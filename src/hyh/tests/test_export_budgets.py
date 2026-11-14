"""Export bounds and compatibility with synthetic records and temporary files."""
import csv
import io
import json
import sys
import tempfile

import pytest
from fastapi.testclient import TestClient

from src.hyh import app as api, db, export_io


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "export.sqlite3")
    files = []

    def temporary():
        stream = tempfile.TemporaryFile(mode="w+b", dir=tmp_path)
        files.append(stream)
        return stream

    # Test only synthetic data here; production private ACL creation is also
    # exercised by the unpatched export contract tests.
    monkeypatch.setattr(export_io, "_temporary_file", temporary)
    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000),
                    headers={"Authorization": "Bearer " + "a" * 43}) as client:
        yield client, files
    assert all(stream.closed for stream in files)


def collect(client, i=0, **changes):
    item = {"url": f"https://example.com/{i}", "title": f"Synthetic {i}", "text": f"文本 {i} 😀",
            "source": "import", "channel": "edu", "ts": 1000 + i}
    item.update(changes)
    assert client.post("/collect", json=item).status_code == 200


@pytest.mark.parametrize("path,params", [
    ("/export/lsj", {"view": "raw"}), ("/export/lsj", {"view": "analysis"}),
    ("/export/lsj/training", {"bare": "true"}), ("/export/lsj/training", {"bare": "false"}),
])
@pytest.mark.parametrize("fmt", ["json", "jsonl", "csv"])
def test_complete_encoded_byte_budget_includes_unicode_and_wrapper(client, monkeypatch, path, params, fmt):
    http, files = client
    collect(http)
    query = {**params, "fmt": fmt}
    baseline = http.get(path, params=query)
    assert baseline.status_code == 200
    size = len(baseline.content)
    assert int(baseline.headers["content-length"]) == size
    monkeypatch.setattr(export_io, "MAX_EXPORT_BYTES", size)
    exact = http.get(path, params=query)
    assert exact.status_code == 200 and exact.content == baseline.content
    monkeypatch.setattr(export_io, "MAX_EXPORT_BYTES", size - 1)
    rejected = http.get(path, params=query)
    assert rejected.status_code == 413
    assert rejected.headers["cache-control"] == "no-store"
    assert "文本" not in rejected.text and all(stream.closed for stream in files)


@pytest.mark.parametrize("fmt", ["json", "jsonl", "csv"])
def test_training_preserves_limit_before_filter_and_dedup_and_timestamp_order(client, fmt):
    http, _ = client
    collect(http, 1, ts=10, text="First input")
    collect(http, 2, ts=10, text="  First   input  ")
    collect(http, 3, ts=5, url="http://localhost/private", text="filtered")
    collect(http, 4, ts=15, text="Outside scanned limit")
    response = http.get("/export/lsj/training", params={"fmt": fmt, "limit_rows": 3, "bare": "false"})
    assert response.status_code == 200
    if fmt == "json":
        assert response.json() == {"items": [{"input": "First input", "label": "edu", "ts": 10,
            "url": "https://example.com/1", "title": "Synthetic 1", "source": "import"}],
            "count": 1, "label_field": "channel", "from_ts": None, "to_ts": None}
    elif fmt == "jsonl":
        assert not response.content.endswith(b"\n")
        assert json.loads(response.content)["input"] == "First input"
    else:
        rows = list(csv.DictReader(io.StringIO(response.text)))
        assert len(rows) == 1 and rows[0]["input"] == "First input"


@pytest.mark.parametrize("field", ["url", "title", "text", "lang", "channel", "author", "tags", "meta"])
def test_oversized_legacy_nullable_field_is_rejected_before_loading_record(client, monkeypatch, field):
    http, _ = client
    collect(http)
    with db.get_conn() as conn:
        conn.execute(f"UPDATE items SET {field} = ?", ("x" * (export_io.MAX_ROW_BYTES + 1),))
    shaped = []
    original = api._shape_export_rows

    def observe(rows, view):
        for row in rows:
            shaped.append(row)
            yield from original([row], view)

    monkeypatch.setattr(api, "_shape_export_rows", observe)
    response = http.get("/export/lsj")
    assert response.status_code == 413 and not shaped


def test_dedup_budget_keeps_exact_keys_and_does_not_charge_duplicates(client, monkeypatch):
    http, _ = client
    collect(http, 1, text="SAME input")
    collect(http, 2, text="same    input")
    normalized = api.normalize_text("", "SAME input")
    monkeypatch.setattr(export_io, "MAX_DEDUP_BYTES", sys.getsizeof(normalized) + 128)
    assert len(http.get("/export/lsj/training").json()) == 1
    collect(http, 3, text="a distinct input")
    assert http.get("/export/lsj/training").status_code == 413
    response = http.get("/export/lsj/training", params={"dedup_by_input": "false"})
    assert response.status_code == 200 and len(response.json()) == 3


@pytest.mark.parametrize("fault", ["write", "flush", "seek"])
def test_storage_failure_is_safe_and_closes_artifact(client, monkeypatch, fault):
    http, files = client
    collect(http)
    original = export_io._temporary_file

    class BrokenFile:
        def __init__(self):
            self.stream = original()

        def __getattr__(self, name):
            if name == fault:
                def fail(*args, **kwargs):
                    raise OSError("synthetic-private-path-and-text")
                return fail
            return getattr(self.stream, name)

    monkeypatch.setattr(export_io, "_temporary_file", BrokenFile)
    response = http.get("/export/lsj")
    assert response.status_code == 507
    assert "synthetic-private" not in response.text
    assert all(stream.closed for stream in files)
    assert http.get("/items").status_code == 200


def test_storage_creation_failure_and_timed_out_preparation_release_resources(client, monkeypatch):
    http, files = client
    original = export_io._temporary_file

    def unavailable():
        raise OSError("synthetic-private-path")

    monkeypatch.setattr(export_io, "_temporary_file", unavailable)
    assert http.get("/export/lsj").status_code == 507
    monkeypatch.setattr(export_io, "_temporary_file", original)
    monkeypatch.setattr(export_io, "PREPARE_SECONDS", 0)
    timeout = http.get("/export/lsj")
    assert timeout.status_code == 503 and timeout.headers["retry-after"] == "5"
    assert all(stream.closed for stream in files)
    monkeypatch.setattr(export_io, "PREPARE_SECONDS", 15)
    assert http.get("/export/lsj").status_code == 200


def test_invalid_legacy_json_cannot_emit_nonfinite_json_or_leak_data(client):
    http, files = client
    collect(http)
    with db.get_conn() as conn:
        conn.execute("UPDATE items SET meta = ?", ('{"private": NaN}',))
    response = http.get("/export/lsj")
    assert response.status_code == 409
    assert "private" not in response.text and all(stream.closed for stream in files)


def test_close_failure_does_not_mask_original_disk_error(client, monkeypatch, caplog):
    http, files = client
    original = export_io._temporary_file

    class BrokenFile:
        def __init__(self):
            self.stream = original()

        def write(self, *_args):
            raise OSError("synthetic-private-write-path")

        def close(self):
            self.stream.close()
            raise OSError("synthetic-private-close-path")

    monkeypatch.setattr(export_io, "_temporary_file", BrokenFile)
    response = http.get("/export/lsj")
    assert response.status_code == 507
    assert all(stream.closed for stream in files)
    assert "synthetic-private" not in response.text + caplog.text


def test_sql_scanning_timeout_closes_read_snapshot_and_allows_next_write(client, monkeypatch):
    from contextlib import contextmanager
    http, files = client
    for i in range(60):
        collect(http, i)
    original = export_io.get_conn

    @contextmanager
    def expire_when_scanning():
        with original() as conn:
            def trace(sql):
                if sql.startswith("SELECT id, ("):
                    monkeypatch.setattr(export_io, "_clock", lambda: float("inf"))
            conn.set_trace_callback(trace)
            yield conn

    with monkeypatch.context() as scoped:
        scoped.setattr(export_io, "get_conn", expire_when_scanning)
        real_clock = export_io._clock
        try:
            response = http.get("/export/lsj")
        finally:
            monkeypatch.setattr(export_io, "_clock", real_clock)
    assert response.status_code == 503
    assert all(stream.closed for stream in files)
    collect(http, 61)
