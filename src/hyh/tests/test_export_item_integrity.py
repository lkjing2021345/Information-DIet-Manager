"""General exports preserve valid stored JSON or fail before delivery.

Only synthetic records and temporary databases/files are used. File creation is
substituted to keep the JSON matrix separate from Windows ACL subprocess tests.
"""
import csv
import io
import json
import tempfile

import pytest
from fastapi.testclient import TestClient

from src.hyh import app as api, db, export_io


PRIVATE_MARKER = "SYNTHETIC_PRIVATE_EXPORT_JSON"
INVALID_KINDS = (
    "malformed", "empty", "invalid_utf8", "NaN", "Infinity", "-Infinity",
    "1e999", "-1e999", "surrogate_value", "surrogate_key", "depth65",
    "depth300", "depth2000",
)
NONFINITE_KINDS = {"NaN", "Infinity", "-Infinity", "1e999", "-1e999"}
VALID_VALUES = (
    None, "null", "0", "-0.0", "false", "1e308",
    "1234567890123456789012345678901234567890", '""', '"旧标签 😀"',
    '["\\ud83d\\ude00"]', "{}", "[]", '{"legacy":[0,null,false,"中文 😀"]}',
    '[[{"nested":true}]]', "[" * 63 + "0" + "]" * 63,
    "[" * 64 + "0" + "]" * 64, b'{"utf8":"\xe6\x96\x87"}',
)


def invalid_value(kind):
    if kind == "malformed":
        return '{"private":"' + PRIVATE_MARKER + '","v":'
    if kind == "empty":
        return ""
    if kind == "invalid_utf8":
        return b'{"v":"\xff"}'
    if kind == "surrogate_value":
        return '"\\ud800"'
    if kind == "surrogate_key":
        return '{"\\udfff":"' + PRIVATE_MARKER + '"}'
    if kind.startswith("depth"):
        depth = int(kind.removeprefix("depth"))
        return "[" * depth + json.dumps(PRIVATE_MARKER) + "]" * depth
    return kind


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "synthetic-export-integrity.sqlite3")
    files = []

    def temporary():
        stream = tempfile.TemporaryFile(mode="w+b", dir=tmp_path)
        files.append(stream)
        return stream

    monkeypatch.setattr(export_io, "_temporary_file", temporary)
    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 51000),
                    headers={"Authorization": "Bearer " + "a" * 43},
                    raise_server_exceptions=False) as http:
        yield http, files
    assert all(stream.closed for stream in files)


def collect(http, index=0):
    response = http.post("/collect", json={
        "url": f"https://example.invalid/export-integrity/{index}",
        "title": PRIVATE_MARKER, "text": f"Synthetic input {index}",
        "ts": 1790208000000 + index, "source": "import", "channel": "edu",
    })
    assert response.status_code == 200 and response.json()["inserted"] == 1


def stored_state():
    with db.get_conn() as conn:
        return ([tuple(row) for row in conn.execute("SELECT * FROM items ORDER BY id")],
                tuple(conn.execute("SELECT * FROM items_revision").fetchone()))


def assert_resources_released(files):
    assert all(stream.closed for stream in files)
    assert not api.app.state.operation_lock.locked()
    assert api.app.state.request_slots._value == 4
    assert api.app.state.export_slots._value == 2


def assert_conflict(response):
    assert response.status_code == 409
    assert response.json() == {"detail": "Stored records cannot be encoded for export"}
    assert "no-store" in response.headers["cache-control"]
    assert PRIVATE_MARKER not in response.text


def exported_rows(response, fmt, *, training=False):
    assert response.status_code == 200, response.text
    assert int(response.headers["content-length"]) == len(response.content)
    if fmt == "json":
        return response.json() if training else response.json()["items"]
    if fmt == "jsonl":
        return [json.loads(line) for line in response.text.splitlines()]
    return list(csv.DictReader(io.StringIO(response.text)))


@pytest.mark.parametrize("field", ["tags", "meta"])
@pytest.mark.parametrize("kind", INVALID_KINDS)
def test_invalid_stored_json_is_explicit_and_does_not_rewrite_records(client, field, kind, caplog):
    http, files = client
    collect(http)
    with db.get_conn() as conn:
        conn.execute(f"UPDATE items SET {field}=? WHERE id=1", (invalid_value(kind),))
    before = stored_state()
    # CSV previously wrote scalar NaN/Infinity directly, unlike JSON encoding.
    fmt = "csv" if kind in NONFINITE_KINDS else "json"
    response = http.get("/export/lsj", params={"view": "raw", "fmt": fmt})
    assert_conflict(response)
    assert stored_state() == before
    assert PRIVATE_MARKER not in caplog.text
    assert_resources_released(files)
    # A failed read must not strand the process's subsequent write capacity.
    collect(http, 1)


@pytest.mark.parametrize("view", ["analysis", "raw"])
@pytest.mark.parametrize("fmt", ["json", "jsonl", "csv"])
def test_later_invalid_row_never_delivers_a_partial_success(client, view, fmt):
    http, files = client
    collect(http, 0)
    collect(http, 1)
    field = "tags" if view == "raw" else "meta"
    with db.get_conn() as conn:
        conn.execute(f"UPDATE items SET {field}=? WHERE id=2", (invalid_value("malformed"),))
    before = stored_state()
    response = http.get("/export/lsj", params={"view": view, "fmt": fmt})
    assert_conflict(response)
    assert stored_state() == before
    assert_resources_released(files)
    # Only selected records are decoded; an earlier valid window still exports.
    valid = http.get("/export/lsj", params={"view": view, "fmt": fmt, "limit_rows": 1})
    assert len(exported_rows(valid, fmt)) == 1
    assert_resources_released(files)


@pytest.mark.parametrize("view", ["analysis", "raw"])
@pytest.mark.parametrize("fmt", ["json", "jsonl", "csv"])
def test_legal_old_json_shapes_nulls_and_utf8_bytes_are_preserved(client, view, fmt):
    http, files = client
    for index, value in enumerate(VALID_VALUES):
        collect(http, index)
        with db.get_conn() as conn:
            conn.execute("UPDATE items SET tags=?,meta=? WHERE id=?", (value, value, index + 1))
    before = stored_state()
    rows = exported_rows(http.get("/export/lsj", params={"view": view, "fmt": fmt}), fmt)
    assert len(rows) == len(VALID_VALUES)
    for row, value in zip(rows, VALID_VALUES):
        expected = None if value is None else json.loads(value)
        if fmt == "csv":
            if isinstance(expected, (dict, list)):
                expected = json.dumps(expected, ensure_ascii=False, allow_nan=False)
            else:
                expected = "" if expected is None else str(expected)
        for field in ("tags", "meta"):
            assert row[field] == expected
            if fmt != "csv":
                # Equality alone would not distinguish False from 0 or -0.0.
                assert json.dumps(row[field], ensure_ascii=False, allow_nan=False) == json.dumps(
                    expected, ensure_ascii=False, allow_nan=False)
    assert stored_state() == before
    assert_resources_released(files)


@pytest.mark.parametrize("fmt", ["json", "jsonl", "csv"])
def test_training_does_not_decode_or_export_unrelated_stored_json(client, fmt):
    http, files = client
    for index, kind in enumerate(INVALID_KINDS):
        collect(http, index)
        with db.get_conn() as conn:
            conn.execute("UPDATE items SET tags=?,meta=? WHERE id=?",
                         (invalid_value(kind), invalid_value(kind), index + 1))
    before = stored_state()
    rows = exported_rows(http.get("/export/lsj/training", params={"fmt": fmt}), fmt, training=True)
    assert [row["input"] for row in rows] == [f"Synthetic input {i}" for i in range(len(INVALID_KINDS))]
    assert all(set(row) == {"input", "label", "ts", "url", "title", "source"} for row in rows)
    assert stored_state() == before
    assert_resources_released(files)
