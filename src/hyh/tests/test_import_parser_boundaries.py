"""Parser failures through ASGI HTTP with synthetic files and temporary SQLite."""
import csv
import io
import json

import pytest
from fastapi.testclient import TestClient

from src.hyh import app as api, db


PRIVATE_MARKER = "SYNTHETIC_PRIVATE_IMPORT_MARKER"
ORIGIN = "http://127.0.0.1:5173"
TABLES = ("items", "embeddings", "analysis_jobs", "analysis_runs", "stats_daily", "items_revision")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "synthetic-import.sqlite3")
    with TestClient(
        api.app, raise_server_exceptions=False, base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
        headers={"Authorization": "Bearer " + "a" * 43, "Origin": ORIGIN},
    ) as client:
        yield client


def item(name):
    return {
        "url": "https://example.invalid/" + name, "title": "Synthetic " + name,
        "text": "Synthetic content", "ts": 1790208000000, "source": "import",
    }


def snapshot():
    with db.get_conn() as conn:
        return {table: [tuple(row) for row in conn.execute("SELECT * FROM " + table)] for table in TABLES}


def csv_payload(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(item("prefix")))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def upload(filename, payload):
    return {"file": (filename, payload.encode("utf-8"), "application/octet-stream")}


def invalid_file(kind):
    prefix = json.dumps(item("prefix"))
    if kind == "oversize_csv":
        invalid = item("oversize")
        invalid["title"] = PRIVATE_MARKER + "x" * (csv.field_size_limit() + 1)
        return "synthetic.csv", csv_payload([item("prefix"), invalid])
    if kind.startswith("deep_"):
        # Well below the 10 MiB HTTP body budget; this exercises the parser's
        # recursion boundary rather than per-item metadata depth validation.
        invalid = '{"title":"' + PRIVATE_MARKER + '","meta":' + "[" * 10000 + "0" + "]" * 10000 + "}"
    else:
        invalid = '{"title":"' + PRIVATE_MARKER + '"'
    if kind.endswith("jsonl"):
        return "synthetic.jsonl", prefix + "\n" + invalid
    return "synthetic.json", "[" + prefix + "," + invalid + "]"


@pytest.mark.parametrize("kind", ["oversize_csv", "deep_json", "deep_jsonl", "malformed_json", "malformed_jsonl"])
def test_actual_parser_failure_is_safe_400_and_preserves_database_then_retry(client, caplog, kind):
    assert client.post("/collect", json=item("existing")).status_code == 200
    assert client.post("/analyze/run").status_code == 200
    before = snapshot()
    filename, payload = invalid_file(kind)
    assert PRIVATE_MARKER in payload
    assert len(payload.encode("utf-8")) < 10 * 1024 * 1024

    response = client.post("/import", files=upload(filename, payload))

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid file contents."}
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["access-control-allow-origin"] == ORIGIN
    assert PRIVATE_MARKER not in response.text + caplog.text
    # Even the valid first CSV record/JSONL line must not be inserted when
    # whole-file parsing fails. Existing records and derived state survive.
    assert snapshot() == before
    assert client.get("/health").status_code == 200

    if filename.endswith(".csv"):
        valid = csv_payload([item("prefix")])
    elif filename.endswith(".jsonl"):
        valid = json.dumps(item("prefix")) + "\n"
    else:
        valid = json.dumps([item("prefix")])
    retry = client.post("/import", files=upload(filename, valid))
    assert retry.status_code == 200
    assert retry.json() == {"inserted": 1, "duplicates": 0, "failed": 0}
    assert client.get("/items").json()["total"] == 2


@pytest.mark.parametrize("error_type", [ValueError, csv.Error, RecursionError])
def test_parser_exception_details_are_not_disclosed(client, monkeypatch, caplog, error_type):
    before = snapshot()

    def fail_parser(_text):
        raise error_type(PRIVATE_MARKER)

    monkeypatch.setattr(api, "_load_items_from_json", fail_parser)
    response = client.post("/import", files=upload("synthetic.json", json.dumps([item("prefix")])))

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid file contents."}
    assert PRIVATE_MARKER not in response.text + caplog.text
    assert snapshot() == before


@pytest.mark.parametrize("filename,body", [
    ("synthetic.json", lambda: "\ufeff" + json.dumps([item("bom-json")])),
    ("synthetic.jsonl", lambda: "\ufeff" + json.dumps(item("bom-jsonl")) + "\n"),
    ("synthetic.csv", lambda: "\ufeff" + csv_payload([item("bom-csv")])),
])
def test_utf8_bom_is_treated_as_file_metadata(client, filename, body):
    response = client.post("/import", files=upload(filename, body()))

    assert response.status_code == 200
    assert response.json() == {"inserted": 1, "duplicates": 0, "failed": 0}
