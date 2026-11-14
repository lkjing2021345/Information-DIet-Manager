"""Import field integrity through real multipart HTTP and temporary SQLite."""
import csv
import io
import json

import pytest
from fastapi.testclient import TestClient

from src.hyh import app as api, db

PRIVATE = "SYNTHETIC_PRIVATE_IMPORT_FIELD"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "synthetic-fields.sqlite3")
    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000),
                    headers={"Authorization": "Bearer " + "a" * 43}) as client:
        yield client


def item(name, **fields):
    return {"url": "https://example.invalid/" + name, "title": "Synthetic " + name,
            "text": "Synthetic content", "ts": 1790208000000, "source": "import", **fields}


def upload(client, extension, rows):
    if extension == "csv":
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=[*item("fields"), "tags", "meta"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
        payload = stream.getvalue()
    elif extension == "jsonl":
        payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    else:
        payload = json.dumps(rows, ensure_ascii=False)
    return client.post("/import", files={"file": ("synthetic." + extension, payload.encode("utf-8"),
                                                  "application/octet-stream")})


def rows_in_database():
    with db.get_conn() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM items ORDER BY id")]


INVALID_FIELDS = [
    ("meta", []), ("meta", [PRIVATE]), ("meta", False), ("meta", 0),
    ("meta", "{" + PRIVATE + ":"), ("meta", json.dumps([PRIVATE])), ("meta", json.dumps(PRIVATE)),
    ("tags", {PRIVATE: "value"}), ("tags", False), ("tags", True), ("tags", 0), ("tags", 1.25),
]


@pytest.mark.parametrize("extension", ["json", "jsonl"])
@pytest.mark.parametrize("field,value", INVALID_FIELDS)
def test_invalid_import_field_counts_as_failed_without_losing_field(client, caplog, extension, field, value):
    existing = item("existing", tags=["saved"], meta={"keep": 1})
    assert client.post("/collect", json=existing).status_code == 200
    before = rows_in_database()[0]
    good = item("valid-neighbor", tags=["ok"], meta={"source": "synthetic"})
    invalid = item("invalid", **{field: value})

    response = upload(client, extension, [good, invalid, existing])

    assert response.status_code == 200
    assert response.json() == {"inserted": 1, "duplicates": 1, "failed": 1}
    assert PRIVATE not in response.text + caplog.text
    rows = rows_in_database()
    assert rows[0] == before
    assert [row["url"] for row in rows] == [existing["url"], good["url"]]
    assert json.loads(rows[1]["tags"]) == good["tags"]
    assert json.loads(rows[1]["meta"]) == good["meta"]


@pytest.mark.parametrize("value", ["[]", "true", "0", json.dumps(PRIVATE), "{" + PRIVATE + ":"])
def test_csv_non_object_or_malformed_meta_is_a_row_failure(client, caplog, value):
    response = upload(client, "csv", [item("bad-csv", meta=value), item("good-csv", meta='{"kept":true}')])
    assert response.status_code == 200
    assert response.json() == {"inserted": 1, "duplicates": 0, "failed": 1}
    assert PRIVATE not in response.text + caplog.text
    rows = rows_in_database()
    assert len(rows) == 1 and rows[0]["url"] == item("good-csv")["url"]
    assert json.loads(rows[0]["meta"]) == {"kept": True}


@pytest.mark.parametrize("extension", ["json", "jsonl", "csv"])
def test_legacy_valid_empty_and_textual_field_forms_remain_compatible(client, extension):
    cases = [
        ({}, None, None),
        ({"tags": None, "meta": None}, None, None),
        ({"tags": "", "meta": ""}, None, None),
        ({"tags": " \t ", "meta": " \t "}, None, None),
        ({"tags": [], "meta": {}}, [], {}),
        ({"tags": ["ai", "政策"], "meta": {"ok": True, "value": [0, None]}},
         ["ai", "政策"], {"ok": True, "value": [0, None]}),
        ({"tags": '["ai","政策"]', "meta": '{"nested":{"key":"值"}}'},
         ["ai", "政策"], {"nested": {"key": "值"}}),
        ({"tags": " ai || 政策 | ", "meta": "null"}, ["ai", "政策"], None),
        ({"tags": "[not-json", "meta": None}, ["[not-json"], None),
        ({"tags": "null", "meta": " null "}, ["null"], None),
        ({"tags": ["x|y"], "meta": {}}, ["x|y"], {}),
    ]
    records = [item(f"valid-{index}", **fields) for index, (fields, _tags, _meta) in enumerate(cases)]
    response = upload(client, extension, records)
    assert response.status_code == 200
    assert response.json() == {"inserted": len(records), "duplicates": 0, "failed": 0}
    actual = {row["url"]: row for row in client.get("/items?page_size=200").json()["items"]}
    for row, (_fields, tags, meta) in zip(records, cases):
        assert actual[row["url"]]["tags"] == tags
        assert actual[row["url"]]["meta"] == meta
    before = rows_in_database()
    retry = upload(client, extension, records)
    assert retry.json() == {"inserted": 0, "duplicates": len(records), "failed": 0}
    assert rows_in_database() == before


@pytest.mark.parametrize("extension", ["json", "jsonl", "csv"])
def test_all_invalid_optional_fields_leave_existing_rows_unchanged(client, extension):
    assert client.post("/collect", json=item("existing", meta={"keep": True})).status_code == 200
    before = rows_in_database()
    response = upload(client, extension, [item("invalid", meta=[])])
    assert response.status_code == 200
    assert response.json() == {"inserted": 0, "duplicates": 0, "failed": 1}
    assert rows_in_database() == before
