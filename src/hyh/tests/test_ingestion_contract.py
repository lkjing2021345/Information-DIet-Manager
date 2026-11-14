"""Ingestion behavior with synthetic, temporary databases; no model inference."""
import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.hyh import app as api
from src.hyh import db
from src.hyh.models import IngestItem, MAX_INGEST_TS, MAX_META_BYTES


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "ingestion.sqlite3")
    with TestClient(api.app, raise_server_exceptions=False, base_url="http://127.0.0.1", client=("127.0.0.1", 50000),
                    headers={"Authorization": "Bearer " + "a" * 43}) as client:
        yield client


def item(**changes):
    return {
        "url": "https://example.invalid/synthetic", "title": "Synthetic title",
        "text": "Synthetic content", "ts": 1790208000000, "source": "plugin",
        **changes,
    }


def test_ack_after_insert_and_response_loss_retry_is_page_idempotent(client):
    first = client.post("/collect", json=item())
    assert first.status_code == 200
    assert first.json() == {"inserted": 1, "duplicates": 0, "failed": 0}

    # Simulate a sender retrying after losing the first response.
    retry = client.post("/collect", json=item())
    assert retry.status_code == 200
    assert retry.json() == {"inserted": 0, "duplicates": 1, "failed": 0}

    # Existing page-level deduplication also ignores a later changed visit.
    changed = client.post("/collect", json=item(title="Later visit", ts=1790208001000))
    assert changed.json() == {"inserted": 0, "duplicates": 1, "failed": 0}
    stored = client.get("/items").json()
    assert stored["total"] == 1
    assert stored["items"][0]["title"] == "Synthetic title"
    assert stored["items"][0]["ts"] == 1790208000000


@pytest.mark.parametrize("ts", [0, MAX_INGEST_TS])
def test_timestamp_inclusive_boundaries_are_accepted(client, ts):
    response = client.post("/collect", json=item(ts=ts))
    assert response.status_code == 200
    assert client.get("/items").json()["items"][0]["ts"] == ts


@pytest.mark.parametrize("ts", [-1, MAX_INGEST_TS + 1, 2**63, True, False, 1.0, "1790208000000", None])
def test_invalid_timestamp_returns_422_without_storage(client, ts):
    response = client.post("/collect", json=item(ts=ts))
    assert response.status_code == 422
    assert client.get("/items").json()["total"] == 0


def test_title_is_trimmed_before_length_validation(client):
    response = client.post("/collect", json=item(title=" \t" + "x" * 300 + "\n "))
    assert response.status_code == 200
    assert client.get("/items").json()["items"][0]["title"] == "x" * 300


@pytest.mark.parametrize("title", ["", " \t\n ", "x" * 301])
def test_invalid_title_returns_422(client, title):
    assert client.post("/collect", json=item(title=title)).status_code == 422
    assert client.get("/items").json()["total"] == 0


def test_tags_accept_documented_limits(client):
    tags = ["标" * 40] * 10
    response = client.post("/collect", json=item(tags=tags))
    assert response.status_code == 200
    assert client.get("/items").json()["items"][0]["tags"] == tags


@pytest.mark.parametrize("tags", [["tag"] * 11, ["x" * 41], [False]])
def test_invalid_tags_return_422(client, tags):
    assert client.post("/collect", json=item(tags=tags)).status_code == 422
    assert client.get("/items").json()["total"] == 0


def nested_meta(levels):
    result = {"value": "leaf"}
    for _ in range(levels - 1):
        result = {"child": result}
    return result


def test_meta_accepts_eight_container_levels(client):
    assert client.post("/collect", json=item(meta=nested_meta(8))).status_code == 200


@pytest.mark.parametrize("meta", [nested_meta(9), {"a": [[[[[[[[0]]]]]]]]}])
def test_meta_rejects_excessive_object_and_array_nesting(client, meta):
    assert client.post("/collect", json=item(meta=meta)).status_code == 422
    assert client.get("/items").json()["total"] == 0


def test_meta_limit_counts_compact_utf8_bytes(client):
    # {"a":""} uses 8 bytes; each literal Chinese character uses 3 UTF-8 bytes.
    meta = {"a": "汉" * 2728}
    assert len(json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) == MAX_META_BYTES
    assert client.post("/collect", json=item(meta=meta)).status_code == 200
    over = {"a": meta["a"] + "x"}
    assert client.post("/collect", json=item(url="https://example.invalid/over", meta=over)).status_code == 422
    assert client.get("/items").json()["total"] == 1


@pytest.mark.parametrize("meta", [{"bad": {1, 2}}, {"bad": float("nan")}, {"bad": float("inf")}, {"bad": object()}])
def test_python_model_rejects_non_json_metadata(meta):
    with pytest.raises(ValidationError):
        IngestItem(**item(meta=meta))


def test_python_model_rejects_cyclic_metadata():
    meta = {}
    meta["cycle"] = meta
    with pytest.raises(ValidationError):
        IngestItem(**item(meta=meta))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf"), "\ud800", "\udfff"])
def test_raw_http_invalid_metadata_returns_safe_422(client, invalid):
    # Raw content deliberately bypasses httpx's rejection of non-finite JSON.
    # ensure_ascii preserves malformed surrogate escapes until server decoding.
    raw = json.dumps(item(meta={"invalid": invalid, "private": "synthetic-secret-marker"}), ensure_ascii=True)
    response = client.post("/collect", content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.headers["content-type"] == "application/json"
    details = response.json()["detail"]
    assert details
    assert all(set(error) == {"type", "loc", "msg"} for error in details)
    assert "synthetic-secret-marker" not in response.text
    assert client.get("/items").json()["total"] == 0


@pytest.mark.parametrize("field", ["url", "title", "text", "lang", "channel", "author", "source", "tags"])
@pytest.mark.parametrize("surrogate", ["\ud800", "\udfff"])
def test_raw_http_lone_surrogates_are_rejected_in_all_string_fields(client, field, surrogate):
    value = f"https://example.invalid/{surrogate}" if field == "url" else surrogate
    if field == "tags":
        value = [surrogate]
    raw = json.dumps(item(**{field: value}), ensure_ascii=True)
    response = client.post("/collect", content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert all("input" not in error and "ctx" not in error for error in response.json()["detail"])
    assert client.get("/items").json()["total"] == 0


def test_valid_unicode_surrogate_pair_round_trips_as_unicode(client):
    payload = item(title="合成测试 😀", text="测试正文 😀", tags=["😀"], meta={"emoji": "😀"})
    # An escaped, correctly paired surrogate represents a valid Unicode code point.
    raw = json.dumps(payload, ensure_ascii=True)
    assert "\\ud83d\\ude00" in raw
    response = client.post("/collect", content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 200
    saved = client.get("/items").json()["items"][0]
    for field in ("title", "text", "tags", "meta"):
        assert saved[field] == payload[field]


def test_optional_null_fields_remain_compatible(client):
    response = client.post("/collect", json=item(
        text=None, lang=None, channel=None, author=None, tags=None, meta=None,
        future_client_field="ignored for compatibility",
    ))
    assert response.status_code == 200
    assert client.get("/items").json()["items"][0]["text"] == "Synthetic title"


def test_import_retains_csv_timestamp_conversion(client):
    csv = "url,title,ts,source\nhttps://example.invalid/csv,Synthetic CSV,1790208000000,import\n"
    response = client.post("/import", files={"file": ("synthetic.csv", csv, "text/csv")})
    assert response.status_code == 200
    assert response.json() == {"inserted": 1, "duplicates": 0, "failed": 0}


def test_import_reports_invalid_item_without_losing_valid_items(client):
    payload = [item(source="import"), item(url="https://example.invalid/invalid", tags=["x"] * 11)]
    response = client.post("/import", files={
        "file": ("synthetic.json", json.dumps(payload), "application/json"),
    })
    assert response.status_code == 200
    assert response.json() == {"inserted": 1, "duplicates": 0, "failed": 1}
    assert client.get("/items").json()["total"] == 1


def test_embedding_failure_rolls_back_item_and_retry_can_succeed(client, monkeypatch):
    original = api._upsert_embedding

    def fail_after_embedding(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("Synthetic failure after both inserts")

    monkeypatch.setattr(api, "_upsert_embedding", fail_after_embedding)
    response = client.post("/collect", json=item())
    assert response.status_code == 500
    with db.get_conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0

    monkeypatch.setattr(api, "_upsert_embedding", original)
    retry = client.post("/collect", json=item())
    assert retry.status_code == 200
    assert retry.json() == {"inserted": 1, "duplicates": 0, "failed": 0}


@pytest.mark.parametrize("invalid_column,invalid_value", [("source", "invalid-source"), ("title", None)])
def test_non_url_database_constraint_is_not_acknowledged_as_duplicate(client, monkeypatch, invalid_column, invalid_value):
    assert client.post("/collect", json=item()).json()["inserted"] == 1
    original = api._row_from_item

    def invalid_row(record):
        row = original(record)
        row[invalid_column] = invalid_value
        return row

    monkeypatch.setattr(api, "_row_from_item", invalid_row)
    payload = item(url="https://example.invalid/constraint")
    response = client.post("/collect", json=payload)
    assert response.status_code == 500
    with db.get_conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 1

    monkeypatch.setattr(api, "_row_from_item", original)
    retry = client.post("/collect", json=payload)
    assert retry.status_code == 200
    assert retry.json() == {"inserted": 1, "duplicates": 0, "failed": 0}


def test_import_storage_constraint_failure_rolls_back_entire_transaction(client, monkeypatch):
    original = api._row_from_item

    def fail_second_row(record):
        row = original(record)
        if str(record.url).endswith("/second"):
            row["source"] = "invalid-source"
        return row

    payload = [item(source="import"), item(url="https://example.invalid/second", source="import")]
    upload = {"file": ("synthetic.json", json.dumps(payload), "application/json")}
    monkeypatch.setattr(api, "_row_from_item", fail_second_row)
    response = client.post("/import", files=upload)
    assert response.status_code == 500
    with db.get_conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0

    monkeypatch.setattr(api, "_row_from_item", original)
    retry = client.post("/import", files=upload)
    assert retry.status_code == 200
    assert retry.json() == {"inserted": 2, "duplicates": 0, "failed": 0}
