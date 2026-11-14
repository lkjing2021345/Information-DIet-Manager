"""Actual versioned backup/restore with synthetic legacy rows and temporary SQLite.

The guarantee is equality of service-generated backup items, not SQLite bytes.
No model is loaded; the existing local hashed embedding is rebuilt normally.
"""
import hashlib
import json

from fastapi.testclient import TestClient
import pytest

from src.hyh import app as api, data_management as management, db
from src.hyh.models import MAX_INGEST_TS
from src.hyh.utils import normalize_text, sha256_hex


TEXT_VALUES = [None, "", " \t\n ", "  合成正文 😀\n尾部\t", "x" * 999,
               "x" * 1000, "x" * 1001, "合成" * 750, "😀" * 2000]
OPTIONAL_VALUES = [
    {"ts": 0, "lang": None, "channel": None, "author": None, "tags": None, "meta": None},
    {"ts": MAX_INGEST_TS, "lang": "", "channel": "", "author": "", "tags": [], "meta": {}},
    {"ts": 1790208000000, "lang": " zh ", "channel": " 合成 ", "author": " 测试 😀 ",
     "tags": ["", " 标记 😀 "], "meta": {"a": [None, False, 0, " 中文\n "], "empty": {}}},
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "synthetic-backup-roundtrip.sqlite3")
    with TestClient(api.app, raise_server_exceptions=False, base_url="http://127.0.0.1",
                    client=("127.0.0.1", 51212), headers={"Authorization": "Bearer " + "a" * 43}) as client:
        yield client


def item(**changes):
    return {"url": "https://example.invalid/roundtrip", "title": "合成标题",
            "text": "Synthetic text", "ts": 1790208000000, "source": "plugin", **changes}


def backup(client):
    response = client.get("/data/backup")
    assert response.status_code == 200
    result = response.json()
    assert result["sha256"] == hashlib.sha256(management.canonical_items(result["items"])).hexdigest()
    return result


@pytest.mark.parametrize("text", TEXT_VALUES,
                         ids=["null", "empty", "whitespace", "untrimmed", "999", "1000", "1001", "1500", "2000-emoji"])
@pytest.mark.parametrize("optional", OPTIONAL_VALUES, ids=["null-fields", "empty-fields", "unicode-fields"])
def test_generated_backup_items_roundtrip_without_applying_ingestion_text_changes(client, text, optional):
    assert client.post("/collect", json=item(**optional)).status_code == 200
    # Store legal historical text explicitly; ordinary ingestion intentionally
    # normalizes it. Export must itself validate and produce the tested backup.
    with db.get_conn() as conn:
        conn.execute("UPDATE items SET text = ?", (text,))
    original = backup(client)
    assert original["items"][0] == {**item(**optional), "text": text}
    for _ in range(2):
        response = client.post("/data/restore", json=original,
                               headers={"X-IDM-Confirm": "replace-records"})
        assert response.status_code == 200
        assert response.json() == {"restored": 1, "analysis_cleared": True}
        assert backup(client)["items"] == original["items"]
        with db.get_conn() as conn:
            row = conn.execute("SELECT text, content_hash FROM items").fetchone()
            assert row["text"] == text
            assert row["content_hash"] == sha256_hex(normalize_text("合成标题", text))
            assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 1


@pytest.mark.parametrize("route", ["collect", "import"])
@pytest.mark.parametrize("text,expected", [
    (None, "合成标题"), ("", "合成标题"), (" \t\n ", "合成标题"),
    ("  合成正文  ", "合成正文"), ("a" * 1500, "a" * 1000), ("😀" * 2000, "😀" * 1000),
], ids=["null", "empty", "whitespace", "untrimmed", "1500", "2000-emoji"])
def test_ordinary_ingestion_keeps_original_text_normalization(client, route, text, expected):
    payload = item(text=text, source="import" if route == "import" else "plugin")
    if route == "collect":
        response = client.post("/collect", json=payload)
    else:
        response = client.post("/import", files={
            "file": ("synthetic.json", json.dumps([payload]), "application/json"),
        })
    assert response.status_code == 200
    assert response.json() == {"inserted": 1, "duplicates": 0, "failed": 0}
    assert client.get("/items").json()["items"][0]["text"] == expected


def tables():
    with db.get_conn() as conn:
        return {table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table}")]
                for table in ("items", "embeddings", "analysis_runs", "analysis_jobs", "stats_daily", "items_revision")}


def seed_long_backup(client):
    for index in range(2):
        assert client.post("/collect", json=item(url=f"https://example.invalid/row-{index}")).status_code == 200
    with db.get_conn() as conn:
        conn.execute("UPDATE items SET text = ?", ("长正文" * 500,))
    original = backup(client)
    assert client.post("/collect", json=item(url="https://example.invalid/sentinel")).status_code == 200
    assert client.post("/analyze/run").status_code == 200  # Local statistics, no model inference.
    return original


def test_long_backup_restore_clears_derived_data_and_rebuilds_embeddings(client):
    original = seed_long_backup(client)
    before = tables()
    assert before["analysis_runs"] and before["stats_daily"]
    response = client.post("/data/restore", json=original, headers={"X-IDM-Confirm": "replace-records"})
    assert response.status_code == 200 and response.json()["restored"] == 2
    assert backup(client)["items"] == original["items"]
    after = tables()
    assert len(after["items"]) == len(after["embeddings"]) == 2
    assert not after["analysis_runs"] and not after["analysis_jobs"] and not after["stats_daily"]


@pytest.mark.parametrize("failure", ["duplicate_url", "embedding_error", "checksum"])
def test_rejected_long_backup_preserves_every_table(client, monkeypatch, failure):
    original = seed_long_backup(client)
    before = tables()
    if failure == "duplicate_url":
        original["items"][1]["url"] = original["items"][0]["url"] + "#same-page"
        original["sha256"] = hashlib.sha256(management.canonical_items(original["items"])).hexdigest()
    elif failure == "checksum":
        original["sha256"] = "0" * 64
    else:
        original_upsert = api._upsert_embedding
        calls = 0
        def fail_second(*args, **kwargs):
            nonlocal calls
            original_upsert(*args, **kwargs)
            calls += 1
            if calls == 2:
                raise OSError("Synthetic storage error")
        monkeypatch.setattr(api, "_upsert_embedding", fail_second)
    response = client.post("/data/restore", json=original, headers={"X-IDM-Confirm": "replace-records"})
    assert response.status_code == (500 if failure == "embedding_error" else 422)
    assert tables() == before
