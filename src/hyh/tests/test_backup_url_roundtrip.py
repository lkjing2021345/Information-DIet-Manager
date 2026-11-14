"""Generated backups must obey the same URL identity contract as restoration.

Only synthetic historical rows in temporary SQLite are used. Hashes in the
database are deliberately absent/stale; no model or external service is used.
"""
from contextlib import contextmanager
import hashlib
import sqlite3

from fastapi.testclient import TestClient
import pytest

from src.hyh import app as api, data_management as management, db
from src.hyh.utils import normalize_url


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "synthetic-backup-url.sqlite3")
    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 51212),
                    headers={"Authorization": "Bearer " + "a" * 43}) as client:
        yield client


def seed(client, urls, hashes):
    for index in range(len(urls)):
        response = client.post("/collect", json={
            "url": f"https://example.invalid/seed-{index}", "title": f"Synthetic {index}",
            "text": "Synthetic body", "ts": index, "source": "import",
        })
        assert response.status_code == 200 and response.json()["inserted"] == 1
    with db.get_conn() as conn:
        ids = [row[0] for row in conn.execute("SELECT id FROM items ORDER BY id")]
        for index, (item_id, url) in enumerate(zip(ids, urls, strict=True)):
            conn.execute("UPDATE items SET url = ?, url_hash = ? WHERE id = ?",
                         (url, f"synthetic-stale-{index}" if hashes == "stale" else None, item_id))
    # Exercise preservation of existing derived rows too; this endpoint only
    # calculates local statistics and does not run the model pipeline.
    assert client.post("/analyze/run").status_code == 200


def snapshot():
    with db.get_conn() as conn:
        schema = [tuple(row) for row in conn.execute("SELECT * FROM sqlite_master ORDER BY name")]
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        return schema, {table: [tuple(row) for row in conn.execute(f'SELECT * FROM "{table}" ORDER BY rowid')]
                        for table in tables}


def watch_connections(monkeypatch):
    original, connections = management.get_conn, []

    @contextmanager
    def watched(*args, **kwargs):
        with original(*args, **kwargs) as conn:
            connections.append(conn)
            yield conn

    monkeypatch.setattr(management, "get_conn", watched)
    return connections


def assert_released(client, connections):
    assert connections
    for conn in connections:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")
    assert not api.app.state.operation_lock.locked()
    assert api.app.state.export_slots._value == 2
    assert api.app.state.request_slots._value == 4
    # An actual independent writer proves the failed read snapshot is gone.
    response = client.post("/collect", json={
        "url": "https://example.invalid/after-rejection", "title": "Synthetic follow-up",
        "ts": 99, "source": "import",
    })
    assert response.status_code == 200 and response.json()["inserted"] == 1


@pytest.mark.parametrize("urls,hashes", [
    (("https://example.invalid/page", "https://example.invalid/page"), "null"),
    (("https://example.invalid/page#first", "https://example.invalid/page#second"), "stale"),
    (("https://example.invalid/page", "https://example.invalid/page/"), "null"),
    (("HTTPS://EXAMPLE.INVALID:443/page", "https://example.invalid/page"), "stale"),
    (("https://example.invalid", "https://example.invalid/"), "null"),
], ids=["identical-null", "fragments-stale", "trailing-slash-null", "validated-host-port-stale", "root-null"])
def test_backup_rejects_unrestorable_url_duplicates_without_mutating_database(client, monkeypatch, urls, hashes):
    seed(client, urls, hashes)
    before = snapshot()
    connections = watch_connections(monkeypatch)
    allocations = []
    allocate = management.BytesIO

    def watched_buffer(body):
        stream = allocate(body)
        allocations.append(stream)
        return stream

    monkeypatch.setattr(management, "BytesIO", watched_buffer)
    response = client.get("/data/backup")
    assert response.status_code == 409
    assert response.json() == {"detail": "Stored records contain duplicate normalized page URLs; backup cannot be restored"}
    assert response.headers["cache-control"] == "no-store"
    assert "content-disposition" not in response.headers
    assert allocations == []
    assert snapshot() == before
    assert_released(client, connections)


@pytest.mark.parametrize("urls,hashes", [
    (("https://example.invalid/page?a=1", "https://example.invalid/page?a=2"), "null"),
    (("https://example.invalid/Page", "https://example.invalid/page"), "stale"),
    (("http://example.invalid/page", "https://example.invalid/page"), "null"),
    (("https://example.invalid:444/page", "https://example.invalid/page"), "null"),
], ids=["query-null", "path-case-stale", "scheme-null", "nondefault-port-null"])
def test_distinct_urls_with_legacy_hashes_generate_restorable_backup(client, monkeypatch, urls, hashes):
    seed(client, urls, hashes)
    before = snapshot()
    connections = watch_connections(monkeypatch)
    response = client.get("/data/backup")
    assert response.status_code == 200
    original = response.json()
    assert len(original["items"]) == 2
    assert original["sha256"] == hashlib.sha256(management.canonical_items(original["items"])).hexdigest()
    assert snapshot() == before
    restored = client.post("/data/restore", json=original, headers={"X-IDM-Confirm": "replace-records"})
    assert restored.status_code == 200 and restored.json() == {"restored": 2, "analysis_cleared": True}
    after = client.get("/data/backup")
    assert after.status_code == 200 and after.json()["items"] == original["items"]
    assert_released(client, connections)


def test_url_identity_validation_remains_inside_preparation_budget(client, monkeypatch):
    seed(client, ("https://example.invalid/one", "https://example.invalid/two"), "null")
    before = snapshot()
    clock = [0.0]
    monkeypatch.setattr(management, "_clock", lambda: clock[0])
    monkeypatch.setattr(management, "BACKUP_PREPARE_SECONDS", 1)

    def delayed_identity(value):
        result = normalize_url(value)
        clock[0] = 2.0
        return result

    monkeypatch.setattr(management, "normalize_url", delayed_identity, raising=False)
    connections = watch_connections(monkeypatch)
    response = client.get("/data/backup")
    assert response.status_code == 503 and response.headers["retry-after"] == "5"
    assert snapshot() == before
    assert_released(client, connections)
