"""Real SQLite/API interleavings; no personal records or model inference."""
import pytest
from fastapi.testclient import TestClient

from src.hyh import app as api, db, item_pagination
from src.hyh.models import IngestItem


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "pagination.sqlite3")
    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000),
                    headers={"Authorization": "Bearer " + "a" * 43}) as client:
        yield client


def seed(count=53, start=0):
    api.insert_items([IngestItem(url=f"https://example.invalid/cursor/{n}", title=f"Synthetic {n}",
                               ts=1790208000000 + n, source="import") for n in range(start, start + count)])


def page(client, cursor=None, size=50):
    params = {"pagination": "cursor", "page_size": size}
    if cursor is not None:
        params["cursor"] = cursor
    return client.get("/items", params=params)


def test_inserts_between_pages_do_not_shift_the_snapshot(client):
    seed()
    first = page(client).json()
    assert first["total"] == 53 and first["has_more"]
    seed(1, start=100)
    second = page(client, first["next_cursor"]).json()
    assert second["total"] == 53 and second["page"] == 2
    assert second["snapshot_at"] == first["snapshot_at"]
    assert not second["has_more"] and second["next_cursor"] is None
    ids = [row["id"] for row in first["items"] + second["items"]]
    assert ids == list(range(53, 0, -1))  # exactly once, in order
    refreshed = page(client).json()
    assert refreshed["total"] == 54 and refreshed["items"][0]["id"] == 54


@pytest.mark.parametrize("operation", ["delete_seen", "delete_unseen", "delete_all", "restore", "update"])
def test_destructive_changes_expire_the_cursor(client, operation):
    seed()
    first = page(client).json()
    if operation.startswith("delete_") and operation != "delete_all":
        target = 53 if operation == "delete_seen" else 1
        assert client.delete(f"/items/{target}", headers={"X-IDM-Confirm": "delete-record"}).status_code == 200
    elif operation == "delete_all":
        assert client.delete("/data", headers={"X-IDM-Confirm": "delete-all"}).status_code == 200
    elif operation == "restore":
        backup = client.get("/data/backup").json()
        assert client.post("/data/restore", json=backup, headers={"X-IDM-Confirm": "replace-records"}).status_code == 200
    else:
        with db.get_conn() as conn:
            conn.execute("UPDATE items SET title = ? WHERE id = 53", ("Updated synthetic record",))
    expired = page(client, first["next_cursor"])
    assert expired.status_code == 409
    assert expired.json()["detail"]["code"] == "items_snapshot_expired"
    assert page(client).status_code == 200


def test_rolled_back_deletion_does_not_expire_the_snapshot(client):
    seed()
    first = page(client).json()
    with db.get_conn() as conn:
        conn.execute("DELETE FROM items WHERE id = 1")
        conn.rollback()
    second = page(client, first["next_cursor"])
    assert second.status_code == 200 and len(second.json()["items"]) == 3


def test_schema_reinitialization_preserves_revision_and_existing_cursor(client):
    seed()
    first = page(client).json()
    db.init_db(api._schema_path())
    assert page(client, first["next_cursor"]).status_code == 200


def test_cursor_after_process_restart_requires_refresh(client, monkeypatch):
    seed()
    first = page(client).json()
    monkeypatch.setattr(item_pagination, "_SIGNING_KEY", b"synthetic-restart-key")
    assert page(client, first["next_cursor"]).status_code == 409


@pytest.mark.parametrize("count,size", [(0, 50), (1, 50), (50, 50), (53, 1)])
def test_page_boundaries_and_empty_state(client, count, size):
    seed(count)
    result = page(client, size=size).json()
    loaded = result["items"][:]
    while result["has_more"]:
        result = page(client, result["next_cursor"], size).json()
        loaded.extend(result["items"])
    assert result["total"] == count and result["next_cursor"] is None
    assert [r["id"] for r in loaded] == list(range(count, 0, -1))


@pytest.mark.parametrize("cursor", ["garbage!", "é", "a"])
def test_malformed_cursor_is_rejected(client, cursor):
    assert page(client, cursor).status_code == 400


def test_changed_page_size_or_mixed_pagination_is_rejected(client):
    seed()
    cursor = page(client).json()["next_cursor"]
    assert page(client, cursor, size=10).status_code == 400
    assert client.get("/items", params={"cursor": cursor}).status_code == 400
    assert client.get("/items?pagination=cursor&page=2").status_code == 400
    assert client.get("/items?pagination=cursor&limit=50").status_code == 400


def test_huge_legacy_page_is_empty_instead_of_sql_overflow(client):
    assert client.get("/items", params={"page": 2**64}).json()["items"] == []
