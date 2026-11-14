"""Stored page JSON is preserved or rejected explicitly; only synthetic DBs."""
import json

import pytest
from fastapi.testclient import TestClient

from src.hyh import app as api, db


PRIVATE_MARKER = "SYNTHETIC_PRIVATE_ITEM_CONTENT"
INVALID_KINDS = ("NaN", "Infinity", "-Infinity", "1e999", "-1e999", "malformed",
                 "empty", "surrogate_value", "surrogate_key", "depth65", "depth300", "depth2000", "invalid_utf8")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "synthetic-item-integrity.sqlite3")
    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 51000),
                    headers={"Authorization": "Bearer " + "a" * 43}, raise_server_exceptions=False) as http:
        response = http.post("/collect", json={
            "url": "https://example.invalid/item-integrity/0", "title": PRIVATE_MARKER,
            "text": PRIVATE_MARKER, "ts": 1790208000000, "source": "import",
        })
        assert response.status_code == 200 and response.json()["inserted"] == 1
        yield http


def stored_state():
    with db.get_conn() as conn:
        return ([tuple(row) for row in conn.execute("SELECT * FROM items ORDER BY id")],
                tuple(conn.execute("SELECT * FROM items_revision").fetchone()))


def get_page(client, pagination, **params):
    return client.get("/items", params={"pagination": pagination, **params})


def assert_invalid(response):
    assert response.status_code == 409
    assert response.json() == {"detail": {"code": "stored_item_invalid", "message": "Stored item JSON is invalid."}}
    assert PRIVATE_MARKER not in response.text
    assert "no-store" in response.headers["cache-control"]


def corrupted_value(kind):
    if kind == "empty":
        return ""
    if kind == "malformed":
        return '{"private":"' + PRIVATE_MARKER + '","v":'
    if kind == "invalid_utf8":
        return b'{"private":"' + PRIVATE_MARKER.encode() + b'","v":"\xff"}'
    if kind == "surrogate_value":
        return '{"private":"' + PRIVATE_MARKER + '","v":"\\ud800"}'
    if kind == "surrogate_key":
        return '{"\\udfff":"' + PRIVATE_MARKER + '"}'
    if kind.startswith("depth"):
        depth = int(kind.removeprefix("depth"))
        return "[" * depth + json.dumps(PRIVATE_MARKER) + "]" * depth
    return '{"private":"' + PRIVATE_MARKER + '","v":' + kind + "}"


@pytest.mark.parametrize("pagination", ["offset", "cursor"])
@pytest.mark.parametrize("field", ["tags", "meta"])
@pytest.mark.parametrize("kind", INVALID_KINDS)
def test_invalid_stored_item_json_is_explicit_without_rewriting_rows(client, pagination, field, kind, caplog):
    value = corrupted_value(kind)
    with db.get_conn() as conn:
        conn.execute(f"UPDATE items SET {field} = ? WHERE id = 1", (value,))
    before = stored_state()
    response = get_page(client, pagination)
    assert_invalid(response)
    assert PRIVATE_MARKER not in caplog.text
    assert stored_state() == before


@pytest.mark.parametrize("pagination", ["offset", "cursor"])
@pytest.mark.parametrize("value", [None, "null", "{}", "[]", "0", "-0.0", "1e308", "false", '""', '"标签 😀"',
                                  "1234567890123456789012345678901234567890", '["\\ud83d\\ude00"]',
                                  '{"中文 😀":[0,null,false,"文本 😀"]}', '[[{"nested":true}]]',
                                  "[" * 64 + "0" + "]" * 64, b'{"utf8":"\xe6\x96\x87"}'])
def test_valid_json_shapes_and_missing_values_are_preserved(client, pagination, value):
    with db.get_conn() as conn:
        conn.execute("UPDATE items SET tags = ?, meta = ? WHERE id = 1", (value, value))
    before = stored_state()
    response = get_page(client, pagination)
    assert response.status_code == 200
    expected = None if value is None else json.loads(value)
    result = response.json()
    assert result["items"][0]["tags"] == result["items"][0]["meta"] == expected
    for field in ("tags", "meta"):
        assert json.dumps(result["items"][0][field], ensure_ascii=False, allow_nan=False) == json.dumps(
            expected, ensure_ascii=False, allow_nan=False)
    assert result["items"][0]["title"] == PRIVATE_MARKER
    assert result["total"] == 1 and result["page"] == 1
    assert stored_state() == before


@pytest.mark.parametrize("pagination", ["offset", "cursor"])
def test_corruption_on_later_page_is_not_hidden_or_reported_as_cursor_expiry(client, pagination):
    with db.get_conn() as conn:
        conn.execute("UPDATE items SET meta = ? WHERE id = 1", (corrupted_value("NaN"),))
    collected = client.post("/collect", json={
        "url": "https://example.invalid/item-integrity/1", "title": "New valid record",
        "ts": 1790208000001, "source": "import",
    })
    assert collected.status_code == 200 and collected.json()["inserted"] == 1
    before = stored_state()
    first = get_page(client, pagination, page_size=1)
    assert first.status_code == 200 and first.json()["items"][0]["id"] == 2
    parameters = {"cursor": first.json()["next_cursor"]} if pagination == "cursor" else {"page": 2}
    assert_invalid(get_page(client, pagination, page_size=1, **parameters))
    assert stored_state() == before
    assert not api.app.state.operation_lock.locked()
    assert api.app.state.request_slots._value == 4
    # A read conflict must leave subsequent authorized writes available.
    deleted = client.delete("/items/1", headers={"X-IDM-Confirm": "delete-record"})
    assert deleted.status_code == 200
    assert get_page(client, pagination).status_code == 200
