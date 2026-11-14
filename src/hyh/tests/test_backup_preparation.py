"""Preparation budgets must apply before loading or decoding legacy records."""
from contextlib import contextmanager
import hashlib
import json
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient

from src.hyh import app as api, data_management as management, db
from src.hyh.models import IngestItem, MAX_META_BYTES


BASE_TS = 1790208000000


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "synthetic-backup-preparation.sqlite3")
    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 51000),
                    headers={"Authorization": "Bearer " + "a" * 43}) as http:
        yield http


def collect(client, index=0):
    response = client.post("/collect", json={
        "url": f"https://example.com/preparation/{index}", "title": f"Synthetic {index}",
        "text": "Synthetic body", "ts": BASE_TS + index, "source": "import", "channel": "edu",
    })
    assert response.status_code == 200 and response.json()["inserted"] == 1


def watch_connections(monkeypatch, *, row_seen=None):
    original = management.get_conn
    connections = []

    @contextmanager
    def watched(*args, **kwargs):
        with original(*args, **kwargs) as conn:
            connections.append(conn)
            if row_seen is not None:
                def row_factory(cursor, row):
                    row_seen(cursor, row)
                    return sqlite3.Row(cursor, row)
                conn.row_factory = row_factory
            yield conn

    monkeypatch.setattr(management, "get_conn", watched)
    return connections


def assert_closed_and_writable(client, connections):
    assert connections
    for conn in connections:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")
    assert not api.app.state.operation_lock.locked()
    assert api.app.state.export_slots._value == 2
    assert api.app.state.request_slots._value == 4
    collect(client, 99)


def test_backup_does_not_materialize_unexported_legacy_hash_columns(client, monkeypatch):
    collect(client)
    with db.get_conn() as conn:
        conn.execute("UPDATE items SET content_hash = ?, url_hash = ?", ("h" * 200000, "u" * 200000))
    fetched_columns = []
    connections = watch_connections(monkeypatch, row_seen=lambda cursor, _row:
                                    fetched_columns.extend(column[0] for column in cursor.description))
    response = client.get("/data/backup")
    assert response.status_code == 200 and len(response.json()["items"]) == 1
    assert {"content_hash", "url_hash"}.isdisjoint(fetched_columns)
    assert_closed_and_writable(client, connections)


@pytest.mark.parametrize("field", management.FIELDS)
def test_oversized_stored_fields_are_rejected_before_python_materialization(client, monkeypatch, field):
    collect(client)
    oversized = 128 * 1024 + 1
    with db.get_conn() as conn:
        # Simulate an externally written legacy DB without relying on its
        # original CHECK constraints. This pragma is local to this test handle.
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute(f"UPDATE items SET {field} = ?", ("x" * oversized,))
    loaded_large_values = []
    connections = watch_connections(monkeypatch, row_seen=lambda _cursor, row:
        loaded_large_values.extend(len(value) for value in row if isinstance(value, (str, bytes)) and len(value) >= oversized))
    response = client.get("/data/backup")
    assert response.status_code == 413 and not loaded_large_values
    assert response.headers["cache-control"] == "no-store"
    assert "xxxx" not in response.text
    assert_closed_and_writable(client, connections)


def test_row_budget_is_the_sum_of_all_stored_fields(client, monkeypatch):
    collect(client)
    # Each field is below 128 KiB; the complete record is not.
    with db.get_conn() as conn:
        conn.execute("UPDATE items SET title = ?, text = ?", ("t" * 70000, "b" * 70000))
    loaded = []
    connections = watch_connections(monkeypatch, row_seen=lambda cursor, _row:
                                    loaded.extend(column[0] for column in cursor.description))
    response = client.get("/data/backup")
    assert response.status_code == 413
    assert "title" not in loaded and "text" not in loaded
    assert_closed_and_writable(client, connections)


@pytest.mark.parametrize("extra,status", [(0, 200), (1, 413)])
def test_stored_row_byte_limit_includes_exact_boundary_and_legacy_json_whitespace(client, extra, status):
    collect(client)
    with db.get_conn() as conn:
        conn.execute("UPDATE items SET meta = ?", ('{"ok":true}',))
        row = conn.execute(f"SELECT {','.join(management.FIELDS)} FROM items").fetchone()
        size = sum(len(str(value).encode("utf-8")) for value in row if value is not None)
        padded = '{"ok":true}' + " " * (128 * 1024 + extra - size)
        conn.execute("UPDATE items SET meta = ?", (padded,))
    response = client.get("/data/backup")
    assert response.status_code == status
    if status == 200:
        assert response.json()["items"][0]["meta"] == {"ok": True}


def test_row_budget_counts_utf8_bytes_and_rejects_json_expansion_before_decoding(client, monkeypatch):
    collect(client)
    raw = '{"synthetic_values":[' + ",".join(["123456"] * 500000) + "]}"
    with db.get_conn() as conn:
        conn.execute("UPDATE items SET meta = ?", (raw,))
    decoded = []
    loads = management.json.loads

    def tracked_loads(value, *args, **kwargs):
        if isinstance(value, str) and value.startswith('{"synthetic_values":'):
            decoded.append(len(value))
        return loads(value, *args, **kwargs)

    monkeypatch.setattr(management.json, "loads", tracked_loads)
    connections = watch_connections(monkeypatch)
    response = client.get("/data/backup")
    assert response.status_code == 413 and decoded == []
    # A character count would miss this separately stored multibyte field.
    with db.get_conn() as conn:
        conn.execute("UPDATE items SET meta = NULL, text = ?", ("文" * 50000,))
    response = client.get("/data/backup")
    assert response.status_code == 413
    assert_closed_and_writable(client, connections)


def test_legitimate_maximum_fields_and_escaped_json_fit_the_stored_row_budget(client):
    meta = {"padding": "😀" * ((MAX_META_BYTES - len('{"padding":""}')) // 4)}
    record = IngestItem.model_validate({
        "url": "https://example.com/" + "x" * (2048 - len("https://example.com/")),
        "title": "😀" * 300, "text": "😀" * 2000, "ts": BASE_TS, "source": "import",
        "lang": "😀" * 16, "channel": "😀" * 32, "author": "😀" * 120,
        "tags": ["😀" * 40 for _ in range(10)], "meta": meta,
    }).model_dump(mode="json")
    stored = {**record, "tags": json.dumps(record["tags"], ensure_ascii=True),
              "meta": json.dumps(record["meta"], ensure_ascii=True), "created_at": BASE_TS}
    fields = [*management.FIELDS, "created_at"]
    with db.get_conn() as conn:
        conn.execute(f"INSERT INTO items({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                     [stored[field] for field in fields])
    response = client.get("/data/backup")
    assert response.status_code == 200 and response.json()["items"] == [record]
    assert response.json()["sha256"] == hashlib.sha256(management.canonical_items([record])).hexdigest()


def test_expired_preparation_does_not_open_database_or_allocate_download_buffer(client, monkeypatch):
    monkeypatch.setattr(management, "BACKUP_PREPARE_SECONDS", 0)

    def unexpected(*args, **kwargs):
        pytest.fail("Expired backup must not enter database or allocate response buffer")

    monkeypatch.setattr(management, "get_conn", unexpected)
    monkeypatch.setattr(management, "BytesIO", unexpected)
    response = client.get("/data/backup")
    assert response.status_code == 503 and response.headers["retry-after"] == "5"
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("phase", ["digest", "encode"])
def test_preparation_expiry_after_final_cpu_phase_never_starts_a_download(client, monkeypatch, phase):
    collect(client)
    clock = [0.0]
    monkeypatch.setattr(management, "_clock", lambda: clock[0])
    monkeypatch.setattr(management, "BACKUP_PREPARE_SECONDS", 1)
    original = management.hashlib.sha256 if phase == "digest" else management.json.dumps

    def expire(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        if ((phase == "digest" and isinstance(value, bytes) and value.startswith(b"["))
                or (phase == "encode" and isinstance(value, dict) and value.get("format") == "idm-page-records")):
            clock[0] = 2.0
        return result

    monkeypatch.setattr(management.hashlib if phase == "digest" else management.json,
                        "sha256" if phase == "digest" else "dumps", expire)
    monkeypatch.setattr(management, "BytesIO", lambda _body: pytest.fail("No download buffer after expiry"))
    connections = watch_connections(monkeypatch)
    response = client.get("/data/backup")
    assert response.status_code == 503 and response.headers["retry-after"] == "5"
    assert_closed_and_writable(client, connections)


def test_expiry_during_download_buffer_allocation_closes_buffer_without_sending(client, monkeypatch):
    collect(client)
    clock, buffers = [0.0], []
    monkeypatch.setattr(management, "_clock", lambda: clock[0])
    monkeypatch.setattr(management, "BACKUP_PREPARE_SECONDS", 1)
    allocate = management.BytesIO

    def delayed_allocation(body):
        stream = allocate(body)
        buffers.append(stream)
        clock[0] = 2.0
        return stream

    monkeypatch.setattr(management, "BytesIO", delayed_allocation)
    connections = watch_connections(monkeypatch)
    response = client.get("/data/backup")
    assert response.status_code == 503 and response.headers["retry-after"] == "5"
    assert len(buffers) == 1 and buffers[0].closed
    assert_closed_and_writable(client, connections)


def test_expiry_while_processing_row_never_loads_the_next_record(client, monkeypatch):
    collect(client, 0)
    collect(client, 1)
    clock = [0.0]
    monkeypatch.setattr(management, "_clock", lambda: clock[0])
    monkeypatch.setattr(management, "BACKUP_PREPARE_SECONDS", 1)
    encode = management.canonical_items

    def expire_after_encoding(value):
        encoded = encode(value)
        if isinstance(value, dict):
            clock[0] = 2.0
        return encoded

    monkeypatch.setattr(management, "canonical_items", expire_after_encoding)
    monkeypatch.setattr(management, "BytesIO", lambda _body: pytest.fail("No download buffer after expiry"))
    loaded_records = []

    def saw_row(cursor, row):
        columns = [column[0] for column in cursor.description]
        if "url" in columns:
            loaded_records.append(row[columns.index("url")])

    connections = watch_connections(monkeypatch, row_seen=saw_row)
    response = client.get("/data/backup")
    assert response.status_code == 503 and response.headers["retry-after"] == "5"
    assert loaded_records == ["https://example.com/preparation/0"]
    assert_closed_and_writable(client, connections)


def test_sqlite_progress_deadline_interrupts_query_and_clears_connection(client, monkeypatch):
    collect(client)
    clock = [0.0]
    monkeypatch.setattr(management, "_clock", lambda: clock[0])
    monkeypatch.setattr(management, "BACKUP_PREPARE_SECONDS", 1)
    original = management.get_conn
    raw_connections, progress_calls = [], []

    class SlowCount:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def set_progress_handler(self, callback, instructions):
            if callback is None:
                return self.connection.set_progress_handler(None, instructions)

            def observe():
                progress_calls.append(True)
                return callback()

            return self.connection.set_progress_handler(observe, instructions)

        def execute(self, sql, *args):
            if sql.strip().upper().startswith("SELECT COUNT(*) FROM ITEMS"):
                clock[0] = 2.0
                sql = "WITH RECURSIVE counter(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM counter WHERE x<1000000) SELECT count(*) FROM counter"
            return self.connection.execute(sql, *args)

    @contextmanager
    def slow_connection(*args, **kwargs):
        with original(*args, **kwargs) as conn:
            raw_connections.append(conn)
            yield SlowCount(conn)

    monkeypatch.setattr(management, "get_conn", slow_connection)
    response = client.get("/data/backup")
    assert response.status_code == 503 and progress_calls
    assert response.headers["retry-after"] == "5"
    assert_closed_and_writable(client, raw_connections)


def test_sqlite_busy_wait_respects_remaining_preparation_budget(client, monkeypatch):
    collect(client)
    monkeypatch.setattr(management, "BACKUP_PREPARE_SECONDS", 0.1)
    external = sqlite3.connect(db.DB_PATH)
    try:
        external.execute("BEGIN EXCLUSIVE")
        started = time.monotonic()
        response = client.get("/data/backup")
        elapsed = time.monotonic() - started
    finally:
        external.rollback()
        external.close()
    assert response.status_code == 503 and response.headers["retry-after"] == "5"
    # Generous scheduling tolerance distinguishes a deadline-aware wait from
    # sqlite3's otherwise five-second default; this is not a real-time claim.
    assert elapsed < 1.5
    assert not api.app.state.operation_lock.locked()
    collect(client, 99)


@pytest.mark.parametrize("failure", ["foreign_keys", "secure_delete", "rollback"])
def test_connection_setup_and_rollback_failures_always_close_handle(tmp_path, monkeypatch, failure):
    raw = sqlite3.connect(tmp_path / "synthetic-initialization.sqlite3")
    seen_timeouts = []

    class FailedConnection:
        def execute(self, sql):
            if failure in sql:
                raise sqlite3.OperationalError("Synthetic setup failure")
            return raw.execute(sql)

        def rollback(self):
            if failure == "rollback":
                raise sqlite3.OperationalError("Synthetic rollback failure")
            return raw.rollback()

        def close(self):
            return raw.close()

    def connect(_path, *, timeout):
        seen_timeouts.append(timeout)
        return FailedConnection()

    monkeypatch.setattr(db.sqlite3, "connect", connect)
    expected_error = RuntimeError if failure == "rollback" else sqlite3.OperationalError
    with pytest.raises(expected_error, match="Synthetic body failure" if failure == "rollback" else "Synthetic setup failure"):
        with db.get_conn(timeout=0.125):
            if failure == "rollback":
                raise RuntimeError("Synthetic body failure")
            pytest.fail("Failed setup must not yield a connection")
    assert seen_timeouts == [0.125]
    with pytest.raises(sqlite3.ProgrammingError):
        raw.execute("SELECT 1")
