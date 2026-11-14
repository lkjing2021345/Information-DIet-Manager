"""Primary SQLite/HTTP failures survive secondary cleanup faults; synthetic DBs."""
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.hyh import app as api, data_management as management, db, export_io


PRIVATE = "SYNTHETIC_PRIVATE_DATABASE_FAILURE"
SCHEMA = Path(db.__file__).with_name("schema.sql")
CLEANUP_FAILURES = [
    pytest.param({"rollback"}, id="rollback"),
    pytest.param({"close"}, id="close"),
    pytest.param({"rollback", "close"}, id="rollback-and-close"),
]


def install_faults(monkeypatch, cleanup, primary_at=None):
    original_connect = sqlite3.connect
    faults = SimpleNamespace(
        connections=[], primary=sqlite3.OperationalError(PRIVATE + " primary"),
        rollback=sqlite3.OperationalError(PRIVATE + " rollback"),
        close=sqlite3.OperationalError(PRIVATE + " close"),
        connect=original_connect,
    )

    class ObservedConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if primary_at == "setup" and sql == "PRAGMA foreign_keys = ON":
                raise faults.primary
            return super().execute(sql, *args, **kwargs)

        def executescript(self, sql):
            try:
                return super().executescript(sql)
            except sqlite3.Error as error:
                faults.primary = error
                raise

        def commit(self):
            if primary_at == "commit":
                raise faults.primary
            return super().commit()

        def rollback(self):
            if "rollback" in cleanup:
                raise faults.rollback
            return super().rollback()

        def close(self):
            # Actually close the SQLite handle, then simulate its error report.
            # This does not claim every OS-level close failure releases a handle.
            super().close()
            if "close" in cleanup:
                raise faults.close

    def connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs, factory=ObservedConnection)
        faults.connections.append(connection)
        return connection

    monkeypatch.setattr(db.sqlite3, "connect", connect)
    return faults


def assert_closed(faults):
    assert faults.connections
    for connection in faults.connections:
        # A property avoids confusing a cross-thread execute error with closure.
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            _ = connection.in_transaction


def stored_state(path, connect=sqlite3.connect):
    connection = connect(path)
    try:
        schema = connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name").fetchall()
        names = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        rows = {name: connection.execute('SELECT * FROM "' + name + '" ORDER BY rowid').fetchall() for name in names}
        return schema, rows
    finally:
        connection.close()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "synthetic-api.sqlite3")
    streams = []

    def temporary_file():
        stream = tempfile.TemporaryFile(mode="w+b", dir=tmp_path)
        streams.append(stream)
        return stream

    monkeypatch.setattr(export_io, "_temporary_file", temporary_file)
    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000),
                    raise_server_exceptions=False,
                    headers={"Authorization": "Bearer " + "a" * 43, "Origin": "http://127.0.0.1:5173"}) as http:
        assert http.post("/collect", json={
            "url": "https://example.invalid/cleanup", "title": "Synthetic title",
            "text": "Synthetic body", "ts": 1790208000000, "source": "import",
        }).status_code == 200
        yield http, streams
    assert all(stream.closed for stream in streams)


@pytest.mark.parametrize("path", ["/export/lsj", "/data/backup"])
@pytest.mark.parametrize("status", [413, 409])
@pytest.mark.parametrize("cleanup", CLEANUP_FAILURES)
def test_export_and_backup_keep_primary_rejection_despite_cleanup_fault(client, monkeypatch, caplog, path, status, cleanup):
    http, streams = client
    if status == 409:
        with db.get_conn() as connection:
            # Both routes reject non-finite legacy JSON; raw export deliberately
            # tolerates some other historical metadata shapes.
            connection.execute("UPDATE items SET meta = ?", ('{"private":"' + PRIVATE + '","invalid":NaN}',))
    before = stored_state(db.DB_PATH)
    with monkeypatch.context() as scoped:
        if status == 413:
            scoped.setattr(export_io, "MAX_ROW_BYTES", 1)
            scoped.setattr(management, "MAX_BACKUP_ROW_BYTES", 1)
        faults = install_faults(scoped, cleanup)
        response = http.get(path)

    assert response.status_code == status
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    assert "retry-after" not in response.headers
    assert PRIVATE not in response.text + caplog.text
    assert_closed(faults)
    assert all(stream.closed for stream in streams)
    assert stored_state(db.DB_PATH) == before
    assert api.app.state.request_slots._value == 4
    assert api.app.state.export_slots._value == 2
    assert not api.app.state.operation_lock.locked()
    if status == 409:
        with db.get_conn() as connection:
            connection.execute("UPDATE items SET meta = NULL")
    assert http.get(path).status_code == 200


@pytest.fixture
def existing_database(tmp_path, monkeypatch):
    path = tmp_path / "synthetic-existing.sqlite3"
    monkeypatch.setattr(db, "DB_PATH", path)
    connection = sqlite3.connect(path)
    try:
        connection.executescript("CREATE TABLE durable(value TEXT); INSERT INTO durable VALUES('original');")
    finally:
        connection.close()
    return path


@pytest.mark.parametrize("primary_at", ["setup", "commit"])
@pytest.mark.parametrize("cleanup", CLEANUP_FAILURES)
def test_connection_setup_and_commit_keep_first_error_and_close(existing_database, monkeypatch, caplog, primary_at, cleanup):
    before = stored_state(existing_database)
    with monkeypatch.context() as scoped:
        faults = install_faults(scoped, cleanup, primary_at)
        with pytest.raises(sqlite3.OperationalError) as raised:
            with db.get_conn() as connection:
                connection.execute("INSERT INTO durable VALUES('must roll back')")
    assert raised.value is faults.primary
    assert_closed(faults)
    assert stored_state(existing_database) == before
    assert PRIVATE not in caplog.text


@pytest.mark.parametrize("cleanup", CLEANUP_FAILURES)
def test_initialization_keeps_real_schema_error_when_cleanup_also_fails(existing_database, tmp_path, monkeypatch, caplog, cleanup):
    broken_schema = tmp_path / "synthetic-invalid-schema.sql"
    broken_schema.write_text(SCHEMA.read_text(encoding="utf-8") + "\nINVALID SYNTHETIC SQL;", encoding="utf-8")
    before = stored_state(existing_database)
    with monkeypatch.context() as scoped:
        faults = install_faults(scoped, cleanup)
        with pytest.raises(sqlite3.OperationalError) as raised:
            db.init_db(broken_schema)
    assert raised.value is faults.primary
    assert_closed(faults)
    assert stored_state(existing_database) == before
    assert PRIVATE not in caplog.text


@pytest.mark.parametrize("operation", ["context", "initialization"])
def test_close_failure_without_primary_error_is_not_silenced(existing_database, monkeypatch, operation):
    with monkeypatch.context() as scoped:
        faults = install_faults(scoped, {"close"})
        with pytest.raises(sqlite3.OperationalError) as raised:
            if operation == "context":
                with db.get_conn() as connection:
                    connection.execute("INSERT INTO durable VALUES('committed before close')")
            else:
                db.init_db(SCHEMA)
    assert raised.value is faults.close
    assert_closed(faults)
    schema, rows = stored_state(existing_database)
    if operation == "context":
        assert rows["durable"] == [("original",), ("committed before close",)]
    else:
        assert rows["durable"] == [("original",)]
        assert any(row[1] == "items" for row in schema)


def test_rollback_system_interrupt_is_not_swallowed_and_still_attempts_close(existing_database, monkeypatch):
    before = stored_state(existing_database)
    interruption = KeyboardInterrupt("Synthetic cleanup interruption")
    with monkeypatch.context() as scoped:
        faults = install_faults(scoped, {"rollback"})
        faults.rollback = interruption
        with pytest.raises(KeyboardInterrupt) as raised:
            with db.get_conn() as connection:
                connection.execute("INSERT INTO durable VALUES('must roll back')")
                raise RuntimeError("Synthetic operation failure")
    assert raised.value is interruption
    assert_closed(faults)
    assert stored_state(existing_database) == before
