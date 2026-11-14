"""Reject incompatible time-column structure without touching synthetic records."""
from contextlib import closing
from pathlib import Path
import sqlite3

from fastapi.testclient import TestClient
import pytest

from src.hyh import app as api, db


SCHEMA = Path(db.__file__).with_name("schema.sql")
LEGACY = Path(__file__).parent / "fixtures/historical_schemas/schema-3d68b00.sql"
ERROR = "Database items.ts must be an INTEGER-affinity table column; automatic migration is not supported."
PRIVATE = "SYNTHETIC_PRIVATE_EXISTING_RECORD"


@pytest.fixture
def database(tmp_path, monkeypatch):
    path = tmp_path / "synthetic-schema.sqlite3"
    monkeypatch.setattr(db, "DB_PATH", path)
    return path


def create_legacy(path, declared_type, *, table_name="items", column_name="ts", seed=True):
    source = LEGACY.read_text(encoding="utf-8")
    source = source.replace("ts INTEGER NOT NULL", f"{column_name} {declared_type} NOT NULL")
    source = source.replace("ON items (ts)", f"ON items ({column_name})")
    source = source.replace("CREATE TABLE IF NOT EXISTS items (", f"CREATE TABLE IF NOT EXISTS {table_name} (")
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(source)
        if seed:
            connection.execute(f"INSERT INTO items(url,title,{column_name},source,created_at) VALUES (?,?,?,?,?)",
                               ("https://example.invalid/legacy", PRIVATE, 5, "import", 6))
            connection.commit()


def snapshot(path):
    with closing(sqlite3.connect(path)) as connection:
        schema = connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name").fetchall()
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        rows = {name: connection.execute('SELECT * FROM "' + name.replace('"', '""') + '" ORDER BY rowid').fetchall()
                for name, in tables}
        return schema, rows


@pytest.mark.parametrize("declared_type", ["TEXT", "VARCHAR(20)", "CLOB", "BLOB", "", "REAL", "FLOAT",
                                            "DOUBLE PRECISION", "NUMERIC", "DECIMAL(10,2)", "BOOLEAN", "DATE", "STRING",
                                            pytest.param("ınt", id="non-ascii-dotless-i")])
def test_incompatible_affinity_fails_before_any_ddl_and_preserves_original_state(database, monkeypatch, declared_type):
    create_legacy(database, declared_type)
    if declared_type == "ınt":
        with closing(sqlite3.connect(database)) as connection:
            # Python upper() would turn this into INT; SQLite folds ASCII only.
            assert connection.execute("SELECT CAST(4.5 AS ınt), CAST(4.5 AS INT)").fetchone() == (4.5, 4)
    before = snapshot(database)
    trace = []
    real_connect = sqlite3.connect

    def observe(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connection.set_trace_callback(trace.append)
        return connection

    with monkeypatch.context() as scoped:
        scoped.setattr(db.sqlite3, "connect", observe)
        with pytest.raises(RuntimeError) as raised:
            db.init_db(SCHEMA)
    assert str(raised.value) == ERROR and PRIVATE not in str(raised.value)
    assert not any(keyword in statement.upper() for statement in trace for keyword in ("CREATE ", "INSERT ", "UPDATE ", "DELETE "))
    assert snapshot(database) == before
    with closing(real_connect(database)) as connection:
        connection.execute("BEGIN EXCLUSIVE")  # Rejection released its connection.
        connection.rollback()


@pytest.mark.parametrize("kind", ["missing-ts", "view", "mixed-case-text"])
def test_invalid_items_object_or_missing_time_column_is_rejected_safely(database, kind):
    if kind == "missing-ts":
        create_legacy(database, "INTEGER", column_name="legacy_ts")
    elif kind == "mixed-case-text":
        create_legacy(database, "TEXT", table_name="ItEmS", column_name="Ts")
    else:
        with closing(sqlite3.connect(database)) as connection:
            source = LEGACY.read_text(encoding="utf-8").replace("items", "synthetic_pages")
            connection.executescript(source + "\nCREATE VIEW items AS SELECT * FROM synthetic_pages;")
            connection.execute("INSERT INTO synthetic_pages(url,title,ts,source,created_at) VALUES (?,?,?,?,?)",
                               ("https://example.invalid/view", PRIVATE, 5, "import", 6))
            connection.commit()
    before = snapshot(database)
    with pytest.raises(RuntimeError) as raised:
        db.init_db(SCHEMA)
    assert str(raised.value) == ERROR
    assert snapshot(database) == before


@pytest.mark.parametrize("declared_type", ["INTEGER", "INT", "bigint", "UNSIGNED BIG INT", "TINYINT", "INT2",
                                            "INT8", "CHARINT", "FLOATING POINT", "INTEGER(8)"])
def test_sqlite_integer_affinity_spellings_keep_http_counts_and_repeat_initialization(database, monkeypatch, declared_type):
    create_legacy(database, declared_type, table_name="ItEmS", column_name="Ts", seed=False)
    # SQLite checks INT before CHAR/TEXT/REAL rules: FLOATING POINT and CHARINT
    # intentionally belong here. Do not equate compatibility with exact spelling.
    monkeypatch.setattr(api, "_execute_lsj_pipeline", lambda rows: {"ok": False, "warning": "synthetic unavailable"})
    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 51234),
                    headers={"Authorization": "Bearer " + "a" * 43}) as client:
        for ts in range(1, 6):
            response = client.post("/collect", json={"url": f"https://example.invalid/{ts}",
                "title": f"Synthetic {ts}", "text": "Synthetic record", "ts": ts, "source": "import"})
            assert response.status_code == 200 and response.json()["inserted"] == 1
        response = client.get("/dashboard/visualization", params={"from_ts": 0, "to_ts": 10})
        assert response.status_code == 200
        assert response.json()["window"]["input_count"] == response.json()["window"]["available_count"] == 5
        assert client.get("/items").json()["total"] == 5
        assert client.get("/data/backup").status_code == 200
    before = snapshot(database)
    db.init_db(SCHEMA)
    db.init_db(SCHEMA)
    assert snapshot(database) == before
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT DISTINCT typeof(ts) FROM items").fetchall() == [("integer",)]


def test_bad_timestamp_table_cannot_start_healthy_api(database):
    create_legacy(database, "TEXT")
    before = snapshot(database)
    with pytest.raises(RuntimeError) as raised:
        with TestClient(api.app) as _client:
            pytest.fail("An incompatible timestamp table must not become a ready service")
    assert str(raised.value) == ERROR
    assert snapshot(database) == before


def test_compatible_affinity_late_ddl_failure_still_rolls_back(database, tmp_path):
    create_legacy(database, "FLOATING POINT")
    before = snapshot(database)
    broken = tmp_path / "synthetic-late-error.sql"
    broken.write_text(SCHEMA.read_text(encoding="utf-8") + "\nINVALID SYNTHETIC SQL;", encoding="utf-8")
    with pytest.raises(sqlite3.OperationalError):
        db.init_db(broken)
    assert snapshot(database) == before


def test_fresh_database_and_repeated_initialization_keep_the_original_schema(database):
    assert not database.exists()
    db.init_db(SCHEMA)
    before = snapshot(database)
    db.init_db(SCHEMA)
    assert snapshot(database) == before
    assert before[1]["items"] == []
