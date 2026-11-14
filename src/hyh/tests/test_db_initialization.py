"""Database path and atomic setup contracts using isolated synthetic databases."""
import importlib.util
from pathlib import Path
import sqlite3
import uuid

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "db.py"
SCHEMA = SOURCE.with_name("schema.sql")
PRIVATE_MARKER = "SYNTHETIC_PRIVATE_DATABASE_PATH"


def load_database_module():
    spec = importlib.util.spec_from_file_location("synthetic_db_" + uuid.uuid4().hex, SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def environment(monkeypatch, tmp_path):
    home = tmp_path / "用户 home"
    local = tmp_path / "用户 应用数据"
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.delenv("IDM_DB_PATH", raising=False)
    return home, local


def schema_rows(path):
    connection = sqlite3.connect(path)
    try:
        return connection.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").fetchall()
    finally:
        connection.close()


@pytest.mark.parametrize("local_value", [None, "", " ", "\t\r\n"])
def test_missing_or_blank_localappdata_has_one_user_default_across_working_directories(environment, monkeypatch, tmp_path, local_value):
    home, _local = environment
    if local_value is None:
        monkeypatch.delenv("LOCALAPPDATA")
    else:
        monkeypatch.setenv("LOCALAPPDATA", local_value)
    expected = home / ".local" / "share" / "InformationDietManager" / "idm.sqlite3"
    for index in range(2):
        cwd = tmp_path / f"launch-{index}"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        module = load_database_module()
        assert module.DB_PATH == expected
        module.init_db(SCHEMA)
        assert not (cwd / "InformationDietManager").exists()
    assert expected.is_file()


def test_normal_localappdata_uses_the_existing_application_directory(environment):
    _home, local = environment
    module = load_database_module()
    assert module.DB_PATH == local / "InformationDietManager" / "idm.sqlite3"
    assert not local.exists()  # Resolving configuration does not create files.


@pytest.mark.parametrize("value", ["", " ", "\t\r\n"])
def test_explicit_blank_database_path_fails_without_silent_default_or_files(environment, monkeypatch, value):
    home, local = environment
    monkeypatch.setenv("IDM_DB_PATH", value)
    with pytest.raises(RuntimeError) as raised:
        load_database_module()
    assert str(raised.value) == "IDM_DB_PATH must be a non-empty database file path."
    assert not home.exists() and not local.exists()


@pytest.mark.parametrize("kind", ["relative", "home", "absolute"])
def test_nonempty_explicit_paths_preserve_relative_home_and_unicode_semantics(environment, monkeypatch, tmp_path, kind):
    home, local = environment
    cwd = tmp_path / "启动目录"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    if kind == "relative":
        configured, expected = "相对目录/浏览记录.sqlite3", cwd / "相对目录" / "浏览记录.sqlite3"
    elif kind == "home":
        configured, expected = "~/私有目录/浏览记录.sqlite3", home / "私有目录" / "浏览记录.sqlite3"
    else:
        expected = tmp_path / "指定目录" / "浏览记录.sqlite3"
        configured = str(expected)
    monkeypatch.setenv("IDM_DB_PATH", configured)
    module = load_database_module()
    assert module.DB_PATH == expected
    module.init_db(SCHEMA)
    assert expected.is_file() and not local.exists()


def test_existing_directory_database_path_has_fixed_error_without_path_disclosure(environment, monkeypatch, tmp_path):
    directory = tmp_path / PRIVATE_MARKER
    directory.mkdir()
    monkeypatch.setenv("IDM_DB_PATH", str(directory))
    with pytest.raises(RuntimeError) as raised:
        load_database_module()
    assert str(raised.value) == "Database path must name a file, not an existing directory."
    assert PRIVATE_MARKER not in str(raised.value)
    assert list(directory.iterdir()) == []
    assert not directory.with_suffix(".lock").exists()
    assert not directory.with_suffix(".credentials.json").exists()


def test_successful_initialization_and_reinitialization_preserve_data_revision_and_schema(environment):
    module = load_database_module()
    module.init_db(SCHEMA)
    with module.get_conn() as connection:
        connection.execute("INSERT INTO items(url,title,ts,source,created_at) VALUES (?,?,?,?,?)",
                           ("https://example.invalid/initialization", "Synthetic row", 1, "import", 1))
        connection.execute("UPDATE items SET title = 'Updated synthetic row' WHERE id = 1")
        revision = tuple(connection.execute("SELECT * FROM items_revision").fetchone())
        item = tuple(connection.execute("SELECT * FROM items").fetchone())
    before_schema = schema_rows(module.DB_PATH)
    module.init_db(SCHEMA)
    assert schema_rows(module.DB_PATH) == before_schema
    with module.get_conn() as connection:
        assert tuple(connection.execute("SELECT * FROM items_revision").fetchone()) == revision
        assert tuple(connection.execute("SELECT * FROM items").fetchone()) == item
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("kind", ["missing", "invalid_utf8"])
def test_schema_read_failure_does_not_create_database_or_parent(environment, tmp_path, kind):
    module = load_database_module()
    schema = tmp_path / "synthetic-schema.sql"
    if kind == "invalid_utf8":
        schema.write_bytes(b"\xff")
    with pytest.raises((FileNotFoundError, UnicodeDecodeError)):
        module.init_db(schema)
    assert not module.DB_PATH.exists() and not module.DB_PATH.parent.exists()


def test_incompatible_existing_table_failure_leaves_no_partial_ddl_or_data_changes(environment):
    module = load_database_module()
    module.DB_PATH.parent.mkdir(parents=True)
    connection = sqlite3.connect(module.DB_PATH)
    try:
        connection.execute("CREATE TABLE analysis_jobs (id INTEGER PRIMARY KEY, status TEXT)")
        connection.execute("INSERT INTO analysis_jobs VALUES (1, 'synthetic-old')")
        connection.commit()
    finally:
        connection.close()
    before = schema_rows(module.DB_PATH)
    with pytest.raises(sqlite3.OperationalError, match="input_hash"):
        module.init_db(SCHEMA)
    assert schema_rows(module.DB_PATH) == before
    with module.get_conn() as connection:
        assert tuple(connection.execute("SELECT * FROM analysis_jobs").fetchone()) == (1, "synthetic-old")
        assert connection.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall() == []
        connection.execute("BEGIN EXCLUSIVE")  # Failed setup left no writer open.


def test_fresh_database_late_schema_error_rolls_back_every_schema_object(environment, tmp_path):
    module = load_database_module()
    schema = tmp_path / "synthetic-late-error.sql"
    schema.write_text(SCHEMA.read_text(encoding="utf-8") + "\nINVALID SYNTHETIC SQL;", encoding="utf-8")
    with pytest.raises(sqlite3.OperationalError):
        module.init_db(schema)
    assert schema_rows(module.DB_PATH) == []
    # The empty file can be initialized normally after the configuration is fixed.
    module.init_db(SCHEMA)
    assert any(row[1] == "items" for row in schema_rows(module.DB_PATH))


@pytest.mark.parametrize("failed", [False, True])
def test_initialization_always_closes_the_connection(environment, monkeypatch, tmp_path, failed):
    module = load_database_module()
    raw = sqlite3.connect(tmp_path / "observed-connection.sqlite3")
    real_script = raw.executescript
    closed = []

    class ObservedConnection:
        def execute(self, *args, **kwargs):
            return raw.execute(*args, **kwargs)

        def executescript(self, text):
            if failed:
                raise sqlite3.OperationalError("Synthetic initialization failure")
            return real_script(text)

        def rollback(self):
            return raw.rollback()

        def close(self):
            closed.append(True)
            raw.close()

    monkeypatch.setattr(module.sqlite3, "connect", lambda *_a, **_k: ObservedConnection())
    if failed:
        with pytest.raises(sqlite3.OperationalError, match="Synthetic initialization failure"):
            module.init_db(SCHEMA)
    else:
        module.init_db(SCHEMA)
    assert closed == [True]
    with pytest.raises(sqlite3.ProgrammingError):
        raw.execute("SELECT 1")


def test_failed_rollback_preserves_initialization_error_and_still_closes_connection(environment, monkeypatch, tmp_path):
    module = load_database_module()
    raw = sqlite3.connect(tmp_path / "rollback-failure.sqlite3")
    original = sqlite3.OperationalError("Synthetic original setup failure")

    class FailedConnection:
        def execute(self, *args, **kwargs):
            return raw.execute(*args, **kwargs)

        def executescript(self, _text):
            raw.execute("BEGIN IMMEDIATE")
            raw.execute("CREATE TABLE synthetic_partial (id INTEGER)")
            raise original

        def rollback(self):
            raise sqlite3.OperationalError("Synthetic secondary rollback failure")

        def close(self):
            raw.close()

    monkeypatch.setattr(module.sqlite3, "connect", lambda *_a, **_k: FailedConnection())
    with pytest.raises(sqlite3.OperationalError) as raised:
        module.init_db(SCHEMA)
    assert raised.value is original
    with pytest.raises(sqlite3.ProgrammingError):
        raw.execute("SELECT 1")
