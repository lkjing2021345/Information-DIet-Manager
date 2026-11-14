"""Lock-path alias protection through real launchers and synthetic files only."""
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from src.hyh.security import credential_path, process_ownership


ROOT = Path(__file__).resolve().parents[3]
PRIVATE_MARKER = "SYNTHETIC_PRIVATE_LOCK_PATH"
ERROR = "Database path conflicts with its process lock file; choose a different database filename."
WINDOWS = pytest.mark.skipif(os.name != "nt", reason="Win32 filename alias behavior")
COLLIDING_NAMES = [
    "synthetic.lock",
    pytest.param("synthetic.LOCK", marks=WINDOWS),
    pytest.param("synthetic.lock ", marks=WINDOWS),
]


@pytest.mark.parametrize("name", COLLIDING_NAMES)
def test_lock_collision_is_rejected_before_creating_parent_or_files(tmp_path, name):
    parent = tmp_path / PRIVATE_MARKER
    with pytest.raises(RuntimeError) as raised:
        with process_ownership(parent / name):
            pytest.fail("A database must never become its own process lock")
    assert str(raised.value) == ERROR
    assert PRIVATE_MARKER not in str(raised.value)
    assert not parent.exists()


def make_database(path):
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE synthetic(value TEXT)")
        connection.execute("INSERT INTO synthetic VALUES ('Synthetic existing record')")
        connection.commit()
    finally:
        connection.close()


@pytest.mark.parametrize("script", ["run_backend.py", "rotate_local_keys.py"])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("name", COLLIDING_NAMES)
def test_real_commands_reject_colliding_paths_without_creating_or_changing_files(
        tmp_path, script, existing, name):
    parent = tmp_path / PRIVATE_MARKER
    path = parent / name
    if existing:
        parent.mkdir()
        make_database(path)
    before = {file.name: file.read_bytes() for file in parent.iterdir()} if existing else None
    env = {**os.environ, "IDM_DB_PATH": str(path), "PYTHONUTF8": "1"}
    for key in ("IDM_ADMIN_TOKEN", "IDM_COLLECTOR_TOKEN", "IDM_FRONTEND_ORIGINS"):
        env.pop(key, None)

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, env=env,
        capture_output=True, timeout=20,
    )

    output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    assert result.returncode != 0
    assert ERROR in output
    assert PRIVATE_MARKER not in output
    assert "Keys rotated:" not in output
    if existing:
        assert {file.name: file.read_bytes() for file in parent.iterdir()} == before
        connection = sqlite3.connect(path)
        try:
            assert connection.execute("SELECT value FROM synthetic").fetchall() == [("Synthetic existing record",)]
        finally:
            connection.close()
    else:
        assert not parent.exists()


def test_normal_database_keeps_existing_lock_and_credential_names(tmp_path):
    path = tmp_path / "普通 目录" / "浏览记录.sqlite3"
    with process_ownership(path):
        assert path.with_suffix(".lock").is_file()
        assert credential_path(path).name == "浏览记录.credentials.json"
        make_database(path)
    assert path.with_suffix(".lock").read_bytes() == b"0"
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT value FROM synthetic").fetchone() == ("Synthetic existing record",)
    finally:
        connection.close()


def test_distinct_database_stems_keep_independent_process_locks(tmp_path):
    first, second = tmp_path / "first.sqlite3", tmp_path / "second.sqlite3"
    with process_ownership(first), process_ownership(second):
        assert first.with_suffix(".lock").is_file()
        assert second.with_suffix(".lock").is_file()
        with pytest.raises(RuntimeError, match="Another IDM process"):
            with process_ownership(first):
                pytest.fail("The original database must remain owned")


def test_same_stem_sidecar_namespace_is_not_silently_migrated(tmp_path):
    first, second = tmp_path / "same.sqlite3", tmp_path / "same.db"
    assert credential_path(first) == credential_path(second)
    with process_ownership(first):
        with pytest.raises(RuntimeError, match="Another IDM process"):
            with process_ownership(second):
                pytest.fail("The existing shared sidecar namespace must not change")


@WINDOWS
def test_trailing_dot_alias_is_allowed_when_derived_lock_is_actually_different(tmp_path):
    path = (tmp_path / "distinct.lock.").resolve()
    # resolve() retains this spelling; Win32 opens the database as distinct.lock.
    assert path.name.endswith(".")
    path.write_bytes(b"Synthetic existing file")
    before = path.read_bytes()
    with process_ownership(path):
        assert not path.samefile(path.with_suffix(".lock"))
        assert path.read_bytes() == before


@WINDOWS
def test_extended_windows_path_retains_literal_trailing_space(tmp_path):
    path = Path("\\\\?\\" + str(tmp_path.resolve())) / "literal.lock "
    path.write_bytes(b"Synthetic literal trailing-space filename")
    before = path.read_bytes()
    try:
        with process_ownership(path):
            assert not path.samefile(path.with_suffix(".lock"))
            assert path.read_bytes() == before
    finally:
        # Use the extended spelling to remove the literal trailing-space file.
        path.unlink()
