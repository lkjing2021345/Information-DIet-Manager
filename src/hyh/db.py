from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


def _default_db_path() -> Path:
    configured = os.environ.get("LOCALAPPDATA")
    base = Path(configured) if configured and configured.strip() else Path.home() / ".local" / "share"
    return base / "InformationDietManager" / "idm.sqlite3"


def _configured_db_path() -> Path:
    configured = os.environ.get("IDM_DB_PATH")
    if configured is not None and not configured.strip():
        raise RuntimeError("IDM_DB_PATH must be a non-empty database file path.")
    path = (Path(configured) if configured is not None else _default_db_path()).expanduser().resolve()
    if path.is_dir():
        raise RuntimeError("Database path must name a file, not an existing directory.")
    return path


DB_PATH = _configured_db_path()


@contextmanager
def _connection_scope(conn: sqlite3.Connection) -> Generator[sqlite3.Connection, None, None]:
    try:
        yield conn
    except BaseException:
        # Cleanup must not replace the first business/setup/commit failure.
        # Even a system-level interruption in rollback must attempt close.
        try:
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
        raise
    else:
        # A standalone close failure is still a failure, even after commit.
        conn.close()


def _validate_existing_timestamp_column(conn: sqlite3.Connection) -> None:
    """Reject known-unsafe time affinity before additive initialization writes."""
    existing = conn.execute(
        "SELECT type FROM sqlite_master WHERE name = 'items' COLLATE NOCASE AND type IN ('table', 'view')"
    ).fetchone()
    if existing is None:
        return
    declared_type = None
    if existing[0] == "table":
        declared_type = next(
            (row[2] for row in conn.execute("PRAGMA main.table_info(items)") if row[1].lower() == "ts"),
            None,
        )
    # SQLite's first affinity rule is ASCII-case-insensitive containment of INT,
    # before the TEXT/REAL rules (so CHARINT and FLOATING POINT also qualify).
    if declared_type is None or re.search("INT", declared_type, re.IGNORECASE | re.ASCII) is None:
        raise RuntimeError(
            "Database items.ts must be an INTEGER-affinity table column; automatic migration is not supported."
        )


def init_db(schema_path: Path) -> None:
    # A missing/unreadable schema must not create an empty database file.
    schema = schema_path.read_text(encoding="utf-8")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connection_scope(sqlite3.connect(DB_PATH)) as conn:
        _validate_existing_timestamp_column(conn)
        # executescript otherwise commits each DDL statement independently.
        # Keep additive setup atomic without attempting to migrate old tables.
        conn.executescript("BEGIN IMMEDIATE;\n" + schema + "\nCOMMIT;")


@contextmanager
def get_conn(*, timeout: float = 5.0) -> Generator[sqlite3.Connection, None, None]:
    with _connection_scope(sqlite3.connect(DB_PATH, timeout=timeout)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA secure_delete = ON")
        yield conn
        conn.commit()
