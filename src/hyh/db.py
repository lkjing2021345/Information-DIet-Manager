from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

SQLITE_TIMEOUT_SECONDS = float(os.getenv("IDM_SQLITE_TIMEOUT_SECONDS", "30"))
SQLITE_BUSY_TIMEOUT_MS = int(os.getenv("IDM_SQLITE_BUSY_TIMEOUT_MS", "30000"))
SQLITE_MAX_RETRIES = int(os.getenv("IDM_SQLITE_MAX_RETRIES", "5"))
SQLITE_RETRY_BASE_DELAY_SECONDS = float(
    os.getenv("IDM_SQLITE_RETRY_BASE_DELAY_SECONDS", "0.05")
)


class RetryConnection(sqlite3.Connection):
    def _run_with_retry(self, operation_name: str, func, *args, **kwargs):
        last_error: sqlite3.OperationalError | None = None
        for attempt in range(SQLITE_MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as exc:
                message = str(exc).lower()
                if "database is locked" not in message and "database table is locked" not in message:
                    raise
                last_error = exc
                if attempt >= SQLITE_MAX_RETRIES:
                    raise
                delay = SQLITE_RETRY_BASE_DELAY_SECONDS * (2**attempt)
                time.sleep(delay)
        if last_error is not None:
            raise last_error
        raise sqlite3.OperationalError(f"{operation_name} failed unexpectedly")

    def execute(self, sql: str, parameters=(), /):
        return self._run_with_retry(
            "execute", super().execute, sql, parameters
        )

    def executemany(self, sql: str, seq_of_parameters, /):
        return self._run_with_retry(
            "executemany", super().executemany, sql, seq_of_parameters
        )

    def executescript(self, sql_script: str, /):
        return self._run_with_retry(
            "executescript", super().executescript, sql_script
        )

    def commit(self):
        return self._run_with_retry("commit", super().commit)


def _default_db_path() -> Path:
    base = Path(__file__).resolve().parent
    data_dir = base / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "idm.sqlite3"


DB_PATH = Path(os.getenv("IDM_DB_PATH", _default_db_path()))


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(
        DB_PATH,
        timeout=SQLITE_TIMEOUT_SECONDS,
        factory=RetryConnection,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def init_db(schema_path: Path) -> None:
    conn = _connect()
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()
    finally:
        conn.close()


@contextmanager
def get_conn() -> Generator[sqlite3.Connection, None, None]:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
