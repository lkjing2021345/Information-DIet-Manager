from __future__ import annotations

import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


def _default_db_path() -> Path:
    base = Path(__file__).resolve().parent
    data_dir = base / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "idm.sqlite3"


DB_PATH = Path(os.getenv("IDM_DB_PATH", _default_db_path()))


def init_db(schema_path: Path) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()
    finally:
        conn.close()


@contextmanager
def get_conn() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
