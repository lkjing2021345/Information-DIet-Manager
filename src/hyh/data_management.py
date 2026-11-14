"""Portable page-record backups; credentials and derived analysis are excluded."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import closing
from io import BytesIO
from typing import Literal

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import export_io
from .db import get_conn
from .models import IngestItem
from .owned_work import run_owned_sync
from .utils import normalize_url, sha256_hex

MAX_BACKUP_RECORDS = 10000
MAX_BACKUP_BYTES = 20 * 1024 * 1024
MAX_BACKUP_ROW_BYTES = 128 * 1024
BACKUP_PREPARE_SECONDS = 15
_clock = time.monotonic
FIELDS = ("url", "title", "text", "ts", "source", "lang", "channel", "author", "tags", "meta")


class PageBackup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["idm-page-records"]
    version: Literal[1]
    exported_at: int = Field(strict=True, ge=0)
    items: list[IngestItem] = Field(max_length=MAX_BACKUP_RECORDS)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def canonical_items(items) -> bytes:
    return json.dumps(items, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def clear_derived(conn):
    for table in ("analysis_jobs", "analysis_runs", "stats_daily"):
        conn.execute(f"DELETE FROM {table}")  # Constants, never user input.


def clear_all(conn):
    clear_derived(conn)
    conn.execute("DELETE FROM embeddings")
    conn.execute("DELETE FROM items")


def require_confirmation(request: Request, value: str):
    if request.headers.get("X-IDM-Confirm") != value:
        raise HTTPException(400, "Explicit operation confirmation required")


class _BackupPreparationBudget:
    def __init__(self):
        self.deadline = _clock() + BACKUP_PREPARE_SECONDS

    def remaining(self):
        remaining = self.deadline - _clock()
        if remaining <= 0:
            raise HTTPException(503, "Backup preparation timed out; retry later",
                                headers={"Retry-After": "5"})
        return remaining

    def check_time(self):
        self.remaining()

    def set_busy_timeout(self, conn):
        # sqlite3's busy handler does not invoke the SQL progress callback.
        # Refresh the wait allowance before each query/row, never exceeding
        # the connection's usual five-second maximum or the remaining budget.
        milliseconds = min(5000, int(self.remaining() * 1000))
        conn.execute(f"PRAGMA busy_timeout = {milliseconds}").close()

    def execute(self, conn, sql, parameters=()):
        self.set_busy_timeout(conn)
        cursor = conn.execute(sql, parameters)
        try:
            self.check_time()
            return cursor
        except BaseException:
            cursor.close()
            raise


def _read_backup_items(conn, budget):
    with closing(budget.execute(conn, "SELECT COUNT(*) FROM items")) as cursor:
        count = cursor.fetchone()[0]
    budget.check_time()
    if count > MAX_BACKUP_RECORDS:
        raise HTTPException(413, "Backup supports at most 10000 page records")
    # SQLite computes UTF-8 byte sizes before Python receives legacy values.
    # Only the backup's fields count; unexported hashes are never loaded.
    size_sql = " + ".join(f"COALESCE(length(CAST({field} AS BLOB)), 0)" for field in FIELDS)
    with closing(budget.execute(conn, f"SELECT id, ({size_sql}) AS byte_size FROM items ORDER BY id")) as cursor:
        items, size = [], 2
        seen_url_hashes = set()
        while True:
            budget.set_busy_timeout(conn)
            sized = cursor.fetchone()
            budget.check_time()
            if sized is None:
                break
            item_id, byte_size = sized
            if byte_size > MAX_BACKUP_ROW_BYTES:
                raise HTTPException(413, "Stored record exceeds backup row budget")
            with closing(budget.execute(conn, f"SELECT {', '.join(FIELDS)} FROM items WHERE id = ?", (item_id,))) as values:
                row = values.fetchone()
            budget.check_time()
            record = {key: row[key] for key in FIELDS}
            for key in ("tags", "meta"):
                budget.check_time()
                record[key] = json.loads(record[key]) if record[key] is not None else None
                budget.check_time()
            record = IngestItem.model_validate(record).model_dump(mode="json")
            budget.check_time()
            # Restoration hashes the validated URL, not a possibly absent or
            # stale legacy hash. A successful backup must be restorable.
            url_hash = sha256_hex(normalize_url(record["url"]))
            budget.check_time()
            if url_hash in seen_url_hashes:
                raise HTTPException(409, "Stored records contain duplicate normalized page URLs; backup cannot be restored")
            seen_url_hashes.add(url_hash)
            size += len(canonical_items(record)) + int(bool(items))
            budget.check_time()
            if size > MAX_BACKUP_BYTES:
                raise HTTPException(413, "Backup exceeds 20 MiB")
            items.append(record)
    return items


def _prepare_backup_response():
    budget = _BackupPreparationBudget()
    budget.check_time()
    try:
        with get_conn(timeout=min(5.0, budget.remaining())) as conn:
            conn.set_progress_handler(lambda: int(_clock() >= budget.deadline), 1000)
            try:
                with closing(budget.execute(conn, "BEGIN")):
                    pass
                items = _read_backup_items(conn, budget)
            finally:
                conn.set_progress_handler(None, 0)
        budget.check_time()
        digest = hashlib.sha256(canonical_items(items)).hexdigest()
        budget.check_time()
        payload = {"format": "idm-page-records", "version": 1, "exported_at": int(time.time() * 1000), "items": items, "sha256": digest}
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        budget.check_time()
    except sqlite3.OperationalError:
        budget.check_time()
        raise HTTPException(503, "Backup database unavailable; retry later",
                            headers={"Retry-After": "5"}) from None
    except (ValueError, TypeError, ValidationError, RecursionError):
        budget.check_time()
        raise HTTPException(409, "Stored legacy records do not satisfy the current backup contract") from None
    if len(body) > MAX_BACKUP_BYTES:
        raise HTTPException(413, "Backup exceeds 20 MiB")
    # Preserve the existing bounded backup encoding. Its completed in-memory
    # artifact uses the same chunking, cancellation and lifetime as file exports.
    stream = BytesIO(body)
    try:
        budget.check_time()
        response = export_io.PreparedExportResponse(stream, len(body), fmt="json", filename="idm-pages-backup")
        response.headers["Content-Disposition"] = 'attachment; filename="idm-pages-backup.json"'
        return response
    except BaseException:
        stream.close()
        raise


def _delete_item_sync(item_id: int):
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if not 1 <= item_id <= 2**63 - 1 or not conn.execute("SELECT id FROM items WHERE id = ?", (item_id,)).fetchone():
            raise HTTPException(404, "Record not found")
        conn.execute("DELETE FROM embeddings WHERE item_id = ?", (item_id,))
        conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
        clear_derived(conn)
    return {"deleted": 1, "analysis_cleared": True}


def _delete_all_sync():
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        clear_all(conn)
    return {"deleted": count, "analysis_cleared": True}


def install_data_routes(app, insert_items):
    @app.get("/data/backup")
    async def backup():
        return await run_owned_sync(_prepare_backup_response, on_cancel=lambda response: response.close())

    @app.delete("/items/{item_id}")
    async def delete_item(item_id: int, request: Request):
        require_confirmation(request, "delete-record")
        return await run_owned_sync(_delete_item_sync, item_id=item_id)

    @app.delete("/data")
    async def delete_all(request: Request):
        require_confirmation(request, "delete-all")
        return await run_owned_sync(_delete_all_sync)

    @app.post("/data/restore")
    async def restore(request: Request):
        require_confirmation(request, "replace-records")
        try:
            raw = await request.json()
            if not isinstance(raw, dict) or type(raw.get("version")) is not int:
                raise ValueError()
            backup = PageBackup.model_validate(raw)
            digest = hashlib.sha256(canonical_items(raw["items"])).hexdigest()
            if digest != backup.sha256:
                raise ValueError()
        except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
            raise HTTPException(422, "Invalid, unsupported or damaged page-record backup") from None
        def replace():
            with get_conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                clear_all(conn)
                inserted, duplicates = insert_items(backup.items, connection=conn, preserve_text=True)
                if duplicates:
                    raise HTTPException(422, "Backup contains duplicate normalized page URLs; nothing changed")
            return {"restored": inserted, "analysis_cleared": True}

        return await run_owned_sync(replace)
