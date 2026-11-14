"""Bounded export preparation, followed by download without a database handle."""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time

from fastapi import HTTPException
from starlette.responses import StreamingResponse

from .db import get_conn
from .owned_work import run_owned_sync
from .security import protect_credential_file

MAX_EXPORT_BYTES = 64 * 1024 * 1024
MAX_ROW_BYTES = 128 * 1024
MAX_DEDUP_BYTES = 32 * 1024 * 1024
PREPARE_SECONDS = 15
DOWNLOAD_SECONDS = 30
CHUNK_BYTES = 64 * 1024
_clock = time.monotonic
FIELDS = ("id", "url", "title", "text", "ts", "source", "lang", "channel",
          "author", "tags", "meta", "created_at")


def _close_file(stream):
    try:
        stream.close()
    except OSError:
        # Buffered close can report the same disk error as a prior write.
        # Preserve the original HTTP error/cancellation, never log the path.
        logging.getLogger("uvicorn.error").error("Export temporary file close failed")


class PreparationBudget:
    def __init__(self):
        self.deadline = _clock() + PREPARE_SECONDS
        self.seen = set()
        self.dedup_bytes = 0

    def check_time(self):
        if _clock() >= self.deadline:
            raise HTTPException(503, "Export preparation timed out; select a smaller window",
                                headers={"Retry-After": "5"})

    def duplicate(self, key):
        if key in self.seen:
            return True
        # Include the Python string and a conservative per-entry set allowance.
        # Keep exact normalized strings: digest collisions cannot merge records.
        cost = sys.getsizeof(key) + 128
        if self.dedup_bytes + cost > MAX_DEDUP_BYTES:
            raise HTTPException(413, "Export deduplication budget exceeded; select a smaller window")
        self.seen.add(key)
        self.dedup_bytes += cost
        return False


def _temporary_file():
    stream = tempfile.TemporaryFile(mode="w+b", prefix="idm-export-")
    try:
        if os.name == "nt":
            # Harden the empty file before any page text is written. POSIX
            # TemporaryFile already uses an unlinked, owner-only file.
            protect_credential_file(Path(stream.name))
        return stream
    except BaseException:
        _close_file(stream)
        raise


def _rows(conn, budget, from_ts, to_ts, limit_rows):
    where = ["1=1"]
    params = []
    if from_ts is not None:
        where.append("ts >= ?")
        params.append(from_ts)
    if to_ts is not None:
        where.append("ts <= ?")
        params.append(to_ts)
    params.append(limit_rows)
    # Inspect sizes in SQLite before materializing possibly oversized legacy
    # text in Python. The ts index also supplies the stable ts/id ordering.
    size_sql = " + ".join(f"COALESCE(length(CAST({field} AS BLOB)), 0)" for field in FIELDS)
    cursor = conn.execute(
        f"SELECT id, ({size_sql}) AS byte_size FROM items WHERE {' AND '.join(where)} "
        "ORDER BY ts ASC, id ASC LIMIT ?", params)
    try:
        for item_id, byte_size in cursor:
            budget.check_time()
            if byte_size > MAX_ROW_BYTES:
                raise HTTPException(413, "Stored record exceeds export row budget")
            row = conn.execute(f"SELECT {', '.join(FIELDS)} FROM items WHERE id = ?", (item_id,)).fetchone()
            yield dict(row)
    finally:
        cursor.close()


class _Writer:
    def __init__(self, stream, budget):
        self.stream, self.budget, self.size = stream, budget, 0

    def write(self, text):
        self.budget.check_time()
        encoded = text.encode("utf-8")
        if self.size + len(encoded) > MAX_EXPORT_BYTES:
            raise HTTPException(413, "Export exceeds 64 MiB; select a smaller window")
        self.stream.write(encoded)
        self.size += len(encoded)


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _encode(stream, budget, rows, fmt, columns, metadata):
    writer = _Writer(stream, budget)
    count = 0
    csv_buffer = io.StringIO(newline="")
    csv_writer = csv.DictWriter(csv_buffer, fieldnames=columns)

    def csv_write(row=None):
        csv_buffer.seek(0)
        csv_buffer.truncate(0)
        if row is None:
            csv_writer.writeheader()
        else:
            csv_writer.writerow({key: _json(value) if isinstance(value, (dict, list)) else value
                                 for key, value in row.items()})
        writer.write(csv_buffer.getvalue())

    if fmt == "json":
        writer.write('[' if metadata is None else '{"items":[')
    elif fmt == "csv":
        csv_write()
    for row in rows:
        budget.check_time()
        if fmt == "csv":
            csv_write(row)
        else:
            if count:
                writer.write("," if fmt == "json" else "\n")
            writer.write(_json(row))
        count += 1
    if fmt == "json":
        writer.write("]")
        if metadata is not None:
            writer.write(',"count":' + str(count))
            for key, value in metadata.items():
                writer.write("," + _json(key) + ":" + _json(value))
            writer.write("}")
    stream.flush()
    budget.check_time()
    stream.seek(0)
    return writer.size


class PreparedExportResponse(StreamingResponse):
    def __init__(self, stream, size, *, fmt, filename):
        self.stream = stream
        media_type = {"json": "application/json", "jsonl": "application/x-ndjson",
                      "csv": "text/csv; charset=utf-8"}[fmt]
        headers = {"Content-Length": str(size)}
        if fmt != "json":
            headers["Content-Disposition"] = f'attachment; filename="{filename}.{fmt}"'
        super().__init__(self._chunks(), media_type=media_type, headers=headers)

    def _chunks(self):
        while chunk := self.stream.read(CHUNK_BYTES):
            yield chunk

    async def __call__(self, scope, receive, send):
        # This response can only be constructed after successful preparation
        # and DB close. The middleware may now release the data-operation gate.
        scope["idm_export_ready"] = True
        try:
            async with asyncio.timeout(DOWNLOAD_SECONDS):
                await super().__call__(scope, receive, send)
        finally:
            # BackgroundTask is insufficient when send raises or is cancelled.
            self.close()

    def close(self):
        _close_file(self.stream)


def prepare_export(*, transform, from_ts, to_ts, limit_rows, fmt, columns, metadata, filename):
    """Complete a bounded artifact before sending 200; never emit partial success."""
    stream = None
    budget = PreparationBudget()
    try:
        stream = _temporary_file()
        budget.check_time()
        with get_conn() as conn:
            conn.execute("BEGIN")
            conn.set_progress_handler(lambda: int(_clock() >= budget.deadline), 1000)
            rows = _rows(conn, budget, from_ts, to_ts, limit_rows)
            try:
                size = _encode(stream, budget, transform(rows, budget), fmt, columns, metadata)
            finally:
                rows.close()
                conn.set_progress_handler(None, 0)
        response = PreparedExportResponse(stream, size, fmt=fmt, filename=filename)
        stream = None  # Ownership transfers to the response only after DB close.
        return response
    except sqlite3.OperationalError:
        budget.check_time()
        raise HTTPException(503, "Export database unavailable; retry later",
                            headers={"Retry-After": "5"}) from None
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        budget.check_time()
        raise HTTPException(507, "Export temporary storage unavailable") from None
    except (ValueError, TypeError, RecursionError):
        raise HTTPException(409, "Stored records cannot be encoded for export") from None
    finally:
        if stream is not None:
            _close_file(stream)


async def prepare_export_response(**kwargs):
    return await run_owned_sync(prepare_export, on_cancel=lambda response: response.close(), **kwargs)
