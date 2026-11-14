"""Bounded, signed record cursors. A cursor never carries an access credential."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time

from fastapi import HTTPException

# A restart intentionally invalidates outstanding cursors. No server-side cursor
# registry or long-lived SQLite transaction is held between HTTP requests.
_SIGNING_KEY = secrets.token_bytes(32)


def _expired() -> HTTPException:
    return HTTPException(409, detail={
        "code": "items_snapshot_expired",
        "message": "Records changed or the server restarted; reload the first page.",
    })


def _encode(state: dict) -> str:
    payload = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("ascii")
    signature = hmac.digest(_SIGNING_KEY, payload, hashlib.sha256)
    return base64.urlsafe_b64encode(signature + payload).decode("ascii").rstrip("=")


def _decode(cursor: str) -> dict:
    try:
        raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
    except (ValueError, UnicodeError):
        raise HTTPException(400, detail="Invalid record cursor.") from None
    if len(raw) < 33:
        raise HTTPException(400, detail="Invalid record cursor.")
    signature, payload = raw[:32], raw[32:]
    if not hmac.compare_digest(signature, hmac.digest(_SIGNING_KEY, payload, hashlib.sha256)):
        raise _expired()
    try:
        state = json.loads(payload)
        numeric = ("upper", "before", "revision", "total", "page", "page_size", "snapshot_at")
        if not isinstance(state, dict) or set(state) != {*numeric, "database"}:
            raise ValueError
        if any(type(state[key]) is not int or not 0 <= state[key] <= 2**63 - 1 for key in numeric):
            raise ValueError
        if not (1 <= state["page_size"] <= 200 and state["page"] >= 1
                and 0 < state["before"] <= state["upper"]):
            raise ValueError
        if not isinstance(state["database"], str) or len(state["database"]) != 32:
            raise ValueError
        return state
    except (ValueError, TypeError, KeyError):
        raise HTTPException(400, detail="Invalid record cursor.") from None


def read_cursor_page(conn, *, page_size: int, cursor: str | None) -> tuple[list, dict]:
    """Call within one read transaction, including the revision/count/rows reads.

    Inserts after the first page are outside its upper ID boundary. Deletion or
    update invalidates the snapshot through transactional database triggers.
    """
    current = conn.execute("SELECT database_id, revision FROM items_revision WHERE singleton = 1").fetchone()
    if cursor is None:
        summary = conn.execute("SELECT COUNT(*) AS total, COALESCE(MAX(id), 0) AS upper_id FROM items").fetchone()
        state = {"upper": summary["upper_id"], "before": summary["upper_id"],
                 "revision": current["revision"], "database": current["database_id"],
                 "total": summary["total"], "page": 1, "page_size": page_size,
                 "snapshot_at": time.time_ns() // 1_000_000}
        rows = conn.execute("SELECT * FROM items WHERE id <= ? ORDER BY id DESC LIMIT ?",
                            (state["upper"], page_size + 1)).fetchall()
    else:
        state = _decode(cursor)
        if state["revision"] != current["revision"] or state["database"] != current["database_id"]:
            raise _expired()
        if state["page_size"] != page_size:
            raise HTTPException(400, detail="Keep page_size unchanged while using a cursor.")
        rows = conn.execute("SELECT * FROM items WHERE id <= ? AND id < ? ORDER BY id DESC LIMIT ?",
                            (state["upper"], state["before"], page_size + 1)).fetchall()
    has_more = len(rows) > page_size
    rows = rows[:page_size]
    next_cursor = _encode({**state, "before": rows[-1]["id"], "page": state["page"] + 1}) if has_more else None
    return rows, {"pagination": "cursor", "page": state["page"], "page_size": page_size,
                  "total": state["total"], "snapshot_at": state["snapshot_at"],
                  "has_more": has_more, "next_cursor": next_cursor}
