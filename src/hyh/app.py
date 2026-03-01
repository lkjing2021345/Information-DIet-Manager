from __future__ import annotations

import csv
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from db import get_conn, init_db
from models import IngestAck, IngestItem
from utils import normalize_text, normalize_url, sha256_hex

from contextlib import asynccontextmanager


def _now_ms() -> int:
    return int(time.time() * 1000)


def _schema_path() -> Path:
    return Path(__file__).resolve().parent / "schema.sql"


def _as_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _parse_tags(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        parts = [p.strip() for p in stripped.split("|") if p.strip()]
        return parts or None
    return None


def _parse_meta(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _coerce_int(value: Any, field: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"Invalid {field}: {value!r}")


def _prepare_item(raw: Dict[str, Any]) -> IngestItem:
    data = {str(k).strip(): v for k, v in raw.items() if k is not None}
    if "tags" in data:
        data["tags"] = _parse_tags(data.get("tags"))
    if "meta" in data:
        data["meta"] = _parse_meta(data.get("meta"))
    if "ts" in data:
        data["ts"] = _coerce_int(data.get("ts"), "ts")
    return IngestItem(**data)


def _row_from_item(item: IngestItem) -> Dict[str, Any]:
    url_norm = normalize_url(str(item.url))
    content_norm = normalize_text(item.title, item.text)
    return {
        "url": str(item.url),
        "title": item.title,
        "text": item.text,
        "ts": item.ts,
        "source": item.source,
        "lang": item.lang,
        "channel": item.channel,
        "author": item.author,
        "tags": _as_json(item.tags),
        "meta": _as_json(item.meta),
        "url_hash": sha256_hex(url_norm),
        "content_hash": sha256_hex(content_norm),
        "created_at": _now_ms(),
    }


def insert_items(items: Iterable[IngestItem]) -> Tuple[int, int]:
    inserted = 0
    duplicates = 0
    sql = """
        INSERT OR IGNORE INTO items (
            url, title, text, ts, source, lang, channel, author, tags, meta,
            url_hash, content_hash, created_at
        ) VALUES (
            :url, :title, :text, :ts, :source, :lang, :channel, :author, :tags, :meta,
            :url_hash, :content_hash, :created_at
        )
    """
    with get_conn() as conn:
        for item in items:
            row = _row_from_item(item)
            cur = conn.execute(sql, row)
            if cur.rowcount == 1:
                inserted += 1
            else:
                duplicates += 1
    return inserted, duplicates


def _read_upload(file: UploadFile) -> str:
    raw = file.file.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Only UTF-8 files are supported.")


def _load_items_from_csv(text: str) -> List[Dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text))
    rows: List[Dict[str, Any]] = []
    for row in reader:
        cleaned = {k: (v if v != "" else None) for k, v in row.items()}
        rows.append(cleaned)
    return rows


def _load_items_from_json(text: str) -> List[Dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        payload = json.loads(stripped)
        if not isinstance(payload, list):
            raise HTTPException(status_code=400, detail="JSON array expected.")
        return payload
    if stripped.startswith("{"):
        payload = json.loads(stripped)
        if isinstance(payload, dict) and "items" in payload:
            items = payload["items"]
            if not isinstance(items, list):
                raise HTTPException(status_code=400, detail="`items` must be a list.")
            return items
        return [payload]
    raise HTTPException(status_code=400, detail="Unsupported JSON format.")


def _load_items_from_jsonl(text: str) -> List[Dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    items: List[Dict[str, Any]] = []
    for line in stripped.splitlines():
        if not line.strip():
            continue
        items.append(json.loads(line))
    return items

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 启动逻辑：和原来的 on_startup 完全一致
    init_db(_schema_path())
    yield  # 分割线：启动完成，应用开始运行
    # 关闭逻辑（如果需要的话，比如关闭数据库连接）
    # 示例：await db.close() （如果有异步数据库连接的话）

app = FastAPI(
    title="Information Diet Manager (MVP)",
    lifespan=lifespan  # 关键：把生命周期函数关联到 app
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^chrome-extension://.*$",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/collect", response_model=IngestAck)
def collect(item: IngestItem) -> IngestAck:
    inserted, duplicates = insert_items([item])
    return IngestAck(inserted=inserted, duplicates=duplicates, failed=0)


@app.post("/import", response_model=IngestAck)
def import_items(file: UploadFile = File(...)) -> IngestAck:
    filename = (file.filename or "").lower()
    text = _read_upload(file)
    failed = 0
    raw_items: List[Dict[str, Any]] = []
    try:
        if filename.endswith(".csv"):
            raw_items = _load_items_from_csv(text)
        elif filename.endswith(".jsonl"):
            raw_items = _load_items_from_jsonl(text)
        elif filename.endswith(".json"):
            raw_items = _load_items_from_json(text)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid file contents: {exc}") from exc
    items: List[IngestItem] = []
    for raw in raw_items:
        try:
            items.append(_prepare_item(raw))
        except Exception:
            failed += 1
    inserted, duplicates = insert_items(items)
    return IngestAck(inserted=inserted, duplicates=duplicates, failed=failed)


@app.get("/items")
def list_items(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
) -> Dict[str, Any]:
    offset = (page - 1) * page_size
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) AS cnt FROM items").fetchone()["cnt"]
        rows = conn.execute(
            "SELECT * FROM items ORDER BY id DESC LIMIT ? OFFSET ?",
            (page_size, offset),
        ).fetchall()
    items: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if item.get("tags"):
            try:
                item["tags"] = json.loads(item["tags"])
            except json.JSONDecodeError:
                item["tags"] = None
        if item.get("meta"):
            try:
                item["meta"] = json.loads(item["meta"])
            except json.JSONDecodeError:
                item["meta"] = None
        items.append(item)
    return {"page": page, "page_size": page_size, "total": total, "items": items}


@app.post("/analyze/run")
def run_analysis() -> Dict[str, Any]:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) AS cnt FROM items").fetchone()["cnt"]
        distinct_content = conn.execute(
            "SELECT COUNT(DISTINCT content_hash) AS cnt FROM items WHERE content_hash IS NOT NULL"
        ).fetchone()["cnt"]
        channel_rows = conn.execute(
            "SELECT channel, COUNT(*) AS cnt FROM items GROUP BY channel"
        ).fetchall()
    repeat_ratio = 0.0
    if total:
        repeat_ratio = max(0.0, min(1.0, 1 - (distinct_content / total)))
    channel_counts = {row["channel"] or "unknown": row["cnt"] for row in channel_rows}
    day = datetime.now(timezone.utc).date().isoformat()
    payload = {
        "day": day,
        "total_count": total,
        "channel_counts": channel_counts,
        "repeat_ratio": repeat_ratio,
        "negative_ratio": None,
        "avg_sentiment": None,
    }
    now_ms = _now_ms()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO stats_daily (
                day, total_count, channel_counts, repeat_ratio, negative_ratio,
                avg_sentiment, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(day) DO UPDATE SET
                total_count=excluded.total_count,
                channel_counts=excluded.channel_counts,
                repeat_ratio=excluded.repeat_ratio,
                negative_ratio=excluded.negative_ratio,
                avg_sentiment=excluded.avg_sentiment,
                updated_at=excluded.updated_at
            """,
            (
                day,
                total,
                _as_json(channel_counts),
                repeat_ratio,
                None,
                None,
                now_ms,
                now_ms,
            ),
        )
    return payload


@app.get("/dashboard/summary")
def dashboard_summary() -> Dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM stats_daily ORDER BY day DESC LIMIT 1"
        ).fetchone()
    if row:
        channel_counts = None
        if row["channel_counts"]:
            try:
                channel_counts = json.loads(row["channel_counts"])
            except json.JSONDecodeError:
                channel_counts = None
        return {
            "day": row["day"],
            "total_count": row["total_count"],
            "channel_counts": channel_counts,
            "repeat_ratio": row["repeat_ratio"],
            "negative_ratio": row["negative_ratio"],
            "avg_sentiment": row["avg_sentiment"],
            "generated_at": row["updated_at"],
        }
    return run_analysis()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
