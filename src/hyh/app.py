from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import struct
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from db import get_conn, init_db
from models import IngestAck, IngestItem
from utils import normalize_text, normalize_url, sha256_hex

from contextlib import asynccontextmanager

MAX_TEXT_LEN = 1000
EMBEDDING_DIM = 128
EMBEDDING_MODEL = "hash-chargram-v1"
_WORD_RE = re.compile(r"\w+", re.UNICODE)
JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_COMPLETED = "completed"
JOB_FAILED = "failed"


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


def _clean_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _truncate_text(value: Optional[str], max_len: int = MAX_TEXT_LEN) -> Optional[str]:
    if value is None:
        return None
    return value[:max_len] if len(value) > max_len else value


def _prepare_item(raw: Dict[str, Any]) -> IngestItem:
    data = {str(k).strip(): v for k, v in raw.items() if k is not None}
    for key in ("url", "title", "text", "source", "lang", "channel", "author"):
        if key in data:
            data[key] = _clean_optional_str(data.get(key))
    if "tags" in data:
        data["tags"] = _parse_tags(data.get("tags"))
    if "meta" in data:
        data["meta"] = _parse_meta(data.get("meta"))
    title = data.get("title")
    text = data.get("text")
    if text is None and title:
        text = title
    data["text"] = _truncate_text(text)
    if "ts" in data:
        data["ts"] = _coerce_int(data.get("ts"), "ts")
    return IngestItem(**data)


def _row_from_item(item: IngestItem) -> Dict[str, Any]:
    title = (item.title or "").strip()
    text = _clean_optional_str(item.text)
    if text is None:
        text = title
    text = _truncate_text(text)
    url_norm = normalize_url(str(item.url))
    content_norm = normalize_text(title, text)
    return {
        "url": str(item.url),
        "title": title,
        "text": text,
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


def _embedding_input_text(title: Optional[str], text: Optional[str]) -> str:
    base = _clean_optional_str(text) or _clean_optional_str(title) or ""
    normalized = normalize_text("", base)
    return _truncate_text(normalized, max_len=MAX_TEXT_LEN) or ""


def _vector_to_blob(vector: List[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _hashed_chargram_embedding(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    vec = [0.0] * dim
    tokens = _WORD_RE.findall(text)
    if not tokens:
        return vec
    for tok in tokens:
        grams: List[str]
        if len(tok) < 3:
            grams = [tok]
        else:
            grams = [tok[i : i + 3] for i in range(len(tok) - 2)]
        for gram in grams:
            h = int.from_bytes(
                hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(),
                "little",
            )
            idx = h % dim
            sign = 1.0 if ((h >> 1) & 1) == 0 else -1.0
            vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _upsert_embedding(conn: Any, item_id: int, title: Optional[str], text: Optional[str]) -> None:
    input_text = _embedding_input_text(title, text)
    vector = _hashed_chargram_embedding(input_text, dim=EMBEDDING_DIM)
    conn.execute(
        """
        INSERT INTO embeddings (item_id, vector, vector_path, model, dim, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(item_id) DO UPDATE SET
            vector=excluded.vector,
            vector_path=excluded.vector_path,
            model=excluded.model,
            dim=excluded.dim,
            created_at=excluded.created_at
        """,
        (
            item_id,
            _vector_to_blob(vector),
            None,
            EMBEDDING_MODEL,
            EMBEDDING_DIM,
            _now_ms(),
        ),
    )


def _backfill_missing_embeddings(conn: Any, limit: int = 2000) -> int:
    rows = conn.execute(
        """
        SELECT i.id, i.title, i.text
        FROM items AS i
        LEFT JOIN embeddings AS e ON e.item_id = i.id
        WHERE e.item_id IS NULL
        ORDER BY i.id ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    count = 0
    for row in rows:
        _upsert_embedding(conn, int(row["id"]), row["title"], row["text"])
        count += 1
    return count


def _payload_from_stats_row(row: Any) -> Dict[str, Any]:
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


def _stable_hash_payload(data: Dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_hex(payload)


def _lsj_algorithms_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "lsj" / "src" / "algorithms"


def _ensure_lsj_import_path() -> None:
    alg_dir = _lsj_algorithms_dir()
    path = str(alg_dir)
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_items_for_analysis(
    conn: Any,
    from_ts: Optional[int],
    to_ts: Optional[int],
    limit_rows: int,
) -> List[Dict[str, Any]]:
    where_parts = ["1=1"]
    params: List[Any] = []
    if from_ts is not None:
        where_parts.append("ts >= ?")
        params.append(from_ts)
    if to_ts is not None:
        where_parts.append("ts <= ?")
        params.append(to_ts)
    where_sql = " AND ".join(where_parts)
    sql = f"""
        SELECT id, url, title, text, ts, channel, created_at
        FROM items
        WHERE {where_sql}
        ORDER BY ts ASC, id ASC
        LIMIT ?
    """
    params.append(limit_rows)
    rows = conn.execute(sql, tuple(params)).fetchall()
    return [dict(r) for r in rows]


def _insert_analysis_job(
    conn: Any,
    *,
    status: str,
    input_hash: str,
    day: str,
    from_ts: Optional[int],
    to_ts: Optional[int],
    limit_rows: int,
    item_max_created_at: int,
    input_count: int,
    cache_hit: bool,
) -> int:
    now_ms = _now_ms()
    cur = conn.execute(
        """
        INSERT INTO analysis_jobs (
            status, input_hash, day, from_ts, to_ts, limit_rows, item_max_created_at,
            input_count, cache_hit, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            status,
            input_hash,
            day,
            from_ts,
            to_ts,
            limit_rows,
            item_max_created_at,
            input_count,
            1 if cache_hit else 0,
            now_ms,
            now_ms,
        ),
    )
    return int(cur.lastrowid)


def _update_analysis_job(
    conn: Any,
    job_id: int,
    *,
    status: str,
    error: Optional[str] = None,
    result_payload: Optional[Dict[str, Any]] = None,
    metrics_json: Optional[Dict[str, Any]] = None,
    duration_ms: Optional[int] = None,
    started_at: Optional[int] = None,
    finished_at: Optional[int] = None,
) -> None:
    now_ms = _now_ms()
    conn.execute(
        """
        UPDATE analysis_jobs
        SET status = ?,
            error = ?,
            result_payload = COALESCE(?, result_payload),
            metrics_json = COALESCE(?, metrics_json),
            duration_ms = COALESCE(?, duration_ms),
            started_at = COALESCE(?, started_at),
            finished_at = COALESCE(?, finished_at),
            updated_at = ?
        WHERE id = ?
        """,
        (
            status,
            error,
            _as_json(result_payload) if result_payload is not None else None,
            _as_json(metrics_json) if metrics_json is not None else None,
            duration_ms,
            started_at,
            finished_at,
            now_ms,
            job_id,
        ),
    )


def _run_lsj_pipeline(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "input_count": 0,
            "category_counts": {},
            "sentiment_counts": {},
            "negative_ratio": 0.0,
            "avg_sentiment": 0.0,
            "repeat_ratio": 0.0,
            "quick_evaluation": None,
            "full_report": None,
        }

    try:
        _ensure_lsj_import_path()
        import pandas as pd  # type: ignore
        from classifier import ContentClassifier  # type: ignore
        from evaluator import InformationQualityEvaluator  # type: ignore
        from sentiment import SentimentAnalyzer  # type: ignore
        from similarity import SimilarityAnalyzer  # type: ignore

        df = pd.DataFrame(rows)
        df["title"] = df["title"].fillna("").astype(str)
        df["text"] = df["text"].fillna(df["title"]).astype(str)
        df["analysis_text"] = df["text"].where(df["text"].str.strip() != "", df["title"])
        df["url"] = df["url"].fillna("").astype(str)
        df["channel"] = df["channel"].fillna("unknown").astype(str)

        classifier = ContentClassifier()
        sentiment = SentimentAnalyzer()
        similarity = SimilarityAnalyzer()
        evaluator = InformationQualityEvaluator(
            sentiment_analyzer=sentiment,
            content_classifier=classifier,
            similarity_analyzer=similarity,
        )

        df1 = classifier.batch_predict(df[["title", "url", "analysis_text", "channel", "ts"]].copy())
        df2 = sentiment.batch_predict(df1, text_column="analysis_text", include_emotions=False, batch_size=500)
        df3 = similarity.batch_calculate_similarity(df2, text_column="analysis_text")
        if "similarity" not in df3.columns and "similarity_to_previous" in df3.columns:
            df3["similarity"] = df3["similarity_to_previous"]
        quick = evaluator.quick_evaluate(df3)
        report = evaluator.evaluate(df3, detailed=False).to_dict()

        sentiment_norm = df3["sentiment"].fillna("").astype(str).str.lower()
        category_norm = df3["category"].fillna("other").astype(str)
        polarity_num = pd.to_numeric(df3["polarity"], errors="coerce").fillna(0.0)
        similarity_num = pd.to_numeric(df3["similarity"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

        negative_ratio = float((sentiment_norm == "negative").mean())
        avg_sentiment = float(polarity_num.mean())
        repeat_ratio = float((similarity_num >= 0.85).mean())

        category_counts = {str(k): int(v) for k, v in category_norm.value_counts().to_dict().items()}
        sentiment_counts = {str(k): int(v) for k, v in df3["sentiment"].value_counts().to_dict().items()}

        return {
            "input_count": int(len(df3)),
            "category_counts": category_counts,
            "sentiment_counts": sentiment_counts,
            "negative_ratio": negative_ratio,
            "avg_sentiment": avg_sentiment,
            "repeat_ratio": repeat_ratio,
            "quick_evaluation": quick,
            "full_report": report,
            "pipeline_warning": None,
        }
    except Exception as exc:
        normalized = [
            normalize_text(str(r.get("title") or ""), str(r.get("text") or r.get("title") or ""))
            for r in rows
        ]
        total = len(normalized)
        distinct = len(set(normalized))
        repeat_ratio = float(0.0 if total == 0 else max(0.0, min(1.0, 1 - (distinct / total))))
        return {
            "input_count": total,
            "category_counts": {"unknown": total},
            "sentiment_counts": {"Neutral": total},
            "negative_ratio": 0.0,
            "avg_sentiment": 0.0,
            "repeat_ratio": repeat_ratio,
            "quick_evaluation": None,
            "full_report": None,
            "pipeline_warning": f"lsj pipeline unavailable: {exc}",
        }


def _get_job_row(conn: Any, job_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM analysis_jobs WHERE id = ? LIMIT 1",
        (job_id,),
    ).fetchone()
    if row is None:
        return None
    data = dict(row)
    for key in ("result_payload", "metrics_json"):
        if data.get(key):
            try:
                data[key] = json.loads(data[key])
            except json.JSONDecodeError:
                data[key] = None
    data["cache_hit"] = bool(data.get("cache_hit", 0))
    return data


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
                _upsert_embedding(conn, int(cur.lastrowid), row.get("title"), row.get("text"))
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
        item: Dict[str, Any] = dict(row)
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
def run_analysis(
    force: bool = Query(False),
    backfill_limit: int = Query(2000, ge=0, le=20000),
) -> Dict[str, Any]:
    day = datetime.now(timezone.utc).date().isoformat()
    with get_conn() as conn:
        item_state = conn.execute(
            "SELECT COALESCE(MAX(created_at), 0) AS max_created_at FROM items"
        ).fetchone()
        max_created_at = int(item_state["max_created_at"] or 0)
        existing = conn.execute(
            "SELECT * FROM stats_daily WHERE day = ? LIMIT 1",
            (day,),
        ).fetchone()
        if (
            not force
            and existing is not None
            and int(existing["updated_at"] or 0) >= max_created_at
        ):
            cached_payload = _payload_from_stats_row(existing)
            cached_payload["cached"] = True
            cached_payload["embeddings_backfilled"] = 0
            conn.execute(
                """
                INSERT INTO analysis_runs (
                    day, total_count, channel_counts, repeat_ratio, negative_ratio, avg_sentiment,
                    payload, cached, item_max_created_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    day,
                    cached_payload["total_count"],
                    _as_json(cached_payload.get("channel_counts")),
                    cached_payload.get("repeat_ratio"),
                    cached_payload.get("negative_ratio"),
                    cached_payload.get("avg_sentiment"),
                    _as_json(cached_payload),
                    1,
                    max_created_at,
                    _now_ms(),
                ),
            )
            return cached_payload

        backfilled = 0
        if backfill_limit > 0:
            backfilled = _backfill_missing_embeddings(conn, limit=backfill_limit)

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
        now_ms = _now_ms()
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
        payload = {
            "day": day,
            "total_count": total,
            "channel_counts": channel_counts,
            "repeat_ratio": repeat_ratio,
            "negative_ratio": None,
            "avg_sentiment": None,
            "generated_at": now_ms,
            "cached": False,
            "embeddings_backfilled": backfilled,
        }
        conn.execute(
            """
            INSERT INTO analysis_runs (
                day, total_count, channel_counts, repeat_ratio, negative_ratio, avg_sentiment,
                payload, cached, item_max_created_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                day,
                total,
                _as_json(channel_counts),
                repeat_ratio,
                None,
                None,
                _as_json(payload),
                0,
                max_created_at,
                now_ms,
            ),
        )
        return payload


@app.get("/analyze/history")
def analyze_history(limit: int = Query(20, ge=1, le=200)) -> Dict[str, Any]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, day, total_count, channel_counts, repeat_ratio, negative_ratio, avg_sentiment,
                   cached, item_max_created_at, created_at
            FROM analysis_runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    runs: List[Dict[str, Any]] = []
    for row in rows:
        run = dict(row)
        if run.get("channel_counts"):
            try:
                run["channel_counts"] = json.loads(run["channel_counts"])
            except json.JSONDecodeError:
                run["channel_counts"] = None
        run["cached"] = bool(run.get("cached", 0))
        runs.append(run)
    return {"total": len(runs), "runs": runs}


@app.post("/analyze/run_full")
def run_full_analysis(
    force: bool = Query(False),
    from_ts: Optional[int] = Query(None),
    to_ts: Optional[int] = Query(None),
    limit_rows: int = Query(5000, ge=1, le=50000),
) -> Dict[str, Any]:
    if from_ts is not None and to_ts is not None and from_ts > to_ts:
        raise HTTPException(status_code=400, detail="from_ts cannot be greater than to_ts")
    day = datetime.now(timezone.utc).date().isoformat()
    with get_conn() as conn:
        item_state = conn.execute(
            "SELECT COALESCE(MAX(created_at), 0) AS max_created_at FROM items"
        ).fetchone()
        max_created_at = int(item_state["max_created_at"] or 0)
        rows = _load_items_for_analysis(conn, from_ts=from_ts, to_ts=to_ts, limit_rows=limit_rows)
        input_count = len(rows)
        job_key = {
            "day": day,
            "from_ts": from_ts,
            "to_ts": to_ts,
            "limit_rows": limit_rows,
            "mode": "run_full",
        }
        input_hash = _stable_hash_payload(job_key)

        if not force:
            cached = conn.execute(
                """
                SELECT id, result_payload FROM analysis_jobs
                WHERE input_hash = ? AND item_max_created_at = ? AND status = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (input_hash, max_created_at, JOB_COMPLETED),
            ).fetchone()
            if cached:
                cached_payload: Dict[str, Any] = {}
                if cached["result_payload"]:
                    try:
                        cached_payload = json.loads(cached["result_payload"])
                    except json.JSONDecodeError:
                        cached_payload = {}
                job_id = _insert_analysis_job(
                    conn,
                    status=JOB_COMPLETED,
                    input_hash=input_hash,
                    day=day,
                    from_ts=from_ts,
                    to_ts=to_ts,
                    limit_rows=limit_rows,
                    item_max_created_at=max_created_at,
                    input_count=input_count,
                    cache_hit=True,
                )
                _update_analysis_job(
                    conn,
                    job_id,
                    status=JOB_COMPLETED,
                    result_payload=cached_payload,
                    metrics_json={
                        "cache_reuse_from_job_id": int(cached["id"]),
                        "input_count": input_count,
                        "cache_hit": True,
                    },
                    duration_ms=0,
                    started_at=_now_ms(),
                    finished_at=_now_ms(),
                )
                return {
                    "job_id": job_id,
                    "status": JOB_COMPLETED,
                    "cached": True,
                    "reused_from_job_id": int(cached["id"]),
                    "result": cached_payload,
                }
        #not cached的情况也自然下落到这里
        job_id = _insert_analysis_job(
            conn,
            status=JOB_QUEUED,
            input_hash=input_hash,
            day=day,
            from_ts=from_ts,
            to_ts=to_ts,
            limit_rows=limit_rows,
            item_max_created_at=max_created_at,
            input_count=input_count,
            cache_hit=False,
        )
        started_at = _now_ms()
        _update_analysis_job(
            conn,
            job_id,
            status=JOB_RUNNING,
            started_at=started_at,
        )

        try:
            pipeline = _run_lsj_pipeline(rows)
            now_ms = _now_ms()

            channel_counts: Dict[str, int] = {}
            for r in rows:
                key = _clean_optional_str(r.get("channel")) or "unknown"
                channel_counts[key] = channel_counts.get(key, 0) + 1

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
                    int(pipeline["input_count"]),
                    _as_json(channel_counts),
                    float(pipeline["repeat_ratio"]),
                    float(pipeline["negative_ratio"]),
                    float(pipeline["avg_sentiment"]),
                    now_ms,
                    now_ms,
                ),
            )

            payload = {
                "day": day,
                "total_count": int(pipeline["input_count"]),
                "channel_counts": channel_counts,
                "category_counts": pipeline["category_counts"],
                "sentiment_counts": pipeline["sentiment_counts"],
                "repeat_ratio": float(pipeline["repeat_ratio"]),
                "negative_ratio": float(pipeline["negative_ratio"]),
                "avg_sentiment": float(pipeline["avg_sentiment"]),
                "quick_evaluation": pipeline["quick_evaluation"],
                "full_report": pipeline["full_report"],
                "pipeline_warning": pipeline.get("pipeline_warning"),
                "generated_at": now_ms,
                "cached": False,
            }
            conn.execute(
                """
                INSERT INTO analysis_runs (
                    day, total_count, channel_counts, repeat_ratio, negative_ratio, avg_sentiment,
                    payload, cached, item_max_created_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    day,
                    int(pipeline["input_count"]),
                    _as_json(channel_counts),
                    float(pipeline["repeat_ratio"]),
                    float(pipeline["negative_ratio"]),
                    float(pipeline["avg_sentiment"]),
                    _as_json(payload),
                    0,
                    max_created_at,
                    now_ms,
                ),
            )
            duration_ms = _now_ms() - started_at
            _update_analysis_job(
                conn,
                job_id,
                status=JOB_COMPLETED,
                result_payload=payload,
                metrics_json={
                    "cache_hit": False,
                    "input_count": input_count,
                    "duration_ms": duration_ms,
                },
                duration_ms=duration_ms,
                finished_at=_now_ms(),
            )
            return {
                "job_id": job_id,
                "status": JOB_COMPLETED,
                "cached": False,
                "result": payload,
            }
        except Exception as exc:
            duration_ms = _now_ms() - started_at
            err_detail = f"{exc}\n{traceback.format_exc(limit=5)}"
            _update_analysis_job(
                conn,
                job_id,
                status=JOB_FAILED,
                error=err_detail[:4000],
                metrics_json={
                    "cache_hit": False,
                    "input_count": input_count,
                    "duration_ms": duration_ms,
                },
                duration_ms=duration_ms,
                finished_at=_now_ms(),
            )
            raise HTTPException(
                status_code=500,
                detail=f"run_full failed, job_id={job_id}, error={exc}",
            ) from exc


@app.get("/analyze/jobs/{job_id}")
def get_analyze_job(job_id: int) -> Dict[str, Any]:
    with get_conn() as conn:
        row = _get_job_row(conn, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    return row


@app.get("/analyze/result/{job_id}")
def get_analyze_result(job_id: int) -> Dict[str, Any]:
    with get_conn() as conn:
        row = _get_job_row(conn, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    if row.get("status") != JOB_COMPLETED:
        raise HTTPException(status_code=409, detail=f"job not completed: {row.get('status')}")
    return {
        "job_id": row["id"],
        "status": row["status"],
        "result": row.get("result_payload"),
        "metrics": row.get("metrics_json"),
        "cache_hit": row.get("cache_hit"),
    }


@app.get("/dashboard/summary")
def dashboard_summary() -> Dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM stats_daily ORDER BY day DESC LIMIT 1"
        ).fetchone()
    if row:
        return _payload_from_stats_row(row)
    return run_analysis()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
