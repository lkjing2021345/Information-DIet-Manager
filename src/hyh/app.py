from __future__ import annotations

import csv
import asyncio
import hashlib
import io
import ipaddress
import json
import math
import re
import struct
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from .db import get_conn, init_db
from .models import IngestAck, IngestItem, MAX_INGEST_TS
from .utils import normalize_text, normalize_url, sha256_hex
from . import db
from .security import LocalAccessMiddleware, LocalSecurity, process_ownership
from .data_management import install_data_routes
from .item_pagination import read_cursor_page
from .visualization_metrics import finite_metric
from . import export_io
from .owned_work import WorkOwner, run_owned_sync
from .analysis_json import InvalidStoredAnalysis, load_analysis_json, validate_analysis_value
from .full_analysis_contract import valid_full_analysis_cache
from .global_statistics_contract import valid_global_statistics_cache

from contextlib import asynccontextmanager, nullcontext

from urllib.parse import urlparse

MAX_TEXT_LEN = 1000
EMBEDDING_DIM = 128
EMBEDDING_MODEL = "hash-chargram-v1"
_WORD_RE = re.compile(r"\w+", re.UNICODE)
JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_COMPLETED = "completed"
JOB_FAILED = "failed"
MAX_ANALYSIS_JOB_HISTORY = 500
MAX_ANALYSIS_RUN_HISTORY = 365
EXPERIMENTAL_ANALYSIS_NOTICE = (
    "Experimental legacy analysis only; not a health, psychological, personality, or diagnostic assessment."
)
DEFAULT_VIS_DAYS = 7
MIN_VIS_RECORDS = 5
DEFAULT_REPEAT_THRESHOLD = 0.85
CATEGORY_ALIAS_MAP = {
    "entertainment": "ent",
    "learning": "edu",
    "news": "news",
    "social": "soc",
    "shopping": "shopping",
    "tools": "tools",
    "other": "other",
}
CHANNEL_CANONICAL_KEYS = ("ent", "edu", "news", "soc", "other")
CHANNEL_ALIAS_MAP = {
    "ent": "ent",
    "edu": "edu",
    "news": "news",
    "soc": "soc",
    "other": "other",
    "entertainment": "ent",
    "learning": "edu",
    "social": "soc",
    "娱乐": "ent",
    "学习": "edu",
    "新闻": "news",
    "社交": "soc",
}


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
    raise ValueError("tags must be a list, a supported text representation, or null")


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
            raise ValueError("meta must contain a JSON object or null") from None
        if parsed is None or isinstance(parsed, dict):
            return parsed
    raise ValueError("meta must be an object or null")


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


def _row_from_item(item: IngestItem, *, preserve_text: bool = False) -> Dict[str, Any]:
    title = (item.title or "").strip()
    # Versioned backups already passed their schema/checksum checks. Preserve
    # their text exactly; new collection/import retains its normal text policy.
    text = item.text
    if not preserve_text:
        text = _clean_optional_str(text)
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


def _upsert_embedding(
    conn: Any, item_id: int, title: Optional[str], text: Optional[str]
) -> None:
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


def _item_dataset_state(conn: Any) -> Dict[str, Any]:
    """Read inside the caller's transaction; timestamps alone miss same-ms inserts."""
    state = dict(conn.execute(
        "SELECT COUNT(*) AS total_count, COALESCE(MAX(id), 0) AS max_id, "
        "COALESCE(MAX(created_at), 0) AS max_created_at FROM items"
    ).fetchone())
    state.update(dict(conn.execute(
        "SELECT database_id, revision FROM items_revision WHERE singleton = 1"
    ).fetchone()))
    return state


def _stable_hash_payload(data: Dict[str, Any]) -> str:
    payload = json.dumps(
        data, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return sha256_hex(payload)


def _lsj_algorithms_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "lsj" / "src" / "algorithms"


def _ensure_lsj_import_path() -> None:
    alg_dir = _lsj_algorithms_dir()
    path = str(alg_dir)
    if path not in sys.path:
        sys.path.insert(0, path)


def _normalize_category_key(value: Any) -> str:
    return str(value or "other").strip().lower() or "other"


def _canonicalize_channel_key(value: Any) -> str:
    key = _clean_optional_str(value) or "other"
    return CHANNEL_ALIAS_MAP.get(key.lower(), "other")


def _normalize_channel_counts(counts: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    if counts is None:
        return None

    normalized: Dict[str, int] = {}
    for raw_key, raw_count in counts.items():
        canonical_key = _canonicalize_channel_key(raw_key)
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError):
            continue
        normalized[canonical_key] = normalized.get(canonical_key, 0) + count

    return {key: normalized.get(key, 0) for key in CHANNEL_CANONICAL_KEYS}


def _default_visualization_payload() -> Dict[str, Any]:
    return {
        "time_series": [],
        "category_distribution": {},
        "sentiment_distribution": {},
        "similarity_histogram": {},
        "hourly_distribution": {},
    }


def _default_visualization_result(
    *,
    from_ts: Optional[int],
    to_ts: Optional[int],
    limit_rows: int,
    input_count: int = 0,
    generated_at: Optional[int] = None,
    pipeline_warning: Optional[str] = None,
) -> Dict[str, Any]:
    aliases = {
        key: alias for key, alias in sorted(CATEGORY_ALIAS_MAP.items(), key=lambda item: item[0])
    }
    return {
        "window": {
            "from_ts": from_ts,
            "to_ts": to_ts,
            "limit_rows": limit_rows,
            "input_count": input_count,
        },
        "global": _default_visualization_payload(),
        "categories": {},
        "category_aliases": aliases,
        "category_counts": {},
        "analysis_status": "not_computed",
        "minimum_records": MIN_VIS_RECORDS,
        "date_timezone": "UTC",
        "pipeline_warning": pipeline_warning,
        "generated_at": generated_at if generated_at is not None else _now_ms(),
    }


def _prepare_analysis_dataframe(rows: List[Dict[str, Any]]) -> Any:
    import pandas as pd  # type: ignore

    df = pd.DataFrame(rows)
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna(df["title"]).astype(str)
    df["analysis_text"] = df["text"].where(
        df["text"].str.strip() != "", df["title"]
    )
    df["url"] = df["url"].fillna("").astype(str)
    df["channel"] = df["channel"].fillna("unknown").astype(str)
    return df


def _execute_lsj_pipeline(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"ok": True, "input_count": 0}

    try:
        _ensure_lsj_import_path()
        from classifier import ContentClassifier  # type: ignore
        from evaluator import InformationQualityEvaluator  # type: ignore
        from sentiment import SentimentAnalyzer  # type: ignore
        from similarity import SimilarityAnalyzer  # type: ignore

        df = _prepare_analysis_dataframe(rows)

        classifier = ContentClassifier()
        sentiment = SentimentAnalyzer()
        similarity = SimilarityAnalyzer()
        evaluator = InformationQualityEvaluator(
            sentiment_analyzer=sentiment,
            content_classifier=classifier,
            similarity_analyzer=similarity,
        )

        df1 = classifier.batch_predict(
            df[["id", "title", "url", "analysis_text", "channel", "ts"]].copy()
        )
        df2 = sentiment.batch_predict(
            df1, text_column="analysis_text", include_emotions=False, batch_size=500
        )
        df3 = similarity.batch_calculate_similarity(df2, text_column="analysis_text")
        if "similarity" not in df3.columns and "similarity_to_previous" in df3.columns:
            df3["similarity"] = df3["similarity_to_previous"]

        return {
            "ok": True,
            "input_count": int(len(df3)),
            "evaluator": evaluator,
            "df3": df3,
        }
    except Exception:
        # Model exceptions can contain source text or private filesystem paths.
        # Do not expose/persist them or replace failed inference with hash scores.
        return {
            "ok": False,
            "input_count": len(rows),
            "warning": "Experimental analysis is unavailable; check the local analysis dependencies and configuration.",
        }


def _build_visualization_result(
    rows: List[Dict[str, Any]],
    *,
    from_ts: Optional[int],
    to_ts: Optional[int],
    limit_rows: int,
) -> Dict[str, Any]:
    base = _default_visualization_result(
        from_ts=from_ts,
        to_ts=to_ts,
        limit_rows=limit_rows,
        input_count=len(rows),
    )
    if not rows:
        base["analysis_status"] = "empty"
        return base
    if len(rows) < MIN_VIS_RECORDS:
        base["analysis_status"] = "insufficient_data"
        return base
    execution = _execute_lsj_pipeline(rows)

    if not execution.get("ok"):
        base["analysis_status"] = "unavailable"
        base["pipeline_warning"] = execution.get("warning")
        return base

    try:
        from .visualization_metrics import build_visualization_metrics

        df3 = execution["df3"]
        # Adjacent-pair results only describe the original (ts, id) sequence.
        # A pipeline that drops or reorders rows must not silently relabel them.
        if (len(df3) != len(rows) or df3["id"].tolist() != [r["id"] for r in rows]
                or df3["ts"].tolist() != [r["ts"] for r in rows]):
            raise ValueError("Pipeline record alignment changed")
        metrics = build_visualization_metrics(
            df3, repeat_threshold=DEFAULT_REPEAT_THRESHOLD, category_aliases=CATEGORY_ALIAS_MAP,
        )
    except Exception:
        # Do not turn a failed metric computation into zeroes or a successful chart.
        base["analysis_status"] = "failed"
        base["pipeline_warning"] = "visualization computation failed"
        return base

    base["window"]["processed_count"] = metrics["processed_count"]
    base["analysis_status"] = "ready"
    base["global"] = metrics["global"]
    base["categories"] = metrics["categories"]
    base["category_counts"] = metrics["category_counts"]
    base["coverage"] = metrics["coverage"]
    return base


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


def _prune_analysis_history(conn: Any) -> None:
    """Keep terminal analysis history bounded without touching active work.

    Jobs are also used as a best-effort cache.  Retaining the newest terminal
    rows keeps recent cache reuse intact while allowing older keys to be
    recomputed instead of letting payloads grow the per-user database forever.
    The latest run rows are retained for the history endpoint; destructive
    data operations still clear all derived state through ``clear_derived``.
    """
    conn.execute(
        """
        DELETE FROM analysis_jobs
        WHERE status IN (?, ?)
          AND id NOT IN (
              SELECT id
              FROM analysis_jobs
              WHERE status IN (?, ?)
              ORDER BY id DESC
              LIMIT ?
          )
        """,
        (JOB_COMPLETED, JOB_FAILED, JOB_COMPLETED, JOB_FAILED,
         max(1, int(MAX_ANALYSIS_JOB_HISTORY))),
    )
    conn.execute(
        """
        DELETE FROM analysis_runs
        WHERE id NOT IN (
            SELECT id
            FROM analysis_runs
            ORDER BY id DESC
            LIMIT ?
        )
        """,
        (max(1, int(MAX_ANALYSIS_RUN_HISTORY)),),
    )


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
    if status in {JOB_COMPLETED, JOB_FAILED}:
        _prune_analysis_history(conn)


def _recover_interrupted_analysis_jobs() -> None:
    """Called only during startup while this process owns the database lock."""
    recovered_at = _now_ms()
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            UPDATE analysis_jobs
            SET status = ?, error = ?, finished_at = ?, updated_at = ?
            WHERE status IN (?, ?)
            """,
            (JOB_FAILED, "Analysis was interrupted before this service start; run it again.",
             recovered_at, recovered_at, JOB_QUEUED, JOB_RUNNING),
        )
        _prune_analysis_history(conn)


class AnalysisUnavailableError(RuntimeError):
    """Expected optional-inference failure; never carries raw model exceptions."""


def _run_lsj_pipeline(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "input_count": 0,
            "category_counts": {},
            "sentiment_counts": {},
            "comparison_count": 0,
            "sentiment_count": 0,
            "polarity_count": 0,
            "negative_ratio": None,
            "avg_sentiment": None,
            "repeat_ratio": None,
            "quick_evaluation": None,
            "full_report": None,
        }

    execution = _execute_lsj_pipeline(rows)
    if not execution.get("ok"):
        raise AnalysisUnavailableError("Experimental analysis is unavailable.")

    df3 = execution["df3"]
    # Scores and adjacent comparisons must describe the original input sequence.
    # Broken adapter output is an internal failure, not incomplete measurements.
    try:
        aligned = (type(execution.get("input_count")) is int
                   and execution["input_count"] == len(rows) and len(df3) == len(rows)
                   and df3["id"].tolist() == [row["id"] for row in rows]
                   and df3["ts"].tolist() == [row["ts"] for row in rows])
    except Exception:
        aligned = False
    if not aligned:
        raise RuntimeError("Pipeline record alignment changed")
    evaluator = execution["evaluator"]
    quick = evaluator.quick_evaluate(df3)
    report = evaluator.evaluate(df3, detailed=False).to_dict()

    category_norm = df3["category"].fillna("other").astype(str)
    similarities: List[float] = []
    polarities: List[float] = []
    sentiment_counts: Dict[str, int] = {}
    for index, row in enumerate(df3.to_dict("records")):
        similarity = finite_metric(row.get("similarity"), 0, 1)
        if (index > 0 and similarity is not None
                and ("similarity_valid" not in row or row["similarity_valid"] is True)):
            similarities.append(similarity)
        sentiment = row.get("sentiment")
        label = sentiment.strip().lower() if isinstance(sentiment, str) else None
        if (label not in {"positive", "neutral", "negative"}
                or ("sentiment_valid" in row and row["sentiment_valid"] is not True)):
            continue
        # Only the metric with missing measurements loses observations; category
        # counts above retain the complete input, including the first record.
        key = label.title()
        sentiment_counts[key] = sentiment_counts.get(key, 0) + 1
        polarity = finite_metric(row.get("polarity"), -1, 1)
        if polarity is not None:
            polarities.append(polarity)
    sentiment_count = sum(sentiment_counts.values())
    negative_ratio = sentiment_counts.get("Negative", 0) / sentiment_count if sentiment_count else None
    avg_sentiment = math.fsum(polarities) / len(polarities) if polarities else None
    repeat_ratio = (sum(value >= DEFAULT_REPEAT_THRESHOLD for value in similarities) / len(similarities)
                    if similarities else None)

    category_counts = {
        str(k): int(v) for k, v in category_norm.value_counts().to_dict().items()
    }
    return {
        "input_count": len(rows),
        "category_counts": category_counts,
        "sentiment_counts": sentiment_counts,
        "comparison_count": len(similarities),
        "sentiment_count": sentiment_count,
        "polarity_count": len(polarities),
        "negative_ratio": negative_ratio,
        "avg_sentiment": avg_sentiment,
        "repeat_ratio": repeat_ratio,
        "quick_evaluation": quick,
        "full_report": report,
        "pipeline_warning": None,
    }


def _get_job_row(conn: Any, job_id: int) -> Optional[Dict[str, Any]]:
    # Python integers are unbounded, but SQLite INTEGER primary keys are not.
    # An unrepresentable identifier cannot name a stored job.
    if not -(2**63) <= job_id <= 2**63 - 1:
        return None
    row = conn.execute(
        "SELECT * FROM analysis_jobs WHERE id = ? LIMIT 1",
        (job_id,),
    ).fetchone()
    if row is None:
        return None
    data = dict(row)
    _validate_stored_analysis(data)
    for key in ("result_payload", "metrics_json"):
        data[key] = _read_stored_analysis(data.get(key))
    data["cache_hit"] = bool(data.get("cache_hit", 0))
    return data


def insert_items(items: Iterable[IngestItem], *, connection=None, preserve_text: bool = False) -> Tuple[int, int]:
    inserted = 0
    duplicates = 0
    sql = """
        INSERT INTO items (
            url, title, text, ts, source, lang, channel, author, tags, meta,
            url_hash, content_hash, created_at
        ) VALUES (
            :url, :title, :text, :ts, :source, :lang, :channel, :author, :tags, :meta,
            :url_hash, :content_hash, :created_at
        ) ON CONFLICT(url_hash) DO NOTHING
    """
    with (get_conn() if connection is None else nullcontext(connection)) as conn:
        for item in items:
            row = _row_from_item(item, preserve_text=True) if preserve_text else _row_from_item(item)
            cur = conn.execute(sql, row)
            if cur.rowcount == 1:
                inserted += 1
                _upsert_embedding(
                    conn, int(cur.lastrowid), row.get("title"), row.get("text")
                )
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


# 以下函数用于辅助“后端->逻辑”的接口实现
def _stored_analysis_conflict() -> HTTPException:
    return HTTPException(status_code=409, detail={
        "code": "stored_analysis_invalid", "message": "Stored analysis data is invalid.",
    })


def _validate_stored_analysis(value: Any) -> None:
    try:
        validate_analysis_value(value)
    except InvalidStoredAnalysis:
        raise _stored_analysis_conflict() from None


def _read_stored_analysis(value: Any) -> Any:
    try:
        return load_analysis_json(value)
    except InvalidStoredAnalysis:
        raise _stored_analysis_conflict() from None


def _load_analysis_cache(value: Any) -> Dict[str, Any]:
    """Invalid stored caches are recomputed; their historical rows stay intact."""
    try:
        payload = load_analysis_json(value)
        return payload if isinstance(payload, dict) else {}
    except InvalidStoredAnalysis:
        return {}


def _cache_count(value: Any, maximum: int, minimum: int = 0) -> bool:
    return type(value) is int and minimum <= value <= maximum


def _valid_visualization_series(value: Any, *, maximum: int, first_date: str, last_date: str) -> bool:
    """Check current daily chart fields, preserving independent missing metrics."""
    if not isinstance(value, list):
        return False
    previous_date = ""
    metric_counts = {
        "avg_polarity": ("polarity_count", -1),
        "avg_similarity": ("comparison_count", 0),
        "repeat_ratio": ("comparison_count", 0),
        "positive_ratio": ("sentiment_count", 0),
        "neutral_ratio": ("sentiment_count", 0),
        "negative_ratio": ("sentiment_count", 0),
    }
    for row in value:
        if not isinstance(row, dict) or not _cache_count(row.get("count"), maximum, 1):
            return False
        date = row.get("date")
        if not isinstance(date, str) or len(date) != 10 or not first_date <= date <= last_date:
            return False
        try:
            if datetime.fromisoformat(date).date().isoformat() != date or date <= previous_date:
                return False
        except ValueError:
            return False
        previous_date = date
        if any(not _cache_count(row.get(key), row["count"])
               for key in ("comparison_count", "sentiment_count", "polarity_count")):
            return False
        if row["polarity_count"] > row["sentiment_count"]:
            return False
        for key, (count_key, lower) in metric_counts.items():
            if key not in row:
                return False
            if row[count_key] == 0:
                if row[key] is not None:
                    return False
            elif finite_metric(row[key], lower, 1) is None:
                return False
        if row["sentiment_count"] and not math.isclose(
            sum(row[f"{label}_ratio"] for label in ("positive", "neutral", "negative")),
            1.0, abs_tol=1e-5,
        ):
            return False
    return sum(row["count"] for row in value) <= maximum


def _valid_visualization_cache(
    payload: Dict[str, Any], *, from_ts: int, to_ts: int, limit_rows: int,
    input_count: int, available_count: int,
) -> bool:
    """Only reuse structurally complete schema-v3 chart results for this snapshot.

    This is a cache boundary, not a historical migration or model-quality check.
    Invalid rows remain in history and the existing analysis path recomputes.
    """
    if (payload.get("analysis_status") != "ready" or payload.get("pipeline_warning")
            or input_count < MIN_VIS_RECORDS or payload.get("date_timezone") != "UTC"
            or type(payload.get("minimum_records")) is not int
            or payload["minimum_records"] != MIN_VIS_RECORDS
            or not _cache_count(payload.get("generated_at"), MAX_INGEST_TS)
            or payload.get("category_aliases") != CATEGORY_ALIAS_MAP):
        return False
    window = payload.get("window")
    expected = {"from_ts": from_ts, "to_ts": to_ts, "limit_rows": limit_rows,
                "input_count": input_count, "processed_count": input_count,
                "available_count": available_count}
    if not isinstance(window, dict) or any(type(window.get(key)) is not int or window[key] != value
                                          for key, value in expected.items()):
        return False
    if window.get("truncated") is not (available_count > input_count):
        return False
    coverage = payload.get("coverage")
    coverage_keys = ("record_count", "timestamp_count", "category_count", "comparison_count", "sentiment_count")
    if not isinstance(coverage, dict) or any(not _cache_count(coverage.get(key), input_count)
                                            for key in coverage_keys):
        return False
    if coverage["record_count"] != input_count or coverage["comparison_count"] > input_count - 1:
        return False
    counts, categories, charts = payload.get("category_counts"), payload.get("categories"), payload.get("global")
    if not all(isinstance(value, dict) for value in (counts, categories, charts)):
        return False
    if (set(counts) - set(CATEGORY_ALIAS_MAP) or set(categories) != set(counts)
            or any(not _cache_count(count, input_count, 1) for count in counts.values())
            or sum(counts.values()) != coverage["category_count"]):
        return False
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    first_date = (epoch + timedelta(milliseconds=from_ts)).date().isoformat()
    last_date = (epoch + timedelta(milliseconds=to_ts)).date().isoformat()
    series = charts.get("time_series")
    if not _valid_visualization_series(series, maximum=input_count, first_date=first_date, last_date=last_date):
        return False
    if sum(row["count"] for row in series) != coverage["timestamp_count"]:
        return False
    missing_dates = input_count - coverage["timestamp_count"]
    if any(not coverage[key] - missing_dates <= sum(row[key] for row in series) <= coverage[key]
           for key in ("comparison_count", "sentiment_count")):
        return False
    by_date = {row["date"]: row for row in series}
    category_daily_counts: Dict[str, int] = {}
    for key, category in categories.items():
        if (not isinstance(category, dict) or category.get("label") != key
                or category.get("alias") != CATEGORY_ALIAS_MAP[key]
                or not _valid_visualization_series(category.get("time_series"), maximum=counts[key],
                                                   first_date=first_date, last_date=last_date)):
            return False
        if sum(row["count"] for row in category["time_series"]) < counts[key] - missing_dates:
            return False
        for row in category["time_series"]:
            global_row = by_date.get(row["date"])
            if global_row is None or any(row[field] > global_row[field] for field in
                                         ("count", "comparison_count", "sentiment_count", "polarity_count")):
                return False
            category_daily_counts[row["date"]] = category_daily_counts.get(row["date"], 0) + row["count"]
    if any(count > by_date[date]["count"] for date, count in category_daily_counts.items()):
        return False
    distribution = charts.get("category_distribution")
    if not isinstance(distribution, dict) or set(distribution) != set(counts):
        return False
    if any(finite_metric(distribution[key], 0, 1) is None or not math.isclose(
        distribution[key], count / coverage["category_count"], abs_tol=1e-6,
    ) for key, count in counts.items()):
        return False
    sentiment = charts.get("sentiment_distribution")
    if (not isinstance(sentiment, dict) or set(sentiment) - {"positive", "neutral", "negative"}
            or any(finite_metric(value, 0, 1) is None or value <= 0 for value in sentiment.values())
            or bool(sentiment) != bool(coverage["sentiment_count"])):
        return False
    if sentiment and not math.isclose(sum(sentiment.values()), 1.0, abs_tol=1e-5):
        return False
    histogram, hourly = charts.get("similarity_histogram"), charts.get("hourly_distribution")
    bins = {"0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"}
    if (not isinstance(histogram, dict) or set(histogram) != (bins if coverage["comparison_count"] else set())
            or any(not _cache_count(value, input_count) for value in histogram.values())
            or sum(histogram.values()) != coverage["comparison_count"]):
        return False
    if (not isinstance(hourly, dict) or set(hourly) - {str(hour) for hour in range(24)}
            or any(not _cache_count(value, input_count, 1) for value in hourly.values())
            or sum(hourly.values()) != coverage["timestamp_count"]):
        return False
    return True


def _shape_export_rows(rows: Iterable[Dict[str, Any]], view: str) -> Iterable[Dict[str, Any]]:
    for r in rows:
        title = _clean_optional_str(r.get("title")) or ""
        text = _clean_optional_str(r.get("text")) or title

        tags = load_analysis_json(r.get("tags"))
        meta = load_analysis_json(r.get("meta"))

        if view == "analysis":
            yield {
                    "id": r.get("id"),
                    "title": title,
                    "url": r.get("url"),
                    "analysis_text": text,
                    "text": text,
                    "channel": _clean_optional_str(r.get("channel")) or "unknown",
                    "ts": r.get("ts"),
                    "source": r.get("source"),
                    "lang": r.get("lang"),
                    "author": r.get("author"),
                    "tags": tags,
                    "meta": meta,
                }
        else:  # raw
            yield {
                    "id": r.get("id"),
                    "url": r.get("url"),
                    "title": title,
                    "text": text,
                    "ts": r.get("ts"),
                    "source": r.get("source"),
                    "lang": r.get("lang"),
                    "channel": r.get("channel"),
                    "author": r.get("author"),
                    "tags": tags,
                    "meta": meta,
                    "created_at": r.get("created_at"),
                }


EXPORT_CSV_COLUMNS = {
    "analysis": ("id", "title", "url", "analysis_text", "text", "channel", "ts",
                 "source", "lang", "author", "tags", "meta"),
    "raw": ("id", "url", "title", "text", "ts", "source", "lang", "channel",
            "author", "tags", "meta", "created_at"),
}
TRAINING_CSV_COLUMNS = ("input", "label", "ts", "url", "title", "source")


def _host_is_private(host: str) -> bool:
    h = (host or "").strip().lower().rstrip(".")
    if not h:
        return False
    if h == "localhost" or h.endswith((".localhost", ".local")):
        return True
    try:
        address = ipaddress.ip_address(h)
    except ValueError:
        # Inspect the URL only: resolving a DNS name leaks browsing hosts and
        # cannot guarantee the address to which it will resolve in the future.
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    # Global/private flags alone miss some reserved, multicast and site-local IPs.
    return (not address.is_global or address.is_multicast or address.is_loopback
            or address.is_link_local or address.is_unspecified or address.is_reserved
            or (isinstance(address, ipaddress.IPv6Address) and address.is_site_local))


def _is_internal_url(url: str) -> bool:
    try:
        u = urlparse(url)
        return _host_is_private(u.hostname or "")
    except Exception:
        return False


def _is_auth_like(url: str, title: Optional[str] = None) -> bool:
    s = f"{url} {title or ''}".lower()
    keywords = [
        "login",
        "signin",
        "sign-in",
        "signup",
        "sign-up",
        "oauth",
        "authorize",
        "sso",
        "callback",
        "logout",
        "register",
        "captcha",
        "verify",
        "auth",
    ]
    return any(k in s for k in keywords)


def _is_search_like(url: str, title: Optional[str] = None) -> bool:
    s = f"{url} {title or ''}".lower()
    keywords = [
        "/search",
        "search?",
        "q=",
        "query=",
        "code search results",
        "google search",
    ]
    return any(k in s for k in keywords)


def _compress_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _prepare_training_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    budget: export_io.PreparationBudget,
    label_field: str,
    exclude_internal: bool,
    exclude_auth_pages: bool,
    exclude_search_pages: bool,
    dedup_by_input: bool,
    max_input_len: int,
) -> Iterable[Dict[str, Any]]:
    for r in rows:
        url = str(r.get("url") or "")
        title = _clean_optional_str(r.get("title")) or ""
        text = _clean_optional_str(r.get("text")) or title

        if exclude_internal and _is_internal_url(url):
            continue
        if exclude_auth_pages and _is_auth_like(url, title):
            continue
        if exclude_search_pages and _is_search_like(url, title):
            continue

        input_text = _compress_ws(text)
        if not input_text:
            continue
        if len(input_text) > max_input_len:
            input_text = input_text[:max_input_len]

        label = _clean_optional_str(r.get(label_field)) or "unknown"

        if dedup_by_input:
            key = normalize_text("", input_text)
            if budget.duplicate(key):
                continue

        yield {
                "input": input_text,
                "label": label,
                "ts": r.get("ts"),
                "url": url,
                "title": title,
                "source": r.get("source"),
            }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    with process_ownership(db.DB_PATH):
        _app.state.security = LocalSecurity.load(db.DB_PATH)
        _app.state.operation_lock = asyncio.Lock()
        _app.state.analysis_lock = asyncio.Lock()
        _app.state.request_slots = asyncio.Semaphore(4)
        _app.state.export_slots = asyncio.Semaphore(2)
        work_owner = WorkOwner()
        _app.state.work_owner = work_owner
        init_db(_schema_path())
        _recover_interrupted_analysis_jobs()
        try:
            yield
        finally:
            # Uvicorn may end lifespan immediately after cancelling requests.
            # Retain the OS database lock until their actual workers finish.
            await work_owner.drain()


app = FastAPI(
    title="Information Diet Manager (MVP)",
    lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None,
)

@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request, exc: RequestValidationError):
    # Do not echo page text, invalid JSON values or exception contexts from a request.
    details = [{"type": error["type"], "loc": error["loc"], "msg": error["msg"]}
               for error in exc.errors()]
    return Response(json.dumps({"detail": details}, ensure_ascii=True),
                    status_code=422, media_type="application/json")


app.add_middleware(LocalAccessMiddleware)
install_data_routes(app, insert_items)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/session")
def session(request: Request):
    return {"role": request.scope["idm_role"]}


def _collect_sync(item: IngestItem) -> IngestAck:
    inserted, duplicates = insert_items([item])
    return IngestAck(inserted=inserted, duplicates=duplicates, failed=0)


@app.post("/collect", response_model=IngestAck)
async def collect(item: IngestItem) -> IngestAck:
    return await run_owned_sync(_collect_sync, item=item)


@app.post("/import", response_model=IngestAck)
async def import_items(file: UploadFile = File(...)) -> IngestAck:
    # Keep FastAPI's request-owned upload open until its worker stops reading.
    return await run_owned_sync(_import_items_sync, file=file)


def _import_items_sync(file: UploadFile) -> IngestAck:
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
    except (ValueError, csv.Error, RecursionError):
        raise HTTPException(
            status_code=400, detail="Invalid file contents."
        ) from None
    items: List[IngestItem] = []
    for raw in raw_items:
        try:
            items.append(_prepare_item(raw))
        except Exception:
            failed += 1
    inserted, duplicates = insert_items(items)
    return IngestAck(inserted=inserted, duplicates=duplicates, failed=failed)


def _read_stored_item_json(value: Any) -> Any:
    """Apply shared JSON safety checks without imposing ingest-field shapes."""
    try:
        return load_analysis_json(value)
    except InvalidStoredAnalysis:
        raise HTTPException(409, detail={
            "code": "stored_item_invalid", "message": "Stored item JSON is invalid.",
        }) from None


@app.get("/items")
def list_items(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    limit: Optional[int] = Query(None, ge=1, le=200),
    pagination: str = Query("offset", pattern="^(offset|cursor)$"),
    cursor: Optional[str] = Query(None, min_length=1, max_length=2048),
) -> Dict[str, Any]:
    if cursor is not None and pagination != "cursor":
        raise HTTPException(400, detail="A cursor requires pagination=cursor.")
    if pagination == "cursor" and (page != 1 or limit is not None):
        raise HTTPException(400, detail="Cursor pagination uses page_size and cursor only.")
    if limit is not None:
        page_size = limit
    offset = (page - 1) * page_size
    with get_conn() as conn:
        conn.execute("BEGIN")
        if pagination == "cursor":
            rows, metadata = read_cursor_page(conn, page_size=page_size, cursor=cursor)
        else:
            # Legacy page-number clients retain their contract, but each count
            # and page now share a SQLite snapshot within this request.
            total = conn.execute("SELECT COUNT(*) AS cnt FROM items").fetchone()["cnt"]
            rows = conn.execute(
                "SELECT * FROM items ORDER BY id DESC LIMIT ? OFFSET ?",
                (page_size, min(offset, 2**63 - 1)),
            ).fetchall()
            metadata = {"page": page, "page_size": page_size, "total": total}
    items: List[Dict[str, Any]] = []
    for row in rows:
        item: Dict[str, Any] = dict(row)
        for field in ("tags", "meta"):
            item[field] = _read_stored_item_json(item.get(field))
        items.append(item)
    return {**metadata, "items": items}


def _global_statistics(*, force: bool, backfill_limit: int, record_run: bool) -> Dict[str, Any]:
    """Whole-library page counts, never windowed model scores.

    stats_daily is a compatibility projection only: old releases wrote arbitrary
    analysis windows there. Only explicitly versioned global run payloads can
    be reused, and all state/count/cache reads share a short write transaction.
    """
    day = datetime.now(timezone.utc).date().isoformat()
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        item_state = _item_dataset_state(conn)
        max_created_at = int(item_state["max_created_at"])
        data_version = _stable_hash_payload(item_state)
        existing = conn.execute(
            "SELECT payload FROM analysis_runs WHERE day = ? ORDER BY id DESC LIMIT 1",
            (day,),
        ).fetchone()
        payload = None
        if not force and existing is not None:
            candidate = _load_analysis_cache(existing["payload"])
            if valid_global_statistics_cache(
                candidate, day=day, data_version=data_version, total_count=item_state["total_count"],
            ):
                payload = {**candidate, "cached": True, "embeddings_backfilled": 0}

        if payload is None:
            backfilled = _backfill_missing_embeddings(conn, limit=backfill_limit) if backfill_limit > 0 else 0
            content = conn.execute(
                "SELECT COUNT(*) AS measured, COUNT(DISTINCT content_hash) AS distinct_count "
                "FROM items WHERE content_hash IS NOT NULL AND content_hash != ''"
            ).fetchone()
            sample_count = int(content["measured"])
            repeat_ratio = ((sample_count - int(content["distinct_count"])) / sample_count
                            if sample_count else None)
            channel_rows = conn.execute("SELECT channel, COUNT(*) AS cnt FROM items GROUP BY channel").fetchall()
            channel_counts = {key: 0 for key in CHANNEL_CANONICAL_KEYS}
            for row in channel_rows:
                channel_counts[_canonicalize_channel_key(row["channel"])] += int(row["cnt"])
            payload = {
                "day": day,
                "total_count": int(item_state["total_count"]),
                "channel_counts": channel_counts,
                "repeat_ratio": repeat_ratio,
                "repeat_sample_count": sample_count,
                "repeat_metric": "duplicate_normalized_content_hash_fraction",
                "negative_ratio": None,
                "avg_sentiment": None,
                "generated_at": _now_ms(),
                "cached": False,
                "embeddings_backfilled": backfilled,
                "statistics_scope": "all_saved_pages",
                "statistics_version": 1,
                "data_version": data_version,
            }

        # Rewrite even a cache hit: this repairs old/mixed daily projections
        # without ever interpreting them as trustworthy global statistics.
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
                payload["total_count"],
                _as_json(payload["channel_counts"]),
                payload["repeat_ratio"],
                None,
                None,
                payload["generated_at"],
                payload["generated_at"],
            ),
        )
        if record_run or not payload["cached"]:
            conn.execute(
                """
                INSERT INTO analysis_runs (
                    day, total_count, channel_counts, repeat_ratio, negative_ratio, avg_sentiment,
                    payload, cached, item_max_created_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (day, payload["total_count"], _as_json(payload["channel_counts"]), payload["repeat_ratio"],
                 None, None, _as_json(payload), int(payload["cached"]), max_created_at, _now_ms()),
            )
        _prune_analysis_history(conn)
        return payload


@app.post("/analyze/run")
async def run_analysis(
    force: bool = Query(False),
    backfill_limit: int = Query(2000, ge=0, le=20000),
) -> Dict[str, Any]:
    return await run_owned_sync(_global_statistics, force=force, backfill_limit=backfill_limit, record_run=True)


@app.get("/analyze/history")
def analyze_history(limit: int = Query(20, ge=1, le=200)) -> Dict[str, Any]:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) AS cnt FROM analysis_runs").fetchone()["cnt"]
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
        _validate_stored_analysis(run)
        run["channel_counts"] = _read_stored_analysis(run.get("channel_counts"))
        run["cached"] = bool(run.get("cached", 0))
        runs.append(run)
    return {"total": int(total), "returned": len(runs), "runs": runs}


@app.post("/analyze/run_full")
async def run_full_analysis(
    force: bool = Query(False),
    from_ts: Optional[int] = Query(None, ge=0, le=MAX_INGEST_TS),
    to_ts: Optional[int] = Query(None, ge=0, le=MAX_INGEST_TS),
    limit_rows: int = Query(5000, ge=1, le=50000),
) -> Dict[str, Any]:
    return await run_owned_sync(
        _run_full_analysis_snapshot, force=force, from_ts=from_ts,
        to_ts=to_ts, limit_rows=limit_rows,
    )


def _failed_analysis_response(conn, job_id, started_at, input_count, status, message):
    duration = _now_ms() - started_at
    _update_analysis_job(
        conn, job_id, status=JOB_FAILED, error=message,
        metrics_json={"cache_hit": False, "input_count": input_count, "duration_ms": duration},
        duration_ms=duration, finished_at=_now_ms(),
    )
    # Returning lets the connection commit the failed job.
    return JSONResponse(status_code=status, content={"job_id": job_id, "status": JOB_FAILED, "detail": message})


def _run_full_analysis_snapshot(*, force, from_ts, to_ts, limit_rows):
    if from_ts is not None and to_ts is not None and from_ts > to_ts:
        raise HTTPException(
            status_code=400, detail="from_ts cannot be greater than to_ts"
        )
    day = datetime.now(timezone.utc).date().isoformat()
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        item_state = _item_dataset_state(conn)
        revision = {key: item_state[key] for key in ("database_id", "revision")}
        max_created_at = int(item_state["max_created_at"])
        rows = _load_items_for_analysis(
            conn, from_ts=from_ts, to_ts=to_ts, limit_rows=limit_rows
        )
        input_count = len(rows)
        job_key = {
            "day": day,
            "from_ts": from_ts,
            "to_ts": to_ts,
            "limit_rows": limit_rows,
            "mode": "run_full",
            "schema_version": 2,
            "dataset": item_state,
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
            cached_payload = _load_analysis_cache(cached["result_payload"]) if cached else {}
            if cached and valid_full_analysis_cache(
                cached_payload, day=day, from_ts=from_ts, to_ts=to_ts,
                limit_rows=limit_rows, input_count=input_count,
            ):
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
                    result_payload={
                        **cached_payload,
                        "experimental": True,
                        "experimental_notice": EXPERIMENTAL_ANALYSIS_NOTICE,
                    },
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
                    "result": {
                        **cached_payload,
                        "experimental": True,
                        "experimental_notice": EXPERIMENTAL_ANALYSIS_NOTICE,
                    },
                }
        # not cached的情况也自然下落到这里
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

    # Inference owns no SQLite connection or data-operation lock. Inserts can
    # proceed; destructive changes invalidate this snapshot before publication.
    failure = None
    try:
        pipeline = _run_lsj_pipeline(rows)
        now_ms = _now_ms()

        channel_counts: Dict[str, int] = {}
        for r in rows:
            key = _canonicalize_channel_key(r.get("channel"))
            channel_counts[key] = channel_counts.get(key, 0) + 1
        channel_counts = _normalize_channel_counts(channel_counts) or {
            key: 0 for key in CHANNEL_CANONICAL_KEYS
        }

        # Windowed experimental metrics belong only to their run/job.
        # stats_daily is reserved for whole-library basic page statistics.
        payload = {
            "day": day,
            "statistics_scope": "analysis_window",
            "statistics_version": 2,
            "experimental": True,
            "experimental_notice": EXPERIMENTAL_ANALYSIS_NOTICE,
            "analysis_status": "ready" if input_count else "empty",
            "repeat_metric": "legacy_adjacent_text_similarity_threshold_fraction",
            "window": {"from_ts": from_ts, "to_ts": to_ts, "limit_rows": limit_rows,
                       "input_count": input_count},
            "total_count": int(pipeline["input_count"]),
            "channel_counts": channel_counts,
            "category_counts": pipeline["category_counts"],
            "sentiment_counts": pipeline["sentiment_counts"],
            "comparison_count": pipeline["comparison_count"],
            "sentiment_count": pipeline["sentiment_count"],
            "polarity_count": pipeline["polarity_count"],
            "repeat_ratio": pipeline["repeat_ratio"],
            "negative_ratio": pipeline["negative_ratio"],
            "avg_sentiment": pipeline["avg_sentiment"],
            "quick_evaluation": pipeline["quick_evaluation"],
            "full_report": pipeline["full_report"],
            "pipeline_warning": pipeline.get("pipeline_warning"),
            "generated_at": now_ms,
            "cached": False,
        }
    except AnalysisUnavailableError:
        failure = (503, "Experimental analysis is unavailable; check the local analysis dependencies and configuration.")
    except ValueError:
        failure = (422, "Legacy scoring lacks complete valid measurements; use /dashboard/visualization for partial statistics.")
    except Exception:
        failure = (500, "Legacy analysis failed; no valid result was produced.")

    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _require_analysis_revision(conn, revision, job_id=job_id)
        if failure is not None:
            return _failed_analysis_response(conn, job_id, started_at, input_count, *failure)
        conn.execute("SAVEPOINT legacy_analysis_result")
        try:
            # Newly published reports must satisfy the same finite, UTF-8 and
            # depth contract as historical reads. Fail inside the savepoint.
            validate_analysis_value(payload)
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
                    payload["repeat_ratio"],
                    payload["negative_ratio"],
                    payload["avg_sentiment"],
                    # Reject non-finite legacy reports inside the savepoint;
                    # never persist a completed job with invalid JSON numbers.
                    json.dumps(payload, ensure_ascii=False, allow_nan=False),
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
            conn.execute("RELEASE legacy_analysis_result")
            return {
                "job_id": job_id,
                "status": JOB_COMPLETED,
                "cached": False,
                "result": payload,
            }
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK TO legacy_analysis_result")
                conn.execute("RELEASE legacy_analysis_result")
            else:
                # A whole-transaction rollback removes the savepoint and
                # releases the writer; another change may now own this data.
                conn.execute("BEGIN IMMEDIATE")
                _require_analysis_revision(conn, revision, job_id=job_id)
            return _failed_analysis_response(
                conn, job_id, started_at, input_count, 500,
                "Legacy analysis failed; no valid result was produced.",
            )


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
        raise HTTPException(
            status_code=409, detail=f"job not completed: {row.get('status')}"
        )
    return {
        "job_id": row["id"],
        "status": row["status"],
        "result": row.get("result_payload"),
        "metrics": row.get("metrics_json"),
        "cache_hit": row.get("cache_hit"),
    }


@app.get("/dashboard/summary")
async def dashboard_summary() -> Dict[str, Any]:
    return await run_owned_sync(_global_statistics, force=False, backfill_limit=2000, record_run=False)


# 以下路由用于“后端->逻辑”的接口
def _require_analysis_revision(conn: Any, expected: Dict[str, Any], *, job_id: Optional[int] = None) -> None:
    actual = dict(conn.execute("SELECT database_id, revision FROM items_revision WHERE singleton = 1").fetchone())
    job = None if job_id is None else conn.execute("SELECT status FROM analysis_jobs WHERE id = ?", (job_id,)).fetchone()
    job_inactive = job_id is not None and (job is None or job["status"] != JOB_RUNNING)
    if actual != expected or job_inactive:
        if job_id is not None:
            # A delete/restore may already have removed this job: UPDATE never
            # recreates it. Direct external updates can leave it present; finish
            # that job before raising so it cannot remain RUNNING indefinitely.
            _update_analysis_job(conn, job_id, status=JOB_FAILED,
                                 error="analysis snapshot expired", finished_at=_now_ms())
            conn.commit()
        raise HTTPException(409, detail={
            "code": "analysis_snapshot_expired",
            "message": "Records changed during analysis; run analysis again.",
        })


@app.get("/dashboard/visualization")
async def dashboard_visualization(
    days: int = Query(DEFAULT_VIS_DAYS, ge=1, le=90),
    from_ts: Optional[int] = Query(None, ge=0, le=MAX_INGEST_TS),
    to_ts: Optional[int] = Query(None, ge=0, le=MAX_INGEST_TS),
    limit_rows: int = Query(5000, ge=1, le=50000),
    force: bool = Query(False),
) -> Dict[str, Any]:
    return await run_owned_sync(
        _dashboard_visualization_snapshot, days=days, from_ts=from_ts,
        to_ts=to_ts, limit_rows=limit_rows, force=force,
    )


def _dashboard_visualization_snapshot(*, days, from_ts, to_ts, limit_rows, force):
    if from_ts is not None and to_ts is not None and from_ts > to_ts:
        raise HTTPException(
            status_code=400, detail="from_ts cannot be greater than to_ts"
        )

    now_ms = _now_ms()
    resolved_to_ts = to_ts if to_ts is not None else now_ms
    resolved_from_ts = (
        from_ts
        if from_ts is not None
        else max(0, resolved_to_ts - days * 24 * 60 * 60 * 1000)
    )
    if resolved_from_ts > resolved_to_ts:
        raise HTTPException(400, detail="from_ts cannot be greater than the resolved to_ts")

    day = datetime.now(timezone.utc).date().isoformat()
    with get_conn() as conn:
        # Keep the coverage count and selected rows in the same read snapshot.
        conn.execute("BEGIN")
        available_count = conn.execute(
            "SELECT COUNT(*) FROM items WHERE ts >= ? AND ts <= ?",
            (resolved_from_ts, resolved_to_ts),
        ).fetchone()[0]
        item_state = conn.execute(
            "SELECT COALESCE(MAX(created_at), 0) AS max_created_at, COALESCE(MAX(id), 0) AS max_id FROM items"
        ).fetchone()
        revision = dict(conn.execute("SELECT database_id, revision FROM items_revision WHERE singleton = 1").fetchone())
        max_created_at = int(item_state["max_created_at"] or 0)
        rows = _load_items_for_analysis(
            conn,
            from_ts=resolved_from_ts,
            to_ts=resolved_to_ts,
            limit_rows=limit_rows,
        )
        input_count = len(rows)
        job_key = {
            "days": days,
            "from_ts": resolved_from_ts,
            "to_ts": resolved_to_ts,
            "limit_rows": limit_rows,
            "mode": "dashboard_visualization",
            "schema_version": 3,
            "data_revision": revision,
            "max_id": int(item_state["max_id"]),
        }
        input_hash = _stable_hash_payload(job_key)

    # Release the read snapshot before writing job/cache metadata.
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _require_analysis_revision(conn, revision)
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
                cached_payload = _load_analysis_cache(cached["result_payload"])
                if _valid_visualization_cache(
                    cached_payload, from_ts=resolved_from_ts, to_ts=resolved_to_ts,
                    limit_rows=limit_rows, input_count=input_count, available_count=int(available_count),
                ):
                    job_id = _insert_analysis_job(
                        conn,
                        status=JOB_COMPLETED,
                        input_hash=input_hash,
                        day=day,
                        from_ts=resolved_from_ts,
                        to_ts=resolved_to_ts,
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
                            "mode": "dashboard_visualization",
                        },
                        duration_ms=0,
                        started_at=_now_ms(),
                        finished_at=_now_ms(),
                    )
                    cached_payload["cached"] = True
                    cached_payload["reused_from_job_id"] = int(cached["id"])
                    return cached_payload

        job_id = _insert_analysis_job(
            conn,
            status=JOB_RUNNING,
            input_hash=input_hash,
            day=day,
            from_ts=resolved_from_ts,
            to_ts=resolved_to_ts,
            limit_rows=limit_rows,
            item_max_created_at=max_created_at,
            input_count=input_count,
            cache_hit=False,
        )

    # Optional inference must not hold SQLite's writer lock and block collection
    # or deletion. Destructive changes invalidate the result before publication.
    started_at = _now_ms()
    payload = _build_visualization_result(
        rows,
        from_ts=resolved_from_ts,
        to_ts=resolved_to_ts,
        limit_rows=limit_rows,
    )
    payload["window"]["available_count"] = int(available_count)
    payload["window"]["truncated"] = available_count > len(rows)
    payload["cached"] = False
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _require_analysis_revision(conn, revision, job_id=job_id)
        conn.execute("SAVEPOINT visualization_result")
        try:
            # Validate the persisted JSON representation: hourly keys are
            # integers in metrics and have always become strings on the wire.
            # Keep serialization and partial write effects in one boundary.
            payload = load_analysis_json(json.dumps(payload, ensure_ascii=False, allow_nan=False))
            _update_analysis_job(
                conn,
                job_id,
                status=JOB_FAILED if payload["analysis_status"] in {"failed", "unavailable"} else JOB_COMPLETED,
                result_payload=payload,
                metrics_json={
                    "input_count": input_count,
                    "cache_hit": False,
                    "mode": "dashboard_visualization",
                    "pipeline_warning": payload.get("pipeline_warning"),
                },
                duration_ms=_now_ms() - started_at,
                started_at=started_at,
                finished_at=_now_ms(),
            )
            conn.execute("RELEASE visualization_result")
            return payload
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK TO visualization_result")
                conn.execute("RELEASE visualization_result")
            else:
                # SQLite can abort the entire transaction (including its
                # savepoints). Reacquire and recheck before recording failure.
                conn.execute("BEGIN IMMEDIATE")
                _require_analysis_revision(conn, revision, job_id=job_id)
            return _failed_analysis_response(
                conn, job_id, started_at, input_count, 500,
                "Visualization analysis failed; no valid result was produced.",
            )


@app.get("/export/lsj")
async def export_lsj(
    from_ts: Optional[int] = Query(None, ge=0, le=MAX_INGEST_TS),
    to_ts: Optional[int] = Query(None, ge=0, le=MAX_INGEST_TS),
    limit_rows: int = Query(5000, ge=1, le=200000),
    view: str = Query("analysis", pattern="^(analysis|raw)$"),
    fmt: str = Query("json", pattern="^(json|jsonl|csv)$"),
) -> Any:
    """
    导出给 lsj 使用：
    - view=analysis: 含 analysis_text/channel/ts（推荐给分析）
    - view=raw: 接近原始采集字段
    - fmt=json|jsonl|csv
    """
    if from_ts is not None and to_ts is not None and from_ts > to_ts:
        raise HTTPException(
            status_code=400, detail="from_ts cannot be greater than to_ts"
        )

    return await export_io.prepare_export_response(
        transform=lambda rows, budget: _shape_export_rows(rows, view),
        from_ts=from_ts, to_ts=to_ts, limit_rows=limit_rows, fmt=fmt,
        columns=EXPORT_CSV_COLUMNS[view],
        metadata={"view": view, "from_ts": from_ts, "to_ts": to_ts},
        filename=f"lsj_export_{view}",
    )


@app.get("/export/lsj/training")
async def export_lsj_training(
    from_ts: Optional[int] = Query(None, ge=0, le=MAX_INGEST_TS),
    to_ts: Optional[int] = Query(None, ge=0, le=MAX_INGEST_TS),
    limit_rows: int = Query(5000, ge=1, le=200000),
    label_field: str = Query("channel", pattern="^(channel|source)$"),
    fmt: str = Query("json", pattern="^(json|jsonl|csv)$"),
    # 新增参数
    bare: bool = Query(True, description="json格式时仅返回数组，适配 data_cleaning.py"),
    exclude_internal: bool = Query(
        True, description="过滤本地域名与非公网、组播 IP 字面量；不解析 DNS"
    ),
    exclude_auth_pages: bool = Query(True, description="过滤登录/oauth/授权回调等页面"),
    exclude_search_pages: bool = Query(False, description="过滤搜索结果页"),
    dedup_by_input: bool = Query(True, description="按 input 去重"),
    max_input_len: int = Query(1000, ge=50, le=5000),
) -> Any:
    """
    导出为 data_cleaning.py 友好结构:
    [{input, label, ts, url, title, source}]
    """
    if from_ts is not None and to_ts is not None and from_ts > to_ts:
        raise HTTPException(
            status_code=400, detail="from_ts cannot be greater than to_ts"
        )

    return await export_io.prepare_export_response(
        transform=lambda rows, budget: _prepare_training_rows(
            rows, budget=budget, label_field=label_field,
            exclude_internal=exclude_internal, exclude_auth_pages=exclude_auth_pages,
            exclude_search_pages=exclude_search_pages, dedup_by_input=dedup_by_input,
            max_input_len=max_input_len),
        from_ts=from_ts, to_ts=to_ts, limit_rows=limit_rows, fmt=fmt,
        columns=TRAINING_CSV_COLUMNS,
        metadata=None if bare else {"label_field": label_field, "from_ts": from_ts, "to_ts": to_ts},
        filename="lsj_training",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
