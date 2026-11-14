"""Structural reuse contract for current full-analysis caches, not model scores."""
from __future__ import annotations

import math
from typing import Any

from .models import MAX_INGEST_TS
from .visualization_metrics import finite_metric


_REQUIRED = frozenset({
    "day", "statistics_scope", "statistics_version", "analysis_status", "repeat_metric", "window",
    "total_count", "channel_counts", "category_counts", "sentiment_counts", "comparison_count",
    "sentiment_count", "polarity_count", "repeat_ratio", "negative_ratio", "avg_sentiment",
    "quick_evaluation", "full_report", "pipeline_warning", "generated_at", "cached",
})
_SENTIMENT_LABELS = frozenset({"Positive", "Neutral", "Negative"})


def _count(value: Any, maximum: int) -> bool:
    return type(value) is int and 0 <= value <= maximum


def _counts(value: Any, total: int) -> bool:
    return (isinstance(value, dict)
            and all(isinstance(key, str) and _count(count, total) for key, count in value.items())
            and sum(value.values()) == total)


def valid_full_analysis_cache(
    payload: Any, *, day: str, from_ts: int | None, to_ts: int | None,
    limit_rows: int, input_count: int,
) -> bool:
    """Check already-safe JSON against the current snapshot before reuse.

    The caller first applies the shared finite/depth/Unicode JSON validation.
    Reports retain their own evolving structure: this only verifies their outer
    object type. A rejected cache remains in history and is normally recomputed.
    """
    if not isinstance(payload, dict) or not _REQUIRED.issubset(payload):
        return False
    if (payload["statistics_scope"] != "analysis_window"
            or type(payload["statistics_version"]) is not int or payload["statistics_version"] != 2
            or payload["day"] != day
            or payload["analysis_status"] != ("ready" if input_count else "empty")
            or payload["repeat_metric"] != "legacy_adjacent_text_similarity_threshold_fraction"
            or payload["pipeline_warning"] is not None
            or type(payload["cached"]) is not bool
            or not _count(payload["generated_at"], MAX_INGEST_TS)
            or not _count(payload["total_count"], input_count)
            or payload["total_count"] != input_count):
        return False
    window = payload["window"]
    expected = {"from_ts": from_ts, "to_ts": to_ts, "limit_rows": limit_rows, "input_count": input_count}
    if not isinstance(window, dict) or any(
        key not in window or type(window[key]) is not type(value) or window[key] != value
        for key, value in expected.items()
    ):
        return False
    if not all(_counts(payload[key], input_count) for key in ("channel_counts", "category_counts")):
        return False
    if (not _count(payload["comparison_count"], max(input_count - 1, 0))
            or not _count(payload["sentiment_count"], input_count)
            or not _count(payload["polarity_count"], payload["sentiment_count"])):
        return False
    sentiments = payload["sentiment_counts"]
    if not _counts(sentiments, payload["sentiment_count"]) or set(sentiments) - _SENTIMENT_LABELS:
        return False
    for metric, count_key, lower in (
        ("repeat_ratio", "comparison_count", 0),
        ("negative_ratio", "sentiment_count", 0),
        ("avg_sentiment", "polarity_count", -1),
    ):
        if payload[count_key] == 0:
            if payload[metric] is not None:
                return False
        elif finite_metric(payload[metric], lower, 1) is None:
            return False
    if payload["sentiment_count"] and not math.isclose(
        payload["negative_ratio"], sentiments.get("Negative", 0) / payload["sentiment_count"],
        rel_tol=1e-9, abs_tol=1e-12,
    ):
        return False
    return all(isinstance(payload[key], dict) or (input_count == 0 and payload[key] is None)
               for key in ("quick_evaluation", "full_report"))
