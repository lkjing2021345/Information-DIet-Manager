"""Reuse only complete basic-statistics caches for the current library state."""
from __future__ import annotations

from typing import Any

from .models import MAX_INGEST_TS
from .visualization_metrics import finite_metric


_REQUIRED = frozenset({
    "day", "total_count", "channel_counts", "repeat_ratio", "repeat_sample_count", "repeat_metric",
    "negative_ratio", "avg_sentiment", "generated_at", "cached", "embeddings_backfilled",
    "statistics_scope", "statistics_version", "data_version",
})
_CHANNELS = frozenset({"ent", "edu", "news", "soc", "other"})


def _count(value: Any, maximum: int) -> bool:
    return type(value) is int and 0 <= value <= maximum


def valid_global_statistics_cache(payload: Any, *, day: str, data_version: str, total_count: int) -> bool:
    """Validate already-safe JSON without recomputing SQL metrics or history.

    This is an outer contract, not a guarantee against self-consistent tampering.
    The caller first applies the shared finite/depth/Unicode JSON checks.
    """
    if not isinstance(payload, dict) or not _REQUIRED.issubset(payload):
        return False
    if (payload["statistics_scope"] != "all_saved_pages"
            or type(payload["statistics_version"]) is not int or payload["statistics_version"] != 1
            or payload["day"] != day or payload["data_version"] != data_version
            or not _count(payload["total_count"], total_count) or payload["total_count"] != total_count
            or payload["repeat_metric"] != "duplicate_normalized_content_hash_fraction"
            or payload["negative_ratio"] is not None or payload["avg_sentiment"] is not None
            or type(payload["cached"]) is not bool
            or not _count(payload["generated_at"], MAX_INGEST_TS)
            or not _count(payload["embeddings_backfilled"], total_count)
            or not _count(payload["repeat_sample_count"], total_count)):
        return False
    channels = payload["channel_counts"]
    if (not isinstance(channels, dict) or set(channels) != _CHANNELS
            or any(not _count(value, total_count) for value in channels.values())
            or sum(channels.values()) != total_count):
        return False
    samples = payload["repeat_sample_count"]
    if samples == 0:
        return payload["repeat_ratio"] is None
    # At least one distinct non-empty hash exists when there are measurements;
    # even an entirely repeated sample therefore cannot have a ratio of 1.
    return finite_metric(payload["repeat_ratio"], 0, (samples - 1) / samples) is not None
