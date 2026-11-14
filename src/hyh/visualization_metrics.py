"""Measured dashboard statistics, independent of the experimental scoring engine.

Rows arrive in (epoch milliseconds, item id) order. A similarity belongs to its
later row, even across a UTC date or category boundary. Filtering for a category
never creates new pairs. Missing measurements do not become zero or remove a
record from unrelated statistics.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

SENTIMENTS = ("positive", "neutral", "negative")
CATEGORY_ALIASES = {
    "entertainment": "ent", "learning": "edu", "news": "news",
    "social": "soc", "shopping": "shopping", "tools": "tools", "other": "other",
}


def finite_metric(value: Any, lower: float, upper: float) -> float | None:
    """Accept finite numeric measurements in range, without clipping bad input."""
    if isinstance(value, (bool, str, bytes)) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and lower <= number <= upper else None


def _utc_time(row: Mapping[str, Any]) -> datetime | None:
    try:
        if "ts" in row:
            # IngestItem explicitly uses epoch milliseconds, including early dates.
            raw = row["ts"]
            number = finite_metric(raw, 0, 253402300799999)
            if number is None or number != int(number):
                return None
            return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=int(number))
        raw = row.get("timestamp")
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00")) if isinstance(raw, str) else raw
        if not isinstance(parsed, datetime):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        parsed = parsed.astimezone(timezone.utc)
        # Also rejects pandas.NaT, whose date/hour accessors are not measurements.
        return parsed if isinstance(parsed.year, int) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _label(value: Any, allowed: Mapping[str, Any] | tuple[str, ...]) -> str | None:
    if not isinstance(value, str):
        return None
    label = value.strip().lower()
    return label if label in allowed else None


def _mean(values: list[float]) -> float | None:
    return round(math.fsum(values) / len(values), 6) if values else None


def _daily_rows(rows: list[dict[str, Any]], repeat_threshold: float) -> list[dict[str, Any]]:
    dates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["time"] is not None:
            dates[row["time"].date().isoformat()].append(row)
    result = []
    for date, group in sorted(dates.items()):
        comparisons = [r["similarity"] for r in group if r["similarity"] is not None]
        polarities = [r["polarity"] for r in group if r["polarity"] is not None]
        sentiments = Counter(r["sentiment"] for r in group if r["sentiment"] is not None)
        sentiment_count = sum(sentiments.values())
        result.append({
            "date": date, "count": len(group),
            "comparison_count": len(comparisons), "sentiment_count": sentiment_count,
            "polarity_count": len(polarities),
            "avg_polarity": _mean(polarities), "avg_similarity": _mean(comparisons),
            "repeat_ratio": _mean([float(value >= repeat_threshold) for value in comparisons]),
            **{f"{label}_ratio": round(sentiments[label] / sentiment_count, 6)
               if sentiment_count else None for label in SENTIMENTS},
        })
    return result


def build_visualization_metrics(
    df: Any, *, repeat_threshold: float = 0.85,
    category_aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Aggregate a pipeline DataFrame; never invoke models or legacy score logic.

    Optional ``similarity_valid`` / ``sentiment_valid`` flags can invalidate a
    measurement but cannot make a bad value valid. The first input row never has
    a predecessor in this window. ``ts`` takes precedence over ``timestamp``.
    """
    if finite_metric(repeat_threshold, 0, 1) is None:
        raise ValueError("repeat_threshold must be finite and between 0 and 1")
    aliases = CATEGORY_ALIASES if category_aliases is None else category_aliases
    rows = []
    for index, source in enumerate(df.to_dict("records")):
        similarity = finite_metric(source.get("similarity", source.get("similarity_to_previous")), 0, 1)
        if index == 0 or ("similarity_valid" in source and source["similarity_valid"] is not True):
            similarity = None
        sentiment = _label(source.get("sentiment"), SENTIMENTS)
        if "sentiment_valid" in source and source["sentiment_valid"] is not True:
            sentiment = None
        rows.append({
            "time": _utc_time(source),
            "category": _label(source.get("category"), aliases),
            "sentiment": sentiment, "similarity": similarity,
            "polarity": finite_metric(source.get("polarity"), -1, 1)
            if sentiment is not None else None,
        })
    category_counts = Counter(row["category"] for row in rows if row["category"] is not None)
    sentiment_counts = Counter(row["sentiment"] for row in rows if row["sentiment"] is not None)
    comparisons = [row["similarity"] for row in rows if row["similarity"] is not None]
    histogram = Counter()
    bins = ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0")
    for similarity in comparisons:
        histogram[bins[min(int(similarity * 5), 4)]] += 1
    category_total, sentiment_total = sum(category_counts.values()), sum(sentiment_counts.values())
    return {
        "processed_count": len(rows),
        "coverage": {
            "record_count": len(rows), "timestamp_count": sum(row["time"] is not None for row in rows),
            "category_count": category_total, "comparison_count": len(comparisons),
            "sentiment_count": sentiment_total,
        },
        "category_counts": dict(sorted(category_counts.items())),
        "global": {
            "time_series": _daily_rows(rows, repeat_threshold),
            "category_distribution": {key: count / category_total for key, count in sorted(category_counts.items())},
            "sentiment_distribution": {key: count / sentiment_total for key, count in sorted(sentiment_counts.items())},
            "similarity_histogram": {key: histogram[key] for key in bins} if comparisons else {},
            "hourly_distribution": dict(sorted(Counter(row["time"].hour for row in rows if row["time"] is not None).items())),
        },
        "categories": {
            key: {"alias": aliases[key], "label": key,
                  "time_series": _daily_rows([row for row in rows if row["category"] == key], repeat_threshold)}
            for key in sorted(category_counts)
        },
    }
