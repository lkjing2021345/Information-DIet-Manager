"""Measured aggregation using synthetic frames; no model inference or database."""
import json

import numpy as np
import pandas as pd
import pytest

from src.hyh.visualization_metrics import build_visualization_metrics


def frame(similarities, *, dates=None, sentiments=None, categories=None, **extra):
    count = len(similarities)
    return pd.DataFrame({
        "ts": dates if dates is not None else [1790208000000 + i for i in range(count)],
        "title": ["synthetic"] * count, "category": categories or ["tools"] * count,
        "sentiment": sentiments if sentiments is not None else ["Positive"] * count,
        "polarity": [1.0] * count, "similarity": similarities, **extra,
    })


def test_first_row_has_no_comparison_but_keeps_its_other_measurements():
    result = build_visualization_metrics(frame([0.0, 1, 1, 1, 1]))
    day = result["global"]["time_series"][0]
    assert day["count"] == day["sentiment_count"] == 5
    assert day["comparison_count"] == 4
    assert day["repeat_ratio"] == day["avg_similarity"] == 1
    assert result["category_counts"] == {"tools": 5}
    assert result["processed_count"] == 5
    assert sum(result["global"]["similarity_histogram"].values()) == 4


def test_cross_day_and_category_pairs_belong_to_the_later_original_record():
    result = build_visualization_metrics(frame(
        [0, 1, 0], dates=[1790207999999, 1790208000000, 1790208000001],
        categories=["tools", "news", "tools"],
    ))
    first, second = result["global"]["time_series"]
    assert first["date"] == "2026-09-23" and first["count"] == 1
    assert first["repeat_ratio"] is None and first["avg_similarity"] is None
    assert first["comparison_count"] == 0 and first["positive_ratio"] == 1
    assert second["repeat_ratio"] == 0.5 and second["comparison_count"] == 2
    assert result["categories"]["news"]["time_series"][0]["repeat_ratio"] == 1
    # The tools record keeps comparison to the preceding news record. No pairing
    # is recomputed after category filtering, and a measured zero remains zero.
    assert result["categories"]["tools"]["time_series"][1]["repeat_ratio"] == 0


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), float("-inf"), -0.1, 1.1, "0.9", True])
def test_invalid_similarity_is_excluded_without_changing_other_counts(bad):
    result = build_visualization_metrics(frame([0, bad, 0, 0.85]))
    day = result["global"]["time_series"][0]
    assert day["count"] == day["sentiment_count"] == 4
    assert day["comparison_count"] == 2 and day["repeat_ratio"] == 0.5
    json.dumps(result, allow_nan=False)


def test_missing_sentiments_use_an_independent_denominator():
    result = build_visualization_metrics(frame(
        [None, 0, 1, 0, 1], sentiments=[" POSITIVE ", "Neutral", None, "nonsense", float("nan")],
    ))
    day = result["global"]["time_series"][0]
    assert day["count"] == 5 and day["sentiment_count"] == day["polarity_count"] == 2
    assert (day["positive_ratio"], day["neutral_ratio"], day["negative_ratio"]) == (0.5, 0.5, 0)
    assert result["category_counts"] == {"tools": 5}
    assert result["global"]["sentiment_distribution"] == {"neutral": 0.5, "positive": 0.5}


def test_all_missing_is_null_but_not_an_empty_record_set():
    result = build_visualization_metrics(frame([None, None], sentiments=[None, None]))
    day = result["global"]["time_series"][0]
    for metric in ("repeat_ratio", "avg_similarity", "avg_polarity", "positive_ratio", "neutral_ratio", "negative_ratio"):
        assert day[metric] is None
    assert day["count"] == 2 and day["comparison_count"] == day["sentiment_count"] == 0
    assert result["category_counts"] == {"tools": 2}
    json.dumps(result, allow_nan=False)


def test_explicit_validity_flags_override_fabricated_legacy_values():
    result = build_visualization_metrics(frame(
        [0, 0.9, 0.9], similarity_valid=[False, False, True], sentiment_valid=[False, False, True],
    ))
    day = result["global"]["time_series"][0]
    assert day["comparison_count"] == day["sentiment_count"] == 1
    assert day["repeat_ratio"] == day["positive_ratio"] == 1


@pytest.mark.parametrize("flag", [False, None, float("nan"), "true", 1])
def test_present_validity_flags_must_be_true_booleans(flag):
    result = build_visualization_metrics(frame(
        [None, 0.9], similarity_valid=pd.Series([flag, flag], dtype=object),
        sentiment_valid=pd.Series([flag, flag], dtype=object),
    ))
    assert result["coverage"]["comparison_count"] == result["coverage"]["sentiment_count"] == 0


def test_pandas_numpy_boolean_columns_convert_to_real_validity_flags():
    result = build_visualization_metrics(frame(
        [None, 0], similarity_valid=np.array([False, True], dtype=np.bool_),
        sentiment_valid=np.array([True, True], dtype=np.bool_),
    ))
    assert result["coverage"]["comparison_count"] == 1
    assert result["coverage"]["sentiment_count"] == 2


def test_epoch_milliseconds_are_not_guessed_as_seconds_and_full_ingest_range_is_supported():
    result = build_visualization_metrics(frame([None, 0, 1], dates=[0, 86_400_000, 253402300799999]))
    assert [row["date"] for row in result["global"]["time_series"]] == ["1970-01-01", "1970-01-02", "9999-12-31"]


def test_timestamp_offsets_normalize_to_utc_and_invalid_ts_does_not_fall_back():
    data = frame([0, 0, 1]).drop(columns=["ts"])
    data["timestamp"] = ["2026-09-24T01:00:00+08:00", "2026-09-24T00:00:00Z", pd.NaT]
    result = build_visualization_metrics(data)
    assert [row["date"] for row in result["global"]["time_series"]] == ["2026-09-23", "2026-09-24"]
    assert result["coverage"]["record_count"] == 3 and result["coverage"]["timestamp_count"] == 2
    data["ts"] = [None, None, None]
    assert build_visualization_metrics(data)["global"]["time_series"] == []


def test_invalid_polarity_and_category_do_not_remove_other_valid_observations():
    data = frame([0, 1, 0], categories=["tools", None, "invalid"], polarity=[float("inf"), -1.1, 0.0])
    result = build_visualization_metrics(data)
    day = result["global"]["time_series"][0]
    assert day["count"] == day["sentiment_count"] == 3
    assert day["polarity_count"] == 1 and day["avg_polarity"] == 0
    assert result["category_counts"] == {"tools": 1}
    assert result["coverage"]["category_count"] == 1


def test_empty_input_has_empty_statistics_and_zero_coverage():
    result = build_visualization_metrics(pd.DataFrame())
    assert result["global"]["time_series"] == []
    assert result["category_counts"] == {} and result["processed_count"] == 0
    assert not any(result["coverage"].values())
