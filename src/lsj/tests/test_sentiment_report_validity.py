"""Real report aggregation on synthetic frames; no model construction/inference."""
import json

import pandas as pd
import pytest

from test_measurement_validity import modules, sentiment_analyzer  # noqa: F401


def report(modules, frame):
    analyzer = sentiment_analyzer(modules["sentiment"])
    original = frame.copy(deep=True)
    result = analyzer.generate_sentiment_report(frame)
    pd.testing.assert_frame_equal(frame, original)
    json.dumps(result, allow_nan=False)
    return result


def test_actual_missing_batch_does_not_become_neutral(modules):
    analyzer = sentiment_analyzer(modules["sentiment"])
    batch = analyzer.batch_predict(pd.DataFrame({"title": [None, "", " "]}))
    result = report(modules, batch)
    assert result["total_records"] == 3
    assert result["valid_records"] == result["confidence_records"] == 0
    assert result["missing_records"] == 3
    assert result["sentiment_distribution"] == {"counts": {}, "percentages": {}}
    assert all(value is None for value in result["polarity_statistics"].values())
    assert all(value is None for value in result["confidence_statistics"].values())
    assert result["overall_summary"] == {
        "dominant_sentiment": "Unknown", "overall_polarity": "Unknown",
        "avg_confidence": None, "high_confidence_ratio": None,
    }
    assert all(value is None for value in result["word_statistics"].values())
    assert result["emotion_distribution"] is None


def test_single_measured_zero_keeps_zero_and_missing_sample_deviation(modules):
    result = report(modules, pd.DataFrame({
        "sentiment": ["Neutral"], "polarity": [0], "confidence": [0],
        "pos_count": [0], "neg_count": [0], "emotions": [{"calm": 0}],
    }))
    assert result["valid_records"] == result["confidence_records"] == 1
    assert result["polarity_statistics"] == {
        "mean": 0, "std": None, "min": 0, "max": 0, "median": 0, "q25": 0, "q75": 0,
    }
    assert result["confidence_statistics"]["std"] is None
    assert result["overall_summary"]["overall_polarity"] == "Neutral"
    assert result["overall_summary"]["high_confidence_ratio"] == 0
    assert result["word_statistics"]["total_positive_words"] == 0
    assert result["emotion_distribution"] == [{"emotion": "calm", "count": 0}]


@pytest.mark.parametrize("bad", [None, "bad", float("nan"), float("inf"), -float("inf"), True, -1.1, 1.1])
def test_invalid_polarity_excludes_prediction_instead_of_inventing_summary(modules, bad):
    result = report(modules, pd.DataFrame({
        "sentiment": ["Positive"], "polarity": [bad], "confidence": [1.0],
    }))
    assert result["valid_records"] == 0
    assert result["overall_summary"]["overall_polarity"] == "Unknown"
    assert result["confidence_statistics"]["mean"] is None


@pytest.mark.parametrize("bad", [None, "bad", float("nan"), float("inf"), -float("inf"), True, -0.1, 1.1])
def test_invalid_confidence_does_not_erase_measured_polarity(modules, bad):
    result = report(modules, pd.DataFrame({
        "sentiment": ["Positive"], "polarity": [0.5], "confidence": [bad],
    }))
    assert result["valid_records"] == 1 and result["confidence_records"] == 0
    assert result["overall_summary"]["overall_polarity"] == "Positive"
    assert result["overall_summary"]["high_confidence_ratio"] is None


def test_mixed_coverage_uses_explicit_measured_denominators(modules):
    result = report(modules, pd.DataFrame({
        "sentiment": ["positive", " Negative ", "Neutral", "Neutral", "unsupported"],
        "polarity": ["0.5", "-0.5", 0, 0, 0],
        "confidence": ["0.9", "0.5", None, 1, 1],
        "sentiment_valid": [True, True, True, False, True],
    }))
    assert result["total_records"] == 5 and result["valid_records"] == 3
    assert result["missing_records"] == 2 and result["confidence_records"] == 2
    assert result["sentiment_distribution"]["counts"] == {"Positive": 1, "Negative": 1, "Neutral": 1}
    assert result["polarity_statistics"]["mean"] == 0
    assert result["confidence_statistics"]["mean"] == 0.7
    assert result["overall_summary"]["high_confidence_ratio"] == 50


@pytest.mark.parametrize("flag", [False, None, 0, 1, "true"])
def test_only_explicit_true_validity_flag_accepts_a_prediction(modules, flag):
    result = report(modules, pd.DataFrame({
        "sentiment": ["Positive"], "polarity": [1], "confidence": [1], "sentiment_valid": [flag],
    }))
    assert result["valid_records"] == 0


def test_optional_counts_exclude_invalid_rows_and_preserve_known_zero(modules):
    result = report(modules, pd.DataFrame({
        "sentiment": ["Positive", "Neutral", "Negative"], "polarity": [1, 0, -1],
        "confidence": [1, 0.5, 1], "sentiment_valid": [True, True, False],
        "pos_count": [None, 0, 100], "neg_count": [float("inf"), -1, 100],
        "emotions": [{"joy": 2, "invalid": float("inf")}, {"joy": 0}, {"joy": 100}],
    }))
    assert result["word_statistics"] == {
        "total_positive_words": 0, "total_negative_words": None,
        "avg_positive_words": 0, "avg_negative_words": None,
    }
    assert result["emotion_distribution"] == [{"emotion": "joy", "count": 2}]


def test_complete_measured_report_keeps_statistics(modules):
    result = report(modules, pd.DataFrame({
        "sentiment": ["Positive", "Negative"], "polarity": [1, -1], "confidence": [0.8, 0.4],
        "pos_count": [2, 0], "neg_count": [0, 4],
    }))
    assert result["polarity_statistics"] == {
        "mean": 0, "std": 1.4142, "min": -1, "max": 1, "median": 0, "q25": -0.5, "q75": 0.5,
    }
    assert result["confidence_statistics"]["mean"] == 0.6
    assert result["word_statistics"]["total_negative_words"] == 4
    assert result["overall_summary"]["high_confidence_ratio"] == 50


@pytest.mark.parametrize("labels", [["Positive", "Neutral", "bad"], ["bad", None, "bad"]])
def test_categorical_labels_do_not_report_unobserved_or_invalid_categories(modules, labels):
    result = report(modules, pd.DataFrame({
        "sentiment": pd.Categorical(labels, categories=["Positive", "Neutral", "Negative", "bad"]),
        "polarity": [0.5, 0, -0.5], "confidence": [0.8, 0.5, 0.9],
    }))
    expected = {"Positive": 1, "Neutral": 1} if labels[0] == "Positive" else {}
    assert result["sentiment_distribution"]["counts"] == expected
    if not expected:
        assert result["overall_summary"]["overall_polarity"] == "Unknown"
        assert result["overall_summary"]["dominant_sentiment"] == "Unknown"


def test_categorical_validity_flags_preserve_missing_and_false(modules):
    result = report(modules, pd.DataFrame({
        "sentiment": ["Positive"] * 3, "polarity": [0.5] * 3, "confidence": [0.8] * 3,
        "sentiment_valid": pd.Categorical([True, False, None]),
    }))
    assert result["valid_records"] == 1 and result["missing_records"] == 2
    assert result["sentiment_distribution"]["counts"] == {"Positive": 1}


def test_empty_and_missing_columns_keep_existing_contract(modules):
    assert report(modules, pd.DataFrame(columns=["sentiment", "polarity", "confidence"])) == {"error": "Empty DataFrame"}
    with pytest.raises(ValueError, match="Missing required columns"):
        report(modules, pd.DataFrame({"sentiment": ["Neutral"]}))
