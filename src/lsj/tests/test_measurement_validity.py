"""Execute real batch methods with small dependency stubs, not real model inference.

Optional ML packages and logger setup are replaced before module import. No model
constructor, download, personal browser data or project database is used.
"""
import importlib.util
import logging
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

ALGORITHMS = Path(__file__).resolve().parents[1] / "src" / "algorithms"


@pytest.fixture
def modules(monkeypatch):
    def stub(name, **attrs):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    stub("jieba")
    stub("yaml")
    stub("utils")
    stub("utils.logger", setup_logger=lambda *_args: logging.getLogger("measurement-fixture"))
    stub("sklearn")
    stub("sklearn.feature_extraction")
    stub("sklearn.feature_extraction.text", TfidfVectorizer=object)
    stub("sklearn.metrics")
    stub("sklearn.metrics.pairwise", linear_kernel=lambda *_args: None)
    stub("sklearn.cluster", KMeans=object, DBSCAN=object, MiniBatchKMeans=object)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    original_find = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == "cntext" else original_find(name, *a, **k))
    # Restore this legacy import-time environment mutation when the fixture exits.
    monkeypatch.setenv("HF_ENDPOINT", "synthetic-import-only")

    loaded = {}
    for filename in ("similarity", "sentiment"):
        name = f"_measurement_fixture_{filename}"
        spec = importlib.util.spec_from_file_location(name, ALGORITHMS / f"{filename}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded[filename] = module
    return loaded


class Vectors:
    """Tiny stand-in for only the sparse operations used by the real batch code."""
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def __getitem__(self, index):
        return Vectors(self.values[index])

    def getnnz(self, axis):
        return np.count_nonzero(self.values, axis=axis)

    def multiply(self, other):
        return Vectors(self.values * other.values)

    def sum(self, axis=None):
        return self.values.sum(axis=axis)


def similarity_analyzer(module):
    analyzer = module.SimilarityAnalyzer.__new__(module.SimilarityAnalyzer)
    analyzer.fit = lambda _texts: analyzer
    analyzer.vectorizer = types.SimpleNamespace(transform=lambda texts: Vectors([
        [1, 0] if text == "A" else [0, 1] if text == "B" else [0, 0] for text in texts
    ]))
    return analyzer


def test_real_batch_similarity_first_row_missing_and_zero_is_measured(modules):
    analyzer = similarity_analyzer(modules["similarity"])
    result = analyzer.batch_calculate_similarity(pd.DataFrame({"title": ["A", "A", "B", "B"]}))
    assert pd.isna(result["similarity_to_previous"].iloc[0])
    assert result["similarity_to_previous"].iloc[1:].tolist() == [1, 0, 1]
    assert result["similarity_valid"].tolist() == [False, True, True, True]


@pytest.mark.parametrize("text", ["", "stopwords-only"])
def test_zero_vectors_invalidate_both_neighboring_pairs(modules, text):
    analyzer = similarity_analyzer(modules["similarity"])
    result = analyzer.batch_calculate_similarity(pd.DataFrame({"title": ["A", text, "B", "B"]}))
    assert result["similarity_to_previous"].iloc[:3].isna().all()
    assert result["similarity_to_previous"].iloc[3] == 1
    assert result["similarity_valid"].tolist() == [False, False, False, True]


def test_empty_batch_content_and_empty_vocabulary_have_no_measurements(modules):
    analyzer = similarity_analyzer(modules["similarity"])
    for texts in ([None, "", " "], ["stop", "stop"]):
        def no_vocabulary(_texts):
            raise ValueError("empty vocabulary; perhaps the documents only contain stop words")
        analyzer.fit = no_vocabulary
        result = analyzer.batch_calculate_similarity(pd.DataFrame({"title": texts}))
        assert result["similarity_to_previous"].isna().all()
        assert not result["similarity_valid"].any()


def test_similarity_statistics_preserve_missing_values_and_measured_zero(modules):
    analyzer = modules["similarity"].SimilarityAnalyzer.__new__(modules["similarity"].SimilarityAnalyzer)
    missing = analyzer.get_similarity_statistics(
        pd.DataFrame({"similarity_to_previous": [np.nan, np.inf, -np.inf]})
    )["similarity_to_previous"]
    assert missing == {"mean": None, "std": None, "min": None, "max": None, "median": None}

    measured = analyzer.get_similarity_statistics(
        pd.DataFrame({"similarity": [0.0, 0.0, 0.0]})
    )["similarity"]
    assert measured == {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}

    single = analyzer.get_similarity_statistics(
        pd.DataFrame({"similarity": [0.0]})
    )["similarity"]
    assert single == {"mean": 0.0, "std": None, "min": 0.0, "max": 0.0, "median": 0.0}


def test_reference_mode_also_preserves_missing_and_actual_zero(modules):
    analyzer = similarity_analyzer(modules["similarity"])
    result = analyzer.batch_calculate_similarity(pd.DataFrame({"title": ["A", "", "B"], "ref": ["B", "A", "B"]}), reference_column="ref")
    assert result["similarity"].iloc[0] == 0
    assert pd.isna(result["similarity"].iloc[1])
    assert result["similarity"].iloc[2] == 1
    assert result["similarity_valid"].tolist() == [True, False, True]


@pytest.mark.parametrize("score", [2.0, -0.01, float("inf"), float("nan")])
@pytest.mark.parametrize("reference", [False, True])
def test_invalid_computed_similarity_is_not_clipped_into_a_measurement(modules, score, reference):
    analyzer = similarity_analyzer(modules["similarity"])
    analyzer.vectorizer.transform = lambda texts: Vectors([[1, 0] if text == "A" else [score, 0] for text in texts])
    if reference:
        result = analyzer.batch_calculate_similarity(pd.DataFrame({"title": ["A"], "ref": ["B"]}), reference_column="ref")
        assert result["similarity"].isna().all()
    else:
        result = analyzer.batch_calculate_similarity(pd.DataFrame({"title": ["A", "B"]}))
        assert result["similarity_to_previous"].isna().all()
    assert not result["similarity_valid"].any()


def test_float32_rounding_tolerance_is_not_confused_with_invalid_output(modules):
    analyzer = similarity_analyzer(modules["similarity"])
    analyzer.vectorizer.transform = lambda texts: Vectors([[1, 0] if text == "A" else [1.0000001, 0] for text in texts])
    result = analyzer.batch_calculate_similarity(pd.DataFrame({"title": ["A", "B"]}))
    assert result["similarity_to_previous"].iloc[1] == 1 and result["similarity_valid"].iloc[1]


def sentiment_analyzer(module):
    analyzer = module.SentimentAnalyzer.__new__(module.SentimentAnalyzer)
    analyzer.custom_dict = {"pos": ["good"], "neg": ["bad"]}
    analyzer._cntext_dict_cache = None
    analyzer.model = None
    analyzer.use_bert = False
    analyzer._segment_text = lambda text: text.split()
    analyzer._analyze_emotions_cntext = lambda _text: {}
    analyzer.cntext_backend = module.CntextSentimentBackend(analyzer)
    return analyzer


def test_real_sentiment_batch_never_converts_failed_prediction_to_neutral(modules):
    analyzer = sentiment_analyzer(modules["sentiment"])
    def predict(text, **_kwargs):
        if text == "error":
            raise RuntimeError("synthetic error")
        if text == "none":
            return None
        return {"sentiment": "Neutral", "polarity": 0.0, "confidence": 0.0}
    analyzer.predict = predict
    result = analyzer.batch_predict(pd.DataFrame({"title": ["", None, "none", "error", "measured neutral"]}), include_emotions=True)
    assert result["sentiment"].iloc[:4].isna().all()
    assert result["polarity"].iloc[:4].isna().all()
    assert result["sentiment"].iloc[4] == "Neutral" and result["polarity"].iloc[4] == 0
    assert result["sentiment_valid"].tolist() == [False, False, False, False, True]


@pytest.mark.parametrize("prediction", [
    {}, {"sentiment": "unsupported", "polarity": 0},
    {"sentiment": "Positive", "polarity": float("inf")},
    {"sentiment": "Positive", "polarity": 1.01},
    {"sentiment": "Neutral", "polarity": 0, "sentiment_valid": False},
])
def test_invalid_prediction_is_marked_missing(modules, prediction):
    analyzer = sentiment_analyzer(modules["sentiment"])
    analyzer.predict = lambda *_args, **_kwargs: prediction
    result = analyzer.batch_predict(pd.DataFrame({"title": ["synthetic"]}))
    assert pd.isna(result["sentiment"].iloc[0]) and not result["sentiment_valid"].iloc[0]


@pytest.mark.parametrize("raw", [
    None, {}, {"pos_num": float("nan"), "neg_num": 0}, {"pos_num": -1, "neg_num": 0},
    {"pos_num": 0.5, "neg_num": 0}, {"乐_num": True}, {"乐_num": float("inf")},
])
def test_backend_invalid_output_reaches_batch_as_missing(modules, raw):
    module = modules["sentiment"]
    module.ct = types.SimpleNamespace(sentiment=lambda *_args, **_kwargs: raw)
    analyzer = sentiment_analyzer(module)
    result = analyzer.batch_predict(pd.DataFrame({"title": ["synthetic"]}))
    assert result["sentiment"].isna().all() and not result["sentiment_valid"].any()


def test_backend_exception_and_real_zero_counts_remain_distinguishable(modules):
    module = modules["sentiment"]
    def error(*_args, **_kwargs):
        raise RuntimeError("synthetic backend failure")
    module.ct = types.SimpleNamespace(sentiment=error)
    analyzer = sentiment_analyzer(module)
    failed = analyzer.batch_predict(pd.DataFrame({"title": ["synthetic"]}))
    assert failed["sentiment"].isna().all()
    module.ct.sentiment = lambda *_args, **_kwargs: {"pos_num": 0, "neg_num": 0}
    measured = analyzer.batch_predict(pd.DataFrame({"title": ["synthetic"]}))
    assert measured["sentiment"].tolist() == ["Neutral"]
    assert measured["polarity"].tolist() == [0]
    assert measured["sentiment_valid"].tolist() == [True]


def test_explicit_score_construction_remains_valid_but_empty_backend_input_is_not(modules):
    module = modules["sentiment"]
    assert module.SentimentScore(pos=2, neg=0).valid is True
    assert module.SentimentScore(pos=0, neg=0).valid is True
    analyzer = sentiment_analyzer(module)
    assert analyzer.cntext_backend.analyze_score("").valid is False
