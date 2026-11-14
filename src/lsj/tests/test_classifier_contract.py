"""Classifier output contracts without loading an optional Transformer model."""
import importlib.util
import logging
import sys
import types
from pathlib import Path

import pandas as pd


def _load_classifier(monkeypatch):
    """Load classifier.py with tiny ML stubs so the test stays offline."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.device = lambda value: value

    def no_grad():
        def decorate(function):
            return function
        return decorate

    torch.no_grad = no_grad
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForSequenceClassification = object
    transformers.AutoTokenizer = object
    logger = types.ModuleType("utils.logger")
    logger.setup_logger = lambda *_args: logging.getLogger("classifier-contract")
    utils = types.ModuleType("utils")
    utils.logger = logger
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "utils", utils)
    monkeypatch.setitem(sys.modules, "utils.logger", logger)

    path = Path(__file__).resolve().parents[1] / "src" / "algorithms" / "classifier.py"
    spec = importlib.util.spec_from_file_location("classifier_contract_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch_classifier_normalizes_object_confidence_without_future_warning(monkeypatch):
    module = _load_classifier(monkeypatch)
    classifier = module.ContentClassifier.__new__(module.ContentClassifier)
    classifier.CATEGORY_OTHER = "Other"

    def prepared(frame):
        result = frame.copy()
        result["category"] = ["News", "Other", "Other"]
        result["confidence"] = ["0.7", None, "invalid"]
        return result

    classifier._batch_predict_by_rules_vectorized = prepared
    result = classifier.batch_predict(pd.DataFrame({"title": ["a", "b", "c"]}))

    assert result["confidence"].dtype.kind == "f"
    assert result["confidence"].tolist() == [0.7, 0.0, 0.0]
