"""Real API/algorithm/logging branches with isolated synthetic dependency faults.

Optional ML dependencies are replaced, not the production catch/logging code.
These tests do not load models or prove which errors real ML packages produce.
"""
import importlib.machinery
import importlib.util
import logging
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.hyh import app as api, db


ALGORITHMS = Path(__file__).resolve().parents[2] / "lsj" / "src" / "algorithms"
ADMIN = "a" * 43
COLLECTOR = "c" * 43
BODY = "SYNTHETIC_PRIVATE_BODY"
TITLE = "SYNTHETIC_PRIVATE_TITLE_FALLBACK"
EXCEPTION_DETAIL = "SYNTHETIC_PRIVATE_EXCEPTION"


class Vectors:
    """Only the sparse operations needed to reach the real similarity branch."""
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    @property
    def shape(self):
        return self.values.shape

    def __getitem__(self, index):
        return Vectors(self.values[index])

    def getnnz(self, axis):
        return np.count_nonzero(self.values, axis=axis)

    def multiply(self, other):
        return Vectors(self.values * other.values)

    def sum(self, axis=None):
        return self.values.sum(axis=axis)


class SyntheticVectorizer:
    def __init__(self, **kwargs):
        self.tokenizer = kwargs["tokenizer"]

    def fit_transform(self, texts):
        return self.transform(texts)

    def transform(self, texts):
        return Vectors([[1.0] if self.tokenizer(text) else [0.0] for text in texts])


@pytest.fixture
def isolated_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "synthetic.sqlite3")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "unused-model-cache"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.syspath_prepend(str(ALGORITHMS))

    def stub(name, **attrs):
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def forbid_model_load(*_args, **_kwargs):
        raise AssertionError("No real model or download is permitted")

    class UnloadedModel:
        from_pretrained = staticmethod(forbid_model_load)

    jieba = stub("jieba", lcut=lambda text: text.split())
    stub("yaml")
    stub("torch", __version__="synthetic-only", no_grad=lambda: lambda function: function,
         device=lambda name: name, cuda=types.SimpleNamespace(is_available=lambda: False))
    stub("transformers", AutoModelForSequenceClassification=UnloadedModel, AutoTokenizer=UnloadedModel,
         BertTokenizer=UnloadedModel, BertForSequenceClassification=UnloadedModel)
    cntext = stub("cntext", __file__="synthetic-cntext", __version__="synthetic-only",
        read_yaml_dict=lambda _name: {"Dictionary": {"pos": ["synthetic"], "neg": []}},
        sentiment=lambda _text, **_kwargs: {"pos_num": 0, "neg_num": 0})
    stub("sklearn")
    stub("sklearn.feature_extraction")
    stub("sklearn.feature_extraction.text", TfidfVectorizer=SyntheticVectorizer)
    stub("sklearn.metrics")
    stub("sklearn.metrics.pairwise", linear_kernel=forbid_model_load)
    stub("sklearn.cluster", KMeans=UnloadedModel, DBSCAN=UnloadedModel, MiniBatchKMeans=UnloadedModel)

    # Use real setup_logger and real FileHandlers without reusing or touching
    # any logger/handler a previous test or the developer may have configured.
    names = ("classifier", "sentiment", "similarity", "evaluator", "markdown_builder")
    loggers = {name: logging.Logger(name) for name in names}
    original_get_logger = logging.getLogger
    monkeypatch.setattr(logging, "getLogger", lambda name=None:
        loggers[name] if name in loggers else original_get_logger(name))
    stub("utils", __path__=[str(ALGORITHMS / "utils")])

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)

    try:
        load("utils.logger", ALGORITHMS / "utils" / "logger.py")
        for name in ("classifier", "sentiment", "similarity", "markdown_builder", "evaluator"):
            load(name, ALGORITHMS / (name + ".py"))
        yield cntext, jieba, tmp_path / "local" / "InformationDietManager" / "logs"
        assert not (tmp_path / "unused-model-cache").exists()
    finally:
        for logger in loggers.values():
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()


@pytest.mark.parametrize("fault", ["sentiment_backend", "tokenizer"])
def test_full_analysis_faults_log_only_event_and_type_without_source_or_keys(
        isolated_pipeline, capsys, fault):
    cntext, jieba, log_dir = isolated_pipeline
    seen = []

    def fail_with_private_input(text, **_kwargs):
        seen.append(text)
        # A dependency can put arbitrary sensitive detail in an exception.
        # The synthetic capability value here is fault-injected, not passed
        # to the production model by the API.
        raise RuntimeError(f"{EXCEPTION_DETAIL}: {text}; synthetic-key={ADMIN}")

    if fault == "sentiment_backend":
        cntext.sentiment = fail_with_private_input
    else:
        jieba.lcut = fail_with_private_input

    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000),
                    headers={"Authorization": "Bearer " + ADMIN}) as client:
        pages = [("Synthetic body page", BODY), (TITLE, "")]
        pages.extend((f"Synthetic extra page {index}", BODY) for index in range(3))
        for index, (title, text) in enumerate(pages):
            response = client.post("/collect", json={
                "url": f"https://example.invalid/log-privacy/{index}", "title": title, "text": text,
                "ts": 1790208000000 + index, "source": "import",
            })
            assert response.status_code == 200

        # Ingestion itself can fill empty text from the title. Preserve a real
        # legacy empty-text row to exercise the API pipeline's own fallback.
        with db.get_conn() as connection:
            connection.execute("UPDATE items SET text = '' WHERE title = ?", (TITLE,))
            assert [tuple(row) for row in connection.execute("SELECT title,text FROM items ORDER BY id")] == pages

        response = client.post("/analyze/run_full?force=true")
        assert response.status_code == 422
        assert response.json()["status"] == api.JOB_FAILED
        assert response.json()["detail"] == (
            "Legacy scoring lacks complete valid measurements; use /dashboard/visualization for partial statistics.")
        assert set(seen) == {BODY, TITLE}
        with db.get_conn() as connection:
            jobs = [tuple(row) for row in connection.execute(
                "SELECT status,error,result_payload,metrics_json FROM analysis_jobs")]
            assert len(jobs) == 1 and jobs[0][0] == api.JOB_FAILED and jobs[0][2] is None
            assert [tuple(row) for row in connection.execute("SELECT title,text FROM items ORDER BY id")] == pages

    console = capsys.readouterr()
    logs = {path.name: path.read_text(encoding="utf-8") for path in log_dir.glob("*.log")}
    assert logs  # The actual production FileHandler must have run.
    if fault == "sentiment_backend":
        assert "情感分析失败 (RuntimeError)" in logs["sentiment.log"]
    else:
        assert "分词失败 (RuntimeError)" in logs["sentiment.log"]
        assert "分词失败 (RuntimeError)" in logs["similarity.log"]
    observed = response.text + str(jobs) + console.out + console.err + "".join(logs.values())
    for private in (BODY, TITLE, EXCEPTION_DETAIL, ADMIN, COLLECTOR):
        assert private not in observed
    assert "Traceback" not in console.err + "".join(logs.values())
