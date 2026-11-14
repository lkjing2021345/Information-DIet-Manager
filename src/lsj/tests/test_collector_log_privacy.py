"""Synthetic response/exception privacy through real collector logging branches.

No HTTP client, model, personal configuration, or browser database is opened.
"""
import asyncio
from dataclasses import asdict
import importlib.util
import io
import json
import logging
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest

UTILS = Path(__file__).resolve().parents[1] / "src" / "algorithms" / "utils"
TEXT = "SYNTHETIC_PRIVATE_GENERATED_TEXT"
KEY = "SYNTHETIC_API_KEY_01234567890123456789"
DETAIL = "SYNTHETIC_PROVIDER_ERROR_DETAIL"


@pytest.fixture(params=["classifier", "sentiment"])
def context(request, monkeypatch, tmp_path):
    kind = request.param
    name = "_collector_privacy_" + kind
    spec = importlib.util.spec_from_file_location(name, UTILS / f"{kind}_data_collector.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    stream = io.StringIO()
    logger = logging.Logger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(stream))
    path = tmp_path / "synthetic.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    logger.addHandler(handler)
    collector = object.__new__(getattr(module, kind.title() + "DataCollector"))
    collector.logger = logger
    collector.temperature = 0.5
    collector.max_tokens = 100
    collector.min_confidence = 0
    collector.cfg = SimpleNamespace(retry=SimpleNamespace(task_timeout=2))
    collector._build_prompt = lambda *_args: "Synthetic prompt"
    fields = dict(label="News" if kind == "classifier" else "平静", scene="x", sentence_type="x",
                  length_type="x", persona="x", intensity="x", event_type="x", tone_style="x")

    def logs():
        handler.flush()
        console = stream.getvalue()
        assert path.read_text(encoding="utf-8") == console
        for private in (TEXT, KEY, DETAIL, "Traceback"):
            assert private not in console
        return [json.loads(line) for line in console.splitlines()]

    yield SimpleNamespace(module=module, collector=collector, fields=fields,
                          text_field="input" if kind == "classifier" else "text", logs=logs, logger=logger)
    for entry in list(logger.handlers):
        logger.removeHandler(entry)
        entry.close()


@pytest.mark.parametrize("retryable", [False, True])
def test_model_pool_failure_preserves_cause_but_not_exception_text(context, monkeypatch, retryable):
    module = context.module
    primary = (module.httpx.NetworkError if retryable else ValueError)(f"{DETAIL} {TEXT} {KEY}")
    calls = []

    async def fail(*args):
        calls.append(args)
        raise primary

    async def no_sleep(_seconds):
        pass

    monkeypatch.setattr(module.asyncio, "sleep", no_sleep)
    pool = object.__new__(module.ModelPool)
    cfg = SimpleNamespace(name="synthetic-model", max_retries=1)
    pool.logger = context.logger
    pool.cfgs = {cfg.name: cfg}
    pool._pick_model = lambda: cfg
    pool.clients = {cfg.name: SimpleNamespace(generate=fail)}
    pool.semaphores = {cfg.name: asyncio.Semaphore(1)}
    pool.breakers = {cfg.name: module.CircuitBreaker()}
    pool.stats = {cfg.name: module.ModelStats()}

    async def run():
        with pytest.raises(RuntimeError) as raised:
            await pool.generate("Synthetic prompt", 0.5, 100)
        assert str(raised.value) == "All models failed."
        assert raised.value.__cause__ is primary
        assert len(calls) == 3
        events = context.logs()
        if retryable:
            assert len(events) == 3
            assert all(event["event"] == "model_request_retry" and event["error_type"] == "NetworkError"
                       and event["model"] == "synthetic-model" for event in events)
        else:
            assert events == []

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["exception", "invalid_response", "label_mismatch"])
def test_generation_failure_logs_diagnostics_without_response_content(context, failure):
    if failure == "invalid_response":
        response = f"{TEXT} {KEY} invalid JSON"
    else:
        response = json.dumps({context.text_field: TEXT, "label": KEY})

    async def generate(*_args):
        if failure == "exception":
            raise RuntimeError(f"{DETAIL} {TEXT} {KEY}")
        return response, "synthetic-model"

    context.collector.pool = SimpleNamespace(generate=generate)

    async def run():
        assert await context.collector._generate_one(context.fields) is None
        events = context.logs()
        assert len(events) == 1
        event = events[0]
        if failure == "exception":
            assert event["event"] == "generate_failed" and event["error_type"] == "RuntimeError"
        elif failure == "invalid_response":
            assert event["event"] == "parse_failed" and event["response_length"] == len(response)
        else:
            assert event["event"] == "label_mismatch"
            assert event["expected"] == context.fields["label"]
            assert event["text_length"] == len(TEXT)
            assert event["received_label_length"] == len(KEY)
        assert not {"reason", "raw_preview", "text_preview", "actual"}.intersection(event)

    asyncio.run(run())


def test_successful_generation_retains_requested_text_and_label(context):
    response = json.dumps({context.text_field: TEXT, "label": context.fields["label"]})

    async def generate(*_args):
        return response, "synthetic-model"

    context.collector.pool = SimpleNamespace(generate=generate)

    async def run():
        record = await context.collector._generate_one(context.fields)
        assert getattr(record, context.text_field) == TEXT
        assert record.label == context.fields["label"]
        assert record.model == "synthetic-model"
        assert context.logs() == []

    asyncio.run(run())


@pytest.mark.parametrize("context", ["classifier"], indirect=True)
@pytest.mark.parametrize("outcome", ["exception", "invalid_response", "rejected", "success"])
def test_relabel_logs_exclude_source_and_untrusted_response(context, outcome):
    collector = context.collector
    collector.enable_relabel_check = True
    collector._build_relabel_prompt = lambda text: "Synthetic check: " + text
    record = context.module.Record(entry_id="synthetic-relabel", input=TEXT, label="News")
    if outcome == "invalid_response":
        response = f"{TEXT} {KEY} invalid JSON"
    else:
        response = json.dumps({"label": "News" if outcome == "success" else KEY, "confidence": 0.95})
    prompts = []

    async def generate(prompt, *_args):
        prompts.append(prompt)
        if outcome == "exception":
            raise RuntimeError(f"{DETAIL} {KEY}")
        return response, "synthetic-model"

    collector.pool = SimpleNamespace(generate=generate)

    async def run():
        assert await collector._relabel_check(record) is (outcome == "success")
        assert prompts == ["Synthetic check: " + TEXT]
        assert record.input == TEXT and record.label == "News"
        events = context.logs()
        if outcome == "success":
            assert events == []
            return
        assert len(events) == 1
        event = events[0]
        if outcome == "exception":
            assert event["event"] == "relabel_failed" and event["error_type"] == "RuntimeError"
            assert event["text_length"] == len(TEXT)
        elif outcome == "invalid_response":
            assert event["event"] == "relabel_parse_failed" and event["response_length"] == len(response)
        else:
            assert event["event"] == "relabel_rejected"
            assert event["original_label"] == "News"
            assert event["received_label_length"] == len(KEY) and event["text_length"] == len(TEXT)
            assert event["confidence"] == 0.95 and event["model"] == "synthetic-model"
        assert not {"reason", "raw_preview", "input_preview", "relabel"}.intersection(event)

    asyncio.run(run())


def test_progress_load_failure_hides_unknown_field_and_preserves_fallback(context, tmp_path):
    path = tmp_path / "synthetic-progress.json"
    invalid = json.dumps({KEY: DETAIL})
    path.write_text(invalid, encoding="utf-8")
    tracker = context.module.ProgressTracker(str(path), context.logger)
    assert tracker.load() is None
    assert path.read_text(encoding="utf-8") == invalid
    events = context.logs()
    assert events[-1]["event"] == "progress_load_failed"
    assert events[-1]["error_type"] == "TypeError"
    assert events[-1]["progress_path"] == str(path)

    state = context.module.ProgressState(output_path="synthetic-output.json", target_count=2,
                                         existing_count=1, generated_count=3)
    valid = json.dumps(asdict(state))
    path.write_text(valid, encoding="utf-8")
    restored = tracker.load()
    assert restored == state and path.read_text(encoding="utf-8") == valid
    assert context.logs()[-1]["event"] == "progress_loaded"


def test_sbert_initialization_failure_logs_type_and_keeps_success_path(context, monkeypatch):
    collector = context.collector
    collector.enable_semantic_dedup = True
    stub = types.ModuleType("sentence_transformers")

    def fail(_name):
        raise RuntimeError(KEY)

    stub.SentenceTransformer = fail
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub)
    assert collector._try_load_sbert() is None
    assert collector.embed_model is None
    event = context.logs()[-1]
    assert event["event"] == "sbert_unavailable" and event["error_type"] == "RuntimeError"

    model = object()
    stub.SentenceTransformer = lambda _name: model
    assert collector._try_load_sbert() is None
    assert collector.embed_model is model
    assert context.logs()[-1]["event"] == "sbert_loaded"


@pytest.mark.parametrize("context", ["sentiment"], indirect=True)
def test_ppl_initialization_failure_logs_type_and_keeps_success_path(context, monkeypatch):
    collector = context.collector
    stub = types.ModuleType("transformers")
    torch = types.ModuleType("torch")

    def fail(_name):
        raise RuntimeError(KEY)

    stub.AutoTokenizer = SimpleNamespace(from_pretrained=fail)
    stub.AutoModelForCausalLM = SimpleNamespace(from_pretrained=fail)
    monkeypatch.setitem(sys.modules, "transformers", stub)
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert collector._try_load_ppl_model() is None
    assert collector.ppl_model is None and collector.ppl_tokenizer is None
    event = context.logs()[-1]
    assert event["event"] == "ppl_model_unavailable" and event["error_type"] == "RuntimeError"

    tokenizer = object()
    evaluated = []
    model = SimpleNamespace(eval=lambda: evaluated.append(True))
    stub.AutoTokenizer.from_pretrained = lambda _name: tokenizer
    stub.AutoModelForCausalLM.from_pretrained = lambda _name: model
    assert collector._try_load_ppl_model() is None
    assert collector.ppl_tokenizer is tokenizer and collector.ppl_model is model
    assert collector._torch is torch and evaluated == [True]
    assert context.logs()[-1]["event"] == "ppl_model_loaded"


def test_history_embedding_failure_hides_source_without_changing_saved_records(context, monkeypatch, tmp_path):
    module = context.module
    cls = type(context.collector)
    monkeypatch.setattr(module, "ModelPool", lambda *_args: SimpleNamespace(stats={}))
    monkeypatch.setattr(module, "setup_logger", lambda *_args: context.logger)
    monkeypatch.setattr(cls, "_try_load_sbert", lambda self: None)
    output = tmp_path / "synthetic-history.json"
    label = context.fields["label"]
    cfg = module.RuntimeConfig(output=str(output), target_count=2, categories=[label], distribution=[1.0])
    collector = cls(cfg, [])
    collector.store.rewrite_all([module.Record(entry_id="synthetic-history", label=label,
                                               **{context.text_field: TEXT})])
    before = output.read_bytes()
    seen = []

    def fail(texts, **_kwargs):
        seen.extend(texts)
        raise RuntimeError(f"{texts[0]} {KEY}")

    collector.embed_model = SimpleNamespace(encode=fail)

    async def run():
        assert await collector._load_existing_state() is None
        assert seen == [TEXT]
        assert collector.state.existing_count == 1 and collector.state.accepted_new_count == 0
        assert collector.state.label_counts == {label: 1}
        assert output.read_bytes() == before
        events = context.logs()
        failed = [event for event in events if event["event"] == "history_embedding_failed"]
        assert len(failed) == 1 and failed[0]["error_type"] == "RuntimeError"

        restored = cls(cfg, [])
        restored.embed_model = SimpleNamespace(encode=lambda texts, **_kwargs: [[1.0, 0.0] for _ in texts])
        assert await restored._load_existing_state() is None
        assert restored.embeddings.shape == (1, 2)
        assert restored.embedding_texts == [TEXT]
        assert restored.state.existing_count == 1 and restored.state.label_counts == {label: 1}
        assert output.read_bytes() == before
        assert context.logs()[-1]["event"] == "history_load_done"

    asyncio.run(run())
