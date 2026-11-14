"""Confidence integrity through real processing and history; generation is offline."""
import asyncio
import csv
import importlib.util
import io
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


PRIVATE = "SYNTHETIC_PRIVATE_CONFIDENCE_TOKEN"
TEXT = "Python reference SYNTHETIC_PRIVATE_PAGE_TEXT"
MISSING = object()
BAD = [
    pytest.param(None, id="null"), pytest.param(True, id="true"), pytest.param(False, id="false"),
    pytest.param({}, id="object"), pytest.param([], id="array"), pytest.param("", id="empty"),
    pytest.param(PRIVATE, id="invalid-text"), pytest.param("NaN", id="nan-text"),
    pytest.param("Infinity", id="infinity-text"), pytest.param("-Infinity", id="negative-infinity-text"),
    pytest.param("1e999", id="overflow-text"), pytest.param(float("nan"), id="nan-number"),
    pytest.param(float("inf"), id="infinity-number"), pytest.param(-float("inf"), id="negative-infinity-number"),
    pytest.param(1.01, id="above-one"), pytest.param(-.01, id="below-zero"),
    pytest.param(10 ** 400, id="overflow-integer"),
]
GOOD = [pytest.param(MISSING, 1.0, id="missing-compatible"), pytest.param(0, 0.0, id="zero"),
        pytest.param(1, 1.0, id="one"), pytest.param(.05, .05, id="low-valid"),
        pytest.param(" 0.95 ", .95, id="numeric-text"), pytest.param("1e0", 1.0, id="numeric-exponent")]


@pytest.fixture
def context(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "src/algorithms/utils/classifier_data_collector.py"
    name = "_collector_confidence_contract"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    stream = io.StringIO()
    logger = logging.Logger(name)
    logger.addHandler(logging.StreamHandler(stream))
    logger.setLevel(logging.DEBUG)
    construction = []

    class Pool:
        def __init__(self, *_args):
            construction.append("pool")
            self.stats, self.responses, self.calls, self.closed = {}, [], 0, False

        async def generate(self, *_args):
            self.calls += 1
            assert self.responses, "Unexpected model request"
            return self.responses.pop(0), "synthetic-model"

        async def close(self):
            self.closed = True

    def setup_logger(*_args):
        construction.append("logger")
        return logger

    monkeypatch.setattr(module, "ModelPool", Pool)
    monkeypatch.setattr(module, "setup_logger", setup_logger)
    monkeypatch.setattr(module.ClassifierDataCollector, "_try_load_sbert", lambda self: None)

    def create(*, output=None, threshold=.9, target=1):
        cfg = module.RuntimeConfig(output=str(output or tmp_path / "synthetic.jsonl"),
            target_count=target, categories=["Tools"], distribution=[1.0],
            min_confidence=threshold, enable_relabel_check=True,
            enable_contrast_pairs=False, flush_every=1)
        return module.ClassifierDataCollector(cfg, [])

    def private_logs_absent():
        assert PRIVATE not in stream.getvalue() and TEXT not in stream.getvalue()
        # Strict JSON diagnostic consumers must not receive NaN/Infinity tokens.
        for line in stream.getvalue().splitlines():
            json.loads(line, parse_constant=lambda value: pytest.fail("Non-finite log value"))

    return SimpleNamespace(module=module, create=create, construction=construction,
                           private_logs_absent=private_logs_absent)


def response(value, *, relabel=False):
    payload = {"label": "Tools"}
    if not relabel:
        payload["input"] = TEXT
    if value is not MISSING:
        payload["confidence"] = value
    return json.dumps(payload)


@pytest.mark.parametrize("value", BAD)
def test_invalid_generation_never_reaches_writer(context, value):
    collector = context.create()
    collector.pool.responses = [response(value), response(.95, relabel=True)]

    async def run():
        assert await collector._process_spec(collector._build_base_spec()) is False
        assert collector.state.attempt_count == collector.state.failed_count == 1
        assert collector.state.generated_count == collector.state.accepted_new_count == 0
        assert collector._pending_flush == [] and collector.pool.calls == 1
        assert not Path(collector.cfg.output).exists()
        await collector.pool.close()

    asyncio.run(run())
    context.private_logs_absent()


@pytest.mark.parametrize("value", BAD)
def test_invalid_relabel_never_reaches_writer(context, value):
    collector = context.create()
    collector.pool.responses = [response(.95), response(value, relabel=True)]

    async def run():
        assert await collector._process_spec(collector._build_base_spec()) is False
        assert collector.state.attempt_count == collector.state.relabel_rejected_count == 1
        assert collector.state.accepted_new_count == collector.state.generated_count == 0
        assert collector._pending_flush == [] and collector.pool.calls == 2
        assert not Path(collector.cfg.output).exists()
        await collector.pool.close()

    asyncio.run(run())
    context.private_logs_absent()


@pytest.mark.parametrize("value,expected", GOOD)
def test_compatible_generation_and_relabel_values(context, value, expected):
    collector = context.create(threshold=0)
    collector.pool.responses = [response(value)]

    async def run():
        record = await collector._generate_one(collector._build_base_spec())
        assert record is not None and record.confidence == expected
        assert record.input == TEXT
        collector.pool.responses = [response(value, relabel=True)]
        assert await collector._relabel_check(record) is (expected >= .7)
        await collector.pool.close()

    asyncio.run(run())
    context.private_logs_absent()


def write_history(path, rows):
    if path.suffix == ".csv":
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            writer.writeheader()
            writer.writerows(rows)
    elif path.suffix == ".jsonl":
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    else:
        path.write_text(json.dumps(rows), encoding="utf-8")


def history_row(value, *, identity="synthetic"):
    row = dict(entry_id=identity, input=TEXT, label="Tools")
    if value is not MISSING:
        row["confidence"] = value
    return row


@pytest.mark.parametrize("extension", ["json", "jsonl", "csv"])
@pytest.mark.parametrize("value,expected", [pytest.param(MISSING, 1.0, id="missing-compatible"),
    pytest.param(" 0.95 ", .95, id="numeric-text"), pytest.param(1, 1.0, id="one")])
def test_valid_generation_relabel_and_persistence_keep_confidence(context, tmp_path, extension, value, expected):
    collector = context.create(output=tmp_path / ("generated." + extension))
    collector.pool.responses = [response(value), response(value, relabel=True)]

    async def run():
        assert await collector._process_spec(collector._build_base_spec()) is True
        records, stats = collector.store.load_existing_records()
        assert len(records) == 1 and records[0].input == TEXT and records[0].confidence == expected
        assert stats["invalid"] == 0
        assert collector.state.accepted_new_count == collector.state.attempt_count == 1
        assert collector.pool.calls == 2
        await collector.pool.close()

    asyncio.run(run())
    context.private_logs_absent()


@pytest.mark.parametrize("extension", ["json", "jsonl", "csv"])
@pytest.mark.parametrize("value", BAD)
def test_invalid_history_aborts_without_rewriting_or_counting(context, tmp_path, extension, value):
    output = tmp_path / ("history." + extension)
    write_history(output, [history_row(.05, identity="valid-low"), history_row(value)])
    before = output.read_bytes()
    collector = context.create(output=output, target=2)

    async def run():
        with pytest.raises(ValueError, match="Historical confidence is invalid") as raised:
            await collector.collect()
        assert PRIVATE not in str(raised.value) and TEXT not in str(raised.value)
        assert output.read_bytes() == before
        assert collector.state.existing_count == collector.state.total_effective_count == 0
        assert collector.state.status == "failed"
        assert collector.pool.calls == 0 and collector.pool.closed

    asyncio.run(run())
    context.private_logs_absent()


@pytest.mark.parametrize("extension", ["json", "jsonl", "csv"])
@pytest.mark.parametrize("value,expected", GOOD)
def test_valid_history_preserves_bytes_and_does_not_apply_new_threshold(context, tmp_path, extension, value, expected):
    output = tmp_path / ("history." + extension)
    write_history(output, [history_row(value)])
    before = output.read_bytes()
    collector = context.create(output=output, threshold=.9)

    async def run():
        await collector.collect()
        records, stats = collector.store.load_existing_records()
        assert len(records) == 1 and records[0].confidence == expected and stats["invalid"] == 0
        assert output.read_bytes() == before
        assert collector.state.existing_count == collector.state.total_effective_count == 1
        assert collector.state.status == "completed"
        assert collector.pool.calls == 0 and collector.pool.closed

    asyncio.run(run())
    context.private_logs_absent()


@pytest.mark.parametrize("extension", ["json", "jsonl", "csv"])
def test_duplicate_history_cannot_hide_invalid_confidence(context, tmp_path, extension):
    output = tmp_path / ("duplicates." + extension)
    write_history(output, [history_row(.95), history_row("NaN")])
    before = output.read_bytes()
    collector = context.create(output=output)
    with pytest.raises(ValueError, match="Historical confidence is invalid"):
        asyncio.run(collector.collect())
    assert output.read_bytes() == before and collector.pool.closed
    assert collector.pool.calls == 0
    context.private_logs_absent()


@pytest.mark.parametrize("value", BAD)
def test_invalid_threshold_rejected_before_logger_or_pool(context, value):
    with pytest.raises(ValueError, match="min_confidence") as raised:
        context.create(threshold=value)
    assert context.construction == []
    assert PRIVATE not in str(raised.value)


@pytest.mark.parametrize("value", [0, 1, .95, " 0.95 "])
def test_valid_threshold_keeps_value(context, value):
    collector = context.create(threshold=value)
    assert collector.min_confidence == float(value)
    assert context.construction == ["logger", "pool"]
    asyncio.run(collector.pool.close())
