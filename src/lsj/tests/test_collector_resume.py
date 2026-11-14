"""Resume collectors against real temporary output and progress files.

Only generation/model construction are stubbed. History loading, file formats,
progress persistence, accepted-record accounting, and collect() are real.
"""
import asyncio
import importlib.util
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


UTILS = Path(__file__).resolve().parents[1] / "src" / "algorithms" / "utils"


@pytest.fixture(params=["classifier", "sentiment"])
def collector_factory(request, monkeypatch):
    kind = request.param
    name = f"_resume_fixture_{kind}"
    spec = importlib.util.spec_from_file_location(name, UTILS / f"{kind}_data_collector.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    cls = getattr(module, f"{kind.title()}DataCollector")

    class NoModelPool:
        def __init__(self, *_args):
            self.stats = {}
            self.closed = False

        async def close(self):
            self.closed = True

    # Sentiment otherwise attempts to initialize an optional embedding model.
    monkeypatch.setattr(cls, "_try_load_sbert", lambda self: None)
    monkeypatch.setattr(module, "ModelPool", NoModelPool)
    monkeypatch.setattr(module, "setup_logger", lambda *_args: logging.getLogger(name))
    labels = ["News", "Tools"] if kind == "classifier" else ["平静", "开心"]
    text_field = "input" if kind == "classifier" else "text"

    def create(output, target=3):
        cfg = module.RuntimeConfig(
            output=str(output), target_count=target,
            categories=labels, distribution=[0.5, 0.5],
            enable_contrast_pairs=False, flush_every=1,
        )
        return cls(cfg, [])

    def record(number, label=None):
        return module.Record(
            entry_id=f"synthetic-{number}", label=label or labels[0],
            **{text_field: f"合成记录 {number}"},
        )

    return SimpleNamespace(create=create, record=record, labels=labels, module=module)


@pytest.mark.parametrize("extension", ["json", "jsonl", "csv"])
def test_repeated_resume_counts_each_persisted_record_once(collector_factory, tmp_path, extension):
    factory = collector_factory
    output = tmp_path / f"records.{extension}"
    dispatched = []

    async def run():
        # Three normal partial runs each persist one record. A fourth startup
        # observes the complete file and must not call generation again.
        for expected_count in (1, 2, 3, 3):
            collector = factory.create(output)

            async def generate_one():
                number = len(dispatched) + 1
                dispatched.append(number)
                spec = collector._build_base_spec()
                await collector._append_record(factory.record(number, spec["label"]), spec)

            collector._run_dispatch_loop = generate_one
            await collector.collect()
            rows, _stats = collector.store.load_existing_records()
            assert len(rows) == expected_count
            assert collector.state.total_effective_count == expected_count
            expected_labels = {label: sum(row.label == label for row in rows) for label in factory.labels}
            assert collector.state.label_counts == expected_labels
            assert collector.pool.closed
            saved = collector.progress.load()
            assert saved.total_effective_count == expected_count
        assert dispatched == [1, 2, 3]
        assert collector.state.existing_count == 3
        assert collector.state.accepted_new_count == 0

    asyncio.run(run())


@pytest.mark.parametrize("extension", ["json", "jsonl", "csv"])
@pytest.mark.parametrize("persisted_count", [0, 2])
def test_stale_sidecar_cannot_invent_records_or_label_counts(
    collector_factory, tmp_path, extension, persisted_count,
):
    factory = collector_factory
    collector = factory.create(tmp_path / f"records.{extension}", target=5)
    records = [factory.record(i, factory.labels[i % 2]) for i in range(persisted_count)]
    if records:
        collector.store.rewrite_all(records)

    async def run():
        restored = factory.module.ProgressState(
            output_path=collector.cfg.output, target_count=5,
            existing_count=8, accepted_new_count=9,
            generated_count=12, attempt_count=13, duplicate_count=4,
            label_counts={factory.labels[0]: 100, factory.labels[1]: 200},
            status="partial",
        )
        await collector.progress.save(restored)
        await collector._load_existing_state()
        assert collector.state.existing_count == persisted_count
        assert collector.state.accepted_new_count == 0
        assert collector._remaining_target() == 5 - persisted_count
        assert not collector._is_done()
        assert collector.state.label_counts == {
            label: sum(row.label == label for row in records) for label in factory.labels
        }
        # Diagnostic attempt/failure history remains resumable; it never adds
        # to the physical-record count used for target completion.
        assert collector.state.generated_count == 12
        assert collector.state.attempt_count == 13
        assert collector.state.duplicate_count == 4
        await collector.pool.close()

    asyncio.run(run())
