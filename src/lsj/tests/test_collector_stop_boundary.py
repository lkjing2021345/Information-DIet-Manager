"""Do not start model stages after an already observed persistence failure.

Use the real dispatcher and record-processing flow. Only model generation,
quality/semantic checks and the failing append are controlled synthetic stubs.
"""
import asyncio

import pytest

from test_collector_failures import factory  # noqa: F401


def test_queued_worker_does_not_start_generation_after_writer_failure(factory, tmp_path, monkeypatch):
    collector = factory.create(tmp_path / "records.jsonl", target=2, flush_every=1)
    generation_states = []
    owned_tasks = []
    append_calls = []
    create_task = asyncio.create_task

    def track_task(coroutine, **kwargs):
        task = create_task(coroutine, **kwargs)
        owned_tasks.append(task)
        return task

    async def generate(spec):
        generation_states.append(collector._persistence_error is not None)
        return factory.record(len(generation_states), spec["label"])

    async def yes(*_args):
        return True

    def fail(records):
        append_calls.append(list(records))
        raise OSError("Synthetic stopped writer")

    monkeypatch.setattr(collector, "_generate_one", generate)
    monkeypatch.setattr(collector, "quality_filter", lambda *_args: True)
    monkeypatch.setattr(collector, "semantic_deduplicate", yes)
    if hasattr(collector, "_relabel_check"):
        monkeypatch.setattr(collector, "_relabel_check", yes)
    monkeypatch.setattr(collector.store, "append_records", fail)
    monkeypatch.setattr(factory.module.asyncio, "create_task", track_task)

    async def run():
        with pytest.raises(factory.module.CollectorPersistenceError):
            await asyncio.wait_for(collector.collect(), timeout=2)
        assert generation_states == [False]
        assert len(append_calls) == 1
        assert len(collector._pending_flush) == 1
        assert collector.state.accepted_new_count == 0
        assert collector.state.status == "failed"
        assert collector.pool.close_calls == 1
        assert len(owned_tasks) == 2 and all(task.done() for task in owned_tasks)

    asyncio.run(run())


@pytest.mark.parametrize("factory", ["classifier"], indirect=True)
def test_classifier_does_not_start_relabel_after_waiting_worker_observes_failure(factory, tmp_path, monkeypatch):
    collector = factory.create(tmp_path / "records.jsonl", target=2, flush_every=1)
    generated = []
    relabel_states = []
    owned_tasks = []
    append_calls = []

    async def run():
        second_waiting = asyncio.Event()
        writer_failed = asyncio.Event()

        async def generate(spec):
            generated.append(len(generated) + 1)
            owned_tasks.append(asyncio.current_task())
            return factory.record(generated[-1], spec["label"])

        async def semantic(entry_id, _text):
            if entry_id == "synthetic-1":
                await second_waiting.wait()
            else:
                second_waiting.set()
                await writer_failed.wait()
            return True

        async def relabel(_record):
            relabel_states.append(collector._persistence_error is not None)
            return True

        def fail(records):
            append_calls.append(list(records))
            writer_failed.set()
            raise OSError("Synthetic failure while another worker awaits deduplication")

        monkeypatch.setattr(collector, "_generate_one", generate)
        monkeypatch.setattr(collector, "quality_filter", lambda *_args: True)
        monkeypatch.setattr(collector, "semantic_deduplicate", semantic)
        monkeypatch.setattr(collector, "_relabel_check", relabel)
        monkeypatch.setattr(collector.store, "append_records", fail)
        with pytest.raises(factory.module.CollectorPersistenceError):
            await asyncio.wait_for(collector.collect(), timeout=2)
        assert generated == [1, 2]
        assert relabel_states == [False]
        assert len(append_calls) == 1
        assert len(collector._pending_flush) == 1
        assert collector.state.accepted_new_count == 0
        assert collector.state.status == "failed"
        assert collector.pool.close_calls == 1
        assert all(task.done() for task in owned_tasks)

    asyncio.run(run())
