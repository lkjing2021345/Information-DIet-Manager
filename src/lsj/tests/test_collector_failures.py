"""Training collector failure boundaries with real temporary files and no models.

Model construction is substituted. Focused lifecycle tests substitute dispatch;
task-ownership tests use the real dispatcher with controlled worker functions.
Successful append means the store returned normally, not fsync durability.
"""
import asyncio
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

UTILS = Path(__file__).resolve().parents[1] / "src" / "algorithms" / "utils"


@pytest.fixture(params=["classifier", "sentiment"])
def factory(request, monkeypatch):
    kind = request.param
    name = f"_failure_fixture_{kind}"
    spec = importlib.util.spec_from_file_location(name, UTILS / f"{kind}_data_collector.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    cls = getattr(module, f"{kind.title()}DataCollector")
    real_pool = module.ModelPool

    class NoModelPool:
        def __init__(self, *_args):
            self.stats = {}
            self.close_calls = 0
            self.close_error = None

        async def close(self):
            self.close_calls += 1
            if self.close_error is not None:
                raise self.close_error

    monkeypatch.setattr(cls, "_try_load_sbert", lambda self: None)
    monkeypatch.setattr(module, "ModelPool", NoModelPool)
    monkeypatch.setattr(module, "setup_logger", lambda *_args: logging.getLogger(name))
    labels = ["News", "Tools"] if kind == "classifier" else ["平静", "开心"]
    field = "input" if kind == "classifier" else "text"

    def create(output, target=2, flush_every=1):
        cfg = module.RuntimeConfig(output=str(output), target_count=target, categories=labels,
                                   distribution=[0.5, 0.5], enable_contrast_pairs=False,
                                   flush_every=flush_every, max_workers=2, batch_size=2)
        return cls(cfg, [])

    def record(number, label=None):
        return module.Record(entry_id=f"synthetic-{number}", label=label or labels[0],
                             **{field: f"合成记录 {number}"})

    return SimpleNamespace(create=create, record=record, labels=labels, module=module,
                           real_pool=real_pool)


def contains_error(error, expected):
    seen = set()
    while error is not None and id(error) not in seen:
        if error is expected:
            return True
        seen.add(id(error))
        # An incidental __context__ on a later cleanup error would still mask
        # the primary failure; only deliberate exception chaining is accepted.
        error = error.__cause__
    return False


@pytest.mark.parametrize("extension", ["json", "jsonl", "csv"])
@pytest.mark.parametrize("partial_write", [False, True])
def test_failed_append_preserves_pending_and_first_error_without_replay(factory, tmp_path, monkeypatch,
                                                                       extension, partial_write):
    output = tmp_path / f"records.{extension}"
    collector = factory.create(output)
    collector.store.rewrite_all([factory.record(0)])
    before = output.read_bytes()
    primary = OSError("synthetic write failed")
    collector.pool.close_error = RuntimeError("synthetic later close failed")
    append_calls = []

    def fail(records):
        append_calls.append(list(records))
        if partial_write:
            with output.open("ab") as stream:
                stream.write(b"\nSYNTHETIC_PARTIAL_BYTES")
        raise primary

    monkeypatch.setattr(collector.store, "append_records", fail)

    async def dispatch():
        spec = collector._build_base_spec()
        await collector._append_record(factory.record(1, spec["label"]), spec)

    collector._run_dispatch_loop = dispatch

    async def run():
        with pytest.raises(Exception) as raised:
            await collector.collect()
        assert contains_error(raised.value, primary)
        assert collector.state.status == "failed"
        assert collector.state.accepted_new_count == 0
        assert collector.state.total_effective_count == 1
        assert [row.entry_id for row in collector._pending_flush] == ["synthetic-1"]
        assert len(append_calls) == 1  # No unsafe retry after a possibly partial write.
        assert collector.pool.close_calls == 1
        assert output.read_bytes() == before + (b"\nSYNTHETIC_PARTIAL_BYTES" if partial_write else b"")
        with pytest.raises(Exception):
            await collector.collect()
        assert len(append_calls) == 1
        assert [row.entry_id for row in collector._pending_flush] == ["synthetic-1"]
        assert collector.state.accepted_new_count == 0
        assert collector.state.total_effective_count == 1
        assert output.read_bytes() == before + (b"\nSYNTHETIC_PARTIAL_BYTES" if partial_write else b"")

    asyncio.run(run())


@pytest.mark.parametrize("stage", ["startup", "after_append", "completed"])
def test_progress_failure_never_reports_completed_and_always_closes(factory, tmp_path, monkeypatch, stage):
    output = tmp_path / "records.jsonl"
    collector = factory.create(output, target=1)
    primary = OSError("synthetic progress failed")
    if stage != "completed":
        collector.pool.close_error = RuntimeError("synthetic later close failed")
    original = collector.progress.save
    failed = []

    async def save(state):
        matches = ((stage == "startup" and state.status in {"ready", "resumed"}) or
                   (stage == "after_append" and state.accepted_new_count == 1 and state.status == "running") or
                   (stage == "completed" and state.status == "completed"))
        if matches and not failed:
            failed.append(True)
            raise primary
        await original(state)

    monkeypatch.setattr(collector.progress, "save", save)

    async def dispatch():
        spec = collector._build_base_spec()
        await collector._append_record(factory.record(1, spec["label"]), spec)

    collector._run_dispatch_loop = dispatch

    async def run():
        with pytest.raises(Exception) as raised:
            await collector.collect()
        assert contains_error(raised.value, primary)
        assert failed
        assert collector.state.status == "failed"
        assert collector.pool.close_calls == 1
        rows, _ = collector.store.load_existing_records()
        assert len(rows) == (0 if stage == "startup" else 1)
        assert collector.state.accepted_new_count == len(rows)

    asyncio.run(run())


@pytest.mark.parametrize("close_fails", [False, True])
def test_cancelled_collect_preserves_unflushed_buffer_and_propagates(factory, tmp_path, close_fails):
    output = tmp_path / "records.jsonl"
    collector = factory.create(output, target=2, flush_every=50)
    if close_fails:
        collector.pool.close_error = OSError("synthetic cleanup after cancellation")

    async def dispatch():
        spec = collector._build_base_spec()
        await collector._append_record(factory.record(1, spec["label"]), spec)
        raise asyncio.CancelledError()

    collector._run_dispatch_loop = dispatch

    async def run():
        with pytest.raises(asyncio.CancelledError):
            await collector.collect()
        assert collector.state.status == "interrupted"
        assert collector.state.accepted_new_count == 0
        assert [row.entry_id for row in collector._pending_flush] == ["synthetic-1"]
        assert not output.exists()
        assert collector.pool.close_calls == 1
        with pytest.raises(Exception):
            await collector.collect()
        assert [row.entry_id for row in collector._pending_flush] == ["synthetic-1"]
        assert collector.state.accepted_new_count == 0
        assert not output.exists()

    asyncio.run(run())


@pytest.mark.parametrize("already_complete", [False, True])
def test_pool_close_failure_is_failed_even_when_output_target_is_met(factory, tmp_path, already_complete):
    output = tmp_path / "records.jsonl"
    collector = factory.create(output, target=1)
    if already_complete:
        collector.store.rewrite_all([factory.record(0)])
    primary = OSError("synthetic pool close failed")
    collector.pool.close_error = primary

    async def dispatch():
        spec = collector._build_base_spec()
        await collector._append_record(factory.record(1, spec["label"]), spec)

    collector._run_dispatch_loop = dispatch

    async def run():
        with pytest.raises(Exception) as raised:
            await collector.collect()
        assert contains_error(raised.value, primary)
        assert collector.state.status == "failed"
        assert collector.pool.close_calls == 1
        rows, _ = collector.store.load_existing_records()
        assert len(rows) == 1
        assert collector.state.total_effective_count == 1

    asyncio.run(run())


def test_normal_final_flush_failure_keeps_buffer_without_retry(factory, tmp_path, monkeypatch):
    output = tmp_path / "records.jsonl"
    collector = factory.create(output, target=2, flush_every=50)
    primary = OSError("synthetic final flush failed")
    calls = []

    def fail(records):
        calls.append(list(records))
        raise primary

    monkeypatch.setattr(collector.store, "append_records", fail)

    async def dispatch():
        spec = collector._build_base_spec()
        await collector._append_record(factory.record(1, spec["label"]), spec)

    collector._run_dispatch_loop = dispatch

    async def run():
        with pytest.raises(Exception) as raised:
            await collector.collect()
        assert contains_error(raised.value, primary)
        assert collector.state.status == "failed"
        assert collector.state.accepted_new_count == 0
        assert len(calls) == 1 and len(collector._pending_flush) == 1
        assert collector.pool.close_calls == 1
        assert not output.exists()

    asyncio.run(run())


def test_fatal_writer_failure_cancels_and_awaits_owned_sibling_tasks(factory, tmp_path, monkeypatch):
    collector = factory.create(tmp_path / "records.jsonl", target=2)
    primary = OSError("synthetic writer failed")
    tasks = []
    append_calls = []

    def fail(records):
        append_calls.append(list(records))
        raise primary

    monkeypatch.setattr(collector.store, "append_records", fail)

    async def run():
        sibling_started = asyncio.Event()
        sibling_cancelled = asyncio.Event()
        never = asyncio.Event()

        async def process(spec):
            tasks.append(asyncio.current_task())
            if len(tasks) == 1:
                await sibling_started.wait()
                await collector._append_record(factory.record(1, spec["label"]), spec)
            else:
                sibling_started.set()
                try:
                    await never.wait()
                except asyncio.CancelledError:
                    sibling_cancelled.set()
                    raise

        collector._process_spec = process
        with pytest.raises(Exception) as raised:
            await asyncio.wait_for(collector.collect(), timeout=2)
        assert contains_error(raised.value, primary)
        assert len(tasks) == 2 and all(task.done() for task in tasks)
        assert sibling_cancelled.is_set()
        assert len(append_calls) == 1
        assert collector.state.status == "failed"
        assert collector.state.accepted_new_count == 0
        assert collector.pool.close_calls == 1

    asyncio.run(run())


def test_buffered_rows_are_counted_only_after_successful_append(factory, tmp_path):
    collector = factory.create(tmp_path / "records.jsonl", target=3, flush_every=2)

    async def dispatch():
        for number in (1, 2, 3):
            spec = collector._build_base_spec()
            await collector._append_record(factory.record(number, spec["label"]), spec)
            if number == 1:
                assert collector.state.accepted_new_count == 0
                assert collector._remaining_target() == 2
            if number == 2:
                assert collector.state.accepted_new_count == 2

    collector._run_dispatch_loop = dispatch

    async def run():
        await collector.collect()
        rows, _ = collector.store.load_existing_records()
        assert len(rows) == 3
        assert collector.state.accepted_new_count == 3
        assert collector.state.status == "completed"
        assert collector._pending_flush == []
        assert collector.pool.close_calls == 1

    asyncio.run(run())


def test_model_pool_closes_all_clients_and_preserves_first_error(factory):
    primary = OSError("synthetic first close failed")
    visited = []

    class Client:
        def __init__(self, number, error=None):
            self.number, self.error = number, error

        async def close(self):
            visited.append(self.number)
            if self.error is not None:
                raise self.error

    pool = object.__new__(factory.real_pool)
    pool.clients = {"one": Client(1, primary), "two": Client(2, ValueError("later")), "three": Client(3)}

    async def run():
        with pytest.raises(Exception) as raised:
            await pool.close()
        assert contains_error(raised.value, primary)
        assert visited == [1, 2, 3]

    asyncio.run(run())


def test_invalid_history_file_closes_pool_without_replacing_load_error(factory, tmp_path, monkeypatch):
    output = tmp_path / "records.json"
    invalid = b'{"synthetic_broken_json":'
    output.write_bytes(invalid)
    collector = factory.create(output)
    collector.pool.close_error = RuntimeError("synthetic later close failed")
    original = collector.store.load_existing_records
    errors = []

    def load():
        try:
            return original()
        except Exception as error:
            errors.append(error)
            raise

    monkeypatch.setattr(collector.store, "load_existing_records", load)

    async def run():
        with pytest.raises(Exception) as raised:
            await collector.collect()
        assert len(errors) == 1 and contains_error(raised.value, errors[0])
        assert collector.state.status == "failed"
        assert collector.pool.close_calls == 1
        assert output.read_bytes() == invalid
        assert collector.state.accepted_new_count == 0

    asyncio.run(run())


def test_append_error_survives_progress_and_close_failures(factory, tmp_path, monkeypatch):
    collector = factory.create(tmp_path / "records.jsonl", target=1)
    primary = OSError("synthetic first write error")
    collector.pool.close_error = RuntimeError("synthetic last close error")
    original_save = collector.progress.save
    writes = []

    def append(records):
        writes.append(list(records))
        raise primary

    async def save(state):
        if writes:
            raise ValueError("synthetic later progress error")
        await original_save(state)

    monkeypatch.setattr(collector.store, "append_records", append)
    monkeypatch.setattr(collector.progress, "save", save)

    async def dispatch():
        spec = collector._build_base_spec()
        await collector._append_record(factory.record(1, spec["label"]), spec)

    collector._run_dispatch_loop = dispatch

    async def run():
        with pytest.raises(Exception) as raised:
            await collector.collect()
        assert contains_error(raised.value, primary)
        assert collector.state.status == "failed"
        assert len(writes) == 1 and len(collector._pending_flush) == 1
        assert collector.state.accepted_new_count == 0
        assert collector.pool.close_calls == 1

    asyncio.run(run())


def test_actual_task_cancellation_reaps_workers_without_flushing_buffer(factory, tmp_path):
    output = tmp_path / "records.jsonl"
    collector = factory.create(output, target=5, flush_every=50)
    tasks = []

    async def run():
        both_started = asyncio.Event()
        never = asyncio.Event()

        async def process(spec):
            tasks.append(asyncio.current_task())
            if len(tasks) == 1:
                await collector._append_record(factory.record(1, spec["label"]), spec)
            else:
                both_started.set()
            await never.wait()

        collector._process_spec = process
        task = asyncio.create_task(collector.collect())
        try:
            await asyncio.wait_for(both_started.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(tasks) == 2 and all(child.done() for child in tasks)
            assert collector.state.status == "interrupted"
            assert collector.state.accepted_new_count == 0
            assert len(collector._pending_flush) == 1
            assert not output.exists()
            assert collector.pool.close_calls == 1
        finally:
            # Also leave a clean test event loop when testing a broken baseline.
            for owned in [task, *tasks]:
                if not owned.done():
                    owned.cancel()
            await asyncio.gather(task, *tasks, return_exceptions=True)

    asyncio.run(run())


def test_actual_dispatch_fast_workers_meet_target_without_extra_rows_or_live_tasks(factory, tmp_path):
    collector = factory.create(tmp_path / "records.jsonl", target=5, flush_every=2)
    tasks = []

    async def run():
        async def process(spec):
            tasks.append(asyncio.current_task())
            number = len(tasks)
            # Alternate yielding and same-tick completion to exercise ownership
            # when a worker finishes before the dispatcher consumes its result.
            if number % 2:
                await asyncio.sleep(0)
            await collector._append_record(factory.record(number, spec["label"]), spec)

        collector._process_spec = process
        await asyncio.wait_for(collector.collect(), timeout=2)
        rows, _ = collector.store.load_existing_records()
        assert len(rows) == 5 and len({row.entry_id for row in rows}) == 5
        assert len(tasks) == 5 and all(task.done() for task in tasks)
        assert collector.state.total_effective_count == 5
        assert collector.state.accepted_new_count == 5
        assert collector.state.status == "completed"
        assert collector._pending_flush == []
        assert collector.pool.close_calls == 1

    asyncio.run(run())


CLI_HARNESS = r'''
import asyncio, runpy, sys
path, fault, output = sys.argv[1:]
def run(coroutine):
    coroutine.close()
    if fault == 'exception':
        raise RuntimeError('SYNTHETIC_PRIVATE_CLI_ERROR kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')
    if fault == 'interrupt':
        raise KeyboardInterrupt()
    if fault == 'cancelled':
        raise asyncio.CancelledError('SYNTHETIC_PRIVATE_CLI_ERROR')
asyncio.run = run
sys.argv = [path, '--output', output]
runpy.run_path(path, run_name='__main__')
'''


@pytest.mark.parametrize("kind", ["classifier", "sentiment"])
@pytest.mark.parametrize("fault,code", [("success", 0), ("exception", 1), ("interrupt", 130), ("cancelled", 130)])
def test_actual_collector_cli_has_safe_exit_status(tmp_path, kind, fault, code):
    output = tmp_path / "unused.jsonl"
    process = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", "-c", CLI_HARNESS,
         str(UTILS / f"{kind}_data_collector.py"), fault, str(output)],
        cwd=tmp_path, env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                          "LOCALAPPDATA": str(tmp_path / "local"), "HF_HOME": str(tmp_path / "hf")},
        capture_output=True, text=True, encoding="utf-8", timeout=20,
    )
    assert process.returncode == code
    assert process.stdout == ""
    if fault == "success":
        assert process.stderr == ""
    else:
        error = json.loads(process.stderr)
        assert error["success"] is False and error["error"]
        assert error["error_type"] == {
            "exception": "RuntimeError", "interrupt": "KeyboardInterrupt", "cancelled": "CancelledError",
        }[fault]
    for private in ("SYNTHETIC_PRIVATE_CLI_ERROR", "k" * 43, "Traceback"):
        assert private not in process.stdout + process.stderr
    assert list(tmp_path.iterdir()) == []
