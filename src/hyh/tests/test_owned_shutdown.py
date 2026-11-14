"""Actual executor completion and Uvicorn shutdown ordering, synthetic data only."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
import uvicorn

from src.hyh import app as api, db
from src.hyh.owned_work import WorkOwner, WorkOwnerClosed, run_owned_sync
from src.hyh.security import process_ownership
from src.hyh.tests.test_export_delivery import Delivery, collect, request
from src.hyh.tests.test_restore_cancellation import pause_restore, snapshot


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.01)


def other_process_lock_available():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from src.hyh.security import process_ownership
try:
    with process_ownership(Path(sys.argv[1])):
        pass
except RuntimeError:
    sys.exit(9)
"""
    result = subprocess.run([sys.executable, "-I", "-c", script, str(db.DB_PATH),
                             str(Path(__file__).resolve().parents[3])],
                            capture_output=True, timeout=10)
    assert result.returncode in {0, 9}, result.stderr
    return result.returncode == 0


@pytest.mark.parametrize("fails", [False, True])
def test_global_task_cancellation_cannot_detach_executor_from_its_owner(fails):
    async def scenario():
        owner = WorkOwner()
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()

        def work():
            entered.set()
            try:
                assert release.wait(10)
                if fails:
                    raise ValueError("Synthetic worker failure")
                return 42
            finally:
                finished.set()

        async def call():
            with owner.bind():
                return await run_owned_sync(work)

        caller = asyncio.create_task(call())
        drain = None
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            drain = asyncio.create_task(owner.drain())
            await until(lambda: owner.closing)
            # Match asyncio.run's shutdown selection: cancel every other Task.
            # Executor Futures must not become independently cancelled Tasks.
            for _ in range(3):
                for task in asyncio.all_tasks() - {asyncio.current_task()}:
                    task.cancel()
                await asyncio.sleep(0)
            assert not caller.done() and not drain.done() and not finished.is_set()
            release.set()
            outcomes = await asyncio.gather(caller, drain, return_exceptions=True)
            assert all(isinstance(value, asyncio.CancelledError) for value in outcomes)
            assert finished.is_set() and not owner._workers
        finally:
            release.set()
            await asyncio.gather(caller, *([drain] if drain else []), return_exceptions=True)
    asyncio.run(scenario())


def test_context_propagation_and_closing_rejection_do_not_start_more_work():
    value = ContextVar("synthetic_context", default="unset")
    async def scenario():
        owner = WorkOwner()
        value.set("synthetic expected")
        with owner.bind():
            assert await run_owned_sync(value.get) == "synthetic expected"
        await owner.drain()
        calls = []
        with owner.bind(), pytest.raises(WorkOwnerClosed):
            await run_owned_sync(lambda: calls.append(True))
        assert calls == [] and not owner._workers
    # A second lifespan/loop cannot inherit a pending Future from the first.
    asyncio.run(scenario())
    asyncio.run(scenario())


def test_owners_on_the_same_loop_drain_independently():
    async def scenario():
        first, second = WorkOwner(), WorkOwner()
        entered, release = threading.Event(), threading.Event()
        def work():
            entered.set()
            assert release.wait(10)
            return "second"
        async def call():
            with second.bind():
                return await run_owned_sync(work)
        task = asyncio.create_task(call())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            await first.drain()
            assert first.closing and not second.closing and not task.done()
            release.set()
            assert await task == "second"
            await second.drain()
        finally:
            release.set()
            await task
    asyncio.run(scenario())


def test_shutdown_also_retains_queued_executor_work_before_thread_start():
    async def scenario():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        occupied, release, executed = threading.Event(), threading.Event(), threading.Event()
        def occupy():
            occupied.set()
            assert release.wait(10)
        blocker = loop.run_in_executor(None, occupy)
        await until(occupied.is_set)
        owner = WorkOwner()
        async def call():
            with owner.bind():
                return await run_owned_sync(executed.set)
        caller = asyncio.create_task(call())
        drain = None
        try:
            await until(lambda: bool(owner._workers))
            drain = asyncio.create_task(owner.drain())
            await until(lambda: owner.closing)
            caller.cancel()
            await asyncio.sleep(0)
            assert not executed.is_set() and not caller.done() and not drain.done()
            release.set()
            result = await asyncio.gather(caller, drain, return_exceptions=True)
            assert isinstance(result[0], asyncio.CancelledError) and result[1] is None
            assert executed.is_set() and not owner._workers
        finally:
            release.set()
            await asyncio.gather(blocker, caller, *([drain] if drain else []), return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize("worker_fails", [False, True])
@pytest.mark.parametrize("cancel_shutdown", [False, True])
def test_uvicorn_timeout_shutdown_retains_database_ownership_until_worker_closes(
        tmp_path, monkeypatch, worker_fails, cancel_shutdown):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "synthetic-uvicorn-shutdown.sqlite3")
    async def scenario():
        lifespan = api.app.router.lifespan_context(api.app)
        await lifespan.__aenter__()
        exited = False
        async def shutdown_lifespan():
            nonlocal exited
            try:
                await lifespan.__aexit__(None, None, None)
            finally:
                exited = True
        task = stop = None
        release = threading.Event()
        try:
            await collect(0)
            backup = (await request("GET", "/data/backup")).json()
            await collect(1)
            before = snapshot()
            entered, release, closed = pause_restore(monkeypatch, worker_fails=worker_fails)
            task = asyncio.create_task(request("POST", "/data/restore", payload=backup,
                headers={"X-IDM-Confirm": "replace-records"}))
            assert await asyncio.to_thread(entered.wait, 5)
            # Exercise the installed Uvicorn shutdown implementation, including
            # its timeout -> cancel requests -> immediately shut lifespan path.
            # Socket/signal setup is omitted; the ASGI app and SQLite are real.
            server = uvicorn.Server(uvicorn.Config(api.app, timeout_graceful_shutdown=0, log_config=None))
            server.servers = []
            server.server_state.tasks.add(task)
            task.add_done_callback(server.server_state.tasks.discard)
            server.lifespan = SimpleNamespace(shutdown=shutdown_lifespan)
            stop = asyncio.create_task(server.shutdown())
            await until(lambda: api.app.state.work_owner.closing)
            if cancel_shutdown:
                for _ in range(3):
                    stop.cancel()
                    await asyncio.sleep(0)
            assert not stop.done() and not task.done() and not closed.is_set()
            with pytest.raises(RuntimeError, match="Another IDM process owns"):
                with process_ownership(db.DB_PATH):
                    pass
            assert not other_process_lock_available()
            rejected = await request("POST", "/collect", payload={},
                                     headers={"Origin": "http://127.0.0.1:5173"})
            assert rejected.status == 503
            assert rejected.json() == {"detail": "Service is shutting down; retry later"}
            assert rejected.headers["retry-after"] == "5"
            assert rejected.headers["cache-control"] == "no-store"
            assert rejected.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
            assert snapshot() == before
            release.set()
            outcomes = await asyncio.wait_for(asyncio.gather(task, stop, return_exceptions=True), 5)
            assert isinstance(outcomes[0], asyncio.CancelledError)
            assert isinstance(outcomes[1], asyncio.CancelledError) if cancel_shutdown else outcomes[1] is None
            assert exited and closed.is_set() and not api.app.state.work_owner._workers
            assert not api.app.state.operation_lock.locked() and api.app.state.request_slots._value == 4
            with process_ownership(db.DB_PATH):
                pass
            assert other_process_lock_available()
            if worker_fails:
                assert snapshot() == before
            else:
                assert len(snapshot()["items"]) == 1
        finally:
            release.set()
            await asyncio.gather(*[value for value in (task, stop) if value], return_exceptions=True)
            if not exited:
                await shutdown_lifespan()
    asyncio.run(scenario())


def test_request_waiting_for_body_keeps_original_lifespan_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "synthetic-stale-owner.sqlite3")
    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            original = api.app.state.work_owner
            entered, release = asyncio.Event(), asyncio.Event()
            delivery = Delivery()
            async def receive():
                entered.set()
                await release.wait()
                return {"type": "http.request", "body": b"{}", "more_body": False}
            scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                     "method": "POST", "scheme": "http", "path": "/collect", "query_string": b"",
                     "headers": [(b"host", b"127.0.0.1"), (b"authorization", b"Bearer " + b"a" * 43),
                                 (b"content-type", b"application/json")],
                     "client": ("127.0.0.1", 51000), "server": ("127.0.0.1", 80)}
            task = asyncio.create_task(api.app(scope, receive, delivery.send))
            try:
                await entered.wait()
                await original.drain()
                # Deliberate state replacement models a later lifespan; the
                # already accepted request must not attach to this new owner.
                newer = WorkOwner()
                api.app.state.work_owner = newer
                release.set()
                await asyncio.wait_for(task, 5)
                assert delivery.status == 503 and not newer._workers
                assert snapshot()["items"] == []
            finally:
                release.set()
                await task
                api.app.state.work_owner = original
    asyncio.run(scenario())
