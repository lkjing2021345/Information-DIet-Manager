"""Restore transactions retain request ownership through ASGI cancellation.

These tests use the real ASGI app and SQLite transactions with synthetic records.
Controlled thread pauses stand in for slow storage, not real browser disconnects.
"""

import asyncio
import contextlib
import threading
from contextlib import contextmanager

import pytest

from src.hyh import app as api, data_management as management, db
from src.hyh.security import process_ownership
from src.hyh.tests.test_export_delivery import Delivery, collect, request


TABLES = ("items", "embeddings", "analysis_jobs", "analysis_runs", "stats_daily")


@pytest.fixture
def restore_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "synthetic-restore-cancellation.sqlite3")


def snapshot():
    with db.get_conn() as conn:
        return {table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table}")]
                for table in TABLES}


async def seed_restore():
    await collect(0)
    await collect(1)
    backup = (await request("GET", "/data/backup")).json()
    await collect(2)
    assert (await request("POST", "/analyze/run")).status == 200
    return backup, snapshot()


def pause_restore(monkeypatch, *, worker_fails, events=None):
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    original_embedding = api._upsert_embedding
    original_connection = management.get_conn

    def held_embedding(*args):
        original_embedding(*args)
        # Pause after a real partial write, inside the replacement transaction.
        if not entered.is_set():
            entered.set()
            assert release.wait(10), "Synthetic restore was not released"
            if worker_fails:
                raise OSError("Synthetic private restore failure")

    @contextmanager
    def tracked_connection(*args, **kwargs):
        try:
            with original_connection(*args, **kwargs) as conn:
                yield conn
        finally:
            closed.set()
            if events is not None:
                events.append("transaction_closed")

    monkeypatch.setattr(api, "_upsert_embedding", held_embedding)
    monkeypatch.setattr(management, "get_conn", tracked_connection)
    return entered, release, closed


async def finish(task, release):
    release.set()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(task), 5)


@pytest.mark.parametrize("cancel_count", [0, 1, 3])
@pytest.mark.parametrize("worker_fails", [False, True])
def test_restore_retains_ownership_until_commit_or_rollback(restore_db, monkeypatch, cancel_count, worker_fails):
    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            backup, before = await seed_restore()
            entered, release, closed = pause_restore(monkeypatch, worker_fails=worker_fails)
            delivery = Delivery()
            task = asyncio.create_task(request("POST", "/data/restore", payload=backup,
                headers={"X-IDM-Confirm": "replace-records", "Origin": "http://127.0.0.1:5173"}, delivery=delivery))
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                for _ in range(cancel_count):
                    task.cancel()
                    await asyncio.sleep(0)
                assert not task.done()
                assert not closed.is_set()
                assert api.app.state.operation_lock.locked()
                assert api.app.state.request_slots._value == 3
                assert snapshot() == before  # Partial replacement is invisible.
                release.set()
                if cancel_count:
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(asyncio.shield(task), 5)
                    assert delivery.status is None and not delivery.body
                else:
                    response = await asyncio.wait_for(task, 5)
                    if worker_fails:
                        assert response.status == 500
                        assert response.json() == {"detail": "Local data operation failed"}
                        assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
                        assert response.headers["cache-control"] == "no-store"
                    else:
                        assert response.status == 200
                        assert response.json() == {"restored": 2, "analysis_cleared": True}
                assert closed.is_set()
                assert not api.app.state.operation_lock.locked()
                assert api.app.state.request_slots._value == 4
                if worker_fails:
                    assert snapshot() == before
                else:
                    after = snapshot()
                    assert len(after["items"]) == len(after["embeddings"]) == 2
                    assert all(not after[table] for table in TABLES[2:])
                    assert (await request("GET", "/data/backup")).json()["items"] == backup["items"]
                await collect(99)  # The connection and write gate are usable again.
            finally:
                await finish(task, release)
    asyncio.run(scenario())


@pytest.mark.parametrize("worker_fails", [False, True])
def test_drained_shutdown_closes_cancelled_restore_transaction_before_releasing_process_lock(
        restore_db, monkeypatch, worker_fails):
    events = []
    owner = api.process_ownership

    @contextmanager
    def tracked_owner(path):
        with owner(path):
            yield
        events.append("ownership_released")

    monkeypatch.setattr(api, "process_ownership", tracked_owner)

    async def scenario():
        ready = asyncio.Event()
        state = {}

        async def serve_until_request_exits():
            # Model server shutdown ordering: drain/cancel request tasks, then
            # stop lifespan. A server bypassing drain after its graceful timeout
            # is outside this test; so is a force-killed OS process.
            async with api.app.router.lifespan_context(api.app):
                backup, _before = await seed_restore()
                state["entered"], state["release"], state["closed"] = pause_restore(
                    monkeypatch, worker_fails=worker_fails, events=events)
                state["task"] = asyncio.create_task(request("POST", "/data/restore", payload=backup,
                    headers={"X-IDM-Confirm": "replace-records"}))
                ready.set()
                with contextlib.suppress(asyncio.CancelledError):
                    await state["task"]

        server = asyncio.create_task(serve_until_request_exits())
        try:
            await asyncio.wait_for(ready.wait(), 5)
            assert await asyncio.to_thread(state["entered"].wait, 5)
            for _ in range(3):
                state["task"].cancel()
                await asyncio.sleep(0)
            assert not state["task"].done()
            assert not server.done()
            assert not state["closed"].is_set()
            with pytest.raises(RuntimeError, match="Another IDM process owns"):
                with process_ownership(db.DB_PATH):
                    pass
            state["release"].set()
            await asyncio.wait_for(server, 5)
            assert state["closed"].is_set()
            assert events == ["transaction_closed", "ownership_released"]
            with process_ownership(db.DB_PATH):
                pass
        finally:
            if "release" in state:
                state["release"].set()
            await asyncio.wait_for(server, 5)
    asyncio.run(scenario())
