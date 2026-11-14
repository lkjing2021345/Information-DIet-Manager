"""Write requests own real SQLite work and uploaded files until workers exit."""

import asyncio
import contextlib
import json
import threading
from contextlib import contextmanager

import httpx
import pytest

from src.hyh import app as api, data_management as management, db
from src.hyh.tests.test_export_delivery import BASE_TS, collect, request
from src.hyh.tests.test_restore_cancellation import snapshot


ROUTES = ("collect", "import", "delete_all", "delete_item", "analyze_run", "summary_miss", "summary_hit")
ADMIN = {"Authorization": "Bearer " + "a" * 43, "Origin": "http://127.0.0.1:5173"}


@pytest.fixture
def write_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "synthetic-write-cancellation.sqlite3")


def new_item():
    return {"url": "https://example.invalid/new-write", "title": "Synthetic write 中文 🚀",
            "source": "import", "ts": BASE_TS}


def http_client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app, client=("127.0.0.1", 51000)),
                             base_url="http://127.0.0.1", headers=ADMIN)


def route_request(client, route, item_id):
    if route == "collect":
        return client.post("/collect", json=new_item())
    if route == "import":
        return client.post("/import", files={"file": ("synthetic.json", json.dumps([new_item()]).encode(), "application/json")})
    if route == "delete_all":
        return client.delete("/data", headers={"X-IDM-Confirm": "delete-all"})
    if route == "delete_item":
        return client.delete(f"/items/{item_id}", headers={"X-IDM-Confirm": "delete-record"})
    if route == "analyze_run":
        return client.post("/analyze/run")
    return client.get("/dashboard/summary")


def pause_transaction(monkeypatch, route, worker_fails):
    module = management if route.startswith("delete") else api
    original = module.get_conn
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    writes = []

    @contextmanager
    def held_connection(*args, **kwargs):
        try:
            with original(*args, **kwargs) as conn:
                conn.set_trace_callback(lambda sql: writes.append(sql.split()[0])
                    if sql.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")) else None)
                yield conn
                # Real writes have occurred but commit is still pending.
                entered.set()
                assert release.wait(10), "Synthetic write transaction was not released"
                if worker_fails:
                    raise OSError("Synthetic private write failure")
        finally:
            closed.set()

    monkeypatch.setattr(module, "get_conn", held_connection)
    return entered, release, closed, writes, lambda: monkeypatch.setattr(module, "get_conn", original)


async def finish(task, release, closed):
    release.set()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(task), 5)
    # Also cleanly retire a worker when running these tests against the old bug.
    assert await asyncio.to_thread(closed.wait, 5)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("worker_fails", [False, True])
def test_write_retains_transaction_ownership_until_worker_exit(write_db, monkeypatch, route, cancel, worker_fails):
    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await collect(0)
            await collect(1)
            if route != "summary_miss":
                assert (await request("POST", "/analyze/run")).status == 200
            else:
                # The GET summary also backfills missing embeddings by default.
                with db.get_conn() as conn:
                    conn.execute("DELETE FROM embeddings WHERE item_id=(SELECT MIN(id) FROM items)")
            before = snapshot()
            item_id = before["items"][0][0]
            entered, release, closed, writes, restore_connection = pause_transaction(monkeypatch, route, worker_fails)
            async with http_client() as client:
                task = asyncio.create_task(route_request(client, route, item_id))
                try:
                    assert await asyncio.to_thread(entered.wait, 5)
                    assert writes, "Every listed route must actually write to SQLite"
                    if cancel:
                        for _ in range(3):
                            task.cancel()
                            await asyncio.sleep(0)
                            assert not task.done()
                            assert not closed.is_set()
                            assert api.app.state.operation_lock.locked()
                            assert api.app.state.request_slots._value == 3
                    assert snapshot() == before
                    release.set()
                    if cancel:
                        with pytest.raises(asyncio.CancelledError):
                            await asyncio.wait_for(asyncio.shield(task), 5)
                    else:
                        response = await asyncio.wait_for(task, 5)
                        if worker_fails:
                            assert response.status_code == 500
                            assert response.json() == {"detail": "Local data operation failed"}
                            assert response.headers["cache-control"] == "no-store"
                            assert response.headers["access-control-allow-origin"] == ADMIN["Origin"]
                        else:
                            assert response.status_code == 200
                            if route in {"collect", "import"}:
                                assert response.json() == {"inserted": 1, "duplicates": 0, "failed": 0}
                            elif route.startswith("delete"):
                                assert response.json() == {"deleted": 2 if route == "delete_all" else 1, "analysis_cleared": True}
                            else:
                                assert response.json()["total_count"] == 2
                                assert response.json()["cached"] is (route != "summary_miss")
                    assert closed.is_set()
                    assert not api.app.state.operation_lock.locked()
                    assert api.app.state.request_slots._value == 4
                    after = snapshot()
                    if worker_fails:
                        assert after == before
                    elif route in {"collect", "import"}:
                        assert len(after["items"]) == len(after["embeddings"]) == 3
                    elif route.startswith("delete"):
                        assert len(after["items"]) == len(after["embeddings"]) == (0 if route == "delete_all" else 1)
                        assert not after["analysis_runs"] and not after["stats_daily"]
                    else:
                        assert len(after["items"]) == len(after["embeddings"]) == 2
                        assert len(after["analysis_runs"]) == (2 if route == "analyze_run" else 1)
                        assert len(after["stats_daily"]) == 1
                    # Restore the connection helper before the next independent write.
                    restore_connection()
                    await collect(99)
                finally:
                    await finish(task, release, closed)
    asyncio.run(scenario())


@pytest.mark.parametrize("spooled", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("worker_fails", [False, True])
def test_import_keeps_uploaded_file_open_until_reader_exits(write_db, monkeypatch, spooled, cancel, worker_fails):
    entered, release, reader_done = threading.Event(), threading.Event(), threading.Event()
    original_read = api._read_upload
    uploaded = []
    close_after_reader = []

    def held_read(file):
        uploaded.append(file)
        original_close = file.close

        async def tracked_close():
            close_after_reader.append(reader_done.is_set())
            await original_close()

        file.close = tracked_close
        entered.set()
        try:
            assert release.wait(10), "Synthetic upload reader was not released"
            if worker_fails:
                raise OSError("Synthetic private upload read failure")
            return original_read(file)
        finally:
            reader_done.set()

    monkeypatch.setattr(api, "_read_upload", held_read)

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            before = snapshot()
            payload = json.dumps([new_item()]).encode()
            if spooled:
                payload += b" " * (2 * 1024 * 1024)
            async with http_client() as client:
                task = asyncio.create_task(client.post("/import", files={"file": ("synthetic.json", payload, "application/json")}))
                try:
                    assert await asyncio.to_thread(entered.wait, 5)
                    assert uploaded[0].file._rolled is spooled
                    if cancel:
                        for _ in range(3):
                            task.cancel()
                            await asyncio.sleep(0)
                            assert not uploaded[0].file.closed
                            assert not task.done()
                            assert api.app.state.operation_lock.locked()
                            assert api.app.state.request_slots._value == 3
                    release.set()
                    if cancel:
                        with pytest.raises(asyncio.CancelledError):
                            await asyncio.wait_for(asyncio.shield(task), 5)
                    else:
                        response = await asyncio.wait_for(task, 5)
                        assert response.status_code == (500 if worker_fails else 200)
                    assert reader_done.is_set()
                    assert uploaded[0].file.closed
                    assert close_after_reader == [True]
                    assert not api.app.state.operation_lock.locked()
                    assert api.app.state.request_slots._value == 4
                    if worker_fails:
                        assert snapshot() == before
                    else:
                        assert len(snapshot()["items"]) == 1
                finally:
                    await finish(task, release, reader_done)
    asyncio.run(scenario())
