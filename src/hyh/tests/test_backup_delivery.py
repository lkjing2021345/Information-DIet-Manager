"""Backup delivery retains its restore contract while sharing download limits."""
import asyncio
import contextlib
import hashlib
import json
import threading

import pytest
from starlette.requests import ClientDisconnect

from src.hyh import app as api, data_management as management, db, export_io
from src.hyh.tests.test_export_delivery import (
    COLLECTOR, BASE_TS, Delivery, collect, finish, request,
    wait_for_delivery_ready, preparation_budgets,
)


@pytest.fixture
def backup_streams(tmp_path, monkeypatch):
    streams = []
    create = management.BytesIO

    def tracked_bytes(data):
        stream = create(data)
        streams.append(stream)
        return stream

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "synthetic-backup-delivery.sqlite3")
    monkeypatch.setattr(management, "BytesIO", tracked_bytes)
    yield streams
    for stream in streams:
        stream.close()


def assert_released(streams):
    assert streams and all(stream.closed for stream in streams)
    assert not api.app.state.operation_lock.locked()
    assert api.app.state.export_slots._value == 2
    assert api.app.state.request_slots._value == 4


@pytest.mark.parametrize("count", [0, 3])
def test_backup_format_checksum_filename_and_restore_round_trip_are_unchanged(backup_streams, count):
    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            for index in range(count):
                await collect(index)
            response = await request("GET", "/data/backup", headers={"Origin": "http://127.0.0.1:5173"})
            assert response.status == 200
            assert response.headers["content-type"] == "application/json"
            assert response.headers["content-disposition"] == 'attachment; filename="idm-pages-backup.json"'
            assert int(response.headers["content-length"]) == len(response.body)
            assert response.headers["cache-control"] == "no-store"
            assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
            payload = response.json()
            assert set(payload) == {"format", "version", "exported_at", "items", "sha256"}
            assert payload["format"] == "idm-page-records" and type(payload["version"]) is int and payload["version"] == 1
            assert type(payload["exported_at"]) is int and payload["exported_at"] > 0
            assert len(payload["items"]) == count
            assert all(set(item) == set(management.FIELDS) for item in payload["items"])
            assert payload["sha256"] == hashlib.sha256(management.canonical_items(payload["items"])).hexdigest()
            assert bytes(response.body) == json.dumps(payload, ensure_ascii=False, allow_nan=False,
                                                      separators=(",", ":")).encode("utf-8")
            await collect(99)
            restored = await request("POST", "/data/restore", payload=payload,
                                     headers={"X-IDM-Confirm": "replace-records"})
            assert restored.status == 200 and restored.json()["restored"] == count
            assert (await request("GET", "/data/backup")).json()["items"] == payload["items"]
            assert_released(backup_streams)
    asyncio.run(scenario())


def test_slow_backup_releases_data_gate_and_keeps_original_snapshot_during_writes(backup_streams):
    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await collect(0)
            original = (await request("GET", "/data/backup")).json()
            slow = Delivery(hold=True)
            task = asyncio.create_task(request("GET", "/data/backup", delivery=slow))
            try:
                await wait_for_delivery_ready(task, slow, "/data/backup")
                assert slow.status == 200 and not api.app.state.operation_lock.locked()
                await collect(1)
                newer = await request("GET", "/data/backup")
                assert newer.status == 200 and len(newer.json()["items"]) == 2
                deleted = await request("DELETE", "/data", headers={"X-IDM-Confirm": "delete-all"})
                assert deleted.status == 200 and deleted.json()["deleted"] == 2
                restored = await request("POST", "/data/restore", payload=newer.json(),
                                         headers={"X-IDM-Confirm": "replace-records"})
                assert restored.status == 200 and restored.json()["restored"] == 2
                assert not task.done()
                slow.release.set()
                await asyncio.wait_for(task, 5)
                assert slow.json()["items"] == original["items"]
                assert slow.json()["sha256"] == original["sha256"]
                assert_released(backup_streams)
            finally:
                await finish(task, slow)
    asyncio.run(scenario())


@pytest.mark.parametrize("paths", [("/data/backup", "/data/backup"),
                                    ("/data/backup", "/export/lsj"),
                                    ("/export/lsj/training", "/data/backup")])
def test_backups_and_exports_share_two_slots_without_blocking_collection(backup_streams, paths):
    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await collect(0)
            deliveries = [Delivery(hold=True), Delivery(hold=True)]
            tasks = []
            try:
                for path, delivery in zip(paths, deliveries):
                    tasks.append(asyncio.create_task(request("GET", path, delivery=delivery)))
                    await wait_for_delivery_ready(tasks[-1], delivery, path)
                    assert delivery.status == 200
                stream_count = len(backup_streams)
                for path in ("/data/backup", "/export/lsj", "/export/lsj/training"):
                    rejected = await request("GET", path)
                    assert rejected.status == 503 and rejected.headers["retry-after"]
                assert len(backup_streams) == stream_count
                await collect(1)
                deliveries[0].release.set()
                await asyncio.wait_for(tasks[0], 5)
                recovered = await request("GET", "/data/backup")
                assert recovered.status == 200 and len(recovered.json()["items"]) == 2
                deliveries[1].release.set()
                await asyncio.wait_for(tasks[1], 5)
                assert_released(backup_streams)
            finally:
                for task, delivery in zip(tasks, deliveries):
                    await finish(task, delivery)
    asyncio.run(scenario())


def test_successful_mixed_export_preparation_can_exceed_old_five_second_wait(backup_streams, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    create_file = export_io._temporary_file
    files = []

    def held_file():
        entered.set()
        assert release.wait(export_io.PREPARE_SECONDS + 5), "Synthetic preparation was not released"
        # Still create the real temporary file and apply real Windows ACLs.
        stream = create_file()
        files.append(stream)
        return stream

    monkeypatch.setattr(export_io, "_temporary_file", held_file)

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await collect(0)
            first, second = Delivery(hold=True), Delivery(hold=True)
            backup = asyncio.create_task(request("GET", "/data/backup", delivery=first))
            export = ready = None
            try:
                await wait_for_delivery_ready(backup, first, "/data/backup")
                export = asyncio.create_task(request("GET", "/export/lsj", delivery=second))
                assert await asyncio.to_thread(entered.wait, export_io.PREPARE_SECONDS + 5)
                ready = asyncio.create_task(wait_for_delivery_ready(export, second, "/export/lsj"))
                # Prove that crossing the old cutoff does not fail a request
                # still within its real preparation budget. This timer releases
                # a controlled gate; there is no unconditional readiness sleep.
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(ready), 5)
                assert not ready.done() and not export.done() and second.status is None
                release.set()
                await ready
                assert second.status == 200 and not api.app.state.operation_lock.locked()
                assert api.app.state.export_slots._value == 0
                for path in ("/data/backup", "/export/lsj", "/export/lsj/training"):
                    assert (await request("GET", path)).status == 503
                await collect(1)
            finally:
                release.set()
                await finish(backup, first)
                if export is not None:
                    await finish(export, second)
                if ready is not None:
                    ready.cancel()
                    with contextlib.suppress(asyncio.CancelledError, AssertionError):
                        await ready
            assert files and all(stream.closed for stream in files)
            assert_released(backup_streams)
    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["send_start", "send_body", "disconnect23", "cancel", "timeout"])
def test_backup_delivery_failure_closes_buffer_and_returns_all_slots(backup_streams, monkeypatch, failure):
    if failure == "timeout":
        monkeypatch.setattr(export_io, "DOWNLOAD_SECONDS", 0.05)

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await collect(0)
            if failure.startswith("send_"):
                with pytest.raises(ClientDisconnect):
                    await request("GET", "/data/backup", delivery=Delivery(fail_at=failure[5:]))
            else:
                disconnected = asyncio.Event()
                slow = Delivery(hold=True)
                task = asyncio.create_task(request("GET", "/data/backup", delivery=slow,
                    asgi_spec="2.3" if failure == "disconnect23" else "2.4", disconnect=disconnected))
                try:
                    await wait_for_delivery_ready(task, slow, "/data/backup")
                    if failure == "disconnect23":
                        disconnected.set()
                        await asyncio.wait_for(task, 5)
                    elif failure == "cancel":
                        task.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await task
                    else:
                        with pytest.raises(TimeoutError):
                            await asyncio.wait_for(asyncio.shield(task), 2)
                        assert task.done(), "The production deadline must end the stalled download"
                finally:
                    await finish(task, slow)
            assert_released(backup_streams)
            assert (await request("GET", "/data/backup")).status == 200
            assert_released(backup_streams)
    asyncio.run(scenario())


@pytest.mark.parametrize("repeat_cancel", [False, True])
@pytest.mark.parametrize("worker_fails", [False, True])
def test_backup_preparation_cancel_waits_for_worker_and_closes_unpublished_buffer(
        backup_streams, monkeypatch, repeat_cancel, worker_fails):
    entered, release = threading.Event(), threading.Event()
    response_type = export_io.PreparedExportResponse

    def paused_response(*args, **kwargs):
        entered.set()
        assert release.wait(5), "Synthetic backup worker was not released"
        if worker_fails:
            raise RuntimeError("Synthetic private backup construction error")
        return response_type(*args, **kwargs)

    monkeypatch.setattr(export_io, "PreparedExportResponse", paused_response)

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await collect(0)
            delivery = Delivery()
            task = asyncio.create_task(request("GET", "/data/backup", delivery=delivery))
            try:
                _, wait_budget = preparation_budgets("/data/backup")
                assert await asyncio.to_thread(entered.wait, wait_budget), (
                    f"Backup worker did not reach response construction: path=/data/backup, status={delivery.status}, "
                    f"request_done={task.done()}, wait_budget={wait_budget}s")
                task.cancel()
                await asyncio.sleep(0)
                if repeat_cancel:
                    task.cancel()
                    await asyncio.sleep(0)
                assert not task.done() and api.app.state.operation_lock.locked()
                assert api.app.state.export_slots._value == 1 and api.app.state.request_slots._value == 3
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 5)
                assert delivery.status is None and not delivery.body
                assert_released(backup_streams)
            finally:
                release.set()
                await finish(task, delivery)
    asyncio.run(scenario())


@pytest.mark.parametrize("token,status", [(None, 401), ("z" * 43, 401), (COLLECTOR, 403)])
def test_unauthorized_backup_does_not_prepare_a_snapshot(backup_streams, token, status):
    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            response = await request("GET", "/data/backup", token=token)
            assert response.status == status and backup_streams == []
            assert api.app.state.export_slots._value == 2 and api.app.state.request_slots._value == 4
    asyncio.run(scenario())


def test_backup_keeps_its_own_exact_byte_limit_including_envelope(backup_streams, monkeypatch):
    monkeypatch.setattr(management.time, "time", lambda: 1790208000)
    monkeypatch.setattr(export_io, "MAX_EXPORT_BYTES", 1)

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            original = await request("GET", "/data/backup")
            assert original.status == 200
            size = len(original.body)
            monkeypatch.setattr(management, "MAX_BACKUP_BYTES", size)
            exact = await request("GET", "/data/backup")
            assert exact.status == 200 and exact.body == original.body
            created = len(backup_streams)
            monkeypatch.setattr(management, "MAX_BACKUP_BYTES", size - 1)
            over = await request("GET", "/data/backup")
            assert over.status == 413 and len(backup_streams) == created
            assert_released(backup_streams)
    asyncio.run(scenario())


def test_backup_keeps_record_limit_and_rejects_invalid_legacy_rows_before_delivery(backup_streams, monkeypatch):
    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await collect(0)
            await collect(1)
            monkeypatch.setattr(management, "MAX_BACKUP_RECORDS", 1)
            assert (await request("GET", "/data/backup")).status == 413
            monkeypatch.setattr(management, "MAX_BACKUP_RECORDS", 10000)
            with db.get_conn() as conn:
                conn.execute("UPDATE items SET title = '' WHERE id = 1")
            assert (await request("GET", "/data/backup")).status == 409
            assert backup_streams == []
            assert api.app.state.export_slots._value == 2 and api.app.state.request_slots._value == 4
            with db.get_conn() as conn:
                assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 2
    asyncio.run(scenario())


def test_backup_uses_default_64k_chunks_with_exact_utf8_length(backup_streams):
    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            with db.get_conn() as conn:
                conn.executemany("INSERT INTO items(url,title,text,ts,source,created_at) VALUES (?,?,?,?,?,?)", [
                    (f"https://example.com/backup/{index}", f"Synthetic 中文 {index}", "文" * 990,
                     BASE_TS + index, "import", BASE_TS) for index in range(100)
                ])
            response = await request("GET", "/data/backup")
            assert response.status == 200 and len(response.body) > 3 * 64 * 1024
            assert int(response.headers["content-length"]) == len(response.body)
            assert len(response.chunks) >= 4 and all(0 < len(chunk) <= 64 * 1024 for chunk in response.chunks)
            payload = response.json()
            assert len(payload["items"]) == 100
            assert payload["sha256"] == hashlib.sha256(management.canonical_items(payload["items"])).hexdigest()
            assert_released(backup_streams)
    asyncio.run(scenario())
