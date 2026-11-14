"""Real ASGI delivery with synthetic data and controlled slow/disconnected clients."""
import asyncio
import contextlib
import csv
import io
import json
import threading
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from starlette.requests import ClientDisconnect

from src.hyh import app as api, db


ADMIN = "a" * 43
COLLECTOR = "c" * 43
BASE_TS = 1790208000000
READINESS_SCHEDULING_MARGIN_SECONDS = 5
SAFE_DELIVERY_FAILURES = {
    "Request capacity reached; retry later": "request_capacity",
    "Export capacity reached; retry later": "export_capacity",
    "Service busy; retry later": "operation_busy",
    "Service is shutting down; retry later": "service_shutdown",
    "Export preparation timed out; select a smaller window": "export_preparation_timeout",
    "Export database unavailable; retry later": "export_database_unavailable",
    "Export temporary storage unavailable": "export_storage_unavailable",
    "Backup preparation timed out; retry later": "backup_preparation_timeout",
    "Backup database unavailable; retry later": "backup_database_unavailable",
}


def safe_delivery_failure(body):
    # Capture only an exact, fixed product diagnostic before a synthetic slow
    # consumer blocks. Never include arbitrary response bytes in CI assertions.
    if len(body) > 4096:
        return "unclassified"
    try:
        payload = json.loads(body)
    except (ValueError, TypeError, RecursionError):
        return "unclassified"
    detail = payload.get("detail") if isinstance(payload, dict) else None
    return SAFE_DELIVERY_FAILURES.get(detail, "unclassified") if isinstance(detail, str) else "unclassified"


@pytest.fixture
def export_files(tmp_path, monkeypatch):
    from src.hyh import export_io

    files = []
    create_file = export_io._temporary_file

    def temporary_file():
        # Exercise the real Windows private-ACL setup before any synthetic text.
        stream = create_file()
        files.append(stream)
        return stream

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "synthetic-export-delivery.sqlite3")
    monkeypatch.setattr(export_io, "_temporary_file", temporary_file)
    yield files
    # Keep a failing test from leaving synthetic temporary handles open.
    for stream in files:
        stream.close()


class Delivery:
    def __init__(self, *, hold=False, fail_at=None):
        self.hold = hold
        self.fail_at = fail_at
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.status = None
        self.failure_kind = None
        self.headers = {}
        self.body = bytearray()
        self.chunks = []

    async def send(self, message):
        if message["type"] == "http.response.start":
            self.status = message["status"]
            self.headers = {key.decode().lower(): value.decode() for key, value in message["headers"]}
            if self.fail_at == "start":
                raise OSError("Synthetic response-start disconnect")
        elif message["type"] == "http.response.body":
            if message.get("body"):
                if self.status != 200 and self.failure_kind is None:
                    self.failure_kind = safe_delivery_failure(message["body"])
                self.entered.set()
                if self.fail_at == "body":
                    raise OSError("Synthetic response-body disconnect")
                if self.hold:
                    await self.release.wait()
            self.body.extend(message.get("body", b""))
            if message.get("body"):
                self.chunks.append(message["body"])

    def json(self):
        return json.loads(self.body)


async def request(method, url, *, payload=None, token=ADMIN, headers=None, delivery=None,
                  asgi_spec="2.4", disconnect=None):
    delivery = delivery or Delivery()
    parsed = urlsplit(url)
    body = json.dumps(payload).encode() if payload is not None else b""
    raw_headers = [(b"host", b"127.0.0.1")]
    if token is not None:
        raw_headers.append((b"authorization", ("Bearer " + token).encode()))
    if payload is not None:
        raw_headers.extend([(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())])
    raw_headers.extend((key.lower().encode(), value.encode()) for key, value in (headers or {}).items())
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": asgi_spec},
             "http_version": "1.1", "method": method, "scheme": "http", "path": parsed.path,
             "raw_path": parsed.path.encode(), "query_string": parsed.query.encode(), "root_path": "",
             "headers": raw_headers, "client": ("127.0.0.1", 51000), "server": ("127.0.0.1", 80)}
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        if disconnect is not None:
            await disconnect.wait()
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()

    await api.app(scope, receive, delivery.send)
    return delivery


async def collect(index):
    response = await request("POST", "/collect", token=COLLECTOR, payload={
        "url": f"https://example.com/delivery/{index}", "title": f"Synthetic page {index}",
        "text": f"Synthetic snapshot {index}", "ts": BASE_TS + index, "source": "import", "channel": "edu",
    })
    assert response.status == 200 and response.json()["inserted"] == 1


def preparation_budgets(path):
    """Return product preparation time and its test observation allowance."""
    from src.hyh import data_management, export_io

    prepare_seconds = (data_management.BACKUP_PREPARE_SECONDS
        if urlsplit(path).path == "/data/backup" else export_io.PREPARE_SECONDS)
    return prepare_seconds, prepare_seconds + READINESS_SCHEDULING_MARGIN_SECONDS


async def wait_for_delivery_ready(task, delivery, path):
    """Wait for HTTP 200 body readiness or fail when the request ends first.

    Preparation has its own product deadline. The extra margin covers admission
    and CI scheduling, not permission for preparation to exceed its real budget.
    Only the event waiter is cancelled here; callers still own request cleanup.
    """
    prepare_seconds, timeout = preparation_budgets(path)
    ready = asyncio.create_task(delivery.entered.wait())
    try:
        await asyncio.wait({task, ready}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        state = (f"path={path}, status={delivery.status}, body_event={delivery.entered.is_set()}, "
                 f"request_done={task.done()}, prepare_budget={prepare_seconds}s, wait_budget={timeout}s, "
                 f"failure_kind={delivery.failure_kind}")
        if delivery.status is not None and delivery.status != 200:
            raise AssertionError(f"Delivery returned non-success HTTP status: {state}")
        if delivery.entered.is_set() and delivery.status == 200:
            # Readiness already happened, even if an intentional short download
            # deadline fired before the event loop resumed this observer. The
            # caller still verifies the request's final result/exception.
            return
        if task.done():
            if task.cancelled():
                raise AssertionError(f"Delivery request cancelled before readiness: {state}")
            error = task.exception()
            if error is not None:
                raise AssertionError(f"Delivery request failed ({type(error).__name__}): {state}") from error
        if task.done():
            raise AssertionError(f"Delivery request ended before successful body readiness: {state}")
        raise AssertionError(f"Delivery readiness timed out: {state}")
    finally:
        ready.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ready


async def finish(task, delivery):
    delivery.release.set()
    if not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), 5)
        except BaseException:
            task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


def assert_released(files):
    assert files and all(stream.closed for stream in files)
    assert all(not isinstance(stream.name, str) or not Path(stream.name).exists() for stream in files)
    assert not api.app.state.operation_lock.locked()
    assert api.app.state.export_slots._value == 2
    assert api.app.state.request_slots._value == 4


@pytest.mark.parametrize("detail,kind", list(SAFE_DELIVERY_FAILURES.items()))
def test_readiness_classifies_fixed_error_before_slow_body_is_released(detail, kind):
    async def scenario():
        delivery = Delivery(hold=True)

        async def failed_request():
            await delivery.send({"type": "http.response.start", "status": 503, "headers": []})
            await delivery.send({"type": "http.response.body", "body": json.dumps({"detail": detail}).encode()})

        task = asyncio.create_task(failed_request())
        try:
            with pytest.raises(AssertionError, match="failure_kind=" + kind):
                await wait_for_delivery_ready(task, delivery, "/export/lsj")
            assert delivery.entered.is_set() and not task.done() and not delivery.body
            assert delivery.failure_kind == kind
        finally:
            await finish(task, delivery)

    asyncio.run(scenario())


@pytest.mark.parametrize("body", [
    b'{"detail":"SYNTHETIC_PRIVATE_TOKEN"}',
    b'{"detail":{"private":"SYNTHETIC_PRIVATE_TOKEN"}}',
    b'["SYNTHETIC_PRIVATE_TOKEN"]',
    b'SYNTHETIC_PRIVATE_TOKEN invalid JSON',
    b'{"detail":"Export preparation timed out; select a smaller window SYNTHETIC_PRIVATE_TOKEN"}',
    b'{"detail":"SYNTHETIC_PRIVATE_TOKEN' + b'x' * 4096 + b'"}',
    b'[' * 1500 + b'"SYNTHETIC_PRIVATE_TOKEN"' + b']' * 1500,
], ids=["text", "object", "array", "malformed", "known-prefix-private-suffix", "oversized", "deeply-nested"])
def test_readiness_never_reflects_unknown_error_body(body):
    async def scenario():
        delivery = Delivery(hold=True)

        async def failed_request():
            await delivery.send({"type": "http.response.start", "status": 503, "headers": []})
            await delivery.send({"type": "http.response.body", "body": body})

        task = asyncio.create_task(failed_request())
        try:
            with pytest.raises(AssertionError, match="failure_kind=unclassified") as raised:
                await wait_for_delivery_ready(task, delivery, "/export/lsj")
            assert "SYNTHETIC_PRIVATE_TOKEN" not in str(raised.value)
            assert not delivery.body and not task.done()
        finally:
            await finish(task, delivery)

    asyncio.run(scenario())


@pytest.mark.parametrize("resource,capacity,kind", [("export_slots", 2, "export_capacity"),
                                                    ("request_slots", 4, "request_capacity")])
def test_readiness_distinguishes_real_admission_rejection(export_files, resource, capacity, kind):
    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            slots = getattr(api.app.state, resource)
            for _ in range(capacity):
                await slots.acquire()
            delivery = Delivery(hold=True)
            task = asyncio.create_task(request("GET", "/export/lsj", delivery=delivery))
            try:
                with pytest.raises(AssertionError, match="failure_kind=" + kind):
                    await wait_for_delivery_ready(task, delivery, "/export/lsj")
                assert delivery.status == 503 and not task.done() and not delivery.body
                assert export_files == [] and not api.app.state.operation_lock.locked()
            finally:
                await finish(task, delivery)
                for _ in range(capacity):
                    slots.release()
            assert api.app.state.export_slots._value == 2 and api.app.state.request_slots._value == 4

    asyncio.run(scenario())


@pytest.mark.parametrize("ending", ["error", "cancelled", "empty_success"])
def test_readiness_reports_request_ending_before_successful_body(ending):
    async def scenario():
        delivery = Delivery()

        async def end_request():
            if ending == "error":
                raise RuntimeError("Synthetic request failure")
            if ending == "cancelled":
                raise asyncio.CancelledError()
            delivery.status = 200

        task = asyncio.create_task(end_request())
        try:
            with pytest.raises(AssertionError, match="path=/export/lsj.*request_done=True"):
                await asyncio.wait_for(wait_for_delivery_ready(task, delivery, "/export/lsj"), 1)
        finally:
            with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                await task
    asyncio.run(scenario())


def test_readiness_rejects_real_preparation_error_without_waiting_for_success(export_files, monkeypatch):
    from src.hyh import export_io

    monkeypatch.setattr(export_io, "PREPARE_SECONDS", 0)
    # This case isolates the readiness helper's response-error branch. Real
    # Windows ACL creation is exercised by the controlled slow-preparation case.
    stream = io.BytesIO()
    monkeypatch.setattr(export_io, "_temporary_file", lambda: stream)

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            delivery = Delivery(hold=True)
            task = asyncio.create_task(request("GET", "/export/lsj", delivery=delivery))
            try:
                with pytest.raises(AssertionError, match="non-success HTTP status:.*status=503"):
                    await asyncio.wait_for(wait_for_delivery_ready(task, delivery, "/export/lsj"), 1)
                assert delivery.failure_kind == "export_preparation_timeout"
                assert stream.closed
            finally:
                await finish(task, delivery)
            assert not api.app.state.operation_lock.locked()
            assert api.app.state.export_slots._value == 2
            assert api.app.state.request_slots._value == 4
    asyncio.run(scenario())


def test_readiness_timeout_does_not_cancel_request_or_hide_state(monkeypatch):
    from src.hyh import export_io

    monkeypatch.setattr(export_io, "PREPARE_SECONDS", 0)
    monkeypatch.setitem(globals(), "READINESS_SCHEDULING_MARGIN_SECONDS", 0.01)

    async def scenario():
        task = asyncio.create_task(asyncio.Event().wait())
        try:
            with pytest.raises(AssertionError, match="readiness timed out:.*status=None.*request_done=False"):
                await wait_for_delivery_ready(task, Delivery(), "/export/lsj")
            assert not task.done()
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    asyncio.run(scenario())


@pytest.mark.parametrize("endpoint", ["/export/lsj", "/export/lsj/training"])
def test_slow_export_allows_collection_delete_restore_and_keeps_its_prepared_snapshot(export_files, endpoint):
    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await collect(0)
            slow = Delivery(hold=True)
            download = asyncio.create_task(request("GET", endpoint, delivery=slow))
            try:
                await wait_for_delivery_ready(download, slow, endpoint)
                assert slow.status == 200 and not api.app.state.operation_lock.locked()
                await collect(1)
                backup = await request("GET", "/data/backup")
                assert backup.status == 200 and len(backup.json()["items"]) == 2
                deleted = await request("DELETE", "/data", headers={"X-IDM-Confirm": "delete-all"})
                assert deleted.status == 200 and deleted.json()["deleted"] == 2
                restored = await request("POST", "/data/restore", payload=backup.json(),
                                         headers={"X-IDM-Confirm": "replace-records"})
                assert restored.status == 200 and restored.json()["restored"] == 2
                assert not download.done()
                slow.release.set()
                await asyncio.wait_for(download, 5)
                payload = slow.json()
                rows = payload if endpoint.endswith("training") else payload["items"]
                assert len(rows) == 1 and rows[0]["title"] == "Synthetic page 0"
                assert_released(export_files)
            finally:
                await finish(download, slow)
    asyncio.run(scenario())


def test_two_slow_downloads_reject_third_without_building_more_files_then_recover(export_files):
    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await collect(0)
            deliveries = [Delivery(hold=True), Delivery(hold=True)]
            tasks = []
            try:
                for endpoint, delivery in zip(("/export/lsj?fmt=jsonl", "/export/lsj/training?fmt=csv"), deliveries):
                    tasks.append(asyncio.create_task(request("GET", endpoint, delivery=delivery)))
                    await wait_for_delivery_ready(tasks[-1], delivery, endpoint)
                    assert delivery.status == 200
                file_count = len(export_files)
                rejected = await request("GET", "/export/lsj")
                assert rejected.status == 503 and rejected.headers["retry-after"]
                assert len(export_files) == file_count
                # Non-export writes still have admission capacity while both downloads wait.
                await collect(1)
                deliveries[0].release.set()
                await asyncio.wait_for(tasks[0], 5)
                recovered = await request("GET", "/export/lsj/training")
                assert recovered.status == 200 and len(recovered.json()) == 2
                deliveries[1].release.set()
                await asyncio.wait_for(tasks[1], 5)
                assert_released(export_files)
            finally:
                for task, delivery in zip(tasks, deliveries):
                    await finish(task, delivery)
    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["start", "body"])
def test_send_disconnect_closes_files_and_releases_admission(export_files, phase):
    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await collect(0)
            with pytest.raises(ClientDisconnect):
                await request("GET", "/export/lsj?fmt=jsonl", delivery=Delivery(fail_at=phase))
            assert_released(export_files)
            assert (await request("GET", "/export/lsj/training?fmt=csv")).status == 200
            assert_released(export_files)
    asyncio.run(scenario())


def test_cancelled_download_closes_files_and_releases_admission(export_files):
    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await collect(0)
            slow = Delivery(hold=True)
            task = asyncio.create_task(request("GET", "/export/lsj?fmt=csv", delivery=slow))
            try:
                await wait_for_delivery_ready(task, slow, "/export/lsj?fmt=csv")
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert_released(export_files)
                assert (await request("GET", "/export/lsj")).status == 200
                assert_released(export_files)
            finally:
                await finish(task, slow)
    asyncio.run(scenario())


def test_download_deadline_closes_files_and_releases_admission(export_files, monkeypatch):
    from src.hyh import export_io

    monkeypatch.setattr(export_io, "DOWNLOAD_SECONDS", 0.05)

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await collect(0)
            slow = Delivery(hold=True)
            task = asyncio.create_task(request("GET", "/export/lsj?fmt=jsonl", delivery=slow))
            try:
                await wait_for_delivery_ready(task, slow, "/export/lsj?fmt=jsonl")
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(task), 2)
                # Shielding means the outer test timeout cannot supply production cleanup.
                assert task.done(), "Download deadline did not stop the stalled sender"
                assert_released(export_files)
            finally:
                await finish(task, slow)
    asyncio.run(scenario())


@pytest.mark.parametrize("endpoint", ["/export/lsj", "/export/lsj/training"])
@pytest.mark.parametrize("token,status", [(None, 401), ("z" * 43, 401), (COLLECTOR, 403)])
def test_unauthorized_export_never_creates_a_temporary_file(export_files, endpoint, token, status):
    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            response = await request("GET", endpoint, token=token)
            assert response.status == status and export_files == []
            assert not api.app.state.operation_lock.locked()
            assert api.app.state.export_slots._value == 2
            assert api.app.state.request_slots._value == 4
    asyncio.run(scenario())


@pytest.mark.parametrize("repeat_cancel", [False, True])
@pytest.mark.parametrize("worker_fails", [False, True])
def test_preparation_cancel_waits_for_worker_and_closes_result_without_sending(
        export_files, monkeypatch, repeat_cancel, worker_fails):
    from src.hyh import export_io

    entered, release = threading.Event(), threading.Event()
    encode = export_io._encode

    def paused_encode(*args, **kwargs):
        entered.set()
        assert release.wait(5), "Synthetic preparation was not released"
        if worker_fails:
            raise OSError("Synthetic storage failure after caller cancellation")
        return encode(*args, **kwargs)

    monkeypatch.setattr(export_io, "_encode", paused_encode)

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await collect(0)
            delivery = Delivery()
            task = asyncio.create_task(request("GET", "/export/lsj", delivery=delivery))
            try:
                _, wait_budget = preparation_budgets("/export/lsj")
                assert await asyncio.to_thread(entered.wait, wait_budget), (
                    f"Export worker did not reach encoding: path=/export/lsj, status={delivery.status}, "
                    f"request_done={task.done()}, wait_budget={wait_budget}s")
                task.cancel()
                await asyncio.sleep(0)  # Let the owner enter its cancellation cleanup.
                if repeat_cancel:
                    task.cancel()
                    await asyncio.sleep(0)
                # A cancelled caller must not free admission while its worker
                # still owns a temporary file and could be reading the database.
                assert not task.done()
                assert api.app.state.operation_lock.locked()
                assert api.app.state.export_slots._value == 1
                assert api.app.state.request_slots._value == 3
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 5)
                assert delivery.status is None and not delivery.body
                assert_released(export_files)
            finally:
                release.set()
                await finish(task, delivery)
    asyncio.run(scenario())


def test_temporary_file_read_failure_closes_files_and_releases_admission(export_files, monkeypatch):
    from src.hyh import export_io

    create_file = export_io._temporary_file

    class UnreadableFile:
        def __init__(self):
            self.file = create_file()

        def __getattr__(self, name):
            return getattr(self.file, name)

        def read(self, *args):
            raise OSError("Synthetic export read failure")

    monkeypatch.setattr(export_io, "_temporary_file", UnreadableFile)

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await collect(0)
            delivery = Delivery()
            with pytest.raises(ClientDisconnect):
                await request("GET", "/export/lsj", delivery=delivery)
            # A late storage failure must terminate the response, never append
            # an error JSON object to a successful export body.
            assert delivery.status == 200 and not delivery.body
            assert_released(export_files)
    asyncio.run(scenario())


@pytest.mark.parametrize("endpoint", ["/export/lsj", "/export/lsj/training"])
@pytest.mark.parametrize("fmt,media_type", [("json", "application/json"), ("jsonl", "application/x-ndjson"),
                                           ("csv", "text/csv")])
def test_download_headers_and_bounded_chunks_match_complete_utf8_body(export_files, monkeypatch, endpoint, fmt, media_type):
    from src.hyh import export_io

    monkeypatch.setattr(export_io, "CHUNK_BYTES", 37)

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            saved = await request("POST", "/collect", token=COLLECTOR, payload={
                "url": "https://example.com/unicode", "title": "合成测试标题", "text": "中文内容" * 50,
                "ts": BASE_TS, "source": "import", "channel": "edu",
            })
            assert saved.status == 200
            response = await request("GET", endpoint + "?fmt=" + fmt,
                                     headers={"Origin": "http://127.0.0.1:5173"})
            assert response.status == 200
            assert response.headers["content-type"].startswith(media_type)
            assert int(response.headers["content-length"]) == len(response.body)
            assert response.headers["cache-control"] == "no-store"
            assert response.headers["x-content-type-options"] == "nosniff"
            assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
            assert len(response.chunks) > 1 and all(0 < len(chunk) <= 37 for chunk in response.chunks)
            text = response.body.decode("utf-8")
            if fmt == "json":
                assert "content-disposition" not in response.headers
                payload = json.loads(text)
                rows = payload if endpoint.endswith("training") else payload["items"]
            else:
                stem = "lsj_training" if endpoint.endswith("training") else "lsj_export_analysis"
                assert response.headers["content-disposition"] == f'attachment; filename="{stem}.{fmt}"'
                if fmt == "jsonl":
                    assert not text.endswith("\n")
                    rows = [json.loads(line) for line in text.splitlines()]
                else:
                    assert text.endswith("\r\n")
                    rows = list(csv.DictReader(io.StringIO(text)))
            assert len(rows) == 1 and rows[0]["title"] == "合成测试标题"
            assert_released(export_files)
    asyncio.run(scenario())


@pytest.mark.parametrize("endpoint", ["/export/lsj", "/export/lsj/training"])
def test_asgi23_disconnect_event_closes_files_and_releases_admission(export_files, endpoint):
    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await collect(0)
            disconnected = asyncio.Event()
            slow = Delivery(hold=True)
            task = asyncio.create_task(request("GET", endpoint, delivery=slow,
                                               asgi_spec="2.3", disconnect=disconnected))
            try:
                await wait_for_delivery_ready(task, slow, endpoint)
                disconnected.set()
                # Uvicorn's ASGI 2.3 receive-disconnect branch cancels the
                # streaming task internally and returns without an exception.
                await asyncio.wait_for(task, 5)
                assert slow.status == 200 and not slow.body
                assert_released(export_files)
                assert (await request("GET", endpoint)).status == 200
                assert_released(export_files)
            finally:
                await finish(task, slow)
    asyncio.run(scenario())


def test_default_delivery_bounds_chunks_to_64k_and_reports_exact_total_bytes(export_files):
    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            with db.get_conn() as conn:
                conn.executemany("INSERT INTO items(url,title,text,ts,source,created_at) VALUES (?,?,?,?,?,?)", [
                    (f"https://example.com/chunks/{index}", f"Synthetic chunk {index}", "文" * 990,
                     BASE_TS + index, "import", BASE_TS) for index in range(70)
                ])
            response = await request("GET", "/export/lsj?fmt=jsonl")
            assert response.status == 200
            assert len(response.body) > 3 * 64 * 1024
            assert int(response.headers["content-length"]) == len(response.body)
            assert len(response.chunks) >= 4 and all(0 < len(chunk) <= 64 * 1024 for chunk in response.chunks)
            rows = [json.loads(line) for line in response.body.decode("utf-8").splitlines()]
            assert len(rows) == 70 and [row["ts"] for row in rows] == list(range(BASE_TS, BASE_TS + 70))
            assert_released(export_files)
    asyncio.run(scenario())
