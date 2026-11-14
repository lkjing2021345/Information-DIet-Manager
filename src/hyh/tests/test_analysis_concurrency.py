"""Real ASGI concurrency with temporary SQLite and controlled synthetic inference.

These tests exercise admission, transactions and publication, not real models.
"""
import asyncio
import contextlib
import hashlib
import json
import sqlite3
import threading
from urllib.parse import urlsplit

import pandas as pd
import pytest

from src.hyh import app as api, db
from src.hyh.data_management import canonical_items


BASE_TS = 1790208000000
PRIVATE_MARKER = "SYNTHETIC_PRIVATE_MODEL_ERROR"
ENDPOINTS = {
    "full": ("POST", "/analyze/run_full"),
    "dashboard": ("GET", f"/dashboard/visualization?from_ts={BASE_TS}&to_ts={BASE_TS + 10000}"),
}


@pytest.fixture(autouse=True)
def temporary_database(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "synthetic-analysis-concurrency.sqlite3")
    # Force equal insertion timestamps so cache invalidation cannot rely on time.
    monkeypatch.setattr(api, "_now_ms", lambda: BASE_TS + 10000)


class Response:
    status = None

    def __init__(self):
        self.body = bytearray()
        self.headers = {}

    async def send(self, message):
        if message["type"] == "http.response.start":
            self.status = message["status"]
            self.headers = {key.decode(): value.decode() for key, value in message["headers"]}
        elif message["type"] == "http.response.body":
            self.body.extend(message.get("body", b""))

    def json(self):
        return json.loads(self.body)


async def request(method, url, *, payload=None, headers=None, token="a" * 43):
    parsed = urlsplit(url)
    body = b"" if payload is None else json.dumps(payload).encode()
    raw_headers = [(b"host", b"127.0.0.1"), (b"authorization", ("Bearer " + token).encode())]
    if payload is not None:
        raw_headers.extend([(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())])
    raw_headers.extend((key.lower().encode(), value.encode()) for key, value in (headers or {}).items())
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
             "http_version": "1.1", "method": method, "scheme": "http", "path": parsed.path,
             "raw_path": parsed.path.encode(), "query_string": parsed.query.encode(), "root_path": "",
             "headers": raw_headers, "client": ("127.0.0.1", 51000), "server": ("127.0.0.1", 80)}
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()

    response = Response()
    await api.app(scope, receive, response.send)
    return response


async def collect(index):
    response = await request("POST", "/collect", token="c" * 43, payload={
        "url": f"https://example.invalid/concurrent/{index}", "title": f"Synthetic page {index}",
        "text": f"Synthetic content {index}", "ts": BASE_TS + index, "source": "import", "channel": "edu",
    })
    assert response.status == 200, response.body
    assert response.json()["inserted"] == 1


async def seed(count=5):
    for index in range(count):
        await collect(index)


def full_result(rows):
    count = len(rows)
    return {"input_count": count, "category_counts": {"Tools": count},
            "sentiment_counts": {"Neutral": count}, "comparison_count": max(0, count - 1),
            "sentiment_count": count, "polarity_count": count,
            "repeat_ratio": 0.0 if count > 1 else None,
            "negative_ratio": 0.0 if count else None, "avg_sentiment": 0.0 if count else None,
            "quick_evaluation": {}, "full_report": {}, "pipeline_warning": None}


def dashboard_result(rows):
    frame = pd.DataFrame([{**row, "category": "tools", "sentiment": "Neutral", "polarity": 0.0,
                           "similarity": 0.0, "sentiment_valid": True, "similarity_valid": True}
                          for row in rows])
    return {"ok": True, "input_count": len(rows), "df3": frame}


class ControlledInference:
    def __init__(self, mode, error=None):
        self.mode = mode
        self.error = error
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def __call__(self, rows):
        self.calls.append([dict(row) for row in rows])
        self.entered.set()
        if not self.release.wait(15):
            raise AssertionError("Test did not release synthetic inference")
        if self.error is not None:
            if self.mode == "dashboard":
                return {"ok": False, "input_count": len(rows), "warning": "analysis pipeline failed"}
            raise self.error(PRIVATE_MARKER)
        return full_result(rows) if self.mode == "full" else dashboard_result(rows)

    async def wait(self):
        assert await asyncio.to_thread(self.entered.wait, 5), "Synthetic inference never started"


def install(monkeypatch, mode, error=None):
    inference = ControlledInference(mode, error)
    monkeypatch.setattr(api, "_run_lsj_pipeline" if mode == "full" else "_execute_lsj_pipeline", inference)
    return inference


async def finish(task, inference):
    inference.release.set()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await asyncio.wait_for(asyncio.shield(task), 5)


def rows_from(table):
    assert table in {"analysis_jobs", "analysis_runs", "stats_daily", "items_revision"}
    with db.get_conn() as conn:
        return [dict(row) for row in conn.execute(f"SELECT * FROM {table}")]


def assert_gates_released():
    assert not api.app.state.analysis_lock.locked()
    assert not api.app.state.operation_lock.locked()
    assert api.app.state.request_slots._value == 4


@pytest.mark.parametrize("mode", ENDPOINTS)
def test_analysis_commits_running_job_releases_writer_and_preserves_insert_snapshot(monkeypatch, mode):
    inference = install(monkeypatch, mode)

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await seed()
            task = asyncio.create_task(request(*ENDPOINTS[mode]))
            try:
                await inference.wait()
                jobs = rows_from("analysis_jobs")
                assert len(jobs) == 1 and jobs[0]["status"] == api.JOB_RUNNING
                job_id = jobs[0]["id"]
                assert (await request("GET", f"/analyze/jobs/{job_id}")).json()["status"] == api.JOB_RUNNING
                assert (await request("GET", f"/analyze/result/{job_id}")).status == 409
                # A separate SQLite writer must acquire immediately during inference.
                conn = sqlite3.connect(db.DB_PATH, timeout=0.1)
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    conn.rollback()
                finally:
                    conn.close()
                assert api.app.state.analysis_lock.locked()
                assert not api.app.state.operation_lock.locked()
                await collect(5)
                assert (await request("GET", "/dashboard/summary")).json()["total_count"] == 6
                inference.release.set()
                first = await task
                assert first.status == 200
                first_result = first.json()["result"] if mode == "full" else first.json()
                assert first_result["window"]["input_count"] == 5
                assert len(inference.calls[0]) == 5
                second = await request(*ENDPOINTS[mode])
                assert second.status == 200 and second.json()["cached"] is False
                second_result = second.json()["result"] if mode == "full" else second.json()
                assert second_result["window"]["input_count"] == 6
                third = await request(*ENDPOINTS[mode])
                assert third.status == 200 and third.json()["cached"] is True
                assert len(inference.calls) == 2
            finally:
                await finish(task, inference)
            assert_gates_released()

    asyncio.run(scenario())


async def mutate(kind):
    if kind == "delete":
        response = await request("DELETE", "/items/1", headers={"X-IDM-Confirm": "delete-record"})
    elif kind == "delete_all":
        response = await request("DELETE", "/data", headers={"X-IDM-Confirm": "delete-all"})
    elif kind == "restore":
        items = [{"url": "https://example.invalid/restored", "title": "Restored synthetic record",
                  "text": "Synthetic replacement", "ts": BASE_TS, "source": "import"}]
        response = await request("POST", "/data/restore", headers={"X-IDM-Confirm": "replace-records"}, payload={
            "format": "idm-page-records", "version": 1, "exported_at": BASE_TS, "items": items,
            "sha256": hashlib.sha256(canonical_items(items)).hexdigest(),
        })
    elif kind == "restore_empty":
        response = await request("POST", "/data/restore", headers={"X-IDM-Confirm": "replace-records"}, payload={
            "format": "idm-page-records", "version": 1, "exported_at": BASE_TS, "items": [],
            "sha256": hashlib.sha256(canonical_items([])).hexdigest(),
        })
    else:
        assert kind == "direct_update"
        with db.get_conn() as conn:
            conn.execute("UPDATE items SET title = ? WHERE id = 1", ("Changed synthetic title",))
        return
    assert response.status == 200, response.body


@pytest.mark.parametrize("mutation", ["delete", "delete_all", "restore", "direct_update"])
@pytest.mark.parametrize("error", [None, api.AnalysisUnavailableError, RuntimeError])
def test_full_late_success_and_failure_cannot_publish_after_destructive_change(monkeypatch, mutation, error):
    inference = install(monkeypatch, "full", error)

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await seed()
            task = asyncio.create_task(request(*ENDPOINTS["full"]))
            try:
                await inference.wait()
                await mutate(mutation)
                inference.release.set()
                response = await task
                assert response.status == 409
                assert response.json()["detail"]["code"] == "analysis_snapshot_expired"
                assert not rows_from("analysis_runs") and not rows_from("stats_daily")
                jobs = rows_from("analysis_jobs")
                if mutation == "direct_update":
                    assert len(jobs) == 1 and jobs[0]["status"] == api.JOB_FAILED
                    assert jobs[0]["result_payload"] is None and jobs[0]["error"] == "analysis snapshot expired"
                else:
                    assert jobs == []
                assert PRIVATE_MARKER not in response.body.decode() + json.dumps(jobs)
                # Recovery uses the current records and a new job, never the expired snapshot.
                inference.error = None
                retried = await request(*ENDPOINTS["full"])
                expected_count = {"delete": 4, "delete_all": 0, "restore": 1, "direct_update": 5}[mutation]
                assert retried.status == 200 and retried.json()["cached"] is False
                assert retried.json()["result"]["total_count"] == expected_count
            finally:
                await finish(task, inference)
            assert_gates_released()

    asyncio.run(scenario())


@pytest.mark.parametrize("mutation", ["delete_all", "restore_empty"])
@pytest.mark.parametrize("error", [None, RuntimeError])
def test_clearing_empty_library_removes_job_without_allowing_history_resurrection(monkeypatch, mutation, error):
    inference = install(monkeypatch, "full", error)

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            original_revision = rows_from("items_revision")
            task = asyncio.create_task(request(*ENDPOINTS["full"]))
            try:
                await inference.wait()
                assert rows_from("analysis_jobs")[0]["status"] == api.JOB_RUNNING
                await mutate(mutation)
                # No item DELETE trigger fires here: job absence must invalidate.
                assert rows_from("items_revision") == original_revision
                inference.release.set()
                response = await task
                assert response.status == 409
                assert not rows_from("analysis_jobs") and not rows_from("analysis_runs")
                assert not rows_from("stats_daily")
            finally:
                await finish(task, inference)
            assert_gates_released()

    asyncio.run(scenario())


@pytest.mark.parametrize("first_mode", ENDPOINTS)
def test_full_and_dashboard_share_one_model_gate(monkeypatch, first_mode):
    other_mode = "dashboard" if first_mode == "full" else "full"
    first = install(monkeypatch, first_mode)
    other = install(monkeypatch, other_mode)
    other.release.set()

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await seed()
            task = asyncio.create_task(request(*ENDPOINTS[first_mode]))
            try:
                await first.wait()
                busy = await request(*ENDPOINTS[other_mode], headers={"Origin": "http://localhost:5173"})
                assert busy.status == 503 and busy.headers["retry-after"] == "5"
                assert busy.headers["access-control-allow-origin"] == "http://localhost:5173"
                assert not other.calls
                first.release.set()
                assert (await task).status == 200
                assert (await request(*ENDPOINTS[other_mode])).status == 200
                assert len(first.calls) == len(other.calls) == 1
            finally:
                await finish(task, first)
            assert_gates_released()

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ENDPOINTS)
@pytest.mark.parametrize("error", [None, RuntimeError])
def test_repeated_cancellation_retains_model_ownership_until_worker_finishes(monkeypatch, mode, error):
    inference = install(monkeypatch, mode, error)
    other_mode = "dashboard" if mode == "full" else "full"
    other = install(monkeypatch, other_mode)
    other.release.set()

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await seed()
            task = asyncio.create_task(request(*ENDPOINTS[mode]))
            try:
                await inference.wait()
                task.cancel()
                for _ in range(3):
                    await asyncio.sleep(0)
                assert not task.done() and api.app.state.analysis_lock.locked()
                await collect(5)
                task.cancel()
                for _ in range(3):
                    await asyncio.sleep(0)
                assert not task.done() and not api.app.state.operation_lock.locked()
                assert (await request(*ENDPOINTS[other_mode])).status == 503
                assert not other.calls
                inference.release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
                job = rows_from("analysis_jobs")[0]
                assert job["status"] == (api.JOB_FAILED if error else api.JOB_COMPLETED)
                assert PRIVATE_MARKER not in json.dumps(job)
                assert_gates_released()
                # The gate becomes available only after worker cleanup/publication.
                assert (await request(*ENDPOINTS[other_mode])).status == 200
                assert len(other.calls) == 1
            finally:
                await finish(task, inference)
            assert_gates_released()

    asyncio.run(scenario())


@pytest.mark.parametrize("error,status", [(api.AnalysisUnavailableError, 503), (ValueError, 422), (RuntimeError, 500)])
def test_full_failure_commits_safe_failed_job_and_recovers(monkeypatch, error, status):
    inference = install(monkeypatch, "full", error)

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await seed()
            task = asyncio.create_task(request(*ENDPOINTS["full"]))
            try:
                await inference.wait()
                assert rows_from("analysis_jobs")[0]["status"] == api.JOB_RUNNING
                inference.release.set()
                response = await task
                assert response.status == status and response.json()["status"] == api.JOB_FAILED
                job_id = response.json()["job_id"]
                job_response = await request("GET", f"/analyze/jobs/{job_id}")
                job = job_response.json()
                assert job["status"] == api.JOB_FAILED and job["result_payload"] is None
                history = await request("GET", "/analyze/history")
                assert history.json()["runs"] == []
                assert PRIVATE_MARKER not in (response.body + job_response.body + history.body).decode()
                assert PRIVATE_MARKER not in json.dumps(rows_from("analysis_jobs"))
                inference.error = None
                recovered = await request(*ENDPOINTS["full"])
                assert recovered.status == 200 and recovered.json()["cached"] is False
                assert recovered.json()["job_id"] != job_id
                assert len(rows_from("analysis_runs")) == 1
                assert (await request(*ENDPOINTS["full"])).json()["cached"] is True
                assert len(inference.calls) == 2
            finally:
                await finish(task, inference)
            assert_gates_released()

    asyncio.run(scenario())


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_full_nonfinite_report_cannot_commit_completed_job_before_response_failure(monkeypatch, bad_value):
    def invalid_report(rows):
        result = full_result(rows)
        result["full_report"] = {"synthetic_score": bad_value}
        return result

    monkeypatch.setattr(api, "_run_lsj_pipeline", invalid_report)

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await seed()
            response = await request(*ENDPOINTS["full"])
            assert response.status == 500
            jobs = rows_from("analysis_jobs")
            assert len(jobs) == 1 and jobs[0]["status"] == api.JOB_FAILED
            assert jobs[0]["result_payload"] is None
            assert not rows_from("analysis_runs") and not rows_from("stats_daily")
            assert_gates_released()

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid_json_number", ["NaN", "Infinity", "-Infinity", "1e999"])
@pytest.mark.parametrize("mode", ENDPOINTS)
def test_analysis_nonfinite_legacy_cache_recomputes_without_removing_history(monkeypatch, invalid_json_number, mode):
    calls = []

    def pipeline(rows):
        calls.append(len(rows))
        return full_result(rows) if mode == "full" else dashboard_result(rows)

    monkeypatch.setattr(api, "_run_lsj_pipeline" if mode == "full" else "_execute_lsj_pipeline", pipeline)

    async def scenario():
        async with api.app.router.lifespan_context(api.app):
            await seed()
            assert (await request("POST", "/analyze/run")).status == 200
            initial = await request(*ENDPOINTS[mode])
            assert initial.status == 200 and initial.json()["cached"] is False
            old_job_id = rows_from("analysis_jobs")[0]["id"]
            original_history = rows_from("analysis_runs")
            assert len(original_history) == 1 + int(mode == "full")
            with db.get_conn() as conn:
                payload = json.loads(conn.execute(
                    "SELECT result_payload FROM analysis_jobs WHERE id = ?", (old_job_id,)
                ).fetchone()[0])
                if mode == "full":
                    payload["full_report"] = {"synthetic_score": "INVALID_NUMBER"}
                else:
                    payload["global"]["time_series"][0]["repeat_ratio"] = "INVALID_NUMBER"
                bad_payload = json.dumps(payload).replace('"INVALID_NUMBER"', invalid_json_number)
                conn.execute("UPDATE analysis_jobs SET result_payload = ? WHERE id = ?", (bad_payload, old_job_id))

            recomputed = await request(*ENDPOINTS[mode])
            assert recomputed.status == 200 and recomputed.json()["cached"] is False
            if mode == "full":
                assert recomputed.json()["result"]["full_report"] == {}
            else:
                assert recomputed.json()["global"]["time_series"][0]["repeat_ratio"] == 0.0
            assert calls == [5, 5]
            # Invalid prior cache is ignored, not destructively removed with history.
            jobs = rows_from("analysis_jobs")
            assert len(jobs) == 2 and jobs[0]["id"] == old_job_id
            assert jobs[0]["result_payload"] == bad_payload
            history = rows_from("analysis_runs")
            assert len(history) == len(original_history) + int(mode == "full")
            assert history[:len(original_history)] == original_history
            cached = await request(*ENDPOINTS[mode])
            assert cached.status == 200 and cached.json()["cached"] is True
            assert cached.json()["reused_from_job_id"] == jobs[1]["id"]
            assert calls == [5, 5]
            assert_gates_released()

    asyncio.run(scenario())
