"""Real ASGI/SQLite publication failures; empty windows never load models."""
import json

import pytest
from fastapi.testclient import TestClient

from src.hyh import app as api, db


PRIVATE_MARKER = "SYNTHETIC_PRIVATE_PUBLICATION_DETAIL"
SAFE_FAILURE = "Visualization analysis failed; no valid result was produced."


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "dashboard-publication.sqlite3")

    def no_model(*_args, **_kwargs):
        raise AssertionError("Empty synthetic windows must not load models")

    monkeypatch.setattr(api, "_execute_lsj_pipeline", no_model)
    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 51234),
                    headers={"Authorization": "Bearer " + "a" * 43},
                    raise_server_exceptions=False) as result:
        yield result


def run(client):
    return client.get("/dashboard/visualization", params={"from_ts": 0, "to_ts": 10})


def jobs():
    with db.get_conn() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM analysis_jobs ORDER BY id")]


def assert_released():
    assert not api.app.state.work_owner._workers
    assert not api.app.state.analysis_lock.locked()
    assert not api.app.state.operation_lock.locked()
    assert api.app.state.request_slots._value == 4


def assert_safe_failure(response):
    assert response.status_code == 500
    stored = jobs()
    assert len(stored) == 1
    job = stored[0]
    assert job["status"] == api.JOB_FAILED
    assert job["error"] == SAFE_FAILURE
    assert job["finished_at"] is not None and job["duration_ms"] >= 0
    assert job["result_payload"] is None
    assert response.json() == {"job_id": job["id"], "status": "failed", "detail": SAFE_FAILURE}
    assert PRIVATE_MARKER not in response.text + json.dumps(stored)
    with db.get_conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM stats_daily").fetchone()[0] == 0
    assert_released()


@pytest.mark.parametrize("failure", ["ABORT", "FAIL", "ROLLBACK"])
def test_sqlite_publication_failure_rolls_back_partial_writes_and_records_failed(client, failure):
    with db.get_conn() as conn:
        conn.execute("CREATE TABLE synthetic_publication_side_effect (detail TEXT)")
        # FAIL deliberately retains statement changes unless our savepoint rolls
        # them back; ROLLBACK ends the outer transaction and removes savepoints.
        conn.execute(f"""CREATE TRIGGER reject_publication AFTER UPDATE OF status ON analysis_jobs
            WHEN NEW.status = 'completed' BEGIN
                INSERT INTO synthetic_publication_side_effect VALUES ('{PRIVATE_MARKER}');
                SELECT RAISE({failure}, '{PRIVATE_MARKER}');
            END""")
    assert_safe_failure(run(client))
    with db.get_conn() as conn:
        assert list(conn.execute("SELECT * FROM synthetic_publication_side_effect")) == []
        conn.execute("DROP TRIGGER reject_publication")
    retry = run(client)
    assert retry.status_code == 200 and retry.json()["analysis_status"] == "empty"
    assert [job["status"] for job in jobs()] == [api.JOB_FAILED, api.JOB_COMPLETED]
    assert jobs()[0]["result_payload"] is None
    assert_released()


@pytest.mark.parametrize("kind", ["nonfinite", "surrogate", "object", "depth"])
def test_unserializable_publication_cannot_leave_running_or_completed_job(client, monkeypatch, kind):
    original = api._build_visualization_result
    invalid = {"nonfinite": float("nan"), "surrogate": "\ud800", "object": object()}.get(kind)
    if kind == "depth":
        invalid = 0
        for _ in range(65):
            invalid = [invalid]

    def malformed_result(*args, **kwargs):
        payload = original(*args, **kwargs)
        payload["synthetic_extra"] = {"private": PRIVATE_MARKER, "invalid": invalid}
        return payload

    monkeypatch.setattr(api, "_build_visualization_result", malformed_result)
    assert_safe_failure(run(client))


@pytest.mark.parametrize("change", ["revision", "deleted_job"])
def test_expired_snapshot_remains_409_before_publication_validation(client, monkeypatch, change):
    original = api._build_visualization_result

    def expire_then_return_bad_payload(*args, **kwargs):
        payload = original(*args, **kwargs)
        with db.get_conn() as conn:
            if change == "revision":
                conn.execute("UPDATE items_revision SET revision = revision + 1")
            else:
                conn.execute("DELETE FROM analysis_jobs")
        payload["synthetic_extra"] = {"private": PRIVATE_MARKER, "invalid": object()}
        return payload

    monkeypatch.setattr(api, "_build_visualization_result", expire_then_return_bad_payload)
    response = run(client)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "analysis_snapshot_expired"
    stored = jobs()
    if change == "revision":
        assert len(stored) == 1 and stored[0]["status"] == api.JOB_FAILED
        assert stored[0]["error"] == "analysis snapshot expired"
        assert stored[0]["result_payload"] is None
    else:
        assert stored == []
    assert PRIVATE_MARKER not in response.text + json.dumps(stored)
    assert_released()


def test_database_rejecting_even_failure_status_does_not_claim_persistence(client):
    with db.get_conn() as conn:
        conn.execute(f"""CREATE TRIGGER reject_all_terminal_updates BEFORE UPDATE OF status ON analysis_jobs
            WHEN NEW.status IN ('completed', 'failed') BEGIN
                SELECT RAISE(ABORT, '{PRIVATE_MARKER}');
            END""")
    response = run(client)
    assert response.status_code == 500
    assert PRIVATE_MARKER not in response.text
    # With both writes refused we cannot persist a terminal state. Startup
    # recovery remains available after the database becomes writable again.
    stored = jobs()
    assert len(stored) == 1 and stored[0]["status"] == api.JOB_RUNNING
    assert stored[0]["result_payload"] is None
    assert_released()


def test_normal_empty_publication_keeps_response_and_stored_result(client):
    response = run(client)
    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis_status"] == "empty" and payload["cached"] is False
    assert payload["window"]["input_count"] == payload["window"]["available_count"] == 0
    stored = jobs()
    assert len(stored) == 1 and stored[0]["status"] == api.JOB_COMPLETED
    assert stored[0]["error"] is None
    assert json.loads(stored[0]["result_payload"]) == payload
    assert_released()


def test_ready_publication_preserves_hour_keys_and_cache(client, monkeypatch):
    import pandas as pd

    calls = []

    def synthetic_inference(rows):
        calls.append(len(rows))
        frame = pd.DataFrame([{**row, "category": "Tools", "sentiment": "Neutral", "polarity": 0.0,
                               "similarity": 0.0, "sentiment_valid": True, "similarity_valid": True}
                              for row in rows])
        return {"ok": True, "input_count": len(rows), "df3": frame}

    monkeypatch.setattr(api, "_execute_lsj_pipeline", synthetic_inference)
    for index in range(5):
        assert client.post("/collect", json={
            "url": f"https://example.invalid/publication/{index}", "title": f"Synthetic {index}",
            "text": f"Synthetic publication content {index}", "ts": index + 1, "source": "import",
        }).status_code == 200
    response = run(client)
    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis_status"] == "ready" and payload["cached"] is False
    assert payload["global"]["hourly_distribution"] == {"0": 5}
    assert json.loads(jobs()[0]["result_payload"]) == payload
    cached = run(client)
    assert cached.status_code == 200 and cached.json()["cached"] is True
    assert cached.json()["global"] == payload["global"] and calls == [5]
    assert_released()
