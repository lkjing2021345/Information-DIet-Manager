"""Real full-analysis publication transactions with synthetic inference only."""
from contextlib import contextmanager
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from src.hyh import app as api, db


PRIVATE_MARKER = "SYNTHETIC_PRIVATE_FULL_PUBLICATION"
SAFE_FAILURE = "Legacy analysis failed; no valid result was produced."


def synthetic_pipeline(rows):
    count = len(rows)
    return {"input_count": count, "category_counts": {"Tools": count},
            "sentiment_counts": {"Neutral": count}, "comparison_count": max(0, count - 1),
            "sentiment_count": count, "polarity_count": count,
            "repeat_ratio": 0.0 if count > 1 else None,
            "negative_ratio": 0.0 if count else None, "avg_sentiment": 0.0 if count else None,
            "quick_evaluation": {}, "full_report": {}, "pipeline_warning": None}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "full-publication.sqlite3")
    monkeypatch.setattr(api, "_run_lsj_pipeline", synthetic_pipeline)
    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 51400),
                    headers={"Authorization": "Bearer " + "a" * 43},
                    raise_server_exceptions=False) as result:
        for index in range(2):
            assert result.post("/collect", json={
                "url": f"https://example.invalid/full-publication/{index}", "title": f"Synthetic {index}",
                "text": f"Synthetic text {index}", "ts": index + 1, "source": "import", "channel": "tools",
            }).status_code == 200
        # Preserve a genuine existing basic-statistics/history baseline, rather
        # than only checking that initially empty tables remain empty.
        assert result.post("/analyze/run").status_code == 200
        with db.get_conn() as conn:
            conn.execute("CREATE TABLE synthetic_publication_side_effect (value INTEGER)")
        yield result


def snapshot():
    with db.get_conn() as conn:
        return {name: [dict(row) for row in conn.execute(f"SELECT * FROM {name} ORDER BY rowid")]
                for name in ("analysis_jobs", "analysis_runs", "stats_daily", "synthetic_publication_side_effect")}


def run(client):
    return client.post("/analyze/run_full", params={"from_ts": 0, "to_ts": 10, "force": True})


def install_trigger(failure, *, reject_failed=False):
    condition = "NEW.status IN ('completed', 'failed')" if reject_failed else "NEW.status = 'completed'"
    with db.get_conn() as conn:
        conn.execute(f"""CREATE TRIGGER reject_full_publication AFTER UPDATE OF status ON analysis_jobs
            WHEN {condition} BEGIN
                INSERT INTO synthetic_publication_side_effect VALUES (1);
                SELECT RAISE({failure}, '{PRIVATE_MARKER}');
            END""")


def assert_released():
    assert not api.app.state.work_owner._workers
    assert not api.app.state.analysis_lock.locked()
    assert not api.app.state.operation_lock.locked()
    assert api.app.state.request_slots._value == 4


def assert_history_unchanged(before, after):
    assert len(before["analysis_runs"]) == len(before["stats_daily"]) == 1
    assert after["analysis_runs"] == before["analysis_runs"]
    assert after["stats_daily"] == before["stats_daily"]
    assert after["synthetic_publication_side_effect"] == []


@pytest.mark.parametrize("failure", ["ABORT", "FAIL", "ROLLBACK"])
def test_publication_failure_records_failed_without_partial_results_and_can_retry(client, failure):
    install_trigger(failure)
    before = snapshot()
    response = run(client)
    assert response.status_code == 500
    after = snapshot()
    assert_history_unchanged(before, after)
    assert len(after["analysis_jobs"]) == 1
    job = after["analysis_jobs"][0]
    assert job["status"] == api.JOB_FAILED
    assert job["error"] == SAFE_FAILURE and job["result_payload"] is None
    assert job["finished_at"] is not None and job["duration_ms"] >= 0
    assert response.json() == {"job_id": job["id"], "status": "failed", "detail": SAFE_FAILURE}
    assert PRIVATE_MARKER not in response.text + json.dumps(after)
    assert_released()
    with db.get_conn() as conn:
        conn.execute("DROP TRIGGER reject_full_publication")
    retry = run(client)
    assert retry.status_code == 200 and retry.json()["status"] == api.JOB_COMPLETED
    recovered = snapshot()
    assert recovered["analysis_jobs"][0] == job
    assert recovered["analysis_jobs"][1]["status"] == api.JOB_COMPLETED
    assert recovered["analysis_runs"][:1] == before["analysis_runs"]
    assert len(recovered["analysis_runs"]) == 2
    assert recovered["stats_daily"] == before["stats_daily"]
    assert_released()


def test_success_keeps_prior_history_and_basic_statistics(client):
    before = snapshot()
    response = run(client)
    assert response.status_code == 200
    after = snapshot()
    assert len(after["analysis_runs"]) == 2 and after["analysis_runs"][:1] == before["analysis_runs"]
    assert after["stats_daily"] == before["stats_daily"]
    job = after["analysis_jobs"][0]
    assert job["status"] == api.JOB_COMPLETED and job["error"] is None
    assert json.loads(job["result_payload"]) == response.json()["result"]
    assert after["synthetic_publication_side_effect"] == []
    assert_released()


@pytest.mark.parametrize("mutation", ["revision", "deleted_job"])
def test_whole_rollback_rechecks_snapshot_after_another_writer_changes_it(client, monkeypatch, mutation):
    install_trigger("ROLLBACK")
    before = snapshot()
    original_get_conn = api.get_conn
    intervened = []

    class InterveningConnection:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def execute(self, sql, *args):
            try:
                return self.connection.execute(sql, *args)
            except sqlite3.IntegrityError:
                if "UPDATE analysis_jobs" in sql and not self.connection.in_transaction and not intervened:
                    # Only timing is controlled: an actual SQLite ROLLBACK has
                    # released the writer, so another actual connection wins.
                    with db.get_conn() as other:
                        if mutation == "revision":
                            other.execute("UPDATE items SET text = 'Synthetic concurrent edit' WHERE id = 1")
                        else:
                            other.execute("DELETE FROM analysis_jobs")
                    intervened.append(mutation)
                raise

    @contextmanager
    def intervening_connections():
        with original_get_conn() as connection:
            yield InterveningConnection(connection)

    monkeypatch.setattr(api, "get_conn", intervening_connections)
    response = run(client)
    assert intervened == [mutation]
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "analysis_snapshot_expired"
    after = snapshot()
    assert_history_unchanged(before, after)
    if mutation == "revision":
        assert len(after["analysis_jobs"]) == 1
        job = after["analysis_jobs"][0]
        assert job["status"] == api.JOB_FAILED and job["error"] == "analysis snapshot expired"
        assert job["result_payload"] is None
    else:
        assert after["analysis_jobs"] == []
    assert PRIVATE_MARKER not in response.text + json.dumps(after)
    assert_released()


def test_database_refusing_failure_write_does_not_claim_persisted_terminal_status(client):
    install_trigger("ROLLBACK", reject_failed=True)
    before = snapshot()
    response = run(client)
    assert response.status_code == 500
    after = snapshot()
    assert_history_unchanged(before, after)
    job = after["analysis_jobs"][0]
    assert job["status"] == api.JOB_RUNNING and job["result_payload"] is None
    assert job["finished_at"] is None
    assert response.json() == {"detail": "Local data operation failed"}
    assert PRIVATE_MARKER not in response.text + json.dumps(after)
    assert_released()
