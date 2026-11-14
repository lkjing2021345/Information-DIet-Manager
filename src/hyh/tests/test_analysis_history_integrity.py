"""Stored analysis corruption must not masquerade as valid missing measurements.

All database state is synthetic. Empty full-analysis windows avoid model calls;
window-cache cases use controlled synthetic outputs, not real model inference.
"""
import json
import math

import pytest
from fastapi.testclient import TestClient

from src.hyh import app as api, db


PRIVATE_MARKER = "SYNTHETIC_PRIVATE_REPORT_CONTENT"
INVALID_NUMBERS = ("NaN", "Infinity", "-Infinity", "1e999")
INVALID_JSON = (*INVALID_NUMBERS, "malformed", "serializer_depth", "deep")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "analysis-history-integrity.sqlite3")
    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000),
                    headers={"Authorization": "Bearer " + "a" * 43}, raise_server_exceptions=False) as client:
        yield client


def corrupted_json(kind):
    if kind == "malformed":
        return '{"private":"' + PRIVATE_MARKER + '","measurement":'
    if kind in ("serializer_depth", "deep"):
        depth = 300 if kind == "serializer_depth" else 2000
        encoded = '{"private":"' + PRIVATE_MARKER + '","measurement":' + "[" * depth + "0" + "]" * depth + "}"
        if kind == "serializer_depth":
            # Finite JSON can exceed the HTTP serializer's depth limit even
            # when the standard-library parser and strict encoder accept it.
            json.dumps(json.loads(encoded), allow_nan=False)
        return encoded
    return '{"private":"' + PRIVATE_MARKER + '","measurement":' + kind + "}"


def nested_containers(depth):
    value = 0
    for level in range(depth):
        value = {"nested": value} if level % 2 == 0 else [value]
    return value


def snapshot():
    with db.get_conn() as conn:
        return {table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
                for table in ("items", "analysis_jobs", "analysis_runs", "stats_daily")}


def assert_stored_invalid(response):
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "stored_analysis_invalid"
    assert PRIVATE_MARKER not in response.text
    assert "no-store" in response.headers["cache-control"]


def empty_job(client):
    response = client.post("/analyze/run_full")
    assert response.status_code == 200
    assert response.json()["result"]["analysis_status"] == "empty"
    return response.json()["job_id"]


@pytest.mark.parametrize("number", (*INVALID_NUMBERS, "depth65"))
@pytest.mark.parametrize("endpoint", ("/dashboard/summary", "/analyze/run"))
def test_corrupt_global_cache_recomputes_from_items_without_destroying_history(client, number, endpoint):
    for index, channel in enumerate(("ent", "edu", "other")):
        response = client.post("/collect", json={
            "url": f"https://example.invalid/finite-history/{index}", "title": f"Synthetic {index}",
            "text": f"Distinct synthetic content {index}", "ts": 1790208000000 + index,
            "source": "import", "channel": channel,
        })
        assert response.status_code == 200 and response.json()["inserted"] == 1
    initial = client.post("/analyze/run?force=true").json()
    expected_counts = initial["channel_counts"]
    if number == "depth65":
        # The report root counts as one container, plus 64 below this field.
        encoded = json.dumps({**initial, "synthetic_depth": nested_containers(64)}, allow_nan=False)
    else:
        poisoned = {**initial, "channel_counts": {**expected_counts, "other": "INVALID_NUMBER"}}
        encoded = json.dumps(poisoned).replace('"INVALID_NUMBER"', number)
    with db.get_conn() as conn:
        old_id = conn.execute("SELECT MAX(id) FROM analysis_runs").fetchone()[0]
        conn.execute("UPDATE analysis_runs SET payload = ? WHERE id = ?", (encoded, old_id))
    before = snapshot()
    response = client.get(endpoint) if endpoint.startswith("/dashboard") else client.post(endpoint)
    assert response.status_code == 200
    result = response.json()
    assert result["cached"] is False
    assert result["total_count"] == 3
    assert result["channel_counts"] == expected_counts
    assert result["repeat_ratio"] == 0.0
    after = snapshot()
    assert after["items"] == before["items"]
    assert after["analysis_jobs"] == before["analysis_jobs"]
    assert after["analysis_runs"][:-1] == before["analysis_runs"]
    with db.get_conn() as conn:
        old_payload = conn.execute("SELECT payload FROM analysis_runs WHERE id = ?", (old_id,)).fetchone()[0]
        projection = conn.execute("SELECT channel_counts, repeat_ratio FROM stats_daily").fetchone()
        new_payload = conn.execute("SELECT payload FROM analysis_runs ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert old_payload == encoded
    assert json.loads(projection["channel_counts"]) == expected_counts and projection["repeat_ratio"] == 0.0
    json.dumps(json.loads(new_payload), allow_nan=False)
    cached = client.get("/dashboard/summary").json()
    assert cached["cached"] is True and cached["channel_counts"] == expected_counts


@pytest.mark.parametrize("kind", INVALID_JSON)
@pytest.mark.parametrize("field", ("result_payload", "metrics_json"))
@pytest.mark.parametrize("endpoint", ("jobs", "result"))
def test_corrupt_job_json_returns_explicit_conflict_without_mutating_stored_job(client, kind, field, endpoint):
    job_id = empty_job(client)
    with db.get_conn() as conn:
        conn.execute(f"UPDATE analysis_jobs SET {field} = ? WHERE id = ?", (corrupted_json(kind), job_id))
    before = snapshot()
    assert_stored_invalid(client.get(f"/analyze/{endpoint}/{job_id}"))
    assert snapshot() == before


@pytest.mark.parametrize("kind", INVALID_JSON)
def test_corrupt_history_json_returns_explicit_conflict_without_mutating_history(client, kind):
    assert client.post("/analyze/run").status_code == 200
    with db.get_conn() as conn:
        conn.execute("UPDATE analysis_runs SET channel_counts = ?", (corrupted_json(kind),))
    before = snapshot()
    assert_stored_invalid(client.get("/analyze/history"))
    assert snapshot() == before


@pytest.mark.parametrize("field", ("repeat_ratio", "negative_ratio", "avg_sentiment"))
@pytest.mark.parametrize("value", (float("inf"), -float("inf")))
def test_history_nonfinite_sqlite_real_cannot_be_silently_rendered_as_null(client, field, value):
    assert client.post("/analyze/run").status_code == 200
    with db.get_conn() as conn:
        conn.execute(f"UPDATE analysis_runs SET {field} = ?", (value,))
        assert not math.isfinite(conn.execute(f"SELECT {field} FROM analysis_runs").fetchone()[0])
    before = snapshot()
    assert_stored_invalid(client.get("/analyze/history"))
    assert snapshot() == before


def test_finite_history_and_legitimate_null_measurements_keep_the_existing_contract(client):
    job_id = empty_job(client)
    assert client.post("/analyze/run").status_code == 200
    with db.get_conn() as conn:
        conn.execute("UPDATE analysis_jobs SET result_payload = ?, metrics_json = NULL WHERE id = ?",
                     (json.dumps({"measurement": None, "valid_zero": 0.0}), job_id))
    before = snapshot()
    job = client.get(f"/analyze/jobs/{job_id}")
    result = client.get(f"/analyze/result/{job_id}")
    history = client.get("/analyze/history")
    assert job.status_code == result.status_code == history.status_code == 200
    assert job.json()["result_payload"] == result.json()["result"] == {"measurement": None, "valid_zero": 0.0}
    assert job.json()["metrics_json"] is result.json()["metrics"] is None
    assert all(run["negative_ratio"] is None and run["avg_sentiment"] is None for run in history.json()["runs"])
    assert snapshot() == before


@pytest.mark.parametrize("status", ("queued", "running", "failed"))
def test_missing_payload_for_unfinished_jobs_keeps_existing_state_semantics(client, status):
    job_id = empty_job(client)
    with db.get_conn() as conn:
        conn.execute("UPDATE analysis_jobs SET status = ?, result_payload = NULL, metrics_json = NULL WHERE id = ?", (status, job_id))
    before = snapshot()
    job = client.get(f"/analyze/jobs/{job_id}")
    result = client.get(f"/analyze/result/{job_id}")
    assert job.status_code == 200 and job.json()["status"] == status
    assert result.status_code == 409 and result.json() == {"detail": f"job not completed: {status}"}
    assert snapshot() == before


@pytest.mark.parametrize("depth", (63, 64, 65))
def test_mixed_container_depth_boundary_counts_the_stored_root_as_one(client, depth):
    job_id = empty_job(client)
    value = nested_containers(depth)
    encoded = json.dumps(value, allow_nan=False)
    with db.get_conn() as conn:
        conn.execute("UPDATE analysis_jobs SET result_payload = ? WHERE id = ?", (encoded, job_id))
    before = snapshot()
    response = client.get(f"/analyze/result/{job_id}")
    if depth <= 64:
        assert response.status_code == 200 and response.json()["result"] == value
    else:
        assert_stored_invalid(response)
    assert snapshot() == before


@pytest.mark.parametrize("encoded", (r'{"measurement":"\ud800"}', r'{"\udfff":1}'))
def test_invalid_unicode_values_and_keys_are_reported_without_serializer_failure(client, encoded):
    job_id = empty_job(client)
    with db.get_conn() as conn:
        conn.execute("UPDATE analysis_jobs SET result_payload = ? WHERE id = ?", (encoded, job_id))
    before = snapshot()
    assert_stored_invalid(client.get(f"/analyze/result/{job_id}"))
    assert snapshot() == before


def test_valid_unicode_surrogate_pair_and_non_ascii_keys_survive_history_reads(client):
    job_id = empty_job(client)
    with db.get_conn() as conn:
        conn.execute("UPDATE analysis_jobs SET result_payload = ? WHERE id = ?",
                     (r'{"\u5408\u6210":"\ud83d\ude00"}', job_id))
    before = snapshot()
    response = client.get(f"/analyze/result/{job_id}")
    assert response.status_code == 200 and response.json()["result"] == {"合成": "😀"}
    assert snapshot() == before


def test_invalid_unicode_global_cache_recomputes_instead_of_publishing_a_bad_string(client):
    initial = client.post("/analyze/run").json()
    encoded = json.dumps({**initial, "synthetic_text": "INVALID_UNICODE"}).replace('"INVALID_UNICODE"', r'"\ud800"')
    with db.get_conn() as conn:
        row_id = conn.execute("SELECT MAX(id) FROM analysis_runs").fetchone()[0]
        conn.execute("UPDATE analysis_runs SET payload = ? WHERE id = ?", (encoded, row_id))
    response = client.get("/dashboard/summary")
    assert response.status_code == 200 and response.json()["cached"] is False
    assert response.json()["total_count"] == 0 and "synthetic_text" not in response.json()
    with db.get_conn() as conn:
        assert conn.execute("SELECT payload FROM analysis_runs WHERE id = ?", (row_id,)).fetchone()[0] == encoded


@pytest.mark.parametrize("mode", ("full", "dashboard"))
def test_overdeep_finite_window_cache_is_recomputed_and_preserved(client, monkeypatch, mode):
    # Reuse only the existing synthetic inference outputs. The HTTP routes,
    # SQLite state, cache keys and metric aggregation remain actual code.
    from src.hyh.tests.test_analysis_concurrency import full_result, dashboard_result

    calls = []

    def pipeline(rows):
        calls.append(len(rows))
        return full_result(rows) if mode == "full" else dashboard_result(rows)

    monkeypatch.setattr(api, "_run_lsj_pipeline" if mode == "full" else "_execute_lsj_pipeline", pipeline)
    for index in range(5):
        assert client.post("/collect", json={
            "url": f"https://example.invalid/depth-window/{index}", "title": f"Synthetic {index}",
            "text": f"Synthetic depth content {index}", "ts": 1790208000000 + index, "source": "import",
        }).status_code == 200
    params = {"from_ts": 1790208000000, "to_ts": 1790208000010}
    request = (lambda: client.post("/analyze/run_full", params=params)) if mode == "full" else (
        lambda: client.get("/dashboard/visualization", params=params))
    initial = request()
    assert initial.status_code == 200 and initial.json()["cached"] is False
    with db.get_conn() as conn:
        row = conn.execute("SELECT id, result_payload FROM analysis_jobs ORDER BY id DESC LIMIT 1").fetchone()
        payload = json.loads(row["result_payload"])
        payload["synthetic_depth"] = nested_containers(64)
        encoded = json.dumps(payload, allow_nan=False)
        conn.execute("UPDATE analysis_jobs SET result_payload = ? WHERE id = ?", (encoded, row["id"]))
    before = snapshot()
    refreshed = request()
    assert refreshed.status_code == 200 and refreshed.json()["cached"] is False
    assert calls == [5, 5]
    after = snapshot()
    assert after["items"] == before["items"]
    assert after["analysis_jobs"][:-1] == before["analysis_jobs"]
    cached = request()
    assert cached.status_code == 200 and cached.json()["cached"] is True
    assert calls == [5, 5]


@pytest.mark.parametrize("report,valid", (
    pytest.param(nested_containers(63), True, id="complete-payload-depth64"),
    pytest.param(nested_containers(64), False, id="complete-payload-depth65"),
    pytest.param({"measurement": "\ud800"}, False, id="invalid-unicode-value"),
    pytest.param({"\udfff": 1}, False, id="invalid-unicode-key"),
))
def test_new_full_reports_obey_the_same_read_contract_before_becoming_completed(client, monkeypatch, report, valid):
    from src.hyh.tests.test_analysis_concurrency import full_result

    def pipeline(rows):
        return {**full_result(rows), "full_report": report}

    monkeypatch.setattr(api, "_run_lsj_pipeline", pipeline)
    response = client.post("/analyze/run_full")
    with db.get_conn() as conn:
        jobs = [dict(row) for row in conn.execute("SELECT * FROM analysis_jobs")]
        runs = conn.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0]
    assert len(jobs) == 1
    job = jobs[0]
    if valid:
        assert response.status_code == 200 and job["status"] == api.JOB_COMPLETED
        assert runs == 1
        historical = client.get(f"/analyze/result/{job['id']}")
        assert historical.status_code == 200 and historical.json()["result"]["full_report"] == report
    else:
        assert response.status_code == 500 and response.json()["status"] == api.JOB_FAILED
        assert job["status"] == api.JOB_FAILED and job["result_payload"] is None
        assert runs == 0
        assert client.get(f"/analyze/jobs/{job['id']}").status_code == 200
        assert client.get(f"/analyze/result/{job['id']}").status_code == 409
