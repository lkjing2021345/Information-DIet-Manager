"""Global vs windowed contracts with temporary SQLite and synthetic inference.

These HTTP tests do not import or validate real model weights or model accuracy.
"""
import json

import pytest
from fastapi.testclient import TestClient

from src.hyh import app as api, db


BASE_TS = 1790208000000
PRIVATE_MARKER = "SYNTHETIC_PRIVATE_EXCEPTION_MARKER"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "legacy-statistics.sqlite3")
    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000),
                    headers={"Authorization": "Bearer " + "a" * 43}) as client:
        yield client


def seed(client, count=20, start=0):
    # Different URLs, with each pair sharing one normalized title/text hash.
    for index in range(start, start + count):
        response = client.post("/collect", json={
            "url": f"https://example.invalid/legacy/{index}",
            "title": f"Synthetic group {index // 2}", "text": "Synthetic content",
            "ts": BASE_TS + index, "source": "import", "channel": "edu",
        })
        assert response.status_code == 200 and response.json()["inserted"] == 1


def successful_pipeline(rows):
    return {"input_count": len(rows), "category_counts": {"Tools": len(rows)},
            "sentiment_counts": {"Neutral": len(rows)}, "negative_ratio": 0.0,
            "comparison_count": max(0, len(rows) - 1), "sentiment_count": len(rows), "polarity_count": len(rows),
            "avg_sentiment": 0.0, "repeat_ratio": 0.75,
            "quick_evaluation": {}, "full_report": {}, "pipeline_warning": None}


def full(client, **params):
    return client.post("/analyze/run_full", params=params)


def global_summary(client):
    response = client.get("/dashboard/summary")
    assert response.status_code == 200, response.text
    return response.json()


def count_rows(table):
    assert table in {"analysis_jobs", "analysis_runs", "stats_daily"}
    with db.get_conn() as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


@pytest.mark.parametrize("basic_first", [False, True])
@pytest.mark.parametrize("window", [{"limit_rows": 6}, {"from_ts": BASE_TS + 5, "to_ts": BASE_TS + 10}])
def test_windowed_full_scores_cannot_replace_global_page_statistics(client, monkeypatch, basic_first, window):
    seed(client)
    monkeypatch.setattr(api, "_run_lsj_pipeline", successful_pipeline)
    if basic_first:
        assert global_summary(client)["total_count"] == 20
    response = full(client, **window)
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["total_count"] == result["window"]["input_count"] == 6
    assert result["statistics_scope"] == "analysis_window"
    assert result["repeat_ratio"] == 0.75
    assert count_rows("stats_daily") == int(basic_first)
    for result in (global_summary(client), client.post("/analyze/run").json()):
        assert result["total_count"] == 20
        assert result["statistics_scope"] == "all_saved_pages"
        assert result["repeat_ratio"] == 0.5
        assert result["repeat_sample_count"] == 20
        assert result["repeat_metric"] == "duplicate_normalized_content_hash_fraction"
        assert result["negative_ratio"] is None and result["avg_sentiment"] is None
    with db.get_conn() as conn:
        projection = conn.execute("SELECT total_count, repeat_ratio FROM stats_daily").fetchone()
        assert tuple(projection) == (20, 0.5)


@pytest.mark.parametrize("bad_payload", ["{broken-json", '{"total_count": 999}', "[]"])
def test_old_or_invalid_cache_payload_and_future_daily_projection_are_not_trusted(client, bad_payload):
    seed(client, 4)
    with db.get_conn() as conn:
        day = api.datetime.now(api.timezone.utc).date().isoformat()
        conn.execute("INSERT INTO stats_daily VALUES (?, 999, '{}', 1, 1, 1, ?, ?)",
                     (day, 2**62, 2**62))
        conn.execute("INSERT INTO analysis_runs (day, total_count, payload, created_at) VALUES (?, 999, ?, ?)",
                     (day, bad_payload, 2**62))
    result = global_summary(client)
    assert result["cached"] is False and result["total_count"] == 4
    assert result["repeat_ratio"] == 0.5
    assert result["statistics_version"] == 1
    with db.get_conn() as conn:
        assert conn.execute("SELECT total_count FROM stats_daily").fetchone()[0] == 4


def test_valid_basic_cache_is_reused_without_creating_history_on_each_summary(client):
    seed(client, 4)
    first = client.post("/analyze/run").json()
    assert first["cached"] is False
    before = count_rows("analysis_runs")
    second = global_summary(client)
    assert second["cached"] is True and second["data_version"] == first["data_version"]
    assert count_rows("analysis_runs") == before
    assert client.post("/analyze/run").json()["cached"] is True
    assert count_rows("analysis_runs") == before + 1
    assert client.post("/analyze/run?force=true").json()["cached"] is False


def test_same_created_millisecond_inserts_invalidate_global_and_full_caches(client, monkeypatch):
    monkeypatch.setattr(api, "_now_ms", lambda: BASE_TS)
    monkeypatch.setattr(api, "_run_lsj_pipeline", successful_pipeline)
    seed(client, 6)
    first = global_summary(client)
    assert full(client).json()["result"]["total_count"] == 6
    assert full(client).json()["cached"] is True
    # Reestablish a global payload as the latest history entry before inserting.
    first = global_summary(client)
    seed(client, 2, start=6)
    second = global_summary(client)
    assert second["total_count"] == 8 and second["cached"] is False
    assert second["data_version"] != first["data_version"]
    result = full(client).json()
    assert result["cached"] is False and result["result"]["total_count"] == 8


def test_update_without_new_timestamp_invalidates_global_and_full_caches(client, monkeypatch):
    seed(client, 6)
    monkeypatch.setattr(api, "_run_lsj_pipeline", successful_pipeline)
    assert full(client).status_code == 200
    first = global_summary(client)
    with db.get_conn() as conn:
        conn.execute("UPDATE items SET content_hash = ? WHERE id = 1", ("f" * 64,))
    updated = global_summary(client)
    assert updated["cached"] is False and updated["data_version"] != first["data_version"]
    assert updated["repeat_ratio"] == pytest.approx(2 / 6)
    assert full(client).json()["cached"] is False


def test_delete_and_same_total_restore_do_not_reuse_old_global_or_windowed_results(client, monkeypatch):
    seed(client, 6)
    monkeypatch.setattr(api, "_run_lsj_pipeline", successful_pipeline)
    backup = client.get("/data/backup").json()
    original = global_summary(client)
    assert full(client).status_code == 200
    assert client.delete("/items/1", headers={"X-IDM-Confirm": "delete-record"}).status_code == 200
    deleted = global_summary(client)
    assert deleted["total_count"] == 5 and deleted["repeat_ratio"] == pytest.approx(2 / 5)
    assert full(client).json()["result"]["total_count"] == 5
    for _ in range(2):
        response = client.post("/data/restore", json=backup, headers={"X-IDM-Confirm": "replace-records"})
        assert response.status_code == 200
        restored = global_summary(client)
        assert restored["total_count"] == 6 and restored["repeat_ratio"] == 0.5
        assert restored["data_version"] != original["data_version"] and restored["cached"] is False
        result = full(client).json()
        assert result["cached"] is False and result["result"]["total_count"] == 6
        original = restored


def test_old_full_cache_key_is_not_reused(client, monkeypatch):
    seed(client, 6)
    monkeypatch.setattr(api, "_run_lsj_pipeline", successful_pipeline)
    assert full(client).status_code == 200
    old_key = api._stable_hash_payload({"day": api.datetime.now(api.timezone.utc).date().isoformat(),
                                      "from_ts": None, "to_ts": None, "limit_rows": 5000, "mode": "run_full"})
    with db.get_conn() as conn:
        conn.execute("UPDATE analysis_jobs SET input_hash = ?", (old_key,))
    assert full(client).json()["cached"] is False


def test_global_exact_hash_denominator_excludes_missing_hashes_and_empty_is_explicit(client):
    empty = global_summary(client)
    assert empty["total_count"] == empty["repeat_sample_count"] == 0 and empty["repeat_ratio"] is None
    seed(client, 4)
    with db.get_conn() as conn:
        conn.execute("UPDATE items SET content_hash = NULL WHERE id <= 2")
    partial = global_summary(client)
    assert partial["total_count"] == 4 and partial["repeat_sample_count"] == 2
    assert partial["repeat_ratio"] == 0.5


@pytest.mark.parametrize("failure,status", [("unavailable", 503), ("validation", 422), ("unexpected", 500)])
def test_failed_full_analysis_is_persisted_safely_without_success_statistics_and_can_recover(client, monkeypatch, failure, status):
    seed(client, 6)
    if failure == "unavailable":
        monkeypatch.setattr(api, "_execute_lsj_pipeline", lambda rows: {"ok": False, "warning": PRIVATE_MARKER})
    else:
        def fail(_rows):
            raise (ValueError if failure == "validation" else RuntimeError)(PRIVATE_MARKER)
        monkeypatch.setattr(api, "_run_lsj_pipeline", fail)
    response = full(client)
    assert response.status_code == status and PRIVATE_MARKER not in response.text
    assert response.json()["status"] == "failed" and "result" not in response.json()
    job = client.get(f"/analyze/jobs/{response.json()['job_id']}")
    assert job.status_code == 200 and PRIVATE_MARKER not in job.text
    assert job.json()["status"] == "failed" and job.json()["result_payload"] is None
    assert client.get(f"/analyze/result/{response.json()['job_id']}").status_code == 409
    history = client.get("/analyze/history")
    assert PRIVATE_MARKER not in history.text and history.json()["total"] == 0
    assert count_rows("stats_daily") == 0
    monkeypatch.setattr(api, "_run_lsj_pipeline", successful_pipeline)
    recovered = full(client).json()
    assert recovered["status"] == "completed" and recovered["cached"] is False
    assert recovered["result"]["analysis_status"] == "ready"


def test_failed_result_persistence_rolls_back_partial_success_history_but_keeps_failed_job(client, monkeypatch):
    seed(client, 6)
    monkeypatch.setattr(api, "_run_lsj_pipeline", successful_pipeline)
    update = api._update_analysis_job
    def fail_completion(conn, job_id, **kwargs):
        if kwargs["status"] == api.JOB_COMPLETED:
            raise RuntimeError(PRIVATE_MARKER)
        return update(conn, job_id, **kwargs)
    monkeypatch.setattr(api, "_update_analysis_job", fail_completion)
    response = full(client)
    assert response.status_code == 500 and PRIVATE_MARKER not in response.text
    assert count_rows("analysis_runs") == count_rows("stats_daily") == 0
    job = client.get(f"/analyze/jobs/{response.json()['job_id']}").json()
    assert job["status"] == "failed" and job["result_payload"] is None


@pytest.mark.parametrize("endpoint", ["legacy", "dashboard"])
def test_dependency_exception_text_does_not_escape_shared_execution_boundary(client, monkeypatch, endpoint):
    seed(client, 6)
    def fail_import():
        raise RuntimeError(PRIVATE_MARKER)
    monkeypatch.setattr(api, "_ensure_lsj_import_path", fail_import)
    response = (full(client) if endpoint == "legacy" else client.get(
        "/dashboard/visualization", params={"from_ts": BASE_TS, "to_ts": BASE_TS + 10}))
    assert PRIVATE_MARKER not in response.text
    with db.get_conn() as conn:
        for table in ("analysis_jobs", "analysis_runs"):
            assert PRIVATE_MARKER not in json.dumps([dict(row) for row in conn.execute(f"SELECT * FROM {table}")])


def test_empty_full_window_does_not_attempt_models_and_is_not_a_failure(client, monkeypatch):
    monkeypatch.setattr(api, "_execute_lsj_pipeline", lambda _rows: pytest.fail("No inference for empty input"))
    response = full(client)
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["analysis_status"] == "empty" and result["total_count"] == 0
    assert result["sentiment_counts"] == {} and result["full_report"] is None
    assert result["comparison_count"] == result["sentiment_count"] == result["polarity_count"] == 0
    for key in ("repeat_ratio", "negative_ratio", "avg_sentiment"):
        assert result[key] is None
    cached = full(client).json()
    assert cached["cached"] is True and cached["result"] == result
    history = client.get("/analyze/history").json()["runs"]
    assert len(history) == 1
    with db.get_conn() as conn:
        stored_payload = json.loads(conn.execute("SELECT payload FROM analysis_runs").fetchone()[0])
    for key in ("repeat_ratio", "negative_ratio", "avg_sentiment"):
        assert history[0][key] is None and stored_payload[key] is None
    assert count_rows("stats_daily") == 0


LEGACY_TIME_ENDPOINTS = [("POST", "/analyze/run_full"), ("GET", "/export/lsj"),
                         ("GET", "/export/lsj/training")]


@pytest.mark.parametrize("method,endpoint", LEGACY_TIME_ENDPOINTS)
@pytest.mark.parametrize("field", ["from_ts", "to_ts"])
@pytest.mark.parametrize("value", [10**100, -1, api.MAX_INGEST_TS + 1])
def test_legacy_time_filters_reject_out_of_range_epoch_milliseconds(client, method, endpoint, field, value):
    response = client.request(method, endpoint, params={field: value})
    assert response.status_code == 422
    assert count_rows("analysis_jobs") == count_rows("analysis_runs") == 0


@pytest.mark.parametrize("method,endpoint", LEGACY_TIME_ENDPOINTS)
@pytest.mark.parametrize("boundary", [0, api.MAX_INGEST_TS])
def test_legacy_time_filters_accept_inclusive_ingest_boundaries(client, method, endpoint, boundary):
    response = client.request(method, endpoint, params={"from_ts": boundary, "to_ts": boundary})
    assert response.status_code == 200


def test_global_channel_aliases_and_missing_groups_are_added_without_overwriting(client):
    seed(client, 8)
    channels = [None, "unknown", "", " ", "edu", "learning", "学习", "unlisted"]
    with db.get_conn() as conn:
        conn.executemany("UPDATE items SET channel = ? WHERE id = ?", [(value, index + 1) for index, value in enumerate(channels)])
    result = global_summary(client)
    assert result["total_count"] == sum(result["channel_counts"].values()) == 8
    assert result["channel_counts"] == {"ent": 0, "edu": 3, "news": 0, "soc": 0, "other": 5}


def use_measured_frame(monkeypatch, measurements):
    """Keep the API aggregation real while substituting inference and score logic."""
    import pandas as pd

    class SyntheticReport:
        def to_dict(self):
            return {"synthetic_report": True}

    class SyntheticEvaluator:
        def quick_evaluate(self, frame):
            return {"synthetic_quick": True}

        def evaluate(self, frame, detailed=False):
            return SyntheticReport()

    def execute(rows):
        assert len(rows) == len(measurements)
        frame = pd.DataFrame([{**row, "category": "Tools", **measurement}
                              for row, measurement in zip(rows, measurements)])
        return {"ok": True, "input_count": len(rows), "df3": frame, "evaluator": SyntheticEvaluator()}

    monkeypatch.setattr(api, "_execute_lsj_pipeline", execute)


@pytest.mark.parametrize("count", [5, 6])
def test_full_repeat_ratio_uses_only_measured_predecessor_pairs(client, monkeypatch, count):
    seed(client, count)
    use_measured_frame(monkeypatch, [{"similarity": 1.0, "similarity_valid": True,
                                     "sentiment": "Negative", "sentiment_valid": True, "polarity": -0.5}] * count)
    response = full(client)
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["repeat_ratio"] == 1 and result["comparison_count"] == count - 1
    assert result["sentiment_count"] == result["polarity_count"] == count
    assert result["category_counts"] == {"Tools": count}
    assert result["quick_evaluation"] == {"synthetic_quick": True}
    assert result["full_report"] == {"synthetic_report": True}


def test_full_each_metric_excludes_only_its_invalid_measurements_without_clipping_or_filling(client, monkeypatch):
    seed(client, 6)
    use_measured_frame(monkeypatch, [
        {"similarity": 1.0, "similarity_valid": True, "sentiment": "Negative", "sentiment_valid": True, "polarity": -1.0},
        {"similarity": 1.0, "similarity_valid": True, "sentiment": "Negative", "sentiment_valid": True, "polarity": None},
        {"similarity": 1.1, "similarity_valid": True, "sentiment": "Positive", "sentiment_valid": True, "polarity": 2.0},
        {"similarity": 0.0, "similarity_valid": False, "sentiment": "Neutral", "sentiment_valid": False, "polarity": 0.0},
        {"similarity": 0.9, "similarity_valid": 1, "sentiment": "Positive", "sentiment_valid": 1, "polarity": 0.5},
        {"similarity": 0.2, "similarity_valid": True, "sentiment": "Positive", "sentiment_valid": True, "polarity": 0.5},
    ])
    response = full(client)
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["comparison_count"] == 2 and result["repeat_ratio"] == 0.5
    assert result["sentiment_count"] == 4 and result["negative_ratio"] == 0.5
    assert result["polarity_count"] == 2 and result["avg_sentiment"] == -0.25
    assert result["sentiment_counts"] == {"Negative": 2, "Positive": 2}
    assert result["category_counts"] == {"Tools": 6} and result["total_count"] == 6


def test_full_missing_measurements_are_null_and_keep_unrelated_category_counts(client, monkeypatch):
    seed(client, 6)
    use_measured_frame(monkeypatch, [{"similarity": value, "sentiment": None, "polarity": value}
                                     for value in (1.0, None, float("nan"), float("inf"), "0.9", True)])
    response = full(client)
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["repeat_ratio"] is result["negative_ratio"] is result["avg_sentiment"] is None
    assert result["comparison_count"] == result["sentiment_count"] == result["polarity_count"] == 0
    assert result["category_counts"] == {"Tools": 6} and result["total_count"] == 6
