"""HTTP contract tests with synthetic SQLite data; no model downloads/inference."""
import pytest
import pandas as pd
from contextlib import contextmanager
from fastapi.testclient import TestClient

from src.hyh import app as api
from src.hyh import db


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "dashboard.sqlite3")
    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000),
                    headers={"Authorization": "Bearer " + "a" * 43}) as client:
        yield client


def seed(client, count):
    items = [{"url": f"https://example.invalid/{i}", "title": f"Synthetic {i}",
              "text": "Synthetic audit text", "ts": 1790208000000 + i,
              "source": "import"} for i in range(count)]
    for item in items:
        response = client.post("/collect", json=item)
        assert response.status_code == 200, response.text
        assert response.json()["inserted"] == 1


def get_analysis(client, **params):
    return client.get("/dashboard/visualization", params={"from_ts": 1790208000000,
                      "to_ts": 1790294400000, **params})


@pytest.mark.parametrize("count,status", [(0, "empty"), (1, "insufficient_data"), (4, "insufficient_data")])
def test_small_samples_return_explicit_state_without_loading_models(client, monkeypatch, count, status):
    seed(client, count)
    def unexpected(_rows):
        pytest.fail("Small samples must not initialize optional model dependencies")
    monkeypatch.setattr(api, "_execute_lsj_pipeline", unexpected)
    response = get_analysis(client)
    assert response.status_code == 200
    result = response.json()
    assert result["analysis_status"] == status
    assert result["window"]["input_count"] == count
    assert result["window"]["available_count"] == count
    assert result["minimum_records"] == 5
    assert result["global"]["time_series"] == []
    assert result["category_counts"] == {}


def test_dependency_failure_has_no_fabricated_measurements(client, monkeypatch):
    seed(client, 5)
    monkeypatch.setattr(api, "_execute_lsj_pipeline", lambda rows: {
        "ok": False, "warning": "synthetic missing dependency", "input_count": len(rows), "repeat_ratio": 0,
    })
    result = get_analysis(client).json()
    assert result["analysis_status"] == "unavailable"
    assert result["pipeline_warning"]
    assert result["global"]["time_series"] == []
    assert result["category_counts"] == {}


def pipeline(fail=False):
    def execute(rows):
        frame = pd.DataFrame({"id": [r["id"] for r in rows], "ts": [r["ts"] for r in rows],
                              "title": ["sample"] * len(rows),
                              "sentiment": ["positive"] * len(rows), "polarity": [1.0] * len(rows),
                              "similarity": [0.0] * len(rows), "category": ["tools"] * 3 + ["shopping"] * (len(rows)-3)})
        if fail:
            frame = frame.iloc[::-1]  # A model must not reassign results to different records.
        return {"ok": True, "input_count": len(rows), "df3": frame}
    return execute


def test_valid_metrics_preserve_zero_and_actual_date_and_categories(client, monkeypatch):
    seed(client, 5)
    monkeypatch.setattr(api, "_execute_lsj_pipeline", pipeline())
    result = get_analysis(client).json()
    assert result["analysis_status"] == "ready"
    assert result["category_counts"] == {"tools": 3, "shopping": 2}
    row = result["global"]["time_series"][0]
    assert row["date"] == "2026-09-24"
    assert (row["positive_ratio"], row["neutral_ratio"], row["negative_ratio"]) == (1.0, 0.0, 0.0)
    assert result["window"]["processed_count"] == 5
    assert result["date_timezone"] == "UTC"
    assert row["comparison_count"] == 4 and row["sentiment_count"] == 5
    assert row["repeat_ratio"] == 0.0


def test_postprocessing_failure_is_explicit_and_empty(client, monkeypatch):
    seed(client, 5)
    monkeypatch.setattr(api, "_execute_lsj_pipeline", pipeline(fail=True))
    result = get_analysis(client).json()
    assert result["analysis_status"] == "failed"
    assert result["global"]["time_series"] == []
    assert result["categories"] == {}


def test_truncation_is_reported_with_actual_window_coverage(client, monkeypatch):
    seed(client, 6)
    monkeypatch.setattr(api, "_execute_lsj_pipeline", pipeline())
    result = get_analysis(client, limit_rows=5).json()
    assert result["window"]["truncated"] is True
    assert result["window"]["available_count"] == 6
    assert result["window"]["input_count"] == 5


def test_saved_records_total_stays_live_and_uses_real_pagination(client):
    seed(client, 51)
    first = client.get("/items?page=1&page_size=50").json()
    second = client.get("/items?page=2&page_size=50").json()
    assert first["total"] == second["total"] == 51
    assert len(first["items"]) == 50 and len(second["items"]) == 1
    assert {x["id"] for x in first["items"]}.isdisjoint(x["id"] for x in second["items"])
    assert "sentiment" not in first["items"][0]


def test_ready_cache_preserves_contract_and_force_recomputes(client, monkeypatch):
    seed(client, 5)
    monkeypatch.setattr(api, "_execute_lsj_pipeline", pipeline())
    first = get_analysis(client).json()
    assert first["analysis_status"] == "ready" and first["cached"] is False

    def fail(_rows):
        return {"ok": False, "warning": "synthetic pipeline outage", "input_count": 5}

    monkeypatch.setattr(api, "_execute_lsj_pipeline", fail)
    cached = get_analysis(client).json()
    assert cached["cached"] is True and cached["analysis_status"] == "ready"
    assert cached["window"]["available_count"] == 5
    assert cached["category_counts"] == first["category_counts"]

    fresh = get_analysis(client, force=True).json()
    assert fresh["cached"] is False and fresh["analysis_status"] == "unavailable"
    assert fresh["global"]["time_series"] == []
    with db.get_conn() as conn:
        assert conn.execute("SELECT status FROM analysis_jobs ORDER BY id DESC LIMIT 1").fetchone()[0] == api.JOB_FAILED


def test_failed_analysis_does_not_prevent_recovery_via_cache(client, monkeypatch):
    seed(client, 5)
    monkeypatch.setattr(api, "_execute_lsj_pipeline", lambda rows: {"ok": False, "warning": "synthetic outage"})
    assert get_analysis(client).json()["analysis_status"] == "unavailable"
    monkeypatch.setattr(api, "_execute_lsj_pipeline", pipeline())
    recovered = get_analysis(client).json()
    assert recovered["analysis_status"] == "ready"
    assert recovered["cached"] is False


def test_same_text_pairs_have_four_comparisons_not_five(client, monkeypatch):
    seed(client, 5)
    def identical(rows):
        result = pipeline()(rows)
        result["df3"]["similarity"] = [0.0, 1.0, 1.0, 1.0, 1.0]
        return result
    monkeypatch.setattr(api, "_execute_lsj_pipeline", identical)
    result = get_analysis(client).json()
    row = result["global"]["time_series"][0]
    assert row["count"] == 5 and row["comparison_count"] == 4
    assert row["repeat_ratio"] == 1.0 and row["sentiment_count"] == 5
    assert sum(result["category_counts"].values()) == 5


def test_moving_window_does_not_reuse_yesterdays_cache(client, monkeypatch):
    seed(client, 5)
    now = 1790208000000 + 7 * 86400000
    monkeypatch.setattr(api, "_now_ms", lambda: now)
    monkeypatch.setattr(api, "_execute_lsj_pipeline", pipeline())
    first = client.get("/dashboard/visualization?days=7").json()
    assert first["analysis_status"] == "ready"
    now += 10  # all five records aged out, without any database write
    second = client.get("/dashboard/visualization?days=7").json()
    assert second["analysis_status"] == "empty" and second["cached"] is False
    assert second["window"]["from_ts"] == 1790208000010


def test_insert_with_same_created_millisecond_invalidates_cache(client, monkeypatch):
    monkeypatch.setattr(api, "_now_ms", lambda: 1790208000100)
    seed(client, 5)
    monkeypatch.setattr(api, "_execute_lsj_pipeline", pipeline())
    assert get_analysis(client).json()["analysis_status"] == "ready"
    assert client.post("/collect", json={"url": "https://example.invalid/new", "title": "New synthetic",
        "ts": 1790208000020, "source": "import"}).status_code == 200
    result = get_analysis(client).json()
    assert result["cached"] is False and result["window"]["processed_count"] == 6


def test_old_metric_schema_cache_is_not_reused(client, monkeypatch):
    seed(client, 5)
    monkeypatch.setattr(api, "_execute_lsj_pipeline", pipeline())
    assert get_analysis(client).json()["analysis_status"] == "ready"
    old_key = api._stable_hash_payload({"days": 7, "from_ts": 1790208000000, "to_ts": 1790294400000,
                                      "limit_rows": 5000, "mode": "dashboard_visualization", "schema_version": 2})
    with db.get_conn() as conn:
        conn.execute("UPDATE analysis_jobs SET input_hash = ?", (old_key,))
    monkeypatch.setattr(api, "_execute_lsj_pipeline", lambda rows: {"ok": False, "warning": "synthetic unavailable"})
    result = get_analysis(client).json()
    assert result["analysis_status"] == "unavailable" and result["cached"] is False


@pytest.mark.parametrize("params,status", [({"from_ts": -1}, 422), ({"to_ts": 2**64}, 422),
                                           ({"from_ts": 4102444800000}, 400)])
def test_invalid_window_is_explicit_client_error(client, params, status):
    assert client.get("/dashboard/visualization", params=params).status_code == status


@pytest.mark.parametrize("operation", ["delete", "restore"])
def test_destructive_change_between_read_and_job_creation_cannot_resurrect_analysis(client, monkeypatch, operation):
    seed(client, 5)
    backup = client.get("/data/backup").json()
    monkeypatch.setattr(api, "_execute_lsj_pipeline", pipeline())
    original = api.get_conn
    first_read = True
    @contextmanager
    def interleaved_connection():
        nonlocal first_read
        inject = first_read
        first_read = False
        with original() as conn:
            yield conn
        if inject:
            if operation == "delete":
                assert client.delete("/data", headers={"X-IDM-Confirm": "delete-all"}).status_code == 200
            else:
                assert client.post("/data/restore", json=backup, headers={"X-IDM-Confirm": "replace-records"}).status_code == 200
    monkeypatch.setattr(api, "get_conn", interleaved_connection)
    response = get_analysis(client, force=True)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "analysis_snapshot_expired"
    with original() as conn:
        assert conn.execute("SELECT COUNT(*) FROM analysis_jobs").fetchone()[0] == 0


@pytest.mark.parametrize("operation", ["delete", "restore"])
def test_destructive_change_during_inference_is_not_blocked_and_old_result_is_discarded(client, monkeypatch, operation):
    seed(client, 5)
    backup = client.get("/data/backup").json()
    def interleaved_pipeline(rows):
        if operation == "delete":
            response = client.delete("/data", headers={"X-IDM-Confirm": "delete-all"})
        else:
            response = client.post("/data/restore", json=backup, headers={"X-IDM-Confirm": "replace-records"})
        assert response.status_code == 200
        return pipeline()(rows)
    monkeypatch.setattr(api, "_execute_lsj_pipeline", interleaved_pipeline)
    response = get_analysis(client, force=True)
    assert response.status_code == 409
    with db.get_conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM analysis_jobs").fetchone()[0] == 0


def test_collection_during_inference_is_not_blocked_and_does_not_change_its_snapshot(client, monkeypatch):
    seed(client, 5)
    def interleaved_pipeline(rows):
        response = client.post("/collect", json={"url": "https://example.invalid/during-analysis", "title": "Synthetic arrival",
                                               "ts": 1790208000030, "source": "import"})
        assert response.status_code == 200 and response.json()["inserted"] == 1
        return pipeline()(rows)
    monkeypatch.setattr(api, "_execute_lsj_pipeline", interleaved_pipeline)
    result = get_analysis(client, force=True).json()
    assert result["analysis_status"] == "ready" and result["window"]["processed_count"] == 5
    assert client.get("/items").json()["total"] == 6
    monkeypatch.setattr(api, "_execute_lsj_pipeline", pipeline())
    next_result = get_analysis(client).json()
    assert next_result["cached"] is False and next_result["window"]["processed_count"] == 6


def test_legacy_score_rejects_incomplete_measurements_without_http500_or_fake_score(client, monkeypatch):
    seed(client, 5)
    class LegacyEvaluator:
        def quick_evaluate(self, df):
            assert pd.isna(df["similarity"].iloc[0])
            raise ValueError("synthetic incomplete measurements")
    def legacy_pipeline(rows):
        result = pipeline()(rows)
        result["df3"].loc[0, "similarity"] = float("nan")
        result["evaluator"] = LegacyEvaluator()
        return result
    monkeypatch.setattr(api, "_execute_lsj_pipeline", legacy_pipeline)
    response = client.post("/analyze/run_full?force=true")
    assert response.status_code == 422
    result = response.json()
    assert result["status"] == "failed" and "result" not in result
    assert "synthetic" not in response.text
    with db.get_conn() as conn:
        job = conn.execute("SELECT status, result_payload FROM analysis_jobs WHERE id = ?", (result["job_id"],)).fetchone()
        assert job["status"] == "failed" and job["result_payload"] is None
        assert conn.execute("SELECT COUNT(*) FROM stats_daily").fetchone()[0] == 0


def test_external_update_during_analysis_finishes_stale_job_without_publishing_result(client, monkeypatch):
    seed(client, 5)
    def changed(rows):
        with db.get_conn() as conn:
            conn.execute("UPDATE items SET title = 'Updated synthetic title' WHERE id = 1")
        return pipeline()(rows)
    monkeypatch.setattr(api, "_execute_lsj_pipeline", changed)
    assert get_analysis(client, force=True).status_code == 409
    with db.get_conn() as conn:
        job = conn.execute("SELECT status, result_payload, error FROM analysis_jobs").fetchone()
        assert job["status"] == "failed" and job["result_payload"] is None
        assert job["error"] == "analysis snapshot expired"
