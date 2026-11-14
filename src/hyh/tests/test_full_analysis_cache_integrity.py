"""Finite JSON is insufficient for reusing a completed full-analysis cache.

Use actual API/temporary SQLite with synthetic inference, never model weights.
"""
import json

from fastapi.testclient import TestClient
import pytest

from src.hyh import app as api, db
from src.hyh.models import MAX_INGEST_TS
from src.hyh.tests.test_analysis_concurrency import full_result
from src.hyh.tests.test_legacy_statistics import use_measured_frame


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "synthetic-full-cache.sqlite3")
    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 51212),
                    headers={"Authorization": "Bearer " + "a" * 43}) as client:
        yield client


def seed(client, count=5):
    for index in range(count):
        response = client.post("/collect", json={
            "url": f"https://example.invalid/full-cache/{index}", "title": f"Synthetic {index}",
            "text": "Synthetic body", "ts": index, "source": "import", "channel": "edu",
        })
        assert response.status_code == 200 and response.json()["inserted"] == 1


def request(client, **params):
    return client.post("/analyze/run_full", params=params)


def install_pipeline(monkeypatch, build=full_result):
    calls = []
    def pipeline(rows):
        calls.append(len(rows))
        return build(rows)
    monkeypatch.setattr(api, "_run_lsj_pipeline", pipeline)
    return calls


def poison_latest(change):
    with db.get_conn() as conn:
        row = conn.execute("SELECT id, result_payload FROM analysis_jobs ORDER BY id DESC LIMIT 1").fetchone()
        payload = json.loads(row["result_payload"])
        change(payload)
        encoded = json.dumps(payload, allow_nan=False)
        conn.execute("UPDATE analysis_jobs SET result_payload=? WHERE id=?", (encoded, row["id"]))
        return row["id"], encoded


def assert_preserved(job_id, encoded):
    with db.get_conn() as conn:
        assert conn.execute("SELECT result_payload FROM analysis_jobs WHERE id=?", (job_id,)).fetchone()[0] == encoded


def replace(path, value):
    def change(payload):
        node = payload
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return change


def remove(path):
    def change(payload):
        node = payload
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]
    return change


def markers_only(payload):
    payload.clear()
    payload.update(statistics_scope="analysis_window", statistics_version=2,
                   analysis_status="ready", pipeline_warning=None)


CORRUPTIONS = [
    ("markers-only", markers_only),
    ("missing-null-from", remove(("window", "from_ts"))),
    ("missing-null-to", remove(("window", "to_ts"))),
    ("missing-warning", remove(("pipeline_warning",))),
    ("missing-total", remove(("total_count",))),
    ("missing-report", remove(("full_report",))),
    ("missing-quick", remove(("quick_evaluation",))),
    ("missing-metric", remove(("repeat_ratio",))),
    ("version-float", replace(("statistics_version",), 2.0)),
    ("wrong-day", replace(("day",), "1970-01-01")),
    ("window-count", replace(("window", "input_count"), 2)),
    ("window-limit", replace(("window", "limit_rows"), 4)),
    ("window-from", replace(("window", "from_ts"), 1)),
    ("wrong-total", replace(("total_count",), 999)),
    ("empty-with-records", replace(("analysis_status",), "empty")),
    ("wrong-repeat-metric", replace(("repeat_metric",), "different_measurement")),
    ("channels-not-map", replace(("channel_counts",), [])),
    ("channels-sum", replace(("channel_counts",), {"edu": 4})),
    ("categories-sum", replace(("category_counts",), {"Tools": 6})),
    ("category-bool", replace(("category_counts",), {"Tools": True, "other": 4})),
    ("category-negative", replace(("category_counts",), {"Tools": 6, "other": -1})),
    ("sentiment-label", replace(("sentiment_counts",), {"neutral": 5})),
    ("sentiment-sum", replace(("sentiment_counts",), {"Neutral": 4})),
    ("sentiment-count", replace(("sentiment_count",), 6)),
    ("too-many-comparisons", replace(("comparison_count",), 5)),
    ("comparison-float", replace(("comparison_count",), 4.0)),
    ("comparison-bool", replace(("comparison_count",), True)),
    ("too-many-polarities", replace(("polarity_count",), 6)),
    ("repeat-negative", replace(("repeat_ratio",), -0.1)),
    ("repeat-bool", replace(("repeat_ratio",), False)),
    ("repeat-null-measured", replace(("repeat_ratio",), None)),
    ("negative-ratio-inconsistent", replace(("negative_ratio",), 0.5)),
    ("polarity-out-of-range", replace(("avg_sentiment",), 1.1)),
    ("polarity-null-measured", replace(("avg_sentiment",), None)),
    ("report-not-map", replace(("full_report",), "ready")),
    ("quick-not-map", replace(("quick_evaluation",), [])),
    ("report-null-with-records", replace(("full_report",), None)),
    ("timestamp-bool", replace(("generated_at",), False)),
    ("timestamp-overflow", replace(("generated_at",), MAX_INGEST_TS + 1)),
    ("cached-not-bool", replace(("cached",), 0)),
]


@pytest.mark.parametrize("_name,change", CORRUPTIONS, ids=[name for name, _ in CORRUPTIONS])
def test_invalid_full_cache_recomputes_without_rewriting_history(client, monkeypatch, _name, change):
    seed(client)
    calls = install_pipeline(monkeypatch)
    initial = request(client)
    assert initial.status_code == 200 and initial.json()["cached"] is False
    expected = initial.json()["result"]
    old_id, encoded = poison_latest(change)
    response = request(client)
    assert response.status_code == 200 and response.json()["cached"] is False
    fresh = response.json()
    assert fresh["result"]["window"] == expected["window"]
    assert fresh["result"]["total_count"] == 5
    assert fresh["result"]["full_report"] == expected["full_report"]
    assert calls == [5, 5]
    assert_preserved(old_id, encoded)
    new_job = client.get(f'/analyze/jobs/{fresh["job_id"]}').json()
    assert new_job["status"] == api.JOB_COMPLETED and new_job["cache_hit"] is False
    cached = request(client)
    assert cached.status_code == 200 and cached.json()["cached"] is True
    assert calls == [5, 5]


@pytest.mark.parametrize("error,status", [(api.AnalysisUnavailableError, 503), (ValueError, 422), (RuntimeError, 500)])
def test_bad_cache_cannot_hide_recompute_failure(client, monkeypatch, error, status):
    seed(client)
    install_pipeline(monkeypatch)
    assert request(client).status_code == 200
    old_id, encoded = poison_latest(markers_only)
    calls = []
    def fail(rows):
        calls.append(len(rows))
        raise error("SYNTHETIC_PRIVATE_FULL_CACHE_FAULT")
    monkeypatch.setattr(api, "_run_lsj_pipeline", fail)
    response = request(client)
    assert response.status_code == status and response.json()["status"] == api.JOB_FAILED
    assert "SYNTHETIC_PRIVATE_FULL_CACHE_FAULT" not in response.text
    assert calls == [5]
    assert_preserved(old_id, encoded)
    job = client.get(f'/analyze/jobs/{response.json()["job_id"]}').json()
    assert job["status"] == api.JOB_FAILED and job["result_payload"] is None and job["cache_hit"] is False
    assert client.get(f'/analyze/result/{job["id"]}').status_code == 409


@pytest.mark.parametrize("case", ["empty", "single", "zeroes", "no-sentiment", "no-comparison", "no-polarity", "partial"])
def test_legal_independent_observations_remain_reusable(client, monkeypatch, case):
    count = 0 if case == "empty" else (1 if case == "single" else 5)
    seed(client, count)
    def build(rows):
        if not rows:
            result = full_result(rows)
            result.update(category_counts={}, sentiment_counts={}, quick_evaluation=None, full_report=None)
            return result
        result = full_result(rows)
        if case in {"no-sentiment", "partial"}:
            result.update(sentiment_counts={}, sentiment_count=0, polarity_count=0,
                          negative_ratio=None, avg_sentiment=None)
        if case == "no-comparison":
            result.update(comparison_count=0, repeat_ratio=None)
        if case == "no-polarity":
            result.update(polarity_count=0, avg_sentiment=None)
        if case == "partial":
            result.update(sentiment_counts={"Negative": 1, "Neutral": 2}, sentiment_count=3,
                          polarity_count=1, negative_ratio=1 / 3, avg_sentiment=-1.0,
                          comparison_count=2, repeat_ratio=0.5,
                          quick_evaluation={"arbitrary": [0, None, False]},
                          full_report={"summary": {"synthetic": True}, "unknown_extension": []})
        return result
    calls = install_pipeline(monkeypatch, build)
    first = request(client)
    assert first.status_code == 200 and first.json()["cached"] is False
    second = request(client)
    assert second.status_code == 200 and second.json()["cached"] is True
    assert second.json()["result"] == first.json()["result"]
    assert calls == [count]


@pytest.mark.parametrize("missing", [False, True])
def test_actual_full_aggregation_with_synthetic_measurements_is_cacheable(client, monkeypatch, missing):
    seed(client)
    use_measured_frame(monkeypatch, [{"similarity": None if missing else 0.0,
                                    "sentiment": None if missing else "Negative",
                                    "polarity": None if missing else -0.5}] * 5)
    first = request(client)
    assert first.status_code == 200
    monkeypatch.setattr(api, "_execute_lsj_pipeline", lambda _rows: pytest.fail("Valid cache must avoid inference"))
    second = request(client)
    assert second.status_code == 200 and second.json()["cached"] is True
    assert second.json()["result"] == first.json()["result"]


def test_explicit_zero_window_bounds_and_force_keep_the_existing_contract(client, monkeypatch):
    seed(client, 1)
    calls = install_pipeline(monkeypatch)
    params = {"from_ts": 0, "to_ts": 0, "limit_rows": 1}
    first = request(client, **params)
    assert first.status_code == 200 and first.json()["result"]["window"]["input_count"] == 1
    assert request(client, **params).json()["cached"] is True
    assert calls == [1]
    old_id, encoded = poison_latest(replace(("window", "from_ts"), False))
    assert request(client, **params).json()["cached"] is False
    assert_preserved(old_id, encoded)
    assert calls == [1, 1]
    assert request(client, force=True, **params).json()["cached"] is False
    assert calls == [1, 1, 1]
