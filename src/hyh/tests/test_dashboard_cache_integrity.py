"""Validate cache reuse via real HTTP/temporary SQLite; model outputs are synthetic."""
import json

import pytest
from fastapi.testclient import TestClient

from src.hyh import app as api, db
from src.hyh.tests.test_dashboard_visualization import get_analysis, pipeline, seed


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "dashboard-cache.sqlite3")
    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000),
                    headers={"Authorization": "Bearer " + "a" * 43}) as client:
        yield client


def poison_latest(change):
    with db.get_conn() as conn:
        row = conn.execute("SELECT id, result_payload FROM analysis_jobs ORDER BY id DESC LIMIT 1").fetchone()
        payload = json.loads(row["result_payload"])
        change(payload)
        encoded = json.dumps(payload, allow_nan=False)
        conn.execute("UPDATE analysis_jobs SET result_payload = ? WHERE id = ?", (encoded, row["id"]))
        return row["id"], encoded


def replace(path, value):
    def change(payload):
        current = payload
        for key in path[:-1]:
            current = current[key]
        current[path[-1]] = value
    return change


def remove(path):
    def change(payload):
        current = payload
        for key in path[:-1]:
            current = current[key]
        del current[path[-1]]
    return change


def ready_only(payload):
    payload.clear()
    payload["analysis_status"] = "ready"


CORRUPTIONS = [
    ("ready-only", ready_only),
    ("missing-global", remove(("global",))),
    ("missing-coverage", remove(("coverage",))),
    ("missing-counts", remove(("category_counts",))),
    ("missing-window", remove(("window",))),
    ("missing-series", remove(("global", "time_series"))),
    ("series-not-list", replace(("global", "time_series"), {})),
    ("input-not-integer", replace(("window", "input_count"), "5")),
    ("processed-mismatch", replace(("window", "processed_count"), 4)),
    ("available-mismatch", replace(("window", "available_count"), 6)),
    ("truncated-mismatch", replace(("window", "truncated"), True)),
    ("window-mismatch", replace(("window", "from_ts"), 1790208000001)),
    ("record-count-mismatch", replace(("coverage", "record_count"), 4)),
    ("count-is-bool", replace(("coverage", "timestamp_count"), True)),
    ("too-many-comparisons", replace(("coverage", "comparison_count"), 5)),
    ("date-not-utc", replace(("date_timezone",), "local")),
    ("generated-not-integer", replace(("generated_at",), "yesterday")),
    ("wrong-category-alias", replace(("category_aliases", "tools"), "other")),
    ("negative-category-count", replace(("category_counts", "tools"), -3)),
    ("category-series-missing", remove(("categories", "tools", "time_series"))),
    ("category-series-drops-records", replace(("categories", "tools", "time_series"), [])),
    ("category-label-mismatch", replace(("categories", "tools", "label"), "other")),
    ("daily-count-mismatch", replace(("global", "time_series", 0, "count"), 4)),
    ("daily-count-is-bool", replace(("global", "time_series", 0, "count"), True)),
    ("daily-field-missing", remove(("global", "time_series", 0, "repeat_ratio"))),
    ("invalid-date", replace(("global", "time_series", 0, "date"), "2026-02-30")),
    ("daily-too-many-comparisons", replace(("global", "time_series", 0, "comparison_count"), 6)),
    ("daily-polarity-count-mismatch", replace(("global", "time_series", 0, "polarity_count"), 6)),
    ("daily-measurement-missing", replace(("global", "time_series", 0, "repeat_ratio"), None)),
    ("daily-measurement-is-bool", replace(("global", "time_series", 0, "avg_similarity"), False)),
    ("daily-sentiment-ratio-mismatch", replace(("global", "time_series", 0, "neutral_ratio"), 1.0)),
    ("category-distribution-mismatch", replace(("global", "category_distribution", "tools"), 0.2)),
    ("sentiment-distribution-not-map", replace(("global", "sentiment_distribution"), [])),
    ("histogram-count-mismatch", replace(("global", "similarity_histogram", "0.0-0.2"), 3)),
    ("hour-out-of-range", replace(("global", "hourly_distribution"), {"24": 5})),
]


@pytest.mark.parametrize("_name,change", CORRUPTIONS, ids=[name for name, _ in CORRUPTIONS])
def test_invalid_ready_cache_recomputes_without_rewriting_history(client, monkeypatch, _name, change):
    seed(client, 5)
    calls = []

    def execute(rows):
        calls.append(len(rows))
        return pipeline()(rows)

    monkeypatch.setattr(api, "_execute_lsj_pipeline", execute)
    initial = get_analysis(client).json()
    assert initial["analysis_status"] == "ready"
    old_id, encoded = poison_latest(change)
    response = get_analysis(client)
    assert response.status_code == 200
    fresh = response.json()
    assert fresh["cached"] is False
    assert fresh["analysis_status"] == "ready"
    assert fresh["global"] == initial["global"]
    assert fresh["window"] == initial["window"]
    assert calls == [5, 5]
    with db.get_conn() as conn:
        assert conn.execute("SELECT result_payload FROM analysis_jobs WHERE id = ?", (old_id,)).fetchone()[0] == encoded
        latest = conn.execute("SELECT id, status, cache_hit FROM analysis_jobs ORDER BY id DESC LIMIT 1").fetchone()
        assert latest["id"] != old_id and latest["status"] == api.JOB_COMPLETED and not latest["cache_hit"]
    assert get_analysis(client).json()["cached"] is True
    assert calls == [5, 5]


@pytest.mark.parametrize("failure", ("unavailable", "failed"))
def test_bad_cache_cannot_hide_recompute_failure(client, monkeypatch, failure):
    seed(client, 5)
    monkeypatch.setattr(api, "_execute_lsj_pipeline", pipeline())
    assert get_analysis(client).json()["analysis_status"] == "ready"
    old_id, encoded = poison_latest(ready_only)
    execute = pipeline(fail=True) if failure == "failed" else lambda rows: {"ok": False, "warning": "synthetic outage"}
    monkeypatch.setattr(api, "_execute_lsj_pipeline", execute)
    response = get_analysis(client)
    assert response.status_code == 200
    result = response.json()
    assert result["analysis_status"] == failure and result["cached"] is False
    assert result["pipeline_warning"] and result["global"]["time_series"] == []
    with db.get_conn() as conn:
        assert conn.execute("SELECT result_payload FROM analysis_jobs WHERE id = ?", (old_id,)).fetchone()[0] == encoded
        row = conn.execute("SELECT status, cache_hit FROM analysis_jobs ORDER BY id DESC LIMIT 1").fetchone()
        assert row["status"] == api.JOB_FAILED and not row["cache_hit"]


@pytest.mark.parametrize("missing", ("none", "category", "sentiment", "similarity", "polarity", "all", "partial", "truncated"))
def test_legitimate_independent_coverage_and_zeroes_are_reusable(client, monkeypatch, missing):
    seed(client, 6 if missing == "truncated" else 5)

    def execute(rows):
        result = pipeline()(rows)
        frame = result["df3"]
        for column in ("category", "sentiment", "similarity", "polarity"):
            if missing in (column, "all"):
                frame[column] = None
        if missing == "partial":
            frame.loc[1, "category"] = None
            frame.loc[2, "sentiment"] = None
            frame.loc[3, "similarity"] = None
            frame.loc[4, "polarity"] = None
        return result

    monkeypatch.setattr(api, "_execute_lsj_pipeline", execute)
    params = {"limit_rows": 5} if missing == "truncated" else {}
    initial = get_analysis(client, **params).json()
    assert initial["analysis_status"] == "ready" and initial["cached"] is False

    def unexpected(_rows):
        pytest.fail("A valid current-schema cache must avoid another model call")

    monkeypatch.setattr(api, "_execute_lsj_pipeline", unexpected)
    cached = get_analysis(client, **params).json()
    assert cached["analysis_status"] == "ready" and cached["cached"] is True
    for key in ("window", "global", "categories", "category_counts", "coverage", "generated_at"):
        assert cached[key] == initial[key]


def test_force_keeps_bypassing_valid_cache(client, monkeypatch):
    seed(client, 5)
    monkeypatch.setattr(api, "_execute_lsj_pipeline", pipeline())
    assert get_analysis(client).json()["analysis_status"] == "ready"
    monkeypatch.setattr(api, "_execute_lsj_pipeline", lambda rows: {"ok": False, "warning": "synthetic outage"})
    result = get_analysis(client, force=True).json()
    assert result["cached"] is False and result["analysis_status"] == "unavailable"


@pytest.mark.parametrize("start,end,record_start,dates", [
    (0, api.MAX_INGEST_TS, 0, ["1970-01-01"]),
    (0, 4, 0, ["1970-01-01"]),
    (api.MAX_INGEST_TS - 4, api.MAX_INGEST_TS, api.MAX_INGEST_TS - 4, ["9999-12-31"]),
    (86399998, 86400002, 86399998, ["1970-01-01", "1970-01-02"]),
], ids=("full-range", "epoch", "upper-bound", "cross-day"))
def test_supported_timestamp_bounds_remain_cacheable(client, monkeypatch, start, end, record_start, dates):
    for index in range(5):
        response = client.post("/collect", json={
            "url": f"https://example.invalid/date-bound/{index}", "title": f"Synthetic {index}",
            "ts": record_start + index, "source": "import",
        })
        assert response.status_code == 200 and response.json()["inserted"] == 1
    monkeypatch.setattr(api, "_execute_lsj_pipeline", pipeline())
    initial_response = get_analysis(client, from_ts=start, to_ts=end)
    assert initial_response.status_code == 200
    initial = initial_response.json()
    assert initial["analysis_status"] == "ready" and initial["cached"] is False
    assert [row["date"] for row in initial["global"]["time_series"]] == dates

    def unexpected(_rows):
        pytest.fail("Valid timestamp bounds must reuse the ready cache")

    monkeypatch.setattr(api, "_execute_lsj_pipeline", unexpected)
    cached_response = get_analysis(client, from_ts=start, to_ts=end)
    assert cached_response.status_code == 200
    cached = cached_response.json()
    assert cached["analysis_status"] == "ready" and cached["cached"] is True
    for key in ("global", "window", "generated_at"):
        assert cached[key] == initial[key]
