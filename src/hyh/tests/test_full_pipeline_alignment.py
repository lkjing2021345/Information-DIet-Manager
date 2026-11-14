"""Full-analysis output contracts with real HTTP/SQLite and synthetic inference.

No model is loaded; faults model a broken adapter boundary, not observed model
behavior. The evaluator is a spy so rejection must precede either scoring call.
"""
import numpy as np
import pandas as pd
import pytest

from src.hyh import app as api, db
from src.hyh.tests.test_legacy_statistics import client, full, seed


PRIVATE_MARKER = "SYNTHETIC_PRIVATE_ALIGNMENT_FAILURE"
SAFE_FAILURE = "Legacy analysis failed; no valid result was produced."


class BrokenFrame:
    def __len__(self):
        raise ValueError(PRIVATE_MARKER)


def install_pipeline(monkeypatch, fault):
    calls = []

    class Report:
        def to_dict(self):
            return {"synthetic_report": True}

    class Evaluator:
        def quick_evaluate(self, frame):
            calls.append("quick")
            return {"synthetic_quick": True}

        def evaluate(self, frame, detailed=False):
            calls.append("full")
            return Report()

    def execute(rows):
        frame = pd.DataFrame([{**row, "category": "Tools", "sentiment": "positive", "polarity": 0.5,
                               "similarity": 0.0, "sentiment_valid": True, "similarity_valid": True}
                              for row in rows])
        count = len(rows)
        if fault == "reindexed":
            frame.index = range(30, 30 + count)
        elif fault == "drop":
            frame = frame.iloc[1:]
        elif fault == "extra":
            frame = pd.concat([frame, frame.iloc[:1]])
        elif fault == "reorder":
            frame = frame.iloc[::-1]
        elif fault == "duplicate":
            frame.iloc[-1] = frame.iloc[0]
        elif fault == "timestamp":
            frame.loc[0, "ts"] += 86400000
        elif fault in {"missing_id", "missing_ts"}:
            frame = frame.drop(columns=[fault.removeprefix("missing_")])
        elif fault == "none_frame":
            frame = None
        elif fault == "list_frame":
            frame = frame.to_dict("records")
        elif fault == "broken_frame":
            frame = BrokenFrame()
        counts = {"bool_count": True, "float_count": float(count), "string_count": str(count),
                  "null_count": None, "list_count": [], "numpy_count": np.int64(count),
                  "high_count": count + 1, "low_count": count - 1}
        result = {"ok": True, "input_count": counts.get(fault, count), "df3": frame, "evaluator": Evaluator()}
        if fault == "missing_count":
            del result["input_count"]
        return result

    monkeypatch.setattr(api, "_execute_lsj_pipeline", execute)
    return calls


@pytest.mark.parametrize("fault", [
    "drop", "extra", "reorder", "duplicate", "timestamp", "missing_id", "missing_ts",
    "none_frame", "list_frame", "broken_frame", "bool_count", "float_count", "string_count",
    "null_count", "list_count", "numpy_count", "high_count", "low_count", "missing_count",
])
def test_full_rejects_misaligned_output_before_scoring_and_does_not_publish(client, monkeypatch, caplog, fault):
    seed(client, 6)
    with db.get_conn() as conn:
        original_items = [tuple(row) for row in conn.execute("SELECT * FROM items ORDER BY id")]
    calls = install_pipeline(monkeypatch, fault)
    response = full(client)
    assert response.status_code == 500
    assert response.json() == {"job_id": response.json()["job_id"], "status": "failed", "detail": SAFE_FAILURE}
    assert calls == []
    job = client.get(f"/analyze/jobs/{response.json()['job_id']}")
    assert job.status_code == 200
    assert job.json()["status"] == "failed" and job.json()["result_payload"] is None
    assert job.json()["error"] == SAFE_FAILURE
    assert client.get(f"/analyze/result/{response.json()['job_id']}").status_code == 409
    assert client.get("/analyze/history").json()["total"] == 0
    assert PRIVATE_MARKER not in response.text + job.text + caplog.text
    with db.get_conn() as conn:
        assert [tuple(row) for row in conn.execute("SELECT * FROM items ORDER BY id")] == original_items
        assert conn.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM stats_daily").fetchone()[0] == 0
    # The failed attempt cannot become a cache hit; a repaired adapter can retry.
    good_calls = install_pipeline(monkeypatch, "aligned")
    recovered = full(client)
    assert recovered.status_code == 200
    assert recovered.json()["status"] == "completed" and recovered.json()["cached"] is False
    assert good_calls == ["quick", "full"]


@pytest.mark.parametrize("fault", ["aligned", "reindexed"])
def test_full_accepts_aligned_frames_without_requiring_a_default_index(client, monkeypatch, fault):
    seed(client, 6)
    calls = install_pipeline(monkeypatch, fault)
    response = full(client)
    assert response.status_code == 200
    result = response.json()["result"]
    assert response.json()["status"] == "completed" and calls == ["quick", "full"]
    assert result["total_count"] == result["window"]["input_count"] == 6
    assert result["category_counts"] == {"Tools": 6}
    assert result["sentiment_count"] == result["polarity_count"] == 6
    assert result["comparison_count"] == 5
    assert result["full_report"] == {"synthetic_report": True}
    assert result["experimental"] is True and "diagnostic" in result["experimental_notice"]


def test_empty_full_analysis_keeps_its_existing_no_inference_path(client, monkeypatch):
    monkeypatch.setattr(api, "_execute_lsj_pipeline", lambda rows: pytest.fail("Empty input must not load models"))
    response = full(client)
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["analysis_status"] == "empty" and result["total_count"] == 0
    assert result["experimental"] is True
    assert result["quick_evaluation"] is result["full_report"] is None
