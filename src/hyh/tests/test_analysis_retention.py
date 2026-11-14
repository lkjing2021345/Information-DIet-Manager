"""Analysis history stays bounded while active jobs remain observable."""
import json

from fastapi.testclient import TestClient

from src.hyh import app as api, db


def test_terminal_history_pruning_keeps_recent_rows_and_active_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "analysis-retention.sqlite3")
    monkeypatch.setattr(api, "MAX_ANALYSIS_JOB_HISTORY", 2)
    monkeypatch.setattr(api, "MAX_ANALYSIS_RUN_HISTORY", 2)

    with TestClient(
        api.app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
        headers={"Authorization": "Bearer " + "a" * 43},
        raise_server_exceptions=False,
    ) as client:
        with db.get_conn() as conn:
            now = 1_800_000_000_000
            for index, status in enumerate((api.JOB_COMPLETED, api.JOB_FAILED, api.JOB_COMPLETED, api.JOB_FAILED), 1):
                conn.execute(
                    """
                    INSERT INTO analysis_jobs (
                        status, input_hash, day, input_count, created_at, updated_at
                    ) VALUES (?, ?, ?, 0, ?, ?)
                    """,
                    (status, f"hash-{index}", "2026-11-12", now + index, now + index),
                )
            conn.execute(
                """
                INSERT INTO analysis_jobs (
                    status, input_hash, day, input_count, created_at, updated_at
                ) VALUES (?, ?, ?, 0, ?, ?)
                """,
                (api.JOB_RUNNING, "active", "2026-11-12", now + 10, now + 10),
            )
            for index in range(4):
                conn.execute(
                    """
                    INSERT INTO analysis_runs (
                        day, total_count, payload, created_at
                    ) VALUES (?, 0, ?, ?)
                    """,
                    ("2026-11-12", json.dumps({"index": index}), now + index),
                )
            api._prune_analysis_history(conn)

        with db.get_conn() as conn:
            terminal = conn.execute(
                "SELECT id, status FROM analysis_jobs WHERE status IN (?, ?) ORDER BY id",
                (api.JOB_COMPLETED, api.JOB_FAILED),
            ).fetchall()
            active = conn.execute(
                "SELECT status FROM analysis_jobs WHERE input_hash = 'active'"
            ).fetchall()
            runs = conn.execute("SELECT payload FROM analysis_runs ORDER BY id").fetchall()

        assert [row["status"] for row in terminal] == [api.JOB_COMPLETED, api.JOB_FAILED]
        assert [row["status"] for row in active] == [api.JOB_RUNNING]
        assert [json.loads(row["payload"])["index"] for row in runs] == [2, 3]

        history = client.get("/analyze/history?limit=1")
        assert history.status_code == 200
        assert history.json()["total"] == 2
        assert history.json()["returned"] == 1
        assert len(history.json()["runs"]) == 1
