"""Recover interrupted jobs only while starting an exclusively owned database.

Synthetic SQLite state and controlled inference exercise startup/transactions;
no model is constructed, downloaded or evaluated.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

import pytest
from fastapi.testclient import TestClient

from src.hyh import app as api, db
from src.hyh.security import process_ownership


DETECTED_AT = 1790208000123
INTERRUPTED = "Analysis was interrupted before this service start; run it again."
STATUSES = ("queued", "running", "completed", "failed", "synthetic-unknown", "RUNNING")
PRIVATE_MARKER = "SYNTHETIC_PRIVATE_RESTART_CONTENT"
SCHEMA = Path(db.__file__).with_name("schema.sql")
PROJECT = Path(__file__).resolve().parents[3]


def insert_job(conn, job_id, status):
    conn.execute(
        "INSERT INTO analysis_jobs(id,status,input_hash,day,from_ts,to_ts,limit_rows,"
        "item_max_created_at,input_count,cache_hit,duration_ms,error,result_payload,"
        "metrics_json,started_at,finished_at,created_at,updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (job_id, status, f"synthetic-hash-{job_id}", "2026-09-24", 10, 20, 30,
         40, 1, 1, 50, PRIVATE_MARKER, '{"synthetic":"retained result"}',
         '{"synthetic":"retained metrics"}', None if status == "queued" else 60,
         None if status in {"queued", "running"} else 70, 80, 90),
    )


@pytest.fixture
def database(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "synthetic-analysis-restart.sqlite3")
    monkeypatch.setattr(api, "_now_ms", lambda: DETECTED_AT)
    db.init_db(SCHEMA)
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO items(id,url,title,text,ts,source,created_at) VALUES (1,?,?,?,?,?,?)",
            ("https://example.invalid/restart", PRIVATE_MARKER, "Synthetic body", 10, "import", 20),
        )
        conn.execute("UPDATE items SET title=title WHERE id=1")  # Nonzero revision.
        conn.execute("INSERT INTO embeddings(item_id,vector,model,dim,created_at) VALUES (1,?,?,2,20)",
                     (b"\0" * 8, "synthetic-vector"))
        conn.execute("INSERT INTO stats_daily(day,total_count,channel_counts,created_at,updated_at) "
                     "VALUES ('2026-09-24',1,'{\"unknown\":1}',20,30)")
        conn.execute("INSERT INTO analysis_runs(day,total_count,payload,created_at) "
                     "VALUES ('2026-09-24',1,'{\"synthetic\":true}',20)")
        for job_id, status in enumerate(STATUSES, 1):
            insert_job(conn, job_id, status)
    return db.DB_PATH


def snapshot():
    # Read independently from the application connection wrapper so injected
    # startup commit failures cannot mask the persisted-state assertion.
    conn = sqlite3.connect(db.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        names = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return {name: [dict(row) for row in conn.execute(f'SELECT * FROM "{name}" ORDER BY rowid')]
                for name in names}
    finally:
        conn.close()


def client():
    return TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 51000),
                      headers={"Authorization": "Bearer " + "a" * 43}, raise_server_exceptions=False)


def test_startup_recovers_only_interrupted_jobs_and_repeated_start_is_idempotent(database, monkeypatch):
    before = snapshot()
    expected = deepcopy(before)
    for row in expected["analysis_jobs"]:
        if row["status"] in {"queued", "running"}:
            row.update(status="failed", error=INTERRUPTED, updated_at=DETECTED_AT, finished_at=DETECTED_AT)

    def no_automatic_model_run(*_args, **_kwargs):
        pytest.fail("Restart recovery must not run analysis automatically")

    monkeypatch.setattr(api, "_run_lsj_pipeline", no_automatic_model_run)
    monkeypatch.setattr(api, "_execute_lsj_pipeline", no_automatic_model_run)
    with client() as http:
        assert http.get("/health").status_code == 200
        assert snapshot() == expected
        for job_id in (1, 2):
            job = http.get(f"/analyze/jobs/{job_id}")
            assert job.status_code == 200
            assert job.json()["status"] == "failed" and job.json()["error"] == INTERRUPTED
            assert job.json()["updated_at"] == job.json()["finished_at"] == DETECTED_AT
            result = http.get(f"/analyze/result/{job_id}")
            assert result.status_code == 409 and result.json() == {"detail": "job not completed: failed"}
        completed = http.get("/analyze/result/3")
        assert completed.status_code == 200
        assert completed.json()["result"] == {"synthetic": "retained result"}
        assert http.get("/items").json()["total"] == 1
        assert snapshot() == expected
    monkeypatch.setattr(api, "_now_ms", lambda: DETECTED_AT + 9999)
    with client() as http:
        assert http.get("/health").status_code == 200
        assert snapshot() == expected
    assert snapshot() == expected


def test_startup_recovery_prunes_old_history_without_dropping_active_jobs(database, monkeypatch):
    """Startup cleanup applies retention after recovery while preserving active work."""
    monkeypatch.setattr(api, "MAX_ANALYSIS_JOB_HISTORY", 2)
    monkeypatch.setattr(api, "MAX_ANALYSIS_RUN_HISTORY", 2)
    with db.get_conn() as conn:
        conn.execute("DELETE FROM analysis_jobs")
        for job_id, status in (
            (1, "completed"),
            (2, "failed"),
            (3, "completed"),
            (4, "queued"),
            (5, "running"),
            (6, "synthetic-active"),
        ):
            insert_job(conn, job_id, status)
        conn.execute("DELETE FROM analysis_runs")
        for index in range(4):
            conn.execute(
                "INSERT INTO analysis_runs(day,total_count,payload,created_at) VALUES (?,?,?,?)",
                (f"2026-09-{index + 20:02d}", index + 1, '{"synthetic":true}', 100 + index),
            )

    with client() as http:
        assert http.get("/health").status_code == 200

    with db.get_conn() as conn:
        jobs = [dict(row) for row in conn.execute(
            "SELECT id,status,error FROM analysis_jobs ORDER BY id"
        )]
        runs = [dict(row) for row in conn.execute(
            "SELECT id,created_at FROM analysis_runs ORDER BY id"
        )]
    assert [job["id"] for job in jobs] == [4, 5, 6]
    assert all(job["status"] == "failed" and job["error"] == INTERRUPTED for job in jobs[:2])
    assert jobs[2] == {"id": 6, "status": "synthetic-active", "error": PRIVATE_MARKER}
    assert len(runs) == 2 and [run["created_at"] for run in runs] == [102, 103]


@pytest.mark.parametrize("failure", ["update", "commit"])
def test_recovery_failure_rolls_back_all_changes_and_prevents_readiness(database, monkeypatch, failure):
    before = snapshot()
    commits = []
    if failure == "update":
        # FAIL leaves changes already performed by this statement pending; the
        # surrounding recovery transaction must roll them back before exit.
        with db.get_conn() as conn:
            conn.execute("CREATE TRIGGER synthetic_recovery_failure AFTER UPDATE OF status ON analysis_jobs "
                         "BEGIN SELECT RAISE(FAIL, 'Synthetic recovery update failure'); END")
    with monkeypatch.context() as patch:
        if failure == "commit":
            original_connect = db.sqlite3.connect

            class FailingCommit(sqlite3.Connection):
                def commit(self):
                    if self.in_transaction and self.execute(
                        "SELECT COUNT(*) FROM analysis_jobs WHERE error=?", (INTERRUPTED,),
                    ).fetchone()[0]:
                        commits.append(True)
                        raise sqlite3.OperationalError("Synthetic recovery commit failure")
                    return super().commit()

            def connect(*args, **kwargs):
                return original_connect(*args, factory=FailingCommit, **kwargs)

            patch.setattr(db.sqlite3, "connect", connect)
        entered = False
        with pytest.raises(sqlite3.DatabaseError, match="Synthetic recovery"):
            with client():
                entered = True
        assert not entered
    if failure == "commit":
        assert commits == [True]
    assert snapshot() == before
    # Both the SQLite transaction and OS process ownership must be released.
    with process_ownership(database):
        with db.get_conn() as conn:
            conn.execute("BEGIN EXCLUSIVE")
            if failure == "update":
                conn.execute("DROP TRIGGER synthetic_recovery_failure")
    with client() as http:
        assert http.get("/health").status_code == 200
        assert http.get("/analyze/jobs/1").json()["error"] == INTERRUPTED


def test_competing_process_is_rejected_before_recovering_active_owner_jobs(database):
    before = snapshot()
    code = """
import asyncio
import sys
sys.path.insert(0, sys.argv[1])
from src.hyh import app as api
async def attempt():
    async with api.lifespan(api.app):
        print('UNEXPECTED_STARTUP_SUCCESS')
try:
    asyncio.run(attempt())
except RuntimeError as exc:
    if 'Another IDM process' in str(exc):
        print('PROCESS_LOCK_REJECTED')
        sys.exit(9)
    raise
"""
    env = {**os.environ, "IDM_DB_PATH": str(database), "PYTHONUTF8": "1",
           "IDM_ADMIN_TOKEN": "a" * 43, "IDM_COLLECTOR_TOKEN": "c" * 43,
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    with process_ownership(database):
        result = subprocess.run([sys.executable, "-I", "-c", code, str(PROJECT)], cwd=PROJECT,
                                env=env, capture_output=True, text=True, encoding="utf-8", timeout=20)
        assert result.returncode == 9, result.stdout + result.stderr
        assert result.stdout.strip() == "PROCESS_LOCK_REJECTED"
        assert PRIVATE_MARKER not in result.stdout + result.stderr
        assert snapshot() == before
    assert snapshot() == before


def test_current_running_job_is_not_recovered_by_ordinary_requests(database, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def controlled_unavailable(_rows):
        entered.set()
        assert release.wait(15)
        raise api.AnalysisUnavailableError("Synthetic unavailable inference")

    monkeypatch.setattr(api, "_run_lsj_pipeline", controlled_unavailable)
    with client() as http, ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(http.post, "/analyze/run_full?force=true")
        try:
            assert entered.wait(5)
            before = snapshot()
            running = [row for row in before["analysis_jobs"] if row["status"] == "running"]
            assert len(running) == 1
            job_id = running[0]["id"]
            for _ in range(2):
                assert http.get("/health").status_code == 200
                job = http.get(f"/analyze/jobs/{job_id}")
                assert job.status_code == 200 and job.json()["status"] == "running"
                result = http.get(f"/analyze/result/{job_id}")
                assert result.status_code == 409
                assert result.json() == {"detail": "job not completed: running"}
                assert snapshot() == before
                assert not pending.done()
        finally:
            release.set()
        assert pending.result(timeout=10).status_code == 503
        final = http.get(f"/analyze/jobs/{job_id}").json()
        assert final["status"] == "failed" and final["error"] != INTERRUPTED
