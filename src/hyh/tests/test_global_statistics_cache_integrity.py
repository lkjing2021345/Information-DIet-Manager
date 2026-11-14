"""Basic global statistics must not reuse finite but contradictory old caches.

Real local API and temporary SQLite only; no inference is involved.
"""
import json

from fastapi.testclient import TestClient
import pytest

from src.hyh import app as api, db
from src.hyh.models import MAX_INGEST_TS


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "synthetic-global-cache.sqlite3")
    monkeypatch.setattr(api, "_execute_lsj_pipeline", lambda _rows: pytest.fail("Basic statistics must not run models"))
    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 51212),
                    headers={"Authorization": "Bearer " + "a" * 43}, raise_server_exceptions=False) as client:
        yield client


def seed(client, count=4):
    for index in range(count):
        response = client.post("/collect", json={
            "url": f"https://example.invalid/basic-cache/{index}", "title": f"Synthetic pair {index // 2}",
            "text": "Synthetic body", "ts": (0, 86399999, 86400000, MAX_INGEST_TS)[index],
            "source": "import", "channel": ("EDU", "unknown", None, "")[index],
        })
        assert response.status_code == 200 and response.json()["inserted"] == 1


def request(client, endpoint):
    return client.get(endpoint) if endpoint == "/dashboard/summary" else client.post(endpoint)


def poison_latest(change):
    with db.get_conn() as conn:
        row = conn.execute("SELECT id, payload FROM analysis_runs ORDER BY id DESC LIMIT 1").fetchone()
        payload = json.loads(row["payload"])
        change(payload)
        encoded = json.dumps(payload, allow_nan=False)
        conn.execute("UPDATE analysis_runs SET payload=? WHERE id=?", (encoded, row["id"]))
        return row["id"], encoded


def snapshot():
    with db.get_conn() as conn:
        return {table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
                for table in ("items", "embeddings", "items_revision", "analysis_runs", "stats_daily")}


def replace(key, value):
    return lambda payload: payload.__setitem__(key, value)


def remove(key):
    return lambda payload: payload.pop(key)


CORRUPTIONS = [
    ("missing-channel-map", remove("channel_counts")),
    ("missing-null-sentiment", remove("negative_ratio")),
    ("missing-samples", remove("repeat_sample_count")),
    ("stale-day", replace("day", "1970-01-01")),
    ("bool-version", replace("statistics_version", True)),
    ("float-total", replace("total_count", 4.0)),
    ("channel-negative", replace("channel_counts", {"ent": 0, "edu": 2, "news": 0, "soc": -1, "other": 3})),
    ("channel-bool", replace("channel_counts", {"ent": 0, "edu": True, "news": 0, "soc": 0, "other": 3})),
    ("channel-wrong-sum", replace("channel_counts", {"ent": 0, "edu": 999, "news": 0, "soc": 0, "other": 3})),
    ("channel-unknown-key", replace("channel_counts", {"ent": 0, "edu": 1, "news": 0, "soc": 0, "alien": 3})),
    ("too-many-samples", replace("repeat_sample_count", 5)),
    ("samples-bool", replace("repeat_sample_count", True)),
    ("null-with-samples", replace("repeat_ratio", None)),
    ("ratio-negative", replace("repeat_ratio", -0.1)),
    ("ratio-impossible", replace("repeat_ratio", 1.0)),
    ("ratio-bool", replace("repeat_ratio", False)),
    ("wrong-repeat-metric", replace("repeat_metric", "different_measurement")),
    ("invented-sentiment", replace("negative_ratio", 0.75)),
    ("invented-polarity", replace("avg_sentiment", -0.5)),
    ("timestamp-bool", replace("generated_at", False)),
    ("timestamp-overflow", replace("generated_at", MAX_INGEST_TS + 1)),
    ("backfill-negative", replace("embeddings_backfilled", -1)),
    ("cached-not-bool", replace("cached", 1)),
]


@pytest.mark.parametrize("_name,change", CORRUPTIONS, ids=[name for name, _ in CORRUPTIONS])
def test_invalid_basic_cache_recomputes_repairs_daily_and_preserves_old_run(client, _name, change):
    seed(client)
    initial = client.post("/analyze/run").json()
    assert initial["total_count"] == 4 and initial["repeat_ratio"] == 0.5
    old_id, encoded = poison_latest(change)
    response = client.get("/dashboard/summary")
    assert response.status_code == 200 and response.json()["cached"] is False
    fresh = response.json()
    for key in ("day", "total_count", "channel_counts", "repeat_ratio", "repeat_sample_count",
                "repeat_metric", "negative_ratio", "avg_sentiment", "data_version"):
        assert fresh[key] == initial[key]
    assert fresh["channel_counts"] == {"ent": 0, "edu": 1, "news": 0, "soc": 0, "other": 3}
    with db.get_conn() as conn:
        assert conn.execute("SELECT payload FROM analysis_runs WHERE id=?", (old_id,)).fetchone()[0] == encoded
        assert conn.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 2
        daily = dict(conn.execute("SELECT * FROM stats_daily").fetchone())
    assert json.loads(daily["channel_counts"]) == fresh["channel_counts"]
    assert daily["repeat_ratio"] == 0.5 and daily["negative_ratio"] is daily["avg_sentiment"] is None
    cached = client.get("/dashboard/summary").json()
    assert cached["cached"] is True and cached["generated_at"] == fresh["generated_at"]
    assert len(snapshot()["analysis_runs"]) == 2


@pytest.mark.parametrize("endpoint", ["/dashboard/summary", "/analyze/run"])
def test_missing_hash_measurements_cannot_become_zero_in_cached_statistics(client, endpoint):
    seed(client)
    with db.get_conn() as conn:
        conn.execute("UPDATE items SET content_hash=NULL")
    initial = client.post("/analyze/run").json()
    assert initial["repeat_sample_count"] == 0 and initial["repeat_ratio"] is None
    poison_latest(replace("repeat_ratio", 0.0))
    response = request(client, endpoint)
    assert response.status_code == 200
    assert response.json()["cached"] is False
    assert response.json()["repeat_sample_count"] == 0 and response.json()["repeat_ratio"] is None
    assert len(snapshot()["analysis_runs"]) == 2


@pytest.mark.parametrize("endpoint", ["/dashboard/summary", "/analyze/run"])
def test_failed_recompute_rolls_back_backfill_and_preserves_existing_daily_and_history(client, monkeypatch, endpoint):
    seed(client)
    assert client.post("/analyze/run").status_code == 200
    with db.get_conn() as conn:
        conn.execute("DELETE FROM embeddings WHERE item_id=1")
    poison_latest(remove("channel_counts"))
    before = snapshot()
    original = api._backfill_missing_embeddings
    calls = []
    def fail_after_backfill(conn, limit):
        calls.append(original(conn, limit=limit))
        raise RuntimeError("SYNTHETIC_PRIVATE_BASIC_RECOMPUTE_FAILURE")
    monkeypatch.setattr(api, "_backfill_missing_embeddings", fail_after_backfill)
    response = request(client, endpoint)
    assert response.status_code == 500 and response.json() == {"detail": "Local data operation failed"}
    assert calls == [1]
    assert snapshot() == before
    assert not api.app.state.work_owner._workers and not api.app.state.operation_lock.locked()
    assert api.app.state.request_slots._value == 4
    monkeypatch.setattr(api, "_backfill_missing_embeddings", original)
    recovered = request(client, endpoint)
    assert recovered.status_code == 200 and recovered.json()["cached"] is False
    assert recovered.json()["embeddings_backfilled"] == 1


@pytest.mark.parametrize("case", ["empty", "single", "missing-hashes", "partial-hashes", "zero", "ordinary"])
def test_valid_basic_caches_keep_null_zero_day_and_history_semantics(client, case):
    count = 0 if case == "empty" else (1 if case == "single" else 4)
    seed(client, count)
    with db.get_conn() as conn:
        if case == "missing-hashes":
            conn.execute("UPDATE items SET content_hash=NULL")
        elif case == "partial-hashes":
            conn.execute("UPDATE items SET content_hash=NULL WHERE id<=2")
        elif case == "zero":
            conn.execute("UPDATE items SET content_hash=CAST(id AS TEXT)")
    initial = client.post("/analyze/run").json()
    cached = client.get("/dashboard/summary").json()
    assert cached == {**initial, "cached": True, "embeddings_backfilled": 0}
    assert len(snapshot()["analysis_runs"]) == 1
    recorded = client.post("/analyze/run").json()
    assert recorded["cached"] is True and recorded["generated_at"] == initial["generated_at"]
    assert len(snapshot()["analysis_runs"]) == 2
    forced = client.post("/analyze/run?force=true").json()
    assert forced["cached"] is False and len(snapshot()["analysis_runs"]) == 3
    if case in {"empty", "missing-hashes"}:
        assert cached["repeat_sample_count"] == 0 and cached["repeat_ratio"] is None
    elif case in {"single", "zero"}:
        assert cached["repeat_ratio"] == 0
    else:
        assert cached["repeat_ratio"] == 0.5
