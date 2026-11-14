"""Known historical SQL schemas with synthetic rows; no historical DB files."""
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest
from fastapi.testclient import TestClient

from src.hyh import app as api, db


FIXTURES = Path(__file__).parent / "fixtures" / "historical_schemas"
SCHEMA = Path(db.__file__).with_name("schema.sql")
# Exact commits and src/hyh/schema.sql Git blobs; no Git process is needed by CI.
BASELINES = [
    pytest.param("3d68b00d40f52cda579c927c6dba09b1e21b5752", "7d5ac26b26f64e6bb6e3dbfa0baa6c71ee2ad11e", id="items-3d68b00"),
    pytest.param("9ca3eb81753bc43b410bcdd669f4eb26903e7f1c", "092f5262257ea13a0f203461bf41fd7bda9e19d6", id="statistics-9ca3eb8"),
    pytest.param("953a2894f94ddd5329bcf7d4d4bd87d302f0f343", "68ba1c969a0f3ed971717fab2538e39ba3c83da7", id="analysis-953a289"),
]


def stored_rows(connection):
    # Names are from the fixed, hash-verified schema fixtures or current schema.
    names = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    return {name: [tuple(row) for row in connection.execute('SELECT * FROM "' + name + '" ORDER BY rowid')]
            for name in names}


def schema_objects(connection):
    return [tuple(row) for row in connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")]


def seed_historical_database(path, source):
    expected = {
        "url": "https://example.invalid/historical-schema", "title": "Synthetic 历史记录",
        "text": "Synthetic content", "ts": 1790208000000, "source": "import",
        "lang": "zh-CN", "channel": "edu", "author": "Synthetic author",
        "tags": ["synthetic"], "meta": {"fixture": "synthetic"},
    }
    connection = sqlite3.connect(path)
    try:
        connection.executescript(source)
        connection.execute(
            "INSERT INTO items(id,url,title,text,ts,source,lang,channel,author,tags,meta,url_hash,content_hash,created_at) "
            "VALUES (7,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (expected["url"], expected["title"], expected["text"], expected["ts"], expected["source"],
             expected["lang"], expected["channel"], expected["author"], json.dumps(expected["tags"]),
             json.dumps(expected["meta"]), hashlib.sha256(expected["url"].encode()).hexdigest(),
             hashlib.sha256(b"Synthetic content").hexdigest(), expected["ts"]),
        )
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "embeddings" in tables:
            connection.execute("INSERT INTO embeddings(item_id,vector,model,dim,created_at) VALUES (7,?,?,2,?)",
                               (b"\0" * 8, "synthetic-vector", expected["ts"]))
        if "stats_daily" in tables:
            connection.execute("INSERT INTO stats_daily(day,total_count,channel_counts,created_at,updated_at) VALUES (?,1,?,?,?)",
                               ("2026-09-24", '{"edu":1}', expected["ts"], expected["ts"]))
        if "analysis_runs" in tables:
            connection.execute("INSERT INTO analysis_runs(day,total_count,payload,created_at) VALUES (?,1,?,?)",
                               ("2026-09-24", '{"synthetic":true}', expected["ts"]))
        if "analysis_jobs" in tables:
            connection.execute("INSERT INTO analysis_jobs(status,input_hash,day,created_at,updated_at) VALUES (?,?,?,?,?)",
                               ("completed", "synthetic-hash", "2026-09-24", expected["ts"], expected["ts"]))
        connection.commit()
        return expected, stored_rows(connection)
    finally:
        connection.close()


@pytest.mark.parametrize("commit,blob", BASELINES)
def test_known_historical_schema_preserves_data_across_upgrade_and_authenticated_reads(tmp_path, monkeypatch, commit, blob):
    # Git may check out SQL with CRLF on Windows. Hash the canonical LF source
    # used by the original blob; every other source byte remains significant.
    source_bytes = (FIXTURES / ("schema-" + commit[:7] + ".sql")).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha1(b"blob " + str(len(source_bytes)).encode("ascii") + b"\0" + source_bytes).hexdigest() == blob
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "synthetic-old.sqlite3")
    expected, old_data = seed_historical_database(db.DB_PATH, source_bytes.decode("utf-8"))

    db.init_db(SCHEMA)
    with db.get_conn() as connection:
        upgraded_data = stored_rows(connection)
        assert all(upgraded_data[name] == rows for name, rows in old_data.items())
        identity = tuple(connection.execute("SELECT * FROM items_revision").fetchone())
        assert identity[0] == 1 and len(identity[1]) == 32 and identity[2] == 0
        # Exercise the newly added update trigger without changing page content.
        connection.execute("UPDATE items SET title=title WHERE id=7")
    with db.get_conn() as connection:
        upgraded_data = stored_rows(connection)
        upgraded_schema = schema_objects(connection)
        revision = tuple(connection.execute("SELECT * FROM items_revision").fetchone())
        assert revision == (1, identity[1], 1)

    db.init_db(SCHEMA)
    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000),
                    raise_server_exceptions=False, headers={"Authorization": "Bearer " + "a" * 43}) as client:
        # Lifespan performs another initialization; both explicit repeat and
        # normal application startup must preserve the upgraded database.
        listing = client.get("/items")
        assert listing.status_code == 200 and listing.json()["total"] == 1
        record = listing.json()["items"][0]
        assert record["id"] == 7
        assert all(record[key] == value for key, value in expected.items())
        backup = client.get("/data/backup")
        assert backup.status_code == 200 and backup.json()["items"] == [expected]

    with db.get_conn() as connection:
        assert stored_rows(connection) == upgraded_data
        assert schema_objects(connection) == upgraded_schema
        assert tuple(connection.execute("SELECT * FROM items_revision").fetchone()) == revision
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
