"""Security and recovery tests use synthetic keys and temporary databases only."""
import hashlib
import base64
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from src.hyh import app as api, db, data_management
from src.hyh.data_management import canonical_items
from src.hyh.security import LocalSecurity, credential_path, process_ownership, write_new_credentials

ADMIN = {"Authorization": "Bearer " + "a" * 43}
COLLECTOR = {"Authorization": "Bearer " + "c" * 43}
ORIGIN = "http://127.0.0.1:5173"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "private" / "test.sqlite3")
    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000), raise_server_exceptions=False) as client:
        yield client


def item(n=1):
    return {"url": f"https://example.invalid/{n}", "title": f"Synthetic {n} 中文 🚀", "text": "Synthetic page content", "ts": 1790208000000 + n, "source": "plugin"}


def seed(client, count=2):
    for n in range(count):
        assert client.post("/collect", headers=COLLECTOR, json=item(n)).status_code == 200


def snapshot():
    with db.get_conn() as conn:
        return {table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table}")]
                for table in ["items", "embeddings", "analysis_jobs", "analysis_runs", "stats_daily"]}


@pytest.mark.parametrize("method,path", [("GET", "/items"), ("GET", "/data/backup"), ("DELETE", "/data"),
    ("POST", "/data/restore"), ("POST", "/collect"), ("POST", "/import"), ("GET", "/dashboard/visualization"),
    ("GET", "/export/lsj"), ("POST", "/analyze/run"), ("GET", "/analyze/history")])
def test_data_endpoints_require_a_key(client, method, path):
    response = client.request(method, path)
    assert response.status_code == 401
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("method,path", [("GET", "/items"), ("GET", "/data/backup"), ("DELETE", "/data"),
    ("DELETE", "/items/1"), ("POST", "/data/restore"), ("POST", "/import"), ("GET", "/dashboard/visualization")])
def test_collector_cannot_read_delete_restore_or_run_analysis(client, method, path):
    assert client.request(method, path, headers=COLLECTOR).status_code == 403


def test_role_and_health_contracts(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/session", headers=COLLECTOR).json() == {"role": "collector"}
    assert client.get("/session", headers=ADMIN).json() == {"role": "admin"}
    assert client.get("/items?token=" + "a" * 43).status_code == 401
    client.cookies.set("token", "a" * 43)
    assert client.get("/items").status_code == 401


@pytest.mark.parametrize("host", ["evil.example", "127.0.0.1.evil.example", "localhost.", "localhost@evil.example", "127.0.0.1:99999"])
def test_dns_rebinding_host_rejected_even_with_valid_key(client, host):
    assert client.get("/items", headers={**ADMIN, "Host": host}).status_code == 403


@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://127.0.0.1:9999", "http://localhost:5173.evil.example"])
def test_foreign_origin_rejected_even_with_valid_key(client, origin):
    response = client.get("/items", headers={**ADMIN, "Origin": origin})
    assert response.status_code == 403
    assert "access-control-allow-origin" not in response.headers


def test_cors_preflight_and_auth_errors_are_readable_only_to_allowed_origin(client):
    for origin in [ORIGIN, "chrome-extension://" + "a" * 32]:
        response = client.options("/collect", headers={"Origin": origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type"})
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin
        assert "access-control-allow-credentials" not in response.headers
    denied = client.get("/items", headers={"Origin": ORIGIN})
    assert denied.status_code == 401 and denied.headers["access-control-allow-origin"] == ORIGIN


def test_duplicate_sensitive_headers_rejected(client):
    response = client.get("/items", headers=[("Authorization", ADMIN["Authorization"]), ("Authorization", COLLECTOR["Authorization"])])
    assert response.status_code == 400


def test_remote_peer_rejected(client):
    remote = TestClient(api.app, base_url="http://127.0.0.1", client=("192.0.2.1", 40000))
    assert remote.get("/items", headers=ADMIN).status_code == 403
    remote.close()


def test_collect_body_limit_applies_without_content_length(client):
    def chunks():
        yield b" " * 40000
        yield b" " * 40000
    response = client.post("/collect", headers=COLLECTOR, content=chunks())
    assert response.status_code == 413
    assert client.get("/items", headers=ADMIN).json()["total"] == 0


def test_request_capacity_and_busy_gate_return_retryable_503_with_cors(client):
    slots = api.app.state.request_slots
    for _ in range(4): client.portal.call(slots.acquire)
    try:
        response = client.get("/items", headers={**ADMIN, "Origin": ORIGIN})
        assert response.status_code == 503 and response.headers["Retry-After"] == "5"
        assert response.headers["Access-Control-Allow-Origin"] == ORIGIN
        assert client.get("/health").status_code == 200
    finally:
        for _ in range(4): client.portal.call(slots.release)
    gate = api.app.state.operation_lock
    client.portal.call(gate.acquire)
    try:
        assert client.get("/items", headers=ADMIN).status_code == 503
    finally:
        client.portal.call(gate.release)
    assert client.get("/items", headers=ADMIN).status_code == 200


def test_roundtrip_delete_restore_and_derived_cleanup(client):
    seed(client)
    assert client.post("/analyze/run", headers=ADMIN).status_code == 200
    before = snapshot()
    response = client.get("/data/backup", headers=ADMIN)
    assert response.status_code == 200
    backup = response.json()
    assert backup["sha256"] == hashlib.sha256(canonical_items(backup["items"])).hexdigest()
    assert "a" * 43 not in response.text and "c" * 43 not in response.text
    assert client.delete("/data", headers=ADMIN).status_code == 400
    assert snapshot() == before
    assert client.delete("/data", headers={**ADMIN, "X-IDM-Confirm": "delete-all"}).json()["deleted"] == 2
    assert all(not rows for rows in snapshot().values())
    restored = client.post("/data/restore", headers={**ADMIN, "X-IDM-Confirm": "replace-records"}, json=backup)
    assert restored.status_code == 200 and restored.json()["restored"] == 2
    after = snapshot()
    assert len(after["items"]) == 2 and len(after["embeddings"]) == 2
    assert not after["analysis_runs"] and not after["analysis_jobs"] and not after["stats_daily"]
    assert client.get("/data/backup", headers=ADMIN).json()["items"] == backup["items"]


def test_single_delete_removes_embedding_and_analysis_cache(client):
    seed(client)
    client.post("/analyze/run", headers=ADMIN)
    item_id = client.get("/items", headers=ADMIN).json()["items"][0]["id"]
    assert client.delete(f"/items/{item_id}", headers={**ADMIN, "X-IDM-Confirm": "delete-record"}).status_code == 200
    state = snapshot()
    assert len(state["items"]) == len(state["embeddings"]) == 1
    assert not state["analysis_runs"] and not state["stats_daily"]
    assert client.delete(f"/items/{item_id}", headers={**ADMIN, "X-IDM-Confirm": "delete-record"}).status_code == 404


@pytest.mark.parametrize("item_id", [-1, 0, 2**63])
def test_invalid_record_ids_cannot_overflow_sqlite(client, item_id):
    assert client.delete(f"/items/{item_id}", headers={**ADMIN, "X-IDM-Confirm": "delete-record"}).status_code == 404


def test_backup_capacity_and_legacy_errors_do_not_change_data(client, monkeypatch):
    seed(client)
    before = snapshot()
    monkeypatch.setattr(data_management, "MAX_BACKUP_RECORDS", 1)
    assert client.get("/data/backup", headers=ADMIN).status_code == 413
    monkeypatch.setattr(data_management, "MAX_BACKUP_RECORDS", 10000)
    monkeypatch.setattr(data_management, "MAX_BACKUP_BYTES", 100)
    assert client.get("/data/backup", headers=ADMIN).status_code == 413
    assert snapshot() == before
    monkeypatch.setattr(data_management, "MAX_BACKUP_BYTES", 20 * 1024 * 1024)
    with db.get_conn() as conn:
        conn.execute("UPDATE items SET meta = 'invalid legacy JSON'")
    legacy = snapshot()
    assert client.get("/data/backup", headers=ADMIN).status_code == 409
    assert snapshot() == legacy


@pytest.mark.parametrize("damage", ["checksum", "version", "boolean_version", "invalid_item", "duplicate", "not_object", "extra"])
def test_invalid_restore_preserves_every_table(client, damage):
    seed(client); client.post("/analyze/run", headers=ADMIN)
    backup = client.get("/data/backup", headers=ADMIN).json()
    before = snapshot()
    if damage == "checksum": backup["sha256"] = "0" * 64
    if damage == "version": backup["version"] = 2
    if damage == "boolean_version": backup["version"] = True
    if damage == "invalid_item": backup["items"][0]["ts"] = -1
    if damage == "duplicate":
        backup["items"].append(backup["items"][0])
        backup["sha256"] = hashlib.sha256(canonical_items(backup["items"])).hexdigest()
    if damage == "not_object": backup = []
    if damage == "extra": backup["unexpected"] = True
    response = client.post("/data/restore", headers={**ADMIN, "X-IDM-Confirm": "replace-records"}, json=backup)
    assert response.status_code == 422
    assert snapshot() == before


def test_partial_restore_storage_failure_rolls_back_deletion_and_new_embeddings(client, monkeypatch):
    seed(client); client.post("/analyze/run", headers=ADMIN)
    backup = client.get("/data/backup", headers=ADMIN).json()
    before = snapshot(); original = api._upsert_embedding; calls = 0
    def fail_second(*args):
        nonlocal calls
        calls += 1
        if calls == 2: raise OSError("synthetic full disk")
        return original(*args)
    monkeypatch.setattr(api, "_upsert_embedding", fail_second)
    response = client.post("/data/restore", headers={**ADMIN, "X-IDM-Confirm": "replace-records", "Origin": ORIGIN}, json=backup)
    assert response.status_code == 500
    assert response.headers["Access-Control-Allow-Origin"] == ORIGIN and response.headers["Cache-Control"] == "no-store"
    assert "synthetic full disk" not in response.text
    assert snapshot() == before


def test_delete_completes_during_analysis_and_prevents_its_late_output(client, monkeypatch):
    seed(client, 5)
    entered, release = threading.Event(), threading.Event()
    def slow_analysis(rows):
        entered.set(); assert release.wait(5)
        return {"ok": False, "warning": "synthetic no model", "input_count": len(rows)}
    monkeypatch.setattr(api, "_execute_lsj_pipeline", slow_analysis)
    with ThreadPoolExecutor(max_workers=2) as pool:
        analysis = pool.submit(client.get, "/dashboard/visualization?from_ts=1790208000000&to_ts=1790294400000", headers=ADMIN)
        assert entered.wait(5)
        delete = pool.submit(client.delete, "/data", headers={**ADMIN, "X-IDM-Confirm": "delete-all"})
        try:
            assert delete.result(timeout=3).status_code == 200
        finally:
            release.set()
        assert analysis.result().status_code == 409
    assert all(not rows for rows in snapshot().values())


def test_busy_dashboard_analysis_does_not_block_records_and_keeps_protected_error_headers(client):
    gate = api.app.state.analysis_lock
    client.portal.call(gate.acquire)
    try:
        response = client.get("/dashboard/visualization", headers={**ADMIN, "Origin": ORIGIN})
        assert response.status_code == 503 and response.headers["Retry-After"] == "5"
        assert response.headers["Access-Control-Allow-Origin"] == ORIGIN
        assert response.headers["Cache-Control"] == "no-store"
        assert client.get("/items", headers=ADMIN).status_code == 200
        assert client.post("/analyze/run_full", headers=ADMIN).status_code == 503
        assert not api.app.state.operation_lock.locked()
    finally:
        client.portal.call(gate.release)


def test_legacy_scoring_does_not_acquire_or_release_another_data_operation_gate(client):
    gate = api.app.state.operation_lock
    client.portal.call(gate.acquire)
    try:
        response = client.post("/analyze/run_full", headers=ADMIN)
        assert response.status_code == 200
        assert response.json()["result"]["analysis_status"] == "empty"
        assert gate.locked()
        assert not api.app.state.analysis_lock.locked()
    finally:
        client.portal.call(gate.release)


def test_second_process_cannot_open_same_database(client):
    script = "from pathlib import Path; from src.hyh.security import process_ownership; import sys\nwith process_ownership(Path(sys.argv[1])): pass"
    result = subprocess.run([sys.executable, "-c", script, str(db.DB_PATH)], capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "Another IDM process" in result.stderr


def test_generated_keys_persist_and_rotate_without_logging_their_values(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    monkeypatch.delenv("IDM_ADMIN_TOKEN"); monkeypatch.delenv("IDM_COLLECTOR_TOKEN")
    path = tmp_path / "fresh.sqlite3"
    with process_ownership(path):
        first = LocalSecurity.load(path); second = LocalSecurity.load(path)
        assert first == second and first.admin_token != first.collector_token
        write_new_credentials(credential_path(path), replace=True)
        rotated = LocalSecurity.load(path)
        assert rotated.admin_token != first.admin_token and rotated.collector_token != first.collector_token
    assert str(credential_path(path)) in caplog.text
    assert first.admin_token not in caplog.text and first.collector_token not in caplog.text
    assert first.admin_token not in repr(first)
    if os.name == "nt":
        # Inspect the applied ACL, not just the permissions helper's exit code. Never read file contents.
        script = """
$ErrorActionPreference = 'Stop'
$acl = [System.IO.File]::GetAccessControl($env:IDM_CREDENTIAL_FILE)
$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$rules = $acl.GetAccessRules($true, $true, [System.Security.Principal.SecurityIdentifier])
if (!$acl.AreAccessRulesProtected -or $rules.Count -ne 2) { exit 1 }
foreach ($rule in $rules) {
    if ($rule.IsInherited -or $rule.AccessControlType -ne 'Allow' -or $rule.IdentityReference.Value -notin @($sid, 'S-1-5-18')) { exit 2 }
}
"""
        powershell = os.path.join(os.environ["SystemRoot"], "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
        result = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-EncodedCommand", base64.b64encode(script.encode("utf-16-le")).decode("ascii")],
                                env={**os.environ, "IDM_CREDENTIAL_FILE": str(credential_path(path))}, capture_output=True,
                                creationflags=subprocess.CREATE_NO_WINDOW, timeout=15)
        assert result.returncode == 0
    else:
        assert credential_path(path).stat().st_mode & 0o777 == 0o600


def test_corrupt_credential_file_is_not_silently_replaced(tmp_path, monkeypatch):
    monkeypatch.delenv("IDM_ADMIN_TOKEN"); monkeypatch.delenv("IDM_COLLECTOR_TOKEN")
    path = tmp_path / "test.sqlite3"
    credential_path(path).write_text("invalid credential JSON", encoding="utf-8")
    with process_ownership(path), pytest.raises(RuntimeError, match="Invalid local credential file"):
        LocalSecurity.load(path)
    assert credential_path(path).read_text(encoding="utf-8") == "invalid credential JSON"


def test_launcher_and_offline_rotation_revoke_old_keys_and_preserve_records(tmp_path):
    root = Path(__file__).resolve().parents[3]
    path = tmp_path / "new-private-directory" / "test.sqlite3"
    env = {**os.environ, "IDM_DB_PATH": str(path), "PYTHONUTF8": "1"}
    for name in ("IDM_ADMIN_TOKEN", "IDM_COLLECTOR_TOKEN", "IDM_FRONTEND_ORIGINS"):
        env.pop(name, None)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0)); port = probe.getsockname()[1]

    @contextmanager
    def running_service():
        process = subprocess.Popen([sys.executable, "scripts/run_backend.py", "--port", str(port)], cwd=root,
                                   env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=2, trust_env=False) as browser:
                deadline = time.monotonic() + 30
                while True:
                    assert process.poll() is None, "Local launcher exited before readiness"
                    try:
                        if browser.get("/health").status_code == 200: break
                    except httpx.HTTPError:
                        pass
                    assert time.monotonic() < deadline, "Local launcher did not become ready"
                    time.sleep(0.1)
                yield browser
        finally:
            process.terminate()
            process.communicate(timeout=10)

    command = [sys.executable, "scripts/rotate_local_keys.py"]
    with running_service() as browser:
        old_file = credential_path(path).read_text(encoding="utf-8")
        old = json.loads(old_file)
        headers = {"Authorization": "Bearer " + old["admin_token"]}
        assert browser.post("/collect", headers=headers, json=item()).status_code == 200
        blocked = subprocess.run(command, cwd=root, env=env, capture_output=True, timeout=20)
        assert blocked.returncode != 0
        assert credential_path(path).read_text(encoding="utf-8") == old_file
    rotated = subprocess.run(command, cwd=root, env=env, capture_output=True, timeout=20)
    assert rotated.returncode == 0
    current = json.loads(credential_path(path).read_text(encoding="utf-8"))
    for name in ("admin_token", "collector_token"):
        assert current[name] != old[name]
        assert current[name].encode() not in rotated.stdout + rotated.stderr
    with running_service() as browser:
        assert browser.get("/items", headers=headers).status_code == 401
        assert browser.post("/collect", headers={"Authorization": "Bearer " + old["collector_token"]}, json=item(2)).status_code == 401
        assert browser.get("/items", headers={"Authorization": "Bearer " + current["admin_token"]}).json()["total"] == 1


@pytest.mark.parametrize("name,value", [("IDM_ADMIN_TOKEN", ""), ("IDM_COLLECTOR_TOKEN", "short"), ("IDM_FRONTEND_ORIGINS", "*"), ("IDM_FRONTEND_ORIGINS", "https://remote.example")])
def test_invalid_security_configuration_fails_closed(tmp_path, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError): LocalSecurity.load(tmp_path / "test.sqlite3")
