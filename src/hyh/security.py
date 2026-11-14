"""Local-process ownership, capability keys and HTTP request boundaries."""
from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import logging
import os
import re
import secrets
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from .owned_work import WorkOwnerClosed

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{43,128}\Z")
EXTENSION_ORIGIN = r"chrome-extension://[a-p]{32}"
DEFAULT_ORIGINS = "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:4173,http://localhost:4173"
BODY_LIMITS = {"/collect": 64 * 1024, "/import": 10 * 1024 * 1024, "/data/restore": 20 * 1024 * 1024}
EXPORT_PATHS = frozenset({"/export/lsj", "/export/lsj/training", "/data/backup"})


def _database_lock_path(db_path: Path) -> Path:
    lock_path = db_path.with_suffix(".lock")
    # with_suffix keeps the parent; compare only the two final names. Normal
    # Win32 file opens discard trailing ASCII spaces/dots, while \\?\ paths
    # preserve them. Keep the actual paths and existing sidecar names unchanged.
    database_name, lock_name = db_path.name, lock_path.name
    if os.name == "nt":
        if not str(db_path).startswith("\\\\?\\"):
            database_name = database_name.rstrip(" .")
            lock_name = lock_name.rstrip(" .")
        database_name, lock_name = database_name.lower(), lock_name.lower()
    if database_name == lock_name:
        raise RuntimeError("Database path conflicts with its process lock file; choose a different database filename.")
    return lock_path


@contextmanager
def process_ownership(db_path: Path):
    """One API process per DB; also used by offline credential rotation."""
    lock_path = _database_lock_path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(lock_path, "a+b")
    try:
        if lock.seek(0, 2) == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("Another IDM process owns this database; stop it before continuing.") from None
        yield
    finally:
        lock.close()  # Closing releases the OS lock, including after a failed operation.


def credential_path(db_path: Path) -> Path:
    return db_path.with_suffix(".credentials.json")


def protect_credential_file(path: Path) -> None:
    if os.name != "nt":
        path.chmod(0o600)
        return
    # Windows ignores POSIX mode bits. Give this file an explicit private NTFS ACL.
    system32 = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32"
    # Replace the DACL in one operation, never reset it to broader inherited rights.
    script = """
$ErrorActionPreference = 'Stop'
$acl = [System.Security.AccessControl.FileSecurity]::new()
$owner = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$acl.SetOwner($owner)
$acl.SetAccessRuleProtection($true, $false)
foreach ($sid in @($owner, [System.Security.Principal.SecurityIdentifier]::new('S-1-5-18'))) {
    $rule = [System.Security.AccessControl.FileSystemAccessRule]::new($sid, 'FullControl', 'Allow')
    $acl.AddAccessRule($rule)
}
[System.IO.File]::SetAccessControl($env:IDM_CREDENTIAL_FILE, $acl)
"""
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    result = subprocess.run([str(system32 / "WindowsPowerShell/v1.0/powershell.exe"), "-NoLogo", "-NoProfile",
                             "-NonInteractive", "-EncodedCommand", encoded], capture_output=True,
                            env={**os.environ, "IDM_CREDENTIAL_FILE": str(path)},
                            creationflags=subprocess.CREATE_NO_WINDOW, timeout=15)
    if result.returncode:
        raise RuntimeError("Could not secure credential file permissions; use a private NTFS data directory.")


def write_new_credentials(path: Path, *, replace: bool = False) -> None:
    payload = {"version": 1, "admin_token": secrets.token_urlsafe(32), "collector_token": secrets.token_urlsafe(32)}
    temporary = path.with_name(path.name + ".new")
    created = False
    # The caller holds process_ownership. A leftover temporary file is never treated as valid credentials.
    try:
        with open(temporary, "x", encoding="utf-8", opener=lambda name, flags: os.open(name, flags, 0o600)) as stream:
            created = True
            protect_credential_file(temporary)
            json.dump(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() and not replace:
            raise RuntimeError("Credential file already exists.")
        os.replace(temporary, path)
    finally:
        if created and temporary.exists():
            temporary.unlink()


def valid_loopback_netloc(value: str) -> bool:
    try:
        parsed = urlsplit("http://" + value)
        return (parsed.hostname in {"localhost", "127.0.0.1", "::1"}
                and not parsed.username and not parsed.password and not parsed.path
                and not parsed.query and not parsed.fragment
                and (parsed.port is None or 1 <= parsed.port <= 65535)
                and value.lower() == parsed.netloc.lower())
    except (ValueError, TypeError):
        return False


@dataclass(frozen=True)
class LocalSecurity:
    admin_token: str = field(repr=False)
    collector_token: str = field(repr=False)
    origins: tuple[str, ...]

    @classmethod
    def load(cls, db_path: Path):
        origins = tuple(value.strip() for value in os.getenv("IDM_FRONTEND_ORIGINS", DEFAULT_ORIGINS).split(",") if value.strip())
        if not origins or any(not value.startswith(("http://", "https://"))
                              or not valid_loopback_netloc(value.split("://", 1)[1]) for value in origins):
            raise RuntimeError("IDM_FRONTEND_ORIGINS must contain exact loopback origins without paths.")
        admin, collector = os.getenv("IDM_ADMIN_TOKEN"), os.getenv("IDM_COLLECTOR_TOKEN")
        if admin is None and collector is None:
            path = credential_path(db_path)
            if not path.exists():
                write_new_credentials(path)
            try:
                protect_credential_file(path)
                if path.stat().st_size > 4096:
                    raise ValueError()
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("version") != 1:
                    raise ValueError()
                admin, collector = payload["admin_token"], payload["collector_token"]
            except (OSError, ValueError, KeyError, AttributeError):
                raise RuntimeError("Invalid local credential file; repair or rotate it while the service is stopped.") from None
            logging.getLogger("uvicorn.error").info("Local access keys: %s (keep this file private)", path)
        if not all(isinstance(value, str) and TOKEN_PATTERN.fullmatch(value) for value in (admin, collector)) or admin == collector:
            raise RuntimeError("Two distinct, valid local access keys are required; authentication cannot be disabled.")
        return cls(admin, collector, origins)

    def role(self, authorization: str) -> str | None:
        if not authorization.startswith("Bearer "):
            return None
        token = authorization[7:]
        if not TOKEN_PATTERN.fullmatch(token):
            return None
        if secrets.compare_digest(token, self.admin_token):
            return "admin"
        if secrets.compare_digest(token, self.collector_token):
            return "collector"
        return None


class LocalAccessMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        security = scope["app"].state.security
        # Capture this lifespan before any await; an old request must never
        # attach its worker to a later lifespan's owner after body/lock waits.
        work_owner = scope["app"].state.work_owner
        headers = {}
        for key, value in scope["headers"]:
            key = key.lower()
            if key in headers and key in {b"host", b"origin", b"authorization", b"content-length"}:
                return await JSONResponse({"detail": "Duplicate request header"}, status_code=400)(scope, receive, send)
            headers[key] = value.decode("latin-1")
        try:
            local_peer = ipaddress.ip_address(scope.get("client", [""])[0]).is_loopback
        except (ValueError, TypeError, IndexError):
            local_peer = False
        if not local_peer or not valid_loopback_netloc(headers.get(b"host", "")):
            return await JSONResponse({"detail": "Local requests only"}, status_code=403)(scope, receive, send)
        origin = headers.get(b"origin")
        if origin is not None and origin not in security.origins and not re.fullmatch(EXTENSION_ORIGIN, origin):
            return await JSONResponse({"detail": "Origin not allowed"}, status_code=403)(scope, receive, send)

        async def authorized(scope, receive, send):
            async def protected_send(message):
                if message["type"] == "http.response.start":
                    message["headers"] = list(message["headers"]) + [(b"cache-control", b"no-store"), (b"x-content-type-options", b"nosniff")]
                await send(message)

            if scope["path"] == "/health" and scope["method"] == "GET":
                return await self.app(scope, receive, protected_send)
            role = security.role(headers.get(b"authorization", ""))
            if role is None:
                return await JSONResponse({"detail": "Local access key required"}, status_code=401,
                                          headers={"WWW-Authenticate": "Bearer"})(scope, receive, protected_send)
            if role == "collector" and (scope["path"], scope["method"]) not in {("/collect", "POST"), ("/session", "GET")}:
                return await JSONResponse({"detail": "Admin access required"}, status_code=403)(scope, receive, protected_send)
            scope["idm_role"] = role
            if work_owner.closing:
                return await JSONResponse({"detail": "Service is shutting down; retry later"}, status_code=503,
                                          headers={"Retry-After": "5"})(scope, receive, protected_send)
            slots = scope["app"].state.request_slots
            try:
                await asyncio.wait_for(slots.acquire(), timeout=0.1)
            except TimeoutError:
                return await JSONResponse({"detail": "Request capacity reached; retry later"}, status_code=503,
                                          headers={"Retry-After": "5"})(scope, receive, protected_send)
            export_slot = None
            try:
                if scope["method"] == "GET" and scope["path"] in EXPORT_PATHS:
                    export_slot = scope["app"].state.export_slots
                    # Do not queue exports behind slow consumers or consume
                    # every general request slot waiting for an export slot.
                    if export_slot.locked():
                        export_slot = None
                        return await JSONResponse({"detail": "Export capacity reached; retry later"}, status_code=503,
                                                  headers={"Retry-After": "5"})(scope, receive, protected_send)
                    await export_slot.acquire()
                return await bounded(scope, receive, protected_send)
            finally:
                if export_slot is not None:
                    export_slot.release()
                slots.release()

        async def bounded(scope, receive, protected_send):
            limit = BODY_LIMITS.get(scope["path"], 64 * 1024)
            length = headers.get(b"content-length")
            if length is not None and (not length.isascii() or not length.isdecimal()):
                return await JSONResponse({"detail": "Invalid content length"}, status_code=400)(scope, receive, protected_send)
            if length is not None and (len(length) > 10 or int(length) > limit):
                return await JSONResponse({"detail": "Request body too large"}, status_code=413)(scope, receive, protected_send)
            body = bytearray()
            deadline = asyncio.get_running_loop().time() + 10
            try:
                while True:
                    message = await asyncio.wait_for(receive(), timeout=max(0, deadline - asyncio.get_running_loop().time()))
                    if message["type"] == "http.disconnect":
                        return
                    body.extend(message.get("body", b""))
                    if len(body) > limit:
                        return await JSONResponse({"detail": "Request body too large"}, status_code=413)(scope, receive, protected_send)
                    if not message.get("more_body", False):
                        break
            except TimeoutError:
                return await JSONResponse({"detail": "Request body timed out"}, status_code=408)(scope, receive, protected_send)
            sent = False

            async def buffered_receive():
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            # Both analysis routes check transactional versions before and
            # after inference. Keep them single-flight without blocking collection
            # and recovery for the duration of optional model loading/inference.
            # Other data operations retain their shared ordering boundary.
            state = scope["app"].state
            if scope["path"] in {"/dashboard/visualization", "/analyze/run_full"}:
                gates = [state.analysis_lock]
            else:
                gates = [state.operation_lock]
            acquired = []
            response_started = False

            async def operation_send(message):
                nonlocal response_started
                if message["type"] == "http.response.start":
                    response_started = True
                    if (scope["method"] == "GET" and scope["path"] in EXPORT_PATHS
                            and scope.get("idm_export_ready") is True and message["status"] == 200):
                        # The complete bounded artifact owns no DB connection.
                        # Slow downloads retain only their export/request slots.
                        for gate in reversed(acquired):
                            gate.release()
                        acquired.clear()
                await protected_send(message)

            try:
                try:
                    for gate in gates:
                        await asyncio.wait_for(gate.acquire(), timeout=2)
                        acquired.append(gate)
                except TimeoutError:
                    return await JSONResponse({"detail": "Service busy; retry later"}, status_code=503,
                                              headers={"Retry-After": "5"})(scope, receive, protected_send)
                try:
                    if work_owner.closing:
                        raise WorkOwnerClosed()
                    with work_owner.bind():
                        await self.app(scope, buffered_receive, operation_send)
                except WorkOwnerClosed:
                    if response_started:
                        raise
                    await JSONResponse({"detail": "Service is shutting down; retry later"}, status_code=503,
                                       headers={"Retry-After": "5"})(scope, buffered_receive, protected_send)
                except Exception as error:
                    if response_started:
                        raise
                    # Keep failures inside the CORS/no-store boundary without reflecting private data.
                    logging.getLogger("uvicorn.error").error("Local data operation failed (%s)", type(error).__name__)
                    await JSONResponse({"detail": "Local data operation failed"}, status_code=500)(scope, buffered_receive, protected_send)
            finally:
                for gate in reversed(acquired):
                    gate.release()

        cors = CORSMiddleware(authorized, allow_origins=list(security.origins), allow_origin_regex=EXTENSION_ORIGIN,
                              allow_methods=["GET", "POST", "DELETE"], allow_headers=["Authorization", "Content-Type", "X-IDM-Confirm"],
                              expose_headers=["Content-Disposition"], allow_credentials=False, max_age=600)
        await cors(scope, receive, send)
