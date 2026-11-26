from __future__ import annotations

import hashlib
import re
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlsplit, urlunparse, urlunsplit


_WS_RE = re.compile(r"\s+")
_PRIVATE_QUERY_PARAM_RE = re.compile(
    r"^(access_token|refresh_token|id_token|token|code|auth|authorization|"
    r"password|passwd|api_key|apikey|key|session|sessionid|sid)$",
    re.IGNORECASE,
)


def sanitize_ingest_url(url: str) -> str:
    """Remove browser-only URL state and credential-bearing components.

    The extension applies the same policy before sending a page. Applying it
    at the service boundary also covers direct API imports and keeps stored
    records from retaining userinfo, fragments or common token parameters.
    """
    parsed = urlsplit(url)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    netloc = hostname
    if port is not None:
        netloc += f":{port}"

    query = parsed.query
    pairs = parse_qsl(query, keep_blank_values=True)
    if any(_PRIVATE_QUERY_PARAM_RE.fullmatch(key) for key, _ in pairs):
        query = urlencode(
            [(key, value) for key, value in pairs
             if not _PRIVATE_QUERY_PARAM_RE.fullmatch(key)],
            doseq=True,
        )
    return urlunsplit((parsed.scheme, netloc, parsed.path, query, ""))


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    # Normalize scheme/host, drop fragment, keep query for now
    netloc = parsed.netloc.lower()
    scheme = parsed.scheme.lower() if parsed.scheme else "http"
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    normalized = urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))
    return normalized


def normalize_text(title: str, text: Optional[str]) -> str:
    parts = [title or "", text or ""]
    joined = " ".join(parts).strip().lower()
    return _WS_RE.sub(" ", joined)


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
