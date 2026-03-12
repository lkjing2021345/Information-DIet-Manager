from __future__ import annotations

import hashlib
import re
from typing import Optional
from urllib.parse import urlparse, urlunparse


_WS_RE = re.compile(r"\s+")


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
