from __future__ import annotations

import json
from typing import Annotated, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, StringConstraints, field_validator


MAX_INGEST_TS = 253402300799999  # 9999-12-31T23:59:59.999Z
MAX_TAGS = 10
MAX_TAG_LENGTH = 40
MAX_META_BYTES = 8192
MAX_META_DEPTH = 8

Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]
Tag = Annotated[str, StringConstraints(max_length=MAX_TAG_LENGTH)]


def _check_json_value(value: object, depth: int = 1) -> None:
    """Bound container nesting before serialization, including cyclic Python inputs."""
    if isinstance(value, (dict, list)):
        if depth > MAX_META_DEPTH:
            raise ValueError(f"meta must have at most {MAX_META_DEPTH} container levels")
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise ValueError("meta object keys must be strings")
            children = value.values()
        else:
            children = value
        for child in children:
            _check_json_value(child, depth + 1)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError("meta values must be JSON serializable")


class IngestItem(BaseModel):
    url: HttpUrl = Field(..., max_length=2048)
    title: Title
    text: Optional[str] = Field(default=None, max_length=2000)
    ts: int = Field(
        ..., strict=True, ge=0, le=MAX_INGEST_TS,
        description="Unix epoch milliseconds (access time), from 1970 through year 9999.",
    )
    source: Literal["plugin", "import"]

    # Optional extension fields (keep stable names for future use)
    lang: Optional[str] = Field(default=None, max_length=16)
    channel: Optional[str] = Field(default=None, max_length=32)
    author: Optional[str] = Field(default=None, max_length=120)
    tags: Optional[List[Tag]] = Field(default=None, max_length=MAX_TAGS)
    meta: Optional[Dict[str, object]] = None

    @field_validator("meta")
    @classmethod
    def validate_meta(cls, value: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
        if value is None:
            return value
        _check_json_value(value)
        try:
            encoded = json.dumps(
                value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("meta must contain finite, UTF-8 JSON values") from exc
        if len(encoded) > MAX_META_BYTES:
            raise ValueError(f"meta must be at most {MAX_META_BYTES} UTF-8 bytes as compact JSON")
        return value


class IngestRequest(BaseModel):
    items: List[IngestItem]


class IngestAck(BaseModel):
    inserted: int
    duplicates: int
    failed: int = 0

