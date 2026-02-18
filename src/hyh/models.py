from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


class IngestItem(BaseModel):
    url: HttpUrl = Field(..., max_length=2048)
    title: str = Field(..., min_length=1, max_length=300)
    text: Optional[str] = Field(default=None, max_length=2000)
    ts: int = Field(..., description="Unix epoch milliseconds (access time).")
    source: Literal["plugin", "import"]

    # Optional extension fields (keep stable names for future use)
    lang: Optional[str] = Field(default=None, max_length=16)
    channel: Optional[str] = Field(default=None, max_length=32)
    author: Optional[str] = Field(default=None, max_length=120)
    tags: Optional[List[str]] = None
    meta: Optional[Dict[str, object]] = None


class IngestRequest(BaseModel):
    items: List[IngestItem]


class IngestAck(BaseModel):
    inserted: int
    duplicates: int
    failed: int = 0

