"""Bounded, finite JSON for stored analysis caches and historical reports."""
from __future__ import annotations

import json
import math
from typing import Any

MAX_ANALYSIS_JSON_DEPTH = 64


class InvalidStoredAnalysis(ValueError):
    """Stored content cannot safely represent a JSON analysis result."""


def _validate_text(value: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InvalidStoredAnalysis("Invalid stored analysis") from exc


def validate_analysis_value(value: Any) -> None:
    """Reject non-finite numbers and more than 64 nested JSON containers.

    The root object/array counts as one container. Iterative traversal avoids
    relying on Python's recursion limit or the HTTP serializer's depth limit.
    Ordinary JSON null is a valid missing measurement and is preserved.
    """
    pending = [(value, 0)]
    while pending:
        item, parent_depth = pending.pop()
        if isinstance(item, (dict, list)):
            depth = parent_depth + 1
            if depth > MAX_ANALYSIS_JSON_DEPTH:
                raise InvalidStoredAnalysis("Invalid stored analysis")
            if isinstance(item, dict):
                for key in item:
                    if not isinstance(key, str):
                        raise InvalidStoredAnalysis("Invalid stored analysis")
                    _validate_text(key)
                pending.extend((child, depth) for child in item.values())
            else:
                pending.extend((child, depth) for child in item)
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise InvalidStoredAnalysis("Invalid stored analysis")
        elif isinstance(item, str):
            _validate_text(item)
        elif item is not None and not isinstance(item, (int, bool)):
            raise InvalidStoredAnalysis("Invalid stored analysis")


def _reject_constant(_value: str) -> None:
    raise InvalidStoredAnalysis("Invalid stored analysis")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise InvalidStoredAnalysis("Invalid stored analysis")
    return number


def load_analysis_json(value: Any) -> Any:
    """Read a stored JSON field; SQL NULL and JSON null remain None."""
    if value is None:
        return None
    try:
        parsed = json.loads(value, parse_constant=_reject_constant, parse_float=_finite_float)
        validate_analysis_value(parsed)
        return parsed
    except (TypeError, ValueError, RecursionError) as exc:
        raise InvalidStoredAnalysis("Invalid stored analysis") from exc
