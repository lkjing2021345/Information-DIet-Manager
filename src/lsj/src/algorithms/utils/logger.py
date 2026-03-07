from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def _normalize_level(level: Union[int, str]) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return getattr(logging, level.upper(), logging.INFO)
    return logging.INFO


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: Union[int, str] = logging.INFO,
    fmt: str = DEFAULT_LOG_FORMAT,
    console: bool = True,
) -> logging.Logger:
    """Create or return a configured logger without duplicating handlers."""
    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(_normalize_level(level))

    if getattr(logger_obj, "_custom_setup_logger_configured", False):
        return logger_obj

    formatter = logging.Formatter(fmt)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger_obj.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger_obj.addHandler(console_handler)

    logger_obj.propagate = False
    logger_obj._custom_setup_logger_configured = True
    return logger_obj
