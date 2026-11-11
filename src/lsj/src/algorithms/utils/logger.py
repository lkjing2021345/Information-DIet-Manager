from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from typing import Optional, Union

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
FILE_LOG_UNAVAILABLE = "File logging is unavailable; using console logging."
FILE_LOG_MAX_BYTES = 5 * 1024 * 1024
FILE_LOG_BACKUP_COUNT = 3


def _resolve_log_path(log_file: str) -> Path:
    path = Path(log_file).expanduser()
    if path.is_absolute():
        return path
    # Existing callers use ../../logs/<name>. Only their filename is relevant:
    # the working directory must never select a location outside the app data.
    configured = os.environ.get("LOCALAPPDATA")
    base = Path(configured) if configured and configured.strip() else Path.home() / ".local" / "share"
    return base.expanduser() / "InformationDietManager" / "logs" / path.name


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
    """Configure once; relative filenames use per-user application data.

    Explicit absolute paths remain supported. File-system failures must not
    prevent optional analysis modules from importing; console=False stays quiet.
    """
    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(_normalize_level(level))

    if getattr(logger_obj, "_custom_setup_logger_configured", False):
        return logger_obj

    formatter = logging.Formatter(fmt)

    file_unavailable = False
    if log_file:
        try:
            log_path = _resolve_log_path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=FILE_LOG_MAX_BYTES,
                backupCount=FILE_LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            logger_obj.addHandler(file_handler)
        except OSError:
            file_unavailable = True

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger_obj.addHandler(console_handler)
    elif not logger_obj.handlers:
        # Avoid logging.lastResort emitting warnings despite console=False.
        logger_obj.addHandler(logging.NullHandler())

    logger_obj.propagate = False
    logger_obj._custom_setup_logger_configured = True
    if file_unavailable and console:
        logger_obj.warning(FILE_LOG_UNAVAILABLE)
    return logger_obj
