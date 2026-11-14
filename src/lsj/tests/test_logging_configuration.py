"""Real logging utility with synthetic names and temporary paths; no models."""
import importlib.util
import logging
from pathlib import Path
import uuid

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src" / "algorithms" / "utils" / "logger.py"
PRIVATE_MARKER = "SYNTHETIC_PRIVATE_LOG_PATH_OR_ERROR"


@pytest.fixture
def logging_utility(monkeypatch, tmp_path):
    root = tmp_path / "用户 应用数据"
    monkeypatch.setenv("LOCALAPPDATA", str(root))
    spec = importlib.util.spec_from_file_location("synthetic_logging_utility", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = []

    def configure(*args, name=None, **kwargs):
        name = name or "idm.synthetic.logging." + uuid.uuid4().hex
        names.append(name)
        return module.setup_logger(name, *args, **kwargs)

    yield module, configure, root
    for name in set(names):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        logging.Logger.manager.loggerDict.pop(name, None)


def file_handler(logger):
    handlers = [handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)]
    assert len(handlers) == 1
    return handlers[0]


def test_relative_legacy_path_is_independent_of_working_directory(logging_utility, monkeypatch, tmp_path):
    _module, configure, root = logging_utility
    expected = root / "InformationDietManager" / "logs" / "sentiment.log"
    for index in range(2):
        cwd = tmp_path / f"launch-{index}" / "parent" / "project"
        cwd.mkdir(parents=True)
        monkeypatch.chdir(cwd)
        logger = configure("../../logs/sentiment.log", console=False, fmt="%(message)s")
        logger.info("Synthetic message %s", index)
        handler = file_handler(logger)
        handler.flush()
        assert Path(handler.baseFilename) == expected
        assert not (cwd / "../../logs").resolve().exists()
    assert expected.read_text(encoding="utf-8").splitlines() == ["Synthetic message 0", "Synthetic message 1"]


def test_file_logging_has_a_bounded_rotation_policy(logging_utility):
    _module, configure, _root = logging_utility
    logger = configure("bounded.log", console=False)
    handler = file_handler(logger)
    assert handler.maxBytes == 5 * 1024 * 1024
    assert handler.backupCount == 3


def test_unicode_path_and_utf8_log_content_are_preserved(logging_utility):
    _module, configure, root = logging_utility
    logger = configure("nested/../../logs/合成日志.log", console=False, fmt="%(message)s")
    logger.info("合成日志 😀")
    handler = file_handler(logger)
    handler.flush()
    expected = root / "InformationDietManager" / "logs" / "合成日志.log"
    assert Path(handler.baseFilename) == expected
    assert expected.read_text(encoding="utf-8") == "合成日志 😀\n"


def test_absolute_log_path_is_honored(logging_utility, tmp_path):
    _module, configure, root = logging_utility
    expected = tmp_path / "指定目录" / "absolute.log"
    logger = configure(str(expected), console=False, fmt="%(message)s")
    logger.info("Synthetic absolute path")
    handler = file_handler(logger)
    handler.flush()
    assert Path(handler.baseFilename) == expected
    assert expected.read_text(encoding="utf-8") == "Synthetic absolute path\n"
    assert not root.exists()


@pytest.mark.parametrize("value", [None, "", " ", "\t\r\n"])
def test_missing_or_empty_localappdata_uses_current_user_data_directory(logging_utility, monkeypatch, tmp_path, value):
    module, configure, _root = logging_utility
    if value is None:
        monkeypatch.delenv("LOCALAPPDATA")
    else:
        monkeypatch.setenv("LOCALAPPDATA", value)
    synthetic_home = tmp_path / "合成 home"
    monkeypatch.setattr(module.Path, "home", classmethod(lambda _cls: synthetic_home))
    logger = configure("../../logs/default.log", console=False)
    expected = synthetic_home / ".local" / "share" / "InformationDietManager" / "logs" / "default.log"
    assert Path(file_handler(logger).baseFilename) == expected
    assert expected.is_file()


@pytest.mark.parametrize("phase", ["directory", "file"])
@pytest.mark.parametrize("console", [True, False])
def test_file_permission_failures_do_not_escape_or_expose_details(logging_utility, monkeypatch, capsys, phase, console):
    module, configure, _root = logging_utility

    def denied(*_args, **_kwargs):
        raise PermissionError(PRIVATE_MARKER)

    if phase == "directory":
        monkeypatch.setattr(module.Path, "mkdir", denied)
    else:
        monkeypatch.setattr(module, "RotatingFileHandler", denied)
    logger = configure("../../logs/" + PRIVATE_MARKER + ".log", console=console, fmt="%(levelname)s|%(message)s")
    captured = capsys.readouterr()
    assert not logger.propagate and len(logger.handlers) == 1
    assert PRIVATE_MARKER not in captured.out + captured.err
    if console:
        assert captured.err == "WARNING|" + module.FILE_LOG_UNAVAILABLE + "\n"
        assert type(logger.handlers[0]) is logging.StreamHandler
        logger.info("Synthetic fallback remains usable")
        assert capsys.readouterr().err == "INFO|Synthetic fallback remains usable\n"
    else:
        assert isinstance(logger.handlers[0], logging.NullHandler)
        logger.error("Synthetic silence check")
        assert captured.out == captured.err == capsys.readouterr().err == ""


def test_existing_non_directory_log_parent_safely_falls_back(logging_utility, capsys):
    module, configure, root = logging_utility
    blocked = root / "InformationDietManager"
    root.mkdir(parents=True)
    blocked.write_text("Synthetic non-directory", encoding="utf-8")
    logger = configure("../../logs/blocked.log", fmt="%(message)s")
    assert type(logger.handlers[0]) is logging.StreamHandler
    assert capsys.readouterr().err == module.FILE_LOG_UNAVAILABLE + "\n"
    assert blocked.read_text(encoding="utf-8") == "Synthetic non-directory"


@pytest.mark.parametrize("file_fails", [False, True])
def test_repeated_configuration_is_idempotent_and_keeps_existing_handlers(logging_utility, monkeypatch, tmp_path, capsys, file_fails):
    module, configure, root = logging_utility
    if file_fails:
        def denied(*_args, **_kwargs):
            raise OSError(PRIVATE_MARKER)
        monkeypatch.setattr(module.Path, "mkdir", denied)
    logger = configure("../../logs/first.log", level="DEBUG", fmt="%(levelname)s|%(message)s")
    original_handlers = list(logger.handlers)
    capsys.readouterr()
    repeated = configure(str(tmp_path / "must-not-create.log"), name=logger.name, level="ERROR", console=False, fmt="CHANGED:%(message)s")
    assert repeated is logger and logger.handlers == original_handlers
    assert logger.level == logging.ERROR
    assert not (tmp_path / "must-not-create.log").exists()
    logger.warning("Suppressed warning")
    logger.error("Synthetic repeated configuration")
    assert capsys.readouterr().err == "ERROR|Synthetic repeated configuration\n"
    assert len(logger.handlers) == (1 if file_fails else 2)
    if not file_fails:
        file_handler(logger).flush()
        assert (root / "InformationDietManager" / "logs" / "first.log").read_text(encoding="utf-8").splitlines() == [
            "ERROR|Synthetic repeated configuration"]


def test_console_only_and_disabled_logging_do_not_create_directories(logging_utility, capsys):
    _module, configure, root = logging_utility
    logger = configure(level="WARNING", fmt="%(levelname)s:%(message)s")
    logger.info("Suppressed info")
    logger.warning("Synthetic console")
    assert capsys.readouterr().err == "WARNING:Synthetic console\n"
    silent = configure(console=False)
    silent.error("Synthetic silence")
    assert isinstance(silent.handlers[0], logging.NullHandler)
    assert capsys.readouterr().err == ""
    assert not root.exists()
