"""Actual CLI entrypoints: safe failure channels with isolated dependency faults.

CSV parsing is real. Exceptions and the evaluator are controlled substitutes;
no optional model is loaded or downloaded by these tests.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT / "src" / "lsj" / "src" / "main.py"
ENTRYPOINTS = ("wrapper-root", "wrapper-external", "legacy-script", "legacy-module", "alias-module")
PRIVATE = "SYNTHETIC_PRIVATE_TITLE"
DETAIL = "SYNTHETIC_EXCEPTION_DETAIL"
KEY = "k" * 43
HARNESS = r'''
import importlib.abc
import json
from pathlib import Path
import runpy
import sys
import types
root, entry, fault, source = sys.argv[1:]
sys.path.insert(0, root)
import pandas as pd
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'cntext', 'jieba', 'huggingface_hub',
                                     'classifier', 'sentiment', 'similarity'}:
            raise AssertionError('No model import is permitted')
sys.meta_path.insert(0, Guard())
original = pd.read_csv
def read_csv(path, *args, **kwargs):
    frame = original(path, *args, **kwargs)
    if fault == 'interrupt':
        raise KeyboardInterrupt()
    if fault == 'exception':
        raise RuntimeError('SYNTHETIC_EXCEPTION_DETAIL ' + frame.iloc[0]['title'] + ' synthetic-key=' + 'k' * 43)
    return frame
pd.read_csv = read_csv
class Report:
    def to_dict(self):
        return {'synthetic_report': True}
class Evaluator:
    def evaluate(self, frame, *, detailed):
        return Report()
    def generate_summary(self, report):
        return 'Synthetic summary'
evaluator = types.ModuleType('evaluator')
evaluator.InformationQualityEvaluator = Evaluator
sys.modules['evaluator'] = evaluator
arguments = ['--mode', 'evaluate', '--input_file', source, '--input_format', 'csv']
if entry.endswith('-module'):
    name = 'src.lsj.src.main' if entry == 'legacy-module' else 'src.analysis_engine.main'
    sys.argv = [name, *arguments]
    runpy.run_module(name, run_name='__main__')
else:
    target = Path(root) / ('src/lsj/src/main.py' if entry == 'legacy-script' else 'scripts/run_analysis.py')
    sys.argv = [str(target), *arguments]
    runpy.run_path(str(target), run_name='__main__')
'''


def invoke(tmp_path, entry, fault):
    source = tmp_path / "synthetic.csv"
    content = ("title,url,category,sentiment,polarity,similarity\n"
               f"{PRIVATE},https://example.invalid,Tools,Neutral,0.0,0.25\n")
    source.write_text(content, encoding="utf-8")
    environment = {**os.environ, "PYTHONUTF8": "1", "LOCALAPPDATA": str(tmp_path / "local"),
                   "HF_HOME": str(tmp_path / "model-cache"), "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1"}
    process = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", "-c", HARNESS, str(ROOT), entry, fault, str(source)],
        cwd=tmp_path if entry in {"wrapper-external", "legacy-script"} else ROOT,
        env=environment, text=True, capture_output=True, encoding="utf-8", timeout=20,
    )
    assert source.read_text(encoding="utf-8") == content
    assert list(tmp_path.iterdir()) == [source]
    return process


@pytest.mark.parametrize("entry", ENTRYPOINTS)
@pytest.mark.parametrize("fault,code,error_type", [
    ("exception", 1, "RuntimeError"), ("interrupt", 130, "KeyboardInterrupt"),
])
def test_actual_cli_failure_is_private_and_never_exits_successfully(tmp_path, entry, fault, code, error_type):
    completed = invoke(tmp_path, entry, fault)
    assert completed.returncode == code
    assert completed.stdout == ""
    error = json.loads(completed.stderr)
    assert error == {
        "success": False,
        "error": ("Analysis was interrupted. Run the command again when ready." if fault == "interrupt" else
                  "Analysis failed. Check the input file, options, and model setup, then retry."),
        "error_type": error_type,
    }
    for private in (PRIVATE, DETAIL, KEY, "Traceback"):
        assert private not in completed.stdout + completed.stderr


@pytest.mark.parametrize("entry", ENTRYPOINTS)
def test_actual_cli_success_keeps_requested_json_output_without_diagnostics(tmp_path, entry):
    completed = invoke(tmp_path, entry, "success")
    assert completed.returncode == 0
    assert completed.stderr == ""
    result = json.loads(completed.stdout)
    assert result["mode"] == "evaluate"
    assert result["summary"] == "Synthetic summary"
    assert result["report"] == {"synthetic_report": True}
    assert result["records"] == [{"title": PRIVATE, "url": "https://example.invalid", "category": "Tools",
                                  "sentiment": "Neutral", "polarity": 0.0, "similarity": 0.25}]
    assert "success" not in result  # Successful analysis output has no new wrapper.


@pytest.mark.parametrize("entry", ENTRYPOINTS)
def test_actual_cli_invalid_json_option_does_not_echo_input(tmp_path, entry):
    source = tmp_path / "synthetic.json"
    private_option = "SYNTHETIC_PRIVATE_INVALID_BATCH_SIZE"
    content = json.dumps({"records": [{"title": "Synthetic title", "url": "https://example.invalid"}],
                          "options": {"batch_size": private_option}})
    source.write_text(content, encoding="utf-8")
    commands = {
        "wrapper-root": ["scripts/run_analysis.py"],
        "wrapper-external": [str(ROOT / "scripts" / "run_analysis.py")],
        "legacy-script": [str(MAIN)],
        "legacy-module": ["-m", "src.lsj.src.main"],
        "alias-module": ["-m", "src.analysis_engine.main"],
    }
    completed = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", *commands[entry], "--input_file", str(source)],
        cwd=tmp_path if entry in {"wrapper-external", "legacy-script"} else ROOT,
        env={**os.environ, "PYTHONUTF8": "1", "LOCALAPPDATA": str(tmp_path / "local"),
             "HF_HOME": str(tmp_path / "model-cache"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        text=True, capture_output=True, encoding="utf-8", timeout=20,
    )
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert json.loads(completed.stderr) == {
        "success": False,
        "error": "Analysis failed. Check the input file, options, and model setup, then retry.",
        "error_type": "ValueError",
    }
    assert private_option not in completed.stdout + completed.stderr
    assert "Traceback" not in completed.stdout + completed.stderr
    assert source.read_text(encoding="utf-8") == content
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("error", [RuntimeError("synthetic direct caller detail"), KeyboardInterrupt()])
def test_direct_main_call_still_raises_original_error(monkeypatch, error):
    spec = importlib.util.spec_from_file_location("synthetic_failure_cli", MAIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["synthetic-cli"])
    def fail(*_args):
        raise error
    monkeypatch.setattr(module, "load_input_data", fail)
    with pytest.raises(type(error)) as raised:
        module.main()
    assert raised.value is error
