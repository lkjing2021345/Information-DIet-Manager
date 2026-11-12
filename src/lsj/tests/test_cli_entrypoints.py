"""CLI startup uses stdlib only; execution wiring uses synthetic model classes."""
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT / "src" / "lsj" / "src" / "main.py"
WRAPPER = ROOT / "scripts" / "run_analysis.py"
ENTRYPOINTS = ("wrapper-root", "wrapper-external", "legacy-script", "legacy-module", "alias-module")
OPTIONAL_IMPORT_GUARD = r'''
import importlib.abc
from pathlib import Path
import runpy
import sys
root, entry, arguments = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
blocked = {"pandas", "numpy", "torch", "transformers", "jieba", "cntext", "huggingface_hub",
           "algorithms", "classifier", "evaluator", "sentiment", "similarity", "markdown_builder",
           "classifier_train", "sentiment_train"}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked or fullname in {"utils.logger", "utils.hf_cache"}:
            raise AssertionError("Unexpected optional import: " + fullname)
sys.meta_path.insert(0, Guard())
if entry.endswith("-module"):
    # Match the repository-root module command's normal sys.path entry.
    sys.path.insert(0, str(root))
    name = "src.lsj.src.main" if entry == "legacy-module" else "src.analysis_engine.main"
    sys.argv = [name, *arguments]
    runpy.run_module(name, run_name="__main__")
else:
    target = root / ("src/lsj/src/main.py" if entry == "legacy-script" else "scripts/run_analysis.py")
    sys.argv = [str(target), *arguments]
    runpy.run_path(str(target), run_name="__main__")
'''


def isolated_environment(tmp_path):
    return {**os.environ, "PYTHONPATH": str(tmp_path / "unused-pythonpath"),
            "LOCALAPPDATA": str(tmp_path / "appdata"), "HOME": str(tmp_path / "home"),
            "USERPROFILE": str(tmp_path / "home"), "HF_HOME": str(tmp_path / "hf-home"),
            "HF_HUB_CACHE": str(tmp_path / "hf-hub"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def entry_command(entry):
    if entry == "wrapper-root":
        return ["scripts/run_analysis.py"]
    if entry == "wrapper-external":
        return [str(WRAPPER)]
    if entry == "legacy-script":
        return [str(MAIN)]
    if entry == "legacy-module":
        return ["-m", "src.lsj.src.main"]
    assert entry == "alias-module"
    return ["-m", "src.analysis_engine.main"]


@pytest.mark.parametrize("entry", ENTRYPOINTS)
@pytest.mark.parametrize("arguments,code", [(["--help"], 0), (["--mode", "invalid"], 2)])
def test_actual_entrypoint_works_without_site_packages_or_pythonpath(tmp_path, entry, arguments, code):
    cwd = tmp_path if entry in {"wrapper-external", "legacy-script"} else ROOT
    completed = subprocess.run(
        [sys.executable, "-E", "-B", "-S", "-X", "utf8", *entry_command(entry), *arguments],
        cwd=cwd, env=isolated_environment(tmp_path), capture_output=True, text=True, encoding="utf-8", timeout=15,
    )
    assert completed.returncode == code, completed.stdout + completed.stderr
    output = completed.stdout if code == 0 else completed.stderr
    assert "usage:" in output
    if code == 0:
        for option in ("--mode", "--input_file", "--output_format", "--batch_size", "--detailed"):
            assert option in output
    else:
        assert "invalid choice" in output
    assert "Traceback" not in completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("entry", ENTRYPOINTS)
@pytest.mark.parametrize("arguments,code", [(["-h"], 0), (["--batch_size", "invalid"], 2)])
def test_help_and_invalid_arguments_never_attempt_optional_imports(tmp_path, entry, arguments, code):
    completed = subprocess.run(
        [sys.executable, "-E", "-B", "-S", "-X", "utf8", "-c", OPTIONAL_IMPORT_GUARD,
         str(ROOT), entry, *arguments],
        cwd=tmp_path, env=isolated_environment(tmp_path), capture_output=True, text=True, encoding="utf-8", timeout=15,
    )
    assert completed.returncode == code, completed.stdout + completed.stderr
    assert "usage:" in completed.stdout + completed.stderr
    assert "Unexpected optional import" not in completed.stdout + completed.stderr
    assert "Traceback" not in completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def cli(monkeypatch):
    # Direct callers remain supported without invoking main() first.
    algorithms = str(MAIN.parent / "algorithms")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != algorithms])
    spec = importlib.util.spec_from_file_location("synthetic_analysis_cli", MAIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_json_and_csv_helpers_remain_directly_callable(cli, tmp_path):
    import pandas as pd

    source = tmp_path / "合成页面.json"
    source.write_text(json.dumps({"records": [{"title": " Synthetic title ", "url": "https://example.invalid"}],
                                  "options": {"batch_size": 3}}), encoding="utf-8")
    frame, options = cli.normalize_input_data(cli.load_input_data(str(source), "auto"))
    assert frame.to_dict(orient="records") == [{"title": "Synthetic title", "url": "https://example.invalid"}]
    assert options == {"batch_size": 3}
    csv = tmp_path / "合成页面.csv"
    pd.DataFrame({"title": [" Synthetic CSV ", " "], "url": [None, "unused"]}).to_csv(csv, index=False)
    frame, options = cli.normalize_input_data(cli.load_input_data(str(csv), "auto"))
    assert frame.to_dict(orient="records") == [{"title": "Synthetic CSV", "url": ""}]
    assert options == {}
    assert cli.build_dataframe([{"title": "Direct helper"}]).to_dict(orient="records") == [
        {"title": "Direct helper", "url": ""}]
    with pytest.raises(TypeError, match="DataFrame"):
        cli.normalize_dataframe_input([])


def test_algorithm_search_path_resolves_project_modules_from_an_unrelated_cwd(cli, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    before = list(sys.path)
    assert str(cli.ALGORITHMS_DIR) not in before
    cli._ensure_algorithm_import_path()
    for name in ("classifier", "evaluator", "sentiment", "similarity"):
        spec = importlib.machinery.PathFinder.find_spec(name, sys.path)
        assert spec is not None and Path(spec.origin) == cli.ALGORITHMS_DIR / (name + ".py")
    # Resolving source locations does not execute the model modules.
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def synthetic_algorithms(monkeypatch):
    calls = []

    class Report:
        def to_dict(self):
            return {"synthetic_report": True}

    class Classifier:
        def __init__(self, *, model_path):
            calls.append(("classifier_init", model_path))

        def batch_predict(self, frame, *, batch_size):
            calls.append(("classifier", batch_size))
            return frame.assign(category="Tools")

    class Sentiment:
        def __init__(self, *, use_bert):
            calls.append(("sentiment_init", use_bert))

        def load_model(self, path):
            calls.append(("sentiment_model", path))

        def batch_predict(self, frame, *, text_column, include_emotions, batch_size):
            calls.append(("sentiment", text_column, include_emotions, batch_size))
            return frame.assign(sentiment="Neutral", polarity=0.0)

    class Similarity:
        def batch_calculate_similarity(self, frame, *, text_column):
            calls.append(("similarity", text_column))
            return frame.assign(similarity_to_previous=0.25)

    class Evaluator:
        def __init__(self, **kwargs):
            calls.append(("evaluator_init", sorted(kwargs)))

        def evaluate(self, frame, *, detailed):
            calls.append(("evaluate", detailed, frame.to_dict(orient="records")))
            return Report()

        def generate_summary(self, report):
            return "Synthetic summary"

        def export_report(self, report, path, *, format):
            calls.append(("export", str(path), format))

    for name, class_name, implementation in (
        ("classifier", "ContentClassifier", Classifier), ("sentiment", "SentimentAnalyzer", Sentiment),
        ("similarity", "SimilarityAnalyzer", Similarity), ("evaluator", "InformationQualityEvaluator", Evaluator),
    ):
        module = types.ModuleType(name)
        setattr(module, class_name, implementation)
        monkeypatch.setitem(sys.modules, name, module)
    return calls


@pytest.mark.parametrize("mode", ["analyze", "evaluate"])
def test_modes_preserve_argument_input_and_output_wiring_with_synthetic_models(cli, synthetic_algorithms, monkeypatch, tmp_path, mode):
    calls = synthetic_algorithms
    source = tmp_path / "synthetic-input.json"
    output = tmp_path / "synthetic-output.md"
    record = {"title": "Synthetic page", "url": "https://example.invalid"}
    if mode == "evaluate":
        record.update(category="Tools", sentiment="Neutral", polarity=0.0, similarity=0.25)
    source.write_text(json.dumps({"records": [record], "options": {
        "batch_size": 7, "include_emotions": True, "detailed": True}}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["synthetic-cli", "--mode", mode, "--input_file", str(source),
                                    "--output_file", str(output), "--batch_size", "2",
                                    "--classifier_model_path", "synthetic-classifier", "--sentiment_model_path", "synthetic-sentiment"])
    cli.main()
    assert ("export", str(output), "markdown") in calls
    evaluated = next(event for event in calls if event[0] == "evaluate")
    assert evaluated[1] is True and evaluated[2][0]["similarity"] == 0.25
    if mode == "analyze":
        assert ("classifier_init", "synthetic-classifier") in calls
        assert ("sentiment_model", "synthetic-sentiment") in calls
        assert ("classifier", 7) in calls and ("sentiment", "title", True, 7) in calls
        assert ("similarity", "title") in calls
        assert ("evaluator_init", ["content_classifier", "sentiment_analyzer", "similarity_analyzer"]) in calls
    else:
        assert ("evaluator_init", []) in calls
        assert all(not event[0].startswith(("classifier", "sentiment", "similarity")) for event in calls)


def test_analyze_rejects_missing_similarity_measurement_without_constructing_evaluator(
        cli, synthetic_algorithms, monkeypatch):
    calls = synthetic_algorithms

    class MissingSimilarity:
        def batch_calculate_similarity(self, frame, *, text_column):
            calls.append(("similarity_missing", text_column))
            return frame.copy()

    monkeypatch.setattr(sys.modules["similarity"], "SimilarityAnalyzer", MissingSimilarity)
    import pandas as pd

    with pytest.raises(ValueError, match="相似度分析未生成"):
        cli.run_full_pipeline(
            pd.DataFrame({"title": ["Synthetic title"], "url": ["https://example.invalid"]}),
            sentiment_model_path=None,
            classifier_model_path=None,
            include_emotions=False,
            batch_size=4,
            detailed=False,
        )

    assert ("similarity_missing", "title") in calls
    assert not any(event[0] == "evaluator_init" for event in calls)


@pytest.mark.parametrize("invalid", [0, -1])
def test_run_full_pipeline_rejects_non_positive_batch_size(cli, synthetic_algorithms, invalid):
    import pandas as pd

    with pytest.raises(ValueError, match="batch_size 必须是正整数"):
        cli.run_full_pipeline(
            pd.DataFrame({"title": ["Synthetic title"], "url": ["https://example.invalid"]}),
            sentiment_model_path=None,
            classifier_model_path=None,
            include_emotions=False,
            batch_size=invalid,
            detailed=False,
        )


def test_dump_output_converts_non_finite_numbers_to_null(cli, capsys):
    cli.dump_output({"nan": float("nan"), "nested": [float("inf"), -float("inf")]}, None)
    output = capsys.readouterr().out
    import json

    parsed = json.loads(output, parse_constant=lambda value: pytest.fail(f"non-standard JSON constant: {value}"))
    assert parsed == {"nan": None, "nested": [None, None]}
