"""Exercise actual report generation/export; optional model dependencies are stubs.

Only synthetic frames and pytest temporary files are used. These tests do not
perform model inference or claim that Markdown renderers sanitize raw HTML.
"""
from dataclasses import replace
from html.parser import HTMLParser
import importlib.util
import json
import logging
from pathlib import Path
import sys
import types

import pandas as pd
import pytest


ALGORITHMS = Path(__file__).resolve().parents[1] / "src" / "algorithms"
PAYLOADS = [
    "</pre><script>window.idm_report_injected=1</script><pre>",
    "</pre><svg onload=window.idm_report_injected=2></svg><pre>",
    "</pre><img src=data:,invalid onerror=window.idm_report_injected=3><pre>",
    "</pre><p id=idm_report_injected>injected</p><pre>",
    "中文 & < > \"双引号\" '单引号' 😀\n下一行 &lt;script&gt;",
]


@pytest.fixture
def report_module(monkeypatch):
    def stub(name, **attrs):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    class NoModel:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("Report tests must not load or download models")

    stub("utils")
    stub("utils.logger", setup_logger=lambda *_args: logging.getLogger("report-html-fixture"))
    stub("sentiment", SentimentAnalyzer=NoModel)
    stub("classifier", ContentClassifier=NoModel)
    stub("similarity", SimilarityAnalyzer=NoModel)
    loaded = {}
    for name, filename in (
        ("markdown_builder", "markdown_builder.py"),
        ("_report_html_fixture_evaluator", "evaluator.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, ALGORITHMS / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded[filename] = module
    return loaded["evaluator.py"]


def generate_report(module, category="learning"):
    # All evaluator and report methods are real. Explicit objects prevent the
    # optional analysis constructors from running; the frame is preclassified.
    evaluator = module.InformationQualityEvaluator(
        sentiment_analyzer=object(), content_classifier=object(), similarity_analyzer=object()
    )
    frame = pd.DataFrame({
        "title": [f"合成条目 {i}" for i in range(5)],
        "url": [f"https://example.invalid/{i}" for i in range(5)],
        "category": [category] * 5,
        "sentiment": ["neutral"] * 5,
        "polarity": [0.0] * 5,
        "similarity": [0.2] * 5,
        "timestamp": pd.date_range("2026-01-01T10:00:00", periods=5, freq="min"),
    })
    return evaluator, evaluator.evaluate(frame)


def test_compare_users_failure_output_redacts_identifiers_and_errors(report_module, caplog, monkeypatch):
    evaluator = report_module.InformationQualityEvaluator(
        sentiment_analyzer=object(), content_classifier=object(), similarity_analyzer=object()
    )
    private_marker = "SYNTHETIC_PRIVATE_USER_AND_ERROR"

    def fail(_frame):
        raise RuntimeError(private_marker)

    monkeypatch.setattr(evaluator, "quick_evaluate", fail)
    with caplog.at_level(logging.ERROR, logger=report_module.logger.name):
        result = evaluator.compare_users({private_marker: pd.DataFrame()})

    failure = result["failed_users"][0]
    assert failure["user_id"] != private_marker
    assert failure["user_id"].startswith("<redacted sha256=")
    assert failure["error"] == "evaluation failed (RuntimeError)"
    assert private_marker not in json.dumps(result, ensure_ascii=False)
    assert private_marker not in caplog.text
    assert "error_type=RuntimeError" in caplog.text

    evaluator.config.min_records = 1
    caplog.clear()
    trend = evaluator.batch_evaluate(pd.DataFrame({"segment": [private_marker]}), group_by="segment")
    assert trend.empty
    assert private_marker not in caplog.text
    assert "error_type=RuntimeError" in caplog.text


class ReportHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.elements = []
        self.pre_texts = []
        self.in_pre = False

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, attrs))
        if tag == "pre":
            self.pre_texts.append("")
            self.in_pre = True

    def handle_endtag(self, tag):
        if tag == "pre":
            self.in_pre = False

    def handle_data(self, data):
        if self.in_pre:
            self.pre_texts[-1] += data


@pytest.mark.parametrize("payload", PAYLOADS)
@pytest.mark.parametrize("sink", ["category", "summary"])
def test_html_report_preserves_text_without_creating_elements(report_module, tmp_path, payload, sink):
    evaluator, report = generate_report(report_module, payload if sink == "category" else "learning")
    assert isinstance(report, report_module.EvaluationReport)
    if sink == "category":
        # This is a real input -> evaluator -> report field, not a fake to_dict.
        assert report.metrics.diversity.dominant_category == payload.strip().lower()
        assert report.metrics.diversity.category_distribution == {payload.strip().lower(): 5}
    else:
        # get_summary has no arbitrary-category field; test its public dataclass
        # text separately without pretending evaluate() supplies this content.
        report.health_status = replace(report.health_status, justification=payload)
        assert payload in report.get_summary()

    expected_summary = report.get_summary()
    expected_data = report.to_dict()
    output = tmp_path / "nested" / "report.html"
    evaluator.export_report(report, str(output), format=" HTML ")
    text = output.read_text(encoding="utf-8")
    parsed = ReportHTML()
    parsed.feed(text)
    parsed.close()

    assert parsed.elements == [
        ("html", [("lang", "zh-CN")]), ("head", []), ("meta", [("charset", "UTF-8")]),
        ("title", []), ("body", []), ("h1", []), ("h2", []), ("pre", []),
        ("h2", []), ("pre", []),
    ]
    assert parsed.pre_texts == [expected_summary, json.dumps(expected_data, ensure_ascii=False, indent=2)]
    assert json.loads(parsed.pre_texts[1]) == json.loads(json.dumps(expected_data))
    assert "信息摄取质量评估报告" in text
    assert report.to_dict() == expected_data
    assert report.get_summary() == expected_summary


@pytest.mark.parametrize("format", ["json", "md", "markdown"])
def test_other_report_formats_keep_existing_serialization(report_module, tmp_path, format):
    evaluator, report = generate_report(report_module, PAYLOADS[0])
    report.health_status = replace(report.health_status, justification=PAYLOADS[-1])
    output = tmp_path / f"report.{format}"
    evaluator.export_report(report, str(output), format=format)
    text = output.read_text(encoding="utf-8")
    if format == "json":
        assert text == json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
        assert json.loads(text) == json.loads(json.dumps(report.to_dict()))
    else:
        expected = report_module.ReportMarkdownGenerator().generate(report, detailed=True)
        assert text == expected
        assert "中文" in text
