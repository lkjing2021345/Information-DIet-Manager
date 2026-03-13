#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

统一算法入口：
1. 接收前端传入的浏览记录数据；
2. 串联分类、情感、相似度、评估四个算法模块；
3. 输出标准 JSON 结果，便于前后端对接。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
ALGORITHMS_DIR = CURRENT_DIR / "algorithms"
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

from algorithms.evaluator import InformationQualityEvaluator
from algorithms.sentiment import SentimentAnalyzer
from algorithms.classifier import ContentClassifier
from algorithms.similarity import SimilarityAnalyzer

def load_json_payload(input_file: Optional[str]) -> Dict[str, Any]:
    """支持从文件或标准输入读取 JSON。"""
    if input_file:
        return json.loads(Path(input_file).read_text(encoding="utf-8"))

    raw = sys.stdin.read().strip()
    if not raw:
        raise ValueError("未读取到输入数据。请通过 --input_file 传入 JSON 文件，或通过 stdin 传入 JSON。")
    return json.loads(raw)


def infer_input_format(input_file: Optional[str], input_format: str) -> str:
    """推断输入格式。"""
    if input_format != "auto":
        return input_format

    if input_file:
        suffix = Path(input_file).suffix.lower()
        if suffix == ".csv":
            return "csv"
        if suffix == ".json":
            return "json"

    return "json"


def infer_output_format(output_file: Optional[str], output_format: str) -> str:
    """推断输出格式。"""
    if output_format != "auto":
        return output_format

    if output_file:
        suffix = Path(output_file).suffix.lower()
        if suffix == ".md":
            return "markdown"
        if suffix == ".html":
            return "html"
        if suffix == ".json":
            return "json"

    return "json"


def load_input_data(input_file: Optional[str], input_format: str) -> Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]:
    """支持读取 JSON 或 CSV 输入。"""
    actual_format = infer_input_format(input_file, input_format)

    if actual_format == "csv":
        if not input_file:
            raise ValueError("CSV 输入必须通过 --input_file 指定文件路径。")
        return pd.read_csv(input_file)

    return load_json_payload(input_file)


def normalize_payload(payload: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """兼容两种 JSON 输入：直接传数组，或传 {records, options} 对象。"""
    if isinstance(payload, list):
        records = payload
        options: Dict[str, Any] = {}
    elif isinstance(payload, dict):
        records = payload.get("records", [])
        options = payload.get("options", {}) or {}
    else:
        raise ValueError("输入 JSON 必须是数组，或包含 records 的对象。")

    if not isinstance(records, list) or not records:
        raise ValueError("records 必须是非空数组。")

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(records, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"records[{idx - 1}] 必须是对象。")

        title = str(item.get("title", "")).strip()
        if not title:
            raise ValueError(f"records[{idx - 1}].title 不能为空。")

        row = dict(item)
        row["title"] = title
        row["url"] = str(item.get("url", "")).strip()
        normalized.append(row)

    return normalized, options


def normalize_dataframe_input(df: pd.DataFrame) -> pd.DataFrame:
    """标准化 CSV / DataFrame 输入。"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("CSV 输入必须可转换为 DataFrame。")
    if df.empty:
        raise ValueError("CSV 输入不能为空。")
    if "title" not in df.columns:
        raise ValueError("CSV 输入缺少必要列: title")

    normalized_df = df.copy()
    normalized_df["title"] = normalized_df["title"].fillna("").astype(str).str.strip()
    normalized_df = normalized_df[normalized_df["title"] != ""].copy()

    if normalized_df.empty:
        raise ValueError("CSV 输入中 title 不能为空。")

    if "url" not in normalized_df.columns:
        normalized_df["url"] = ""
    else:
        normalized_df["url"] = normalized_df["url"].fillna("").astype(str).str.strip()

    return normalized_df.reset_index(drop=True)


def normalize_input_data(payload: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """统一标准化 JSON / CSV 输入。"""
    if isinstance(payload, pd.DataFrame):
        return normalize_dataframe_input(payload), {}

    records, options = normalize_payload(payload)
    return build_dataframe(records), options


def build_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """前端 records 数组转 DataFrame。"""
    df = pd.DataFrame(records)
    if "url" not in df.columns:
        df["url"] = ""
    return df


def run_full_pipeline(
    df: pd.DataFrame,
    *,
    sentiment_model_path: Optional[str],
    classifier_model_path: Optional[str],
    include_emotions: bool,
    batch_size: int,
    detailed: bool,
) -> Tuple[Dict[str, Any], Any, InformationQualityEvaluator]:
    """原始数据全流程：分类 -> 情感 -> 相似度 -> 评估。"""
    sentiment_analyzer = SentimentAnalyzer(use_bert=False)
    if sentiment_model_path:
        sentiment_analyzer.load_model(sentiment_model_path)

    content_classifier = ContentClassifier(model_path=classifier_model_path)
    similarity_analyzer = SimilarityAnalyzer()

    classified_df = content_classifier.batch_predict(df, batch_size=batch_size)
    sentiment_df = sentiment_analyzer.batch_predict(
        classified_df,
        text_column="title",
        include_emotions=include_emotions,
        batch_size=batch_size,
    )
    similarity_df = similarity_analyzer.batch_calculate_similarity(sentiment_df, text_column="title")

    if "similarity" not in similarity_df.columns:
        if "similarity_to_previous" in similarity_df.columns:
            similarity_df["similarity"] = similarity_df["similarity_to_previous"]
        else:
            similarity_df["similarity"] = 0.0

    evaluator = InformationQualityEvaluator(
        sentiment_analyzer=sentiment_analyzer,
        content_classifier=content_classifier,
        similarity_analyzer=similarity_analyzer,
    )
    report = evaluator.evaluate(similarity_df, detailed=detailed)

    result = {
        "mode": "analyze",
        "summary": evaluator.generate_summary(report),
        "records": similarity_df.to_dict(orient="records"),
        "report": report.to_dict(),
    }
    return result, report, evaluator


def run_evaluate_only(df: pd.DataFrame, *, detailed: bool) -> Tuple[Dict[str, Any], Any, InformationQualityEvaluator]:
    """已包含 category/sentiment/polarity/similarity 的结果直接评估。"""
    evaluator = InformationQualityEvaluator()
    report = evaluator.evaluate(df, detailed=detailed)
    result = {
        "mode": "evaluate",
        "summary": evaluator.generate_summary(report),
        "records": df.to_dict(orient="records"),
        "report": report.to_dict(),
    }
    return result, report, evaluator


def validate_evaluate_input(df: pd.DataFrame) -> None:
    """evaluate 模式下校验前端是否传入完整评估字段。"""
    required = {"title", "url", "category", "sentiment", "polarity", "similarity"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "evaluate 模式缺少必要字段: "
            f"{missing}。"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Information Diet Manager main entry")
    parser.add_argument(
        "--mode",
        choices=["analyze", "evaluate"],
        default="analyze",
        help="analyze=原始数据全流程分析；evaluate=已加工数据直接评估",
    )
    parser.add_argument("--input_file", type=str, default=None, help="输入文件路径；支持 JSON / CSV。不传则从 stdin 读取 JSON")
    parser.add_argument(
        "--input_format",
        choices=["auto", "json", "csv"],
        default="auto",
        help="输入格式，默认 auto：按文件扩展名自动识别；无文件时默认按 JSON stdin 处理",
    )
    parser.add_argument("--output_file", type=str, default=None, help="输出文件路径；可配合 --output_format 导出 json / markdown / html")
    parser.add_argument(
        "--output_format",
        choices=["auto", "json", "markdown", "html"],
        default="auto",
        help="输出格式，默认 auto：按 output_file 扩展名自动识别；无 output_file 时打印 JSON 到 stdout",
    )
    parser.add_argument("--sentiment_model_path", type=str, default=None, help="情感模型目录，可选")
    parser.add_argument("--classifier_model_path", type=str, default=None, help="分类模型目录，可选")
    parser.add_argument("--include_emotions", action="store_true", help="是否输出情绪分析结果")
    parser.add_argument("--batch_size", type=int, default=128, help="批处理大小")
    parser.add_argument("--detailed", action="store_true", help="是否生成详细评估报告")
    return parser


def dump_output(result: Dict[str, Any], output_file: Optional[str]) -> None:
    content = json.dumps(result, ensure_ascii=False, indent=2)
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(content, encoding="utf-8")
        print(f"结果已写入: {output_file}")
    else:
        print(content)


def export_result(
    result: Dict[str, Any],
    report: Any,
    evaluator: InformationQualityEvaluator,
    output_file: Optional[str],
    output_format: str,
) -> None:
    """按指定格式输出结果。"""
    actual_format = infer_output_format(output_file, output_format)

    if output_file is None:
        dump_output(result, None)
        return

    if actual_format == "json":
        evaluator.export_report(report, output_file, format="json")
        print(f"结果已写入: {output_file}")
        return

    if actual_format in {"markdown", "html"}:
        evaluator.export_report(report, output_file, format=actual_format)
        print(f"结果已写入: {output_file}")
        return

    raise ValueError(f"不支持的输出格式: {actual_format}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    payload = load_input_data(args.input_file, args.input_format)
    df, options = normalize_input_data(payload)

    include_emotions = bool(options.get("include_emotions", args.include_emotions))
    batch_size = int(options.get("batch_size", args.batch_size))
    detailed = bool(options.get("detailed", args.detailed))

    if args.mode == "analyze":
        result, report, evaluator = run_full_pipeline(
            df,
            sentiment_model_path=args.sentiment_model_path,
            classifier_model_path=args.classifier_model_path,
            include_emotions=include_emotions,
            batch_size=batch_size,
            detailed=detailed,
        )
    else:
        validate_evaluate_input(df)
        result, report, evaluator = run_evaluate_only(df, detailed=detailed)

    export_result(
        result=result,
        report=report,
        evaluator=evaluator,
        output_file=args.output_file,
        output_format=args.output_format,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as exc:
        error_payload = {
            "success": False,
            "error": str(exc),
        }
        print(json.dumps(error_payload, ensure_ascii=False, indent=2))
        sys.exit(1)
