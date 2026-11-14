#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classifier_data_collector.py

基于多模型并发蒸馏的浏览器标签页分类训练数据生成器，输出格式兼容 classifier_train.py。

核心能力：
1) 历史数据加载与去重：启动时自动读取已有数据，按业务唯一键 entry_id 去重
2) 断点续跑：进度持久化到 progress sidecar 文件，异常退出后可继续补齐差额
3) 精确目标控制：通过 target_count 精确补齐到目标总量
4) 并发增强：异步任务队列 + worker + 单写入器，保证线程安全与幂等性
5) 稳定性增强：重试、超时、异常分级、中间结果安全落盘、结构化日志
6) 适配 classifier_train.py：默认输出 JSON 数组，记录字段包含 input、label（并保留 entry_id 等元数据）

唯一条目判定规则：
- entry_id = sha1(f"{label}\t{normalize_text(input)}")
- 若历史文件已存在 entry_id / id 字段则优先使用；否则自动按上述规则回填

默认标签集合：
- News
- Tools
- Learning
- Shopping
- Social
- Entertainment
- Other

示例：
python classifier_data_collector.py \
  --output ./classifier_train.json \
  --target_count 100000 \
  --progress_path ./classifier_train.progress.json \
  --model-config '[{"name":"gpt-4o","provider":"openai","concurrency":4,"weight":0.4},{"name":"claude-3-5-sonnet","provider":"anthropic","concurrency":3,"weight":0.4},{"name":"qwen2.5-72b-instruct","provider":"local","base_url":"http://localhost:8000/v1","concurrency":6,"weight":0.2}]' \
  --temperature 0.9 \
  --batch_size 50 \
  --max_workers 30 \
  --enable_relabel_check
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np


# -----------------------------
# Utilities / Logging
# -----------------------------
def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("ClassifierDataCollector")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    if not logger.handlers:
        ch = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def log_event(logger: logging.Logger, level: str, event: str, **kwargs):
    payload = {"event": event, **kwargs}
    msg = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    getattr(logger, level.lower(), logger.info)(msg)


class InvalidConfidenceError(ValueError):
    """Explicitly invalid confidence must not become a successful record."""


def _validated_confidence(value: Any) -> float:
    message = "Confidence must be a finite number between 0 and 1."
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise InvalidConfidenceError(message)
    try:
        confidence = float(value)
    except (TypeError, ValueError, OverflowError):
        raise InvalidConfidenceError(message) from None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise InvalidConfidenceError(message)
    return confidence


class CollectorPersistenceError(RuntimeError):
    """A failed write may have partially succeeded; never replay it automatically."""

    def __init__(self, stage: str):
        self.stage = stage
        super().__init__("Collector persistence failed; inspect the output before resuming.")


class BloomFilter:
    """简单 BloomFilter 实现（无外部依赖）"""

    def __init__(self, capacity: int = 300_000, error_rate: float = 0.01):
        m = -capacity * math.log(error_rate) / (math.log(2) ** 2)
        self.size = max(8, int(m))
        k = (self.size / max(1, capacity)) * math.log(2)
        self.hash_count = max(2, int(k))
        self.bit_array = bytearray((self.size + 7) // 8)

    def _hashes(self, item: str):
        b = item.encode("utf-8", errors="ignore")
        h1 = int(hashlib.md5(b).hexdigest(), 16)
        h2 = int(hashlib.sha1(b).hexdigest(), 16)
        for i in range(self.hash_count):
            yield (h1 + i * h2) % self.size

    def add(self, item: str):
        for idx in self._hashes(item):
            self.bit_array[idx // 8] |= 1 << (idx % 8)

    def __contains__(self, item: str):
        for idx in self._hashes(item):
            if not (self.bit_array[idx // 8] & (1 << (idx % 8))):
                return False
        return True


LABEL_ALIASES = {
    "news": "News",
    "资讯": "News",
    "新闻": "News",
    "tools": "Tools",
    "tool": "Tools",
    "工具": "Tools",
    "learning": "Learning",
    "learn": "Learning",
    "学习": "Learning",
    "shopping": "Shopping",
    "shop": "Shopping",
    "购物": "Shopping",
    "social": "Social",
    "社交": "Social",
    "entertainment": "Entertainment",
    "entertain": "Entertainment",
    "娱乐": "Entertainment",
    "other": "Other",
    "其它": "Other",
    "其他": "Other",
}


def safe_json_extract(text: str) -> Optional[Any]:
    text = str(text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                continue
    return None


def normalize_text(s: Any) -> str:
    s = str(s or "").strip().replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.strip("“”\"'")
    return s


def normalize_label(label: Any) -> str:
    raw = normalize_text(label)
    if not raw:
        return ""
    return LABEL_ALIASES.get(raw.lower(), raw)


def compute_entry_id(text: str, label: str) -> str:
    canonical = f"{normalize_label(label)}\t{normalize_text(text)}"
    return hashlib.sha1(canonical.encode("utf-8", errors="ignore")).hexdigest()


# -----------------------------
# Config Models
# -----------------------------
@dataclass
class RetryConfig:
    max_retries: int = 3
    backoff_base: float = 0.5
    backoff_max: float = 8.0
    request_timeout: float = 30.0
    task_timeout: float = 60.0


@dataclass
class RuntimeConfig:
    output: str
    target_count: int
    categories: List[str]
    distribution: List[float]
    temperature: float = 0.9
    max_tokens: int = 220
    batch_size: int = 50
    max_workers: int = 30
    similarity_threshold: float = 0.92
    random_seed: int = 42
    log_level: str = "INFO"
    enable_semantic_dedup: bool = False
    progress_path: Optional[str] = None
    flush_every: int = 200
    max_attempt_factor: int = 40
    output_format: Optional[str] = None
    enable_relabel_check: bool = False
    min_confidence: float = 0.0
    domains: List[str] = field(default_factory=lambda: [
        "资讯门户", "电商平台", "学习平台", "社交社区", "视频娱乐", "生产力工具", "企业办公", "论坛博客", "下载资源", "搜索导航"
    ])
    domain_distribution: Optional[List[float]] = None
    page_types: List[str] = field(default_factory=lambda: [
        "首页", "详情页", "列表页", "搜索结果页", "文档页", "登录页", "帖子页", "播放页", "商品页", "设置页"
    ])
    page_type_distribution: Optional[List[float]] = None
    language_styles: List[str] = field(default_factory=lambda: [
        "简洁标题", "营销风格", "教程风格", "社区风格", "官方风格"
    ])
    language_style_distribution: Optional[List[float]] = None
    brand_presence: List[str] = field(default_factory=lambda: ["含品牌名", "不含品牌名"])
    brand_presence_distribution: Optional[List[float]] = None
    title_length_types: List[str] = field(default_factory=lambda: ["短标题", "中标题", "长标题"])
    title_length_distribution: Optional[List[float]] = None
    include_chinese_ratio: float = 0.7
    include_english_ratio: float = 0.2
    include_mixed_ratio: float = 0.1
    enable_contrast_pairs: bool = True
    contrast_pair_ratio: float = 0.25
    retry: RetryConfig = field(default_factory=RetryConfig)


@dataclass
class ModelConfig:
    name: str
    provider: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    concurrency: int = 3
    weight: float = 1.0
    timeout: float = 30.0
    max_retries: int = 3
    qps_limit: float = 5.0


@dataclass
class ModelStats:
    total_calls: int = 0
    success_calls: int = 0
    failed_calls: int = 0
    total_latency: float = 0.0
    latencies: List[float] = field(default_factory=list)

    def success_rate(self) -> float:
        return 0.0 if self.total_calls == 0 else self.success_calls / self.total_calls

    def avg_latency(self) -> float:
        return 0.0 if self.success_calls == 0 else self.total_latency / self.success_calls

    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        idx = int(0.95 * (len(sorted_l) - 1))
        return sorted_l[idx]


@dataclass
class ProgressState:
    output_path: str
    target_count: int
    existing_count: int = 0
    deduped_existing_count: int = 0
    accepted_new_count: int = 0
    generated_count: int = 0
    duplicate_count: int = 0
    filtered_count: int = 0
    failed_count: int = 0
    relabel_rejected_count: int = 0
    attempt_count: int = 0
    label_counts: Dict[str, int] = field(default_factory=dict)
    domain_counts: Dict[str, int] = field(default_factory=dict)
    page_type_counts: Dict[str, int] = field(default_factory=dict)
    language_style_counts: Dict[str, int] = field(default_factory=dict)
    brand_presence_counts: Dict[str, int] = field(default_factory=dict)
    title_length_counts: Dict[str, int] = field(default_factory=dict)
    language_mix_counts: Dict[str, int] = field(default_factory=dict)
    contrast_pair_count: int = 0
    last_update_ts: float = 0.0
    status: str = "initialized"

    @property
    def total_effective_count(self) -> int:
        return self.existing_count + self.accepted_new_count


@dataclass
class Record:
    entry_id: str
    input: str
    label: str
    confidence: float = 1.0
    model: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "input": self.input,
            "label": self.label,
            "confidence": self.confidence,
            "model": self.model,
            "created_at": self.created_at,
        }


# -----------------------------
# Atomic File IO / Progress
# -----------------------------
class AtomicFileIO:
    @staticmethod
    def atomic_write_text(path: str, content: str, encoding: str = "utf-8"):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, encoding=encoding, dir=str(path_obj.parent)) as tf:
            tf.write(content)
            temp_path = tf.name
        os.replace(temp_path, path)


class ProgressTracker:
    def __init__(self, progress_path: str, logger: logging.Logger):
        self.progress_path = progress_path
        self.logger = logger
        self.lock = asyncio.Lock()

    def load(self) -> Optional[ProgressState]:
        if not self.progress_path or not os.path.exists(self.progress_path):
            return None
        try:
            with open(self.progress_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            state = ProgressState(**data)
            log_event(self.logger, "info", "progress_loaded", progress_path=self.progress_path)
            return state
        except Exception as e:
            log_event(self.logger, "warning", "progress_load_failed", progress_path=self.progress_path,
                      error_type=type(e).__name__)
            return None

    async def save(self, state: ProgressState):
        if not self.progress_path:
            return
        async with self.lock:
            state.last_update_ts = time.time()
            AtomicFileIO.atomic_write_text(
                self.progress_path,
                json.dumps(asdict(state), ensure_ascii=False, indent=2),
            )
            log_event(
                self.logger,
                "info",
                "progress_saved",
                progress_path=self.progress_path,
                total_effective_count=state.total_effective_count,
                accepted_new_count=state.accepted_new_count,
            )


# -----------------------------
# Historical Data Loader / Writer
# -----------------------------
class DataStore:
    """加载历史数据、按业务唯一键去重、幂等写入、兼容 JSON/JSONL/CSV。"""

    def __init__(self, output_path: str, logger: logging.Logger, output_format: Optional[str] = None):
        self.output_path = output_path
        self.logger = logger
        self.output_format = output_format or self._detect_format(output_path)

    @staticmethod
    def _detect_format(path: str) -> str:
        suffix = Path(path).suffix.lower()
        if suffix == ".jsonl":
            return "jsonl"
        if suffix == ".csv":
            return "csv"
        return "json"

    def load_existing_records(self) -> Tuple[List[Record], Dict[str, int]]:
        if not os.path.exists(self.output_path):
            return [], {"loaded": 0, "dedup_removed": 0, "invalid": 0}

        try:
            if self.output_format == "csv":
                rows = self._read_csv(self.output_path)
            elif self.output_format == "jsonl":
                rows = self._read_jsonl(self.output_path)
            else:
                rows = self._read_json(self.output_path)
        except Exception as e:
            raise IOError(f"读取历史数据失败: {e}") from e

        unique: Dict[str, Record] = {}
        invalid = 0
        for row in rows:
            try:
                # Validate even duplicate rows. An invalid score must stop
                # initialization before history cleanup can rewrite evidence.
                confidence = _validated_confidence(row.get("confidence", 1.0))
                text = normalize_text(row.get("input") or row.get("text") or row.get("title") or "")
                label = normalize_label(row.get("label", ""))
                if not text or not label:
                    invalid += 1
                    continue
                entry_id = str(row.get("entry_id") or row.get("id") or compute_entry_id(text, label)).strip()
                if not entry_id:
                    invalid += 1
                    continue
                if entry_id not in unique:
                    unique[entry_id] = Record(
                        entry_id=entry_id,
                        input=text,
                        label=label,
                        confidence=confidence,
                        model=str(row.get("model", "")),
                        created_at=float(row.get("created_at", time.time())),
                    )
            except InvalidConfidenceError:
                raise InvalidConfidenceError(
                    "Historical confidence is invalid; the original output is unchanged."
                ) from None
            except Exception:
                invalid += 1

        records = list(unique.values())
        stats = {
            "loaded": len(rows),
            "dedup_removed": max(0, len(rows) - len(records)),
            "invalid": invalid,
        }
        return records, stats

    def rewrite_all(self, records: List[Record]):
        try:
            if self.output_format == "csv":
                self._write_csv(records)
            elif self.output_format == "jsonl":
                self._write_jsonl(records)
            else:
                self._write_json(records)
        except Exception as e:
            raise IOError(f"重写数据文件失败: {e}") from e

    def append_records(self, records: List[Record]):
        if not records:
            return
        try:
            if self.output_format == "csv":
                Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
                fields = self._csv_append_fields()
                needs_header = fields is None
                fields = fields or ["entry_id", "input", "label", "confidence", "model", "created_at"]
                needs_separator = not needs_header and self._append_needs_separator()
                with open(self.output_path, "a", encoding="utf-8", newline="") as f:
                    if needs_separator:
                        f.write("\n")
                    writer = csv.DictWriter(f, fieldnames=fields)
                    if needs_header:
                        writer.writeheader()
                    for r in records:
                        row = r.to_dict()
                        row.update(id=r.entry_id, text=r.input, title=r.input)
                        writer.writerow({field: row.get(field, "") for field in fields})
            elif self.output_format == "jsonl":
                Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
                needs_separator = self._append_needs_separator()
                with open(self.output_path, "a", encoding="utf-8") as f:
                    if needs_separator:
                        f.write("\n")
                    for r in records:
                        f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            else:
                existing, _ = self.load_existing_records()
                merged = {r.entry_id: r for r in existing}
                for r in records:
                    merged[r.entry_id] = r
                self.rewrite_all(list(merged.values()))
        except Exception as e:
            raise IOError(f"写入结果失败: {e}") from e

    def _csv_append_fields(self) -> Optional[List[str]]:
        if not os.path.exists(self.output_path):
            return None
        with open(self.output_path, "r", encoding="utf-8-sig", newline="") as f:
            fields = next(csv.reader(f, strict=True), None)
        if fields is not None and (
            not all(field.strip() for field in fields)
            or len(set(fields)) != len(fields)
            or "label" not in fields
            or not {"input", "text", "title"}.intersection(fields)
        ):
            raise ValueError("CSV header is not supported")
        return fields

    def _append_needs_separator(self) -> bool:
        if not os.path.exists(self.output_path):
            return False
        with open(self.output_path, "rb") as f:
            if f.seek(0, os.SEEK_END) == 0:
                return False
            f.seek(-1, os.SEEK_END)
            return f.read(1) not in (b"\r", b"\n")

    def _read_csv(self, path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    def _read_jsonl(self, path: str) -> List[Dict[str, Any]]:
        rows = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _read_json(self, path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON 文件内容必须为数组")
        return data

    def _write_csv(self, records: List[Record]):
        parent = Path(self.output_path).parent
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", newline="", dir=str(parent)) as tf:
            writer = csv.DictWriter(
                tf,
                fieldnames=["entry_id", "input", "label", "confidence", "model", "created_at"],
            )
            writer.writeheader()
            for row in records:
                writer.writerow(row.to_dict())
            temp_path = tf.name
        os.replace(temp_path, self.output_path)

    def _write_jsonl(self, records: List[Record]):
        content = "\n".join(json.dumps(r.to_dict(), ensure_ascii=False) for r in records)
        if content:
            content += "\n"
        AtomicFileIO.atomic_write_text(self.output_path, content)

    def _write_json(self, records: List[Record]):
        AtomicFileIO.atomic_write_text(
            self.output_path,
            json.dumps([r.to_dict() for r in records], ensure_ascii=False, indent=2),
        )


# -----------------------------
# Model Layer
# -----------------------------
class CircuitBreaker:
    def __init__(self, fail_threshold: int = 5, reset_timeout: float = 30.0):
        self.fail_threshold = fail_threshold
        self.reset_timeout = reset_timeout
        self.fail_count = 0
        self.open_until = 0.0

    def is_open(self) -> bool:
        return time.time() < self.open_until

    def on_success(self):
        self.fail_count = 0
        self.open_until = 0.0

    def on_failure(self):
        self.fail_count += 1
        if self.fail_count >= self.fail_threshold:
            self.open_until = time.time() + self.reset_timeout


class BaseModelClient:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.http: Optional[httpx.AsyncClient] = None
        self._closed = False
        self._close_failed = False
        self._close_lock = asyncio.Lock()
        self.last_request_ts = 0.0
        self.min_interval = 1.0 / max(0.1, cfg.qps_limit)
        self._rate_limit_lock = asyncio.Lock()

    def _get_http(self) -> httpx.AsyncClient:
        if self._closed:
            raise RuntimeError("Model client is closed.")
        if self.http is None:
            # Allocate at the async request site, never during construction.
            # No await separates the check, allocation, and ownership assignment.
            self.http = httpx.AsyncClient(timeout=self.cfg.timeout)
        return self.http

    async def _rate_limit_wait(self):
        async with self._rate_limit_lock:
            now = time.time()
            delta = now - self.last_request_ts
            if delta < self.min_interval:
                await asyncio.sleep(self.min_interval - delta)
            self.last_request_ts = time.time()

    async def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        raise NotImplementedError

    async def close(self):
        self._closed = True
        async with self._close_lock:
            if self._close_failed:
                raise RuntimeError("Model client cleanup previously failed; resource state is unknown.") from None
            if self.http is not None:
                try:
                    await self.http.aclose()
                except BaseException:
                    # HTTPX may already mark itself closed; another aclose can
                    # be a no-op, so retain ownership and the failed state.
                    self._close_failed = True
                    raise
                self.http = None


class OpenAIClient(BaseModelClient):
    async def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        await self._rate_limit_wait()
        base = self.cfg.base_url or "https://api.openai.com/v1"
        url = f"{base.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg.name,
            "messages": [
                {
                    "role": "system",
                    "content": "你是网页标题分类训练数据生成助手。输出必须严格 JSON，不允许额外解释。",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        http = self._get_http()
        r = await http.post(url, headers=headers, json=payload)
        if r.status_code in {400, 404, 422}:
            payload.pop("response_format", None)
            r = await http.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return self._extract_message_text(data)

    @staticmethod
    def _extract_message_text(data: Dict[str, Any]) -> str:
        message = data["choices"][0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            joined = "".join(parts).strip()
            if joined:
                return joined
        for key in ["reasoning_content", "text", "output_text"]:
            if isinstance(message.get(key), str) and message[key].strip():
                return message[key]
        return json.dumps(data, ensure_ascii=False)


class AnthropicClient(BaseModelClient):
    async def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        await self._rate_limit_wait()
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.cfg.api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.cfg.name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        http = self._get_http()
        r = await http.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        if isinstance(data.get("content"), list):
            return "".join(str(p.get("text", "")) for p in data["content"] if isinstance(p, dict))
        return json.dumps(data, ensure_ascii=False)


class LocalOpenAICompatibleClient(OpenAIClient):
    pass


class ModelPool:
    def __init__(self, model_configs: List[ModelConfig], logger: logging.Logger):
        model_configs = list(model_configs)
        for cfg in model_configs:
            if isinstance(cfg.concurrency, bool) or not isinstance(cfg.concurrency, int) or cfg.concurrency <= 0:
                raise ValueError("Model concurrency must be a positive integer.")
        self.logger = logger
        self.cfgs: Dict[str, ModelConfig] = {c.name: c for c in model_configs}
        self.clients: Dict[str, BaseModelClient] = {}
        self.semaphores: Dict[str, asyncio.Semaphore] = {}
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.stats: Dict[str, ModelStats] = {}
        self._init_clients()

    def _init_clients(self):
        for cfg in self.cfgs.values():
            provider = cfg.provider.lower()
            if provider == "openai":
                client = OpenAIClient(cfg)
            elif provider == "anthropic":
                client = AnthropicClient(cfg)
            elif provider == "local":
                client = LocalOpenAICompatibleClient(cfg)
            else:
                raise ValueError(f"Unsupported provider: {cfg.provider}")
            self.clients[cfg.name] = client
            self.semaphores[cfg.name] = asyncio.Semaphore(cfg.concurrency)
            self.breakers[cfg.name] = CircuitBreaker()
            self.stats[cfg.name] = ModelStats()

    def _available_models(self) -> List[ModelConfig]:
        return [cfg for name, cfg in self.cfgs.items() if not self.breakers[name].is_open()]

    def _pick_model(self) -> Optional[ModelConfig]:
        available = self._available_models()
        if not available:
            return None
        weights = [max(0.0, c.weight) for c in available]
        return random.choice(available) if sum(weights) == 0 else random.choices(available, weights=weights, k=1)[0]

    async def generate(self, prompt: str, temperature: float, max_tokens: int) -> Tuple[str, str]:
        last_err = None
        max_attempts = max(3, len(self.cfgs) * 2)
        for _ in range(max_attempts):
            cfg = self._pick_model()
            if cfg is None:
                await asyncio.sleep(0.5)
                continue

            name = cfg.name
            client = self.clients[name]
            sem = self.semaphores[name]
            st = self.stats[name]

            for retry in range(cfg.max_retries):
                try:
                    async with sem:
                        st.total_calls += 1
                        t0 = time.time()
                        resp = await client.generate(prompt, temperature, max_tokens)
                        latency = time.time() - t0
                    st.success_calls += 1
                    st.total_latency += latency
                    st.latencies.append(latency)
                    self.breakers[name].on_success()
                    return resp, name
                except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
                    last_err = e
                    st.failed_calls += 1
                    self.breakers[name].on_failure()
                    backoff = 0.5 * (2 ** retry) + random.uniform(0, 0.3)
                    log_event(
                        self.logger,
                        "warning",
                        "model_request_retry",
                        model=name,
                        retry=retry + 1,
                        error_type=type(e).__name__,
                        backoff=round(backoff, 3),
                    )
                    await asyncio.sleep(backoff)
                except Exception as e:
                    last_err = e
                    st.failed_calls += 1
                    self.breakers[name].on_failure()
                    await asyncio.sleep(0.2)
        raise RuntimeError("All models failed.") from last_err

    async def close(self):
        first_error = None
        for client in self.clients.values():
            try:
                await client.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


# -----------------------------
# Core Collector
# -----------------------------
class ClassifierDataCollector:
    """面向 classifier_train.py 的高吞吐蒸馏数据生成器。"""

    def __init__(self, runtime_cfg: RuntimeConfig, model_configs: List[ModelConfig]):
        # Invalid configuration must fail before logger/client/file side effects.
        try:
            min_confidence = _validated_confidence(runtime_cfg.min_confidence)
        except InvalidConfidenceError:
            raise InvalidConfidenceError(
                "min_confidence must be a finite number between 0 and 1."
            ) from None
        self.cfg = runtime_cfg
        self.logger = setup_logger(runtime_cfg.log_level)
        random.seed(runtime_cfg.random_seed)
        np.random.seed(runtime_cfg.random_seed)

        self.pool = ModelPool(model_configs, self.logger)
        self.store = DataStore(runtime_cfg.output, self.logger, runtime_cfg.output_format)
        progress_path = runtime_cfg.progress_path or f"{runtime_cfg.output}.progress.json"
        self.progress = ProgressTracker(progress_path, self.logger)

        self.categories = runtime_cfg.categories
        self.distribution = runtime_cfg.distribution
        self.target_count = runtime_cfg.target_count
        self.temperature = runtime_cfg.temperature
        self.max_tokens = runtime_cfg.max_tokens
        self.batch_size = max(1, runtime_cfg.batch_size)
        self.max_workers = max(1, runtime_cfg.max_workers)
        self.similarity_threshold = runtime_cfg.similarity_threshold
        self.enable_semantic_dedup = bool(runtime_cfg.enable_semantic_dedup)
        self.flush_every = max(1, runtime_cfg.flush_every)
        self.enable_relabel_check = bool(runtime_cfg.enable_relabel_check)
        self.min_confidence = min_confidence

        self.domains = runtime_cfg.domains
        self.page_types = runtime_cfg.page_types
        self.language_styles = runtime_cfg.language_styles
        self.brand_presence = runtime_cfg.brand_presence
        self.title_length_types = runtime_cfg.title_length_types
        self.enable_contrast_pairs = bool(runtime_cfg.enable_contrast_pairs)
        self.contrast_pair_ratio = max(0.0, min(1.0, float(runtime_cfg.contrast_pair_ratio)))
        self._queued_specs: List[Dict[str, str]] = []

        self.disallowed_patterns = [
            r"^示例", r"^模板", r"^标题", r"^网页标题", r"^test", r"^demo", r"^untitled$"
        ]
        self.login_keywords = ["登录", "注册", "验证码", "sign in", "log in"]
        self.settings_keywords = ["设置", "偏好", "账号安全", "preferences", "settings"]

        self.exact_set: set = set()
        self.bloom = BloomFilter(capacity=max(50000, runtime_cfg.target_count * 3), error_rate=0.005)
        self.id_set: set = set()

        self.embed_model = None
        self.embeddings = None
        self.embedding_texts: List[str] = []
        self._embed_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._writer_lock = asyncio.Lock()
        self._pending_flush: List[Record] = []
        self._pending_flush_ids: set = set()
        self._last_progress_log = 0
        self._stop_requested = False
        self._persistence_error: Optional[CollectorPersistenceError] = None

        self._try_load_sbert()

        self.category_targets = self._build_targets(self.categories, self.distribution, self.target_count)
        self.domain_targets = self._build_targets(
            self.domains,
            runtime_cfg.domain_distribution or [1 / len(self.domains)] * len(self.domains),
            self.target_count,
        )
        self.page_type_targets = self._build_targets(
            self.page_types,
            runtime_cfg.page_type_distribution or [1 / len(self.page_types)] * len(self.page_types),
            self.target_count,
        )
        self.language_style_targets = self._build_targets(
            self.language_styles,
            runtime_cfg.language_style_distribution or [1 / len(self.language_styles)] * len(self.language_styles),
            self.target_count,
        )
        self.brand_presence_targets = self._build_targets(
            self.brand_presence,
            runtime_cfg.brand_presence_distribution or [0.55, 0.45][: len(self.brand_presence)] if len(self.brand_presence) <= 2 else [1 / len(self.brand_presence)] * len(self.brand_presence),
            self.target_count,
        )
        self.title_length_targets = self._build_targets(
            self.title_length_types,
            runtime_cfg.title_length_distribution or [0.25, 0.5, 0.25][: len(self.title_length_types)] if len(self.title_length_types) <= 3 else [1 / len(self.title_length_types)] * len(self.title_length_types),
            self.target_count,
        )
        self.language_mix_targets = self._build_targets(
            ["中文", "英文", "中英混合"],
            [runtime_cfg.include_chinese_ratio, runtime_cfg.include_english_ratio, runtime_cfg.include_mixed_ratio],
            self.target_count,
        )

        self.state = ProgressState(
            output_path=self.cfg.output,
            target_count=self.target_count,
            label_counts={c: 0 for c in self.categories},
            domain_counts={k: 0 for k in self.domain_targets},
            page_type_counts={k: 0 for k in self.page_type_targets},
            language_style_counts={k: 0 for k in self.language_style_targets},
            brand_presence_counts={k: 0 for k in self.brand_presence_targets},
            title_length_counts={k: 0 for k in self.title_length_targets},
            language_mix_counts={k: 0 for k in self.language_mix_targets},
            status="initializing",
        )
        self.start_ts = time.time()
        self.initial_existing_count = 0

    def _try_load_sbert(self):
        if not self.enable_semantic_dedup:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            log_event(self.logger, "info", "sbert_loaded", model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        except Exception as e:
            self.embed_model = None
            log_event(self.logger, "warning", "sbert_unavailable", error_type=type(e).__name__)

    @staticmethod
    def _build_targets(keys: List[str], ratios: List[float], total: int) -> Dict[str, int]:
        ratios_arr = np.array(ratios, dtype=float)
        if ratios_arr.sum() <= 0:
            ratios_arr = np.ones(len(keys), dtype=float)
        ratios_arr = ratios_arr / ratios_arr.sum()
        raw = ratios_arr * total
        floor = np.floor(raw).astype(int)
        remain = total - floor.sum()
        frac_idx = np.argsort(-(raw - floor))
        for i in range(remain):
            floor[frac_idx[i]] += 1
        return {k: int(v) for k, v in zip(keys, floor)}

    def _choose_from_remaining(self, targets: Dict[str, int], used: Dict[str, int]) -> str:
        remain_items = [(k, targets[k] - used.get(k, 0)) for k in targets]
        remain_items = [(k, v) for k, v in remain_items if v > 0]
        if not remain_items:
            return random.choice(list(targets.keys()))
        keys, weights = zip(*remain_items)
        return random.choices(keys, weights=weights, k=1)[0]

    def _choose_contrast_label(self, label: str) -> str:
        contrast_map = {
            "News": ["Entertainment", "Social", "Other"],
            "Tools": ["Entertainment", "Shopping", "Social"],
            "Learning": ["Entertainment", "Social", "News"],
            "Shopping": ["Tools", "Learning", "News"],
            "Social": ["Tools", "Learning", "News"],
            "Entertainment": ["Learning", "Tools", "News"],
            "Other": ["News", "Tools", "Shopping"],
        }
        candidates = [x for x in contrast_map.get(label, self.categories) if x != label]
        return random.choice(candidates or [x for x in self.categories if x != label])

    def _build_base_spec(self) -> Dict[str, str]:
        return {
            "label": self._choose_from_remaining(self.category_targets, self.state.label_counts),
            "domain": self._choose_from_remaining(self.domain_targets, self.state.domain_counts),
            "page_type": self._choose_from_remaining(self.page_type_targets, self.state.page_type_counts),
            "language_style": self._choose_from_remaining(self.language_style_targets, self.state.language_style_counts),
            "brand_presence": self._choose_from_remaining(self.brand_presence_targets, self.state.brand_presence_counts),
            "title_length": self._choose_from_remaining(self.title_length_targets, self.state.title_length_counts),
            "language_mix": self._choose_from_remaining(self.language_mix_targets, self.state.language_mix_counts),
        }

    def _build_one_spec(self) -> Dict[str, str]:
        if self._queued_specs:
            return self._queued_specs.pop(0)

        spec = self._build_base_spec()
        if self.enable_contrast_pairs and random.random() < self.contrast_pair_ratio and self._remaining_target() > 1:
            contrast_group_id = hashlib.sha1(
                f"{time.time()}-{random.random()}-{spec['domain']}-{spec['page_type']}".encode("utf-8")
            ).hexdigest()[:16]
            contrast_label = self._choose_contrast_label(spec["label"])
            spec["contrast_group_id"] = contrast_group_id
            spec["contrast_label"] = contrast_label
            paired_spec = dict(spec)
            paired_spec["label"] = contrast_label
            paired_spec["contrast_label"] = spec["label"]
            self._queued_specs.append(paired_spec)
        return spec

    def _build_prompt(self, spec: Dict[str, str]) -> str:
        length_map = {
            "短标题": [8, 18],
            "中标题": [19, 36],
            "长标题": [37, 60],
        }
        lang_examples = {
            "中文": ["知乎 - 职场转行经验分享", "淘宝网 - 夏季新款连衣裙"],
            "英文": ["GitHub - Issue Tracker Dashboard", "Netflix - Continue Watching"],
            "中英混合": ["Bilibili - Python 入门教程合集", "小红书 - Travel Vlog Ideas"]
        }
        label_examples = {
            "News": ["财联社 - 今日早报：A股盘前重要资讯", "BBC News - World Headlines"],
            "Tools": ["Notion - 团队项目看板", "GitHub - Repository Settings"],
            "Learning": ["Coursera - Machine Learning Week 3", "菜鸟教程 - Python while 循环"],
            "Shopping": ["京东 - iPhone 15 Pro 手机报价", "Amazon.com - Gaming Laptop Deals"],
            "Social": ["微博 - 热门评论与互动", "Discord - Community Chat"],
            "Entertainment": ["爱奇艺 - 热播电视剧全集", "Spotify - Daily Mix 1"],
            "Other": ["登录 - 账号安全验证", "系统设置 - 偏好配置"]
        }
        prompt = {
            "task": "生成一条高质量网页/浏览器标签页标题分类训练样本，用于 classifier_train.py。",
            "output_requirements": {
                "format": '{"input":"...","label":"...","confidence":0.98}',
                "must_follow": [
                    "只输出一个 JSON 对象",
                    "input 字段必须是可直接用于训练的真实网页标题",
                    "label 必须与 target_label 完全一致",
                    "confidence 必须在 0.80 到 1.00 之间",
                    "不要输出解释、不要输出 markdown"
                ],
            },
            "constraints": {
                "target_label": spec["label"],
                "domain": spec["domain"],
                "page_type": spec["page_type"],
                "language_style": spec["language_style"],
                "brand_presence": spec["brand_presence"],
                "title_length_chars": length_map.get(spec["title_length"], [18, 36]),
                "language_mix": spec["language_mix"],
                "naturalness": "标题必须像真实网站标题/网页 tab 标题，而不是句子解释或需求描述",
                "diversity": [
                    "避免模板化重复，如“欢迎来到某某网站”",
                    "优先生成具体栏目名、产品名、课程名、帖子标题、视频标题、页面功能标题",
                    "同类标签在不同 domain/page_type 下标题风格要明显不同",
                    "可包含品牌、栏目、站点分隔符，如 - | _，但要自然"
                ],
                "forbidden": [
                    "明显占位词",
                    "AI 说明文字",
                    "与 target_label 冲突的页面用途",
                    "乱码",
                    "超长标题",
                    "重复模板句"
                ],
            },
            "contrast_guidance": (
                f"如果可行，请让该标题与 {spec.get('contrast_label', '')} 标签在相近场景下具有明确可区分性，但不要提及对照标签名。"
                if spec.get("contrast_label") else ""
            ),
            "few_shot": [
                {"input": label_examples[spec["label"]][0], "label": spec["label"], "confidence": 0.98},
                {"input": label_examples[spec["label"]][-1], "label": spec["label"], "confidence": 0.97},
                {"input": random.choice(lang_examples.get(spec["language_mix"], [label_examples[spec["label"]][0]])), "label": spec["label"], "confidence": 0.95},
            ],
        }
        return json.dumps(prompt, ensure_ascii=False)

    def _build_relabel_prompt(self, title: str) -> str:
        prompt = {
            "task": "判断给定网页标题最合适的单标签类别。",
            "allowed_labels": self.categories,
            "rules": [
                "News: 新闻、资讯、时事、媒体报道、快讯、公告新闻页",
                "Tools: 工具、控制台、文档平台中的编辑器/管理台、生产力软件、开发工具",
                "Learning: 教程、课程、题库、文档学习页、知识讲解",
                "Shopping: 商品详情、购物平台、订单购物、促销、购买行为",
                "Social: 社区、论坛、聊天、私信、评论互动、社交平台主页或帖子页",
                "Entertainment: 视频、音乐、直播、游戏、影视、娱乐内容消费",
                "Other: 登录、设置、错误页、下载页、本地页、无法明确归类页"
            ],
            "output_format": '{"label":"...","confidence":0.98}',
            "input": title,
        }
        return json.dumps(prompt, ensure_ascii=False)

    def quality_filter(self, text: str, label: str) -> bool:
        text = normalize_text(text)
        label = normalize_label(label)
        if not text or not label:
            return False
        if label not in self.categories:
            return False
        if len(text) < 4 or len(text) > 80:
            return False
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in self.disallowed_patterns):
            return False
        if re.search(r"[\n\r\t]", text):
            return False
        if re.search(r"[{}\[\]]", text):
            return False
        if re.search(r"(请生成|输出json|分类结果|训练样本)", text, flags=re.IGNORECASE):
            return False

        ascii_ratio = sum(1 for ch in text if ord(ch) < 128) / max(1, len(text))
        if ascii_ratio > 0.95 and label not in {"Tools", "Entertainment", "News", "Shopping"}:
            return False

        hard_conflicts = {
            "Shopping": ["教程", "课程", "lesson"],
            "Learning": ["立即购买", "加入购物车", "秒杀"],
            "Social": ["下单", "结算", "checkout"],
            "Entertainment": ["登录验证", "系统设置"],
        }
        for bad in hard_conflicts.get(label, []):
            if bad.lower() in text.lower():
                return False

        if label == "Other":
            if not any(k.lower() in text.lower() for k in self.login_keywords + self.settings_keywords + ["404", "error", "下载", "download"]):
                return False

        return True

    def _quick_duplicate(self, entry_id: str, text: str) -> bool:
        return entry_id in self.id_set or (text in self.bloom and text in self.exact_set)

    async def semantic_deduplicate(self, entry_id: str, text: str) -> bool:
        if self._quick_duplicate(entry_id, text):
            return False

        if self.embed_model is None:
            self.id_set.add(entry_id)
            self.exact_set.add(text)
            self.bloom.add(text)
            return True

        async with self._embed_lock:
            if entry_id in self.id_set:
                return False
            emb = self.embed_model.encode([text], normalize_embeddings=True)[0]
            emb = np.asarray(emb, dtype=np.float32)
            if self.embeddings is not None and len(self.embeddings) > 0:
                sims = self.embeddings @ emb
                if float(np.max(sims)) >= self.similarity_threshold:
                    return False
            self.embeddings = emb.reshape(1, -1) if self.embeddings is None else np.vstack([self.embeddings, emb.reshape(1, -1)])
            self.embedding_texts.append(text)
            self.id_set.add(entry_id)
            self.exact_set.add(text)
            self.bloom.add(text)
            return True

    async def _load_existing_state(self):
        log_event(self.logger, "info", "history_load_start", output=self.cfg.output)
        records, stats = self.store.load_existing_records()
        if stats["dedup_removed"] > 0 or stats["invalid"] > 0:
            log_event(
                self.logger,
                "warning",
                "history_rewrite_required",
                dedup_removed=stats["dedup_removed"],
                invalid=stats["invalid"],
            )
            self.store.rewrite_all(records)

        self.initial_existing_count = len(records)
        self.state.existing_count = len(records)
        self.state.deduped_existing_count = len(records)
        # Persisted rows are the current baseline, including earlier runs' additions.
        # A sidecar may predate or outlive the output and cannot establish row counts.
        self.state.accepted_new_count = 0
        self.state.label_counts = {c: 0 for c in self.categories}

        for r in records:
            self.id_set.add(r.entry_id)
            self.exact_set.add(r.input)
            self.bloom.add(r.input)
            if r.label in self.state.label_counts:
                self.state.label_counts[r.label] += 1

        if self.embed_model is not None and records:
            try:
                embs = self.embed_model.encode([r.input for r in records], normalize_embeddings=True, batch_size=128)
                self.embeddings = np.asarray(embs, dtype=np.float32)
                self.embedding_texts = [r.input for r in records]
            except Exception as e:
                log_event(self.logger, "warning", "history_embedding_failed", error_type=type(e).__name__)

        restored = self.progress.load()
        if restored and restored.output_path == self.cfg.output and restored.target_count == self.target_count:
            self.state.generated_count = restored.generated_count
            self.state.duplicate_count = restored.duplicate_count
            self.state.filtered_count = restored.filtered_count
            self.state.failed_count = restored.failed_count
            self.state.relabel_rejected_count = restored.relabel_rejected_count
            self.state.attempt_count = restored.attempt_count
            self.state.domain_counts.update(restored.domain_counts)
            self.state.page_type_counts.update(restored.page_type_counts)
            self.state.language_style_counts.update(restored.language_style_counts)
            self.state.brand_presence_counts.update(restored.brand_presence_counts)
            self.state.title_length_counts.update(restored.title_length_counts)
            self.state.language_mix_counts.update(restored.language_mix_counts)
            self.state.contrast_pair_count = restored.contrast_pair_count
            self.state.status = "resumed"
        else:
            self.state.status = "ready"

        log_event(
            self.logger,
            "info",
            "history_load_done",
            loaded=stats["loaded"],
            deduped_existing_count=self.state.deduped_existing_count,
            dedup_removed=stats["dedup_removed"],
            invalid=stats["invalid"],
        )

    def _remaining_target(self) -> int:
        # Pending candidates reserve capacity but are not confirmed output rows.
        return max(0, self.target_count - self.state.total_effective_count - len(self._pending_flush))

    def _is_done(self) -> bool:
        return self.state.total_effective_count >= self.target_count

    async def _generate_one(self, spec: Dict[str, str]) -> Optional[Record]:
        prompt = self._build_prompt(spec)
        try:
            raw, model_name = await asyncio.wait_for(
                self.pool.generate(prompt, self.temperature, self.max_tokens),
                timeout=self.cfg.retry.task_timeout,
            )
        except asyncio.TimeoutError:
            log_event(self.logger, "warning", "task_timeout", spec=spec)
            return None
        except Exception as e:
            log_event(self.logger, "warning", "generate_failed", spec=spec, error_type=type(e).__name__)
            return None

        obj = safe_json_extract(raw)
        if not isinstance(obj, dict):
            log_event(self.logger, "warning", "parse_failed", response_length=len(raw))
            return None

        text = normalize_text(obj.get("input") or obj.get("title") or obj.get("text") or "")
        label = normalize_label(obj.get("label", spec["label"]))
        if label != spec["label"]:
            log_event(
                self.logger,
                "warning",
                "label_mismatch",
                expected=spec["label"],
                received_label_length=len(label),
                text_length=len(text),
            )
            return None

        try:
            confidence = _validated_confidence(obj.get("confidence", 1.0))
        except InvalidConfidenceError:
            log_event(self.logger, "warning", "invalid_confidence", stage="generation")
            return None

        if confidence < self.min_confidence:
            log_event(self.logger, "warning", "confidence_too_low", confidence=confidence, threshold=self.min_confidence)
            return None

        return Record(
            entry_id=compute_entry_id(text, label),
            input=text,
            label=label,
            confidence=confidence,
            model=model_name,
        )

    async def _relabel_check(self, record: Record) -> bool:
        if not self.enable_relabel_check:
            return True
        prompt = self._build_relabel_prompt(record.input)
        try:
            raw, model_name = await asyncio.wait_for(
                self.pool.generate(prompt, min(0.2, self.temperature), 80),
                timeout=self.cfg.retry.task_timeout,
            )
        except Exception as e:
            log_event(self.logger, "warning", "relabel_failed",
                      text_length=len(record.input), error_type=type(e).__name__)
            return False

        obj = safe_json_extract(raw)
        if not isinstance(obj, dict):
            log_event(self.logger, "warning", "relabel_parse_failed", response_length=len(raw))
            return False

        relabel = normalize_label(obj.get("label", ""))
        try:
            confidence = _validated_confidence(obj.get("confidence", 1.0))
        except InvalidConfidenceError:
            log_event(self.logger, "warning", "invalid_confidence", stage="relabel")
            return False
        if relabel != record.label or confidence < max(0.7, self.min_confidence):
            log_event(
                self.logger,
                "warning",
                "relabel_rejected",
                original_label=record.label,
                received_label_length=len(relabel),
                confidence=confidence,
                model=model_name,
                text_length=len(record.input),
            )
            return False
        return True

    async def _process_spec(self, spec: Dict[str, str]) -> bool:
        if self._persistence_error is not None:
            raise self._persistence_error
        if self._stop_requested:
            return False
        record = await self._generate_one(spec)
        async with self._state_lock:
            self.state.attempt_count += 1
        if self._persistence_error is not None:
            raise self._persistence_error
        if self._stop_requested:
            return False

        if record is None:
            async with self._state_lock:
                self.state.failed_count += 1
            return False

        if not self.quality_filter(record.input, record.label):
            async with self._state_lock:
                self.state.filtered_count += 1
            return False

        uniq = await self.semantic_deduplicate(record.entry_id, record.input)
        if not uniq:
            async with self._state_lock:
                self.state.duplicate_count += 1
            return False

        # Deduplication may have yielded while another worker failed to persist.
        # Do not start a new relabel request after that failure.
        if self._persistence_error is not None:
            raise self._persistence_error
        if self._stop_requested:
            return False
        if not await self._relabel_check(record):
            async with self._state_lock:
                self.state.relabel_rejected_count += 1
            return False

        await self._append_record(record, spec)
        return True

    async def _append_record(self, record: Record, spec: Dict[str, str]):
        async with self._writer_lock:
            if self._persistence_error is not None:
                raise self._persistence_error
            if self._stop_requested or self._remaining_target() == 0:
                return
            if record.entry_id in self._pending_flush_ids:
                self.state.duplicate_count += 1
                return

            self._pending_flush.append(record)
            self._pending_flush_ids.add(record.entry_id)
            self.state.generated_count += 1
            self.state.label_counts[record.label] = self.state.label_counts.get(record.label, 0) + 1
            self.state.domain_counts[spec["domain"]] = self.state.domain_counts.get(spec["domain"], 0) + 1
            self.state.page_type_counts[spec["page_type"]] = self.state.page_type_counts.get(spec["page_type"], 0) + 1
            self.state.language_style_counts[spec["language_style"]] = self.state.language_style_counts.get(spec["language_style"], 0) + 1
            self.state.brand_presence_counts[spec["brand_presence"]] = self.state.brand_presence_counts.get(spec["brand_presence"], 0) + 1
            self.state.title_length_counts[spec["title_length"]] = self.state.title_length_counts.get(spec["title_length"], 0) + 1
            self.state.language_mix_counts[spec["language_mix"]] = self.state.language_mix_counts.get(spec["language_mix"], 0) + 1
            if spec.get("contrast_group_id"):
                self.state.contrast_pair_count += 1

            if len(self._pending_flush) >= self.flush_every or self._remaining_target() == 0:
                self._flush_pending_sync()
                await self._save_progress()

            if self.state.total_effective_count - self._last_progress_log >= max(200, self.flush_every):
                self._last_progress_log = self.state.total_effective_count
                self._log_progress()

    def _flush_pending_sync(self):
        if self._persistence_error is not None:
            raise self._persistence_error
        if not self._pending_flush:
            return
        records = list(self._pending_flush)
        try:
            self.store.append_records(records)
        except Exception as exc:
            self.state.failed_count += len(self._pending_flush)
            error = self._record_persistence_failure("output", exc)
            raise error
        self.state.accepted_new_count += len(records)
        self._pending_flush.clear()
        self._pending_flush_ids.clear()
        log_event(self.logger, "info", "records_flushed", count=len(records), output=self.cfg.output)

    def _record_persistence_failure(self, stage: str, exc: Exception) -> CollectorPersistenceError:
        self._stop_requested = True
        if self._persistence_error is None:
            self._persistence_error = CollectorPersistenceError(stage)
            self._persistence_error.__cause__ = exc
            log_event(self.logger, "error", "persistence_failed", stage=stage,
                      error_type=type(exc).__name__, pending_count=len(self._pending_flush))
        return self._persistence_error

    async def _save_progress(self):
        try:
            await self.progress.save(self.state)
        except Exception as exc:
            error = self._record_persistence_failure("progress", exc)
            raise error

    async def _run_dispatch_loop(self):
        max_attempts = self.target_count * max(1, self.cfg.max_attempt_factor)
        in_flight: set = set()

        async def consume(done):
            for task in done:
                try:
                    await task
                except CollectorPersistenceError:
                    raise
                except Exception as exc:
                    async with self._state_lock:
                        self.state.failed_count += 1
                    log_event(self.logger, "error", "worker_crashed", error_type=type(exc).__name__)
                finally:
                    # Keep ownership until the result, including failure, is read.
                    in_flight.discard(task)

        try:
            while not self._is_done() and self.state.attempt_count < max_attempts and not self._stop_requested:
                remaining_target = self._remaining_target()
                available_slots = max(0, self.max_workers - len(in_flight))
                if available_slots == 0:
                    done, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                    await consume(done)
                    continue

                to_submit = min(self.batch_size, available_slots, remaining_target)
                if to_submit <= 0:
                    break

                for _ in range(to_submit):
                    spec = self._build_one_spec()
                    in_flight.add(asyncio.create_task(self._process_spec(spec)))

                log_event(self.logger, "info", "tasks_submitted", batch=to_submit,
                          in_flight=len(in_flight), remaining_target=self._remaining_target())
                if in_flight:
                    done, _ = await asyncio.wait(in_flight, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
                    await consume(done)

            while in_flight:
                done, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                await consume(done)
            if self._persistence_error is not None:
                raise self._persistence_error
        finally:
            self._stop_requested = True
            for task in in_flight:
                if not task.done():
                    task.cancel()
            if in_flight:
                await asyncio.gather(*in_flight, return_exceptions=True)
            in_flight.clear()

    async def collect(self):
        primary_error = None
        try:
            if self._persistence_error is not None:
                raise self._persistence_error
            if self._stop_requested:
                raise RuntimeError("Collector has stopped; create a new instance to resume.")
            await self._load_existing_state()
            await self._save_progress()
            if not self._is_done():
                log_event(self.logger, "info", "collector_start", target_count=self.target_count,
                          existing_count=self.state.existing_count, remaining_target=self._remaining_target(),
                          max_workers=self.max_workers, output=self.cfg.output)
                self.state.status = "running"
                await self._run_dispatch_loop()
            else:
                log_event(self.logger, "info", "target_already_satisfied",
                          target_count=self.target_count, existing_count=self.state.existing_count)
            # Flush only after a successful run, never during error cleanup:
            # an unsuccessful append may already have written part of this batch.
            async with self._writer_lock:
                self._flush_pending_sync()
        except BaseException as exc:
            primary_error = exc
        finally:
            self._stop_requested = True
            try:
                await self.pool.close()
            except BaseException as exc:
                if primary_error is None:
                    primary_error = exc
                else:
                    log_event(self.logger, "error", "collector_cleanup_failed",
                              error_type=type(exc).__name__)

        if primary_error is None:
            self.state.status = "completed" if self._is_done() and not self._pending_flush else "partial"
            try:
                await self._save_progress()
            except BaseException as exc:
                primary_error = exc

        if primary_error is not None:
            self.state.status = (
                "interrupted" if isinstance(primary_error, (KeyboardInterrupt, asyncio.CancelledError)) else "failed"
            )
            log_event(self.logger, "error", "collector_failed", status=self.state.status,
                      error_type=type(primary_error).__name__)
            # Sidecar persistence is best effort when the storage itself is failing.
            # Its error must not hide the original failure, or retry output writes.
            try:
                await self.progress.save(self.state)
            except BaseException as exc:
                log_event(self.logger, "error", "failure_progress_save_failed",
                          error_type=type(exc).__name__)
            raise primary_error

        self._log_summary(final=True)

    def _log_progress(self):
        elapsed = max(1e-6, time.time() - self.start_ts)
        speed = self.state.accepted_new_count / elapsed
        dedup_rate = self.state.duplicate_count / max(1, self.state.attempt_count)
        filter_rate = self.state.filtered_count / max(1, self.state.attempt_count)
        log_event(
            self.logger,
            "info",
            "progress",
            target_count=self.target_count,
            existing_count=self.state.existing_count,
            accepted_new_count=self.state.accepted_new_count,
            total_effective_count=self.state.total_effective_count,
            remaining_target=self._remaining_target(),
            attempts=self.state.attempt_count,
            speed=round(speed, 2),
            dedup_rate=round(dedup_rate, 4),
            filter_rate=round(filter_rate, 4),
            relabel_rejected_count=self.state.relabel_rejected_count,
            label_dist=self.state.label_counts,
        )
        for name, st in self.pool.stats.items():
            log_event(
                self.logger,
                "info",
                "model_stats",
                model=name,
                calls=st.total_calls,
                succ_rate=round(st.success_rate(), 4),
                avg_latency=round(st.avg_latency(), 3),
                p95_latency=round(st.p95_latency(), 3),
            )

    def _log_summary(self, final: bool = False):
        elapsed = max(1e-6, time.time() - self.start_ts)
        log_event(
            self.logger,
            "info",
            "final_summary" if final else "summary",
            target_count=self.target_count,
            original_count=self.initial_existing_count,
            added_count=self.state.accepted_new_count,
            deduped_total_count=self.state.total_effective_count,
            failed_count=self.state.failed_count,
            duplicate_count=self.state.duplicate_count,
            filtered_count=self.state.filtered_count,
            relabel_rejected_count=self.state.relabel_rejected_count,
            attempt_count=self.state.attempt_count,
            elapsed_seconds=round(elapsed, 3),
            status=self.state.status,
        )


# -----------------------------
# CLI
# -----------------------------
def parse_model_configs(args) -> List[ModelConfig]:
    if args.model_config:
        raw = args.model_config.strip()
        if (raw.endswith(".json") or os.path.sep in raw) and os.path.isfile(raw):
            try:
                with open(raw, "r", encoding="utf-8") as f:
                    raw = f.read()
                print(f"已从文件加载配置: {args.model_config}")
            except Exception as e:
                raise ValueError(f"无法读取配置文件 {args.model_config}: {e}")

        if raw.startswith("\ufeff"):
            raw = raw[1:]

        try:
            config_list = json.loads(raw)
        except json.JSONDecodeError as e:
            preview = raw[:100] if len(raw) > 0 else "<空>"
            raise ValueError(f"JSON 解析失败: {e}\n内容预览: {preview}...")

        if not isinstance(config_list, list):
            raise ValueError(f"配置必须是数组，得到: {type(config_list).__name__}")

        cfgs = []
        for i, m in enumerate(config_list):
            if not isinstance(m, dict):
                raise ValueError(f"第 {i + 1} 项必须是字典")
            required = ["name", "provider"]
            missing = [f for f in required if f not in m]
            if missing:
                raise ValueError(f"第 {i + 1} 项缺少必填字段: {missing}")

            cfgs.append(
                ModelConfig(
                    name=m["name"],
                    provider=m["provider"],
                    api_key=m.get("api_key"),
                    base_url=m.get("base_url"),
                    concurrency=int(m.get("concurrency", 3)),
                    weight=float(m.get("weight", 1.0)),
                    timeout=float(m.get("timeout", 30)),
                    max_retries=int(m.get("max_retries", 3)),
                    qps_limit=float(m.get("qps_limit", 5)),
                )
            )
        return cfgs

    return [
        ModelConfig(
            name=args.model,
            provider=args.provider,
            api_key=args.api_key,
            base_url=args.base_url,
            concurrency=args.concurrency,
            weight=1.0,
            timeout=args.timeout,
            max_retries=args.max_retries,
            qps_limit=args.qps_limit,
        )
    ]


def build_arg_parser():
    p = argparse.ArgumentParser(description="Multi-LLM classifier training data collector")
    p.add_argument("--output", type=str, required=True, help="输出文件路径，建议使用 .json 以兼容 classifier_train.py")
    p.add_argument("--target_count", type=int, default=100000, help="目标总条数（最终总量）")
    p.add_argument("--total_samples", type=int, default=None, help="兼容旧参数，等价于 target_count")

    p.add_argument("--categories", type=str, default="News,Tools,Learning,Shopping,Social,Entertainment,Other")
    p.add_argument("--distribution", type=str, default="0.143,0.143,0.143,0.143,0.143,0.143,0.142")

    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max_tokens", type=int, default=220)
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--max_workers", type=int, default=30, help="并发 worker 数")
    p.add_argument("--flush_every", type=int, default=200, help="累计多少条后安全落盘")
    p.add_argument("--progress_path", type=str, default=None, help="断点续跑进度文件路径")
    p.add_argument("--max_attempt_factor", type=int, default=40, help="最大尝试次数系数")
    p.add_argument("--output_format", type=str, default=None, help="可显式指定 json/jsonl/csv，默认按输出文件后缀判断")

    p.add_argument("--domains", type=str, default="资讯门户,电商平台,学习平台,社交社区,视频娱乐,生产力工具,企业办公,论坛博客,下载资源,搜索导航")
    p.add_argument("--domain_distribution", type=str, default=None)
    p.add_argument("--page_types", type=str, default="首页,详情页,列表页,搜索结果页,文档页,登录页,帖子页,播放页,商品页,设置页")
    p.add_argument("--page_type_distribution", type=str, default=None)
    p.add_argument("--language_styles", type=str, default="简洁标题,营销风格,教程风格,社区风格,官方风格")
    p.add_argument("--language_style_distribution", type=str, default=None)
    p.add_argument("--brand_presence", type=str, default="含品牌名,不含品牌名")
    p.add_argument("--brand_presence_distribution", type=str, default="0.55,0.45")
    p.add_argument("--title_length_types", type=str, default="短标题,中标题,长标题")
    p.add_argument("--title_length_distribution", type=str, default="0.25,0.5,0.25")
    p.add_argument("--include_chinese_ratio", type=float, default=0.7)
    p.add_argument("--include_english_ratio", type=float, default=0.2)
    p.add_argument("--include_mixed_ratio", type=float, default=0.1)
    p.add_argument("--enable_contrast_pairs", action="store_true", help="启用相近场景下不同类别对照样本生成")
    p.add_argument("--contrast_pair_ratio", type=float, default=0.25)

    p.add_argument("--enable_relabel_check", action="store_true", help="生成后再做一次复核分类")
    p.add_argument("--min_confidence", type=float, default=0.0, help="最低接受置信度")
    p.add_argument("--enable_semantic_dedup", action="store_true", help="启用向量语义去重")
    p.add_argument("--similarity_threshold", type=float, default=0.92)

    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--provider", type=str, default="openai")
    p.add_argument("--api_key", type=str, default=None)
    p.add_argument("--base_url", type=str, default=None)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--qps_limit", type=float, default=5.0)
    p.add_argument("--model-config", type=str, default=None, help="JSON list for multi-model configs")

    p.add_argument("--request_timeout", type=float, default=30.0)
    p.add_argument("--task_timeout", type=float, default=60.0)
    p.add_argument("--backoff_base", type=float, default=0.5)
    p.add_argument("--backoff_max", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", type=str, default="INFO")
    return p


def _parse_csv_list(raw: Optional[str], cast=str) -> Optional[List[Any]]:
    if raw is None:
        return None
    parts = [x.strip() for x in str(raw).split(",") if x.strip()]
    return [cast(x) for x in parts]


async def async_main():
    parser = build_arg_parser()
    args = parser.parse_args()

    target_count = args.target_count if args.total_samples is None else args.total_samples
    if target_count <= 0:
        raise ValueError("target_count 必须大于 0")

    categories = [normalize_label(x) for x in args.categories.split(",") if x.strip()]
    distribution = [float(x.strip()) for x in args.distribution.split(",") if x.strip()]
    if len(categories) != len(distribution):
        raise ValueError("categories 与 distribution 长度必须一致")
    if sum(distribution) <= 0:
        raise ValueError("distribution 总和必须大于 0")
    if abs(sum(distribution) - 1.0) > 1e-3:
        total = sum(distribution)
        distribution = [x / total for x in distribution]

    domains = _parse_csv_list(args.domains) or []
    page_types = _parse_csv_list(args.page_types) or []
    language_styles = _parse_csv_list(args.language_styles) or []
    brand_presence = _parse_csv_list(args.brand_presence) or []
    title_length_types = _parse_csv_list(args.title_length_types) or []

    def norm_ratio_list(name: str, value: Optional[str], expected_len: int) -> Optional[List[float]]:
        if value is None:
            return None
        ratios = _parse_csv_list(value, float)
        if ratios is None:
            return None
        if len(ratios) != expected_len:
            raise ValueError(f"{name} 长度必须与对应类别数量一致")
        total = sum(ratios)
        if total <= 0:
            raise ValueError(f"{name} 总和必须大于 0")
        return [x / total for x in ratios]

    model_configs = parse_model_configs(args)
    retry_cfg = RetryConfig(
        max_retries=args.max_retries,
        backoff_base=args.backoff_base,
        backoff_max=args.backoff_max,
        request_timeout=args.request_timeout,
        task_timeout=args.task_timeout,
    )
    runtime_cfg = RuntimeConfig(
        output=args.output,
        target_count=target_count,
        categories=categories,
        distribution=distribution,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        similarity_threshold=args.similarity_threshold,
        random_seed=args.seed,
        log_level=args.log_level,
        enable_semantic_dedup=args.enable_semantic_dedup,
        progress_path=args.progress_path,
        flush_every=args.flush_every,
        max_attempt_factor=args.max_attempt_factor,
        output_format=(args.output_format.lower() if args.output_format else None),
        enable_relabel_check=args.enable_relabel_check,
        min_confidence=args.min_confidence,
        domains=domains,
        domain_distribution=norm_ratio_list("domain_distribution", args.domain_distribution, len(domains)),
        page_types=page_types,
        page_type_distribution=norm_ratio_list("page_type_distribution", args.page_type_distribution, len(page_types)),
        language_styles=language_styles,
        language_style_distribution=norm_ratio_list("language_style_distribution", args.language_style_distribution, len(language_styles)),
        brand_presence=brand_presence,
        brand_presence_distribution=norm_ratio_list("brand_presence_distribution", args.brand_presence_distribution, len(brand_presence)),
        title_length_types=title_length_types,
        title_length_distribution=norm_ratio_list("title_length_distribution", args.title_length_distribution, len(title_length_types)),
        include_chinese_ratio=args.include_chinese_ratio,
        include_english_ratio=args.include_english_ratio,
        include_mixed_ratio=args.include_mixed_ratio,
        enable_contrast_pairs=args.enable_contrast_pairs,
        contrast_pair_ratio=args.contrast_pair_ratio,
        retry=retry_cfg,
    )

    if abs(runtime_cfg.include_chinese_ratio + runtime_cfg.include_english_ratio + runtime_cfg.include_mixed_ratio - 1.0) > 1e-3:
        total = runtime_cfg.include_chinese_ratio + runtime_cfg.include_english_ratio + runtime_cfg.include_mixed_ratio
        if total <= 0:
            raise ValueError("语言比例总和必须大于 0")
        runtime_cfg.include_chinese_ratio /= total
        runtime_cfg.include_english_ratio /= total
        runtime_cfg.include_mixed_ratio /= total

    collector = ClassifierDataCollector(runtime_cfg=runtime_cfg, model_configs=model_configs)
    await collector.collect()


def cli_main() -> int:
    try:
        asyncio.run(async_main())
    except (KeyboardInterrupt, asyncio.CancelledError) as exc:
        print(json.dumps({"success": False, "error": "Collector interrupted; inspect output before resuming.",
                          "error_type": type(exc).__name__}), file=sys.stderr)
        return 130
    except Exception as exc:
        print(json.dumps({"success": False, "error": "Collector failed; inspect output before resuming.",
                          "error_type": type(exc).__name__}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
