#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sentiment_data_collector.py

基于多模型并发蒸馏的六分类中文情感数据生成器（平静、开心、伤心、生气、惊讶、厌恶）

本次重构重点增强：
1) 历史数据加载与去重：启动时自动读取已有数据，按业务唯一键 entry_id 去重
2) 断点续跑：进度持久化到 sidecar progress 文件，异常退出后可继续补齐差额
3) 精确目标控制：通过 target_count 精确补齐到目标总量
4) 并发增强：异步任务队列 + worker + 单写入器，保证线程安全与幂等性
5) 稳定性增强：重试、超时、异常分级、中间结果安全落盘、结构化日志

唯一条目判定规则：
- entry_id = sha1(f"{label}\t{normalize_text(text)}")
- 若历史文件内已存在 id 字段则优先使用；否则自动回填该规则生成的 entry_id
- 该规则为“业务唯一键”，不是全文模糊相似判断

兼容输出格式：
- CSV（默认，兼容当前项目 text,label 结构；会新增 entry_id 列）
- JSONL
- JSON（数组）

依赖建议：
pip install httpx numpy
# 可选：
pip install sentence-transformers torch
pip install transformers

示例：
python sentiment_data_collector.py \
  --output ./sentiment_train.csv \
  --target_count 10000 \
  --progress_path ./sentiment_train.progress.json \
  --model-config '[{"name":"gpt-4","provider":"openai","concurrency":3,"weight":0.4},{"name":"claude-3-opus","provider":"anthropic","concurrency":3,"weight":0.4},{"name":"qwen-72b","provider":"local","base_url":"http://localhost:8000/v1","concurrency":4,"weight":0.2}]' \
  --temperature 0.8 \
  --max_workers 20
"""

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
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
import numpy as np


# -----------------------------
# Utilities / Logging
# -----------------------------
def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("SentimentDataCollector")
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


class BloomFilter:
    """简单 BloomFilter 实现（无外部依赖）"""

    def __init__(self, capacity: int = 200_000, error_rate: float = 0.01):
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


def safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def normalize_text(s: str) -> str:
    s = s.strip().replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.strip("“”\"'")
    return s


def compute_entry_id(text: str, label: str) -> str:
    canonical = f"{label.strip()}\t{normalize_text(text)}"
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
    task_timeout: float = 45.0


@dataclass
class RuntimeConfig:
    output: str
    target_count: int
    categories: List[str]
    distribution: List[float]
    temperature: float = 0.8
    max_tokens: int = 120
    batch_size: int = 20
    max_workers: int = 20
    similarity_threshold: float = 0.85
    random_seed: int = 42
    log_level: str = "INFO"
    enable_ppl: bool = False
    progress_path: Optional[str] = None
    flush_every: int = 20
    max_attempt_factor: int = 30
    retry: RetryConfig = field(default_factory=RetryConfig)


@dataclass
class ModelConfig:
    name: str
    provider: str  # openai / anthropic / local
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
    attempt_count: int = 0
    label_counts: Dict[str, int] = field(default_factory=dict)
    style_counts: Dict[str, int] = field(default_factory=dict)
    length_counts: Dict[str, int] = field(default_factory=dict)
    scene_counts: Dict[str, int] = field(default_factory=dict)
    last_update_ts: float = 0.0
    status: str = "initialized"

    @property
    def total_effective_count(self) -> int:
        return self.existing_count + self.accepted_new_count


@dataclass
class Record:
    entry_id: str
    text: str
    label: str
    model: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "text": self.text,
            "label": self.label,
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

    @staticmethod
    def atomic_write_bytes(path: str, content: bytes):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path_obj.parent)) as tf:
            tf.write(content)
            temp_path = tf.name
        os.replace(temp_path, path)


class ProgressTracker:
    """职责：持久化执行进度，支持断点续跑与恢复。"""

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
            log_event(
                self.logger,
                "warning",
                "progress_load_failed",
                progress_path=self.progress_path,
                reason=str(e),
            )
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
    """职责：加载历史数据、按业务唯一键去重、幂等写入、兼容 CSV/JSONL/JSON。"""

    def __init__(self, output_path: str, logger: logging.Logger):
        self.output_path = output_path
        self.logger = logger
        self.output_format = self._detect_format(output_path)

    @staticmethod
    def _detect_format(path: str) -> str:
        suffix = Path(path).suffix.lower()
        if suffix == ".jsonl":
            return "jsonl"
        if suffix == ".json":
            return "json"
        return "csv"

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
                text = normalize_text(str(row.get("text", "")))
                label = str(row.get("label", "")).strip()
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
                        text=text,
                        label=label,
                        model=str(row.get("model", "")),
                        created_at=float(row.get("created_at", time.time())),
                    )
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
        """
        幂等追加：调用方必须保证传入 records 已按 entry_id 去重且不与历史冲突。
        CSV/JSONL 支持追加；JSON 采用全量重写避免结构损坏。
        """
        if not records:
            return
        try:
            if self.output_format == "csv":
                file_exists = os.path.exists(self.output_path)
                Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.output_path, "a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=["entry_id", "text", "label", "model", "created_at"]
                    )
                    if not file_exists or os.path.getsize(self.output_path) == 0:
                        writer.writeheader()
                    for r in records:
                        writer.writerow(r.to_dict())
            elif self.output_format == "jsonl":
                Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.output_path, "a", encoding="utf-8") as f:
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

    def _read_csv(self, path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _read_jsonl(self, path: str) -> List[Dict[str, Any]]:
        rows = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def _read_json(self, path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON 文件内容必须为数组")
        return data

    def _write_csv(self, records: List[Record]):
        rows = [r.to_dict() for r in records]
        out = []
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", newline="", dir=str(Path(self.output_path).parent or Path("."))) as tf:
            writer = csv.DictWriter(
                tf, fieldnames=["entry_id", "text", "label", "model", "created_at"]
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            temp_path = tf.name
        os.replace(temp_path, self.output_path)
        out.clear()

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
        self.http = httpx.AsyncClient(timeout=cfg.timeout)
        self.last_request_ts = 0.0
        self.min_interval = 1.0 / max(0.1, cfg.qps_limit)
        self._rate_limit_lock = asyncio.Lock()

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
        await self.http.aclose()


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
                {"role": "system", "content": "你是情感文本数据生成助手。输出必须严格JSON。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = await self.http.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


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
        r = await self.http.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        if "content" in data and data["content"]:
            parts = data["content"]
            if isinstance(parts, list):
                return "".join([p.get("text", "") for p in parts])
        return json.dumps(data, ensure_ascii=False)


class LocalOpenAICompatibleClient(OpenAIClient):
    pass


class ModelPool:
    """职责：多模型健康检查、限流、熔断、失败重试与故障转移。"""

    def __init__(self, model_configs: List[ModelConfig], logger: logging.Logger):
        self.logger = logger
        self.cfgs: Dict[str, ModelConfig] = {c.name: c for c in model_configs}
        self.clients: Dict[str, BaseModelClient] = {}
        self.semaphores: Dict[str, asyncio.Semaphore] = {}
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.stats: Dict[str, ModelStats] = {}
        self._init_clients()

    def _init_clients(self):
        for cfg in self.cfgs.values():
            if cfg.provider.lower() == "openai":
                client = OpenAIClient(cfg)
            elif cfg.provider.lower() == "anthropic":
                client = AnthropicClient(cfg)
            elif cfg.provider.lower() == "local":
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
        av = self._available_models()
        if not av:
            return None
        weights = [max(0.0, c.weight) for c in av]
        return random.choice(av) if sum(weights) == 0 else random.choices(av, weights=weights, k=1)[0]

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
                        reason=str(e),
                        backoff=round(backoff, 3),
                    )
                    await asyncio.sleep(backoff)
                except Exception as e:
                    last_err = e
                    st.failed_calls += 1
                    self.breakers[name].on_failure()
                    await asyncio.sleep(0.2)

        raise RuntimeError(f"All models failed. last_error={last_err}")

    async def close(self):
        for client in self.clients.values():
            await client.close()


# -----------------------------
# Core Collector
# -----------------------------
class SentimentDataCollector:
    """
    核心职责拆分：
    - 历史数据加载与去重
    - 配额目标计算
    - 任务生产与并发执行
    - 结果去重 / 质量过滤 / 安全写入
    - 进度恢复与中间结果落盘
    """

    def __init__(self, runtime_cfg: RuntimeConfig, model_configs: List[ModelConfig]):
        self.cfg = runtime_cfg
        self.logger = setup_logger(runtime_cfg.log_level)
        random.seed(runtime_cfg.random_seed)
        np.random.seed(runtime_cfg.random_seed)

        self.pool = ModelPool(model_configs, self.logger)
        self.store = DataStore(runtime_cfg.output, self.logger)
        progress_path = runtime_cfg.progress_path or f"{runtime_cfg.output}.progress.json"
        self.progress = ProgressTracker(progress_path, self.logger)

        self.categories = runtime_cfg.categories
        self.distribution = runtime_cfg.distribution
        self.target_count = runtime_cfg.target_count
        self.temperature = runtime_cfg.temperature
        self.max_tokens = runtime_cfg.max_tokens
        self.batch_size = runtime_cfg.batch_size
        self.max_workers = runtime_cfg.max_workers
        self.similarity_threshold = runtime_cfg.similarity_threshold
        self.enable_ppl = runtime_cfg.enable_ppl
        self.flush_every = max(1, runtime_cfg.flush_every)

        self.scenes = ["工作场景", "日常生活", "社交互动", "健康状况", "学习考试"]
        self.sentence_types = {"陈述句": 0.4, "疑问句": 0.2, "感叹句": 0.3, "祈使句": 0.1}
        self.length_types = {"短文本": (5, 15, 0.3), "中文本": (16, 30, 0.4), "长文本": (31, 50, 0.3)}
        self.sensitive_words = {"暴恐", "涉黄", "政治极端", "仇恨言论"}

        self.exact_set = set()
        self.bloom = BloomFilter(capacity=max(10000, runtime_cfg.target_count * 3), error_rate=0.005)
        self.id_set = set()

        self.embed_model = None
        self.embeddings = None
        self.embedding_texts: List[str] = []
        self._embed_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._writer_lock = asyncio.Lock()
        self._pending_flush: List[Record] = []
        self._last_progress_log = 0
        self._stop_requested = False

        self.ppl_model = None
        self.ppl_tokenizer = None
        self._try_load_sbert()
        if self.enable_ppl:
            self._try_load_ppl_model()

        self.category_targets = self._build_targets(self.categories, self.distribution, self.target_count)
        self.style_targets = self._build_targets(
            list(self.sentence_types.keys()),
            list(self.sentence_types.values()),
            self.target_count,
        )
        self.length_targets = self._build_targets(
            list(self.length_types.keys()),
            [self.length_types[k][2] for k in self.length_types],
            self.target_count,
        )
        self.scene_targets = self._build_targets(
            self.scenes, [1 / len(self.scenes)] * len(self.scenes), self.target_count
        )

        self.state = ProgressState(
            output_path=self.cfg.output,
            target_count=self.target_count,
            label_counts={c: 0 for c in self.categories},
            style_counts={k: 0 for k in self.style_targets},
            length_counts={k: 0 for k in self.length_targets},
            scene_counts={k: 0 for k in self.scene_targets},
            status="initializing",
        )
        self.start_ts = time.time()
        self.initial_existing_count = 0

    def _try_load_sbert(self):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.embed_model = SentenceTransformer("shibing624/text2vec-base-chinese")
            log_event(self.logger, "info", "sbert_loaded", model="shibing624/text2vec-base-chinese")
        except Exception as e:
            self.embed_model = None
            log_event(self.logger, "warning", "sbert_unavailable", reason=str(e))

    def _try_load_ppl_model(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            import torch  # type: ignore

            name = "uer/gpt2-chinese-cluecorpussmall"
            self.ppl_tokenizer = AutoTokenizer.from_pretrained(name)
            self.ppl_model = AutoModelForCausalLM.from_pretrained(name)
            self.ppl_model.eval()
            self._torch = torch
            log_event(self.logger, "info", "ppl_model_loaded", model=name)
        except Exception as e:
            self.ppl_model = None
            self.ppl_tokenizer = None
            log_event(self.logger, "warning", "ppl_model_unavailable", reason=str(e))

    @staticmethod
    def _build_targets(keys: List[str], ratios: List[float], total: int) -> Dict[str, int]:
        ratios_arr = np.array(ratios, dtype=float)
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

    def _build_prompt(self, category: str, scene: str, sentence_type: str, length_type: str) -> str:
        lo, hi, _ = self.length_types[length_type]
        few_shot = [
            {"text": "这个项目终于完成了，感觉松了一口气。", "label": "平静"},
            {"text": "哇！我被心仪的大学录取了！", "label": "开心"},
            {"text": "最近总是失眠，心情很低落。", "label": "伤心"},
            {"text": "这种服务态度实在太差了！", "label": "生气"},
            {"text": "天哪，这怎么可能发生？", "label": "惊讶"},
            {"text": "看到这一幕我简直想吐。", "label": "厌恶"},
        ]
        prompt = {
            "task": "生成一条中文情感文本，用于BERT六分类训练。",
            "constraints": {
                "target_label": category,
                "scene": scene,
                "sentence_type": sentence_type,
                "length_chars_range": [lo, hi],
                "naturalness": "口语化、自然、不生硬",
                "forbidden": ["敏感内容", "暴恐涉黄", "明显乱码", "重复模板句"],
                "output_format": '严格输出JSON: {"text":"...","label":"..."}，不要输出额外解释',
            },
            "few_shot": few_shot,
        }
        return json.dumps(prompt, ensure_ascii=False)

    def _heuristic_ppl(self, text: str) -> float:
        if len(text) < 5:
            return 1000.0
        uniq = len(set(text))
        rep_ratio = 1 - uniq / len(text)
        punct = len(re.findall(r"[，。！？；,.!?]", text))
        return 20 + rep_ratio * 200 - min(10, punct) * 0.5

    def _model_ppl(self, text: str) -> float:
        if self.ppl_model is None or self.ppl_tokenizer is None:
            return self._heuristic_ppl(text)
        try:
            import torch  # type: ignore

            enc = self.ppl_tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                out = self.ppl_model(**enc, labels=enc["input_ids"])
                loss = out.loss.item()
            return float(math.exp(min(20, loss)))
        except Exception:
            return self._heuristic_ppl(text)

    def quality_filter(self, text: str, label: str) -> bool:
        text = text.strip()
        if not text:
            return False
        if any(w in text for w in self.sensitive_words):
            return False
        if re.search(r"[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：、“”‘’\s,.!?-]", text):
            return False

        length = len(re.sub(r"\s+", "", text))
        if length < 5 or length > 50:
            return False

        ppl = self._model_ppl(text)
        if ppl > 150:
            return False

        conflict_map = {
            "开心": ["恶心", "愤怒", "难过"],
            "伤心": ["太棒", "开心", "激动"],
            "生气": ["放松", "淡定", "安心"],
            "惊讶": ["平静", "习以为常"],
            "厌恶": ["喜欢", "享受", "美好"],
            "平静": ["气死", "恶心", "崩溃"],
        }
        for c in conflict_map.get(label, []):
            if c in text:
                return False
        return True

    def _quick_duplicate(self, entry_id: str, text: str) -> bool:
        return entry_id in self.id_set or (text in self.bloom and text in self.exact_set)

    async def semantic_deduplicate(self, entry_id: str, text: str) -> bool:
        """True 表示保留；False 表示重复。"""
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

            self.embeddings = emb.reshape(1, -1) if self.embeddings is None else np.vstack(
                [self.embeddings, emb.reshape(1, -1)]
            )
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

        for r in records:
            self.id_set.add(r.entry_id)
            self.exact_set.add(r.text)
            self.bloom.add(r.text)
            if r.label in self.state.label_counts:
                self.state.label_counts[r.label] += 1

        if self.embed_model is not None and records:
            try:
                embs = self.embed_model.encode(
                    [r.text for r in records], normalize_embeddings=True, batch_size=64
                )
                self.embeddings = np.asarray(embs, dtype=np.float32)
                self.embedding_texts = [r.text for r in records]
            except Exception as e:
                log_event(self.logger, "warning", "history_embedding_failed", reason=str(e))

        restored = self.progress.load()
        if restored and restored.output_path == self.cfg.output and restored.target_count == self.target_count:
            self.state.accepted_new_count = restored.accepted_new_count
            self.state.generated_count = restored.generated_count
            self.state.duplicate_count = restored.duplicate_count
            self.state.filtered_count = restored.filtered_count
            self.state.failed_count = restored.failed_count
            self.state.attempt_count = restored.attempt_count
            self.state.style_counts.update(restored.style_counts)
            self.state.length_counts.update(restored.length_counts)
            self.state.scene_counts.update(restored.scene_counts)
            for k, v in restored.label_counts.items():
                if k in self.state.label_counts:
                    self.state.label_counts[k] = max(self.state.label_counts[k], v)
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
        return max(0, self.target_count - self.state.total_effective_count)

    def _is_done(self) -> bool:
        return self.state.total_effective_count >= self.target_count

    async def _generate_one(self, spec: Dict[str, str]) -> Optional[Record]:
        prompt = self._build_prompt(
            spec["label"], spec["scene"], spec["sentence_type"], spec["length_type"]
        )
        try:
            raw, model_name = await asyncio.wait_for(
                self.pool.generate(prompt, self.temperature, self.max_tokens),
                timeout=self.cfg.retry.task_timeout,
            )
        except asyncio.TimeoutError:
            log_event(self.logger, "warning", "task_timeout", spec=spec)
            return None
        except Exception as e:
            log_event(self.logger, "warning", "generate_failed", spec=spec, reason=str(e))
            return None

        obj = safe_json_extract(raw)
        if not obj or "text" not in obj:
            log_event(self.logger, "warning", "parse_failed", raw_preview=raw[:200])
            return None

        text = normalize_text(str(obj["text"]))
        label = str(obj.get("label", spec["label"])).strip()
        if label != spec["label"]:
            log_event(
                self.logger,
                "warning",
                "label_mismatch",
                expected=spec["label"],
                actual=label,
                text_preview=text[:50],
            )
            return None

        return Record(
            entry_id=compute_entry_id(text, label),
            text=text,
            label=label,
            model=model_name,
        )

    async def _process_spec(self, spec: Dict[str, str]) -> bool:
        """单任务处理。返回 True 表示最终新增成功，False 表示失败/重复/过滤。"""
        record = await self._generate_one(spec)
        async with self._state_lock:
            self.state.attempt_count += 1

        if record is None:
            async with self._state_lock:
                self.state.failed_count += 1
            return False

        if not self.quality_filter(record.text, record.label):
            async with self._state_lock:
                self.state.filtered_count += 1
            return False

        uniq = await self.semantic_deduplicate(record.entry_id, record.text)
        if not uniq:
            async with self._state_lock:
                self.state.duplicate_count += 1
            return False

        await self._append_record(record, spec)
        return True

    async def _append_record(self, record: Record, spec: Dict[str, str]):
        async with self._writer_lock:
            if self._is_done():
                return
            if record.entry_id in {r.entry_id for r in self._pending_flush}:
                self.state.duplicate_count += 1
                return

            self._pending_flush.append(record)
            self.state.accepted_new_count += 1
            self.state.generated_count += 1
            self.state.label_counts[record.label] = self.state.label_counts.get(record.label, 0) + 1
            self.state.style_counts[spec["sentence_type"]] = self.state.style_counts.get(spec["sentence_type"], 0) + 1
            self.state.length_counts[spec["length_type"]] = self.state.length_counts.get(spec["length_type"], 0) + 1
            self.state.scene_counts[spec["scene"]] = self.state.scene_counts.get(spec["scene"], 0) + 1

            if len(self._pending_flush) >= self.flush_every or self._is_done():
                self._flush_pending_sync()
                await self.progress.save(self.state)

            if self.state.total_effective_count - self._last_progress_log >= max(50, self.flush_every):
                self._last_progress_log = self.state.total_effective_count
                self._log_progress()

    def _flush_pending_sync(self):
        if not self._pending_flush:
            return
        try:
            records = list(self._pending_flush)
            self.store.append_records(records)
            self._pending_flush.clear()
            log_event(self.logger, "info", "records_flushed", count=len(records), output=self.cfg.output)
        except Exception as e:
            self.state.failed_count += len(self._pending_flush)
            log_event(self.logger, "error", "flush_failed", reason=str(e), count=len(self._pending_flush))
            self._pending_flush.clear()

    def _build_one_spec(self) -> Dict[str, str]:
        label = self._choose_from_remaining(self.category_targets, self.state.label_counts)
        scene = self._choose_from_remaining(self.scene_targets, self.state.scene_counts)
        stype = self._choose_from_remaining(self.style_targets, self.state.style_counts)
        ltype = self._choose_from_remaining(self.length_targets, self.state.length_counts)
        return {"label": label, "scene": scene, "sentence_type": stype, "length_type": ltype}

    async def _run_dispatch_loop(self):
        max_attempts = self.target_count * max(1, self.cfg.max_attempt_factor)
        in_flight: set = set()

        while not self._is_done() and self.state.attempt_count < max_attempts and not self._stop_requested:
            remaining_target = self._remaining_target()
            available_slots = max(0, self.max_workers - len(in_flight))
            if available_slots == 0:
                done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    try:
                        await task
                    except Exception as e:
                        async with self._state_lock:
                            self.state.failed_count += 1
                        log_event(self.logger, "error", "worker_crashed", reason=str(e))
                continue

            to_submit = min(self.batch_size, available_slots, remaining_target)
            if to_submit <= 0:
                break

            for _ in range(to_submit):
                spec = self._build_one_spec()
                task = asyncio.create_task(self._process_spec(spec))
                in_flight.add(task)
                task.add_done_callback(lambda t, bag=in_flight: bag.discard(t))

            log_event(
                self.logger,
                "info",
                "tasks_submitted",
                batch=to_submit,
                in_flight=len(in_flight),
                remaining_target=self._remaining_target(),
            )

            if in_flight:
                done, _ = await asyncio.wait(in_flight, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    try:
                        await task
                    except Exception as e:
                        async with self._state_lock:
                            self.state.failed_count += 1
                        log_event(self.logger, "error", "worker_crashed", reason=str(e))

        if in_flight:
            results = await asyncio.gather(*list(in_flight), return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    async with self._state_lock:
                        self.state.failed_count += 1
                    log_event(self.logger, "error", "worker_crashed", reason=str(r))

    async def collect(self):
        await self._load_existing_state()
        await self.progress.save(self.state)

        if self._is_done():
            log_event(
                self.logger,
                "info",
                "target_already_satisfied",
                target_count=self.target_count,
                existing_count=self.state.existing_count,
            )
            self._log_summary(final=True)
            await self.pool.close()
            return

        log_event(
            self.logger,
            "info",
            "collector_start",
            target_count=self.target_count,
            existing_count=self.state.existing_count,
            remaining_target=self._remaining_target(),
            max_workers=self.max_workers,
            output=self.cfg.output,
        )

        try:
            self.state.status = "running"
            await self._run_dispatch_loop()
        except KeyboardInterrupt:
            self._stop_requested = True
            self.state.status = "interrupted"
            log_event(self.logger, "warning", "collector_interrupted")
            raise
        except Exception as e:
            self.state.status = "failed"
            log_event(self.logger, "error", "collector_failed", reason=str(e))
            raise
        finally:
            async with self._writer_lock:
                self._flush_pending_sync()
                await self.progress.save(self.state)
            await self.pool.close()

        self.state.status = "completed" if self._is_done() else "partial"
        await self.progress.save(self.state)
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
    p = argparse.ArgumentParser(description="Multi-LLM sentiment data collector for 6-class task")
    p.add_argument("--output", type=str, required=True, help="输出文件路径，支持 .csv/.jsonl/.json")
    p.add_argument("--target_count", type=int, default=10000, help="目标总条数（最终总量）")
    p.add_argument("--total_samples", type=int, default=None, help="兼容旧参数，等价于 target_count")

    p.add_argument("--categories", type=str, default="平静,开心,伤心,生气,惊讶,厌恶")
    p.add_argument("--distribution", type=str, default="0.167,0.167,0.167,0.167,0.166,0.166")

    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_tokens", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=20)
    p.add_argument("--max_workers", type=int, default=20, help="并发 worker 数")
    p.add_argument("--flush_every", type=int, default=20, help="累计多少条后安全落盘")
    p.add_argument("--progress_path", type=str, default=None, help="断点续跑进度文件路径")
    p.add_argument("--max_attempt_factor", type=int, default=30, help="最大尝试次数系数")

    p.add_argument("--model", type=str, default="gpt-4")
    p.add_argument("--provider", type=str, default="openai")
    p.add_argument("--api_key", type=str, default=None)
    p.add_argument("--base_url", type=str, default=None)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--qps_limit", type=float, default=5.0)
    p.add_argument("--model-config", type=str, default=None, help="JSON list for multi-model configs")

    p.add_argument("--request_timeout", type=float, default=30.0, help="单次请求超时")
    p.add_argument("--task_timeout", type=float, default=45.0, help="单任务总超时")
    p.add_argument("--backoff_base", type=float, default=0.5)
    p.add_argument("--backoff_max", type=float, default=8.0)

    p.add_argument("--similarity_threshold", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--enable_ppl", action="store_true")
    return p


async def async_main():
    parser = build_arg_parser()
    args = parser.parse_args()

    target_count = args.target_count if args.total_samples is None else args.total_samples
    if target_count <= 0:
        raise ValueError("target_count 必须大于 0")

    categories = [x.strip() for x in args.categories.split(",") if x.strip()]
    distribution = [float(x.strip()) for x in args.distribution.split(",") if x.strip()]
    if len(categories) != len(distribution):
        raise ValueError("categories 与 distribution 长度必须一致")
    if sum(distribution) <= 0:
        raise ValueError("distribution 总和必须大于 0")
    if abs(sum(distribution) - 1.0) > 1e-3:
        s = sum(distribution)
        distribution = [x / s for x in distribution]

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
        enable_ppl=args.enable_ppl,
        progress_path=args.progress_path,
        flush_every=args.flush_every,
        max_attempt_factor=args.max_attempt_factor,
        retry=retry_cfg,
    )

    collector = SentimentDataCollector(runtime_cfg=runtime_cfg, model_configs=model_configs)
    await collector.collect()


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Fatal error: {e}")