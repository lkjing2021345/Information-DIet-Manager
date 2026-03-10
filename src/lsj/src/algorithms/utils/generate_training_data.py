"""
训练数据生成脚本
使用多模型并发 + 批量请求对浏览器标签页标题进行分类，并增量写入训练数据集。

重构目标：
1. 参考 sentiment_data_collector.py 的架构，引入断点续跑、进度持久化、去重、单写入器、结构化统计。
2. 面向 generate_training_data.py 的业务场景，保留 7 类标题分类规则与 OpenAI 兼容接口。
3. 重点优化长时间运行吞吐：批量请求、连接复用、异步 worker 队列、JSONL 追加写、失败重试、限流与监控。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import aiohttp

from logger import setup_logger

logger = setup_logger(__name__, "../../../logs/generate_training_data.log")


# -----------------------------
# Utilities
# -----------------------------
def log_event(level: str, event: str, **kwargs: Any) -> None:
    payload = {"event": event, **kwargs}
    message = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    getattr(logger, level.lower(), logger.info)(message)


def normalize_input_text(text: str) -> str:
    return " ".join(str(text).replace("\u3000", " ").strip().split())


def compute_entry_id(text: str) -> str:
    return hashlib.sha1(normalize_input_text(text).encode("utf-8", errors="ignore")).hexdigest()


def safe_json_extract(text: str) -> Optional[Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start_obj = cleaned.find("{")
    end_obj = cleaned.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        try:
            return json.loads(cleaned[start_obj : end_obj + 1])
        except Exception:
            pass

    start_arr = cleaned.find("[")
    end_arr = cleaned.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        try:
            return json.loads(cleaned[start_arr : end_arr + 1])
        except Exception:
            return None
    return None


def build_professional_system_prompt(enable_reasoning: bool) -> str:
    reasoning_requirement = "reasoning 为必填字段，使用 8-30 个中文字符简要说明分类依据。" if enable_reasoning else "reasoning 字段必须返回，可为空字符串。"
    return f"""你是一个专业的数据构建与标注专家，负责生成可直接用于监督学习训练、评估分析和规则验证的“浏览器标签页标题分类数据”。

【数据生成目标】
基于输入的浏览器标签页标题数组，生成高质量、结构化、可复用的标题分类结果，用于训练和优化自动分类模型。输出结果必须满足：字段完整、格式统一、语义清晰、低重复、低噪声、符合真实业务场景、便于后续训练、评估与分析。

【数据类型】
结构化文本分类数据。每条样本表示一个浏览器标签页标题及其唯一分类标签。

【使用场景】
- 浏览器标签页自动归类
- 智能信息整理与标签聚合
- 用户浏览行为分析
- 个性化推荐与工作流优化
- 文本分类模型训练、验证与测试集构建

【目标受众】
- 机器学习工程师
- NLP 训练数据标注人员
- 数据分析师
- 产品算法团队
- 需要高一致性分类样本的数据治理团队

【任务说明】
你会收到一个标题数组，每个元素包含 id 和 title。你需要对每个 title 进行单标签分类，并返回结构化 JSON 结果。禁止遗漏任何输入 id，禁止新增输入中不存在的 id。

【字段结构】
输出中每条结果必须包含以下字段：
- id: 输入中的原始唯一编号，必须原样返回
- label: 分类标签，必须为 News、Tools、Learning、Shopping、Social、Entertainment、Other 之一
- confidence: 0.50 到 1.00 之间的小数，保留两位
- reasoning: 简要说明判定依据，{reasoning_requirement}

【格式要求】
- 只输出 JSON 对象
- 顶层结构必须严格为：
{{
  "results": [
    {{
      "id": "原始id",
      "label": "News|Tools|Learning|Shopping|Social|Entertainment|Other",
      "confidence": 0.95,
      "reasoning": "判定依据"
    }}
  ]
}}
- 不允许输出 markdown
- 不允许输出注释
- 不允许输出任何解释、前后缀说明或额外文本
- results 的数量必须与输入标题数量完全一致
- 每个输入 id 必须且只能出现一次

【样本数量要求】
- 输出样本数量必须与输入标题数量完全一致
- 不可遗漏、不可重复、不可额外生成无关样本

【内容范围】
输入标题可能来自以下场景：
- 新闻网站首页、资讯文章、财经快讯、科技媒体
- 开发平台、在线文档、协同办公、云盘、翻译工具、可视化工具
- 官方文档、教程文章、在线课程、学术检索、技术问答
- 电商平台、商品详情、购物车、订单页、优惠活动页
- 社交平台首页、社区帖子、讨论串、个人动态、消息页
- 视频平台、音乐平台、直播页、电影详情页、游戏平台
- 浏览器设置页、扩展管理页、登录授权页、错误页、本地文件页

【分类标签集合】
标签只能从以下 7 个类别中选择，且每条标题只能对应 1 个标签：
1. News
2. Tools
3. Learning
4. Shopping
5. Social
6. Entertainment
7. Other

【标签定义与判定规范】
1) News：以新闻报道、快讯、媒体资讯、时效性内容为核心用途的页面。
2) Tools：以完成任务、提高效率、开发协作、在线处理、内容编辑、管理控制为核心目的的页面。
3) Learning：以学习知识、阅读教程、查阅文档、课程学习、论文检索为核心目的的页面。
4) Shopping：以商品浏览、购买、支付、订单管理、优惠促销、比价为核心目的的页面。
5) Social：以社区互动、社交关系、动态交流、评论讨论、帖子内容为核心目的的页面。
6) Entertainment：以休闲消费内容为核心，包括视频、音乐、影视、直播、游戏、动漫等。
7) Other：无法明确归入以上类别，或属于系统页、本地页、登录页、错误页、空白页、设置页、授权页等。

【优先级规则】
当一个标题同时具有多个类别特征时，按照以下优先级判定：
Shopping > Social > Entertainment > Learning > Tools > News > Other

【例外修正规则】
- 科技媒体报道页优先标注为 News，不标为 Learning
- 官方文档、教程页优先标注为 Learning，不因品牌平台而标为 Tools
- 代码仓库、在线编辑器、控制台、协作平台优先标注为 Tools
- 纯登录页、错误页、设置页、跳转页、权限页优先标注为 Other
- 信息过少、用途不清、歧义严重时标注为 Other，且 confidence 不高于 0.65

【数据质量标准】
输出结果必须满足：
1. 字段完整
2. 格式统一
3. 语义清晰
4. 低重复
5. 低噪声
6. 场景真实
7. 类别边界清晰
8. 适合后续模型训练与分析
9. 标注逻辑一致
10. 结果可人工抽查验证

【约束条件】
- 必须仅依据标题的“当前页面核心用途”分类，不按品牌名机械分类
- 不得输出空 label、非法 label、缺失字段、错误类型值
- 不得输出与标题明显冲突的标签
- confidence 必须在 [0.50, 1.00] 范围内
- 若无法确定，也必须返回合法标签，优先 Other
- 不得返回任何敏感、无关、虚构解释文本

【去重要求】
- 不允许在 results 中重复返回同一个 id
- 不允许一条输入产生多条结果
- 每条结果必须与输入一一对应

【异常值控制】
以下情况禁止出现：
- id 缺失或与输入不一致
- label 不在允许集合内
- confidence 越界
- reasoning 过长、无关或完全空洞（如“无法判断”“随便猜测”）
- 输出结构不完整或 JSON 非法

【风格一致性要求】
- reasoning 风格保持简洁、客观、稳定
- 对相似标题使用一致的判定逻辑
- confidence 与确定性一致：越明确越高，越模糊越低

【标注规范】
- 以“当前页面核心用途”作为唯一判定依据
- 一个标题只能对应一个标签
- 边界样本采用保守策略
- 对近似场景保持一致标注

【输出示例】
{{
  "results": [
    {{
      "id": "sample_001",
      "label": "Tools",
      "confidence": 0.98,
      "reasoning": "代码仓库与开发协作工具"
    }},
    {{
      "id": "sample_002",
      "label": "Learning",
      "confidence": 0.97,
      "reasoning": "官方文档用于知识查阅"
    }},
    {{
      "id": "sample_003",
      "label": "Shopping",
      "confidence": 0.99,
      "reasoning": "核心用途是商品购买"
    }}
  ]
}}

【执行要求】
现在请严格按照以上规则，对输入标题逐条分类。除合法 JSON 结果外，不输出任何其他内容。"""


class AtomicFileIO:
    @staticmethod
    def atomic_write_text(path: str, content: str, encoding: str = "utf-8") -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, encoding=encoding, dir=str(path_obj.parent)) as tf:
            tf.write(content)
            temp_path = tf.name
        os.replace(temp_path, path)


@dataclass
class RetryConfig:
    request_timeout: float = 30.0
    task_timeout: float = 45.0
    max_retries: int = 3
    backoff_base: float = 0.5
    backoff_max: float = 8.0


@dataclass
class ModelStats:
    total_calls: int = 0
    success_calls: int = 0
    failed_calls: int = 0
    total_items: int = 0
    total_latency: float = 0.0
    latencies: List[float] = field(default_factory=list)

    def success_rate(self) -> float:
        return 0.0 if self.total_calls == 0 else self.success_calls / self.total_calls

    def avg_latency(self) -> float:
        return 0.0 if self.success_calls == 0 else self.total_latency / self.success_calls

    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(0.95 * (len(sorted_latencies) - 1))
        return sorted_latencies[idx]


class CircuitBreaker:
    def __init__(self, fail_threshold: int = 5, reset_timeout: float = 30.0):
        self.fail_threshold = fail_threshold
        self.reset_timeout = reset_timeout
        self.fail_count = 0
        self.open_until = 0.0

    def is_open(self) -> bool:
        return time.time() < self.open_until

    def on_success(self) -> None:
        self.fail_count = 0
        self.open_until = 0.0

    def on_failure(self) -> None:
        self.fail_count += 1
        if self.fail_count >= self.fail_threshold:
            self.open_until = time.time() + self.reset_timeout


@dataclass
class ModelConfig:
    name: str
    provider: str = "openai"
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    concurrent_requests: int = 5
    qps_limit: float = 10.0
    max_retries: int = 3
    weight: float = 1.0
    timeout: float = 30.0


@dataclass
class RuntimeConfig:
    training_data_path: str
    progress_path: str
    input_file: Optional[str] = None
    output_format: str = "jsonl"
    flush_every: int = 100
    queue_maxsize: int = 50000
    batch_size_per_request: int = 20
    writer_batch_size: int = 2000
    max_pending_tasks: int = 500
    max_samples: int = 200000
    target_count: int = 200000
    generate_mode: bool = True
    random_seed: int = 42
    min_confidence: float = 0.0
    enable_reasoning: bool = False
    report_every: int = 500
    strict_relabel_match: bool = False
    temperature: float = 0.0
    max_workers: Optional[int] = None
    retry: RetryConfig = field(default_factory=RetryConfig)


@dataclass
class ProgressState:
    training_data_path: str
    target_count: int = 0
    existing_count: int = 0
    processed_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    duplicate_count: int = 0
    invalid_count: int = 0
    attempt_count: int = 0
    generated_count: int = 0
    input_exhausted: bool = False
    status: str = "initialized"
    label_counts: Dict[str, int] = field(default_factory=dict)
    model_counts: Dict[str, int] = field(default_factory=dict)
    last_update_ts: float = 0.0


@dataclass
class TitleRecord:
    entry_id: str
    input: str
    label: str
    confidence: float
    model: str
    reasoning: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self, include_reasoning: bool = False) -> Dict[str, Any]:
        payload = {
            "entry_id": self.entry_id,
            "input": self.input,
            "label": self.label,
            "confidence": self.confidence,
            "model": self.model,
            "created_at": self.created_at,
        }
        if include_reasoning:
            payload["reasoning"] = self.reasoning
        return payload


# -----------------------------
# Progress / Store
# -----------------------------
class ProgressTracker:
    def __init__(self, progress_path: str):
        self.progress_path = progress_path
        self._lock = asyncio.Lock()

    def load(self) -> Optional[ProgressState]:
        if not self.progress_path or not os.path.exists(self.progress_path):
            return None
        try:
            with open(self.progress_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ProgressState(**data)
        except Exception as e:
            log_event("warning", "progress_load_failed", path=self.progress_path, reason=str(e))
            return None

    async def save(self, state: ProgressState) -> None:
        async with self._lock:
            state.last_update_ts = time.time()
            AtomicFileIO.atomic_write_text(
                self.progress_path,
                json.dumps(asdict(state), ensure_ascii=False, indent=2),
            )


class TrainingDataStore:
    def __init__(self, output_path: str, output_format: str = "jsonl", include_reasoning: bool = False):
        self.output_path = output_path
        self.output_format = output_format.lower()
        self.include_reasoning = include_reasoning

    def load_existing_ids(self) -> Tuple[Set[str], int]:
        ids: Set[str] = set()
        total_rows = 0
        path = Path(self.output_path)
        if not path.exists():
            return ids, total_rows

        if self.output_format == "json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for row in data:
                    if not isinstance(row, dict):
                        continue
                    text = normalize_input_text(row.get("input") or row.get("text") or "")
                    entry_id = str(row.get("entry_id") or compute_entry_id(text)) if text else ""
                    if entry_id:
                        ids.add(entry_id)
                        total_rows += 1
            return ids, total_rows

        if self.output_format == "csv":
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = normalize_input_text(row.get("input") or row.get("text") or "")
                    entry_id = str(row.get("entry_id") or compute_entry_id(text)) if text else ""
                    if entry_id:
                        ids.add(entry_id)
                        total_rows += 1
            return ids, total_rows

        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                text = normalize_input_text(row.get("input") or row.get("text") or "")
                entry_id = str(row.get("entry_id") or compute_entry_id(text)) if text else ""
                if entry_id:
                    ids.add(entry_id)
                    total_rows += 1
        return ids, total_rows

    def append_records(self, records: List[TitleRecord]) -> None:
        if not records:
            return
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.output_format == "csv":
            file_exists = path.exists() and path.stat().st_size > 0
            with open(path, "a", encoding="utf-8", newline="") as f:
                fieldnames = ["entry_id", "input", "label", "confidence", "model", "created_at"]
                if self.include_reasoning:
                    fieldnames.append("reasoning")
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                for record in records:
                    writer.writerow(record.to_dict(include_reasoning=self.include_reasoning))
            return

        if self.output_format == "jsonl":
            with open(path, "a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record.to_dict(include_reasoning=self.include_reasoning), ensure_ascii=False) + "\n")
            return

        existing: List[Dict[str, Any]] = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
                if isinstance(payload, list):
                    existing = payload
        existing.extend(record.to_dict(include_reasoning=self.include_reasoning) for record in records)
        AtomicFileIO.atomic_write_text(path.as_posix(), json.dumps(existing, ensure_ascii=False, indent=2))


# -----------------------------
# Input Loader
# -----------------------------
class InputSource:
    def __init__(self, input_file: Optional[str], inline_inputs: Optional[List[str]] = None):
        self.input_file = input_file
        self.inline_inputs = inline_inputs or []

    def __iter__(self) -> Iterable[str]:
        if self.inline_inputs:
            for item in self.inline_inputs:
                text = normalize_input_text(item)
                if text:
                    yield text
            return

        if not self.input_file:
            return

        path = Path(self.input_file)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                candidates = ["input", "title", "text"]
                for row in reader:
                    for col in candidates:
                        if row.get(col):
                            text = normalize_input_text(row[col])
                            if text:
                                yield text
                            break
            return

        if suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(row, dict):
                        text = normalize_input_text(row.get("input") or row.get("title") or row.get("text") or "")
                        if text:
                            yield text
                    elif isinstance(row, str):
                        text = normalize_input_text(row)
                        if text:
                            yield text
            return

        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        text = normalize_input_text(item.get("input") or item.get("title") or item.get("text") or "")
                    else:
                        text = normalize_input_text(item)
                    if text:
                        yield text
            return

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = normalize_input_text(line)
                if text:
                    yield text


# -----------------------------
# Model Layer
# -----------------------------
class ModelClient:
    def __init__(self, cfg: ModelConfig, retry_cfg: RetryConfig):
        self.cfg = cfg
        self.retry_cfg = retry_cfg
        self.semaphore = asyncio.Semaphore(max(1, cfg.concurrent_requests))
        self._rate_lock = asyncio.Lock()
        self._last_request_ts = 0.0
        self._min_interval = 1.0 / max(0.1, float(cfg.qps_limit))

    async def _rate_limit_wait(self) -> None:
        async with self._rate_lock:
            now = time.time()
            delta = now - self._last_request_ts
            if delta < self._min_interval:
                await asyncio.sleep(self._min_interval - delta)
            self._last_request_ts = time.time()

    async def _request_json_once(
        self,
        session: aiohttp.ClientSession,
        payload: Dict[str, Any],
    ) -> Tuple[int, str]:
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        async with session.post(
            url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=self.retry_cfg.request_timeout),
        ) as response:
            return response.status, await response.text()

    async def _request_json(
        self,
        session: aiohttp.ClientSession,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> List[Dict[str, Any]]:
        base_payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload_variants = [
            {**base_payload, "response_format": {"type": "json_object"}},
            base_payload,
        ]

        max_retries = max(0, int(self.cfg.max_retries or self.retry_cfg.max_retries))

        for attempt in range(max_retries + 1):
            try:
                await self._rate_limit_wait()
                async with self.semaphore:
                    status = 0
                    body = ""
                    for variant_index, payload in enumerate(payload_variants):
                        status, body = await self._request_json_once(session, payload)
                        if status == 200:
                            try:
                                parsed_body = json.loads(body)
                                content = parsed_body["choices"][0]["message"]["content"]
                                parsed_results = self._parse_batch_response(content)
                                if parsed_results:
                                    return parsed_results
                                log_event(
                                    "warning",
                                    "empty_or_unparseable_model_response",
                                    model=self.cfg.name,
                                    variant=variant_index,
                                    body_preview=body[:300],
                                )
                            except Exception as parse_error:
                                log_event(
                                    "warning",
                                    "model_response_parse_failed",
                                    model=self.cfg.name,
                                    variant=variant_index,
                                    reason=str(parse_error),
                                    body_preview=body[:300],
                                )
                            continue

                        if status == 429:
                            break

                        if status in {400, 404, 415, 422} and variant_index == 0:
                            log_event(
                                "warning",
                                "model_request_variant_fallback",
                                model=self.cfg.name,
                                status=status,
                                body_preview=body[:200],
                            )
                            continue
                        break

                    if status == 429:
                        wait_seconds = min(
                            self.retry_cfg.backoff_max,
                            self.retry_cfg.backoff_base * (2 ** attempt) + random.uniform(0, 0.5),
                        )
                        log_event(
                            "warning",
                            "rate_limited",
                            model=self.cfg.name,
                            attempt=attempt + 1,
                            wait_seconds=round(wait_seconds, 3),
                        )
                        await asyncio.sleep(wait_seconds)
                        continue

                    if attempt < max_retries:
                        backoff = min(
                            self.retry_cfg.backoff_max,
                            self.retry_cfg.backoff_base * (2 ** attempt) + random.uniform(0, 0.5),
                        )
                        log_event(
                            "warning",
                            "model_request_retry",
                            model=self.cfg.name,
                            status=status,
                            attempt=attempt + 1,
                            backoff=round(backoff, 3),
                            body_preview=body[:200],
                        )
                        await asyncio.sleep(backoff)
                        continue
                    return []
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    backoff = min(self.retry_cfg.backoff_max, self.retry_cfg.backoff_base * (2 ** attempt))
                    await asyncio.sleep(backoff)
                    continue
                log_event("warning", "request_timeout", model=self.cfg.name)
                return []
            except Exception as e:
                if attempt < max_retries:
                    backoff = min(self.retry_cfg.backoff_max, self.retry_cfg.backoff_base * (2 ** attempt))
                    await asyncio.sleep(backoff)
                    continue
                log_event("error", "batch_classify_failed", model=self.cfg.name, reason=str(e))
                return []
        return []

    async def classify_batch(
        self,
        session: aiohttp.ClientSession,
        system_prompt: str,
        batch_inputs: List[Tuple[str, str]],
        enable_reasoning: bool,
        temperature: float,
    ) -> List[Dict[str, Any]]:
        user_prompt = self._build_batch_prompt(batch_inputs, enable_reasoning)
        return await self._request_json(
            session=session,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=min(4000, 200 + len(batch_inputs) * 80),
            temperature=temperature,
        )

    async def generate_labeled_batch(
        self,
        session: aiohttp.ClientSession,
        system_prompt: str,
        batch_size: int,
        desired_labels: List[str],
        enable_reasoning: bool,
        temperature: float,
    ) -> List[Dict[str, Any]]:
        user_prompt = self._build_generation_prompt(batch_size, desired_labels, enable_reasoning)
        return await self._request_json(
            session=session,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=min(4000, 300 + batch_size * 100),
            temperature=temperature,
        )

    @staticmethod
    def _build_batch_prompt(batch_inputs: List[Tuple[str, str]], enable_reasoning: bool) -> str:
        items = [{"id": item_id, "title": title} for item_id, title in batch_inputs]
        reasoning_rule = (
            "reasoning 为必填字段，使用 8-30 个中文字符，简要说明判定依据"
            if enable_reasoning
            else "reasoning 字段必须返回，可为空字符串"
        )
        return (
            "请基于系统规则，对下面输入标题生成高质量结构化分类结果。"
            "输出目标是可直接用于训练和分析的低噪声标注数据。"
            "严格保证字段完整、格式统一、语义清晰、低重复、低噪声、符合真实业务场景。"
            "顶层只允许返回一个 JSON 对象，格式必须为 "
            '{"results":[{"id":"原始id","label":"News|Tools|Learning|Shopping|Social|Entertainment|Other",'
            '"confidence":0.95,"reasoning":"判定依据"}]}'
            f"。{reasoning_rule}。results 数量必须与输入数量完全一致，每个输入 id 必须且只能返回一次，"
            "禁止遗漏、禁止重复、禁止新增无关样本、禁止输出任何额外文本。\n"
            f"输入数据: {json.dumps(items, ensure_ascii=False)}"
        )

    @staticmethod
    def _build_generation_prompt(batch_size: int, desired_labels: List[str], enable_reasoning: bool) -> str:
        reasoning_rule = (
            "reasoning 为必填字段，使用 8-30 个中文字符，简要说明标题核心用途与标签关系"
            if enable_reasoning
            else "reasoning 字段必须返回，可为空字符串"
        )
        label_plan = [{"index": i + 1, "label": label} for i, label in enumerate(desired_labels)]
        return (
            "请直接生成一批可用于浏览器标签页分类蒸馏训练的高质量候选标题数据。"
            "你当前阶段只负责生成候选标题，并给出你预估的标签与理由，后续会由另一个标注阶段复核。"
            "请根据给定的标签规划，为每一项生成一个真实、自然、低重复、具有明确页面用途的浏览器标签页标题，"
            "标题必须像真实网页 title，不能是句子续写、不能是解释说明、不能是模板化占位文本。"
            "生成的数据必须覆盖真实网页场景，如新闻、工具、学习、电商、社交、娱乐、系统页等。"
            "不得生成乱码、模板化重复标题、无意义短语、明显虚构站点堆砌。"
            "每条结果必须包含 title、label、confidence、reasoning 四个字段。"
            f"{reasoning_rule}。"
            "顶层只允许输出一个 JSON 对象，格式必须为 "
            '{"results":[{"title":"页面标题","label":"News|Tools|Learning|Shopping|Social|Entertainment|Other","confidence":0.95,"reasoning":"判定依据"}]}'
            f"。results 数量必须严格等于 {batch_size}。"
            "每个标题都要唯一且风格多样，避免同义改写式重复。"
            f"标签规划: {json.dumps(label_plan, ensure_ascii=False)}"
        )

    @staticmethod
    def _parse_batch_response(content: str) -> List[Dict[str, Any]]:
        payload = safe_json_extract(content)
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            return payload["results"]
        if isinstance(payload, list):
            return payload
        return []


class ModelPool:
    def __init__(self, model_configs: List[ModelConfig], retry_cfg: RetryConfig):
        self.model_cfgs = model_configs
        self.retry_cfg = retry_cfg
        self.clients = {cfg.name: ModelClient(cfg, retry_cfg) for cfg in model_configs}
        self.model_stats: Dict[str, ModelStats] = {
            cfg.name: ModelStats() for cfg in model_configs
        }
        self.breakers: Dict[str, CircuitBreaker] = {
            cfg.name: CircuitBreaker() for cfg in model_configs
        }

    def _available_models(self) -> List[ModelConfig]:
        return [cfg for cfg in self.model_cfgs if not self.breakers[cfg.name].is_open()]

    def pick_model(self) -> ModelConfig:
        candidates = self._available_models() or self.model_cfgs
        weights = [max(0.0, cfg.weight) for cfg in candidates]
        if sum(weights) <= 0:
            return random.choice(candidates)
        return random.choices(candidates, weights=weights, k=1)[0]

    async def classify_batch(
        self,
        session: aiohttp.ClientSession,
        system_prompt: str,
        batch_inputs: List[Tuple[str, str]],
        enable_reasoning: bool,
        temperature: float,
    ) -> Tuple[List[Dict[str, Any]], str]:
        cfg = self.pick_model()
        stats = self.model_stats[cfg.name]
        stats.total_calls += 1
        t0 = time.time()
        results = await self.clients[cfg.name].classify_batch(
            session,
            system_prompt,
            batch_inputs,
            enable_reasoning,
            temperature,
        )
        latency = time.time() - t0
        stats.latencies.append(latency)
        stats.total_items += len(batch_inputs)
        if results:
            stats.success_calls += 1
            stats.total_latency += latency
            self.breakers[cfg.name].on_success()
        else:
            stats.failed_calls += 1
            self.breakers[cfg.name].on_failure()
        return results, cfg.name

    async def generate_labeled_batch(
        self,
        session: aiohttp.ClientSession,
        system_prompt: str,
        batch_size: int,
        desired_labels: List[str],
        enable_reasoning: bool,
        temperature: float,
    ) -> Tuple[List[Dict[str, Any]], str]:
        cfg = self.pick_model()
        stats = self.model_stats[cfg.name]
        stats.total_calls += 1
        t0 = time.time()
        results = await self.clients[cfg.name].generate_labeled_batch(
            session,
            system_prompt,
            batch_size,
            desired_labels,
            enable_reasoning,
            temperature,
        )
        latency = time.time() - t0
        stats.latencies.append(latency)
        stats.total_items += batch_size
        if results:
            stats.success_calls += 1
            stats.total_latency += latency
            self.breakers[cfg.name].on_success()
        else:
            stats.failed_calls += 1
            self.breakers[cfg.name].on_failure()
        return results, cfg.name


# -----------------------------
# Core Generator
# -----------------------------
class TrainingDataGenerator:
    """高吞吐训练数据生成器。"""

    VALID_LABELS = ["News", "Tools", "Learning", "Shopping", "Social", "Entertainment", "Other"]

    def __init__(self, runtime_cfg: RuntimeConfig, model_configs: List[ModelConfig]):
        self.runtime_cfg = runtime_cfg
        self.model_configs = model_configs
        self.pool = ModelPool(model_configs, runtime_cfg.retry)
        self.store = TrainingDataStore(
            runtime_cfg.training_data_path,
            output_format=runtime_cfg.output_format,
            include_reasoning=runtime_cfg.enable_reasoning,
        )
        self.progress = ProgressTracker(runtime_cfg.progress_path)
        self.state = ProgressState(
            training_data_path=runtime_cfg.training_data_path,
            target_count=runtime_cfg.target_count,
            label_counts={label: 0 for label in self.VALID_LABELS},
            model_counts={cfg.name: 0 for cfg in model_configs},
        )

        self.seen_entry_ids: Set[str] = set()
        self.pending_entry_ids: Set[str] = set()
        self.input_queue: asyncio.Queue[Optional[Any]] = asyncio.Queue(maxsize=runtime_cfg.queue_maxsize)
        self.result_queue: asyncio.Queue[Optional[TitleRecord]] = asyncio.Queue(maxsize=runtime_cfg.queue_maxsize)
        self.start_ts = time.time()
        self._writer_lock = asyncio.Lock()
        random.seed(runtime_cfg.random_seed)

        self.system_prompt = build_professional_system_prompt(runtime_cfg.enable_reasoning)
        self.flush_every = max(1, int(runtime_cfg.flush_every))
        self.report_every = max(1, int(runtime_cfg.report_every))
        self.strict_relabel_match = bool(runtime_cfg.strict_relabel_match)
        base_target = max(1, runtime_cfg.target_count)
        per_label = base_target // len(self.VALID_LABELS)
        remainder = base_target % len(self.VALID_LABELS)
        self.label_generation_targets = {
            label: per_label + (1 if idx < remainder else 0)
            for idx, label in enumerate(self.VALID_LABELS)
        }

    def _total_effective_count(self) -> int:
        return max(0, int(self.state.existing_count) + int(self.state.success_count))

    def _remaining_target(self) -> int:
        return max(0, int(self.runtime_cfg.target_count) - self._total_effective_count())

    def _is_done(self) -> bool:
        return self._total_effective_count() >= int(self.runtime_cfg.target_count)

    def _pick_generation_labels(self, batch_size: int) -> List[str]:
        planned: List[str] = []
        simulated_counts = dict(self.state.label_counts)
        for _ in range(batch_size):
            remaining = [
                (label, max(1, self.label_generation_targets.get(label, 1) - simulated_counts.get(label, 0)))
                for label in self.VALID_LABELS
            ]
            labels, weights = zip(*remaining)
            selected = random.choices(labels, weights=weights, k=1)[0]
            planned.append(selected)
            simulated_counts[selected] = simulated_counts.get(selected, 0) + 1
        return planned

    def restore(self) -> None:
        existing_ids, total_rows = self.store.load_existing_ids()
        self.seen_entry_ids = existing_ids
        restored = self.progress.load()
        if restored and restored.training_data_path == self.runtime_cfg.training_data_path:
            self.state = restored
            self.state.target_count = self.runtime_cfg.target_count
            self.state.existing_count = len(existing_ids)
            self.state.label_counts = {label: restored.label_counts.get(label, 0) for label in self.VALID_LABELS}
            for cfg in self.model_configs:
                self.state.model_counts.setdefault(cfg.name, 0)
            log_event(
                "info",
                "progress_resumed",
                processed=self.state.processed_count,
                success=self.state.success_count,
                existing_count=self.state.existing_count,
                target_count=self.state.target_count,
            )
        else:
            self.state.target_count = self.runtime_cfg.target_count
            self.state.existing_count = len(existing_ids)
            self.state.label_counts = {label: 0 for label in self.VALID_LABELS}
            self.state.model_counts = {cfg.name: 0 for cfg in self.model_configs}
        log_event(
            "info",
            "existing_data_loaded",
            existing_rows=total_rows,
            unique_ids=len(existing_ids),
            remaining_target=self._remaining_target(),
        )

    async def produce_inputs(self, source: Optional[InputSource]) -> None:
        enqueued_count = 0
        max_samples = max(0, int(self.runtime_cfg.max_samples))
        target_remaining = self._remaining_target()
        enqueue_limit = target_remaining if max_samples <= 0 else min(max_samples, target_remaining)
        try:
            if self.runtime_cfg.generate_mode:
                for _ in range(enqueue_limit):
                    if self._is_done():
                        log_event("info", "target_already_reached_before_enqueue", target_count=self.runtime_cfg.target_count)
                        break
                    await self.input_queue.put({"mode": "generate"})
                    enqueued_count += 1
                log_event(
                    "info",
                    "generation_tasks_enqueued",
                    enqueued_count=enqueued_count,
                    enqueue_limit=enqueue_limit,
                    target_count=self.runtime_cfg.target_count,
                )
                return

            if source is None:
                return

            for text in source:
                if self._is_done():
                    log_event("info", "target_already_reached_before_enqueue", target_count=self.runtime_cfg.target_count)
                    break
                if enqueue_limit and enqueued_count >= enqueue_limit:
                    log_event(
                        "info",
                        "input_limit_reached",
                        max_samples=max_samples,
                        enqueue_limit=enqueue_limit,
                        target_count=self.runtime_cfg.target_count,
                    )
                    break

                normalized = normalize_input_text(text)
                if not normalized:
                    continue
                entry_id = compute_entry_id(normalized)
                if entry_id in self.seen_entry_ids:
                    self.state.duplicate_count += 1
                    self.state.processed_count += 1
                    continue
                await self.input_queue.put(normalized)
                enqueued_count += 1
        finally:
            self.state.input_exhausted = True
            worker_count = self.runtime_cfg.max_workers or sum(max(1, cfg.concurrent_requests) for cfg in self.model_configs)
            for _ in range(worker_count):
                await self.input_queue.put(None)

    async def worker(self, worker_id: int, session: aiohttp.ClientSession) -> None:
        batch_size = max(1, self.runtime_cfg.batch_size_per_request)
        while True:
            if self.runtime_cfg.generate_mode:
                batch_tokens: List[Dict[str, Any]] = []
                first_item = await self.input_queue.get()
                if first_item is None:
                    self.input_queue.task_done()
                    await self.result_queue.put(None)
                    return
                batch_tokens.append(first_item)
                self.input_queue.task_done()

                for _ in range(batch_size - 1):
                    try:
                        item = self.input_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if item is None:
                        self.input_queue.task_done()
                        await self.input_queue.put(None)
                        break
                    batch_tokens.append(item)
                    self.input_queue.task_done()

                desired_labels = self._pick_generation_labels(len(batch_tokens))
                self.state.attempt_count += len(batch_tokens)
                try:
                    generated_items, model_name = await asyncio.wait_for(
                        self.pool.generate_labeled_batch(
                            session,
                            self.system_prompt,
                            len(batch_tokens),
                            desired_labels,
                            self.runtime_cfg.enable_reasoning,
                            self.runtime_cfg.temperature,
                        ),
                        timeout=self.runtime_cfg.retry.task_timeout,
                    )
                except asyncio.TimeoutError:
                    self.state.failed_count += len(batch_tokens)
                    log_event(
                        "warning",
                        "worker_generate_batch_timeout",
                        worker_id=worker_id,
                        batch_size=len(batch_tokens),
                    )
                    continue
                except Exception as e:
                    self.state.failed_count += len(batch_tokens)
                    log_event(
                        "error",
                        "worker_generate_batch_failed",
                        worker_id=worker_id,
                        batch_size=len(batch_tokens),
                        reason=str(e),
                    )
                    continue

                candidate_inputs: List[Tuple[str, str]] = []
                candidate_meta: Dict[str, Dict[str, Any]] = {}
                seen_titles_in_batch: Set[str] = set()
                for item in generated_items:
                    if not isinstance(item, dict):
                        continue
                    title = normalize_input_text(str(item.get("title", "")).strip())
                    label = str(item.get("label", "")).strip()
                    if not title or label not in self.VALID_LABELS:
                        self.state.invalid_count += 1
                        continue
                    if title in seen_titles_in_batch:
                        self.state.duplicate_count += 1
                        continue

                    entry_id = compute_entry_id(title)
                    if entry_id in self.seen_entry_ids or entry_id in self.pending_entry_ids:
                        self.state.duplicate_count += 1
                        continue

                    seen_titles_in_batch.add(title)
                    self.pending_entry_ids.add(entry_id)
                    candidate_inputs.append((entry_id, title))
                    self.state.generated_count += 1
                    candidate_meta[entry_id] = {
                        "title": title,
                        "generated_label": label,
                        "generated_reasoning": str(item.get("reasoning", "")).strip(),
                    }

                if not candidate_inputs:
                    log_event(
                        "info",
                        "worker_generate_batch_done",
                        worker_id=worker_id,
                        model=model_name,
                        batch_size=len(batch_tokens),
                        valid_records=0,
                        missing=max(0, len(batch_tokens) - len(generated_items)),
                        filtered_by_relabel=0,
                        desired_labels=desired_labels,
                    )
                    continue

                try:
                    relabeled_items, relabel_model_name = await asyncio.wait_for(
                        self.pool.classify_batch(
                            session,
                            self.system_prompt,
                            candidate_inputs,
                            self.runtime_cfg.enable_reasoning,
                            self.runtime_cfg.temperature,
                        ),
                        timeout=self.runtime_cfg.retry.task_timeout,
                    )
                except asyncio.TimeoutError:
                    self.state.failed_count += len(candidate_inputs)
                    for entry_id in candidate_meta:
                        self.pending_entry_ids.discard(entry_id)
                    log_event(
                        "warning",
                        "worker_relabel_batch_timeout",
                        worker_id=worker_id,
                        batch_size=len(candidate_inputs),
                    )
                    continue
                except Exception as e:
                    self.state.failed_count += len(candidate_inputs)
                    for entry_id in candidate_meta:
                        self.pending_entry_ids.discard(entry_id)
                    log_event(
                        "error",
                        "worker_relabel_batch_failed",
                        worker_id=worker_id,
                        batch_size=len(candidate_inputs),
                        reason=str(e),
                    )
                    continue

                relabeled_index: Dict[str, Dict[str, Any]] = {}
                for item in relabeled_items:
                    if not isinstance(item, dict):
                        continue
                    entry_id = str(item.get("id", "")).strip()
                    if entry_id in candidate_meta:
                        relabeled_index[entry_id] = item

                valid_records = 0
                filtered_by_relabel = 0
                for entry_id, meta in candidate_meta.items():
                    relabeled = relabeled_index.get(entry_id)
                    if relabeled is None:
                        self.state.failed_count += 1
                        self.pending_entry_ids.discard(entry_id)
                        continue

                    relabeled_label = str(relabeled.get("label", "")).strip()
                    if relabeled_label not in self.VALID_LABELS:
                        self.state.invalid_count += 1
                        self.pending_entry_ids.discard(entry_id)
                        continue

                    if self.strict_relabel_match and relabeled_label != meta["generated_label"]:
                        filtered_by_relabel += 1
                        self.state.invalid_count += 1
                        self.pending_entry_ids.discard(entry_id)
                        continue

                    try:
                        confidence = float(relabeled.get("confidence", 1.0))
                    except (TypeError, ValueError):
                        confidence = 1.0
                    confidence = max(0.0, min(1.0, confidence))
                    if confidence < self.runtime_cfg.min_confidence:
                        self.state.invalid_count += 1
                        self.pending_entry_ids.discard(entry_id)
                        continue

                    record = TitleRecord(
                        entry_id=entry_id,
                        input=meta["title"],
                        label=relabeled_label,
                        confidence=confidence,
                        model=f"{model_name}->{relabel_model_name}",
                        reasoning=str(relabeled.get("reasoning", meta["generated_reasoning"])).strip(),
                    )
                    await self.result_queue.put(record)
                    valid_records += 1

                missing_count = max(0, len(batch_tokens) - len(generated_items))
                if missing_count:
                    self.state.failed_count += missing_count

                log_event(
                    "info",
                    "worker_generate_batch_done",
                    worker_id=worker_id,
                    model=model_name,
                    relabel_model=relabel_model_name,
                    batch_size=len(batch_tokens),
                    generated_candidates=len(candidate_inputs),
                    valid_records=valid_records,
                    missing=missing_count,
                    filtered_by_relabel=filtered_by_relabel,
                    desired_labels=desired_labels,
                )
                continue

            batch_texts: List[str] = []
            first_item = await self.input_queue.get()
            if first_item is None:
                self.input_queue.task_done()
                await self.result_queue.put(None)
                return
            batch_texts.append(first_item)
            self.input_queue.task_done()

            for _ in range(batch_size - 1):
                try:
                    item = self.input_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    self.input_queue.task_done()
                    await self.input_queue.put(None)
                    break
                batch_texts.append(item)
                self.input_queue.task_done()

            batch_inputs: List[Tuple[str, str]] = []
            batch_index: Dict[str, str] = {}
            for text in batch_texts:
                entry_id = compute_entry_id(text)
                if entry_id in self.seen_entry_ids or entry_id in self.pending_entry_ids:
                    self.state.duplicate_count += 1
                    self.state.processed_count += 1
                    continue
                batch_index[entry_id] = text
                batch_inputs.append((entry_id, text))
                self.pending_entry_ids.add(entry_id)

            if not batch_inputs:
                continue

            try:
                self.state.attempt_count += len(batch_inputs)
                results, model_name = await asyncio.wait_for(
                    self.pool.classify_batch(
                        session,
                        self.system_prompt,
                        batch_inputs,
                        self.runtime_cfg.enable_reasoning,
                        self.runtime_cfg.temperature,
                    ),
                    timeout=self.runtime_cfg.retry.task_timeout,
                )
            except asyncio.TimeoutError:
                self.state.failed_count += len(batch_inputs)
                for entry_id in batch_index:
                    self.pending_entry_ids.discard(entry_id)
                log_event(
                    "warning",
                    "worker_classify_batch_timeout",
                    worker_id=worker_id,
                    batch_size=len(batch_inputs),
                )
                continue
            except Exception as e:
                self.state.failed_count += len(batch_inputs)
                for entry_id in batch_index:
                    self.pending_entry_ids.discard(entry_id)
                log_event(
                    "error",
                    "worker_classify_batch_failed",
                    worker_id=worker_id,
                    batch_size=len(batch_inputs),
                    reason=str(e),
                )
                continue

            returned_ids = set()
            valid_records = 0
            for item in results:
                if not isinstance(item, dict):
                    continue
                entry_id = str(item.get("id", "")).strip()
                label = str(item.get("label", "")).strip()
                if entry_id not in batch_index:
                    continue
                returned_ids.add(entry_id)
                if label not in self.VALID_LABELS:
                    self.state.invalid_count += 1
                    self.pending_entry_ids.discard(entry_id)
                    continue

                try:
                    confidence = float(item.get("confidence", 1.0))
                except (TypeError, ValueError):
                    confidence = 1.0
                confidence = max(0.0, min(1.0, confidence))
                if confidence < self.runtime_cfg.min_confidence:
                    self.state.invalid_count += 1
                    self.pending_entry_ids.discard(entry_id)
                    continue

                record = TitleRecord(
                    entry_id=entry_id,
                    input=batch_index[entry_id],
                    label=label,
                    confidence=confidence,
                    model=model_name,
                    reasoning=str(item.get("reasoning", "")).strip(),
                )
                await self.result_queue.put(record)
                valid_records += 1

            missing_ids = set(batch_index) - returned_ids
            if missing_ids:
                self.state.failed_count += len(missing_ids)
                for missing_id in missing_ids:
                    self.pending_entry_ids.discard(missing_id)

            log_event(
                "info",
                "worker_batch_done",
                worker_id=worker_id,
                model=model_name,
                batch_size=len(batch_inputs),
                valid_records=valid_records,
                missing=len(missing_ids),
            )

    async def writer(self, worker_done_count: int) -> None:
        done_workers = 0
        pending: List[TitleRecord] = []
        flush_threshold = max(1, min(self.runtime_cfg.writer_batch_size, self.flush_every))
        while done_workers < worker_done_count:
            item = await self.result_queue.get()
            if item is None:
                done_workers += 1
                self.result_queue.task_done()
                if pending:
                    await self.flush_records(pending)
                    pending.clear()
                continue

            pending.append(item)
            self.result_queue.task_done()
            if len(pending) >= flush_threshold:
                await self.flush_records(pending)
                pending.clear()

        if pending:
            await self.flush_records(pending)
        self.state.status = "completed"
        await self.progress.save(self.state)

    async def flush_records(self, records: List[TitleRecord]) -> None:
        async with self._writer_lock:
            remaining_target = self._remaining_target()
            if remaining_target <= 0:
                for record in records:
                    self.pending_entry_ids.discard(record.entry_id)
                return

            unique_records: List[TitleRecord] = []
            for record in records:
                if len(unique_records) >= remaining_target:
                    self.pending_entry_ids.discard(record.entry_id)
                    continue
                if record.entry_id in self.seen_entry_ids:
                    self.state.duplicate_count += 1
                    self.pending_entry_ids.discard(record.entry_id)
                    continue
                self.seen_entry_ids.add(record.entry_id)
                self.pending_entry_ids.discard(record.entry_id)
                unique_records.append(record)

            if not unique_records:
                return

            self.store.append_records(unique_records)
            self.state.success_count += len(unique_records)
            self.state.processed_count += len(unique_records)
            for record in unique_records:
                self.state.label_counts[record.label] = self.state.label_counts.get(record.label, 0) + 1
                self.state.model_counts[record.model] = self.state.model_counts.get(record.model, 0) + 1

            if self.state.success_count % self.report_every < len(unique_records):
                self.log_progress()
            await self.progress.save(self.state)

    def log_progress(self) -> None:
        elapsed = max(1e-6, time.time() - self.start_ts)
        throughput = self.state.success_count / elapsed
        log_event(
            "info",
            "progress",
            processed_count=self.state.processed_count,
            success_count=self.state.success_count,
            existing_count=self.state.existing_count,
            total_effective_count=self._total_effective_count(),
            target_count=self.runtime_cfg.target_count,
            remaining_target=self._remaining_target(),
            duplicate_count=self.state.duplicate_count,
            failed_count=self.state.failed_count,
            invalid_count=self.state.invalid_count,
            throughput_per_sec=round(throughput, 2),
            label_counts=self.state.label_counts,
            model_counts=self.state.model_counts,
        )
        for model_name, stats in self.pool.model_stats.items():
            log_event(
                "info",
                "model_stats",
                model=model_name,
                calls=stats.total_calls,
                success=stats.success_calls,
                fail=stats.failed_calls,
                items=stats.total_items,
                succ_rate=round(stats.success_rate(), 4),
                avg_latency=round(stats.avg_latency(), 3),
                p95_latency=round(stats.p95_latency(), 3),
                circuit_open=self.pool.breakers[model_name].is_open(),
            )

    def _log_summary(self, final: bool = False) -> None:
        elapsed = max(1e-6, time.time() - self.start_ts)
        log_event(
            "info",
            "final_summary" if final else "summary",
            processed_count=self.state.processed_count,
            success_count=self.state.success_count,
            existing_count=self.state.existing_count,
            total_effective_count=self._total_effective_count(),
            target_count=self.runtime_cfg.target_count,
            remaining_target=self._remaining_target(),
            generated_count=self.state.generated_count,
            attempt_count=self.state.attempt_count,
            duplicate_count=self.state.duplicate_count,
            failed_count=self.state.failed_count,
            invalid_count=self.state.invalid_count,
            elapsed_seconds=round(elapsed, 3),
            status=self.state.status,
        )

    async def classify_batch(self, inputs: List[str]) -> List[Dict[str, Any]]:
        temp_output_path = str(Path(tempfile.gettempdir()) / f"generate_training_data_{int(time.time() * 1000)}_{random.randint(1000, 9999)}.{self.runtime_cfg.output_format}")
        temp_progress_path = f"{temp_output_path}.progress.json"
        temp_runtime = RuntimeConfig(
            training_data_path=temp_output_path,
            progress_path=temp_progress_path,
            output_format=self.runtime_cfg.output_format,
            flush_every=self.runtime_cfg.flush_every,
            queue_maxsize=max(len(inputs) * 2, 1000),
            batch_size_per_request=self.runtime_cfg.batch_size_per_request,
            writer_batch_size=max(1, min(len(inputs), self.runtime_cfg.writer_batch_size)),
            max_pending_tasks=self.runtime_cfg.max_pending_tasks,
            max_samples=min(len(inputs), self.runtime_cfg.max_samples),
            target_count=min(len(inputs), self.runtime_cfg.target_count),
            generate_mode=False,
            random_seed=self.runtime_cfg.random_seed,
            min_confidence=self.runtime_cfg.min_confidence,
            enable_reasoning=self.runtime_cfg.enable_reasoning,
            report_every=self.runtime_cfg.report_every,
            strict_relabel_match=self.runtime_cfg.strict_relabel_match,
            temperature=self.runtime_cfg.temperature,
            max_workers=self.runtime_cfg.max_workers,
            retry=self.runtime_cfg.retry,
        )
        generator = TrainingDataGenerator(temp_runtime, self.model_configs)
        await generator.run(InputSource(None, inline_inputs=inputs))
        path = Path(temp_runtime.training_data_path)
        if not path.exists():
            return []

        results: List[Dict[str, Any]] = []
        try:
            if temp_runtime.output_format == "json":
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                    if isinstance(payload, list):
                        results.extend(payload)
            elif temp_runtime.output_format == "csv":
                with open(path, "r", encoding="utf-8-sig", newline="") as f:
                    results.extend(list(csv.DictReader(f)))
            else:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            results.append(json.loads(line))
        finally:
            for cleanup_path in [temp_output_path, temp_progress_path]:
                try:
                    if os.path.exists(cleanup_path):
                        os.remove(cleanup_path)
                except Exception:
                    pass
        return results

    async def collect(self, source: Optional[InputSource]) -> None:
        self.restore()
        await self.progress.save(self.state)
        if self._is_done():
            self.state.status = "completed"
            await self.progress.save(self.state)
            self.log_progress()
            self._log_summary(final=True)
            return

        self.state.status = "running"
        await self.progress.save(self.state)

        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300, ssl=False)
        timeout = aiohttp.ClientTimeout(total=self.runtime_cfg.retry.request_timeout)
        worker_count = self.runtime_cfg.max_workers or sum(max(1, cfg.concurrent_requests) for cfg in self.model_configs)

        try:
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                producer_task = asyncio.create_task(self.produce_inputs(source))
                worker_tasks = [
                    asyncio.create_task(self.worker(i + 1, session))
                    for i in range(worker_count)
                ]
                writer_task = asyncio.create_task(self.writer(worker_count))

                await producer_task
                await asyncio.gather(*worker_tasks)
                await self.result_queue.join()
                await writer_task
        except KeyboardInterrupt:
            self.state.status = "interrupted"
            await self.progress.save(self.state)
            self._log_summary(final=True)
            raise
        except Exception as e:
            self.state.status = "failed"
            log_event("error", "collector_failed", reason=str(e))
            await self.progress.save(self.state)
            self._log_summary(final=True)
            raise

        if not self.state.status == "completed":
            self.state.status = "partial" if not self._is_done() else "completed"
        await self.progress.save(self.state)
        self.log_progress()
        self._log_summary(final=True)

    async def run(self, source: Optional[InputSource]) -> None:
        await self.collect(source)

    def append_to_training_data(self, new_data: List[Dict[str, Any]], training_data_path: str) -> None:
        records = []
        for item in new_data:
            text = normalize_input_text(item.get("input") or item.get("text") or "")
            label = str(item.get("label", "")).strip()
            if not text or label not in self.VALID_LABELS:
                continue
            records.append(
                TitleRecord(
                    entry_id=compute_entry_id(text),
                    input=text,
                    label=label,
                    confidence=float(item.get("confidence", 1.0)),
                    model=str(item.get("model", "manual")),
                    reasoning=str(item.get("reasoning", "")),
                )
            )
        TrainingDataStore(training_data_path, self.runtime_cfg.output_format, self.runtime_cfg.enable_reasoning).append_records(records)


# -----------------------------
# Config Helpers / CLI
# -----------------------------
def _load_model_config_payload(raw_value: Any, base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return raw_value
    if not isinstance(raw_value, str):
        raise ValueError("model_config/models 必须为 JSON 数组、文件路径或数组对象")

    raw_text = raw_value.strip()
    if not raw_text:
        return []

    candidate_paths: List[Path] = []
    if base_dir is not None:
        candidate_paths.append((base_dir / raw_text).resolve())
    candidate_paths.append(Path(raw_text).resolve())

    for candidate in candidate_paths:
        if candidate.is_file():
            with open(candidate, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, list):
                raise ValueError("models/model_config 内容必须是数组")
            return payload

    payload = json.loads(raw_text)
    if not isinstance(payload, list):
        raise ValueError("models/model_config 内容必须是数组")
    return payload


def parse_model_configs(config: Dict[str, Any], base_dir: Optional[Path] = None) -> List[ModelConfig]:
    models_payload = None
    if config.get("models") is not None:
        models_payload = config.get("models")
    elif config.get("model_config") is not None:
        models_payload = config.get("model_config")

    if models_payload is not None:
        models = _load_model_config_payload(models_payload, base_dir=base_dir)
        if not models:
            raise ValueError("models/model_config 配置为空")

        cfgs = []
        for idx, item in enumerate(models):
            if not isinstance(item, dict):
                raise ValueError(f"models[{idx}] 必须为对象")
            provider = str(item.get("provider", "openai")).strip().lower()
            for key in ["name"]:
                if not item.get(key):
                    raise ValueError(f"models[{idx}] 缺少必需字段: {key}")

            model_name = str(item.get("model") or item.get("name") or "").strip()
            if provider in {"openai", "local"}:
                for key in ["base_url"]:
                    if not item.get(key):
                        raise ValueError(f"models[{idx}] 缺少必需字段: {key}")
                if not model_name:
                    raise ValueError(f"models[{idx}] 缺少必需字段: model 或 name")

            cfgs.append(
                ModelConfig(
                    name=str(item["name"]),
                    provider=provider,
                    api_key=str(item.get("api_key") or ""),
                    base_url=str(item.get("base_url") or "").rstrip("/"),
                    model=model_name,
                    concurrent_requests=int(item.get("concurrency", item.get("concurrent_requests", 5))),
                    qps_limit=float(item.get("qps_limit", config.get("qps_limit", 10.0))),
                    max_retries=int(item.get("max_retries", config.get("max_retries", 3))),
                    weight=float(item.get("weight", 1.0)),
                    timeout=float(item.get("timeout", config.get("request_timeout", 30.0))),
                )
            )
        return cfgs

    required_fields = ["api_key", "base_url", "model"]
    missing_fields = [field for field in required_fields if not config.get(field)]
    if missing_fields:
        raise ValueError(f"配置文件缺少必需字段: {', '.join(missing_fields)}")

    return [
        ModelConfig(
            name=str(config.get("name", "default")),
            provider=str(config.get("provider", "openai")).strip().lower(),
            api_key=str(config["api_key"]),
            base_url=str(config["base_url"]).rstrip("/"),
            model=str(config["model"]),
            concurrent_requests=int(config.get("concurrent_requests", config.get("concurrency", 5))),
            qps_limit=float(config.get("qps_limit", 10.0)),
            max_retries=int(config.get("max_retries", 3)),
            weight=1.0,
            timeout=float(config.get("timeout", config.get("request_timeout", 30.0))),
        )
    ]


def build_runtime_config(config: Dict[str, Any], args: argparse.Namespace) -> RuntimeConfig:
    training_data_path = args.output or config.get("training_data_path")
    if not training_data_path:
        raise ValueError("缺少 training_data_path/--output 配置")

    return RuntimeConfig(
        training_data_path=training_data_path,
        progress_path=str(config.get("progress_path") or f"{training_data_path}.progress.json"),
        input_file=args.input_file or config.get("input_file"),
        output_format=str(config.get("output_format", "jsonl")).lower(),
        flush_every=int(config.get("flush_every", 100)),
        queue_maxsize=int(config.get("queue_maxsize", 50000)),
        batch_size_per_request=int(config.get("batch_size_per_request", 20)),
        writer_batch_size=int(config.get("writer_batch_size", 2000)),
        max_pending_tasks=int(config.get("max_pending_tasks", 500)),
        max_samples=max(0, int(args.max_samples if args.max_samples is not None else config.get("max_samples", 200000))),
        target_count=max(0, int(args.target_count if args.target_count is not None else config.get("target_count", config.get("max_samples", 200000)))),
        generate_mode=bool(config.get("generate_mode", True)),
        random_seed=int(config.get("random_seed", 42)),
        min_confidence=float(config.get("min_confidence", 0.0)),
        enable_reasoning=bool(config.get("enable_reasoning", False)),
        report_every=int(config.get("report_every", 500)),
        strict_relabel_match=bool(config.get("strict_relabel_match", False)),
        retry=RetryConfig(
            request_timeout=float(config.get("request_timeout", 30.0)),
            task_timeout=float(config.get("task_timeout", 45.0)),
            max_retries=int(config.get("max_retries", 3)),
            backoff_base=float(config.get("backoff_base", 0.5)),
            backoff_max=float(config.get("backoff_max", 8.0)),
        ),
    )


def build_runtime_configs(config: Dict[str, Any], args: argparse.Namespace) -> List[RuntimeConfig]:
    jobs = config.get("jobs")
    if isinstance(jobs, list) and jobs:
        runtime_cfgs: List[RuntimeConfig] = []
        for idx, job in enumerate(jobs):
            if not isinstance(job, dict):
                raise ValueError(f"jobs[{idx}] 必须为对象")
            merged = dict(config)
            merged.update(job)
            merged.pop("jobs", None)
            runtime_cfgs.append(build_runtime_config(merged, args))
        return runtime_cfgs
    return [build_runtime_config(config, args)]


async def main() -> None:
    parser = argparse.ArgumentParser(description="高吞吐训练数据生成和分类工具")
    parser.add_argument("--config", default="config.json", help="配置文件路径（默认: config.json）")
    parser.add_argument("--input", nargs="+", help="直接传入标题列表（测试模式）")
    parser.add_argument("--input_file", type=str, help="输入文件路径，支持 .txt/.csv/.json/.jsonl")
    parser.add_argument("--output", type=str, help="输出文件路径，默认读取 config 中 training_data_path")
    parser.add_argument("--max_samples", type=int, help="本次最多处理多少条输入标题")
    parser.add_argument("--target_count", type=int, help="目标最终总量（包含历史已存在数据）")
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        raise SystemExit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_configs = parse_model_configs(config, base_dir=config_path.parent)
    runtime_cfgs = build_runtime_configs(config, args)

    for runtime_cfg in runtime_cfgs:
        generator = TrainingDataGenerator(runtime_cfg=runtime_cfg, model_configs=model_configs)
        log_event(
            "info",
            "run_config",
            input_file=runtime_cfg.input_file,
            output_path=runtime_cfg.training_data_path,
            progress_path=runtime_cfg.progress_path,
            max_samples=runtime_cfg.max_samples,
            target_count=runtime_cfg.target_count,
            generate_mode=runtime_cfg.generate_mode,
            batch_size_per_request=runtime_cfg.batch_size_per_request,
            writer_batch_size=runtime_cfg.writer_batch_size,
            flush_every=runtime_cfg.flush_every,
            strict_relabel_match=runtime_cfg.strict_relabel_match,
            models=[cfg.name for cfg in model_configs],
        )

        if args.input:
            runtime_cfg.generate_mode = False
            source = InputSource(None, inline_inputs=args.input)
        elif runtime_cfg.input_file:
            runtime_cfg.generate_mode = False
            source = InputSource(runtime_cfg.input_file)
        elif runtime_cfg.generate_mode:
            source = None
        else:
            sample_inputs = [
                "GitHub - microsoft/vscode: Visual Studio Code",
                "Python Documentation - Built-in Functions",
                "Amazon.com: Online Shopping for Electronics",
                "CNN - Breaking News, Latest News and Videos",
                "YouTube - Broadcast Yourself",
                "Stack Overflow - Where Developers Learn",
                "Netflix - Watch TV Shows Online",
                "Twitter / X - Home",
                "Google Translate",
                "Spotify - Web Player",
                "淘宝网 - 淘！我喜欢",
                "知乎 - 有问题，就会有答案",
                "哔哩哔哩 (゜-゜)つロ 干杯~",
                "微博 - 随时随地发现新鲜事",
                "百度网盘 - 自由存，随心看",
                "CSDN - 专业开发者社区",
                "人民网 - 网上的人民日报",
                "豆瓣电影 - 你的光影记录",
                "腾讯新闻 - 事实派",
                "网易云音乐 - 听见好时光",
            ]
            source = InputSource(None, inline_inputs=sample_inputs)

        await generator.run(source)


if __name__ == "__main__":
    asyncio.run(main())