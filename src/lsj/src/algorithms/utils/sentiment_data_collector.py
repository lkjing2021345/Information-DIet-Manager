#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sentiment_data_collector.py

基于多模型并发蒸馏的六分类中文情感数据生成器（平静、开心、伤心、生气、惊讶、厌恶）

主要特性：
1) asyncio 并发 + 多模型池（OpenAI / Anthropic / 本地 OpenAI 兼容）
2) 细粒度配额控制（情感/场景/句式/长度）
3) 规则过滤 + 困惑度检测（可选）+ Sentence-BERT 语义去重
4) BloomFilter + HashSet 全局唯一
5) 模型健康检查、限流、重试、熔断与故障转移
6) 流式写入 CSV，支持大规模样本生成

依赖建议：
pip install httpx numpy
# 可选：
pip install sentence-transformers torch
pip install transformers

示例：
python sentiment_data_collector.py \
  --output ./sentiment_train.csv \
  --total_samples 10000 \
  --model-config '[{"name":"gpt-4","provider":"openai","concurrency":3,"weight":0.4},{"name":"claude-3-opus","provider":"anthropic","concurrency":3,"weight":0.4},{"name":"qwen-72b","provider":"local","base_url":"http://localhost:8000/v1","concurrency":4,"weight":0.2}]' \
  --temperature 0.8
"""

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import math
import random
import re
import time
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np


# -----------------------------
# Utilities
# -----------------------------
def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("SentimentDataCollector")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        ch = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


class BloomFilter:
    """简单 BloomFilter 实现（无外部依赖）"""

    def __init__(self, capacity: int = 200_000, error_rate: float = 0.01):
        m = -capacity * math.log(error_rate) / (math.log(2) ** 2)
        self.size = max(8, int(m))
        k = (self.size / capacity) * math.log(2)
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
    # 优先直接解析
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 从文本中提取 JSON 块
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


# -----------------------------
# Model Layer
# -----------------------------
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
    qps_limit: float = 5.0  # 每秒请求数


@dataclass
class ModelStats:
    total_calls: int = 0
    success_calls: int = 0
    failed_calls: int = 0
    total_latency: float = 0.0
    latencies: List[float] = field(default_factory=list)

    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.success_calls / self.total_calls

    def avg_latency(self) -> float:
        if self.success_calls == 0:
            return 0.0
        return self.total_latency / self.success_calls

    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        idx = int(0.95 * (len(sorted_l) - 1))
        return sorted_l[idx]


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

    async def _rate_limit_wait(self):
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
        # Anthropic 返回结构兼容处理
        if "content" in data and data["content"]:
            parts = data["content"]
            if isinstance(parts, list):
                return "".join([p.get("text", "") for p in parts])
        return json.dumps(data, ensure_ascii=False)


class LocalOpenAICompatibleClient(OpenAIClient):
    pass


class ModelPool:
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
        av = []
        for name, cfg in self.cfgs.items():
            if not self.breakers[name].is_open():
                av.append(cfg)
        return av

    def _pick_model(self) -> Optional[ModelConfig]:
        av = self._available_models()
        if not av:
            return None
        weights = [max(0.0, c.weight) for c in av]
        if sum(weights) == 0:
            return random.choice(av)
        return random.choices(av, weights=weights, k=1)[0]

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
# Collector
# -----------------------------
class SentimentDataCollector:
    def __init__(
        self,
        model_configs: List[ModelConfig],
        categories: List[str],
        distribution: List[float],
        total_samples: int,
        output: str,
        temperature: float = 0.8,
        max_tokens: int = 120,
        batch_size: int = 20,
        similarity_threshold: float = 0.85,
        random_seed: int = 42,
        log_level: str = "INFO",
        enable_ppl: bool = False,
    ):
        self.logger = setup_logger(log_level)
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        self.pool = ModelPool(model_configs, self.logger)
        self.categories = categories
        self.distribution = distribution
        self.total_samples = total_samples
        self.output = output
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold
        self.enable_ppl = enable_ppl

        self.scenes = ["工作场景", "日常生活", "社交互动", "健康状况", "学习考试"]
        self.sentence_types = {"陈述句": 0.4, "疑问句": 0.2, "感叹句": 0.3, "祈使句": 0.1}
        self.length_types = {"短文本": (5, 15, 0.3), "中文本": (16, 30, 0.4), "长文本": (31, 50, 0.3)}

        self.sensitive_words = {"暴恐", "涉黄", "政治极端", "仇恨言论"}
        self.exact_set = set()
        self.bloom = BloomFilter(capacity=max(10000, total_samples * 3), error_rate=0.005)

        # SBERT 去重相关
        self.embed_model = None
        self.embeddings = None  # np.ndarray [N, D]
        self.embedding_texts = []
        self._embed_lock = asyncio.Lock()
        self._try_load_sbert()

        # PPL 检测（可选）
        self.ppl_model = None
        self.ppl_tokenizer = None
        if self.enable_ppl:
            self._try_load_ppl_model()

        self.generated_count = 0
        self.accepted_count = 0
        self.duplicate_count = 0
        self.filtered_count = 0
        self.label_counts = {c: 0 for c in categories}
        self.start_ts = time.time()

        self.category_targets = self._build_targets(self.categories, self.distribution, total_samples)
        self.style_targets = self._build_targets(
            list(self.sentence_types.keys()),
            list(self.sentence_types.values()),
            total_samples,
        )
        self.length_targets = self._build_targets(
            list(self.length_types.keys()),
            [self.length_types[k][2] for k in self.length_types],
            total_samples,
        )
        self.scene_targets = self._build_targets(
            self.scenes, [1 / len(self.scenes)] * len(self.scenes), total_samples
        )

    def _try_load_sbert(self):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.embed_model = SentenceTransformer("shibing624/text2vec-base-chinese")
            self.logger.info("Sentence-BERT loaded: shibing624/text2vec-base-chinese")
        except Exception as e:
            self.embed_model = None
            self.logger.warning(f"Sentence-BERT not available, semantic dedup disabled. reason={e}")

    def _try_load_ppl_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
            import torch  # type: ignore

            name = "uer/gpt2-chinese-cluecorpussmall"
            self.ppl_tokenizer = AutoTokenizer.from_pretrained(name)
            self.ppl_model = AutoModelForCausalLM.from_pretrained(name)
            self.ppl_model.eval()
            self._torch = torch
            self.logger.info(f"PPL model loaded: {name}")
        except Exception as e:
            self.ppl_model = None
            self.ppl_tokenizer = None
            self.logger.warning(f"PPL model unavailable, fallback heuristic used. reason={e}")

    @staticmethod
    def _build_targets(keys: List[str], ratios: List[float], total: int) -> Dict[str, int]:
        ratios = np.array(ratios, dtype=float)
        ratios = ratios / ratios.sum()
        raw = ratios * total
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

    async def _generate_one(self, spec: Dict[str, str]) -> Optional[Dict[str, str]]:
        prompt = self._build_prompt(
            spec["label"], spec["scene"], spec["sentence_type"], spec["length_type"]
        )
        try:
            raw, model_name = await self.pool.generate(prompt, self.temperature, self.max_tokens)
            obj = safe_json_extract(raw)
            if not obj or "text" not in obj:
                return None
            text = normalize_text(str(obj["text"]))
            label = str(obj.get("label", spec["label"])).strip()
            if label != spec["label"]:
                return None
            return {"text": text, "label": label, "model": model_name}
        except Exception as e:
            self.logger.debug(f"generate_one failed: {e}")
            return None

    async def generate_batch(self, batch_specs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        tasks = [self._generate_one(spec) for spec in batch_specs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = []
        for r in results:
            if isinstance(r, dict):
                out.append(r)
        return out

    def _quick_duplicate(self, text: str) -> bool:
        if text in self.bloom and text in self.exact_set:
            return True
        return False

    async def semantic_deduplicate(self, text: str) -> bool:
        """
        True: 保留（非重复）
        False: 丢弃（重复）
        """
        if self._quick_duplicate(text):
            return False

        if self.embed_model is None:
            self.exact_set.add(text)
            self.bloom.add(text)
            return True

        async with self._embed_lock:
            emb = self.embed_model.encode([text], normalize_embeddings=True)[0]
            emb = np.asarray(emb, dtype=np.float32)

            if self.embeddings is not None and len(self.embeddings) > 0:
                sims = self.embeddings @ emb
                if float(np.max(sims)) >= self.similarity_threshold:
                    return False

            if self.embeddings is None:
                self.embeddings = emb.reshape(1, -1)
            else:
                self.embeddings = np.vstack([self.embeddings, emb.reshape(1, -1)])
            self.embedding_texts.append(text)
            self.exact_set.add(text)
            self.bloom.add(text)
            return True

    def _heuristic_ppl(self, text: str) -> float:
        # fallback：基于字符转移的粗略困惑度代理（值越小越好）
        if len(text) < 5:
            return 1000.0
        uniq = len(set(text))
        rep_ratio = 1 - uniq / len(text)
        punct = len(re.findall(r"[，。！？；,.!?]", text))
        score = 20 + rep_ratio * 200 - min(10, punct) * 0.5
        return score

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
        if ppl > 150:  # 可根据实际语料调参
            return False

        # 轻量标签一致性检查（关键词非强约束）
        emotion_hints = {
            "开心": ["开心", "高兴", "太棒", "激动", "喜悦"],
            "伤心": ["难过", "伤心", "失落", "低落", "痛苦"],
            "生气": ["生气", "愤怒", "气死", "火大", "恼火"],
            "惊讶": ["天哪", "竟然", "没想到", "怎么可能", "太意外"],
            "厌恶": ["恶心", "讨厌", "反感", "想吐", "厌烦"],
            "平静": ["平静", "放松", "安心", "踏实", "淡定"],
        }
        hints = emotion_hints.get(label, [])
        # 若完全没有线索，不强制拒绝；若出现明显冲突词则拒绝
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

    def save_to_csv(self, writer: csv.writer, text: str, label: str):
        writer.writerow([text, label])

    async def collect(self):
        used_style = {k: 0 for k in self.style_targets}
        used_len = {k: 0 for k in self.length_targets}
        used_scene = {k: 0 for k in self.scene_targets}

        max_attempts = self.total_samples * 30
        attempts = 0

        with open(self.output, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])

            while self.accepted_count < self.total_samples and attempts < max_attempts:
                remain = self.total_samples - self.accepted_count
                current_batch = min(self.batch_size, remain)
                batch_specs = []

                for _ in range(current_batch):
                    label = self._choose_from_remaining(self.category_targets, self.label_counts)
                    scene = self._choose_from_remaining(self.scene_targets, used_scene)
                    stype = self._choose_from_remaining(self.style_targets, used_style)
                    ltype = self._choose_from_remaining(self.length_targets, used_len)

                    batch_specs.append(
                        {"label": label, "scene": scene, "sentence_type": stype, "length_type": ltype}
                    )

                    # 预占位（若最终失败，不回滚，能促使配额整体趋近目标）
                    used_scene[scene] += 1
                    used_style[stype] += 1
                    used_len[ltype] += 1

                samples = await self.generate_batch(batch_specs)
                self.generated_count += len(samples)
                attempts += current_batch

                for s in samples:
                    text, label = s["text"], s["label"]

                    if not self.quality_filter(text, label):
                        self.filtered_count += 1
                        continue

                    uniq = await self.semantic_deduplicate(text)
                    if not uniq:
                        self.duplicate_count += 1
                        continue

                    self.save_to_csv(writer, text, label)
                    self.label_counts[label] += 1
                    self.accepted_count += 1

                    if self.accepted_count >= self.total_samples:
                        break

                if self.accepted_count % max(100, self.batch_size * 2) < self.batch_size:
                    self._log_progress()

        self._log_progress(final=True)
        await self.pool.close()

    def _log_progress(self, final: bool = False):
        elapsed = max(1e-6, time.time() - self.start_ts)
        speed = self.accepted_count / elapsed
        dedup_rate = self.duplicate_count / max(1, self.generated_count)
        filter_rate = self.filtered_count / max(1, self.generated_count)

        title = "FINAL REPORT" if final else "PROGRESS"
        self.logger.info(
            f"[{title}] accepted={self.accepted_count}/{self.total_samples}, "
            f"generated={self.generated_count}, speed={speed:.2f}条/秒, "
            f"dedup_rate={dedup_rate:.2%}, filter_rate={filter_rate:.2%}, "
            f"label_dist={self.label_counts}"
        )
        for name, st in self.pool.stats.items():
            self.logger.info(
                f"  model={name}, calls={st.total_calls}, succ_rate={st.success_rate():.2%}, "
                f"avg_lat={st.avg_latency():.2f}s, p95={st.p95_latency():.2f}s"
            )


# -----------------------------
# CLI
# -----------------------------
def parse_model_configs(args) -> List[ModelConfig]:
    if args.model_config:
        raw = args.model_config.strip()

        # ✅ 新增：检测是否为文件路径
        if (raw.endswith('.json') or os.path.sep in raw) and os.path.isfile(raw):
            try:
                with open(raw, 'r', encoding='utf-8') as f:
                    raw = f.read()
                print(f"✅ 已从文件加载配置: {args.model_config}")
            except Exception as e:
                raise ValueError(f"无法读取配置文件 {args.model_config}: {e}")

        # 清理可能的 BOM 头
        if raw.startswith('\ufeff'):
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

            # 必填字段检查
            required = ['name', 'provider']
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
    # 兼容单模型模式
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

    # 兼容单模型
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
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--total_samples", type=int, default=10000)

    p.add_argument("--categories", type=str, default="平静,开心,伤心,生气,惊讶,厌恶")
    p.add_argument("--distribution", type=str, default="0.167,0.167,0.167,0.167,0.166,0.166")

    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_tokens", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=20)

    # 单模型参数（兼容）
    p.add_argument("--model", type=str, default="gpt-4")
    p.add_argument("--provider", type=str, default="openai")
    p.add_argument("--api_key", type=str, default=None)
    p.add_argument("--base_url", type=str, default=None)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--qps_limit", type=float, default=5.0)

    # 多模型 JSON
    p.add_argument("--model-config", type=str, default=None, help="JSON list for multi-model configs")

    p.add_argument("--similarity_threshold", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--enable_ppl", action="store_true")
    return p


async def async_main():
    parser = build_arg_parser()
    args = parser.parse_args()

    categories = [x.strip() for x in args.categories.split(",") if x.strip()]
    distribution = [float(x.strip()) for x in args.distribution.split(",") if x.strip()]
    if len(categories) != len(distribution):
        raise ValueError("categories 与 distribution 长度必须一致")
    if abs(sum(distribution) - 1.0) > 1e-3:
        # 自动归一化
        s = sum(distribution)
        distribution = [x / s for x in distribution]

    model_configs = parse_model_configs(args)

    collector = SentimentDataCollector(
        model_configs=model_configs,
        categories=categories,
        distribution=distribution,
        total_samples=args.total_samples,
        output=args.output,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        similarity_threshold=args.similarity_threshold,
        random_seed=args.seed,
        log_level=args.log_level,
        enable_ppl=args.enable_ppl,
    )

    await collector.collect()


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Fatal error: {e}")