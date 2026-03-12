"""
文本分类训练脚本。

职责：
    提供分类训练所需的数据检查、清洗、划分、训练、评估与模型导出流程。

说明：
    该文件更偏向离线训练与实验使用，因此保留了较详细的数据质量日志，
    便于在训练前尽早发现标注、分布和样本质量问题。
"""
import json
import os
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).parent))

from classifier import ContentClassifier
from utils.logger import setup_logger

# Windows 固定 Hugging Face 缓存目录，确保后续训练优先复用本地缓存。
HF_HUB_CACHE_DIR = Path(r"C:\Users\Administrator\.cache\huggingface\hub")
HF_HOME_DIR = HF_HUB_CACHE_DIR.parent
HF_PERSISTENT_MODELS_DIR = HF_HUB_CACHE_DIR / "persistent_models"


def configure_huggingface_cache() -> Path:
    """统一指定 Hugging Face 默认缓存目录，避免落到临时目录。"""
    HF_HOME_DIR.mkdir(parents=True, exist_ok=True)
    HF_HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    HF_PERSISTENT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(HF_HOME_DIR)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HUB_CACHE_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(HF_HUB_CACHE_DIR)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    return HF_HUB_CACHE_DIR


configure_huggingface_cache()

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None

logger = setup_logger(__name__, "../../logs/classifier_train.log")


def _sanitize_model_cache_name(model_name: str) -> str:
    """将 repo id/path 转为稳定目录名，适配 Windows 文件系统。"""
    return model_name.replace("\\", "__").replace("/", "__").replace(":", "_")



def ensure_model_cached(model_name_or_path: str) -> str:
    """
    首次运行自动下载模型到固定缓存目录；后续运行优先从本地持久化缓存加载。
    保存内容包括模型权重、配置文件、tokenizer 文件以及 Hugging Face 所需依赖文件。
    """
    input_path = Path(model_name_or_path)
    if input_path.exists():
        return str(input_path)

    configure_huggingface_cache()
    persistent_dir = HF_PERSISTENT_MODELS_DIR / _sanitize_model_cache_name(model_name_or_path)
    config_file = persistent_dir / "config.json"
    tokenizer_candidates = [
        persistent_dir / "tokenizer.json",
        persistent_dir / "tokenizer_config.json",
        persistent_dir / "vocab.txt",
        persistent_dir / "sentencepiece.bpe.model",
    ]
    has_tokenizer_files = any(path.exists() for path in tokenizer_candidates)
    has_weight_files = any(
        (persistent_dir / file_name).exists()
        for file_name in ["pytorch_model.bin", "model.safetensors", "tf_model.h5", "flax_model.msgpack"]
    )

    if config_file.exists() and has_tokenizer_files and has_weight_files:
        logger.info("Using persisted Hugging Face model cache: %s", persistent_dir)
        return str(persistent_dir)

    persistent_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Persistent Hugging Face cache miss, downloading model: %s", model_name_or_path)

    if snapshot_download is not None:
        snapshot_download(
            repo_id=model_name_or_path,
            cache_dir=str(HF_HUB_CACHE_DIR),
            local_dir=str(persistent_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        logger.info("Model snapshot downloaded to persistent cache: %s", persistent_dir)
        return str(persistent_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=str(HF_HUB_CACHE_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        cache_dir=str(HF_HUB_CACHE_DIR),
    )
    tokenizer.save_pretrained(persistent_dir)
    model.save_pretrained(persistent_dir)
    logger.info("Model/tokenizer saved to persistent cache: %s", persistent_dir)
    return str(persistent_dir)


@dataclass
class TransformerTrainingConfig:
    """Transformer 分类训练配置。"""
    pretrained_model_name: str = "xlm-roberta-large"
    output_dir: str = str(Path(__file__).parent / "models" / "classifier_xlm_roberta")
    max_length: int = 128
    train_batch_size: int = 4
    eval_batch_size: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    num_epochs: int = 50
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    random_seed: int = 42
    val_size: float = 0.1
    test_size: float = 0.15
    num_workers: int = 0
    use_fp16: bool = True
    balance_strategy: str = "upsample"
    min_length: int = 3
    monitor: str = "val_loss"
    early_stopping_patience: int = 5
    early_stopping_mode: str = "min"
    restore_best_weights: bool = True
    early_stopping_min_delta: float = 1e-4


class TextClassificationDataset(Dataset):
    """将原始文本与标签包装为可供 DataLoader 使用的数据集。"""
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]],
        tokenizer,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx]).strip()
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        item = {key: value.squeeze(0) for key, value in encoded.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class TransformerTrainer:
    """
    Transformer 分类训练器。

    负责模型初始化、数据装载、训练循环、验证评估以及最终权重导出。
    """
    def __init__(
        self,
        label2id: Dict[str, int],
        config: TransformerTrainingConfig,
    ) -> None:
        if not label2id:
            raise ValueError("label2id 不能为空")
        if config.train_batch_size <= 0 or config.eval_batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        if config.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps 必须大于 0")
        if config.max_length <= 0:
            raise ValueError("max_length 必须大于 0")
        if config.monitor != "val_loss":
            raise ValueError("当前仅支持 monitor='val_loss'")
        if config.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience 必须大于 0")
        if config.early_stopping_mode != "min":
            raise ValueError("当前仅支持 early_stopping_mode='min'")

        self.config = config
        self.label2id = label2id
        self.id2label = {idx: label for label, idx in label2id.items()}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_source = ensure_model_cached(config.pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            cache_dir=str(HF_HUB_CACHE_DIR),
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=len(label2id),
            id2label={int(key): value for key, value in self.id2label.items()},
            label2id=label2id,
            cache_dir=str(HF_HUB_CACHE_DIR),
            local_files_only=True,
        )
        self.model.to(self.device)

        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            # 在 GPU 上按 8 对齐通常更利于 Tensor Core 发挥吞吐。
            pad_to_multiple_of=8 if self.device.type == "cuda" else None,
        )
        self.use_amp = self.device.type == "cuda" and config.use_fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        logger.info("训练设备: %s", self.device)
        logger.info("预训练模型: %s", config.pretrained_model_name)
        logger.info("AMP 混合精度: %s", "开启" if self.use_amp else "关闭")

    def create_dataloader(
        self,
        texts: List[str],
        labels: Optional[List[int]],
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        dataset = TextClassificationDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=self.data_collator,
            pin_memory=(self.device.type == "cuda"),
        )

    def train(
        self,
        train_data: List[Dict[str, str]],
        val_data: List[Dict[str, str]],
    ) -> Dict[str, float]:
        train_texts = [item["text"] for item in train_data]
        train_labels = [self.label2id[item["label"]] for item in train_data]
        val_texts = [item["text"] for item in val_data]
        val_labels = [self.label2id[item["label"]] for item in val_data]

        if not train_texts:
            raise ValueError("训练集为空，无法训练模型")
        if not val_texts:
            raise ValueError("验证集为空，无法进行验证")

        train_loader = self.create_dataloader(
            train_texts,
            train_labels,
            batch_size=self.config.train_batch_size,
            shuffle=True,
        )
        val_loader = self.create_dataloader(
            val_texts,
            val_labels,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
        )

        total_train_steps = max(
            1,
            # 配合梯度累积时，将 loss 按累积步数缩放，保持整体梯度量级稳定。
            (len(train_loader) * self.config.num_epochs) // self.config.gradient_accumulation_steps,
        )
        warmup_steps = int(total_train_steps * self.config.warmup_ratio)

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps,
        )

        best_val_loss = float("inf")
        best_val_accuracy = 0.0
        best_metrics: Dict[str, float] = {}
        best_state_dict = None
        epochs_without_improvement = 0

        logger.info("开始 Transformer 训练")
        logger.info("训练样本数: %d", len(train_data))
        logger.info("验证样本数: %d", len(val_data))
        logger.info("训练步数: %d", total_train_steps)
        logger.info("Warmup 步数: %d", warmup_steps)
        logger.info(
            "早停配置: monitor=%s, patience=%d, mode=%s, restore_best_weights=%s, min_delta=%.6f",
            self.config.monitor,
            self.config.early_stopping_patience,
            self.config.early_stopping_mode,
            self.config.restore_best_weights,
            self.config.early_stopping_min_delta,
        )

        for epoch in range(self.config.num_epochs):
            self.model.train()
            optimizer.zero_grad()
            running_loss = 0.0

            for step, batch in enumerate(train_loader, start=1):
                batch = {key: value.to(self.device) for key, value in batch.items()}

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.config.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
                running_loss += loss.item() * self.config.gradient_accumulation_steps

                if (
                    step % self.config.gradient_accumulation_steps == 0
                    or step == len(train_loader)
                ):
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

            avg_train_loss = running_loss / max(1, len(train_loader))
            val_metrics = self.evaluate(val_loader)

            logger.info(
                "Epoch %d/%d - train_loss: %.4f - val_loss: %.4f - val_acc: %.4f",
                epoch + 1,
                self.config.num_epochs,
                avg_train_loss,
                val_metrics["loss"],
                val_metrics["accuracy"],
            )

            current_val_loss = val_metrics["loss"]
            improved = current_val_loss < (best_val_loss - self.config.early_stopping_min_delta)

            if improved:
                best_val_loss = current_val_loss
                best_val_accuracy = val_metrics["accuracy"]
                best_metrics = {
                    "train_loss": avg_train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "best_epoch": epoch + 1,
                }
                # 提前缓存最佳权重，便于早停后恢复到验证集表现最好的时刻。
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                epochs_without_improvement = 0
                logger.info("val_loss improved to %.4f at epoch %d", best_val_loss, epoch + 1)
            else:
                epochs_without_improvement += 1
                logger.info(
                    "val_loss did not improve for %d/%d epoch(s)",
                    epochs_without_improvement,
                    self.config.early_stopping_patience,
                )

                if epochs_without_improvement >= self.config.early_stopping_patience:
                    logger.info(
                        "Early stopping triggered at epoch %d because val_loss did not improve for %d consecutive epochs",
                        epoch + 1,
                        self.config.early_stopping_patience,
                    )
                    break

        if self.config.restore_best_weights and best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            logger.info("已恢复验证损失最低时的模型权重")

        logger.info(
            "最佳验证结果 - epoch: %s, val_loss: %.4f, val_acc: %.4f",
            best_metrics.get("best_epoch", "N/A"),
            best_val_loss,
            best_val_accuracy,
        )
        return best_metrics

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        predictions: List[int] = []
        references: List[int] = []

        for batch in dataloader:
            batch = {key: value.to(self.device) for key, value in batch.items()}
            outputs = self.model(**batch)
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.detach().cpu().tolist())
            references.extend(batch["labels"].detach().cpu().tolist())

        avg_loss = total_loss / max(1, len(dataloader))
        accuracy = accuracy_score(references, predictions) if references else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}

    @torch.no_grad()
    def predict(self, data: List[Dict[str, str]]) -> Tuple[List[str], List[float]]:
        texts = [item["text"] for item in data]
        dataloader = self.create_dataloader(
            texts=texts,
            labels=None,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
        )

        self.model.eval()
        all_labels: List[str] = []
        all_confidences: List[float] = []

        for batch in dataloader:
            batch = {key: value.to(self.device) for key, value in batch.items()}
            outputs = self.model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidence, pred_ids = torch.max(probs, dim=-1)

            all_labels.extend(self.id2label[idx] for idx in pred_ids.detach().cpu().tolist())
            all_confidences.extend(confidence.detach().cpu().tolist())

        return all_labels, all_confidences

    def save(self, output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(output_path / "label2id.json", "w", encoding="utf-8") as f:
            json.dump(self.label2id, f, ensure_ascii=False, indent=2)

        with open(output_path / "id2label.json", "w", encoding="utf-8") as f:
            json.dump(
                {str(key): value for key, value in self.id2label.items()},
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(output_path / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, ensure_ascii=False, indent=2)

        logger.info("Transformer 模型已保存到: %s", output_path)


def set_seed(seed: int) -> None:
    """统一设置随机种子，尽量降低训练结果的波动。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def step1_load_and_inspect(data_path):
    """
    加载原始训练数据并做基础结构检查。

    该步骤主要用于尽早暴露文件格式、字段缺失和空值问题，
    避免后续训练阶段才出现难以定位的异常。
    """
    logger.info("步骤1：加载并检查数据")

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.error("数据格式错误：根节点必须为列表")
            sys.exit(1)

        logger.info("加载数据成功: %s", data_path)
        logger.info("总数据量: %d 条", len(data))

        if data:
            logger.info("数据结构示例（第1条）:")
            first_item = data[0]
            if isinstance(first_item, dict):
                for key, value in first_item.items():
                    logger.info("  %s: %s", key, value)
            else:
                logger.warning("第1条数据不是对象类型: %s", type(first_item))

        logger.info("开始检查数据完整性...")
        missing_count = 0
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                logger.warning("第%d条数据不是对象类型", index + 1)
                missing_count += 1
                continue

            if "input" not in item or "label" not in item:
                logger.warning("第%d条数据缺失字段", index + 1)
                missing_count += 1
            elif not item.get("input") or not item.get("label"):
                logger.warning("第%d条数据字段为空", index + 1)
                missing_count += 1

        if missing_count == 0:
            logger.info("所有数据字段完整")
        else:
            logger.warning("发现 %d 条数据有问题", missing_count)

        return data

    except FileNotFoundError:
        logger.error("文件不存在: %s", data_path)
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error("JSON 解析失败: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("加载数据失败: %s", e)
        sys.exit(1)


def step2_check_text_length(data):
    """统计文本长度分布，用于评估截断风险与脏数据比例。"""
    logger.info("步骤2：检查文本长度")

    try:
        texts = [item.get("input", "") for item in data]
        lengths = [len(text) for text in texts]

        if not lengths:
            logger.warning("数据为空，无法检查文本长度")
            return data

        avg_length = sum(lengths) / len(lengths)
        min_length = min(lengths)
        max_length = max(lengths)

        logger.info("文本长度统计:")
        logger.info("  平均长度: %.1f 字符", avg_length)
        logger.info("  最短: %d 字符", min_length)
        logger.info("  最长: %d 字符", max_length)

        short_texts = [(index, text) for index, text in enumerate(texts) if len(text) < 3]

        if short_texts:
            logger.warning("发现 %d 条过短文本（<3字符）", len(short_texts))
            for index, text in short_texts[:5]:
                logger.info("  [%d] '%s'", index + 1, text)
            if len(short_texts) > 5:
                logger.info("  ... 还有 %d 条", len(short_texts) - 5)
        else:
            logger.info("没有发现过短文本")

        logger.info("长度分布:")
        ranges = [(0, 10), (10, 50), (50, 100), (100, 500), (500, float("inf"))]
        for start, end in ranges:
            count = sum(1 for length in lengths if start <= length < end)
            percentage = count / len(lengths) * 100
            logger.info(
                "  %s-%s 字符: %d 条 (%.1f%%)",
                start,
                end if end != float("inf") else "∞",
                count,
                percentage,
            )

    except Exception as e:
        logger.exception("检查文本长度时发生错误: %s", e)

    return data


def step3_check_duplicates(data):
    """检查重复文本，帮助评估数据冗余和潜在泄漏风险。"""
    logger.info("步骤3：检查重复数据")

    try:
        text_counts: Dict[str, int] = {}

        for item in data:
            text = item.get("input", "")
            if text in text_counts:
                text_counts[text] += 1
            else:
                text_counts[text] = 1

        duplicates = {text: count for text, count in text_counts.items() if count > 1}

        total_texts = len(data)
        unique_texts = len(text_counts)
        duplicate_rate = (total_texts - unique_texts) / total_texts * 100 if total_texts else 0.0

        if duplicates:
            logger.warning("发现重复数据")
            logger.info("  总数据量: %d 条", total_texts)
            logger.info("  唯一文本: %d 条", unique_texts)
            logger.info("  重复率: %.2f%%", duplicate_rate)

            logger.info("重复最多的文本（前5个）:")
            sorted_duplicates = sorted(duplicates.items(), key=lambda item: item[1], reverse=True)
            for text, count in sorted_duplicates[:5]:
                logger.info("  出现%d次: '%s...'", count, text[:50])
        else:
            logger.info("没有发现重复数据")

    except Exception as e:
        logger.exception("检查重复数据时发生错误: %s", e)

    return data


def step4_check_label_distribution(data):
    """统计标签分布，为后续是否做类别平衡提供依据。"""
    logger.info("步骤4：检查标签分布")

    try:
        labels = [item.get("label", "") for item in data]
        label_counts = Counter(labels)

        if not labels:
            logger.warning("数据为空，无法检查标签分布")
            return data

        total = len(labels)
        logger.info("标签分布:")
        for label, count in sorted(label_counts.items()):
            percentage = count / total * 100
            bar = "█" * int(percentage / 2)
            logger.info("  %-15s: %5d (%5.1f%%) %s", label, count, percentage, bar)

        min_count = min(label_counts.values())
        max_count = max(label_counts.values())
        balance_ratio = min_count / max_count if max_count else 0.0

        logger.info("平衡度分析:")
        logger.info("  最少类别: %d 条", min_count)
        logger.info("  最多类别: %d 条", max_count)
        logger.info("  平衡比例: %.2f", balance_ratio)

        if balance_ratio >= 0.8:
            logger.info("数据较为平衡")
        elif balance_ratio >= 0.5:
            logger.warning("数据轻度不平衡，建议平衡处理")
        else:
            logger.warning("数据严重不平衡，必须进行平衡处理")

    except Exception as e:
        logger.exception("检查标签分布时发生错误: %s", e)

    return data


def step5_check_label_consistency(data):
    """检查同一文本是否被标注为多个类别，以发现标注冲突。"""
    logger.info("步骤5：检查标签一致性")

    try:
        text_labels: Dict[str, List[str]] = {}
        for item in data:
            text = item.get("input", "")
            label = item.get("label", "")

            if text not in text_labels:
                text_labels[text] = []
            text_labels[text].append(label)

        conflicts = []
        for text, labels in text_labels.items():
            unique_labels = set(labels)
            if len(unique_labels) > 1:
                conflicts.append(
                    {
                        "text": text,
                        "labels": list(unique_labels),
                        "counts": Counter(labels),
                    }
                )

        if conflicts:
            logger.warning("发现 %d 条文本标签不一致", len(conflicts))
            for index, conflict in enumerate(conflicts[:5], start=1):
                logger.warning("  [%d] 文本: '%s...'", index, conflict["text"][:50])
                logger.warning("      标签: %s", conflict["labels"])
                logger.warning("      分布: %s", dict(conflict["counts"]))

            if len(conflicts) > 5:
                logger.warning("  ... 还有 %d 条冲突", len(conflicts) - 5)

            logger.warning("建议：人工检查这些样本，统一标签")
        else:
            logger.info("所有文本标签一致")

    except Exception as e:
        logger.exception("检查标签一致性时发生错误: %s", e)

    return data


def step6_remove_short_texts(data, min_length=3):
    """移除过短文本，减少无信息样本对模型训练的干扰。"""
    logger.info("步骤6：移除短于%d字符的文本", min_length)

    try:
        original_count = len(data)

        filtered_data = []
        removed_texts = []

        for item in data:
            text = item.get("input", "")
            if len(text) >= min_length:
                filtered_data.append(item)
            else:
                removed_texts.append(text)

        removed_count = original_count - len(filtered_data)
        removed_ratio = (removed_count / original_count * 100) if original_count else 0.0

        logger.info("原始数据: %d 条", original_count)
        logger.info("过滤后: %d 条", len(filtered_data))
        logger.info("移除: %d 条 (%.1f%%)", removed_count, removed_ratio)

        if removed_texts:
            logger.info("移除的文本示例（前5个）:")
            for text in removed_texts[:5]:
                logger.info("  '%s'", text)

        return filtered_data

    except Exception as e:
        logger.exception("移除短文本时发生错误: %s", e)
        return data


def step7_remove_duplicates(data):
    """按文本去重，避免重复样本过度影响训练分布。"""
    logger.info("步骤7：移除重复数据")

    try:
        original_count = len(data)

        seen_texts: Dict[str, bool] = {}
        dedup_data = []

        for item in data:
            text = item.get("input", "")
            if text not in seen_texts:
                seen_texts[text] = True
                dedup_data.append(item)

        removed_count = original_count - len(dedup_data)
        removed_ratio = (removed_count / original_count * 100) if original_count else 0.0

        logger.info("原始数据: %d 条", original_count)
        logger.info("去重后: %d 条", len(dedup_data))
        logger.info("移除: %d 条 (%.1f%%)", removed_count, removed_ratio)

        return dedup_data

    except Exception as e:
        logger.exception("移除重复数据时发生错误: %s", e)
        return data


def step8_balance_data(data, strategy="downsample"):
    """
    按类别做简单重采样平衡。

    说明：
        - downsample 通过裁剪多数类降低偏置；
        - upsample 通过复制少数类提升覆盖，但可能增加过拟合风险。
    """
    logger.info("步骤8：平衡数据（策略：%s）", strategy)

    try:
        label_groups: Dict[str, List[Dict[str, str]]] = {}
        for item in data:
            label = item.get("label", "")
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(item)

        logger.info("原始分布:")
        for label, items in sorted(label_groups.items()):
            logger.info("  %s: %d 条", label, len(items))

        if not label_groups:
            return data

        if strategy == "downsample":
            min_count = min(len(items) for items in label_groups.values())
            logger.info("下采样目标: 每类 %d 条", min_count)

            balanced_data = []
            for _, items in label_groups.items():
                random.shuffle(items)
                balanced_data.extend(items[:min_count])

        elif strategy == "upsample":
            max_count = max(len(items) for items in label_groups.values())
            logger.info("上采样目标: 每类 %d 条", max_count)

            balanced_data = []
            for _, items in label_groups.items():
                sampled_items = items.copy()
                while len(sampled_items) < max_count:
                    sampled_items.extend(
                        random.sample(items, min(len(items), max_count - len(sampled_items)))
                    )
                balanced_data.extend(sampled_items[:max_count])
        else:
            logger.warning("未知平衡策略: %s，跳过平衡", strategy)
            return data

        logger.info("平衡后分布:")
        balanced_labels = Counter(item.get("label", "") for item in balanced_data)
        for label, count in sorted(balanced_labels.items()):
            logger.info("  %s: %d 条", label, count)

        logger.info("总数据量: %d → %d", len(data), len(balanced_data))
        return balanced_data

    except Exception as e:
        logger.exception("平衡数据时发生错误: %s", e)
        return data


def load_and_prepare_data(data_path, min_length=3, balance_strategy="downsample"):
    """
    按既定清洗流程加载并标准化训练数据。

    返回结果统一为 {"text": ..., "label": ...} 结构，便于后续训练流程直接消费。
    """
    logger.info("=" * 60)
    logger.info("开始完整数据清洗流程")
    logger.info("=" * 60)

    try:
        data = step1_load_and_inspect(data_path)
        step2_check_text_length(data)
        step3_check_duplicates(data)
        step4_check_label_distribution(data)
        step5_check_label_consistency(data)

        logger.info("转换数据格式...")
        formatted_data = []
        for item in data:
            if not isinstance(item, dict):
                continue

            raw_text = item.get("input", "")
            raw_label = item.get("label", "")
            text = str(raw_text).strip() if raw_text is not None else ""
            label = str(raw_label).strip() if raw_label is not None else ""
            if text and label:
                formatted_data.append({"input": text, "label": label})

        logger.info("格式化后数据: %d 条", len(formatted_data))

        cleaned_data = step6_remove_short_texts(formatted_data, min_length=min_length)
        cleaned_data = step7_remove_duplicates(cleaned_data)
        cleaned_data = step8_balance_data(cleaned_data, strategy=balance_strategy)

        final_data = []
        for item in cleaned_data:
            final_data.append(
                {
                    "text": item["input"],
                    "label": item["label"],
                }
            )

        if not final_data:
            logger.error("清洗后数据为空，无法继续训练")
            sys.exit(1)

        logger.info("=" * 60)
        logger.info("数据清洗完成，最终数据量: %d 条", len(final_data))
        logger.info("=" * 60)

        return final_data

    except Exception:
        logger.exception("加载和准备数据时发生错误")
        sys.exit(1)


def create_stratified_splits(
    data: List[Dict[str, str]],
    test_size: float = 0.15,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    logger.info("划分数据集（test_size=%s, val_size=%s）", test_size, val_size)

    try:
        if not data:
            raise ValueError("数据为空，无法划分数据集")
        if not (0 < test_size < 1):
            raise ValueError("test_size 必须在 0 和 1 之间")
        if not (0 < val_size < 1):
            raise ValueError("val_size 必须在 0 和 1 之间")
        if test_size + val_size >= 1:
            raise ValueError("test_size + val_size 必须小于 1")

        texts = [item["text"] for item in data]
        labels = [item["label"] for item in data]
        label_counts = Counter(labels)
        min_label_count = min(label_counts.values()) if label_counts else 0

        use_stratify = min_label_count >= 2
        if not use_stratify:
            # 样本过少时 sklearn 分层切分会失败，此处主动降级为普通随机切分。
            logger.warning("部分类别样本数不足 2，降级为非分层划分")

        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_seed,
            stratify=labels if use_stratify else None,
        )

        adjusted_val_size = val_size / (1 - test_size)
        train_val_counts = Counter(train_val_labels)
        use_val_stratify = train_val_counts and min(train_val_counts.values()) >= 2
        if not use_val_stratify:
            # 第二次切分后也可能出现极小类别，同样需要保护性降级。
            logger.warning("验证集划分阶段部分类别样本数不足 2，降级为非分层划分")

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts,
            train_val_labels,
            test_size=adjusted_val_size,
            random_state=random_seed,
            stratify=train_val_labels if use_val_stratify else None,
        )

        train_data = [{"text": text, "label": label} for text, label in zip(train_texts, train_labels)]
        val_data = [{"text": text, "label": label} for text, label in zip(val_texts, val_labels)]
        test_data = [{"text": text, "label": label} for text, label in zip(test_texts, test_labels)]

        if not train_data or not val_data or not test_data:
            raise ValueError("训练/验证/测试集存在空集，无法继续训练")

        logger.info("训练集: %d 条", len(train_data))
        logger.info("验证集: %d 条", len(val_data))
        logger.info("测试集: %d 条", len(test_data))

        return train_data, val_data, test_data

    except Exception:
        logger.exception("划分数据集时发生错误")
        sys.exit(1)


def build_label_mappings(data: List[Dict[str, str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """根据训练样本中的标签构建双向映射。"""
    labels = sorted({item["label"] for item in data if item.get("label")})
    if not labels:
        raise ValueError("未找到有效标签，无法构建标签映射")

    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    logger.info("标签映射: %s", label2id)
    return label2id, id2label


def train_transformer_model(
    train_data: List[Dict[str, str]],
    val_data: List[Dict[str, str]],
    config: TransformerTrainingConfig,
) -> Tuple[ContentClassifier, Dict[str, float]]:
    logger.info("开始基于 Transformer 的文本分类训练")

    try:
        if not train_data:
            logger.error("训练数据为空，无法开始训练")
            sys.exit(1)
        if not val_data:
            logger.error("验证数据为空，无法开始训练")
            sys.exit(1)

        label2id, _ = build_label_mappings(train_data + val_data)
        trainer = TransformerTrainer(label2id=label2id, config=config)
        train_metrics = trainer.train(train_data=train_data, val_data=val_data)
        trainer.save(config.output_dir)

        classifier = ContentClassifier(
            model_path=config.output_dir,
            pretrained_model_name=config.pretrained_model_name,
            max_length=config.max_length,
            inference_batch_size=config.eval_batch_size,
        )
        return classifier, train_metrics

    except Exception:
        logger.exception("Transformer 训练失败")
        sys.exit(1)


def evaluate_on_test_set(classifier: ContentClassifier, test_data: List[Dict[str, str]]):
    """
    在测试集上评估模型。

    除整体准确率外，还会输出详细分类报告与误判样本，
    便于后续分析类别边界和清洗数据问题。
    """
    logger.info("测试集评估")

    try:
        if not test_data:
            logger.error("测试集为空，无法评估")
            sys.exit(1)

        test_texts = [item["text"] for item in test_data]
        test_labels = [item["label"] for item in test_data]

        predictions, confidences = classifier.predict_texts(
            test_texts,
            return_confidence=True,
            batch_size=classifier.inference_batch_size,
        )
        test_acc = accuracy_score(test_labels, predictions)

        logger.info("测试准确率: %.4f", test_acc)
        if confidences:
            logger.info("平均置信度: %.4f", float(np.mean(confidences)))

        logger.info("详细分类报告:")
        report = classification_report(test_labels, predictions, digits=4)
        for line in report.split("\n"):
            if line.strip():
                logger.info("  %s", line)

        errors = []
        for true_label, pred_label, text, confidence in zip(
            test_labels,
            predictions,
            test_texts,
            confidences,
        ):
            if true_label != pred_label:
                errors.append(
                    {
                        "text": text,
                        "true": true_label,
                        "pred": pred_label,
                        "confidence": confidence,
                    }
                )

        error_rate = len(errors) / len(test_labels) * 100 if test_labels else 0.0
        logger.info("错误样本数: %d/%d (%.2f%%)", len(errors), len(test_labels), error_rate)

        if errors:
            logger.info("前10个错误样本:")
            for index, err in enumerate(errors[:10], start=1):
                logger.info("  [%d] 文本: %s...", index, err["text"][:60])
                logger.info(
                    "      真实: %s | 预测: %s | 置信度: %.4f",
                    err["true"],
                    err["pred"],
                    err["confidence"],
                )

        return {
            "accuracy": test_acc,
            "errors": errors,
            "report": report,
        }

    except Exception:
        logger.exception("评估测试集时发生错误")
        sys.exit(1)


def main():
    """命令行训练入口：串联数据清洗、训练、评估与结果汇总。"""
    try:
        if len(sys.argv) < 2:
            logger.error("参数不足")
            logger.info("用法: python classifier_train.py <数据文件路径>")
            sys.exit(1)

        data_path = sys.argv[1]
        if not Path(data_path).exists():
            logger.error("数据文件不存在: %s", data_path)
            sys.exit(1)

        config = TransformerTrainingConfig()
        set_seed(config.random_seed)

        logger.info("=" * 60)
        logger.info("Transformer 文本分类训练 - 基于 xlm-roberta-large")
        logger.info("=" * 60)
        logger.info("训练配置: %s", asdict(config))

        logger.info("步骤 1: 完整数据清洗")
        clean_data = load_and_prepare_data(
            data_path,
            min_length=config.min_length,
            balance_strategy=config.balance_strategy,
        )

        logger.info("步骤 2: 划分训练/验证/测试集")
        train_data, val_data, test_data = create_stratified_splits(
            clean_data,
            test_size=config.test_size,
            val_size=config.val_size,
            random_seed=config.random_seed,
        )

        logger.info("步骤 3: Transformer 训练")
        classifier, train_metrics = train_transformer_model(
            train_data=train_data,
            val_data=val_data,
            config=config,
        )

        logger.info("步骤 4: 测试集评估")
        test_results = evaluate_on_test_set(classifier, test_data)

        logger.info("=" * 60)
        logger.info("训练总结")
        logger.info("=" * 60)
        logger.info("训练损失: %.4f", train_metrics.get("train_loss", 0.0))
        logger.info("验证损失: %.4f", train_metrics.get("val_loss", 0.0))
        logger.info("验证准确率: %.4f", train_metrics.get("val_accuracy", 0.0))
        logger.info("测试集准确率: %.4f", test_results["accuracy"])
        logger.info("模型目录: %s", config.output_dir)
        logger.info("=" * 60)

        if test_results["accuracy"] >= 0.98:
            logger.info("恭喜！模型准确率达到98%以上！")
        elif test_results["accuracy"] >= 0.95:
            logger.info("模型表现良好，准确率超过95%")
            logger.info("改进建议:")
            logger.info("  1. 增加更多训练数据")
            logger.info("  2. 分析错误样本，优化标注质量")
            logger.info("  3. 适当增大 max_length 或训练轮次")
        else:
            logger.info("当前准确率: %.2f%%", test_results["accuracy"] * 100)
            logger.info("改进建议:")
            logger.info("  1. 检查数据质量和标注一致性")
            logger.info("  2. 增加更多训练数据")
            logger.info("  3. 调整学习率、batch size、epoch 数")
            logger.info("  4. 针对长文本适度提高 max_length")

    except KeyboardInterrupt:
        logger.warning("用户中断操作")
        sys.exit(0)
    except Exception:
        logger.exception("程序执行失败")
        sys.exit(1)


if __name__ == "__main__":
    main()