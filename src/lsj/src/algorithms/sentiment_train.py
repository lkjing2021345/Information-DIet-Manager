from __future__ import annotations

"""
情感训练模块。

职责：
    封装基于 BERT 的情感分类训练、验证、测试评估与模型产物保存流程。

特点：
    - 提供从清洗数据到保存模型的一站式训练管线；
    - 支持在线加载与本地离线回退；
    - 兼容 sentiment.py 的旧版推理加载方式。
"""

import json
import os
import pickle
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover
    matplotlib = None
    plt = None
    MATPLOTLIB_AVAILABLE = False

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from utils.logger import setup_logger

# Windows 固定 Hugging Face 缓存目录，确保模型权重、配置和 tokenizer 文件持久化保存。
HF_HUB_CACHE_DIR = Path(r"C:\Users\Administrator\.cache\huggingface\hub")
HF_HOME_DIR = HF_HUB_CACHE_DIR.parent
HF_PERSISTENT_MODELS_DIR = HF_HUB_CACHE_DIR / "persistent_models"


def configure_huggingface_cache() -> Path:
    """统一设置 Hugging Face/Transformers 缓存目录到固定 Windows 路径。"""
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
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        BertForSequenceClassification,
        BertTokenizer,
        get_linear_schedule_with_warmup,
    )

    try:
        from huggingface_hub import snapshot_download
    except Exception:  # pragma: no cover
        snapshot_download = None

    BERT_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    F = None
    DataLoader = None
    Dataset = object
    AdamW = None
    BertForSequenceClassification = None
    BertTokenizer = None
    get_linear_schedule_with_warmup = None
    snapshot_download = None
    BERT_AVAILABLE = False

logger = setup_logger(__name__, "../../logs/sentiment_train.log")
MODEL_API_VERSION = "1.0"
DEFAULT_MODEL_OUTPUT_DIR = str(Path(__file__).resolve().parent / "models")
DEFAULT_TRAIN_VIS_DIRNAME = "train_vis"


def _sanitize_model_cache_name(model_name: str) -> str:
    """将 repo id/path 转为稳定目录名，避免 Windows 路径非法字符问题。"""
    return model_name.replace("\\", "__").replace("/", "__").replace(":", "_")



def ensure_model_cached(model_name_or_path: str, token: Optional[str] = None) -> str:
    """
    确保模型与 tokenizer 文件被持久化到固定 Hugging Face 缓存目录。

    - 若传入本地目录，则直接返回本地目录；
    - 若固定缓存目录下已存在完整快照，则优先复用；
    - 首次不存在时自动下载到固定目录，并保存后续复用所需文件。
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
    logger.info("Persistent Hugging Face cache miss, preparing download: %s", model_name_or_path)

    if snapshot_download is not None:
        snapshot_download(
            repo_id=model_name_or_path,
            cache_dir=str(HF_HUB_CACHE_DIR),
            local_dir=str(persistent_dir),
            local_dir_use_symlinks=False,
            token=token,
            resume_download=True,
        )
        logger.info("Model snapshot downloaded to persistent cache: %s", persistent_dir)
        return str(persistent_dir)

    # 兜底：若 huggingface_hub 不可用，则通过 transformers 下载后再显式保存。
    if BertTokenizer is None or BertForSequenceClassification is None:
        raise RuntimeError("huggingface_hub unavailable and transformers runtime incomplete.")

    tokenizer_kwargs: Dict[str, Any] = {"cache_dir": str(HF_HUB_CACHE_DIR)}
    model_kwargs: Dict[str, Any] = {"cache_dir": str(HF_HUB_CACHE_DIR)}
    if token:
        tokenizer_kwargs["token"] = token
        model_kwargs["token"] = token

    tokenizer = BertTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    model = BertForSequenceClassification.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer.save_pretrained(persistent_dir)
    model.save_pretrained(persistent_dir)
    logger.info("Model/tokenizer saved to persistent cache: %s", persistent_dir)
    return str(persistent_dir)


@dataclass
class TrainResult:
    """训练流程的统一返回结构。"""
    model_name: str
    save_dir: str
    train_summary: Dict[str, Any]
    test_summary: Dict[str, Any]


@dataclass
class TrainConfig:
    """训练配置，集中管理数据列、超参数与输出目录命名。"""
    text_column: str = "text"
    label_column: str = "sentiment"
    test_size: float = 0.15
    val_size: float = 0.15
    random_seed: int = 42

    model_name: str = "hfl/chinese-macbert-base"
    max_length: int = 128
    batch_size: int = 48
    epochs: int = 15
    learning_rate: float = 2.25e-5
    weight_decay: float = 0.03
    warmup_ratio: float = 0.05
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0005
    label_smoothing: float = 0.05

    model_output_name: str = "sentiment_train"


class SentimentTrainDataset(Dataset):
    """将文本与标签包装为适配 PyTorch DataLoader 的数据集。"""
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if torch is None:
            raise RuntimeError("Torch runtime is unavailable.")

        encoding = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BaseTrainer(ABC):
    """训练器抽象基类。"""
    @abstractmethod
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
        raise NotImplementedError


class BasePredictor(ABC):
    """评估器抽象基类。"""
    @abstractmethod
    def evaluate_test(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        raise NotImplementedError


class SentimentTrainer(BaseTrainer, BasePredictor):
    """BERT 情感分类训练器，负责完整训练生命周期管理。"""
    def __init__(self, config: Optional[TrainConfig] = None):
        if not BERT_AVAILABLE:
            raise ImportError("BERT dependencies unavailable. Install torch and transformers.")
        if torch is None or BertTokenizer is None or BertForSequenceClassification is None:
            raise ImportError("BERT runtime dependencies are not fully initialized.")

        self.config = config or TrainConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer: Any = None
        self.model: Any = None
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}

        self._set_seed(self.config.random_seed)
        logger.info("Trainer initialized. device=%s", self.device)

    @staticmethod
    def _set_seed(seed: int) -> None:
        if torch is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            # 同步设置所有 CUDA 设备随机种子，提升实验可复现性。
            torch.cuda.manual_seed_all(seed)

    def load_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """清洗原始训练数据，保留有效文本与标签并去重。"""
        cfg = self.config
        required = {cfg.text_column, cfg.label_column}
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = data[[cfg.text_column, cfg.label_column]].copy()
        df[cfg.text_column] = df[cfg.text_column].astype(str).str.strip()
        df[cfg.label_column] = df[cfg.label_column].astype(str).str.strip()

        df = df[df[cfg.text_column].str.len() > 0]
        df = df[df[cfg.label_column].str.len() > 0]
        df = df.drop_duplicates(subset=[cfg.text_column, cfg.label_column]).reset_index(drop=True)

        if df.empty:
            raise ValueError("No valid rows left after cleaning.")

        logger.info("Data cleaned. rows=%d", len(df))
        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """按配置将数据稳定拆分为训练、验证和测试集合。"""
        cfg = self.config
        train_df, temp_df = train_test_split(
            df,
            test_size=cfg.test_size + cfg.val_size,
            random_state=cfg.random_seed,
            stratify=df[cfg.label_column],
        )

        test_ratio_in_temp = cfg.test_size / (cfg.test_size + cfg.val_size)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_ratio_in_temp,
            random_state=cfg.random_seed,
            stratify=temp_df[cfg.label_column],
        )

        logger.info("Split done. train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))
        return train_df, val_df, test_df

    def _build_label_mapping(self, labels: List[str]) -> None:
        """建立标签到整数 ID 的双向映射。"""
        unique_labels = sorted(set(labels))
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def _to_loader(self, df: pd.DataFrame, shuffle: bool) -> Any:
        """将 DataFrame 转换为 DataLoader，供训练或评估阶段使用。"""
        cfg = self.config
        dataset = SentimentTrainDataset(
            texts=df[cfg.text_column].tolist(),
            labels=[self.label2id[label] for label in df[cfg.label_column].tolist()],
            tokenizer=self.tokenizer,
            max_length=cfg.max_length,
        )
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

    def _evaluate_loader(self, data_loader: Any) -> Dict[str, Any]:
        """在给定数据加载器上执行前向评估并汇总核心指标。"""
        if torch is None or F is None:
            raise RuntimeError("Torch runtime is unavailable.")
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call train(...) first.")

        model = self.model
        if not callable(model):
            raise RuntimeError("Model is not callable.")

        model.eval()
        preds: List[int] = []
        labels: List[int] = []
        total_loss = 0.0

        with torch.inference_mode():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                y = batch["label"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(
                    outputs.logits,
                    y,
                    label_smoothing=max(0.0, float(self.config.label_smoothing)),
                )
                total_loss += float(loss.item())
                pred = torch.argmax(outputs.logits, dim=1)

                preds.extend(pred.cpu().tolist())
                labels.extend(y.cpu().tolist())

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="weighted",
            zero_division=0,
        )
        return {
            "loss": total_loss / max(1, len(data_loader)),
            "accuracy": accuracy_score(labels, preds),
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
            "pred_ids": preds,
            "true_ids": labels,
        }

    def _save_training_visualizations(
        self,
        save_dir: Path,
        history: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """保存训练历史及可选的损失/精度曲线图。"""
        vis_dir = save_dir / DEFAULT_TRAIN_VIS_DIRNAME
        vis_dir.mkdir(parents=True, exist_ok=True)

        history_df = pd.DataFrame(history)
        metrics_csv_path = vis_dir / "metrics.csv"
        metrics_json_path = vis_dir / "metrics.json"
        history_df.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")
        history_df.to_json(metrics_json_path, orient="records", force_ascii=False, indent=2)

        saved_files: Dict[str, str] = {
            "metrics_csv": str(metrics_csv_path),
            "metrics_json": str(metrics_json_path),
        }

        if not MATPLOTLIB_AVAILABLE or plt is None:
            logger.warning("matplotlib is unavailable, skipping curve plotting.")
            return saved_files

        epochs = history_df["epoch"].tolist()

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history_df["train_loss"].tolist(), marker="o", label="Train Loss")
        plt.plot(epochs, history_df["val_loss"].tolist(), marker="o", label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        loss_curve_path = vis_dir / "loss_curve.png"
        plt.savefig(loss_curve_path, dpi=150)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history_df["train_accuracy"].tolist(), marker="o", label="Train Accuracy")
        plt.plot(epochs, history_df["val_accuracy"].tolist(), marker="o", label="Validation Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        metric_curve_path = vis_dir / "metric_curve.png"
        plt.savefig(metric_curve_path, dpi=150)
        plt.close()

        saved_files["loss_curve"] = str(loss_curve_path)
        saved_files["metric_curve"] = str(metric_curve_path)
        logger.info("Training visualizations saved to: %s", vis_dir)
        return saved_files

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
        if torch is None or F is None:
            raise RuntimeError("Torch runtime is unavailable.")

        cfg = self.config
        if not (0.0 <= float(cfg.label_smoothing) < 1.0):
            raise ValueError(f"label_smoothing must be in [0.0, 1.0), got {cfg.label_smoothing}")
        if cfg.max_grad_norm < 0:
            raise ValueError(f"max_grad_norm must be >= 0, got {cfg.max_grad_norm}")
        if cfg.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {cfg.warmup_steps}")

        self._build_label_mapping(train_df[cfg.label_column].tolist())

        if BertTokenizer is None or BertForSequenceClassification is None:
            raise RuntimeError("Transformers runtime is unavailable.")

        tokenizer_cls = BertTokenizer
        model_cls = BertForSequenceClassification

        hf_token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
        )
        model_source = ensure_model_cached(cfg.model_name, token=hf_token)

        tokenizer_common_kwargs: Dict[str, Any] = {
            "cache_dir": str(HF_HUB_CACHE_DIR),
        }
        model_common_kwargs: Dict[str, Any] = {
            "num_labels": len(self.label2id),
            "id2label": self.id2label,
            "label2id": self.label2id,
            "use_safetensors": False,
            "cache_dir": str(HF_HUB_CACHE_DIR),
        }
        if hf_token:
            tokenizer_common_kwargs["token"] = hf_token
            model_common_kwargs["token"] = hf_token

        def _load_from_source(source: str, local_only: bool) -> None:
            tokenizer_kwargs = dict(tokenizer_common_kwargs)
            tokenizer_kwargs["local_files_only"] = local_only
            model_kwargs = dict(model_common_kwargs)
            model_kwargs["local_files_only"] = local_only

            self.tokenizer = tokenizer_cls.from_pretrained(source, **tokenizer_kwargs)
            self.model = model_cls.from_pretrained(source, **model_kwargs).to(self.device)

        try:
            # 优先从固定持久化缓存目录加载，避免后续重复联网下载。
            _load_from_source(model_source, local_only=True)
            logger.info("Loaded model/tokenizer from persisted local cache: %s", model_source)
        except Exception as local_error:
            logger.warning("Local cache loading failed, fallback to hub cache download. err=%r", local_error)
            try:
                _load_from_source(cfg.model_name, local_only=False)
                logger.info("Loaded model/tokenizer from remote or hub cache: %s", cfg.model_name)
            except Exception as online_error:
                raise RuntimeError(
                    "Failed to load model from both persisted cache and remote source. "
                    f"local_error={local_error!r}, online_error={online_error!r}"
                )

        train_loader = self._to_loader(train_df, shuffle=True)
        val_loader = self._to_loader(val_df, shuffle=False)

        if AdamW is None or get_linear_schedule_with_warmup is None or torch is None:
            raise RuntimeError("Torch optimizer/scheduler runtime is unavailable.")
        if self.model is None:
            raise RuntimeError("Model init failed.")

        model = self.model
        if not callable(model):
            raise RuntimeError("Model is not callable.")

        optimizer = AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        total_steps = len(train_loader) * cfg.epochs
        warmup_steps = cfg.warmup_steps if cfg.warmup_steps > 0 else int(total_steps * cfg.warmup_ratio)
        # warmup 步数支持显式指定；若未指定，则按总步数比例自动推导。
        warmup_steps = min(warmup_steps, total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        use_amp = self.device.type == "cuda"
        # 仅在 CUDA 上启用混合精度，兼顾训练速度与显存占用。
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_val_loss = float("inf")
        best_state = None
        no_improve_epochs = 0
        history: List[Dict[str, Any]] = []

        for epoch in range(cfg.epochs):
            model.train()
            total_loss = 0.0
            train_preds: List[int] = []
            train_labels: List[int] = []

            for batch in train_loader:
                # 标准训练步骤：前向、反向、梯度裁剪、优化器更新、调度器更新。
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                y = batch["label"].to(self.device)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = F.cross_entropy(
                        outputs.logits,
                        y,
                        label_smoothing=float(cfg.label_smoothing),
                    )

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.max_grad_norm))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += float(loss.item())
                batch_pred = torch.argmax(outputs.logits.detach(), dim=1)
                train_preds.extend(batch_pred.cpu().tolist())
                train_labels.extend(y.detach().cpu().tolist())

            train_loss = total_loss / max(1, len(train_loader))
            train_accuracy = accuracy_score(train_labels, train_preds) if train_labels else 0.0
            val_metrics = self._evaluate_loader(val_loader)
            current_lr = float(optimizer.param_groups[0]["lr"])

            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "train_accuracy": train_accuracy,
                "val_accuracy": val_metrics["accuracy"],
                "train_metric": train_accuracy,
                "val_metric": val_metrics["accuracy"],
                "val_f1_weighted": val_metrics["f1_weighted"],
                "learning_rate": current_lr,
            }
            history.append(epoch_record)

            logger.info(
                "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f | train_accuracy=%.6f | val_accuracy=%.6f | lr=%.10f",
                epoch + 1,
                cfg.epochs,
                train_loss,
                float(val_metrics["loss"]),
                train_accuracy,
                float(val_metrics["accuracy"]),
                current_lr,
            )

            current_val_loss = float(val_metrics["loss"])
            if current_val_loss < (best_val_loss - cfg.early_stopping_min_delta):
                # 仅在验证损失出现有效改善时刷新最优权重。
                best_val_loss = current_val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if cfg.early_stopping_patience > 0 and no_improve_epochs >= cfg.early_stopping_patience:
                    logger.info(
                        "Early stopping triggered at epoch %d (best_val_loss=%.6f, patience=%d, min_delta=%.6f)",
                        epoch + 1,
                        best_val_loss,
                        cfg.early_stopping_patience,
                        cfg.early_stopping_min_delta,
                    )
                    break

        if best_state is not None:
            # 训练结束后恢复到验证集表现最好的参数，而不是最后一个 epoch。
            model.load_state_dict(best_state)

        return {
            "best_val_loss": best_val_loss,
            "history": history,
            "visualization_metric_name": "accuracy",
        }

    def evaluate_test(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """在测试集上评估最终模型，并生成分类报告与混淆矩阵。"""
        test_loader = self._to_loader(test_df, shuffle=False)
        metrics = self._evaluate_loader(test_loader)

        pred_labels = [self.id2label[idx] for idx in metrics["pred_ids"]]
        true_labels = [self.id2label[idx] for idx in metrics["true_ids"]]

        report = classification_report(true_labels, pred_labels, digits=4)
        cm = confusion_matrix(true_labels, pred_labels, labels=sorted(self.label2id.keys()))

        return {
            "accuracy": metrics["accuracy"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
            "f1_weighted": metrics["f1_weighted"],
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "labels": sorted(self.label2id.keys()),
        }

    def save_artifacts(
        self,
        output_dir: str,
        training_summary: Dict[str, Any],
        test_summary: Dict[str, Any],
    ) -> Path:
        """保存模型权重、配置、评估结果和兼容性元数据。"""
        cfg = self.config
        save_dir = Path(output_dir) / cfg.model_output_name
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model/tokenizer is not initialized. Train model before saving.")

        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        visualization_files: Dict[str, str] = {}
        history = training_summary.get("history", [])
        if history:
            visualization_files = self._save_training_visualizations(save_dir, history)

        metadata = {
            "config": asdict(cfg),
            "label2id": self.label2id,
            "id2label": self.id2label,
            "training_summary": training_summary,
            "test_summary": test_summary,
            "visualization_files": visualization_files,
            "api_version": MODEL_API_VERSION,
        }
        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        with open(save_dir / "train_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

        model_card = (
            f"# sentiment_train\n\n"
            f"- api_version: {MODEL_API_VERSION}\n"
            f"- base_model: {cfg.model_name}\n"
            f"- batch_size: {cfg.batch_size}\n"
            f"- epochs: {cfg.epochs}\n"
            f"- learning_rate: {cfg.learning_rate}\n"
        )
        with open(save_dir / "model_card.md", "w", encoding="utf-8") as f:
            f.write(model_card)

        # 为旧版 sentiment.py 推理加载逻辑额外写出 metadata.pkl。
        try:
            from sklearn.preprocessing import LabelEncoder

            encoder = LabelEncoder()
            encoder.fit(sorted(self.label2id.keys()))
            pkl_metadata = {
                "model_type": "BERT",
                "label_encoder": encoder,
                "bert_model_name": cfg.model_name,
            }
            with open(save_dir / "metadata.pkl", "wb") as f:
                pickle.dump(pkl_metadata, f)
        except Exception as e:
            logger.warning("Failed to save metadata.pkl compatibility file: %s", e)

        logger.info("Model artifacts saved to: %s", save_dir)
        return save_dir

    def save_model(
        self,
        output_dir: str,
        training_summary: Dict[str, Any],
        test_summary: Dict[str, Any],
    ) -> Path:
        """Backward-compatible save entry for training module."""
        return self.save_artifacts(output_dir, training_summary, test_summary)


def run_training_pipeline(
    df: pd.DataFrame,
    output_dir: str = DEFAULT_MODEL_OUTPUT_DIR,
    config: Optional[TrainConfig] = None,
) -> Dict[str, Any]:
    """一站式训练入口：清洗、切分、训练、测试评估并保存产物。"""
    trainer = SentimentTrainer(config=config)
    clean_df = trainer.load_and_clean_data(df)
    train_df, val_df, test_df = trainer.split_data(clean_df)
    train_summary = trainer.train(train_df, val_df)
    test_summary = trainer.evaluate_test(test_df)
    save_dir = trainer.save_model(output_dir, train_summary, test_summary)

    final_result = TrainResult(
        model_name=trainer.config.model_output_name,
        save_dir=str(save_dir),
        train_summary=train_summary,
        test_summary=test_summary,
    )
    logger.info("Training pipeline finished. test_f1=%.4f", test_summary["f1_weighted"])
    return asdict(final_result)


def train(config_path: Path, resume_from: Optional[Path] = None) -> TrainResult:
    """契约式训练入口：从配置文件读取参数并返回结构化训练结果。"""
    with open(config_path, "r", encoding="utf-8") as f:
        config_payload = json.load(f)

    data_path = Path(config_payload["data_path"])
    output_dir = config_payload.get("output_dir", DEFAULT_MODEL_OUTPUT_DIR)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    cfg_fields = {key: value for key, value in config_payload.items() if key in TrainConfig.__dataclass_fields__}
    cfg = TrainConfig(**cfg_fields)

    if resume_from is not None:
        cfg.model_name = str(resume_from)

    frame = pd.read_csv(data_path)
    if "label" in frame.columns and cfg.label_column not in frame.columns:
        frame = frame.rename(columns={"label": cfg.label_column})

    result = run_training_pipeline(frame, output_dir=output_dir, config=cfg)
    return TrainResult(**result)


def finetune(base_model_path: Path, new_data_path: Path, output_dir: Path) -> Path:
    """微调入口：以已有模型为起点，在新数据上继续训练并输出新模型。"""
    if not new_data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {new_data_path}")

    frame = pd.read_csv(new_data_path)
    if "label" in frame.columns and "sentiment" not in frame.columns:
        frame = frame.rename(columns={"label": "sentiment"})

    cfg = TrainConfig(model_name=str(base_model_path), model_output_name="sentiment_train")
    result = run_training_pipeline(frame, output_dir=str(output_dir), config=cfg)
    return Path(result["save_dir"])


if __name__ == "__main__":
    # 最小可运行示例：读取 CSV 后直接执行完整训练流水线
    csv_path = "../training_data/converted_dataset.csv"
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    if "label" in frame.columns and "sentiment" not in frame.columns:
        frame = frame.rename(columns={"label": "sentiment"})

    result = run_training_pipeline(frame, output_dir=DEFAULT_MODEL_OUTPUT_DIR)
    print(json.dumps(result, ensure_ascii=False, indent=2))