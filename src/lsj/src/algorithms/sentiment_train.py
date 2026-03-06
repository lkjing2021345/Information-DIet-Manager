from __future__ import annotations

import json
import logging
import os
import pickle
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

# Keep mirror endpoint for regions with limited HF connectivity
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW
    from transformers import (
        BertForSequenceClassification,
        BertTokenizer,
        get_linear_schedule_with_warmup,
    )
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
    BERT_AVAILABLE = False


def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(level)

    if logger_obj.handlers:
        return logger_obj

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger_obj.addHandler(file_handler)
    logger_obj.addHandler(stream_handler)
    logger_obj.propagate = False
    return logger_obj


logger = setup_logger(__name__, "../../logs/sentiment_train.log")
MODEL_API_VERSION = "1.0"
DEFAULT_MODEL_OUTPUT_DIR = str(Path(__file__).resolve().parent / "models")


@dataclass
class TrainResult:
    model_name: str
    save_dir: str
    train_summary: Dict[str, Any]
    test_summary: Dict[str, Any]


@dataclass
class TrainConfig:
    text_column: str = "text"
    label_column: str = "sentiment"
    test_size: float = 0.15
    val_size: float = 0.15
    random_seed: int = 42

    model_name: str = "hfl/chinese-roberta-wwm-ext"
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    label_smoothing: float = 0.0

    model_output_name: str = "sentiment_train"


class SentimentTrainDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: Any, max_length: int):
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
    @abstractmethod
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
        raise NotImplementedError


class BasePredictor(ABC):
    @abstractmethod
    def evaluate_test(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        raise NotImplementedError


class SentimentTrainer(BaseTrainer, BasePredictor):

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
            torch.cuda.manual_seed_all(seed)

    def load_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        required = {cfg.text_column, cfg.label_column}
        missing = [c for c in required if c not in data.columns]
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
        unique_labels = sorted(set(labels))
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def _to_loader(self, df: pd.DataFrame, shuffle: bool) -> Any:
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
        if torch is None or F is None:
            raise RuntimeError("Torch runtime is unavailable.")
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call train(...) first.")
        model = self.model
        if not callable(model):
            raise RuntimeError("Model is not callable.")
        model.eval()
        preds, labels = [], []
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
            labels, preds, average="weighted", zero_division=0
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
        model_source = cfg.model_name

        tokenizer_common_kwargs: Dict[str, Any] = {}
        model_common_kwargs: Dict[str, Any] = {
            "num_labels": len(self.label2id),
            "id2label": self.id2label,
            "label2id": self.label2id,
            "use_safetensors": False,
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

        online_error: Optional[Exception] = None
        try:
            _load_from_source(model_source, local_only=False)
            logger.info("Loaded model/tokenizer from remote or cache: %s", model_source)
        except Exception as e:
            online_error = e
            logger.warning("Online model loading failed, switching to offline fallback. err=%r", e)

            local_candidates: List[str] = []
            if Path(model_source).exists():
                local_candidates.append(str(Path(model_source)))

            local_model_dir = os.getenv("HF_LOCAL_MODEL_DIR")
            if local_model_dir and Path(local_model_dir).exists():
                local_candidates.append(local_model_dir)

            loaded_offline = False
            last_offline_error: Optional[Exception] = None

            # fallback 1: explicit local paths
            for candidate in local_candidates:
                try:
                    _load_from_source(candidate, local_only=True)
                    logger.info("Loaded model/tokenizer from local path: %s", candidate)
                    loaded_offline = True
                    break
                except Exception as offline_e:
                    last_offline_error = offline_e
                    logger.warning("Offline loading from %s failed: %r", candidate, offline_e)

            # fallback 2: local HF cache only
            if not loaded_offline:
                try:
                    _load_from_source(model_source, local_only=True)
                    logger.info("Loaded model/tokenizer from local HF cache: %s", model_source)
                    loaded_offline = True
                except Exception as offline_cache_e:
                    last_offline_error = offline_cache_e

            if not loaded_offline:
                raise RuntimeError(
                    "Failed to load model in both online and offline modes. "
                    f"online_error={online_error!r}, offline_error={last_offline_error!r}. "
                    "You can set HF_TOKEN and optionally HF_LOCAL_MODEL_DIR to a local model directory."
                )

        train_loader = self._to_loader(train_df, shuffle=True)
        val_loader = self._to_loader(val_df, shuffle=False)

        if AdamW is None or get_linear_schedule_with_warmup is None or torch is None:
            raise RuntimeError("Torch optimizer/scheduler runtime is unavailable.")
        optimizer_cls = AdamW
        scheduler_factory = get_linear_schedule_with_warmup

        if self.model is None:
            raise RuntimeError("Model init failed.")
        model = self.model
        if not callable(model):
            raise RuntimeError("Model is not callable.")

        optimizer = optimizer_cls(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        total_steps = len(train_loader) * cfg.epochs
        warmup_steps = cfg.warmup_steps if cfg.warmup_steps > 0 else int(total_steps * cfg.warmup_ratio)
        warmup_steps = min(warmup_steps, total_steps)
        scheduler = scheduler_factory(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_val_loss = float("inf")
        best_state = None
        no_improve_epochs = 0
        history: List[Dict[str, Any]] = []

        for epoch in range(cfg.epochs):
            model.train()
            total_loss = 0.0

            for batch in train_loader:
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

            train_loss = total_loss / max(1, len(train_loader))
            val_metrics = self._evaluate_loader(val_loader)
            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_weighted": val_metrics["f1_weighted"],
            }
            history.append(epoch_record)
            logger.info("Epoch %d/%d => %s", epoch + 1, cfg.epochs, epoch_record)

            current_val_loss = float(val_metrics["loss"])
            if current_val_loss < (best_val_loss - cfg.early_stopping_min_delta):
                best_val_loss = current_val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
            model.load_state_dict(best_state)

        return {
            "best_val_loss": best_val_loss,
            "history": history,
        }

    def evaluate_test(self, test_df: pd.DataFrame) -> Dict[str, Any]:
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

    def save_artifacts(self, output_dir: str, training_summary: Dict[str, Any], test_summary: Dict[str, Any]) -> Path:
        cfg = self.config
        save_dir = Path(output_dir) / cfg.model_output_name
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model/tokenizer is not initialized. Train model before saving.")

        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        metadata = {
            "config": asdict(cfg),
            "label2id": self.label2id,
            "id2label": self.id2label,
            "training_summary": training_summary,
            "test_summary": test_summary,
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

        # Backward compatibility for sentiment.py loader (expects metadata.pkl)
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

    def save_model(self, output_dir: str, training_summary: Dict[str, Any], test_summary: Dict[str, Any]) -> Path:
        """Backward-compatible save entry for training module."""
        return self.save_artifacts(output_dir, training_summary, test_summary)


def run_training_pipeline(df: pd.DataFrame, output_dir: str = DEFAULT_MODEL_OUTPUT_DIR, config: Optional[TrainConfig] = None) -> Dict[str, Any]:
    """One-call API: clean -> split -> train -> evaluate -> save sentiment_train."""
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
    """Contract API: train(config_path, resume_from) -> TrainResult."""
    with open(config_path, "r", encoding="utf-8") as f:
        config_payload = json.load(f)

    data_path = Path(config_payload["data_path"])
    output_dir = config_payload.get("output_dir", DEFAULT_MODEL_OUTPUT_DIR)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    cfg_fields = {k: v for k, v in config_payload.items() if k in TrainConfig.__dataclass_fields__}
    cfg = TrainConfig(**cfg_fields)

    if resume_from is not None:
        cfg.model_name = str(resume_from)

    frame = pd.read_csv(data_path)
    if "label" in frame.columns and cfg.label_column not in frame.columns:
        frame = frame.rename(columns={"label": cfg.label_column})

    result = run_training_pipeline(frame, output_dir=output_dir, config=cfg)
    return TrainResult(**result)


def finetune(base_model_path: Path, new_data_path: Path, output_dir: Path) -> Path:
    """Contract API: finetune(base_model_path, new_data_path, output_dir) -> Path."""
    if not new_data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {new_data_path}")

    frame = pd.read_csv(new_data_path)
    if "label" in frame.columns and "sentiment" not in frame.columns:
        frame = frame.rename(columns={"label": "sentiment"})

    cfg = TrainConfig(model_name=str(base_model_path), model_output_name="sentiment_train")
    result = run_training_pipeline(frame, output_dir=str(output_dir), config=cfg)
    return Path(result["save_dir"])


if __name__ == "__main__":
    # Minimal CLI example
    csv_path = "../training_data/converted_dataset.csv"
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    if "label" in frame.columns and "sentiment" not in frame.columns:
        frame = frame.rename(columns={"label": "sentiment"})

    result = run_training_pipeline(frame, output_dir=DEFAULT_MODEL_OUTPUT_DIR)
    print(json.dumps(result, ensure_ascii=False, indent=2))
