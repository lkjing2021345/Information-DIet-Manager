import json
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


@dataclass
class TransformerTrainingConfig:
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
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.pretrained_model_name,
            num_labels=len(label2id),
            id2label={int(k): v for k, v in self.id2label.items()},
            label2id=label2id,
        )
        self.model.to(self.device)
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8 if self.device.type == "cuda" else None,
        )
        self.use_amp = self.device.type == "cuda" and config.use_fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        logger.info(f"训练设备: {self.device}")
        logger.info(f"预训练模型: {config.pretrained_model_name}")
        logger.info(f"AMP 混合精度: {'开启' if self.use_amp else '关闭'}")

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
            pin_memory=self.device.type == "cuda",
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
            (len(train_loader) * self.config.num_epochs)
            // self.config.gradient_accumulation_steps,
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
        best_metrics = {}
        best_state_dict = None
        epochs_without_improvement = 0

        logger.info("开始 Transformer 训练")
        logger.info(f"训练样本数: {len(train_data)}")
        logger.info(f"验证样本数: {len(val_data)}")
        logger.info(f"训练步数: {total_train_steps}")
        logger.info(f"Warmup 步数: {warmup_steps}")
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
                batch = {k: v.to(self.device) for k, v in batch.items()}

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
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                epochs_without_improvement = 0
                logger.info(
                    "val_loss improved to %.4f at epoch %d",
                    best_val_loss,
                    epoch + 1,
                )
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
        predictions = []
        references = []

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
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
        all_labels = []
        all_confidences = []

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
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
            json.dump({str(k): v for k, v in self.id2label.items()}, f, ensure_ascii=False, indent=2)

        with open(output_path / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, ensure_ascii=False, indent=2)

        logger.info(f"Transformer 模型已保存到: {output_path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def step1_load_and_inspect(data_path):
    """
    步骤1：加载数据并进行初步检查

    学习目标：
    - 了解数据的基本结构
    - 发现明显的问题
    """
    logger.info("步骤1：加载并检查数据")

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.error("数据格式错误：根节点必须为列表")
            sys.exit(1)

        logger.info(f"加载数据成功: {data_path}")
        logger.info(f"总数据量: {len(data)} 条")

        if data:
            logger.info("数据结构示例（第1条）:")
            first_item = data[0]
            if isinstance(first_item, dict):
                for key, value in first_item.items():
                    logger.info(f"  {key}: {value}")
            else:
                logger.warning(f"第1条数据不是对象类型: {type(first_item)}")

        logger.info("开始检查数据完整性...")
        missing_count = 0
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                logger.warning(f"第{i + 1}条数据不是对象类型")
                missing_count += 1
                continue

            if 'input' not in item or 'label' not in item:
                logger.warning(f"第{i + 1}条数据缺失字段")
                missing_count += 1
            elif not item.get('input') or not item.get('label'):
                logger.warning(f"第{i + 1}条数据字段为空")
                missing_count += 1

        if missing_count == 0:
            logger.info("所有数据字段完整")
        else:
            logger.warning(f"发现 {missing_count} 条数据有问题")

        return data

    except FileNotFoundError:
        logger.error(f"文件不存在: {data_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"加载数据失败: {e}")
        sys.exit(1)


def step2_check_text_length(data):
    logger.info("步骤2：检查文本长度")

    try:
        texts = [item.get('input', '') for item in data]
        lengths = [len(text) for text in texts]

        if not lengths:
            logger.warning("数据为空，无法检查文本长度")
            return data

        avg_length = sum(lengths) / len(lengths)
        min_length = min(lengths)
        max_length = max(lengths)

        logger.info("文本长度统计:")
        logger.info(f"  平均长度: {avg_length:.1f} 字符")
        logger.info(f"  最短: {min_length} 字符")
        logger.info(f"  最长: {max_length} 字符")

        short_texts = [(i, text) for i, text in enumerate(texts) if len(text) < 3]

        if short_texts:
            logger.warning(f"发现 {len(short_texts)} 条过短文本（<3字符）")
            for i, text in short_texts[:5]:
                logger.info(f"  [{i + 1}] '{text}'")
            if len(short_texts) > 5:
                logger.info(f"  ... 还有 {len(short_texts) - 5} 条")
        else:
            logger.info("没有发现过短文本")

        logger.info("长度分布:")
        ranges = [(0, 10), (10, 50), (50, 100), (100, 500), (500, float('inf'))]
        for start, end in ranges:
            count = sum(1 for l in lengths if start <= l < end)
            percentage = count / len(lengths) * 100
            logger.info(f"  {start}-{end if end != float('inf') else '∞'} 字符: {count} 条 ({percentage:.1f}%)")

    except Exception as e:
        logger.exception(f"检查文本长度时发生错误: {e}")

    return data


def step3_check_duplicates(data):
    logger.info("步骤3：检查重复数据")

    try:
        text_counts = {}

        for item in data:
            text = item.get('input', '')
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
            logger.info(f"  总数据量: {total_texts} 条")
            logger.info(f"  唯一文本: {unique_texts} 条")
            logger.info(f"  重复率: {duplicate_rate:.2f}%")

            logger.info("重复最多的文本（前5个）:")
            sorted_dups = sorted(duplicates.items(), key=lambda x: x[1], reverse=True)
            for text, count in sorted_dups[:5]:
                logger.info(f"  出现{count}次: '{text[:50]}...'")
        else:
            logger.info("没有发现重复数据")

    except Exception as e:
        logger.exception(f"检查重复数据时发生错误: {e}")

    return data


def step4_check_label_distribution(data):
    logger.info("步骤4：检查标签分布")

    try:
        labels = [item.get('label', '') for item in data]
        label_counts = Counter(labels)

        if not labels:
            logger.warning("数据为空，无法检查标签分布")
            return data

        total = len(labels)
        logger.info("标签分布:")
        for label, count in sorted(label_counts.items()):
            percentage = count / total * 100
            bar = "█" * int(percentage / 2)
            logger.info(f"  {label:15s}: {count:5d} ({percentage:5.1f}%) {bar}")

        min_count = min(label_counts.values())
        max_count = max(label_counts.values())
        balance_ratio = min_count / max_count if max_count else 0.0

        logger.info("平衡度分析:")
        logger.info(f"  最少类别: {min_count} 条")
        logger.info(f"  最多类别: {max_count} 条")
        logger.info(f"  平衡比例: {balance_ratio:.2f}")

        if balance_ratio >= 0.8:
            logger.info("数据较为平衡")
        elif balance_ratio >= 0.5:
            logger.warning("数据轻度不平衡，建议平衡处理")
        else:
            logger.warning("数据严重不平衡，必须进行平衡处理")

    except Exception as e:
        logger.exception(f"检查标签分布时发生错误: {e}")

    return data


def step5_check_label_consistency(data):
    logger.info("步骤5：检查标签一致性")

    try:
        text_labels = {}
        for item in data:
            text = item.get('input', '')
            label = item.get('label', '')

            if text not in text_labels:
                text_labels[text] = []
            text_labels[text].append(label)

        conflicts = []
        for text, labels in text_labels.items():
            unique_labels = set(labels)
            if len(unique_labels) > 1:
                conflicts.append({
                    'text': text,
                    'labels': list(unique_labels),
                    'counts': Counter(labels)
                })

        if conflicts:
            logger.warning(f"发现 {len(conflicts)} 条文本标签不一致")
            for i, conflict in enumerate(conflicts[:5], 1):
                logger.warning(f"  [{i}] 文本: '{conflict['text'][:50]}...'")
                logger.warning(f"      标签: {conflict['labels']}")
                logger.warning(f"      分布: {dict(conflict['counts'])}")

            if len(conflicts) > 5:
                logger.warning(f"  ... 还有 {len(conflicts) - 5} 条冲突")

            logger.warning("建议：人工检查这些样本，统一标签")
        else:
            logger.info("所有文本标签一致")

    except Exception as e:
        logger.exception(f"检查标签一致性时发生错误: {e}")

    return data


def step6_remove_short_texts(data, min_length=3):
    logger.info(f"步骤6：移除短于{min_length}字符的文本")

    try:
        original_count = len(data)

        filtered_data = []
        removed_texts = []

        for item in data:
            text = item.get('input', '')
            if len(text) >= min_length:
                filtered_data.append(item)
            else:
                removed_texts.append(text)

        removed_count = original_count - len(filtered_data)
        removed_ratio = (removed_count / original_count * 100) if original_count else 0.0

        logger.info(f"原始数据: {original_count} 条")
        logger.info(f"过滤后: {len(filtered_data)} 条")
        logger.info(f"移除: {removed_count} 条 ({removed_ratio:.1f}%)")

        if removed_texts:
            logger.info("移除的文本示例（前5个）:")
            for text in removed_texts[:5]:
                logger.info(f"  '{text}'")

        return filtered_data

    except Exception as e:
        logger.exception(f"移除短文本时发生错误: {e}")
        return data


def step7_remove_duplicates(data):
    logger.info("步骤7：移除重复数据")

    try:
        original_count = len(data)

        seen_texts = {}
        dedup_data = []

        for item in data:
            text = item.get('input', '')
            if text not in seen_texts:
                seen_texts[text] = True
                dedup_data.append(item)

        removed_count = original_count - len(dedup_data)
        removed_ratio = (removed_count / original_count * 100) if original_count else 0.0

        logger.info(f"原始数据: {original_count} 条")
        logger.info(f"去重后: {len(dedup_data)} 条")
        logger.info(f"移除: {removed_count} 条 ({removed_ratio:.1f}%)")

        return dedup_data

    except Exception as e:
        logger.exception(f"移除重复数据时发生错误: {e}")
        return data


def step8_balance_data(data, strategy='downsample'):
    logger.info(f"步骤8：平衡数据（策略：{strategy}）")

    try:
        label_groups = {}
        for item in data:
            label = item.get('label', '')
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(item)

        logger.info("原始分布:")
        for label, items in sorted(label_groups.items()):
            logger.info(f"  {label}: {len(items)} 条")

        if not label_groups:
            return data

        if strategy == 'downsample':
            min_count = min(len(items) for items in label_groups.values())
            logger.info(f"下采样目标: 每类 {min_count} 条")

            balanced_data = []
            for label, items in label_groups.items():
                random.shuffle(items)
                balanced_data.extend(items[:min_count])

        elif strategy == 'upsample':
            max_count = max(len(items) for items in label_groups.values())
            logger.info(f"上采样目标: 每类 {max_count} 条")

            balanced_data = []
            for label, items in label_groups.items():
                sampled_items = items.copy()
                while len(sampled_items) < max_count:
                    sampled_items.extend(random.sample(items, min(len(items), max_count - len(sampled_items))))
                balanced_data.extend(sampled_items[:max_count])
        else:
            logger.warning(f"未知平衡策略: {strategy}，跳过平衡")
            return data

        logger.info("平衡后分布:")
        balanced_labels = Counter(item.get('label', '') for item in balanced_data)
        for label, count in sorted(balanced_labels.items()):
            logger.info(f"  {label}: {count} 条")

        logger.info(f"总数据量: {len(data)} → {len(balanced_data)}")

        return balanced_data

    except Exception as e:
        logger.exception(f"平衡数据时发生错误: {e}")
        return data


def load_and_prepare_data(data_path, min_length=3, balance_strategy='downsample'):
    """
    加载并准备训练数据（使用现有清洗流程）

    Args:
        data_path: 数据文件路径
        min_length: 最小文本长度
        balance_strategy: 数据平衡策略 ('downsample' 或 'upsample')

    Returns:
        清洗后的数据列表
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

            raw_text = item.get('input', '')
            raw_label = item.get('label', '')
            text = str(raw_text).strip() if raw_text is not None else ''
            label = str(raw_label).strip() if raw_label is not None else ''
            if text and label:
                formatted_data.append({'input': text, 'label': label})

        logger.info(f"格式化后数据: {len(formatted_data)} 条")

        cleaned_data = step6_remove_short_texts(formatted_data, min_length=min_length)
        cleaned_data = step7_remove_duplicates(cleaned_data)
        cleaned_data = step8_balance_data(cleaned_data, strategy=balance_strategy)

        final_data = []
        for item in cleaned_data:
            final_data.append({
                'text': item['input'],
                'label': item['label']
            })

        if not final_data:
            logger.error("清洗后数据为空，无法继续训练")
            sys.exit(1)

        logger.info("=" * 60)
        logger.info(f"数据清洗完成，最终数据量: {len(final_data)} 条")
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
    logger.info(f"划分数据集（test_size={test_size}, val_size={val_size}）")

    try:
        if not data:
            raise ValueError("数据为空，无法划分数据集")
        if not (0 < test_size < 1):
            raise ValueError("test_size 必须在 0 和 1 之间")
        if not (0 < val_size < 1):
            raise ValueError("val_size 必须在 0 和 1 之间")
        if test_size + val_size >= 1:
            raise ValueError("test_size + val_size 必须小于 1")

        texts = [item['text'] for item in data]
        labels = [item['label'] for item in data]
        label_counts = Counter(labels)
        min_label_count = min(label_counts.values()) if label_counts else 0

        use_stratify = min_label_count >= 2
        if not use_stratify:
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
            logger.warning("验证集划分阶段部分类别样本数不足 2，降级为非分层划分")

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts,
            train_val_labels,
            test_size=adjusted_val_size,
            random_state=random_seed,
            stratify=train_val_labels if use_val_stratify else None,
        )

        train_data = [{'text': text, 'label': label} for text, label in zip(train_texts, train_labels)]
        val_data = [{'text': text, 'label': label} for text, label in zip(val_texts, val_labels)]
        test_data = [{'text': text, 'label': label} for text, label in zip(test_texts, test_labels)]

        if not train_data or not val_data or not test_data:
            raise ValueError("训练/验证/测试集存在空集，无法继续训练")

        logger.info(f"训练集: {len(train_data)} 条")
        logger.info(f"验证集: {len(val_data)} 条")
        logger.info(f"测试集: {len(test_data)} 条")

        return train_data, val_data, test_data

    except Exception:
        logger.exception("划分数据集时发生错误")
        sys.exit(1)


def build_label_mappings(data: List[Dict[str, str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted({item['label'] for item in data if item.get('label')})
    if not labels:
        raise ValueError("未找到有效标签，无法构建标签映射")
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    logger.info(f"标签映射: {label2id}")
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
    在测试集上评估模型
    """
    logger.info("测试集评估")

    try:
        if not test_data:
            logger.error("测试集为空，无法评估")
            sys.exit(1)

        test_texts = [item['text'] for item in test_data]
        test_labels = [item['label'] for item in test_data]

        predictions, confidences = classifier.predict_texts(
            test_texts,
            return_confidence=True,
            batch_size=classifier.inference_batch_size,
        )
        test_acc = accuracy_score(test_labels, predictions)

        logger.info(f"测试准确率: {test_acc:.4f}")
        if confidences:
            logger.info(f"平均置信度: {float(np.mean(confidences)):.4f}")

        logger.info("详细分类报告:")
        report = classification_report(test_labels, predictions, digits=4)
        for line in report.split('\n'):
            if line.strip():
                logger.info(f"  {line}")

        errors = []
        for true, pred, text, conf in zip(test_labels, predictions, test_texts, confidences):
            if true != pred:
                errors.append({
                    'text': text,
                    'true': true,
                    'pred': pred,
                    'confidence': conf,
                })

        error_rate = len(errors) / len(test_labels) * 100 if test_labels else 0.0
        logger.info(f"错误样本数: {len(errors)}/{len(test_labels)} ({error_rate:.2f}%)")

        if errors:
            logger.info("前10个错误样本:")
            for i, err in enumerate(errors[:10], 1):
                logger.info(f"  [{i}] 文本: {err['text'][:60]}...")
                logger.info(
                    f"      真实: {err['true']} | 预测: {err['pred']} | 置信度: {err['confidence']:.4f}"
                )

        return {
            'accuracy': test_acc,
            'errors': errors,
            'report': report,
        }

    except Exception:
        logger.exception("评估测试集时发生错误")
        sys.exit(1)


def main():
    try:
        if len(sys.argv) < 2:
            logger.error("参数不足")
            logger.info("用法: python classifier_train.py <数据文件路径>")
            sys.exit(1)

        data_path = sys.argv[1]

        if not Path(data_path).exists():
            logger.error(f"数据文件不存在: {data_path}")
            sys.exit(1)

        config = TransformerTrainingConfig()
        set_seed(config.random_seed)

        logger.info("=" * 60)
        logger.info("Transformer 文本分类训练 - 基于 xlm-roberta-large")
        logger.info("=" * 60)
        logger.info(f"训练配置: {asdict(config)}")

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
        logger.info(f"训练损失: {train_metrics.get('train_loss', 0.0):.4f}")
        logger.info(f"验证损失: {train_metrics.get('val_loss', 0.0):.4f}")
        logger.info(f"验证准确率: {train_metrics.get('val_accuracy', 0.0):.4f}")
        logger.info(f"测试集准确率: {test_results['accuracy']:.4f}")
        logger.info(f"模型目录: {config.output_dir}")
        logger.info("=" * 60)

        if test_results['accuracy'] >= 0.98:
            logger.info("恭喜！模型准确率达到98%以上！")
        elif test_results['accuracy'] >= 0.95:
            logger.info("模型表现良好，准确率超过95%")
            logger.info("改进建议:")
            logger.info("  1. 增加更多训练数据")
            logger.info("  2. 分析错误样本，优化标注质量")
            logger.info("  3. 适当增大 max_length 或训练轮次")
        else:
            logger.info(f"当前准确率: {test_results['accuracy']:.2%}")
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