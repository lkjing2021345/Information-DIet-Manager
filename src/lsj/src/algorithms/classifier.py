"""
文本分类模块。

职责：
    基于浏览记录的标题与 URL 输出业务类别标签。

分类策略：
    1. 先执行关键词规则，优先处理高确定性样本；
    2. 规则未命中时，再调用 Transformer 模型兜底；
    3. 输入为空、模型未加载或推理失败时，统一回退为 Other。
"""
import json
from typing import Dict, List, Optional, Tuple, Union

from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.logger import setup_logger

logger = setup_logger(__name__, '../../logs/classifier.log')

class ContentClassifier:
    """
    浏览记录内容分类器。

    设计上将“可解释的规则匹配”与“泛化能力更强的模型预测”结合，
    使高频常见样本走低成本路径，长尾样本交由模型处理。
    """

    CATEGORY_NEWS = "News"
    CATEGORY_ENTERTAINMENT = "Entertainment"
    CATEGORY_LEARNING = "Learning"
    CATEGORY_SOCIAL = "Social"
    CATEGORY_SHOPPING = "Shopping"
    CATEGORY_TOOLS = "Tools"
    CATEGORY_OTHER = "Other"

    def __init__(
        self,
        keyword_dict: Optional[Dict[str, List[str]]] = None,
        model_path: Optional[str] = None,
        pretrained_model_name: str = "xlm-roberta-large",
        max_length: int = 128,
        inference_batch_size: int = 8,
        device: Optional[str] = None,
    ):
        self.rules = keyword_dict if keyword_dict is not None else self._load_default_rules()
        self.categories = list(self.rules.keys())

        self.pretrained_model_name = pretrained_model_name
        self.max_length = max_length if max_length > 0 else 128
        self.inference_batch_size = inference_batch_size if inference_batch_size > 0 else 8

        try:
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        except Exception:
            logger.warning("传入的 device 非法，自动回退到可用设备")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.tokenizer = None
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}

        if model_path:
            self.load_model(model_path)

        logger.info(f"ContentClassifier 初始化完成，设备: {self.device}")

    def _load_default_rules(self) -> Dict[str, List[str]]:
        """加载默认分类规则，并将配置文件中的类别键映射为类内统一常量。"""
        current_dir = Path(__file__).parent
        config_path = current_dir.joinpath("rules", "default_classify_rules.json")

        try:
            with open(config_path, 'r', encoding="utf-8") as f:
                raw_data = json.load(f)

            category_mapping = {
                "Social": self.CATEGORY_SOCIAL,
                "Learning": self.CATEGORY_LEARNING,
                "Shopping": self.CATEGORY_SHOPPING,
                "Entertainment": self.CATEGORY_ENTERTAINMENT,
                "News": self.CATEGORY_NEWS,
                "Tools": self.CATEGORY_TOOLS,
                "Other": self.CATEGORY_OTHER,
            }

            result = {}
            for key, category_const in category_mapping.items():
                result[category_const] = raw_data.get(key, [])

            return result

        except FileNotFoundError:
            logger.error(f"配置文件 {config_path} 未找到，使用空规则")
            return {}
        except json.JSONDecodeError:
            logger.error(f"配置文件 {config_path} 格式错误，请检查 JSON 语法")
            return {}
        except Exception as e:
            logger.error(f"出现异常错误: {e}")
            return {}

    def _normalize_text(self, text: Optional[str]) -> str:
        """将输入统一转为可安全处理的字符串，避免后续拼接与推理时出现空值问题。"""
        if text is None:
            return ""
        return str(text).strip()

    def _prepare_text_for_model(self, text: Optional[str], url: Optional[str] = None) -> str:
        """构造模型输入文本；当标题与 URL 同时存在时显式保留字段语义。"""
        normalized_text = self._normalize_text(text)
        normalized_url = self._normalize_text(url)

        if normalized_text and normalized_url:
            return f"标题: {normalized_text} [SEP] URL: {normalized_url}"
        if normalized_text:
            return normalized_text
        if normalized_url:
            return normalized_url
        return ""

    def _predict_by_model(self, text: str, url: Optional[str] = None) -> Tuple[str, float]:
        """使用 Transformer 模型预测单条样本；模型不可用或输入为空时返回兜底结果。"""
        if self.model is None or self.tokenizer is None:
            logger.warning("Transformer 模型未加载，无法预测")
            return self.CATEGORY_OTHER, 0.0

        prepared_text = self._prepare_text_for_model(text=text, url=url)
        if not prepared_text:
            return self.CATEGORY_OTHER, 0.0

        labels, confidences = self.predict_texts(
            [prepared_text],
            return_confidence=True,
            batch_size=1,
        )
        if not labels:
            return self.CATEGORY_OTHER, 0.0
        return labels[0], confidences[0]

    def _batch_predict_by_rules_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用 pandas 向量化字符串匹配批量执行规则分类。

        这样做的主要目的是减少逐行遍历带来的 Python 开销，提升大批量数据处理效率。
        """
        import re

        df = df.copy()
        df['category'] = None
        df['confidence'] = None

        df['title'] = df['title'].fillna('')
        df['url'] = df['url'].fillna('')
        # 提前生成小写副本，避免在每条规则上重复做字符串转换。
        df['title_lower'] = df['title'].str.lower()
        df['url_lower'] = df['url'].str.lower()

        for category, keywords in self.rules.items():
            if not keywords:
                continue

            escaped_keywords = [re.escape(kw.lower()) for kw in keywords if kw]
            if not escaped_keywords:
                continue
            pattern = '|'.join(escaped_keywords)

            try:
                title_match = df['title_lower'].str.contains(pattern, case=False, na=False, regex=True)
                url_match = df['url_lower'].str.contains(pattern, case=False, na=False, regex=True)
                matched = title_match | url_match
                # 仅覆盖此前尚未命中的记录，保证规则顺序就是优先级顺序。
                need_update = matched & df['category'].isna()
                df.loc[need_update, 'category'] = category
                df.loc[need_update, 'confidence'] = 1.0
            except Exception as e:
                logger.warning(f"类别 {category} 的规则匹配失败: {e}")
                continue

        return df.drop(columns=['title_lower', 'url_lower'])

    def predict_by_rules(self, text: str, url: Optional[str] = None) -> Optional[str]:
        """按“先 URL、后标题”的顺序执行规则匹配；URL 通常更稳定，因此优先级更高。"""
        text_lower = str(text).lower() if text else ""
        url_lower = str(url).lower() if url else ""

        if url_lower:
            for category, keywords in self.rules.items():
                for keyword in keywords:
                    if keyword and keyword.lower() in url_lower:
                        logger.info(f"URL匹配成功: '{keyword}' -> {category}")
                        return category

        if text_lower:
            for category, keywords in self.rules.items():
                for keyword in keywords:
                    if keyword and keyword.lower() in text_lower:
                        logger.info(f"标题匹配成功: '{keyword}' -> {category}")
                        return category

        logger.debug("规则匹配失败，返回 None")
        return None

    @torch.no_grad()
    def predict_texts(
        self,
        texts: List[str],
        return_confidence: bool = False,
        batch_size: Optional[int] = None,
    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        """
        批量执行模型预测。

        返回值会根据 return_confidence 变化：
        - False：仅返回标签列表；
        - True：返回 (标签列表, 置信度列表)。
        """
        if not isinstance(texts, list):
            raise TypeError("texts 必须是字符串列表")

        if not texts:
            return ([], []) if return_confidence else []

        if self.model is None or self.tokenizer is None:
            logger.warning("Transformer 模型未加载，返回 Other")
            fallback = [self.CATEGORY_OTHER] * len(texts)
            fallback_conf = [0.0] * len(texts)
            return (fallback, fallback_conf) if return_confidence else fallback

        effective_batch_size = batch_size or self.inference_batch_size
        if effective_batch_size <= 0:
            effective_batch_size = self.inference_batch_size if self.inference_batch_size > 0 else 8

        normalized_texts = [self._normalize_text(text) for text in texts]

        predictions: List[str] = []
        confidences: List[float] = []

        self.model.eval()

        for start in range(0, len(normalized_texts), effective_batch_size):
            batch_texts = normalized_texts[start:start + effective_batch_size]
            if not batch_texts:
                continue

            encoded = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            outputs = self.model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1)
            batch_confidences, batch_pred_ids = torch.max(probs, dim=-1)

            predictions.extend(
                self.id2label.get(pred_id, self.CATEGORY_OTHER)
                for pred_id in batch_pred_ids.detach().cpu().tolist()
            )
            confidences.extend(batch_confidences.detach().cpu().tolist())

        return (predictions, confidences) if return_confidence else predictions

    def predict(self, text: str, url: Optional[str] = None) -> str:
        """单条预测主入口：规则优先，模型兜底。"""
        result = self.predict_by_rules(text=text, url=url)
        if result:
            logger.info(f"规则匹配成功: {result}")
            return result

        model_result, confidence = self._predict_by_model(text=text, url=url)
        logger.info(f"模型预测成功: {model_result}, confidence={confidence:.4f}")
        return model_result or self.CATEGORY_OTHER

    def predict_with_confidence(self, text: str, url: Optional[str] = None) -> Tuple[str, float]:
        """返回标签与置信度；规则命中时置信度固定为 1.0。"""
        result = self.predict_by_rules(text=text, url=url)
        if result:
            return result, 1.0
        return self._predict_by_model(text=text, url=url)

    def batch_predict(
        self,
        df: pd.DataFrame,
        use_parallel: bool = True,
        n_workers: int = None,
        batch_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        对整批浏览记录执行分类。

        当前实现保留了并行相关参数，但核心流程仍是：
        先批量规则匹配，再仅对未命中的样本做模型推理。
        """
        import time

        if df.empty:
            logger.warning("输入数据为空")
            return df

        if 'title' not in df.columns:
            raise ValueError("输入 DataFrame 必须包含 title 列")
        if 'url' not in df.columns:
            df = df.copy()
            df['url'] = ''
        if batch_size is not None and batch_size <= 0:
            batch_size = self.inference_batch_size

        start_time = time.time()
        logger.info(f"开始处理 {len(df)} 条数据...")

        df = self._batch_predict_by_rules_vectorized(df)

        rule_matched = df['category'].notna().sum()
        rule_ratio = rule_matched / len(df) * 100
        logger.info(f"规则匹配: {rule_matched}/{len(df)} ({rule_ratio:.1f}%)")

        need_model = df['category'].isna()
        need_model_count = int(need_model.sum())

        if need_model_count > 0:
            logger.info(f"进入 Transformer 批量推理: {need_model_count} 条")
            # 仅对规则未命中的记录做模型推理，避免对全量数据进行高成本推理。
            need_model_df = df.loc[need_model, ['title', 'url']].copy()
            prepared_texts = [
                self._prepare_text_for_model(title, url)
                for title, url in zip(need_model_df['title'].tolist(), need_model_df['url'].tolist())
            ]

            predictions, confidences = self.predict_texts(
                prepared_texts,
                return_confidence=True,
                batch_size=batch_size or self.inference_batch_size,
            )

            df.loc[need_model, 'category'] = predictions
            df.loc[need_model, 'confidence'] = confidences

        df['category'] = df['category'].fillna(self.CATEGORY_OTHER)
        df['confidence'] = df['confidence'].fillna(0.0)

        elapsed = time.time() - start_time
        logger.info(f"处理完成，耗时: {elapsed:.2f}秒")
        logger.info(f"平均速度: {len(df) / elapsed:.0f} 条/秒" if elapsed > 0 else "处理完成")

        return df

    def save_model(self, path: str) -> None:
        """保存模型、分词器及标签映射，保证训练产物可独立用于推理。"""
        if self.model is None or self.tokenizer is None:
            logger.error("模型未加载，无法保存")
            return

        if not path:
            logger.error("模型保存路径为空")
            return

        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            self.model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)

            with open(output_path / "label2id.json", "w", encoding="utf-8") as f:
                json.dump(self.label2id, f, ensure_ascii=False, indent=2)

            with open(output_path / "id2label.json", "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in self.id2label.items()}, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ Transformer 模型已保存到: {output_path}")
        except Exception as e:
            logger.error(f"模型保存失败: {e}")

    def load_model(self, path: str) -> bool:
        """
        从目录加载训练好的分类模型。

        会优先读取显式保存的 label2id / id2label；若缺失，则退回模型配置中的映射。
        """
        if not path:
            logger.error("模型目录路径为空")
            return False

        model_dir = Path(path)
        if not model_dir.exists() or not model_dir.is_dir():
            logger.error(f"模型目录不存在: {path}")
            return False

        try:
            label2id_path = model_dir / "label2id.json"
            id2label_path = model_dir / "id2label.json"

            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.model.to(self.device)
            self.model.eval()

            if label2id_path.exists():
                with open(label2id_path, 'r', encoding='utf-8') as f:
                    loaded_label2id = json.load(f)
                self.label2id = {str(k): int(v) for k, v in loaded_label2id.items()}
            else:
                self.label2id = {str(k): int(v) for k, v in dict(self.model.config.label2id).items()}

            if id2label_path.exists():
                with open(id2label_path, 'r', encoding='utf-8') as f:
                    raw_id2label = json.load(f)
                self.id2label = {int(k): str(v) for k, v in raw_id2label.items()}
            else:
                self.id2label = {int(k): str(v) for k, v in dict(self.model.config.id2label).items()}

            if not self.id2label and self.label2id:
                self.id2label = {idx: label for label, idx in self.label2id.items()}
            if not self.label2id and self.id2label:
                self.label2id = {label: idx for idx, label in self.id2label.items()}

            if self.id2label:
                ordered_categories = [self.id2label[idx] for idx in sorted(self.id2label.keys())]
                self.categories = ordered_categories

            logger.info(f"✅ Transformer 模型已加载: {path}")
            logger.info(f"当前设备: {self.device}")
            return True

        except Exception as e:
            self.model = None
            self.tokenizer = None
            self.label2id = {}
            self.id2label = {}
            logger.error(f"模型加载失败: {e}")
            return False

    def get_category_distribution(self, df: pd.DataFrame) -> pd.Series:
        """统计分类结果分布；输入不合法时返回空 Series，便于调用方安全处理。"""
        if df.empty:
            logger.error("数据为空")
            return pd.Series(dtype='int64')

        if 'category' not in df.columns:
            logger.error("数据缺少 'category' 列")
            return pd.Series(dtype='int64')

        return df['category'].value_counts()


if __name__ == "__main__":
    classifier = ContentClassifier(
        model_path="./models/classifier_xlm_roberta"
    )

    history_df = pd.read_json('utils/output/history_data.jsonl', lines=True)
    result_df = classifier.batch_predict(history_df)
    result_df.to_csv('utils/output/result.csv', index=False)