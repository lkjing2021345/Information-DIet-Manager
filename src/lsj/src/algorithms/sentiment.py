# -*- coding: utf-8 -*-  # 声明文件编码，避免中文乱码
from __future__ import annotations  # 让 Python 3.8/3.9 支持 | 类型注解
"""
情感分析模块。

职责：
    基于 cntext 词典能力与可选的机器学习模型，对文本执行情感、情绪、可读性与趋势分析。

说明：
    - 词典分析适合快速、可解释的基础判断；
    - 自定义模型适合在特定业务标注数据上提升一致性；
    - 两者可以组合使用，以在可解释性与泛化能力之间取得平衡。
"""
# ======== 环境变量设置 ========
import os  # 系统环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 为 HuggingFace 下载配置国内镜像
# ======== 标准库导入 ========
import pickle  # 模型持久化
from pathlib import Path  # 路径处理
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import csv  # 用于识别和读取 CSV 词典
import importlib.util  # 动态检查依赖是否安装
# ======== 第三方库导入 ========
import jieba  # 中文分词
import pandas as pd  # 数据处理
import yaml  # YAML 读取

from utils.logger import setup_logger

# 初始化日志器
logger = setup_logger(__name__, "../../logs/sentiment.log")
MODEL_API_VERSION = "1.0"
# ==================== 判断依赖是否存在 ====================
def _pkg_exists(name: str) -> bool:
    """判断某个包是否可被导入"""
    return importlib.util.find_spec(name) is not None  # 返回包是否可用
# ==================== cntext 导入 ====================
ct = None  # 默认占位，避免后续直接引用时报 NameError
CNTEXT_AVAILABLE = False  # 标记 cntext 是否可用
if not _pkg_exists("cntext"):  # 如果未安装 cntext
    logger.warning("cntext 未安装（find_spec 找不到），请运行: python -m pip install cntext")
else:
    try:
        import cntext as ct  # 延迟导入，避免依赖缺失时模块直接崩溃
        CNTEXT_AVAILABLE = True  # 标记可用
        logger.info("cntext 加载成功: %s, version=%s", ct.__file__, getattr(ct, "__version__", "未知"))
    except Exception as e:
        CNTEXT_AVAILABLE = False  # 导入失败，后续功能需走降级或抛错
        ct = None  # 清空引用，避免误用半初始化对象
        logger.exception("cntext 已安装但导入失败。异常: %r", e)
# ==================== BERT 相关导入 ====================
torch = None  # 先给出默认占位，避免依赖缺失时报 NameError
BertTokenizer = None
BertForSequenceClassification = None
BERT_AVAILABLE = False  # 标记 BERT 相关依赖是否可用
try:
    import torch  # PyTorch
    from transformers import BertTokenizer, BertForSequenceClassification  # BERT 推理/训练组件
    BERT_AVAILABLE = True  # 标记可用
    logger.info("BERT 相关依赖加载成功：torch=%s, transformers 已可用", torch.__version__)
except Exception as e:
    BERT_AVAILABLE = False  # 标记不可用
    logger.exception("BERT 相关依赖导入失败。异常: %r", e)


@dataclass
class SentimentScore:
    """词典情感打分结构。"""
    pos: int = 0
    neg: int = 0
    pos_word: List[str] = field(default_factory=list)
    neg_word: List[str] = field(default_factory=list)
    categories: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pos': self.pos,
            'neg': self.neg,
            'pos_word': self.pos_word,
            'neg_word': self.neg_word,
            'categories': self.categories,
            'raw': self.raw,
        }


@dataclass
class SentimentPrediction:
    """统一预测输出结构（对外仍可转 dict 兼容）。"""
    sentiment: str
    polarity: float
    pos_count: int
    neg_count: int
    confidence: float
    pos_words: List[str] = field(default_factory=list)
    neg_words: List[str] = field(default_factory=list)
    emotions: Optional[Dict[str, int]] = None
    model_sentiment: Optional[str] = None

    def to_dict(self, include_words: bool = True, include_emotions: bool = True) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            'sentiment': self.sentiment,
            'polarity': self.polarity,
            'pos_count': self.pos_count,
            'neg_count': self.neg_count,
            'confidence': self.confidence,
        }
        if self.model_sentiment:
            result['model_sentiment'] = self.model_sentiment
        if include_words:
            result['pos_words'] = self.pos_words
            result['neg_words'] = self.neg_words
        if include_emotions and self.emotions is not None:
            result['emotions'] = self.emotions
        return result


class CntextSentimentBackend:
    """封装基于 cntext 词典的情感与情绪分析逻辑。"""

    def __init__(self, analyzer: "SentimentAnalyzer"):
        self.analyzer = analyzer

    def analyze_score(self, text: str) -> SentimentScore:
        if self.analyzer._is_empty_text(text):
            return SentimentScore()

        try:
            if ct is None:
                logger.error("cntext 未安装，无法进行情感分析")
                return SentimentScore()

            if self.analyzer.custom_dict:
                raw = ct.sentiment(str(text), diction=self.analyzer.custom_dict)
            elif self.analyzer._cntext_dict_cache is not None:
                raw = ct.sentiment(str(text), diction=self.analyzer._cntext_dict_cache)
            else:
                default_yaml = ct.read_yaml_dict('zh_common_DUTIR.yaml')
                default_dict = default_yaml.get('Dictionary', default_yaml)
                raw = ct.sentiment(str(text), diction=default_dict)

            pos = raw.get('pos_num', raw.get('pos', 0))
            neg = raw.get('neg_num', raw.get('neg', 0))
            pos_words = raw.get('pos_word', raw.get('pos_words', []))
            neg_words = raw.get('neg_word', raw.get('neg_words', []))

            if pos == 0 and neg == 0:
                # 某些词典不会直接返回 pos/neg，需要从细粒度情绪字段回推正负倾向。
                for key in ['乐_num', '喜_num', '好_num']:
                    if key in raw:
                        pos += raw[key]
                for key in ['怒_num', '愤_num', '哀_num', '悲_num', '惧_num', '恐_num', '恶_num', '厌_num', '惊_num', '惊讶_num']:
                    if key in raw:
                        neg += raw[key]

            if not isinstance(pos_words, list):
                pos_words = []
            if not isinstance(neg_words, list):
                neg_words = []

            exclude = {'stopword_num', 'word_num', 'sentence_num'}
            # 仅保留 *_num 形式的情绪统计字段，供上层做更细粒度分析。
            categories = {
                k: v for k, v in raw.items()
                if isinstance(k, str) and k.endswith('_num') and k not in exclude
            }

            return SentimentScore(
                pos=int(pos) if pd.notna(pos) else 0,
                neg=int(neg) if pd.notna(neg) else 0,
                pos_word=list(pos_words),
                neg_word=list(neg_words),
                categories=categories,
                raw=raw,
            )

        except Exception as e:
            logger.exception(f"情感分析失败: {e}")
            return SentimentScore()

    def analyze_emotions(self, text: str) -> Dict[str, int]:
        if self.analyzer._is_empty_text(text):
            logger.error("传入文本为空")
            return {}

        expected_emotion_keys = {
            '乐', '怒', '哀', '惧', '恶', '惊', '好',
            '喜', '愤', '悲', '恐', '厌', '惊讶',
        }

        def _looks_like_emotion_dict(d: Any) -> bool:
            if not isinstance(d, dict) or not d:
                return False
            return any((k in expected_emotion_keys) for k in d.keys())

        diction_for_emotion: Optional[Dict[str, Any]] = None
        if _looks_like_emotion_dict(self.analyzer.custom_dict):
            diction_for_emotion = self.analyzer.custom_dict
        elif _looks_like_emotion_dict(self.analyzer._cntext_dict_cache):
            diction_for_emotion = self.analyzer._cntext_dict_cache
        else:
            return {}

        emotion_key_mapping = {
            '乐': self.analyzer.EMOTION_JOY,
            '喜': self.analyzer.EMOTION_JOY,
            '怒': self.analyzer.EMOTION_ANGER,
            '愤': self.analyzer.EMOTION_ANGER,
            '哀': self.analyzer.EMOTION_SADNESS,
            '悲': self.analyzer.EMOTION_SADNESS,
            '惧': self.analyzer.EMOTION_FEAR,
            '恐': self.analyzer.EMOTION_FEAR,
            '恶': self.analyzer.EMOTION_DISGUST,
            '厌': self.analyzer.EMOTION_DISGUST,
            '惊': self.analyzer.EMOTION_SURPRISE,
            '惊讶': self.analyzer.EMOTION_SURPRISE,
            '好': self.analyzer.EMOTION_GOOD,
        }

        try:
            if ct is None:
                logger.error("cntext 未安装，无法进行情绪分析")
                return {}

            raw = ct.sentiment(str(text), diction=diction_for_emotion)
            exclude = {'stopword_num', 'word_num', 'sentence_num'}
            emotions: Dict[str, int] = {}

            for key, value in raw.items():
                if not (isinstance(key, str) and key.endswith('_num')) or key in exclude:
                    continue
                category = key[:-4]
                if category not in expected_emotion_keys:
                    continue
                mapped = emotion_key_mapping.get(category)
                if mapped is None:
                    continue
                try:
                    emotions[mapped] = emotions.get(mapped, 0) + int(value)
                except Exception:
                    continue
            return emotions
        except Exception as e:
            logger.exception(f"情绪分析失败: {e}")
            return {}


class ModelBackend:
    """封装 BERT 与朴素贝叶斯模型的加载、预测与兼容入口。"""

    def __init__(self, analyzer: "SentimentAnalyzer"):
        self.analyzer = analyzer

    def predict(self, text: str) -> str | None:
        if self.analyzer._is_empty_text(text):
            logger.warning("文本为空")
            return None

        try:
            if self.analyzer.use_bert and self.analyzer.bert_model is not None:
                logger.info("predict_by_model: 当前使用 BERT 进行预测")
                return self.predict_by_bert(text)
            if self.analyzer.model is not None:
                logger.info("predict_by_model: 当前使用 Naive Bayes 进行预测")
                return self.predict_by_naive_bayes(text)
            logger.error("predict_by_model: 没有可用模型（BERT/NB 都未加载）")
            return None
        except Exception as e:
            logger.exception(f"模型预测失败: {e}")
            return None

    def predict_by_bert(self, text: str) -> str:
        self.analyzer.bert_model.eval()
        encoding = self.analyzer.bert_tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.analyzer.device)
        attention_mask = encoding["attention_mask"].to(self.analyzer.device)

        with torch.no_grad():
            outputs = self.analyzer.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]

        if self.analyzer.label_encoder is not None:
            prediction_label = self.analyzer.label_encoder.inverse_transform([prediction])[0]
        else:
            prediction_label = str(prediction)

        logger.debug(f"BERT 预测结果: {prediction_label}")
        return prediction_label

    def predict_by_naive_bayes(self, text: str) -> str | None:
        if self.analyzer.model is None:
            logger.error("模型未加载")
            return None
        if self.analyzer.vectorizer is None:
            logger.error("向量化器未加载")
            return None

        words = self.analyzer._segment_text(text)
        text_processed = ' '.join(words)
        X = self.analyzer.vectorizer.transform([text_processed])
        prediction = self.analyzer.model.predict(X)[0]
        logger.debug(f"朴素贝叶斯预测结果: {prediction}")
        return str(prediction)

    def train_bert_model(self,
                         train_df: pd.DataFrame,
                         text_column: str,
                         label_column: str,
                         test_size: float,
                         epochs: int,
                         batch_size: int,
                         learning_rate: float,
                         max_length: int) -> Dict[str, Any]:
        """训练逻辑已迁移到 sentiment_train.py。"""
        raise NotImplementedError(
            "Training has been moved to sentiment_train.py. "
            "Please use sentiment_train.train(...) or run_training_pipeline(...)."
        )

    def train_naive_bayes_model(self,
                                train_df: pd.DataFrame,
                                text_column: str,
                                label_column: str,
                                test_size: float) -> Dict[str, Any]:
        """训练逻辑已迁移到 sentiment_train.py。"""
        raise NotImplementedError(
            "Training has been moved to sentiment_train.py. "
            "Please use sentiment_train.train(...) or run_training_pipeline(...)."
        )

    def load_bert_model(self, model_dir: str) -> bool:
        if not BERT_AVAILABLE:
            logger.error("BERT 不可用")
            return False

        try:
            model_path = Path(model_dir)
            self.analyzer.bert_model = BertForSequenceClassification.from_pretrained(model_path)
            self.analyzer.bert_tokenizer = BertTokenizer.from_pretrained(model_path)
            self.analyzer.bert_model.to(self.analyzer.device)

            metadata_json_path = model_path / 'metadata.json'
            if metadata_json_path.exists():
                # 通过 api_version 校验训练产物格式，避免旧模型与新代码不兼容。
                import json
                with open(metadata_json_path, 'r', encoding='utf-8') as f:
                    metadata_json = json.load(f)
                api_version = str(metadata_json.get('api_version', ''))
                if api_version and api_version != MODEL_API_VERSION:
                    raise ValueError(
                        f"模型 api_version 不匹配: model={api_version}, expected={MODEL_API_VERSION}"
                    )

            metadata_path = model_path / 'metadata.pkl'
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                self.analyzer.label_encoder = metadata.get('label_encoder')
                self.analyzer.bert_model_name = metadata.get('bert_model_name', 'bert-base-chinese')

            self.analyzer.use_bert = True
            return True
        except Exception as e:
            logger.exception(f"加载 BERT 模型失败: {e}")
            return False

    def load_naive_bayes_model(self, path: str) -> bool:
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)

            if isinstance(model_data, dict):
                self.analyzer.model = model_data.get('model')
                self.analyzer.vectorizer = model_data.get('vectorizer')
            else:
                self.analyzer.model = model_data
                self.analyzer.vectorizer = None

            self.analyzer.use_bert = False
            return True
        except Exception as e:
            logger.exception(f"加载朴素贝叶斯模型失败: {e}")
            return False
# ==================== SentimentAnalyzer 主类 ====================
class SentimentAnalyzer:
    """
    情感分析主入口。

    统一对外暴露词典分析、模型分析、批量预测、趋势统计和报告生成等能力。
    """
    # ==== 类常量 ====
    SENTIMENT_POSITIVE = "Positive"
    SENTIMENT_NEGATIVE = "Negative"
    SENTIMENT_NEUTRAL = "Neutral"
    EMOTION_JOY = "Joy"
    EMOTION_ANGER = "Anger"
    EMOTION_SADNESS = "Sadness"
    EMOTION_FEAR = "Fear"
    EMOTION_DISGUST = "Disgust"
    EMOTION_SURPRISE = "Surprise"
    EMOTION_GOOD = "Good"
    PSYCH_ATTITUDE = "Attitude"
    PSYCH_COGNITION = "Cognition"
    PSYCH_EMOTION = "Emotion"
    def __init__(self,
                 diction: str = 'zh_common_DUTIR.yaml',
                 custom_dict_path: Optional[str] = None,
                 model_path: Optional[str] = None,
                 stopwords_path: Optional[str] = None,
                 bert_model_name: str = 'hfl/chinese-roberta-wwm-ext',
                 use_bert: bool = False):
        if not CNTEXT_AVAILABLE:
            logger.error("没有安装 cntext 库，建议 pip install cntext")
            raise ImportError("cntext is required but not installed")
        self.diction = diction  # 保存当前使用的词典配置名或词典对象
        self._cntext_dict_cache: Optional[Dict[str, Any]] = None  # 缓存解析后的词典，避免重复读取
        default_stopwords_path = Path(__file__).resolve().parent / "rules" / "hit_stopwords.txt"
        self.stopwords_path = str(default_stopwords_path if stopwords_path is None else stopwords_path)  # 停用词文件路径
        self._stopwords_cache: set[str] | None = None  # 停用词缓存，避免重复读取文件
        # ===== 加载词典 =====
        if isinstance(self.diction, dict):
            loaded = self.diction
        elif isinstance(self.diction, str):
            try:
                if self.diction.lower().endswith((".yaml", ".yml")):
                    loaded = ct.read_yaml_dict(self.diction)
                else:
                    logger.warning("diction 建议使用内置 yaml 名或 yaml 文件路径")
                    loaded = None
            except Exception as e:
                logger.exception(f"读取 YAML 词典失败: {e}")
                loaded = None
        else:
            loaded = None
        if isinstance(loaded, dict):
            self._cntext_dict_cache = loaded.get('Dictionary', loaded)
            if self._cntext_dict_cache:
                logger.info(f"词典加载成功，已缓存，包含键: {list(self._cntext_dict_cache.keys())[:10]}")
        else:
            logger.warning("词典加载失败")
            self._cntext_dict_cache = None
        # ===== 加载自定义词典 =====
        if custom_dict_path is not None:
            try:
                self.custom_dict = self._load_custom_dict(custom_dict_path)
                if isinstance(self.custom_dict, dict) and 'Dictionary' in self.custom_dict:
                    self.custom_dict = self.custom_dict.get('Dictionary')
            except Exception as e:
                logger.exception(f"加载自定义词典失败: {e}")
                self.custom_dict = None
        else:
            self.custom_dict = None
        # ===== 加载模型 =====
        if model_path is not None:
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    self.model = data.get('model')
                    self.vectorizer = data.get('vectorizer')
                else:
                    self.model = data
                    self.vectorizer = None
                    logger.warning("模型文件中未包含 vectorizer，传统模型能力可能受限")
            except Exception as e:
                logger.exception(f"加载模型失败: {e}")
                self.model = None
                self.vectorizer = None
        else:
            self.model = None
            self.vectorizer = None
        # ===== BERT 设置 =====
        self.use_bert = use_bert and BERT_AVAILABLE
        self.bert_model_name = bert_model_name
        self.bert_model = None
        self.bert_tokenizer = None
        self.label_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if BERT_AVAILABLE else None
        self.cntext_backend = CntextSentimentBackend(self)
        self.model_backend = ModelBackend(self)
        if self.use_bert:
            logger.info(f"BERT 模式已启用，设备: {self.device}")
        logger.info(f"初始化完成 - 词典: {diction}, BERT: {self.use_bert}")

    @classmethod
    def load(cls, model_path: Path) -> "SentimentAnalyzer":
        """稳定推理入口：加载模型并返回可用的分析器实例。"""
        analyzer = cls(use_bert=False)
        if not analyzer.load_model(str(model_path)):
            raise ValueError(f"Failed to load model from: {model_path}")
        return analyzer

    # ==================== 静态方法 ====================
    
    @staticmethod
    def get_available_dicts() -> List[str]:
        """
        获取 cntext 可用的内置词典列表
        
        返回:
            List[str]: 可用词典名称列表
        """
        if not CNTEXT_AVAILABLE or ct is None:
            logger.error("cntext 未安装")
            return []
        cntext_dict_lists = ct.get_dict_list()
        return cntext_dict_lists

    @staticmethod
    def identify_file_format(file_path: str) -> str:
        """
        识别文件格式

        返回:
            'yaml', 'csv', 或 'unknown'
        """
        try:
            ext = Path(file_path).suffix.lower()
            if ext in ['.yaml', '.yml']:
                return 'yaml'
            elif ext in ['.csv', '.tsv']:
                return 'csv'

            with open(file_path, 'r', encoding='utf-8') as f:
                sample = f.read(2048)

            try:
                data = yaml.safe_load(sample)
                if isinstance(data, (dict, list)):
                    return 'yaml'
            except Exception as e:
                logger.exception(f"读取 YAML 类型词典出现异常: {e}")

            try:
                dialect = csv.Sniffer().sniff(sample)
                if dialect.delimiter in [',', '\t', ';', '|']:
                    return 'csv'
            except Exception as e:
                logger.exception(f"读取 CSV 类型词典出现异常: {e}")

            return 'unknown'

        except Exception as e:
            logger.error(f"识别文件格式失败: {e}")
            return 'unknown'

    @staticmethod
    def _is_empty_text(text: Any) -> bool:
        """统一判断文本是否为空、缺失或仅包含空白字符。"""
        return text is None or pd.isna(text) or str(text).strip() == ''

    def _empty_cntext_score_result(self) -> Dict[str, Any]:
        """cntext 打分失败/空文本时的统一返回。"""
        return SentimentScore().to_dict()

    def _load_stopwords(self) -> set[str]:
        """加载停用词并缓存，避免每次分词都重复读取文件。"""
        if self._stopwords_cache is not None:
            return self._stopwords_cache

        if not self.stopwords_path:
            self._stopwords_cache = set()
            return self._stopwords_cache

        try:
            with open(self.stopwords_path, 'r', encoding='utf-8') as f:
                self._stopwords_cache = {line.strip() for line in f if line.strip()}
        except OSError:
            logger.warning(f"停用词词典不存在: {self.stopwords_path}")
            self._stopwords_cache = set()
        except Exception as e:
            logger.exception(f"加载停用词失败: {e}")
            self._stopwords_cache = set()

        return self._stopwords_cache

    # ==================== 私有方法（内部使用）====================

    def _load_custom_dict(self, path: str) -> Dict[str, Any]:
        """
        加载自定义词典（补充 cntext）

        参数:
            path: 词典文件路径

        返回:
            Dict[str, Any]: 自定义词典
        """

        FILE_TYPE = self.identify_file_format(file_path=path)
        # 先识别词典格式，再走对应解析逻辑。

        if FILE_TYPE == 'yaml':
            try:
                if ct is None:
                    raise ImportError("cntext is not available")
                yaml_dict = ct.read_yaml_dict(path)
                logger.info(f"成功加载 YAML 词典: {path}")
                return yaml_dict
            except OSError as e:
                logger.error(f"YAML 词典不存在或者损坏: {e}")
                return {}
            except Exception as e:
                logger.exception(f"出现未知异常，加载 YAML 词典失败: {e}")
                return {}

        elif FILE_TYPE == 'csv':
            try:
                df = pd.read_csv(path, encoding='utf-8', sep=None, engine='python')
                if 'word' not in df.columns or 'sentiment' not in df.columns:
                    logger.error(f"CSV 缺少必要的列 (word, sentiment)")
                    return {}

                df['sentiment'] = df['sentiment'].fillna('').astype(str).str.strip().str.lower()

                custom_dict = {
                    'pos': df[df['sentiment'] == 'positive']['word'].dropna().astype(str).tolist(),
                    'neg': df[df['sentiment'] == 'negative']['word'].dropna().astype(str).tolist(),
                }

                logger.info(f"成功加载 CSV 词典: {len(custom_dict['pos'])} 积极词, {len(custom_dict['neg'])} 消极词")

                return custom_dict
            except OSError as e:
                logger.error(f"CSV 词典不存在或者损坏: {e}")
                return {}
            except Exception as e:
                logger.exception(f"出现未知异常，加载 CSV 词典失败: {e}")
                return {}
        else:
            logger.error(f"不支持的文件格式: {path}")
            return {}
    
    def _segment_text(self, text: str) -> List[str]:
        """
        对文本进行分词

        参数:
            text: 待分词的文本

        返回:
            List[str]: 分词结果
        """
        if self._is_empty_text(text):
            logger.error("传入文本为空")
            return []

        try:
            words = jieba.lcut(str(text))
        except Exception as e:
            logger.exception(f"出现异常，分词失败: {e}")
            return []

        stopwords_set = self._load_stopwords()
        # 分词后去掉停用词与空白 token，减少后续噪声。

        filtered_words = [word for word in words if word not in stopwords_set and len(word.strip()) > 0]

        return filtered_words
    
    def _calculate_sentiment_score_cntext(self, text: str) -> Dict[str, Any]:
        """使用 cntext 计算情感相关统计（委托给 CntextSentimentBackend）。"""
        return self.cntext_backend.analyze_score(text).to_dict()


    def _analyze_emotions_cntext(self, text: str) -> Dict[str, Any]:
        """使用 cntext 分析具体情绪（委托给 CntextSentimentBackend）。"""
        return self.cntext_backend.analyze_emotions(text)

    def _score_to_sentiment(self, pos_count: int, neg_count: int,
                           threshold: float = 0.1) -> str:
        """将正负词数量映射为离散情感标签。"""
        total_count = pos_count + neg_count

        if total_count == 0:
            return self.SENTIMENT_NEUTRAL

        polarity_score = (pos_count - neg_count) / total_count

        if polarity_score > threshold:
            return self.SENTIMENT_POSITIVE
        elif polarity_score < -threshold:
            return self.SENTIMENT_NEGATIVE
        else:
            return self.SENTIMENT_NEUTRAL
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空文本或分析失败时使用的默认预测结果。"""
        return {
            'sentiment': self.SENTIMENT_NEUTRAL,
            'polarity': 0.0,
            'pos_count': 0,
            'neg_count': 0,
            'confidence': 0.0,
            'pos_words': [],
            'neg_words': []
        }
    
    # ==================== 核心公共方法 ====================
    
    def predict_by_cntext(self, text: str) -> Dict[str, Any]:
        """
        使用 cntext 进行多维度情感分析

        参数:
            text: 待分析的文本

        返回:
            Dict[str, Any]: {
                'sentiment': 情感类别,
                'polarity': 情感极性分数 (-1 到 1),
                'sentiment_scores': {pos: 积极词数, neg: 消极词数},
                'emotions': 情绪分析结果,
                'pos_words': 积极词列表,
                'neg_words': 消极词列表
            }
        """
        if self._is_empty_text(text):
            logger.error("传入文本为空")
            return {
                'sentiment': self.SENTIMENT_NEUTRAL,
                'polarity': 0,
                'sentiment_scores': {'pos': 0, 'neg': 0},
                'emotions': None,
                'pos_words': [],
                'neg_words': []
            }

        sentiment_score_obj = self.cntext_backend.analyze_score(text)
        sentiment_score = sentiment_score_obj.to_dict()

        pos_count = sentiment_score['pos']
        neg_count = sentiment_score['neg']
        pos_words = sentiment_score.get('pos_word', [])
        neg_words = sentiment_score.get('neg_word', [])

        polarity = self.calculate_polarity(pos_count=pos_count, neg_count=neg_count)

        sentiment = self._score_to_sentiment(pos_count=pos_count, neg_count=neg_count)

        result = {
            'sentiment': sentiment,
            'polarity': polarity,
            'sentiment_scores': {
                'pos': pos_count,
                'neg': neg_count
            },
            'pos_words': pos_words,
            'neg_words': neg_words
        }

        emotions = self._analyze_emotions_cntext(text)
        if emotions:
            result['emotions'] = emotions
            logger.debug(f"情绪分析结果: {emotions}")

        logger.debug(f"情感分析完成: {sentiment}, 极性: {polarity:.3f}")
        return result

    def predict_by_model(self, text: str) -> str | None:
        """基于自定义机器学习模型进行情感分析（委托给 ModelBackend）。"""
        return self.model_backend.predict(text)

    def _predict_by_bert(self, text: str) -> str:
        """使用 BERT 模型预测（委托给 ModelBackend）。"""
        return self.model_backend.predict_by_bert(text)

    def _predict_by_naive_bayes(self, text: str) -> str | None:
        """使用朴素贝叶斯模型预测（委托给 ModelBackend）。"""
        return self.model_backend.predict_by_naive_bayes(text)

    def predict(self, text: str,
               include_emotions: bool = True,
               include_words: bool = True,
               use_custom_model: bool = False) -> Dict[str, Any] | None:
        """
        综合预测情感（主入口方法）

        参数:
            text: 待分析的文本
            include_emotions: 是否包含具体情绪分析
            include_words: 是否包含匹配的词语列表
            use_custom_model: 是否使用自定义模型

        返回:
            Dict: {
                'sentiment': 情感类别,
                'polarity': 情感极性 (-1 到 1),
                'pos_count': 积极词数,
                'neg_count': 消极词数,
                'confidence': 置信度,
                'pos_words': 积极词列表 (可选),
                'neg_words': 消极词列表 (可选),
                'emotions': 情绪分析结果 (可选)
            }
        """
        if self._is_empty_text(text):
            logger.error("传入文本为空，无法进行预测")
            return None

        result = self.predict_by_cntext(text)

        sentiment = result['sentiment']
        polarity = result['polarity']
        pos_count = result['sentiment_scores']['pos']
        neg_count = result['sentiment_scores']['neg']

        words = self._segment_text(text)
        total_words = len(words)
        sentiment_words = pos_count + neg_count

        if total_words > 0:
            confidence = min(sentiment_words / total_words, 1.0)
        else:
            confidence = 0.0

        prediction = SentimentPrediction(
            sentiment=sentiment,
            polarity=polarity,
            pos_count=pos_count,
            neg_count=neg_count,
            confidence=confidence,
            pos_words=result.get('pos_words', []),
            neg_words=result.get('neg_words', []),
            emotions=result.get('emotions') if 'emotions' in result else None,
        )

        has_custom_model = (self.model is not None) or (self.use_bert and self.bert_model is not None)
        if use_custom_model and has_custom_model:
            # 词典结果与模型结果同时可用时，用模型结果辅助修正并调整置信度。
            model_sentiment = self.predict_by_model(text)
            if model_sentiment:
                if model_sentiment != sentiment:
                    logger.info(f"模型预测 ({model_sentiment}) 与词典预测 ({sentiment}) 不一致")
                    prediction.confidence *= 0.7
                    prediction.model_sentiment = model_sentiment
                    prediction.sentiment = model_sentiment
                else:
                    prediction.confidence = min(confidence * 1.2, 1.0)

        final_result = prediction.to_dict(
            include_words=include_words,
            include_emotions=include_emotions
        )

        logger.info(
            f"预测完成: {final_result['sentiment']} (置信度: {final_result['confidence']:.3f})"
        )
        return final_result
    
    def batch_predict(self, df: pd.DataFrame,
                     text_column: str = 'title',
                     include_emotions: bool = False,
                     batch_size: int = 1000) -> pd.DataFrame:
        """
        批量预测 DataFrame 中的情感

        参数:
            df: 输入数据
            text_column: 文本列名
            include_emotions: 是否包含情绪分析
            batch_size: 批处理大小

        返回:
            pd.DataFrame: 添加了情感分析列的 DataFrame
        """
        if text_column not in df.columns:
            logger.error(f"列 '{text_column}' 不存在于 DataFrame 中")
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        if df.empty:
            logger.warning("输入 DataFrame 为空")
            return df

        logger.info(f"开始批量预测，总数据量: {len(df)}, 批次大小: {batch_size}")

        sentiments = []
        polarities = []
        pos_counts = []
        neg_counts = []
        confidences = []
        emotions_list = [] if include_emotions else None

        total_batches = (len(df) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]

            logger.info(f"处理批次 {batch_idx + 1}/{total_batches} (行 {start_idx}-{end_idx})")

            for idx, row in batch_df.iterrows():
                text = row[text_column]

                if text is None or pd.isna(text) or str(text).strip() == '':
                    sentiments.append(self.SENTIMENT_NEUTRAL)
                    polarities.append(0.0)
                    pos_counts.append(0)
                    neg_counts.append(0)
                    confidences.append(0.0)
                    if include_emotions:
                        emotions_list.append({})
                    continue

                try:
                    result = self.predict(
                        text=str(text),
                        include_emotions=include_emotions,
                        include_words=False,
                        use_custom_model=True
                    )

                    if result is None:
                        # 预测失败时退回默认中性结果，避免中断整批任务。
                        sentiments.append(self.SENTIMENT_NEUTRAL)
                        polarities.append(0.0)
                        pos_counts.append(0)
                        neg_counts.append(0)
                        confidences.append(0.0)
                        if include_emotions:
                            emotions_list.append({})
                    else:
                        # 提取结果
                        sentiments.append(result.get('sentiment', self.SENTIMENT_NEUTRAL))
                        polarities.append(result.get('polarity', 0.0))
                        pos_counts.append(result.get('pos_count', 0))
                        neg_counts.append(result.get('neg_count', 0))
                        confidences.append(result.get('confidence', 0.0))

                        if include_emotions:
                            emotions_list.append(result.get('emotions', {}))

                except Exception as e:
                    logger.error(f"处理索引 {idx} 时出错: {e}")
                    sentiments.append(self.SENTIMENT_NEUTRAL)
                    polarities.append(0.0)
                    pos_counts.append(0)
                    neg_counts.append(0)
                    confidences.append(0.0)
                    if include_emotions:
                        emotions_list.append({})

            processed = end_idx
            progress = (processed / len(df)) * 100
            logger.info(f"已处理: {processed}/{len(df)} ({progress:.1f}%)")

        # 将结果写回副本，避免直接污染调用方传入的原始数据。
        # 创建副本避免修改原数据
        result_df = df.copy()
        result_df['sentiment'] = sentiments
        result_df['polarity'] = polarities
        result_df['pos_count'] = pos_counts
        result_df['neg_count'] = neg_counts
        result_df['confidence'] = confidences

        if include_emotions and emotions_list is not None:
            result_df['emotions'] = emotions_list

        logger.info(f"批量预测完成，共处理 {len(result_df)} 条数据")

        return result_df

    # ==================== 模型训练方法（已抽离到 sentiment_train.py） ====================

    def train_model(self,
                    train_df: pd.DataFrame,
                    text_column: str = 'text',
                    label_column: str = 'sentiment',
                    test_size: float = 0.2,
                    use_bert: bool = True,
                    epochs: int = 20,
                    batch_size: int = 16,
                    learning_rate: float = 2e-5,
                    max_length: int = 128) -> Dict[str, Any]:
        """训练逻辑已迁移到 sentiment_train.py。"""
        raise NotImplementedError(
            "Training has been moved to sentiment_train.py. "
            "Please call sentiment_train.train(...) / sentiment_train.finetune(...)."
        )

    def _train_bert_model(self,
                         train_df: pd.DataFrame,
                         text_column: str,
                         label_column: str,
                         test_size: float,
                         epochs: int,
                         batch_size: int,
                         learning_rate: float,
                         max_length: int) -> Dict[str, Any]:
        """兼容旧接口：训练逻辑已迁移到 sentiment_train.py。"""
        return self.train_model(
            train_df=train_df,
            text_column=text_column,
            label_column=label_column,
            test_size=test_size,
            use_bert=True,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
        )

    def _train_naive_bayes_model(self,
                                train_df: pd.DataFrame,
                                text_column: str,
                                label_column: str,
                                test_size: float) -> Dict[str, Any]:
        """兼容旧接口：训练逻辑已迁移到 sentiment_train.py。"""
        return self.train_model(
            train_df=train_df,
            text_column=text_column,
            label_column=label_column,
            test_size=test_size,
            use_bert=False,
            epochs=6,
            batch_size=16,
            learning_rate=2e-5,
            max_length=128,
        )

    # ==================== 高级分析方法 ====================
    
    def calculate_polarity(self, pos_count: int, neg_count: int) -> float:
        """
        计算情感极性分数
        
        参数:
            pos_count: 积极词数量
            neg_count: 消极词数量
            
        返回:
            float: 极性分数，范围 [-1, 1]
        """
        if pos_count < 0:
            logger.error(f"积极词不能为负: {pos_count}")
            return 0.0
        if neg_count < 0:
            logger.error(f"消极词不能为负: {neg_count}")
            return 0.0

        total_count = pos_count + neg_count
        if total_count == 0:
            return 0.0
        return (pos_count - neg_count) / total_count
    
    def analyze_readability(self, text: str) -> dict[Any, Any] | None:
        """
        分析文本可读性
        
        参数:
            text: 待分析的文本
            
        返回:
            可读性指标字典
        """
        if text is None or pd.isna(text) or str(text).strip() == '':
            logger.error("文本为空，无法分析文本可读性")
            return None

        try:
            if ct is None:
                logger.error("cntext 未安装，无法分析可读性")
                return None
            readability_dict = ct.readability(text)
            logger.debug(f"可读性分析完成: {readability_dict}")
            return readability_dict
        except Exception as e:
            logger.exception(f"可读性分析失败: {e}")
            return None
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        提取关键词（基于 TF-IDF）
        
        参数:
            text: 待分析的文本
            top_k: 返回前 k 个关键词
            
        返回:
            List[Tuple[str, float]]: [(关键词, 权重), ...]
         """
        if text is None or pd.isna(text) or str(text).strip() == '':
            logger.error("文本为空，无法提取关键词")
            return []

        try:
            import jieba.analyse
            keywords = jieba.analyse.extract_tags(text, topK=top_k, withWeight=True)
            logger.debug(f"成功提取 {len(keywords)} 个关键词")
            return keywords
        except Exception as e:
            logger.error(f"提取关键词失败: {e}")
            return []
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float | None:
        """
        计算两段文本的余弦相似度
        
        参数:
            text1: 文本1
            text2: 文本2
            
        返回:
            float: 相似度分数 (0-1)
        """
        if self._is_empty_text(text1) or self._is_empty_text(text2):
            logger.error("传入的文本1或文本2为空，无法计算余弦相似度")
            return None

        try:
            if ct is None:
                logger.error("cntext 未安装，无法计算相似度")
                return None
            similarity = ct.cosine_sim(text1, text2, lang='chinese')
            logger.debug(f"相似度: {similarity}")
            return float(similarity)
        except Exception as e:
            logger.exception(f"相似度计算失败: {e}")
            return 0.0
    
    # ==================== 模型持久化方法 ====================
    
    def save_model(self, path: str) -> None:
        """模型保存逻辑已迁移到 sentiment_train.py。"""
        raise NotImplementedError(
            "Model saving has been moved to sentiment_train.py. "
            "Please use sentiment_train.run_training_pipeline(...) / sentiment_train.train(...)."
        )

    def load_model(self, path: str) -> bool:
        """
        加载模型（自动识别 BERT 或朴素贝叶斯）

        参数:
            path: 模型文件路径

        返回:
            bool: 是否加载成功
        """
        try:
            path_obj = Path(path)

            # 检查是否是 BERT 模型目录
            if path_obj.is_dir() and (path_obj / 'config.json').exists():
                return self._load_bert_model(path)
            elif path_obj.suffix == '.pkl' or path_obj.is_file():
                return self._load_naive_bayes_model(path)
            else:
                logger.error(f"无法识别模型类型: {path}")
                return False

        except Exception as e:
            logger.exception(f"加载模型失败: {e}")
            return False

    def _load_bert_model(self, model_dir: str) -> bool:
        """加载 BERT 模型（委托给 ModelBackend）。"""
        return self.model_backend.load_bert_model(model_dir)

    def _load_naive_bayes_model(self, path: str) -> bool:
        """加载朴素贝叶斯模型（委托给 ModelBackend）。"""
        return self.model_backend.load_naive_bayes_model(path)

    # ==================== 统计分析方法 ====================
    
    def get_sentiment_distribution(self, df: pd.DataFrame) -> pd.Series:
        """
        统计情感分布

        参数:
            df: 包含 'sentiment' 列的 DataFrame

        返回:
            pd.Series: 各情感类别的数量统计
        """
        if 'sentiment' not in df.columns:
            logger.error("DataFrame 中不存在 'sentiment' 列")
            raise ValueError("Column 'sentiment' not found in DataFrame")

        if df.empty:
            logger.warning("输入 DataFrame 为空")
            return pd.Series(dtype=int)

        # 统计各情感类别的数量，用于快速观察整体情绪结构。
        distribution = df['sentiment'].value_counts().sort_index()

        logger.info(f"情感分布统计完成: {dict(distribution)}")

        return distribution

    def get_emotion_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        统计情绪分布
        
        参数:
            df: 包含情绪列的 DataFrame
            
        返回:
            pd.DataFrame: 各情绪类型的统计
        """
        if 'emotions' not in df.columns:
            logger.error("DataFrame 中不存在 'emotions' 列")
            raise ValueError("Column 'emotions' not found in DataFrame")

        if df.empty:
            logger.warning("输入 DataFrame 为空")
            return pd.DataFrame()

        emotion_counts = {}

        for emotions_dict in df['emotions']:
            if not isinstance(emotions_dict, dict):
                continue

            for emotion, count in emotions_dict.items():
                if emotion not in emotion_counts:
                    emotion_counts[emotion] = 0
                try:
                    emotion_counts[emotion] += int(count)
                except (ValueError, TypeError):
                    logger.warning(f"无效的情绪计数值: {emotion}={count}")
                    continue

        # 转换为 DataFrame，便于后续展示与排序。
        if not emotion_counts:
            logger.warning("未找到有效的情绪数据")
            return pd.DataFrame(columns=['emotion', 'count'])

        result_df = pd.DataFrame([
            {'emotion': emotion, 'count': count}
            for emotion, count in emotion_counts.items()
        ]).sort_values('count', ascending=False).reset_index(drop=True)

        logger.info(f"情绪分布统计完成，共 {len(result_df)} 种情绪")

        # 返回统计结果
        return result_df

    def analyze_sentiment_trend(self, df: pd.DataFrame,
                               time_column: str = 'visit_time',
                               freq: str = 'D') -> pd.DataFrame:
        """
        分析情感随时间的变化趋势

        参数:
            df: 包含情感和时间信息的 DataFrame
            time_column: 时间列名
            freq: 时间频率 ('D'=天, 'W'=周, 'M'=月)

        返回:
            pd.DataFrame: 时间序列的情感统计
        """
        required_columns = [time_column, 'sentiment', 'polarity']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"DataFrame 缺少必要的列: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        if df.empty:
            logger.warning("输入 DataFrame 为空")
            return pd.DataFrame()

        # 创建副本避免修改原数据
        df_copy = df.copy()

        # 确保时间列是 datetime 类型，便于按频率分组。
        if not pd.api.types.is_datetime64_any_dtype(df_copy[time_column]):
            try:
                df_copy[time_column] = pd.to_datetime(df_copy[time_column])
                logger.info(f"已将 '{time_column}' 列转换为 datetime 类型")
            except Exception as e:
                logger.error(f"无法将 '{time_column}' 转换为 datetime: {e}")
                raise ValueError(f"Cannot convert '{time_column}' to datetime: {e}")

        # 按时间频率聚合情感统计。
        try:
            grouped = df_copy.groupby(pd.Grouper(key=time_column, freq=freq))

            trend_data = grouped.agg({
                'polarity': ['mean', 'std', 'min', 'max'],
                'sentiment': 'count'
            }).reset_index()

            # 展平多层列名，便于下游直接使用。
            trend_data.columns = [
                time_column,
                'polarity_mean',
                'polarity_std',
                'polarity_min',
                'polarity_max',
                'count'
            ]

            # 标准差在仅一个样本的分组中会是 NaN，这里统一补 0。
            trend_data['polarity_std'] = trend_data['polarity_std'].fillna(0)

            # 计算移动平均，帮助观察趋势而非单点波动。
            if len(trend_data) >= 3:
                window_size = min(7, len(trend_data))
                trend_data['polarity_ma'] = trend_data['polarity_mean'].rolling(
                    window=window_size,
                    min_periods=1
                ).mean()
            else:
                trend_data['polarity_ma'] = trend_data['polarity_mean']

            # 统计各时间段的情感分布，并并入主结果。
            sentiment_dist = df_copy.groupby(
                [pd.Grouper(key=time_column, freq=freq), 'sentiment']
            ).size().unstack(fill_value=0)

            # 合并情感分布到趋势数据
            trend_data = trend_data.merge(
                sentiment_dist,
                left_on=time_column,
                right_index=True,
                how='left'
            )

            # 若某些时间段缺少某个情感类别，也补齐对应列，保持结构稳定。
            for sentiment_type in [self.SENTIMENT_POSITIVE, self.SENTIMENT_NEGATIVE, self.SENTIMENT_NEUTRAL]:
                if sentiment_type not in trend_data.columns:
                    trend_data[sentiment_type] = 0

            logger.info(f"情感趋势分析完成，共 {len(trend_data)} 个时间段")

            return trend_data

        except Exception as e:
            logger.exception(f"分析情感趋势时出错: {e}")
            raise

    def generate_sentiment_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        生成综合情感分析报告
        
        参数:
            df: 包含情感分析结果的 DataFrame
            
        返回:
            Dict[str, Any]: 综合报告
        """
        required_columns = ['sentiment', 'polarity', 'confidence']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"DataFrame 缺少必要的列: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        if df.empty:
            logger.warning("输入 DataFrame 为空")
            return {'error': 'Empty DataFrame'}

        df = df.copy()  # 使用副本，避免修改调用方原始数据
        df['polarity'] = pd.to_numeric(df['polarity'], errors='coerce')  # 转成数值，非法值记为 NaN
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')  # 转成数值，便于统计
        report: Dict[str, Any] = {
            'total_records': len(df),  # 报告覆盖的总记录数
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')  # 报告生成时间
        }

        sentiment_dist = df['sentiment'].value_counts(dropna=True).to_dict()
        sentiment_pct = (df['sentiment'].value_counts(normalize=True, dropna=True) * 100).round(2).to_dict()
        report['sentiment_distribution'] = {
            'counts': sentiment_dist,
            'percentages': sentiment_pct
        }

        polarity_stats = df['polarity'].describe().to_dict()
        report['polarity_statistics'] = {
            'mean': round(polarity_stats.get('mean', 0) or 0, 4),
            'std': round(polarity_stats.get('std', 0) or 0, 4),
            'min': round(polarity_stats.get('min', 0) or 0, 4),
            'max': round(polarity_stats.get('max', 0) or 0, 4),
            'median': round(df['polarity'].median() if df['polarity'].notna().any() else 0, 4),
            'q25': round(polarity_stats.get('25%', 0) or 0, 4),
            'q75': round(polarity_stats.get('75%', 0) or 0, 4)
        }

        if 'emotions' in df.columns:
            try:
                emotion_df = self.get_emotion_distribution(df)
                report['emotion_distribution'] = emotion_df.to_dict('records') if not emotion_df.empty else None
            except Exception as e:
                logger.warning(f"统计情绪分布出错: {e}")
                report['emotion_distribution'] = None
        else:
            report['emotion_distribution'] = None

        confidence_stats = df['confidence'].describe().to_dict()
        report['confidence_statistics'] = {
            'mean': round(confidence_stats.get('mean', 0) or 0, 4),
            'std': round(confidence_stats.get('std', 0) or 0, 4),
            'min': round(confidence_stats.get('min', 0) or 0, 4),
            'max': round(confidence_stats.get('max', 0) or 0, 4),
            'median': round(df['confidence'].median() if df['confidence'].notna().any() else 0, 4)
        }

        if 'pos_count' in df.columns and 'neg_count' in df.columns:
            report['word_statistics'] = {
                'total_positive_words': int(df['pos_count'].sum()),
                'total_negative_words': int(df['neg_count'].sum()),
                'avg_positive_words': round(df['pos_count'].mean(), 2),
                'avg_negative_words': round(df['neg_count'].mean(), 2)
            }

        if sentiment_dist:
            dominant = max(sentiment_dist, key=sentiment_dist.get)
        else:
            dominant = 'Unknown'
        report['overall_summary'] = {
            'dominant_sentiment': dominant,
            'overall_polarity': 'Positive' if report['polarity_statistics']['mean'] > 0.1
            else 'Negative' if report['polarity_statistics']['mean'] < -0.1
            else 'Neutral',
            'avg_confidence': round(report['confidence_statistics']['mean'], 4),
            'high_confidence_ratio': round(
                (df['confidence'] >= 0.7).sum() / len(df) * 100, 2
            )
        }

        logger.info("情感分析报告生成完成")
        return report

if __name__ == "__main__":
    if not CNTEXT_AVAILABLE:
        print("错误: cntext 未安装")
        print("请运行: pip install cntext")
        exit(1)

    print(f"cntext 版本: {getattr(ct, '__version__', '未知')}")

    if BERT_AVAILABLE:
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
    else:
        print("警告: BERT 功能不可用")
        print("如需使用 BERT，请安装: pip install torch transformers")

    def run_smoke_tests() -> None:
        """快速冒烟测试（不依赖训练模型）。"""
        print("\n=== 运行 sentiment.py 内置冒烟测试 ===")
        analyzer = SentimentAnalyzer(use_bert=False)

        samples = [
            "今天心情很好，工作很顺利，太开心了！",
            "糟糕透了，事情一团糟，我很难过。",
            "今天下雨了，我按时吃饭然后休息。",
            "",
            None,
        ]

        for i, text in enumerate(samples, start=1):
            try:
                result = analyzer.predict(text, include_emotions=True, include_words=False, use_custom_model=False)
                print(f"[单条预测-{i}] 输入: {repr(text)}")
                print(f"           输出: {result}")
            except Exception as e:
                print(f"[单条预测-{i}] 异常: {e}")

        df = pd.DataFrame({
            'title': [
                "非常满意这次体验，服务很好",
                "真失望，问题一直没解决",
                "天气一般，心情也一般",
                None,
            ],
            'visit_time': pd.date_range('2024-01-01', periods=4, freq='D')
        })

        result_df = analyzer.batch_predict(df, text_column='title', include_emotions=True, batch_size=2)
        print("\n[批量预测结果前几行]")
        print(result_df.head())

        report = analyzer.generate_sentiment_report(result_df)
        print("\n[综合报告]")
        print(report)

        trend = analyzer.analyze_sentiment_trend(result_df, time_column='visit_time', freq='D')
        print("\n[趋势分析]")
        print(trend.head())

        sim = analyzer.calculate_semantic_similarity("我今天很开心", "我现在很高兴")
        print(f"\n[语义相似度] {sim}")

        keywords = analyzer.extract_keywords("今天心情很好，工作效率提升，大家都很开心", top_k=5)
        print(f"[关键词] {keywords}")

        readability = analyzer.analyze_readability("今天阳光明媚，我们一起去公园散步。")
        print(f"[可读性] {readability}")

        print("=== 冒烟测试完成 ===\n")

    print("\nchoose:1.模型相关；2.词典相关；3.冒烟测试\n")
    op = int(input().strip())

    if op == 1:
        analyzer = SentimentAnalyzer(use_bert=False)
        analyzer.load_model('./models/sentiment_train')

        text1 = '''
        今天我非常高兴！因为在家庭拼音本上得到了红星和“优”，这是因为我的拼音书写得很工整。老师在大家面前表扬了我，还要我也分享了我的小经验。我心\
        里像吃了蜂蜜一样甜，开心极了。今后我一定要更加认真地写作业，争取每天都得到红星！ 
        '''
        text2 = '''
        今天，我拿到乐理考试的成绩单，上面是一个鲜红的“不合格”。心瞬间揪在了一起，眼泪在眼眶里打转。虽然老师和妈妈安慰我说明年还可以考，但我心里\
        还是很难受，觉得辜负了这段时间的努力。看着书桌上的乐谱，我暗暗发誓：一定要加倍努力，下一次我一定要考过！
        '''
        text3 = '''
        清晨，太阳悄悄爬上地平线，金色的阳光洒满校园。操场上，露珠在绿草尖上闪烁。同学们背着书包，陆陆续续走进校门。操场那边，几位晨练的老师正在\
        慢跑。教学楼里传出朗朗的读书声，开启了新的一天。校园的早晨真安静，也充满了生机。
        '''

        df = pd.DataFrame({
            'title': [
                text1,
                text2,
                text3,
            ],
            'visit_time': pd.date_range('2024-01-01', periods=3, freq='D')
        })

        result_df = analyzer.batch_predict(df, text_column='title', include_emotions=True, batch_size=2)
        print("\n[批量预测结果前几行]")
        print(result_df.head())

    elif op == 2:
        d = ct.read_yaml_dict("zh_common_NTUSD.yaml")
        dict_only = d.get("Dictionary", {})
        print("词典键:", d.keys())
        text = "今天心情很好！"
        raw = ct.sentiment(text, diction=dict_only)
        print("raw:", raw)
    elif op == 3:
        run_smoke_tests()
    else:
        print("无效选项，请输入 1/2/3")
