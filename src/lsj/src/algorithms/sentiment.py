# -*- coding: utf-8 -*-  # 声明文件编码，避免中文乱码
# from __future__ import annotations  # 让 Python 3.8/3.9 支持 | 类型注解
"""
情感分析模块
功能概述：
    使用 cntext 库对浏览记录进行多维度情感和心理分析
    - 情感分析：积极、消极、中性情感倾向
    - 情绪分析：喜悦、愤怒、悲伤、恐惧等具体情绪
    - 心理特征：态度、认知、价值观等抽象构念
    - 语义分析：主题、关键词、语义相似度
依赖库：
    - cntext: 中文文本分析工具包（核心）
    - jieba: 中文分词
    - pandas: 数据处理
    
安装 cntext：
    pip install cntext
    
参考文档：
    - cntext GitHub: https://github.com/hidadeng/cntext
"""
# ======== 环境变量设置（必须在 transformers 之前设置） ========
import os  # 导入系统环境变量模块
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 设置 HuggingFace 国内镜像
# ======== 标准库导入 ========
import logging  # 日志模块
import pickle  # 模型持久化
from pathlib import Path  # 跨平台路径处理
from typing import List, Dict, Optional, Tuple, Any  # 类型标注
import csv  # CSV 文件格式识别
# ======== 第三方库导入 ========
import jieba  # 中文分词
import pandas as pd  # 数据处理
import yaml  # YAML 读取
# ==================== Logger 标准化 ====================
def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """创建标准化 logger，避免重复 handler"""
    logger_obj = logging.getLogger(name)  # 获取 logger
    logger_obj.setLevel(level)  # 设置日志级别
    # 如果 handler 已存在，直接返回避免重复
    if logger_obj.handlers:
        return logger_obj
    # 创建日志文件夹（如不存在）
    log_path = Path(log_file)  # 转成 Path 对象
    log_path.parent.mkdir(parents=True, exist_ok=True)  # 创建目录
    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # 文件输出
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    # 添加 handler
    logger_obj.addHandler(file_handler)
    logger_obj.addHandler(console_handler)
    # 防止日志重复传播
    logger_obj.propagate = False
    return logger_obj
# 初始化 logger
logger = setup_logger(__name__, "../../logs/sentiment.log")
# ==================== 判断依赖是否存在 ====================
import importlib.util  # 动态检查模块是否安装
def _pkg_exists(name: str) -> bool:
    """判断某个包是否可被导入"""
    return importlib.util.find_spec(name) is not None  # 返回 True/False
# ==================== cntext 导入 ====================
ct = None  # 默认值，避免 NameError
CNTEXT_AVAILABLE = False  # 标记 cntext 是否可用
if not _pkg_exists("cntext"):  # 如果未安装 cntext
    logger.warning("cntext 未安装（find_spec 找不到），请运行: python -m pip install cntext")
else:
    try:
        import cntext as ct  # 尝试导入
        CNTEXT_AVAILABLE = True  # 标记可用
        logger.info("cntext 加载成功: %s, version=%s", ct.__file__, getattr(ct, "__version__", "未知"))
    except Exception as e:
        CNTEXT_AVAILABLE = False  # 导入失败
        ct = None  # 清空引用
        logger.exception("cntext 已安装但导入失败。异常: %r", e)
# ==================== BERT 相关导入 ====================
torch = None  # 默认 torch
Dataset = object  # 默认 Dataset
DataLoader = None  # 默认 DataLoader
BERT_AVAILABLE = False  # 标记 BERT 是否可用
try:
    import torch  # PyTorch
    from torch.utils.data import Dataset, DataLoader  # 数据集与加载器
    from transformers import BertTokenizer, BertForSequenceClassification  # BERT 模型
    from torch.optim import AdamW  # 优化器
    try:
        from transformers import get_linear_schedule_with_warmup  # 新版位置
    except Exception:
        from transformers.optimization import get_linear_schedule_with_warmup  # 旧版位置
    BERT_AVAILABLE = True  # 标记可用
    logger.info("BERT 相关依赖加载成功：torch=%s, transformers 已可用", torch.__version__)
except Exception as e:
    BERT_AVAILABLE = False  # 标记不可用
    logger.exception("BERT 相关依赖导入失败。异常: %r", e)
# ==================== 只有 BERT 可用时才定义 Dataset ====================
if BERT_AVAILABLE:
    class SentimentDataset(Dataset):
        """BERT 情感分析数据集"""
        def __init__(self, texts: List[str], labels: List[int],
                     tokenizer, max_length: int = 128):
            self.texts = texts  # 保存文本
            self.labels = labels  # 保存标签
            self.tokenizer = tokenizer  # 保存 tokenizer
            self.max_length = max_length  # 最大长度
        def __len__(self):
            return len(self.texts)  # 返回样本数量
        def __getitem__(self, idx):
            text = str(self.texts[idx])  # 取出文本
            label = self.labels[idx]  # 取出标签
            # 编码文本
            encoding = self.tokenizer(
                text,
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
                "label": torch.tensor(label, dtype=torch.long),
            }
else:
    class SentimentDataset:
        """BERT 不可用时的占位类"""
        def __init__(self, *args, **kwargs):
            raise ImportError("BERT 不可用，无法使用 SentimentDataset")
# ==================== SentimentAnalyzer 主类 ====================
class SentimentAnalyzer:
    """
    情感分析器（基于 cntext）
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
                 stopwords_path: Optional[str] = './rules/hit_stopwords.txt',
                 bert_model_name: str = 'bert-base-chinese',
                 use_bert: bool = False):
        if not CNTEXT_AVAILABLE:
            logger.error("没有安装 cntext 库，建议 pip install cntext")
            raise ImportError("cntext is required but not installed")
        self.diction = diction  # 保存词典名
        self._cntext_dict_cache: Optional[Dict[str, Any]] = None  # 缓存词典
        self.stopwords_path = stopwords_path  # 停用词路径
        # ===== 加载词典 =====
        if isinstance(self.diction, dict):
            loaded = self.diction
        elif isinstance(self.diction, str):
            try:
                if self.diction.lower().endswith((".yaml", ".yml")):
                    loaded = ct.read_yaml_dict(self.diction)
                else:
                    logger.warning("diction 建议使用内置 yaml 名")
                    loaded = None
            except Exception as e:
                logger.exception(f"读取 YAML 词典失败: {e}")
                loaded = None
        else:
            loaded = None
        if isinstance(loaded, dict):
            self._cntext_dict_cache = loaded.get('Dictionary', loaded)
            if self._cntext_dict_cache:
                logger.info(f"词典加载成功，包含键: {list(self._cntext_dict_cache.keys())[:10]}")
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
                    logger.warning("模型文件中没有 vectorizer")
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
        if self.use_bert:
            logger.info(f"BERT 模式已启用，设备: {self.device}")
        logger.info(f"初始化完成 - 词典: {diction}, BERT: {self.use_bert}")

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
        if text is None or pd.isna(text) or str(text).strip() == "":
            logger.error("传入文本为 None")
            return []

        try:
            words = jieba.lcut(text)
        except Exception as e:
            logger.exception(f"出现异常，分词失败: {e}")
            return []

        def load_stopwords(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    stopwords = {line.strip() for line in f if line.strip()}
                return stopwords
            except OSError:
                logger.warning(f"停用词词典不存在: {path}")
                return {}
            except Exception as e:
                logger.exception(f"出现未知错误: {e}")
                return {}

        stopwords_set = load_stopwords(self.stopwords_path)

        filtered_words = [word for word in words if word not in stopwords_set and len(word.strip()) > 0]

        return filtered_words
    
    def _calculate_sentiment_score_cntext(self, text: str) -> Dict[str, Any]:
        """
        使用 cntext 计算情感相关统计

        参数:
            text: 待分析的文本

        返回:
            Dict[str, Any]: {
                'pos': int,              # 正向词数（若词典不提供则为 0）
                'neg': int,              # 负向词数（若词典不提供则为 0）
                'pos_word': List[str],   # cntext 文档的标准返回不一定包含，保留兼容
                'neg_word': List[str],
                'categories': Dict[str, int],  # 其它维度的计数（*_num）
                'raw': Dict[str, Any],   # 原始返回，便于调试
            }
        """
        if text is None or pd.isna(text) or str(text).strip() == "":
            return {'pos': 0, 'neg': 0, 'pos_word': [], 'neg_word': [], 'categories': {}, 'raw': {}}

        try:
            if ct is None:
                logger.error("cntext 未安装，无法进行情感分析")
                return {'pos': 0, 'neg': 0, 'pos_word': [], 'neg_word': [], 'categories': {}, 'raw': {}}

            # 优先使用自定义词典，否则使用缓存的词典
            if self.custom_dict:
                logger.debug("使用自定义词典进行分析")
                raw = ct.sentiment(str(text), diction=self.custom_dict)
            elif self._cntext_dict_cache is not None:
                logger.debug("使用缓存词典进行分析")
                raw = ct.sentiment(str(text), diction=self._cntext_dict_cache)
            else:
                # 尝试使用默认词典
                logger.warning("未加载词典，尝试使用 cntext 默认行为")
                try:
                    raw = ct.sentiment(str(text), diction=ct.read_yaml_dict('zh_common_DUTIR.yaml'))
                except Exception as e:
                    logger.error(f"使用默认词典失败: {e}")
                    return {'pos': 0, 'neg': 0, 'pos_word': [], 'neg_word': [], 'categories': {}, 'raw': {}}

            # 调试：打印原始结果
            logger.debug(f"cntext 原始返回: {raw}")

            # ---- 字段归一化：兼容不同版本/不同词典的返回格式 ----
            # 尝试多种可能的字段名
            pos = raw.get('pos_num', raw.get('pos', 0))
            neg = raw.get('neg_num', raw.get('neg', 0))

            # 处理词列表字段
            pos_words = raw.get('pos_word', raw.get('pos_words', []))
            neg_words = raw.get('neg_word', raw.get('neg_words', []))

            # 如果没有 pos/neg，尝试从情绪类别映射（DUTIR 词典）
            if pos == 0 and neg == 0:
                # 积极情绪：乐(喜)、好
                pos_emotions = ['乐_num', '喜_num', '好_num']
                # 消极情绪：怒(愤)、哀(悲)、惧(恐)、恶(厌)、惊(惊讶)
                neg_emotions = ['怒_num', '愤_num', '哀_num', '悲_num', '惧_num', '恐_num', '恶_num', '厌_num', '惊_num', '惊讶_num']

                # 计算积极情绪总数
                for key in pos_emotions:
                    if key in raw:
                        pos += raw[key]

                # 计算消极情绪总数
                for key in neg_emotions:
                    if key in raw:
                        neg += raw[key]

                logger.debug(f"从情绪类别映射 - pos: {pos}, neg: {neg}")

            # 确保是列表类型
            if not isinstance(pos_words, list):
                pos_words = []
            if not isinstance(neg_words, list):
                neg_words = []

            logger.debug(f"提取结果 - pos: {pos}, neg: {neg}, pos_words: {len(pos_words)}, neg_words: {len(neg_words)}")

            # 收集其它 *_num 计数字段（排除通用统计字段）
            exclude = {'stopword_num', 'word_num', 'sentence_num'}
            categories = {
                k: v for k, v in raw.items()
                if isinstance(k, str) and k.endswith('_num') and k not in exclude
            }

            if pos == 0 and neg == 0 and categories:
                logger.debug(
                    "当前 diction 返回的结果未包含 pos/neg 维度，可能是多维度计数词典。"
                    "你可以用 categories 做进一步的情绪/心理维度分析。"
                )

            return {
                'pos': int(pos) if pd.notna(pos) else 0,
                'neg': int(neg) if pd.notna(neg) else 0,
                'pos_word': list(pos_words) if isinstance(pos_words, list) else [],
                'neg_word': list(neg_words) if isinstance(neg_words, list) else [],
                'categories': categories,
                'raw': raw,
            }

        except Exception as e:
            logger.exception(f"情感分析失败: {e}")
            return {'pos': 0, 'neg': 0, 'pos_word': [], 'neg_word': [], 'categories': {}, 'raw': {}}

    
    def _analyze_emotions_cntext(self, text: str) -> Dict[str, Any]:
        """使用 cntext 分析具体情绪（仅“情绪类别型词典”支持；DUTIR 属于这一类）

        参数:
            text: 待分析文本

        返回:
            Dict[str, Any]: {"Joy": 1, "Anger": 0, ...}
        """
        if text is None or pd.isna(text) or str(text).strip() == "":
            logger.error("传入文本为空")
            return {}

        expected_emotion_keys = {
            '乐', '怒', '哀', '惧', '恶', '惊', '好',
            '喜', '愤', '悲', '恐', '厌', '惊讶',
        }

        def _looks_like_emotion_dict(d: Any) -> bool:
            """判断一个 diction（Python dict）是否像“情绪类别型词典”。"""
            if not isinstance(d, dict) or not d:
                return False
            return any((k in expected_emotion_keys) for k in d.keys())

        diction_for_emotion: Optional[Dict[str, Any]] = None
        if _looks_like_emotion_dict(self.custom_dict):
            diction_for_emotion = self.custom_dict
        elif _looks_like_emotion_dict(self._cntext_dict_cache):
            diction_for_emotion = self._cntext_dict_cache
        else:
            logger.info(
                "当前词典不包含情绪类别 key（如 乐/怒/哀/惧/恶/惊/好），跳过情绪分析。"
            )
            return {}

        emotion_key_mapping = {
            '乐': self.EMOTION_JOY,
            '喜': self.EMOTION_JOY,
            '怒': self.EMOTION_ANGER,
            '愤': self.EMOTION_ANGER,
            '哀': self.EMOTION_SADNESS,
            '悲': self.EMOTION_SADNESS,
            '惧': self.EMOTION_FEAR,
            '恐': self.EMOTION_FEAR,
            '恶': self.EMOTION_DISGUST,
            '厌': self.EMOTION_DISGUST,
            '惊': self.EMOTION_SURPRISE,
            '惊讶': self.EMOTION_SURPRISE,
            '好': self.EMOTION_GOOD,
        }

        try:
            if ct is None:
                logger.error("cntext 未安装，无法进行情绪分析")
                return {}
            raw = ct.sentiment(str(text), diction=diction_for_emotion)

            exclude = {'stopword_num', 'word_num', 'sentence_num'}

            emotions: Dict[str, int] = {}
            for key, value in raw.items():
                if not (isinstance(key, str) and key.endswith('_num')):
                    continue
                if key in exclude:
                    continue

                category = key[:-4]

                if category not in expected_emotion_keys:
                    continue

                mapped = emotion_key_mapping.get(category)
                if mapped is None:
                    logger.debug(f"发现未映射的情绪类别: {category}")
                    continue

                try:
                    emotions[mapped] = emotions.get(mapped, 0) + int(value)
                except Exception:
                    continue

            return emotions

        except Exception as e:
            logger.exception(f"情绪分析失败: {e}")
            return {}

    def _score_to_sentiment(self, pos_count: int, neg_count: int,
                           threshold: float = 0.1) -> str:
        """
        将情感分数转换为情感类别

        参数:
            pos_count: 积极词数量
            neg_count: 消极词数量
            threshold: 判断阈值

        返回:
            str: 情感类别（SENTIMENT_POSITIVE/NEGATIVE/NEUTRAL）
        """
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
        """
        返回空文本的默认结果
        
        返回:
            Dict[str, Any]: 默认结果
        """
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
        if text is None or pd.isna(text) or str(text).strip() == '':
            logger.error("传入文本为 None")
            return {
                'sentiment': self.SENTIMENT_NEUTRAL,
                'polarity': 0,
                'sentiment_scores': {'pos': 0, 'neg': 0},
                'emotions': None,
                'pos_words': [],
                'neg_words': []
            }

        sentiment_score = self._calculate_sentiment_score_cntext(text)

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
        """
        基于自定义机器学习模型进行情感分析（支持 BERT 和朴素贝叶斯）

        参数:
            text: 待分析的文本

        返回:
            str: 情感类别
        """
        # 检查文本
        if text is None or pd.isna(text) or str(text).strip() == "":
            logger.warning("文本为空")
            return None
        
        try:
            if self.use_bert and self.bert_model is not None:
                logger.info("predict_by_model: 当前使用 BERT 进行预测")
                return self._predict_by_bert(text)
            elif self.model is not None:
                logger.info("predict_by_model: 当前使用 Naive Bayes 进行预测")
                return self._predict_by_naive_bayes(text)
            else:
                logger.error("predict_by_model: 没有可用模型（BERT/NB 都未加载）")
                return None
        except Exception as e:
            logger.exception(f"模型预测失败: {e}")
            return None
    
    def _predict_by_bert(self, text: str) -> str:
        """
        使用 BERT 模型预测

        参数:
            text: 待分析的文本

        返回:
            str: 情感类别
        """
        self.bert_model.eval()

        # 编码文本
        # 直接调用 tokenizer(...)，兼容 Transformers 5.x
        encoding = self.bert_tokenizer(
            text,  # 输入文本
            add_special_tokens=True,  # 自动加特殊 token
            max_length=128,  # 最大长度
            padding="max_length",  # padding 到固定长度
            truncation=True,  # 超长截断
            return_attention_mask=True,  # 返回 mask
            return_tensors="pt",  # 返回 torch 张量
        )
        # input_ids / attention_mask shape 是 [1, 128]，直接放到 device 上
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # 预测
        with torch.no_grad():
            outputs = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]

        # 转换回原始标签
        if self.label_encoder is not None:
            prediction_label = self.label_encoder.inverse_transform([prediction])[0]
        else:
            prediction_label = str(prediction)

        logger.debug(f"BERT 预测结果: {prediction_label}")
        return prediction_label

    def _predict_by_naive_bayes(self, text: str) -> str | None:
        """
        使用朴素贝叶斯模型预测（原有实现）

        参数:
            text: 待分析的文本

        返回:
            str: 情感类别
        """
        if self.model is None:
            logger.error("模型未加载")
            return None

        if self.vectorizer is None:
            logger.error("向量化器未加载")
            return None

        # 文本分词
        words = self._segment_text(text)
        text_processed = ' '.join(words)

        # 文本向量化
        X = self.vectorizer.transform([text_processed])

        # 模型预测
        prediction = self.model.predict(X)[0]

        logger.debug(f"朴素贝叶斯预测结果: {prediction}")
        return str(prediction)

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
        if text is None or pd.isna(text) or str(text).strip() == '':
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

        final_result = {
            'sentiment': sentiment,
            'polarity': polarity,
            'pos_count': pos_count,
            'neg_count': neg_count,
            'confidence': confidence
        }

        has_custom_model = (self.model is not None) or (self.use_bert and self.bert_model is not None)
        if use_custom_model and has_custom_model:
            model_sentiment = self.predict_by_model(text)
            if model_sentiment:
                if model_sentiment != sentiment:
                    logger.info(f"模型预测 ({model_sentiment}) 与词典预测 ({sentiment}) 不一致")
                    final_result['confidence'] *= 0.7
                    final_result['model_sentiment'] = model_sentiment
                    final_result['sentiment'] = model_sentiment
                else:
                    final_result['confidence'] = min(confidence * 1.2, 1.0)

        if include_words:
            final_result['pos_words'] = result.get('pos_words', [])
            final_result['neg_words'] = result.get('neg_words', [])

        if include_emotions and 'emotions' in result:
            final_result['emotions'] = result['emotions']

        logger.info(f"预测完成: {sentiment} (置信度: {confidence:.3f})")
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
                        # 预测失败，使用默认值
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

        # 将结果添加到 DataFrame
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

    # ==================== 模型训练方法 ====================
    
    def train_model(self,
                    train_df: pd.DataFrame,
                    text_column: str = 'text',
                    label_column: str = 'sentiment',
                    test_size: float = 0.2,
                    use_bert: bool = True,
                    epochs: int = 3,
                    batch_size: int = 16,
                    learning_rate: float = 2e-5,
                    max_length: int = 128) -> Dict[str, Any]:
        """
        训练情感分析模型（支持 BERT 和朴素贝叶斯）

        参数:
            train_df: 训练数据
            text_column: 文本列名
            label_column: 标签列名
            test_size: 测试集比例
            use_bert: 是否使用 BERT（默认 True）
            epochs: 训练轮数（仅 BERT）
            batch_size: 批次大小（仅 BERT）
            learning_rate: 学习率（仅 BERT）
            max_length: 最大序列长度（仅 BERT）

        返回:
            Dict[str, Any]: 训练结果，包含准确率、损失等信息
        """
        if text_column not in train_df.columns or label_column not in train_df.columns:
            raise ValueError(f"列 '{text_column}' 或 '{label_column}' 不存在")

        logger.info(f"开始训练模型，数据量: {len(train_df)}")

        if use_bert and BERT_AVAILABLE:
            return self._train_bert_model(
                train_df, text_column, label_column, test_size,
                epochs, batch_size, learning_rate, max_length
            )
        else:
            return self._train_naive_bayes_model(
                train_df, text_column, label_column, test_size
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
        """
        使用 BERT 训练模型

        参数:
            train_df: 训练数据
            text_column: 文本列名
            label_column: 标签列名
            test_size: 测试集比例
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            max_length: 最大序列长度

        返回:
            Dict[str, Any]: 训练结果
        """
        global predictions, true_labels, accuracy, f1
        from sklearn.model_selection import train_test_split  # 数据集切分
        from sklearn.preprocessing import LabelEncoder  # 标签编码
        from sklearn.metrics import classification_report, accuracy_score, f1_score  # 评估指标

        logger.info("=" * 50)
        logger.info("开始 BERT 模型训练")
        logger.info("=" * 50)

        texts = train_df[text_column].astype(str).tolist()  # 文本列表
        labels = train_df[label_column].tolist()  # 标签列表

        self.label_encoder = LabelEncoder()  # 初始化编码器
        labels_encoded = self.label_encoder.fit_transform(labels)  # 转换为数字标签
        num_labels = len(self.label_encoder.classes_)  # 类别数
        logger.info(f"标签类别: {self.label_encoder.classes_}")
        logger.info(f"类别数量: {num_labels}")

        X_train, X_test, y_train, y_test = train_test_split(
            texts,  # 文本
            labels_encoded,  # 标签
            test_size=test_size,  # 测试比例
            random_state=42,  # 随机种子
            stratify=labels_encoded  # 保持类别分布一致
        )

        logger.info(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

        logger.info(f"加载 BERT 模型: {self.bert_model_name}")

        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)  # 分词器
        self.bert_model = BertForSequenceClassification.from_pretrained(
            self.bert_model_name,  # 模型名称
            num_labels=num_labels  # 类别数
        )

        self.bert_model.to(self.device)  # 放到 GPU/CPU

        train_dataset = SentimentDataset(X_train, y_train, self.bert_tokenizer, max_length)  # 训练集
        test_dataset = SentimentDataset(X_test, y_test, self.bert_tokenizer, max_length)  # 测试集

        # DataLoader 加速参数（CPU/GPU 都可用）
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # 打乱数据
            num_workers=2,  # 子进程加载（Windows 建议 0~2）
            pin_memory=True  # GPU 加速拷贝
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True
        )

        optimizer = AdamW(self.bert_model.parameters(), lr=learning_rate)  # AdamW 优化器
        total_steps = len(train_loader) * epochs  # 总步数
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,  # 不做 warmup
            num_training_steps=total_steps
        )

        # 混合精度
        use_amp = torch.cuda.is_available()  # 仅在 GPU 上用混合精度
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  # AMP 缩放器

        logger.info("开始训练...")
        training_stats = []  # 用于记录训练指标

        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            logger.info("-" * 50)
            # ===== 训练阶段 =====
            self.bert_model.train()  # 开启训练模式
            total_train_loss = 0  # 训练损失累计

            try:
                from tqdm import tqdm  # 进度条
                train_iterator = tqdm(train_loader, desc=f"训练 Epoch {epoch + 1}")
            except ImportError:
                tqdm = None
                train_iterator = train_loader
            for batch in train_iterator:

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = self.bert_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss  # 取 loss
                total_train_loss += loss.item()  # 记录 loss

                scaler.scale(loss).backward()  # 反向传播 + 缩放

                torch.nn.utils.clip_grad_norm_(self.bert_model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_loader)
            logger.info(f"平均训练损失: {avg_train_loss:.4f}")

            logger.info("开始验证...")
            self.bert_model.eval()  # 开启验证模式

            predictions = []  # 保存预测结果
            true_labels = []  # 保存真实标签
            total_eval_loss = 0  # 验证损失累计

            with torch.inference_mode():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    outputs = self.bert_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    total_eval_loss += loss.item()
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    predictions.extend(preds)
                    true_labels.extend(labels.cpu().numpy())

            avg_eval_loss = total_eval_loss / len(test_loader)
            accuracy = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions, average='weighted')

            logger.info(f"验证损失: {avg_eval_loss:.4f}")
            logger.info(f"准确率: {accuracy:.4f}")
            logger.info(f"F1 分数: {f1:.4f}")

            training_stats.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'eval_loss': avg_eval_loss,
                'accuracy': accuracy,
                'f1_score': f1
            })

        logger.info("\n" + "=" * 50)
        logger.info("训练完成！最终评估结果：")
        logger.info("=" * 50)
        predictions_labels = self.label_encoder.inverse_transform(predictions)
        true_labels_original = self.label_encoder.inverse_transform(true_labels)
        report = classification_report(true_labels_original, predictions_labels)
        logger.info("\n分类报告:\n" + report)
        self.use_bert = True  # 标记 BERT 可用
        return {
            'model_type': 'BERT',
            'accuracy': accuracy,
            'f1_score': f1,
            'training_stats': training_stats,
            'classification_report': report
        }

    def _train_naive_bayes_model(self,
                                train_df: pd.DataFrame,
                                text_column: str,
                                label_column: str,
                                test_size: float) -> Dict[str, Any]:
        """
        使用朴素贝叶斯训练模型（原有实现）

        参数:
            train_df: 训练数据
            text_column: 文本列名
            label_column: 标签列名
            test_size: 测试集比例

        返回:
            Dict[str, Any]: 训练结果
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.metrics import classification_report, accuracy_score

        logger.info("=" * 50)
        logger.info("开始朴素贝叶斯模型训练")
        logger.info("=" * 50)

        # 数据预处理（分词）
        logger.info("正在分词...")
        texts = []
        for text in train_df[text_column]:
            words = self._segment_text(str(text))
            texts.append(' '.join(words))

        labels = train_df[label_column].values

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42
        )

        logger.info(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

        # 特征提取（TF-IDF）
        logger.info("正在提取特征...")
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # 模型训练（朴素贝叶斯）
        logger.info("正在训练模型...")
        self.model = MultinomialNB()
        self.model.fit(X_train_vec, y_train)

        # 模型评估
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        logger.info(f"模型训练完成，准确率: {accuracy:.4f}")
        logger.info("\n分类报告:\n" + report)

        return {
            'model_type': 'Naive Bayes',
            'accuracy': accuracy,
            'classification_report': report
        }

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
        def is_text_Empty(text) -> bool:
            if text is None or pd.isna(text) or str(text).strip() == '':
                return True
            return False
        if is_text_Empty(text1) or is_text_Empty(text2):
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
        """
        保存训练好的模型（支持 BERT 和朴素贝叶斯）

        参数:
            path: 模型保存路径
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self.use_bert and self.bert_model is not None:
            # 保存 BERT 模型
            logger.info("保存 BERT 模型...")

            # 创建模型目录
            model_dir = save_path.parent / f"{save_path.stem}_bert"
            model_dir.mkdir(exist_ok=True)

            # 保存模型和 tokenizer
            self.bert_model.save_pretrained(model_dir)
            self.bert_tokenizer.save_pretrained(model_dir)

            # 保存标签编码器和其他元数据
            metadata = {
                'model_type': 'BERT',
                'label_encoder': self.label_encoder,
                'bert_model_name': self.bert_model_name,
                'diction': self.diction
            }

            with open(model_dir / 'metadata.pkl', 'wb') as f:
                pickle.dump(metadata, f)

            logger.info(f"BERT 模型已保存到: {model_dir}")

        elif self.model is not None:
            # 保存朴素贝叶斯模型
            logger.info("保存朴素贝叶斯模型...")

            model_data = {
                'model_type': 'Naive Bayes',
                'model': self.model,
                'vectorizer': self.vectorizer,
                'diction': self.diction
            }

            with open(path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"朴素贝叶斯模型已保存到: {path}")
        else:
            logger.error("没有可保存的模型")
            raise ValueError("No model to save")

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
        """
        加载 BERT 模型

        参数:
            model_dir: BERT 模型目录

        返回:
            bool: 是否加载成功
        """
        if not BERT_AVAILABLE:
            logger.error("BERT 不可用")
            return False

        try:
            model_path = Path(model_dir)

            logger.info(f"加载 BERT 模型: {model_path}")

            # 加载模型和 tokenizer
            self.bert_model = BertForSequenceClassification.from_pretrained(model_path)
            self.bert_tokenizer = BertTokenizer.from_pretrained(model_path)
            self.bert_model.to(self.device)

            # 加载元数据
            metadata_path = model_path / 'metadata.pkl'
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)

                self.label_encoder = metadata.get('label_encoder')
                self.bert_model_name = metadata.get('bert_model_name', 'bert-base-chinese')
                logger.info(f"模型使用的词典: {metadata.get('diction')}")

            self.use_bert = True
            logger.info("BERT 模型加载成功")
            return True

        except Exception as e:
            logger.exception(f"加载 BERT 模型失败: {e}")
            return False

    def _load_naive_bayes_model(self, path: str) -> bool:
        """
        加载朴素贝叶斯模型（原有实现）

        参数:
            path: 模型文件路径

        返回:
            bool: 是否加载成功
        """
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)

            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.vectorizer = model_data.get('vectorizer')
                loaded_diction = model_data.get('diction')

                if loaded_diction:
                    logger.info(f"模型使用的词典: {loaded_diction}")
            else:
                self.model = model_data
                self.vectorizer = None

            self.use_bert = False
            logger.info(f"朴素贝叶斯模型已从 {path} 加载")
            return True

        except Exception as e:
            logger.exception(f"加载朴素贝叶斯模型失败: {e}")
            return False

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

        # 统计各情感类别的数量
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

        # 转换为 DataFrame
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

        # 确保时间列是 datetime 类型
        if not pd.api.types.is_datetime64_any_dtype(df_copy[time_column]):
            try:
                df_copy[time_column] = pd.to_datetime(df_copy[time_column])
                logger.info(f"已将 '{time_column}' 列转换为 datetime 类型")
            except Exception as e:
                logger.error(f"无法将 '{time_column}' 转换为 datetime: {e}")
                raise ValueError(f"Cannot convert '{time_column}' to datetime: {e}")

        # 按时间分组统计
        try:
            grouped = df_copy.groupby(pd.Grouper(key=time_column, freq=freq))

            trend_data = grouped.agg({
                'polarity': ['mean', 'std', 'min', 'max'],
                'sentiment': 'count'
            }).reset_index()

            # 扁平化列名
            trend_data.columns = [
                time_column,
                'polarity_mean',
                'polarity_std',
                'polarity_min',
                'polarity_max',
                'count'
            ]

            # 填充 NaN 值（标准差在只有一个样本时为 NaN）
            trend_data['polarity_std'] = trend_data['polarity_std'].fillna(0)

            # 计算移动平均（7 期窗口）
            if len(trend_data) >= 3:
                window_size = min(7, len(trend_data))
                trend_data['polarity_ma'] = trend_data['polarity_mean'].rolling(
                    window=window_size,
                    min_periods=1
                ).mean()
            else:
                trend_data['polarity_ma'] = trend_data['polarity_mean']

            # 统计各时间段的情感分布
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

            # 填充缺失的情感类别列
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

        df = df.copy()  # 防止修改原数据
        df['polarity'] = pd.to_numeric(df['polarity'], errors='coerce')  # 转换为数字
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')  # 转换为数字
        report: Dict[str, Any] = {
            'total_records': len(df),  # 总记录数
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')  # 分析时间
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

    print(f"cntext 版本: {ct.__version__}")

    if BERT_AVAILABLE:
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
    else:
        print("警告: BERT 功能不可用")
        print("如需使用 BERT，请安装: pip install torch transformers")