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
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import logging
import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import jieba
import pandas as pd
import yaml
import csv

# logger 基本设置
logs_folder_path = "../../logs"
if not os.path.exists(logs_folder_path):
    os.makedirs(logs_folder_path)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../../logs/sentiment.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

import importlib.util
def _pkg_exists(name: str) -> bool:
    """判断某个包在当前解释器环境里是否可被找到。"""
    return importlib.util.find_spec(name) is not None
# cntext 相关导入
ct = None  # 先给默认值，避免后面 NameError
CNTEXT_AVAILABLE = False
if not _pkg_exists("cntext"):
    logger.warning("cntext 未安装（find_spec 找不到），请运行: python -m pip install cntext")
else:
    try:
        import cntext as ct
        CNTEXT_AVAILABLE = True
        logger.info("cntext 加载成功: %s, version=%s", ct.__file__, getattr(ct, "__version__", "未知"))
    except Exception as e:
        CNTEXT_AVAILABLE = False
        ct = None
        logger.exception("cntext 已安装但导入失败（不是未安装）。真实异常如下：%r", e)
# BERT 相关导入
torch = None
Dataset = object
DataLoader = None
BERT_AVAILABLE = False
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import BertTokenizer, BertForSequenceClassification
    from torch.optim import AdamW
    try:
        from transformers import get_linear_schedule_with_warmup  # 有的版本在顶层
    except Exception:
        from transformers.optimization import get_linear_schedule_with_warmup  # 有的版本在 optimization
    BERT_AVAILABLE = True
    logger.info("BERT 相关依赖加载成功：torch=%s, transformers 已可用", torch.__version__)
except Exception as e:
    BERT_AVAILABLE = False
    logger.exception("BERT 相关依赖导入失败（不一定是未安装）。真实异常如下：%r", e)

class SentimentDataset(Dataset):
    """
    BERT 情感分析数据集

    用于将文本数据转换为 BERT 模型可接受的格式
    """

    def __init__(self, texts: List[str], labels: List[int],
                 tokenizer, max_length: int = 128):
        """
        初始化数据集

        参数:
            texts: 文本列表
            labels: 标签列表（整数编码）
            tokenizer: BERT tokenizer
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # 使用 tokenizer 编码文本
        encoding = self.tokenizer(
            text,  # 需要编码的文本
            add_special_tokens=True,  # 自动加 [CLS] [SEP]
            max_length=self.max_length,  # 最大长度，超过就截断
            padding="max_length",  # 不够就补到 max_length
            truncation=True,  # 允许截断
            return_attention_mask=True,  # 返回 attention_mask（告诉模型哪些是 padding）
            return_tensors="pt",  # 直接返回 PyTorch 张量（torch.Tensor）
        )
        return {
            # encoding['input_ids'] shape 通常是 [1, max_length]，所以 squeeze(0) 去掉 batch 维度
            "input_ids": encoding["input_ids"].squeeze(0),
            # 同理 squeeze(0)
            "attention_mask": encoding["attention_mask"].squeeze(0),
            # label 转成 torch tensor，供 loss 计算使用
            "label": torch.tensor(label, dtype=torch.long),
        }


class SentimentAnalyzer:
    """
    情感分析器（基于 cntext）
    
    设计思路：
        采用 cntext 多维度分析策略：
        1. 基础情感分析：使用 cntext 内置词典（大连理工、知网等）
        2. 情绪维度分析：识别具体情绪类型（喜怒哀惧等）
        3. 心理特征分析：测量态度、认知、价值观等抽象构念
        4. 语义分析：主题提取、关键词、语义相似度
        5. 可选的自定义模型：支持用户训练的机器学习模型
    """
    
    # ==================== 类常量 ====================
    # 基础情感类别
    SENTIMENT_POSITIVE = "Positive"
    SENTIMENT_NEGATIVE = "Negative"
    SENTIMENT_NEUTRAL = "Neutral"
    
    # 具体情绪类别（基于大连理工情感词典）
    EMOTION_JOY = "Joy"              # 喜悦 (乐)
    EMOTION_ANGER = "Anger"          # 愤怒 (怒)
    EMOTION_SADNESS = "Sadness"      # 悲伤 (哀)
    EMOTION_FEAR = "Fear"            # 恐惧 (惧)
    EMOTION_DISGUST = "Disgust"      # 厌恶 (恶)
    EMOTION_SURPRISE = "Surprise"    # 惊讶 (惊)
    EMOTION_GOOD = "Good"            # 好评
    
    # 心理维度
    PSYCH_ATTITUDE = "Attitude"
    PSYCH_COGNITION = "Cognition"
    PSYCH_EMOTION = "Emotion"
    
    # ==================== 初始化方法 ====================
    
    def __init__(self,
                 diction: str = 'zh_common_DUTIR.yaml',
                 custom_dict_path: Optional[str] = None,
                 model_path: Optional[str] = None,
                 stopwords_path: Optional[str] = './rules/hit_stopwords.txt',
                 bert_model_name: str = 'bert-base-chinese',
                 use_bert: bool = False):
        """
        初始化情感分析器

        参数:
            diction: 使用的词典

                因此本参数推荐直接传入 cntext 的内置 yaml 文件名，例如：
                - 'zh_common_DUTIR.yaml'
                - 'zh_common_HowNet.yaml'
                - 'zh_common_NTUSD.yaml'
                - 'zh_common_FinanceSenti.yaml'

                也支持你直接传入一个 Python dict（格式参考 cntext 文档中的 diction 示例）。

            custom_dict_path: 自定义词典文件路径（可选）
            model_path: 已训练模型的路径（可选）
            stopwords_path: 停用词文件路径（可选）
            bert_model_name: BERT 模型名称（默认使用中文 BERT）
            use_bert: 是否使用 BERT 模型（默认 False）
        """
        if not CNTEXT_AVAILABLE:
            logger.error("没有安装 cntext 库，建议使用指令 pip install cntext 安装。否则"
                        "将有很多功能无法使用")
            raise ImportError("cntext is required but not installed. Run: pip install cntext")

        self.diction = diction
        self._cntext_dict_cache: Optional[Dict[str, Any]] = None
        self.stopwords_path = stopwords_path

        if isinstance(self.diction, dict):
            loaded = self.diction
        elif isinstance(self.diction, str):
            try:
                if self.diction.lower().endswith((".yaml", ".yml")):
                    if ct is None:
                        raise ImportError("cntext is not available")
                    loaded = ct.read_yaml_dict(self.diction)
                else:
                    logger.warning(
                        "diction 建议使用内置词典的 yaml 文件名（例如 zh_common_DUTIR.yaml）。"
                        f"当前收到: {self.diction!r}"
                    )
                    loaded = None
            except Exception as e:
                logger.exception(f"读取 YAML 词典失败: {self.diction}, error={e}")
                loaded = None
        else:
            loaded = None

        if isinstance(loaded, dict):
            # 优先使用 Dictionary 字段，如果没有则使用整个字典
            if 'Dictionary' in loaded:
                self._cntext_dict_cache = loaded['Dictionary']
            else:
                self._cntext_dict_cache = loaded

            # 调试：打印词典结构
            if self._cntext_dict_cache:
                logger.info(f"词典加载成功，包含的键: {list(self._cntext_dict_cache.keys())[:10]}")
        else:
            self._cntext_dict_cache = None
            logger.warning("词典加载失败，_cntext_dict_cache 为 None")

        if custom_dict_path is not None:
            try:
                self.custom_dict = self._load_custom_dict(custom_dict_path)

                if isinstance(self.custom_dict, dict) and 'Dictionary' in self.custom_dict:
                    self.custom_dict = self.custom_dict.get('Dictionary')

            except OSError as e:
                logger.error(f"自定义词典不存在，请检查路径: {e}")
                self.custom_dict = None
            except UnicodeError as e:
                logger.error(f"自定义词典内容编码损坏: {e}")
                self.custom_dict = None
            except Exception as e:
                logger.exception(f"加载自定义词典时发生异常: {e}")
                self.custom_dict = None
        else:
            self.custom_dict = None

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
                    logger.warning("模型文件中没有 vectorizer，predict_by_model() 将无法使用")

            except OSError as e:
                logger.error(f"模型不存在，请检查路径: {e}")
                self.model = None
                self.vectorizer = None
            except UnicodeError as e:
                logger.error(f"模型内容编码损坏: {e}")
                self.model = None
                self.vectorizer = None
            except Exception as e:
                logger.exception(f"加载模型时发生异常: {e}")
                self.model = None
                self.vectorizer = None
        else:
            self.model = None
            self.vectorizer = None

        # 添加 BERT 相关属性
        self.use_bert = use_bert and BERT_AVAILABLE
        self.bert_model_name = bert_model_name
        self.bert_model = None
        self.bert_tokenizer = None
        self.label_encoder = None  # 用于标签编码
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if BERT_AVAILABLE else None

        if self.use_bert:
            if not BERT_AVAILABLE:
                logger.warning("BERT 不可用，将使用词典方法")
                self.use_bert = False
            else:
                logger.info(f"BERT 模式已启用，使用设备: {self.device}")

        model_status = "已加载" if self.model is not None else "未加载"
        custom_dict_status = "已加载" if self.custom_dict else "未加载"
        bert_status = "已启用" if self.use_bert else "未启用"
        logger.info(f"初始化完成 - 词典: {diction}, 自定义词典: {custom_dict_status}, 模型: {model_status}, BERT: {bert_status}")

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
        pos_words = sentiment_score.get('pos_words', sentiment_score.get('pos_word', []))
        neg_words = sentiment_score.get('neg_words', sentiment_score.get('neg_word', []))

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
            
        提示:
            - 分批处理避免内存溢出
            - 对每条文本调用 predict()
            - 将结果添加为新列：sentiment, polarity, pos_count, neg_count, confidence
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
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import classification_report, accuracy_score, f1_score

        logger.info("=" * 50)
        logger.info("开始 BERT 模型训练")
        logger.info("=" * 50)

        # 准备数据
        texts = train_df[text_column].astype(str).tolist()
        labels = train_df[label_column].tolist()

        # 标签编码
        self.label_encoder = LabelEncoder()
        labels_encoded = self.label_encoder.fit_transform(labels)
        num_labels = len(self.label_encoder.classes_)

        logger.info(f"标签类别: {self.label_encoder.classes_}")
        logger.info(f"类别数量: {num_labels}")

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels_encoded, test_size=test_size, random_state=42, stratify=labels_encoded
        )

        logger.info(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

        # 加载 BERT tokenizer 和模型
        logger.info(f"加载 BERT 模型: {self.bert_model_name}")
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        self.bert_model = BertForSequenceClassification.from_pretrained(
            self.bert_model_name,
            num_labels=num_labels
        )
        self.bert_model.to(self.device)

        # 创建数据集和数据加载器
        train_dataset = SentimentDataset(X_train, y_train, self.bert_tokenizer, max_length)
        test_dataset = SentimentDataset(X_test, y_test, self.bert_tokenizer, max_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # 设置优化器和学习率调度器
        optimizer = AdamW(self.bert_model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # 训练循环
        logger.info("开始训练...")
        training_stats = []

        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            logger.info("-" * 50)

            # 训练阶段
            self.bert_model.train()
            total_train_loss = 0

            try:
                from tqdm import tqdm
                train_iterator = tqdm(train_loader, desc=f"训练 Epoch {epoch + 1}")
            except ImportError:
                train_iterator = train_loader

            for batch in train_iterator:
                # 将数据移到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # 清零梯度
                self.bert_model.zero_grad()

                # 前向传播
                outputs = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_train_loss += loss.item()

                # 反向传播
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.bert_model.parameters(), 1.0)

                # 更新参数
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_loader)
            logger.info(f"平均训练损失: {avg_train_loss:.4f}")

            # 验证阶段
            logger.info("开始验证...")
            self.bert_model.eval()

            predictions = []
            true_labels = []
            total_eval_loss = 0

            with torch.no_grad():
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

        # 最终评估
        logger.info("\n" + "=" * 50)
        logger.info("训练完成！最终评估结果：")
        logger.info("=" * 50)

        # 将预测结果转换回原始标签
        predictions_labels = self.label_encoder.inverse_transform(predictions)
        true_labels_original = self.label_encoder.inverse_transform(true_labels)

        report = classification_report(true_labels_original, predictions_labels)
        logger.info("\n分类报告:\n" + report)

        # 标记使用 BERT
        self.use_bert = True

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
            
        提示:
            使用 df['sentiment'].value_counts()
        """
        # TODO: 检查 sentiment 列是否存在
        
        # TODO: 统计各情感类别的数量
        
        # TODO: 返回统计结果
        pass
    
    def get_emotion_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        统计情绪分布
        
        参数:
            df: 包含情绪列的 DataFrame
            
        返回:
            pd.DataFrame: 各情绪类型的统计
            
        提示:
            - 展开 emotions 列中的字典
            - 统计各情绪的总数
        """
        # TODO: 检查 emotions 列是否存在
        
        # TODO: 展开情绪字典
        
        # TODO: 统计各情绪类型的总数
        
        # TODO: 返回统计结果
        pass
    
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
            
        提示:
            - 使用 pd.Grouper 按时间分组
            - 计算每个时间段的平均极性、标准差、数量
            - 可选：计算移动平均
        """
        # TODO: 检查必要的列是否存在
        
        # TODO: 确保时间列是 datetime 类型
        
        # TODO: 按时间分组统计
        
        # TODO: 计算移动平均（可选）
        
        # TODO: 返回趋势数据
        pass
    
    def generate_sentiment_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        生成综合情感分析报告
        
        参数:
            df: 包含情感分析结果的 DataFrame
            
        返回:
            Dict[str, Any]: 综合报告
            
        提示:
            包含：情感分布、极性统计、情绪分布、置信度统计、总体统计
        """
        # TODO: 初始化报告字典
        
        # TODO: 统计整体情感分布
        
        # TODO: 统计情感极性（均值、标准差、最小值、最大值）
        
        # TODO: 统计情绪分布（如果有）
        
        # TODO: 统计置信度
        
        # TODO: 添加总体统计信息
        
        # TODO: 返回完整报告
        pass

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