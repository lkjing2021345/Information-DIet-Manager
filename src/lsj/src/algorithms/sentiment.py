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

# cntext 相关导入
try:
    import cntext as ct
    CNTEXT_AVAILABLE = True
    logger.info(f'cntext 版本: {ct.__version__}')
except ImportError:
    CNTEXT_AVAILABLE = False
    logger.warning('cntext 未安装，请运行: pip install cntext')

# BERT 相关导入
try:
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    from transformers import AdamW, get_linear_schedule_with_warmup
    from torch.utils.data import Dataset, DataLoader
    BERT_AVAILABLE = True
    logger.info('PyTorch 和 Transformers 可用')
except ImportError:
    BERT_AVAILABLE = False
    logger.warning('BERT 功能不可用，请安装: pip install torch transformers')

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
                 stopwords_path: Optional[str] = './rules/hit_stopwords.txt'):
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
            self._cntext_dict_cache = loaded.get('Dictionary', loaded)
        else:
            self._cntext_dict_cache = None

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

        model_status = "已加载" if self.model is not None else "未加载"
        custom_dict_status = "已加载" if self.custom_dict else "未加载"
        logger.info(f"初始化完成 - 词典: {diction}, 自定义词典: {custom_dict_status}, 模型: {model_status}")
    
    # ==================== 静态方法 ====================
    
    @staticmethod
    def get_available_dicts() -> List[str]:
        """
        获取 cntext 可用的内置词典列表
        
        返回:
            List[str]: 可用词典名称列表
        """
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
            if self.custom_dict:
                raw = ct.sentiment(str(text), diction=self.custom_dict)
            elif self._cntext_dict_cache is not None:
                raw = ct.sentiment(str(text), diction=self._cntext_dict_cache)
            else:
                logger.warning(
                    "未能加载任何可用词典（custom_dict 和 _cntext_dict_cache 都为空），返回空结果。"
                )
                return {'pos': 0, 'neg': 0, 'pos_word': [], 'neg_word': [], 'categories': {}, 'raw': {}}

            # ---- 字段归一化：兼容不同版本/不同词典的返回格式 ----
            pos = raw.get('pos', raw.get('pos_num', 0))
            neg = raw.get('neg', raw.get('neg_num', 0))

            pos_words = raw.get('pos_word', raw.get('pos_words', []))
            neg_words = raw.get('neg_word', raw.get('neg_words', []))

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
                           threshold: float = 0.3) -> str:
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
        基于自定义机器学习模型进行情感分析

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

        if text is None or pd.isna(text) or str(text).strip() == "":
            logger.warning("文本为空")
            return None
        
        try:
            words = self._segment_text(text)
            text_processed = ' '.join(words)

            X = self.vectorizer.transform([text_processed])

            prediction = self.model.predict(X)[0]

            logger.debug(f"模型预测结果: {prediction}")
            return str(prediction)

        except Exception as e:
            logger.exception(f"模型预测失败: {e}")
            return None
    
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
        # TODO: 检查文本列是否存在
        
        # TODO: 初始化结果列表
        
        # TODO: 分批处理数据
        
        # TODO: 对每条文本调用 predict()
        
        # TODO: 记录处理进度
        
        # TODO: 将结果添加到 DataFrame
        
        # TODO: 如果包含情绪分析，添加 emotions 列
        
        # TODO: 返回处理后的 DataFrame
        pass
    
    # ==================== 模型训练方法 ====================
    
    def train_model(self, train_df: pd.DataFrame,
                   text_column: str = 'text',
                   label_column: str = 'sentiment',
                   test_size: float = 0.2):
        """
        训练自定义情感分析模型（补充 cntext）

        参数:
            train_df: 训练数据
            text_column: 文本列名
            label_column: 标签列名
            test_size: 测试集比例
            
        提示:
            1. 使用 _segment_text() 对文本分词
            2. 使用 TfidfVectorizer 提取特征
            3. 使用 train_test_split 划分数据集
            4. 使用 MultinomialNB 训练模型
            5. 使用 classification_report 评估模型
        """
        # TODO: 导入必要的 sklearn 模块
        
        # TODO: 数据预处理（分词）
        
        # TODO: 划分训练集和测试集
        
        # TODO: 特征提取（TF-IDF）
        
        # TODO: 模型训练（朴素贝叶斯）
        
        # TODO: 模型评估
        
        # TODO: 记录训练结果
        
        # TODO: 返回准确率
        pass
    
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
        保存训练好的自定义模型

        参数:
            path: 模型保存路径
            
        提示:
            - 使用 pickle 保存模型和向量化器
            - 同时保存词典名称
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
        加载自定义模型

        参数:
            path: 模型文件路径

        返回:
            bool: 是否加载成功
            
        提示:
            使用 pickle 加载模型和向量化器
        """
        # TODO: 使用 pickle 加载模型数据
        
        # TODO: 恢复模型、向量化器和词典名称
        
        # TODO: 记录加载日志
        
        # TODO: 返回加载结果
        pass
    
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


# ==================== 测试代码 ====================
def test_basic_sentiment():
    """测试基础情感分析"""
    # TODO: 初始化分析器
    
    # TODO: 准备测试文本
    
    # TODO: 对每条文本进行分析
    
    # TODO: 打印结果
    pass


def test_batch_prediction():
    """测试批量预测"""
    # TODO: 初始化分析器
    
    # TODO: 创建测试 DataFrame
    
    # TODO: 批量分析
    
    # TODO: 打印结果和统计
    pass


def test_advanced_features():
    """测试高级功能"""
    # TODO: 初始化分析器
    
    # TODO: 测试关键词提取
    
    # TODO: 测试可读性分析
    
    # TODO: 测试语义相似度
    pass


def test_cntext_dicts():
    """测试 cntext 内置词典"""
    # TODO: 显示可用词典
    
    # TODO: 测试不同词典的效果
    pass


def test_custom_dict():
    """测试自定义词典"""
    # TODO: 创建自定义词典
    
    # TODO: 使用自定义词典进行分析
    pass


if __name__ == "__main__":
    # TODO: 检查 cntext 是否可用
    
    # TODO: 显示 cntext 版本
    
    # TODO: 运行所有测试
    pass