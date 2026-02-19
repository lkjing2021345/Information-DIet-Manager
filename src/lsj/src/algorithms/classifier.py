"""
文本分类器模块

功能概述：
    对浏览记录的标题/URL进行自动分类，判断用户访问的是新闻、娱乐、学习等哪类内容

主要技术：
    - jieba: 中文分词
    - sklearn: TF-IDF 特征提取 + 朴素贝叶斯分类器

学习要点：
    - 类的封装设计
    - 规则匹配 vs 机器学习方法的选择
    - 文本预处理流程
"""

import jieba
import pandas as pd
import os
import pickle
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any

from pandas.core.interchange.dataframe_protocol import DataFrame
# removed import
# removed import

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# logger 基本设置
logs_folder_path = "../../logs"
if not os.path.exists(logs_folder_path):
    os.makedirs(logs_folder_path)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../../logs/classifier.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ContentClassifier:
    """
    浏览记录内容分类器

    设计思路：
        采用"规则优先，模型兜底"的混合策略：
        1. 先用关键词规则快速匹配（准确率高、速度快）
        2. 规则未命中时，使用训练好的机器学习模型（泛化能力强）
        3. 都失败则归类为 Other

    属性说明：
        categories: 支持的分类类别列表
        rules: 关键词规则字典 {类别: [关键词列表]}
        model: 机器学习模型（朴素贝叶斯）
        vectorizer: TF-IDF 向量化器
    """

    # ==================== 类常量 ====================
    # 使用类常量定义默认类别，方便统一管理
    CATEGORY_NEWS = "News"           # 新闻
    CATEGORY_ENTERTAINMENT = "Entertainment"  # 娱乐
    CATEGORY_LEARNING = "Learning"   # 学习
    CATEGORY_SOCIAL = "Social"        # 社交
    CATEGORY_SHOPPING = "Shopping"    # 购物
    CATEGORY_TOOLS = "Tools"          # 工具
    CATEGORY_OTHER = "Other"          # 其他

    # ==================== 初始化方法 ====================

    def __init__(self,
                 keyword_dict: Optional[Dict[str, List[str]]] = None,
                 model_path: Optional[str] = None):
        """
        初始化分类器

        参数:
            keyword_dict: 自定义关键词字典，格式为 {'类别': ['词1', '词2']}
                          如果为 None，则使用默认规则库
            model_path: 已训练模型的路径，如果提供则自动加载

        学习要点:
            - Optional 类型提示表示参数可以为 None
            - __init__ 方法不应包含耗时操作
        """
        # 初始化分类类别列表
        self.categories = []

        # 加载关键词规则库
        self.rules = keyword_dict if keyword_dict is not None else self._load_default_rules()

        self.categories = list(self.rules.keys())

        # TODO: 初始化机器学习相关属性（初始为 None）
        self.model = None       # 朴素贝叶斯模型
        self.vectorizer = None  # TF-IDF 向量化器

        # 如果提供了模型路径，尝试加载模型
        if model_path:
            self.load_model(model_path)

        logger.info("✅ ContentClassifier 初始化完成")

    # ==================== 私有方法（内部使用）====================

    def _load_default_rules(self) -> Dict[str, List[str]]:
        """
        加载默认的关键词规则库

        返回:
            Dict[str, List[str]]: 关键词规则字典
        """
        current_dir = Path(__file__).parent
        json_dir = current_dir.joinpath("rules")
        config_path = json_dir.joinpath("default_classify_rules.json")

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
                if key in raw_data:
                    result[category_const] = raw_data[key]
                else:
                    result[category_const] = []

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

    def _segment_text(self, text: str) -> List[str]:
        """
        对文本进行分词

        参数:
            text: 待分词的文本字符串

        返回:
            List[str]: 分词后的词语列表
        """
        if text is None:
            logger.error("输入的文本为 None")
            return []

        try:
            words = jieba.lcut(text)
            return words

        except Exception as e:
            logger.error(f"分词失败: {e}")
            return []


    def _remove_stopwords(self, words: List[str]) -> List[str]:
        """
        移除停用词

        参数:
            words: 分词后的词语列表

        返回:
            List[str]: 移除停用词后的词语列表
        """
        def load_stopwords(path):
            with open(path, "r", encoding="utf-8") as f:
                stopwords = {line.strip() for line in f if line.strip()}

            return stopwords

        rules_path = Path(__file__).parent.joinpath("rules")
        stopwords_path = rules_path.joinpath("hit_stopwords.txt")
        stopwords_set = load_stopwords(stopwords_path)

        filtered_words = [word for word in words if word not in stopwords_set and len(word.strip()) > 0]

        return filtered_words

    def _extract_domain(self, url: str) -> str:
        """
        从 URL 中提取域名

        参数:
            url: 完整的 URL 字符串

        返回:
            str: 域名部分，如 "baidu.com"
        """
        from urllib.parse import urlparse

        if not url:
            logger.error("输入的 url 有误")
            return ""

        try:
            parsed = urlparse(url)
            domain = parsed.netloc

            if domain.startswith("www."):
                domain = domain[4:]

            return domain

        except Exception as e:
            logger.error(f"域名提取失败: {url}, 错误: {e}")
            return ""

    def _predict_by_model(self, text: str) -> str:
        """
        使用机器学习模型进行预测

        参数:
            text: 待预测的文本

        返回:
            str: 预测的类别
        """
        if not self.model or not self.vectorizer:
            logger.warning("模型未训练，无法预测")
            return self.CATEGORY_OTHER

        try:
            words = self._segment_text(text)
            clean_words = self._remove_stopwords(words)
            processed_text = "".join(clean_words)

            text_vec = self.vectorizer.transform([processed_text])

            prediction = self.model.predict(text_vec)[0]
            return prediction

        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            return self.CATEGORY_OTHER

    # ==================== 核心公共方法 ====================

    def predict_by_rules(self, text: str, url: Optional[str] = None) -> Optional[str]:
        """
        基于关键词规则进行分类

        参数:
            text: 网页标题
            url: 网页 URL（可选，辅助判断）

        返回:
            Optional[str]: 匹配到的类别，未匹配返回 None
        """
        text_lower = str(text).lower() if text else ""
        url_lower = str(url).lower() if url else ""

        # 1. URL 匹配
        if url_lower:
            for category, keywords in self.rules.items():
                for keyword in keywords:
                    if keyword.lower() in url_lower:
                        logger.info(f"URL匹配成功: '{keyword}' -> {category}")
                        return category

        # 2. 标题文本匹配
        if text_lower:
            for category, keywords in self.rules.items():
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        logger.info(f"标题匹配成功: '{keyword}' -> {category}")
                        return category

        # 3. 匹配失败
        logger.debug("规则匹配失败，返回 None")
        return None

    def train_model(self,
                    texts: List[str],
                    labels: List[str],
                    test_size: float = 0.2,
                    fixed_test_set: Optional[Tuple[List[str], List[str]]] = None,
                    random_state: int = 42,
                    remove_duplicates: bool = True) -> Dict[str, Union[float, int, str]]:
        """
        训练朴素贝叶斯分类器

        参数:
            texts: 训练文本列表
            labels: 对应的标签列表
            test_size: 测试集比例，默认 0.2
            fixed_test_set: 固定测试集 (texts, labels)，用于稳定评估
            random_state: 随机种子

        返回:
            Dict[str, float]: 包含准确率等评估指标的字典
        """
        logger.info("正在训练模型")

        # 数据质量检查
        logger.info(f"原始数据: {len(texts)} 条")

        # 检查标签分布
        from collections import Counter
        label_dist = Counter(labels)
        logger.info(f"标签分布: {dict(label_dist)}")

        # 去重处理
        if remove_duplicates:
            unique_pairs = list(set(zip(texts, labels)))
            if len(unique_pairs) < len(texts):
                dup_count = len(texts) - len(unique_pairs)
                dup_rate = dup_count / len(texts)
                logger.warning(f"发现重复数据: {dup_count} 条 ({dup_rate:.2%})")
                texts, labels = zip(*unique_pairs)
                texts, labels = list(texts), list(labels)
                logger.info(f"去重后数据量: {len(texts)} 条")
        else:
            logger.info("跳过去重处理")

        # 数据预处理
        processed_texts = []
        valid_indices = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                logger.warning(f"跳过空文本: index {i}")
                continue

            words = self._segment_text(text)
            clean_words = self._remove_stopwords(words)

            if not clean_words:
                logger.warning(f"预处理后为空: '{text[:50]}...'")
                continue

            # 用空格连接，保持词边界
            processed_texts.append(" ".join(clean_words))
            valid_indices.append(i)

        # 过滤对应的标签
        valid_labels = [labels[i] for i in valid_indices]

        logger.info(f"有效数据: {len(processed_texts)} 条")

        if len(processed_texts) < 10:
            logger.error("有效数据太少，无法训练")
            return {"accuracy": 0.0, "error": "insufficient_data"}

        # 使用固定测试集或划分新测试集
        if fixed_test_set:
            X_train, y_train = processed_texts, valid_labels
            X_test_raw, y_test = fixed_test_set

            # 预处理固定测试集
            X_test = []
            for text in X_test_raw:
                words = self._segment_text(text)
                clean_words = self._remove_stopwords(words)
                X_test.append(" ".join(clean_words))

            logger.info(f"使用固定测试集: {len(X_test)} 条")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                processed_texts, valid_labels,
                test_size=test_size,
                random_state=random_state,
                stratify=valid_labels  # 保持标签分布
            )
            logger.info(f"划分数据集 - 训练: {len(X_train)}, 测试: {len(X_test)}")

        # 动态调整 TF-IDF 参数
        vocab_size = min(len(processed_texts) * 5, 20000)
        vocab_size = max(vocab_size, 1000)

        self.vectorizer = TfidfVectorizer(
            max_features=vocab_size,
            min_df=2,  # 至少出现2次
            max_df=0.95,  # 最多95%文档包含
            ngram_range=(1, 2)  # 1-2gram
        )

        try:
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)

            logger.info(f"特征维度: {X_train_vec.shape[1]}")

            # 训练模型
            self.model = MultinomialNB(alpha=0.1)  # 平滑参数
            self.model.fit(X_train_vec, y_train)

            # 评估
            train_accuracy = self.model.score(X_train_vec, y_train)
            test_accuracy = self.model.score(X_test_vec, y_test)

            logger.info(f"训练准确率: {train_accuracy:.4f}")
            logger.info(f"测试准确率: {test_accuracy:.4f}")

            # 详细分类报告
            y_pred = self.model.predict(X_test_vec)
            report = classification_report(y_test, y_pred, output_dict=True)

            # 打印每类指标
            logger.info("\n各类别性能:")
            for label in sorted(set(y_test)):
                if label in report:
                    metrics = report[label]
                    logger.info(f"  {label:12}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1-score']:.3f} ({metrics['support']} samples)")

            return {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "macro_f1": report['macro avg']['f1-score'],
                "weighted_f1": report['weighted avg']['f1-score'],
                "feature_count": X_train_vec.shape[1],
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            }

        except Exception as e:
            logger.error(f"训练失败: {e}")
            return {"accuracy": 0.0, "error": str(e)}

    def predict(self, text: str, url: Optional[str] = None) -> str:
        """
        预测单条文本的类别（主入口方法）

        参数:
            text: 网页标题
            url: 网页 URL（可选）

        返回:
            str: 预测的类别
        """
        result = self.predict_by_rules(text=text, url=url)
        if result:
            logger.info(f"规则匹配成功: {result}")
            return result

        if self.model is not None:
            model_result = self._predict_by_model(text=text)
            if model_result:
                logger.info(f"模型预测成功: {model_result}")
                return model_result

        return self.CATEGORY_OTHER


    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        批量预测 DataFrame 中的数据

        参数:
            df: 包含 'title' 和 'url' 列的 DataFrame

        返回:
            pd.DataFrame: 添加了 'category' 列的 DataFrame
        """
        # TODO: 多线程优化
        # TODO: 向量化操作

        if df.empty:
            logger.warning("输入数据为空")
            return df

        if df is None:
            logger.warning("输出数据为 None")
            return DataFrame

        logger.info(f"正在处理 {len(df)} 条数据...")

        result_df = df.copy()
        result_df['category'] = result_df.apply(
            lambda row: self.predict(
                text=row.get('title', ''),
                url=row.get('url', '')
            ),
            axis=1
        )

        logger.info("处理完成")

        return result_df

    # ==================== 模型持久化方法 ====================

    def save_model(self, path: str) -> None:
        """
        保存训练好的模型到文件

        参数:
            path: 模型保存路径

        说明:
            模型保存后，下次启动可以直接加载，无需重新训练
        """
        if not self.model or not self.vectorizer:
            logger.error("模型未训练，无法保存")
            return

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # 保存模型、向量化器和类别列表
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'categories': self.categories
            }

            with open(path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"✅ 模型已保存到: {path}")

        except Exception as e:
            logger.error(f"模型保存失败: {e}")

    def load_model(self, path: str) -> bool:
        """
        从文件加载模型

        参数:
            path: 模型文件路径

        返回:
            bool: 加载是否成功

        注意:
            加载前检查文件是否存在
        """
        if not os.path.exists(path):
            logger.error(f"模型文件不存在: {path}")
            return False

        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.categories = model_data['categories']

            logger.info(f"✅ 模型已加载: {path}")
            return True

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def get_category_distribution(self, df: pd.DataFrame) -> pd.Series:
        """
        统计分类结果的分布情况

        参数:
            df: 包含 'category' 列的 DataFrame

        返回:
            pd.Series: 各类别的数量统计
        """
        if df.empty:
            logger.error("数据为空")
            return pd.Series()

        if 'category' not in df.columns:
            logger.error("数据缺少 \'category\' 列")
            return pd.Series()

        distribution = df['category'].value_counts()

        return distribution

# ==================== 测试代码 ====================
if __name__ == "__main__":
    """
    单元测试：直接运行此文件来测试分类器功能
    
    测试步骤:
        1. 实例化分类器
        2. 测试单条文本预测
        3. 测试批量预测
        4. (可选) 测试模型训练和保存
    """

    print("=" * 50)
    print("ContentClassifier 单元测试")
    print("=" * 50)

    # 1. 实例化分类器
    classifier = ContentClassifier()

    # 2. 测试分词功能
    print("\n--- 测试分词 ---")
    words = classifier._segment_text("我爱用Python写代码")
    print(f"分词结果: {words}")
    # 预期输出: ['我', '爱', '用', 'Python', '写', '代码']

    # 3. 测试规则预测功能
    print("\n--- 测试规则预测 ---")

    # 测试1: URL 包含 "jd" (购物)
    res1 = classifier.predict_by_rules("首页", "https://www.jd.com")
    print(f"京东测试: {res1}")  # 预期输出: Shopping

    # 测试2: 标题包含 "Python" (学习)
    res2 = classifier.predict_by_rules("Python基础教程", "https://www.baidu.com")
    print(f"Python测试: {res2}")  # 预期输出: Learning

    # 测试3: 都不匹配
    res3 = classifier.predict_by_rules("今天天气真好", "https://www.unknown.com")
    print(f"未知测试: {res3}")  # 预期输出: None

    print("\n--- 测试 predict 主入口 ---")
    res = classifier.predict("Python入门教程", None)
    print(f"预测结果: {res}")  # 预期: learning

    res = classifier.predict("未知标题", "https://unknown.com")
    print(f"预测结果: {res}")  # 预期: other

    # 测试批量预测
    print("\n--- 测试 batch_predict ---")
    test_data = pd.DataFrame([
        {"title": "GitHub - 开源项目", "url": "https://github.com"},
        {"title": "京东购物", "url": "https://www.jd.com"},
        {"title": "今日新闻", "url": "https://news.baidu.com"},
    ])
    result_df = classifier.batch_predict(test_data)
    # print(result_df.columns.tolist())
    print(result_df[['title', 'category']])

    print("\n--- 测试域名提取 ---")
    url1 = "https://www.bilibili.com/video/BV1xx"
    domain1 = classifier._extract_domain(url1)
    print(f"提取结果: {domain1}")  # 预期: bilibili.com

    # 5. 测试分类统计
    print("\n--- 测试分类统计 ---")
    # 假设我们有一些已分类的数据
    test_df = pd.DataFrame({
        'title': ['Python教程', '淘宝购物', '微博热搜', '京东下单'],
        'category': ['Learning', 'Shopping', 'Social', 'Shopping']
    })
    dist = classifier.get_category_distribution(test_df)
    print("分类统计结果:")
    print(dist)
    # 预期输出:
    # Shopping    2
    # Learning    1
    # Social      1