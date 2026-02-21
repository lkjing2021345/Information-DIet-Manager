"""
文本分类器模块

功能概述：
    对浏览记录的标题/URL进行自动分类，判断用户访问的是新闻、娱乐、学习等哪类内容
"""
import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import jieba
import numpy as np
import pandas as pd

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

        self.model = None       # 集成模型
        self.vectorizer = None  # TF-IDF 向量化器（词级）
        self.char_vectorizer = None  # 字符级向量化器
        self.high_freq_vectorizer = None  # 高频词向量化器

        if model_path:
            self.load_model(model_path)

        logger.info("ContentClassifier 初始化完成")

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

    def _extract_features(self, text: str) -> np.ndarray:
        """
        提取多种特征（必须与训练时保持一致）

        参数:
            text: 待处理的文本

        返回:
            特征向量
        """
        from scipy.sparse import hstack

        # 预处理文本
        words = self._segment_text(text)
        clean_words = self._remove_stopwords(words)
        word_text = " ".join(clean_words)

        # 提取所有特征
        feature_list = []

        # 1. 词级特征（1-5gram）
        if self.vectorizer:
            word_vec = self.vectorizer.transform([word_text])
            feature_list.append(word_vec)

        # 2. 字符级特征（2-6gram）
        if self.char_vectorizer:
            char_vec = self.char_vectorizer.transform([text])
            feature_list.append(char_vec)

        # 3. 高频词特征（1-2gram）
        if self.high_freq_vectorizer:
            high_freq_vec = self.high_freq_vectorizer.transform([word_text])
            feature_list.append(high_freq_vec)

        # 合并所有特征
        if len(feature_list) > 1:
            combined = hstack(feature_list)
            return combined
        elif len(feature_list) == 1:
            return feature_list[0]
        else:
            raise ValueError("没有可用的向量化器")

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
            features = self._extract_features(text)
            prediction = self.model.predict(features)[0]
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

    def batch_predict(self, df: pd.DataFrame,
                      use_parallel: bool = True,
                      n_workers: int = None) -> pd.DataFrame:
        """
        批量预测 DataFrame 中的数据
        优化策略：
        1. 向量化规则匹配（快速处理大部分数据）
        2. 多进程模型预测（处理规则未匹配的数据）

        参数:
            df: 包含 'title' 和 'url' 列的 DataFrame
            use_parallel: 是否使用多进程（默认True）
            n_workers: 进程数（默认为CPU核心数）

        返回:
            pd.DataFrame: 添加了 'category' 列的 DataFrame
        """
        import time

        if df.empty:
            logger.warning("输入数据为空")
            return df

        start_time = time.time()
        logger.info(f"开始处理 {len(df)} 条数据...")

        logger.info("阶段1: 向量化规则匹配")
        df = self._batch_predict_by_rules_vectorized(df)

        # 统计规则匹配结果
        rule_matched = df['category'].notna().sum()
        rule_ratio = rule_matched / len(df) * 100
        logger.info(f"  规则匹配: {rule_matched}/{len(df)} ({rule_ratio:.1f}%)")

        need_model = df['category'].isna()
        need_model_count = need_model.sum()

        if need_model_count == 0:
            logger.info("所有数据都通过规则匹配完成")
            elapsed = time.time() - start_time
            logger.info(f"处理完成，耗时: {elapsed:.2f}秒")
            return df

        logger.info(f"阶段2: 模型预测 ({need_model_count} 条)")

        if use_parallel and need_model_count > 100:
            df = self._batch_predict_by_model_parallel(df, need_model, n_workers)
        else:
            df = self._batch_predict_by_model_sequential(df, need_model)

        df['category'] = df['category'].fillna(self.CATEGORY_OTHER)

        elapsed = time.time() - start_time
        logger.info(f"处理完成，耗时: {elapsed:.2f}秒")
        logger.info(f"  规则匹配: {rule_matched} 条")
        logger.info(f"  模型预测: {need_model_count} 条")
        logger.info(f"  平均速度: {len(df)/elapsed:.0f} 条/秒")

        return df

    def _batch_predict_by_rules_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        向量化的规则匹配
        """
        import re

        df = df.copy()
        df['category'] = None

        df['title'] = df['title'].fillna('')
        df['url'] = df['url'].fillna('')

        df['title_lower'] = df['title'].str.lower()
        df['url_lower'] = df['url'].str.lower()

        for category, keywords in self.rules.items():
            if not keywords:
                continue

            # 构建正则表达式
            escaped_keywords = [re.escape(kw.lower()) for kw in keywords]
            pattern = '|'.join(escaped_keywords)

            try:
                title_match = df['title_lower'].str.contains(
                    pattern,
                    case=False,
                    na=False,
                    regex=True
                )

                url_match = df['url_lower'].str.contains(
                    pattern,
                    case=False,
                    na=False,
                    regex=True
                )

                matched = title_match | url_match

                need_update = matched & df['category'].isna()

                df.loc[need_update, 'category'] = category

            except Exception as e:
                logger.warning(f"类别 {category} 的规则匹配失败: {e}")
                continue

        df = df.drop(columns=['title_lower', 'url_lower'])

        return df

    def _batch_predict_by_model_sequential(self, df: pd.DataFrame,
                                           need_model: pd.Series) -> pd.DataFrame:
        """
        单进程模型预测
        """
        need_model_df = df[need_model]

        texts = need_model_df['title'].tolist()

        if self.model and self.vectorizer:
            try:
                features_list = []
                for text in texts:
                    words = self._segment_text(text)
                    clean_words = self._remove_stopwords(words)
                    word_text = " ".join(clean_words)
                    features_list.append(word_text)

                # 提取所有特征
                feature_matrices = []

                # 词级特征
                if self.vectorizer:
                    word_vecs = self.vectorizer.transform(features_list)
                    feature_matrices.append(word_vecs)

                # 字符级特征
                if self.char_vectorizer:
                    char_vecs = self.char_vectorizer.transform(texts)
                    feature_matrices.append(char_vecs)

                # 高频词特征
                if self.high_freq_vectorizer:
                    high_freq_vecs = self.high_freq_vectorizer.transform(features_list)
                    feature_matrices.append(high_freq_vecs)

                # 合并特征
                from scipy.sparse import hstack
                if len(feature_matrices) > 1:
                    combined_features = hstack(feature_matrices)
                else:
                    combined_features = feature_matrices[0]

                predictions = self.model.predict(combined_features)

                df.loc[need_model, 'category'] = predictions

            except Exception as e:
                logger.error(f"批量模型预测失败: {e}")
                for idx, text in zip(need_model_df.index, texts):
                    try:
                        category = self._predict_by_model(text)
                        df.loc[idx, 'category'] = category
                    except Exception as e2:
                        logger.error(f"预测失败 (索引 {idx}): {e2}")
                        df.loc[idx, 'category'] = self.CATEGORY_OTHER
        else:
            df.loc[need_model, 'category'] = self.CATEGORY_OTHER

        return df

    def _batch_predict_by_model_parallel(self, df: pd.DataFrame,
                                         need_model: pd.Series,
                                         n_workers: int = None) -> pd.DataFrame:
        """
        多进程模型预测（内部方法）
        """
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing

        if n_workers is None:
            n_workers = min(multiprocessing.cpu_count(), 4)

        need_model_df = df[need_model].copy().reset_index(drop=False)
        original_indices = need_model_df['index'].tolist()

        chunk_size = max(1, len(need_model_df) // n_workers)
        chunks = []
        for i in range(0, len(need_model_df), chunk_size):
            chunk_data = need_model_df.iloc[i:i + chunk_size]
            if len(chunk_data) > 0:
                chunks.append(chunk_data)

        logger.info(f"  使用 {n_workers} 个进程并行处理")

        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(self._predict_chunk_worker, chunk.to_dict('records'), chunk['index'].tolist())
                    for chunk in chunks
                ]

                all_results = []
                all_indices = []
                for i, future in enumerate(futures):
                    try:
                        results, indices = future.result(timeout=300)
                        all_results.extend(results)
                        all_indices.extend(indices)
                        logger.debug(f"  进程 {i+1}/{n_workers} 完成")
                    except Exception as e:
                        logger.error(f"  进程 {i+1} 失败: {e}")
                        # 使用当前块的索引和默认类别
                        all_results.extend([self.CATEGORY_OTHER] * len(chunks[i]))
                        all_indices.extend(chunks[i].index.tolist())

            # 使用索引来正确设置结果
            for idx, category in zip(all_indices, all_results):
                df.loc[idx, 'category'] = category

        except Exception as e:
            logger.error(f"多进程预测失败，降级为单进程: {e}")
            df = self._batch_predict_by_model_sequential(df, need_model)

        return df

    def _predict_chunk_worker(self, chunk_records: List[Dict], indices: List[int]) -> Tuple[List[str], List[int]]:
        """
        工作函数：在子进程中运行

        参数:
            chunk_records: 记录列表（字典格式）
            indices: 对应的索引列表

        返回:
            Tuple[List[str], List[int]]: (预测结果列表, 索引列表)
        """
        results = []

        for record in chunk_records:
            text = record.get('title', '')

            try:
                # 调用模型预测
                category = self._predict_by_model(text)
                results.append(category)
            except Exception:
                # 预测失败，标记为 Other
                results.append(self.CATEGORY_OTHER)

        return results, indices

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

            # 保存模型、所有向量化器和类别列表
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'char_vectorizer': self.char_vectorizer,
                'high_freq_vectorizer': self.high_freq_vectorizer,
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
            self.char_vectorizer = model_data.get('char_vectorizer', None)
            self.high_freq_vectorizer = model_data.get('high_freq_vectorizer', None)
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
    classifier = ContentClassifier(
        model_path="./models/advanced_model.pkl"
    )

    history_df = pd.read_json('../utils/output/history_data.jsonl', lines=True)

    result_df = classifier.batch_predict(history_df)

    result_df.to_csv('../utils/output/result.csv', index=False)