# -*- coding: utf-8 -*-
"""
文本相似度分析模块。

职责：
    基于 TF-IDF、编辑距离与可选词向量能力，对文本执行相似度计算、聚类与重复检测。

适用场景：
    - 检测重复或高度相似的浏览记录；
    - 发现内容同质化现象；
    - 为上层评估模块提供多样性与相似度特征。
"""

import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set

import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans

from utils.logger import setup_logger


logger = setup_logger(__name__, "../../logs/similarity.log")


class SimilarityAnalyzer:
    """
    文本相似度分析主入口。

    负责维护分词、TF-IDF 向量化、相似度计算、聚类和模型持久化等状态。
    """

    # ==================== 类常量 ====================
    SIMILARITY_HIGH = "High"      # 高相似度 (>0.8)
    SIMILARITY_MEDIUM = "Medium"  # 中等相似度 (0.5-0.8)
    SIMILARITY_LOW = "Low"        # 低相似度 (<0.5)

    def __init__(
        self,
        stopwords_path: Optional[str] = None,
        use_word_vectors: bool = False,
        word_vector_model: Optional[str] = None,
        max_features: int = 5000
    ):
        """
        初始化相似度分析器

        参数:
            stopwords_path: 停用词文件路径
            use_word_vectors: 是否使用词向量（需安装 gensim）
            word_vector_model: 词向量模型路径（如 Word2Vec）
            max_features: TF-IDF 最大特征数
        """
        default_stopwords_path = Path(__file__).resolve().parent / "rules" / "hit_stopwords.txt"
        self.stopwords_path = str(default_stopwords_path if stopwords_path is None else stopwords_path)
        self.stopwords = self._load_stopwords()

        # TF-IDF 向量化器：统一负责中文分词后的特征抽取。
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            tokenizer=self._tokenize,
            token_pattern=None,
            lowercase=True,
            ngram_range=(1, 4),   # 允许 1~4 gram，兼顾短语与局部上下文
            dtype=np.float32,
            sublinear_tf=True,
            norm="l2"
        )

        self.tfidf_matrix = None
        self.documents: List[str] = []
        self._doc_indices: List[int] = []  # TF-IDF 行号到原始输入索引的映射

        # 词向量相关：仅在显式开启时参与语义相似度计算。
        self.use_word_vectors = use_word_vectors
        self.word_vectors = None

        if use_word_vectors and word_vector_model:
            self._load_word_vectors(word_vector_model)

        logger.info("SimilarityAnalyzer 初始化完成")

    # ==================== 私有方法 ====================

    def _load_stopwords(self) -> Set[str]:
        """
        加载停用词

        返回:
            Set[str]: 停用词集合
        """
        if not self.stopwords_path or not Path(self.stopwords_path).exists():
            logger.warning(f"停用词文件不存在: {self.stopwords_path}")
            return set()

        try:
            with open(self.stopwords_path, "r", encoding="utf-8") as f:
                stopwords = {line.strip() for line in f if line.strip()}
            logger.info(f"加载停用词 {len(stopwords)} 个")
            return stopwords
        except Exception as e:
            logger.error(f"加载停用词失败: {e}")
            return set()

    def _tokenize(self, text: str) -> List[str]:
        """
        分词并去除停用词

        参数:
            text: 待分词文本

        返回:
            List[str]: 分词结果
        """
        if text is None or pd.isna(text):
            return []

        try:
            words = jieba.lcut(str(text))
            # 过滤停用词和纯空白 token，降低噪声特征对相似度的干扰。
            filtered = [w for w in words if w.strip() and w not in self.stopwords]
            return filtered
        except Exception as e:
            logger.error(f"分词失败: {e}")
            return []

    def _load_word_vectors(self, model_path: str) -> None:
        """
        加载词向量模型（Word2Vec/FastText）

        参数:
            model_path: 模型文件路径
        """
        try:
            from gensim.models import KeyedVectors

            logger.info(f"加载词向量模型: {model_path}")
            self.word_vectors = KeyedVectors.load_word2vec_format(
                model_path,
                binary=True
            )
            logger.info("词向量模型加载成功")
        except ImportError:
            logger.error("gensim 未安装，无法使用词向量功能")
            self.use_word_vectors = False
        except Exception as e:
            logger.error(f"加载词向量模型失败: {e}")
            self.use_word_vectors = False

    def _text_to_vector(self, text: str) -> Optional[np.ndarray]:
        """
        将文本转换为平均词向量

        参数:
            text: 输入文本

        返回:
            np.ndarray 或 None
        """
        if not self.word_vectors:
            return None

        words = self._tokenize(text)
        if not words:
            return None

        vectors = []
        for word in words:
            try:
                vectors.append(self.word_vectors[word])
            except KeyError:
                # OOV（未登录词）直接忽略，避免词向量查找失败打断流程。
                continue

        if not vectors:
            return None

        return np.mean(vectors, axis=0)

    def _calculate_edit_distance(self, text1: str, text2: str) -> int:
        """
        计算编辑距离（Levenshtein Distance）
        使用滚动数组优化，空间复杂度 O(min(m,n))

        参数:
            text1: 文本1
            text2: 文本2

        返回:
            int: 编辑距离
        """
        if not text1 or not text2:
            return max(len(text1 or ""), len(text2 or ""))

        # 确保 text1 是较短的字符串，以降低滚动数组实现的空间占用。
        if len(text1) > len(text2):
            text1, text2 = text2, text1

        m, n = len(text1), len(text2)

        # prev[j] 表示上一行 dp[i-1][j]，curr[j] 表示当前行 dp[i][j]。
        prev = list(range(n + 1))  # 初始化第 0 行：[0, 1, 2, ..., n]
        curr = [0] * (n + 1)

        # 状态转移：删除、插入、替换三种操作中取最小代价。
        for i in range(1, m + 1):
            curr[0] = i  # 边界：dp[i][0] = i

            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    # 字符相同，不需要操作
                    curr[j] = prev[j - 1]
                else:
                    curr[j] = min(
                        prev[j] + 1,      # 删除 text1[i-1]
                        curr[j - 1] + 1,  # 插入 text2[j-1]
                        prev[j - 1] + 1   # 替换 text1[i-1] -> text2[j-1]
                    )

            prev, curr = curr, prev

        return prev[n]

    def _similarity_to_category(self, score: float) -> str:
        """将连续相似度分数映射为高/中/低三档类别。"""
        if score >= 0.8:
            return self.SIMILARITY_HIGH
        elif score >= 0.5:
            return self.SIMILARITY_MEDIUM
        else:
            return self.SIMILARITY_LOW

    # ==================== 核心公共方法 ====================

    def fit(self, texts: List[str]) -> "SimilarityAnalyzer":
        """
        训练 TF-IDF 模型（必须先调用）

        参数:
            texts: 原始文本列表（允许包含空值）

        返回:
            self
        """
        if not texts:
            logger.error("输入文本列表为空")
            raise ValueError("texts cannot be empty")

        logger.info(f"开始训练 TF-IDF 模型，文档数: {len(texts)}")

        valid_pairs = [
            (idx, str(t))
            for idx, t in enumerate(texts)
            if t is not None and not pd.isna(t) and str(t).strip()
        ]

        if not valid_pairs:
            logger.error("没有有效的文本")
            raise ValueError("No valid texts found")

        self._doc_indices = [idx for idx, _ in valid_pairs]
        self.documents = [txt for _, txt in valid_pairs]

        # 稀疏 TF-IDF 矩阵，仅对有效文本建模。
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

        logger.info(f"TF-IDF 矩阵形状: {self.tfidf_matrix.shape}")
        return self

    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的余弦相似度（L2 下 linear_kernel 等价余弦）
        """
        if not text1 or not text2 or pd.isna(text1) or pd.isna(text2):
            logger.warning("输入文本为空")
            return 0.0

        try:
            if self.tfidf_matrix is None:
                # 若尚未 fit，则仅用当前两个文本临时构建词表。
                self.fit([str(text1), str(text2)])

            vec1 = self.vectorizer.transform([str(text1)])
            vec2 = self.vectorizer.transform([str(text2)])

            similarity = linear_kernel(vec1, vec2)[0][0]
            return float(similarity)

        except Exception as e:
            logger.error(f"计算余弦相似度失败: {e}")
            return 0.0

    def calculate_semantic_similarity(self, text1: str, text2: str) -> Optional[float]:
        """
        计算语义相似度（基于词向量）

        返回:
            float 或 None（词向量不可用时）
        """
        if not self.use_word_vectors or self.word_vectors is None:
            logger.warning("词向量模型未加载")
            return None

        vec1 = self._text_to_vector(text1)
        vec2 = self._text_to_vector(text2)

        if vec1 is None or vec2 is None:
            return 0.0

        # 防止除以 0，避免零向量导致 NaN。
        denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        if denom == 0:
            return 0.0

        similarity = np.dot(vec1, vec2) / denom
        return float(similarity)

    def calculate_edit_similarity(self, text1: str, text2: str, normalize: bool = True) -> float:
        """
        计算编辑距离相似度
        """
        if text1 is None or text2 is None:
            return 0.0

        text1 = str(text1)
        text2 = str(text2)

        if not text1 and not text2:
            return 1.0 if normalize else 0.0
        if not text1 or not text2:
            return 0.0 if normalize else float(max(len(text1), len(text2)))

        distance = self._calculate_edit_distance(text1, text2)

        if normalize:
            max_len = max(len(text1), len(text2))
            return 1.0 - (distance / max_len) if max_len > 0 else 1.0

        return float(distance)

    def find_similar_texts(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Tuple[int, float]]:
        """
        查找与 query 最相似的文档

        返回:
            [(原始文档索引, 相似度), ...]
        """
        if self.tfidf_matrix is None:
            logger.error("请先调用 fit() 方法训练模型")
            raise ValueError("Model not fitted. Call fit() first.")

        if not query or pd.isna(query):
            logger.warning("查询文本为空")
            return []

        try:
            query_vec = self.vectorizer.transform([str(query)])

            similarities = linear_kernel(query_vec, self.tfidf_matrix).ravel()

            valid_indices = np.where(similarities >= threshold)[0]
            if len(valid_indices) == 0:
                logger.info("没有找到相似度超过阈值的文档")
                return []

            sorted_indices = valid_indices[np.argsort(-similarities[valid_indices])][:top_k]

            # 返回原始索引，避免过滤空文本后出现索引错位。
            results = [(int(self._doc_indices[i]), float(similarities[i])) for i in sorted_indices]

            logger.info(f"找到 {len(results)} 个相似文档")
            return results

        except Exception as e:
            logger.error(f"查找相似文本失败: {e}")
            return []

    def detect_duplicates(self, texts: List[str], threshold: float = 0.9) -> List[Tuple[int, int, float]]:
        """
        检测重复或高度相似文本

        返回:
            [(原始索引1, 原始索引2, 相似度), ...]
        """
        if not texts:
            logger.warning("输入文本列表为空")
            return []

        logger.info(f"开始检测重复文本，阈值: {threshold}")

        try:
            self.fit(texts)
        except ValueError:
            return []

        # 稀疏矩阵乘法：X @ X.T，直接获得两两文档点积。
        # 因为 TF-IDF 已做 L2 归一化，所以点积可直接视为余弦相似度。
        X = self.tfidf_matrix.tocsr()
        sim_coo = (X @ X.T).tocoo()

        duplicates = []
        for i, j, v in zip(sim_coo.row, sim_coo.col, sim_coo.data):
            # 只取上三角，避免重复配对，同时排除文本与自身的比较。
            if i < j and v >= threshold:
                orig_i = self._doc_indices[i]
                orig_j = self._doc_indices[j]
                duplicates.append((int(orig_i), int(orig_j), float(v)))

        # 按相似度降序排列，便于业务优先处理最相似的样本。
        duplicates.sort(key=lambda x: x[2], reverse=True)

        logger.info(f"检测到 {len(duplicates)} 对重复文本")
        return duplicates

    def cluster_texts(
        self,
        texts: List[str],
        n_clusters: Optional[int] = None,
        method: str = "kmeans",
        min_samples: int = 2
    ) -> np.ndarray:
        """
        对文本进行聚类

        参数:
            texts: 文本列表
            n_clusters: 聚类数量（KMeans 必需）
            method: 'kmeans' 或 'dbscan'
            min_samples: DBSCAN 最小样本数

        返回:
            np.ndarray: 聚类标签（顺序对应“有效文本”）
        """
        if not texts:
            logger.error("输入文本列表为空")
            raise ValueError("texts cannot be empty")

        logger.info(f"开始文本聚类，方法: {method}")
        self.fit(texts)

        if method == "kmeans":
            if n_clusters is None:
                n_clusters = max(2, int(np.sqrt(len(self.documents) / 2)))
                logger.info(f"自动确定聚类数: {n_clusters}")

            if len(self.documents) > 2000:
                clusterer = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    batch_size=1024,
                    n_init="auto"
                )
            else:
                clusterer = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init="auto"
                )

            labels = clusterer.fit_predict(self.tfidf_matrix)

        elif method == "dbscan":
            # 使用稀疏矩阵，避免 toarray() 带来的额外内存压力。
            clusterer = DBSCAN(
                eps=0.5,
                min_samples=min_samples,
                metric="cosine"
            )
            labels = clusterer.fit_predict(self.tfidf_matrix)

        else:
            logger.error(f"不支持的聚类方法: {method}")
            raise ValueError(f"Unsupported clustering method: {method}")

        unique_labels = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"聚类完成，共 {unique_labels} 个簇")
        return labels

    def batch_calculate_similarity(
        self,
        df: pd.DataFrame,
        text_column: str = "title",
        reference_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        批量计算相似度

        参数:
            df: 输入 DataFrame
            text_column: 文本列名
            reference_column: 参考列名（如果为 None，则计算相邻两条）

        返回:
            pd.DataFrame
        """
        if text_column not in df.columns:
            logger.error(f"列 '{text_column}' 不存在")
            raise ValueError(f"Column '{text_column}' not found")

        if df.empty:
            logger.warning("输入 DataFrame 为空")
            return df

        logger.info(f"开始批量计算相似度，数据量: {len(df)}")
        result_df = df.copy()

        if reference_column:
            if reference_column not in df.columns:
                logger.error(f"参考列 '{reference_column}' 不存在")
                raise ValueError(f"Reference column '{reference_column}' not found")

            # 统一处理缺失值，避免 transform 阶段报错。
            texts = result_df[text_column].fillna("").astype(str).tolist()
            refs = result_df[reference_column].fillna("").astype(str).tolist()

            self.fit(texts + refs)

            A = self.vectorizer.transform(texts)
            B = self.vectorizer.transform(refs)

            # 行对行点积（L2 归一化下即余弦相似度），向量化方式快于逐行循环。
            sims = np.asarray(A.multiply(B).sum(axis=1)).ravel()
            result_df["similarity"] = sims.astype(float)

        else:
            texts = result_df[text_column].fillna("").astype(str).tolist()

            # 若全为空文本，直接返回全 0，避免无意义建模。
            if all(not t.strip() for t in texts):
                result_df["similarity_to_previous"] = [0.0] * len(result_df)
                return result_df

            # 训练词表，确保后续相邻比较使用同一特征空间。
            self.fit(texts)

            # 一次性向量化后，相邻行做稀疏点积，避免重复 transform。
            vecs = self.vectorizer.transform(texts)

            similarities = [0.0]  # 第一条记录没有“上一条”可供比较
            for i in range(1, len(texts)):
                sim = float(vecs[i - 1].multiply(vecs[i]).sum())
                similarities.append(sim)

            result_df["similarity_to_previous"] = similarities

        logger.info("批量相似度计算完成")
        return result_df

    # ==================== 模型持久化 ====================

    def save_model(self, path: str) -> None:
        """保存当前向量化器、语料状态与索引映射。"""
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            model_data = {
                "vectorizer": self.vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
                "documents": self.documents,
                "stopwords": self.stopwords,
                "_doc_indices": self._doc_indices
            }

            with open(path, "wb") as f:
                pickle.dump(model_data, f)

            logger.info(f"✅ 模型已保存到: {path}")

        except Exception as e:
            logger.error(f"保存模型失败: {e}")

    def load_model(self, path: str) -> bool:
        """加载已保存的相似度分析模型及其内部状态。"""
        if not Path(path).exists():
            logger.error(f"模型文件不存在: {path}")
            return False

        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

            self.vectorizer = model_data["vectorizer"]
            self.tfidf_matrix = model_data["tfidf_matrix"]
            self.documents = model_data["documents"]
            self.stopwords = model_data.get("stopwords", set())
            # 兼容旧模型：若缺少 _doc_indices，则按当前 documents 顺序回填。
            self._doc_indices = model_data.get("_doc_indices", list(range(len(self.documents))))

            logger.info(f"✅ 模型已加载: {path}")
            return True

        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False

    # ==================== 统计分析 ====================

    def get_similarity_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        统计相似度分布

        参数:
            df: 包含相似度列的 DataFrame

        返回:
            Dict[str, Any]
        """
        similarity_cols = [col for col in df.columns if "similarity" in col.lower()]

        if not similarity_cols:
            logger.error("DataFrame 中没有相似度列")
            return {}

        stats: Dict[str, Any] = {}

        for col in similarity_cols:
            # 强制转成数值，非法值记为 NaN，避免 describe 统计时报错。
            series = pd.to_numeric(df[col], errors="coerce")
            col_stats = series.describe().to_dict()

            stats[col] = {
                "mean": round(float(col_stats.get("mean", 0) or 0), 4),
                "std": round(float(col_stats.get("std", 0) or 0), 4),
                "min": round(float(col_stats.get("min", 0) or 0), 4),
                "max": round(float(col_stats.get("max", 0) or 0), 4),
                "median": round(float(series.median() if not series.empty else 0), 4)
            }

        logger.info("相似度统计完成")
        return stats


# ==================== 测试代码 ====================
if __name__ == "__main__":
    analyzer = SimilarityAnalyzer()

    df = pd.read_csv("utils/output/result.csv")

    texts = df['title'].values.tolist()
    # print(texts)
    # 训练模型
    analyzer.fit(texts)

    # 两文本相似度
    sim = analyzer.calculate_cosine_similarity("今天天气很好", "天气不错适合散步")
    print(f"相似度: {sim:.4f}")

    # 查找相似文本（返回原始索引）
    results = analyzer.find_similar_texts("天气很好", top_k=3, threshold=0.1)
    print(f"相似文本: {results}")

    # 检测重复
    duplicates = analyzer.detect_duplicates(texts, threshold=0.5)
    print(f"重复文本对: {duplicates}")

    # 聚类
    labels = analyzer.cluster_texts(texts, n_clusters=2)
    print(f"聚类标签(有效文本): {labels}")

    # 批量相似度（reference_column）
    df = pd.DataFrame({
        "title": ["天气很好", "Python教程", "机器学习入门"],
        "ref": ["今天天气不错", "Python 编程", "深度学习基础"]
    })
    out_df = analyzer.batch_calculate_similarity(df, text_column="title", reference_column="ref")
    print(out_df)

    logger.info("测试完成")
