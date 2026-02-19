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
from typing import List, Dict, Optional, Tuple

# ==================== 可选导入 ====================
# Day 4 之后取消注释，用于机器学习分类
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

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

        # TODO: 如果提供了模型路径，尝试加载模型
        # 提示：调用 self.load_model(model_path)

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

        学习要点:
            - jieba.cut() 返回生成器，需要用 list() 转换
            - jieba.lcut() 直接返回列表，更方便
            - 可以加载自定义词典提高分词准确率

        jieba 常用方法:
            - jieba.cut(text): 精确模式分词
            - jieba.lcut(text): 返回列表
            - jieba.add_word(word): 添加自定义词
            - jieba.load_userdict(path): 加载自定义词典文件
        """
        if text is None:
            logger.error("输入的文本为 None")
            return []

        try:
            words = jieba.lcut(text)
            return words

        except Exception as e:
            logger.exception(f"分词失败: {e}")
            return []


    def _remove_stopwords(self, words: List[str]) -> List[str]:
        """
        移除停用词

        参数:
            words: 分词后的词语列表

        返回:
            List[str]: 移除停用词后的词语列表

        说明:
            停用词是指"的"、"是"、"在"等无实际意义的词语
            移除停用词可以提高分类准确率

        TODO:
            1. 定义停用词列表或从文件加载
            2. 过滤掉停用词
        """
        pass

    def _extract_domain(self, url: str) -> str:
        """
        从 URL 中提取域名

        参数:
            url: 完整的 URL 字符串

        返回:
            str: 域名部分，如 "www.baidu.com"

        提示:
            - 可以使用字符串的 split('/') 方法
            - 或使用 urllib.parse.urlparse() 解析

        示例:
            输入: "https://www.bilibili.com/video/xxx"
            输出: "www.bilibili.com" 或 "bilibili.com"
        """
        # TODO: 实现域名提取
        pass

    def _predict_by_model(self, text: str) -> str:
        """
        使用机器学习模型进行预测

        参数:
            text: 待预测的文本

        返回:
            str: 预测的类别

        前置条件:
            self.model 和 self.vectorizer 必须已训练

        实现步骤:
            1. 对文本进行分词和预处理
            2. 使用 vectorizer 转换为 TF-IDF 向量
            3. 使用 model.predict() 预测类别
        """
        # TODO: Day 4 实现机器学习预测
        pass

    # ==================== 核心公共方法 ====================

    def predict_by_rules(self, text: str, url: Optional[str] = None) -> Optional[str]:
        """
        基于关键词规则进行分类

        参数:
            text: 网页标题
            url: 网页 URL（可选，辅助判断）

        返回:
            Optional[str]: 匹配到的类别，未匹配返回 None

        设计思路:
            1. URL 匹配优先（域名更准确）
            2. 标题关键词匹配次之
            3. 只要匹配到就返回，不进行多类别判断

        提示:
            - 字符串的 in 操作符可以判断子串
            - 使用 .lower() 统一转为小写，提高匹配率
        """
        text_lower = str(text).lower() if text else ""
        url_lower = str(url).lower() if url else ""

        if url_lower:
            for category, keywords in self.rules.items():
                for keyword in keywords:
                    if keyword.lower() in keywords:
                        logger.info(f"url类别成功匹配: {category}")
                        return category

        logger.warning(f"url类别匹配失败，继续使用标题关键词匹配")
        if text_lower:
            for category, keywords in self.rules.items():
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        logger.info(f"标题关键字匹配成功: {category}")
                        return category

        logger.warning("规则匹配失败")
        return None

    def train_model(self,
                    texts: List[str],
                    labels: List[str],
                    test_size: float = 0.2) -> Dict[str, float]:
        """
        训练朴素贝叶斯分类器

        参数:
            texts: 训练文本列表
            labels: 对应的标签列表
            test_size: 测试集比例，默认 0.2

        返回:
            Dict[str, float]: 包含准确率等评估指标的字典

        实现步骤:
            1. 文本预处理（分词、去停用词）
            2. 划分训练集和测试集
            3. 创建 TfidfVectorizer 并转换文本
            4. 训练 MultinomialNB 模型
            5. 评估模型性能

        sklearn 关键方法:
            - TfidfVectorizer(): 创建 TF-IDF 向量化器
              - fit_transform(texts): 拟合并转换
              - transform(texts): 仅转换（用于新数据）
            - MultinomialNB(): 朴素贝叶斯分类器
              - fit(X, y): 训练模型
              - predict(X): 预测
              - score(X, y): 计算准确率
            - train_test_split(): 划分数据集

        TODO: Day 4 实现模型训练
        """
        print("🔄 正在训练模型...")
        # TODO: 实现训练逻辑
        pass

    def predict(self, text: str, url: Optional[str] = None) -> str:
        """
        预测单条文本的类别（主入口方法）

        参数:
            text: 网页标题
            url: 网页 URL（可选）

        返回:
            str: 预测的类别

        分类策略:
            1. 优先使用规则匹配（快速、准确）
            2. 规则未命中且有模型时，使用模型预测
            3. 都失败则返回 Other

        这是类最重要的对外接口！
        """
        # TODO: 实现预测逻辑
        # 提示：调用 self.predict_by_rules() 和 self._predict_by_model()
        pass

    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        批量预测 DataFrame 中的数据

        参数:
            df: 包含 'title' 和 'url' 列的 DataFrame

        返回:
            pd.DataFrame: 添加了 'category' 列的 DataFrame

        Pandas 技巧:
            - df.apply(func, axis=1): 对每行应用函数
            - df['col'].apply(func): 对单列应用函数
            - 使用进度条库 tqdm 可以显示处理进度

        性能优化建议:
            - 对于大量数据，可以考虑向量化操作
            - 或者使用多进程处理
        """
        if df.empty:
            print("⚠️ 输入数据为空")
            return df

        print(f"📊 正在处理 {len(df)} 条数据...")

        # TODO: 实现批量预测
        # 提示：使用 df.apply(lambda row: self.predict(...), axis=1)

        return df

    # ==================== 模型持久化方法 ====================

    def save_model(self, path: str) -> None:
        """
        保存训练好的模型到文件

        参数:
            path: 模型保存路径

        说明:
            模型保存后，下次启动可以直接加载，无需重新训练

        Python 持久化方法:
            - pickle.dump(obj, file): 序列化对象
            - pickle.load(file): 反序列化对象
            - 也可以使用 joblib（sklearn 推荐）

        需要保存的内容:
            - self.model (分类器)
            - self.vectorizer (向量化器)
            - self.categories (类别列表)
        """
        # TODO: 实现模型保存
        # 提示：
        # with open(path, 'wb') as f:
        #     pickle.dump({...}, f)
        pass

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
        # TODO: 实现模型加载
        pass

    def get_category_distribution(self, df: pd.DataFrame) -> pd.Series:
        """
        统计分类结果的分布情况

        参数:
            df: 包含 'category' 列的 DataFrame

        返回:
            pd.Series: 各类别的数量统计

        用途:
            用于分析用户浏览习惯，生成报告

        Pandas 方法:
            - df['col'].value_counts(): 统计各值出现次数
        """
        # TODO: 实现统计逻辑
        pass


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

    # TODO: 编写测试代码

    # 1. 实例化分类器
    classifier = ContentClassifier()

    # 2. 测试单条预测
    # test_cases = [
    #     {"title": "Python 基础教程", "url": "https://www.runoob.com/python"},
    #     {"title": "京东 - 正品低价", "url": "https://www.jd.com"},
    #     ...
    # ]
    # for case in test_cases:
    #     result = classifier.predict(case['title'], case['url'])
    #     print(f"标题: {case['title']} -> {result}")

    # 3. 测试批量预测
    # df = pd.DataFrame(test_cases)
    # result_df = classifier.batch_predict(df)
    # print(result_df)

    print("\n✅ 测试完成")