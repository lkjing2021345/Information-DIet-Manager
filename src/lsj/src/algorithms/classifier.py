import jieba
import pandas as pd
import os


# 如果后续要用机器学习，需要引入 sklearn
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB

class ContentClassifier:
    """
    浏览记录内容分类器

    功能：
    1. 对浏览标题进行分词处理
    2. 基于关键词规则进行快速分类
    3. (预留) 基于机器学习模型进行分类
    """

    def __init__(self, keyword_dict=None):
        """
        初始化分类器

        :param keyword_dict: 自定义的关键词字典，格式为 {'类别': ['词1', '词2']}
        """
        # 定义核心类别 (参考你的开发计划)
        self.categories = [
            "News",  # 新闻
            "Entertainment",  # 娱乐
            "Learning",  # 学习
            "Social",  # 社交
            "Shopping",  # 购物
            "Tools",  # 工具
            "Other"  # 其他
        ]

        # 1. 初始化规则库 (如果没有传入，则使用默认的空字典，后续需要你填充)
        self.rules = keyword_dict if keyword_dict else self._load_default_rules()

        # 2. 初始化机器学习模型相关变量 (占位，Day 3-4 后期实现)
        self.model = None
        self.vectorizer = None

        print("✅ ContentClassifier 初始化完成")

    def _load_default_rules(self):
        """
        (私有方法) 加载默认的关键词规则库
        这是你 Day 3 需要重点填充的部分
        """
        return {
            "Social": ["微信", "知乎", "微博", "weibo", "bilibili"],
            "Learning": ["教程", "文档", "python", "course", "学习", "CSDN", "Stack Overflow"],
            "Shopping": ["淘宝", "京东", "亚马逊", "价格", "优惠券"],
            "Entertainment": ["电影", "小说", "游戏", "直播", "漫画"],
            "News": ["新闻", "日报", "头条", "news", "report"],
            "Tools": ["翻译", "邮箱", "日历", "网盘", "转换"]
        }

    def _preprocess(self, text):
        """
        (私有方法) 文本预处理：分词

        :param text: 原始标题字符串
        :return: 分词后的列表或空格分隔的字符串
        """
        if not isinstance(text, str):
            return ""

        # 使用 jieba 进行分词
        words = jieba.cut(text)
        # 过滤停用词逻辑可以在这里添加
        return list(words)

    def predict_by_rules(self, text, url=None):
        """
        基于规则和关键词的分类 (P0 优先级)

        :param text: 网页标题
        :param url: 网页链接 (辅助判断，例如 domain 包含 'bilibili')
        :return: 匹配到的类别，如果没有匹配则返回 None
        """
        # 1. URL 规则检查 (通常 URL 的域名最准确)
        if url:
            for category, keywords in self.rules.items():
                for kw in keywords:
                    if kw.lower() in url.lower():
                        return category

        # 2. 标题关键词检查
        for category, keywords in self.rules.items():
            for kw in keywords:
                if kw in text:
                    return category

        return None

    def train_model(self, training_data, training_labels):
        """
        (Day 4 任务) 训练朴素贝叶斯分类器

        :param training_data: 文本列表
        :param training_labels: 对应的标签列表
        """
        print("🔄 正在训练模型... (待实现)")
        # 伪代码逻辑：
        # 1. self.vectorizer = TfidfVectorizer()
        # 2. X = self.vectorizer.fit_transform(training_data)
        # 3. self.model = MultinomialNB()
        # 4. self.model.fit(X, training_labels)
        pass

    def predict(self, text, url=None):
        """
        主预测函数：对外暴露的唯一接口
        逻辑：优先使用规则匹配，如果规则未命中，且有模型，则用模型，否则返回 Other
        """
        # 1. 尝试规则匹配
        category = self.predict_by_rules(text, url)
        if category:
            return category

        # 2. (未来) 尝试模型预测
        # if self.model:
        #     return self._predict_by_model(text)

        # 3. 兜底策略
        return "Other"

    def batch_predict(self, df):
        """
        批量预测 pandas DataFrame

        :param df: 包含 'title' 和 'url' 列的 DataFrame
        :return: 增加了 'category' 列的 DataFrame
        """
        if df.empty:
            return df

        print(f"📊 正在处理 {len(df)} 条数据...")

        # 使用 apply 函数应用 predict 方法
        # axis=1 表示按行处理
        df['category'] = df.apply(
            lambda row: self.predict(row.get('title', ''), row.get('url', '')),
            axis=1
        )
        return df


# --- 单元测试代码 (用于直接运行此文件测试) ---
if __name__ == "__main__":
    # 1. 实例化
    classifier = ContentClassifier()

    # 2. 测试单条数据
    test_title = "Python 教程 - 廖雪峰的官方网站"
    test_url = "https://www.liaoxuefeng.com/wiki/python"

    result = classifier.predict(test_title, test_url)
    print(f"测试标题: {test_title}")
    print(f"分类结果: {result}")  # 应该输出 'Learning'

    # 3. 测试 DataFrame
    data = {
        'title': ['京东超市', 'Bilibili 视频', '未知网页'],
        'url': ['jd.com', 'bilibili.com', 'unknown.com']
    }
    df = pd.DataFrame(data)
    result_df = classifier.batch_predict(df)
    print("\n批量测试结果:")
    print(result_df)