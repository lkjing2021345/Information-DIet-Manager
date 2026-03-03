# -*- coding: utf-8 -*-
"""
信息摄取质量评估模块

功能概述：
    整合分类、情感、相似度分析结果，评估用户信息摄取质量
    - 信息茧房检测：识别内容单一、重复度高的浏览模式
    - 信息毒品识别：检测负面情绪、娱乐过度的内容倾向
    - 健康度评分：综合评估信息摄取的多样性和质量
    - 时间趋势分析：追踪用户浏览习惯的变化
    - 个性化建议：基于评估结果提供改进建议
"""
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, fields, is_dataclass
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from datetime import datetime

import pandas as pd
import numpy as np

# 导入已完成的模块
from sentiment import SentimentAnalyzer
from classifier import ContentClassifier
from similarity import SimilarityAnalyzer
from markdown_builder import ReportMarkdownGenerator


# ========= Logger 初始化 =========
def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(level)

    if logger_obj.handlers:
        return logger_obj

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger_obj.addHandler(file_handler)
    logger_obj.addHandler(console_handler)

    logger_obj.propagate = False
    return logger_obj

logger = setup_logger(__name__, "../../logs/evaluator.log")


# ========= 枚举类定义 =========
class HealthLevel(Enum):
    """健康等级"""
    EXCELLENT = "优秀"      # 90-100分
    GOOD = "良好"           # 75-89分
    FAIR = "一般"           # 60-74分
    WARNING = "警告"        # 40-59分
    CRITICAL = "危险"       # 0-39分


class RiskType(Enum):
    """风险类型"""
    ECHO_CHAMBER = "信息茧房"
    TOXIC_CONTENT = "信息毒品"
    TIME_WASTE = "时间浪费"
    EMOTION_POLLUTION = "情绪污染"
    CONTENT_MONOTONY = "内容单一"
    EXCESSIVE_ENTERTAINMENT = "过度娱乐"


class Priority(Enum):
    """优先级"""
    URGENT = "紧急"
    IMPORTANT = "重要"
    NORMAL = "一般"


class Difficulty(Enum):
    """实施难度"""
    EASY = "容易"
    MEDIUM = "中等"
    HARD = "困难"


# ==================== 核心指标数据类 ====================

@dataclass
class DiversityMetrics:
    """多样性维度指标"""
    # 类别多样性
    category_diversity_score: float  # 0-1
    category_count: int
    category_entropy: float  # 香农熵
    dominant_category: str
    dominant_category_ratio: float

    # 内容多样性
    content_diversity_score: float  # 0-1
    avg_similarity: float
    duplicate_ratio: float
    cluster_count: int

    # 原始数据（用于详细分析）
    category_distribution: Dict[str, int] = field(default_factory=dict)
    similarity_distribution: List[float] = field(default_factory=list)


@dataclass
class SentimentHealthMetrics:
    """情感健康维度指标"""
    # 情感健康分数
    sentiment_health_score: float  # 0-1

    # 情感分布
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float

    # 情感极性统计
    polarity_mean: float
    polarity_std: float
    extreme_emotion_count: int

    # 情绪分布（如果有）
    emotion_distribution: Optional[Dict[str, int]] = None
    emotion_stability: Optional[float] = None

    # 原始数据
    sentiment_distribution: Dict[str, int] = field(default_factory=dict)
    polarity_values: List[float] = field(default_factory=list)


@dataclass
class ContentQualityMetrics:
    """内容质量维度指标"""
    # 质量分数
    content_quality_score: float  # 0-1
    weighted_quality_score: float  # 加权分数

    # 类别占比
    learning_ratio: float
    news_ratio: float
    tools_ratio: float
    entertainment_ratio: float
    social_ratio: float
    shopping_ratio: float
    other_ratio: float

    # 原始数据
    category_time_distribution: Dict[str, float] = field(default_factory=dict)
    category_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class TimeAllocationMetrics:
    """时间分配维度指标"""
    # 时间分配合理性
    time_allocation_score: float  # 0-1

    # 时段利用率
    peak_hour_efficiency: float
    off_hour_waste_ratio: float

    # 碎片化程度
    fragmentation_score: float  # 0-1，越高越碎片化
    avg_session_duration: float  # 平均会话时长（分钟）

    # 时间浪费
    low_efficiency_duration: float  # 小时
    late_night_entertainment_duration: float  # 小时

    # 时间占比
    category_time_ratios: Dict[str, float] = field(default_factory=dict)

    # 原始数据
    hourly_distribution: Dict[int, int] = field(default_factory=dict)  # 24小时分布
    weekday_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class EvaluationMetrics:
    """综合评估指标"""
    # 各维度指标
    diversity: DiversityMetrics
    sentiment_health: SentimentHealthMetrics
    content_quality: ContentQualityMetrics
    time_allocation: TimeAllocationMetrics

    # 综合得分
    overall_score: float  # 0-100

    # 权重配置
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        'diversity': 0.25,
        'sentiment_health': 0.25,
        'content_quality': 0.30,
        'time_allocation': 0.20
    })


# ==================== 风险警报数据类 ====================

@dataclass
class Evidence:
    """证据数据"""
    key_statistics: Dict[str, Any] = field(default_factory=dict)  # 关键统计数字
    problem_examples: List[str] = field(default_factory=list)  # 问题内容示例
    time_distribution: Optional[Dict[str, Any]] = None  # 时间分布
    benchmark_comparison: Optional[Dict[str, float]] = None  # 与基准对比


@dataclass
class RiskAlert:
    """风险警报"""
    # 基本信息
    risk_type: RiskType
    severity: int  # 1-5

    # 描述
    brief_description: str  # 一句话简述
    detailed_description: str  # 详细说明

    # 证据
    evidence: Evidence

    # 影响分析
    impact_analysis: str
    potential_consequences: List[str] = field(default_factory=list)

    # 改进建议
    suggestions: List[str] = field(default_factory=list)
    priority: Priority = Priority.NORMAL


# ==================== 详细分析数据类 ====================

@dataclass
class CategoryAnalysis:
    """类别分析"""
    # 分布统计
    distribution_table: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # 格式: {category: {count: int, ratio: float, duration: float}}

    # 时间序列
    time_series: Dict[str, List[Tuple[datetime, int]]] = field(default_factory=dict)
    # 格式: {category: [(timestamp, count), ...]}

    # 转换矩阵
    transition_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # 格式: {from_category: {to_category: probability}}


@dataclass
class SentimentAnalysis:
    """情感分析"""
    # 分布数据（饼图）
    sentiment_pie_data: Dict[str, float] = field(default_factory=dict)

    # 时间序列
    sentiment_time_series: List[Tuple[datetime, str, float]] = field(default_factory=list)
    # 格式: [(timestamp, sentiment, polarity), ...]

    # 交叉分析
    sentiment_by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # 格式: {category: {sentiment: ratio}}

    # Top内容
    top_positive_content: List[Dict[str, Any]] = field(default_factory=list)
    top_negative_content: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SimilarityAnalysis:
    """相似度分析"""
    # 分布数据（直方图）
    similarity_histogram: Dict[str, int] = field(default_factory=dict)
    # 格式: {bin_range: count}

    # 高相似度对
    high_similarity_pairs: List[Tuple[int, int, float]] = field(default_factory=list)
    # 格式: [(idx1, idx2, similarity), ...]

    # 聚类结果
    cluster_labels: List[int] = field(default_factory=list)
    cluster_centers: Optional[List[str]] = None  # 每个簇的代表性文本


@dataclass
class TimePatternAnalysis:
    """时间模式分析"""
    # 24小时热力图
    hourly_heatmap: Dict[int, Dict[str, int]] = field(default_factory=dict)
    # 格式: {hour: {category: count}}

    # 工作日vs周末
    weekday_vs_weekend: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 高峰时段
    peak_hours: List[int] = field(default_factory=list)
    peak_categories: Dict[int, str] = field(default_factory=dict)

    # 浏览时长分布
    duration_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class AnomalyDetection:
    """异常检测"""
    # 异常行为列表
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    # 格式: [{timestamp, type, description, severity}, ...]

    # 突变点
    change_points: List[Tuple[datetime, str, float]] = field(default_factory=list)
    # 格式: [(timestamp, metric_name, change_magnitude), ...]

    # 周期性模式
    periodic_patterns: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetailedAnalysis:
    """详细分析"""
    category_analysis: CategoryAnalysis
    sentiment_analysis: SentimentAnalysis
    similarity_analysis: SimilarityAnalysis
    time_pattern_analysis: TimePatternAnalysis
    anomaly_detection: AnomalyDetection


# ==================== 改进建议数据类 ====================

@dataclass
class ActionableRecommendation:
    """可操作建议"""
    action: str  # 具体行动
    reason: str  # 预期效果
    difficulty: Difficulty
    expected_improvement: float  # 预期改善幅度（0-1）


@dataclass
class CategoryBalanceRecommendations:
    """类别平衡建议"""
    categories_to_increase: List[str] = field(default_factory=list)
    categories_to_decrease: List[str] = field(default_factory=list)
    specific_content_suggestions: List[str] = field(default_factory=list)
    actions: List[ActionableRecommendation] = field(default_factory=list)


@dataclass
class TimeManagementRecommendations:
    """时间管理建议"""
    time_slot_adjustments: List[str] = field(default_factory=list)
    fragmentation_reduction: List[str] = field(default_factory=list)
    time_limits: Dict[str, float] = field(default_factory=dict)  # {category: hours}
    actions: List[ActionableRecommendation] = field(default_factory=list)


@dataclass
class EmotionRegulationRecommendations:
    """情感调节建议"""
    reduce_negative_content: List[str] = field(default_factory=list)
    increase_positive_content: List[str] = field(default_factory=list)
    emotion_buffer_strategies: List[str] = field(default_factory=list)
    actions: List[ActionableRecommendation] = field(default_factory=list)


@dataclass
class ContentQualityRecommendations:
    """内容质量提升建议"""
    increase_learning_content: List[str] = field(default_factory=list)
    optimize_information_sources: List[str] = field(default_factory=list)
    deep_reading_suggestions: List[str] = field(default_factory=list)
    actions: List[ActionableRecommendation] = field(default_factory=list)


@dataclass
class Recommendations:
    """改进建议"""
    # 按优先级分组
    urgent_recommendations: List[ActionableRecommendation] = field(default_factory=list)
    important_recommendations: List[ActionableRecommendation] = field(default_factory=list)
    normal_recommendations: List[ActionableRecommendation] = field(default_factory=list)

    # 分类建议
    category_balance: CategoryBalanceRecommendations = field(default_factory=CategoryBalanceRecommendations)
    time_management: TimeManagementRecommendations = field(default_factory=TimeManagementRecommendations)
    emotion_regulation: EmotionRegulationRecommendations = field(default_factory=EmotionRegulationRecommendations)
    content_quality: ContentQualityRecommendations = field(default_factory=ContentQualityRecommendations)


# ==================== 趋势分析数据类 ====================

@dataclass
class HistoricalComparison:
    """历史对比"""
    comparison_period: str  # "上周" / "上月"
    metric_changes: Dict[str, float] = field(default_factory=dict)  # {metric: change_rate}
    improvement_trends: List[str] = field(default_factory=list)
    deterioration_trends: List[str] = field(default_factory=list)


@dataclass
class PredictiveInsights:
    """预测性提示"""
    risk_predictions: List[str] = field(default_factory=list)
    potential_issues: List[str] = field(default_factory=list)
    preventive_suggestions: List[str] = field(default_factory=list)


@dataclass
class Milestones:
    """里程碑记录"""
    best_performance_date: Optional[datetime] = None
    best_performance_score: Optional[float] = None
    worst_performance_date: Optional[datetime] = None
    worst_performance_score: Optional[float] = None
    turning_points: List[Tuple[datetime, str]] = field(default_factory=list)
    # 格式: [(timestamp, description), ...]


@dataclass
class TrendAnalysis:
    """趋势分析"""
    historical_comparison: HistoricalComparison
    predictive_insights: PredictiveInsights
    milestones: Milestones


# ==================== 顶层报告结构 ====================

@dataclass
class ReportMetadata:
    """报告元信息"""
    # 时间范围
    start_date: datetime
    end_date: datetime

    # 数据统计
    total_records: int
    valid_records: int
    time_span_days: int

    # 生成信息
    generated_at: datetime
    evaluator_version: str
    config_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """健康状态"""
    level: HealthLevel
    score: float  # 0-100
    justification: str  # 判定依据


@dataclass
class EvaluationReport:
    """完整评估报告"""
    # 元信息
    metadata: ReportMetadata

    # 健康状态
    health_status: HealthStatus

    # 核心指标
    metrics: EvaluationMetrics

    # 风险警报
    risk_alerts: List[RiskAlert] = field(default_factory=list)

    # 详细分析
    detailed_analysis: Optional[DetailedAnalysis] = None

    # 改进建议
    recommendations: Recommendations = field(default_factory=Recommendations)

    # 趋势信息（可选）
    trend_analysis: Optional[TrendAnalysis] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于JSON序列化）"""
        def dataclass_to_dict(obj: Any) -> Any:
            if obj is None:
                return None

            if is_dataclass(obj):
                result = {}
                for filed in fields(obj):
                    value = getattr(obj, filed.name)
                    result[filed.name] = dataclass_to_dict(value)
                return result

            # 处理枚举
            if isinstance(obj, Enum):
                return obj.value

            # 处理 datetime
            if isinstance(obj, datetime):
                return obj.isoformat()

            # 处理列表
            if isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]

            # 处理元组
            if isinstance(obj, tuple):
                return tuple(dataclass_to_dict(item) for item in obj)

            # 处理字典
            if isinstance(obj, dict):
                return {key: dataclass_to_dict(value) for key, value in obj.items()}

            return obj

        return dataclass_to_dict(self)

    def to_json(self, filepath: str, indent: int = 2) -> None:
        """导出为JSON"""
        data = self.to_dict()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)

        logger.info(f"报告导出到: {filepath}")

    def to_markdown(self, filepath: str, detailed: bool = True) -> None:
        """导出为Markdown"""
        generator = ReportMarkdownGenerator()

        generator.save(self, filepath=filepath, detailed=detailed)

        logger.info(f"Markdown 报告已保存到: {filepath}")

    def get_summary(self) -> str:
        """获取文字摘要"""
        # ===== 1. 基础信息 =====
        total = self.metadata.total_records or 0  # 总记录数
        valid = self.metadata.valid_records or 0  # 有效记录数
        valid_rate = (valid / total * 100) if total > 0 else 0.0  # 防止除零
        # ===== 2. 四个维度分数 =====
        dimension_scores = {
            "多样性": self.metrics.diversity.category_diversity_score * 100,
            "情感健康": self.metrics.sentiment_health.sentiment_health_score * 100,
            "内容质量": self.metrics.content_quality.content_quality_score * 100,
            "时间分配": self.metrics.time_allocation.time_allocation_score * 100,
        }
        # 取最好和最弱维度
        best_dim = max(dimension_scores, key=dimension_scores.get)
        worst_dim = min(dimension_scores, key=dimension_scores.get)
        # ===== 3. 风险统计 =====
        risk_total = len(self.risk_alerts) if self.risk_alerts else 0
        risk_critical = len([r for r in self.risk_alerts if r.severity >= 4]) if self.risk_alerts else 0
        # ===== 4. 建议统计 =====
        urgent_count = len(self.recommendations.urgent_recommendations) if self.recommendations else 0
        important_count = len(self.recommendations.important_recommendations) if self.recommendations else 0
        normal_count = len(self.recommendations.normal_recommendations) if self.recommendations else 0
        # ===== 5. 组装摘要文本 =====
        lines = [
            "信息摄取质量评估摘要",
            f"- 时间范围: {self.metadata.start_date.strftime('%Y-%m-%d')} ~ {self.metadata.end_date.strftime('%Y-%m-%d')}（{self.metadata.time_span_days} 天）",
            f"- 数据情况: 总记录 {total} 条，有效 {valid} 条（有效率 {valid_rate:.1f}%）",
            f"- 综合结果: {self.health_status.level.value}（{self.health_status.score:.1f}/100）",
            f"- 最佳维度: {best_dim}（{dimension_scores[best_dim]:.1f}/100）",
            f"- 薄弱维度: {worst_dim}（{dimension_scores[worst_dim]:.1f}/100）",
            f"- 风险情况: 共 {risk_total} 项，其中严重风险 {risk_critical} 项",
            f"- 改进建议: 紧急 {urgent_count} 条 / 重要 {important_count} 条 / 一般 {normal_count} 条",
            f"- 判定依据: {self.health_status.justification}",
        ]
        return "\n".join(lines)


# ========= 主评估器类 =========
class InformationQualityEvaluator:
    """
    信息摄取质量评估器

    设计思路：
        1. 整合三个分析器的结果
        2. 计算多维度评估指标
        3. 识别潜在风险模式
        4. 生成综合评估报告
    """

    # ==================== 类常量 ====================
    # 定义阈值常量
    # 信息茧房：平均相似度阈值
    ECHO_CHAMBER_SIMILARITY_LIMIT = 0.75
    # 信息茧房：主导类别占比阈值
    DOMINANT_CATEGORY_RATIO_LIMIT = 0.60
    # 情感健康：负面内容占比警戒线
    NEGATIVE_RATIO_WARNING = 0.40
    # 时间分配：娱乐内容占比警戒线
    ENTERTAINMENT_RATIO_WARNING = 0.50
    # 权重
    DEFAULT_WEIGHTS = {
        "diversity": 0.25,  # 多样性权重
        "sentiment_health": 0.25,  # 情感健康权重
        "content_quality": 0.30,  # 内容质量权重
        "time_allocation": 0.20  # 时间分配权重
    }

    def __init__(
        self,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
        content_classifier: Optional[ContentClassifier] = None,
        similarity_analyzer: Optional[SimilarityAnalyzer] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化评估器

        参数:
            sentiment_analyzer: 情感分析器实例
            content_classifier: 内容分类器实例
            similarity_analyzer: 相似度分析器实例
            config: 自定义配置（阈值、权重等）
        """
        # 初始化三个分析器
        self.sentiment = sentiment_analyzer if sentiment_analyzer else SentimentAnalyzer()
        self.content = content_classifier if content_classifier else ContentClassifier()
        self.similarity = similarity_analyzer if similarity_analyzer else SimilarityAnalyzer()

        # 加载配置（阈值、权重、评分规则）
        self.config = self.get_default_config()
        if config is not None:
            self.update_config(config)

        # 初始化缓存和状态变量
        self._cache : Dict[str, Any] = {}
        self.last_report: Optional[EvaluationReport] = None

        logger.info("InformationQualityEvaluator 初始化完成")

    # ==================== 私有方法：数据预处理 ====================

    def _attach_timestamp_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        统一时间字段到 timestamp 列
        支持输入列: timestamp / visit_time / ts
        """
        working_df = df.copy()
        parsed_ts = None

        if "timestamp" in working_df.columns:
            parsed_ts = pd.to_datetime(working_df["timestamp"], errors="coerce")
        elif "visit_time" in working_df.columns:
            parsed_ts = pd.to_datetime(working_df["visit_time"], errors="coerce")
        elif "ts" in working_df.columns:
            ts_num = pd.to_numeric(working_df["ts"], errors="coerce")
            if ts_num.notna().any():
                inferred_unit = "ms" if float(ts_num.dropna().median()) > 1e11 else "s"
                parsed_ts = pd.to_datetime(ts_num, unit=inferred_unit, errors="coerce")
            else:
                parsed_ts = pd.to_datetime(working_df["ts"], errors="coerce")

        if parsed_ts is not None:
            working_df["timestamp"] = parsed_ts
        return working_df

    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        验证输入数据格式
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("输入数据必须是 pandas.DataFrame")
            raise TypeError("输入数据必须是 pandas.DataFrame")

        if df.empty:
           logger.error("输入 DataFrame 为空，无法评估")
           raise ValueError("输入 DataFrame 为空，无法评估")

        required_columns = {"title", "url", "category", "sentiment", "polarity", "similarity"}
        missing_columns = sorted(required_columns - set(df.columns))

        if missing_columns:
            logger.error(f"缺少必需列: {missing_columns}")
            raise ValueError(f"缺少必需列: {missing_columns}")

        min_records = int(self.config.get('min_records', 5))

        if len(df) < min_records:
            logger.warning(f"样本量不足：当前 {len(df)} 条，至少需要 {min_records} 条")
            raise ValueError(f"样本量不足：当前 {len(df)} 条，至少需要 {min_records} 条")

        return True


    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        """
        processed_df = df.copy()

        processed_df['title'] = processed_df['title'].astype(str).str.strip()
        processed_df['url'] = processed_df['url'].astype(str).str.strip()
        processed_df['category'] = processed_df['category'].astype(str).str.strip()
        processed_df['sentiment'] = processed_df['sentiment'].astype(str).str.strip()

        processed_df = processed_df.dropna(subset=["category", "sentiment", "polarity", "similarity"])

        processed_df['polarity'] = pd.to_numeric(processed_df['polarity'], errors='coerce')
        processed_df['similarity'] = pd.to_numeric(processed_df['similarity'], errors='coerce')

        processed_df = processed_df.dropna(subset=["polarity", "similarity"])

        processed_df["polarity"] = processed_df["polarity"].clip(-1.0, 1.0)
        processed_df["similarity"] = processed_df["similarity"].clip(0.0, 1.0)

        processed_df = self._attach_timestamp_column(processed_df)

        if "timestamp" in processed_df.columns:
            processed_df = processed_df.dropna(subset=["timestamp"])
            processed_df = processed_df.sort_values(by=["timestamp"], ascending=True)

            processed_df["date"] = processed_df["timestamp"].dt.date
            processed_df["hour"] = processed_df["timestamp"].dt.hour
            processed_df["weekday"] = processed_df["timestamp"].dt.day_name()

        processed_df = processed_df.reset_index(drop=True)

        min_records = int(self.config.get("min_records", 5))
        if len(processed_df) < min_records:
            raise ValueError(
                f"预处理后有效样本不足：当前 {len(processed_df)} 条，至少需要 {min_records} 条"
            )

        logger.info(f"数据预处理完成：原始 {len(df)} 条 -> 有效 {len(processed_df)} 条")
        return processed_df

    def _filter_by_time_range(
            self,
            df: pd.DataFrame,
            time_range: Optional[Tuple[str, str]] = None
    ) -> pd.DataFrame:
        """
        按 time_range 过滤数据（闭区间）：[start, end]
        """
        if time_range is None:
            return df

        if not isinstance(time_range, tuple) or len(time_range) != 2:
            raise ValueError("time_range 必须是 (start, end) 二元组")

        working_df = self._attach_timestamp_column(df)

        if "timestamp" not in working_df.columns:
            raise ValueError("指定了 time_range，但数据中没有可用时间列（timestamp/visit_time/ts）")

        start = pd.to_datetime(time_range[0], errors="coerce")
        end = pd.to_datetime(time_range[1], errors="coerce")

        if pd.isna(start) or pd.isna(end):
            raise ValueError(f"time_range 解析失败: {time_range}，请使用可识别时间格式")
        if start > end:
            raise ValueError(f"time_range 起始时间不能晚于结束时间: start={start}, end={end}")

        working_df = working_df.dropna(subset=["timestamp"]).copy()

        mask = (working_df["timestamp"] >= start) & (working_df["timestamp"] <= end)
        filtered_df = working_df.loc[mask].copy()

        if filtered_df.empty:
            raise ValueError(
                f"time_range 过滤后无数据: [{start}, {end}]。"
                "请检查时间范围是否覆盖到你的数据。"
            )

        filtered_df = filtered_df.sort_values("timestamp").reset_index(drop=True)
        logger.info(
            f"time_range 过滤完成: [{start}, {end}]，"
            f"过滤前 {len(df)} 条 -> 过滤后 {len(filtered_df)} 条"
        )

        return filtered_df

    # ==================== 私有方法：多样性分析 ====================

    def _calculate_category_diversity(self, df: pd.DataFrame) -> float:
        """
        计算类别多样性（基于分类结果）
        """
        category_counts = df['category'].value_counts()
        probs = category_counts / category_counts.sum()

        H = calculate_shannon_entropy(probs.tolist())

        n_categories = len(category_counts)
        if n_categories <= 1:
            H_max = 0.0
        else:
            H_max = np.log2(n_categories)

        if H_max == 0.0:
            score = 0.0
        else:
            score = H / H_max

        score = float(np.clip(score, 0.0, 1.0))

        return score

    def _calculate_content_diversity(self, df: pd.DataFrame) -> float:
        """
        计算内容多样性（基于相似度）
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入数据必须是 pandas.DataFrame")
        if "similarity" not in df.columns:
            raise ValueError("缺少 similarity 列，无法计算内容多样性")

        sim = pd.to_numeric(df["similarity"], errors="coerce").dropna().clip(0.0, 1.0)
        if sim.empty:
            # 无有效相似度时 给中性分 缓存空细节
            self._cache["content_diversity_details"] = {
                "avg_similarity": 0.5,
                "duplicate_ratio": 0.0,
                "cluster_count": 0,
                "concentration": 0.0,
                "similarity_distribution": []
            }
            return 0.5

        avg_similarity = float(sim.mean())
        duplicate_threshold = 0.85
        duplicate_ratio = float((sim >= duplicate_threshold).mean())

        bins = [0.0, 0.4, 0.7, 1.01]
        labels = ["low", "mid", "high"]
        bucket = pd.cut(sim, bins=bins, labels=labels, right=False, include_lowest=True)

        bucket_counts = bucket.value_counts(dropna=True)
        cluster_count = int((bucket_counts > 0).sum())
        concentration = float(bucket_counts.max() / len(sim)) if not bucket_counts.empty else 0.0

        score = 1.0 - (0.50 * avg_similarity + 0.35 * duplicate_ratio + 0.15 * concentration)
        score = float(np.clip(score, 0.0, 1.0))

        self._cache["content_diversity_details"] = {
            "avg_similarity": avg_similarity,
            "duplicate_ratio": duplicate_ratio,
            "cluster_count": cluster_count,
            "concentration": concentration,
            "similarity_distribution": sim.tolist(),
            "bucket_distribution": {str(k): int(v) for k, v in bucket_counts.to_dict().items()}
        }

        return score

    def _detect_echo_chamber(self, df: pd.DataFrame) -> Tuple[bool, float, Dict[str, Any]]:
        """
        检测信息茧房

        TODO: 综合类别集中度和内容相似度
        TODO: 分析时间窗口内的变化趋势
        TODO: 返回 (是否存在, 严重程度, 详细证据)
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df 必须是 pandas.DataFrame")
        if df.empty:
            return False, 0.0, {"reason": "empty_dataframe"}

        category_series = df["category"].astype(str).str.lower().str.strip() \
            if "category" in df.columns \
            else pd.Series([], dtype=str)

        sim_series = pd.to_numeric(df["similarity"], errors="coerce").dropna().clip(0.0, 1.0) \
            if "similarity" in df.columns \
            else pd.Series([], dtype=float)

        if category_series.empty:
            dominant_ratio = 0.0
            dominant_category = "other"
        else:
            counts = category_series.value_counts()
            dominant_category = str(counts.index[0])
            dominant_ratio = float(counts.iloc[0] / counts.sum())

        avg_similarity = float(sim_series.mean()) if not sim_series.empty else 0.0
        high_similarity_ratio = float((sim_series >= 0.85).mean()) if not sim_series.empty else 0.0

        dom_limit = float(self.config.get("thresholds", {}).get("dominant_category_ratio", self.DOMINANT_CATEGORY_RATIO_LIMIT))
        sim_limit = float(self.config.get("thresholds", {}).get("echo_chamber_similarity", self.ECHO_CHAMBER_SIMILARITY_LIMIT))

        dom_exceed = max(0.0, (dominant_ratio - dom_limit) / max(1e-6, 1.0 - dom_limit))
        sim_exceed = max(0.0, (avg_similarity - sim_limit) / max(1e-6, 1.0 - sim_limit))

        severity = float(np.clip(0.55 * dom_exceed + 0.45 * sim_exceed, 0.0, 1.0))
        exists = bool((dominant_ratio > dom_limit) or (avg_similarity > sim_limit))

        evidence = {
            "dominant_category": dominant_category,
            "dominant_category_ratio": dominant_ratio,
            "dominant_ratio_limit": dom_limit,
            "avg_similarity": avg_similarity,
            "similarity_limit": sim_limit,
            "high_similarity_ratio": high_similarity_ratio,
            "record_count": int(len(df)),
        }

        return exists, severity, evidence


    # ==================== 私有方法：情感健康分析 ====================

    def _calculate_sentiment_health(self,
                                    df: pd.DataFrame,
                                    std_scale: float = 1.0,
                                    alpha: float = 0.7,
                                    neutral_weight: float = 0.5,
                                    neutral_eps: float = 0.0,
                                    empty_score: float = 0.5
                                    ) -> float:
        """
        计算情感健康分数
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("输入数据必须是 pandas.DataFrame")
            raise TypeError("输入数据必须是 pandas.DataFrame")

        if df.empty or df is None:
            logger.error("输入 DataFrame 为空，无法评估")
            raise ValueError("输入 DataFrame 为空，无法评估")

        s = df['polarity']
        s_valid = s.dropna()

        if s_valid.empty:
            return float(np.clip(empty_score, 0.0, 1.0))

        s_num = pd.to_numeric(s_valid, errors='coerce')
        if s_num.notna().all():
            total = len(s_num)

            positive_mask = s_num > neutral_eps
            negative_mask = s_num < -neutral_eps
            neutral_mask = ~(positive_mask | negative_mask)

            positive_ratio = float(positive_mask.mean())
            negative_ratio = float(negative_mask.mean())
            neutral_ratio = float(neutral_mask.mean())

            polarity_std = float(s_num.std(ddof=1)) if total >= 2 else 0.0
        else:
            s_str = s_valid.astype(str).str.lower()
            total = len(s_str)

            positive_ratio = float((s_str == "positive").mean())
            negative_ratio = float((s_str == "negative").mean())
            neutral_ratio = float((s_str == "neutral").mean())

            polarity_std = 0.0

        polarity_std_norm = float(np.clip(polarity_std / std_scale, 0.0, 1.0))

        balance = positive_ratio + neutral_weight * neutral_ratio - negative_ratio
        balance_norm = float(np.clip((balance + 1.0) / 2.0, 0.0, 1.0))

        stability = 1.0 - polarity_std_norm

        score = alpha * balance_norm + (1.0 - alpha) * stability
        score = float(np.clip(score, 0.0, 1.0))

        return score



    def _detect_toxic_content(self, df: pd.DataFrame) -> Tuple[bool, float, List[Dict]]:
        """
        检测信息毒品（负面、成瘾性内容）

        TODO: 识别高负面情绪内容
        TODO: 检测娱乐内容过度消费
        TODO: 分析访问频率和时长模式
        TODO: 返回 (是否存在, 严重程度, 问题内容列表)
        """
        pass

    def _analyze_emotion_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析情绪模式

        TODO: 统计各类情绪的分布
        TODO: 分析情绪随时间的变化
        TODO: 识别情绪触发因素（类别、时间段）
        """
        pass

    # ==================== 私有方法：内容质量分析 ====================

    def _calculate_content_quality(self,
                                   df: pd.DataFrame,
                                   category_col: str = "category",
                                   polarity_col: str = "polarity",
                                   use_sentiment_adjust: bool = True,
                                   default_weight: float = 0.4,
                                   category_weights: Optional[dict[str, float]] = None) -> float:
        """
        计算内容质量分数
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("输入数据必须是 pandas.DataFrame")
            raise TypeError("输入数据必须是 pandas.DataFrame")

        if df.empty or df is None:
            logger.error("输入 DataFrame 为空，无法计算内容质量分数")
            raise ValueError("输入 DataFrame 为空，无法计算内容质量分数")

        if category_weights is None:
            category_weights = {
                "learning": 1.0,
                "news": 0.8,
                "tools": 0.75,
                "social": 0.5,
                "shopping": 0.45,
                "entertainment": 0.3,
                "other": 0.4,
            }

        if category_col not in df.columns:
            logger.error("缺少类别列 \'category\'，无法计算类别质量")
            raise TypeError("缺少类别列 \'category\'，无法计算类别质量")

        category = df[category_col].dropna()

        if category.empty:
            logger.warning("去除无效值后，表为空，无法计算类别质量")
            raise ValueError("去除无效值后，表为空，无法计算类别质量")

        category = category.astype(str).str.lower()
        ratios = category.value_counts(normalize=True)

        base_score = 0.0
        for c, r in ratios.items():
            c = str(c)
            w = float(category_weights.get(c, default_weight))
            base_score += float(r) * float(w)

        adjust = 0.0
        if use_sentiment_adjust and polarity_col in df.columns:
            s = df[polarity_col].dropna()

            if not s.empty:
                s_num = pd.to_numeric(s, errors="coerce")
                if s_num.notna().all():
                    positive_ratio = float((s_num > 0).mean())
                    negative_ratio = float((s_num < 0).mean())
                else:
                    s_str = s.astype(str).str.lower()
                    positive_ratio = float((s_str == "positive").mean())
                    negative_ratio = float((s_str == "negative").mean())

                adjust = 0.1 * positive_ratio - 0.1 * negative_ratio

        score = base_score + adjust
        score = float(np.clip(score, 0.0, 1.0))

        return score


    def _analyze_learning_ratio(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        分析学习类内容占比

        TODO: 计算学习类内容的数量和时长占比
        TODO: 对比不同时间段的变化
        TODO: 返回详细统计
        """
        pass

    # ==================== 私有方法：时间分配分析 ====================

    def _analyze_time_allocation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析时间分配合理性

        兼容时间字段：timestamp / visit_time / ts
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入数据必须是 pandas.DataFrame")

        if df.empty:
            raise ValueError("输入 DataFrame 为空，无法分析时间分配")

        working_df = df.copy()

        time_col = "timestamp"

        if time_col in working_df.columns:
            working_df[time_col] = pd.to_datetime(working_df[time_col], errors="coerce")
        else:
            working_df = self._attach_timestamp_column(working_df)

        if time_col not in working_df.columns:
            raise ValueError("缺少可用时间列（timestamp/visit_time/ts），无法分析时间分配")

        working_df = working_df.dropna(subset=[time_col]).sort_values(time_col).copy()

        if working_df.empty:
            raise ValueError("可用时间数据为空，无法分析时间分配")

        working_df["hour"] = working_df[time_col].dt.hour
        working_df["weekday"] = working_df[time_col].dt.day_name()

        # 计算类别时间占比 在没有真实 duration 时 用记录数占比近似
        if "category" in working_df.columns:
            category_series = working_df["category"].astype(str).str.lower().str.strip()
            category_time_ratios = category_series.value_counts(normalize=True).to_dict()
        else:
            category_series = pd.Series(["other"] * len(working_df), index=working_df.index)
            category_time_ratios = {"other": 1.0}

        # 时段分布
        hourly_distribution = {}
        weekday_distribution = {}
        if working_df["hour"].notna().any():
            hourly_distribution = (
                working_df["hour"]
                .dropna()
                .astype(int)
                .value_counts()
                .sort_index()
                .to_dict()
            )
        if working_df["weekday"].notna().any():
            weekday_distribution = (
                working_df["weekday"]
                .dropna()
                .astype(str)
                .value_counts()
                .to_dict()
            )

        # 低效时段与深夜娱乐
        off_hours = {23, 0, 1, 2, 3, 4, 5, 6}
        off_hour_mask = working_df["hour"].isin(off_hours) if "hour" in working_df.columns else pd.Series(False, index=working_df.index)
        off_hour_count = int(off_hour_mask.sum())
        off_hour_waste_ratio = float(off_hour_count / len(working_df))

        entertainment_mask = category_series.eq("entertainment")
        late_night_entertainment_count = int((off_hour_mask & entertainment_mask).sum())

        # 会话碎片化 通过相邻访问时间差估算
        avg_session_duration = 5.0  # 默认值（分钟）
        fragmentation_score = 0.5
        if time_col is not None and len(working_df) >= 2:
            deltas = working_df[time_col].diff().dt.total_seconds().div(60).dropna()
            # 间隔大于 30 分钟视作新会话切分点
            boundary_points = deltas[deltas > 30].index.tolist()

            session_starts = [working_df.index.min()] + boundary_points
            session_ends = boundary_points + [working_df.index.max()]

            session_durations = []
            for s_idx, e_idx in zip(session_starts, session_ends):
                start_time = working_df.loc[s_idx, time_col]
                end_time = working_df.loc[e_idx, time_col]
                duration_min = max((end_time - start_time).total_seconds() / 60.0, 1.0)
                session_durations.append(duration_min)

            if session_durations:
                avg_session_duration = float(np.mean(session_durations))

            # 平均会话越短，碎片化越高
            fragmentation_score = float(np.clip(1.0 - (avg_session_duration / 30.0), 0.0, 1.0))

        # 峰时效率（09:00-18:00 中高价值类别占比）
        productive_categories = {"learning", "news", "tools"}
        peak_mask = working_df["hour"].between(9, 18) if "hour" in working_df.columns else pd.Series(False, index=working_df.index)
        peak_total = int(peak_mask.sum())
        if peak_total > 0:
            peak_productive = int((peak_mask & category_series.isin(productive_categories)).sum())
            peak_hour_efficiency = float(peak_productive / peak_total)
        else:
            peak_hour_efficiency = 0.5

        # 时长近似：无 duration，用“每条记录约 3 分钟”估算
        minutes_per_record = 3.0
        low_efficiency_duration = float(off_hour_count * minutes_per_record / 60.0)
        late_night_entertainment_duration = float(late_night_entertainment_count * minutes_per_record / 60.0)

        # 8) 综合分 惩罚项越高分越低
        late_night_ratio = float(late_night_entertainment_count / len(working_df))
        time_allocation_score = 1.0 - (
            0.45 * off_hour_waste_ratio +
            0.30 * fragmentation_score +
            0.25 * late_night_ratio
        )
        time_allocation_score = float(np.clip(time_allocation_score, 0.0, 1.0))

        return {
            "time_allocation_score": time_allocation_score,
            "peak_hour_efficiency": float(np.clip(peak_hour_efficiency, 0.0, 1.0)),
            "off_hour_waste_ratio": float(np.clip(off_hour_waste_ratio, 0.0, 1.0)),
            "fragmentation_score": float(np.clip(fragmentation_score, 0.0, 1.0)),
            "avg_session_duration": float(avg_session_duration),
            "low_efficiency_duration": float(max(low_efficiency_duration, 0.0)),
            "late_night_entertainment_duration": float(max(late_night_entertainment_duration, 0.0)),
            "category_time_ratios": category_time_ratios,
            "hourly_distribution": hourly_distribution,
            "weekday_distribution": weekday_distribution,
        }

    def _detect_time_waste(self, df: pd.DataFrame) -> List[RiskAlert]:
        """
        检测时间浪费模式

        TODO: 识别过度娱乐时段
        TODO: 检测碎片化浏览
        TODO: 分析低效浏览模式
        """
        pass

    # ==================== 私有方法：趋势分析 ====================

    def _analyze_temporal_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析时间趋势

        TODO: 计算各指标的时间序列变化
        TODO: 识别改善或恶化趋势
        TODO: 预测未来风险
        """
        pass

    def _compare_time_periods(
        self,
        df: pd.DataFrame,
        period1: Tuple[str, str],
        period2: Tuple[str, str]
    ) -> Dict[str, Any]:
        """
        对比不同时间段

        TODO: 计算各指标的变化率
        TODO: 识别显著变化
        TODO: 生成对比报告
        """
        pass

    # ==================== 私有方法：风险识别 ====================

    def _identify_risks(self, df: pd.DataFrame, metrics: EvaluationMetrics) -> List[RiskAlert]:
        """
        综合识别风险

        依据 metrics 中已计算指标进行判定，避免重复扫描原始数据。
        """
        if not isinstance(metrics, EvaluationMetrics):
            raise TypeError("metrics 必须是 EvaluationMetrics 实例")

        risks: List[RiskAlert] = []
        thresholds = self.config.get("thresholds", {})

        def build_alert(
            risk_type: RiskType,
            severity: int,
            brief_description: str,
            detailed_description: str,
            impact_analysis: str,
            key_statistics: Dict[str, Any],
            suggestions: List[str],
        ) -> RiskAlert:
            severity = int(np.clip(severity, 1, 5))
            priority = Priority.URGENT if severity >= 5 else Priority.IMPORTANT if severity >= 4 else Priority.NORMAL
            return RiskAlert(
                risk_type=risk_type,
                severity=severity,
                brief_description=brief_description,
                detailed_description=detailed_description,
                evidence=Evidence(key_statistics=key_statistics),
                impact_analysis=impact_analysis,
                potential_consequences=[],
                suggestions=suggestions,
                priority=priority,
            )

        # 信息茧房：类别过于集中 + 内容相似度过高
        dom_ratio = float(metrics.diversity.dominant_category_ratio)
        avg_similarity = float(metrics.diversity.avg_similarity)
        dom_limit = float(thresholds.get("dominant_category_ratio", self.DOMINANT_CATEGORY_RATIO_LIMIT))
        sim_limit = float(thresholds.get("echo_chamber_similarity", self.ECHO_CHAMBER_SIMILARITY_LIMIT))

        if dom_ratio > dom_limit or avg_similarity > sim_limit:
            exceed_ratio = max(
                (dom_ratio - dom_limit) / max(1e-6, 1 - dom_limit),
                (avg_similarity - sim_limit) / max(1e-6, 1 - sim_limit),
            )
            severity = 3 + int(np.clip(np.ceil(exceed_ratio * 2), 0, 2))
            risks.append(
                build_alert(
                    risk_type=RiskType.ECHO_CHAMBER,
                    severity=severity,
                    brief_description="内容同质化明显，存在信息茧房倾向",
                    detailed_description=(
                        f"主导类别占比 {dom_ratio:.1%}（阈值 {dom_limit:.1%}），"
                        f"平均相似度 {avg_similarity:.2f}（阈值 {sim_limit:.2f}）。"
                    ),
                    impact_analysis="信息输入结构单一，可能导致认知视角收窄。",
                    key_statistics={
                        "dominant_category": metrics.diversity.dominant_category,
                        "dominant_category_ratio": dom_ratio,
                        "avg_similarity": avg_similarity,
                    },
                    suggestions=[
                        "主动增加学习/新闻/工具类内容比例",
                        "每周设定至少 2 个新主题进行探索",
                    ],
                )
            )

        # 情绪污染：负面占比高
        negative_ratio = float(metrics.sentiment_health.negative_ratio)
        negative_limit = float(thresholds.get("negative_ratio_warning", self.NEGATIVE_RATIO_WARNING))
        if negative_ratio > negative_limit:
            exceed = (negative_ratio - negative_limit) / max(1e-6, 1 - negative_limit)
            severity = 3 + int(np.clip(np.ceil(exceed * 2), 0, 2))
            risks.append(
                build_alert(
                    risk_type=RiskType.EMOTION_POLLUTION,
                    severity=severity,
                    brief_description="负面内容占比较高，情绪健康受损",
                    detailed_description=(
                        f"负面内容占比 {negative_ratio:.1%}，超过预警阈值 {negative_limit:.1%}。"
                    ),
                    impact_analysis="长期暴露于负面信息可能增加焦虑与压力。",
                    key_statistics={
                        "negative_ratio": negative_ratio,
                        "polarity_std": float(metrics.sentiment_health.polarity_std),
                    },
                    suggestions=[
                        "降低高负面来源订阅频率",
                        "增加中性/积极信息配比，平衡情绪负荷",
                    ],
                )
            )

        # 过度娱乐：娱乐内容占比高
        entertainment_ratio = float(metrics.content_quality.entertainment_ratio)
        entertainment_limit = float(thresholds.get("entertainment_ratio_warning", self.ENTERTAINMENT_RATIO_WARNING))
        if entertainment_ratio > entertainment_limit:
            exceed = (entertainment_ratio - entertainment_limit) / max(1e-6, 1 - entertainment_limit)
            severity = 3 + int(np.clip(np.ceil(exceed * 2), 0, 2))
            risks.append(
                build_alert(
                    risk_type=RiskType.EXCESSIVE_ENTERTAINMENT,
                    severity=severity,
                    brief_description="娱乐内容消费偏高，影响信息质量",
                    detailed_description=(
                        f"娱乐类占比 {entertainment_ratio:.1%}，超过预警阈值 {entertainment_limit:.1%}。"
                    ),
                    impact_analysis="高娱乐占比会稀释高价值信息摄入。",
                    key_statistics={
                        "entertainment_ratio": entertainment_ratio,
                        "learning_ratio": float(metrics.content_quality.learning_ratio),
                    },
                    suggestions=[
                        "设定娱乐内容每日时长上限",
                        "将学习类内容固定到高注意力时段",
                    ],
                )
            )

        # 时间浪费：非高效时段与碎片化显著
        off_hour_waste_ratio = float(metrics.time_allocation.off_hour_waste_ratio)
        fragmentation_score = float(metrics.time_allocation.fragmentation_score)
        late_night_ent_duration = float(metrics.time_allocation.late_night_entertainment_duration)
        if off_hour_waste_ratio > 0.35 or fragmentation_score > 0.70 or late_night_ent_duration > 1.0:
            severity = 3
            if off_hour_waste_ratio > 0.50 or fragmentation_score > 0.85 or late_night_ent_duration > 2.0:
                severity = 4
            if off_hour_waste_ratio > 0.65 or late_night_ent_duration > 3.0:
                severity = 5

            risks.append(
                build_alert(
                    risk_type=RiskType.TIME_WASTE,
                    severity=severity,
                    brief_description="时间利用效率偏低，存在明显浪费",
                    detailed_description=(
                        f"低效时段占比 {off_hour_waste_ratio:.1%}，"
                        f"碎片化评分 {fragmentation_score:.2f}，"
                        f"深夜娱乐时长约 {late_night_ent_duration:.2f} 小时。"
                    ),
                    impact_analysis="碎片化和深夜浏览会降低专注力并压缩高质量输入时间。",
                    key_statistics={
                        "off_hour_waste_ratio": off_hour_waste_ratio,
                        "fragmentation_score": fragmentation_score,
                        "late_night_entertainment_duration": late_night_ent_duration,
                    },
                    suggestions=[
                        "将高价值阅读安排在白天固定时段",
                        "晚间设置娱乐截止时间，减少睡前连续刷屏",
                    ],
                )
            )

        # 内容单一：多样性分过低
        category_diversity_score = float(metrics.diversity.category_diversity_score)
        if category_diversity_score < 0.35:
            severity = 4 if category_diversity_score < 0.20 else 3
            risks.append(
                build_alert(
                    risk_type=RiskType.CONTENT_MONOTONY,
                    severity=severity,
                    brief_description="内容类别结构单一，输入广度不足",
                    detailed_description=f"类别多样性评分为 {category_diversity_score:.2f}，低于健康建议范围。",
                    impact_analysis="长期单一摄入不利于建立完整的信息结构。",
                    key_statistics={
                        "category_diversity_score": category_diversity_score,
                        "category_count": int(metrics.diversity.category_count),
                    },
                    suggestions=[
                        "按周制定跨类别内容计划（学习/新闻/工具）",
                        "降低单一类别连续浏览时长",
                    ],
                )
            )

        return risks

    def _generate_risk_alert(
        self,
        risk_type: RiskType,
        severity: int,
        evidence: Dict[str, Any]
    ) -> RiskAlert:
        """
        生成单个风险警报

        TODO: 根据风险类型生成描述
        TODO: 提取关键证据
        TODO: 生成针对性建议
        """
        pass

    # ==================== 私有方法：建议生成 ====================

    def _generate_suggestions(
        self,
        metrics: EvaluationMetrics,
        risks: List[RiskAlert]
    ) -> List[str]:
        """
        生成改进建议
        """
        merged: List[str] = []
        seen = set()

        def add_suggestion(text: str) -> None:
            t = str(text).strip()
            if t and t not in seen:
                seen.add(t)
                merged.append(t)

        sorted_risk = sorted(risks, key=lambda r: r.severity, reverse=True)
        for risk in sorted_risk:
            for s in risk.suggestions:
                add_suggestion(s)

        if not merged:
            if metrics.diversity.category_diversity_score < 0.5:
                add_suggestion("增加跨类别信息摄入（学习/新闻/工具），降低单一内容连续浏览。")
            if metrics.sentiment_health.negative_ratio > 0.35:
                add_suggestion("减少高负面信息源暴露，增加中性与积极内容比例。")
            if metrics.content_quality.entertainment_ratio > 0.45:
                add_suggestion("设置娱乐内容时长上限，将高价值内容前置到白天。")
            if metrics.time_allocation.off_hour_waste_ratio > 0.35:
                add_suggestion("减少深夜浏览，固定高效阅读时段并控制碎片化使用。")

        if not merged:
            add_suggestion("当前信息摄取结构总体稳定，建议每周复盘一次并保持多样化输入。")

        return merged

    def _generate_category_suggestions(self, df: pd.DataFrame) -> List[str]:
        """
        生成类别平衡建议

        TODO: 识别缺失或不足的类别
        TODO: 建议增加的内容类型
        """
        pass

    def _generate_time_management_suggestions(self, df: pd.DataFrame) -> List[str]:
        """
        生成时间管理建议

        TODO: 基于时间分配分析
        TODO: 建议调整浏览时段
        TODO: 推荐时间管理策略
        """
        pass

    # ==================== 核心公共方法 ====================

    def evaluate(
        self,
        df: pd.DataFrame,
        time_range: Optional[Tuple[str, str]] = None,
        detailed: bool = True
    ) -> EvaluationReport:
        """
        执行完整评估（主入口）

        参数:
            df: 包含分析结果的 DataFrame
                必需列: title, url, category, sentiment, polarity, similarity
            time_range: 评估时间范围 (start, end)
            detailed: 是否生成详细报告

        返回:
            EvaluationReport: 完整评估报告
        """
        if not self._validate_dataframe(df):
            raise TypeError("数据无效，无法评估")

        processed_df = self._preprocess_data(df)
        processed_df = self._filter_by_time_range(processed_df, time_range=time_range)

        category_diversity_score = self._calculate_category_diversity(df=processed_df)
        content_diversity_score = self._calculate_content_diversity(df=processed_df)
        content_quality_score = self._calculate_content_quality(df=processed_df)
        sentiment_health_score = self._calculate_sentiment_health(df=processed_df)
        time_alloc = self._analyze_time_allocation(df=processed_df)

        category_counts = processed_df["category"].astype(str).str.lower().value_counts()
        sentiment_counts = processed_df['sentiment'].astype(str).str.lower().value_counts()
        polarity_series = pd.to_numeric(processed_df['polarity'], errors='coerce').dropna()


        # 构造 DiversityMetrics
        content_div_details = self._cache.get('content_diversity_details', {})

        category_distribution = {str(k): int(v) for k, v in category_counts.to_dict().items()}
        category_ratios = (category_counts / category_counts.sum()).tolist() if len(category_counts) > 0 else []
        category_entropy = calculate_shannon_entropy(category_ratios)

        if len(category_counts) > 0:
            dominant_category = str(category_counts.index[0])
            dominant_category_ratio = float(category_counts.iloc[0] / category_counts.sum())
        else:
            dominant_category = "other"
            dominant_category_ratio = 0.0

        diversity_metrics = DiversityMetrics(
            category_diversity_score=float(category_diversity_score),
            category_count=int(len(category_counts)),
            category_entropy=float(category_entropy),
            dominant_category=dominant_category,
            dominant_category_ratio=float(dominant_category_ratio),

            content_diversity_score=float(content_diversity_score),
            avg_similarity=float(content_div_details.get("avg_similarity", 0.0)),
            duplicate_ratio=float(content_div_details.get("duplicate_ratio", 0.0)),
            cluster_count=int(content_div_details.get("cluster_count", 0)),

            category_distribution=category_distribution,
            similarity_distribution=list(content_div_details.get("similarity_distribution", [])),
        )

        # 构造 SentimentHealthMetrics
        total_sent = max(len(processed_df), 1)
        positive_ratio = float(sentiment_counts.get("positive", 0) / total_sent)
        negative_ratio = float(sentiment_counts.get("negative", 0) / total_sent)
        neutral_ratio = float(sentiment_counts.get("neutral", 0) / total_sent)

        polarity_mean = float(polarity_series.mean()) if not polarity_series.empty else 0.0
        polarity_std = float(polarity_series.std()) if len(polarity_series) >= 2 else 0.0
        extreme_emotion_count = int((polarity_series.abs() >= 0.8).sum()) if not polarity_series.empty else 0

        sentiment_metrics = SentimentHealthMetrics(
            sentiment_health_score=float(sentiment_health_score),

            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
            neutral_ratio=neutral_ratio,

            polarity_mean=polarity_mean,
            polarity_std=polarity_std,
            extreme_emotion_count=extreme_emotion_count,

            sentiment_distribution={str(k): int(v) for k, v in sentiment_counts.to_dict().items()},
            polarity_values=polarity_series.tolist(),
        )

        # 构造 ContentQualityMetrics
        category_ratio_map = (category_counts / category_counts.sum()).to_dict() if len(category_counts) > 0 else {}

        content_metrics = ContentQualityMetrics(
            content_quality_score=float(content_quality_score),
            weighted_quality_score=float(content_quality_score),

            learning_ratio=float(category_ratio_map.get("learning", 0.0)),
            news_ratio=float(category_ratio_map.get("news", 0.0)),
            tools_ratio=float(category_ratio_map.get("tools", 0.0)),
            entertainment_ratio=float(category_ratio_map.get("entertainment", 0.0)),
            social_ratio=float(category_ratio_map.get("social", 0.0)),
            shopping_ratio=float(category_ratio_map.get("shopping", 0.0)),
            other_ratio=float(category_ratio_map.get("other", 0.0)),

            category_time_distribution={str(k): float(v) for k, v in category_ratio_map.items()},
            category_weights={},
        )

        # 构造 TimeAllocationMetrics
        time_metrics = TimeAllocationMetrics(
            time_allocation_score=float(time_alloc.get("time_allocation_score", 0.0)),
            peak_hour_efficiency=float(time_alloc.get("peak_hour_efficiency", 0.0)),
            off_hour_waste_ratio=float(time_alloc.get("off_hour_waste_ratio", 0.0)),
            fragmentation_score=float(time_alloc.get("fragmentation_score", 0.0)),
            avg_session_duration=float(time_alloc.get("avg_session_duration", 0.0)),
            low_efficiency_duration=float(time_alloc.get("low_efficiency_duration", 0.0)),
            late_night_entertainment_duration=float(time_alloc.get("late_night_entertainment_duration", 0.0)),
            category_time_ratios=dict(time_alloc.get("category_time_ratios", {})),
            hourly_distribution=dict(time_alloc.get("hourly_distribution", {})),
            weekday_distribution=dict(time_alloc.get("weekday_distribution", {})),
        )

        # 拼接为 EvaluationMetrics
        dimension_scores = {
            "diversity": float(diversity_metrics.category_diversity_score),
            "sentiment_health": float(sentiment_metrics.sentiment_health_score),
            "content_quality": float(content_metrics.content_quality_score),
            "time_allocation": float(time_metrics.time_allocation_score),
        }
        overall_score = float(weighted_average(dimension_scores, self.config["weights"]) * 100)

        metrics = EvaluationMetrics(
            diversity=diversity_metrics,
            sentiment_health=sentiment_metrics,
            content_quality=content_metrics,
            time_allocation=time_metrics,
            overall_score=overall_score,
            dimension_weights=self.config["weights"].copy(),
        )

        # 构造 ReportMetadata
        if overall_score >= 90:
            level = HealthLevel.EXCELLENT
        elif overall_score >= 75:
            level = HealthLevel.GOOD
        elif overall_score >= 60:
            level = HealthLevel.FAIR
        elif overall_score >= 40:
            level = HealthLevel.WARNING
        else:
            level = HealthLevel.CRITICAL

        health_status = HealthStatus(
            level=level,
            score=round(overall_score, 2),
            justification=(
                f"综合评分 {overall_score:.1f}，"
                f"多样性 {diversity_metrics.category_diversity_score * 100:.1f}，"
                f"情感健康 {sentiment_metrics.sentiment_health_score * 100:.1f}，"
                f"内容质量 {content_metrics.content_quality_score * 100:.1f}，"
                f"时间分配 {time_metrics.time_allocation_score * 100:.1f}"
            )
        )

        risk_alerts = self._identify_risks(processed_df, metrics)

        if "timestamp" in processed_df.columns and not processed_df["timestamp"].empty:
            start_date = pd.to_datetime(processed_df["timestamp"], errors="coerce").min()
            end_date = pd.to_datetime(processed_df["timestamp"], errors="coerce").max()
        else:
            # 没有时间列时，用当前时间兜底
            start_date = datetime.now()
            end_date = datetime.now()

        if pd.isna(start_date):
            start_date = datetime.now()
        if pd.isna(end_date):
            end_date = datetime.now()

        time_span_days = max((end_date.date() - start_date.date()).days + 1, 1)

        metadata = ReportMetadata(
            start_date=start_date,
            end_date=end_date,
            total_records=int(len(df)),
            valid_records=int(len(processed_df)),
            time_span_days=int(time_span_days),
            generated_at=datetime.now(),
            evaluator_version="0.1.0",
            config_info=self.config.copy()
        )

        # 组装 EvaluationReport
        report = EvaluationReport(
            metadata=metadata,
            health_status=health_status,
            metrics=metrics,
            risk_alerts=risk_alerts,
            detailed_analysis=None if not detailed else None,  # 先占位，后续再填详细分析
            recommendations=Recommendations(),  # 先空，后续接 _generate_suggestions
            trend_analysis=None
        )

        self.last_report = report
        return report

    def quick_evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        快速评估（简化版）
        """
        self._validate_dataframe(df)
        processed_df = self._preprocess_data(df)

        category_counts = processed_df["category"].value_counts()
        category_probs = (category_counts / category_counts.sum()).tolist()
        entropy = calculate_shannon_entropy(category_probs)
        max_entropy = np.log2(len(category_probs)) if len(category_probs) > 1 else 1.0
        diversity_score = float(entropy / max_entropy) if max_entropy > 0 else 0.0
        diversity_score = float(np.clip(diversity_score, 0.0, 1.0))

        negative_ratio = float((processed_df["sentiment"] == "negative").mean())
        sentiment_health_score = float(np.clip(1.0 - negative_ratio, 0.0, 1.0))

        entertainment_ratio = float((processed_df["category"] == "entertainment").mean())
        content_quality_score = float(np.clip(1.0 - entertainment_ratio, 0.0, 1.0))

        avg_similarity = float(processed_df["similarity"].mean())
        time_allocation_score = float(np.clip(1.0 - avg_similarity, 0.0, 1.0))

        dimension_scores = {
            "diversity": diversity_score,
            "sentiment_health": sentiment_health_score,
            "content_quality": content_quality_score,
            "time_allocation": time_allocation_score,
        }
        overall_0_1 = weighted_average(dimension_scores, self.config["weights"])
        overall_score = round(overall_0_1 * 100, 2)

        if overall_score >= 90:
            level = HealthLevel.EXCELLENT.value
        elif overall_score >= 75:
            level = HealthLevel.GOOD.value
        elif overall_score >= 60:
            level = HealthLevel.FAIR.value
        elif overall_score >= 40:
            level = HealthLevel.WARNING.value
        else:
            level = HealthLevel.CRITICAL.value

        return {
            "overall_score": overall_score,
            "health_level": level,
            "dimension_scores": {
                "diversity": round(diversity_score * 100, 2),
                "sentiment_health": round(sentiment_health_score * 100, 2),
                "content_quality": round(content_quality_score * 100, 2),
                "time_allocation": round(time_allocation_score * 100, 2),
            },
            "sample_info": {
                "total_records": int(len(df)),
                "valid_records": int(len(processed_df)),
            },
        }

    # ==================== 批量和对比分析 ====================

    def batch_evaluate(
        self,
        df: pd.DataFrame,
        group_by: str = "date"
    ) -> pd.DataFrame:
        """
        批量评估（按时间段分组）

        TODO: 按指定维度分组
        TODO: 对每组执行评估
        TODO: 返回评估结果 DataFrame
        """
        pass

    def compare_users(
        self,
        user_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        对比多个用户的信息摄取质量

        TODO: 对每个用户执行评估
        TODO: 计算相对排名
        TODO: 生成对比报告
        """
        pass

    # ==================== 可视化支持 ====================

    def get_visualization_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        准备可视化数据

        TODO: 提取关键指标的时间序列
        TODO: 准备分布图数据
        TODO: 生成热力图数据
        """
        pass

    # ==================== 报告导出 ====================

    def export_report(
        self,
        report: EvaluationReport,
        output_path: str,
        format: str = "json"
    ) -> None:
        """
        导出评估报告
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        fmt = format.strip().lower()

        if fmt == "json":
            report.to_json(str(output))
            return

        if fmt in {"markdown", "md"}:
            report.to_markdown(str(output), detailed=True)
            return

        if fmt == "html":
            data = report.to_dict()
            summary = report.get_summary()
            html_content = (
                "<!DOCTYPE html>\n"
                "<html lang='zh-CN'>\n"
                "<head><meta charset='UTF-8'><title>信息摄取质量评估报告</title></head>\n"
                "<body>\n"
                "<h1>信息摄取质量评估报告</h1>\n"
                "<h2>摘要</h2>\n"
                f"<pre>{summary}</pre>\n"
                "<h2>完整数据（JSON）</h2>\n"
                f"<pre>{json.dumps(data, ensure_ascii=False, indent=2)}</pre>\n"
                "</body>\n"
                "</html>\n"
            )
            output.write_text(html_content, encoding="utf-8")
            logger.info(f"HTML 报告已保存到: {output}")
            return

        raise ValueError("不支持的导出格式，请使用 json / markdown(md) / html")

    def generate_summary(self, report: EvaluationReport) -> str:
        """
        生成文字摘要
        """
        return report.get_summary()

    # ==================== 配置管理 ====================

    def update_config(self, config: Dict[str, Any]) -> None:
        """
        更新评估配置
        """
        if not isinstance(config, dict):
            raise TypeError("config 必须是 dict")

         # 先以当前配置为基础；若不存在则用默认
        current = self.config.copy() if isinstance(self.config, dict) else self.get_default_config()

        if "thresholds" not in current or not isinstance(current["thresholds"], dict):
            current["thresholds"] = self.get_default_config()["thresholds"].copy()
        if "weights" not in current or not isinstance(current["weights"], dict):
            current["weights"] = self.get_default_config()["weights"].copy()

        # 1) min_records
        if "min_records" in config:
            min_records = config["min_records"]
            if not isinstance(min_records, int) or min_records <= 0:
                raise ValueError("min_records 必须是正整数")
            current["min_records"] = min_records

        # 2) thresholds
        if "thresholds" in config:
            thresholds = config["thresholds"]
            if not isinstance(thresholds, dict):
                raise TypeError("thresholds 必须是 dict")

            allowed_thresholds = set(self.get_default_config()["thresholds"].keys())
            unknown_keys = set(thresholds.keys()) - allowed_thresholds
            if unknown_keys:
                raise ValueError(f"未知阈值配置项: {sorted(unknown_keys)}")

            for k, v in thresholds.items():
                if not isinstance(v, (int, float)):
                    raise TypeError(f"thresholds['{k}'] 必须是数值")
                v = float(v)
                if v < 0 or v > 1:
                    raise ValueError(f"thresholds['{k}'] 必须在 [0,1] 区间")
                current["thresholds"][k] = v

        # 3) weights
        if "weights" in config:
            weights = config["weights"]
            if not isinstance(weights, dict):
                raise TypeError("weights 必须是 dict")

            required_keys = set(self.DEFAULT_WEIGHTS.keys())
            if set(weights.keys()) != required_keys:
                raise ValueError(f"weights 必须且只能包含: {sorted(required_keys)}")

            cleaned_weights = {}
            for k, v in weights.items():
                if not isinstance(v, (int, float)):
                    raise TypeError(f"weights['{k}'] 必须是数值")
                v = float(v)
                if v < 0:
                    raise ValueError(f"weights['{k}'] 不能为负数")
                cleaned_weights[k] = v

            total = sum(cleaned_weights.values())
            if total <= 0:
                raise ValueError("weights 总和必须大于 0")

            # 自动归一化
            current["weights"] = {k: v / total for k, v in cleaned_weights.items()}

        self.config = current
        logger.info(f"评估配置已更新: {self.config}")

    def get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置

        返回默认阈值和权重
        """
        return {
            "min_records": 5,  # 最低样本量
            "thresholds": {
                "echo_chamber_similarity": self.ECHO_CHAMBER_SIMILARITY_LIMIT,
                "dominant_category_ratio": self.DOMINANT_CATEGORY_RATIO_LIMIT,
                "negative_ratio_warning": self.NEGATIVE_RATIO_WARNING,
                "entertainment_ratio_warning": self.ENTERTAINMENT_RATIO_WARNING,
            },
            'weights' : self.DEFAULT_WEIGHTS.copy()
        }


# ==================== 辅助函数 ====================

def calculate_shannon_entropy(distribution: List[float]) -> float:
    """
    计算香农熵（衡量分布均匀度）
    """
    if not distribution:
        return 0.0

    entropy = 0.0

    for p in distribution:
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy


def normalize_score(value: float, min_val: float, max_val: float) -> float:
    """
    标准化分数到 0-1 区间
    """
    if max_val == min_val:  # 如果最大值和最小值相同
        return 0.0  # 无法归一化，直接返回 0

    normalized = (value - min_val) / (max_val - min_val)

    if normalized < 0:
        return 0.0
    if normalized > 1:
        return 1.0

    return normalized


def weighted_average(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    计算加权平均
    """
    total_weight = sum(weights.values())

    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError("权重之和必须为 1")

    weighted_sum = 0.0

    for key, score in scores.items():
        weight = weights.get(key, 0.0)
        weighted_sum += score * weight

    return weighted_sum



# ==================== 测试代码 ====================
if __name__ == "__main__":
    try:
        # ===== 路径配置 =====
        raw_input_path = "./utils/output/history_data.csv"   # 原始浏览记录
        result_csv_path = "./utils/output/result.csv"        # 构造后的完整结果
        report_json = "cache/evaluation_report.json"
        report_md = "cache/evaluation_report.md"
        report_html = "cache/evaluation_report.html"

        # ===== 读取原始数据 =====
        raw_df = pd.read_csv(raw_input_path)
        logger.info(f"已加载原始数据: {raw_input_path}, 共 {len(raw_df)} 条")

        # ===== 初始化三个分析器 =====
        sentiment_analyzer = SentimentAnalyzer(model_path='./models/sentiment_model_bert', use_bert=True)  # 可按需传 model_path/use_bert
        content_classifier = ContentClassifier(model_path="./models/classifier_model.pkl")
        similarity_analyzer = SimilarityAnalyzer()

        # ===== 构造 category =====
        df1 = content_classifier.batch_predict(raw_df)

        # ===== 构造 sentiment / polarity =====
        df2 = sentiment_analyzer.batch_predict(df1, text_column="title", include_emotions=False)

        # ===== 构造 similarity =====
        df3 = similarity_analyzer.batch_calculate_similarity(df2, text_column="title")
        if "similarity" not in df3.columns:
            if "similarity_to_previous" in df3.columns:
                df3["similarity"] = df3["similarity_to_previous"]
            else:
                df3["similarity"] = 0.0

        # ===== 保存构造后的 result.csv =====
        Path(result_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df3.to_csv(result_csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"已生成完整结果文件: {result_csv_path}")

        # ===== 评估 =====
        evaluator = InformationQualityEvaluator(
            sentiment_analyzer=sentiment_analyzer,
            content_classifier=content_classifier,
            similarity_analyzer=similarity_analyzer
        )
        report = evaluator.evaluate(df3, detailed=True)

        # ===== 输出摘要 导出报告 =====
        print("\n" + "=" * 60)
        print(evaluator.generate_summary(report))
        print("=" * 60 + "\n")

        evaluator.export_report(report, report_json, format="json")
        evaluator.export_report(report, report_md, format="markdown")
        evaluator.export_report(report, report_html, format="html")

        logger.info("完整流程执行完成：构造 result.csv + 评估 + 报告导出")

    except FileNotFoundError as e:
        logger.error(f"文件不存在: {e}")
    except Exception as e:
        logger.exception(f"执行失败: {e}")