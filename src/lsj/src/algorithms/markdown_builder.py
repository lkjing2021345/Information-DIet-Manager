# -*- coding: utf-8 -*-
"""
Markdown 构建器模块

提供 Markdown 文档构建和评估报告生成功能
"""
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
from datetime import datetime
from urllib.parse import quote

# 避免循环导入
if TYPE_CHECKING:
    from lsj.src.algorithms.evaluator import (
        EvaluationReport, HealthLevel, RiskAlert,
        ActionableRecommendation, DiversityMetrics,
        SentimentHealthMetrics, ContentQualityMetrics,
        TimeAllocationMetrics
    )


class MarkdownBuilder:
    """Markdown 构建器"""

    def __init__(self):
        self.lines: List[str] = []

    def add_line(self, text: str = "") -> "MarkdownBuilder":
        """添加一行"""
        self.lines.append(text)
        return self

    def add_heading(self, text: str, level: int = 1) -> "MarkdownBuilder":
        """添加标题"""
        self.lines.append(f"{'#' * level} {text}")
        self.lines.append("")
        return self

    def add_paragraph(self, text: str) -> "MarkdownBuilder":
        """添加段落"""
        self.lines.append(text)
        self.lines.append("")
        return self

    def add_bold(self, text: str) -> str:
        """粗体文本"""
        return f"**{text}**"

    def add_italic(self, text: str) -> str:
        """斜体文本"""
        return f"*{text}*"

    def add_code(self, text: str) -> str:
        """行内代码"""
        return f"`{text}`"

    def add_list_item(self, text: str, level: int = 0) -> "MarkdownBuilder":
        """添加列表项"""
        indent = "  " * level
        self.lines.append(f"{indent}- {text}")
        return self

    def add_numbered_item(self, text: str, number: int = 1) -> "MarkdownBuilder":
        """添加编号列表项"""
        self.lines.append(f"{number}. {text}")
        return self

    def add_table(self, headers: List[str], rows: List[List[Any]]) -> "MarkdownBuilder":
        """添加表格"""
        # 表头
        self.lines.append("| " + " | ".join(headers) + " |")
        # 分隔线
        self.lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        # 数据行
        for row in rows:
            self.lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        self.lines.append("")
        return self

    def add_blockquote(self, text: str) -> "MarkdownBuilder":
        """添加引用块"""
        lines = text.split("\n")
        for line in lines:
            self.lines.append(f"> {line}")
        self.lines.append("")
        return self

    def add_code_block(self, code: str, language: str = "") -> "MarkdownBuilder":
        """添加代码块"""
        self.lines.append(f"```{language}")
        self.lines.append(code)
        self.lines.append("```")
        self.lines.append("")
        return self

    def add_horizontal_rule(self) -> "MarkdownBuilder":
        """添加分隔线"""
        self.lines.append("---")
        self.lines.append("")
        return self

    def add_badge(self, label: str, value: str, color: str = "blue") -> str:
        """生成徽章（GitHub风格）"""
        # 对 label/value 做 URL 编码，避免中文、空格导致图片链接失效
        safe_label = quote(str(label))
        safe_value = quote(str(value))
        safe_color = quote(str(color))
        return f"![{label}](https://img.shields.io/badge/{safe_label}-{safe_value}-{safe_color})"

    def add_progress_bar(self, value: float, max_value: float = 100, width: int = 20) -> str:
        """生成进度条"""
        # 防止 max_value <= 0 导致除零错误
        if max_value <= 0:
            max_value = 100
        # 将 value 约束到 [0, max_value]，防止进度条越界
        safe_value = max(0.0, min(float(value), float(max_value)))
        # 计算百分比
        percentage = safe_value / max_value
        # 计算已填充和未填充长度
        filled = int(percentage * width)
        empty = width - filled
        # 构建条形图
        bar = "█" * filled + "░" * empty
        # 展示原始值，更直观（如果你想展示裁剪值，可换成 safe_value）
        return f"`{bar}` {value:.1f}/{max_value}"

    def build(self) -> str:
        """构建最终的 Markdown 文本"""
        return "\n".join(self.lines)

    def save(self, filepath: str) -> None:
        """保存到文件"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.build())


class ReportMarkdownGenerator:
    """评估报告 Markdown 生成器"""

    def __init__(self):
        self.md = MarkdownBuilder()

    def generate(self, report: 'EvaluationReport', detailed: bool = True) -> str:
        """
        生成完整的 Markdown 报告

        参数:
            report: 评估报告对象
            detailed: 是否包含详细分析

        返回:
            str: Markdown 文本
        """
        self.md = MarkdownBuilder()  # 重置

        # 1. 标题和元信息
        self._add_header(report)

        # 2. 执行摘要
        self._add_executive_summary(report)

        # 3. 健康状态
        self._add_health_status(report)

        # 4. 核心指标
        self._add_metrics(report)

        # 5. 风险警报
        self._add_risk_alerts(report)

        # 6. 详细分析（可选）
        if detailed and report.detailed_analysis:
            self._add_detailed_analysis(report)

        # 7. 改进建议
        self._add_recommendations(report)

        # 8. 趋势分析（如果有）
        if report.trend_analysis:
            self._add_trend_analysis(report)

        # 9. 附录
        self._add_appendix(report)

        return self.md.build()

    def save(self, report: 'EvaluationReport', filepath: str, detailed: bool = True) -> None:
        """
        生成并保存 Markdown 报告

        参数:
            report: 评估报告对象
            filepath: 保存路径
            detailed: 是否包含详细分析
        """
        content = self.generate(report, detailed)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    # ==================== 私有方法：各部分生成 ====================

    def _add_header(self, report: 'EvaluationReport') -> None:
        """添加标题和元信息"""
        self.md.add_heading("📊 信息摄取质量评估报告", level=1)
        self.md.add_paragraph(
            f"**评估时间范围**: {report.metadata.start_date.strftime('%Y-%m-%d')} "
            f"至 {report.metadata.end_date.strftime('%Y-%m-%d')} "
            f"（共 {report.metadata.time_span_days} 天）"
        )
        # 防止 total_records 为 0 时出现除零错误
        total_records = report.metadata.total_records or 0
        valid_records = report.metadata.valid_records or 0
        valid_rate = (valid_records / total_records * 100) if total_records > 0 else 0.0
        self.md.add_paragraph(
            f"**数据统计**: 总记录 {total_records} 条，"
            f"有效记录 {valid_records} 条 "
            f"（有效率 {valid_rate:.1f}%）"
        )
        self.md.add_paragraph(
            f"**生成时间**: {report.metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.md.add_horizontal_rule()

    def _add_appendix(self, report: 'EvaluationReport') -> None:
        """添加附录"""
        self.md.add_heading("📎 附录", level=2)
        self.md.add_heading("评分标准", level=3)
        self.md.add_paragraph("**健康等级划分**:")
        self.md.add_list_item("优秀: 90-100 分")
        self.md.add_list_item("良好: 75-89 分")
        self.md.add_list_item("一般: 60-74 分")
        self.md.add_list_item("警告: 40-59 分")
        self.md.add_list_item("危险: 0-39 分")
        self.md.add_line()
        self.md.add_paragraph("**维度权重配置**:")
        for dimension, weight in report.metrics.dimension_weights.items():
            self.md.add_list_item(f"{dimension}: {weight * 100:.0f}%")
        self.md.add_line()
        # 增加一个方法说明，便于读者理解“分数不是医学诊断”
        self.md.add_paragraph("**说明**:")
        self.md.add_list_item("本报告用于行为模式与信息摄取结构分析，不构成医疗或心理诊断建议。")
        self.md.add_list_item("建议结合长期趋势观察，不建议仅凭单周期结果做重大决策。")
        self.md.add_line()
        self.md.add_horizontal_rule()
        self.md.add_paragraph(f"_报告生成完成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")

    def _add_executive_summary(self, report: 'EvaluationReport') -> None:
        """添加执行摘要"""
        self.md.add_heading("📋 执行摘要", level=2)

        # 健康等级徽章
        level_color = self._get_health_level_color(report.health_status.level)
        badge = self.md.add_badge("健康等级", report.health_status.level.value, level_color)
        self.md.add_paragraph(badge)

        # 综合得分
        score_bar = self.md.add_progress_bar(report.health_status.score, 100)
        self.md.add_paragraph(f"**综合得分**: {score_bar}")

        # 关键发现
        self.md.add_paragraph("**关键发现**:")
        findings = self._generate_key_findings(report)
        for finding in findings:
            self.md.add_list_item(finding)

        self.md.add_line()
        self.md.add_horizontal_rule()

    def _add_health_status(self, report: 'EvaluationReport') -> None:
        """添加健康状态"""
        self.md.add_heading("🏥 健康状态", level=2)

        level_emoji = self._get_health_level_emoji(report.health_status.level)
        self.md.add_paragraph(
            f"{level_emoji} **等级**: {report.health_status.level.value} | "
            f"**分数**: {report.health_status.score:.1f}/100"
        )

        self.md.add_blockquote(report.health_status.justification)
        self.md.add_horizontal_rule()

    def _add_metrics(self, report: 'EvaluationReport') -> None:
        """添加核心指标"""
        self.md.add_heading("📈 核心指标", level=2)
        self.md.add_heading("指标总览", level=3)

        headers = ["维度", "得分", "状态", "说明"]
        rows = [
            [
                "多样性",
                f"{report.metrics.diversity.category_diversity_score * 100:.1f}",
                self._get_score_status(report.metrics.diversity.category_diversity_score),
                f"类别数: {report.metrics.diversity.category_count}"
            ],
            [
                "情感健康",
                f"{report.metrics.sentiment_health.sentiment_health_score * 100:.1f}",
                self._get_score_status(report.metrics.sentiment_health.sentiment_health_score),
                f"积极比例: {report.metrics.sentiment_health.positive_ratio * 100:.1f}%"
            ],
            [
                "内容质量",
                f"{report.metrics.content_quality.content_quality_score * 100:.1f}",
                self._get_score_status(report.metrics.content_quality.content_quality_score),
                f"学习占比: {report.metrics.content_quality.learning_ratio * 100:.1f}%"
            ],
            [
                "时间分配",
                f"{report.metrics.time_allocation.time_allocation_score * 100:.1f}",
                self._get_score_status(report.metrics.time_allocation.time_allocation_score),
                f"碎片化: {report.metrics.time_allocation.fragmentation_score * 100:.1f}%"
            ]
        ]

        self.md.add_table(headers, rows)

        # 详细指标
        self._add_diversity_metrics(report.metrics.diversity)
        self._add_sentiment_metrics(report.metrics.sentiment_health)
        self._add_content_quality_metrics(report.metrics.content_quality)
        self._add_time_allocation_metrics(report.metrics.time_allocation)

        self.md.add_horizontal_rule()

    def _add_diversity_metrics(self, diversity: 'DiversityMetrics') -> None:
        """添加多样性指标详情"""
        self.md.add_heading("🎨 多样性分析", level=3)

        self.md.add_paragraph("**类别分布**:")

        if diversity.category_distribution:
            sorted_categories = sorted(
                diversity.category_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            )

            total = sum(diversity.category_distribution.values())
            for category, count in sorted_categories:
                ratio = count / total * 100 if total > 0 else 0
                bar = self.md.add_progress_bar(ratio, 100, width=15)
                self.md.add_list_item(f"{category}: {count} 条 ({ratio:.1f}%) {bar}")

        self.md.add_line()

        self.md.add_paragraph("**关键指标**:")
        self.md.add_list_item(
            f"类别多样性: {diversity.category_diversity_score:.2f} (香农熵: {diversity.category_entropy:.2f})")
        self.md.add_list_item(f"内容多样性: {diversity.content_diversity_score:.2f}")
        self.md.add_list_item(f"平均相似度: {diversity.avg_similarity:.2f}")
        self.md.add_list_item(f"重复内容比例: {diversity.duplicate_ratio * 100:.1f}%")
        self.md.add_list_item(f"聚类数量: {diversity.cluster_count}")

        self.md.add_line()

    def _add_sentiment_metrics(self, sentiment: 'SentimentHealthMetrics') -> None:
        """添加情感健康指标详情"""
        self.md.add_heading("😊 情感健康分析", level=3)

        self.md.add_paragraph("**情感分布**:")

        sentiments = [
            ("积极", sentiment.positive_ratio, "🟢"),
            ("中性", sentiment.neutral_ratio, "🟡"),
            ("消极", sentiment.negative_ratio, "🔴")
        ]

        for name, ratio, emoji in sentiments:
            bar = self.md.add_progress_bar(ratio * 100, 100, width=15)
            self.md.add_list_item(f"{emoji} {name}: {ratio * 100:.1f}% {bar}")

        self.md.add_line()

        self.md.add_paragraph("**情感极性**:")
        self.md.add_list_item(f"平均极性: {sentiment.polarity_mean:.3f} (标准差: {sentiment.polarity_std:.3f})")
        self.md.add_list_item(f"极端情绪事件: {sentiment.extreme_emotion_count} 次")

        if sentiment.emotion_distribution:
            self.md.add_line()
            self.md.add_paragraph("**具体情绪分布**:")

            sorted_emotions = sorted(
                sentiment.emotion_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for emotion, count in sorted_emotions[:5]:
                self.md.add_list_item(f"{emotion}: {count} 次")

        self.md.add_line()

    def _add_content_quality_metrics(self, quality: 'ContentQualityMetrics') -> None:
        """添加内容质量指标详情"""
        self.md.add_heading("⭐ 内容质量分析", level=3)

        self.md.add_paragraph("**类别占比**:")

        categories = [
            ("学习", quality.learning_ratio, "📚"),
            ("新闻", quality.news_ratio, "📰"),
            ("工具", quality.tools_ratio, "🔧"),
            ("娱乐", quality.entertainment_ratio, "🎮"),
            ("社交", quality.social_ratio, "💬"),
            ("购物", quality.shopping_ratio, "🛒"),
            ("其他", quality.other_ratio, "📦")
        ]

        for name, ratio, emoji in categories:
            if ratio > 0:
                bar = self.md.add_progress_bar(ratio * 100, 100, width=15)
                self.md.add_list_item(f"{emoji} {name}: {ratio * 100:.1f}% {bar}")

        self.md.add_line()

        self.md.add_paragraph("**质量评分**:")
        self.md.add_list_item(f"基础质量分: {quality.content_quality_score * 100:.1f}/100")
        self.md.add_list_item(f"加权质量分: {quality.weighted_quality_score * 100:.1f}/100")

        self.md.add_line()

    def _add_time_allocation_metrics(self, time_alloc: 'TimeAllocationMetrics') -> None:
        """添加时间分配指标详情"""
        self.md.add_heading("⏰ 时间分配分析", level=3)

        self.md.add_paragraph(f"**时间分配合理性**: {time_alloc.time_allocation_score * 100:.1f}/100")
        self.md.add_line()

        self.md.add_paragraph("**关键指标**:")
        self.md.add_list_item(f"平均会话时长: {time_alloc.avg_session_duration:.1f} 分钟")
        self.md.add_list_item(f"碎片化程度: {time_alloc.fragmentation_score * 100:.1f}%")
        self.md.add_list_item(f"高峰时段效率: {time_alloc.peak_hour_efficiency * 100:.1f}%")
        self.md.add_list_item(f"低效浏览时长: {time_alloc.low_efficiency_duration:.1f} 小时")
        self.md.add_list_item(f"深夜娱乐时长: {time_alloc.late_night_entertainment_duration:.1f} 小时")

        self.md.add_line()

    def _add_risk_alerts(self, report: 'EvaluationReport') -> None:
        """添加风险警报"""
        self.md.add_heading("⚠️ 风险警报", level=2)

        if not report.risk_alerts:
            self.md.add_paragraph("✅ 未检测到显著风险，继续保持！")
            self.md.add_horizontal_rule()
            return

        # 按严重程度分组
        critical_risks = [r for r in report.risk_alerts if r.severity >= 4]
        warning_risks = [r for r in report.risk_alerts if 2 <= r.severity < 4]
        info_risks = [r for r in report.risk_alerts if r.severity < 2]

        if critical_risks:
            self.md.add_heading("🚨 严重风险", level=3)
            for risk in critical_risks:
                self._add_risk_detail(risk)

        if warning_risks:
            self.md.add_heading("⚠️ 需要注意", level=3)
            for risk in warning_risks:
                self._add_risk_detail(risk)

        if info_risks:
            self.md.add_heading("ℹ️ 改进建议", level=3)
            for risk in info_risks:
                self._add_risk_detail(risk)

        self.md.add_horizontal_rule()

    def _add_risk_detail(self, risk: 'RiskAlert') -> None:
        """添加单个风险详情"""
        severity_emoji = "🔴" if risk.severity >= 4 else "🟡" if risk.severity >= 2 else "🟢"
        self.md.add_heading(
            f"{severity_emoji} {risk.risk_type.value} (严重度: {risk.severity}/5)",
            level=4
        )

        self.md.add_paragraph(f"**问题**: {risk.brief_description}")
        self.md.add_blockquote(risk.detailed_description)

        if risk.evidence.key_statistics:
            self.md.add_paragraph("**关键数据**:")
            for key, value in risk.evidence.key_statistics.items():
                if isinstance(value, float):
                    self.md.add_list_item(f"{key}: {value:.2f}")
                else:
                    self.md.add_list_item(f"{key}: {value}")
            self.md.add_line()

        self.md.add_paragraph(f"**影响**: {risk.impact_analysis}")

        if risk.suggestions:
            self.md.add_paragraph("**建议措施**:")
            for i, suggestion in enumerate(risk.suggestions, 1):
                self.md.add_numbered_item(suggestion, i)
            self.md.add_line()

        self.md.add_line()

    def _add_detailed_analysis(self, report: 'EvaluationReport') -> None:
        """添加详细分析"""
        self.md.add_heading("🔍 详细分析", level=2)

        if report.detailed_analysis.category_analysis:
            self._add_category_analysis_detail(report.detailed_analysis.category_analysis)

        if report.detailed_analysis.sentiment_analysis:
            self._add_sentiment_analysis_detail(report.detailed_analysis.sentiment_analysis)

        if report.detailed_analysis.time_pattern_analysis:
            self._add_time_pattern_detail(report.detailed_analysis.time_pattern_analysis)

        self.md.add_horizontal_rule()

    def _add_category_analysis_detail(self, analysis) -> None:
        """添加类别分析详情"""
        self.md.add_heading("类别深度分析", level=3)

        if analysis.distribution_table:
            headers = ["类别", "数量", "占比", "时长(小时)"]
            rows = []

            for category, stats in analysis.distribution_table.items():
                rows.append([
                    category,
                    stats.get('count', 0),
                    f"{stats.get('ratio', 0) * 100:.1f}%",
                    f"{stats.get('duration', 0):.1f}"
                ])

            self.md.add_table(headers, rows)

        self.md.add_line()

    def _add_sentiment_analysis_detail(self, analysis) -> None:
        """添加情感分析详情"""
        self.md.add_heading("情感深度分析", level=3)

        if analysis.top_positive_content:
            self.md.add_paragraph("**最积极的内容** (Top 5):")
            for i, content in enumerate(analysis.top_positive_content[:5], 1):
                title = content.get('title', '未知')
                polarity = content.get('polarity', 0)
                self.md.add_numbered_item(f"{title} (极性: {polarity:.2f})", i)
            self.md.add_line()

        if analysis.top_negative_content:
            self.md.add_paragraph("**最消极的内容** (Top 5):")
            for i, content in enumerate(analysis.top_negative_content[:5], 1):
                title = content.get('title', '未知')
                polarity = content.get('polarity', 0)
                self.md.add_numbered_item(f"{title} (极性: {polarity:.2f})", i)
            self.md.add_line()

        self.md.add_line()

    def _add_time_pattern_detail(self, analysis) -> None:
        """添加时间模式详情"""
        self.md.add_heading("时间模式分析", level=3)

        if analysis.peak_hours:
            self.md.add_paragraph("**浏览高峰时段**:")
            peak_hours_str = ", ".join([f"{h}:00" for h in analysis.peak_hours])
            self.md.add_list_item(peak_hours_str)
            self.md.add_line()

        if analysis.weekday_vs_weekend:
            self.md.add_paragraph("**工作日 vs 周末对比**:")

            weekday = analysis.weekday_vs_weekend.get('weekday', {})
            weekend = analysis.weekday_vs_weekend.get('weekend', {})

            self.md.add_list_item(f"工作日平均: {weekday.get('avg_count', 0):.1f} 条/天")
            self.md.add_list_item(f"周末平均: {weekend.get('avg_count', 0):.1f} 条/天")
            self.md.add_line()

        self.md.add_line()

    def _add_recommendations(self, report: 'EvaluationReport') -> None:
        """添加改进建议"""
        self.md.add_heading("💡 改进建议", level=2)

        if report.recommendations.urgent_recommendations:
            self.md.add_heading("🔴 紧急建议（立即改进）", level=3)
            for i, rec in enumerate(report.recommendations.urgent_recommendations, 1):
                self._add_recommendation_detail(rec, i)

        if report.recommendations.important_recommendations:
            self.md.add_heading("🟡 重要建议（近期改进）", level=3)
            for i, rec in enumerate(report.recommendations.important_recommendations, 1):
                self._add_recommendation_detail(rec, i)

        if report.recommendations.normal_recommendations:
            self.md.add_heading("🟢 一般建议（长期优化）", level=3)
            for i, rec in enumerate(report.recommendations.normal_recommendations, 1):
                self._add_recommendation_detail(rec, i)

        self.md.add_horizontal_rule()

    def _add_recommendation_detail(self, rec: 'ActionableRecommendation', number: int) -> None:
        """添加单个建议详情"""
        difficulty_emoji = {"容易": "🟢", "中等": "🟡", "困难": "🔴"}.get(rec.difficulty.value, "⚪")

        self.md.add_paragraph(
            f"**{number}. {rec.action}** {difficulty_emoji} ({rec.difficulty.value})"
        )

        self.md.add_list_item(f"原因: {rec.reason}", level=1)
        self.md.add_list_item(f"预期改善: {rec.expected_improvement * 100:.0f}%", level=1)
        self.md.add_line()

    def _add_trend_analysis(self, report: 'EvaluationReport') -> None:
        """添加趋势分析"""
        self.md.add_heading("📊 趋势分析", level=2)

        trend = report.trend_analysis

        self.md.add_heading("历史对比", level=3)
        self.md.add_paragraph(f"**对比周期**: {trend.historical_comparison.comparison_period}")

        if trend.historical_comparison.metric_changes:
            self.md.add_paragraph("**指标变化**:")

            for metric, change in trend.historical_comparison.metric_changes.items():
                change_emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                change_text = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"
                self.md.add_list_item(f"{change_emoji} {metric}: {change_text}")

            self.md.add_line()

        if trend.historical_comparison.improvement_trends:
            self.md.add_paragraph("**改善趋势** ✅:")
            for item in trend.historical_comparison.improvement_trends:
                self.md.add_list_item(item)
            self.md.add_line()

        if trend.historical_comparison.deterioration_trends:
            self.md.add_paragraph("**恶化趋势** ⚠️:")
            for item in trend.historical_comparison.deterioration_trends:
                self.md.add_list_item(item)
            self.md.add_line()

        if trend.milestones:
            self.md.add_heading("里程碑", level=3)

            if trend.milestones.best_performance_date:
                self.md.add_list_item(
                    f"🏆 最佳表现: {trend.milestones.best_performance_date.strftime('%Y-%m-%d')} "
                    f"(得分: {trend.milestones.best_performance_score:.1f})"
                )

            if trend.milestones.worst_performance_date:
                self.md.add_list_item(
                    f"⚠️ 最差表现: {trend.milestones.worst_performance_date.strftime('%Y-%m-%d')} "
                    f"(得分: {trend.milestones.worst_performance_score:.1f})"
                )

            self.md.add_line()

        self.md.add_horizontal_rule()

    def _get_health_level_color(self, level) -> str:
        """
        将健康等级映射为徽章颜色
        支持 enum.value / 字符串，两种输入都能兜底处理。
        """
        # 尽量取 value（若是枚举），否则转字符串
        level_text = str(getattr(level, "value", level)).strip().lower()
        # 中英混合关键词映射，尽可能鲁棒
        if any(k in level_text for k in ["优秀", "excellent", "great"]):
            return "brightgreen"
        if any(k in level_text for k in ["良好", "good"]):
            return "green"
        if any(k in level_text for k in ["一般", "average", "fair"]):
            return "yellow"
        if any(k in level_text for k in ["警告", "warning", "caution"]):
            return "orange"
        if any(k in level_text for k in ["危险", "risk", "danger", "critical"]):
            return "red"
        # 未知等级兜底颜色
        return "lightgrey"

    def _get_health_level_emoji(self, level) -> str:
        """将健康等级映射为 emoji"""
        level_text = str(getattr(level, "value", level)).strip().lower()
        if any(k in level_text for k in ["优秀", "excellent", "great"]):
            return "🟢"
        if any(k in level_text for k in ["良好", "good"]):
            return "✅"
        if any(k in level_text for k in ["一般", "average", "fair"]):
            return "🟡"
        if any(k in level_text for k in ["警告", "warning", "caution"]):
            return "⚠️"
        if any(k in level_text for k in ["危险", "risk", "danger", "critical"]):
            return "🔴"
        return "⚪"

    def _get_score_status(self, score: float) -> str:
        """
        根据 0~1 分数返回状态文案
        例如：0.82 -> 良好 ✅
        """
        # 将输入尽量转换成 float，异常则按 0 处理
        try:
            s = float(score)
        except Exception:
            s = 0.0
        # 边界保护
        s = max(0.0, min(1.0, s))
        if s >= 0.90:
            return "优秀 🟢"
        if s >= 0.75:
            return "良好 ✅"
        if s >= 0.60:
            return "一般 🟡"
        if s >= 0.40:
            return "警告 ⚠️"
        return "危险 🔴"

    def _generate_key_findings(self, report: 'EvaluationReport') -> List[str]:
        """
        生成执行摘要中的“关键发现”
        规则尽量简单直观，方便后续维护与调参。
        """
        findings: List[str] = []
        # 1) 总体结论
        findings.append(
            f"综合健康等级为「{report.health_status.level.value}」，总体得分 {report.health_status.score:.1f}/100。"
        )
        # 2) 维度得分中最好/最弱
        dimension_scores = {
            "多样性": report.metrics.diversity.category_diversity_score,
            "情感健康": report.metrics.sentiment_health.sentiment_health_score,
            "内容质量": report.metrics.content_quality.content_quality_score,
            "时间分配": report.metrics.time_allocation.time_allocation_score,
        }
        best_dim = max(dimension_scores, key=dimension_scores.get)
        worst_dim = min(dimension_scores, key=dimension_scores.get)
        findings.append(
            f"表现最佳维度为「{best_dim}」({dimension_scores[best_dim] * 100:.1f}/100)。"
        )
        findings.append(
            f"相对薄弱维度为「{worst_dim}」({dimension_scores[worst_dim] * 100:.1f}/100)，建议优先优化。"
        )
        # 3) 风险摘要
        risk_count = len(report.risk_alerts) if report.risk_alerts else 0
        critical_count = len([r for r in report.risk_alerts if r.severity >= 4]) if report.risk_alerts else 0
        if risk_count == 0:
            findings.append("未发现显著风险项，当前信息摄取行为整体稳定。")
        else:
            findings.append(f"检测到 {risk_count} 项风险，其中严重风险 {critical_count} 项。")
        # 4) 建议摘要
        urgent_count = len(report.recommendations.urgent_recommendations) \
            if report.recommendations and report.recommendations.urgent_recommendations else 0
        if urgent_count > 0:
            findings.append(f"建议优先执行 {urgent_count} 条紧急改进措施，以快速降低主要风险。")
        return findings