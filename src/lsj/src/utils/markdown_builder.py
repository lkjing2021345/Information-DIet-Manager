# -*- coding: utf-8 -*-
from typing import List, Dict, Any
from pathlib import Path
import textwrap

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
        return f"![{label}](https://img.shields.io/badge/{label}-{value}-{color})"

    def add_progress_bar(self, value: float, max_value: float = 100, width: int = 20) -> str:
        """生成进度条"""
        percentage = value / max_value
        filled = int(percentage * width)
        empty = width - filled
        bar = "█" * filled + "░" * empty
        return f"`{bar}` {value:.1f}/{max_value}"

    def build(self) -> str:
        """构建最终的 Markdown 文本"""
        return "\n".join(self.lines)

    def save(self, filepath: str) -> None:
        """保存到文件"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.build())
