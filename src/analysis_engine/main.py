"""
统一分析入口（语义化命名）：
- 新入口：python -m src.analysis_engine.main ...
- 实际实现仍在 src.lsj.src.main，功能不变
"""

import sys

from src.lsj.src.main import cli_main, main  # 保留直接调用 main 的兼容接口


if __name__ == "__main__":
    sys.exit(cli_main())
