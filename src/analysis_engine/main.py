"""
统一分析入口（语义化命名）：
- 新入口：python -m src.analysis_engine.main ...
- 实际实现仍在 src.lsj.src.main，功能不变
"""

from src.lsj.src.main import main  # 复用原逻辑


if __name__ == "__main__":
    main()
