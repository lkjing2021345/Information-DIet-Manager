"""
把命令行参数原样透传给 src.lsj.src.main
用法示例：
python scripts/run_analysis.py --mode analyze --input_file data.json
"""

from src.lsj.src.main import main

if __name__ == "__main__":
    main()
