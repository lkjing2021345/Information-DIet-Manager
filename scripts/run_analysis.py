"""
把命令行参数原样透传给 src.lsj.src.main
用法示例：
python scripts/run_analysis.py --mode analyze --input_file data.json
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lsj.src.main import cli_main, main

if __name__ == "__main__":
    sys.exit(cli_main())
