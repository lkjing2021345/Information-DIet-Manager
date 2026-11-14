# 分析命令行入口启动修复（2026-10-03）

本轮只修复入口路径和可选依赖的加载时机，不修改分类、情感、相似度、评分、输入格式或报告格式，也不把启动与模型替身测试视为真实推理验收。

## 复现与修复

README 中的 `python scripts/run_analysis.py --help` 在不借助 `PYTHONPATH` 的环境中原先报 `ModuleNotFoundError: No module named 'src'`。脚本入口现在与已有后端启动器采用相同方式，根据自身文件位置加入仓库根路径，支持从仓库根调用，也支持从其他工作目录通过脚本绝对路径调用。

原 `src/lsj/src/main.py` 在解析参数前就导入 pandas 和算法模块，因此仅修复仓库根路径仍不能让帮助命令脱离可选分析环境。它还将 `algorithms` 目录加入搜索路径后使用 `algorithms.*` 包名，入口启动方式不同会改变这些导入是否可解析。

现在该模块顶层只有标准库依赖。`build_arg_parser()` 保留唯一一份参数定义，帮助和 argparse 非法参数检查先完成。pandas 在实际 CSV/DataFrame 辅助函数中局部导入；这些函数仍能被其他代码直接调用，不要求先运行 `main()`。算法类只在实际全流程或评估函数中导入，并根据当前文件位置使用与后端 API 一致的算法目录及 sibling 导入名称。

以下入口的帮助及参数解析保持可用：

- 仓库根：`python scripts/run_analysis.py --help`。
- 任意工作目录：通过 `scripts/run_analysis.py` 的绝对路径运行。
- 历史脚本：通过 `src/lsj/src/main.py` 的绝对路径运行。
- 仓库根旧模块入口：`python -m src.lsj.src.main --help`。
- 仓库根语义别名：`python -m src.analysis_engine.main --help`。

模块形式仍遵循 Python 的模块搜索规则；本轮没有安装全局命令，也不承诺从任意目录直接运行 `-m src...`。默认模式、参数名称、命令行与输入 `options` 的优先关系及输出格式没有改变。

## 验证

新增 `src/lsj/tests/test_cli_entrypoints.py` 的 24 项回归通过：

- 五种入口位置/形式分别以真实子进程运行帮助和非法模式，使用 `-E -B -S` 排除 `PYTHONPATH`、字节码写入和第三方 site-packages 的影响。
- 同样的五种入口以导入拦截器运行 `-h` 和非法整数参数，确认不会尝试导入 pandas、NumPy、模型模块、日志工具或 Hugging Face 依赖。临时应用数据与缓存目录没有被创建。
- 真实 pandas 验证 JSON、CSV、DataFrame 辅助函数能直接使用；源码位置解析确认算法导入指向本仓库文件，不执行模型代码。
- 两种模式使用合成模型类验证调用顺序、模型路径参数、批大小、情绪开关、详细报告与 Markdown 导出接线。

测试中的 `-X utf8` 仅用于稳定捕获包含中文的帮助输出。实际推理和训练仍需要独立安装并验收可选模型环境；本轮没有下载、加载或推理真实模型，也没有验证评分有效性。完整回归和对应提交的 CI 结果另行记录。
