# 分析运行日志的异常内容保护

## 证据与修复范围

`POST /analyze/run_full` 的真实调用链为 `_run_full_analysis_snapshot` → `_run_lsj_pipeline` → `_execute_lsj_pipeline`，之后依次进入分类、情感、相似度批处理及评估。API 已将分析失败转换为安全响应和任务错误，但底层算法可能先把原异常写入文件和控制台。

在临时合成数据库上，通过依赖替身让 cntext 或分词函数抛出含其输入文本的异常，真实算法捕获分支和真实 `FileHandler` 将正文、空正文回退标题写入了临时日志。API 响应和任务错误记录没有这些标记。这是对可达异常处理的确定性故障注入，不证明真实第三方库每次出错都会包含源文，也不是日常成功路径直接记录正文的证据。

本批仅修改三处已复现日志：

- `sentiment.py` 的 `CntextSentimentBackend.analyze_score()` 异常分支。
- `sentiment.py` 的 `SentimentAnalyzer._segment_text()` 异常分支。
- `similarity.py` 的 `SimilarityAnalyzer._tokenize()` 异常分支。

这些位置现在用 `logger.error` 记录固定事件和异常类型，不插入原异常文本，也不附加 traceback。只删除字符串中的异常文本并保留 `logger.exception` 仍可能通过 traceback 暴露原异常，所以同时取消了这两个分支的异常堆栈输出。

各分支原有的无效情感测量或空分词结果保持不变，未调整评分、数据缺失语义、API 响应、训练脚本或独立 CLI。其他未确认分支的日志未在本批统一清理，因此不能据此声称所有模型运行日志都已完成隐私验收。

## 回归与边界

新增 `src/hyh/tests/test_analysis_log_privacy.py` 两个有效故障场景：cntext 失败；分词失败同时进入情感与相似度模块。

测试通过真实 ASGI API 采集合成记录，保留一条合成旧数据的空正文来执行分析管线自身的标题回退。分类、情感、相似度、评估模块和日志辅助函数均从实际源码加载，日志写入临时 `LOCALAPPDATA` 下的真实文件，并捕获控制台输出。PyTorch、Transformers、cntext、jieba、scikit-learn 是受控依赖替身，未加载或下载模型，不代表真实推理质量或真实依赖故障验收。

断言覆盖：

- 异常处理确实收到正文和回退标题；日志保留固定事件及 `RuntimeError` 类型。
- 文件日志、控制台、API 响应和 `analysis_jobs` 错误/结果字段没有源文、异常细节或合成能力密钥，没有 traceback。
- API 保持 `422` 和失败任务，无有效结果；原始页面记录及正文保持原样。没有通过删除页面内容来消除日志标记。

异常里的能力密钥是故障注入的合成值，用于检查任意敏感异常内容不会落盘；生产 API 并不把认证密钥作为模型输入，本批未发现那样的数据路径。

新增两项在旧日志实现上失败，修复后与全部 `src/lsj/tests` 及旧统计回归合计 **149 passed，15.73 秒**。输出有一条已有 Starlette/httpx 测试客户端弃用提示，以及两个真实分类批处理触发的 pandas `fillna` 后续行为变更提示；本批未改动它们。`git diff --check` 通过。完整回归与提交后的 GitHub CI 结果由总验证另行报告。
