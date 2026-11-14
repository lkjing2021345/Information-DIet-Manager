# HTML 报告文本边界修复

日期：2026-10-03。范围为独立评估器 `InformationQualityEvaluator.export_report(..., format="html")`，不修改前端界面、扩展、模型推理、评分或 Markdown 语义。

## 确证问题

旧实现直接将摘要字符串及 `json.dumps(report.to_dict())` 放入 HTML 的两个 `<pre>` 中。JSON 字符串编码并不提供 HTML 文本上下文的转义：`</pre><script>...</script><pre>` 会被浏览器解析为元素和脚本；普通 `&lt;...&gt;` 原文也会被误解码，显示不再等于报告原文。

合成回归实际调用评估器 `evaluate()`，输入满足最低样本数的 5 条预分类 DataFrame；任意 `category` 经已有小写、去边缘空白规范化后，进入真实 `EvaluationReport` 的 `dominant_category` 和 `category_distribution`，随后抵达实际导出文件。没有伪造 `to_dict()` 或直接拼接一份假报告。另通过公开的真实 `HealthStatus.justification` 字段验证摘要边界；这不表示正常 `evaluate()` 会从分类字段生成该摘要内容。

这证明独立导出器接收不可信预分类数据或报告字段时存在存储型 HTML 注入；没有据此宣称当前仪表盘 API 存在相同可达路径。实际模型通常产生受限分类标签，本轮没有进行真实模型推理或检查个人浏览记录。

## 实现

只在两个 HTML 插值处调用标准库 `html.escape`。报告对象及其原始字符串、中文、emoji、实体字面量、换行和 JSON 序列化不变；浏览器通过 HTML 实体解码后可恢复原文。JSON、Markdown 导出分支和原 HTML 文档结构保持不变。

## 回归证据

`src/lsj/tests/test_report_html_safety.py` 共 13 项：

- 摘要和真实 category 路径分别覆盖 script、SVG 事件、data URL 图片错误事件、DOM 元素以及中文/引号/实体/换行/emoji 共 10 个案例。
- 读取真实导出文件，用 `HTMLParser` 检查只剩原有标签和属性，并比较两个 `<pre>` 的完整解码文本与原始摘要、JSON 字符串；确认报告未被修改。
- JSON、md、markdown 三种导出与既有真实序列化方法逐字比较。JSON 对象比较按 JSON 固有的整型键转字符串语义归一化。
- 修复前同一组回归为 **10 failed / 3 passed**；修复后为 **13 passed**，使用仓库 Windows CPython 3.12 锁定测试环境。

可选模型依赖与日志初始化使用导入替身；评估、报告数据类、JSON、Markdown 生成器和 HTML 导出器均为真实实现。所有输入为合成数据，写入 pytest 临时目录。JSON/Markdown 兼容测试不表示任意下游 Markdown 渲染器能够安全处理原始 HTML。

供真实隔离 Chromium 验收的 `before.html`、`after.html`、`expected.json` 及生成脚本保存在忽略目录 `output/playwright/html-report-20261003/`。两个版本使用相同合成数据、固定生成时间；旧文件在修改代码前由真实导出器产生。载荷只设置窗口标记或合成 DOM 标记，无远程请求、数据读取或破坏操作。

主任务使用仓库锁定 Playwright、已安装的 Chromium（`channel=chromium`），分别创建隔离上下文通过 `file://` 打开旧、新文件，拒绝非文件网络请求。实际结果：旧文件三个窗口标记均为 `1`，注入 DOM 标记有 5 个，`pre` 有 9 个；新文件三个标记均不存在，注入 DOM 为 0，`pre` 恰为 2，两个 `textContent` 与预期摘要及 JSON 字符串逐字相同。两次浏览均没有网络请求，上下文已关闭。浏览器脚本及结果保存在同一忽略目录的 `browser-check.cjs`、`browser-result.json`。首次默认 headless shell 路径因未安装未能启动，改用已有锁定 Chromium 后验收成功，没有新增浏览器下载。
