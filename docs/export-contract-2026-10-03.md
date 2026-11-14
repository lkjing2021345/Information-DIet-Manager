# 导出过滤与 CSV 契约修复（2026-10-03）

本批解决训练导出的内部地址漏筛、公开域名误筛，以及空 CSV 无法按列读取的问题。没有修改前端或模型，不改变训练文本的既有清理规则。

## 地址过滤范围

`GET /export/lsj/training` 默认 `exclude_internal=true`。过滤仅检查已保存 URL 中的主机名：

- `localhost`、其子域以及 `.local` 域名，不区分大小写，允许末尾的 DNS 根点。
- 使用 Python 标准库 `ipaddress` 解析 IP 字面量，排除非全球可达地址、保留地址、组播、回环、链路本地、已废弃的 IPv6 站点本地地址和未指定地址。IPv4 映射 IPv6 地址先转换为 IPv4 再判断，避免误删映射的公网地址。
- 普通 DNS 名不会因为 `10.`、`172.16.` 或 `192.168.` 前缀而被删除。不查询 DNS，也不连接导出中的站点。

`exclude_internal=false` 保留这些记录。原始/分析交换导出 `/export/lsj` 仍不启用此过滤，两类导出都要求管理员能力密钥。

这是基于 URL 的有限筛选，不是匿名化：普通域名仍可能解析到内网，公开地址中的正文、路径和查询参数也可能包含私人信息。规则没有声称识别这些内容。IP 分类沿用受支持 Python 运行时的标准库；具体定义见 [Python ipaddress 文档](https://docs.python.org/3.12/library/ipaddress.html)。

## CSV 列与原文

CSV 使用固定列顺序。空数据库、空时间窗口以及训练过滤后没有剩余记录时，仍返回 `200` 和表头，Python `csv.DictReader`、pandas 都可读取零行表格。

| 导出 | 列顺序 |
| --- | --- |
| `/export/lsj?view=analysis` | `id,title,url,analysis_text,text,channel,ts,source,lang,author,tags,meta` |
| `/export/lsj?view=raw` | `id,url,title,text,ts,source,lang,channel,author,tags,meta,created_at` |
| `/export/lsj/training` | `input,label,ts,url,title,source` |

沿用标准 CSV 引号、逗号、换行编码，字典和列表单元格继续使用 JSON。训练 `input` 仍按既有规则合并空白、截断长度和去重；本批没有增加新的文本清理。

以 `=` 等字符开头的原文也保持原样，没有添加单引号等电子表格公式转义。该格式面向程序读取，不应把未知页面文本当作可信公式在电子表格中执行；若需要电子表格专用安全格式，应提供独立的显式选项，而不是悄悄改变训练数据。背景见 [OWASP CSV Injection](https://community.owasp.org/attacks/CSV_Injection)。

## 验证

新增 `src/hyh/tests/test_export_contract.py`，使用独立临时 SQLite、合成记录、管理员/采集器测试密钥和真实 FastAPI 请求。17 项测试覆盖：

- JSON、JSONL、CSV 三种训练格式中的内部地址过滤与显式关闭过滤。
- IPv4 回环、私网、链路本地、共享地址、组播，IPv6 ULA、链路本地、站点本地、保留地址、组播及 IPv4 映射地址；本地域名末尾根点和公开域名保留。
- 空库、空窗口和过滤空结果的固定表头及 pandas 解析。
- 中文、逗号、引号、换行、JSON 单元格与公式形原文的往返读取。
- 普通交换导出保持未筛选语义，采集器密钥不能导出。

最终版本执行 `python -m pytest src/hyh/tests src/lsj/tests -q -p no:cacheprovider`：**277 项通过**，用时 75.67 秒。测试输出有一条 Starlette 关于当前 `httpx` 测试适配器弃用的提示；本批未切换测试依赖。

测试不访问个人数据库、不发起 DNS 请求或模型推理，不等同于真实模型验收或完整浏览器验收。GitHub CI 状态由提交报告单独记录。

## 尚未解决的大规模导出内存问题

更新：本节是 `00c9156` 时的基线。后续已完成有预算的临时文件导出和下载锁隔离，见[导出资源与下载隔离](bounded-exports-2026-10-03.md)；下面保留修复前测量，不应继续当作当前实现。

目前最多允许 `limit_rows=200000`，但实现仍先 `fetchall`、构造列表并完成 JSONL/CSV 编码，之后才通过 `StreamingResponse` 返回；响应类名称不能证明输出是有界内存流式处理。

本轮独立临时数据库试验中，1,000 行合成记录（每行约 990 个中文文本字符、7,000 字符的 ASCII 元数据值，均在现有字段限制内）在尚未消费响应前就已全部完成 JSONL 编码。Python `tracemalloc` 记录峰值约 **69.72 MiB**、当时保留约 **35.37 MiB**。这是该次合成负载的 Python 分配测量，不是进程总内存、通用性能基准或 20 万行压力测试；没有尝试耗尽主机内存。

后续应独立实现分块读取、明确输出字节预算、必要时写入有限额临时文件，再关闭数据库快照后提供下载。训练去重集合也需要内存预算；不能只把编码移入生成器就宣布问题解决。当前版本尚未证明在允许的最大规模下可靠导出。
