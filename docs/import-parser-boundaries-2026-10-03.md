# 导入文件解析错误边界

## 已复现的问题

`/import` 在准备数据库写入前先完整解析 CSV、JSON 或 JSONL，但此前只捕获 `ValueError`（以及它的子类 `JSONDecodeError`）。Python 3.12.5 的默认 CSV 单字段上限为 131072 个字符；超过该解析器上限会抛出 `csv.Error`。约 20 KiB、10000 层嵌套的合成 JSON/JSONL 会抛出 `RecursionError`。这些文件都低于现有 10 MiB 请求体预算，却返回了服务器错误 500，而合同要求无法解析的导入文件返回 400。

修复前的临时库 HTTP 复现没有发现部分写入：即使 CSV/JSONL 的首条记录有效，后续解析失败时六张业务表仍保持原样。原安全中间件也没有在 500 响应或捕获日志中暴露合成隐私标记。本次修复针对错误分类和异常详情边界，不将该问题描述为已发生的数据损坏或隐私泄漏。

## 当前行为

文件解析边界捕获 `ValueError`、`csv.Error` 和 `RecursionError`，统一返回：

```json
{"detail":"Invalid file contents."}
```

HTTP 状态为 400，响应不包含原异常文本，也不保留供错误链显示的原解析异常。CSV 字段限制、HTTP 请求体预算和逐条记录验证保持原值。能够完成文件解析、但包含不符合记录字段要求的数据，仍按既有 `inserted`、`duplicates`、`failed` 合同处理；存储失败仍沿用事务回滚和服务器错误处理。

## 验证范围

新增 `src/hyh/tests/test_import_parser_boundaries.py` 共 8 项：

- 5 项通过真实 ASGI 应用和 HTTP multipart 上传，覆盖超长 CSV 字段、超深 JSON/JSONL、语法错误 JSON/JSONL。使用独立临时 SQLite 数据库，检查六张业务表不变、CORS 与 `no-store` 响应头、服务可继续使用及相同格式的有效文件重试成功。
- 3 项为解析器异常注入，分别验证上述异常类型即使带有合成私密内容，也只返回固定 400，响应和捕获日志不含该内容。这些用例验证异常处理边界，不代表真实解析器会产生相同异常文本。

新增测试与既有 `test_ingestion_contract.py`、`test_write_cancellation.py` 共 101 项通过。测试使用隔离的 Python 3.12.5 环境；存在一条已有 Starlette/httpx 测试客户端弃用提示。没有使用个人数据库、浏览器资料、模型或网络推理。该批结果是本地合成 HTTP/事务回归，不等同于真实模型验收或全部资源压力测试。
