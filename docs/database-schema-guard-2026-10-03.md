# 启动时间列结构检查：2026-10-03

## 已确认问题

`CREATE TABLE IF NOT EXISTS` 不会验证同名旧表的列类型。使用首版历史 SQL，仅将 `items.ts` 改为 TEXT，应用仍能正常启动、健康检查返回 200。五次合成采集请求传入严格整数时间戳 `1..5`，均返回插入成功，记录列表总数也是 5；查询 `from_ts=0&to_ts=10` 时却返回 200、`insufficient_data`，输入和窗口可用记录数均为 1。原因是 TEXT 亲和性把整数保存为文字，窗口比较随后按文字排序。

同一探针的原 INTEGER 结构返回正确数量 5。两库的 SQLite `integrity_check` 均为正常；结构检查正常不等于业务时间字段兼容。TEXT 库备份返回 409，没有观察到失败删除或数据丢失。全部数据来自新建临时 SQLite，未读取任何用户数据库。

## 本次处理

`init_db()` 在同一连接执行任何初始化 DDL 前，只读检查现有 `items` 对象及 `ts` 列：

- `items` 不存在时，继续原空库创建流程。
- 已有对象必须是表，并有普通 `ts` 列，其声明类型必须具有 SQLite INTEGER 亲和性。视图、缺列和其他亲和性返回固定错误，不包含原记录、路径或 SQL 内容。
- 表名、列名按 SQLite 的大小写兼容方式识别；类型不要求精确拼写为 `INTEGER`。SQLite 首先检查声明中是否含有 ASCII 大小写不敏感的 `INT`，因此 `BIGINT`、`UNSIGNED BIG INT`、`CHARINT` 和 `FLOATING POINT` 都可以具有 INTEGER 亲和性。[SQLite 官方类型规则](https://www.sqlite.org/datatype3.html#determination_of_column_affinity)

检查失败时不执行初始化 DDL/DML，原逻辑 schema 和记录保持不变。检查通过后仍使用原有单次 `BEGIN IMMEDIATE … COMMIT` 脚本，后续 DDL 失败仍整体回滚。没有自动改列、转换时间值或清理历史数据。

## 验证

新增 `test_database_schema_guard.py` 最终 30 项，包含：不兼容声明拒绝、失败前无 DDL/DML、原 schema/记录保持、连接释放、缺列、视图、混合大小写、十种 INTEGER 亲和性声明的真实 HTTP 采集/窗口/备份、拒绝应用就绪、晚期 SQL 失败回滚、空库和重复初始化。

首批 29 项在原实现上实测 **17 failed / 12 passed，4.33 秒**。自查又发现直接使用 Python `upper()` 会把 Unicode 类型名 `ınt` 当作 `INT`，而真实 SQLite 的 CAST 对照表明它属于 NUMERIC；追加该项初版失败后，改用 ASCII 大小写匹配并通过。未将所有不兼容类型都描述为具有相同的 TEXT 漏数行为。

与原初始化、三个精确历史 SQL、二次清理错误及备份往返测试联动，最终 **120 passed / 1 warning，9.35 秒**；既有 Starlette/httpx 适配器弃用提示保留。原初始化测试的两个手写连接替身仅增加真实 `execute` 委托，原关闭和首错断言保持。`git diff --check` 通过。

HTTP 检查的分析流水线为显式不可用桩，没有加载真实模型。完整集成回归、浏览器和精确提交 CI 由主流程另行记录，不能用这些定向测试替代。

独立复核新结构门禁、旧结构兼容和初始化测试，53 项通过（4.78 秒），保留 1 条既有 Starlette 警告；未发现新的阻断。

## 限制

这只检查已确证问题涉及的 `items.ts` 结构，不是完整 schema 认证、通用版本迁移或全库数据质量扫描。SQLite 即使声明 INTEGER 亲和性，也允许存入部分其他类型；本次不验证每条旧记录的实际值，不保证其他列、约束或触发器正确。其他未知结构仍可能在既有初始化语句或后续操作中被拒绝。

前置读取不改变原 DDL 事务结构，也不承诺阻止绕过应用进程锁的外部程序在检查与初始化之间修改 schema。失败后应保留原库，使用经过确认的迁移或修复方案；本次不会自行把不兼容存量转换成另一种结构。
