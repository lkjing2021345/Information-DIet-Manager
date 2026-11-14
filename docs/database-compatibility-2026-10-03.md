# 已验证的历史数据库结构兼容范围

本批将已通过临时库复核的三个历史 schema 固化为持续回归，不修改数据库初始化代码，不引入通用迁移器。夹具只含 Git 中的 SQL 文本，没有复制历史数据库或个人记录。

## SQL 来源

以下文件均来自对应提交的 `src/hyh/schema.sql`，保存于 `src/hyh/tests/fixtures/historical_schemas/`：

| SQL 夹具 | 完整提交 | 原 Git blob SHA-1 | 该阶段已有表 |
| --- | --- | --- | --- |
| `schema-3d68b00.sql` | `3d68b00d40f52cda579c927c6dba09b1e21b5752` | `7d5ac26b26f64e6bb6e3dbfa0baa6c71ee2ad11e` | `items` |
| `schema-9ca3eb8.sql` | `9ca3eb81753bc43b410bcdd669f4eb26903e7f1c` | `092f5262257ea13a0f203461bf41fd7bda9e19d6` | 再增加 `embeddings`、`stats_daily` |
| `schema-953a289.sql` | `953a2894f94ddd5329bcf7d4d4bd87d302f0f343` | `68ba1c969a0f3ed971717fab2538e39ba3c83da7` | 再增加 `analysis_runs`、`analysis_jobs` |

夹具从原 blob 直接复制，没有修改 SQL。测试在执行前验证原 Git blob 哈希；允许 Windows Git 检出时自动将 LF 改为 CRLF，校验前仅将 CRLF 还原为 LF。测试运行不调用 Git，因此无需完整提交历史。这里的 SHA-1 用于匹配 Git 对象身份，不作为外部不可信数据的安全签名。

暂存差异检查对 `schema-3d68b00.sql` 提示文件末尾多一个空行；该空行存在于原 Git blob，已核对暂存对象与原对象一致，因此有意保留，避免改写历史夹具。其余暂存文件的空白检查通过。

## 持续回归

`src/hyh/tests/test_database_compatibility.py` 只有一个参数化流程，分别执行上述三个基线：

1. 在临时路径用历史 SQL 创建数据库，向当时存在的业务表写入合成记录和合成派生数据。
2. 运行当前 `init_db()`，确认旧表全部行保留；新增数据库身份和初始 revision 正常。
3. 通过合成更新触发新增 revision 触发器，再显式初始化一次并正常启动应用。确认已有数据、数据库身份、非零 revision、表、索引及触发器不因重复初始化被重置。
4. 用合成管理密钥向实际 ASGI 应用发出 HTTP 请求，确认 `/items` 和 `/data/backup` 返回原合成页面及其可选字段。
5. 检查 `PRAGMA integrity_check` 返回 `ok`，`foreign_key_check` 没有结果。

新增 3 项与现有 `test_db_initialization.py` 的 20 项合计 **23 项通过**，环境为 Windows、Python 3.12.5。现有初始化测试已经覆盖不兼容旧表、晚期 SQL 错误及失败回滚，本批不重复这些用例。测试存在一条已有 Starlette/httpx 测试客户端弃用提示。

## 支持边界

这三个历史结构可以通过现有的追加表、索引和触发器机制初始化；结论限于所列 SQL 和当前字段契约内的合成记录。不保证未知手工改表、缺列、损坏文件、断电、磁盘故障、任意旧数据质量或全部历史提交都能升级，也不自动补齐未知旧列。初始化遇到不兼容结构仍失败回滚，迁移前仍需停止服务并备份自己的原文件。

另一次只读启动探针核对了现存最小环境的 14 个运行包与当前运行锁逐项一致；在未安装 pandas 的情况下，`scripts/run_backend.py` 从外部临时工作目录启动成功，回环 HTTP 健康检查和认证记录查询均成功，使用的是临时数据库和合成密钥。该旧探针环境的 pip 仍为 24.2，不是当前安装器锁的 26.2.1；这只证明运行依赖闭包和启动路径，不代表本批重新执行或验证了一次完整哈希安装。README 的常规安装仍应先消费安装器锁，再消费运行锁。

上述测试和探针没有使用个人数据库、浏览器资料或模型，不等同于真实模型推理、全面升级认证或故障注入压力验收。
