# 数据库路径与原子初始化（2026-10-03）

本轮修复两个经独立临时数据库复现的问题：空配置导致数据库定位异常，以及 schema 初始化失败后留下部分已提交的结构。仅修改 `src/hyh/db.py`，不增加历史数据库自动迁移或替换逻辑。

## 配置输入

- 没有设置 `IDM_DB_PATH` 时，默认仍是 `%LOCALAPPDATA%/InformationDietManager/idm.sqlite3`。
- `LOCALAPPDATA` 缺失、为空或只有空白字符时，回退到当前用户的 `~/.local/share/InformationDietManager/idm.sqlite3`。改变启动工作目录不会改变该回退位置。
- **显式设置但为空或只有空白字符的 `IDM_DB_PATH` 会报错**：`IDM_DB_PATH must be a non-empty database file path.`。不会悄悄改开默认数据库，避免用户把配置错误误认为原数据消失。
- 非空显式路径沿用原有语义：相对路径按启动目录解析，支持 `~`、绝对路径和 Unicode。有效路径内容不会被统一 `strip()`，以免改变用户指定的文件名。
- 如果最终路径已经是目录，在配置解析阶段返回固定错误 `Database path must name a file, not an existing directory.`，不把该路径写进错误文字，也不会继续创建锁文件或密钥文件。

修复前，空 `LOCALAPPDATA` 使两个启动目录分别创建各自的 `InformationDietManager/idm.sqlite3`；空 `IDM_DB_PATH` 则被解析成当前目录，直到 SQLite 打开时才失败。上述现象均在独立临时目录复现，未访问个人数据库。

## 初始化事务

`init_db()` 先以 UTF-8 完整读取 schema，再创建数据库父目录和打开连接。读取失败时，这个函数不会创建数据库文件或父目录。此处仅指初始化函数；应用启动的进程锁和凭据准备仍有其自己的生命周期。

schema 在显式 `BEGIN IMMEDIATE` 和 `COMMIT` 之间执行。任一语句失败时回滚整个初始化事务；即使回滚本身又抛异常，也保留原始初始化错误，并在 `finally` 中关闭连接。

修复前，一个仅有 `analysis_jobs(id, status)` 的合成旧库会在创建 `input_hash` 索引时失败，但此前已新增并提交 16 个表、索引或触发器。修复后，同一失败保留原始 schema 和已有合成行，不留下部分 DDL。没有把这种部分修改描述成已经丢失数据，也不尝试补齐未知旧表。

成功初始化以及对当前 schema 的重复初始化保留已有页面、数据库身份、revision、索引和触发器。新建数据库在执行 SQL 后失败可能留下空数据库文件；它不含部分业务结构，可以在修复初始化原因后正常重试。本轮没有新增删除失败文件的行为。

页面备份恢复仍在原表中事务性删除和插入记录，不替换 SQLite 文件或运行迁移。独立合成恢复探针确认恢复前后的 18 个 schema 对象完全相同，URL 唯一索引继续识别重复记录，`integrity_check` 正常且 `foreign_key_check` 无结果；这不代表任意历史库都能迁移。

## 验证边界

新增 `src/hyh/tests/test_db_initialization.py` 的 20 项回归通过，覆盖默认目录与两个启动目录、显式空配置、相对/主目录/中文路径、目录误配置、成功及重复初始化、schema 读取失败、不兼容旧表与新库的晚期 SQL 失败、连接关闭以及回滚二次异常保留原错误。

测试使用临时路径、合成旧表和故障注入，不读取个人数据库或运行模型。完整后端和对应提交的 CI 结果由最终集成验证另行报告，本说明不替代它们。
