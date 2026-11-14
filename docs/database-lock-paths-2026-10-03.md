# 数据库与进程锁路径自碰撞保护

## 已复现的问题

进程锁沿用数据库路径替换后缀的命名方式：`records.sqlite3` 对应 `records.lock`。此前允许把数据库本身配置为 `records.lock`，导致锁文件与数据库重合。

在 Windows CPython 3.12.5 的临时合成目录中，真实 `scripts/run_backend.py` 因此以退出码 3 结束，SQLite 报 `disk I/O error`。新路径留下 1 字节锁哨兵；已有非空合成数据库启动失败，但数据库字节和记录没有改变，不能将该证据描述为既有数据损坏。真实 `scripts/rotate_local_keys.py` 则仍返回成功；新数据库路径同样被创建为 1 字节锁文件。

`records.LOCK` 和 `records.lock ` 也在真实 Windows 文件系统上与派生锁文件重合。仅比较原始字符串或依赖 `Path.resolve()` 不足以处理后一种情况：对尚不存在的路径，当前 Python 保留了尾空格。

## 当前保护与兼容性

`process_ownership()` 在创建父目录或打开任何文件之前，检查数据库与派生锁文件是否同名。两个路径始终位于同一父目录，因此检查仅比较最终文件名，不引入全局路径改写。碰撞时返回固定错误，不包含用户路径：

```text
Database path conflicts with its process lock file; choose a different database filename.
```

普通 Win32 路径的比较处理大小写及尾部 ASCII 空格、点；明确使用 `\\?\` 扩展前缀的路径保留尾字符。Microsoft 的[文件路径格式说明](https://learn.microsoft.com/en-us/dotnet/standard/io/file-path-formats#skip-normalization)指出该前缀跳过常规路径规范化，[文件命名说明](https://learn.microsoft.com/en-us/windows/win32/fileio/naming-a-file)说明了默认大小写行为及扩展命名空间；[NTFS 故障排查说明](https://learn.microsoft.com/en-us/troubleshoot/windows-server/backup-and-storage/cannot-delete-file-folder-on-ntfs-file-system#cause-6-the-file-name-includes-an-invalid-name-in-the-win32-name-space)进一步解释了普通打开方式如何去除尾空格和点。实现同时用真实临时文件验证这些边界。

这不是一律禁止带尾点的数据库名称。例如 `distinct.lock.` 对应的锁名为 `distinct.lock..lock`，二者实际不同，应继续允许。扩展路径下的 `literal.lock ` 与 `literal.lock` 同样保持区分。

启动和离线轮换都经过此检查。正常数据库、锁和凭据命名不变，没有改写、迁移或删除旧数据库及密钥。不同文件主名仍可独立持锁；同目录 `same.sqlite3` 和 `same.db` 仍沿用既有的同主名锁与凭据空间，不能将这两个名字视为独立实例。本次没有改变该约定，也不是通用硬链接、重解析点或文件系统竞争审计。

## 验证范围

新增 `src/hyh/tests/test_database_lock_paths.py` 使用临时目录及合成数据，覆盖：

- `.lock`、Windows 大写后缀及尾空格在创建父目录前被拒绝。
- 真实启动和轮换脚本，对新路径及已有合成 SQLite 数据库均明确失败；新路径不产生父目录或文件，旧目录文件集合和全部字节不变，原记录仍可读取，输出不包含合成私密路径。
- 普通中文路径、既有 sidecar 命名、不同主名独立锁、同主名共享锁的兼容性。
- 普通尾点路径和扩展路径下字面尾空格文件不会被误判为同一文件。

新增 20 项与既有 `test_local_security_and_data.py` 合计 **77 passed，30.43 秒**。其中包含现有真实启动、密钥轮换、旧密钥失效及记录保留测试；存在一条已有 Starlette/httpx 测试客户端弃用提示。`git diff --check` 通过。未访问个人数据库或凭据，未执行模型或修改前端；完整回归与提交后的 GitHub CI 结果由总验证另行报告。
