# 最小 Python 环境的依赖锁与完整审计

本轮锁定的是本地 API 和合成数据测试所需的 Python 依赖。模型推理、训练、浏览器二进制和 Node.js 依赖不在这些锁文件中。目标是 **Windows / CPython 3.12 / AMD64**；本机安装验收使用 Windows 11 和 CPython 3.12.5，不据此承诺其他平台、解释器版本或真实模型可用。

## 文件与职责

| 文件 | 用途 |
| --- | --- |
| `requirements-runtime.txt` | 人工维护的 4 个运行直接依赖 |
| `requirements-test.txt` | 引入运行输入，并增加 3 个测试直接依赖 |
| `requirements/locks/installer.txt` | 固定 pip 26.2.1 官方 wheel 的 SHA-256，单独审计安装器 |
| `requirements/locks/runtime-windows-py312.txt` | 14 个运行包的完整版本与允许的 SHA-256 |
| `requirements/locks/test-windows-py312.txt` | 29 个运行/测试包的完整版本与允许的 SHA-256 |
| `requirements/locks/toolchain-windows-py312.txt` | 独立审计/锁生成工具及其传递依赖的完整哈希锁；后续治理批次新增 |
| `scripts/lock_python_dependencies.py` | 先生成运行锁，再用运行锁约束测试锁 |
| `scripts/verify_python_dependencies.py` | 对比输入、锁、实际安装和审计的包名/版本集合 |

普通安装直接消费锁文件，不现场重新选择传递依赖。安装使用 `--require-hashes --only-binary=:all:`，拒绝未知哈希和源码构建。哈希用于校验取得的制品是否在允许清单中，不能证明依赖没有漏洞或恶意代码。[pip 安装安全说明](https://pip.pypa.io/en/stable/topics/secure-installs/)

## 新环境安装

从仓库根目录执行；`.venv` 应是为本项目新建的隔离环境。不要把维护工具、审计工具或模型依赖塞进待验收的最小环境。

```powershell
py -3.12 -m venv .venv
$appPython = '.venv/Scripts/python.exe'
& $appPython -m pip --isolated install --disable-pip-version-check --require-hashes --only-binary=:all: --index-url https://pypi.org/simple -r requirements/locks/installer.txt
& $appPython -m pip --isolated install --disable-pip-version-check --require-hashes --only-binary=:all: --index-url https://pypi.org/simple -r requirements/locks/runtime-windows-py312.txt
& $appPython -m pip check
```

测试环境将最后一个安装命令的运行锁换成 `requirements/locks/test-windows-py312.txt`。测试锁已经包含全部运行依赖，不需要再安装未锁定的输入文件。每一步失败后应停止，不继续启动服务或运行后续验收。

## 审计覆盖必须和实际安装一致

在 pip-audit 2.10.1 的 requirements 解析路径中，临时环境先安装 pip、wheel、setuptools，再执行没有 `--ignore-installed` 的 `pip install --dry-run --report`，最后只审计报告中的待安装包。当前 wheel 依赖 packaging，导致 pytest 所需的 packaging 已被临时环境满足、未列入待安装报告。已在本轮复现：直接输入审计得到 28 包，实际环境是 29 包；漏项为 packaging 26.3。[pip-audit 2.10.1 源码](https://github.com/pypa/pip-audit/blob/v2.10.1/pip_audit/_virtual_env.py)

因此，门禁对完整哈希锁使用 `--disable-pip --require-hashes`，让审计器直接检查显式清单。`--disable-pip` 本身不证明依赖闭包完整；完整性由干净环境的带哈希安装、`pip check` 和下列集合对比共同验证。[pip-audit 使用说明](https://github.com/pypa/pip-audit#usage)

审计工具必须放在独立环境，先安装 `installer.txt` 再安装完整 `toolchain-windows-py312.txt` 哈希锁，不能仅安装直接工具版本。见[工具链锁说明](toolchain-locks-2026-10-03.md)。以下命令假定 `$auditPython` 指向该环境、`$appPython` 指向前述新测试环境：

```powershell
New-Item -ItemType Directory -Force output | Out-Null
& $auditPython -m pip_audit --strict --disable-pip --require-hashes -r requirements/locks/test-windows-py312.txt --vulnerability-service pypi --progress-spinner off --format json --output output/python-application-audit.json
& $auditPython -m pip_audit --strict --disable-pip --require-hashes -r requirements/locks/installer.txt --vulnerability-service pypi --progress-spinner off --format json --output output/python-installer-audit.json
& $appPython scripts/verify_python_dependencies.py --lock requirements/locks/test-windows-py312.txt --installer-lock requirements/locks/installer.txt --runtime-lock requirements/locks/runtime-windows-py312.txt --input requirements-test.txt --audit output/python-application-audit.json --installer-audit output/python-installer-audit.json
```

验证器默认通过当前解释器的 `importlib.metadata` 获取安装集合，也可以用 `--installed-json` 检查目标环境的 `pip list --format=json` 快照。以下任一情况都会失败：

- 直接输入版本发生变化，锁文件未同步；运行锁与测试锁共享版本冲突。
- 实际环境漏包、多包、版本不同或包名别名重复。只有安装器锁中的 pip 是应用锁外允许的包。
- 审计漏包、多包、版本不同、跳过包、缺少漏洞结果或发现已知漏洞。
- 锁文件包含未固定版本、缺失 SHA-256 或本项目不支持的要求语法。

这能阻止“包数看起来差不多”“审计器没有报漏洞”被误当作完整覆盖。漏洞数据库是查询时的已知信息，零告警不等于没有未知漏洞。

## 维护与更新

生成器要求 Windows CPython 3.12 AMD64、`pip==26.2.1` 和 `pip-tools==7.6.1`，并检查整个维护环境与工具/安装器锁逐包一致。单独创建维护环境，先安装安装器锁，再安装完整工具链锁；早期仅安装 pip-tools 的示例已经被替代。生成器清除继承的 pip 源、额外源、约束和本机配置，使用相对输入路径与固定的文件头命令，避免把本机路径或私有索引写进锁文件。

解释器、平台或工具版本不符合上述要求时，生成脚本以非零状态退出并说明要求，不静默生成其他平台的锁。安装器锁中 pip 26.2.1 wheel 的 SHA-256 来自官方发布元数据；本轮也单独通过 pip-audit 查询该版本，未发现已知漏洞。该结论仅对应本次查询日期，后续仍须持续审计和升级。[pip 26.2.1 官方元数据](https://pypi.org/pypi/pip/26.2.1/json)

```powershell
py -3.12 -m venv output/python-lock-tools
$lockPython = 'output/python-lock-tools/Scripts/python.exe'
& $lockPython -m pip --isolated install --disable-pip-version-check --require-hashes --only-binary=:all: --index-url https://pypi.org/simple -r requirements/locks/installer.txt
& $lockPython -m pip --isolated install --disable-pip-version-check --require-hashes --only-binary=:all: --index-url https://pypi.org/simple -r requirements/locks/toolchain-windows-py312.txt
& $lockPython scripts/lock_python_dependencies.py
# 需要主动更新传递依赖时，再使用：
& $lockPython scripts/lock_python_dependencies.py --upgrade
```

默认生成会尽量保留已有锁中的版本；`--upgrade` 才主动重新选择满足输入要求的传递版本。运行锁先生成，测试锁通过 `-c` 受其约束。[pip-tools 分层依赖工作流](https://pip-tools.readthedocs.io/en/stable/#workflow-for-layered-requirements)

不要把每次重新解析依赖放进普通 CI。CI 应安装受审查的锁、检查输入一致性并验收。更新锁后需要新环境安装、审计、完整测试、检查差异，再按项目工作流提交。

本轮没有选择旧 pip 来换取生成速度：pip 25.3 的官方元数据已有漏洞记录；pip-tools 7.5.3 配 pip 26.2.1 的实际运行也因 pip 内部接口变化失败。当前固定组合能成功生成，但 7.6.1 获取 PyPI JSON 哈希时遇到 404，会退回下载候选 wheel 计算哈希。首次生成观察到超过 1 GiB 缓存；已有锁的两次连续重生成合计约 12 秒、输出字节一致。此性能限制属于维护过程，不会使最终用户安装所有平台的 wheel。[pip-tools 7.6.1 实现](https://github.com/jazzband/pip-tools/blob/v7.6.1/piptools/repositories/pypi.py)

锁文件可能列出同一包多个平台 wheel 的允许哈希，依赖解析和验收范围仍仅为上述 Windows 环境。扩展平台前必须在相应环境重新生成并验收，不能把多平台哈希误当作跨平台测试通过。[pip-tools 跨环境说明](https://pip-tools.readthedocs.io/en/stable/#cross-environment-usage-of-requirements-in-requirements-txt-and-pip-compile)

## 本轮实测与边界

2026-10-03 的隔离验证结果：

- 运行锁 14 包安装成功，`pip check` 和最小 API 模块导入通过。
- 测试锁 29 包在新环境使用 pip 26.2.1、强制哈希、仅 wheel 安装成功；没有源码构建。
- 测试锁、实际环境和应用审计清单的 29 个包逐项一致；安装器 pip 另审计 1 包；均无跳过、无已知漏洞。
- 用错误的合成 SHA-256 执行真实 pip 安装预演，确认下载被拒绝。
- 验证器 14 项回归通过，包含漏审 packaging、版本漂移、重复别名、错误哈希格式、跳过包、安装器遗漏和真实 CLI 退出码检查。
- 此初始批次未锁生成工具与 pip-audit 自身的全部传递依赖；后续已经补齐独立工具链哈希锁及安装/审计集合校验，见[后续验收](toolchain-locks-2026-10-03.md)。模型环境及操作系统/解释器安全补丁验收仍不包含在这些结果中。

原始安装报告、审计 JSON、生成缓存和诊断输出仅留在被忽略的 `output/`，不随源码提交。

整合复核另从零创建应用环境，按 CI 的 `pip --isolated install` 命令安装上述哈希锁；`pip check`、**260 项后端测试**及 **26 项仓库/依赖脚本测试**通过。该新环境下的生产页面真实浏览器 **25 项**、扩展真实浏览器 **10 项**均通过。仍有已有 Starlette/httpx 弃用警告和前端主包超过 500 kB 的构建提示。这些检查不包括真实模型推理，远端 CI 状态以对应提交的 Actions 为准。
