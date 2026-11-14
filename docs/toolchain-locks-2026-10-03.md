# Python 维护工具的完整哈希锁

本批次补齐审计器和锁生成器的传递依赖，替代旧维护流程中直接执行 `pip install pip-audit==...` 或 `pip install pip-tools==...` 的步骤。应用运行锁、测试锁和安装器锁未改变；工具依然与应用环境隔离。验收范围为 Windows / CPython 3.12 / AMD64。

## 输入、闭包和安装器

`requirements/locks/toolchain.in` 固定 `pip-audit==2.10.1` 和 `pip-tools==7.6.1`；`toolchain-windows-py312.txt` 锁定其 35 个直接及传递包的完整版本和 SHA-256，包括 packaging、setuptools、wheel 和审计器自身。直接工具版本保持不变。[pip-audit 官方发布元数据](https://pypi.org/pypi/pip-audit/2.10.1/json)、[pip-tools 官方发布元数据](https://pypi.org/pypi/pip-tools/7.6.1/json)。

工具依赖中的 pip 使用已有 `installer.txt` 单独锁定、安装和审计。生成命令通过 `--unsafe-package=pip --no-allow-unsafe` 显式替换 pip-tools 的默认排除集合，只把 pip 留给安装器锁，并使用安装器锁约束解析版本；setuptools 等其他工具依赖均进入工具锁。**这不是漏洞忽略清单，审计不允许忽略 advisory 或跳过任何包。**

因此，工具锁末尾保留 pip-tools 关于 pip 未在该文件中固定的生成警告。它说明工具锁必须与安装器锁配套使用；本轮已经从空环境验证“先安装安装器，再安装工具”的顺序。不能把这个工具锁当作不需要安装器的独立安装入口。

## 建立隔离维护环境

从仓库根目录运行，每条命令失败后立即停止。以下环境只供维护，不能用于验收应用的最小安装集合。

```powershell
py -3.12 -m venv output/python-maintainer-tools
if ($LASTEXITCODE -ne 0) { throw 'Could not create maintainer environment' }
$toolPython = 'output/python-maintainer-tools/Scripts/python.exe'
& $toolPython -m pip --isolated install --disable-pip-version-check --index-url https://pypi.org/simple --only-binary=:all: --require-hashes -r requirements/locks/installer.txt
if ($LASTEXITCODE -ne 0) { throw 'Installer bootstrap failed' }
& $toolPython -m pip --isolated install --disable-pip-version-check --index-url https://pypi.org/simple --only-binary=:all: --require-hashes -r requirements/locks/toolchain-windows-py312.txt
if ($LASTEXITCODE -ne 0) { throw 'Toolchain installation failed' }
& $toolPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Toolchain dependencies are inconsistent' }
```

## 审计自身并逐包对账

```powershell
$auditCache = 'output/python-maintainer-audit-cache'
& $toolPython -m pip_audit --strict --disable-pip --require-hashes -r requirements/locks/toolchain-windows-py312.txt --vulnerability-service pypi --cache-dir $auditCache --progress-spinner off --format json --output output/python-toolchain-audit.json
if ($LASTEXITCODE -ne 0) { throw 'Toolchain audit failed' }
& $toolPython -m pip_audit --strict --disable-pip --require-hashes -r requirements/locks/installer.txt --vulnerability-service pypi --cache-dir $auditCache --progress-spinner off --format json --output output/python-installer-audit.json
if ($LASTEXITCODE -ne 0) { throw 'Installer audit failed' }
& $toolPython scripts/verify_python_dependencies.py --lock requirements/locks/toolchain-windows-py312.txt --installer-lock requirements/locks/installer.txt --input requirements/locks/toolchain.in --audit output/python-toolchain-audit.json --installer-audit output/python-installer-audit.json
if ($LASTEXITCODE -ne 0) { throw 'Installed packages and audit coverage do not match' }
```

继续使用现有严格验证器：工具锁与安装器锁不能重叠，两者的并集必须与实际安装包名及版本完全一致，工具审计和安装器审计分别必须完整。没有新增“允许未知包”“跳过工具包”或按包数近似通过的例外。即使 setuptools 是构建工具、packaging 已存在于审计器环境，它们仍须出现在工具锁和显式审计 JSON 中。

## 更新流程

生成脚本现在检查整套维护环境，缺包、多包、版本漂移或平台不符都会在解析前失败。`--scope` 可独立更新应用锁或工具锁；默认仍是应用锁，避免一次普通更新隐式改变工具链。

```powershell
# 默认保留已有版本：
& $toolPython scripts/lock_python_dependencies.py --scope toolchain
# 明确重新解析工具的传递依赖：
& $toolPython scripts/lock_python_dependencies.py --scope toolchain --upgrade
# 应用运行/测试锁仍使用原来的默认入口：
& $toolPython scripts/lock_python_dependencies.py
```

生成新工具锁后，在新的隔离环境重装并审计，再用新环境重新生成验证；旧环境不符合新锁时，生成器会拒绝继续工作。若更新 pip-tools 或 pip 的直接版本，还需明确修改并验证脚本中的生成器版本配对。不要在现有应用测试环境安装维护工具来绕过检查。

CI 先从安装器锁和工具锁建立独立环境、执行 `pip check`，再审计工具和安装器并逐包对账，通过后才使用它审计应用。CI 只消费提交的锁，不每次重新解析传递依赖；锁生成是维护者主动执行和审查的操作。

## 本轮证据与边界

2026-10-03，在 Windows 11 / CPython 3.12.5 的新环境验证：

- 安装器 1 包、工具 35 包全部按官方 PyPI 哈希以 wheel 安装，`pip check` 通过。
- 工具锁、实际环境和工具审计 JSON 的 35 个包逐项一致；pip 安装器另审计 1 包。均为零跳过、零已知漏洞。
- 使用新工具环境重新审计原应用测试锁：29 个应用/测试包、14 个运行子集与原隔离测试环境一致，安装器另计，零跳过、零已知漏洞。
- 新环境实际运行 `--scope toolchain`，重生成后的锁文件字节完全一致。
- 依赖验证和工具锁回归共 19 项通过，覆盖漏审构建工具、包集污染、版本漂移、安装器重叠、生成范围和安装器唯一分锁等边界。

生成器首次计算候选 wheel 哈希仍可能较慢，这是既有 pip-tools 行为；重生成可以复用锁内哈希。允许多平台 wheel 哈希不代表本轮做过跨平台验收。

上述结果是本机检查和查询时的已知漏洞信息，远端 CI 必须对应最终提交核验。它们不证明没有未知漏洞、构建来源完全可信或达成完整供应链认证。模型环境、操作系统/解释器补丁、签名制品和发布来源证明仍在本批次范围之外。原始安装报告、审计 JSON、工具环境及缓存均留在被忽略的 `output/`。
