# 2026-11-26 质量状态与迭代记录

本轮基于原作者已经合并的 `f4891f2` 继续维护，后续提交仍推送到 fork 的 `main`。前端源码和视觉文件没有改动，本轮完成依赖锁安全升级、报告确定性、锁生成范围复现、CI 隔离环境校验和离线分析 CLI 契约收紧。

## 本轮已完成

| 提交 | 结果 |
| --- | --- |
| `3039fbe` | 前端 npm 审计改为解析官方 registry 的 JSON 报告，未知或过期漏洞仍阻断 CI；已知漏洞必须进入带到期日的显式策略 |
| `9e69f8b` | 测试依赖增加 `httpx2==2.16.2`，并补齐 `httpcore2`、`truststore` 的哈希锁，消除 Starlette TestClient 弃用提示 |
| `abff2b5` | 仓库卫生测试清理宿主注入的 `GIT_CONFIG_*` 环境变量，避免临时 Git 仓库测试受 Codex 宿主配置影响 |
| `f150bc4` | 生产仓库卫生检查脚本也隔离 `GIT_CONFIG_*`，并增加宿主配置回归用例 |
| `291ce25` | npm 审计策略要求版本号和非空原因，防止无说明的长期漏洞豁免 |
| `fd71d2c` | npm 审计策略严格校验整数版本和 GitHub Advisory URL 类型 |
| `f703044` | Python 工具链与测试锁升级 `certifi`、`filelock`、`setuptools`，四套锁文件审计均无已知漏洞；Markdown 报告页脚复用报告生成时间，避免连续导出跨秒产生不稳定内容 |
| `f8ac511` | 记录 Windows / Python 3.12 / Node 24 Quality CI 通过结果 |
| `5bba8bd` | `--scope all` 为 runtime、test、toolchain 分别写入最小可复现的 `CUSTOM_COMPILE_COMMAND`，并增加三次调用的回归校验 |
| `91d9578` | CI 的 Python 依赖兼容性检查显式调用隔离的测试解释器，确保 `pip check` 覆盖测试锁实际安装环境 |
| `b610d97` | 离线分析 CLI 对显式模型路径检查加载结果，失败时终止分析；`include_emotions` 和 `detailed` 只接受 JSON 布尔值，并补齐入口回归测试 |

依赖审计策略目前记录 8 个来自官方 npm 报告的 Advisory URL，其中包含 Vue 服务端渲染链路、`source-map-js` 和 Vite 8.3.2 开发工具链。所有条目到期日为 2026-12-31，升级前端依赖后应删除对应策略项并重新完成浏览器验收。

## 当前验证证据

| 检查 | 结果 | 证据边界 |
| --- | --- | --- |
| Python 完整回归 | 1789 passed；195 个 subtests；无警告 | 合成数据、契约和状态测试，不代表真实模型质量 |
| Starlette TestClient 定向回归 | 68 passed，无弃用警告 | 只验证测试客户端兼容性 |
| npm 审计策略 | 12 项策略测试通过；实际官方 registry 报告中的 8 个 Advisory 均被显式到期策略识别 | 允许列表只延迟治理，不修复漏洞 |
| Python 锁文件审计 | runtime、test、toolchain、installer 四套锁均报告 No known vulnerabilities found；维护环境 `pip check` 和生成器校验通过 | 只覆盖 PyPI 公布的漏洞，不能替代运行时行为验收 |
| 前端单元、构建和真实浏览器 E2E | 35 passed；构建通过；31 个 E2E 场景通过 | 本轮未修改前端，Vite 仍报告约 1.23 MB 主 JS chunk |
| 扩展单元和真实浏览器 E2E | 38 passed；10 个 E2E 场景通过 | 使用隔离 Chrome for Testing、临时数据库和合成密钥 |
| GitHub Actions | run `45696322091` / `91d9578` 的 Windows / Python 3.12 / Node 24 job 成功 | 远程 runner 结果与本地验证分开记录 |
| 仓库卫生与 Python 编译 | 通过 | 历史 `docs/audit-2026-09-24/` 仍未跟踪，未进入提交 |

## 仍未闭环

- 8 个 npm Advisory 仍等待前端依赖升级，当前按到期策略跟踪；
- 模型执行仍没有可强制终止的独立进程和硬内存预算，现有取消语义会等待线程完成以保护数据库和任务状态；
- 可选实验模型的 legacy 兼容层仍会直接加载本地 `pickle` 产物，当前没有模型上传 API；正式发行前应迁移到安全的权重格式，并把旧格式设为显式信任边界；
- 分类、情感、相似度和综合评分缺少独立人工标注基准、校准和真实模型资源验收，仍属于实验性能力；
- 前端主 JS 包体积约 1.23 MB，尚未做拆包或按需加载；
- 许可证、第三方通知、模型/数据来源证明、采集授权、保留周期和正式发行制品治理仍需权利人确认。

这些验证说明本地契约、生命周期和门禁行为，不等同于真实模型准确率、所有 Chrome 版本兼容性或维护者环境中的 GitHub CI 结果。
