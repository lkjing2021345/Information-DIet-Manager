# 最小运行环境的真实验收与 CI 门禁

2026-10-03，在 `5832ab35fef74eeec6c1a4ba6c1377fc5b3e1f26` 产品源码上新建 Windows CPython 3.12.5 AMD64 环境，先按 installer 哈希锁安装 pip 26.2.1，再按 runtime 哈希锁安装 14 个运行包。仅使用官方 PyPI wheel。独立维护环境新审计运行包和安装器，实际安装、锁与审计集合逐项一致，无跳过、无已知漏洞，pip check 通过。没有 pandas、NumPy、pytest、httpx、torch 或 transformers；用户包与 PYTHONPATH 已隔离。

现有前端 Playwright **31 场景**、扩展 **10 场景**在该环境全部通过。前端覆盖鉴权、分页、备份、删除、恢复和失败处理；扩展覆盖采集、持久队列和重新投递。原首页与右抽屉截图已检查。图表就绪数据仍含浏览器端合成响应，扩展异常 ACK 含 Node 替身，不能作为真实模型推理证据。浏览器工具仍使用独立锁定的 Node/Playwright 依赖；“最小 Python 环境”不意味着验收工具无需额外依赖。

另以新临时 SQLite 和真实 HTTP 保存 5 条近期合成记录，确保越过数据不足短路。基础统计正常；仪表盘明确返回 unavailable，完整模型分析返回安全 503 和 failed 任务。业务记录、向量、版本及既有基础统计历史保持，合法新增两个失败任务；随后健康检查与备份正常。未替换模型函数，没有下载模型，失败实际发生于缺少 pandas 的导入处，不扩大为各模型逐一故障测试。真实服务身份通过 PID、解释器、prefix 和 nonce 核对，launcher 与实际 worker 均正常退出 0，临时库已清理。

浏览器观察器确认本次观察到的后代进程均退出；只归档本次新报告和截图并记录 SHA-256。隔离浏览器合成资料暂留 ignored 目录供诊断，不是个人 profile。执行后再次核对包清单，14 个运行包与安装器保持不变。

CI 新增独立运行环境的哈希安装、兼容检查、新运行包审计及实际集合门禁，全部成功后才将两套浏览器的 `IDM_TEST_PYTHON` 切到该解释器。原测试环境和 PATH 仍负责 Python 全套及其它工具。两个浏览器步骤明确禁用用户包、清除 PYTHONPATH；没有改动产品逻辑、UI、依赖版本或第三方 action。新工作流尚须按推送后的精确提交核验，不能拿上述本地验收或上一提交 CI 代替。

独立复核检查了安装/审计集合、浏览器新产物与指纹、HTTP 探针和 CI 变更；未重复负载。证据留在 `output/playwright/quality-20261003/` 的 `runtime-acceptance/`、`runtime-browser-frontend/`、`runtime-browser-chrome-extension/` 与 `runtime-http-acceptance-result/`。这不补齐真实模型效果、长期并发、断电恢复或正式发布验收。
