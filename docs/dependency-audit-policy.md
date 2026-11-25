# npm 依赖审计策略

CI 使用官方 npm registry 生成前端依赖审计 JSON，再由
`scripts/check_npm_audit.py` 对照 `scripts/npm-audit-allowlist.json` 检查。

允许列表只用于记录已经确认、仍待升级的漏洞。策略必须声明
`schema_version: 1`，每一项必须包含可追踪的 GitHub Advisory URL、到期日期
和非空原因。未知漏洞、没有 Advisory URL 的异常、重复或已过期条目都会使
检查失败。允许列表不会修改依赖，也不会降低本地审计报告中的严重级别。

当前条目对应现有前端依赖中的 Vue 服务端渲染链路、`source-map-js` 和 Vite
8.3.2 开发工具链。它们的到期日是 2026-12-31。升级前端依赖后，应删除对应
条目并重新运行前端单元测试、构建和真实浏览器 E2E。

扩展依赖继续使用严格的 `npm audit` 门禁，因为当前没有允许列表条目。Python
依赖继续使用完整哈希锁和 `pip-audit` 检查。
