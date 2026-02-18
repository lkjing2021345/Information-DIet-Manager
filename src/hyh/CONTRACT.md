2# 信息饮食管理器 - 最小可用数据契约 (v0.1)

目标：前端/插件/导入统一一个最小可用的短文本 Payload，便于后端先跑通统计、去重与情绪分析。

## 1. 核心 Payload
**资源：** `IngestItem`

字段：
`url` string, **必填**  
约束：合法 URL，允许重复但用于去重与聚类。建议长度 ≤ 2048。

`title` string, **必填**  
约束：标题或内容摘要。建议长度 ≤ 300。

`text` string, **可选**  
约束：description 或正文截断；允许空字符串或省略。建议长度 ≤ 2000。

`ts` integer, **必填**  
约束：访问时间的 Unix 毫秒时间戳。示例：`1708243200000`。

`source` enum, **必填**  
取值：`plugin` | `import`

**示例：**
```json
{
  "url": "https://example.com/news/123",
  "title": "AI 相关政策新动向",
  "text": "政策文件摘要或正文截断...",
  "ts": 1708243200000,
  "source": "plugin"
}
```

## 2. 建议补充字段（后续版本）
这些不是 MVP 必需，但建议预留在接口层，方便扩展：

`lang` string，可选  
例：`zh-CN`、`en-US`。用于情绪与分词策略。

`channel` string，可选  
例：`short_video` / `news` / `social`。用于统计占比。

`author` string，可选  
作者或账号名。

`tags` string[]，可选  
内容标签或主题。

`meta` object，可选  
承载插件端的额外字段（如 `platform`, `duration`, `likes`）。

## 3. 去重与相似度建议
后端建议预留两个内部字段（不要求客户端上传）：

`url_hash`：对 `url` 规范化后再 hash，用于强去重  
`content_hash`：对 `title + text` 规范化后 hash，用于弱去重

## 4. 错误与校验
推荐的最小校验：
1. `url` 非空且可解析
2. `title` 非空
3. `ts` 为合理时间范围（允许前后 1 年，避免异常数据）
4. `source` 必须在枚举内

## 5. 简要接口约定（建议）
这些是建议的 API 形状，便于前后端对齐：

`POST /v1/items`  
请求体：`IngestItem` 或 `IngestItem[]`  
响应体：`{"inserted": n, "duplicates": m}`

`POST /v1/import`  
请求体：`multipart/form-data`，支持 `csv`/`jsonl`  
响应体：`{"inserted": n, "duplicates": m, "failed": k}`

`GET /v1/stats?from=...&to=...`  
响应体：占比统计、情绪分布、重复率等

`GET /v1/insights?from=...&to=...`  
响应体：建议文案列表（如“信息过载”）

