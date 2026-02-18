# 文件导入格式 (CSV / JSONL)

目标：让导入文件与 `IngestItem` 一一对应，字段命名保持一致，避免二次映射。

## JSONL
每行一个 JSON 对象，必须包含 `url` / `title` / `ts` / `source`。

**示例：**
```json
{"url":"https://example.com/a","title":"A","text":"...","ts":1708243200000,"source":"import","channel":"news"}
{"url":"https://example.com/b","title":"B","ts":1708243500000,"source":"import","tags":["ai","policy"]}
```

## CSV
编码：UTF-8  
首行表头：字段名与 `IngestItem` 一致（小写、下划线）  
推荐分隔符：`,`

**推荐字段（最少 4 个）：**
`url,title,text,ts,source`

**可选字段：**
`lang,channel,author,tags,meta`

**tags 规则：**
1. 允许使用 JSON 数组字符串，如 `["ai","policy"]`
2. 或使用 `|` 分隔，如 `ai|policy`

**meta 规则：**
1. 允许 JSON 字符串，如 `{"platform":"bilibili","duration":95}`
2. 若为空，留空即可

**示例：**
```csv
url,title,text,ts,source,channel,tags,meta
https://example.com/a,AI政策更新,摘要...,1708243200000,import,news,"ai|policy","{""platform"":""weibo""}"
```

