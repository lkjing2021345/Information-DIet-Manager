# Information-DIet-Manager
2Manage the information you take in every day just like you manage your diet.

You can define your own database location by setting the IDM_DB_PATH environment variable, otherwise the database location will be created in this directory.

## 项目简介

Information Diet Manager 是一个用于评估用户“信息茧房”程度的工具。它支持多种数据采集方式，包括网站后端数据抓取和 Chrome 浏览器插件自动采集用户浏览信息。通过分析用户的浏览行为和内容分布，帮助用户了解自身的信息摄入多样性和偏向性。

## 主要功能
- 数据模型与数据库结构定义（src/hyh/）
- 主程序与数据分析逻辑（src/lsj/src/main.py）
- 数据抓取工具（src/lsj/src/utils/fetch_data.py）
- Chrome 插件采集用户浏览信息（url、标题、meta、正文、停留时长等）
- 后端 API 接收并存储浏览数据
- 信息茧房程度评估与分析

## 项目结构
```
Information-DIet-Manager/
├── README.md
├── src/
│   ├── hyh/
│   │   ├── CONTRACT.md           # 接口与约定说明
│   │   ├── import_formats.md     # 数据导入格式说明
│   │   ├── ingest_item.schema.json # 数据结构定义
│   │   ├── models.py             # 数据模型
│   │   └── schema.sql            # 数据库结构
│   └── lsj/
│       ├── requirements.txt      # 依赖库
│       └── src/
│           ├── main.py           # 主程序入口
│           └── utils/
│               └── fetch_data.py # 数据抓取工具
```

## Chrome 插件集成说明
1. 用户安装 Chrome 插件后，插件会自动采集浏览信息（包括 url、标题、meta description、正文、停留时长等）。
2. 插件将采集到的数据通过 HTTPS POST 上传到网站后端 API（如 `/api/ingest_chrome_data`）。
3. 后端接收数据并存储到数据库，后续可进行信息茧房程度分析。

## 技术栈
- Python（后端与数据分析）
- Chrome Extension（前端数据采集）
- RESTful API（数据上传与交互）
- 数据库（结构见 schema.sql）

## 快速开始
1. 安装依赖：
   ```cmd
   pip install -r src/lsj/requirements.txt
   ```
2. 启动后端服务：
   ```cmd
   python src/lsj/src/main.py
   ```
3. 安装 Chrome 插件（详见插件目录及安装说明）
4. 浏览网页，插件自动采集并上传数据
5. 在网站端查看和分析信息茧房评估结果

## 安全与隐私
- 插件采集内容仅用于信息多样性分析，传输过程采用加密。
- 用户可随时卸载插件或关闭数据上传。

## 贡献与开发
- 欢迎提交 issue 或 pull request。
- 新增插件或分析模块请遵循模块化设计原则。
