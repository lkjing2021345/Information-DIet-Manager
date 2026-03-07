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
│   ├── hyh/                        # 后端服务（FastAPI + SQLite）
│   │   ├── CONTRACT.md             # 接口与约定说明
│   │   ├── import_formats.md       # 数据导入格式说明
│   │   ├── ingest_item.schema.json # 数据结构定义（JSON Schema）
│   │   ├── models.py               # Pydantic 数据模型
│   │   ├── schema.sql              # 数据库表结构
│   │   ├── app.py                  # FastAPI 应用（API 路由）
│   │   ├── db.py                   # 数据库连接管理
│   │   └── utils.py                # URL/文本规范化与哈希
│   ├── lsj/                        # 本地数据抓取与分析
│   │   ├── requirements.txt        # 依赖库
│   │   └── src/
│   │       ├── main.py             # 主程序入口
│   │       └── utils/
│   │           └── fetch_data.py   # Chrome 本地历史记录抓取
│   └── chrome-extension/           # Chrome 浏览器插件
│       ├── manifest.json           # 插件清单（Manifest V3）
│       ├── background.js           # Service Worker：停留时长追踪与数据上传
│       ├── content.js              # Content Script：提取页面元信息与正文
│       ├── popup.html              # 插件弹窗界面
│       └── popup.js                # 弹窗交互逻辑（API 地址配置、开关）
```

## Chrome 插件集成说明
1. 用户安装 Chrome 插件后，插件会自动采集浏览信息（包括 url、标题、meta description、正文、停留时长等）。
2. 插件将采集到的数据通过 HTTP POST 上传到后端 API（`/collect`），数据格式遵循 `IngestItem` 契约。
3. 后端接收数据并存储到数据库，后续可通过 `/analyze/run` 和 `/dashboard/summary` 进行信息茧房程度分析。

### 安装 Chrome 插件
1. 启动后端服务：`python src/hyh/app.py`
2. 打开 Chrome 浏览器，进入 `chrome://extensions/`
3. 打开右上角 **开发者模式**
4. 点击 **加载已解压的扩展程序**，选择 `src/chrome-extension/` 目录
5. 插件安装完成后，点击工具栏图标可配置后端 API 地址和采集开关
6. 正常浏览网页即可，插件会自动采集并上传浏览数据

### 插件工作原理
- **background.js (Service Worker)**：监听标签页切换和关闭事件，计算每个页面的停留时长，将数据发送到后端 `/collect` 接口
- **content.js (Content Script)**：在页面加载完成后提取 meta description 或正文摘要（截取前 2000 字符），发送给 Service Worker
- **popup.html / popup.js**：提供简单的配置界面，用户可设置后端 API 地址和启用/禁用数据采集
- 忽略停留时间小于 2 秒的页面访问，以减少噪声数据
- 仅采集 HTTP/HTTPS 页面，不采集浏览器内部页面

## 技术栈
- Python（后端与数据分析）
- Chrome Extension（前端数据采集）
- RESTful API（数据上传与交互）
- 数据库（结构见 schema.sql）

## 快速开始
1. 安装后端依赖：
   ```cmd
   pip install fastapi uvicorn pydantic
   ```
2. 启动后端服务：
   ```cmd
   python src/hyh/app.py
   ```
3. 安装 Chrome 插件：
   - 打开 `chrome://extensions/`，开启开发者模式
   - 点击"加载已解压的扩展程序"，选择 `src/chrome-extension/` 目录
4. 浏览网页，插件自动采集并上传数据
5. 访问 `http://127.0.0.1:8000/docs` 查看 API 文档
6. 调用 `POST /analyze/run` 运行分析，调用 `GET /dashboard/summary` 查看信息茧房评估结果

## 安全与隐私
- 插件采集内容仅用于信息多样性分析。本地开发默认使用 HTTP，生产环境建议配置 HTTPS 以加密传输。
- 用户可随时卸载插件或通过弹窗界面关闭数据上传。

## 贡献与开发
- 欢迎提交 issue 或 pull request。
- 新增插件或分析模块请遵循模块化设计原则。
