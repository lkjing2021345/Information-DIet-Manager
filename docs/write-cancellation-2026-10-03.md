# 同步写入接口的取消归属（2026-10-03）

## 确证范围

在先前恢复接口的修复之后，继续用临时 SQLite 数据库和真实 ASGI 请求核实，以下六个接口仍存在同类问题：

- `POST /collect`：采集写入。
- `POST /import`：上传解析及导入。
- `DELETE /items/{item_id}`：单条删除及派生结果清除。
- `DELETE /data`：全部删除。
- `POST /analyze/run`：全库可信统计及持久化。
- `GET /dashboard/summary`：虽然是 GET，仍会更新 `stats_daily`；缓存未命中时还会插入分析记录并按现有上限补齐 embedding。

探针在真实事务提交前暂停工作线程，再对 ASGI 请求任务执行 `asyncio.Task.cancel()`。六个接口均出现“请求已结束、工作线程未结束、操作锁已释放、可用请求名额恢复到 4”的状态，释放线程后仍有数据库提交。summary 分别验证了缓存未命中和命中：前者执行两条 INSERT，后者仍执行一条统计表 UPSERT；它不是纯读路径。此探针没有把普通浏览器断连等同于服务器取消任务。

导入另有文件生命周期风险：取消原同步端点后，FastAPI 可以关闭上传文件，尚未执行 `read()` 的工作线程仍持有该文件对象。修复前的内存上传回归已直接观察到这个状态。

## 最小改动

六个路由改为异步薄包装，通过既有 `run_owned_sync` 等待同步工作；不在中央 ASGI 层屏蔽整个请求的取消。

- 采集、导入和两类删除的原有逻辑移入小型同步 helper。
- 删除的确认头仍在路由中验证，Request 不跨线程传递。
- 两个统计路由直接调用已有 `_global_statistics`，参数不变；summary 保留 `backfill_limit=2000`。
- 导入将读上传文件、解析、校验和写库保留在同一个受管理的 worker 中。异步包装完成之前，FastAPI 的外层文件清理不会先行退出；项目没有另行抢先关闭上传文件。

原参数类型、Query 范围、UploadFile 必填规则、响应模型、重复 URL 行为和事务边界保持不变。仓库 `src/`、`scripts/` 的调用搜索未发现依赖这些端点函数同步返回的直接调用者。这里的取消语义仍然是等待已授权的工作完成：成功则提交，工作线程异常则回滚，不承诺取消请求能够撤销已经开始的写入。

## 验证

`src/hyh/tests/test_write_cancellation.py` 新增 36 项回归，全部使用临时数据库及合成 multipart 文件：

- 六路由，其中 summary 分缓存命中 / 未命中，共 7 种场景；分别验证普通完成或重复取消、事务成功或故障回滚，共 28 项。
- 真实上传文件分别保留在内存或超过阈值滚落临时磁盘文件；各自验证普通完成 / 重复取消与读入成功 / 故障，共 8 项。
- 每次取消后检查请求仍未退出、操作锁仍占用、可用请求名额仍为 3；事务结束后检查释放、原子提交 / 全表回滚，以及后续写入可用。
- 上传测试检查读线程未退出时文件仍打开，退出后才由 FastAPI 关闭；保留正常成功和安全错误响应。

修复前：**18 failed，18 passed**。取消相关场景全部失败，正常成功 / 故障路径通过。最小路由修复后：**36 passed**。

随后在 Windows CPython 3.12 隔离锁定环境中合并运行本文件，以及 `test_ingestion_contract.py`、`test_legacy_statistics.py`、`test_query_bounds.py`、`test_local_security_and_data.py`、`test_restore_cancellation.py`、`test_export_delivery.py`：**259 passed**。一条既有 Starlette/httpx TestClient 弃用警告。该组覆盖原输入与响应合同、可信统计、安全及恢复流程、导出请求生命周期；真实浏览器验收另行报告。

本组测试不验证真实模型推理，也不单独证明 Uvicorn 超时停服安全。服务级工作登记、停止接收与 lifespan drain 的修改及证据由独立生命周期测试负责；本说明只描述各写入请求对事务和上传文件的归属。
