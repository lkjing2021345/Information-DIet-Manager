# 模型缓存与环境配置修复（2026-10-03）

## 确认的问题与本次行为

`sentiment.py` 原先在导入时将 `HF_ENDPOINT` 无条件改为镜像地址。两个训练模块原先在导入时创建写死的 `C:\Users\Administrator\.cache\huggingface\hub` 目录，并覆盖 `HF_HOME`、`HUGGINGFACE_HUB_CACHE`、`TRANSFORMERS_CACHE`。分类训练模块还在改环境之前就导入了 Transformers，配置生效顺序不可靠。

本次删除上述环境改写和导入时的 Hugging Face 缓存目录创建。两个训练模块在模型依赖导入前通过纯标准库 helper 解析缓存位置，优先级如下：

1. `HF_HUB_CACHE`。
2. 兼容旧名称 `HUGGINGFACE_HUB_CACHE`。
3. `HF_HOME/hub`。
4. `XDG_CACHE_HOME/huggingface/hub`。
5. 当前用户 `~/.cache/huggingface/hub`。

路径按上游顺序分两阶段展开 `~` 与环境变量：先展开 HF home，再展开最终 hub 路径。支持 Windows `%变量%`、`$变量` / `${变量}`、中文及 emoji。优先级及导入前设置环境的要求见 [Hugging Face 官方环境文档](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)；两阶段展开核对了 [huggingface_hub v1.4.1 官方 constants.py](https://github.com/huggingface/huggingface_hub/blob/v1.4.1/src/huggingface_hub/constants.py#L116-L140)。

只有明确调用 `configure_huggingface_cache()` 或非本地模型的 `ensure_model_cached()` 才创建所选 hub 内的 `persistent_models` 目录。已有本地模型路径直接返回；完整快照继续复用；显式下载和 Transformers 备用路径继续接收同一 `cache_dir`。目录不可创建时向调用者报告错误，不改写环境或静默转存到别处。

设置需在启动进程或导入模型模块之前完成。代码不再默认切换镜像；如需镜像，用户自行预先设置 `HF_ENDPOINT`。已有 endpoint、离线选项、认证及其他 Hugging Face 环境配置均不由本次代码改写。`TRANSFORMERS_CACHE` 不被改写；训练函数显式传入的 `cache_dir` 按上述 Hub 设置解析。

## 验证与边界

在 Windows CPython 3.12 的隔离锁定测试环境执行：

```text
python -m pytest src/lsj/tests -q --tb=short -p no:cacheprovider
59 passed
```

其中新增 27 项模型配置测试，覆盖所有优先级、默认用户目录、变量与波浪号的组合展开、中文 / emoji、三个真实模块的导入接线、保留缺省或指定 endpoint、导入不创建 HF 缓存、显式创建、错误目录、本地路径、快照复用以及 Transformers 备用接线。可选模型依赖与 logger 由桩替代，文件全部位于 pytest 临时目录。既有 32 项 lsj 测试也通过。`git diff --check` 通过，Git 仅提示既有 Windows 换行转换策略。

本次没有读取个人模型缓存或凭证文件，没有下载模型，也没有进行真实模型训练、推理或网络镜像验收。原有 logger 在真实导入时仍可能创建日志文件；因此本次仅证明项目的 HF 缓存配置不再产生上述导入副作用，不宣称整个训练模块导入完全没有文件写入。训练算法、模型质量、模型版本固定与产物完整性校验仍需独立验收；合成快照仅验证既有文件发现和参数传递。
