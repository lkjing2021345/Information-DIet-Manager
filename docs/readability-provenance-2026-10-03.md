# Readability 来源证据（2026-10-03）

`chrome-extension/readability.js` 的 Git LF 文件与 Mozilla Readability 在不可变提交 [`d7949dc47dd9ed9ee1d3b34ffdcf3bce28cde435`](https://github.com/mozilla/readability/commit/d7949dc47dd9ed9ee1d3b34ffdcf3bce28cde435) 下的 [`Readability.js`](https://raw.githubusercontent.com/mozilla/readability/d7949dc47dd9ed9ee1d3b34ffdcf3bce28cde435/Readability.js) 原始字节完全一致。核查日期为 2026-10-03；本说明沿用当日已保存的官方文本和逐字节比较结果，不执行第三方脚本，不安装包。

| 已核对对象 | 字节数 | SHA-256 |
| --- | ---: | --- |
| Git LF 文件 / 上述上游原始文件 | 90944 | `e9330028c8a5a4aa7d75147be2605d520f7f213c7b28474947dc0e9c984e9bed` |
| 本机 Windows CRLF 工作文件 | 93756 | `ffadf8434be4558c87edbe25c9a4cf740ae7a7d6c687c7471a216d805b0ae05d` |

CRLF 工作文件与上游原始字节不同；仅将 CRLF 转为 LF 后才完全相同，未作其他空白、编码或内容归一化。有限的本地文件历史中，首次加入（`83f63e5`）、删除后恢复（`d65a837`）与核查时 Git 文件的字节相同；提交信息未记录下载地址。这证明文件内容对应关系，不能证明原作者实际使用的下载渠道，也不能证明只有这一个上游提交包含相同文件。

精确发布版本保持未知。核查时官方返回的 8 个 tag 对应文件均未与本地文件完全匹配，包括 [`0.6.0` 对应源文件](https://raw.githubusercontent.com/mozilla/readability/04fd32f72b448c12b02ba6c40928b67e510bac49/Readability.js)。匹配提交的 [`package.json`](https://raw.githubusercontent.com/mozilla/readability/d7949dc47dd9ed9ee1d3b34ffdcf3bce28cde435/package.json) 虽声明 `0.6.0`，不能据此将该快照称为正式 0.6.0 release。文件头 Arc90 `(1.7.1)` 描述代码祖先，也不是当前快照版本。

## 库存中的记录与保护

`chrome-extension/readability.provenance.json` 保存上述提交、不可变 URL、两种字节指纹及核查日期，作为第 14 个固定输入。库存 schema 2 / generator 1.1.0 离线校验当前文件与该元数据的关系，再输出 `source_match`；其中 `method` 为 `byte-identical` 或 `crlf-only`，`upstream_package_version_declaration` 仅保留上游声明，`release_version` 为 null。`version`、`verified_source` 与 `verified_license` 仍为 null，实际源码和元数据的原始 SHA-256 均保留。

未匹配的内容、缺失或非法元数据在写入库存前失败，不覆盖已有产物。允许的换行变体是显式记录的 LF / CRLF，不接受任意格式化或编码变化。来源元数据自身也参与输入指纹，并受禁止覆盖输入文件的输出路径检查保护。

这是以受审查元数据为依据的离线一致性检查，不是签名认证。修改源码时必须重新核实、审查并更新元数据；生成器不会联网判断新填入的提交或指纹是否真实。两次确定性生成与 CI 附件也不能单独证明作者下载渠道、完整软件供应链或构建二进制来源。

## 许可证与验收边界

文件头保留 Arc90 版权、Apache-2.0 声明和链接；匹配提交的 package 元数据同样声明 Apache-2.0。该提交的 [`LICENSE.md`](https://raw.githubusercontent.com/mozilla/readability/d7949dc47dd9ed9ee1d3b34ffdcf3bce28cde435/LICENSE.md) 是 553 字节的简短声明与链接，不是完整许可证正文；[当时根目录](https://api.github.com/repos/mozilla/readability/git/trees/d7949dc47dd9ed9ee1d3b34ffdcf3bce28cde435) 没有 `NOTICE`。核查前项目没有另附完整 Apache 许可证正文；本批也不复制许可证或决定项目自身许可证。来源匹配不表示完成第三方分发材料或法律义务审核。

冻结实现的库存专项为 **32 passed / 151 subtests，5.49 秒**；`scripts/tests` 联动为 **63 passed / 181 subtests，12.29 秒**。覆盖 LF / CRLF 合法输入、源码变化拒绝、元数据字段和双指纹关系、元数据自身指纹、禁止覆盖输入及旧输出保留；合成输入测试不复制整份第三方脚本为 fixture。

本批验收仅针对固定输入、元数据校验、字节对应关系与生成失败保护；不加载真实模型或访问个人数据，不将本地测试视为精确提交 CI、完整 SBOM 或企业发布认证。独立复核、全项目回归与最终提交 CI 证据由本轮总报告另行列明。
