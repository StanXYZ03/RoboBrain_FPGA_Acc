# SpatialEval VQA 外部确认适配准备报告

状态：`ADAPTER_DRAFT_READY`，不是预注册，也不得运行正式外部确认。

## 选择依据

SpatialEval 发表于 NeurIPS 2024，覆盖空间关系、位置理解、计数和导航。官方
Hugging Face 数据卡给出的 VQA test 为 4635 条单图样本，官方仓库使用 exact
match 评价。它比含多图与非空间任务的完整 BLINK 更贴合当前一图一问的 Selector
输入合同，也未被当前 CV-Bench 发展流程用于调参。

## 本次完成

- 增加只读取显式本地 parquet 的离线 manifest 构建器；使用 `pyarrow`，不依赖
  Hugging Face `datasets` 包，不联网。
- 增加严格的 answer-only exact-match 草案指标和 CPU 夹具测试。
- 增加 `DRAFT_NOT_PREREGISTERED` 配置；所有数据、模型、代码和输出哈希仍为空，
  `must_not_run=true`。

草案指标不是 SpatialEval 官方仓库针对不同模型编写的启发式解析器。真实 schema、
Prompt、期望答案字段和官方 evaluator 未审计前，不得把该草案分数称为官方结果。

## 当前阻塞

- 数据尚未下载，任何真实样本和标签均未查看，GPU 推理次数为 0。
- 2026-10-09 检查时 `/home` 只剩约 950 MB；官方 VQA 数据卡给出的下载量约
  712 MB、展开后约 733 MB，缓存和导出图片还需要额外空间。
- `127.0.0.1:7891` 代理当时未运行，服务器无法访问 Hugging Face 和 DeepSeek API。
- 官方数据卡和代码仓库未明确给出数据集许可。得到作者或权威许可文件确认前，
  不应下载后再分发，也不应将其冻结为论文正式外部集。

## 进入正式预注册的条件

先恢复至少 2 GB 可用空间和网络，确认许可，下载并固定 parquet SHA256；随后仅做
schema/完整性审计，冻结全部 4635 条样本、Prompt、解析器、三臂代码、模型身份、
bootstrap 种子和输出路径。完成这些步骤后，配置才可升级为
`PREREGISTERED_NOT_EXECUTED`，并只执行一次三臂外部确认。
