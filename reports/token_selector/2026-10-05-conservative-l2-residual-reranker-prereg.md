# 保守修正 L2 主干 + 有界整数任务残差重排 v1 预注册报告

- 机制：`conservative_l2_residual_rerank_v1`
- 状态：**`PREREGISTERED_NOT_EXECUTED`**（新臂输出尚未生成，`executed=false`）
- 口径：**仅发展性实验**；不是论文确认、不是可部署性结论、不是 FPGA 时延/加速主张。
- 本轮：CPU-only 设计、实现、聚焦测试与预注册；`gpu_runs=0`；无模型权重、无网络、无训练、无依赖安装。
- 预注册 config：`configs/2026-10-05-conservative-l2-residual-reranker-full2638-development.preregistered.json`
  - SHA256：`6b158d22d52f1df66eba75b1b7bb4c7c614793681152c1501c43079fecd97512`

## 1. 背景与必须保留的修正

| 证据 | 数值 |
| --- | --- |
| Dense BF16 官方 CV-Bench | `0.8727298774` |
| 修正整数 L2 B1 K96 | `0.7668517265`（dense 的 87.8682%） |
| 既往 full-score Prompt + 修正 L2 整数融合 | `0.777362`（dense 的 89.0725%） |
| 该融合相对修正 L2 的配对增益 | `+0.010510`，95% 配对 bootstrap CI `[-0.005430,+0.025699]`（**未证明** Prompt 增益） |

- full CV-Bench 2638 已被反复观测，**只能作为发展性证据，永远不是未触碰确认集**。
- 当前全尺寸输入的精确压缩布局是**动态 87 real + 9 context**；3x3 的 L2 anchors 是这 87 个 real 分组的子集。
- `local_candidate_cap=16` 是**每区域候选上限**，不是交换次数；不得混用。
- 直接绝对梯度全替换 teacher **永久 STOPPED**；本轮不在 24 样本 answer-margin pilot 上调任何参数。

## 2. 唯一机制合同

1. 基线 real 选择来自既有修正整数 L2 K96 路径：每个非空 3x3 区域一个 L2 anchor，并保留既有动态 context 预留。
2. 所有 L2 anchor 冻结，**从不替换**。
3. 每个区域保留其被选 real 分组数量，因此每次替换都是**同区域一对一交换**，不会消耗该区域预留的未选 context 分组。
4. 有符号任务残差严格定义为 `semantic_norm_q15 - corrected_l2_norm_q15`：语义项来自已接受的整数语义链（seed-20261003 `Wt`/`Wv_q` Q0.15），L2 项来自修正整数 patch-L2 Q0.15。**不使用**标签、答案 token、梯度、浮点 lambda、学习到的新权重、task/source ID。
5. 在每个区域内，将当前被选的非 anchor 分组与未选候选分组配对，只取**严格正**残差增益；候选枚举受既有 `local_candidate_cap=16` 限制；平局按**更小原始分组索引**确定性打破。
6. 全局至多应用 `S=16` 个最高正增益、互不冲突的同区域交换。若合法正交换少于 16 个，就少做；每次一对一，因此 **K 恒为 96**。`S=16` 是一个独立的固定硬件工作量子，**不是**从已停止的 24 样本 pilot 推断，也**不是** local candidate cap。
7. 用既有布局合同重建 87 real + 9 context。**context pooling 权重仍基于修正整数 L2**，因此 Prompt 信息只影响有界的残差交换。
8. 任何违约 fail closed。**不引入**对 `S`、`K`、cut 点、anchor、公式、阈值或分数混合的扫描。

## 3. 87 real + 9 context 的含义

本轮 full-size 输入的合并网格为 3x3=9 个非空区域，每个区域成员数 ≥2，因此动态 context 数 `C=9`；real 预算 `K-C=87`。其中每个区域恰好 1 个 L2 anchor（共 9 个 anchor，均为 87 个 real 的子集），其余 78 个 real 为两阶段有界 Top-K 明细组。identity bypass（`G<=K`）仍然合法，单独计数，不参与残差交换。

## 4. 发展性判定门（只有全满足才 GO）

输出二值之一：`DEVELOPMENT_GO_EXTERNAL_VALIDATION` 或 `DEVELOPMENT_STOP`。

1. 2638 个唯一样本全部完成且输出可解析；
2. 每个压缩样本满足精确 K96 与冻结 real/context/布局合同；合法 `G<=K` identity bypass 单独计数；
3. 官方 CV-Bench 分数 `>= 0.78545688966`（dense BF16 的 90%）；
4. 点估计不差于最强同预算发展基线 `0.777362`；
5. 新臂减修正整数 L2 的配对 bootstrap 95% CI 下界 `>= 0.0`（5000 次，seed `20261005`，按 `task::source_dataset` 分层）；
6. 无禁用标签/答案/task/source 泄漏。

通过仅是**获取并运行一个新的、未触碰的外部空间推理基准**的许可；**不是**官方确认、可部署性、FPGA 时延或论文精度主张。看到 full2638 后**不得**再调 v1。

## 5. 运行器与冒烟

- runner：`evaluation/run_hf_conservative_l2_residual_rerank.py`
- 固定：本地 BF16/SDPA RoboBrain2.5-4B、确定性 greedy、`max_new_tokens=16`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`；每个计时段用 `torch.cuda.synchronize` 括起；模型加载时间不计入稳态计时；raw 逐样本 JSONL + 安全 fail-closed 续跑。
- `--smoke` 为**合同冒烟**：六单元（ADE20K/COCO × Count/Relation + Omni3D × Depth/Distance）各取 1 个确定性样本；拒绝写入 full2638 预注册输出路径；**不得**用于质量判定。

完整 2638 命令（宿主、尚未执行）：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python evaluation/run_hf_conservative_l2_residual_rerank.py \
  --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B \
  --model-label RoboBrain2.5-4B-BF16 \
  --manifest datasets/cv_bench_full2638.jsonl \
  --output artifacts/token_selector/2026-10-05-conservative-l2-residual-reranker-full2638/conservative_l2_residual_rerank_v1_k96_full2638.jsonl \
  --budget 96 --region-grid 3 --local-candidate-cap 16 --max-swaps 16 \
  --dtype bf16 --device cuda:0
```

汇总判定（新臂缺失时 fail closed、退出码 2、不写报告）：

```bash
python evaluation/summarize_conservative_l2_residual_rerank_full2638.py
```

## 6. Pins

本预注册在写 config 时对 **30** 个输入完成现场 SHA256 复核（`pins_verified=true`），包括：
full manifest、dense、corrected-L2 float/integer、prompt+corrected-L2、旧 current integer、
修正 L2 full2638 正式报告与 config、机制/集成/runner/汇总器/4 个新聚焦测试、
既有 token selector / B1 integration / 修正 L2 core / runner / evaluator / metrics /
current_selector_integer / 两个 audit 模块 / `evaluation_protocol.json`、
以及 3 个部署 checkpoint（projection npz、Wv int8、Wv 逐张量 scale）。
新臂输出 SHA256 为 `null`、状态 `TO_BE_GENERATED`。完整 pin 表见机器可读报告
`reports/token_selector/2026-10-05-conservative-l2-residual-reranker-prereg.json`。

## 7. 限制

- full2638 是反复观测过的发展集，不是未触碰确认集。
- 通过只是外部验证许可，不是论文/可部署/FPGA 时延主张。
- Selector 路径是软件桥；不做 FPGA/板卡/时延/加速声明。
- 看到 full2638 后不得调参 v1。
- 新臂输出尚不存在，状态 `PREREGISTERED_NOT_EXECUTED`。
