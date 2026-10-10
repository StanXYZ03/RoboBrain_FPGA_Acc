# L2 守卫整数重排 v2 预注册报告

- 机制：`conservative_l2_guarded_rerank_v2`
- 状态：**`PREREGISTERED_NOT_EXECUTED`**（新臂输出尚未生成，`executed=false`）
- 口径：**仅发展性实验**；不是论文确认、不是可部署性结论、不是 FPGA 时延/加速主张。
- 本轮：CPU-only 设计、实现、聚焦测试与预注册；`gpu_runs=0`；无模型权重、无网络、无训练、无依赖安装。
- 预注册 config：`configs/2026-10-06-conservative-l2-guarded-reranker-v2-full2638-development.preregistered.json`
  - SHA256：`7499b6ea00939c58506e73b496c1c62bed08162dc8d6d535229f2ffa703e74eb`
- 前身 v1：`conservative_l2_residual_rerank_v1` → **`PREEXECUTION_DESIGN_STOP`**（执行前被独立复核阻断，从未运行、从未修补；v1 源/配置/报告字节不变）。

## 1. v1 阻断缺陷与 v2 修正

v1 的交换排序量 `(semantic_j - L2_j) - (semantic_i - L2_i)` 展开为 `semantic_gain + l2_loss`，
符号与“保护修正整数 L2”相反。反例：`semantic_i=semantic_j=10000`、`L2_i=30000`、`L2_j=1000`
→ v1 gain `29000>0` 错误接受；v2 `legal_gain = 0 - 29000 = -29000 < 0` 正确拒绝。

v2 唯一修正量：

- `utility_q16[g] = semantic_q15[g] + l2_q15[g]`，范围 `[0, 65534]`，加宽 `int64`；
- `semantic_gain = semantic_q15[j] - semantic_q15[i]`；
- `l2_loss = l2_q15[i] - l2_q15[j]`；
- `legal_gain = semantic_gain - l2_loss = utility_q16[j] - utility_q16[i]`；
- 仅当 `legal_gain > 0` 才允许交换。

## 2. 背景与必须保留的修正

| 证据 | 数值 |
| --- | --- |
| Dense BF16 官方 CV-Bench（精确 pin） | `0.8727298774444869` |
| 90% dense 门（计算 `0.9*dense`，非手写近似） | `0.7854568897000382` |
| 修正整数 L2 B1 K96 | `0.7668517265` |
| 既往 Prompt + 修正 L2 整数融合（**仅描述**） | `0.7773617325561999` |

- 先验 Prompt 融合分**不是**门，因为 90%-dense 门已隐含它。
- full CV-Bench 2638 已被反复观测，**只能作为发展性证据，永远不是未触碰确认集**。
- 当前全尺寸输入的精确压缩布局是**动态 87 real + 9 context**；3x3 的 L2 anchors 是这 87 个 real 分组的子集。
- `local_candidate_cap=16` 是**每区域候选上限**，不是交换次数；不得混用。
- 直接绝对梯度全替换 teacher **永久 STOPPED**；本轮不在 24 样本 answer-margin pilot 上调任何参数。

## 3. 唯一机制合同

1. 基线 real 选择来自既有修正整数 L2 K96 路径：每个非空 3x3 区域一个 L2 anchor，并保留既有动态 context 预留。
2. 所有 L2 anchor 冻结，**从不替换**。
3. 每个区域保留其被选 real 分组数量，因此每次替换都是**同区域一对一交换**，不会消耗该区域预留的未选 context 分组。
4. 每区域未选候选按 `utility_q16` **降序**枚举，平局取**更小原始分组索引**，受既有 `local_candidate_cap=16` 限制。
5. 只接受 `legal_gain > 0`；全局按 `legal_gain` 降序、再按 `(added_group_id, removed_group_id, region_id)` 升序排序；至多应用 `S=16` 个互不冲突交换。若合法正交换少于 16 个，就少做；每次一对一，因此 **K 恒为 96**。`S=16` 是独立固定硬件工作量子，**不是** local candidate cap。
6. 用既有布局合同（复用 v1 已验证的 `rebuild_hardware_selection`）重建 87 real + 9 context。**context pooling 权重仍基于修正整数 L2**，因此 Prompt 信息只影响有界合法交换。
7. **不使用**标签、答案 token、梯度、浮点 lambda、学习到的新权重、task/source ID；不引入对 `S`、`K`、cut 点、anchor、公式、阈值或分数混合的扫描。任何违约 fail closed。

## 4. 87 real + 9 context 的含义

本轮 full-size 输入的合并网格为 3x3=9 个非空区域，每个区域成员数 ≥2，因此动态 context 数 `C=9`；real 预算 `K-C=87`。其中每个区域恰好 1 个 L2 anchor（共 9 个），均为 87 个 real 的子集。identity bypass（`G<=K`）仍然合法，单独计数，不参与交换。

## 5. 修正后的发展性判定门（只有全满足才 GO）

输出二值之一：`DEVELOPMENT_GO_EXTERNAL_VALIDATION` 或 `DEVELOPMENT_STOP`。

1. 2638 个唯一样本全部完成且输出可解析；
2. 每个压缩样本满足精确 K96 与冻结 real/context/布局合同；合法 `G<=K` identity bypass 单独计数；
3. 官方 CV-Bench 分数 `>= 0.7854568897000382`（精确 `0.9*dense`，计算存储）；
4. 新臂减修正整数 L2 的配对 bootstrap 95% CI 下界 `>= 0.0`（5000 次，seed `20261006`，按 `task::source_dataset` 分层）；
5. 无禁用标签/答案/task/source 泄漏。

通过仅是**获取并运行一个新的、未触碰的外部空间推理基准**的许可；**不是**官方确认、可部署性、FPGA 时延或论文精度主张。看到 full2638 后**不得**再调 v2。

## 6. 运行器与冒烟

- runner：`evaluation/run_hf_conservative_l2_guarded_rerank.py`
- 固定：本地 BF16/SDPA RoboBrain2.5-4B、确定性 greedy、`max_new_tokens=16`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`；每个计时段用 `torch.cuda.synchronize` 括起；模型加载时间不计入稳态计时；raw 逐样本 JSONL + 安全 fail-closed 续跑。
- `--smoke` 为**合同冒烟**：六单元（ADE20K/COCO × Count/Relation + Omni3D × Depth/Distance）各取 1 个确定性样本；拒绝写入 full2638 预注册输出路径；**不得**用于质量判定。

完整 2638 命令（宿主、尚未执行）：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python evaluation/run_hf_conservative_l2_guarded_rerank.py \
  --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B \
  --model-label RoboBrain2.5-4B-BF16 \
  --manifest datasets/cv_bench_full2638.jsonl \
  --output artifacts/token_selector/2026-10-06-conservative-l2-guarded-reranker-v2-full2638/conservative_l2_guarded_rerank_v2_k96_full2638.jsonl \
  --budget 96 --region-grid 3 --local-candidate-cap 16 --max-swaps 16 \
  --dtype bf16 --device cuda:0
```

汇总判定（新臂缺失时 fail closed、退出码 2、不写报告）：

```bash
python evaluation/summarize_conservative_l2_guarded_rerank_full2638.py
```

## 7. Pins

本预注册在写 config 时对 **36** 个输入完成现场 SHA256 复核（`pins_verified=true`）；
新臂输出 SHA256 为 `null`、状态 `TO_BE_GENERATED`。完整 pin 表见机器可读报告
`reports/token_selector/2026-10-06-conservative-l2-guarded-reranker-v2-prereg.json`。

## 8. 限制

- full2638 是反复观测过的发展集，不是未触碰确认集。
- 通过只是外部验证许可，不是论文/可部署/FPGA 时延主张。
- Selector 路径是软件桥；不做 FPGA/板卡/时延/加速声明。
- 看到 full2638 后不得调参 v2。
- 先验 Prompt 融合分仅为描述性同预算比较。
- v1 永久 `PREEXECUTION_DESIGN_STOP`，不得运行或修补。
- 新臂输出尚不存在，状态 `PREREGISTERED_NOT_EXECUTED`。
