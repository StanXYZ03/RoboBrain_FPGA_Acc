# Wv 逐行→逐张量 scale 硬件正则化审计 — 2026-10-03-selector-wv-scale-regularization

- 决策：**PER_TENSOR_WV_ACCEPTABLE_FOR_HLS**
- 范围：CPU-only、离线、validation-only、只读；未训练、未调参、未加载 4B、未读原图、未访问 internal test。
- 参考臂：`PER_ROW_WV`（逐输出行 INT8 Wv scale）；候选臂：`PER_TENSOR_WV`（整张 Wv 单一 scale `d=max(abs(Wv))/127`）；float 参考为同一冻结 scorer。
- 聚合：每记录跨 3 seed 取中位数，再按 38 张独立图像取两任务均值并做 2000 次 percentile bootstrap 95% CI；不使用 p 值。

## 冻结判据

| 判据 | 通过 |
|---|---|
| `pins_isolation_int32_fixture_pass` | True |
| `candidate_vs_per_row_mean_jaccard_each_budget_at_least_0.98` | True |
| `candidate_vs_per_row_mean_replacement_count_each_budget_at_most_1.0` | True |
| `candidate_vs_per_row_mean_score_spearman_at_least_0.995` | True |
| `candidate_vs_float_mass_delta_each_budget_at_least_-0.01` | True |
| `candidate_vs_float_ndcg_delta_each_budget_at_least_-0.01` | True |
| `pointing_k48_prompt_utility_gap_delta_candidate_vs_float_at_least_-0.01` | True |
| `each_seed_k48_candidate_vs_per_row_mean_jaccard_at_least_0.97` | True |

## 候选 vs 逐行参考（correct prompt，overall 图像级 mean）

| K | Jaccard mean [95% CI] | exact-set rate（仅描述） | replacement count | score Spearman | mass delta | NDCG delta |
|---|---|---|---|---|---|---|
| 48 | 0.998389 [0.996241, 1.000000] | 0.960526 | 0.039474 | 0.999982 | 0.000000 | -0.000072 |
| 64 | 0.998785 [0.997571, 1.000000] | 0.960526 | 0.039474 | 0.999982 | 0.000000 | -0.000063 |
| 96 | 0.998644 [0.997558, 0.999729] | 0.934211 | 0.065789 | 0.999982 | 0.000000 | -0.000066 |

## 候选 vs float（correct prompt，overall 图像级 mean）

| K | Jaccard | mass delta | NDCG delta |
|---|---|---|---|
| 48 | 0.959612 | -0.000321 | -0.000255 |
| 64 | 0.962334 | -0.000373 | 0.000048 |
| 96 | 0.976286 | -0.003206 | -0.000385 |

## K48 各臂 prompt-utility gap（correct − wrong retained mass）delta

- Pointing（候选 − float）mean=-0.009432 CI=[-0.038977, 0.007120]
- Grounding（候选 − float，仅描述：目标嵌套不构成有效负样本）mean=-0.000362 CI=[-0.007036, 0.005913]
- Pointing（候选 − 逐行）mean=0.000000 CI=[0.000000, 0.000000]
- Grounding（候选 − 逐行，仅描述：目标嵌套不构成有效负样本）mean=-0.000024 CI=[-0.000120, 0.000040]

## 逐 seed K48 候选 vs 逐行 mean Jaccard

- seed 20261003: 0.998926
- seed 20261004: 0.994629
- seed 20261005: 0.994629

## INT32 解析最坏界与观测

| dot | length | |acc| bound | fits INT32 |
|---|---|---|---|
| group_mean_dot_wv_row | 1024 | 16646144 | True |
| prompt_dot_wt_row | 2560 | 41615360 | True |
| query_dot | 128 | 2080768 | True |

- 观测最大 |INT32 累加|：group_mean×Wv 逐行=227969，逐张量=145258，no_overflow=True。
- 导出 fixture 回读重算逐位一致=True；主 bundle 不导出 Wt。

## 隔离与 pins

- 打开 val NPZ 76（仅 val=True）；test opened=[]；train opened=[]；val aggregate 匹配预注册=True。
- 运行后 pins 未变=True（14 项）。

## 限制

- Float normalization, reciprocal square root, L2 and min-max remain software golden.
- Passing does not establish complete fixed-point HLS equivalence, FPGA latency, end-to-end accuracy or publication readiness.
- Validation has already been used for development; internal test remains sealed.
- Float L2 normalisation, reciprocal square root, min-max and row normalisation are software golden operations in this audit.
- Passing does not establish complete fixed-point HLS equivalence, FPGA latency, end-to-end accuracy or publication readiness.
- The candidate export stores the per-tensor scale as float32 while the metric chain uses the float64 value; the difference is far below the Q0.15 resolution and is validated by the fixture round-trip.

> 明确说明：float 归一化/倒数平方根/min-max 仍是本审计的 software golden；本审计未证明完整定点 FPGA scorer，也不作精度、延迟、端到端加速或发表声明。
