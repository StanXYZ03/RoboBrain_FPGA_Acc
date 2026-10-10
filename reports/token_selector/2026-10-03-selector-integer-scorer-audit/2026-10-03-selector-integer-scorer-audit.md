# 位精确整数 Selector scorer 审计 — 2026-10-03-selector-integer-scorer-audit

- 决策：**INTEGER_SCORER_NUMERICS_PASS**
- 范围：CPU-only、离线、validation-only、只读；未训练、未调参、未加载 4B、未读原图、未访问 internal test。
- 参考臂：`PER_TENSOR_FLOAT_NORM`（已接受的 per-tensor Wv float-normalisation 链路）；候选臂：`INTEGER_SCORER_Q16`（输入量化后纯整数，semantic 小数位冻结为 16）。
- 聚合：每记录跨 3 seed 取中位数，再按 38 张独立图像取两任务均值并做 2000 次 percentile bootstrap 95% CI；不使用 p 值。

## 冻结判据

| 判据 | 通过 |
|---|---|
| `pins_isolation_fixture_width_pass` | True |
| `candidate_vs_per_tensor_mean_jaccard_each_budget_at_least_0.98` | True |
| `candidate_vs_per_tensor_mean_replacement_count_each_budget_at_most_1.0` | True |
| `candidate_vs_per_tensor_mean_score_spearman_at_least_0.995` | True |
| `candidate_vs_float_mass_delta_each_budget_at_least_-0.01` | True |
| `candidate_vs_float_ndcg_delta_each_budget_at_least_-0.01` | True |
| `pointing_k48_prompt_utility_gap_delta_candidate_vs_float_at_least_-0.01` | True |
| `each_seed_k48_candidate_vs_per_tensor_mean_jaccard_at_least_0.97` | True |

## 候选 vs per-tensor 参考（correct prompt，overall 图像级 mean）

| K | Jaccard mean [95% CI] | exact-set rate（仅描述） | replacement count | score Spearman | mass delta | NDCG delta |
|---|---|---|---|---|---|---|
| 48 | 0.994092 [0.990857, 0.997315] | 0.855263 | 0.144737 | 0.999957 | 0.000000 | 0.000005 |
| 64 | 0.993939 [0.991118, 0.996356] | 0.815789 | 0.197368 | 0.999957 | -0.000040 | -0.000043 |
| 96 | 0.996744 [0.995117, 0.998101] | 0.842105 | 0.157895 | 0.999957 | -0.000000 | 0.000006 |

## 候选 vs 原始 float（correct prompt，overall 图像级 mean）

| K | Jaccard | mass delta | NDCG delta |
|---|---|---|---|
| 48 | 0.959012 | -0.000244 | -0.000171 |
| 64 | 0.962775 | -0.000164 | -0.000029 |
| 96 | 0.974957 | -0.002779 | -0.000223 |

## K48 各臂 prompt-utility gap（correct − wrong retained mass）delta

- Pointing（候选 − float）mean=-0.009432 CI=[-0.038977, 0.007120]
- Grounding（候选 − float，仅描述：目标嵌套不构成有效负样本）mean=0.000375 CI=[-0.006223, 0.006651]
- Pointing（候选 − per-tensor）mean=0.000000 CI=[0.000000, 0.000000]
- Grounding（候选 − per-tensor，仅描述：目标嵌套不构成有效负样本）mean=0.000736 CI=[-0.000019, 0.001742]

## 逐 seed K48 候选 vs per-tensor mean Jaccard

- seed 20261003: 0.986617
- seed 20261004: 0.988786
- seed 20261005: 0.991944

## 整数宽度与操作合同

| 操作 | 宽度 | 解析上界 | 观测最大 | 均放得下 |
|---|---|---|---|---|
| `visual_projection_accumulator` | INT32 | 16646144 | 145258 | True |
| `semantic_dot_accumulator` | INT64 | 272730423296 | 156390982 | True |
| `semantic_norm2_accumulator` | INT64 | 35468046088798208 | 558499699697 | True |
| `semantic_shifted_numerator` | INT64 | 17873661021126656 | 10249239396352 | True |
| `semantic_q16` | INT32 | 201326592 | 20902468 | True |
| `l2_squared_sum` | INT64 | 16777216 | 369467 | True |
| `l2_int` | INT32 | 4096 | 607 | True |
| `semantic_norm_q15` | UINT16 | 32767 | 32767 | True |
| `l2_norm_q15` | UINT16 | 32767 | 32767 | True |
| `score_q` | UINT16 | 32767 | 30581 | True |

- 全部分析/观测值放得下声明宽度：`True`；fixture 回读逐位一致：`True`；fixture 不复制 Wv。

## 隔离与 pins

- 打开 val NPZ 76（仅 val=True）；test opened=[]；train opened=[]；val aggregate 匹配预注册=True。
- 运行后 pins 未变=True（17 项）。

## 限制

- This is a Python bit-exact numerical reference, not HLS synthesis or board execution.
- Passing permits HLS implementation but does not prove latency, resources, task accuracy or end-to-end acceleration.
- Validation has already been used in development; internal test remains sealed.
- This is a Python bit-exact integer reference, not HLS synthesis, board execution, FPGA latency or end-to-end accuracy evidence.
- Python arbitrary-precision integers are used only to prove fixed widths and to run the exact rational divisions/square roots.
- The reference float-normalisation arm remains the accepted prior chain and is not re-derived here.

> 明确说明：本审计是 Python 位精确整数参考，非 HLS 综合/上板证据；通过仅授权实现 HLS，不证明时延、资源、任务精度或端到端加速。
