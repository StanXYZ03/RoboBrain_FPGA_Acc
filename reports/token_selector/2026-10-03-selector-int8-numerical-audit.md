# Selector INT8 数值审计 — 2026-10-03-selector-int8-numerical-audit

- 决策：**INT8_NUMERICS_NOT_ESTABLISHED**
- 范围：CPU-only、离线、validation-only、只读；未训练、未调参、未加载 4B、未读原图、未访问 internal test。
- 主臂：`SPLIT_WV_Q8_A8_Q8QUERY`；描述性对照：`FULL_WT_WV_Q8_A8`（不参与决策）。
- 聚合：每记录跨 3 seed 取中位数，再按 38 张独立图像报告 mean 与 2000 次 percentile bootstrap 95% CI；不使用 p 值。

## 冻结判据

| 判据 | 通过 |
|---|---|
| `pins_and_isolation_pass` | True |
| `int32_worst_case_bounds_below_2147483647` | True |
| `mean_jaccard_each_budget_at_least_0.95` | True |
| `exact_set_match_rate_each_budget_at_least_0.75` | False |
| `mean_retained_mass_delta_each_budget_at_least_-0.01` | True |
| `mean_selected_ndcg_delta_each_budget_at_least_-0.01` | True |
| `pointing_k48_prompt_utility_gap_delta_at_least_-0.01` | True |
| `each_seed_k48_mean_jaccard_at_least_0.95` | True |

## 主臂量化 vs float（overall）

| budget | Jaccard mean [95% CI] | exact-set rate | mass delta | NDCG delta |
|---|---|---|---|---|
| 48 | 0.959074 [0.952345, 0.965369] | 0.223684 | -0.000253 | -0.000093 |
| 64 | 0.962334 [0.957574, 0.967476] | 0.144737 | -0.000373 | 0.000200 |
| 96 | 0.976552 [0.973584, 0.979492] | 0.184211 | -0.002636 | -0.000054 |

## K48 Pointing prompt-utility retained-mass gap delta (quant - float)

- Pointing mean=-0.009432 CI=[-0.038977, 0.007120]
- Grounding（仅描述，嵌套目标不构成有效负样本）mean=-0.000338 CI=[-0.007045, 0.005950]

## INT32 解析最坏界

| dot | length | |acc| bound | fits INT32 |
|---|---|---|---|
| group_mean_dot_wv_row | 1024 | 16646144 | True |
| prompt_dot_wt_row | 2560 | 41615360 | True |
| query_dot | 128 | 2080768 | True |

- 观测最大 |INT32 累加|：group_mean×Wv=227969，prompt×Wt=1023052，no_overflow=True。

## 隔离与 pins

- 打开 val NPZ 76（仅 val=True）；test opened=[]；train opened=[]；val aggregate 匹配预注册=True。
- 运行后 pins 未变=True（15 项）。

## 描述性对照臂

| budget | Jaccard | exact-set rate | mass delta | NDCG delta |
|---|---|---|---|---|
| 48 | 0.959611 | 0.236842 | -0.000151 | 0.000061 |
| 64 | 0.961954 | 0.157895 | -0.000020 | 0.000285 |
| 96 | 0.976286 | 0.184211 | -0.004215 | -0.000305 |

## 限制

- Grounding same-image Pointing prompt is descriptive, not a valid negative, because targets are nested.
- Float L2 normalization, reciprocal square root and min-max are numerical golden operations in this audit.
- Passing does not establish complete fixed-point HLS equivalence, FPGA latency, end-to-end accuracy or publication readiness.
- Float L2 normalisation, reciprocal square root, min-max and row normalisation are software golden operations in this audit.
- This audit is an engineering-equivalence check; it does not prove a complete fixed-point FPGA scorer, HLS/bit-level equivalence, FPGA latency, end-to-end accuracy or publication readiness.
- The Pointing K48 prompt-utility gate uses the same-image other-task prompt; Grounding is reported descriptively only because its targets are nested.
- The validation split has already been used in development; this is not a final or CV-Bench accuracy result.

> 明确说明：float L2 归一化/倒数平方根/min-max 仍是本审计的 software golden；本审计尚未证明完整的定点 FPGA scorer，也不作精度、延迟、端到端加速或发表声明。
