# 2026-10-02 冻结权重 scorer-reduction 消融（中文主报告）

> CPU-only、只读、冻结权重：**不训练、不调参、不加载 4B 模型、不使用 GPU、不联网、不跑 CV-Bench**。复用既有 472 条 NPZ 派生特征、当前 `Wt`/`Wv`、现有 train/val/test split 与 v2 统计工具。

- 预注册：`b2686bc394833c4026ff8a2e69359fba0e8e6b94f57815be1b70e7768c96d508`（seed 20261002，读取任何 test 数值前写盘）；候选顺序 `hard_max, mean_cos, topk_mean_k2, topk_mean_k4, norm_lse_tau003, norm_lse_tau010, norm_lse_tau030, softmax_mean_tau003, softmax_mean_tau010, softmax_mean_tau030, prompt_mean_vector`

## 1. 目标

隔离并修复现有 prompt-conditioned selector 的 hard-max prompt collapse：v2 诊断显示 `max_t cosine(Wv·group_mean, Wt·prompt_t)` 在每个 group 上由单一 prompt token 主导，同图 grounding/pointing 的 correct 分数 `np.array_equal` 精确相等。本消融只替换 **prompt-token 归约**，其余（视觉投影、prompt 投影、split、teacher target、指标定义）全部冻结。

## 2. 预注册候选集合（不得在 test 后增删/改参）

| # | candidate | kind | params | token competition |
| --- | --- | --- | --- | --- |
| 1 | hard_max | hard_max | `{}` | True |
| 2 | mean_cos | mean_cos | `{}` | False |
| 3 | topk_mean_k2 | topk_mean | `{"k": 2}` | True |
| 4 | topk_mean_k4 | topk_mean | `{"k": 4}` | True |
| 5 | norm_lse_tau003 | norm_lse | `{"tau": 0.03}` | True |
| 6 | norm_lse_tau010 | norm_lse | `{"tau": 0.1}` | True |
| 7 | norm_lse_tau030 | norm_lse | `{"tau": 0.3}` | True |
| 8 | softmax_mean_tau003 | softmax_mean | `{"tau": 0.03}` | True |
| 9 | softmax_mean_tau010 | softmax_mean | `{"tau": 0.1}` | True |
| 10 | softmax_mean_tau030 | softmax_mean | `{"tau": 0.3}` | True |
| 11 | prompt_mean_vector | prompt_mean_vector | `{}` | False |

## 3. val promotion gate（全部只看 val）

- A prompt sensitivity：same-image-other-task 的 Spearman 与 NDCG-full paired delta 均 >0，95% CI 下界 >0，|dz|≥0.2，Holm-adjusted p<0.05
- B shuffled discrimination：cross-image-same-task 的 Spearman 与 NDCG-full paired delta 均 >0，95% CI 下界 >0，|dz|≥0.2，Holm-adjusted p<0.05
- C collapse removed：exact_prompt_invariance_fraction ≤ 0.05
- D teacher non-regression：候选 val correct Spearman 与 NDCG-full 均不低于 hard_max baseline 超过 0.01
- E 数值有限且硬件实现合同完整
- Holm-Bonferroni family：{ (candidate, condition, metric) : candidate in NON_BASELINE_CANDIDATES (10 candidates), condition in {same_image_other_task, cross_image_same_task_shuffled}, metric in {spearman, ndcg_full} }; m = 40

- **winner = NO_PROMOTION**；no candidate passed all pre-registered val gates A-E

| candidate | A | B | C | D | E | passed_all | exact_inv frac |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hard_max | False | False | False | True | True | False | 1.0000 |
| mean_cos | False | False | True | False | True | False | 0.0000 |
| topk_mean_k2 | False | False | False | True | True | False | 0.9474 |
| topk_mean_k4 | False | False | True | True | True | False | 0.0000 |
| norm_lse_tau003 | False | False | True | True | True | False | 0.0000 |
| norm_lse_tau010 | False | False | True | True | True | False | 0.0000 |
| norm_lse_tau030 | False | False | True | False | True | False | 0.0000 |
| softmax_mean_tau003 | False | False | True | True | True | False | 0.0263 |
| softmax_mean_tau010 | False | False | True | True | True | False | 0.0000 |
| softmax_mean_tau030 | False | False | True | False | True | False | 0.0000 |
| prompt_mean_vector | False | False | True | False | True | False | 0.0000 |

## 4. val 关键数值（correct 与两个 paired control，图像级聚类）

- val：n_rec=76，n_img=38，prompt token min/mean/max = 38/40.6/45，G min/mean/max = 160/268.6/360

| candidate | correct rho | correct NDCG-full | Δsame rho [CI] | Δsame NDCG [CI] | Δcross rho [CI] | Δcross NDCG [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| hard_max | 0.0565 | 0.8499 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0112 [-0.0336, +0.0000] | -0.0027 [-0.0082, +0.0000] |
| mean_cos | 0.0129 | 0.8343 | +0.0004 [-0.0018, +0.0027] | +0.0000 [-0.0004, +0.0004] | +0.0001 [-0.0011, +0.0011] | +0.0000 [-0.0003, +0.0004] |
| topk_mean_k2 | 0.0563 | 0.8510 | -0.0000 [-0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] | -0.0002 [-0.0126, +0.0105] | +0.0016 [-0.0013, +0.0056] |
| topk_mean_k4 | 0.0574 | 0.8513 | +0.0015 [-0.0008, +0.0041] | +0.0001 [-0.0002, +0.0005] | -0.0002 [-0.0077, +0.0068] | +0.0003 [-0.0015, +0.0020] |
| norm_lse_tau003 | 0.0564 | 0.8499 | -0.0000 [-0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] | -0.0112 [-0.0337, +0.0000] | -0.0027 [-0.0082, +0.0000] |
| norm_lse_tau010 | 0.0554 | 0.8501 | +0.0002 [-0.0003, +0.0008] | +0.0001 [+0.0000, +0.0002] | -0.0102 [-0.0318, +0.0014] | -0.0024 [-0.0076, +0.0004] |
| norm_lse_tau030 | 0.0292 | 0.8397 | +0.0000 [-0.0032, +0.0032] | +0.0001 [-0.0006, +0.0010] | -0.0008 [-0.0052, +0.0024] | +0.0002 [-0.0005, +0.0011] |
| softmax_mean_tau003 | 0.0564 | 0.8499 | -0.0000 [-0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] | -0.0113 [-0.0338, +0.0000] | -0.0028 [-0.0082, -0.0000] |
| softmax_mean_tau010 | 0.0593 | 0.8493 | -0.0004 [-0.0015, +0.0004] | +0.0001 [-0.0001, +0.0004] | -0.0119 [-0.0363, +0.0007] | -0.0028 [-0.0084, +0.0002] |
| softmax_mean_tau030 | 0.0427 | 0.8458 | -0.0000 [-0.0033, +0.0037] | -0.0000 [-0.0006, +0.0005] | -0.0090 [-0.0293, +0.0028] | -0.0014 [-0.0038, +0.0003] |
| prompt_mean_vector | 0.0129 | 0.8343 | +0.0004 [-0.0017, +0.0027] | +0.0000 [-0.0004, +0.0004] | +0.0001 [-0.0011, +0.0010] | +0.0000 [-0.0003, +0.0004] |

### 4.1 dominant-token concentration（含 token competition 的 reducer）

| candidate | concentration | single-argmax fraction | mean distinct tokens |
| --- | --- | --- | --- |
| hard_max | 0.9999 | 0.9737 | 1.026 |
| mean_cos | N/A | N/A | N/A |
| topk_mean_k2 | 0.5000 | 0.9737 | 1.026 |
| topk_mean_k4 | 0.2500 | 0.9737 | 1.026 |
| norm_lse_tau003 | 0.9975 | 0.9737 | 1.026 |
| norm_lse_tau010 | 0.8363 | 0.9737 | 1.026 |
| norm_lse_tau030 | 0.1567 | 0.9737 | 1.026 |
| softmax_mean_tau003 | 0.9975 | 0.9737 | 1.026 |
| softmax_mean_tau010 | 0.8363 | 0.9737 | 1.026 |
| softmax_mean_tau030 | 0.1567 | 0.9737 | 1.026 |
| prompt_mean_vector | N/A | N/A | N/A |

### 4.2 prompt invariance（correct vs 同图另一任务）

| candidate | exact (np.array_equal) | tolerance 1e-9 | pairs | max_abs_diff max |
| --- | --- | --- | --- | --- |
| hard_max | 1.0000 | 1.0000 | 76 | 0.000e+00 |
| mean_cos | 0.0000 | 0.0000 | 76 | 1.580e-02 |
| topk_mean_k2 | 0.9474 | 0.9474 | 76 | 5.185e-03 |
| topk_mean_k4 | 0.0000 | 0.0000 | 76 | 2.531e-02 |
| norm_lse_tau003 | 0.0000 | 0.0000 | 76 | 3.710e-03 |
| norm_lse_tau010 | 0.0000 | 0.0000 | 76 | 1.459e-02 |
| norm_lse_tau030 | 0.0000 | 0.0000 | 76 | 2.352e-02 |
| softmax_mean_tau003 | 0.0263 | 0.0263 | 76 | 2.599e-05 |
| softmax_mean_tau010 | 0.0000 | 0.0000 | 76 | 1.879e-02 |
| softmax_mean_tau030 | 0.0000 | 0.0000 | 76 | 3.009e-02 |
| prompt_mean_vector | 0.0000 | 0.0000 | 76 | 2.909e-02 |

### 4.3 K=48/64/96 teacher top-K overlap/recall/Jaccard/saliency mass（val correct）

| candidate | recall48/64/96 | Jaccard48/64/96 | saliency48/64/96 | NDCG@48/64/96 |
| --- | --- | --- | --- | --- |
| hard_max | 0.2198/0.2800/0.3984 | 0.1271/0.1665/0.2548 | 0.2534/0.3388/0.4956 | 0.5264/0.5561/0.6085 |
| mean_cos | 0.1938/0.2564/0.3771 | 0.1094/0.1500/0.2361 | 0.1641/0.2438/0.3956 | 0.4813/0.5172/0.5734 |
| topk_mean_k2 | 0.2141/0.2765/0.3995 | 0.1231/0.1645/0.2557 | 0.2417/0.3090/0.4529 | 0.5260/0.5549/0.6069 |
| topk_mean_k4 | 0.2220/0.2778/0.3973 | 0.1289/0.1656/0.2541 | 0.2455/0.3234/0.4630 | 0.5274/0.5568/0.6086 |
| norm_lse_tau003 | 0.2198/0.2800/0.3987 | 0.1271/0.1665/0.2550 | 0.2534/0.3388/0.4956 | 0.5264/0.5561/0.6085 |
| norm_lse_tau010 | 0.2193/0.2794/0.3991 | 0.1265/0.1661/0.2552 | 0.2516/0.3419/0.4960 | 0.5250/0.5573/0.6089 |
| norm_lse_tau030 | 0.2037/0.2650/0.3899 | 0.1157/0.1556/0.2463 | 0.2048/0.2675/0.4258 | 0.4979/0.5304/0.5859 |
| softmax_mean_tau003 | 0.2204/0.2800/0.3983 | 0.1275/0.1665/0.2547 | 0.2534/0.3388/0.4955 | 0.5266/0.5561/0.6084 |
| softmax_mean_tau010 | 0.2237/0.2806/0.4009 | 0.1295/0.1675/0.2571 | 0.2590/0.3426/0.4859 | 0.5268/0.5555/0.6092 |
| softmax_mean_tau030 | 0.2133/0.2712/0.3930 | 0.1223/0.1607/0.2495 | 0.2486/0.3182/0.4445 | 0.5142/0.5441/0.5973 |
| prompt_mean_vector | 0.1938/0.2564/0.3771 | 0.1094/0.1500/0.2361 | 0.1641/0.2438/0.3956 | 0.4813/0.5172/0.5734 |

## 5. 硬件代价（解析符号计数，未综合、未实测 ZCU102）

- analytic symbol counts only (G=visual group count, T=prompt token count, D=128); the prompt reducer runs on the host/Jetson and the scorer on the FPGA.  NOT synthesised and NOT measured on ZCU102; no latency, resource or speedup claim is made.  INT8 bytes assume one signed int8 per element (128-byte 1x128 query / 16-byte 128-bit alignment not modelled here).
- 评估点：G=269，T=41，D=128

| candidate | FPGA dot/group | cmp/group | exp/group | reduce/group | total/group | total/image | query shape | query INT8 B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hard_max | 5248 | 40 | 0 | 0 | 5288 | 1422472 | [41, 128] | 5248 |
| mean_cos | 5248 | 0 | 0 | 40 | 5288 | 1422472 | [41, 128] | 5248 |
| topk_mean_k2 | 5248 | 82 | 0 | 1 | 5331 | 1434039 | [41, 128] | 5248 |
| topk_mean_k4 | 5248 | 164 | 0 | 3 | 5415 | 1456635 | [41, 128] | 5248 |
| norm_lse_tau003 | 5248 | 41 | 41 | 42 | 5372 | 1445068 | [41, 128] | 5248 |
| norm_lse_tau010 | 5248 | 41 | 41 | 42 | 5372 | 1445068 | [41, 128] | 5248 |
| norm_lse_tau030 | 5248 | 41 | 41 | 42 | 5372 | 1445068 | [41, 128] | 5248 |
| softmax_mean_tau003 | 5248 | 41 | 41 | 82 | 5412 | 1455828 | [41, 128] | 5248 |
| softmax_mean_tau010 | 5248 | 41 | 41 | 82 | 5412 | 1455828 | [41, 128] | 5248 |
| softmax_mean_tau030 | 5248 | 41 | 41 | 82 | 5412 | 1455828 | [41, 128] | 5248 |
| prompt_mean_vector | 128 | 0 | 0 | 0 | 128 | 34432 | [1, 128] | 128 |

重点：`prompt_mean_vector` 只需 **1x128 INT8 query**（128 B）传输，FPGA 侧为 **Gx128 dot**（每 group 128 MAC，无 compare/exp/归约），彻底移除 token 竞争；其余矩阵型 reducer 需传输 `[T,128]` query 并在 FPGA 上做 `T*D` dot 与归约。

## 6. test（仅在 winner 写盘后评一次，不得据 test 反选）

- evaluated=True，evaluated_after_winner_written=True，used_for_selection=False
- test：n_rec=78，n_img=39
- NO_PROMOTION：baseline `hard_max` test correct Spearman=0.0536，NDCG-full=0.8493（仅冻结诊断）

## 7. 结论与下一决策

- **winner = NO_PROMOTION**：本批预注册候选无人通过全部门槛 A-E，按预注册规则**不得事后挑“最不差”方案晋级**，也不得据 test 反选。
- 所有数字均为 teacher-target 排序/相关性内部指标，**不是 CV-Bench 任务精度**；硬件代价为解析符号计数，**未综合、未实测 ZCU102**。

## 8. 复现

- python 3.10.20 / numpy 2.2.6
- seed 20261002；derangement seeds {'train': 20261002, 'val': 20261103, 'test': 20261204}
- index_sha256 `02a50868a8467541…`；projection_sha256 `132a6d9fcd35ee94…`；preregistration `b2686bc394833c40…`
- elapsed 21.65 s
