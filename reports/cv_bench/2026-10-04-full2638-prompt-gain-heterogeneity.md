# full2638 Prompt 增益异质性诊断（只读、事后探索）

- 生成时间：`2026-10-04T18:36:36.410701+08:00`
- 范围：post-hoc exploratory diagnostic of the frozen formal result PROMPT_GAIN_NOT_CONFIRMED_FULL2638; CPU-only, read-only; NOT a new primary metric and NOT a formal gate
- 冻结正式结论（不变）：**PROMPT_GAIN_NOT_CONFIRMED_FULL2638**；唯一主指标 `score(...)["cv_bench_accuracy"]`
> All subgroup / bootstrap / overlap numbers in this report are post-hoc exploratory descriptions of the frozen PROMPT_GAIN_NOT_CONFIRMED_FULL2638 result. They are not a new primary metric, not a formal gate and must not be quoted as evidence for a new claim.

- pre-selector real group count 定义：Number of real (non-context) candidate groups before selection = full_llm_visual_tokens (one group per 2x2 merge unit). The raw JSONL field 'real_group_count' is the POST-selection count of REAL source slots (87 for every non-bypass sample; G for bypass samples) and is reported separately for audit.
- bootstrap：seed=20261004 / repeats=5000 / level=0.95；**不作为正式门。**

## 1. 总体迁移与配对差

- n=2638；wrong→right=220（率 0.083397）、right→wrong=194（率 0.073541）、不变对=1820、不变错=404。
- 净修正=+26；unweighted 配对差=+0.009856；official `cv_bench_accuracy` Δ=+0.010510，95% CI [-0.005430, +0.025699]（与冻结报告逐位一致：True）。

## 2. 按 task

| 层 | n | W→R | R→W | 不变对 | 不变错 | 净修正 | int(unweighted) | prompt(unweighted) | 配对差 | 95% CI（unweighted，探索性） | official Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Count | 788 | 84 | 60 | 348 | 296 | +24 | 0.517766 | 0.548223 | +0.030457 | [+0.001269, +0.060914] | n/a |
| Depth | 600 | 49 | 50 | 472 | 29 | -1 | 0.870000 | 0.868333 | -0.001667 | [-0.033333, +0.030000] | n/a |
| Distance | 600 | 66 | 50 | 436 | 48 | +16 | 0.810000 | 0.836667 | +0.026667 | [-0.008333, +0.061667] | n/a |
| Relation | 650 | 21 | 34 | 564 | 31 | -13 | 0.920000 | 0.900000 | -0.020000 | [-0.041538, +0.001538] | n/a |

## 3. 按 source_dataset

| 层 | n | W→R | R→W | 不变对 | 不变错 | 净修正 | int(unweighted) | prompt(unweighted) | 配对差 | 95% CI（unweighted，探索性） | official Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| ADE20K Validation Set | 633 | 44 | 34 | 374 | 181 | +10 | 0.644550 | 0.660348 | +0.015798 | [-0.011058, +0.044234] | n/a |
| COCO 2017 Validation Set | 805 | 61 | 60 | 538 | 146 | +1 | 0.742857 | 0.744099 | +0.001242 | [-0.024845, +0.028571] | n/a |
| Omni3D_Hypersim | 400 | 61 | 36 | 271 | 32 | +25 | 0.767500 | 0.830000 | +0.062500 | [+0.015000, +0.107500] | n/a |
| Omni3D_SUNRGBD | 400 | 6 | 28 | 365 | 1 | -22 | 0.982500 | 0.927500 | -0.055000 | [-0.082500, -0.027500] | n/a |
| Omni3D_nuScenes | 400 | 48 | 36 | 272 | 44 | +12 | 0.770000 | 0.800000 | +0.030000 | [-0.015000, +0.075000] | n/a |

## 4. 按 source

| 层 | n | W→R | R→W | 不变对 | 不变错 | 净修正 | int(unweighted) | prompt(unweighted) | 配对差 | 95% CI（unweighted，探索性） | official Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| ADE20K | 633 | 44 | 34 | 374 | 181 | +10 | 0.644550 | 0.660348 | +0.015798 | [-0.011058, +0.044234] | n/a |
| COCO | 805 | 61 | 60 | 538 | 146 | +1 | 0.742857 | 0.744099 | +0.001242 | [-0.024845, +0.028571] | n/a |
| Omni3D | 1200 | 115 | 100 | 908 | 77 | +15 | 0.840000 | 0.852500 | +0.012500 | [-0.010833, +0.035833] | n/a |

## 5. 按 identity bypass

| 层 | n | W→R | R→W | 不变对 | 不变错 | 净修正 | int(unweighted) | prompt(unweighted) | 配对差 | 95% CI（unweighted，探索性） | official Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| False | 2495 | 220 | 194 | 1716 | 365 | +26 | 0.765531 | 0.775952 | +0.010421 | [-0.005210, +0.026854] | +0.011371 |
| True | 143 | 0 | 0 | 104 | 39 | +0 | 0.727273 | 0.727273 | +0.000000 | [+0.000000, +0.000000] | n/a |

## 6. 按 pre-selector real group count 桶（= full_llm_visual_tokens）

| 层 | n | W→R | R→W | 不变对 | 不变错 | 净修正 | int(unweighted) | prompt(unweighted) | 配对差 | 95% CI（unweighted，探索性） | official Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| <=96 | 143 | 0 | 0 | 104 | 39 | +0 | 0.727273 | 0.727273 | +0.000000 | [+0.000000, +0.000000] | n/a |
| 97-256 | 781 | 31 | 45 | 641 | 64 | -14 | 0.878361 | 0.860435 | -0.017926 | [-0.039693, +0.003841] | -0.019935 |
| 257-512 | 706 | 56 | 63 | 458 | 129 | -7 | 0.737960 | 0.728045 | -0.009915 | [-0.039660, +0.021246] | -0.007472 |
| 513-1024 | 448 | 65 | 42 | 293 | 48 | +23 | 0.747768 | 0.799107 | +0.051339 | [+0.004464, +0.095982] | n/a |
| >1024 | 560 | 68 | 44 | 324 | 124 | +24 | 0.657143 | 0.700000 | +0.042857 | [+0.007143, +0.078571] | n/a |

## 7. 按原始字段 real_group_count 桶（审计用；该字段为选择后计数）

| 层 | n | W→R | R→W | 不变对 | 不变错 | 净修正 | int(unweighted) | prompt(unweighted) | 配对差 | 95% CI（unweighted，探索性） | official Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| <=96 | 2638 | 220 | 194 | 1820 | 404 | +26 | 0.763457 | 0.773313 | +0.009856 | n/a | +0.010510 |

## 8. 非 bypass 样本的选择索引重叠

- 范围：non-bypass samples only (both arms emit K=96)；n=2495；精确匹配=0（率 0.000000）；Jaccard mean=0.285257、min=0.043478。

| task | n | 精确匹配 | 精确匹配率 | Jaccard mean | Jaccard min |
|---|---:|---:|---:|---:|---:|
| Count | 707 | 0 | 0.000000 | 0.312494 | 0.043478 |
| Depth | 600 | 0 | 0.000000 | 0.193961 | 0.049180 |
| Distance | 600 | 0 | 0.000000 | 0.194858 | 0.060773 |
| Relation | 588 | 0 | 0.000000 | 0.437909 | 0.060773 |

## 9. 增益集中度与回归

- 毛增益 W→R=220、毛损失 R→W=194、净修正=+26、总翻转=414；净值为大额对向流的残差：True。
- task 净值：{'Count': 24, 'Depth': -1, 'Distance': 16, 'Relation': -13}；source_dataset 净值：{'ADE20K Validation Set': 10, 'COCO 2017 Validation Set': 1, 'Omni3D_Hypersim': 25, 'Omni3D_SUNRGBD': -22, 'Omni3D_nuScenes': 12}；pre-selector 桶净值：{'<=96': 0, '97-256': -14, '257-512': -7, '513-1024': 23, '>1024': 24}。
- 最大正向 task={'stratum': 'Count', 'net_corrections': 24}；最大负向 task={'stratum': 'Relation', 'net_corrections': -13}；最大正向 source_dataset={'stratum': 'Omni3D_Hypersim', 'net_corrections': 25}；最大负向 source_dataset={'stratum': 'Omni3D_SUNRGBD', 'net_corrections': -22}。
- 负点估计层数=3；bootstrap 95% CI 整体 <0 的层数=1。
  - `source_dataset::Omni3D_SUNRGBD`：n=400，unweighted 配对差=-0.055000，CI [-0.082500, -0.027500]。

## 10. dense 描述性上下文（非门）

- dense official `cv_bench_accuracy`=0.872730（descriptive context only; never a gate）。
- dense unweighted by task：{'Count': 0.682741, 'Depth': 0.981667, 'Distance': 0.911667, 'Relation': 0.947692}
- dense unweighted by source：{'ADE20K': 0.767773, 'COCO': 0.829814, 'Omni3D': 0.946667}

## 11. 输入 SHA256

| 输入 | SHA256 |
|---|---|
| formal_report | `53ec2ee10fde7b31b08a840263bd26409e35781ed25b89b20166ea90e84af6ea` |
| full_manifest | `22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91` |
| corrected_l2_integer_arm | `d0e302f6b91279374ddd507db4d300ca0c29e16f522db57f47c8de059f103a5f` |
| prompt_plus_corrected_l2_integer_arm | `6d3b97e8ce768066a99c9b9f517ddf602480fa75d57ce2e36b54e26baabc90fa` |

## 12. 限制

- The formal conclusion PROMPT_GAIN_NOT_CONFIRMED_FULL2638 is unchanged; every subgroup number here is post-hoc and exploratory.
- The official cv_bench_accuracy requires all three sources and is undefined inside single-source task/source strata; those stratum CIs use unweighted sample accuracy and are not a gate.
- No GPU, model, board, network or training is used; this is a pure CPU read-only re-analysis of frozen outputs.
- The raw JSONL real_group_count is the post-selection real-slot count; the requested pre-selector bins use full_llm_visual_tokens.

> CPU-only 只读诊断：未改任何权威输入、既有配置/报告或 `configs/evaluation_protocol.json`；无 FPGA/时延/加速声明。
