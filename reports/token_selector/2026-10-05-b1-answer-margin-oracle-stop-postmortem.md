# B1 answer-margin Oracle challenge/control pilot24 STOP 事后尸检（CPU-only、只读）

- 日期：2026-10-05
- 设计：`2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24`
- 状态：**PILOT_STOP_POSTMORTEM_COMPLETE**；冻结决策：**PILOT_STOP**
- 范围：CPU-only、只读冻结结果；无 GPU / 模型权重 / 网络 / 训练；未改任何冻结 manifest / config / raw JSONL / summary；未调阈值 / K / anchor。

## 1. 口径（必须先读）

- 本报告**不推翻、不重新论证**预注册 STOP；只做描述性事后分析。
- **outcome/label leakage**：样本按 dense/control 正确性分层，非代表性、不可部署；禁止官方绝对精度主张。
- **ground-truth answer leakage**：oracle 用官方正确答案分叉 token，只是可证伪上界。
- n=24 且 outcome-enriched；所有关联仅探索性，**不使用 CI / p 值 / 显著性**。

## 2. 冻结决策与验证

| 项 | 值 |
| --- | --- |
| challenge 修复 | 10/12 |
| control 回归 | 3/12 |
| oracle parse | 24/24 |
| 精确 K96/87+9 布局 | 24/24 |
| 冻结状态 | **PILOT_STOP** |
| 输入 SHA / record SHA / ID / 迁移标签 | 全部通过（0bbe413d5575…） |

## 3. 逐样本指标

| sample_id | cell | stratum | 迁移 | margin | G | 压缩比 | 全槽 Jaccard | real 槽 Jaccard | context 槽 Jaccard | Oracle 保留质量 | L2 保留质量 | 差值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 229 | ADE20K x Count | challenge | W->R | 4.2500 | 204 | 0.4706 | 0.3813 | 0.3488 | 0.0588 | 0.7984 | 0.4015 | 0.3969 |
| 290 | ADE20K x Count | challenge | W->R | 0.3750 | 345 | 0.2783 | 0.1294 | 0.0943 | 0.1250 | 0.6128 | 0.1847 | 0.4281 |
| 305 | ADE20K x Count | control | unchanged_correct | 2.2500 | 288 | 0.3333 | 0.1497 | 0.1083 | 0.3846 | 0.6593 | 0.2510 | 0.4083 |
| 334 | ADE20K x Count | control | R->W | 1.2500 | 228 | 0.4211 | 0.2308 | 0.2000 | 0.0000 | 0.7301 | 0.3285 | 0.4016 |
| 425 | ADE20K x Relation | challenge | unchanged_wrong | 0.2500 | 144 | 0.6667 | 0.4769 | 0.4380 | 0.0588 | 0.8282 | 0.6801 | 0.1482 |
| 519 | ADE20K x Relation | challenge | W->R | 2.5000 | 180 | 0.5333 | 0.3427 | 0.2794 | 0.3846 | 0.7460 | 0.4527 | 0.2933 |
| 439 | ADE20K x Relation | control | unchanged_correct | 2.1250 | 180 | 0.5333 | 0.3521 | 0.2985 | 0.1250 | 0.7682 | 0.4019 | 0.3663 |
| 471 | ADE20K x Relation | control | R->W | 0.5000 | 180 | 0.5333 | 0.4222 | 0.3701 | 0.1250 | 0.7879 | 0.5845 | 0.2034 |
| 748 | COCO x Count | challenge | W->R | 0.5000 | 300 | 0.3200 | 0.2549 | 0.2168 | 0.0588 | 0.6488 | 0.3373 | 0.3115 |
| 969 | COCO x Count | challenge | unchanged_wrong | 7.5000 | 300 | 0.3200 | 0.2308 | 0.1837 | 0.2857 | 0.6374 | 0.3928 | 0.2446 |
| 636 | COCO x Count | control | unchanged_correct | 5.8750 | 320 | 0.3000 | 0.1779 | 0.1523 | 0.2000 | 0.7072 | 0.1919 | 0.5152 |
| 719 | COCO x Count | control | unchanged_correct | 6.5000 | 300 | 0.3200 | 0.1779 | 0.1373 | 0.2857 | 0.7001 | 0.1799 | 0.5201 |
| 1221 | COCO x Relation | challenge | W->R | 0.2500 | 180 | 0.5333 | 0.3714 | 0.3282 | 0.1250 | 0.8143 | 0.4862 | 0.3281 |
| 1406 | COCO x Relation | challenge | W->R | 3.5000 | 300 | 0.3200 | 0.1566 | 0.1154 | 0.2857 | 0.6958 | 0.1851 | 0.5107 |
| 1234 | COCO x Relation | control | unchanged_correct | 5.6250 | 180 | 0.5333 | 0.3333 | 0.2889 | 0.1250 | 0.7631 | 0.4056 | 0.3575 |
| 1328 | COCO x Relation | control | unchanged_correct | 3.7500 | 160 | 0.6000 | 0.4656 | 0.4380 | 0.1250 | 0.8045 | 0.5783 | 0.2262 |
| 1701 | Omni3D x Depth | challenge | W->R | 3.6250 | 391 | 0.2455 | 0.2229 | 0.2000 | 0.3846 | 0.5547 | 0.2769 | 0.2778 |
| 1814 | Omni3D x Depth | challenge | W->R | 5.3750 | 234 | 0.4103 | 0.3333 | 0.3282 | 0.0588 | 0.5896 | 0.4385 | 0.1511 |
| 1908 | Omni3D x Depth | control | unchanged_correct | 6.5000 | 1400 | 0.0686 | 0.0787 | 0.0545 | 0.3846 | 0.3564 | 0.0748 | 0.2815 |
| 1932 | Omni3D x Depth | control | unchanged_correct | 3.5000 | 1400 | 0.0686 | 0.0667 | 0.0482 | 0.2000 | 0.3529 | 0.0727 | 0.2802 |
| 2108 | Omni3D x Distance | challenge | W->R | 3.2500 | 768 | 0.1250 | 0.1163 | 0.1083 | 0.2000 | 0.3656 | 0.1690 | 0.1965 |
| 2178 | Omni3D x Distance | challenge | W->R | 4.7500 | 768 | 0.1250 | 0.0909 | 0.0807 | 0.2000 | 0.3725 | 0.1140 | 0.2585 |
| 2518 | Omni3D x Distance | control | R->W | 3.6250 | 1400 | 0.0686 | 0.0971 | 0.0807 | 0.2857 | 0.3001 | 0.0983 | 0.2018 |
| 2566 | Omni3D x Distance | control | unchanged_correct | 4.5000 | 1400 | 0.0686 | 0.0667 | 0.0482 | 0.2857 | 0.2988 | 0.0799 | 0.2189 |

## 4. 3 个回归样本与 2 个未修复 challenge 样本

| 类型 | sample_id | cell | stratum | margin | G | real 槽 Jaccard | 保留质量差值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control 回归 R→W | 334 | ADE20K x Count | control | 1.2500 | 228 | 0.2000 | 0.4016 |
| control 回归 R→W | 471 | ADE20K x Relation | control | 0.5000 | 180 | 0.3701 | 0.2034 |
| control 回归 R→W | 2518 | Omni3D x Distance | control | 3.6250 | 1400 | 0.0807 | 0.2018 |
| 未修复 challenge | 425 | ADE20K x Relation | challenge | 0.2500 | 144 | 0.4380 | 0.1482 |
| 未修复 challenge | 969 | COCO x Count | challenge | 7.5000 | 300 | 0.1837 | 0.2446 |

## 5. 迁移组描述统计（mean / median；仅探索性）

### W->R（n=10）

| 变量 | mean | median | min | max |
| --- | --- | --- | --- | --- |
| oracle_margin | 2.8375 | 3.3750 | 0.2500 | 5.3750 |
| G | 367.0000 | 300.0000 | 180.0000 | 768.0000 |
| compression_ratio | 0.3361 | 0.3200 | 0.1250 | 0.5333 |
| all_slot_intersection | 36.0000 | 37.0000 | 16.0000 | 53.0000 |
| all_slot_jaccard | 0.2400 | 0.2389 | 0.0909 | 0.3813 |
| real_slot_intersection | 29.2000 | 30.0000 | 13.0000 | 45.0000 |
| real_slot_jaccard | 0.2100 | 0.2084 | 0.0807 | 0.3488 |
| context_slot_intersection | 2.7000 | 2.5000 | 1.0000 | 5.0000 |
| context_slot_jaccard | 0.1881 | 0.1625 | 0.0588 | 0.3846 |
| oracle_retained_mass_real | 0.6198 | 0.6308 | 0.3656 | 0.8143 |
| l2_retained_mass_real | 0.3046 | 0.3071 | 0.1140 | 0.4862 |
| retained_mass_advantage | 0.3152 | 0.3024 | 0.1511 | 0.5107 |

### R->W（n=3）

| 变量 | mean | median | min | max |
| --- | --- | --- | --- | --- |
| oracle_margin | 1.7917 | 1.2500 | 0.5000 | 3.6250 |
| G | 602.6667 | 228.0000 | 180.0000 | 1400.0000 |
| compression_ratio | 0.3410 | 0.4211 | 0.0686 | 0.5333 |
| all_slot_intersection | 36.6667 | 36.0000 | 17.0000 | 57.0000 |
| all_slot_jaccard | 0.2500 | 0.2308 | 0.0971 | 0.4222 |
| real_slot_intersection | 29.6667 | 29.0000 | 13.0000 | 47.0000 |
| real_slot_jaccard | 0.2169 | 0.2000 | 0.0807 | 0.3701 |
| context_slot_intersection | 2.0000 | 2.0000 | 0.0000 | 4.0000 |
| context_slot_jaccard | 0.1369 | 0.1250 | 0.0000 | 0.2857 |
| oracle_retained_mass_real | 0.6060 | 0.7301 | 0.3001 | 0.7879 |
| l2_retained_mass_real | 0.3371 | 0.3285 | 0.0983 | 0.5845 |
| retained_mass_advantage | 0.2689 | 0.2034 | 0.2018 | 0.4016 |

### unchanged_correct（n=9）

| 变量 | mean | median | min | max |
| --- | --- | --- | --- | --- |
| oracle_margin | 4.5139 | 4.5000 | 2.1250 | 6.5000 |
| G | 625.3333 | 300.0000 | 160.0000 | 1400.0000 |
| compression_ratio | 0.3140 | 0.3200 | 0.0686 | 0.6000 |
| all_slot_intersection | 31.1111 | 29.0000 | 12.0000 | 61.0000 |
| all_slot_jaccard | 0.2076 | 0.1779 | 0.0667 | 0.4656 |
| real_slot_intersection | 24.2222 | 21.0000 | 8.0000 | 53.0000 |
| real_slot_jaccard | 0.1749 | 0.1373 | 0.0482 | 0.4380 |
| context_slot_intersection | 3.3333 | 3.0000 | 2.0000 | 5.0000 |
| context_slot_jaccard | 0.2351 | 0.2000 | 0.1250 | 0.3846 |
| oracle_retained_mass_real | 0.6012 | 0.7001 | 0.2988 | 0.8045 |
| l2_retained_mass_real | 0.2485 | 0.1919 | 0.0727 | 0.5783 |
| retained_mass_advantage | 0.3527 | 0.3575 | 0.2189 | 0.5201 |

### unchanged_wrong（n=2）

| 变量 | mean | median | min | max |
| --- | --- | --- | --- | --- |
| oracle_margin | 3.8750 | 3.8750 | 0.2500 | 7.5000 |
| G | 222.0000 | 222.0000 | 144.0000 | 300.0000 |
| compression_ratio | 0.4933 | 0.4933 | 0.3200 | 0.6667 |
| all_slot_intersection | 49.0000 | 49.0000 | 36.0000 | 62.0000 |
| all_slot_jaccard | 0.3538 | 0.3538 | 0.2308 | 0.4769 |
| real_slot_intersection | 40.0000 | 40.0000 | 27.0000 | 53.0000 |
| real_slot_jaccard | 0.3108 | 0.3108 | 0.1837 | 0.4380 |
| context_slot_intersection | 2.5000 | 2.5000 | 1.0000 | 4.0000 |
| context_slot_jaccard | 0.1723 | 0.1723 | 0.0588 | 0.2857 |
| oracle_retained_mass_real | 0.7328 | 0.7328 | 0.6374 | 0.8282 |
| l2_retained_mass_real | 0.5364 | 0.5364 | 0.3928 | 0.6801 |
| retained_mass_advantage | 0.1964 | 0.1964 | 0.1482 | 0.2446 |

### Spearman（有符号迁移结果 +1/0/−1；仅探索性，无 p 值）

| 变量 | Spearman rho |
| --- | --- |
| oracle_margin | -0.1003 |
| G | 0.0472 |
| all_slot_jaccard | 0.0150 |
| real_slot_jaccard | 0.0485 |
| context_slot_jaccard | -0.0085 |
| retained_mass_advantage | 0.0835 |

## 6. 可能解释的逐项审计

### low_dense_answer_margin

- 判定：`unproven_as_sufficient_explanation`；已确立事实：`False`
- 说明：The full-sequence answer margin is recorded only for the Oracle arm. The failure groups are not confined to low margins (largest failure-group margin = 7.5000); a low dense margin is therefore not a supported sufficient explanation of the STOP.

### excessive_selection_churn_relative_to_l2

- 判定：`not_supported_by_observed_association`；已确立事实：`False`
- 说明：Real-slot Jaccard is low overall (mean 0.2061), but the unrepaired challenges have the highest mean Jaccard (0.3108) and the unchanged-correct group the lowest (0.1749), so churn alone does not order the outcomes.

### unsigned_absolute_gradient_mass

- 判定：`plausible_but_unproven`；已确立事实：`False`
- retained_mass_advantage_positive_all_24：`True`
- failure_group_mean_advantage：`0.23989728163288765`
- unchanged_correct_mean_advantage：`0.35270098401871314`
- 说明：The Oracle retains more recorded absolute |grad*activation| mass than corrected L2 in every sample, yet still regresses 3 controls; the observed failure groups have a lower mean advantage. Because the score is unsigned and K96 pruning is nonlinear and discrete, retaining more absolute mass does not by itself imply the correct-answer margin survives. This is a plausible but untested mechanism.

### context_anchor_rule_interaction

- 判定：`cannot_be_isolated_from_recorded_metadata`；已确立事实：`False`
- 说明：Per-slot scores are not stored for either arm and there is no context-reservation / anchor ablation arm in this pilot, so the context/anchor rule interaction cannot be isolated from the frozen records. Context-representative overlap is measurable and reported, but it is descriptive only.

## 7. 可结论 / 不可结论

**可以结论：**
- the frozen pilot decision is PILOT_STOP (10/12 repairs, 3/12 control regressions, parse 24/24, exact K96/87+9 layout 24/24) and every frozen input hash, record hash, id, transition label and layout verified
- the direct absolute-gradient full-replacement teacher repaired 10/12 challenges but produced 3/12 control regressions and left 2/12 challenges unrepaired, so full replacement fails the preregistered control-regression gate
- Oracle and corrected L2 select largely different slots (mean real-slot Jaccard 0.2061); the Oracle retains a larger fraction of its recorded absolute gradient saliency mass on the real slots (mean paired advantage 0.3136)
- the observed associations between margin/G/Jaccard/retained-mass advantage and the signed transition outcome are exploratory only at n=24 with an outcome-enriched manifest

**不能结论：**
- any official CV-Bench accuracy, deployability, FPGA/latency/speedup or representative-sampling statement
- a causal mechanism for the 3 regressions or the 2 unrepaired challenges; n=24 outcome-enriched samples cannot support it
- that a lower dense answer margin alone explains the failures (the unrepaired challenge includes a 7.5-margin sample)
- that excessive selection churn relative to L2 explains the failures (the unrepaired challenges have the highest real-slot Jaccard)
- that the unsigned absolute |grad*activation| signal preserves the correct-answer margin after the nonlinear discrete K96 pruning; the pilot only records retained saliency mass, not a margin after pruning
- that the context/anchor reservation rule causes any observed difference; the per-slot scores are not recorded and no context/anchor ablation arm exists in this pilot
- any tuned numeric mixture, threshold, K or anchor change on these 24 samples

## 8. 机器可读决策边界

```json
{
  "direct_absolute_gradient_full_replacement_teacher": "STOPPED",
  "preregistered_pilot_status": "PILOT_STOP",
  "preregistered_stop_overturned": false,
  "reproducible_on_frozen_records": true,
  "thresholds_K_anchors_or_mixtures_tuned_on_24_samples": false,
  "max_next_mechanism_classes": 1,
  "recommended_next_mechanism_class": "conservative_l2_backbone_residual_task_aware_reranking",
  "recommendation_status": "unproven_hypothesis_for_next_falsifiable_test",
  "recommendation_evidence": {
    "retained_mass_advantage_positive_all_24": true,
    "failure_group_mean_advantage_lt_unchanged_correct_mean": true,
    "full_replacement_repairs": 10,
    "full_replacement_regressions": 3
  },
  "numeric_mixture_specified": false,
  "note": "The direct absolute-gradient full-replacement teacher remains stopped and the preregistered STOP is not overturned. If (and only if) the next step is a bounded, conservative residual re-ranking on the corrected-L2 backbone, it is proposed here solely as an unproven, falsifiable hypothesis; no numeric mixture, threshold, K or anchor is specified or tuned on these 24 samples."
}
```

禁止据此报告声称：deployable selector、paper accuracy、official CV-Bench accuracy from 24 enriched samples、CV-Bench accuracy improvement confirmed、FPGA speedup or latency、representative sampling、using the 24 enriched samples for training or threshold tuning。
