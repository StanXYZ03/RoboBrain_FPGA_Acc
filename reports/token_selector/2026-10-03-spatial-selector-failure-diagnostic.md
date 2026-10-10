# 空间学生部署链路失效分解（探索性诊断）

- 日期：2026-10-03；状态：`SPATIAL_SELECTOR_FAILURE_DIAGNOSED`
- 判因标签：**HEADROOM_PRESENT**
- 这是**已使用 validation 的探索性诊断**，只能指导下一次预注册，不能作为最终精度结论，且未触碰 internal test。

## 关键 K48 图像级均值 delta（mode − l2_anchor，retained spatial-target mass）

| 对比 | delta |
| --- | --- |
| semantic_anchor_minus_l2_anchor | +0.237814 |
| combined_lambda_0.50_minus_l2_anchor | +0.246583 |
| semantic_plain_minus_l2_plain | +0.261678 |
| oracle_anchor_minus_l2_anchor | +0.737096 |
| combined_lambda_0.50_grounding_minus_l2_anchor | +0.121060 |
| combined_lambda_0.50_pointing_minus_l2_anchor | +0.371635 |

## Lambda sweep（K48 overall，anchor，retained mass mean；仅描述）

| lambda | mean mass | delta vs l2_anchor | jaccard vs l2_anchor |
| --- | --- | --- | --- |
| 0.00 | 0.150366 | +0.000000 | 1.0000 |
| 0.05 | 0.208426 | +0.058060 | 0.7722 |
| 0.10 | 0.267889 | +0.117523 | 0.6004 |
| 0.20 | 0.327170 | +0.176804 | 0.3877 |
| 0.30 | 0.366607 | +0.216242 | 0.2803 |
| 0.50 | 0.396948 | +0.246583 | 0.1797 |
| 0.70 | 0.401517 | +0.251152 | 0.1320 |
| 1.00 | 0.388179 | +0.237814 | 0.0864 |

## 分任务（K48，lambda=0.5 anchor 相对 L2+anchor）

| 任务 | delta mass | delta ndcg |
| --- | --- | --- |
| grounding | +0.121060 | +0.142116 |
| pointing | +0.371635 | +0.141228 |

## Headroom（oracle+anchor 相对 L2+anchor，K48 retained mass）

- mean delta = +0.737096；门槛 = 0.1；HEADROOM_PRESENT = True

## 各模式 K48 overall 指标（图像级 median-of-seeds 后汇总）

| 模式 | retained mass mean | mass delta | ndcg mean | ndcg delta | jaccard vs L2+anchor | better/equal/worse |
| --- | --- | --- | --- | --- | --- | --- |
| l2_plain | 0.150420 | +0.000055 | 0.135870 | +0.000695 | 0.9574 | 6/27/5 |
| l2_anchor | 0.150366 | +0.000000 | 0.135175 | +0.000000 | 1.0000 | 0/38/0 |
| semantic_plain | 0.412098 | +0.261733 | 0.313493 | +0.178318 | 0.0871 | 28/0/10 |
| semantic_anchor | 0.388179 | +0.237814 | 0.303065 | +0.167890 | 0.0864 | 28/0/10 |
| combined_anchor_lambda_0.00 | 0.150366 | +0.000000 | 0.135175 | +0.000000 | 1.0000 | 0/38/0 |
| combined_anchor_lambda_0.05 | 0.208426 | +0.058060 | 0.163816 | +0.028641 | 0.7722 | 24/11/3 |
| combined_anchor_lambda_0.10 | 0.267889 | +0.117523 | 0.193343 | +0.058168 | 0.6004 | 28/7/3 |
| combined_anchor_lambda_0.20 | 0.327170 | +0.176804 | 0.233259 | +0.098084 | 0.3877 | 28/3/7 |
| combined_anchor_lambda_0.30 | 0.366607 | +0.216242 | 0.259906 | +0.124731 | 0.2803 | 28/2/8 |
| combined_anchor_lambda_0.50 | 0.396948 | +0.246583 | 0.279194 | +0.144019 | 0.1797 | 28/0/10 |
| combined_anchor_lambda_0.70 | 0.401517 | +0.251152 | 0.284897 | +0.149722 | 0.1320 | 28/0/10 |
| combined_anchor_lambda_1.00 | 0.388179 | +0.237814 | 0.303065 | +0.167890 | 0.0864 | 28/0/10 |
| oracle_anchor | 0.887461 | +0.737096 | 0.974950 | +0.839775 | 0.1071 | 38/0/0 |

## 判因说明（由本报告数值实时生成，非手写）

- Frozen diagnosis labels (fixed order): HEADROOM_PRESENT. K48 image-level mean retained-spatial-target-mass delta vs L2+anchor: semantic+anchor +0.237814, combined(lambda=0.5)+anchor +0.246583, oracle+anchor +0.737096. On the preregistered sign the retained-mass deployment link did NOT fail. The prior report's G3 primary test is sign-flipped relative to its stated definition (see prior_report_consistency).
- 任务内异质性：combined lambda=0.5 + anchor is better on 28/38 images and worse on 10/38; negative grid strata: 10x16 (G=160, delta -0.487764). Lowest grid-stratum delta: 10x16 (-0.487764).
- retained-mass 部署链路健康（semantic+anchor 与 lambda=0.5 combined+anchor 均 > L2+anchor）：**True**

## 与既有报告的符号一致性（只读核对）

- 既有报告 `reports/token_selector/2026-10-03-spatial-student-trainval.json` 的 `B_selector_minus_L2_anchor_mass_k48` mean = -0.246583（CI [-0.352744, -0.141682]）
- 按预注册定义 selector − L2+anchor 重算 mean = +0.246583
- 与定义符号一致：**False**
- 说明：Prior aggregate_validation computes base - metric = l2_anchor - selector for this key, while its definition states selector - l2_anchor; the reported mean is therefore the negative of the preregistered definition. Reported only; the prior report is not modified.

## 隔离与溯源

- 打开 NPZ 数：76（仅 val：True）
- test NPZ 打开：[]；train NPZ 打开：[]
- 预注册 config 运行前后 SHA 不变：True
- 预注册 report 运行前后 SHA 不变：True
- 所有 pin 输入不变：True
- val NPZ aggregate 与预注册一致：True

## 限制

- Exploratory diagnostic on already-used validation; not a final accuracy conclusion.
- No internal test NPZ was resolved, hashed, opened or scored.
- Lambda sweep is descriptive only; no significance or promotion claim.
- The oracle+anchor mode is an observational reachable upper bound, not a deployable selector.
- Image-level statistics treat the 38 images as independent units.
- This run reuses the frozen spatial targets, scorer, combine and anchor rules; it does not retrain.
- The prior report's G3 primary test is computed as l2_anchor - selector by aggregate_validation, the negative of its stated selector - l2_anchor definition; this decomposition uses the stated sign and reports the discrepancy without editing the prior report.
- Retained-mass deltas are descriptive and are not a significance or promotion test; the large oracle headroom only bounds achievable mass, not task accuracy.
