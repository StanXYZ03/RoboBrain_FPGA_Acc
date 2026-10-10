# 空间监督目标可行性审计（TRAIN-only，零更新）

- 决策：`SPATIAL_SUPERVISION_SIGNAL_PRESENT`
- 训练记录/图像：318 / 159
- 隔离：`val_test_npz_opened=[]`，opened_npz_count=318
- 预注册：`bfa1963597993ea9e3482735f16b2d430638ac3bf8e0682b5078f4e112974490`

## 目标定义（冻结，不调参，不与语义 teacher 混合）

- `grounding_area_target`：bbox 与网格单元精确交集面积，按总面积归一化。
- `pointing_bilinear_target`：point→连续中心坐标→四邻可分离双线性权重，合并重复边界索引并归一化；无高斯、无带宽参数。

## 同图任务分离（image 为独立单位）

### 语义 teacher（缓存 teacher_soft_target）

| 指标 | count | mean | median | std | p5 | p95 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| js_nats | 159 | 0.051067 | 0.009236 | 0.095606 | 0.000910 | 0.266627 | 0.000174 | 0.497356 |
| symmetric_kl | 159 | 0.385209 | 0.051271 | 0.835520 | 0.004203 | 2.090502 | 0.000723 | 5.024718 |
| tv | 159 | 0.127210 | 0.051953 | 0.180563 | 0.009564 | 0.558156 | 0.003025 | 0.835933 |

- argmax 不同比例：0.119497

### 空间 teacher（grounding 面积 vs pointing 双线性）

| 指标 | count | mean | median | std | p5 | p95 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| js_nats | 159 | 0.518015 | 0.539648 | 0.115080 | 0.292991 | 0.646514 | 0.121034 | 0.671053 |
| symmetric_kl | 159 | 11.248110 | 11.831743 | 1.944630 | 7.224536 | 13.055809 | 2.869189 | 13.405080 |
| tv | 159 | 0.866312 | 0.903662 | 0.118069 | 0.625317 | 0.979157 | 0.386761 | 0.991747 |

- argmax 不同比例：0.962264

## 配对差 spatial − semantic（bootstrap 2000 / sign-flip 10000，seed 20261003）

| 指标 | n | mean | median | std | CI95 low | CI95 high | p(双侧) | Cohen dz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diff_js_nats | 159 | 0.466949 | 0.502143 | 0.165044 | 0.440491 | 0.492981 | 0.000100 | 2.829235 |
| diff_tv | 159 | 0.739101 | 0.817175 | 0.235751 | 0.700647 | 0.774453 | 0.000100 | 3.135090 |
| diff_k48_disagreement | 159 | 0.492348 | 0.530292 | 0.378132 | 0.433178 | 0.552661 | 0.000100 | 1.302054 |
| diff_k64_disagreement | 159 | 0.412434 | 0.474828 | 0.392853 | 0.350117 | 0.474256 | 0.000100 | 1.049843 |
| diff_k96_disagreement | 159 | 0.282972 | 0.273101 | 0.386262 | 0.223701 | 0.343878 | 0.000100 | 0.732591 |

无多重候选搜索，报告原始 p 值，不自创 Holm 校正。

## 语义/空间兼容性（Spearman 与 full NDCG，relevance=teacher_soft_target）

| 范围 | Spearman mean | NDCG mean |
| --- | --- | --- |
| overall | 0.125667 | 0.388966 |
| grounding | 0.180759 | 0.411134 |
| pointing | 0.070574 | 0.366799 |

低一致性只说明目标不同，并不自动说明某一方错误。

## 硬件规整基线（L2 / 3x3 anchor，K=48/64/96）

| 基线 | K | 范围 | NDCG mean | 保留目标质量 mean | count |
| --- | --- | --- | --- | --- | --- |
| L2 | 48 | overall | 0.134532 | 0.167937 | 318 |
| L2 | 48 | grounding | 0.217840 | 0.173358 | 159 |
| L2 | 48 | pointing | 0.051223 | 0.162515 | 159 |
| L2 | 64 | overall | 0.154315 | 0.240022 | 318 |
| L2 | 64 | grounding | 0.240733 | 0.238987 | 159 |
| L2 | 64 | pointing | 0.067897 | 0.241056 | 159 |
| L2 | 96 | overall | 0.187391 | 0.348751 | 318 |
| L2 | 96 | grounding | 0.287698 | 0.357225 | 159 |
| L2 | 96 | pointing | 0.087084 | 0.340277 | 159 |
| anchor | 48 | overall | 0.135611 | 0.171013 | 318 |
| anchor | 48 | grounding | 0.219177 | 0.175990 | 159 |
| anchor | 48 | pointing | 0.052045 | 0.166036 | 159 |
| anchor | 64 | overall | 0.155977 | 0.244024 | 318 |
| anchor | 64 | grounding | 0.243338 | 0.243704 | 159 |
| anchor | 64 | pointing | 0.068616 | 0.244344 | 159 |
| anchor | 96 | overall | 0.190026 | 0.356207 | 318 |
| anchor | 96 | grounding | 0.291110 | 0.363556 | 159 |
| anchor | 96 | pointing | 0.088943 | 0.348857 | 159 |

anchor 行 NDCG 为**仅对 anchor 选定组按 L2 降序排序**的受限 NDCG（低索引打破平局，分母为全体组 ideal DCG@K）；与其保留目标质量使用同一选定集合。

## 更正记录：anchor NDCG 选择集缺陷（旧值 vs 校正值）

- 标签：`CORRECTION_ANCHOR_NDCG_RESTRICTED_TO_ANCHOR_SELECTED_GROUPS`（2026-10-03）
- 缺陷：anchor_baseline_record computed NDCG as ndcg_at_k(scores, target, effective), which ignores the anchor-selected set and therefore duplicated the pure-L2 NDCG even when anchor retained mass differed.
- 旧定义：`ndcg_at_k(all L2 scores, spatial_target, effective)`
- 校正定义：`ndcg_for_selected_indices(anchor_selected, L2 scores, spatial_target, effective): DCG over only the selected groups ordered by descending L2 score with deterministic low-index tie-break, divided by the all-group ideal DCG@K`
- 保留目标质量不变：`True`（最大绝对差 0.000000）
- 主指标/JS/TV 决策不变：`True`

| K | 范围 | anchor NDCG 旧 | anchor NDCG 校正 |
| --- | --- | --- | --- |
| 48 | overall | 0.134532 | 0.135611 |
| 48 | grounding | 0.217840 | 0.219177 |
| 48 | pointing | 0.051223 | 0.052045 |
| 64 | overall | 0.154315 | 0.155977 |
| 64 | grounding | 0.240733 | 0.243338 |
| 64 | pointing | 0.067897 | 0.068616 |
| 96 | overall | 0.187391 | 0.190026 |
| 96 | grounding | 0.287698 | 0.291110 |
| 96 | pointing | 0.087084 | 0.088943 |

- The pre-correction anchor NDCG equals the pure-L2 NDCG because the ranking ignored the anchor-selected set.  Retained target mass already used the selected set and is unchanged.  The preregistered spatial targets, JS/TV decision rule, data split, inputs and decision label were not modified.

## 限制

- The audit does not train anything, does not build a student and does not claim the target is learnable.
- No downstream task accuracy (CV-Bench or otherwise) is measured or claimed.
- No FPGA latency, resource or speedup number is measured or claimed.
- The semantic comparator is the existing cached max-over-prompt-token teacher only; no new semantic model is loaded.
- The grounding area target and the pointing bilinear target have different structural support (an area over many cells vs at most four cells), so part of the same-image spatial separation reflects that structural difference and not only task-relevant geometry.
- Low spatial/semantic agreement means the objectives differ, not automatically that either target is wrong.
- The hardware-regular anchor baseline is a descriptive reference over the L2 saliency score in this zero-update audit; it is not a trained selector.
- All numbers are TRAIN-split, image-level and CPU float64 internal measurements, never a promotion gate.

> 决策规则：仅当所有完整性检查通过且 `spatial-semantic` 的 95% bootstrap CI 下界对 JS 与 TV 同时 > 0 时，才判 `SPATIAL_SUPERVISION_SIGNAL_PRESENT`；top-K 仅为描述性，不能覆盖该规则。
> 即使 signal present，可学习性与下游任务精度仍未验证；本报告不声称精度提升或 FPGA 加速。
