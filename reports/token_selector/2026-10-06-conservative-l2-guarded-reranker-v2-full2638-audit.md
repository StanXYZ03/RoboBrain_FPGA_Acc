# L2 守卫整数重排 v2 full2638 发展性结果只读审计

- 日期：2026-10-06
- 审计结论：**`AUDIT_PASS`**
- 冻结发展判定：**`DEVELOPMENT_STOP`**（理由：failed gates: score_at_least_90pct_dense）
- 口径：CPU-only、只读；`gpu_runs=0`、`audited_gpu_runs=1`、`audited_samples=2638`；未加载模型权重/未联网/未训练/未装依赖。

## 1. 被审计原始 full2638

- 路径：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/artifacts/token_selector/2026-10-06-conservative-l2-guarded-reranker-v2-full2638/conservative_l2_guarded_rerank_v2_k96_full2638.jsonl`
- 记录数：2638；唯一 `sample_id`：2638
- SHA256（现场重算）：`3018f7593d31fd02854f3f63f992b6fc6293ba64c24185e2905877a71d88fdb1`；与声明一致：`True`
- 缺失 ID：无；多余 ID：无

## 2. 八项审计检查

| 审计项 | 通过 | 子检查数 |
| --- | --- | --- |
| `1_raw_sha_2638_unique_manifest_ids_and_record_hashes` | ✅ | 7 |
| `2_model_config_checkpoint_source_pins_and_leakage_flags` | ✅ | 23 |
| `3_identity_bypass_contract_and_exact_k96_87_real_9_context` | ✅ | 7 |
| `4_anchors_frozen_same_region_swaps_counts_and_nonempty_context` | ✅ | 5 |
| `5_swap_gain_identities_and_deterministic_array_hashes` | ✅ | 13 |
| `6_frozen_official_primary_metric_all_four_arms` | ✅ | 6 |
| `7_stratified_paired_bootstrap_5000_seed_20261006_and_90pct_dense_floor` | ✅ | 4 |
| `8_decision_gates_recomputed_development_stop` | ✅ | 3 |

- 全部审计检查通过：`True`；失败项：无

## 3. 唯一主指标（冻结 `score(...)["cv_bench_accuracy"]`）

| 臂 | cv_bench_accuracy | 未加权 | 样本数 |
| --- | --- | --- | --- |
| new | 0.7837404943432142 | 0.7801364670204700 | 2638 |
| dense | 0.8727298774444869 | 0.8680818802122821 | 2638 |
| corrected_l2_integer | 0.7668517264725796 | 0.7634571645185747 | 2638 |
| prompt_plus_corrected_l2_integer | 0.7773617325561999 | 0.7733131159969674 | 2638 |

## 4. 配对 bootstrap 与 90% dense 门

- v2 − 修正整数 L2 点估计：0.0168887678706346
- 分层 `task::source_dataset` 95% CI：[0.0061575927506795, 0.0271543129924543]；seed 20261006，repeats 5000
- 90% dense 门（现场计算 `0.9 × 0.8727298774444869`）：0.7854568897000382
- 绝对缺口：0.0017163953568240；相对门：0.0021852190480874；相对 dense：0.0019666971432786
- v2 保留 dense 分数比例：0.8980333028567213

## 5. 决策门（现场重算）

| 门 | 通过 |
| --- | --- |
| `all_2638_complete_parseable` | 是 |
| `exact_k96_layout_contract` | 是 |
| `score_at_least_90pct_dense` | 否 |
| `paired_ci_lower_nonnegative` | 是 |
| `no_forbidden_leakage` | 是 |

- 最终判定：**`DEVELOPMENT_STOP`**；失败门：['score_at_least_90pct_dense']

## 6. 布局与交换摘要

- 压缩样本：2495；identity bypass：143；压缩样本中 0 交换：0

## 7. 冻结输入字节一致性

- 非空 config pin 现场复核通过：36；不匹配：无；待生成（null）：['new_arm_jsonl']

## 8. 发现

- `parse_gate_implementation_defect`（implementation_defect_not_artifact）：the frozen gate all_2638_complete_parseable is computed from score(...)['parse_success_rate'], which is identically 1.0 for the summariser's arm rows because the 'correct' key has no '_correct' substring to rewrite to '_answer'; honest parse_choice shows real failures in every arm
- `model_weight_sha_pin_absent`（informational）：the frozen config SHA-pins the three checkpoint files but not the model weights; model identity is metadata-only (model_path/model_label)

## 9. 限制

- full2638 has been repeatedly observed and is development evidence only; it is not an untouched confirmation set.
- This audit is CPU-only and read-only: no model weights were loaded, no GPU was used, no network, no training and no dependency installation; gpu_runs=0.
- The Selector path is a software bridge; no FPGA latency, board deployment or acceleration claim is made.
- No paper-accuracy or external-validation claim is made; the DEVELOPMENT_STOP outcome grants no external-validation permission for v2.
- Do not tune v2 after observing full2638. The postmortem is diagnostic/development-only and must not be used to select a v3 rule.
- v1 remains permanently PREEXECUTION_DESIGN_STOP and was neither run nor patched.

