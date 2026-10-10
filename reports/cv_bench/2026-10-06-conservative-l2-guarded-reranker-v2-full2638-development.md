# L2 守卫整数重排 v2（full2638 发展性判定）

- 机制：`conservative_l2_guarded_rerank_v2`
- 判定：`DEVELOPMENT_STOP`
- 理由：failed gates: score_at_least_90pct_dense
- 口径：**仅发展性证据**，不是论文确认、不是可部署性、不是 FPGA 时延/加速主张。

## 1. 唯一主指标

`score(...)["cv_bench_accuracy"]`（官方 CV-Bench 组合分）。

## 2. 分数

| 臂 | cv_bench_accuracy | 未加权 | parse |
| --- | --- | --- | --- |
| new | 0.783740 | 0.780136 | 1.000000 |
| corrected_l2_integer | 0.766852 | 0.763457 | 1.000000 |
| dense | 0.872730 | 0.868082 | 1.000000 |

## 3. 配对比较（new − corrected integer-L2）

- 点估计：0.016889
- 95% CI：[0.006158, 0.027154]
- 分层：task::source_dataset，seed 20261006，repeats 5000

## 4. 布局审计

- 压缩样本：2495（其中 87 real + 9 context：2495）
- identity bypass：143
- 发生交换的样本数：2495

## 5. 门

| 门 | 通过 |
| --- | --- |
| all_2638_complete_parseable | 是 |
| exact_k96_layout_contract | 是 |
| score_at_least_90pct_dense | 否 |
| paired_ci_lower_nonnegative | 是 |
| no_forbidden_leakage | 是 |

## 6. 输入 SHA256

| 输入 | SHA256 |
| --- | --- |
| full_manifest | `22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91` |
| new_arm_jsonl | `3018f7593d31fd02854f3f63f992b6fc6293ba64c24185e2905877a71d88fdb1` |
| corrected_l2_integer_jsonl | `d0e302f6b91279374ddd507db4d300ca0c29e16f522db57f47c8de059f103a5f` |
| dense_full_jsonl | `3ff3b77ea4bddae96a8231c653bf08db1ce396a9c125fb414466417b1ea69f63` |

## 7. 限制

- Development evidence only; the full CV-Bench 2638 was observed repeatedly and is never an untouched confirmation set.
- Passing is permission to acquire/run a fresh untouched external spatial-reasoning benchmark, not an official confirmation, deployability claim, FPGA latency claim or paper accuracy claim.
- The Selector path is a software bridge; no FPGA/board/latency claim.
- Do not tune v2 after seeing full2638.
- The prior prompt-fusion score is a descriptive same-budget comparator only and is not a redundant gate.
