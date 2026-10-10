# Layer-5 L2 vs B1 L2 vs dense（screening100 诊断）

- 生成时间：`2026-10-04T00:59:30.358728+00:00`
- 范围：**diagnostic screening100; NOT a formal full2638 CV-Bench conclusion**
> ⚠️ Fixed stratified 100-sample screening: a cut-point attribution signal only, NOT a formal full2638 CV-Bench conclusion. It must not be quoted as the official CV-Bench accuracy of any approach.

- 正式主指标（唯一）：`score(...)["cv_bench_accuracy"]` = `((ADE20K + COCO)/2 + Omni3D)/2`；`unweighted_sample_accuracy` 仅为次要诊断，**不得写作 CV-Bench 主分**，也不参与决策。
- 禁用主指标：pointing, grounding, ndcg, retained_mass
- dense 仅为参考（reference only），不参与决策。

## 1. 固定配置

- 样本：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/datasets/cv_bench_screening100.jsonl`；K=96；切点：B1=0 / layer5=5；region_grid=3；local_cap=16
- dtype=bfloat16；attention=sdpa；decoding=greedy_argmax；score_mode=l2；bootstrap seed=20261004 / repeats=5000
- layer5 不使用 B1 训练的 Wv：✅

## 2. 输入合同校验

- screening 样本数：100
- B1 L2 样本数：100
- layer5 L2 样本数：100
- 100 个 sample_id 完全一致：✅
- score_mode 全为 `l2`：✅
- requested_budget 全为 96（K=96）：✅
- layer5 prune_after_layer 全为 5：✅
- B1 prune_after_layer 缺失或 0：✅
- layer5 输入 grid 与 B1 一致（(tuple(image_grid_thw), full_patch_tokens, full_llm_visual_tokens)）：✅
- 输入 SHA 复核：✅（15 项）

## 3. 三臂正式主分

| 臂 | 样本 | **CV-Bench 主分**（`cv_bench_accuracy`） | unweighted_sample_accuracy（次要） | 相对 dense 保留（参考） |
|---|---:|---:|---:|---:|
| dense | 100 | **0.891667** | 0.900000 | 1.000000 |
| b1_l2_block0 | 100 | **0.662500** | 0.660000 | 0.742991 |
| layer5_l2_block5 | 100 | **0.741667** | 0.750000 | 0.831776 |

## 4. 配对分层 bootstrap（95% CI，主指标）

| 对比 | 点估计 | 95% CI 下界 | 95% CI 上界 |
|---|---:|---:|---:|
| layer5 − b1 | +0.079167 | +0.016667 | +0.145833 |
| layer5 − dense | -0.150000 | -0.225000 | -0.075000 |
| b1 − dense | -0.229167 | -0.308333 | -0.150000 |

## 5. 预注册决策（仅用 layer5 − B1 主指标）

- 结论：**CUTPOINT_EFFECT_NOT_CONFIRMED**
- 依据：layer5_minus_b1 CI lower 0.016667 <= 0.02
- layer5−B1 95% CI：[+0.016667, +0.145833]；阈值 > 0.02
- dense 作用：reference_only（不参与决策）

## 6. 输入 SHA256

| 输入 | SHA256 |
|---|---|
| preregistered_config | `1dd6eb63c2b215e8d2459a30ee500a9361b12fe94f51e6b20ac5a7f9ec0f8891` |
| screening_manifest | `31bf313286c81fa50911ca3822118184ac03ca74df2cf2c260250cab2768b50c` |
| full_manifest | `22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91` |
| dense_full_jsonl | `3ff3b77ea4bddae96a8231c653bf08db1ce396a9c125fb414466417b1ea69f63` |
| b1_l2_screening_jsonl | `983300c1816a1d54a2f89354c14606cdc69b7fc4528fa299ce875c07d13849c6` |
| layer5_l2_screening_jsonl | `5395d5e2846022a75f8bb0317d4fa8ccd8e62771582437b55811bd08c64a571e` |

## 7. 限制

- Screening100 is a fixed diagnostic subset, not the formal 2638-sample benchmark.
- Both Selector arms are software L2 only; no FPGA, latency or acceleration claim.
- Dense is the frozen custom dense path and is reference only; it never drives the decision.
- The decision uses score(...)["cv_bench_accuracy"] exclusively.
- Forbidden primary metrics (pointing/grounding/ndcg/retained_mass) are never reported.
- The layer-5 arm never uses the B1-trained Wv projection.

> CPU-only 只读汇总：未改原始 JSONL、正式 quality report 或 `configs/evaluation_protocol.json`；无 FPGA/时延/加速声明。
