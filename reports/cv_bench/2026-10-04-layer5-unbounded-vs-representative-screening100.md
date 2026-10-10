# Layer-5 unbounded real-only vs raw-representative（screening100 诊断）

- 生成时间：`2026-10-04T01:27:24.004219+00:00`
- 范围：**diagnostic screening100; NOT a formal full2638 CV-Bench conclusion**
> ⚠️ Fixed stratified 100-sample screening: a single-variable selection-constraint attribution signal only, NOT a formal full2638 CV-Bench conclusion. It must not be quoted as the official CV-Bench accuracy of any approach.

- 正式主指标（唯一）：`score(...)["cv_bench_accuracy"]` = `((ADE20K + COCO)/2 + Omni3D)/2`；`unweighted_sample_accuracy` 仅为次要诊断，**不得写作 CV-Bench 主分**，也不参与决策。
- 禁用主指标：pointing, grounding, ndcg, retained_mass
- dense 仅为参考（reference only），不参与决策。

## 1. 固定配置

- 样本：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/datasets/cv_bench_screening100.jsonl`；K=96；切点：5；region_grid=3；local_cap=None
- dtype=bfloat16；attention=sdpa；decoding=greedy_argmax；score_mode=l2；bootstrap seed=20261004 / repeats=5000
- 变量：representative=raw representative 2x2 context groups under the frozen hardware-regular selector (context pooling + slot reservation + local_candidate_cap=16)；unbounded selection_rule=unbounded_real_only_global_l2_3x3_anchor；unbounded 无 context pooling / 无 context slot reservation / 无 local_candidate_cap

## 2. 输入合同校验

- screening 样本数：100
- representative layer5 样本数：100
- unbounded layer5 样本数：100
- 100 个 sample_id 完全一致：✅
- score_mode 全为 `l2`：✅
- requested_budget 全为 96（K=96）：✅
- prune_after_layer 全为 5：✅
- 输入 grid 两臂一致（(tuple(image_grid_thw), full_patch_tokens, full_llm_visual_tokens)）：✅
- representative 臂声明 context_group_values_source=`representative_raw_group`：✅
- unbounded 臂 all-real / 无 context（pooling、slot reservation、local_candidate_cap）/ 精确 K 或 identity：✅
- 输入 SHA 复核：✅（16 项）

## 3. 三臂正式主分

| 臂 | 样本 | **CV-Bench 主分**（`cv_bench_accuracy`） | unweighted_sample_accuracy（次要） | 相对 dense 保留（参考） |
|---|---:|---:|---:|---:|
| dense | 100 | **0.891667** | 0.900000 | 1.000000 |
| layer5_representative_block5 | 100 | **0.729167** | 0.740000 | 0.817757 |
| layer5_unbounded_real_only_block5 | 100 | **0.745833** | 0.760000 | 0.836449 |

## 4. 配对分层 bootstrap（95% CI，主指标）

| 对比 | 点估计 | 95% CI 下界 | 95% CI 上界 |
|---|---:|---:|---:|
| unbounded − representative | +0.016667 | -0.041667 | +0.070833 |
| unbounded − dense | -0.145833 | -0.229167 | -0.062500 |
| representative − dense | -0.162500 | -0.237500 | -0.091667 |

## 5. 预注册决策（仅用 unbounded − representative 主指标）

- 结论：**HARDWARE_SELECTION_CONSTRAINT_HARM_NOT_CONFIRMED**
- 依据：unbounded_minus_representative CI lower -0.041667 <= 0.02
- unbounded−representative 95% CI：[-0.041667, +0.070833]；阈值 > 0.02
- dense 作用：reference_only（不参与决策）

## 6. 输入 SHA256

| 输入 | SHA256 |
|---|---|
| preregistered_config | `33d46aa0fa5779bf586fa9262b656bf72caf48eb45f958028cd1ca70e6cff020` |
| screening_manifest | `31bf313286c81fa50911ca3822118184ac03ca74df2cf2c260250cab2768b50c` |
| full_manifest | `22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91` |
| dense_full_jsonl | `3ff3b77ea4bddae96a8231c653bf08db1ce396a9c125fb414466417b1ea69f63` |
| representative_layer5_l2_screening_jsonl | `f09f5a539141d389d362af1fd2284fe4ab8cc5edc384c1a1904408ae491e796c` |
| unbounded_layer5_l2_screening_jsonl | `fd22c5a4b1941a8ad0a2c47bf15f0abcbeaa56553a9f9a790deb6e14054df94c` |

## 7. 限制

- Screening100 is a fixed diagnostic subset, not the formal 2638-sample benchmark.
- Both Selector arms are software L2 only; no FPGA, latency or acceleration claim.
- Dense is the frozen custom dense path and is reference only; it never drives the decision.
- The decision uses score(...)["cv_bench_accuracy"] exclusively.
- Forbidden primary metrics (pointing/grounding/ndcg/retained_mass) are never reported.
- Both arms are raw real groups; they differ only in the selection constraint and indexing (bounded context-pooling selector vs unbounded global-L2 + 3x3 anchor).
- Neither arm uses the B1-trained Wv projection.

> CPU-only 只读汇总：未改原始 JSONL、正式 quality report 或 `configs/evaluation_protocol.json`；无 FPGA/时延/加速声明。
