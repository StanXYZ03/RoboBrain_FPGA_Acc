# Layer-5 context-pooling ablation（screening100 诊断）

- 生成时间：`2026-10-04T01:14:03.147981+00:00`
- 范围：**diagnostic screening100; NOT a formal full2638 CV-Bench conclusion**
> ⚠️ Fixed stratified 100-sample screening: a single-variable context-pooling attribution signal only, NOT a formal full2638 CV-Bench conclusion. It must not be quoted as the official CV-Bench accuracy of any approach.

- 正式主指标（唯一）：`score(...)["cv_bench_accuracy"]` = `((ADE20K + COCO)/2 + Omni3D)/2`；`unweighted_sample_accuracy` 仅为次要诊断，**不得写作 CV-Bench 主分**，也不参与决策。
- 禁用主指标：pointing, grounding, ndcg, retained_mass
- dense 仅为参考（reference only），不参与决策。

## 1. 固定配置

- 样本：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/datasets/cv_bench_screening100.jsonl`；K=96；切点：5；region_grid=3；local_cap=16
- dtype=bfloat16；attention=sdpa；decoding=greedy_argmax；score_mode=l2；bootstrap seed=20261004 / repeats=5000
- 消融：SOURCE_CONTEXT pooled group -> raw representative 2x2 group

## 2. 输入合同校验

- screening 样本数：100
- pooled layer5 样本数：100
- representative layer5 样本数：100
- 100 个 sample_id 完全一致：✅
- score_mode 全为 `l2`：✅
- requested_budget 全为 96（K=96）：✅
- prune_after_layer 全为 5：✅
- 输入 grid 两臂一致（(tuple(image_grid_thw), full_patch_tokens, full_llm_visual_tokens)）：✅
- selection metadata/indices/positions 两臂逐元素一致（selected_group_indices, source_type, region_id, valid）：✅
- representative 臂声明 context_group_values_source=`representative_raw_group`：✅
- 输入 SHA 复核：✅（14 项）

## 3. 三臂正式主分

| 臂 | 样本 | **CV-Bench 主分**（`cv_bench_accuracy`） | unweighted_sample_accuracy（次要） | 相对 dense 保留（参考） |
|---|---:|---:|---:|---:|
| dense | 100 | **0.891667** | 0.900000 | 1.000000 |
| layer5_pooled_block5 | 100 | **0.741667** | 0.750000 | 0.831776 |
| layer5_representative_block5 | 100 | **0.729167** | 0.740000 | 0.817757 |

## 4. 配对分层 bootstrap（95% CI，主指标）

| 对比 | 点估计 | 95% CI 下界 | 95% CI 上界 |
|---|---:|---:|---:|
| representative − pooled | -0.012500 | -0.066667 | +0.037500 |
| representative − dense | -0.162500 | -0.237500 | -0.091667 |
| pooled − dense | -0.150000 | -0.225000 | -0.075000 |

## 5. 预注册决策（仅用 representative − pooled 主指标）

- 结论：**CONTEXT_POOLING_HARM_NOT_CONFIRMED**
- 依据：representative_minus_pooled CI lower -0.066667 <= 0.02
- representative−pooled 95% CI：[-0.066667, +0.037500]；阈值 > 0.02
- dense 作用：reference_only（不参与决策）

## 6. 输入 SHA256

| 输入 | SHA256 |
|---|---|
| preregistered_config | `e5230e9028ae004ee02db515f36fe601562cd5789cc5435840412a22c9967eec` |
| screening_manifest | `31bf313286c81fa50911ca3822118184ac03ca74df2cf2c260250cab2768b50c` |
| full_manifest | `22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91` |
| dense_full_jsonl | `3ff3b77ea4bddae96a8231c653bf08db1ce396a9c125fb414466417b1ea69f63` |
| pooled_layer5_l2_screening_jsonl | `5395d5e2846022a75f8bb0317d4fa8ccd8e62771582437b55811bd08c64a571e` |
| representative_layer5_l2_screening_jsonl | `f09f5a539141d389d362af1fd2284fe4ab8cc5edc384c1a1904408ae491e796c` |

## 7. 限制

- Screening100 is a fixed diagnostic subset, not the formal 2638-sample benchmark.
- Both Selector arms are software L2 only; no FPGA, latency or acceleration claim.
- Dense is the frozen custom dense path and is reference only; it never drives the decision.
- The decision uses score(...)["cv_bench_accuracy"] exclusively.
- Forbidden primary metrics (pointing/grounding/ndcg/retained_mass) are never reported.
- The single changed variable is the SOURCE_CONTEXT group value; all selection metadata, indices and positions are identical.
- Neither arm uses the B1-trained Wv projection.

> CPU-only 只读汇总：未改原始 JSONL、正式 quality report 或 `configs/evaluation_protocol.json`；无 FPGA/时延/加速声明。
