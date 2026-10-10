# Layer-5 adaptive-budget unbounded real-only（screening100 诊断）

- 生成时间：`2026-10-04T04:09:51.101203+00:00`
- 范围：**diagnostic screening100; NOT a formal full2638 CV-Bench conclusion**
> ⚠️ Fixed stratified 100-sample screening: a diagnostic adaptive-budget signal only, NOT a formal full2638 CV-Bench conclusion. It must not be quoted as the official CV-Bench accuracy of any approach.

- 正式主指标（唯一）：`score(...)["cv_bench_accuracy"]` = `((ADE20K + COCO)/2 + Omni3D)/2`；`unweighted_sample_accuracy` 仅为次要诊断，**不得写作 CV-Bench 主分**，也不参与决策。
- 禁用主指标：pointing, grounding, ndcg, retained_mass
- dense 仅为参考（reference only）；adaptive/dense 比例为项目解释量，**不是 CV-Bench 官方指标**。

## 1. 固定配置与预算公式

- 样本：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/datasets/cv_bench_screening100.jsonl`；切点：5；region_grid=3；local_cap=None
- dtype=bfloat16；attention=sdpa；decoding=greedy_argmax；score_mode=l2；bootstrap seed=20261004 / repeats=5000
- 预算公式：`G = full_llm_visual_tokens; K_raw = max(96, ceil(0.25 * G)); K_eff = min(G, 16 * ceil(K_raw / 16)); identity_bypass if K_eff >= G`
- budget_min=96；retention=0.25；alignment=16；fixed_k96=96
- selection_rule=`unbounded_real_only_global_l2_3x3_anchor`；无 context pooling / 无 context slot reservation / 无 local_candidate_cap

## 2. 输入合同校验

- screening 样本数：100
- dense 全量样本数：2638
- fixed-K96 unbounded 样本数：100
- adaptive 样本数：100
- 100 个 sample_id 完全一致：✅
- score_mode 全为 `l2`：✅
- prune_after_layer 全为 5：✅
- adaptive 预算公式（K_raw/K_eff/retention/identity）逐样本复核：✅
- fixed-K96 精确 K=96 或 G<=96 identity：✅
- adaptive 输入 grid 与 fixed-K96 一致（(tuple(image_grid_thw), full_patch_tokens, full_llm_visual_tokens)）：✅
- 两臂 all-real / 无 context（pooling、slot reservation、local_candidate_cap）：✅
- 输入 SHA 复核：✅（19 项）

## 3. 自适应预算概览

- K_raw 范围：96–897；K_eff 范围：64–912；K_eff 均值：200.480
- retention 均值：0.356865；identity bypass：2 个样本 ['195', '523']
- adaptive K_eff 不超过 fixed-K96=96：❌

## 4. 三臂正式主分

| 臂 | 样本 | **CV-Bench 主分**（`cv_bench_accuracy`） | unweighted_sample_accuracy（次要） |
|---|---:|---:|---:|
| dense | 100 | **0.891667** | 0.900000 |
| layer5_unbounded_real_only_block5 | 100 | **0.745833** | 0.760000 |
| layer5_adaptive_budget_unbounded_real_only_block5 | 100 | **0.762500** | 0.770000 |

- 项目解释量（**非官方指标**）：adaptive/dense=0.855140；fixed-K96/dense=0.836449；adaptive/fixed-K96=1.022346

## 5. 配对分层 bootstrap（95% CI，主指标）

| 对比 | 点估计 | 95% CI 下界 | 95% CI 上界 |
|---|---:|---:|---:|
| adaptive − fixed_k96 | +0.016667 | -0.045833 | +0.083333 |
| adaptive − dense | -0.129167 | -0.204167 | -0.054167 |
| fixed_k96 − dense | -0.145833 | -0.229167 | -0.062500 |

## 6. 诊断结论（仅主指标，dense 不参与）

- 类型：descriptive_only；结果：**DIAGNOSTIC_NO_PROMOTION_THRESHOLD**
- 说明：screening100 diagnostic only: the paired stratified bootstrap CI on score(...)["cv_bench_accuracy"] is reported descriptively; no CONFIRMED/NOT_CONFIRMED promotion claim is made.

## 7. 输入 SHA256

| 输入 | SHA256 |
|---|---|
| preregistered_config | `55a39d9f1b2f0df264391e6e8c99198d8c4086221116bbbd2509d3e00e52025f` |
| screening_manifest | `31bf313286c81fa50911ca3822118184ac03ca74df2cf2c260250cab2768b50c` |
| full_manifest | `22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91` |
| dense_full_jsonl | `3ff3b77ea4bddae96a8231c653bf08db1ce396a9c125fb414466417b1ea69f63` |
| fixed_k96_unbounded_screening_jsonl | `fd22c5a4b1941a8ad0a2c47bf15f0abcbeaa56553a9f9a790deb6e14054df94c` |
| adaptive_budget_screening_jsonl | `8f9442dd0f533e8a475fd51784682b729a64decb31f3166a7f32d72b4405d3e3` |

## 8. 限制

- Screening100 is a fixed diagnostic subset, not the formal 2638-sample benchmark.
- The adaptive budget is derived from the measured real G; it is a software selection diagnostic and makes no FPGA/latency/acceleration claim.
- Both Selector arms use the same frozen unbounded real-only global-L2 + 3x3 anchor algorithm and raw real groups; only the per-sample budget differs.
- Dense is the frozen custom dense path and is reference only; it never drives the decision.
- Only score(...)["cv_bench_accuracy"] is reported as the primary metric; the adaptive/dense ratio is project interpretation, not an official CV-Bench metric.
- Forbidden primary metrics (pointing/grounding/ndcg/retained_mass) are never reported.
- Neither arm uses the B1-trained Wv projection.

> CPU-only 只读汇总：未改原始 JSONL、正式 quality report 或 `configs/evaluation_protocol.json`；无 FPGA/时延/加速声明。
