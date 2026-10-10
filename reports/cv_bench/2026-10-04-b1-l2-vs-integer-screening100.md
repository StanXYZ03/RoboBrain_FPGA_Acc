# B1 L2 vs current integer Selector（screening100 诊断）

- 生成时间：`2026-10-04T00:43:08.785072+00:00`
- 范围：**diagnostic screening100; NOT a formal full2638 CV-Bench conclusion**
> ⚠️ Fixed stratified 100-sample screening: an attribution signal only, NOT a formal full2638 CV-Bench conclusion. It must not be quoted as the official CV-Bench accuracy of any approach.

- 正式主指标（唯一）：`score(...)["cv_bench_accuracy"]` = `((ADE20K + COCO)/2 + Omni3D)/2`；`unweighted_sample_accuracy` 仅为次要诊断，**不得写作 CV-Bench 主分**，也不参与决策。
- 禁用主指标：pointing, grounding, ndcg, retained_mass

## 1. 固定配置

- 样本：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/datasets/cv_bench_screening100.jsonl`；K=96；切点：B1 = after vision block 0；region_grid=3；local_cap=16
- dtype=bfloat16；attention=sdpa；decoding=greedy_argmax；bootstrap seed=20261004 / repeats=5000

## 2. 输入合同校验

- screening 样本数：100
- L2 样本数：100
- 100 个 sample_id 完全一致：✅
- 无重复 sample_id：✅
- L2 score_mode 全为 `l2`：✅
- L2 requested_budget 全为 96：✅
- L2 输入 grid 与 current 一致（(tuple(image_grid_thw), full_patch_tokens, full_llm_visual_tokens)）：✅

## 3. 三臂正式主分

| 臂 | 样本 | **CV-Bench 主分**（`cv_bench_accuracy`） | unweighted_sample_accuracy（次要） | 相对 dense 保留 |
|---|---:|---:|---:|---:|
| dense | 100 | **0.891667** | 0.900000 | 1.000000 |
| current_integer | 100 | **0.812500** | 0.820000 | 0.911215 |
| l2 | 100 | **0.662500** | 0.660000 | 0.742991 |

## 4. 配对分层 bootstrap（95% CI，主指标）

| 对比 | 点估计 | 95% CI 下界 | 95% CI 上界 |
|---|---:|---:|---:|
| l2 − current | -0.150000 | -0.233333 | -0.070833 |
| l2 − dense | -0.229167 | -0.308333 | -0.150000 |
| current − dense | -0.079167 | -0.150000 | -0.008333 |

## 5. 预注册决策

- 结论：**B1_PREMATURE_PRUNING_PRIMARY**
- 依据：l2_minus_current CI upper -0.070833 <= 0.02 and l2 retention 0.742991 < 0.92 and current retention 0.911215 < 0.92
- L2−current 95% CI：[-0.233333, -0.070833]；L2 相对 dense 保留=0.742991；current 相对 dense 保留=0.911215

## 6. 输入 SHA256

| 输入 | SHA256 |
|---|---|
| preregistered_config | `1bad36e057166cbce4de2ab0f4f47890a96f0c174c3817994749092927d13451` |
| screening_manifest | `31bf313286c81fa50911ca3822118184ac03ca74df2cf2c260250cab2768b50c` |
| full_manifest | `22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91` |
| dense_full_jsonl | `3ff3b77ea4bddae96a8231c653bf08db1ce396a9c125fb414466417b1ea69f63` |
| current_full_jsonl | `b8976a5befdde7ae84cae925d14e37b2f5dd4e4d60b30eb4bdd879b18b87241d` |
| l2_screening_jsonl | `983300c1816a1d54a2f89354c14606cdc69b7fc4528fa299ce875c07d13849c6` |

## 7. 限制

- Screening100 is a fixed diagnostic subset, not the formal 2638-sample benchmark.
- Software Selector path only; no FPGA, latency or acceleration claim.
- Dense is the frozen custom dense path and is only a reference.
- The decision uses score(...)["cv_bench_accuracy"] exclusively.
- Forbidden primary metrics (pointing/grounding/ndcg/retained_mass) are never reported.

> CPU-only 只读汇总：未改原始 JSONL、正式 quality report 或 `configs/evaluation_protocol.json`；无 FPGA/时延/加速声明。
