# B1 corrected-L2 / Prompt 消融（screening100 诊断）

- 生成时间：`2026-10-04T09:28:43.144859+00:00`
- 范围：**diagnostic screening100; NOT a formal full2638 CV-Bench conclusion**
> ⚠️ Fixed stratified 100-sample screening: a corrected-L2 / Prompt ablation signal only, NOT a formal full2638 CV-Bench conclusion. It must not be quoted as the official CV-Bench accuracy of any approach.

- 正式主指标（唯一）：`score(...)["cv_bench_accuracy"]` = `((ADE20K + COCO)/2 + Omni3D)/2`；`unweighted_sample_accuracy` 仅为次要诊断，**不得写作 CV-Bench 主分**，也不参与决策。
- 禁用主指标：pointing, grounding, ndcg, retained_mass

## 1. 固定配置

- 样本：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/datasets/cv_bench_screening100.jsonl`；K=96；切点：B1 = after vision block 0；region_grid=3；local_cap=16
- dtype=bfloat16；attention=sdpa；decoding=greedy_argmax；bootstrap seed=20261004 / repeats=5000；quantization CI 下界阈值=-0.02；prompt CI 下界阈值=0.0

## 2. 输入合同校验

- screening 样本数：100
- 三臂样本数：{'corrected_l2_float': 100, 'corrected_l2_integer': 100, 'prompt_plus_corrected_l2_integer': 100}
- 100 个 sample_id 完全一致：✅
- 三臂 budget=96 / region_grid=3 / local_cap=16：✅
- 非 bypass 臂输出 K=96：✅
- 三臂输入 grid 一致（(tuple(image_grid_thw), full_patch_tokens, full_llm_visual_tokens)）：✅

## 3. 六臂正式主分

| 臂 | 角色 | 样本 | **CV-Bench 主分** | unweighted（次要） |
|---|---|---:|---:|---:|
| corrected_l2_float | primary | 100 | **0.733333** | 0.750000 |
| corrected_l2_integer | primary | 100 | **0.733333** | 0.750000 |
| prompt_plus_corrected_l2_integer | primary | 100 | **0.820833** | 0.830000 |
| dense | descriptive | 100 | **0.891667** | 0.900000 |
| old_defective_l2 | descriptive | 100 | **0.662500** | 0.660000 |
| old_current_integer | descriptive | 100 | **0.812500** | 0.820000 |

## 4. 配对分层 bootstrap（95% CI，主指标）

| 对比 | 决策相关 | 点估计 | 95% CI |
|---|---:|---:|---:|
| corrected_l2_integer − corrected_l2_float | 是 | +0.000000 | [+0.000000, +0.000000] |
| prompt_plus_corrected_l2_integer − corrected_l2_integer | 是 | +0.087500 | [+0.016667, +0.162500] |
| dense − corrected_l2_float | 否（descriptive） | +0.158333 | [+0.083333, +0.233333] |
| old_defective_l2 − corrected_l2_float | 否（descriptive） | -0.070833 | [-0.150000, +0.012500] |
| old_current_integer − corrected_l2_float | 否（descriptive） | +0.079167 | [+0.008333, +0.154167] |

## 5. 预注册决策

- 结论：**CORRECTED_PROMPT_SELECTOR_SCREENING_PASS**
- 依据：quantization gate passes (CI lower 0.000000 >= -0.02) and prompt_minus_integer_l2 CI lower 0.016667 > 0.0
- integer_minus_float：[+0.000000, +0.000000]，下界 +0.000000；量化门 ✅ 通过（阈值 >= -0.02）
- prompt_minus_integer_l2：[+0.016667, +0.162500]，下界 +0.016667；Prompt 增益 ✅ 确认（阈值 > 0.0）
- dense 与两个旧选择器为 descriptive reference，**不得覆盖**上述决策。

## 6. 输入 SHA256

| 输入 | SHA256 |
|---|---|
| screening_manifest | `31bf313286c81fa50911ca3822118184ac03ca74df2cf2c260250cab2768b50c` |
| full_manifest | `22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91` |
| dense_full_jsonl | `3ff3b77ea4bddae96a8231c653bf08db1ce396a9c125fb414466417b1ea69f63` |
| old_defective_l2_screening_jsonl | `983300c1816a1d54a2f89354c14606cdc69b7fc4528fa299ce875c07d13849c6` |
| old_current_integer_full_jsonl | `b8976a5befdde7ae84cae925d14e37b2f5dd4e4d60b30eb4bdd879b18b87241d` |
| corrected_l2_float_jsonl | `c9b25550ebf579844779067289d7a63140afefbad7ed8f3d44b5ca518b002ba3` |
| corrected_l2_integer_jsonl | `313beb877ceeaf0cbbdf025ff36e306c52ccf8a174cce84fd0532f32c01e6a7d` |
| prompt_plus_corrected_l2_integer_jsonl | `fbaf08e5ba69a5ff604ab56c28a2d1cf778f0d24e83561d017b836abf5667b70` |
| preregistered_config | `80f2db140cc1997ced73f939d40a0b5c172993a73a64ab75d10ea46d5a563319` |

## 7. 限制

- Screening100 is a fixed diagnostic subset, not the formal 2638-sample benchmark.
- Software Selector path only; no FPGA, latency or acceleration claim.
- Dense and the two old selectors are descriptive references only and never drive the decision.
- The decision uses score(...)["cv_bench_accuracy"] exclusively.
- Forbidden primary metrics (pointing/grounding/ndcg/retained_mass) are never reported.

> CPU-only 只读汇总：未改原始 JSONL、正式 quality report 或 `configs/evaluation_protocol.json`；无 FPGA/时延/加速声明。
