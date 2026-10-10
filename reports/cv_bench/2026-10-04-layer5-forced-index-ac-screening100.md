# Layer-5 forced-index（forced vs control，同预算 K_C）screening100 因果诊断

- 生成时间：`2026-10-04T13:35:39.192652+08:00`
- 范围：**fixed 100-sample CV-Bench screening causal diagnostic (forced vs control, same per-sample K_C); NOT a formal full2638 conclusion**
> ⚠️ Fixed 100-sample screening causal diagnostic: the forced vs control signal is diagnostic only and is NOT a formal full2638 CV-Bench conclusion. It must not be quoted as the official CV-Bench accuracy of any approach.

- 唯一主指标（唯一决策依据）：`score(...)["cv_bench_accuracy"]` = `((ADE20K + COCO)/2 + Omni3D)/2`；`unweighted_sample_accuracy`（`score(...)["accuracy"]`）仅为次要诊断，**不得写作 CV-Bench 主分**，也不参与决策。
- 禁用主指标：pointing, grounding, ndcg, retained_mass

## 1. 单变量合同与固定配置

- 唯一变量：`selected_group_indices`；实现：`evaluation/hardware_regular_layer5_forced_index_ablation.py (one unified forward, layer-5 cut)`；两臂逐样本同 K_C：✅
- control：control rule, exactly K_C real groups；forced：exact frozen shallow_r0250 historical indices, same K_C
- alignment 仅为诊断放宽（非 FPGA 候选）：relaxed=True；fpga_deployable_candidate=False；causal_diagnostic_only=True
- manifest=`datasets/cv_bench_screening100.jsonl`；reference=`evaluation/results/cv_bench_l2_budget64_pilot100_20260927/shallow_r0250.jsonl`（selector=shallow_r0250）；样本数=100
- cut_point=5；region_grid=3；score_mode=l2；local_candidate_cap=None
- dtype=bf16；attention=sdpa；decoding=greedy_argmax；do_sample=False；bootstrap seed=20261004 / repeats=5000 / stratified_by=task::source_dataset

## 2. 输入合同校验（fail-closed）

- 100 个 sample_id 唯一且四输入一致：✅
- reference 索引与 hash 复核：✅
- forced 选择精确等于 reference：✅
- control/forced 每样本 K_C 相同：✅
- control/forced/reference 每样本 G 相同：✅
- 两臂 BF16/SDPA/greedy：dtype=bf16、attn=sdpa、decoding=greedy_argmax
- 两臂 all-real / 无 context：✅
- 预注册 SHA pin 复核：✅（已校验 14 项；null 待生成 pin 2 项：['control_arm', 'forced_arm']）

## 3. 两臂官方主分

| 臂 | 样本 | **CV-Bench 主分**（`cv_bench_accuracy`） | unweighted_sample_accuracy（次要） | 2D | 3D | parse |
|---|---:|---:|---:|---:|---:|---:|
| control | 100 | **0.670833** | 0.700000 | 0.525000 | 0.816667 | 1.000000 |
| forced | 100 | **0.795833** | 0.810000 | 0.725000 | 0.866667 | 1.000000 |

- forced − control 点估计：**+0.125000**；95% CI：[+0.041667, +0.208333]

## 4. 唯一决策（forced − control，主指标）

- 阈值：`delta_ci95_lower > 0`；结果：**HISTORICAL_INDEX_SET_BETTER_CONFIRMED**
- 点估计=0.125000；CI 下界=0.041667；历史索引集合更好=True
- 边界：screening100 causal diagnostic only; this is NOT a full-dataset (2638-sample) conclusion

## 5. Correctness transitions

- 都对：66；仅 control 对：4；仅 forced 对：15；都错：15
- 输出文本变化：20（0.200000）；答案变化：20（0.200000）

## 6. K / G 分布（reference = 逐样本 K_C）

- G：64–3588（mean=724.140000）；K_C：16–897（mean=181.110000、total=18111）
- retention：mean=0.250253（0.250000–0.252137）；identity bypass=0
- G 桶：G<=96=2、97-256=27、257-768=46、G>768=25；K_C 桶：K<=48=20、49-96=21、97-192=34、K>192=25
- K_C 不同取值数：22；直方图：{'16': 2, '27': 2, '36': 3, '40': 2, '44': 1, '45': 7, '48': 3, '59': 7, '63': 2, '65': 6, '70': 1, '75': 3, '80': 1, '91': 1, '98': 11, '104': 1, '192': 22, '300': 1, '350': 20, '671': 1, '768': 1, '897': 2}

## 7. 索引集合 Jaccard / 嵌套（control vs forced）

- Jaccard：mean=0.603351（0.333333–0.849057）
- control ⊇ forced：0；forced ⊇ control：0；集合相等：0；forced == reference：100

## 8. 时间（仅描述，**不用于决策**）

- used_for_decision=False

## 9. 输入 SHA256

| 输入 | SHA256 |
|---|---|
| preregistered_config | `ef1ffeae6c2805b4ec1b3ff3f3551169b17c9ccf1b125ac956fb7f5603ce4457` |
| screening_manifest | `31bf313286c81fa50911ca3822118184ac03ca74df2cf2c260250cab2768b50c` |
| reference_jsonl | `ce019edd1ee35d11c83a3ff7c11bd9db46539cf56745aadcf0780f15abdbf7d0` |
| control_arm | `838cbb847c957412b73010b4c54999d1cad486a83c6e54592a1c42d35420d989` |
| forced_arm | `f78530ee2e781774dae66feb9e9287e678f1beca378c55ba186f7e6470b8b083` |

## 10. 限制

- Screening100 is a fixed diagnostic subset, not the formal 2638-sample benchmark.
- The relaxed K_C alignment is a diagnostic setting; it is NOT an FPGA-deployable candidate.
- Both arms run the same unified forward; the only variable is selected_group_indices.
- Only score(...)["cv_bench_accuracy"] drives the decision.
- Forbidden primary metrics (pointing/grounding/ndcg/retained_mass) are never reported.
- Timings are descriptive only and never enter the decision.
- Neither arm uses the B1-trained Wv projection.

> CPU-only 只读汇总：未改原始 JSONL、正式 quality report 或 `configs/evaluation_protocol.json`；无 FPGA/时延/加速声明。
