# edge576 CV-Bench 三臂 r4 汇总（CPU-only 只读）

- 生成时间（UTC）：`2026-10-08T04:47:06.214152+00:00`
- 样本数：每臂 `2638`（唯一 `sample_id`，三臂同集合，且与固定 edge576 manifest 一致）
- 主指标：官方 `cv_bench_accuracy = ((ADE20K+COCO)/2 + Omni3D)/2`，由 `evaluation/cv_bench_metric_v2.compute_cv_bench_metric(correct_key='correct', parsed_answer_key='parsed_answer', expected_count=2638)` 现场计算。
- 配对 bootstrap：`task::source_dataset` 分层、seed `20261007`、`5000` 次、95% 百分位区间；实现语义与冻结 `evaluation.build_cv_bench_pruning_report.paired_bootstrap_delta` 一致（聚焦测试逐位对照）。

## 口径与限制（必须先读）

- **软件 Selector 时间不是 FPGA 实测**：`selector_cpu_roundtrip_seconds` 是 Python 软件桥计时，不代表 FPGA/上板时延，也不是加速主张。
- **全量结果不是外部验证**：full2638 是被反复观测的发展集，本报告不是未触碰确认集，也不构成论文级精度主张。
- **诚实解析**：`parsed_answer` 显式字段为 `str` 或 `null`；解析失败计为错误答案并单独报告失败数/率，不设 100% 解析门。
- **未发明阈值**：本报告仅对预注册 r4 config 中冻结的门做参考判定，阈值原文读取，未自行新增或放宽任何阈值。
- 本汇总 CPU-only、只读；未修改三份原始 JSONL、r4 config、模型或 `CURRENT_STATUS.md`。

## 1. 输入与 SHA256 校验

| 臂 | mechanism_id | 原始 JSONL | SHA256 |
| --- | --- | --- | --- |
| dense_edge576 | dense_edge576 | `artifacts/token_selector/2026-10-07-edge576-cvbench/dense_edge576_full2638.jsonl` | `707868c61e683cefc24962bdf3c2b508bd2286d355d4427200da78533190e784` |
| corrected_l2_integer_b1_k96 | corrected_l2_integer | `artifacts/token_selector/2026-10-07-edge576-cvbench/corrected_l2_integer_k96_full2638.jsonl` | `e11dba93b324f5bcce18b84434b2cda2d022a09e66f2cee2db895dcc4bf88a8c` |
| v2_edge576_b1_k96 | conservative_l2_guarded_rerank_v2 | `artifacts/token_selector/2026-10-07-edge576-cvbench/v2_edge576_k96_full2638.jsonl` | `8b2a28854ec8741b53be61bfd461d1e4170be2f3d9bef863ec05222cc2f792f8` |

edge576 manifest `datasets/2026-10-07-cv-bench-edge576-full2638.jsonl` SHA256 `b5355ba1636011b4af101ace16c170b061ecf44a69e5c75fe2a64f4288c7dd1f`；r4 config `configs/2026-10-07-edge576-cvbench-three-arm-development-r4.preregistered.json` SHA256 `4f11d47981fd6c04c07608eba8b7e85e27e035bcb8d2d459a7ebc8a8c4b682c5`。

## 2. 官方质量分（官方 cv_bench_accuracy 及来源/任务分解）

| 臂 | 官方 cv_bench_accuracy | 相对同输入 Dense | 未加权样本精度 | 2D | 3D |
| --- | --- | --- | --- | --- | --- |
| dense_edge576 | 0.873958 | 1.000000 | 0.869219 | 0.799583 | 0.948333 |
| corrected_l2_integer_b1_k96 | 0.793787 | 0.908267 | 0.788476 | 0.714241 | 0.873333 |
| v2_edge576_b1_k96 | 0.807846 | 0.924353 | 0.802502 | 0.721525 | 0.894167 |

| 臂 | ADE20K | COCO | Omni3D | Count | Depth | Distance | Relation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dense_edge576 | 0.769352 | 0.829814 | 0.948333 | 0.684010 | 0.980000 | 0.916667 | 0.947692 |
| corrected_l2_integer_b1_k96 | 0.685624 | 0.742857 | 0.873333 | 0.549492 | 0.925000 | 0.821667 | 0.921538 |
| v2_edge576_b1_k96 | 0.684044 | 0.759006 | 0.894167 | 0.563452 | 0.953333 | 0.835000 | 0.923077 |

| 臂 | 解析成功 | 解析失败 | 解析成功率 | 解析失败率 |
| --- | --- | --- | --- | --- |
| dense_edge576 | 2602 | 36 | 0.986353 | 0.013647 |
| corrected_l2_integer_b1_k96 | 2596 | 42 | 0.984079 | 0.015921 |
| v2_edge576_b1_k96 | 2597 | 41 | 0.984458 | 0.015542 |

## 3. 配对差值（官方 cv_bench_accuracy）

| 对比 | 点估计 | 95% CI 下界 | 95% CI 上界 | 分层 |
| --- | --- | --- | --- | --- |
| L2 − Dense | -0.080171 | -0.093117 | -0.067591 | task::source_dataset |
| v2 − Dense | -0.066112 | -0.077846 | -0.054795 | task::source_dataset |
| v2 − L2 | 0.014059 | 0.004458 | 0.023341 | task::source_dataset |

## 4. 预注册门参考判定（阈值来自 r4 config，未发明）

- v2 官方分 / Dense 官方分 = `0.924353`，预注册阈值 `>= 0.9` → **PASS**。
- v2 − L2 点估计 = `0.014059`（阈值 `>= 0.0`）→ **PASS**；95% CI `[0.004458, 0.023341]`，下界阈值 `>= 0.0` → **PASS**。

## 5. 时延（秒；null = 该臂未记录该字段）

| 阶段 | 字段映射 Dense | Dense mean / p95 | L2 mean / p95 | v2 mean / p95 |
| --- | --- | --- | --- | --- |
| application_end_to_end_seconds | null | null | 0.150077 / 0.190141 | 0.185061 / 0.240654 |
| model_inference_seconds | elapsed_seconds | 0.157044 / 0.196484 | 0.140278 / 0.168488 | 0.174562 / 0.218795 |
| vision_seconds | vision_encoder_seconds | 0.025566 / 0.038780 | 0.035997 / 0.058722 | 0.067963 / 0.106744 |
| selector_cpu_roundtrip_seconds | null | null | 0.024436 / 0.045709 | 0.052795 / 0.086229 |
| llm_prefill_seconds | llm_prefill_seconds | 0.054821 / 0.085775 | 0.028359 / 0.029086 | 0.029679 / 0.032566 |
| decode_seconds | decoder_seconds | 0.069371 / 0.079988 | 0.068335 / 0.078186 | 0.069021 / 0.078613 |

- Dense 未记录 `application_end_to_end_seconds`，故 Dense 的应用端到端为 `null`；Dense 的 `model_inference_seconds` 诚实映射其真实字段 `elapsed_seconds`（vision 调用到 decode），`vision_seconds` 映射 `vision_encoder_seconds`。
- L2/v2 的应用端到端、模型推理、vision、selector CPU roundtrip 均来自各自真实字段；`vision_seconds` 取 `vision_total_seconds`。

## 6. Token G/K、平均保留率、G<=96 identity 数

| 臂 | G mean / median / p95 / min / max | K | 平均 LLM 视觉 token 保留率 | G<=96 数 | G>96 数 | identity_bypass 标志数 |
| --- | --- | --- | --- | --- | --- | --- |
| dense_edge576 | 357.066338 / 300.000000 / 576.000000 / 64.0 / 576.0 | null | 1.000000 | 143 | 2495 | 2638 |
| corrected_l2_integer_b1_k96 | 357.066338 / 300.000000 / 576.000000 / 64.0 / 576.0 | 96 | 0.365104 | 143 | 2495 | 143 |
| v2_edge576_b1_k96 | 357.066338 / 300.000000 / 576.000000 / 64.0 / 576.0 | 96 | 0.365104 | 143 | 2495 | 143 |

注：Dense 不裁剪（每样本 `output_group_count == G`，保留率 1.0），其 `identity_bypass` 字段恒为 true，属 producer 语义，不代表压缩臂的 G<=96 绕过；压缩臂的 G<=96 identity 数以上表为准。

## 7. 复现命令

```bash
python3 evaluation/summarize_edge576_cvbench_three_arm_r4.py
```

