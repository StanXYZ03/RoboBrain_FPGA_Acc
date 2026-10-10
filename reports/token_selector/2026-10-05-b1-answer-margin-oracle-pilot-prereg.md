# B1 answer-margin oracle K96 pilot 预注册（未执行 GPU）

- 日期：2026-10-05
- 状态：**PRE-REGISTERED_NOT_EXECUTED**
- 设计：`2026-10-05-b1-answer-margin-oracle-pilot24`
- 范围：Label-aware oracle upper-bound pilot for the B1 answer-margin saliency; CPU-only pre-registration, GPU not executed.

## 1. 结论口径（必须保留）

- **ground-truth answer leakage = true**：oracle 用官方正确答案分叉 token 计算
  B1 `|grad×activation|` group saliency，只是可证伪的**上界**。
- **不可部署**：这不是 selector，不能上板、不能用于真实推理。
- **非论文精度**：24 样本只是泄漏 oracle 筛选；本轮 **未运行 GPU**、未加载模型权重、
  未训练、未生成任何 oracle 输出，因此不得使用 CI 或『精度提升已确认』措辞。
- `gpu_executed = false`、`model_was_run = false`、`gpu_runs = 0`。

## 2. 固定 24 样本

- manifest：`datasets/2026-10-05-b1-answer-margin-oracle-pilot24.jsonl`（SHA256 `9c8d15a65ef429eb1315ae1b73a80246de0ff2005373370fc184f0de1a2813c9`，24 条）。
- 三个固定 `source_dataset` 各 8 条：`ADE20K Validation Set`、`COCO 2017 Validation Set`、`Omni3D_Hypersim`。
- 样本 ID：`['245', '178', '327', '344', '356', '152', '351', '394', '1039', '886', '1064', '671', '1392', '1391', '1181', '1047', '1467', '2120', '1587', '2221', '1631', '1438', '2212', '1547']`。
- 各源数量：`{'ADE20K Validation Set': 8, 'COCO 2017 Validation Set': 8, 'Omni3D_Hypersim': 8}`。

## 3. 选样规则（不读正确性）

- 只用 `datasets/cv_bench_full2638.jsonl` 原始元数据、真实本地 tokenizer 标签分叉
  合同、`full_llm_visual_tokens > 96`，并按 `sha256(sample_id)` 升序取前 8。
- 排除：`sample_id=1`、`datasets/cv_bench_screening100.jsonl` 全部 ID、
  同图反事实 100 manifest 全部 ID。
- 排除计数：`199`。
- **不读取 dense/L2 正确性**；选样函数没有任何 correctness 输入。

## 4. 固定运行合同

| 项 | 值 |
| --- | --- |
| 模型 | `/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B`，BF16/SDPA |
| cutpoint | `B1_after_vision_block_0`（prune_after_layer=0） |
| K / region_grid / local_cap | 96 / 3 / 16 |
| 解码 | greedy，max_new_tokens=16 |
| 期望布局 | output=96，real=87，context=9 |

- saliency 定义：per merged group g: sum over its 4 patches j and feature dim d of abs(grad[g*4+j, d] * activation[g*4+j, d]), from the full (uncut) B1 visual sequence and the official correct-answer branch token
- score 注入：existing integer_scorer hook of forward_hardware_regular_b1_vision -> combined_scores of select_hardware_regular_groups (layout unchanged)

## 5. 汇总结论门（仅 24 样本泄漏 oracle 筛选）

- oracle − control net ≥ 3；
- W→R ≥ 4；R→W ≤ 1；
- oracle parse ≥ 23/24；
- 24/24 精确 K96 布局合同通过；
- 满足 → `GO`，否则 `STOP`；不使用置信区间。

## 6. 输入与脚本 SHA256 pin

- `full_manifest`: `datasets/cv_bench_full2638.jsonl` → `22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91`
- `screening100_manifest`: `datasets/cv_bench_screening100.jsonl` → `31bf313286c81fa50911ca3822118184ac03ca74df2cf2c260250cab2768b50c`
- `counterfactual100_manifest`: `datasets/2026-10-04-same-image-resolution-counterfactual-ade20k100.jsonl` → `c53c7b7dd733b11f55edc85ba65745d72951d31f456c7403ec21e91b0031d4c5`
- `selection_script`: `evaluation/select_b1_answer_margin_oracle_pilot24.py` → `067008c264903bee9b6c95ca87c0d8ac54dc42f38460025cb034a97a8242fd2e`
- `shared_helpers`: `evaluation/b1_answer_margin_oracle.py` → `58f2843ad329f723a15388f2a95422b4a55572b769ca62100df585a5bb28c752`
- `gpu_runner`: `evaluation/run_b1_answer_margin_oracle_pilot.py` → `972b504afa41027b1c07588e2cd5cd26f310e7d7ccc3e5d395c82dfb117fb6c2`
- `summarizer`: `evaluation/summarize_b1_answer_margin_oracle_pilot.py` → `8c8556d0e342348ddf2e9f17e3e8a1faf23d98079d613306425a5f5e0615981b`
- `selection_test`: `evaluation/tests/test_select_b1_answer_margin_oracle_pilot24.py` → `3b38b982ca85ebbbcaffdf0f5e864625eb80a9a19b83dfade0492e358ec5a2d8`
- `runner_test`: `evaluation/tests/test_run_b1_answer_margin_oracle_pilot.py` → `b4e32be6073047f4fe36fdc63a1ca2d036efc91d164150a8ca9e2a0298581bc9`
- `summarizer_test`: `evaluation/tests/test_summarize_b1_answer_margin_oracle_pilot.py` → `28d5b9773573331bc4ad66b030a15d8c4925ee483d3e152335afc1d6b7d0116c`

## 7. 汇总器输入

- dense：`evaluation/results/cv_bench_l2_0875_full2638_20260916/shallow_r1000.jsonl` → `3ff3b77ea4bddae96a8231c653bf08db1ce396a9c125fb414466417b1ea69f63`
- control：`artifacts/token_selector/2026-10-04-b1-corrected-l2-prompt-ablation-full2638/corrected_l2_integer_k96_full2638.jsonl` → `d0e302f6b91279374ddd507db4d300ca0c29e16f522db57f47c8de059f103a5f`
- oracle（待 GPU 生成）：`artifacts/token_selector/2026-10-05-b1-answer-margin-oracle-pilot24/oracle_answer_margin_b1_saliency_k96.jsonl`

## 8. 下一条命令（尚未执行）

```
python evaluation/run_b1_answer_margin_oracle_pilot.py --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --model-label RoboBrain2.5-4B-BF16 --manifest datasets/2026-10-05-b1-answer-margin-oracle-pilot24.jsonl --config configs/2026-10-05-b1-answer-margin-oracle-pilot24.preregistered.json --output artifacts/token_selector/2026-10-05-b1-answer-margin-oracle-pilot24/oracle_answer_margin_b1_saliency_k96.jsonl
```

禁止据此报告声称：deployable selector、paper accuracy、CV-Bench accuracy improvement confirmed、FPGA speedup or latency、using dense/L2 correctness to select samples。
