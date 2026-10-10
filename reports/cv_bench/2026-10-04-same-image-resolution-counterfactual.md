# 同图双分辨率反事实（ADE20K100）汇总

- 生成时间：`2026-10-04T16:25:36.835526+00:00`
- 范围：**preregistered same-image dual-pixel-budget counterfactual on frozen ADE20K samples**
- 正式结论（full2638，不因本实验改变）：**PROMPT_GAIN_NOT_CONFIRMED_FULL2638**

- 唯一主估计量：`mean_i [ (Prompt_correct - Integer_correct)_high - (Prompt_correct - Integer_correct)_low ]`；官方 `score(...)["cv_bench_accuracy"]` 在单一 ADE20K 数据源内未定义，本报告**不计算**。
- 禁用主指标：pointing, grounding, ndcg, retained_mass

## 1. 固定配置

- 样本：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/datasets/2026-10-04-same-image-resolution-counterfactual-ade20k100-separated.jsonl`（100）；门控阈值 G=512；K=96；切点：B1_after_vision_block_0；region_grid=3；local_cap=16
- low pixel budget：{'min_pixels': 65536, 'max_pixels': 524288}；high pixel budget：{'min_pixels': 525312, 'max_pixels': 16777216}
- dtype=bfloat16；attention=sdpa；decoding=greedy_argmax；max_new_tokens=16；bootstrap seed=20261004 / repeats=5000

## 2. 输入合同校验

- 四文件样本数：{'corrected_l2_integer::low': 100, 'corrected_l2_integer::high': 100, 'prompt_plus_corrected_l2_integer::low': 100, 'prompt_plus_corrected_l2_integer::high': 100}
- 层（stratum）：G<=512::Count::integer_correct, G<=512::Count::integer_wrong, G<=512::Relation::integer_correct, G<=512::Relation::integer_wrong, G>512::Count::integer_correct, G>512::Count::integer_wrong, G>512::Relation::integer_correct, G>512::Relation::integer_wrong
- 每文件恰好 100 个唯一 sample_id：✅
- 四文件 ID 集合一致且等于 separated manifest：✅
- 权威 processor dry run 且 100/100 resolved：✅
- low 每样本 G <= 512：✅
- high 每样本 G > 512：✅
- G 与权威 dry-run 测量一致：✅
- 同图 low/high 成对：✅
- 同预算 integer/prompt 两臂 grid 完全一致：✅
- 同预算 integer/prompt 两臂 G 完全一致：✅
- low/high grid 不同：✅
- 无混臂 / 无重复或多余记录（断点续跑污染）：✅
- record_sha256 复核通过：✅
- checkpoint / Wv 哈希一致：✅

## 3. 各档指标（W→R / R→W / net、parse rate）

| 档 | G 范围 | Integer 准确率 | Prompt 准确率 | Integer parse | Prompt parse | W→R | R→W | net |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| low | 64..504 | 0.560000 | 0.620000 | 0.990000 | 0.990000 | 16 | 10 | +6 |
| high | 520..3588 | 0.530000 | 0.520000 | 0.980000 | 1.000000 | 11 | 12 | -1 |

## 4. 主估计量配对分层 bootstrap

- 估计量：`mean_i [ (Prompt_correct - Integer_correct)_high - (Prompt_correct - Integer_correct)_low ]`（unweighted per-sample paired difference (the official cv_bench_accuracy is undefined inside the single ADE20K source)）
- 点估计：**-0.070000**；95% CI：**[-0.200000, +0.060000]**；分层：original_stratum；seed=20261004；repeats=5000

| 层 | 样本 | 层内均值 |
|---|---:|---:|
| G<=512::Count::integer_correct | 13 | +0.076923 |
| G<=512::Count::integer_wrong | 13 | -0.153846 |
| G<=512::Relation::integer_correct | 14 | +0.000000 |
| G<=512::Relation::integer_wrong | 13 | -0.538462 |
| G>512::Count::integer_correct | 13 | -0.076923 |
| G>512::Count::integer_wrong | 14 | +0.285714 |
| G>512::Relation::integer_correct | 13 | -0.384615 |
| G>512::Relation::integer_wrong | 7 | +0.428571 |

## 5. 预注册决策

- 结论：**RESOLUTION_DEPENDENT_PROMPT_GAIN_NOT_CONFIRMED**
- 依据：95% stratified paired bootstrap CI lower bound -0.200000 <= 0.0 (mean_i [ (Prompt_correct - Integer_correct)_high - (Prompt_correct - Integer_correct)_low ])
- CI 下界 -0.200000 <= 0.0 → NOT_CONFIRMED
- 无论确认与否，正式 full2638 结论 `PROMPT_GAIN_NOT_CONFIRMED_FULL2638` **不变**。

## 6. 输入 SHA256

| 输入 | SHA256 |
|---|---|
| preregistered_config | `ae1ba23936896f9c7b836164e092ec941af17c05611bf6c0368b71e913857f83` |
| separated_manifest | `c53c7b7dd733b11f55edc85ba65745d72951d31f456c7403ec21e91b0031d4c5` |
| gate_separation_report | `2d2abe2db0079b3238b109c91ba020194a9e6bb4e04d9fcaee89c7161217021e` |
| checkpoint | `1f1d379e3229f310d11fee4ff620480e13fea9005abefc762c66f0d331ca72ff` |
| wv | `126bd8421dd128a318115c301bf9d1c7d336912150d19d2ec35203ed55d89d3b` |
| wv_scale | `93585a3bd2d584a612b40d827f8d15f43c87acdc764d7e217d99df473428b13f` |
| corrected_l2_integer::low | `83588ffa368c2f41939cf2cbaabc1145fafb4c8b890ba86686dee8ace71865bb` |
| corrected_l2_integer::high | `eb99fbd56387c23c1d0f67355b51085f4b38c3e5c18cacf8e176836b26330293` |
| prompt_plus_corrected_l2_integer::low | `7e806097e38d5d13cd3740535bd804b4fa9bbf90e88b759cfe0c16b8a54f695e` |
| prompt_plus_corrected_l2_integer::high | `25fa904763cb8d9d173d3edbef81ae715f206fc82512a723ca700fad330ecfeb` |

## 7. 限制

- Software Selector bridge only; no FPGA, latency or acceleration claim is made.
- The official score(...)["cv_bench_accuracy"] is undefined inside the single ADE20K source; only the preregistered unweighted paired difference is computed.
- Pointing / grounding / nDCG / retained-mass are never reported.
- Either outcome leaves PROMPT_GAIN_NOT_CONFIRMED_FULL2638 unchanged.
- The four GPU arm files were produced with a single A6000 BF16/SDPA forward; the counterfactual only varies the processor pixel budget of the same frozen image.

> CPU-only 只读汇总：未改原始 JSONL、预注册 config 或既有 gate 报告；**不作时延或 FPGA 加速声明**。
