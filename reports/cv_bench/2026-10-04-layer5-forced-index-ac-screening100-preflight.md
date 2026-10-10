# Layer-5 forced-index（forced vs control，同预算 K_C）screening100 因果诊断：执行前预检

日期：2026-10-04  
状态：`PREREGISTERED / SOFTWARE_READY / GPU_RUN_PENDING`

> 本文件是**运行前**预检，只做 CPU-only、只读的工程与哈希记录。两个 arm 的原始 JSONL 尚未生成（预注册中 `sha256=null` / `TO_BE_GENERATED`），正式汇总报告也未生成。本预检**未运行 GPU、未加载模型、未联网、未训练、未调参、未安装依赖**。

## 1. 实验锁与唯一变量

- manifest：`datasets/cv_bench_screening100.jsonl`（固定 100 样本）。
- reference（只提供逐样本 `K_C` 与 forced 索引）：`evaluation/results/cv_bench_l2_budget64_pilot100_20260927/shallow_r0250.jsonl`（冻结历史 shallow-block r=0.25，`prune_after_layer=5`）。
- control 输出：`artifacts/token_selector/2026-10-04-layer5-forced-index-ac/control_same_budget_screening100.jsonl`（**待生成**）。
- forced 输出：`artifacts/token_selector/2026-10-04-layer5-forced-index-ac/forced_r25_indices_screening100.jsonl`（**待生成**）。
- 同一新 integration：`evaluation/hardware_regular_layer5_forced_index_ablation.py`（单一 layer-5 `cut=5` 统一 forward，control/forced 共用）。
- 同一新 runner：`evaluation/run_hf_layer5_forced_index_ablation.py`（每次只跑一个 arm）。
- 固定：逐样本同一 `K_C = len(reference.selected_group_indices)`、`region_grid=3`、BF16 / SDPA / greedy / `do_sample=false`、全 real、无 context pooling / 无 context slot reservation / 无 `local_candidate_cap`、DeepStack index 5 在 compact hidden 上调用、blocks 6..23 在压缩序列上运行、禁用 B1-trained Wv。
- **唯一变量：`selected_group_indices`**。control 用 control 规则重选恰好 `K_C` 个组；forced 精确使用历史冻结索引。
- `hardware_alignment_relaxed_for_diagnostic=true`、`causal_diagnostic_only=true`、`fpga_deployable_candidate=false`：放宽的 `K_C` 对齐只是**诊断**设置，**不是** FPGA 可部署候选。

## 2. 真实哈希（写入预注册并复核通过）

预注册文件：`configs/2026-10-04-layer5-forced-index-ac-screening100.preregistered.json`
当前 SHA256：`ef1ffeae6c2805b4ec1b3ff3f3551169b17c9ccf1b125ac956fb7f5603ce4457`

> 运行后、首次成功汇总前发生一次非分析性元数据契约修正：runner 记录 CLI 拼写 `bf16`，初版 summarizer 错误要求 `bfloat16`。原始两臂数据、数值精度、指标、bootstrap 和判定阈值均未改变；详见 `2026-10-04-layer5-forced-index-ac-screening100-amendment.md`。

| 输入 | SHA256 | 预注册 pin |
|---|---|---|
| `datasets/cv_bench_screening100.jsonl` | `31bf313286c81fa50911ca3822118184ac03ca74df2cf2c260250cab2768b50c` | 非 null，已复核 |
| `evaluation/results/cv_bench_l2_budget64_pilot100_20260927/shallow_r0250.jsonl` | `ce019edd1ee35d11c83a3ff7c11bd9db46539cf56745aadcf0780f15abdbf7d0` | 非 null，已复核 |
| `evaluation/build_cv_bench_pruning_report.py`（官方 score/pin 解析） | `58a6bc67399e32b6edac08acbc96babf7c4d5e83173b3cc8eb41a2db64d8a61b` | 非 null，已复核 |
| `evaluation/hardware_regular_layer5_forced_index_ablation.py`（integration） | `50e1adc8935488c320cdeab037cc4f5eb10612c9f3b31f3d9149daaff9615927` | 非 null，已复核 |
| `evaluation/run_hf_layer5_forced_index_ablation.py`（runner） | `0ee857726a7f0ea99bc25b0b15e37792137048aa8cf8ae7854af10e36e4bd151` | 非 null，已复核 |
| `evaluation/summarize_layer5_forced_index_ac_screening100.py`（summarizer） | `6356d3ac9571ded14f717a562a946af3f6aed2a437286bed7d2bb441b6528087` | 非 null，已复核 |
| `evaluation/metrics.py` | `9c5c12428289374f827be1018c7e95a59d61b5d5aedb5934970f5e2fa4c5efe8` | 非 null，已复核 |
| `evaluation/hardware_regular_token_selector.py` | `c55cfdc39e798e7566cd609a77ac388dca85df463ec20f536448284e2f85932b` | 非 null，已复核 |
| `evaluation/hardware_regular_layer5_unbounded_l2.py` | `de05d3877848228937d00b566afad26d45f450ef145ede9175092a97dda0b78a` | 非 null，已复核 |
| `evaluation/shallow_block_pruning.py` | `a753972f9a98d2bbfc5ae0fff40df8c05da8935147be52d5afcd0d006bcff84a` | 非 null，已复核 |
| `evaluation/run_hf_shallow_block_pruning.py` | `41a7e9c8a777dbf88dd39c6e713e201be75770b50ebc41a9936c42f9e09ed5b9` | 非 null，已复核 |
| `evaluation/run_hf_token_reduction.py` | `28e9870d17c9f1b63389e55d09f92118291ad15adb3c3c33c99686a973df5c8e` | 非 null，已复核 |
| `evaluation/token_reduction.py` | `6af9b507fb4a612b4013937150d6a7a59c8cf76d7e310c427b7fdb918f9a77b8` | 非 null，已复核 |
| `configs/evaluation_protocol.json` | `997ccf1a6707cd6b133d41f294ae14d28b8f03d781d6629d3d62bb16e015d643` | 非 null，已复核 |
| `.../control_same_budget_screening100.jsonl` | `null`（`TO_BE_GENERATED`） | 运行后回填 |
| `.../forced_r25_indices_screening100.jsonl` | `null`（`TO_BE_GENERATED`） | 运行后回填 |

预注册 `inputs_sha256` 共 16 项：14 项非 null 已逐一复核一致，2 项 arm 为 `null`/`TO_BE_GENERATED`（按设计在正式运行前跳过摘要校验，但文件必须存在）。

## 3. 样本 / K / G 分布（由冻结 reference 只读统计）

- 样本：manifest 100 唯一 id；reference 100 唯一 id；两者集合完全一致；`task` 与 manifest 一致；`prune_after_layer=5` 全部通过；`llm_visual_tokens == len(selected_group_indices)`；`selected_group_indices` 严格升序、唯一、SHA-256 可复核。
- 任务分布：Count 20、Relation 20、Depth 30、Distance 30。
- 来源分布：ADE20K 20、COCO 20、Omni3D 60（`source_dataset` 5 类）。
- G（`baseline_llm_visual_tokens`）：min/max/mean = **64 / 3588 / 724.14**；桶：`G<=96` 2、`97-256` 27、`257-768` 46、`G>768` 25。
- K_C（= reference `selected_group_indices` 长度）：min/max/mean = **16 / 897 / 181.11**；不同取值 **22** 个；桶：`K<=48` 20、`49-96` 21、`97-192` 34、`K>192` 25。
- K_C 直方图：`16:2, 27:2, 36:3, 40:2, 44:1, 45:7, 48:3, 59:7, 63:2, 65:6, 70:1, 75:3, 80:1, 91:1, 98:11, 104:1, 192:22, 300:1, 350:20, 671:1, 768:1, 897:2`。
- retention（K_C/G）：mean/min/max = **0.250253 / 0.25 / 0.252137**；identity bypass = **0**（全部样本 K_C < G）。
- 对齐诊断依据：100 个样本中 **71 个 K_C 不是 16 的整数倍**，故 `K_C` 对齐被显式放宽为诊断设置，禁止当作 FPGA 部署配置。

## 4. 汇总器与 fail-closed 设计

- 唯一主指标：`evaluation.build_cv_bench_pruning_report.score(...)["cv_bench_accuracy"]`；不得自创主指标；`score(...)["accuracy"]`（unweighted sample accuracy）仅次要诊断。
- 多选答案解析沿用官方 `answer_letter` / `parse_choice`（与官方 `score` 同模块），构造官方 score 记录。
- 配对 bootstrap：`paired_bootstrap_delta`，按 `task::source_dataset` 分层，`seed=20261004`、`repeats=5000`。
- 唯一决策：forced − control 点估计及 95% CI；**CI 下界 > 0 → `HISTORICAL_INDEX_SET_BETTER_CONFIRMED`，否则 `NOT_CONFIRMED`**。screening100 仅因果诊断，不宣称全数据集结论。
- 严格校验（任一项失败即 `InputContractError`，**不写任何报告**）：非 null SHA pin 复核；manifest/reference/control/forced 各 100 唯一且相同 `sample_id`；每条 arm 记录的 `arm`/flags/`manifest_sha256`/`reference_jsonl_sha256`/`K_C`/`G`/reference 索引及其 hash；**forced 选择精确等于 reference**；control/forced 每样本 `K_C` 相同；run metadata BF16/SDPA/greedy。
- 输出：`reports/cv_bench/2026-10-04-layer5-forced-index-ac-screening100.{json,csv,md}`；CSV 逐样本含 `source`/`source_dataset`/`task`/truth/两臂答案与正确性/K/索引 hash/输出是否变化。JSON/MD 另报两臂官方分、差值、CI、correctness transitions、K/G 分布、索引 Jaccard/嵌套；时间字段仅描述、`used_for_decision=false`。

## 5. 已通过的 CPU 测试（本预检阶段，无 GPU）

- 新增 summarizer 测试：`evaluation/tests/test_summarize_layer5_forced_index_ac_screening100.py` **29/29 OK**（覆盖官方 `score` 调用、CI 决策两分支、fail-closed、失败时不写报告、pin 校验、CSV 列、真实预注册 pin 与 integration 常量一致性）；测试使用临时合成数据，**未运行真实 summarizer**（结果文件尚不存在）。
- 既有 forced-index integration 回归：`evaluation/tests/test_hardware_regular_layer5_forced_index_ablation.py` **23/23 OK**。
- 既有 forced-index runner 回归：`evaluation/tests/test_run_hf_layer5_forced_index_ablation.py` **41/41 OK**。
- `py_compile` 通过。以上均 CPU-only、只读；**未运行 GPU、未加载模型**。

## 6. 正式运行前的 GPU 归属审计（强制）

正式运行前必须先做只读 GPU 归属审计，确认空闲且不干扰他人：

```bash
nvidia-smi
```

- 记录显存占用、利用率与所有计算进程的属主；只使用空闲且属于本任务的 GPU。
- **不得 kill / pause / 修改其他用户的进程**；若显存不足则等待或更换空闲设备，不得抢占。
- 本预检阶段**没有**执行任何 CUDA 模型加载。

## 7. 可直接执行的命令（运行前已预注册）

两条 arm 命令（同一 runner，仅 `--arm` 与 `--output` 不同）：

```bash
python evaluation/run_hf_layer5_forced_index_ablation.py --model <frozen BF16 RoboBrain2.5-4B> --model-label <label> --manifest datasets/cv_bench_screening100.jsonl --reference-jsonl evaluation/results/cv_bench_l2_budget64_pilot100_20260927/shallow_r0250.jsonl --output artifacts/token_selector/2026-10-04-layer5-forced-index-ac/control_same_budget_screening100.jsonl --arm control --region-grid 3 --dtype bf16 --device cuda:0
```

```bash
python evaluation/run_hf_layer5_forced_index_ablation.py --model <frozen BF16 RoboBrain2.5-4B> --model-label <label> --manifest datasets/cv_bench_screening100.jsonl --reference-jsonl evaluation/results/cv_bench_l2_budget64_pilot100_20260927/shallow_r0250.jsonl --output artifacts/token_selector/2026-10-04-layer5-forced-index-ac/forced_r25_indices_screening100.jsonl --arm forced --region-grid 3 --dtype bf16 --device cuda:0
```

汇总命令（两臂完成后，CPU-only）：

```bash
python evaluation/summarize_layer5_forced_index_ac_screening100.py --preregistered configs/2026-10-04-layer5-forced-index-ac-screening100.preregistered.json --manifest datasets/cv_bench_screening100.jsonl --reference-jsonl evaluation/results/cv_bench_l2_budget64_pilot100_20260927/shallow_r0250.jsonl --control-jsonl artifacts/token_selector/2026-10-04-layer5-forced-index-ac/control_same_budget_screening100.jsonl --forced-jsonl artifacts/token_selector/2026-10-04-layer5-forced-index-ac/forced_r25_indices_screening100.jsonl --output-json reports/cv_bench/2026-10-04-layer5-forced-index-ac-screening100.json --output-csv reports/cv_bench/2026-10-04-layer5-forced-index-ac-screening100.csv --output-md reports/cv_bench/2026-10-04-layer5-forced-index-ac-screening100.md
```

## 8. 边界与限制

- screening100 是固定诊断子集，不是正式 2638 样本全量结论；不得据此宣称全数据集结论。
- 放宽的 `K_C` 对齐是诊断设置，**不是** FPGA 可部署候选；本实验无 FPGA / 时延 / 加速声明。
- control 与 forced 共用同一实现且逐样本同 `K_C`，唯一变量是 `selected_group_indices`；时间字段仅描述、不参与决策。
- reference 仅提供 `K_C` 与冻结索引，不使用其输出或分数；forced 臂由新 runner 在本 workspace 重新前向产生。
- 未改原始 JSONL、正式 quality report 或 `configs/evaluation_protocol.json`；两臂文件生成前本预检不写任何正式汇总报告。
