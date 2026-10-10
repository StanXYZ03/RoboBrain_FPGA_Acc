# edge576 CV-Bench 三臂预注册 r4（2026-10-07，token 契约修复层）

- 状态：`PREREGISTERED_NOT_EXECUTED`（`executed=false`、`gpu_runs=0`、仅 CPU/只读、未加载模型权重）；三臂状态 `NOT_EXECUTED`
- 取代 r3：`configs/2026-10-07-edge576-cvbench-three-arm-development-r3.preregistered.json` SHA `e7ff56c98a39f764deb6cd6b099211f5da45025ce6958b44e8f72c641ae83eed`；r3 `SUPERSEDED_AFTER_FIRST_GPU_SMOKE_FAILURE`（首次 GPU smoke 于归一化阶段 fail-closed，写出记录 0、未触碰 full2638），字节全部保留且不得再运行
- 失败证据：`reports/token_selector/2026-10-07-edge576-cvbench-r3-first-gpu-attempt-failure.json`；作废记录：`reports/token_selector/2026-10-07-edge576-cvbench-r3-superseded.json`
- pin 总数：**63** = r3 既有 54 项 + 新增 9 项

## 唯一作废 r3 的理由

**r3 硬性要求 `output_token_ids`，而冻结 dense 生产者从不返回 token id。** `evaluation/run_hf_shallow_block_pruning.py::run_one` 只返回 `output_text`、真实整数计数 `output_tokens` 和 `selected_group_indices`；只有冻结 L2/v2 生产者返回 `output_token_ids`。r3 的第一次真实执行（`dense_edge576`、`sample_id=1838`、`cuda:0`）在模型加载、一次 warmup、一次样本推理之后，于写文件前 fail-closed：`frozen dense_edge576 record is missing 'output_token_ids' (fail closed)`。r3 **未写任何记录、未触碰任何 full2638 输出**，作废不损失任何结果；这是 r3 被取代的**唯一**理由。

> 机器可读英文原文：r3 hard-required 'output_token_ids' for every arm, but the frozen dense producer evaluation/run_hf_shallow_block_pruning.py::run_one never emits token ids (only the real integer count 'output_tokens'). The first real r3 GPU smoke attempt (dense_edge576, sample_id=1838) completed model load, one warmup and one sample inference and then fail-closed inside normalize_record before writing any file. r3 wrote no record and touched no full2638 output, so superseding it costs no results; this is the only reason r3 is superseded.

## r4 冻结不变（与 r3 逐字节一致）

- 三臂：`dense_edge576` / `corrected_l2_integer_b1_k96` / `v2_edge576_b1_k96`；
- 主指标：官方 `cv_bench_accuracy = ((ADE20K + COCO)/2 + Omni3D)/2`；
- 成功门、`task::source_dataset` 分层配对 bootstrap（seed 20261007、5000 次、95%）、K96 布局（87 real + 9 context，`G<=96` identity bypass）、2638 唯一完整输出（`parsed_answer` 可 `null`、解析失败计错并报告、无 100% 解析门）、无标签/答案泄漏；
- 三条 full2638 输出路径不变；edge576 运行域与处理参数不变。

## r4 修的 token 契约（唯一改动）

- `output_token_ids` **可选**，只有冻结生产者真实返回时才记录，**绝不伪造**；
- 必备可审计字段 `output_token_count` 由真实字段规范映射：有 `output_token_ids` 取其真实长度、有 `output_tokens` 取其真实整数值、两者都在必须一致、两者都无则 fail-closed；
- 必备 `output_token_ids_available`（bool）标注 token id 是否真实存在；
- 核心字段 `output_text`、`selected_group_indices` 缺失仍 fail-closed；
- 主指标不消费 token id（只用 `output_text` + 官方答案诚实解析）。

## 独立单样本 smoke 执行层

- 入口：`evaluation/run_edge576_cvbench_three_arm_smoke.py`（SHA `52003e39f2f1e2ae4c858bf80a820f3c10e9efd6028f3a563f94abc0191e3044`）；
- 测试：`evaluation/tests/test_run_edge576_cvbench_three_arm_smoke.py`（SHA `b0a7ed6e4825f9382d4393287062020c09b532e9bbc2e0105eb10424841ee1a5`）；
- 唯一输出目录：`artifacts/token_selector/2026-10-07-edge576-smoke`，文件名 `<arm>_<sample_id>.json`；**绝不触碰** full2638 输出：
  - `artifacts/token_selector/2026-10-07-edge576-cvbench/dense_edge576_full2638.jsonl`
  - `artifacts/token_selector/2026-10-07-edge576-cvbench/corrected_l2_integer_k96_full2638.jsonl`
  - `artifacts/token_selector/2026-10-07-edge576-cvbench/v2_edge576_k96_full2638.jsonl`
- 每次只跑一个 arm + 一个 `sample_id`（`1838`，G=576，grid=[1, 36, 64]）；
- 复用（不复制）`load_edge576_manifest` / `verify_dependency_pins` / `mechanism_identity` / `verify_model_identity` / `Edge576Processor` / `build_arm_state` / `run_multimodal_warmup` / `run_one_for_arm` / `normalize_record`；
- 必须执行：17 项依赖 pin + 机制身份、模型身份（含 `model.safetensors`）、真实 warmup（`max_new_tokens=1`、记录丢弃、`torch.cuda.synchronize`）、实际 `image_grid_thw` 断言；保存一个完整正式样本记录 + smoke 摘要；
- 禁止复制任何选择算法（`selection_algorithm_copied=false`）。

## 明确执行顺序（r4 冻结）

1. `model_identity` — 模型身份：在任何 CUDA/scorer/权重加载之前，校验 pinned 模型身份清单与每个声明文件（相对路径 + 字节数 + SHA256，含 `model.safetensors`）。
2. `three_arm_single_sample_smoke` — 三臂单样本 smoke：用 `evaluation/run_edge576_cvbench_three_arm_smoke.py` 依次对 `dense_edge576`、`corrected_l2_integer_b1_k96`、`v2_edge576_b1_k96` 各跑一个样本（`sample_id=1838`）；每次调用只写自己的 smoke 文件，必须保存完整记录 + smoke 摘要。Dense 无真实 token id，记录必须带 `output_token_count` 且 `output_token_ids_available=false`。**三臂全部通过前不得进入任何 full2638 运行。**
3. `dense_edge576_full2638` — Dense@edge576 full2638（写 pinned 输出路径）。
4. `corrected_l2_integer_b1_k96_full2638` — 修正整数 L2@B1/K96 full2638（写 pinned 输出路径）。
5. `v2_edge576_b1_k96_full2638` — v2@B1/K96 full2638（写 pinned 输出路径）。
6. `official_scoring_and_paired_bootstrap` — 官方评分与分层配对 bootstrap：用 `evaluation/cv_bench_metric_v2.py` 对每条完成臂评分（显式 `parsed_answer`、解析失败计错），并做 `task::source_dataset` 配对 bootstrap（seed 20261007、5000 次、95%）之后，才允许外部验证或硬件主张。

**硬门**：第 2 步三臂单样本 smoke 必须全部通过，否则不得开始任何 full2638 运行。

## 新增/继承 pin

| pin | 文件 | SHA256 |
| --- | --- | --- |
| `edge576_runner_contract` | `evaluation/edge576_cvbench_runner_contract.py` | `36f7c2e44a974350ddb3863871788ea14de3b901eca2ee6817e1e5cde8a83813` |
| `edge576_runner` | `evaluation/run_edge576_cvbench_three_arm.py` | `5e6b408746c9f5410b3a893ff3269ecc596596b9a346ef1188b5a75dbb60a83a` |
| `edge576_runner_test` | `evaluation/tests/test_run_edge576_cvbench_three_arm.py` | `6f8d8c2f0923bc1486aa8033a9f7ef25894945e6a7cc92df2be0ce47a84e3048` |
| `smoke_entry` | `evaluation/run_edge576_cvbench_three_arm_smoke.py` | `52003e39f2f1e2ae4c858bf80a820f3c10e9efd6028f3a563f94abc0191e3044` |
| `smoke_test` | `evaluation/tests/test_run_edge576_cvbench_three_arm_smoke.py` | `b0a7ed6e4825f9382d4393287062020c09b532e9bbc2e0105eb10424841ee1a5` |
| `r3_first_gpu_attempt_failure_json` | `reports/token_selector/2026-10-07-edge576-cvbench-r3-first-gpu-attempt-failure.json` | `605127121cf06c1b5292e406a5f2d3bf180f45c7458f4551f6e78558891bdcd8` |
| `r3_first_gpu_attempt_failure_md` | `reports/token_selector/2026-10-07-edge576-cvbench-r3-first-gpu-attempt-failure.md` | `98bd60504f048e1b8411e1c6e968860210c5142055dd3c33fafa746c38e99bae` |
| `r3_superseded_json` | `reports/token_selector/2026-10-07-edge576-cvbench-r3-superseded.json` | `9ff117f1d38ec0eea910cf029ad31ef3d6d91b6b6043b1a0f09f4eef941b54e0` |
| `r3_superseded_md` | `reports/token_selector/2026-10-07-edge576-cvbench-r3-superseded.md` | `2a7ba9333c46c6b8d4135f23bd266cb4a4449a2a2777149a795ccb19b76d602b` |
| `r3_prereg_config` | `configs/2026-10-07-edge576-cvbench-three-arm-development-r3.preregistered.json` | `e7ff56c98a39f764deb6cd6b099211f5da45025ce6958b44e8f72c641ae83eed` |
| `r3_prereg_report_json` | `reports/token_selector/2026-10-07-edge576-cvbench-prereg-r3.json` | `42a4aabddd8895a2b502befff2052056ad757af80608a106f8f5ad76e3158c49` |
| `r3_prereg_report_md` | `reports/token_selector/2026-10-07-edge576-cvbench-prereg-r3.md` | `ce3561e2635185375275e1cea22e1e6bd97ee233f5710ca206632f2bc79d9051` |
| `prereg_generator_r4` | `scripts/preregister_edge576_cvbench_r4_20261007.py` | `d7572e20287c535c339ca25cf178f458252c569d1e3147c3face186e9c58dff6` |
| `prereg_test_r4` | `evaluation/tests/test_edge576_cvbench_prereg_r4.py` | `3536908509a33c14c4f1417719539b9fabc31140fb3a332a60527a37c0d5ee94` |

（其余 r3 冻结依赖现场重算后原样继承；r3 未重新生成，字节保留。）

## 验证

- `py_compile`：4 个新增/修改文件；
- runner 聚焦测试：**66 OK**；
- smoke 聚焦测试：**34 OK**；
- r4 预注册聚焦测试：**21 OK**；
- 全部 pin 现场重算 `pins_recomputed_on_disk=true`；
- `gpu_runs=0`、`weights_loaded=false`、`model_loaded=false`、未联网、未训练。

## 边界与风险

- 本轮**不启动模型、不联网、不训练**；r4 与修复后的 smoke 均未执行任何 GPU 运行；
- r3 的失败尝试消耗了 1 次 GPU smoke，未产出任何记录，不构成精度/加速/部署证据；
- 模型身份核验会在加载前流式读取并 SHA256 整个 `model.safetensors`（约 9.65 GB），每次正式运行增加一次性 CPU/IO 开销；
- 三臂 smoke 与 full2638 的 GPU 路径尚未在修复后与真实模型联调；
- full2638 是反复观测的发展集，通过门也只允许一次未触碰外部基准确认，不构成发表/部署/加速主张；
- r3 及其预注册、报告、生成器、测试、失败/作废证据字节全部保留，不得重新生成。
