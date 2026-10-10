# edge576 CV-Bench 三臂预注册 r3（2026-10-07，可执行最终层）

- 状态：`PREREGISTERED_NOT_EXECUTED`（`executed=false`、`gpu_runs=0`、仅 CPU/只读、未加载模型权重）；三臂状态 `NOT_EXECUTED`
- 取代 r2：`configs/2026-10-07-edge576-cvbench-three-arm-development-r2.preregistered.json` SHA `8f265557bfb86998135b6985023e136d7f397eb8e06eb64551a97308cc958d09`；r2 `PREEXECUTION_SUPERSEDED`，从未执行且不得运行，字节全部保留
- 作废证据：`reports/token_selector/2026-10-07-edge576-cvbench-r2-preexecution-superseded.json`
- pin 总数：**54** = r2 既有 40 项 + 新增执行层 14 项

## 唯一作废 r2 的理由

**r2 未 pin 新增的 adapter / smoke 执行层。** 具体未 pin 的文件为：`evaluation/edge576_cvbench_runner_contract.py`、`evaluation/run_edge576_cvbench_three_arm.py` 及其聚焦测试、runner 适配报告 JSON/MD，以及新的独立单样本 smoke 入口 `evaluation/run_edge576_cvbench_three_arm_smoke.py` 及其聚焦测试。r2 **从未执行**（`gpu_runs=0`、`arm_runs=0`），作废不损失任何结果；这是 r2 被取代的**唯一**理由。

> 机器可读英文原文：r2 did not pin the newly added adapter/smoke execution layer: evaluation/edge576_cvbench_runner_contract.py, evaluation/run_edge576_cvbench_three_arm.py, their focused test, the runner adaptation report JSON/MD, and the new independent single-sample smoke entry evaluation/run_edge576_cvbench_three_arm_smoke.py and its focused test. r2 was never executed, so superseding it costs no results; this is the only reason r2 is superseded.

## r3 冻结不变（与 r2 逐字节一致）

- 三臂：`dense_edge576` / `corrected_l2_integer_b1_k96` / `v2_edge576_b1_k96`；
- 主指标：官方 `cv_bench_accuracy = ((ADE20K + COCO)/2 + Omni3D)/2`；
- 成功门：同输入 Dense@edge576 保留率 `>=0.90`；相对修正整数 L2 的 `task::source_dataset` 分层配对 bootstrap（seed 20261007、5000 次、95%）CI 下界 `>=0`；K96 布局（87 real + 9 context，`G<=96` identity bypass）；2638 个唯一完整输出（`parsed_answer` 可 `null`、解析失败计错并报告、无 100% 解析门）；无标签/答案泄漏；
- 三条 full2638 输出路径不变；edge576 运行域与处理参数不变。

## 新增独立单样本 smoke 执行层（本轮才 pin）

- 入口：`evaluation/run_edge576_cvbench_three_arm_smoke.py`（SHA `3c956d7ea9a64a52dc6affbec0c85ae032b44ce58197c8b166d89480dc2ec255`）；
- 测试：`evaluation/tests/test_run_edge576_cvbench_three_arm_smoke.py`（SHA `458a45101d8ca2a861b4a4d7f507e650402bee00e164e5991ea0ca1c508d1068`）；
- 唯一输出目录：`artifacts/token_selector/2026-10-07-edge576-smoke`，文件名 `<arm>_<sample_id>.json`；**绝不触碰** full2638 输出：
  - `artifacts/token_selector/2026-10-07-edge576-cvbench/dense_edge576_full2638.jsonl`
  - `artifacts/token_selector/2026-10-07-edge576-cvbench/corrected_l2_integer_k96_full2638.jsonl`
  - `artifacts/token_selector/2026-10-07-edge576-cvbench/v2_edge576_k96_full2638.jsonl`
- 每次只跑一个 arm + 一个 `sample_id`（建议 `1838`，G=576，grid=[1, 36, 64]）；
- 复用（不复制）`load_edge576_manifest` / `verify_dependency_pins` / `mechanism_identity` / `verify_model_identity` / `Edge576Processor` / `build_arm_state` / `run_multimodal_warmup` / `run_one_for_arm` / `normalize_record`；
- 必须执行：17 项依赖 pin + 机制身份、模型身份（含 `model.safetensors`）、真实 warmup（`max_new_tokens=1`、记录丢弃、`torch.cuda.synchronize`）、实际 `image_grid_thw` 断言；保存一个完整正式样本记录 + smoke 摘要；
- 禁止复制任何选择算法（`selection_algorithm_copied=false`）。

## 明确执行顺序（r3 冻结）

1. `model_identity` — 模型身份：在任何 CUDA/scorer/权重加载之前，校验 pinned 模型身份清单与每个声明文件（相对路径 + 字节数 + SHA256，含 `model.safetensors`）。
2. `three_arm_single_sample_smoke` — 三臂单样本 smoke：用 `evaluation/run_edge576_cvbench_three_arm_smoke.py` 依次对 `dense_edge576`、`corrected_l2_integer_b1_k96`、`v2_edge576_b1_k96` 各跑一个样本（建议 `sample_id=1838`）；每次调用只写自己的 smoke 文件，必须保存完整记录 + smoke 摘要。**三臂全部通过前不得进入任何 full2638 运行。**
3. `dense_edge576_full2638` — Dense@edge576 full2638（写 pinned 输出路径）。
4. `corrected_l2_integer_b1_k96_full2638` — 修正整数 L2@B1/K96 full2638（写 pinned 输出路径）。
5. `v2_edge576_b1_k96_full2638` — v2@B1/K96 full2638（写 pinned 输出路径）。
6. `official_scoring_and_paired_bootstrap` — 官方评分与分层配对 bootstrap：用 `evaluation/cv_bench_metric_v2.py` 对每条完成臂评分（显式 `parsed_answer`、解析失败计错），并做 `task::source_dataset` 配对 bootstrap（seed 20261007、5000 次、95%）之后，才允许外部验证或硬件主张。

**硬门**：第 2 步三臂单样本 smoke 必须全部通过，否则不得开始任何 full2638 运行。

## 新增/继承 pin

| pin | 文件 | SHA256 |
| --- | --- | --- |
| `edge576_runner_contract` | `evaluation/edge576_cvbench_runner_contract.py` | `fdea7df65556ef47c297a8082d9aa1238fcb645bdb1e5910d02f5453107590db` |
| `edge576_runner` | `evaluation/run_edge576_cvbench_three_arm.py` | `b1fb2ca1bc67af4355bfe6f0ce52725adf788f3339e77f5c6ea4188788e3eb85` |
| `edge576_runner_test` | `evaluation/tests/test_run_edge576_cvbench_three_arm.py` | `6d4f513568faccd005ded47e6b963d89b783194b84411b5c80d25fabf700fa55` |
| `runner_adaptation_report_json` | `reports/token_selector/2026-10-07-edge576-cvbench-three-arm-runner-adaptation.json` | `485947b8f44801d8c74488d8e446fed9ea6079149f8b9c2a98482df5fb783784` |
| `runner_adaptation_report_md` | `reports/token_selector/2026-10-07-edge576-cvbench-three-arm-runner-adaptation.md` | `6e496348d9aef43ae89aaebfa8d20e83bb82bb95490a0effc5622b5fef7655fe` |
| `smoke_entry` | `evaluation/run_edge576_cvbench_three_arm_smoke.py` | `3c956d7ea9a64a52dc6affbec0c85ae032b44ce58197c8b166d89480dc2ec255` |
| `smoke_test` | `evaluation/tests/test_run_edge576_cvbench_three_arm_smoke.py` | `458a45101d8ca2a861b4a4d7f507e650402bee00e164e5991ea0ca1c508d1068` |
| `prereg_generator_r3` | `scripts/preregister_edge576_cvbench_r3_20261007.py` | `03ed9f42862f90b664e7a9f3fdc8732690831669de48e00e0c2ace26d38f3eaa` |
| `prereg_test_r3` | `evaluation/tests/test_edge576_cvbench_prereg_r3.py` | `632f988a0f07f86ab697823e97d97f01e5791491ebb335541b8364440321713e` |
| `r2_prereg_config` | `configs/2026-10-07-edge576-cvbench-three-arm-development-r2.preregistered.json` | `8f265557bfb86998135b6985023e136d7f397eb8e06eb64551a97308cc958d09` |
| `r2_prereg_report_json` | `reports/token_selector/2026-10-07-edge576-cvbench-prereg-r2.json` | `ab2d22659850ea9757563dc84522ef5b50502cb28a788bf9a9e8bb285661d95b` |
| `r2_prereg_report_md` | `reports/token_selector/2026-10-07-edge576-cvbench-prereg-r2.md` | `a2254d205e80240d9cfe6f1b8a11126f452edbe98de441398091e95a3b5e2b27` |
| `r2_preexecution_superseded_json` | `reports/token_selector/2026-10-07-edge576-cvbench-r2-preexecution-superseded.json` | `ca853d6c5951be9792b87a6d2b734a581bc28551e6e0904ef29df5b8aed7bc37` |
| `r2_preexecution_superseded_md` | `reports/token_selector/2026-10-07-edge576-cvbench-r2-preexecution-superseded.md` | `f9593219aa383be7bdef2951edf70a75b0652e15f6e659e73af88c82e1b0e2a4` |

（其余 40 项 r2 冻结依赖现场重算后原样继承。）

## 验证

- `py_compile`：4 个新增文件；
- smoke 聚焦测试：**31 OK**；
- r3 预注册聚焦测试：**20 OK**；
- 全部 pin 现场重算 `pins_recomputed_on_disk=true`；
- `gpu_runs=0`、`weights_loaded=false`、`model_loaded=false`、未联网、未训练。

## 边界与风险

- 本轮**不启动模型、不联网、不训练**；smoke 入口与 r3 均未执行任何 GPU 运行；
- 模型身份核验会在加载前流式读取并 SHA256 整个 `model.safetensors`（约 9.65 GB），每次正式运行增加一次性 CPU/IO 开销；
- 三臂 smoke 与 full2638 的 GPU 路径尚未与真实模型联调；
- full2638 是反复观测的发展集，通过门也只允许一次未触碰外部基准确认，不构成发表/部署/加速主张；
- r2 及其预注册、报告、生成器、测试字节全部保留，不得重新生成或执行。
