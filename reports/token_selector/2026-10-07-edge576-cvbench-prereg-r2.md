# edge576 CV-Bench 三臂未来预注册 r2（2026-10-07）

- 状态：`PREREGISTERED_NOT_EXECUTED`（`executed=false`、`gpu_runs=0`、仅 CPU/只读、未加载模型权重）；第三臂状态 `NOT_EXECUTED`
- 取代已作废的 r1：`configs/2026-10-07-edge576-cvbench-three-arm-development.preregistered.json` SHA `8958cbd6f61589217b73e983e0f57257f0cf9bc951c5adbd8506df5d6156b3f3`（r1 `PREEXECUTION_SUPERSEDED`，从未执行且不得运行）
- edge576 manifest：`datasets/2026-10-07-cv-bench-edge576-full2638.jsonl` SHA `b5355ba1636011b4af101ace16c170b061ecf44a69e5c75fe2a64f4288c7dd1f`
- 源冻结 full manifest：SHA `22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91`
- 修正后审计器：`evaluation/audit_edge576_cv_bench_manifest.py` SHA `a999fe0a48b962104b9991df1c62329789a2164f530fe977767dd95ddbad352b`；重新审计报告：`reports/token_selector/2026-10-07-edge576-cv-bench-manifest-audit-r2.json` SHA `34f1385fbe71aa7a947798614b71fe4da9e64ef56449d3b41ac6c52e7886469e`
- 样本数：2638（唯一 2638）；全部 `G<=576`：True
- G：min/max = 64/576；分桶 <=96=143、97-256=781、257-512=706、513-576=1008

## r1 三项缺陷与 r2 修正

1. **A（第三臂未预注册）**：r1 第三臂 `mechanism_id=null`，不是已预注册候选。r2 将其明确冻结为**现有** `conservative_l2_guarded_rerank_v2`，`is_new_mechanism=false`，不发明新机制。
2. **B（记录完整性口径错误）**：r1 写「2638 全部可解析」与诚实解析口径冲突（Dense 也会有不可解析输出）。r2 只要求「2638 个唯一完整输出」，`parsed_answer` 可 `null`，解析失败按错误计并单独报告成功/失败数/率，**不设 100% 解析门**。
3. **C（审计器 `--model` 丢失）**：r1 审计器 CLI 接受 `--model` 却未传入 `audit()`，`verify_processor` 硬用 `DEFAULT_MODEL`。r2 已修复：`audit()` 显式接受 `model_path`，`main` 传入 `--model`，报告记录实际处理器路径。

## 三臂共享合同（已锁定）

- 同一 edge576 manifest 与同一 Prompt；BF16/SDPA；`cuda:0`；
- `do_sample=false`（greedy）、`max_new_tokens=16`；
- 处理参数 `min_pixels=65536`、`max_pixels=589824`、保持长宽比、`patch_size=16`、`merge_size=2`；
- 每臂必须断言 processor 的 `image_grid_thw` 与 manifest 记录逐样本一致，否则 fail closed；禁止使用原生分辨率预算。

## 三臂（与 r1 相同的输入与切点）

1. `dense_edge576`（同输入质量分母）；
2. `corrected_l2_integer_b1_k96`（B1 after vision block 0、K96、grid3、cap16、`G<=96` identity bypass）——冻结整数 L2 控制臂；
3. `v2_edge576_b1_k96`（同切点、同 K）——冻结现有 `conservative_l2_guarded_rerank_v2`，输出路径 `artifacts/token_selector/2026-10-07-edge576-cvbench/v2_edge576_k96_full2638.jsonl`，`status=NOT_EXECUTED`、`output_sha256=null`。

## 成功门（保留 r1 的有效门）

- 主指标仅官方 `cv_bench_accuracy`（`((ADE20K+COCO)/2+Omni3D)/2`）；
- 相对**同输入** Dense@edge576 保留率 `>= 0.90`；
- 相对同切点/同 K 修正整数 L2，按 `task::source_dataset` 分层配对 bootstrap（5000 次、seed 20261007、95%）CI 下界 `>=0`；
- 布局 K96/87 real+9 context（`G<=96` identity bypass）；
- 记录完整性：2638 个唯一完整输出（解析可失败、可 `null`，失败计错并报告）；
- 无标签/答案/task/source 泄漏。

## 口径与边界

- 这是**现有 v2 机制在新边缘运行域（G<=576）的发展性复测**，不是新机制，也不在 edge576 上调参；
- 2026-10-06 原生分辨率 v2 `DEVELOPMENT_STOP` 不追溯改写，原生结果只读；
- full2638 是反复观测的发展集，通过只允许一次新的未触碰外部基准确认；
- 本预注册不代表任何精度/部署/加速/FPGA 时延主张，也未启动任何模型推理；
- r1 从未执行、不得运行，仅作为未执行设计缺陷证据保留字节不变。

## 下一步 runner 适配入口

- Dense 适配入口：`evaluation/run_hf_shallow_block_pruning.py with --retentions 1.0 (the frozen producer of the dense reference shallow_r1000.jsonl), re-pointed to the edge576 manifest and processor budgets`
- 修正整数 L2 入口：`evaluation/run_hf_hardware_regular_b1_corrected_l2.py`
- v2 候选入口：`evaluation/run_hf_conservative_l2_guarded_rerank.py`
- 必须消费 edge576 manifest，`min_pixels=65536`、`max_pixels=589824`，并逐样本断言 `image_grid_thw` 等于 manifest；未适配前不运行任何模型。
