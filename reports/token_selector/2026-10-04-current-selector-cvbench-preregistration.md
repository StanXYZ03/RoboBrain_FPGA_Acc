# 2026-10-04 Current Selector CV-Bench 预注册（PREFLIGHT）

- 状态：`PREREGISTERED_IMMUTABLE`（CPU-only、只读审计、未读取任何 proposed CV-Bench 输出）
- 权威提示：`reports/token_selector/prompts/2026-10-04-current-selector-cvbench-gate-harness-prompt.md`（SHA256 `8e1ae30e…76eea`）
- 配置：`configs/current_selector_cvbench_20261004.preregistered.json`（SHA256 `f49f86b0…09864`）
- 本报告：`reports/token_selector/2026-10-04-current-selector-cvbench-preregistration.json`（SHA256 `942c829f…426dd`），全部 pin 复核 `all_pins_verified=true`

## 1. 唯一部署 checkpoint/seed

- **seed = 20261003**
- checkpoint：`artifacts/token_selector/spatial_prompt_contrastive_trainval/B_SPATIAL_KL_CONTRASTIVE/20261003/best.npz`（SHA256 `1f1d379e…ca72ff`）
- 已验收 per-tensor INT8 Wv：`artifacts/token_selector/selector_wv_scale_regularization/20261003/Wv_int8_rowmajor.bin`（`126bd842…d89d3b`），scale：`.../Wv_per_tensor_scale_f32le.bin`（`93585a3b…28b13f`）

**选择依据（非 CV-Bench）**：2026-10-03 已验收的紧凑 HLS/IP/OOC/Zynq-BD 全链只由 seed 20261003 的 Wv bundle 构建——
`fpga_accel/integer_selector_pipeline_compact_hls/tools/generate_fixtures.py` 与
`artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/manifest.json`
都 pin `selector_wv_scale_regularization/20261003/Wv_int8_rowmajor.bin`，且 scorer/interface/pipeline 三个 HLS 报告只引用 seed 20261003。
seed 20261004/20261005 只出现在多 seed 数值审计统计与其预注册中，从未进入任何已验收硬件实现。
**禁止**三 seed 平均、禁止按 CV-Bench 回选；无歧义，故不 fail。

## 2. 当前 Selector 链（不得退回 L2、不得发明新算法）

1. `group_mean[G,1024]` → `selector_int8_numerical_audit.quantize_activation`（每图单一 INT8 scale）
2. `prompt_embeddings[T,2560]` + float `Wt` → `selector_int8_numerical_audit.primary_quantized_query`（float prompt-mean query → 每 query 单一 INT8 scale）
3. 已验收 per-tensor INT8 `Wv_q` → `selector_integer_scorer_audit.integer_chain`（精确 INT32 投影、带符号 INT64 dot、非负 INT64 norm2、精确 floor sqrt、精确有理 round-half-away、Q0.15 combine）
4. `score_q` → `combined = score_q / 2**15` → `hardware_regular_token_selector.select_hardware_regular_groups`（3x3 anchor + 有界 Top-K + 动态 context 保留，`combined_scores` 原样使用）

## 3. 冻结数据与评测

- manifest：`datasets/cv_bench_full2638.jsonl`（`22399a76…ce3d91`，2638 条，唯一 id 2638）
- 图片：`datasets/cv_bench_full2638_images`（2638 文件、404,722,392 B、聚合 SHA256 `abac17f6…110fc`）
- 官方 evaluator：`evaluation/build_cv_bench_pruning_report.py`（`58a6bc67…d8a61b`）的 `parse_choice/answer_letter/score/paired_bootstrap_delta`；公式 `((2D_ADE + 2D_COCO)/2 + 3D_Omni3D)/2`，禁止自定义公式
- dense baseline：native BF16 `reports/cv_bench/cv_bench_bf16_full2638.json`（`fa5bdcc9…405d4f`）；custom dense `evaluation/results/cv_bench_l2_0875_full2638_20260916/shallow_r1000.jsonl`（`3ff3b77e…9f63`）
- 模型：`/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B`（BF16；`model.safetensors` `e2f6e3c3…6eb2f`，config/generation/preprocessor/tokenizer/chat_template 全部 pin）
- operating points：K = 48/64/96（先验固定，绝不由 CV-Bench 回选）
- 解码：`do_sample=false`，greedy argmax，`max_new_tokens=16`（manifest 全为 16），模型加载不计入稳态

## 4. 计时边界（`configs/evaluation_protocol.json`，`997ccf1a…5d643`）

- model-E2E = vision_total + llm_prepare + prefill + decoder
- application-E2E = image_open_decode + processor + H2D + prompt_text_embedding + model_inference + postprocess
- 所有 GPU 段前后 `torch.cuda.synchronize`；`selector_software_seconds` 单列并排除在 `gpu_vision_seconds` 之外，绝不标注 FPGA/板卡加速

## 5. 正式质量与统计

- 仅当某 K 恰有 2638 个唯一完整样本时才输出正式分数；部分结果只能 `IN_PROGRESS`
- 主指标：官方组合分、2D/3D、Count/Relation/Depth/Distance、绝对差、相对 dense BF16 保持率、解析成功率、配对 bootstrap 95% CI（5000 次，seed 20261004，按 task×source_dataset 分层）
- **禁止**作为主指标或选择依据：Pointing、Grounding、selected-set NDCG、retained spatial mass

## 6. 门与产物

- identity ≤5（要求 generated token ids、文本、dense vision `image_embeds` 与官方 dense 逐项一致）→ 结构 smoke ≤5（只查 shape/K/placeholder/M-RoPE/有限值/INT 宽度，不聚合质量）→ K48/K64/K96 full2638
- runner：`evaluation/run_hf_current_selector_cvbench.py`（断点续跑 `resume_state` 对 mode/budget/seed/score_mode 不一致 fail-closed，绝不重复样本）
- 汇总器：`evaluation/summarize_current_selector_cvbench.py`
- host driver：`scripts/run_current_selector_cvbench.sh`（阶段顺序固定；100×3 latency 命令只打印不运行）
- 本轮未运行 GPU、未安装依赖、未读取任何 proposed CV-Bench 输出
