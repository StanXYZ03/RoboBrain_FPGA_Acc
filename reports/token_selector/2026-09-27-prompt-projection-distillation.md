# 2026-09-27 B1 Prompt 投影蒸馏基础设施（评估契约加固，未运行抽取/GPU）

## 结论

- 状态：`PROMPT_PROJECTION_DISTILLATION_EVAL_CONTRACT_HARDENED / CPU_SYNTHETIC_TESTS_PASS / EXTRACTOR_NOT_RUN / GPU_NOT_USED`。
- 本轮只修改上一轮 7 个蒸馏文件并跑 CPU 合成测试、selector/B1 回归与 `py_compile`；未加载任何真实模型、未运行抽取器、未访问 GPU、未访问封存数据、未访问 ZCU102。
- 契约加固四点：图像 SHA256 分组 split（同图同 split、缺 hash 拒绝正式训练）、硬件真实 group 指标的 matched-real-budget 与 total-budget-coverage 双口径、split 评估的 L2 基线与确定性 shuffled-prompt 对照、以及 val 晋升判据（正式方法须优于 L2 且正确 prompt 须优于错配 prompt；测试集冻结后仅评估一次，不预设任意数值阈值）。

## 交付文件（本轮修改，未改任何其他既有源文件或模型）

| 文件 | SHA256（前 16） | 作用 |
| --- | --- | --- |
| `evaluation/prompt_projection_distillation.py` | `21c6bec09af608a1` | 冻结定义：教师/学生分数、指标、图像分组 split、对照、promotion gate、缓存 schema、损失 |
| `evaluation/extract_prompt_projection_features.py` | `aeaf2049bfab91e8` | offline 抽取器（可注销 hook + 图像 SHA256 + nvidia-smi 门禁） |
| `evaluation/train_prompt_projection.py` | `f34a7865f815299f` | 确定性训练器、受控 split 评估与 float32 导出 |
| `evaluation/tests/test_prompt_projection_distillation.py` | `a1bb19cae059739d` | CPU 合成测试（33 项） |
| `configs/prompt_projection_distillation.json` | `6c46c28216378f22` | 全部超参、比例、对照与晋升判据 |
| `reports/token_selector/2026-09-27-prompt-projection-distillation.md` | 本文件 | 中文报告 |
| `artifacts/token_selector/2026-09-27-prompt-projection-distillation-spec.json` | 机器可读 spec | 冻结契约工件 |

## 1. 教师分数契约

- 教师 = 完整 24 层视觉编码后的 merged image embedding `[G,2560]`（dense vision tower + merger），student 输入 = 同一图像的 B1 block0 后每组 4 patch 均值 `[G,1024]`。
- prompt = **仅** tokenize 原始用户 prompt 文本（`add_special_tokens=False`，无 chat template、无图像 token），再过输入 embedding 表得到 `[T,2560]`。
- `teacher_raw[g] = max_t cosine(merged_embeddings[g], prompt_embeddings[t])`，保存为 float32。
- `teacher_normalized = minmax(teacher_raw)`（`[0,1]`，全等则置 0）。
- `teacher_soft_target = softmax(teacher_normalized / temperature)`，逐样本和为 1。
- 学生：`student_raw[g] = max_t cosine(group_mean[g] @ Wv.T, prompt_embeddings[t] @ Wt.T)`；把 `group_mean` 广播到 4 个 patch 位置后与冻结 selector 的 `semantic_cosine_scores` 数值一致（测试已断言）。

## 2. 特征缓存 schema（每样本一个 npz，不保存全 dense hidden）

| key | dtype / shape | 说明 |
| --- | --- | --- |
| `group_mean` | float16 `[G,1024]` | B1 block0 每组 4 patch 均值 |
| `prompt_embeddings` | float16 `[T,2560]` | 原始 prompt token embedding |
| `teacher_scores` | float32 `[G]` | 原始最大 cosine |
| `teacher_scores_normalized` | float32 `[G]` | minmax 原分数 |
| `teacher_soft_target` | float32 `[G]` | temperature soft target |
| `meta_*` | 标量 | 新增 `source_image_sha256`；其余为 sample_id/task/grid/group_count/prompt_token_count/temperature/dense_vision_blocks/capture_layer/prompt_tokenization/add_special_tokens/source_manifest/source_image/config_sha256/sample_sha256 |

- 禁止键（加载时直接报错）：`hidden`、`dense_hidden`、`block0_hidden`、`deepstack`、`image_embeds`、`merged_embeddings` 等。
- `sample_sha256` 为指定 5 个数组的规范化内容哈希，`load_feature_cache` 会重新校验。
- JSONL 索引每行含 `sample_id/npz/group_count/prompt_token_count/grid/teacher min-max/sample_sha256/source_image_sha256/config_sha256/bytes`；抽取器读取已有索引的 `sample_id` 实现断点续跑。

## 3. 抽取器（本轮未运行）

- dense 前向时在 `visual.blocks[0]` 注册可注销 forward hook 捕获 block0 输出，并校验 `block0 行数 = 4*G`、merged 行数 = `G`、`grid_h*grid_w = G`，任一不一致即报错。
- 抽取时对**源图片文件字节**做流式 SHA256（`image_file_sha256`），写入 npz 的 `meta_source_image_sha256` 与索引记录的 `source_image_sha256`。
- 正式运行前强制 GPU 门禁 `gpu_preflight_audit`：只读调用 `nvidia-smi`/`--query-compute-apps`/`ps -o user= -p <pid>`；要求所有 GPU 计算进程属主为当前用户且空闲显存 ≥ `required_free_mib`（默认 20480 MiB）。不通过则写审计 JSON 并 **在加载模型前退出**，不 kill/暂停/修改任何进程。
- 本轮未执行抽取器，因此没有任何真实特征文件产生；门禁行为已用注入式假命令在 CPU 测试中覆盖（PASS/非属主/显存不足/nvidia-smi 失败四种）。

## 4. Split 契约（图像分组哈希，禁止按 sample_id 独立划分）

- 正式训练按 `source_image_sha256` 做 `sha256(salt|seed|source_image_sha256)` 哈希划分 `train/val/test`；**同一图像的所有 prompt 必然进入同一 split**，绝不跨 split。
- `split_sample_ids` 只接受带 `sample_id` 与 `source_image_sha256` 的记录；直接传 `sample_id` 字符串、或 hash 缺失/为空/非 64-hex 一律抛 `DistillationContractError`，即正式训练直接拒绝。
- 划分后同时断言 sample_id 两两不交，并由 `split_group_summary` 报告每 split 的 `sample_count` 与 `image_count`，且不同 split 的 image hash 集合必须两两不交。
- 合成测试用 4 张图 × 3 个 prompt（12 样本）覆盖“同图多 sample 绝不泄漏”：每个 image hash 只归属一个 split，全部 image 数守恒。

## 5. 训练器契约

- 确定性：`seed` 同时固定 numpy/torch 与参数初始化（`torch.randn(generator)`）；minibatch 顺序由 `np.random.default_rng(seed)` 固定。
- 主损失：`KL(teacher_soft || softmax(student/T))`（默认乘 `T^2`）；`score_mse_weight > 0` 时叠加逐样本 minmax 分数的 MSE（默认 0）。
- 选择纪律：best epoch 仅按 **val** listwise KL 选；**test 只在 best 冻结后评估一次**，报告写 `test_used_for_selection=false`、`test_evaluated_once_after_freeze=true`。
- 导出：float32 `Wt`/`Wv` + `weights_sha256 = canonical_projection_sha256(Wt,Wv)`，与 `evaluation/run_hf_hardware_regular_b1.py::canonical_projection_sha256` 逐位一致；npz 元数据记录 `trained=True`、`training_complete=True`、best_epoch、`split_group_by=source_image_sha256`。

## 6. 指标（全部自行实现，不用 SciPy）

- `pearson`；`spearman` 用自实现 `stable_rank`（stable sort + 并列取平均秩）。
- `ndcg@K`：以 `teacher_scores_normalized` 为 graded relevance，按 `student_raw` 排序。
- 对 `K=48/64/96`：`topk_recall = |student_topK ∩ teacher_topK| / K`、`jaccard`、`saliency_mass_retained`（`teacher_soft_target` 在 student topK 上的质量和）。
- 硬件约束指标：对每个预算调用冻结 `select_hardware_regular_groups`，只在 `source_type==REAL` 的真实 group 集合上报告，并拆成两个明确口径：
  - `matched_real_budget`：取 `R = real_group_count`，与 teacher top-R 比 recall/jaccard/mass（公平的真实 group 对真实预算比较）。
  - `total_budget_coverage`：与 teacher top-K 比 recall/jaccard/mass；context group 占用了 K 的一部分，因此是覆盖上界（`upper_bound_only=true`），**明确不是满分**。
  - 不再输出含糊的顶层 `teacher_topk_recall`。
- 因 selector 的 real 决策只依赖每组均值，缓存用 `mean` 广播到 `[G,4,1024]` 在数值上与原实现等价。

## 7. Split 评估与对照

- 每个 split（train/val/test）同时报告 `student_correct`、`l2_baseline`、`student_shuffled_prompt` 及 `correct_minus_shuffled`。
- `l2_baseline`：按 `group_mean` 的每组 L2 范数排序，用同一 rank/K 指标评估（数值等于 selector 的 `group_l2_saliency` 在广播后的结果）。
- `student_shuffled_prompt`：**同一 split 内**按 `sample_id` 升序循环错配 prompt embedding（样本 i 使用样本 i+1 mod n 的 prompt），用同一 `Wt`/`Wv` 算分；对 n≥2 是无自配对的错排，结果只依赖排序后的 sample_id 与 prompt。
- `correct_minus_shuffled`：correct 减 shuffled 在 `pearson`、`spearman`、`ndcg_full` 及每个预算 K 的 `ndcg/topk_recall/jaccard/saliency_mass_retained` 上的配对差值。
- 样本数 < 2 时 shuffled 标记 `available=false` 且 reason 为“split has fewer than two samples”，差值同样 `available=false`；**不伪造**。

## 8. 晋升判据（写入 config 与报告，不预设任意数值阈值）

- 正式方法必须在 **val** 上优于 L2 基线；正确 prompt 必须优于 shuffled 错配 prompt。
- 判据只取配对差值的符号（严格 > 0），主指标为 `spearman` 与 `ndcg_full`；`no_arbitrary_threshold` 明确不预设数值 margin。
- 测试集只在 best epoch 冻结后评估一次，绝不影响选 epoch、学习率或任何超参。
- 报告写 `promotion_gate`（status/student_beats_l2/correct_beats_shuffled/各指标差值）、`val_gate`、`primary_promotion_metrics`；val 少于 2 样本时 gate 标记 `UNAVAILABLE`。

## 9. 验证

- `py_compile`：4 个 py 文件通过。
- 新测试：`python -m unittest evaluation.tests.test_prompt_projection_distillation` → **33/33 OK**（CPU，约 4 s）。
- 回归：既有 `test_hardware_regular_token_selector` + `test_hardware_regular_b1_integration` → **42/42 OK**，证明未破坏现有 selector/runner 契约。
- 覆盖点：图像分组 split 稳定性/同图不泄漏、缺/非法 hash 拒绝、R 与 K 双口径指标、shuffled 确定性与少样本 unavailable、正确 prompt 合成数据严格优于错配 prompt、promotion gate 的 EVALUATED/UNAVAILABLE 两态。
- 合成短训（24 样本、12 epoch）训练 loss 下降，导出 npz 可被现有 `load_projection` 加载。

## 10. 已知边界

- 未运行真实抽取，`group_mean`/`prompt_embeddings` 的真实分布、`G` 规模、真实可学性均未验证；本轮所有数字都是 CPU 合成。
- 无真实 Wt/Wv 训练结果，因此不能宣称任何真实指标、精度或加速；合成短训未证明 val 一定优于 L2/错配 prompt，只证明报告与判据可正确给出布尔结论。
- 教师 soft target 的具体数值取决于 `temperature`；temperature 属于配置，本轮未在真实数据上选择。
- GPU 门禁的 `nvidia-smi` 解析针对标准 CSV 输出；非标准驱动/容器输出会进入 `BLOCKED_NVIDIA_SMI_UNPARSABLE`。
- `select_hardware_regular_groups` 在预算不足以覆盖非空区域时会抛 `InsufficientBudgetError`；真实 `G` 较小（`G<=K`）时走 identity bypass，此时 real 集合为全部 group。
- 图像 SHA256 分组只能保证“同文件”不泄漏；同一图像的不同编码副本会得到不同 hash，需在构建 manifest 时人工保证。

## 11. 精确下一步命令（需在通过 GPU 门禁的会话执行，本轮不执行）

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/zhaoxiangyu/conda-envs/robobrain25/bin/python evaluation/extract_prompt_projection_features.py --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --manifest datasets/cv_bench_screening100_budget64_diagnostic.jsonl --config configs/prompt_projection_distillation.json --out-dir artifacts/token_selector/prompt_projection_features --limit-per-task 4
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/zhaoxiangyu/conda-envs/robobrain25/bin/python evaluation/train_prompt_projection.py --config configs/prompt_projection_distillation.json --index artifacts/token_selector/prompt_projection_features/index.jsonl --out artifacts/token_selector/prompt_projection_WtWv.npz --report reports/token_selector/2026-09-27-prompt-projection-training.json --device cuda:0
```

随后如需推理，必须用 `evaluation/run_hf_hardware_regular_b1.py --score-mode prompt --projection-npz <上一步导出的 npz>`，禁止使用任何未训练权重。

## 12. 合规

- 未修改任何既有源文件、模型或数据（本轮只改上一轮 7 个蒸馏文件）；未删除文件；未 kill/暂停/修改任何进程。
- 未访问封存（blind/validation）数据，未访问 ZCU102。
- 未运行抽取器、未加载 CUDA 模型、未声称任何 GPU 结果。
