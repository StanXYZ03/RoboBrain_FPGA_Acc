# 2026-09-28 Prompt 投影特征抽取器同图复用（CPU 审计 + 可注入执行层）

- 状态：`PROMPT_FEATURE_IMAGE_REUSE_IMPLEMENTED / CPU_TESTS_PASS / EXTRACTOR_NOT_RUN / GPU_NOT_USED`
- 范围：只审计并重构 `evaluation/extract_prompt_projection_features.py` 的抽取编排；**未使用 GPU、未加载真实模型、未运行真实抽取/训练/推理、未改教师/学生定义、未改 NPZ schema、未改 split、未改 manifest pin**。
- 冻结来源：`datasets/prompt_projection_clean_inductive_v1.jsonl`（472 记录 / 236 图，sha256 前缀 `acab1a32…9213`；236 个唯一图片文件，共 35,054,288 B；每图 2 条 prompt，grounding 236 + pointing 236）。

## 1. 审计结论：哪些输出是图像相关、哪些是 prompt 相关

对真实执行路径逐项核对（`dense_vision_with_block0_capture`、`group_mean_from_block0`、`prompt_embeddings_from_text`、`teacher_target`）：

| 输出 | 依赖 | 是否可跨 prompt 复用 |
| --- | --- | --- |
| processor image `pixel_values` / `image_grid_thw` | 仅图像 | 是 |
| block0 capture、`group_mean[G,1024]` | 仅图像（视觉前向 + hook） | 是（**关键**：这是学生输入） |
| 完整 24 层 dense vision + merger `merged_embeddings[G,2560]` | 仅图像 | 是（**关键**：教师特征的图像侧） |
| merge grid `(grid_h, grid_w)` | 仅图像 | 是 |
| `prompt_embeddings[T,2560]` | 仅 prompt 文本（`add_special_tokens=False`） | **否，逐记录独立** |
| `teacher_scores` / `_normalized` / `_soft_target` | 图像 embedding × 本条 prompt embedding | **否，逐记录独立** |

事实成立，因此可以按 `(resolved image path, source_image_sha256)` 分组做**一次视觉前向、多 prompt 复用**；prompt embedding 与 teacher 最大 cosine 仍逐条独立计算。

一个必须指出的数据结构事实：冻结 manifest **不是**按图像连续排列的——它是“先 236 条 grounding、再 236 条 pointing”，首次出现块数 = 472 而非 236。因此“只缓存当前图”的朴素流式实现对本数据会产生 **472 次**前向（无复用）。本实现按全 manifest 分组，并用有序写入器保持输出顺序。

## 2. 复用契约与 resume 语义

- 分组键：`(resolved path, source_image_sha256)`，跨整个 manifest；组内按 manifest 顺序。
- **每组最多一次视觉前向**：只要该图还有 pending prompt，就做一次；该图全部完成则零前向。
- **prompt embedding 逐记录调用**：`prompt_embeddings_fn` 每条 pending 记录一次，绝不跨 prompt 复用。
- **缓存仅驻留当前图**：组内 capture/image 在 `finally` 中置空，CUDA 下每组后 `torch.cuda.empty_cache()`；不会跨 236 图累积 tensor。
- **index 顺序 = manifest 顺序**：`_OrderedIndexWriter` 缓冲“提前就绪”的记录（例如 pointing 记录远在后方），按预先算出的 pending manifest 位置顺序 flush，因此分组顺序不影响输出顺序。
- **resume**：`--overwrite` 以 `"w"` 截断重建；否则以 `"a"` 追加。已完成的 sample 不写 NPZ/index。
- **同路径内容 hash 变化 = hard fail**：
  - 跨 run：当前文件 SHA256 与已完成样本记录的 `source_image_sha256` 不一致 → `DistillationContractError`；
  - run 内：同一 resolved path 两次得到不同 SHA256 → `DistillationContractError`。

## 3. 可注入、CPU 可测的执行层

新增 `run_prompt_projection_extraction(...)`，所有模型/处理器交互都通过注入回调：

`resolve_image_fn`、`image_sha256_fn`、`image_loader_fn`、`vision_capture_fn`、`prompt_embeddings_fn`，以及可替换的 `save_cache_fn` / `index_record_fn` / `release_fn`。
真实 `main()` 只负责 GPU 审计、模型加载与把真实处理器/视觉塔封装进这些回调；执行层本身不 import `transformers`/PIL、不碰 CUDA。

保留 `dense_vision_with_block0_capture`、`group_mean_from_block0`、`prompt_embeddings_from_text` 原语义不变；新增 `validate_vision_capture` 早失败校验。

## 4. 测试（新增 10 项，全部 CPU）

文件：`evaluation/tests/test_prompt_projection_feature_reuse.py`

- 同图 2 prompt = 1 次 vision + 2 次 prompt；
- 异图 = 各 1 次；
- partial resume = 1 次 vision（只写 pending 的 NPZ/index）；
- full resume = 0 次 vision；
- 同路径内容 hash 变化（resume / run 内）= hard fail；
- 同图分组输出与“逐样本 legacy 参考”在 `group_mean`/`prompt_embeddings`/`teacher_scores`/`teacher_scores_normalized`/`teacher_soft_target` 上**逐数组相等**；
- 交错图组仍按 manifest 顺序输出 index；
- `plan_image_groups` 首现顺序与 manifest 索引；
- `--validate-manifest-only` 子进程 import guard：不 import `transformers`/PIL。

回归结果（`/home/zhaoxiangyu/conda-envs/robobrain25/bin/python`）：

- `py_compile`：`extract_prompt_projection_features.py`、新测试、`prompt_projection_distillation.py`、`train_prompt_projection.py`、`run_hf_token_reduction.py` 全部通过；
- `test_prompt_projection_distillation`：37/37 OK；
- `test_prompt_projection_feature_reuse`：10/10 OK；
- 全量 `unittest discover -s evaluation/tests`：**222 tests OK**。

未使用 GPU，未 import 真实模型，未访问封存数据或 ZCU102。

## 5. 理论收益（非实测）

| 指标 | 无复用 | 本实现 | 说明 |
| --- | --- | --- | --- |
| vision forward | 472（每记录一次） | **236（每图一次）** | 结构计数，减少 **2.0×** |
| prompt embedding | 472 | 472 | 逐记录独立，不减少 |
| index 记录 | 472 | 472 | schema 不变 |

**边界**：2.0× 是 472/236 的**理论结构计数**。本任务未运行真实抽取，**不提供、不伪造任何实测 wall-clock 加速或端到端时间**。真实加速还取决于 processor/解码/写入占比。

## 6. 门禁与边界

- fail-closed：manifest pin 校验（相对路径 + sha256）仍在 GPU 审计与模型加载**之前**；
- GPU 审计仍先于真实模型加载；`--validate-manifest-only` 仍为纯 CPU、返回 `gpu_used=false, model_loaded=false`；
- **未**改教师/学生定义、NPZ schema、split 契约、manifest pin；
- 未运行真实抽取/训练/推理，未下载网络内容。

## 7. 文件

- `evaluation/extract_prompt_projection_features.py`（重构：分组复用 + 可注入执行层 + 有序 index 写入器）
- `evaluation/tests/test_prompt_projection_feature_reuse.py`（新增 10 项 CPU 测试）
- `reports/token_selector/2026-09-28-prompt-feature-image-reuse.{md,json}`
- `CURRENT_STATUS.md`

## 8. 精确下一步（未执行）

GPU 审计通过（无非本用户 compute 进程、空闲 ≥20480 MiB、非满载）后，才在独立授权门下运行：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/zhaoxiangyu/conda-envs/robobrain25/bin/python \
  evaluation/extract_prompt_projection_features.py \
  --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B \
  --manifest datasets/prompt_projection_clean_inductive_v1.jsonl \
  --out-dir artifacts/token_selector/prompt_projection_features
```

预期报告 `vision_forwards=236`（472 记录 / 236 图）；随后再训练 `Wt/Wv`。在此之前不得加载模型或用 GPU。
