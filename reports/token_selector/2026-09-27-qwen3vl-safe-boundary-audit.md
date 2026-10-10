# Qwen3-VL 最早安全视觉 Token 压缩边界审计

- 日期：2026-09-27
- 状态：`READ_ONLY_STRUCTURAL_AUDIT / NO_GPU / NO_ALGORITHM_CHANGE`
- 工作区：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`
- 结论一句话：最早“结构上可实现”的边界是 `B0`（Patch Embedding 之后、首个视觉块之前）；最早“低风险”的边界是 `B5`（第 1 个 DeepStack 提取点，零基 layer 5，前 6 块）。二者不同，原因见第 8 节。

本报告只做只读代码/文档审计，不实现算法、不训练、不运行 GPU、不修改任何既有项目文件。除本报告与同名 JSON 产物以及 `/tmp/robobrain_harness_result.json` 外没有新增/修改文件。

## 0. 审计依据与证据边界

已读取（只读）：

- `AGENTS.md`、`CURRENT_STATUS.md`（第 3-17、67-137、236-284 行关于 layer-5 L2、Q8-A/Q8-B、Cutpoint Oracle 的记录）。
- 项目实现：`evaluation/shallow_block_pruning.py`、`evaluation/run_hf_shallow_block_pruning.py`、`evaluation/token_reduction.py`、`evaluation/cutpoint_oracle.py`、`evaluation/run_hf_cutpoint_oracle.py`、`evaluation/run_cv_bench_refinement.py`、`evaluation/visual_refinement.py`、`evaluation/extract_task_conditioned_roi_scout.py`、`evaluation/extract_roi_vision_prefix_features.py`。
- 测试：`evaluation/tests/test_shallow_block_pruning.py`、`evaluation/tests/test_cutpoint_oracle.py`、`evaluation/tests/test_cv_refinement.py`。
- 本地 Qwen3-VL 源码（conda site-packages 可读，非 blocked）：`/home/zhaoxiangyu/conda-envs/robobrain25/lib/python3.10/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py`（transformers 4.57.0）、`configuration_qwen3_vl.py`。
- 本地部署路径：`/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B/config.json`、`profiling_src/llama.cpp/tools/mtmd/models/qwen3vl.cpp`、`profiling_src/llama.cpp/tools/mtmd/clip.cpp`。
- 既有实测报告：`reports/2026-09-04-cutpoint-oracle-preflight.md`、`reports/2026-09-04-cutpoint-oracle-summary.md`、`reports/2026-09-04-q8-cutpoint-path-audit.md`、`reports/2026-09-10-q8b-real-q8-validation.md`、`reports/2026-09-11-q8b-expanded-diagnostic.md`、`reports/2026-09-11-q8-token-pruning-failure-analysis.md`、`reports/2026-09-11-q8-stage-attribution.md`。

事实约束（来自 `RoboBrain2.5-4B/config.json`）：

| 字段 | 值 |
|---|---|
| `vision_config.depth` | 24 |
| `vision_config.hidden_size` | 1024 |
| `vision_config.num_heads` | 16 |
| `vision_config.out_hidden_size` | 2560 |
| `vision_config.spatial_merge_size` | 2 |
| `vision_config.deepstack_visual_indexes` | `[5, 11, 17]`（零基层号 5 是第一个 DeepStack 点） |
| `text_config.hidden_size` | 2560 |
| `text_config.num_hidden_layers` | 36 |
| `image_token_id` | 151655 |

本审计不重新评估精度，不把旧报告的数字当成新实验。

## 1. 现有 layer5 L2 pruning 与 global/local refinement 实现与测试

### 1.1 layer5 L2 pruning（当前冻结边界）

- 视觉前向入口：`evaluation/shallow_block_pruning.py::forward_shallow_pruned_vision`（第 381-847 行）。
  - Patch Embedding + 学习位置：第 416-418 行。
  - 视觉 RoPE、`cu_seqlens`：第 419-429 行。
  - 现有默认 `prune_after_layer=5`（`evaluation/run_hf_shallow_block_pruning.py` 第 416 行），即先完整跑 6 个视觉块。
  - layer 5 处：先由 `deepstack_merger_list[0]` 对完整格点算 DeepStack 特征（第 501-504 行），再按 `selected_groups` 压紧（第 813-816 行）。
  - 选择器 `l2_anchor`：块内 hidden state 按 2x2 group 取 L2 范数（第 530 行），由 `structured_group_indices`（第 86-136 行）选出高分 group 加 3x3 区域锚点。
  - 压紧：`hidden_states`、视觉 `position_embeddings`、`cu_seqlens` 同步按 `selected_patches` 索引（第 818-826 行）；DeepStack 已存特征按 `selected_groups` 压紧（第 806-811 行）。
  - Merger：第 830 行；随后校验 DeepStack 与主视觉 token 对齐（第 831-832 行）。
- LLM 侧接入：`prepare_pruned_multimodal_inputs`（第 850-891 行）。
  - 用**完整** `input_ids` + `image_grid_thw` 调 `model.model.get_rope_index` 得到原始 M-RoPE（第 865-870 行），再按保留掩码切片（第 871-876 行），因此保留 token 保留原二维/三维位置，不重编号。
  - 占位符计数、`inputs_embeds`、`deepstack_visual_embeds` 同步（第 877-882 行）。
- 延迟/运行：`evaluation/run_hf_shallow_block_pruning.py::run_one`（第 198-406 行），默认 `--prune-after-layer 5`、`--anchor-grid 3`、`--selector l2_anchor`（第 416-432 行）。
- 测试：`evaluation/tests/test_shallow_block_pruning.py` 覆盖区域锚点（第 46-49 行）、精确保留数/确定性（第 51-57 行）、100% 恒等（第 59-61 行）、局部残差（第 63-77 行）、区域配额（第 79-87 行）、文本相关（第 89-105 行）、几何 oracle（第 107-127 行）、以及**所有切点保持 DeepStack 对齐**（第 129-176 行，`cut in (-1, 0, 2, 5)`）。

### 1.2 global/local refinement（64 Token 协议）

- 入口：`evaluation/run_cv_bench_refinement.py`，策略 `dense / global256 / cosine_global_roi`（第 33 行、第 81-123 行）。
  - `global256`：原图长边缩到 256（第 81 行）。
  - `cosine_global_roi`：低清图先跑视觉前 6 层 + 第一个 DeepStack 投影得每格特征（`spatial_scout_features`，`evaluation/extract_task_conditioned_roi_scout.py` 第 36-77 行），文本 embedding 与之算最大余弦（第 66-69 行），3x3 平滑选峰后裁 600x600 框、局部长边 384（第 108-112 行），再与低清全图一起送入模型（第 114-123 行）。
  - 纯函数在 `evaluation/visual_refinement.py`：`resize_long_edge`、`crop_image_exact_1000`、`make_global_roi_messages`、`grid_token_counts`。
- 测试：`evaluation/tests/test_cv_refinement.py`；64 Token 结果见 `CURRENT_STATUS.md` 第 3-17 行（`global_local` 被 `global256` 与 `l2_50` 支配）。该实现是**输入端两图方案**，与视觉 Token 删除是两条独立路径。

## 2. Qwen3-VL 本地源码数据流（HF transformers 4.57.0）

`modeling_qwen3_vl.py`：

1. Patch Embedding：`Qwen3VLVisionPatchEmbed`（第 59-76 行），`Conv3d(temporal_patch_size=2, 16, 16)`，输出 `(n_patches, 1024)`。
2. 学习位置：`fast_pos_embed_interpolate`（第 642-701 行），把 patch 重排成 **2x2 group 连续**顺序：`view(t, h/2, 2, w/2, 2, -1).permute(0,1,3,2,4,5).flatten(0,4)`（第 694-698 行）。
3. 视觉 RoPE：`rot_pos_emb`（第 603-640 行），每个 patch 用全分辨率 `(row, col)` 二维坐标；在 `forward` 中拼成 `(cos, sin)`（第 719-725 行）。
4. `cu_seqlens`：按每图 `t*h*w` 累加（第 727-735 行）；**本实现没有 `window_index`**，视觉注意力在每张图内是全局的（`Qwen3VLVisionAttention.is_causal=False`，第 180 行）。
5. 24 个 `Qwen3VLVisionBlock`：第 738-744 行循环；`config.depth=24`。
6. DeepStack：层号命中 `deepstack_visual_indexes=[5,11,17]` 时，由 `deepstack_merger_list[i]` 生成特征并收集（第 745-749 行）。
7. 主 Merger：`self.merger`，`use_postshuffle_norm=False`（第 751 行；类定义第 93-106 行）。
8. 返回 `(image_embeds, deepstack_feature_lists)`（第 753 行）；`get_image_features` 按图切分（第 1050-1064 行）。
9. LLM 接入：`Qwen3VLModel.forward` 用 `masked_scatter` 把视觉 token 填进占位符（第 1137-1143 行），构造 `visual_pos_masks` 与 `deepstack_visual_embeds`（第 1153-1175 行），M-RoPE 由 `get_rope_index` 产生（第 1177-1221 行、第 916-1033 行）。
10. 文本层内 DeepStack 注入：`_deepstack_process` 把每层 DeepStack 特征加到对应视觉位置（第 876-883 行）。

PatchMerger 结构：`hidden_size = 1024 * spatial_merge_size**2 = 4096`；`forward` 中 `view(-1, 4096)` -> `LayerNorm` -> `Linear(4096,4096)` -> `GELU` -> `Linear(4096,2560)`（第 96-106 行）。DeepStack 版本 `use_postshuffle_norm=True`（第 591-599 行）。

GGUF/量化部署图 `profiling_src/llama.cpp/tools/mtmd/models/qwen3vl.cpp`：

- 固定 `n_pos=n_patches`（第 9 行）；可选 `q8b_patch_indices`（第 11-13 行）在第 1 个块后 gather（第 162-166 行），并切换压缩后的 M-RoPE 位置（第 122-128 行）。
- DeepStack 在压缩序列上 reshape 生成（第 168-183 行）；主投影在第 193-201 行；拼接在第 203-205 行。
- `clip.cpp` 计算 `compact_positions` 并写入 `q8b_positions`（第 3845-3859 行），并校验输出 token 数（第 4514-4525 行）。
- 现状：**B1 在真实 Q8 图内已经实现**；`B0`、`B5`、`BM` 目前在 HF 路径有实现，GGUF 路径没有对应固定预算 gather 分支。

## 3. 候选边界逐项审计

约定：`B0`=Patch Embedding 后、块 0 前（`prune_after_layer=-1`，完成 0 块）；`B1`=块 0 后（`prune_after_layer=0`，完成 1 块）；`B5`=零基层 5 后（`prune_after_layer=5`，完成 6 块，第一个 DeepStack 点）；`BM`=Merger 后、进 LLM 前（用 `prune_after_layer=23` + 固定逐 group 掩码实现，完成 24 块）。映射定义见 `evaluation/cutpoint_oracle.py` 第 17-30 行。

视编码器 MAC 比例来自 `reports/2026-09-04-cutpoint-oracle-preflight.md` 的静态代理（不含 patch embedding、merger、DeepStack 投影、文本 token），只作相对比较。

### B0 Patch Embedding 之后、首个 Vision Block 之前

- 可行性：**结构上可行；可部署选择器“有条件可行”**。
  - 代码证据：`shallow_block_pruning.py` 第 452-489 行的 `prune_after_layer == -1` 分支；单元测试 `test_all_cutpoint_classes_keep_deepstack_aligned` 用 `cut=-1` 验证（`test_shallow_block_pruning.py` 第 163-176 行）。
  - 条件：该分支只允许 `fixed_indices` 或 `geometric_oracle`（第 474-477 行），`l2_anchor` 等内容相关选择器会抛错；此时任何内容相关分数只能来自 patch embedding 原始输出，尚无块级上下文。
- 压缩时同步更新的 Tensor/元数据：
  - `hidden_states` 按 `selected_patches` 索引（第 482 行）。
  - 视觉 `position_embeddings=(cos,sin)` 按同一 patch 索引（第 483-486 行，保留原始全分辨率 RoPE 位置）。
  - `cu_seqlens=[0, selected_patches.numel()]`（第 487-489 行）。
  - `selected_groups/selected_patches/group_scores`；主 Merger 输入 token 数（第 830 行）。
  - `deepstack_features` 此刻为空（第一个提取点是层 5）。
  - LLM 侧：`input_ids/attention_mask` 掩码、`get_rope_index` 原始 M-RoPE 切片、图像占位符、`visual_pos_masks`、`deepstack_visual_embeds`（`prepare_pruned_multimodal_inputs` 第 865-882 行）。
- 是否降低 vision encoder 主体计算：**是，且最大**。全部 24 块在压缩序列上运行；静态编码器 MAC 比例 0.75/0.50/0.25 保留时为 `0.731/0.474/0.231`（preflight 第 10-12 行）。
- 是否破坏 DeepStack/Merger/M-RoPE：**结构上不破坏**。DeepStack `[5,11,17]` 全在切点之后，特征直接在压缩序列上生成，无跨层压紧；Merger 每 4 个连续 patch 合并；M-RoPE 不重编号。
- 需要修改的具体函数与行号（若要加入内容相关选择器）：`evaluation/shallow_block_pruning.py` 第 452-489 行（放宽选择器、补分数来源）；`evaluation/run_hf_shallow_block_pruning.py` 第 267-325 行（文本 embedding 与选择器参数传入）。GGUF 路径需在 `qwen3vl.cpp` patch/位置相加之后新增 gather 与压缩位置（当前无）。
- 已测风险：几何 Oracle@75% 门控 **FAIL**（Pointing PCK@0.1 `0.750 -> 0.500`；`reports/2026-09-04-cutpoint-oracle-summary.md` 第 10 行，每任务仅 4 条烟雾样本）。

### B1 第 1 个 Vision Block 之后

- 可行性：**可行**。
  - HF：`prune_after_layer=0`，块循环第 491-496 行后进入切点逻辑，所有选择器可用。
  - 真实 Q8：已在 `qwen3vl.cpp` 第 162-166 行（`il==0` 后 `ggml_get_rows`）、第 122-128 行（`q8b_positions`）、`clip.cpp` 第 3845-3859 行与第 4514-4525 行实现。
- 同步更新的 Tensor/元数据：`hidden_states`（第 819 行）、视觉 `position_embeddings`（第 820-823 行）、`cu_seqlens`（第 824-826 行）、主 Merger 输入、LLM 侧同 B0（第 865-882 行）；`deepstack_features` 为空，后续 5/11/17 在压缩序列上生成。
- 是否降低 vision encoder 主体计算：**是**。1 块全序列 + 23 块压缩；静态编码器 MAC 比例 0.75/0.50/0.25 为 `0.742/0.496/0.263`（preflight 第 14-16 行）。
- 是否破坏 DeepStack/Merger/M-RoPE：**不破坏**。所有 DeepStack 提取点都在块 0 之后，无需跨层压紧；M-RoPE 用原始 patch/二维索引（Q8 侧 `positions_cur` 对 `il==0` 用完整 `positions`，之后用 `q8b_positions`）。
- 已测风险：真实 Q8-B@87.5% 扩大 96 图上 Pointing PCK@0.1 `0.6458 -> 0.5104`，差值 `-0.1354`，95% 区间 `[-0.2190,-0.0521]`，确认退化（`reports/2026-09-11-q8b-expanded-diagnostic.md` 第 9 行；`reports/2026-09-11-q8-token-pruning-failure-analysis.md` 第 16 行）；Q8-B 相对 Q8-A 的 Pointing PCK@0.1 为 `-0.2069`，区间 `[-0.3793,-0.0345]`（`reports/2026-09-11-q8-stage-attribution.md` 第 24 行）。100% 身份门通过。

### B5 第 5 层/前 6 块边界（当前冻结边界）

- 可行性：**可行，且是当前生产路径**。
- 同步更新的 Tensor/元数据：层 5 的 DeepStack 特征先由 `deepstack_merger_list[0]` 从完整 hidden state 算出（第 501-504 行），再按 `selected_groups` 压紧（第 813-816 行）；此前 DeepStack 列表为空（第 806-811 行的压紧循环只对已存特征生效）；`hidden_states/position_embeddings/cu_seqlens` 同步压紧（第 818-826 行）；主 Merger 第 830 行；LLM 侧第 865-882 行。
- 是否降低 vision encoder 主体计算：**是**。6 块全序列 + 18 块压缩；静态编码器 MAC 比例 0.75/0.50/0.25 为 `0.798/0.606/0.423`（preflight 第 22-24 行）。
- 是否破坏 DeepStack/Merger/M-RoPE：**不破坏**。切点恰好等于第一个 DeepStack 索引 5，特征在同一层完成“全序列计算 + 按 group 压紧”，语义与稠密路径对保留 group 逐元素一致；后续 11/17 自然生成在压缩序列上。
- 已测风险：CV-Bench 完整 2638 题、真实 BF16、87.5% 保留时相对分数保留 `0.9995474605`，配对 95% 区间 `[-0.0045079013,+0.0037806757]` 含零（`CURRENT_STATUS.md` 第 112-113 行）；这是唯一有全量配对质量证据的边界。默认真实运行命令使用 `--prune-after-layer 5 --anchor-grid 3 --selector l2_anchor`（`run_hf_shallow_block_pruning.py` 第 416-432 行）。

### BM PatchMerger 之后、进入 LLM 之前

- 可行性：**可行（结构）**；用 `prune_after_layer=23` 的固定逐 group 掩码实现，在每 group merger 固定时与“先 merger 再选”数值等价（`evaluation/cutpoint_oracle.py` 第 26-28 行；`reports/2026-09-04-cutpoint-oracle-preflight.md` 第 32 行）。
- 同步更新的 Tensor/元数据：层 5/11/17 已捕获的 DeepStack 特征按 `selected_groups` 压紧（第 806-811 行）；层 23 处 `hidden_states/position_embeddings/cu_seqlens` 压紧（第 818-826 行）；`visual.merger` 只对保留 group 运算（第 830 行）；LLM 侧第 865-882 行。
- 是否降低 vision encoder 主体计算：**否**。全部 24 块在完整序列上运行，静态编码器 MAC 比例在所有保留率下为 `1.000`（preflight 第 26-28 行）；只降低 LLM 视觉 Prefill（prefill MAC 比例 0.75/0.50/0.25 约 `0.747/0.496/0.247`）。`reports/2026-09-04-q8-cutpoint-path-audit.md` 第 37 行明确把它定为“只能减少 LLM 视觉 Prefill，不能减少 Vision Encoder，不能作为最终性能方案”。
- 是否破坏 DeepStack/Merger/M-RoPE：**不破坏**。Merger 与 DeepStack 在完整视觉栈上执行后按同一 group 索引压紧。
- 已测风险：真实 Q8-A@87.5% 扩大 96 图上 Pointing PCK@0.1 `-0.0938`，区间 `[-0.1667,-0.0208]`；Grounding Mean IoU `-0.0510`，区间 `[-0.0755,-0.0269]`（`reports/2026-09-11-q8-token-pruning-failure-analysis.md` 第 20-21 行）。即“只动 Merger 后”也不保证质量无损，且无编码器收益。

## 4. 最小合法选择单位：2x2 merge group、窗口、空间原始索引

- **最小合法选择单位 = 1 个完整 2x2 空间 merge group = 4 个连续 patch token**（`merge_unit = spatial_merge_size**2 = 4`）。
  - PatchMerger 固定消费 4 个连续 patch：`hidden_size = config.hidden_size * spatial_merge_size**2 = 4096`，`forward` 中 `view(-1, 4096)`（`modeling_qwen3_vl.py` 第 96、104 行）。
  - DeepStack merger 同样按 4 patch 拼接后归一（`use_postshuffle_norm=True`，第 591-599 行）。
  - `fast_pos_embed_interpolate` 已把 patch 排成 group 连续顺序（第 694-698 行），`rot_pos_emb` 的 `(row,col)` 也按同一 group 顺序展开（第 623-627 行）。
  - LLM 每个 group 只有一个图像占位符与一个 M-RoPE 位置：`get_rope_index` 的 `llm_grid_h=h//2`、`llm_grid_w=w//2`，每格一个位置（第 987-1001 行）；占位符数 `full_llm_visual_tokens = full_patch_tokens / merge_unit`（`run_hf_shallow_block_pruning.py` 第 244-248 行）。
  - Q8 图显式断言 `n_pos_q8b % 4 == 0`（`qwen3vl.cpp` 第 13 行）。
- **窗口**：本 transformers 4.57.0 的 Qwen3-VL 视觉注意力**没有滑动窗口/`window_index`**（`grep window_index` 无命中）；只有按图的 `cu_seqlens` 变长全注意力（第 727-735 行）。因此没有“窗口对齐”这一约束，唯一的局部单位就是 2x2 merge group。
- **空间原始索引**：选择必须保留**原始**索引，不重编号：
  - 视觉侧每个 patch 保留 `rot_pos_emb` 的原始 `(row,col)`，靠 `index_select` 取得（第 820-823 行）。
  - LLM 侧对完整 `input_ids` 先算 `get_rope_index`，再按保留掩码切片（第 865-876 行），保留 token 的二维 M-RoPE 不变。
  - group 内顺序 = `(intra_row, intra_col)`；group 间 = merged 行优先。选择越界或重复索引由 `validate_fixed_group_indices` 拒绝（第 54-83 行）。

## 5. Prompt 查询向量的结构事实（只给结构，不设计训练方法）

- LLM token embedding 维度：`text_config.hidden_size = 2560`；`Qwen3VLTextModel.embed_tokens = nn.Embedding(vocab_size=151936, hidden_size=2560)`（`modeling_qwen3_vl.py` 第 771 行）。运行器已经从 LLM embedding 取文本向量：`model.get_input_embeddings()(batch.input_ids[0, text_mask])`（`run_hf_shallow_block_pruning.py` 第 267-284 行），形状 `[n_text, 2560]`。
- vision hidden 维度：1024（`vision_config.hidden_size`，patch embed 输出与块 hidden 均为 1024）。
- 现有 projector 方向：`Qwen3VLVisionPatchMerger` 把 4x1024 拼成 4096，经 `Linear(4096,4096)->GELU->Linear(4096,2560)` 输出 2560（`modeling_qwen3_vl.py` 第 93-106 行）；`visual.merger` 与 `deepstack_merger_list` 都输出 `out_hidden_size=2560`（第 585-599 行）。**在 merger MLP 之外不存在独立的 1024->2560 投影**。
- 现有文本-视觉打分要求同维：`text_relevance_group_scores` 断言 `projected_groups.shape[1] == text_embeddings.shape[1]`（`shallow_block_pruning.py` 第 354 行）。因此 2560 维的 Prompt 查询向量只能与**已 merger/DeepStack 投影后的** 2560 维视觉 group 直接比较，不能与 1024 维原始块 hidden 直接比较。
- 侦察路径的现有做法（结构事实）：`fixed_projection(2560, 8, seed)` 用固定 ±1 随机矩阵把 2560 降到 8（`extract_roi_vision_prefix_features.py` 第 31-35 行），作用在 `deepstack_merger_list[i](hidden)` 的输出上（`extract_task_conditioned_roi_scout.py` 第 66-68 行）。此处只陈述结构，不评价其效果，也不提出新的训练/投影方案。

## 6. 汇总表

| 边界 | 零基层 | 完成块 | 可行性 | 是否减 Vision Encoder | 编码器 MAC 比 @0.75/0.50/0.25 | DeepStack | Merger | M-RoPE | 已测质量风险 |
|---|---|---|---|---|---|---|---|---|---|
| B0 | -1 | 0 | 结构可行 / 选择器有条件可行 | 是（全部 24 块） | 0.731 / 0.474 / 0.231 | 切点后生成，不破坏 | 不破坏 | 保留原索引 | Oracle@75% FAIL（PCK@0.1 0.75->0.50） |
| B1 | 0 | 1 | 可行（HF + 真实 Q8 图） | 是（23 块） | 0.742 / 0.496 / 0.263 | 切点后生成，不破坏 | 不破坏 | 保留原索引 | Q8@87.5% 扩大 96 图 Pointing PCK@0.1 -0.1354（确认退化） |
| B5 | 5 | 6 | 可行（当前冻结） | 是（18 块） | 0.798 / 0.606 / 0.423 | 与首个提取点同层，专门处理 | 不破坏 | 保留原索引 | CV-Bench 2638@87.5% 相对 0.99955，CI 含零 |
| BM | 23 | 24 | 可行（固定逐 group 掩码） | **否** | 1.000 / 1.000 / 1.000 | 全栈后压紧，不破坏 | 不破坏 | 保留原索引 | Q8-A@87.5% Pointing -0.0938、IoU -0.0510 |

同步张量/元数据清单（四个边界共同）：保留 group/patch 索引、视觉 `position_embeddings`、`cu_seqlens`、主视觉 token 顺序、全部 DeepStack 通道、LLM `input_ids/attention_mask/position_ids/visual_pos_masks`、图像占位符计数与 `image_grid_thw`。差异仅在 DeepStack 是否有“已存特征需要跨层压紧”。

## 7. 结论

- **最早结构上可实现边界：`B0`**。在 Patch Embedding + 学习位置之后，2x2 group 结构与原始视觉 RoPE 已经定义，M-RoPE 由保留掩码切片保持，DeepStack 全在切点之后，主 Merger 天然按 group 工作；HF 代码已实现（`prune_after_layer=-1`，`shallow_block_pruning.py` 第 452-489 行）且有 `cut=-1` 单元测试。但它目前只接受固定掩码或几何 Oracle，不存在可部署的内容相关选择器。
- **最早低风险边界：`B5`**。它是当前冻结的 layer-5 L2 边界：唯一具备全量配对质量证据（CV-Bench 2638@87.5% 相对 0.99955、CI 含零）、默认生产选择器（`l2_anchor`+3x3 锚点）、且切点恰好落在第一个 DeepStack 提取索引 5 上，DeepStack 压紧语义精确处理；测试覆盖最完整。
- 两者不同的原因：**“结构上可实现”只问张量/元数据契约能否闭合；“低风险”还必须包含可部署选择器是否存在与已测质量是否成立**。`B0` 满足结构契约但没有可用内容信号，且其几何 Oracle@75% 已 FAIL；`B1` 虽然比 `B5` 浅、结构也干净、真实 Q8 图已实现，但在 87.5% 的真实 Q8 扩大诊断上确认了 Pointing PCK@0.1 退化，因此不能算低风险；`BM` 结构风险最低但没有 Vision Encoder 收益，且 Q8-A 在同一 87.5% 上也出现显著下降。综合结构、选择器可用性、已测质量和编码器收益，最早的低风险边界是 `B5`。

## 8. 明确未做的事与证据边界

- 未运行 GPU、未加载模型、未修改任何项目文件；未访问封存 512/校准 96 集；未访问或配置 ZCU102；未读取凭据/SSH/代理/许可证内容。
- 未实现新算法、未训练投影或选择器、未提出研究路线。
- 静态 MAC 比来自既有 preflight，不含 patch embedding、Merger、DeepStack 投影、文本 token，不是硬件实测。
- B0 的“无内容选择器”是当前代码事实，不代表原理上不可行；本审计不外推。
- B5 的 CV-Bench 结果是在该集合上观察曲线的一部分，不是未观察盲测；B1/BM 的负结果来自几何 Oracle 诊断，选择器不同，不能当作同策略的直接对照。
