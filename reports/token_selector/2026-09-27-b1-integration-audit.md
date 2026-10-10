# hardware_regular_token_selector 在 Qwen3-VL B1（vision block 0 之后）接入的精确接口地图

- 日期：2026-09-27
- 状态：`READ_ONLY_INTERFACE_AUDIT / NO_GPU / NO_ALGORITHM_CHANGE / NO_MODEL_EDIT`
- 工作区：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`
- 边界：`B1` = 跑完零基 vision block 0、在做 block 1 之前；`prune_after_layer=0`，`completed_vision_blocks=1`。
- 结论一句话：B1 接入在结构上闭合；`shallow_block_pruning.py` 已给出“真实 group gather”的全部同步点，`hardware_regular_token_selector.py` 给出的 `groups/group_indices/patch_indices` 可直接替换其中 `index_select` 的索引来源；**唯一需要改变 gather 语义的是 synthetic context group —— 它必须用选择器的 `groups.reshape(K*4,1024)` 作为新的 hidden，而不能用 `patch_indices` 去 `index_select` 原激活**。真实 group 与 context 代表索引全局唯一，因此 K 个输出 group 天然对应 K 个唯一 placeholder，文本次序不变。
- 本文件只回答接口问题，不实现接入、不训练、不运行 GPU、不修改原模型目录。除本报告、同名 JSON 与 `/tmp/robobrain_harness_result.json` 外未新增/修改任何文件。

## 0. 证据范围（全部只读）

已读取：

- 复用候选：`evaluation/run_hf_shallow_block_pruning.py`、`evaluation/run_hf_token_reduction.py`、`evaluation/run_hf_cutpoint_oracle.py`、`evaluation/run_hf_spatial_baseline.py`。
- 选择与压紧：`evaluation/shallow_block_pruning.py`、`evaluation/token_reduction.py`、`evaluation/cutpoint_oracle.py`、`evaluation/hardware_regular_token_selector.py`、`configs/hardware_regular_token_selector.json`。
- 测试：`evaluation/tests/test_shallow_block_pruning.py`、`evaluation/tests/test_cutpoint_oracle.py`、`evaluation/tests/test_hardware_regular_token_selector.py`。
- 本地 transformers 4.57.0 只读源码：`/home/zhaoxiangyu/conda-envs/robobrain25/lib/python3.10/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py`（sha256 `dd63ed3b...b22da`），未修改。
- 既有 B1 的 GGUF/Q8 路径：`profiling_src/llama.cpp/tools/mtmd/models/qwen3vl.cpp`、`profiling_src/llama.cpp/tools/mtmd/clip.cpp`（只读，不提交、不修改）。
- 既有审计：`reports/token_selector/2026-09-27-qwen3vl-safe-boundary-audit.md`、`.../2026-09-27-hardware-regular-selector-contract.md`、`CURRENT_STATUS.md`。

环境事实：`numpy 2.2.6`、`torch 2.5.1+cu124`、本沙箱 `torch.cuda.is_available() == False`。
证据文件 sha256：

| 文件 | sha256 |
|---|---|
| `evaluation/shallow_block_pruning.py` | `a753972f9a98d2bbfc5ae0fff40df8c05da8935147be52d5afcd0d006bcff84a` |
| `evaluation/run_hf_shallow_block_pruning.py` | `41a7e9c8a777dbf88dd39c6e713e201be75770b50ebc41a9936c42f9e09ed5b9` |
| `evaluation/hardware_regular_token_selector.py` | `7e55f930c966812091ebca0fecf2a07b68e5c13a36e39ffdf2cb78c6d0239a79` |
| `evaluation/token_reduction.py` | `6af9b507fb4a612b4013937150d6a7a59c8cf76d7e310c427b7fdb918f9a77b8` |
| `evaluation/cutpoint_oracle.py` | `750f3eb1ef8f6385f239fcaa708506f91656298edea96ef5fe6dcd3e03f742e6` |
| `evaluation/run_hf_cutpoint_oracle.py` | `820c3626fbcd70bd67e5d92061c3905fe506c51e7bdfd66c25b4023654780d50` |
| `evaluation/run_hf_token_reduction.py` | `28e9870d17c9f1b63389e55d09f92118291ad15adb3c3c33c99686a973df5c8e` |
| `modeling_qwen3_vl.py` | `dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da` |

固定模型事实（`RoboBrain2.5-4B/config.json`）：`vision_depth=24`、`vision_hidden=1024`、`num_heads=16`、`out_hidden_size=2560`、`spatial_merge_size=2`、`deepstack_visual_indexes=[5,11,17]`、`text_hidden=2560`、`text_depth=36`、`image_token_id=151655`。

## 1. 可直接复用的 runner、hook 与函数（名字 + 行号）

### 1.1 没有 monkey patch；接入方式是“重写视觉前向、调用原模块”

项目里**不存在对 HF `forward` 的 monkey patch**。全仓库只有一处 forward hook：

- `evaluation/run_hf_token_reduction.py::CudaCallTimer`（第 34-62 行）：`register_forward_pre_hook` / `register_forward_hook`（第 38-39 行），仅用于逐 block 计时，`close()` 注销（第 60-62 行）。

真正的“接入机制”是 `evaluation/shallow_block_pruning.py::forward_shallow_pruned_vision`（第 381-847 行）：它不调用 `Qwen3VLVisionModel.forward`，而是**手工重跑** `patch_embed → pos → rotary → blocks → deepstack → merger`，从而能在任意层插入压紧。B1 复用这一模式即可。

### 1.2 可直接复用的函数清单

| 可复用项 | 位置 | 用途 |
|---|---|---|
| `forward_shallow_pruned_vision` | `shallow_block_pruning.py:381-847` | B1 视觉前向骨架；把 `prune_after_layer=0` 的 `index_select` 换成选择器输出 |
| `prepare_pruned_multimodal_inputs` | `shallow_block_pruning.py:850-891` | LLM 侧 placeholder/`inputs_embeds`/M-RoPE/`visual_pos_masks`/DeepStack 同步 |
| `PrunedVisionOutput` | `shallow_block_pruning.py:26-40` | 把选择器结果包装成既有 dataclass 以便复用上两个函数 |
| `PrunedMultimodalInputs` | `shallow_block_pruning.py:43-51` | LLM prefill 入参 |
| `validate_fixed_group_indices` | `shallow_block_pruning.py:54-83` | 固定索引校验（唯一、范围、升序），可作为接入的对照实现 |
| `text_relevance_group_scores` | `shallow_block_pruning.py:347-360` | 2560 维文本-视觉余弦（现有 Wt/Wv 之外的同一打分语义参照） |
| `structured_group_indices` | `shallow_block_pruning.py:86-136` | 3x3 anchor + Top-K，现有 `l2_anchor` 选择器 |
| `greedy_generate_from_pruned` | `run_hf_shallow_block_pruning.py:117-190` | 自定义贪心解码；直接消费 `PrunedMultimodalInputs` |
| `run_one` | `run_hf_shallow_block_pruning.py:198-406` | 单样本端到端：预处理→视觉→准备→生成→记录 |
| `main` | `run_hf_shallow_block_pruning.py:409-740` | 可续跑 CLI、manifest 扫描、warmup |
| 默认 CLI | `run_hf_shallow_block_pruning.py:415-432` | `--prune-after-layer 5 --selector l2_anchor`；B1 改 `--prune-after-layer 0` |
| CUDA 前置校验 | `run_hf_shallow_block_pruning.py:473-474` | 无 CUDA 直接报错（沙箱内不可跑） |
| `limit_each_task` / `completed_ids` | `run_hf_shallow_block_pruning.py:85-95` / `51-52` | 采样与断点续跑 |
| `cuda_seconds` | `run_hf_shallow_block_pruning.py:98-105` | GPU 计时 |
| B1 的 `run_one` 复用范例 | `run_hf_cutpoint_oracle.py:150-186` | 传 `prune_after_layer=point.prune_after_layer`，B1 即 `0` |
| `CUT_POINTS["vision_block_1"]` | `cutpoint_oracle.py:17-30` | `prune_after_layer=0, completed_vision_blocks=1, transfer_hidden=1024` |
| `transfer_accounting` | `cutpoint_oracle.py:40-106` | 传输量核算（B1 属于 `patch_hidden`） |
| `qwen3_vl_shallow_pruned_compute_proxy` | `token_reduction.py:183-235` | 逐层 MAC 代理（B1 用 `prune_after_layer=0`） |
| `retention_label` | `token_reduction.py:238-241` | 输出命名 |
| `select_hardware_regular_groups` | `hardware_regular_token_selector.py:623-789` | 主选择入口（K 固定、动态 context、全局唯一代表） |
| `_identity_bypass` | `hardware_regular_token_selector.py:593-620` | `G <= K` 逐元素旁路 |
| `select_real_groups` | `hardware_regular_token_selector.py:420-518` | anchor + 两级有界 Top-K + context 保留位 |
| `build_regions` / `region_id_by_group` | `hardware_regular_token_selector.py:292-339` / `396-403` | 3x3 区域与 group→region 映射 |
| `context_pooling_weights` / `pool_region_context` | `hardware_regular_token_selector.py:526-536` / `539-560` | context 池化（`[4,1024]`） |
| `region_context_representative` | `hardware_regular_token_selector.py:356-393` | 距加权质心最近的未选 group |
| `LowRankProjection` / `score_groups` | `hardware_regular_token_selector.py:163-181` / `254-270` | `Wt[128,2560]`、`Wv[128,1024]` 与三分数 |
| `describe_fpga_layout` | `hardware_regular_token_selector.py:903-920` | 传输段布局描述 |

### 1.3 runner 装配点（B1 复用的最短路径）

`run_one` 的可替换点非常集中：

- 文本 embedding（做 `Wt` 输入）：`run_hf_shallow_block_pruning.py:267-284`，取 `model.get_input_embeddings()(batch.input_ids[0, text_mask])`，形状 `[T, 2560]`。
- 视觉调用：`run_hf_shallow_block_pruning.py:286-325`（`vision_call`）。
- 视觉→LLM 装配：`run_hf_shallow_block_pruning.py:327-343`（`prepare_pruned_multimodal_inputs` + `greedy_generate_from_pruned`）。
- 逐 block 计时 hook：`run_hf_shallow_block_pruning.py:618`（`CudaCallTimer(model.model.visual.blocks[i])`）。

## 2. B1 张量形状与索引流（单张静图，`grid_thw=[[1,H,W]]`）

记 `P = H*W`（patch 数）、`G = P/4`（2x2 merged group 数）、`K`（选择器固定输出预算，16 的倍数，生产 `48/64/96`）。patch 顺序 = group-major，组内顺序 = `intra_row*2 + intra_col`，由 `fast_pos_embed_interpolate` 第 694-698 行的 `view(t,h//2,2,w//2,2,-1).permute(0,1,3,2,4,5).flatten(0,4)` 与 `rot_pos_emb` 第 623-627 行共同保证。

| 阶段 | 代码位置 | 形状 / 索引 |
|---|---|---|
| `pixel_values` | processor 输出 | `[P, 3*2*16*16]`（`Conv3d(2,16,16)` 展平） |
| `patch_embed` | `modeling:59-76` | `[P, 1024]`（第 72-75 行 `view(-1,3,2,16,16)` → `view(-1,1024)`） |
| 学习位置 | `modeling:642-701` | `[P, 1024]`，已 group-major |
| 相加 | `modeling:716-717` / `shallow:416-418` | `[P, 1024]` |
| `rot_pos_emb` | `modeling:603-640` | `[P, 64]`（`(row,col)` 各 32），group-major |
| 视觉 RoPE | `modeling:719-725` / `shallow:419-424` | `emb=[P,128]`，`position_embeddings=(cos[P,128], sin[P,128])` |
| `cu_seqlens` | `modeling:727-735` / `shallow:425-429` | 单图 `[0, P]`（int32） |
| block 0 | `modeling:738-744` / `shallow:491-496` | `[P, 1024]`（图内全注意力，`is_causal=False`，`modeling:180`） |
| **B1 切点** | `shallow:506`, `shallow:818-826` | `hidden.view(G,4,1024)` 即 `hardware_regular_token_selector` 的输入 |
| 选择器输出 | `hardware:623-789` | `groups[K,4,1024]`、`group_indices[K]`、`patch_indices[K,4]`、`source_type[K]`、`region_id[K]`、`valid[K]` |
| 压紧 hidden | 现有 `shallow:819`；接入改为 `groups.reshape(K*4,1024)` | `[K*4, 1024]` |
| 压紧 RoPE | 现有 `shallow:820-823`；接入用 `patch_indices.reshape(-1)` | `(cos[K*4,128], sin[K*4,128])` |
| 新 `cu_seqlens` | 现有 `shallow:824-826` | `[0, K*4]`（int32） |
| blocks 1..23 | `shallow:491-496` | `[K*4, 1024]` |
| DeepStack 5/11/17 | `modeling:745-749` / `shallow:501-504` | 每个 `deepstack_merger_list[i](hidden)`：`[K*4,1024] → [K,2560]`，共 3 个 |
| 主 merger | `modeling:751` / `shallow:830` | `[K*4,1024] → [K,2560]` |
| LLM placeholder | `shallow:850-891` | 全序列 `[1,S]` → 保留 `S-(G-K)`；`visual_mask` 恰 `K` 个 True |
| M-RoPE | `modeling:916-1033` / `shallow:865-876` | 由**完整** `input_ids` 算 `[3,1,S]`，再按 `keep_mask` 切为 `[3,1,S-(G-K)]` |

关键结构事实：
- `PatchMerger.hidden_size = 1024*4 = 4096`，`view(-1,4096)`（`modeling:96,104`），所以**每 4 个连续 patch 必须成组**；`patch_indices` 必须是 `4g+[0,1,2,3]` 形式。
- `deepstack_merger_list` 用 `use_postshuffle_norm=True`（`modeling:591-599`），同样按 4 patch 拼接后归一。
- 每个 merged group 在 LLM 侧只有 1 个 image placeholder 和 1 个 M-RoPE 位置（`modeling:987-1001`，`llm_grid_h=h//2, llm_grid_w=w//2`）。
- `get_image_features` 按 `image_grid_thw.prod(-1)//4` 切图（`modeling:1062-1064`）；B1 单图不需要切。

## 3. 现有“真实 group gather”如何处理 cu_seqlens / position_embeddings / DeepStack / placeholder

“真实 group gather”指 `forward_shallow_pruned_vision` 在 `layer_num == prune_after_layer` 处的动作，全部为 `index_select`：

1. `selected_patches`：由 `selected_groups[:,None]*4 + offsets` 展平（`shallow:803-804`），保证真实 group 保留原连续 4 patch。
2. **hidden**：`hidden_states = hidden_states.index_select(0, selected_patches)`（`shallow:819`）。
3. **position_embeddings**：`cos/sin` 各 `index_select(0, selected_patches)`（`shallow:820-823`），**保留原始 `(row,col)`，不重编号**。
4. **cu_seqlens**：直接重建为 `[0, selected_patches.numel()]`（`shallow:824-826`），单图下合法。
5. **DeepStack**：切点之前已捕获的特征按 **group 级** `index_select(0, selected_groups)`（`shallow:806-811`）；切点当层的 feature 先在全序列算好、再同样按 group 压紧（`shallow:813-816`）；切点之后的层在压缩序列上自然生成，无需再压紧。B1 的 `min(deepstack_visual_indexes)=5 > 0`，所以 **B1 时 `deepstack_features` 必为空**，不存在跨层压紧。
6. **placeholder / LLM 侧**（`prepare_pruned_multimodal_inputs`）：
   - 先校验完整 placeholder 数等于 `full_llm_visual_tokens=G`（`shallow:861-863`）——**必须用完整 `input_ids`，不能预先删 placeholder**。
   - 用完整 `input_ids` 调 `get_rope_index` 得原始 M-RoPE（`shallow:865-870`）。
   - `keep_mask`：先全 True，再把 image 位置全置 False，再把 `image_positions[selected_group_indices]` 置 True（`shallow:871-873`）。
   - 切 `input_ids`/`attention_mask`/`position_ids`（`shallow:874-876`），重建 `inputs_embeds`（`shallow:877`），把 `image_embeds` 写进 `visual_mask` 行（`shallow:878-881`），DeepStack 逐通道搬设备（`shallow:882`）。
   - 末尾对齐校验：`int(visual_mask.sum()) == image_embeds.shape[0]`（`shallow:879-880`）。

Q8/GGUF 侧对 B1 的现有实现（`qwen3vl.cpp:162-166` 在 `il==0` 后 `ggml_get_rows`；`qwen3vl.cpp:122-128` 从 block 1 起改用 `q8b_positions`；`clip.cpp:3845-3859` 由完整 `positions` gather 出 `compact_positions`；`clip.cpp:4514-4525` 断言输出 token 数 = `len(q8b_patch_indices)/4`）与 HF 的 `index_select` 语义一致：**只做真实 patch 的 gather，不做池化**，因此它没有、也不能表达 synthetic context group。

## 4. 加入 synthetic context group 必须同步修改的张量，以及“原位置复用”合法性

### 4.1 必须同步修改/重建的清单

选择器的输出已经把需要同步的量编码齐全（`groups`、`group_indices`、`patch_indices`、`source_type`、`region_id`、`valid`）。接入时逐项：

| 张量 | B1 之后的动作 | 与真实 gather 的差异 |
|---|---|---|
| vision `hidden_states` | `selection.groups.reshape(K*4,1024)`；**不能** `index_select(patch_indices)` | context 行是池化值，gather 只会取回代表 group 的原始激活，丢失 context |
| vision `position_embeddings` (cos,sin) | `index_select(0, patch_indices.reshape(-1))` | 与真实 gather 同构；context 用代表 group 的 4 个原始位置 |
| vision `cu_seqlens` | `[0, K*4]` int32 | 同真实 gather |
| DeepStack 已捕获特征 | B1 处为空，无需压紧 | 同真实 gather |
| DeepStack 后续 5/11/17 | 在 `[K*4,1024]` 上自然生成，每个 `[K,2560]` | context 与真实 token 混在同一压缩序列，按行对齐 |
| 主 merger | `visual.merger([K*4,1024]) → [K,2560]` | 同真实 gather |
| LLM `input_ids/attention_mask` | `keep_mask`：保留全部文本 + `image_positions[group_indices]` | 关键差异：索引来自 `group_indices`（含 context 代表），不是 `selected_groups` |
| LLM `position_ids` | 完整 `get_rope_index` 后按同一 `keep_mask` 切 | 同真实 gather；context 继承代表 group 的原始 `(t,h,w)` |
| LLM `inputs_embeds` | 重新 embedding + 覆盖 `visual_mask` 行为 `image_embeds` | 同真实 gather |
| `visual_pos_masks` | `pruned_ids == image_token_id`，恰 K 个 True | 同真实 gather |
| `deepstack_visual_embeds` | 3 个 `[K,2560]`，注入层 0/1/2（`modeling:862-867`） | 顺序必须与 `visual_pos_masks` 的 True 顺序一致 |
| 硬件元数据 | `group_indices/source_type/region_id/valid` 透传（FPGA 段布局见 `hardware:903-920`） | 新增：真实 gather 没有 `source_type=1` |

选择器契约对同步有利的两点：

- `patch_indices [K,4]` 已经给出每个输出 group 应使用的**原始 patch 位置**：real 为 `4g+[0,1,2,3]`，context 为代表 group 的 `4*rep+[0,1,2,3]`（`hardware:712`、`740-742`）。
- `entries.sort(key=lambda entry: (entry[0], entry[1], entry[2]))`（`hardware:757`）使输出按代表 `original_group_index` 升序，因此 `patch_indices` 展平后天然升序且唯一。

### 4.2 context 代表原位置能否合法复用：可以，但只在“结构合法”意义上

判定依据（逐条可验证，均只涉及位置/顺序契约，不涉及精度）：

1. **索引合法**：context 代表取自该 region 的**未选** group（`hardware:725,739`），一定落在 `[0,G)`，故其 4 个 patch 是合法原 patch。
2. **全局唯一**：real 来自被选集合，context 代表来自未选集合，两者不相交（`hardware:75` 的契约 + `select_real_groups` 的 context reservation，`hardware:496-499`），因此 `patch_indices` 的行两两不相交，不会抢占同一 placeholder。
3. **视觉 RoPE 一致**：context 使用代表 group 的 `rot_pos_emb` 坐标（通过 `patch_indices` gather），与 LLM 侧该 placeholder 的 M-RoPE `(h,w)` 是同一个原始 grid 单元（`modeling:987-1001`）。
4. **注意力无窗口约束**：Qwen3-VL 视觉注意力为每图全注意力、`is_causal=False`、无 `window_index`（`modeling:180,727-735`），所以“池化 token 与真实 token 混排”不违反注意力结构。
5. **merger/DeepStack 只看 4 连续 patch**：`view(-1,4096)`（`modeling:104`）只要求组内 4 patch 连续，不要求这 4 个值来自同一次前向的真实位置。
6. **LLM 注入只按 mask 行对齐**：`_deepstack_process` 用 `hidden_states[visual_pos_masks]` 顺序相加（`modeling:876-883`），顺序由 `keep_mask` 决定。

必须明确：以上只证明**位置与索引合法**。它不证明池化 context 的表征质量，也不改变 B1 real-gather 已有退化证据；本审计不做精度判断。

### 4.3 接入时唯一不能照抄的地方

现有真实 gather 是“用索引取原激活”。context 行必须改为**直接使用 `selection.groups`**。正确写法是先构造压紧 hidden：

```text
compact_hidden = torch.as_tensor(selection.groups, dtype=hidden.dtype, device=hidden.device).reshape(K*4, 1024)
compact_cos   = cos.index_select(0, patch_idx_flat)
compact_sin   = sin.index_select(0, patch_idx_flat)
```

若误用 `block0_hidden.index_select(0, patch_idx_flat)`，real 行数值相同，但 context 行会退化成代表 group 的原激活，`source_type=1` 的语义被静默破坏（且不会报错）。

## 5. 如何保证 K 个输出 group 对应 K 个唯一 placeholder 且文本次序不变

构造性保证 + 接入侧应做的断言：

1. **唯一性来源**：`select_hardware_regular_groups` 保证 real 与 context 代表 `original_group_index` **全局唯一**（`hardware:75,776`），`group_indices` 是这 K 个代表并按升序排序（`hardware:757,765`）。
2. **K 个 placeholder**：`keep_mask[image_positions] = False` 后仅 `keep_mask[image_positions[group_indices]] = True`（对照 `shallow:871-873`）。因索引唯一且 `< G`，恰好保留 K 个 image placeholder。
3. **文本次序**：`keep_mask` 是从全 True 出发只删 G-K 个 image 位置，所有文本 token 的 `keep_mask` 恒为 True，`input_ids[:, keep_mask]` 是原序列的**保序子序列**，文本相对次序不变。
4. **视觉 token 与 placeholder 的配对顺序**：`keep_mask` 使保留的 placeholder 按原位置升序排列；`group_indices` 已升序，且 `groups` 与 `group_indices` 由同一次 `entries.sort` 同步排序（`hardware:757`），所以第 i 行视觉特征恰好对应第 i 个保留 placeholder。接入应显式断言：

```text
assert np.all(np.diff(group_indices) > 0)                      # 严格升序，无重复
assert selection.output_group_count == K == int(selection.valid.sum())
assert set(group_indices).__len__() == K
# 保序子序列：保留的 image placeholder 原位置升序 == image_positions[group_indices]
```

5. **数量自检**：`int(visual_mask.sum()) == vision.image_embeds.shape[0]`（`shallow:879-880`）在 context 路径下同样成立，因为 `visual_mask` 由 `group_indices` 生成、`image_embeds` 行数也是 K。
6. **若上游不保证升序**：接入侧必须先按 `group_indices` 对 `groups`/`patch_indices` 重排，再构造 `keep_mask`；不要依赖“恰好已排序”。

## 6. identity（K >= G）时的逐元素等价测试怎么做

`G <= budget` 时选择器走 `_identity_bypass`（`hardware:593-620`，入口判断 `hardware:661-662`）：`groups = hidden.copy()`、`group_indices = arange(G)`、`patch_indices = 4g+[0,1,2,3]`、`source_type` 全 0、`context_group_count = 0`、`valid` 全 True。等价测试分三层，越往后端到端约束越强：

1. **纯 CPU 原语层（无需模型，可立即跑）**
   - `torch.equal(torch.from_numpy(sel.groups), torch.from_numpy(hidden))`；
   - `sel.group_indices == arange(G)`、`sel.patch_indices == arange(4G).view(G,4)`、`sel.valid.all()`、`sel.context_group_count == 0`。
2. **视觉数值层（CPU 可用 `FakeVisual`，GPU 用真模型）**
   - `compact_hidden` 与 block 0 输出逐元素相等；
   - gather 出的 `(compact_cos, compact_sin)` 与原始 `position_embeddings` 逐元素相等；
   - 继续跑 blocks 1..23，输出与同一输入下的稠密 `Qwen3VLVisionModel` 逐元素相等（`K=G` 时形状完全相同，不引入额外归约顺序差异，允许 `torch.equal`）；
   - `deepstack_embeds` 三个通道、`image_embeds` 与稠密路径逐元素相等；长度都为 K。
   - 参照 `evaluation/tests/test_shallow_block_pruning.py:129-176` 的 `FakeVisual` 手法与 `test_full_retention_is_identity`（第 59-61 行）。
3. **端到端层（需 CUDA）**
   - `keep_mask` 全 True → `pruned_ids == input_ids`、`pruned_positions == full_position_ids`；
   - `greedy_generate_from_pruned` 的生成 token ids 与稠密基线（`run_hf_spatial_baseline.py:120-124` 的 `model.generate(do_sample=False)`）逐 token 一致；这正是既有 Q8-B 身份门的做法（`CURRENT_STATUS.md:1135`：20 个生成 Token ID 逐项一致）。
   - 注意：`run_hf_shallow_block_pruning.py:473-474` 强制 CUDA，沙箱内 `torch.cuda.is_available()==False`，端到端层当前无法执行。

对 `bf16` 的说明：`K=G` 时计算图形状完全一致，理论上是逐位等价；若接入实现用了不同的 `reshape/contiguous` 布局，仍应期望 `torch.equal` 成立（元素级运算未变），而不是引入容差。

## 7. 最小实现文件清单与验收测试（≥8）

### 7.1 最小实现文件清单（均为新增，不改既有实现）

| 文件 | 作用 |
|---|---|
| `evaluation/hardware_regular_b1_integration.py` | 桥接层：`(a)` block0 输出 `[P,1024] → [G,4,1024]`；`(b)` 调 `select_hardware_regular_groups`（numpy，需显式 `LowRankProjection`）；`(c)` 用 `groups.reshape(K*4,1024)` + `patch_indices` gather 位置，继续 blocks 1..23、DeepStack、merger；`(d)` 包装成 `PrunedVisionOutput` 以复用 `prepare_pruned_multimodal_inputs`；`(e)` 输出 `source_type/region_id/valid` |
| `evaluation/run_hf_hardware_regular_b1.py` | runner：复用 `CudaCallTimer`、`make_messages/processor_inputs/resolve_image`、`run_one` 骨架、`greedy_generate_from_pruned`、`selection_sha256`、`retention_label` 与 `qwen3_vl_shallow_pruned_compute_proxy(prune_after_layer=0)` |
| `evaluation/tests/test_hardware_regular_b1_integration.py` | 下列 13 条验收测试（CPU 子集 + 可选 GPU 端到端） |
| `evaluation/build_hardware_regular_b1_report.py`（可选） | 汇总 JSONL → 报告 |
| 复用不修改 | `evaluation/shallow_block_pruning.py`、`evaluation/hardware_regular_token_selector.py`、`configs/hardware_regular_token_selector.json` |

### 7.2 验收测试（13 条）

1. `test_identity_bypass_k_ge_g_hidden_and_patch_indices_elementwise`——`_identity_bypass` 下 `groups`、`group_indices`、`patch_indices`、`valid` 严格逐元素相等。
2. `test_identity_bypass_k_ge_g_deepstack_and_merger_equal_dense`——blocks 1..23、DeepStack 3 通道、merger 输出与稠密路径 `torch.equal`。
3. `test_real_only_matches_existing_fixed_indices_gather`——用 `forward_shallow_pruned_vision(prune_after_layer=0, selector="fixed_indices")` 作对照，real-only 路径逐元素一致。
4. `test_context_rows_use_representative_patch_positions`——`source_type==1` 行满足 `patch_indices[i] == 4*group_indices[i]+[0,1,2,3]`，且代表来自该 region 未选集合。
5. `test_output_exactly_k_and_source_type_counts`——`output_group_count==K`、`real+context==K`、`valid` 全 True、无 padding/`-1`。
6. `test_group_indices_globally_unique_and_strictly_ascending`——`len(set)==K` 且 `np.diff>0`。
7. `test_placeholder_keep_mask_yields_k_visual_tokens_and_preserves_text_order`——`visual_mask.sum()==K`；非 image token 子序列与完整序列中的文本子序列逐元素相同。
8. `test_mrope_retained_positions_equal_full_positions`——`pruned_positions` 等于完整 `get_rope_index` 结果在保留列上的切片，文本 token 位置不变。
9. `test_cu_seqlens_rebuilt_and_downstream_shapes`——新 `cu_seqlens==[0,K*4]`；`image_embeds.shape==(K,2560)`；DeepStack 为 3×`(K,2560)`。
10. `test_block0_full_then_blocks_1_23_compacted`——用 `CudaCallTimer`/计数 hook 证明 block 0 输入 `P`、block 1..23 输入 `K*4`。
11. `test_end_to_end_greedy_ids_match_dense_at_identity`——`K>=G` 时生成 token ids 与稠密基线逐 token 一致（需 CUDA，沙箱跳过并标记）。
12. `test_invalid_budget_and_insufficient_context_raise`——K 非 16 倍数、`budget-C < 非空区域数`、投影形状错误、边界非法均抛 `SelectorContractError`/`InsufficientBudgetError`。
13. `test_selection_is_deterministic_with_lower_index_tiebreak`——同输入两次运行 `group_indices/groups` 完全一致；全同分取更小 `original_group_index`。

## 8. 当前最大未知 / 阻塞

1. **`Wt/Wv` 权重不存在（最大阻塞）**：压缩路径强制要求显式 `LowRankProjection`（`hardware:664-675`），`configs/hardware_regular_token_selector.json` 的 `training.weights_generated=false`，仓库内没有任何 `Wt/Wv` 产物。没有权重，B1 压缩选择无法产生真实语义分数；identity 旁路可以跑，压缩不能。
2. **无 GPU / 沙箱 `torch.cuda.is_available()==False`**：`run_hf_shallow_block_pruning.py:473-474` 强制 CUDA；端到端身份与质量门当前不可执行。`CURRENT_STATUS.md:25` 已记录 `BLOCKED_GPU_ACCESS_UNAVAILABLE_IN_SANDBOX`。
3. **选择器是 CPU float64 numpy**：block0 的 `[G,4,1024]` 需从 GPU/bf16 转到 host float64，再转回；这条数据通路没有现成函数，性能与数值回写未验证。
4. **context 语义未验证**：结构上合法（第 4.2 节）不等于质量安全；B1 的 real-gather 在真实 Q8 扩大诊断上已记录 Pointing PCK@0.1 退化（`reports/2026-09-11-q8b-expanded-diagnostic.md`），本审计不重估精度。
5. **M-RoPE 下一步位置假设**：`greedy_generate_from_pruned` 用 `prepared.position_ids.max()+1`（`run_hf_shallow_block_pruning.py:145`）。仅当保留序列中仍存在“原最大位置”的文本 token 时才与稠密 `rope_delta` 等价；若 prompt 以 image placeholder 结尾，该假设可能失效。需要第 8 条验收测试之外的显式边界测试。
6. **必须用完整 `input_ids`**：`prepare_pruned_multimodal_inputs` 要求完整 placeholder 数 == G（`shallow:861-863`）。任何提前裁剪 `input_ids` 的接入都会在装配处报错。
7. **`patch_indices` 升序依赖**：若未来选择器改为非升序输出，接入需自行重排，否则 placeholder 与视觉行错配（第 5 节第 6 点）。

## 9. 结论

- B1 接入的复用面完整：runner 骨架（`run_one`/`greedy_generate_from_pruned`）、计时 hook（`CudaCallTimer`）、同步函数（`forward_shallow_pruned_vision`/`prepare_pruned_multimodal_inputs`）均可直接使用。
- 与真实 group gather 的唯一语义分歧是 synthetic context：hidden 必须来自 `selection.groups`，position/cu_seqlens/placeholder/M-RoPE 沿用 `patch_indices` 与 `group_indices` 的原位置 gather/slice。
- K 个输出 group ↔ K 个唯一 placeholder 由“代表索引全局唯一 + 升序 + keep_mask 保序”共同保证。
- 当前不可执行压缩路径的三个硬阻塞：缺 `Wt/Wv`、无 CUDA、CPU float64 桥未实现。
