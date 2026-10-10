# Qwen3-VL B1（vision block 0 之后）hardware-regular selector 软件集成实现

- 日期：2026-09-27
- 状态：`IMPLEMENTED_CPU_ONLY / NO_GPU / NO_MODEL_EDIT / NO_TRAINING / NO_ALGORITHM_SEARCH / TIMING_AND_HASH_CONTRACT_FIXED`
- 工作区：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`
- 边界：`B1` = 跑完零基 vision block 0、在做 block 1 之前（`prune_after_layer=0`，`completed_vision_blocks=1`）。
- 范围约束：只新增/修改任务允许的 7 个文件；未改原始模型、`app/`、`api.py`、`shallow_block_pruning.py`；未训练；未运行 GPU；未访问封存数据与 ZCU102。

本文件只记录本轮实现的接口、结构、测试与已知边界，不给出精度或加速结论，不做研究方向决策。

## 0. 修改范围与哈希

| 文件 | 动作 | sha256 |
|---|---|---|
| `evaluation/hardware_regular_token_selector.py` | 最小扩展 | `c55cfdc3…2f85932b` |
| `evaluation/hardware_regular_b1_integration.py` | 新增 + 计时契约修复 | `1854f437…e91f3d61` |
| `evaluation/run_hf_hardware_regular_b1.py` | 新增 + 命名计时/同步/哈希修复 | `8e18b6bf…52370611` |
| `evaluation/tests/test_hardware_regular_token_selector.py` | 追加回归 | `024fa71e…93fda5` |
| `evaluation/tests/test_hardware_regular_b1_integration.py` | 新增 + 计时/哈希测试 | `53d6154e…03166ad` |
| `reports/token_selector/2026-09-27-b1-integration-implementation.md` | 新增（本文件） | — |
| `artifacts/token_selector/2026-09-27-b1-integration-implementation.json` | 新增 | — |

复用但不修改：`evaluation/shallow_block_pruning.py`（sha256 仍为 `a753972f…cff84a`，与 B1 审计一致）、`evaluation/run_hf_shallow_block_pruning.py`（`greedy_generate_from_pruned`/`completed_ids`/`limit_each_task`/`selection_sha256`）、`evaluation/run_hf_token_reduction.py`（`make_messages`/`processor_inputs`/`resolve_image`）。

## 1. A) selector 的最小扩展

`evaluation/hardware_regular_token_selector.py` 只做了压缩路径的入口扩展，未改选择数学、区域规则、context 规则与输出契约：

- `GroupScores` 的 `semantic`/`l2` 变为可选（`combined` 始终存在），因为 l2 模式与显式分数路径没有 prompt 投影。
- 新增公开函数：
  - `minmax_normalize(values)`：`combined` 的 L2 单项归一化（`lambda_semantic=0` 时的 `combine_scores` 等价量）。
  - `scores_from_precomputed(semantic, l2, combined, lambda_semantic)`：直接装配三分数；`combined` 缺失时必须由显式 `semantic+l2+lambda_semantic` 推导，否则报错。
- `select_hardware_regular_groups(...)` 新增 `semantic_scores` / `l2_scores` / `combined_scores`：
  - 任一提供即进入显式分数路径，分数**按原值使用，不重新生成**；`combined.size` 必须等于 `G`。
  - 都不提供时保持原有 `text_embeddings + LowRankProjection + lambda_semantic` prompt 路径，行为不变。
  - `G <= K` 的 identity bypass 在任何分数参数提供之前返回，不需要任何分数或投影。
  - 模块不训练、不随机生成 `Wt/Wv`。

回归测试确认：显式 `semantic/l2/combined` 与 prompt 路径输出完全相同的 `group_indices`；l2-only 显式分数与 `lambda_semantic=0` 的 prompt 路径选择一致；尺寸不符、无法推导 combined、`lambda` 越界均显式报错。

## 2. B) B1 集成实现

`evaluation/hardware_regular_b1_integration.py` 复用 `shallow_block_pruning` 的骨架，但没有修改它。主入口：

```text
forward_hardware_regular_b1_vision(
    visual, pixel_values, grid_thw, *,
    budget, score_mode="l2", text_embeddings=None,
    projection=None, lambda_semantic=None,
    region_grid=3, local_candidate_cap=16, boundary="B1",
    synchronize=None,
)
```

数据流（单张静图，`P = H*W` patch，`G = P/4`，输出预算 `K`）：

1. `patch_embed → +pos → rot_pos_emb → (cos, sin)`，`cu_seqlens=[0,P]`；这一段（含 `fast_pos_embed_interpolate` 学习位置、rotary/RoPE 与初始 `cu_seqlens`）单独计时为 `vision_preamble_seconds`。
2. **block 0 在完整 `P` 上运行**（`block_input_shapes[0] == P`）。
3. `[P,1024] → [G,4,1024]`；转 host float64 调 `select_hardware_regular_groups`；再转回原 dtype/device。这一步单独计时为 `selector_cpu_seconds`。
4. 压紧：
   - `hidden = selection.groups.reshape(K*4, 1024)`（context 行是**池化值**，这是与真实 gather 唯一语义不同点）；
   - `cos/sin = full_cos/full_sin.index_select(0, patch_indices.reshape(-1))`（context 用代表 group 的原始位置）；
   - `cu_seqlens = [0, K*4]` int32。
5. blocks `1..N-1` 在 `K*4` 上运行；DeepStack `5/11/17` 各产生 `[K,2560]`；`merger` 产生 `image_embeds [K,2560]`。
6. 返回 `HardwareB1VisionOutput`：`pruned`（`PrunedVisionOutput`，可直接喂给 `prepare_pruned_multimodal_inputs`）、`block_input_shapes`、分阶段秒数（`vision_preamble_seconds`/`block0_seconds`/`selector_cpu_seconds`/`remaining_vision_seconds`，以及只读属性 `vision_total_seconds` 与 `gpu_vision_seconds`）、`compact_hidden/cos/sin/cu_seqlens`、`group_indices/patch_indices/source_type/region_id/valid` 等。
7. `prepare_b1_multimodal_inputs(model, batch, result)` 先断言 `group_indices` 严格升序唯一、`valid` 与 `K` 一致，再调用既有 `prepare_pruned_multimodal_inputs`。

### 2.1 视觉计时契约定理

- `vision_total_seconds == vision_preamble_seconds + block0_seconds + selector_cpu_seconds + remaining_vision_seconds`（`HardwareB1VisionOutput` 的属性定义，测试逐项断言）。
- `gpu_vision_seconds == vision_preamble_seconds + block0_seconds + remaining_vision_seconds`，即 `vision_total_seconds - selector_cpu_seconds`；CPU selector 往返永远从 GPU 视觉时间中剔除，不得计为 GPU compute gain。
- 每个计时段（preamble、block 0、selector、remaining）开始与结束均调用调用方传入的 `synchronize`（`_clock` 在读取起止 `perf_counter` 前各调用一次）；测试用计数回调确认至少 8 次调用。
- preamble 未单独计时是上一轮的缺陷：它把 `patch_embed`、学习位置、rotary/RoPE 与初始 `cu_seqlens` 排除在视觉总时间之外；本轮修复。

## 3. C) CUDA runner

`evaluation/run_hf_hardware_regular_b1.py`：

- `--score-mode l2`：`combined` 由 B1 group-mean L2 经 `minmax_normalize` 得到，不需要也不接受投影；传 `--projection-npz` 直接报错。
- `--score-mode prompt`：必须 `--projection-npz` 显式加载 `Wt[128,2560]`/`Wv[128,1024]`；文件缺失、键缺失、shape 不符、`--projection-sha256`（整文件哈希）或 npz 内 `weights_sha256`（数组内容哈希）不符都会报错。权重不训练、不随机生成。
- 生成固定为贪心 `argmax`（`do_sample=False`），模型 `local_files_only=True`、`HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`。
- 每样本 JSONL 按阶段命名计时，全部以 `torch.cuda.synchronize` 括住边界：
  - 应用层：`image_open_decode_seconds`、`processor_seconds`、`host_to_device_seconds`、`postprocess_seconds`、`application_end_to_end_seconds`。
  - prompt：`prompt_text_embedding_seconds`（仅 prompt 模式非零）。
  - 视觉：`vision_preamble_seconds`、`block0_seconds`、`selector_cpu_roundtrip_seconds`、`remaining_vision_seconds`、`vision_total_seconds`、`gpu_vision_seconds`。
  - LLM：`llm_prepare_seconds`、`llm_prefill_seconds`、`decoder_seconds`、`model_inference_seconds`。
  - 其它既有字段：`G`（`full_llm_visual_tokens`）、`K`（`output_group_count`）、`context_group_count`、`real_group_count`、`image_grid_thw`、`output_text`、`peak_cuda_bytes`、`group_indices/source_type/region_id/valid` 等。
- 旧字段 `elapsed_seconds`、`timed_stage_seconds` 保留为兼容别名，并在记录内 `deprecated_alias` 映射到 `model_inference_seconds`（视觉总时间 + LLM prepare/prefill/decode）；`vision_encoder_seconds` 保留且等于 `vision_total_seconds`。
- 记录固定 `"cpu_selector_roundtrip_is_acceleration_claim": false` 与 `"selector_cpu_roundtrip_excluded_from_gpu_vision": true`。**CPU selector 往返是把 `[G,4,1024]` 搬到 host 再搬回的开销，不是加速项**；`gpu_vision_seconds` 已剔除它，本报告不把它计入任何加速声明。
- 复用 `shallow_block_pruning` 的 CPU 选择器与既有贪心解码/预处理函数，未改原模型代码。

本轮无 GPU，runner 只做了 `py_compile`、import 与 CLI 校验（`--help`、prompt 缺投影报错、l2 误传投影报错），未做端到端推理。

### 3.1 投影哈希契约

- `--projection-sha256` 只校验**整个 npz 文件**（`file_sha256`）。
- 删除上一轮“把 npz 内 `sha256` 键与文件自身哈希比较”的逻辑：文件不可能包含自身摘要，该比较恒不成立，属于不可能的比较；遗留 `sha256` 键现在被忽略。
- 新增 `canonical_projection_sha256(Wt, Wv)`：把 `Wt`/`Wv` 固定转成 `float32`、C 连续，并把键名、dtype、shape 与字节一起做 SHA256；同内容不同来源 dtype/内存布局得到相同摘要，任何数值、shape 或 dtype 变化都会改变摘要。
- npz 可选键 `weights_sha256` 与 `canonical_projection_sha256(Wt, Wv)` 比较；不匹配报错。metadata 另记录 `projection_weights_sha256`。

## 4. D) placeholder 与 M-RoPE

- 复用完整 `input_ids` 先算 `get_rope_index`，再用 `keep_mask`（全 True → 删除全部 image 位置 → 只把 `image_positions[group_indices]` 置 True）切 `input_ids/attention_mask/position_ids`。
- `group_indices` 由 selector 保证全局唯一且升序，因此保留的恰好是 `K` 个视觉 placeholder，视觉行与 placeholder 一一对应。
- 文本 token 的 `keep_mask` 恒为 True，`input_ids[:, keep_mask]` 是原序列的保序子序列，文本相对次序不变。
- 测试验证：`visual_pos_masks.sum()==K`、image placeholder 恰 `K` 个、文本子序列逐元素相同、`position_ids` 等于完整 `get_rope_index` 结果在保留列上的切片、`get_rope_index` 收到的输入长度仍是完整序列长度。

## 5. E) CPU/fake 测试

`evaluation/tests/test_hardware_regular_b1_integration.py` 使用 fake visual（`hidden=1024`、`merge_unit=4`、`out_hidden=2560`、DeepStack `[5,11,17]`）与 fake model（`get_rope_index`、embedding），不加载真实权重、不需要 CUDA。覆盖：

1. identity 与稠密路径逐元素等价（`image_embeds` 与全部 DeepStack 通道）。
2. B1 只在 block 0 后压缩：block 0 输入 `P`，blocks 1..N-1 输入 `K*4`。
3. context 行确为池化值：等于 `selection.groups` 的池化行，且与代表 group 的原始 block0 激活不同；real 行仍等于原始激活。
4. cos/sin 用代表位置：等于完整 cos/sin 在 `patch_indices` 上的 gather，代表位置为 `4g+[0,1,2,3]` 且全局唯一。
5. `cu_seqlens == [0, K*4]` int32。
6. DeepStack 3 通道与 merger 形状均为 `[K,2560]`。
7. `K` 个 placeholder 且文本顺序不变。
8. M-RoPE 用完整 `input_ids` 先算再切片，文本位置不变。
9. 唯一升序索引。
10. l2 模式不需要投影（`projection=None`、`text_embeddings=None`）。
11. prompt 缺权重失败；提供显式投影时 prompt 模式可运行。
12. identity bypass 即使 score-mode 为 prompt 也不要求任何分数。
13. 投影加载器：合法加载并返回整文件哈希；文件缺失/键缺失/shape 不符/整文件 hash 不符分别报错。
14. 视觉计时契约：四个分段均有限且非负，`vision_total_seconds` 恰等于四段之和，`gpu_vision_seconds` 恰为总时间减去 selector CPU 往返；计数 `synchronize` 回调确认每段起止各调用一次（≥8 次）。
15. `canonical_projection_sha256` 对同一数组稳定，对副本 / 不同来源 dtype / Fortran 布局不变，对单个元素篡改或 shape 变化必变。
16. `weights_sha256` 键与内容哈希一致时加载成功，篡改数组或错哈希时报错；遗留 `sha256` 键被忽略而不是与文件哈希比较。

`evaluation/tests/test_hardware_regular_token_selector.py` 追加既有选择器回归：显式分数与 prompt 路径一致、l2-only 不需要投影、combined 推导规则、尺寸不符报错、`minmax_normalize` 等价。

## 6. 测试与编译结果

```bash
/home/zhaoxiangyu/conda-envs/robobrain25/bin/python -m unittest \
  evaluation.tests.test_hardware_regular_token_selector \
  evaluation.tests.test_hardware_regular_b1_integration \
  evaluation.tests.test_shallow_block_pruning
```

结果：`Ran 55 tests ... OK`（selector 23、B1 集成 19、shallow 13；failed=0）。

```bash
/home/zhaoxiangyu/conda-envs/robobrain25/bin/python -m py_compile \
  evaluation/hardware_regular_token_selector.py \
  evaluation/hardware_regular_b1_integration.py \
  evaluation/run_hf_hardware_regular_b1.py \
  evaluation/tests/test_hardware_regular_token_selector.py \
  evaluation/tests/test_hardware_regular_b1_integration.py
```

结果：`PY_COMPILE_OK`。

## 7. 已知边界

- 本轮**未运行 GPU**；runner 未做端到端推理，只有编译/import/CLI 校验。
- 计时契约只在 CPU/fake 路径上验证分段求和与同步调用次数；没有真实 GPU 秒数，因此不给出任何性能或加速结论。
- 仓库内不存在 `Wt/Wv` 权重；prompt 模式必须由调用方显式提供 npz，否则拒绝运行。
- CPU selector 往返是 host 往返开销，不是加速；本报告不作任何加速或精度声明。
- context 池化的合法性仅是结构/索引层面的；其表征质量未验证。
- `B0` 保留为同接口消融标签，但没有单独的 B0 集成测试。
- 解码下一步 M-RoPE 位置沿用既有 `greedy_generate_from_pruned` 的 `position_ids.max()+1` 假设。
- 因任务限定允许文件集合，未修改 `CURRENT_STATUS.md`（与 `AGENTS.md` 第 7 条的常规要求有意偏离）。

## 8. 机读产物

- `artifacts/token_selector/2026-09-27-b1-integration-implementation.json`
- 交接审计（只读）：`reports/token_selector/2026-09-27-b1-integration-audit.md`、`...json`
