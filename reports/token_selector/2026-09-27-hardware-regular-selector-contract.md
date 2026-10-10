# 硬件规则化 Token 选择与 FPGA 传输布局合同（纯 CPU）

- 日期：2026-09-27（修订：动态 context 契约）
- 状态：`CONTRACT_IMPLEMENTED_CPU_ONLY / NO_MODEL / NO_TRAINING / NO_GPU`
- 工作区：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`
- 边界：主插入边界 `B1`（vision block 0 之后）；`B0`（Patch Embedding 之后、block 0 之前）仅作为同一接口的消融标签，选择数学完全相同。

本文件只写实现合同、测试结果与已知未接入项，不给出研究建议，不重新评估精度。

## 0. 修改范围

按任务冻结范围，仅新增/修改以下文件：

- `evaluation/hardware_regular_token_selector.py`（实现，本次修订）
- `evaluation/tests/test_hardware_regular_token_selector.py`（测试，本次修订）
- `configs/hardware_regular_token_selector.json`（配置，本次修订）
- `reports/token_selector/2026-09-27-hardware-regular-selector-contract.md`（本文件）
- `artifacts/token_selector/2026-09-27-hardware-regular-selector-contract.json`（机读产物）
- `/tmp/robobrain_harness_result.json`

未修改原始模型、`app/`、`api.py`，未训练，未运行 GPU，未访问封存数据或 ZCU102。因任务范围明确限定允许文件，`CURRENT_STATUS.md` 未修改（与 `AGENTS.md` 第 7 条的常规要求有意偏离，以便遵守本次冻结范围）。

## 1. 实现合同

### 1.1 最小单位与边界

- 最小选择单位是完整 2x2 merge group，输入 hidden 形状固定为 `[G, 4, 1024]`。
- `evaluation/hardware_regular_token_selector.py` 不接入模型，只接受 NumPy 数组；`MERGE_UNIT=4`、`FEATURE_DIM=1024`、`PROMPT_DIM=2560`、`SCORE_DIM=128`。
- `boundary` 只接受 `B1` / `B0` 两个标签，写入结果元数据；默认 `B1`。

### 1.2 Prompt 输入与低秩投影接口

- Prompt 输入接口为 `text_embeddings[T, 2560]`。
- 仅定义可训练低秩投影接口 `Wt: [128, 2560]`、`Wv: [128, 1024]`（`LowRankProjection` 数据类，构造时校验形状与有限性）。
- 模块不训练、不随机生成正式权重；权重必须由调用方显式提供。视觉组表示为 4 个 patch 的均值。

### 1.3 三个分数接口

| 接口 | 函数 | 定义 |
|---|---|---|
| semantic cosine | `semantic_cosine_scores` | 组均值经 `Wv` 投影、文本经 `Wt` 投影后逐行 L2 归一，取每个视觉组对全部 prompt token 的最大余弦 |
| group L2 saliency | `group_l2_saliency` | 组均值表示的 L2 范数 |
| normalized weighted combined | `combine_scores` / `score_groups` | `lambda_semantic * minmax(semantic) + (1 - lambda_semantic) * minmax(l2)`，`lambda_semantic` 必须显式配置且落在 `[0,1]` |

`score_groups` 一次性返回三者；`GroupScores` 分别保存 `semantic`、`l2`、`combined` 与 `lambda_semantic`，互不共享内存。

### 1.4 固定总预算 K、动态 context 与 bypass

- 总输出预算 `K` 仍严格固定：必须是 16 的正倍数，生产预算为 `48/64/96`（配置 `budgets`）。
- `G <= K` 时走 identity bypass：逐元素返回原 hidden，`source_type` 全为 real，输出组数等于 `G`，context 数为 0，不需要 prompt 或投影。
- 压缩路径输出严格 `K` 组；`K` 被拆成**动态**的真实组数 `K - C` 与 context 组数 `C`，`C` 不再是固定 9。

### 1.5 固定 3x3 区域、anchor 与两级 Top-K

- 空间按 `build_regions` 固定划分为 3x3 区域；每个非空区域至少保留 1 个真实 group 作为 anchor（区内 combined 最高分，同分取更小 `original_group_index`）。
- 其余 detail group 通过两级 Top-K：
  1. 每区先取有界 local candidates（`local_candidate_cap`，默认 16，上限为区内可用非 anchor 组数）；
  2. 只在候选集合（`<= 9 * cap`）上取全局最高 combined 分，同分按 `original_group_index` 升序。
- 不对全部 `G` 个 group 做全局全排序；候选不足时抛 `InsufficientBudgetError`。
- 对需要生成 context 的区域，候选池再减 1，从结构上保证该区域至少留下 1 个未选 group。

### 1.6 动态 context 槽（最多 9）

- context 槽数量动态，最多 `MAX_CONTEXT_SLOTS = 9`（即 3x3 区域数）。
- **context eligible region** 判定：该 region 至少含 2 个输入 group，且压缩后仍有未选 group。压缩选择被强制为每个 eligible region 至少保留 1 个未选 group，因此 `C = context eligible region 数`。
- 只有 eligible region 才生成 context；空 region 或仅含 1 个 group 的 region 不生成任何 context，也不产生零 token。
- context 由该 region **全部**未选 group 以非负权重 `max(combined, 0)` 归一化后池化得到 `[4, 1024]` synthetic group（`context_pooling_weights` + `pool_region_context`）；全为 0 时退化为均匀权重。
- context 代表索引必须来自本 region 未选 group，取距**加权质心**最近者（`region_context_representative`），同距取更小 `original_group_index`，结果稳定确定。
- 若 `budget - C < 非空区域数`（例如 `K=16`、9 个非空区域），无法为每个非空区域保留 anchor，抛 `InsufficientBudgetError`。

### 1.7 输出契约

- `groups [K,4,1024]`、`group_indices [K]`、`patch_indices [K,4]`、`source_type [K]`、`region_id [K]`、`valid [K]`。
- 真实 group 的 patch 索引是原连续 4 patch（`4g + [0,1,2,3]`）；context 使用代表 group 的 4 个位置并标记 synthetic。
- 所有 real 与 context 的代表 `original_group_index` **全局唯一**：real 来自被选真实 group，context 来自各自 region 的未选 group，二者不相交。
- **不允许 invalid/zero padding token**：输出 `valid` 恒为全 `true`，不存在 `original_group_index = -1` 或 `patch_indices = -1` 的占位条目；context 数即为实际生成数。
- 输出按代表 `original_group_index` 稳定升序；索引已全局唯一，故排序无歧义。所有排序同分时按 `original_group_index` 升序。

### 1.8 硬件读取 metadata 与 FPGA 传输布局（只描述，不写二进制）

- 硬件必须读取 `source_type`（`SOURCE_REAL=0`、`SOURCE_CONTEXT=1`）来区分动态 context group 与真实 group，不能假定固定 9 个 context 或固定位置。
- 段顺序：`tokens_int8`（`K*4*1024` 字节）→ `group_index_uint16`（`2K`）→ `source_type_uint8`（`K`）→ `region_id_uint8`（`K`）→ `valid_uint8`（`K`）。
- 每段 offset 64 字节对齐，总长度按 64 字节对齐；布局按总预算 `K` 计算，与动态 context 数无关。
- 计算函数：`int8_token_bytes`、`uint16_group_index_bytes`、`uint8_source_bytes`、`uint8_region_bytes`、`uint8_valid_bytes`、`align_up`、`fpga_segment_offsets`、`fpga_segment_sizes`、`fpga_transfer_layout`、`describe_fpga_layout`。

| K | tokens offset/size | group index offset/size | source/region/valid offset | total bytes |
|---|---|---|---|---|
| 48 | 0 / 196608 | 196608 / 96 | 196736 / 196800 / 196864 | 196928 |
| 64 | 0 / 262144 | 262144 / 128 | 262272 / 262336 / 262400 | 262464 |
| 96 | 0 / 393216 | 393216 / 192 | 393408 / 393536 / 393664 | 393792 |

所有 offset 均为 64 的倍数，`total_bytes = align_up(最后一段 end, 64)`。

## 2. 测试结果

测试命令（CPU，不加载模型权重）：

```bash
/home/zhaoxiangyu/conda-envs/robobrain25/bin/python -m unittest \
  evaluation.tests.test_hardware_regular_token_selector \
  evaluation.tests.test_token_reduction \
  evaluation.tests.test_shallow_block_pruning \
  evaluation.tests.test_cutpoint_oracle
```

结果：`Ran 42 tests ... OK`（passed=42，failed=0）。原始模型文件未改动。

| 测试模块 | 用例数 | 结果 |
|---|---|---|
| `test_hardware_regular_token_selector` | 18 | 通过 |
| `test_token_reduction` | 8 | 通过 |
| `test_shallow_block_pruning` | 13 | 通过 |
| `test_cutpoint_oracle` | 3 | 通过 |

新测试覆盖（逐条对应任务要求）：

- 多种 grid 与 `K=48/64/96`（另含 `K=32`）下严格 `K`、`K % 16 == 0`、动态 `C <= 9`、真实数 `= K - C`、每个非空区域至少一个 anchor。
- 真实 group 与 context 代表的 `original_group_index` 全局唯一，`source_type` 正确区分 real/context。
- context eligible 区域判定正确：仅含 1 个 group 或空 region 不生成 context；`2x9`（含 3 个空区）与 `5x4`（含 2 个单组区）输出仍严格 `K` 且无零/`-1` padding。
- 每个 context 区域强制保留至少 1 个未选 group（含直接对 `select_real_groups` 的 reservation 检查）。
- context 池化只用该 region 全部未选 group，数值与手工 `pool_region_context` 对照一致。
- context 代表等于该区域未选 group 距加权质心最近者，来自未选集合，patch 索引为 `4g+[0,1,2,3]`。
- `G <= K` identity bypass 逐元素不变（`(4,4)/48`、`(6,6)/48`、`(8,8)/64`），`valid` 全 true。
- 输出索引稳定升序、重复运行完全一致、全同分时确定性地取更小 `original_group_index`。
- 两级 Top-K 只用有界 local candidates：全局第三高分但不在区内 top-2 的 group 不被选中。
- FPGA 布局每段 offset 64 字节对齐、段大小公式正确、总长度正确。
- 维度错误、`K` 非 16 倍数、`lambda_semantic` 越界、投影形状错误、文本维度错误、边界非法、`max_context_slots<=0`、预算不足均明确报错。
- `import` 该模块不会引入 `torch`（子进程断言 `'torch' not in sys.modules`），证明不依赖 CUDA 与模型权重。

## 3. 已知未接入项

- 未接入任何模型或 `run_hf_*` 运行器；本模块是独立纯 CPU 原语，不改变现有推理路径。
- `Wt/Wv` 只是接口；没有提供训练好或随机的正式权重，任何真实分数都需要调用方显式传入权重。
- 未生成二进制传输文件、RTL 或板级传输；FPGA 部分只有布局描述与 offset/size 计算。
- `B0` 只是同接口标签，没有针对 `B0` 的独立打分或验证运行。
- 未做精度、时延、功耗或 ZCU102 实测；本报告不声称任何加速或精度结论。
- 未修改 `CURRENT_STATUS.md`（原因见第 0 节的范围约束）。
