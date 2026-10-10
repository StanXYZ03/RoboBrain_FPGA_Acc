# Selector INT8 / Q0.15 量化契约

> 状态：`CONTRACT_DEFINED / NO_MODEL / NO_CALIBRATION`。
> 本文件只定义 Selector 边界上的数值语义，不产生真实权重或真实分布。

## 1. 输入 token 段

- 形状 `int8[G, 4, 1024]`，每个元素是完整 2x2 merge group 的 4 个 patch 位置
  之一。
- 全图共享一个**对称** scale `s > 0`，zero-point 固定为 0：
  `x_real ≈ q / s`，其中 `q = clamp(round(x_real * s), -128, 127)`。
- 对称量化没有 zero-point，因此同一 scale 下池化仍然是保序的凸组合。
- `scale` 只通过 AXI-Lite 透传（`scale_in → scale_out`），**不参与**选择、
  Top-K、或 context 池化的任何比较与除法。

### 1.1 128-bit packed token 传输（2026-09-27 v3）

- token 数据面按 **128-bit word** 传输：word `L` lane 是 16 个连续 int8 的
  小端打包，lane `L` 占 bits `[8L+7:8L]`。
- 一个 group 4096 B = **256 words**；`(group, w)` word 对应原
  `tokens_int8` 连续段字节 `group*4096 + w*16`。地址严格等价。
- 综合用 `ap_uint<128>`（ZCU102 HPC 现实数据宽度按 128-bit 记录）；
  g++ 非综合用 `unsigned __int128`，little-endian host 上字节布局与
  `ap_uint<128>` 的 AXI 小端 lane 完全一致。
- host 5 段布局、64 B 对齐、段字节数与总字节数**完全不变**；token 段只是
  `size_bytes/16` 个 word（K=48 时 12288 words）。
- context 池化仍按 region 只读取未选 group，每个 word 一次取 16 lane、
  16 路 `int8 * uint16` 并行 `int64` 累加；权重、round-half-away、全零均匀
  与 §3 完全一致。**不缓存全部 G tokens**（只用一个 word 寄存器 + 4096 项
  `int64` 累加器）。

## 2. 输入分数段

- `score_q` 为 `uint16`，Q0.15：`score = score_q / 32768`。
- 有效范围 `[0, 32767]`（即 `score ∈ [0, 0.99997...]`），由 CPU 侧
  `combine_scores` 的归一化组合分量化得到。
- 选择、anchor、两级 Top-K 全部直接比较 `score_q` 整数值；同分取更小
  `original_group_index`。因为 Q0.15 是同一个正缩放，整数比较与浮点比较
  的**排序完全一致**。

## 3. context 池化（在量化域内完成）

对某区域的全部未选 group `remaining`：

1. 权重 `w_i = score_q[group_i]`（非负整数）；若 `Σ w_i == 0`，令
   `w_i = 1`（均匀）。
2. 每个 patch/feature 累加 `num = Σ_i w_i * tokens_q[group_i][p][f]`，使用
   `int64`。packed 数据面下每个 128-bit word 的 16 个 lane 并行累加：
   `weight * lane` 先按 32 位精确乘（`|product| <= 32767*128 < 2^23`），再
   符号扩展进 `int64` 累加器；整数加法的可交换/可结合性保证结果与逐元素
   顺序完全一致。
3. `pooled = round_half_away_from_zero(num, Σ w_i)`，再写回 `int8`。

由于 `tokens_q` 是带共享 scale `s` 的对称量化，且权重非负、和为 1，
`pooled / s` 就是实域加权平均的量化近似；整数池化与实域池化的差仅为
round-half-away-from-zero 的一个量化步长。**不需要**额外的 scale/zero-point
段：池化权重是归一化的，`s` 在分子分母中相消。

## 4. context 代表点

代表点是距该区域未选 group 加权质心最近的**未选** group：

- 质心 `c = (Σ w_i * r_i / W, Σ w_i * c_i / W)`，`W = Σ w_i`。
- 为避免除法与浮点，比较 `((r_i*W - Σ w_j r_j)^2 + (c_i*W - Σ w_j c_j)^2)`，
  **全部用 64 位整数**（Python 任意精度只作为对拍 golden），同距取低 index。
- `int64` 安全性由 `MAX_GRID_DIM = 128` 保证：单个 3x3 区域成员数
  `n <= 491`、`W <= 491*32767 = 16088597`、
  `|delta| <= W*42 = 675721074`，因此
  `distance <= 2*675721074^2 = 9.13e17 < 2^63-1`；穷举最大值为
  `4.4776e17`（grid `31x127`）。池化累加 `|A| <= 491*32767*128 = 2.06e9`。
  证明与源码哈希见 `artifacts/token_selector/2026-09-27-selector-overflow-proof.json`。
- 早期版本 C++ 曾用 `__int128`；现已全部移除，`fpga_accel/token_selector_hls/src/token_selector_hls.cpp`
  只保留 `int64_t`/`uint64_t`，以匹配可综合路径。
- 代表 index 只作为元数据写出，不代表对 token 做重采样。

## 5. 输出段

- `tokens_int8`：真实组直接复制原 4x1024 token（packed 下 256 word 原样复制）；
  context 组写入 §3 的池化结果。字节内容与 int8 版本逐字节一致。
- `group_index_uint16`：代表 `original_group_index`，全局唯一、升序，因此要求
  `G <= 65535`（实际契约 `G <= 4096`，且 `1 <= grid_h, grid_w <= 128`；单维
  超过 128 直接返回 `ERR_GRID`，这是 `int64` 质心证明的前提）。
- `source_type_uint8`：`0=real`、`1=context`；硬件必须读它区分动态 context。
- `region_id_uint8`：`[0, 9)`。
- `valid_uint8`：恒为 1，不产生零/-1 padding。

## 6. 未定义 / 未验证

- 未提供真实 `s`、`Wt/Wv` 或真实 `score_q`；本契约不含标定结果。
- 未做 INT4、per-group scale 或非对称量化；这些不在本版本范围内。
- 未做量化误差对选择精度影响的端到端评估。
