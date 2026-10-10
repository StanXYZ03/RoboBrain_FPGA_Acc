# Selector 128-bit packed token 数据面 + 受控 csynth v3

- 日期：2026-09-27
- 决策：`SELECTOR_V3_PACKED_128BIT_EQUIVALENT_LATENCY_AND_RESOURCES_PASS_TIMING_GATE_FAIL`（status `REDESIGN_REQUIRED`）
- 范围：只做 **选择与合并** 的硬件规则化 Token Selector；把 token 数据面从 8-bit 改为
  **128-bit packed AXI word**，并只跑 g++/Python 对拍与受控 `csynth_design`。
- 未做：GPU、ZCU102、C/RTL cosim、IP 导出、bitstream。未改变选择数学与 host 5 段字节布局。

## 1. 目标与约束

要求（本轮）：

1. 保留现有 `int8` core/golden 接口用于软件回归；新增综合 top 使 tokens 输入/输出为
   128-bit word（16 个连续 int8，小端 lane），G 个 group 每组 4096 B = 256 words，
   地址严格等价原 `tokens_int8` 连续段；metadata 端口不变。
2. context 池化按 region 只读取其未选 group，每个 128-bit word 一次取 16 lane，
   16 路 `int8 * uint16` 并行 `int64` 累加，round-half-away 与全零权重均匀不变；
   real group 按 256 word 原样复制；**不得缓存全部 G tokens**。
3. ZCU102 HPC 现实宽度按 128-bit 记录；host 64 B 对齐与总字节数不变；新增 pack/unpack
   测试，任意 byte offset/lane 保持一致。
4. 普通、ASan、Python 回归与 ≥150 随机 fixtures 逐字节对拍全部通过。
5. `/tmp` 用新 `scripts/run_csynth_v3.tcl`，top 为 packed top，part 与 5 ns/default
   uncertainty 不变，只 `csynth_design`；最多两轮修复。

## 2. 128-bit packed 契约

- `ts_word_t`：综合时 `ap_uint<128>`；g++ 时 `unsigned __int128`，little-endian host 上
  16 字节内存布局与 `ap_uint<128>` 的 AXI lane 完全一致。
- lane：`lane L` 占 bits `[8L+7:8L]`，即 `word.range(8L+7,8L)`；`lane = byte % 16`。
- 分组：`TS_GROUP_WORDS = 4096/16 = 256`；`word(group,w)` ↔ 原字节
  `group*4096 + w*16`（严格等价，无偏移）。
- host 5 段布局、64 B 对齐、段字节数与总字节数**完全不变**：token 段 K=48/64/96 分别为
  196608/262144/393216 B = 12288/16384/24576 words。
- 综合端口：`gmem_in_tokens` / `gmem_out_tokens` 数据宽度 128 bit（报告
  `gmem_out_tokens_WDATA` = 128），metadata 端口与全部 `s_axilite` 标量不变。
- 访问器：C++ `ts_word_lane` / `ts_word_set_lane` / `ts_pack_int8_to_words` /
  `ts_unpack_words_to_int8`；Python `evaluation/token_selector_packed_contract.py`。

## 3. 算法等价性

- `token_selector_core_packed` 与冻结的 `token_selector_core` 逐行比对：差异**仅**为注释、
  identity/real 的 256-word 复制循环、以及 context 池化调用；所有 anchor / 两级 Top-K /
  稳定插入排序 / 代表点 / 元数据排序代码逐行相同。差异证据：
  `artifacts/token_selector/2026-09-27-selector-csynth-v3/core_selection_diff.txt`。
- context 池化只改取数/循环结构：每个 word 的 16 个 lane 并行做
  `weight * lane`（32 位精确乘，`|product| <= 32767*128 < 2^23`）再符号扩展进 `int64`；
  整数加法的可交换/可结合性保证结果与逐元素顺序完全一致。round-half-away 与全零均匀权重
  不变；real group 256 word 原样复制；**无** per-group staging、**无**全 G token 缓存
  （只有 1 个 word 寄存器 + `int64[4096]` 累加器）。
- Python golden `evaluation/hardware_regular_selector_fixedpoint.py` 未改动
  （SHA256 `23e8d825…d489e71`）。

## 4. 测试

| 项 | 命令 | 结果 |
|---|---|---|
| g++ 普通 | `make -C fpga_accel/token_selector_hls test` | `passed=207 failed=0` |
| g++ ASan+UBSan | `make -C fpga_accel/token_selector_hls asan_run` | `passed=207 failed=0` |
| Python | 4 个模块（含新 `test_token_selector_packed_contract`） | `Ran 75 tests OK` |
| fixtures | 184（34 固定 + **150 随机**） | int8 top 与 packed top 均逐字节对拍 golden |
| pack/unpack | `pack_lane_offsets`、`pack_byte_offset_tail` | 任意 lane 轮转（0..15）与非 16 倍数尾部/非零 word offset 一致 |

每个 fixture 同时运行 int8 top 与 packed top：status/counts/scale、tokens、group_index、
source_type、region_id、valid 全部逐字节一致；随机 case `rand000`…`rand149`。

Python 新增测试（`evaluation/tests/test_token_selector_packed_contract.py`，9 项）：
pack/unpack 任意 byte offset 往返、前缀/尾部零填充、小端 lane 序、`byte_to_word_lane`、
`group_word_address` 与字节地址等价、token 段 bytes/words 不变、word 视图与连续字节重解释一致、
word 池化 = int8 golden、全零权重均匀。

## 5. 受控 csynth（Vitis HLS 2023.2）

- 脚本：`scripts/run_csynth_v3.tcl`；project `/tmp/robobrain_token_selector_hls_csynth_v3`；
  top `token_selector_top_packed`；part `xczu9eg-ffvb1156-2-e`；`create_clock -period 5.0`；
  **只 `csynth_design`**（无 cosim、无 IP、无 bitstream、未访问板卡）。所有中间文件留在 `/tmp`，
  仅小报告复制回 `artifacts/token_selector/2026-09-27-selector-csynth-v3/`。
- 修复轮次：**2 轮**。
  - 第 1 轮：HLS 214-274 拒绝 pipelined loop 内的 `#pragma HLS UNROLL`；改为显式手写 16 lane。
  - 第 2 轮：unrolled rounding loop 被展开成约 4096 个 divider/packer 实例（FF 520%、LUT 824%）；
    改为「4096 元素单次共享除法 round + 16-lane 装配」，并把代表点质心乘法收窄为 32 位操作数。
- 结果：
  - 0 error；未改 part / 5 ns / 默认 1.35 ns uncertainty。
  - **latency 有限**：best 2 / average 124880 / **worst 2499989** cycles = **12.50 ms @200 MHz
    < 33.3 ms**（通过）。
  - **资源 <50%**：BRAM 130/1824 = 7.13%、DSP 67/2520 = 2.66%、FF 185662/548160 = 33.87%、
    LUT 91752/274080 = 33.48%、URAM 0（通过）。
  - **timing 未通过**：estimated 4.423 ns > effective budget 3.650 ns，slack **-0.773 ns**。
    除 `ts_pool_context_packed` 子模块（3.650 ns，刚好达标）外，其余 csynth 子模块均达标；
    残余关键路径在**共享选择逻辑**的 top-level 平面逻辑内，两轮允许的修复内未能解决。
    uncertainty 未修改，未掩盖。
- 门控判定：`REDESIGN_REQUIRED`（latency 与资源通过，5 ns 时序未过）。

## 6. 与 v2 的对比

| 项 | v2（8-bit） | v3（128-bit packed） |
|---|---|---|
| worst latency | 20,561,009 cycles / 102.81 ms | **2,499,989 cycles / 12.50 ms** |
| BRAM | 40 (2.19%) | 130 (7.13%) |
| DSP | 73 (2.90%) | 67 (2.66%) |
| FF | 184,171 (33.60%) | 185,662 (33.87%) |
| LUT | 89,111 (32.51%) | 91,752 (33.48%) |
| estimated period | 4.423 ns | 4.423 ns |
| timing slack | -0.773 ns | -0.773 ns |

v2 的 8-bit `gmem_in_tokens` 读带宽（1 byte/cycle，II=8）导致 context 池化 18,996,156 cycles；
v3 用 128-bit port 直接把该瓶颈消除，worst latency 降到 12.5 ms（约 8.2×）。timing 数值与 v2
相同，说明残余关键路径不在 token 取数/池化路径，而在两版共享的选择逻辑。

## 7. 结论与精确下一步

- 完成：`PACKED_128BIT_TOKEN_CONTRACT_EQUIVALENT / CSIM_AND_PYTHON_REGRESSION_PASS /
  CSYNTH_LATENCY_AND_RESOURCES_PASS / CSYNTH_TIMING_GATE_FAIL`。
- 未通过项：默认 uncertainty 下 5 ns 时序（estimated 4.423 ns，slack -0.773 ns）。
- 精确下一步（未执行）：在不改选择数学 / host 5 段布局 / 默认 uncertainty 前提下，定位并
  打拍/流水化共享选择逻辑中的关键路径（重点：完全分区的 `local_top[9][16]` 插入 mux 与
  代表点/质心逻辑），使 estimated period `<= 3.65 ns`，然后重跑同一条 `csynth_design`。
- 在此之前不写驱动、不 cosim、不导出 IP、不上板。

## 8. 复现命令

```bash
make -C fpga_accel/token_selector_hls test
make -C fpga_accel/token_selector_hls asan_run
/home/zhaoxiangyu/conda-envs/robobrain25/bin/python -m unittest \
  evaluation.tests.test_hardware_regular_selector_fixedpoint \
  evaluation.tests.test_hardware_regular_token_selector \
  evaluation.tests.test_hardware_regular_b1_integration \
  evaluation.tests.test_token_selector_packed_contract

mkdir -p /tmp/robobrain_selector_csynth_run_v3 && cd /tmp/robobrain_selector_csynth_run_v3
export LD_LIBRARY_PATH=/home/xilinx/xic/lib/lnx64.o/SuSE:$LD_LIBRARY_PATH
/home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f \
  /home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/scripts/run_csynth_v3.tcl
```
