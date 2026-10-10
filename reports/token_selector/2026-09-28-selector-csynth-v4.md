# Selector packed top v4：多遍 max-select 关闭 5 ns 时序门槛

- 日期：2026-09-28
- 决策：`SELECTOR_V4_MULTIPASS_CLOSES_5NS_TIMING_GATE`（status **PASS**）
- 范围：只做 **选择与合并** 的硬件规则化 Token Selector 的 v4 时序重构，并只跑
  g++/Python 对拍与受控 `csynth_design`。
- 未做：GPU、ZCU102、C/RTL cosim、IP 导出、bitstream、驱动。未改变选择数学、
  host 5 段字节布局、128-bit packed context 路径与默认 clock uncertainty。

## 1. 目标

v3 已达成：packed 128-bit token 数据面位精确、latency 有限 worst 12.50 ms、
资源全部 <50%、0 error，但 5 ns 默认 uncertainty 下 estimated period 4.423 ns
（slack -0.773 ns）。本轮唯一目标：在不改选择数学/host 布局/default uncertainty
的前提下，把 estimated period 压到 `<= 3.650 ns`。

## 2. v3 真实 critical path（先提取，后修改）

来源：`/tmp/robobrain_selector_csynth_run_v3/console_v3_fix2.log`，
保存摘要 `artifacts/token_selector/2026-09-28-selector-csynth-v4/v3_critical_path_summary.md`。

```
WARNING: [HLS 200-871] Estimated clock period (4.423 ns) exceeds the target
(target clock period: 5.000 ns, clock uncertainty: 1.350 ns, effective delay budget: 3.650 ns).
WARNING: [HLS 200-1016] The critical path in module 'token_selector_top_packed' consists of the following:
    'getelementptr' operation 16 bit ('gmem_out_group_addr_1', ...cpp:1132->:1217) [73393]  (0.000 ns)
    bus request operation ('empty_110', ...cpp:1132->:1217) on port 'gmem_out_group' (...:1132->:1217) [73394]  (3.650 ns)
    blocking operation 0.773 ns on control path)
```

即 v3 残余关键路径不在 token 取数/`ts_pool_context_packed`（3.650 ns 恰好达标），
而在顶层共享选择逻辑经 `entry` 物化循环写入 `gmem_out_group` 的地址/控制锥。v3 的
`local_top[9][16]`、`pool`、`selected`、`entry_*` 全部 `ARRAY_PARTITION complete`，
稳定插入 `while` 以可变下标访问完全分区寄存器堆，正是该扁平组合锥的来源。

## 3. v4 修改（selection maths 不变）

1. **删除 `local_top[9][16]` 完全分区 bank 与所有 16/144 路插入结构。**
   每个 region 的精确 top-`local_cap` 候选改为**顺序、确定性多遍 max-select**：
   对 `rank = 0..local_cap-1` 扫描该 region 的 row/col band 一次，跳过 anchor 与
   已选组，按“score 高优先、tie 低 index”取最优，直接追加进有界 `pool`；
   候选集合与 v3 稳定插入**完全相同**（同一全序，集合与顺序一致）。
2. **全局 details 与排序改为单比较器多遍 select。** `<=144` pool、`<=96` selected
   （按 index 升序）与 `entry`（按 `(representative, source_type, region_id)`）
   都用多遍 select + 标量 best 寄存器；下标只来自循环计数或标量，**无数据相关读口**，
   无 144/96 路组合 mux。
3. **centroid representative 显式窄化到 int32。** 利用已证明
   `|delta| <= 675721074`（`distance <= 9.13e17 < 2^63-1`），`total/row_sum/col_sum`
   与 delta 均显式 int32；两个平方用 32x32->64 乘法，
   `#pragma HLS BIND_OP variable=<sq> op=mul impl=dsp latency=2` 指定 DSP 且至少
   2-cycle latency（流水寄存）。平方欧氏距离与 tie 规则位精确不变。
4. **128-bit packed context 路径不动。** `ts_pool_context_packed` 与其 16-lane
   并行 MAC、round-half-away、256-word real 复制逐行不变；v3 该模块已达 3.650 ns。
5. **v3 int8 内核保留为测试参考。** 冻结的 v3 bounded-insertion
   `token_selector_core`/`token_selector_top` 用 `#ifndef __SYNTHESIS__` 包住，
   只由 g++/ASan testbench 调用，**永不综合**。

源代码差异：`artifacts/token_selector/2026-09-28-selector-csynth-v4/core_selection_diff.txt`。
golden `evaluation/hardware_regular_selector_fixedpoint.py` SHA256 仍为
`23e8d825…d489e71`（未改）。

## 4. 测试

| 项 | 命令 | 结果 |
|---|---|---|
| g++ 普通 | `make -C fpga_accel/token_selector_hls test` | `passed=475 failed=0` |
| g++ ASan+UBSan | `make -C fpga_accel/token_selector_hls asan_run` | `passed=475 failed=0` |
| Python | 4 个模块 | `Ran 76 tests OK` |
| fixtures | 184（34 固定 + **150 随机**） | int8 v3 top 与 packed v4 top 逐字节一致 |
| **随机差分 vs 旧 v3 insertion** | testbench 内 220 例（`randdiff_v3_000..219`） | `passed=220 failed=0`，逐字节 |
| **确定性边界差分 vs 旧 v3 insertion** | testbench 内 48 例（`bounddiff_v4_00..47`） | `passed=48 failed=0`，逐字节 |
| **Python v3 insertion 差分** | `test_v4_multipass_matches_v3_insertion_200plus` | 220 例对 golden 全过 |

475 = 207（184 fixture + 布局 3 + guard 18 + pack/unpack 2） + 220 随机差分 + 48 边界差分。
220 随机差分覆盖 30 种 grid、预算 16/32/48/64/80/96、`local_cap` 1..16，分数由
`score_mod` 量化（上界 32767/score_mod、按 score_mod 取整；score_mod=1 即未量化的
合同范围，较大值只是提高 tie 频率，并非显式全 tie 模式）与 identity bypass。
48 确定性边界差分显式覆盖大网格 32x128/64x64/128x32/31x127/127x31 与
16x16/24x24/32x32，全部预算 K=16..96，`local_cap` 1/16，以及四种 tie/饱和模式
（全相等、0/65535 交替、重复档位、随机饱和值），并逐例释放输入/输出内存；
其中 **0/65535 交替只是 `uint16` 原始载荷鲁棒性压力测试，不是合同内的量化输入**
（正式 `score_q` 合同是 unsigned Q0.15，合法上限 32767；见
`fpga_accel/token_selector_hls/quantization_contract.md` §2），该用例保留不删除；
普通与 ASan/UBSan 两遍均逐字节通过，无 sanitizer 报错。

## 5. 受控 csynth（Vitis HLS 2023.2）

- 脚本：`scripts/run_csynth_v4.tcl`；project `/tmp/robobrain_token_selector_hls_csynth_v4`；
  top `token_selector_top_packed`；part `xczu9eg-ffvb1156-2-e`；`create_clock -period 5.0`；
  **只 `csynth_design`**（无 cosim、无 IP、无 bitstream、未访问板卡）。
- 修复轮次：**1 轮**（首轮即通过，无需第二轮）。
- 结果：
  - **0 error**；未改 part / 5 ns / 默认 1.35 ns uncertainty。
  - **timing 通过**：estimated period **3.650 ns** = effective budget 3.650 ns，
    slack **0.000 ns**（Fmax 273.97 MHz）；**不再有** `HLS 200-871` / `200-1016`。
  - **latency 有限**：best 2 / average 47,782 / **worst 1,708,124 cycles
    = 8.541 ms @200 MHz < 33.3 ms**。
  - **资源 <50%**：BRAM 138/1824 = 7.57%、DSP 87/2520 = 3.45%、
    FF 41,441/548,160 = 7.56%、LUT 44,966/274,080 = 16.41%、URAM 0。
  - `HLS 200-790` 仍报 loop constraints NOT satisfied（可变界内层扫描未达 target II），
    但这不是 correctness/timing 门槛：0 error、timing met、资源与 latency 均过。
- 门控判定：**PASS**（estimated period、latency、资源、0 error 四项全过）。

## 6. 与 v3 对比

| 项 | v3（8-bit/packed insertion） | v4（packed multi-pass） |
|---|---|---|
| estimated period | 4.423 ns | **3.650 ns** |
| timing slack | -0.773 ns | **0.000 ns** |
| worst latency | 2,499,989 cycles / 12.50 ms | 1,708,124 cycles / **8.541 ms** |
| BRAM | 130 (7.13%) | 138 (7.57%) |
| DSP | 67 (2.66%) | 87 (3.45%) |
| FF | 185,662 (33.87%) | **41,441 (7.56%)** |
| LUT | 91,752 (33.48%) | **44,966 (16.41%)** |
| errors | 0 | 0 |

FF 从 33.87% 降到 7.56%、LUT 33.48% 降到 16.41%，正是移除完全分区插入 bank/mux
的直接结果；DSP 略升 67→87 来自 centroid 平方显式绑定 DSP。packed context 模块的
worst 仍为 140,662 cycles，未被改动。

## 7. 结论与精确下一步

- 完成：`MULTIPASS_SELECTION_EQUIVALENT / CSIM_ASAN_PYTHON_PASS /
  CSYNTH_TIMING_LATENCY_RESOURCES_PASS`。
- 门槛：`estimated period 3.650 ns <= 3.650 ns`；latency worst 8.541 ms < 33.3 ms；
  各资源 <50%；0 error —— 全部满足，status **PASS**。
- 精确下一步（本轮未做）：只有在通过后续独立的板卡/授权门后，才写驱动、
  做 C/RTL cosim 与导出 IP；本轮不 cosim、不导出、不上板。

## 8. 复现命令

```bash
make -C fpga_accel/token_selector_hls test
make -C fpga_accel/token_selector_hls asan_run
/home/zhaoxiangyu/conda-envs/robobrain25/bin/python -m unittest \
  evaluation.tests.test_hardware_regular_selector_fixedpoint \
  evaluation.tests.test_hardware_regular_token_selector \
  evaluation.tests.test_hardware_regular_b1_integration \
  evaluation.tests.test_token_selector_packed_contract

mkdir -p /tmp/robobrain_selector_csynth_run_v4 && cd /tmp/robobrain_selector_csynth_run_v4
export LD_LIBRARY_PATH=/home/xilinx/xic/lib/lnx64.o/SuSE:$LD_LIBRARY_PATH
/home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f \
  /home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/scripts/run_csynth_v4.tcl
```
