# 定点 Selector v2：有界重构、等价性回归与受控 csynth（门控 REDESIGN_REQUIRED）

- 日期：2026-09-27
- 状态：`SELECTOR_V2_REFACTOR_EQUIVALENT / FULL_REGRESSION_PASS / CSYNTH_LATENCY_AND_TIMING_GATE_FAIL / REDESIGN_REQUIRED`
- 工作区：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`
- 范围：按审校方案把定点 HLS Selector 改成**编译期有界**实现，保持选择数学与
  host 5 段布局不变；跑 Python/C++/fixtures 普通与 ASan 回归，加 150 组随机
  对拍；在 `/tmp` 用相同 ZCU102 5 ns 设置只跑 `csynth_design`。**不**运行
  GPU/板卡/cosim/IP 导出/bitstream，**不**改 clock uncertainty。

## 1. 一句话结论

重构后的选择核与冻结 Python golden 在所有 fixtures 与 150 组随机 case 上逐字节
一致（Python 66/66，g++ 普通与 ASan/UBSan 均 `passed=205 failed=0`），
`csynth_design` 0 error、资源任一类 <50%、顶层 latency **已有限**；但
worst latency = 20,561,009 cycles（200 MHz 下 102.8 ms > 33.3 ms），5 ns 在默认
1.35 ns uncertainty 下 slack `-0.773 ns`，因此门控**未全满足**，如实判
`REDESIGN_REQUIRED`，**未写 PASS**。

## 2. 审校方案对应的 8 项改动

1. **固化 `TS_MAX_BUDGET=96`**：Python `MAX_BUDGET`、C++ `TS_MAX_BUDGET` 同值；
   运行时仍接受 16 的正倍数且 `<=96`（保留 16/32/48/64/80/96 边界 fixtures），
   生产仅 48/64/96；`K=112/128` 返回 `ERR_BUDGET`。所有 output/selected/entry
   循环按 96 定长。
2. **区内不再收集最多 4096 项后全排序**：扫描 region 成员时用稳定插入只维护
   精确 `top-local_cap`（`local_cap<=16`），结果与旧算法逐项相同。候选池最大
   `9*16=144`；全局用固定有界插入排序选出 `details_needed<=budget<=96`
   （实际可行解通常 <=87），彻底去掉对 4096 组的 O(G²) 全排序。
3. **anchor 扫描缓存 `anchor_score`**，不再每次动态重读 `score_q[best]`；
   所有动态 G 循环加真实 `#pragma HLS LOOP_TRIPCOUNT max=4096`。
4. **输出 `m_axi` 拆分**：`gmem_out_tokens` / `gmem_out_group` /
   `gmem_out_source` / `gmem_out_region` / `gmem_out_valid` 各自独立 bundle，
   输入也拆成 `gmem_in_tokens` / `gmem_in_score`；**host 5 段字节布局未变**。
5. **context 池化位精确不变**：region/selected 改为流式扫描（去掉
   `region_members[9][4096]`），权重、int64 累加、round-half-away、代表点交叉
   乘法与同距低 index 规则全部不变。
6. **普通与 ASan 全回归**：见 §4。
7. **受控 csynth**：见 §5、§6。
8. **文档**：本报告、`artifacts/token_selector/2026-09-27-selector-csynth-v2/summary.json`、
   `fpga_accel/token_selector_hls/README.md`、`register_map_draft.md`、
   `artifacts/token_selector/2026-09-27-fixedpoint-selector-spec.json`、
   `CURRENT_STATUS.md`。

## 3. 与原算法逐项相同的证明路径

- **选择数学冻结**：区域划分、anchor（最高分/同分低 index）、`C` 动态、
  `real=K-C`、区内有界候选、全局 Top-K、代表全局唯一升序、context 权重/舍入/
  代表点规则均未改。
- **区内候选等价**：原实现把非 anchor 成员全排序后取前
  `take=min(local_count, local_cap, allowed)`；新实现维护 `limit=min(local_cap,
  allowed)` 的稳定插入 top-K。排序键 `(score desc, index asc)` 是全序（index
  唯一），两者得到同一集合与同一顺序。
- **全局选择等价**：候选池按 region 顺序拼接后做有界插入排序（<=144），再取
  前 `details_needed`；与旧“候选集合全排序取前 K”一致。
- **context 等价**：`remaining` 流式扫描得到与旧 `region_members` 相同的升序
  集合；整数加法可交换，池化/代表点结果不变。
- **独立参考实现对拍**：`evaluation/tests/test_hardware_regular_selector_fixedpoint.py`
  内 `_legacy_select` 是“收集全部成员 + 全排序”的独立实现，
  `test_random_differential_matches_legacy_reference` 用 120 组确定性随机 case
  对其逐项（status/counts/tokens_out/group_index/source_type/region_id/valid）
  对拍 golden，全部通过。
- **C++ 对 golden**：fixtures 由 golden 生成，C testbench 对每个输出字节比对，
  184 个 case（34 固定 + 150 随机）普通与 ASan/UBSan 均 `passed=205 failed=0`。

## 4. 测试与 fixtures

| 项 | 命令 | 结果 |
|---|---|---|
| Python | `python -m unittest evaluation.tests.test_hardware_regular_selector_fixedpoint evaluation.tests.test_hardware_regular_token_selector evaluation.tests.test_hardware_regular_b1_integration` | `Ran 66 tests ... OK` |
| 随机对拍 | 同上（新增 `test_random_differential_matches_legacy_reference`） | 120 组逐项一致 |
| C++ 普通 | `make -C fpga_accel/token_selector_hls test` | `passed=205 failed=0` |
| C++ ASan/UBSan | `make -C fpga_accel/token_selector_hls asan_run` | `passed=205 failed=0` |
| fixtures | `tools/generate_fixtures.py` | 184 case（34 固定 + 150 随机） |

fixtures 目录：`artifacts/token_selector/2026-09-27-fixedpoint-selector-fixtures/`。

## 5. 受控 csynth（v2）

- 命令（中间文件全在 `/tmp`，运行前记录 `df`）：
  ```bash
  mkdir -p /tmp/robobrain_selector_csynth_run_v2
  cd /tmp/robobrain_selector_csynth_run_v2
  export LD_LIBRARY_PATH=/home/xilinx/xic/lib/lnx64.o/SuSE
  /home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f \
    /home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/scripts/run_csynth_v2.tcl
  ```
- `open_project /tmp/robobrain_token_selector_hls_csynth_v2`、top
  `token_selector_top`、part `xczu9eg-ffvb1156-2-e`、clock 5.0 ns，只
  `csynth_design`；0 error。
- 两轮修复：
  - 第 1 轮：加编译期上界（预算/`local_cap`/候选池/区域成员）、所有动态循环
    `LOOP_TRIPCOUNT`、把 materialise 拆成“real 拷贝（<=96）”与
    “context 池化（<=9）”，输出 m_axi 拆 bundle。顶层 latency 变有限，但 worst
    仍 87,228,752 cycles。
  - 第 2 轮：context 池化改为循环分区 int64 累加器 + 8 路连续 MAC。worst 降到
    20,561,009 cycles，但 `gmem_in_tokens` 8-bit 端口读带宽使内层 II=8。

## 6. 客观门槛逐项

| 门槛 | 结果 | 判定 |
|---|---|---|
| top latency 有限 | best 2 / avg 592,391 / worst 20,561,009 cycles | 通过 |
| 200 MHz 下 worst < 33.3 ms | 20,561,009 / 200e6 = **102.81 ms** | **失败** |
| 资源任一类 < 50% | BRAM 40(2.19%) / DSP 73(2.90%) / FF 184171(33.60%) / LUT 89111(32.51%) / URAM 0 | 通过 |
| 无 unsynthesizable/error | 0 error，综合完成 | 通过 |
| 5 ns（默认 uncertainty 1.35 ns） | 估计 4.423 ns，可用 3.650 ns，slack **-0.773 ns** | **失败** |

- 未修改 clock uncertainty；失败来自真实估计周期 4.423 ns，如实记录。
- 主瓶颈证据：`VITIS_LOOP_610_28`（context 池化调用，trip 9）=
  18,996,156 cycles；其内层 `VITIS_LOOP_184_2_VITIS_LOOP_188_3`（trip
  262,144）Final II=8，HLS 200-880 报 `gmem_in_tokens` 读操作 carried
  dependence。即池化受 8-bit token 端口 1 byte/cycle 限制。
- 次要项：`VITIS_LOOP_441_10`（区内稳定插入，4096 次）827,392 cycles，
  II=11/12；`VITIS_LOOP_380_6`/`VITIS_LOOP_598_27` 各约 395k cycles（96 组
  4096-byte 拷贝）；均为有界但未继续优化。

## 7. 验证哈希

源码 SHA256（v2，供复现）：

```text
23e8d825d7966719774726c8debcdf3f8c6c33bab49a3fdcc231d7ec8d489e71  evaluation/hardware_regular_selector_fixedpoint.py
c9160a41d57b7a3c4444c92973e25f804481b624f4f0c098dfcfee1783ca6d85  evaluation/tests/test_hardware_regular_selector_fixedpoint.py
671e93af3347d65c3af4fc1e25483031fabb7ae46ea89f6f16a3f7110dc17347  fpga_accel/token_selector_hls/include/token_selector_hls.h
308d5f323f43b81d965aab1d270c2fbbf02250108acfed486da8c771b2474737  fpga_accel/token_selector_hls/src/token_selector_hls.cpp
94973958686576037cc6b8cec58ed7e8bc0410b215fa32353b260ca4faded5ee  fpga_accel/token_selector_hls/tb/tb_token_selector_hls.cpp
9019d38923bbf7b56d8c7f89eac32100807389deee2ccded66068536aeef98b1  fpga_accel/token_selector_hls/tools/generate_fixtures.py
f797859333cfa4853504360749eeb5f59e7ba6c78189051aedc6314a1c190768  scripts/run_csynth_v2.tcl
```

## 8. 已知限制 / 未做

- 只做选择与整数池化，不打分、不训练、不加载模型/权重、不访问 GPU/ZCU102。
- 只 `csynth_design`：未 cosim、未生成 RTL/IP/bitstream、无板级时序/资源/吞吐。
- 未改选择数学、未改 host 5 段布局、未改 clock uncertainty。
- `int64` 溢出证明仍以 `MAX_GRID_DIM=128` 为前提，见
  `artifacts/token_selector/2026-09-27-selector-overflow-proof.json`（其
  `source_sha256` 记录的是 v2 之前的源；v2 源哈希见 §7）。

## 9. 精确下一步（REDESIGN_REQUIRED，未执行）

在**不改 host 5 段布局、不改选择数学、不改默认 uncertainty** 的前提下：
把 token 取数通路加宽（更宽的内部 token 端口 / 预取 staging buffer / 多读口）
并/或把 int64 MAC 用带寄存的 DSP 链流水化，目标是把 context 池化的
1 byte/cycle 提到 ≥3 byte/cycle，使 worst < 6.66e6 cycles，同时把估计周期压到
≤3.65 ns；然后重跑同一条 `csynth_design`。在此之前不写驱动、不做 cosim、不导出
IP、不上板。
