# Selector v4 测试缺口修复 + 独立 C/RTL cosim（Vitis HLS 2023.2）

- 日期：2026-09-28
- 决策：`SELECTOR_V4_BOUNDARY_DIFFERENTIAL_48_PASS / SELECTOR_V4_CSIM_3_PASS / SELECTOR_V4_COSIM_3_PASS_FUNCTIONAL_ONLY`
- 范围：只做选择与合并的 packed v4 Token Selector 的**测试缺口修复**与
  **C simulation + C/RTL co-simulation**。
- 未做：GPU、ZCU102、IP 导出、bitstream、驱动。未改变选择数学、量化、host 5 段
  字节布局、clock uncertainty，也未改 `token_selector_hls.cpp` / `.h`（哈希不变）。
- 时序边界：v4 csynth 是 **pre-route 0 ns 估算裕量**；cosim **只验证功能**，
  **不证明 post-route 200 MHz**。

## 1. 测试缺口修复（普通 + ASan/UBSan）

保留原有 220 组内存内随机差分（`randdiff_v3_000..219`），新增 **48 组确定性边界
差分**（`bounddiff_v4_00..47`），每例都用新 packed v4
（`token_selector_top_packed`）对冻结 v3 bounded-insertion int8
（`token_selector_top`，仅 `#ifndef __SYNTHESIS__` 参考）逐字节比较，
并在每例结束时释放 tokens/score/输出/解包向量。

边界差分覆盖：

- 有效大网格：32x128、64x64、128x32、31x127、127x31，以及典型 16x16、24x24、32x32。
- 全部预算：K = 16/32/48/64/80/96。
- `local_cap` = 1 与 16。
- 四种 score 模式：全相等；0/65535 交替（仅 uint16 原始载荷鲁棒性压力测试）；重复档位
  `{0,10922,21844,32766}`；随机饱和值（约一半取 Q0.15 合同上限 32767）。
- **score_q 合同澄清**：正式合同是 unsigned Q0.15，合法量化上限为 **32767**
  （`score = score_q / 32768 ∈ [0, 0.99997...]`，见
  `fpga_accel/token_selector_hls/quantization_contract.md` §2）。边界差分里的
  `0/65535` 交替**只是对 `uint16` 原始载荷的鲁棒性压力测试，不是合同内的量化输入**；
  该用例保留、不删除。
- 0/65535 交替时每个区域至多一半权重非零，仍落在已证明的 centroid 界内，ASan/UBSan
  无溢出。
- 关于旧 220 随机差分：分数按 `score_mod` 量化（上界 32767/score_mod），
  `score_mod=1` 即未量化合同范围；较大 `score_mod` 只是**提高 tie 频率**，
  并非“强制大量 tie”，本次报告与代码注释均已按此更正。

结果（2026-09-28）：

| 项 | 命令 | 结果 |
|---|---|---|
| g++ 普通 | `make -C fpga_accel/token_selector_hls test` | `passed=475 failed=0` |
| g++ ASan+UBSan | `make -C fpga_accel/token_selector_hls asan_run` | `passed=475 failed=0`，无 sanitizer 报错 |
| Python 回归 | 4 个模块 | `Ran 76 tests OK` |
| 随机差分 | testbench 内 220 例 | 逐字节一致 |
| 边界差分 | testbench 内 48 例 | 逐字节一致 |

475 = 207（184 fixtures + 3 layout + 18 guard + 2 pack/unpack） + 220 + 48。
日志：`artifacts/token_selector/2026-09-28-selector-cosim-v4/{tb_v4_normal.log,tb_v4_asan.log,py_v4_regression.log,boundary_differential_pass.txt}`。

## 2. 独立 cosim 专用 testbench 与 3 个代表 case

新增 `fpga_accel/token_selector_hls/tb/tb_token_selector_cosim_v4.cpp`，**独立于
184 fixtures + 220 随机的 g++ 差分**，只回放 3 个代表 case。输入与 C golden 由
`tools/generate_cosim_v4_fixtures.py` 用冻结参考
`evaluation/hardware_regular_selector_fixedpoint.py` 生成并固化为
`TSIN`/`TSEX` 文件。

| case | 网格 | G | K | local_cap | 说明 | golden out/real/context |
|---|---|---|---|---|---|---|
| `cosim_identity_4x4_K48` | 4x4 | 16 | 48 | 16 | identity 小例（无 context） | 16 / 16 / 0 |
| `cosim_16x16_K64_alltie` | 16x16 | 256 | 64 | 16 | context + 全 tie | 64 / 55 / 9 |
| `cosim_32x32_K96_mixedextreme` | 32x32 | 1024 | 96 | 16 | context + int8 饱和 + 分数 0/32767 极值 | 96 / 87 / 9 |

每例在 csim 和 cosim post-check 中都核对 **status、output_count/real_count/
context_count、scale_out、全部 token 字节、全部 metadata（group_index /
source_type / region_id / valid）**，并检查 `valid==1` 与 representative 严格升序唯一。

## 3. Vitis HLS 2023.2 csim + C/RTL cosim 结果

- 脚本：`scripts/run_cosim_v4.tcl`；top `token_selector_top_packed`；part
  `xczu9eg-ffvb1156-2-e`；`create_clock -period 5.0`。
- C simulation：`0 errors`，`[COSIM-PASS]` 3/3。
- C/RTL cosim：RTL **Verilog**、`-trace_level port`、tool `xsim`、
  **`*** C/RTL co-simulation finished: PASS ***`**，post-check `passed=3 failed=0`。
- cosim latency（`token_selector_top_packed_cosim.rpt`）：
  min 5,668 / avg 209,518 / max 440,899 cycles，total 628,452 cycles。
- 证据：`artifacts/token_selector/2026-09-28-selector-cosim-v4/` 下
  `csim_report.log`、`cosim_report.rpt`、`cosim_options.xml`、
  `console_cosim_v4.log`、`summary.json`、`cosim_fixtures_sha256.txt`、
  `source_sha256.txt`。

## 4. cosim 与 csynth 的 /tmp 工程隔离

- csynth-only v4：`/tmp/robobrain_token_selector_hls_csynth_v4`
- cosim v4：`/tmp/robobrain_token_selector_hls_cosim_v4`

两目录不同，cosim 的 csynth 只在自身工程内进行，不与 v4 csynth 工程共享。

## 5. Ubuntu 24.04 主机工具链 workaround（只影响 host testbench）

1. HLS 2023.2 自带 binutils 2.37 读不了系统 glibc 2.39 的 `.relr.dyn`，链接失败。
2. 换系统 GCC 13 会让 HLS clang-3.9 的 `xilinx-tb-process` 无法解析 GCC 13 的
   libstdc++ 头，top 调用不会被改写成 RTL stub（`COSIM 212-330`）。
3. 最终方案：host 仍用自带 **GCC 8.3.0** 编译，用 `COMPILER_PATH` 指向系统 `ld`，
   用 `LIBRARY_PATH=/usr/lib/x86_64-linux-gnu` 提供多架构 crt，并 `-no-pie` 链接。
   该 workaround 由 `scripts/run_cosim_v4.tcl` 自行设置。DUT/RTL/选择数学未变。

## 6. 边界与限制

- 本轮 cosim **只验证功能**：csim/cosim 全部 PASS，但**不证明 post-route 200 MHz**。
- v4 csynth 结论是 **pre-route 0 ns 估算裕量**（estimated period 3.650 ns = effective
  budget 3.650 ns，slack 0.000 ns，默认 1.35 ns uncertainty 未改），与 cosim 功能
  验证是两件事。
- 未导出 IP、未生成 bitstream、未访问板卡或驱动。

## 7. 复现命令

```bash
make -C fpga_accel/token_selector_hls test
make -C fpga_accel/token_selector_hls asan_run
/home/zhaoxiangyu/conda-envs/robobrain25/bin/python -m unittest \
  evaluation.tests.test_hardware_regular_selector_fixedpoint \
  evaluation.tests.test_hardware_regular_token_selector \
  evaluation.tests.test_hardware_regular_b1_integration \
  evaluation.tests.test_token_selector_packed_contract

mkdir -p /tmp/robobrain_selector_cosim_run_v4 && cd /tmp/robobrain_selector_cosim_run_v4
export LD_LIBRARY_PATH=/home/xilinx/xic/lib/lnx64.o/SuSE:$LD_LIBRARY_PATH
/home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f \
  /home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/scripts/run_cosim_v4.tcl
```

`scripts/run_cosim_v4.tcl` 内部会生成 3 个 fixture、执行 `csim_design`、
`csynth_design` 和 `cosim_design -rtl verilog -trace_level port`；调用方用
`timeout` 包住整个工具调用即可设置合理超时。
