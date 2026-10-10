# Selector v4 HLS IP OOC 导出 + Vivado OOC synth/opt/place/route

- 日期：2026-09-28
- 决策：`SELECTOR_V4_IP_OOC_ROUTE_PASS`（status **PASS**）
- 范围：只做 packed v4 Token Selector `token_selector_top_packed` 的 **Verilog IP
  导出**与 **Vivado out-of-context（OOC）synth + opt + place + route**。使用
  Vitis HLS 2023.2 `export_design -flow impl`（等价于仓库中既有的成功 HLS-IP
  OOC route 流程，见 `fpga_accel/q8q8_pe16_hls/scripts/run_hls.tcl`）。
- 未做：GPU、ZCU102、驱动、bitstream；未改选择数学、量化、host 5 段字节布局、
  IP 端口/寄存器布局；**未改 HLS 默认 clock uncertainty，也未放宽 clock**。
- 所有 Vivado/HLS project、cache、checkpoint、IP zip 都在 `/tmp`，仓库只保存小
  report/summary/hash。

## 0. score_q 合同澄清（本轮先修正文档）

- 正式 `score_q` 合同：**unsigned Q0.15**，`score = score_q / 32768`，合法上限
  **32767**（见 `fpga_accel/token_selector_hls/quantization_contract.md` §2）。
- `bounddiff_v4_00..47` 中的 `0/65535` 交替**只是对 `uint16` 原始载荷的鲁棒性
  压力测试，不是合同内的量化输入**；该用例保留、不删除。
- 已修正 `reports/token_selector/2026-09-28-selector-cosim-v4.md`（“每年区域”→
  “每个区域”、合同澄清）与 `reports/token_selector/2026-09-28-selector-csynth-v4.md`、
  `fpga_accel/token_selector_hls/README.md`（同一合同澄清）。

## 1. df 审计（运行前 / 运行后）

| 文件系统 | 容量 | 已用 | 可用 | 挂载点 |
|---|---|---|---|---|
| `/dev/nvme0n1p3`（含 `/tmp`） | 397G | 109G | **269G** | `/` |
| `/dev/sda1`（含 `/home`） | 1.8T | 1.7T | 70G | `/home` |

- 运行前后快照一致（`/tmp` 上 269G 可用），本流程未对大文件落入仓库。
- 原始快照：`artifacts/token_selector/2026-09-28-selector-ooc-v4/df_before.txt`、
  `df_after.txt`。

## 2. 流程与可复现命令

新增脚本：

- `scripts/export_selector_v4_ip_ooc.tcl`：`open_project -reset`（`/tmp`）→
  `set_top token_selector_top_packed` → `set_part xczu9eg-ffvb1156-2-e` →
  `create_clock -period 5.0 -name default`（**不传 `-uncertainty`**，保持 HLS 默认）
  → `csynth_design` → `export_design -rtl verilog -format ip_catalog -flow impl ...`。
- `scripts/run_selector_v4_ip_ooc.sh`：df 审计 + 运行上面的 tcl + 调用 collector。
- `scripts/summarize_selector_ooc.py`：解析并复制 route status / timing summary /
  utilization / DRC / clock / critical path，写 `summary.json`，计算 IP zip 与
  source 的 sha256，判定硬门槛。

运行命令：

```bash
bash scripts/run_selector_v4_ip_ooc.sh
# 内部等价于：
mkdir -p /tmp/robobrain_selector_export_run_v4 && cd /tmp/robobrain_selector_export_run_v4
export LD_LIBRARY_PATH=/home/xilinx/xic/lib/lnx64.o/SuSE:$LD_LIBRARY_PATH
/home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f \
  /home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/scripts/export_selector_v4_ip_ooc.tcl
python3 scripts/summarize_selector_ooc.py \
  --project-dir /tmp/robobrain_token_selector_hls_export_v4 \
  --ip-zip /tmp/robobrain_selector_ip_ooc_v4/ip/token_selector_top_packed_ip.zip \
  --artifact-dir artifacts/token_selector/2026-09-28-selector-ooc-v4
```

工具版本与 `/tmp` 路径：

| 项 | 值 |
|---|---|
| HLS | Vitis HLS 2023.2，`csynth_design` estimated Fmax 273.97 MHz（= 3.650 ns） |
| Vivado | `Vivado v.2023.2 (lin64) Build 4029153 Fri Oct 13 20:13:54 MDT 2023` |
| HLS/Vivado project | `/tmp/robobrain_token_selector_hls_export_v4` |
| Vivado impl run | `/tmp/robobrain_token_selector_hls_export_v4/solution1/impl/verilog` |
| IP zip | `/tmp/robobrain_selector_ip_ooc_v4/ip/token_selector_top_packed_ip.zip` |
| IP zip 大小 / sha256 | **719,284 B** / `3ea8b74f1f7dd47de5b59c3e0d78a168ed3000364b8af887be1a9368d1f3f265` |
| HLS 运行日志 | `/tmp/robobrain_selector_export_run_v4/console_export_v4.log`（281,857 B，sha256 `2566a0e31ea282a4e254e1b06759855c24bf38df11c22306d5422004939b1edd`，`ERROR` 行 0） |
| 源码 sha256 | `src/token_selector_hls.cpp` `078a3a18…d7da`；`include/token_selector_hls.h` `f76ac1dd…ade9c`（与 v4 csynth 冻结哈希一致，未改） |

关于 clock uncertainty：`csynth_design` 仍使用 HLS 默认 uncertainty（`create_clock`
默认 27% × 5 ns = **1.35 ns**，未改）。HLS 导出的 OOC XDC 只含
`create_clock -name ap_clk -period 5.000`，因此 Vivado OOC 路由后的 timing 含
Vivado 自身 jitter（routed 报 0.035 ns），**不含** HLS 的 1.35 ns 调度 uncertainty——
这是 HLS-IP OOC 标准流程，clock 未被放宽；见 §5 的三方区分。

## 3. OOC 结果

### 3.1 route status（`report_route_status`，Fully Routed）

| 项 | 值 |
|---|---|
| logical nets | 145,683 |
| routable nets | 52,474 |
| fully routed nets | 52,474 |
| **nets with routing errors** | **0** |
| **unrouted nets** | **0** |

### 3.2 timing summary（`report_timing_summary -max_paths 10 -report_unconstrained`，Routed）

| 指标 | 值 |
|---|---|
| Clock `ap_clk` | 5.000 ns / 200.000 MHz |
| **WNS (setup)** | **+0.689 ns** |
| TNS | 0.000 ns（failing endpoints 0 / 70,186） |
| **WHS (hold)** | **+0.036 ns** |
| THS | 0.000 ns（failing endpoints 0 / 70,186） |
| WPWS / TPWS | 1.958 ns / 0.000 ns |
| 结论 | “All user specified timing constraints are met.” |

### 3.3 utilization（`report_utilization`，Fully Placed）

| 资源 | 用量 | 器件容量 | 利用率 |
|---|---|---|---|
| LUT (CLB LUTs) | 24,949 | 274,080 | **9.10%** |
| FF (CLB Registers) | 29,458 | 548,160 | **5.37%** |
| BRAM (Block RAM Tile) | 42 (RAMB36 = 37，RAMB18 = 10) | 912 | **4.61%** |
| BRAM_18K 等价 | 84 | 1,824 | **4.61%** |
| DSP | 85 | 2,520 | **3.37%** |
| URAM | 0 | 0（器件无 URAM） | 0% |

### 3.4 DRC（`report_drc`，Fully Routed）

- **Error = 0；Critical Warning = 0**。
- Warning 共 130：`DPIP-2` 16（DSP 输入流水）、`DPOP-3` 38、`DPOP-4` 75
  （DSP 输出流水建议）、`RTSTAT-10` 1（无可路由负载）。这些是 OOC IP 单跑的
  典型非致命告警，不是 DRC Error。
- 另有 OOC 期待的 `Route 35-197/198`（无 `HD.CLK_SRC` / `HD.PARTPIN_LOCS`）与
  `check_timing` 的 `no_input_delay` / `no_output_delay`（无外部 I/O 约束），
  均不构成 DRC Error。

### 3.5 post-route critical path（真实路由后）

`report_timing -max_paths 10`（`critical_paths_routed.rpt` / `critical_path_routed.md`）：

| 项 | 值 |
|---|---|
| Slack | **MET +0.689 ns**，requirement 5.000 ns |
| Source | `grp_ts_pool_context_packed_fu_1990/weights_U/ram_reg_bram_0/CLKARDCLK`（RAMB18E2） |
| Destination | `grp_ts_pool_context_packed_fu_1990/.../mul_ln290_reg_1294_reg/DSP_OUTPUT_INST/ALU_OUT[10]` |
| Data path delay | **4.258 ns**（logic 2.278 ns = 53.5%，route 1.980 ns = 46.5%） |
| Logic levels | 5（全为 DSP48E2 内部级：DSP_A_B_DATA/DSP_PREADD_DATA/DSP_MULTIPLIER/DSP_M_DATA/DSP_ALU） |

即 **post-route 最差路径落在未改动的 `ts_pool_context_packed`（int8×Q0.15 权重
乘加绑 DSP）内**，而不是 v4 多遍 max-select 的选择逻辑；这解释了为什么 v4 在
pre-route 估 0 ns 裕量、post-route 反而有 +0.689 ns 正裕量（HLS OOC XDC 不含
1.35 ns 调度 uncertainty）。

## 4. 硬门槛判定

| 门槛 | 实测 | 判定 |
|---|---|---|
| export 0 error | HLS 日志 `ERROR` 行 0；`export_design` 成功生成 IP zip | PASS |
| route 0 error | Vivado `route_design completed successfully`，0 Errors | PASS |
| WNS ≥ 0 | +0.689 ns | PASS |
| WHS ≥ 0 | +0.036 ns | PASS |
| unrouted = 0 | 0（routing errors 0） | PASS |
| DRC Error = 0 | 0 | PASS |
| BRAM < 50% | 4.61% | PASS |
| DSP < 50% | 3.37% | PASS |
| FF < 50% | 5.37% | PASS |
| LUT < 50% | 9.10% | PASS |

全部满足 → status **PASS**（决策 `SELECTOR_V4_IP_OOC_ROUTE_PASS`）。clock 未放宽、
选择数学/host 布局/default uncertainty 均未改。

## 5. 必须区分的三件事

| 结论 | 证据 | 说明 |
|---|---|---|
| **csynth pre-route 估算** | `reports/token_selector/2026-09-28-selector-csynth-v4.md` | estimated 3.650 ns = budget 3.650 ns、slack **0.000 ns**，含 HLS 默认 1.35 ns uncertainty；pre-route，非签核 |
| **C/RTL cosim 功能** | `reports/token_selector/2026-09-28-selector-cosim-v4.md` | 3 个代表 case `PASS`，只证明功能等价，**不证明 post-route 200 MHz** |
| **OOC post-route 时序/资源** | 本报告 + `artifacts/token_selector/2026-09-28-selector-ooc-v4/` | WNS +0.689 / WHS +0.036 @ 5.000 ns（Vivado OOC XDC，不含 HLS 1.35 ns 调度 uncertainty），资源 <10%，DRC Error 0，unrouted 0 |

后两者不能互相替代：cosim 不证明时序；OOC post-route 也不等于整机（含 PS/互连）
签核。

## 6. 证据

`artifacts/token_selector/2026-09-28-selector-ooc-v4/`（总约 480 KB）：

- `summary.json`（含工具版本、命令、`/tmp` 路径、IP zip hash/size、门槛判定）
- `route_status_routed.rpt`、`route_status_hls_routed.rpt`
- `timing_summary_routed.rpt`、`timing_hls_routed.rpt`
- `utilization_placed.rpt`、`utilization_hls_routed.rpt`
- `drc_routed_summary.rpt`
- `clock_utilization_routed.rpt`
- `critical_paths_routed.rpt`、`critical_path_routed.md`
- `df_before.txt`、`df_after.txt`

IP zip 本体与其 Vivado checkpoint 均留在 `/tmp`，未复制入仓库；仓库只记录
大小与 sha256。

## 7. 限制与未做

- 这是单 IP OOC 流程：设计顶层是 HLS 生成的 IP 包装 `bd_0_wrapper`，**不是**
  含 ZCU102 PS、DDR、SmartConnect 的整机实现；不声称整机时序/带宽/功耗。
- **未生成 bitstream、未访问板卡、未写驱动、未跑 GPU**。
- HLS `HLS 200-790`（可变界内层扫描未达 target II）仍存在，不影响本轮正确性/
  时序门槛（0 error、timing met）。
- 未做第二轮优化（本轮无需，门槛已全过）。

## 8. 精确下一步

- 本里程碑到此为止：IP 已导出并完成 OOC route，所有硬门槛满足。
- 后续如要上板，需要独立授权门：把该 IP 集成进含 PS/DDR 的 ZCU102 系统
  （`launch_runs impl_1 -to_step route_design` 的整机 route）、生成 bitstream、
  写驱动，再做板级 200 MHz 与功能验证；这些本轮均未做。
