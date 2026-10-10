# HLS Selector 只读审计：工具链、可复用模块与动态 context 契约核对

- 日期：2026-09-27
- 状态：`READ_ONLY_AUDIT / NO_SYNTHESIS / NO_RTL_EDIT / NO_GPU / NO_BOARD`
- 工作区：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`
- 边界：主插入边界 `B1`（vision block 0 之后），消融标签 `B0`。
- 范围：只读列目录、读已有报告/源码、`--version` 与文件查找。未启动综合，未修改任何已有源码，未访问 ZCU102。
- 一句话结论：CPU 侧 `hardware_regular` selector 契约与 B1 集成已闭合；**PL 侧不存在任何 Selector/TopK/context 池化 HLS、RTL、寄存器草案或 testbench**，现有硬件全部是 Q8×Q8 矩阵/GEMV/FFN 引擎；INT8 token 段只有字节数、没有量化契约，INT4 完全不存在。

## 1. 工具链与 ZCU102 器件证据

安装根：`/home/xilinx`（由 `/home/xilinx/*/2023.2/settings64.sh` 与可执行文件确认）。

| 工具 | 版本 | 证据 |
|---|---|---|
| Vivado | v2023.2 (64-bit)，SW Build 4029153 (2023-10-13) | `/home/xilinx/Vivado/2023.2/bin/vivado -version` |
| Vitis | v2023.2 (64-bit)，SW Build 4026344 (2023-10-11) | `/home/xilinx/Vitis/2023.2/bin/vitis -version`（另报 User HOME 磁盘 <100 MB，非工具故障） |
| Vitis HLS | v2023.2，SW Build 4023990 (2023-10-11) | `LD_LIBRARY_PATH=/home/xilinx/xic/lib/lnx64.o/SuSE vitis_hls -version`；默认 PATH 下缺 `libtinfo.so.5`，仓库根 `vitis_hls.log` 亦为 v2023.2 |

- License：`/home/xilinx/...` 未在沙箱 `env` 中出现 `XILINXD_LICENSE_FILE`/`LM_LICENSE_FILE`；用户级 `~/.Xilinx/Xilinx.lic -> /home/zhaoxiangyu/Xilinx.lic`，含 `VIVADO_HLS`、`Vivado_System_Edition` 等 INCREMENT（permanent, HOSTID=ANY）。仅 `--version` 成功即可判断可用，未做 license checkout 测试。
- 器件/板卡：ZCU102 board store 存在 `3.3` 与 `3.4` 两版；`board.xml` 的 `part0` = `xczu9eg-ffvb1156-2-e`，与 `installed_devices.txt` 一致。即上次实现所用 part 在当前安装中可用。
- 上次最小综合证据（只读，未重跑）：`fpga_accel/q8q8_gemv_stream6_hls/artifacts/stream6_ooc_impl_20260901/summary.json`
  - 决策 `STREAM6_OOC_200MHZ_PASS / FULL_SYSTEM_INTEGRATION_NEXT`，Vivado 2023.2，part `xczu9eg-ffvb1156-2-e`。
  - 目标 5.0 ns；post-route period 4.453 ns（≈224.57 MHz），WNS +0.546568 ns、WHS +0.022683 ns、unrouted 0；LUT 19563 / FF 26227 / DSP 240 / BRAM 123；无 bitstream、无上板。
  - 更早的同类证据：`fpga_accel/q8q8_pe16_hls/artifacts/vitis_hls_csynth.log`、`q8q8_pe16_dataflow_hls`、`pl_resident_ffn_hls/artifacts/20260830T*`。均非 Selector。

## 2. hardware / HLS / RTL / Vivado / Vitis 目录与状态

### 2.1 HLS 计算核（全部为规则稠密 Q8×Q8 / 向量，无 Selector）

| 目录 | 源/测试台 | 状态（依据既有报告） |
|---|---|---|
| `fpga_accel/q8_gemm_hls` | `src/q8_gemm.cpp`, `tb/tb_q8_gemm.cpp` | Q8 GEMM 原型（2026-08-22） |
| `fpga_accel/q8q8_gemm_hls` | `src/q8q8_gemm.cpp`, `tb/tb_q8q8_gemm.cpp` | Q8×Q8 整数 GEMM |
| `fpga_accel/q8q8_dataflow_hls` | `src/q8q8_gemm.cpp`, `tb/tb_q8q8_gemm.cpp` | DATAFLOW load/compute/store |
| `fpga_accel/q8q8_weight_stationary_hls` | `src/q8q8_gemm.cpp` | 权重驻留变体 |
| `fpga_accel/q8q8_output_burst_hls` | `src/q8q8_gemm.cpp`, `tb/tb_q8q8_gemm.cpp` | 128-bit output burst v3 基线 |
| `fpga_accel/q8q8_pe16_hls` | `src/q8q8_gemm.cpp`, `tb/tb_q8q8_gemm.cpp` | PE16，C 9/9、板端数值 `max_abs=0` |
| `fpga_accel/q8q8_pe16_dataflow_hls` | `src/q8q8_gemm.cpp`, `tb/tb_q8q8_gemm.cpp` | DATAFLOW IP，OOC 时序通过 |
| `fpga_accel/q8q8_gemv_stream6_hls` | `src/q8q8_gemv_stream6.cpp`, `tb/tb_q8q8_gemv_stream6.cpp` | 6 通道 Decoder GEMV，C/RTL cosim 9/9 通过，OOC 200 MHz 通过 |
| `fpga_accel/pl_resident_ffn_hls` | `src/ffn_vector.cpp`, `tb/tb_ffn_vector.cpp` | 向量/FFN 驻留，proxy 数值通过 |

### 2.2 Vivado 完整系统

`fpga_accel/{q8q8_zcu102_vivado, q8q8_v2_zcu102_vivado, q8q8_v3_zcu102_vivado, q8q8_pe16_zcu102_vivado, q8q8_pe16_dataflow_zcu102_vivado, q8q8_gemv_stream6_zcu102_vivado, pl_resident_ffn_zcu102_vivado}`：每个含 `scripts/build_system.tcl`（PE16 DATAFLOW 分支另含 `export_bitstream.tcl`、`generate_dataflow_fsbl.tcl`）。PE16 DATAFLOW 分支已出 bitstream；stream6 只有 OOC synth/impl。

### 2.3 PS 运行时 / 桥接 / 描述符 / RTL smoke

- `fpga_accel/q8q8_runtime`：AArch64 UIO 运行时（`src/q8q8_uio.c`、`q8q8_arena.c`、`q8q8_dispatch.c`、`q8q8_apm.c`），寄存器映射测试 `scripts/test_register_map.sh`。
- `fpga_accel/ggml_q8q8_bridge`：GGML Q8_0↔硬件桥接，含 `tests/test_upstream_quantizer.c`。
- `fpga_accel/token_sparse_descriptor`：Token/Block Sparse Descriptor v1（C 编码/解析 + CRC），普通与 ASan 10 类测试通过；`build/` 下有已编译二进制。
- `fpga_accel/pl_smoke/rtl`：`pl_jtag_smoke_core.sv`、`pl_uart_smoke_core.sv` 及 top。
- `fpga_accel/heterogeneous_partition`、`fpga_accel/runtime_activation_capture`、`fpga_accel/profiling`：Python/契约分析，非 HLS。
- `deployment/`：PE16 RAM Linux 上板包、FSBL、board/probes/reports；`deployment/scripts/*registers*.tcl` 为 PS 寄存器探查，非 Selector。

## 3. 对照 `hardware_regular` selector 契约：可复用模块（准确文件/函数/接口）

### 3.1 CPU golden / 选择数学（`evaluation/hardware_regular_token_selector.py`，当前 sha256 `c55cfdc3…2f85932b`）

- 校验/常量：`validate_hidden`(122)、`validate_budget`(136)、`validate_boundary`(150)、`MERGE_UNIT=4`、`FEATURE_DIM=1024`、`PROMPT_DIM=2560`、`SCORE_DIM=128`、`REGION_GRID=3`、`MAX_CONTEXT_SLOTS=9`。
- 打分：`LowRankProjection`(163)、`group_representation`(204)、`group_l2_saliency`(209)、`semantic_cosine_scores`(215)、`combine_scores`(250)、`score_groups`(267)、`scores_from_precomputed`(286)。
- 区域/锚点：`build_regions`(370)、`region_representative`(420)、`region_id_by_group`(474)。
- 选择：`select_real_groups`(498)、`context_pooling_weights`(604)、`pool_region_context`(617)、`region_context_representative`(434)、`select_hardware_regular_groups`(701)。
- 传输布局：`align_up`(897)、`int8_token_bytes`/`uint16_group_index_bytes`/`uint8_source_bytes`/`uint8_region_bytes`/`uint8_valid_bytes`(907-924)、`fpga_transfer_layout`(964)、`describe_fpga_layout`(1003)。
- 配置加载：`load_selector_config`(1032) + `configs/hardware_regular_token_selector.json`（`context_slots_policy=dynamic_up_to_max`、`max_context_slots=9`、`fpga_layout.alignment_bytes=64`）。

### 3.2 软件 B1 集成（可作 C 仿真 golden 向量来源）

- `evaluation/hardware_regular_b1_integration.py`（`1854f437…e91f3d61`）：`forward_hardware_regular_b1_vision`、`prepare_b1_multimodal_inputs`，含 `compact_hidden = groups.reshape(K*4,1024)`、`cos/sin = index_select(patch_indices)`、`cu_seqlens=[0,K*4]`。
- `evaluation/run_hf_hardware_regular_b1.py`（`8e18b6bf…52370611`）：CLI、`--score-mode l2|prompt`、投影哈希契约。
- `evaluation/tests/test_hardware_regular_token_selector.py`、`test_hardware_regular_b1_integration.py`：CPU 回归。

### 3.3 可复用的硬件/系统资产（非 Selector，但接口可套用）

- AXI 接口模板：`fpga_accel/q8q8_pe16_hls/src/q8q8_gemm.cpp:241-254` 的 `m_axi`（bundle/gmem、burst、outstanding）与 `s_axilite bundle=control`；`q8q8_gemv_stream6.cpp:319-334` 的 6 端口模板。
- Block Design 模板：`fpga_accel/q8q8_pe16_dataflow_zcu102_vivado/scripts/build_system.tcl`（PS `M_AXI_HPM0_FPD` 控 `s_axi_control`；`S_AXI_HPC0/HPC1/HP0` + SmartConnect 承载数据端口）。
- 运行时/寄存器校验模板：`fpga_accel/q8q8_runtime/scripts/test_register_map.sh`（从 IP zip 的 `xq8q8_gemm_hw.h` 校验 10 个偏移）。
- 描述符合同：`fpga_accel/token_sparse_descriptor/include/token_sparse_descriptor.h`（`tsd_encode`/`tsd_decode`/`tsd_view_entry`/`tsd_crc32`，64 B 头）。

## 4. 缺失模块（仓库中不存在）

1. Selector HLS 顶层：无 score/guard/select/compact 任意一个 C++ 实现。
2. 硬件两级 Top-K / anchor / 区域配额引擎：无。
3. 动态 context 池化硬件（加权均值 + 加权质心最近代表）：无。
4. Selector AXI4-Lite 寄存器草案/驱动头：无（唯一寄存器图是 GEMM 的 10 寄存器）。
5. token INT8 量化契约（scale/zero-point、per-tensor/per-group/per-token）与打包器：无。
6. INT4 路径：无任何硬件或格式（`docs/ROADMAP.md` 仅列为未来 W4A8/W4A16 方向）。
7. Selector 专用 AXI4-Stream / AXI DMA 数据面：无；现有核均为 `m_axi + s_axilite`，BD 中未实例化 `axi_dma`。
8. Selector HLS testbench / C 仿真 / C-RTL cosim：无。
9. 5 段传输布局的二进制序列化器：无（`writes_binary_file=false`，只有 offset/size 计算）。
10. 硬件侧 `source_type/region_id/valid` 读取与分支：无。
11. `Wt/Wv` 权重（软件阻塞）：仓库内无产物。

## 5. INT8/INT4 packing、64B 布局、寄存器草案 vs 当前动态 context 元数据

当前动态 context 元数据：`source_type∈{0,1}`、`region_id∈[0,9)`、`valid` 恒 true、`group_indices` 全局唯一升序、`C=eligible region 数(≤9)`、`real=K-C`、无 padding。

| 已有一项 | 现状 | 与动态 context 的一致性 |
|---|---|---|
| INT8 packing | 仅矩阵引擎 Q8_0：34 B/32 权重（FP16 scale + 32×INT8）；stream6 为 1 尺度头 + 8 Q8 块 = 288 B 承载 272 B，94.44% | **不一致/缺失**。Selector 的 `tokens_int8` 段只是 `K*4*1024` 字节计数，没有量化 scale/zero-point 契约，也没有随包尺度段；`groups` 是 float64，INT8 转换未定义 |
| INT4 packing | 无 | **完全缺失**，谈不上一致 |
| 64B 布局 | `fpga_transfer_layout`：`tokens_int8 → group_index_uint16 → source_type_uint8 → region_id_uint8 → valid_uint8`，每段 64 B 对齐，按 K 计；K=48/64/96 总量 196928/262464/393792 | **部分一致**。段序、对齐、按 K 计算与动态 C 无关，符合契约；但 `group_index` 由软件 `int64` 截为 `uint16` 且无 `G<65536` 守卫；无 `C`/`real_count` 段，硬件需自行累计 `source_type`；`valid` 恒 true 仍占段；`tokens_int8` 的 dtype 无量化语义 |
| 寄存器草案 | 只有 Q8Q8 GEMM 的 10 个偏移（control 0x00、ap_return 0x10、weights_q 0x18、weights_scale 0x24、activations_q 0x30、activations_scale 0x3c、output 0x48、m 0x54、n 0x5c、k 0x64） | **不一致**。没有任何 Selector 寄存器（K/预算、grid_h/grid_w、region_grid、max_context_slots、C、lambda_semantic、5 段基址）可由现有草案驱动 |
| TSD v1 | 64 B 头 + `uint32 original_index` + 0–4 位置分量 + 双 CRC；条目须严格递增唯一 | **两套并行合同**。TSD v1 无 `source_type/region_id`，无法把 context 代表标记为 synthetic；Selector 5 段布局也未复用 TSD 头/CRC |
| 产物哈希 | `artifacts/token_selector/2026-09-27-hardware-regular-selector-contract.json` 记 `hardware_regular_token_selector.py = 7e55f930…` | **陈旧**。当前文件为 `c55cfdc3…2f85932b`（B1 集成后扩展了显式分数入口）；B1 审计报告引用的行号（如 623-789、903-920）也随之过期，当前为 701-889、964-1020 |

## 6. 下一次实现允许新增/修改的最小文件清单（本轮不实现）

只新增，不改任何既有源码/模型；不决定 PE 数、位宽、Top-K 并行度或时钟目标。

| 文件 | 作用 |
|---|---|
| `fpga_accel/token_selector_hls/include/token_selector_hls.h` | 参数化类型/常量（K、grid、region_grid、max_context_slots、段序），位宽保持模板参数，不在此冻结 |
| `fpga_accel/token_selector_hls/src/token_selector_hls.cpp` | score→guard→select→compact 顶层；AXI-Lite 控制 + 5 段 `m_axi`；按 `source_type` 输出 dynamic context |
| `fpga_accel/token_selector_hls/tools/selector_layout_fixture.c` | 从 CPU 契约导出 5 段 golden 二进制/offset 表，供 C 仿真对照 |
| `fpga_accel/token_selector_hls/tb/tb_token_selector_hls.cpp` | C testbench（下节验收项） |
| `fpga_accel/token_selector_hls/scripts/run_hls.tcl` | 仅 `csim_design`；csynth/cosim 本轮不启动 |
| `fpga_accel/token_selector_hls/Makefile` | 本地编译/运行 C 仿真 |
| `fpga_accel/token_selector_hls/README.md` | 参数化说明与未冻结项 |

可选（若需 Python↔C 布局对拍）：`evaluation/tests/test_hls_selector_layout_parity.py`（新增）。不修改 `evaluation/hardware_regular_token_selector.py` 的现有语义；如确需 `G` 上界守卫，另在报告中提出，不在本轮改。

## 7. C 仿真验收项（下一轮）

1. Identity bypass：`G ≤ K` 时输出 token 与输入逐字节一致，`C=0`、`source_type` 全 0。
2. 压缩路径 `K∈{48,64,96}`：输出恰 `K` 组，`real+context=K`。
3. 动态 `C`：`2x9`（3 空区）与 `5x4`（2 单组区）下 `C` 等于 eligible region 数且 `≤9`，无零/`-1` padding。
4. 每非空区域至少 1 个 real anchor。
5. 每个 eligible region 至少保留 1 个未选 group。
6. `group_indices` 全局唯一、严格升序；同分取更小 `original_group_index`。
7. context 池化值等于 CPU `pool_region_context`（定点容差内），代表为距加权质心最近的未选 group。
8. 5 段 offset/size 与 Python `fpga_transfer_layout` 在 K=48/64/96 上完全一致（64 B 对齐）。
9. `source_type` 正确区分 real/context，`region_id∈[0,9)`，`valid` 全 true。
10. 确定性：同输入两次运行逐字节一致。
11. 非法预算（非 16 倍数）与预算不足（`budget-C < 非空区域数`）被拒绝。
12. 至少对 K=48/64/96 复现 `selector_layout_fixture.c` 的 frozen golden。
13. `tokens_int8` 段仅校验字节布局；量化数值正确性在 scale/zero-point 契约定义前不作声明。

## 8. 已知未验证 / 阻塞

- 本轮**未启动任何综合**、未生成 RTL/bitstream、未访问 ZCU102。
- 工具链默认 shell 缺 `libtinfo.so.5`（`vitis_hls` 需 `LD_LIBRARY_PATH`），未做 license checkout 与完整 `v++` 流程验证。
- Selector 无任何硬件实现，所有 PL 结论均为“存在/缺失”，不是时序、资源或精度结论。
- `Wt/Wv` 权重缺失，压缩路径真实语义分数不可生成（沿用 B1 审计结论）。
- context 池化质量、INT8/INT4 精度、端到端加速均未验证，本报告不作任何此类声明。
