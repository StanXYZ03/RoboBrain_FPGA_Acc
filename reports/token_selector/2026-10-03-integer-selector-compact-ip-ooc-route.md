# 紧凑整数 Selector IP 导出 + Vivado OOC 布线报告

- 日期：2026-10-03
- 决策：**INTEGER_SELECTOR_COMPACT_IP_OOC_ROUTE_PASS**
- part：`xczu9eg-ffvb1156-2-e`，top：`integer_selector_pipeline_compact`，时钟：`ap_clk = 5.000 ns`
- 流程：Vitis HLS 2023.2 csynth_design + export_design -format ip_catalog (-flow syn); Vivado 2023.2 OOC synth_design + opt_design + place_design + phys_opt_design + route_design

## 命令与工具
- `hls`：export LD_LIBRARY_PATH=/home/xilinx/xic/lib/lnx64.o/SuSE:$LD_LIBRARY_PATH && /home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f fpga_accel/integer_selector_pipeline_compact_ooc/scripts/run_hls_export.tcl
- `vivado`：/home/xilinx/Vivado/2023.2/bin/vivado -mode batch -source fpga_accel/integer_selector_pipeline_compact_ooc/scripts/run_vivado_ooc.tcl -log /tmp/.../vivado_ooc.log
- `driver`：bash fpga_accel/integer_selector_pipeline_compact_ooc/scripts/run_ooc.sh
- `timeouts_s`：{'hls': 7200, 'vivado_total': 5400}
- 工具版本：
```
=== tool versions ===
Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.2 (64-bit)
SW Build 4023990 on Oct 11 2023
IP Build 4028589 on Sat Oct 14 00:45:43 MDT 2023
Tool Version Limit: 2023.10
vivado v2023.2 (64-bit)
Tool Version Limit: 2023.10
SW Build 4029153 on Fri Oct 13 20:13:54 MDT 2023
IP Build 4028589 on Sat Oct 14 00:45:43 MDT 2023
```
- license：synthesis+implementation 获取成功=True，license error 行=0
- 环境 CRITICAL WARNING：['Common 17-741']（[Common 17-741] Tcl store 写权限，非设计 DRC，不影响实现/时序）

## 接口门（来自 component.xml/csynth.xml）
- m_axi bundles：['input_arena', 'output_arena', 'wv_q']（宽度 {'input_arena': '128', 'output_arena': '128', 'wv_q': '128'}）
- s_axi bundles：['control']（control WDATA 32 bit）
- 无 control_r：True；无 score_q 端口：True
- three_128bit_axi4_masters_input_arena_wv_q_output_arena：True
- one_axi4lite_slave_control：True
- no_control_r_interface：True
- no_external_score_q_port：True
- interface_gate_pass：True

## HLS 身份
- accepted estimated clock 3.65 ns / new 3.65 ns
- accepted resources {'BRAM_18K': 295, 'DSP': 212, 'FF': 45156, 'LUT': 91089, 'URAM': 0}
- new resources {'BRAM_18K': 295, 'DSP': 212, 'FF': 45156, 'LUT': 91089, 'URAM': 0}
- identical：True
- HLS export log：errors=0 fatals=0 csynth_done=True export_done=True
- VLNV：`robobrain.local:hls:integer_selector_pipeline_compact:1.0`
- IP archive：ip/export.zip，1051551 B，sha256 `b4e0654792e6591afd388aeb34bff713cd2b8688d068be1942a987bdff9a4090`
- routed DCP：vivado/post_route.dcp，33501830 B（<=200MiB：True），sha256 `25f1ca2f5ef1ecb28f119f1300daf53aff0735723e5c36a7ed66a7a50696a4d3`

## OOC 阶段
- synth_design：rc=0，164 s
- opt_design：rc=0，15 s
- place_design：rc=0，96 s
- phys_opt_design：rc=0，1 s
- route_design：rc=0，64 s
- HLS exit=0，Vivado exit=0，timeout=False

## 时序
- post-route WNS=0.529 ns / TNS=0.0 ns / WHS=0.028 ns / THS=0.0 ns
- 由 5 ns 需求推导的 achieved frequency：223.6636 MHz
- 关键路径：grp_ts_pool_context_packed_fu_2549/weights_U/ram_reg_bram_0/CLKARDCLK -> grp_ts_pool_context_packed_fu_2549/grp_ts_pool_context_packed_Pipeline_VITIS_LOOP_276_2_VITIS_LOOP_281_3_fu_303/mul_ln290_reg_1294_reg/DSP_OUTPUT_INST/ALU_OUT[15]（slack 0.529 ns，data path 4.418 ns）

## 资源（post-route）
- lut：used 27607 / available 274080 = 10.07%
- ff：used 30997 / available 548160 = 5.65%
- bram_tile：used 122.5 / available 912 = 13.43%
- dsp：used 210 / available 2520 = 8.33%
- uram：used 0 / available 0 = 0.0%（XCZU9EG has no URAM）
- post-synth：LUT 28427/274080=10.37%，FF 31049/548160=5.66%，BRAM 122.5/912=13.43%，DSP 210/2520=8.33%

## 布线 / DRC
- {'present': True, 'logical_nets': 251921, 'routable_nets': 59967, 'fully_routed_nets': 59967, 'routing_errors': 0, 'unrouted_nets': 0}
- {'present': True, 'violations_found': 125, 'by_severity': {'Warning': 125}, 'error': 0, 'critical_warning': 0, 'warning': 125}

## 门汇总
- protected_files_byte_identical：True
- hls_csynth_and_export_no_error_fatal：True
- vivado_ooc_synth_ok：True
- opt_design_ok：True
- place_design_ok：True
- phys_opt_design_ok：True
- route_design_ok：True
- post_route_setup_wns_ge_0：True
- post_route_hold_whs_ge_0：True
- no_unrouted_nets：True
- drc_no_error_no_critical_warning：True
- resources_each_below_80_percent：True
- part_exact：True
- clock_exact_5_000_ns：True
- interface_gates：True
- no_bitstream_generated：True

## 受保护文件 SHA 审计
- before=2260 文件 / aggregate `8fec4b4988ed014b5ac0e4e453ab46153ec731dc825ab2d6810608240ca5dfc5`
- after=2260 文件 / aggregate `8fec4b4988ed014b5ac0e4e453ab46153ec731dc825ab2d6810608240ca5dfc5`
- 未变：True（changed=[] missing=[] added=[]）

## 产物存储审计
- 总字节 40852056 / 文件 44
- 最大的 20 个文件：
  - vivado/post_route.dcp：33501830 B
  - vivado/vivado_ooc.log：1194323 B
  - vivado/vivado_console.log：1139379 B
  - ip/export.zip：1051551 B
  - hls/csynth.xml：793285 B
  - protected_sha_after.json：444839 B
  - protected_sha_before.json：444839 B
  - ip/component.xml：387049 B
  - hls/hls_export.log：314226 B
  - vivado/timing_summary_post_route.rpt：274273 B
  - vivado/timing_summary_post_synth.rpt：272750 B
  - hls/csynth.rpt：235559 B
  - vivado/utilization_hier_post_route.rpt：185086 B
  - vivado/utilization_hier_post_synth.rpt：184972 B
  - vivado/timing_summary_post_place.rpt：112048 B
  - vivado/drc_post_route.rpt：98002 B
  - vivado/clock_utilization_post_route.rpt：57860 B
  - scripts/summarize_ooc.py：40383 B
  - vivado/clock_utilization_post_synth.rpt：18810 B
  - vivado/critical_paths_post_route.rpt：17575 B

## 限制
- Out-of-context (OOC) only: no PS, no DDR, no AXI interconnect contention, no DMA and no full ZCU102 block design.
- Post-route timing/resources are the OOC Vivado result for the exported IP; they are not a board measurement.
- No bitstream was generated and the board was not accessed.
- No end-to-end model latency, throughput, speedup or task-accuracy claim is made.
- The exported HLS OOC XDC carries only create_clock ap_clk 5.000 ns; it has no explicit set_clock_uncertainty. Vitis HLS csynth used its default 1.35 ns scheduling uncertainty, which is not present in the post-route OOC constraints.
- The exported IP interface gate is parsed from the generated component.xml/csynth.xml, not from the C++ pragmas.

## 建议下一步
- Build the minimal Zynq UltraScale+ MPSoC block design around this exported IP (PS, clock/reset, three 128-bit AXI4 masters, AXI4-Lite control, memory map and DMA path), validate it, and only then consider a bitstream.
