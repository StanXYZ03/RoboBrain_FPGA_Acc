# 整数 Selector 最小 ZCU102 Zynq MPSoC Block Design 验证

- 决策：**INTEGER_SELECTOR_MINIMAL_ZYNQ_BD_PASS**
- 工程日期：2026-10-03；工具：Vivado 2023.2（Tool Version Limit 2023.10）
- part：`xczu9eg-ffvb1156-2-e`；board part：`xilinx.com:zcu102:part0:3.4`（available）
- Selector VLNV：`robobrain.local:hls:integer_selector_pipeline_compact:1.0`
- 工程记录：`fpga_accel/integer_selector_minimal_zynq_bd/`；证据：`artifacts/token_selector/2026-10-03-integer-selector-minimal-zynq-bd/`

## 门（Gates）

| gate | result |
|---|---|
| protected_files_unchanged | PASS |
| exact_part_and_board_recorded | PASS |
| selector_vlnv_instantiated | PASS |
| selector_interface_inventory_exact | PASS |
| three_master_to_ps_ddr_path | PASS |
| ps_master_to_control_and_base | PASS |
| clocks_resets_connected | PASS |
| pl_clock_200mhz | PASS |
| three_ddr_segments_nonempty | PASS |
| validate_bd_design_no_error | PASS |
| output_products_and_wrapper | PASS |
| no_bitstream_no_board | PASS |

## AXI 接口清单

| pin | mode | protocol | data width | freq (Hz) |
|---|---|---|---|---|
| `/integer_selector_pipeline_compact_0/m_axi_input_arena` | Master | AXI4 | 128 | 199998001 |
| `/integer_selector_pipeline_compact_0/m_axi_output_arena` | Master | AXI4 | 128 | 199998001 |
| `/integer_selector_pipeline_compact_0/m_axi_wv_q` | Master | AXI4 | 128 | 199998001 |
| `/integer_selector_pipeline_compact_0/s_axi_control` | Slave | AXI4LITE | 32 | 199998001 |
| `/smartconnect_control/M00_AXI` | Master | AXI4LITE | 32 | 199998001 |
| `/smartconnect_control/S00_AXI` | Slave | AXI4 | 128 | 199998001 |
| `/smartconnect_ddr/M00_AXI` | Master | AXI4 | 128 | 199998001 |
| `/smartconnect_ddr/S00_AXI` | Slave | AXI4 | 128 | 199998001 |
| `/smartconnect_ddr/S01_AXI` | Slave | AXI4 | 128 | 199998001 |
| `/smartconnect_ddr/S02_AXI` | Slave | AXI4 | 128 | 199998001 |
| `/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD` | Master | AXI4 | 128 | 199998001 |
| `/zynq_ultra_ps_e_0/S_AXI_HP0_FPD` | Slave | AXI4 | 128 | 199998001 |

## AXI 连接图

- `selector_master_nets` -> {'m_axi_input_arena': ['/integer_selector_pipeline_compact_0/m_axi_input_arena', '/smartconnect_ddr/S00_AXI'], 'm_axi_wv_q': ['/integer_selector_pipeline_compact_0/m_axi_wv_q', '/smartconnect_ddr/S01_AXI'], 'm_axi_output_arena': ['/integer_selector_pipeline_compact_0/m_axi_output_arena', '/smartconnect_ddr/S02_AXI']}
- `smartconnect_ddr_M00_AXI` -> ['/smartconnect_ddr/M00_AXI', '/zynq_ultra_ps_e_0/S_AXI_HP0_FPD']
- `smartconnect_control_M00_AXI` -> ['/smartconnect_control/M00_AXI', '/integer_selector_pipeline_compact_0/s_axi_control']
- `ps_M_AXI_HPM0_FPD` -> ['/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD', '/smartconnect_control/S00_AXI']

## 时钟 / 复位

- `ap_clk` = 199.998 MHz（目标 200 MHz；PS `pl_clk0`，IOPLL）
- `proc_sys_reset` `C_EXT_RESET_HIGH` = 0（0 = 外部复位低有效）
- 复位连接：{'ext_reset_in_driven_by': ['/proc_sys_reset_0/ext_reset_in'], 'peripheral_aresetn_members': ['/proc_sys_reset_0_peripheral_aresetn'], 'ap_rst_n_net': ['/proc_sys_reset_0_peripheral_aresetn']}

## 地址映射

- AXI-Lite control 基地址：`0x00A0000000`
- DDR 段（每个 Selector master）：
  - `m_axi_input_arena`：`0x0000000000000000` .. range `0x0000000080000000`（/integer_selector_pipeline_compact_0/Data_m_axi_input_arena/SEG_zynq_ultra_ps_e_0_HP0_DDR_LOW）
  - `m_axi_wv_q`：`0x0000000000000000` .. range `0x0000000080000000`（/integer_selector_pipeline_compact_0/Data_m_axi_wv_q/SEG_zynq_ultra_ps_e_0_HP0_DDR_LOW）
  - `m_axi_output_arena`：`0x0000000000000000` .. range `0x0000000080000000`（/integer_selector_pipeline_compact_0/Data_m_axi_output_arena/SEG_zynq_ultra_ps_e_0_HP0_DDR_LOW）

## 校验 / 综合

- `validate_bd_design` rc = 0，ERROR = 0，CRITICAL WARNING = 0，设计 WARNING = 15（另有 17 条证据导出的 BD 5-234）
- HDL wrapper：/tmp/robobrain_integer_selector_bd/vivado/project/integer_selector_minimal_zynq_bd.gen/sources_1/bd/integer_selector_minimal_zynq_bd/hdl/integer_selector_minimal_zynq_bd_wrapper.v
- 综合：OK（wall 284 s）
- 综合资源（post-synth，逻辑 + PS）：{"CLB_LUTs": {"used": 32514, "fixed": 0, "prohibited": 0, "available": 274080, "pct": 11.86}, "CLB_Registers": {"used": 37264, "fixed": 0, "prohibited": 0, "available": 548160, "pct": 6.8}, "Block_RAM_Tile": {"used": 122.5, "fixed": 0, "prohibited": 0, "available": 912, "pct": 13.43}, "DSPs": {"used": 210, "fixed": 0, "prohibited": 0, "available": 2520, "pct": 8.33}, "URAM": null, "PS8": {"used": 1, "fixed": 0, "prohibited": 0, "available": 1, "pct": 100.0}}
- 综合时序（post-synth，未实现/未布线）：{"WNS_ns": "1.184", "TNS_ns": "0.000", "TNS_failing": "0", "TNS_total": "145506", "WHS_ns": "-0.096", "THS_ns": "-1958.720", "THS_failing": "57404", "THS_total": "145506", "WPWS_ns": "1.000", "TPWS_ns": "0.000", "TPWS_failing": "0", "TPWS_total": "42319"}
- 时钟约束报告：`vivado/clocks_post_synth_bd.rpt`

## 受限说明

- No Ethernet/Jetson software path is implemented.
- No PS software (bare-metal/Linux) driver is implemented.
- No DDR traffic contention / arbitration bandwidth is measured.
- No full-system timing sign-off (no implementation, no bitstream).
- No bitstream is generated and the board is never accessed.
- No board measurement is performed.
- No model end-to-end latency is measured.
- No task-accuracy claim is made.

**下一步**：Implement the PS-side bare-metal/Linux register-and-DDR test harness using static JPG-derived fixture buffers, before any Jetson Ethernet integration.
