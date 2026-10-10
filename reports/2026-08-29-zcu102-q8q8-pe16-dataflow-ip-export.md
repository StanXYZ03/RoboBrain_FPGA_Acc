# ZCU102 Q8×Q8 PE16 DATAFLOW IP导出与OOC实现报告

状态：`DATAFLOW_IP_OOC_ROUTE_PASS / FULL_SYSTEM_ROUTE_NEXT`

## 结论

- 已导出独立IP `robobrain.local:hls:q8q8_pe16_dataflow:1.0`，未覆盖板端PE16基线。
- IP在ZCU102器件上完成200 MHz OOC布局布线：WNS `+0.015 ns`、WHS `+0.028 ns`、未布线网络`0`。
- OOC资源为21,534 LUT、22,636 FF、661 DSP、838 BRAM18K等价资源。
- 本阶段未生成bitstream，未访问、复位或烧录ZCU102。

## 实现证据

| 项目 | 结果 |
|---|---:|
| 目标频率 | 200.000 MHz |
| 实现后关键周期 | 4.985 ns |
| WNS / WHS | +0.015 / +0.028 ns |
| TNS / THS | 0 / 0 ns |
| 未布线网络 | 0 |
| LUT / FF | 21,534 / 22,636 |
| DSP | 661 / 2520（26.23%） |
| BRAM tile | 419 / 912（45.94%） |

默认DRC共750项：606项Warning、144项Advisory、0项Error，内容为DSP流水化建议和一项无可路由负载提示。方法学的1000项Warning均为OOC外部AXI/控制端口缺少I/O delay（TIMING-18），不是完整系统签核结果。

## 功耗边界

Vivado vectorless OOC估算为`1.478 W`（动态`0.850 W`、静态`0.628 W`，Medium confidence）。该值不包含完整PS与系统互连，也没有使用板端活动率，不能作为ZCU102整板功耗或论文能效数据。

## 冻结产物

- IP：`fpga_accel/q8q8_pe16_dataflow_hls/artifacts/q8q8_pe16_dataflow_ip.zip`
- IP SHA256：`309a4bad50a4d377eedfa63a1a19fa9b1007a9ad81c9b355159868631306f723`
- 导出日志：`fpga_accel/q8q8_pe16_dataflow_hls/artifacts/20260829T132902Z_vitis_hls_export_impl.log`
- 机器可读摘要：`fpga_accel/q8q8_pe16_dataflow_hls/artifacts/pe16_dataflow_ip_export_summary.json`

## 下一门

建立独立完整系统分支，将该IP接入与PE16基线相同的ZynqMP PS、HPC0/HPC1/HP0、AXI-Lite及200 MHz时钟拓扑，重新执行完整系统validate、综合、布局布线、DRC和功耗检查。只有完整系统通过后才考虑生成bitstream；当前不执行上板。
