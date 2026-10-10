# ZCU102 Q8×Q8 PE16 DATAFLOW完整系统布局布线报告

状态：`DATAFLOW_FULL_SYSTEM_ROUTE_PASS / OFFLINE_BITSTREAM_NEXT`

## 结论

PE16 DATAFLOW核已接入完整ZynqMP PS、SmartConnect及DDR通道，并通过200 MHz布局布线。最终WNS为`+0.078 ns`、WHS为`+0.010 ns`、未布线网络为`0`。本阶段没有生成bitstream，也没有访问或改变ZCU102。

## 完整系统实现结果

| 指标 | DATAFLOW | 已上板PE16基线 | 变化 |
|---|---:|---:|---:|
| LUT | 25,966（9.47%） | 25,592 | +374 |
| FF | 30,473（5.56%） | 29,406 | +1,067 |
| BRAM tile | 419（45.94%） | 366.5 | +52.5 |
| DSP | 661（26.23%） | 656 | +5 |
| WNS | +0.078 ns | +0.067 ns | +0.011 ns |
| WHS | +0.010 ns | +0.011 ns | -0.001 ns |

DATAFLOW的主要硬件代价是52.5个额外BRAM tile，用于激活和输出FIFO；DSP只增加5个，LUT/FF增幅较小。全部资源仍低于80%冻结门槛。

## 接口兼容性

- WQ→HPC0、XQ→HPC1、WS/XS/Y→HP0的拓扑不变。
- 输出AXI仍为128 bit，控制地址仍为`0xA0000000 / 64 KiB`。
- 新旧IP的运行时寄存器头文件逐字节一致，SHA256均为
  `0810fe9a8a030fd2858dfd00bba2ea1ac7f3c9c1dcb6c30c2603934c36a0144d`。

## DRC与功耗边界

- 默认DRC：765项，Warning 621、Advisory 144、Error 0；均为DSP流水化或属性建议。
- 方法学：2项时钟Advisory（CLKC-40、CLKC-56），与已上板PE16基线相同。
- Vivado vectorless估算为`4.684 W`（Medium confidence）；PE16基线当前报告为`5.042 W`。由于未注入实测切换率，这个差值不能用于宣称省电，论文能效必须用板端功耗与实测吞吐量计算。

## 性能证据边界

当前只允许引用HLS排程的一阶投影：Language Prefill约`198.942 ms`、Vision Encoder约`293.048 ms`。它们不是DATAFLOW板端实测。Decoder投影不形成加速声明，因为已冻结的PE16 Decoder GEMV是memory/weight-stream-bound。

## 冻结证据

- 原始运行目录：`fpga_accel/q8q8_pe16_dataflow_zcu102_vivado/artifacts/runs/20260829T135117Z_impl`
- routed DCP SHA256：`8eda6c2deba32c331044a88009ff39bec6d3fd4f3b02b1d968b40248aa33efca`
- Vivado日志 SHA256：`eb9f8e50873ffa3b1f51bef83fd3dbf96cf5630b2c0d2dec904597f9eafa0880`
- 机器可读摘要：`fpga_accel/q8q8_pe16_dataflow_zcu102_vivado/artifacts/pe16_dataflow_vivado_summary.json`

## 下一门

在不接触板卡的前提下生成独立bitstream/XSA并校验内嵌bit一致性，然后复用既有运行时自检与三类microbenchmark测试包。任何PL配置、复位或板测仍需单独授权。
