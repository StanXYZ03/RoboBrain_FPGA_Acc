# ZCU102 Q8×Q8 PE16 DATAFLOW HLS门控

生成时间：2026-08-29T13:20:49.356511+00:00
状态：`DATAFLOW_HLS_GATE_PASS / DIRECTED_RTL_NEXT`

## 本阶段实现

- 三级并发流水：DDR连续读取激活 → PE16 Q8×Q8 GEMM → 128-bit结果写回。
- 片上FIFO容纳一个最大tile，允许load、compute、store跨tile重叠。
- 将1026周期的累加器清零融合到首个K-block，不改变后续累加顺序。

## 验证结果

- C testbench：9/9 PASS，最大绝对误差`3.05175781e-05`。
- HLS明确综合出DATAFLOW区域；主MAC循环II=1。
- 估算时钟：3.884 ns（约257.47 MHz），通过200 MHz门槛。
- 资源：BRAM 816/1824（44.74%），DSP 406/2520（16.11%），LUT 42110/274080（15.36%）。

## 冻结workload的保守排程模型

| workload | PE16串行/ms | DATAFLOW/ms | 降幅 | 证据边界 |
|---|---:|---:|---:|---|
| language_prefill | 210.695 | 166.829 | 20.82% | 投影 |
| vision_encoder | 408.822 | 253.031 | 38.11% | 投影 |
| decoder_gemv | 10.364 | 8.824 | 14.86% | 仅观察 |

模型将compute串行路径全部计入，并额外保留最后一次store；因此没有把理想完全重叠当作实测。decoder受权重流/内存限制，本阶段不宣称加速。

## 决策

全部冻结HLS门槛通过。下一步只做生成RTL的定向数值/控制仿真；在RTL和布局布线门槛通过前，不生成板端烧录结论。

机器可读证据：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/fpga_accel/q8q8_pe16_dataflow_hls/artifacts/pe16_dataflow_hls_summary.json`。
