# ZCU102 Q8×Q8 PE16 RTL验证报告

生成时间：2026-08-27T07:06:20.873517+00:00
状态：`COMPUTE_CONTROL_RTL_PASS / TOP_AXI_COSIM_RESOURCE_STOP / BITSTREAM_GENERATION_NEXT`

## 结论边界

- 全尺寸C数值测试7/7通过，最大绝对误差`2.50339508e-05`。
- Vitis HLS顶层AXI cosim没有完成RTL阶段，不能记为通过。
- 生成的`compute_cached_tile`控制RTL完成定向XSim；算术单元使用零数据行为stub，因此本项只证明控制、地址和有效位调度。
- PE16板端数值和性能尚未测量。

## 顶层cosim工具限制

- HLS的C侧RTL smoke 5/5通过，小容量综合保持II=1。
- Vitis HLS 2023.2生成5个AXI主端口模型，仿真目录约30.74 GiB。
- `xelab`仅解析首个模型时RSS采样已约20.5 GiB；为避免OOM主动停止。
- 原始日志：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/fpga_accel/q8q8_pe16_hls/artifacts/cosim_small.log`。

## 定向计算控制RTL

- 测试shape：`M=3, N=64, K=64`，16个PE bank。
- 每个bank写24次，覆盖12个合法地址，每个地址写2次。
- XSim完成时间：792500 ps。
- 最终原始目录：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/fpga_accel/q8q8_pe16_hls/artifacts/compute_rtl_smoke/20260827T070133Z`。
- 前两次失败分别用于补齐flow-control RTL依赖和修正`ap_start/ap_ready`握手，日志保留且未覆盖。

## 决策

PE16的新M-row交错控制及16-bank累加地址映射通过生成RTL的定向验证；数值正确性由C testbench覆盖。顶层AXI cosim因工具资源风险停止。结合独立完整系统报告已经通过的200 MHz routed门控，下一步可以生成bitstream，再以板端self-test补足真实数值与性能证据；不得将本报告写成PE16板端通过。

机器可读证据：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/fpga_accel/q8q8_pe16_hls/artifacts/pe16_rtl_validation_summary.json`。
