# 六通道 Q8_0 Decoder GEMV：HLS 与 RTL 协同仿真记录

## 结论

判定：`STREAM6_RTL_COSIM_PASS / OOC_TIMING_NEXT`。

- C/RTL 协同仿真通过，9/9 个 RTL 事务完成；6 个数值用例和 1 个非法维度用例均通过，最大绝对误差为 `0`。
- 六个权重 AXI 端口均推断出连续 256-bit burst；权重读取循环 II=1，九相计算循环 II=1。
- HLS 估算周期 `4.413 ns`（约 `226.60 MHz`），低于 5 ns 名义目标；但在 1.35 ns 时钟不确定度下仍报告 Timing 风险，必须继续做导出 IP 后的 OOC/实现时序验证。
- 资源估算：BRAM18K 395（21.66%）、DSP 144（5.71%）、LUT 37763（13.78%）、FF 32194（5.87%）。

## 数据组织与吞吐含义

每个端口、每个 K-block 使用 9 个 256-bit 字：1 个尺度头和 8 个 Q8 权重块，对应 288 B 传输承载 272 B 原生 Q8_0 数据，打包效率 94.44%。200 MHz 下，HLS 内部六端口请求上限为 38.4 GB/s，整数计算能力为 34.13 GMAC/s；这两个数都不是 ZCU102 DDR 实测值，接入 PS 端口后会受到位宽转换、仲裁和 DDR 总带宽限制。

按已冻结的模型权重流量，每生成 1 token 需读取约 4.274 GB 权重。DDR4-2133 线速只给出 3.99 token/s 的理论上界；按线速 85% 的规划情形为 3.39 token/s。要达到 5 token/s，仍需减少约 32.1% 的每 token 权重流量。

## 证据

- 机器可读摘要：`fpga_accel/q8q8_gemv_stream6_hls/artifacts/stream6_hls_gate_20260901/summary.json`
- RTL 协同仿真日志：`fpga_accel/q8q8_gemv_stream6_hls/artifacts/20260901T151232Z_vitis_hls_cosim.log`
- HLS 综合报告：`fpga_accel/q8q8_gemv_stream6_hls/build/q8q8_gemv_stream6_prj/solution1/syn/report/csynth.rpt`
- 异构边界与 Decoder 带宽约束：`fpga_accel/heterogeneous_partition/artifacts/partition_gate_20260901/summary.json`

## 尚未完成

尚未进行导出 IP 的 OOC 综合、全系统布线、真实 DDR 并发带宽测试和板端推理集成，因此不能据此声称板端 token/s 或端到端加速比。下一门槛是导出 IP，并在 200 MHz 目标下获得实现级时序和资源结果。
