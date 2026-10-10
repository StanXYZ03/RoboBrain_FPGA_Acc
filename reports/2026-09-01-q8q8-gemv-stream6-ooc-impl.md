# 六通道 Q8_0 Decoder GEMV：OOC 实现记录

## 结论

判定：`STREAM6_OOC_200MHZ_PASS / FULL_SYSTEM_INTEGRATION_NEXT`。

独立计算核在 XCZU9EG 上完成 Vivado OOC 综合、布局和布线。目标周期为 5.000 ns（200 MHz），后综合周期 2.777 ns，布线后周期 4.453 ns；WNS 为 +0.547 ns，WHS 为 +0.023 ns，未布通网络为 0。因此，HLS 阶段的时序风险在计算核 OOC 实现中已经消除。

## 实现级资源

| 资源 | 用量 | XCZU9EG 可用 | 占比 |
|---|---:|---:|---:|
| LUT | 19563 | 274080 | 7.14% |
| FF | 26227 | 548160 | 4.78% |
| DSP | 240 | 2520 | 9.52% |
| BRAM18K | 123 | 1824 | 6.74% |

## 判断边界

这证明六通道 GEMV 核单体可以在 200 MHz 下闭合，并留有较多逻辑和算术资源。OOC 环境没有连接 PS、SmartConnect、位宽转换器和 DDR 控制器；工具也明确提示 `ap_clk` 缺少全系统 `HD.CLK_SRC`，外部 AXI 端口缺少系统级位置。因此，这一结果不能替代全系统时序、DDR 并发带宽和板端 token/s 测量。

## 证据

- 机器可读摘要：`fpga_accel/q8q8_gemv_stream6_hls/artifacts/stream6_ooc_impl_20260901/summary.json`
- 冻结的实现报告与 IP：`fpga_accel/q8q8_gemv_stream6_hls/artifacts/stream6_ooc_impl_20260901/raw`
- 前置 C/RTL 协同仿真：`fpga_accel/q8q8_gemv_stream6_hls/artifacts/stream6_hls_gate_20260901/summary.json`

下一步是构建只含 PS、六路高性能端口、必要互连和该 IP 的最小系统工程，先做地址/位宽/时钟静态检查与仿真，再做全系统实现；此阶段仍不需要烧板。
