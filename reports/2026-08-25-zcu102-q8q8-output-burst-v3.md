# ZCU102 Q8×Q8 v3：128-bit 输出突发阶段报告

日期：2026-08-25  
状态：`STAGE_PASS / BOARD_PASS / NUMERIC_PASS / TIMING_PASS`

## 1. 阶段结论

v3 将 v2 的逐 FP32 标量输出写改为 `128-bit` 打包、每次最多 `16 beat`
的 AXI4 burst。真实 ZCU102 上，目标 workload `vision_encoder` 从
`1,404.381 ms` 降到 `628.165 ms`，耗时降低 **55.27%（2.236×）**，超过
“至少降低 25%”的阶段门槛。三类 workload 的板端数值抽检均为
`max_abs=0`。

该结果是矩阵计算核的板端 microbenchmark，不等同于完整 RoboBrain 图片推理
已达到 0.628 s；完整模型尚未把所有层和算子接入该 PL runtime。

## 2. 实现改动

- HLS 输出端口由 32-bit FP32 标量改为 128-bit，每 beat 打包 4 个 FP32。
- 每个 64 列输出 tile 合并成 4 个、每个 16 beat 的 AXI burst。
- DDR 中输出仍保持 row-major FP32 布局，PS runtime 接口和参考结果不变。
- 为避免尾部 read-modify-write，当前要求 `N % 4 == 0`；非法尺寸明确拒绝。
- PS/DDR 拓扑不变：WQ→HPC0、XQ→HPC1、scale/output→HP0，PL 200 MHz。
- UART shell 保留；JTAG 启动只写易失 RAM/PL，不写 SD 或 QSPI。

## 3. 仿真、综合和实现门控

| 门控 | 结果 |
|---|---|
| C testbench | 6/6 PASS，最大绝对误差 `2.503e-05` |
| 定向 store RTL/XSim testbench | PASS，`width=128`、`rows=3`、`words=3` |
| HLS 估算 | Fmax `238.95 MHz` |
| 完整系统实现 | `199.980 MHz`，WNS `+0.129 ns`，WHS `+0.009 ns` |
| 路由完整性 | unrouted nets `0`，bitstream generated |
| 板端 UIO 自检 | `M=2,N=4,K=32`，8 个输出全部正确，`max_abs=0` |

说明：Vitis HLS 自动顶层 cosim 因 2023.2 工具生成异常巨大的 AXI 仿真模型
未作为通过项；这里不声称完整顶层自动 cosim 通过。上板前采用了 C testbench
和对本次改动关键模块的定向 RTL/XSim testbench，随后才生成并加载位流。

实现后系统资源：18,048 LUT、21,731 FF、314.5 BRAM tile、320 DSP；这是完整
PS+互连+加速核设计的 Vivado routed 结果。vectorless power report 为 4.634 W、
confidence=Medium，仅作工具估计，不能当作板卡实测功耗。

## 4. 板端性能

三次重复的 kernel wall time：

| workload | v2 | v3 | 耗时降低 | 加速比 | v3 有效性能 |
|---|---:|---:|---:|---:|---:|
| Language prefill | 586.339 ms | 368.570 ms | 37.14% | 1.591× | 69.190 GOPS |
| Vision encoder | 1,404.381 ms | 628.165 ms | **55.27%** | **2.236×** | 57.690 GOPS |
| Decoder GEMV | 19.664 ms | 18.511 ms | 5.86% | 1.062× | 2.691 GOPS |

vision 三次总 kernel time 分别为 `628.157 / 628.182 / 628.156 ms`，极差仅
`0.026 ms`；每个 chunk 的抽样均 `max_abs=0`。

## 5. APM 证据与新瓶颈

vision 的 HP0 输出写入量保持 `70,778,880 B`，但真实 AXI write transaction
从 v2 的 `17,694,720` 降到 v3 的 `276,480`，恰好减少 **64×**。v3 每事务
平均 `256 B`，即 16 个 128-bit beat；这直接证明性能提升来自预期的输出合并，
不是测试规模变化。

v3 latency profile：

| 路径 | 平均读/写延迟 | 最大读延迟 |
|---|---:|---:|
| CCI→DDR slot 1 | 83.30 cycle | 427 cycle |
| CCI→DDR slot 2 | 81.36 cycle | 414 cycle |
| HP0 read | 55.14 cycle | — |
| HP0 write | 17.00 cycle/transaction | — |

vision 实测 `628.165 ms`，当前排程模型为 `585.423 ms`，实测仅为模型的
`1.073×`；v2 为 `2.399×`。标量写协议开销已基本消除，下一主瓶颈转为
**计算/排程吞吐**，不是 DDR 峰值带宽。

Decoder 提升很小符合预期：`M=1` 时输出只有 38,912 B，主要耗时仍是约
26.9 MB 权重/scale 的流式读取，属于 memory-bound 路径。

## 6. 下一阶段

下一阶段不再继续扩大输出 burst，而以 vision/prefill 的计算阵列为目标：

1. 评估 `PE_N: 8→16` 的 DSP/BRAM、200 MHz 时序和理论吞吐；保留同一数值门控。
2. 若单纯加 PE 受存储端口或 BRAM banking 限制，则实现 load/compute/store
   双缓冲 `DATAFLOW`，用 APM 和 kernel wall time区分收益来源。
3. Decoder 保持独立 GEMV 路线，不用高 M 的 GEMM 优化结论代替。
4. 完成 C/RTL testbench、实现时序后才再次上板；最终接入真实 Transformer
   线性层 runtime，而不是直接将 microbenchmark 当作端到端模型成果。

## 7. 可复现产物

- HLS：`fpga_accel/q8q8_output_burst_hls/`
- 系统：`fpga_accel/q8q8_v3_zcu102_vivado/`
- runtime：`fpga_accel/q8q8_runtime/`
- UART 原始日志：`reports/2026-08-25-zcu102-q8q8-v3-uart.log`
- directed RTL log：
  `fpga_accel/q8q8_output_burst_hls/artifacts/q8q8_output_burst_store_rtl_xsim.log`

SHA-256：

- bitstream：`76c4e63ee04d1220c1eebee49631400ed1a9dae7f7b2f324291b15e137033fab`
- HLS IP：`f4b8a955649557b54ed528e5397bfbf588306470d652f58a7754371791db3ebe`
- ARM64 benchmark：`4a2657681ec02263e5c94d5ca7a11c17cb2e06a8e86fe28b60a69111d29d96de`
- UART log：`c1fcb60cbfb33791cd2ba5e21fd940521ed13329ae6ed59da20bf0f48d5b693f`

最终板态：v3 PL 已配置，accelerator control=`0x00000004`（idle），UART shell
alive。当前最小 RAM Linux 未枚举 thermal zone，因此本轮没有伪造温度数据。
