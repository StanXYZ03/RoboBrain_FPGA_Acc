# ZCU102 Q8×Q8 DATAFLOW HLS 原型验证

日期：2026-08-22  
状态：C 仿真、HLS 综合、Verilog RTL 仿真及 RTL 输出后校验通过；未生成 bitstream，未烧录板卡。

## 1. 本步目标

在不改变已验证 Q8_0×Q8_0 数值语义的前提下，将激活读取、权重读取和 tile 计算由串行改为并行 DATAFLOW，利用片上 FIFO 实现两级弹性缓冲，使 DDR 读取与计算重叠。

计算仍严格对应固定 llama.cpp/GGML Q8_0 路径：

```text
sum += dot_i8x32(weight.qs, activation.qs) * weight.d * activation.d
```

- 权重、激活量化值：每块 32×INT8，打包为 256 bit。
- 权重、激活 scale：各为 FP16。
- 块级乘法和输出累加：FP32。
- tile：TM=16、TN=64；N 方向 8 PE。

## 2. 实现

HLS 实际提取 3 个并行进程：

1. 激活读取器：独立 AXI q/scale 端口，输出 288-bit block stream。
2. 权重读取器：独立 AXI q/scale 端口，输出 288-bit block stream。
3. 计算器：同时接收两条 stream，完成 32 路 INT8 点积、scale 融合和 FP32 累加。

FIFO 深度分别为 32 和 128，可缓存两套 activation/weight tile。FP16 subnormal 归一化由可变次数循环改成等价的前导零优先编码器，使两个读取循环均达到 II=1。

## 3. 验证结果

### 3.1 C 仿真

六组测试全部通过：small tile、GEMV、M/N tile 边界、K=2560、K=1024、非法维度；测试数据包含 FP16 最小 subnormal scale。最大绝对误差为 `2.47955322e-05`。

### 3.2 综合

| 指标 | DATAFLOW 版 | 非重叠整数基线 | 变化 |
|---|---:|---:|---:|
| K-block 计算周期 | 220 | 378 | -41.8% |
| DATAFLOW 最小启动间隔 | 222 | 不适用 | — |
| DSP | 179 | 179 | 0 |
| BRAM18K | 180 | 148 | +32 |
| LUT | 24,764 | 24,942 | -178 |
| FF | 31,000 | 30,238 | +762 |
| HLS 估计周期 | 4.185 ns | 4.185 ns | 0 |

阶段明细：

- activation reader：II=1；首个 K-block 约 33 cycles。
- weight reader：II=1；首个 K-block 79 cycles。
- compute ingest：66 cycles。
- 8-PE MAC/accumulate：150 cycles，内部循环 II=1。
- compute 每 K-block：220 cycles。
- 完整 DATAFLOW：`77 + 220 × K_blocks` cycles；最小调用 interval 为 222 cycles。

HLS 给出估计 Fmax 238.95 MHz。需注意：4.185 ns 小于名义 5 ns 周期，但大于考虑 1.35 ns uncertainty 后的 3.65 ns 内部预算，因此 200 MHz 是否真正闭合必须以后续 Vivado place-and-route 报告为准。

### 3.3 RTL 仿真

- XSim 完成 5/5 个 RTL transaction，仿真结束时间 62,672,500 ps，无 DATAFLOW 死锁。
- HLS 自动生成的 C post-check 对 small、M 边界、N 边界和非法输入逐项比较，全部 PASS。
- Vivado 2023.2 原生流程在加载 DATAFLOW GUI 波形时出现 `unexpected exception`；移走只用于 GUI 的 `.wcfg` 后运行同一 elaborated RTL snapshot 可通过。此问题不涉及 DUT 修改，工程中已保留专用 workaround 脚本。

## 4. 200 MHz 周期投影

综合报告给出的每个输出 tile 周期为：

```text
C_tile = 2143 + 220 × (K/32)
T = ceil(M/16) × ceil(N/64) × C_tile / 200 MHz
```

| 代表矩阵 M×N×K | DATAFLOW 投影 | 非重叠整数基线 | 加速比 |
|---|---:|---:|---:|
| 512×9728×2560 | 0.480 s | 0.786 s | 1.64× |
| 4320×4096×1024 | 0.793 s | 1.224 s | 1.54× |
| 1×9728×2560 | 15.0 ms | 24.6 ms | 1.64× |

以上只计算本 HLS kernel 的调度周期，不包含 PS 调用、真实 DDR 争用、量化、RoPE、softmax 或其他模型算子，不是整模型或板上实测延迟。

## 5. 结论与下一步

本步证明 load/compute overlap 有效：在 DSP 数不增加的情况下，K-block 调度周期减少 41.8%，代价为 32 个 BRAM18K。下一步应先做 Vivado IP 集成与 place-and-route，确认 200 MHz 时序和 HP/HPC 端口带宽，再进行上板微基准；在此之前不应把 0.480 s 等投影当作板上结果。
