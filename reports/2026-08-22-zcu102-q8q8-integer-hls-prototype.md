# ZCU102 Q8_0×Q8_0 整数矩阵乘原型

日期：2026-08-22  
器件：XCZU9EG-FFVB1156-2-E  
工具：Vitis HLS 2023.2  
状态：C 仿真、HLS 综合、Verilog C/RTL 联合仿真通过；未生成 bitstream，未烧录。

## 1. 语义依据

固定版本 llama.cpp（commit `178a6c44937154dc4c4eff0d166f4a044c4fceba`）为
`GGML_TYPE_Q8_0` 指定：

- `from_float = quantize_row_q8_0`；
- `vec_dot = ggml_vec_dot_q8_0_q8_0`；
- `vec_dot_type = GGML_TYPE_Q8_0`。

标量参考语义为：

`sum += dot_i8x32(weight.qs, activation.qs) * weight.d * activation.d`

激活动态量化采用每32元素绝对值最大值，`d=amax/127`、`q=round(x/d)`，scale以
FP16存储。该原型复现现有GGML路径，不改模型权重，也不额外降低已部署Q8_0基线精度。

## 2. 架构

- 权重和激活均拆成对齐的256-bit int8面与16-bit FP16 scale面；
- `TM=16`、`TN=64`、`K-block=32`；
- 8个并行N方向PE，每个PE包含32路INT8乘法和整数平衡归约树；
- 每block使用INT32范围点积，乘两个scale后累加至FP32输出；
- 五个AXI master分别访问权重q/scale、激活q/scale和输出；
- 激活动态量化器本版作为上游，testbench严格复现GGML量化规则。

## 3. 正确性

C仿真全部通过：小矩阵、GEMV、M/N tile边界、`K=2560`、`K=1024`、零block、
FP16 subnormal scale、int8端点以及非法维度。相对独立GGML语义参考的最大绝对误差
为`2.47955322e-05`，来自HLS C模型与参考half转换的浮点仿真差异。

XSim Verilog C/RTL联合仿真5/5事务通过：

- RTL latency：79 / 3316 / 7174 cycles（min/avg/max）；
- 总执行：16,524 cycles；
- 最终状态：`C/RTL co-simulation finished: PASS`。

## 4. 综合结果与FP32原型对比

| 指标 | Q8权重×FP32激活 | Q8×Q8整数版 | 变化 |
|---|---:|---:|---:|
| 并行PE | 4 | 8 | 2倍 |
| K-block最低周期 | 982 | 378 | 2.60倍加速 |
| DSP | 655 | 179 | -72.7% |
| BRAM18K | 80 | 148 | +85.0% |
| LUT | 58,902 | 24,942 | -57.7% |
| FF | 81,202 | 30,238 | -62.8% |
| HLS估算周期 | 3.650 ns | 4.185 ns | 均短于5 ns时钟周期 |

整数计算主循环达到`II=1`。BRAM增加主要来自新增的256-bit激活量化AXI master及其
接口缓冲，当前仍只占ZU9EG总BRAM约8.1%。HLS对4.185 ns路径给出相对3.650 ns有效
预算的警告；它小于5 ns，但200 MHz是否真正闭合必须由Vivado布局布线确认。

曾实验将8个浮点累加器强制映射为更深的DSP流水线，时序估算改善，但跨8个N group
的累加依赖使II从1退化到2，因此已撤销；吞吐优先的II=1版本作为当前基线。

## 5. 代表矩阵的综合级估算

采用200 MHz和正常scale的最低调度：

`cycles_tile = 2065 + (K/32) * 378`

| 矩阵MxNxK | 整数版 | FP32原型 | 内核调度加速 |
|---|---:|---:|---:|
| 512×9728×2560 | 0.786 s | 1.961 s | 2.50倍 |
| 4320×4096×1024 | 1.224 s | 2.893 s | 2.37倍 |
| 1×9728×2560 | 24.6 ms | 61.3 ms | 2.50倍 |

以上是单次矩阵乘的HLS调度估算，不包含DDR争用、驱动、DMA、布局布线和激活动态
量化时间，不能作为板上或整模型实测结果。

## 6. 新瓶颈与下一步

正常scale下每K-block约由激活加载32周期、权重加载192周期、计算150周期及控制开销
组成。整数化后加载约占调度周期59%，瓶颈已经从昂贵FP32计算明显向数据供给迁移。

下一版应：

1. 使用ping-pong tile buffer和HLS DATAFLOW重叠权重/激活加载与计算；
2. 合并或缩减AXI接口FIFO，降低148个BRAM开销；
3. 集成流式FP32→Q8_0激活量化器，并与当前预量化输入模式对比；
4. 为M=1增加专用GEMV调度；
5. 再进行Vivado block design、布局布线和AXI性能仿真，通过后才上板。
