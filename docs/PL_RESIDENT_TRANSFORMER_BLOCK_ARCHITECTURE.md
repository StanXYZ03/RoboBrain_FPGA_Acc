# PL驻留Transformer Block架构规格（阶段A）

日期：2026-08-30  
状态：向量IP与双IP完整系统均已布线通过；真实模型数值门待完成  
范围：RoboBrain2.5-4B Q8_0，ZCU102，现有PE16 DATAFLOW GEMM

## 1. 实测驱动

当前单算子桥接已证明：

- M=512/1080/4320大矩阵DATAFLOW kernel为174.545/368.889/252.071 ms。
- A53四核NEON激活量化为2.701/5.474/8.644 ms。
- 26.46 MB LLM FFN权重拆分并写入真实UIO为12.038 ms。
- 19.92/37.98/58.72 MB F32输出从UIO回PS即使用NEON四核仍需
  48.957/83.899/150.351 ms。

因此，单个`MUL_MAT`通过PS往返的后端可以用于bring-up，但不应成为最终结构。阶段A目标是让
FFN子图的中间张量停留在PL可访问DDR，仅在子图边界交付结果。

既有Grounding profile中，Vision Prefill的SWIGLU与RMS_NORM仅占阶段时间约0.2185%和
0.1528%，Vision Encoder的GELU与NORM约占0.6044%和0.1353%。因此新增向量核的首要目的
不是单独加速这些小算子，而是消除F32中间张量回PS；论文口径必须报告融合前后数据流量与
子图端到端时间，不能用向量算子自身的小延迟夸大收益。

## 2. 两类驻留子图

### 2.1 LLM SwiGLU FFN

```text
F32 residual/input
  → RMSNorm
  → F32→Q8_0
  → Gate GEMM (N=9728)
  → Up GEMM   (N=9728)
  → SiLU(gate) × up
  → F32→Q8_0
  → Down GEMM (N=2560, K=9728)
  → residual add
  → F32 output
```

必须新增的向量操作：RMSNorm、SiLU、逐元素乘、残差加、block-wise Q8_0量化。

### 2.2 Vision GELU FFN

```text
F32 residual/input (D=1024)
  → LayerNorm
  → F32→Q8_0
  → FC1 GEMM (N=4096)
  → GELU
  → F32→Q8_0
  → FC2 GEMM (N=1024, K=4096)
  → residual add
  → F32 output
```

必须新增的向量操作：LayerNorm、GELU、残差加、block-wise Q8_0量化。

## 3. Block级活跃区间感知分块

现有auto chunk只保证单个GEMM装入64 MiB，因此得到LLM `M=976`、Vision `M=3584`。
完整FFN同时存活gate/up或FC1中间结果、残差、Q8输入、输出和当前权重，不能沿用该上限。

首版规划器对每个候选`chunk_m`计算：

```text
arena_required = max_t Σ buffer_bytes(buffer在时刻t存活)
```

而不是把所有缓冲区简单相加，也不是只看单个GEMM。规划器必须输出每个buffer的birth/death、
可复用offset和峰值时刻，并拒绝任何物理区间重叠错误。

64 MiB下的初始保守搜索范围：

- LLM SwiGLU：`chunk_m ∈ {128, 256, 384, 400, 512}`。
- Vision GELU：`chunk_m ∈ {512, 1024, 1536, 2048, 2560}`。

这些只是待验证候选，不是已证明最优值。选择条件为端到端子图时间最短，并满足全量数值门。

第一版精确搜索与区间复用属性测试得到：LLM SwiGLU `chunk_m=432`（1080行分3块，
峰值65,678,848 B），Vision GELU `chunk_m=2896`（4320行分2块，峰值66,917,376 B）。
二者均为容量最大值，不代表时间最优值；后续仍需扫描较小候选以考虑kernel效率和调度开销。

## 4. 权重与输出策略

- 权重：阶段A采用已测四核NEON快速staging；每个不变权重只在其GEMM前写入一次，跨M chunk复用。
- 输出：中间结果不回PS；只有FFN最终F32输出允许回读，后续若残差/下一Norm也驻留则继续延后。
- Decoder M=1：不进入当前PL路径。
- 原生34-byte Q8_0 PL前端：保留为后续优化，不是阶段A前置门。
- 线程：量化/staging/readback统一由backend线程预算管理，禁止每个节点无约束创建4线程。

## 5. 控制与ABI

PL不接收General/Pointing/Grounding/Trajectory任务名。阶段A描述符只包含：

```text
subgraph_type, M, hidden, ffn, chunk_m,
input/output/residual addresses,
gate/up/down or fc1/fc2 weight addresses,
scratch arena address/size,
epsilon, activation_mode
```

首版可以由PS顺序启动GEMM与向量核；待数值和性能通过后再引入PL命令队列。所有入口默认关闭，
失败时整段子图回退CPU，不能在半个Block后混合回退。

## 6. 验证顺序

1. 纯C golden：两类FFN、边界M、NaN/零方差、Q8舍入。
2. 活跃区间规划器：区间不重叠、64 MiB峰值、随机M属性测试。
3. 向量核HLS C simulation与独立testbench。
4. GEMM+向量核软件模拟端到端，逐元素对照CPU reference。
5. RTL定向仿真、综合、完整系统时序。
6. 上板单FFN子图：kernel、staging、向量操作、最终一次回读分项计时。
7. 首个真实Transformer Block对照；通过后才进入默认关闭的GGML backend hook。

阶段A验收：FFN子图全量数值通过，且相对“每个MUL_MAT都回PS”的快速桥接路径有可重复的
端到端收益。未达到该门槛时不得宣称Block级或模型端到端加速。

## 7. 2026-08-31向量IP实现状态

- `fpga_accel/pl_resident_ffn_hls`已实现统一模式核，3个AXI主口，任务字段为0。
- 单算子与两类小规模完整FFN组合测试9/9通过；LLM/Vision组合最大误差分别为
  0.000711/0.002926。
- XCZU9EG独立布局布线通过：4.198 ns、WNS +0.802 ns、WHS +0.023 ns、未布线0；
  26,685 LUT、47,063 FF、154 DSP、64 BRAM Tile。
- 该结果只完成原验证顺序第3、4步及独立IP时序，不是第5步完整系统结果，更不是第6、7步
  板端/真实Block结果。
- 详细证据：`reports/2026-08-31-pl-resident-ffn-vector-ip.md`。

## 8. 2026-08-31双IP完整系统实现状态

- PE16 DATAFLOW GEMM与统一向量核已集成到同一ZynqMP系统；GEMM/向量控制地址固定为
  `0xA0000000`/`0xA0010000`。
- 200 MHz完整布局布线通过：WNS +0.060 ns、WHS +0.009 ns、未布线0，DRC与方法学
  Error均为0。
- 完整系统资源为58,445 LUT、88,156 FF、483 BRAM Tile、815 DSP，最高占用为BRAM
  52.96%，未触发80%资源门。
- 这完成验证顺序第5步的完整系统综合与布线部分；尚未完成冻结真实张量数值门，也未执行
  第6、7步上板与真实Block验证。
- Vivado vectorless功耗5.942 W仅作布线审计，不是板级实测或能效证据。
- 详细证据：`reports/2026-08-31-pl-resident-ffn-full-system.md`。
