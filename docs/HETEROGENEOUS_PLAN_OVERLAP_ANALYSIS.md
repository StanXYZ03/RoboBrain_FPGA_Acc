# Jetson + FPGA 异构方案与原计划重合项分析

更新时间：2026-09-01

## 1. 暂定系统划分

本文先假设采用以下异构方案，再判断原计划中哪些工作可以保留：

```text
Jetson Thor
  ├─ 图像预处理
  ├─ Vision Encoder
  ├─ 视觉/文本 Prefill
  └─ 系统调度

FPGA
  ├─ 自回归 Decoder 中适合定制的带宽受限计算
  ├─ 结构化稀疏与权重流
  ├─ 约束输出或专用动作 Token
  └─ 确定性实时数据通路
```

这只是用于梳理工作的候选划分，不代表最终决定。若 Prefill 产生的 KV Cache 无法以足够低的代价传到 FPGA，Prefill 与 Decode 可能需要重新放到同一设备。

## 2. 可以原样保留的目标

| 原计划内容 | 重合程度 | 说明 |
|---|---:|---|
| RoboBrain BF16、Q8 基线与任务精度验证 | 完全重合 | 异构系统仍需相同模型和精度基线 |
| Pointing、Grounding 等真实指标 | 完全重合 | 平台变化不改变 PCK、IoU 等评价方式 |
| 逐阶段 Profiling 与 Roofline | 完全重合 | 用于决定任务放在 Jetson 还是 FPGA |
| Compute-bound/Memory-bound 判断 | 完全重合 | 异构划分必须由实测瓶颈支持 |
| 视觉 Token 减少 | 完全重合 | 同时减少 Prefill、传输和后续计算 |
| 结构化稀疏 | 基本重合 | 可用于 FPGA 权重流与 Decoder |
| 双速率连续推理 | 完全重合 | 仍是实现 3–5 Hz 控制更新的主要路径 |
| 精度—延迟—功耗 Pareto | 完全重合 | 增加 Thor-only 和异构基线即可 |
| 先仿真、后上板 | 完全重合 | 硬件验证顺序不变 |
| 端到端功耗、资源与复现包 | 完全重合 | 异构系统需额外报告整机和通信功耗 |

已经完成的 ZCU102 PS 端模型部署、网页接口、真实图片输出和逐阶段计时继续作为正确性与低端平台基线。

## 3. 重要性提高的工作

### 3.1 Decoder GEMV 与权重流

已有实测表明：Vision Encoder 与 Prefill 的大矩阵偏计算/调度受限，Decoder 的 `M=1` GEMV 偏权重带宽受限；DATAFLOW 对大矩阵有效，对 Decoder 基本无收益。异构方案应保留并加强以下工作：

- 连续权重突发读取；
- 权重预取、双缓冲与片上缓存；
- Decoder 算子融合；
- KV Cache 分块与压缩；
- 块稀疏或 N:M 稀疏；
- 减少每个 Token 必须读取的权重量。

### 3.2 PL 驻留 Transformer 子图

逐个 `MUL_MAT` 下发 FPGA 并回读结果的方案已经因往返开销被否决。正在实现的 PL 驻留 FFN 包含 RMSNorm、SiLU/GELU、门控乘法、残差和 Q8_0 重量化，在异构系统中仍可复用。目标改为让完整 Decoder Layer 或连续子图驻留 FPGA，减少 Jetson 与 FPGA 之间的交互次数。

### 3.3 Q8/GGML 数据桥接

已经完成的 GGUF Q8_0 解析、INT8 payload 与 FP16 scale 拆分、权重预打包、激活量化、自动分块和数值参考可以继续使用。若主机由 ZCU102 PS 改为 Jetson，底层传输代码需要重写，但模型格式合同和硬件数据布局仍有效。

### 3.4 Token 减少与双速率推理

异构系统也不能只依赖输出 Token 吞吐达到 3–5 Hz。原计划中的关键帧完整 VLM、视觉/KV状态缓存、中间帧变化 Token 更新和高频结构化输出仍是主线。L2@87.5%继续作为冻结安全基线；已经失败的学习式残差头不再继续。

## 4. 需要修改定位的工作

### 4.1 Vision Encoder 与视觉 Prefill 的 FPGA 加速

采用 Thor 后，不再默认将完整 Vision Encoder 或视觉 Prefill 搬到 FPGA。现有 PE16 DATAFLOW 大矩阵核保留为可复现基线和候选核；只有 Thor 实测显示某个算子功耗高、延迟不稳定或软件后端支持不足时，才考虑下沉 FPGA。

### 4.2 非线性近似

非线性研究缩小到 FPGA 实际承载的子图。Jetson 侧 Vision Encoder 的 GELU、Softmax 和 Norm 不重复实现；FPGA Decoder 内的 RMSNorm、RoPE、Softmax、SiLU只有在 Profiling 证明成为瓶颈后才进入近似设计。

### 4.3 任务感知主机与任务无关 PL

原架构：

```text
ZCU102 PS：任务调度
ZCU102 PL：统一计算
```

异构架构：

```text
Jetson：任务、Prompt、视觉计算和全局调度
ZCU102 PS：FPGA运行时、DMA与通信管理
ZCU102 PL：任务无关计算子图
```

PL描述符继续只包含矩阵形状、地址、Token长度和量化格式，不包含 RoboBrain 的四种任务名称。

## 5. 降级为对照或停止扩展的工作

1. 把完整 Vision Encoder 逐层全部搬到 ZCU102；
2. 用现有 GEMM 阵列继续堆 PE 来解决 Decoder；
3. 为四种 RoboBrain API 分别设计硬件；
4. 继续优化已经验证无效的学习式 Token 残差头；
5. 把“单块 ZCU102 对每张新图完整运行 4B VLM 达到 3–5 Hz”作为硬目标；
6. 在异构通信边界确定前继续开发大量 PL 视觉算子。

## 6. 异构方案新增工作

### 6.1 设备间通信

Jetson 与 ZCU102 不共享 DDR，需要确定 PCIe、高速以太网或其他互连，并测量单次数据量、有效带宽、P50/P95延迟、复制次数、启动开销和通信能耗。普通 1 GbE 不适合逐层往返，系统边界必须至少覆盖完整阶段或完整子图。

### 6.2 KV Cache 迁移

若 Jetson 完成 Prefill、FPGA 完成 Decode，就必须迁移所有层的 KV Cache。1080个视觉Token对应的未压缩Cache可能达到百MB量级，因此需要研究KV Cache量化、增量更新、跨帧复用和高速互连；否则应调整分工，使 Prefill 与 Decode 位于同一设备。

### 6.3 新增实验基线

| 基线 | 作用 |
|---|---|
| ZCU102 PS-only | 已有基础部署基线 |
| ZCU102 PS+PL | 单板低功耗基线 |
| Jetson Thor-only | 现有机器人平台基线 |
| Jetson Thor+FPGA | 异构系统基线 |
| 异构+Token/KV压缩 | 算法—硬件协同方案 |

## 7. 原工作包调整

| 原工作包 | 异构方案处理 |
|---|---|
| WP0 基线与任务集 | 原样保留，增加 Thor-only |
| WP1 Profiling 与 Roofline | 原样保留，增加 Thor 和通信 Roofline |
| WP2 稠密 PL 基线 | 保留已有结果，转向 Decoder/驻留 Block |
| WP3 视觉 Token 削减 | 保留，重点转向跨帧复用和传输削减 |
| WP4 结构化稀疏 | 保留，优先服务 FPGA 权重流 |
| WP5 非线性与融合 | 缩小到 FPGA 实际承载的子图 |
| WP6 双速率推理 | 原样保留，仍是 3–5 Hz 的主要路线 |
| WP7 系统评测 | 增加 Thor-only、通信开销和整机功耗 |

## 8. 当前判断与下一门槛

原计划中最值得保留的部分是瓶颈分析、Token缩减、GEMV权重流、PL驻留子图、任务精度和双速率推理。需要调整的是把“大矩阵计算全部下沉PL”改为“依据实测进行阶段划分”，并新增跨设备通信和KV Cache管理。

下一门槛不是立即修改硬件，而是：

1. 完成当前真实运行时FFN激活捕获和PL驻留子图数值门；
2. 冻结两到三个异构切分候选；
3. 计算并实测每种切分的中间张量/KV Cache传输量；
4. 据此决定FPGA接管Decoder、完整语言模型还是特定实时子图。

原计划见 [`PL_ACCELERATION_RESEARCH_PLAN_2027.md`](PL_ACCELERATION_RESEARCH_PLAN_2027.md)，平台调研见 [`ROBOT_COMPUTE_PLATFORM_AND_FPGA_ROLE_SURVEY.md`](ROBOT_COMPUTE_PLATFORM_AND_FPGA_ROLE_SURVEY.md)。
