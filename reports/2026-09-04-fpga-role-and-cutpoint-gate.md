# FPGA角色复核与多切入点门控

日期：2026-09-04  
状态：`BF16_SMOKE_COMPLETE / Q8_0_GATE_REQUIRED`

## 1. “单向输入”和“状态/子图驻留”

“单向输入”不是指链路只能朝一个方向工作，而是模型主张量只沿计算图前进一次。例如：

```text
图像 → ZCU102 Patch/浅层块/选择/压紧 → 紧凑Token → GPU或NPU后续模型
```

若GPU先计算稠密特征，再执行`GPU → FPGA Selector → GPU`，一次选择要搬入完整张量并搬回紧凑张量。这是“外置往返”，同步、布局转换和驱动开销都位于关键路径。

“状态/子图驻留”指权重、KV或多个连续算子留在FPGA一侧，不随每层、每Token重复传输。例如FPGA连续执行完整Decoder，或持有KV并连续执行Attention；主平台只按请求、层组或较大Head块交换数据。

二者至少满足一项，才能避免通信吞掉小算子收益；但仍须比较FPGA计算、通信和被替代的主平台时间，因此只是必要条件。

## 2. 顶会工作中的实际角色

| 工作 | 发表 | FPGA承担的角色 | 对本项目的约束 |
|---|---|---|---|
| [HeatViT](https://doi.org/10.1109/HPCA56546.2023.10071047) | HPCA 2023 | Selector、Token打包和后续ViT主体位于同一FPGA数据流；仅LayerNorm留给ARM | Selector收益来自压紧后续计算，不是独立跨板评分 |
| [Fitop-Trans](https://ieeexplore.ieee.org/document/10705515/) | FPL 2024 | 首层前固定长度剪枝，Token selector与分区脉动阵列同板 | 早切且后续消费者驻留，减少片外流量 |
| [DFX](https://arxiv.org/abs/2209.10797) | MICRO 2022 | 四块U280执行完整GPT-2推理，模型与中间状态按HBM/板卡划分 | 依赖完整子图和高带宽存储驻留 |
| [FlightLLM](https://arxiv.org/abs/2401.03868) | FPGA 2024 | 稀疏GEMM/GEMV、Attention和非线性组成完整LLM数据流 | 稀疏索引与计算在同一驻留流水线内消费 |
| [Hummingbird](https://arxiv.org/abs/2507.03308) | ICCAD 2025 | 嵌入式FPGA上驻留低比特Decoder和权重，连续生成Token | 适合单板能效路线，但不能外推其文本速度到4B VLM图片端到端 |
| [HILOS](https://arxiv.org/abs/2502.09921) | ASPLOS 2026 | SmartSSD FPGA在KV所在位置执行Attention | FPGA靠近状态，避免KV跨主机往返 |
| [HPU](https://arxiv.org/abs/2504.16112) | 预印本 | GPU执行稠密FC/FFN，U55C持有KV并执行Attention，经PCIe 4.0×8批量交换Head块 | 高速PCIe、大粒度传输和KV驻留缺一不可 |
| [ReCoVLM](https://doi.org/10.1109/FCCM68464.2026.00026) | FCCM 2026 | Jetson GPU执行视觉编码与跨模态Prefill，VU9P执行完整Decoder；视觉Token压到5% | 公开摘要没有披露物理链路，不能据此假定ZCU102接口性能 |
| [FNM-Trans](https://ieeexplore.ieee.org/document/11294873/) | DAC 2024 | 用统一N:M稀疏模式覆盖QKV、Attention和FFN的连续Transformer计算 | 若本项目选择N:M，必须联合训练、格式和流水线，不能只做单个稀疏GEMM |

此前计划引用的`arXiv:2512.24713`已由作者撤回，撤回说明指出机器学习流程与独立开发的FPGA加速器存在明显不一致，故不再作为方案依据。

## 3. 当前创新边界

较稳妥的主张不是“FPGA比Jetson/昇腾更快”，而是：

1. 为具身空间输出建立保留二维位置、MRoPE、DeepStack和全局锚点的结构化Token削减；
2. 把Patch或浅层视觉前缀、评分、选择、合并和压紧组成ZCU102单向驻留数据流；
3. 用精度、被删主算子和通信字节共同选择切入点，并对无净收益的边界执行止损。

只有Merger后Oracle通过时，ZCU102若只做Selector仍可能没有足够工作量来抵消往返，应转为连续视觉前缀驻留或其他稀疏子图。早期Oracle也失败时，停止该Token硬件路线，不继续为已有L2规则写RTL。

## 4. 已执行门控

- 新建16图、32请求的`cutpoint_oracle_dev`开发集，与此前416张已用COCO图像重合为0。
- 实现五个切入点：Patch Embedding、视觉块1、视觉块3、第一个DeepStack层5（零基索引5，共完成6块）和Merger后。
- 实现读取真实框的几何Oracle。它先保留所有与目标相交的规则group，再用空间锚点和邻近上下文填满预算；这是不可部署上界。
- 支持100%、75%、50%、25%保留率，保留原始视觉RoPE、LLM MRoPE和DeepStack映射。
- 41个单元测试通过，其中包含所有切入点的DeepStack对齐功能仿真。
- 静态审计基于真实HF预处理网格，同时单列既有板端4320 Patch合同。

4320 Patch合同时，25%保留的Patch/浅层单向FP16载荷约2.113 MiB，10GbE串行化下界约2.216 ms；若作为外置Selector往返，则约10.551 MiB、11.064 ms。1GbE对应约22.162 ms和110.635 ms，往返路径超过5 Hz周期40%通信预算。所有数值均未计协议、DMA、驱动、布局转换和FPGA计算，不是链路实测。

开发集与静态结果：[`2026-09-04-cutpoint-oracle-preflight.md`](2026-09-04-cutpoint-oracle-preflight.md)  
实验锁：[`2026-09-04-cutpoint-oracle.lock.json`](2026-09-04-cutpoint-oracle.lock.json)

## 5. BF16烟雾结果与Q8_0约束

A6000空闲后已完成20组、160次BF16真实前向。100%路径在五个切入点输出一致；75%保留时，视觉块1、首个DeepStack层和Merger后通过小样本门，Patch Embedding失败。视觉块1@75%的主MAC比例为0.746，Pointing PCK@0.1/PCK@0.2为0.75/1.00，Grounding Mean IoU/Acc@0.5为0.888/1.00；每任务仅4条，不能视为精度结论。

最终部署对象是`RoboBrain2.5-4B.Q8_0.gguf + RoboBrain2.5-4B.mmproj-Q8_0.gguf`。上述BF16结果只用于排除明显不可行边界，不能冻结硬件。当前GGUF路径的Qwen3-VL视觉图在`tools/mtmd/models/qwen3vl.cpp`中以固定`n_pos`构图，24层后才输出merger和DeepStack拼接结果；没有运行时Token mask或中间压紧接口。因此下一步先在不改变权重的情况下实现Merger后Q8_0 Token列压紧与位置同步门，再实现视觉块1后的Q8图内gather。两级Q8门都必须使用真实Q8_0主模型和mmproj。

BF16结果：[`2026-09-04-cutpoint-oracle-summary.md`](2026-09-04-cutpoint-oracle-summary.md)
