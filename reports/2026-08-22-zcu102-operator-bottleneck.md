# RoboBrain2.5-4B 在 ZCU102 PS 端的算子级瓶颈分析

日期：2026-08-22  
状态：`PS_BASELINE_PROFILE_PASS / FIRST_PL_TARGET_SELECTED`

## 1. 目的与结论

本次工作在不使用 PL 加速的基线上，将一次文字请求和一次 Grounding 图片请求拆分到 GGML 算子、功能模块及具体矩阵形状，以确定首个 PL 加速对象。

结论：首个 PL 原型应实现一套可变形状的 `Q8_0 权重 × FP32/FP16 激活` 矩阵乘引擎，优先覆盖 FFN 与注意力投影，并预留复用同一 MAC 阵列实现 `QK^T`、`P×V` 的数据通路。只加速 GELU/SwiGLU 没有意义；图片推理还必须同步减少视觉 token，单靠算子加速无法达到 3–5 Hz。

## 2. 实验配置

- 板卡：Xilinx ZCU102，XCZU9EG；PS 为 4 核 Cortex-A53，实测频率 `1,199,999 kHz`。
- 内存：16 GiB DDR4 SODIMM；Linux 可见 `14,310,760 KiB`。
- DDR 顺序带宽实测：读 `2.158 GB/s`，写 `8.209 GB/s`，读写拷贝合计 `3.632 GB/s`；256 MiB 缓冲区、4 次迭代。
- 模型：`RoboBrain2.5-4B.Q8_0.gguf`，来源 `mradermacher/RoboBrain2.5-4B-GGUF`，上游模型 `BAAI/RoboBrain2.5-4B`。
- 视觉投影：`RoboBrain2.5-4B.mmproj-Q8_0.gguf`。
- 模型与 mmproj 均为 GGUF `Q8_0`，文件 SHA-256 分别为 `835b78ff...349eb`、`fcc27825...fde2`。
- llama.cpp 基线提交：`178a6c44937154dc4c4eff0d166f4a044c4fceba`。
- 推理参数：`ctx=2048`、`batch=2048`、`ubatch=512`、`threads=4`、`parallel=1`、`mlock`。
- 图片：COCO `000000022371.jpg`；Grounding prompt 要求返回 0–1000 归一化框。
- 图片经过视觉编码器产生 `4320` 个 patch token，2×2 merge 后进入语言模型的是 `1080` 个视觉 token，实际按 `512 + 512 + 56` 三个 ubatch 执行。
- 为只获得一个 decode 图并控制总时长，Grounding 生成上限临时设为 2 token；返回 `[0` 不用于精度评价。

## 3. 测量方法与边界

在 GGML CPU graph 中为每个节点插入线程屏障和单调时钟，记录算子、张量类型、输入/输出形状、耗时、矩阵乘 FLOP 及理想最低访存量。仅在 `GGML_CPU_PROFILE=1` 时启用；正常推理二进制未替换。

- 矩阵乘按每个输出元素 `2K FLOP` 计数。
- Flash Attention 只计 `QK^T` 与 `P×V`，不计 softmax，因此总 FLOP 是保守值。
- 访存量是假设每个张量至少读/写一次的理想下界，不包含 cache miss、写分配和重复搬运。
- profiling 会改变同步与日志开销；算子占比和排序可用于选核，端到端性能应由无插桩版本复测。
- Roofline 使用 `38.4 GFLOP/s = 4×1.2 GHz×8 FP32 FLOP/cycle` 作为 A53 FP32 NEON 理论参考，用 `2.158 GB/s` 实测读带宽；分界点为 `17.79 FLOP/B`。该峰值不是 Q8 内核的实测峰值。
- 对 FLOP 未建模的激活、归一化和搬运算子，不根据报告中的 `memory` 标签作强结论。

## 4. 端到端阶段耗时

### 文字请求

Prompt：`Reply with only the word OK.`；输入 15 token、输出 2 token，结果为 `OK`。

| 阶段 | 时间 |
|---|---:|
| 文本 prefill | 12.940 s |
| decode compute | 1.043 s |
| 首 token（服务端） | 12.945 s |

文字 prefill 中 `MUL_MAT` 占 `98.9%`，达到 `8.43 GFLOP/s`，属于计算受限；decode 中 `MUL_MAT` 占 `97.7%`，算术强度仅 `1.878 FLOP/B`，属于内存带宽受限。

### Grounding 图片请求

端到端耗时 `1695.01 s`，服务端首 token `1693.454 s`。

| 阶段 | 时间 | 端到端占比 |
|---|---:|---:|
| 图片解析与请求准备 | 0.204 s | 0.01% |
| 视觉编码器与 projector | 625.425 s | 36.90% |
| 视觉 token 进入 LLM 的 prefill | 1003.155 s | 59.18% |
| 文本 prefill | 63.732 s | 3.76% |
| decode compute | 1.562 s | 0.09% |

视觉编码与视觉 prefill 合计约占 `96.1%`，因此当前图片瓶颈不是解码，而是长视觉序列的编码和预填充。

## 5. Grounding 功能模块分解

### 视觉编码器：625.294 s 的算子节点时间

| 模块 | 时间 | 阶段占比 |
|---|---:|---:|
| Attention core（QK、softmax、PV 融合算子） | 278.022 s | 44.5% |
| FFN 线性层 | 204.751 s | 32.7% |
| Attention Q/K/V/O 投影 | 102.769 s | 16.4% |
| DeepStack 投影 | 20.640 s | 3.3% |
| 多模态 projector | 6.886 s | 1.1% |
| GELU | 3.779 s | 0.6% |
| 其余搬运、归一化、RoPE、逐元素等 | 8.448 s | 1.4% |

### 视觉 prefill：1002.913 s 的算子节点时间

| 模块 | 时间 | 阶段占比 |
|---|---:|---:|
| FFN 线性层 | 668.157 s | 66.6% |
| Attention 投影 | 241.295 s | 24.1% |
| Attention core | 88.331 s | 8.8% |
| SwiGLU | 2.192 s | 0.2% |
| 其余 | 2.937 s | 0.3% |

### 文本 prefill 与 decode

- 文本 prefill：FFN `47.8%`、attention core `34.5%`、attention projection `17.2%`。
- 单 token decode：FFN `43.4%`、attention core `34.0%`、attention projection `15.2%`、LM head `6.2%`；这些主矩阵在该阶段均呈低算术强度、带宽受限特征。

## 6. 关键矩阵形状

耗时最大的规则矩阵包括：

| 阶段与功能 | 权重/输入主形状 | 调用数 | 累计时间 |
|---|---|---:|---:|
| 视觉 prefill，FFN gate/up | `2560×9728` × `512 token` | 140 | 424.792 s |
| 视觉 prefill，FFN down | `9728×2560` × `512 token` | 70 | 208.647 s |
| 视觉编码器，FFN up | `1024×4096` × `4320 token` | 24 | 102.684 s |
| 视觉编码器，FFN down | `4096×1024` × `4320 token` | 24 | 102.067 s |
| 视觉 prefill，attention QKV | `2560×4096` × `512 token` | 72 | 92.060 s |
| 视觉编码器，attention QKV | `1024×3072` × `4320 token` | 24 | 77.033 s |

这说明首核不能只为固定的语言层尺寸硬编码，至少需要支持视觉编码器的 `1024/4096` 和语言模型的 `2560/9728` 两组主维度，以及 `M=1` 的 decode 与 `M=56/512/4320` 的 prefill/encoder 两种工作模式。

## 7. Roofline 与 Amdahl 判断

- encoder/prefill 主矩阵算术强度约 `75–1440 FLOP/B`，远高于 `17.79 FLOP/B` 分界，当前 PS 基线主要是计算受限。
- decode 主矩阵算术强度约 `1.9–4.0 FLOP/B`，低于分界，主要是内存带宽受限。
- Grounding 各请求阶段中，普通 `MUL_MAT` 累计约 `1289.36 s`，约占 `76.1%`；fused attention core 约 `388.84 s`，约占 `23.0%`，两者合计约 `99%`。
- 若只把普通矩阵乘无限加速，Amdahl 上限约为 `4.18×`；普通矩阵乘加速 10×时，总加速约 `3.17×`。
- 若矩阵乘与 attention core 都加速 10×，理论仍约需 `169.5 s/图`，仅约 `0.0059 Hz`。
- 当前约 `0.00059 Hz`；达到 3 Hz 和 5 Hz 分别需要约 `5085×`、`8475×` 端到端加速。
- 本次保守建模得到每图至少 `13.443 TFLOP`，3 Hz/5 Hz 分别要求至少 `40.3/67.2 TFLOP/s` 的持续有效吞吐，还未计 softmax 等运算。完整 4B、1080 视觉 token 流程不可能仅靠 ZCU102 PL 的一个算子核达到该目标。

## 8. 首个 PL 原型定义

第一阶段实现统一的矩阵引擎，而不是按 Grounding/Pointing/Trajectory/General 各做硬件：

1. 权重格式：原生读取 GGUF `Q8_0` 的 32 元素 block，执行 block scale 解码后 MAC；激活先保留 FP16/FP32 接口以复现现有数值路径。
2. 数据通路：DDR → DMA → 片上双缓冲 → 可分块 MAC 阵列 → 累加/缩放 → DDR；支持权重、激活和输出 tile 重用。
3. 两种模式：`GEMM` 模式覆盖 encoder/prefill 的多 token 高算术强度矩阵；`GEMV` 模式覆盖 decode，重点优化连续权重流和带宽利用率。
4. 接口：运行时传入 M/N/K、stride、量化 block 参数，不为四类任务各复制计算阵列。
5. 扩展：MAC 阵列和片上缓存预留 `QK^T` 与 `P×V` 路径；softmax 第一版可留在 PS，第二版再做在线 softmax/fused attention。只做普通矩阵乘的系统收益上限已经由 Amdahl 定量限制。

## 9. 下一项实验及验收门槛

在上板前先完成一个代表性 kernel 的 C/RTL/HLS 数值对拍与 testbench：

- 首测尺寸：`Q8_0 [2560×9728] × FP16/FP32 [512×2560]`，再覆盖 `1024×4096×4320` 和 `M=1`。
- 正确性：与现有 GGML CPU 输出逐元素比较，报告最大绝对误差、相对误差和任务输出一致性。
- 性能：报告周期数、有效 GOP/s、DDR 实际带宽、DMA/计算重叠率、DSP/BRAM/URAM/LUT 占用与板级功耗。
- 系统复测：同一张 Grounding 图片、同一 prompt、同一 token 数，与本报告 PS 基线逐阶段对比。
- 并行开展视觉 token 消减：从 `4320→1080` 之后继续做空间感知 pooling/pruning；必须分别验证 grounding 坐标误差、pointing 点误差、trajectory 轨迹误差和 general 文本质量，避免只追求 token 数。

## 10. 运行状态与可复现文件

- 本轮使用 JTAG RAM Linux，未写 SD 卡或 Flash；只读介质证明：`write_counters_unchanged=1`。
- 分析结束时模型服务仍在线且空闲；PS SysMon 当前 `28.86 °C`，本次上电最大 `38.32 °C`，未见过热。
- 原始日志、JSON、自动分析报告、分析脚本、单元测试和 llama.cpp 插桩补丁均已随本报告归档。

