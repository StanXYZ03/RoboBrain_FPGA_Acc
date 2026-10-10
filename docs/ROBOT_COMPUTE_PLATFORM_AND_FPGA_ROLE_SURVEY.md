# 具身智能大模型计算平台与 FPGA 加速路线调研

更新时间：2026-09-01

本文讨论 RoboBrain2.5-4B 一类具身视觉语言模型的本地推理。这里所说的“适合部署”包含两层含义：一是硬件容量和算力是否足够，二是开发者能否获得完整的软件栈、模型转换工具和硬件接口。车规芯片公布了很高的 TOPS，并不等于普通研究团队可以直接把 RoboBrain 部署上去。

## 1. 现有机器人计算平台是否适合部署 RoboBrain2.5-4B

### 1.1 总体判断

现有高端机器人计算平台能够部署 4B 级视觉语言模型，其中 NVIDIA Jetson AGX Thor 最合适。RoboBrain2.5-4B 的 Q8_0 主模型约 4.69 GB，视觉投影模型约 454 MB；即使考虑 KV Cache、视觉中间结果和运行时空间，也远低于 Thor 的 128 GB 内存。真正的问题不是“能不能装下”，而是模型能否转换到高效的 GPU 推理后端，以及视觉编码、Prefill 和 Decode 能否充分利用 Tensor Core 和内存带宽。

现有平台可以分为三类：

| 平台类型 | 代表平台 | 部署判断 | 主要限制 |
|---|---|---|---|
| 通用机器人 GPU 平台 | Jetson Thor、Jetson AGX Orin、Orin NX | 适合，工具链相对完整 | 功耗高于小型 FPGA；峰值算力不能直接等同于单请求延迟 |
| NPU/BPU 平台 | 地平线、地瓜机器人 S600、高通 Ride、部分国产机器人芯片 | 容量和算力可能足够，需要逐模型验证 | 算子覆盖、量化格式、动态图和自定义算子受 SDK 约束 |
| 封闭车规或自研平台 | Tesla AI、蔚来神玑、理想 M100、华为 MDC、小鹏图灵等 | 厂商内部可能适合，通常不适合作为独立课题组的实验平台 | 开发板、编译器、驱动或模型部署接口不公开 |

Mobileye EyeQ 一类芯片更偏向已经固化的感知和自动驾驶流水线。EyeQ6H 的公开指标为 34 INT8 TOPS，它适合 ADAS，但不应仅凭 TOPS 判断其适合运行任意生成式 VLM。[Mobileye EyeQ 平台](https://www.mobileye.com/technology/eyeq-chip/)

地瓜机器人 RDK S600 提供最高 560 TOPS 的 BPU 和 32 GB LPDDR5 版本，从容量上可以容纳 4B Q8 模型；是否能高效运行 RoboBrain，仍取决于 Transformer、注意力、视觉投影和自定义后处理的编译支持。[RDK S600 文档](https://developer.d-robotics.cc/rdk_s_doc/en/01_Quick_start/01_hardware_introduction/02_rdk_s600/01_rdk_s600_kit)

高通 Snapdragon Ride Flex 支持异构计算和生成式 AI，但其主要目标是座舱、ADAS 和混合安全等级任务，部署自由度取决于高通及 Tier-1 的汽车软件栈，不像 Jetson 那样适合快速开展开放模型实验。[Qualcomm Ride Flex](https://www.qualcomm.com/news/releases/2023/01/qualcomm-unveils-snapdragon-ride-flex---the-automotive-industry-)

因此，对本课题最合理的基线不是证明“现有平台都不能部署”，而是承认它们能够部署，再研究 FPGA 能否在功耗、确定性延迟或某些特定阶段上提供额外价值。

### 1.2 Jetson AGX Thor

Jetson AGX Thor 使用 Blackwell GPU，而不是单独的传统 NPU。T5000 模组包含 2560 个 CUDA Core 和 96 个第五代 Tensor Core，并提供 Transformer Engine、FP4/FP8 支持和 MIG。主要规格如下：

| 项目 | Jetson AGX Thor T5000 |
|---|---:|
| 峰值 AI 算力 | 2070 TFLOPS，稀疏 FP4 |
| 稠密 FP4 峰值 | 1035 TFLOPS |
| 内存 | 128 GB、256-bit LPDDR5X |
| 内存带宽 | 273 GB/s |
| CPU | 14 核 Arm Neoverse-V3AE |
| 功耗范围 | 40–130 W |
| 软件环境 | JetPack 7、CUDA 13、TensorRT、vLLM、Isaac ROS/GR00T |

2070 TFLOPS 是稀疏 FP4 峰值，不能直接换算成 RoboBrain 的 token/s。模型当前采用 GGUF Q8_0，若继续使用 llama.cpp CUDA 后端，可以较快完成兼容性验证，但未必能充分使用 Thor 的 FP4 Transformer Engine。若追求性能，需要进一步转换到受 TensorRT-LLM、TensorRT 或 vLLM 支持的格式，并重新验证精度。

NVIDIA 给出的 MAXN、上下文 2048、输出 128 token、最大并发 8 的结果为：Qwen2.5-VL-3B 达到 356.86 output token/s，Qwen2.5-VL-7B 达到 252 output token/s，GR00T N1.5 达到 41.5 output token/s。这些是特定软件配置下的并发吞吐，不是 RoboBrain 单请求实测值，也不能直接当作图像帧率。[Jetson Thor 技术说明与基准](https://developer.nvidia.com/blog/?p=104879)

以模型规模看，RoboBrain2.5-4B 位于上述 3B 和 7B VLM 之间。可以合理预期 Thor 会明显快于当前 ZCU102 PS 端，但具体 TTFT、单请求 token/s 和单图延迟必须实测。对 Pointing、Grounding、Trajectory 这类短输出任务，首 token 时间通常比峰值 token/s 更重要。

综合来看，Thor 适合作为：

1. RoboBrain 的高性能部署平台；
2. FPGA 方案的性能和能效对照基线；
3. GPU+FPGA 异构系统中的主机侧视觉编码和 Prefill 设备。

它的不足主要是 40–130 W 功耗、开发套件成本，以及闭源 GPU 微结构不便于开展细粒度硬件创新。这些不足并不妨碍部署，但给 FPGA 的能效和可定制数据流研究留下了空间。

### 1.3 Jetson Orin 与其他平台

Jetson AGX Orin 提供最高 275 TOPS、32/64 GB LPDDR5 和 15–60 W 功耗；Orin NX 提供最高 157 TOPS、8/16 GB 内存和 10–40 W 功耗。两者都可以运行 4B 量化模型。AGX Orin 的内存余量更充分；Orin NX 16 GB 可以容纳模型，但同时运行视觉、语言和机器人软件时需要控制上下文、KV Cache 和中间张量。[Jetson Orin 规格](https://www.nvidia.cn/autonomous-machines/embedded-systems/jetson-orin/)

从研究可复现性看，Jetson 的优势不只在芯片算力，还包括 CUDA、TensorRT、JetPack、Isaac ROS 和较成熟的模型部署社区。相比之下，部分汽车和机器人 AI 芯片的纸面算力并不低，但第三方难以取得同等完整的编译与运行环境。

## 2. 国内外人形机器人的计算平台

人形机器人通常不是用一颗芯片完成全部工作，而是分成至少两层：实时运动控制由 MCU、实时 CPU 或专用控制器负责；视觉、语言、规划和 VLA/VLM 推理由 Jetson 或自研 AI 计算平台负责。训练一般仍在服务器 GPU 上完成。因此，“机器人采用某计算平台”通常指高层 AI 计算单元，不代表关节闭环也运行在该平台。

### 2.1 公开信息较明确的案例

| 地区 | 机器人 | 公开的计算平台 | 说明 |
|---|---|---|---|
| 中国 | 宇树 G1 EDU | NVIDIA Jetson Orin，常见开发计算单元为 Orin NX 16 GB | 另有独立运动控制计算单元；开发者使用高层计算单元进行二次开发 |
| 中国 | 小鹏 IRON | 3 颗自研图灵 AI 芯片，官方称有效算力 2250 TOPS | 面向 VLT、VLA、VLM 的自研软硬件系统，不是开放通用开发板 |
| 国际 | Agility Robotics Digit | Jetson AGX Thor；后续安全系统还采用 IGX Thor | 用于实时感知、导航和自主决策 |
| 国际/研究参考设计 | NVIDIA Isaac GR00T Reference Humanoid、Unitree G1 工作流 | Jetson Thor | 面向 GR00T 模型、Isaac ROS 和实机部署的参考方案 |

宇树官方页面列出了 G1 的 NVIDIA Jetson Orin 扩展算力；公开的 G1-EDU 开发资料进一步给出 Orin NX 16 GB 开发计算单元，并明确它与不可由用户访问的运动控制单元分离。[Unitree G1](https://www.unitree-robot.com/cn/g1/index.html)、[G1-EDU 计算单元说明](https://docs.quadruped.de/projects/g1/html/g1_overview.html)

小鹏官方披露的新一代 IRON 使用 3 颗图灵 AI 芯片和 2250 TOPS 有效算力，并运行 VLT、VLA、VLM 组合。该案例说明机器人厂商也在采用自研 AI SoC，但平台并未作为通用研究开发板开放。[小鹏 IRON 官方信息](https://www.xiaopeng.com/news/company_news/5511.html)

Agility Digit 已公开采用 Jetson AGX Thor；NVIDIA 还宣布在其安全相关系统中引入 IGX Thor 和 Halos。这里的 IGX Thor 更强调工业安全、故障监测和认证，不等同于普通 Jetson 开发套件。[Digit 与 Jetson Thor](https://investor.nvidia.com/news/press-release-details/2025/NVIDIA-and-US-Manufacturing-and-Robotics-Leaders-Drive-Americas-Reindustrialization-With-Physical-AI/default.aspx)、[Digit 与 IGX Thor](https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-Announces-Halos-for-Robotics-the-Industrys-First-Full-Stack-Safety-System-for-Physical-AI/default.aspx)

NVIDIA 已发布基于 Jetson Thor 的 Unitree G1 实机 GR00T 工作流，包含数据采集、模型训练、仿真评估和部署。这比单纯的厂商合作名单更能说明 Thor 已经进入可复现的人形机器人开发流程。[Unitree G1 + Jetson Thor 工作流](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/index.html)

### 2.2 计算平台未完整公开的案例

Tesla Optimus、Figure 02、Boston Dynamics Atlas、Apptronik Apollo、1X NEO、智元、傅利叶 GR-2、优必选 Walker S 等机器人都公开展示了较强的感知或大模型能力，但许多厂商没有披露量产版本的完整板载计算配置。NVIDIA 将 Agility、Boston Dynamics、Figure 等列为 Jetson Thor 的早期采用者，并称 1X 等公司正在评估 Thor；这只能证明合作或评估关系，不能据此断言每个机器人当前量产版本都内置 Thor。[NVIDIA Jetson Thor 发布信息](https://nvidianews.nvidia.com/news/nvidia-blackwell-powered-jetson-thor-now-available-accelerating-the-age-of-general-robotics)

Figure 使用 NVIDIA 加速计算训练 Helix，UBTECH Walker S 官方说明包含多模态大模型决策，均未公开足以复现的板载主控型号。因此表格中应保留“未披露”，不能用训练平台反推板载平台。[Walker S 官方页面](https://www.ubtrobot.com/en/humanoid/products/walker-s)

从公开案例可以得到一个相对稳妥的判断：研究型和通用型机器人广泛使用 Jetson Orin，并开始向 Jetson Thor 迁移；具备芯片研发能力的汽车或机器人公司会采用自研 AI SoC；实时运动控制仍由独立控制系统承担。单板 FPGA 作为整机唯一大脑的公开案例很少。

## 3. FPGA 在具身智能大模型中的作用

### 3.1 单板 FPGA：以功耗和带宽利用率为主要目标

单板 FPGA 的优势不是峰值算力，而是可以围绕固定量化格式、数据流和外部存储接口裁剪硬件。对 batch=1 的自回归 Decode，每生成一个 token 都要读取大部分模型权重，GEMV 常受 DDR 带宽限制。此时增加乘法器不一定有效，提高突发传输效率、融合算子和减少中间数据搬运更重要。

| 工作 | 会议 | 平台与模型 | 主要结果 | 边界 |
|---|---|---|---|---|
| Pushing up to the Limit of Memory Bandwidth and Capacity Utilization | DATE 2025 | KV260，4 GB DDR4，LLaMA2-7B，4-bit | 约 5 token/s；使用 93.3% 内存容量；达到理论带宽上限的 85% | 裸机、文本 Decode，不包含视觉编码 |
| Hummingbird | ICCAD 2025 | KV260/ZCU104，LLaMA3-8B，GPTQ W4、KV Cache INT8，4K上下文 | KV260 4.8 token/s，ZCU104 8.6 token/s；模型带宽利用率 93–94% | 文本 LLM，不是完整 VLM |

DATE 2025 的工作在 KV260 上通过裸机运行、算子融合和专用数据排布，把有限 DDR 容量和带宽用于 LLaMA2-7B Decode。[DATE 2025 论文](https://arxiv.org/abs/2502.10659)

Hummingbird 在此基础上支持 LLaMA3-8B 和更长上下文，并通过权重卸载解决嵌入式 FPGA 的容量限制。论文报告相对已有研究减少 67% LUT、39% DSP 和 42% 功耗，但摘要没有给出所有平台统一口径的绝对板级功耗，因此不宜自行补算瓦数。[Hummingbird 论文](https://arxiv.org/abs/2507.03308)、[ICCAD DOI](https://doi.org/10.1109/ICCAD66269.2025.11241002)

这些结果说明单板 FPGA 可以把文本 Decode 做到接近 DDR 带宽上限，但不说明单板 FPGA 能以相同 token/s 完成图像编码和跨模态 Prefill。RoboBrain 的视觉编码、视觉 Prefill 和文本 Decode 特征不同，需要至少两类数据流。若坚持单板 ZCU102 路线，比较合理的论文目标是：在较低功耗下完成端到端 VLM，并证明单位任务能耗、带宽利用率或确定性延迟优于通用处理器，而不是追求超过 Jetson Thor 的绝对吞吐。

### 3.2 异构系统：把不同阶段交给更合适的计算单元

VLM 推理通常包括视觉编码、跨模态 Prefill 和自回归 Decode：

```text
图片 → 视觉编码器 → 视觉 token / 投影 → LLM Prefill → 自回归 Decode → 坐标、轨迹或文本
```

视觉编码和 Prefill 具有较大的矩阵维度，适合 GPU 或 NPU；batch=1 Decode 更接近带宽受限的 GEMV，可由 FPGA 使用定制权重流和算子融合处理。FPGA 还可以承担传感器预处理、确定性低时延接口和安全监控，但这些功能需要与大模型推理贡献分开评价。

ReCoVLM 是目前最接近这一思路的完整 VLM 工作：

| 项目 | ReCoVLM |
|---|---|
| 发表 | FCCM 2026 |
| 平台 | Jetson Orin Nano 8 GB + Xilinx VU9P FPGA |
| GPU任务 | 视觉编码、跨模态 Prefill |
| FPGA任务 | 带宽敏感的自回归 Decode、MoE 权重流 |
| 视觉 token | 保留 5% |
| 任务性能 | 保留基线的 88.9% |
| FPGA有效带宽利用率 | 87.2% |
| 端到端吞吐 | 18.1 token/s |
| 能效 | 相对 RTX 4090 提高 2.67 倍 |

ReCoVLM 的 18.1 token/s 是生成吞吐，不是每秒处理 18.1 张图片；公开摘要也没有给出可直接换算的单图延迟。[ReCoVLM 论文信息](https://doi.org/10.1109/FCCM68464.2026.00026)、[论文摘要与参数](https://eurekamag.com/research/108/122/108122390.php)

NANOMIND 虽然不含 FPGA，但提供了另一种可参考的异构分工：在 RK3566 上使用 NPU 运行视觉编码器、GPU 运行 W4A16 LLM Decoder、CPU 负责调度和零拷贝内存管理。论文覆盖 LLaVA-OneVision-0.5B、Qwen2-VL-2B 和 SmolVLM-500M，报告相对对照实现降低 36.2% 端到端延迟和 42.3% 能耗。[NANOMIND，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b20260e3e1bd7ea108831925d65ce0a6-Abstract-Conference.html)

对本课题而言，异构方案比“用 ZCU102 替代 Jetson”更符合实际：Jetson Thor 或 Orin 负责算力密集、软件变化快的视觉编码和 Prefill，FPGA 负责经过充分分析后确定的 Decode 权重流、结构化输出、稀疏跳过或低时延传感器通路。论文创新点应落在阶段划分、跨设备数据布局、token/权重压缩和功耗—精度联合优化，而不是简单把两个开发板连接起来。

### 3.3 异构通信方式与本项目选择

现有GPU/FPGA大模型研究主要把FPGA作为同一服务器中的PCIe加速卡。GLITCHES通过主机PCIe将GPU逐层生成并量化的KV Cache送入U280 HBM，同时把传输与后续层Prefill重叠；HPU使用PCIe 4.0×8 QDMA交换每层Q/K/V和Attention结果，并依靠Head合并、大Batch和流水隐藏通信；INF²与HillInfer则把KV尽量留在SmartSSD内的FPGA附近，只传Query、增量KV或筛选结果。ReCoVLM采用Orin Nano完成视觉编码和Prefill、VU9P完成Decode，但公开会议资料没有说明物理接口，不能写成已经确认使用PCIe或以太网。[GLITCHES](https://ieee-hpec.org/wp-content/uploads/2024/09/112.pdf)、[HPU](https://arxiv.org/abs/2504.16112)、[INF²](https://arxiv.org/abs/2502.09921)、[HillInfer](https://arxiv.org/abs/2602.18750)、[ReCoVLM](https://doi.org/10.1109/FCCM68464.2026.00026)

本项目中的Jetson和ZCU102是两块独立开发板，不能直接照搬服务器PCIe卡结构。ZCU102的4路SFP+连接到PL侧GTH；GTH只是高速SerDes，若要与普通Jetson或服务器网卡通信，还需在PL中实现10G/40G Ethernet MAC、PCS/PMA、DMA和AXI4-Stream数据通路。单路可作10GbE，4路经通道对齐可组成1×40GbE；AMD已有对应ZCU102参考设计。[ZCU102 UG1182](https://docs.amd.com/v/u/en-US/ug1182-zcu102-eval-bd)、[XAPP1326](https://docs.amd.com/v/u/en-US/xapp1326-10g-40g-switching)

当前默认采用10GbE传输压缩视觉状态和小结果，KV Cache继续留在Jetson。按4.0 GB/s规划有效载荷估算，40GbE传输85.32 MiB Q8 KV的纯线路下界约22.4 ms；但ZCU102 Decode仍受DDR实测带宽限制，因此暂不恢复“Jetson Prefill、ZCU102 Decode”的切分。通信接口、各payload下界和平台淘汰表见[边缘AI计算平台综合比较](EDGE_AI_PLATFORM_TOTAL_COMPARISON_2026.md)。

## 4. 本项目当前进展与下一步计划

### 4.1 已完成工作

1. **基础部署。** ZCU102 PS 端 16 GB DDR 已完成跨地址空间读写验证；RoboBrain2.5-4B Q8_0 已完成文字、图片和四种 API 输出模式的本地推理，以太网和网页交互已验证。
2. **端到端基线。** 单图 affordance 测试总耗时 1725.623 s，其中 1142 个 Prompt token 用时 1709.905 s，生成 11 token 用时 15.436 s，实测生成速度 0.713 token/s。
3. **瓶颈分析。** 视觉编码、视觉 Prefill 和文本 Prefill 以大矩阵计算为主；逐 token Decode 以 `M=1` GEMV 为主，主要受权重读取和存储带宽限制。
4. **统一 Q8×Q8 GEMM 核。** PE16 DATAFLOW 核已在 ZCU102 以 200 MHz 完成易失性 JTAG 上板验证，未写入 SD/QSPI。Language Prefill、Vision Encoder 和 Decoder GEMV 的板测分别为 146.09、143.77 和 3.398 GOPS。相对上一版 PE16，前两者加速 1.391 倍和 1.649 倍，Decoder 基本没有变化。
5. **GGML 数据路径。** 已完成 GGUF Q8_0 到 PL 数据格式映射、NEON 四核激活量化、快速 UIO 权重搬运和自动分块。逐算子结果反复回到 PS 的方案已被否决，改为研究 PL 驻留 FFN/Transformer 子图。
6. **PL 驻留 FFN。** 已实现统一向量核，支持 RMSNorm/LayerNorm、SiLU 门控、GELU、残差和 Q8_0 重量化；C 仿真 9/9 通过。真实 Q8 权重代理测试 relative L2 为 1.235%，但输入仍是代理激活，不是运行时真实隐藏状态。
7. **视觉 token 减少。** 已完成独立盲测，保留 L2@87.5% 作为冻结基线，即保留 87.5% 视觉组。后续学习式联合删除没有在真实 PCK/IoU 指标上证明优于该基线，已停止，不把失败结果写成精度提升。

详细记录见[当前状态](../CURRENT_STATUS.md)、[PL加速研究计划](PL_ACCELERATION_RESEARCH_PLAN_2027.md)和[DATAFLOW板端复测](../reports/2026-08-31-zcu102-dataflow-rerun.md)。

### 4.2 当前资源占用

当前有两套需要区分的实现。

**已经上板验证的 DATAFLOW GEMM 系统：**

| 资源 | 使用量 | ZCU102占比 |
|---|---:|---:|
| LUT | 25,966 | 9.47% |
| FF | 30,473 | 5.56% |
| BRAM Tile | 419 | 45.94% |
| DSP | 661 | 26.23% |
| 频率 | 200 MHz | WNS +0.078 ns |

**已完成布局布线、尚未上板的 GEMM + FFN向量核双IP系统：**

| 资源 | 使用量 | ZCU102占比 |
|---|---:|---:|
| LUT | 58,445 | 21.32% |
| FF | 88,156 | 16.08% |
| BRAM Tile | 483 | 52.96% |
| DSP | 815 | 32.34% |
| 频率 | 200 MHz | WNS +0.060 ns |

Vivado 对双 IP 系统的 vectorless 功耗估算为 5.942 W，置信度为 Medium。它不是板级实测功耗，不能用来声称系统能效。

### 4.3 当前性能边界

目前能够正式引用的是算子级板测，而不是完整 RoboBrain PL 加速结果。根据最新板测吞吐替换可覆盖矩阵计算，并保留 CPU Decoder 后得到的一阶单图投影为 538.727 s，即约 8.98 min；这还没有通过完整后端实测。

生成 8 token 的当前投影为 10.749 s，即约 0.744 token/s，只比原始 0.713 token/s 略高。原因是现有核加速了大矩阵 GEMM，却没有解决 Decode 的 GEMV 权重带宽问题。与 Hummingbird 的 4.8/8.6 token/s 和 ReCoVLM 的 18.1 token/s 相比，当前 token 生成速度还不能作为主要成果。

现阶段较完整的表述是：已经完成端到端基线、算子级瓶颈判定、Q8×Q8 大矩阵核实板验证和 PL 驻留 FFN 的离线实现；尚未完成完整 Transformer Block、llama.cpp/GGML 后端接入、端到端精度复验和板级能耗测量。

### 4.4 建议的下一步

短期仍应完成当前 ZCU102 路线，避免停留在孤立算子：

1. 捕获 RoboBrain CPU 推理中的真实 FFN 输入激活，替换当前 embedding 代理输入；
2. 用真实权重和真实激活完成 FFN 数值门，失败时先修正量化边界，不生成 bitstream；
3. 增加 PL 自主 FFN 控制和缓冲区驻留，避免每个矩阵算子都由 PS 启动并回读；
4. 完成双 IP bitstream、JTAG 易失性上板和 FFN 子图实测，不修改 SD 卡；
5. 接入 GGML/llama.cpp 后端，测量完整 TTFT、各阶段延迟、生成 token/s、DDR流量、温度和板级功耗；
6. 单独设计面向 `M=1` GEMV 的 Decode 权重流。评价重点是有效 DDR 带宽、每 token 读取量和算子融合，而不是继续增加 GEMM PE；
7. 在硬件路径稳定后，再研究时序复用、跨帧视觉 token 复用和任务输出压缩。新的 token 方法必须使用未观察数据，并同时报告精度与延迟。

同时建议增加一条对照实验：在 Jetson Thor 或至少 Jetson AGX Orin 上运行相同 RoboBrain 模型、相同图片和相同输出长度。这样才能回答 FPGA 相对于现有机器人平台究竟改善了什么。

后续路线需要在以下两种目标中确定主次：

- **单板 ZCU102：** 重点考察低功耗、DDR带宽利用率和确定性延迟，接受绝对性能低于 Thor；
- **Jetson + FPGA：** 由 Jetson 负责视觉编码和 Prefill，FPGA负责 Decode或其他经过实测确认的瓶颈，重点考察端到端性能和能效。

在当前数据下，先完成真实激活数值门和完整 FFN 上板更合适；取得端到端结果后，再根据生成速度和功耗决定是否继续单板路线，或转向 Jetson+FPGA 异构系统。
