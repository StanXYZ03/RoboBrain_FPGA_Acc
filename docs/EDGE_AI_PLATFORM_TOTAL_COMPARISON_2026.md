# RoboBrain2.5-4B 边缘 AI 计算平台综合比较

更新时间：2026-09-03  
适用模型：RoboBrain2.5-4B，当前 GGUF Q8_0 主模型约 4.69 GB，视觉投影约 454 MB，既有运行时占用约 9 GB。

## 1. 比较口径

本表覆盖可以买到或实验室已经具备的代表性平台，不包括只能通过车企合作获得、软件栈不公开的车规芯片。不同厂商的 TOPS 往往对应不同精度、稀疏率和测试条件，不能直接横向相除。因此，平台选择同时看以下五项：

1. 内存是否能容纳模型、KV Cache、视觉中间张量和机器人程序；
2. Vision Encoder 与 Prefill 所需的持续矩阵计算能力；
3. Decode 每生成一个 token 所需的权重读取带宽；
4. 当前 Q8_0、多模态算子和推理框架能否落到加速器，而不是退回 CPU；
5. 模组或加速卡功耗、主机功耗、散热和采购成本。

价格按 2026-09-03 可查信息记录。美元公开价不换算成人民币，以免汇率、税费和渠道差异造成假精确。成本等级仅用于方案筛选：

| 等级 | 采购成本口径 |
|---|---|
| 已有 | 实验室新增硬件成本为 0，不代表维护和开发成本为 0 |
| A | 不高于约 1500 元 |
| B | 约 1500–5000 元 |
| C | 约 5000–15000 元 |
| D | 约 15000–30000 元 |
| E | 高于约 30000 元 |
| 询价 | 没有可靠的统一公开零售价，或必须连同主机、载板和散热采购 |

## 2. 本模型逐阶段负载

代表性 Grounding 图片产生 4320 个 Vision Encoder token，2×2 合并后为 1080 个视觉 token，完整 Prompt 为 1142 token。ZCU102 PS-only 插桩实测为 1695.01 s；该数据用于判断负载构成，不用于预测其他平台的绝对时间。

| 阶段 | ZCU102 PS 实测 | 主要运算和瓶颈 | 平台选择含义 |
|---|---:|---|---|
| 以太网接收、HTTP/API、图片文件处理 | 0.204 s（含请求准备） | CPU、网络和内存复制 | 所有候选均能承担；不应为此购买高 TOPS 平台 |
| JPEG/PNG 解码 | 未独立计时 | 固定功能视频/图像解码器或 CPU | Jetson、RK3588、昇腾板均有相应硬件；在当前总时延中不是主矛盾 |
| Resize、归一化、布局变换 | 未独立计时 | 规则像素处理，偏带宽 | ISP/PVA、GPU、NPU或PL均可；适合与后级融合，避免重复拷贝 |
| Patch Embedding 与位置处理 | 包含在视觉编码器 | 卷积/投影、位置编码 | 大批量规则计算，GPU/NPU/AIE/PL均可，前提是算子被编译器覆盖 |
| Vision Attention core | 278.022 s | QK、Softmax、PV；长序列计算和片上缓存压力 | 需要高持续算力及融合 Attention；单纯提高 DDR 带宽不够 |
| Vision Attention 投影 | 102.769 s | Q/K/V/O 大矩阵乘 | Compute-bound；CUDA Tensor Core、高算力 NPU、Versal AIE 或 PL GEMM 适合 |
| Vision FFN | 204.751 s | `1024↔4096` GEMM 与 GELU | 线性层占主导；GELU 本身只有 3.779 s，不值得单独做硬件 |
| DeepStack、Patch Merger、Projector | 27.526 s（20.640+6.886） | 多尺度特征、合并、跨模态投影 | 计算量次于主干，但决定跨设备边界；输出隐藏状态约 5.27 MiB/图（FP16） |
| 视觉 LLM Prefill | 1003.155 s | 36 层大矩阵；FFN 668.157 s、投影 241.295 s、Attention 88.331 s | 全流程最大瓶颈；应放在 CUDA GPU、高算力 NPU或完整驻留的定制加速器上 |
| 文本 Prefill | 63.732 s | 短序列 GEMM/Attention；仍偏计算 | 与视觉 Prefill 使用同一语言层，通常不值得跨设备拆开 |
| KV Cache 写入与管理 | 已包含在 Prefill，未独立计时 | 容量、带宽和数据布局 | Prefill 与 Decode 分离会传输约 160.59 MiB FP16 KV；普通 1 GbE 不适合逐帧搬运 |
| 自回归 Decode | 1.562 s（本次 2 token） | M=1 GEMV、KV Attention、LM head；主要受权重带宽限制 | 当前每 token 约需读取 4.274 GB 权重；选型应优先看内存带宽和实际 Q8 kernel |
| Sampling、坐标/轨迹文本解析 | 未独立计时 | CPU 标量运算 | 耗时很小；可在主机侧完成，不需要 NPU |
| 动作头、安全逻辑和实时控制 | 当前模型测试未覆盖 | 小矩阵、约束输出、确定性 I/O | MCU/实时 CPU/FPGA 更合适；不应让非实时 Linux 大模型循环直接承担关节闭环 |

已测结论是：视觉编码器和 Prefill 的大矩阵主要是 compute-bound，Decode 主要是 memory-bound。每张新图的保守计算下界为 13.443 TFLOP；3 Hz 和 5 Hz 分别要求至少 40.3 和 67.2 TFLOP/s 的持续有效吞吐，尚未计入 Softmax、数据搬运和调度。因而，峰值达到这一数值仍不等于端到端达到 3–5 Hz。

## 3. 平台、通信与异构淘汰

### 3.1 先确定允许的切分边界

以5 Hz为最严格目标，每帧周期为200 ms。若通信最多占周期的20%，单向通信预算为40 ms。当前已知的两类中间数据为：

| 切分位置 | FP16/BF16 | Q8 payload | 40 ms内所需有效带宽 |
|---|---:|---:|---:|
| Vision Merger → LLM | 5.27 MiB | 2.80 MiB | FP16约138 MB/s；Q8约73 MB/s |
| Prefill → Decode，全部36层KV Cache | 160.59 MiB | 85.32 MiB | FP16约4.21 GB/s；Q8约2.24 GB/s |

据此冻结通信门槛：

1. **不允许逐层、逐算子在两块板之间来回传输。** 当前每算子下发和回读方案在单块ZCU102内部已经因往返开销被否决，跨板只会更差。
2. **默认不允许把Prefill放在主平台、Decode放在ZCU102。** 即使使用10GbE，Q8 KV单向传输仍需约89.5 ms，超过40 ms；ZCU102自身的Decode权重带宽也没有形成补偿收益。
3. **允许在Vision Merger前后切分，但最低应达到2.5GbE或USB 3.0有效带宽。** 1GbE传FP16隐藏状态约55.3 ms，不通过5 Hz门槛；传Q8约29.4 ms，理论通过但余量很小，只作过渡验证。
4. **如果主平台完成整个VLM，ZCU102只收发图片、Token掩码、坐标或动作向量，1GbE足够。** 这种划分通信最稳，但FPGA的论文贡献必须来自视觉Token削减、传感器前处理、动作头或确定性控制，而不能声称加速了整个Decoder。

推荐的默认拓扑因此调整为：

```text
相机/传感器
    │
    ▼
ZCU102：采集、ROI/视觉Token筛选、可选图像前处理
    │  10GbE为目标链路；1GbE只用于协议和小数据验证
    ▼
主AI平台：Vision Encoder + 视觉/文本Prefill + KV Cache + Decode
    │  坐标、轨迹或固定动作向量（小数据）
    ▼
ZCU102：动作头、安全约束、确定性I/O与实时控制接口
```

Prefill和Decode保留在同一主AI平台，KV Cache不跨板。ZCU102通过减少进入主模型的视觉Token来降低计算量，并负责小而固定的低时延子图；这比把完整KV Cache搬到FPGA更符合现有硬件条件。

### 3.2 ZCU102可用通信接口

| ZCU102接口 | 理论能力 | 实际用途 | 当前判断 |
|---|---:|---|---|
| PS端RJ45 | 1GbE；规划按100 MB/s有效载荷 | Prompt、压缩图片、结果、控制命令 | 已验证可用；不适合FP16视觉隐藏状态的5 Hz传输，更不适合KV Cache |
| PS端USB 3.0 Micro-AB，Device模式 | 5 Gb/s线速；规划按350 MB/s有效载荷 | 与Jetson/PC点到点传视觉隐藏状态 | 估算FP16/Q8隐藏状态为15.8/8.4 ms；需确认跳线、Device Tree和USB gadget，尚未实测 |
| P1 PCIe插槽 | PS PCIe Gen2×4 Root Port；规划有效约1.5 GB/s | 在ZCU102上安装10GbE NIC或其他Endpoint卡 | **不能直接接另一块Root Port主机板。** 可用于给ZCU102增加10GbE网卡；还需ARM64驱动和实测 |
| PL端4路SFP/SFP+ | 每路GTH可承载10GbE；4路可组成1×40GbE | 10GbE、40GbE或定制串行链路 | 适合作为最终低抖动链路，但需要新增IP、bitstream、驱动及光模块/DAC，不是板载即用高速网卡 |
| UART/CAN/GPIO | 低带宽、确定性较好 | 状态、调试、动作和安全控制 | 可保留为控制面，不传模型张量 |

ZCU102的PCIe插槽默认面向插入Endpoint卡；Jetson AGX Orin开发套件的x16插槽同样是主机侧插槽。因此，“一根PCIe线直接连接ZCU102和Jetson”不作为当前方案。若以后一定要直连PCIe，需要额外的PL Endpoint、FMC/线缆或定制载板，并单独解决枚举、DMA和驱动问题。

GTH与SFP+不是同一层。GTH是XCZU9EG PL内部的高速SerDes，负责串并转换、时钟恢复和信号均衡；SFP+是板上的可插拔电气/光学接口。ZCU102把Quad 230的4条GTH通道接到SFP0～SFP3。若要让普通Jetson或服务器网卡识别，完整数据通路应为：

```text
PL计算核 / 缓冲区
    ↕ AXI4-Stream
DMA或硬件包处理器
    ↕
10G/40G Ethernet MAC + PCS/PMA
    ↕
GTH SerDes → SFP+模块或DAC → 主AI平台网卡
```

单路10GbE在线路侧运行于10.3125 Gb/s，4路可以作为4个独立10GbE口，也可以由40G MAC/PCS完成通道对齐后组成1个40GbE口；插入4根线本身不会自动形成40GbE。AMD的ZCU102参考设计已经验证1×10GbE和1×40GbE两种接口，40G模式使用4×10.3125 Gb/s，用户侧通过AXI接口连接。若采用Aurora或自定义裸链路，可以降低协议开销，但标准Jetson网卡不能直接接收，因而不作为当前跨平台首选。

### 3.3 链路时间下界

下表按规划有效载荷速率计算，未计协议栈、内存复制、量化/反量化和排队。它只用于淘汰，不作为实测成绩。

| 链路 | 规划有效载荷 | Vision FP16 5.27 MiB | Vision Q8 2.80 MiB | KV FP16 160.59 MiB | KV Q8 85.32 MiB | 5 Hz通信门 |
|---|---:|---:|---:|---:|---:|---|
| 1GbE | 100 MB/s | 55.3 ms | 29.4 ms | 1683.9 ms | 894.6 ms | 仅Q8视觉边界勉强通过；KV淘汰 |
| 2.5GbE | 250 MB/s | 22.1 ms | 11.8 ms | 673.6 ms | 357.8 ms | 视觉边界通过；KV淘汰 |
| USB 3.0 Device | 350 MB/s | 15.8 ms | 8.4 ms | 481.1 ms | 255.6 ms | 视觉边界通过；KV淘汰 |
| 5GbE | 500 MB/s | 11.1 ms | 5.9 ms | 336.8 ms | 178.9 ms | 视觉边界通过；KV淘汰 |
| 10GbE | 1.0 GB/s | 5.5 ms | 2.9 ms | 168.4 ms | 89.5 ms | 视觉边界通过；KV淘汰 |
| PCIe Gen2×4规划值 | 1.5 GB/s | 3.7 ms | 2.0 ms | 112.3 ms | 59.6 ms | 视觉边界通过；KV仍不通过，而且当前两板不能直接连接 |
| PL 40GbE规划值 | 4.0 GB/s | 1.4 ms | 0.7 ms | 42.1 ms | 22.4 ms | Q8 KV仅在线路下界上通过；需PL直通、对端40G和端到端实测 |

40GbE的Q8 KV传输下界进入40 ms预算，并不自动推翻当前切分结论。当前ZCU102 Decode仍受2.158 GB/s实测顺序读带宽和约4.274 GB/token权重流限制；若数据还要经过PS Linux、TCP和DDR多次复制，40GbE线路速度也无法转化为端到端收益。40GbE只在PL侧MAC/PCS、DMA和计算流水直接连接，并且对端具备兼容40G接口时才值得验证。

### 3.4 现有异构推理研究如何通信

现有GPU/FPGA大模型研究大多把FPGA做成同一服务器内的PCIe加速卡，而不是用两块独立开发板通过以太网通信。它们取得收益的共同原因不是PCIe本身，而是减少跨设备状态、批量DMA并把传输隐藏在计算之后。

| 工作 | 异构切分 | 物理链路与数据路径 | 实际传输内容 | 通信优化及适用边界 |
|---|---|---|---|---|
| [ReCoVLM，FCCM 2026](https://doi.org/10.1109/FCCM68464.2026.00026) | Jetson Orin Nano完成视觉编码和跨模态Prefill，VU9P完成Decode | 可公开的会议摘要没有说明物理接口，不能据此写成PCIe或以太网 | 从阶段边界可确定需要交接Prefill状态、首token和控制信息，但具体格式尚未公开 | 先把视觉token压缩到5%，再进行异构执行；18.1 token/s是生成吞吐，不是图像帧率 |
| [GLITCHES，HPEC 2024](https://ieee-hpec.org/wp-content/uploads/2024/09/112.pdf) | A100/V100S完成Prefill，U280完成Decode | 同一服务器PCIe；GPU显存→主机内存→FPGA HBM | GPU按Transformer层产生并量化KV Cache；CPU还传首token、token ID和启动寄存器 | 每层KV传输与GPU后续层Prefill重叠；模型权重预加载到设备，不随请求发送 |
| [HPU，2025预印本](https://arxiv.org/abs/2504.16112) | L40S执行线性层/FFN，U55C执行Attention并保存KV | PCIe 4.0×8 QDMA；原型因缺少直接P2P而经CPU中转 | 每层Q/K/V、命令描述符和Attention结果 | 将256个Attention Head合并成大块，并用子Batch交错隐藏通信；通信仍约占生成时间10%，主要面向大Batch，不适合直接套用到机器人batch=1 |
| [INF²，2025预印本](https://arxiv.org/abs/2502.09921) | GPU执行主要矩阵运算，SmartSSD内FPGA就近处理Attention/KV | 主机和SmartSSD通过PCIe，SSD与板内FPGA也通过PCIe连接 | Query、新增KV和Attention结果；历史KV尽量留在存储侧 | 计算靠近KV数据，合并细碎写入并用流水隐藏搬运，避免完整KV反复经过主机互连 |
| [HillInfer，2026预印本](https://arxiv.org/abs/2602.18750) | GPU执行模型，SmartSSD FPGA管理分级KV | GPU、CPU内存与SmartSSD组成PCIe存储层次 | token重要性、被选中的KV和控制元数据 | 分级淘汰和自适应预取，只移动当前需要的KV，并把传输与GPU计算重叠 |

这些工作形成三种可区分的通信范式：

1. **阶段级交接。** Prefill与Decode只在边界处交换一次KV或状态，GLITCHES最接近这种做法。
2. **算子级协处理。** 每层交换Q/K/V和结果，HPU证明它需要PCIe 4.0×8、大块合并、大Batch和充分流水；不满足这些条件时通信很容易抵消加速收益。
3. **数据驻留。** INF²和HillInfer把KV留在FPGA/SmartSSD附近，只移动Query、增量KV或筛选结果；本质是“计算靠近数据”，而不是追求搬运完整KV的更快链路。

对本项目，10/40GbE是由ZCU102和Jetson均为独立开发板这一物理形态决定的，并非相关论文的主流连接方式。可借鉴的是状态驻留和通信流水：权重在两端预加载；KV留在主AI平台；ZCU102只发送压缩视觉状态、token掩码或动作结果；数据面从PL AXI4-Stream直接进入MAC/DMA，PS侧只保留配置、监测和异常处理。只有后续实测证明ZCU102 Decode快于主平台时，才重新开启“量化KV一次性交接”的40GbE候选。

### 3.5 总表：从入门板到高端平台

“Decode 理论上限”按 `内存峰值带宽 ÷ 4.274 GB/token` 计算，只表示完全连续读取权重、无其他开销时的天花板；没有公开带宽或不能直接运行当前 Q8_0 kernel 的平台不计算。实际值必然更低。

表内先列通用部署门槛较低的平台，再列中高端机器人/工作站平台；ZCU102和VEK280放在最后作为定制研究平台，不参加开箱即用性能排名。

| 档位 | 平台与形态 | 标称 AI 算力 | 内存与带宽 | 功耗口径 | 成本快照 | 完整 4B VLM 判断 | 本项目适用阶段与主要限制 |
|---:|---|---|---|---|---|---|---|
| 入门 | [RDK X5](https://developer.d-robotics.cc/en/rdkx5)，国产 BPU 单板 | 10 TOPS | 4/8 GB LPDDR4 | 官方页面给出 5V/5A 供电要求，不等于持续功耗 | A–B，随套件变化 | **不建议完整部署**：8 GB 小于当前约 9 GB 运行时需求 | 适合图像预处理、小视觉模型或机器人控制；4B 多模态模型需重新转换且容量不足 |
| 入门 | [Orange Pi 5 / RK3588S](https://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-5.html) | 6 TOPS NPU | 4/8/16 GB LPDDR4/4X | 5V/4A 是电源规格，不是实测功耗 | A–B | 16 GB 版容量可放入，**但不能据此认为 NPU 可完整运行** | NPU适合已支持的视觉图；LLM/VLM 常受 RKNN 算子、动态图和量化格式限制，退回 CPU 后性能有限 |
| 入门 | 实验室原始 **Ascend 310** 卡/模组 | 16 INT8 TOPS；芯片公开功耗低于 8 W | 容量和带宽取决于实验室具体 Atlas 卡或载板，尚未盘点 | `<8 W` 只指芯片，不含主机与载板 | **已有** | **待定**：先确认型号、显存和 CANN 版本 | 值得先做零成本兼容性门；当前 llama.cpp CANN 已验证列表主要是 310P/910B，不应把原始 310 当作即插即用 |
| 入门 | 图片中的 **Orange Pi AIpro 8T/20T**，Ascend 310B 系单板 | 8 或 20 INT8 TOPS；[20T 官方页](https://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-AIpro%2820t%29.html) | 图片商品含 8/16 GB 变体；新版 20T 官方为 12/24 GB LPDDR4X-4266 | 20T 官方给 65 W PD 电源，不能当作板卡实耗 | 图片商品券后 **1094.09 元起**；具体内存/算力 SKU 未确认 | 16/24 GB 容量可行；8/12 GB 边缘或不足，**软件风险高** | 可验证 CANN/ATC 转换后的视觉和 Prefill；llama.cpp 曾有 310B1 构建兼容问题，当前 GGUF Q8_0 不能默认直接加速 |
| 低功耗协处理 | [Hailo-10H M.2](https://hailo.ai/hailo-files/hailo-10h-m-2-et-product-brief-en/)，NPU 协处理器 | 40 INT4 / 20 INT8 TOPS | 4/8 GB LPDDR4/4X | 2.5 W typical，仅指模块 | 询价，另需 x86/ARM 主机 | **不能单独容纳当前完整运行时** | 低功耗视觉/小型生成模型协处理有价值；8 GB 容量和专用编译流程限制完整 RoboBrain 部署 |
| 入门 CUDA | [Jetson Orin Nano Super](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) | 67 sparse INT8 TOPS | 8 GB，102 GB/s | 7–25 W | 官方开发套件 [**$249**](https://developer.nvidia.com/embedded/buy-jetson?product=jetson_nano)，B | **不建议原样完整部署**：当前运行时约 9 GB | CUDA生态最好且便宜，可作为裁剪/更低比特模型基线；带宽给出的 Decode 绝对上限约 23.9 token/s |
| 中低端 CUDA | Jetson Xavier NX 16 GB，旧 CUDA 模组 | 21 TOPS | 16 GB，59.7 GB/s | 10–20 W | 官方 1KU 模组 **$899**；开发套件已 EOL，C | 容量可行，性能和生命周期一般 | 可完成基线，但 Decode 带宽上限约 14.0 token/s，Vision/Prefill明显弱于 Orin；不建议作为新异构平台采购 |
| 中低端 | [Orange Pi 6 Plus](https://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-6-Plus.html)，国产 CPU+GPU+NPU 单板 | 45 combined TOPS，不能视为单一 NPU 的 45 TOPS | 16/32/64 GB，128-bit LPDDR5 | 100 W 电源规格不是持续功耗 | 新平台，渠道价浮动，约 C/询价 | 32/64 GB 容量充足，**后端成熟度待测** | 适合比较国产统一内存平台；需要逐算子验证 NPU/GPU 协同，不能只用“合计算力”预测 VLM 延迟 |
| 中端 CUDA | [Jetson Orin NX 16 GB](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) | 157 sparse INT8 TOPS | 16 GB，102.4 GB/s | 10–40 W | 官方 1KU 模组 **$999**，载板另计，C | **可部署但余量较紧** | 当前最低一档较实用 CUDA 机器人主机；Vision/Prefill明显强于 Nano，Decode 带宽上限仍约 24.0 token/s；上下文和并发要受控 |
| 中端工作站 | [AMD Ryzen AI Max+ 395 开发平台](https://developer.amd.com/playbooks/user-guide/)，x86 APU | NPU 50 TOPS；实际 LLM 通常主要用 40-CU Radeon GPU | 128 GB LPDDR5X，256 GB/s | 120 W 平台 TDP | 整机询价，通常 D | **可部署，容量很宽裕** | 适合紧凑工作站或机器人上位机；Decode 带宽上限约 59.9 token/s。ROCm/Vulkan需实测，不能把 NPU 50 TOPS当成完整 VLM 性能 |
| 中高端 dGPU | [RTX 4000 SFF Ada](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-rtx-4000-sff-ada-datasheet-2616456-web-1.pdf)，低矮 dGPU | 19.2 FP32 TFLOPS；306.8 Tensor TFLOPS（厂商口径） | 20 GB GDDR6，280 GB/s | 70 W 板卡，主机另计 | 询价，约 C–D | **适合** | 低功耗机载 CUDA dGPU 候选；20 GB 足够，Decode 带宽上限约 65.5 token/s；摄像头、CPU和实时控制仍需主机平台 |
| 高端机器人 | [Jetson AGX Orin 64 GB](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) | 275 sparse INT8 TOPS | 64 GB，204.8 GB/s | 15–60 W（当前开发套件规格） | 官方开发套件 **$3499**，D | **适合，当前最均衡** | 机器人 I/O、CUDA、容量和功耗较平衡；适合承担 Vision Encoder、视觉/文本 Prefill 和 Decode，带宽上限约 47.9 token/s；可与 ZCU102 做异构研究 |
| 高端 NPU | [Atlas 300I Duo / Ascend 310P](https://e.huawei.com/cn/products/computing/ascend/atlas-300i-duo)，PCIe NPU 卡 | 280 INT8 TOPS / 140 FP16 TFLOPS | 48/96 GB LPDDR4X，408 GB/s | 最大 150 W，主机另计 | 询价，约 D–E | 容量和硬件性能适合，**转换工作量大** | 理论 Decode 带宽上限约 95.5 token/s；适合国产高算力对照。当前 llama.cpp 对 310P 的 Q8_0 `MUL_MAT`仍非完整支持，需 MindIE/ATC/自定义算子路线 |
| 高端 dGPU | [RTX PRO 4000 Blackwell Embedded](https://www.nvidia.com/zh-tw/products/workstations/rtx-embedded/)，嵌入式 dGPU | 第五代 Tensor Core，支持 FP4；厂商未在同表给统一可比的本模型指标 | 16 GB GDDR7，896 GB/s | 115 W 图形子系统，主机另计 | 询价，D–E | **适合，但 16 GB 余量有限** | 机载高带宽 CUDA 方案；Decode 带宽上限约 209.6 token/s。是否优于 Thor取决于主机、拷贝和 Vision/Prefill kernel，不能只看带宽 |
| 旗舰机器人 | [Jetson AGX Thor T5000](https://www.nvidia.com/en-gb/autonomous-machines/embedded-systems/jetson-thor/) | 2070 sparse FP4 TFLOPS | 128 GB LPDDR5X，273 GB/s | 40–130 W | 官方开发套件 **$5499**，E | **最适合机器人端完整部署** | Blackwell GPU、CUDA/TensorRT、摄像头和机器人 I/O 完整；Decode 带宽上限约 63.9 token/s。官方 Qwen2.5-VL-3B 在特定 vLLM、2048/128 配置下为 71.7 output token/s（并发1），不能换算成 RoboBrain 新图 Hz |
| 高端桌面 | [DGX Spark](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)，桌面 Grace-Blackwell | 1 PFLOP sparse FP4 | 128 GB LPDDR5X，273 GB/s | GB10 SoC TDP 140 W；240 W 电源 | 官方 **$4699**，E | **适合开发与大模型实验，不是首选机载件** | 容量和 CUDA 软件很好，Decode 带宽上限同样约 63.9 token/s；更像实验室开发机，缺少 Jetson 的机器人载板定位 |
| 高端服务器 | 实验室 [RTX A6000](https://www.nvidia.com/en-us/products/workstations/rtx-a6000/) 服务器 GPU | 38.7 FP32 / 309.7 Tensor TFLOPS | 48 GB GDDR6，768 GB/s | 300 W 板卡，服务器其余功耗另计 | **已有** | **适合，最应先做统一软件基线** | 新增成本最低且 CUDA 已跑通过；Decode 带宽上限约 179.7 token/s。功耗和体积不适合机器人上机，但最适合校准模型、精度和优化后端 |
| 定制研究 | 现有 **ZCU102 + 16 GB DDR4**，FPGA SoC | 没有可直接等同 GPU/NPU TOPS 的统一指标 | 16 GB DDR4-2133；线速 17.065 GB/s，实测顺序读 2.158 GB/s | 尚缺同口径板级实测 | **已有** | 已完成 PS-only 部署；PL完整模型尚未接通 | 适合低功耗、确定性延迟、结构化稀疏、动作头和实时 I/O；Decode 线速上限 3.99 token/s，单板追求绝对性能不占优 |
| 高端自适应研究 | [AMD VEK280 / Versal AI Edge](https://www.amd.com/en/products/adaptive-socs-and-fpgas/evaluation-boards/vek280.html)，AIE+PL+Arm | 超过 200 dense INT8 TOPS | 12 GB LPDDR4，192-bit、3733 Mb/s，推算约 89.6 GB/s | 无统一固定板级推理功耗，应以实际设计测量 | 官方 **$6995**，E | 容量勉强，工程工作量最大 | AIE承担 GEMM、PL承担稀疏/融合/控制；Decode 带宽上限约 21.0 token/s。适合研究新架构，不适合追求最快得到完整 VLM 基线 |

### 3.6 各平台与ZCU102的通信筛选

表中的“通过”只表示所列通信方式能承载视觉边界或小数据边界，不表示该平台的模型后端已经通过。凡是需要逐帧迁移完整KV Cache的组合均直接淘汰。

| 主AI平台 | 与ZCU102最现实的连接 | 可承载的切分 | 通信结论 | 平台结论 |
|---|---|---|---|---|
| RDK X5 | 板载1GbE ↔ ZCU102 PS 1GbE | 压缩图片、Q8视觉边界或小结果 | FP16视觉边界不通过5 Hz门 | **淘汰**：内存、算力和通信均无优势 |
| Orange Pi 5 | 板载1GbE ↔ ZCU102 PS 1GbE | 压缩图片、Q8视觉边界或小结果 | 同上 | **淘汰**：即使16 GB可装入，NPU后端与通信不足以支撑主方案 |
| 实验室原始Ascend 310 | 若为PCIe卡，经宿主机10GbE NIC；若为独立板，取决于实际网口 | 宿主机方案可传视觉状态；独立板待盘点 | 型号未知，暂不能通过通信门 | **条件保留**：先执行`npu-smi info`并确认载体和网口 |
| Orange Pi AIpro | 双2.5GbE ↔ ZCU102的多速率10GbE NIC/交换机，或先降速到1GbE | 2.5GbE可传FP16/Q8视觉状态；不能传KV | 2.5GbE视觉边界通过 | **条件保留**：通信可用，但CANN/Q8_0和实际算力仍需门控 |
| Hailo-10H | Hailo先插入x86/ARM主机M.2，再由主机连接ZCU102 | 取决于宿主机；不能单独成为主机 | 没有板到板直接链路 | **淘汰为主平台**：只能作为第三个协处理器，增加系统复杂度 |
| Jetson Orin Nano Super | 板载1GbE；或Jetson作USB Host、ZCU102作USB 3.0 Device | USB 3.0可传视觉状态；1GbE只适合Q8视觉或小数据 | USB方案通信可通过 | **淘汰原样4B方案**：8 GB容量先失败；可用于更小模型对照 |
| Jetson Xavier NX 16 GB | 1GbE或USB 3.0；M.2 PCIe不能直接接ZCU102 Root Port | USB 3.0可传视觉状态 | 通信可通过，但需新USB Device软件 | **淘汰新采购**：性能和生命周期不如Orin NX |
| Orange Pi 6 Plus | 双5GbE ↔ ZCU102多速率10GbE NIC/交换机 | FP16/Q8视觉状态、小结果 | 5GbE视觉边界通过 | **条件保留**：通信较好，后端成熟度需实测 |
| Jetson Orin NX 16 GB | 标准载板常见1GbE；USB 3.0；或采购带5/10GbE的第三方载板 | USB 3.0或高速载板可传视觉状态 | 默认1GbE载板不够稳，升级后通过 | **条件保留**：作为最低成本CUDA主机时，必须把高速载板计入成本 |
| Ryzen AI Max+ 395平台 | 官方开发平台10GbE RJ45 ↔ ZCU102 10GbE NIC/PL SFP+ | FP16/Q8视觉状态、小结果 | 10GbE通过 | **保留**：通信和容量好，主要风险是ROCm/Vulkan后端与120 W平台功耗 |
| RTX 4000 SFF Ada主机 | 主机10GbE NIC ↔ ZCU102 10GbE；GPU本身不直接连接ZCU102 | FP16/Q8视觉状态、小结果 | 配置10GbE后通过 | **保留**：适合机载GPU对照，但需把CPU、NIC和整机功耗一起计算 |
| Jetson AGX Orin 64 GB | **板载10GbE RJ45** ↔ ZCU102 10GbE NIC，或ZCU102 PL SFP+经兼容模块/交换机 | FP16/Q8视觉状态、小结果 | 现成主机侧10GbE，路径最清楚 | **首选**：计算、通信、功耗和开发难度最平衡 |
| Atlas 300I Duo | PCIe卡插宿主机；宿主机10GbE NIC ↔ ZCU102 | FP16/Q8视觉状态、小结果 | 宿主机配10GbE后通过 | **条件保留**：通信不是主风险，CANN模型转换和主机功耗才是 |
| RTX PRO 4000 Blackwell Embedded主机 | GPU插宿主机；宿主机10GbE NIC ↔ ZCU102 | FP16/Q8视觉状态、小结果 | 10GbE通过 | **保留**：带宽强，但整机集成和价格需询价 |
| Jetson AGX Thor | 板载5GbE RJ45；QSFP28为更高速接口，但与ZCU102 SFP+的速率和模块兼容需实测 | 5GbE已能传视觉状态；不传KV | 5GbE视觉边界通过；QSFP28不作为未经验证的25GbE承诺 | **高预算保留**：计算最强，但通信并没有使其相对AGX Orin形成决定性优势 |
| DGX Spark | 板载10GbE RJ45 ↔ ZCU102 10GbE；ConnectX-7只在兼容网络设备下使用 | FP16/Q8视觉状态、小结果 | 10GbE通过 | **保留为实验室对照，淘汰机载方案** |
| 实验室RTX A6000服务器 | 当前1GbE链路用于控制；若传视觉状态，服务器增加10GbE NIC | 现有1GbE适合输入/输出；10GbE适合中间视觉状态 | 当前链路不做跨板Tensor性能结论 | **首要软件基线**；异构性能实验前补10GbE |
| VEK280 | 不建议与ZCU102串联；它本身含AIE、PL和Arm | 作为ZCU102替代平台重新划分片上任务 | 串联两块FPGA没有通信收益 | **不作为“ZCU102+？”候选**，只作为升级替代路线 |

通信筛选后的主候选缩小为三类：

1. **首选：ZCU102 + Jetson AGX Orin 64 GB + 10GbE。** AGX Orin完成Vision Encoder、全部Prefill、KV Cache和Decode；ZCU102完成传感器接入、输入侧Token/ROI削减、动作头和实时安全通路。
2. **最高性能：ZCU102 + Jetson Thor + 5/10GbE。** 任务划分与Orin相同。只有Orin实测算力不够、Thor能显著改善端到端延迟时才值得增加成本。
3. **实验室基线：ZCU102 + A6000服务器 + 10GbE。** 适合尽快验证算法和通信协议，但300 W GPU及服务器功耗不作为机器人最终能效结果。

Ascend 310/Orange Pi AIpro属于国产NPU条件候选：通信本身可以通过2.5/10GbE解决，但在CANN关键算子覆盖和完整模型实测通过前，不进入最终二选一。

## 4. 需要特别解释的三个平台

### 4.1 实验室 Ascend 310

它的优点是新增硬件成本为零，因此应先测，不应先淘汰。但“实验室有昇腾 310”还不足以估算 RoboBrain 性能：原始 310、310B 和 310P 的算子、内存及软件支持不同，Atlas 卡、模组和单板也不同。

当前 `llama.cpp` CANN 文档列出的已验证硬件主要是 Atlas 300I Duo（310P）和 910B；310P 对 Q8_0 的数据变换已有支持，但量化 `MUL_MAT` 仍标为不支持。这意味着当前 GGUF 文件即使能加载，也可能在最关键的线性层退回其他路径。实验室设备的第一步不是跑总模型，而是记录：

```text
npu-smi info
cat /usr/local/Ascend/driver/version.info
free -h
lspci -nn | grep -i 19e5
```

随后用一个 `2560×9728` 的线性层和视觉 Attention 子图做最小 CANN/ACL 基准，再决定采用 `llama.cpp CANN`、MindIE，还是将 Hugging Face 模型转换为昇腾原生图。

### 4.2 图片中的 Orange Pi AIpro

截图中的标题同时写有 8/20 TOPS、8G/16G，1094.09 元是“起售价”，不能据此认定实验室拿到的是 20 TOPS、16 GB 版本。新版 AIpro 20T 官方规格则为 12/24 GB。必须读取实机 `npu-smi info` 和 `free -h` 后再填具体型号。

若为 8 GB 版，当前约 9 GB 的运行时没有安全余量；若为 16/24 GB，容量可行，但主要风险转为 Q8_0、视觉算子和 CANN 版本的兼容性。它适合做“低成本国产 NPU 能否运行 4B VLM”的对照，不宜在未通过算子覆盖率测试前作为主平台。

### 4.3 Jetson Thor

Thor 是当前表中最完整的机器人端高性能平台：128 GB 统一内存、273 GB/s 带宽、40–130 W 功耗档以及 CUDA/TensorRT 软件栈都适合本模型。NVIDIA 公布的 Qwen2.5-VL-3B 基准为并发 1 时 71.7 output token/s，并发 8 时总计 356.86 output token/s；条件为 JetPack 7.0、CUDA 13.0、TensorRT 10.13、vLLM、输入/输出长度 2048/128。

这个结果证明 Thor 能高效运行同量级 VLM，但没有给出新图 Vision Encoder 和 Prefill 的单帧首 token 时间，也不是 RoboBrain2.5-4B 的结果。因此，Thor 很可能显著缩短本项目的推理时间，但是否达到 3–5 张“全新图片”每秒仍须用同模型、同图片和同输出格式实测。更现实的 3–5 Hz 路径仍是关键帧完整 VLM、跨帧视觉/KV复用、视觉 token 减少和固定动作头。

## 5. 选型结论

| 目标 | 建议平台 | 与ZCU102的通信 | 理由 |
|---|---|---|---|
| 立即建立高性能正确性与性能基线 | **实验室 RTX A6000** | 当前1GbE只传输入/输出；跨板Tensor实验前增加10GbE NIC | 已有、CUDA环境最熟悉；先排除GGUF CPU-only或低效backend对结果的干扰 |
| 零采购成本验证国产 NPU | **实验室 Ascend 310** | 先确认它是独立板还是PCIe卡；卡形态由宿主机10GbE连接ZCU102 | 先做型号、内存、CANN和关键算子门控；失败也能形成有边界的对照证据 |
| 低成本国产单板探索 | **Orange Pi AIpro 16/24 GB** | 2.5GbE连接ZCU102新增多速率NIC/PL网络口 | 视觉边界通信可通过，但软件适配风险高于Jetson |
| 最低可用 CUDA 机器人原型 | **Orin NX 16 GB** | USB 3.0，或选配带5/10GbE的载板；默认1GbE不作为正式Tensor链路 | 成本和功耗较低，能容纳模型；内存余量、带宽和载板条件有限 |
| 性能、功耗、开发难度最平衡 | **Jetson AGX Orin 64 GB + ZCU102** | **10GbE**；Orin已有10GbE，ZCU102侧增加10GbE NIC或PL SFP+数据通路 | Orin保留Vision、Prefill、KV和Decode；FPGA做输入Token削减、动作头和实时通路 |
| 追求机器人端最高性能 | **Jetson Thor + ZCU102** | 先用5GbE；更高速QSFP28与ZCU102的兼容性另测 | Thor运行完整VLM；FPGA承担有实测价值的前后端子图，不能迁移完整KV |
| 研究国产高算力 NPU | **Atlas 300I Duo宿主机 + ZCU102** | 宿主机10GbE NIC | 容量、带宽和算力充分，但CANN/量化算子适配工作量大 |
| 研究高端自适应计算架构 | **VEK280 或后续 Versal Gen2** | 替代ZCU102，不与ZCU102串联 | AIE+PL能形成更完整的定制架构；采购成本和开发周期较高 |

当前最合理的顺序是：先在A6000上建立同模型的CUDA基线，同时对两套已有昇腾设备做兼容性门；通信原型先沿用1GbE验证消息格式，再用USB 3.0或10GbE验证视觉张量，正式异构方案以10GbE为准。若采购机器人主机，默认选择AGX Orin 64 GB；只有它的端到端实测不足且预算允许时再上Thor。无论选择哪一个主平台，都不再采用“主平台Prefill、ZCU102 Decode”的KV跨板方案。

## 6. 资料与数据边界

- 本项目阶段计时与 Roofline：[`2026-08-22-zcu102-operator-bottleneck.md`](../reports/2026-08-22-zcu102-operator-bottleneck.md)
- 异构边界、KV传输量和 Decode 权重流：[`partition_gate_20260901/report.md`](../fpga_accel/heterogeneous_partition/artifacts/partition_gate_20260901/report.md)
- 当前 `llama.cpp` CANN 支持说明：[`CANN.md`](../third_party/llama.cpp/docs/backend/CANN.md)
- Jetson 当前规格与官方价格：[Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)、[Jetson FAQ](https://developer.nvidia.com/embedded/faq)
- Thor 同量级 VLM 参考基准：[Jetson Benchmarks](https://developer.nvidia.com/embedded/jetson-benchmarks)
- 原始 Ascend 310 为 16 INT8 TOPS、芯片功耗低于 8 W 的资料：[Huawei ICT Insights](https://e-file.huawei.com/-/media/EBG/Download_Files/Publications/en/ICT25-en.pdf)
- ZCU102的PS 1GbE、USB 3.0、PCIe Gen2×4 Root Port及PL SFP+硬件依据：[ZCU102 UG1182](https://docs.amd.com/v/u/en-US/ug1182-zcu102-eval-bd)
- ZCU102的1×10GbE、1×40GbE及AXI用户接口参考实现：[AMD XAPP1326](https://docs.amd.com/v/u/en-US/xapp1326-10g-40g-switching)
- GPU Prefill与FPGA Decode经PCIe交换分层KV的实现：[GLITCHES，HPEC 2024](https://ieee-hpec.org/wp-content/uploads/2024/09/112.pdf)
- PCIe QDMA逐层卸载Attention并以合并和流水隐藏通信：[HPU](https://arxiv.org/abs/2504.16112)
- 将KV驻留在SmartSSD附近处理的两种实现：[INF²](https://arxiv.org/abs/2502.09921)、[HillInfer](https://arxiv.org/abs/2602.18750)
- AGX Orin开发套件10GbE与PCIe接口：[AGX Orin Developer Kit Guide](https://developer.nvidia.com/sites/default/files/akamai/Jetson_AGX_Orin_Developer_Kit_RG_0.pdf)
- Thor的5GbE、QSFP28及40–130 W规格：[Jetson Thor](https://www.nvidia.com/en-gb/autonomous-machines/embedded-systems/jetson-thor/)
- Orange Pi 6 Plus双5GbE接口：[Orange Pi 6 Plus](https://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-6-Plus.html)
- DGX Spark的10GbE和ConnectX-7接口：[DGX Spark Hardware](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)

表中理论上限不是实测成绩；峰值 TOPS 不是端到端吞吐；电源适配器额定功率不是板卡实耗。平台最终排序必须由同一 RoboBrain 模型、相同图片、相同 token 数、相同精度门和墙上功耗测量完成。
