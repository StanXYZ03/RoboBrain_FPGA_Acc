# RoboBrain2.5-4B / ZCU102 平台无关稀疏协处理研究计划

版本：2026-09-04（加入多切入点Oracle门和驻留条件）  
状态：L2@87.5%冻结为安全基线；BF16多切入点烟雾门完成，视觉块1@75%为临时候选；转入最终Q8_0 GGUF两级复验  
目标平台：Xilinx ZCU102（XCZU9EG，PS DDR4 16 GiB）

## 2026-10-10 VSR 短决策发展实验（最新权威更新）

VSR zero-shot dev 340 条已完成 edge576、True/False、2-token 稳定输出预算曲线。Dense 准确率
0.7912；v2 K96/K144 为 0.6853/0.7382，相对 Dense 0.8662/0.9331；同预算相对修正 L2
均只提高 0.0206，配对 bootstrap 95% CI 仍跨 0。当前 compact HLS 仅支持 K<=96；K96 距
0.88 宽松开发线差 4 条，K144 虽过质量线但不满足现有硬件合同。v2 K192 因冻结整数 scorer
内部 cap16 审计选择不可行而在写记录前 fail-closed，未修改已审计核心。

按本轮真实 G、实测 A6000 阶段时延及已接受 200 MHz HLS 周期公式，v2 K96 在 1 GbE 串行
往返下投影为 91.56 ms，对 Dense model inference 88.98 ms 无净收益；假设 10 GbE 为 76.55 ms
（1.162x）。这些是分析投影，不是 Jetson/ZCU102 实测。下一阶段不得靠扩大 K 推进：固定
K96/3x3/整数布局，在 VSR zero-shot train 发展成对关系保持目标；同时把 PCIe/共享内存或
避免完整 block0 特征往返的切点设计作为架构前提。证据见
`reports/token_selector/2026-10-10-vsr-zeroshot-dev-budget-curve.*` 和
`reports/token_selector/2026-10-10-vsr-zeroshot-dev-outcome-analysis.*`。

## 2026-10-09 外部确认结果（权威更新）

SpatialEval VQA 三臂已各完成 4500 条。项目预注册等权宏平均为 Dense 0.583778、
L2@K96 0.439111、v2@K96 0.434444；v2/Dense 仅 0.744195，v2-L2 的分层配对 bootstrap
95% CI 为 [-0.013778,+0.004444]。因此两个外部门均失败，当前固定 K96 v2 不再作为主方案晋级。
SpatialEval test 已经解封，今后不得在该集合调 K、权重或锚点，也不得再次称其为未见外部集。

平均视觉 token 从 468.98 降至 96（减少 79.53%），但 A6000 软件 elapsed 从 1426.78 ms
增至 1472.96 ms。Dense Decode 为 1220.92 ms，占 85.57%；即使所有非 Decode 阶段理想化
为零，Amdahl 上限也只有 1.169x。Token Selector 因而只能作为联合方案的一部分：下一轮在
原发展数据研究自适应预算、空间关系保持，并同步减少自回归输出长度或引入非自回归结构化头。
新版冻结后必须更换未观察公开基准做一次外部确认。正式证据见
`reports/token_selector/2026-10-09-spatialeval-three-arm-summary.*` 与
`reports/token_selector/2026-10-09-spatialeval-external-failure-analysis.*`。

## 2026-10-09 外部确认协议（权威更新）

edge576 CV-Bench 仍是已观察的发展集，不能继续承担泛化确认。外部确认现已固定为
SpatialEval VQA test 的三个官方可评分任务：Spatial-Map、Maze-Nav、Spatial-Grid，各 1500 条，
共 4500 条。Spatial-Real 135 条不进入主门，因为固定官方 commit `d82ba382…` 的 evaluator
没有该任务的有效 handler；官方也没有跨四任务总分。

正式预注册继任配置为
`configs/2026-10-09-spatialeval-vqa-three-arm-external-confirmation-r2.preregistered.json`
（SHA256 `bd8e1a23…708014`）。r1 在 Dense 产生 93 条、尚未读取或评分答案时发现 checkpoint
直接固定不完整，已标记 `R1_SUPERSEDED_BEFORE_SCORING`；r2 不改算法、数据、Prompt、指标、
门槛或输出路径，直接 pin v2 实际加载的三个 checkpoint 和不可变 Dense 前缀快照。它固定
Dense、修正整数 L2@B1/K96 和 v2@B1/K96 三臂，
共享 edge576 输入、官方解释型 Prompt、BF16/SDPA、greedy、max64。逐条主判断复现官方
parser 后的答案子串比较；严格相等只作诊断。官方只定义逐任务准确率，本项目额外定义的
三任务等权宏平均必须明确称为“项目预注册汇总”，不得称官方总分。

外部确认门保持与发展阶段一致且在看结果前冻结：v2/Dense 项目宏平均保留率 `>=0.90`；
按 `task::question_id` 分层的配对 bootstrap（seed 20261009、5000 次）中，v2−L2 的 95% CI
下界 `>=0`。逐任务结果完整报告但不另设门。标签已经程序化物化并做聚合审计，因此本实验
不是 label-blind；但这些标签从未用于 Selector 训练、调参或路由，运行器也会在机制入口前
移除所有 oracle 字段。观察本次输出后禁止在该 test split 上调整 Selector 或门槛。

截至当前，r2 的 35 个数据/代码/checkpoint/模型身份 pin 和 35 项 CPU 合同测试通过，4500
条问题对应的 1459 个去重图像已在 `/tmp` 逐图校验；Dense 第一臂正从固定的 93 条未评分
前缀 append-only 续跑。任何 A6000
软件计时都不是 Jetson/ZCU102 端到端加速或功耗结论。

## 2026-10-08 执行结论（权威更新，覆盖旧计划）

2026-10-08 已按 edge576 冻结协议真实执行三臂发展性实验。本节给出执行结论，凡与下文旧计划
冲突，以本节和 `CURRENT_STATUS.md` 顶部为准。

1. **r3 作废、r4 接管。** r3 首次真实 GPU smoke 因冻结 dense 生产者不返回 `output_token_ids`，
   在写文件前的归一化阶段 fail-closed，未写任何文件；r3 标记 `superseded`（字节全部保留、
   不得重新运行）。r4 预注册 config SHA256
   `4f11d47981fd6c04c07608eba8b7e85e27e035bcb8d2d459a7ebc8a8c4b682c5`、**63 个 pin 全部复核
   通过**，修复为 `output_token_ids` 可选、`output_token_count` 必备，且 token 计数只由真实字段
   映射、**绝不伪造**。
2. **三臂 full2638 完成。** 三臂 smoke 全通过；Dense、修正整数 L2 与 v2 均完成 2638 条唯一记录
   且三臂同 `sample_id` 集合。raw SHA256：Dense `707868c6…e784`、L2 `e11dba93…8a8c`、
   v2 `8b2a2885…92f8`。
3. **预注册发展门通过，但仅限发展集。** 官方 `cv_bench_accuracy`：Dense `0.8739581555`、
   L2 `0.7937869555`（相对同输入 Dense `0.908267`）、v2 `0.8078459446`（相对同输入 Dense
   `0.924353`）；诚实解析失败 36/42/41，失败计为错误。`task::source_dataset` 分层配对
   bootstrap（seed `20261007`、5000 次、95%）给出 v2−L2 `+0.0140590` CI
   `[+0.0044580,+0.0233407]`。预注册两门（v2/Dense `>=0.90`、v2−L2 CI 下界 `>=0`）全部通过，
   状态记为 `DEVELOPMENT_GATES_PASS_GO_EXTERNAL_VALIDATION`；**这只允许进入一次未调参外部
   确认，不代表论文级精度或外部泛化已确认**。
4. **时延口径。** 上述均为软件时延：L2/v2 `application_end_to_end` mean/p95 分别为
   `150.08/190.14 ms` 与 `185.06/240.65 ms`，v2 `selector_cpu_roundtrip` mean/p95
   `52.79/86.23 ms`。Dense 没有 `application_end_to_end_seconds` 字段，必须为 `null`；其
   `elapsed_seconds` 只映射为 `model_inference` mean/p95 `157.04/196.48 ms`，不得与
   application 字段直接比较。软件 selector 不是 FPGA/板端实测，不构成加速结论。
5. **证据。** 正式汇总
   `reports/token_selector/2026-10-08-edge576-cvbench-three-arm-r4-summary.{json,csv,md}`，
   artifact SHA256 `1957c323…6549`；汇总器与测试 28 OK。
6. **下一里程碑。** 在查看任何结果前冻结一个此前未用于调参的公开空间 VLM 外部确认基准
   （样本、Prompt、官方指标、阈值、模型身份、输出路径全部预先固定），然后**只运行一次**
   确认；不得继续在 CV-Bench full2638 上调整 v2。并行工程任务：接入 Jetson Xavier NX 后
   重测真实阶段/链路延迟；ZCU102 只在净收益合同下评估 selector，当前不宣称 FPGA 加速。

## 2026-10-07 边缘部署运行域与成功判定（权威更新）

本节覆盖下文与输入分辨率、视觉 Token 范围和 Selector 晋升标准冲突的旧条款。项目目标是
**资源受限边缘部署**，不是证明一个固定 K96 Selector 能覆盖 Qwen3-VL 的全部动态高分辨率范围。

### 边缘部署主运行域

1. 未来主实验对所有样本统一使用保持长宽比的动态缩放，并将 Patch Merger 后、进入 LLM 的
   视觉 Token 数限制为 `G <= 576`。以当前 `patch_size=16`、`spatial_merge_size=2` 计算，
   候选 `max_pixels` 为 `576 * (16*2)^2 = 589824`；实际执行前必须以 processor 产生的
   `image_grid_thw` 验证每条记录确实满足 `G <= 576`，不能只相信像素参数。
2. 不按运行结果删除高 Token 样本。应对完整 CV-Bench 2638 样本重新预处理，使 Dense、L2
   对照和候选 Selector 接收完全相同的边缘分辨率输入。
3. 当前原生动态分辨率数据（中位数 G=300、P90=1400、最大 G=3588、560/2638 条
   `G>1024`）保留为冻结的高分辨率压力测试，不再作为边缘主档位的晋升阻断门，也不得
   反过来重写其既有 `DEVELOPMENT_STOP` 结论。
4. 主硬件预算保持 `K=96`、固定规则布局；在 `G<=96` 时 identity bypass，在 `96<G<=576`
   时执行压缩。K64/K128/K160 或更高分辨率只能作为预注册消融，不得混入主结果。

### 模型质量与“成功”判定

1. 主指标继续使用公开 CV-Bench 的官方 `cv_bench_accuracy`：
   `((ADE20K + COCO)/2 + Omni3D)/2`。不得重新使用 General/Pointing/Grounding/Trajectory
   四个 API 路由的 PCK、IoU、ADE/FDE 作为 Selector 的主晋升门，也不得自行发明 E/G/Q、
   retained mass 或合成“成功率”替代官方主分。
2. 边缘 Dense 基线必须在**同一 G<=576 预处理、同一模型、同一 Prompt、同一 greedy
   解码和同一输出上限**下重新运行。质量保留率定义为
   `selector_cv_bench_accuracy / dense_edge576_cv_bench_accuracy`，不得拿边缘 Selector
   与原生高分辨率 Dense 混比。
3. 下一候选的发展门固定为：质量保留率 `>=90%`；相对同切点、同 K 的修正整数 L2
   硬件基线，按 `task::source_dataset` 分层配对 bootstrap 的 95% CI 下界 `>=0`；同时满足
   K96/87 real+9 context（若该布局沿用）、无标签泄漏和记录完整性合同。
4. 使用 `evaluation/cv_bench_metric_v2.py` 的显式 `correct_key` 与 `parsed_answer_key`，分别
   报告主分、解析成功数、解析失败数和解析率；解析失败按错误计，不允许由布尔 correctness
   字段伪造 100% 解析率。
5. full2638 是已反复观察的发展集。通过发展门只表示允许在一个新的、未用于调参的公开空间
   VLM 基准上做一次预注册确认，不能直接声称论文级泛化、板端加速或精度无损。
6. 最终论文同时报告边缘主档位和原生高分辨率压力测试，但论文的主要部署结论限定为
   `G<=576`。压力测试用于说明适用边界，不用不属于目标运行域的极端输入否定边缘方案。

### 下一次实验的冻结顺序

先生成并审计 `edge576` manifest（2638 条、逐条记录 `image_grid_thw/G`），再分别运行
Dense@edge576、修正整数 L2@K96 和预注册候选@K96；最后用未来评测协议 v2 只读计分与审计。
在 manifest、模型身份、代码 SHA、成功门和输出路径全部冻结前，不启动新的全量 GPU 推理。

## 2026-10-05 权威更新（覆盖下文冲突条款）

下文保留了 2026-09-04 时的计划和阶段记录，便于审计；凡与本节冲突，以本节和 `CURRENT_STATUS.md` 顶部为准。

1. **异构平台已暂定为 Jetson Xavier NX + ZCU102。** Xavier NX 负责 RoboBrain2.5 主模型和规则稠密计算；ZCU102 只承担经过净收益门验证的 Token 评分、空间约束、选择、压紧及可选连续子图。Jetson 尚未接入，因此当前只做平台无关合同、A6000 模型实验和 ZCU102 单端工程验证。
2. **正式模型质量标准改为公开基准的官方指标。** 当前唯一完成全量正式评测的是 CV-Bench full2638 的 `score(...)["cv_bench_accuracy"]`。General/Pointing/Grounding/Trajectory 是 API 路由，不再作为 Selector promotion 的四个独立任务，也不得用 PCK、IoU、ADE/FDE、NDCG 或 retained mass 替代当前官方主分。以后增加公开空间 VLM 基准时，分别沿用其官方指标，不自行发明合成总分。
3. **当前冻结对照更新为 B1、K96、纠正整数 L2 + 3x3 anchor。** dense BF16 主分 0.872730；纠正整数 L2 主分 0.766852。该方案约保留 dense 分数的 87.87%，是硬件规则基线，不是精度等价结论，也不是足够的算法创新。
4. **当前 Prompt 条件分支没有通过正式门。** full2638 Prompt 相对整数 L2 增量 +0.010510，95% CI [-0.005430,+0.025699]；同图双分辨率反事实交互效应 -0.07，95% CI [-0.20,+0.06]。因此禁止继续调 `G>512` 阈值或把它写成因果机制；Prompt 条件化仍可作为研究目标，但必须换成新的监督/评分机制并重新预注册。
5. **禁止用 G、数据源名或任务标签做开发路由。** 新候选第一轮必须在相同切点、相同 K、相同解码和相同官方 scorer 下与纠正整数 L2 做单变量配对比较；只有未用于候选选择的数据上增量 CI 下界 >0，才允许进入 full2638、HLS 适配和异构端到端实验。
6. **精度运行时不再预先锁死为旧 GGUF Q8_0。** BF16 可用于算法筛选和教师信号；最终部署主张必须在 Xavier NX 实际采用的量化格式和运行时上复验。旧 Q8_0 GGUF 结果保留为 ZCU102 PS 端容量与软件基线，但不得强迫未来 Xavier 路径使用同一格式。
7. **现有 Selector IP 是工程基线，不是算法有效性证据。** 含 `Wv_q/query_q` 的整数投影流水线已通过 HLS、OOC route 和最小 Zynq BD；但其 Prompt 算法效用未确认。新算法开发门通过前，不继续为该评分公式生成 bitstream、驱动或板端性能结论。
8. **1 GbE 往返目前没有明确净收益。** K96 的现有解析投影为 FPGA 计算均值约 9.16 ms、1 GbE 传输约 37.29 ms，投影应用端约 183.71 ms；与软件 Selector 的约 182.87 ms 基本持平。后续必须实测 Xavier 链路，并比较高带宽连接、计算/通信重叠或“视觉前缀与 Selector 同侧驻留”，不能只报告 FPGA 核延迟。

当前证据边界见 `reports/token_selector/2026-10-05-post-counterfactual-evidence-ledger.md`；后续新实验使用 `configs/evaluation_protocol_20261005.json`（SHA256 `2cdc375e...ddb3`），旧 `configs/evaluation_protocol.json` 仅为历史实验复现保留。下一候选机制在独立策略审查任务 `20261005-001` 返回前不冻结。

## 1. 研究目标

在已经完成 RoboBrain2.5-4B PS 端部署、以太网交互和官方四种推理模式验证的基础上，开展以下两条主线的算法—硬件协同研究：

1. 由 ZCU102 在模型内部生成并执行结构保持的视觉 Token 压缩、块稀疏描述和数据重排，减少主 AI 平台的实际工作量。
2. 只把经端到端收益门证明有利的动态/不规则稀疏子图下沉到 PL；规则稠密 GEMM 默认交给未来选定的 GPU/NPU/AIE 主平台。

最终目标：

- Pointing、Grounding、Trajectory等结构化具身输出以独立新图3–5 Hz为长期目标；连续帧复用结果另行报告。
- 保留普通语言交互能力。
- 给出完整的精度、延迟、吞吐、功耗和 FPGA 资源利用结果。
- 形成可投 FPL、ASAP、ICCAD、DAC 或 MICRO 的算法—硬件协同成果。

> 3–5 Hz在本文中定义为模型端到端图片推理频率，不包含机器人控制、绘图或网络页面时间。未经裁剪的完整4B VLM不能仅靠ZCU102达到该目标，必须由主AI平台与模型工作量削减共同完成。

### 1.1 任务定义修正

`General / Pointing / Grounding / Trajectory` 是 RoboBrain 官方推理 API 的四种输出路由，不是学界公认的四类具身智能任务，也不是四个独立网络。当前官方封装主要通过追加 Prompt、约束输出格式以及选择点/框/轨迹解析器来区分模式，四种模式最终调用同一个自回归模型。

本项目按计算特征将其归纳为：

| 工作负载类别 | RoboBrain 模式或能力 | 输出形式 |
|---|---|---|
| 自由文本 | General | 可变长度文本 |
| 空间定位 | Pointing、Grounding；Affordance 通过 Pointing 表达 | 点或边界框 |
| 空间轨迹 | Trajectory | `(u,v,d)` 路径点序列 |

该归纳只用于软件策略、数据集和评价指标组织；PL 不复制三套数据通路，也不硬编码任务名称。

### 1.2 冻结ZCU102的平台无关角色

无论未来主计算平台是Jetson、昇腾、服务器GPU还是其他NPU，ZCU102固定定位为：

> **模型工作量削减协处理器（model-workload reduction coprocessor）**：在一个或少数几个粗粒度模型边界上，对视觉Token进行评分、结构约束、选择、合并、压紧和索引生成，并输出主平台可以直接消费的紧凑Token及稀疏描述符。

固定职责分成三层：

1. **控制面：**接收序列长度、Token预算、原始二维坐标、MRoPE/DeepStack映射和小尺寸Prompt摘要；生成任务无关的保留掩码、块位图和数据布局描述符。
2. **数据面：**以AXI4-Stream/AXI DMA完成Token评分、规则块选择、全局锚点保护、Token合并、压紧和位置索引同步，避免主平台对已删除Token继续执行后续层。
3. **可选计算面：**只有当`FPGA计算 + 双向通信 < 主平台被替代计算`时，才执行块稀疏Attention、稀疏GEMM/GEMV或KV筛选；否则PL只输出掩码和紧凑Token，不与主平台竞争稠密矩阵乘法。

这一角色不依赖具体主平台，但依赖明确的阶段边界和统一张量合同。平台选择只决定物理传输后端、主平台适配器和收益阈值，不改变PL的Token/稀疏描述符语义。

### 1.3 明确不再作为默认主线的方案

- 不默认把完整Vision Encoder、36层Prefill或完整Decoder搬到ZCU102。
- 不在两块板之间逐层传输Q/K/V、FFN中间张量或完整KV Cache。
- 不以ZCU102替代主平台的规则稠密GEMM来论证异构收益。
- 不把动作头、传感器I/O或机器人闭环控制计入本计划的模型端到端加速贡献。
- 已完成的稠密Q8×Q8 DATAFLOW和PL驻留FFN不删除；它们作为数值、资源、带宽、融合和可选稀疏计算面的硬件基线。

### 1.4 FPGA子图成立的最低条件与当前创新边界

结合已发表的边缘Transformer/LLM FPGA工作，FPGA子图至少满足下列一项，才值得进入RTL：

1. **单向输入：** FPGA从原始图像或上游流开始执行连续前缀，只向主平台发送一次压紧后的Token和位置元数据。主平台不先生成稠密中间张量再送到FPGA、取回后继续算。
2. **状态或连续子图驻留：** 权重、KV或多个连续算子长期留在FPGA/HBM/DDR中，主平台按较大块传入任务，避免每层或每个算子往返。

这两项是必要条件而不是充分条件。HeatViT（HPCA 2023）和Fitop-Trans（FPL 2024）把Selector与后续Transformer计算放在同一FPGA流水线；DFX（MICRO 2022）、FlightLLM和Hummingbird把模型或Decoder连续子图及权重驻留在FPGA；HILOS（ASPLOS 2026）把Attention放到KV所在的近存储FPGA；HPU则依靠PCIe大块传输、FPGA侧KV驻留和批量Head处理。共同点是避免“小算子跨板来回搬”。

据此把当前创新点限定为三个互相约束的部分：

- 面向Pointing/Grounding/Trajectory的空间保持Token削减，同时维护原二维索引、MRoPE、DeepStack和全局锚点；
- 从Patch/浅层视觉块到评分、选择、合并、压紧的单向驻留数据流，而不是独立外挂Selector；
- 用任务精度、被删除计算量和跨板字节共同选择切入点，并以实测通信否决没有净收益的划分。

在证明早期切入点的精度上界之前，不开发新的Selector RTL。若使用真实框的几何Oracle在早期切入点仍不能保住精度，则停止“外置Token选择器”路线；若只有较深切入点通过，则必须把从输入到该点的连续视觉前缀一并驻留，不能只跨板执行选择。

## 2. 已有基线与可行性边界

### 2.1 已验证条件

- RoboBrain2.5-4B Q8_0 主模型约 4.69 GB。
- Q8_0 视觉投影模型约 454 MB。
- 16 GiB PS DDR4 已完成跨 16 GiB 地址空间最小读写验证。
- 模型能够在 ZCU102 A53 PS 端运行文字和图片推理。
- 图片与 Prompt 可通过以太网提交，UART 可保留为系统 Shell 和调试通道。

### 2.2 当前图片推理基线

代表性图片输入经过 resize 后形成 4320 个 Encoder token，经 2×2 Patch Merger 后形成约 1080 个 LLM 视觉 token；文本和特殊 token 约 62 个。

| 阶段 | 实测时间 | 占总时间 |
|---|---:|---:|
| Vision Encoder | 625.812 s | 36.45% |
| 视觉 token 的 LLM Prefill | 1002.344 s | 58.39% |
| 文本 Prefill | 77.485 s | 4.51% |
| 生成 8 token | 10.749 s | 0.63% |
| 完整图片推理 | 1716.707 s | 100% |

Encoder 与视觉 Prefill 合计占 94.8%，是第一优先级。当前约为 0.00058 张/s；达到 3–5 张/s 需要约 5150–8580 倍端到端加速。

按照当前输入规模和 `1 MAC = 2 OPS` 的口径估算：

- Vision Encoder：约 4.69 TOP/图。
- LLM Prompt：约 8.68 TOP/图。
- 完整图片推理：约 13.47 TOP/图。
- 3–5 图/s：需要约 40.4–67.4 TOPS 的持续有效计算能力。

因此，单独进行 PL 算子映射不足以达到目标，必须同步降低运算量。

### 2.3 必须新增的模型与量化基线

官方论文主要报告 RoboBrain2.5-8B，而本项目部署的是 4B 第三方 GGUF Q8_0；不能直接将 8B BF16 结论移植到 4B Q8_0。所有算法优化之前必须固定以下三层基线：

| 基线 | 目的 |
|---|---|
| Qwen3-VL-4B BF16 | 判断原始通用底座在相同规模下的能力 |
| RoboBrain2.5-4B BF16 | 隔离具身后训练带来的收益或通用能力退化 |
| RoboBrain2.5-4B Q8_0 | 隔离 GGUF Q8_0 带来的精度、容量和速度变化 |

三者必须使用相同图片、Prompt、解码参数和真实标注。Pointing 使用 PCK/坐标误差，Grounding 使用 IoU/Acc@0.5，Trajectory 使用 ADE/FDE/成功率，General 仅作为通用能力回归测试。

当前 Q8_0 不是严格的纯 8 bit 权重格式。每 32 个 INT8 权重共享 1 个 FP16 scale：

```text
32 × INT8 + 1 × FP16 scale = 34 byte = 8.5 bit/weight
```

FP16 scale 用于恢复该权重块的实际幅值；精度收益来自细粒度分块缩放，不能笼统归因于“保留一个 FP16 权重”。Norm、Bias 和部分张量仍可能保持浮点格式，因此硬件精度口径必须按真实张量类型统计。

最终部署与论文系统结果固定使用`RoboBrain2.5-4B.Q8_0.gguf`和
`RoboBrain2.5-4B.mmproj-Q8_0.gguf`。BF16只允许用于结构可行性、教师信号和提前排除边界；任何Token保留率、切入点或精度门都必须在真实Q8_0 GGUF执行路径复验后才能冻结。

### 2.4 当前 PL 实测里程碑（2026-08-28）

统一 Q8×Q8 GEMM 已在 ZCU102 PL 200 MHz 上完成 C testbench、定向 RTL/XSim、
Vivado 时序和板端 UIO 数值验证。v3 采用 128-bit、16-beat 输出 burst；此后
PE16 将 N 向 PE 从 8 扩为 16，真实板端结果为：

| workload microbenchmark | v3 PE8 | PE16 | PE16 相对 v3 |
|---|---:|---:|---:|
| Language prefill shape | 368.570 ms | 242.808 ms | 1.518× |
| Vision encoder shape | 628.165 ms | 448.839 ms | 1.400× |
| Decoder GEMV shape | 18.511 ms | 14.647 ms | 1.264× |

PE16 启动前后自检及三类工作负载全部数值抽检 `max_abs=0`，APM 无溢出。
Prefill/Vision 仍为 compute/schedule-bound，Decoder 仍为 memory/weight-stream-bound。
这只是代表性矩阵 shape 的计算核结果，尚未完成 Transformer block 或完整模型的
PS+PL 接入。下一硬件子目标是保持 PE16，重叠 load/compute/store 并扩大 DMA
分块；decoder 继续按独立 GEMV 路线处理。完整证据见
`reports/2026-08-28-zcu102-q8q8-pe16-board.md`。

### 2.5 PE16 DATAFLOW离线候选（2026-08-30）

PE16上板基线未被覆盖。独立DATAFLOW候选将load/compute/store通过FIFO流水连接，并把累加器
清零融合到首个K block。HLS C测试9/9、定向控制RTL、200 MHz完整系统布局布线、bit/XSA、
16 GB动态DDR FSBL及RAM Linux离线上板包均通过。完整系统WNS=`+0.078 ns`、WHS=`+0.010 ns`，
资源为25,966 LUT、30,473 FF、419 BRAM tile、661 DSP。

同时完成任务无关PS描述符、GGML Q8_0双平面拆分和64 MiB DMA自动分块。规划器在代表shape上
给出视觉Prefill `chunk_m=976`、Vision Encoder `chunk_m=3584`、Decoder `chunk_m=1`，
普通与消毒器测试各5/5通过。以上性能仍是排程/容量结果，不是DATAFLOW板端实测；下一门必须是
板端双自检、固定512与auto chunk对照、三次性能/APM测量。证据见
`reports/2026-08-30-zcu102-q8q8-pe16-dataflow-preboard.md`。

为避免口径混淆，`language_prefill M=512`只保留为跨硬件同形状kernel基线；完整图片代表
视觉Prefill单列为`M=1080`并比较固定`512+512+56`与自动`976+104`。15阶段板测协议、逐阶段
`ap_idle`、UART启动识别/单次命令发送和结果汇总器均已完成离线正/负测试。APM平均延迟统一采用
“累计延迟÷累计事务数”；旧PE16报告中放大3倍的平均延迟已追加修正，原始性能和瓶颈结论不变。

随后完成授权板测。相对冻结PE16同shape基线，DATAFLOW在Language Prefill、Vision Encoder、
Decoder上分别为`1.391×/1.649×/0.998×`；这证明流水化对大M矩阵有效，对M=1 GEMV无收益。
同工作量下自动M分块相对固定512又使完整视觉Prefill和Vision Encoder获得`1.032×/1.080×`，
Decoder仍基本不变。所有数值抽检`max_abs=0`、APM无溢出、最终控制为idle；未写SD/QSPI。

### 2.6 GGML到PL桥接门（2026-08-30）

新建项目自有 `fpga_accel/ggml_q8q8_bridge`，不修改上游 llama.cpp。桥接层固定
`src0[K,N] / src1[K,M] / dst[N,M]` 与硬件 `X[M,K]×W[N,K]^T` 的对应关系，支持连续
Q8_0权重、Q8_0或动态量化的FP32激活、FP32输出；batch、broadcast、view、转置、重排和
越过硬件上限的shape均显式回退CPU。

原生与ASan/UBSan测试通过，mock端到端12输出`max_abs=0`；当前上游Q8_0布局确认是
34 byte（FP16 scale偏移0、INT8 payload偏移2），19行/152块量化结果与上游参考实现逐字节一致。
公共`ggml_tensor`适配与AArch64交叉编译通过。以上仍是软件契约门，不代表完整模型已调用PL；
下一门是默认关闭的生产入口、逐shape命中/回退统计和首个Transformer Block数值对照。

### 2.7 真实覆盖率与PS准备开销门（2026-08-30）

既有Grounding逐算子日志的只读审计表明，大矩阵合同覆盖率足够进入后端工作：decode、KV cache、
text prefill、vision encode、vision prefill的可命中调用分别为`252/253`、`249/253`、
`252/253`、`104/106`、`747/759`。这只是形状/类型覆盖；M=1 decoder虽多数合同可命中，
但DATAFLOW板测无收益，因此生产策略必须按盈利性回退CPU。

板端进一步量化了GGML→PL准备开销。标量FP32→Q8_0在M=512/1080/4320分别需要
`31.758/66.970/107.172 ms`。新增NEON和4核row并行后降至`2.701/5.474/8.644 ms`，
相对标量为`11.76×/12.23×/12.40×`；六种真实shape的完整输出与标量参考逐字节一致。
此时激活量化仅占对应PL kernel的`1.55%/1.48%/3.43%`，不再是首要瓶颈。

随后以NEON四核重写权重拆分，真实UIO中6.96/26.46/4.46 MB权重stage降到
`3.207/12.038/2.188 ms`，三shape全量逐字节一致。对大M，权重stage与激活量化合计只相当于
当前kernel的`4.30–8.44%`，因此原生Q8_0 PL前端不再是首个Block的前置门，保留为后续优化。

新的主要数据路径瓶颈是F32输出回读。19.92/37.98/58.72 MB从非缓存UIO回PS，即使NEON四核
仍需`48.957/83.899/150.351 ms`，有效带宽仅约0.39–0.45 GB/s。逐算子PS往返可以用于bring-up，
但被拒绝为最终架构。下一阶段必须：

1. 先实现LLM SwiGLU FFN和Vision GELU FFN的PL驻留子图；
2. 按整个子图的buffer live range而非单个GEMM容量选择chunk；
3. 在PL内完成Norm、SiLU/GELU、逐元素乘、残差和再量化，仅在子图边界回读；
4. 后续再比较原生Q8_0权重前端和更大/可缓存共享DMA内存。

详细规格见`docs/PL_RESIDENT_TRANSFORMER_BLOCK_ARCHITECTURE.md`。在FFN子图和首个真实
Transformer Block得到数值/端到端时间前，生产hook保持默认关闭，decoder保持CPU回退。

第一版live-range规划器已证明单GEMM chunk不能直接用于完整FFN：64 MiB下LLM SwiGLU从
`976`收紧为`432`（1080行3块），Vision GELU从`3584`收紧为`2896`（4320行2块），物理区间
重叠为0。该结果是容量上界，不是时间最优；后续结合实际向量核和GEMM时间扫描候选chunk。

现有profile显示SWIGLU/RMS_NORM在Vision Prefill分别仅占约0.2185%/0.1528%，GELU/NORM在
Vision Encoder约占0.6044%/0.1353%。所以向量核创新点不是“单算子本身很慢”，而是通过融合
阻止大F32中间张量在PS与非缓存UIO之间往返；评价必须看子图端到端延迟和DDR流量下降。

## 3. 模型逐阶段拆解

模型结构以官方 [RoboBrain2.5-4B](https://huggingface.co/BAAI/RoboBrain2.5-4B)、[config.json](https://huggingface.co/BAAI/RoboBrain2.5-4B/blob/main/config.json) 和 [RoboBrain2.5 论文](https://arxiv.org/abs/2601.14352)为依据。

### 3.0 ZCU102逐阶段参与矩阵

“参与”不等于每一层都跨板调用一次。默认只允许图像/早期Patch、选定Vision层、Vision Merger三类粗粒度边界；其余阶段通过前面产生的Token掩码和索引间接受益。

| 模型阶段 | 当前稀疏性 | ZCU102固定参与方式 | 可选扩展 | 禁止的默认切分 |
|---|---|---|---|---|
| 图像预处理、Patch Embedding | 局部卷积有规则连接，但不是一般SpMM | 流式分块、轻量显著性、空间group构造、原始坐标登记 | 低成本Patch投影或变化检测 | 为很小的Resize/Normalize收益单独往返 |
| Vision Attention core | 当前为稠密全Attention | 生成Token mask、块位图和紧凑Q/K/V索引 | 动态块稀疏QK/PV与在线Softmax | 24层逐层在主平台与PL间搬Q/K/V |
| Vision QKV/O投影 | 当前权重稠密 | 通过删除Token减少矩阵M维；记录被保留Head/Channel描述 | 经训练后的块/N:M稀疏计算 | 从高端NPU卸载普通稠密投影 |
| Vision FFN | 当前权重稠密，GELU不产生天然零值 | 通过删除Token减少FC1/FC2行数 | 高稀疏率块权重FFN；复用现有驻留FFN核 | 每个GEMM结果回传主平台 |
| DeepStack、Patch Merger、Projector | 合并/重排规则，Projector稠密 | 执行结构保持合并、全局锚点保护、Token压紧和位置映射同步 | 粗到细可训练Merger | 丢弃原始位置后再重建坐标 |
| 视觉LLM Prefill | FFN/投影稠密，因果Mask不等于稀疏权重 | 向主平台提供缩短后的视觉序列、MRoPE/DeepStack索引和预算描述符 | 生成块稀疏Attention布局 | 36层逐层跨板执行FFN或Attention |
| 文本Prefill | 短序列为主 | 仅提供小尺寸Prompt/任务摘要给视觉选择器；不搬文本主干张量 | 固定Prompt缓存描述 | 为短文本Prefill单独下沉PL |
| KV Cache | 默认稠密 | 视觉Token削减自然减少KV；输出可复现的Token映射供主平台同步删除 | 长上下文的页/Head保留位图 | 每帧把160.59 MiB FP16 KV搬到ZCU102 |
| 自回归Decode | M=1稠密GEMV和KV Attention | 默认不直接执行；通过更短视觉上下文和更小KV间接受益 | 仅在权重/KV常驻且跨越收益门时启用稀疏Decode | 把主平台Prefill与ZCU102 Decode作为默认架构 |

逐阶段研究的共同产物不是多套硬件，而是一套`Token/Block Sparse Descriptor`：包含原始长度、保留长度、group大小、原始Token索引、二维坐标、MRoPE索引、DeepStack层级、块位图、数据类型和校验字段。描述符不包含`General/Pointing/Grounding/Trajectory`字符串。

### 3.1 图像预处理与 Patch Embedding

```text
图像
  → resize / normalize
  → Conv3D，kernel=[2,16,16]
  → 4320 个视觉 token
```

主要运算：图像缩放、归一化、卷积/矩阵乘法。

候选加速：

- 将归一化、Patch分组、轻量评分和数据布局转换组成单遍流式前端。
- 记录每个Patch的原始二维坐标，在压缩后继续携带坐标而非重新编号。
- 比较“原图直接进入主平台”和“ZCU102生成紧凑Patch/浅层Token”两个单边界方案。
- Conv3D只有在与Token选择融合且总延迟下降时才复用现有矩阵引擎。

优先级：P2。必须优化，但不是当前最大耗时阶段。

### 3.2 Vision Encoder

结构：24 层、hidden size 1024、FFN size 4096、16 个 Attention Head，输入约 4320 token。

单层流程：

```text
LayerNorm
  → Q/K/V Projection
  → RoPE
  → QK^T
  → Softmax
  → Attention × V
  → Output Projection
  → Residual
  → LayerNorm
  → FC1
  → GELU
  → FC2
  → Residual
```

候选加速：

- 冻结一个早期选择点，首选已经验证计算图可行的layer-5；不在24层间反复跨板。
- PL生成规则2×2/4×4块的Token mask、全局锚点和压紧索引，主平台继续执行剩余稠密层。
- 动态块稀疏Attention只作为可选计算面：完整QK/PV/Softmax子图必须驻留PL且通过通信收益门。
- QKV、Output Projection、FC1、FC2默认留在主平台；只有经过训练的高稀疏块权重才进入PL候选。
- 连续视频中只更新变化显著的 token。

若保留 token 比例为 `r`：

- QKV、Projection、FFN 的运算量近似缩小到 `r`。
- QK 和 AV 的运算量近似缩小到 `r²`。

优先级：P0。

### 3.3 Patch Merger 与 DeepStack

当前 2×2 空间合并将 4320 个 Encoder token 压缩为约 1080 个 LLM 视觉 token，并包含多组：

```text
4×1024 → 4096 → GELU → 2560
```

候选加速：

- 将本阶段作为默认的跨设备单边界之一，执行Token合并、压紧和描述符封装。
- 使用粗粒度全局 token 与少量高分辨率局部 token。
- 研究可训练的空间保持型 Token Merger；冻结L2@87.5%作为正确性和安全基线。
- 保留原始二维位置、MRoPE坐标和DeepStack层级信息，并对掩码与张量做CRC/长度校验。

优先级：P1。

### 3.4 视觉 Prefill

视觉 Prefill 是 1080 个视觉 token 进入 36 层语言模型后的计算，不等同于 Vision Encoder。

语言模型参数：hidden size 2560、FFN size 9728、32 个 Query Head、8 个 KV Head。

单层流程：

```text
RMSNorm
  → Q/K/V Projection
  → Q/K RMSNorm
  → MRoPE
  → Causal Attention
  → Output Projection
  → Residual
  → RMSNorm
  → Gate Projection + Up Projection
  → SiLU(gate) × up
  → Down Projection
  → Residual
```

候选加速：

- 主平台承担QKV、Output和SwiGLU等规则稠密矩阵；PL的首要贡献是让这些矩阵处理更少的Token行。
- 将视觉Token从1080扫描至512、256、128，建立精度—主平台时间—通信—PL时间联合Pareto曲线。
- 使用Prompt摘要引导视觉Token选择，但Prompt摘要必须小于视觉张量且不引入逐层往返。
- 对权重引入N:M或块结构化稀疏仅作为第二阶段；先证明Token稀疏的端到端收益。
- 可选块稀疏Attention必须同时报告索引生成、通信、DDR流量和主平台未执行部分，不能只报告FLOPs下降。

优先级：P0。

### 3.5 文本 Prefill

- 多 token Prompt 主要表现为 GEMM，通常偏计算受限。
- 单 token 时逐渐退化为 GEMV，受权重带宽影响增大。
- 固定系统 Prompt 可缓存 KV，避免多轮对话重复计算。
- ZCU102只接收池化后的Prompt摘要或预算参数用于条件选择，不执行文本主干。

优先级：P2。当前仅占图片推理约 4.5%。

### 3.6 Decoder 与输出头

Decoder 在 batch=1 时以 GEMV 为主，每生成一个 token 都需要流式读取大部分语言模型权重，因此偏 memory-bound。

候选加速：

- 默认由主平台执行GEMV、KV Attention和LM head，避免把权重/KV搬到较低带宽的ZCU102。
- 视觉Token削减直接缩短Decode可见上下文和KV；该间接收益计入模型端到端结果。
- 稀疏Decode只保留为条件分支：要求压缩权重和KV在ZCU102侧跨Token常驻，并实测快于主平台。
- 在不改变模型的基线中，由 PS 执行坐标词表约束、停止条件和结果解析，避免无效输出 token。
- 轻量点/框/轨迹头属于后续需要训练的新模型方案，不作为首版 PL 加速器的必需组成，也不预设其一定优于原始自回归输出。
- 原始自回归 Decoder 承担普通语言交互，并作为所有结构化输出的精度基线。

优先级：视觉上下文缩短带来的Decode间接收益为P1；ZCU102直接Decode为条件探索项，不作为近期交付门。

### 3.7 任务路由与硬件边界

当前 `task` 参数不改变 Vision Encoder、LLM Prefill 或 Decoder 的主体计算图。因此，本研究不为四种 API 模式分别设计硬件，而采用：

```text
主平台：任务/Prompt → 小尺寸Prompt摘要、预算上限、阶段边界
                              ↓ 稀疏控制描述符
PL：统一执行Token评分、结构约束、选择、合并、压紧与索引生成
                              ↓ 紧凑Token/块描述符
主平台：执行规则稠密主干；仅在收益门通过时调用PL稀疏计算面
```

工作负载描述符包含序列长度、Token预算、group形状、原始位置、MRoPE/DeepStack映射、token/block mask地址、可选`M/N/K`、量化格式和缓冲区地址。PL不接收`General/Pointing/Grounding/Trajectory`名称。这样任务策略可以迭代，而数据面、DSP和BRAM不随任务数量增长。

## 4. 稀疏性判断与研究路线

### 4.1 当前模型不存在可直接利用的天然稀疏 GEMM

- QKV、Output、FFN 权重均为稠密矩阵。
- Vision Attention 为完整非因果 Attention。
- LLM 因果 Mask 的三角结构不等于稀疏权重矩阵。
- SiLU 和 GELU 不会自然产生大量严格为零的激活。

### 4.2 应主动引入的稀疏性

1. N:M 结构化权重稀疏，例如 2:4、4:8。
2. 规则块稀疏，便于突发读取和 PE 负载均衡。
3. 视觉 token 稀疏，减少矩阵行数和 Attention 序列长度。
4. 局部窗口或块稀疏 Attention。
5. 连续帧变化 token 稀疏。

优先选择结构化稀疏，不优先采用无结构稀疏。后者容易引入索引开销、随机访存和 PE 负载不均衡，未必能得到真实加速。

## 5. 非线性算子处理

“将非线性变为线性”不是无损转换，计划分为两级。

### 5.1 保持模型结构的硬件近似

| 原始算子 | FPGA 实现候选 |
|---|---|
| GELU / SiLU | 分段线性、二/三阶多项式、LUT |
| Softmax exp | base-2 exp、LUT、定点范围缩放 |
| Softmax 归一化 | 在线最大值/求和、倒数 LUT+Newton |
| RMSNorm / LayerNorm | 流水归约、rsqrt LUT+Newton |
| RoPE sin/cos | 预计算表或 CORDIC |
| Top-k | 比较树或 PS 端保留 |

### 5.2 改变模型算子的近似

- 以硬件友好的激活函数替换 GELU/SiLU。
- 以 Softermax/ConSmax 类结构替换 Softmax。
- 使用原始 4B 模型进行层级蒸馏与任务蒸馏。

只有在逐算子 Profiling 证明非线性单元成为显著瓶颈后，才进入第二级。空间坐标任务必须重新验证，不能直接沿用分类或 NLP 精度结论。

## 6. Compute-bound / Memory-bound 实测计划

### 6.1 采集项

每个算子记录：

- 算子类型与 `M/N/K` 或 Attention Shape。
- 数据精度和布局。
- 理论 MAC/OPS。
- PS/PL 执行周期。
- DDR 实际读写字节数。
- DMA 等待、AXI stall、PE idle 和 PE active 周期。
- 各阶段功耗和温度。

### 6.2 Roofline 判定

```text
Arithmetic Intensity = 实际 OPS / 实际 DDR 传输字节
Ridge Point          = PL 实测峰值 OPS / PL 实测 DDR 带宽
```

- `Arithmetic Intensity < Ridge Point`：memory-bound。
- `Arithmetic Intensity > Ridge Point`：compute-bound。
- 最终结论同时参考 AXI stall 与 PE 利用率。

### 6.3 必须完成的微基准

- PS DDR 顺序读、写、读写带宽。
- PL 通过单 HP 端口和多 HP 端口访问 DDR 的带宽。
- 不同 burst、tile 和并发 DMA 配置。
- GEMM：Encoder/Prefill 代表 Shape。
- GEMV：Decoder 代表 Shape。
- QK、AV、Softmax、Norm、GELU/SiLU。
- 稠密、2:4、4:8 和块稀疏对比。
- Token评分、约束选择、gather/merge/compact及位置索引同步。
- 三个候选切分点的输入/输出字节、序列化、DMA和可配置链路回放。

### 6.4 当前待验证假设

| 阶段 | 初步分类 | 后续证据 |
|---|---|---|
| Vision Encoder | compute-bound | GEMM/Attention PE 利用率、DDR 流量 |
| 视觉 Prefill | compute-bound | Prompt 长度扫描、GEMM 利用率 |
| 长文本 Prefill | compute-bound | token 数与吞吐变化 |
| 单 token Decoder | memory-bound | 权重流量、带宽利用率 |
| Softmax/Norm | latency/control-bound | 算子周期、流水停顿 |
| PS↔PL 搬运 | 未确定 | HP/HPC AXI 实测 |
| Token Selector/Compactor | 未确定 | 每Token周期、读写字节、Top-k/配额控制停顿 |
| 主平台↔ZCU102 | 平台与边界相关 | 单次阶段交接P50/P95、有效带宽、主平台空等时间 |

## 7. 平台无关的稀疏协处理架构

### 7.1 逻辑架构

```text
图片/早期Patch或选定层视觉特征       Prompt池化摘要、预算与精度约束
              │                                  │
              └──────── 主平台适配层 ────────────┘
                               │
                    单次、粗粒度张量交接
                               ▼
                     ZCU102 Sparse Front-end
           ┌──────────────────────────────────────┐
           │ Score：L2基线 / 可编程轻量评分       │
           │ Guard：空间分区、全局锚点、预算约束  │
           │ Compact：选择、合并、重排、压紧       │
           │ Map：原始索引、2D坐标、MRoPE、层级    │
           │ Optional：块稀疏Attention/GEMM/GEMV  │
           └──────────────────────────────────────┘
                               │
                    紧凑Token + 稀疏描述符
                               ▼
              主平台继续Vision/LLM稠密主干与Decode
```

### 7.2 稳定合同与可替换部分

| 层次 | 与主平台选择的关系 | 本项目冻结内容 |
|---|---|---|
| 算法语义 | 无关 | 规则group、Token预算、空间锚点、原位置保持、可选Prompt条件评分 |
| 稀疏描述符 | 无关 | 版本、shape、keep count、原索引、坐标/MRoPE/DeepStack映射、块位图、dtype、CRC |
| PL数据面 | 无关 | 流式评分、约束选择、合并、压紧、索引同步和性能计数器 |
| 物理接口 | 有关 | 先以文件/内存参考实现和AXI4-Stream验证；最终替换为10/40GbE、USB或可用高速互连 |
| 主平台后端 | 有关 | CUDA/TensorRT、CANN或其他运行时分别实现同一合同的适配器 |
| 可选稀疏算子 | 有关 | 只有主平台对该稀疏模式效率低且通信可隐藏时启用 |

### 7.3 三个候选切分点

| 切分点 | 能减少的计算 | 代价与风险 | 当前顺序 |
|---|---|---|---|
| Patch/浅层前 | 后续Vision Encoder、Merger和视觉Prefill | 选择信息最弱，需要训练/蒸馏；模型改动最大 | P1，长期收益最高 |
| Vision layer-5 | 后18层Vision Encoder、Merger和视觉Prefill | 需要一次中间特征交接；已有计算图可行性证据 | **P0，首个研究边界** |
| Merger后、LLM前 | 36层视觉Prefill与后续KV | 不减少Vision Encoder；接口最简单，适合先做系统合同 | **P0，首个系统边界** |

不会同时实现三个边界的硬件。先用同一软件模型和描述符比较净收益，再冻结一个主边界、一个消融边界。

### 7.4 端到端收益门

对任意主平台和切分点，只有满足下式才接受异构方案：

```text
T_send + T_PL_score/compact + T_return + T_main_reduced
    < T_main_full
```

并同时满足：

- 使用P50和P95实测，而不是链路线速或峰值TOPS；
- 总系统能量包含主平台、ZCU102和互连，主平台没有因空等而增加的能量也必须计入；
- 每帧最多一次主数据往返，不允许24/36层逐层往返；
- 不迁移完整KV Cache；只有索引、位图和增量小状态可以跨板；
- 若主平台自身运行相同Selector更快且更省电，保留软件算法结果，但淘汰ZCU102异构claim；
- 新方法必须通过未参与设计的测试集，已揭示的blind/validation清单只用于复现冻结结果。

现有稠密GEMM、DATAFLOW和PL驻留FFN继续用于测量ZCU102计算/带宽上界和可选稀疏算子，不再要求它们串成完整4B主干后才进入Token协处理研究。

ZCU102片上BRAM约4 MB，且没有UltraRAM，无法保存数GB权重，因此不能直接照搬依赖HBM或大容量片上SRAM的FPGA LLM方案。

## 8. Token 与运算量削减方案

本节先在平台无关软件参考实现上冻结算法和张量合同，再映射PL。不得因为某一主平台方便实现而改变Token语义，也不得先做硬件后使用已揭示测试集反向选参。

### 8.1 无训练/少训练基线

- 动态图片分辨率。
- 文本引导视觉 token 选择。
- Token 合并而不是直接丢弃。
- 保留原二维位置 ID 和 MRoPE 坐标。
- 固定 Prompt KV Cache。
- 相同图片的视觉特征缓存。
- 冻结L2@87.5%作为硬件合同、全长兼容和故障回退基线；它不是最终创新方法。

### 8.2 需要微调/蒸馏的主线

- 首轮只比较Patch/浅层前、Vision layer-5和Merger后三个冻结边界；边界选定后才在其邻近层做局部搜索。
- 全局空间锚点与局部高分辨率 token 并存。
- 将 Patch Merger 改为可学习的粗到细压缩器。
- 原始 RoboBrain2.5-4B 作为 Teacher，蒸馏隐藏状态、Attention 与任务输出。
- 优先研究空间定位与轨迹共享的结构保持型 token 选择；新增点、框和轨迹任务头仅作为长期探索。
- 主平台根据实测精度约束和图像复杂度选择token预算，只向PL发送任务无关描述符和小尺寸Prompt摘要。
- 使用全新开发/测试划分；当前已经揭示的开发、blind、validation和task-routing清单全部禁止用于新方法选参。

### 8.3 参数扫描

```text
LLM 视觉 token：1080 / 512 / 256 / 128
Encoder 保留率：100% / 75% / 50% / 25%
权重密度：100% / 75% / 50%
输入最大边：原始 / 512 / 384 / 256
```

所有扫描均生成任务精度—延迟—功耗 Pareto 曲线。

除Token数外，还必须报告主平台被删除的真实算子行数、Attention有效块数、KV字节数、PL处理字节数和跨设备传输字节数，防止把“Token减少”等同于“系统一定加速”。

## 9. 连续具身任务的 3–5 Hz 路线

本计划的第一评价对象是每张独立新图的模型端到端推理；连续帧复用作为第二阶段扩展，不能用应用侧控制频率替代模型推理频率。连续场景可进一步采用双速率推理：

```text
关键帧或场景明显变化
  → 完整/压缩后的 Encoder + LLM 语义更新

相邻普通帧
  → 变化检测
  → 只更新变化 token
  → 复用视觉特征和上下文
  → 原始约束解码，或经精度验证后的可选轻量输出头
```

这样将目标拆分为：

- 独立新图完整模型频率：单独报告，不与缓存帧混合平均。
- 完整语义刷新频率：连续场景中单独报告。
- Token增量更新频率：3–5 Hz或更高，明确其中有多少帧调用完整模型。
- 普通语言生成：保留原 Decoder，允许较低速度。

若要求完整4B模型对每一张独立新图达到3–5 Hz，则需要足够强的主AI平台、显著的视觉工作量削减或更小学生模型，不能仅靠ZCU102 PL优化完成。

## 10. 相关文献与可迁移内容

| 运算/研究点 | 文献 | 可借鉴内容 | ZCU102 注意事项 |
|---|---|---|---|
| 动态视觉 token | [HeatViT](https://arxiv.org/abs/2211.08110) | Token Selector、非线性多项式近似；ZCU102 上报告 3.46–4.89× | 最接近本平台，但原模型规模较小 |
| 固定长度Token剪枝 | [Fitop-Trans](https://ieeexplore.ieee.org/document/10705515/)（FPL 2024） | Selector与后续阵列同板，压紧Token后由分区脉动阵列继续消费 | 说明Selector单独跨板不是其收益来源 |
| 完整GPT推理 | [DFX](https://arxiv.org/abs/2209.10797)（MICRO 2022） | 四块U280上端到端执行GPT-2，利用HBM和模型驻留 | 多卡HBM结果不能外推到ZCU102 DDR4 |
| 低功耗Decoder驻留 | [Hummingbird](https://arxiv.org/abs/2507.03308)（ICCAD 2025） | Decoder和低比特权重连续驻留嵌入式FPGA | 可作单板能效对照，不代表4B VLM独立新图性能 |
| 近存储KV Attention | [HILOS](https://arxiv.org/abs/2502.09921)（ASPLOS 2026） | 在SmartSSD FPGA上就近处理KV/Attention | 依赖存储侧并行和KV驻留，角色不同于外置小Selector |
| 空间保持 token | [Nüwa](https://arxiv.org/abs/2602.02951) | 全局空间锚点、两阶段 token 压缩 | 需适配 MRoPE 和 DeepStack |
| Grounding 位置保持 | [GAP](https://arxiv.org/abs/2506.21873) | 剪枝后保持原 position ID | 必须在空间定位与轨迹指标上复测 |
| Training-free token 削减 | [SparseVLM](https://arxiv.org/abs/2410.04417) | 文本引导选择、token recycling | 主要减少 LLM Prefill，不能减少全部 Encoder 计算 |
| Projector 压缩 | [TokenPacker](https://arxiv.org/abs/2407.02392) | 粗到细 Projector、局部细节注入 | 需要训练或蒸馏 |
| 连续帧复用 | [Eventful Transformers](https://arxiv.org/abs/2308.13494) | 仅重算变化 token，论文报告约 2–4× 计算节省 | 需要增加跨帧状态管理 |
| 稀疏 DSP 数据流 | [FlightLLM](https://arxiv.org/abs/2401.03868) | 可配置 Sparse DSP Chain、混合精度 | 原平台为 U280/VHK158，不能照搬 HBM 性能 |
| 动态稀疏Prefill | [FAST-Prefill](https://arxiv.org/abs/2602.20515) | 输入/Head相关稀疏索引、融合索引生成与双层缓存 | 原实验为U280及4K–128K长上下文；本项目必须重新验证1142-token图文输入 |
| CPU-FPGA 异构 | [EdgeLLM](https://arxiv.org/abs/2407.21325) | GEMM/FFN 混合数据流、结构化稀疏、异步搬运 | 原平台 VCU128，需重新设计 DDR 调度 |
| GPU-FPGA Attention协处理 | [HPU](https://arxiv.org/abs/2504.16112) | GPU保留密集线性层，FPGA驻留KV并处理带宽受限Attention | 依赖PCIe 4.0×8、大块合并和较大Batch；不能套用到独立板1GbE逐层切分 |
| VLM阶段级异构 | [ReCoVLM](https://doi.org/10.1109/FCCM68464.2026.00026) | 视觉Token压缩后再划分Prefill/Decode，说明先减状态再异构的重要性 | 公开摘要未说明物理链路；不得假设其接口或直接套用18.1 token/s |
| Attention/量化协同 | [AccLLM](https://arxiv.org/abs/2505.03745) | 剪枝、稀疏 Attention、可重构引擎 | 原平台 U280，仅借鉴架构思想 |
| N:M 稀疏 | [FNM-Trans](https://ieeexplore.ieee.org/document/11294873/)（DAC 2024） | 用统一N:M模式覆盖QKV、Attention和FFN | 需要在本模型重新训练并实测索引、负载均衡和精度 |
| Softmax | [Softermax](https://arxiv.org/abs/2103.09301) | base replacement、低精度、在线归一化 | 需验证空间输出误差 |
| Softmax 替换 | [ConSmax](https://arxiv.org/abs/2402.10930) | 减少 max 搜索与分母归约 | 属于模型变化，需要微调 |
| Softmax/Norm | [Hardware-Efficient Softmax and LayerNorm, 2026](https://arxiv.org/abs/2604.23647) | 保持归一化约束的硬件近似 | ASIC 结果不能直接视为 FPGA 性能 |
| 整数非线性 | [I-BERT](https://arxiv.org/abs/2101.01321) | GELU、Softmax、LayerNorm 整数近似 | NLP 精度结论不能替代具身任务评测 |

## 11. 预期论文主线

暂定研究主题：

> 面向异构具身VLM的空间保持型自适应Token稀疏化与平台无关FPGA协处理器

预期贡献：

1. 在Patch、Vision layer-5和Merger后三个候选边界统一描述视觉Token削减，自动选择一个粗粒度边界，并维护二维坐标、MRoPE与DeepStack映射。
2. 定义不依赖CUDA/CANN等后端的Token/Block Sparse Descriptor，ZCU102用同一数据面完成评分、空间约束、合并、压紧和索引同步。
3. 建立包含主平台计算、跨板通信和PL处理的收益模型；只有动态/不规则稀疏子图通过交叉点后才启用可选PL计算面。
4. 用Qwen3-VL-4B BF16、RoboBrain2.5-4B BF16、RoboBrain2.5-4B Q8_0和各级算法/硬件优化构成完整证据链。
5. 在真实ZCU102及至少一个主AI平台上报告模型端到端延迟、任务精度、总系统功耗、能效、通信和资源使用。

投稿匹配：

- FPL、ASAP：Token数据面、稀疏执行和算法—硬件协同，优先目标。
- DAC、ICCAD：需要进一步形成切分点、Token预算、通信与PL参数的自动联合搜索或编译方法。
- MICRO：需要证明同一描述符和微体系结构能跨至少两类主平台/模型工作，而非单板工程。
- 若“ASSP”实际指 ASP-DAC 2027，其截稿已过；若指 ASAP 2027，则仍可作为主目标。

## 12. DAC 2027 加速版路线

### 12.1 定位与时间边界

DAC 2027 已公布会期为 2027 年 7 月，但截至本计划版本日期，Research Manuscript 的正式截稿时间尚未公布。参考 DAC 2025/2026 均在前一年 11 月中旬截稿，本项目设置：

- 内部完整实验冻结：2026-10-31。
- 内部论文初稿：2026-11-05。
- 内部投稿就绪：2026-11-10。
- 正式日期：以 DAC 2027 CFP 发布后更新，不把历史日期当作官方日期。

DAC 加速版与完整路线共用前期基线、Profiling、Roofline 和稠密 PL 引擎；在算法协同阶段主动收缩任务范围，换取完整、可信、具有设计方法贡献的证据链。

### 12.2 收缩后的问题定义

主问题限定为：

> 在保持点和边界框空间精度的条件下，如何联合选择视觉Token压缩边界、预算、通信格式和FPGA参数，并由平台无关的ZCU102稀疏协处理器减少异构RoboBrain2.5推理的实际工作量？

任务范围：

| 层级 | 范围 | 用途 |
|---|---|---|
| 主评测 | Pointing + Grounding | 共同代表空间定位，分别报告 PCK 与 IoU |
| 回归检查 | 少量 General 样本 | 证明接口和普通问答没有被破坏，不作为主要优化对象 |
| 可选迁移 | 小规模 Trajectory | 只用于方法迁移性验证；不阻塞投稿 |
| 暂缓 | 连续视频、密集时间价值、完整轨迹头、多机器人规划 | 留给完整路线或后续会议 |

减少任务数量不等于降低论文标准。DAC 版本通过收窄 claim，将工作量集中到一个算法—编译/搜索—体系结构闭环。

### 12.3 DAC 版核心贡献组合

1. **精度约束的空间Token压缩。** 面向点和框输出保留空间锚点、原position ID/MRoPE、DeepStack对应关系与局部高分辨率区域。
2. **平台无关稀疏合同。** 用统一描述符表达Token、位置和块稀疏关系，使CUDA、CANN或其他主平台后端不改变PL数据面。
3. **切分点—预算—通信—硬件联合搜索。** 根据精度约束、主平台实测曲线、链路带宽和ZCU102资源选择一个粗粒度边界，而不是手工固定“某层一定下沉”。
4. **真实异构系统的PPA/能效闭环。** 报告主平台全量、主平台单独执行同一Selector、ZCU102协处理三条基线，并计入双板和互连的总延迟、总能量及FPGA资源。

其中至少需要一项成为明确的方法创新：优先选择“空间精度约束的Token压缩 + 异构切分自动映射”。只有一个手工RTL GEMM核、单纯移植Q8_0或把Selector从GPU搬到FPGA，不足以支撑DAC Research Manuscript。

### 12.4 DAC 版明确不做的内容

- 不同时实现 Point、Box、Trajectory 三套专用输出头。
- 不同时铺开 Dense、2:4、4:8、任意块稀疏的全部组合；只保留 Roofline 证明最有效的一种。
- 不在截稿前同时替换 Softmax、SiLU、GELU、Norm；仅优化实测瓶颈。
- 不把完整 4B 独立新图达到 3–5 Hz 作为 DAC 投稿硬门槛。
- 不以官方 8B 结果替代本项目 4B BF16/Q8_0 实测。
- 不宣称覆盖全部具身任务；论文结论限定为空间定位工作负载，并用可选 Trajectory 检查迁移性。

### 12.5 DAC 版必须完成的实验矩阵

模型精度基线：

```text
Qwen3-VL-4B BF16
→ RoboBrain2.5-4B BF16
→ RoboBrain2.5-4B Q8_0
→ Q8_0 + 通用Token压缩基线
→ Q8_0 + 本文空间保持压缩
```

硬件性能基线：

```text
ZCU102 PS-only
→ 主AI平台完整模型
→ 主AI平台 + 主平台软件Selector
→ 主AI平台 + ZCU102固定L2 Selector/Compactor
→ 主AI平台 + ZCU102本文稀疏方法
→ 可选PL稀疏算子（仅在收益门通过时）
```

必须报告：

- Pointing：PCK、像素/归一化坐标误差。
- Grounding：IoU、Acc@0.5、中心点误差。
- Vision Encoder、视觉Prefill、文本Prefill、Decoder、通信、Selector/Compactor和端到端P50/P95延迟。
- 主平台、ZCU102和整套系统平均/峰值功耗、J/图、有效GOPS、FPS/W。
- Fmax、DSP/LUT/FF/BRAM、DDR 带宽、AXI stall、PE 利用率。
- Token 数、剪枝位置、矩阵形状、理论运算下降与真实加速的差距。
- 消融：切分点、空间锚点、position/MRoPE/DeepStack保持、预算搜索、通信格式和硬件映射各自贡献。

### 12.6 DAC 加速时间表与止损门

| 时间 | 任务 | 验收/止损条件 |
|---|---|---|
| 已完成至09-04 | Q8_0部署、逐阶段Profiling、稠密Q8 PL能力基线和既有Token路线止损 | 保留实测与负结果，不再等待完整稠密Transformer接入 |
| 09-04 至 09-15 | 稀疏描述符、三个边界合同、参数化通信/收益模型 | 同一合同覆盖三个边界并冻结主边界/消融边界 |
| 09-08 至 09-30 | 全长/L2基线软件路径与PL Selector/Merger/Compactor testbench、HLS和路由 | 全长逐字一致、位置映射一致、资源/时序通过；上板前仿真通过 |
| 09-10 至 10-15 | 全新数据上的空间保持Token方法、预算与切分联合搜索 | 相对L2或通用压缩形成更优PCK/IoU—算量Pareto；否则只保留硬件基线 |
| 10-01 至 10-31 | 至少一个主平台适配、真实通信、端到端功耗、资源、消融与对比 | 超过主平台软件Selector；若无端到端净收益，降为后续会议或修改claim |
| 11-01 至官方截稿 | 论文、图表、复现材料和内部评审 | 所有数字可由日志与脚本追溯，禁止补估算值 |

### 12.7 “达到 DAC 水平”的内部判据

以下条件同时满足才按 DAC Research Manuscript 推进：

1. 不是单纯部署：至少有一个新的算法—硬件协同方法或自动映射方法。
2. 不是理论加速：真实 ZCU102 有端到端延迟、功耗和资源结果。
3. 不是自比较：包含 Qwen/RoboBrain/Q8 三层模型基线和至少一种已有 token 压缩/映射基线。
4. 不是只看文本一致性：Pointing/Grounding 使用真实标注和公认几何指标。
5. 不是堆模块：通过消融证明每个新增机制的必要性。
6. 结论可推广：PL 只接收工作负载描述符，方法不依赖四个 RoboBrain API 字符串。

若无法在 2026-10-31 前满足这些条件，应保留完整研究路线，转投时间更合适的 FPL/ASAP/ICCAD，而不是用不完整数据强投 DAC。

## 13. 完整路线工作包、时间表与验收门

### WP0：基线冻结与真实任务集（2026-08 至 2026-09）

- [ ] 固定模型 hash、推理软件版本、图片尺寸和 Prompt 模板。
- [ ] 固定 Qwen3-VL-4B BF16、RoboBrain2.5-4B BF16 和 RoboBrain2.5-4B Q8_0 三层模型基线。
- [ ] 为 General、空间定位和轨迹工作负载准备带真实标注的测试集；Affordance 归入 Pointing 输出协议。
- [ ] 固定 PS-only、A6000 BF16/Q8_0 参考结果。
- [ ] 保留逐阶段计时、温度、内存和网页日志。

验收条件：相同输入可重复得到阶段延迟和任务指标，误差来源可追踪。

### WP1：逐算子 Profiling 与 Roofline（2026-08 至 2026-09）

- [ ] 记录完整算子 Shape、OPS 和字节流量。
- [ ] 测量 PS DDR 与 PL AXI 带宽。
- [ ] 完成 GEMM、GEMV、Attention、Softmax、Norm 微基准。
- [ ] 对每个阶段给出 compute/memory/control-bound 结论。

验收条件：形成实测 Roofline，能够解释每个主要阶段的性能上限。

输出文件：

```text
operator_inventory.csv
measured_roofline.json
microbenchmark_report.md
```

### WP2：ZCU102计算与数据路径能力基线（已完成主体，冻结）

- [x] 统一Q8×Q8 Dense GEMM/GEMV计算核、PE16与load/compute/store DATAFLOW候选；
  当前仅为算子microbenchmark，完整Transformer Block接入仍未完成。
- [x] 项目自有GGML Q8_0桥接契约、FP32激活动态量化、公共`ggml_tensor`适配和安全回退；
  与上游参考量化152块逐字节一致，尚未安装完整模型生产hook。
- [x] 真实算子覆盖率审计与A53准备开销门；NEON四核量化六shape逐字节一致，M512/1080/4320
  相对标量`11.76×/12.23×/12.40×`，权重每次重排路线明确拒绝。
- [x] DATAFLOW完成15阶段板测：相对PE16同shape的Language/Vision/Decoder为
  `1.391×/1.649×/0.998×`，确认大矩阵收益与Decoder无收益边界。
- [ ] Patch/Conv数据通路仅在与Token前端融合时实现，不再单独追求稠密主干覆盖率。
- [ ] QK/AV与Online Softmax降为可选稀疏计算面，不是主线前置门。
- [x] Norm/Activation统一向量IP和双IP完整系统完成OOC/布局布线；真实运行时激活门尚未完成。
- [ ] AXI DMA、多端口和双缓冲。
  - [x] HPC0/HPC1/HP0多端口、64 MiB保留区和任务无关自动M分块。
  - [ ] 原生交织Q8_0权重流式前端或受控预打包驻留、真正DMA队列与更深层双缓冲。

每个内核必须依次通过：

```text
C/C++ Golden
→ 单元测试
→ HLS C Simulation
→ RTL Testbench
→ C/RTL Co-simulation
→ 综合/时序检查
→ 上板
```

验收条件：现有结果作为ZCU102 Roofline、数值和资源基线冻结；不再要求先把完整Transformer Block接入PL。只有可选稀疏算子进入最终方案时，才恢复其真实激活和完整子图验收。

### WP3：平台无关稀疏合同与边界选择（2026-09）

- [x] 建立与此前416张已用图像零重叠的16图/32请求开发集，并锁定manifest与排除清单哈希。
- [x] 扩展真实模型路径以支持Patch Embedding、视觉块1/3、首个DeepStack层5和Merger后五个切入点；完成小模型全切入点DeepStack对齐测试。
- [x] 建立真实预处理网格上的MAC与通信静态审计，分别计算单向驻留和外置往返在1/2.5/10/40GbE下的串行化下界。
- [x] 在A6000完成BF16五切入点×四保留率×每任务4条、共160次几何Oracle前向；Patch Embedding失败，视觉块1@75%、首个DeepStack@75%和Merger后@75%为小样本候选。
- [x] Q8-A：真实Q8_0主模型+mmproj路径的Merger后Token列压紧、DeepStack同步与MRoPE位置同步已通过；100%旁路Token ID逐项一致。75%几何Oracle开发门因Pointing PCK@0.1下降0.25而失败，不作为候选结论。
- [x] Q8-B：已在Qwen3-VL GGML视觉图的第1块后加入固定预算group gather，使后续23块和LLM视觉Prefill真实减算；100%身份门通过。
- [x] Q8_0路径已复验Q8-A@75%、Q8-B@75%与Q8-B@87.5%；三者均未通过Pointing质量门，因此不冻结描述符、边界或Selector RTL。
- [x] 定义并版本化`Token/Block Sparse Descriptor v1`，完成C参考编码/零拷贝解析、双CRC及10类正常/损坏输入测试；仅冻结通用格式，未固化失败的Oracle策略，也未进入HLS/RTL。
- [ ] 为Patch前、Vision layer-5、Merger后三个边界冻结输入/输出张量、字节数、可删除算子和位置映射。
- [x] 建立平台无关符号收益模型：使用既有BF16阶段中位数扫描5个切入点、4档链路和4档前缀加速，共80组；输出单向驻留/外挂往返的延迟、假设能耗及链路/计算交叉点。目标平台实测参数待选型后回填。
- [ ] 在软件中实现全长、L2@87.5%、固定预算和可编程mask四条路径，100%路径必须逐字一致。
- [ ] 用既有冻结结果复现L2@87.5%，但新算法只使用全新开发数据，并建立新的未观察测试集。

验收条件：BF16只负责候选筛选；Q8_0 GGUF路径形成精度—算量—传输量Pareto。早期边界必须同时通过Q8精度门和单向驻留收益门，才能冻结一个主边界和一个消融边界。不能只用模型原始输出作为唯一精度真值。

#### 2026-08-25 提前完成的可行性烟雾实验

- [x] RoboBrain2.5-4B BF16 在 A6000 上完成 uniform/oracle、5 个预算点、640 次真实图片推理。
- [x] 记录实际视觉 Token、主运算 MAC proxy、Vision Encoder/LLM prefill/decode 延迟及 Pointing/Grounding 指标。
- [x] 约 1/4 Token 的匹配预算证明空间选择显著优于统一缩放，允许进入无真值选择器研究。
- [ ] 论文级数据集、置信区间和无真值方法尚未完成；当前精度门未通过，不得写成算法有效性结论。

阶段报告：[`2026-08-25-token-reduction-novelty-gap.md`](../reports/2026-08-25-token-reduction-novelty-gap.md)。
最新文献表明文本引导、自适应比例和空间保持剪枝均已有大量工作；本项目后续创新边界收紧为
“坐标精度约束 + MRoPE/DeepStack 对齐 + FPGA 规则块压缩 + 精度—硬件自动映射”。

#### 2026-08-25 无真值 Scout–Refine 止损结果

- [x] 实现低分辨率预测点/框驱动的规则裁剪，不使用真实标注选择区域。
- [x] 完成 4 组配置、256 次 A6000 推理；Token、算力代理和延迟均计入两遍总成本。
- [x] 任务协议 hybrid 使用 34.9% Token，PCK@0.1=0.5000、Mean IoU=0.6790。
- [x] 判定不接 PL：相对 uniform@0.50，PCK@0.1 增加 0.0312，但 Mean IoU 与 Acc@0.5 均下降 0.0625，且顺序延迟约翻倍。

该负结果排除了“完整低分辨率推理后再裁剪”的硬件路线。下一候选必须改为单遍浅层特征选择，
保留原始 position ID/MRoPE，用规则 2×2/4×4 块和分区全局锚点降低后续层工作量。
报告：[`2026-08-25-scout-refine-no-gt.md`](../reports/2026-08-25-scout-refine-no-gt.md)。

#### 2026-08-25 单遍浅层结构化剪枝结果

- [x] 在原始 BF16 模型中实现真实的 layer-5 选择：第 0–5 层全量执行，第 6–23 层与 LLM prefill 处理保留 token。
- [x] 选择单位严格对齐 Qwen3-VL 的 2×2 merger group，保留原始视觉 RoPE、LLM MRoPE position ID 和 DeepStack 对齐关系。
- [x] 完成 7 个保留率、448 次 A6000 推理；100% 路径与既有 stock 路径 64/64 输出完全一致。
- [x] 87.5% 保留率为临时 smoke 候选：Encoder/LLM prefill MAC proxy 分别为 89.7%/87.3%，PCK@0.1、PCK@0.2 各下降 1/32，Mean IoU 下降 0.0142，Acc@0.5 不变。
- [x] 按既有 PS 阶段实测做一阶工作量投影约 1502.6 s/图、1.13×；该数值不是剪枝版板端实测。
- [x] 门控判定暂不进入 RTL/HLS 集成：当前样本量不能证明精度等价，且固定 L2+分区锚点是可审计基线，不是足够的论文创新。

该结果证明“单遍、保留全局坐标系、浅层后结构化减算”在模型计算图中可行，也给出了精度开始快速下降的边界；
下一阶段应扩大数据，并设计面向点/框坐标误差约束且可流式映射到 FPGA 的统一评分/预算方法。
报告：[`2026-08-25-shallow-block-pruning.md`](../reports/2026-08-25-shallow-block-pruning.md)。

#### 2026-08-25 扩大集与坐标敏感选择器结果

- [x] 从 COCO val2017 新建 96 图、192 查询扩大集，显式排除原 32 图 smoke 集，重叠为 0。
- [x] 实现统一的 `spatial residual + regional quota`：四邻域特征残差与 RMS 能量融合，按 3×3 区域分配最小配额和剩余预算。
- [x] 方法只使用视觉运行时特征，Pointing/Grounding 共用同一评分、预算和数据通路；选择单位保持为 2×2 merger group。
- [x] 完成 full、L2 87.5%/75%、residual 87.5%/75% 共 960 次真实推理，并做 n=96/task 的成对 bootstrap。
- [x] 75% 预算相对 L2：PCK@0.1 `+0.0729`、PCK@0.2 `+0.0312`、Mean IoU `+0.0072`，但 Acc@0.5 `-0.0208`；四项 95% CI 均跨 0。
- [x] 门控判定继续算法研究、暂不进入 PL：存在坐标保持信号，但没有严格 Pareto 支配或统计上已确认的优势。

下一步使用隔离的开发/评测划分扫描残差—能量融合与区域预算，避免在同一 96 图集合上调参后报告；
只有统一配置在独立评测子集形成稳定优势，才定义可编程 selector 描述符并进入 HLS/RTL。
报告：[`2026-08-25-expanded-spatial-selector.md`](../reports/2026-08-25-expanded-spatial-selector.md)。

#### 2026-08-25 参数冻结与盲验证止损

- [x] 在 96 图开发集预注册能量权重 `{0, 0.25, 0.5, 0.75, 1.0}`，以 PCK@0.1、PCK@0.2、Mean IoU、Acc@0.5 等权均值自动选参。
- [x] 冻结 `75% Token + layer-5 + 3×3 区域配额 + 纯四邻域残差`；开发集相对 L2 四项点估计均提高。
- [x] 在首次推理前锁定另一套 96 图盲集；它与 smoke/开发集共 128 张已见图片零重叠。
- [x] 完成 full、L2、冻结 residual 共 576 次盲验证和成对 bootstrap。
- [x] 盲集相对 L2：PCK@0.1 `-0.0104`、PCK@0.2 `+0.0000`、Mean IoU `+0.0041`、Acc@0.5 `-0.0104`，全部 95% CI 跨 0。
- [x] 止损：开发集优势未泛化，停止纯视觉固定残差规则，不进入 HLS/RTL。

该实验说明继续手调 L2、能量或局部残差不足以形成论文方法。下一候选必须加入可训练或文本条件信息，
但输出仍限定为规则 2×2 块、统一任务无关描述符，并保持 MRoPE/DeepStack 原位置对齐。
冻结记录：[`2026-08-25-selector-tuning.md`](../reports/2026-08-25-selector-tuning.md)。
盲验证：[`2026-08-25-selector-blind-validation.md`](../reports/2026-08-25-selector-blind-validation.md)。

#### 2026-08-25 无训练文本相关性止损

- [x] 复用 layer-5 DeepStack merger，将视觉 group 与 Prompt token embedding 投到同一 2560 维空间。
- [x] 实现文本最大余弦相关性、局部残差与区域配额的统一选择路径；100% 路径兼容。
- [x] 在 8 条/任务上测试文本权重 0.10/0.25；0.10 已降低 PCK@0.1 与 Acc@0.5，0.25 出现明显空间偏移。
- [x] 止损：不进行全量扫描，停止“未经训练的跨模态余弦”路线。

下一候选为坐标损失/教师显著性蒸馏的轻量评分头：训练阶段允许使用任务标注和完整模型梯度，
推理阶段仅使用浅层视觉特征、Prompt 摘要和固定小网络，输出硬件友好的规则块索引。
报告：[`2026-08-25-text-selector-pilot.md`](../reports/2026-08-25-text-selector-pilot.md)。

#### 2026-08-25 坐标监督学习头开发门控

- [x] 生成96图×2任务的 layer-5 `|activation × gradient|` 教师，192条均有限、非退化且可重复。
- [x] 提取推理可用的残差、能量、文本相关性、x/y五维特征；同图静态特征完全一致，文本特征随任务变化。
- [x] 训练113参数统一评分头并完成去文本、去坐标、去区域损失消融；按图做64/32训练—校准拆分。
- [x] 完成75%和87.5%真实模型输出门控；学习头未超过同预算L2，且未满足相对full四指标下降均不超过2个百分点。
- [x] 止损：不冻结、不建立第三套盲集、不进入HLS/RTL。

该结果说明从零学习绝对排序会破坏已经可靠的L2基线，且单group一阶显著性不足以表达联合删除的
非线性交互。下一候选改为 `normalized L2 + residual MLP`：末层零初始化保证初始行为严格等于L2，
仅学习小幅改序并正则化残差；L2是显式安全下界。只有开发集真实输出过门后才冻结并建立全新盲集。
报告：[`2026-08-25-coordinate-selector-development.md`](../reports/2026-08-25-coordinate-selector-development.md)。

#### 2026-08-26 L2安全锚定残差头盲验证止损

- [x] 以零初始化残差头保持初始选择严格等于L2，并按开发校准集冻结 `lambda=2`、87.5% Token、layer-5和3×3区域配额。
- [x] 冻结后新建第三套96图/192任务盲集；与此前224张已见图片零重叠，并在推理前锁定manifest与配置哈希。
- [x] full、同预算L2和冻结残差头各运行192条，共576条；运行中未重跑、改参或提前查看局部精度。
- [x] 冻结候选相对full保持PCK@0.1，PCK@0.2、Mean IoU和Acc@0.5点估计分别提高0.0208、0.0015和0.0208。
- [x] 冻结候选相对同预算L2的PCK@0.1、PCK@0.2和Mean IoU分别下降0.0104、0.0208和0.0022，仅Acc@0.5提高0.0208；所有成对95% bootstrap CI跨0。
- [x] 止损：停止一阶 `activation×gradient` 显著性蒸馏，不把113参数残差头映射到PL；保留87.5%浅层结构化L2作为安全基线。

该盲测证明残差头可以在当前样本上保持full点估计，但没有证明其优于简单L2，也没有建立统计非劣性。
下一算法阶段只使用开发集，直接以多个group联合删除后的真实坐标NLL构造集合级监督；在新的开发门通过前，
禁止再次使用本盲集或开始PL集成。
报告：[`2026-08-26-l2-residual-blind-validation.md`](../reports/2026-08-26-l2-residual-blind-validation.md)。
机器汇总：[`2026-08-26-l2-residual-blind-summary.json`](../reports/2026-08-26-l2-residual-blind-summary.json)。

#### 2026-08-26 联合删除坐标NLL最小门控

- [x] 增加固定group索引前向，校验类型、数量、唯一性、边界和Token预算。
- [x] 从L2@87.5%生成保持3×3区域计数的1/2/4-group交换候选。
- [x] 实现按真实teacher-forced坐标NLL训练的集合级pairwise ranking loss。
- [x] 在开发集每类1条上完成14个候选前向；所有掩码唯一、损失有限。
- [x] 独立重复中14个掩码和NLL逐一一致，最大NLL差异为0。
- [x] Grounding与Pointing最佳候选相对L2的NLL分别降低0.004640和0.051981。
- [x] 数据隔离保护会在加载模型前拒绝已揭盲的 `blind` 和 `validation` 清单。

该结果只证明集合级监督链路可运行且存在可学习差异，不是任务精度提升结论。下一门槛为未揭盲开发集
每类8条的规模门控；之后才能覆盖固定64/32图拆分并训练集合级排序头。报告：
[`2026-08-26-joint-deletion-coordinate-nll-pilot.md`](../reports/2026-08-26-joint-deletion-coordinate-nll-pilot.md)。

#### 2026-08-27 联合删除规模化、交互头与新测试止损

- [x] 修复同图双任务候选种子分叉，并完成96图开发集192条记录、1344次候选前向。
- [x] 逐组可加头校准pairwise accuracy为0.4903，未达到0.52门槛，停止该结构。
- [x] 545参数候选交互头在固定64/32图拆分上通过坐标NLL开发门：
  pairwise accuracy 0.5374，top-1 regret 0.0288，相对L2平均NLL降低0.0101。
- [x] 真实输出校准中Pointing PCK@0.1相对L2下降0.0625，统一交互头停止。
- [x] 将后验提出的Grounding交互头/Pointing回退L2路由在推理前冻结，并建立与既有320图零重叠的96图新测试。
- [x] 新测试上Pointing指标严格等于L2；Grounding Mean IoU下降0.0099、Acc@0.5下降0.0104，没有主指标改善。
- [x] 止损：停止当前学习式联合删除与后验阈值搜索，不把113参数或545参数头接入PL；冻结L2@87.5%。

该阶段给出了从开发教师信号、开发真实输出到未触碰新测试的完整证据链。开发坐标NLL门通过没有转化为
真实任务指标改善，因此后续PL工作只集成L2@87.5%掩码，不集成学习头。新测试、旧blind和开发校准
输出均已揭示，只可用于复现，不得继续选参。报告：
[`2026-08-27-joint-deletion-scale-and-additive-selector.md`](../reports/2026-08-27-joint-deletion-scale-and-additive-selector.md)、
[`2026-08-27-candidate-interaction-calibration.md`](../reports/2026-08-27-candidate-interaction-calibration.md)、
[`2026-08-27-task-routing-test.md`](../reports/2026-08-27-task-routing-test.md)。

### WP4：ZCU102 Token/稀疏数据面（2026-09 至 2026-10）

- [ ] HLS实现流式group评分、Top-k/预算选择、空间分区配额和全局锚点保护。
- [ ] HLS实现Token gather/merge/compact及原始索引、二维坐标、MRoPE、DeepStack映射同步。
- [ ] 支持全长旁路、L2冻结基线和可编程分数三种模式；模式差异只改变描述符和参数。
- [ ] 输出处理Token数、保留数、周期、输入/输出字节、stall和错误状态计数器。
- [ ] 先通过C Golden、HLS C仿真、定向RTL、综合时序和系统路由；取得授权后才易失性上板，不写SD/QSPI。

验收条件：全长旁路逐字一致；冻结mask的紧凑Token、位置和层级映射逐项一致；实际PL时间和流量低于收益模型上限，并报告资源和板级功耗。

### WP5：传输后端与主平台适配（2026-10 至 2026-12）

- [ ] 先完成平台无关文件/共享内存回放器，确保合同不依赖网络或某个运行时。
- [ ] 建立AXI4-Stream数据面和控制面；PS只负责配置、状态、异常和日志。
- [ ] 主平台确定前，用可配置带宽/固定延迟仿真1/2.5/5/10/40GbE、USB或PCIe情景。
- [ ] 主平台确定后，只新增一个物理传输后端和一个运行时适配器，不改PL算法语义。
- [ ] 比较主平台完整模型、主平台软件Selector、ZCU102固定Selector、ZCU102本文Selector四条路径。

验收条件：每帧最多一次主张量往返；端到端P50/P95和整机能量均实测。若ZCU102路径不能超过主平台软件Selector，则停止异构性能claim，保留算法与硬件负结果。

### WP6：可选稀疏计算面与连续帧扩展（2026-11 至 2027-03）

- [ ] 根据主平台实测交叉点，只选择块稀疏Attention、块/N:M权重或KV页筛选中的一种进入硬件。
- [ ] 复用课题组稀疏矩阵乘法积累及现有Q8 DATAFLOW/向量IP，不同时铺开全部稀疏格式。
- [ ] 仅当权重/KV可跨请求驻留时评估稀疏Decode；否则保持主平台Decode。
- [ ] 独立完成连续帧变化检测、Token复用和视觉状态管理，不与独立新图结果混合。
- [ ] 非线性近似只在可选稀疏子图中成为实测瓶颈后开展。

验收条件：同时给出理论稀疏率、实际传输/DDR节省、PE利用率、模型端到端加速和精度；连续帧另报完整刷新率与增量更新率。

### WP7：系统评测与论文（2027-01 至截稿）

- [ ] PS-only、主平台完整模型、主平台软件Selector、固定L2 PL、本文PL方法和可选稀疏计算面逐级消融。
- [ ] 独立新图与连续帧分开报告P50/P95、吞吐、总系统功耗、温度、J/图和FPS/W。
- [ ] 报告通信字节/时间、主平台空等时间、DSP/LUT/FF/BRAM、Fmax、DDR带宽和PL利用率。
- [ ] 至少在一个真实主平台完成端到端；若主张可移植性，再增加第二类后端，不以带宽模拟替代。
- [ ] 完整复现包、bitstream、驱动、运行时、模型合同和测试脚本。
- [ ] 根据正式CFP冻结投稿会议。

## 14. 评测指标

### 14.1 精度

| 任务 | 主要指标 |
|---|---|
| Pointing | 像素误差、归一化坐标误差、PCK |
| Affordance（使用 Pointing 输出协议） | 关键点误差、可抓取区域命中率、PCK |
| Grounding | IoU、Acc@0.5、中心点误差 |
| Trajectory | ADE、FDE、成功率、碰撞率 |
| 普通语言 | 输出一致性、任务完成率、困惑度或参考答案指标 |

### 14.2 性能与能效

- Vision Encoder、视觉 Prefill、文本 Prefill、Decoder 分阶段延迟。
- TTFT、每token延迟、独立新图FPS、连续帧完整刷新率与增量更新率。
- P50/P95 延迟和长时间运行稳定性。
- PL DDR 有效带宽、PE 利用率和计算/传输重叠率。
- 主平台、ZCU102、互连和整套系统的平均/峰值功耗、J/图、FPS/W。
- DSP、LUT、FF、BRAM、Fmax 和时序裕量。

## 15. 关键决策顺序

1. 先完成 Qwen3-VL-4B BF16、RoboBrain2.5-4B BF16、RoboBrain2.5-4B Q8_0 三层基线，确认模型专用化收益和量化损失。
2. 再完成真实任务集、逐算子 Profiling 和实测 Roofline。
3. 冻结ZCU102为平台无关的Token/稀疏数据面，而不是预先指定它替代某个平台的稠密算子。
4. 先定义三个候选边界和统一描述符，用主平台参数化收益模型冻结一个主边界、一个消融边界。
5. 首个硬件交付是Selector/Merger/Compactor；现有稠密GEMM/FFN作为能力基线和可选计算面。
6. Token削减优先进入Vision Encoder内部；Merger后边界用于先打通系统合同和对照。
7. 主平台执行规则稠密GEMM；动态/不规则稀疏算子只有通过端到端收益门才进入PL。
8. Decode默认留在主平台；只有权重/KV常驻ZCU102且实测交叉点成立时才重新开启稀疏Decode。
9. 非线性近似只有在已选PL子图内成为实测瓶颈后才提升优先级。
10. 独立新图3–5 Hz与连续帧增量3–5 Hz分别报告；不得用应用控制频率替代模型推理频率。
11. 所有硬件修改先通过testbench/协同仿真，再生成bitstream和上板。

## 16. 近期立即执行项

- [x] 冻结 Qwen3-VL-4B BF16、RoboBrain2.5-4B BF16 和 RoboBrain2.5-4B Q8_0 的 hash、Prompt、解码参数与评价脚本。
- [x] 为切入点Oracle建立与既有416图零重叠的16图开发清单；正式新算法测试集仍须在冻结后建立且标签不可见。
- [x] 在推理后端增加逐算子类型、Shape、耗时和调用次数统计。
- [x] 设计 PL DDR/AXI 带宽微基准及 RTL testbench。
- [x] 建立 Vision Encoder 和 LLM Prefill 的 GEMM Shape 清单。
- [x] 编写`Token/Block Sparse Descriptor v1`规范、C参考实现、序列化往返与损坏描述符负例测试；普通与ASan/UBSan均10/10通过。
- [x] 完成Patch Embedding、视觉块1/3、首个DeepStack层5、Merger后五个边界的静态shape/字节/MAC审计；正式描述符合同待Oracle选边界后冻结。
- [x] 建立单向驻留与外置往返通信模型；当前数值为串行化下界，主平台计算和真实链路测量仍待补充。
- [x] BF16在全新开发集完成100%/75%/50%/25%几何Oracle烟雾门；既有L2@87.5%只作冻结回退基线。
- [x] 最终Q8_0 GGUF路径两级复验完成：Q8-A Merger后实现通过、75%质量失败；Q8-B视觉块1后100%身份通过、75%与87.5%质量失败。
- [x] 定义 PS 工作负载描述符；PL 接口不包含 RoboBrain 任务名称，并通过5/5模拟执行测试。
- [x] 固化并执行完整视觉Prefill/Vision/Decoder固定分块与自动分块板测A/B、APM和追加日志协议；
  15阶段全部通过，SD/QSPI未写入。
- [x] 既有第一个PL计算核为Vision/LLM共用分块GEMM，作为能力基线保留。
- [ ] 新主线第一个PL内核：任务无关的规则group Selector/Merger/Compactor；因真实Q8质量门失败暂缓，不冻结当前Oracle配置或进入RTL。
- [x] 确认 Vivado 综合许可证和工具链可用性。

精确下一步（不访问板卡、不使用GPU）：

```bash
sed -n '1,260p' reports/2026-09-10-q8b-real-q8-validation.md
```

随后建立平台无关的切入点收益模型，量化不同主算力平台、链路带宽与驻留策略下的净延迟和能量交叉点。两级Q8机制代码保留为实验路径；当前质量门未通过，不进入Selector/Merger/Compactor RTL，也不在已观察8样本上继续调参。
