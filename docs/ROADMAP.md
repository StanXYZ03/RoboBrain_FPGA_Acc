# ZCU102 部署与加速路线图

> 2026-10-10 最新状态：VSR zero-shot dev 340 条短 True/False 预算曲线已完成。Dense 0.7912；
> v2 K96/K144 0.6853/0.7382（相对 Dense 0.8662/0.9331），两档相对同预算 L2 均 +0.0206
> 但 95% CI 跨 0。现有 compact HLS 只支持 K<=96，K96 距 0.88 宽松质量线差 4 条；K144
> 虽过质量线但不兼容现有接口。用已接受 HLS 周期模型投影，K96 经 1 GbE 往返为 91.56 ms，
> 对 Dense model inference 88.98 ms 无净收益；假设 10 GbE 为 76.55 ms。下一步固定 K96，
> 用 VSR zero-shot train 发展关系保持目标，并优先解决 PCIe/共享内存或切点重构；test 保持封存。

> 2026-10-09 外部确认结论（最新权威状态）：SpatialEval 三臂各 4500 条已经完成。
> Dense/L2@K96/v2@K96 的项目等权宏平均为 0.583778/0.439111/0.434444；v2/Dense
> 仅 74.42%，v2-L2=-0.004667，95% CI [-0.013778,+0.004444]，两项预注册门均失败。
> 当前固定 K96 v2 不再作为主方案晋级，SpatialEval test 不得用于后续调参。视觉 token
> 虽减少 79.53%，但 Decode 占 Dense elapsed 85.57%，当前软件 v2 反而慢 3.24%；下一轮
> 必须把自适应空间关系保持与结构化短解码/非自回归动作头联合设计，并换未观察公开基准
> 做下一次外部确认。详见 `reports/token_selector/2026-10-09-spatialeval-three-arm-summary.*`
> 和 `reports/token_selector/2026-10-09-spatialeval-external-failure-analysis.*`。

> 2026-10-09 外部确认状态（权威更新）：SpatialEval VQA 主集已在查看模型输出前冻结为
> Spatial-Map、Maze-Nav、Spatial-Grid 各 1500、共 4500 条；Spatial-Real 135 条因官方
> evaluator 缺少有效 handler 而只作未来独立诊断。正式配置
> `configs/2026-10-09-spatialeval-vqa-three-arm-external-confirmation-r2.preregistered.json`
> SHA `bd8e1a23…708014`，35 pins 与 35 项 CPU 测试通过。r1 在 93 条未评分 Dense 前缀后
> 发现 v2 checkpoint 直接固定缺口，已在不观察答案/分数、也不改变算法或门槛的条件下由 r2
> 接管并固定前缀与三个真实 checkpoint；三臂为同输入 Dense、修正整数
> L2@B1/K96、冻结 v2@B1/K96。主指标复现官方 parser+子串判断；三任务等权宏平均是项目
> 预注册汇总，不是官方总分。门槛为 v2/Dense 宏平均保留 `>=0.90` 且 v2−L2 的
> `task::question_id` 分层配对 bootstrap 95% CI 下界 `>=0`；逐任务只报告。标签已物化，
> 所以不是 label-blind，但 Selector 输入不含 oracle，且本 split 从未用于调参。1459 个去重
> 图像已在 `/tmp` 校验，Dense 第一臂正在 append-only 执行、尚未评分；观察输出后不得再调
> Selector。A6000 软件计时仍不等于 Jetson/ZCU102 板端加速。

> 2026-10-08 完成状态（权威更新）：edge576 CV-Bench 三臂发展性实验已完成。r3 因 dense 生产者
> 缺 `output_token_ids` 在写文件前 fail-closed 并被 r4 取代；r4 config SHA256
> `4f11d47981fd6c04c07608eba8b7e85e27e035bcb8d2d459a7ebc8a8c4b682c5`（63 pins）修复为
> token ids 可选、`output_token_count` 必备且不伪造。三臂 smoke 全通过，三份 full2638 唯一且
> 同集合（Dense `707868c6…`、L2 `e11dba93…`、v2 `8b2a2885…`）。官方 `cv_bench_accuracy`：
> Dense `0.8739581555`、L2 `0.7937869555`（0.908267 Dense）、v2 `0.8078459446`（0.924353
> Dense）；诚实解析失败 36/42/41。`task::source_dataset` 分层配对 bootstrap（seed
> 20261007/5000）v2−L2 `+0.0140590` CI `[+0.0044580,+0.0233407]`，v2/Dense `>=0.90` 与
> v2−L2 CI 下界 `>=0` 两门通过，状态 `DEVELOPMENT_GATES_PASS_GO_EXTERNAL_VALIDATION`（仅
> 发展集；不得写成论文级精度或外部泛化已确认）。时延均为软件时延、非 FPGA/板端实测；Dense
> 无 `application_end_to_end` 字段，必须为 `null`。正式汇总见
> `reports/token_selector/2026-10-08-edge576-cvbench-three-arm-r4-summary.{json,csv,md}`
> （artifact SHA `1957c323…`，汇总器/测试 28 OK）。**下一里程碑**：在查看结果前冻结一个此前
> 未用于调参的公开空间 VLM 外部确认基准并只运行一次确认；并行接入 Jetson Xavier NX 重测真实
> 阶段/链路延迟，ZCU102 仅在净收益合同下评估 selector，当前不宣称 FPGA 加速。

> 2026-10-07 权威更新：本项目按**边缘部署**定义主运行域。未来 Selector 主实验将完整
> CV-Bench 2638 样本统一缩放到 Patch Merger 后 `G<=576`，Dense、修正整数 L2 和候选方法
> 使用完全相同的输入、Prompt 与解码配置；固定主预算为 K96。原生动态分辨率（最高 G=3588）
> 只保留为压力测试，不再阻断边缘主档位。主质量指标使用 CV-Bench 官方
> `cv_bench_accuracy`，成功定义为相对同分辨率 Dense 保留 `>=90%`，且相对同预算修正整数
> L2 的分层配对 bootstrap 95% CI 下界 `>=0`；解析率按显式 parsed-answer 字段诚实统计。
> General/Pointing/Grounding/Trajectory 仅是 API 路由，不再用 PCK/IoU/ADE/FDE 作为
> Selector 主晋升门。详细冻结口径见
> [PL_ACCELERATION_RESEARCH_PLAN_2027.md](PL_ACCELERATION_RESEARCH_PLAN_2027.md) 顶部
> “2026-10-07 边缘部署运行域与成功判定”。旧原生分辨率结果及其 `DEVELOPMENT_STOP`
> 保持只读，不追溯改判。

> 2026-08-22 更新：面向 PL 加速、视觉 token 削减、结构化稀疏和 2027 投稿的详细研究计划见
> [PL_ACCELERATION_RESEARCH_PLAN_2027.md](PL_ACCELERATION_RESEARCH_PLAN_2027.md)。本文保留为早期部署与首轮硬件迭代路线。

## 1. 基线冻结（质量基线已完成；部署性能基线待补）

- [x] 固定模型、transformers/PyTorch 版本、CV-Bench manifest、Prompt 和 greedy 解码设置。
- [x] 主部署档位固定为完整 2638 样本、Patch Merger 后 `G<=576`；逐条保存实际
  `image_grid_thw` 与 G，禁止运行后按 G 或正确性删样本。
- [x] 在相同 edge576 输入上分别冻结 Dense、修正整数 L2@K96 和候选 Selector@K96。
- [x] 在 A6000 软件路径记录视觉编码、prefill、decode 等分阶段耗时；Dense 缺少统一的
  `application_end_to_end_seconds`，该项保持 `null`，不得与另外两臂的 application 字段直比。
- [x] 记录 BF16 输出和 CV-Bench 官方主分，作为后续量化回归参考。
- [ ] 接入 Jetson Xavier NX 后，以统一边界补测三臂 application end-to-end、TTFT、tokens/s、
  峰值内存和链路开销；完成前不得称部署性能基线已经闭环。

完成状态：官方主分、诚实解析率和 A6000 分阶段软件计时已完成；统一的 Jetson 部署性能报告待补。

## 2. 模型压缩

- 优先验证 W4A8、W4A16 和分层混合精度，不直接假设全模型 INT4 可用。
- 对视觉编码器、语言层和输出头分别做敏感性分析。
- 控制图像分辨率与视觉 token 数，量化精度和时延之间的 Pareto 前沿。
- 若 4B 模型仍无法达到实时目标，再评估蒸馏后的学生模型。

完成条件：确定可接受精度下降下的权重、激活和 KV cache 格式。

## 3. PL 计算内核

- 第一版实现带 AXI4-Stream/AXI4 master 接口的分块 GEMV/GEMM。
- 融合 INT4 解包、零点/尺度处理和累加，避免中间张量回写 DDR。
- 用 C/RTL 协同仿真验证数值，综合报告记录 LUT、FF、BRAM、URAM、DSP 和 Fmax。
- 通过双缓冲、burst 对齐和计算/搬运重叠提升有效带宽。

完成条件：板端内核结果与软件参考一致，并提供实测带宽与 TOPS。

## 4. PS 运行时与模型拆分

- 在 A53 上实现缓冲区分配、寄存器控制、中断/轮询和层级调度。
- 明确哪些算子留在 PS，哪些下沉 PL；避免小算子频繁跨边界。
- 支持逐层权重流式加载和 KV cache 管理。
- 将运行时封装为稳定 C/C++ API，再接入 Python 应用。

完成条件：完成至少一个 Transformer block 的端到端板上执行。

## 5. 集成与验收

- 接入 RoboBrain 应用 API；四种 API 路由用于功能回归，不作为 Selector 主质量门。
- 对比 CPU 基线、主机 GPU 参考、PS-only 和 PS+PL 四种配置。
- 主质量使用 edge576 CV-Bench 官方分数及相对同输入 Dense 的保留率；另报诚实解析率。
- 记录 P50/P95 延迟、吞吐、功耗、温度和质量；注明估算或实测。
- 生成可复现部署包：bitstream、设备树/overlay、运行时、量化权重和启动脚本。

完成条件：在 ZCU102 冷启动后可离线运行固定验收集，并产生统一报告。

## 首个实现迭代

首个硬件迭代建议只实现语言模型中的单层量化线性算子。它能快速验证权重格式、
AXI 带宽、数值精度和运行时控制路径。确认真实瓶颈后再决定是否融合 RMSNorm、
RoPE、激活函数或 attention，避免在缺少板卡数据时过早扩大硬件范围。
