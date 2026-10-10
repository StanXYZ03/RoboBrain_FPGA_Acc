# 视觉 Token 缩减阶段结论与创新边界

日期：2026-08-25  
状态：`A6000_FEASIBILITY_COMPLETE / ACCURACY_GATE_NOT_PASSED / SELECTOR_RESEARCH_GO`

## 1. 本阶段完成项

- 在 RTX A6000 上加载原始 RoboBrain2.5-4B BF16，一次驻留完成 10 组配置、640 次真实图片推理。
- 扫描 `uniform` 与不可部署的 `oracle_crop`，请求保留率为 100%、75%、50%、25%、12.5%。
- 实际改变图片网格和视觉 Token 数，并记录 Vision Encoder、LLM prefill、decode、总延迟和峰值显存。
- 以 32 张 COCO 图片上的 32 条 Pointing、32 条 Grounding 真实标注计算 PCK、IoU 和 Acc@0.5。
- 没有修改 PL、板端软件、模型权重或量化格式。

## 2. 核心证据

匹配的约 1/4 Token 预算下，`uniform` 实际保留 23.9%，`oracle_crop` 实际保留 24.6%。

| 指标 | Uniform | Oracle | Oracle 改善 |
|---|---:|---:|---:|
| PCK@0.1 | 0.3438 | 0.5312 | +0.1875 |
| PCK@0.2 | 0.6875 | 0.8125 | +0.1250 |
| Mean IoU | 0.6274 | 0.7545 | +0.1271 |
| Acc@0.5 | 0.6875 | 0.8750 | +0.1875 |

结论：相同 Token 数下，空间选择策略显著影响定位精度，开发选择器有明确价值；但 oracle 使用真实框，不能部署，也不能作为论文算法结果。

相对完整输入，oracle@1/4 的 Encoder 与 LLM prefill 主运算 MAC proxy 分别为 22.5% 和 24.3%。把该比例一阶映射到既有 ZCU102 PS 实测阶段时间：

`625.425×0.225 + 1003.155×0.243 + 65.498 ≈ 449.985 s/图`

即理论约 `3.77×`、`0.00222 Hz`。该估算假设选择器零开销且时间随 MAC 线性缩放，不是板端实测；距离 3 Hz 仍约差 1350×。因此 Token 缩减必须与 PL 矩阵/Attention 加速、跨帧复用或更小模型共同使用。

## 3. 为什么当前精度门没有通过

完整输入的 PCK@0.1、Mean IoU、Acc@0.5 分别为 0.5625、0.8090、0.9062；oracle@1/4 为 0.5312、0.7545、0.8750。二值指标只变化 1/32，但 Mean IoU 下降 0.0545，超过烟雾门的 0.02。

原定“精度下降不超过 2 个百分点”也无法在每任务 32 条上检验：单个样本就对应 3.125 个百分点。因此结论只能是可行性信号，不是论文级精度结论。

## 4. 截至 2026-08-25 的文献边界

以下结论来自论文原文或会议正式页面；这是一轮定向检索，不宣称穷尽全部文献。

| 已有方向 | 代表工作 | 对本项目的约束 |
|---|---|---|
| 文本/注意力引导剪枝 | [SparseVLM, ICML 2025](https://proceedings.mlr.press/v267/zhang25s.html)、[TransPrune, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_TransPrune_Token_Transition_Pruning_for_Efficient_Large_Vision-Language_Model_CVPR_2026_paper.html) | “文本引导选择”本身不是创新点。 |
| 实例/逐层自适应预算 | [ATP-LLaVA, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Ye_ATP-LLaVA_Adaptive_Token_Pruning_for_Large_Vision_Language_Models_CVPR_2025_paper.html) | “不同图片或层使用不同保留率”本身不是创新点。 |
| 空间覆盖与多样性 | [FEATHER, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Endo_Feather_the_Throttle_Revisiting_Visual_Token_Pruning_for_Vision-Language_Model_ICCV_2025_paper.html)、[CoIn, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Du_CoIn_Coverage_and_Informativeness-Guided_Token_Reduction_for_Efficient_Large_Multimodal_CVPR_2026_paper.html) | “保留全局覆盖或多样 token”本身也不够。 |
| 空间定位专用剪枝 | [Nüwa, 2026](https://arxiv.org/abs/2602.02951)、[LiteLVLM, ICML 2026](https://arxiv.org/abs/2605.13178) | 已有工作明确发现通用剪枝破坏 grounding，并提出空间完整性或文本引导方法。 |
| 场景/任务自适应 | [Prune2Drive, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Xiong_Prune2Drive_A_Plug-and-Play_Framework_for_Accelerating_Vision-Language_Models_in_Autonomous_CVPR_2026_paper.html) | 具身/自动驾驶场景的任务相关预算也已有先例。 |
| FPGA 大模型计算 | [LUT-LLM, 2025](https://arxiv.org/abs/2511.06174) | 仅把 LLM 线性层搬到 FPGA、仅做低比特或 LUT 计算不足以形成 VLM 创新。 |

## 5. 可继续推进的创新假设

暂定主线不是“再发明一种 top-k”，而是以下联合问题：

> 在 Pointing/Grounding 坐标精度约束下，生成保持原始二维 MRoPE 与 DeepStack 对齐的规则块 Token；再用精度—Roofline 联合代价模型自动选择层、预算、块形状和 FPGA 映射，使动态压缩仍能转化为连续 AXI burst 和高利用率稠密 GEMM。

可检验贡献拆为三项：

1. **坐标保持选择目标**：同时约束目标相关性、全局空间锚点和坐标误差，不以通用 VQA 平均分替代空间精度。
2. **硬件友好的规则块压缩**：选择 2×2/4×4 层次块和少量全局锚点，避免任意 top-k 导致不规则索引、短 burst 和 PE 负载不均。
3. **精度—硬件联合映射**：PS 根据输入复杂度和精度约束生成 token 预算及 M/N/K、tile、层位置描述符；PL 保持一套任务无关阵列。

这三项必须共同验证。只完成任意一种通用剪枝、固定分辨率缩放或稠密 HLS GEMM，都不足以支撑 DAC/FPL 级方法贡献。

## 6. 下一阶段及止损门

先不接 PL，先在 A6000 实现无真值的规则块选择器：

- 基线：uniform、random、L2/视觉显著性、文本相似度，以及至少一个可适配的公开空间剪枝方法。
- 候选：浅层视觉特征 + prompt 相关性 + 分区锚点；只输出规则块索引并保持原 position ID/MRoPE。
- 预算：50% 与 25% 为主，12.5% 仅作极限点。
- 数据：开发集 Pointing/Grounding 每类至少 250 条；补充复杂 referring expression，之后再加 Affordance/Trajectory 回归集。
- 统计：正式 held-out 集先做配对非劣效功效分析再定样本量。若按独立二项比例粗估，95% 区间半宽 2 个百分点在准确率约 0.9 时需约 865 条/任务，最坏 0.5 附近约需 2401 条/任务；250 条只用于筛选方法，不能证明 2 个百分点非劣。

进入 PL 动态压缩的门槛：在不使用真值的情况下，实际 Token≤25%，空间指标相对完整输入的下降落入预设置信界，同时显著优于 matched uniform，并且规则化后的理论 DDR/PE 利用率优于任意 top-k。未满足时，停止该选择器路线，保留稠密 PL 主线。

## 7. 可复现材料

- 完整 Pareto 报告：`reports/2026-08-25-token-reduction-pareto.md`
- 机器可读汇总：`reports/2026-08-25-token-reduction-pareto.json`
- 640 条原始预测：`evaluation/results/token_reduction_full/`
- 扫描脚本：`evaluation/run_hf_token_reduction.py`
- 汇总脚本：`evaluation/build_token_reduction_report.py`
- 坐标与 MAC proxy：`evaluation/token_reduction.py`
- 单元测试：`evaluation/tests/test_token_reduction.py`
