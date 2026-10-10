# Token Selector 反事实实验后的证据清单

日期：2026-10-05

这份清单只回答两件事：哪些结论已有直接证据，哪些结论仍不能成立。它不是新实验，也不替代原始报告。

## 一、已经成立的工程结论

1. **固定预算、规则分组和空间锚点的数据合同可以实现。** 当前实现按视觉 group 工作，支持 K48/K64/K96、3x3 区域锚点、确定性排序、原始位置与元数据输出，并有身份旁路。
2. **整数评分与选择流水线已完成软件和 HLS 功能验证。** `integer_selector_pipeline` 的 Python fixture、g++、ASan/UBSan、HLS C-sim 共 23 个 case、383 个检查全部通过，失败数为 0。
3. **紧凑 Selector IP 的 200 MHz OOC 实现通过。** ZCU102 器件 `xczu9eg-ffvb1156-2-e` 上 post-route WNS=+0.529 ns、WHS=+0.028 ns；资源为 LUT 10.07%、FF 5.65%、DSP 8.33%、BRAM tile 13.43%。这是 OOC 布局布线结果，不是板端结果。
4. **最小 Zynq BD 的结构验证通过。** 已连接 PS、时钟复位、三个 128-bit AXI master、AXI-Lite 控制与 DDR 地址段，`validate_bd_design` 无错误；尚未生成 bitstream、驱动或板端测量。

注意：上述 OOC IP 的 scorer 输入包含 `Wv_q` 和 `query_q`，对应现有整数投影/Prompt 评分硬件，不等同于 full2638 正式对照使用的纯纠正 L2 分数。硬件可实现性通过不能反向证明该 Prompt 评分算法有效；纯 L2 的专用精简实现也尚未单独签核。

## 二、已经成立的模型侧事实

1. CV-Bench full2638 的 dense BF16 官方主分为 **0.872730**。
2. 当前纠正后的整数 L2+3x3 anchor、B1、K96 主分为 **0.766852**，约为 dense 分数的 **87.87%**；绝对差约 **-0.10588**。这可以作为可复现压缩基线，但不能称为精度等价。
3. 当前 Prompt 分支 K96 主分为 **0.777362**；相对整数 L2 的增量为 **+0.010510**，95% CI **[-0.005430,+0.025699]**，增益未确认。
4. 同图双分辨率四臂反事实的预注册交互效应为 **-0.07**，95% CI **[-0.20,+0.06]**。low 档 Prompt 净修正 +6，high 档净修正 -1。因此不能确认 Prompt 增益随视觉 token 数或 `G>512` 增强。
5. 先前内部 OOF `G>512` 路由与数据源高度相关（source 与 gate 的 Cramer's V=0.880783），只能保留为探索性诊断，不能作为因果机制或论文主张。

## 三、性能证据的正确边界

1. 软件 Selector 的同机 A6000 测量中，旧 current-selector K96 路径的模型推理均值从 dense 的 237.32 ms 降至 168.55 ms，约 **1.408x**；应用端均值从 256.25 ms 降至 185.41 ms，约 **1.382x**。这是软件路径结果，不是 FPGA 加速结果，也不是纠正 L2 K96 的重新计时。
2. 现有异构结果仅为投影。K96 在 100 张图上的 FPGA 计算估计均值为 9.16 ms；假设 1 GbE 有效 110 MB/s、无重叠时，传输估计均值为 37.29 ms，投影应用端均值约 183.71 ms，几乎抵消 FPGA Selector 的计算优势。
3. 假设 10 GbE 有效 1.1 GB/s 时，K96 投影应用端均值约 150.15 ms；该数字依然不是链路或板端实测。
4. 目前不能声称 FPGA 端到端加速、板级能效、真实通信带宽或 Jetson Xavier NX 端到端性能。

## 四、已经被否定或尚未建立的主张

- **否定：** `G>512` 是启用 Prompt 条件分支的有效因果门。
- **未建立：** 当前 Prompt 投影/对比学习分支能稳定提高官方任务精度。
- **未建立：** 当前 K96 Selector 与 dense 精度等价。
- **未建立：** 当前异构往返在 1 GbE 下带来可重复端到端收益。
- **未建立：** OOC 时序通过等同于完整 BD、bitstream 或板端通过。

## 五、仍可复用的研究资产

- 固定预算、16 对齐、规则 group 和 3x3 anchor 合同；
- 整数 L2 评分、确定性 top-K、紧凑 arena 与元数据布局；
- 已通过 OOC 的紧凑 Selector IP 和最小 Zynq BD；
- full2638 官方 CV-Bench scorer、配对分层 bootstrap 与 fail-closed 报告链；
- K96 整数 L2 基线，作为后续新机制必须超过的同预算对照；
- 同图反事实方法，作为排除数据源/分辨率混杂的模板。

## 六、下一候选不得原样重复的失败路线

1. **纯空间 KL 学生：** `SPATIAL_STUDENT_NOT_ESTABLISHED`。虽然空间监督相对语义控制的排序相关性提高，但 Prompt 敏感性和部署 K48 效用门失败。
2. **空间 + Prompt 对比学生：** `PROMPT_CONTRASTIVE_STUDENT_NOT_ESTABLISHED`。K48 retained-target-mass 门通过，但两项 Prompt 区分门没有全部通过，且未验证 CV-Bench 官方精度。
3. **旧 Prompt 投影蒸馏：** validation 上 `correct_beats_shuffled=false`，最佳 checkpoint 接近初始化；其 grounding/pointing 文本分布也与 CV-Bench 问题分布不同。
4. **旧 operational Prompt 诊断：** `OPERATIONAL_PROMPT_UTILITY_NOT_ESTABLISHED`；不能把内部 retained mass 或排序相关性代替官方任务精度。
5. **只按 G 自适应预算：** screening100 的 adaptive-K 相对 fixed-K96 增量为 +0.01667，95% CI `[-0.04583,+0.08333]`，没有确认；且 G 与数据源存在强混杂。
6. **只改 context pooling 或 bounded/unbounded 选择约束：** 两项 screening100 CI 均跨零，没有证明是主要精度缺口来源。

新的机制若仍采用上述教师、损失或路由变量，必须明确指出新增了哪个可检验变量；换名字或追加阈值不算新候选。

## 七、下一阶段的硬门

新算法候选必须在**相同 B1、相同 K、相同解码与相同官方 CV-Bench 指标**下与纠正整数 L2 配对比较。开发阶段不得使用 `G`、数据源名称或任务标签进行路由；只有冻结候选在未用于选择的样本上，其相对 L2 的官方分数增量置信区间下界大于 0，才允许进入 full2638 和 HLS 映射。若开发门失败，停止该机制，不用事后阈值挽救。

下一候选机制尚未冻结，等待独立策略助手任务 `20261005-001` 的文献与可证伪性审查后由 Codex 决策。

## 证据来源

- `reports/cv_bench/2026-10-04-b1-corrected-l2-prompt-ablation-full2638.json`
- `reports/cv_bench/2026-10-04-same-image-resolution-counterfactual.json`
- `reports/cv_bench/2026-10-04-gate-source-confounding.json`
- `reports/cv_bench/2026-10-04-current-selector-quality-latency.json`
- `reports/token_selector/2026-10-04-current-selector-heterogeneous-latency-projection.json`
- `reports/token_selector/2026-10-03-integer-selector-pipeline-hls.json`
- `reports/token_selector/2026-10-03-integer-selector-compact-ip-ooc-route.json`
- `reports/token_selector/2026-10-03-integer-selector-minimal-zynq-bd.json`
