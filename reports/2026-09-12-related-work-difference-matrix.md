# 按需视觉细化相关工作差异矩阵

日期：2026-09-12。范围：核对与当前“低清全局观察、按需恢复局部细节”最直接重合的四项工作。网页与公开代码会更新，投稿前需按冻结版本再次审计。

| 工作 | 选择/恢复机制 | 训练 | 任务与评价重点 | 缓存与开销边界 | 与本项目的直接重合 | 本项目仍可验证的差异 |
|---|---|---|---|---|---|---|
| AdaptVision，CVPR 2026 Highlight | 先看下采样全图；模型生成bbox工具调用；追加原图裁剪后再回答，允许直接停止 | 基于Qwen2.5-VL，使用RL和精度/工具代价奖励；公开实现依赖veRL/vLLM | 通用视觉推理与细节恢复，报告视觉Token效率 | 公开评估代码把新裁剪追加到`multi_modal_data`并再次调用`generate`；未发现面向FPGA链路、DDR字节或ROI流水的报告 | 主动判断是否重看、低清全局加局部裁剪、两轮生成 | 空间误差收益预测而非长推理文本工具调用；固定硬件预算；显式累计Token/字节/端到端时延；RoboBrain Q8与FPGA协同 |
| Visual Perception Tokens，ICCV 2025 | Region Selection Token触发区域裁剪重编码；Vision Re-Encoding Token以隐藏状态指导额外视觉编码，追加视觉嵌入后继续生成 | 829k样本，覆盖通用、细粒度、空间和OCR任务；需训练专用控制Token | 空间推理、细粒度理解、OCR/VQA | 论文描述追加额外视觉嵌入；没有可直接移植的ZCU102数据流与链路成本结论 | 模型自主触发重看、区域选择、附加视觉表示 | 不改词表的轻量价值预测器；针对坐标误差的监督；Q8后端兼容；硬件可执行的有界ROI队列与停止门 |
| TEVA，ICCV 2025 | Relevant Area Proposal识别主体区域，动态多尺度Patch采样，同时保留全局Patch且不增加Token总数 | 四阶段训练，补充材料报告64张A100；数据构建用开放词汇检测筛选主体问答 | 高分辨率细节、主体导向理解；不以文本密集OCR为主要目标 | 单次重组视觉采样；不是自回归按需停止，也未报告跨设备缓存 | 全局上下文加局部高分辨率Patch、任务相关区域 | 以实测空间损失降低量选择区域/尺度/停止；累计成本约束；面向异构DMA和重复ROI去重 |
| Reroute，2026预印本 | Decoder内被延后Token绕过若干层，后续重新进入候选池；不重新读取像素 | 免训练，复用FastV/PDrop/Nuwa评分与分层排程 | RefCOCO系列Grounding与通用VQA；强调激进裁剪下空间信息恢复 | 保持所替代裁剪基线的理论FLOPs与KV预算级别；恢复的是已有视觉Token，不是高分辨率像素细节 | “先不计算、后按需恢复”以及空间任务对不可逆删除敏感 | 从原始RGB获取此前未编码的细节；区域动作的端到端价值模型；图像链路和FPGA预处理成本 |

## 不能宣称的新颖性

- “先低清看图，再放大局部”不是新颖点。
- “模型自动决定是否重看”不是新颖点。
- “保留全局Patch并加入局部高分辨率Patch”不是新颖点。
- “不永久丢弃Token、后续允许恢复”也不是新颖点。

## 当前可守住的研究问题

候选贡献应收敛为：对具身空间任务，在固定端到端硬件预算下预测每个`(区域, 尺度, 停止)`动作带来的真实空间误差降低量，并把累计视觉编码、Prefill、链路字节和调度成本纳入同一个决策目标；再证明有界区域请求、去重和批量DMA在ZCU102异构通路上带来可测净收益。

这仍只是待验证假设。若最终只实现固定真值框裁剪、多图Prompt或普通Resize，不能作为算法创新。

## 一手来源

- AdaptVision项目与代码：https://adaptvision.github.io/ ，https://github.com/AdaptVision/AdaptVision
- Visual Perception Tokens论文：https://openaccess.thecvf.com/content/ICCV2025/html/Yu_Auto-Controlled_Image_Perception_in_MLLMs_via_Visual_Perception_Tokens_ICCV_2025_paper.html
- TEVA论文：https://openaccess.thecvf.com/content/ICCV2025/html/Jiang_Token-Efficient_VLM_High-Resolution_Image_Understanding_via_Dynamic_Region_Proposal_ICCV_2025_paper.html
- Reroute论文与代码：https://arxiv.org/abs/2606.12412 ，https://github.com/elmma/mllm-reroute

