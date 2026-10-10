# 冻结权重 scorer-reduction 消融独立审校

## 审校结论

Codex 独立检查了实现、预注册、split 加载顺序、Holm-Bonferroni 校正、promotion gate、硬件代价口径和输出报告，并在独立临时目录复跑验证集。

- 预注册文件先于 val/test 结果生成；val 完成并将 `NO_PROMOTION` 写入 `val-gate.json` 后，代码才按 split 加载 train/test NPZ。
- donor 为 split-local；跨图 donor 是 image-level 固定 seed derangement，并保持 same-task；paired 统计以 image 为独立单位。
- Holm family 为 10 个非 baseline 候选 x 2 个 control x 2 个 primary metric，共 40 项。
- 独立复跑的 11 个候选 val Spearman/NDCG 与正式报告逐项完全一致，最大绝对差为 0.0；winner、passing list 和 preregistration hash 均一致。
- 独立测试：聚焦 36/36 OK；`evaluation/tests` 全量 382/382 OK。

因此，`winner=NO_PROMOTION` 是可复现的。更准确的表述是：多数平滑归约已经消除逐元素 prompt 不变量，但冻结的 `Wt/Wv` 下没有恢复可统计区分的正确 prompt 信号，同时部分方案降低了 teacher alignment；不能把这一结果表述为“所有归约仍然发生 exact collapse”。

## 解释边界

`mean_cos` 与 `prompt_mean_vector` 的组内排序在数学上等价：后者只是将平均 prompt unit vector 再除以一个对该样本所有视觉 group 相同的正标量。因此二者的 Spearman、NDCG 和 top-K 指标相同是预期现象，不应作为两份独立算法证据。二者的硬件实现代价不同，`prompt_mean_vector` 仍可作为单条 `1x128` query 的硬件实现形式。

这两个高度相关候选被同时纳入 Holm family 会使校正略保守，但本次所有候选连未校正的效应量和置信区间门槛也没有同时通过，所以不影响 `NO_PROMOTION`。

硬件表仅是解析运算与传输计数，未计定点归一化、量化 scale、控制开销和 DMA 对齐，也不是 ZCU102 综合或延迟实测结果。

## 下一研究决策

不应继续在冻结 `Wt/Wv` 上增加更多归约函数。下一步应在 train/val 上重新训练一个固定、硬件友好的 `prompt_mean_vector` scorer，或改变 prompt-sensitive supervision；候选、损失、seed 和 gate 必须在重新训练前预注册，test 保持封存。该步骤会改变训练方法，属于需要 PI 确认的研究决策。
