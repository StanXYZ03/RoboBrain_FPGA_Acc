# 坐标监督轻量 Token Selector 训练规格

状态：评分头、损失、教师显著性与推理特征链路均已通过真实模型验证；进入开发集训练。

## 目标

训练阶段用完整 RoboBrain2.5-4B 和空间标注产生 group 级教师显著性；推理阶段冻结大模型和
113 参数评分头，在 vision layer-5 对规则 2×2 merger group 评分，保留原始 RoPE/MRoPE 与
DeepStack 索引。Pointing、Grounding 使用同一网络和同一预算。

## 推理特征

每个 group 仅使用五个标量：四邻域残差、RMS 能量、Prompt 相关性、归一化 x、归一化 y。
三项内容特征逐图标准化。评分头为 `Linear(5,16) → ReLU → Linear(16,1)`：113 个参数、
每 group 96 MAC；相对 4B 主模型可忽略，并适合定点化或直接映射到 PS/PL 控制单元。

## 教师标签

1. full 模型按标注坐标构造目标答案并执行 teacher forcing；模型权重冻结。
2. 保留 vision layer-5 group feature 梯度。
3. group saliency 定义为四个 patch 的 `|activation × gradient|` 均值。
4. Pointing 使用点坐标 token NLL；Grounding 使用四个框坐标 token NLL。
5. 教师标签只在训练阶段使用；验证与推理禁止读取标注。

## 损失与选择

先将非负教师显著性归一化为概率质量，再在对数概率上施加温度，避免约 `1e-5` 的原始量级
经 softmax 后退化为均匀分布；该定义对显著性整体缩放不变。评分分布做 KL 蒸馏，并增加
3×3 区域概率质量一致性项。推理时沿用确定性
区域配额和稳定 top-k，输出排序后的 2×2 group index，不改变后续矩阵核接口。

## 门控

```text
单样本梯度链正确且有限
→ 教师标签可重复
→ 开发集训练/消融
→ 推理前冻结权重与新盲集 hash
→ 同预算超过 L2，且相对 full 四项下降均 ≤2pp
→ 才进入 HLS/RTL
```

必须消融：无文本、无坐标、无区域损失、纯 L2、纯残差、参数量/位宽。任何盲验证失败均保留
为负结果，不在同一验证集上重新调参。
