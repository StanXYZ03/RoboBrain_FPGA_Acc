# 无训练文本条件选择器止损实验

日期：2026-08-25  
状态：`COMPATIBILITY_PASS / SMALL_PILOT_FAIL / STOP_UNTRAINED_COSINE`

## 方法

在视觉 layer-5 使用模型已有 DeepStack merger 将每个 2×2 group 投影到 LLM hidden space，
与当前 Prompt 的非图像 token embedding 计算最大余弦相关性，再与纯局部残差融合并执行 3×3 区域配额。
Pointing/Grounding 共用相同算法和参数；100% 保留路径与 full 输出一致。

## 75% Token 小样本结果

每项任务仅 8 条，只用于快速止损。

| 配置 | PCK@0.1 | PCK@0.2 | Mean IoU | Acc@0.5 |
|---|---:|---:|---:|---:|
| L2+anchor | 0.500 | 0.875 | 0.7590 | 1.000 |
| pure residual | 0.500 | 0.875 | 0.7684 | 0.875 |
| text weight 0.10 | 0.375 | 0.875 | 0.7547 | 0.875 |
| text weight 0.25 | 0.500 | 0.750 | 0.6976 | 0.750 |

## 结论

原始词嵌入与浅层 DeepStack 投影虽维度一致，但没有证据表明余弦距离直接表达目标区域相关性。
文本权重 0.10 已无一致收益，0.25 出现严重坐标偏移，因此不做全量扫描、不进入 PL。
下一方案改为坐标损失/教师显著性监督的轻量评分头；推理时仍只输出规则块描述符。

原始结果：

- `evaluation/results/text_residual_pilot_w0100`
- `evaluation/results/text_residual_pilot_w0250`
- `evaluation/results/text_residual_pilot`
