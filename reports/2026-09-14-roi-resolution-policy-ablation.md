# ROI分辨率策略特征消融

日期：2026-09-14。状态：`FIXED_POOL_LINEAR_POLICY_STOP`。

范围：192张已揭示开发图，Grounding与Pointing各192条；5折按COCO图像ID分组，折内标准化，固定岭系数10和零阈值。所有ROI均由真值框扩张得到，因此这里只评价“区域正确给定后的尺度选择”，不评价区域定位。

| 特征 | 任务 | 动作平衡准确率 | 收益相关系数 | 主指标差[95% CI] | 最终Token比 | Vision工作比 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| 全图64×64统计 | Grounding | 0.5405 | -0.0153 | -0.0070 [-0.0293,+0.0139] | 0.501 | 0.501 | FAIL |
| 全图64×64统计 | Pointing | 0.5676 | -0.0305 | -0.0312 [-0.0938,+0.0312] | 0.477 | 0.477 | FAIL |
| ROI 64×64统计 | Grounding | 0.4877 | 0.1454 | -0.0009 [-0.0239,+0.0211] | 0.505 | 0.505 | FAIL |
| ROI 64×64统计 | Pointing | 0.5506 | 0.1376 | -0.0104 [-0.0677,+0.0469] | 0.494 | 0.494 | FAIL |
| Vision Prefix 1层 | Grounding | 0.4554 | 0.0721 | +0.0061 [-0.0161,+0.0279] | 0.494 | 0.502 | FAIL |
| Vision Prefix 1层 | Pointing | 0.4566 | -0.0026 | -0.0260 [-0.0885,+0.0365] | 0.498 | 0.507 | FAIL |
| Vision Prefix 3层 | Grounding | 0.5287 | 0.0234 | -0.0017 [-0.0262,+0.0208] | 0.474 | 0.495 | FAIL |
| Vision Prefix 3层 | Pointing | 0.4574 | -0.0098 | -0.0521 [-0.1146,+0.0104] | 0.485 | 0.508 | FAIL |
| Vision Prefix 6层 | Grounding | 0.4954 | -0.0173 | -0.0012 [-0.0244,+0.0213] | 0.482 | 0.527 | FAIL |
| Vision Prefix 6层 | Pointing | 0.5266 | 0.0482 | -0.0260 [-0.0885,+0.0365] | 0.472 | 0.513 | FAIL |
| 6层+Patch-Text空间相关性 | Grounding | 0.4973 | -0.0127 | -0.0018 [-0.0238,+0.0193] | 0.486 | 0.532 | FAIL |
| 6层+Patch-Text空间相关性 | Pointing | 0.5262 | 0.0456 | -0.0156 [-0.0781,+0.0469] | 0.475 | 0.518 | FAIL |

门控要求三项同时满足：动作平衡准确率至少0.55、成本比不高于0.5、主指标相对Dense的配对95%置信下界不低于-0.02。没有一个特征配置在两项任务上同时通过。

Vision Prefix仅计时Patch Embedding、位置处理和指定Vision Block，不含ROI取数、processor、LLM Prefill或Decode。A6000中位/p95：1层2.804/3.083 ms，3层3.553/3.951 ms，6层4.638/5.308 ms；1层最大16.743 ms为运行抖动离群值。这些不是目标异构平台延迟。

## 决策

1. 按`configs/roi_resolution_vision_prefix_depth_ablation.lock.json`停止增加前缀深度或在同一开发集调岭系数、阈值。
2. 当前固定池化将空间Token压成全局统计，可能丢失“任务目标与局部细节是否匹配”的信息；这是由结果提出的设计假设，不是已证明根因。
3. 预注册的任务条件Patch-Text空间相关性候选也未通过，当前384/512尺度控制器分支正式停止。
4. 不访问校准/测试集；下一研究动作必须改变动作空间或监督目标，例如联合区域与尺度的多候选价值预测、可复用Token增量细化，或回到同预算固定ROI/L2对照。不能继续在同一动作矩阵上增加模型容量并把开发集结果当作新证据。

关键原始产物：

- `evaluation/results/roi384_vision_prefix{1,3,6}_features_dev192_20260914.jsonl`
- `reports/2026-09-14-roi-resolution-{cheap,crop}-policy-cv192.{json,md}`
- `reports/2026-09-14-roi-resolution-vision-prefix{1,3,6}-policy-cv192.{json,md}`
- `reports/2026-09-14-roi-resolution-task-relevance-policy-cv192.{json,md}`
- `evaluation/results/roi384_vision_prefix6_task_relevance_features_dev384_20260914.jsonl`

证据边界：本轮未访问校准集和封存测试集；失败不等于主动视觉细化普遍不可行，只否定当前特征压缩与线性预测合同。
