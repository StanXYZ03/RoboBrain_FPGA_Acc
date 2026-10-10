# 候选交互头：开发集真实输出校准

日期：2026-08-27  
状态：`REAL_OUTPUT_CALIBRATION_FAIL / UNIFIED_HEAD_STOP`

## 协议

- 数据为固定64/32开发拆分中的32张校准图，共64条Grounding/Pointing记录。
- 选择器为开发坐标NLL门控选出的 `full_task_lambda_0p5`，运行前已固定。
- 交互头只能在L2基线及1/2/4组区域保持交换形成的7个候选中选择。
- 已揭示blind/validation数据未参与本次训练、选参或真实输出校准。
- 主指标门控要求：每项相对L2下降不超过0.02，且至少一项提高。

## 真实标注结果

| 配置 | PCK@0.1 | PCK@0.2 | Point NME | Mean IoU | Acc@0.5 |
|---|---:|---:|---:|---:|---:|
| L2@87.5 | 0.6250 | 0.8125 | 0.2436 | 0.7976 | 0.8438 |
| 交互头@87.5 | 0.5625 | 0.8125 | 0.2400 | 0.8032 | 0.8750 |

## 相对L2的逐样本差值

每项为均值差和10,000次paired bootstrap的95%置信区间，n=32/任务。

| PCK@0.1 | PCK@0.2 | Mean IoU | Acc@0.5 |
|---:|---:|---:|---:|
| -0.0625 [-0.1562, 0.0000] | 0.0000 [-0.0938, 0.0938] | +0.0056 [-0.0208, +0.0412] | +0.0312 [0.0000, +0.0938] |

## 门控决策

Grounding的Mean IoU提高0.0056，Acc@0.5提高0.0312；Pointing平均归一化误差
略有降低，但PCK@0.1下降0.0625，超过预注册的0.02容忍度。因此统一双任务
交互头门控失败，不能进入PL，也不能在已揭示盲集上继续尝试。

当前结果可用于提出一个新设计：Grounding使用交互头，Pointing回退L2。该路由
是观察本次结果后形成，不能用同一校准集证明有效；必须先冻结策略，再在与全部
既有图片零重叠的新测试集上比较同预算L2。

L2与交互头输出不是同一时段采集，延迟值只作各自描述，不用于声明加速。

## 证据

- 机器汇总：`reports/2026-08-27-candidate-interaction-calibration.json`
- L2原始输出：`evaluation/results/shallow_block_expanded/shallow_r0875.jsonl`
- 交互头原始输出：
  `evaluation/results/candidate_interaction_real_output/interaction_full_task_lambda_0p5_r0875.jsonl`
- 训练摘要：
  `evaluation/results/candidate_interaction_selector_development_summary.json`
