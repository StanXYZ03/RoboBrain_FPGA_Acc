# 任务路由候选交互头：冻结新测试

日期：2026-08-27  
状态：`UNTOUCHED_TEST_COMPLETE / ROUTING_GATE_FAIL / RETAIN_L2`

## 冻结协议

- 策略锁在推理前写入：
  `reports/2026-08-27-task-routing-test.lock.json`
- 新测试集包含96张COCO val2017图片、192条Grounding/Pointing记录。
- 与smoke、开发、校准、旧validation、旧blind共320张既有图片的交集均为0。
- 冻结路由：Grounding使用 `full_task_lambda_0p5` 交互头，Pointing回退L2。
- 两端保留率均为87.5%，相同layer-5剪枝和3×3区域约束。
- 对照L2与路由策略在同一时段、同一GPU状态和同一代码版本下分别运行。

## 真实标注结果

| 配置 | PCK@0.1 | PCK@0.2 | Mean IoU | Acc@0.5 |
|---|---:|---:|---:|---:|
| L2@87.5 | 0.5417 | 0.7500 | 0.7442 | 0.8229 |
| 冻结任务路由@87.5 | 0.5417 | 0.7500 | 0.7343 | 0.8125 |

Pointing路由实现核验：

- 输出文本逐条一致：96/96；
- 选择掩码逐条一致：96/96；
- 实际选择器计数：96条 `l2_anchor`、96条
  `candidate_interaction_quota`。

Grounding交互头在96条中改变了54条输出，没有选择baseline候选。

## 相对L2的逐样本差值

每项为均值差和10,000次paired bootstrap的95%置信区间。

| PCK@0.1 | PCK@0.2 | Mean IoU | Acc@0.5 |
|---:|---:|---:|---:|
| 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | -0.0099 [-0.0291, +0.0025] | -0.0104 [-0.0312, 0.0000] |

## 门控

冻结门槛要求：

1. 解析率不得下降；
2. 任一主指标点估计下降不得超过0.02；
3. 至少一个主指标严格提高；
4. bootstrap区间如实报告，不用于揭示后调参。

解析率和点估计最大下降条件通过，但没有任何主指标提高；Mean IoU置信区间下界
也低于-0.02。因此总体门控失败。

## 最终决策

- 停止学习式联合删除、候选交互头及基于当前结果的后验fallback/阈值搜索。
- 不在旧blind、新测试集或其输出上继续调参。
- 不将113参数逐组头或545参数候选头接入PL。
- 保留layer-5 L2@87.5%作为当前软件和硬件协同的冻结视觉token基线。
- 后续研究转回可验证的PL算子/数据流加速和L2掩码下的真实端到端集成。

该结论只否定当前特征、候选生成和小头结构，不能推广为“学习式token选择普遍无效”。

## 证据

- 机器汇总：`reports/2026-08-27-task-routing-test.json`
- L2原始输出：
  `evaluation/results/task_routing_test/l2/shallow_r0875.jsonl`
- 路由原始输出：
  `evaluation/results/task_routing_test/routed/routed_interaction_full_task_lambda_0p5_r0875.jsonl`
- 新测试清单：`datasets/dac_spatial_task_routing_test.jsonl`
