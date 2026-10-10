# ROI分辨率动作互补性上界

状态：`ROI_RESOLUTION_REVEALED_ORACLE_COMPLETE`。本分析读取开发集真值逐图选择384或512，只是控制器潜力上界，不是可部署策略。

| 任务 | 512更好/相同/更差 | 策略 | 质量主指标 | 次指标 | 选512比例 | 最终Token/Dense | 累计Token/Dense |
|---|---:|---|---:|---:|---:|---:|---:|
| grounding | 23/0/9 | fixed_low_384 | IoU 0.7989 | Acc@0.5 1.0000 | 0.000 | 0.290 | 0.290 |
| grounding | 23/0/9 | fixed_high_512 | IoU 0.8330 | Acc@0.5 1.0000 | 1.000 | 0.516 | 0.516 |
| grounding | 23/0/9 | revealed_one_shot_oracle | IoU 0.8540 | Acc@0.5 1.0000 | 0.719 | 0.443 | 0.443 |
| grounding | 23/0/9 | revealed_active_escalation_oracle | IoU 0.8540 | Acc@0.5 1.0000 | 0.719 | 0.443 | 0.637 |
| pointing | 14/3/15 | fixed_low_384 | PCK@0.1 0.6875 | PCK@0.2 0.9375 | 0.000 | 0.290 | 0.290 |
| pointing | 14/3/15 | fixed_high_512 | PCK@0.1 0.6562 | PCK@0.2 0.8750 | 1.000 | 0.516 | 0.516 |
| pointing | 14/3/15 | revealed_one_shot_oracle | PCK@0.1 0.7188 | PCK@0.2 0.9375 | 0.438 | 0.389 | 0.389 |
| pointing | 14/3/15 | revealed_active_escalation_oracle | PCK@0.1 0.7188 | PCK@0.2 0.9375 | 0.438 | 0.389 | 0.520 |

## 解释边界

- `revealed_one_shot_oracle`假设编码前已知最佳尺度，不能部署，只回答两种动作是否互补。
- `revealed_active_escalation_oracle`先处理384，再对被选样本重编码512；质量相同，但累计Token计入重复视觉处理。
- 若Oracle明显优于两个固定尺度，只说明存在可学习信号的必要条件；仍需使用不读真值的特征训练控制器，并在校准集和封存测试集验证。
- 本报告不比较跨运行延迟，因为GPU负载条件不同。
