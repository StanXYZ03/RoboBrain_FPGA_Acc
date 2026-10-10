# ROI/L2同预算失效归因

状态：`ROI_L2_MATCHED_FAILURE_ATTRIBUTION_COMPLETE`。这是已揭示开发集上的事后描述分析，不是新假设检验。

## 目标覆盖审计

L2保留目标中心网格：`150/384`；完整保留全部目标相交网格：`10/384`；目标相交网格总体保留率：`0.367`。
同图Grounding/Pointing的L2选择集合一致：`True`。

## 关键分组

`Delta`为L2主指标减Dense主指标；分组只用于定位机制，不作显著性结论。

| 任务 | 分组 | n | 保留率 | 目标覆盖 | L2-Dense Delta | L2-ROI Delta |
|---|---|---:|---:|---:|---:|---:|
| grounding | retention_bin=[0.00,0.20) | 24 | 0.141 | 0.123 | -0.4547 | -0.3966 |
| grounding | retention_bin=[0.20,0.35) | 66 | 0.265 | 0.281 | -0.3640 | -0.3393 |
| grounding | retention_bin=[0.35,0.50) | 69 | 0.398 | 0.442 | -0.2125 | -0.1861 |
| grounding | retention_bin=[0.50,1.00] | 33 | 0.631 | 0.703 | -0.1142 | -0.1369 |
| grounding | target_center_cell_retained=False | 117 | 0.316 | 0.271 | -0.3541 | -0.3292 |
| grounding | target_center_cell_retained=True | 75 | 0.431 | 0.580 | -0.1592 | -0.1433 |
| grounding | coverage_bin=(0,0.5) | 123 | 0.295 | 0.256 | -0.3579 | -0.3257 |
| grounding | coverage_bin=0 | 5 | 0.180 | 0.000 | -0.8481 | -0.7264 |
| grounding | coverage_bin=1 | 5 | 0.675 | 1.000 | -0.0475 | -0.0433 |
| grounding | coverage_bin=[0.5,1) | 59 | 0.485 | 0.655 | -0.0825 | -0.0908 |
| pointing | retention_bin=[0.00,0.20) | 24 | 0.141 | 0.123 | -0.4167 | -0.3750 |
| pointing | retention_bin=[0.20,0.35) | 66 | 0.265 | 0.281 | -0.4091 | -0.2879 |
| pointing | retention_bin=[0.35,0.50) | 69 | 0.398 | 0.442 | -0.2609 | -0.2174 |
| pointing | retention_bin=[0.50,1.00] | 33 | 0.631 | 0.703 | -0.0909 | -0.1818 |
| pointing | target_center_cell_retained=False | 117 | 0.316 | 0.271 | -0.3675 | -0.3077 |
| pointing | target_center_cell_retained=True | 75 | 0.431 | 0.580 | -0.2000 | -0.1733 |
| pointing | coverage_bin=(0,0.5) | 123 | 0.295 | 0.256 | -0.3740 | -0.2846 |
| pointing | coverage_bin=0 | 5 | 0.180 | 0.000 | -0.6000 | -0.8000 |
| pointing | coverage_bin=1 | 5 | 0.675 | 1.000 | +0.0000 | +0.0000 |
| pointing | coverage_bin=[0.5,1) | 59 | 0.485 | 0.655 | -0.1525 | -0.1695 |

## 相关性

| 任务 | 保留率与L2-Dense | 目标覆盖与L2-Dense |
|---|---:|---:|
| grounding | 0.3215 | 0.5289 |
| pointing | 0.2441 | 0.3677 |

## 解释边界

如果保留目标中心或提高目标网格覆盖只能部分缩小损失，则失败不能简化为单个中心Token漏选；还可能涉及部件细节、连续邻域、全局参照和裁剪后重采样差异。当前分析只能形成下一动作空间的设计依据，不能证明因果。
