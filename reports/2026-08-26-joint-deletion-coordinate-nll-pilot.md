# 联合删除坐标NLL：最小可行性门控

日期：2026-08-26  
状态：`PILOT_PASS / REPEATABILITY_PASS / NOT_AN_ACCURACY_CLAIM`

## 目的

此前单group的 `activation×gradient` 教师和L2残差头没有在独立盲集上证明优于简单L2。
本阶段不再拟合独立group重要性，而是保持相同Token预算和粗区域配额，直接测量多个group
联合交换后的teacher-forced坐标token NLL，为集合级排序提供监督。

## 实现

- 在layer-5浅层裁剪入口增加固定group索引模式。
- 固定索引经过整数类型、数量、唯一性、边界和Token预算校验。
- 从L2+anchor@87.5%生成1/2/4个group的区域内交换候选，每个规模2个。
- 所有候选保持168/192个视觉group及相同3×3粗区域计数。
- 集合级pairwise loss要求坐标NLL较低的整个保留集合获得更高分数。
- 提取脚本显式拒绝文件名含 `blind` 或 `validation` 的已揭盲清单。
- 原始结果为append-only JSONL；相同命令续跑时跳过已完成sample。

## 门控配置

| 项目 | 值 |
|---|---|
| 模型 | RoboBrain2.5-4B BF16，本地权重 |
| GPU | NVIDIA RTX A6000 |
| 数据 | `dac_spatial_selector_calibration.jsonl` |
| 样本 | 同一开发图片的Grounding与Pointing各1条 |
| Token预算 | 87.5%，168/192 group |
| 候选 | 每条1个L2基线 + 6个联合交换，共7个 |
| 总前向 | 14 |
| 目标 | 真实标注坐标的teacher-forced token NLL |

## 结果

| 任务 | L2基线NLL | 最佳候选 | 最佳NLL | 相对L2 |
|---|---:|---|---:|---:|
| Grounding | 0.663793 | `swap_004_00` | 0.659153 | -0.004640 |
| Pointing | 1.415355 | `swap_002_00` | 1.363374 | -0.051981 |

12个非基线候选中有4个低于L2基线，变化范围为 `[-0.051981, +0.016530]`。
这说明联合掩码产生了可测、可排序且任务相关的坐标损失差异，允许继续构建集合级训练数据。

独立重复运行中，14个掩码及其SHA-256逐一一致，14个NLL逐一完全相同，
`max_abs_nll_repeat_delta=0`。

## 结论与限制

最小实现和重复性门控通过，但当前只有1张图片、2个任务，不能声称任务精度提高，
也不能据此选择硬件配置。候选计时受目标token长度影响，未按正式warmup/repeats协议采集，
因此不作为性能结果。

下一门槛是在未揭盲开发集上先扩展到每类8条，检查候选生成成功率、NLL排序多样性和任务平衡；
通过后再覆盖固定64/32图训练-校准拆分，训练L2安全锚定的集合级排序头。任何真实输出精度门
通过之前，联合删除方法不得进入PL。

## 复现产物

- 实现：`evaluation/joint_deletion.py`
- 提取入口：`evaluation/extract_joint_deletion_teacher.py`
- 原始结果：`evaluation/results/joint_deletion_pilot/coordinate_nll.jsonl`
- 机器汇总：`evaluation/results/joint_deletion_pilot/summary.json`
- 独立重复：`evaluation/results/joint_deletion_pilot_repeat/coordinate_nll.jsonl`
- 单元测试：`evaluation/tests/test_learned_selector.py`

SHA-256：

- 原始结果：`90a8731f26752652345c5240a916b68a49f31cb2dcf05fa04380ba1ac755ba1a`
- 机器汇总：`f708fc2e46cf6281a4b873e76a5864fcfd1a6ecf0683cdccea6500586aeb5595`

