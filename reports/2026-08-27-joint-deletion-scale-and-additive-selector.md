# 联合删除规模实验与可加选择器门控

日期：2026-08-27  
结论：`SCALE_GATE_PASS / ADDITIVE_SELECTOR_STOP / REAL_OUTPUT_SKIPPED`

## 1. 实验边界

- 训练与选参仅使用 `dac_spatial_eval_expanded.jsonl` 的96张开发图。
- 每张图配对Grounding与Pointing，共192条任务记录。
- 固定64图训练、32图校准，拆分哈希为
  `49adb2dfa7859cade1f941674fecb8d2e39bc8683add46fd3523824e5485faba`。
- 已揭示的blind/validation清单未参与训练、门控或算法设计。
- 指标是教师强制的坐标token NLL，不等同于PCK、IoU或任务精度。

## 2. 候选协议

以layer-5 L2@87.5%为安全基线。对每个样本生成6个候选：

- 1组交换×2；
- 2组交换×2；
- 4组交换×2。

交换只在同一个3×3粗区域内发生，因此7个候选具有相同token预算和完全相同的
区域计数。候选随机种子由图片路径而非任务ID决定，同一图片的Grounding与
Pointing使用完全相同的候选掩码。

## 3. 规模门控

首轮8图/任务实验发现候选种子曾包含任务ID，导致同图双任务候选不可配对。该
版本保留为审计证据，不用于结论。修正后v2结果如下：

| 任务 | 非基线候选 | 优于L2数量 | 候选delta均值 | 每样本oracle delta均值 |
|---|---:|---:|---:|---:|
| Grounding | 48 | 20 | +0.002613 | -0.015385 |
| Pointing | 48 | 22 | +0.001772 | -0.034700 |

同图候选掩码8/8一致，但两任务最佳候选仅2/8一致，候选NLL排序的同图平均
Spearman相关系数为0.1563。该结果支持继续测试任务相关集合监督，但不能证明
精度改善。

完整开发集共完成192条记录、1344次候选前向：

| 任务 | 非基线候选 | 优于L2数量 | 候选delta均值 | 每样本oracle delta均值 | 存在更优候选样本 |
|---|---:|---:|---:|---:|---:|
| Grounding | 576 | 282 | +0.002135 | -0.038334 | 91/96 |
| Pointing | 576 | 319 | -0.005490 | -0.054767 | 93/96 |

结论是候选空间中普遍存在更低坐标NLL的掩码，规模门控通过。

## 4. 113参数可加选择器

训练头以标准化L2为固定主分数，只学习5维逐组特征的残差修正。部署选择被限制
在上述7个候选中，最多只偏离L2四组。共训练full与no-text两种特征消融、8个
正则强度，共16个候选头。

预注册开发门控要求同时满足：

1. 候选pairwise排序准确率不低于0.52；
2. 平均坐标NLL优于L2；
3. top-1 regret相对L2至少降低5%；
4. 与L2掩码的平均重叠率不低于0.95。

诊断最优为 `no_text_lambda_0`：

| 指标 | L2基线 | 诊断最优 |
|---|---:|---:|
| pairwise accuracy | 0.4707 | 0.4903 |
| candidate Spearman | -0.0675 | -0.0128 |
| mean top-1 regret | 0.038897 | 0.035146 |
| mean delta NLL vs L2 | 0 | -0.003751 |
| oracle hit rate | 0.0625 | 0.2656 |
| mean L2 overlap | 1.0000 | 0.9868 |
| p95 top-1 regret | 0.1175 | 0.1402 |

该头通过了平均NLL、regret和重叠率三项，但pairwise accuracy未达到0.52；
此外Grounding平均delta为+0.008188，且总体只有56.25%的样本不劣于L2。
no-text优于full也表明当前文本相关逐组特征没有形成可靠任务条件化收益。

## 5. 决策

- 停止“逐组打分后求集合和”的可加选择器路线。
- 不执行该头的GPU真实输出校准；这是预注册门控的结果，不是资源不足。
- 不将该权重接入PL，也不声称PCK、IoU或端到端精度提高。
- 下一步保留7候选安全边界，改为显式编码加入组与删除组交互的候选级小头。
- 候选级方法若再次失败，则停止学习式联合删除并保留L2@87.5%。

## 6. 证据

- 规模门控v2：
  `evaluation/results/joint_deletion_scale_gate_v2/coordinate_nll.jsonl`
- 完整教师：
  `evaluation/results/joint_deletion_development/coordinate_nll.jsonl`
- 训练摘要：
  `evaluation/results/joint_deletion_selector_development_summary.json`
- 诊断权重：
  `evaluation/results/joint_deletion_selector_development.pt`

关键SHA-256：

```text
scale gate v2 raw:
be8c53f354f11fee775881f19a91a321f7074fb47187627884d624c42f242f08
development teacher raw:
a16477a52dc9b29a3d81c5396249b9f79e22a4f0e21fa2dd6b83c1a9b29031d0
additive training summary:
ecd72f52be07a73f688077a486216199d017a19a921e57a5c167a0e3e8240721
additive diagnostic artifact:
34daba682f19a5c6099f4e8df8234a5246d826c7064ab4d37935440bf190a0b4
```
