# Q8 87.5%视觉Token裁剪扩大诊断与失效归因

状态：`EXPANDED_DIAGNOSTIC_COMPLETE / ROOT_CAUSE_PARTIALLY_ISOLATED / BF16_SAME_MASK_PENDING`。

## 证据边界

- 数据为COCO 2017 validation中的96张图、30个类别；每张图各形成Grounding和Pointing任务，共192条任务。
- 该数据已用于此前L2研究，不是新盲测；几何Oracle读取真实目标框，不能部署。本报告只用于诊断机制。
- Q8-B完整路径共运行stock和87.5%两路，384条推理全部成功；Q8-A在同一完整集运行192条87.5%推理，并复用相同stock记录。
- Bootstrap为配对重采样，固定种子，10,000次重复；不能用某一张图是否失败替代总体判断。

## 扩大结果

| 路径 | 指标 | stock | 87.5%裁剪 | 差值 | 95%配对Bootstrap区间 | 判断 |
|---|---|---:|---:|---:|---:|---|
| Q8-B，视觉块1后裁剪 | Pointing PCK@0.1 | 0.6458 | 0.5104 | -0.1354 | [-0.2190, -0.0521] | 相对5个百分点非劣界限确认退化 |
| Q8-B，视觉块1后裁剪 | Pointing PCK@0.2 | 0.8333 | 0.8229 | -0.0104 | [-0.0521, +0.0312] | 不确定，不能称为退化 |
| Q8-B，视觉块1后裁剪 | Grounding Mean IoU | 0.7379 | 0.7227 | -0.0152 | [-0.0444, +0.0185] | 不确定 |
| Q8-B，视觉块1后裁剪 | Grounding Acc@0.5 | 0.8646 | 0.8750 | +0.0104 | [-0.0417, +0.0625] | 不确定 |
| Q8-A，Merger后裁剪 | Pointing PCK@0.1 | 0.6458 | 0.5521 | -0.0938 | [-0.1667, -0.0208] | 相对零显著下降，非劣判定仍不确定 |
| Q8-A，Merger后裁剪 | Grounding Mean IoU | 0.7379 | 0.6868 | -0.0510 | [-0.0755, -0.0269] | 相对零显著下降，非劣判定仍不确定 |

Q8-B的结论只针对严格Pointing指标。此前四图结果中的PCK@0.2单样本失败没有在扩大集上形成确认退化，不能继续用“四图失败”概括全部精度。

## 目标覆盖审计

- 目标中心网格保留：192/192。
- 与目标框相交网格保留：8510/8566，比例99.346%。
- 目标相交网格全部保留：188/192条任务。
- Q8-B的16个PCK@0.1退化样本中，15个目标相交网格被全部保留。
- Q8-A的11个PCK@0.1退化样本全部属于目标相交网格完整保留组。

这排除了“主要因为Oracle直接删掉目标区域”的解释。4条覆盖不完整任务来自2张大目标图，不能解释总体严格定位下降。

## 当前归因

1. **Token集合或共享压紧路径是主要嫌疑。** Q8-A在完整视觉编码后才删除同一组Token，仍出现Pointing和Grounding下降，因此退化不依赖早期23层视觉计算被跳过。
2. **早期裁剪不是统一主因。** 完整集逐样本规则中，48条支持Token集合或共享压紧影响，15条支持早期裁剪影响；但Q8-B相对Q8-A的PCK@0.1差值为-0.0417，95%区间[-0.1042,+0.0208]，不足以证明早期裁剪普遍更差。Q8-B的Grounding Mean IoU反而比Q8-A高0.0358，区间[+0.0122,+0.0638]。
3. **上下文删除比目标删除更符合现有证据。** 几何Oracle只知道真实框和空间锚点，不知道模型依赖的关系对象、边界参照和全局布局。Pointing误差增量在小目标约+0.0430、中目标约+0.0273、大目标接近0；边缘目标约+0.0386、中心目标约+0.0059。这是关联证据，不是因果证明。
4. **未发现明显位置重编号错误。** Q8视觉侧从原始patch索引提取紧凑位置；LLM MRoPE由保留Token的原始二维索引计算。100% identity逐Token一致，稀疏路径在96种图像上均成功运行。该证据降低了位置实现错误的可能性，但不能替代跨后端数值对照。

当前最合理判断是：**87.5%裁剪对粗粒度Grounding和PCK@0.2影响有限，但当前bbox-only几何选择会破坏严格Pointing所需的上下文和坐标校准；量化与Q8后端是否放大该效应尚未完全排除。**

## 尚缺的决定性实验

在BF16 Transformers路径上使用完全相同的96图、同一87.5%索引，分别测试Merger后和视觉块1后裁剪：

- 若BF16复现相同退化，主因支持裁剪策略/上下文缺失。
- 若BF16不退化而Q8退化，主因转向Q8量化交互或自定义后端实现。
- 若只有视觉块1后退化，主因支持早期视觉表征传播；若Merger后也退化，说明Token集合本身已不足。

2026-09-11检查时，GPU由用户`seey6`的训练进程占用并保持高利用率，因此未并发启动本实验，也未干预该进程。待GPU空闲后执行：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/zhaoxiangyu/conda-envs/robobrain25/bin/python evaluation/run_hf_cutpoint_oracle.py \
  --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B \
  --model-label RoboBrain2.5-4B-BF16-expanded-diagnostic \
  --manifest datasets/dac_spatial_l2res_blind.jsonl \
  --output-dir evaluation/results/hf_oracle0875_expanded96_20260911 \
  --limit-per-task 96 --cut-points vision_block_1 post_merger \
  --retentions 1.0 0.875 --allow-revealed-diagnostic \
  --expected-masks evaluation/results/q8_geometric_oracle_masks_r0875_expanded96_20260911.jsonl
```

运行前必须重新检查GPU进程归属和剩余显存。该实验是精度归因，不把并发环境下的时间作为性能结果。

完成后生成配对报告：

```bash
/home/zhaoxiangyu/conda-envs/robobrain25/bin/python evaluation/build_hf_same_mask_diagnostic_report.py \
  --manifest datasets/dac_spatial_l2res_blind.jsonl \
  --results-dir evaluation/results/hf_oracle0875_expanded96_20260911 \
  --json reports/2026-09-11-hf-same-mask-expanded-diagnostic.json \
  --markdown reports/2026-09-11-hf-same-mask-expanded-diagnostic.md
```
