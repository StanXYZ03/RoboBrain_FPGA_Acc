# 32图前缀抽样门控作废记录

日期：2026-09-12。状态：`SUPERSEDED_PREFIX_SAMPLING_BIAS`。

`global_roi_bf16_representation_gate32_20260912.jsonl`虽完成768条确定性推理，但旧版`choose_samples()`按manifest文件顺序截取前32张图。该子集有25/32个边缘目标，30/32个真值扩张ROI触碰图像边界，不能代表完整192图开发集。因此其质量结果只保留为发现实现问题的诊断，不进入表示门控结论。

`global_roi_bf16_representation_gate32_timingv2_20260912.jsonl`在发现该问题后由本任务主动中止，是不完整运行，也不得聚合。

修正版按目标面积、边缘位置、bbox重叠代理和场景复杂度执行确定性比例分层，并保证同图Pointing/Grounding成对入选。正式替代结果将写入：

```text
evaluation/results/global_roi_bf16_representation_gate32_stratified_20260912.jsonl
reports/2026-09-12-global-roi-bf16-representation-gate32-stratified.md
```

旧文件不删除，以保留失败证据和审计链。
