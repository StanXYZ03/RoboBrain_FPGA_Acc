# 全局与ROI表示失败归因

状态：`GLOBAL_ROI_FAILURE_ATTRIBUTION_COMPLETE`。范围为32张已揭示开发图，不是盲测。

真值框经扩张后按构造完整包含目标，坐标回映射单元测试通过，全部候选格式解析率见下表。因此当前损失不能归因为简单的漏选目标或解析失败；仍可能来自裁剪后的视觉分布变化、上下文缺失、多图干扰和分辨率/边距选择。

| 候选 | 任务 | n | 主指标差 | 连续误差变化 | 平均裁剪面积 | 触边率 |
|---|---|---:|---:|---:|---:|---:|
| roi_only | grounding | 32 | -0.0419 | +0.0419 | 0.468 | 0.938 |
| roi_only | pointing | 32 | -0.0312 | +0.0066 | 0.468 | 0.938 |
| roi_global_v2 | grounding | 32 | -0.0342 | +0.0342 | 0.468 | 0.938 |
| roi_global_v2 | pointing | 32 | -0.1250 | +0.0040 | 0.468 | 0.938 |

## 分层差值

### roi_only / grounding

- `target_area_bin`：large: n=8, Δ=-0.0872, medium: n=14, Δ=-0.0024, small: n=10, Δ=-0.0607。
- `edge_bin`：edge: n=25, Δ=-0.0192, interior: n=7, Δ=-0.1228。
- `overlap_bin`：clear: n=15, Δ=-0.0081, overlap: n=17, Δ=-0.0716。
- `scene_complexity_bin`：dense: n=5, Δ=+0.0694, medium: n=10, Δ=-0.0568, sparse: n=17, Δ=-0.0658。
- `crop_touches_image_edge`：false: n=2, Δ=-0.0128, true: n=30, Δ=-0.0438。

### roi_only / pointing

- `target_area_bin`：large: n=8, Δ=+0.0000, medium: n=14, Δ=+0.0000, small: n=10, Δ=-0.1000。
- `edge_bin`：edge: n=25, Δ=-0.0400, interior: n=7, Δ=+0.0000。
- `overlap_bin`：clear: n=15, Δ=+0.1333, overlap: n=17, Δ=-0.1765。
- `scene_complexity_bin`：dense: n=5, Δ=-0.2000, medium: n=10, Δ=-0.1000, sparse: n=17, Δ=+0.0588。
- `crop_touches_image_edge`：false: n=2, Δ=+0.0000, true: n=30, Δ=-0.0333。

### roi_global_v2 / grounding

- `target_area_bin`：large: n=8, Δ=-0.0418, medium: n=14, Δ=+0.0095, small: n=10, Δ=-0.0893。
- `edge_bin`：edge: n=25, Δ=-0.0021, interior: n=7, Δ=-0.1490。
- `overlap_bin`：clear: n=15, Δ=-0.0133, overlap: n=17, Δ=-0.0527。
- `scene_complexity_bin`：dense: n=5, Δ=+0.1047, medium: n=10, Δ=-0.0827, sparse: n=17, Δ=-0.0465。
- `crop_touches_image_edge`：false: n=2, Δ=-0.0913, true: n=30, Δ=-0.0304。

### roi_global_v2 / pointing

- `target_area_bin`：large: n=8, Δ=-0.2500, medium: n=14, Δ=+0.0000, small: n=10, Δ=-0.2000。
- `edge_bin`：edge: n=25, Δ=-0.2000, interior: n=7, Δ=+0.1429。
- `overlap_bin`：clear: n=15, Δ=+0.0000, overlap: n=17, Δ=-0.2353。
- `scene_complexity_bin`：dense: n=5, Δ=-0.2000, medium: n=10, Δ=-0.2000, sparse: n=17, Δ=-0.0588。
- `crop_touches_image_edge`：false: n=2, Δ=+0.0000, true: n=30, Δ=-0.1333。

## 当前归因边界

- ROI-only用于判断局部表示本身；ROI-first加全局图用于判断上下文是否恢复。二者差值若很小，不能把全部退化简单归因于缺少全局图。
- 真值ROI仍读取标注，不反映可部署控制器的定位错误；下一步必须在开发集测试边距/尺度，再接入不读真值的轻量策略。
- 分层子组样本很小，只用于提出后续假设，不作显著性结论。
