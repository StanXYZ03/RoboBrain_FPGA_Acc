# 全局与ROI表示失败归因

状态：`GLOBAL_ROI_FAILURE_ATTRIBUTION_COMPLETE`。范围为32张已揭示开发图，不是盲测。

真值框经扩张后按构造完整包含目标，坐标回映射单元测试通过，全部候选格式解析率见下表。因此当前损失不能归因为简单的漏选目标或解析失败；仍可能来自裁剪后的视觉分布变化、上下文缺失、多图干扰和分辨率/边距选择。

| 候选 | 任务 | n | 主指标差 | 连续误差变化 | 平均裁剪面积 | 触边率 |
|---|---|---:|---:|---:|---:|---:|
| roi_only | grounding | 32 | -0.0122 | +0.0122 | 0.418 | 0.938 |
| roi_only | pointing | 32 | -0.0312 | -0.0373 | 0.418 | 0.938 |
| roi_global_v2 | grounding | 32 | -0.0123 | +0.0123 | 0.418 | 0.938 |
| roi_global_v2 | pointing | 32 | -0.1562 | -0.0103 | 0.418 | 0.938 |

## 分层差值

### roi_only / grounding

- `target_area_bin`：large: n=8, Δ=-0.0003, medium: n=10, Δ=-0.0336, small: n=14, Δ=-0.0036。
- `edge_bin`：edge: n=26, Δ=+0.0034, interior: n=6, Δ=-0.0795。
- `overlap_bin`：clear: n=14, Δ=+0.0206, overlap: n=18, Δ=-0.0376。
- `scene_complexity_bin`：dense: n=7, Δ=+0.0070, medium: n=9, Δ=-0.0775, sparse: n=16, Δ=+0.0162。
- `crop_touches_image_edge`：false: n=2, Δ=+0.0436, true: n=30, Δ=-0.0159。

### roi_only / pointing

- `target_area_bin`：large: n=8, Δ=+0.1250, medium: n=10, Δ=-0.1000, small: n=14, Δ=-0.0714。
- `edge_bin`：edge: n=26, Δ=-0.0385, interior: n=6, Δ=+0.0000。
- `overlap_bin`：clear: n=14, Δ=+0.0000, overlap: n=18, Δ=-0.0556。
- `scene_complexity_bin`：dense: n=7, Δ=+0.1429, medium: n=9, Δ=-0.1111, sparse: n=16, Δ=-0.0625。
- `crop_touches_image_edge`：false: n=2, Δ=+0.0000, true: n=30, Δ=-0.0333。

### roi_global_v2 / grounding

- `target_area_bin`：large: n=8, Δ=-0.0177, medium: n=10, Δ=-0.0562, small: n=14, Δ=+0.0221。
- `edge_bin`：edge: n=26, Δ=+0.0020, interior: n=6, Δ=-0.0743。
- `overlap_bin`：clear: n=14, Δ=+0.0207, overlap: n=18, Δ=-0.0380。
- `scene_complexity_bin`：dense: n=7, Δ=+0.0223, medium: n=9, Δ=-0.0810, sparse: n=16, Δ=+0.0112。
- `crop_touches_image_edge`：false: n=2, Δ=+0.0610, true: n=30, Δ=-0.0172。

### roi_global_v2 / pointing

- `target_area_bin`：large: n=8, Δ=+0.1250, medium: n=10, Δ=-0.3000, small: n=14, Δ=-0.2143。
- `edge_bin`：edge: n=26, Δ=-0.1923, interior: n=6, Δ=+0.0000。
- `overlap_bin`：clear: n=14, Δ=-0.0714, overlap: n=18, Δ=-0.2222。
- `scene_complexity_bin`：dense: n=7, Δ=-0.1429, medium: n=9, Δ=-0.2222, sparse: n=16, Δ=-0.1250。
- `crop_touches_image_edge`：false: n=2, Δ=+0.0000, true: n=30, Δ=-0.1667。

## 当前归因边界

- ROI-only用于判断局部表示本身；ROI-first加全局图用于判断上下文是否恢复。二者差值若很小，不能把全部退化简单归因于缺少全局图。
- 真值ROI仍读取标注，不反映可部署控制器的定位错误；下一步必须在开发集测试边距/尺度，再接入不读真值的轻量策略。
- 分层子组样本很小，只用于提出后续假设，不作显著性结论。
