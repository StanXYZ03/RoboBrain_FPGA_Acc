# BF16同掩码扩大诊断

状态：已揭示诊断，不是新盲测。所有裁剪记录均通过与Q8固定掩码的逐索引一致性检查。

| 切入点 | 指标 | stock | 裁剪 | 差值 | 95%配对Bootstrap区间 | 相对零变化 |
|---|---|---:|---:|---:|---:|---|
| vision_block_1 | pointing_pck_0_1 | 0.5729 | 0.5208 | -0.0521 | [-0.1250, +0.0208] | UNCERTAIN |
| vision_block_1 | pointing_pck_0_2 | 0.7708 | 0.8021 | +0.0312 | [-0.0208, +0.0833] | UNCERTAIN |
| vision_block_1 | grounding_iou_mean | 0.7804 | 0.7413 | -0.0391 | [-0.0680, -0.0071] | DECREASE |
| vision_block_1 | grounding_acc_0_5 | 0.8854 | 0.8646 | -0.0208 | [-0.0729, +0.0312] | UNCERTAIN |
| post_merger | pointing_pck_0_1 | 0.5729 | 0.5000 | -0.0729 | [-0.1458, +0.0000] | UNCERTAIN |
| post_merger | pointing_pck_0_2 | 0.7708 | 0.7812 | +0.0104 | [-0.0208, +0.0521] | UNCERTAIN |
| post_merger | grounding_iou_mean | 0.7804 | 0.7341 | -0.0463 | [-0.0747, -0.0156] | DECREASE |
| post_merger | grounding_acc_0_5 | 0.8854 | 0.8854 | +0.0000 | [-0.0417, +0.0417] | UNCERTAIN |

解释规则：Merger后也退化支持Token集合/上下文缺失；仅视觉块1后退化支持早期视觉传播；BF16均不退化而Q8退化则应优先审查量化交互或Q8后端。
