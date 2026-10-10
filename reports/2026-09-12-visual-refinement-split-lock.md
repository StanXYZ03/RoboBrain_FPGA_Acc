# 按需视觉细化数据集冻结验收

状态：`VISUAL_REFINEMENT_SPLIT_LOCK_PASS`。

| 划分 | 图像 | Pointing记录 | Grounding记录 | 图像字节 |
|---|---:|---:|---:|---:|
| development | 192 | 192 | 192 | 27541113 |
| calibration | 96 | 96 | 96 | 14832199 |
| test | 512 | 512 | 512 | 76480871 |

- 三个划分共 `800` 张唯一原始分辨率JPEG，目录实有 `800` 张。
- 划分间重叠：`0`；与历史 `432` 张图像重叠：`0`。
- 图像集合内容摘要：`29e5b8084a4fa86e92c9b81042a59cc10b835bea79f0ef1b1b7a63b42320c529`。
- 冻结文件摘要：`f052d10adcaa46e9274629493587a715edd9df90d890b62c1040d798b48bd646`。
- 每张图均完成PIL解码、实际尺寸与manifest尺寸核对，并核验Pointing/Grounding成对存在。

证据边界：该集合可用于Grounding和目标框中心Pointing；COCO框中心不是可供性点，框重叠也不是真实人工遮挡标注。测试集在策略冻结前不得用于调参。
