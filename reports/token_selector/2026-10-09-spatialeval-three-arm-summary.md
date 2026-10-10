# SpatialEval 三臂外部确认结果

- 状态：`EXTERNAL_CONFIRMATION_GATES_FAIL`
- 主集：Spatial-Map、Maze-Nav、Spatial-Grid，共 4500 条。
- Spatial-Real 135 条不进入主门槛，因为官方仓库没有有效解析器。
- 下表宏平均是本项目预注册汇总，不是 SpatialEval 官方总分。

| 臂 | 官方兼容宏平均 | 严格相等宏平均 |
| --- | ---: | ---: |
| dense_edge576 | 0.583778 | 0.578889 |
| corrected_l2_integer_b1_k96 | 0.439111 | 0.438000 |
| v2_edge576_b1_k96 | 0.434444 | 0.433333 |

- v2/Dense 宏平均保留率：`0.7441948991244766`
- v2-L2：`-0.004667`，95% CI `[-0.013778, 0.004444]`。
- 两项主门全部通过：`False`。
