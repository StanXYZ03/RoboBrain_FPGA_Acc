# Layer-5 score-formula replay screening100（2026-10-04-layer5-score-formula-replay-screening100）

- **决定**：`L2_AGGREGATION_DIFFERENCE_FULLY_EXPLAINS_INDEX_SET`
- **判据**：100/100 exact candidate/reference matches => L2_AGGREGATION_DIFFERENCE_FULLY_EXPLAINS_INDEX_SET; otherwise PARTIAL_OR_NOT_EXPLAINED
- **样本**：100/100
- **精确匹配（candidate==reference）**：100/100
- **不匹配 sample_id**：无

## 分数公式（唯一变量）

- control：`norm(mean(x_0, x_1, x_2, x_3))`
- candidate：`mean(norm(x_0), norm(x_1), norm(x_2), norm(x_3))`

## Jaccard（candidate/control vs 冻结参考）

| 对比 | exact | mean | min |
| --- | --- | --- | --- |
| candidate vs reference | 100/100 | 1.000000 | 1.000000 |
| control vs reference | 0/100 | 0.603351 | 0.333333 |

## G / K 分布

- G：{'64': 2, '108': 2, '144': 3, '160': 2, '176': 1, '180': 7, '192': 3, '234': 7, '252': 2, '260': 6, '280': 1, '300': 3, '320': 1, '361': 1, '391': 11, '414': 1, '768': 22, '1200': 1, '1400': 20, '2684': 1, '3072': 1, '3588': 2}
- K_C：{'16': 2, '27': 2, '36': 3, '40': 2, '44': 1, '45': 7, '48': 3, '59': 7, '63': 2, '65': 6, '70': 1, '75': 3, '80': 1, '91': 1, '98': 11, '104': 1, '192': 22, '300': 1, '350': 20, '671': 1, '768': 1, '897': 2}

## 计时（仅描述，不参与决定）

- vision_total_seconds：{'count': 100, 'mean': 0.09669960609753617, 'min': 0.010890451027080417, 'max': 3.2951192050240934}
- model_inference_seconds：{'count': 100, 'mean': 0.09733938813093118, 'min': 0.011104395030997694, 'max': 3.2970478780334815}
- peak_cuda_bytes：{'count': 100, 'mean': 9013682964.48, 'min': 8898853888.0, 'max': 9517211648.0}

## 范围与限制

- 固定 100 样本 screening100 因果诊断（layer-5 cut=5、region_grid=3、每样本 K_C 取自冻结参考、仅改变每组分聚合公式）。不是正式全量 2638 结论，不是 FPGA 可部署配置；K_C 对齐放宽仅为诊断设置。
- 本诊断不是正式全量 2638 结论，也不是 FPGA 可部署配置。
- 计时仅描述，绝不参与决定。

## 输入 SHA256

- manifest：`31bf313286c81fa50911ca3822118184ac03ca74df2cf2c260250cab2768b50c`
- reference (`shallow_r0250`)：`ce019edd1ee35d11c83a3ff7c11bd9db46539cf56745aadcf0780f15abdbf7d0`
- control：`838cbb847c957412b73010b4c54999d1cad486a83c6e54592a1c42d35420d989`
