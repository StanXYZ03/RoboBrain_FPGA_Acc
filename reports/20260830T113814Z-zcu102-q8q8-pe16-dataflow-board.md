# ZCU102 Q8×Q8 PE16 DATAFLOW 板端验证

日期：2026-08-30  
状态：`DATAFLOW_BOARD_PASS / NUMERIC_PASS / CHUNK_AB_PASS / APM_PASS`

## DATAFLOW 与已冻结 PE16 基线

| 同形状工作负载 | PE16基线 | DATAFLOW | 加速比 |
|---|---:|---:|---:|
| language_prefill | 242.808 ms | 174.545 ms | 1.391× |
| vision_encoder | 448.839 ms | 272.232 ms | 1.649× |
| decoder_gemv | 14.647 ms | 14.670 ms | 0.998× |

## DMA分块A/B

| 完整工作负载 | 固定512 | 自动分块 | 加速比 |
|---|---:|---:|---:|
| visual_prefill | 380.711 ms | 368.889 ms | 1.032× |
| vision_encoder | 272.232 ms | 252.071 ms | 1.080× |
| decoder_gemv | 14.670 ms | 14.669 ms | 1.000× |

自动分块只改变单次调用的M行数；当前PL仍在每次调用重新读取权重，因此这里不把收益
解释为跨调用权重常驻。所有阶段前后均保持`ap_idle`，协议前后自检及全部数值抽检
`max_abs=0`。

## 自动分块APM

| 工作负载 | 每次流量 | 实效带宽 | OP/B | CCI1/CCI2/HP0平均读延迟(cycle) |
|---|---:|---:|---:|---:|
| visual_prefill | 541.492 MB | 1.468 GB/s | 99.340 | 96.71 / 92.60 / 63.79 |
| vision_encoder | 380.507 MB | 1.510 GB/s | 95.238 | 87.88 / 84.54 / 58.39 |
| decoder_gemv | 26.915 MB | 1.835 GB/s | 1.851 | 96.02 / 90.15 / 61.21 |

平均延迟按“累计延迟÷累计事务数”计算；两个profile均覆盖相同3次重复和相同chunk，
不能先把事务数除以3，否则结果会被错误放大3倍。

## 边界

- 结果仅是矩阵核microbenchmark，不是完整RoboBrain端到端推理。
- 未测板级功耗和温度，不给出能效结论。
- UART：`reports/20260830T113814Z-zcu102-q8q8-pe16-dataflow-board-uart.log`。
