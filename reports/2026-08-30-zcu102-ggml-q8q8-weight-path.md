# ZCU102 Q8_0权重数据路径实测

结论：`FAST_UIO_WEIGHT_STAGING_PASS / BYTE_EXACT_3_OF_3 / LARGE_M_STAGING_FEASIBLE / NATIVE_Q8_PL_FRONTEND_DEFERRED`。

| 权重shape | 原生Q8_0/MB | cache参考/ms | cache四核/ms | UIO参考/ms | UIO四核/ms | UIO加速 |
|---|---:|---:|---:|---:|---:|---:|
| llm_square_2560 | 6.96 | 38.240 | 2.461 | 58.624 | 3.207 | 18.28× |
| llm_ffn_9728 | 26.46 | 146.935 | 9.109 | 222.700 | 12.038 | 18.50× |
| vision_fc_4096 | 4.46 | 25.017 | 1.753 | 37.726 | 2.188 | 17.24× |

三种shape在cacheable DDR和真实UIO非缓存区均完成全量逐字节对照。基准只映射UIO数据区，
未访问PL控制寄存器，未启动计算核，未写SD/QSPI。

## 与已测kernel的组合口径

| workload | 权重stage/ms | 激活NEON四核/ms | PL kernel/ms | 已测准备组件/kernel |
|---|---:|---:|---:|---:|
| language_prefill_m512 | 12.038 | 2.701 | 174.545 | 8.44% |
| visual_prefill_m1080 | 12.038 | 5.474 | 368.889 | 4.75% |
| vision_encoder_m4320 | 2.188 | 8.644 | 252.071 | 4.30% |

- 对大M算子，快速权重stage与激活量化合计约为当前kernel的4.30–8.44%，可先用于完整block接入。
- 原生交织Q8_0 PL前端仍可能进一步降低带宽和PS占用，但不再是首个Transformer Block的前置门。
- 下一步必须把相同快速拆分用于每个activation chunk；当前runtime内部仍调用旧串行拆分。
- 上表尚未包含activation拆分/拷贝、输出拷贝、backend调度和GGML图开销，不能当端到端延迟。
- Decoder继续CPU回退，避免为M=1支付无收益的PL kernel与stage开销。

原始UART日志：`reports/20260830T124000Z-zcu102-ggml-weight-path-benchmark-uart.log`。
