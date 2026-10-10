# ZCU102 A53 NEON并行Q8_0激活量化

结论：`A53_NEON_PARALLEL_QUANTIZATION_PASS / BYTE_EXACT_6_OF_6 / ACTIVATION_QUANTIZATION_NO_LONGER_PRIMARY_PREP_BOTTLENECK`。

| case | 标量参考/ms | NEON单核/ms | NEON四核/ms | 单核加速 | 四核加速 | 四核量化/PL kernel |
|---|---:|---:|---:|---:|---:|---:|
| decoder_m1 | 0.061 | 0.019 | 0.019 | 3.28× | 3.26× | — |
| text_prefill_m49 | 2.999 | 0.891 | 0.345 | 3.37× | 8.69× | — |
| visual_tail_m56 | 3.423 | 1.015 | 0.372 | 3.37× | 9.19× | — |
| language_prefill_m512 | 31.758 | 9.815 | 2.701 | 3.24× | 11.76× | 1.55% |
| visual_prefill_m1080 | 66.970 | 20.729 | 5.474 | 3.23× | 12.23× | 1.48% |
| vision_encoder_m4320 | 107.172 | 33.119 | 8.644 | 3.24× | 12.40× | 3.43% |

## 数值门

- 六个真实尺度的全部Q8_0输出逐字节比较通过：标量参考 = NEON单核 = NEON四核。
- 另有上游`quantize_row_q8_0_ref`差分测试、half-way舍入和NaN回退测试通过。
- 板端静态二进制SHA256回读一致；只写RAM中的`/tmp`，未写SD/QSPI。

## 判断

- 单核NEON约为标量的3.2×；大矩阵四核约为标量的11.8–12.4×。
- M=512、M=1080、M=4320的四核量化仅约占同shape PL kernel的1.55%、1.48%、3.43%。
- 激活量化已不再是首要准备瓶颈；当前首要问题转为权重从原生交织Q8_0到PL双平面UIO格式的
  变换与搬运。
- 四线程策略必须由未来backend统一管理，避免与GGML线程池重复并行；生产入口仍默认关闭。

原始UART日志：`reports/20260830T123000Z-zcu102-ggml-neon-numeric-benchmark-uart.log`。
