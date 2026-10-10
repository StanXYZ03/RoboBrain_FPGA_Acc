# ZCU102 UIO输出回读实测

结论：`FAST_UIO_READBACK_PASS / BYTE_EXACT_3_OF_3 / PER_OP_PS_ROUNDTRIP_REMAINS_MAJOR / PL_RESIDENT_BLOCK_PIPELINE_REQUIRED`。

| case | 输出/MB | 逐字节参考/ms | NEON单核/ms | NEON四核/ms | 四核加速 | 四核GB/s | 单chunk回读/workload kernel |
|---|---:|---:|---:|---:|---:|---:|---:|
| language_m512 | 19.92 | 2308.914 | 152.729 | 48.957 | 47.16× | 0.407 | 28.05% |
| visual_chunk_m976 | 37.98 | 4401.855 | 293.422 | 83.899 | 52.47× | 0.453 | 22.74% |
| vision_chunk_m3584 | 58.72 | 6805.901 | 454.277 | 150.351 | 45.27× | 0.391 | 59.65% |

三种输出均完成全量逐字节一致性验证。基准只读取UIO数据区，没有访问PL控制寄存器、启动kernel
或写SD/QSPI。

## 设计结论

- NEON四核消除了旧逐字节实现的灾难性开销，但非缓存UIO回读仍只有约0.39–0.45 GB/s。
- M=512完整输出回读仍相当于同shape kernel的28.05%；视觉/Encoder表中只计最大chunk，
  完整workload还有尾chunk，不能把该比例误当完整回读总量。
- 若每个MUL_MAT都回到PS，输出回读会成为端到端主要开销。因此完整Block不能只是“逐算子RPC”。
- 下一架构门：中间张量驻留PL可访问DDR，融合RMSNorm/残差/SiLU或在PL端直接requantize，
  只在Block边界或确有CPU消费者时回读。

原始UART日志：`reports/20260830T125000Z-zcu102-ggml-output-path-benchmark-uart.log`。
