# ZCU102 GGML→PL准备开销实测

结论：`PREPARE_COST_MEASURED / WEIGHT_REPACK_PER_CALL_REJECTED / ACTIVATION_QUANTIZATION_OPTIMIZATION_REQUIRED / DECODER_CPU_FALLBACK`。

| case | M | N | K | 激活FP32→Q8_0 | 权重拆分（cacheable DDR） | 同shape PL kernel | 量化/kernel |
|---|---:|---:|---:|---:|---:|---:|---:|
| decoder_m1 | 1 | 2560 | 2560 | 0.070 | 39.809 | — | — |
| text_prefill_m49 | 49 | 2560 | 2560 | 3.416 | 38.152 | — | — |
| visual_tail_m56 | 56 | 9728 | 2560 | 3.909 | 149.006 | — | — |
| language_prefill_m512 | 512 | 9728 | 2560 | 36.017 | 149.223 | 174.545 | 20.6% |
| visual_prefill_m1080 | 1080 | 9728 | 2560 | 76.038 | 147.152 | 368.889 | 20.6% |
| vision_encoder_m4320 | 4320 | 4096 | 1024 | 121.676 | 26.221 | 252.071 | 48.3% |

单位均为ms，三次取中位数。同shape对照只列入M/N/K完全一致的三项。板端程序经UART写入
`/tmp`并执行，SHA256回读一致；未写SD或QSPI。

## 关键判断

- `M=512`与`M=1080`的激活量化约占当前PL kernel的20.6%；`M=4320` Encoder约占48.3%。
- `N=9728,K=2560`权重在普通可缓存DDR中拆分仍需约147–149 ms；既有UIO实测约354 ms。
  不允许生产路径在每次算子调用时重复这一步。
- 需要比较两条路线：受控的预打包权重驻留/复用，或PL直接消费原生34-byte GGML Q8_0块。
- 当前decoder GEMV板端没有kernel收益，保持CPU回退，避免权重准备反向拖慢。
- 下一门是NEON/多线程激活量化与权重数据路径原型；未完成前不宣称端到端加速。

## 口径边界

本次新测量的目标是普通cacheable DDR；既有`average_prepare_us`还包含确定性激活生成和向UIO
保留区打包，二者不能直接相减或当成相同测量。全部结果仍是算子桥接开销，不是完整模型延迟。

原始UART日志：`reports/20260830T120500Z-zcu102-ggml-prepare-benchmark-uart.log`。
