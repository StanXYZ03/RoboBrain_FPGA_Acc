# ZCU102 Q8×Q8 大矩阵首次上板基准

日期：2026-08-23  
状态：`LARGE_MATRIX_BOARD_BENCH_PASS / NUMERIC_SAMPLE_PASS`

## 1. 结论

当前 `Q8_0 × Q8_0` DATAFLOW HLS 核已在真实 ZCU102 上完成三种代表
Shape 的重复测试，所有抽样输出 `max_abs=0`，无超时、AXI 错误或PL失锁。

| 工作负载 | M×N×K | 板上均值 | 有效吞吐 | HLS排程 | 实测/排程 |
|---|---:|---:|---:|---:|---:|
| 语言视觉Prefill | 512×9728×2560 | 2.304381 s | 11.066 GOPS | 0.480150 s | 4.799×慢 |
| Vision Encoder | 4320×4096×1024 | 2.501725 s | 14.486 GOPS | 0.793411 s | 3.153×慢 |
| Decoder GEMV | 1×9728×2560 | 56.180 ms | 0.887 GOPS | 15.005 ms | 3.744×慢 |

相对现有PS算子基线：

| 工作负载 | PS参考 | PL实测 | PL/PS |
|---|---:|---:|---:|
| 512×9728×2560 | 3.034 s/次（140次累计均值） | 2.304 s | **1.317×加速** |
| 4320×4096×1024 | 4.279 s/次（24次累计均值） | 2.502 s | **1.710×加速** |
| 1×9728×2560 | 约6.164 ms（由8.08 GFLOP/s推算） | 56.180 ms | **0.110×，约9.11倍更慢** |

前两项PS值来自同一板卡的算子级插桩；Decoder值不是该Shape的独立计时，
只用于方向判断。PL计时不包含激活量化，尚不能视为llama.cpp端到端收益。

## 2. 测试方法

- 位流：此前通过C testbench、RTL co-sim及200 MHz布局布线的
  `q8q8_dataflow`。
- 内存：UIO map1，`0xB7C000000`起始的64 MiB保留DDR。
- 数据：权重和激活均为每32个INT8配一个FP16 scale；本基准scale固定为1。
- 正确性：每个分块抽查首行、中间行、末行及多列，与逐元素标量Golden比较。
- 重复：每个工作负载3次；权重只准备一次。
- `4320×4096×1024`的完整FP32输出约67.5 MiB，超过保留区，因此按
  `8×512 + 224`行执行并累加内核时间。
- 所有大矩阵测试结束后再次运行`M=2,N=3,K=32`自检，结果仍为
  `max_abs=0`，板卡保持空闲UART shell。

## 3. 数据流量与瓶颈证据

当前HLS循环顺序是`m_tile → n_tile → k_block`。每个M tile都会重新从
DDR读取对应的全部权重，当前核没有跨M tile缓存权重。根据源码实际请求的
量化面、scale面和输出面计算：

| 工作负载 | 逻辑AXI请求量 | 实测有效流量 | 算术强度 |
|---|---:|---:|---:|
| 语言Prefill | 1.078 GB | 0.468 GB/s | 23.649 OP/B |
| Vision Encoder | 1.575 GB | 0.629 GB/s | 23.011 OP/B |
| Decoder GEMV | 26.913 MB | 0.479 GB/s | 1.851 OP/B |

这里的“有效流量”是源码级有效字节数除以内核时间，不是AXI总线分析仪测得
的物理beat带宽；它不包含协议开销、重试或无效传输。相较此前PS侧顺序读
带宽`2.158 GB/s`，当前只得到约22%至29%的有效字节率，同时比理想HLS
排程慢3.15至4.80倍，说明主要优化对象是AXI事务效率、数据复用和流水停顿，
而不是继续孤立增加INT8乘法器。

权重重复读取尤其严重：

- 语言Prefill的25 MB唯一权重被32个M tile重复请求，权重量化面请求约
  797 MB。
- Vision Encoder的4 MB唯一权重被270个M tile重复请求，权重量化面请求约
  1.132 GB。

这为下一版采用`N tile`外层、片上weight-stationary缓存和ping-pong DMA
提供了直接实测依据。精确区分`AR/AW stall`、DDR等待和计算空泡仍需在下一
版位流加入AXI Performance Monitor或内核周期/停顿计数器。

## 4. 下一步设计门槛

1. 将一个`64×K` Q8_0权重tile保留在BRAM/URAM，跨全部M tile复用。
2. 为权重、激活和输出分别记录请求字节、有效周期和等待周期。
3. 保留当前实现作为baseline，用完全相同三种Shape比较GOPS与数值结果。
4. 大矩阵实际吞吐稳定提高后，再接入llama.cpp算子；Decoder另做专用GEMV
   调度，不能沿用当前16行GEMM阵列。

## 5. 可复现产物

- 基准源码：`fpga_accel/q8q8_runtime/src/q8q8_benchmark.c`
- DMA分配器：`fpga_accel/q8q8_runtime/src/q8q8_arena.c`
- ARM64二进制SHA-256：
  `3d236ab5dfea1ec2a953384bdfcde82ee98b4414dcf672097da780d32b51e035`
- UART原始日志：
  `reports/2026-08-23-zcu102-q8q8-large-matrix-benchmark-uart.log`
