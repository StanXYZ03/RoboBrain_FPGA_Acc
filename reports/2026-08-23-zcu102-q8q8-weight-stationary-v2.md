# ZCU102 Q8×Q8 Weight-Stationary v2 阶段报告

日期：2026-08-23  
状态：`STAGE_PASS / BOARD_NUMERIC_PASS / >=2X_GATE_PASS`

## 1. 阶段结论

实现了 `Q8_0 × Q8_0` weight-stationary v2：一个 `64×K` 权重 tile
完整驻留 BRAM，并跨全部 `16×K` 的 M tile 复用；权重和激活按行连续突发。
在真实 ZCU102 上，语言 prefill 相对 v1 加速 **3.930×**，超过阶段目标
“至少一项代表矩阵达到 2×”的门槛；decoder 达到 **2.854×**，视觉 encoder
达到 **1.781×**。全部板端抽样结果 `max_abs=0`。

| 工作负载 | M×N×K | v1 板端 | v2 板端 | v2 吞吐 | v2/v1 |
|---|---:|---:|---:|---:|---:|
| Language prefill | 512×9728×2560 | 2.304381 s | 0.586333 s | 43.493 GOPS | **3.930×** |
| Vision encoder | 4320×4096×1024 | 2.501725 s | 1.404407 s | 25.804 GOPS | **1.781×** |
| Decoder GEMV | 1×9728×2560 | 56.180 ms | 19.688 ms | 2.530 GOPS | **2.854×** |

语言 prefill 三次内核计时为 586.362、586.328、586.309 ms，极差仅
0.053 ms。v1 数据来自同日、同板卡、同矩阵及同一运行时口径的基线报告。

## 2. 算法与数据流

- 数据格式：每 32 个 INT8 配一个 FP16 scale，保持 GGML Q8_0 算术。
- tile：`TM=16`、`TN=64`、`QK=32`，8 路 N 方向 PE。
- 循环顺序：`n_tile → load_weight → m_tile → load_activation → compute → store`。
- 权重 tile：量化值和 scale 均驻留片上 BRAM，跨 M tile 复用。
- AXI：权重 Q、激活 Q、辅助 scale/输出分别走 HPC0、HPC1、HP0。
- 控制：A53 经 AXI4-Lite 启动内核；Linux UIO 映射控制寄存器与
  `0xB7C000000` 起始的 64 MiB 保留 DMA 区。

相对 v1 的 `m_tile → n_tile` 顺序，语言 prefill 的源码模型有效 DDR
请求量从约 1.078 GB 降至 258.064 MB（238.141 MB 读、19.923 MB 写），
下降约 76%。这与 3.930× 板端加速相符，证明 v1 的主要瓶颈确实是权重
重复读取和 AXI/DDR 等待，而非 INT8 乘法器数量不足。

## 3. 验证链

1. C Golden：6/6 case 通过，覆盖小矩阵、GEMV、tile 边界、K=2560、
   K=1024 和非法输入。
2. Verilog RTL：XSim 完成 5/5 transaction；C post-check 覆盖 3 个数值
   case 和非法输入，最大绝对误差 `6.5267086e-06`，`TB_RESULT=PASS`。
3. 核级实现：200 MHz 导出 IP 成功；实现后最差周期 4.903 ns。
4. 系统级实现：199.98001 MHz，WNS `+0.123 ns`、WHS `+0.002 ns`、
   未布线网络 0。
5. JTAG 启动门：SPD magic `0x53504434`、2 ranks、16 GiB；PL configured；
   AXI-Lite control `0x00000004`（idle）后启动 Linux。
6. 板端自检：`M=2,N=3,K=32`，82 µs，`max_abs=0`；三类大矩阵
   benchmark 均 `result=PASS`。

XSim 2023.2 在不可写 Vivado 用户状态目录的受限沙箱中会抛出
`unexpected exception`；在正常主机权限下直接运行同一 elaborated snapshot
通过。工程脚本和 PASS 日志均已保留。

## 4. 资源、时序与功耗边界

| 项目 | 完整 PS+PL 系统 | ZU9EG 占比 |
|---|---:|---:|
| LUT | 17,897 | 约 6.53% |
| FF | 21,151 | 3.86% |
| BRAM tile | 301 | 33.00% |
| DSP | 316 | 12.54% |

Vivado 无向量活动传播估计：总片上功耗 4.608 W，其中动态 3.876 W、静态
0.732 W，置信度 Medium。这不是电源轨实测，不能作为论文能效结论；后续需
读取板载功耗监控或外接功率计，并报告 J/token 或 J/inference。

## 5. 可观测数据与瓶颈判断

运行时逐 chunk 输出 begin/done、wall time、按 200 MHz 换算的周期数、源码
模型 DDR 读写字节、有效带宽及权重复用次数。这里的 `pl_cycles_200mhz` 是
板端 wall time 换算值，DDR bytes/bandwidth 是源码请求模型，均不是 AXI APM
硬件事件计数，报告中不得混称为物理总线 beat。

| 工作负载 | 实测/无停顿排程 | 模型读/写 | 解释 |
|---|---:|---:|---|
| Language prefill | 1.750× | 238.141/19.923 MB | 权重复用已生效，剩余为 AXI 与流水停顿 |
| Vision encoder | 2.421× | 340.918/70.779 MB | 64 MiB DMA 迫使 9 chunks，且停顿更明显 |
| Decoder GEMV | 1.381× | 26.874/0.039 MB | 权重流量主导，需独立 GEMV 数据路 |

因此下一阶段的优先级是：先接入 AXI Performance Monitor 的真实 AR/R/W
beat、stall 与 busy 计数，再做 activation/output ping-pong 和计算/访存重叠；
视觉路径同时扩大 DMA 保留区以消除 9 次 chunk 调用。不能仅凭当前模型带宽
继续增加 PE，否则可能增加功耗而不提高有效吞吐。

## 6. 可复现产物

- HLS 源码：`fpga_accel/q8q8_weight_stationary_hls/src/q8q8_gemm.cpp`
- RTL XSim 日志：`fpga_accel/q8q8_weight_stationary_hls/artifacts/q8q8_weight_stationary_rtl_xsim.log`
- RTL post-check：`fpga_accel/q8q8_weight_stationary_hls/artifacts/q8q8_weight_stationary_rtl_postcheck.log`
- 位流/XSA：`fpga_accel/q8q8_v2_zcu102_vivado/artifacts/q8q8_zcu102.bit`、
  `q8q8_zcu102.xsa`
- 系统时序/资源/功耗：`fpga_accel/q8q8_v2_zcu102_vivado/artifacts/`
- 板端原始 UART：`reports/2026-08-23-zcu102-q8q8-weight-stationary-v2-uart.log`
- v1 对照：`reports/2026-08-23-zcu102-q8q8-large-matrix-benchmark.md`

关键 SHA-256：bitstream
`6af76a72f8733bceb1a3d381e255cf29a828ce5f8b3f5b42dd73e10f7b8e0079`；
XSA `e9d5b30c91666636a43be386dd506ee4f65bd0353b08c3bdfeda91fa9bb084e5`；
FSBL `0c8ed3df03dfbd78f8daa9259098273e99af4e7edc8ea50e6bc5efe05018f76f`；
initramfs `a6b042bf8cf0cf3e7e96beb817691460073054fff41b6aba1ce8c3c9fd33304d`。
