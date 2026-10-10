# ZCU102 Q8×Q8 v2：DDR APM 真实瓶颈判定

日期：2026-08-23  
状态：`STAGE_PASS / APM_COUNTER_PASS / BOARD_NUMERIC_PASS`

## 1. 结论

板上硬化 DDR AXI Performance Monitor（APM）已经接入现有 benchmark，三类
矩阵均取得真实 DDR 事务、字节、延迟、写 beat 和 AXI idle 计数，所有计数器
无溢出，全部数值抽检 `max_abs=0`。本阶段没有修改或重烧 PL 位流。

- **Language prefill、vision encoder：不是 DDR 峰值带宽瓶颈。** 算术强度
  分别为 98.81 和 88.01 OP/B，远高于本设计约 6.0 OP/B 的 roofline 分界；
  实际 DDR 带宽却只占 DDR4-2133 理论峰值 2.58% 和 1.72%。主要损失是
  load/compute/store 串行、标量输出写事务和流水调度，而不是 DDR 已饱和。
- **Decoder GEMV：memory-bound。** 算术强度只有 1.85 OP/B，低于分界；
  每个 token 必须流过约 26.9 MB 权重/scale，实测 1.369 GB/s、2.533 GOPS。
  这是权重流与访存路径效率瓶颈，但 DDR 物理带宽仍未饱和。
- v1 曾因重复装载权重而显著 memory-bound；v2 weight-stationary 已消除该主因，
  因而 prefill/vision 的下一优先级应转向并行数据流与输出合并，而非直接增加
  DDR 频率或盲目增加 PE。

## 2. 观测方法与拓扑边界

ZynqMP PS 内置 DDR APM 位于 `0xFD0B0000`，有 6 个 slot、10 个 32-bit
Advanced-mode counter。Linux 将其枚举为 `uio2/axi-pmon`。每个 accelerator
chunk 前清零并启动 APM，`q8q8_run()` 返回后立即停止、读取并检查 overflow。

| APM slot | 实际监视点 | 本设计流量 | 解释限制 |
|---:|---|---|---|
| 1、2 | 两条 CCI→DDR 交错端口 | HPC0 权重 Q + HPC1 激活 Q + 极少 CPU 背景 | 必须合计，不能分别命名为 HPC0/HPC1 |
| 3 | DDR port 3（含 PL HP0） | weight/activation scale + FP32 output | 可独立统计 AUX 路径 |

官方 TRM 说明 DDR APM 的 6 个 slot 对应 DDR 控制器 6 个端口；UG1221 给出
slot 1/2 为两条 CCI ACE-Lite→DDR 路径、slot 3 含 PL HP0。APM 指标定义遵循
PG037：metric 0/1/2/3 为事务与字节，4 为 write beat，5/6 为总读写延迟，
7/8 为 slave-write/master-read idle，15 为最大读延迟。

参考：

- AMD UG1085：<https://docs.amd.com/r/en-US/ug1085-zynq-ultrascale-trm/AXI-Performance-Monitor>
- AMD UG1221：<https://docs.amd.com/r/en-US/ug1221-zcu102-base-trd/Performance-Monitor-Library>
- AMD PG037：<https://docs.amd.com/v/u/en-US/pg037_axi_perf_mon>

## 3. 真实流量结果

主表采用补充的 write-beat 采样；latency 采用紧邻的第二次运行。两次 wall time
高度一致。带宽使用 APM 实测总字节 / kernel wall time，GB 按十进制计算。

| 工作负载 | Wall time | APM 读/写 | APM 总带宽 | DDR4-2133 峰值占比 | APM/源码读量差 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| Language prefill | 586.339 ms | 238.157 / 19.923 MB | 0.440 GB/s | 2.58% | +15,872 B（0.007%） | PASS |
| Vision encoder | 1,404.381 ms | 341.049 / 70.779 MB | 0.293 GB/s | 1.72% | +130,240 B（0.038%） | PASS |
| Decoder GEMV | 19.664 ms | 26.879 / 0.039 MB | 1.369 GB/s | 8.02% | +5,312 B（0.020%） | PASS |

APM 与源码请求模型的差异低于 0.03%，说明 bracket 内 CPU/内核背景流量很小，
也验证了 weight-stationary 的真实读量。APM global counter / wall time 稳定为
533.28–533.38 MHz；这是实测 APM counter clock，不把它混同为 AXI 或 DDR
数据率。

### 事务粒度与忙/停顿

| 工作负载 | CCI1/CCI2 平均读事务 | HP0 平均读事务 | 输出写事务/beat | HP0 写 beat / 采样周期 |
|---|---:|---:|---:|---:|
| Language prefill | 255.94 / 255.95 B | 155.15 B | 4,980,736 / 4,980,736 | 1.593% |
| Vision encoder | 255.69 / 255.69 B | 64.00 B | 17,694,720 / 17,694,720 | 2.363% |
| Decoder GEMV | 255.78 / 255.90 B | 155.22 B | 9,728 / 9,728 | 0.093% |

CCI 读事务接近 256 B，说明量化权重/激活的读 burst 已正常形成。相反，输出
每个 FP32 元素都是一个独立 4 B AXI write transaction，且一事务一 beat；
这是 prefill/vision 最明确、可直接优化的硬件证据。当前 APM 的 MM 指标没有
read-beat counter，因此不能把 `read_bytes/16` 伪称为精确读忙周期；报告只给
真实 write beat。全部运行中 `Mst_Rd_Idle` 为 0–4 cycle，`Slv_Wr_Idle=0`，
即没有可观测的读消费者 backpressure 或 DDR 写侧 backpressure；它们不覆盖
“DDR 未及时给出读数据”的周期，该部分由读延迟指标反映。

## 4. 延迟结果

| 工作负载 | CCI1 平均/最大读延迟 | CCI2 平均/最大读延迟 | HP0 平均读延迟 |
|---|---:|---:|---:|
| Language prefill | 95.63 / 358 cycle | 93.37 / 394 cycle | 65.09 cycle |
| Vision encoder | 82.71 / 402 cycle | 82.73 / 415 cycle | 56.25 cycle |
| Decoder GEMV | 93.92 / 396 cycle | 91.39 / 397 cycle | 63.15 cycle |

按本次实测 APM 约 533.3 MHz 换算，CCI 平均读延迟约 155–179 ns，最大约
671–778 ns；HP0 平均约 105–122 ns。`Total Write Latency / Write Txn = 1`
是该 monitor 点看到的缓冲响应延迟，不能解释为 DRAM CAS latency。

## 5. Roofline 判定依据

条件：PL 200 MHz，8 个 N 向 PE，每 PE 每周期完成一个 32-lane INT8 dot，
理论峰值为 `8×32×200M = 51.2 GMAC/s = 102.4 GOPS`；DDR4-2133、64-bit
理论峰值为 `2133.125 MT/s×8 B = 17.065 GB/s`。因此机器平衡点约为：

`102.4 GOPS / 17.065 GB/s = 6.00 OP/B`

| 工作负载 | 算术强度 | 实测 GOPS | 计算峰值利用率 | 无停顿排程/实测 | 判定 |
|---|---:|---:|---:|---:|---|
| Language prefill | 98.81 OP/B | 43.49 | 42.47% | 1 / 1.750 | compute/schedule-bound |
| Vision encoder | 88.00 OP/B | 25.80 | 25.20% | 1 / 2.399 | compute/schedule-bound，串行开销更重 |
| Decoder GEMV | 1.85 OP/B | 2.533 | 2.47% | 1 / 1.380 | memory/weight-stream-bound |

“compute/schedule-bound”不代表访存无影响：prefill/vision 的实测时间仍分别是
HLS 无停顿串行排程的 1.75×、2.40×。准确解释是 DDR 容量充足，但当前微架构
没有把 load、compute、store 重叠，且数百万个标量写事务放大了协议与流水开销。

## 6. 下一阶段设计目标

阶段目标：保持数值与三类 workload 功能不变，先把 vision encoder 的板端
kernel time 从约 1.404 s 降低至少 25%，并取得 APM 证据证明优化来源。

1. 将 FP32 output 改为 128/256/512-bit pack + burst。当前一元素一事务；按
   `64` 列 tile 合并，事务数理论上可从 4.98M/17.69M 降到约 77.8K/276.5K，
   128-bit 数据 beat 至少减少约 4×。
2. 对 activation load、compute、output store 做双缓冲和 `DATAFLOW` 重叠；
   优先 vision，因为其排程倍率 2.40×。
3. 将 reserved DMA arena 扩到至少 128 MiB，验证单次 vision 调用，消除 9 个
   chunk 的重复权重装载与启动边界。
4. decoder 单列专用 GEMV 路径：增加输出通道并行、跨端口权重流或结构化稀疏；
   不用 prefill 的高 M tile 结构硬套 GEMV。
5. 每次改动继续执行 C/RTL testbench → 实现时序 → JTAG 上板 → APM traffic/
   latency → 数值 golden 的完整门控；性能结论只使用硬件计数与板端 wall time。

## 7. 可复现产物

- APM runtime：`fpga_accel/q8q8_runtime/include/q8q8_apm.h`、
  `src/q8q8_apm.c`
- APM 单元测试：`fpga_accel/q8q8_runtime/tests/test_q8q8_apm.c`
- benchmark：`fpga_accel/q8q8_runtime/src/q8q8_benchmark.c`
- 最终 traffic+latency 原始日志：
  `reports/2026-08-23-zcu102-q8q8-ddr-apm-final.log`
- 中间交叉验证日志：
  `reports/2026-08-23-zcu102-q8q8-ddr-apm-matrix.log`
- write-beat 补充日志：
  `reports/2026-08-23-zcu102-q8q8-ddr-apm-write-beats.log`
- UART 操作日志：`reports/2026-08-23-zcu102-q8q8-ddr-apm-uart.log`

SHA-256：最终 ARM64 benchmark
`806da58f960ab54f08d80495539fcc2527192dc2a3613243b85751d7254b463c`；
final log
`1d5f47d8d07417c7425afa1539712d19dff140719eb2688024eb3bae95deb584`；
write-beat log
`a9f1b5ab8df4884227f8f3f02555bec8600f4f7b27f028c4ff700175ef40d25f`。

最终板态：accelerator `ap_ctrl=0x4`（idle），APM `control=0x0`（stopped），
UART shell alive；未重烧位流，板上模型/web 服务未恢复。
