# ZCU102 DATAFLOW 板端复测

结论：`BOARD_RERUN_PASS / KERNEL_ASSUMPTION_STABLE / END_TO_END_NOT_MEASURED`。

本次直接使用板上既有 DATAFLOW RAM Linux 与 UART shell 重跑冻结协议。没有复位、没有重新配置
PL，也没有写入 SD/QSPI。协议前后自检和全部抽样均为 `max_abs=0`，APM 无溢出，结束后控制
寄存器为 `0x00000004`（idle）。

## 实测结果

| workload | 本次板测 | 有效吞吐 | 相对 2026-08-30 延迟变化 |
|---|---:|---:|---:|
| Language Prefill, M=512 | 174.559 ms | 146.090 GOPS | +0.0082% |
| Visual Prefill, M=1080 auto | 368.879 ms | 145.826 GOPS | -0.0029% |
| Vision Encoder, M=4320 auto | 252.063 ms | 143.769 GOPS | -0.0033% |
| Decoder GEMV, M=1 auto | 14.660 ms | 3.398 GOPS | -0.0591% |

四项与上次板测的差异均小于 `0.06%`。因此，整图一阶投影采用的 `144–146 GOPS` 大 M
矩阵核假设稳定，没有出现数量级偏差。

## 更新后的整图投影

将冻结的 PS 完整图片基线中可覆盖的 `MUL_MAT` 时间替换为本次 PL 吞吐，保留未覆盖算子、
CPU Decoder，并加入此前实测数据路径的一阶开销：

| 阶段 | 投影时间 |
|---|---:|
| Vision Encoder | 322.200 s |
| 视觉 token 的 LLM Prefill | 164.841 s |
| 文本 Prefill | 40.937 s |
| 生成 8 token | 10.749 s |
| 合计 | **538.727 s（8.979 min）** |

这仍不是整图板端实测。当前完整 GGML/llama.cpp Transformer 后端尚未连接 PL，且部分小 M、
转置 shape、完整调度与驻留 Block 尚未实测。因此本次只能说明矩阵核理论依据稳定，不能消除
端到端集成误差；逐算子 PS 往返也不应成为最终架构。

原始证据：`reports/20260831T-board-rerun-uart.log`，SHA256=
`436a3947a509777c550dbb129bc406fd08b30a654e7ad44c217dde929d6756ef`。
