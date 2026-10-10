# 当前 Selector 异构时延投影（K48/K64/K96）

> **PURE ANALYSIS ESTIMATE — not a ZCU102 or Jetson measurement**

- 生成时间：`2026-10-04T00:21:20.602438+00:00`
- measurement_status：`not_measured`
- 投影公式：`projected_model_E2E = measured_model_inference - measured_software_selector + fpga_compute + jetson_to_fpga_transfer + fpga_to_jetson_transfer`
- 条件：Wv resident、`G<=K` 本地 identity bypass、无重叠（严格串行）

## 1. 假设

- Wv resident：每图 0 B（一次性 131072 B 摊销）
- 1 GbE 有效带宽：**110 MB/s**
- 假设 10 GbE 有效带宽：**1100 MB/s**（假设值，非实测）
- FPGA 目标频率：200 MHz（accepted）
- 重叠：none (strictly serial J->F, compute, F->J)

## 2. 每 K 汇总（ms，100 样本）

| K | identity 样本 | FPGA compute mean | 传输 1GbE mean | 传输 10GbE mean | 实测 model mean | 实测 selector-soft mean | 投影 model 1GbE mean | 投影 model 10GbE mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K48 | 0/100 | 8.967 | 35.497 | 3.550 | 156.913 | 45.399 | 155.977 | 124.030 |
| K64 | 2/100 | 9.026 | 36.093 | 3.609 | 158.027 | 45.619 | 157.526 | 125.043 |
| K96 | 2/100 | 9.162 | 37.287 | 3.729 | 166.039 | 45.610 | 166.877 | 133.320 |

## 3. 每 K 明细（mean / median / p95，ms）

### K48

| 量 | mean | median | p95 |
|---|---:|---:|---:|
| FPGA compute | 8.967 | 5.187 | 16.639 |
| 传输 1GbE | 35.497 | 19.991 | 66.955 |
| 传输 10GbE(假设) | 3.550 | 1.999 | 6.696 |
| 实测 model inference | 156.913 | 145.160 | 205.424 |
| 实测 selector software | 45.399 | 31.554 | 78.064 |
| 投影 model E2E 1GbE | 155.977 | 148.945 | 211.472 |
| 投影 model E2E 10GbE(假设) | 124.030 | 117.846 | 161.424 |
| 投影 − 实测 model 1GbE | -0.935 | 2.377 | 6.544 |
| 投影 − 实测 model 10GbE(假设) | -32.882 | -26.344 | -11.570 |

### K64

| 量 | mean | median | p95 |
|---|---:|---:|---:|
| FPGA compute | 9.026 | 5.248 | 16.700 |
| 传输 1GbE | 36.093 | 20.586 | 67.551 |
| 传输 10GbE(假设) | 3.609 | 2.059 | 6.755 |
| 实测 model inference | 158.027 | 148.206 | 211.817 |
| 实测 selector software | 45.619 | 24.687 | 91.879 |
| 投影 model E2E 1GbE | 157.526 | 147.793 | 216.897 |
| 投影 model E2E 10GbE(假设) | 125.043 | 121.699 | 156.101 |
| 投影 − 实测 model 1GbE | -0.501 | 1.707 | 6.905 |
| 投影 − 实测 model 10GbE(假设) | -32.984 | -21.296 | -10.742 |

### K96

| 量 | mean | median | p95 |
|---|---:|---:|---:|
| FPGA compute | 9.162 | 5.387 | 16.839 |
| 传输 1GbE | 37.287 | 21.780 | 68.745 |
| 传输 10GbE(假设) | 3.729 | 2.178 | 6.874 |
| 实测 model inference | 166.039 | 152.231 | 212.215 |
| 实测 selector software | 45.610 | 26.596 | 82.129 |
| 投影 model E2E 1GbE | 166.877 | 158.356 | 220.696 |
| 投影 model E2E 10GbE(假设) | 133.320 | 131.214 | 165.026 |
| 投影 − 实测 model 1GbE | 0.839 | 3.355 | 8.791 |
| 投影 − 实测 model 10GbE(假设) | -32.719 | -21.666 | -11.570 |

## 4. 输入 SHA256

| 输入 | SHA256 |
|---|---|
| accepted_hls_report | `8871aebd587793466520bb6541d54f9910fa2b93614a34e24366f55ef5ce85e2` |
| accepted_csynth_model | `693455fbb9c052ee011c672f081d10167cdcd91826b825b972f2331d854db250` |
| transfer_audit_module | `5bc312386148b38721afd3224797047a817955734b20fe8e5ed1a3409b9bc610` |
| latency100_K48 | `7a416e2be0b9d0ffef84d092c2729f65b64259b1172485a21c4c327ccd95946c` |
| latency100_K64 | `d52cbbda0437cf4697b6b88befc940348c9710d6ea82487e47bd1c62a856e7e5` |
| latency100_K96 | `84190141b6eb8d967f2814ae1b7d1170f880e4eed8eb7f41824a1f365b44525b` |

## 5. 限制（必须随结论引用）

- Pure analytic projection: no bitstream, no board, no link measurement and no model run.
- 1 GbE time uses an explicit 110 MB/s effective-payload-bandwidth assumption.
- 10 GbE time assumes a hypothetical 1.1 GB/s effective payload bandwidth; it is not a measured link.
- Wv is assumed resident: one 131,072 B load amortised away, 0 B per image.
- FPGA compute uses the accepted compact-pipeline formula at the accepted 200 MHz target.
- No pipelining / overlap: Jetson->FPGA, FPGA compute and FPGA->Jetson are strictly serial.
- The measured baselines are the latency-100 software-Selector runs, not FPGA runs.
- The accepted cycle model is a formula estimate built from per-loop unit costs; it is not a measured FPGA latency.
- Latency-100 G values differ from the accepted production grids, so the cycle model is extrapolated over the observed CV-Bench grid range.
- The measured baselines include GPU/CPU scheduling noise; only their medians over 3 repeats are used and then aggregated over 100 samples.

> **本文件是纯分析估算，不是 ZCU102/Jetson 实测。** 未上板、未测链路、未跑模型；不得作为加速/时延实测或精度结论引用。
