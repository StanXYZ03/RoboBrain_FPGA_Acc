# ZCU102 GGML operator profile

Parsed 1386 operator samples from 2 graphs.

> The instrumented run inserts synchronization and log output. Operator shares and rankings are valid for bottleneck discovery; use an uninstrumented run for end-to-end latency.

## Measured operator time by stage

| Stage | Operator | Calls | Time (ms) | Stage share | Matmul GFLOP/s | Ideal traffic (GB/s) | AI (FLOP/B) | Roofline |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| text_prefill | MUL_MAT | 253 | 12774.263 | 98.9% | 8.43 | 0.34 | 24.496 | compute |
| decode | MUL_MAT | 253 | 995.233 | 97.7% | 8.08 | 4.30 | 1.878 | memory |
| text_prefill | FLASH_ATTN_EXT | 36 | 59.899 | 0.5% | 0.00 | 0.93 | 0.000 | memory |
| text_prefill | SWIGLU | 36 | 32.985 | 0.3% | 0.00 | 1.86 | 0.000 | memory |
| text_prefill | RMS_NORM | 145 | 22.794 | 0.2% | 0.00 | 1.95 | 0.000 | memory |
| text_prefill | ROPE | 72 | 8.533 | 0.1% | 0.00 | 2.59 | 0.000 | memory |
| decode | SWIGLU | 36 | 8.138 | 0.8% | 0.00 | 0.52 | 0.000 | memory |
| text_prefill | SET_ROWS | 72 | 7.492 | 0.1% | 0.00 | 1.18 | 0.000 | memory |
| decode | FLASH_ATTN_EXT | 36 | 6.335 | 0.6% | 0.00 | 6.14 | 0.000 | memory |
| text_prefill | ADD | 75 | 4.403 | 0.0% | 0.00 | 8.04 | 0.000 | memory |
| decode | RMS_NORM | 145 | 3.934 | 0.4% | 0.00 | 0.75 | 0.000 | memory |
| decode | SET_ROWS | 72 | 2.003 | 0.2% | 0.00 | 0.29 | 0.000 | memory |
| decode | ROPE | 72 | 1.376 | 0.1% | 0.00 | 1.07 | 0.000 | memory |
| decode | ADD | 75 | 1.168 | 0.1% | 0.00 | 1.97 | 0.000 | memory |
| text_prefill | PAD | 1 | 0.769 | 0.0% | 0.00 | 1.00 | 0.000 | memory |
| decode | PAD | 1 | 0.166 | 0.0% | 0.00 | 0.31 | 0.000 | memory |
| text_prefill | GET_ROWS | 3 | 0.107 | 0.0% | 0.00 | 2.20 | 0.000 | memory |
| decode | GET_ROWS | 3 | 0.025 | 0.0% | 0.00 | 2.16 | 0.000 | memory |

## Interpretation constraints

- timing: barrier-synchronized CPU wall time per GGML graph node
- flops: 2*K per output element for MUL_MAT/MUL_MAT_ID only
- bytes: ideal lower-bound tensor traffic; cache reuse and write allocation are not modeled
- warning: profiling changes scheduling and logging overhead; use shares and operator ranking, not profiled end-to-end latency
