# ZCU102 GGML operator profile

Parsed 4730 operator samples from 7 graphs.

> The instrumented run inserts synchronization and log output. Operator shares and rankings are valid for bottleneck discovery; use an uninstrumented run for end-to-end latency.

## Measured operator time by stage

| Stage | Functional block | Calls | Time (ms) | Stage share | GFLOP/s | AI (FLOP/B) | Roofline |
|---|---:|---:|---:|---:|---:|---:|---|
| vision_prefill | ffn_linear | 324 | 668156.724 | 66.6% | 8.45 | 399.250 | compute |
| vision_encode | attention_core | 24 | 278022.105 | 44.5% | 6.60 | 1440.000 | compute |
| vision_prefill | attention_projection | 432 | 241295.404 | 24.1% | 8.45 | 329.143 | compute |
| vision_encode | ffn_linear | 48 | 204750.658 | 32.7% | 8.50 | 389.958 | compute |
| vision_encode | attention_projection | 48 | 102769.028 | 16.4% | 8.46 | 327.583 | compute |
| vision_prefill | attention_core | 108 | 88331.377 | 8.8% | 7.48 | 374.417 | compute |
| text_prefill | ffn_linear | 108 | 30421.686 | 47.8% | 8.43 | 82.427 | compute |
| text_prefill | attention_core | 36 | 21959.274 | 34.5% | 1.68 | 150.048 | compute |
| vision_encode | vision_deepstack | 6 | 20639.750 | 3.3% | 8.56 | 632.455 | compute |
| text_prefill | attention_projection | 144 | 10980.611 | 17.2% | 8.42 | 80.617 | compute |
| vision_encode | multimodal_projector | 2 | 6885.529 | 1.1% | 8.55 | 632.455 | compute |
| vision_encode | activation | 28 | 3779.307 | 0.6% | 0.00 | 0.000 | memory |
| vision_encode | matmul_other | 2 | 2366.100 | 0.4% | 5.74 | 247.299 | compute |
| vision_prefill | activation | 108 | 2191.624 | 0.2% | 0.00 | 0.000 | memory |
| vision_encode | data_movement | 58 | 2117.390 | 0.3% | 0.00 | 0.000 | memory |
| vision_encode | elementwise | 259 | 1848.885 | 0.3% | 0.00 | 0.000 | memory |
| vision_prefill | normalization | 435 | 1532.635 | 0.2% | 0.00 | 0.000 | memory |
| vision_encode | normalization | 52 | 846.067 | 0.1% | 0.00 | 0.000 | memory |
| vision_encode | rope | 48 | 822.807 | 0.1% | 0.00 | 0.000 | memory |
| decode | ffn_linear | 108 | 667.080 | 43.4% | 8.06 | 1.879 | memory |
| kv_cache | ffn_linear | 108 | 647.588 | 71.8% | 8.08 | 1.827 | memory |
| vision_prefill | rope | 216 | 540.887 | 0.1% | 0.00 | 0.000 | memory |
| decode | attention_core | 36 | 522.662 | 34.0% | 1.44 | 3.975 | memory |
| vision_prefill | data_movement | 222 | 500.713 | 0.0% | 0.00 | 0.000 | memory |
| vision_prefill | elementwise | 225 | 363.169 | 0.0% | 0.00 | 0.000 | memory |
| vision_encode | upscale | 1 | 347.359 | 0.1% | 0.00 | 0.000 | memory |
| decode | attention_projection | 144 | 233.914 | 15.2% | 8.07 | 1.877 | memory |
| kv_cache | attention_projection | 144 | 233.504 | 25.9% | 8.08 | 1.877 | memory |
| text_prefill | activation | 36 | 104.983 | 0.2% | 0.00 | 0.000 | memory |
| vision_encode | im2col | 2 | 99.423 | 0.0% | 0.00 | 0.000 | memory |
| decode | lm_head | 1 | 95.303 | 6.2% | 8.16 | 1.880 | memory |
| text_prefill | lm_head | 1 | 95.186 | 0.1% | 8.17 | 1.880 | memory |
| text_prefill | normalization | 145 | 70.267 | 0.1% | 0.00 | 0.000 | memory |
| text_prefill | data_movement | 76 | 26.175 | 0.0% | 0.00 | 0.000 | memory |
| text_prefill | rope | 72 | 25.235 | 0.0% | 0.00 | 0.000 | memory |
| text_prefill | elementwise | 75 | 14.307 | 0.0% | 0.00 | 0.000 | memory |
| decode | activation | 36 | 8.145 | 0.5% | 0.00 | 0.000 | memory |
| kv_cache | activation | 36 | 7.957 | 0.9% | 0.00 | 0.000 | memory |
| decode | normalization | 145 | 3.938 | 0.3% | 0.00 | 0.000 | memory |
| kv_cache | normalization | 145 | 3.915 | 0.4% | 0.00 | 0.000 | memory |
| kv_cache | attention_core | 36 | 3.775 | 0.4% | 40.00 | 3.879 | memory |
| decode | data_movement | 76 | 2.189 | 0.1% | 0.00 | 0.000 | memory |
| kv_cache | data_movement | 76 | 2.173 | 0.2% | 0.00 | 0.000 | memory |
| decode | rope | 72 | 1.368 | 0.1% | 0.00 | 0.000 | memory |
| kv_cache | rope | 72 | 1.330 | 0.1% | 0.00 | 0.000 | memory |
| decode | elementwise | 75 | 1.145 | 0.1% | 0.00 | 0.000 | memory |
| kv_cache | elementwise | 75 | 1.111 | 0.1% | 0.00 | 0.000 | memory |
| vision_prefill | lm_head | 3 | 0.002 | 0.0% | 0.00 | 0.000 | memory |
| kv_cache | lm_head | 1 | 0.001 | 0.0% | 0.00 | 0.000 | memory |

## GGML operators

| Stage | Operator | Calls | Time (ms) | Stage share | Matmul GFLOP/s | Ideal traffic (GB/s) | AI (FLOP/B) | Roofline |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| vision_prefill | MUL_MAT | 759 | 909452.130 | 90.7% | 8.45 | 0.02 | 356.192 | compute |
| vision_encode | MUL_MAT | 106 | 337411.065 | 54.0% | 8.47 | 0.02 | 378.937 | compute |
| vision_encode | FLASH_ATTN_EXT | 24 | 278022.105 | 44.5% | 6.60 | 0.00 | 1440.000 | compute |
| vision_prefill | FLASH_ATTN_EXT | 108 | 88331.377 | 8.8% | 7.48 | 0.02 | 374.417 | compute |
| text_prefill | MUL_MAT | 253 | 41497.483 | 65.1% | 8.43 | 0.11 | 74.847 | compute |
| text_prefill | FLASH_ATTN_EXT | 36 | 21959.274 | 34.5% | 1.68 | 0.01 | 150.048 | compute |
| vision_encode | GELU | 28 | 3779.307 | 0.6% | 0.00 | 0.94 | 0.000 | memory |
| vision_prefill | SWIGLU | 108 | 2191.624 | 0.2% | 0.00 | 2.01 | 0.000 | memory |
| vision_encode | CPY | 48 | 1770.569 | 0.3% | 0.00 | 1.44 | 0.000 | memory |
| vision_encode | ADD | 207 | 1619.822 | 0.3% | 0.00 | 7.66 | 0.000 | memory |
| vision_prefill | RMS_NORM | 435 | 1532.635 | 0.2% | 0.00 | 2.09 | 0.000 | memory |
| decode | MUL_MAT | 253 | 996.297 | 64.9% | 8.07 | 4.30 | 1.878 | memory |
| kv_cache | MUL_MAT | 253 | 881.093 | 97.8% | 8.08 | 4.86 | 1.662 | memory |
| vision_encode | NORM | 52 | 846.067 | 0.1% | 0.00 | 2.18 | 0.000 | memory |
| vision_encode | ROPE | 48 | 822.807 | 0.1% | 0.00 | 4.13 | 0.000 | memory |
| vision_prefill | ROPE | 216 | 540.887 | 0.1% | 0.00 | 2.95 | 0.000 | memory |
| decode | FLASH_ATTN_EXT | 36 | 522.662 | 34.0% | 1.44 | 0.36 | 3.975 | memory |
| vision_prefill | SET_ROWS | 216 | 500.707 | 0.0% | 0.00 | 1.27 | 0.000 | memory |
| vision_prefill | ADD | 225 | 363.169 | 0.0% | 0.00 | 7.03 | 0.000 | memory |
| vision_encode | UPSCALE | 1 | 347.359 | 0.1% | 0.00 | 0.08 | 0.000 | memory |
| vision_encode | MUL | 52 | 229.063 | 0.0% | 0.00 | 8.03 | 0.000 | memory |
| vision_encode | CONCAT | 3 | 181.958 | 0.0% | 0.00 | 1.09 | 0.000 | memory |
| vision_encode | CONT | 7 | 164.863 | 0.0% | 0.00 | 1.50 | 0.000 | memory |
| text_prefill | SWIGLU | 36 | 104.983 | 0.2% | 0.00 | 1.91 | 0.000 | memory |
| vision_encode | IM2COL | 2 | 99.423 | 0.0% | 0.00 | 0.46 | 0.000 | memory |
| text_prefill | RMS_NORM | 145 | 70.267 | 0.1% | 0.00 | 2.06 | 0.000 | memory |
| text_prefill | ROPE | 72 | 25.235 | 0.0% | 0.00 | 2.87 | 0.000 | memory |
| text_prefill | SET_ROWS | 72 | 23.927 | 0.0% | 0.00 | 1.21 | 0.000 | memory |
| text_prefill | ADD | 75 | 14.307 | 0.0% | 0.00 | 8.10 | 0.000 | memory |
| decode | SWIGLU | 36 | 8.145 | 0.5% | 0.00 | 0.52 | 0.000 | memory |
| kv_cache | SWIGLU | 36 | 7.957 | 0.9% | 0.00 | 0.51 | 0.000 | memory |
| decode | RMS_NORM | 145 | 3.938 | 0.3% | 0.00 | 0.75 | 0.000 | memory |
| kv_cache | RMS_NORM | 145 | 3.915 | 0.4% | 0.00 | 0.75 | 0.000 | memory |
| kv_cache | FLASH_ATTN_EXT | 36 | 3.775 | 0.4% | 40.00 | 10.31 | 3.879 | memory |
| text_prefill | PAD | 1 | 2.145 | 0.0% | 0.00 | 1.17 | 0.000 | memory |
| decode | SET_ROWS | 72 | 1.999 | 0.1% | 0.00 | 0.30 | 0.000 | memory |
| kv_cache | SET_ROWS | 72 | 1.996 | 0.2% | 0.00 | 0.30 | 0.000 | memory |
| decode | ROPE | 72 | 1.368 | 0.1% | 0.00 | 1.08 | 0.000 | memory |
| kv_cache | ROPE | 72 | 1.330 | 0.1% | 0.00 | 1.11 | 0.000 | memory |
| decode | ADD | 75 | 1.145 | 0.1% | 0.00 | 2.01 | 0.000 | memory |
| kv_cache | ADD | 75 | 1.111 | 0.1% | 0.00 | 2.02 | 0.000 | memory |
| kv_cache | PAD | 1 | 0.166 | 0.0% | 0.00 | 0.31 | 0.000 | memory |
| decode | PAD | 1 | 0.166 | 0.0% | 0.00 | 0.31 | 0.000 | memory |
| text_prefill | GET_ROWS | 3 | 0.103 | 0.0% | 0.00 | 6.57 | 0.000 | memory |
| decode | GET_ROWS | 3 | 0.024 | 0.0% | 0.00 | 2.25 | 0.000 | memory |
| kv_cache | GET_ROWS | 3 | 0.011 | 0.0% | 0.00 | 1.18 | 0.000 | memory |
| vision_prefill | GET_ROWS | 6 | 0.006 | 0.0% | 0.00 | 0.00 | - | unclassified |

## Interpretation constraints

- timing: barrier-synchronized CPU wall time per GGML graph node
- flops: 2*K per output element for matmul; QK and PV terms for FLASH_ATTN_EXT; softmax is excluded
- bytes: ideal lower-bound tensor traffic; cache reuse and write allocation are not modeled
- warning: profiling changes scheduling and logging overhead; use shares and operator ranking, not profiled end-to-end latency
