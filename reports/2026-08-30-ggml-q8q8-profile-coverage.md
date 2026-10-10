# 现有GGML日志的Q8×Q8 PL桥接覆盖率审计

状态：`PROFILE_COVERAGE_AUDIT_PASS / NO_RUNTIME_DISPATCH_CLAIM`

## 按阶段覆盖

| 阶段 | 可命中/矩阵乘调用 | 可命中MUL_MAT耗时 | 可命中阶段耗时 | 零耗时Amdahl上界 |
|---|---:|---:|---:|---:|
| decode | 252/253 | 90.43% | 58.67% | 2.419× |
| kv_cache | 249/253 | 100.00% | 97.75% | 44.481× |
| text_prefill | 252/253 | 99.77% | 65.00% | 2.857× |
| vision_encode | 104/106 | 99.30% | 53.58% | 2.154× |
| vision_prefill | 747/759 | 100.00% | 90.68% | 10.731× |

“可命中”要求Q8_0权重、FP32激活/输出、二维非view、shape满足PL上限且至少一行能装入
64 MiB保留区。零耗时Amdahl只回答即使可命中部分瞬间完成，整个阶段最多能加速多少，
不是实际PL速度预测。

## 耗时最高的shape/拒绝原因

| 阶段 | 分类 | M | N | K | 调用 | CPU profile耗时 |
|---|---|---:|---:|---:|---:|---:|
| vision_prefill | eligible | 512 | 9728 | 2560 | 140 | 424.792 s |
| vision_prefill | eligible | 512 | 2560 | 9728 | 70 | 208.647 s |
| vision_encode | eligible | 4320 | 4096 | 1024 | 24 | 102.684 s |
| vision_encode | eligible | 4320 | 1024 | 4096 | 24 | 102.067 s |
| vision_prefill | eligible | 512 | 4096 | 2560 | 72 | 92.060 s |
| vision_prefill | eligible | 512 | 2560 | 4096 | 72 | 90.427 s |
| vision_encode | eligible | 4320 | 3072 | 1024 | 24 | 77.033 s |
| vision_prefill | eligible | 512 | 1024 | 2560 | 144 | 46.274 s |
| vision_encode | eligible | 4320 | 1024 | 1024 | 24 | 25.736 s |
| vision_prefill | eligible | 56 | 9728 | 2560 | 70 | 23.283 s |
| text_prefill | eligible | 49 | 9728 | 2560 | 70 | 20.377 s |
| vision_encode | eligible | 1080 | 4096 | 4096 | 4 | 16.935 s |
| vision_prefill | eligible | 56 | 2560 | 9728 | 35 | 11.435 s |
| vision_encode | eligible | 1080 | 2560 | 4096 | 4 | 10.591 s |
| text_prefill | eligible | 49 | 2560 | 9728 | 35 | 10.026 s |

## 边界

- 数据来自既有插桩日志，未重新运行模型，也未读取任何盲测精度数据。
- 本报告不证明生产后端已经调用PL；真实运行仍必须逐张量检查stride和view。
- 原始日志：`reports/2026-08-22-zcu102-grounding-operator-profile.log.gz`。
