# GGML Q8_0 到 PL 运行时桥接门控

## 结论

项目自有桥接层通过软件契约门控，可把当前 GGML `MUL_MAT` 的连续 Q8_0 权重、
Q8_0/FP32 激活和 FP32 输出映射到既有 Q8×Q8 PL 运行时。不修改
`profiling_src/llama.cpp`；不满足条件时只返回明确原因，由原 CPU 路径回退。

决策：`GGML_Q8Q8_BRIDGE_CONTRACT_PASS / PRODUCTION_HOOK_AND_COVERAGE_NEXT`。

## 已确认的数据语义

GGML 的 `src0` 逻辑形状为 `[K,N]`，`src1` 为 `[K,M]`，输出为 `[N,M]`；
对应硬件数学形式：

```text
Y[M,N] = X[M,K] × W[N,K]^T
```

当前只接受二维、连续、未做 view/transpose/repack 的张量。Q8_0 块由一个 FP16
scale 和 32 个 INT8 组成，共 34 byte。FP32 激活按当前上游参考公式动态量化；
NaN/Inf 不送 PL。

## 验证结果

| 门控 | 结果 |
|---|---|
| 原生单测 | 4/4 PASS |
| 安全回退 | 8种不支持条件均拒绝 |
| ASan/UBSan | PASS |
| 端到端 mock | 12输出，`max_abs=0` |
| 上游块布局 | `QK=32, size=34, scale@0, payload@2` |
| 上游量化差分 | 19行、152块逐字节一致 |
| 公共 `ggml_tensor` 适配 | PASS，view明确回退 |
| AArch64交叉编译 | 两个测试程序均通过；未上板执行 |

机器汇总：
`fpga_accel/ggml_q8q8_bridge/artifacts/ggml_bridge_summary_20260830.json`。

冻结日志位于：
`fpga_accel/ggml_q8q8_bridge/artifacts/ggml_bridge_20260830_final/`。

## 证据边界

本阶段证明格式、矩阵方向、量化和调度接口相容，不证明完整 llama.cpp 已调用 PL，
也不证明模型端到端精度或速度提升。下一阶段先增加默认关闭的生产调用入口和逐 shape
命中/回退统计；只有真实模型图覆盖率明确后，才选择第一个完整 Transformer Block 接入点。
