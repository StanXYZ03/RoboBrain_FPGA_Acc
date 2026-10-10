# 真实运行时激活上的 FFN 数值门槛

## 结论

判定：`RUNTIME_FFN_NUMERIC_PASS`。

从 llama.cpp 实际推理图中捕获 `ffn_inp-0`，其位置是第 0 层注意力残差相加之后、FFN RMSNorm 之前。捕获张量包含 2 行、每行 2560 个 FP32 元素，数据全部有限；本次没有用随机激活替代真实运行数据。

候选路径为：RMSNorm → 激活 Q8_0 → gate/up Q8×Q8 → 分段线性 SiLU → 中间结果 Q8_0 → down Q8×Q8 → 残差。参考路径使用同一组已解量化的真实 Q8_0 权重、FP32 激活和精确 SiLU，因此本门槛衡量的是拟上 PL 数值路径相对当前 Q8 模型的附加误差，而不是相对原始 FP16 模型的量化误差。

## 结果

| 指标 | 结果 | 门槛 |
|---|---:|---:|
| 最大绝对误差 | 0.039874 | — |
| RMSE | 0.002889 | — |
| 相对 L2 误差 | 1.372% | ≤ 8% |
| 第 0 行相对 L2 | 1.503% | ≤ 8% |
| 第 1 行相对 L2 | 1.255% | ≤ 8% |

## 证据

- 机器可读结果：`fpga_accel/runtime_activation_capture/artifacts/runtime_ffn_numeric_20260901/summary.json`
- 捕获张量：`fpga_accel/runtime_activation_capture/artifacts/runtime_activation_capture_20260901/ffn_inp_0.bin`
- 捕获 SHA-256：`345268b2265f3fdfa404a9391bc5c017aac3a97d644b5d1e25d8f7059dcb3922`
- 模型：`RoboBrain2.5-4B.Q8_0.gguf`
- 模型 SHA-256：`835b78f173a8229f4a17d9f4861662eb219f28a6ea76416abbce57ce28ef49eb`

## 适用范围

本结果只覆盖第 0 层、一个文本提示产生的 2 行 FFN 输入。它尚不能代表全部 36 层、视觉输入、不同任务提示或端到端任务精度，也不是 FPGA 板端性能结果。下一轮数值验证应扩展到多层、多提示及图像任务，并在冻结的公开校准集上检查最终任务输出。
