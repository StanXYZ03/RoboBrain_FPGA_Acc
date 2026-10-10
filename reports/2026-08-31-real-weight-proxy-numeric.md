# RoboBrain真实Q8权重FFN代理数值门

日期：2026-08-31  
结论：`REAL_WEIGHT_PROXY_NUMERIC_PASS / RUNTIME_ACTIVATION_CAPTURE_NEXT`

## 范围

- 权重：真实`RoboBrain2.5-4B.Q8_0.gguf`第0层`ffn_norm/gate/up/down`。
- 输入：4个真实token embedding解量化后的行，作为确定性代理激活。
- 候选路径：RMSNorm→Q8_0→Gate/Up Q8×Q8→SiLU PWL→Q8_0→Down Q8×Q8→残差。
- 参考路径：相同真实Q8权重解量化，FP32激活与精确SiLU。
- 未加载CUDA，未访问板卡，未生成bitstream，未修改SD/QSPI。

代理输入不是第0层运行时隐藏状态，因此本门只验证真实权重数值可行性，不是完整模型精度门。

## 结果

| 用例 | max abs | RMSE | relative L2 | 结果 |
|---|---:|---:|---:|---:|
| 完整2560→9728→2560 FFN，4行 | 0.03017 | 0.002165 | 1.235% | PASS（门槛8%） |
| HLS真实权重切片：2560→64→64，4行 | 0.001716 | 0.000346 | 0.685% | PASS |

HLS C仿真原9项回归与新增真实权重切片共10/10通过。新增切片直接调用当前
`ffn_vector.cpp`，不是独立Python模拟结果。

## 证据

- 机器摘要：`fpga_accel/pl_resident_ffn_hls/artifacts/real_weight_proxy_20260831_v2/summary.json`
- 冻结切片：`fpga_accel/pl_resident_ffn_hls/artifacts/real_weight_proxy_20260831_v2/real_weight_proxy_slice.bin`
- HLS C仿真日志：`fpga_accel/pl_resident_ffn_hls/artifacts/20260830T174858Z_vitis_hls_csim.log`
- 摘要 SHA256：`269ebad274ae0c891ecdbb81228774c01a03699b1c9f5c92d67466f915a7e7ca`
- 切片 SHA256：`d7390a298088fa0cf048f3bb8ee18d953c881478ee857c3e32b276332c89b9b0`
- 仿真日志 SHA256：`4fcbad030a50f6857d72c3e1c8e42eafe1139ab64c53418debc521646e4a69c9`

下一门是CPU运行RoboBrain时捕获真实FFN输入隐藏状态，再冻结同样的全层差分结果；该门通过前
仍不生成新bitstream。
