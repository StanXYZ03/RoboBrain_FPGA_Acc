# PL驻留FFN向量核：HLS与布线后验证

日期：2026-08-31  
结论：`PL_RESIDENT_VECTOR_IP_ROUTE_PASS / FULL_SYSTEM_INTEGRATION_NEXT`

## 实现范围

统一、任务无关的HLS核覆盖RMSNorm、LayerNorm、SiLU门控乘、GELU、残差加及Q8_0重
量化。SiLU/GELU采用`[-8,8]`内步长0.25的分段线性表。接口由7个AXI主口压缩到3个：
双输入各保留独立读通道，结果使用独立写通道，参数在计算前复用第一读通道加载到BRAM。

## 数值门

9/9测试通过。独立算子最大误差为：RMSNorm 0.007011、LayerNorm 0.006794、SiLU门控
0.028351、GELU 0.026097、残差加0。组合子图对独立FP32参考的结果为：

| 子图 | 流程 | 最大绝对误差 | RMSE |
|---|---|---:|---:|
| LLM FFN | RMSNorm→Gate/Up→SiLU×Up→Down→Residual | 0.000711 | 0.000304 |
| Vision FFN | LayerNorm→FC1→GELU→FC2→Residual | 0.002926 | 0.001043 |

测试使用小规模确定性Q8矩阵，只证明运算链与数据格式正确，不等同于冻结RoboBrain模型精度。

## 综合与布局布线

目标200 MHz；全部循环约束满足。FP32统计累加按真实反馈延迟设`II=5`，参数双向量共享总线
预载设`II=2`，激活与重量化内层均为`II=1`。

| 指标 | 结果 |
|---|---:|
| 布线后周期 / 等效Fmax | 4.198 ns / 238.21 MHz |
| WNS / WHS | +0.802 ns / +0.023 ns |
| 未布线网络 | 0 |
| LUT / FF | 26,685 / 47,063 |
| DSP | 154 |
| BRAM Tile | 64 |

与已布线DATAFLOW完整系统简单相加后约为19.21% LUT、14.14% FF、32.34% DSP、
52.96% BRAM Tile，低于80%门限，但该结果未计入新增SmartConnect和布线开销，不能代替完整
系统实现。

## 证据与限制

- IP：`fpga_accel/pl_resident_ffn_hls/artifacts/pl_resident_ffn_vector_ip.zip`，SHA256
  `0a8959b1ea6478f228608fe20df4deb3fe6fc566bd5e8b10d19f88095b786ba0`。
- 机器摘要：`fpga_accel/pl_resident_ffn_hls/artifacts/pl_resident_ffn_vector_summary.json`。
- 原始日志：`fpga_accel/pl_resident_ffn_hls/artifacts/20260830T165327Z_vitis_hls_export_impl.log`。
- 本阶段未生成系统bitstream、未访问或配置板卡，也未改动SD卡。
- 下一门：把向量核与PE16 DATAFLOW GEMM同时接入ZynqMP完整BD，完成地址、DRC、资源和
  200 MHz布局布线；之后再做真实RoboBrain张量精度门。
