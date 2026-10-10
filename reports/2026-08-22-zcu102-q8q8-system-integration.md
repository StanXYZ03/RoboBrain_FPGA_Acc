# ZCU102 Q8×Q8 DATAFLOW 系统集成报告（2026-08-22）

## 结论

已将通过 C/RTL 仿真的 `Q8_0 × Q8_0` DATAFLOW GEMM 封装为 Vivado IP，并完成 ZCU102 的 PS、AXI、PL 加速器和 PS DDR 系统级集成与布局布线。199.98001 MHz 实际配置下满足 200 MHz 时序约束：WNS `+0.439 ns`、WHS `+0.010 ns`、未布线网络 `0`。

本阶段没有生成 bitstream、XSA 或启动镜像，也没有烧录板卡。

## 系统结构

| 功能 | AXI 路径 | 用途 |
|---|---|---|
| 寄存器控制 | A53 `M_AXI_HPM0_FPD` → SmartConnect → `s_axi_control` | 启动、尺寸、DDR 地址和状态；基地址 `0xA0000000` |
| Q8 权重读 | `m_axi_gmem_wq` → SmartConnect → `S_AXI_HPC0_FPD` | 独立 128-bit DDR 读通路 |
| Q8 激活读 | `m_axi_gmem_xq` → SmartConnect → `S_AXI_HPC1_FPD` | 独立 128-bit DDR 读通路 |
| scale 与输出 | 3 个 AXI master → SmartConnect → `S_AXI_HP0_FPD` | FP16 scale 读取和 FP32 结果写回 |
| PL 时钟 | PS 99.990005 MHz → MMCM ×12/÷6 | 配置输出 199.98001 MHz |

HPC0/HPC1 是支持一致性的高性能端口，但是否产生一致性事务仍取决于软件映射和 AXI cache 属性；当前不能仅凭连线宣称已实现缓存一致性。

## DDR 地址空间

- DDR low：`0x0000000000000000` 起，2 GiB 映射。
- DDR high：`0x0000000800000000` 起，32 GiB AXI aperture。
- 因而硬件地址宽度可覆盖更换后的 16 GiB DIMM；有效物理容量仍由启动阶段已验证的 SPD 感知 DDR 初始化决定。
- Vivado 的官方 ZCU102 board preset 描述默认板级连接，不等于重新训练或验证 16 GiB DIMM。

## 验证链

| 层级 | 结果 |
|---|---|
| C testbench | 6/6 通过，包含小矩阵、边界尺寸、RoboBrain `K=2560`、视觉 `K=1024` 和非法输入 |
| RTL XSim | 5/5 transaction 通过，数值后检查通过 |
| HLS IP OOC 布局布线 | 5.000 ns 约束，关键路径 4.127 ns；WNS `+0.873 ns`、WHS `+0.039 ns` |
| Vivado Block Design | 地址分配和 `validate_bd_design` 通过 |
| 完整 PS+AXI+PL 布局布线 | WNS `+0.439 ns`、WHS `+0.010 ns`、0 未布线网络 |
| 方法学检查 | 原 `TIMING-2/TIMING-4` 已消除；仅剩 2 条 MMCM 布局 Advisory |

当前验证证明算子逻辑、AXI 结构和 200 MHz 可实现性；尚未执行 PS 软件驱动参与的系统级 AXI 功能仿真，因此上板前仍需完成 host/driver 的寄存器与 DMA 事务测试。

## 布局布线资源

| 范围 | LUT | FF | BRAM tile | DSP |
|---|---:|---:|---:|---:|
| 加速器 OOC | 15,549 | 20,784 | 21.5 | 308 |
| PS+AXI+加速器完整系统 | 19,947 | 28,539 | 21.5 | 308 |

HLS 综合阶段曾估算 179 DSP；Vivado 实际布局布线为 308 DSP。后续论文和资源预算必须采用 Vivado 实现值。

## 功耗口径与待优化项

- Vivado vectorless 估算：片上总功耗 4.250 W，动态 3.523 W，静态 0.727 W；其中 PS8 估算 2.748 W。
- 该结果没有输入真实推理切换活动，且报告提示 reset 活动不可靠，因此不能替代上板后的电源轨实测。
- DRC 只有性能类告警：`DPIP-2=37`、`DPOP-3=48`、`DPOP-4=162`。主要指向 DSP 输入、乘法器和输出寄存器不足，是下一轮改善频率/能效的明确位置，不影响本次 200 MHz 收敛。

## 下一道上板门槛

1. 编写 PS 端寄存器控制、连续物理内存分配、cache flush/invalidate 和超时恢复代码。
2. 用确定性小矩阵建立软件 golden result，并先完成可执行的 AXI 事务测试。
3. 通过后才能生成 bitstream/XSA；上板先运行小矩阵并比对逐元素结果，再接入模型算子。

IP 包 SHA-256：`542170b371b901b63821d680b6134e165ba6b1949576cecff33005635747ed40`。

