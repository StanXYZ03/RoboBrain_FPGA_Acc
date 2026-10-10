# PL驻留FFN双IP完整系统布局布线

日期：2026-08-31  
结论：`PL_RESIDENT_FULL_SYSTEM_ROUTE_PASS / REAL_TENSOR_NUMERIC_GATE_NEXT`

## 结果

现有PE16 DATAFLOW Q8×Q8 GEMM与新统一FFN向量核已集成到同一个ZCU102 ZynqMP系统。
GEMM控制地址保持`0xA0000000`，向量核使用`0xA0010000`；两个核共享PS DDR，但通过
HPC0、HPC1与HP0 SmartConnect分配读写通道。

| 门控 | 结果 |
|---|---:|
| 目标频率 | 200 MHz |
| WNS / WHS | +0.060 ns / +0.009 ns |
| TNS / THS | 0 / 0 ns |
| 未布线网络 | 0 |
| DRC Error / Methodology Error | 0 / 0 |
| bitstream / 板卡访问 / SD修改 | NO / NO / NO |

| 资源 | 完整系统 | XCZU9EG占比 | 相对仅DATAFLOW系统增量 |
|---|---:|---:|---:|
| LUT | 58,445 | 21.32% | +32,479 |
| FF | 88,156 | 16.08% | +57,683 |
| BRAM Tile | 483 | 52.96% | +64 |
| DSP | 815 | 32.34% | +154 |

层次报告中，向量核占26,638 LUT、47,022 FF、64 BRAM Tile和154 DSP；增量LUT/FF还包含
新增AXI与控制互连。所有主要资源低于80%门槛。

Vivado无激励功耗估计为5.942 W（Medium confidence），相比仅DATAFLOW系统增加1.258 W。
该值依赖工具假定的翻转率，不是板级功耗实测，不能用于宣称能效优势。

## 当前边界

- 已证明两个IP能在完整PS+AXI+PL系统中共同实现并满足200 MHz时序。
- 尚未用冻结RoboBrain真实张量通过数值门；现有组合测试是确定性小矩阵。
- 两个IP目前由PS顺序启动，尚无PL命令队列或自主完整FFN控制器。
- 尚未生成bitstream、未上板，因此不宣称Block或模型端到端加速。

## 证据

- 机器摘要：`fpga_accel/pl_resident_ffn_zcu102_vivado/artifacts/pl_resident_ffn_system_summary.json`
- 实现目录：`fpga_accel/pl_resident_ffn_zcu102_vivado/artifacts/runs/20260830T171212Z_impl`
- 路由DCP SHA256：`8cb2c550cdcbafcbb32ca344141b1b91123b3fa73e2b6174739c46e06159ec2d`
- 时序报告 SHA256：`344ebdef5d29fff8ffb8be61320e96ffb37517abc085225eaca2150cccf6bf6c`
- 资源报告 SHA256：`a1eec37c2bd4e77f03d00af0924c0542cd3bb1d63b43661b0588c2d6b2ebf1cf`

下一步先建立真实RoboBrain FFN张量的只读导出、冻结与CPU差分门，未通过前不生成新上板包。
