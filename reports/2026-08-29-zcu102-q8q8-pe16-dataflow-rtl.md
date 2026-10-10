# ZCU102 Q8×Q8 PE16 DATAFLOW RTL门控

生成时间：2026-08-29T13:28:29.053983+00:00
状态：`DIRECTED_RTL_AND_DATAFLOW_STRUCTURE_PASS / EXPORT_IP_NEXT`

## 定向XSim

- HLS生成的融合初始化计算控制RTL通过：16个PE bank各写24次，覆盖12个合法地址，每地址写2次。
- 测试shape为`M=3,N=64,K=64`；完成时间807500 ps。
- 算术单元使用行为stub，本项证明启动/完成握手、bank选择、地址遍历和`kb=0`直接写；数值由9组C testbench覆盖。
- 沙箱内XSim会在时间仿真入口发生工具内部异常；相同snapshot在沙箱外运行通过。失败和通过目录均保留。

## DATAFLOW生成结构

- RTL中同时存在load、compute、store三个独立process。
- 激活Q FIFO：256 bit × 4864；激活scale FIFO：32 bit × 4864；输出FIFO：128 bit × 256。
- 静态连接检查确认`load → activation FIFO → compute → output FIFO → store`完整。

## 证据边界

没有重试已知会产生约30.7 GiB模型并有OOM风险的顶层AXI cosim；本阶段也没有导出bitstream或访问板卡。下一步先导出独立HLS IP，再进入Vivado布局布线门控。

机器可读证据：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/fpga_accel/q8q8_pe16_dataflow_hls/artifacts/pe16_dataflow_rtl_summary.json`。
