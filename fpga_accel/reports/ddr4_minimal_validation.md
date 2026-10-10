# ZCU102 PS DDR4 最小验证状态

日期：2026-07-20

## 已完成

- JTAG cable：Digilent JTAG-SMT2NC `210308A7A408`
- Device：Zynq UltraScale+ `xczu9`
- PS targets：PMU、R5、4× Cortex-A53 均已枚举
- 动态 PS 配置：DDR4、64-bit、8× x8、双 Rank、总容量 16GiB
- DDR PHY 初始化执行完成
- 八个稀疏地址写入/回读正确，覆盖 0.25、1.25、2、3、4、8、12、15GiB
- 测试算法 testbench 通过：健康内存通过，地址镜像和单 bit 损坏均被检出
- A53 裸机 ELF 已构建到 OCM，不依赖 DDR 启动
- OCM ELF 完整 16GiB 测试通过：16 次检查、0 次失败

## 板端结果

早期直接通过 DAP 重复读取高 DDR 窗口时曾出现 transaction timeout，DAP 进入
sticky error `0x30000021`；该状态已通过 `rst -dap` 清除。改用运行于 OCM 的 A53
测试程序并关闭缓存后，动态 FSBL 从 SPD 得到双 Rank、每 Rank 8GiB，并得到：

```text
DDR16_SPD magic=0x53504434 ranks=2 rank_bytes=8589934592 total_bytes=17179869184
DDR16_DDRC mstr=0x43040010 active_ranks=0x3 addrmap0=0x00000018 rank_address_bit=0x18
DDR16_STATUS magic=0x44445234 phase=0x0000600D checks=16 failures=0
DDR16_RESULT=PASS
```

因此可以确认：动态控制器配置识别到完整 16GiB 双 Rank 容量，低 2GiB窗口和高
14GiB窗口中的八个稀疏位置未发现地址镜像或读写错误。该结果证明了完整地址范围的
最小功能，不等同于长时间、全地址或温度覆盖压力测试。

## 远程判定方式

不观察 LED 或屏幕。板端 ELF 将结果写入 OCM `0xFFFEF000`，XSCT 脚本轮询并把
`magic`、`phase`、检查次数、失败次数和最后错误地址保存到
`reports/ddr16_dynamic_test.log`。`DDR16_RESULT=PASS` 是唯一通过条件。

本流程不配置 PL、不烧写 flash，只向 OCM 临时下载 ELF；断电后消失。
