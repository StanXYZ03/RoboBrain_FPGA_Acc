# ZCU102 Q8×Q8 加速器：PS 运行时与上板前验证记录

日期：2026-08-23  
范围：PS 控制、DDR 物理缓冲区、确定性自检及 HLS RTL 联合仿真。**未生成位流，未烧录或复位板卡。**

## 1. 结论

- PS 端运行时、自检程序与设备树集成已实现。
- 软件单元测试 3/3 通过；ASan/UBSan 3/3 通过。
- AArch64 交叉编译通过。
- 16 GB 设备树地址图及 64 MiB 保留区合并检查通过。
- 运行时的 10 个寄存器偏移与 HLS IP 自动导出头文件完全一致。
- Verilog C/RTL 联合仿真通过：5/5 事务完成，数值后检查全部通过。
- 结合此前完整 PS+AXI+DDR 系统实现结果（200 MHz，WNS +0.439 ns，WHS +0.010 ns，0 未布线网络），已满足进入受控上板自检前的软件和仿真条件。

## 2. PS—PL 执行路径

1. Linux 将 16 GB DDR 顶部 64 MiB `0xB_7C00_0000–0xB_7FFF_FFFF` 标记为 `no-map`，避免被页分配器使用。
2. `generic-uio` 暴露两个映射：控制寄存器 `0xA000_0000`（64 KiB）和上述 DDR 测试区。
3. 用户态程序在非缓存 UIO 映射中准备 Q8_0 权重、Q8_0 激活、FP16 scale 与 FP32 输出区；传给 PL 的是物理地址，而非普通进程虚拟地址。
4. 驱动先同步 5 个缓冲区，再写入五个 64 位地址和 `M/N/K`，执行内存屏障后置 `ap_start`。
5. 驱动轮询 `ap_done`，读取 `ap_return`，同步并逐元素核对输出。
6. 超时后锁存 fault，禁止直接再次启动；只有 IP 已回到 idle 才允许软件恢复，否则必须复位 PL 域。超时不能被解释为已安全终止在途 AXI 事务。

## 3. 地址与接口

| 对象 | 地址/范围 | 连接 |
|---|---:|---|
| AXI4-Lite 控制 | `0xA000_0000` | PS `M_AXI_HPM0_FPD` |
| Q8 权重 / FP16 权重 scale | DDR 高地址 | PS `S_AXI_HPC0_FPD` |
| Q8 激活 / FP16 激活 scale | DDR 高地址 | PS `S_AXI_HPC1_FPD` |
| FP32 输出 | DDR 高地址 | PS `S_AXI_HP0_FPD` |
| 自检保留区 | `0xB_7C00_0000–0xB_7FFF_FFFF` | 64 MiB、非缓存 UIO 映射 |

控制寄存器：`AP_CTRL 0x00`、`AP_RETURN 0x10`、`weights_q 0x18`、`weights_scale 0x24`、`activations_q 0x30`、`activations_scale 0x3c`、`output 0x48`、`M 0x54`、`N 0x5c`、`K 0x64`。

## 4. 确定性自检向量

- 尺寸：`M=2, N=3, K=32`。
- scale：全部为 FP16 `1.0`（编码 `0x3c00`）。
- 预期 FP32 输出：`[64, 0, -32, -96, 0, 48]`。
- 地址位于 `0xB_7C00_0000` 以上，可同时验证 64 位地址高 32 位的写入。

软件 mock 自检结果：精确相等、14 次寄存器写、5 次 device 方向同步、1 次 host 方向同步；非法尺寸/对齐被拒绝；50 µs 超时、fault 锁存与 idle 恢复均通过。

## 5. 验证分层与边界

| 层次 | 覆盖内容 | 结果 |
|---|---|---|
| C 单元测试 | 寄存器顺序、64 位地址、数值、非法输入、超时 | 3/3 PASS |
| ASan/UBSan | 主机端内存与未定义行为检查 | 3/3 PASS |
| AArch64 交叉构建 | 板端 ELF 可生成 | PASS |
| DT 合并测试 | 16 GB memory、64 MiB no-map、两个 UIO reg | PASS |
| 寄存器表一致性 | 软件常量与 HLS 导出头文件 | 10/10 PASS |
| HLS Verilog C/RTL co-sim | AXI master RTL、控制协议、数值后检查 | 5/5 PASS |
| PS+AXI+DDR Vivado 实现 | 结构、寻址、布线、200 MHz 时序 | PASS |

以上不是一个包含 Linux 的单体全系统仿真：软件 mock 验证控制协议，HLS co-sim 验证实际 RTL AXI master 与数值，Vivado 实现验证互连和时序。真实 DDR 信号完整性、板上 UIO 枚举及物理数据通路仍必须由最终上板自检确认。

## 6. 后续上板验收步骤（暂未执行）

1. 只使用与当前 IP、地址图和设备树完全匹配的硬件产物。
2. 经 UART 保留 shell，启动带 `UIO=y`、`UIO_PDRV_GENIRQ=y` 和 overlay 的 RAM Linux。
3. 检查 `/sys/class/uio/uio*/name` 与两个 map 的物理地址/大小。
4. 运行 `/opt/robobrain/q8q8-selftest --timeout-ms 5000`。
5. 只有出现 `SELFTEST_PASS m=2 n=3 k=32 outputs=6 max_abs=0` 才判定成功；超时或 AXI 异常后不重启任务，先复位 PL 域并保存 UART 日志。
