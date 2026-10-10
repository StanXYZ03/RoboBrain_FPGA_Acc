# ZCU102 Q8×Q8 加速器离线包与首次上板记录

日期：2026-08-23  
范围：bitstream、XSA、匹配 FSBL、UIO RAM Linux、自检与 JTAG 启动链。  
状态：**离线门禁与易失性 JTAG 上板数值验证均完成；未写入 SD/QSPI。**

## 1. 结论

- 已从通过验证的 `Q8_0 × Q8_0` DATAFLOW HLS IP 生成完整 ZCU102 PS+AXI+PL bitstream 与 XSA。
- 200 MHz 约束下布局布线通过：实际 `199.98001 MHz`，WNS `+0.439 ns`，WHS `+0.010 ns`，未布线网络 `0`。
- 独立 `.bit` 与 XSA 内嵌 `.bit` 的 SHA-256 完全相同。
- 已由该 XSA 生成匹配的 JTAG 专用 FSBL；保留 PS 时钟、DDR、DDR PHY、PS–PL AXI 和复位初始化。
- 已构建启用内建 UIO 驱动的 ARM64 RAM Linux；设备树与 initramfs 离线校验通过。
- 全包校验脚本输出：`Q8Q8_BUNDLE_PASS ... board_programmed=no`。
- 板上 16 GiB SPD/双 rank 门禁、PL 配置、PS–PL 隔离/复位释放、AXI-Lite `ap_idle` 门禁全部通过。
- `M=2, N=3, K=32` 数值自检首次运行 `83 us`，随后连续 5 次为 `75/79/79/80/76 us`（均值 `77.8 us`）；每次 `max_abs=0`。

## 2. 前置仿真与实现门禁

| 层级 | 结果 |
|---|---|
| C testbench | 6/6 PASS |
| HLS Verilog C/RTL co-sim | 5/5 transaction PASS，数值后检查 PASS |
| PS 运行时单元测试 | 3/3 PASS |
| ASan/UBSan | 3/3 PASS |
| 寄存器表一致性 | 10/10 PASS |
| HLS IP OOC 实现 | 200 MHz，WNS `+0.873 ns`，WHS `+0.039 ns` |
| 完整 PS+AXI+PL 实现 | 200 MHz，WNS `+0.439 ns`，WHS `+0.010 ns`，0 未布线网络 |
| RAM Linux 离线包 | UIO、DT、initramfs、校验和全部 PASS |
| ZCU102 数值自检 | 首次 + 连续 5 次全部 PASS，`max_abs=0` |

布局布线资源：`19,947 LUT`、`28,539 FF`、`21.5 BRAM tile`、`308 DSP`。Vivado vectorless 估算总功耗 `4.250 W`，其中动态 `3.523 W`；该数值不是板上实测功耗。

## 3. 匹配启动链

1. PMUFW 启动并释放 ZynqMP 初始化门禁。
2. 从当前 XSA 生成的 `q8q8_fsbl.elf` 初始化 PS、16 GiB DDR、PL 时钟及 PS–PL AXI 端口。
3. 读取 FSBL 的 SPD 状态区，要求 magic `0x53504434`、双 rank、总容量 `17179869184` 字节。
4. 通过 JTAG 把匹配的 `q8q8_zcu102.bit` 临时配置到 PL，不写 SD/QSPI。
5. 调用同一 FSBL ELF 中由当前 XSA 生成的 `psu_ps_pl_isolation_removal_data()` 与 `psu_ps_pl_reset_config_data()`，再读取 AXI-Lite 控制口，要求 `ap_idle=1`。
6. 启动 ATF、U-Boot、RAM Linux；Linux 用 `generic-uio` 暴露控制口和 DDR DMA 区。
7. 专用 `robobrain.pl_selftest=1` 模式跳过模型下载，板端立即运行确定性 `M=2, N=3, K=32` 自检，期望输出 `[64, 0, -32, -96, 0, 48]`，随后保持 UART shell。

FSBL 为 JTAG 专用构建，启用 Xilinx 的 `FSBL_PARTITION_LOAD_EXCLUDE`：完成 PS/DDR 初始化后，无论物理拨码为何值都进入 JTAG 停留路径，不访问或改写 SD。它以当前 XSA 的 `psu_init.c` 为基础，因此保留 PL 所需的 AFI/AXI 配置；随后应用此前已在该 16 GiB DIMM 上实测通过的两处动态 DDR 补丁：按 SPD 几何计算双 rank 地址位，并把 rank/容量证据写入 OCM。由于分区加载被裁掉，标准 FSBL 原本在位流后执行的 PS–PL 隔离解除和 PL 复位释放也会被跳过；JTAG 脚本现从最终 ELF 元数据读取这两个生成函数的地址并显式调用，随后用 `0xA0000000 == 0x00000004` 作为上 Linux 前的 AXI 可达门禁。

## 4. 板上故障定位与修复

第一次 Linux 自检在访问控制寄存器时使总线阻塞。原因不是位流配置失败，而是裁剪 FSBL 分区加载后漏掉了位流后的 PS–PL 隔离解除和复位释放。补齐官方生成函数后，JTAG 读取控制口得到 `0x00000004`，`ap_idle` 门禁通过。

第二次自检在 `prepare` 阶段触发 `SIGBUS`。板端 `devmem` 对 `0xB7C000000` 的 32 位写入与回读一致，证明高地址 DDR 可用；根因是 UIO 将保留 DDR 映射为 ARM 非缓存设备内存，而优化的 `memset/memcpy` 可能使用宽向量访问或 `DC ZVA`。共享区访问改为显式 `volatile` 字节事务后，AArch64 反汇编确认使用 `STRB/LDRB`，板端数值自检稳定通过。

## 5. Linux 与 DDR 映射

| 对象 | 配置 |
|---|---|
| Linux 内存描述 | low `0x0–0x7FEFFFFF`；high `0x800000000–0xB7FFFFFFF` |
| 保留 DMA 区 | `0xB7C000000–0xB7FFFFFFF`，64 MiB，`no-map` |
| AXI-Lite 控制口 | `0xA0000000–0xA000FFFF` |
| 驱动 | `CONFIG_UIO=y`，`CONFIG_UIO_PDRV_GENIRQ=y` |
| UIO 名称 | `q8q8-gemm` |
| 板端程序 | `/opt/robobrain/q8q8-selftest`，AArch64 ELF |

RAM Linux 构建锁定 `linux-xlnx a19da02cf5b44420ec6afb1eef348c21d9e8cda2` 与 `BusyBox 1a64f6a20aaf6ea4dbba68bbfa8cc1ab7e5c57c4`。与旧 base-platform PL 外设对应的 `/amba_pl@0` 子树保持禁用；Q8Q8 UIO 是 overlay 添加的独立根节点，不受该禁用项影响。

## 6. 关键产物

| 产物 | SHA-256 |
|---|---|
| HLS IP ZIP | `542170b371b901b63821d680b6134e165ba6b1949576cecff33005635747ed40` |
| bitstream | `ee84fb8b60ad6143bb9adc5402d7b3a0331934741c142adf6fd20d64666a3499` |
| XSA | `cd4fd7e8c307f8dc0546033ad17af17ebb1c4440d61d21a9d9a0935358684087` |
| routed DCP | `974f21e7b12404144ebdffad1c70ab7251ea8e6bfb2ff714084441d5939af25c` |
| FSBL | `3693d9e1fe672519b2379503af29bd130fc74c36440b2f6c52a97041d70876da` |
| RAM Linux Image | `fa7ac626d0b616e4a644577b7ff2ed2dea2b0821e0d5767b350a718d2731470e` |
| 加速版 DTB | `4c9f0a0f1bfb90243143190c041b95a7880a84c51d6ea59df5695f3550e47cc6` |
| initramfs | `4e136c048e240fa3218e9e3dad14c128b99ad66e9f7c9c1674dc0f15828b6d26` |
| 板端自检日志 | `0f94e2cbd88cca5eb5ed4f5778c70d02b6fa7a908400b0db11128afc56efb7be` |

全量清单见 `deployment/artifacts/q8q8_accel_bundle.sha256`。离线复验命令：

```bash
sh deployment/scripts/verify_q8q8_accel_bundle.sh
sha256sum -c deployment/artifacts/q8q8_accel_bundle.sha256
```

## 7. 已证明与尚未证明的边界

- 已证明：16 GiB 动态初始化、真实 PL 配置、UIO 枚举、AXI-Lite 可达、高地址 DDR 小范围访问，以及独立 Q8×Q8 GEMM 小矩阵数值正确性和重复可运行性。
- 当前是独立 Q8×Q8 GEMM 自检，不是 llama.cpp/RoboBrain 算子替换；不能据此宣称模型已经获得加速。
- 当前 `75–83 us` 仅对应 `M=2,N=3,K=32`，不能外推端到端模型吞吐。
- 当前功耗仅为 Vivado vectorless 估算；温度节点在该最小内核中未暴露，板上功耗、温度和大矩阵吞吐仍须专项实测。
- 下一步可以进入大矩阵基准与 llama.cpp/RoboBrain 算子接入，但接入前应先解决通用 DMA 缓冲管理，避免大张量采用逐字节 UIO 搬运。

## 8. 工作副本位置

本轮产物暂存于 Codex 可写工作副本：
`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`。
同步到主工程 `/home/zhaoxiangyu/RoboBrain2.5` 因当前 Codex 用量限制未执行；不得把主工程中的旧文件误认为本轮匹配产物。
