# ZCU102 RAM-only 部署调试交接（2026-07-21）

## 目标与约束

- ZCU102 更换为 Micron DDR4-2666 16 GiB；模型为 RoboBrain2.5-4B Q8_0。
- 无 UART、远程操作；主机通过 `enp4s0` 直连 PS GEM。
- 不写 Flash，不修改/写入 SD；Linux 使用 JTAG + DDR + initramfs。

## 已通过项目

- DDR 动态 FSBL：SPD magic `0x53504434`，双 Rank，总容量 `17179869184` 字节。
- 16 GiB 稀疏地址测试通过（覆盖到 15 GiB），DDR testbench 通过。
- 以太网最小测试及 testbench 通过。
- ARM64 RAM Linux QEMU 完整启动通过：`RAM_LINUX_QEMU_PASS`。
- 模型：`RoboBrain2.5-4B.Q8_0.gguf`，4,693,672,576 字节，SHA256
  `835b78ff8c298acc358d682d3bf44d620d084a4ccd4c90da07e4e1afba3349eb`。
- 主机 llama.cpp smoke test 通过；ARM64 `llama-server --version` 在 QEMU 通过。

## 板端启动问题与定位

1. hw_server 2023.2 复位后必须先在 PSU 写 `0xFFCA0038 = 0x1FF`，才会出现可下载的
   `MicroBlaze PMU` 子目标。
2. FSBL 不应停在 `XFsbl_Exit` 函数入口；应停在 `0xFFFC943C`，让 TLB/I-cache/EL3
   cache 清理先完成。
3. 最初 JTAG 状态邮箱放在 `0xFFFEF100`，与 BL31 加载范围
   `0xFFFE0000..0xFFFFE000` 冲突；已移到设备树排除的 `0x7FF00000`。
4. A53 在 BL31 `plat_panic_handler` (`0xFFFEABD0`) 循环。OCM 栈回溯为：
   `bl31_early_platform_setup2 -> zynqmp_get_bootmode -> pm_mmio_read -> pm_ipi_send_sync`。
5. PMUFW 本身正常停在 `XPfw_Main` 事件循环，不是 PMUFW 崩溃。
6. 决定性发现：`BOOT_MODE_USER=0x0000000E`，板卡物理拨码为 SD 模式。
   BL31 因非零 boot mode 进入 `xbl_handover`，但纯 JTAG 手工加载没有 SD/XBL 分区交接表，
   因而 panic。此问题与 SD 内容无关。
7. `BOOT_MODE_USER` 在此阶段不可写。当前修复是在 RAM 中对固定的 Vitis 2023.2 BL31
   指令 `0xFFFEB0DC: 0x350002E0` 做受门禁补丁为 NOP `0xD503201F`，强制走
   preloaded-BL33/JTAG 路径。脚本会校验原值和写回值；不匹配即退出。

## 当前文件与未完成测试

- 主启动脚本：`deployment/scripts/run_ram_linux_jtag.tcl`。
- 只读诊断：`inspect_jtag_targets.tcl`、`inspect_bl31_panic.tcl`、`inspect_pmufw.tcl`、
  `read_boot_mode.tcl`。
- 状态邮箱：`0x7FF00000`；初值 `0x4A544147` (`JTAG`)，Linux PID 1 后写 `RBI*`，
  网络/SD 门禁阶段写 `RBB*`。
- 最后一次运行已在大文件 JTAG 下载阶段按用户要求中止，因此 BL31 NOP 补丁尚未完成板上验证。
- 下次从完整运行 `run_ram_linux_jtag.tcl` 开始；确认输出
  `RAM_BOOT_BL31_JTAG_PATCH=PASS`，再读取邮箱、ping `172.21.127.254`，查看
  `deployment/reports/board-runtime-nosudo.log`。
- 网络成功后才传约 5.15 GB 模型，运行 `host_chat_test.py`，最后运行
  `final_sd_check.py` 验证 SD 前后哈希一致。

## 主机网络参数

- 主机借用地址：`172.21.51.222`；强制绑定接口：`enp4s0`。
- 板端：`172.21.127.254/17`；日志端口 9000，模型分发 8000，API 8080，最终证明 9001。
- 不需要 sudo：主机 socket 使用 `SO_BINDTODEVICE(enp4s0)`，rp_filter 为 loose (`2`)。

## 本次结束状态

- 用户要求保存日志、复位板卡并结束。
- 当前 JTAG 下载已中止；随后执行系统复位；没有执行 Flash/SD 写操作。

## 2026-07-21 后续：无 UART RAM Linux 与 SD PetaLinux

### RAM Linux 新结论

- U-Boot 重定位地址固定为 `0x7FC85000`，`reloc_off=0x6FC05000`，重定位
  `main_loop=0x7FCB1BDC`。
- U-Boot RAM 探测会覆盖 `boot.scr`；正确流程是在 `main_loop` 断点后下载镜像，调用
  `invalidate_dcache_range(0x00100000, 0x21000000)`，再直接调用
  `image_source_script(0x20000000)`。
- 自制 `make_uboot_script.py` 原先错误地生成“header + 文本”。`IH_TYPE_SCRIPT` payload
  必须为“大端脚本长度、零终止长度表、脚本文本”；已修复。板端 format、HCRC、DCRC
  均返回 `1`。
- 修复后阶段字达到 `0x53435202` (`SCR02`)，`booti` 不返回。44 字节入口探针把邮箱写成
  `0x4B454E54` (`KENT`)，证明 Linux 已进入 `primary_entry`。
- 早期内存证明内核已创建 identity map、开启 MMU 并进入 `__primary_switched`：
  `boot_args x0=0x74994000`、`__fdt_pointer=0x74994000`、CPU mode `0xE12` (EL2)。
  但尚未到 PID 1；RAM Linux 早期内核问题仍待继续定位。

### 现成 SD PetaLinux 路线

- 物理启动模式始终为 SD：`BOOT_MODE_USER=0xE`。
- 单独 `rst -system` 会把 APU/DAP 留在 reset/AXI AP error；恢复顺序为顶层 DAP
  `rst -system` 后 `rst -dap`，再写安全门 `0xFFCA0038=0x1FF`。
- 新脚本 `run_sd_petalinux_jtag_bootstrap.tcl` 使用动态 16 GiB FSBL、PMUFW、BL31、U-Boot，
  只加载 control-DTB，不加载主机 Linux。U-Boot 可达 `main_loop`。
- 该路径的 U-Boot 能检测 SDHCI1 卡插入和供电，但 `mmc_init()` 返回 `-95`：
  `find_mmc_device(0)` 有效、`mmc_getcd=1`、`mmc1` 不存在；`mmc_get_op_cond()` 等待
  CMD8/ACMD41 响应超时。SDHCI1 Present State 为 `0x01FF0000`，3.3 V/时钟已启用。
- 完整运行官方 ZCU102 stock FSBL（`run_stock_fsbl_sd_boot.tcl`）可以从 SD 的 BOOT.BIN
  启动：`enp4s0` RX 从 `43424` 增至 `43713`，增加 289 包/约 98 KB。随后 DAP 被
  stock Linux/非 JTAG 启动链锁定，且板卡持续发 DHCP 流量。这是 SD PetaLinux 已运行的
  当前最强证据。
- 服务器 `enp4s0` 没有地址/DHCP 服务。当前用户无 `CAP_NET_RAW`，不能 tcpdump；
  `sudo -n` 需要密码；NetworkManager/PolicyKit 也拒绝创建共享连接。因此无法自动获得
  DHCP 地址。无需 UART 的下一步是用户在服务器终端手动配置临时 DHCP。

### 当前建议命令（用户终端输入 sudo 密码）

```sh
sudo ip address add 192.168.50.1/24 dev enp4s0
sudo dnsmasq --no-daemon --interface=enp4s0 --bind-interfaces --port=0 \
  --dhcp-authoritative \
  --dhcp-range=192.168.50.2,192.168.50.20,255.255.255.0,1h --log-dhcp
```

看到 `DHCPACK` 后记录地址并停止 dnsmasq；清理命令：

```sh
sudo ip address del 192.168.50.1/24 dev enp4s0
```

### 新增诊断脚本

- `run_sd_petalinux_jtag_bootstrap.tcl`、`run_stock_fsbl_sd_boot.tcl`

## 2026-07-21 DHCP 管理员配置后的复测

- 管理员已把服务器 `enp4s0` 配成 `192.168.50.1/24`，并在 tmux 中运行仅绑定
  `enp4s0` 的 `dnsmasq`，地址池为 `192.168.50.2-192.168.50.20`。
- 重新运行 stock SD FSBL 后，`enp4s0` RX 仍停在 `43713`，租约为空；说明现成 SD
  PetaLinux 尚未运行到网络阶段，问题不在 DHCP 服务。
- `run_sd_petalinux_jtag_bootstrap.tcl` 和 `run_stock_fsbl_sd_boot.tcl` 已加入
  `rst -system` 后的 `rst -dap` 恢复序列。
- 新增 `probe_uboot_dhcp.tcl`，在临时 U-Boot 内执行 `setenv autoload no; dhcp` 和
  `ping 192.168.50.1`，两条命令返回值均为 `0`。
- `dnsmasq` 租约确认板卡 MAC `00:0a:35:00:22:01` 获得 `192.168.50.6`；服务器反向
  `ping -I enp4s0 -c 3 192.168.50.6` 为 `3/3`、零丢包、平均约 `0.053 ms`。
- 结论：服务器 DHCP、PS GEM、PHY、网线和双向 IPv4 均已验证正常；当前
  `192.168.50.6` 属于易失的 JTAG U-Boot 环境，不代表 SD PetaLinux 已启动成功。
- `probe_uboot_mmc.tcl`、`probe_uboot_mmc_api.tcl`、`inspect_sdhci1_registers.tcl`
- `probe_stock_fsbl_sd_init.tcl`、`intercept_stock_sd_kernel.tcl`、
  `intercept_stock_sd_stages.tcl`
- `inspect_a53_runtime.tcl`、`inspect_kernel_early_memory.tcl`
