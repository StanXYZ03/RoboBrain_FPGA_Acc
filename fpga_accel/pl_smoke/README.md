# ZCU102 PL 最小可用性验证

本工程提供两种 PL 最基本功能验证：

1. **首选：JTAG/VIO 闭环验证。** PL 使用内部 50 MHz `CFGMCLK`，运行
   32 位心跳计数器，并计算 `response = challenge XOR 0xA5C35A7E`；服务器
   经 JTAG 写入 challenge、读回 response 和 heartbeat，不依赖 LED、PL UART
   或外部 PL 时钟。
2. **备选：PL UART 验证。** PL 使用板载 300 MHz SI570 差分时钟，以
   115200 baud 周期发送 `PL_SMOKE_PASS\n`，同时驱动 LED0 心跳。

远程环境以 JTAG/VIO 闭环为主要验收条件：至少两个 challenge 的 response
完全正确，并且两次采样间 heartbeat 发生变化。

工程不包含 PS，不改动 SD 卡和 Flash。通过 JTAG 下载 bitstream 只临时配置 PL，断电或重新配置后消失。

## 验证顺序

```text
RTL testbench
→ synthesis
→ implementation/timing/DRC
→ JTAG 临时配置
→ VIO 挑战响应与心跳回读
```

主要脚本：

- `scripts/build_jtag_bitstream.tcl`：综合、实现、bitstream 与 `.ltx` 生成。
- `scripts/program_and_test_jtag_vio.tcl`：JTAG 临时下载并自动判定 PASS/FAIL。
- `artifacts/jtag/validation_result.md`：最近一次板端验证结果。
