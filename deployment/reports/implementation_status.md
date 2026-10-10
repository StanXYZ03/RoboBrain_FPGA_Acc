# RoboBrain2.5-4B Q8_0 ZCU102 部署实现状态

日期：2026-07-20

## 已完成并验证

- llama.cpp 固定到 `178a6c44937154dc4c4eff0d166f4a044c4fceba`，源码包含
  `qwen3vl` 和 Qwen3-VL mmproj 支持。
- x86_64 `llama-server`/`llama-cli` 构建通过，版本 `10069 (178a6c44)`。
- Cortex-A53 ARMv8-A CPU-only 交叉编译通过；未启用 dot-product、SVE、CUDA、
  Vulkan 或 OpenMP，避免 ZCU102 非法指令和额外动态库依赖。
- ARM64 `llama-server --version` 已通过 Vitis QEMU user-mode 执行。
- ARM64 二进制 SHA-256：
  - `llama-server`: `eb2420264bf6201e0f2879e55bdd75583620de16325cc18b6f63f5991467c132`
  - `llama-cli`: `3a77e62913f8ef45724110f6febb9ab7b9d3aa57fe4fd8ed7ef72b12e9a38a86`
- 部署单元/安全测试：7 项全部通过。
- DDR testbench：`TB_RESULT=PASS`；以太网协议 testbench：`TB_ETH_RESULT=PASS`。
- 既有板端证据：双 Rank 16GiB、8 个稀疏位置覆盖到 15GiB，
  `DDR16_RESULT=PASS`；PS GEM3 物理以太网测试 `ETH_RESULT=PASS`。
- 2026-07-20 再次只读枚举 JTAG：PMU、PL、PSU、RPU、APU 和四个 A53 均在线，
  A53 #0 正在运行。未复位、未烧录、未启动 SD。

## SD 零写入实现

- 只能从 RAM initramfs 进入部署流程；发现任一 SD 分区以 `rw` 挂载或 SD swap
  已启用时立即失败。
- 挂载 SD 前使用内核 block-device read-only 锁，并同时检查 sysfs 状态。
- 板端模型、二进制和日志只写入 DDR tmpfs。
- 开始与结束分别验证整卡 SHA-256、首尾各 4MiB SHA-256，以及 block layer 的
  write、write-sector、discard、flush 计数。
- 下载端只暴露三个白名单文件，支持断点续传；板端校验 SHA 后才重命名和执行。

## 当前硬阻塞

1. 系统 DNS 把 `huggingface.co` 错误解析到非 Hugging Face CDN 地址，导致 TLS
   `EOF occurred in violation of protocol`。通过公共 DoH 得到正确 CDN 地址后已读取
   官网 API，并把完整 revision、两个文件的精确大小和官方 LFS SHA-256 写入策略；
   GGUF 尚未落盘，x86 模型推理门禁和板端模型传输尚不能执行。
2. 服务器上没有与 SD 中 PetaLinux 匹配的 `Image`、16GB `system.dtb` 和 RAM
   `rootfs.cpio`。为遵守 SD 零写入红线，不能直接启动 SD 后再设置只读；必须先准备
   JTAG 加载的 RAM Linux。工程已提供 initramfs staging 和自动启动脚本，但没有
   猜测或生成不匹配的 Linux 镜像。

在以上两项解决前，板卡保持原状态，不执行复位、烧录或 SD 启动。
