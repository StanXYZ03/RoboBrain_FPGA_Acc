# RoboBrain2.5-4B Q8_0：ZCU102 只读 SD 部署

本目录保留 CPU-only 模型验证路径，并新增 Q8×Q8 PL 算子的受控上板自检路径。主模型与 mmproj 来自社区仓库，非 BAAI
官方量化产物；仓库、revision、文件名、大小范围和 GGUF 元数据均由
`config/model.json` 锁定。完整 revision 为
`a43f8ba7a3499579a8f43258ea36af018f26a17c`；工程不会执行任何量化。

## 安全红线

SD 卡不得删除、覆盖、新增任何内容。板端流程必须从 JTAG 加载的 RAM initramfs
启动，SD 在挂载前通过 `blockdev --setro` 锁定，且整个测试期间不挂载 SD。
如果检测到 SD 曾以读写方式挂载或启用了 SD swap，流程立即失败。不能从 SD 正常
启动后再补做只读保护，因为那时不能证明启动阶段没有发生写入。

板端模型、程序、临时文件和日志只存在于 DDR tmpfs；断电后消失。服务器保存下载
文件、SHA 锁文件、板端日志和验收报告。

## 服务器端门禁

```bash
make deploy-test
make llama-host
make llama-arm64
make model-download
make model-smoke
```

如果服务器到 Hugging Face 的代理不可用，可指定兼容端点；仓库和 revision 仍不会
改变：

```bash
HF_ENDPOINT=https://your-approved-mirror.example make model-download
```

`model-download` 通过公共 DoH 临时解析 Hugging Face 与 CDN，再用 curl 的
`--connect-to` 纠正本次连接；它不修改系统 DNS。下载使用 `.part` 断点续传，完成后
验证：

- 主模型精确大小 `4693672576` 字节、SHA-256 `835b78ff…49eb`、架构
  `qwen3vl`、`general.file_type=7`（Q8_0）。
- mmproj 精确大小 `453974336` 字节、SHA-256 `fcc27825…fde2`、架构 `clip`。
- 两个文件的 SHA-256 和元数据写入 `deployment/models/model.lock.json`。

## RAM initramfs 要求

当前服务器没有用户 SD 卡对应的 PetaLinux `Image`、16GB `system.dtb` 和
`rootfs.cpio`，因此工程不会重启或烧录板卡。准备与当前 PetaLinux ABI 匹配的
RAM initramfs 后，将其解包到临时目录并执行：

```bash
sh deployment/scripts/stage_initramfs.sh /path/to/unpacked-initramfs
```

initramfs 必须使用 glibc、包含 `/bin/sh`、`blockdev`、`ip`、`sha256sum`、`curl`
或 `wget`，并把 `/run`、`/dev/shm` 挂载为 tmpfs。启动系统在这些挂载完成后自动
执行 `/opt/robobrain/boot_entrypoint.sh`。不要把生成的 initramfs 写入 SD；由 JTAG
从服务器 DDR 临时加载。

Linux Device Tree 必须描述已经验证的完整 16GB DDR。板端 `inventory.sh` 要求
`uname -m=aarch64` 且 `MemTotal >= 15000000 KiB`，否则拒绝下载模型。

## 以太网运行顺序

服务器端先执行：

```bash
sudo sh deployment/scripts/configure_host_network.sh
python3 deployment/scripts/log_receiver.py \
  --output deployment/reports/board-runtime.log
python3 deployment/scripts/artifact_server.py \
  --model-lock deployment/models/model.lock.json \
  --model-dir deployment/models \
  --llama-server deployment/artifacts/arm64/llama-server
```

随后通过 JTAG 启动 RAM Linux。板端固定为 `192.168.10.2/24`，会自动完成 SD
只读门禁、16GB 内存检查、三个白名单文件的断点传输和 SHA 校验，并启动
`llama-server`。服务器验收命令：

```bash
python3 deployment/scripts/host_chat_test.py \
  --result deployment/reports/board-chat.json
```

文本测试通过后才允许在后续命令中增加匹配的 mmproj；当前启动脚本不会加载图像
投影模型。结束前在板端执行 `verify_sd_unchanged.sh`，要求 SD 仍为内核只读，整卡
SHA-256、首尾各 4MiB 哈希以及 block layer 的 write/discard/flush 计数均与开始时
一致。整卡哈希会延长启动和收尾时间，但默认不允许跳过。

## 验收结果

通过条件为：离线单元测试、DDR/以太网 testbench、x86 模型冒烟测试、AArch64
QEMU 启动检查、SD 只读门禁、Linux 识别 16GB、板端三次文本回复和结束时 SD
哈希检查全部通过。本阶段只记录加载时间、首 token 时间和 tokens/s，不设置性能
下限。

## Q8×Q8 PL 加速器离线包

加速版 RAM Linux 使用 `generic-uio` 映射控制寄存器 `0xA0000000`，并把
16 GB DDR 顶部的 64 MiB `0xB7C000000–0xB7FFFFFFF` 作为 `no-map` DMA
自检区。生成包已包含 UIO 内核配置、设备树节点和 ARM64
`/opt/robobrain/q8q8-selftest`。

只读验证（不会连接或操作板卡）：

```bash
sh deployment/scripts/verify_q8q8_accel_bundle.sh
sha256sum -c deployment/artifacts/q8q8_accel_bundle.sha256
```

待明确批准上板后，才可用 `deployment/scripts/run_q8q8_accel_jtag.tcl` 启动。
该脚本使用同一 XSA 生成的 FSBL 与 bitstream，先验证动态 DDR 初始化得到
16 GiB/双 rank，再配置 PL 并读回 `ap_idle`，随后启动 RAM Linux。最终通过标准是
板端打印 `SELFTEST_PASS m=2 n=3 k=32 outputs=6 max_abs=0`。此步骤只证明独立
GEMM 数据通路，尚不表示 llama.cpp/RoboBrain 的真实算子已接入 PL。
