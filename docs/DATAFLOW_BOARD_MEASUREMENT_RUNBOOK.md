# PE16 DATAFLOW上板测量操作单

状态：`OFFLINE_VALIDATED / EXPLICIT_BOARD_AUTHORIZATION_REQUIRED`

本操作会复位当前易失性RAM Linux并替换PL配置，但不写SD或QSPI。必须先取得用户对本次
DATAFLOW上板的明确授权。UART采集必须先于JTAG复位启动，否则启动证据不完整。

## 1. 离线门控

```bash
cd /home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102
deployment/scripts/verify_q8q8_pe16_dataflow_bundle.sh
```

唯一允许继续的结果是`Q8Q8_PE16_DATAFLOW_BUNDLE_PASS`且
`board_accessed=no board_programmed=no`。

## 2. 先启动UART采集

为本次运行选择从未使用的时间戳；以下`RUN_ID`只是格式示例，实际执行时替换为当前UTC时间：

```bash
RUN_ID=20260830T000000Z
python3 deployment/scripts/capture_q8q8_dataflow_uart.py \
  --device /dev/ttyUSB0 \
  --log reports/${RUN_ID}-zcu102-q8q8-pe16-dataflow-board-uart.log
```

看到`DATAFLOW_UART_CAPTURE_READY`后才能进入下一步。采集器会等待新系统的启动自检，随后
只发送一次`/opt/robobrain/q8q8-dataflow-measure`。它不会复位或烧录板卡，不会在失败后
自动重试，也拒绝覆盖既有日志。

## 3. JTAG易失性启动

确认`hw_server`正在本机`127.0.0.1:3121`监听；若没有，由有权限的终端启动：

```bash
/home/xilinx/Vivado/2023.2/bin/hw_server -s tcp::3121
```

另一个终端执行：

```bash
cd /home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102
RUN_ID=20260830T000000Z  # 必须与UART终端使用完全相同的实际时间戳
set -a
. deployment/artifacts/q8q8_pe16_dataflow_jtag.env
set +a
/home/xilinx/Vitis/2023.2/bin/xsct -nodisp \
  deployment/scripts/run_q8q8_accel_jtag.tcl \
  2>&1 | tee reports/${RUN_ID}-zcu102-q8q8-pe16-dataflow-board-xsct.log
```

成功时XSCT必须包含16 GB双Rank DDR、PL configured、PS–PL隔离/复位释放、AXI idle和
JTAG handoff全部PASS。UART采集器只有看到协议最终PASS才会退出；结束后板上控制寄存器
应为`0x00000004`，加速器idle，UART shell仍然存在。

## 4. 冻结结果

```bash
RUN_ID=20260830T000000Z  # 必须与前两步使用完全相同的实际时间戳
python3 deployment/scripts/summarize_q8q8_pe16_dataflow_board.py \
  --uart reports/${RUN_ID}-zcu102-q8q8-pe16-dataflow-board-uart.log \
  --xsct reports/${RUN_ID}-zcu102-q8q8-pe16-dataflow-board-xsct.log \
  --json-out deployment/artifacts/${RUN_ID}-q8q8-pe16-dataflow-board-summary.json \
  --markdown-out reports/${RUN_ID}-zcu102-q8q8-pe16-dataflow-board.md
```

汇总器拒绝覆盖既有输出，并强制检查15阶段顺序、15次`ap_idle`、协议前后自检、所有chunk
数值抽检、固定/自动分块身份、APM 10/10计数器和零溢出。任何门控失败都保留原始日志并
停止，不重跑任务、不发送Ctrl-C、不声称板端加速结果。
