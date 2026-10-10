# ZCU102 PS DDR4 最小验证

执行顺序固定为：主机 testbench、生成板端程序、JTAG 运行、JTAG 回读状态。
在 testbench 通过之前禁止下载或运行板端程序。

```bash
make -C fpga_accel/ddr_test sim
```

testbench 必须同时满足：健康内存通过，地址镜像被检出，单 bit 读错误被检出。

16GB 测试点覆盖 0.25、1.25、2、3、4、8、12 和 15GiB 附近。ZynqMP 将
前2GiB映射到低地址窗口，其余容量映射到从 `0x8_0000_0000` 开始的高地址窗口。

板端程序最终将状态写到 OCM `0xFFFEF000` 固定区域，远程测试脚本读取以下字段：

- `magic == 0x44445234`
- `phase == 0x600D` 表示通过，`0xDEAD` 表示失败
- `checks`、`failures`
- `last_address`、`expected`、`actual`

因此不依赖 LED、显示器或人工观察串口。JTAG 命令输出同时保存为日志，作为测试记录。
