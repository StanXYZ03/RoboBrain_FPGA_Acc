# ZCU102 Q8×Q8 PE16 DATAFLOW bitstream离线验收

状态：`DATAFLOW_BITSTREAM_PASS / OFFLINE_PREBOARD_NEXT`

## 结论

PE16 DATAFLOW完整系统已生成独立bitstream和含bitstream的XSA。200 MHz实现保持
WNS=`+0.078 ns`、WHS=`+0.010 ns`、未布线网络=`0`，默认DRC为`0 Error`。
本阶段没有访问、复位或配置ZCU102，因此不形成板端性能或正确性结论。

## 完整性门控

| 项目 | 结果 |
|---|---:|
| bitstream生成 | PASS |
| XSA包含bitstream | PASS |
| 独立bit与XSA内嵌bit一致 | PASS |
| 时序与布线 | PASS |
| DRC Error | 0 |
| 板卡访问/配置 | NO / NO |

## 冻结产物

- 原始运行目录：`fpga_accel/q8q8_pe16_dataflow_zcu102_vivado/artifacts/runs/20260829T141332Z_bitstream`
- bitstream：26,510,904 bytes，SHA256=`c70d08a386d2a69b4319094107eea28e0591119ad1a86bfabf5598a806600036`
- XSA：2,943,078 bytes，SHA256=`6a1fd18fde062ebc192c37b1288ca0116c8013cfe9caa1ca8800705482b43f04`
- XSA内嵌bit SHA256=`c70d08a386d2a69b4319094107eea28e0591119ad1a86bfabf5598a806600036`
- Vivado日志 SHA256=`1112b2d87be1a01b3c719c2cc149327261a159d312a1e50718189982702e7db9`
- 机器可读摘要：`fpga_accel/q8q8_pe16_dataflow_zcu102_vivado/artifacts/pe16_dataflow_bitstream_summary.json`

## 下一门

使用本XSA离线生成保留SPD动态双Rank 16 GB DDR补丁的JTAG-only FSBL；随后核对寄存器映射、
AArch64运行时、UIO设备树、initramfs载荷和JTAG脚本。通过后只生成上板候选包，仍不配置板卡。
