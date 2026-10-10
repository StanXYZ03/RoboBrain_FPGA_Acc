# ZCU102 Q8×Q8 PE16 bitstream离线验收

生成时间：2026-08-27T16:22:41.848368+00:00
状态：`PE16_BITSTREAM_PASS / BOARD_AUTHORIZATION_REQUIRED`

## 结论

- PE16完整系统已生成独立bitstream和含bitstream的XSA。
- 本阶段只做文件生成与完整性检查；未连接、复位或烧写ZCU102。
- 这不证明PE16板端数值正确或获得任何实测加速。

## 门控结果

| 项目 | 结果 |
|---|---:|
| summary_says_generated | PASS |
| summary_says_board_not_programmed | PASS |
| timing_pass | PASS |
| route_pass | PASS |
| drc_zero_error | PASS |
| xsa_contains_bit | PASS |
| xsa_bit_matches_standalone | PASS |
| output_axi_128 | PASS |

## 实现与产物

- 频率：199.98001 MHz；WNS=0.067 ns；WHS=0.011 ns；未布线网络=0。
- DRC：Error 0，Warning 613，Advisory 144。
- bitstream：26510904 bytes，SHA256=`d8ab6e8215d1ce41f320be8afad51aff5d4e7d90b4135e7994c90d9e1f810f65`。
- XSA：2884218 bytes，SHA256=`d2d5a85aced516fb5be3e511d8ddeb712902cabd6bb6476b5d5a3e9a00f16934`。
- XSA内嵌bit与独立bit：一致。

## 下一门

先用PE16 IP重新核对PS runtime寄存器映射并冻结板测包；获得明确上板授权后，才可临时配置PL并依次运行确定性self-test、language_prefill、vision_encoder和decoder_gemv microbenchmark。

机器可读证据：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/fpga_accel/q8q8_pe16_zcu102_vivado/artifacts/pe16_bitstream_summary.json`。
