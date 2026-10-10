# ZCU102 Q8×Q8 PE16 DATAFLOW上板前离线封板报告

状态：`DATAFLOW_OFFLINE_PREBOARD_PASS / BOARD_AUTHORIZATION_REQUIRED`

## 结论

DATAFLOW候选系统的完整离线上板包已通过一致性门控。新XSA生成的FSBL保留SPD动态
双Rank 16 GB DDR初始化补丁；其PS初始化和补丁后DDR初始化与已上板PE16基线逐字节一致。
本阶段没有连接、复位或配置ZCU102。

## 门控结果

| 层次 | 结果 |
|---|---:|
| bit/XSA完整性、200 MHz时序、DRC Error | PASS / PASS / 0 |
| FSBL来源、AArch64符号、16 GB DDR补丁 | PASS |
| runtime单测 | 4/4 PASS |
| runtime ASan/UBSan | 4/4 PASS |
| benchmark自检/消毒器 | PASS / PASS |
| APM单测 | PASS |
| DATAFLOW寄存器映射 | 10/10 PASS |
| 16 GB DTB、64 MiB DMA、UIO | PASS |
| 任务无关描述符、Q8_0拆分和DMA自动分块 | 5/5 PASS |
| 描述符ASan/UBSan | 5/5 PASS |
| initramfs载荷哈希 | PASS |
| 板测协议dry-run/模拟idle | 15/15 PASS |
| 板测汇总器及APM公式负向测试 | 4/4 PASS |
| UART启动识别/指令发送/追加日志状态机 | PTY PASS |
| 新RAM Linux QEMU启动 | PASS |
| JTAG脚本语法与路径覆盖 | PASS |

新RAM Linux包包含任务无关描述符、自动DMA分块版benchmark和板端测量协议。冻结的
`language_prefill`仍是用于PE8/PE16/DATAFLOW同形状比较的`M=512`单次调用；另增独立
`visual_prefill M=1080`，用于固定`512+512+56`与自动`976+104`的同工作量A/B。
Vision Encoder自动分块为`3584+736`，Decoder为`1`。规划器结果只说明容量和接口合法，
不能替代板端性能结果。

测量协议包含15个fail-fast阶段：协议前后数值自检、三类同形状kernel对照、完整视觉
Prefill/Vision/Decoder固定与自动分块A/B，以及自动策略的traffic/latency APM。每个阶段后
强制验证`ap_idle`，并拒绝并发运行。汇总器用“累计延迟÷累计事务数”计算平均延迟；
旧PE16报告中被放大3倍的平均延迟已另行追加修正记录，原始日志和旧报告未覆盖。

## 关键产物

| 产物 | SHA256 |
|---|---|
| DATAFLOW bitstream | `c70d08a386d2a69b4319094107eea28e0591119ad1a86bfabf5598a806600036` |
| DATAFLOW XSA | `6a1fd18fde062ebc192c37b1288ca0116c8013cfe9caa1ca8800705482b43f04` |
| DATAFLOW FSBL | `0f2c39fe97f121a31bef848996d460be71f174ae10b65dc28f05b90d8fe4481a` |
| AArch64 selftest | `7057bd3a9d0cd9a639f39b23b2d3224b7b13860752382582f8faa461ba24ac6b` |
| AArch64 benchmark | `9efd7c52dd27d5f4041b0600fc9126a02b0460adc24d203cd3348f12445e8925` |
| 板端测量协议 | `e9b79f4d16abf4a163aadbe416f9a0da56b52c637f6c68f32d7108b67ef70edf` |
| 板测汇总器 | `a755403805d749dec2570971625c7e0ed8bbd4f9f3924c98d48acbf9ad250380` |
| UART自动采集器 | `d897e71931d88e638a50e9495745c8dee9dae5ab1f96cc7cfce345372ad4ace9` |
| 新RAM Linux Image | `73c131879b1237d4cfb8de4eb24cf7a288d27f6defda488609d9d86c87457e6b` |
| 新initramfs | `927cb283c56687a0f8f7b53b5d3c48fc03030443206e1c371ce069b6461d8972` |

机器可读记录：`deployment/artifacts/q8q8_pe16_dataflow_preboard_summary.json`。

## 上板边界

下一步需要JTAG易失性复位和PL配置，会替换当前板上PE16 RAM Linux状态，但不会写SD/QSPI。
只有取得明确授权后才能执行。板测必须依次通过DDR识别、PL configured、PS–PL reset/isolation、
AXI-Lite idle、Linux UIO、协议前后双自检、固定/自动分块对照及APM无溢出门控。
QEMU只验证Linux/initramfs可启动，不能替代上述任何板端门控。
