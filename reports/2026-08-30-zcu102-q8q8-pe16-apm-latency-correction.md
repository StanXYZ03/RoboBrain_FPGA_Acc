# PE16 APM平均延迟公式修正

日期：2026-08-30  
状态：`APM_LATENCY_FORMULA_CORRECTED / PERFORMANCE_AND_BOTTLENECK_CONCLUSIONS_UNCHANGED`

2026-08-28 PE16板测的原始APM数据有效，但旧汇总脚本把三次累计延迟除以“平均每次
事务数”，使四个平均延迟字段统一放大3倍。两个APM profile都覆盖相同的3次重复，
正确公式是：

```text
平均延迟 = 三次累计延迟 / 三次累计事务数
```

| 工作负载 | CCI1读平均/最大 | CCI2读平均/最大 | HP0读平均 | HP0写平均 |
|---|---:|---:|---:|---:|
| Language prefill | 32.38 / 458 | 30.61 / 455 | 20.30 | 5.67 |
| Vision encoder | 27.15 / 418 | 26.61 / 411 | 17.98 | 5.67 |
| Decoder GEMV | 32.61 / 430 | 30.44 / 442 | 20.72 | 5.32 |

单位为APM cycle。最大延迟、流量、kernel时间、GOPS、数值正确性和
compute/schedule-bound或memory/weight-stream-bound判定均不受此公式错误影响。
旧JSON与Markdown作为历史证据保留、不覆盖；机器可读修正见
`deployment/artifacts/q8q8_pe16_apm_latency_correction.json`。本次只重算原始日志，
未访问或配置板卡。
