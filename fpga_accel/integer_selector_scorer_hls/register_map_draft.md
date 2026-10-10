# register-map draft — integer_selector_scorer

状态：**draft**。本轮只做 `csynth_design`，未 `export_design`，因此下面的
AXI-Lite 偏移是**预期布局**（Vitis HLS 2023.2 默认 s_axilite 顺序），最终偏移
以 export 生成的 `.hwh`/driver 为准。数据面宽度 128 bit。

## 1. AXI-Lite control bundle（`control`）

| offset | 名称 | 位宽 | 方向 | 说明 |
|---:|---|---:|---|---|
| 0x00 | AP_CTRL | 32 | RW | `ap_start` / `ap_done` / `ap_idle` / `ap_ready`（块控制） |
| 0x04 | GIE | 32 | RW | 全局中断使能 |
| 0x08 | IER | 32 | RW | 中断使能 |
| 0x0C | ISR | 32 | RW | 中断状态 |
| 0x10 | group_count | 32 | W | group 数，合法 1..4096；非法返回 `ERR_GROUP_COUNT` |
| 0x18 | group_mean_q_base_lo | 32 | W | `group_mean_q` AXI 基址低 32 位 |
| 0x1C | group_mean_q_base_hi | 32 | W | 基址高 32 位 |
| 0x20 | wv_q_base_lo | 32 | W | `wv_q` AXI 基址低 32 位 |
| 0x24 | wv_q_base_hi | 32 | W | 基址高 32 位 |
| 0x28 | query_q_base_lo | 32 | W | `query_q` AXI 基址低 32 位 |
| 0x2C | query_q_base_hi | 32 | W | 基址高 32 位 |
| 0x30 | score_q_base_lo | 32 | W | `score_q` AXI 基址低 32 位 |
| 0x34 | score_q_base_hi | 32 | W | 基址高 32 位 |
| 0x38 | return (status) | 32 | R | `0=OK`、`1=ERR_NULL`、`2=ERR_GROUP_COUNT` |

> 偏移/寄存器个数取决于 `offset=slave` 与地址位宽；若 HLS 把 64-bit 地址压成
> 单个 64-bit 寄存器，则对应 low/hi 合并为一个 64-bit 槽。`return` 由
> `#pragma HLS INTERFACE s_axilite port=return bundle=control` 暴露。

## 2. m_axi 数据 bundle

| bundle | 端口 | 方向 | 元素 | 传输宽度 | 深度（beat 数） | 说明 |
|---|---|---|---:|---|---:|---|
| `gmem_gm` | `group_mean_q` | in | `ap_uint<128>` | 128 bit | `G*64`（max 262144） | 16 int8/word 小端 |
| `gmem_wv` | `wv_q` | in | `ap_uint<128>` | 128 bit | 8192 | 每次调用缓存一次 |
| `gmem_query` | `query_q` | in | `ap_uint<128>` | 128 bit | 8 | 128 int8 |
| `gmem_score` | `score_q` | out | `uint16` | 16 bit（可打包到 128-bit burst） | `G`（max 4096） | Q0.15 |

`m_axi` 参数：`offset=slave`；`group_mean_q`/`wv_q`/`score_q`
`max_*_burst_length=64`、`num_*_outstanding=16`，`query_q` burst 16 /
outstanding 4。空的 AXI base address（0）触发 `ERR_NULL`。

## 3. 软件写入顺序（driver 草案）

1. 写 4 个 `*_base` 寄存器指向 host 缓冲。
2. 写 `group_count`。
3. 写 `AP_CTRL.ap_start=1`，轮询 `ap_done`。
4. 读 `return`，0 时读 `score_q`（`group_count` 个 uint16）。

## 4. 尚未完成（非本轮）

- `export_design` / 真实 `.hwh` 偏移、中断聚合、DMA 驱动、cache 一致性处理。
- 与 `fpga_accel/token_selector_hls` 的 `score_q` 端口对接（见报告 §对接分析）。
