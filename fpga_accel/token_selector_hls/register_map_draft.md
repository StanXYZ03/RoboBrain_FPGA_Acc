# Selector AXI4-Lite 寄存器（csynth 生成，未做 RTL cosim / 未上板）

> 状态：`CSYNTH_GENERATED / NO_COSIM / NO_BOARD / NO_BITSTREAM`。
> 下表来自 2026-09-27 v3 受控 Vitis HLS 2023.2 `csynth_design`（top
> `token_selector_top_packed`，part `xczu9eg-ffvb1156-2-e`，clock 5.0 ns），原始报告见
> `artifacts/token_selector/2026-09-27-selector-csynth-v3/`。这是工具综合估计，
> **未**做 C/RTL cosim、未导出 IP、未生成 bitstream，也未在 ZCU102 上验证。
> 控制/状态 bundle 的标量集合与 v2 完全相同，因此 offset 不变；只有 token `m_axi`
> 数据宽度改为 128 bit。

## 1. 控制与状态 bundle（s_axi_control，offset 24，地址宽度 7）

| Offset | 名称 | 方向 | 宽度 | 说明 |
|---|---|---|---|---|
| 0x00 | `CTRL` | RW | 32 | ap_start/ap_done/ap_idle/ap_ready |
| 0x04 | `GIER` | RW | 32 | 全局中断使能 |
| 0x08 | `IP_IER` | RW | 32 | 通道中断使能 |
| 0x0c | `IP_ISR` | RW | 32 | 通道中断状态 |
| 0x10 | `ap_return` | R | 32 | `status`（见 §3） |
| 0x18 | `group_count` | W | 32 | `G`，1..4096，且 <=65535 |
| 0x20 | `grid_h` | W | 32 | 区域行数，1..128 |
| 0x28 | `grid_w` | W | 32 | 区域列数，1..128，`grid_h*grid_w == G` |
| 0x30 | `budget` | W | 32 | `K`，16 的正倍数且 `<= TS_MAX_BUDGET=96`（生产 48/64/96） |
| 0x38 | `local_cap` | W | 32 | 区内候选上限，`[1, 16]`，生产固定 16 |
| 0x40 | `scale_in` | W | 32 | 全图共享对称 scale，仅透传 |
| 0x48 | `scale_out` | R | 32 | `scale_in` 透传寄存器（IEEE-754 float 位型） |
| 0x4c | `scale_out_ctrl` | R | 32 | `scale_out_ap_vld` |
| 0x58 | `output_count` | R | 32 | 输出组数（identity 为 `G`，压缩为 `K`） |
| 0x5c | `output_count_ctrl` | R | 32 | `output_count_ap_vld` |
| 0x68 | `real_count` | R | 32 | 真实组数（identity 为 `G`，压缩为 `K-C`） |
| 0x6c | `real_count_ctrl` | R | 32 | `real_count_ap_vld` |
| 0x78 | `context_count` | R | 32 | context 组数 `C`（identity 为 0） |
| 0x7c | `context_count_ctrl` | R | 32 | `context_count_ap_vld` |

## 2. 数据面 m_axi 端口与基址寄存器（s_axi_control_r，offset 16，地址宽度 7）

v3 把输入 tokens 与 5 个输出段的 tokens、以及输入 score 各自放到独立 `m_axi` bundle；
token 端口为 **128-bit**（`ap_uint<128>`，16 个 int8 lane/beat），metadata 端口不变。
host 侧 5 段字节布局未改变，只是多了基址寄存器（每段一个 64 位地址）。

| 端口 | bundle | 方向 | 段 | 元素类型 | 数据宽度 | 深度（元素/words）| control_r offset |
|---|---|---|---|---|---|---|---|
| `tokens_q` | `gmem_in_tokens` | R | 输入 tokens | `ap_uint<128>` | **128 bit** | `G * 256`（最大 1048576） | 0x10 / 0x14 |
| `score_q` | `gmem_in_score` | R | 输入分数 | uint16 | 16 bit | `G` | 0x1c / 0x20 |
| `out_tokens` | `gmem_out_tokens` | W | 段 1 `tokens_int8` | `ap_uint<128>` | **128 bit** | `max(G,K) * 256` | 0x28 / 0x2c |
| `out_group_index` | `gmem_out_group` | W | 段 2 `group_index_uint16` | uint16 | 16 bit | `max(G,K)` | 0x34 / 0x38 |
| `out_source_type` | `gmem_out_source` | W | 段 3 `source_type_uint8` | uint8 | 8 bit | `max(G,K)` | 0x40 / 0x44 |
| `out_region_id` | `gmem_out_region` | W | 段 4 `region_id_uint8` | uint8 | 8 bit | `max(G,K)` | 0x4c / 0x50 |
| `out_valid` | `gmem_out_valid` | W | 段 5 `valid_uint8` | uint8 | 8 bit | `max(G,K)` | 0x58 / 0x5c |

`m_axi` 参数：地址宽度 64、max read/write burst 16、outstanding read/write 16。
`gmem_in_tokens` / `gmem_out_tokens` 的 `WDATA`/`RDATA` 为 **128 bit**（报告确认
`gmem_out_tokens_WDATA` = 128），即 v2 的 1 byte/cycle token 端口已消除；字节序为小端，
lane `L` = 地址 `group*4096 + w*16 + L`。

## 3. status 编码

| 值 | 名称 | 含义 |
|---|---|---|
| 0 | `OK` | 成功 |
| 1 | `ERR_NULL` | 空指针 |
| 2 | `ERR_GROUP_LIMIT` | `G<1` 或 `G>4096`/`>65535` |
| 3 | `ERR_GRID` | grid 非正、单维 >128，或 `grid_h*grid_w != G` |
| 4 | `ERR_BUDGET` | `K<=0`、`K%16 != 0` 或 `K>96` |
| 5 | `ERR_INSUFFICIENT` | `K-C < 非空区域数` |
| 6 | `ERR_CANDIDATE_POOL` | 有界候选池不足以填满 `real` |
| 7 | `ERR_INTERNAL` | 内部不变量被破坏 |
| 8 | `ERR_LOCAL_CAP` | `local_cap` 不在 `[1,16]` |

## 4. 5 段传输布局（按 K，host 布局未变）

| K | total B | tokens off/size | group off/size | source off | region off | valid off |
|---|---|---|---|---|---|---|
| 48 | 196928 | 0 / 196608 | 196608 / 96 | 196736 | 196800 | 196864 |
| 64 | 262464 | 0 / 262144 | 262144 / 128 | 262272 | 262336 | 262400 |
| 96 | 393792 | 0 / 393216 | 393216 / 192 | 393408 | 393536 | 393664 |

## 5. csynth 结果与后续门控（v3，packed 128-bit top）

- 综合完成、0 error、资源任一类 <50%：BRAM 130（7.13%）、DSP 67（2.66%）、
  FF 185662（33.87%）、LUT 91752（33.48%）、URAM 0。
- 顶层 latency **有限且通过**：best 2 / average 124880 / worst **2499989** cycles
  = **12.50 ms @200 MHz < 33.3 ms**（v2 为 102.81 ms）。
- **门控唯一未过项**：5.0 ns 目标在默认 1.35 ns uncertainty 下估计周期 **4.423 ns**、
  slack `-0.773 ns`（未改 uncertainty）。`ts_pool_context_packed` 子模块自身为 3.650 ns，
  达标；残余关键路径在共享选择逻辑的 top-level 平面逻辑内，两轮允许的修复内未解决。
- v2 的 8-bit `gmem_in_tokens` 读带宽瓶颈（II=8，context 池化 18,996,156 cycles，HLS 200-880）
  已被 128-bit 端口消除。
- 决策 `REDESIGN_REQUIRED`：在 5 ns timing 解决前，不写驱动、不做 cosim、不导出 IP、不上板。
