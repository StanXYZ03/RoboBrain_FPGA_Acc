# Jetson Xavier NX + ZCU102 Selector 异构需求模型（基线修正版 v2）

- 日期：2026-09-28
- 决策：`HETEROGENEOUS_SELECTOR_BASELINE_CORRECTED / NO_FULL_BREAK_EVEN / NO_MEASURED_SPEEDUP_CLAIM`
- 范围：**只做模型与参数扫描**。不使用 GPU、不访问 ZCU102、不联网、不做板级实测，不声称当前已加速。

## 0. 为什么必须修正基线（先读）

旧 v1 用单一 `J_ONLY` 作基线，而 `J_ONLY` 已包含 Jetson 软件 selector **和** K-token 后续视觉；
`J_F_J` 与它的 K-token 后续视觉完全相同，因此不能用“post-B1 节省”抵消 FPGA 往返。
若改对 Dense 基线（`J_DENSE`），score 生成（`T_score`）不再抵消，必须显式保留。
本版据此拆成 `J_DENSE` 与 `J_PRUNED` 两个基线，并输出两套严格公式。
- **禁止**再输出相对旧 `J_ONLY` 的 9–12 ms“完整 break-even”。数表只给 communication + FPGA-selector 的 lower bound，
  必须以 `+T_score` 或“`T_sel_sw` 需超过 X”表达。

## 1. 四个 placement 定义

| placement | score | selector | K tokens | 后续视觉 | 链路腿 | 固定同步开销 | 总延迟 |
|---|---|---|---|---|---|---:|---|
| `J_DENSE` | 无 | 无 | 不裁剪 | dense Jetson vision | 0 | 0 | `T_vision_dense` |
| `J_PRUNED` | 在 Jetson | Jetson 软件 | 留 Jetson | K-token Jetson vision | 0 | 0 | `T_score + T_sel_sw + T_vision_pruned` |
| `J_F_J` | 在 Jetson | FPGA | 返回 Jetson | 同一 K-token Jetson vision | 2 | 2 | `T_score + T_link_dense + T_sel_fpga + T_link_return + 2F + T_vision_pruned` |
| `J_F_CONTINUE` | 在 Jetson | FPGA | 留 FPGA | 真实 FPGA 下游核 | 1 | 1 | `T_score + T_link_dense + T_sel_fpga + F + T_fpga_downstream` |

**Selector IP 不含 Wt/Wv score 生成。** `T_score` 单列，且不计入 FPGA 核 cycles。
它在 `J_F_J` vs `J_PRUNED` 中抵消（连同 `T_vision_pruned`），但在对 `J_DENSE` 的算法收益中**必须保留**。

## 2. 两套严格公式

记 `S_prune = T_vision_dense - T_vision_pruned`，`F` 为每条链路腿的一次固定软件+DMA 开销。

### A. 算法收益 vs `J_DENSE`（`J_DENSE = T_vision_dense`）

```
J_PRUNED       : S_prune > T_score + T_sel_sw
J_F_J          : S_prune > T_score + T_link_dense + T_sel_fpga + T_link_return + 2F
J_F_CONTINUE   : T_vision_dense - T_fpga_downstream > T_score + T_link_dense + T_sel_fpga + F
```

`J_F_J` 对 dense 基线的要求里**含 `T_score` 符号项**，不能把 score 当作抵消。

### B. 部署选择 vs `J_PRUNED`（`J_PRUNED = T_score + T_sel_sw + T_vision_pruned`）

```
J_F_J          : T_sel_sw > T_link_dense + T_sel_fpga + T_link_return + 2F
J_F_CONTINUE   : T_sel_sw + T_vision_pruned > T_link_dense + T_sel_fpga + F + T_fpga_downstream
```

`J_F_J` 对 `J_PRUNED` 的公式**不含任何 vision saving**：`T_vision_pruned` 与 `T_score` 严格抵消，
只有 `T_sel_sw` 超过“往返 + FPGA selector + 2F”时才更快。

## 3. 字节合同（从真实实现推导）

- 来源：`evaluation/hardware_regular_selector_fixedpoint.py` 的 `fpga_transfer_layout` 与 `GROUP_ELEMENTS=4096`。
- dense 输入 = `G*4096 B`（int8 `[G,4,1024]`）+ `G*2 B`（uint16 Q0.15 score）。
- 输出 = 5 段布局：`tokens_int8 K*4096 B` + `group_index_uint16 K*2` + `source_type_uint8 K` + `region_id_uint8 K` + `valid_uint8 K`，每段 64 B 对齐。

| G | K | dense 输入/B | 选中输出/B | 往返/B |
|---:|---:|---:|---:|---:|
| 256 | 48 | 1049088 | 196928 | 1246016 |
| 256 | 64 | 1049088 | 262464 | 1311552 |
| 256 | 96 | 1049088 | 393792 | 1442880 |
| 576 | 48 | 2360448 | 196928 | 2557376 |
| 576 | 64 | 2360448 | 262464 | 2622912 |
| 576 | 96 | 2360448 | 393792 | 2754240 |
| 1024 | 48 | 4196352 | 196928 | 4393280 |
| 1024 | 64 | 4196352 | 262464 | 4458816 |
| 1024 | 96 | 4196352 | 393792 | 4590144 |

## 4. Selector 周期证据（读取，不手填）

- v4 csynth：worst bound **1,708,124 cycles** = 8.5406 ms @ 200 MHz（pre-route 静态界，对 G=256/576/1024 保守）。
- v4 cosim：功能 PASS，3 个代表 case，聚合 latency min/avg/max = **5,668 / 209,518 / 440,899 cycles**；cosim **只验证功能，不证明 200 MHz**。
- 200 MHz 换算的时序依据是独立 OOC route：period 5.0 ns、WNS +0.689 ns、status PASS（OOC，非整机/上板）。

| cosim case | G | K |
|---|---:|---:|
| `cosim_identity_4x4_K48` | 16 | 48 |
| `cosim_16x16_K64_alltie` | 256 | 64 |
| `cosim_32x32_K96_mixedextreme` | 1024 | 96 |

## 5. 数表：只有 communication + FPGA-selector lower bound

以下为 `csynth worst bound`、2000 MB/s、固定开销 0.5 ms 的代表点。
`known lower bound` 只含链路纯序列化 + FPGA selector + 固定项，**不含任何未实测的 Jetson 项**；
因此它只是所需量（`S_prune` / `T_vision_dense - T_fpga_downstream` / `T_sel_sw` 等）的**下界**。

| G | K | comparison | placement | T_link_dense/ms | T_sel_fpga/ms | T_link_return/ms | 固定项/ms | known lower bound/ms | 严格需求 |
|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| 256 | 64 | `A_vs_J_DENSE` | `J_PRUNED` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `S_prune > T_score + T_sel_sw` |
| 256 | 64 | `A_vs_J_DENSE` | `J_F_J` | 0.5245 | 8.5406 | 0.1312 | 1.0000 | 10.1964 | `S_prune > T_score + 10.196396 ms` |
| 256 | 64 | `A_vs_J_DENSE` | `J_F_CONTINUE` | 0.5245 | 8.5406 | 0.0000 | 0.5000 | 9.5652 | `T_vision_dense - T_fpga_downstream > T_score + 9.565164 ms` |
| 256 | 64 | `B_vs_J_PRUNED` | `J_F_J` | 0.5245 | 8.5406 | 0.1312 | 1.0000 | 10.1964 | `T_sel_sw > 10.196396 ms` |
| 256 | 64 | `B_vs_J_PRUNED` | `J_F_CONTINUE` | 0.5245 | 8.5406 | 0.0000 | 0.5000 | 9.5652 | `T_sel_sw + T_vision_pruned > 9.565164 ms + T_fpga_downstream` |
| 1024 | 96 | `A_vs_J_DENSE` | `J_PRUNED` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `S_prune > T_score + T_sel_sw` |
| 1024 | 96 | `A_vs_J_DENSE` | `J_F_J` | 2.0982 | 8.5406 | 0.1969 | 1.0000 | 11.8357 | `S_prune > T_score + 11.835692 ms` |
| 1024 | 96 | `A_vs_J_DENSE` | `J_F_CONTINUE` | 2.0982 | 8.5406 | 0.0000 | 0.5000 | 11.1388 | `T_vision_dense - T_fpga_downstream > T_score + 11.138796 ms` |
| 1024 | 96 | `B_vs_J_PRUNED` | `J_F_J` | 2.0982 | 8.5406 | 0.1969 | 1.0000 | 11.8357 | `T_sel_sw > 11.835692 ms` |
| 1024 | 96 | `B_vs_J_PRUNED` | `J_F_CONTINUE` | 2.0982 | 8.5406 | 0.0000 | 0.5000 | 11.1388 | `T_sel_sw + T_vision_pruned > 11.138796 ms + T_fpga_downstream` |

最快扫描链路（3000 MB/s，固定开销 0.1 ms）下最乐观的 known lower bound：
- `J_F_J` vs `J_PRUNED`：`T_sel_sw > 9.155959 ms`（即要求 `T_sel_sw > 9.1560 ms`）。
- `J_F_J` vs `J_DENSE`：`S_prune > T_score + 9.155959 ms`。
- `J_F_CONTINUE` vs `J_DENSE`：`T_vision_dense - T_fpga_downstream > T_score + 8.990316 ms`。
- `J_F_CONTINUE` vs `J_PRUNED`：`T_sel_sw + T_vision_pruned > 8.990316 ms + T_fpga_downstream`。

> These are communication + FPGA-selector lower bounds only. Do not quote them as a full break-even relative to the superseded J_ONLY baseline, and always attach '+ T_score' or 'T_sel_sw must exceed X'.

## 6. 扫描摘要

- 扫描行数：3240。
- comparison：A_vs_J_DENSE, B_vs_J_PRUNED；baseline：J_DENSE, J_PRUNED。
- selector 周期场景：csynth_worst_bound, cosim_observed_max, cosim_observed_avg, cosim_observed_min。
- 有效吞吐扫描（MB/s）：[100, 250, 500, 1000, 2000, 3000]。
- 单次固定软件+DMA 开销扫描（ms）：[0.1, 0.5, 1.0]。
- `symbolic_missing_terms`：T_score, T_sel_sw, T_vision_dense, T_vision_pruned, T_fpga_downstream。
- 全部行的 `lower_bound_only=True`；没有任何一行的数值可单独判定胜负。

## 7. 必须补齐的实测（Gate）

1. Jetson 上实测 `T_score`（Wt/Wv score 生成）。
2. Jetson 上实测 `T_sel_sw`（软件 selector）；实测后代入 B 组公式。
3. Jetson 上实测 `T_vision_dense`、`T_vision_pruned`（`J_DENSE` / `J_PRUNED` 基线）。
4. 实测 Jetson↔ZCU102 有效吞吐与固定软硬件+DMA 开销（当前为参数扫描）。
5. 若走 `J_F_CONTINUE`：必须存在并实测真实 FPGA 下游核 `T_fpga_downstream`；当前仓库没有该核。
6. 板级 200 MHz 与功能验证（当前仅 OOC route 报告 + 功能 cosim）。

在上述实测完成前，**不得**声称异构 selector 已加速，也不得给出“完整 break-even 毫秒数”。

## 8. 复现命令

```bash
python3 scripts/analyze_heterogeneous_selector_break_even.py \
  --config configs/heterogeneous_selector_break_even.json
```

