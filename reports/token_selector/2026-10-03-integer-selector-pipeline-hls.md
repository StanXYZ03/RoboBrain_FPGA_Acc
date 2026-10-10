# 整数打分核与硬件规则选择核片上串接 `integer_selector_pipeline`：实现、验证与受控 HLS csynth

- 日期：2026-10-03
- 决策：**`INTEGER_SELECTOR_PIPELINE_HLS_PASS`**
- 范围：**只 `csynth_design`**（Vitis HLS 2023.2、part `xczu9eg-ffvb1156-2-e`、`create_clock -period 5.0`、tool 默认 uncertainty **1.35 ns**，未放宽）；无 C/RTL cosim、无 IP 导出、无 implementation/bitstream、无板卡、无 GPU、无联网、无训练、无调参。
- 大工程只在 `/tmp`；仓库只保存源码、小 fixture、小报告与日志。
- 只读复用既有资产（seed-20261003 Wv bundle、grounding/pointing validation fixture），**未改**两个既有 HLS 目录的源码/测试/报告/产物，也未改模型/checkpoint/manifest/config/既有报告。

## 0. 结论摘要

新增独立顶层 `integer_selector_pipeline`，把两个已验收模块串成单核数据通路：

```
group_mean_q + wv_q + query_q
     -> 已验收整数 scorer 核  -> score_q[G]（片上局部缓冲，不是端口）
     -> 已验收 packed v4 selector 核 -> selected tokens + metadata
```

- 四道功能门逐字节通过：Python 7/0、g++ 383/0、ASan+UBSan 383/0、HLS C-sim 383/0（0 error）。
- `csynth_design` 0 error；estimated period **3.65 ns** ≤ effective budget **3.65 ns**（slack **0.0**，Fmax **273.97 MHz**）。
- 资源占 ZCU102：BRAM_18K **17.98%**、DSP **8.41%**、FF **9.16%**、LUT **33.16%**、URAM **0.00%**（均 < 50%）。
- 顶层无外部 `score_q` 数据端口（m_axi bundle：`gmem_in_gm`, `gmem_in_query`, `gmem_in_tokens`, `gmem_in_wv`, `gmem_out_group`, `gmem_out_region`, `gmem_out_source`, `gmem_out_tokens`, `gmem_out_valid`；score 端口数 = 0）。
- **公式推导** G=400/K=96 总周期 **1,073,128** = **5.366 ms @200 MHz** ≤ 6,000,000 cycles / 30 ms。

## 1. 顶层合同

```
int integer_selector_pipeline(
    const ts_word_t* tokens_q,       // [G*256] 128-bit packed int8
    const is_word_t* group_mean_q,   // [G*64]  128-bit packed int8
    const is_word_t* wv_q,           // [8192]  128-bit packed int8
    const is_word_t* query_q,        // [8]     128-bit packed int8
    ts_word_t* out_tokens,           // [K*256] 128-bit packed int8
    uint16_t* out_group_index, uint8_t* out_source_type,
    uint8_t* out_region_id, uint8_t* out_valid,
    int group_count, int grid_h, int grid_w, int budget, int local_cap,
    float scale_in, float* scale_out,
    int* output_count, int* real_count, int* context_count);
```

| 项 | 定义 |
|---|---|
| `score_q` | `uint16_t score_local[IS_MAX_GROUPS]` **顶层函数内局部数组**；源码静态检查与综合接口表双重确认它不是端口 |
| 输入 | `tokens_q`/`group_mean_q`/`wv_q`/`query_q`，全部 128-bit packed INT8，lane 小端 |
| 输出 | 沿用 token selector 的 5 段语义与 64 B 对齐合同（selected tokens + group_index + source_type + region_id + valid），加 `scale_out`/三计数 |
| 状态码 | union：0 OK、1 NULL、2 G 非法（两核同值）、3 GRID、4 BUDGET、5 INSUFFICIENT、6 CANDIDATE_POOL、7 INTERNAL、8 LOCAL_CAP |

## 2. 复用策略与最小暴露

- `fpga_accel/integer_selector_scorer_hls/src/integer_selector_scorer.cpp` 被 **`#include`** 进 wrapper：`is_scorer_core<>` 位于该文件的匿名命名空间（静态可见性），无法从独立 TU 链接；只读复用该 .cpp 是“最小、可审计的 core 暴露”。
- `fpga_accel/token_selector_hls/src/token_selector_hls.cpp` 同样 `#include`，直接调用 `token_selector_core_packed`，使整个融合数据通路成为单一综合单元并被 HLS 内联。
- 两个既有目录**逐字节未改**；未复制第二套数学（Python 静态测试断言 wrapper 中不出现 `floor_sqrt`/`round_half_away`/`minmax`/投影数组）。
- 综合可组合性审计：Vitis HLS 2023.2 的 `HLSGen` 在“两个 core 融合进 helper 函数/C++ 函数模板”时会 segfault（已用最小变体复现并记录）；因此融合体**直接写在顶层 `extern "C"` 函数内**，host-only debug 孪生单独用 `#ifndef __SYNTHESIS__` 隔离。上游 core 自身的模板保持不变。

## 3. 功能门与 fixtures

fixture 生成器直接复用冻结 golden `integer_chain`（中间 `score_q`）与 `select_fixedpoint`（最终选择），Wv 按 SHA256 复用已接受 bundle。真实 grounding/pointing scene 的 `group_mean_q`/grid/query 来自已接受 validation NPZ；该 NPZ 无原始 4x1024 tokens，故 token 字节用文档化的确定性 xorshift64* 流（scorer golden 与 token 无关，selector golden 就在这批字节上计算）。

| fixture | 覆盖 |
|---|---|
| scenes | 7 个不同 G：[9, 16, 50, 80, 128, 260, 400]，grids [[2, 25], [3, 3], [4, 4], [8, 10], [8, 16], [13, 20], [20, 20]] |
| budgets | 48 / 64 / 96 |
| real | grounding + pointing 的 correct prompt（G=260, 13x20, K=48/64/96），另含 wrong prompt K=48 |
| identity bypass | G=16 K=48、G=9 K=96 |
| 空/全等分数 | zero_g128（全 0 score）K=48/64/96、all_equal_g128（全等 score）K=48/64/96 |
| 空 region | empty_2x25 K=48 |
| 同分按小 index | 全 0/全等分数在 G>K 时进入完整 anchor/context/multi-pass 选择路径 |

| 门 | 结果 | 日志 |
|---|---|---|
| Python fixture/合同测试 | **Ran 7 tests, OK** | `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/logs/01_python_fixtures.log` |
| g++ 普通 | **383 checks, 0 fail；ISP_RESULT PASS** | `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/logs/02_gpp_test.log` |
| ASan + UBSan | **383 checks, 0 fail；ISP_RESULT PASS** | `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/logs/03_asan_ubsan.log` |
| Vitis HLS C-sim | **CSim 0 error；383 checks, 0 fail** | `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/logs/04_hls_csim.log` |

测试台逐字节核对：

1. 片上中间 `score_q` vs 冻结 Python scorer golden（debug 孪生）；
2. 最终 5 段输出（tokens/group_index/source_type/region_id/valid + status/counts/scale）vs 冻结 Python selector golden；
3. pipeline 最终输出 vs “既有 standalone `integer_selector_scorer` + `token_selector_core_packed`”顺序组合，逐字节一致；
4. packed v4 vs int8 v3 selector 差分，逐字节一致；
5. 同一输入重复运行逐字节确定；
6. 非法 G（0/4097/-1）、grid、budget（47）、local_cap（0/17）与空指针状态码。

## 4. 受控 csynth 结果

- project `/tmp/robobrain_integer_selector_pipeline_hls_csynth`；只 `csynth_design`；**0 error**、142 warning。
- `HLS 200-871` count = 0、`HLS 200-1016` count = 0、`HLS 200-790` loop constraint NOT satisfied = 1、II warning `200-885` = 5、`200-878` = 2。
- 无 timing violation，因此工具未报告 `200-1016` 关键路径；estimated period == effective budget（3.650 ns，slack 0.000），限制路径即为有效时钟预算本身。II 告警来自一次性的 Wv 载入流水（achieved II=8，与已验收 scorer 一致）。

### 4.1 时序与资源（工具报告值）

| 项 | 值 |
|---|---|
| target / uncertainty | 5.00 ns / 1.35 ns（默认，未改） |
| effective budget | **3.650 ns** |
| estimated period | **3.650 ns**（Fmax 273.97 MHz） |
| slack | **0.000 ns** |
| BRAM_18K | 328 / 1,824 = **17.98%** |
| DSP | 212 / 2,520 = **8.41%** |
| FF | 50,206 / 548,160 = **9.16%** |
| LUT | 90,891 / 274,080 = **33.16%** |
| URAM | 0 / 0 = **0.00%** |

### 4.2 时延（报告值 vs 公式推导值）

- **报告值**（compile-time trip-count 上界，G=4096/K=96/64x64 等最大界）：best 1、average undef、**worst 9,312,990 cycles = 46.565 ms @200 MHz**。
- **报告值**（scorer 子模块）：Wv 载入 **65,539 cycles**（一次性）、pass 1 **1,440 cycles/group**、pass 2 **402 cycles/group**。
- **公式推导**（`scripts/summarize_csynth.py` 的 model）：`scorer(G)=wv_load+G*(pass1+pass2)`；selector 项按 anchor（5.5 cycles/group）、per-region multi-pass（4.0 cycles/cell）、context pooling（1.4 cycles/word）与 bookkeeping 保守估计，单位成本取自本次 RPT 的 top loop 表。

| G | K | grid | scorer cycles（推导） | selector cycles（推导） | 总周期（推导） | ms @200 MHz（推导） |
|---:|---:|:---|---:|---:|---:|---:|
| 80 | 48 | 8x10 | 212,899 | 103,445 | **316,344** | 1.582 |
| 80 | 64 | 8x10 | 212,899 | 111,637 | **324,536** | 1.623 |
| 80 | 96 | 8x10 | 212,899 | 80,800 | **293,699** | 1.468 |
| 260 | 48 | 13x20 | 544,459 | 183,235 | **727,694** | 3.639 |
| 260 | 64 | 13x20 | 544,459 | 191,427 | **735,886** | 3.679 |
| 260 | 96 | 13x20 | 544,459 | 210,883 | **755,342** | 3.777 |
| 400 | 48 | 20x20 | 802,339 | 243,141 | **1,045,480** | 5.227 |
| 400 | 64 | 20x20 | 802,339 | 251,333 | **1,053,672** | 5.268 |
| 400 | 96 | 20x20 | 802,339 | 270,789 | **1,073,128** | 5.366 |

- 模型自校验（最大界 G=4096/K=96/64x64）：模型 9,462,678 cycles vs 工具报告 worst 9,312,990 cycles，ratio **1.0161**（保守 ~1.6%）。
- **G=400/K=96 推导总周期 1,073,128 ≤ 6,000,000**，5.366 ms ≤ 30 ms。

关键 selector 子函数时延（报告值，compile-time 上界）：

| 子函数 / 循环 | 最大 cycles |
|---|---:|
| context pooling per region (packed 16-lane MAC) | 140,653 |
| context-region pooling outer loop | 1,305,459 |
| per-region multi-pass max-select | 251,190 |
| single region band scan | 1,855 |
| context entry construction | 55,883 |
| output materialisation | 26,688 |

## 5. 异构传输审计（模型估算，非测量；已按冻结 5 段布局更正）

**Codex 审校更正（2026-10-03）**：旧 `image_bytes` 把 `selected_tokens + metadata` 只做**一次**尾部 64 B `align_up`，未按冻结合同 `evaluation.hardware_regular_token_selector.fpga_transfer_layout(budget)` 的 **5 段逐 offset** 对齐，K=48/K=96 的 FPGA→Jetson 总量与由它派生的网络/DDR 时延因此被**低估**。现在 `image_bytes` **直接调用**冻结函数实时产生 `fixed_budget_arena_bytes` 并保存 5 段 offset/size；**旧传输总量/时延作废**，以更正后的 arena 口径为准。决策保持 **`INTEGER_SELECTOR_PIPELINE_HLS_PASS`** 不变：HLS 功能/csim/csynth 证据未改，功能门、时序、资源与周期推导数字全部不变。

| G | K | output_count | 旧 F->J（单次 align_up）B | 新 logical payload B | 新 fixed arena B | arena budget | 低估 B |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | 48 | 48 | 196,864 | 196,848 | **196,928** | 48 | 64 |
| 80 | 64 | 64 | 262,464 | 262,464 | **262,464** | 64 | 0 |
| 80 | 96 | 80 | 328,128 | 328,080 | **393,792** | 96 | 65,664 |
| 260 | 48 | 48 | 196,864 | 196,848 | **196,928** | 48 | 64 |
| 260 | 64 | 64 | 262,464 | 262,464 | **262,464** | 64 | 0 |
| 260 | 96 | 96 | 393,728 | 393,696 | **393,792** | 96 | 64 |
| 400 | 48 | 48 | 196,864 | 196,848 | **196,928** | 48 | 64 |
| 400 | 64 | 64 | 262,464 | 262,464 | **262,464** | 64 | 0 |
| 400 | 96 | 96 | 393,728 | 393,696 | **393,792** | 96 | 64 |

- **权威口径**：`logical_payload_bytes` = 实际 `output_count` 个 token + 四种 metadata 的原始有效字节（无 padding）；`fixed_budget_arena_bytes` = 冻结 `fpga_transfer_layout(K).total_bytes`（5 段各自 64 B 对齐），**与 runtime `group_count` 无关**。G=80/K=96 identity bypass 的 `output_count=80` 与固定预算 arena K=96 分开报告。
- 网络/DDR 保守时间**统一基于 `fixed_budget_arena_bytes`**；scatter-gather payload 时间仅作为明确标注的**下界**，不替代固定 arena 口径。

- 1 GbE 有效带宽假设：线速率 125 MB/s，有效载荷 central 110 MB/s，敏感性区间 [90, 118] MB/s。
- 理论本地 DDR 路径：6 GB/s。
- Wv = 131,072 B；cold 口径每图随传，resident 口径一次性 131,072 B。
- 片上 `score_q` **不计入**任何外部传输。

Jetson→FPGA 每图字节数（cold 口径，含每图 Wv；数值未更正）：

| G | K | J->F tokens | J->F group_mean | J->F query | J->F Wv | J->F 合计 | 1GbE J->F low/central/high (us) | 6GB/s J->F (us) |
|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| 80 | 48 | 327,680 | 81,920 | 128 | 131,072 | 540,800 | 6008.9/4916.4/4583.1 | 90.1 |
| 80 | 64 | 327,680 | 81,920 | 128 | 131,072 | 540,800 | 6008.9/4916.4/4583.1 | 90.1 |
| 80 | 96 | 327,680 | 81,920 | 128 | 131,072 | 540,800 | 6008.9/4916.4/4583.1 | 90.1 |
| 260 | 48 | 1,064,960 | 266,240 | 128 | 131,072 | 1,462,400 | 16248.9/13294.5/12393.2 | 243.7 |
| 260 | 64 | 1,064,960 | 266,240 | 128 | 131,072 | 1,462,400 | 16248.9/13294.5/12393.2 | 243.7 |
| 260 | 96 | 1,064,960 | 266,240 | 128 | 131,072 | 1,462,400 | 16248.9/13294.5/12393.2 | 243.7 |
| 400 | 48 | 1,638,400 | 409,600 | 128 | 131,072 | 2,179,200 | 24213.3/19810.9/18467.8 | 363.2 |
| 400 | 64 | 1,638,400 | 409,600 | 128 | 131,072 | 2,179,200 | 24213.3/19810.9/18467.8 | 363.2 |
| 400 | 96 | 1,638,400 | 409,600 | 128 | 131,072 | 2,179,200 | 24213.3/19810.9/18467.8 | 363.2 |

FPGA→Jetson 每图字节数（cold 口径；所有时间基于 fixed arena）：

| G | K | output_count | logical payload B | fixed arena B | 1GbE arena central (us) | 1GbE arena low/high (us) | payload 下界 central (us) | 6GB/s arena (us) |
|---:|---:|---:|---:|---:|:---:|:---:|:---:|---:|
| 80 | 48 | 48 | 196,848 | **196,928** | 1790.3 | 2188.1/1668.9 | 1789.5（下界） | 32.8 |
| 80 | 64 | 64 | 262,464 | **262,464** | 2386.0 | 2916.3/2224.3 | 2386.0（下界） | 43.7 |
| 80 | 96 | 80 | 328,080 | **393,792** | 3579.9 | 4375.5/3337.2 | 2982.5（下界） | 65.6 |
| 260 | 48 | 48 | 196,848 | **196,928** | 1790.3 | 2188.1/1668.9 | 1789.5（下界） | 32.8 |
| 260 | 64 | 64 | 262,464 | **262,464** | 2386.0 | 2916.3/2224.3 | 2386.0（下界） | 43.7 |
| 260 | 96 | 96 | 393,696 | **393,792** | 3579.9 | 4375.5/3337.2 | 3579.1（下界） | 65.6 |
| 400 | 48 | 48 | 196,848 | **196,928** | 1790.3 | 2188.1/1668.9 | 1789.5（下界） | 32.8 |
| 400 | 64 | 64 | 262,464 | **262,464** | 2386.0 | 2916.3/2224.3 | 2386.0（下界） | 43.7 |
| 400 | 96 | 96 | 393,696 | **393,792** | 3579.9 | 4375.5/3337.2 | 3579.1（下界） | 65.6 |

固定 arena 的 5 段 offset/size（由冻结 `describe_fpga_layout(K)` 实时产生；与 cold/resident 无关）：

| K | total B | tokens offset/size | group_index offset/size | source_type offset/size | region_id offset/size | valid offset/size |
|---:|---:|:---:|:---:|:---:|:---:|:---:|
| 48 | 196,928 | 0/196608 | 196608/96 | 196736/48 | 196800/48 | 196864/48 |
| 64 | 262,464 | 0/262144 | 262144/128 | 262272/64 | 262336/64 | 262400/64 |
| 96 | 393,792 | 0/393216 | 393216/192 | 393408/96 | 393536/96 | 393664/96 |

resident 口径下每图少传 Wv 131,072 B（Jetson->FPGA），1 GbE central 下约 1191.6 us/图。

与“score_q 外部落 DDR 后再读回”的两核串接基线比较（同一 (G,K)）：

| G | K | 基线 score_q DDR 写+读 bytes | 片上融合节省 DDR bytes | 6 GB/s 理想节省 (us) |
|---:|---:|---:|---:|---:|
| 80 | 48 | 320 | 320 | 0.053 |
| 80 | 64 | 320 | 320 | 0.053 |
| 80 | 96 | 320 | 320 | 0.053 |
| 260 | 48 | 1,040 | 1,040 | 0.173 |
| 260 | 64 | 1,040 | 1,040 | 0.173 |
| 260 | 96 | 1,040 | 1,040 | 0.173 |
| 400 | 48 | 1,600 | 1,600 | 0.267 |
| 400 | 64 | 1,600 | 1,600 | 0.267 |
| 400 | 96 | 1,600 | 1,600 | 0.267 |

**边界**：片上融合消除的是中间 `score_q` 的 **FPGA 本地 DDR 往返**，**不消除** Jetson↔FPGA 的输入/输出往返。`group_mean_q` 仍是当前额外输入；若未来在 PL 内从 `tokens_q` 重建它，可潜在节省每图 G=80 时 81,920 B、G=260 时 266,240 B、G=400 时 409,600 B（随 G 线性，见 JSON `future_group_mean_reconstruction_saving`），但本轮**不改变**已验收的 group-mean 量化合同，top 继续同时接收 `tokens_q` 与 `group_mean_q`。

## 6. 冻结工程门与决策

| gate | 结果 |
|---|---|
| 顶层无外部 `score_q` 数据端口（源码 + 综合接口） | ✅ |
| 中间 `score_q` 与冻结 Python golden 逐字节一致 | ✅ |
| pipeline == standalone scorer + packed selector 顺序组合（逐字节） | ✅ |
| budget 48/64/96、G 80/260/400、grounding/pointing correct、identity/空/全等/同分/非法参数/确定性 | ✅ |
| 普通 g++ / ASan+UBSan / HLS C-sim 均 0 mismatch、0 sanitizer/runtime error | ✅ |
| csynth 0 error | ✅ |
| estimated period ≤ 3.650 ns | ✅（3.650 ns） |
| 各资源占比 < 50% | ✅ |
| G=400/K=96 推导总周期 ≤ 6,000,000 | ✅（1,073,128） |

→ **`INTEGER_SELECTOR_PIPELINE_HLS_PASS`**

## 7. 限制

- csynth_design only: not C/RTL cosimulation, not place-and-route, not a bitstream, not a board measurement; timing/latency/resource numbers are synthesis estimates and the per-configuration cycle counts are formula-derived.
- The on-chip fusion removes the intermediate score_q FPGA-DDR round trip; it does not remove the Jetson <-> FPGA round trip of the inputs/outputs.
- The transfer audit is a model with an explicit bandwidth assumption and a sensitivity band; it is not a measured link throughput.
- group_mean_q is still a separate input; reconstructing it from tokens_q in the PL is only quantified as a future saving and the accepted group-mean quantisation contract is unchanged this round.
- The derived model is conservative (1.6% above the tool's worst case at the maximum compile-time bounds) but is still an estimate, not a measurement.
- Codex review correction (2026-10-03): the first transfer audit used a single trailing align_up for FPGA -> Jetson and underestimated K=48/K=96. The deprecated transfer totals/times are void; only fixed_budget_arena_bytes / logical_payload_bytes and the arena-based times are authoritative. The HLS functional/timing/resource evidence is untouched and the decision is unchanged.

## 8. 复现命令

```bash
cd fpga_accel/integer_selector_pipeline_hls
make python_test
make test
make asan_run
make csim
make csynth
/home/zhaoxiangyu/conda-envs/robobrain25/bin/python scripts/summarize_csynth.py --report-dir ../../artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth --console-log ../../artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/console.log --out ../../artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/summary.json
/home/zhaoxiangyu/conda-envs/robobrain25/bin/python tools/build_report.py
```

## 9. 证据路径

- 源码：`fpga_accel/integer_selector_pipeline_hls/{README.md,Makefile,include/,src/,tb/,tools/,tests/,scripts/}`
- fixtures：`artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/`
- csynth：`artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/`
- 日志：`artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/logs/`
- 机读摘要：`reports/token_selector/2026-10-03-integer-selector-pipeline-hls.json`
- harness result：`reports/token_selector/logs/2026-10-03-integer-selector-pipeline-hls-harness-result.json`

## 10. 文件与 SHA256

| 文件 | sha256 |
|---|---|
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/console.log` | `291153151b0732e4e3545bc7b2de24887e707808728bf58e5763f35846d2d382` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/csynth.rpt` | `789fdb6b2a13f6531f3023bd4c71f0538e499298f3444c39ca9622eb1692dadf` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/csynth.xml` | `d85079ca9db447a0a7c96f17528377c1354284505a04e25724869bb1eeef97e0` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/csynth_design_size.rpt` | `e11fe0076e7ef7690caf829ef074aa98f09562a5777da1eaa33f0db354b3b228` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/csynth_design_size.xml` | `94e9dba8f6d1496aa17f76ae481565306e17c24aa5cb9e7d3f183927929d228f` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1013_9_csynth.rpt` | `bb756a44abe1b0c1be124f6a49759fb054d63e0669e65c18501896397766d22f` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1013_9_csynth.xml` | `571e9d7bc009ac1a1c762ee5dc85c7158eecfc3d1d33d00c7762a45d9d24cea5` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_103_1_csynth.rpt` | `375d1f2bcb1fb851d48a302c559c4859dcf286b3f5db21ae5a7859ed8b1ea5f5` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_103_1_csynth.xml` | `24668f50b9fab39e2f187b6c38318041663e5eca20f9109b0ae51534628459ff` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1040_12_VITIS_LOOP_1042_13_csynth.rpt` | `1269c83dcf17e68f32433849742670f88c7a79e6e5a686fa8c13679e65e598e3` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1040_12_VITIS_LOOP_1042_13_csynth.xml` | `07d2b977363f1585a925bc0a1f6c9e1ec5c328f41e82f6e3f7879c11a285df1b` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1074_14_csynth.rpt` | `0e4e9bd928aa3505c2d2aee2f4c87083eaeb6e7deff6733d974a4aaa055e5705` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1074_14_csynth.xml` | `58aeadccc43e2f569ce76f5873cb672d1e55ec529b7c305c51f5614ae0d3794f` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1079_15_csynth.rpt` | `3b3b1df63f74545bcd360c24ddbebb94401c460dbcc22980538a65f638928e18` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1079_15_csynth.xml` | `2c93e59015bb4aea7743cec62fd1096555927600222dc4bd06d56bfbe28e176d` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1089_17_csynth.rpt` | `09d665a5b6adf10df656cf81c4d49803fe6b10f66b3ac4c0c8ced5b52bfb0490` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1089_17_csynth.xml` | `f9c35a750ae868a0057b9ddb99a02d16d02f3c2047c4a1215aa08c3774f124a7` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_109_2_csynth.rpt` | `d034ff7dc9d7ef46598f0bf48ccc1d1d4d996b49b293f96e486dcdf7cf07acc8` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_109_2_csynth.xml` | `551e4f284904a3fec03c24a2ad03a049f109a47989572922f3316d1351c906f7` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1107_18_csynth.rpt` | `5d18f7c5f57b77ef06f40694c554c6cbb86e23a963e44c318c60d9057b08cce7` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1107_18_csynth.xml` | `86862d4ab64b4de8692915fde1da0d95a49341e75e7fa4ab2cb2894c4eb2c639` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1116_20_csynth.rpt` | `60a335b38466a608f00bbbb65c9f1c0f6b77ef3bc8d1477218a9d076332a1b9b` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1116_20_csynth.xml` | `b5df3e1289e888cc0b74bd439abc1c121112674a4f9f302e78f6f91c0e911456` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1132_21_csynth.rpt` | `084d9e8a1f8291ef70f96c1d1fe5e0cbc14290c6e4600e183939a1e155723079` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1132_21_csynth.xml` | `621cb5a20086c28146afdc4181bad44395ed886791b0ce986213e548f83fec90` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1137_22_csynth.rpt` | `5b538c322801035d1a8ccbb0a4832a47d6f0e2b66f47502b896a1a1c512da19d` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1137_22_csynth.xml` | `416ab9bd409755a77572d6d8dc3670961b9de25730baea33215bffa326843882` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1166_24_csynth.rpt` | `2d1bca0f4b36dfc1927bff9b77b7c61f2ef16aa2c2f587800540dd81ee7c7b4e` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1166_24_csynth.xml` | `aed2b25761bc16d387a9af51b03c96b5d5298a3ebf7f1e095e9bda0cbbe2f8ae` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1176_26_csynth.rpt` | `b93d9bc1c025c3921461c268cf2299dab3aa656af8cac2e76e5bec5548fba117` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1176_26_csynth.xml` | `8c4fa337803e169de5972aeb1d3b3832c3d60909b47175d78ac0054024bed8e3` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1196_27_csynth.rpt` | `1d01a1446a23eb7244a779254b3f158882b6a51dd7d1ad480bf7a18f00d02abf` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1196_27_csynth.xml` | `70cd8991b3bfbb72d944e088720ec2adc2eacce70bdbb7bcd0dc0addac2722b4` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1212_29_csynth.rpt` | `1c215896b1e73c238edd33f9222a393938e2336cca5574f5c4604b46f03c0e0d` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1212_29_csynth.xml` | `fed54799c017b208814d87134f4b355c5d463b74e45d972cf64fd96dad47b398` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1227_31_csynth.rpt` | `751202559d91510bbee217d799a4c337916f04645fda77e4c0915fa20787ad6c` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_1227_31_csynth.xml` | `8053eefb3d0ade27dcbd326bfc0b23009e932594f19be3dd59ebd9f17deb748f` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_127_12_csynth.rpt` | `4fde0003f738278b48685ad4fee2b0a408de9c56e3ae8fda29d20447d48a522b` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_127_12_csynth.xml` | `37005af87059f7a2eeef77461a7af05f75defa963e3413df518f1cdc2448261c` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_127_1_csynth.rpt` | `77099baf2f06fde5151c0b8d9d194be7e2871f69c39cef59e7a2a17de121c2ab` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_127_1_csynth.xml` | `fd336afe268181194b31b5120b86ad660e2fae8b643c86779c7a9fa72f41aad5` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_151_1_csynth.rpt` | `4c8f0eb56877a247756ee6aa642ab5046b79930b9031dc5a16e88f23efcbb5f0` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_151_1_csynth.xml` | `2675a0c711ecdcd9e35be05d6af65552ffc98dab33617cfc41c9f073a2d5c882` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_175_2_csynth.rpt` | `0afae099603a5075a4f18835d265a4f75cec6a947544d160312b97a9f541a9ee` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_175_2_csynth.xml` | `275da97c047773a5e6fb67f33d1c9dd1b617a23593269c50d9eb98855a499727` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_66_11_csynth.rpt` | `4725809b83088b4e65759e1b621924043a6bff4af227a320e233b0ad6194ad83` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_66_11_csynth.xml` | `e3a9690dd7c9451e4b4b8eda52e21d067ede96c3456403f5bcbf1c58e80e4d63` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_66_1_csynth.rpt` | `b97c6e291bba96040624dbdc7302c675f6932e8f2fc53b6accc81d0c7a2c4892` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_66_1_csynth.xml` | `45841cd7d6682ee317192f56984bd475f4d31030733450924f8de6238646b92e` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_944_5_csynth.rpt` | `4f6e5d74a87f7c5ac6196a1b98b9048e85668b1921f251b3089f4844caab7298` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_944_5_csynth.xml` | `dd5f32f1b62a5faaeb61a7928765794c9329ad75cb06f1980f900421d743604a` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_969_7_csynth.rpt` | `ad4bda47f78af081e5078fe436dc1cea31d91d19b3b5409752131851025ae85b` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_969_7_csynth.xml` | `8a5ade1f60e38c17598a155c38727d51a54bc1a4f53abc31fbe3360f45d757e3` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_985_8_csynth.rpt` | `7f84bddd544828b3c8f6816d3c721807953bbed2e5e78237690e64c2ff171c50` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_Pipeline_VITIS_LOOP_985_8_csynth.xml` | `4ce52d5b96ca3ddcd44f9695d3f10cfe1ee2d4654629ad8379c9b160f6b73a39` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_csynth.rpt` | `9c73e75de4273d28e48f9986c870cb9916c21df73405d1c2d242dbb5810894e7` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/integer_selector_pipeline_csynth.xml` | `f6cb15233f0924e19c55041dae0f941bfff2ab5387aed5a90e2ec7b07cb2b595` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_136_1_csynth.rpt` | `889a92e7e28cc835de9c71d7d5844057a8b650c6acfd241ec0f0caa8bb5956e6` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_136_1_csynth.xml` | `7c3f0e1097c58d29216fa9e7636f66a44e85fb33c82e0bfab6b56b7f6d12f1d7` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_148_3_csynth.rpt` | `45633166b80b484cdd7f5d3f3a19ccdd175dc0ddeaaee75f1feea51f6c66bf7f` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_148_3_csynth.xml` | `1d79131ab75a837d13145d9e52578df95f382335426241ba7c4230d9b1ead8f3` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_168_6_csynth.rpt` | `45299f3420027e734bbfc016c826a38e3047dd50488430dc5c59ef373e16cc93` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_168_6_csynth.xml` | `f4299b7224125a9bf801db99724499551a408cea5015382b8578f39672c9dee9` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_188_9_csynth.rpt` | `c36f180e50eee17aa269bf3e44fc1028b389c619ca4d944eb5b0d231f0a210c8` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_188_9_csynth.xml` | `8cba59cdd015b005abb45d5ff4c8332f4e1a47bbdcba4af8f436b14f37355ab8` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_201_11_csynth.rpt` | `211e1d11e1ae5bccf0401f48209006882a99f3fc16206e09fed612f63877b6ab` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_201_11_csynth.xml` | `573c1474016c3a8e5b6b948d0568f1769dd9645c2f6cbabb5ce9868d3503cf3e` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_59_13_csynth.rpt` | `f2a1a80d220de2120b9a1b0cf09a0c706c8755f9d273e9f833e3c1199f351a76` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_59_13_csynth.xml` | `840de9958b6639b27b91ae8d9a5e296efb3e31e8d5ce5481f8710be3fb61628a` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_59_1_csynth.rpt` | `58e60c4a0bfbd2f7a49b6f12bd26b3bb8394bca16d7700b8515646e2a5d454fa` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_59_1_csynth.xml` | `c587b48ad17c2f8e69ff30d2f087542f0d32bf1d6518f789ff77c71c1a80e766` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_14_csynth.rpt` | `5cbedb7716e5a06fbd7a6d70f11913af0c6828144a3b3bda2ea4cbbf311446af` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_14_csynth.xml` | `8c7951cdd26bef2d48ae2d8ec9c3bd9325f54417d41c6f08a927cc76f528711e` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_15_csynth.rpt` | `1b445b8133d379c412a87478036467de82be36fad3bb3175d1b0f2fffd0bde8a` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_15_csynth.xml` | `4a9cec8365ab96258efd5737a051c01b74d424897e6f2f93373a2ecdaf5783a6` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_16_csynth.rpt` | `118ad6b736e1a30ecd62b7c2cd1f5ee60e84cb8664d33066c1fc78ef81fb594a` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_16_csynth.xml` | `2a61d3962795e16821392de838bf36e71d2b0685c3e794f56c49ccd74a2b2725` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_1_csynth.rpt` | `bc507aefcb5fddf9f263b94eef37fa5100e51e0448e780a4d32be94896eceab1` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_1_csynth.xml` | `6fe229dbdc41347edf6fe7043897c404056f51081cce698914f57b55b82cdc47` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_s_csynth.rpt` | `4fd102d7e758f1516255e932e74869584b4947c01565f877d8320619930c7873` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/p_anonymous_namespace_is_scorer_core_false_s_csynth.xml` | `4563c01264ed04c811a6726eeb1b744213b7c217e6635e483e4a50c739f8ee1d` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/summary.json` | `f2bd188b032d85e7f28c1b1238f2b7a029df3ff1c0201ed13907845371e88fe5` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_103_1_csynth.rpt` | `05c600437a6cebc774a3d9a089fcb350f4459e05ee47c31ab404125a24ba2093` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_103_1_csynth.xml` | `f3f748349f1489fcc9a3cf5e149ff9920f0dacbc1ffa17df32678b1b26060c2f` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_109_2_csynth.rpt` | `fa174d4cb6130081909728dce9bc7c19f648bdb573a522578a3d17704d605e93` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_109_2_csynth.xml` | `dc6fa3e37b5a4431720cb3ba1c8f76a6e736c374f2edf6d4e51a2dafe9675a93` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_272_1_csynth.rpt` | `dbaee674b04aaa585b6c1c0d0fdca05ab26613fbf8554940b75c433c1125dc1b` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_272_1_csynth.xml` | `e98c0a9a9de6c70d5e56446e9707df88329287dd2baa7f3ab2820dd1d1047655` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_276_2_VITIS_LOOP_281_3_csynth.rpt` | `3f9018f2800cda05e50f5a4ffe762069c180849dd2c17b4074b771b35caf85fc` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_276_2_VITIS_LOOP_281_3_csynth.xml` | `4b27b45c99228fecfd982e9e84da83ccf93e460bb1e9e315341b2fbc7d82339b` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_315_4_csynth.rpt` | `4c0ba98d4e75714f8b6eb2a51f00f03a8de88746cc8fd2717ecc6801a2b5868b` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_315_4_csynth.xml` | `0e7395818cf7592f7ef7abeb9e47c891157b9c285e17cb2a4b62e79f9c8a03e2` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_320_5_csynth.rpt` | `a2e603aed9480d39d62e80694bbc594ee3689b517ef2aef8549cdd1ae8249115` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_320_5_csynth.xml` | `971595b8be138b84f18b287d6fd90fed95dbf136bbab83a174aaa3682933ee1e` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/ts_pool_context_packed_csynth.rpt` | `2cd7bc92274a226aced4d2b81a08e9da200528a5bb5912a6d5fbebd170a9f5f5` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/ts_pool_context_packed_csynth.xml` | `1080d282a155f3a235a6cd92ab56eace367968086a6ae78224c196dab9b3e95a` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/FORMAT.md` | `78e848cfe523ac8e2db4c2e8aec09185ac0e8652f328e6e691de820b5163aebb` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/all_equal_g128_K48_cap16.exp.bin` | `a013d90d4086c2c05eec4ceba0401e3803356e0f5287b421a6013ef815e3e567` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/all_equal_g128_K64_cap16.exp.bin` | `a634ce8fa57ffaafcae515c982a2db365869d2740468f2e035e0591973f1809a` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/all_equal_g128_K96_cap16.exp.bin` | `08f5ee3227d58d87465597a7f0b70b82719c7cb6efaa6604e30794c57c5b5c8f` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/empty_2x25_K48_cap16.exp.bin` | `01e901182347930be41aae754a69ed607440b64c9bd84cdc7934712dabc3903b` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/identity_g16_K48_cap16.exp.bin` | `4ae9bb8820abbae4c1a30cb1f8547dbfcfd68538affe2c9663dc02265ba49727` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/identity_g9_K96_cap16.exp.bin` | `c99b638e5ef7ee678165741cb88aed72d89c3c970d36bad740ef1d5e5046c33f` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/manifest.json` | `db4debe3c8dee5e925af869abbb29a42f36e8e413474b5f14a2f94ce0a8785dd` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/manifest.txt` | `1aaaa8ab3af84776e944b11b34e2081a402b5829b8ce5c348babafac1d68f037` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/real_grounding_correct_K48_cap16.exp.bin` | `0a8ddafc8c2060bd00d88b97d0f49babb4d5c6b83f040b94a986c1f554d3a4f5` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/real_grounding_correct_K64_cap16.exp.bin` | `5bab6b67f02584c9b22044aad2940f8e94ad9919af9d58d6757fd13fd26982aa` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/real_grounding_correct_K96_cap16.exp.bin` | `1e84f8bc3b76f0070b1dcd1f1343031e81cc05e342d2dc6f3453b5adac7a49de` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/real_grounding_wrong_K48_cap16.exp.bin` | `2c84c54a83e2774eb351cb65346de107ebd3785bf5307454b3f55526d8795757` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/real_pointing_correct_K48_cap16.exp.bin` | `1f868d572bd15a3142f246c432a3278611226e64e9436649fcc1ace1ea69a758` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/real_pointing_correct_K64_cap16.exp.bin` | `3e8505f68158c3b660f27c27f39fd2178b1911d9ed5083147a72fb9b39fc35ca` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/real_pointing_correct_K96_cap16.exp.bin` | `161c4ea3348bbb5607e8d6b6ff742fdd886e2e5c3a58db422d4d57bd262a4144` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/real_pointing_wrong_K48_cap16.exp.bin` | `cbfca2dd627bde206ac4c10571e8feb5548a057aac324f24fd6e893485e025cc` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/scene_all_equal_g128.bin` | `2d1e8c7dc90ce9d3c7a5dfad88630d77ffa67e2f082fc22d3304a1124064684e` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/scene_empty_2x25.bin` | `2de8b7aae53aa544013e8f8d180232ae471ad76a3eaeb45f826a5842561bb3f8` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/scene_identity_g16.bin` | `65eda767fb30a1f5d35fe69065790744b1a89403cfba1ed368970f4ace4c59b3` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/scene_identity_g9.bin` | `49da82151ed629ab70b34541c2acb77825d44107bffdc6106df8ac7bf606f012` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/scene_real_grounding_correct.bin` | `e550c5b7f5c441f24b41396b81d5d356c2b86e62c810fb82e3ef1ef983e0156c` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/scene_real_grounding_wrong.bin` | `25e45429d9e1d176d10d0d1e93f628042dd29194c3ae66a984fca4ae841ed39d` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/scene_real_pointing_correct.bin` | `f95a9ea7cf5e350605bca4ac020e9d70b688208266d89a72d5693904c98e9181` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/scene_real_pointing_wrong.bin` | `5c71f1ddc9ded24a9642b4f578c1bba61d8b223c8ea676dafe2e5eebde60d096` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/scene_synth_g400.bin` | `3201f1fadb04160474ea920af8e82e2ab311da90fd0fa050f367741aa0495b9f` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/scene_synth_g80.bin` | `a73fabbc485ed91279b65985474c7bdcab6f214e3f7d22f2e84f504c3b1e1662` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/scene_zero_g128.bin` | `7306421a7d3567a40b8cbb1317a7efee94043c282a399b6a7beef2a8d5685f40` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/synth_g400_K48_cap16.exp.bin` | `2c6b37489b06424a5f0fc9396b678cc2cdb689c898ca2a4481f6e0d11158b003` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/synth_g400_K64_cap16.exp.bin` | `cbd3e532dfc86ad18a2558368b02ea967d5d8b6564cdc7fa141f33698369d561` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/synth_g400_K96_cap16.exp.bin` | `e58c697beb51b6a1eaa3ea508ddf1bb37e9c32af5d73907abccfb820b658a4bd` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/synth_g80_K48_cap16.exp.bin` | `e8b68b99cf467e3cf5a0d177268b91e4046a905fefd6f518c8e940a442958aa3` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/synth_g80_K64_cap16.exp.bin` | `b20e2174d3293342e3b0a9d8427f3a4a7f879881079b269f21ca402986121c18` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/synth_g80_K96_cap16.exp.bin` | `960b083a38de6f38e7760ffaf9ff45fe7edd5d59bbc344501be7e939c61e9539` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/wv_q.bin` | `126bd8421dd128a318115c301bf9d1c7d336912150d19d2ec35203ed55d89d3b` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/zero_g128_K48_cap16.exp.bin` | `11ad094308c2460247025d1dc7aa3c3b0ebdfa500b4e2df04c7eb91d8cb07aff` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/zero_g128_K64_cap16.exp.bin` | `6d063923f240e594c24025a10910b64b493900d047cdcdaeda39b00a86ed5fd1` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/zero_g128_K96_cap16.exp.bin` | `7ef7fd0204ceb9e962ee412d3e4eb9b2dad0d3b49d3c6f8e036edd21682ff4c1` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/logs/01_python_fixtures.log` | `d757dc32fd0ea686e88f37b2bd8e5c533be74b695672c5cca0ffdecf7e222a64` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/logs/02_gpp_test.log` | `a80acc21aedd3a972b60b06acb8bc51af7f733b5b03be4c6fd2434ab131df67a` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/logs/03_asan_ubsan.log` | `1fe1cdbffc15b7aaf5d3b0580a933af177a3b520827fa8f1010895bbeb46e223` |
| `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/logs/04_hls_csim.log` | `c58b97374ded4a7a92f22aac832668df6b08b90012642db9535faafd9b2db1b2` |
| `fpga_accel/integer_selector_pipeline_hls/Makefile` | `df9339bf78cf635e79dd68cb725768bbdc39dbd72180e5145b9299f5c2b4328b` |
| `fpga_accel/integer_selector_pipeline_hls/README.md` | `878ed5963c5ce07ec9387dd89cf8ab90b69d8324408c6a241272ad7cd848799d` |
| `fpga_accel/integer_selector_pipeline_hls/include/integer_selector_pipeline.h` | `f3496454efc57c591c95f16ad42f6158c66d7df3ce93691446bf9b0ff4e7b725` |
| `fpga_accel/integer_selector_pipeline_hls/scripts/run_csim.tcl` | `b8c18b9fb4ee2104a7890db9f57d49b5427589edc9173387f3a5ddb2ea7dea6c` |
| `fpga_accel/integer_selector_pipeline_hls/scripts/run_csynth.tcl` | `4af56843d6f66b268df27ab5ebb74de68582ded3a6ddc53b14daf286a1a248c5` |
| `fpga_accel/integer_selector_pipeline_hls/scripts/run_stage.sh` | `3a1717096b8b2fba4e1fce9b85334e2625ff9fc61f018fe572e2148b5123fc34` |
| `fpga_accel/integer_selector_pipeline_hls/scripts/summarize_csynth.py` | `eba57b947a99ec13718ee8579701785aa0aaf458cddae93ef77b4425335758da` |
| `fpga_accel/integer_selector_pipeline_hls/src/integer_selector_pipeline.cpp` | `bf75cd4427d3ff288bba5c3fcae6a22b64598eb2ed8d227bbf4c06e89c731cdb` |
| `fpga_accel/integer_selector_pipeline_hls/tb/tb_integer_selector_pipeline.cpp` | `1c31c8f3c6ec31a398b756a71776922617d85cd1d87748160873e66c2af15a0b` |
| `fpga_accel/integer_selector_pipeline_hls/tests/test_integer_selector_pipeline_hls.py` | `7f4e19f339b0d37553e6cd74eb16afca176f641c055715c019871ec2334074c2` |
| `fpga_accel/integer_selector_pipeline_hls/tools/build_report.py` | `17ad978262927473c4d892c4e9f0cb85af6c09ce0878cfe495170577de2c8526` |
| `fpga_accel/integer_selector_pipeline_hls/tools/fixture_format.md` | `78e848cfe523ac8e2db4c2e8aec09185ac0e8652f328e6e691de820b5163aebb` |
| `fpga_accel/integer_selector_pipeline_hls/tools/generate_fixtures.py` | `701b54c102c992a1376978588d0c112126ea3bf24ea4cc70febbe13b2683c63b` |
| `fpga_accel/integer_selector_pipeline_hls/tools/transfer_audit.py` | `d6218b5456629bdc33e0964dd22446e9ea035bb3f6345dc17564cf2faf27302a` |

