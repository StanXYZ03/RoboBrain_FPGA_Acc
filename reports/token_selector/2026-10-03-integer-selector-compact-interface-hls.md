# 紧凑 AXI 接口整数 Selector 流水线 `integer_selector_pipeline_compact`：实现、验证与受控 HLS csynth

- 日期：2026-10-03
- 决策：**`INTEGER_SELECTOR_COMPACT_INTERFACE_HLS_PASS`**
- 范围：**只 `csynth_design`**（Vitis HLS 2023.2、part `xczu9eg-ffvb1156-2-e`、`create_clock -period 5.0`、tool 默认 uncertainty，未放宽）；无 C/RTL cosim、无 IP 导出、无 implementation/bitstream、无板卡、无 GPU、无联网、无训练、无调参。
- 只读复用两个已验收 HLS 核源码与冻结 Python golden；**未改**受保护目录。

## 0. 结论摘要

新增独立顶层 `integer_selector_pipeline_compact`，把已验收的片上 scorer + packed selector 包进三个 128-bit `m_axi` bundle 与一个 AXI-Lite `control` bundle：

```
input_arena (tokens | group_mean | query) + wv_q
     -> 已验收整数 scorer 核 -> score_q[G]（片上局部缓冲，不是端口）
     -> 已验收 packed v4 selector 核 -> output_arena（tokens + 4 段 metadata）
```

- 四道功能门：Python 16/0、g++ 244/0、ASan+UBSan 244/0、HLS C-sim 244/0（0 error）；g++/ASan/C-sim 均 dlopen 已验收 `integer_selector_pipeline` 参考实现并逐字节比对。
- 生成接口：m_axi = `input_arena`, `output_arena`, `wv_q`（**3 个**）；AXI-Lite = `control`（**1 个**）；外部 score 端口 = 0。
- `csynth_design` 0 error；estimated period **3.65 ns** ≤ requested 5.0 ns，effective budget **3.65 ns**，slack **0.0**，Fmax **273.97 MHz**。
- 资源占 ZCU102：BRAM_18K **16.17%**、DSP **8.41%**、FF **8.24%**、LUT **33.23%**、URAM **0.00%**（均 < 50%）。
- 输出 arena 权威总量 K=48/64/96 = **196,928 / 262,464 / 393,792 B**，5 段逐 offset 64 B 对齐，gap/unused 全 0。
- **公式推导** 生产用例：G80/K48 **328,699 cycles = 1.643 ms**；G260/K64 **752,342 = 3.762 ms**；G400/K96 **1,097,802 = 5.489 ms** @200 MHz。

## 1. 顶层接口合同

```
int integer_selector_pipeline_compact(
    const ts_word_t* input_arena,   // [G*320 + 8] 128-bit words
    const is_word_t* wv_q,          // [8192] 128-bit words
    ts_word_t* output_arena,        // [total_bytes/16] 128-bit words
    int group_count, int grid_h, int grid_w, int budget, int local_cap,
    float scale_in, float* scale_out,
    int* output_count, int* real_count, int* context_count);
```

| 项 | 定义 |
|---|---|
| `score_q` | `uint16_t score_local[IS_MAX_GROUPS]` **顶层函数内局部数组**；无端口、不落 DDR |
| m_axi | 仅 `input_arena`、`wv_q`、`output_arena` 三个 128-bit bundle |
| AXI-Lite | 仅 `control` 一个；三个 m_axi 指针各自的 offset 寄存器用显式 `s_axilite port=<ptr> bundle=control` 合并进 `control`（否则 Vitis 2023.2 会另建 `control_r`） |
| 状态码 | 与 `integer_selector_pipeline` 逐值一致（`static_assert` 编译期锁定） |

## 2. 冻结内存合同与布局证据

输入 arena（偏移为 arena 基址的字节偏移，均 64 B 对齐）：

| 段 | 偏移 | 大小 |
|---|---|---|
| `tokens_int8` | 0 | `G*4096` |
| `group_mean_int8` | `G*4096` | `G*1024` |
| `query_int8` | `G*5120` | 128 |
| 合计 | | `G*5120 + 128` |

输出 arena 权威（`evaluation.hardware_regular_token_selector.fpga_transfer_layout`，5 段各自 64 B 对齐）：

| K | total B | tokens off/size | group_index off/size | source off/size | region off/size | valid off/size |
|---:|---:|:---:|:---:|:---:|:---:|:---:|
| 48 | 196,928 | 0/196608 | 196608/96 | 196736/48 | 196800/48 | 196864/48 |
| 64 | 262,464 | 0/262144 | 262144/128 | 262272/64 | 262336/64 | 262400/64 |
| 96 | 393,792 | 0/393216 | 393216/192 | 393408/96 | 393536/96 | 393664/96 |

Python 布局测试对 K=16/32/48/64/80/96 与 G=9/16/50/80/128/260/400/4096 同时核对公式与冻结权威函数，并编译 C++ 端的 `ispc_fpga_layout`/`ispc_input_layout` 镜像逐字段比对。

## 3. 功能验证

fixture 生成器只读复用已验收 pipeline 的 scene/case/Wv（`accepted_generator_sha256`），golden 直接来自冻结 `integer_chain` + `select_fixedpoint`，输出偏移实时来自冻结 `fpga_transfer_layout`/`describe_fpga_layout`；未复制第二套数学或 offset 算法。

| 门 | 结果 | 日志 |
|---|---|---|
| Python 布局/fixture/接口测试 | **Ran 16 tests, OK** | `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/logs/01_python_fixtures.log` |
| g++（dlopen 已验收 pipeline） | **244 checks, 0 fail；PASS** | `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/logs/02_gpp_test.log` |
| ASan + UBSan | **244 checks, 0 fail；PASS** | `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/logs/03_asan_ubsan.log` |
| Vitis HLS C-sim | **0 error；244 checks, 0 fail** | `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/logs/04_hls_csim.log` |

主机测试台逐字节核对：compact arena vs 冻结 Python golden、vs dlopen 的已验收 `integer_selector_pipeline`、vs 导出 `integer_selector_scorer` + `token_selector_core_packed` 顺序组合；并检查 status / 三计数 / `scale_out` / 全部 padding / arena 尾部 / canary / 重复确定性。

fixture 覆盖：K=48/64/96、G=9/16/50/80/128/260/400、真实 grounding/pointing correct+wrong prompt、identity bypass（G16/K48、G9/K96、G80/K96）、全 0 分数、全等分数、空 region（2x25）；非法 G/grid/budget/local_cap/空指针均为兼容状态码且不越过 canary。

## 4. 受控 csynth 结果

- project `/tmp/robobrain_integer_selector_compact_hls_csynth`；只 `csynth_design`；**0 error**、190 warning。
- `HLS 200-871` = 0、`HLS 200-1016` = 0、`HLS 200-790` loop constraint NOT satisfied = 1、II 200-885 = 5、200-878 = 2。

### 4.1 时序与资源（工具报告值）

| 项 | 值 |
|---|---|
| requested / uncertainty | 5.00 ns / 1.35 ns（默认，未改） |
| effective budget | **3.650 ns** |
| estimated period | **3.650 ns**（Fmax 273.97 MHz） |
| slack | **0.000 ns** |
| BRAM_18K | 295 / 1,824 = **16.17%** |
| DSP | 212 / 2,520 = **8.41%** |
| FF | 45,156 / 548,160 = **8.24%** |
| LUT | 91,089 / 274,080 = **33.23%** |
| URAM | 0 / 0 = **0.00%** |

### 4.2 时延（工具报告值 vs 公式推导值）

- **工具报告**（compile-time 上界）：best 1、average undef、**worst 9,337,705 cycles = 46.688 ms @200 MHz**。
- **工具报告** scorer 子模块：Wv 载入 **65,539 cycles**、pass1 **1,440 cycles/group**、pass2 **402 cycles/group**。

| G | K | grid | scorer（推导） | selector（推导） | compact 开销（推导） | 总周期（推导） | ms @200 MHz（推导） |
|---:|---:|:---|---:|---:|---:|---:|---:|
| 80 | 48 | 8x10 | 212,899 | 103,445 | 12,355 | **328,699** | 1.643 |
| 260 | 64 | 13x20 | 544,459 | 191,427 | 16,456 | **752,342** | 3.762 |
| 400 | 96 | 20x20 | 802,339 | 270,789 | 24,674 | **1,097,802** | 5.489 |

- 模型自校验（G=4096/K=96/64x64）：模型 9,487,352 vs 工具报告 worst 9,337,705，ratio **1.0160**（保守）。
- 以上生产用例周期为**公式估计**，不是工具测量；工具只报告 compile-time 上界时延。

## 5. 异构传输估计（模型，非测量；fixed arena + Wv-resident）

- 1 GbE 有效带宽假设 central 110 MB/s（灵敏度 90–118 MB/s）；本地 DDR 6 GB/s。
- 输入 arena = `G*5120 + 128` B；Wv = 131,072 B，resident 模式每图 0 B。
- FPGA→Jetson 网络/DDR 时间统一基于 fixed arena（`fpga_transfer_layout(K).total_bytes`），`logical_payload_bytes` 仅作为下界参考。

| G | K | input arena B | J→F Wv-resident B | J→F 1GbE central (us) | fixed arena B | F→J 1GbE central (us) | F→J 6GB/s (us) |
|---:|---:|---:|---:|:---:|---:|:---:|---:|
| 80 | 48 | 409,728 | 131,072 | 4916.4 | **196,928** | 1790.3 | 32.8 |
| 80 | 48 | 409,728 | 0 | 3724.8 | **196,928** | 1790.3 | 32.8 |
| 260 | 64 | 1,331,328 | 131,072 | 13294.5 | **262,464** | 2386.0 | 43.7 |
| 260 | 64 | 1,331,328 | 0 | 12103.0 | **262,464** | 2386.0 | 43.7 |
| 400 | 96 | 2,048,128 | 131,072 | 19810.9 | **393,792** | 3579.9 | 65.6 |
| 400 | 96 | 2,048,128 | 0 | 18619.3 | **393,792** | 3579.9 | 65.6 |

- identity bypass（G80/K96）`output_count=80`，但固定预算 arena 仍为 K=96 的 393,792 B。
- 片上 `score_q` 不计入任何外部传输；紧凑化消除的是本地 DDR 中间往返。

## 6. 门与决策

| gate | 结果 |
|---|---|
| protected_trees_unchanged | ✅ |
| functional_gates_pass | ✅ |
| exactly_three_m_axi_bundles | ✅ |
| exactly_one_axilite_bundle | ✅ |
| no_external_score_port | ✅ |
| zero_errors | ✅ |
| estimated_period_le_5ns | ✅ |
| timing_met | ✅ |
| all_resources_under_50_percent | ✅ |
| output_arena_k48_k64_k96_exact | ✅ |
| derived_production_cases_finite | ✅ |

→ **`INTEGER_SELECTOR_COMPACT_INTERFACE_HLS_PASS`**

## 7. 限制

- HLS evidence only: csynth_design plus host g++/ASan/UBSan and Vitis HLS C-simulation.  This is NOT RTL co-simulation, NOT place-and-route/routed timing, NOT a bitstream, NOT a board measurement and NOT a model end-to-end latency or task-accuracy result.
- The per-configuration G80/K48, G260/K64 and G400/K96 cycle counts are FORMULA estimates built from the tool-reported per-loop unit costs; only the reported best/average/worst latency (compile-time trip-count bounds) is a tool number.
- The transfer audit is a model with an explicit 1 GbE effective-bandwidth assumption and a sensitivity band; it is not a measured link throughput.
- The compact envelope removes the intermediate score_q FPGA-DDR round trip and merges the input pointers into one arena; it does not remove the Jetson <-> FPGA round trip.
- group_mean_q is still supplied inside the input arena; reconstructing it from tokens_q in the PL is not attempted here and the accepted group-mean quantisation contract is unchanged.
- Wv is assumed to remain resident in FPGA-attached memory in the corrected resident mode; the cold mode number is reported for completeness.
- No IP export, no cosim and no implementation/PnR were run; the AXI-Lite interface count is proven from the generated csynth report/XML only.

## 8. 受保护目录 SHA256

- baseline `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/protected_trees_before.json`（2051 文件）
- `fpga_accel/integer_selector_scorer_hls`：14/14 未变，changed 0，missing 0 → OK
- `fpga_accel/token_selector_hls`：14/14 未变，changed 0，missing 0 → OK
- `fpga_accel/integer_selector_pipeline_hls`：18/18 未变，changed 0，missing 0 → OK
- `evaluation`：2005/2005 未变，changed 0，missing 0 → OK

## 9. 复现命令

```bash
cd fpga_accel/integer_selector_pipeline_compact_hls
make python_test
make test
make asan_run
make csim
make csynth
/home/zhaoxiangyu/conda-envs/robobrain25/bin/python -B scripts/summarize_csynth.py --report-dir ../../artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth --console-log ../../artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/console.log --out ../../artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/summary.json
/home/zhaoxiangyu/conda-envs/robobrain25/bin/python -B tools/build_report.py
```

## 10. 证据路径

- 源码：`fpga_accel/integer_selector_pipeline_compact_hls/{README.md,Makefile,include/,src/,tb/,tools/,tests/,scripts/}`
- fixtures：`artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/`
- csynth：`artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/`
- 日志：`artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/logs/`
- 机读报告：`reports/token_selector/2026-10-03-integer-selector-compact-interface-hls.json`
- harness result：`reports/token_selector/logs/2026-10-03-integer-selector-compact-interface-hls-harness-result.json`

## 11. 文件与 SHA256

| 文件 | sha256 |
|---|---|
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/console.log` | `4a3ab90aef5d667660db7f9e18a3396ec24a4694fc7ea156031eb425498dd83e` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/csynth.rpt` | `0c8181218d5b2eae1ed2d4fa968d8eaf0c8b69aa674e509678169b1fff4dc227` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/csynth.xml` | `b62baac70f0fd616bfb7eb0baebc9095da915fb02f71a3c35e214683995fe016` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/csynth_design_size.rpt` | `2bfb932a221af65da058cd83d314abc621f0ce103a3cf5f2428073347623552a` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/csynth_design_size.xml` | `6ce9d4e63727f7b654328cd9dce6ffb67d7da1455dcfe7684ac36ad6efdb0b92` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1013_9_csynth.rpt` | `5a40ab5fcb56ee5220e073e43763cdb28d0b5b95d210f352d1f4033957d7e760` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1013_9_csynth.xml` | `71f5f4fabca61191ec3f7a408462c29d7c0127c4df6ed9fc265eb97fa91e1831` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_103_1_csynth.rpt` | `f6f3e7e019e74d7fda246cbbc1b2a367bf9d310b7e75995c714916e0b408d05f` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_103_1_csynth.xml` | `e34ace749a4fc253dc2fd6b6626b96897aeca6c77d9fec11e51776f9f3d24490` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1040_12_VITIS_LOOP_1042_13_csynth.rpt` | `53d224df4d0b9832313f6b39f7b765db3eb69d778ef3e5d0fd5f5816d7d90600` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1040_12_VITIS_LOOP_1042_13_csynth.xml` | `4f2f55bb5006b5ab189043d59f9aa7c15654f038e76117c806ac5a7b48b0cd3c` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1074_14_csynth.rpt` | `ec84a64cda596173f405d319c6b46202e5f44da88de8e86b0139f2330022546d` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1074_14_csynth.xml` | `cc6a4e9acd6ac8f490a34e466a973eeb0d2f4a4a270a5d395961cc19eca2883a` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1079_15_csynth.rpt` | `bef5a2c4d984928d8c74fcffb87442a0b2fdf933bb328c3ad344ffae1d00c8e5` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1079_15_csynth.xml` | `4843176849debaa80ed4e368f5999fae9f9e19f1da6c455d51b70a0315633eba` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1089_17_csynth.rpt` | `f7e03be04562b67c17fc8220cf5e02f68219b42cfa77b341dc7e9e841b0f797f` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1089_17_csynth.xml` | `1836a13e4fba06eadce9e01c33cfe60c1ba77a0fdcf810ff127c6f465345ef78` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_109_2_csynth.rpt` | `6579fdf4609b619db2a048bd2154c4c86811096ce3a10ed553dcb3a0a4a36ebe` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_109_2_csynth.xml` | `ac4850cb2ad4e2e3dfc6c4ca023073257d20f35687eebee2a5877dfab59ecab5` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1107_18_csynth.rpt` | `fa772d7f8101c3c69abd8e1d41a9baa1cd25644fa95f1200cfbdf94e94394801` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1107_18_csynth.xml` | `92deda4cf0bfbde6fa8b4d9b1dad76ee03608b812590a759e465e947b283f3ea` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1116_20_csynth.rpt` | `604cffe04a1fde15c45f01cf4a07dfd8bc72c23fb4f0e8ca1b3392bcf379ab4e` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1116_20_csynth.xml` | `7b3100d320bf5c5f9e5dbb23399493c4baa25d53a8e0e447061914617ab2e32a` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1132_21_csynth.rpt` | `211a3e81ea902f54bf16af4ac1245801bb78d197232016eef2c181418261158b` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1132_21_csynth.xml` | `8c7e0287de66b8ab117b6c8a6024b849c6497bec0be795904392755ab03c4f86` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1137_22_csynth.rpt` | `6c18ddc399c4603439df258785a47b5d868a4ce9f2a128f06a2f0ff577c38f4d` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1137_22_csynth.xml` | `2ee7f97269a61f3d44261ea0766c4e75dc0bc8aaabb477ba7d185bc2e2650da4` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1166_24_csynth.rpt` | `df8c7e648e3f5e7a5fd2695b2c79cb0a279aa83681ed71f8805854d0a47314a4` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1166_24_csynth.xml` | `52fdc6b4dc19497faa9421feca490716174aba3a8f5e29ff50d1c39adfa3e96c` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1176_26_csynth.rpt` | `1378351e0d70aff70fb2d71562e17a6f6bc398e366d6e5efdce32ffcac325a40` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1176_26_csynth.xml` | `5b9cd2940dc5f6acc3bdd59d6e55b9ef731f23488f60539cdf6661fe8b55964d` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1196_27_csynth.rpt` | `6d02e23d4df8ac69210d716d838c708029ec4849f2df36fa856d449a71f3dfc6` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1196_27_csynth.xml` | `c07675d38d188f50543eafaac20aae931ae450f4aa14a6c5d5134daeb4b4369d` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1212_29_csynth.rpt` | `fd3f149f5839732815aff2f79f31220b2ff2822fa9fa4ff8fe621588394c2a92` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1212_29_csynth.xml` | `1c08e07c613adeddbf9e396c0335710876340278a257217d1a5c8c47042c64cf` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1227_31_csynth.rpt` | `f047d5fa38ad96c188aff2cde29a07cd1ecd7fb807864ba36bf7ed3918031bdf` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_1227_31_csynth.xml` | `1c375920241a5efb56e60869edff49effef21be3197da0c60d94001c1355060d` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_126_1_csynth.rpt` | `15176d2432b5db3ee48323ef7836c0c2a09603462c0b89efdc70985694ed5fea` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_126_1_csynth.xml` | `d5ef277bde2fce946a720a8739594672cdc8d4ab711565ddc786eb52d55463ec` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_127_12_csynth.rpt` | `8a308927ea58cc263a0be51b522e8dd61c2a48ec19c86b45dfbc21ab96eef897` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_127_12_csynth.xml` | `2f06dbbf4843df99a2a1350190516b8a297aabccc800f41ff1bd5a7054c53776` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_127_1_csynth.rpt` | `6c96f069fb1df1f7f634501b6adbca686177f2b5e2c09b020032ba4075a92392` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_127_1_csynth.xml` | `438b55be3be4aa0c0cc9cf5ca8adfced257814b948af1a523adef7171e72fea5` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_129_1_csynth.rpt` | `9e829b2cbbbd018e792e7364ee6beb54140f9ce959c18445339cae60a6c660eb` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_129_1_csynth.xml` | `1c309239ce9ade1ae3c09a6783dff594ff7b6d27b53cef6b87fe367dbc5f5934` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_151_1_csynth.rpt` | `6ff29ed2422095b731842df92c8cf124c3b3349a25084bcf967e569af04036f9` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_151_1_csynth.xml` | `e6c8786e3a7ab478038e2dcc2c53988df067db2abdb56b15961c49cac3aad87d` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_175_2_csynth.rpt` | `de1a5497eae5739f5c674c75df29594e8c3fc0a37d30be1bf3d9083a72ee09d4` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_175_2_csynth.xml` | `c1256d0d736e568e35a64014ac0f957a06f4f31f363233e487485409102be6d4` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_197_1_csynth.rpt` | `9bfdc816e56d1955b98001524aece4ea641afd740c55e6d3cbce1a135552b81d` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_197_1_csynth.xml` | `85dc0500262b266f9baf5c7f7ab3a7d20e8ad9b94db62ccf1641ddf3c79a9633` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_201_1_csynth.rpt` | `3f597d55472976cb296ad9d693e7a1e5deee0feb1f1a59c57697d9f11f760bc8` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_201_1_csynth.xml` | `678b5602782f7f9d1aded5d36c8f0d7ef18a2c955f58fce99f3c969d3353631d` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_66_11_csynth.rpt` | `38372cf3177595845406f00089285d233240f6cf089540ac561d859769e04d8f` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_66_11_csynth.xml` | `9df5a5ffc09d19c670c3558251755d281bce6a8e8ce556ee3489c38b2ae5dfa1` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_66_1_csynth.rpt` | `fb7b034dbdaccbd3401510d72845975de09afb353d595f9b52513d77896f8efd` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_66_1_csynth.xml` | `5eddde4096e8405ea07e7693d9d815f624585ea6f014cc96180c232cd545e759` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_92_1_csynth.rpt` | `a36eaccf786e10a0e3ffcc5e4d7c007efdd91c91e7ce02e92f8752809eb534cf` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_92_1_csynth.xml` | `88ea5c84ca2472e2167b2ae70104ab740bc3bf87a0f07eefc91e80e997955592` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_944_5_csynth.rpt` | `84a58043cb2a24411dbedbd5616956bbc73cfa6dcc1721ff856c8107ef8fe10a` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_944_5_csynth.xml` | `dc8e8cbeeb4c7de109ce10a692b4ea3eb2085aaf8d06ea43e0c7ae639f0db0ca` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_969_7_csynth.rpt` | `53725f5014758ca55df9a8a923e3436d75ef890b365404a2ae69e1ca6aef236e` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_969_7_csynth.xml` | `9cebf8e7c05a98400e781306ef45c876307dcea2c039b761b4e13563c08fd265` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_985_8_csynth.rpt` | `d93f30a380acaadbd2d75ce9fd5e470e7aa8652b76f38c04de5ca00052f727c1` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_Pipeline_VITIS_LOOP_985_8_csynth.xml` | `1afb21ba2efc7bc671dfe59366ff95cfc64121e3ce2418131b2d8fc56ea1b992` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_csynth.rpt` | `faf07976da097ece28c82aa4733b24c8a8ec042cee8e24c30e665a20a3bf2eb3` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/integer_selector_pipeline_compact_csynth.xml` | `a0a1b756c94f782589908a2ab549a72fc2bda9c5884dee4067074b1ba3751edd` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/interface_evidence.json` | `9f4fe75c7a3890d7343fb243085d9f19d6738187cb683eb2e8861d1a2952d9b8` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/ispc_put_u8_segment_Pipeline_VITIS_LOOP_110_1_csynth.rpt` | `ccdc5f9cea48ef55a60ba554198780faf6cd008634585f032961b965dece6952` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/ispc_put_u8_segment_Pipeline_VITIS_LOOP_110_1_csynth.xml` | `a611bbd963dc373254819d0d984cac1554d93988454e7b2fcad620e2a1310c7b` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/ispc_put_u8_segment_Pipeline_VITIS_LOOP_111_1_csynth.rpt` | `bd27c5e919c73d050618c06fc44f8ed829e94eba62f0c823bbb33093e01388f2` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/ispc_put_u8_segment_Pipeline_VITIS_LOOP_111_1_csynth.xml` | `994768a1b73ebbdaf36cb08768065436b5ee6fd894fd69a1865515bdcac4726d` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/ispc_put_u8_segment_csynth.rpt` | `eed4a15b1a79d5b875d4f0c896ff795b19843228ec22c569f1d933b1771851c7` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/ispc_put_u8_segment_csynth.xml` | `921f09d3cef766dcb932a4ac50daeed9d6772c22a8f37aa60a133e9da4c6354c` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/layout_evidence.json` | `1530dec74355a378e860a1919705e5134ebf341b3d4aae5d2a6e3eb7299f66ed` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_136_1_csynth.rpt` | `56adfc1cbf02c12a8654953b88e365cceee78a13d40719b28131165b687eece7` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_136_1_csynth.xml` | `2522bdca0817ffc5e09fbe0f67df7b1e7e08bf27f6408c8a480ef2006c4ceb99` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_148_3_csynth.rpt` | `dd96bd58cd57982e10fb429ed407f07cebbb22f288f5aa9f05bba89a876c63e7` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_148_3_csynth.xml` | `700abc4f41c04d7bbc22a33581ea97a7f8d102569e25026819d0b270b190ba48` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_168_6_csynth.rpt` | `d5223d5c75055454a8f03e8d471973f2340ba66409ede3e891c81482a2f6fd29` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_168_6_csynth.xml` | `2dbbfbeccdc8846dd8cc1b68bffafe1c776df9dff243a824afc4faff76ba7a33` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_188_9_csynth.rpt` | `420e2f8ea3145f82250c687177c65436a3ee91be3ccb9d66d089352dc68240c3` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_188_9_csynth.xml` | `8cba59cdd015b005abb45d5ff4c8332f4e1a47bbdcba4af8f436b14f37355ab8` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_201_11_csynth.rpt` | `2f1927992895202ca6325736aec1b350e813a58da47c8b787a030c8cc3cf9ae4` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_201_11_csynth.xml` | `573c1474016c3a8e5b6b948d0568f1769dd9645c2f6cbabb5ce9868d3503cf3e` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_59_13_csynth.rpt` | `4995249ba34c344377841e2fef2e0722e8e89f1544ac6df0ebc224015e801a60` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_59_13_csynth.xml` | `840de9958b6639b27b91ae8d9a5e296efb3e31e8d5ce5481f8710be3fb61628a` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_59_1_csynth.rpt` | `c22f3c78b6a0bf16b06f019a655d25c0398cf45f254c223f0a37bf040906ef15` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_59_1_csynth.xml` | `c587b48ad17c2f8e69ff30d2f087542f0d32bf1d6518f789ff77c71c1a80e766` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_14_csynth.rpt` | `ca3dc7bd625d98caf29ae912f2937f7252c53ca278fdcc5e3b84e42d130f3638` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_14_csynth.xml` | `8c7951cdd26bef2d48ae2d8ec9c3bd9325f54417d41c6f08a927cc76f528711e` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_15_csynth.rpt` | `4b313c5f75a584d1626f5f43aca2ac99f053b1260615ae6e8af66a532d933535` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_15_csynth.xml` | `b9fd778658fe7ba1055bd0cdf053cf06b43cbe5b07a001cb221c2b4181d85633` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_16_csynth.rpt` | `305928b5664ee5d233f954ae7fcd95d20a29f1f1ba2177b5283c2c58d208233f` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_16_csynth.xml` | `2a61d3962795e16821392de838bf36e71d2b0685c3e794f56c49ccd74a2b2725` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_1_csynth.rpt` | `27d5093725bbe216dede1474f721f6f8713d166669a07e104869a0db2def3f57` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_1_csynth.xml` | `63825d30e43e7f6bbd4651570f9043b5e0fbab586243b8fed0868524cd3b3b0a` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_s_csynth.rpt` | `4e99338c519fc8faef19bb547fd5de53176492ef0180fa86856ce8278b008236` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/p_anonymous_namespace_is_scorer_core_false_s_csynth.xml` | `db9f3d4e09a9dc864f9f69822776b1b96359703e293ed185e0d358584e08789e` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/summary.json` | `ad6530499df67c4cabadaaaa2e2af03dc820da02fd0329937f6a2e8121c15426` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_103_1_csynth.rpt` | `568b871d80bc4e9d5fa25dd86a497e65f3cc3d444183e09b638831a5b83658ca` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_103_1_csynth.xml` | `f3f748349f1489fcc9a3cf5e149ff9920f0dacbc1ffa17df32678b1b26060c2f` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_109_2_csynth.rpt` | `c75f47ae1b4569659dc334fd384eaf303678dc31bae387483f04187fe192ceb5` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_109_2_csynth.xml` | `dc6fa3e37b5a4431720cb3ba1c8f76a6e736c374f2edf6d4e51a2dafe9675a93` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_272_1_csynth.rpt` | `351a89035b7d9784f7aa23b6af87f32e2316c38fb05f422215e9d31015aa8a78` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_272_1_csynth.xml` | `e98c0a9a9de6c70d5e56446e9707df88329287dd2baa7f3ab2820dd1d1047655` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_276_2_VITIS_LOOP_281_3_csynth.rpt` | `3beda204338b2f4a994071e63cac4d9c2cc1f443a75df2d569e2111252e432dc` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_276_2_VITIS_LOOP_281_3_csynth.xml` | `7c8304a9046818813a00b0c1445051ad4e756916a67f2bf7f7d481b3aed42636` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_315_4_csynth.rpt` | `5966c3b57179ed7d3766d7c5abbe459636ecb9eea33769ec4d28a7c0512c0f78` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_315_4_csynth.xml` | `0e7395818cf7592f7ef7abeb9e47c891157b9c285e17cb2a4b62e79f9c8a03e2` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_320_5_csynth.rpt` | `3b89f3fe9bc9244d773ea9b6269d8d4faa7a303db32b5e37b34459d6e8eb8c7d` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/ts_pool_context_packed_Pipeline_VITIS_LOOP_320_5_csynth.xml` | `23b680c3600c8847193770ab0d3338cb2b5b6383a3a5f9a2d140566fb6ad304e` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/ts_pool_context_packed_csynth.rpt` | `0f3cbb404797e3c21eaf217dd5d34f772c7ff5943739a2ff1bd3f1fe203cf89a` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/ts_pool_context_packed_csynth.xml` | `9b049bd209117e8a78d5056600ab49e82546776dee13e1ff3abefa2c26fbf814` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/FORMAT.md` | `e5b19747f6c17320b40d12f8a8890863501e65882d4f5b445b0bdd6a0200522f` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/all_equal_g128_K48_cap16.exp.bin` | `d61f5393f7370ac8a1bf73ab57be1a71b6e67084de34c8d619fac2c28cf0802a` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/all_equal_g128_K64_cap16.exp.bin` | `0aec8e0fe905af83be62d5819729442139635a341eb861a3fa7caabe5cfab206` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/all_equal_g128_K96_cap16.exp.bin` | `ab555efb0e93079a510b670d5af07a4b3cc17931bc5bbb63433fec6ef36d37a8` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/arena_all_equal_g128.bin` | `74ca06c7cefb06a70fa426f3e61da460e8a9130b72794f0f982642bae9ab56a6` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/arena_empty_2x25.bin` | `0b6b6aa43f0baf7fd8bd73dbf219f8daad9c24182dc87d09eea8cdb09d60b79c` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/arena_identity_g16.bin` | `035f3d9ee4c171aff671e3c5f8ee01c97f78546bb09203714c8579fca5ea4a33` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/arena_identity_g9.bin` | `9b46d65607ffc65df905a72937c63443d6e0e1e9210fb2ac25cdc14c101bb61d` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/arena_real_grounding_correct.bin` | `9357530693dccd327a85f49e096f96cfe69afcc169b51ae770a4ef8df00a827a` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/arena_real_grounding_wrong.bin` | `60d568dd80739f743ccd6b9fbc8e418667b4162c4768dfb8b1c615f10d36be4a` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/arena_real_pointing_correct.bin` | `dd07e9bf431103281bd7923bdb7600deaffd23f3e07dbba4d7658b7c59d60566` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/arena_real_pointing_wrong.bin` | `9c4ea91c94b02a541931abe0d43b6fe239a0b0d0fef0be42f58c43a51e4eab82` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/arena_synth_g400.bin` | `f19d90369155e8bd2dd9f8bb39470614205a3bad5c00d2282a99491cf5c263f5` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/arena_synth_g80.bin` | `af69e32a890c439472127a39a925c06db1f43427eeb39153ce1d985c09a5c87a` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/arena_zero_g128.bin` | `8d0662ae40307e71b23c66d10603c7bf2c0f641ccf6ed6f86cb1d234d1b9914c` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/empty_2x25_K48_cap16.exp.bin` | `d4a7c32ddd8aed4129e0292a94a12aab78c780769068c329188c0caef3436073` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/identity_g16_K48_cap16.exp.bin` | `ad3bce8e97bd70fe4462a9f08e9db9373418ac8e9990a252c72ff83f47c20d41` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/identity_g9_K96_cap16.exp.bin` | `ad97ff256add534a9354f4bc5696fb6c1d1621e05a35e2546ae54b2941629120` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/manifest.json` | `51c5828a78e7c68f265fa85a3e7ea10a3e92d2b150202574f84cdb7bc8c10f8e` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/manifest.txt` | `492cea0546a3a0487f0cc378da181d66efbd8168f286c1d6ec3316da9b868f2e` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/real_grounding_correct_K48_cap16.exp.bin` | `8ff0db0658c9fd720e293ecfcdcda5f4853f8839bd43fd0934230ea3cfdac944` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/real_grounding_correct_K64_cap16.exp.bin` | `f7b226c83e0c2b02b906c9b49f1fb8cdd0b279582ee53edbeed22da5495c0235` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/real_grounding_correct_K96_cap16.exp.bin` | `ae96ffd5482eebaaaf0e655f95be10d176b0e23d404aa824ab219c3a00e69d37` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/real_grounding_wrong_K48_cap16.exp.bin` | `67e80590ffe75e3560811a90f780f4b603daa0c5ba69517702767a2f5608ff70` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/real_pointing_correct_K48_cap16.exp.bin` | `afafdbb318a52d9289b15ac652b5386f5edf5301b3db9d5902b34bf7817dafda` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/real_pointing_correct_K64_cap16.exp.bin` | `f5da85ae2a19353e3af6c3eae8065717faa608fcb7efa10dc886d02cb04fe02d` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/real_pointing_correct_K96_cap16.exp.bin` | `86e0f67ca9f2e69d0232da4c0ec5492fc030bcfd5798a86a3b438ec0d03508c7` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/real_pointing_wrong_K48_cap16.exp.bin` | `abcc87df504e8eb4136c987edde3e81cd2ab8098669076d849987de94d1ea28b` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/synth_g400_K48_cap16.exp.bin` | `72c2d39dca5aef8d4df5744e4886a327d723d5e9a1498f0f2ee1906bca8e6420` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/synth_g400_K64_cap16.exp.bin` | `a69705319f954a93c227b4967cc8ac00e69cb1ac88bcbca63c10cac4dbd7a620` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/synth_g400_K96_cap16.exp.bin` | `89930c550fab88302fdf76b837024642ed5c9043d0c103dbae53135fe9eb72dc` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/synth_g80_K48_cap16.exp.bin` | `780d888ae9455ec6d6d21ee3a2825b0c22ffda21d1ddae1b4e8a7bab6f195377` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/synth_g80_K64_cap16.exp.bin` | `04c234264811bb8d8b02d13067ceb85baf02f172a0f1defda375a81cd541d30f` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/synth_g80_K96_cap16.exp.bin` | `9b5784b35ff165298376f0e032e19c7d43eceb5e09a1c3e3688698ef3a927d88` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/wv_q.bin` | `126bd8421dd128a318115c301bf9d1c7d336912150d19d2ec35203ed55d89d3b` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/zero_g128_K48_cap16.exp.bin` | `f432452dcbeebcbe4b3fb5abf3c33ef036f94c4a43364ca9243e3c615bbfb63d` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/zero_g128_K64_cap16.exp.bin` | `8ab22c8da70f749d95e481cb58c6eca6c8883d92ded39f9e9e2927feea84c275` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/zero_g128_K96_cap16.exp.bin` | `d52bb4c18ee77b31b960db692243433f96f032727433187b5d9589df2de73a22` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/logs/01_python_fixtures.log` | `e3ae1a8c961a1f935f445f0bd135f8b4132300f9b92487dead49c40678238595` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/logs/02_gpp_test.log` | `5220b7a7ac471a97ce3eeb166c7db504cb2de80343829f620adc41a8a0aac0f8` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/logs/03_asan_ubsan.log` | `4ff1507cc268079792b06e14901819a965ea3bf7ab9981896b5e9280c92b1d7a` |
| `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/logs/04_hls_csim.log` | `9320ca1e49cd959fc667a7ab0f027a92ae26ce9f548db1ffee2d654acec3fdc3` |
| `fpga_accel/integer_selector_pipeline_compact_hls/Makefile` | `e00b2700aac0db6a7531331c4108a66f77ae6ec5c7578c7a7c53e9d15601b657` |
| `fpga_accel/integer_selector_pipeline_compact_hls/README.md` | `96e69a0d21740d5ea2b74974274c179189e0774945d2a2289463af6f97e264bd` |
| `fpga_accel/integer_selector_pipeline_compact_hls/include/integer_selector_pipeline_compact.h` | `628369a28c6e00d97761f41ef89d76929a16d7e1ac3cb15045d422b81da7e91a` |
| `fpga_accel/integer_selector_pipeline_compact_hls/register_map_draft.md` | `7127345ac5c523a9ae5eb1efc60df567d7cecfaf281a689802b2b96af5086321` |
| `fpga_accel/integer_selector_pipeline_compact_hls/scripts/run_csim.tcl` | `4617693c602a4353fc5403f2bd03012fcc338be5b8537cdeafa8aeee75e000b9` |
| `fpga_accel/integer_selector_pipeline_compact_hls/scripts/run_csynth.tcl` | `a986d9819bb78930aa7dd8b217c495529bdc1361f228f957b8d6c4aaeb880757` |
| `fpga_accel/integer_selector_pipeline_compact_hls/scripts/run_stage.sh` | `42579f3037191a5fa247c3a06c945e36ded3c54dae48485715713813b9968a88` |
| `fpga_accel/integer_selector_pipeline_compact_hls/scripts/summarize_csynth.py` | `693455fbb9c052ee011c672f081d10167cdcd91826b825b972f2331d854db250` |
| `fpga_accel/integer_selector_pipeline_compact_hls/src/integer_selector_pipeline_compact.cpp` | `e64e2306fe993ec9f2f38e78eaf6944b60ff57a6450b338215c4f9d7c7b8f883` |
| `fpga_accel/integer_selector_pipeline_compact_hls/tb/tb_integer_selector_pipeline_compact.cpp` | `0abbf5bf3ec5b5c82cbffe3c451b64547b1165a2db5699ea9c0809964f5bbc8a` |
| `fpga_accel/integer_selector_pipeline_compact_hls/tests/test_integer_selector_pipeline_compact_hls.py` | `7db9b80a47a66c4592ed1eee3262abbd3bb9d6d9ce1abfd7cce20fb67984bc8b` |
| `fpga_accel/integer_selector_pipeline_compact_hls/tools/build_report.py` | `43d3bae8c39ae2b75932a2340ac1264d6963e39295761dc39b03a5e810ddf81f` |
| `fpga_accel/integer_selector_pipeline_compact_hls/tools/compact_layout_dump.cpp` | `0c7bb335bf7530cc1dc574e19f299e990c84134ab31500b5bdc36345f4537547` |
| `fpga_accel/integer_selector_pipeline_compact_hls/tools/fixture_format.md` | `e5b19747f6c17320b40d12f8a8890863501e65882d4f5b445b0bdd6a0200522f` |
| `fpga_accel/integer_selector_pipeline_compact_hls/tools/generate_fixtures.py` | `6822d26eb5a109f988b303a06acd211c5a27dc6d6187552dc9b337e8fd0c30fc` |
| `fpga_accel/integer_selector_pipeline_compact_hls/tools/transfer_audit.py` | `5bc312386148b38721afd3224797047a817955734b20fe8e5ed1a3409b9bc610` |

