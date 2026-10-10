# 独立位精确整数 Selector 打分核 `integer_selector_scorer`：实现、验证与受控 HLS csynth

- 日期：2026-10-03
- 决策：**`INTEGER_SELECTOR_SCORER_HLS_PASS`**（全部 gate 通过）
- 范围：**只 `csynth_design`**（Vitis HLS 2023.2、part `xczu9eg-ffvb1156-2-e`、
  `create_clock -period 5.0`、tool 默认 uncertainty **1.35 ns**，未放宽）；无 C/RTL
  cosim、无 IP 导出、无 implementation/bitstream、无板卡、无 GPU、无联网、无训练。
- 大工程只在 `/tmp`；仓库只保存源码、小 fixture、小报告与日志。
- 只读复用既有资产（seed-20261003 Wv bundle、grounding/pointing fixture），
  **未改**任何既有 Python 审计、selector/HLS kernel、模型/checkpoint/cache/manifest/
  config/report 文件。

## 0. 结论摘要

新增独立顶层 `integer_selector_scorer`，把冻结整数链
（`INTEGER_SCORER_Q16`，golden `evaluation/selector_integer_scorer_audit.py::integer_chain`）
实现为纯整数 HLS 核：INT8×INT8→INT32 投影、INT64 dot/norm2、精确 floor 整数
平方根、精确有符号有理 round-half-away、整数 min-max 到 Q0.15、λ=0.5 半进位合并
与 clamp。DUT 内**无 float/double、无负值左移 UB**。

四道数值门全部逐字节通过（Python 21/21、g++ 103/103、ASan+UBSan 103/103、HLS
C-sim 0 error 103/103），随后 `csynth_design` 一次通过：**estimated period
3.650 ns ≤ effective budget 3.650 ns（slack 0.0）**，资源
**BRAM 10.42% / DSP 5.24% / FF 2.32% / LUT 17.99% / URAM 0%**（每类 < 50%），
由工具报告推导的 **G=400 时延 803,539 cycles = 4.018 ms @200 MHz**
（含 Wv 载入 65,539 cycles）**≤ 4,000,000 cycles / 20 ms**。判
**`INTEGER_SELECTOR_SCORER_HLS_PASS`**。

## 1. 顶层合同

```
int integer_selector_scorer(
    const is_word_t* group_mean_q,   // [G*64]  128-bit words, G in 1..4096
    const is_word_t* wv_q,           // [8192]  128-bit words (128x1024 INT8)
    const is_word_t* query_q,        // [8]     128-bit words (128 INT8)
    uint16_t*        score_q,        // [G]     uint16 Q0.15 in [0,32767]
    int              group_count);   // 1..4096
```

| 项 | 定义 |
|---|---|
| `group_mean_q` | 行主序有符号 INT8 `[G,1024]`，16 lane/128-bit word，lane 小端（lane `L` 占 bits `[8L+7:8L]`） |
| `wv_q` | 行主序有符号 INT8 `[128,1024]`，同打包；**每次调用缓存一次**到片上（128 bank × 1 KiB）后所有 group 复用 |
| `query_q` | 有符号 INT8 `[128]`，8 个 128-bit word |
| `score_q` | `uint16 [G]` Q0.15，范围 `[0,32767]` |
| `group_count` | 标量，合法 `1..4096`；固定维度 1024/128 |
| 状态码 | `IS_STATUS_OK=0`、`IS_STATUS_ERR_NULL=1`、`IS_STATUS_ERR_GROUP_COUNT=2` |
| AXI | 4 个独立 `m_axi` bundle（`gmem_gm`/`gmem_wv`/`gmem_query`/`gmem_score`）+ `s_axilite control`（`group_count`、`return`） |

- g++ 下 `is_word_t = unsigned __int128`（小端 host 位布局相同），综合下
  `is_word_t = ap_uint<128>`；两种模式共用同一份源码与 lane 访问器。
- 空指针（host C++；综合下 AXI base address=0）返回 `ERR_NULL`；非法
  `group_count`（0/4097/负）返回 `ERR_GROUP_COUNT` 且不写输出。
- 只做**打分**，不做 anchor/选择/合并；选择仍在 `fpga_accel/token_selector_hls`。

## 2. 位精确整数链（与审计逐位一致）

1. `p[G,128] = group_mean_q @ Wv_q^T`，精确 INT32 累加。
2. `dot = Σ_d p[d]*query_q[d]`（INT64），`norm2 = Σ_d p[d]^2`（非负 INT64）。
3. `semantic_q16 = round_half_away((dot*2^16) / floor_sqrt(norm2))`；`norm2==0` → 0。
   `2^16` 用**有符号乘法**（`dot * (int64_t)((int64_t)1 << 16)`），不是负值左移。
4. `l2_int = floor_sqrt(Σ_1024 group_mean_q^2)`。
5. `semantic_q16`、`l2_int` 各自独立精确有理 min-max 到 `[0,32767]`（全等 → 0）。
6. `score_q = clamp(round_half_away((semantic_norm_q15 + l2_norm_q15)/2), 0, 32767)`。

`floor_sqrt` 为逐位整数算法（32 次固定迭代，不触碰 float）；`round_half_away`
为 64 次无符号移位-减法长除法（`(2|num|+den)/(2den)`）后还原符号。DUT 内所有
`float`/`double` 出现次数 **0**（Python 静态测试断言），`sqrt(`/`pow(` 内建调用
**0**。

冻结分析界（`static_assert` 编译期检查 + `quantization_contract.md`）：
`|p| ≤ 1024*128*127 = 16,646,144 < 2^31-1`；
`|dot| ≤ 128*16,646,144*128 = 2.73e11 < 2^63-1`；
`norm2 ≤ 128*16,646,144^2 = 3.55e16 < 2^63-1`；
`|dot*2^16| ≤ 1.79e16 < 2^63-1`；
`|semantic_q16| ≤ 2*128*12*2^16 = 201,326,592 < 2^31-1`；
`l2_int ≤ floor(sqrt(1024*128^2)) = 4096`；`score_q ∈ [0,32767]`。

## 3. 片上结构与两趟流程

| 数组 | 规模 | 说明 |
|---|---|---|
| `wv_local[128][1024]` int8 | 128 KiB，`dim=1 complete`（128 bank，各 1 BRAM18） | 每次调用一次缓存，group 间复用 |
| `group_row[1024]` int8 | 1 KiB，`cyclic factor=16` | 当前 group 的 packed 行，16 lane/周期载入 |
| `semantic_arr[4096]` int32 | 16 KiB | 第 1 趟写、第 2 趟 min-max/输出 |
| `l2_arr[4096]` uint16 | 8 KiB | 同上 |

- 投影为 **128 路并行 INT8×INT8→INT32 MAC**（每个输出行一个 bank，`#pragma HLS
  UNROLL` + `PIPELINE II=1`），约 **1029 cycles/group**（含流水开销）。
- 随后 INT64 dot/norm2 归约（约 133 cycles）、`floor_sqrt`（2×34 cycles）、
  `round_half_away` 除法（约 130 cycles）构成 pass 1。
- pass 2 独立 min-max 并写 `score_q`（约 403 cycles/group）。
- Wv 载入 8192 个 128-bit word，achieved II=8 → **65,539 cycles**（一次性）。

## 4. fixtures 与四道验证

### 4.1 fixtures（`artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/fixtures/`）

Python 生成器 `tools/generate_fixtures.py` **直接调用冻结 golden**
`integer_chain`（以及 `floor_sqrt_int64`/`round_half_away_div`/`minmax_normalize_q15`），
不复制数学；输出格式见 `tools/fixture_format.md`（`ISF1`/`ISP1` 头，小端）。

| fixture | 内容 |
|---|---|
| `wv_q.bin` | 已接受 seed-20261003 Wv bundle 的逐字节副本（源 SHA 与副本 SHA 均 pin） |
| `full_grounding.bin` / `full_pointing.bin` | 真实 fixture（correct+wrong prompt），G=260，13×20，含 `score_q/semantic_q16/l2_int/semantic_norm_q15/l2_norm_q15` golden |
| `synthetic_zero.bin` | 全零 group_mean → 全 0 |
| `synthetic_all_equal.bin` | 全等非零 group_mean → 全 0 |
| `synthetic_g1.bin` | G=1 → 全等 → 0 |
| `synthetic_mixed_sign.bin` | 混合符号（dot 与 semantic 均含正负） |
| `stress_g400.bin` | G=400 **流式** xorshift64* 生成（不嵌 400×1024 行），golden `score_q` + FNV |
| `primitives.bin` | 45 个 floor-sqrt、52 个 round-half-away（含正/负精确 .5 tie）、12 组 min-max 向量 |

真实 fixture 的 golden 与既有 NPZ 内 `score_q_correct/wrong` 等逐字节一致
（两次独立生成也逐字节一致）。

### 4.2 四道门

| 门 | 结果 | 日志 |
|---|---|---|
| Python fixture/合同测试 | **Ran 21 tests, OK**（0 fail） | `artifacts/.../logs/01_python_fixtures.log` |
| g++ 普通 | **103 checks, 0 fail；IS_RESULT PASS** | `artifacts/.../logs/02_gpp_test.log` |
| ASan + UBSan | **103 checks, 0 fail；IS_RESULT PASS** | `artifacts/.../logs/03_asan_ubsan.log` |
| Vitis HLS C-sim | **CSim done with 0 errors；103 checks, 0 fail；IS_RESULT PASS** | `artifacts/.../logs/04_hls_csim.log` |

测试台逐字节比较每个 `score_q`（生产顶层与 debug 孪生各一次）、`semantic_q16`、
`l2_int`、`semantic_norm_q15`、`l2_norm_q15`，独立 FNV-1a 校验 golden，
并直接对拍 `floor_sqrt`/`round_half_away`（含正负 .5 tie）/`minmax` 原语向量，
以及 G=0/4097/-1 与空指针状态码。**全部在 csynth 之前通过。**

## 5. 受控 csynth 结果

- project `/tmp/robobrain_integer_selector_scorer_hls_csynth`；只 `csynth_design`；
  0 error；54 warning（均为 loop constraint/信息类）；**无** `HLS 200-871` /
  `HLS 200-1016` 时序告警。
- 报告：`artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/`
  （`csynth.xml`、`csynth.rpt`、`integer_selector_scorer_csynth.rpt`、
  `csynth_design_size.rpt`、各 pipeline rpt、`summary.json`、`console.log`）。

### 5.1 时序与资源

| 项 | 值 |
|---|---|
| target / uncertainty | 5.00 ns / 1.35 ns（默认，未改） |
| effective budget | **3.650 ns** |
| estimated period | **3.650 ns**（Fmax 273.97 MHz） |
| slack | **0.000 ns**（timing met） |
| BRAM_18K | 190 / 1824 = **10.42%** |
| DSP | 132 / 2520 = **5.24%** |
| FF | 12,725 / 548,160 = **2.32%** |
| LUT | 49,298 / 274,080 = **17.99%** |
| URAM | 0 / 0 = **0.00%** |

每类资源均 **< 50%**。

### 5.2 时延（综合估计，非实测）

- 工具对硬边界 **G=4096** 报告：latency best 1、average `undef`、**worst
  7,622,678 cycles = 38.113 ms @200 MHz**（合法 G 的上界；非法输入的 early
  return 给出 best=1）。
- **Wv 载入 cycles = 65,539**（8192 words，achieved II=8，一次性）。
- pass 1 每 group **1442 cycles**（`VITIS_LOOP_164_5`，投影+归约+sqrt+除法，
  含数据相关分支最坏值；迭代区间 1238~1442）。
- pass 2 每 group **403 cycles**（`VITIS_LOOP_250_12`，min-max+输出）。
- 每 group 合计 **1845 cycles**。

**由工具报告推导**（`derived = 65,539 + G × 1845`，200 MHz；综合估计，非测量）：

| G | cycles | ms @200 MHz |
|---:|---:|---:|
| 80 | 213,139 | 1.0657 |
| 260 | 545,239 | 2.7262 |
| **400** | **803,539** | **4.0177** |

G=400 的 803,539 cycles **≤ 4,000,000 cycles** 且 4.018 ms **≤ 20 ms**，
满足门槛。

## 6. 与既有 token selector `score_q` 的对接

- 本核输出 `uint16 score_q[G]`，Q0.15、范围 `[0,32767]`；既有
  `fpga_accel/token_selector_hls` 的输入端口就是 `const uint16_t* score_q`
  （同样是 Q0.15、`[0,32767]`、G 个元素）。
- 因此 **`score_q` 数组可被 token selector 直接消费**：本核填好 `score_q`
  缓冲后，selector 以其现有 `score_q` 端口读取即可；元素类型、语义、范围完全
  一致。selector 仍在自身内部做 anchor/选择/合并，本核不改变其行为。
- 需要 host/driver 侧把两核串起来的**数据搬运与顺序**（先打分、后选择），
  本轮未实现、未 cosim、未上板。

## 7. 决策与 gate 表

| gate | 结果 |
|---|---|
| Python/g++/sanitizer/C-sim fixture 全部逐字节一致 | ✅ |
| csynth 无 error | ✅（0） |
| timing 满足 effective budget | ✅（3.650 ≤ 3.650 ns，slack 0.0） |
| 每类资源 < 50% ZCU102 | ✅（BRAM 10.42/DSP 5.24/FF 2.32/LUT 17.99/URAM 0） |
| 有限可辩护的 G=400 时延 ≤ 20 ms（≤ 4,000,000 cycles，含 Wv 载入） | ✅（803,539 cycles / 4.018 ms） |

→ **`INTEGER_SELECTOR_SCORER_HLS_PASS`**。无 bottleneck；不需要 redesign。

## 8. 限制

- 只 `csynth_design`：不是 C/RTL cosim、不是 place/route、不是 bitstream、
  不是板级实测；时延为**综合估计 + 由报告推导**，不是测量值。
- 只做打分：不写选择/anchor/输出布局，不声称端到端 selector 完成。
- 未加载 4B/模型/checkpoint，未训练，未跑 CV-Bench，不声称任务精度或加速。
- 输入量化后整数链与冻结 golden 逐位一致；量化前的 A8/per-tensor INT8 量化
  本身沿用既有已接受结果，本轮未重算精度。
- 合法 G 的上界（4096）worst-case 为 38.113 ms > 20 ms；门槛只规定 G=400
  （803,539 cycles / 4.018 ms）。若需 G>~2100 保持 20 ms 以内，需要进一步提高
  投影并行度或降低 group 级开销，本轮未做。

## 9. 复现命令

```bash
cd fpga_accel/integer_selector_scorer_hls
make python_test        # fixtures + 21 个 Python 合同测试
make test               # g++ 103 checks
make asan_run           # ASan+UBSan 103 checks
make csim               # Vitis HLS C-sim（只 csim_design）
make csynth             # Vitis HLS csynth（只 csynth_design，报告归档到 artifacts）
/home/zhaoxiangyu/conda-envs/robobrain25/bin/python scripts/summarize_csynth.py \
  --report-dir ../../artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth \
  --console-log ../../artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/console.log \
  --out ../../artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/summary.json
```

## 10. 证据路径

- 源码：`fpga_accel/integer_selector_scorer_hls/{README.md,quantization_contract.md,register_map_draft.md,Makefile,include/,src/,tb/,tools/,tests/,scripts/}`
- fixtures：`artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/fixtures/`
- csynth：`artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/`
- 日志：`artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/logs/{01_python_fixtures,02_gpp_test,03_asan_ubsan,04_hls_csim}.log`
- 机读摘要：`reports/token_selector/2026-10-03-integer-selector-scorer-hls.json`
- authority prompt：`reports/token_selector/prompts/2026-10-03-integer-selector-scorer-hls-harness-prompt.md`（SHA256 `9cb35db572b7c5efb3c2dd39dfcfcba7901a8d39614e3d2721ddcd2bf712f7e5`，运行前后未改）

**精确下一步（未执行）**：只有在独立授权的下一阶段，才考虑与既有 token
selector 的串接（driver/数据搬运）、C/RTL cosim 与 IP 导出；在此之前不
implementation、不 bitstream、不上板，也不改变本轮的位精确整数链与资源/时序结论。

## 11. 文件与 SHA256

下列为源码/合同/fixture/csynth/日志文件（报告本身与 JSON 的 SHA 见 JSON 摘要）。

| 文件 | bytes | sha256 |
|---|---:|---|
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/console.log` | 68372 | `60ac9611f19a7ea00df306bf91737334a537f5ca88d0f106f7a43d6da56d9902` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/csynth.rpt` | 97619 | `7610bb4555e5bc9fd43be29f2735fec82b78e44a284964e6f35e0a9c95c559a6` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/csynth.xml` | 340715 | `2827a3e5aa1d681a193abe0eac8f43963865d8832e82533e2ccee8691762d8dd` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/csynth_design_size.rpt` | 7318 | `0a3f18b457c81d6557984e18e7038f57191ac3416d5adeefcb2461fd841b9f52` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/csynth_design_size.xml` | 5199 | `25f9b13a2efbe1dbfcbc4f787242451be2bc54376793698e9af6d53248b3d575` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/integer_selector_scorer_csynth.rpt` | 31010 | `b442ed1fee62cf087e61bc3b377c65eedebb6467a0d964ea1faaf82908c2dc52` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/integer_selector_scorer_csynth.xml` | 56107 | `04fc8f2b7f57b9ff7923c75a436fe30973188d3872e2f9f185de75439ad1ccfb` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_112_1_csynth.rpt` | 274545 | `6034048b3d631761ae4ac51ea3a50b45101dd6bcfa1909f17e00eb83fc1598a1` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_112_1_csynth.xml` | 255261 | `763dba0d1e50eb1552d889664f5a8abbbbd0266a35bcb9653dd0cc25a3dde20f` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_124_3_csynth.rpt` | 24392 | `d61061eb96fe50c3af4e0f6b94e30adbabc625ee19c5b71f4a666e9921644cb4` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_124_3_csynth.xml` | 30311 | `1f4a5ebf231ce5270e4887abf076c345b7f67eab1da9d99d39888f654b337a06` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_136_1_csynth.rpt` | 274545 | `056a498b0e04f15c70191ee17265d98f73659a22d19d8fdb278276233af1df61` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_136_1_csynth.xml` | 255261 | `91c8c36a15b8703492c82ce605139df000dc2140814013879bf2913927b630db` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_144_6_csynth.rpt` | 23932 | `728296ebb327a0ae0b497f29c43a436139af2b971951fb20a36c2db4442b98ab` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_144_6_csynth.xml` | 29844 | `0d006f025d5f01fbd6a5e59ecbae227d1c8486c020e802cc70abc764f97b166d` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_148_3_csynth.rpt` | 24392 | `354bdef5d87557b8420e2dcd9491137b5c12d778d2c63482e6bff3be525dca38` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_148_3_csynth.xml` | 30311 | `819db18e581004e804b9c4c0e619543f8981e67e857d29f03d8fc4093d451dec` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_164_9_csynth.rpt` | 142859 | `59710731f32eb9819865efee2ebfd0e44166b6edefc2765ef191d18ffa7d03bb` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_164_9_csynth.xml` | 163010 | `756d0fee2f6635d9b120faf77bae81c7df4bc679216c0d504f6e075d457659c5` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_168_6_csynth.rpt` | 23932 | `198c9362317fc757e8927cc6e8314db8e87443590211e40bf7c0eec098ec9f97` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_168_6_csynth.xml` | 29844 | `ebd6e897311bba1e70dde70b7ae6b4144f2ad48e114133df93637a1090c47b95` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_177_11_csynth.rpt` | 31234 | `6c60efc29f1f4af4c4885001b87eaedfad989cca424d66298c044bcc8b1026e6` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_177_11_csynth.xml` | 39947 | `d147224a9fea341250691fb97836d4c0562ee6ca697e9b941d6f9c0500777bae` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_188_9_csynth.rpt` | 142859 | `941cd7e4ad4416e7aad2251e9afc3a2f591eeb0cc656062e202d9561fb52df96` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_188_9_csynth.xml` | 163010 | `a9312430836d77c2580964823afafbf11551ec2da8d89aecefefb43269bf1800` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_201_11_csynth.rpt` | 31234 | `fea2e9aaa82b43b94e8a2322955dfe22165da6cbbba0e05d01ac9312ef4d651f` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_201_11_csynth.xml` | 39947 | `f56643318c9ba4221734cc076b36ef464ef9897bb49d6a4529ec69988f84b059` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_35_11_csynth.rpt` | 8252 | `7898543f23dcc3609aaf29ccfcdbe3a3551ef33ca6fc15ff29c56024f72c65e4` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_35_11_csynth.xml` | 4652 | `43e616bf44efee73abdd8285990a4c220c239fb2682dae9436788220ed5bf6c3` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_35_1_csynth.rpt` | 8212 | `5dd6b447565fa618b97b2b5d20c924804cc970d0b40193474bac3a3f993a3099` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_35_1_csynth.xml` | 4637 | `5531fd96be1c78a28bcf927725a244e0e9818face1eff91f3a22350275d7a447` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_54_12_csynth.rpt` | 8751 | `18fa3853a0e6ae8e1225cdfd1ac370f441d4cb1c40bc768232c3028c96d2e71b` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_54_12_csynth.xml` | 5254 | `6aba6ba834440d64d3d4cde1f7f9c47cb28598b52507986f3cdc9df73a41f3ce` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_54_13_csynth.rpt` | 8781 | `cd3923da1ebdf4245c4cfa286eb6d1e48a2ba23c91238fef7a9efe830d07dbf4` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_54_13_csynth.xml` | 5252 | `7219d318acd2c69a7455a43e0922af4e4d1bcab2b88a1312543ac24e3f449c6c` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_54_14_csynth.rpt` | 8039 | `f92d0098afcccd13f133babbbbee1dcb0a3a16c83a7c5bc4a55d53cdaefa1021` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_54_14_csynth.xml` | 4738 | `8db4177171494ae51c4c7f4befebbd81d4dfeea4a070308bffa12c086c605c40` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_54_1_csynth.rpt` | 8780 | `da6e8c152c00a4c3f20bc9a86dafd317e918b75b20c8deaa916a4a22e3790e09` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_54_1_csynth.xml` | 5169 | `18902b4a02dc1ad693bc80c942602340210613d1447430352bc125454f8a2109` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_59_11_csynth.rpt` | 8252 | `e8cc6168425aa0ded1c3c2bf36157bd4a79c0e25b7993d63104a311bdd6fa7fe` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_59_11_csynth.xml` | 4652 | `6e887fe274f94e0b20d1a12a0689c35a3bee2aca8292b89fecd17576c6d491c1` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_59_1_csynth.rpt` | 8212 | `73536fbfb568952973154d2a6249019783e44f6c0c29967354f028bc8ba001fb` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_59_1_csynth.xml` | 4637 | `7a6b8252f05a3fb1c4bfcb1691bc1e3334601ca5cedf6fb4ba1e88893f5c75ca` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_12_csynth.rpt` | 8751 | `a4629e9bfe522e8b58a9c2f1a284f00ba9d7a114821b8ba55ed75295e1cbaae4` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_12_csynth.xml` | 5256 | `7608eaca5aab7cc2dd3303ae5a42caab53926bd680dca13c93cb4caa9d6527d5` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_13_csynth.rpt` | 8781 | `d127339e00cbebcf07cb5322dcb532762d1eb10a334a788f0bf7881d21dd4a0f` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_13_csynth.xml` | 5254 | `aeec366d805c201d7a05c7a23e68fa27be4ab8305f9efc9056cdc2fade72b2a2` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_14_csynth.rpt` | 8039 | `967f5a769ed69a97a6736136129f44fe4e1a0c1f8c444d41b97583783c23bd20` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_14_csynth.xml` | 4738 | `17c6564af080913662fb1848b2707817fc1ee89a8d51120e7991233e9534ab37` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_1_csynth.rpt` | 8780 | `8c97b754bfe13893bb56c44ef932f076f8f92fd431fca524dff5b4b288f3f095` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_Pipeline_VITIS_LOOP_78_1_csynth.xml` | 5169 | `d0cfb318300777fee3c88e3eba88fccd2c14433d95a248d4d392c4351bacea97` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_s_csynth.rpt` | 133755 | `15e3ea441dddd927836a1efd8e305c3fbff627e10d16c35815917a7db64d97d8` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/p_anonymous_namespace_is_scorer_core_false_s_csynth.xml` | 49021 | `2c1a7411bc6fa49316108598fc3085b91420d4c1767d81f341cce6ad230ba02c` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/csynth/summary.json` | 2713 | `5ec6d9a914de97567380b09e3b19dd46d885fb297ef053a8f2c00f8a307131e9` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/fixtures/FORMAT.md` | 4002 | `cb43c42f330dcfaf0ee4d202706b14faeded3fa364fd48ef443191e7bae27a82` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/fixtures/full_grounding.bin` | 272864 | `c495381644662df8adce105c6e9ec65bda053996cead29a7dea614e35e971441` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/fixtures/full_pointing.bin` | 272864 | `530d68e775fc7403972d2fec5242c38e70bef90130ed15ee58ace30543bdd19d` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/fixtures/manifest.json` | 4872 | `93f48bd5b279a364d23b67df82e4e9a962f4952699407adfb57f7c7ea2eafebc` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/fixtures/primitives.bin` | 2942 | `7c0bb30e86f6570eed41664f55e2634ed1692fc7a4e196c6c55a7f53c971807f` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/fixtures/stress_g400.bin` | 5056 | `6920894bcf2f858ef615be86397d7c5cc68e99ec970b66b11e416b81935b9724` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/fixtures/synthetic_all_equal.bin` | 8544 | `c0f694ded0349be7ad467d95861a496f1fab9bca8fdf4e0b434576bdff0964a5` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/fixtures/synthetic_g1.bin` | 1292 | `c52dedb82ce57892afb59939455f0660d06abab905d30a5aed8ebf3475b8066a` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/fixtures/synthetic_mixed_sign.bin` | 16832 | `a7cd903d8fb08ee8f484cf68ca65564bb323412b207b67bca06ae4c710610a40` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/fixtures/synthetic_zero.bin` | 8544 | `8b23425e6fa0500a1e6eb808a2d7ef3b1599645692b1541417fcd9beec405727` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/fixtures/wv_q.bin` | 131072 | `126bd8421dd128a318115c301bf9d1c7d336912150d19d2ec35203ed55d89d3b` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/logs/01_python_fixtures.log` | 3192 | `a8293385db8936dd323ee2a347a06d3afdb80d1901efafd6eeda2147e709e821` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/logs/02_gpp_test.log` | 798 | `a77420d3868141be1e54f2ebb0cc9c932743bb96051ec786ef563de422eb2916` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/logs/03_asan_ubsan.log` | 805 | `b5b413340ac94db0662c1038c0da9773945be55b95e9fb44384533b7a9189fd1` |
| `artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/logs/04_hls_csim.log` | 4165 | `baeda64cc5aa0da367f0bb77df662d91dfb33d206312558644243ee7335b7d91` |
| `fpga_accel/integer_selector_scorer_hls/Makefile` | 2206 | `6c099fcc9d9dfdd43bc19df74d7e46357f8ea0398b285e117e15a1a7d6cf142b` |
| `fpga_accel/integer_selector_scorer_hls/README.md` | 5403 | `f40b0543e61d3e115b825e7d7db8ef893920faa19e610fe7bb6da4b50ee57e2b` |
| `fpga_accel/integer_selector_scorer_hls/include/integer_selector_scorer.h` | 7416 | `ae80d415356f2ceb45ab9a7b7fafbfbe17242fc8fe9916b8c24ad9ac7a4135de` |
| `fpga_accel/integer_selector_scorer_hls/quantization_contract.md` | 3813 | `a07a7bbd47ad4b5c57c49805e549648399bd6ba668f3a8688b72d0f547ed00ae` |
| `fpga_accel/integer_selector_scorer_hls/register_map_draft.md` | 2797 | `316afddbed16078a85f4dac128d0b55860830afa43b86e1f29356ba9657dcc7b` |
| `fpga_accel/integer_selector_scorer_hls/scripts/run_csim.tcl` | 2045 | `f13d18740bb5656dbed71a7e5d67961d9c3029f3895c254fa7ba84eaacfbc3f0` |
| `fpga_accel/integer_selector_scorer_hls/scripts/run_csynth.tcl` | 1077 | `83a97b8180365b480c870d7e3ebd7db67e662b92a161489ea5f719a1f622ffb7` |
| `fpga_accel/integer_selector_scorer_hls/scripts/run_stage.sh` | 2344 | `70b39159dd6ab012ea882c3a1ea1c2957093c89307a2edb6588613b2a5516067` |
| `fpga_accel/integer_selector_scorer_hls/scripts/summarize_csynth.py` | 10766 | `1468b0e2c6c8df57bdb83859262b6f65d57ec51b107709091d6f0978c3baa904` |
| `fpga_accel/integer_selector_scorer_hls/src/integer_selector_scorer.cpp` | 13547 | `57dd9ca56b5e61d3609923a2d3cd72ed1d593335e804292f4784d7aca2e43b50` |
| `fpga_accel/integer_selector_scorer_hls/tb/tb_integer_selector_scorer.cpp` | 16118 | `a6d77c4d049028b648874efed2fc6737e4483a880f3bbf364b516e5dd46a991f` |
| `fpga_accel/integer_selector_scorer_hls/tests/test_integer_selector_scorer_hls.py` | 11643 | `334192b284820a5f1e43ad721d5ae395c4c7be9612cb963461b2eedf6a6213e9` |
| `fpga_accel/integer_selector_scorer_hls/tools/fixture_format.md` | 4002 | `cb43c42f330dcfaf0ee4d202706b14faeded3fa364fd48ef443191e7bae27a82` |
| `fpga_accel/integer_selector_scorer_hls/tools/generate_fixtures.py` | 20453 | `bee29a74b4a6ecaa18c962265381eea4dbfed1301fa1d1ff9d6735d83af5d3fb` |

