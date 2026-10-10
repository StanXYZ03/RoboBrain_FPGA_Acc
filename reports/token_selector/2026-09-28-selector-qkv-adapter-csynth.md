# Selector→LayerNorm-Q8_0 适配器与独立 QKV bias-add：受控 HLS csynth

- 日期：2026-09-28
- 决策：`SELECTOR_QKV_ADAPTER_HLS_IMPLEMENTED / ADAPTER_LEGAL_M_192_256_384 / PARAM_CACHE_8KIB_BRAM / FUSED_LAYERNORM_Q8_0_BYTE_EXACT_VS_PYTHON_GOLDEN / INDEPENDENT_QKV_BIAS_ADD_KERNEL / GPP_ASAN_HLS_CSIM_PASS / CONTROLLED_CSYNTH_RUN / ATTENTION_STILL_MISSING / NO_END_TO_END_SPEEDUP_CLAIM / ADAPTER_CSYNTH_TIMING_FAIL / BIAS_CSYNTH_GATE_PASS / NO_COSIM_NO_ROUTE`
- 范围：**只 `csynth_design`**（Vitis HLS 2023.2、part `xczu9eg-ffvb1156-2-e`、`create_clock -period 5.0`、默认 uncertainty 不放宽）；无 C/RTL cosim、无 IP 导出、无 Vivado/BD/驱动、无板卡、无 GPU、无联网。
- 大工程只放 `/tmp`；仓库只保存小报告与 hash。未改现有 selector/q8q8/pl_resident_ffn 源码与模型文件。

## 0. 前置合同修正（本轮）

- adapter 合法 rows：`rows<=0 || rows>4320 (SQA_MAX_ROWS=4320)` → **`rows in {192,256,384} (SQA_MAX_ROWS=384, explicit legal check)`**；拒绝示例 `[0, -1, 1, 191, 193, 385, 4320, 385]`。qkv_bias_add keeps the general row gate 1..4320; formal qkv case is M=192/256/384,N=3072。
- 参数存储：`norm_gamma/norm_beta re-read from m_axi per row/block` → **`norm_gamma/norm_beta read once into 256x128-bit on-chip caches (4KiB each, 8KiB total) explicitly BIND_STORAGE type=ram_2p impl=bram; rows read from cache`**；不变量：math, Q8_0 payload/scale, port layout unchanged; byte-exact golden stays 0 mismatch。

## 1. 逐字节 C-sim 回归（修正后重跑，0 mismatch）

| 门 | 结果 | 证据 |
|---|---|---|
| g++ 普通 | 18/18 PASS；CSim0err=False；runtime_err=0；asan_err=0 | `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/gpp_test.log` |
| ASan+UBSan | 18/18 PASS；CSim0err=False；runtime_err=0；asan_err=0 | `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/asan_ubsan_run.log` |
| HLS C-sim(adapter) | 18/18 PASS；CSim0err=True；runtime_err=0；asan_err=0 | `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/hls_csim_adapter.log` |
| HLS C-sim(bias) | 18/18 PASS；CSim0err=True；runtime_err=0；asan_err=0 | `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/hls_csim_bias.log` |

- g++ 普通：`SQA_FIXTURE name=adapter_real_block1 rows=192 cols=1024 scale=33.7386169 payload_byte_mismatch=0 scale_byte_mismatch=0 fused_vs_direct[max_abs=4.29153442e-06 relative_l2=2.78197902e-07 cosine=1] quant_vs_direct[max_abs=0.0276881605 rmse=0.0050942842 relative_l2=0.00722061636 cosine=0.999973933] result=PASS`
- g++ 普通：`SQA_FIXTURE name=qkv_bias_add_real rows=384 cols=3072 byte_mismatch=0 result=PASS`
- ASan+UBSan：`SQA_FIXTURE name=adapter_real_block1 rows=192 cols=1024 scale=33.7386169 payload_byte_mismatch=0 scale_byte_mismatch=0 fused_vs_direct[max_abs=4.29153442e-06 relative_l2=2.78197902e-07 cosine=1] quant_vs_direct[max_abs=0.0276881605 rmse=0.0050942842 relative_l2=0.00722061636 cosine=0.999973933] result=PASS`
- ASan+UBSan：`SQA_FIXTURE name=qkv_bias_add_real rows=384 cols=3072 byte_mismatch=0 result=PASS`
- HLS C-sim(adapter)：`SQA_FIXTURE name=adapter_real_block1 rows=192 cols=1024 scale=33.7386169 payload_byte_mismatch=0 scale_byte_mismatch=0 fused_vs_direct[max_abs=4.29153442e-06 relative_l2=2.78197902e-07 cosine=1] quant_vs_direct[max_abs=0.0276881605 rmse=0.0050942842 relative_l2=0.00722061636 cosine=0.999973933] result=PASS`
- HLS C-sim(adapter)：`SQA_FIXTURE name=qkv_bias_add_real rows=384 cols=3072 byte_mismatch=0 result=PASS`
- HLS C-sim(bias)：`SQA_FIXTURE name=adapter_real_block1 rows=192 cols=1024 scale=33.7386169 payload_byte_mismatch=0 scale_byte_mismatch=0 fused_vs_direct[max_abs=4.29153442e-06 relative_l2=2.78197902e-07 cosine=1] quant_vs_direct[max_abs=0.0276881605 rmse=0.0050942842 relative_l2=0.00722061636 cosine=0.999973933] result=PASS`
- HLS C-sim(bias)：`SQA_FIXTURE name=qkv_bias_add_real rows=384 cols=3072 byte_mismatch=0 result=PASS`

两条 fixture 的 payload/scale 与 bias 输出对 Python 冻结 golden 逐字节 mismatch 均为 **0**。

## 2. adapter csynth（top `selector_ln_q8_adapter`）

- project：`/tmp/robobrain_selector_qkv_adapter_csynth_adapter`；只 `csynth_design`。
- estimated period：**3.945 ns**；effective budget：**3.65 ns**（target 5.0 ns − uncertainty 1.35 ns）；slack：**-0.295 ns**；Fmax 253.49 MHz。
- latency cycles：best 1.0、average 649044.0、worst **2601752.0**；@200MHz：best 0.0 ms、average 3.2452 ms、worst **13.0088 ms** (< 33.3 ms)。best=1 来自非法 rows 的提前返回路径，不是 M=192。
- 资源：BRAM_18K 2.96%、DSP 1.90%、FF 2.08%、LUT 6.42%、URAM 0.00%（used `{'BRAM_18K': 54, 'DSP': 48, 'FF': 11419, 'LUT': 17596, 'URAM': 0}` / available `{'BRAM_18K': 1824, 'DSP': 2520, 'FF': 548160, 'LUT': 274080, 'URAM': 0}`）。任一资源 <50%。
- II warnings / critical path：critical path 为 quantize amax 归约 `fcmp→select→store`（3.945 ns），模块 `selector_ln_q8_adapter_Pipeline_VITIS_LOOP_129_1`；`HLS 200-871`/`200-1016` 在合法 M=384 下仍触发。`HLS 200-885` 另报参数 cache 载入循环与 bias 内层循环的 II violation（gmem0 端口限制），不改变结果。

### 2.1 允许的调度/存储 pragma 修复轮（最多 2 轮，均无法关闭 timing）

| 轮 | 变更 | estimated period | 结果 | 证据 |
|---|---|---|---|---|
| 0 | original design (no scheduling pragma) | 3.945 ns | csynth completed; timing gate FAIL (3.945 > 3.650 ns) | `adapter_round0/console.log` |
| 1 | scheduling pragma: HLS PIPELINE II=2 on the Q8_0 amax loop | 3.945 ns | csynth completed; timing gate still FAIL (3.945 ns, same fcmp->select->store path), latency reaches II=2 but the combinational compare is not broken | `adapter_round1/console.log` |
| 2 | scheduling pragma: named compare + BIND_OP fcmp impl=fabric latency=1 | - | pre-synthesis ERROR: HLS 214-372 (fcmp impl=fabric unsupported on xczu9eg) and HLS 214-383 (fcmp latency range [-1,-1]); not a valid synthesized design | `adapter_round2_bindop_unsupported/console.log` |

- **真实根因**：Q8_0 quantize_block 的 32 路 amax 归约是 loop-carried 依赖 (amax -> fcmp -> select -> amax)。2023.2 在该器件上把单精度浮点 `>` 实现为约 2.782 ns 的 fcmp，加上 select 0.449 ns 与 store 0.427 ns，单周期组合路径 3.945 ns > 默认 uncertainty 下的 3.650 ns effective budget。只允许调度/存储 pragma 的等价修复无法打断该组合路径：II=2 不生效；`BIND_OP op=fcmp` 在该器件不支持。同一 3.945 ns 路径也存在于现有 pl_resident_ffn 的 quantize_block csynth 中，证明这是算法实现的固有路径，而非本轮参数 cache 引入。
- gate：**FAIL**；failed_checks `['estimated_period_le_effective_budget', 'timing_slack_non_negative']`；按门槛判 FAIL，未继续 cosim/route。
- 合法 M 可推导周期（row-loop trip [192, 384]）：M=192 → **1294848 cycles / 6.4742 ms**（工具直接报告）；M=384 → **2601216 cycles / 13.0061 ms**（工具直接报告）；M=256 → 区间 [1726464, 1734144] cycles（**有界推导，非工具单独报告**）。

## 3. bias csynth（top `qkv_bias_add`）

- project：`/tmp/robobrain_selector_qkv_adapter_csynth_bias`；0 error。
- estimated period：**3.65 ns**；effective budget：**3.65 ns**；slack **0.0 ns**；Fmax 273.97 MHz。
- latency cycles：best 1.0、average None（动态 rows/cols 下 tool 报 `undef`）、worst **21012512.0**（M=4320,N=9728 硬边界）；@200MHz：best 0.0 ms、average None ms（undef）、worst **105.0626 ms**。
- 资源：BRAM_18K 1.75%、DSP 0.36%、FF 0.73%、LUT 1.41%、URAM 0.00%（used `{'BRAM_18K': 32, 'DSP': 9, 'FF': 4023, 'LUT': 3869, 'URAM': 0}` / available `{'BRAM_18K': 1824, 'DSP': 2520, 'FF': 548160, 'LUT': 274080, 'URAM': 0}`）。任一资源 <50%。
- 正式 M=384,N=3072 边界：words 294912、achieved II 2 → **589847 cycles / 2.9492 ms @200MHz**（由工具报告的 achieved II 推导，非直接报告值）；工具硬边界 M=4320,N=9728 = 21012512 cycles。
- II warnings：`HLS 200-885` bias 内层平坦流水线 II violation（gmem0 端口限制，achieved II=2）；`HLS 200-790` loop constraints 未满足，不改变结果。
- gate：**PASS**；checks `{'zero_errors': True, 'latency_finite': True, 'estimated_period_le_effective_budget': True, 'timing_slack_non_negative': True, 'any_resource_below_50_percent': True}`。

## 4. 测试

| 门 | 结果 |
|---|---|
| 聚焦 Python 合同测试 | Ran 24 tests, OK |
| 全量 evaluation/tests 回归 | Ran 300 tests, OK |
| adapter csynth gate | FAIL |
| bias csynth gate | PASS |

## 5. 边界与限制

- 仅 csynth_design：不是 C/RTL cosim、不是 place/route、不是 bitstream、不是板级结果。
- 激活为合成，非真实采集；只有 norm1 gamma/beta 与 qkv.bias 是真实张量。
- 数值链止于 +qkv.bias；reshape/split、RoPE、softmax、PV、attn.proj、残差、norm2、fc1/GELU/fc2 与 blocks 2..23 全部缺失。
- 不声称端到端加速，也不判任务精度；Q8Q8 权重侧工作不在本轮范围。
- adapter 的 M=256 周期只能由 row-loop 迭代延迟范围给出有界推导，不是工具单独报告值。
- bias 的 cols 为运行时标量，M=384,N=3072 周期由 achieved II 推导；工具硬边界为 M=4320,N=9728。
- HLS 200-790 仍报 loop constraints 未满足（adapter 参数 cache 载入循环与 bias 内层循环存在 II violation），这不是 correctness/timing 门槛；gate 以 0 error/有限 latency/timing/资源判定。
- adapter csynth timing gate FAIL：estimated period 3.945 ns > 3.650 ns effective budget；按门槛如实判 FAIL，未继续 cosim/route。

**精确下一步（未执行）**：adapter csynth 在默认 1.35 ns uncertainty / 5.0 ns 下 estimated 3.945 ns > 3.650 ns，未过 timing 门槛；下一步只有在独立授权下才可用不等价手段（例如把 32 路 amax 归约改成平衡树/分段最大值并对 fcmp 打拍，或重排量化块调度）以打破该 loop-carried 组合路径，再重跑同一条只 csynth_design 的流程；在此之前不 cosim、不导出 IP、不 route、不建 BD、不写驱动、不上板。+qkv.bias 之后的 attention 各算子与真实 block0 hidden 采集仍待完成。

## 6. 复现命令

```bash
cd fpga_accel/selector_qkv_adapter_hls
make test && make asan_run && make csim && make csim_bias
bash scripts/run_csynth.sh adapter   # 只 csynth_design，工程在 /tmp
bash scripts/run_csynth.sh bias
```

## 7. 文件变更

- `README.md`（4431 B，sha256 `2db43f9a6598f412…`）
- `fpga_accel/selector_qkv_adapter_hls/README.md`（8499 B，sha256 `50f406da17bd5239…`）
- `fpga_accel/selector_qkv_adapter_hls/Makefile`（2459 B，sha256 `402c2d186cd38819…`）
- `fpga_accel/selector_qkv_adapter_hls/include/selector_qkv_adapter.h`（4668 B，sha256 `36c8e3e830b2fd51…`）
- `fpga_accel/selector_qkv_adapter_hls/src/selector_qkv_adapter.cpp`（12345 B，sha256 `0b4ae41fee0e2ce1…`）
- `fpga_accel/selector_qkv_adapter_hls/tb/tb_selector_qkv_adapter.cpp`（39868 B，sha256 `b21ab4494c63160d…`）
- `fpga_accel/selector_qkv_adapter_hls/tools/selector_qkv_golden.py`（16571 B，sha256 `c401a0a63291da8b…`）
- `fpga_accel/selector_qkv_adapter_hls/tools/generate_fixtures.py`（11721 B，sha256 `d19487eed01ea32b…`）
- `fpga_accel/selector_qkv_adapter_hls/scripts/run_hls.tcl`（2488 B，sha256 `bec2b3e1897ba82e…`）
- `fpga_accel/selector_qkv_adapter_hls/scripts/run_stage.sh`（1881 B，sha256 `f230412033efbf7a…`）
- `fpga_accel/selector_qkv_adapter_hls/scripts/run_csynth.tcl`（1589 B，sha256 `4b6ee5b23e86a728…`）
- `fpga_accel/selector_qkv_adapter_hls/scripts/run_csynth.sh`（2108 B，sha256 `d02e26b44c488618…`）
- `scripts/validate_selector_qkv_adapter_csim.py`（30846 B，sha256 `59cdcb0d5a81a16c…`）
- `scripts/validate_selector_qkv_adapter_csynth.py`（36901 B，sha256 `e9d6cc53ce2b1b87…`）
- `scripts/summarize_selector_csynth.py`（9148 B，sha256 `ea170b14bf634bb5…`）
- `evaluation/tests/test_selector_qkv_adapter_hls.py`（17120 B，sha256 `2b11f64db95d20b7…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/asan_ubsan_run.log`（4774 B，sha256 `92a322ccd59b06e7…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/fixture_manifest.json`（4375 B，sha256 `c56b278877f301e2…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/gpp_test.log`（4688 B，sha256 `33fcfb41828f850b…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/hls_csim_adapter.log`（6997 B，sha256 `6327252579113390…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/hls_csim_bias.log`（6988 B，sha256 `2191aefa9262cdd2…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/real_block1_adapter_fixture.bin`（413740 B，sha256 `02e16b51424e858d…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/real_block1_bias_fixture.bin`（9449500 B，sha256 `7461ad9bd9bea144…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter/console.log`（47337 B，sha256 `0f4159ef1920607d…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter/csynth.rpt`（51983 B，sha256 `baee24263e7118f6…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter/csynth.xml`（169332 B，sha256 `481b9957d445911f…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter/csynth_design_size.rpt`（11526 B，sha256 `01d22c4364ebccba…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter/df_before.txt`（89 B，sha256 `9d215559433a2396…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter/selector_ln_q8_adapter_csynth.rpt`（38715 B，sha256 `ecdecec17770eb98…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter/summary.json`（4862 B，sha256 `f65eb83c654a4484…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter_round0/console.log`（47344 B，sha256 `612a8101601f85b7…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter_round0/csynth.rpt`（51983 B，sha256 `b3bc7a1353bda4ac…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter_round0/csynth.xml`（169332 B，sha256 `e706922d0779ef71…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter_round0/csynth_design_size.rpt`（11526 B，sha256 `638aeba1e6952978…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter_round0/df_before.txt`（89 B，sha256 `9d215559433a2396…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter_round0/selector_ln_q8_adapter_csynth.rpt`（38715 B，sha256 `ad4b82e4c2feb837…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter_round0/summary.json`（4862 B，sha256 `62fb438d202d236a…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter_round1/console.log`（47337 B，sha256 `0f4159ef1920607d…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter_round1/csynth.rpt`（51983 B，sha256 `baee24263e7118f6…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter_round1/csynth.xml`（169332 B，sha256 `481b9957d445911f…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter_round1/csynth_design_size.rpt`（11526 B，sha256 `01d22c4364ebccba…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter_round1/df_before.txt`（89 B，sha256 `9d215559433a2396…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter_round1/selector_ln_q8_adapter_csynth.rpt`（38715 B，sha256 `ecdecec17770eb98…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter_round1/summary.json`（4862 B，sha256 `f65eb83c654a4484…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter_round2_bindop_unsupported/console.log`（22460 B，sha256 `93ed8244316e7641…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/adapter_round2_bindop_unsupported/df_before.txt`（89 B，sha256 `9d215559433a2396…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/bias/console.log`（22638 B，sha256 `41aa3827837a6b8f…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/bias/csynth.rpt`（34031 B，sha256 `f99a71c555ed14f8…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/bias/csynth.xml`（99689 B，sha256 `1fc7cefab78d6d05…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/bias/csynth_design_size.rpt`（6332 B，sha256 `5a334f2df529e336…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/bias/df_before.txt`（89 B，sha256 `9d215559433a2396…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/bias/qkv_bias_add_csynth.rpt`（20370 B，sha256 `1c51827fe07532c1…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/bias/summary.json`（2087 B，sha256 `d2b1d4de5fd8db06…`）

