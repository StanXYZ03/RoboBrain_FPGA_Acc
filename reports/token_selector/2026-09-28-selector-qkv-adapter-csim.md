# Selector→LayerNorm-Q8_0 适配器与独立 QKV bias-add 参考核（g++/Sanitizer/HLS C-sim）

- 日期：2026-09-28
- 决策：`SELECTOR_QKV_ADAPTER_HLS_IMPLEMENTED / ADAPTER_LEGAL_M_192_256_384 / PARAM_CACHE_8KIB_BRAM / FUSED_LAYERNORM_Q8_0_BYTE_EXACT_VS_PYTHON_GOLDEN / INDEPENDENT_QKV_BIAS_ADD_KERNEL / GPP_ASAN_HLS_CSIM_PASS / CSIM_NOT_CSYNTH / ATTENTION_STILL_MISSING / NO_END_TO_END_SPEEDUP_CLAIM`
- 范围：CPU/HLS **C-sim only**；只读真实 BF16 张量 header+切片；不加载完整模型；无 GPU/板卡/综合/cosim/IP 导出/Vivado/联网；未改现有 selector/q8q8/pl_resident_ffn HLS 源码与模型文件。

## 0. 边界声明（必读）

- 激活为**确定性合成**，非真实采集 hidden；仅 `blocks.1.norm1.weight/bias` 与 `blocks.1.attn.qkv.bias` 为真实张量。
- 数值链只到 `+qkv.bias`；下游 attention 缺失，**不**声称完整、**不**判任务精度、**不**声称端到端加速。
- HLS C-sim **不是**综合/时序/资源/板级结果。

## 1. 接口合同

| 顶层 | 输入 | 输出 |
|---|---|---|
| `selector_ln_q8_adapter` | `int8[M,1024]` 128-bit 小端 packed + 全局 `s>0` + 真实 `norm1` gamma/beta (4×fp32/128-bit) | Q8_0 `[M,32]` 256-bit payload + per-block binary16 scale（与现有 q8q8/pl_resident_ffn 合同一致） |
| `qkv_bias_add` | Q8Q8 输出 4×fp32/128-bit `[M,3072]` + fp32 `bias[3072]` | 同布局 `y_in + bias[col]` |

- Selector clamp `[-128,127]`；Q8_0 饱和 `[-127,127]`；`d=amax/127`；`scale=float_to_half_bits(d)`；round 半离零。
- `s<=0`/NaN/±inf → `SQA_ERR_SCALE`；adapter 的 `rows` 不属于 `{192,256,384}`（含 `1/191/193/385/4320`）→ `SQA_ERR_DIMENSION`。
- bias-add 保留通用 `rows∈[1,4320]`、`cols∈[1,9728]` 且 `cols%4==0`；本轮正式 qkv case 为 `M=192/256/384, N=3072`。

### 1.1 本轮合同修正（逐条）
- adapter 合法 rows：`rows<=0 || rows>4320 (SQA_MAX_ROWS=4320)` → **`rows in {192,256,384} (SQA_MAX_ROWS=384, explicit legal check)`**。原因：Selector legal budgets are K=48/64/96 only, so M=4K is 192/256/384。拒绝示例：`[0, -1, 1, 191, 193, 385, 4320, 385]`。
- bias-add：qkv_bias_add keeps the general row gate 1..4320; formal qkv case is M=192/256/384,N=3072。
- 参数存储：`norm_gamma/norm_beta re-read from m_axi per row/block` → **`norm_gamma/norm_beta read once into 256x128-bit on-chip caches (4KiB each, 8KiB total) explicitly BIND_STORAGE type=ram_2p impl=bram; rows read from cache`**；不变量：math, Q8_0 payload/scale, port layout unchanged; byte-exact golden stays 0 mismatch。

## 2. 数学等价推导

令 `x=q/s`，则 `mean_x=mean_q/s`、`var_x=var_q/s²`（population variance）：

```
LN(x) = (x-mean_x)/sqrt(var_x+eps)*gamma + beta
      = (q-mean_q)/sqrt(var_q + eps*s*s)*gamma + beta
```

- 因此可在 int8 域直接计算，不物化 `x`；`1/s` 广播只是反量化，**不等于** LayerNorm。
- `var_q = E[q²]-E[q]²`（负值截 0），与 `ffn_vector.cpp::row_statistics` 的 population variance 完全一致；常量行 `var_q=0` 且 `eps*s*s=0` 时取 `inverse_norm=0`，结果恰为 `beta`。
- `q=-128` 参与统计；只有输出 Q8_0 饱和到 `-127`。

## 3. 真实张量 fixture 取证（只读，未加载完整模型）

- shard：`model-00002-of-00002.safetensors`（3908490048 B），header sha256 `87e6341399ae7de5…`，读后未改动：`True`。

| 张量 | dtype | shape | data_offsets | raw sha256 |
|---|---|---|---|---|
| `model.visual.blocks.1.norm1.weight` | BF16 | [1024] | [3128115712, 3128117760] | `41b623b3b4f0de8d…` |
| `model.visual.blocks.1.norm1.bias` | BF16 | [1024] | [3128113664, 3128115712] | `855c11e4acc32c28…` |
| `model.visual.blocks.1.attn.qkv.bias` | BF16 | [3072] | [3105028608, 3105034752] | `83c5f50d34c576dc…` |

- adapter fixture：`real_block1_adapter_fixture.bin` sha256 `02e16b51424e858d…`，M=192 (K=48)，s=33.7386。
- bias fixture：`real_block1_bias_fixture.bin` sha256 `7461ad9bd9bea144…`，M=384，N=3072。

## 4. 逐字节对拍与逐级误差

C++/HLS 内核对 Python 冻结 golden（fused 公式 + repository half）逐字节比较：
- HLS C-sim(adapter)：`SQA_FIXTURE name=adapter_real_block1 rows=192 cols=1024 scale=33.7386169 payload_byte_mismatch=0 scale_byte_mismatch=0 fused_vs_direct[max_abs=4.29153442e-06 relative_l2=2.78197902e-07 cosine=1] quant_vs_direct[max_abs=0.0276881605 rmse=0.0050942842 relative_l2=0.00722061636 cosine=0.999973933] result=PASS`
- HLS C-sim(adapter)：`SQA_FIXTURE name=qkv_bias_add_real rows=384 cols=3072 byte_mismatch=0 result=PASS`
- g++ 普通：`SQA_FIXTURE name=adapter_real_block1 rows=192 cols=1024 scale=33.7386169 payload_byte_mismatch=0 scale_byte_mismatch=0 fused_vs_direct[max_abs=4.29153442e-06 relative_l2=2.78197902e-07 cosine=1] quant_vs_direct[max_abs=0.0276881605 rmse=0.0050942842 relative_l2=0.00722061636 cosine=0.999973933] result=PASS`
- g++ 普通：`SQA_FIXTURE name=qkv_bias_add_real rows=384 cols=3072 byte_mismatch=0 result=PASS`
- ASan+UBSan：`SQA_FIXTURE name=adapter_real_block1 rows=192 cols=1024 scale=33.7386169 payload_byte_mismatch=0 scale_byte_mismatch=0 fused_vs_direct[max_abs=4.29153442e-06 relative_l2=2.78197902e-07 cosine=1] quant_vs_direct[max_abs=0.0276881605 rmse=0.0050942842 relative_l2=0.00722061636 cosine=0.999973933] result=PASS`
- ASan+UBSan：`SQA_FIXTURE name=qkv_bias_add_real rows=384 cols=3072 byte_mismatch=0 result=PASS`

### 4.1 adapter 真实 block1（M=192, K=48, 真实 gamma/beta）

| 对象 | max_abs | RMSE | relative_L2 | cosine |
|---|---|---|---|---|
| fused golden vs direct FP32 LN | 9.53674e-07 | 5.64961e-08 | 8.00774e-08 | 1.000000000 |
| golden Q8_0 dequant vs direct FP32 LN | 0.0276882 | 0.00509428 | 0.00722062 | 0.999973933 |

- fused golden payload 对 direct-quant payload 字节差：0/196608，scale 值差：2/6144。
- 上表为 Python golden 的数值；表下 C++/HLS 日志中 `fused_vs_direct.max_abs≈4.3e-6`（C++ direct 用顺序 float32 求和，Python 用向量化求和），两者都远小于 Q8_0 量化步长。
- C++/HLS 的 adapter payload/scale 与 bias 逐字节差均为 **0**，即与 Python 冻结 golden **逐字节一致**。

### 4.2 必需案例（合成激活，K=48/64/96 → M=192/256/384，多 seed）

| case | M | K | s | 饱和 | fused_vs_direct rel_L2 | dequant_vs_direct rel_L2 | cosine |
|---|---|---|---|---|---|---|---|
| m192_k48_seed20260928 | 192 | 48 | 33.7386 | 1 | 7.91e-08 | 0.00535 | 0.999985679 |
| m192_k48_seed20260929 | 192 | 48 | 32.2339 | 1 | 7.78e-08 | 0.00533 | 0.999985817 |
| m192_k48_seed20260930 | 192 | 48 | 30.1383 | 1 | 8.2e-08 | 0.00535 | 0.999985663 |
| m256_k64_seed20260930 | 256 | 64 | 30.6102 | 1 | 8.31e-08 | 0.00534 | 0.999985736 |
| m256_k64_seed20261001 | 256 | 64 | 28.4206 | 0 | 8.43e-08 | 0.00533 | 0.999985772 |
| m384_k96_seed20261001 | 384 | 96 | 29.9629 | 0 | 7.77e-08 | 0.00534 | 0.999985734 |
| m384_k96_seed20261002 | 384 | 96 | 29.9577 | 1 | 7.68e-08 | 0.00535 | 0.999985712 |

## 5. 测试与原始日志

| 门 | 结果 | 证据 |
|---|---|---|
| 聚焦 Python 合同测试 | Ran 24 tests, OK | `evaluation/tests/test_selector_qkv_adapter_hls.py` |
| 全量 evaluation/tests 回归 | Ran 300 tests, OK | `evaluation/tests` |
| g++ 普通 | 18/18 PASS；CSim0err=False；runtime_err=0；asan_err=0 | `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/gpp_test.log` |
| ASan+UBSan | 18/18 PASS；CSim0err=False；runtime_err=0；asan_err=0 | `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/asan_ubsan_run.log` |
| HLS C-sim(adapter) | 18/18 PASS；CSim0err=True；runtime_err=0；asan_err=0 | `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/hls_csim_adapter.log` |
| HLS C-sim(bias) | 18/18 PASS；CSim0err=True；runtime_err=0；asan_err=0 | `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/hls_csim_bias.log` |

## 6. 主机工具链 workaround（仅 host testbench）

Ubuntu 24.04 / glibc 2.39 下复用仓库既有可复现 workaround：
1. `LD_LIBRARY_PATH=/home/xilinx/xic/lib/lnx64.o/SuSE` 提供 Tcl 需要的 `libtinfo.so.5`；
2. `COMPILER_PATH=<symlink ld → /usr/bin/ld>`、`LIBRARY_PATH=/usr/lib/x86_64-linux-gnu`、`AP_GCC_PATH`=自带 GCC 8.3.0、`config_csim -ldflags "-no-pie"` 规避 binutils 2.37 无法读取系统 `libm.so.6 .relr.dyn`；DUT/算法未变。
3. ASan/UBSan 使用 `-fno-sanitize=shift`：Vitis HLS `ap_int` 头内部 left-shift-of-negative 惯用法会误报；本内核位移全部作用于无符号值且位移量有界，未丢失内核侧检查。

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
- `evaluation/tests/test_selector_qkv_adapter_hls.py`（17120 B，sha256 `2b11f64db95d20b7…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/asan_ubsan_run.log`（4774 B，sha256 `92a322ccd59b06e7…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/fixture_manifest.json`（4375 B，sha256 `c56b278877f301e2…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/gpp_test.log`（4688 B，sha256 `33fcfb41828f850b…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/hls_csim_adapter.log`（6997 B，sha256 `6327252579113390…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/hls_csim_bias.log`（6988 B，sha256 `2191aefa9262cdd2…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/real_block1_adapter_fixture.bin`（413740 B，sha256 `02e16b51424e858d…`）
- `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/real_block1_bias_fixture.bin`（9449500 B，sha256 `7461ad9bd9bea144…`）

## 8. 限制与下一步

- 仅 HLS C-sim：不是综合，不是时序/资源/功耗，不是板级结果。
- 激活为合成，非真实采集；只有 norm1 gamma/beta 与 qkv.bias 是真实张量。
- 数值链止于 +qkv.bias；reshape/split、RoPE、softmax、PV、attn.proj、残差、norm2、fc1/GELU/fc2 与 blocks 2..23 全部缺失。
- 不声称端到端加速，也不判任务精度。
- 仓库不存在 Vision tower 的 Q8_0 激活/权重产物；Q8Q8 权重侧工作不在本轮范围。
- Python golden 的 direct 参考使用向量化求和而 C++ 使用顺序求和，fused_vs_direct 的 max_abs 数量级略有差异（均远小于 Q8_0 步长）。

**精确下一步（未执行）**：在 +qkv.bias 之后实现 attention 各算子（reshape/split、RoPE、softmax、PV、attn.proj+bias、残差、norm2、fc1/GELU/fc2），仅在独立授权下再考虑 csynth/BD/驱动/上板；同时采集真实 block0 hidden 激活以做 fixture 级数值验证。

