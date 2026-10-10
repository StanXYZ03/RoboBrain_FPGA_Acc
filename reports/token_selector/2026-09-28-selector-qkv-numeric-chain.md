# Selector → 真实 block1 fused QKV CPU 数值链验证（合成激活）

- 日期：2026-09-28
- 决策：`SELECTOR_QKV_CPU_NUMERIC_CHAIN_IMPLEMENTED / INTERFACE_AND_IMPLEMENTATION_SELF_CONSISTENT / SYNTHETIC_ACTIVATION / NO_TASK_ACCURACY_CLAIM`
- 范围：CPU-only；无 GPU、无板卡、无综合、无联网；只读真实 BF16 safetensors 的 4 个 block1 张量；不加载完整模型；不改任何 HLS/RTL/模型文件。

## 0. 边界声明（必读）

- `x_fp32` 是**确定性合成**的 block0 residual hidden，**不是**真实模型激活；不得声称是真实激活。
- 本报告只判**接口/数值实现是否自洽**，**不**给出“任务精度通过”结论。
- 数值链只覆盖到 `+qkv.bias`；reshape/split、RoPE、softmax、PV、`attn.proj`、残差、`norm2`、`fc1/GELU/fc2` 及 blocks 2..23 均未实现。
- 无时延/吞吐加速声明；下表 wall-clock 仅是本机 CPU 计时。

## 1. 量化合同

- Selector：`q = clamp(round_half_away(x * s), -128, 127)`，反量化 `x ≈ q/s`，zero point 0，全图共享单一对称 `s`；本报告 `s = 128.0 / amax(x)`。饱和区间 `[-128, 127]`。
- Q8_0：block=32，`d = amax/127`，`q = clamp(round_half_away(x/d), -127, 127)`，`x ≈ q*d`，scale 存 binary16。复用既有 `export_real_tensor_case.py::q8_quantize`（宽度整除 32 时直接委托；尾块通用实现由单测证明与之一致）。
- **差异**：selector 允许 `-128`，Q8_0 饱和到 `-127`；已用自洽检查 `selector_minus128_vs_q8_0_minus127` 记录。
- `1/s` 广播**只是反量化**，**不等价于** LayerNorm；链中显式执行 LayerNorm(norm1, eps=1e-6)。

## 2. 真实权重取证（只读）

- shard：`model-00002-of-00002.safetensors`（3908490048 B），header sha256 `87e6341399ae7de5098b310951993b0010876a5c805dbe0b2c0cfbef7c8cb94e`，读后未改动文件：`True`。

| 张量 | dtype | shape | shard | data_offsets | raw sha256 |
|---|---|---|---|---|---|
| `model.visual.blocks.1.norm1.weight` | BF16 | [1024] | model-00002-of-00002.safetensors | [3128115712, 3128117760] | `41b623b3b4f0de8d…` |
| `model.visual.blocks.1.norm1.bias` | BF16 | [1024] | model-00002-of-00002.safetensors | [3128113664, 3128115712] | `855c11e4acc32c28…` |
| `model.visual.blocks.1.attn.qkv.weight` | BF16 | [3072, 1024] | model-00002-of-00002.safetensors | [3105034752, 3111326208] | `2a6ffd8e261ee288…` |
| `model.visual.blocks.1.attn.qkv.bias` | BF16 | [3072] | model-00002-of-00002.safetensors | [3105028608, 3105034752] | `83c5f50d34c576dc…` |

- 确认 `qkv.weight` shape = `[3072, 1024]`、`qkv.bias` shape = `[3072]`；`norm1.weight/bias` shape = `[1024]`。

## 3. 案例与合成激活

| case_id | seed | M | K_groups | hidden | N | scale s | 饱和数 | x amax | x std |
|---|---|---|---|---|---|---|---|---|---|
| m192_k48_seed20260928 | 20260928 | 192 | 48 | 1024 | 3072 | 33.7386 | 1 | 3.79387 | 0.841429 |
| m192_k48_seed20260929 | 20260929 | 192 | 48 | 1024 | 3072 | 32.2339 | 1 | 3.97097 | 0.846775 |
| m192_k48_seed20260930 | 20260930 | 192 | 48 | 1024 | 3072 | 30.1383 | 1 | 4.24708 | 0.844758 |
| m8_k2_seed20260928 | 20260928 | 8 | 2 | 1024 | 3072 | 39.1867 | 0 | 3.26641 | 0.836507 |

## 4. 逐级误差（相对 A：FP32 LN(x) · W + b）

- A：`LayerNorm(x_fp32)` → FP32 `W`+bias。
- B：selector INT8 量化/反量化 → LayerNorm → FP32 `W`+bias。
- C：在 B 基础上把 LayerNorm 输出经现有 Q8_0 量化/反量化 → FP32 `W`+bias。
- D：在 C 基础上真实 `W` 也按 Q8_0 量化，INT32 分块累加（Q8Q8 等价）再 +bias。

### m192_k48_seed20260928（seed=20260928, M=192, N=3072, K=1024）

| 对象 | max_abs | RMSE | relative_L2 | cosine |
|---|---|---|---|---|
| B vs A | 0.0480076 | 0.00580303 | 0.00572778 | 0.99998360 |
| C vs A | 0.0591552 | 0.00712943 | 0.00703698 | 0.99997525 |
| D vs A | 0.0598283 | 0.00797435 | 0.00787095 | 0.99996903 |

诊断分量：

| 对象 | max_abs | RMSE | relative_L2 | cosine |
|---|---|---|---|---|
| selector 输入量化 | 0.0206575 | 0.00855993 | 0.0101731 | 0.99994826 |
| LayerNorm 后 | 0.0386539 | 0.00690548 | 0.00978714 | 0.99995210 |
| 激活 Q8_0 | 0.0276882 | 0.0050943 | 0.00722063 | 0.99997393 |
| 权重 Q8_0 | 0.00205231 | 0.000157556 | 0.00623745 | 0.99998055 |
| INT32 模拟 vs 反量化 FP32 | 2.38419e-06 | 2.10164e-07 | 2.07465e-07 | 1.00000000 |

### m192_k48_seed20260929（seed=20260929, M=192, N=3072, K=1024）

| 对象 | max_abs | RMSE | relative_L2 | cosine |
|---|---|---|---|---|
| B vs A | 0.0442096 | 0.00609518 | 0.00604536 | 0.99998173 |
| C vs A | 0.0537496 | 0.00734633 | 0.00728629 | 0.99997346 |
| D vs A | 0.0585415 | 0.008152 | 0.00808537 | 0.99996731 |

诊断分量：

| 对象 | max_abs | RMSE | relative_L2 | cosine |
|---|---|---|---|---|
| selector 输入量化 | 0.0310233 | 0.00895203 | 0.0105719 | 0.99994412 |
| LayerNorm 后 | 0.0397134 | 0.00719965 | 0.0102664 | 0.99994730 |
| 激活 Q8_0 | 0.025315 | 0.00506354 | 0.00722079 | 0.99997393 |
| 权重 Q8_0 | 0.00205231 | 0.000157556 | 0.00623745 | 0.99998055 |
| INT32 模拟 vs 反量化 FP32 | 2.38419e-06 | 2.08394e-07 | 2.06699e-07 | 1.00000000 |

### m192_k48_seed20260930（seed=20260930, M=192, N=3072, K=1024）

| 对象 | max_abs | RMSE | relative_L2 | cosine |
|---|---|---|---|---|
| B vs A | 0.0508291 | 0.00647258 | 0.00640544 | 0.99997949 |
| C vs A | 0.0583228 | 0.00765742 | 0.00757799 | 0.99997129 |
| D vs A | 0.0683191 | 0.00843811 | 0.00835059 | 0.99996513 |

诊断分量：

| 对象 | max_abs | RMSE | relative_L2 | cosine |
|---|---|---|---|---|
| selector 输入量化 | 0.0331802 | 0.00957396 | 0.0113334 | 0.99993578 |
| LayerNorm 后 | 0.0424907 | 0.00770018 | 0.0109547 | 0.99994000 |
| 激活 Q8_0 | 0.0256731 | 0.00508528 | 0.00723499 | 0.99997383 |
| 权重 Q8_0 | 0.00205231 | 0.000157556 | 0.00623745 | 0.99998055 |
| INT32 模拟 vs 反量化 FP32 | 2.6226e-06 | 2.08533e-07 | 2.06368e-07 | 1.00000000 |

### m8_k2_seed20260928（seed=20260928, M=8, N=3072, K=1024）

| 对象 | max_abs | RMSE | relative_L2 | cosine |
|---|---|---|---|---|
| B vs A | 0.0323121 | 0.00523577 | 0.00513883 | 0.99998680 |
| C vs A | 0.0423176 | 0.00659633 | 0.0064742 | 0.99997906 |
| D vs A | 0.0486687 | 0.00750427 | 0.00736533 | 0.99997289 |

诊断分量：

| 对象 | max_abs | RMSE | relative_L2 | cosine |
|---|---|---|---|---|
| selector 输入量化 | 0.0127582 | 0.00735356 | 0.00879077 | 0.99996136 |
| LayerNorm 后 | 0.0292263 | 0.00601561 | 0.00856194 | 0.99996335 |
| 激活 Q8_0 | 0.0200251 | 0.00512071 | 0.0072887 | 0.99997344 |
| 权重 Q8_0 | 0.00205231 | 0.000157556 | 0.00623745 | 0.99998055 |
| INT32 模拟 vs 反量化 FP32 | 1.66893e-06 | 2.036e-07 | 1.99867e-07 | 1.00000000 |

## 5. fused QKV 输出布局

`[M,3072]` 按 Q=`[:,0:1024]` | K=`[:,1024:2048]` | V=`[:,2048:3072]` 切分（与 `reshape(seq,3,num_heads,head_dim)` 一致）。

## 6. 自洽性检查（只判实现，不判任务精度）

| 检查 | 通过 | 说明 |
|---|---|---|
| q8_0_generalized_matches_existing_reference | True | width 96 (multiple of 32) delegated to existing export_real_tensor_case.q8_quantize |
| blocked_fp32_matches_direct_small_case | True | x[4,64] W[8,64] block_k=32 |
| int32_accumulation_matches_dequantized_fp32_small_case | True | x[4,64] W[8,64], same Q8_0 payloads |
| fused_qkv_layout_matches_three_segment_gemms | True | [M,3072] split at 1024/2048 equals three GEMMs on W[0:1024]/[1024:2048]/[2048:3072] |
| selector_minus128_vs_q8_0_minus127 | True | selector keeps -128; Q8_0 clamps the same value to -127 |

- 确定性：case `m192_k48_seed20260928` 重复两次，字段逐字节一致 = `True`，x sha256 `b812dc7f1b67139e…`。

## 7. 资源与耗时

- 权重读取（文件 IO + BF16→FP32，**不计入** kernel 链计时）：0.0200914 s。
- 权重 Q8_0 量化（一次性预处理，不计入每 case kernel 链计时）：0.059266 s。
- CPU count = 20；BLAS 线程环境 = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}。

| case_id | kernel_total_s | A matmul s | B matmul s | C matmul s | D int32 s | peak_rss_kb |
|---|---|---|---|---|---|---|
| m192_k48_seed20260928 | 0.262524 | 0.0102422 | 0.0100974 | 0.0100742 | 0.218616 | 173348 |
| m192_k48_seed20260929 | 0.262545 | 0.0100287 | 0.00975271 | 0.0097981 | 0.219567 | 188588 |
| m192_k48_seed20260930 | 0.26502 | 0.0100969 | 0.0098925 | 0.00987008 | 0.221567 | 188588 |
| m8_k2_seed20260928 | 0.0171423 | 0.00180785 | 0.001559 | 0.00144192 | 0.0102801 | 188592 |

## 8. 限制与下一步

- `x_fp32` 是确定性合成的 block0 hidden，不是真实采集激活。
- 纯 CPU 软件模拟：无 GPU、无 ZCU102 板卡、无综合/bitstream/cosim。
- 数值链只到 `+qkv.bias`；block1 后续 attention/MLP 路径完全缺失。
- 不建模也不声明吞吐/时延加速；表中 wall-clock 仅是本机 CPU 串行计时。
- selector 全局 scale 由 `s = 128/amax(x)` 推导，不是标定后的生产 scale。
- 结论只判接口/数值实现自洽，不判下游任务精度。

**精确下一步（未执行）**：仅在本 CPU 链被接受后：再决定是否在 HLS 中实现 int8→fp32/LayerNorm-Q8_0 适配器与显式 bias-add 算子；没有独立授权不得综合、不得建 BD/驱动、不得上板。任务精度仍取决于完整下游路径。
