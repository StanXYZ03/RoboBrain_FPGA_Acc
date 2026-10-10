# selector_qkv_adapter_hls — Selector→LayerNorm-Q8_0 适配器 + 独立 QKV bias-add 参考核

本目录实现 2026-09-28 修正版下游兼容性审计确定的**唯一 model-correct
直接边界**的前两级 PL 算子（HLS C-sim + g++/ASan + 受控 `csynth_design`；无 cosim/route/上板）：

```
Selector 段0 int8[M,1024] (全局 s>0, x≈q/s)
        │
        ▼
selector_ln_q8_adapter   Dequant+LayerNorm(norm1, eps=1e-6)+Q8_0 requant
        │                输出与 q8q8_pe16_dataflow 的 activations 合同一致
        ▼
Q8_0 activation payload[M,32]×256-bit + binary16 scale[M,32]
        │
        ▼  （现有 q8q8_pe16_dataflow：X·Wqkvᵀ，无 bias）
fused QKV GEMM 输出 4×fp32/128-bit [M,3072]
        │
        ▼
qkv_bias_add             y = y_q8q8 + qkv.bias    ← 本目录的**独立**算子
```

本目录**只**覆盖到 `+qkv.bias`。reshape/split、RoPE、softmax、PV、`attn.proj`、
残差、norm2、fc1/GELU/fc2、blocks 2..23 **全部缺失**，因此不得声称下游完整，
也不得给出任何端到端加速。

## 1. 顶层接口

`include/selector_qkv_adapter.h`：

| 顶层 | 说明 |
|---|---|
| `selector_ln_q8_adapter` | `tokens_q`(128-bit packed int8, 每 word 16 lane 小端) + `norm_gamma/norm_beta`(128-bit=4×fp32) → `act_q`(256-bit=32×int8) + `act_scale`(binary16)；标量 `global_scale`(float)、`rows`(int) |
| `qkv_bias_add` | `y_in`/`bias`/`y_out` 均为 128-bit=4×fp32，`word = row*(cols/4)+col/4`，`lane=col%4`；标量 `rows`、`cols` |

尺寸合同：`HIDDEN=1024`、`BLOCK=32`、`BLOCKS_PER_ROW=32`、
**adapter 合法 rows 严格为 `M=192/256/384`（Selector 合法 `K=48/64/96`），
`MAX_ROWS=384`**、`MAX_COLS=9728`、`QKV_COLS=3072`、`EPSILON=1e-6`；
bias-add 保留通用 rows 上限 `BIAS_MAX_ROWS=4320`，本轮正式 qkv case 为
`M=192/256/384, N=3072`。

### 输出 Q8_0 合同（与现有核一致）

- `act_q[row*32+block]`：32 个 int8 lane 小端打包进 256-bit（`bits[8i+7:8i]`）。
- `act_scale[row*32+block]`：per-32 block 的 binary16 scale。
- `d = amax/127`（amax 为该 block 32 个值的最大绝对值）；`q8 =
  clamp(round_half_away(v/d), -127, 127)`；`scale = float_to_half_bits(d)`。
- 与 `ffn_vector.cpp::quantize_block` 的 round/clamp/half 语义**逐字节一致**。

### Selector 输入合同

- 128-bit word 内 lane `L` 占 `bits[8L+7:8L]`，即 16 个连续 int8 小端；
  `word = (row*1024+col)/16`, `lane = col%16`。
- 全局对称 `s>0`，zero-point 0，`x≈q/s`；输入 int8 允许 `-128`（Selector 合同），
  仅**输出** Q8_0 饱和到 `[-127,127]`。
- `s<=0`、NaN、±inf 返回 `SQA_ERR_SCALE`；adapter 的 `rows` 不属于
  `{192,256,384}`（含 `1/191/193/385/4320`）返回 `SQA_ERR_DIMENSION`。
- bias-add 的 `rows` 仍为通用 `1..4320`，`cols` 为 `1..9728` 且 `cols%4==0`。

### norm_gamma / norm_beta 片上缓存（存储/调度修正，语义不变）

- adapter 调用开始时把 `norm_gamma`/`norm_beta` 各 256 个 packed 128-bit word
  一次性读入片上 cache（各 4 KiB，共 8 KiB），用
  `#pragma HLS BIND_STORAGE variable=... type=ram_2p impl=bram` 明确绑定 BRAM；
  行内 pass 2 从 cache 读取，不再每行重复 m_axi 取参数。
- 仅改变参数读取的存储/调度；数学、Q8_0 payload/scale、端口布局与逐字节 golden
  完全不变。

## 2. 数学等价融合（不是 scale 广播）

对每行 `q`（int8）、`s>0`，直接链在实数域做 `x=q/s` 后的 LayerNorm：

```
mean_x = mean_q / s,      mean_q = (1/1024)Σq
var_x  = var_q / s²,      var_q  = (1/1024)Σq² - mean_q²   (population variance)
LN(x)  = (x-mean_x)/sqrt(var_x+eps) * gamma + beta
       = (q-mean_q)/sqrt(var_q + eps*s*s) * gamma + beta
```

所以适配器**在 int8 域**直接计算上式，不物化 `x`。`1/s` 广播只是反量化，
**不**中心化、**不**归一化、**不**引入 gamma/beta，绝不等于 LayerNorm。

- variance 采用 **population variance**（`E[x²]-E[x]²`，负值截 0），与
  `ffn_vector.cpp::norm_quant` / `row_statistics` 完全一致。
- `eps*s*s` 用左结合 `(eps*s)*s`；`eps*s*s==0` 且 `var_q==0`（常量行）时取
  `inverse_norm=0`，结果恰为 `beta`，与直接链 `x-mean_x==0` 一致。
- Python golden 与 C++ 内核按相同运算顺序逐操作镜像，因此 payload 和 binary16
  scale 可逐字节对拍。

## 3. 逐字节冻结 golden 与真实权重 fixture

- `tools/selector_qkv_golden.py`：bit-exact Python golden（含 repository
  `float_to_half_bits` 的精确复刻），并单独提供 direct dequant→FP32 LayerNorm
  参考用于误差对拍。
- `tools/generate_fixtures.py`：**只读**真实 shard
  `models/Qwen3-VL-4B-Instruct/model-00002-of-00002.safetensors` 的
  `blocks.1.norm1.weight/bias` 与 `blocks.1.attn.qkv.bias`（BF16→FP32），
  生成小 fixture 到 `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/`：
  - `real_block1_adapter_fixture.bin`：真实 norm1 gamma/beta + 合成 int8 输入 +
    Python golden payload/scale 字节；`rows_M=192`（合法 `M∈{192,256,384}`）；
  - `real_block1_bias_fixture.bin`：真实 qkv.bias + 合成 Q8Q8-like fp32 输出 +
    精确 fp32 和；`rows_M=384, cols_N=3072`（本轮正式 qkv 边界）。
- 不加载完整模型；记录 tensor dtype/shape/data_offsets/raw sha256/fp32 sha256、
  shard、header sha256、读后 mtime/size 不变。

## 4. 构建与运行

```bash
make -C fpga_accel/selector_qkv_adapter_hls test       # g++ 普通 + fixture 逐字节
make -C fpga_accel/selector_qkv_adapter_hls asan_run   # ASan + UBSan
make -C fpga_accel/selector_qkv_adapter_hls csim       # HLS C-sim: selector_ln_q8_adapter
make -C fpga_accel/selector_qkv_adapter_hls csim_bias  # HLS C-sim: qkv_bias_add
```
- g++/ASan 需要 `HLS_INCLUDE ?= /home/xilinx/Vitis_HLS/2023.2/include`（提供
  `ap_int.h`），构建目录在 `/tmp/robobrain_selector_qkv_adapter_build`。
- ASan/UBSan 使用 `-fno-sanitize=shift`：Vitis HLS `ap_int` 头内部存在
  left-shift-of-negative 惯用法，UBSan 会误报；本内核所有位移都作用于无符号值且
  位移量有界，因此没有丢失任何内核侧检查，ASan 与其余 UBSan 检查仍全开。
- HLS C-sim 仅调用 `csim_design`，工程在
  `/tmp/robobrain_selector_qkv_adapter_hls_prj`，**不** csynth/cosim/export/上板。
- 受控 csynth（2026-09-28 新增，**只** `csynth_design`）：

  ```bash
  bash fpga_accel/selector_qkv_adapter_hls/scripts/run_csynth.sh adapter
  bash fpga_accel/selector_qkv_adapter_hls/scripts/run_csynth.sh bias
  ```

  工程分别在 `/tmp/robobrain_selector_qkv_adapter_csynth_{adapter,bias}`，part
  `xczu9eg-ffvb1156-2-e`、`create_clock -period 5.0`、默认 uncertainty 不放宽；
  小报告复制到 `artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/`。
  结果：**bias PASS**（estimated 3.650 ns = effective budget 3.650 ns，资源全 <50%）；
  **adapter FAIL timing**（estimated 3.945 ns > 3.650 ns，slack -0.295 ns），根因是
  Q8_0 `quantize_block` 的 32 路 amax loop-carried `fcmp→select→store` 组合路径
  （3.945 ns）；最多 2 轮调度/存储 pragma 修复无法打断（见
  `reports/token_selector/2026-09-28-selector-qkv-adapter-csynth.md`）。

### 已知主机工具链 workaround（与仓库既有 cosim 脚本一致）

Ubuntu 24.04 / glibc 2.39 下 Vitis HLS 2023.2：

1. Tcl 需要 SuSE `libtinfo.so.5`，由
   `LD_LIBRARY_PATH=/home/xilinx/xic/lib/lnx64.o/SuSE` 提供（`run_stage.sh`）；
2. 自带 binutils 2.37 读不了系统 `libm.so.6` 的 `.relr.dyn`，host 链接失败；
   由 `run_hls.tcl` 用 `COMPILER_PATH=<symlink dir>`, `LIBRARY_PATH=/usr/lib/x86_64-linux-gnu`,
   `AP_GCC_PATH` 指向自带 GCC 8.3.0，并 `config_csim -ldflags "-no-pie"` 复用
   `scripts/run_cosim_v4.tcl` 的既有可复现 workaround。仅影响 host testbench，
   DUT/算法不变。

## 5. 边界

- C-sim 与 csynth 都**不是** C/RTL cosim、不是 place/route、不是 bitstream、不是板级结果。
- 已做受控 `csynth_design`：bias 过 5.0 ns 默认 uncertainty 门槛；adapter 过 0 error、
  有限 latency、资源 <50%，但 estimated period 3.945 ns > 3.650 ns，**如实判 timing FAIL**，
  未继续 cosim/route。
- 激活为**合成**，非真实采集；真实张量仅 norm1 gamma/beta 与 qkv.bias。
- 下游 attention 仍缺失；本目录不声称端到端加速，也不判任务精度。
