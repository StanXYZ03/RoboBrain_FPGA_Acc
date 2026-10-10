# Selector v4 与 Q8Q8/PE16 DATAFLOW/PL-resident FFN 下游核接口兼容性审计（拓扑修正版）

- 日期：2026-09-28（修正记录见 §0.1）
- 决策：`SELECTOR_DOWNSTREAM_COMPAT_AUDIT_CORRECTED / FUSED_QKV_ONLY_DIRECT_PATH / DEQUANT_THEN_LAYERNORM_REQUIRED / SCALE_BROADCAST_NOT_EQUIVALENT_TO_LAYERNORM / FC1_NOT_DIRECT / DOWNSTREAM_INCOMPLETE / NO_FUSION_IMPLEMENTED`
- 状态：**AUDIT ONLY — CORRECTED**（只读取证 + 只写报告）。未使用 GPU/板卡，未运行综合/bitstream/cosim，未修改任何算法/RTL/HLS/驱动，未联网。
- 范围：修正 2026-09-28 首版兼容性审计中的真实模型拓扑错误，只审计 Selector v4 输出与仓库内**实际存在**的 `q8q8_pe16_dataflow_hls` / `pl_resident_ffn_hls` 之间的模型正确接口；给出唯一 model-correct 直接候选与最小 CPU 数值验证候选；**本轮不实施任何融合/数值链**。

## 0.1 修正记录（相对首版 2026-09-28 report 的差异）

首版报告基于“Vision Q/K/V 是三个独立 Linear、fc1 是 selector 直接下游”的错误假设，得出 `Q/K/V/fc1 全部 DIRECT + 仅需 scale broadcast adapter` 的结论。本版依据**实际 Transformers 源码 + 真实权重索引/header**修正如下：

| # | 首版错误 | 修正后事实 | 证据 |
|---|---|---|---|
| 1 | Q/K/V 是三个独立 `q_proj/k_proj/v_proj` 权重 | 实际是**一个 fused** `attn.qkv` Linear `[1024 -> 3072]`（带 bias），`attn.proj` `[1024->1024]`（带 bias）；**不存在**独立 Q/K/V 权重张量 | `modeling_qwen3_vl.py:175-176`；index/header 只有 `blocks.N.attn.qkv.{weight,bias}` 与 `blocks.N.attn.proj.{weight,bias}` |
| 2 | selector 段 0 可直接喂 qkv/fc1，只需 `x_scale=1/s` 广播 | selector 段 0 是 **block0 输出 residual hidden**；block1 的第一个算子必须先做 `norm1` LayerNorm，再 fused QKV GEMM。`1/s` 广播只是**反量化**，**不**等价于 LayerNorm | `modeling_qwen3_vl.py:267-274`、`:254-255`；`quantization_contract.md:10-11` |
| 3 | fc1 是 selector 直接边界 | fc1 输入是 `norm2(hidden_after_attn_residual)`，中间隔着完整 attention（RoPE/softmax/PV/proj/bias）+ 残差 + norm2；**不是**直接边界 | `modeling_qwen3_vl.py:267-274` |
| 4 | scale broadcast adapter 足够 | 需要 **Dequant(global s) → LayerNorm(norm1, eps=1e-6) → Q8_0 requant** 链；现有 `pl_resident_ffn` LayerNorm-Q8_0 核输入是 **4×fp32/128-bit**，不是 selector int8，故还需 int8→fp32 适配或新建 int8-LayerNorm-Q8_0 变体 | `ffn_vector.h`、`ffn_vector.cpp:233-278` |
| 5 | 下游可视为“GEMM 即可” | 即使 QKV GEMM 完成，仍需 3072 bias、reshape/split、RoPE、softmax、PV、proj+bias、residual、norm2、fc1/GELU/fc2；这些**在仓库 PL 侧不存在**，不能声称下游完整 | 源码拓扑 + §3 inventory |

## 0. 证据范围（全部实际存在，只读）

| 类别 | 文件 / 事实 | 关键点 |
|---|---|---|
| **实际 Transformers 源码** | `/home/zhaoxiangyu/conda-envs/robobrain25/lib/python3.10/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py`，版本 **4.57.0**，sha256 `dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da` | `Qwen3VLVisionAttention`、`Qwen3VLVisionBlock`、`Qwen3VLVisionMLP`、`apply_rotary_pos_emb_vision` 的真实定义 |
| 真实权重索引 | `models/Qwen3-VL-4B-Instruct/model.safetensors.index.json`，sha256 `58a7841d7bff2548dd91577d216274a83cf1b500bc6a534b809d6c1b1707cf2b` | 713 个张量、2 shards；block0/1 全部张量列出（§1.3） |
| 真实 safetensors header（只读） | `models/Qwen3-VL-4B-Instruct/model-00002-of-00002.safetensors`（3,908,490,048 B） | 只读 8+N 字节 header + 少量切片，**未加载整模型**；block0/1 权重全在该 shard |
| 模型配置 | `models/Qwen3-VL-4B-Instruct/config.json`，sha256 `edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d` | vision `hidden=1024, intermediate=4096, num_heads=16, head_dim=64, depth=24, hidden_act=gelu_pytorch_tanh` |
| Selector 量化契约 | `fpga_accel/token_selector_hls/quantization_contract.md`，sha256 `b383234b49d68b505eb3b0e05c34f78794c74a3e2542f80319278e6137aa0a53` | 全局对称 `s`：`x_real ≈ q/s`，`q=clamp(round(x_real*s),-128,127)`，zero-point 0 |
| 下游 Q8Q8 | `fpga_accel/q8q8_pe16_dataflow_hls/include/q8q8_gemm.h` sha256 `b4fa360b…c3f4`、`src/q8q8_gemm.cpp` sha256 `34b1073849aa7629be6c7143242da8421a8f242b7eb27ee234a898a8263736dc` | 纯 `X·Wᵀ`，**无 bias 端口/参数**；Q8_0 每 32 元素 scale |
| 下游 PL-resident FFN | `fpga_accel/pl_resident_ffn_hls/include/ffn_vector.h` sha256 `011d7ecd…2d034`、`src/ffn_vector.cpp` sha256 `991ddfc3a898e994854ef9b675ae894d41c3dcdb9dd7feca872e22e8e6b858ce` | 输入 `ap_uint<128>`=4×fp32；LayerNorm-Q8_0 输出 256-bit+fp16 |
| 首版审计 | `reports/token_selector/2026-09-28-selector-downstream-compatibility.{md,json}` | 被本版修正 |
| B1 接入语义 | `reports/token_selector/2026-09-27-b1-integration-audit.md` | B1 = block0 之后、block1 之前；selector 输入 = `block0_hidden.view(G,4,1024)` |

> 明确未作为证据：任何不存在的 Selector+GEMM fused IP、任何独立 Q/K/V 权重、任何“selector 段 0 直接喂 fc1”、任何未实测的 `T_fpga_downstream`。

## 1. 真实模型拓扑（Transformers 4.57.0 源码 + 真实 header）

### 1.1 `Qwen3VLVisionAttention`：fused QKV（不是 3 个独立投影）

```python
# modeling_qwen3_vl.py:168-180
class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config):
        self.dim = config.hidden_size            # 1024
        self.num_heads = config.num_heads        # 16
        self.head_dim = self.dim // self.num_heads  # 64
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)  # [1024 -> 3072] + bias
        self.proj = nn.Linear(self.dim, self.dim)                # [1024 -> 1024] + bias (default True)
        self.scaling = self.head_dim ** -0.5
```

- **`attn.qkv`**：`Linear(1024, 3072, bias=True)` → 权重 `[3072,1024]`、bias `[3072]`。源码 `forward` 中
  `self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1)`（`:191-193`）说明 3072 维布局为
  **Q(0..1023) | K(1024..2047) | V(2048..3071)**（按 `(3, num_heads, head_dim)` 展开），每段 16 heads × 64。
- **`attn.proj`**：`Linear(1024,1024)`，默认 `bias=True`；header 中确有 `attn.proj.bias [1024]`。
- **不存在** `q_proj`/`k_proj`/`v_proj` 三个独立权重张量；仓库 index/header 中只有 `attn.qkv` 与 `attn.proj`。

### 1.2 `Qwen3VLVisionBlock`：norm1→attn→residual；norm2→mlp→residual

```python
# modeling_qwen3_vl.py:251-275
class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, config, ...):
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)  # 1024
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)  # 1024
        self.attn  = Qwen3VLVisionAttention(config)
        self.mlp   = Qwen3VLVisionMLP(config)
    def forward(self, hidden_states, ...):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), ...)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states
```

- `Qwen3VLVisionMLP`（`:46-56`）：`linear_fc1 = Linear(1024, 4096, bias=True)`、`linear_fc2 = Linear(4096,1024,bias=True)`、`act_fn=ACT2FN["gelu_pytorch_tanh"]`（config）。
- **B1 语义**：selector 输入是 block0 输出 residual hidden；selector 段 0 的输出是压紧后的 hidden，下一步是 **block1 的 `self.attn(self.norm1(hidden))`**。
- 因此正确顺序是：`selector_int8 → dequant(global s) → LayerNorm(hidden, norm1.weight, norm1.bias, eps=1e-6) → Q8_0 requant → fused QKV GEMM → +qkv.bias`。**selector 段 0 不能直接喂 qkv，更不能直接喂 fc1。**

### 1.3 block0 / block1 真实张量（shape / dtype / shard / header data_offsets）

全部位于 `model-00002-of-00002.safetensors`（只读 header 取证，未加载整模型）。

| 张量名 | shape | dtype | shard | data_offsets |
|---|---|---|---|---|
| `model.visual.blocks.0.attn.qkv.weight` | [3072, 1024] | BF16 | model-00002-of-00002.safetensors | 3079842304..3086133760 |
| `model.visual.blocks.0.attn.qkv.bias` | [3072] | BF16 | model-00002-of-00002.safetensors | 3079836160..3079842304 |
| `model.visual.blocks.0.attn.proj.weight` | [1024, 1024] | BF16 | model-00002-of-00002.safetensors | 3077739008..3079836160 |
| `model.visual.blocks.0.attn.proj.bias` | [1024] | BF16 | model-00002-of-00002.safetensors | 3077736960..3077739008 |
| `model.visual.blocks.0.norm1.weight` | [1024] | BF16 | model-00002-of-00002.safetensors | 3102923264..3102925312 |
| `model.visual.blocks.0.norm1.bias` | [1024] | BF16 | model-00002-of-00002.safetensors | 3102921216..3102923264 |
| `model.visual.blocks.0.norm2.weight` | [1024] | BF16 | model-00002-of-00002.safetensors | 3102927360..3102929408 |
| `model.visual.blocks.0.norm2.bias` | [1024] | BF16 | model-00002-of-00002.safetensors | 3102925312..3102927360 |
| `model.visual.blocks.0.mlp.linear_fc1.weight` | [4096, 1024] | BF16 | model-00002-of-00002.safetensors | 3086141952..3094530560 |
| `model.visual.blocks.0.mlp.linear_fc1.bias` | [4096] | BF16 | model-00002-of-00002.safetensors | 3086133760..3086141952 |
| `model.visual.blocks.0.mlp.linear_fc2.weight` | [1024, 4096] | BF16 | model-00002-of-00002.safetensors | 3094532608..3102921216 |
| `model.visual.blocks.0.mlp.linear_fc2.bias` | [1024] | BF16 | model-00002-of-00002.safetensors | 3094530560..3094532608 |
| `model.visual.blocks.1.attn.qkv.weight` | [3072, 1024] | BF16 | model-00002-of-00002.safetensors | 3105034752..3111326208 |
| `model.visual.blocks.1.attn.qkv.bias` | [3072] | BF16 | model-00002-of-00002.safetensors | 3105028608..3105034752 |
| `model.visual.blocks.1.attn.proj.weight` | [1024, 1024] | BF16 | model-00002-of-00002.safetensors | 3102931456..3105028608 |
| `model.visual.blocks.1.attn.proj.bias` | [1024] | BF16 | model-00002-of-00002.safetensors | 3102929408..3102931456 |
| `model.visual.blocks.1.norm1.weight` | [1024] | BF16 | model-00002-of-00002.safetensors | 3128115712..3128117760 |
| `model.visual.blocks.1.norm1.bias` | [1024] | BF16 | model-00002-of-00002.safetensors | 3128113664..3128115712 |
| `model.visual.blocks.1.norm2.weight` | [1024] | BF16 | model-00002-of-00002.safetensors | 3128119808..3128121856 |
| `model.visual.blocks.1.norm2.bias` | [1024] | BF16 | model-00002-of-00002.safetensors | 3128117760..3128119808 |
| `model.visual.blocks.1.mlp.linear_fc1.weight` | [4096, 1024] | BF16 | model-00002-of-00002.safetensors | 3111334400..3119723008 |
| `model.visual.blocks.1.mlp.linear_fc1.bias` | [4096] | BF16 | model-00002-of-00002.safetensors | 3111326208..3111334400 |
| `model.visual.blocks.1.mlp.linear_fc2.weight` | [1024, 4096] | BF16 | model-00002-of-00002.safetensors | 3119725056..3128113664 |
| `model.visual.blocks.1.mlp.linear_fc2.bias` | [1024] | BF16 | model-00002-of-00002.safetensors | 3119723008..3119725056 |

**切片可读性校验（只读、取证用，非数值链）**：从 header `data_offsets` 定位后读取 `blocks.1.norm1.weight`、`blocks.1.attn.qkv.weight`、`blocks.1.attn.qkv.bias`、`blocks.1.attn.proj.bias` 的前 4 个 BF16 并按 IEEE BF16 解码（`u16<<16`）：分别为 `[0.621094, 0.302734, 0.324219, 0.285156]`、`[0.065430, 0.061035, 0.000404, 0.034424]`、`[-0.057129, 0.882812, -0.996094, -0.168945]`、`[-0.087891, 0.135742, 0.077148, -0.412109]`。这只证明 header+切片读取路径与 dtype，**不构成**任何数值/误差验证。

## 2. Selector 契约与 global scale 的精确定义

`quantization_contract.md` §1：

```
x_real ≈ q / s          (反量化)
q = clamp(round(x_real * s), -128, 127)   (量化),  zero-point = 0
```

- `s > 0` 是**整个 selector 输入张量共享的一个全局对称 scale**，随 `scale_in → scale_out` 原样透传（`s_axilite`，float），不进入 5 段输出布局，也不参与任何 Top-K/池化比较。
- **首版所谓 `x_scale=1/s` 广播**：把 `1/s` 填到 Q8Q8 的 `activations_scale[M,32]`（每 32 元素同一常量）后，Q8Q8 的 `x_dequant = q * activation_scale`（`q8q8_gemm.cpp:177-186`）恰好等于 `q/s`，即 selector int8 的**反量化**。这一步在“反量化”意义上是自洽的。
- **但反量化 ≠ LayerNorm**：
  1. LayerNorm 是先对每个 1024 维向量做 `mean/var` 统计，再 `(x-mean)/sqrt(var+eps)*gamma+beta`；`1/s` 广播是逐元素的**线性缩放**，不改变向量内的相对比例、不中心化、不归一化、不引入 `norm1.weight/bias`。
  2. `s` 是针对 block0 输出 hidden 标定的；LayerNorm 后的分布与 `s` 无关。即使对 block0 hidden 反量化完全正确，也**没有**产生 block1 需要的 `norm1(hidden)`。
  3. 因此**不能**未经证明地写 “`x_scale=1/s` 广播等价于 LayerNorm”，也不能据此把 block1 QKV 当成 selector 直接下游。
- 正确链必须显式插入 LayerNorm（见 §4）。

## 3. 下游核接口审计（实际 header/source）

### 3.1 Q8Q8 PE16 DATAFLOW

- 形状门：`M<=4320`、`N<=9728`、`K<=9728`；`k%32==0`、`n%4==0`；`TM=16,TN=64,PE_N=16,OUTPUT_LANES=4`。
- 量化：GGML Q8_0，权重与激活各带 per-32 binary16 scale；`dot_q8_0_block` 为纯 int8 dot，`contribution = dot * weight_scale * activation_scale`（`q8q8_gemm.cpp:181-186`）。
- **是否含 bias：没有。** 函数签名 `q8q8_gemm(weights_q, weights_scale, activations_q, activations_scale, output, m,n,k)`（`q8q8_gemm.h:35-43`）无 bias 指针/参数，源码也从未加任何 bias 项。**fused QKV 的 `[3072]` bias 必须由外部（新）算子添加**，不能指望现有 Q8Q8。
- 输出 `ap_uint<128>` = 4×fp32（`q8q8_output_word_t`），地址 `(m0+mi)*(n/4)+n0/4`。

### 3.2 PL-resident FFN `ffn_vector`（LayerNorm+Q8_0）

- 输入 `src0` 类型 `ffnv_f32x4_t = ap_uint<128>` = **4×fp32**（`ffn_vector.h:11-12`；`ffn_vector.cpp:102-105` `lane()` 按 32-bit float 取）。**不是** selector 的 int8，也不是 Q8_0 payload。
- LayerNorm 模式（`ffn_vector.cpp:233-278`）：`mean=sum/cols`，`variance=mean_square-mean^2`，`inverse_norm=1/sqrt(variance+eps)`，`value=(x-mean)*inverse_norm*gamma[col]+beta[col]`；`gamma=param0`、`beta=param1`（fp32），`eps` 由调用方传入（可实现 1e-6）。列数需 `cols%32==0`。
- 输出：`ffnv_q8x32_t`（256-bit，32 int8 LE）+ `ap_uint<16>` binary16 scale；量化 `d=amax/127`、`q=round(x/d)` 并 clamp 到 `[-127,127]`（`ffn_vector.cpp:209-231`）。这与 Q8Q8 的 Q8_0 反量化 `x=q*d` 自洽。
- **适配缺口**：selector 段 0 是 **int8 + 一个全局 float `s`**，而此核只吃 **4×fp32/128-bit**。因此要么
  (a) 新增 **int8-global-scale → fp32 adapter**（`x=q/s`，可用每 block 常量 `1/s` 的极简 elementwise 核），把 fp32 结果再接现有 LayerNorm-Q8_0 核；要么
  (b) 新建**直接吃 int8 + global scale 的 LayerNorm-Q8_0 变体**（在统计/归一化内部先 `q/s`）。
- 现有核**不能**直接以 selector int8 为输入。

### 3.3 其它

- `q8q8_pe16_hls` 与 DATAFLOW 同构（同 header sha256）。
- PE8 族（`q8q8_gemm_hls`/`q8q8_dataflow_hls`/`q8q8_weight_stationary_hls`/`q8q8_output_burst_hls`）：同 Q8_0 block 布局，但同样**无 bias**。
- `q8q8_gemv_stream6_hls`：M=1 GEMV，对 Vision prefill **INCOMPATIBLE**（背景）。
- 仓库 PL 侧**没有** RoPE / softmax / PV / attention 的 HLS/RTL；`pl_resident_ffn` 只有 norm/激活/残差/重量化。

## 4. 修正后的兼容性矩阵（Vision 真实形状）

真实：vision hidden=1024、intermediate=4096、heads=16、head_dim=64；selector `M=4K`（K=48/64/96 → M=192/256/384）、每行 1024。

| 候选 | 真实 GEMM（Y=X·Wᵀ） | 维度门 | 数据流可行性 | 判定 |
|---|---|---|---|---|
| **Segment0 → Dequant → LayerNorm-Q8_0 → fused QKV → Bias** | `X[4K,1024]·Wqkv[3072,1024]ᵀ → [4K,3072]` | M≤4320✓ N=3072%4/%64✓ K=1024%32✓ | selector int8 → `x=q/s` → LayerNorm(norm1,eps=1e-6) → Q8_0 → Q8Q8 → +qkv.bias | **MODEL-CORRECT DIRECT CANDIDATE**（唯一） |
| Decoder-style 独立 Q/K/V | — | — | 模型**不存在**独立 Q/K/V 张量 | **NOT-A-MODEL-PATH** |
| fc1 直接从 segment0 | `[4K,1024]·W1[4096,1024]ᵀ` | 维度本身 PASS | 输入必须是 `norm2(hidden+attn(...))`，中间有完整 attention+残差+norm2 | **NOT A DIRECT BOUNDARY** |
| O proj (`attn.proj`) | `[4K,1024]·[1024,1024]ᵀ` | PASS | 输入是 attention 输出，非 segment0 | **NOT A DIRECT BOUNDARY** |
| fc2 | `[4K,4096]·[1024,4096]ᵀ` | PASS | 输入是 `gelu(fc1)` | **NOT A DIRECT BOUNDARY** |

**结论**：
- 唯一 model-correct 的 selector 直接边界是 **`Selector → Dequant(global s) → LayerNorm-Q8_0(norm1, eps=1e-6) → fused QKV GEMM(M=4K,N=3072,K=1024) → +qkv.bias`**。
- `fc1`/`O proj`/`fc2` **都不是** selector 直接边界，必须依赖当前仓库 PL 侧**不存在**的 attention/激活算子。
- QKV GEMM 完成后得到的 `[4K,3072]` 仍需 reshape/split 到 Q/K/V、RoPE（`apply_rotary_pos_emb_vision`）、`QKᵀ*scaling`、softmax、PV、`attn.proj`+bias、残差、`norm2`、`fc1`、`gelu_pytorch_tanh`、`fc2`+bias、残差。**因此不能声称 selector 下游在 PL 上已完整可跑。**

## 5. J_F_CONTINUE 零返回路径审计（修正）

- 数据面事实不变：selector 与 Q8Q8 均为 `m_axi + s_axilite`，无 `axis`；BD 无 `axi_dma`；`m_axi` master→master 不能直连 stream；无 selector BD/驱动。
- 修正点：即使采用 shared-DDR（方案 A）零返回，selector 段 0 **不能**直接作为 Q8Q8 的激活输入；中间必须新增 **int8→fp32（global scale）→LayerNorm→Q8_0** 的 PL elementwise 链（属新算子，本任务不实现），再做 QKV GEMM，再新增 **bias add**。这些都可留在 PL/共享内存内、不返回 Jetson，但**不再**是“selector 写、GEMM 直接读”的单一 buffer 关系。
- 真正“零 memory materialization”仍需方案 C（AXI4-Stream variant + 宽度/FIFO 适配），仓库不存在。
- `T_fpga_downstream` 依然未实测。

## 6. 最小可验证候选（本轮不实现，需独立授权）

**最小 CPU 数值链（替代首版“scale plane 对拍”）：**

用**真实 block1 的 `norm1`/`qkv` 切片**（从 header `data_offsets` 定位、按需流式读），或**真实完整一层流式读取**，在 CPU 上验证：

1. 取真实 `norm1.weight/bias`、`qkv.weight/bias`（BF16→FP32）与一段真实 hidden 切片；
2. 构造 selector 风格量化：用全局 `s` 得 `q=round(x*s)`；
3. 依次执行 `dequant(q,s) → LayerNorm(eps=1e-6, gamma,beta) → Q8_0 requant → Q8Q8(M=4K,N=3072,K=1024) → +bias`；
4. 以 BF16/FP32 `F.linear` 参考比较逐元素误差（max_abs / rmse），分别报告 dequant-only、+LayerNorm、+Q8_0、+Q8Q8、+bias 各级误差。

**明确缺失（不能声称下游完整）**：reshape/split Q/K/V、RoPE、`scaling`、softmax、PV、`attn.proj`+bias、attention 残差、`norm2`、`fc1`、`gelu_pytorch_tanh`、`fc2`+bias、MLP 残差，以及后续 22 层与 merger/DeepStack。CPU 链只覆盖到 `+qkv.bias`。

被排除：任何独立 Q/K/V 方案、任何“segment0 直接喂 fc1/O/fc2”、任何“`1/s` 广播即 LayerNorm”的说法、任何“下游完整”的声明。

## 7. 阻塞项与警告

### 7.1 Blocker

1. **模型拓扑阻断**：selector 段 0 是 block0 residual hidden，必须先经 `norm1` LayerNorm；缺少该步则 QKV 结果在模型语义上错误。
2. **无 int8-global-scale 输入适配**：`pl_resident_ffn` LayerNorm-Q8_0 核只吃 4×fp32/128-bit；需要 int8→fp32 adapter 或新建 int8 LayerNorm-Q8_0 变体。仓库无此件。
3. **Q8Q8 无 bias**：fused QKV 的 `[3072]` bias 无 PL 算子可加。
4. **Vision Q8_0 权重不存在**：真实 QKV/FFN 权重仅有 BF16 safetensors；无 Vision tower 的 Q8_0 产物。
5. **PL 侧无 attention/RoPE/softmax/PV**：`+qkv.bias` 之后全部缺失，下游不完整。
6. **无 selector↔下游 BD/驱动、无 AXI4-Stream 数据面**。
7. **`T_fpga_downstream` 未实测**；selector 仅有 OOC route + 3-case cosim，无整机/上板签核。

### 7.2 Warning

1. selector int8 clamp 为 `[-128,127]`（`quantization_contract.md:11`），而 `pl_resident_ffn` quantize clamp 为 `[-127,127]`（`ffn_vector.cpp:227-228`）；跨核重量化时 `-128` 的处理需统一。
2. `scale` 只透传，需 CPU/PS 侧维护 `s` 并在 dequant 时使用；`s` 本身未标定。
3. `LayerNorm` 的 `gamma=norm1.weight`、`beta=norm1.bias` 需 BF16→FP32。
4. 现有 `pl_resident_ffn` GELU/SiLU 为 65 项 0.25 步长 PWL（`vision_ffn_composition max_abs≈0.0029255`），而模型 `hidden_act=gelu_pytorch_tanh`；若未来做到 fc1 需注意近似。
5. selector metadata（`source_type` 等）不参与 GEMM，但 context 行是池化合成值。
6. 本报告全部数字为接口/维度/源码推断；**未**做端到端数值或时延实测。

## 8. 文件变更与精确下一步

变更：`reports/token_selector/2026-09-28-selector-downstream-compatibility.md`、`...json`、`CURRENT_STATUS.md`、`/tmp/robobrain_harness_result.json`。

**精确下一步（未执行，需独立授权）**：
1. CPU-only 脚本按 §6 实现 `dequant + LayerNorm + Q8_0 + Q8Q8 + bias` 数值链，对 BF16/FP32 参考报误差；只读 safetensors 切片/流式，不加载整模型、不用 GPU。
2. 该链通过前，不新建 BD、不改 HLS 接口、不写驱动、不综合、不上板、不实施融合。

## 9. 边界声明

未使用 GPU；未访问/复位 ZCU102；未运行综合/cosim/bitstream/IP 导出；未修改任何算法/RTL/HLS/驱动；未联网。所有结论仅基于上表实际存在的 Transformers 源码、权重索引、只读 safetensors header 与仓库源文件。本轮**未实现**任何数值链。
