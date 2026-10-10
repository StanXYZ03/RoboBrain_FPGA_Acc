# quantization / interface contract — integer_selector_scorer

冻结来源：预注册审计 `evaluation/selector_integer_scorer_audit.py`
（`INTEGER_SCORER_NUMERICS_PASS`）与
`artifacts/token_selector/selector_integer_scorer_audit/integer_width_contract.json`。
本核是 `INTEGER_SCORER_Q16` 臂的硬件实现，输入量化后**没有任何 float/double**。

## 1. 量化约定

| 张量 | dtype / 布局 | 量化 |
|---|---|---|
| `group_mean_q` | int8 `[G,1024]` 行主序 | 每图全局 A8：`s=128/amax`，`q=clamp(round_half_away(x*s),-128,127)`（复用 `quantize_activation`） |
| `wv_q` | int8 `[128,1024]` 行主序 | 已接受 seed-20261003 **per-tensor** Wv INT8，`d=max|Wv|/127`，`q=clamp(round_half_away(Wv/d),-127,127)` |
| `query_q` | int8 `[128]` | 已冻结逐向量 INT8 prompt query |
| `score_q` | uint16 `[G]` | Q0.15，范围 `[0,32767]` |

`group_mean`/`Wv` 的 scale 与 query 的 scale/L2 范数都是**正公共因子**，在余弦与
逐图 min-max 中相消，故整数链**故意省略**（与审计一致）。

舍入统一为 **round-half-away-from-zero**；`sqrt` 为**精确 floor 整数平方根**；
除法为**精确有符号有理 round-half-away**（非 float）。semantic 小数位固定
**16**，不 sweep；score 小数位 **15**；`lambda_semantic = 0.5`。

## 2. 逐算子位宽（分析上界 + 观测上界）

| 算子 | 表达式 | 位宽 | 分析上界 | 观测上界 | 判定 |
|---|---|---|---|---|---|
| visual projection | `Σ_1024 group_mean_q*Wv_q` | INT32 | 16,646,144 | 145,258 | 放得下 |
| semantic dot | `Σ_128 p*query_q` | INT64 | 272,730,423,296 | 156,390,982 | 放得下 |
| semantic norm2 | `Σ_128 p^2` | INT64 | 35,468,046,088,798,208 | 558,499,699,697 | 放得下 |
| shifted numerator | `dot*2^16` | INT64 | 17,873,661,021,126,656 | 10,249,239,396,352 | 放得下 |
| semantic_q16 | `round_half_away((dot*2^16)/floor_sqrt(norm2))` | INT32 | 201,326,592 | 20,902,468 | 放得下 |
| l2 squared sum | `Σ_1024 group_mean_q^2` | INT64 | 16,777,216 | 369,467 | 放得下 |
| l2_int | `floor_sqrt(Σ group_mean_q^2)` | UINT16 | 4096 | 607 | 放得下 |
| semantic_norm_q15 | 精确 min-max | UINT16 | 32767 | 32767 | 放得下 |
| l2_norm_q15 | 精确 min-max | UINT16 | 32767 | 32767 | 放得下 |
| score_q | `clamp(round_half_away((sn+ln)/2),0,32767)` | UINT16 | 32767 | 30581 | 放得下 |

分析上界取 `1024*128*127 = 16,646,144`（`|group_mean_q|<=128`、`|Wv_q|<=127`），
`DOT_BOUND = 128*PROJECTION_BOUND*128`，`NORM2_BOUND = 128*PROJECTION_BOUND^2`，
`SHIFTED_NUMERATOR_BOUND = DOT_BOUND<<16`，
`SEMANTIC_Q16_BOUND = 2*(128*12)*2^16 = 201,326,592`（Cauchy–Schwarz +
`sqrt/floor_sqrt < 2`），`L2_INT_BOUND = 4096 = floor(sqrt(1024*128^2))`。
`all_fit_declared_widths = true`（分析与观测均为 true）。

### 2.1 DUT 侧静态保证

- `projection[128]` 为 `int32_t`；`dot`/`norm2`/`l2_squared`/`numerator` 为
  `int64_t`；`semantic_q16` 为 `int32_t`；`l2_int`/`*_norm_q15`/`score_q` 为
  `uint16_t`。
- `dot*2^16` 用乘法而非有符号左移：`dot * (int64_t)((int64_t)1 << 16)`，避免
  negative-left-shift UB。
- `round_half_away` 分子 `2|num| + den <= 3.575e16 < 2^63-1`，无 int64 溢出；
  `floor_sqrt` 输入 `norm2 <= 3.55e16 < 2^63-1`。
- 除法用 64 次无符号移位-减法；余数在每次左移前 `< den <= 2^63`，`rem<<1`
  不会回绕 uint64。

## 3. 边界语义

- `norm2 == 0` → `semantic_q16 = 0`（不用除法）。
- 全等 `semantic_q16` 或 `l2_int` → 对应 min-max 全 0。
- `group_count == 1` → 两项都全等 → `score_q = 0`。
- 全零 `group_mean_q` → `score_q = 0`。
- `group_count` 非法（0、4097、负）→ `IS_STATUS_ERR_GROUP_COUNT`，不写输出。
- 空指针（host C++ / AXI base 为 0）→ `IS_STATUS_ERR_NULL`。
