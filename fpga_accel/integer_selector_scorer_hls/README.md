# integer_selector_scorer_hls — 独立位精确整数 Selector 打分核（HLS）

本目录是**独立**的 HLS 顶层 `integer_selector_scorer`：它只做打分，把已接受的
seed-20261003 per-tensor INT8 `Wv` 与 INT8 `group_mean_q` / INT8 `query_q`
按冻结整数链算成 uint16 Q0.15 `score_q`。位精确 golden 是
`evaluation/selector_integer_scorer_audit.py` 的 `integer_chain`（预注册审计
`INTEGER_SCORER_NUMERICS_PASS`）。本目录**不复制**第二套数学：Python fixture
生成器直接调用该 golden。

本核**不做选择/合并**（选择与 anchor 合并仍由 `fpga_accel/token_selector_hls`
负责），不训练、不加载模型、不跑 GPU、不上板。

## 1. 顶层接口

```
int integer_selector_scorer(
    const is_word_t* group_mean_q,   // [G*64]  128-bit words, G in 1..4096
    const is_word_t* wv_q,           // [8192]  128-bit words (128x1024 int8)
    const is_word_t* query_q,        // [8]     128-bit words (128 int8)
    uint16_t*        score_q,        // [G]     uint16 Q0.15 in [0,32767]
    int              group_count);   // scalar, legal 1..4096
```

| 端口 | 方向 | 打包 | AXI |
|---|---|---|---|
| `group_mean_q` | in | `[G,1024]` int8 行主序，16 lane/128-bit word，小端 | m_axi `gmem_gm` |
| `wv_q` | in | `[128,1024]` int8 行主序，同打包；每次调用缓存到片上 | m_axi `gmem_wv` |
| `query_q` | in | `[128]` int8，8 个 128-bit word | m_axi `gmem_query` |
| `score_q` | out | `[G]` uint16 | m_axi `gmem_score` |
| `group_count` | in | 标量 | s_axilite `control` |
| `return` | out | 状态码 | s_axilite `control` |

- 固定维度：`embed_dim = 1024`、`query_dim = 128`、`Wv` 输出行 128；仅
  `group_count` 为运行时标量。
- `group_count` 非法（<1 或 >4096）→ `IS_STATUS_ERR_GROUP_COUNT`；任一指针为
  空（host C++ / AXI base address 为 0）→ `IS_STATUS_ERR_NULL`；合法 → `IS_STATUS_OK`。
- 每类数据端口使用独立 AXI bundle，便于 dma/连接；标量与返回走 AXI-Lite。
  详细寄存器见 `register_map_draft.md`。

128-bit packed word 契约与既有 packed kernel 一致：lane `L` 占 bits
`[8L+7:8L]`，地址 `row*1024 + word*16 + lane` 的字节即原始行主序 int8 字节。
g++ 下 `is_word_t = unsigned __int128`（小端 host 上位布局相同），综合下
`is_word_t = ap_uint<128>`。

## 2. 冻结整数链（DUT 内无 float/double）

1. `p[G,128] = group_mean_q @ Wv_q^T`，精确 INT32 累加（`|p| <= 16,646,144`）。
2. `dot = Σ_d p[d]*query_q[d]`（INT64），`norm2 = Σ_d p[d]^2`（非负 INT64）。
3. `semantic_q16 = round_half_away((dot*2^16)/floor_sqrt(norm2))`；`norm2==0` 时
   `0`。左移写成有符号乘法 `dot * 65536`，绝不使用负值左移（无 UB）。
4. `l2_int = floor_sqrt(Σ_1024 group_mean_q^2) <= 4096`。
5. `semantic_q16` 与 `l2_int` 各自精确有理 min-max 到 `[0,32767]`（全等→0）。
6. `score_q = clamp(round_half_away((semantic_norm_q15 + l2_norm_q15)/2), 0, 32767)`。

`floor_sqrt` 是逐位整数算法（32 次固定迭代）；`round_half_away` 用 64 位无符号
长除法（64 次迭代，`(2|num|+den)/(2den)`）后还原符号。所有中间位宽见
`quantization_contract.md`。

## 3. 片上结构（两趟 min-max / 输出）

| 数组 | 规模 | 用途 |
|---|---|---|
| `wv_local[128][1024]` int8 | 128 KiB，dim1 完全分区（128 bank） | 每次调用缓存一次 Wv，全部 group 复用 |
| `group_row[1024]` int8 | 1 KiB，cyclic-16 | 当前 group 的 packed 行 |
| `semantic_arr[4096]` int32 | 16 KiB | 第 1 趟写，第 2 趟 min-max/输出 |
| `l2_arr[4096]` uint16 | 8 KiB | 同上 |

投影用 128 路 INT8×INT8→INT32 并行 MAC（一个输出行一个 bank），每 group 约
1024 周期；然后 INT64 dot/norm2、floor_sqrt、round-half-away 除法各一遍，最后
第二趟独立 min-max 并输出 `score_q`。

## 4. 目录与命令

```
include/integer_selector_scorer.h   顶层契约与 packed 辅助
src/integer_selector_scorer.cpp     DUT（g++ 与 HLS 共用）
tb/tb_integer_selector_scorer.cpp   g++ / HLS C-sim 测试台
tools/generate_fixtures.py          Python fixture 生成器（复用冻结 golden）
tools/fixture_format.md             fixture 二进制格式文档
tests/test_integer_selector_scorer_hls.py  聚焦 Python 合同测试
scripts/run_csim.tcl                csim_design only
scripts/run_csynth.tcl              csynth_design only
scripts/run_stage.sh                csim / csynth 调度与日志归档
Makefile
```

```bash
cd fpga_accel/integer_selector_scorer_hls
make python_test    # 生成 fixtures + 聚焦 Python 测试
make test           # g++ 普通
make asan_run       # ASan + UBSan
make csim           # Vitis HLS C-sim（只 csim_design）
make csynth         # Vitis HLS csynth（只 csynth_design，报告归档）
```

工具链：Vitis HLS 2023.2，part `xczu9eg-ffvb1156-2-e`，`create_clock -period 5.0`，
tool 默认 uncertainty（不放大）。HLS 工程只在 `/tmp`，仓库只存小报告/日志。

## 5. 限制

- 只做打分：不写选择/anchor/输出布局，不接 token selector 的 host 布局。
- 只 `csynth_design`：不是 C/RTL cosim、不是 place/route、不是 bitstream、不是板级。
- 时延为综合估计（工具报告 + 由 `group_count` 推导），不是实测。
- 不声称端到端 selector 完成、任务精度或加速。
