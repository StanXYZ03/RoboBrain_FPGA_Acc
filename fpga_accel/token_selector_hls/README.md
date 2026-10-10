# token_selector_hls — 位精确定点 Selector 选择核（C 仿真 + 受控 csynth）

本目录是固定点硬件规则化 Token Selector 的 **HLS 顶层 + g++ C testbench**。
它只做**选择与合并**，不做打分、不训练、不加载模型、不运行 GPU。位精确 golden 是
`evaluation/hardware_regular_selector_fixedpoint.py`。

2026-09-27 v3：在**不改变选择数学、不改变 host 5 段字节布局**的前提下，把
token 数据面从 8-bit 改成 **128-bit packed AXI word**：

- `tokens_q` 输入与 `out_tokens` 输出改为 `ap_uint<128>`（HLS）/ 128-bit word
  （g++）；每个 word 是 **16 个连续 int8 lane，小端序**：lane `L` 占
  bits `[8L+7:8L]`，即地址 `group*4096 + word*16 + lane` 的字节。
- 一个 group 4096 B = **256 words**；word 地址与旧 `tokens_int8` 连续段严格等价。
- metadata 端口（`group_index/source_type/region_id/valid`）与全部标量寄存器不变。
- context 池化每个 region 只流式读取其未选 group，每个 128-bit word 一次取
  16 lane，**16 路 `int8 * uint16` 并行 `int64` 累加**；round-half-away-from-zero
  与全零权重均匀不变。real group 按 256 word 原样复制。**不缓存全部 G tokens**。

2026-09-28 v4：在**不改选择数学 / host 5 段布局 / default 1.35 ns uncertainty /
packed context 路径**的前提下关闭 5 ns 时序门槛（v3 estimated 4.423 ns）：

- 删除完全分区的 `local_top[9][16]` 稳定插入 bank 与所有 16/144 路插入结构。
  每个 region 的精确 top-`local_cap` 候选改为**顺序、确定性多遍 max-select**：
  每个 rank 扫描一次 region，跳过 anchor 与已选组，按“score 高优先、tie 低 index”
  取最优，直接写入有界 `pool`；候选集合与 v3 稳定插入**完全相同**。
- 全局 details（`<=144` pool）、升序 selected（`<=96`）、entry
  （`(representative, source_type, region_id)`）都用**单比较器多遍 select** +
  标量 best 寄存器，下标只来自循环计数或标量，无数据相关读口。
- 加权质心代表点显式窄化到 int32（已证明 `|delta| <= 675721074`），两个
  32x32→64 平方用 `#pragma HLS BIND_OP ... impl=dsp latency=2` 绑定 DSP 且
  ≥2-cycle 流水寄存；平方欧氏距离与 tie 规则位精确。
- `ts_pool_context_packed` 128-bit context 路径**逐行不变**（v3 已达 3.650 ns）。
- 冻结的 v3 bounded-insertion int8 kernel/top 保留在 `#ifndef __SYNTHESIS__` 下，
  只作 g++/ASan testbench 的旧实现差分参考，**永不综合**。

v2 遗留的 8-bit `int8[G,4,1024]` core（`token_selector_core` /
`token_selector_top`）**原样保留**用于软件回归；v3 新增 packed 孪生接口
（`token_selector_core_packed` / `token_selector_top_packed`），二者逐字节对拍。

## 1. 冻结接口

| 项 | 定义 |
|---|---|
| `tokens_q` | 软件回归：`int8[G, 4, 1024]`；综合 top：`uint128[G*256]`（16 lane/word，小端） |
| `score_q` | `uint16` Q0.15 组合分，范围 `[0, 32767]` |
| `scale` | 全图共享对称 scale；**仅透传/寄存器语义**，不参与选择或池化 |
| `grid_h * grid_w = G` | `1 <= grid_h, grid_w <= MAX_GRID_DIM=128`，且 `G <= 4096`；单维 >128 返回 `ERR_GRID`。另加 `G <= 65535`（uint16 代表索引）守卫 |
| `K` | 16 的正倍数且 `K <= TS_MAX_BUDGET=96`；生产预算 `48/64/96`；边界 `16/32/48/64/80/96`；`G <= K` 时 identity |
| region | 固定 3x3 |
| `local_cap` | `[1, 16]`，生产固定 16 |
| host 布局 | 5 段不变，64 B 对齐；每段字节数与总字节数不变，token 段只是按 16 B 重新分元素 |

选择数学与 CPU 浮点 selector 完全一致：每个非空区域 1 个 anchor（最高分，
同分低 index）；`size>=2` 的区域预留至少 1 个未选 group 并生成 1 个 context；
`real = K - C`；区内有界候选后做全局 Top-K；输出代表 index 全局唯一、升序。
context 用该区域全部未选 group 以 `score_q` 非负权重池化（全 0 则均匀），
`int64` 累加、round-half-away-from-zero；代表点用整数交叉乘法比较到加权质心
的平方距离，同距取低 index。

编译期上界（对应 HLS 有限 latency）：

- `TS_MAX_BUDGET = 96`：所有 output/selected/entry 循环按此定长。
- `TS_MAX_LOCAL_CANDIDATE_CAP = 16`：区内多遍 max-select 只保留精确 `local_cap` 个。
- `TS_MAX_CANDIDATE_POOL = 9*16 = 144`：全局候选池定长，多遍 select 取前 `details_needed` 个。
- `TS_MAX_REGION_MEMBERS = 512`：单个 3x3 区域成员数的编译期上界（解析界 491）。
- `TS_MAX_GROUPS = 4096`：所有动态 G 循环的真实 `max` tripcount。
- `TS_GROUP_WORDS = 256`、`TS_MAX_WORDS = 4096*256 = 1048576`：packed token 段。

## 2. 128-bit packed word 契约

- 小端 lane：`lane = byte_offset % 16`，`word = byte_offset / 16`；`word.range(8L+7,8L)`。
- g++ 兼容类型：非综合时 `ts_word_t = unsigned __int128`，在 little-endian
  host 上与 16 个连续 int8 字节**位精确同布局**；访问器 `ts_word_lane` /
  `ts_word_set_lane` 在 HLS 与 g++ 下语义一致。
- 打包/解包：`ts_pack_int8_to_words` / `ts_unpack_words_to_int8`（C++）与
  `evaluation/token_selector_packed_contract.py`（Python）支持**任意 byte
  offset / lane**，尾部零填充；`group_word_address(g,w)` 与
  `g*4096 + w*16` 字节地址等价。
- ZCU102 HPC 现实宽度按 **128-bit** 记录：token m_axi 数据宽度 128 bit，
  一个 burst beat 取 16 个 int8；host 64 B 对齐与总字节数完全不变。

## 3. 文件

```text
include/token_selector_hls.h        常量、status、layout、packed word 类型/访问器、core/top 声明
src/token_selector_hls.cpp          packed multi-pass core/top（综合）+ v3 int8 insertion core/top（#ifndef __SYNTHESIS__ 参考）
tb/tb_token_selector_hls.cpp        g++ C testbench，int8/packed 双 top 对拍 + pack/unpack 契约
tb/tb_token_selector_cosim_v4.cpp   独立 C/RTL cosim testbench（只回放 3 个代表 case）
tools/generate_fixtures.py          用 Python golden 生成固定 + 随机 fixtures
tools/generate_cosim_v4_fixtures.py 用 Python golden 生成 3 个 cosim fixtures
Makefile                            make test / make asan_run
register_map_draft.md               AXI-Lite 寄存器草案（csynth 生成，未上板）
quantization_contract.md            INT8/Q0.15 量化契约
```

Python 侧新增 `evaluation/token_selector_packed_contract.py` 与
`evaluation/tests/test_token_selector_packed_contract.py`（pack/unpack、
任意 offset/lane、word 地址等价、word 池化 = int8 golden）。

## 4. 顶层接口

`token_selector_top_packed` 使用 `#pragma HLS INTERFACE m_axi`：输入端口
`gmem_in_tokens`（128-bit）/ `gmem_in_score`，5 个输出段各自独立 bundle
`gmem_out_tokens`（128-bit）/ `gmem_out_group` / `gmem_out_source` /
`gmem_out_region` / `gmem_out_valid`；标量走 `s_axilite bundle=control`。返回
`status`，并通过 s_axilite 写回 `output_count` / `real_count` /
`context_count` / `scale_out`。旧的 `token_selector_top` 8-bit 端口保留。

5 个输出段仍然固定为 `tokens_int8 → group_index_uint16 → source_type_uint8 →
region_id_uint8 → valid_uint8`，每段 64 B 对齐，按总预算 `K` 计算；`valid`
恒为 1。**host 5 段布局未改变**，与 CPU `fpga_transfer_layout` 逐值一致
（K=48 时总长 196928 B）；token 段只是 `K*4096 B = K*256 words`。

## 5. 运行

```bash
# 生成 fixtures（固定 34 + 随机 150 = 184 个 case）
PYTHON=/home/zhaoxiangyu/conda-envs/robobrain25/bin/python make fixtures

# g++ 普通编译 + C 仿真（int8 top 与 packed top 同时跑，均对拍 golden）
make test

# ASan + UBSan
make asan_run

# Python 回归
PYTHON -m unittest evaluation.tests.test_hardware_regular_selector_fixedpoint \
  evaluation.tests.test_hardware_regular_token_selector \
  evaluation.tests.test_hardware_regular_b1_integration \
  evaluation.tests.test_token_selector_packed_contract

# C/RTL cosim（3 个代表 case；csim + csynth + cosim，Verilog/port，独立 /tmp 工程）
mkdir -p /tmp/robobrain_selector_cosim_run_v4 && cd /tmp/robobrain_selector_cosim_run_v4
export LD_LIBRARY_PATH=/home/xilinx/xic/lib/lnx64.o/SuSE:$LD_LIBRARY_PATH
timeout 7200 /home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f \
  /home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/scripts/run_cosim_v4.tcl

# Verilog IP 导出 + Vivado OOC synth/opt/place/route（project/cache/IP zip 全在 /tmp）
bash /home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/scripts/run_selector_v4_ip_ooc.sh
```

## 6. 覆盖与当前结果

fixtures 共 184 个 case（34 固定 + 150 确定性随机）。testbench 对每个 case
同时运行 **int8 v3 insertion top** 与 **packed v4 multi-pass top**，两者与 Python
golden 在 tokens/group_index/source_type/region_id/valid 上逐字节一致，并额外做
`pack_lane_offsets` / `pack_byte_offset_tail` 两项 128-bit pack/unpack 契约检查。
v4 另加 **220 组内存内确定性随机差分**（`randdiff_v3_000..219`，30 种 grid、
预算 16/32/48/64/80/96、`local_cap` 1..16；分数按 `score_mod` 量化，较大
`score_mod` 提高 tie 频率，但这不是显式全 tie 模式）与 **48 组确定性边界差分**
（`bounddiff_v4_00..47`，显式覆盖全相等、0/65535 交替、重复档位、随机饱和值四种
score 模式，以及 32x128/64x64/128x32/31x127/127x31/16x16/24x24/32x32 网格和
K=16..96、`local_cap` 1/16），直接对比新 multi-pass packed 与旧 v3 insertion int8
kernel；输入/输出向量逐例释放；普通 g++ 与 ASan/UBSan 下均
`passed=475 failed=0`（207 + 220 + 48），无 sanitizer 报错。

> 合同澄清：正式 `score_q` 合同是 unsigned Q0.15，合法上限 **32767**；
> `bounddiff` 里的 `0/65535` 交替仅是对 `uint16` 原始载荷的鲁棒性压力测试，
> **不是合同内量化输入**（见 `quantization_contract.md` §2）。该用例保留、不删除。


- identity：`4x4/K48`、`8x8/K64`、`9x9/K96`、`2x9/K48`、`5x4/K48`、`1x96/K96`。
- 压缩：`K=48/64/96`，`9x9`、`10x12`；`2x25/K48`（3 个空区）、`2x9/K16`、`5x4/K16`（动态 `C=6/7`）。
- `MAX_GRID_DIM=128` 边界与极端长宽比：`1x128`、`128x1`、`2x128` 接受；`1x129`、`2x129`、`129x2` 返回 `ERR_GRID`。
- 预算上界：`K=112/128` 与 `local_cap=17` 返回 `ERR_BUDGET` / `ERR_LOCAL_CAP`。
- 全同分、全零分（均匀权重）。
- 5 段布局 offset/size 与 Python 一致，5 个输出 bundle 与输入 bundle 分离。
- 150 组随机 case 与旧 Python golden 逐字节对拍（`rand000`…`rand149`）。

Python 侧：`evaluation.tests.test_hardware_regular_selector_fixedpoint`、
`test_hardware_regular_token_selector`、`test_hardware_regular_b1_integration`
与 `test_token_selector_packed_contract` 合计 **76 项全部通过**；其中
`test_v4_multipass_matches_v3_insertion_200plus` 对 frozen golden 跑 **220 组**
`candidate_mode="v3_insertion"` 差分。

v4 受控 csynth（`scripts/run_csynth_v4.tcl`，`token_selector_top_packed`，
part/clock/default uncertainty 不变，只 `csynth_design`，**1 轮修复即通过**）：
0 error；**timing 通过 estimated 3.650 ns = 3.650 ns budget（slack 0.000 ns）**，
无 `HLS 200-871`/`200-1016`；latency 有限 **worst 1,708,124 cycles = 8.541 ms @200 MHz
< 33.3 ms**；资源 **BRAM 7.57% / DSP 3.45% / FF 7.56% / LUT 16.41% / URAM 0** 全部 <50%。
全部四项门槛满足，status **PASS**。这是 **pre-route 0 ns 估算裕量**，不是 post-route
200 MHz 结论。详见 `artifacts/token_selector/2026-09-28-selector-csynth-v4/summary.json`
与 `reports/token_selector/2026-09-28-selector-csynth-v4.md`；v3 真实 critical path
摘要见同目录 `v3_critical_path_summary.md`。

v4 C/RTL cosim（`scripts/run_cosim_v4.tcl`，**独立** cosim testbench
`tb/tb_token_selector_cosim_v4.cpp`，只回放 3 个固定代表 case，不是 184+220 全量）：
C simulation `0 errors`、cosim RTL Verilog + `-trace_level port`、tool xsim，
**`C/RTL co-simulation finished: PASS`**，post-check `passed=3 failed=0`，每例核对
status/count/scale/全部 token 字节/全部 metadata。3 例为：identity 4x4/K48；
16x16/K64 全 tie + context；32x32/K96 极值 + context。cosim 工程目录与 csynth
工程目录分离。cosim **只验证功能，不证明 post-route 200 MHz**。证据：
`artifacts/token_selector/2026-09-28-selector-cosim-v4/`、
`reports/token_selector/2026-09-28-selector-cosim-v4.md`。

v4 Verilog IP 导出 + **Vivado OOC synth/opt/place/route**（`scripts/export_selector_v4_ip_ooc.tcl`，
`export_design -rtl verilog -format ip_catalog -flow impl`，等价于既有
`fpga_accel/q8q8_pe16_hls/scripts/run_hls.tcl` 的成功 OOC 流程；project/cache/IP
zip 全在 `/tmp`）：part/5 ns/default uncertainty 不变，clock 未放宽。Vivado
v.2023.2 结果 **WNS +0.689 ns / WHS +0.036 ns @ 5.000 ns、unrouted 0、DRC Error 0、
LUT 9.10% / FF 5.37% / BRAM 4.61% / DSP 3.37%**，post-route 最差路径在未改动的
`ts_pool_context_packed`（DSP MAC）内；IP zip 719,284 B（sha256 `3ea8b74f…f265`）只
留 `/tmp`。**注意**：OOC XDC 只含 `create_clock -period 5.000`，因此 routed timing
不含 HLS 的 1.35 ns 调度 uncertainty；与 csynth pre-route 3.650 ns/0 ns 裕量、
cosim 功能 PASS 是三个不同结论。证据：
`artifacts/token_selector/2026-09-28-selector-ooc-v4/`、
`reports/token_selector/2026-09-28-selector-ooc-v4.md`。

## 7. 已知未做 / 限制

- 已运行 csim + `csynth_design` + C/RTL cosim（3 例，见 §6）以及 **Verilog IP 导出 +
  Vivado OOC synth/opt/place/route**（见上；0 error、WNS/WHS ≥ 0、unrouted 0、
  DRC Error 0、资源 <10%）；**未**生成 bitstream、未访问 ZCU102、未写驱动。cosim
  只证明功能等价；OOC post-route 是单 IP 包装，不是含 PS/DDR 的整机签核，也不含
  HLS 的 1.35 ns 调度 uncertainty。
- 顶层 latency、资源、timing 与门控结论见 v4 summary。
- 顶层大数组不再做完全分区；`pool/selected/entry` 等小数组由 HLS 推断为 auto RAM，
  选择为多遍 select。寄存器偏移由 csynth 生成并写入 `register_map_draft.md`
  （仍是综合估计，未做 RTL/上板确认；v4 未改端口/寄存器布局）。
- 只验证选择与整数池化的位精确性，不声明任何精度、加速或能效结论。
- `Wt/Wv` 与真实 `score_q` 的产生不在本目录范围内。
- 加权质心 `int64` 安全性覆盖 `MAX_GRID_DIM=128`；预算上界 `K<=96` 已固化，
  因此所有 output/selected/entry 循环均有编译期定长。
