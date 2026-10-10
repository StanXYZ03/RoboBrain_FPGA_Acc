# 固定点 Selector：溢出边界、C++ 仿真与受控 csynth

> **2026-09-27 v2 后续**：本文件的“预算无编译期上界 / local_cap 上限 64 /
> 31 fixtures / 49 checks / Python 65”是 v2 重构前的记录。v2 已固化
> `TS_MAX_BUDGET=96`、`local_cap<=16`、候选池 `<=144`，把 csynth 顶层 latency
> 变有限，并新增 150 组随机对拍（Python 66/66，C++ 205/205）。v2 的门控结论是
> `SELECTOR_V2_REFACTOR_EQUIVALENT / CSYNTH_LATENCY_AND_TIMING_GATE_FAIL /
> REDESIGN_REQUIRED`，见
> `reports/token_selector/2026-09-27-selector-csynth-v2.md` 与
> `artifacts/token_selector/2026-09-27-selector-csynth-v2/summary.json`。
> 下文保留为 v1 历史记录。

- 日期：2026-09-27
- 状态：`FIXEDPOINT_SELECTOR_CSIM_PASS / OVERFLOW_BOUND_PROVEN / CSYNTH_FAIL_REDESIGN_REQUIRED`
- 工作区：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`
- 范围：位精确固定点选择核 + `int64` 溢出证明 + g++ C 仿真对拍 + 一次只做
  `csynth_design` 的受控 Vitis HLS。不运行 GPU、不访问 ZCU102、不做 cosim/
  IP 导出/bitstream，不改动模型或选择算法语义。

## 1. 一句话结论

新增的定点 golden 与 g++ C 仿真在 31 个冻结 fixtures、49 项检查上逐字节一致
（普通 g++ 与 ASan/UBSan 均 `passed=49 failed=0`），Python 回归 65/65 通过；
加权质心已在 `MAX_GRID_DIM=128` 下证明 `int64` 无溢出并移除了全部 `__int128`。
受控 csynth 可完成、资源在 ZCU102 内，但顶层 latency 为 `?`（不有限）且有 II
与 timing 违反，按客观门槛状态为 `FAIL_REDESIGN_REQUIRED`，**未写 PASS**。

## 2. 冻结接口

| 项 | 定义 |
|---|---|
| `tokens_q` | `int8[G, 4, 1024]`，完整 2x2 merge group |
| `score_q` | `uint16` Q0.15 组合分，`[0, 32767]` |
| `scale` | 全图共享对称 scale；**仅透传/寄存器语义**，不参与选择 |
| grid | `1 <= grid_h, grid_w <= 128`，`grid_h*grid_w = G`，`G <= 4096`；单维 >128 返回 `ERR_GRID` |
| `K` | 16 的正倍数；生产 `48/64/96`；`G <= K` identity |
| region | 固定 3x3 |
| `local_cap` | 16 |

输出 5 段 `tokens_int8 → group_index_uint16 → source_type_uint8 →
region_id_uint8 → valid_uint8`，每段 64 B 对齐，按 `K` 计算，`valid` 恒为 1。

## 3. 选择数学与整数规则

- 每个非空区域 1 个 anchor：`score_q` 最高，同分取更小 `original_group_index`。
- `size>=2` 的区域预留至少 1 个未选 group 并生成 1 个 context；`C` 动态且 `<=9`。
- `real = K - C`；详细组来自两级 Top-K：区内有界候选（anchor 除外，context
  区域再减 1 个名额），再在候选池上取全局最高分。
- 输出代表 index 全局唯一、按升序排列；同分低 index。
- context 池化：该区域全部未选 group，权重为原始 `score_q`（非负），全 0 则
  均匀；`int64` 累加；除法用 **round-half-away-from-zero**。
- context 代表：用整数交叉乘法比较到加权质心平方距离，同距取低 index。
- identity：`G <= K` 直接逐字节复制，`C=0`，`source_type` 全 real。

与既有浮点 selector 的同构性由测试
`test_selection_structure_matches_float_selector` 覆盖。

## 4. `int64` 溢出边界（本轮新增）

`grid_h/grid_w` 各自限制到 `MAX_GRID_DIM = 128`（Python 与 C++ 同值），
超限直接返回 `ERR_GRID`。证明：

- 单个 3x3 区域的成员数 `n = b_row*b_col`，`b <= ceil(128/3) = 43`；
  在 `grid_h*grid_w <= 4096` 下 `(grid_h+2)(grid_w+2) <= 4420`（`32x128` 取到），
  故 `n <= floor(4420/9) = 491`。
- `W = Σ w_i <= 491*32767 = 16088597`；
  `|delta| <= W*(b-1) <= W*42 = 675721074`；
  `distance <= 2*675721074^2 = 9.13197939695426952e17 < 2^63-1`（约 10.1x 余量）。
- 穷举全部 9686 个合法 `(grid_h, grid_w)` 与 9 个区域带，最大
  `distance = 4.47756141868573384e17`（`grid 31x127`），仅占 `INT64_MAX` 的 4.85%。
- 池化累加 `|A| <= 491*32767*128 = 2.059340416e9`，`2|A|+W` 远在 `int64` 内。

证明与源码哈希：`artifacts/token_selector/2026-09-27-selector-overflow-proof.json`。
C++ 内核 `fpga_accel/token_selector_hls/src/token_selector_hls.cpp` 已删除
`__int128`/`unsigned __int128`，只用 `int64_t`/`uint64_t`。既有观测到的
Qwen3VL merged grid 最大维度为 8（`5x6/6x8/4x5/7x6/8x8`），规划中最大
LLM 视觉 token 1080 的方格维度也 <128，故 128 覆盖现有网格。

## 5. 新增文件

- `evaluation/hardware_regular_selector_fixedpoint.py`（`MAX_GRID_DIM=128`、`grid_dims_ok`）
- `evaluation/tests/test_hardware_regular_selector_fixedpoint.py`（新增边界/极端/溢出证明测试）
- `fpga_accel/token_selector_hls/include/token_selector_hls.h`（`TS_MAX_GRID_DIM`）
- `fpga_accel/token_selector_hls/src/token_selector_hls.cpp`（`int64`、inline、grid 守卫）
- `fpga_accel/token_selector_hls/tb/tb_token_selector_hls.cpp`（grid 边界检查）
- `fpga_accel/token_selector_hls/tools/generate_fixtures.py`（新增 7 个 case）
- `scripts/run_csynth.tcl`
- `artifacts/token_selector/2026-09-27-selector-overflow-proof.json`
- `artifacts/token_selector/2026-09-27-selector-csynth/`（csynth 小报告与 summary）
- `artifacts/token_selector/2026-09-27-fixedpoint-selector-spec.json`（更新）

## 6. HLS 源码可综合性静态审计

`include/*.h` 与 `src/*.cpp`（被综合部分）：

- 无 `new/delete/malloc/calloc/realloc/free/alloca` 等动态分配；
- 无递归（全部 helper 都是被 `token_selector_core` 单向调用）；
- 无 `throw/try/catch/noexcept` 异常；
- 无 `virtual/typeid/dynamic_cast`、无 STL 堆容器（`<vector>/<string>` 只在 tb）；
- 只有 `memcpy`（常量长度）与固定大小局部数组。

`tb/` 使用 `std::vector`/`std::string`，但它们只服务 g++ 仿真，不参与综合。

## 7. 受控 csynth

- 命令（所有中间文件在 `/tmp`，运行前记录了 `df`）：
  ```bash
  cd /tmp/robobrain_selector_csynth_run
  export LD_LIBRARY_PATH=/home/xilinx/xic/lib/lnx64.o/SuSE
  /home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f \
    /home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/scripts/run_csynth.tcl
  ```
- `open_project /tmp/robobrain_token_selector_hls_csynth`、`top=token_selector_top`、
  `part=xczu9eg-ffvb1156-2-e`、`clock=5.0ns`，只 `csynth_design`。
- 第 1 轮：综合“成功”但顶层被优化为空（latency 0、指针参数 unused、154 条
  dangling m_axi 警告）。根因是 `token_selector_core` 未强制内联，前端把其指针
  参数当作非顶层标量端口而删除了全部访存。修复：给 core 加
  `#pragma HLS INLINE`（`src/token_selector_hls.cpp:237`）。
- 第 2 轮：得到真实数据通路。资源 BRAM 128（7%）、DSP 75（2%）、FF 25933（4%）、
  LUT 29026（10%）、URAM 0（器件可用 1824/2520/548160/274080/0）。
- 结果：估计周期 3.762 ns（265.82 MHz），但 5.0 ns 目标减去 1.35 ns 不确定度后
  可用预算 3.65 ns，slack `-0.11 ns`；`VITIS_LOOP_344_10`（`src:343-349`
  anchor 比较）存在 II 违反，最高 II=11；顶层 latency 报告为 `?`。
- 门控：`finite_latency=false` → `FAIL_REDESIGN_REQUIRED`。
- 报告：`artifacts/token_selector/2026-09-27-selector-csynth/{csynth.rpt,csynth.xml,token_selector_top_csynth.rpt,summary.json,...}`。

## 8. Fixtures 与测试结果

- fixtures 31 个 case：原 24 个 + `1x128/K48`、`128x1/K48`、`2x128/K64`、
  `1x128/K128`（identity）、`1x129/K48`、`2x129/K48`、`129x2/K48`（后三者 `ERR_GRID`）。
- Python：`evaluation.tests.test_hardware_regular_selector_fixedpoint` 23/23；
  与既有 selector/B1 回归合并 `Ran 65 tests ... OK`。
- g++ 普通：`make test` → `TB token_selector_hls: passed=49 failed=0`。
- ASan/UBSan：`make asan_run` → `passed=49 failed=0`，无 sanitizer 报告。

命令：

```bash
cd /home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102
/home/zhaoxiangyu/conda-envs/robobrain25/bin/python -m unittest \
  evaluation.tests.test_hardware_regular_selector_fixedpoint \
  evaluation.tests.test_hardware_regular_token_selector \
  evaluation.tests.test_hardware_regular_b1_integration
make -C fpga_accel/token_selector_hls test
make -C fpga_accel/token_selector_hls asan_run
```

## 9. 已知限制

- 只做选择与整数池化，不打分、不训练、不加载任何模型/权重。
- csynth 门控失败：顶层 latency 不有限、II 与 timing 未收敛；未做 cosim、
  未生成 RTL/IP/bitstream、未访问 ZCU102。
- `int64` 证明只覆盖 `MAX_GRID_DIM=128`；`budget` 仍无编译期上界，这是顶层
  latency 不有限的主要来源。
- 不声明任何精度/加速/能效结论；`Wt/Wv` 与真实 `score_q` 不在本轮范围。

## 10. 精确下一步（未执行）

重设计（`REDESIGN_REQUIRED`）：给选择核加编译期循环上界（例如显式
`budget <= 4096` 契约 + `#pragma HLS LOOP_TRIPCOUNT` 或把 `budget` 设为模板/常量），
再重跑同一条 `csynth_design` 命令，直到顶层 latency 有限且 5.0 ns 收敛；在此之前
不写驱动、不做 cosim、不导出 IP、不上板。
