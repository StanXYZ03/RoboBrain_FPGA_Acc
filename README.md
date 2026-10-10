# RoboBrain2.5 on Xilinx ZCU102

面向具身智能多模态模型 RoboBrain2.5 的 ZCU102 本地部署与异构加速工程。

## 项目目标

1. 建立可复现的 CPU/GPU 推理基线与端到端性能指标。
2. 通过量化、视觉词元裁剪和模型拆分降低内存与计算量。
3. 将高占比的矩阵乘、反量化与缓存访问映射到 ZCU102 PL。
4. 在 ARM PS 上完成预后处理、调度和机器人应用接口集成。
5. 用真实板卡数据验证吞吐、首 token 延迟、功耗和任务精度。

## 目录

```text
RoboBrain2.5-4B/   本地模型检查点（不纳入 Git）
app/               原始推理、API 与任务基准
fpga_accel/        量化、性能分析、硬件规格与 Vivado 验证
docs/              系统设计、部署路线和实验规范
scripts/           顶层环境与项目自检
Makefile           常用任务入口
```

## 快速开始

```bash
cd /home/zhaoxiangyu/RoboBrain2.5
make check
make test
make analyze
```

`make check` 不加载模型权重。`make analyze` 只读取 safetensors 元数据。完整 GPU
基准、量化实验和 Vivado 流程见 [`fpga_accel/README.md`](fpga_accel/README.md)。
Makefile 默认使用 `/home/zhaoxiangyu/conda-envs/robobrain25/bin/python`；如环境位置
不同，可通过 `make PYTHON=/path/to/python test` 覆盖。

## 推荐架构

当前应采用 PS/PL 异构方案，而不是尝试把 4B 多模态模型完整装入 PL：

- PS：tokenizer、图像预处理、算子调度、采样、机器人控制接口。
- PL：INT4/INT8 权重矩阵乘、反量化融合、RMSNorm/激活等候选热点。
- DDR：量化权重和 KV cache，采用分块与双缓冲隐藏搬运延迟。
- 主机：模型转换、量化校准、bitstream 构建和部署包生成。

所有估算结果必须与 ZCU102 实测结果分开标注。项目的主要验收指标是任务精度、
首 token 延迟、解码 tokens/s、峰值内存、板级功耗和相对基线加速比。

## 当前阶段

现有 `fpga_accel` 已具备模型结构分析、量化敏感性、Roofline 估算、规格导出、
单元测试以及最小 Vivado 流程。下一里程碑是实现可综合的矩阵乘内核和 PS 端运行时，
再将真实板卡测量接入统一 benchmark。详见 [`docs/ROADMAP.md`](docs/ROADMAP.md)。

Token Selector 的未来主实验按边缘部署运行域执行：完整 CV-Bench 2638 样本统一限制为
Patch Merger 后 `G<=576`，Dense 与 K96 Selector 使用相同输入；主质量指标为 CV-Bench
官方分数及相对 Dense 的保留率，原生高分辨率只作压力测试。General/Pointing/Grounding/
Trajectory 是 API 路由，不作为 Selector 的独立晋升门。详细规则见
[`docs/PL_ACCELERATION_RESEARCH_PLAN_2027.md`](docs/PL_ACCELERATION_RESEARCH_PLAN_2027.md)
顶部 2026-10-07 权威更新。

## Selector v4 HLS IP OOC（2026-09-28）

`fpga_accel/token_selector_hls` 的 packed v4 Token Selector
（`token_selector_top_packed`）已导出 Verilog IP 并完成 Vivado OOC
synth/opt/place/route（Vitis HLS/Vivado 2023.2、`xczu9eg-ffvb1156-2-e`、
`create_clock -period 5.000`，未放宽 clock）：0 error、**WNS +0.689 ns**、
**WHS +0.036 ns**、unrouted 0、DRC Error 0，LUT/FF/BRAM/DSP 均 <10%。
这是**单 IP OOC** 结果，不是含 PS/DDR 的整机或上板签核；IP project、cache 与
zip 均留在 `/tmp`，仓库只保存小报告与 hash。复现脚本
`scripts/run_selector_v4_ip_ooc.sh`；详见
[`reports/token_selector/2026-09-28-selector-ooc-v4.md`](reports/token_selector/2026-09-28-selector-ooc-v4.md)。

## Selector→LayerNorm-Q8_0 adapter + QKV bias-add 受控 csynth（2026-09-28）

`fpga_accel/selector_qkv_adapter_hls` 的 `selector_ln_q8_adapter` 与独立
`qkv_bias_add` 已完成受控 `csynth_design`（Vitis HLS 2023.2、`xczu9eg-ffvb1156-2-e`、
5.0 ns、默认 uncertainty 不放宽，只综合，无 cosim/route/IP/BD/驱动/板卡）。
adapter 合法 rows 收紧为严格 `M=192/256/384`（`SQA_MAX_ROWS=384`），
`norm_gamma/norm_beta` 改为调用开始一次性读入 8 KiB 片上 BRAM cache。
结果：g++/ASan+UBSan/HLS C-sim 逐字节 golden **0 mismatch**；**bias csynth PASS**
（estimated 3.650 ns = 3.650 ns，资源全 <50%）；**adapter estimated 3.945 ns >
3.650 ns，如实判 timing FAIL**（根因：Q8_0 amax 归约 `fcmp→select→store`
loop-carried 组合路径；2 轮调度/存储 pragma 无法打断），未继续 cosim/route，
不声称端到端加速。详见
[`reports/token_selector/2026-09-28-selector-qkv-adapter-csynth.md`](reports/token_selector/2026-09-28-selector-qkv-adapter-csynth.md)。

## Q8_0 只读 SD 部署

`deployment/` 提供 RoboBrain2.5-4B Q8_0 GGUF 的服务器可信门禁、Cortex-A53
交叉编译、DDR-only 传输、OpenAI 兼容文本测试以及 SD 零写入保护流程。该流程不做
量化、不写 SD；详细的执行顺序与硬失败条件见 [`deployment/README.md`](deployment/README.md)。
