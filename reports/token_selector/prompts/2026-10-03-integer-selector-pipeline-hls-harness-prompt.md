# Harness 执行任务：整数打分核与硬件规则选择核片上串接

你是执行工程师。Codex 已作出技术决策：下一步把两个已验收模块串成一个 FPGA 可执行的端到端 Selector 数据通路，不再做训练或参数搜索。请直接实现、测试、综合、记录证据；最终回复只保留结论、关键数字、产物路径和阻塞，避免叙述思考过程。

工作区：
`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`

权威输入（只读复用，禁止改变其数学和现有文件）：

- `fpga_accel/integer_selector_scorer_hls/`：已验收 `INTEGER_SELECTOR_SCORER_HLS_PASS`。
- `fpga_accel/token_selector_hls/`：已验收 packed v4 selector、C/RTL cosim 与 OOC route。
- `evaluation/selector_integer_scorer_audit.py`：位精确 scorer golden。
- `evaluation/hardware_regular_selector_fixedpoint.py`：selector golden。
- `reports/token_selector/2026-10-03-integer-selector-scorer-hls.{md,json}`。
- `reports/token_selector/2026-09-28-selector-{csynth-v4,cosim-v4,ooc-v4}.md`。

## 目标

新增独立目录 `fpga_accel/integer_selector_pipeline_hls/`，实现单一 HLS top，将：

`group_mean_q + Wv_q + query_q -> score_q -> anchor/固定预算选择/context pooling -> selected token + metadata`

串接起来。`score_q[G]` 必须是片上局部缓冲，不得成为 top-level `m_axi` 端口，也不得往返 DDR。输入 token 使用既有 128-bit packed INT8 布局，输出沿用 token selector 的 5 段语义与 64 B 对齐合同。允许 top 同时接收 `tokens_q` 和 `group_mean_q`，本轮不擅自改变已验收的 group-mean 量化合同。

## 实施边界

1. 优先直接链接/调用现有 scorer 与 selector core，不复制第二套数学。若 HLS 因顶层 pragma 或静态可见性不能组合，只允许做最小、可审计的新 wrapper/core 暴露；不得修改两个既有目录中的源码、测试、报告或产物。
2. 只新增 pipeline 目录、测试、fixture 生成器、报告和小型综合摘要。大型 HLS project/cache 放 `/tmp`。
3. 不加载 4B 模型、不使用 GPU、不联网、不训练、不调参、不访问或哈希 internal-test；不得读原始图片。只复用已经接受的 validation fixture/NPZ 和公开 golden。
4. 本轮只运行 host g++、ASan+UBSan、HLS C-sim 和 `csynth_design`。不运行 RTL cosim、IP export、Vivado implementation、bitstream 或板卡。
5. Vivado/Vitis HLS 2023.2，part `xczu9eg-ffvb1156-2-e`，target period 5.0 ns，保持默认 1.35 ns uncertainty；不得放宽时钟或 uncertainty。
6. 不修改模型、checkpoint、数据 manifest、既有 config、既有报告和 `CURRENT_STATUS.md` 的旧内容；完成后只在 `CURRENT_STATUS.md` 顶部追加一条事实记录。

## 必须覆盖的功能门

1. Top-level interface 没有外部 `score_q` 数据端口；源码/综合接口双重核验。
2. scorer 中间 `score_q` 与冻结 Python golden 逐字节一致。
3. pipeline 最终输出与“既有 standalone scorer + packed selector”顺序组合逐字节一致，包括 selected tokens、group index、source type、region id、valid、output/real/context count 和 scale。
4. 覆盖 budget `48/64/96`；至少覆盖 G 约 `80/260/400`，并包含 grounding/pointing 的 correct prompt fixture；若接受 fixture 的实际 G 不完全等于这些值，应如实记录，不伪造。
5. 覆盖 identity bypass、空/全等分数、同分按小 index、非法 G/grid/budget/local-cap，且状态码与既有合同一致。
6. 同一输入重复运行结果逐字节确定。
7. 普通 g++、ASan+UBSan、HLS C-sim 均为 0 mismatch、0 sanitizer/runtime error。

## 综合门与报告口径

执行受控 `csynth_design`，提取 XML/RPT 中的真实 top 数据：

- estimated period、effective period、slack、Fmax；
- BRAM/DSP/FF/LUT/URAM 绝对值和 ZCU102 占比；
- latency best/average/worst 和关键子函数 latency；
- G=80/260/400、K=48/64/96 的周期与按 200 MHz 换算时间，明确哪些是报告值、哪些是公式推导；
- HLS error 与 200-871/200-1016/II warning 数量和关键路径。

冻结工程门：estimated period <= 3.650 ns；各资源占比 < 50%；G=400/K=96 推导总周期 <= 6,000,000（30 ms @200 MHz）；功能门全过。任何一项失败都报告 `PIPELINE_HLS_NOT_ESTABLISHED`，不得弱化门槛。全部通过才报告 `INTEGER_SELECTOR_PIPELINE_HLS_PASS`。综合估计不得写成上板实测或端到端模型加速。

## 异构传输审计

报告必须单列每图字节数，至少给 G=80/260/400 与 K=48/64/96：

- Jetson -> FPGA：`tokens_q`、`group_mean_q`、`query_q`；Wv 分冷启动每次传输和 FPGA DDR/PL 常驻两种口径。
- FPGA -> Jetson：selected tokens 和全部 metadata；不要把片上 `score_q` 算入外部传输。
- 分别给 1 GbE（使用明确的有效带宽假设并做敏感性区间）及理论 6 GB/s 本地 DDR 路径的传输时间；这是模型估算，不是测量。
- 与“score_q 外部落 DDR 后再读回”的两核串接基线比较，报告片上融合节省的字节和理想时间；不得声称消除了 Jetson↔FPGA 往返。
- 明确 `group_mean_q` 是当前额外输入，评估未来从 `tokens_q` 在 PL 内重建它的潜在节省，但本轮不得改变数值合同。

## 产物

- `fpga_accel/integer_selector_pipeline_hls/{include,src,tb,scripts,tools,README.md,Makefile}`。
- `artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/`：fixtures、紧凑日志、csynth XML/RPT 和机器可读 summary；不要复制大型临时工程。
- `reports/token_selector/2026-10-03-integer-selector-pipeline-hls.md`
- `reports/token_selector/2026-10-03-integer-selector-pipeline-hls.json`
- `reports/token_selector/logs/2026-10-03-integer-selector-pipeline-hls-harness-result.json`
- 在 `CURRENT_STATUS.md` 顶部追加结果、边界和下一决策，不重写历史。

最终自检报告所有新增/修改文件、命令退出码、测试计数、SHA256、是否读取 internal-test、是否使用 GPU/网络/模型/板卡。不要打开 internal-test 来证明没有打开它。
