# Harness 有界更正：pipeline 传输布局计数

你是执行工程师。Codex 审校已接受 `integer_selector_pipeline` 的功能、csim 与 csynth 证据，但发现传输审计对 5 段 64 B 布局计数错误。请只做下述有界更正，最终回复仅给结论、修正数字、测试和文件路径。

工作区：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`

## 已确认缺陷

`fpga_accel/integer_selector_pipeline_hls/tools/transfer_audit.py::image_bytes` 当前以
`align_up(selected_tokens + raw_metadata)` 计算 FPGA→Jetson 总量；这不等于冻结合同
`evaluation.hardware_regular_token_selector.fpga_transfer_layout(budget)`，后者要求 5 个
segment 的**每个起始 offset**分别 64 B 对齐。K=48/K=96 因而低估。

## 允许修改

- `fpga_accel/integer_selector_pipeline_hls/tools/transfer_audit.py`
- `fpga_accel/integer_selector_pipeline_hls/tests/test_integer_selector_pipeline_hls.py`
- `fpga_accel/integer_selector_pipeline_hls/tools/build_report.py`
- `fpga_accel/integer_selector_pipeline_hls/README.md`
- 本任务对应主 JSON/MD、harness-result 和 `CURRENT_STATUS.md` 顶部更正记录

禁止修改 HLS header/source/TB/TCL/Makefile、fixtures、csynth/csim 日志与 XML/RPT、两个既有核、模型/checkpoint/manifest/config/既有历史报告。

## 更正合同

1. 直接调用冻结 `fpga_transfer_layout(budget)` / `describe_fpga_layout(budget)`，不得复制第二套 offset 算法。
2. 对每个 (G,K) 同时报告：
   - `logical_payload_bytes`：实际 `output_count` 个 token + 四种 metadata 的原始有效字节，不含 padding；
   - `fixed_budget_arena_bytes`：严格等于 `fpga_transfer_layout(K).total_bytes`，并保存 5 段 offset/size；
   - 网络/DDR保守传输时间统一基于 `fixed_budget_arena_bytes`；若另给 scatter-gather payload 时间，必须明确是下界且不能替代固定 arena 口径。
3. 固定 arena 的权威值应由函数实时产生，不硬编码；预期 K=48/64/96 分别为 196928/262464/393792 B，但以运行结果为准。
4. G=80,K=96 identity bypass 的 `output_count=80` 与 arena 仍按 runtime fixed budget K=96 分开报告，避免混淆有效 payload 和固定预算缓冲区。
5. Jetson→FPGA、score_q DDR 节省、HLS 功能/时序/资源数字保持不变。
6. 主报告增加一段 Codex 审校更正：旧输出总量/时延被低估，不改 `INTEGER_SELECTOR_PIPELINE_HLS_PASS`，但旧传输数字作废，以更正后字段为准。
7. 主 JSON 的 `authority_prompt` 写入本提示路径与 SHA256，不得为 null。

## 验证

- 新增会令旧实现失败的测试：K=48/64/96 的 offsets、sizes、total 与冻结函数逐字段一致；K=48/K96 明确防回归；identity G80/K96 同时验证 payload 与 arena 两口径。
- 运行 pipeline 聚焦 Python 测试；无需重跑 g++/ASan/HLS csim/csynth。
- 校验 HLS source/header/TB/TCL、fixtures、csynth XML/RPT/log 的 SHA 在本更正前后不变，并写入更正 harness-result。
- internal test、GPU、网络、模型、训练、板卡均不得使用。

产物：更新主 `reports/token_selector/2026-10-03-integer-selector-pipeline-hls.{json,md}`；新增
`reports/token_selector/logs/2026-10-03-integer-selector-pipeline-transfer-correction-harness-result.json`
和 focused-test 日志。不要覆盖原 pipeline harness-result。
