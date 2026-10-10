# 整数 Selector 紧凑 IP 有界 RTL C/RTL 协同仿真（Vitis HLS 2023.2 / XSim）

- 日期：2026-10-03
- 决策：`INTEGER_SELECTOR_COMPACT_RTL_COSIM_PASS`
- 范围：对已验收顶层 `integer_selector_pipeline_compact` 运行 5 个冻结代表用例的 C/RTL 协同仿真（Verilog RTL、XSim、`trace_level none`）；只读引用已验收源码/头文件与 fixture。
- 未做：GPU、联网、训练、调参、IP 导出、Vivado implementation、bitstream、板卡；未改已验收源码/头文件/pragmas/报告/fixture。

## 1. 五个代表用例（对已验收 manifest 逐行解析）

| slot | case | G | grid | K |
|---|---|---|---|---|
| production 非 identity G80/K48 | `synth_g80_K48_cap16` | 80 | 8x10 | 48 |
| real correct-prompt G260/K64 | `real_grounding_correct_K64_cap16` | 260 | 13x20 | 64 |
| production stress G400/K96 | `synth_g400_K96_cap16` | 400 | 20x20 | 96 |
| identity G80/K96 | `synth_g80_K96_cap16` | 80 | 8x10 | 96 |
| 确定性 all-zero G128/K48 | `zero_g128_K48_cap16` | 128 | 8x16 | 48 |

用例解析门：`all_ok=true`（逐行对已验收 `manifest.txt` 校验、SHA 一致）。

## 2. Host 聚焦编译/测试（cosim 之前）

- g++ -O2：cases=5 checks=51 failed=0 scalar_mismatches=0 arena_mismatches=0 result=['PASS']
- ASan+UBSan：cases=5 checks=51 failed=0 scalar_mismatches=0 arena_mismatches=0 result=['PASS']

## 3. C/RTL 协同仿真

- 工具：Vitis HLS 2023.2；RTL 语言：verilog；模拟器：xsim
- part：xczu9eg-ffvb1156-2-e；clock：5.00；trace：`none`
- `C/RTL co-simulation finished: PASS` = True；XSim 启动 = True
- 墙钟：596 s（vitis_hls exit=0）
- cosim latency（aggregate over 5 transactions）：min=266036.0 avg=557566.0 max=1062296.0 cycles；total=2787614.0 cycles；status=Pass
- cosim 内 C TB post-check summaries：[{"cases": 5, "expected_cases": 5, "checks": 51, "passed": 51, "failed": 0, "scalar_mismatches": 0, "arena_mismatches": 0}, {"cases": 5, "expected_cases": 5, "checks": 51, "passed": 51, "failed": 0, "scalar_mismatches": 0, "arena_mismatches": 0}] results=['PASS', 'PASS']

### 3.1 逐 transaction 延迟（`result.transaction.rpt`）

| tx | case | G | grid | K | latency (cycles) | @200MHz (ms) |
|---|---|---|---|---|---|---|
| 0 | `synth_g80_K48_cap16` | 80 | 8x10 | 48 | 338225.0 | 1.691125 |
| 1 | `real_grounding_correct_K64_cap16` | 260 | 13x20 | 64 | 737081.0 | 3.685405 |
| 2 | `synth_g400_K96_cap16` | 400 | 20x20 | 96 | 1062296.0 | 5.31148 |
| 3 | `synth_g80_K96_cap16` | 80 | 8x10 | 96 | 266036.0 | 1.33018 |
| 4 | `zero_g128_K48_cap16` | 128 | 8x16 | 48 | 384196.0 | 1.92098 |

## 4. 生成 RTL 接口

- m_axi bundles：['input_arena', 'output_arena', 'wv_q']，data widths：{'input_arena': ['128'], 'wv_q': ['128'], 'output_arena': ['128']}
- s_axi bundles：['control']；control_r present：False
- score ports：[]

## 5. 日志告警/错误与波形/存储

- warnings=191（{"HLS 207-4706": 9, "HLS 214-398": 6, "XFORM 203-561": 8, "HLS 200-960": 9, "SYN 201-103": 12, "HLS 200-885": 5, "HLS 200-878": 2, "RTGEN 206-101": 139, "XSIM 43-3431": 1}）
- errors=0（{}）；fatals=0
- waveform files=0；trace_level=`none`
- artifact total：17 files / 1284950 B（1.225 MiB）
- 20 largest files：
  - `artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/protected_sha_after.json` — 364184 B
  - `artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/protected_sha_before.json` — 364184 B
  - `artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/logs/03_cosim.log` — 287250 B
  - `artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/cosim/syn_report/integer_selector_pipeline_compact_csynth.rpt` — 122961 B
  - `artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/cosim/sim_report/integer_selector_pipeline_compact.log` — 60839 B
  - `artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/cosim/syn_report/integer_selector_pipeline_compact_csynth.xml` — 51575 B
  - `artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/cases/rtl_cosim_cases_resolved.json` — 12624 B
  - `artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/logs/00_case_resolution.log` — 12624 B
  - `artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/cosim/sim_report/cosim_options.xml` — 2548 B
  - `artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/logs/01_host_test.log` — 1713 B
  - `artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/logs/02_host_test_asan_ubsan.log` — 1713 B
  - `artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/cosim/sim_report/integer_selector_pipeline_compact_cosim.rpt` — 1255 B
  - `artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/protected_sha_verify.json` — 547 B
  - `artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/cosim/sim_report/result.transaction.rpt` — 318 B
  - `artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/cosim/archive_index.txt` — 258 B
  - `artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/cosim/sim_report/lat.rpt` — 189 B
  - `artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/cosim/tmp_project_audit.txt` — 168 B

## 6. 受保护文件 SHA 审计

- before files=2229 aggregate=2478d5e7ea068772d4830686e1cd8534cc212f4e48f76d8998b7c732ab4aa948
- verify unchanged=True changed=0 missing=0 added=0

## 7. 门与限制

- protected_files_unchanged: True
- host_test_five_cases_zero_mismatch: True
- host_asan_ubsan_zero_mismatch: True
- cosim_finished_pass: True
- cosim_c_tb_five_cases_zero_mismatch: True
- cosim_zero_errors: True
- cosim_zero_fatals: True
- interface_exactly_three_m_axi: True
- interface_all_m_axi_128bit: True
- interface_exactly_one_axilite_control: True
- interface_no_control_r: True
- interface_no_external_score_port: True
- trace_level_none: True
- zero_waveform_files: True
- artifacts_bounded: True
- xsim_timeout_flag_absent: True

- 限制：Functional C/RTL co-simulation only: it proves RTL transaction behaviour against the frozen expected arenas, not routed timing, bitstream, board latency, model end-to-end speed or task accuracy.
- 限制：XSim cosim is pre-implementation; the accepted csynth estimate (3.650 ns <= 5.0 ns) is a pre-route estimate, not a post-route timing sign-off.
- 限制：The five cases are a bounded representative subset of the 23 accepted compact fixtures; they do not exhaust the fixture space.
- 限制：Vitis reports per-transaction latency in result.transaction.rpt and the aggregate min/avg/max in the cosim report; the measured cycle counts are pre-implementation RTL numbers, not post-route/board latency.
- 限制：Input/output canaries are checked in the co-simulation C phase and one word past the declared m_axi port depth; memory smaller than the declared cosim port depth is not independently modelled.

## 8. 推荐下一步

Export the accepted IP (export_design) and run ZCU102 out-of-context synthesis/implementation only after this RTL co-simulation gate passes; route timing is still required before any bitstream/board claim.

