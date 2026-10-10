# integer_selector_pipeline_hls

Single Vitis HLS top that fuses the two accepted Selector kernels on chip:

```
group_mean_q + wv_q + query_q
     │
     ▼
┌─────────────────────────┐
│ is_scorer_core          │   accepted integer_selector_scorer core
│ (bit-exact integer Q16) │
└───────────┬─────────────┘
            │  score_local[G]   ← on-chip BRAM only, NEVER a port, NEVER DDR
            ▼
┌─────────────────────────┐
│ token_selector_core_    │   accepted packed v4 selector core
│ packed (anchor / budget │
│ / context pooling)      │
└───────────┬─────────────┘
            ▼
selected tokens + group_index + source_type + region_id + valid (+ counts/scale)
```

## Why this directory exists

`fpga_accel/integer_selector_scorer_hls` produces `uint16 score_q[G]` on an
`m_axi` port; `fpga_accel/token_selector_hls` consumes `score_q[G]` on another
`m_axi` port.  Chaining the two standalone kernels forces the intermediate
through DDR.  This module keeps that array inside the device.

**Nothing upstream is modified.**  The two accepted translation units are
compiled into this wrapper and their cores are called directly:

* `is_scorer_core<>` lives in an anonymous namespace of
  `../../integer_selector_scorer_hls/src/integer_selector_scorer.cpp`
  (static visibility), so the accepted `.cpp` is `#include`d here — the minimal
  auditable way to call it without editing the accepted directory.
* `token_selector_core_packed` is a normal external symbol in
  `../../token_selector_hls/src/token_selector_hls.cpp`; that file is also
  included so the whole fused datapath is one synthesis unit and HLS inlines
  both phases.

There is exactly one copy of every arithmetic rule (INT8×INT8→INT32 projection,
INT64 dot/norm2, exact integer floor-sqrt and round-half-away, independent
min-max to Q0.15, λ=0.5 combine; anchor / bounded local candidates / global
Top-K / weighted-centroid context pooling / round-half-away).  The wrapper adds
no arithmetic of its own.

## Top-level interface

```
int integer_selector_pipeline(
    const ts_word_t* tokens_q,       // [G*256] 128-bit packed int8 words
    const is_word_t* group_mean_q,   // [G*64]  128-bit packed int8 words
    const is_word_t* wv_q,           // [8192]  128-bit packed int8 words
    const is_word_t* query_q,        // [8]     128-bit packed int8 words
    ts_word_t* out_tokens,           // [budget*256] 128-bit packed words
    uint16_t* out_group_index,       // [budget]
    uint8_t*  out_source_type,       // [budget]
    uint8_t*  out_region_id,         // [budget]
    uint8_t*  out_valid,             // [budget]
    int group_count, int grid_h, int grid_w, int budget, int local_cap,
    float scale_in, float* scale_out,
    int* output_count, int* real_count, int* context_count);
```

There is **no `score_q` port**.  `score_q[G]` is `uint16_t
score_local[IS_MAX_GROUPS]` inside the top.  The 5 output segments keep the
accepted token selector's 64-byte-aligned semantic layout (see
`../token_selector_hls/quantization_contract.md` and
`evaluation/hardware_regular_token_selector.fpga_transfer_layout`, identical to
`evaluation/hardware_regular_selector_fixedpoint.fpga_transfer_layout`).

Status codes are the union of the two frozen contracts:

| code | meaning |
|---:|---|
| 0 | OK |
| 1 | NULL pointer |
| 2 | illegal group count (== scorer `ERR_GROUP_COUNT` == selector `ERR_GROUP_LIMIT`) |
| 3 | illegal grid |
| 4 | illegal budget |
| 5 | insufficient budget for non-empty regions |
| 6 | candidate pool too small |
| 7 | internal |
| 8 | illegal local_cap |

## Layout

| path | what |
|---|---|
| `include/integer_selector_pipeline.h` | top + host-only debug twin declarations |
| `src/integer_selector_pipeline.cpp` | the wrapper (includes the two accepted `.cpp`) |
| `tb/tb_integer_selector_pipeline.cpp` | g++ / HLS C-sim testbench |
| `tools/generate_fixtures.py` | fixture generator (re-uses the frozen goldens) |
| `tools/fixture_format.md` | binary fixture format |
| `tests/` | focused Python tests (ledger, golden re-derivation, interface) |
| `scripts/` | `run_csim.tcl`, `run_csynth.tcl`, `run_stage.sh`, `summarize_csynth.py` |

## Commands

```bash
cd fpga_accel/integer_selector_pipeline_hls
make python_test   # regenerate fixtures + focused Python tests
make test          # g++ testbench
make asan_run      # ASan+UBSan testbench
make csim          # Vitis HLS C-simulation (csim_design only)
make csynth        # Vitis HLS C-synthesis  (csynth_design only)

/home/zhaoxiangyu/conda-envs/robobrain25/bin/python scripts/summarize_csynth.py \
  --report-dir ../../artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth \
  --console-log ../../artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/console.log \
  --out ../../artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/summary.json
```

## Fixtures and goldens

`tools/generate_fixtures.py` re-uses the frozen goldens
`evaluation.selector_integer_scorer_audit.integer_chain` (on-chip intermediate
`score_q`) and `evaluation.hardware_regular_selector_fixedpoint.select_fixedpoint`
(final selection).  The accepted seed-20261003 Wv bundle is copied by SHA.  The
accepted validation NPZ supplies `group_mean_q`/grid/query for the real
grounding/pointing scenes; it contains no raw `tokens_q`, so the raw 4×1024
token bytes come from the documented deterministic xorshift64\* stream (the
scorer-side golden is token-independent and the selector-side golden is computed
over exactly those bytes).

## Transfer audit

`tools/transfer_audit.py` reports the FPGA → Jetson output with **two explicit
counters** per `(G, K)`:

* `logical_payload_bytes` — the raw valid bytes of the `output_count` returned
  tokens plus the four metadata arrays, *without* any padding;
* `fixed_budget_arena_bytes` — the authoritative, budget-sized 5-segment buffer,
  produced **live** by
  `evaluation.hardware_regular_token_selector.fpga_transfer_layout(K)`
  (every segment starts at its own 64-byte-aligned offset; the 5 offsets/sizes
  are saved verbatim in `fixed_budget_arena.segments`).

Every reported 1 GbE / local-DDR FPGA → Jetson time is based on
`fixed_budget_arena_bytes`. A scatter-gather time over `logical_payload_bytes`
is reported only as an explicit **lower bound** and never replaces the fixed
arena basis. For the G=80/K=96 identity bypass, `output_count=80` (logical
payload) and the runtime fixed-budget K=96 arena are reported separately.

Correction history (2026-10-03 Codex review): the first revision summed tokens +
metadata and applied a single trailing `align_up`; that is not the frozen
per-segment contract and underestimated K=48/K=96. Those old transfer
totals/times are void. The HLS functional/timing/resource evidence and the
`INTEGER_SELECTOR_PIPELINE_HLS_PASS` decision are unchanged.

## Scope

This round runs host `g++`, ASan+UBSan, HLS C-simulation and `csynth_design`
only.  No C/RTL cosimulation, no IP export, no Vivado implementation, no
bitstream, no board, no GPU, no network, no training, no parameter search.  All
timing/resource numbers from synthesis are **estimates**, not board
measurements.
