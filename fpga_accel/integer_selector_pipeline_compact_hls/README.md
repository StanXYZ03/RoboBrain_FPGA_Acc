# `integer_selector_pipeline_compact` - compact AXI envelope for the accepted integer Selector pipeline

This directory contains a *new* HLS wrapper around the already-accepted fused
integer Selector pipeline.  It does not change any arithmetic: the accepted
scorer and packed selector translation units are `#include`d byte-for-byte and
their cores (`is_scorer_core<false>` and `token_selector_core_packed`) are called
directly.  Only the memory envelope changes.

## Interface

* exactly **three** 128-bit `m_axi` bundles:
  * `input_arena` -- one contiguous arena:
    `tokens_int8 @ 0` (`G*4096` B), `group_mean_int8 @ G*4096` (`G*1024` B),
    `query_int8 @ G*5120` (`128` B); total `G*5120 + 128` B, every offset 64 B
    aligned with no extra padding;
  * `wv_q` -- the 128 KiB projection weight (8192 packed words), separate so it
    can stay resident in FPGA-attached memory;
  * `output_arena` -- the canonical fixed-budget 5-segment arena from
    `evaluation.hardware_regular_token_selector.fpga_transfer_layout(K)`
    (`K=48/64/96` totals `196,928 / 262,464 / 393,792` B).  Tokens start at
    offset 0; the bounded `K<=96` metadata is kept on chip and packed into the
    arena after selection; every gap and unused entry is deterministic zero
    through the runtime budget.
* exactly **one** AXI-Lite bundle `control`.  Each `m_axi` pointer also carries an
  explicit `#pragma HLS INTERFACE s_axilite port=<ptr> bundle=control` so all
  offset registers are merged into `control`.  Without it, Vitis HLS 2023.2
  silently creates a second `control_r` AXI-Lite bundle (as the accepted
  pipeline/scorer reports show).
* **no** `score_q` port: the intermediate is `uint16_t score_local[IS_MAX_GROUPS]`
  inside the top and never round-trips DDR.
* status codes, guards and the `scale_out`/`output_count`/`real_count`/
  `context_count` scalars are identical to `integer_selector_pipeline`
  (compile-time `static_assert`).

## Reuse strategy (no second mathematics)

```
#include "../../integer_selector_scorer_hls/src/integer_selector_scorer.cpp"
#include "../../token_selector_hls/src/token_selector_hls.cpp"
```

`is_scorer_core` lives in an anonymous namespace in the accepted scorer `.cpp`
(static visibility), so including the unmodified translation unit is the minimal
auditable exposure.  The accepted selector core is called directly.  The
accepted directories are never modified.

## Layout authority

The output-arena offsets/sizes/total are produced by the frozen Python contract
`evaluation.hardware_regular_token_selector.fpga_transfer_layout(budget)` /
`describe_fpga_layout(budget)`.  The C++ mirror `ispc_fpga_layout` and the input
formula `ispc_input_layout` are cross-checked against that authority (and against
explicit formulas) by the focused Python tests for `K=16/32/48/64/80/96` and
`G=9/16/50/80/128/260/400/4096`.

## Quick start

```bash
make python_test    # focused Python layout/fixture/interface tests
make test           # build the reference .so + g++ testbench and run it
make asan_run       # the same testbench under ASan + UBSan
make csim           # Vitis HLS C-simulation (csim_design only)
make csynth         # Vitis HLS C-synthesis (csynth_design only)
make report         # assemble the JSON/Markdown reports + harness result
```

The host testbench `dlopen()`s a shared object built from the accepted,
unmodified `integer_selector_pipeline` and compares the compact output arena
byte-for-byte against it, as well as against the frozen Python golden and the
exact `integer_selector_scorer` + `token_selector_core_packed` composition.

`make csynth` writes the large HLS project under
`/tmp/robobrain_integer_selector_compact_hls_csynth`; only small reports/logs are
copied into
`artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/`.

## Scope

CPU-only, read-only reuse of the accepted assets; no network, GPU, training,
parameter search, C/RTL co-simulation, IP export, implementation/PnR,
bitstream or board access.  The csynth evidence is HLS-only: it is not routed
timing, not board measurement and not a model end-to-end or task-accuracy
result.
