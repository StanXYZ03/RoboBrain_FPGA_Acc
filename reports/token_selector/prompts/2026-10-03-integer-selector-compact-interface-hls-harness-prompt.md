# Harness task: compact AXI interface for the accepted integer selector pipeline

Work only in:
`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`

Return one concise JSON object only after completing the work. Do not narrate intermediate reasoning.

## Authority and immutability

- Treat `CURRENT_STATUS.md` and the accepted 2026-10-03 integer scorer/pipeline reports as authority.
- Do not modify any existing source, test, report, artifact, fixture, prompt, or log except:
  - add one short current-status entry at the top of `CURRENT_STATUS.md` after all evidence exists;
  - create the new paths explicitly requested below.
- In particular, do not modify:
  - `fpga_accel/integer_selector_scorer_hls/`
  - `fpga_accel/token_selector_hls/`
  - `fpga_accel/integer_selector_pipeline_hls/`
  - `evaluation/`
- Record SHA-256 before/after for those four protected trees and report whether every pre-existing file is unchanged.
- Reuse the accepted scorer and packed selector implementation; do not create a second mathematical implementation and do not change quantisation, ranking, anchor, context-pooling, tie-break, rounding, saturation, or metadata semantics.

## Objective

Create `fpga_accel/integer_selector_pipeline_compact_hls/`: a new HLS wrapper around the accepted fused pipeline suitable for later ZCU102 block-design integration. Replace the current nine external `m_axi` bundles with exactly three physical 128-bit `m_axi` bundles and expose exactly one AXI-Lite control bundle:

1. `input_arena`: tokens, group means, and query in one contiguous arena.
2. `wv_q`: the 128 KiB projection weight, separate so it can remain resident in FPGA-attached memory.
3. `output_arena`: selected tokens and four metadata segments in the canonical fixed-budget arena.

The scorer-to-selector `score_q` must remain on chip and must not become a port.

## Frozen memory contracts

All offsets are byte offsets from an arena base and are 64-byte aligned.

Input arena for runtime group count `G`:

- `tokens_int8`: offset 0, size `G * 4096` bytes (`G * 256` packed words).
- `group_mean_int8`: immediately after tokens, size `G * 1024` bytes (`G * 64` packed words).
- `query_int8`: immediately after group means, size 128 bytes (8 packed words).
- Total input arena: `G * 5120 + 128` bytes.
- These offsets are already 64-byte aligned for every legal integer G. Do not invent extra padding.

`wv_q`:

- row-major INT8, 128 x 1024 = 131,072 bytes = 8,192 packed words.

Output arena:

- Its authority is the existing frozen Python contract:
  `evaluation.hardware_regular_token_selector.fpga_transfer_layout(budget)` and `describe_fpga_layout(budget)`.
- Five segment starts are aligned independently to 64 bytes:
  `tokens_int8`, `group_index_uint16`, `source_type_uint8`, `region_id_uint8`, `valid_uint8`.
- Required total bytes: K48=196,928; K64=262,464; K96=393,792.
- Token bytes occupy packed words at the start of `output_arena`. Small metadata may be held in bounded on-chip arrays and packed into the canonical output arena after selection.
- Write deterministic zero bytes to every alignment gap and to unused entries through runtime `budget`; do not leak uninitialised data.
- Identity cases (`G <= K`) still allocate/write the fixed K arena and must preserve the accepted `output_count` semantics.

## Implementation constraints

- Synthesised top name: `integer_selector_pipeline_compact`.
- Data bus type: 128-bit packed word, identical little-endian lane order to accepted modules.
- Exactly three top-level `m_axi` bundles. No hidden fourth data master, no score port, no separate metadata masters.
- Exactly one AXI-Lite bundle in the generated component interface, including addresses and scalar control/status. If Vitis creates `control_r` or another AXI-Lite bundle, the interface gate fails and the decision cannot be PASS.
- Keep all runtime guards/status codes and scalar outputs compatible with `integer_selector_pipeline`.
- Do not buffer the complete input token arena on chip. `score_q` and bounded K<=96 metadata may be local.
- Do not install dependencies, use GPU, retrain, change datasets, or run implementation/place-and-route in this task.
- Target `xczu9eg-ffvb1156-2-e`, 5.0 ns requested period, Vitis/Vivado 2023.2 already present.

## Required verification

Create focused host tests, deterministic fixtures, scripts, README, and report tooling under the new directory. At minimum verify:

1. Python layout tests compare every input/output offset and byte count against formulas and the frozen Python output-layout authority for K=16/32/48/64/80/96.
2. Host g++ test compares compact output arena byte-for-byte with outputs from the accepted `integer_selector_pipeline` for all existing valid fixture cases that can be reused, including production G/K cases G80/K48, G260/K64, G400/K96, identity cases, ties/zero scores, and real correct-prompt fixtures. Compare status, all scalar outputs, tokens, every metadata element, padding bytes, and arena tail.
3. Invalid/null/boundary guards remain compatible and cannot write outside canaries.
4. Run the focused Python suite.
5. Run host g++ tests normally and with ASan+UBSan.
6. Run Vitis HLS C simulation.
7. Run Vitis HLS C synthesis and parse the generated component/interface metadata and reports, not source pragmas alone.

## Gates

Decision may be `INTEGER_SELECTOR_COMPACT_INTERFACE_HLS_PASS` only if all are true:

- Protected pre-existing files are byte-for-byte unchanged.
- Python, host g++, ASan+UBSan, and HLS C-sim all pass with zero mismatches.
- Exactly 3 `m_axi` interfaces and exactly 1 AXI-Lite control interface are proven from generated HLS/IP metadata.
- No `score_q` external port exists.
- Output layout is byte-exact to the frozen authority, including K48/K64/K96 totals and zero padding.
- Estimated effective clock period is <=5.0 ns.
- BRAM, DSP, FF, LUT, and URAM are each <50% of XCZU9EG capacity.
- No synthesis error or timing-failure diagnostic is present. Report loop-constraint warnings honestly; do not relabel a failed loop target as achieved.

If a gate fails, diagnose and make bounded changes only inside the new directory. Do not weaken the gate. If still unresolved, emit `INTEGER_SELECTOR_COMPACT_INTERFACE_HLS_BLOCKED` with exact evidence.

## Required outputs

- `fpga_accel/integer_selector_pipeline_compact_hls/` with include/src/tb/scripts/tools/tests/README.md/Makefile as appropriate.
- `reports/token_selector/2026-10-03-integer-selector-compact-interface-hls.md`
- `reports/token_selector/2026-10-03-integer-selector-compact-interface-hls.json`
- `reports/token_selector/logs/2026-10-03-integer-selector-compact-interface-hls-harness-result.json`
- `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/` containing raw test, C-sim, C-synthesis, interface, timing, resource, latency, and layout evidence.
- A concise top entry in `CURRENT_STATUS.md` with decision, measured facts, limitations, and next action.

The Markdown/JSON report must distinguish measured HLS results from formula estimates and must include:

- exact generated port/interface list;
- canonical input/output offsets and sizes;
- test case counts and mismatch counts;
- requested/effective/estimated period;
- absolute and percentage resource use;
- reported worst-case HLS latency and formula-derived G80/K48, G260/K64, G400/K96 latency at 200 MHz;
- corrected 1 GbE transfer estimates using fixed output arena bytes and Wv-resident mode;
- explicit statement that this is HLS evidence, not RTL co-simulation, routed timing, board measurement, model end-to-end latency, or task-accuracy evidence.

Final stdout must be a compact JSON summary with decision, changed/new files, test counts, interface counts/names, timing, resources, production-case latency, transfer estimates, protected-tree check, limitations, report paths, and one recommended next action. Do not include chain-of-thought.
