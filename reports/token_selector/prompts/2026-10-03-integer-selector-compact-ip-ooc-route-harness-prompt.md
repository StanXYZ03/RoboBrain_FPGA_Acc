# Harness task: export accepted compact selector IP and run ZCU102 OOC route

Work only in:
`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`

Return one concise JSON object only after completing the work. Do not narrate intermediate reasoning.

## Authority and immutability

Read the top of `CURRENT_STATUS.md` and the accepted compact-interface and RTL-cosim reports. Freeze every pre-existing file under:

- `fpga_accel/integer_selector_scorer_hls/`
- `fpga_accel/token_selector_hls/`
- `fpga_accel/integer_selector_pipeline_hls/`
- `fpga_accel/integer_selector_pipeline_compact_hls/`
- `fpga_accel/integer_selector_pipeline_compact_cosim/`
- `evaluation/`
- all existing compact-interface and compact-RTL-cosim reports, logs, prompts, and artifacts

Record SHA-256 before/after. Do not modify accepted source, pragmas, tests, fixtures, reports, artifacts, or prior evidence. Add only the new paths below plus one concise top entry in `CURRENT_STATUS.md` after evidence exists.

## Objective

Using the same accepted source and top `integer_selector_pipeline_compact`:

1. Recreate an isolated Vitis HLS 2023.2 project in `/tmp` and run `csynth_design`.
2. Export the synthesized design as an IP catalog component (`export_design -format ip_catalog`) without changing source or interface.
3. Use Vivado 2023.2 to perform out-of-context synthesis and implementation through `opt_design`, `place_design`, `phys_opt_design` when legal, and `route_design` for `xczu9eg-ffvb1156-2-e`.
4. Constrain `ap_clk` to 5.000 ns (200 MHz). Preserve the exported HLS clock/reset/control assumptions and document any generated clock uncertainty.
5. Do not generate a bitstream and do not access the board.

This phase is intended to obtain post-route timing/resource/DRC evidence. It is not a full ZCU102 block design and cannot establish board or end-to-end model performance.

## New isolated work area

Create only:

- `fpga_accel/integer_selector_pipeline_compact_ooc/`
- `artifacts/token_selector/2026-10-03-integer-selector-compact-ip-ooc-route/`
- `reports/token_selector/2026-10-03-integer-selector-compact-ip-ooc-route.{md,json}`
- `reports/token_selector/logs/2026-10-03-integer-selector-compact-ip-ooc-route-harness-result.json`

Keep all Vitis/Vivado projects in `/tmp`. Do not copy complete projects, generated RTL trees, `.cache`, `.gen`, `.runs`, simulation data, or journals into the repository.

## Required implementation evidence

Archive only bounded, useful outputs:

- HLS export log and exported `component.xml`/VLNV metadata;
- Vivado Tcl scripts and console logs;
- post-synth, post-place, and post-route timing summaries;
- post-synth and post-route hierarchical utilization reports;
- clock/interconnect/high-fanout reports useful for diagnosing timing;
- route status and DRC reports;
- QoR suggestions summary when available;
- a routed DCP only if it is <=200 MiB; otherwise store its SHA-256/size and keep it out of the repository;
- exact generated interface inventory from `component.xml` and/or the synthesized design.

Audit total artifact bytes and list the 20 largest files.

## Interface and design gates

The exported/routed design must still expose:

- exactly three 128-bit AXI4 master interfaces: `input_arena`, `wv_q`, `output_arena`;
- exactly one AXI4-Lite slave control interface: `control`;
- no `control_r` interface;
- no external `score_q` port;
- part exactly `xczu9eg-ffvb1156-2-e`;
- target clock exactly 5.000 ns.

Parse generated metadata/reports, not source pragmas alone.

## Decision gates

Decision may be `INTEGER_SELECTOR_COMPACT_IP_OOC_ROUTE_PASS` only if all are true:

- all protected files remain byte-for-byte unchanged;
- HLS csynth and IP export finish with no error/fatal;
- Vivado OOC synthesis, opt, placement, and routing finish successfully;
- post-route setup WNS >= 0.000 ns for the 5.000 ns clock;
- post-route hold WHS >= 0.000 ns;
- no unrouted nets;
- no DRC ERROR and no CRITICAL WARNING that invalidates implementation or timing;
- LUT, FF, BRAM, DSP, and URAM utilization are each below 80% of XCZU9EG capacity;
- generated interface and part/clock gates above pass;
- no bitstream was generated.

Do not weaken gates or relabel failed timing as PASS. Use these failure decisions as appropriate:

- `INTEGER_SELECTOR_COMPACT_IP_EXPORT_BLOCKED`
- `INTEGER_SELECTOR_COMPACT_OOC_SYNTH_BLOCKED`
- `INTEGER_SELECTOR_COMPACT_OOC_ROUTE_BLOCKED`
- `INTEGER_SELECTOR_COMPACT_OOC_TIMING_FAIL`

If route runs longer than 90 minutes, terminate only this task's Vivado process group, archive evidence, and report `INTEGER_SELECTOR_COMPACT_OOC_ROUTE_BLOCKED_TIMEOUT`.

## Required reports

JSON/Markdown/final stdout must include:

- exact commands, tool versions, license outcome, part and top;
- exported IP VLNV and archive size;
- generated interface names/types/data widths and gate results;
- HLS timing/resources for identity with the accepted design;
- OOC synth/place/route statuses and elapsed wall times;
- post-route WNS/TNS/WHS/THS and achieved frequency derived from the 5 ns requirement;
- post-synth and post-route absolute/percentage resources;
- route status, unrouted nets, DRC error/critical-warning counts;
- key post-route critical path startpoint/endpoint when reported;
- protected SHA audit;
- artifact storage audit;
- limitations: OOC only, no PS/DDR/AXI interconnect contention, no full block design, no bitstream, no board measurement, no end-to-end or accuracy claim;
- one recommended next action based on the actual outcome.

If OOC route passes, the recommended next action is to build the minimal Zynq MPSoC block design with PS, clocks/resets, three AXI masters, AXI-Lite control, memory map and DMA path, then validate the design before any bitstream. If timing fails, recommend the smallest evidence-driven RTL/HLS optimization based on the critical path instead.
