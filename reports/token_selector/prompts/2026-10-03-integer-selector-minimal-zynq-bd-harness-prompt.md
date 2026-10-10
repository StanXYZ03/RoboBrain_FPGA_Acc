# Harness task: minimal Zynq MPSoC block-design validation for the routed Selector IP

Work only in:
`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`

Return one concise JSON object only after completing the work. Do not narrate intermediate reasoning.

## Authority and immutability

Use the accepted IP archive and metadata from:
`artifacts/token_selector/2026-10-03-integer-selector-compact-ip-ooc-route/`

Freeze all pre-existing Selector source, tests, evaluation code, prompts, reports, logs, and artifacts, including the compact HLS, cosim, and OOC work areas/evidence. Record SHA-256 before/after. Do not modify accepted files. Add only the new paths below plus one concise top entry in `CURRENT_STATUS.md` after evidence exists.

## Objective

Create and validate a minimal Vivado 2023.2 ZCU102 block design that represents the intended PS-DDR/PL integration of `integer_selector_pipeline_compact`:

- Zynq UltraScale+ MPSoC PS is the control/DDR owner.
- PS master AXI programs the Selector's single AXI4-Lite `control` interface.
- Selector masters `input_arena`, `wv_q`, and `output_arena` are 128-bit AXI4 and connect through a 3-input SmartConnect/interconnect to one enabled PS high-performance DDR slave port.
- The three Selector masters must share DDR arbitration without being merged at the IP boundary.
- Configure a 200 MHz PL fabric clock where supported and use `proc_sys_reset` for active-low peripheral reset.
- Assign and report AXI-Lite and DDR address segments.
- Validate the block design and generate its HDL wrapper/output products.

The Selector directly reads/writes DDR through its AXI masters, so do not add a DMA merely to move its arenas. A future Ethernet/Jetson software path may have PS software place arenas in DDR and program base-address registers.

## New paths

Create only:

- `fpga_accel/integer_selector_minimal_zynq_bd/`
- `artifacts/token_selector/2026-10-03-integer-selector-minimal-zynq-bd/`
- `reports/token_selector/2026-10-03-integer-selector-minimal-zynq-bd.{md,json}`
- `reports/token_selector/logs/2026-10-03-integer-selector-minimal-zynq-bd-harness-result.json`

Keep the Vivado project in `/tmp`. Copy only Tcl, logs, generated block-design Tcl/metadata, validation reports, address maps, interface/clock/reset inventories, and bounded synthesis reports if synthesis is run. Do not copy `.cache`, `.gen`, `.runs`, complete project trees, or generated simulation data. Do not generate a bitstream and do not access the board.

## Required flow

1. Confirm the ZCU102 board part is installed; otherwise use the exact device part and record that the board preset was unavailable.
2. Unpack the accepted IP archive in `/tmp`, add it as an IP repository, refresh the catalog, and instantiate the exact VLNV `robobrain.local:hls:integer_selector_pipeline_compact:1.0`.
3. Instantiate the Zynq MPSoC PS, SmartConnect/interconnect, and reset logic.
4. Enable the required PS AXI master and one suitable 128-bit PS high-performance slave interface to DDR. Record the exact PS interface names used.
5. Connect all clocks, active-low resets, AXI interfaces, and interrupt if used; do not leave a required interface floating.
6. Assign addresses and export the exact address map, including the Selector AXI-Lite register range and DDR segments visible to each Selector master.
7. Run `validate_bd_design`, `save_bd_design`, `generate_target all`, and create the HDL wrapper.
8. Run a bounded synthesis/elaboration check if possible without bitstream; archive synthesis status/utilization/timing only when generated.
9. Export `write_bd_tcl` so the design is reproducible.

## Gates

Decision may be `INTEGER_SELECTOR_MINIMAL_ZYNQ_BD_PASS` only if all are true:

- protected files are unchanged;
- exact part is `xczu9eg-ffvb1156-2-e` and ZCU102 board preset/part status is recorded;
- exact accepted Selector VLNV is instantiated;
- Selector interface inventory remains 3x 128-bit AXI4 masters plus 1x AXI4-Lite slave, no `control_r`, no score port;
- all three Selector AXI masters reach a PS DDR high-performance slave through a validated 3-to-1 interconnect path;
- PS master reaches Selector `control` and an AXI-Lite base address is assigned;
- all required AXI clocks/resets are connected consistently; intended PL clock is 200 MHz or an explicit, evidenced tool limitation is reported;
- all three DDR address segments are assigned and non-empty;
- `validate_bd_design` succeeds with no ERROR;
- output products and HDL wrapper are generated;
- no bitstream/board access occurs.

If validation or required connectivity cannot be achieved, use `INTEGER_SELECTOR_MINIMAL_ZYNQ_BD_BLOCKED` and preserve exact diagnostics. Do not call a diagram-only or unvalidated design PASS.

## Reports

Report exact:

- Vivado/tool/license/board-part outcomes;
- cell/VLNV inventory;
- AXI connectivity graph and widths;
- PS interfaces chosen and rationale;
- clocks and frequencies, reset polarity/connectivity;
- address map for control and all three DDR master address spaces;
- validation messages and error/critical-warning counts;
- output-product/wrapper status;
- optional synthesis status/resources/timing;
- artifact storage and protected SHA audits;
- limitations: no Ethernet stack, PS software, DDR traffic contention measurement, full-system timing sign-off, bitstream, board measurement, model end-to-end latency, or task-accuracy claim;
- one evidence-driven next action.

If PASS, recommend next implementing the PS-side bare-metal/Linux register-and-DDR test harness using static JPG-derived fixture buffers, before any Jetson Ethernet integration.
