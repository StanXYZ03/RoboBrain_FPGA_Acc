#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$ROOT_DIR/artifacts/compute_rtl_smoke/$RUN_ID"
RTL="$ROOT_DIR/build/q8q8_pe16_cosim_prj/solution1/syn/verilog/q8q8_gemm_p_anonymous_namespace_compute_cached_tile.v"
FLOW_CONTROL="$ROOT_DIR/build/q8q8_pe16_cosim_prj/solution1/syn/verilog/q8q8_gemm_flow_control_loop_pipe_sequential_init.v"
STUBS="$ROOT_DIR/tb/arithmetic_control_stubs.sv"
TB="$ROOT_DIR/tb/tb_compute_control_rtl.sv"
XVLOG="${XVLOG:-/home/xilinx/Vivado/2023.2/bin/xvlog}"
XELAB="${XELAB:-/home/xilinx/Vivado/2023.2/bin/xelab}"
XSIM="${XSIM:-/home/xilinx/Vivado/2023.2/bin/xsim}"

if [[ ! -f "$RTL" ]]; then
    echo "Missing generated compute RTL: $RTL" >&2
    echo "Run ./scripts/run_cosim_small.sh until csynth_design completes." >&2
    exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing run directory: $RUN_DIR" >&2
    exit 2
fi

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"
"$XVLOG" --sv "$STUBS" "$FLOW_CONTROL" "$RTL" "$TB" 2>&1 | tee compile.log
"$XELAB" tb_compute_control_rtl -s tb_compute_control_rtl_sim 2>&1 | tee elaborate.log
"$XSIM" tb_compute_control_rtl_sim -runall 2>&1 | tee simulation.log
sha256sum "$RTL" "$FLOW_CONTROL" "$STUBS" "$TB" > input_sha256.txt
printf 'RUN_DIR=%s\n' "$RUN_DIR"
