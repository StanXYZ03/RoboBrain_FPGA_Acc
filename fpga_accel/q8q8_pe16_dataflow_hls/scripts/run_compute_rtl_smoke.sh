#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_dir="$root_dir/artifacts/compute_rtl_smoke/$run_id"
rtl="$root_dir/build/q8q8_pe16_dataflow_prj/solution1/syn/verilog/q8q8_gemm_p_anonymous_namespace_compute_cached_tile4.v"
flow_control="$root_dir/build/q8q8_pe16_dataflow_prj/solution1/syn/verilog/q8q8_gemm_flow_control_loop_pipe_sequential_init.v"
stubs="$root_dir/../q8q8_pe16_hls/tb/arithmetic_control_stubs.sv"
address_stubs="$root_dir/tb/address_control_stubs.sv"
tb="$root_dir/tb/tb_compute_control_dataflow_rtl.sv"
xvlog="${XVLOG:-/home/xilinx/Vivado/2023.2/bin/xvlog}"
xelab="${XELAB:-/home/xilinx/Vivado/2023.2/bin/xelab}"
xsim="${XSIM:-/home/xilinx/Vivado/2023.2/bin/xsim}"

for input in "$rtl" "$flow_control" "$stubs" "$address_stubs" "$tb"; do
    if [[ ! -f "$input" ]]; then
        printf 'missing RTL input: %s\n' "$input" >&2
        exit 2
    fi
done
if [[ -e "$run_dir" ]]; then
    printf 'refusing to overwrite run directory: %s\n' "$run_dir" >&2
    exit 3
fi

mkdir -p "$run_dir"
cd "$run_dir"
"$xvlog" --sv "$stubs" "$address_stubs" "$flow_control" "$rtl" "$tb" \
    2>&1 | tee compile.log
"$xelab" tb_compute_control_dataflow_rtl \
    -s tb_compute_control_dataflow_rtl_sim 2>&1 | tee elaborate.log
"$xsim" tb_compute_control_dataflow_rtl_sim -runall 2>&1 | tee simulation.log
if ! grep -q 'RTL_CONTROL_RESULT=PASS' simulation.log; then
    printf 'directed RTL PASS marker missing\n' >&2
    exit 4
fi
sha256sum "$rtl" "$flow_control" "$stubs" "$address_stubs" "$tb" \
    > input_sha256.txt
printf 'RUN_DIR=%s\n' "$run_dir"
