#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
rtl_dir="$project_dir/build/q8q8_output_burst_prj/solution1/syn/verilog"
run_dir=$(mktemp -d "$project_dir/artifacts/store_rtl_xsim.XXXXXX")
xvlog_bin=${XVLOG_BIN:-/home/xilinx/Vivado/2023.2/bin/xvlog}
xelab_bin=${XELAB_BIN:-/home/xilinx/Vivado/2023.2/bin/xelab}
xsim_bin=${XSIM_BIN:-/home/xilinx/Vivado/2023.2/bin/xsim}

cd "$run_dir"
"$xvlog_bin" \
    "$rtl_dir/q8q8_gemm_mul_32ns_31ns_63_1_1.v" \
    "$rtl_dir/q8q8_gemm_mul_33s_31s_60_1_1.v" \
    "$rtl_dir/q8q8_gemm_flow_control_loop_pipe_sequential_init.v" \
    "$rtl_dir/q8q8_gemm_p_anonymous_namespace_store_output_tile_Pipeline_VITIS_LOOP_210_1_VITIS_LOOP_214_s.v" \
    "$rtl_dir/q8q8_gemm_p_anonymous_namespace_store_output_tile.v" \
    "$project_dir/tb/tb_store_output_rtl.v" \
    2>&1 | tee "$project_dir/artifacts/q8q8_output_burst_store_xvlog.log"
"$xelab_bin" work.tb_store_output_rtl --O1 \
    -s tb_store_output_rtl 2>&1 | \
    tee "$project_dir/artifacts/q8q8_output_burst_store_xelab.log"
"$xsim_bin" tb_store_output_rtl \
    -tclbatch "$project_dir/scripts/xsim_run_all.tcl" 2>&1 | \
    tee "$project_dir/artifacts/q8q8_output_burst_store_rtl_xsim.log"
grep -q "TB_RESULT=PASS test=store_output_rtl" \
    "$project_dir/artifacts/q8q8_output_burst_store_rtl_xsim.log"
echo "STORE_RTL_XSIM_RESULT=PASS"
