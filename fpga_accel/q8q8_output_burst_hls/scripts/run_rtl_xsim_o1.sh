#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
sim_root="$project_dir/build/q8q8_output_burst_prj/solution1/sim"
sim_dir="$sim_root/verilog"
post_dir="$sim_root/wrapc_pc"
artifact_dir="$project_dir/artifacts"
xelab_bin=${XELAB_BIN:-/home/xilinx/Vivado/2023.2/bin/xelab}
xsim_bin=${XSIM_BIN:-/home/xilinx/Vivado/2023.2/bin/xsim}
snapshot=q8q8_gemm_o1
xelab_opt=${XELAB_OPT:-O1}
snapshot="q8q8_gemm_${xelab_opt,,}"

if [[ ! -f "$sim_dir/q8q8_gemm.prj" || ! -x "$post_dir/cosim.pc.exe" ]]; then
    echo "Generated RTL or C post-check executable is missing." >&2
    exit 2
fi
mkdir -p "$artifact_dir"

(
    cd "$sim_dir"
    "$xelab_bin" xil_defaultlib.apatb_q8q8_gemm_top glbl \
        "--${xelab_opt}" --stats --prj q8q8_gemm.prj \
        -L smartconnect_v1_0 \
        -L axi_protocol_checker_v1_1_12 \
        -L axi_protocol_checker_v1_1_13 \
        -L axis_protocol_checker_v1_1_11 \
        -L axis_protocol_checker_v1_1_12 \
        -L xil_defaultlib -L unisims_ver -L xpm \
        -L floating_point_v7_1_16 -L floating_point_v7_0_21 \
        --lib ieee_proposed=./ieee_proposed -s "$snapshot"
) 2>&1 | tee "$artifact_dir/q8q8_output_burst_rtl_xelab_o1.log"

(
    cd "$sim_dir"
    "$xsim_bin" --noieeewarnings "$snapshot" \
        -tclbatch q8q8_gemm.tcl
) 2>&1 | tee "$artifact_dir/q8q8_output_burst_rtl_xsim.log"

(
    cd "$post_dir"
    ./cosim.pc.exe --rtl-smoke
) 2>&1 | tee "$artifact_dir/q8q8_output_burst_rtl_postcheck.log"
grep -q "TB_RESULT=PASS mode=rtl_smoke" \
    "$artifact_dir/q8q8_output_burst_rtl_postcheck.log"
echo "COSIM_RESULT=PASS path=xsim-2023.2-${xelab_opt,,}-no-wave"
