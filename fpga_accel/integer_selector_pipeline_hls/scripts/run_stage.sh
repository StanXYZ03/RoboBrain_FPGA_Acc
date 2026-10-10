#!/usr/bin/env bash
# Run one controlled Vitis HLS stage for the fused integer Selector pipeline.
#
#   scripts/run_stage.sh csim     -> csim_design only (host C testbench)
#   scripts/run_stage.sh csynth   -> csynth_design only (reports archived)
#
# Forbidden here: C/RTL cosimulation, export_design, Vivado, bitstream, board,
# GPU, network.  The HLS project lives in /tmp; only small reports/logs are
# copied into the repository artifact directory.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
mode="${1:-csynth}"
vitis_hls_bin="${VITIS_HLS_BIN:-/home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls}"
artifact_dir="${ARTIFACT_DIR:-$project_root/../../artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls}"

case "$mode" in
    csim)
        tcl="scripts/run_csim.tcl"
        project_dir="${ISP_HLS_PROJECT_DIR:-/tmp/robobrain_integer_selector_pipeline_hls_csim}"
        ;;
    csynth)
        tcl="scripts/run_csynth.tcl"
        project_dir="${ISP_HLS_PROJECT_DIR:-/tmp/robobrain_integer_selector_pipeline_hls_csynth}"
        ;;
    *)
        printf 'unsupported mode: %s\n' "$mode" >&2
        exit 2
        ;;
esac

test -x "$vitis_hls_bin" || { printf 'missing Vitis HLS: %s\n' "$vitis_hls_bin" >&2; exit 3; }

mkdir -p "$artifact_dir/logs" "$artifact_dir/csynth"
cd "$project_root"

# SuSE libtinfo.so.5 needed by the 2023.2 Tcl shell on modern hosts.
export LD_LIBRARY_PATH="/home/xilinx/xic/lib/lnx64.o/SuSE:${LD_LIBRARY_PATH:-}"

if [ "$mode" = "csim" ]; then
    export ISP_FIXTURE_DIR="${ISP_FIXTURE_DIR:-$artifact_dir/fixtures}"
    log_path="$artifact_dir/logs/04_hls_csim.log"
    test -d "$ISP_FIXTURE_DIR" || { printf 'missing fixture dir: %s\n' "$ISP_FIXTURE_DIR" >&2; exit 4; }
    export ISP_HLS_PROJECT_DIR="$project_dir"
    "$vitis_hls_bin" -f "$tcl" 2>&1 | tee "$log_path"
else
    log_path="$artifact_dir/csynth/console.log"
    export ISP_HLS_PROJECT_DIR="$project_dir"
    "$vitis_hls_bin" -f "$tcl" 2>&1 | tee "$log_path"

    report_dir="$project_dir/solution1/syn/report"
    if [ -d "$report_dir" ]; then
        for report in "$report_dir"/*.rpt "$report_dir"/*.xml; do
            [ -e "$report" ] && cp -f "$report" "$artifact_dir/csynth/"
        done
    fi
    printf 'archived csynth reports -> %s\n' "$artifact_dir/csynth"
fi
