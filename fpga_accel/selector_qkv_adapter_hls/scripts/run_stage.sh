#!/usr/bin/env bash
# Run a single Vitis HLS C-simulation for one top function and keep the raw log.
#
#   scripts/run_stage.sh csim_adapter   (top: selector_ln_q8_adapter)
#   scripts/run_stage.sh csim_bias      (top: qkv_bias_add)
#
# Forbidden here: csynth, cosim, export_design, Vivado, board, GPU, network.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"
mode="${1:-csim_adapter}"
vitis_hls_bin="${VITIS_HLS_BIN:-/home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls}"
artifact_dir="${SQA_ARTIFACT_DIR:-$project_dir/../../artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim}"

case "$mode" in
    csim_adapter|csim_bias) ;;
    *) printf 'unsupported mode: %s\n' "$mode" >&2; exit 2 ;;
esac
test -x "$vitis_hls_bin" || { printf 'missing Vitis HLS: %s\n' "$vitis_hls_bin" >&2; exit 3; }

mkdir -p "$artifact_dir"
cd "$project_dir"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_path="${HLS_LOG_PATH:-$artifact_dir/${run_stamp}_vitis_hls_${mode}.log}"
test ! -e "$log_path" || { printf 'refusing to overwrite %s\n' "$log_path" >&2; exit 4; }
printf 'HLS_LOG=%s\n' "$log_path"

# Frozen real-block1 golden fixtures feed the byte-exact comparison.
export SQA_ADAPTER_FIXTURE="${SQA_ADAPTER_FIXTURE:-$artifact_dir/real_block1_adapter_fixture.bin}"
export SQA_BIAS_FIXTURE="${SQA_BIAS_FIXTURE:-$artifact_dir/real_block1_bias_fixture.bin}"
export SQA_HLS_PROJECT_DIR="${SQA_HLS_PROJECT_DIR:-/tmp/robobrain_selector_qkv_adapter_hls_prj}"

# Repo-reproducible Ubuntu 24.04 host workaround (also used by the selector
# cosim scripts): Vitis HLS 2023.2 Tcl needs the SuSE libtinfo.so.5, which the
# modern host no longer ships. The Xilinx xic SuSE dir provides it.
export LD_LIBRARY_PATH="/home/xilinx/xic/lib/lnx64.o/SuSE:${LD_LIBRARY_PATH:-}"

"$vitis_hls_bin" -f scripts/run_hls.tcl "$mode" 2>&1 | tee "$log_path"
