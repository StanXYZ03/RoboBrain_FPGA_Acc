#!/usr/bin/env bash
# Controlled single-run Vitis HLS C-synthesis for one top function and keep the
# small raw reports in the repository artifact directory.
#
#   scripts/run_csynth.sh adapter   (top: selector_ln_q8_adapter)
#   scripts/run_csynth.sh bias      (top: qkv_bias_add)
#
# ONLY csynth_design is run: no cosim, no IP export, no Vivado, no bitstream,
# no board, no GPU, no network. The large project stays under /tmp.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
mode="${1:-adapter}"
vitis_hls_bin="${VITIS_HLS_BIN:-/home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls}"
artifact_root="${SQA_CSYNTH_ARTIFACT_DIR:-$project_root/../../artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth}"

case "$mode" in
    adapter) top="selector_ln_q8_adapter" ;;
    bias)    top="qkv_bias_add" ;;
    *) printf 'unsupported mode: %s\n' "$mode" >&2; exit 2 ;;
esac
test -x "$vitis_hls_bin" || { printf 'missing Vitis HLS: %s\n' "$vitis_hls_bin" >&2; exit 3; }

mode_dir="$artifact_root/$mode"
mkdir -p "$mode_dir"
run_dir="/tmp/robobrain_selector_qkv_adapter_csynth_run_$mode"
mkdir -p "$run_dir"

export SQA_HLS_CSYNTH_PROJECT_DIR="${SQA_HLS_CSYNTH_PROJECT_DIR:-/tmp/robobrain_selector_qkv_adapter_csynth_$mode}"
export LD_LIBRARY_PATH="/home/xilinx/xic/lib/lnx64.o/SuSE:${LD_LIBRARY_PATH:-}"

df -h /tmp > "$mode_dir/df_before.txt"

console="$mode_dir/console.log"
set +e
( cd "$run_dir" && "$vitis_hls_bin" -f "$script_dir/run_csynth.tcl" "$mode" ) \
    > "$console" 2>&1
hls_status=$?
set -e
printf 'HLS_STATUS=%d\n' "$hls_status" >> "$console"

solution_dir="$SQA_HLS_CSYNTH_PROJECT_DIR/solution1"
report_dir="$solution_dir/syn/report"
for name in csynth.rpt csynth.xml "${top}_csynth.rpt" csynth_design_size.rpt; do
    if [ -f "$report_dir/$name" ]; then
        cp -f "$report_dir/$name" "$mode_dir/$name"
    fi
done

if [ "$hls_status" -ne 0 ]; then
    printf 'csynth failed (exit %d); see %s\n' "$hls_status" "$console" >&2
    exit "$hls_status"
fi
printf 'csynth done: mode=%s report_dir=%s\n' "$mode" "$mode_dir"
