#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"
mode="${1:-csynth}"
vitis_hls_bin="${VITIS_HLS_BIN:-/home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls}"

case "$mode" in
    csim|csynth|export_impl) ;;
    *) printf 'unsupported mode: %s\n' "$mode" >&2; exit 2 ;;
esac
test -x "$vitis_hls_bin" || { printf 'missing Vitis HLS: %s\n' "$vitis_hls_bin" >&2; exit 3; }
mkdir -p "$project_dir/artifacts"
cd "$project_dir"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_path="${HLS_LOG_PATH:-artifacts/${run_stamp}_vitis_hls_${mode}.log}"
test ! -e "$log_path" || { printf 'refusing to overwrite %s\n' "$log_path" >&2; exit 4; }
printf 'HLS_LOG=%s\n' "$log_path"
"$vitis_hls_bin" -f scripts/run_hls.tcl "$mode" 2>&1 | tee "$log_path"
