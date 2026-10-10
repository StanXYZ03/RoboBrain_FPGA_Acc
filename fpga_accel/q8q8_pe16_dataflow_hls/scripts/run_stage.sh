#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"
mode="${1:-csynth}"
vitis_hls_bin="${VITIS_HLS_BIN:-/home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls}"

case "$mode" in
    csim|csynth|cosim|export_impl|all) ;;
    *)
        printf 'unsupported mode: %s\n' "$mode" >&2
        exit 2
        ;;
esac

if [[ ! -x "$vitis_hls_bin" ]]; then
    printf 'vitis_hls not executable: %s\n' "$vitis_hls_bin" >&2
    exit 3
fi

mkdir -p "$project_dir/artifacts"
cd "$project_dir"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_path="${HLS_LOG_PATH:-artifacts/${run_stamp}_vitis_hls_${mode}.log}"
if [[ -e "$log_path" ]]; then
    printf 'refusing to overwrite experiment log: %s\n' "$log_path" >&2
    exit 4
fi
printf 'HLS_LOG=%s\n' "$log_path"
"$vitis_hls_bin" -f scripts/run_hls.tcl "$mode" \
    2>&1 | tee "$log_path"
