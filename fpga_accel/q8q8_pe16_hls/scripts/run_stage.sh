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
"$vitis_hls_bin" -f scripts/run_hls.tcl "$mode" \
    2>&1 | tee "artifacts/vitis_hls_${mode}.log"
