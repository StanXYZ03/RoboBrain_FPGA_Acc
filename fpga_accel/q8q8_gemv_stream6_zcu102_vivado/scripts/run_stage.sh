#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"
mode="${1:-validate}"
vivado_bin="${VIVADO_BIN:-/home/xilinx/Vivado/2023.2/bin/vivado}"

case "$mode" in
    validate|synth|impl) ;;
    *)
        printf 'unsupported mode: %s\n' "$mode" >&2
        exit 2
        ;;
esac
if [[ ! -x "$vivado_bin" ]]; then
    printf 'vivado not executable: %s\n' "$vivado_bin" >&2
    exit 3
fi
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
artifact_dir="$project_dir/artifacts/runs/${stamp}_${mode}"
mkdir -p "$artifact_dir"
cd "$project_dir"
"$vivado_bin" -mode batch -source scripts/build_system.tcl \
    -tclargs "$mode" "$artifact_dir" \
    2>&1 | tee "$artifact_dir/vivado_${mode}.log"
