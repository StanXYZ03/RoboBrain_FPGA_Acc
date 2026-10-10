#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$root/artifacts/runs/${stamp}_bitstream"
mkdir -p "$run_dir"

vivado=${VIVADO:-/home/xilinx/Vivado/2023.2/bin/vivado}
echo "RUN_DIR=$run_dir"
exec "$vivado" -mode batch \
    -log "$run_dir/vivado.log" \
    -journal "$run_dir/vivado.jou" \
    -source "$root/scripts/export_bitstream.tcl" \
    -tclargs "$run_dir"
