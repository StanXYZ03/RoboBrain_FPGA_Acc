#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 validate|synth|impl" >&2
    exit 2
fi
stage=$1
case "$stage" in
    validate|synth|impl) ;;
    *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$root/artifacts/runs/${stamp}_${stage}"
test ! -e "$run_dir" || { echo "refusing to overwrite $run_dir" >&2; exit 3; }
mkdir -p "$run_dir"

vivado=${VIVADO:-/home/xilinx/Vivado/2023.2/bin/vivado}
test -x "$vivado" || { echo "missing Vivado: $vivado" >&2; exit 4; }
echo "RUN_DIR=$run_dir"
exec "$vivado" -mode batch \
    -log "$run_dir/vivado.log" \
    -journal "$run_dir/vivado.jou" \
    -source "$root/scripts/build_system.tcl" \
    -tclargs "$stage" "$run_dir"
