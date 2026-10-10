#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VITIS_HLS_BIN="${VITIS_HLS_BIN:-/home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls}"

cd "$ROOT_DIR"
"$VITIS_HLS_BIN" -f scripts/run_cosim_small.tcl 2>&1 | tee artifacts/cosim_small.log
