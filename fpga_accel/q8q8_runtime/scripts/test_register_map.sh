#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
IP_ZIP=${Q8Q8_IP_ZIP:-$ROOT/../q8q8_dataflow_hls/artifacts/q8q8_dataflow_ip.zip}
HEADER=$(mktemp)
trap 'rm -f "$HEADER"' EXIT HUP INT TERM

[ -f "$IP_ZIP" ] || {
    echo "REGISTER_MAP_FAIL missing_ip_zip=$IP_ZIP" >&2
    exit 1
}
unzip -p "$IP_ZIP" \
    drivers/q8q8_gemm_v1_0/src/xq8q8_gemm_hw.h > "$HEADER"

check_define() {
    name=$1
    expected=$2
    actual=$(awk -v key="$name" '$1 == "#define" && $2 == key { print $3 }' "$HEADER")
    [ "$actual" = "$expected" ] || {
        echo "REGISTER_MAP_FAIL name=$name expected=$expected actual=${actual:-missing}" >&2
        exit 1
    }
}

check_define XQ8Q8_GEMM_CONTROL_ADDR_AP_CTRL 0x00
check_define XQ8Q8_GEMM_CONTROL_ADDR_AP_RETURN 0x10
check_define XQ8Q8_GEMM_CONTROL_ADDR_WEIGHTS_Q_DATA 0x18
check_define XQ8Q8_GEMM_CONTROL_ADDR_WEIGHTS_SCALE_DATA 0x24
check_define XQ8Q8_GEMM_CONTROL_ADDR_ACTIVATIONS_Q_DATA 0x30
check_define XQ8Q8_GEMM_CONTROL_ADDR_ACTIVATIONS_SCALE_DATA 0x3c
check_define XQ8Q8_GEMM_CONTROL_ADDR_OUTPUT_R_DATA 0x48
check_define XQ8Q8_GEMM_CONTROL_ADDR_M_DATA 0x54
check_define XQ8Q8_GEMM_CONTROL_ADDR_N_DATA 0x5c
check_define XQ8Q8_GEMM_CONTROL_ADDR_K_DATA 0x64

echo "REGISTER_MAP_PASS registers=10 source=$(basename "$IP_ZIP")"
