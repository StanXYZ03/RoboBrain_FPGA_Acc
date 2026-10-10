#!/bin/sh
set -eu

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
FDT_BIN="${FDT_BIN:-/home/xilinx/Vitis/2023.2/data/emulation/qemu/comp/qemu/sysroots/x86_64-petalinux-linux/usr/bin}"
BASE_DTB="${BASE_DTB:-/home/xilinx/Vitis/2023.2/base_platforms/xilinx_zcu102_base_202320_1/sw/xilinx_zcu102_base_202320_1/boot/system.dtb}"
ARTIFACTS="$PROJECT_ROOT/artifacts"
DTC="$FDT_BIN/dtc"
FDTOVERLAY="$FDT_BIN/fdtoverlay"
FDTGET="$FDT_BIN/fdtget"
FDTPUT="$FDT_BIN/fdtput"

mkdir -p "$ARTIFACTS"
"$DTC" -@ -I dts -O dtb \
    -o "$ARTIFACTS/q8q8-uio-overlay.dtbo" \
    "$PROJECT_ROOT/dts/q8q8-uio-overlay.dts"
cp "$BASE_DTB" "$ARTIFACTS/q8q8-base-16gb-test.dtb"
"$FDTPUT" -t x "$ARTIFACTS/q8q8-base-16gb-test.dtb" /memory reg \
    0 0 0 7ff00000 8 0 3 80000000
"$FDTOVERLAY" -i "$ARTIFACTS/q8q8-base-16gb-test.dtb" \
    -o "$ARTIFACTS/q8q8-runtime-test.dtb" \
    "$ARTIFACTS/q8q8-uio-overlay.dtbo"

memory_reg="$("$FDTGET" -t x "$ARTIFACTS/q8q8-runtime-test.dtb" /memory reg)"
arena_reg="$("$FDTGET" -t x "$ARTIFACTS/q8q8-runtime-test.dtb" \
    /reserved-memory/q8q8-dma@b7c000000 reg)"
uio_reg="$("$FDTGET" -t x "$ARTIFACTS/q8q8-runtime-test.dtb" \
    /q8q8-gemm@a0000000 reg)"
compatible="$("$FDTGET" -t s "$ARTIFACTS/q8q8-runtime-test.dtb" \
    /q8q8-gemm@a0000000 compatible)"

[ "$memory_reg" = "0 0 0 7ff00000 8 0 3 80000000" ]
[ "$arena_reg" = "b 7c000000 0 4000000" ]
[ "$uio_reg" = "0 a0000000 0 10000 b 7c000000 0 4000000" ]
[ "$compatible" = "generic-uio" ]
echo "DT_TEST_PASS memory='$memory_reg' arena='$arena_reg' uio='$uio_reg'"
