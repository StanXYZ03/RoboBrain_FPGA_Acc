#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
XSCT=${XSCT:-/home/xilinx/Vitis/2023.2/bin/xsct}
BAREMETAL_BIN=${BAREMETAL_BIN:-/home/xilinx/Vitis/2023.2/gnu/aarch64/lin/aarch64-none/bin}
BUILD=$ROOT/build/q8q8_fsbl
OUTPUT=$ROOT/artifacts/q8q8_fsbl.elf

[ -x "$BAREMETAL_BIN/aarch64-none-elf-gcc" ] || {
    echo "Q8Q8_FSBL_FAIL bare-metal toolchain missing" >&2
    exit 1
}
PATH="$BAREMETAL_BIN:$PATH"
export PATH

"$XSCT" -nodisp "$ROOT/scripts/generate_accel_fsbl.tcl"
patch -d "$BUILD" -p0 < "$ROOT/scripts/dynamic_ddr16_fsbl.patch"
# This FSBL is a JTAG-only PS initializer.  SD/QSPI/NAND, secure-boot and
# BOOT.BIN bitstream loaders are unreachable on that path and overflow OCM
# when linked with the full board BSP.  Excluding them retains the generated
# PS/DDR/clock/AXI initialization and JTAG handoff used by this deployment.
FSBL_DEFINES="-DFSBL_BS_EXCLUDE -DFSBL_NAND_EXCLUDE -DFSBL_QSPI_EXCLUDE"
FSBL_DEFINES="$FSBL_DEFINES -DFSBL_SD_EXCLUDE -DFSBL_SECURE_EXCLUDE"
FSBL_DEFINES="$FSBL_DEFINES -DFSBL_WDT_EXCLUDE"
FSBL_DEFINES="$FSBL_DEFINES -DFSBL_PARTITION_LOAD_EXCLUDE"
make -C "$BUILD" CFLAGS="$FSBL_DEFINES"

elf=$(find "$BUILD" -maxdepth 2 -type f -name '*.elf' | head -n 1)
[ -n "$elf" ] && [ -r "$elf" ] || {
    echo "Q8Q8_FSBL_FAIL no ELF generated" >&2
    exit 1
}
cp "$elf" "$OUTPUT"

readelf_bin=/home/xilinx/Vitis/2023.2/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-readelf
exit_addr=$($readelf_bin -sW "$OUTPUT" | awk '$8 == "XFsbl_Exit" { print "0x" $2; exit }')
loop_addr=$($readelf_bin -sW "$OUTPUT" | awk '$8 == "XFsbl_Loop" { print "0x" $2; exit }')
isolation_addr=$($readelf_bin -sW "$OUTPUT" | awk '$8 ~ /^psu_ps_pl_isolation_removal_data/ { print "0x" $2; exit }')
reset_addr=$($readelf_bin -sW "$OUTPUT" | awk '$8 ~ /^psu_ps_pl_reset_config_data/ { print "0x" $2; exit }')
[ -n "$exit_addr" ] && [ -n "$loop_addr" ] && \
        [ -n "$isolation_addr" ] && [ -n "$reset_addr" ] || {
    echo "Q8Q8_FSBL_FAIL required symbols missing" >&2
    exit 1
}
{
    echo "XFSBL_EXIT=$exit_addr"
    echo "XFSBL_LOOP=$loop_addr"
    echo "PS_PL_ISOLATION_REMOVAL=$isolation_addr"
    echo "PS_PL_RESET_CONFIG=$reset_addr"
    echo "DDR_INIT_MODE=SPD_DYNAMIC_DUAL_RANK"
    echo "BOOT_PATH=PARTITION_LOAD_EXCLUDED_JTAG"
    (cd "$ROOT" && sha256sum scripts/dynamic_ddr16_fsbl.patch)
    (cd "$BUILD" && sha256sum psu_init.c xfsbl_ddr_init.c)
    (cd "$ROOT/artifacts" && sha256sum q8q8_fsbl.elf)
} > "$ROOT/artifacts/q8q8_fsbl_metadata.txt"

echo "Q8Q8_FSBL_PASS elf=$OUTPUT exit=$exit_addr loop=$loop_addr isolation=$isolation_addr reset=$reset_addr"
