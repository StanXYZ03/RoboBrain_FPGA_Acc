#!/bin/sh
# Offline-only consistency gate for the Q8Q8 PL-accelerated RAM boot bundle.
# This script never opens JTAG/UART, resets the board or programs the PL.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
ACCEL="$ROOT/fpga_accel/q8q8_zcu102_vivado/artifacts"
RUNTIME="$ROOT/fpga_accel/q8q8_runtime"
RAM="$ROOT/deployment/artifacts/ram_linux"
BUILD="$ROOT/build/ram-linux"
JTAG="$ROOT/deployment/scripts/run_q8q8_accel_jtag.tcl"
VITIS_ROOT=${VITIS_ROOT:-/home/xilinx/Vitis/2023.2}
FDT_BIN="$VITIS_ROOT/data/emulation/qemu/comp/qemu/sysroots/x86_64-petalinux-linux/usr/bin"
READELF="$VITIS_ROOT/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-readelf"
XSCT="$VITIS_ROOT/bin/xsct"

fail() {
    echo "Q8Q8_BUNDLE_FAIL: $*" >&2
    exit 1
}

for file in \
    "$ACCEL/q8q8_zcu102.bit" \
    "$ACCEL/q8q8_zcu102.xsa" \
    "$ACCEL/q8q8_zcu102_routed.dcp" \
    "$ACCEL/q8q8_fsbl.elf" \
    "$ACCEL/q8q8_fsbl_metadata.txt" \
    "$ACCEL/implementation_summary.txt" \
    "$RAM/Image" \
    "$RAM/system-16gb.dtb" \
    "$RAM/rootfs.cpio.gz" \
    "$RAM/boot.scr" \
    "$RAM/SHA256SUMS" \
    "$RAM/NETWORK_CONFIG" \
    "$JTAG"; do
    [ -r "$file" ] || fail "missing artifact: $file"
done
for command in sha256sum unzip gzip cpio grep awk mktemp; do
    command -v "$command" >/dev/null 2>&1 || fail "missing host tool: $command"
done
[ -x "$FDT_BIN/fdtget" ] || fail "fdtget missing"
[ -x "$READELF" ] || fail "readelf missing"
[ -x "$XSCT" ] || fail "XSCT missing"

grep -qx 'TARGET_MHZ=200' "$ACCEL/implementation_summary.txt" || fail "target clock mismatch"
grep -qx 'WNS_NS=0.439' "$ACCEL/implementation_summary.txt" || fail "setup timing gate failed"
grep -qx 'WHS_NS=0.010' "$ACCEL/implementation_summary.txt" || fail "hold timing gate failed"
grep -qx 'UNROUTED_NETS=0' "$ACCEL/implementation_summary.txt" || fail "unrouted nets remain"
grep -qx 'BITSTREAM_GENERATED=YES' "$ACCEL/implementation_summary.txt" || fail "bitstream gate failed"
grep -qx 'BOARD_PROGRAMMED=NO' "$ACCEL/implementation_summary.txt" || fail "unexpected board operation marker"

bit_hash=$(sha256sum "$ACCEL/q8q8_zcu102.bit" | awk '{print $1}')
xsa_bit_hash=$(unzip -p "$ACCEL/q8q8_zcu102.xsa" q8q8_zcu102.bit | sha256sum | awk '{print $1}')
[ "$bit_hash" = "$xsa_bit_hash" ] || fail "standalone and XSA bitstreams differ"

(cd "$ACCEL" && tail -n 1 q8q8_fsbl_metadata.txt | sha256sum -c - >/dev/null) \
    || fail "FSBL checksum mismatch"
grep -qx 'DDR_INIT_MODE=SPD_DYNAMIC_DUAL_RANK' "$ACCEL/q8q8_fsbl_metadata.txt" \
    || fail "FSBL metadata does not select dynamic dual-rank DDR"
grep -qx 'BOOT_PATH=PARTITION_LOAD_EXCLUDED_JTAG' "$ACCEL/q8q8_fsbl_metadata.txt" \
    || fail "FSBL metadata does not select the non-media JTAG path"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
"$READELF" -sW "$ACCEL/q8q8_fsbl.elf" > "$tmp/fsbl.symbols"
for symbol in XFsbl_DdrInit XFsbl_DdrcPhyTraining XFsbl_Exit XFsbl_Loop; do
    grep -q " $symbol$" "$tmp/fsbl.symbols" || fail "FSBL symbol missing: $symbol"
done
grep -q ' psu_ps_pl_isolation_removal_data' "$tmp/fsbl.symbols" \
    || fail "FSBL PS-PL isolation-removal helper missing"
grep -q ' psu_ps_pl_reset_config_data' "$tmp/fsbl.symbols" \
    || fail "FSBL PS-PL reset helper missing"
loop_metadata=$(awk -F= '$1 == "XFSBL_LOOP" {print tolower($2)}' "$ACCEL/q8q8_fsbl_metadata.txt")
loop_elf=$(awk '$8 == "XFsbl_Loop" {print "0x" tolower($2); exit}' "$tmp/fsbl.symbols")
[ "$loop_metadata" = "$loop_elf" ] || fail "FSBL loop metadata mismatch"
isolation_metadata=$(awk -F= '$1 == "PS_PL_ISOLATION_REMOVAL" {print tolower($2)}' "$ACCEL/q8q8_fsbl_metadata.txt")
isolation_elf=$(awk '$8 ~ /^psu_ps_pl_isolation_removal_data/ {print "0x" tolower($2); exit}' "$tmp/fsbl.symbols")
[ "$isolation_metadata" = "$isolation_elf" ] || fail "FSBL isolation helper metadata mismatch"
reset_metadata=$(awk -F= '$1 == "PS_PL_RESET_CONFIG" {print tolower($2)}' "$ACCEL/q8q8_fsbl_metadata.txt")
reset_elf=$(awk '$8 ~ /^psu_ps_pl_reset_config_data/ {print "0x" tolower($2); exit}' "$tmp/fsbl.symbols")
[ "$reset_metadata" = "$reset_elf" ] || fail "FSBL reset helper metadata mismatch"

(cd "$RAM" && sha256sum -c SHA256SUMS >/dev/null) || fail "RAM Linux checksum mismatch"
grep -qx 'pl_accel_mode=1' "$RAM/NETWORK_CONFIG" || fail "RAM Linux is not accelerator-enabled"
grep -qx 'CONFIG_UIO=y' "$BUILD/linux/.config" || fail "CONFIG_UIO is not built in"
grep -qx 'CONFIG_UIO_PDRV_GENIRQ=y' "$BUILD/linux/.config" || fail "generic UIO driver is not built in"

memory_reg=$("$FDT_BIN/fdtget" -t x "$RAM/system-16gb.dtb" /memory reg)
arena_reg=$("$FDT_BIN/fdtget" -t x "$RAM/system-16gb.dtb" /reserved-memory/q8q8-dma@b7c000000 reg)
uio_reg=$("$FDT_BIN/fdtget" -t x "$RAM/system-16gb.dtb" /q8q8-gemm@a0000000 reg)
uio_name=$("$FDT_BIN/fdtget" -t s "$RAM/system-16gb.dtb" /q8q8-gemm@a0000000 linux,uio-name)
bootargs=$("$FDT_BIN/fdtget" -t s "$RAM/system-16gb.dtb" /chosen bootargs)
[ "$memory_reg" = '0 0 0 7ff00000 8 0 3 80000000' ] || fail "16GB memory map mismatch"
[ "$arena_reg" = 'b 7c000000 0 4000000' ] || fail "64MiB DMA arena mismatch"
[ "$uio_reg" = '0 a0000000 0 10000 b 7c000000 0 4000000' ] || fail "UIO resource map mismatch"
[ "$uio_name" = 'q8q8-gemm' ] || fail "UIO name mismatch"
case "$bootargs" in *uio_pdrv_genirq.of_id=generic-uio*) ;; *) fail "generic-uio boot argument missing";; esac
case "$bootargs" in *robobrain.pl_selftest=1*) ;; *) fail "dedicated PL self-test boot mode missing";; esac

mkdir "$tmp/rootfs"
(cd "$tmp/rootfs" && gzip -dc "$RAM/rootfs.cpio.gz" | cpio -id --quiet \
    opt/robobrain/q8q8-selftest opt/robobrain/selftest/llama-server \
    lib/ld-linux-aarch64.so.1 usr/lib/libstdc++.so.6 init) \
    || fail "cannot extract required initramfs files"
for file in \
    opt/robobrain/q8q8-selftest \
    opt/robobrain/selftest/llama-server \
    lib/ld-linux-aarch64.so.1 \
    usr/lib/libstdc++.so.6; do
    [ -s "$tmp/rootfs/$file" ] || fail "initramfs payload missing: $file"
done
grep -q 'RAM_LINUX_PL_SELFTEST_PASS' "$tmp/rootfs/init" \
    || fail "initramfs does not run the dedicated PL self-test"
"$READELF" -h "$tmp/rootfs/opt/robobrain/q8q8-selftest" | grep -q 'Machine:.*AArch64' \
    || fail "self-test is not AArch64"
archive_selftest_hash=$(sha256sum "$tmp/rootfs/opt/robobrain/q8q8-selftest" | awk '{print $1}')
source_selftest_hash=$(sha256sum "$RUNTIME/artifacts/q8q8-selftest-arm64" | awk '{print $1}')
[ "$archive_selftest_hash" = "$source_selftest_hash" ] || fail "staged self-test differs from runtime build"

grep -q 'q8q8_fsbl.elf' "$JTAG" || fail "JTAG script does not select matching FSBL"
grep -q 'q8q8_zcu102.bit' "$JTAG" || fail "JTAG script does not select matching bitstream"
grep -q 'fpga -file \$bitstream' "$JTAG" || fail "JTAG script does not configure PL"
grep -q 'set pl_state \[fpga -state\]' "$JTAG" || fail "JTAG FPGA configuration-state gate missing"
grep -q 'XFSBL_LOOP=' "$JTAG" || fail "JTAG script does not read FSBL loop metadata"
grep -q 'bpadd -addr \$fsbl_loop' "$JTAG" || fail "JTAG script FSBL breakpoint is not metadata-driven"
grep -q 'RAM_BOOT_Q8Q8_PL_GATE=PASS' "$JTAG" || fail "JTAG PL read-back gate missing"
grep -q 'call_fsbl_helper \$ps_pl_isolation_removal' "$JTAG" \
    || fail "JTAG PS-PL isolation removal missing"
grep -q 'call_fsbl_helper \$ps_pl_reset_config' "$JTAG" \
    || fail "JTAG PS-PL reset release missing"
grep -q 'RAM_BOOT_Q8Q8_AXI_GATE=PASS' "$JTAG" \
    || fail "JTAG AXI-Lite reachability gate missing"
tcl_complete=$("$XSCT" -nodisp -quiet -eval \
    "set fd [open {$JTAG} r]; set script [read \$fd]; close \$fd; puts [info complete \$script]")
[ "$tcl_complete" = 1 ] || fail "JTAG Tcl script is syntactically incomplete"

sh -n "$ROOT/deployment/scripts/build_ram_linux.sh"
sh -n "$ROOT/fpga_accel/q8q8_zcu102_vivado/scripts/build_accel_fsbl.sh"
sh -n "$ROOT/deployment/scripts/verify_q8q8_accel_bundle.sh"
grep -q -- '-DFSBL_PARTITION_LOAD_EXCLUDE' \
    "$ROOT/fpga_accel/q8q8_zcu102_vivado/scripts/build_accel_fsbl.sh" \
    || fail "FSBL is not forced onto the non-media JTAG path"
grep -q '0x53504434U' "$ROOT/fpga_accel/q8q8_zcu102_vivado/build/q8q8_fsbl/xfsbl_ddr_init.c" \
    || fail "dynamic DDR SPD evidence patch is absent"
grep -q 'AddrMapCsBit0 = PDimmPtr->NumBgAddr' \
    "$ROOT/fpga_accel/q8q8_zcu102_vivado/build/q8q8_fsbl/xfsbl_ddr_init.c" \
    || fail "dynamic dual-rank address-map patch is absent"
grep -q 'AFIFM0_AFIFM_RDCTRL_OFFSET' \
    "$ROOT/fpga_accel/q8q8_zcu102_vivado/build/q8q8_fsbl/psu_init.c" \
    || fail "matching XSA PS-PL AFI configuration is absent"

echo "Q8Q8_BUNDLE_PASS bit_sha256=$bit_hash memory_bytes=17178820608 dma_bytes=67108864 control=0xA0000000 board_programmed=no"
