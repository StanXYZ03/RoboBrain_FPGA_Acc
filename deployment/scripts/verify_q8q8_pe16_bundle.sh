#!/bin/sh
# Offline-only PE16 bitstream, FSBL and versioned RAM-Linux consistency gate.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
ACCEL="$ROOT/fpga_accel/q8q8_pe16_zcu102_vivado/artifacts"
HLS="$ROOT/fpga_accel/q8q8_pe16_hls/artifacts"
RUNTIME="$ROOT/fpga_accel/q8q8_runtime"
RUNTIME_ARTIFACTS="$RUNTIME/artifacts/pe16_20260828"
RAM="$ROOT/deployment/artifacts/ram_linux_pe16_20260828"
BUILD="$ROOT/build/ram-linux-pe16-20260828"
JTAG="$ROOT/deployment/scripts/run_q8q8_accel_jtag.tcl"
VITIS_ROOT=${VITIS_ROOT:-/home/xilinx/Vitis/2023.2}
FDT_BIN="$VITIS_ROOT/data/emulation/qemu/comp/qemu/sysroots/x86_64-petalinux-linux/usr/bin"
READELF="$VITIS_ROOT/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-readelf"
XSCT="$VITIS_ROOT/bin/xsct"

fail() {
    echo "Q8Q8_PE16_BUNDLE_FAIL: $*" >&2
    exit 1
}

for file in \
    "$HLS/q8q8_pe16_ip.zip" \
    "$ACCEL/q8q8_pe16_zcu102.bit" \
    "$ACCEL/q8q8_pe16_zcu102.xsa" \
    "$ACCEL/q8q8_pe16_zcu102_routed.dcp" \
    "$ACCEL/q8q8_pe16_fsbl.elf" \
    "$ACCEL/q8q8_pe16_fsbl_metadata.txt" \
    "$ACCEL/implementation_summary.txt" \
    "$RUNTIME_ARTIFACTS/q8q8-selftest-arm64" \
    "$RUNTIME_ARTIFACTS/q8q8-benchmark-arm64" \
    "$RAM/Image" "$RAM/system-16gb.dtb" "$RAM/rootfs.cpio.gz" \
    "$RAM/boot.scr" "$RAM/SHA256SUMS" "$RAM/NETWORK_CONFIG" "$JTAG"; do
    [ -r "$file" ] || fail "missing artifact: $file"
done
for command in sha256sum unzip gzip cpio grep awk mktemp; do
    command -v "$command" >/dev/null 2>&1 || fail "missing host tool: $command"
done
[ -x "$FDT_BIN/fdtget" ] || fail "fdtget missing"
[ -x "$READELF" ] || fail "readelf missing"
[ -x "$XSCT" ] || fail "XSCT missing"

grep -qx 'TARGET_MHZ=200' "$ACCEL/implementation_summary.txt" || fail "clock target mismatch"
grep -qx 'WNS_NS=0.067' "$ACCEL/implementation_summary.txt" || fail "setup timing gate failed"
grep -qx 'WHS_NS=0.011' "$ACCEL/implementation_summary.txt" || fail "hold timing gate failed"
grep -qx 'UNROUTED_NETS=0' "$ACCEL/implementation_summary.txt" || fail "unrouted nets remain"
grep -qx 'OUTPUT_AXI_WIDTH=128' "$ACCEL/implementation_summary.txt" || fail "AXI output width mismatch"
grep -qx 'BITSTREAM_GENERATED=YES' "$ACCEL/implementation_summary.txt" || fail "bitstream gate failed"
grep -qx 'BOARD_PROGRAMMED=NO' "$ACCEL/implementation_summary.txt" || fail "unexpected board marker"

bit_hash=$(sha256sum "$ACCEL/q8q8_pe16_zcu102.bit" | awk '{print $1}')
xsa_bit_hash=$(unzip -p "$ACCEL/q8q8_pe16_zcu102.xsa" \
    q8q8_pe16_zcu102.bit | sha256sum | awk '{print $1}')
[ "$bit_hash" = "$xsa_bit_hash" ] || fail "standalone and XSA bitstreams differ"

(cd "$ACCEL" && tail -n 1 q8q8_pe16_fsbl_metadata.txt | sha256sum -c - >/dev/null) \
    || fail "FSBL checksum mismatch"
grep -qx 'DDR_INIT_MODE=SPD_DYNAMIC_DUAL_RANK' "$ACCEL/q8q8_pe16_fsbl_metadata.txt" \
    || fail "dynamic DDR mode missing"
grep -qx 'BOOT_PATH=PARTITION_LOAD_EXCLUDED_JTAG' "$ACCEL/q8q8_pe16_fsbl_metadata.txt" \
    || fail "JTAG-only FSBL mode missing"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
"$READELF" -sW "$ACCEL/q8q8_pe16_fsbl.elf" > "$tmp/fsbl.symbols"
for symbol in XFsbl_DdrInit XFsbl_DdrcPhyTraining XFsbl_Exit XFsbl_Loop; do
    grep -q " $symbol$" "$tmp/fsbl.symbols" || fail "FSBL symbol missing: $symbol"
done
grep -q ' psu_ps_pl_isolation_removal_data' "$tmp/fsbl.symbols" \
    || fail "PS-PL isolation helper missing"
grep -q ' psu_ps_pl_reset_config_data' "$tmp/fsbl.symbols" \
    || fail "PS-PL reset helper missing"
grep -q '0x53504434U' "$ROOT/fpga_accel/q8q8_pe16_zcu102_vivado/build/q8q8_pe16_fsbl/xfsbl_ddr_init.c" \
    || fail "SPD evidence patch missing"
grep -q 'AddrMapCsBit0 = PDimmPtr->NumBgAddr' \
    "$ROOT/fpga_accel/q8q8_pe16_zcu102_vivado/build/q8q8_pe16_fsbl/xfsbl_ddr_init.c" \
    || fail "dual-rank address patch missing"

(cd "$RAM" && sha256sum -c SHA256SUMS >/dev/null) || fail "RAM-Linux checksums differ"
grep -qx 'pl_accel_mode=1' "$RAM/NETWORK_CONFIG" || fail "accelerator mode disabled"
grep -qx 'CONFIG_UIO=y' "$BUILD/linux/.config" || fail "UIO not built in"
grep -qx 'CONFIG_UIO_PDRV_GENIRQ=y' "$BUILD/linux/.config" || fail "generic UIO not built in"

memory_reg=$("$FDT_BIN/fdtget" -t x "$RAM/system-16gb.dtb" /memory reg)
arena_reg=$("$FDT_BIN/fdtget" -t x "$RAM/system-16gb.dtb" \
    /reserved-memory/q8q8-dma@b7c000000 reg)
uio_reg=$("$FDT_BIN/fdtget" -t x "$RAM/system-16gb.dtb" \
    /q8q8-gemm@a0000000 reg)
[ "$memory_reg" = '0 0 0 7ff00000 8 0 3 80000000' ] || fail "16GB map mismatch"
[ "$arena_reg" = 'b 7c000000 0 4000000' ] || fail "DMA arena mismatch"
[ "$uio_reg" = '0 a0000000 0 10000 b 7c000000 0 4000000' ] || fail "UIO map mismatch"

mkdir "$tmp/rootfs"
(cd "$tmp/rootfs" && gzip -dc "$RAM/rootfs.cpio.gz" | cpio -id --quiet \
    opt/robobrain/q8q8-selftest opt/robobrain/q8q8-benchmark init) \
    || fail "cannot extract PE16 runtime payload"
for name in q8q8-selftest q8q8-benchmark; do
    [ -s "$tmp/rootfs/opt/robobrain/$name" ] || fail "payload missing: $name"
    "$READELF" -h "$tmp/rootfs/opt/robobrain/$name" | grep -q 'Machine:.*AArch64' \
        || fail "$name is not AArch64"
done
[ "$(sha256sum "$tmp/rootfs/opt/robobrain/q8q8-selftest" | awk '{print $1}')" = \
  "$(sha256sum "$RUNTIME_ARTIFACTS/q8q8-selftest-arm64" | awk '{print $1}')" ] \
    || fail "staged self-test mismatch"
[ "$(sha256sum "$tmp/rootfs/opt/robobrain/q8q8-benchmark" | awk '{print $1}')" = \
  "$(sha256sum "$RUNTIME_ARTIFACTS/q8q8-benchmark-arm64" | awk '{print $1}')" ] \
    || fail "staged benchmark mismatch"
grep -q 'RAM_LINUX_PL_SELFTEST_PASS' "$tmp/rootfs/init" \
    || fail "init does not run PL self-test"

Q8Q8_IP_ZIP="$HLS/q8q8_pe16_ip.zip" sh "$RUNTIME/scripts/test_register_map.sh" \
    > "$tmp/register-map.log"
grep -q 'REGISTER_MAP_PASS registers=10' "$tmp/register-map.log" \
    || fail "PE16 register map mismatch"

for variable in Q8Q8_RAM_ARTIFACT_DIR Q8Q8_ACCEL_FSBL \
        Q8Q8_ACCEL_FSBL_METADATA Q8Q8_ACCEL_BITSTREAM; do
    grep -q "$variable" "$JTAG" || fail "JTAG path override missing: $variable"
done
tcl_complete=$("$XSCT" -nodisp -quiet -eval \
    "set fd [open {$JTAG} r]; set script [read \$fd]; close \$fd; puts [info complete \$script]")
[ "$tcl_complete" = 1 ] || fail "JTAG Tcl is incomplete"

echo "Q8Q8_PE16_BUNDLE_PASS bit_sha256=$bit_hash memory_bytes=17178820608 dma_bytes=67108864 control=0xA0000000 board_programmed=no"
