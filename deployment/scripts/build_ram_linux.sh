#!/bin/sh
# Build a reproducible, RAM-only Linux/initramfs bundle for ZCU102 Rev1.0.
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
LINUX_SOURCE="${LINUX_SOURCE:-$ROOT/third_party/linux-xlnx}"
BUSYBOX_SOURCE="${BUSYBOX_SOURCE:-$ROOT/third_party/busybox}"
BUILD_ROOT="${RAM_LINUX_BUILD_ROOT:-$ROOT/build/ram-linux}"
OUTPUT="${RAM_LINUX_OUTPUT:-$ROOT/deployment/artifacts/ram_linux}"
LINUX_COMMIT=a19da02cf5b44420ec6afb1eef348c21d9e8cda2
BUSYBOX_COMMIT=1a64f6a20aaf6ea4dbba68bbfa8cc1ab7e5c57c4
VITIS_ROOT="${VITIS_ROOT:-/home/xilinx/Vitis/2023.2}"
VITIS_TOOLCHAIN="$VITIS_ROOT/gnu/aarch64/lin/aarch64-linux/bin"
VITIS_CC="$VITIS_TOOLCHAIN/aarch64-linux-gnu-gcc"
VITIS_CXX="$VITIS_TOOLCHAIN/aarch64-linux-gnu-g++"
CROSS_COMPILE="${CROSS_COMPILE:-aarch64-linux-gnu-}"
JOBS="${BUILD_JOBS:-4}"
BASE_DTB="${BASE_DTB:-$VITIS_ROOT/base_platforms/xilinx_zcu102_base_202320_1/sw/xilinx_zcu102_base_202320_1/boot/system.dtb}"
FDT_BIN="$VITIS_ROOT/data/emulation/qemu/comp/qemu/sysroots/x86_64-petalinux-linux/usr/bin"
HOST_TOOLS="${HOST_TOOLS:-$ROOT/third_party/host-tools/root/usr/bin}"
HOST_SOURCE_IFACE="${HOST_SOURCE_IFACE:-wlo1}"
BOARD_ADDRESS="${BOARD_ADDRESS:-172.21.127.254/17}"
PROFILE_MODE="${ROBOBRAIN_PROFILE_MODE:-0}"
PL_ACCEL_MODE="${ROBOBRAIN_PL_ACCEL_MODE:-0}"
RUNTIME_ARTIFACTS="${Q8Q8_RUNTIME_ARTIFACTS:-$ROOT/fpga_accel/q8q8_runtime/artifacts}"

if [ -d "$HOST_TOOLS" ]; then
    PATH="$HOST_TOOLS:$PATH"
    M4="$HOST_TOOLS/m4"
    host_lib="$(dirname -- "$HOST_TOOLS")/lib/x86_64-linux-gnu"
    BISON_PKGDATADIR="$(dirname -- "$HOST_TOOLS")/share/bison"
    LD_LIBRARY_PATH="$host_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export M4 BISON_PKGDATADIR LD_LIBRARY_PATH
    export PATH
fi

die() { echo "RAM_LINUX_BUILD_FAIL: $*" >&2; exit 1; }
[ "$(git -C "$LINUX_SOURCE" rev-parse HEAD)" = "$LINUX_COMMIT" ] || die "linux-xlnx commit mismatch"
[ "$(git -C "$BUSYBOX_SOURCE" rev-parse HEAD)" = "$BUSYBOX_COMMIT" ] || die "BusyBox commit mismatch"
[ -x "$VITIS_CC" ] && [ -x "$VITIS_CXX" ] || die "Vitis ARM64 toolchain missing"
[ -r "$BASE_DTB" ] || die "base ZCU102 device tree missing"
for command in make cpio gzip python3 "${CROSS_COMPILE}gcc"; do
    command -v "$command" >/dev/null 2>&1 || die "missing build command: $command"
done
HOST_REACHABLE_IP="${ROBOBRAIN_HOST_IP:-$(ip -4 -o address show dev "$HOST_SOURCE_IFACE" scope global | awk 'NR == 1 {split($4, value, "/"); print value[1]}')}"
case "$HOST_REACHABLE_IP" in ''|*[!0-9.]*) die "invalid or missing host IPv4 on $HOST_SOURCE_IFACE";; esac
case "$BOARD_ADDRESS" in ''|*[!0-9./]*) die "invalid board address";; esac
case "$PROFILE_MODE" in 0|1) ;; *) die "invalid profile mode";; esac
case "$PL_ACCEL_MODE" in 0|1) ;; *) die "invalid PL accelerator mode";; esac

LINUX_BUILD="$BUILD_ROOT/linux"
BUSYBOX_BUILD="$BUILD_ROOT/busybox"
ROOTFS="$BUILD_ROOT/rootfs"
mkdir -p "$LINUX_BUILD" "$BUSYBOX_BUILD" "$OUTPUT"

make -C "$LINUX_SOURCE" O="$LINUX_BUILD" ARCH=arm64 CROSS_COMPILE="$CROSS_COMPILE" xilinx_zynqmp_defconfig
"$LINUX_SOURCE/scripts/config" --file "$LINUX_BUILD/.config" \
    -e BLK_DEV_INITRD -e DEVTMPFS -e DEVTMPFS_MOUNT -e TMPFS \
    -e MACB -e PHYLIB -e MMC -e MMC_BLOCK -e MMC_SDHCI_OF_ARASAN \
    -e RD_GZIP -e BINFMT_ELF -e IKCONFIG -e IKCONFIG_PROC \
    -e SERIAL_AMBA_PL011 -e SERIAL_AMBA_PL011_CONSOLE \
    -d MODULE_SIG -d CFG80211_USE_KERNEL_REGDB_KEYS -d CFG80211 -d WLAN \
    -d SYSTEM_DATA_VERIFICATION -d SYSTEM_TRUSTED_KEYRING -d SYSTEM_REVOCATION_LIST
if [ "$PL_ACCEL_MODE" = 1 ]; then
    "$LINUX_SOURCE/scripts/config" --file "$LINUX_BUILD/.config" \
        -e UIO -e UIO_PDRV_GENIRQ
fi
make -C "$LINUX_SOURCE" O="$LINUX_BUILD" ARCH=arm64 CROSS_COMPILE="$CROSS_COMPILE" olddefconfig
make -C "$LINUX_SOURCE" O="$LINUX_BUILD" ARCH=arm64 CROSS_COMPILE="$CROSS_COMPILE" -j"$JOBS" Image

make -C "$BUSYBOX_SOURCE" O="$BUSYBOX_BUILD" ARCH=arm64 CROSS_COMPILE="$CROSS_COMPILE" defconfig
busybox_enable() {
    option="$1"
    if grep -q "^# CONFIG_${option} is not set$" "$BUSYBOX_BUILD/.config"; then
        sed -i "s/^# CONFIG_${option} is not set$/CONFIG_${option}=y/" "$BUSYBOX_BUILD/.config"
    elif ! grep -q "^CONFIG_${option}=y$" "$BUSYBOX_BUILD/.config"; then
        printf 'CONFIG_%s=y\n' "$option" >> "$BUSYBOX_BUILD/.config"
    fi
}
for option in STATIC FEATURE_HTTPD_CGI WGET NC BLOCKDEV SHA256SUM IP MOUNT POWEROFF FEATURE_SH_STANDALONE SETSID CTTYHACK; do
    busybox_enable "$option"
done
make -C "$BUSYBOX_SOURCE" O="$BUSYBOX_BUILD" ARCH=arm64 CROSS_COMPILE="$CROSS_COMPILE" -j"$JOBS"

rm -rf "$ROOTFS"
mkdir -p "$ROOTFS"
make -C "$BUSYBOX_SOURCE" O="$BUSYBOX_BUILD" ARCH=arm64 CROSS_COMPILE="$CROSS_COMPILE" CONFIG_PREFIX="$ROOTFS" install
mkdir -p "$ROOTFS/dev" "$ROOTFS/proc" "$ROOTFS/sys" "$ROOTFS/run" \
    "$ROOTFS/tmp" "$ROOTFS/etc" "$ROOTFS/lib" "$ROOTFS/usr/lib" \
    "$ROOTFS/dev/shm" "$ROOTFS/opt/robobrain/selftest"
cp "$ROOT/deployment/ram_linux/init" "$ROOTFS/init"
chmod 0755 "$ROOTFS/init"
sh "$ROOT/deployment/scripts/stage_initramfs.sh" "$ROOTFS"
if [ "$PL_ACCEL_MODE" = 1 ]; then
    make -C "$ROOT/fpga_accel/q8q8_runtime" \
        ARTIFACTS="$RUNTIME_ARTIFACTS" arm64
    cp "$RUNTIME_ARTIFACTS/q8q8-selftest-arm64" \
        "$ROOTFS/opt/robobrain/q8q8-selftest"
    chmod 0755 "$ROOTFS/opt/robobrain/q8q8-selftest"
    cp "$RUNTIME_ARTIFACTS/q8q8-benchmark-arm64" \
        "$ROOTFS/opt/robobrain/q8q8-benchmark"
    chmod 0755 "$ROOTFS/opt/robobrain/q8q8-benchmark"
    cp "$ROOT/deployment/ram_linux/q8q8_dataflow_measure.sh" \
        "$ROOTFS/opt/robobrain/q8q8-dataflow-measure"
    chmod 0755 "$ROOTFS/opt/robobrain/q8q8-dataflow-measure"
fi
cp "$ROOT/deployment/artifacts/arm64/llama-server" "$ROOTFS/opt/robobrain/selftest/llama-server"
chmod 0755 "$ROOTFS/opt/robobrain/selftest/llama-server"

SYSROOT="$($VITIS_CC -print-sysroot)"
copy_runtime() {
    source="$1"; destination="$2"
    [ -r "$source" ] || die "missing runtime library: $source"
    cp -L "$source" "$ROOTFS/$destination"
}
copy_runtime "$SYSROOT/lib/ld-linux-aarch64.so.1" lib/ld-linux-aarch64.so.1
copy_runtime "$($VITIS_CC -print-file-name=libc.so.6)" lib/libc.so.6
copy_runtime "$($VITIS_CC -print-file-name=libm.so.6)" lib/libm.so.6
copy_runtime "$($VITIS_CC -print-file-name=libgcc_s.so.1)" lib/libgcc_s.so.1
copy_runtime "$($VITIS_CXX -print-file-name=libstdc++.so.6)" usr/lib/libstdc++.so.6

cat > "$ROOTFS/etc/passwd" <<'EOF'
root:x:0:0:root:/root:/bin/sh
EOF
cat > "$ROOTFS/etc/group" <<'EOF'
root:x:0:
EOF
cat > "$ROOTFS/etc/nsswitch.conf" <<'EOF'
hosts: files dns
EOF
cat > "$ROOTFS/etc/hosts" <<'EOF'
127.0.0.1 localhost zcu102-robobrain
EOF

(cd "$ROOTFS" && find . -print0 | LC_ALL=C sort -z | cpio --null -o --format=newc --owner=0:0 2>/dev/null | gzip -n -9 > "$OUTPUT/rootfs.cpio.gz")
cp "$LINUX_BUILD/arch/arm64/boot/Image" "$OUTPUT/Image"
cp "$BASE_DTB" "$OUTPUT/system-16gb.dtb"
"$FDT_BIN/fdtput" -t x "$OUTPUT/system-16gb.dtb" /memory reg \
    0 0 0 7ff00000 8 0 3 80000000
bootargs_accel=""
if [ "$PL_ACCEL_MODE" = 1 ]; then
    "$FDT_BIN/dtc" -@ -I dts -O dtb \
        -o "$BUILD_ROOT/q8q8-uio-overlay.dtbo" \
        "$ROOT/fpga_accel/q8q8_runtime/dts/q8q8-uio-overlay.dts"
    "$FDT_BIN/fdtoverlay" -i "$OUTPUT/system-16gb.dtb" \
        -o "$BUILD_ROOT/system-16gb-accel.dtb" \
        "$BUILD_ROOT/q8q8-uio-overlay.dtbo"
    cp "$BUILD_ROOT/system-16gb-accel.dtb" "$OUTPUT/system-16gb.dtb"
    bootargs_accel=" uio_pdrv_genirq.of_id=generic-uio robobrain.pl_selftest=1"
fi
# Disable the unrelated PL peripherals inherited from the base-platform DT.
# In accelerator mode the Q8Q8 UIO node is a separate root-level node added by
# the overlay above, so disabling this stale subtree does not disable Q8Q8.
# Probing the stale subtree previously stalled at an unmapped AXI interrupt
# controller.
"$FDT_BIN/fdtput" -t s "$OUTPUT/system-16gb.dtb" /amba_pl@0 status disabled
# Early irqchip discovery walks interrupt-controller nodes independently of
# normal platform-device parent availability, so disable the PL INTC itself.
"$FDT_BIN/fdtput" -t s "$OUTPUT/system-16gb.dtb" \
    /amba_pl@0/interrupt-controller@80020000 status disabled
"$FDT_BIN/fdtput" -t s "$OUTPUT/system-16gb.dtb" /chosen bootargs \
    "console=ttyPS0,115200 earlycon root=/dev/ram0 rw rdinit=/init clk_ignore_unused robobrain.host_ip=$HOST_REACHABLE_IP robobrain.board_address=$BOARD_ADDRESS robobrain.profile=$PROFILE_MODE$bootargs_accel"
memory_reg="$("$FDT_BIN/fdtget" -t x "$OUTPUT/system-16gb.dtb" /memory reg)"
[ "$memory_reg" = "0 0 0 7ff00000 8 0 3 80000000" ] || die "unexpected 16GB DT memory map: $memory_reg"
pl_status="$("$FDT_BIN/fdtget" -t s "$OUTPUT/system-16gb.dtb" /amba_pl@0 status)"
[ "$pl_status" = "disabled" ] || die "stale base-platform PL subtree must be disabled"
pl_intc_status="$("$FDT_BIN/fdtget" -t s "$OUTPUT/system-16gb.dtb" \
    /amba_pl@0/interrupt-controller@80020000 status)"
[ "$pl_intc_status" = "disabled" ] || die "PL interrupt controller must be disabled"
if [ "$PL_ACCEL_MODE" = 1 ]; then
    arena_reg="$("$FDT_BIN/fdtget" -t x "$OUTPUT/system-16gb.dtb" \
        /reserved-memory/q8q8-dma@b7c000000 reg)"
    uio_reg="$("$FDT_BIN/fdtget" -t x "$OUTPUT/system-16gb.dtb" \
        /q8q8-gemm@a0000000 reg)"
    [ "$arena_reg" = "b 7c000000 0 4000000" ] || die "bad Q8Q8 arena: $arena_reg"
    [ "$uio_reg" = "0 a0000000 0 10000 b 7c000000 0 4000000" ] \
        || die "bad Q8Q8 UIO resources: $uio_reg"
fi

ramdisk_size="$(wc -c < "$OUTPUT/rootfs.cpio.gz")"
cat > "$OUTPUT/boot.cmd" <<EOF
echo ROBOBRAIN_JTAG_SCRIPT_BEGIN
mw.l 0x03000100 0x53435201
setenv bootargs console=ttyPS0,115200 earlycon root=/dev/ram0 rw rdinit=/init clk_ignore_unused robobrain.host_ip=$HOST_REACHABLE_IP robobrain.board_address=$BOARD_ADDRESS robobrain.profile=$PROFILE_MODE$bootargs_accel
mw.l 0x03000100 0x53435202
booti 0x18000000 0x04000000:${ramdisk_size} 0x00100000
mw.l 0x03000100 0x53435203
echo ROBOBRAIN_JTAG_SCRIPT_FAIL
EOF
python3 "$ROOT/deployment/scripts/make_uboot_script.py" \
    --command-file "$OUTPUT/boot.cmd" --output "$OUTPUT/boot.scr"

(cd "$OUTPUT" && sha256sum Image system-16gb.dtb rootfs.cpio.gz boot.scr > SHA256SUMS)
printf '%s\n' "$LINUX_COMMIT" > "$OUTPUT/LINUX_COMMIT"
printf '%s\n' "$BUSYBOX_COMMIT" > "$OUTPUT/BUSYBOX_COMMIT"
{
    echo "host_source_iface=$HOST_SOURCE_IFACE"
    echo "host_reachable_ip=$HOST_REACHABLE_IP"
    echo "board_address=$BOARD_ADDRESS"
    echo "profile_mode=$PROFILE_MODE"
    echo "pl_accel_mode=$PL_ACCEL_MODE"
} > "$OUTPUT/NETWORK_CONFIG"
if [ -d "$ROOT/third_party/host-tools/packages" ]; then
    sha256sum "$ROOT"/third_party/host-tools/packages/*.deb > "$OUTPUT/HOST_TOOL_SHA256SUMS"
fi
echo "RAM_LINUX_BUILD_PASS output=$OUTPUT memory_reg=$memory_reg initramfs_bytes=$ramdisk_size host_ip=$HOST_REACHABLE_IP board=$BOARD_ADDRESS"
