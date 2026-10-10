#!/bin/sh
# Boot the real ARM64 Image/initramfs on a virtual machine before touching HW.
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
ARTIFACTS="${RAM_LINUX_OUTPUT:-$ROOT/deployment/artifacts/ram_linux}"
QEMU="${QEMU_SYSTEM_AARCH64:-/home/xilinx/Vitis/2023.2/data/emulation/qemu/comp/qemu/sysroots/x86_64-petalinux-linux/usr/bin/qemu-system-aarch64}"
REPORT="${RAM_LINUX_QEMU_REPORT:-$ROOT/deployment/reports/ram-linux-qemu.log}"

[ -x "$QEMU" ] || { echo "missing qemu-system-aarch64: $QEMU" >&2; exit 1; }
for file in Image rootfs.cpio.gz; do
    [ -r "$ARTIFACTS/$file" ] || { echo "missing $ARTIFACTS/$file" >&2; exit 1; }
done
mkdir -p "$(dirname -- "$REPORT")"
set +e
timeout 120 "$QEMU" \
    -machine virt -cpu cortex-a53 -smp 4 -m 2048 \
    -display none -serial stdio -no-reboot \
    -kernel "$ARTIFACTS/Image" -initrd "$ARTIFACTS/rootfs.cpio.gz" \
    -append 'console=ttyAMA0 earlycon=pl011,0x09000000 rdinit=/init robobrain.selftest=1' \
    > "$REPORT" 2>&1
rc=$?
set -e
grep -q 'RAM_LINUX_QEMU_PASS' "$REPORT" || {
    tail -100 "$REPORT" >&2
    echo "RAM_LINUX_QEMU_FAIL rc=$rc report=$REPORT" >&2
    exit 1
}
echo "RAM_LINUX_QEMU_PASS report=$REPORT"
