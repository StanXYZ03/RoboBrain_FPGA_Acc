#!/bin/sh
# Prepare volatile DDR and networking. No persistent filesystem is touched.
set -eu

RUN_DIR="${ROBOBRAIN_RUN_DIR:-/run/robobrain}"
RUNTIME_DIR="${ROBOBRAIN_RUNTIME_DIR:-/dev/shm/robobrain}"
BOARD_IFACE="${BOARD_IFACE:-eth0}"
BOARD_ADDRESS="${BOARD_ADDRESS:-172.21.127.254/17}"
# The 16 GiB DIMM is fully described by the DT (16776192 KiB at boot), while
# ZynqMP firmware/reserved regions and CMA reduce /proc/meminfo MemTotal to
# about 14310760 KiB.  Keep a high gate that still rejects the factory 4 GiB.
MIN_MEM_KIB="${MIN_MEM_KIB:-13500000}"
TMPFS_SIZE="${ROBOBRAIN_TMPFS_SIZE:-8G}"

die() { echo "PREPARE_FAIL: $*" >&2; exit 1; }
[ -f "$RUN_DIR/sd-readonly.ok" ] || die "SD guard marker missing"
MMC_DEVICE="${MMC_DEVICE:-/dev/mmcblk0}"
if [ -b "$MMC_DEVICE" ]; then
    [ "$(blockdev --getro "$MMC_DEVICE")" = 1 ] || die "SD is no longer read-only"
else
    grep -qx 'state=absent' "$RUN_DIR/sd-readonly.audit" || die "missing absent-SD audit"
fi

machine="$(uname -m)"
[ "$machine" = aarch64 ] || die "expected aarch64, got $machine"
mem_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
[ "$mem_kib" -ge "$MIN_MEM_KIB" ] || die "Linux sees only ${mem_kib} KiB RAM"

awk '$2 == "/dev/shm" && $3 == "tmpfs" { ok=1 } END { exit !ok }' /proc/mounts \
    || die "/dev/shm is not tmpfs"
mkdir -p "$RUNTIME_DIR"
if ! awk -v path="$RUNTIME_DIR" '$2 == path && $3 == "tmpfs" { ok=1 } END { exit !ok }' /proc/mounts; then
    mount -t tmpfs -o "size=$TMPFS_SIZE,mode=0700,nosuid,nodev" robobrain-ddr "$RUNTIME_DIR"
fi

if [ -r /proc/swaps ] && [ "$(awk 'NR > 1 {n++} END {print n+0}' /proc/swaps)" -ne 0 ]; then
    swapoff -a
fi
command -v ip >/dev/null 2>&1 || die "ip command is required"
ip link set "$BOARD_IFACE" up
# early_network.sh normally installed this address already.  BusyBox ip may
# implement "replace" as an add and return EEXIST on the second call, which is
# fatal under set -e.  Only repair the address when it is actually missing.
if ! ip address show dev "$BOARD_IFACE" | grep -Fq "inet $BOARD_ADDRESS "; then
    ip address flush dev "$BOARD_IFACE"
    ip address add "$BOARD_ADDRESS" dev "$BOARD_IFACE"
fi

echo "runtime_dir=$RUNTIME_DIR" > "$RUN_DIR/runtime.env"
echo "board_iface=$BOARD_IFACE" >> "$RUN_DIR/runtime.env"
echo "board_address=$BOARD_ADDRESS" >> "$RUN_DIR/runtime.env"
echo "PREPARE_PASS mem_kib=$mem_kib runtime=$RUNTIME_DIR address=$BOARD_ADDRESS"
df -k "$RUNTIME_DIR"
ip address show dev "$BOARD_IFACE"
