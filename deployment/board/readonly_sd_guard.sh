#!/bin/sh
# Establish and attest SD read-only state. Run from a RAM-backed initramfs.
set -eu

MMC_DEVICE="${MMC_DEVICE:-/dev/mmcblk0}"
RUN_DIR="${ROBOBRAIN_RUN_DIR:-/run/robobrain}"

die() {
    echo "SD_GUARD_FAIL: $*" >&2
    exit 1
}

[ "$(id -u)" = 0 ] || die "must run as root"
[ -r /proc/mounts ] || die "/proc is not mounted"

# A failed/unavailable SD controller exposes no block device.  In a RAM-only
# boot this is a safe, non-writable state; record it explicitly and require the
# device to remain absent in the final attestation.
if [ ! -b "$MMC_DEVICE" ]; then
    mkdir -p "$RUN_DIR"
    {
        echo "schema=1"
        echo "state=absent"
        echo "device=$MMC_DEVICE"
        echo "media_sha256=absent"
        echo "write_ios=0"
        echo "write_sectors=0"
        echo "discard_ios=0"
        echo "flush_ios=0"
    } > "$RUN_DIR/sd-readonly.audit"
    touch "$RUN_DIR/sd-readonly.ok"
    echo "SD_GUARD_PASS state=absent device=$MMC_DEVICE"
    exit 0
fi

# The guard is deliberately unable to 'repair' an unsafe boot. A writable SD
# mount or SD swap means writes may already have occurred, so fail immediately.
if awk -v dev="$MMC_DEVICE" '
    index($1, dev) == 1 && ("," $4 ",") ~ /,rw,/ { found=1 }
    END { exit !found }
' /proc/mounts; then
    die "an SD filesystem is mounted read-write; reboot via RAM initramfs"
fi
if [ -r /proc/swaps ] && awk -v dev="$MMC_DEVICE" 'NR > 1 && index($1, dev) == 1 { found=1 } END { exit !found }' /proc/swaps; then
    die "SD-backed swap is active"
fi

command -v blockdev >/dev/null 2>&1 || die "blockdev is required"
blockdev --setro "$MMC_DEVICE"
[ "$(blockdev --getro "$MMC_DEVICE")" = 1 ] || die "kernel did not lock SD read-only"

base="${MMC_DEVICE##*/}"
[ -r "/sys/block/$base/ro" ] || die "missing sysfs read-only state"
[ "$(cat "/sys/block/$base/ro")" = 1 ] || die "sysfs does not report read-only"

# The marker and audit evidence must be placed in RAM, never on the rootfs.
awk '$2 == "/run" && $3 == "tmpfs" { ok=1 } END { exit !ok }' /proc/mounts \
    || die "/run is not a dedicated tmpfs"
mkdir -p "$RUN_DIR"

sectors="$(blockdev --getsz "$MMC_DEVICE")"
[ "$sectors" -gt 16384 ] || die "implausible SD size"
tail_skip=$((sectors - 8192))
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"
head_hash="$(dd if="$MMC_DEVICE" bs=512 count=8192 2>/dev/null | sha256sum | awk '{print $1}')"
tail_hash="$(dd if="$MMC_DEVICE" bs=512 skip="$tail_skip" count=8192 2>/dev/null | sha256sum | awk '{print $1}')"
media_hash="$(sha256sum "$MMC_DEVICE" | awk '{print $1}')"
write_ios="$(awk '{print $5+0}' "/sys/block/$base/stat")"
write_sectors="$(awk '{print $7+0}' "/sys/block/$base/stat")"
discard_ios="$(awk '{print $12+0}' "/sys/block/$base/stat")"
flush_ios="$(awk '{print $16+0}' "/sys/block/$base/stat")"

{
    echo "schema=1"
    echo "state=read-only"
    echo "device=$MMC_DEVICE"
    echo "sectors=$sectors"
    echo "head_4m_sha256=$head_hash"
    echo "tail_4m_sha256=$tail_hash"
    echo "media_sha256=$media_hash"
    echo "write_ios=$write_ios"
    echo "write_sectors=$write_sectors"
    echo "discard_ios=$discard_ios"
    echo "flush_ios=$flush_ios"
    echo "sysfs_ro=1"
    echo "blockdev_ro=1"
} > "$RUN_DIR/sd-readonly.audit"
touch "$RUN_DIR/sd-readonly.ok"

echo "SD_GUARD_PASS device=$MMC_DEVICE media_sha256=$media_hash"
cat /proc/mounts
