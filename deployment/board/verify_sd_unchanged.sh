#!/bin/sh
# End-of-run SD immutability attestation. It never writes the media.
set -eu

MMC_DEVICE="${MMC_DEVICE:-/dev/mmcblk0}"
RUN_DIR="${ROBOBRAIN_RUN_DIR:-/run/robobrain}"
AUDIT="$RUN_DIR/sd-readonly.audit"
die() { echo "SD_FINAL_FAIL: $*" >&2; exit 1; }

[ -f "$RUN_DIR/sd-readonly.ok" ] && [ -r "$AUDIT" ] || die "initial audit missing"
state="$(awk -F= '$1 == "state" {print $2}' "$AUDIT")"
if [ "$state" = absent ]; then
    [ ! -b "$MMC_DEVICE" ] || die "SD device appeared after absent-state audit"
    echo "SD_FINAL_PASS state=absent device=$MMC_DEVICE media_sha256=absent write_counters_unchanged=1"
    exit 0
fi
[ "$state" = read-only ] || die "unknown initial SD state"
[ "$(blockdev --getro "$MMC_DEVICE")" = 1 ] || die "device is no longer read-only"
if awk -v dev="$MMC_DEVICE" 'index($1, dev) == 1 && ("," $4 ",") ~ /,rw,/ { found=1 } END { exit !found }' /proc/mounts; then
    die "an SD filesystem is mounted read-write"
fi
sectors="$(blockdev --getsz "$MMC_DEVICE")"
tail_skip=$((sectors - 8192))
head_hash="$(dd if="$MMC_DEVICE" bs=512 count=8192 2>/dev/null | sha256sum | awk '{print $1}')"
tail_hash="$(dd if="$MMC_DEVICE" bs=512 skip="$tail_skip" count=8192 2>/dev/null | sha256sum | awk '{print $1}')"
media_hash="$(sha256sum "$MMC_DEVICE" | awk '{print $1}')"
expected_head="$(awk -F= '$1 == "head_4m_sha256" {print $2}' "$AUDIT")"
expected_tail="$(awk -F= '$1 == "tail_4m_sha256" {print $2}' "$AUDIT")"
expected_media="$(awk -F= '$1 == "media_sha256" {print $2}' "$AUDIT")"
[ "$head_hash" = "$expected_head" ] || die "first 4MiB changed"
[ "$tail_hash" = "$expected_tail" ] || die "last 4MiB changed"
[ "$media_hash" = "$expected_media" ] || die "full-device SHA-256 changed"
base="${MMC_DEVICE##*/}"
for field in write_ios:5 write_sectors:7 discard_ios:12 flush_ios:16; do
    name="${field%:*}"; number="${field#*:}"
    expected="$(awk -F= -v name="$name" '$1 == name {print $2}' "$AUDIT")"
    current="$(awk -v number="$number" '{print $(number)+0}' "/sys/block/$base/stat")"
    [ "$current" = "$expected" ] || die "$name changed ($expected -> $current)"
done
echo "SD_FINAL_PASS device=$MMC_DEVICE media_sha256=$media_hash write_counters_unchanged=1"
