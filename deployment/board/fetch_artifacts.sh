#!/bin/sh
# Pull exactly the allow-listed artifacts into DDR tmpfs and verify SHA-256.
set -eu

RUN_DIR="${ROBOBRAIN_RUN_DIR:-/run/robobrain}"
RUNTIME_DIR="${ROBOBRAIN_RUNTIME_DIR:-/dev/shm/robobrain}"
HOST_IP="${HOST_IP:-172.21.51.222}"
ARTIFACT_URL="${ARTIFACT_URL:-http://$HOST_IP:8000}"
FILES="llama-server RoboBrain2.5-4B.Q8_0.gguf RoboBrain2.5-4B.mmproj-Q8_0.gguf"

die() { echo "FETCH_FAIL: $*" >&2; exit 1; }
[ -f "$RUN_DIR/sd-readonly.ok" ] || die "SD guard marker missing"
MMC_DEVICE="${MMC_DEVICE:-/dev/mmcblk0}"
if [ -b "$MMC_DEVICE" ]; then
    [ "$(blockdev --getro "$MMC_DEVICE")" = 1 ] || die "SD is no longer read-only"
else
    grep -qx 'state=absent' "$RUN_DIR/sd-readonly.audit" || die "missing absent-SD audit"
fi
awk -v path="$RUNTIME_DIR" '$2 == path && $3 == "tmpfs" { ok=1 } END { exit !ok }' /proc/mounts \
    || die "$RUNTIME_DIR is not a dedicated tmpfs"

fetch() {
    url="$1"; output="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 -C - -o "$output" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -c -O "$output" "$url"
    else
        die "curl or wget is required"
    fi
}

manifest="$RUNTIME_DIR/artifacts.sha256.part"
fetch "$ARTIFACT_URL/artifacts.sha256" "$manifest"
[ "$(wc -l < "$manifest")" -eq 3 ] || die "manifest must contain exactly three files"

for name in $FILES; do
    expected="$(awk -v name="$name" '$2 == name {print $1}' "$manifest")"
    [ "${#expected}" -eq 64 ] || die "manifest does not contain a 64-character hash for $name"
    case "$expected" in *[!0-9a-f]*) die "manifest hash is not lowercase hexadecimal for $name";; esac
    part="$RUNTIME_DIR/$name.part"
    final="$RUNTIME_DIR/$name"
    if [ -f "$final" ]; then
        actual="$(sha256sum "$final" | awk '{print $1}')"
        [ "$actual" = "$expected" ] || die "existing $name has the wrong hash"
    else
        fetch "$ARTIFACT_URL/$name" "$part"
        actual="$(sha256sum "$part" | awk '{print $1}')"
        [ "$actual" = "$expected" ] || die "$name hash mismatch"
        mv "$part" "$final"
    fi
    echo "FETCH_VERIFIED $name sha256=$expected"
done
mv "$manifest" "$RUNTIME_DIR/artifacts.sha256"
chmod 0700 "$RUNTIME_DIR/llama-server"
echo "FETCH_PASS"
