#!/bin/sh
# Read-only Linux/DDR inventory. Output is streamed to the host.
set -eu

RUN_DIR="${ROBOBRAIN_RUN_DIR:-/run/robobrain}"
[ -f "$RUN_DIR/sd-readonly.ok" ] || {
    echo "INVENTORY_FAIL: SD read-only guard has not passed" >&2
    exit 1
}

echo "=== identity ==="
uname -a
printf 'machine='; uname -m
printf 'libc='; (ldd --version 2>&1 || true) | sed -n '1p'
echo "=== cpu ==="
grep -E '^(processor|model name|CPU architecture|Features)' /proc/cpuinfo || true
echo "=== memory ==="
grep -E '^(MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo
echo "=== device-tree memory ==="
for node in /proc/device-tree/memory*; do
    [ -d "$node" ] || continue
    echo "node=$node"
    if command -v hexdump >/dev/null 2>&1; then
        hexdump -v -e '1/4 "%08x " "\n"' "$node/reg" 2>/dev/null || true
    else
        od -An -tx4 "$node/reg" 2>/dev/null || true
    fi
done
echo "=== mounts ==="
cat /proc/mounts
echo "=== sd guard ==="
cat "$RUN_DIR/sd-readonly.audit"
echo "INVENTORY_PASS"
