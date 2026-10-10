#!/bin/sh
set -eu

RUN_DIR="${ROBOBRAIN_RUN_DIR:-/run/robobrain}"
RUNTIME_DIR="${ROBOBRAIN_RUNTIME_DIR:-/dev/shm/robobrain}"
MMC_DEVICE="${MMC_DEVICE:-/dev/mmcblk0}"
BOARD_ADDRESS="${BOARD_ADDRESS:-172.21.127.254/17}"
BOARD_IP="${BOARD_IP:-${BOARD_ADDRESS%/*}}"
PORT="${PORT:-8080}"
BOARD_DIR="${ROBOBRAIN_BOARD_DIR:-/opt/robobrain}"
PROFILE_MODE="${ROBOBRAIN_PROFILE_MODE:-0}"

die() { echo "SERVER_FAIL: $*" >&2; exit 1; }
[ -f "$RUN_DIR/sd-readonly.ok" ] || die "SD guard marker missing"
if [ -b "$MMC_DEVICE" ]; then
    [ "$(blockdev --getro "$MMC_DEVICE")" = 1 ] || die "SD is no longer read-only"
else
    grep -qx 'state=absent' "${ROBOBRAIN_RUN_DIR:-/run/robobrain}/sd-readonly.audit" \
        || die "missing absent-SD audit"
fi
case "$PROFILE_MODE" in
    0) SERVER_BIN="$RUNTIME_DIR/llama-server" ;;
    1) SERVER_BIN="$BOARD_DIR/llama-server-profile" ;;
    *) die "invalid profile mode: $PROFILE_MODE" ;;
esac
[ -x "$SERVER_BIN" ] || die "llama-server missing: $SERVER_BIN"
[ -f "$RUNTIME_DIR/RoboBrain2.5-4B.Q8_0.gguf" ] || die "Q8_0 model missing"
[ -f "$RUNTIME_DIR/RoboBrain2.5-4B.mmproj-Q8_0.gguf" ] || die "Q8_0 mmproj missing"
(cd "$RUNTIME_DIR" && sha256sum -c artifacts.sha256) || die "artifact hash check failed"

echo "SERVER_START machine=$(uname -m) mem_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
export GGML_CPU_PROFILE="$PROFILE_MODE"
echo "SERVER_PROFILE mode=$PROFILE_MODE binary=$SERVER_BIN"
exec "$SERVER_BIN" \
    -m "$RUNTIME_DIR/RoboBrain2.5-4B.Q8_0.gguf" \
    --mmproj "$RUNTIME_DIR/RoboBrain2.5-4B.mmproj-Q8_0.gguf" \
    --no-mmproj-offload \
    --image-min-tokens 1024 \
    --image-max-tokens 1024 \
    --ctx-size 2048 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --threads 4 \
    --threads-batch 4 \
    --parallel 1 \
    --mlock \
    --jinja \
    --metrics \
    --slots \
    --perf \
    --log-timestamps \
    --host "$BOARD_IP" \
    --port "$PORT"
