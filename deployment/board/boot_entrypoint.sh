#!/bin/sh
# Initramfs entrypoint for unattended, zero-SD-write deployment.
set -eu

BOARD_DIR="${ROBOBRAIN_BOARD_DIR:-/opt/robobrain}"
LOG="/dev/shm/robobrain-boot.log"
HOST_IP="${HOST_IP:-172.21.51.222}"
LOG_PORT="${LOG_PORT:-9000}"

exec 3>/dev/console
exec >> "$LOG" 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo "ROBOBRAIN_BOOT_FAIL rc=$rc"; echo "ROBOBRAIN_UART_LOG_REPLAY_BEGIN" >&3; tail -n 120 "$LOG" >&3 2>/dev/null || true; echo "ROBOBRAIN_UART_LOG_REPLAY_END" >&3; fi' 0
echo "ROBOBRAIN_BOOT_BEGIN"
devmem 0x7FF00000 32 0x52424201 || true
"$BOARD_DIR/early_network.sh"
devmem 0x7FF00000 32 0x52424202 || true

# Keep a second, Ethernet-backed log path so failures before model loading are
# observable remotely even if the UART client disconnects.
if command -v nc >/dev/null 2>&1; then
    (
        while :; do
            tail -n +1 -f "$LOG" | nc "$HOST_IP" "$LOG_PORT" || true
            sleep 1
        done
    ) &
fi
"$BOARD_DIR/readonly_sd_guard.sh"
devmem 0x7FF00000 32 0x52424203 || true
"$BOARD_DIR/prepare_runtime.sh"
"$BOARD_DIR/inventory.sh"
if [ "${ROBOBRAIN_PROFILE_MODE:-0}" = 1 ]; then
    echo "DDR_BW_BEGIN"
    "$BOARD_DIR/ddr_bandwidth" 256 4
    echo "DDR_BW_END"
fi
"$BOARD_DIR/final_attestation_server.sh"
"$BOARD_DIR/fetch_artifacts.sh"
devmem 0x7FF00000 32 0x52424204 || true
trap - 0
exec "$BOARD_DIR/start_text_server.sh"
