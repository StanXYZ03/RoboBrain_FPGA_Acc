#!/bin/sh
# Establish the only remote observability channel before any long SD hashing.
set -eu

BOARD_IFACE="${BOARD_IFACE:-eth0}"
BOARD_ADDRESS="${BOARD_ADDRESS:-172.21.127.254/17}"

command -v ip >/dev/null 2>&1 || {
    echo "EARLY_NET_FAIL: ip command is required" >&2
    exit 1
}
ip link set "$BOARD_IFACE" up
ip address replace "$BOARD_ADDRESS" dev "$BOARD_IFACE"

# Link negotiation can finish asynchronously; later HTTP retries tolerate it.
echo "EARLY_NET_PASS iface=$BOARD_IFACE address=$BOARD_ADDRESS"
ip address show dev "$BOARD_IFACE" || true
