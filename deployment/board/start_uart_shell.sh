#!/bin/sh
set -eu

UART_DEVICE="${ROBOBRAIN_UART_DEVICE:-/dev/ttyPS0}"

if [ ! -c "$UART_DEVICE" ]; then
    echo "UART_SHELL_SKIP device=$UART_DEVICE reason=missing"
    exit 0
fi

echo "UART_SHELL_START device=$UART_DEVICE baud=115200"
(
    while :; do
        setsid cttyhack sh -i < "$UART_DEVICE" > "$UART_DEVICE" 2>&1 || true
        sleep 1
    done
) &
