#!/bin/sh
# Add only deployment scripts to an unpacked RAM initramfs tree.
set -eu

[ "$#" -eq 1 ] || { echo "usage: $0 UNPACKED_INITRAMFS_ROOT" >&2; exit 2; }
ROOTFS="$1"
PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
[ -d "$ROOTFS" ] || { echo "not a directory: $ROOTFS" >&2; exit 1; }
[ -x "$ROOTFS/bin/sh" ] || { echo "initramfs must contain /bin/sh" >&2; exit 1; }
for command in blockdev dd sha256sum awk mount ip df tail nc wget httpd setsid cttyhack; do
    [ -x "$ROOTFS/bin/$command" ] || [ -x "$ROOTFS/sbin/$command" ] || \
        [ -x "$ROOTFS/usr/bin/$command" ] || [ -x "$ROOTFS/usr/sbin/$command" ] || {
        echo "initramfs missing required command: $command" >&2
        exit 1
    }
done
install -d -m 0755 "$ROOTFS/opt/robobrain"
for script in early_network.sh readonly_sd_guard.sh inventory.sh prepare_runtime.sh fetch_artifacts.sh start_text_server.sh verify_sd_unchanged.sh final_attestation_server.sh boot_entrypoint.sh start_uart_shell.sh; do
    install -m 0755 "$PROJECT_ROOT/deployment/board/$script" "$ROOTFS/opt/robobrain/$script"
done
if [ -f "$PROJECT_ROOT/deployment/artifacts/arm64/ddr_bandwidth" ]; then
    install -m 0755 "$PROJECT_ROOT/deployment/artifacts/arm64/ddr_bandwidth" \
        "$ROOTFS/opt/robobrain/ddr_bandwidth"
fi
if [ -f "$PROJECT_ROOT/deployment/artifacts/arm64/llama-server-profile" ]; then
    install -m 0755 "$PROJECT_ROOT/deployment/artifacts/arm64/llama-server-profile" \
        "$ROOTFS/opt/robobrain/llama-server-profile"
fi
echo "INITRAMFS_STAGE_PASS root=$ROOTFS"
echo "Configure the RAM init system to run /opt/robobrain/boot_entrypoint.sh after mounting proc, sysfs, devtmpfs, /run tmpfs and /dev/shm tmpfs."
