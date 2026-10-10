#!/bin/sh
# Expose read-only CGI endpoints for SD attestation and runtime observations.
set -eu

BOARD_DIR="${ROBOBRAIN_BOARD_DIR:-/opt/robobrain}"
CONTROL_ROOT="${ROBOBRAIN_CONTROL_ROOT:-/dev/shm/robobrain-control}"
CONTROL_PORT="${ROBOBRAIN_CONTROL_PORT:-9001}"

command -v httpd >/dev/null 2>&1 || {
    echo "ATTEST_SERVER_FAIL: BusyBox httpd is required" >&2
    exit 1
}
mkdir -p "$CONTROL_ROOT/cgi-bin"
cat > "$CONTROL_ROOT/cgi-bin/final" <<EOF
#!/bin/sh
printf 'Content-Type: text/plain\r\n\r\n'
exec "$BOARD_DIR/verify_sd_unchanged.sh"
EOF
chmod 0700 "$CONTROL_ROOT/cgi-bin/final"
cat > "$CONTROL_ROOT/cgi-bin/runtime" <<'EOF'
#!/bin/sh
printf 'Content-Type: text/plain\r\n\r\n'
printf 'runtime_status=ok\n'
printf 'uptime_seconds='
cut -d' ' -f1 /proc/uptime
awk '/^(MemTotal|MemAvailable):/ {print tolower($1) "=" $2 " " $3}' /proc/meminfo
printf 'loadavg='
cat /proc/loadavg
printf '%s\n' 'log_tail_begin'
tail -n 200 /dev/shm/robobrain-boot.log 2>/dev/null || true
printf '%s\n' 'log_tail_end'
EOF
chmod 0700 "$CONTROL_ROOT/cgi-bin/runtime"
httpd -f -p "$CONTROL_PORT" -h "$CONTROL_ROOT" &
echo "ATTEST_SERVER_PASS port=$CONTROL_PORT endpoints=/cgi-bin/final,/cgi-bin/runtime"
