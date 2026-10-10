#!/usr/bin/env python3
"""Push one gzip-compressed artifact to RAM over UART and run it.

The remote path is restricted to /tmp and the script never invokes SD/QSPI
utilities.  The UART output log is append-only evidence.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import os
import re
import select
import sys
import termios
import time
from pathlib import Path


FAIL_MARKERS = (b"PREP_BENCH_FAIL", b"UART_PUSH_DONE status=1")


def configure_uart(fd: int, baud: int) -> None:
    speeds = {
        9600: termios.B9600,
        19200: termios.B19200,
        38400: termios.B38400,
        57600: termios.B57600,
        115200: termios.B115200,
    }
    if baud not in speeds:
        raise ValueError(f"unsupported baud rate: {baud}")
    attrs = termios.tcgetattr(fd)
    attrs[0] = termios.IGNBRK
    attrs[1] = 0
    attrs[2] = speeds[baud] | termios.CS8 | termios.CREAD | termios.CLOCAL
    attrs[3] = 0
    attrs[4] = speeds[baud]
    attrs[5] = speeds[baud]
    attrs[6][termios.VMIN] = 1
    attrs[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    termios.tcflush(fd, termios.TCIFLUSH)


def send_all(fd: int, payload: bytes, deadline: float) -> None:
    offset = 0
    while offset < len(payload):
        if time.monotonic() >= deadline:
            raise TimeoutError("UART write timed out")
        _, writable, _ = select.select([], [fd], [], 0.5)
        if not writable:
            continue
        try:
            written = os.write(fd, payload[offset : offset + 4096])
        except BlockingIOError:
            continue
        offset += written


def make_payload(local: Path, remote: str) -> tuple[bytes, str, int]:
    content = local.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    encoded = base64.b64encode(gzip.compress(content, compresslevel=9, mtime=0))
    lines = b"\n".join(encoded[index : index + 76]
                       for index in range(0, len(encoded), 76))
    command = (
        b"\003\r"
        b"stty -echo\r"
        + f"base64 -d <<'RB_UART_EOF' | gzip -d > {remote}\n".encode()
        + lines
        + b"\nRB_UART_EOF\n"
        + b"stty echo\n"
        + f"chmod 700 {remote} && {remote}; rb_status=$?; ".encode()
        + f"sha256sum {remote}; ".encode()
        + b"echo UART_PUSH_DONE status=$rb_status\n"
    )
    return command, digest, len(content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/ttyUSB1")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--remote", default="/tmp/prepare-benchmark")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--pass-marker", default="PREP_BENCH_PASS")
    args = parser.parse_args()
    if not args.local.is_file():
        raise RuntimeError(f"local artifact not found: {args.local}")
    if not re.fullmatch(r"/tmp/[A-Za-z0-9._-]+", args.remote):
        raise RuntimeError("remote path must be a simple /tmp filename")
    if args.log.exists():
        raise RuntimeError(f"refusing to overwrite append-only log: {args.log}")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    payload, expected_digest, source_bytes = make_payload(args.local, args.remote)
    pass_marker = args.pass_marker.encode("ascii")
    deadline = time.monotonic() + args.timeout
    fd = os.open(args.device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    configure_uart(fd, args.baud)
    print(
        f"UART_PUSH_READY device={args.device} remote={args.remote} "
        f"source_bytes={source_bytes} wire_bytes={len(payload)} "
        f"sha256={expected_digest}",
        flush=True,
    )
    recent = b""
    saw_pass = False
    try:
        with args.log.open("xb", buffering=0) as log:
            send_all(fd, payload, deadline)
            print("UART_PUSH_SENT destination=ram_only", flush=True)
            while time.monotonic() < deadline:
                ready, _, _ = select.select([fd], [], [], 1.0)
                if not ready:
                    continue
                try:
                    block = os.read(fd, 65536)
                except BlockingIOError:
                    continue
                if not block:
                    continue
                log.write(block)
                sys.stdout.buffer.write(block)
                sys.stdout.buffer.flush()
                recent = (recent + block)[-262144:]
                if pass_marker in recent:
                    saw_pass = True
                if any(marker in recent for marker in FAIL_MARKERS):
                    print("UART_PUSH_FAIL reason=board_marker", file=sys.stderr)
                    return 1
                done = re.search(rb"UART_PUSH_DONE status=(\d+)", recent)
                if done:
                    status = int(done.group(1))
                    digest_seen = expected_digest.encode() in recent
                    if status == 0 and saw_pass and digest_seen:
                        print(
                            "UART_PUSH_PASS marker=1 digest=1 "
                            "destination=ram_only sd_qspi_written=no",
                            flush=True,
                        )
                        return 0
                    print(
                        f"UART_PUSH_FAIL status={status} pass={int(saw_pass)} "
                        f"digest={int(digest_seen)}",
                        file=sys.stderr,
                    )
                    return 1
            print("UART_PUSH_FAIL reason=timeout", file=sys.stderr)
            return 1
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
