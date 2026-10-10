#!/usr/bin/env python3
"""Capture a fresh UART boot, launch the staged protocol, and stop on its gate."""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
from pathlib import Path


BOOT_READY = b"RAM_LINUX_PL_SELFTEST_HOLD uart_shell=enabled"
BOOT_FAILURES = (b"RAM_LINUX_PL_SELFTEST_FAIL", b"RAM_LINUX_INIT_FAIL")
MEASURE_PASS = b"DATAFLOW_MEASURE_PASS protocol=1"
MEASURE_FAIL = b"DATAFLOW_MEASURE_FAIL"
COMMAND = b"/opt/robobrain/q8q8-dataflow-measure\r"


def configure_uart(fd: int, baud: int) -> None:
    speeds = {9600: termios.B9600, 19200: termios.B19200, 38400: termios.B38400,
              57600: termios.B57600, 115200: termios.B115200}
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--boot-timeout", type=float, default=600.0)
    parser.add_argument("--measure-timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if args.log.exists():
        raise RuntimeError(f"refusing to overwrite append-only UART log: {args.log}")
    args.log.parent.mkdir(parents=True, exist_ok=True)

    fd = os.open(args.device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    configure_uart(fd, args.baud)
    phase = "boot"
    deadline = time.monotonic() + args.boot_timeout
    recent = b""
    print(f"DATAFLOW_UART_CAPTURE_READY device={args.device} log={args.log}", flush=True)
    try:
        with args.log.open("xb", buffering=0) as log:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    print(f"DATAFLOW_UART_CAPTURE_FAIL phase={phase} reason=timeout", file=sys.stderr)
                    return 1
                ready, _, _ = select.select([fd], [], [], min(1.0, remaining))
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
                recent = (recent + block)[-131072:]
                if phase == "boot":
                    if any(marker in recent for marker in BOOT_FAILURES):
                        print("DATAFLOW_UART_CAPTURE_FAIL phase=boot reason=board_marker", file=sys.stderr)
                        return 1
                    if BOOT_READY in recent:
                        os.write(fd, b"\r" + COMMAND)
                        phase = "measure"
                        deadline = time.monotonic() + args.measure_timeout
                        recent = b""
                        print("DATAFLOW_UART_COMMAND_SENT protocol=1", flush=True)
                else:
                    if MEASURE_FAIL in recent:
                        print("DATAFLOW_UART_CAPTURE_FAIL phase=measure reason=board_marker", file=sys.stderr)
                        return 1
                    if MEASURE_PASS in recent:
                        print("DATAFLOW_UART_CAPTURE_PASS protocol=1", flush=True)
                        return 0
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())

