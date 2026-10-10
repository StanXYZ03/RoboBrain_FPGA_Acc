#!/usr/bin/env python3
"""Minimal dependency-free UART console with a byte-exact server-side log."""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
from pathlib import Path


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    serial_fd = os.open(args.device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    configure_uart(serial_fd, args.baud)
    stdin_fd = sys.stdin.fileno()
    readers = [serial_fd, stdin_fd]

    print(
        f"UART_CONSOLE_READY device={args.device} baud={args.baud} "
        f"log={args.log}",
        flush=True,
    )
    try:
        with args.log.open("wb", buffering=0) as log:
            while True:
                ready, _, _ = select.select(readers, [], [])
                if serial_fd in ready:
                    try:
                        block = os.read(serial_fd, 65536)
                    except BlockingIOError:
                        block = b""
                    if block:
                        log.write(block)
                        sys.stdout.buffer.write(block)
                        sys.stdout.buffer.flush()
                if stdin_fd in ready:
                    block = os.read(stdin_fd, 4096)
                    if block:
                        os.write(serial_fd, block)
                    else:
                        readers.remove(stdin_fd)
    finally:
        os.close(serial_fd)


if __name__ == "__main__":
    raise SystemExit(main())
