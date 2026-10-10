#!/usr/bin/env python3
"""Receive board boot/runtime logs and persist them only on the server."""

from __future__ import annotations

import argparse
import socket
import socketserver
import threading
from pathlib import Path


class LogHandler(socketserver.StreamRequestHandler):
    output: Path
    lock = threading.Lock()

    def handle(self) -> None:
        with self.output.open("ab", buffering=0) as stream:
            while block := self.request.recv(65536):
                # Do not hold the lock for the lifetime of a board connection.
                # A stale reconnect must not prevent the active boot from
                # appending live inference-stage observations.
                with self.lock:
                    stream.write(block)


class DeviceBoundTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, handler, interface: str):
        self.interface = interface
        super().__init__(server_address, handler)

    def server_bind(self) -> None:
        if self.interface:
            self.socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_BINDTODEVICE,
                self.interface.encode("ascii") + b"\0",
            )
        super().server_bind()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", required=True)
    parser.add_argument("--interface", default="enp4s0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    LogHandler.output = args.output
    with DeviceBoundTCPServer(
        (args.bind, args.port), LogHandler, args.interface
    ) as server:
        print(
            f"receiving board logs on {args.bind}:{args.port} via "
            f"{args.interface} -> {args.output}"
        )
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
