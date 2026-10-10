#!/usr/bin/env python3
"""Trigger the board's read-only end-of-run SD attestation endpoint."""

from __future__ import annotations

import argparse
import http.client
import socket
from urllib.parse import urlsplit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://172.21.127.254:9001/cgi-bin/final")
    parser.add_argument("--interface", default="enp4s0")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    parsed = urlsplit(args.url)
    connection = http.client.HTTPConnection(
        parsed.hostname, parsed.port or 80, timeout=args.timeout
    )
    connection.connect = lambda: _connect_bound(connection, args.interface)
    try:
        connection.request("GET", parsed.path or "/")
        response = connection.getresponse()
        body = response.read().decode("utf-8", errors="replace")
    finally:
        connection.close()
    print(body, end="" if body.endswith("\n") else "\n")
    if "SD_FINAL_PASS" not in body:
        raise SystemExit("FINAL_SD_CHECK_FAIL: pass marker missing")
    print("FINAL_SD_CHECK_PASS")
    return 0


def _connect_bound(connection: http.client.HTTPConnection, interface: str) -> None:
    connection.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.sock.settimeout(connection.timeout)
    connection.sock.setsockopt(
        socket.SOL_SOCKET, socket.SO_BINDTODEVICE,
        interface.encode("ascii") + b"\0",
    )
    connection.sock.connect((connection.host, connection.port))


if __name__ == "__main__":
    raise SystemExit(main())
