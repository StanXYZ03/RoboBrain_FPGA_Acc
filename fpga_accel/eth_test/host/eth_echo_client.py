#!/usr/bin/env python3
import argparse
import socket
import struct
import sys


def fnv1a(data: bytes) -> int:
    value = 2166136261
    for byte in data:
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def request(sequence: int) -> bytes:
    header = b"RBET" + bytes((1, 1, 0, 0)) + struct.pack("<I", sequence)
    return header + struct.pack("<I", fnv1a(header))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="255.255.255.255")
    parser.add_argument("--port", type=int, default=7007)
    parser.add_argument("--board-port", type=int, default=7008)
    parser.add_argument("--sequence", type=lambda value: int(value, 0),
                        default=0x10203040)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--interface", default="enp4s0")
    parser.add_argument("--source", default="0.0.0.0")
    args = parser.parse_args()

    expected = request(args.sequence)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind((args.source, args.port))
        sock.settimeout(args.timeout)
        received, peer = sock.recvfrom(2048)
        if received != expected:
            print(f"HOST_ETH_RESULT=FAIL expected={expected.hex()} "
                  f"actual={received.hex()} peer={peer}")
            return 1
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE,
                        args.interface.encode() + b"\0")
        sock.sendto(received, (args.host, args.board_port))

    print(f"HOST_ETH_RESULT=PASS bytes={len(received)} "
          f"sequence=0x{args.sequence:08X} peer={peer[0]}:{peer[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
