#!/usr/bin/env python3
"""Range-capable, allow-list-only HTTP server for board DDR transfers."""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import re
import socket
from pathlib import Path
from urllib.parse import unquote, urlsplit


CHUNK = 8 * 1024 * 1024


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            value.update(block)
    return value.hexdigest()


def parse_range(header: str | None, size: int) -> tuple[int, int, int]:
    """Return (HTTP status, start, end) for a single byte range."""
    if not header:
        return 200, 0, size - 1
    match = re.fullmatch(r"bytes=(\d+)-(\d*)", header.strip())
    if not match:
        raise ValueError("invalid byte range")
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else size - 1
    if start >= size or end < start or end >= size:
        raise ValueError("byte range outside artifact")
    return 206, start, end


class ArtifactHandler(http.server.BaseHTTPRequestHandler):
    artifacts: dict[str, tuple[Path, str]] = {}
    manifest = b""
    protocol_version = "HTTP/1.1"

    def do_HEAD(self) -> None:
        self._serve(send_body=False)

    def do_GET(self) -> None:
        self._serve(send_body=True)

    def _serve(self, *, send_body: bool) -> None:
        name = unquote(urlsplit(self.path).path).lstrip("/")
        if name == "artifacts.sha256":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=ascii")
            self.send_header("Content-Length", str(len(self.manifest)))
            self.end_headers()
            if send_body:
                self.wfile.write(self.manifest)
            return
        item = self.artifacts.get(name)
        if item is None:
            self.send_error(404)
            return
        path, _ = item
        size = path.stat().st_size
        try:
            status, start, end = parse_range(self.headers.get("Range"), size)
        except ValueError:
            self.send_error(416)
            return
        count = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(count))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not send_body:
            return
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = count
            while remaining:
                block = stream.read(min(CHUNK, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"artifact-server {self.address_string()} {fmt % args}", flush=True)


class DeviceBoundHTTPServer(http.server.ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler, interface: str):
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
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--bind", required=True)
    parser.add_argument("--interface", default="enp4s0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    lock = json.loads(args.model_lock.read_text(encoding="utf-8"))
    if lock.get("official") is not False or len(lock.get("files", [])) != 2:
        raise SystemExit("invalid model lock")
    paths: dict[str, Path] = {"llama-server": args.llama_server.resolve()}
    expected = {item["name"]: item["sha256"] for item in lock["files"]}
    for name in expected:
        paths[name] = (args.model_dir / name).resolve()
    if set(paths) != {
        "llama-server", "RoboBrain2.5-4B.Q8_0.gguf",
        "RoboBrain2.5-4B.mmproj-Q8_0.gguf",
    }:
        raise SystemExit("artifact allow-list mismatch")

    artifacts = {}
    for name, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"missing artifact: {path}")
        actual = digest(path)
        if name in expected and actual != expected[name]:
            raise SystemExit(f"hash differs from model lock: {name}")
        artifacts[name] = (path, actual)
        print(f"verified {actual}  {name}")
    ArtifactHandler.artifacts = artifacts
    ArtifactHandler.manifest = "".join(
        f"{sha}  {name}\n" for name, (_, sha) in artifacts.items()
    ).encode("ascii")
    server = DeviceBoundHTTPServer(
        (args.bind, args.port), ArtifactHandler, args.interface
    )
    print(
        f"serving three allow-listed artifacts on http://{args.bind}:{args.port} "
        f"via {args.interface}"
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
