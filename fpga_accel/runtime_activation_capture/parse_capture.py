#!/usr/bin/env python3
"""Validate and describe an append-only GGML CPU F32 activation capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


HEADER = struct.Struct("<8sIIQiI4qQ64s")
MAGIC = b"RBAC0001"
VERSION = 1
GGML_TYPE_F32 = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_capture(path: Path, scan_finite: bool = True) -> dict:
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw_header = handle.read(HEADER.size)
        if len(raw_header) != HEADER.size:
            raise ValueError("truncated capture header")
        unpacked = HEADER.unpack(raw_header)
        magic, version, tensor_type, graph_id, node_index, reserved = unpacked[:6]
        ne = tuple(int(item) for item in unpacked[6:10])
        payload_bytes = int(unpacked[10])
        name = unpacked[11].split(b"\0", 1)[0].decode("utf-8")
        if magic != MAGIC or version != VERSION or reserved != 0:
            raise ValueError("invalid capture identity/header")
        if tensor_type != GGML_TYPE_F32:
            raise ValueError(f"capture tensor type is {tensor_type}, expected F32")
        if any(item <= 0 for item in ne):
            raise ValueError(f"invalid dimensions {ne}")
        elements = math.prod(ne)
        if payload_bytes != elements * 4:
            raise ValueError(f"payload size {payload_bytes} does not match F32 dimensions {ne}")
        if size != HEADER.size + payload_bytes:
            raise ValueError(f"file size {size} does not match header payload {payload_bytes}")
        finite = None
        minimum = None
        maximum = None
        if scan_finite:
            finite = True
            for _ in range(elements):
                raw = handle.read(4)
                if len(raw) != 4:
                    raise ValueError("truncated F32 payload")
                value = struct.unpack("<f", raw)[0]
                if not math.isfinite(value):
                    finite = False
                minimum = value if minimum is None else min(minimum, value)
                maximum = value if maximum is None else max(maximum, value)
        return {
            "schema_version": 1,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "header_bytes": HEADER.size,
            "payload_bytes": payload_bytes,
            "ggml_type": "F32",
            "graph_id": int(graph_id),
            "node_index": int(node_index),
            "tensor_name": name,
            "ne": ne,
            "elements": elements,
            "all_finite": finite,
            "minimum": minimum,
            "maximum": maximum,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--no-scan", action="store_true")
    parser.add_argument("--json", type=Path, help="new output path; existing files are refused")
    args = parser.parse_args()
    result = parse_capture(args.capture, not args.no_scan)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.json:
        with args.json.open("x", encoding="utf-8") as handle:
            handle.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
