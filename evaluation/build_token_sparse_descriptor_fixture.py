#!/usr/bin/env python3
"""Generate a C-authored descriptor fixture and verify byte equality in Python."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.token_sparse_descriptor import decode, encode


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--c-writer",
        type=Path,
        default=ROOT / "fpga_accel/token_sparse_descriptor/build/export_fixture",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "fpga_accel/token_sparse_descriptor/artifacts",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    binary = args.output_dir / "golden_sparse_v1.bin"
    subprocess.run([str(args.c_writer), str(binary)], check=True)
    decoded = decode(binary.read_bytes())
    python_bytes = encode(decoded["full_count"], decoded["entries"], decoded["flags"])
    if python_bytes != binary.read_bytes():
        raise SystemExit("C/Python byte mismatch")
    summary = {
        "format": "token-sparse-descriptor-golden-v1",
        "status": "C_PYTHON_BYTE_EQUAL_PASS",
        "binary": str(binary.relative_to(ROOT)),
        "binary_bytes": binary.stat().st_size,
        "binary_sha256": sha256(binary),
        "decoded": decoded,
        "c_writer": str(args.c_writer.relative_to(ROOT)),
    }
    output = args.output_dir / "golden_sparse_v1.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "sha256": summary["binary_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
