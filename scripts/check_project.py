#!/usr/bin/env python3
"""Fast, dependency-free project layout and toolchain check."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "RoboBrain2.5-4B/config.json",
    "RoboBrain2.5-4B/model.safetensors",
    "app/inference.py",
    "fpga_accel/README.md",
    "fpga_accel/scripts/analyze_model_structure.py",
    "fpga_accel/tests",
)


def main() -> int:
    print(f"project: {ROOT}")
    missing = []
    for relative in REQUIRED:
        path = ROOT / relative
        state = "ok" if path.exists() else "missing"
        print(f"[{state:7}] {relative}")
        if not path.exists():
            missing.append(relative)

    for tool in ("python3", "vivado", "vitis_hls"):
        location = shutil.which(tool)
        print(f"[tool   ] {tool}: {location or 'not on PATH'}")

    config_path = ROOT / "RoboBrain2.5-4B/config.json"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
        print(f"[model  ] type: {config.get('model_type', 'unknown')}")
        print(f"[model  ] architectures: {config.get('architectures', [])}")

    if missing:
        print("\nProject check failed; required files are missing.", file=sys.stderr)
        return 1
    print("\nProject layout is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

