#!/usr/bin/env python3
"""Create a reproducible legacy U-Boot script image without host mkimage."""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path


MAGIC = 0x27051956
IH_OS_LINUX = 5
IH_ARCH_ARM64 = 22
IH_TYPE_SCRIPT = 6
IH_COMP_NONE = 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="RoboBrain RAM Linux")
    args = parser.parse_args()

    script = args.command_file.read_bytes()
    if not script.endswith(b"\n"):
        script += b"\n"
    # IH_TYPE_SCRIPT uses the legacy multi-image payload convention: a
    # big-endian component-length table, a zero terminator, then script text.
    # image_source_script() consumes this table before calling run_command_list.
    data = struct.pack(">II", len(script), 0) + script
    name = args.name.encode("ascii")[:31].ljust(32, b"\0")
    header = struct.pack(
        ">7I4B32s",
        MAGIC, 0, 0, len(data), 0, 0, zlib.crc32(data) & 0xFFFFFFFF,
        IH_OS_LINUX, IH_ARCH_ARM64, IH_TYPE_SCRIPT, IH_COMP_NONE, name,
    )
    header_crc = zlib.crc32(header) & 0xFFFFFFFF
    header = struct.pack(">I", MAGIC) + struct.pack(">I", header_crc) + header[8:]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(header + data)
    print(
        f"UBOOT_SCRIPT_PASS output={args.output} bytes={len(header) + len(data)} "
        f"script_bytes={len(script)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
