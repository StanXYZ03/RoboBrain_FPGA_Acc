#!/usr/bin/env python3
"""Dependency-free, bounded GGUF metadata reader.

Only metadata is decoded; tensor payloads are never mapped or evaluated. Large
tokenizer arrays are skipped in place so inspecting an untrusted multi-GB file
does not consume comparable RAM.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import BinaryIO, Any


GGUF_TYPES = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2),
    4: ("I", 4), 5: ("i", 4), 6: ("f", 4), 7: ("?", 1),
    8: (None, None), 9: (None, None), 10: ("Q", 8), 11: ("q", 8),
    12: ("d", 8),
}
MAX_KEY_BYTES = 1 << 20
MAX_STRING_CAPTURE = 1 << 20


class GGUFError(ValueError):
    pass


def _exact(stream: BinaryIO, count: int) -> bytes:
    data = stream.read(count)
    if len(data) != count:
        raise GGUFError("unexpected end of GGUF metadata")
    return data


def _u32(stream: BinaryIO) -> int:
    return struct.unpack("<I", _exact(stream, 4))[0]


def _u64(stream: BinaryIO) -> int:
    return struct.unpack("<Q", _exact(stream, 8))[0]


def _string(stream: BinaryIO, *, capture: bool) -> str | None:
    length = _u64(stream)
    if capture and length <= MAX_STRING_CAPTURE:
        return _exact(stream, length).decode("utf-8", errors="strict")
    stream.seek(length, 1)
    return None


def _value(stream: BinaryIO, value_type: int, *, capture: bool) -> Any:
    if value_type not in GGUF_TYPES:
        raise GGUFError(f"unknown GGUF metadata type {value_type}")
    if value_type == 8:
        return _string(stream, capture=capture)
    if value_type == 9:
        element_type = _u32(stream)
        length = _u64(stream)
        if length > (1 << 40):
            raise GGUFError("implausible GGUF array length")
        # Capturing arrays is intentionally unsupported: keys needed by the
        # deployment gate are scalars, while tokenizer arrays can be enormous.
        for _ in range(length):
            _value(stream, element_type, capture=False)
        return {"array_type": element_type, "length": length} if capture else None
    fmt, size = GGUF_TYPES[value_type]
    assert fmt is not None and size is not None
    raw = _exact(stream, size)
    return struct.unpack("<" + fmt, raw)[0] if capture else None


def read_metadata(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        if _exact(stream, 4) != b"GGUF":
            raise GGUFError("not a GGUF file")
        version = _u32(stream)
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        tensor_count = _u64(stream)
        metadata_count = _u64(stream)
        if metadata_count > 1_000_000 or tensor_count > 10_000_000:
            raise GGUFError("implausible GGUF header counts")
        result: dict[str, Any] = {
            "gguf.version": version,
            "gguf.tensor_count": tensor_count,
            "gguf.metadata_count": metadata_count,
        }
        wanted_prefixes = ("general.", "tokenizer.chat_template")
        wanted_suffixes = (".context_length", ".block_count", ".embedding_length")
        for _ in range(metadata_count):
            key_len = _u64(stream)
            if key_len > MAX_KEY_BYTES:
                raise GGUFError("implausible GGUF metadata key length")
            key = _exact(stream, key_len).decode("utf-8", errors="strict")
            value_type = _u32(stream)
            capture = key.startswith(wanted_prefixes) or key.endswith(wanted_suffixes)
            value = _value(stream, value_type, capture=capture)
            if capture:
                result[key] = value
        result["gguf.metadata_end"] = stream.tell()
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(read_metadata(args.path), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
