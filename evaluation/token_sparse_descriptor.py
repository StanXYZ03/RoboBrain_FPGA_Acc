"""Independent Python codec for Token/Block Sparse Descriptor v1 golden vectors."""

from __future__ import annotations

import struct
import zlib


MAGIC = 0x44535452
VERSION = 1
HEADER_BYTES = 64
HAS_POSITIONS = 0x1
IDENTITY = 0x2
SORTED_UNIQUE = 0x4
ALLOWED_FLAGS = HAS_POSITIONS | IDENTITY | SORTED_UNIQUE
HEADER = struct.Struct("<IHHIIIHHIII7I")


def encode(full_count: int, entries: list[dict[str, object]], flags: int) -> bytes:
    if not entries:
        raise ValueError("entries must not be empty")
    position_dims = len(entries[0]["position"])
    indices = [int(entry["original_index"]) for entry in entries]
    if (
        full_count <= 0 or len(entries) > full_count or position_dims > 4
        or flags & ~ALLOWED_FLAGS or not flags & SORTED_UNIQUE
        or bool(flags & HAS_POSITIONS) != bool(position_dims)
        or indices != sorted(set(indices)) or indices[-1] >= full_count
        or any(len(entry["position"]) != position_dims for entry in entries)
    ):
        raise ValueError("invalid descriptor")
    if flags & IDENTITY and (len(entries) != full_count or indices != list(range(full_count))):
        raise ValueError("invalid identity descriptor")
    entry_bytes = 4 * (1 + position_dims)
    payload = b"".join(
        struct.pack("<" + "I" * (1 + position_dims), index, *entry["position"])
        for index, entry in zip(indices, entries)
    )
    values = [
        MAGIC, VERSION, HEADER_BYTES, flags, full_count, len(entries),
        entry_bytes, position_dims, len(payload), zlib.crc32(payload), 0,
        *([0] * 7),
    ]
    header = bytearray(HEADER.pack(*values))
    struct.pack_into("<I", header, 32, zlib.crc32(header))
    return bytes(header) + payload


def decode(data: bytes) -> dict[str, object]:
    if len(data) < HEADER_BYTES:
        raise ValueError("truncated header")
    values = list(HEADER.unpack_from(data))
    magic, version, header_bytes, flags, full_count, kept_count = values[:6]
    entry_bytes, position_dims, payload_bytes, payload_crc, header_crc = values[6:11]
    if magic != MAGIC or version != VERSION or header_bytes != HEADER_BYTES:
        raise ValueError("bad header identity")
    header = bytearray(data[:HEADER_BYTES])
    struct.pack_into("<I", header, 32, 0)
    if zlib.crc32(header) != header_crc:
        raise ValueError("bad header CRC")
    if any(values[11:]) or flags & ~ALLOWED_FLAGS or not flags & SORTED_UNIQUE:
        raise ValueError("unsupported header")
    if (
        full_count <= 0 or kept_count <= 0 or kept_count > full_count
        or position_dims > 4 or bool(flags & HAS_POSITIONS) != bool(position_dims)
        or entry_bytes != 4 * (1 + position_dims)
        or payload_bytes != kept_count * entry_bytes
        or len(data) != HEADER_BYTES + payload_bytes
    ):
        raise ValueError("inconsistent lengths")
    payload = data[HEADER_BYTES:]
    if zlib.crc32(payload) != payload_crc:
        raise ValueError("bad payload CRC")
    entries = []
    for ordinal in range(kept_count):
        values = struct.unpack_from("<" + "I" * (1 + position_dims), payload, ordinal * entry_bytes)
        entries.append({"original_index": values[0], "position": list(values[1:])})
    indices = [entry["original_index"] for entry in entries]
    if indices != sorted(set(indices)) or indices[-1] >= full_count:
        raise ValueError("invalid indices")
    if flags & IDENTITY and (kept_count != full_count or indices != list(range(full_count))):
        raise ValueError("invalid identity")
    return {
        "version": version,
        "flags": flags,
        "full_count": full_count,
        "kept_count": kept_count,
        "position_dims": position_dims,
        "entry_bytes": entry_bytes,
        "entries": entries,
    }
