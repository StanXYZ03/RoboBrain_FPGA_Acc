# Compact integer selector pipeline fixture format

All integers are little-endian.  Fixtures are produced by
`tools/generate_fixtures.py`; the golden values are re-used from the accepted
frozen Python entries and are never re-derived by a second implementation.

## Wv bundle

* `wv_q.bin` -- byte copy of the accepted seed-20261003 Wv tensor
  (128 x 1024 row-major INT8, 131072 B, SHA-pinned to the accepted bundle).

## Input arena

* `arena_<scene>.bin` -- exactly the `input_arena` bytes consumed by the compact
  top.  Three back-to-back segments, no extra padding because every offset is
  already 64-byte aligned:

  | segment | offset | size |
  |---|---|---|
  | `tokens_int8` (`G x 4 x 1024` INT8) | `0` | `G * 4096` |
  | `group_mean_int8` (`G x 1024` INT8) | `G * 4096` | `G * 1024` |
  | `query_int8` (`128` INT8) | `G * 5120` | `128` |

  Total `G * 5120 + 128` bytes.  The 128-bit packed lane order is the accepted
  little-endian order: byte `B` of a segment is lane `B % 16` of word `B / 16`.

## Expected output

* `<case>.exp.bin`

  52-byte header:

  | offset | type | field |
  |---|---|---|
  | 0 | `char[8]` | magic `ISPCMP01` |
  | 8 | `int32` | status |
  | 12 | `int32` | `output_count` |
  | 16 | `int32` | `real_count` |
  | 20 | `int32` | `context_count` |
  | 24 | `float32` | `scale_out` |
  | 28 | `int32` | runtime `budget` (`K`) |
  | 32 | `int32` | `total_bytes` of the canonical arena |
  | 36 | `uint64` | FNV-1a-64 of the arena bytes |
  | 44 | `uint64` | payload bytes (== `total_bytes`) |

  Followed by `total_bytes` bytes of the canonical fixed-budget output arena
  produced live by
  `evaluation.hardware_regular_token_selector.fpga_transfer_layout(K)`:

  | segment | dtype | size |
  |---|---|---|
  | `tokens_int8` | int8 | `K * 4096` |
  | `group_index_uint16` | uint16 | `K * 2` |
  | `source_type_uint8` | uint8 | `K` |
  | `region_id_uint8` | uint8 | `K` |
  | `valid_uint8` | uint8 | `K` |

  Each segment start is independently 64-byte aligned; every inter-segment
  alignment gap, every unused entry (identity bypass writes only
  `output_count <= K` entries) and the trailing padding to the 64-byte aligned
  total are deterministic zero.

## Manifest

* `manifest.txt` -- `case arena_file budget local_cap scale grid_h grid_w` lines.
* `manifest.json` -- SHA256 ledger, coverage, the frozen golden entry points and
  the per-budget authoritative layout produced by `describe_fpga_layout`.
