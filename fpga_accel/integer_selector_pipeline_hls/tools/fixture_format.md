# integer_selector_pipeline HLS fixture format (v1)

All files are little-endian.  The generator is
`fpga_accel/integer_selector_pipeline_hls/tools/generate_fixtures.py`; it calls
the **frozen** goldens
`evaluation.selector_integer_scorer_audit.integer_chain` and
`evaluation.hardware_regular_selector_fixedpoint.select_fixedpoint`, never a
second copy of the maths.

The accepted validation NPZ contains only the pooled `group_mean_q` (there is no
raw `tokens_q`), so the raw 4x1024 token bytes are drawn from the documented
deterministic xorshift64* stream.  That does not change the scorer-side golden
(which is token independent) and the selector-side golden is computed over
exactly those bytes.

## `wv_q.bin`

Raw signed INT8, row-major `[128, 1024]` (131 072 bytes).  Byte-for-byte copy of
the accepted seed-20261003 bundle
`artifacts/token_selector/selector_wv_scale_regularization/20261003/Wv_int8_rowmajor.bin`
(SHA256 `126bd842…89d3b`); the source path and both SHA256 values are pinned in
`manifest.json`.

## `scene_<name>.bin`

### 64-byte header

| offset | type    | field                             |
|-------:|:--------|:----------------------------------|
| 0      | char[8] | magic `ISPSCN01`                  |
| 8      | u32     | `group_count` G                   |
| 12     | u32     | `grid_h`                          |
| 16     | u32     | `grid_w`                          |
| 20     | u32     | flags (0)                         |
| 24     | u64     | FNV-1a-64 of the `tokens_q` bytes |
| 32     | u64     | FNV-1a-64 of the `group_mean_q` bytes |
| 40     | u64     | FNV-1a-64 of the `query_q` bytes  |
| 48     | u64     | `payload_bytes`                   |
| 56     | u64     | reserved, zero                    |

### Payload

- `int8 tokens_q[G*4*1024]`   row-major, group-major (one group = 4096 bytes)
- `int8 group_mean_q[G*1024]` row-major
- `int8 query_q[128]`

`tokens_q` is also the 128-bit packed AXI layout: byte `B` maps to word `B/16`,
lane `B%16` (little-endian).  The testbench packs the raw bytes with the frozen
`ts_pack_int8_to_words` helper from `fpga_accel/token_selector_hls`.

## `<case>.exp.bin`

### 80-byte header

| offset | type    | field                              |
|-------:|:--------|:-----------------------------------|
| 0      | char[8] | magic `ISPEX001`                   |
| 8      | i32     | `status`                           |
| 12     | i32     | `output_count`                     |
| 16     | i32     | `real_count`                       |
| 20     | i32     | `context_count`                    |
| 24     | f32     | `scale_out`                        |
| 28     | u32     | reserved, zero                     |
| 32     | u64     | FNV-1a-64 of `score_q` bytes       |
| 40     | u64     | FNV-1a-64 of `tokens_out` bytes    |
| 48     | u64     | FNV-1a-64 of `group_index` bytes   |
| 56     | u64     | FNV-1a-64 of metadata bytes (`source_type`+`region_id`+`valid`) |
| 64     | u64     | `payload_bytes`                    |
| 72     | u64     | reserved, zero                     |

### Payload

Always:

- `uint16 score_q[G]` — the on-chip intermediate, from `integer_chain`.

When `status == 0` additionally, in order:

- `int8 tokens_out[output_count*4096]`
- `uint16 group_index[output_count]`
- `uint8 source_type[output_count]`
- `uint8 region_id[output_count]`
- `uint8 valid[output_count]`

## `manifest.txt`

One line per case:

```
<case_name> <scene_file> <budget> <local_cap> <scale>
```

## `manifest.json`

Coverage ledger: per-scene and per-case SHA256, FNV hashes, status/counts, the
frozen golden source SHA256 values and the pinned Wv bundle.
