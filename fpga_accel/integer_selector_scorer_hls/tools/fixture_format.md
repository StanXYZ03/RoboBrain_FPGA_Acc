# integer_selector_scorer HLS fixture format (v1)

All files are little-endian.  The generator is
`fpga_accel/integer_selector_scorer_hls/tools/generate_fixtures.py`; it calls the
**frozen** golden `evaluation.selector_integer_scorer_audit.integer_chain`
(re-using its `floor_sqrt_int64` / `round_half_away_div`), never a second copy of
the maths.

## `wv_q.bin`

Raw signed INT8, row-major `[128, 1024]` (131 072 bytes).  Byte-for-byte copy of
the accepted seed-20261003 bundle
`artifacts/token_selector/selector_wv_scale_regularization/20261003/Wv_int8_rowmajor.bin`
(SHA256 `126bd842…89d3b`); the source path and both SHA256 values are pinned in
`manifest.json`.

## `*.bin` scorer fixtures (kinds 0, 1, 2)

### 128-byte header

| offset | type    | field                             |
|-------:|:--------|:----------------------------------|
| 0      | char[4] | magic `ISF1`                      |
| 4      | u32     | version (1)                       |
| 8      | u32     | kind: 0 real full, 1 synthetic, 2 stress spec |
| 12     | u32     | `group_count` G                   |
| 16     | u32     | `grid_h`                          |
| 20     | u32     | `grid_w`                          |
| 24     | u32     | `role_count` (1 or 2)             |
| 28     | u32     | `generator` (0 none, 1 xorshift64*) |
| 32     | u64     | `generator_seed`                  |
| 40     | u64     | FNV-1a-64 of the concatenated golden `score_q` bytes (role order) |
| 48     | u64     | FNV-1a-64 of the concatenated golden `semantic_q16` (int32 LE) |
| 56     | u64     | FNV-1a-64 of the concatenated golden `l2_int` (u16 LE) |
| 64     | u64     | FNV-1a-64 of `group_mean_q` bytes (stored even when generated) |
| 72     | u64     | `payload_bytes`                   |
| 80     | u64[6]  | reserved, zero                    |

### Payload (kind 0 / 1)

- `int8 group_mean_q[G*1024]` — present for kinds 0/1; **absent** for kind 2,
  where the testbench regenerates it with the deterministic generator below.
- then for each role `r = 0..role_count-1`, in order:
  - `int8 query_q[128]`
  - `uint16 golden_score_q[G]`
  - `int32 golden_semantic_q16[G]`
  - `uint16 golden_l2_int[G]`
  - `uint16 golden_semantic_norm_q15[G]`
  - `uint16 golden_l2_norm_q15[G]`

Role 0 is the correct prompt, role 1 the wrong prompt (same group means).

### Deterministic generator (kind 2, `generator == 1`)

`xorshift64*`, 64-bit unsigned wrap-around:

```
state = seed
next():
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    out = state * 0x2545F4914F6CDD1D
    return state, (out >> 56) & 0xFF      # one signed INT8 lane
```

- `G*1024` consecutive `next()` bytes are the row-major `group_mean_q`
  (byte `g*1024 + d`), interpreted as signed INT8 (`b >= 128 -> b - 256`).
- the following 128 `next()` bytes are `query_q`.

The `group_mean_fnv1a64` header field lets the testbench prove it regenerated the
exact bytes the golden used, for both explicitly stored (kinds 0/1) and
generator-derived (kind 2) group means.

## `primitives.bin` (kind 3)

### 64-byte header

| offset | type    | field                       |
|-------:|:--------|:----------------------------|
| 0      | char[4] | magic `ISP1`                |
| 4      | u32     | version (1)                 |
| 8      | u32     | `n_sqrt`                    |
| 12     | u32     | `n_div`                     |
| 16     | u32     | `n_minmax`                  |
| 20     | u32     | `minmax_total_values`       |
| 24     | u64     | `payload_bytes`             |
| 32     | u64[4]  | reserved, zero              |

### Payload

- `n_sqrt` × `u64 input, u64 expected` (`expected = floor_sqrt_int64(input)`)
- `n_div`  × `i64 numerator, i64 denominator, i64 expected`
  (`expected = round_half_away_div(num, den)`, `den > 0`), including exact
  positive and negative `.5` ties
- `n_minmax` × `i64 low, i64 high, u32 count, i64 value[count],
  u16 expected[count]` (`expected = minmax_normalize_q15(value; low, high)`)
