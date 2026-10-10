# Register map draft - `integer_selector_pipeline_compact`

This is a draft description of the single AXI4-Lite `control` interface.  Exact
byte offsets are assigned by Vitis HLS during `csynth_design`; the generated
component metadata/report is the authority.  No hand-written offset table is
maintained here on purpose (it would be a second source of truth).

## AXI4-Lite `control`

### Data ports (offset registers for the three `m_axi` bundles)

| port | direction | description |
|---|---|---|
| `input_arena` | in | 64-bit base address of the input arena |
| `wv_q` | in | 64-bit base address of the 128 KiB Wv tensor |
| `output_arena` | in | 64-bit base address of the canonical output arena |

### Scalar control

| port | type | description |
|---|---|---|
| `group_count` | `int` | runtime group count `G`, `1..4096` |
| `grid_h` | `int` | 3x3 region grid rows, `1..128`, `grid_h*grid_w == G` |
| `grid_w` | `int` | 3x3 region grid columns |
| `budget` | `int` | output budget `K`, positive multiple of 16, `<= 96` |
| `local_cap` | `int` | per-region candidate cap, `1..16` |
| `scale_in` | `float` | register pass-through input |

### Scalar status

| port | type | description |
|---|---|---|
| `scale_out` | `float` | `scale_in` pass-through |
| `output_count` | `int` | number of valid arena entries written (`<= K`) |
| `real_count` | `int` | real (non-context) selected entries |
| `context_count` | `int` | context entries (`<= 9`) |
| `return` | `int` | status code (see below) |

### Status codes (identical to `integer_selector_pipeline`)

| value | name |
|---|---|
| 0 | `OK` |
| 1 | `ERR_NULL` |
| 2 | `ERR_GROUP_COUNT` |
| 3 | `ERR_GRID` |
| 4 | `ERR_BUDGET` |
| 5 | `ERR_INSUFFICIENT` |
| 6 | `ERR_CANDIDATE_POOL` |
| 7 | `ERR_INTERNAL` |
| 8 | `ERR_LOCAL_CAP` |

## Address-map artifact

`artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/`
contains the generated `*_csynth.rpt` / `*_csynth.xml` and
`csynth_design_size.rpt`, which enumerate the final register map and the exact
AXI-Lite bundle count.  A single `control` bundle is required; a second
`control_r` bundle is a gate failure.
