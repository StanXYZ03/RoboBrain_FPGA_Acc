#!/usr/bin/env python3
"""Heterogeneous transfer audit for the compact integer Selector pipeline.

This is a **model estimate**, not a measurement.  It reports, per image and per
production (G, K) configuration, the *corrected* accounting requested for the
compact envelope:

* Jetson -> FPGA: ONE contiguous input arena of ``G*5120 + 128`` bytes
  (tokens | group means | query) plus the 128 KiB Wv tensor in cold mode; in the
  corrected **Wv-resident** mode the weight is loaded once and costs zero bytes
  per image.
* FPGA -> Jetson: the authoritative fixed-budget arena produced live by
  ``evaluation.hardware_regular_token_selector.fpga_transfer_layout(K)`` (the
  fixed arena, independent of the runtime group count), with the raw
  ``output_count`` logical payload reported separately.
* 1 GbE time under an explicit effective-payload-bandwidth assumption
  (90/110/118 MB/s) and the theoretical 6 GB/s local DDR time, both based on the
  fixed arena bytes.

The on-chip ``score_q`` is a local buffer and is never external traffic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.hardware_regular_token_selector import (  # noqa: E402
    describe_fpga_layout,
    fpga_transfer_layout,
)

GROUP_ELEMENTS = 4096
EMBED_DIM = 1024
QUERY_DIM = 128
WV_BYTES = 128 * 1024

GROUP_COUNTS = (80, 260, 400)
BUDGETS = (48, 64, 96)

GBE_EFFECTIVE_MBPS_CENTRAL = 110.0
GBE_EFFECTIVE_MBPS_LOW = 90.0
GBE_EFFECTIVE_MBPS_HIGH = 118.0
DDR_BYTES_PER_SECOND = 6.0e9

LAYOUT_AUTHORITY = "evaluation.hardware_regular_token_selector.fpga_transfer_layout"


def input_arena_bytes(group_count: int) -> int:
    return group_count * 5120 + 128


def output_group_count(group_count: int, budget: int) -> int:
    return group_count if group_count <= budget else budget


def _time_us(byte_count: int, bytes_per_second: float) -> float:
    return byte_count / bytes_per_second * 1e6


def _gbe_times(byte_count: int) -> dict[str, float]:
    return {
        "central_us": _time_us(byte_count, GBE_EFFECTIVE_MBPS_CENTRAL * 1e6),
        "low_bandwidth_us": _time_us(byte_count, GBE_EFFECTIVE_MBPS_LOW * 1e6),
        "high_bandwidth_us": _time_us(byte_count, GBE_EFFECTIVE_MBPS_HIGH * 1e6),
    }


def image_bytes(group_count: int, budget: int, wv_resident: bool) -> dict[str, Any]:
    layout = fpga_transfer_layout(budget)
    layout_description = describe_fpga_layout(budget)
    arena_in = input_arena_bytes(group_count)
    wv_in = 0 if wv_resident else WV_BYTES
    out_groups = output_group_count(group_count, budget)
    logical_payload = out_groups * GROUP_ELEMENTS + out_groups * 5
    return {
        "group_count": group_count,
        "budget": budget,
        "wv_resident": bool(wv_resident),
        "output_group_count": out_groups,
        "layout_authority": LAYOUT_AUTHORITY,
        "jetson_to_fpga": {
            "input_arena_bytes": arena_in,
            "wv_q": wv_in,
            "total": arena_in + wv_in,
        },
        "fpga_to_jetson": {
            "output_group_count": out_groups,
            "logical_payload_bytes": logical_payload,
            "fixed_budget_arena_bytes": layout.total_bytes,
            "fixed_budget_arena": {
                "budget": layout.budget,
                "alignment": layout.alignment,
                "total_bytes": layout.total_bytes,
                "segment_count": len(layout.segments),
                "segments": layout_description["segments"],
            },
        },
        "baseline_score_q_ddr_round_trip": {
            "total_bytes": 4 * group_count,
            "ideal_time_us_at_6gbs": 4 * group_count / DDR_BYTES_PER_SECOND * 1e6,
        },
    }


def build_transfer_audit() -> dict[str, Any]:
    configs: dict[str, Any] = {}
    for group_count in GROUP_COUNTS:
        for budget in BUDGETS:
            for resident in (False, True):
                key = f"G{group_count}_K{budget}_{'resident' if resident else 'cold'}"
                record = image_bytes(group_count, budget, resident)
                jf = record["jetson_to_fpga"]["total"]
                fj = record["fpga_to_jetson"]
                arena = fj["fixed_budget_arena_bytes"]
                record["times"] = {
                    "jetson_to_fpga_1gbe": _gbe_times(jf),
                    "fpga_to_jetson_1gbe_arena": _gbe_times(arena),
                    "fpga_to_jetson_1gbe_payload_lower_bound": {
                        **_gbe_times(fj["logical_payload_bytes"]),
                        "basis": "logical_payload_bytes",
                        "note": "lower bound only",
                    },
                    "jetson_to_fpga_local_ddr_6gbs_us": _time_us(jf, DDR_BYTES_PER_SECOND),
                    "fpga_to_jetson_local_ddr_6gbs_us": _time_us(arena, DDR_BYTES_PER_SECOND),
                    "baseline_score_q_ddr_round_trip_6gbs_us": record[
                        "baseline_score_q_ddr_round_trip"
                    ]["ideal_time_us_at_6gbs"],
                }
                configs[key] = record

    return {
        "scope": "model estimate, not a measurement",
        "transport_time_basis": "fixed_budget_arena_bytes",
        "layout_authority": {
            "module": "evaluation.hardware_regular_token_selector",
            "functions": ["fpga_transfer_layout", "describe_fpga_layout"],
            "note": "fixed arena values and the 5 segment offsets/sizes are "
                    "produced live by the frozen contract",
        },
        "input_arena": {
            "tokens_int8": "offset 0, size G*4096",
            "group_mean_int8": "offset G*4096, size G*1024",
            "query_int8": "offset G*5120, size 128",
            "total_bytes": "G*5120 + 128",
        },
        "assumptions": {
            "gbe_effective_mbytes_per_s_central": GBE_EFFECTIVE_MBPS_CENTRAL,
            "gbe_effective_mbytes_per_s_low": GBE_EFFECTIVE_MBPS_LOW,
            "gbe_effective_mbytes_per_s_high": GBE_EFFECTIVE_MBPS_HIGH,
            "local_ddr_bytes_per_s": DDR_BYTES_PER_SECOND,
            "wv_bytes": WV_BYTES,
            "layout_alignment": fpga_transfer_layout(48).alignment,
        },
        "wv_resident_one_time_bytes": WV_BYTES,
        "not_counted_as_external": ["on-chip score_q (local buffer, never DDR)"],
        "configs": configs,
    }


if __name__ == "__main__":  # pragma: no cover
    import json

    print(json.dumps(build_transfer_audit(), indent=2))
