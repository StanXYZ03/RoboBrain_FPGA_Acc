#!/usr/bin/env python3
"""Heterogeneous transfer audit for the fused integer Selector pipeline.

This is a **model estimate**, not a measurement.  It reports, per image and per
(G, budget) configuration:

* Jetson -> FPGA input bytes: ``tokens_q``, ``group_mean_q``, ``query_q`` and
  the Wv tensor under a cold-start and a PL/DDR-resident accounting;
* FPGA -> Jetson output bytes in **two explicit counters**:

  - ``logical_payload_bytes`` -- the raw valid bytes of the ``output_count``
    returned tokens plus the four metadata arrays, *without* any padding;
  - ``fixed_budget_arena_bytes`` -- the authoritative fixed-budget buffer size
    produced *live* by the frozen contract
    ``evaluation.hardware_regular_token_selector.fpga_transfer_layout(budget)``,
    in which every one of the 5 segments starts at its own 64-byte-aligned
    offset (saved verbatim under ``fixed_budget_arena.segments``).

* 1 GbE time under an explicit effective-payload-bandwidth assumption with a
  sensitivity band, and the theoretical 6 GB/s local DDR time.  These
  conservative network/DDR numbers are **all based on
  ``fixed_budget_arena_bytes``**.  A scatter-gather time over the raw payload is
  also reported, but only as an explicit **lower bound**; it never replaces the
  fixed-arena basis.
* the external-DDR ``score_q`` round-trip of the two-kernel baseline and the
  on-chip fusion saving.

The on-chip ``score_q`` is a local buffer and is never counted as external
traffic.  The on-chip fusion does **not** remove the Jetson <-> FPGA round
trip; it removes the extra FPGA-DDR round trip of the intermediate.

Correction history (2026-10-03, Codex review): the first revision summed the
selected tokens plus the metadata and applied a *single* trailing ``align_up``.
That is not the frozen 5-segment contract, which aligns **each** segment
offset, so K=48/K=96 FPGA -> Jetson totals (and the times derived from them)
were underestimated.  The ``fixed_budget_arena_bytes`` field is now the
authoritative one; the old single-``align_up`` value is no longer produced here.
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

MERGE_UNIT = 4
FEATURE_DIM = 1024
GROUP_ELEMENTS = MERGE_UNIT * FEATURE_DIM  # 4096
EMBED_DIM = 1024
QUERY_DIM = 128
WV_ROWS = 128
WV_BYTES = WV_ROWS * EMBED_DIM  # 131072

#: The frozen 5-segment layout is authoritative; this constant is only kept for
#: the human-readable assumptions block and is cross-checked against it.
LAYOUT_ALIGNMENT = fpga_transfer_layout(48).alignment
LAYOUT_AUTHORITY = "evaluation.hardware_regular_token_selector.fpga_transfer_layout"

#: Production configurations covered by the audit.
GROUP_COUNTS = (80, 260, 400)
BUDGETS = (48, 64, 96)

#: 1 GbE assumption: 1 Gbit/s line rate = 125 MB/s; the effective TCP/UDP
#: payload after framing/headers is assumed 90-118 MB/s, central 110 MB/s.
GBE_LINE_RATE_MBPS = 125.0
GBE_EFFECTIVE_MBPS_CENTRAL = 110.0
GBE_EFFECTIVE_MBPS_LOW = 90.0
GBE_EFFECTIVE_MBPS_HIGH = 118.0

#: Theoretical local DDR path used for the model (not a measurement).
DDR_BYTES_PER_SECOND = 6.0e9


def output_group_count(group_count: int, budget: int) -> int:
    """Accepted selector output count: identity when G <= K, else K."""
    return group_count if group_count <= budget else budget


def image_bytes(group_count: int, budget: int, wv_resident: bool) -> dict[str, Any]:
    """Per-image Jetson -> FPGA and FPGA -> Jetson bytes for one (G, K).

    ``fpga_to_jetson`` carries the two corrected counters.  The fixed-budget
    arena is produced live by the frozen contract and is independent of the
    runtime ``group_count`` (identity bypass included), while the logical
    payload only counts the ``output_count`` valid groups.
    """
    layout = fpga_transfer_layout(budget)
    layout_description = describe_fpga_layout(budget)

    tokens_in = group_count * GROUP_ELEMENTS
    group_mean_in = group_count * EMBED_DIM
    query_in = QUERY_DIM
    wv_in = 0 if wv_resident else WV_BYTES

    out_groups = output_group_count(group_count, budget)
    selected_tokens = out_groups * GROUP_ELEMENTS
    group_index = out_groups * 2
    source_type = out_groups
    region_id = out_groups
    valid = out_groups
    metadata = group_index + source_type + region_id + valid

    return {
        "group_count": group_count,
        "budget": budget,
        "output_group_count": out_groups,
        "wv_resident": bool(wv_resident),
        "layout_authority": LAYOUT_AUTHORITY,
        "jetson_to_fpga": {
            "tokens_q": tokens_in,
            "group_mean_q": group_mean_in,
            "query_q": query_in,
            "wv_q": wv_in,
            "total": tokens_in + group_mean_in + query_in + wv_in,
        },
        "fpga_to_jetson": {
            "output_group_count": out_groups,
            "selected_tokens": selected_tokens,
            "group_index": group_index,
            "source_type": source_type,
            "region_id": region_id,
            "valid": valid,
            "metadata": metadata,
            # Raw valid bytes of output_count groups; no padding.
            "logical_payload_bytes": selected_tokens + metadata,
            # Authoritative fixed-budget buffer, produced live by the frozen
            # 5-segment 64-byte-aligned contract (uses K, not output_count).
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
            "score_q_bytes_written": group_count * 2,
            "score_q_bytes_read": group_count * 2,
            "total": 4 * group_count,
        },
        "on_chip_fusion_saving": {
            "ddr_bytes": 4 * group_count,
            "ideal_time_us_at_6gbs": 4 * group_count / DDR_BYTES_PER_SECOND * 1e6,
        },
        "future_group_mean_reconstruction_saving": {
            "jetson_to_fpga_bytes": group_mean_in,
            "note": "potential only; the top keeps group_mean_q as a separate "
                    "input and the accepted group-mean quantisation contract is "
                    "unchanged this round",
        },
    }


def _time_us(byte_count: int, bytes_per_second: float) -> float:
    return byte_count / bytes_per_second * 1e6


def _gbe_times(byte_count: int) -> dict[str, float]:
    return {
        "central_us": _time_us(byte_count, GBE_EFFECTIVE_MBPS_CENTRAL * 1e6),
        "low_bandwidth_us": _time_us(byte_count, GBE_EFFECTIVE_MBPS_LOW * 1e6),
        "high_bandwidth_us": _time_us(byte_count, GBE_EFFECTIVE_MBPS_HIGH * 1e6),
    }


def build_transfer_audit() -> dict[str, Any]:
    configs: dict[str, Any] = {}
    for group_count in GROUP_COUNTS:
        for budget in BUDGETS:
            for resident in (False, True):
                key = f"G{group_count}_K{budget}_{'resident' if resident else 'cold'}"
                record = image_bytes(group_count, budget, resident)
                jf = record["jetson_to_fpga"]["total"]
                arena = record["fpga_to_jetson"]["fixed_budget_arena_bytes"]
                payload = record["fpga_to_jetson"]["logical_payload_bytes"]
                record["times"] = {
                    "jetson_to_fpga_1gbe": _gbe_times(jf),
                    # Authoritative FPGA -> Jetson link time: fixed budget arena.
                    "fpga_to_jetson_1gbe_arena": _gbe_times(arena),
                    # Explicit lower bound only; never replaces the arena basis.
                    "fpga_to_jetson_1gbe_payload_scatter_gather_lower_bound": {
                        **_gbe_times(payload),
                        "basis": "logical_payload_bytes",
                        "note": "lower bound only: assumes the host gathers just "
                                "the output_count valid groups; it is not a "
                                "substitute for the fixed-budget arena accounting",
                    },
                    "jetson_to_fpga_local_ddr_6gbs_us": _time_us(jf, DDR_BYTES_PER_SECOND),
                    "fpga_to_jetson_local_ddr_6gbs_us": _time_us(
                        arena, DDR_BYTES_PER_SECOND
                    ),
                    "fpga_to_jetson_payload_local_ddr_6gbs_lower_bound_us": _time_us(
                        payload, DDR_BYTES_PER_SECOND
                    ),
                    "baseline_score_q_ddr_round_trip_6gbs_us": record[
                        "on_chip_fusion_saving"
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
                    "produced live by the frozen contract; no second offset "
                    "algorithm is kept in this module",
        },
        "assumptions": {
            "gbe_line_rate_mbytes_per_s": GBE_LINE_RATE_MBPS,
            "gbe_effective_mbytes_per_s_central": GBE_EFFECTIVE_MBPS_CENTRAL,
            "gbe_effective_mbytes_per_s_low": GBE_EFFECTIVE_MBPS_LOW,
            "gbe_effective_mbytes_per_s_high": GBE_EFFECTIVE_MBPS_HIGH,
            "local_ddr_bytes_per_s": DDR_BYTES_PER_SECOND,
            "wv_bytes": WV_BYTES,
            "layout_alignment": LAYOUT_ALIGNMENT,
        },
        "wv_resident_one_time_bytes": WV_BYTES,
        "not_counted_as_external": ["on-chip score_q (local buffer, never DDR)"],
        "configs": configs,
    }


if __name__ == "__main__":  # pragma: no cover - manual inspection
    import json

    print(json.dumps(build_transfer_audit(), indent=2))
