#!/usr/bin/env python3
"""Cycle-level schedule model for the PE16 coarse DATAFLOW experiment."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLOCK_MHZ = 200.0
QK = 32
TM = 16
TN = 64
PE_N = 16
INIT_CYCLES = 1026
STORE_CYCLES = 1034
COMPUTE_DEPTH = 25

WORKLOADS = (
    ("language_prefill", 512, 512, 9728, 2560, 242808.0),
    ("vision_encoder", 4320, 512, 4096, 1024, 448838.667),
    ("decoder_gemv", 1, 1, 9728, 2560, 14647.0),
)


def component_cycles(m: int, n: int, k: int) -> dict[str, int]:
    k_blocks = k // QK
    m_tiles = math.ceil(m / TM)
    n_tiles = math.ceil(n / TN)
    weight = TN * k_blocks + 14
    activation = TM * k_blocks + 14
    compute = TM * (TN // PE_N) * k_blocks + COMPUTE_DEPTH
    compute_stage = INIT_CYCLES + compute

    serial_per_n = weight + m_tiles * (
        activation + compute_stage + STORE_CYCLES
    )

    # Prime the first activation bank.  Each subsequent coarse step overlaps
    # load(t+1), compute(t), and store(t-1); then the final store drains.
    overlapped_steps = 0
    for tile in range(m_tiles):
        load_next = activation if tile + 1 < m_tiles else 0
        store_previous = STORE_CYCLES if tile > 0 else 0
        overlapped_steps += max(load_next, compute_stage, store_previous)
    dataflow_per_n = weight + activation + overlapped_steps + STORE_CYCLES
    return {
        "m_tiles": m_tiles,
        "n_tiles": n_tiles,
        "k_blocks": k_blocks,
        "weight_tile_cycles": weight,
        "activation_tile_cycles": activation,
        "compute_tile_cycles": compute,
        "initialize_and_compute_cycles": compute_stage,
        "store_tile_cycles": STORE_CYCLES,
        "serial_cycles": n_tiles * serial_per_n,
        "dataflow_cycles": n_tiles * dataflow_per_n,
    }


def workload(total_m: int, chunk_m: int, n: int, k: int) -> dict[str, float]:
    serial = 0
    dataflow = 0
    calls = 0
    for m0 in range(0, total_m, chunk_m):
        row = component_cycles(min(chunk_m, total_m - m0), n, k)
        serial += row["serial_cycles"]
        dataflow += row["dataflow_cycles"]
        calls += 1
    serial_us = serial / CLOCK_MHZ
    dataflow_us = dataflow / CLOCK_MHZ
    return {
        "runtime_calls": calls,
        "serial_cycles": serial,
        "dataflow_cycles": dataflow,
        "serial_schedule_us": serial_us,
        "dataflow_schedule_us": dataflow_us,
        "schedule_reduction": 1.0 - dataflow_us / serial_us,
        "schedule_speedup": serial_us / dataflow_us,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/dataflow_cycle_model.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite experiment record: {args.output}")

    rows = []
    for name, total_m, chunk_m, n, k, measured_us in WORKLOADS:
        model = workload(total_m, chunk_m, n, k)
        residual_us = measured_us - model["serial_schedule_us"]
        rows.append(
            {
                "name": name,
                "shape": {
                    "total_m": total_m,
                    "chunk_m": chunk_m,
                    "n": n,
                    "k": k,
                },
                **model,
                "pe16_board_measured_us": measured_us,
                "pe16_measured_minus_serial_model_us": residual_us,
                "first_order_projected_board_us": (
                    residual_us + model["dataflow_schedule_us"]
                ),
                "projection_scope": (
                    "cycle-model projection; preserves the measured PE16/model "
                    "residual; not a board measurement"
                ),
            }
        )

    output = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ANALYTICAL_SCHEDULE_MODEL_ONLY",
        "architecture": {
            "clock_mhz": CLOCK_MHZ,
            "tm": TM,
            "tn": TN,
            "pe_n": PE_N,
            "qk": QK,
            "compute_pipeline_ii": 1,
            "compute_pipeline_depth": COMPUTE_DEPTH,
            "initialize_cycles": INIT_CYCLES,
            "store_cycles": STORE_CYCLES,
        },
        "formula": {
            "serial_per_n_tile": (
                "weight + M_tiles*(activation + initialize_compute + store)"
            ),
            "dataflow_per_n_tile": (
                "weight + prime_activation + sum(max(load_next, "
                "initialize_compute, store_previous)) + drain_store"
            ),
        },
        "workloads": rows,
        "limitations": [
            "This is a deterministic HLS schedule model, not RTL or board timing.",
            "Concurrency must still be proved by HLS DATAFLOW reports and RTL simulation.",
            "DDR contention and AXI back-pressure are represented only by the frozen board residual.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
