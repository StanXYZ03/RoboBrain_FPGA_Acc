#!/usr/bin/env python3
"""Recompute the PE16 APM latency correction from immutable UART evidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UART = ROOT / "reports/2026-08-28-zcu102-q8q8-pe16-board-uart.log"
CORRECTION = ROOT / "deployment/artifacts/q8q8_pe16_apm_latency_correction.json"
CASES = ("language_prefill", "vision_encoder", "decoder_gemv")


def fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", line))


def main() -> int:
    data = json.loads(CORRECTION.read_text())
    digest = hashlib.sha256(UART.read_bytes()).hexdigest()
    assert digest == data["source_uart"]["sha256"]
    totals: dict[tuple[str, str], dict[int, int]] = {}
    for line in UART.read_text(errors="replace").splitlines():
        if not line.startswith("APM_TOTAL "):
            continue
        item = fields(line)
        key = (item["case"], item["profile"])
        totals.setdefault(key, {})[int(item["counter"])] = int(item["value"])
    for case in CASES:
        traffic = totals[(case, "traffic")]
        latency = totals[(case, "latency")]
        expected = {
            "cci1_average_read": latency[0] / traffic[1],
            "cci1_max_read": float(latency[2]),
            "cci2_average_read": latency[3] / traffic[3],
            "cci2_max_read": float(latency[5]),
            "hp0_average_read": latency[6] / traffic[5],
            "hp0_average_write": latency[8] / traffic[7],
        }
        actual = data["corrected_latency_cycles"][case]
        for name, value in expected.items():
            assert abs(actual[name] - value) < 1e-12, (case, name, actual[name], value)
    assert data["board_accessed"] is False
    assert data["board_programmed"] is False
    print("PE16_APM_LATENCY_CORRECTION_PASS cases=3 formula=total_over_total board_accessed=no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

