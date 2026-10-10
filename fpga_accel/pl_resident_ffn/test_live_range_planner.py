#!/usr/bin/env python3
"""Deterministic and property tests for the PL-resident FFN arena planner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("live_range_planner.py")
SPEC = importlib.util.spec_from_file_location("live_range_planner", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def main() -> int:
    capacity = 64 * 1024 * 1024
    llm = MODULE.maximum_chunk("llm_swiglu", 1080, capacity)
    vision = MODULE.maximum_chunk("vision_gelu", 4320, capacity)
    assert llm["m"] % 16 == 0 and llm["chunk_count"] >= 2
    assert vision["m"] % 16 == 0 and vision["chunk_count"] >= 2
    for kind, total in (("llm_swiglu", 1080), ("vision_gelu", 4320)):
        previous = 0
        for m in list(range(1, 65)) + [127, 128, 255, 256, 511, 512, total]:
            plan = MODULE.allocate(kind, m)
            assert plan["arena_bytes"] >= previous
            previous = plan["arena_bytes"]
            if plan["arena_bytes"] <= capacity:
                MODULE.validate(plan, capacity)
    assert MODULE.allocate("llm_swiglu", llm["m"] + 16)["arena_bytes"] > capacity
    assert MODULE.allocate("vision_gelu", vision["m"] + 16)["arena_bytes"] > capacity
    print(
        "PL_RESIDENT_FFN_PLANNER_TEST_PASS "
        f"llm_chunk={llm['m']} llm_chunks={llm['chunk_count']} "
        f"vision_chunk={vision['m']} vision_chunks={vision['chunk_count']} "
        "alignment=64 overlap=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
