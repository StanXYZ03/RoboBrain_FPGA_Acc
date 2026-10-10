#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "analyze_profile_coverage.py"
SPEC = importlib.util.spec_from_file_location("coverage", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def line(graph: int, node: int, *, wtype: str = "q8_0", m: int = 8,
         n: int = 256, k: int = 128, name: str = "out",
         src1: str = "act", elapsed: int = 100) -> str:
    return (
        f"GGML_OP_PROFILE graph={graph} node={node} op=MUL_MAT name={name} "
        f"out_type=f32 out_ne={n}x{m}x1x1 src0=w src0_type={wtype} "
        f"src0_ne={k}x{n}x1x1 src1={src1} src1_type=f32 "
        f"src1_ne={k}x{m}x1x1 src2=- src2_type=- src2_ne=0x0x0x0 "
        f"elapsed_us={elapsed} flops={2*m*n*k} bytes=1\n"
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        log = root / "profile.log"
        profile = root / "profile.json"
        log.write_text(
            line(1, 0) +
            line(1, 1, wtype="f16") +
            line(1, 2, src1="act (view)") +
            line(1, 3, n=255) +
            line(1, 4, n=9728, k=9728)
        )
        profile.write_text(
            '{"graphs":{"1":{"stage":"vision_prefill"}},'
            '"stage_time_ms":{"vision_prefill":1.0}}\n'
        )
        result = MODULE.analyze(log, profile)
        stage = result["stages"]["vision_prefill"]
        assert result["parsed_mul_mat_records"] == 5
        assert stage["eligible_calls"] == 1
        assert stage["reasons"]["weight_type_not_q8_0"]["calls"] == 1
        assert stage["reasons"]["view_or_permutation"]["calls"] == 1
        assert stage["reasons"]["hardware_limit"]["calls"] == 1
        assert stage["reasons"]["arena_capacity"]["calls"] == 1
        assert "零耗时Amdahl" in MODULE.markdown(result)
    print("GGML_Q8Q8_PROFILE_COVERAGE_TEST_PASS cases=5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
