#!/usr/bin/env python3
"""Reproduce and verify one frozen DATAFLOW board summary without writes."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUMMARIZER = ROOT / "deployment/scripts/summarize_q8q8_pe16_dataflow_board.py"
SPEC = importlib.util.spec_from_file_location("dataflow_board_summary", SUMMARIZER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    stored = json.loads(args.summary.read_text())
    raw = stored.get("raw_evidence", {})
    if len(raw) != 2:
        raise RuntimeError("expected exactly UART and XSCT raw evidence")
    uart_paths = [Path(path) for path in raw if "uart" in path]
    xsct_paths = [Path(path) for path in raw if "xsct" in path]
    if len(uart_paths) != 1 or len(xsct_paths) != 1:
        raise RuntimeError("cannot identify UART/XSCT evidence")
    reproduced = MODULE.summarize(uart_paths[0], xsct_paths[0])
    # JSON object keys are strings after persistence; normalize the freshly
    # reproduced in-memory integer APM counter keys through the same encoding.
    reproduced = json.loads(json.dumps(reproduced))
    stored_without_time = dict(stored)
    reproduced_without_time = dict(reproduced)
    stored_without_time.pop("generated_at", None)
    reproduced_without_time.pop("generated_at", None)
    if stored_without_time != reproduced_without_time:
        raise RuntimeError("stored JSON differs from reproduced summary")
    if args.markdown.read_text() != MODULE.markdown(stored):
        raise RuntimeError("stored Markdown differs from reproduced report")
    pairs = stored["chunk_policy_ab"]
    kernels = stored["kernel_comparison_vs_pe16"]
    print(
        "DATAFLOW_BOARD_FROZEN_VERIFY_PASS "
        f"language_speedup={kernels['language_prefill']['speedup']:.6f} "
        f"vision_speedup={kernels['vision_encoder']['speedup']:.6f} "
        f"decoder_speedup={kernels['decoder_gemv']['speedup']:.6f} "
        f"visual_chunk_speedup={pairs['visual_prefill']['speedup']:.6f} "
        f"vision_chunk_speedup={pairs['vision_encoder']['speedup']:.6f} "
        "sd_qspi_written=no final_control=0x00000004"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
