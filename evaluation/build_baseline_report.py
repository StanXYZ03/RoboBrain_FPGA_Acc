#!/usr/bin/env python3
"""Create a compact Markdown comparison from completed metric summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def value(payload: dict[str, Any], task: str, key: str) -> str:
    raw = payload.get("tasks", {}).get(task, {}).get(key)
    if raw is None:
        return "—"
    return f"{float(raw):.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", action="append", nargs=2, metavar=("LABEL", "SUMMARY"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    rows = []
    for label, path in args.entry:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.append(
            "| " + " | ".join(
                [
                    label,
                    value(payload, "pointing", "parse_rate"),
                    value(payload, "pointing", "pck_0_1"),
                    value(payload, "pointing", "pck_0_2"),
                    value(payload, "pointing", "point_nme_bbox_diag_mean"),
                    value(payload, "pointing", "latency_seconds_median"),
                    value(payload, "grounding", "parse_rate"),
                    value(payload, "grounding", "iou_mean"),
                    value(payload, "grounding", "acc_0_5"),
                    value(payload, "grounding", "latency_seconds_median"),
                ]
            ) + " |"
        )

    markdown = f"""# Qwen / RoboBrain / Q8 spatial smoke baseline

Manifest: `{args.manifest}`

> This COCO-derived set is a reproducible pipeline gate, not the final publication benchmark. Pointing is defined as object-center pointing and does not measure affordance quality.

| Model | Point parse | PCK@0.1 | PCK@0.2 | Point NME / box diag | Point p50 (s) | Box parse | Mean IoU | Acc@0.5 | Box p50 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Interpretation gate

- Compare Qwen BF16 with RoboBrain BF16 to isolate embodied post-training effects at 4B scale.
- Compare RoboBrain BF16 with RoboBrain Q8_0 to isolate the deployed format/runtime effect.
- Latency is reported for audit only: BF16 uses A6000/Transformers while Q8_0 uses host CPU/llama.cpp, so it is not a quantization speed comparison.
- Do not enter token-pruning accuracy claims until all three rows are reproducible and the final held-out benchmark is frozen.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
