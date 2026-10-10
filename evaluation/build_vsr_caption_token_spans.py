#!/usr/bin/env python3
"""Map each VSR raw prompt embedding to its variable statement token span."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from evaluation.extract_vsr_b1_margin_teacher import DEFAULT_MANIFEST, DEFAULT_MODEL, sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/token_selector/2026-10-10-vsr-b1-margin-teacher/caption_token_spans.json"
REPORT = ROOT / "reports/token_selector/2026-10-10-vsr-caption-token-spans.json"
PREFIX = "Statement: "
SUFFIX = "\nBased only on the image, is this statement true or false? Answer with exactly True or False."


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_MODEL), local_files_only=True, use_fast=True)
    rows = [json.loads(line) for line in DEFAULT_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    spans = {}; lengths = []
    for row in rows:
        prompt = str(row["prompt"]); caption = str(row["caption"])
        if prompt != PREFIX + caption + SUFFIX:
            raise ValueError(f"prompt template mismatch: {row['sample_id']}")
        start_char, end_char = len(PREFIX), len(PREFIX) + len(caption)
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        selected = [index for index, (start, end) in enumerate(offsets) if end > start_char and start < end_char]
        if not selected or selected != list(range(selected[0], selected[-1] + 1)):
            raise ValueError(f"caption token span is empty/non-contiguous: {row['sample_id']}")
        decoded = tokenizer.decode(encoded["input_ids"][selected[0]:selected[-1] + 1]).strip()
        if caption.strip() not in decoded and decoded not in caption.strip():
            raise ValueError(f"caption span decode mismatch: {row['sample_id']} {decoded!r}")
        spans[str(row["sample_id"])] = {"start": selected[0], "end_exclusive": selected[-1] + 1,
                                        "full_token_count": len(encoded["input_ids"]), "caption_token_count": len(selected)}
        lengths.append(len(selected))
    OUTPUT.write_text(json.dumps({"schema": "half-open token spans over raw prompt embeddings", "spans": spans}, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    report = {"status": "VSR_CAPTION_TOKEN_SPANS_COMPLETE", "records": len(spans),
              "caption_token_min": min(lengths), "caption_token_max": max(lengths),
              "caption_token_mean": sum(lengths) / len(lengths), "output": str(OUTPUT),
              "output_sha256": sha256_file(OUTPUT), "uses_relation_annotation": False,
              "uses_label": False, "rule": "character offsets of the variable caption inside the fixed raw Prompt template"}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
