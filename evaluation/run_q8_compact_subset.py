#!/usr/bin/env python3
"""Run only a compact Q8 path for a preselected attribution subset."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import load_jsonl
from evaluation.run_q8_expanded_validation import ROOT, run_one


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--oracle-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("q8a", "q8b"), required=True)
    parser.add_argument("--retention", type=float, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--model-root", type=Path, default=Path("/home/zhaoxiangyu/RoboBrain2.5/deployment/models"))
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")

    samples = load_jsonl(args.manifest)
    masks = {record["sample_id"]: record for record in load_jsonl(args.oracle_records)}
    missing = [sample["sample_id"] for sample in samples if sample["sample_id"] not in masks]
    if missing:
        raise SystemExit(f"missing masks: {len(missing)}")
    args.output_dir.mkdir(parents=True)
    executable = ROOT / "profiling_src/llama.cpp/build-capture/bin/llama-mtmd-cli"
    executable_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
    mode = f"oracle{int(round(args.retention * 1000)):04d}"
    futures = []
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for ordinal, sample in enumerate(samples, 1):
            mask = masks[sample["sample_id"]]
            futures.append(pool.submit(
                run_one, ordinal=ordinal, total=len(samples), sample=sample,
                mode=mode, stage=args.stage,
                keep=[int(value) for value in mask["selected_group_indices"]],
                full_tokens=int(mask["baseline_llm_visual_tokens"]),
                executable=executable, executable_sha256=executable_sha256,
                model_root=args.model_root, output=args.output_dir,
            ))
        with (args.output_dir / "raw_records.jsonl").open("x", encoding="utf-8") as stream:
            for completed, future in enumerate(as_completed(futures), 1):
                record = future.result()
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                if completed % 12 == 0 or completed == len(futures):
                    failures = sum(item["exit_code"] != 0 for item in records)
                    print(f"PROGRESS {completed}/{len(futures)} failures={failures}", flush=True)
    complete = len(records) == len(samples) and all(
        record["exit_code"] == 0 and record["token_ids"] for record in records
    )
    summary = {
        "format": "q8-compact-attribution-subset-v1",
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "stage": args.stage,
        "mode": mode,
        "sample_count": len(samples),
        "oracle_uses_ground_truth_bbox": True,
        "not_deployable": True,
        "not_a_latency_benchmark": True,
        "executable_sha256": executable_sha256,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"STATUS {summary['status']}", flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
