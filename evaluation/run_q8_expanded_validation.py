#!/usr/bin/env python3
"""Run a paired stock/compact Q8 diagnostic with bounded process parallelism."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import load_jsonl, score_record, summarize
from evaluation.run_q8a_oracle_validation import command_for, response_text


ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATTERN = re.compile(r"Q8A_TOKEN_IDS=([0-9,]+)")
MARKER = "Q8A_RESPONSE_BEGIN"


def run_one(
    *,
    ordinal: int,
    total: int,
    sample: dict[str, Any],
    mode: str,
    stage: str,
    keep: list[int],
    full_tokens: int,
    executable: Path,
    executable_sha256: str,
    model_root: Path,
    output: Path,
) -> dict[str, Any]:
    stem = f"{ordinal:03d}_{sample['task']}_{mode}"
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        MTMD_Q8A_PRINT_TOKEN_IDS="1",
        MTMD_TEST_RESPONSE_MARKER=MARKER,
    )
    env.pop("MTMD_Q8A_KEEP_INDICES", None)
    env.pop("MTMD_Q8B_KEEP_INDICES", None)
    env.pop("GGML_CPU_PROFILE", None)
    if mode != "stock":
        env[f"MTMD_{stage.upper()}_KEEP_INDICES"] = ",".join(str(index) for index in keep)
    started = time.monotonic()
    with (output / f"{stem}.stdout").open("x") as stdout_stream, \
            (output / f"{stem}.stderr").open("x") as stderr_stream:
        try:
            completed = subprocess.run(
                command_for(executable, model_root, sample),
                cwd=ROOT, env=env, stdout=stdout_stream, stderr=stderr_stream,
                timeout=1800,
            )
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
    stdout = (output / f"{stem}.stdout").read_text(errors="replace")
    stderr = (output / f"{stem}.stderr").read_text(errors="replace")
    matches = TOKEN_PATTERN.findall(stdout + stderr)
    return {
        "sample_id": sample["sample_id"],
        "task": sample["task"],
        "ordinal": ordinal,
        "stage": stage,
        "mode": mode,
        "output_text": response_text(stdout),
        "token_ids": [int(value) for value in matches[-1].split(",")] if matches else [],
        "exit_code": exit_code,
        "process_wall_s_including_load": time.monotonic() - started,
        "full_visual_tokens": full_tokens,
        "kept_visual_tokens": full_tokens if mode == "stock" else len(keep),
        "actual_retention": 1.0 if mode == "stock" else len(keep) / full_tokens,
        "keep_indices": None if mode == "stock" else keep,
        "executable_sha256": executable_sha256,
        "performance_benchmark": False,
        "development_oracle_uses_ground_truth": mode != "stock",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--oracle-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("q8a", "q8b"), default="q8b")
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
    compact_mode = f"oracle{int(round(args.retention * 1000)):04d}"
    futures = []
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for ordinal, sample in enumerate(samples, 1):
            mask = masks[sample["sample_id"]]
            keep = [int(value) for value in mask["selected_group_indices"]]
            full_tokens = int(mask["baseline_llm_visual_tokens"])
            for mode in ("stock", compact_mode):
                futures.append(pool.submit(
                    run_one, ordinal=ordinal, total=len(samples), sample=sample,
                    mode=mode, stage=args.stage, keep=keep, full_tokens=full_tokens,
                    executable=executable, executable_sha256=executable_sha256,
                    model_root=args.model_root, output=args.output_dir,
                ))
        completed_count = 0
        with (args.output_dir / "raw_records.jsonl").open("x", encoding="utf-8") as stream:
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                completed_count += 1
                if completed_count % 12 == 0 or completed_count == len(futures):
                    failures = sum(item["exit_code"] != 0 for item in records)
                    print(f"PROGRESS {completed_count}/{len(futures)} failures={failures}", flush=True)

    by_mode = {
        mode: [record for record in records if record["mode"] == mode]
        for mode in ("stock", compact_mode)
    }
    sample_by_id = {sample["sample_id"]: sample for sample in samples}
    scored = {
        mode: [score_record(sample_by_id[record["sample_id"]], record) for record in mode_records]
        for mode, mode_records in by_mode.items()
    }
    complete = all(
        len(mode_records) == len(samples)
        and all(record["exit_code"] == 0 and record["token_ids"] for record in mode_records)
        for mode_records in by_mode.values()
    )
    summary = {
        "format": "q8-expanded-paired-diagnostic-v1",
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "expanded_diagnostic_not_blind_validation": True,
        "prior_dataset_use_disclosed": True,
        "oracle_uses_ground_truth_bbox": True,
        "not_deployable": True,
        "not_a_latency_benchmark": True,
        "manifest": str(args.manifest),
        "oracle_records": str(args.oracle_records),
        "executable_sha256": executable_sha256,
        "workers": args.workers,
        "sample_count": len(samples),
        "unique_image_count": len({sample["image"] for sample in samples}),
        "metrics": {mode: summarize(items) for mode, items in scored.items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"STATUS {summary['status']}", flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
