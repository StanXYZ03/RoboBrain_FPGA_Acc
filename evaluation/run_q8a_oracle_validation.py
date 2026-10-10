#!/usr/bin/env python3
"""Validate real Q8_0 post-merger compaction with frozen development-oracle masks."""

from __future__ import annotations

import argparse
from collections import defaultdict
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


ROOT = Path(__file__).resolve().parents[1]
MARKER = "Q8A_RESPONSE_BEGIN"
TOKEN_PATTERN = re.compile(r"Q8A_TOKEN_IDS=([0-9,]+)")


def select_samples(records: list[dict[str, Any]], limit_per_task: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        task = str(record["task"])
        if counts[task] < limit_per_task:
            selected.append(record)
            counts[task] += 1
    return selected


def response_text(stdout: str) -> str:
    if MARKER not in stdout:
        return ""
    response = stdout.split(MARKER, 1)[1]
    if "Q8A_TOKEN_IDS=" in response:
        response = response.split("Q8A_TOKEN_IDS=", 1)[0]
    return response.strip()


def command_for(executable: Path, model_root: Path, sample: dict[str, Any]) -> list[str]:
    return [
        str(executable),
        "-m", str(model_root / "RoboBrain2.5-4B.Q8_0.gguf"),
        "--mmproj", str(model_root / "RoboBrain2.5-4B.mmproj-Q8_0.gguf"),
        "--image", str(ROOT / "datasets" / sample["image"]),
        "-p", str(sample["prompt"]),
        "-n", str(sample["max_new_tokens"]),
        "-c", "2048", "-b", "512", "-ub", "256", "-t", "4", "-tb", "4",
        "-ngl", "0", "--no-mmproj-offload", "--temp", "0", "--seed", "42",
    ]


def metric_delta(compact: dict[str, Any], stock: dict[str, Any], key: str) -> float:
    return float(compact[key]) - float(stock[key])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("q8a", "q8b"), default="q8a")
    parser.add_argument("--retention", type=float, default=0.75)
    parser.add_argument("--limit-per-task", type=int, default=4)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "datasets/dac_cutpoint_oracle_dev.jsonl",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--oracle-records",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("/home/zhaoxiangyu/RoboBrain2.5/deployment/models"),
    )
    args = parser.parse_args()
    if args.limit_per_task <= 0:
        parser.error("--limit-per-task must be positive")
    if not (0.0 < args.retention < 1.0):
        parser.error("--retention must be in (0, 1)")
    retention_tag = int(round(args.retention * 1000))
    compact_mode = f"oracle{retention_tag:04d}"
    if args.oracle_records is None:
        if retention_tag != 750:
            parser.error("--oracle-records is required when --retention is not 0.75")
        args.oracle_records = ROOT / "evaluation/results/cutpoint_oracle_dev/oracle_post_merger_r0750.jsonl"

    manifest_path = args.manifest
    samples = select_samples(load_jsonl(manifest_path), args.limit_per_task)
    oracle_by_id = {record["sample_id"]: record for record in load_jsonl(args.oracle_records)}
    missing = [sample["sample_id"] for sample in samples if sample["sample_id"] not in oracle_by_id]
    if missing:
        raise SystemExit(f"missing oracle mask for {missing[0]}")

    output = args.output_dir or (
        ROOT / "evaluation/results" / time.strftime(f"{args.stage}_{compact_mode}_%Y%m%dT%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=False)
    executable = ROOT / "profiling_src/llama.cpp/build-capture/bin/llama-mtmd-cli"
    executable_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
    records: list[dict[str, Any]] = []

    for sample_index, sample in enumerate(samples, 1):
        oracle = oracle_by_id[sample["sample_id"]]
        keep = [int(index) for index in oracle["selected_group_indices"]]
        full_tokens = int(oracle["baseline_llm_visual_tokens"])
        for mode in ("stock", compact_mode):
            stem = f"{sample_index:02d}_{sample['task']}_{mode}"
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
            if mode == compact_mode:
                env[f"MTMD_{args.stage.upper()}_KEEP_INDICES"] = ",".join(str(index) for index in keep)
            command = command_for(executable, args.model_root, sample)
            print(
                f"RUN {sample_index}/{len(samples)} task={sample['task']} mode={mode} "
                f"tokens={full_tokens if mode == 'stock' else len(keep)}/{full_tokens}",
                flush=True,
            )
            started = time.monotonic()
            with (output / f"{stem}.stdout").open("x") as stdout_stream, \
                    (output / f"{stem}.stderr").open("x") as stderr_stream:
                try:
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=stdout_stream,
                        stderr=stderr_stream,
                        timeout=1800,
                    )
                    exit_code = completed.returncode
                except subprocess.TimeoutExpired:
                    exit_code = 124
            stdout = (output / f"{stem}.stdout").read_text(errors="replace")
            stderr = (output / f"{stem}.stderr").read_text(errors="replace")
            matches = TOKEN_PATTERN.findall(stdout + stderr)
            token_ids = [int(value) for value in matches[-1].split(",")] if matches else []
            text = response_text(stdout)
            record = {
                "sample_id": sample["sample_id"],
                "task": sample["task"],
                "stage": args.stage,
                "mode": mode,
                "output_text": text,
                "token_ids": token_ids,
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
            records.append(record)
            with (output / "raw_records.jsonl").open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"DONE exit={exit_code} output={text!r}", flush=True)
            if exit_code != 0:
                break
        if records[-1]["exit_code"] != 0:
            break

    by_mode = {
        mode: [record for record in records if record["mode"] == mode]
        for mode in ("stock", compact_mode)
    }
    samples_by_id = {sample["sample_id"]: sample for sample in samples}
    scored = {
        mode: [score_record(samples_by_id[record["sample_id"]], record) for record in mode_records]
        for mode, mode_records in by_mode.items()
    }
    metrics = {mode: summarize(mode_scored) for mode, mode_scored in scored.items()}
    complete = all(
        len(by_mode[mode]) == len(samples)
        and all(record["exit_code"] == 0 and record["token_ids"] for record in by_mode[mode])
        for mode in by_mode
    )
    parse_complete = complete and all(
        item["parse_success"] for mode_scored in scored.values() for item in mode_scored
    )
    quality_pass = False
    deltas: dict[str, float] = {}
    if parse_complete and {"pointing", "grounding"}.issubset(metrics["stock"]["tasks"]):
        stock_point = metrics["stock"]["tasks"]["pointing"]
        compact_point = metrics[compact_mode]["tasks"]["pointing"]
        stock_ground = metrics["stock"]["tasks"]["grounding"]
        compact_ground = metrics[compact_mode]["tasks"]["grounding"]
        deltas = {
            "pck_0_1": metric_delta(compact_point, stock_point, "pck_0_1"),
            "pck_0_2": metric_delta(compact_point, stock_point, "pck_0_2"),
            "iou_mean": metric_delta(compact_ground, stock_ground, "iou_mean"),
            "acc_0_5": metric_delta(compact_ground, stock_ground, "acc_0_5"),
        }
        quality_pass = (
            deltas["pck_0_1"] >= -0.125
            and deltas["pck_0_2"] >= -0.125
            and deltas["iou_mean"] >= -0.05
            and deltas["acc_0_5"] >= -0.125
        )
    status = (
        f"{args.stage.upper()}_{compact_mode.upper()}_SMOKE_CANDIDATE"
        if quality_pass
        else f"{args.stage.upper()}_MECHANICS_PASS_QUALITY_FAIL"
        if parse_complete
        else f"{args.stage.upper()}_COMPACTION_NOT_PASSED"
    )
    summary = {
        "format": f"{args.stage}-{compact_mode}-development-v1",
        "stage": args.stage,
        "requested_retention": args.retention,
        "status": status,
        "development_only": True,
        "not_deployable": True,
        "not_a_latency_benchmark": True,
        "oracle_uses_ground_truth_bbox": True,
        "manifest": str(manifest_path),
        "oracle_records": str(args.oracle_records),
        "executable_sha256": executable_sha256,
        "sample_count": len(samples),
        "metrics": metrics,
        "compact_minus_stock": deltas,
        "gate_thresholds": {
            "pck_0_1_max_drop": 0.125,
            "pck_0_2_max_drop": 0.125,
            "iou_mean_max_drop": 0.05,
            "acc_0_5_max_drop": 0.125,
        },
        "limitations": [
            "Masks were frozen from the BF16 development oracle and use ground-truth boxes.",
            "Process wall time includes model loading and is not a steady-state performance result.",
            "Q8-A reduces only LLM visual prefill; Q8-B reduces vision blocks 2-24 and LLM visual prefill.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"STATUS {status}", flush=True)
    print(f"OUTPUT {output}", flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
