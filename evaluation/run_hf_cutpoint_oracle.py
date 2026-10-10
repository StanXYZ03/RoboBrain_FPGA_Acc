#!/usr/bin/env python3
"""Run a resumable geometric-oracle scan at several vision cut points."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluation.cutpoint_oracle import CUT_POINTS, cut_point, transfer_accounting
from evaluation.joint_deletion import require_development_manifest
from evaluation.metrics import load_jsonl
from evaluation.run_hf_shallow_block_pruning import (
    CudaCallTimer,
    completed_ids,
    limit_each_task,
    run_one,
)
from evaluation.token_reduction import retention_label


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output_path(output_dir: Path, cut_name: str, retention: float) -> Path:
    return output_dir / f"oracle_{cut_name}_{retention_label(retention)}.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-revealed-diagnostic", action="store_true")
    parser.add_argument("--expected-masks", type=Path)
    parser.add_argument(
        "--cut-points", nargs="+", choices=tuple(CUT_POINTS), default=tuple(CUT_POINTS)
    )
    parser.add_argument(
        "--retentions", type=float, nargs="+", default=(1.0, 0.75, 0.5, 0.25)
    )
    parser.add_argument("--limit-per-task", type=int, default=4)
    parser.add_argument("--anchor-grid", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()
    manifest_records = load_jsonl(args.manifest)
    manifest_role = str(manifest_records[0].get("dataset_role"))
    if args.allow_revealed_diagnostic:
        if manifest_role != "selector_validation":
            parser.error("revealed diagnostic mode requires dataset_role=selector_validation")
        if args.expected_masks is None:
            parser.error("revealed diagnostic mode requires --expected-masks")
    else:
        try:
            require_development_manifest(str(args.manifest))
        except ValueError as error:
            parser.error(str(error))
        if manifest_role != "cutpoint_oracle_dev":
            parser.error("manifest must declare dataset_role=cutpoint_oracle_dev")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("cut-point oracle scan requires CUDA")
    for retention in args.retentions:
        retention_label(retention)

    samples = limit_each_task(manifest_records, args.limit_per_task)
    expected_masks = (
        {record["sample_id"]: record for record in load_jsonl(args.expected_masks)}
        if args.expected_masks is not None else {}
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending: dict[tuple[str, float], tuple[Path, list[dict[str, Any]]]] = {}
    for cut_name in args.cut_points:
        for retention in args.retentions:
            path = output_path(args.output_dir, cut_name, retention)
            done = completed_ids(path)
            work = [sample for sample in samples if sample["sample_id"] not in done]
            pending[(cut_name, retention)] = path, work
    pending_total = sum(len(work) for _, work in pending.values())
    print(
        f"LOAD samples={len(samples)} configs={len(pending)} pending={pending_total}",
        flush=True,
    )
    if pending_total == 0:
        return 0

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model),
        dtype=dtype,
        device_map=args.device,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    block_timers = [CudaCallTimer(block) for block in model.model.visual.blocks]
    common_run = {
        "format": "cutpoint-geometric-oracle-v1",
        "status": "DEVELOPMENT_ORACLE_NOT_DEPLOYABLE",
        "model_label": args.model_label,
        "model_path": str(args.model.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "dtype": args.dtype,
        "device": args.device,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "selector": "geometric_oracle",
        "oracle_uses_ground_truth_bbox": True,
        "original_position_ids_preserved": True,
        "custom_greedy_decode": True,
        "revealed_diagnostic": args.allow_revealed_diagnostic,
        "expected_masks_sha256": (
            file_sha256(args.expected_masks) if args.expected_masks is not None else None
        ),
    }
    try:
        for config_index, ((cut_name, retention), (path, work)) in enumerate(
            pending.items(), 1
        ):
            print(
                f"CONFIG {config_index}/{len(pending)} cut={cut_name} "
                f"retention={retention:.3f} pending={len(work)}",
                flush=True,
            )
            if not work:
                continue
            point = cut_point(cut_name)
            # Warm the exact shape for this configuration without recording it.
            warmup = run_one(
                model,
                processor,
                work[0],
                args.manifest,
                retention=retention,
                prune_after_layer=point.prune_after_layer,
                anchor_grid=args.anchor_grid,
                selector="geometric_oracle",
                residual_energy_weight=0.0,
                text_weight=0.0,
                selector_head=None,
                selector_feature_mask=(),
                joint_candidate_config=None,
                candidate_task_conditioned=False,
                block_timers=block_timers,
            )
            print(f"WARMUP output={warmup['output_text']!r}", flush=True)
            with path.open("a", encoding="utf-8", buffering=1) as stream:
                for sample_index, sample in enumerate(work, 1):
                    record = run_one(
                        model,
                        processor,
                        sample,
                        args.manifest,
                        retention=retention,
                        prune_after_layer=point.prune_after_layer,
                        anchor_grid=args.anchor_grid,
                        selector="geometric_oracle",
                        residual_energy_weight=0.0,
                        text_weight=0.0,
                        selector_head=None,
                        selector_feature_mask=(),
                        joint_candidate_config=None,
                        candidate_task_conditioned=False,
                        block_timers=block_timers,
                    )
                    record["cut_point"] = {
                        "name": point.name,
                        "prune_after_layer_zero_based": point.prune_after_layer,
                        "completed_vision_blocks": point.completed_vision_blocks,
                        "transfer_space": point.transfer_space,
                    }
                    record["communication"] = transfer_accounting(
                        cut_name=cut_name,
                        full_patch_tokens=int(record["full_encoder_tokens"]),
                        kept_groups=int(record["llm_visual_tokens"]),
                    )
                    expected_mask = expected_masks.get(sample["sample_id"])
                    if expected_mask is not None and abs(
                        retention - float(expected_mask["requested_retention"])
                    ) < 1e-12:
                        if record["selected_group_indices"] != expected_mask["selected_group_indices"]:
                            raise RuntimeError(
                                f"mask mismatch for {sample['sample_id']} at retention {retention}"
                            )
                        record["expected_mask_match"] = True
                    record["run"] = common_run
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(
                        f"RESULT {sample_index}/{len(work)} id={sample['sample_id']} "
                        f"vision={record['vision_encoder_seconds']:.4f}s "
                        f"prefill={record['llm_prefill_seconds']:.4f}s "
                        f"total={record['elapsed_seconds']:.4f}s "
                        f"output={record['output_text']!r}",
                        flush=True,
                    )
    finally:
        for timer in block_timers:
            timer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
