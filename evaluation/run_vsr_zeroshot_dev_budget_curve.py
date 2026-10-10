#!/usr/bin/env python3
"""Run the preregistered VSR zero-shot dev Dense/L2/v2 budget curve."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from evaluation.edge576_cvbench_runner_contract import Edge576Processor, verify_model_identity
from evaluation.run_edge576_cvbench_three_arm import load_arm_model
from evaluation.vsr_zeroshot_dev_contract import (
    ARMS,
    ARM_DENSE,
    MAX_NEW_TOKENS,
    MAX_SWAPS,
    REGION_GRID,
    EVAL_MANIFEST,
    VSRContractError,
    arm_kind_budget,
    canonical_record_sha256,
    load_eval_manifest,
    local_candidate_cap,
    output_path,
    parse_true_false,
    prompt_sha256,
    sanitize_for_selector,
    sha256_file,
    validate_resume,
    verify_preregistration,
    write_eval_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B")
RUN_MODE = "vsr_zeroshot_dev_budget_curve"


def build_state(kind: str, budget: int | None) -> dict[str, Any]:
    if kind == "dense":
        return {"scorer": None, "checkpoint_identity": {}}
    if budget is None:
        raise VSRContractError("pruned arm has no budget")
    if kind == "l2":
        from evaluation.hardware_regular_b1_corrected_l2 import ARM_INTEGER, build_b1_scorer
        from evaluation.run_hf_hardware_regular_b1_corrected_l2 import load_arm_checkpoint

        _, _, identity = load_arm_checkpoint(ARM_INTEGER)
        return {"scorer": build_b1_scorer(ARM_INTEGER, Wt=None, wv_q=None, budgets=(budget,)), "checkpoint_identity": identity}
    from evaluation.conservative_l2_guarded_reranker import build_guarded_scorer
    from evaluation.run_hf_conservative_l2_residual_rerank import load_residual_checkpoint

    Wt, wv_q, identity = load_residual_checkpoint()
    cap = local_candidate_cap(budget)
    scorer = build_guarded_scorer(
        Wt, wv_q, budgets=(budget,), region_grid=REGION_GRID,
        local_candidate_cap=cap, max_swaps=MAX_SWAPS,
    )
    return {"scorer": scorer, "checkpoint_identity": identity}


def run_frozen(kind: str, budget: int | None, model: Any, processor: Any, sample: dict[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    if kind == "dense":
        from evaluation.run_hf_shallow_block_pruning import run_one

        return run_one(
            model, processor, sample, EVAL_MANIFEST, retention=1.0,
            prune_after_layer=5, anchor_grid=3, selector="l2_anchor",
            residual_energy_weight=0.25, text_weight=0.5, selector_head=None,
            selector_feature_mask=(0, 1, 2, 3, 4), joint_candidate_config=None,
            candidate_task_conditioned=False, block_timers=[],
        )
    if budget is None:
        raise VSRContractError("pruned arm has no budget")
    cap = local_candidate_cap(budget)
    if kind == "l2":
        from evaluation.hardware_regular_b1_corrected_l2 import ARM_INTEGER
        from evaluation.run_hf_hardware_regular_b1_corrected_l2 import run_one

        return run_one(
            model, processor, sample, EVAL_MANIFEST, arm=ARM_INTEGER, budget=budget,
            region_grid=REGION_GRID, local_candidate_cap=cap,
            scorer=state["scorer"], manifest_sha256=sha256_file(EVAL_MANIFEST),
            checkpoint_identity=dict(state["checkpoint_identity"]),
        )
    from evaluation.run_hf_conservative_l2_guarded_rerank import run_one

    return run_one(
        model, processor, sample, EVAL_MANIFEST, budget=budget,
        region_grid=REGION_GRID, local_candidate_cap=cap,
        max_swaps=MAX_SWAPS, scorer=state["scorer"],
        manifest_sha256=sha256_file(EVAL_MANIFEST),
        checkpoint_identity=dict(state["checkpoint_identity"]), run_mode=RUN_MODE,
    )


def normalize(arm: str, sample: Mapping[str, Any], frozen: Mapping[str, Any], wall_seconds: float, manifest_sha: str, run_metadata: Mapping[str, Any]) -> dict[str, Any]:
    kind, budget = arm_kind_budget(arm)
    grid = [int(value) for value in frozen["image_grid_thw"]]
    if len(grid) != 3 or grid[0] != 1 or (grid[0] * grid[1] * grid[2]) % 4:
        raise VSRContractError(f"invalid processor grid {grid}")
    full_llm = grid[0] * grid[1] * grid[2] // 4
    selected = [int(value) for value in frozen["selected_group_indices"]]
    bypass = bool(frozen.get("identity_bypass", len(selected) == full_llm))
    expected = full_llm if kind == "dense" or bypass else budget
    if len(selected) != expected:
        raise VSRContractError(f"{arm} emitted K={len(selected)}, expected {expected}")
    parsed = parse_true_false(str(frozen["output_text"]))
    record = dict(frozen)
    record.update(
        {
            "sample_id": sample["sample_id"], "arm": arm, "arm_kind": kind,
            "requested_budget": budget, "run_mode": RUN_MODE,
            "manifest_sha256": manifest_sha, "image_sha256": sample["image_sha256"],
            "relation": sample["relation"], "official_label": int(sample["official_label"]),
            "prompt_sha256": prompt_sha256(str(sample["prompt"])),
            "parsed_answer": parsed, "parse_success": parsed is not None,
            "correct": parsed is not None and parsed == bool(sample["official_label"]),
            "runner_wall_seconds": float(wall_seconds),
            "full_llm_visual_tokens": full_llm, "output_group_count": len(selected),
            "identity_bypass": bypass, "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False, "decoding": "greedy_argmax", "dtype": "bfloat16",
            "attention_implementation": "sdpa", "uses_labels": False,
            "uses_answer_tokens": False, "uses_gradients": False,
            "run": dict(run_metadata),
        }
    )
    record["record_sha256"] = canonical_record_sha256(record)
    return record


def execute_arm(arm: str, model: Any, processor: Any, samples: Sequence[Mapping[str, Any]], manifest_sha: str, run_metadata: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    kind, budget = arm_kind_budget(arm)
    path = output_path(arm)
    done = validate_resume(path, samples, arm, manifest_sha)
    pending = [sample for sample in samples if sample["sample_id"] not in done]
    if not pending:
        return {"arm": arm, "already_done": len(done), "written": 0}
    state = build_state(kind, budget)
    warmup = dict(pending[0])
    warmup["max_new_tokens"] = 1
    run_frozen(kind, budget, model, processor, sanitize_for_selector(warmup), state)
    torch.cuda.synchronize()
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        for index, sample in enumerate(pending, 1):
            sanitized = sanitize_for_selector(sample)
            torch.cuda.synchronize()
            started = time.perf_counter()
            frozen = run_frozen(kind, budget, model, processor, sanitized, state)
            torch.cuda.synchronize()
            wall = time.perf_counter() - started
            record = normalize(arm, sample, frozen, wall, manifest_sha, run_metadata)
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            written += 1
            print(f"RESULT arm={arm} {index}/{len(pending)} id={sample['sample_id']} G={record['full_llm_visual_tokens']} K={record['output_group_count']} tokens={record.get('output_tokens')} wall={wall:.4f}s", flush=True)
    return {"arm": arm, "already_done": len(done), "written": written, "output": str(path)}


def preflight(arms: Sequence[str]) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    for arm in arms:
        arm_kind_budget(arm)
    config = verify_preregistration()
    manifest_sha = sha256_file(EVAL_MANIFEST)
    samples = load_eval_manifest(config["pins"]["eval_manifest"]["sha256"])
    for arm in arms:
        validate_resume(output_path(arm), samples, arm, manifest_sha)
    return samples, manifest_sha, config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-manifest", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_manifest:
        print(json.dumps(write_eval_manifest(), ensure_ascii=False, indent=2))
        return 0
    samples, manifest_sha, _ = preflight(args.arms)
    summary = {arm: len(validate_resume(output_path(arm), samples, arm, manifest_sha)) for arm in args.arms}
    if args.validate_only:
        print(json.dumps({"status": "PASS", "samples": len(samples), "manifest_sha256": manifest_sha, "completed": summary}, indent=2))
        return 0

    import torch
    import transformers

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise VSRContractError("CUDA is required")
    verify_model_identity(args.model)
    model = load_arm_model(args.model, args.device)
    processor = Edge576Processor(transformers.AutoProcessor.from_pretrained(str(args.model), local_files_only=True))
    run_metadata = {
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "device": args.device,
        "model": str(args.model.resolve()), "warmup_per_arm": 1,
        "model_load_excluded": True, "test_split_opened": False,
    }
    results = []
    for arm in args.arms:
        results.append(execute_arm(arm, model, processor, samples, manifest_sha, run_metadata))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
