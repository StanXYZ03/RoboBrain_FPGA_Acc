#!/usr/bin/env python3
"""SpatialEval VQA 4500 three-arm external-confirmation runner (raw JSONL).

Runs exactly one arm of
``configs/2026-10-09-spatialeval-vqa-external-confirmation.preregistered.json``:

* ``dense_edge576`` - the frozen dense producer at retention ``1.0``;
* ``corrected_l2_integer_b1_k96`` - the frozen corrected integer L2 B1 arm;
* ``v2_edge576_b1_k96`` - the existing frozen
  ``conservative_l2_guarded_rerank_v2`` mechanism.

No selection algorithm is copied or rewritten here.  The frozen Dense / L2 / v2
producers are imported through :func:`evaluation.run_edge576_cvbench_three_arm.run_one_for_arm`
and :func:`evaluation.run_edge576_cvbench_three_arm.build_arm_state`; this module
only adapts the runtime domain to the 4500 SpatialEval VQA samples.

The processor is pinned to ``min_pixels=65536`` / ``max_pixels=589824``, the
prompt is the official ``text + w_reason`` string, decoding is greedy
``do_sample=false`` / ``max_new_tokens=64`` / BF16 / SDPA.  Every sample asserts
a single-image grid within ``G<=576`` (fail closed), records the real output
token fields, never fabricates token ids, never feeds a label into the selector
and writes append-only raw JSONL with a safe resume.

``--validate-only`` / ``--dry-run`` verify the run manifest, the frozen
dependency SHA256 pins, the mechanism identity and the resume state *without
loading a model or touching CUDA*.  In run mode the same CPU-only preflight runs
first, then the ``--model`` identity is verified, and only then are the frozen
scorer and the local model opened.  Each arm runs exactly one real multimodal
warmup (``max_new_tokens=1``, discarded, ``torch.cuda.synchronize()``).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.spatialeval_vqa_contract import (  # noqa: E402
    ARM_IS_NEW_MECHANISM,
    ARM_MECHANISM_IDS,
    ARM_OUTPUT_PATHS,
    ARMS,
    ATTENTION_IMPLEMENTATION,
    DECODING,
    DO_SAMPLE,
    DTYPE,
    K_BUDGET,
    LOCAL_CANDIDATE_CAP,
    MAX_NEW_TOKENS,
    MAX_PIXELS,
    MAX_SWAPS,
    MIN_PIXELS,
    REGION_GRID,
    RUN_MANIFEST,
    RUN_MODE,
    SpatialEvalContractError,
    assert_spatialeval_grid,
    compact_model_identity,
    ensure_output_path,
    load_run_manifest,
    mechanism_identity,
    prompt_sha256,
    record_sha256,
    resolve_output_token_fields,
    sanitize_sample_for_selector,
    sha256_file,
    validate_arm,
    validate_resume,
    verify_dependency_pins,
    verify_model_identity,
    wrap_processor,
)
from evaluation.edge576_cvbench_runner_contract import (  # noqa: E402
    OUTPUT_TOKEN_COUNT_FIELD,
    OUTPUT_TOKEN_IDS_AVAILABLE_FIELD,
    OUTPUT_TOKEN_IDS_FIELD,
)
from evaluation.run_edge576_cvbench_three_arm import (  # noqa: E402
    build_arm_state,
    run_one_for_arm,
)

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = Path("/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B")
DEFAULT_MODEL_LABEL = "RoboBrain2.5-4B-BF16"

RunOne = Callable[[Any, Any, dict[str, Any], Path], dict[str, Any]]


# ---------------------------------------------------------------------------
# Record normalisation + append-only execution
# ---------------------------------------------------------------------------
def normalize_record(
    arm: str,
    sample: Mapping[str, Any],
    frozen_record: Mapping[str, Any],
    *,
    manifest_sha256: str,
    run_metadata: Mapping[str, Any],
    run_mode: str,
) -> dict[str, Any]:
    """Attach the SpatialEval contract fields to a frozen producer record.

    ``output_text`` and ``selected_group_indices`` are mandatory.  Token ids are
    optional (the dense producer never emits them) and are never fabricated; the
    mandatory ``output_token_count`` is mapped from the real frozen fields.
    """

    record = dict(frozen_record)
    for required in ("output_text", "selected_group_indices"):
        if required not in record:
            raise SpatialEvalContractError(
                f"frozen {arm} record is missing {required!r} (fail closed)"
            )
    if "image_grid_thw" not in record:
        raise SpatialEvalContractError(
            f"frozen {arm} record is missing image_grid_thw (fail closed)"
        )
    token_ids, token_count = resolve_output_token_fields(frozen_record)
    if token_ids is None:
        record.pop(OUTPUT_TOKEN_IDS_FIELD, None)
    else:
        record[OUTPUT_TOKEN_IDS_FIELD] = token_ids
    record[OUTPUT_TOKEN_COUNT_FIELD] = token_count
    record[OUTPUT_TOKEN_IDS_AVAILABLE_FIELD] = token_ids is not None
    record["sample_id"] = sample["sample_id"]
    record["task"] = sample["task"]
    record["question_id"] = int(sample["question_id"])
    record["arm"] = arm
    record["mechanism_id"] = ARM_MECHANISM_IDS[arm]
    record["manifest_sha256"] = manifest_sha256
    record["min_pixels"] = MIN_PIXELS
    record["max_pixels"] = MAX_PIXELS
    record["image"] = sample["image"]
    record["image_sha256"] = sample["image_sha256"]
    record["width"] = int(sample["width"])
    record["height"] = int(sample["height"])
    grid = [int(value) for value in record["image_grid_thw"]]
    record["image_grid_thw"] = grid
    full_llm = assert_spatialeval_grid(sample, grid)
    record["full_llm_visual_tokens"] = full_llm
    selected = [int(value) for value in record["selected_group_indices"]]
    record["selected_group_indices"] = selected
    record["output_group_count"] = len(selected)
    record["identity_bypass"] = bool(
        record.get("identity_bypass", len(selected) == full_llm)
    )
    record["prompt_sha256"] = prompt_sha256(sample["prompt"])
    record["max_new_tokens"] = int(sample["max_new_tokens"])
    record["do_sample"] = DO_SAMPLE
    record["decoding"] = DECODING
    record["dtype"] = DTYPE
    record["attention_implementation"] = ATTENTION_IMPLEMENTATION
    record["uses_labels"] = False
    record["uses_answer_tokens"] = False
    record["uses_gradients"] = False
    record["run_mode"] = run_mode
    record["run"] = dict(run_metadata)
    record["record_sha256"] = record_sha256(record)
    return record


def execute_arm(
    *,
    arm: str,
    model: Any,
    processor: Any,
    samples: Sequence[Mapping[str, Any]],
    output: Path,
    manifest_path: Path,
    manifest_sha256: str,
    run_one: RunOne,
    run_metadata: Mapping[str, Any],
    run_mode: str,
) -> dict[str, Any]:
    """Append pending samples one at a time with safe resume (fail closed)."""

    validate_arm(arm)
    output = Path(output)
    done = validate_resume(
        output, samples, arm=arm, manifest_sha256=manifest_sha256
    )
    pending = [
        sample for sample in samples if str(sample["sample_id"]) not in done
    ]
    stats: dict[str, Any] = {
        "arm": arm,
        "mechanism_id": ARM_MECHANISM_IDS[arm],
        "run_mode": run_mode,
        "manifest_sha256": manifest_sha256,
        "total": len(samples),
        "already_done": len(done),
        "pending": len(pending),
        "written": 0,
        "output_path": str(output),
    }
    if not pending:
        print(
            f"LOAD arm={arm} samples={len(samples)} done={len(done)} pending=0",
            flush=True,
        )
        return stats
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", buffering=1) as stream:
        for index, sample in enumerate(pending, start=1):
            sanitized = sanitize_sample_for_selector(sample)
            frozen_record = run_one(model, processor, sanitized, manifest_path)
            record = normalize_record(
                arm,
                sample,
                frozen_record,
                manifest_sha256=manifest_sha256,
                run_metadata=run_metadata,
                run_mode=run_mode,
            )
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            stats["written"] += 1
            print(
                "RESULT "
                f"{index}/{len(pending)} id={sample['sample_id']} arm={arm} "
                f"G={record['full_llm_visual_tokens']} "
                f"K={record['output_group_count']} "
                f"output={record['output_text']!r}",
                flush=True,
            )
    return stats


# ---------------------------------------------------------------------------
# CUDA / model loading / real multimodal warmup
# ---------------------------------------------------------------------------
def require_cuda(device: str) -> None:
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise SpatialEvalContractError("this experiment requires a CUDA device")


def load_arm_model(model_path: Path, device: str) -> Any:
    import torch  # noqa: PLC0415
    from transformers import AutoModelForImageTextToText  # noqa: PLC0415

    torch.manual_seed(0)
    return AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation=ATTENTION_IMPLEMENTATION,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()


def load_arm_processor(model_path: Path) -> Any:
    from transformers import AutoProcessor  # noqa: PLC0415

    return wrap_processor(
        AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    )


def run_multimodal_warmup(
    arm: str,
    model: Any,
    processor: Any,
    sample: Mapping[str, Any],
    manifest_path: Path,
    *,
    manifest_sha256: str,
    scorer: Any,
    checkpoint_identity: Mapping[str, Any],
    run_mode: str,
    max_new_tokens: int = 1,
    torch_module: Any = None,
) -> dict[str, Any]:
    """Run exactly one real multimodal warmup for ``arm`` and discard it."""

    validate_arm(arm)
    if torch_module is None:
        import torch as torch_module  # noqa: PLC0415

    warmup_sample = dict(sample)
    warmup_sample["max_new_tokens"] = int(max_new_tokens)
    sanitized = sanitize_sample_for_selector(warmup_sample)
    import time  # noqa: PLC0415

    started = time.perf_counter()
    frozen_record = run_one_for_arm(
        arm,
        model,
        processor,
        sanitized,
        manifest_path,
        manifest_sha256=manifest_sha256,
        scorer=scorer,
        checkpoint_identity=checkpoint_identity,
        run_mode=run_mode,
    )
    torch_module.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started
    full_llm = assert_spatialeval_grid(sample, frozen_record.get("image_grid_thw"))
    return {
        "arm": arm,
        "mechanism_id": ARM_MECHANISM_IDS[arm],
        "count": 1,
        "sample_id": str(sample["sample_id"]),
        "max_new_tokens": int(max_new_tokens),
        "elapsed_seconds": elapsed_seconds,
        "full_llm_visual_tokens": full_llm,
        "output_text": frozen_record.get("output_text"),
        "wrote_record": False,
        "output_written": False,
        "cuda_synchronized": True,
    }


# ---------------------------------------------------------------------------
# Validation / dry-run summary
# ---------------------------------------------------------------------------
def build_run_metadata(
    arm: str,
    *,
    model_path: Path,
    model_label: str,
    device: str,
    manifest_sha256: str,
    checkpoint_identity: Mapping[str, Any],
    model_identity: Mapping[str, Any] | None = None,
    warmup: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415

    metadata = {
        "model_label": model_label,
        "model_path": str(Path(model_path).resolve()),
        "dtype": DTYPE,
        "device": device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "arm": arm,
        "mechanism_id": ARM_MECHANISM_IDS[arm],
        "is_new_mechanism": bool(ARM_IS_NEW_MECHANISM[arm]),
        "budget": K_BUDGET,
        "region_grid": REGION_GRID,
        "local_candidate_cap": LOCAL_CANDIDATE_CAP,
        "max_swaps": MAX_SWAPS if arm == "v2_edge576_b1_k96" else None,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "manifest_sha256": manifest_sha256,
        "do_sample": DO_SAMPLE,
        "decoding": DECODING,
        "max_new_tokens": MAX_NEW_TOKENS,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "run_mode": RUN_MODE,
        "training": False,
        "uses_labels": False,
        "uses_answer_tokens": False,
        "uses_gradients": False,
        "official_w_reason_prompt": True,
        "labels_observed": True,
        "not_label_blind": True,
        "not_used_for_selector_tuning": True,
    }
    metadata.update(dict(checkpoint_identity))
    metadata["model_identity"] = (
        compact_model_identity(model_identity) if model_identity is not None else None
    )
    metadata["model_identity_verified"] = model_identity is not None
    metadata["warmup"] = dict(warmup) if warmup is not None else None
    metadata["warmup_count"] = 1 if warmup is not None else 0
    return metadata


def validation_summary(
    arm: str,
    samples: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    """CPU-only verification of inputs, parameters, outputs, identity and pins."""

    validate_arm(arm)
    dependencies = verify_dependency_pins()
    identity = mechanism_identity(arm)
    manifest_sha256 = sha256_file(Path(manifest_path))
    done = validate_resume(
        output, samples, arm=arm, manifest_sha256=manifest_sha256
    )
    g_values = [int(sample["full_llm_visual_tokens"]) for sample in samples]
    return {
        "arm": arm,
        "mechanism_id": ARM_MECHANISM_IDS[arm],
        "is_new_mechanism": bool(ARM_IS_NEW_MECHANISM[arm]),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "sample_count": len(samples),
        "unique_sample_id_count": len({str(s["sample_id"]) for s in samples}),
        "all_g_within_budget": max(g_values) <= 576,
        "g_min": min(g_values),
        "g_max": max(g_values),
        "processor": {
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
            "keep_aspect_ratio": True,
            "patch_size": 16,
            "merge_size": 2,
        },
        "locked_parameters": {
            "K": K_BUDGET,
            "region_grid": REGION_GRID,
            "local_candidate_cap": LOCAL_CANDIDATE_CAP,
            "max_swaps": MAX_SWAPS,
            "do_sample": DO_SAMPLE,
            "decoding": DECODING,
            "max_new_tokens": MAX_NEW_TOKENS,
            "dtype": DTYPE,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
        },
        "output_path": str(output),
        "existing_done_count": len(done),
        "pending_count": len(samples) - len(done),
        "dependency_pins": dependencies,
        "mechanism_identity": identity,
        "weights_loaded": False,
        "model_loaded": False,
        "gpu_runs": 0,
        "cuda_required": False,
    }


def planned_command(arm: str, output: Path, manifest_path: Path) -> str:
    try:
        rel_output = output.relative_to(ROOT)
    except ValueError:
        rel_output = output
    return (
        "python evaluation/run_spatialeval_vqa_three_arm.py"
        f" --arm {arm}"
        f" --model {DEFAULT_MODEL_PATH}"
        f" --model-label {DEFAULT_MODEL_LABEL}"
        f" --manifest {manifest_path}"
        f" --output {rel_output}"
    )


def run_arm(
    *,
    arm: str,
    samples: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    output: Path,
    model_path: Path | None,
    model_label: str,
    device: str,
    run_mode: str = RUN_MODE,
    validation_fn: Callable[..., dict[str, Any]] = validation_summary,
    model_identity_fn: Callable[..., dict[str, Any]] = verify_model_identity,
    require_cuda_fn: Callable[[str], None] = require_cuda,
    build_state_fn: Callable[[str], dict[str, Any]] = build_arm_state,
    load_model_fn: Callable[..., Any] = load_arm_model,
    load_processor_fn: Callable[..., Any] = load_arm_processor,
    warmup_fn: Callable[..., dict[str, Any]] = run_multimodal_warmup,
    execute_fn: Callable[..., dict[str, Any]] = execute_arm,
) -> dict[str, Any]:
    """Run one arm with a strict fail-closed preflight before any model load."""

    validate_arm(arm)
    manifest_path = Path(manifest_path)
    output = Path(output)
    manifest_sha256 = sha256_file(manifest_path)

    validation = validation_fn(
        arm, samples, manifest_path=manifest_path, output=output
    )
    pending = int(validation["pending_count"])
    if pending == 0:
        return {
            "status": "SPATIALEVAL_RUNNER_ARM_ALREADY_COMPLETE",
            "arm": arm,
            "mechanism_id": ARM_MECHANISM_IDS[arm],
            "run_mode": run_mode,
            "manifest_sha256": manifest_sha256,
            "total": len(samples),
            "already_done": int(validation["existing_done_count"]),
            "pending": 0,
            "written": 0,
            "output_path": str(output),
            "validation": validation,
            "model_identity": None,
            "warmup": None,
            "model_identity_verified": False,
            "warmup_count": 0,
            "weights_loaded": False,
            "model_loaded": False,
            "gpu_runs": 0,
        }

    if model_path is None:
        raise SpatialEvalContractError(
            "--model is required to run an arm (fail closed)"
        )
    require_cuda_fn(device)
    model_identity = model_identity_fn(model_path)
    state = build_state_fn(arm)
    model = load_model_fn(model_path, device)
    processor = load_processor_fn(model_path)

    if not samples:
        raise SpatialEvalContractError("cannot warm up an empty sample set (fail closed)")
    warmup = warmup_fn(
        arm,
        model,
        processor,
        samples[0],
        manifest_path,
        manifest_sha256=manifest_sha256,
        scorer=state["scorer"],
        checkpoint_identity=state["checkpoint_identity"],
        run_mode=run_mode,
    )
    run_metadata = build_run_metadata(
        arm,
        model_path=Path(model_path),
        model_label=model_label,
        device=device,
        manifest_sha256=manifest_sha256,
        checkpoint_identity=state["checkpoint_identity"],
        model_identity=model_identity,
        warmup=warmup,
    )

    def run_one(
        current_model: Any,
        current_processor: Any,
        sample: dict[str, Any],
        manifest: Path,
    ) -> dict[str, Any]:
        return run_one_for_arm(
            arm,
            current_model,
            current_processor,
            sample,
            manifest,
            manifest_sha256=manifest_sha256,
            scorer=state["scorer"],
            checkpoint_identity=state["checkpoint_identity"],
            run_mode=run_mode,
        )

    stats = execute_fn(
        arm=arm,
        model=model,
        processor=processor,
        samples=samples,
        output=output,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        run_one=run_one,
        run_metadata=run_metadata,
        run_mode=run_mode,
    )
    stats["status"] = "SPATIALEVAL_RUNNER_ARM_COMPLETE"
    stats["manifest_sha256"] = manifest_sha256
    stats["validation"] = validation
    stats["model_identity"] = compact_model_identity(model_identity)
    stats["warmup"] = dict(warmup)
    stats["model_identity_verified"] = True
    stats["warmup_count"] = 1
    stats["weights_loaded"] = True
    stats["model_loaded"] = True
    stats["gpu_runs"] = 1
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    parser.add_argument("--manifest", type=Path, default=RUN_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16",), default=DTYPE)
    parser.add_argument("--budget", type=int, default=K_BUDGET)
    parser.add_argument("--region-grid", type=int, default=REGION_GRID)
    parser.add_argument("--local-candidate-cap", type=int, default=LOCAL_CANDIDATE_CAP)
    parser.add_argument("--max-swaps", type=int, default=MAX_SWAPS)
    parser.add_argument("--min-pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def _enforce_locks(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.budget != K_BUDGET:
        parser.error(f"--budget is frozen at {K_BUDGET}")
    if args.region_grid != REGION_GRID:
        parser.error(f"--region-grid is frozen at {REGION_GRID}")
    if args.local_candidate_cap != LOCAL_CANDIDATE_CAP:
        parser.error(f"--local-candidate-cap is frozen at {LOCAL_CANDIDATE_CAP}")
    if args.max_swaps != MAX_SWAPS:
        parser.error(f"--max-swaps is frozen at {MAX_SWAPS}")
    if args.min_pixels != MIN_PIXELS:
        parser.error(f"--min-pixels is frozen at {MIN_PIXELS}")
    if args.max_pixels != MAX_PIXELS:
        parser.error(f"--max-pixels is frozen at {MAX_PIXELS}")
    if args.dtype != DTYPE:
        parser.error(f"--dtype is frozen at {DTYPE}")
    if args.validate_only and args.dry_run:
        parser.error("choose either --validate-only or --dry-run, not both")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _enforce_locks(parser, args)

    manifest_path = Path(args.manifest)
    samples = load_run_manifest(manifest_path)
    output = ensure_output_path(args.arm, args.output)

    if args.validate_only or args.dry_run:
        summary = validation_summary(
            args.arm, samples, manifest_path=manifest_path, output=output
        )
        summary["status"] = (
            "SPATIALEVAL_RUNNER_VALIDATE_ONLY_OK"
            if args.validate_only
            else "SPATIALEVAL_RUNNER_DRY_RUN_OK"
        )
        if args.dry_run:
            summary["planned_command"] = planned_command(args.arm, output, manifest_path)
        report = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        if args.json_out is not None:
            json_out = Path(args.json_out)
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(report + "\n", encoding="utf-8")
        print(report, flush=True)
        return 0

    result = run_arm(
        arm=args.arm,
        samples=samples,
        manifest_path=manifest_path,
        output=output,
        model_path=args.model,
        model_label=args.model_label,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
