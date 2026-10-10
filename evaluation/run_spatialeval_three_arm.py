#!/usr/bin/env python3
"""Run one frozen Dense/L2/v2 arm on SpatialEval VQA primary test4500."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from evaluation.edge576_cvbench_runner_contract import (  # noqa: E402
    resolve_output_token_fields,
    verify_model_identity,
)
from evaluation.run_edge576_cvbench_three_arm import (  # noqa: E402
    build_arm_state,
    load_arm_model,
    load_arm_processor,
    run_one_for_arm,
)
from evaluation.spatialeval_external_contract import (  # noqa: E402
    ARM_DENSE,
    ARM_L2,
    ARM_MECHANISM_IDS,
    ARM_OUTPUT_PATHS,
    ARM_V2,
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
    PRIMARY_SAMPLE_COUNT,
    REFERENCE_MANIFEST,
    REGION_GRID,
    SpatialEvalExternalContractError,
    canonical_record_sha256,
    ensure_output_path,
    load_reference_manifest,
    materialize_primary_samples,
    prompt_sha256,
    sanitize_sample_for_selector,
    sha256_file,
    verify_parquet_shards,
    verify_preregistered_pins,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path("/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B")
RUN_MODE = "spatialeval_vqa_external_test4500"


def validate_actual_grid(grid: Sequence[int]) -> tuple[list[int], int]:
    values = [int(value) for value in grid]
    if len(values) != 3 or values[0] != 1:
        raise SpatialEvalExternalContractError(
            f"processor grid must describe one image, got {values}"
        )
    full_patch_tokens = values[0] * values[1] * values[2]
    if full_patch_tokens % 4:
        raise SpatialEvalExternalContractError(
            f"processor grid {values} is not aligned to merge unit 4"
        )
    full_llm = full_patch_tokens // 4
    if full_llm > 576:
        raise SpatialEvalExternalContractError(
            f"edge576 processor produced G={full_llm} > 576"
        )
    return values, full_llm


def normalize_output_record(
    arm: str,
    sample: Mapping[str, Any],
    frozen: Mapping[str, Any],
    *,
    manifest_sha256: str,
    run_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    for field in ("output_text", "image_grid_thw", "selected_group_indices"):
        if field not in frozen:
            raise SpatialEvalExternalContractError(
                f"frozen {arm} output missing {field!r}"
            )
    token_ids, token_count = resolve_output_token_fields(frozen)
    grid, full_llm = validate_actual_grid(frozen["image_grid_thw"])
    selected = [int(value) for value in frozen["selected_group_indices"]]
    if selected and not all(b > a for a, b in zip(selected, selected[1:])):
        raise SpatialEvalExternalContractError("selected indices are not increasing")
    identity_bypass = bool(frozen.get("identity_bypass", len(selected) == full_llm))
    if arm == ARM_DENSE and len(selected) != full_llm:
        raise SpatialEvalExternalContractError(
            f"dense arm selected {len(selected)} groups but G={full_llm}"
        )
    if arm in (ARM_L2, ARM_V2):
        expected = full_llm if identity_bypass else K_BUDGET
        if len(selected) != expected:
            raise SpatialEvalExternalContractError(
                f"{arm} selected {len(selected)} groups, expected {expected}"
            )

    record = dict(frozen)
    if token_ids is None:
        record.pop("output_token_ids", None)
    else:
        record["output_token_ids"] = token_ids
    record.update(
        {
            "sample_id": str(sample["sample_id"]),
            "id": str(sample["sample_id"]),
            "task": str(sample["task"]),
            "question_id": int(sample["question_id"]),
            "arm": arm,
            "mechanism_id": ARM_MECHANISM_IDS[arm],
            "run_mode": RUN_MODE,
            "manifest_sha256": manifest_sha256,
            "image_sha256": str(sample["image_sha256"]),
            "prompt_sha256": prompt_sha256(str(sample["prompt"])),
            "oracle_answer": sample["oracle_answer"],
            "oracle_option": sample["oracle_option"],
            "oracle_full_answer": sample["oracle_full_answer"],
            "image_grid_thw": grid,
            "full_llm_visual_tokens": full_llm,
            "output_group_count": len(selected),
            "selected_group_indices": selected,
            "identity_bypass": identity_bypass,
            "output_token_count": token_count,
            "output_token_ids_available": token_ids is not None,
            "max_new_tokens": MAX_NEW_TOKENS,
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
            "do_sample": DO_SAMPLE,
            "decoding": DECODING,
            "dtype": DTYPE,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "uses_labels": False,
            "uses_answer_tokens": False,
            "uses_gradients": False,
            "run": dict(run_metadata),
        }
    )
    record["record_sha256"] = canonical_record_sha256(record)
    return record


def validate_resume(
    output: Path,
    samples: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    manifest_sha256: str,
) -> set[str]:
    if not output.is_file() or output.stat().st_size == 0:
        return set()
    samples_by_id = {str(sample["sample_id"]): sample for sample in samples}
    done: set[str] = set()
    for line_number, line in enumerate(
        output.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SpatialEvalExternalContractError(
                f"invalid resume JSON at line {line_number}"
            ) from exc
        sample_id = str(record.get("sample_id"))
        if sample_id in done or sample_id not in samples_by_id:
            raise SpatialEvalExternalContractError(
                f"duplicate or stale resume sample {sample_id!r}"
            )
        sample = samples_by_id[sample_id]
        expected = {
            "arm": arm,
            "mechanism_id": ARM_MECHANISM_IDS[arm],
            "run_mode": RUN_MODE,
            "manifest_sha256": manifest_sha256,
            "image_sha256": sample["image_sha256"],
            "prompt_sha256": prompt_sha256(str(sample["prompt"])),
            "max_new_tokens": MAX_NEW_TOKENS,
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
            "do_sample": DO_SAMPLE,
            "decoding": DECODING,
            "dtype": DTYPE,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "uses_labels": False,
            "uses_answer_tokens": False,
            "uses_gradients": False,
        }
        mismatched = [key for key, value in expected.items() if record.get(key) != value]
        if mismatched:
            raise SpatialEvalExternalContractError(
                f"resume sample {sample_id!r} mismatched fields {mismatched}"
            )
        if record.get("oracle_answer") != sample["oracle_answer"]:
            raise SpatialEvalExternalContractError(
                f"resume oracle mismatch for {sample_id!r}"
            )
        validate_actual_grid(record.get("image_grid_thw", ()))
        if canonical_record_sha256(record) != record.get("record_sha256"):
            raise SpatialEvalExternalContractError(
                f"resume record hash mismatch for {sample_id!r}"
            )
        done.add(sample_id)
    return done


def require_cuda(device: str) -> None:
    import torch

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise SpatialEvalExternalContractError("a CUDA device is required")


def run_warmup(
    arm: str,
    model: Any,
    processor: Any,
    sample: Mapping[str, Any],
    *,
    manifest_sha256: str,
    scorer: Any,
    checkpoint_identity: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    warmup = sanitize_sample_for_selector(sample)
    warmup["max_new_tokens"] = 1
    started = time.perf_counter()
    result = run_one_for_arm(
        arm,
        model,
        processor,
        warmup,
        REFERENCE_MANIFEST,
        manifest_sha256=manifest_sha256,
        scorer=scorer,
        checkpoint_identity=checkpoint_identity,
        run_mode=RUN_MODE,
    )
    torch.cuda.synchronize()
    grid, _ = validate_actual_grid(result["image_grid_thw"])
    return {
        "count": 1,
        "sample_id": str(sample["sample_id"]),
        "max_new_tokens": 1,
        "elapsed_seconds": time.perf_counter() - started,
        "image_grid_thw": grid,
        "output_written": False,
        "cuda_synchronized": True,
    }


def execute_arm(
    *,
    arm: str,
    samples: Sequence[Mapping[str, Any]],
    output: Path,
    model: Any,
    processor: Any,
    scorer: Any,
    checkpoint_identity: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    done = validate_resume(
        output, samples, arm=arm, manifest_sha256=manifest_sha256
    )
    pending = [sample for sample in samples if str(sample["sample_id"]) not in done]
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("a", encoding="utf-8", buffering=1) as handle:
        for index, sample in enumerate(pending, 1):
            selector_sample = sanitize_sample_for_selector(sample)
            frozen = run_one_for_arm(
                arm,
                model,
                processor,
                selector_sample,
                REFERENCE_MANIFEST,
                manifest_sha256=manifest_sha256,
                scorer=scorer,
                checkpoint_identity=checkpoint_identity,
                run_mode=RUN_MODE,
            )
            record = normalize_output_record(
                arm,
                sample,
                frozen,
                manifest_sha256=manifest_sha256,
                run_metadata=run_metadata,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            written += 1
            print(
                f"RESULT {index}/{len(pending)} id={sample['sample_id']} "
                f"arm={arm} G={record['full_llm_visual_tokens']} "
                f"K={record['output_group_count']}",
                flush=True,
            )
    return {
        "arm": arm,
        "total": len(samples),
        "already_done": len(done),
        "written": written,
        "pending_after": len(samples) - len(done) - written,
        "output": str(output),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only and args.prepare_only:
        parser.error("choose only one of --validate-only and --prepare-only")
    if not args.arm and not (args.validate_only or args.prepare_only):
        parser.error("--arm is required for a GPU run")

    verified_pins = verify_preregistered_pins()
    references = load_reference_manifest()
    verify_parquet_shards()
    manifest_sha = sha256_file(REFERENCE_MANIFEST)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "SPATIALEVAL_EXTERNAL_INPUTS_VALID",
                    "reference_records": len(references),
                    "primary_records": PRIMARY_SAMPLE_COUNT,
                    "manifest_sha256": manifest_sha,
                    "verified_pin_count": len(verified_pins),
                    "gpu_runs": 0,
                    "weights_loaded": False,
                },
                indent=2,
            )
        )
        return 0

    samples = materialize_primary_samples(references)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "SPATIALEVAL_EXTERNAL_IMAGES_PREPARED",
                    "sample_count": len(samples),
                    "unique_image_count": len({sample["image_sha256"] for sample in samples}),
                    "gpu_runs": 0,
                    "weights_loaded": False,
                },
                indent=2,
            )
        )
        return 0

    arm = str(args.arm)
    output = ensure_output_path(arm, args.output)
    done = validate_resume(output, samples, arm=arm, manifest_sha256=manifest_sha)
    if len(done) == len(samples):
        print(json.dumps({"status": "ALREADY_COMPLETE", "arm": arm, "count": len(done)}))
        return 0

    require_cuda(args.device)
    model_identity = verify_model_identity(args.model)
    state = build_arm_state(arm)
    model = load_arm_model(args.model, args.device)
    processor = load_arm_processor(args.model)
    warmup = run_warmup(
        arm,
        model,
        processor,
        samples[0],
        manifest_sha256=manifest_sha,
        scorer=state["scorer"],
        checkpoint_identity=state["checkpoint_identity"],
    )
    import torch
    import transformers

    run_metadata = {
        "model_path": str(args.model.resolve()),
        "model_identity_manifest": model_identity,
        "arm": arm,
        "mechanism_id": ARM_MECHANISM_IDS[arm],
        "checkpoint_identity": state["checkpoint_identity"],
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "warmup": warmup,
        "training": False,
    }
    result = execute_arm(
        arm=arm,
        samples=samples,
        output=output,
        model=model,
        processor=processor,
        scorer=state["scorer"],
        checkpoint_identity=state["checkpoint_identity"],
        run_metadata=run_metadata,
        manifest_sha256=manifest_sha,
    )
    result["status"] = "SPATIALEVAL_EXTERNAL_ARM_COMPLETE"
    result["warmup"] = warmup
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
