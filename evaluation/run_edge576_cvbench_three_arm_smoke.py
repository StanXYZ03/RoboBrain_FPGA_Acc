#!/usr/bin/env python3
"""Independent single-sample edge576 GPU smoke entry (one arm, one sample).

This is the last fail-closed layer before the formal
``configs/2026-10-07-edge576-cvbench-three-arm-development-r3.preregistered.json``
execution order.  It runs **exactly one** frozen arm on **exactly one** manifest
sample and writes a single file under

    artifacts/token_selector/2026-10-07-edge576-smoke/<arm>_<sample_id>.json

which holds one complete formal sample record plus a smoke summary.  It never
touches the full2638 arm outputs under
``artifacts/token_selector/2026-10-07-edge576-cvbench/``.

Everything runtime is reused from the already frozen adapter layer; this file
contains **no selection algorithm** and re-implements no forward/scorer step:

* :func:`evaluation.edge576_cvbench_runner_contract.load_edge576_manifest`
* :func:`evaluation.edge576_cvbench_runner_contract.verify_dependency_pins`
* :func:`evaluation.edge576_cvbench_runner_contract.mechanism_identity`
* :func:`evaluation.edge576_cvbench_runner_contract.verify_model_identity`
* :class:`evaluation.edge576_cvbench_runner_contract.Edge576Processor`
* :func:`evaluation.run_edge576_cvbench_three_arm.build_arm_state`
* :func:`evaluation.run_edge576_cvbench_three_arm.load_arm_model`
* :func:`evaluation.run_edge576_cvbench_three_arm.load_arm_processor`
* :func:`evaluation.run_edge576_cvbench_three_arm.require_cuda`
* :func:`evaluation.run_edge576_cvbench_three_arm.run_multimodal_warmup`
* :func:`evaluation.run_edge576_cvbench_three_arm.run_one_for_arm`
* :func:`evaluation.run_edge576_cvbench_three_arm.normalize_record`

The smoke performs, in order and all fail-closed:

1. the 17 frozen dependency SHA256 pins and the arm mechanism identity;
2. the ``--model`` identity verification (relative path + byte size + SHA256,
   including ``model.safetensors``) **before** any CUDA/scorer/weight access;
3. the CUDA requirement and the frozen scorer/model/processor load;
4. exactly one real multimodal warmup (same mechanism, same parameters,
   ``max_new_tokens=1``) whose record is discarded and never written;
5. the real frozen ``run_one`` for the requested sample, with the actual
   processor ``image_grid_thw`` asserted against the edge576 manifest grid;
6. the complete normalised record and the smoke summary written atomically to
   the single smoke output file.

The normalised record always carries ``output_text`` and an auditable
``output_token_count``.  ``output_token_ids`` is optional and is only recorded
when the frozen producer really emitted it (the dense producer does not; the
L2/v2 producers do).  Token ids are never fabricated.

``--dry-run`` runs steps 1-2 plus the sample lookup and the output-path
isolation check without loading any weight or touching CUDA.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.edge576_cvbench_runner_contract import (  # noqa: E402
    ARM_IS_NEW_MECHANISM,
    ARM_MECHANISM_IDS,
    ARM_OUTPUT_PATHS,
    ARMS,
    ATTENTION_IMPLEMENTATION,
    DECODING,
    DO_SAMPLE,
    DTYPE,
    EDGE576_MANIFEST,
    EDGE576_MANIFEST_SHA256,
    K_BUDGET,
    MAX_NEW_TOKENS,
    MAX_PIXELS,
    MIN_PIXELS,
    Edge576ContractError,
    Edge576Processor,
    assert_edge576_grid,
    compact_model_identity,
    load_edge576_manifest,
    mechanism_identity,
    prompt_sha256,
    sanitize_sample_for_selector,
    sha256_file,
    validate_arm,
    verify_dependency_pins,
    verify_model_identity,
)
from evaluation.run_edge576_cvbench_three_arm import (  # noqa: E402
    DEFAULT_MODEL_LABEL,
    DEFAULT_MODEL_PATH,
    build_arm_state,
    build_run_metadata,
    load_arm_model,
    load_arm_processor,
    normalize_record,
    require_cuda,
    run_multimodal_warmup,
    run_one_for_arm,
)

ROOT = Path(__file__).resolve().parents[1]

#: Dedicated smoke output directory.  It is deliberately disjoint from the
#: full2638 ``2026-10-07-edge576-cvbench`` directory.
SMOKE_DIR = (
    ROOT / "artifacts" / "token_selector" / "2026-10-07-edge576-smoke"
)

SMOKE_RUN_MODE = "smoke"
SMOKE_STATUS_PASS = "EDGE576_SMOKE_PASS"
SMOKE_STATUS_DRY_RUN_OK = "EDGE576_SMOKE_DRY_RUN_OK"

#: The smoke sample must exercise real compression.  ``1838`` has G=576 (the
#: edge576 maximum, grid ``[1,36,64]``) and is already the independent main
#: review's processor-wrapper evidence sample; the same sample is used for all
#: three arms so the smoke is directly comparable.
DEFAULT_SMOKE_SAMPLE_ID = "1838"

RunSmokeOne = Callable[[str, Any, Any, dict[str, Any], Path], dict[str, Any]]


# ---------------------------------------------------------------------------
# Smoke output path contract (isolation from every full2638 output)
# ---------------------------------------------------------------------------
def _validate_sample_id_token(sample_id: str) -> str:
    """Return a path-safe sample id or fail closed."""

    token = str(sample_id)
    if (
        not token
        or token in {".", ".."}
        or "/" in token
        or "\\" in token
        or Path(token).name != token
    ):
        raise Edge576ContractError(
            f"sample_id {sample_id!r} is not a safe single path component "
            "(fail closed)"
        )
    return token


def smoke_output_path(
    arm: str, sample_id: str, *, smoke_dir: Path = SMOKE_DIR
) -> Path:
    """The one pinned destination for this ``(arm, sample_id)`` smoke."""

    validate_arm(arm)
    token = _validate_sample_id_token(sample_id)
    return Path(smoke_dir) / f"{arm}_{token}.json"


def ensure_smoke_output_path(
    arm: str,
    sample_id: str,
    output: Path | None = None,
    *,
    smoke_dir: Path = SMOKE_DIR,
) -> Path:
    """Return the pinned smoke path; reject any other or full2638 destination."""

    pinned = smoke_output_path(arm, sample_id, smoke_dir=smoke_dir)
    if output is not None:
        candidate = Path(output)
        try:
            same = candidate.resolve() == pinned.resolve()
        except OSError:
            same = False
        if not same:
            raise Edge576ContractError(
                f"--output {candidate} != pinned smoke path {pinned} for "
                f"({arm}, {sample_id}) (fail closed)"
            )
    resolved = pinned.resolve()
    root = Path(smoke_dir).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise Edge576ContractError(
            f"smoke output {resolved} escapes the smoke directory {root} "
            "(fail closed)"
        ) from exc
    for other_arm, full in ARM_OUTPUT_PATHS.items():
        if resolved == full.resolve():
            raise Edge576ContractError(
                f"smoke output collides with the full2638 output of arm "
                f"{other_arm!r} (fail closed)"
            )
        try:
            full.resolve().relative_to(root)
        except ValueError:
            continue
        raise Edge576ContractError(
            f"full2638 output {full} would live inside the smoke directory "
            f"{root} (fail closed)"
        )
    return pinned


def find_smoke_sample(
    samples: Sequence[Mapping[str, Any]], sample_id: str
) -> Mapping[str, Any]:
    """Return the unique manifest sample for ``sample_id`` or fail closed."""

    token = _validate_sample_id_token(sample_id)
    matches = [
        sample for sample in samples if str(sample["sample_id"]) == token
    ]
    if len(matches) != 1:
        raise Edge576ContractError(
            f"sample_id {token!r} matched {len(matches)} manifest samples; "
            "exactly one is required (fail closed)"
        )
    return matches[0]


def assert_edge576_processor(processor: Any) -> Any:
    """Fail closed unless the processor is the shared Edge576Processor wrapper."""

    if not isinstance(processor, Edge576Processor):
        raise Edge576ContractError(
            "smoke processor is not wrapped in Edge576Processor; the pinned "
            "edge576 pixel budget would not be applied (fail closed)"
        )
    return processor


# ---------------------------------------------------------------------------
# CPU-only validation
# ---------------------------------------------------------------------------
def smoke_validation(
    arm: str,
    samples: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    """17 dependency pins + mechanism identity + smoke output isolation."""

    validate_arm(arm)
    dependencies = verify_dependency_pins()
    identity = mechanism_identity(arm)
    return {
        "arm": arm,
        "mechanism_id": ARM_MECHANISM_IDS[arm],
        "is_new_mechanism": bool(ARM_IS_NEW_MECHANISM[arm]),
        "manifest": str(manifest_path),
        "manifest_sha256": EDGE576_MANIFEST_SHA256,
        "sample_count": len(samples),
        "unique_sample_id_count": len({str(s["sample_id"]) for s in samples}),
        "output_path": str(output),
        "output_is_smoke_path": True,
        "touches_full2638_output": False,
        "dependency_pins": dependencies,
        "mechanism_identity": identity,
        "pins_verified": bool(dependencies["all_match"]),
        "pin_count": int(dependencies["verified_count"]),
        "weights_loaded": False,
        "model_loaded": False,
        "gpu_runs": 0,
    }


def _display_path(path: Path) -> str:
    """Repo-relative display path, falling back to the absolute path."""

    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def planned_command(
    arm: str,
    sample_id: str,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    model_label: str = DEFAULT_MODEL_LABEL,
    manifest_path: Path = EDGE576_MANIFEST,
    output: Path,
) -> str:
    return (
        "python evaluation/run_edge576_cvbench_three_arm_smoke.py"
        f" --arm {arm}"
        f" --sample-id {sample_id}"
        f" --model {model_path}"
        f" --model-label {model_label}"
        f" --manifest {_display_path(manifest_path)}"
        f" --output {_display_path(output)}"
    )


# ---------------------------------------------------------------------------
# Smoke summary + orchestration
# ---------------------------------------------------------------------------
def build_smoke_summary(
    arm: str,
    sample: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    validation: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    warmup: Mapping[str, Any],
    output: Path,
    manifest_path: Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "status": SMOKE_STATUS_PASS,
        "arm": arm,
        "mechanism_id": ARM_MECHANISM_IDS[arm],
        "is_new_mechanism": bool(ARM_IS_NEW_MECHANISM[arm]),
        "run_mode": SMOKE_RUN_MODE,
        "single_sample_only": True,
        "sample_id": str(sample["sample_id"]),
        "image_grid_thw": [int(v) for v in record["image_grid_thw"]],
        "full_llm_visual_tokens": int(record["full_llm_visual_tokens"]),
        "output_group_count": int(record["output_group_count"]),
        "identity_bypass": bool(record["identity_bypass"]),
        "output_token_count": int(record["output_token_count"]),
        "output_token_ids_available": bool(record["output_token_ids_available"]),
        "output_token_ids_recorded": "output_token_ids" in record,
        "parsed_answer": record["parsed_answer"],
        "parse_success": bool(record["parse_success"]),
        "correct": bool(record["correct"]),
        "record_sha256": record["record_sha256"],
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "decoding": DECODING,
        "dtype": DTYPE,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "budget": K_BUDGET,
        "dependency_pin_count": int(validation["dependency_pins"]["verified_count"]),
        "dependency_pins_all_match": bool(validation["dependency_pins"]["all_match"]),
        "mechanism_identity": dict(validation["mechanism_identity"]),
        "model_identity": compact_model_identity(model_identity),
        "model_identity_verified": True,
        "warmup": dict(warmup),
        "warmup_count": 1,
        "warmup_recorded_in_output": False,
        "records_written": 1,
        "output_path": str(output),
        "weights_loaded": True,
        "model_loaded": True,
        "gpu_runs": 1,
        "full2638_output_touched": False,
        "is_accuracy_claim": False,
        "is_deployment_claim": False,
        "is_speed_claim": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def run_smoke(
    *,
    arm: str,
    sample_id: str,
    samples: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    output: Path | None,
    model_path: Path | None,
    model_label: str,
    device: str,
    smoke_dir: Path = SMOKE_DIR,
    validation_fn: Callable[..., dict[str, Any]] = smoke_validation,
    model_identity_fn: Callable[..., dict[str, Any]] = verify_model_identity,
    require_cuda_fn: Callable[[str], None] = require_cuda,
    build_state_fn: Callable[[str], dict[str, Any]] = build_arm_state,
    load_model_fn: Callable[..., Any] = load_arm_model,
    load_processor_fn: Callable[..., Any] = load_arm_processor,
    warmup_fn: Callable[..., dict[str, Any]] = run_multimodal_warmup,
    run_one_fn: RunSmokeOne = run_one_for_arm,
    metadata_fn: Callable[..., dict[str, Any]] = build_run_metadata,
) -> dict[str, Any]:
    """Run exactly one arm on exactly one sample and write one smoke file."""

    validate_arm(arm)
    manifest_path = Path(manifest_path)
    sample = find_smoke_sample(samples, sample_id)
    resolved_output = ensure_smoke_output_path(
        arm, sample_id, output, smoke_dir=smoke_dir
    )
    manifest_sha256 = sha256_file(manifest_path)

    # (1) 17 dependency pins + mechanism identity + output isolation.
    validation = validation_fn(
        arm, samples, manifest_path=manifest_path, output=resolved_output
    )

    if model_path is None:
        raise Edge576ContractError(
            "--model is required to run the smoke (fail closed)"
        )
    # (2) Fail-closed model identity BEFORE CUDA, scorer or weight access.
    model_identity = model_identity_fn(model_path)

    # (3) CUDA requirement, frozen scorer, local model/processor.
    require_cuda_fn(device)
    state = build_state_fn(arm)
    model = load_model_fn(model_path, device)
    processor = assert_edge576_processor(load_processor_fn(model_path))

    # (4) Exactly one real multimodal warmup; its record is discarded.
    warmup = warmup_fn(
        arm,
        model,
        processor,
        dict(sample),
        manifest_path,
        manifest_sha256=manifest_sha256,
        scorer=state["scorer"],
        checkpoint_identity=state["checkpoint_identity"],
        run_mode=SMOKE_RUN_MODE,
        max_new_tokens=1,
    )

    # (5) The real frozen run_one for the requested sample.
    sanitized = sanitize_sample_for_selector(sample)
    frozen_record = run_one_fn(
        arm,
        model,
        processor,
        sanitized,
        manifest_path,
        manifest_sha256=manifest_sha256,
        scorer=state["scorer"],
        checkpoint_identity=state["checkpoint_identity"],
        run_mode=SMOKE_RUN_MODE,
    )
    assert_edge576_grid(sample, frozen_record.get("image_grid_thw"))
    run_metadata = metadata_fn(
        arm,
        model_path=Path(model_path),
        model_label=model_label,
        device=device,
        manifest_sha256=manifest_sha256,
        checkpoint_identity=state["checkpoint_identity"],
        model_identity=model_identity,
        warmup=warmup,
    )
    record = normalize_record(
        arm,
        sample,
        frozen_record,
        manifest_sha256=manifest_sha256,
        run_metadata=run_metadata,
        run_mode=SMOKE_RUN_MODE,
    )

    # (6) One record + one summary -> one file.
    summary = build_smoke_summary(
        arm,
        sample,
        record,
        validation=validation,
        model_identity=model_identity,
        warmup=warmup,
        output=resolved_output,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )
    payload = {
        "status": SMOKE_STATUS_PASS,
        "arm": arm,
        "sample_id": str(sample["sample_id"]),
        "record": record,
        "smoke_summary": summary,
    }
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def dry_run_summary(
    arm: str,
    sample_id: str,
    samples: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    sample = find_smoke_sample(samples, sample_id)
    validation = smoke_validation(
        arm, samples, manifest_path=manifest_path, output=output
    )
    validation["status"] = SMOKE_STATUS_DRY_RUN_OK
    validation["sample_id"] = str(sample["sample_id"])
    validation["image_grid_thw"] = [int(v) for v in sample["image_grid_thw"]]
    validation["full_llm_visual_tokens"] = int(
        sample["full_llm_visual_tokens"]
    )
    validation["planned_command"] = planned_command(
        arm, sample_id, manifest_path=manifest_path, output=output
    )
    return validation


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    parser.add_argument("--manifest", type=Path, default=EDGE576_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--smoke-dir",
        type=Path,
        default=SMOKE_DIR,
        help="advanced/testing override; defaults to the pinned smoke directory",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    samples = load_edge576_manifest(manifest_path)
    output = ensure_smoke_output_path(
        args.arm, args.sample_id, args.output, smoke_dir=args.smoke_dir
    )

    if args.dry_run:
        summary = dry_run_summary(
            args.arm,
            args.sample_id,
            samples,
            manifest_path=manifest_path,
            output=output,
        )
        report = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        if args.json_out is not None:
            json_out = Path(args.json_out)
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(report + "\n", encoding="utf-8")
        print(report, flush=True)
        return 0

    result = run_smoke(
        arm=args.arm,
        sample_id=args.sample_id,
        samples=samples,
        manifest_path=manifest_path,
        output=output,
        model_path=args.model,
        model_label=args.model_label,
        device=args.device,
        smoke_dir=args.smoke_dir,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "arm": result["arm"],
                "sample_id": result["sample_id"],
                "record_sha256": result["record"]["record_sha256"],
                "output_path": result["smoke_summary"]["output_path"],
                "gpu_runs": result["smoke_summary"]["gpu_runs"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Edge576ContractError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
