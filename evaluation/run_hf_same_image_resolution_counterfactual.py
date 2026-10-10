#!/usr/bin/env python3
"""Scaffold runner for the same-image dual-resolution counterfactual.

.. warning::
   This runner exists to make the pre-registered counterfactual executable and
   auditable.  It has **not been executed**: no model was loaded, no GPU was
   used and no arm output exists.  Before any run, the authoritative
   processor-only dry run must resolve all 100 frozen samples, and this runner
   refuses to start unless that clearance report exists and is consistent with
   the ``--manifest`` it is given.

The runner delegates the frozen B1 forward, checkpoint loading, resume contract
and greedy decoding to :mod:`evaluation.run_hf_hardware_regular_b1_corrected_l2`
and only injects a processor pixel budget (``min_pixels`` / ``max_pixels``) into
the otherwise frozen processor call.  Every written record is tagged with the
pixel budget label and bounds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

DESIGN = "2026-10-04-same-image-resolution-counterfactual"
SEPARATION_REPORT_JSON = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-same-image-resolution-counterfactual-gate-separation.json"
)

DEFAULT_BUDGET = 96
DEFAULT_REGION_GRID = 3
DEFAULT_LOCAL_CANDIDATE_CAP = 16

PIXEL_BUDGETS: dict[str, dict[str, int]] = {
    "low": {"min_pixels": 65536, "max_pixels": 524288},
    "high": {"min_pixels": 525312, "max_pixels": 16777216},
}


class CounterfactualRunnerError(RuntimeError):
    """Raised when the runner contract is violated (fail closed)."""


def pixel_budget_metadata(label: str, min_pixels: int, max_pixels: int) -> dict[str, Any]:
    if label not in PIXEL_BUDGETS:
        raise CounterfactualRunnerError(
            f"pixel budget label must be one of {sorted(PIXEL_BUDGETS)}, got {label!r}"
        )
    return {
        "design": DESIGN,
        "label": label,
        "min_pixels": int(min_pixels),
        "max_pixels": int(max_pixels),
    }


def resolve_pixel_budget(
    label: str, min_pixels: int | None, max_pixels: int | None
) -> tuple[int, int]:
    if label not in PIXEL_BUDGETS:
        raise CounterfactualRunnerError(
            f"unknown pixel budget label {label!r}; expected one of {sorted(PIXEL_BUDGETS)}"
        )
    default = PIXEL_BUDGETS[label]
    if (min_pixels is None) != (max_pixels is None):
        raise CounterfactualRunnerError(
            "--min-pixels and --max-pixels must be supplied together"
        )
    if min_pixels is None:
        min_pixels = default["min_pixels"]
        max_pixels = default["max_pixels"]
    assert min_pixels is not None and max_pixels is not None
    if min_pixels <= 0 or max_pixels <= 0 or min_pixels > max_pixels:
        raise CounterfactualRunnerError(
            f"invalid pixel budget: min_pixels={min_pixels}, max_pixels={max_pixels}"
        )
    return int(min_pixels), int(max_pixels)


def load_gate_clearance(
    report_path: Path = SEPARATION_REPORT_JSON,
) -> dict[str, Any]:
    """Validate the authoritative processor dry run and return its manifest pin."""

    if not report_path.is_file():
        raise CounterfactualRunnerError(
            f"missing authoritative processor dry-run report {report_path}; run "
            "'python evaluation/verify_same_image_resolution_gate_separation.py "
            "--mode processor --model <local-model>' first (fail closed)"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("mode") != "processor" or not report.get("authoritative"):
        raise CounterfactualRunnerError(
            "the gate-separation report is not the authoritative processor dry run "
            "(fail closed)"
        )
    plan = report.get("plan") or {}
    if not plan.get("resolved"):
        raise CounterfactualRunnerError(
            "the processor dry run did not resolve all 100 samples (fail closed)"
        )
    final = report.get("final_manifest")
    if not isinstance(final, Mapping) or not final.get("sha256"):
        raise CounterfactualRunnerError(
            "the processor dry run did not pin a final separated manifest (fail closed)"
        )
    return {
        "report_path": str(report_path),
        "final_manifest_path": str(final["path"]),
        "final_manifest_sha256": str(final["sha256"]),
        "gate_threshold": int(report.get("gate_threshold", 512)),
    }


def validate_pixel_budget_resume(
    output: Path, *, label: str, min_pixels: int, max_pixels: int
) -> None:
    """Refuse to resume an output file written under a different pixel budget."""

    if not output.is_file() or output.stat().st_size == 0:
        return
    for line_number, line in enumerate(
        output.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        for field, expected in (
            ("pixel_budget_label", label),
            ("min_pixels", int(min_pixels)),
            ("max_pixels", int(max_pixels)),
        ):
            if record.get(field) != expected:
                raise CounterfactualRunnerError(
                    f"output line {line_number} {field}={record.get(field)!r} != "
                    f"current run {expected!r} (mixed pixel budget, fail closed)"
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--pixel-budget", choices=sorted(PIXEL_BUDGETS), required=True)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--region-grid", type=int, default=DEFAULT_REGION_GRID)
    parser.add_argument(
        "--local-candidate-cap", type=int, default=DEFAULT_LOCAL_CANDIDATE_CAP
    )
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--separation-report", type=Path, default=SEPARATION_REPORT_JSON)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    min_pixels, max_pixels = resolve_pixel_budget(
        args.pixel_budget, args.min_pixels, args.max_pixels
    )
    clearance = load_gate_clearance(args.separation_report)

    import torch  # noqa: PLC0415

    import evaluation.run_hf_hardware_regular_b1_corrected_l2 as base  # noqa: PLC0415
    from evaluation.run_hf_token_reduction import (  # noqa: PLC0415
        processor_inputs as real_processor_inputs,
    )

    if not args.model.is_dir():
        raise CounterfactualRunnerError(
            f"--model must be a local directory (strict offline): {args.model}"
        )
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise CounterfactualRunnerError("this experiment requires a CUDA device")
    if args.arm not in base.ALLOWED_ARMS:
        raise CounterfactualRunnerError(
            f"--arm must be one of {base.ALLOWED_ARMS}, got {args.arm!r}"
        )

    manifest_sha256 = base.file_sha256(args.manifest)
    if manifest_sha256 != clearance["final_manifest_sha256"]:
        raise CounterfactualRunnerError(
            "the --manifest is not the final separated manifest pinned by the "
            "processor dry run (fail closed)"
        )

    Wt, wv_q, checkpoint_identity = base.load_arm_checkpoint(args.arm)
    scorer = base.build_b1_scorer(args.arm, Wt=Wt, wv_q=wv_q, budgets=(args.budget,))
    samples = base.limit_each_task(
        [
            json.loads(line)
            for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
        args.limit_per_task,
    )
    done = base.validate_resume(
        args.output,
        samples,
        arm=args.arm,
        manifest_sha256=manifest_sha256,
        checkpoint_identity=checkpoint_identity,
        budget=args.budget,
        region_grid=args.region_grid,
        local_candidate_cap=args.local_candidate_cap,
    )
    validate_pixel_budget_resume(
        args.output,
        label=args.pixel_budget,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    pending = [sample for sample in samples if sample["sample_id"] not in done]
    if not pending:
        return 0

    dtype = {"bf16": torch.bfloat16}[args.dtype]
    torch.manual_seed(0)
    from transformers import (  # noqa: PLC0415
        AutoModelForImageTextToText,
        AutoProcessor,
    )

    # Inject the pixel budget into the frozen processor call used by run_one.
    base.processor_inputs = lambda processor, messages, image: real_processor_inputs(  # noqa: E731
        processor, messages, image, min_pixels=min_pixels, max_pixels=max_pixels
    )
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model),
        dtype=dtype,
        device_map=args.device,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    metadata = {
        "model_label": args.model_label,
        "model_path": str(args.model.resolve()),
        "dtype": args.dtype,
        "device": args.device,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "arm": args.arm,
        "integration_score_mode": base.SCORE_MODE_INTEGER,
        "prune_after_layer": 0,
        "budget": args.budget,
        "region_grid": args.region_grid,
        "local_candidate_cap": args.local_candidate_cap,
        "manifest_sha256": manifest_sha256,
        "do_sample": False,
        "custom_greedy_decode": True,
        "original_position_ids_preserved": True,
        "training": False,
        "same_image_resolution_counterfactual": True,
        "pixel_budget": pixel_budget_metadata(
            args.pixel_budget, min_pixels, max_pixels
        ),
        "cpu_selector_roundtrip_is_acceleration_claim": False,
        **checkpoint_identity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8", buffering=1) as stream:
        for sample in pending:
            record = base.run_one(
                model,
                processor,
                sample,
                args.manifest,
                arm=args.arm,
                budget=args.budget,
                region_grid=args.region_grid,
                local_candidate_cap=args.local_candidate_cap,
                scorer=scorer,
                manifest_sha256=manifest_sha256,
                checkpoint_identity=checkpoint_identity,
            )
            record["pixel_budget_label"] = args.pixel_budget
            record["min_pixels"] = int(min_pixels)
            record["max_pixels"] = int(max_pixels)
            record["run"] = metadata
            record["record_sha256"] = base.record_sha256(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
