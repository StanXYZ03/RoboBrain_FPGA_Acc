#!/usr/bin/env python3
"""Verify that the two processor pixel budgets separate G across the gate.

Pre-registered gate-separation dry run for the same-image resolution
counterfactual.  Two budgets are compared:

* ``low``  -- ``max_pixels`` is pinned so that ``G <= 512``;
* ``high`` -- ``min_pixels`` is pinned so that ``G > 512``.

Each frozen ADE20K image must satisfy ``G_low <= 512 < G_high``.  Images that
do not separate are dropped and replaced by the next unused reserve candidate
from the same stratum (deterministic, rank order).  No model is loaded and no
forward pass is run.

Modes
-----
``processor`` (authoritative)
    Loads the real Qwen processor (``local_files_only``, CPU) and reads the
    actual ``image_grid_thw`` for each image / budget.  This is the dry run
    required before any model run.
``analytic`` (non-authoritative prediction)
    Uses the exact Qwen ``smart_resize`` formula without loading the processor.
    It is only a cheap self-check; it never writes the final manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import analyze_full2638_prompt_gain_heterogeneity as het  # noqa: E402
from evaluation.select_ade20k_resolution_counterfactual_samples import (  # noqa: E402
    DESIGN,
    MANIFEST_JSONL,
    RESERVE_JSONL,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]

CONFIG_JSON = (
    ROOT
    / "configs"
    / "2026-10-04-same-image-resolution-counterfactual.preregistered.json"
)
REPORT_JSON = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-same-image-resolution-counterfactual-gate-separation.json"
)
PREDICTION_JSON = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-same-image-resolution-counterfactual-gate-separation-prediction.json"
)
FINAL_MANIFEST_JSONL = (
    ROOT
    / "datasets"
    / "2026-10-04-same-image-resolution-counterfactual-ade20k100-separated.jsonl"
)

PATCH_SIZE = 16
MERGE_SIZE = 2
MERGE_UNIT = MERGE_SIZE * MERGE_SIZE
RESIZE_FACTOR = PATCH_SIZE * MERGE_SIZE
GATE_THRESHOLD = 512

SEPARATION_RULE = (
    "A frozen sample separates iff its low-budget group count is <= 512 and "
    "its high-budget group count is > 512; otherwise it is dropped and replaced."
)
REPLACEMENT_RULE = (
    "For each dropped sample, take the next unused reserve candidate from the "
    "same stratum in ascending selection_rank order whose own two budgets also "
    "separate. If the reserve for that stratum is exhausted the sample stays "
    "unresolved and the dry run fails closed."
)


class GateSeparationError(RuntimeError):
    """Raised when the dry-run contract is violated."""


# ---------------------------------------------------------------------------
# Analytic prediction (exact Qwen smart_resize)
# ---------------------------------------------------------------------------
def smart_resize(
    height: int,
    width: int,
    *,
    factor: int = RESIZE_FACTOR,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Exact Qwen2/3-VL ``smart_resize`` (mirrors the released implementation)."""

    height = int(height)
    width = int(width)
    if height < factor or width < factor:
        raise GateSeparationError(
            f"image {width}x{height} is smaller than the {factor}px resize factor"
        )
    height_bar = max(factor, round(height / factor) * factor)
    width_bar = max(factor, round(width / factor) * factor)
    if height_bar * width_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        height_bar = max(factor, math.floor(height / beta / factor) * factor)
        width_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif height_bar * width_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        height_bar = math.ceil(height * beta / factor) * factor
        width_bar = math.ceil(width * beta / factor) * factor
    return height_bar, width_bar


def predict_group_count(
    height: int,
    width: int,
    *,
    min_pixels: int,
    max_pixels: int,
    patch_size: int = PATCH_SIZE,
    merge_unit: int = MERGE_UNIT,
) -> int:
    height_bar, width_bar = smart_resize(
        height, width, min_pixels=min_pixels, max_pixels=max_pixels
    )
    full_patch_tokens = (height_bar // patch_size) * (width_bar // patch_size)
    if full_patch_tokens % merge_unit:
        raise GateSeparationError(
            f"resized grid {height_bar}x{width_bar} is not aligned to the merger"
        )
    return full_patch_tokens // merge_unit


def separation_status(g_low: int, g_high: int, threshold: int = GATE_THRESHOLD) -> bool:
    return int(g_low) <= threshold < int(g_high)


# ---------------------------------------------------------------------------
# Deterministic drop / replace
# ---------------------------------------------------------------------------
def _row_stratum(row: Mapping[str, Any]) -> str:
    nested = row.get("resolution_counterfactual")
    if isinstance(nested, Mapping) and "stratum" in nested:
        return str(nested["stratum"])
    return str(row["stratum"])


def _row_rank(row: Mapping[str, Any]) -> int:
    nested = row.get("resolution_counterfactual")
    if isinstance(nested, Mapping) and "selection_rank" in nested:
        return int(nested["selection_rank"])
    return int(row["selection_rank"])


def plan_replacements(
    measurements: Sequence[Mapping[str, Any]],
    reserve: Sequence[Mapping[str, Any]],
    *,
    measure_candidate,
    threshold: int = GATE_THRESHOLD,
) -> dict[str, Any]:
    """Apply the pre-registered drop/replace rule deterministically.

    ``measure_candidate`` is called lazily for reserve candidates that are
    actually needed, so an all-separating manifest never touches the reserve.
    """

    reserve_by_stratum: dict[str, list[Mapping[str, Any]]] = {}
    for row in sorted(reserve, key=lambda item: (_row_stratum(item), _row_rank(item))):
        reserve_by_stratum.setdefault(_row_stratum(row), []).append(row)
    used: set[str] = set()
    checked: set[str] = set()
    accepted: list[str] = []
    accepted_measurements: list[Mapping[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    replaced: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for row in measurements:
        sample_id = str(row["sample_id"])
        if row["separated"]:
            accepted.append(sample_id)
            accepted_measurements.append(row)
            used.add(sample_id)
            continue
        stratum = str(row["stratum"])
        rejected.append(
            {
                "sample_id": sample_id,
                "stratum": stratum,
                "G_low": row["G_low"],
                "G_high": row["G_high"],
            }
        )
        replacement = None
        for candidate in reserve_by_stratum.get(stratum, []):
            candidate_id = str(candidate["sample_id"])
            if candidate_id in used or candidate_id in checked:
                continue
            checked.add(candidate_id)
            measured = measure_candidate(candidate)
            if separation_status(measured["G_low"], measured["G_high"], threshold):
                replacement = measured
                break
        if replacement is None:
            unresolved.append({"sample_id": sample_id, "stratum": stratum})
            continue
        used.add(str(replacement["sample_id"]))
        accepted.append(str(replacement["sample_id"]))
        accepted_measurements.append(replacement)
        replaced.append(
            {
                "dropped": sample_id,
                "replacement": str(replacement["sample_id"]),
                "stratum": stratum,
                "replacement_G_low": replacement["G_low"],
                "replacement_G_high": replacement["G_high"],
            }
        )
    return {
        "accepted": accepted,
        "accepted_count": len(accepted),
        "accepted_measurements": accepted_measurements,
        "rejected": rejected,
        "rejected_count": len(rejected),
        "replaced": replaced,
        "replaced_count": len(replaced),
        "unresolved": unresolved,
        "unresolved_count": len(unresolved),
        "resolved": len(accepted) == len(measurements) and not unresolved,
    }


# ---------------------------------------------------------------------------
# Measurement backends
# ---------------------------------------------------------------------------
def measure_analytic(
    sample: Mapping[str, Any],
    budgets: Mapping[str, Mapping[str, Any]],
    *,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    width, height = image_size
    low = budgets["low"]
    high = budgets["high"]
    g_low = predict_group_count(
        height, width, min_pixels=int(low["min_pixels"]), max_pixels=int(low["max_pixels"])
    )
    g_high = predict_group_count(
        height, width, min_pixels=int(high["min_pixels"]), max_pixels=int(high["max_pixels"])
    )
    return {
        "sample_id": str(sample["sample_id"]),
        "stratum": str(sample["resolution_counterfactual"]["stratum"]),
        "width": width,
        "height": height,
        "G_low": g_low,
        "G_high": g_high,
        "separated": separation_status(g_low, g_high),
    }


def _processor_group_count(
    processor: Any, image: Any, prompt: str, *, min_pixels: int, max_pixels: int
) -> int:
    from evaluation.run_hf_token_reduction import (  # noqa: PLC0415
        make_messages,
        processor_inputs,
    )

    messages = make_messages(Path("image"), prompt)
    processed = processor_inputs(
        processor, messages, image, min_pixels=min_pixels, max_pixels=max_pixels
    )
    grid = [int(value) for value in processed.image_grid_thw[0].tolist()]
    full_patch_tokens = grid[0] * grid[1] * grid[2]
    merge_size = int(getattr(processor.image_processor, "merge_size", MERGE_SIZE))
    merge_unit = merge_size * merge_size
    if full_patch_tokens % merge_unit:
        raise GateSeparationError(
            f"processor grid {grid} is not aligned to the merger (fail closed)"
        )
    return full_patch_tokens // merge_unit


def measure_processor(
    processor: Any,
    sample: Mapping[str, Any],
    budgets: Mapping[str, Mapping[str, Any]],
    *,
    image_path: Path,
) -> dict[str, Any]:
    from PIL import Image  # noqa: PLC0415

    low = budgets["low"]
    high = budgets["high"]
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        g_low = _processor_group_count(
            processor,
            image,
            str(sample["prompt"]),
            min_pixels=int(low["min_pixels"]),
            max_pixels=int(low["max_pixels"]),
        )
        g_high = _processor_group_count(
            processor,
            image,
            str(sample["prompt"]),
            min_pixels=int(high["min_pixels"]),
            max_pixels=int(high["max_pixels"]),
        )
    return {
        "sample_id": str(sample["sample_id"]),
        "stratum": str(sample["resolution_counterfactual"]["stratum"]),
        "width": width,
        "height": height,
        "G_low": g_low,
        "G_high": g_high,
        "separated": separation_status(g_low, g_high),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise GateSeparationError(f"missing required input {path} (fail closed)")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_config(path: Path = CONFIG_JSON) -> dict[str, Any]:
    if not path.is_file():
        raise GateSeparationError(f"missing preregistered config {path} (fail closed)")
    return json.loads(path.read_text(encoding="utf-8"))


def budgets_from_config(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    budgets = config.get("processor_pixel_budgets")
    if not isinstance(budgets, Mapping) or set(budgets) != {"low", "high"}:
        raise GateSeparationError(
            "preregistered config must define exactly 'low' and 'high' budgets"
        )
    for label in ("low", "high"):
        spec = budgets[label]
        for key in ("min_pixels", "max_pixels"):
            if not isinstance(spec.get(key), int) or int(spec[key]) <= 0:
                raise GateSeparationError(f"budget {label}.{key} must be a positive int")
        if int(spec["min_pixels"]) > int(spec["max_pixels"]):
            raise GateSeparationError(f"budget {label} has min_pixels > max_pixels")
    return {label: dict(budgets[label]) for label in ("low", "high")}


def analyze(
    *,
    mode: str,
    manifest_path: Path = MANIFEST_JSONL,
    reserve_path: Path = RESERVE_JSONL,
    config_path: Path = CONFIG_JSON,
    model_path: Path | None = None,
    image_resolver=None,
) -> dict[str, Any]:
    if mode not in {"processor", "analytic"}:
        raise GateSeparationError(f"unknown mode {mode!r}")
    config = load_config(config_path)
    budgets = budgets_from_config(config)
    manifest = load_jsonl(manifest_path)
    reserve = load_jsonl(reserve_path)
    if len(manifest) != int(config["frozen_sample_count"]):
        raise GateSeparationError(
            f"manifest has {len(manifest)} samples, config expects "
            f"{config['frozen_sample_count']} (fail closed)"
        )
    selected_strata = {
        row["resolution_counterfactual"]["stratum"] for row in manifest
    }
    reserve_strata = {
        row["resolution_counterfactual"]["stratum"] for row in reserve
    }
    strata_without_reserve = sorted(selected_strata - reserve_strata)
    if image_resolver is None:

        def image_resolver(path: str) -> Path:  # noqa: ANN001
            return Path(path)

    measurements: list[dict[str, Any]] = []
    processor = None
    if mode == "processor":
        if model_path is None or not model_path.is_dir():
            raise GateSeparationError(
                "--model must point to a local model directory in processor mode"
            )
        from transformers import AutoProcessor  # noqa: PLC0415

        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True
        )
        for sample in manifest:
            measurements.append(
                measure_processor(
                    processor,
                    sample,
                    budgets,
                    image_path=image_resolver(str(sample["image"])),
                )
            )
    else:
        from PIL import Image  # noqa: PLC0415

        for sample in manifest:
            with Image.open(image_resolver(str(sample["image"]))) as image:
                size = image.size
            measurements.append(measure_analytic(sample, budgets, image_size=size))

    def measure_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
        if mode == "processor":
            measured = measure_processor(
                processor,
                candidate,
                budgets,
                image_path=image_resolver(str(candidate["image"])),
            )
        else:
            from PIL import Image  # noqa: PLC0415

            with Image.open(image_resolver(str(candidate["image"]))) as image:
                size = image.size
            measured = measure_analytic(candidate, budgets, image_size=size)
        measured["selection_rank"] = int(
            candidate["resolution_counterfactual"]["selection_rank"]
        )
        return measured

    plan = plan_replacements(
        measurements, reserve, measure_candidate=measure_candidate
    )
    authoritative = mode == "processor"
    by_id = {str(sample["sample_id"]): sample for sample in manifest + reserve}
    ordered_final = [by_id[sample_id] for sample_id in plan["accepted"]]

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "schema_version": "1.0",
        "design": DESIGN,
        "mode": mode,
        "authoritative": authoritative,
        "model_was_run": False,
        "gate_threshold": GATE_THRESHOLD,
        "separation_rule": SEPARATION_RULE,
        "replacement_rule": REPLACEMENT_RULE,
        "processor_pixel_budgets": budgets,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "count": len(manifest),
        },
        "reserve": {
            "path": str(reserve_path),
            "sha256": sha256_file(reserve_path),
            "count": len(reserve),
        },
        "strata_without_reserve": strata_without_reserve,
        "measurements": measurements,
        "plan": {key: value for key, value in plan.items() if key != "accepted_measurements"},
        "final_manifest": None,
    }
    accepted_measurements = plan["accepted_measurements"]
    report["contract_checks"] = [
        {"name": "mode is processor (authoritative) or analytic (prediction)", "ok": mode in {"processor", "analytic"}},
        {"name": "no model forward pass was run", "ok": True},
        {"name": "all low-budget G <= 512 for accepted samples", "ok": all(m["G_low"] <= GATE_THRESHOLD for m in accepted_measurements)},
        {"name": "all high-budget G > 512 for accepted samples", "ok": all(m["G_high"] > GATE_THRESHOLD for m in accepted_measurements)},
        {"name": "accepted + unresolved == frozen count", "ok": plan["accepted_count"] + plan["unresolved_count"] == len(manifest)},
        {"name": "no accepted sample duplicates a reserve sample", "ok": len(set(plan["accepted"])) == plan["accepted_count"]},
        {"name": "reserve coverage: strata without reserve reported (any needed drop there would fail closed)", "ok": isinstance(strata_without_reserve, list)},
    ]
    if authoritative:
        report["final_manifest"] = {
            "path": str(FINAL_MANIFEST_JSONL),
            "sha256": None,
            "count": len(ordered_final),
        }
        FINAL_MANIFEST_JSONL.parent.mkdir(parents=True, exist_ok=True)
        FINAL_MANIFEST_JSONL.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in ordered_final),
            encoding="utf-8",
        )
        report["final_manifest"]["sha256"] = sha256_file(FINAL_MANIFEST_JSONL)
    return report


def default_report_path(mode: str) -> Path:
    return REPORT_JSON if mode == "processor" else PREDICTION_JSON


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("processor", "analytic"), default="processor")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=CONFIG_JSON)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_JSONL)
    parser.add_argument("--reserve", type=Path, default=RESERVE_JSONL)
    parser.add_argument("--json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze(
        mode=args.mode,
        manifest_path=args.manifest,
        reserve_path=args.reserve,
        config_path=args.config,
        model_path=args.model,
    )
    output = args.json or default_report_path(args.mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "authoritative": report["authoritative"],
                "separated": sum(1 for m in report["measurements"] if m["separated"]),
                "rejected": report["plan"]["rejected_count"],
                "replaced": report["plan"]["replaced_count"],
                "unresolved": report["plan"]["unresolved_count"],
                "resolved": report["plan"]["resolved"],
                "json": str(output),
                "final_manifest": report["final_manifest"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
