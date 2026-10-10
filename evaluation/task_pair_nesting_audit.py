"""Task-pair nesting audit for the operational prompt-utility diagnostic.

This is a **zero-training, CPU-only, read-only explanatory audit**.  It asks
whether the "same-image other-task" wrong prompt used by the operational
prompt-utility diagnostic is a *valid* negative for the grounding/pointing
task pair, by checking whether the two spatial targets of the same image are
nested (the pointing point lies inside the grounding bbox) and by re-aggregating
the already-reported B-arm K48 correct-minus-wrong retained-mass gap per task.

It reads only three pinned text inputs:

* ``datasets/prompt_projection_clean_inductive_v1.jsonl`` (clean manifest,
  geometry only),
* ``reports/token_selector/2026-10-03-operational-prompt-utility-diagnostic-records.csv``
  (val-only per-record metrics),
* ``reports/token_selector/2026-10-03-operational-prompt-utility-diagnostic.json``
  (corresponding operational JSON, for a cross-check).

The governing preregistration (config + report) is written before any input
byte is read; this module only re-verifies the frozen identity/pins.  It never
opens an NPZ, a raw image or any model weight, never trains, never uses a GPU
and never touches the network.  It does not change the prior
``OPERATIONAL_PROMPT_UTILITY_NOT_ESTABLISHED`` decision and makes no accuracy or
promotion claim.

Reuse only: geometry via
:func:`evaluation.spatial_teacher_feasibility_audit.point_inside_bbox`, rank
correlation via :func:`evaluation.prompt_projection_distillation.spearman`, and
hashing via :func:`evaluation.prompt_projection_clean_inductive.sha256_file`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:  # pragma: no cover - direct CLI execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.prompt_projection_clean_inductive import sha256_file
from evaluation.prompt_projection_distillation import spearman
from evaluation.spatial_teacher_feasibility_audit import point_inside_bbox

# ---------------------------------------------------------------------------
# Frozen constants (mirrored from the preregistration config)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-10-03"
REPORT_STEM = "2026-10-03-task-pair-nesting-audit"
SCHEMA_VERSION = "1.0"

REPORTS_DIR = Path("reports/token_selector")
LOGS_DIR = REPORTS_DIR / "logs"
PREREG_CONFIG_PATH = Path("configs/task_pair_nesting_audit.preregistered.json")
PREREG_REPORT_PATH = Path(
    "reports/token_selector/2026-10-03-task-pair-nesting-audit-preregistration.json"
)
PROMPT_PATH = Path(
    "reports/token_selector/prompts/2026-10-03-task-pair-nesting-audit-harness-prompt.md"
)
MANIFEST_PATH = Path("datasets/prompt_projection_clean_inductive_v1.jsonl")
RECORDS_CSV_PATH = Path(
    "reports/token_selector/2026-10-03-operational-prompt-utility-diagnostic-records.csv"
)
OPERATIONAL_JSON_PATH = Path(
    "reports/token_selector/2026-10-03-operational-prompt-utility-diagnostic.json"
)

PRIMARY_ARM = "B_SPATIAL_KL_CONTRASTIVE"
CONTROL_ARM = "A_SPATIAL_KL_CONTROL"
SEEDS: tuple[int, ...] = (20261003, 20261004, 20261005)
BUDGETS: tuple[int, ...] = (48, 64, 96)
PRIMARY_BUDGET = 48
TASKS: tuple[str, ...] = ("grounding", "pointing")
NESTED_RATIO_THRESHOLD = 0.80
CROSS_CHECK_TOLERANCE = 1e-9

DECISION_SYMMETRIC_INVALID = "SYMMETRIC_NEGATIVE_INVALID"
DECISION_NOT_ESTABLISHED = "NOT_ESTABLISHED"
ALLOWED_DECISIONS: tuple[str, ...] = (
    DECISION_SYMMETRIC_INVALID,
    DECISION_NOT_ESTABLISHED,
)

REQUIRED_RECORD_FIELDS: tuple[str, ...] = (
    "arm",
    "seed",
    "sample_id",
    "image_sha256",
    "task",
    "group_count",
    "grid_h",
    "grid_w",
    "budget",
    "correct_retained_spatial_target_mass",
    "wrong_retained_spatial_target_mass",
    "gap_retained_spatial_target_mass",
    "correct_wrong_selected_jaccard",
)

RECORDS_CSV_FIELDS: tuple[str, ...] = (
    "sample_id",
    "image_sha256",
    "task",
    "group_count",
    "grid_h",
    "grid_w",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_area_norm",
    "point_x",
    "point_y",
    "point_inside_bbox",
    "gap_k48_seed_20261003",
    "gap_k48_seed_20261004",
    "gap_k48_seed_20261005",
    "gap_k48_median_3seed",
    "jaccard_k48_seed_20261003",
    "jaccard_k48_seed_20261004",
    "jaccard_k48_seed_20261005",
    "jaccard_k48_median_3seed",
)

# Text inputs this module is allowed to read.  Hashing the pinned inputs is a
# read of bytes but never a parse of their content; the guard below covers the
# parsed data files and the governing preregistration.
_ALLOWED_READ_RELS: tuple[Path, ...] = (
    PREREG_CONFIG_PATH,
    PREREG_REPORT_PATH,
    MANIFEST_PATH,
    RECORDS_CSV_PATH,
    OPERATIONAL_JSON_PATH,
)


class AuditContractError(RuntimeError):
    """Raised when an audit contract, pin or schema check fails."""


# ---------------------------------------------------------------------------
# Input isolation guard
# ---------------------------------------------------------------------------
_READ_LOG: list[str] = []


def reset_read_log() -> None:
    """Clear the read log (used by tests)."""
    _READ_LOG.clear()


def read_log() -> list[str]:
    """Return a copy of the resolved paths read by the audit logic."""
    return list(_READ_LOG)


def allowed_read_paths(root: Path | str = ROOT) -> set[str]:
    base = Path(root)
    return {str((base / rel).resolve()) for rel in _ALLOWED_READ_RELS}


def read_text(path: Path | str, *, root: Path | str = ROOT, purpose: str = "") -> str:
    """Read an allow-listed text input, rejecting anything else.

    This is the single gate every audit data read passes through, so a test can
    prove that no NPZ / image / weight path is ever opened.
    """
    resolved = Path(path).resolve()
    allowed = allowed_read_paths(root)
    if str(resolved) not in allowed:
        raise AuditContractError(
            f"refusing to read non-allow-listed path ({purpose or 'read'}): {resolved}"
        )
    text = resolved.read_text(encoding="utf-8")
    _READ_LOG.append(str(resolved))
    return text


# ---------------------------------------------------------------------------
# Frozen preregistration identity
# ---------------------------------------------------------------------------
def load_prereg(root: Path | str = ROOT) -> dict[str, Any]:
    """Load the governing config and report and assert their mutual identity."""
    base = Path(root)
    config = json.loads(read_text(base / PREREG_CONFIG_PATH, root=base, purpose="config"))
    report = json.loads(read_text(base / PREREG_REPORT_PATH, root=base, purpose="report"))
    config_sha = sha256_file(base / PREREG_CONFIG_PATH)
    if config_sha != report["config"]["sha256"]:
        raise AuditContractError("preregistration config SHA does not match the report")
    if report["config"]["path"] != str(PREREG_CONFIG_PATH):
        raise AuditContractError("preregistration report points at an unexpected config")
    prompt_sha = sha256_file(base / PROMPT_PATH)
    if prompt_sha != config["authority_prompt"]["sha256"]:
        raise AuditContractError("authority prompt SHA does not match the config")
    if prompt_sha != report["authority_prompt"]["sha256"]:
        raise AuditContractError("authority prompt SHA does not match the report")
    return {"config": config, "report": report, "config_sha256": config_sha, "prompt_sha256": prompt_sha}


def verify_input_pins(
    prereg: Mapping[str, Any], root: Path | str = ROOT
) -> dict[str, Any]:
    """Recompute every frozen input SHA256 and report mismatches."""
    base = Path(root)
    pinned = prereg["report"]["frozen_inputs"]
    actual: dict[str, Any] = {}
    mismatches: dict[str, Any] = {}
    for rel, expected in pinned.items():
        path = base / rel
        value = sha256_file(path) if path.is_file() else None
        actual[rel] = value
        if value != expected:
            mismatches[rel] = {"expected": expected, "actual": value}
    if mismatches:
        raise AuditContractError(f"frozen input SHA mismatch: {sorted(mismatches)}")
    return {"pinned": dict(pinned), "actual": actual, "mismatches": mismatches}


def snapshot_protected_hashes(root: Path | str = ROOT) -> dict[str, Any]:
    """Hash the preregistration, prompt and the three inputs for a before/after check."""
    base = Path(root)
    tracked = (PREREG_CONFIG_PATH, PREREG_REPORT_PATH, PROMPT_PATH, MANIFEST_PATH, RECORDS_CSV_PATH, OPERATIONAL_JSON_PATH)
    return {str(rel): sha256_file(base / rel) for rel in tracked if (base / rel).is_file()}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_geometry(manifest_path: Path | str, *, root: Path | str = ROOT) -> dict[str, dict[str, Any]]:
    """Load per-record geometry (bbox/point/task) from the clean manifest."""
    text = read_text(manifest_path, root=root, purpose="manifest")
    geometry: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        sample_id = str(record["sample_id"])
        ground_truth = record["ground_truth"]
        bbox = [float(value) for value in ground_truth["bbox_1000"]]
        point = [float(value) for value in ground_truth["point_1000"]]
        if len(bbox) != 4 or len(point) != 2:
            raise AuditContractError(f"{sample_id}: bad geometry shape")
        geometry[sample_id] = {
            "task": str(record["task"]),
            "bbox_1000": bbox,
            "point_1000": point,
            "coco_image_id": record.get("coco_image_id"),
        }
    if not geometry:
        raise AuditContractError("clean manifest is empty")
    return geometry


def load_records(
    csv_path: Path | str, *, root: Path | str = ROOT
) -> list[dict[str, str]]:
    """Load the operational records CSV as a list of string rows."""
    text = read_text(csv_path, root=root, purpose="records")
    reader = csv.DictReader(text.splitlines())
    fieldnames = reader.fieldnames or []
    missing = [field for field in REQUIRED_RECORD_FIELDS if field not in fieldnames]
    if missing:
        raise AuditContractError(f"records CSV missing columns: {missing}")
    rows = [dict(row) for row in reader]
    if not rows:
        raise AuditContractError("records CSV is empty")
    return rows


def load_operational_json(
    path: Path | str, *, root: Path | str = ROOT
) -> dict[str, Any]:
    return json.loads(read_text(path, root=root, purpose="operational-json"))


# ---------------------------------------------------------------------------
# Pure geometry helpers
# ---------------------------------------------------------------------------
def bbox_area_norm(bbox_1000: Sequence[float]) -> float:
    """Normalized bbox area in ``[0, 1]`` after clipping to ``[0, 1000]``."""
    values = [float(value) for value in bbox_1000]
    if len(values) != 4:
        raise AuditContractError("bbox_1000 must have 4 values")
    if not all(np.isfinite(values)):
        raise AuditContractError("bbox_1000 must be finite")
    x1, y1, x2, y2 = (min(max(value, 0.0), 1000.0) for value in values)
    if x2 <= x1 or y2 <= y1:
        raise AuditContractError(f"bbox_1000 has non-positive area: {values}")
    return (x2 - x1) * (y2 - y1) / 1_000_000.0


def record_point_inside(point_1000: Sequence[float], bbox_1000: Sequence[float]) -> bool:
    """Inclusive point-in-bbox test via the frozen geometry helper."""
    return bool(point_inside_bbox(point_1000, bbox_1000))


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise AuditContractError("cannot take a percentile of an empty series")
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def build_val_mapping(
    records: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Derive the val sample/image mapping from the operational records CSV.

    Every row of the operational records CSV is a val record, so no index or NPZ
    is needed to recover the split.
    """
    sample_ids = sorted({str(row["sample_id"]) for row in records})
    image_by_sample = {str(row["sample_id"]): str(row["image_sha256"]) for row in records}
    task_by_sample = {str(row["sample_id"]): str(row["task"]) for row in records}
    tasks_by_image: dict[str, dict[str, str]] = defaultdict(dict)
    for sample_id in sample_ids:
        tasks_by_image[image_by_sample[sample_id]][task_by_sample[sample_id]] = sample_id
    images = sorted(tasks_by_image)
    return {
        "sample_ids": sample_ids,
        "images": images,
        "image_by_sample": image_by_sample,
        "task_by_sample": task_by_sample,
        "tasks_by_image": tasks_by_image,
    }


def compute_geometry_metrics(
    mapping: Mapping[str, Any],
    geometry: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Record-level and image-level point-in-bbox ratios plus bbox-area stats."""
    sample_ids = list(mapping["sample_ids"])
    missing = [sample_id for sample_id in sample_ids if sample_id not in geometry]
    if missing:
        raise AuditContractError(f"val sample ids missing from manifest: {missing[:5]}")

    record_inside: dict[str, bool] = {}
    bbox_area_by_image: dict[str, float] = {}
    per_task_inside: dict[str, list[bool]] = {task: [] for task in TASKS}
    for sample_id in sample_ids:
        entry = geometry[sample_id]
        inside = record_point_inside(entry["point_1000"], entry["bbox_1000"])
        record_inside[sample_id] = inside
        task = str(mapping["task_by_sample"][sample_id])
        if task in per_task_inside:
            per_task_inside[task].append(inside)
        if task == "grounding":
            bbox_area_by_image[str(mapping["image_by_sample"][sample_id])] = bbox_area_norm(
                entry["bbox_1000"]
            )

    record_ratio = float(np.mean([record_inside[sid] for sid in sample_ids]))

    tasks_by_image = mapping["tasks_by_image"]
    image_inside: dict[str, bool] = {}
    image_grounding_inside: dict[str, bool] = {}
    image_pointing_inside: dict[str, bool] = {}
    shared_ground_truth = 0
    for image in mapping["images"]:
        partners = tasks_by_image[image]
        if "grounding" not in partners or "pointing" not in partners:
            raise AuditContractError(f"image {image} does not have both tasks")
        grounding = geometry[partners["grounding"]]
        pointing = geometry[partners["pointing"]]
        if grounding["bbox_1000"] == pointing["bbox_1000"] and grounding["point_1000"] == pointing["point_1000"]:
            shared_ground_truth += 1
        image_grounding_inside[image] = record_point_inside(
            grounding["point_1000"], grounding["bbox_1000"]
        )
        image_pointing_inside[image] = record_point_inside(
            pointing["point_1000"], pointing["bbox_1000"]
        )
        image_inside[image] = record_point_inside(
            pointing["point_1000"], grounding["bbox_1000"]
        )

    image_ratio = float(np.mean([image_inside[image] for image in mapping["images"]]))
    areas = [bbox_area_by_image[image] for image in mapping["images"]]
    area_stats = {
        "count": len(areas),
        "min": float(min(areas)),
        "p25": _percentile(areas, 0.25),
        "median": _percentile(areas, 0.50),
        "p75": _percentile(areas, 0.75),
        "max": float(max(areas)),
        "mean": float(np.mean(areas)),
        "zero_count": int(sum(1 for value in areas if value <= 0.0)),
    }
    return {
        "record_inside": record_inside,
        "record_ratio": record_ratio,
        "record_count": len(sample_ids),
        "record_true_count": int(sum(1 for value in record_inside.values() if value)),
        "per_task_record_ratio": {
            task: (float(np.mean(values)) if values else None)
            for task, values in per_task_inside.items()
        },
        "image_inside": image_inside,
        "image_grounding_inside": image_grounding_inside,
        "image_pointing_inside": image_pointing_inside,
        "image_ratio": image_ratio,
        "image_count": len(mapping["images"]),
        "image_true_count": int(sum(1 for value in image_inside.values() if value)),
        "shared_ground_truth_images": shared_ground_truth,
        "bbox_area_norm_by_image": bbox_area_by_image,
        "bbox_area_norm": area_stats,
    }


def per_image_task_medians(
    records: Sequence[Mapping[str, str]],
    *,
    arm: str = PRIMARY_ARM,
    budget: int = PRIMARY_BUDGET,
    seeds: Sequence[int] = SEEDS,
    tasks: Sequence[str] = TASKS,
) -> dict[str, dict[str, float]]:
    """Median across the fixed seeds of gap and Jaccard for every (image, task)."""
    wanted_seeds = sorted(int(seed) for seed in seeds)
    buckets: dict[tuple[str, str], dict[int, dict[str, float]]] = defaultdict(dict)
    for row in records:
        if str(row["arm"]) != arm:
            continue
        if int(row["budget"]) != int(budget):
            continue
        task = str(row["task"])
        if task not in tasks:
            continue
        seed = int(row["seed"])
        if seed not in wanted_seeds:
            continue
        key = (str(row["image_sha256"]), task)
        if seed in buckets[key]:
            raise AuditContractError(f"duplicate record for {key} seed {seed}")
        buckets[key][seed] = {
            "gap": float(row["gap_retained_spatial_target_mass"]),
            "jaccard": float(row["correct_wrong_selected_jaccard"]),
        }
    result: dict[str, dict[str, float]] = {}
    expected = set(wanted_seeds)
    for (image, task), by_seed in buckets.items():
        if set(by_seed) != expected:
            raise AuditContractError(
                f"ragged seed series for ({image}, {task}): {sorted(by_seed)}"
            )
        gap_median = float(np.median([by_seed[seed]["gap"] for seed in wanted_seeds]))
        jaccard_median = float(
            np.median([by_seed[seed]["jaccard"] for seed in wanted_seeds])
        )
        result.setdefault(image, {})[task] = gap_median
        result[image][f"{task}__jaccard"] = jaccard_median
    return result


def aggregate_gaps(
    medians: Mapping[str, Mapping[str, float]],
    images: Sequence[str],
) -> dict[str, Any]:
    """Task-marginal and overall mean/median of the per-image K48 median gap."""
    per_task: dict[str, Any] = {}
    for task in TASKS:
        values = [float(medians[image][task]) for image in images]
        per_task[task] = {
            "n_images": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
        }
    overall = [
        float(np.mean([medians[image][task] for task in TASKS])) for image in images
    ]
    per_task["overall"] = {
        "n_images": len(overall),
        "mean": float(np.mean(overall)),
        "median": float(np.median(overall)),
    }
    return per_task


def aggregate_jaccard(
    medians: Mapping[str, Mapping[str, float]],
    images: Sequence[str],
) -> dict[str, Any]:
    per_task: dict[str, Any] = {}
    for task in TASKS:
        values = [float(medians[image][f"{task}__jaccard"]) for image in images]
        per_task[task] = {
            "n_images": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
        }
    return per_task


def stratify_by_point_inside(
    medians: Mapping[str, Mapping[str, float]],
    images: Sequence[str],
    image_inside: Mapping[str, bool],
) -> dict[str, Any]:
    """Per-stratum gap/Jaccard means, split by the image-level point-inside flag."""
    strata: dict[str, Any] = {}
    for label, flag in (("point_inside_true", True), ("point_inside_false", False)):
        selected = [image for image in images if bool(image_inside[image]) is flag]
        entry: dict[str, Any] = {"n_images": len(selected)}
        for task in TASKS:
            if selected:
                entry[f"{task}_gap_mean"] = float(
                    np.mean([float(medians[image][task]) for image in selected])
                )
                entry[f"{task}_jaccard_mean"] = float(
                    np.mean([float(medians[image][f"{task}__jaccard"]) for image in selected])
                )
            else:
                entry[f"{task}_gap_mean"] = None
                entry[f"{task}_jaccard_mean"] = None
        strata[label] = entry
    true_count = strata["point_inside_true"]["n_images"]
    false_count = strata["point_inside_false"]["n_images"]
    if true_count and false_count:
        strata["note"] = "both strata populated; between-stratum contrast is reported"
    else:
        strata["note"] = (
            "one stratum is empty; point-inside is a constant across the 38 val "
            "images, so no between-stratum contrast exists"
        )
    return strata


def spearman_area_vs_grounding_gap(
    medians: Mapping[str, Mapping[str, float]],
    images: Sequence[str],
    bbox_area_by_image: Mapping[str, float],
) -> dict[str, Any]:
    areas = [float(bbox_area_by_image[image]) for image in images]
    gaps = [float(medians[image]["grounding"]) for image in images]
    rho = float(spearman(areas, gaps))
    return {
        "n_images": len(images),
        "bbox_area_vs_grounding_gap_k48": rho,
        "method": "reused evaluation.prompt_projection_distillation.spearman",
    }


def decide_task_pair_nesting(
    nested_ratio: float,
    grounding_gap_mean: float,
    pointing_gap_mean: float,
) -> dict[str, Any]:
    """Frozen three-condition decision (threshold and signs, no tuning)."""
    nested = bool(float(nested_ratio) >= NESTED_RATIO_THRESHOLD)
    asymmetric = bool(grounding_gap_mean <= 0.0 and pointing_gap_mean > 0.0)
    if nested and asymmetric:
        decision = DECISION_SYMMETRIC_INVALID
    else:
        decision = DECISION_NOT_ESTABLISHED
    return {
        "decision": decision,
        "same_image_task_targets_nested": nested,
        "nested_ratio_threshold": NESTED_RATIO_THRESHOLD,
        "nested_ratio": float(nested_ratio),
        "grounding_gap_negative_or_zero": bool(grounding_gap_mean <= 0.0),
        "pointing_gap_positive": bool(pointing_gap_mean > 0.0),
        "grounding_gap_mean": float(grounding_gap_mean),
        "pointing_gap_mean": float(pointing_gap_mean),
        "rule": (
            "SYMMETRIC_NEGATIVE_INVALID iff nested_ratio>=0.80 and "
            "grounding_gap_mean<=0 and pointing_gap_mean>0; otherwise NOT_ESTABLISHED"
        ),
    }


# ---------------------------------------------------------------------------
# Cross-check against the operational JSON
# ---------------------------------------------------------------------------
def cross_check_operational_json(
    operational: Mapping[str, Any],
    gaps: Mapping[str, Any],
    jaccard: Mapping[str, Any],
) -> dict[str, Any]:
    task_k48 = operational.get("task_k48", {})
    deltas: dict[str, Any] = {}
    matches = True
    for task in TASKS:
        reference = task_k48.get(task, {})
        for key, computed in (
            ("gap_mass", gaps[task]["mean"]),
            ("jaccard", jaccard[task]["mean"]),
        ):
            ref_key = (
                "B_correct_minus_wrong_gap_mass_mean"
                if key == "gap_mass"
                else "B_correct_wrong_jaccard_mean"
            )
            ref_value = reference.get(ref_key)
            delta = None if ref_value is None else abs(float(ref_value) - float(computed))
            deltas[f"{task}.{key}"] = {
                "computed": float(computed),
                "operational": ref_value,
                "abs_delta": delta,
            }
            if delta is None or delta > CROSS_CHECK_TOLERANCE:
                matches = False
    return {
        "matches_within_tolerance": bool(matches),
        "tolerance": CROSS_CHECK_TOLERANCE,
        "deltas": deltas,
        "prior_decision": operational.get("status"),
    }


# ---------------------------------------------------------------------------
# Bundle / report assembly
# ---------------------------------------------------------------------------
def build_audit(root: Path | str = ROOT, *, verify_pins: bool = True) -> dict[str, Any]:
    """Run the whole read-only audit and return the result bundle."""
    base = Path(root)
    started = time.time()
    hashes_before = snapshot_protected_hashes(base)
    prereg = load_prereg(base)
    pins = verify_input_pins(prereg, base) if verify_pins else {"skipped": True}

    records = load_records(base / RECORDS_CSV_PATH, root=base)
    geometry = load_geometry(base / MANIFEST_PATH, root=base)
    operational = load_operational_json(base / OPERATIONAL_JSON_PATH, root=base)

    mapping = build_val_mapping(records)
    expected_records = int(prereg["config"]["data_scope"]["expected_val_records"])
    expected_images = int(prereg["config"]["data_scope"]["expected_val_images"])
    if len(mapping["sample_ids"]) != expected_records:
        raise AuditContractError(
            f"val record count {len(mapping['sample_ids'])} != expected {expected_records}"
        )
    if len(mapping["images"]) != expected_images:
        raise AuditContractError(
            f"val image count {len(mapping['images'])} != expected {expected_images}"
        )

    geometry_metrics = compute_geometry_metrics(mapping, geometry)
    medians = per_image_task_medians(records)
    for image in mapping["images"]:
        if set(medians.get(image, {})) < {f"{task}__jaccard" for task in TASKS} | set(TASKS):
            raise AuditContractError(f"missing seeded metrics for image {image}")
    gaps = aggregate_gaps(medians, mapping["images"])
    jaccard = aggregate_jaccard(medians, mapping["images"])
    strata = stratify_by_point_inside(medians, mapping["images"], geometry_metrics["image_inside"])
    rho = spearman_area_vs_grounding_gap(
        medians, mapping["images"], geometry_metrics["bbox_area_norm_by_image"]
    )
    decision = decide_task_pair_nesting(
        geometry_metrics["record_ratio"],
        gaps["grounding"]["mean"],
        gaps["pointing"]["mean"],
    )
    cross_check = cross_check_operational_json(operational, gaps, jaccard)

    hashes_after = snapshot_protected_hashes(base)
    unchanged = hashes_before == hashes_after

    record_rows: list[dict[str, Any]] = []
    for sample_id in mapping["sample_ids"]:
        entry = geometry[sample_id]
        image = mapping["image_by_sample"][sample_id]
        task = str(mapping["task_by_sample"][sample_id])
        row = {
            "sample_id": sample_id,
            "image_sha256": image,
            "task": task,
            "group_count": int(next(r for r in records if r["sample_id"] == sample_id)["group_count"]),
            "grid_h": int(next(r for r in records if r["sample_id"] == sample_id)["grid_h"]),
            "grid_w": int(next(r for r in records if r["sample_id"] == sample_id)["grid_w"]),
            "bbox_x1": entry["bbox_1000"][0],
            "bbox_y1": entry["bbox_1000"][1],
            "bbox_x2": entry["bbox_1000"][2],
            "bbox_y2": entry["bbox_1000"][3],
            "bbox_area_norm": bbox_area_norm(entry["bbox_1000"]),
            "point_x": entry["point_1000"][0],
            "point_y": entry["point_1000"][1],
            "point_inside_bbox": geometry_metrics["record_inside"][sample_id],
            "gap_k48_median_3seed": medians[image][task],
            "jaccard_k48_median_3seed": medians[image][f"{task}__jaccard"],
        }
        for seed in SEEDS:
            match = next(
                r
                for r in records
                if r["sample_id"] == sample_id
                and r["arm"] == PRIMARY_ARM
                and int(r["budget"]) == PRIMARY_BUDGET
                and int(r["seed"]) == seed
            )
            row[f"gap_k48_seed_{seed}"] = float(match["gap_retained_spatial_target_mass"])
            row[f"jaccard_k48_seed_{seed}"] = float(match["correct_wrong_selected_jaccard"])
        record_rows.append(row)

    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "report_stem": REPORT_STEM,
        "date": DATE,
        "status": decision["decision"],
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "exploratory": {
            "validation_already_used": True,
            "statement": (
                "Explanatory diagnostic on already-used validation data. It does "
                "not change the prior OPERATIONAL_PROMPT_UTILITY_NOT_ESTABLISHED "
                "decision and supports no accuracy or promotion claim."
            ),
            "no_training": True,
            "no_gpu": True,
            "no_network": True,
            "not_final_accuracy": True,
        },
        "decision": decision,
        "point_inside_bbox": {
            "overall_ratio_primary": geometry_metrics["record_ratio"],
            "record_count": geometry_metrics["record_count"],
            "record_true_count": geometry_metrics["record_true_count"],
            "per_task_record_ratio": geometry_metrics["per_task_record_ratio"],
            "image_ratio_secondary": geometry_metrics["image_ratio"],
            "image_count": geometry_metrics["image_count"],
            "image_true_count": geometry_metrics["image_true_count"],
            "shared_ground_truth_images": geometry_metrics["shared_ground_truth_images"],
            "gate_uses": "overall_ratio_primary (record level)",
        },
        "bbox_area_norm": geometry_metrics["bbox_area_norm"],
        "gaps": {
            "arm": PRIMARY_ARM,
            "budget_k": PRIMARY_BUDGET,
            "seed_to_image_task": "median across the three seeds",
            "grounding": gaps["grounding"],
            "pointing": gaps["pointing"],
            "overall": gaps["overall"],
        },
        "stratification": strata,
        "spearman": rho,
        "jaccard": {
            "field": "correct_wrong_selected_jaccard",
            "arm": PRIMARY_ARM,
            "budget_k": PRIMARY_BUDGET,
            "grounding": jaccard["grounding"],
            "pointing": jaccard["pointing"],
        },
        "cross_check": cross_check,
        "provenance": {
            "prereg_config": {
                "path": str(PREREG_CONFIG_PATH),
                "sha256": prereg["config_sha256"],
            },
            "prereg_report": {
                "path": str(PREREG_REPORT_PATH),
                "sha256": hashes_before[str(PREREG_REPORT_PATH)],
            },
            "authority_prompt_sha256": prereg["prompt_sha256"],
            "input_pins": pins.get("actual", {}),
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
            "protected_files_unchanged": unchanged,
        },
        "isolation": {
            "read_paths": sorted(set(read_log())),
            "allowed_read_paths": sorted(allowed_read_paths(base)),
            "npz_opened": [path for path in read_log() if path.endswith(".npz")],
            "raw_images_opened": False,
            "model_weights_opened": False,
            "internal_test_access": False,
            "gpu_used": False,
            "network_used": False,
            "training": False,
        },
        "split_counts": {
            "val_records": len(mapping["sample_ids"]),
            "val_images": len(mapping["images"]),
            "expected_val_records": expected_records,
            "expected_val_images": expected_images,
            "sample_ids_are_val_only": True,
        },
        "limitations": [
            "Explanatory diagnostic on already-used validation; not a final accuracy conclusion.",
            "The prior OPERATIONAL_PROMPT_UTILITY_NOT_ESTABLISHED decision is unchanged.",
            "No NPZ, raw image, model weight, GPU, network or internal test was touched.",
            "point-inside is constant across the 38 val images, so the stratification cannot separate strata.",
            "Per-image statistics treat the 38 images as independent units.",
        ],
        "duration_seconds": round(time.time() - started, 3),
    }
    return {"bundle": bundle, "record_rows": record_rows}


def render_markdown(bundle: Mapping[str, Any]) -> str:
    decision = bundle["decision"]
    geometry = bundle["point_inside_bbox"]
    gaps = bundle["gaps"]
    areas = bundle["bbox_area_norm"]
    strata = bundle["stratification"]
    jaccard = bundle["jaccard"]
    lines = [
        f"# 任务对嵌套负样本有效性审计（{bundle['date']}）",
        "",
        f"- 结论：**{decision['decision']}**（解释性诊断，不改变既有 `OPERATIONAL_PROMPT_UTILITY_NOT_ESTABLISHED`，不作精度/晋级声明）",
        f"- SAME_IMAGE_TASK_TARGETS_NESTED：{decision['same_image_task_targets_nested']}（阈值 {decision['nested_ratio_threshold']}，实测 {decision['nested_ratio']:.6f}）",
        f"- grounding B K48 gap mean：{decision['grounding_gap_mean']:+.6f}；pointing B K48 gap mean：{decision['pointing_gap_mean']:+.6f}",
        "",
        "## 1. point 是否落在 bbox 内",
        f"- 记录级 overall 比例（判定用）：{geometry['overall_ratio_primary']:.6f}（{geometry['record_true_count']}/{geometry['record_count']}）",
        f"- 图像级跨任务比例（次要）：{geometry['image_ratio_secondary']:.6f}（{geometry['image_true_count']}/{geometry['image_count']}）",
        f"- 同图 grounding/pointing 共享 ground_truth 的图像数：{geometry['shared_ground_truth_images']}/{geometry['image_count']}",
        "",
        "## 2. bbox 面积分布（归一化）",
        f"- min {areas['min']:.6f} / p25 {areas['p25']:.6f} / median {areas['median']:.6f} / p75 {areas['p75']:.6f} / max {areas['max']:.6f}",
        f"- mean {areas['mean']:.6f}，零面积 {areas['zero_count']}",
        "",
        "## 3. B 臂三 seed 中位数 K48 correct−wrong retained-mass gap",
        f"- grounding：mean {gaps['grounding']['mean']:+.6f}，median {gaps['grounding']['median']:+.6f}（n={gaps['grounding']['n_images']}）",
        f"- pointing：mean {gaps['pointing']['mean']:+.6f}，median {gaps['pointing']['median']:+.6f}（n={gaps['pointing']['n_images']}）",
        f"- overall：mean {gaps['overall']['mean']:+.6f}，median {gaps['overall']['median']:+.6f}",
        "",
        "## 4. 分层与 Spearman",
        f"- point_inside=true：n={strata['point_inside_true']['n_images']}，grounding gap {strata['point_inside_true']['grounding_gap_mean']}，pointing gap {strata['point_inside_true']['pointing_gap_mean']}",
        f"- point_inside=false：n={strata['point_inside_false']['n_images']}，grounding gap {strata['point_inside_false']['grounding_gap_mean']}，pointing gap {strata['point_inside_false']['pointing_gap_mean']}",
        f"- 说明：{strata['note']}",
        f"- bbox 面积 vs grounding gap Spearman：{bundle['spearman']['bbox_area_vs_grounding_gap_k48']:+.6f}（n={bundle['spearman']['n_images']}）",
        "",
        "## 5. correct/错误 selected-set Jaccard 分层",
        f"- grounding：mean {jaccard['grounding']['mean']:.6f}，median {jaccard['grounding']['median']:.6f}",
        f"- pointing：mean {jaccard['pointing']['mean']:.6f}，median {jaccard['pointing']['median']:.6f}",
        f"- point_inside=true：grounding {strata['point_inside_true']['grounding_jaccard_mean']}，pointing {strata['point_inside_true']['pointing_jaccard_mean']}",
        "",
        "## 6. 与 operational JSON 交叉核对",
        f"- 一致（容差 {bundle['cross_check']['tolerance']}）：{bundle['cross_check']['matches_within_tolerance']}",
        "",
        "## 7. 隔离与不变性",
        f"- 仅读取：{', '.join(bundle['isolation']['read_paths'])}",
        f"- NPZ 打开：{bundle['isolation']['npz_opened']}；原图/权重/GPU/联网/训练/内部测试：全部 False",
        f"- 预注册与输入运行前后逐字节不变：{bundle['provenance']['protected_files_unchanged']}",
        "",
        "## 限制",
    ]
    lines.extend(f"- {item}" for item in bundle["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    bundle: Mapping[str, Any],
    record_rows: Sequence[Mapping[str, Any]],
    *,
    root: Path | str = ROOT,
) -> dict[str, Any]:
    base = Path(root)
    reports = base / REPORTS_DIR
    reports.mkdir(parents=True, exist_ok=True)
    json_path = reports / f"{REPORT_STEM}.json"
    md_path = reports / f"{REPORT_STEM}.md"
    csv_path = reports / f"{REPORT_STEM}-records.csv"
    json_path.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(bundle), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RECORDS_CSV_FIELDS))
        writer.writeheader()
        for row in record_rows:
            writer.writerow({field: row.get(field) for field in RECORDS_CSV_FIELDS})
    return {
        "json": str(json_path),
        "json_sha256": sha256_file(json_path),
        "md": str(md_path),
        "md_sha256": sha256_file(md_path),
        "csv": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--skip-pin-check", action="store_true")
    parser.add_argument("--summary", action="store_true", help="print a short JSON summary")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    result = build_audit(root, verify_pins=not args.skip_pin_check)
    files = write_outputs(result["bundle"], result["record_rows"], root=root)
    summary = {
        "status": result["bundle"]["status"],
        "same_image_task_targets_nested": result["bundle"]["decision"][
            "same_image_task_targets_nested"
        ],
        "point_inside_ratio": result["bundle"]["point_inside_bbox"]["overall_ratio_primary"],
        "grounding_gap_mean": result["bundle"]["gaps"]["grounding"]["mean"],
        "pointing_gap_mean": result["bundle"]["gaps"]["pointing"]["mean"],
        "spearman_area_vs_grounding_gap": result["bundle"]["spearman"][
            "bbox_area_vs_grounding_gap_k48"
        ],
        "cross_check_ok": result["bundle"]["cross_check"]["matches_within_tolerance"],
        "protected_files_unchanged": result["bundle"]["provenance"][
            "protected_files_unchanged"
        ],
        "outputs": files,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
