#!/usr/bin/env python3
"""Exploratory spatial-selector deployment-link failure decomposition.

This is a **read-only, CPU-only, exploratory diagnostic** on validation data that
was already used by the spatial-student train/val experiment.  It can only inform
the design of the *next* preregistered experiment; it is not a final accuracy
conclusion, not a promotion decision, and it never touches the internal test
split.

What it measures per validation record, per ``B_SPATIAL_KL_PRIMARY`` seed and per
budget ``K in {48, 64, 96}``:

* ``l2_plain``                -- top-K of the no-training L2 baseline
* ``l2_anchor``               -- fixed 3x3 anchor selection on L2
* ``semantic_plain``          -- top-K of the ``prompt_mean_vector`` student
* ``semantic_anchor``         -- 3x3 anchor selection on the student
* ``combined_anchor_lambda_*`` -- 3x3 anchor on min-max combined semantic+L2
* ``oracle_anchor``           -- 3x3 anchor on the exact spatial target (upper bound)

Every mode reports retained spatial-target mass; anchor modes report the
corrected selected-set NDCG; every mode reports its delta and selected-set
Jaccard/overlap against ``l2_anchor``.

Reuse only (no second implementation of any target / scorer / selection /
metric / statistic):

* targets: :func:`evaluation.train_spatial_prompt_selector.attach_spatial_targets`
* scorer: :func:`evaluation.frozen_scorer_reduction_ablation.score_candidate`
* L2: :func:`evaluation.prompt_projection_distillation.l2_baseline_scores`
* combine / anchor: :func:`evaluation.hardware_regular_token_selector.{combine_scores,select_real_groups}`
* metrics: :func:`evaluation.prompt_projection_distillation.{topk_indices,saliency_mass_retained,ndcg_at_k,ndcg_for_selected_indices,jaccard}`
* statistics: :func:`evaluation.prompt_identifiability_diagnostic.paired_effect`
* loader / split: :func:`evaluation.train_pairwise_prompt_selector.load_split_samples`,
  :func:`evaluation.prompt_projection_distillation.split_sample_ids`
* checkpoint: :func:`evaluation.run_hf_hardware_regular_b1.load_projection`
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluation.frozen_scorer_reduction_ablation import score_candidate
from evaluation.hardware_regular_token_selector import combine_scores
from evaluation.prompt_identifiability_diagnostic import paired_effect
from evaluation.prompt_projection_distillation import (
    DistillationContractError,
    jaccard,
    l2_baseline_scores,
    ndcg_at_k,
    ndcg_for_selected_indices,
    saliency_mass_retained,
    split_sample_ids,
    topk_indices,
)
from evaluation.run_hf_hardware_regular_b1 import load_projection
from evaluation.spatial_teacher_feasibility_audit import load_clean_manifest
from evaluation.teacher_pair_signal_audit import load_index_metadata
from evaluation.train_pairwise_prompt_selector import load_split_samples, sha256_file
from evaluation.train_spatial_prompt_selector import (
    _select_groups as select_anchor_groups,
    attach_spatial_targets,
)

REPORT_STEM = "2026-10-03-spatial-selector-failure-diagnostic"
DATE = "2026-10-03"
SCHEMA_VERSION = "1.0"

ROOT = Path(__file__).resolve().parents[1]

PREREG_CONFIG_PATH = Path("configs/spatial_selector_failure_diagnostic.preregistered.json")
PREREG_REPORT_PATH = Path(
    "reports/token_selector/2026-10-03-spatial-selector-failure-diagnostic-preregistration.json"
)
CONFIG_PATH = Path("configs/prompt_projection_distillation.json")
INDEX_PATH = Path("artifacts/token_selector/prompt_projection_features/index.jsonl")
MANIFEST_PATH = Path("datasets/prompt_projection_clean_inductive_v1.jsonl")
LOCK_PATH = Path("datasets/prompt_projection_clean_inductive_v1.lock.json")
SPATIAL_AUDIT_JSON_PATH = Path(
    "reports/token_selector/2026-10-03-spatial-teacher-feasibility.json"
)
SPATIAL_AUDIT_SCRIPT_PATH = Path("evaluation/spatial_teacher_feasibility_audit.py")
HARDWARE_CONFIG_PATH = Path("configs/hardware_regular_token_selector.json")
HARDWARE_CODE_PATH = Path("evaluation/hardware_regular_token_selector.py")

ARTIFACTS_DIR = Path("artifacts/token_selector/spatial_student_trainval")
BEST_ARM = "B_SPATIAL_KL_PRIMARY"
REPORTS_DIR = Path("reports/token_selector")
LOGS_DIR = REPORTS_DIR / "logs"
PRIOR_REPORT_PATH = Path("reports/token_selector/2026-10-03-spatial-student-trainval.json")
PRIOR_PRIMARY_KEY = "B_selector_minus_L2_anchor_mass_k48"

SEEDS: tuple[int, ...] = (20261003, 20261004, 20261005)
BUDGETS: tuple[int, ...] = (48, 64, 96)
LAMBDAS: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
LAMBDA_PRIMARY = 0.5
STUDENT_SCORER = "prompt_mean_vector"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261003
PERMUTATION_RESAMPLES = 10000
CI_LEVEL = 0.95
EQUALITY_TOLERANCE = 1e-12
HEADROOM_THRESHOLD = 0.10

SIZE_BANDS: tuple[tuple[str, int, int], ...] = (
    ("small", 0, 176),
    ("medium", 177, 260),
    ("large", 261, 10**9),
)

DIAGNOSIS_ORDER: tuple[str, ...] = (
    "FUSION_DAMAGE",
    "HEAD_SIGNAL_WEAK",
    "ANCHOR_BUDGET_DAMAGE",
    "TASK_SPECIFIC",
    "HEADROOM_PRESENT",
)

CSV_FIELDS: tuple[str, ...] = (
    "seed",
    "sample_id",
    "image_sha256",
    "task",
    "group_count",
    "grid_h",
    "grid_w",
    "budget",
    "mode_id",
    "family",
    "anchor_selection",
    "lambda_semantic",
    "retained_spatial_target_mass",
    "ndcg",
    "delta_mass_vs_l2_anchor",
    "delta_ndcg_vs_l2_anchor",
    "selected_jaccard_vs_l2_anchor",
    "selected_overlap_count_vs_l2_anchor",
    "selected_count",
)


# ---------------------------------------------------------------------------
# Frozen mode registry
# ---------------------------------------------------------------------------
def mode_specs() -> list[dict[str, Any]]:
    """The frozen 7 mode families / 13 concrete modes (8 lambda sweep values)."""
    specs: list[dict[str, Any]] = [
        {
            "mode_id": "l2_plain",
            "family": "l2_plain",
            "anchor_selection": False,
            "source": "l2",
            "lambda": None,
        },
        {
            "mode_id": "l2_anchor",
            "family": "l2_anchor",
            "anchor_selection": True,
            "source": "l2",
            "lambda": None,
        },
        {
            "mode_id": "semantic_plain",
            "family": "semantic_plain",
            "anchor_selection": False,
            "source": "semantic",
            "lambda": None,
        },
        {
            "mode_id": "semantic_anchor",
            "family": "semantic_anchor",
            "anchor_selection": True,
            "source": "semantic",
            "lambda": None,
        },
    ]
    for lam in LAMBDAS:
        specs.append(
            {
                "mode_id": combined_mode_id(lam),
                "family": (
                    "combined_anchor" if lam == LAMBDA_PRIMARY else "lambda_sweep_anchor"
                ),
                "anchor_selection": True,
                "source": "combined",
                "lambda": float(lam),
            }
        )
    specs.append(
        {
            "mode_id": "oracle_anchor",
            "family": "oracle_anchor",
            "anchor_selection": True,
            "source": "oracle",
            "lambda": None,
        }
    )
    return specs


def combined_mode_id(lam: float) -> str:
    return f"combined_anchor_lambda_{float(lam):.2f}"


def _mode_scores(
    source: str,
    *,
    semantic_scores: np.ndarray,
    l2_scores: np.ndarray,
    target: np.ndarray,
    lam: float | None,
) -> np.ndarray:
    if source == "l2":
        return np.asarray(l2_scores, dtype=np.float64)
    if source == "semantic":
        return np.asarray(semantic_scores, dtype=np.float64)
    if source == "combined":
        if lam is None:
            raise DistillationContractError("combined mode requires a lambda")
        return combine_scores(
            np.asarray(semantic_scores, dtype=np.float64),
            np.asarray(l2_scores, dtype=np.float64),
            float(lam),
        )
    if source == "oracle":
        return np.asarray(target, dtype=np.float64)
    raise DistillationContractError(f"unknown mode source {source!r}")


def _mode_selection(
    scores: np.ndarray,
    *,
    anchor_selection: bool,
    grid_h: int,
    grid_w: int,
    budget: int,
) -> np.ndarray:
    if anchor_selection:
        return np.asarray(select_anchor_groups(scores, grid_h, grid_w, budget), dtype=np.int64)
    return np.asarray(topk_indices(scores, budget), dtype=np.int64)


def record_mode_metrics(
    *,
    semantic_scores: np.ndarray,
    l2_scores: np.ndarray,
    target: np.ndarray,
    grid_h: int,
    grid_w: int,
    budgets: Sequence[int] = BUDGETS,
    specs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, dict[int, dict[str, float | int]]]:
    """Per-mode, per-budget retained mass / NDCG / delta / Jaccard for one record."""
    specs = list(specs) if specs is not None else mode_specs()
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    group_count = int(target.size)
    result: dict[str, dict[int, dict[str, float | int]]] = {}
    for budget in budgets:
        budget = int(budget)
        effective = min(budget, group_count)
        l2_sel = _mode_selection(
            np.asarray(l2_scores, dtype=np.float64),
            anchor_selection=True,
            grid_h=grid_h,
            grid_w=grid_w,
            budget=effective,
        )
        l2_anchor_mass = float(saliency_mass_retained(l2_sel, target))
        l2_anchor_ndcg = float(
            ndcg_for_selected_indices(l2_sel, np.asarray(l2_scores, dtype=np.float64), target, effective)
        )
        l2_anchor_set = {int(v) for v in l2_sel}
        for spec in specs:
            scores = _mode_scores(
                str(spec["source"]),
                semantic_scores=semantic_scores,
                l2_scores=l2_scores,
                target=target,
                lam=spec["lambda"],
            )
            selected = _mode_selection(
                scores,
                anchor_selection=bool(spec["anchor_selection"]),
                grid_h=grid_h,
                grid_w=grid_w,
                budget=effective,
            )
            mass = float(saliency_mass_retained(selected, target))
            if bool(spec["anchor_selection"]):
                ndcg = float(ndcg_for_selected_indices(selected, scores, target, effective))
            else:
                ndcg = float(ndcg_at_k(scores, target, effective))
            selected_set = {int(v) for v in selected}
            overlap = len(selected_set & l2_anchor_set)
            result.setdefault(str(spec["mode_id"]), {})[budget] = {
                "retained_mass": mass,
                "ndcg": ndcg,
                "delta_mass_vs_l2_anchor": mass - l2_anchor_mass,
                "delta_ndcg_vs_l2_anchor": ndcg - l2_anchor_ndcg,
                "selected_jaccard_vs_l2_anchor": float(
                    jaccard(selected, sorted(l2_anchor_set))
                ),
                "selected_overlap_count_vs_l2_anchor": int(overlap),
                "selected_count": int(selected.size),
            }
    return result


# ---------------------------------------------------------------------------
# Image-level aggregation
# ---------------------------------------------------------------------------
def _median(values: Sequence[float]) -> float:
    return float(np.median([float(v) for v in values]))


def per_image_median(
    per_seed_values: Mapping[int, Sequence[float]],
) -> list[float]:
    """Collapse the three seeds to one median per position (image/record)."""
    seeds = sorted(per_seed_values)
    if not seeds:
        raise DistillationContractError("no seeds to aggregate")
    lengths = {len(list(per_seed_values[s])) for s in seeds}
    if len(lengths) != 1:
        raise DistillationContractError("per-seed series must share a length")
    matrix = np.asarray([list(per_seed_values[s]) for s in seeds], dtype=np.float64)
    return [float(v) for v in np.median(matrix, axis=0)]


def summarize(
    values: Mapping[str, float],
    *,
    reference: Mapping[str, float] | None = None,
    seed: int = BOOTSTRAP_SEED,
    n_bootstrap: int = BOOTSTRAP_RESAMPLES,
    n_permutation: int = PERMUTATION_RESAMPLES,
    equality_tolerance: float = EQUALITY_TOLERANCE,
) -> dict[str, Any]:
    """Mean / median / 95% bootstrap CI plus better-equal-worse vs a reference."""
    keys = sorted(values)
    if not keys:
        return {
            "n_images": 0,
            "mean": None,
            "median": None,
            "ci_low": None,
            "ci_high": None,
            "better": None,
            "equal": None,
            "worse": None,
        }
    array = np.asarray([float(values[k]) for k in keys], dtype=np.float64)
    effect = paired_effect(
        array.tolist(),
        seed=int(seed),
        n_bootstrap=int(n_bootstrap),
        n_permutation=int(n_permutation),
        sample_count=len(keys),
        independent_image_units=len(keys),
    )
    better = equal = worse = None
    if reference is not None:
        ref = np.asarray([float(reference[k]) for k in keys], dtype=np.float64)
        diff = array - ref
        better = int(np.sum(diff > equality_tolerance))
        equal = int(np.sum(np.abs(diff) <= equality_tolerance))
        worse = int(np.sum(diff < -equality_tolerance))
    return {
        "n_images": len(keys),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci_low": effect["ci_low"],
        "ci_high": effect["ci_high"],
        "cohens_dz": effect["cohens_dz"],
        "p_value_two_sided_descriptive_only": effect["p_value_two_sided"],
        "better": better,
        "equal": equal,
        "worse": worse,
    }


def series_delta(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, float]:
    keys = sorted(set(a) & set(b))
    return {k: float(a[k]) - float(b[k]) for k in keys}


def subset(values: Mapping[str, float], images: Sequence[str]) -> dict[str, float]:
    return {img: float(values[img]) for img in images if img in values}


# ---------------------------------------------------------------------------
# Diagnosis rules (frozen)
# ---------------------------------------------------------------------------
def diagnose(
    *,
    series: Mapping[str, Mapping[int, Mapping[str, Mapping[str, float]]]],
    series_task: Mapping[
        str, Mapping[int, Mapping[str, Mapping[str, Mapping[str, float]]]]
    ],
    images: Sequence[str],
    task_images: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Apply the preregistration-frozen multi-label diagnosis rules."""
    zero_lam = combined_mode_id(0.0)
    primary = combined_mode_id(LAMBDA_PRIMARY)

    def mean_delta(mode: str, stratum_images: Sequence[str], task: str | None = None) -> float:
        if task is None:
            mode_vals = subset(series[mode][48]["retained_mass"], stratum_images)
            base_vals = subset(series["l2_anchor"][48]["retained_mass"], stratum_images)
        else:
            mode_vals = subset(series_task[mode][48]["retained_mass"][task], stratum_images)
            base_vals = subset(
                series_task["l2_anchor"][48]["retained_mass"][task], stratum_images
            )
        keys = sorted(set(mode_vals) & set(base_vals))
        if not keys:
            return 0.0
        return float(np.mean([mode_vals[k] - base_vals[k] for k in keys]))

    def mean_plain_delta(stratum_images: Sequence[str]) -> float:
        sem = subset(series["semantic_plain"][48]["retained_mass"], stratum_images)
        l2p = subset(series["l2_plain"][48]["retained_mass"], stratum_images)
        keys = sorted(set(sem) & set(l2p))
        if not keys:
            return 0.0
        return float(np.mean([sem[k] - l2p[k] for k in keys]))

    def relative_lambda(lam: float, stratum_images: Sequence[str], task: str | None = None) -> float:
        mode = combined_mode_id(lam)
        if task is None:
            lam_vals = subset(series[mode][48]["retained_mass"], stratum_images)
            base_vals = subset(series[zero_lam][48]["retained_mass"], stratum_images)
        else:
            lam_vals = subset(series_task[mode][48]["retained_mass"][task], stratum_images)
            base_vals = subset(
                series_task[zero_lam][48]["retained_mass"][task], stratum_images
            )
        keys = sorted(set(lam_vals) & set(base_vals))
        if not keys:
            return 0.0
        return float(np.mean([lam_vals[k] - base_vals[k] for k in keys]))

    d_semantic_anchor = mean_delta("semantic_anchor", images)
    d_primary = mean_delta(primary, images)
    d_semantic_plain = mean_plain_delta(images)
    d_oracle = mean_delta("oracle_anchor", images)
    d_primary_grounding = mean_delta(primary, task_images["grounding"], task="grounding")
    d_primary_pointing = mean_delta(primary, task_images["pointing"], task="pointing")

    positive_lambdas = [lam for lam in LAMBDAS if lam > 0.0]
    lambda_relative: dict[str, dict[str, float]] = {}
    for lam in positive_lambdas:
        lambda_relative[f"{lam:.2f}"] = {
            "overall": relative_lambda(lam, images),
            "grounding": relative_lambda(lam, task_images["grounding"], task="grounding"),
            "pointing": relative_lambda(lam, task_images["pointing"], task="pointing"),
        }

    def _sign(value: float) -> int:
        if value > EQUALITY_TOLERANCE:
            return 1
        if value < -EQUALITY_TOLERANCE:
            return -1
        return 0

    head_signal_weak = d_semantic_anchor <= 0.0 and all(
        entry["overall"] <= 0.0 and entry["grounding"] <= 0.0 and entry["pointing"] <= 0.0
        for entry in lambda_relative.values()
    )
    evaluations = {
        "FUSION_DAMAGE": {
            "condition": "D(semantic_anchor) > 0 and D(combined lambda=0.5) <= 0",
            "d_semantic_anchor": d_semantic_anchor,
            "d_combined_lambda_0.50": d_primary,
            "triggered": bool(d_semantic_anchor > 0.0 and d_primary <= 0.0),
        },
        "HEAD_SIGNAL_WEAK": {
            "condition": (
                "D(semantic_anchor) <= 0 and every lambda>0 relative to lambda=0 "
                "has overall<=0 and grounding<=0 and pointing<=0 at K48"
            ),
            "d_semantic_anchor": d_semantic_anchor,
            "lambda_relative_to_lambda_0": lambda_relative,
            "triggered": bool(head_signal_weak),
        },
        "ANCHOR_BUDGET_DAMAGE": {
            "condition": "D_plain(semantic_plain - l2_plain) > 0 and D(semantic_anchor) <= 0",
            "d_semantic_plain_vs_l2_plain": d_semantic_plain,
            "d_semantic_anchor": d_semantic_anchor,
            "triggered": bool(d_semantic_plain > 0.0 and d_semantic_anchor <= 0.0),
        },
        "TASK_SPECIFIC": {
            "condition": "sign(grounding K48 delta) * sign(pointing K48 delta) < 0",
            "d_combined_lambda_0.50_grounding": d_primary_grounding,
            "d_combined_lambda_0.50_pointing": d_primary_pointing,
            "grounding_sign": _sign(d_primary_grounding),
            "pointing_sign": _sign(d_primary_pointing),
            "triggered": bool(
                _sign(d_primary_grounding) * _sign(d_primary_pointing) < 0
            ),
        },
        "HEADROOM_PRESENT": {
            "condition": "mean(oracle_anchor - l2_anchor) at K48 > 0.10",
            "d_oracle_anchor": d_oracle,
            "threshold": HEADROOM_THRESHOLD,
            "triggered": bool(d_oracle > HEADROOM_THRESHOLD),
        },
    }
    labels = [label for label in DIAGNOSIS_ORDER if evaluations[label]["triggered"]]
    return {
        "labels": labels,
        "evaluations": evaluations,
        "k48_image_level_mean_deltas": {
            "semantic_anchor_minus_l2_anchor": d_semantic_anchor,
            "combined_lambda_0.50_minus_l2_anchor": d_primary,
            "semantic_plain_minus_l2_plain": d_semantic_plain,
            "oracle_anchor_minus_l2_anchor": d_oracle,
            "combined_lambda_0.50_grounding_minus_l2_anchor": d_primary_grounding,
            "combined_lambda_0.50_pointing_minus_l2_anchor": d_primary_pointing,
        },
    }


# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------
def verify_prereg_pins(
    prereg_config: Mapping[str, Any],
    *,
    root: Path = ROOT,
    strict: bool = True,
) -> dict[str, Any]:
    """Recompute every pinned SHA256; report per-input unchanged booleans."""
    frozen = prereg_config.get("frozen_inputs", {})
    results: dict[str, Any] = {}
    mismatches: list[str] = []
    for name, entry in frozen.items():
        if isinstance(entry, Mapping) and "path" in entry and "sha256" in entry:
            path = root / str(entry["path"])
            actual = sha256_file(path) if path.is_file() else None
            ok = actual == str(entry["sha256"])
            results[str(entry["path"])] = {"expected": entry["sha256"], "actual": actual, "unchanged": ok}
            if not ok:
                mismatches.append(str(entry["path"]))
        elif name == "val_npz_sha256" and isinstance(entry, Mapping):
            for rel, expected in entry.items():
                path = root / str(rel)
                actual = sha256_file(path) if path.is_file() else None
                ok = actual == str(expected)
                results[str(rel)] = {"expected": expected, "actual": actual, "unchanged": ok}
                if not ok:
                    mismatches.append(str(rel))
        elif name in {"best_checkpoints", "best_checkpoint_metadata", "restricted_val_report_inputs"} and isinstance(entry, Mapping):
            for rel, expected in entry.items():
                path = root / str(rel)
                actual = sha256_file(path) if path.is_file() else None
                ok = actual == str(expected)
                results[str(rel)] = {"expected": expected, "actual": actual, "unchanged": ok}
                if not ok:
                    mismatches.append(str(rel))
    report = {
        "checked": len(results),
        "mismatches": mismatches,
        "pins_unchanged": not mismatches,
        "per_input": results,
    }
    if mismatches and strict:
        raise DistillationContractError(f"pinned inputs changed: {mismatches}")
    return report


# ---------------------------------------------------------------------------
# Prior-report sign cross-check (read-only)
# ---------------------------------------------------------------------------
def prior_gate_sign_cross_check(
    *,
    recomputed_selector_minus_l2anchor: float,
    root: Path = ROOT,
    prior_report_path: Path = PRIOR_REPORT_PATH,
) -> dict[str, Any]:
    """Compare the prior G3 mean with the frozen-definition recomputation.

    The prior trainer's ``aggregate_validation`` computes every non-arm-delta
    test as ``base - metric``.  For the selector test that is
    ``l2_anchor - selector``, the negative of the stated
    ``selector - l2_anchor`` definition.  This function only reports the
    discrepancy; it never edits the prior report.
    """
    path = root / prior_report_path
    if not path.is_file():
        return {"available": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = (
        payload.get("validation_statistics", {})
        .get("primary_tests", {})
        .get(PRIOR_PRIMARY_KEY)
    )
    if not entry:
        return {"available": False}
    prior_mean = entry.get("mean")
    prior_ci = [entry.get("ci_low"), entry.get("ci_high")]
    recomputed = float(recomputed_selector_minus_l2anchor)
    consistent = (
        prior_mean is not None
        and np.sign(float(prior_mean)) == np.sign(recomputed)
        and abs(float(prior_mean)) > 0.0
        and abs(recomputed) > 0.0
    )
    return {
        "available": True,
        "prior_report": str(prior_report_path),
        "prior_report_sha256": sha256_file(path),
        "prior_primary_key": PRIOR_PRIMARY_KEY,
        "prior_definition": entry.get("definition"),
        "prior_mean": None if prior_mean is None else float(prior_mean),
        "prior_ci_low": prior_ci[0],
        "prior_ci_high": prior_ci[1],
        "prior_seed_means": entry.get("seed_means"),
        "recomputed_selector_minus_l2anchor_mean": recomputed,
        "sign_consistent_with_stated_definition": bool(consistent),
        "note": (
            "Prior aggregate_validation computes base - metric = l2_anchor - "
            "selector for this key, while its definition states selector - "
            "l2_anchor; the reported mean is therefore the negative of the "
            "preregistered definition. Reported only; the prior report is not modified."
        ),
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_diagnostic(
    *,
    root: Path = ROOT,
    prereg_config_path: Path = PREREG_CONFIG_PATH,
    prereg_report_path: Path = PREREG_REPORT_PATH,
    index_path: Path = INDEX_PATH,
    manifest_path: Path = MANIFEST_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
    features_dir: Path | None = None,
    seeds: Sequence[int] = SEEDS,
    budgets: Sequence[int] = BUDGETS,
    cache_loader: Callable[..., dict[str, Any]] | None = None,
    verify_pins: bool = True,
    strict_pins: bool = True,
) -> dict[str, Any]:
    """Run the full read-only diagnostic and return the report bundle."""
    started = time.time()
    cfg_path = root / prereg_config_path
    rep_path = root / prereg_report_path
    prereg = json.loads(cfg_path.read_text(encoding="utf-8"))
    prereg_sha_before = sha256_file(cfg_path)
    prereg_report_sha_before = sha256_file(rep_path)

    if prereg_sha_before != json.loads(rep_path.read_text(encoding="utf-8"))["config"]["sha256"]:
        raise DistillationContractError("preregistration config changed since the report was pinned")

    pin_report = (
        verify_prereg_pins(prereg, root=root, strict=strict_pins)
        if verify_pins
        else {"skipped": True}
    )

    index_records = load_index_metadata(root / index_path)
    split_rule = prereg["frozen_split"]
    splits = split_sample_ids(
        index_records,
        seed=int(split_rule["seed"]),
        ratios=split_rule["ratios"],
        salt=str(split_rule["salt"]),
    )
    val_ids = [str(v) for v in splits["val"]]
    test_ids = {str(v) for v in splits["test"]}
    train_ids = {str(v) for v in splits["train"]}
    directory = Path(features_dir) if features_dir else (root / index_path).parent

    # Internal test paths are computed as strings for the isolation log only.
    test_path_strings = {
        str(directory / str(record["npz"]))
        for record in index_records
        if str(record["sample_id"]) in test_ids
    }

    opened: set[str] = set()
    val_samples_raw, val_aggregate, val_paths = load_split_samples(
        root / index_path,
        directory,
        val_ids,
        opened=opened,
        cache_loader=cache_loader,
    )
    opened_list = sorted(opened)
    test_opened = sorted(path for path in opened_list if path in test_path_strings)
    train_opened = sorted(
        path
        for path in opened_list
        if path
        in {
            str(directory / str(record["npz"]))
            for record in index_records
            if str(record["sample_id"]) in train_ids
        }
    )
    if test_opened:
        raise DistillationContractError(f"internal test NPZ were opened: {test_opened}")
    if train_opened:
        raise DistillationContractError(f"train NPZ were opened during a val-only diagnostic: {train_opened}")

    manifest_by_id, _ = load_clean_manifest(root / manifest_path)
    val_samples = attach_spatial_targets(val_samples_raw, manifest_by_id)

    image_task: dict[str, str] = {}
    image_group_count: dict[str, int] = {}
    image_grid: dict[str, tuple[int, int]] = {}
    task_images: dict[str, set[str]] = {"grounding": set(), "pointing": set()}
    for sample in val_samples:
        image = str(sample["source_image_sha256"])
        task = str(sample["task"])
        image_task[image] = task
        image_group_count[image] = int(sample["group_count"])
        image_grid[image] = (int(sample["grid_h"]), int(sample["grid_w"]))
        task_images[task].add(image)
    images = sorted(image_group_count)
    task_images_sorted = {task: sorted(v) for task, v in task_images.items()}

    specs = mode_specs()
    csv_rows: list[dict[str, Any]] = []
    # seed -> (mode, budget, metric, task) -> list of (image, record value)
    seed_image_task_values: dict[int, dict[tuple[str, int, str, str], list[tuple[str, float]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    checkpoint_hashes: dict[str, Any] = {}
    for seed in seeds:
        seed = int(seed)
        ckpt_path = root / artifacts_dir / BEST_ARM / str(seed) / "best.npz"
        projection, file_sha = load_projection(ckpt_path)
        canonical = None
        with np.load(ckpt_path, allow_pickle=False) as archive:
            if "weights_sha256" in archive:
                canonical = str(np.asarray(archive["weights_sha256"]).reshape(-1)[0])
        checkpoint_hashes[str(seed)] = {
            "path": str(ckpt_path.relative_to(root)),
            "file_sha256": file_sha,
            "canonical_weights_sha256": canonical,
        }
        Wt, Wv = projection.Wt, projection.Wv
        for sample in val_samples:
            sample_id = str(sample["sample_id"])
            target = np.asarray(sample["spatial_target"], dtype=np.float64)
            semantic = np.asarray(
                score_candidate(
                    STUDENT_SCORER,
                    sample["group_mean"],
                    sample["prompt_embeddings"],
                    Wt,
                    Wv,
                ),
                dtype=np.float64,
            )
            l2_scores = np.asarray(l2_baseline_scores(sample["group_mean"]), dtype=np.float64)
            metrics = record_mode_metrics(
                semantic_scores=semantic,
                l2_scores=l2_scores,
                target=target,
                grid_h=int(sample["grid_h"]),
                grid_w=int(sample["grid_w"]),
                budgets=budgets,
                specs=specs,
            )
            image = str(sample["source_image_sha256"])
            task = str(sample["task"])
            for spec in specs:
                mode_id = str(spec["mode_id"])
                for budget in budgets:
                    entry = metrics[mode_id][int(budget)]
                    csv_rows.append(
                        {
                            "seed": seed,
                            "sample_id": sample_id,
                            "image_sha256": image,
                            "task": task,
                            "group_count": int(sample["group_count"]),
                            "grid_h": int(sample["grid_h"]),
                            "grid_w": int(sample["grid_w"]),
                            "budget": int(budget),
                            "mode_id": mode_id,
                            "family": str(spec["family"]),
                            "anchor_selection": bool(spec["anchor_selection"]),
                            "lambda_semantic": (
                                "" if spec["lambda"] is None else float(spec["lambda"])
                            ),
                            "retained_spatial_target_mass": entry["retained_mass"],
                            "ndcg": entry["ndcg"],
                            "delta_mass_vs_l2_anchor": entry["delta_mass_vs_l2_anchor"],
                            "delta_ndcg_vs_l2_anchor": entry["delta_ndcg_vs_l2_anchor"],
                            "selected_jaccard_vs_l2_anchor": entry[
                                "selected_jaccard_vs_l2_anchor"
                            ],
                            "selected_overlap_count_vs_l2_anchor": entry[
                                "selected_overlap_count_vs_l2_anchor"
                            ],
                            "selected_count": entry["selected_count"],
                        }
                    )
                    for metric in (
                        "retained_mass",
                        "ndcg",
                        "delta_mass_vs_l2_anchor",
                        "delta_ndcg_vs_l2_anchor",
                        "selected_jaccard_vs_l2_anchor",
                        "selected_overlap_count_vs_l2_anchor",
                    ):
                        key = (mode_id, int(budget), metric, task)
                        seed_image_task_values[seed][key].append((image, float(entry[metric])))

    # Build per-image-per-task medians across seeds.
    series_task: dict[str, dict[int, dict[str, dict[str, dict[str, float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    all_keys = set()
    for seed in seeds:
        all_keys.update(seed_image_task_values[int(seed)].keys())
    for key in sorted(all_keys):
        mode_id, budget, metric, task = key
        # group per-seed values by image (one record per image-task per seed)
        per_image_seed: dict[str, dict[int, float]] = defaultdict(dict)
        for seed in seeds:
            for image, value in seed_image_task_values[int(seed)][key]:
                per_image_seed[image][int(seed)] = value
        for image, seed_vals in per_image_seed.items():
            values = [seed_vals[s] for s in seeds if s in seed_vals]
            series_task[mode_id][budget][metric][task][image] = _median(values)

    # Overall image series: per seed average the grounding/pointing records, then
    # take the per-image median across seeds.
    image_seed_tasks: dict[
        tuple[str, int, str], dict[str, dict[int, list[float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for seed in seeds:
        for key, entries in seed_image_task_values[int(seed)].items():
            mode_id, budget, metric, task = key
            for image, value in entries:
                image_seed_tasks[(mode_id, int(budget), metric)][image][int(seed)].append(
                    float(value)
                )
    series: dict[str, dict[int, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for (mode_id, budget, metric), per_image in image_seed_tasks.items():
        for image, seed_tasks in per_image.items():
            per_seed_means = [
                float(np.mean(seed_tasks[s])) for s in seeds if s in seed_tasks
            ]
            if per_seed_means:
                series[mode_id][budget][metric][image] = _median(per_seed_means)
    modes_present = sorted({key[0] for key in all_keys})
    metrics_present = sorted({key[2] for key in all_keys})
    budgets_present = sorted({key[1] for key in all_keys})

    strata: dict[str, list[str]] = {"overall": images}
    strata["grounding"] = task_images_sorted["grounding"]
    strata["pointing"] = task_images_sorted["pointing"]
    by_group_count: dict[str, list[str]] = defaultdict(list)
    by_grid: dict[str, list[str]] = defaultdict(list)
    by_size_band: dict[str, list[str]] = {name: [] for name, _, _ in SIZE_BANDS}
    for image in images:
        group_count = image_group_count[image]
        grid = image_grid[image]
        by_group_count[str(group_count)].append(image)
        by_grid[f"{grid[0]}x{grid[1]}"].append(image)
        for name, low, high in SIZE_BANDS:
            if low <= group_count <= high:
                by_size_band[name].append(image)
                break

    def _values(
        mode_id: str,
        budget: int,
        metric: str,
        images_sub: Sequence[str],
        task: str | None = None,
    ) -> dict[str, float]:
        if task is not None:
            source = series_task[mode_id][budget][metric].get(task, {})
        else:
            source = series[mode_id][budget][metric]
        return subset(source, images_sub)

    def _mode_summary(mode_id: str) -> dict[str, Any]:
        out: dict[str, Any] = {
            "mode_id": mode_id,
            "family": next(s["family"] for s in specs if s["mode_id"] == mode_id),
            "anchor_selection": next(
                bool(s["anchor_selection"]) for s in specs if s["mode_id"] == mode_id
            ),
        }
        strata_specs: list[tuple[str, Sequence[str], str | None]] = [
            ("overall", strata["overall"], None),
            ("grounding", strata["grounding"], "grounding"),
            ("pointing", strata["pointing"], "pointing"),
        ]
        for key in sorted(by_group_count):
            strata_specs.append((f"group_count={key}", by_group_count[key], None))
        for key in sorted(by_grid):
            strata_specs.append((f"grid={key}", by_grid[key], None))
        for key in by_size_band:
            strata_specs.append((f"size_band={key}", by_size_band[key], None))
        for lookup, images_sub, task in strata_specs:
            per_budget: dict[str, Any] = {}
            for budget in budgets_present:
                mass = _values(mode_id, budget, "retained_mass", images_sub, task)
                ndcg = _values(mode_id, budget, "ndcg", images_sub, task)
                base_mass = _values(
                    "l2_anchor", budget, "retained_mass", images_sub, task
                )
                base_ndcg = _values("l2_anchor", budget, "ndcg", images_sub, task)
                jac = _values(
                    mode_id, budget, "selected_jaccard_vs_l2_anchor", images_sub, task
                )
                overlap = _values(
                    mode_id,
                    budget,
                    "selected_overlap_count_vs_l2_anchor",
                    images_sub,
                    task,
                )
                delta_mass = series_delta(mass, base_mass)
                delta_ndcg = series_delta(ndcg, base_ndcg)
                per_budget[f"k{int(budget)}"] = {
                    "retained_spatial_target_mass": summarize(
                        mass, reference=base_mass
                    ),
                    "ndcg": summarize(ndcg, reference=base_ndcg),
                    "delta_mass_vs_l2_anchor": summarize(
                        delta_mass, reference={k: 0.0 for k in delta_mass}
                    ),
                    "delta_ndcg_vs_l2_anchor": summarize(
                        delta_ndcg, reference={k: 0.0 for k in delta_ndcg}
                    ),
                    "selected_jaccard_vs_l2_anchor_mean": (
                        float(np.mean(list(jac.values()))) if jac else None
                    ),
                    "selected_overlap_count_vs_l2_anchor_mean": (
                        float(np.mean(list(overlap.values()))) if overlap else None
                    ),
                }
            out[lookup] = per_budget
        return out

    modes = {str(spec["mode_id"]): _mode_summary(str(spec["mode_id"])) for spec in specs}

    diagnosis = diagnose(
        series=series,
        series_task=series_task,
        images=images,
        task_images=task_images_sorted,
    )
    prior_consistency = prior_gate_sign_cross_check(
        recomputed_selector_minus_l2anchor=float(
            modes[combined_mode_id(LAMBDA_PRIMARY)]["overall"]["k48"][
                "delta_mass_vs_l2_anchor"
            ]["mean"]
        ),
        root=root,
    )

    # Narrative interpretation is *derived* from the frozen metric bundle; nothing
    # here is hand-written so a rerun can never keep a stale statement.
    primary_mode = combined_mode_id(LAMBDA_PRIMARY)
    primary_k48_delta = modes[primary_mode]["overall"]["k48"]["delta_mass_vs_l2_anchor"]
    grid_k48_means = {
        key.split("=", 1)[1]: value["k48"]["delta_mass_vs_l2_anchor"]["mean"]
        for key, value in modes[primary_mode].items()
        if key.startswith("grid=")
    }
    grid_group_count: dict[str, int | None] = {}
    for grid_key in grid_k48_means:
        grid_group_count[grid_key] = None
        if "x" in grid_key:
            grid_h_key, grid_w_key = (int(part) for part in grid_key.split("x"))
            for image in images:
                if image_grid[image] == (grid_h_key, grid_w_key):
                    grid_group_count[grid_key] = image_group_count[image]
                    break
    negative_grids = sorted(key for key, value in grid_k48_means.items() if value < 0.0)
    lowest_grid = min(grid_k48_means, key=grid_k48_means.get) if grid_k48_means else None
    link_healthy = bool(
        diagnosis["k48_image_level_mean_deltas"]["semantic_anchor_minus_l2_anchor"] > 0.0
        and diagnosis["k48_image_level_mean_deltas"][
            "combined_lambda_0.50_minus_l2_anchor"
        ]
        > 0.0
    )
    heterogeneity = (
        f"combined lambda=0.5 + anchor is better on "
        f"{primary_k48_delta['better']}/{primary_k48_delta['n_images']} images and worse on "
        f"{primary_k48_delta['worse']}/{primary_k48_delta['n_images']}"
    )
    if negative_grids:
        described = ", ".join(
            f"{grid} (G={grid_group_count[grid]}, delta {grid_k48_means[grid]:+.6f})"
            for grid in negative_grids
        )
        heterogeneity += f"; negative grid strata: {described}"
    else:
        heterogeneity += "; no grid stratum has a negative delta"
    heterogeneity += (
        f". Lowest grid-stratum delta: {lowest_grid} "
        f"({grid_k48_means[lowest_grid]:+.6f})."
        if lowest_grid is not None
        else "."
    )
    labels_text = "、".join(diagnosis["labels"]) if diagnosis["labels"] else "（无）"
    summary = (
        f"Frozen diagnosis labels (fixed order): {labels_text}. "
        "K48 image-level mean retained-spatial-target-mass delta vs L2+anchor: "
        "semantic+anchor "
        f"{diagnosis['k48_image_level_mean_deltas']['semantic_anchor_minus_l2_anchor']:+.6f}, "
        "combined(lambda=0.5)+anchor "
        f"{diagnosis['k48_image_level_mean_deltas']['combined_lambda_0.50_minus_l2_anchor']:+.6f}, "
        "oracle+anchor "
        f"{diagnosis['k48_image_level_mean_deltas']['oracle_anchor_minus_l2_anchor']:+.6f}."
    )
    summary += (
        " On the preregistered sign the retained-mass deployment link did NOT fail."
        if link_healthy
        else " On the preregistered sign the retained-mass deployment link failed."
    )
    if not prior_consistency.get("sign_consistent_with_stated_definition", True):
        summary += (
            " The prior report's G3 primary test is sign-flipped relative to its stated "
            "definition (see prior_report_consistency)."
        )
    interpretation = {
        "retained_mass_link_healthy": link_healthy,
        "task_signs": {
            "grounding": diagnosis["evaluations"]["TASK_SPECIFIC"]["grounding_sign"],
            "pointing": diagnosis["evaluations"]["TASK_SPECIFIC"]["pointing_sign"],
        },
        "within_task_heterogeneity": heterogeneity,
        "summary": summary,
    }

    prereg_sha_after = sha256_file(cfg_path)
    prereg_report_sha_after = sha256_file(rep_path)

    lambda_sweep = {
        f"{lam:.2f}": {
            "mode_id": combined_mode_id(lam),
            "combined_anchor_k48_overall": modes[combined_mode_id(lam)]["overall"]["k48"],
        }
        for lam in LAMBDAS
    }

    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "report_stem": REPORT_STEM,
        "date": DATE,
        "status": "SPATIAL_SELECTOR_FAILURE_DIAGNOSED",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "exploratory": {
            "validation_already_used": True,
            "statement": (
                "This is an exploratory failure decomposition on already-used "
                "validation data; it may only inform the next preregistration and "
                "cannot support a final accuracy or promotion conclusion."
            ),
            "no_internal_test": True,
            "no_gpu": True,
            "no_training": True,
            "lambda_sweep_descriptive_only": True,
        },
        "provenance": {
            "prereg_config": {
                "path": str(prereg_config_path),
                "sha256_before": prereg_sha_before,
                "sha256_after": prereg_sha_after,
                "unchanged": prereg_sha_before == prereg_sha_after,
            },
            "prereg_report": {
                "path": str(prereg_report_path),
                "sha256_before": prereg_report_sha_before,
                "sha256_after": prereg_report_sha_after,
                "unchanged": prereg_report_sha_before == prereg_report_sha_after,
            },
            "pins": pin_report,
            "checkpoints": checkpoint_hashes,
            "val_npz_aggregate_sha256": val_aggregate,
            "val_npz_aggregate_matches_prereg": (
                val_aggregate
                == prereg["frozen_inputs"].get("val_npz_aggregate_sha256")
            ),
            "val_npz_count": len(val_paths),
        },
        "isolation": {
            "opened_npz_count": len(opened_list),
            "opened_only_val_npz": not test_opened and not train_opened,
            "test_npz_opened": test_opened,
            "train_npz_opened": train_opened,
            "test_npz_resolved_hashed_or_opened": False,
            "internal_test_report_files_parsed": False,
            "restricted_val_report_files_parsed": bool(
                prior_consistency.get("available")
            ),
            "restricted_val_report_parsed_note": (
                "restricted val report files are byte-hashed by verify_prereg_pins; "
                "the trainval report JSON is additionally read only for the G3 sign "
                "cross-check and is never used as a metric source; the val-samples CSV "
                "is never parsed by the diagnostic itself"
            ),
            "restricted_val_samples_csv_or_json_parsed_as_metric_source": False,
            "prior_report_json_parsed_for_sign_cross_check_only": bool(
                prior_consistency.get("available")
            ),
        },
        "split_counts": {
            "train_records": len(splits["train"]),
            "val_records": len(splits["val"]),
            "test_records": len(splits["test"]),
            "val_images": len(images),
            "val_grounding_images": len(task_images_sorted["grounding"]),
            "val_pointing_images": len(task_images_sorted["pointing"]),
            "expected_val_records": 76,
            "expected_val_images": 38,
        },
        "config": {
            "best_arm": BEST_ARM,
            "seeds": [int(s) for s in seeds],
            "budgets": [int(b) for b in budgets],
            "lambda_sweep": [float(v) for v in LAMBDAS],
            "lambda_primary": LAMBDA_PRIMARY,
            "student_scorer": STUDENT_SCORER,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "equality_tolerance": EQUALITY_TOLERANCE,
            "headroom_threshold": HEADROOM_THRESHOLD,
            "size_bands": [
                {"name": name, "low": low, "high": high}
                for name, low, high in SIZE_BANDS
            ],
            "mode_families": sorted({str(s["family"]) for s in specs}),
        },
        "modes": modes,
        "lambda_sweep": lambda_sweep,
        "diagnosis": diagnosis,
        "interpretation": interpretation,
        "prior_report_consistency": prior_consistency,
        "headroom": {
            "oracle_anchor_minus_l2_anchor_k48": modes["oracle_anchor"]["overall"]["k48"][
                "delta_mass_vs_l2_anchor"
            ],
            "threshold": HEADROOM_THRESHOLD,
            "present": diagnosis["evaluations"]["HEADROOM_PRESENT"]["triggered"],
        },
        "size_strata": {
            "by_group_count": {k: len(v) for k, v in sorted(by_group_count.items())},
            "by_grid": {k: len(v) for k, v in sorted(by_grid.items())},
            "by_size_band": {k: len(v) for k, v in by_size_band.items()},
        },
        "limitations": [
            "Exploratory diagnostic on already-used validation; not a final accuracy conclusion.",
            "No internal test NPZ was resolved, hashed, opened or scored.",
            "Lambda sweep is descriptive only; no significance or promotion claim.",
            "The oracle+anchor mode is an observational reachable upper bound, not a deployable selector.",
            "Image-level statistics treat the 38 images as independent units.",
            "This run reuses the frozen spatial targets, scorer, combine and anchor rules; it does not retrain.",
            "The prior report's G3 primary test is computed as l2_anchor - selector by aggregate_validation, the negative of its stated selector - l2_anchor definition; this decomposition uses the stated sign and reports the discrepancy without editing the prior report.",
            "Retained-mass deltas are descriptive and are not a significance or promotion test; the large oracle headroom only bounds achievable mass, not task accuracy.",
        ],
        "duration_seconds": round(time.time() - started, 3),
    }
    return {"bundle": bundle, "csv_rows": csv_rows, "prereg": prereg}


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def render_markdown(bundle: Mapping[str, Any]) -> str:
    diag = bundle["diagnosis"]
    labels = "、".join(diag["labels"]) if diag["labels"] else "（无命中标签）"
    prov = bundle["provenance"]
    iso = bundle["isolation"]
    lines = [
        "# 空间学生部署链路失效分解（探索性诊断）",
        "",
        f"- 日期：{bundle['date']}；状态：`{bundle['status']}`",
        f"- 判因标签：**{labels}**",
        "- 这是**已使用 validation 的探索性诊断**，只能指导下一次预注册，不能作为最终精度结论，且未触碰 internal test。",
        "",
        "## 关键 K48 图像级均值 delta（mode − l2_anchor，retained spatial-target mass）",
        "",
        "| 对比 | delta |",
        "| --- | --- |",
    ]
    for key, value in diag["k48_image_level_mean_deltas"].items():
        lines.append(f"| {key} | {value:+.6f} |")
    lines += [
        "",
        "## Lambda sweep（K48 overall，anchor，retained mass mean；仅描述）",
        "",
        "| lambda | mean mass | delta vs l2_anchor | jaccard vs l2_anchor |",
        "| --- | --- | --- | --- |",
    ]
    for lam, entry in bundle["lambda_sweep"].items():
        leaf = entry["combined_anchor_k48_overall"]
        lines.append(
            f"| {lam} | {leaf['retained_spatial_target_mass']['mean']:.6f} | "
            f"{leaf['delta_mass_vs_l2_anchor']['mean']:+.6f} | "
            f"{leaf['selected_jaccard_vs_l2_anchor_mean']:.4f} |"
        )
    lines += [
        "",
        "## 分任务（K48，lambda=0.5 anchor 相对 L2+anchor）",
        "",
        "| 任务 | delta mass | delta ndcg |",
        "| --- | --- | --- |",
    ]
    for task in ("grounding", "pointing"):
        leaf = bundle["modes"][combined_mode_id(LAMBDA_PRIMARY)][task]["k48"]
        lines.append(
            f"| {task} | {leaf['delta_mass_vs_l2_anchor']['mean']:+.6f} | "
            f"{leaf['delta_ndcg_vs_l2_anchor']['mean']:+.6f} |"
        )
    head = bundle["headroom"]["oracle_anchor_minus_l2_anchor_k48"]
    lines += [
        "",
        "## Headroom（oracle+anchor 相对 L2+anchor，K48 retained mass）",
        "",
        f"- mean delta = {head['mean']:+.6f}；门槛 = {bundle['headroom']['threshold']}；"
        f"HEADROOM_PRESENT = {bundle['headroom']['present']}",
        "",
        "## 各模式 K48 overall 指标（图像级 median-of-seeds 后汇总）",
        "",
        "| 模式 | retained mass mean | mass delta | ndcg mean | ndcg delta | jaccard vs L2+anchor | better/equal/worse |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode_id, mode in bundle["modes"].items():
        leaf = mode["overall"]["k48"]
        dm = leaf["delta_mass_vs_l2_anchor"]
        lines.append(
            f"| {mode_id} | {leaf['retained_spatial_target_mass']['mean']:.6f} | "
            f"{dm['mean']:+.6f} | {leaf['ndcg']['mean']:.6f} | "
            f"{leaf['delta_ndcg_vs_l2_anchor']['mean']:+.6f} | "
            f"{leaf['selected_jaccard_vs_l2_anchor_mean']:.4f} | "
            f"{dm['better']}/{dm['equal']}/{dm['worse']} |"
        )
    interp = bundle.get("interpretation", {})
    if interp:
        lines += [
            "",
            "## 判因说明（由本报告数值实时生成，非手写）",
            "",
            f"- {interp.get('summary', '')}",
            f"- 任务内异质性：{interp.get('within_task_heterogeneity', '')}",
            f"- retained-mass 部署链路健康（semantic+anchor 与 lambda=0.5 combined+anchor 均 > L2+anchor）："
            f"**{interp.get('retained_mass_link_healthy')}**",
        ]
    prior = bundle.get("prior_report_consistency", {})
    lines += ["", "## 与既有报告的符号一致性（只读核对）", ""]
    if prior.get("available"):
        lines += [
            f"- 既有报告 `{prior['prior_report']}` 的 `{prior['prior_primary_key']}` mean = "
            f"{prior['prior_mean']:+.6f}（CI [{prior['prior_ci_low']:+.6f}, {prior['prior_ci_high']:+.6f}]）",
            f"- 按预注册定义 selector − L2+anchor 重算 mean = "
            f"{prior['recomputed_selector_minus_l2anchor_mean']:+.6f}",
            f"- 与定义符号一致：**{prior['sign_consistent_with_stated_definition']}**",
            f"- 说明：{prior['note']}",
        ]
    else:
        lines.append("- 既有报告不可用，跳过。")
    lines += [
        "",
        "## 隔离与溯源",
        "",
        f"- 打开 NPZ 数：{iso['opened_npz_count']}（仅 val：{iso['opened_only_val_npz']}）",
        f"- test NPZ 打开：{iso['test_npz_opened']}；train NPZ 打开：{iso['train_npz_opened']}",
        f"- 预注册 config 运行前后 SHA 不变：{prov['prereg_config']['unchanged']}",
        f"- 预注册 report 运行前后 SHA 不变：{prov['prereg_report']['unchanged']}",
        f"- 所有 pin 输入不变：{prov['pins'].get('pins_unchanged')}",
        f"- val NPZ aggregate 与预注册一致：{prov['val_npz_aggregate_matches_prereg']}",
        "",
        "## 限制",
        "",
    ]
    for item in bundle["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    bundle: Mapping[str, Any],
    csv_rows: Sequence[Mapping[str, Any]],
    *,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, str]:
    json_path = reports_dir / f"{REPORT_STEM}.json"
    md_path = reports_dir / f"{REPORT_STEM}.md"
    csv_path = reports_dir / f"{REPORT_STEM}-records.csv"
    _write_json(json_path, bundle)
    md_path.write_text(render_markdown(bundle), encoding="utf-8")
    _write_csv(csv_path, csv_rows)
    return {
        "json": str(json_path),
        "md": str(md_path),
        "csv": str(csv_path),
        "json_sha256": sha256_file(json_path),
        "md_sha256": sha256_file(md_path),
        "csv_sha256": sha256_file(csv_path),
    }


def harness_payload(bundle: Mapping[str, Any], files: Mapping[str, str]) -> dict[str, Any]:
    modes = bundle.get("modes", {})
    primary = combined_mode_id(LAMBDA_PRIMARY)

    def _k48_delta(mode_id: str, stratum: str) -> float | None:
        try:
            return float(
                modes[mode_id][stratum]["k48"]["delta_mass_vs_l2_anchor"]["mean"]
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _k48_mass(mode_id: str, stratum: str) -> float | None:
        try:
            return float(
                modes[mode_id][stratum]["k48"]["retained_spatial_target_mass"]["mean"]
            )
        except (KeyError, TypeError, ValueError):
            return None

    lambda_best: dict[str, Any] = {"lambda": None, "retained_mass_mean": None}
    for lam, entry in sorted(bundle.get("lambda_sweep", {}).items()):
        try:
            mean = float(
                entry["combined_anchor_k48_overall"]["retained_spatial_target_mass"][
                    "mean"
                ]
            )
        except (KeyError, TypeError, ValueError):
            continue
        if lambda_best["retained_mass_mean"] is None or mean > lambda_best["retained_mass_mean"]:
            lambda_best = {"lambda": lam, "retained_mass_mean": mean}
    lambda_best["note"] = "descriptive only; no significance or promotion claim"

    return {
        "schema_version": SCHEMA_VERSION,
        "report_stem": REPORT_STEM,
        "status": bundle["status"],
        "diagnosis_labels": bundle["diagnosis"]["labels"],
        "k48_image_level_mean_deltas": bundle["diagnosis"]["k48_image_level_mean_deltas"],
        "task_stratification": {
            "grounding": {
                primary: _k48_delta(primary, "grounding"),
                "semantic_anchor": _k48_delta("semantic_anchor", "grounding"),
                "oracle_anchor": _k48_delta("oracle_anchor", "grounding"),
            },
            "pointing": {
                primary: _k48_delta(primary, "pointing"),
                "semantic_anchor": _k48_delta("semantic_anchor", "pointing"),
                "oracle_anchor": _k48_delta("oracle_anchor", "pointing"),
            },
        },
        "l2_anchor_k48_overall_mass_mean": _k48_mass("l2_anchor", "overall"),
        "lambda_sweep_best": lambda_best,
        "headroom_present": bundle["headroom"]["present"],
        "retained_mass_link_healthy": bundle.get("interpretation", {}).get(
            "retained_mass_link_healthy"
        ),
        "prior_report_sign_consistent": bundle.get(
            "prior_report_consistency", {}
        ).get("sign_consistent_with_stated_definition"),
        "isolation": bundle["isolation"],
        "provenance": {
            "prereg_config_unchanged": bundle["provenance"]["prereg_config"]["unchanged"],
            "prereg_report_unchanged": bundle["provenance"]["prereg_report"]["unchanged"],
            "pins_unchanged": bundle["provenance"]["pins"].get("pins_unchanged"),
            "val_npz_aggregate_matches_prereg": bundle["provenance"][
                "val_npz_aggregate_matches_prereg"
            ],
        },
        "limitations": list(bundle.get("limitations", [])),
        "files": dict(files),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--no-pin-verify", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    result = run_diagnostic(
        root=root,
        verify_pins=not args.no_pin_verify,
    )
    bundle = result["bundle"]
    files: dict[str, str] = {}
    if not args.no_write:
        files = write_outputs(bundle, result["csv_rows"], reports_dir=root / REPORTS_DIR)
        harness_path = root / LOGS_DIR / f"{REPORT_STEM}-harness-result.json"
        _write_json(harness_path, harness_payload(bundle, files))
        files["harness_result"] = str(harness_path)
    summary = {
        "status": bundle["status"],
        "diagnosis_labels": bundle["diagnosis"]["labels"],
        "k48_deltas": bundle["diagnosis"]["k48_image_level_mean_deltas"],
        "headroom_present": bundle["headroom"]["present"],
        "files": files,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
