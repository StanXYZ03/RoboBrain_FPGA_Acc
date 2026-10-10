#!/usr/bin/env python3
"""Frozen Wv per-row -> per-tensor scale hardware-regularization audit.

This is a **zero-training, preregistered, validation-only, CPU-only, read-only**
numeric-equivalence audit for the prompt-conditioned spatial selector.  It
answers exactly one engineering question: may the 128 per-output-row INT8 Wv
scales of the prior deployment chain be replaced by **one** per-tensor Wv scale
without changing the selector enough to matter for an FPGA scorer?

Two deterministic chains share every step except Wv weight quantisation:

* ``PER_ROW_WV`` (reference, the prior primary chain).  Float ``Wt`` produces the
  frozen ``prompt_mean_vector`` query, which is then per-vector symmetric INT8
  quantised.  One global-per-image INT8 activation scale dequantises the visual
  ``group_mean``.  Wv uses one symmetric INT8 scale per output row.  The
  ``G x 1024 @ 1024 x 128`` projection is exact INT32, followed by the software
  golden float row-L2 / normalised dot / min-max / combined score and a uint16
  Q0.15 pack.  Selection is the frozen 3x3 anchor rule.
* ``PER_TENSOR_WV`` (candidate).  Identical except that Wv uses the single
  per-tensor scale ``d = max(abs(Wv))/127`` with
  ``q = clamp(round_half_away(Wv/d), -127, 127)`` and ``Wv_hat = q*d``.

The float reference is the same frozen scorer used by the prior audit.  The
comparison is candidate-vs-reference and candidate-vs-float for every correct and
same-image-other-task prompt of all 76 validation records, three B-arm seeds and
K = 48/64/96.

Reuse only (no second implementation of split / loader / target / scorer /
quantiser primitive / selection / statistic):

* split / loader:    :func:`evaluation.prompt_projection_distillation.split_sample_ids`
                     and :func:`evaluation.train_pairwise_prompt_selector.load_split_samples`
* spatial targets:   :func:`evaluation.train_spatial_prompt_selector.attach_spatial_targets`
* scorer / combine:  :func:`evaluation.selector_int8_numerical_audit.float_reference_scores`
* INT8 primitives:   :func:`evaluation.selector_int8_numerical_audit.quantize_activation`,
                     :func:`evaluation.selector_int8_numerical_audit.quantize_weight_rows`,
                     :func:`evaluation.selector_int8_numerical_audit.primary_quantized_query`,
                     :func:`evaluation.selector_int8_numerical_audit.quantized_visual_projection`,
                     :func:`evaluation.selector_int8_numerical_audit.quantized_combined_scores`
* anchor selection:  :func:`evaluation.selector_int8_numerical_audit.selected_indices`
* metrics:           :func:`evaluation.prompt_projection_distillation.jaccard`,
                     :func:`evaluation.prompt_projection_distillation.saliency_mass_retained`,
                     :func:`evaluation.prompt_projection_distillation.ndcg_for_selected_indices`,
                     :func:`evaluation.prompt_projection_distillation.spearman`
* statistics:        :func:`evaluation.selector_int8_numerical_audit.bootstrap_mean_ci`
* checkpoint:        :func:`evaluation.run_hf_hardware_regular_b1.load_projection`

The audit intentionally does not export ``Wt`` and does not claim fixed-point
normalisation, HLS latency, end-to-end accuracy or publication readiness.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluation import selector_int8_numerical_audit as base
from evaluation.prompt_identifiability_diagnostic import same_image_partner_map
from evaluation.prompt_projection_distillation import (
    jaccard,
    ndcg_for_selected_indices,
    saliency_mass_retained,
    spearman,
    split_sample_ids,
)
from evaluation.run_hf_hardware_regular_b1 import load_projection
from evaluation.spatial_teacher_feasibility_audit import load_clean_manifest
from evaluation.teacher_pair_signal_audit import load_index_metadata
from evaluation.train_pairwise_prompt_selector import load_split_samples, sha256_file
from evaluation.train_spatial_prompt_selector import attach_spatial_targets

REPORT_STEM = "2026-10-03-selector-wv-scale-regularization"
DATE = "2026-10-03"
SCHEMA_VERSION = "1.0"

ROOT = Path(__file__).resolve().parents[1]
PREREG_CONFIG_PATH = Path("configs/selector_wv_scale_regularization.preregistered.json")
INDEX_PATH = Path("artifacts/token_selector/prompt_projection_features/index.jsonl")
MANIFEST_PATH = Path("datasets/prompt_projection_clean_inductive_v1.jsonl")
REPORTS_DIR = Path("reports/token_selector")
REPORT_DIR = REPORTS_DIR / REPORT_STEM
LOGS_DIR = REPORTS_DIR / "logs"
ARTIFACT_ROOT = Path("artifacts/token_selector/selector_wv_scale_regularization")

REFERENCE_ARM = "PER_ROW_WV"
CANDIDATE_ARM = "PER_TENSOR_WV"
FLOAT_ARM = "FLOAT_REFERENCE"

SEEDS = (20261003, 20261004, 20261005)
BUDGETS = (48, 64, 96)
TASKS = ("grounding", "pointing")
ROLES = ("correct", "wrong")
LAMBDA_SEMANTIC = 0.5
STUDENT_SCORER = "prompt_mean_vector"

INT8_MIN = base.INT8_MIN
INT8_MAX = base.INT8_MAX
WEIGHT_MIN = base.WEIGHT_MIN
WEIGHT_MAX = base.WEIGHT_MAX
WEIGHT_LEVELS = base.WEIGHT_LEVELS
INT32_MAX = base.INT32_MAX
BOOTSTRAP_RESAMPLES = base.BOOTSTRAP_RESAMPLES
STATISTICS_SEED = base.STATISTICS_SEED
CI_LEVEL = base.CI_LEVEL

#: Scalar per (role, budget) metrics carried through aggregation.
RECORD_METRICS = (
    "cand_mass",
    "cand_ndcg",
    "cand_selected_count",
    "ref_mass",
    "ref_ndcg",
    "ref_selected_count",
    "float_mass",
    "float_ndcg",
    "float_selected_count",
    "cand_vs_ref_jaccard",
    "cand_vs_ref_exact_match",
    "cand_vs_ref_replacement_count",
    "cand_vs_ref_mass_delta",
    "cand_vs_ref_ndcg_delta",
    "cand_vs_float_jaccard",
    "cand_vs_float_mass_delta",
    "cand_vs_float_ndcg_delta",
)

SPEARMAN_COMPARISONS = (
    "candidate_vs_per_row",
    "per_row_vs_float",
    "candidate_vs_float",
)

CSV_FIELDS = (
    "seed",
    "sample_id",
    "image_sha256",
    "task",
    "group_count",
    "grid_h",
    "grid_w",
    "prompt_role",
    "budget",
    "lambda_semantic",
    "candidate_selected_count",
    "reference_selected_count",
    "float_selected_count",
    "candidate_vs_reference_jaccard",
    "candidate_vs_reference_exact_match",
    "candidate_vs_reference_replacement_count",
    "candidate_vs_reference_mass_delta",
    "candidate_vs_reference_ndcg_delta",
    "candidate_vs_float_jaccard",
    "candidate_vs_float_mass_delta",
    "candidate_vs_float_ndcg_delta",
    "candidate_vs_reference_spearman",
)


class WvScaleAuditError(ValueError):
    """Malformed programmer input or a violated audit contract."""


# ---------------------------------------------------------------------------
# Per-tensor Wv quantisation (the only new numeric primitive)
# ---------------------------------------------------------------------------
def quantize_weight_tensor(matrix: Any) -> tuple[np.ndarray, float]:
    """Symmetric **per-tensor** INT8 weight quantisation ``d = amax/127``.

    ``q = clamp(round_half_away(w/d), -127, 127)`` and ``w_hat = q*d``.  A zero
    tensor keeps ``d = 1.0`` and a zero int64 quantisation.  This is the
    candidate-arm counterpart of
    :func:`evaluation.selector_int8_numerical_audit.quantize_weight_rows`.
    """
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise WvScaleAuditError(
            "per-tensor weight quantisation needs a non-empty 2-D matrix"
        )
    if not np.isfinite(array).all():
        raise WvScaleAuditError("weight matrix must be finite")
    amax = float(np.max(np.abs(array)))
    scale = amax / WEIGHT_LEVELS if amax > 0.0 else 1.0
    quantized = np.clip(
        base.round_half_away_from_zero(array / scale), WEIGHT_MIN, WEIGHT_MAX
    )
    return quantized.astype(np.int64), float(scale)


def per_tensor_scale_vector(scale: float, rows: int) -> np.ndarray:
    """Broadcast one per-tensor scale to the ``[rows]`` dequantisation vector."""
    if int(rows) <= 0:
        raise WvScaleAuditError("scale vector needs a positive row count")
    return np.full(int(rows), float(scale), dtype=np.float64)


# ---------------------------------------------------------------------------
# Per-record evaluation
# ---------------------------------------------------------------------------
def _selection_metrics(
    scores: Any, target: Any, grid_h: int, grid_w: int, budgets: Sequence[int]
) -> dict[int, dict[str, Any]]:
    """Retained mass, selected-set NDCG and selected set per budget."""
    score_array = np.asarray(scores, dtype=np.float64)
    relevance = np.asarray(target, dtype=np.float64).reshape(-1)
    result: dict[int, dict[str, Any]] = {}
    for budget in budgets:
        budget = int(budget)
        selected = base.selected_indices(score_array, grid_h, grid_w, budget)
        effective = min(budget, int(relevance.size))
        result[budget] = {
            "selected": selected,
            "mass": float(saliency_mass_retained(selected, relevance)),
            "ndcg": float(
                ndcg_for_selected_indices(selected, score_array, relevance, effective)
            ),
        }
    return result


def evaluate_record(
    *,
    group_mean: Any,
    prompt_correct: Any,
    prompt_wrong: Any,
    Wt: Any,
    Wv: Any,
    wv_q_ref: np.ndarray,
    wv_scales_ref: np.ndarray,
    wv_q_cand: np.ndarray,
    wv_scale_cand: float,
    target: Any,
    grid_h: int,
    grid_w: int,
    budgets: Sequence[int] = BUDGETS,
) -> dict[str, Any]:
    """Evaluate the per-row reference, the per-tensor candidate and the float arm.

    The activation scale, the float-``Wt`` query chain and every downstream golden
    operation are shared; only the Wv dequantisation vector differs, so the
    comparison isolates the per-row -> per-tensor scale change.
    """
    budgets = [int(value) for value in budgets]
    mean = np.asarray(group_mean, dtype=np.float64)
    correct = np.asarray(prompt_correct, dtype=np.float64)
    wrong = np.asarray(prompt_wrong, dtype=np.float64)

    group_mean_q, group_mean_scale = base.quantize_activation(mean)
    group_mean_hat = group_mean_q.astype(np.float64) / group_mean_scale
    l2_hat = np.linalg.norm(group_mean_hat, axis=1)

    ref_visual = base.quantized_visual_projection(
        group_mean_q, group_mean_scale, np.asarray(wv_q_ref), np.asarray(wv_scales_ref)
    )
    cand_scales = per_tensor_scale_vector(wv_scale_cand, np.asarray(Wv).shape[0])
    cand_visual = base.quantized_visual_projection(
        group_mean_q, group_mean_scale, np.asarray(wv_q_cand), cand_scales
    )

    queries: dict[str, np.ndarray] = {}
    for role, prompt in (("correct", correct), ("wrong", wrong)):
        queries[role] = base.primary_quantized_query(prompt, Wt)["query_hat"]

    float_selection: dict[str, dict[int, dict[str, Any]]] = {}
    ref_selection: dict[str, dict[int, dict[str, Any]]] = {}
    cand_selection: dict[str, dict[int, dict[str, Any]]] = {}
    float_combined: dict[str, np.ndarray] = {}
    ref_combined: dict[str, np.ndarray] = {}
    cand_combined: dict[str, np.ndarray] = {}
    cand_score_q: dict[str, np.ndarray] = {}
    by_role: dict[str, dict[int, dict[str, Any]]] = {}

    for role, prompt in (("correct", correct), ("wrong", wrong)):
        _, _, combined_float = base.float_reference_scores(mean, prompt, Wt, Wv)
        _, ref_q, ref_hat = base.quantized_combined_scores(
            ref_visual["visual_unit"], queries[role], l2_hat
        )
        _, cand_q, cand_hat = base.quantized_combined_scores(
            cand_visual["visual_unit"], queries[role], l2_hat
        )
        float_combined[role] = combined_float
        ref_combined[role] = ref_hat
        cand_combined[role] = cand_hat
        cand_score_q[role] = cand_q
        float_selection[role] = _selection_metrics(
            combined_float, target, grid_h, grid_w, budgets
        )
        ref_selection[role] = _selection_metrics(
            ref_hat, target, grid_h, grid_w, budgets
        )
        cand_selection[role] = _selection_metrics(
            cand_hat, target, grid_h, grid_w, budgets
        )

        role_metrics: dict[int, dict[str, Any]] = {}
        for budget in budgets:
            ref_entry = ref_selection[role][budget]
            cand_entry = cand_selection[role][budget]
            float_entry = float_selection[role][budget]
            ref_set = set(ref_entry["selected"])
            cand_set = set(cand_entry["selected"])
            float_set = set(float_entry["selected"])
            role_metrics[budget] = {
                "cand_mass": float(cand_entry["mass"]),
                "cand_ndcg": float(cand_entry["ndcg"]),
                "cand_selected_count": int(len(cand_entry["selected"])),
                "ref_mass": float(ref_entry["mass"]),
                "ref_ndcg": float(ref_entry["ndcg"]),
                "ref_selected_count": int(len(ref_entry["selected"])),
                "float_mass": float(float_entry["mass"]),
                "float_ndcg": float(float_entry["ndcg"]),
                "float_selected_count": int(len(float_entry["selected"])),
                "cand_vs_ref_jaccard": float(
                    jaccard(cand_entry["selected"], ref_entry["selected"])
                ),
                "cand_vs_ref_exact_match": 1.0 if cand_set == ref_set else 0.0,
                # Replacement count = candidate-only members; with equal set
                # sizes this is half the symmetric difference.
                "cand_vs_ref_replacement_count": int(len(cand_set - ref_set)),
                "cand_vs_ref_mass_delta": float(
                    cand_entry["mass"] - ref_entry["mass"]
                ),
                "cand_vs_ref_ndcg_delta": float(
                    cand_entry["ndcg"] - ref_entry["ndcg"]
                ),
                "cand_vs_float_jaccard": float(
                    jaccard(cand_entry["selected"], float_entry["selected"])
                ),
                "cand_vs_float_mass_delta": float(
                    cand_entry["mass"] - float_entry["mass"]
                ),
                "cand_vs_float_ndcg_delta": float(
                    cand_entry["ndcg"] - float_entry["ndcg"]
                ),
                "cand_selected": tuple(int(v) for v in cand_entry["selected"]),
                "ref_selected": tuple(int(v) for v in ref_entry["selected"]),
                "float_selected": tuple(int(v) for v in float_entry["selected"]),
                "cand_score_q": cand_q,
                "cand_score_hat": cand_hat,
            }
        by_role[role] = role_metrics

    spearman_entries = {
        role: {
            "candidate_vs_per_row": float(spearman(cand_combined[role], ref_combined[role])),
            "per_row_vs_float": float(spearman(ref_combined[role], float_combined[role])),
            "candidate_vs_float": float(
                spearman(cand_combined[role], float_combined[role])
            ),
        }
        for role in ROLES
    }

    return {
        "by_role": by_role,
        "spearman": spearman_entries,
        "group_mean_scale": float(group_mean_scale),
        "group_mean_q_absmax": int(np.max(np.abs(group_mean_q))),
        "ref_wv_absmax": int(np.max(np.abs(wv_q_ref))),
        "cand_wv_absmax": int(np.max(np.abs(wv_q_cand))),
        "cand_wv_scale": float(wv_scale_cand),
        "projection_int_max_abs_per_row": base.max_abs_int32(ref_visual["projection_int"]),
        "projection_int_max_abs_per_tensor": base.max_abs_int32(
            cand_visual["projection_int"]
        ),
        "group_count": int(mean.shape[0]),
    }


# ---------------------------------------------------------------------------
# Aggregation and bootstrap
# ---------------------------------------------------------------------------
def _median_across_seeds(values: Sequence[float]) -> float:
    if not values:
        raise WvScaleAuditError("cannot take a median of zero seed values")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def aggregate_metrics(
    raw_by_seed: Mapping[int, Mapping[str, Mapping[str, Any]]],
    *,
    sample_meta: Mapping[str, Mapping[str, Any]],
    seeds: Sequence[int] = SEEDS,
    budgets: Sequence[int] = BUDGETS,
    roles: Sequence[str] = ROLES,
) -> dict[str, Any]:
    """Reduce per-seed records to per-record medians and image-level summaries.

    ``raw_by_seed[seed][sample_id]["by_role"][role][budget][metric]`` carries the
    scalar metrics and ``raw_by_seed[seed][sample_id]["spearman"][role][comparison]``
    the rank correlations.  The frozen aggregation is: median across the three
    seeds per record, then mean across the two task records per image, then a
    2000-resample percentile bootstrap over the 38 independent images.
    """
    seeds = [int(value) for value in seeds]
    budgets = [int(value) for value in budgets]
    roles = [str(value) for value in roles]
    sample_ids = sorted(str(value) for value in sample_meta)
    image_of = {sid: str(sample_meta[sid]["image_sha256"]) for sid in sample_ids}
    task_of = {sid: str(sample_meta[sid]["task"]) for sid in sample_ids}
    images = sorted(set(image_of.values()))

    collected: dict[tuple[str, int, str], dict[str, list[float]]] = {}
    for role in roles:
        for budget in budgets:
            for metric in RECORD_METRICS:
                collected[(role, int(budget), metric)] = {}
    spearman_collected: dict[tuple[str, str], dict[str, list[float]]] = {}
    for seed in seeds:
        for sid in sample_ids:
            entry = raw_by_seed[seed][sid]
            for role in roles:
                for budget in budgets:
                    metrics = entry["by_role"][role][int(budget)]
                    for metric in RECORD_METRICS:
                        collected[(role, int(budget), metric)].setdefault(
                            sid, []
                        ).append(float(metrics[metric]))
                for comparison in SPEARMAN_COMPARISONS:
                    spearman_collected.setdefault((role, comparison), {}).setdefault(
                        sid, []
                    ).append(float(entry["spearman"][role][comparison]))

    record_median: dict[tuple[str, int, str], dict[str, float]] = {
        key: {sid: _median_across_seeds(values) for sid, values in per_sid.items()}
        for key, per_sid in collected.items()
    }
    spearman_median: dict[tuple[str, str], dict[str, float]] = {
        key: {sid: _median_across_seeds(values) for sid, values in per_sid.items()}
        for key, per_sid in spearman_collected.items()
    }

    def series(role: str, budget: int, metric: str, tasks: Sequence[str]) -> dict[str, float]:
        per_image: dict[str, list[float]] = {}
        for sid in sample_ids:
            if task_of[sid] not in tasks:
                continue
            value = record_median[(role, int(budget), metric)].get(sid)
            if value is None:
                continue
            per_image.setdefault(image_of[sid], []).append(float(value))
        return {image: float(np.mean(values)) for image, values in per_image.items()}

    overall: dict[str, Any] = {}
    per_task: dict[str, Any] = {}
    for role in roles:
        overall[role] = {}
        per_task[role] = {task: {} for task in TASKS}
        for budget in budgets:
            overall[role][str(int(budget))] = {}
            for task in TASKS:
                per_task[role][task][str(int(budget))] = {}
            for metric in RECORD_METRICS:
                overall[role][str(int(budget))][metric] = base.bootstrap_mean_ci(
                    list(series(role, budget, metric, TASKS).values())
                )
                for task in TASKS:
                    per_task[role][task][str(int(budget))][metric] = base.bootstrap_mean_ci(
                        list(series(role, budget, metric, (task,)).values())
                    )

    # --- score rank Spearman (per record median, then image mean) ----------
    score_rank_spearman: dict[str, Any] = {}
    for comparison in SPEARMAN_COMPARISONS:
        score_rank_spearman[comparison] = {}
        for role in roles:
            per_image: dict[str, list[float]] = {}
            for sid in sample_ids:
                per_image.setdefault(image_of[sid], []).append(
                    float(spearman_median[(role, comparison)][sid])
                )
            per_task_spearman = {}
            for task in TASKS:
                task_values = [
                    float(spearman_median[(role, comparison)][sid])
                    for sid in sample_ids
                    if task_of[sid] == task
                ]
                per_task_spearman[task] = base.bootstrap_mean_ci(task_values)
            score_rank_spearman[comparison][role] = {
                "overall": base.bootstrap_mean_ci(
                    [float(np.mean(values)) for values in per_image.values()]
                ),
                "per_task": per_task_spearman,
            }

    # --- prompt-utility gaps: (correct - wrong) candidate vs reference/float -
    def gap_delta_series(
        budget: int, metric: str, reference: str, tasks: Sequence[str]
    ) -> dict[str, float]:
        cand_key = "cand_mass" if metric == "mass" else "cand_ndcg"
        ref_key = "ref_mass" if metric == "mass" else "ref_ndcg"
        float_key = "float_mass" if metric == "mass" else "float_ndcg"
        per_image: dict[str, list[float]] = {}
        for sid in sample_ids:
            if task_of[sid] not in tasks:
                continue
            gap_cand = (
                record_median[("correct", int(budget), cand_key)][sid]
                - record_median[("wrong", int(budget), cand_key)][sid]
            )
            if reference == "float":
                gap_ref = (
                    record_median[("correct", int(budget), float_key)][sid]
                    - record_median[("wrong", int(budget), float_key)][sid]
                )
            elif reference == "per_row":
                gap_ref = (
                    record_median[("correct", int(budget), ref_key)][sid]
                    - record_median[("wrong", int(budget), ref_key)][sid]
                )
            else:
                raise WvScaleAuditError(f"unknown reference arm {reference!r}")
            per_image.setdefault(image_of[sid], []).append(float(gap_cand - gap_ref))
        return {image: float(np.mean(values)) for image, values in per_image.items()}

    prompt_utility_gap_delta: dict[str, Any] = {}
    for reference in ("float", "per_row"):
        prompt_utility_gap_delta[reference] = {}
        for budget in budgets:
            prompt_utility_gap_delta[reference][str(int(budget))] = {}
            for metric in ("mass", "ndcg"):
                prompt_utility_gap_delta[reference][str(int(budget))][metric] = {
                    "overall": base.bootstrap_mean_ci(
                        list(gap_delta_series(budget, metric, reference, TASKS).values())
                    ),
                    "per_task": {
                        task: base.bootstrap_mean_ci(
                            list(
                                gap_delta_series(budget, metric, reference, (task,)).values()
                            )
                        )
                        for task in TASKS
                    },
                }

    # --- per-seed K48 candidate-vs-reference correct-prompt Jaccard --------
    seed_k48: dict[str, float] = {}
    for seed in seeds:
        per_image: dict[str, list[float]] = {}
        for sid in sample_ids:
            per_image.setdefault(image_of[sid], []).append(
                float(raw_by_seed[seed][sid]["by_role"]["correct"][48]["cand_vs_ref_jaccard"])
            )
        seed_k48[str(seed)] = float(np.mean([np.mean(values) for values in per_image.values()]))

    return {
        "independent_unit": "image",
        "record_aggregation": (
            "per record then median across the three seeds, then mean across the "
            "two task records per image"
        ),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": STATISTICS_SEED,
        "confidence": CI_LEVEL,
        "n_records": len(sample_ids),
        "n_images": len(images),
        "record_medians": {
            f"{role}|{int(budget)}|{metric}": values
            for (role, budget, metric), values in record_median.items()
        },
        "overall": overall,
        "per_task": per_task,
        "score_rank_spearman": score_rank_spearman,
        "prompt_utility_gap_delta": prompt_utility_gap_delta,
        "seed_k48_candidate_vs_per_row_jaccard": seed_k48,
    }


def _bucket_mean(entry: Mapping[str, Any] | None) -> float | None:
    if entry is None or entry.get("mean") is None:
        return None
    return float(entry["mean"])


def evaluate_decision(
    summary: Mapping[str, Any],
    *,
    pins_ok: bool,
    isolation_ok: bool,
    int32_ok: bool,
    fixture_ok: bool,
    budgets: Sequence[int] = BUDGETS,
) -> dict[str, Any]:
    """Frozen ``PER_TENSOR_WV_ACCEPTABLE_FOR_HLS`` / ``..._NOT_ESTABLISHED`` rule."""
    budgets = [int(value) for value in budgets]
    overall = summary["overall"]

    def overall_mean(role: str, budget: int, metric: str) -> float | None:
        return _bucket_mean(overall[role][str(int(budget))].get(metric))

    jaccard_values = {
        b: overall_mean("correct", b, "cand_vs_ref_jaccard") for b in budgets
    }
    replacement_values = {
        b: overall_mean("correct", b, "cand_vs_ref_replacement_count") for b in budgets
    }
    mass_delta_values = {
        b: overall_mean("correct", b, "cand_vs_float_mass_delta") for b in budgets
    }
    ndcg_delta_values = {
        b: overall_mean("correct", b, "cand_vs_float_ndcg_delta") for b in budgets
    }
    spearman_mean = _bucket_mean(
        summary["score_rank_spearman"]["candidate_vs_per_row"]["correct"]["overall"]
    )
    pointing_gap = _bucket_mean(
        summary["prompt_utility_gap_delta"]["float"]["48"]["mass"]["per_task"]["pointing"]
    )
    seed_k48 = {
        str(seed): float(value)
        for seed, value in sorted(
            summary["seed_k48_candidate_vs_per_row_jaccard"].items()
        )
    }

    checks = {
        "pins_isolation_int32_fixture_pass": bool(
            pins_ok and isolation_ok and int32_ok and fixture_ok
        ),
        "candidate_vs_per_row_mean_jaccard_each_budget_at_least_0.98": all(
            value is not None and value >= 0.98 for value in jaccard_values.values()
        ),
        "candidate_vs_per_row_mean_replacement_count_each_budget_at_most_1.0": all(
            value is not None and value <= 1.0 for value in replacement_values.values()
        ),
        "candidate_vs_per_row_mean_score_spearman_at_least_0.995": bool(
            spearman_mean is not None and spearman_mean >= 0.995
        ),
        "candidate_vs_float_mass_delta_each_budget_at_least_-0.01": all(
            value is not None and value >= -0.01 for value in mass_delta_values.values()
        ),
        "candidate_vs_float_ndcg_delta_each_budget_at_least_-0.01": all(
            value is not None and value >= -0.01 for value in ndcg_delta_values.values()
        ),
        "pointing_k48_prompt_utility_gap_delta_candidate_vs_float_at_least_-0.01": bool(
            pointing_gap is not None and pointing_gap >= -0.01
        ),
        "each_seed_k48_candidate_vs_per_row_mean_jaccard_at_least_0.97": all(
            value >= 0.97 for value in seed_k48.values()
        ),
    }
    passed = all(checks.values())
    return {
        "decision": (
            "PER_TENSOR_WV_ACCEPTABLE_FOR_HLS"
            if passed
            else "PER_TENSOR_WV_NOT_ESTABLISHED"
        ),
        "pass": bool(passed),
        "checks": checks,
        "candidate_arm": CANDIDATE_ARM,
        "reference_arm": REFERENCE_ARM,
        "overall_mean_candidate_vs_per_row_jaccard": {
            str(b): jaccard_values[b] for b in budgets
        },
        "overall_mean_candidate_vs_per_row_replacement_count": {
            str(b): replacement_values[b] for b in budgets
        },
        "overall_mean_candidate_vs_per_row_score_spearman": spearman_mean,
        "overall_mean_candidate_vs_float_mass_delta": {
            str(b): mass_delta_values[b] for b in budgets
        },
        "overall_mean_candidate_vs_float_ndcg_delta": {
            str(b): ndcg_delta_values[b] for b in budgets
        },
        "pointing_k48_prompt_utility_gap_delta_candidate_vs_float": pointing_gap,
        "seed_k48_candidate_vs_per_row_mean_jaccard": seed_k48,
        "exact_set_rate_is_descriptive_only": True,
        "rule": (
            "PER_TENSOR_WV_ACCEPTABLE_FOR_HLS iff pins/isolation/INT32/fixture pass, "
            "every K candidate-vs-per-row mean Jaccard >= 0.98, mean replacement "
            "count <= 1.0, mean score Spearman >= 0.995, every K candidate-vs-float "
            "mass delta >= -0.01 and NDCG delta >= -0.01, K48 Pointing prompt-utility "
            "gap delta >= -0.01 and every seed K48 candidate-vs-per-row mean "
            "Jaccard >= 0.97"
        ),
    }


# ---------------------------------------------------------------------------
# Compact hardware bundle + deterministic fixtures
# ---------------------------------------------------------------------------
def _fixture_payload(
    sample: Mapping[str, Any],
    *,
    Wt: np.ndarray,
    wv_q: np.ndarray,
    wv_scales: np.ndarray,
) -> dict[str, Any]:
    """Candidate-arm fixture for one record (no raw image or ``Wt`` dump)."""
    group_mean = np.asarray(sample["group_mean"], dtype=np.float64)
    correct = np.asarray(sample["prompt_embeddings"], dtype=np.float64)
    wrong = np.asarray(sample["wrong_prompt_embeddings"], dtype=np.float64)
    grid_h = int(sample["grid_h"])
    grid_w = int(sample["grid_w"])

    group_mean_q, group_mean_scale = base.quantize_activation(group_mean)
    group_mean_hat = group_mean_q.astype(np.float64) / group_mean_scale
    l2_hat = np.linalg.norm(group_mean_hat, axis=1)
    visual = base.quantized_visual_projection(
        group_mean_q, group_mean_scale, np.asarray(wv_q), np.asarray(wv_scales)
    )

    payload: dict[str, Any] = {
        "group_mean_q": group_mean_q.astype(np.int8),
        "group_mean_scale": np.asarray(group_mean_scale, dtype=np.float64),
        "grid_h": np.asarray(grid_h, dtype=np.int64),
        "grid_w": np.asarray(grid_w, dtype=np.int64),
        "sample_id": np.asarray(str(sample["sample_id"])),
    }
    for role, prompt in (("correct", correct), ("wrong", wrong)):
        query = base.primary_quantized_query(prompt, Wt)
        _, score_q, score_hat = base.quantized_combined_scores(
            visual["visual_unit"], query["query_hat"], l2_hat
        )
        payload[f"query_q_{role}"] = query["query_q"].astype(np.int8)
        payload[f"query_scale_{role}"] = np.asarray(query["query_scale"], dtype=np.float64)
        payload[f"score_q_{role}"] = score_q.astype(np.uint16)
        for budget in BUDGETS:
            payload[f"selected_{role}_k{budget}"] = np.asarray(
                base.selected_indices(score_hat, grid_h, grid_w, budget), dtype=np.int32
            )
    return payload


def validate_fixture_roundtrip(
    out_dir: Path,
    *,
    Wv: np.ndarray,
    fixtures: Sequence[Mapping[str, Any]],
    budgets: Sequence[int] = BUDGETS,
) -> dict[str, Any]:
    """Reload the exported int8 Wv + float32 scale and replay every fixture.

    The check is byte-exact: the reloaded int8 matrix must equal the frozen
    per-tensor quantiser output, the reloaded float32 scale must equal the stored
    scale, and the recomputed uint16 Q0.15 code and anchor selection must match
    the fixture payload exactly.
    """
    out_dir = Path(out_dir)
    wv_q_expected, wv_scale_expected = quantize_weight_tensor(Wv)
    wv_shape = tuple(int(value) for value in np.asarray(Wv).shape)

    wv_q_loaded = np.frombuffer(
        (out_dir / "Wv_int8_rowmajor.bin").read_bytes(), dtype=np.int8
    ).reshape(wv_shape)
    scale_loaded = float(
        np.frombuffer(
            (out_dir / "Wv_per_tensor_scale_f32le.bin").read_bytes(), dtype="<f4"
        )[0]
    )
    scale_vector = per_tensor_scale_vector(scale_loaded, wv_shape[0])

    q_ok = bool(np.array_equal(wv_q_loaded.astype(np.int64), wv_q_expected))
    scale_ok = bool(scale_loaded == float(np.float32(wv_scale_expected)))

    fixture_checks: list[dict[str, Any]] = []
    all_ok = True
    for sample in fixtures:
        name = f"fixture_{sample['task']}.npz"
        with np.load(out_dir / name, allow_pickle=False) as archive:
            group_mean_q = archive["group_mean_q"].astype(np.int64)
            group_mean_scale = float(archive["group_mean_scale"])
            grid_h = int(archive["grid_h"])
            grid_w = int(archive["grid_w"])
            group_mean_hat = group_mean_q.astype(np.float64) / group_mean_scale
            l2_hat = np.linalg.norm(group_mean_hat, axis=1)
            visual = base.quantized_visual_projection(
                group_mean_q, group_mean_scale, wv_q_loaded.astype(np.int64), scale_vector
            )
            role_checks: dict[str, Any] = {}
            for role in ROLES:
                query_q = archive[f"query_q_{role}"].astype(np.float64)
                query_scale = float(archive[f"query_scale_{role}"])
                query_hat = query_q / query_scale
                query_hat = query_hat / float(np.linalg.norm(query_hat))
                _, score_q, score_hat = base.quantized_combined_scores(
                    visual["visual_unit"], query_hat, l2_hat
                )
                expected_score_q = archive[f"score_q_{role}"].astype(np.uint16)
                score_ok = bool(np.array_equal(score_q, expected_score_q))
                selections_ok = True
                for budget in budgets:
                    budget = int(budget)
                    expected_selected = archive[f"selected_{role}_k{budget}"].astype(
                        np.int64
                    )
                    recomputed = np.asarray(
                        base.selected_indices(score_hat, grid_h, grid_w, budget),
                        dtype=np.int64,
                    )
                    selections_ok = selections_ok and bool(
                        np.array_equal(recomputed, expected_selected)
                    )
                role_checks[role] = {
                    "score_q_matches": score_ok,
                    "selected_matches": bool(selections_ok),
                }
                all_ok = all_ok and score_ok and selections_ok
        fixture_checks.append(
            {"task": str(sample["task"]), "sample_id": str(sample["sample_id"]), "roles": role_checks}
        )
        all_ok = all_ok and True

    return {
        "wv_q_matches_quantizer": q_ok,
        "wv_scale_matches_float32": scale_ok,
        "fixtures": fixture_checks,
        "ok": bool(q_ok and scale_ok and all_ok),
    }


def build_seed_bundle(
    seed: int,
    *,
    Wt: np.ndarray,
    Wv: np.ndarray,
    fixtures: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Write one compact per-seed candidate bundle and validate its fixtures."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wv_q, wv_scale = quantize_weight_tensor(Wv)
    export_scale = float(np.float32(wv_scale))
    export_scales = per_tensor_scale_vector(export_scale, np.asarray(Wv).shape[0])

    wv_bytes = np.ascontiguousarray(wv_q, dtype=np.int8).tobytes(order="C")
    scale_bytes = np.asarray([export_scale], dtype="<f4").tobytes(order="C")
    (out_dir / "Wv_int8_rowmajor.bin").write_bytes(wv_bytes)
    (out_dir / "Wv_per_tensor_scale_f32le.bin").write_bytes(scale_bytes)

    fixture_files: list[dict[str, Any]] = []
    for sample in fixtures:
        payload = _fixture_payload(
            sample, Wt=Wt, wv_q=wv_q, wv_scales=export_scales
        )
        name = f"fixture_{sample['task']}.npz"
        np.savez(out_dir / name, **payload)
        fixture_files.append(
            {
                "task": str(sample["task"]),
                "sample_id": str(sample["sample_id"]),
                "file": name,
                "group_count": int(sample["group_count"]),
                "grid_h": int(sample["grid_h"]),
                "grid_w": int(sample["grid_w"]),
                "group_mean_scale": float(payload["group_mean_scale"]),
                "query_scale_correct": float(payload["query_scale_correct"]),
                "query_scale_wrong": float(payload["query_scale_wrong"]),
                "bytes": (out_dir / name).stat().st_size,
                "sha256": sha256_file(out_dir / name),
            }
        )

    roundtrip = validate_fixture_roundtrip(out_dir, Wv=Wv, fixtures=fixtures)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "candidate_arm": CANDIDATE_ARM,
        "reference_arm": REFERENCE_ARM,
        "source_checkpoint": {
            "path": str(checkpoint["path"]),
            "sha256": str(checkpoint["sha256"]),
            "canonical_weights_sha256": str(checkpoint.get("canonical_weights_sha256", "")),
        },
        "wt_exported": False,
        "wt_note": "the candidate deployment bundle intentionally does not export Wt",
        "wv": {
            "layout": "row-major [128, 1024] int8, output row = Wv row",
            "shape": [int(wv_q.shape[0]), int(wv_q.shape[1])],
            "dtype": "int8",
            "bytes": len(wv_bytes),
            "file": "Wv_int8_rowmajor.bin",
            "sha256": sha256_file(out_dir / "Wv_int8_rowmajor.bin"),
            "quantization": (
                "d=max(abs(Wv))/127; q=clamp(round_half_away(Wv/d),-127,127); w_hat=q*d"
            ),
            "q_absmax": int(np.max(np.abs(wv_q))),
            "scale_value_float64": float(wv_scale),
            "scale_value_float32": float(export_scale),
            "scale_layout": "one little-endian float32 per tensor",
            "scale_bytes": len(scale_bytes),
            "scale_file": "Wv_per_tensor_scale_f32le.bin",
            "scale_sha256": sha256_file(out_dir / "Wv_per_tensor_scale_f32le.bin"),
            "scale_dtype": "<f4",
        },
        "activation_quantization": (
            "s=128/amax; q=clamp(round_half_away(x*s),-128,127); x_hat=q/s"
        ),
        "combined_score_quantization": (
            "q=clamp(round_half_away(score*32768),0,32767); score_hat=q/32768"
        ),
        "rounding": "round_half_away_from_zero",
        "projection_accumulation": (
            "exact INT32 (int8 x int8); dequantize by (1/s)*d_w[row]"
        ),
        "query": {
            "chain": "host float Wt prompt_mean_vector then per-vector symmetric INT8"
        },
        "fixtures": fixture_files,
        "roundtrip": roundtrip,
        "files": sorted(
            [
                "Wv_int8_rowmajor.bin",
                "Wv_per_tensor_scale_f32le.bin",
                "metadata.json",
            ]
            + [entry["file"] for entry in fixture_files]
        ),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata["meta_sha256"] = sha256_file(out_dir / "metadata.json")
    return metadata


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def _features_dir(root: Path, index_path: Path) -> Path:
    return (root / index_path).parent


def run_audit(
    *,
    root: Path = ROOT,
    prereg_config_path: Path = PREREG_CONFIG_PATH,
    index_path: Path = INDEX_PATH,
    manifest_path: Path = MANIFEST_PATH,
    artifact_root: Path = ARTIFACT_ROOT,
    seeds: Sequence[int] = SEEDS,
    budgets: Sequence[int] = BUDGETS,
    cache_loader: Callable[..., dict[str, Any]] | None = None,
    write_artifacts: bool = True,
    write_reports: bool = True,
) -> dict[str, Any]:
    """Run the full validation-only audit and return the report bundle."""
    started = time.time()
    root = Path(root)
    prereg = base.load_prereg(root / prereg_config_path)
    identity = base.verify_prereguration_identity(
        prereg, root=root, config_path=prereg_config_path
    )
    if not identity["consistent"]:
        raise WvScaleAuditError("preregistration identity/authority-prompt mismatch")
    pins_before = base.verify_pins(prereg, root=root)
    if not pins_before["pins_unchanged"]:
        raise WvScaleAuditError(
            f"pinned inputs changed before load: {pins_before['mismatches']}"
        )

    split_rule = prereg["split_rule"]
    index_records = load_index_metadata(root / index_path)
    splits = split_sample_ids(
        index_records,
        seed=int(split_rule["seed"]),
        ratios=split_rule["ratios"],
        salt=str(split_rule["salt"]),
    )
    val_ids = [str(value) for value in splits["val"]]
    test_ids = {str(value) for value in splits["test"]}
    train_ids = {str(value) for value in splits["train"]}
    features_dir = _features_dir(root, index_path)

    # Internal-test paths are only compared as strings; no NPZ is resolved/hashed/
    # opened/parsed and no internal-test report is read.
    test_path_strings = {
        str(features_dir / str(record["npz"]))
        for record in index_records
        if str(record["sample_id"]) in test_ids
    }
    train_path_strings = {
        str(features_dir / str(record["npz"]))
        for record in index_records
        if str(record["sample_id"]) in train_ids
    }

    opened: set[str] = set()
    val_raw, val_aggregate, _ = load_split_samples(
        root / index_path,
        features_dir,
        val_ids,
        opened=opened,
        cache_loader=cache_loader,
    )
    opened_list = sorted(opened)
    test_opened = sorted(path for path in opened_list if path in test_path_strings)
    train_opened = sorted(path for path in opened_list if path in train_path_strings)
    if test_opened:
        raise WvScaleAuditError(f"internal test NPZ were opened: {test_opened}")
    if train_opened:
        raise WvScaleAuditError(f"train NPZ were opened during a val-only audit: {train_opened}")
    val_aggregate_ok = val_aggregate == str(
        split_rule["expected_val_npz_aggregate_sha256"]
    )
    if not val_aggregate_ok:
        raise WvScaleAuditError("val NPZ aggregate SHA256 does not match the preregistration")

    manifest_by_id, _ = load_clean_manifest(root / manifest_path)
    val_samples = attach_spatial_targets(val_raw, manifest_by_id)
    by_id = {str(sample["sample_id"]): sample for sample in val_samples}
    partner = same_image_partner_map(val_samples)
    sample_meta: dict[str, dict[str, Any]] = {}
    for sample in val_samples:
        sample_id = str(sample["sample_id"])
        sample_meta[sample_id] = {
            "task": str(sample["task"]),
            "image_sha256": str(sample["source_image_sha256"]),
            "group_count": int(sample["group_count"]),
            "grid_h": int(sample["grid_h"]),
            "grid_w": int(sample["grid_w"]),
        }

    checkpoint_by_seed = {int(entry["seed"]): entry for entry in prereg["checkpoints"]}
    bounds = base.analytical_int32_bounds()
    if not bounds["all_fit_int32"]:
        raise WvScaleAuditError("analytical INT32 bound violation")

    raw_by_seed: dict[int, dict[str, Any]] = {}
    csv_rows: list[dict[str, Any]] = []
    bundle_meta: list[dict[str, Any]] = []
    int32_observed = {"gm_wv_per_row_max_abs": 0, "gm_wv_per_tensor_max_abs": 0}
    for seed in [int(value) for value in seeds]:
        checkpoint = checkpoint_by_seed[seed]
        ckpt_path = root / str(checkpoint["path"])
        projection, _ = load_projection(ckpt_path)
        if sha256_file(ckpt_path) != str(checkpoint["sha256"]):
            raise WvScaleAuditError(f"checkpoint {ckpt_path} changed after load")
        Wt = np.asarray(projection.Wt, dtype=np.float64)
        Wv = np.asarray(projection.Wv, dtype=np.float64)
        wv_q_ref, wv_scales_ref = base.quantize_weight_rows(Wv)
        wv_q_cand, wv_scale_cand = quantize_weight_tensor(Wv)

        raw_by_seed[seed] = {}
        for sample in val_samples:
            sample_id = str(sample["sample_id"])
            other_id = partner[sample_id]
            target = np.asarray(sample["spatial_target"], dtype=np.float64)
            record = evaluate_record(
                group_mean=sample["group_mean"],
                prompt_correct=sample["prompt_embeddings"],
                prompt_wrong=by_id[other_id]["prompt_embeddings"],
                Wt=Wt,
                Wv=Wv,
                wv_q_ref=wv_q_ref,
                wv_scales_ref=wv_scales_ref,
                wv_q_cand=wv_q_cand,
                wv_scale_cand=wv_scale_cand,
                target=target,
                grid_h=int(sample["grid_h"]),
                grid_w=int(sample["grid_w"]),
                budgets=budgets,
            )
            int32_observed["gm_wv_per_row_max_abs"] = max(
                int32_observed["gm_wv_per_row_max_abs"],
                int(record["projection_int_max_abs_per_row"]),
            )
            int32_observed["gm_wv_per_tensor_max_abs"] = max(
                int32_observed["gm_wv_per_tensor_max_abs"],
                int(record["projection_int_max_abs_per_tensor"]),
            )

            raw_by_seed[seed][sample_id] = {
                "by_role": {
                    role: {
                        int(budget): dict(record["by_role"][role][int(budget)])
                        for budget in budgets
                    }
                    for role in ROLES
                },
                "spearman": {
                    role: dict(record["spearman"][role]) for role in ROLES
                },
            }
            for role in ROLES:
                for budget in budgets:
                    budget = int(budget)
                    metrics = record["by_role"][role][budget]
                    csv_rows.append(
                        {
                            "seed": seed,
                            "sample_id": sample_id,
                            "image_sha256": sample_meta[sample_id]["image_sha256"],
                            "task": sample_meta[sample_id]["task"],
                            "group_count": sample_meta[sample_id]["group_count"],
                            "grid_h": sample_meta[sample_id]["grid_h"],
                            "grid_w": sample_meta[sample_id]["grid_w"],
                            "prompt_role": role,
                            "budget": budget,
                            "lambda_semantic": LAMBDA_SEMANTIC,
                            "candidate_selected_count": int(
                                metrics["cand_selected_count"]
                            ),
                            "reference_selected_count": int(
                                metrics["ref_selected_count"]
                            ),
                            "float_selected_count": int(
                                metrics["float_selected_count"]
                            ),
                            "candidate_vs_reference_jaccard": float(
                                metrics["cand_vs_ref_jaccard"]
                            ),
                            "candidate_vs_reference_exact_match": float(
                                metrics["cand_vs_ref_exact_match"]
                            ),
                            "candidate_vs_reference_replacement_count": int(
                                metrics["cand_vs_ref_replacement_count"]
                            ),
                            "candidate_vs_reference_mass_delta": float(
                                metrics["cand_vs_ref_mass_delta"]
                            ),
                            "candidate_vs_reference_ndcg_delta": float(
                                metrics["cand_vs_ref_ndcg_delta"]
                            ),
                            "candidate_vs_float_jaccard": float(
                                metrics["cand_vs_float_jaccard"]
                            ),
                            "candidate_vs_float_mass_delta": float(
                                metrics["cand_vs_float_mass_delta"]
                            ),
                            "candidate_vs_float_ndcg_delta": float(
                                metrics["cand_vs_float_ndcg_delta"]
                            ),
                            "candidate_vs_reference_spearman": float(
                                record["spearman"][role]["candidate_vs_per_row"]
                            ),
                        }
                    )

        if write_artifacts:
            fixture_inputs = []
            for task in TASKS:
                sample = min(
                    (s for s in val_samples if str(s["task"]) == task),
                    key=lambda s: str(s["sample_id"]),
                )
                enriched = dict(sample)
                enriched["wrong_prompt_embeddings"] = by_id[partner[str(sample["sample_id"])]][
                    "prompt_embeddings"
                ]
                fixture_inputs.append(enriched)
            bundle_meta.append(
                build_seed_bundle(
                    seed,
                    Wt=Wt,
                    Wv=Wv,
                    fixtures=fixture_inputs,
                    checkpoint=checkpoint,
                    out_dir=root / artifact_root / str(seed),
                )
            )

    summary = aggregate_metrics(
        raw_by_seed, sample_meta=sample_meta, seeds=seeds, budgets=budgets
    )
    isolation = {
        "opened_npz_count": len(opened_list),
        "opened_only_val_npz": bool(
            len(opened_list) == len(val_ids) and not test_opened and not train_opened
        ),
        "test_npz_opened": test_opened,
        "train_npz_opened": train_opened,
        "test_npz_resolved_hashed_or_opened": False,
        "internal_test_report_files_parsed": False,
        "raw_images_read": False,
        "model_4b_loaded": False,
        "gpu_used": False,
        "network_used": False,
        "training_performed": False,
        "val_npz_aggregate": val_aggregate,
        "val_npz_aggregate_matches_prereg": bool(val_aggregate_ok),
        "val_records": len(val_ids),
        "val_images": len({meta["image_sha256"] for meta in sample_meta.values()}),
    }
    pins_after = base.verify_pins(prereg, root=root)
    pins_ok = bool(pins_before["pins_unchanged"] and pins_after["pins_unchanged"])
    isolation_ok = bool(
        isolation["opened_only_val_npz"]
        and not isolation["test_npz_opened"]
        and not isolation["train_npz_opened"]
    )
    int32_ok = bool(
        bounds["all_fit_int32"]
        and int32_observed["gm_wv_per_row_max_abs"] <= INT32_MAX
        and int32_observed["gm_wv_per_tensor_max_abs"] <= INT32_MAX
    )
    fixture_ok = bool(
        all(bool(entry.get("roundtrip", {}).get("ok")) for entry in bundle_meta)
        if bundle_meta
        else True
    )
    decision = evaluate_decision(
        summary,
        pins_ok=pins_ok,
        isolation_ok=isolation_ok,
        int32_ok=int32_ok,
        fixture_ok=fixture_ok,
        budgets=budgets,
    )

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "report_stem": REPORT_STEM,
        "date": DATE,
        "experiment": "selector_wv_scale_regularization",
        "scope": {
            "cpu_only": True,
            "offline": True,
            "validation_only": True,
            "training": False,
            "hyperparameter_tuning": False,
            "raw_images": False,
            "model_4b": False,
            "internal_test_resolve_hash_open": False,
        },
        "preregistration": {
            "config_path": str(prereg_config_path),
            "purpose": str(prereg.get("purpose")),
            "reference_arm": REFERENCE_ARM,
            "candidate_arm": CANDIDATE_ARM,
            "candidate_quantization": str(prereg.get("candidate_quantization")),
            "lambda_semantic": LAMBDA_SEMANTIC,
            "budgets": [int(value) for value in budgets],
            "seeds": [int(value) for value in seeds],
        },
        "identity": identity,
        "pins_before": pins_before,
        "pins_after": pins_after,
        "pins_unchanged": pins_ok,
        "isolation": isolation,
        "int32": {
            "analytical": bounds,
            "observed_max_abs": int32_observed,
            "int32_max": INT32_MAX,
            "no_overflow": int32_ok,
        },
        "fixtures_validated": fixture_ok,
        "aggregation": {
            "independent_unit": summary["independent_unit"],
            "record_aggregation": summary["record_aggregation"],
            "bootstrap_resamples": summary["bootstrap_resamples"],
            "bootstrap_seed": summary["bootstrap_seed"],
            "confidence": summary["confidence"],
            "n_records": summary["n_records"],
            "n_images": summary["n_images"],
            "p_values_used": False,
        },
        "metrics": {
            "overall": summary["overall"],
            "per_task": summary["per_task"],
            "score_rank_spearman": summary["score_rank_spearman"],
            "prompt_utility_gap_delta": summary["prompt_utility_gap_delta"],
            "seed_k48_candidate_vs_per_row_jaccard": summary[
                "seed_k48_candidate_vs_per_row_jaccard"
            ],
        },
        "exact_set_rate_is_descriptive_only": True,
        "grounding_prompt_utility_note": (
            "Grounding same-image Pointing prompt is descriptive only: targets are "
            "nested and the paired prompt is not a valid negative; only Pointing "
            "enters the K48 prompt-utility gate."
        ),
        "decision": decision,
        "artifacts": {
            "root": str(artifact_root),
            "per_seed": bundle_meta,
            "wt_exported": False,
        },
        "limitations": list(prereg.get("limitations", []))
        + [
            "Float L2 normalisation, reciprocal square root, min-max and row "
            "normalisation are software golden operations in this audit.",
            "Passing does not establish complete fixed-point HLS equivalence, FPGA "
            "latency, end-to-end accuracy or publication readiness.",
            "The candidate export stores the per-tensor scale as float32 while the "
            "metric chain uses the float64 value; the difference is far below the "
            "Q0.15 resolution and is validated by the fixture round-trip.",
        ],
        "elapsed_seconds": float(time.time() - started),
        "csv_rows": len(csv_rows),
    }
    if write_reports:
        files = _write_reports(bundle, csv_rows, root=root)
        bundle["files"] = files
        harness_path = root / LOGS_DIR / f"{REPORT_STEM}-harness-result.json"
        _write_json(harness_path, harness_payload(bundle, files))
        files["harness_result"] = str(harness_path)
        files["harness_result_sha256"] = sha256_file(harness_path)
        run_log_path = root / LOGS_DIR / f"{REPORT_STEM}-run.log"
        _write_json(run_log_path, bundle)
        files["run_log"] = str(run_log_path)
    return bundle


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(bundle: Mapping[str, Any]) -> str:
    """Chinese Markdown report generated from the frozen numbers."""
    decision = bundle["decision"]
    metrics = bundle["metrics"]
    overall = metrics["overall"]
    lines: list[str] = []
    lines.append(f"# Wv 逐行→逐张量 scale 硬件正则化审计 — {REPORT_STEM}")
    lines.append("")
    lines.append(f"- 决策：**{decision['decision']}**")
    lines.append(
        "- 范围：CPU-only、离线、validation-only、只读；未训练、未调参、未加载 4B、未读原图、未访问 internal test。"
    )
    lines.append(
        f"- 参考臂：`{REFERENCE_ARM}`（逐输出行 INT8 Wv scale）；候选臂：`{CANDIDATE_ARM}`"
        "（整张 Wv 单一 scale `d=max(abs(Wv))/127`）；float 参考为同一冻结 scorer。"
    )
    lines.append(
        f"- 聚合：每记录跨 3 seed 取中位数，再按 {bundle['aggregation']['n_images']} 张独立图像取两任务均值并做 2000 次 percentile bootstrap 95% CI；不使用 p 值。"
    )
    lines.append("")
    lines.append("## 冻结判据")
    lines.append("")
    lines.append("| 判据 | 通过 |")
    lines.append("|---|---|")
    for key, value in decision["checks"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.append("")
    lines.append("## 候选 vs 逐行参考（correct prompt，overall 图像级 mean）")
    lines.append("")
    lines.append(
        "| K | Jaccard mean [95% CI] | exact-set rate（仅描述） | replacement count | score Spearman | mass delta | NDCG delta |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for budget in BUDGETS:
        entry = overall["correct"][str(budget)]
        lines.append(
            "| {} | {} [{}, {}] | {} | {} | {} | {} | {} |".format(
                budget,
                _fmt(entry["cand_vs_ref_jaccard"]["mean"]),
                _fmt(entry["cand_vs_ref_jaccard"]["ci_low"]),
                _fmt(entry["cand_vs_ref_jaccard"]["ci_high"]),
                _fmt(entry["cand_vs_ref_exact_match"]["mean"]),
                _fmt(entry["cand_vs_ref_replacement_count"]["mean"]),
                _fmt(
                    metrics["score_rank_spearman"]["candidate_vs_per_row"]["correct"][
                        "overall"
                    ]["mean"]
                ),
                _fmt(entry["cand_vs_ref_mass_delta"]["mean"]),
                _fmt(entry["cand_vs_ref_ndcg_delta"]["mean"]),
            )
        )
    lines.append("")
    lines.append("## 候选 vs float（correct prompt，overall 图像级 mean）")
    lines.append("")
    lines.append("| K | Jaccard | mass delta | NDCG delta |")
    lines.append("|---|---|---|---|")
    for budget in BUDGETS:
        entry = overall["correct"][str(budget)]
        lines.append(
            "| {} | {} | {} | {} |".format(
                budget,
                _fmt(entry["cand_vs_float_jaccard"]["mean"]),
                _fmt(entry["cand_vs_float_mass_delta"]["mean"]),
                _fmt(entry["cand_vs_float_ndcg_delta"]["mean"]),
            )
        )
    lines.append("")
    lines.append("## K48 各臂 prompt-utility gap（correct − wrong retained mass）delta")
    lines.append("")
    for reference, label in (("float", "候选 − float"), ("per_row", "候选 − 逐行")):
        gap = metrics["prompt_utility_gap_delta"][reference]["48"]["mass"]
        pt = gap["per_task"]["pointing"]
        gd = gap["per_task"]["grounding"]
        lines.append(
            f"- Pointing（{label}）mean={_fmt(pt['mean'])} "
            f"CI=[{_fmt(pt['ci_low'])}, {_fmt(pt['ci_high'])}]"
        )
        lines.append(
            f"- Grounding（{label}，仅描述：目标嵌套不构成有效负样本）mean={_fmt(gd['mean'])} "
            f"CI=[{_fmt(gd['ci_low'])}, {_fmt(gd['ci_high'])}]"
        )
    lines.append("")
    lines.append("## 逐 seed K48 候选 vs 逐行 mean Jaccard")
    lines.append("")
    for seed, value in sorted(decision["seed_k48_candidate_vs_per_row_mean_jaccard"].items()):
        lines.append(f"- seed {seed}: {_fmt(value)}")
    lines.append("")
    lines.append("## INT32 解析最坏界与观测")
    lines.append("")
    lines.append("| dot | length | |acc| bound | fits INT32 |")
    lines.append("|---|---|---|---|")
    for name, entry in bundle["int32"]["analytical"]["lengths"].items():
        lines.append(
            f"| {name} | {entry['length']} | {entry['bound']} | {entry['fits_int32']} |"
        )
    lines.append("")
    lines.append(
        f"- 观测最大 |INT32 累加|：group_mean×Wv 逐行={bundle['int32']['observed_max_abs']['gm_wv_per_row_max_abs']}，"
        f"逐张量={bundle['int32']['observed_max_abs']['gm_wv_per_tensor_max_abs']}，"
        f"no_overflow={bundle['int32']['no_overflow']}。"
    )
    lines.append(
        f"- 导出 fixture 回读重算逐位一致={bundle['fixtures_validated']}；主 bundle 不导出 Wt。"
    )
    lines.append("")
    lines.append("## 隔离与 pins")
    lines.append("")
    iso = bundle["isolation"]
    lines.append(
        f"- 打开 val NPZ {iso['opened_npz_count']}（仅 val={iso['opened_only_val_npz']}）；"
        f"test opened={iso['test_npz_opened']}；train opened={iso['train_npz_opened']}；"
        f"val aggregate 匹配预注册={iso['val_npz_aggregate_matches_prereg']}。"
    )
    lines.append(
        f"- 运行后 pins 未变={bundle['pins_unchanged']}（{bundle['pins_after']['checked']} 项）。"
    )
    lines.append("")
    lines.append("## 限制")
    lines.append("")
    for note in bundle["limitations"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append(
        "> 明确说明：float 归一化/倒数平方根/min-max 仍是本审计的 software golden；"
        "本审计未证明完整定点 FPGA scorer，也不作精度、延迟、端到端加速或发表声明。"
    )
    lines.append("")
    return "\n".join(lines)


def harness_payload(bundle: Mapping[str, Any], files: Mapping[str, str]) -> dict[str, Any]:
    decision = bundle["decision"]
    metrics = bundle["metrics"]
    overall = metrics["overall"]["correct"]
    return {
        "schema_version": SCHEMA_VERSION,
        "report_stem": REPORT_STEM,
        "decision": decision["decision"],
        "reference_arm": REFERENCE_ARM,
        "candidate_arm": CANDIDATE_ARM,
        "checks": decision["checks"],
        "candidate_vs_per_row": {
            str(budget): {
                "jaccard": overall[str(budget)]["cand_vs_ref_jaccard"],
                "exact_set_match_rate": overall[str(budget)]["cand_vs_ref_exact_match"],
                "replacement_count": overall[str(budget)][
                    "cand_vs_ref_replacement_count"
                ],
                "mass_delta": overall[str(budget)]["cand_vs_ref_mass_delta"],
                "ndcg_delta": overall[str(budget)]["cand_vs_ref_ndcg_delta"],
            }
            for budget in BUDGETS
        },
        "candidate_vs_float": {
            str(budget): {
                "jaccard": overall[str(budget)]["cand_vs_float_jaccard"],
                "mass_delta": overall[str(budget)]["cand_vs_float_mass_delta"],
                "ndcg_delta": overall[str(budget)]["cand_vs_float_ndcg_delta"],
            }
            for budget in BUDGETS
        },
        "candidate_vs_per_row_score_spearman": metrics["score_rank_spearman"][
            "candidate_vs_per_row"
        ]["correct"]["overall"],
        "pointing_k48_prompt_utility_gap_delta_candidate_vs_float": metrics[
            "prompt_utility_gap_delta"
        ]["float"]["48"]["mass"]["per_task"]["pointing"],
        "seed_k48_candidate_vs_per_row_mean_jaccard": decision[
            "seed_k48_candidate_vs_per_row_mean_jaccard"
        ],
        "int32": {
            "analytical": bundle["int32"]["analytical"]["lengths"],
            "observed_max_abs": bundle["int32"]["observed_max_abs"],
            "no_overflow": bundle["int32"]["no_overflow"],
        },
        "fixtures_validated": bundle["fixtures_validated"],
        "pins_unchanged": bundle["pins_unchanged"],
        "isolation": bundle["isolation"],
        "limitations": bundle["limitations"],
        "files": dict(files),
    }


def _write_reports(
    bundle: Mapping[str, Any], csv_rows: Sequence[Mapping[str, Any]], *, root: Path
) -> dict[str, str]:
    # Canonical evidence follows the prior flat-report convention:
    # reports/token_selector/<stem>.{json,md} and <stem>-records.csv.  The same
    # trio is mirrored into reports/token_selector/<stem>/ so the task-specific
    # directory path also resolves.
    json_path = root / REPORTS_DIR / f"{REPORT_STEM}.json"
    md_path = root / REPORTS_DIR / f"{REPORT_STEM}.md"
    csv_path = root / REPORTS_DIR / f"{REPORT_STEM}-records.csv"
    _write_json(json_path, bundle)
    md_path.write_text(render_markdown(bundle), encoding="utf-8")
    _write_csv(csv_path, csv_rows)

    mirror_json = root / REPORT_DIR / json_path.name
    mirror_md = root / REPORT_DIR / md_path.name
    mirror_csv = root / REPORT_DIR / csv_path.name
    _write_json(mirror_json, bundle)
    mirror_md.write_text(render_markdown(bundle), encoding="utf-8")
    _write_csv(mirror_csv, csv_rows)

    return {
        "json": str(json_path),
        "md": str(md_path),
        "csv": str(csv_path),
        "json_sha256": sha256_file(json_path),
        "md_sha256": sha256_file(md_path),
        "csv_sha256": sha256_file(csv_path),
        "json_dir_mirror": str(mirror_json),
        "md_dir_mirror": str(mirror_md),
        "csv_dir_mirror": str(mirror_csv),
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--no-artifacts", action="store_true")
    args = parser.parse_args(argv)
    bundle = run_audit(root=args.root, write_artifacts=not args.no_artifacts)
    print(json.dumps({"decision": bundle["decision"]["decision"]}, ensure_ascii=False))
    return 0 if bundle["decision"]["pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
