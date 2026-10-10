#!/usr/bin/env python3
"""Frozen INT8/INT32 numerical audit for the prompt-conditioned selector.

This is a **zero-training, preregistered, validation-only, CPU-only, read-only**
numeric-equivalence audit.  It answers exactly one engineering question: does the
trained prompt-conditioned spatial selector survive an FPGA-oriented INT8/INT32
numerical boundary without retraining or threshold tuning?

Two deterministic round-half-away-from-zero INT8 chains are implemented:

* ``SPLIT_WV_Q8_A8_Q8QUERY`` (primary deployment arm).  The host/Jetson computes
  the frozen float ``prompt_mean_vector`` query with the float ``Wt``, then
  per-vector symmetric INT8 quantises the final 128-D unit query.  The FPGA
  boundary receives that query plus the visual data: one global-per-image INT8
  ``group_mean``, per-``Wv``-output-row INT8 weights, exact INT32 projection,
  float dequantisation, row-L2 normalisation and the normalised visual-query
  dot.  L2 is computed from the dequantised activations, combined with the frozen
  ``lambda_semantic=0.5`` and quantised to a uint16 Q0.15 score.  Selection
  consumes the dequantised Q0.15 score, hence is exactly rank-equivalent to the
  hardware ``score_q`` including deterministic index tie-breaks.
* ``FULL_WT_WV_Q8_A8`` (descriptive control).  Additionally quantises ``Wt`` by
  output row and the prompt embeddings per prompt tensor, uses an INT32 prompt
  projection and then the same frozen project/row-normalise/mean/final-normalise
  query definition.  This is **not** the proposed FPGA partition and can never
  determine the decision.

Nothing is trained, tuned, synthesised or measured: the float L2 norm, the
min-max normalisation, the reciprocal square root and the row normalisation are
explicit software golden operations.  Passing does not prove a complete
fixed-point HLS scorer.

Reuse only (no second implementation of split / loader / target / scorer /
combine / anchor / metric):

* split / loader:   :func:`evaluation.prompt_projection_distillation.split_sample_ids`
                    and :func:`evaluation.train_pairwise_prompt_selector.load_split_samples`
* spatial targets:  :func:`evaluation.train_spatial_prompt_selector.attach_spatial_targets`
* scorer:           :func:`evaluation.frozen_scorer_reduction_ablation.score_candidate`
* combine:          :func:`evaluation.hardware_regular_token_selector.combine_scores`
* anchor selection: :func:`evaluation.train_spatial_prompt_selector._select_groups`
* metrics:          :func:`evaluation.prompt_projection_distillation.{saliency_mass_retained,ndcg_for_selected_indices,jaccard,spearman}`
* checkpoint:       :func:`evaluation.run_hf_hardware_regular_b1.load_projection`
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

from evaluation.frozen_scorer_reduction_ablation import score_candidate
from evaluation.hardware_regular_token_selector import combine_scores
from evaluation.prompt_identifiability_diagnostic import same_image_partner_map
from evaluation.prompt_projection_distillation import (
    DistillationContractError,
    jaccard,
    l2_baseline_scores,
    ndcg_for_selected_indices,
    saliency_mass_retained,
    spearman,
    split_sample_ids,
)
from evaluation.run_hf_hardware_regular_b1 import load_projection
from evaluation.spatial_teacher_feasibility_audit import load_clean_manifest
from evaluation.teacher_pair_signal_audit import load_index_metadata
from evaluation.train_pairwise_prompt_selector import load_split_samples, sha256_file
from evaluation.train_spatial_prompt_selector import _select_groups, attach_spatial_targets

REPORT_STEM = "2026-10-03-selector-int8-numerical-audit"
DATE = "2026-10-03"
SCHEMA_VERSION = "1.0"

ROOT = Path(__file__).resolve().parents[1]
PREREG_CONFIG_PATH = Path("configs/selector_int8_numerical_audit.preregistered.json")
INDEX_PATH = Path("artifacts/token_selector/prompt_projection_features/index.jsonl")
MANIFEST_PATH = Path("datasets/prompt_projection_clean_inductive_v1.jsonl")
REPORTS_DIR = Path("reports/token_selector")
LOGS_DIR = REPORTS_DIR / "logs"
ARTIFACT_ROOT = Path("artifacts/token_selector/selector_int8_numerical_audit")

PRIMARY_ARM = "SPLIT_WV_Q8_A8_Q8QUERY"
CONTROL_ARM = "FULL_WT_WV_Q8_A8"
FLOAT_ARM = "FLOAT_REFERENCE"
ARMS = (PRIMARY_ARM, CONTROL_ARM)
#: Arm labels that carry a quantised-vs-float comparison.
QUANT_ARMS = (PRIMARY_ARM, CONTROL_ARM)

SEEDS = (20261003, 20261004, 20261005)
BUDGETS = (48, 64, 96)
TASKS = ("grounding", "pointing")
ROLES = ("correct", "wrong")
LAMBDA_SEMANTIC = 0.5
STUDENT_SCORER = "prompt_mean_vector"

INT8_MIN = -128
INT8_MAX = 127
WEIGHT_MIN = -127
WEIGHT_MAX = 127
ACTIVATION_LEVELS = 128.0
WEIGHT_LEVELS = 127.0
SCORE_Q_SCALE = 1 << 15  # Q0.15 denominator
SCORE_Q_MAX = 32767
INT32_MAX = 2_147_483_647

BOOTSTRAP_RESAMPLES = 2000
STATISTICS_SEED = 20261003
CI_LEVEL = 0.95
TIE_EPS = 1e-12

#: Metric keys stored per (arm, role, budget).
BUDGET_METRICS = (
    "mass",
    "ndcg",
    "selected_count",
    "jaccard",
    "exact_match",
    "mass_delta",
    "ndcg_delta",
)

CSV_FIELDS = (
    "seed",
    "arm",
    "sample_id",
    "image_sha256",
    "task",
    "group_count",
    "grid_h",
    "grid_w",
    "prompt_role",
    "budget",
    "lambda_semantic",
    "quant_selected_count",
    "float_selected_count",
    "selected_set_jaccard",
    "exact_set_match",
    "retained_mass_quant",
    "retained_mass_float",
    "retained_mass_delta",
    "selected_ndcg_quant",
    "selected_ndcg_float",
    "selected_ndcg_delta",
    "score_rank_spearman",
)


class Int8AuditError(ValueError):
    """Malformed programmer input or a violated audit contract."""


# ---------------------------------------------------------------------------
# Rounding and quantisation primitives
# ---------------------------------------------------------------------------
def round_half_away_from_zero(values: Any) -> np.ndarray:
    """Element-wise round-half-away-from-zero for a float array.

    ``0.5 -> 1``, ``1.5 -> 2``, ``-0.5 -> -1`` and ``-1.5 -> -2``.  This is the
    frozen rounding rule of both the activation and the weight quantiser and of
    the Q0.15 combined-score packer.
    """
    array = np.asarray(values, dtype=np.float64)
    return np.sign(array) * np.floor(np.abs(array) + 0.5)


def quantize_activation(values: Any) -> tuple[np.ndarray, float]:
    """Symmetric INT8 activation quantisation ``s = 128/amax``.

    ``q = clamp(round_half_away(x*s), -128, 127)`` and the reconstructed value is
    ``x_hat = q/s``.  The whole tensor shares one scale.  A zero tensor returns a
    zero int64 quantisation with scale ``1.0``.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise Int8AuditError("activation quantisation needs a non-empty tensor")
    if not np.isfinite(array).all():
        raise Int8AuditError("activation tensor must be finite")
    amax = float(np.max(np.abs(array)))
    if amax <= 0.0:
        return np.zeros(array.shape, dtype=np.int64), 1.0
    scale = ACTIVATION_LEVELS / amax
    quantized = np.clip(round_half_away_from_zero(array * scale), INT8_MIN, INT8_MAX)
    return quantized.astype(np.int64), float(scale)


def quantize_weight_rows(matrix: Any) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-row symmetric INT8 weight quantisation ``d = amax/127``.

    ``q = clamp(round_half_away(w/d), -127, 127)`` and ``w_hat = q*d``.  A zero
    row keeps ``d = 1.0`` and a zero int64 quantisation.
    """
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise Int8AuditError("weight quantisation needs a non-empty 2-D matrix")
    if not np.isfinite(array).all():
        raise Int8AuditError("weight matrix must be finite")
    amax = np.max(np.abs(array), axis=1)
    scales = np.where(amax > 0.0, amax / WEIGHT_LEVELS, 1.0)
    quantized = np.clip(
        round_half_away_from_zero(array / scales[:, None]), WEIGHT_MIN, WEIGHT_MAX
    )
    return quantized.astype(np.int64), scales.astype(np.float64)


def quantize_score_q(combined: Any) -> np.ndarray:
    """Pack a combined score in ``[0, 1]`` to a uint16 Q0.15 code."""
    array = np.asarray(combined, dtype=np.float64)
    quantized = np.clip(
        round_half_away_from_zero(array * float(SCORE_Q_SCALE)),
        0.0,
        float(SCORE_Q_MAX),
    )
    return quantized.astype(np.uint16)


def dequantize_score_q(score_q: Any) -> np.ndarray:
    """Dequantise a uint16 Q0.15 code back to a float combined score."""
    return np.asarray(score_q, dtype=np.float64) / float(SCORE_Q_SCALE)


# ---------------------------------------------------------------------------
# Projection / query mathematics
# ---------------------------------------------------------------------------
def _row_l2(matrix: Any, name: str) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= TIE_EPS):
        raise Int8AuditError(f"{name} contains a zero-norm row")
    return array / norms[:, None]


def exact_int_matmul(left: Any, right: Any) -> np.ndarray:
    """Exact integer matrix product via float64 with an integrality assertion.

    Every operand is an int8-valued integer, each product is at most
    ``128*127 = 16256`` and the accumulated magnitude is far below ``2**53``, so
    every partial sum is exactly representable as an IEEE-754 double and the BLAS
    product is exact.  The result is validated to be integral and returned as
    int64; callers separately assert the INT32 bound.
    """
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    product = left_array @ right_array
    if product.size and not bool(np.all(product == np.rint(product))):
        raise Int8AuditError("exact_int_matmul produced a non-integral accumulator")
    return product.astype(np.int64)


def max_abs_int32(accumulator: Any) -> int:
    """Maximum absolute accumulator value as a Python int (0 for empty)."""
    array = np.asarray(accumulator, dtype=np.int64)
    if array.size == 0:
        return 0
    return int(np.max(np.abs(array)))


def analytical_int32_bound(length: int) -> dict[str, Any]:
    """Analytic worst-case INT32 accumulator bound for an int8 x int8 dot.

    Activations are clamped to ``[-128, 127]`` (max magnitude 128) and weights to
    ``[-127, 127]`` (max magnitude 127), so ``|acc| <= length * 128 * 127``.
    """
    length = int(length)
    if length <= 0:
        raise Int8AuditError("dot length must be positive")
    bound = length * 128 * 127
    return {
        "length": length,
        "activation_max_abs": 128,
        "weight_max_abs": 127,
        "bound": int(bound),
        "int32_max": INT32_MAX,
        "fits_int32": bool(bound <= INT32_MAX),
    }


def analytical_int32_bounds() -> dict[str, Any]:
    """Frozen INT32 bounds for lengths 1024, 2560 and 128."""
    lengths = {"group_mean_dot_wv_row": 1024, "prompt_dot_wt_row": 2560, "query_dot": 128}
    entries = {name: analytical_int32_bound(length) for name, length in lengths.items()}
    return {
        "rule": "|INT32 accumulator| <= length * 128 * 127; int32 max = 2147483647",
        "lengths": entries,
        "all_fit_int32": all(entry["fits_int32"] for entry in entries.values()),
    }


def float_prompt_query(prompt_embeddings: Any, Wt: Any) -> np.ndarray:
    """Frozen float ``prompt_mean_vector`` query: project, row-L2, mean, L2."""
    prompt = np.asarray(prompt_embeddings, dtype=np.float64)
    weight = np.asarray(Wt, dtype=np.float64)
    projection = prompt @ weight.T
    unit = _row_l2(projection, "float prompt projection")
    query = unit.mean(axis=0)
    norm = float(np.linalg.norm(query))
    if norm <= TIE_EPS:
        raise Int8AuditError("float prompt mean query has zero norm")
    return query / norm


def float_reference_scores(
    group_mean: Any, prompt_embeddings: Any, Wt: Any, Wv: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frozen float B-checkpoint scorer and float combined score path."""
    semantic = np.asarray(
        score_candidate(STUDENT_SCORER, group_mean, prompt_embeddings, Wt, Wv),
        dtype=np.float64,
    )
    l2 = np.asarray(l2_baseline_scores(group_mean), dtype=np.float64)
    combined = np.asarray(combine_scores(semantic, l2, LAMBDA_SEMANTIC), dtype=np.float64)
    return semantic, l2, combined


def primary_quantized_query(prompt_embeddings: Any, Wt: Any) -> dict[str, Any]:
    """Per-vector symmetric INT8 quantisation of the final float query."""
    float_query = float_prompt_query(prompt_embeddings, Wt)
    query_q, query_scale = quantize_activation(float_query)
    query_hat = query_q.astype(np.float64) / query_scale
    query_hat = query_hat / float(np.linalg.norm(query_hat))
    return {
        "query_float": float_query,
        "query_q": query_q.astype(np.int8),
        "query_scale": float(query_scale),
        "query_hat": query_hat,
    }


def control_quantized_query(
    prompt_embeddings: Any, Wt: Any, wt_q: np.ndarray, wt_scales: np.ndarray
) -> dict[str, Any]:
    """FULL_WT_WV_Q8_A8 query: quantise Wt rows + prompt, INT32 project, float golden.

    The prompt projection uses exact INT32 accumulation, is dequantised by
    ``(1/prompt_scale) * wt_scales[row]`` and then follows the same frozen
    project-row-normalise/mean/final-normalise definition.  ``wt_q``/``wt_scales``
    are the pre-computed per-seed ``Wt`` row quantisation.
    """
    prompt_q, prompt_scale = quantize_activation(prompt_embeddings)
    projection_int = exact_int_matmul(prompt_q, np.asarray(wt_q).T)
    projection = (
        projection_int.astype(np.float64)
        * (1.0 / prompt_scale)
        * np.asarray(wt_scales, dtype=np.float64)[None, :]
    )
    unit = _row_l2(projection, "control prompt projection")
    query = unit.mean(axis=0)
    norm = float(np.linalg.norm(query))
    if norm <= TIE_EPS:
        raise Int8AuditError("control prompt mean query has zero norm")
    return {
        "query_hat": query / norm,
        "prompt_q": prompt_q.astype(np.int8),
        "prompt_scale": float(prompt_scale),
        "projection_int": projection_int,
    }


def quantized_visual_projection(
    group_mean_q: np.ndarray,
    group_mean_scale: float,
    wv_q: np.ndarray,
    wv_scales: np.ndarray,
) -> dict[str, Any]:
    """Exact INT32 ``group_mean @ Wv.T`` then dequantise by ``(1/s)*d_w[row]``."""
    projection_int = exact_int_matmul(np.asarray(group_mean_q), np.asarray(wv_q).T)
    projection = (
        projection_int.astype(np.float64)
        * (1.0 / float(group_mean_scale))
        * np.asarray(wv_scales, dtype=np.float64)[None, :]
    )
    visual_unit = _row_l2(projection, "quantized visual projection")
    return {"projection_int": projection_int, "visual_unit": visual_unit}


def quantized_combined_scores(
    visual_unit: np.ndarray, query_hat: np.ndarray, l2_hat: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalised visual-query dot, combine, and uint16 Q0.15 pack."""
    semantic = np.asarray(visual_unit, dtype=np.float64) @ np.asarray(
        query_hat, dtype=np.float64
    )
    combined = np.asarray(
        combine_scores(semantic, l2_hat, LAMBDA_SEMANTIC), dtype=np.float64
    )
    score_q = quantize_score_q(combined)
    score_hat = dequantize_score_q(score_q)
    return combined, score_q, score_hat


# ---------------------------------------------------------------------------
# Selection and record metrics
# ---------------------------------------------------------------------------
def selected_indices(scores: Any, grid_h: int, grid_w: int, budget: int) -> tuple[int, ...]:
    """Frozen anchor selection on a score vector, deterministic low-index ties."""
    array = np.asarray(scores, dtype=np.float64)
    effective = min(int(budget), int(array.shape[0]))
    selection = _select_groups(array, int(grid_h), int(grid_w), effective)
    return tuple(sorted({int(value) for value in np.asarray(selection).reshape(-1)}))


def score_selection_metrics(
    scores: Any, target: Any, grid_h: int, grid_w: int, budgets: Sequence[int]
) -> dict[int, dict[str, Any]]:
    """Per-budget retained mass and selected-set NDCG for a score vector."""
    score_array = np.asarray(scores, dtype=np.float64)
    relevance = np.asarray(target, dtype=np.float64).reshape(-1)
    result: dict[int, dict[str, Any]] = {}
    for budget in budgets:
        budget = int(budget)
        selected = selected_indices(score_array, grid_h, grid_w, budget)
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
    wt_q: np.ndarray,
    wt_scales: np.ndarray,
    wv_q: np.ndarray,
    wv_scales: np.ndarray,
    target: Any,
    grid_h: int,
    grid_w: int,
    budgets: Sequence[int] = BUDGETS,
) -> dict[str, Any]:
    """Evaluate both quantised arms and the float reference for one record.

    ``wt_q``/``wt_scales``/``wv_q``/``wv_scales`` are the per-seed quantisations
    shared by every record so no duplicate quantiser semantics are created.
    """
    budgets = [int(value) for value in budgets]
    mean = np.asarray(group_mean, dtype=np.float64)
    correct = np.asarray(prompt_correct, dtype=np.float64)
    wrong = np.asarray(prompt_wrong, dtype=np.float64)

    group_mean_q, group_mean_scale = quantize_activation(mean)
    group_mean_hat = group_mean_q.astype(np.float64) / group_mean_scale
    l2_hat = np.linalg.norm(group_mean_hat, axis=1)

    visual = quantized_visual_projection(group_mean_q, group_mean_scale, wv_q, wv_scales)
    visual_unit = visual["visual_unit"]
    projection_int = visual["projection_int"]

    # Float reference: frozen scorer + float combine for both prompts.
    float_selection: dict[str, dict[int, dict[str, Any]]] = {}
    float_combined: dict[str, np.ndarray] = {}
    for role, prompt in (("correct", correct), ("wrong", wrong)):
        _, _, combined = float_reference_scores(mean, prompt, Wt, Wv)
        float_combined[role] = combined
        float_selection[role] = score_selection_metrics(
            combined, target, grid_h, grid_w, budgets
        )

    arms: dict[str, Any] = {}
    for arm in QUANT_ARMS:
        prompt_queries: dict[str, np.ndarray] = {}
        extra: dict[str, Any] = {}
        if arm == PRIMARY_ARM:
            for role, prompt in (("correct", correct), ("wrong", wrong)):
                entry = primary_quantized_query(prompt, Wt)
                prompt_queries[role] = entry["query_hat"]
                extra[f"{role}_query_q"] = entry["query_q"]
                extra[f"{role}_query_scale"] = entry["query_scale"]
            extra["prompt_projection_int_max_abs"] = 0
        else:
            for role, prompt in (("correct", correct), ("wrong", wrong)):
                entry = control_quantized_query(prompt, Wt, wt_q, wt_scales)
                prompt_queries[role] = entry["query_hat"]
                extra[f"{role}_prompt_q"] = entry["prompt_q"]
                extra[f"{role}_prompt_scale"] = entry["prompt_scale"]
                extra[f"{role}_prompt_projection_int"] = entry["projection_int"]
            extra["prompt_projection_int_max_abs"] = max(
                max_abs_int32(extra["correct_prompt_projection_int"]),
                max_abs_int32(extra["wrong_prompt_projection_int"]),
            )

        arm_entry: dict[str, Any] = {"by_role": {}, **extra}
        for role in ROLES:
            _, score_q, score_hat = quantized_combined_scores(
                visual_unit, prompt_queries[role], l2_hat
            )
            quant = score_selection_metrics(score_hat, target, grid_h, grid_w, budgets)
            role_metrics: dict[int, dict[str, Any]] = {}
            for budget in budgets:
                float_entry = float_selection[role][budget]
                quant_entry = quant[budget]
                quant_set = set(quant_entry["selected"])
                float_set = set(float_entry["selected"])
                role_metrics[budget] = {
                    "mass": quant_entry["mass"],
                    "ndcg": quant_entry["ndcg"],
                    "selected_count": len(quant_entry["selected"]),
                    "selected": quant_entry["selected"],
                    "float_selected": float_entry["selected"],
                    "jaccard": float(
                        jaccard(quant_entry["selected"], float_entry["selected"])
                    ),
                    "exact_match": 1.0 if quant_set == float_set else 0.0,
                    "mass_delta": quant_entry["mass"] - float_entry["mass"],
                    "ndcg_delta": quant_entry["ndcg"] - float_entry["ndcg"],
                    "score_q": score_q,
                    "score_hat": score_hat,
                }
            arm_entry["by_role"][role] = role_metrics
            arm_entry[f"spearman_{role}"] = float(spearman(score_hat, float_combined[role]))
        arms[arm] = arm_entry

    return {
        "arms": arms,
        "float": float_selection,
        "group_mean_scale": float(group_mean_scale),
        "group_mean_q_absmax": int(np.max(np.abs(group_mean_q))),
        "wv_absmax": int(np.max(np.abs(wv_q))),
        "projection_int_max_abs": max_abs_int32(projection_int),
        "group_count": int(mean.shape[0]),
    }


# ---------------------------------------------------------------------------
# Aggregation and bootstrap
# ---------------------------------------------------------------------------
def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int = STATISTICS_SEED,
    n_bootstrap: int = BOOTSTRAP_RESAMPLES,
    ci_level: float = CI_LEVEL,
) -> dict[str, Any]:
    """Mean and percentile bootstrap CI over independent units (images)."""
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    n = int(array.size)
    entry: dict[str, Any] = {
        "n": n,
        "mean": float(array.mean()) if n else None,
        "median": float(np.median(array)) if n else None,
        "ci_low": None,
        "ci_high": None,
        "ci_level": float(ci_level),
        "bootstrap_resamples": int(n_bootstrap),
    }
    if n == 0:
        return entry
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, n, size=(int(n_bootstrap), n))
    boot = array[draws].mean(axis=1)
    low, high = np.percentile(
        boot, [100.0 * (1.0 - ci_level) / 2.0, 100.0 * (1.0 + ci_level) / 2.0]
    )
    entry["ci_low"] = float(low)
    entry["ci_high"] = float(high)
    return entry


def _median_across_seeds(values: Sequence[float]) -> float:
    if not values:
        raise Int8AuditError("cannot take a median of zero seed values")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def aggregate_metrics(
    raw_by_seed: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    *,
    sample_meta: Mapping[str, Mapping[str, Any]],
    seeds: Sequence[int] = SEEDS,
    arms: Sequence[str] = (PRIMARY_ARM, CONTROL_ARM, FLOAT_ARM),
    budgets: Sequence[int] = BUDGETS,
    roles: Sequence[str] = ROLES,
) -> dict[str, Any]:
    """Reduce per-seed records to per-record medians and image-level summaries.

    ``raw_by_seed[seed][arm][sample_id][role][budget]["metrics"][metric]`` is the
    frozen input and ``raw_by_seed[seed][arm][sample_id]["spearman_correct"]`` is
    the rank correlation.  The float reference is carried as :data:`FLOAT_ARM` so
    that quant-minus-float deltas are computed on the same per-record medians.
    """
    seeds = [int(value) for value in seeds]
    budgets = [int(value) for value in budgets]
    sample_ids = sorted(str(value) for value in sample_meta)
    image_of = {sid: str(sample_meta[sid]["image_sha256"]) for sid in sample_ids}
    task_of = {sid: str(sample_meta[sid]["task"]) for sid in sample_ids}
    images = sorted(set(image_of.values()))

    # --- per-record median across seeds -----------------------------------
    collected: dict[tuple[str, str, int, str], dict[str, list[float]]] = {}
    for arm in arms:
        for role in roles:
            for budget in budgets:
                for metric in BUDGET_METRICS:
                    collected[(arm, role, budget, metric)] = {}
    spearman_collected: dict[tuple[str, str], list[float]] = {}
    for seed in seeds:
        for arm in arms:
            for sid in sample_ids:
                if arm in QUANT_ARMS:
                    spearman_collected.setdefault((arm, sid), []).append(
                        float(raw_by_seed[seed][arm][sid]["spearman_correct"])
                    )
                for role in roles:
                    for budget in budgets:
                        metrics = raw_by_seed[seed][arm][sid][role][budget]["metrics"]
                        for metric in BUDGET_METRICS:
                            collected[(arm, role, budget, metric)].setdefault(
                                sid, []
                            ).append(float(metrics[metric]))
    record_median: dict[tuple[str, str, int, str], dict[str, float]] = {
        key: {sid: _median_across_seeds(values) for sid, values in per_sid.items()}
        for key, per_sid in collected.items()
    }
    spearman_median = {
        key: _median_across_seeds(values) for key, values in spearman_collected.items()
    }

    # --- seed-level overall correct-prompt Jaccard ------------------------
    seed_overall: dict[str, dict[int, dict[str, float]]] = {}
    for arm in arms:
        seed_overall[arm] = {}
        for seed in seeds:
            seed_overall[arm][seed] = {}
            for budget in budgets:
                per_image: dict[str, list[float]] = {}
                for sid in sample_ids:
                    per_image.setdefault(image_of[sid], []).append(
                        float(
                            raw_by_seed[seed][arm][sid]["correct"][budget]["metrics"][
                                "jaccard"
                            ]
                        )
                    )
                seed_overall[arm][seed][str(budget)] = float(
                    np.mean([np.mean(values) for values in per_image.values()])
                )

    # --- image-level series ----------------------------------------------
    def series(
        arm: str, role: str, budget: int, metric: str, tasks: Sequence[str]
    ) -> dict[str, float]:
        per_image: dict[str, list[float]] = {}
        for sid in sample_ids:
            if task_of[sid] not in tasks:
                continue
            value = record_median[(arm, role, budget, metric)].get(sid)
            if value is None:
                continue
            per_image.setdefault(image_of[sid], []).append(float(value))
        return {image: float(np.mean(values)) for image, values in per_image.items()}

    overall: dict[str, Any] = {}
    per_task: dict[str, Any] = {}
    overall_by_role: dict[str, Any] = {}
    for arm in arms:
        overall[arm] = {str(budget): {} for budget in budgets}
        per_task[arm] = {task: {str(budget): {} for budget in budgets} for task in TASKS}
        for budget in budgets:
            for metric in BUDGET_METRICS:
                overall[arm][str(budget)][metric] = bootstrap_mean_ci(
                    list(series(arm, "correct", budget, metric, TASKS).values())
                )
                for task in TASKS:
                    per_task[arm][task][str(budget)][metric] = bootstrap_mean_ci(
                        list(series(arm, "correct", budget, metric, (task,)).values())
                    )
    for role in roles:
        overall_by_role[role] = {arm: {str(budget): {} for budget in budgets} for arm in arms}
        for arm in arms:
            for budget in budgets:
                for metric in BUDGET_METRICS:
                    overall_by_role[role][arm][str(budget)][metric] = bootstrap_mean_ci(
                        list(series(arm, role, budget, metric, TASKS).values())
                    )

    # --- prompt-utility gaps (correct - wrong) ---------------------------
    gap: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for arm in arms:
        gap[arm] = {str(budget): {} for budget in budgets}
        for budget in budgets:
            for metric in ("mass", "ndcg"):
                gap[arm][str(budget)][metric] = {
                    sid: record_median[(arm, "correct", budget, metric)][sid]
                    - record_median[(arm, "wrong", budget, metric)][sid]
                    for sid in sample_ids
                }
    gap_delta: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for arm in QUANT_ARMS:
        gap_delta[arm] = {str(budget): {} for budget in budgets}
        for budget in budgets:
            for metric in ("mass", "ndcg"):
                gap_delta[arm][str(budget)][metric] = {
                    sid: gap[arm][str(budget)][metric][sid]
                    - gap[FLOAT_ARM][str(budget)][metric][sid]
                    for sid in sample_ids
                }

    def gap_series(
        arm: str, budget: int, metric: str, tasks: Sequence[str]
    ) -> dict[str, float]:
        per_image: dict[str, list[float]] = {}
        for sid in sample_ids:
            if task_of[sid] not in tasks:
                continue
            per_image.setdefault(image_of[sid], []).append(
                float(gap_delta[arm][str(budget)][metric][sid])
            )
        return {image: float(np.mean(values)) for image, values in per_image.items()}

    prompt_utility: dict[str, Any] = {}
    for arm in QUANT_ARMS:
        prompt_utility[arm] = {str(budget): {} for budget in budgets}
        for budget in budgets:
            for metric in ("mass", "ndcg"):
                prompt_utility[arm][str(budget)][metric] = {
                    "overall": bootstrap_mean_ci(
                        list(gap_series(arm, budget, metric, TASKS).values())
                    ),
                    "per_task": {
                        task: bootstrap_mean_ci(
                            list(gap_series(arm, budget, metric, (task,)).values())
                        )
                        for task in TASKS
                    },
                }

    # --- score rank Spearman (correct prompt) ----------------------------
    spearman_summary: dict[str, Any] = {}
    for arm in QUANT_ARMS:
        spearman_summary[arm] = {
            "overall": bootstrap_mean_ci(
                [spearman_median[(arm, sid)] for sid in sample_ids]
            ),
            "per_task": {
                task: bootstrap_mean_ci(
                    [
                        spearman_median[(arm, sid)]
                        for sid in sample_ids
                        if task_of[sid] == task
                    ]
                )
                for task in TASKS
            },
        }

    return {
        "independent_unit": "image",
        "record_aggregation": "per record then median across the three seeds",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": STATISTICS_SEED,
        "confidence": CI_LEVEL,
        "n_records": len(sample_ids),
        "n_images": len(images),
        "record_medians": {
            f"{arm}|{role}|{budget}|{metric}": values
            for (arm, role, budget, metric), values in record_median.items()
        },
        "overall": overall,
        "per_task": per_task,
        "overall_by_role": overall_by_role,
        "prompt_utility_gap_delta": prompt_utility,
        "score_rank_spearman": spearman_summary,
        "seed_overall": seed_overall,
    }


def _get_metric(summary: Mapping[str, Any], arm: str, budget: int, metric: str) -> float | None:
    entry = summary["overall"][arm][str(int(budget))].get(metric)
    if entry is None or entry.get("mean") is None:
        return None
    return float(entry["mean"])


def evaluate_decision(
    summary: Mapping[str, Any],
    *,
    pins_ok: bool,
    isolation_ok: bool,
    int32_ok: bool,
    budgets: Sequence[int] = BUDGETS,
) -> dict[str, Any]:
    """Frozen ``INT8_NUMERICS_PASS`` / ``INT8_NUMERICS_NOT_ESTABLISHED`` rule."""
    budgets = [int(value) for value in budgets]
    arm = PRIMARY_ARM
    jaccard = {budget: _get_metric(summary, arm, budget, "jaccard") for budget in budgets}
    exact = {budget: _get_metric(summary, arm, budget, "exact_match") for budget in budgets}
    mass_delta = {budget: _get_metric(summary, arm, budget, "mass_delta") for budget in budgets}
    ndcg_delta = {budget: _get_metric(summary, arm, budget, "ndcg_delta") for budget in budgets}

    pointing_gap = summary["prompt_utility_gap_delta"][arm]["48"]["mass"]["per_task"][
        "pointing"
    ]
    pointing_gap_mean = (
        None if pointing_gap.get("mean") is None else float(pointing_gap["mean"])
    )
    seed_k48 = {
        str(seed): float(summary["seed_overall"][arm][seed]["48"])
        for seed in sorted(summary["seed_overall"][arm])
    }

    checks = {
        "pins_and_isolation_pass": bool(pins_ok and isolation_ok),
        "int32_worst_case_bounds_below_2147483647": bool(int32_ok),
        "mean_jaccard_each_budget_at_least_0.95": all(
            value is not None and value >= 0.95 for value in jaccard.values()
        ),
        "exact_set_match_rate_each_budget_at_least_0.75": all(
            value is not None and value >= 0.75 for value in exact.values()
        ),
        "mean_retained_mass_delta_each_budget_at_least_-0.01": all(
            value is not None and value >= -0.01 for value in mass_delta.values()
        ),
        "mean_selected_ndcg_delta_each_budget_at_least_-0.01": all(
            value is not None and value >= -0.01 for value in ndcg_delta.values()
        ),
        "pointing_k48_prompt_utility_gap_delta_at_least_-0.01": bool(
            pointing_gap_mean is not None and pointing_gap_mean >= -0.01
        ),
        "each_seed_k48_mean_jaccard_at_least_0.95": all(
            value >= 0.95 for value in seed_k48.values()
        ),
    }
    passed = all(checks.values())
    return {
        "decision": "INT8_NUMERICS_PASS" if passed else "INT8_NUMERICS_NOT_ESTABLISHED",
        "pass": bool(passed),
        "checks": checks,
        "primary_arm": arm,
        "overall_mean_jaccard": {str(b): jaccard[b] for b in budgets},
        "overall_exact_set_match_rate": {str(b): exact[b] for b in budgets},
        "overall_mean_retained_mass_delta": {str(b): mass_delta[b] for b in budgets},
        "overall_mean_ndcg_delta": {str(b): ndcg_delta[b] for b in budgets},
        "pointing_k48_prompt_utility_gap_delta_mean": pointing_gap_mean,
        "seed_k48_mean_jaccard": seed_k48,
        "rule": (
            "INT8_NUMERICS_PASS iff pins/isolation pass, no INT32 overflow, every K "
            "mean Jaccard >= 0.95, exact-set rate >= 0.75, mass delta >= -0.01, "
            "NDCG delta >= -0.01, K48 Pointing prompt-utility gap delta >= -0.01 "
            "and all three seeds K48 mean Jaccard >= 0.95"
        ),
    }


# ---------------------------------------------------------------------------
# Preregistration pins and isolation
# ---------------------------------------------------------------------------
def load_prereg(path: Path = PREREG_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_pins(prereg: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    """Recompute every frozen input and B-checkpoint SHA256 in the prereg."""
    results: dict[str, Any] = {}
    mismatches: list[str] = []
    for rel, expected in prereg.get("frozen_inputs", {}).items():
        path = Path(root) / str(rel)
        actual = sha256_file(path) if path.is_file() else None
        ok = actual == str(expected)
        results[str(rel)] = {
            "kind": "frozen_input",
            "expected": str(expected),
            "actual": actual,
            "unchanged": bool(ok),
        }
        if not ok:
            mismatches.append(str(rel))
    for checkpoint in prereg.get("checkpoints", []):
        rel = str(checkpoint["path"])
        path = Path(root) / rel
        actual = sha256_file(path) if path.is_file() else None
        ok = actual == str(checkpoint["sha256"])
        results[rel] = {
            "kind": "checkpoint",
            "seed": int(checkpoint["seed"]),
            "expected": str(checkpoint["sha256"]),
            "actual": actual,
            "unchanged": bool(ok),
        }
        if not ok:
            mismatches.append(rel)
    return {
        "checked": len(results),
        "mismatches": mismatches,
        "pins_unchanged": not mismatches,
        "per_input": results,
    }


def verify_prereguration_identity(
    prereg: Mapping[str, Any], *, root: Path = ROOT, config_path: Path = PREREG_CONFIG_PATH
) -> dict[str, Any]:
    """Verify config SHA, status and authority-prompt SHA agreement."""
    cfg_path = Path(root) / Path(config_path)
    authority = prereg.get("authority_prompt", {})
    authority_path = Path(root) / str(authority.get("path"))
    authority_actual = sha256_file(authority_path) if authority_path.is_file() else None
    identity = {
        "config_path": str(config_path),
        "config_sha256": sha256_file(cfg_path),
        "status": str(prereg.get("status")),
        "authority_prompt_path": str(authority.get("path")),
        "authority_prompt_sha256": authority_actual,
        "authority_prompt_expected": str(authority.get("sha256")),
        "authority_prompt_matches": authority_actual == str(authority.get("sha256")),
        "status_is_validation_only": str(prereg.get("status"))
        == "PREREGISTERED_VALIDATION_ONLY",
    }
    identity["consistent"] = bool(
        identity["authority_prompt_matches"] and identity["status_is_validation_only"]
    )
    return identity


# ---------------------------------------------------------------------------
# Compact hardware bundle + deterministic fixtures
# ---------------------------------------------------------------------------
def _fixture_record(
    sample: Mapping[str, Any],
    *,
    Wt: np.ndarray,
    Wv: np.ndarray,
    wv_q: np.ndarray,
    wv_scales: np.ndarray,
) -> dict[str, Any]:
    """Primary-arm fixture for one record (no raw image or raw feature dump)."""
    group_mean = np.asarray(sample["group_mean"], dtype=np.float64)
    correct = np.asarray(sample["prompt_embeddings"], dtype=np.float64)
    wrong = np.asarray(sample["wrong_prompt_embeddings"], dtype=np.float64)
    grid_h = int(sample["grid_h"])
    grid_w = int(sample["grid_w"])

    group_mean_q, group_mean_scale = quantize_activation(group_mean)
    group_mean_hat = group_mean_q.astype(np.float64) / group_mean_scale
    l2_hat = np.linalg.norm(group_mean_hat, axis=1)
    visual = quantized_visual_projection(group_mean_q, group_mean_scale, wv_q, wv_scales)

    payload: dict[str, Any] = {
        "group_mean_q": group_mean_q.astype(np.int8),
        "group_mean_scale": np.asarray(group_mean_scale, dtype=np.float64),
        "grid_h": np.asarray(grid_h, dtype=np.int64),
        "grid_w": np.asarray(grid_w, dtype=np.int64),
        "sample_id": np.asarray(str(sample["sample_id"])),
    }
    for role, prompt in (("correct", correct), ("wrong", wrong)):
        query = primary_quantized_query(prompt, Wt)
        _, score_q, score_hat = quantized_combined_scores(
            visual["visual_unit"], query["query_hat"], l2_hat
        )
        payload[f"query_q_{role}"] = query["query_q"].astype(np.int8)
        payload[f"query_scale_{role}"] = np.asarray(query["query_scale"], dtype=np.float64)
        payload[f"score_q_{role}"] = score_q.astype(np.uint16)
        for budget in BUDGETS:
            payload[f"selected_{role}_k{budget}"] = np.asarray(
                selected_indices(score_hat, grid_h, grid_w, budget), dtype=np.int32
            )
        _, _, combined_float = float_reference_scores(group_mean, prompt, Wt, Wv)
        for budget in BUDGETS:
            payload[f"float_selected_{role}_k{budget}"] = np.asarray(
                selected_indices(combined_float, grid_h, grid_w, budget), dtype=np.int32
            )
    return payload


def build_seed_bundle(
    seed: int,
    *,
    Wt: np.ndarray,
    Wv: np.ndarray,
    fixtures: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Write one compact hardware bundle and return its metadata record."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wv_q, wv_scales = quantize_weight_rows(Wv)
    wv_bytes = np.ascontiguousarray(wv_q, dtype=np.int8).tobytes(order="C")
    scale_bytes = np.ascontiguousarray(wv_scales, dtype="<f4").tobytes(order="C")
    (out_dir / "Wv_int8_rowmajor.bin").write_bytes(wv_bytes)
    (out_dir / "Wv_row_scales_f32le.bin").write_bytes(scale_bytes)

    fixture_files: list[dict[str, Any]] = []
    for sample in fixtures:
        payload = _fixture_record(sample, Wt=Wt, Wv=Wv, wv_q=wv_q, wv_scales=wv_scales)
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

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "primary_arm": PRIMARY_ARM,
        "descriptive_control": CONTROL_ARM,
        "source_checkpoint": {
            "path": str(checkpoint["path"]),
            "sha256": str(checkpoint["sha256"]),
            "canonical_weights_sha256": str(checkpoint.get("canonical_weights_sha256", "")),
        },
        "wt_exported": False,
        "wt_note": "the primary deployment bundle intentionally does not export Wt",
        "wv": {
            "layout": "row-major [128, 1024] int8, output row = Wv row",
            "shape": [int(wv_q.shape[0]), int(wv_q.shape[1])],
            "dtype": "int8",
            "bytes": len(wv_bytes),
            "file": "Wv_int8_rowmajor.bin",
            "sha256": sha256_file(out_dir / "Wv_int8_rowmajor.bin"),
            "quantization": (
                "d=max(abs(row))/127; q=clamp(round_half_away(w/d),-127,127); w_hat=q*d"
            ),
            "scale_layout": "float32 little-endian [128] per output row",
            "scale_bytes": len(scale_bytes),
            "scale_file": "Wv_row_scales_f32le.bin",
            "scale_sha256": sha256_file(out_dir / "Wv_row_scales_f32le.bin"),
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
            "primary": "host float Wt prompt_mean_vector then per-vector symmetric INT8",
            "control": (
                "per-Wt-row and per-prompt-tensor INT8 then INT32 projection and float "
                "query golden"
            ),
        },
        "fixtures": fixture_files,
        "files": sorted(
            ["Wv_int8_rowmajor.bin", "Wv_row_scales_f32le.bin", "metadata.json"]
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
    prereg = load_prereg(root / prereg_config_path)
    identity = verify_prereguration_identity(
        prereg, root=root, config_path=prereg_config_path
    )
    if not identity["consistent"]:
        raise Int8AuditError("preregistration identity/authority-prompt mismatch")
    pins_before = verify_pins(prereg, root=root)
    if not pins_before["pins_unchanged"]:
        raise Int8AuditError(
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
        raise Int8AuditError(f"internal test NPZ were opened: {test_opened}")
    if train_opened:
        raise Int8AuditError(f"train NPZ were opened during a val-only audit: {train_opened}")
    val_aggregate_ok = val_aggregate == str(
        split_rule["expected_val_npz_aggregate_sha256"]
    )
    if not val_aggregate_ok:
        raise Int8AuditError("val NPZ aggregate SHA256 does not match the preregistration")

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
    bounds = analytical_int32_bounds()
    if not bounds["all_fit_int32"]:
        raise Int8AuditError("analytical INT32 bound violation")

    raw_by_seed: dict[int, dict[str, Any]] = {}
    csv_rows: list[dict[str, Any]] = []
    bundle_meta: list[dict[str, Any]] = []
    int32_observed = {"gm_wv_max_abs": 0, "prompt_wt_max_abs": 0}
    for seed in [int(value) for value in seeds]:
        checkpoint = checkpoint_by_seed[seed]
        ckpt_path = root / str(checkpoint["path"])
        projection, _ = load_projection(ckpt_path)
        if sha256_file(ckpt_path) != str(checkpoint["sha256"]):
            raise Int8AuditError(f"checkpoint {ckpt_path} changed after load")
        Wt = np.asarray(projection.Wt, dtype=np.float64)
        Wv = np.asarray(projection.Wv, dtype=np.float64)
        wt_q, wt_scales = quantize_weight_rows(Wt)
        wv_q, wv_scales = quantize_weight_rows(Wv)

        raw_by_seed[seed] = {arm: {} for arm in (PRIMARY_ARM, CONTROL_ARM, FLOAT_ARM)}
        for arm in raw_by_seed[seed]:
            for sample_id in sample_meta:
                raw_by_seed[seed][arm][sample_id] = {role: {} for role in ROLES}

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
                wt_q=wt_q,
                wt_scales=wt_scales,
                wv_q=wv_q,
                wv_scales=wv_scales,
                target=target,
                grid_h=int(sample["grid_h"]),
                grid_w=int(sample["grid_w"]),
                budgets=budgets,
            )
            int32_observed["gm_wv_max_abs"] = max(
                int32_observed["gm_wv_max_abs"], int(record["projection_int_max_abs"])
            )
            for arm in QUANT_ARMS:
                int32_observed["prompt_wt_max_abs"] = max(
                    int32_observed["prompt_wt_max_abs"],
                    int(record["arms"][arm]["prompt_projection_int_max_abs"]),
                )

            # Float reference stored as an arm-like entry so quant-minus-float is
            # computed on the same per-record medians.
            for role in ROLES:
                raw_by_seed[seed][FLOAT_ARM][sample_id][role] = {
                    budget: {
                        "metrics": {
                            "mass": float(record["float"][role][budget]["mass"]),
                            "ndcg": float(record["float"][role][budget]["ndcg"]),
                            "selected_count": len(record["float"][role][budget]["selected"]),
                            "jaccard": 1.0,
                            "exact_match": 1.0,
                            "mass_delta": 0.0,
                            "ndcg_delta": 0.0,
                        }
                    }
                    for budget in budgets
                }
            for arm in QUANT_ARMS:
                for role in ROLES:
                    raw_by_seed[seed][arm][sample_id][role] = {
                        budget: {
                            "metrics": {
                                "mass": float(
                                    record["arms"][arm]["by_role"][role][budget]["mass"]
                                ),
                                "ndcg": float(
                                    record["arms"][arm]["by_role"][role][budget]["ndcg"]
                                ),
                                "selected_count": int(
                                    record["arms"][arm]["by_role"][role][budget][
                                        "selected_count"
                                    ]
                                ),
                                "jaccard": float(
                                    record["arms"][arm]["by_role"][role][budget]["jaccard"]
                                ),
                                "exact_match": float(
                                    record["arms"][arm]["by_role"][role][budget][
                                        "exact_match"
                                    ]
                                ),
                                "mass_delta": float(
                                    record["arms"][arm]["by_role"][role][budget][
                                        "mass_delta"
                                    ]
                                ),
                                "ndcg_delta": float(
                                    record["arms"][arm]["by_role"][role][budget][
                                        "ndcg_delta"
                                    ]
                                ),
                            }
                        }
                        for budget in budgets
                    }
                    raw_by_seed[seed][arm][sample_id][f"spearman_{role}"] = float(
                        record["arms"][arm][f"spearman_{role}"]
                    )
                    for budget in budgets:
                        budget = int(budget)
                        csv_rows.append(
                            {
                                "seed": seed,
                                "arm": arm,
                                "sample_id": sample_id,
                                "image_sha256": sample_meta[sample_id]["image_sha256"],
                                "task": sample_meta[sample_id]["task"],
                                "group_count": sample_meta[sample_id]["group_count"],
                                "grid_h": sample_meta[sample_id]["grid_h"],
                                "grid_w": sample_meta[sample_id]["grid_w"],
                                "prompt_role": role,
                                "budget": budget,
                                "lambda_semantic": LAMBDA_SEMANTIC,
                                "quant_selected_count": int(
                                    record["arms"][arm]["by_role"][role][budget][
                                        "selected_count"
                                    ]
                                ),
                                "float_selected_count": len(
                                    record["float"][role][budget]["selected"]
                                ),
                                "selected_set_jaccard": float(
                                    record["arms"][arm]["by_role"][role][budget]["jaccard"]
                                ),
                                "exact_set_match": float(
                                    record["arms"][arm]["by_role"][role][budget][
                                        "exact_match"
                                    ]
                                ),
                                "retained_mass_quant": float(
                                    record["arms"][arm]["by_role"][role][budget]["mass"]
                                ),
                                "retained_mass_float": float(
                                    record["float"][role][budget]["mass"]
                                ),
                                "retained_mass_delta": float(
                                    record["arms"][arm]["by_role"][role][budget][
                                        "mass_delta"
                                    ]
                                ),
                                "selected_ndcg_quant": float(
                                    record["arms"][arm]["by_role"][role][budget]["ndcg"]
                                ),
                                "selected_ndcg_float": float(
                                    record["float"][role][budget]["ndcg"]
                                ),
                                "selected_ndcg_delta": float(
                                    record["arms"][arm]["by_role"][role][budget][
                                        "ndcg_delta"
                                    ]
                                ),
                                "score_rank_spearman": float(
                                    record["arms"][arm][f"spearman_{role}"]
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
    pins_after = verify_pins(prereg, root=root)
    pins_ok = bool(pins_before["pins_unchanged"] and pins_after["pins_unchanged"])
    isolation_ok = bool(
        isolation["opened_only_val_npz"]
        and not isolation["test_npz_opened"]
        and not isolation["train_npz_opened"]
    )
    int32_ok = bool(
        bounds["all_fit_int32"]
        and int32_observed["gm_wv_max_abs"] <= INT32_MAX
        and int32_observed["prompt_wt_max_abs"] <= INT32_MAX
    )
    decision = evaluate_decision(
        summary,
        pins_ok=pins_ok,
        isolation_ok=isolation_ok,
        int32_ok=int32_ok,
        budgets=budgets,
    )

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "report_stem": REPORT_STEM,
        "date": DATE,
        "experiment": "selector_int8_numerical_audit",
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
            "primary_arm": PRIMARY_ARM,
            "descriptive_control": CONTROL_ARM,
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
            "overall_by_role": summary["overall_by_role"],
            "prompt_utility_gap_delta": summary["prompt_utility_gap_delta"],
            "score_rank_spearman": summary["score_rank_spearman"],
            "seed_overall": summary["seed_overall"],
        },
        "control_arm_is_descriptive_only": True,
        "control_arm_note": (
            "FULL_WT_WV_Q8_A8 is a descriptive control, is not the proposed FPGA "
            "partition and cannot determine the decision."
        ),
        "grounding_prompt_utility_note": (
            "Grounding same-image Pointing prompt is descriptive only: the targets are "
            "nested and the paired prompt is not a valid negative; only Pointing enters "
            "the K48 prompt-utility gate."
        ),
        "decision": decision,
        "artifacts": {
            "root": str(artifact_root),
            "per_seed": bundle_meta,
            "wt_exported_in_primary_bundle": False,
        },
        "limitations": list(prereg.get("limitations", []))
        + [
            "Float L2 normalisation, reciprocal square root, min-max and row "
            "normalisation are software golden operations in this audit.",
            "This audit is an engineering-equivalence check; it does not prove a "
            "complete fixed-point FPGA scorer, HLS/bit-level equivalence, FPGA latency, "
            "end-to-end accuracy or publication readiness.",
            "The Pointing K48 prompt-utility gate uses the same-image other-task prompt; "
            "Grounding is reported descriptively only because its targets are nested.",
            "The validation split has already been used in development; this is not a "
            "final or CV-Bench accuracy result.",
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
    decision = bundle["decision"]
    metrics = bundle["metrics"]
    lines: list[str] = []
    lines.append(f"# Selector INT8 数值审计 — {REPORT_STEM}")
    lines.append("")
    lines.append(f"- 决策：**{decision['decision']}**")
    lines.append(
        "- 范围：CPU-only、离线、validation-only、只读；未训练、未调参、未加载 4B、未读原图、未访问 internal test。"
    )
    lines.append(
        f"- 主臂：`{PRIMARY_ARM}`；描述性对照：`{CONTROL_ARM}`（不参与决策）。"
    )
    lines.append(
        f"- 聚合：每记录跨 3 seed 取中位数，再按 {bundle['aggregation']['n_images']} 张独立图像报告 mean 与 2000 次 percentile bootstrap 95% CI；不使用 p 值。"
    )
    lines.append("")
    lines.append("## 冻结判据")
    lines.append("")
    lines.append("| 判据 | 通过 |")
    lines.append("|---|---|")
    for key, value in decision["checks"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.append("")
    lines.append("## 主臂量化 vs float（overall）")
    lines.append("")
    lines.append("| budget | Jaccard mean [95% CI] | exact-set rate | mass delta | NDCG delta |")
    lines.append("|---|---|---|---|---|")
    for budget in BUDGETS:
        over = metrics["overall"][PRIMARY_ARM][str(budget)]
        lines.append(
            "| {} | {} [{}, {}] | {} | {} | {} |".format(
                budget,
                _fmt(over["jaccard"]["mean"]),
                _fmt(over["jaccard"]["ci_low"]),
                _fmt(over["jaccard"]["ci_high"]),
                _fmt(over["exact_match"]["mean"]),
                _fmt(over["mass_delta"]["mean"]),
                _fmt(over["ndcg_delta"]["mean"]),
            )
        )
    lines.append("")
    lines.append("## K48 Pointing prompt-utility retained-mass gap delta (quant - float)")
    lines.append("")
    gap = metrics["prompt_utility_gap_delta"][PRIMARY_ARM]["48"]["mass"]
    pt = gap["per_task"]["pointing"]
    lines.append(
        f"- Pointing mean={_fmt(pt['mean'])} CI=[{_fmt(pt['ci_low'])}, {_fmt(pt['ci_high'])}]"
    )
    gd = gap["per_task"]["grounding"]
    lines.append(
        f"- Grounding（仅描述，嵌套目标不构成有效负样本）mean={_fmt(gd['mean'])} "
        f"CI=[{_fmt(gd['ci_low'])}, {_fmt(gd['ci_high'])}]"
    )
    lines.append("")
    lines.append("## INT32 解析最坏界")
    lines.append("")
    lines.append("| dot | length | |acc| bound | fits INT32 |")
    lines.append("|---|---|---|---|")
    for name, entry in bundle["int32"]["analytical"]["lengths"].items():
        lines.append(
            f"| {name} | {entry['length']} | {entry['bound']} | {entry['fits_int32']} |"
        )
    lines.append("")
    lines.append(
        f"- 观测最大 |INT32 累加|：group_mean×Wv={bundle['int32']['observed_max_abs']['gm_wv_max_abs']}，"
        f"prompt×Wt={bundle['int32']['observed_max_abs']['prompt_wt_max_abs']}，"
        f"no_overflow={bundle['int32']['no_overflow']}。"
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
    lines.append("## 描述性对照臂")
    lines.append("")
    lines.append("| budget | Jaccard | exact-set rate | mass delta | NDCG delta |")
    lines.append("|---|---|---|---|---|")
    for budget in BUDGETS:
        over = metrics["overall"][CONTROL_ARM][str(budget)]
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                budget,
                _fmt(over["jaccard"]["mean"]),
                _fmt(over["exact_match"]["mean"]),
                _fmt(over["mass_delta"]["mean"]),
                _fmt(over["ndcg_delta"]["mean"]),
            )
        )
    lines.append("")
    lines.append("## 限制")
    lines.append("")
    for note in bundle["limitations"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append(
        "> 明确说明：float L2 归一化/倒数平方根/min-max 仍是本审计的 software golden；"
        "本审计尚未证明完整的定点 FPGA scorer，也不作精度、延迟、端到端加速或发表声明。"
    )
    lines.append("")
    return "\n".join(lines)


def harness_payload(bundle: Mapping[str, Any], files: Mapping[str, str]) -> dict[str, Any]:
    decision = bundle["decision"]
    metrics = bundle["metrics"]
    pointing_gap = metrics["prompt_utility_gap_delta"][PRIMARY_ARM]["48"]["mass"][
        "per_task"
    ]["pointing"]
    return {
        "schema_version": SCHEMA_VERSION,
        "report_stem": REPORT_STEM,
        "decision": decision["decision"],
        "primary_arm": PRIMARY_ARM,
        "descriptive_control": CONTROL_ARM,
        "checks": decision["checks"],
        "overall": {
            str(budget): {
                "jaccard": metrics["overall"][PRIMARY_ARM][str(budget)]["jaccard"],
                "exact_set_match_rate": metrics["overall"][PRIMARY_ARM][str(budget)][
                    "exact_match"
                ],
                "mass_delta": metrics["overall"][PRIMARY_ARM][str(budget)]["mass_delta"],
                "ndcg_delta": metrics["overall"][PRIMARY_ARM][str(budget)]["ndcg_delta"],
            }
            for budget in BUDGETS
        },
        "pointing_k48_prompt_utility_gap_delta": pointing_gap,
        "seed_k48_mean_jaccard": decision["seed_k48_mean_jaccard"],
        "int32": {
            "analytical": bundle["int32"]["analytical"]["lengths"],
            "observed_max_abs": bundle["int32"]["observed_max_abs"],
            "no_overflow": bundle["int32"]["no_overflow"],
        },
        "pins_unchanged": bundle["pins_unchanged"],
        "isolation": bundle["isolation"],
        "limitations": bundle["limitations"],
        "files": dict(files),
    }


def _write_reports(
    bundle: Mapping[str, Any], csv_rows: Sequence[Mapping[str, Any]], *, root: Path
) -> dict[str, str]:
    json_path = root / REPORTS_DIR / f"{REPORT_STEM}.json"
    md_path = root / REPORTS_DIR / f"{REPORT_STEM}.md"
    csv_path = root / REPORTS_DIR / f"{REPORT_STEM}-records.csv"
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
