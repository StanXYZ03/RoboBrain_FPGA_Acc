#!/usr/bin/env python3
"""Bit-exact integer Selector scorer audit (``INTEGER_SCORER_Q16``).

This is a **zero-training, preregistered, validation-only, CPU-only, read-only**
numerical audit.  It answers exactly one engineering question: may the accepted
per-tensor-Wv selector chain be re-expressed as a pure-integer scorer -- one that
contains **no float operation after input quantisation** -- without changing the
selected token groups enough to matter for an FPGA implementation?

Three deterministic chains share the frozen A8 activation quantiser and the
frozen per-vector INT8 prompt query:

* ``PER_TENSOR_FLOAT_NORM`` (reference, the accepted prior chain).  Per-tensor
  INT8 Wv, exact INT32 projection, float row-L2 normalisation, normalised
  visual-query dot, float min-max combine with ``lambda_semantic=0.5`` and a
  uint16 Q0.15 pack.
* ``FLOAT_REFERENCE`` (original float).  The frozen float scorer with the
  original float ``Wt``/``Wv``.
* ``INTEGER_SCORER_Q16`` (candidate).  The whole scorer after input quantisation
  is integer-only:

  1. visual projection ``p[G,128]`` from ``group_mean_q`` (INT8) and per-tensor
     ``Wv_q`` (INT8) with exact INT32 sums;
  2. signed ``dot = sum_d p[d]*query_q[d]`` accumulated in INT64 and
     ``norm2 = sum_d p[d]^2`` accumulated in nonnegative INT64;
  3. ``semantic_q16 = round_half_away((dot << 16) / floor_sqrt(norm2))`` with an
     exact signed-rational round-half-away division; zero norm maps to zero.
     The activation/Wv/query scales and the query norm are positive common
     factors that cancel in the cosine and in the per-image min-max, so they are
     omitted deliberately;
  4. ``l2_int = floor_sqrt(sum_1024 group_mean_q^2)``;
  5. semantic and L2 terms independently min-max normalised to ``[0, 32767]``
     with an exact rational round-half-away map (all-equal maps to zero);
  6. ``score_q = clamp(round_half_away((semantic_norm_q15 + l2_norm_q15)/2),
     0, 32767)``.

The Python ``int`` type is used only to *prove* fixed hardware widths and to run
the exact rational divisions/square roots; every declared width is also checked
analytically and on the observed values.  Passing is permission to implement
HLS, not evidence of synthesis, latency, task accuracy or end-to-end
acceleration.

Reuse only (no second implementation of split / loader / target / quantiser /
scorer / metric / selection / statistic):

* split / loader:   :func:`evaluation.prompt_projection_distillation.split_sample_ids`
                    and :func:`evaluation.train_pairwise_prompt_selector.load_split_samples`
* spatial targets:  :func:`evaluation.train_spatial_prompt_selector.attach_spatial_targets`
* per-tensor Wv:    :func:`evaluation.selector_wv_scale_regularization.quantize_weight_tensor`
                    and :func:`evaluation.selector_wv_scale_regularization.per_tensor_scale_vector`
* INT8 primitives:  :func:`evaluation.selector_int8_numerical_audit.quantize_activation`,
                    :func:`evaluation.selector_int8_numerical_audit.primary_quantized_query`,
                    :func:`evaluation.selector_int8_numerical_audit.quantized_visual_projection`,
                    :func:`evaluation.selector_int8_numerical_audit.quantized_combined_scores`,
                    :func:`evaluation.selector_int8_numerical_audit.float_reference_scores`,
                    :func:`evaluation.selector_int8_numerical_audit.exact_int_matmul`
* anchor selection: :func:`evaluation.selector_int8_numerical_audit.selected_indices`
* metrics:          :func:`evaluation.prompt_projection_distillation.{jaccard,saliency_mass_retained,ndcg_for_selected_indices,spearman}`
* statistics:       :func:`evaluation.selector_int8_numerical_audit.bootstrap_mean_ci`
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

from evaluation import selector_int8_numerical_audit as base
from evaluation import selector_wv_scale_regularization as wv_audit
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

REPORT_STEM = "2026-10-03-selector-integer-scorer-audit"
DATE = "2026-10-03"
SCHEMA_VERSION = "1.0"

ROOT = Path(__file__).resolve().parents[1]
PREREG_CONFIG_PATH = Path("configs/selector_integer_scorer_audit.preregistered.json")
INDEX_PATH = Path("artifacts/token_selector/prompt_projection_features/index.jsonl")
MANIFEST_PATH = Path("datasets/prompt_projection_clean_inductive_v1.jsonl")
REPORTS_DIR = Path("reports/token_selector")
REPORT_DIR = REPORTS_DIR / REPORT_STEM
LOGS_DIR = REPORTS_DIR / "logs"
ARTIFACT_ROOT = Path("artifacts/token_selector/selector_integer_scorer_audit")

REFERENCE_ARM = "PER_TENSOR_FLOAT_NORM"
CANDIDATE_ARM = "INTEGER_SCORER_Q16"
FLOAT_ARM = "FLOAT_REFERENCE"

SEEDS = (20261003, 20261004, 20261005)
BUDGETS = (48, 64, 96)
TASKS = ("grounding", "pointing")
ROLES = ("correct", "wrong")
LAMBDA_SEMANTIC = 0.5
SEMANTIC_FRACTIONAL_BITS = 16
SCORE_FRACTIONAL_BITS = 15
SEMANTIC_SCALE = 1 << SEMANTIC_FRACTIONAL_BITS
SCORE_Q_MAX = 32767

EMBED_DIM = 1024
QUERY_DIM = 128
ACTIVATION_MAX_ABS = 128
WEIGHT_MAX_ABS = 127
QUERY_MAX_ABS = 128
INT32_MAX = 2_147_483_647
INT64_MAX = 9_223_372_036_854_775_807
UINT16_MAX = 65_535

#: Analytical worst-case magnitudes for the integer chain (arbitrary-precision
#: Python integers; used only to prove the fixed hardware widths).
PROJECTION_BOUND = EMBED_DIM * ACTIVATION_MAX_ABS * WEIGHT_MAX_ABS
DOT_BOUND = QUERY_DIM * PROJECTION_BOUND * QUERY_MAX_ABS
NORM2_BOUND = QUERY_DIM * PROJECTION_BOUND * PROJECTION_BOUND
SHIFTED_NUMERATOR_BOUND = DOT_BOUND << SEMANTIC_FRACTIONAL_BITS
#: ``|dot| <= ||p|| * ||q||`` (Cauchy-Schwarz); ``||q|| < 128*12`` and
#: ``sqrt(norm2)/isqrt(norm2) < 2`` for every ``norm2 >= 1``.
SEMANTIC_Q16_BOUND = 2 * (QUERY_DIM * 12) * SEMANTIC_SCALE
L2_SQUARED_SUM_BOUND = EMBED_DIM * ACTIVATION_MAX_ABS * ACTIVATION_MAX_ABS
L2_INT_BOUND = 4096

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
    "candidate_vs_per_tensor",
    "per_tensor_vs_float",
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


class IntegerScorerAuditError(ValueError):
    """Malformed programmer input or a violated audit contract."""


# ---------------------------------------------------------------------------
# Exact integer primitives (the only new numeric operations)
# ---------------------------------------------------------------------------
def floor_sqrt_int64(value: int) -> int:
    """Exact ``floor(sqrt(value))`` for an integer in ``[0, 2**63)``.

    Implemented with a bit-by-bit integer algorithm so every intermediate value
    provably fits the declared INT64 width; it never touches a float.
    """
    if not isinstance(value, (int, np.integer)):
        raise IntegerScorerAuditError("floor_sqrt_int64 needs an integer")
    value = int(value)
    if value < 0:
        raise IntegerScorerAuditError("floor_sqrt_int64 needs a nonnegative value")
    if value > INT64_MAX:
        raise IntegerScorerAuditError("floor_sqrt_int64 input exceeds INT64")
    if value < 2:
        return value
    remainder = value
    result = 0
    bit = 1 << (((value.bit_length() - 1) // 2) * 2)
    while bit:
        if remainder >= result + bit:
            remainder -= result + bit
            result = (result >> 1) + bit
        else:
            result >>= 1
        bit >>= 2
    return int(result)


def round_half_away_div(numerator: int, denominator: int) -> int:
    """Exact signed rational round-half-away-from-zero ``numerator/denominator``.

    ``round_half_away(0.5)=1``, ``round_half_away(1.5)=2``, ``round_half_away(-0.5)=-1``.
    The denominator must be strictly positive.  This is integer-only: no float
    ever enters the candidate scoring path.
    """
    if not isinstance(numerator, (int, np.integer)) or not isinstance(
        denominator, (int, np.integer)
    ):
        raise IntegerScorerAuditError("round_half_away_div needs integer operands")
    numerator = int(numerator)
    denominator = int(denominator)
    if denominator <= 0:
        raise IntegerScorerAuditError("round_half_away_div denominator must be positive")
    magnitude = (2 * abs(numerator) + denominator) // (2 * denominator)
    return -magnitude if numerator < 0 else magnitude


def minmax_normalize_q15(values: np.ndarray) -> np.ndarray:
    """Exact min-max map to ``[0, 32767]``; an all-equal vector maps to zero."""
    array = np.asarray(values, dtype=np.int64).reshape(-1)
    if array.size == 0:
        raise IntegerScorerAuditError("min-max normalisation needs a non-empty vector")
    low = int(array.min())
    high = int(array.max())
    if high == low:
        return np.zeros(array.shape, dtype=np.int64)
    span = high - low
    return np.asarray(
        [round_half_away_div((int(value) - low) * SCORE_Q_MAX, span) for value in array],
        dtype=np.int64,
    )


def _checked_abs_max(array: np.ndarray) -> int:
    array = np.asarray(array, dtype=np.int64)
    if array.size == 0:
        return 0
    return int(np.max(np.abs(array)))


def integer_chain(
    group_mean_q: Any,
    query_q: Any,
    wv_q: Any,
    grid_h: int,
    grid_w: int,
    *,
    budgets: Sequence[int] = BUDGETS,
) -> dict[str, Any]:
    """Run the frozen integer scorer for one image and one prompt query.

    Raises :class:`IntegerScorerAuditError` if any observed magnitude would not
    fit its declared fixed width, so a silent overflow can never pass unnoticed.
    """
    mean_q = np.asarray(group_mean_q, dtype=np.int64)
    weights = np.asarray(wv_q, dtype=np.int64)
    query = np.asarray(query_q, dtype=np.int64).reshape(-1)
    if mean_q.ndim != 2 or mean_q.shape[1] != EMBED_DIM:
        raise IntegerScorerAuditError(
            f"group_mean_q must be [G, {EMBED_DIM}], got {mean_q.shape}"
        )
    if weights.ndim != 2 or weights.shape[1] != EMBED_DIM:
        raise IntegerScorerAuditError(
            f"wv_q must be [128, {EMBED_DIM}], got {weights.shape}"
        )
    if query.size != weights.shape[0]:
        raise IntegerScorerAuditError("query_q size must match Wv_q output rows")
    for name, array, channel_max in (
        ("group_mean_q", mean_q, ACTIVATION_MAX_ABS),
        ("wv_q", weights, WEIGHT_MAX_ABS),
        ("query_q", query, QUERY_MAX_ABS),
    ):
        if int(np.max(np.abs(array))) > channel_max:
            raise IntegerScorerAuditError(f"{name} exceeds its INT8 magnitude bound")

    # 1. Exact INT32 visual projection p[G,128] = group_mean_q @ Wv_q^T.
    projection_int = base.exact_int_matmul(mean_q, weights.T)
    projection_absmax = _checked_abs_max(projection_int)
    if projection_absmax > INT32_MAX:
        raise IntegerScorerAuditError("projection accumulator exceeds INT32")

    # 2. Signed INT64 dot and nonnegative INT64 norm2 per group.  The explicit
    #    per-row bounds prove the numpy int64 accumulation cannot wrap.
    max_abs_per_row = np.max(np.abs(projection_int), axis=1)
    dot_bound = int(np.max(max_abs_per_row)) * int(np.sum(np.abs(query)))
    norm2_bound = QUERY_DIM * int(np.max(max_abs_per_row)) ** 2
    if dot_bound > INT64_MAX:
        raise IntegerScorerAuditError("semantic dot accumulator exceeds INT64")
    if norm2_bound > INT64_MAX:
        raise IntegerScorerAuditError("semantic norm2 accumulator exceeds INT64")
    dot = projection_int @ query
    norm2 = np.einsum("gd,gd->g", projection_int, projection_int)
    if np.any(norm2 < 0):
        raise IntegerScorerAuditError("norm2 accumulator wrapped negative")
    dot_absmax = _checked_abs_max(dot)
    norm2_max = int(norm2.max()) if norm2.size else 0

    # 3. semantic_q16 = round_half_away((dot << 16) / floor_sqrt(norm2)).
    shifted_absmax = dot_absmax << SEMANTIC_FRACTIONAL_BITS
    if shifted_absmax > INT64_MAX:
        raise IntegerScorerAuditError("semantic shifted numerator exceeds INT64")
    semantic_q16 = np.zeros(mean_q.shape[0], dtype=np.int64)
    for index in range(mean_q.shape[0]):
        if int(norm2[index]) == 0:
            semantic_q16[index] = 0
            continue
        semantic_q16[index] = round_half_away_div(
            int(dot[index]) << SEMANTIC_FRACTIONAL_BITS,
            floor_sqrt_int64(int(norm2[index])),
        )
    semantic_absmax = _checked_abs_max(semantic_q16)
    if semantic_absmax > INT32_MAX:
        raise IntegerScorerAuditError("semantic_q16 exceeds its declared INT32 width")

    # 4. l2_int = floor_sqrt(sum_1024 group_mean_q^2).
    l2_squared_sum = np.einsum("gd,gd->g", mean_q, mean_q)
    l2_squared_max = int(l2_squared_sum.max()) if l2_squared_sum.size else 0
    if l2_squared_max > INT64_MAX:
        raise IntegerScorerAuditError("l2 squared sum exceeds INT64")
    l2_int = np.asarray(
        [floor_sqrt_int64(int(value)) for value in l2_squared_sum], dtype=np.int64
    )
    l2_max = int(l2_int.max()) if l2_int.size else 0

    # 5. Independent exact min-max normalisation to [0, 32767].
    semantic_norm_q15 = minmax_normalize_q15(semantic_q16)
    l2_norm_q15 = minmax_normalize_q15(l2_int)

    # 6. lambda = 0.5 combine, clamped to [0, 32767].
    score_q = np.asarray(
        [
            min(
                max(
                    round_half_away_div(
                        int(semantic_norm_q15[index]) + int(l2_norm_q15[index]), 2
                    ),
                    0,
                ),
                SCORE_Q_MAX,
            )
            for index in range(mean_q.shape[0])
        ],
        dtype=np.uint16,
    )

    selections = {
        int(budget): base.selected_indices(
            score_q.astype(np.float64), int(grid_h), int(grid_w), int(budget)
        )
        for budget in budgets
    }
    return {
        "projection_int": projection_int,
        "dot": dot,
        "norm2": norm2,
        "semantic_q16": semantic_q16,
        "l2_int": l2_int,
        "semantic_norm_q15": semantic_norm_q15,
        "l2_norm_q15": l2_norm_q15,
        "score_q": score_q,
        "selections": selections,
        "bounds": {
            "projection_int_absmax": int(projection_absmax),
            "projection_int_sum": int(projection_int.sum()),
            "dot_absmax": int(dot_absmax),
            "dot_bound": int(dot_bound),
            "norm2_max": int(norm2_max),
            "norm2_bound": int(norm2_bound),
            "shifted_numerator_absmax": int(shifted_absmax),
            "semantic_q16_absmax": int(semantic_absmax),
            "l2_squared_sum_max": int(l2_squared_max),
            "l2_int_max": int(l2_max),
            "score_q_max": int(score_q.max()) if score_q.size else 0,
        },
    }


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
    wv_scale_ref: float,
    target: Any,
    grid_h: int,
    grid_w: int,
    budgets: Sequence[int] = BUDGETS,
) -> dict[str, Any]:
    """Evaluate the integer candidate, the per-tensor reference and the float arm.

    ``wv_q_ref``/``wv_scale_ref`` are the accepted per-tensor Wv quantisation
    (from :func:`evaluation.selector_wv_scale_regularization.quantize_weight_tensor`);
    the same tensor feeds the reference float chain and the integer candidate, so
    the comparison isolates the float-vs-integer scoring change.
    """
    budgets = [int(value) for value in budgets]
    mean = np.asarray(group_mean, dtype=np.float64)
    correct = np.asarray(prompt_correct, dtype=np.float64)
    wrong = np.asarray(prompt_wrong, dtype=np.float64)

    group_mean_q, group_mean_scale = base.quantize_activation(mean)
    group_mean_hat = group_mean_q.astype(np.float64) / group_mean_scale
    l2_hat = np.linalg.norm(group_mean_hat, axis=1)

    # Reference: accepted per-tensor float-normalisation chain.
    ref_scales = wv_audit.per_tensor_scale_vector(wv_scale_ref, np.asarray(Wv).shape[0])
    ref_visual = base.quantized_visual_projection(
        group_mean_q, group_mean_scale, np.asarray(wv_q_ref), ref_scales
    )
    ref_visual_unit = ref_visual["visual_unit"]

    by_role: dict[str, dict[int, dict[str, Any]]] = {}
    spearman_entries: dict[str, dict[str, float]] = {}
    integer_entries: dict[str, dict[str, Any]] = {}
    for role, prompt in (("correct", correct), ("wrong", wrong)):
        query = base.primary_quantized_query(prompt, Wt)
        chain = integer_chain(
            group_mean_q, query["query_q"], wv_q_ref, grid_h, grid_w, budgets=budgets
        )
        integer_entries[role] = chain

        _, _, ref_hat = base.quantized_combined_scores(
            ref_visual_unit, query["query_hat"], l2_hat
        )
        _, _, float_combined = base.float_reference_scores(mean, prompt, Wt, Wv)

        candidate_selection = {
            int(budget): {
                "selected": chain["selections"][int(budget)],
                "mass": float(
                    saliency_mass_retained(
                        chain["selections"][int(budget)],
                        np.asarray(target, dtype=np.float64).reshape(-1),
                    )
                ),
                "ndcg": float(
                    ndcg_for_selected_indices(
                        chain["selections"][int(budget)],
                        chain["score_q"].astype(np.float64),
                        np.asarray(target, dtype=np.float64).reshape(-1),
                        min(int(budget), int(np.asarray(target).size)),
                    )
                ),
            }
            for budget in budgets
        }
        ref_selection = _selection_metrics(ref_hat, target, grid_h, grid_w, budgets)
        float_selection = _selection_metrics(
            float_combined, target, grid_h, grid_w, budgets
        )

        role_metrics: dict[int, dict[str, Any]] = {}
        for budget in budgets:
            budget = int(budget)
            cand_entry = candidate_selection[budget]
            ref_entry = ref_selection[budget]
            float_entry = float_selection[budget]
            cand_set = set(cand_entry["selected"])
            ref_set = set(ref_entry["selected"])
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
                "cand_selected": tuple(int(value) for value in cand_entry["selected"]),
                "ref_selected": tuple(int(value) for value in ref_entry["selected"]),
                "float_selected": tuple(int(value) for value in float_entry["selected"]),
                "score_q": chain["score_q"],
            }
        by_role[role] = role_metrics
        spearman_entries[role] = {
            "candidate_vs_per_tensor": float(
                spearman(chain["score_q"].astype(np.float64), ref_hat)
            ),
            "per_tensor_vs_float": float(spearman(ref_hat, float_combined)),
            "candidate_vs_float": float(
                spearman(chain["score_q"].astype(np.float64), float_combined)
            ),
        }

    return {
        "by_role": by_role,
        "spearman": spearman_entries,
        "integer": integer_entries,
        "group_mean_scale": float(group_mean_scale),
        "group_mean_q_absmax": int(np.max(np.abs(group_mean_q))),
        "wv_q_absmax": int(np.max(np.abs(np.asarray(wv_q_ref)))),
        "wv_scale": float(wv_scale_ref),
        "group_count": int(mean.shape[0]),
    }


# ---------------------------------------------------------------------------
# Aggregation and bootstrap
# ---------------------------------------------------------------------------
def _median_across_seeds(values: Sequence[float]) -> float:
    if not values:
        raise IntegerScorerAuditError("cannot take a median of zero seed values")
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
    the rank correlations.  Frozen aggregation: median across the three seeds per
    record, mean across the two task records per image, then a 2000-resample
    percentile bootstrap over the 38 independent images.
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
            score_rank_spearman[comparison][role] = {
                "overall": base.bootstrap_mean_ci(
                    [float(np.mean(values)) for values in per_image.values()]
                ),
                "per_task": {
                    task: base.bootstrap_mean_ci(
                        [
                            float(spearman_median[(role, comparison)][sid])
                            for sid in sample_ids
                            if task_of[sid] == task
                        ]
                    )
                    for task in TASKS
                },
            }

    # --- prompt-utility gaps: (correct - wrong) candidate vs float ---------
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
            elif reference == "per_tensor":
                gap_ref = (
                    record_median[("correct", int(budget), ref_key)][sid]
                    - record_median[("wrong", int(budget), ref_key)][sid]
                )
            else:
                raise IntegerScorerAuditError(f"unknown reference arm {reference!r}")
            per_image.setdefault(image_of[sid], []).append(float(gap_cand - gap_ref))
        return {image: float(np.mean(values)) for image, values in per_image.items()}

    prompt_utility_gap_delta: dict[str, Any] = {}
    for reference in ("float", "per_tensor"):
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
                float(
                    raw_by_seed[seed][sid]["by_role"]["correct"][48][
                        "cand_vs_ref_jaccard"
                    ]
                )
            )
        seed_k48[str(seed)] = float(
            np.mean([np.mean(values) for values in per_image.values()])
        )

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
        "seed_k48_candidate_vs_per_tensor_jaccard": seed_k48,
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
    fixture_ok: bool,
    width_ok: bool,
    budgets: Sequence[int] = BUDGETS,
) -> dict[str, Any]:
    """Frozen ``INTEGER_SCORER_NUMERICS_PASS`` / ``..._NOT_ESTABLISHED`` rule."""
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
        summary["score_rank_spearman"]["candidate_vs_per_tensor"]["correct"]["overall"]
    )
    pointing_gap = _bucket_mean(
        summary["prompt_utility_gap_delta"]["float"]["48"]["mass"]["per_task"]["pointing"]
    )
    seed_k48 = {
        str(seed): float(value)
        for seed, value in sorted(
            summary["seed_k48_candidate_vs_per_tensor_jaccard"].items()
        )
    }

    checks = {
        "pins_isolation_fixture_width_pass": bool(
            pins_ok and isolation_ok and fixture_ok and width_ok
        ),
        "candidate_vs_per_tensor_mean_jaccard_each_budget_at_least_0.98": all(
            value is not None and value >= 0.98 for value in jaccard_values.values()
        ),
        "candidate_vs_per_tensor_mean_replacement_count_each_budget_at_most_1.0": all(
            value is not None and value <= 1.0 for value in replacement_values.values()
        ),
        "candidate_vs_per_tensor_mean_score_spearman_at_least_0.995": bool(
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
        "each_seed_k48_candidate_vs_per_tensor_mean_jaccard_at_least_0.97": all(
            value >= 0.97 for value in seed_k48.values()
        ),
    }
    passed = all(checks.values())
    return {
        "decision": (
            "INTEGER_SCORER_NUMERICS_PASS"
            if passed
            else "INTEGER_SCORER_NUMERICS_NOT_ESTABLISHED"
        ),
        "pass": bool(passed),
        "checks": checks,
        "candidate_arm": CANDIDATE_ARM,
        "reference_arm": REFERENCE_ARM,
        "overall_mean_candidate_vs_per_tensor_jaccard": {
            str(b): jaccard_values[b] for b in budgets
        },
        "overall_mean_candidate_vs_per_tensor_replacement_count": {
            str(b): replacement_values[b] for b in budgets
        },
        "overall_mean_candidate_vs_per_tensor_score_spearman": spearman_mean,
        "overall_mean_candidate_vs_float_mass_delta": {
            str(b): mass_delta_values[b] for b in budgets
        },
        "overall_mean_candidate_vs_float_ndcg_delta": {
            str(b): ndcg_delta_values[b] for b in budgets
        },
        "pointing_k48_prompt_utility_gap_delta_candidate_vs_float": pointing_gap,
        "seed_k48_candidate_vs_per_tensor_mean_jaccard": seed_k48,
        "exact_set_rate_is_descriptive_only": True,
        "rule": (
            "INTEGER_SCORER_NUMERICS_PASS iff pins/isolation/fixture/width checks pass, "
            "every K candidate-vs-per-tensor mean Jaccard >= 0.98, mean replacement "
            "count <= 1.0, mean score Spearman >= 0.995, every K candidate-vs-float "
            "mass delta >= -0.01 and NDCG delta >= -0.01, K48 Pointing prompt-utility "
            "gap delta >= -0.01 and every seed K48 candidate-vs-per-tensor mean "
            "Jaccard >= 0.97"
        ),
    }


# ---------------------------------------------------------------------------
# Hardware width / operation contract
# ---------------------------------------------------------------------------
def declared_width_contract() -> dict[str, Any]:
    """Analytical integer width and operation contract for the candidate chain."""
    operations = [
        {
            "op": "visual_projection_accumulator",
            "inputs": "group_mean_q INT8 x Wv_q INT8",
            "accumulation": f"INT32 exact, length {EMBED_DIM}",
            "width": "INT32",
            "max_value": INT32_MAX,
            "analytical_bound": int(PROJECTION_BOUND),
            "fits": bool(PROJECTION_BOUND <= INT32_MAX),
        },
        {
            "op": "semantic_dot_accumulator",
            "inputs": "projection INT32 x query_q INT8",
            "accumulation": f"INT64 exact, length {QUERY_DIM}",
            "width": "INT64",
            "max_value": INT64_MAX,
            "analytical_bound": int(DOT_BOUND),
            "fits": bool(DOT_BOUND <= INT64_MAX),
        },
        {
            "op": "semantic_norm2_accumulator",
            "inputs": "projection INT32 squared",
            "accumulation": f"INT64 nonnegative, length {QUERY_DIM}",
            "width": "INT64",
            "max_value": INT64_MAX,
            "analytical_bound": int(NORM2_BOUND),
            "fits": bool(NORM2_BOUND <= INT64_MAX),
        },
        {
            "op": "semantic_shifted_numerator",
            "inputs": f"dot << {SEMANTIC_FRACTIONAL_BITS}",
            "accumulation": "INT64 exact",
            "width": "INT64",
            "max_value": INT64_MAX,
            "analytical_bound": int(SHIFTED_NUMERATOR_BOUND),
            "fits": bool(SHIFTED_NUMERATOR_BOUND <= INT64_MAX),
        },
        {
            "op": "semantic_q16",
            "inputs": "round_half_away((dot << 16) / floor_sqrt(norm2))",
            "accumulation": "exact signed rational, integer-only",
            "width": "INT32",
            "max_value": INT32_MAX,
            "analytical_bound": int(SEMANTIC_Q16_BOUND),
            "fits": bool(SEMANTIC_Q16_BOUND <= INT32_MAX),
        },
        {
            "op": "l2_squared_sum",
            "inputs": "group_mean_q INT8 squared over 1024",
            "accumulation": "INT64 nonnegative",
            "width": "INT64",
            "max_value": INT64_MAX,
            "analytical_bound": int(L2_SQUARED_SUM_BOUND),
            "fits": bool(L2_SQUARED_SUM_BOUND <= INT64_MAX),
        },
        {
            "op": "l2_int",
            "inputs": "floor_sqrt(sum_1024 group_mean_q^2)",
            "accumulation": "exact integer square root",
            "width": "INT32",
            "max_value": INT32_MAX,
            "analytical_bound": int(L2_INT_BOUND),
            "fits": bool(L2_INT_BOUND <= INT32_MAX),
        },
        {
            "op": "semantic_norm_q15",
            "inputs": "exact min-max of semantic_q16",
            "accumulation": "exact rational round-half-away",
            "width": "UINT16",
            "max_value": SCORE_Q_MAX,
            "analytical_bound": SCORE_Q_MAX,
            "fits": True,
        },
        {
            "op": "l2_norm_q15",
            "inputs": "exact min-max of l2_int",
            "accumulation": "exact rational round-half-away",
            "width": "UINT16",
            "max_value": SCORE_Q_MAX,
            "analytical_bound": SCORE_Q_MAX,
            "fits": True,
        },
        {
            "op": "score_q",
            "inputs": "round_half_away((semantic_norm_q15 + l2_norm_q15)/2)",
            "accumulation": "exact rational, clamped [0, 32767]",
            "width": "UINT16",
            "max_value": SCORE_Q_MAX,
            "analytical_bound": SCORE_Q_MAX,
            "fits": True,
        },
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_arm": CANDIDATE_ARM,
        "reference_arm": REFERENCE_ARM,
        "semantic_fractional_bits": SEMANTIC_FRACTIONAL_BITS,
        "score_fractional_bits": SCORE_FRACTIONAL_BITS,
        "lambda_semantic": LAMBDA_SEMANTIC,
        "float_after_input_quantization": False,
        "rounding": "round_half_away_from_zero",
        "sqrt": "floor integer square root (bit-by-bit, INT64 input)",
        "division": "exact signed rational round-half-away-from-zero",
        "frozen_constants": {
            "activation_levels": 128,
            "weight_levels": 127,
            "embed_dim": EMBED_DIM,
            "query_dim": QUERY_DIM,
            "int32_max": INT32_MAX,
            "int64_max": INT64_MAX,
            "score_q_max": SCORE_Q_MAX,
        },
        "operations": operations,
        "analytical_all_fit": all(entry["fits"] for entry in operations),
        "omitted_positive_common_factors": [
            "group_mean activation scale",
            "per-tensor Wv scale",
            "per-vector query scale",
            "query L2 norm",
        ],
    }


def finalize_width_contract(
    contract: Mapping[str, Any], observed: Mapping[str, int]
) -> dict[str, Any]:
    """Attach observed maxima and the observed-fit verdict to the contract."""
    result = json.loads(json.dumps(contract))
    observed_map = {
        "visual_projection_accumulator": observed.get("projection_int_absmax", 0),
        "semantic_dot_accumulator": observed.get("dot_absmax", 0),
        "semantic_norm2_accumulator": observed.get("norm2_max", 0),
        "semantic_shifted_numerator": observed.get("shifted_numerator_absmax", 0),
        "semantic_q16": observed.get("semantic_q16_absmax", 0),
        "l2_squared_sum": observed.get("l2_squared_sum_max", 0),
        "l2_int": observed.get("l2_int_max", 0),
        "semantic_norm_q15": observed.get("semantic_norm_max", 0),
        "l2_norm_q15": observed.get("l2_norm_max", 0),
        "score_q": observed.get("score_q_max", 0),
    }
    observed_ok = True
    for entry in result["operations"]:
        value = int(observed_map.get(entry["op"], 0))
        entry["observed_max"] = value
        entry["observed_fits"] = bool(value <= int(entry["max_value"]))
        observed_ok = observed_ok and entry["observed_fits"]
    result["observed_maxima"] = {key: int(value) for key, value in observed.items()}
    result["observed_all_fit"] = bool(observed_ok)
    result["all_fit_declared_widths"] = bool(
        result["analytical_all_fit"] and observed_ok
    )
    return result


# ---------------------------------------------------------------------------
# Preregistration pins and isolation
# ---------------------------------------------------------------------------
def load_prereg(path: Path = PREREG_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_pins(prereg: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    """Recompute every frozen input, checkpoint and accepted Wv bundle SHA256."""
    base_result = base.verify_pins(prereg, root=root)
    results = dict(base_result["per_input"])
    mismatches = list(base_result["mismatches"])
    for bundle in prereg.get("accepted_wv_bundles", []):
        seed = int(bundle["seed"])
        for kind, key, hash_key in (
            ("accepted_wv_int8", "wv_path", "wv_sha256"),
            ("accepted_wv_scale", "scale_path", "scale_sha256"),
        ):
            rel = str(bundle[key])
            path = Path(root) / rel
            actual = sha256_file(path) if path.is_file() else None
            ok = actual == str(bundle[hash_key])
            results[rel] = {
                "kind": kind,
                "seed": seed,
                "expected": str(bundle[hash_key]),
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


# ---------------------------------------------------------------------------
# Compact bit-exact fixtures + accepted-bundle reference
# ---------------------------------------------------------------------------
def load_accepted_wv_bundle(root: Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Load the accepted per-tensor Wv INT8 tensor plus its float32 scale."""
    wv_path = Path(root) / str(bundle["wv_path"])
    scale_path = Path(root) / str(bundle["scale_path"])
    wv_q = np.frombuffer(wv_path.read_bytes(), dtype=np.int8).reshape(128, EMBED_DIM)
    scale = float(np.frombuffer(scale_path.read_bytes(), dtype="<f4")[0])
    return {
        "wv_q": wv_q.astype(np.int64),
        "scale_float32": scale,
        "wv_path": str(bundle["wv_path"]),
        "wv_sha256": str(bundle["wv_sha256"]),
        "scale_path": str(bundle["scale_path"]),
        "scale_sha256": str(bundle["scale_sha256"]),
    }


def _fixture_payload(
    sample: Mapping[str, Any],
    *,
    Wt: np.ndarray,
    wv_q: np.ndarray,
    budgets: Sequence[int] = BUDGETS,
) -> dict[str, Any]:
    """Bit-exact integer fixture for one record (no raw image, no Wv copy)."""
    group_mean = np.asarray(sample["group_mean"], dtype=np.float64)
    group_mean_q, _ = base.quantize_activation(group_mean)
    grid_h = int(sample["grid_h"])
    grid_w = int(sample["grid_w"])
    payload: dict[str, Any] = {
        "group_mean_q": group_mean_q.astype(np.int8),
        "grid_h": np.asarray(grid_h, dtype=np.int64),
        "grid_w": np.asarray(grid_w, dtype=np.int64),
        "sample_id": np.asarray(str(sample["sample_id"])),
    }
    for role, prompt in (
        ("correct", np.asarray(sample["prompt_embeddings"], dtype=np.float64)),
        ("wrong", np.asarray(sample["wrong_prompt_embeddings"], dtype=np.float64)),
    ):
        query = base.primary_quantized_query(prompt, Wt)
        chain = integer_chain(
            group_mean_q, query["query_q"], wv_q, grid_h, grid_w, budgets=budgets
        )
        payload[f"query_q_{role}"] = query["query_q"].astype(np.int8)
        payload[f"semantic_q16_{role}"] = chain["semantic_q16"].astype(np.int64)
        payload[f"l2_int_{role}"] = chain["l2_int"].astype(np.int64)
        payload[f"semantic_norm_q15_{role}"] = chain["semantic_norm_q15"].astype(np.uint16)
        payload[f"l2_norm_q15_{role}"] = chain["l2_norm_q15"].astype(np.uint16)
        payload[f"score_q_{role}"] = chain["score_q"].astype(np.uint16)
        payload[f"dot_{role}"] = chain["dot"].astype(np.int64)
        payload[f"norm2_{role}"] = chain["norm2"].astype(np.int64)
        payload[f"projection_int_sum_{role}"] = np.asarray(
            chain["bounds"]["projection_int_sum"], dtype=np.int64
        )
        payload[f"projection_int_max_abs_{role}"] = np.asarray(
            chain["bounds"]["projection_int_absmax"], dtype=np.int64
        )
        for budget in budgets:
            payload[f"selected_{role}_k{int(budget)}"] = np.asarray(
                chain["selections"][int(budget)], dtype=np.int32
            )
    return payload


def validate_fixture_roundtrip(
    out_dir: Path,
    *,
    wv_q: np.ndarray,
    fixtures: Sequence[Mapping[str, Any]],
    budgets: Sequence[int] = BUDGETS,
) -> dict[str, Any]:
    """Reload each fixture and replay the integer chain byte-exactly."""
    out_dir = Path(out_dir)
    fixture_checks: list[dict[str, Any]] = []
    all_ok = True
    for sample in fixtures:
        name = f"fixture_{sample['task']}.npz"
        with np.load(out_dir / name, allow_pickle=False) as archive:
            group_mean_q = archive["group_mean_q"].astype(np.int64)
            grid_h = int(archive["grid_h"])
            grid_w = int(archive["grid_w"])
            role_checks: dict[str, Any] = {}
            for role in ROLES:
                query_q = archive[f"query_q_{role}"].astype(np.int64)
                chain = integer_chain(
                    group_mean_q, query_q, wv_q, grid_h, grid_w, budgets=budgets
                )
                role_ok = bool(
                    np.array_equal(
                        chain["semantic_q16"], archive[f"semantic_q16_{role}"].astype(np.int64)
                    )
                    and np.array_equal(
                        chain["l2_int"], archive[f"l2_int_{role}"].astype(np.int64)
                    )
                    and np.array_equal(
                        chain["semantic_norm_q15"],
                        archive[f"semantic_norm_q15_{role}"].astype(np.int64),
                    )
                    and np.array_equal(
                        chain["l2_norm_q15"],
                        archive[f"l2_norm_q15_{role}"].astype(np.int64),
                    )
                    and np.array_equal(
                        chain["score_q"], archive[f"score_q_{role}"].astype(np.uint16)
                    )
                    and int(chain["bounds"]["projection_int_sum"])
                    == int(archive[f"projection_int_sum_{role}"])
                    and int(chain["bounds"]["projection_int_absmax"])
                    == int(archive[f"projection_int_max_abs_{role}"])
                )
                for budget in budgets:
                    budget = int(budget)
                    role_ok = role_ok and bool(
                        np.array_equal(
                            np.asarray(chain["selections"][budget], dtype=np.int64),
                            archive[f"selected_{role}_k{budget}"].astype(np.int64),
                        )
                    )
                role_checks[role] = {"bit_exact": bool(role_ok)}
                all_ok = all_ok and role_ok
        fixture_checks.append(
            {
                "task": str(sample["task"]),
                "sample_id": str(sample["sample_id"]),
                "roles": role_checks,
            }
        )
    return {"fixtures": fixture_checks, "ok": bool(all_ok)}


def build_seed_bundle(
    seed: int,
    *,
    Wt: np.ndarray,
    Wv: np.ndarray,
    fixtures: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    accepted_bundle: Mapping[str, Any],
    out_dir: Path,
    budgets: Sequence[int] = BUDGETS,
) -> dict[str, Any]:
    """Write one compact per-seed fixture bundle and validate its round-trip.

    The accepted per-tensor Wv INT8 tensor is **referenced** (path + pinned
    SHA256), never duplicated into the fixture files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wv_q_expected, wv_scale_expected = wv_audit.quantize_weight_tensor(Wv)
    wv_q = np.asarray(accepted_bundle["_wv_q"], dtype=np.int64)
    if not np.array_equal(wv_q, wv_q_expected):
        raise IntegerScorerAuditError(
            f"accepted Wv bundle for seed {seed} disagrees with the quantiser"
        )
    if float(np.float32(wv_scale_expected)) != float(accepted_bundle["_scale_float32"]):
        raise IntegerScorerAuditError(
            f"accepted Wv scale for seed {seed} disagrees with the quantiser"
        )

    fixture_files: list[dict[str, Any]] = []
    for sample in fixtures:
        payload = _fixture_payload(sample, Wt=Wt, wv_q=wv_q, budgets=budgets)
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
                "bytes": (out_dir / name).stat().st_size,
                "sha256": sha256_file(out_dir / name),
            }
        )

    roundtrip = validate_fixture_roundtrip(
        out_dir, wv_q=wv_q, fixtures=fixtures, budgets=budgets
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "candidate_arm": CANDIDATE_ARM,
        "reference_arm": REFERENCE_ARM,
        "source_checkpoint": {
            "path": str(checkpoint["path"]),
            "sha256": str(checkpoint["sha256"]),
        },
        "wv_reference": {
            "note": "the accepted per-tensor Wv INT8 tensor is referenced, not duplicated",
            "path": str(accepted_bundle["wv_path"]),
            "sha256": str(accepted_bundle["wv_sha256"]),
            "scale_path": str(accepted_bundle["scale_path"]),
            "scale_sha256": str(accepted_bundle["scale_sha256"]),
            "shape": [int(wv_q.shape[0]), int(wv_q.shape[1])],
            "dtype": "int8",
            "q_absmax": int(np.max(np.abs(wv_q))),
            "scale_value_float64": float(wv_scale_expected),
            "scale_value_float32": float(np.float32(wv_scale_expected)),
            "layout": "row-major [128, 1024] int8, output row = Wv row",
        },
        "activation_quantization": (
            "s=128/amax; q=clamp(round_half_away(x*s),-128,127); x_hat=q/s"
        ),
        "integer_contract": "integer_width_contract.json (parent artifact directory)",
        "rounding": "round_half_away_from_zero",
        "sqrt": "floor integer square root",
        "fixtures": fixture_files,
        "roundtrip": roundtrip,
        "files": sorted(["metadata.json"] + [entry["file"] for entry in fixture_files]),
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


def _observed_from_chain(observed: dict[str, int], chain: Mapping[str, Any]) -> None:
    bounds = chain["bounds"]
    observed["projection_int_absmax"] = max(
        observed.get("projection_int_absmax", 0), int(bounds["projection_int_absmax"])
    )
    observed["dot_absmax"] = max(
        observed.get("dot_absmax", 0), int(bounds["dot_absmax"])
    )
    observed["norm2_max"] = max(
        observed.get("norm2_max", 0), int(bounds["norm2_max"])
    )
    observed["shifted_numerator_absmax"] = max(
        observed.get("shifted_numerator_absmax", 0),
        int(bounds["shifted_numerator_absmax"]),
    )
    observed["semantic_q16_absmax"] = max(
        observed.get("semantic_q16_absmax", 0), int(bounds["semantic_q16_absmax"])
    )
    observed["l2_squared_sum_max"] = max(
        observed.get("l2_squared_sum_max", 0), int(bounds["l2_squared_sum_max"])
    )
    observed["l2_int_max"] = max(observed.get("l2_int_max", 0), int(bounds["l2_int_max"]))
    observed["semantic_norm_max"] = max(
        observed.get("semantic_norm_max", 0),
        int(np.max(chain["semantic_norm_q15"])) if np.asarray(chain["semantic_norm_q15"]).size else 0,
    )
    observed["l2_norm_max"] = max(
        observed.get("l2_norm_max", 0),
        int(np.max(chain["l2_norm_q15"])) if np.asarray(chain["l2_norm_q15"]).size else 0,
    )
    observed["score_q_max"] = max(
        observed.get("score_q_max", 0), int(bounds["score_q_max"])
    )


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
    """Run the full validation-only integer scorer audit."""
    started = time.time()
    root = Path(root)
    prereg = load_prereg(root / prereg_config_path)
    identity = base.verify_prereguration_identity(
        prereg, root=root, config_path=prereg_config_path
    )
    if not identity["consistent"]:
        raise IntegerScorerAuditError("preregistration identity/authority-prompt mismatch")
    pins_before = verify_pins(prereg, root=root)
    if not pins_before["pins_unchanged"]:
        raise IntegerScorerAuditError(
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
        raise IntegerScorerAuditError(f"internal test NPZ were opened: {test_opened}")
    if train_opened:
        raise IntegerScorerAuditError(
            f"train NPZ were opened during a val-only audit: {train_opened}"
        )
    val_aggregate_ok = val_aggregate == str(
        split_rule["expected_val_npz_aggregate_sha256"]
    )
    if not val_aggregate_ok:
        raise IntegerScorerAuditError(
            "val NPZ aggregate SHA256 does not match the preregistration"
        )

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
    bundle_by_seed = {int(entry["seed"]): entry for entry in prereg["accepted_wv_bundles"]}
    contract = declared_width_contract()
    if not contract["analytical_all_fit"]:
        raise IntegerScorerAuditError("analytical integer width contract violation")

    observed: dict[str, int] = {}
    raw_by_seed: dict[int, dict[str, Any]] = {}
    csv_rows: list[dict[str, Any]] = []
    bundle_meta: list[dict[str, Any]] = []
    for seed in [int(value) for value in seeds]:
        checkpoint = checkpoint_by_seed[seed]
        ckpt_path = root / str(checkpoint["path"])
        projection, _ = load_projection(ckpt_path)
        if sha256_file(ckpt_path) != str(checkpoint["sha256"]):
            raise IntegerScorerAuditError(f"checkpoint {ckpt_path} changed after load")
        Wt = np.asarray(projection.Wt, dtype=np.float64)
        Wv = np.asarray(projection.Wv, dtype=np.float64)
        wv_q_ref, wv_scale_ref = wv_audit.quantize_weight_tensor(Wv)

        accepted = load_accepted_wv_bundle(root, bundle_by_seed[seed])
        if not np.array_equal(accepted["wv_q"], wv_q_ref):
            raise IntegerScorerAuditError(
                f"accepted Wv INT8 bundle for seed {seed} disagrees with the quantiser"
            )
        if float(accepted["scale_float32"]) != float(np.float32(wv_scale_ref)):
            raise IntegerScorerAuditError(
                f"accepted Wv scale for seed {seed} disagrees with the quantiser"
            )

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
                wv_q_ref=accepted["wv_q"],
                wv_scale_ref=wv_scale_ref,
                target=target,
                grid_h=int(sample["grid_h"]),
                grid_w=int(sample["grid_w"]),
                budgets=budgets,
            )
            for role in ROLES:
                _observed_from_chain(observed, record["integer"][role])

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
                            "float_selected_count": int(metrics["float_selected_count"]),
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
                                record["spearman"][role]["candidate_vs_per_tensor"]
                            ),
                        }
                    )

        if write_artifacts:
            fixture_inputs = []
            for task in TASKS:
                sample = min(
                    (entry for entry in val_samples if str(entry["task"]) == task),
                    key=lambda entry: str(entry["sample_id"]),
                )
                enriched = dict(sample)
                enriched["wrong_prompt_embeddings"] = by_id[
                    partner[str(sample["sample_id"])]
                ]["prompt_embeddings"]
                fixture_inputs.append(enriched)
            bundle_reference = dict(bundle_by_seed[seed])
            bundle_reference["_wv_q"] = accepted["wv_q"]
            bundle_reference["_scale_float32"] = accepted["scale_float32"]
            bundle_meta.append(
                build_seed_bundle(
                    seed,
                    Wt=Wt,
                    Wv=Wv,
                    fixtures=fixture_inputs,
                    checkpoint=checkpoint,
                    accepted_bundle=bundle_reference,
                    out_dir=root / artifact_root / str(seed),
                    budgets=budgets,
                )
            )

    summary = aggregate_metrics(
        raw_by_seed, sample_meta=sample_meta, seeds=seeds, budgets=budgets
    )
    width_contract = finalize_width_contract(contract, observed)
    width_ok = bool(width_contract["all_fit_declared_widths"])

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
    fixture_ok = bool(
        all(bool(entry.get("roundtrip", {}).get("ok")) for entry in bundle_meta)
        if bundle_meta
        else True
    )
    decision = evaluate_decision(
        summary,
        pins_ok=pins_ok,
        isolation_ok=isolation_ok,
        fixture_ok=fixture_ok,
        width_ok=width_ok,
        budgets=budgets,
    )

    contract_path = root / artifact_root / "integer_width_contract.json"
    if write_artifacts:
        _write_json(contract_path, width_contract)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "report_stem": REPORT_STEM,
        "date": DATE,
        "experiment": "selector_integer_scorer_audit",
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
            "semantic_fractional_bits": SEMANTIC_FRACTIONAL_BITS,
            "score_fractional_bits": SCORE_FRACTIONAL_BITS,
            "lambda_semantic": LAMBDA_SEMANTIC,
            "budgets": [int(value) for value in budgets],
            "seeds": [int(value) for value in seeds],
        },
        "identity": identity,
        "pins_before": pins_before,
        "pins_after": pins_after,
        "pins_unchanged": pins_ok,
        "isolation": isolation,
        "integer_width_contract": width_contract,
        "integer_width_contract_path": str(artifact_root / "integer_width_contract.json"),
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
            "seed_k48_candidate_vs_per_tensor_jaccard": summary[
                "seed_k48_candidate_vs_per_tensor_jaccard"
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
            "wv_duplicated_in_fixtures": False,
            "integer_width_contract": str(contract_path),
        },
        "limitations": list(prereg.get("limitations", []))
        + [
            "This is a Python bit-exact integer reference, not HLS synthesis, "
            "board execution, FPGA latency or end-to-end accuracy evidence.",
            "Python arbitrary-precision integers are used only to prove fixed "
            "widths and to run the exact rational divisions/square roots.",
            "The reference float-normalisation arm remains the accepted prior chain "
            "and is not re-derived here.",
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
    lines.append(f"# 位精确整数 Selector scorer 审计 — {REPORT_STEM}")
    lines.append("")
    lines.append(f"- 决策：**{decision['decision']}**")
    lines.append(
        "- 范围：CPU-only、离线、validation-only、只读；未训练、未调参、未加载 4B、未读原图、未访问 internal test。"
    )
    lines.append(
        f"- 参考臂：`{REFERENCE_ARM}`（已接受的 per-tensor Wv float-normalisation 链路）；"
        f"候选臂：`{CANDIDATE_ARM}`（输入量化后纯整数，semantic 小数位冻结为 {SEMANTIC_FRACTIONAL_BITS}）。"
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
    lines.append("## 候选 vs per-tensor 参考（correct prompt，overall 图像级 mean）")
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
                    metrics["score_rank_spearman"]["candidate_vs_per_tensor"]["correct"][
                        "overall"
                    ]["mean"]
                ),
                _fmt(entry["cand_vs_ref_mass_delta"]["mean"]),
                _fmt(entry["cand_vs_ref_ndcg_delta"]["mean"]),
            )
        )
    lines.append("")
    lines.append("## 候选 vs 原始 float（correct prompt，overall 图像级 mean）")
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
    for reference, label in (("float", "候选 − float"), ("per_tensor", "候选 − per-tensor")):
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
    lines.append("## 逐 seed K48 候选 vs per-tensor mean Jaccard")
    lines.append("")
    for seed, value in sorted(
        decision["seed_k48_candidate_vs_per_tensor_mean_jaccard"].items()
    ):
        lines.append(f"- seed {seed}: {_fmt(value)}")
    lines.append("")
    lines.append("## 整数宽度与操作合同")
    lines.append("")
    lines.append("| 操作 | 宽度 | 解析上界 | 观测最大 | 均放得下 |")
    lines.append("|---|---|---|---|---|")
    contract = bundle["integer_width_contract"]
    for entry in contract["operations"]:
        lines.append(
            f"| `{entry['op']}` | {entry['width']} | {entry['analytical_bound']} | "
            f"{entry.get('observed_max', 'n/a')} | {entry.get('observed_fits')} |"
        )
    lines.append("")
    lines.append(
        f"- 全部分析/观测值放得下声明宽度：`{contract['all_fit_declared_widths']}`；"
        f"fixture 回读逐位一致：`{bundle['fixtures_validated']}`；fixture 不复制 Wv。"
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
        "> 明确说明：本审计是 Python 位精确整数参考，非 HLS 综合/上板证据；"
        "通过仅授权实现 HLS，不证明时延、资源、任务精度或端到端加速。"
    )
    lines.append("")
    return "\n".join(lines)


def harness_payload(bundle: Mapping[str, Any], files: Mapping[str, str]) -> dict[str, Any]:
    decision = bundle["decision"]
    metrics = bundle["metrics"]
    overall = metrics["overall"]["correct"]
    contract = bundle["integer_width_contract"]
    return {
        "schema_version": SCHEMA_VERSION,
        "report_stem": REPORT_STEM,
        "decision": decision["decision"],
        "reference_arm": REFERENCE_ARM,
        "candidate_arm": CANDIDATE_ARM,
        "checks": decision["checks"],
        "candidate_vs_per_tensor": {
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
        "candidate_vs_per_tensor_score_spearman": metrics["score_rank_spearman"][
            "candidate_vs_per_tensor"
        ]["correct"]["overall"],
        "pointing_k48_prompt_utility_gap_delta_candidate_vs_float": metrics[
            "prompt_utility_gap_delta"
        ]["float"]["48"]["mass"]["per_task"]["pointing"],
        "seed_k48_candidate_vs_per_tensor_mean_jaccard": decision[
            "seed_k48_candidate_vs_per_tensor_mean_jaccard"
        ],
        "integer_width_contract": {
            "all_fit_declared_widths": contract["all_fit_declared_widths"],
            "analytical_all_fit": contract["analytical_all_fit"],
            "observed_all_fit": contract["observed_all_fit"],
            "operations": contract["operations"],
            "observed_maxima": contract["observed_maxima"],
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
