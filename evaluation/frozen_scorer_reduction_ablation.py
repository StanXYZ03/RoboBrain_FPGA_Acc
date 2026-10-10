#!/usr/bin/env python3
"""Frozen-weight scorer-reduction ablation (CPU-only, diagnosis only).

The v2 prompt-identifiability diagnostic established that the current student
score ``max_t cosine(Wv @ group_mean, Wt @ prompt_t)`` collapses: the max over
prompt tokens is dominated by one projected token whose direction is almost
identical for the grounding and pointing prompts of the same image, so the
correct-prompt and same-image-other-task scores are exactly equal.

This module isolates that failure.  It does **not** train, does **not** tune a
hyper-parameter, does **not** load a 4B model, does **not** use a GPU and never
touches the network.  It reuses the frozen feature cache (472 records / 236
images), the frozen ``Wt``/``Wv`` and the frozen train/val/test split, then
scores the same visual groups with a small, **pre-registered** candidate set of
prompt-token reducers.

Discipline enforced by this file:

* a pre-registration JSON is written **before** any test value is read/computed;
* the promotion gate is evaluated on **val only**; the winner is written to disk
  before the test split is loaded for the first and only time;
* every donor map is strictly split-local and uses a fixed-seed deterministic
  derangement on images (bijective, no self-pair); student donors keep the task;
* the correct-vs-control paired deltas are image-clustered (grounding/pointing
  averaged per image) with 2000 bootstrap resamples and 10000 two-sided sign-flip
  permutation resamples, matching the v2 design;
* exact prompt invariance uses ``np.array_equal`` and is reported next to the
  explicitly named ``1e-9`` tolerance variant;
* the Holm-Bonferroni correction is applied to the non-baseline candidates' val
  primary tests.

All score/metric/load primitives are imported from the frozen modules
(``evaluation.prompt_projection_distillation``,
``evaluation.prompt_identifiability_diagnostic`` and
``evaluation.train_prompt_projection``); the only new math here is the token
reducer family, the token-weight concentration statistic and the analytic
hardware cost contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluation.prompt_identifiability_diagnostic import (
    BOOTSTRAP_SEED,
    derange_images,
    image_task_index,
    paired_effect,
    same_image_partner_map,
    sample_donor_map,
    score_invariance_stats,
)
from evaluation.prompt_projection_distillation import (
    DistillationContractError,
    FEATURE_DIM,
    PROMPT_DIM,
    SCORE_DIM,
    config_sha256,
    load_distillation_config,
    load_feature_cache,
    metric_bundle,
    split_sample_ids,
)
from evaluation.run_hf_hardware_regular_b1 import load_projection
from evaluation.train_prompt_projection import index_file_sha256

_EPS = 1e-12

# ---------------------------------------------------------------------------
# Pre-registered constants (fixed BEFORE the test split is read).
# ---------------------------------------------------------------------------
PREREGISTRATION_DATE = "2026-10-02"
PREREGISTRATION_SEED = 20261002
BOOTSTRAP_RESAMPLES = 2000
PERMUTATION_RESAMPLES = 10000
ALPHA = 0.05
CI_LEVEL = 0.95
EFFECT_SIZE_FLOOR = 0.2
COLLAPSE_EXACT_INVARIANCE_MAX = 0.05
TEACHER_NON_REGRESSION_TOLERANCE = -0.01
PROMPT_INVARIANCE_TOLERANCE = 1e-9
PRIMARY_METRICS = ("spearman", "ndcg_full")
DELTA_METRICS = ("pearson", "spearman", "ndcg_full")
ROLE_CORRECT = "correct"
ROLE_SAME = "same_image_other_task"
ROLE_CROSS = "cross_image_same_task_shuffled"
ROLES = (ROLE_CORRECT, ROLE_SAME, ROLE_CROSS)
# Fixed seed offsets for the per-split deterministic derangements (same values
# as the v2 diagnostic so the pairing is fully pre-registered).
DERANGEMENT_SPLIT_OFFSETS: dict[str, int] = {"train": 0, "val": 101, "test": 202}

REPORT_VERSION = "1.0"
REPORT_STEM = "2026-10-02-frozen-scorer-reduction"

# ---------------------------------------------------------------------------
# The frozen candidate set.  The order is part of the pre-registration: it is
# the final tie-break of the winner selection.
# ---------------------------------------------------------------------------
CANDIDATES: list[dict[str, Any]] = [
    {"name": "hard_max", "kind": "hard_max", "params": {}, "token_competition": True,
     "description": "existing baseline: per group max over prompt-token cosines"},
    {"name": "mean_cos", "kind": "mean_cos", "params": {}, "token_competition": False,
     "description": "arithmetic mean over all prompt-token cosines"},
    {"name": "topk_mean_k2", "kind": "topk_mean", "params": {"k": 2}, "token_competition": True,
     "description": "mean of the highest 2 cosines; T<2 uses all"},
    {"name": "topk_mean_k4", "kind": "topk_mean", "params": {"k": 4}, "token_competition": True,
     "description": "mean of the highest 4 cosines; T<4 uses all"},
    {"name": "norm_lse_tau003", "kind": "norm_lse", "params": {"tau": 0.03}, "token_competition": True,
     "description": "tau*(logsumexp(sim/tau) - log(T)), tau=0.03"},
    {"name": "norm_lse_tau010", "kind": "norm_lse", "params": {"tau": 0.10}, "token_competition": True,
     "description": "tau*(logsumexp(sim/tau) - log(T)), tau=0.10"},
    {"name": "norm_lse_tau030", "kind": "norm_lse", "params": {"tau": 0.30}, "token_competition": True,
     "description": "tau*(logsumexp(sim/tau) - log(T)), tau=0.30"},
    {"name": "softmax_mean_tau003", "kind": "softmax_mean", "params": {"tau": 0.03}, "token_competition": True,
     "description": "softmax(sim/tau)-weighted mean of sim, tau=0.03"},
    {"name": "softmax_mean_tau010", "kind": "softmax_mean", "params": {"tau": 0.10}, "token_competition": True,
     "description": "softmax(sim/tau)-weighted mean of sim, tau=0.10"},
    {"name": "softmax_mean_tau030", "kind": "softmax_mean", "params": {"tau": 0.30}, "token_competition": True,
     "description": "softmax(sim/tau)-weighted mean of sim, tau=0.30"},
    {"name": "prompt_mean_vector", "kind": "prompt_mean_vector", "params": {}, "token_competition": False,
     "description": (
         "normalise each prompt token Wt projection (existing cosine convention), "
         "mean over tokens, re-normalise; every visual group does a single "
         "cosine/dot against this one 128-d query"
     )},
]
CANDIDATE_ORDER: tuple[str, ...] = tuple(str(candidate["name"]) for candidate in CANDIDATES)
CANDIDATE_SPEC: dict[str, dict[str, Any]] = {
    str(candidate["name"]): candidate for candidate in CANDIDATES
}
BASELINE_CANDIDATE = "hard_max"
NON_BASELINE_CANDIDATES: tuple[str, ...] = tuple(
    name for name in CANDIDATE_ORDER if name != BASELINE_CANDIDATE
)
# The Holm family is fixed: every (non-baseline candidate, condition, primary
# metric) val test belongs to the same family.
HOLM_FAMILY_SIZE = len(NON_BASELINE_CANDIDATES) * 2 * len(PRIMARY_METRICS)
HOLM_FAMILY_DEFINITION = (
    "{ (candidate, condition, metric) : candidate in NON_BASELINE_CANDIDATES "
    f"({len(NON_BASELINE_CANDIDATES)} candidates), condition in "
    "{same_image_other_task, cross_image_same_task_shuffled}, metric in "
    "{spearman, ndcg_full} }; m = "
    f"{HOLM_FAMILY_SIZE}"
)

HARDWARE_SCOPE_NOTE = (
    "analytic symbol counts only (G=visual group count, T=prompt token count, "
    "D=128); the prompt reducer runs on the host/Jetson and the scorer on the "
    "FPGA.  NOT synthesised and NOT measured on ZCU102; no latency, resource or "
    "speedup claim is made.  INT8 bytes assume one signed int8 per element "
    "(128-byte 1x128 query / 16-byte 128-bit alignment not modelled here)."
)


# ---------------------------------------------------------------------------
# Reducer family
# ---------------------------------------------------------------------------
def _unit_rows(matrix: Any, name: str) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise DistillationContractError(f"{name} must be 2-D")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= _EPS):
        raise DistillationContractError(f"{name} contains a zero-norm row")
    return array / norms[:, None]


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def prompt_cosine_matrix(
    group_mean: Any, prompt_embeddings: Any, Wt: Any, Wv: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(cosines[G,T], visual_unit[G,D], prompt_unit[T,D])``.

    This is exactly the frozen cosine convention of
    ``prompt_projection_distillation.student_cosine_scores`` (rows are L2
    normalised before the dot product); only the reduction over ``T`` is new.
    """
    visual = np.asarray(group_mean, dtype=np.float64) @ np.asarray(Wv, dtype=np.float64).T
    prompt = np.asarray(prompt_embeddings, dtype=np.float64) @ np.asarray(Wt, dtype=np.float64).T
    visual_unit = _unit_rows(visual, "visual_projection")
    prompt_unit = _unit_rows(prompt, "prompt_projection")
    return visual_unit @ prompt_unit.T, visual_unit, prompt_unit


def reduce_cosine_matrix(
    name: str,
    cosines: np.ndarray,
    visual_unit: np.ndarray | None = None,
    prompt_unit: np.ndarray | None = None,
) -> np.ndarray:
    """Apply one pre-registered reducer to a ``[G,T]`` cosine matrix."""
    if name not in CANDIDATE_SPEC:
        raise DistillationContractError(f"unknown reducer {name!r}")
    spec = CANDIDATE_SPEC[name]
    kind = str(spec["kind"])
    params = spec["params"]
    matrix = np.asarray(cosines, dtype=np.float64)
    if matrix.ndim != 2:
        raise DistillationContractError("cosines must be a [G,T] matrix")
    group_count, token_count = matrix.shape
    if kind == "hard_max":
        return matrix.max(axis=1)
    if kind == "mean_cos":
        return matrix.mean(axis=1)
    if kind == "topk_mean":
        k = min(int(params["k"]), token_count)
        order = np.argsort(-matrix, axis=1, kind="stable")[:, :k]
        rows = np.arange(group_count)[:, None]
        return matrix[rows, order].mean(axis=1)
    if kind == "norm_lse":
        tau = float(params["tau"])
        scaled = matrix / tau
        row_max = scaled.max(axis=1, keepdims=True)
        log_sum = row_max + np.log(
            np.sum(np.exp(scaled - row_max), axis=1, keepdims=True)
        )
        return (tau * (log_sum - math.log(token_count))).reshape(-1)
    if kind == "softmax_mean":
        tau = float(params["tau"])
        weights = _softmax_rows(matrix / tau)
        return (weights * matrix).sum(axis=1)
    if kind == "prompt_mean_vector":
        if visual_unit is None or prompt_unit is None:
            raise DistillationContractError(
                "prompt_mean_vector needs the unit visual/prompt projections"
            )
        query = np.asarray(prompt_unit, dtype=np.float64).mean(axis=0)
        norm = float(np.linalg.norm(query))
        if norm <= _EPS:
            return np.zeros(group_count, dtype=np.float64)
        query = query / norm
        return np.asarray(visual_unit, dtype=np.float64) @ query
    raise DistillationContractError(f"unsupported reducer kind {kind!r}")


def token_weight_matrix(name: str, cosines: np.ndarray) -> np.ndarray | None:
    """Per-group token weight matrix for competitive reducers, else ``None``.

    ``None`` marks a reducer without token competition (``mean_cos`` uses one
    uniform weight, ``prompt_mean_vector`` never forms a per-token score).
    """
    spec = CANDIDATE_SPEC[name]
    kind = str(spec["kind"])
    params = spec["params"]
    matrix = np.asarray(cosines, dtype=np.float64)
    group_count, token_count = matrix.shape
    if kind == "hard_max":
        weights = np.zeros_like(matrix)
        weights[np.arange(group_count), matrix.argmax(axis=1)] = 1.0
        return weights
    if kind == "topk_mean":
        k = min(int(params["k"]), token_count)
        order = np.argsort(-matrix, axis=1, kind="stable")[:, :k]
        weights = np.zeros_like(matrix)
        weights[np.arange(group_count)[:, None], order] = 1.0 / float(k)
        return weights
    if kind in {"norm_lse", "softmax_mean"}:
        return _softmax_rows(matrix / float(params["tau"]))
    return None


def score_candidate(
    name: str, group_mean: Any, prompt_embeddings: Any, Wt: Any, Wv: Any
) -> np.ndarray:
    """Convenience one-shot scorer used by the focused tests."""
    cosines, visual_unit, prompt_unit = prompt_cosine_matrix(
        group_mean, prompt_embeddings, Wt, Wv
    )
    return reduce_cosine_matrix(name, cosines, visual_unit, prompt_unit).astype(np.float32)


# ---------------------------------------------------------------------------
# Analytic hardware contract
# ---------------------------------------------------------------------------
def hardware_contract_for(
    name: str,
    *,
    group_count: int = 1,
    token_count: int = 1,
    dimension: int = SCORE_DIM,
    feature_dim: int = FEATURE_DIM,
    prompt_dim: int = PROMPT_DIM,
) -> dict[str, Any]:
    """Per-reducer MAC / compare / exp / reduction / transfer contract.

    The prompt side (host/Jetson) projects ``[T, prompt_dim]`` to ``[T, D]`` and
    optionally reduces it; the scorer side (FPGA) projects each visual group and
    reduces the ``[G, T]`` cosine matrix.  Every matrix reducer must ship the
    full ``[T, D]`` prompt query, ``prompt_mean_vector`` ships a single
    ``[1, D]`` query and therefore replaces the FPGA ``T x D`` dot with ``D``.
    """
    g = int(group_count)
    t = int(token_count)
    d = int(dimension)
    kind = str(CANDIDATE_SPEC[name]["kind"])
    params = CANDIDATE_SPEC[name]["params"]
    if kind == "prompt_mean_vector":
        query_tokens = 1
        host_reducer_mac = t * d  # per-token L2 normalisation
        host_reducer_compare = 0
        host_reducer_exp = 0
        host_reducer_add = t * d + d  # mean over tokens + final L2 normalisation
        dot_mac = d
        compare = 0
        exp = 0
        reduction_add = 0
    else:
        query_tokens = t
        host_reducer_mac = 0
        host_reducer_compare = 0
        host_reducer_exp = 0
        host_reducer_add = 0
        dot_mac = t * d
        if kind == "hard_max":
            compare, exp, reduction_add = max(0, t - 1), 0, 0
        elif kind == "mean_cos":
            compare, exp, reduction_add = 0, 0, max(0, t - 1)
        elif kind == "topk_mean":
            k = min(int(params["k"]), t)
            compare, exp, reduction_add = k * t, 0, max(0, k - 1)
        elif kind == "norm_lse":
            compare, exp, reduction_add = t, t, t + 1  # max, exp, sum-exp + log
        elif kind == "softmax_mean":
            compare, exp, reduction_add = t, t, 2 * t  # max, exp, sum-exp + weighted mean
        else:  # pragma: no cover - defensive
            raise DistillationContractError(f"unsupported reducer kind {kind!r}")
    total_per_group = dot_mac + compare + exp + reduction_add
    return {
        "candidate": name,
        "kind": kind,
        "token_competition": bool(CANDIDATE_SPEC[name]["token_competition"]),
        "symbols": {"G": g, "T": t, "D": d},
        "prompt_reducer_side": "host/Jetson (prompt projection + optional prompt-side reduction)",
        "scorer_side": "FPGA (visual projection + per-group token reduction)",
        "host_prompt_projection_mac_per_sample": t * d * prompt_dim,
        "host_prompt_query_tokens": query_tokens,
        "host_prompt_query_shape": [query_tokens, d],
        "host_prompt_reducer_mac_per_sample": host_reducer_mac,
        "host_prompt_reducer_compare_per_sample": host_reducer_compare,
        "host_prompt_reducer_exp_per_sample": host_reducer_exp,
        "host_prompt_reducer_reduction_add_per_sample": host_reducer_add,
        "prompt_query_transfer_bytes_fp32_per_sample": query_tokens * d * 4,
        "prompt_query_transfer_bytes_int8_per_sample": query_tokens * d * 1,
        "fpga_visual_projection_mac_per_group": d * feature_dim,
        "fpga_scorer_dot_mac_per_group": dot_mac,
        "fpga_scorer_compare_per_group": compare,
        "fpga_scorer_exp_per_group": exp,
        "fpga_scorer_reduction_add_per_group": reduction_add,
        "fpga_scorer_total_ops_per_group": total_per_group,
        "fpga_scorer_total_ops_per_image": total_per_group * g,
        "formulas": {
            "hard_max": "dot=T*D, compare=T-1, exp=0, reduction=0",
            "mean_cos": "dot=T*D, compare=0, exp=0, reduction=T-1",
            "topk_mean": "dot=T*D, compare=K*T, exp=0, reduction=K-1, K=min(k,T)",
            "norm_lse": "dot=T*D, compare=T, exp=T, reduction=T+1",
            "softmax_mean": "dot=T*D, compare=T, exp=T, reduction=2T",
            "prompt_mean_vector": "dot=D, compare=0, exp=0, reduction=0; query=[1,D]",
        }[kind],
    }


def hardware_contract(
    *, group_count: int, token_count: int, dimension: int = SCORE_DIM
) -> dict[str, Any]:
    return {
        "scope_note": HARDWARE_SCOPE_NOTE,
        "symbols": {
            "G": "visual group count per image",
            "T": "prompt token count per sample",
            "D": SCORE_DIM,
            "visual_feature_dim": FEATURE_DIM,
            "prompt_dim": PROMPT_DIM,
        },
        "evaluation_point": {"G": int(group_count), "T": int(token_count)},
        "per_reducer": {
            name: hardware_contract_for(
                name, group_count=group_count, token_count=token_count, dimension=dimension
            )
            for name in CANDIDATE_ORDER
        },
    }


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def holm_bonferroni(p_values: Mapping[str, float], *, alpha: float = ALPHA) -> dict[str, float]:
    """Holm-Bonferroni step-down adjustment of a family of p-values."""
    items = sorted(
        ((str(key), float(value)) for key, value in p_values.items()),
        key=lambda item: (item[1], item[0]),
    )
    count = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (key, value) in enumerate(items):
        candidate = min(1.0, max(running, (count - index) * value))
        adjusted[key] = candidate
        running = candidate
    return adjusted


def _mean_nested(dicts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not dicts:
        return {}
    result: dict[str, Any] = {}
    for key in dicts[0]:
        values = [entry[key] for entry in dicts]
        first = values[0]
        if isinstance(first, Mapping):
            result[key] = _mean_nested(values)  # type: ignore[arg-type]
        elif isinstance(first, bool):
            result[key] = bool(first)
        elif isinstance(first, (int, float)) and not isinstance(first, bool):
            result[key] = float(np.mean([float(value) for value in values]))
        else:
            result[key] = first
    return result


def _paired_seed(split_name: str, candidate: str, role: str, metric: str) -> int:
    candidate_index = CANDIDATE_ORDER.index(candidate)
    role_index = (ROLE_SAME, ROLE_CROSS).index(role)
    metric_index = DELTA_METRICS.index(metric)
    return (
        BOOTSTRAP_SEED
        + DERANGEMENT_SPLIT_OFFSETS.get(split_name, 0)
        + 101 * candidate_index
        + 11 * role_index
        + metric_index
    )


# ---------------------------------------------------------------------------
# Per-split evaluation
# ---------------------------------------------------------------------------
def evaluate_split(
    split_name: str,
    samples: Sequence[Mapping[str, Any]],
    Wt: np.ndarray,
    Wv: np.ndarray,
    *,
    budgets: Sequence[int],
    seed: int,
) -> dict[str, Any]:
    """Evaluate every pre-registered candidate with split-local donors."""
    by_id = {str(sample["sample_id"]): sample for sample in samples}
    ordered = sorted(by_id)
    if not ordered:
        raise DistillationContractError(f"split {split_name} has no samples")
    budgets = [int(b) for b in budgets]
    partner = same_image_partner_map(samples)
    image_index = image_task_index(samples)
    image_ids = sorted(image_index)
    donor_by_image = derange_images(image_ids, seed=int(seed))
    student_donor = sample_donor_map(samples, donor_by_image, same_task=True)
    image_of = {sid: str(by_id[sid]["source_image_sha256"]) for sid in ordered}

    role_scores: dict[str, dict[str, dict[str, np.ndarray]]] = {
        role: {} for role in ROLES
    }
    bundles: dict[str, dict[str, dict[str, Any]]] = {role: {} for role in ROLES}
    invariance: dict[str, dict[str, dict[str, Any]]] = {}
    concentration: dict[str, dict[str, Any]] = {}
    token_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}

    for sample_id in ordered:
        sample = by_id[sample_id]
        group_mean = sample["group_mean"]
        teacher_raw = sample["teacher_raw"]
        teacher_normalized = sample["teacher_normalized"]
        teacher_soft = sample["teacher_soft"]
        cosines, visual_unit, prompt_unit = prompt_cosine_matrix(
            group_mean, sample["prompt_embeddings"], Wt, Wv
        )
        group_counts[sample_id] = int(group_mean.shape[0])
        token_counts[sample_id] = int(sample["prompt_embeddings"].shape[0])

        other_id = partner.get(sample_id)
        donor_id = student_donor.get(sample_id)
        if other_id is None or donor_id is None:
            raise DistillationContractError(
                f"{sample_id}: missing same-image partner or cross-image donor"
            )
        prompts = {
            ROLE_CORRECT: sample["prompt_embeddings"],
            ROLE_SAME: by_id[other_id]["prompt_embeddings"],
            ROLE_CROSS: by_id[donor_id]["prompt_embeddings"],
        }
        role_units: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
            ROLE_CORRECT: (cosines, visual_unit, prompt_unit)
        }
        for role in (ROLE_SAME, ROLE_CROSS):
            role_units[role] = prompt_cosine_matrix(group_mean, prompts[role], Wt, Wv)
        for role in ROLES:
            role_cosines, role_visual_unit, role_prompt_unit = role_units[role]
            role_scores[role][sample_id] = {}
            bundles[role][sample_id] = {}
            for candidate in CANDIDATE_ORDER:
                scores = reduce_cosine_matrix(
                    candidate, role_cosines, role_visual_unit, role_prompt_unit
                )
                scores = np.asarray(scores, dtype=np.float32)
                role_scores[role][sample_id][candidate] = scores
                bundles[role][sample_id][candidate] = metric_bundle(
                    scores,
                    teacher_raw,
                    teacher_normalized,
                    teacher_soft,
                    budgets=budgets,
                )

        invariance[sample_id] = {}
        concentration[sample_id] = {}
        for candidate in CANDIDATE_ORDER:
            invariance[sample_id][candidate] = score_invariance_stats(
                role_scores[ROLE_CORRECT][sample_id][candidate],
                role_scores[ROLE_SAME][sample_id][candidate],
                tolerance=PROMPT_INVARIANCE_TOLERANCE,
            )
            weights = token_weight_matrix(candidate, cosines)
            if weights is None:
                concentration[sample_id][candidate] = None
                continue
            mean_weights = weights.mean(axis=0)
            argmax_tokens = cosines.argmax(axis=1)
            concentration[sample_id][candidate] = {
                "dominant_token_concentration": float(mean_weights.max()),
                "single_argmax_token_fraction": float(np.unique(argmax_tokens).size == 1),
                "mean_distinct_argmax_tokens": float(np.unique(argmax_tokens).size),
            }

    role_means = {
        role: {
            candidate: _mean_nested(
                [bundles[role][sample_id][candidate] for sample_id in ordered]
            )
            for candidate in CANDIDATE_ORDER
        }
        for role in ROLES
    }

    deltas: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATE_ORDER:
        deltas[candidate] = {}
        for role in (ROLE_SAME, ROLE_CROSS):
            deltas[candidate][role] = {}
            for metric in DELTA_METRICS:
                per_image: dict[str, list[float]] = defaultdict(list)
                for sample_id in ordered:
                    per_image[image_of[sample_id]].append(
                        float(
                            bundles[ROLE_CORRECT][sample_id][candidate][metric]
                            - bundles[role][sample_id][candidate][metric]
                        )
                    )
                image_deltas = [float(np.mean(values)) for values in per_image.values()]
                deltas[candidate][role][metric] = {
                    "definition": (
                        f"image-clustered mean of (correct - {role}) student {metric} "
                        "over the grounding/pointing samples of one image"
                    ),
                    "independent_unit": "image",
                    **paired_effect(
                        image_deltas,
                        seed=_paired_seed(split_name, candidate, role, metric),
                        n_bootstrap=BOOTSTRAP_RESAMPLES,
                        n_permutation=PERMUTATION_RESAMPLES,
                        sample_count=len(ordered),
                        independent_image_units=len(image_deltas),
                    ),
                }

    invariance_summary: dict[str, Any] = {}
    for candidate in CANDIDATE_ORDER:
        stats = [
            invariance[sample_id][candidate]
            for sample_id in ordered
            if candidate in invariance[sample_id]
        ]
        diffs = np.asarray([entry["max_abs_diff"] for entry in stats], dtype=np.float64)
        invariance_summary[candidate] = {
            "candidate": candidate,
            "pairs": len(stats),
            "exact_uses_np_array_equal": True,
            "tolerance_used": PROMPT_INVARIANCE_TOLERANCE,
            "exact_prompt_invariance_fraction": float(np.mean([entry["exact"] for entry in stats]))
            if stats
            else None,
            "tolerance_prompt_invariance_fraction": float(
                np.mean([entry["tolerance"] for entry in stats])
            )
            if stats
            else None,
            "max_abs_diff_max": float(diffs.max()) if diffs.size else None,
            "max_abs_diff_median": float(np.median(diffs)) if diffs.size else None,
            "max_abs_diff_p95": float(np.percentile(diffs, 95)) if diffs.size else None,
        }

    concentration_summary: dict[str, Any] = {}
    for candidate in CANDIDATE_ORDER:
        specs = [
            concentration[sample_id][candidate]
            for sample_id in ordered
            if concentration[sample_id][candidate] is not None
        ]
        if not specs:
            concentration_summary[candidate] = {
                "candidate": candidate,
                "token_competition": bool(CANDIDATE_SPEC[candidate]["token_competition"]),
                "dominant_token_concentration": None,
                "dominant_token_concentration_status": "N/A",
                "note": "reducer has no per-token competition",
            }
            continue
        concentration_summary[candidate] = {
            "candidate": candidate,
            "token_competition": True,
            "dominant_token_concentration": float(
                np.mean([entry["dominant_token_concentration"] for entry in specs])
            ),
            "dominant_token_concentration_status": "REPORTED",
            "single_argmax_token_fraction": float(
                np.mean([entry["single_argmax_token_fraction"] for entry in specs])
            ),
            "mean_distinct_argmax_tokens": float(
                np.mean([entry["mean_distinct_argmax_tokens"] for entry in specs])
            ),
            "definition": (
                "per sample: mean over groups of the reducer token weights w[g,t], "
                "then max over t; for hard_max this is the fraction of groups "
                "dominated by the modal token, for the soft family it is the modal "
                "softmax mass"
            ),
        }

    return {
        "split": split_name,
        "count": len(ordered),
        "independent_image_units": len(image_ids),
        "budgets": budgets,
        "roles": role_means,
        "deltas": deltas,
        "invariance": invariance_summary,
        "concentration": concentration_summary,
        "prompt_token_count": {
            "min": int(min(token_counts.values())),
            "max": int(max(token_counts.values())),
            "mean": float(np.mean(list(token_counts.values()))),
        },
        "group_count": {
            "min": int(min(group_counts.values())),
            "max": int(max(group_counts.values())),
            "mean": float(np.mean(list(group_counts.values()))),
        },
    }


# ---------------------------------------------------------------------------
# Holm correction, gates and winner selection
# ---------------------------------------------------------------------------
def apply_holm_to_val(val_result: dict[str, Any]) -> dict[str, Any]:
    family: dict[str, float] = {}
    for candidate in NON_BASELINE_CANDIDATES:
        for role in (ROLE_SAME, ROLE_CROSS):
            for metric in PRIMARY_METRICS:
                family[f"{candidate}|{role}|{metric}"] = float(
                    val_result["deltas"][candidate][role][metric]["p_value_two_sided"]
                )
    adjusted = holm_bonferroni(family)
    for key, value in adjusted.items():
        candidate, role, metric = key.split("|")
        entry = val_result["deltas"][candidate][role][metric]
        entry["holm_adjusted_p"] = float(value)
        entry["holm_family_size"] = HOLM_FAMILY_SIZE
        entry["holm_family_definition"] = HOLM_FAMILY_DEFINITION
    for candidate in NON_BASELINE_CANDIDATES:
        for role in (ROLE_SAME, ROLE_CROSS):
            for metric in PRIMARY_METRICS:
                val_result["deltas"][candidate][role][metric].setdefault(
                    "holm_adjusted_p", None
                )
    return val_result


def _delta_gate(entry: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "mean_positive": bool(float(entry["mean"]) > 0.0),
        "ci_low_positive": bool(entry["ci_low"] is not None and float(entry["ci_low"]) > 0.0),
        "effect_size": bool(
            entry["cohens_dz"] is not None and abs(float(entry["cohens_dz"])) >= EFFECT_SIZE_FLOOR
        ),
        "holm_significant": bool(
            entry.get("holm_adjusted_p") is not None
            and float(entry["holm_adjusted_p"]) < ALPHA
        ),
    }


def evaluate_gates(
    val_result: Mapping[str, Any], *, hardware: Mapping[str, Any]
) -> dict[str, Any]:
    baseline_correct = val_result["roles"][ROLE_CORRECT][BASELINE_CANDIDATE]
    gate: dict[str, Any] = {
        "split": "val",
        "fixed_before_test": True,
        "primary_metrics": list(PRIMARY_METRICS),
        "collapse_exact_invariance_max": COLLAPSE_EXACT_INVARIANCE_MAX,
        "teacher_non_regression_tolerance": TEACHER_NON_REGRESSION_TOLERANCE,
        "holm_family_definition": HOLM_FAMILY_DEFINITION,
        "thresholds": {
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "permutation_resamples": PERMUTATION_RESAMPLES,
            "alpha": ALPHA,
            "effect_size_floor_cohens_dz": EFFECT_SIZE_FLOOR,
        },
        "candidates": {},
    }
    for candidate in CANDIDATE_ORDER:
        correct = val_result["roles"][ROLE_CORRECT][candidate]
        same_checks = {
            metric: _delta_gate(val_result["deltas"][candidate][ROLE_SAME][metric])
            for metric in PRIMARY_METRICS
        }
        cross_checks = {
            metric: _delta_gate(val_result["deltas"][candidate][ROLE_CROSS][metric])
            for metric in PRIMARY_METRICS
        }
        gate_a = all(
            all(same_checks[metric].values()) for metric in PRIMARY_METRICS
        )
        gate_b = all(
            all(cross_checks[metric].values()) for metric in PRIMARY_METRICS
        )
        exact_fraction = val_result["invariance"][candidate]["exact_prompt_invariance_fraction"]
        gate_c = bool(
            exact_fraction is not None
            and float(exact_fraction) <= COLLAPSE_EXACT_INVARIANCE_MAX
        )
        spearman_delta = float(correct["spearman"]) - float(baseline_correct["spearman"])
        ndcg_delta = float(correct["ndcg_full"]) - float(baseline_correct["ndcg_full"])
        gate_d = bool(
            spearman_delta >= TEACHER_NON_REGRESSION_TOLERANCE
            and ndcg_delta >= TEACHER_NON_REGRESSION_TOLERANCE
        )
        finite = bool(
            np.isfinite(correct["spearman"])
            and np.isfinite(correct["ndcg_full"])
            and np.isfinite(val_result["deltas"][candidate][ROLE_SAME]["spearman"]["mean"])
            and np.isfinite(val_result["deltas"][candidate][ROLE_CROSS]["ndcg_full"]["mean"])
        )
        contract = hardware["per_reducer"][candidate]
        contract_ok = bool(
            contract["host_prompt_query_shape"][1] == SCORE_DIM
            and contract["prompt_query_transfer_bytes_fp32_per_sample"]
            == contract["host_prompt_query_tokens"] * SCORE_DIM * 4
            and contract["prompt_query_transfer_bytes_int8_per_sample"]
            == contract["host_prompt_query_tokens"] * SCORE_DIM
            and contract["fpga_scorer_total_ops_per_group"]
            == contract["fpga_scorer_dot_mac_per_group"]
            + contract["fpga_scorer_compare_per_group"]
            + contract["fpga_scorer_exp_per_group"]
            + contract["fpga_scorer_reduction_add_per_group"]
        )
        gate_e = bool(finite and contract_ok)
        gate["candidates"][candidate] = {
            "A_prompt_sensitivity": {
                "pass": gate_a,
                "per_metric": same_checks,
                "role": ROLE_SAME,
            },
            "B_shuffled_discrimination": {
                "pass": gate_b,
                "per_metric": cross_checks,
                "role": ROLE_CROSS,
            },
            "C_collapse_removed": {"pass": gate_c, "exact_prompt_invariance_fraction": exact_fraction},
            "D_teacher_non_regression": {
                "pass": gate_d,
                "correct_spearman_delta_vs_baseline": spearman_delta,
                "correct_ndcg_full_delta_vs_baseline": ndcg_delta,
            },
            "E_numeric_and_contract": {"pass": gate_e, "finite": finite, "contract_ok": contract_ok},
            "passed_all": bool(gate_a and gate_b and gate_c and gate_d and gate_e),
        }
    gate["passing_candidates"] = [
        candidate for candidate in CANDIDATE_ORDER if gate["candidates"][candidate]["passed_all"]
    ]
    return gate


def select_winner(
    val_result: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    hardware: Mapping[str, Any],
) -> dict[str, Any]:
    """Lexicographic winner selection fixed in the pre-registration."""
    passing = list(gate["passing_candidates"])
    if not passing:
        return {
            "winner": "NO_PROMOTION",
            "reason": "no candidate passed all pre-registered val gates A-E",
            "ranking": [],
        }
    rows: list[dict[str, Any]] = []
    for candidate in passing:
        same_spearman = float(val_result["deltas"][candidate][ROLE_SAME]["spearman"]["mean"])
        same_ndcg = float(val_result["deltas"][candidate][ROLE_SAME]["ndcg_full"]["mean"])
        cross_spearman = float(val_result["deltas"][candidate][ROLE_CROSS]["spearman"]["mean"])
        cross_ndcg = float(val_result["deltas"][candidate][ROLE_CROSS]["ndcg_full"]["mean"])
        min_signal = min(same_spearman, same_ndcg, cross_spearman, cross_ndcg)
        correct = val_result["roles"][ROLE_CORRECT][candidate]
        alignment = float(correct["spearman"]) + float(correct["ndcg_full"])
        contract = hardware["per_reducer"][candidate]
        hardware_key = (
            int(contract["fpga_scorer_total_ops_per_image"]),
            int(contract["prompt_query_transfer_bytes_int8_per_sample"]),
        )
        rows.append(
            {
                "candidate": candidate,
                "min_primary_delta": min_signal,
                "correct_spearman_plus_ndcg_full": alignment,
                "fpga_scorer_total_ops_per_image": hardware_key[0],
                "prompt_query_transfer_bytes_int8": hardware_key[1],
                "candidate_order_rank": CANDIDATE_ORDER.index(candidate),
                "_key": (
                    -min_signal,
                    -alignment,
                    hardware_key,
                    CANDIDATE_ORDER.index(candidate),
                ),
            }
        )
    rows.sort(key=lambda row: row["_key"])
    best = rows[0]
    for row in rows:
        row.pop("_key")
    return {
        "winner": str(best["candidate"]),
        "reason": "highest min primary val delta, then teacher alignment, then lowest cost, then candidate order",
        "ranking": rows,
    }


# ---------------------------------------------------------------------------
# Pre-registration artefact
# ---------------------------------------------------------------------------
def build_preregistration(
    *,
    config_path: Path,
    index_path: Path,
    projection_path: Path,
    split_summary: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": REPORT_VERSION,
        "report_version": REPORT_VERSION,
        "preregistration_date": PREREGISTRATION_DATE,
        "seed": int(seed),
        "scope": (
            "CPU-only frozen-weight scorer-reduction ablation; no training, no "
            "tuning, no 4B model load, no GPU, no network, no CV-Bench"
        ),
        "baseline_candidate": BASELINE_CANDIDATE,
        "candidates": [
            {
                "name": candidate["name"],
                "kind": candidate["kind"],
                "parameters": candidate["params"],
                "token_competition": candidate["token_competition"],
                "description": candidate["description"],
            }
            for candidate in CANDIDATES
        ],
        "candidate_order": list(CANDIDATE_ORDER),
        "candidate_selection_rule": (
            "the candidate set, its parameters and its order are frozen before "
            "the test split is read; no candidate may be added, removed or "
            "re-parameterised after test is seen (no post-hoc tuning)"
        ),
        "primary_metrics": list(PRIMARY_METRICS),
        "statistics": {
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "permutation_resamples": PERMUTATION_RESAMPLES,
            "alpha": ALPHA,
            "ci_level": CI_LEVEL,
            "effect_size_floor_cohens_dz": EFFECT_SIZE_FLOOR,
            "independent_unit": "image (grounding/pointing averaged before resampling)",
            "donor_scope": "strictly split-local",
            "cross_image_donor": (
                "fixed-seed deterministic derangement on images (bijective, no "
                "self-pair); student donors keep the task"
            ),
            "exact_invariance": "np.array_equal; tolerance variant explicitly named 1e-9",
            "holm_bonferroni_family": HOLM_FAMILY_DEFINITION,
            "derangement_split_offsets": dict(DERANGEMENT_SPLIT_OFFSETS),
        },
        "promotion_gate": {
            "A_prompt_sensitivity": (
                "same-image-other-task spearman and ndcg_full paired delta > 0, "
                "95% CI lower bound > 0, |dz| >= 0.2 and Holm-adjusted p < 0.05"
            ),
            "B_shuffled_discrimination": (
                "cross-image-same-task shuffled spearman and ndcg_full paired "
                "delta > 0, 95% CI lower bound > 0, |dz| >= 0.2 and "
                "Holm-adjusted p < 0.05"
            ),
            "C_collapse_removed": f"exact_prompt_invariance_fraction <= {COLLAPSE_EXACT_INVARIANCE_MAX}",
            "D_teacher_non_regression": (
                "val correct-prompt spearman and ndcg_full both >= hard_max "
                f"baseline + {TEACHER_NON_REGRESSION_TOLERANCE}"
            ),
            "E_numeric_and_contract": "all metrics finite and the hardware contract complete",
        },
        "winner_selection_lexicographic": [
            "maximise min(same spearman delta, same ndcg_full delta, cross spearman delta, cross ndcg_full delta)",
            "then maximise val correct-prompt spearman + ndcg_full",
            "then minimise per-group MAC/compare/exp and prompt-side transfer bytes",
            "then follow the fixed candidate order",
        ],
        "test_discipline": {
            "val_decides": True,
            "test_evaluated_once_after_winner_written": True,
            "no_post_hoc_selection_on_test": (
                "if no candidate passes, winner=NO_PROMOTION and the single test "
                "evaluation is a frozen diagnostic only and may not be used to "
                "pick a candidate"
            ),
        },
        "hardware_cost_contract": {
            "scope_note": HARDWARE_SCOPE_NOTE,
            "symbols": ["G=visual group count", "T=prompt token count", "D=128"],
        },
        "inputs": {
            "config": str(config_path),
            "config_sha256": config_sha256(config_path),
            "index": str(index_path),
            "index_sha256": index_file_sha256(index_path),
            "projection": str(projection_path),
            "projection_npz_sha256": hashlib.sha256(Path(projection_path).read_bytes()).hexdigest(),
        },
        "split_summary": dict(split_summary),
    }
    payload["preregistration_sha256"] = preregistration_sha256(payload)
    return payload


def preregistration_sha256(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "preregistration_sha256"}
    blob = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Loading helpers (split-aware so val never reads test npz files)
# ---------------------------------------------------------------------------
def load_index_records(index_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in Path(index_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        raise DistillationContractError(f"index {index_path} contains no samples")
    return records


def load_split_samples(
    index_path: Path,
    features_dir: Path | None,
    records: Sequence[Mapping[str, Any]],
    split_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Load only the npz files of the requested split (test stays untouched)."""
    directory = Path(features_dir) if features_dir else Path(index_path).parent
    by_id = {str(record["sample_id"]): record for record in records}
    samples: list[dict[str, Any]] = []
    for sample_id in split_ids:
        record = by_id[str(sample_id)]
        npz_path = Path(record["npz"])
        if not npz_path.is_absolute():
            npz_path = directory / npz_path
        cache = load_feature_cache(npz_path)
        samples.append(
            {
                "sample_id": str(sample_id),
                "task": str(record.get("task", cache["metadata"].get("task", ""))),
                "source_image_sha256": str(
                    record.get("source_image_sha256", cache["metadata"].get("source_image_sha256"))
                ).lower(),
                "grid_h": int(record["grid_h"]),
                "grid_w": int(record["grid_w"]),
                "group_mean": cache["group_mean"].astype(np.float32),
                "prompt_embeddings": cache["prompt_embeddings"].astype(np.float32),
                "teacher_raw": cache["teacher_scores"].astype(np.float32),
                "teacher_normalized": cache["teacher_scores_normalized"].astype(np.float32),
                "teacher_soft": cache["teacher_soft_target"].astype(np.float32),
                "npz": str(npz_path),
            }
        )
    return samples


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_ablation(
    *,
    config_path: Path,
    index_path: Path,
    features_dir: Path | None,
    projection_path: Path,
    split_names: Sequence[str] = ("train", "val", "test"),
    seed: int = PREREGISTRATION_SEED,
    out_dir: Path | None = None,
    write_artifacts: bool = False,
) -> dict[str, Any]:
    """Run the ablation.  Pre-registration is written before any test read."""
    started = time.time()
    config = load_distillation_config(config_path)
    records = load_index_records(index_path)
    split = split_sample_ids(
        records,
        seed=int(config["split"]["seed"]),
        ratios=config["split"]["ratios"],
        salt=str(config["split"]["salt"]),
    )
    image_hashes = {str(record["sample_id"]): str(record["source_image_sha256"]) for record in records}
    split_summary = {
        name: {
            "sample_count": len(ids),
            "image_count": len({image_hashes[sample_id] for sample_id in ids}),
        }
        for name, ids in split.items()
    }
    projection, projection_sha = load_projection(projection_path)
    Wt = np.asarray(projection.Wt, dtype=np.float32)
    Wv = np.asarray(projection.Wv, dtype=np.float32)

    paths: dict[str, Path] = {}
    if write_artifacts:
        if out_dir is None:
            raise DistillationContractError("write_artifacts requires out_dir")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = _output_paths(out_dir)

    preregistration = build_preregistration(
        config_path=config_path,
        index_path=index_path,
        projection_path=projection_path,
        split_summary=split_summary,
        seed=seed,
    )
    if write_artifacts:
        _write_json(paths["preregistration"], preregistration)
        if not paths["preregistration"].is_file():  # pragma: no cover - defensive
            raise DistillationContractError("pre-registration was not written")

    budgets = [int(value) for value in config["budgets"]]
    requested = [str(name) for name in split_names]
    for name in requested:
        if name not in split:
            raise DistillationContractError(f"unknown split {name!r}")

    results: dict[str, Any] = {}
    if "val" in requested:
        val_samples = load_split_samples(index_path, features_dir, records, split["val"])
        results["val"] = evaluate_split(
            "val",
            val_samples,
            Wt,
            Wv,
            budgets=budgets,
            seed=int(seed) + DERANGEMENT_SPLIT_OFFSETS["val"],
        )
        apply_holm_to_val(results["val"])
    gate: dict[str, Any] | None = None
    winner: dict[str, Any] | None = None
    if "val" in results:
        val_group_mean = int(round(results["val"]["group_count"]["mean"]))
        val_token_mean = int(round(results["val"]["prompt_token_count"]["mean"]))
        hardware = hardware_contract(group_count=val_group_mean, token_count=val_token_mean)
        gate = evaluate_gates(results["val"], hardware=hardware)
        winner = select_winner(results["val"], gate, hardware=hardware)
    else:
        hardware = hardware_contract(group_count=1, token_count=1)

    # Val selection is complete and the winner is on disk before test is read.
    if write_artifacts and gate is not None and winner is not None:
        _write_json(
            paths["val_gate"],
            {
                "split": "val",
                "seed": int(seed),
                "winner": winner["winner"],
                "winner_reason": winner["reason"],
                "winner_ranking": winner["ranking"],
                "val_gate": gate,
                "preregistration_sha256": preregistration["preregistration_sha256"],
                "test_not_yet_evaluated": True,
            },
        )

    for name in requested:
        if name == "val":
            continue
        samples = load_split_samples(index_path, features_dir, records, split[name])
        results[name] = evaluate_split(
            name,
            samples,
            Wt,
            Wv,
            budgets=budgets,
            seed=int(seed) + DERANGEMENT_SPLIT_OFFSETS[name],
        )

    bundle: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "status": "COMPUTED",
        "scope": preregistration["scope"],
        "seed": int(seed),
        "preregistration": preregistration,
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "candidate_order": list(CANDIDATE_ORDER),
        "baseline_candidate": BASELINE_CANDIDATE,
        "split_summary": split_summary,
        "splits": results,
        "splits_evaluated": list(results),
        "hardware": hardware,
        "val_gate": gate,
        "winner": winner,
        "test_once": {
            "evaluated": "test" in results,
            "evaluated_after_winner_written": bool(
                "test" in results and winner is not None and write_artifacts
            ),
            "used_for_selection": False,
            "note": (
                "test is loaded once, after the val winner/NO_PROMOTION is written; "
                "it may never change the winner"
            ),
        },
        "provenance": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "seed": int(seed),
            "derangement_seeds": {
                name: int(seed) + DERANGEMENT_SPLIT_OFFSETS[name] for name in ("train", "val", "test")
            },
            "config_sha256": preregistration["inputs"]["config_sha256"],
            "index_sha256": preregistration["inputs"]["index_sha256"],
            "projection_sha256": projection_sha,
            "projection_npz_sha256": preregistration["inputs"]["projection_npz_sha256"],
            "elapsed_seconds": None,
        },
    }
    bundle["provenance"]["elapsed_seconds"] = time.time() - started
    if write_artifacts:
        bundle["artifacts"] = {key: str(value) for key, value in paths.items()}
    return bundle


def _output_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "preregistration": out_dir / f"{REPORT_STEM}-preregistration.json",
        "val_gate": out_dir / f"{REPORT_STEM}-val-gate.json",
        "json": out_dir / f"{REPORT_STEM}.json",
        "markdown": out_dir / f"{REPORT_STEM}.md",
        "candidates_csv": out_dir / f"{REPORT_STEM}-candidates.csv",
        "splits_csv": out_dir / f"{REPORT_STEM}-splits.csv",
        "log": out_dir / "logs" / f"{REPORT_STEM}-run.log",
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# CSV projections of the results
# ---------------------------------------------------------------------------
def _flat_bundle(bundle: Mapping[str, Any]) -> dict[str, float]:
    flat = {
        "pearson": float(bundle["pearson"]),
        "spearman": float(bundle["spearman"]),
        "ndcg_full": float(bundle["ndcg_full"]),
    }
    for budget in (48, 64, 96):
        entry = bundle["per_budget"][str(budget)]
        flat[f"ndcg{budget}"] = float(entry["ndcg"])
        flat[f"recall{budget}"] = float(entry["topk_recall"])
        flat[f"jaccard{budget}"] = float(entry["jaccard"])
        flat[f"saliency{budget}"] = float(entry["saliency_mass_retained"])
    return flat


def _role_rows(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("train", "val", "test"):
        result = bundle["splits"].get(name)
        if result is None:
            continue
        for role in ROLES:
            for candidate in CANDIDATE_ORDER:
                row = {
                    "candidate": candidate,
                    "split": name,
                    "role": role,
                    "count": result["count"],
                    "independent_image_units": result["independent_image_units"],
                    "token_competition": CANDIDATE_SPEC[candidate]["token_competition"],
                }
                row.update(_flat_bundle(result["roles"][role][candidate]))
                rows.append(row)
    return rows


def _candidate_rows(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gate = bundle.get("val_gate") or {}
    hardware = bundle["hardware"]["per_reducer"]
    for name in ("train", "val", "test"):
        result = bundle["splits"].get(name)
        if result is None:
            continue
        for candidate in CANDIDATE_ORDER:
            correct = _flat_bundle(result["roles"][ROLE_CORRECT][candidate])
            same = _flat_bundle(result["roles"][ROLE_SAME][candidate])
            cross = _flat_bundle(result["roles"][ROLE_CROSS][candidate])
            row: dict[str, Any] = {
                "candidate": candidate,
                "split": name,
                "is_baseline": candidate == BASELINE_CANDIDATE,
                "token_competition": CANDIDATE_SPEC[candidate]["token_competition"],
                "count": result["count"],
                "independent_image_units": result["independent_image_units"],
            }
            for metric, value in correct.items():
                row[f"correct_{metric}"] = value
            for metric in ("pearson", "spearman", "ndcg_full"):
                row[f"same_{metric}"] = same[metric]
                row[f"cross_{metric}"] = cross[metric]
            for role_key, role in (("same", ROLE_SAME), ("cross", ROLE_CROSS)):
                for metric in DELTA_METRICS:
                    entry = result["deltas"][candidate][role][metric]
                    row[f"delta_{role_key}_{metric}_mean"] = entry["mean"]
                    row[f"delta_{role_key}_{metric}_ci_low"] = entry["ci_low"]
                    row[f"delta_{role_key}_{metric}_ci_high"] = entry["ci_high"]
                    row[f"delta_{role_key}_{metric}_dz"] = entry["cohens_dz"]
                    row[f"delta_{role_key}_{metric}_p"] = entry["p_value_two_sided"]
                    row[f"delta_{role_key}_{metric}_holm_p"] = entry.get("holm_adjusted_p")
            inv = result["invariance"][candidate]
            row.update(
                {
                    "exact_prompt_invariance_fraction": inv["exact_prompt_invariance_fraction"],
                    "tolerance_prompt_invariance_fraction": inv["tolerance_prompt_invariance_fraction"],
                    "invariance_pairs": inv["pairs"],
                    "invariance_max_abs_diff_max": inv["max_abs_diff_max"],
                    "invariance_max_abs_diff_median": inv["max_abs_diff_median"],
                    "invariance_max_abs_diff_p95": inv["max_abs_diff_p95"],
                }
            )
            conc = result["concentration"][candidate]
            row.update(
                {
                    "dominant_token_concentration": conc.get("dominant_token_concentration"),
                    "dominant_token_concentration_status": conc["dominant_token_concentration_status"],
                    "single_argmax_token_fraction": conc.get("single_argmax_token_fraction"),
                    "mean_distinct_argmax_tokens": conc.get("mean_distinct_argmax_tokens"),
                }
            )
            contract = hardware[candidate]
            row.update(
                {
                    "hardware_fpga_scorer_total_ops_per_image": contract["fpga_scorer_total_ops_per_image"],
                    "hardware_prompt_query_tokens": contract["host_prompt_query_tokens"],
                    "hardware_prompt_query_bytes_fp32": contract["prompt_query_transfer_bytes_fp32_per_sample"],
                    "hardware_prompt_query_bytes_int8": contract["prompt_query_transfer_bytes_int8_per_sample"],
                }
            )
            if name == "val" and gate:
                candidate_gate = gate["candidates"][candidate]
                row["gate_A_pass"] = candidate_gate["A_prompt_sensitivity"]["pass"]
                row["gate_B_pass"] = candidate_gate["B_shuffled_discrimination"]["pass"]
                row["gate_C_pass"] = candidate_gate["C_collapse_removed"]["pass"]
                row["gate_D_pass"] = candidate_gate["D_teacher_non_regression"]["pass"]
                row["gate_E_pass"] = candidate_gate["E_numeric_and_contract"]["pass"]
                row["passed_all"] = candidate_gate["passed_all"]
            else:
                # gates are val-only; keep the columns present but empty elsewhere
                for field in (
                    "gate_A_pass",
                    "gate_B_pass",
                    "gate_C_pass",
                    "gate_D_pass",
                    "gate_E_pass",
                    "passed_all",
                ):
                    row[field] = None
            rows.append(row)
    return rows


ROLE_CSV_FIELDS: list[str] = [
    "candidate", "split", "role", "count", "independent_image_units", "token_competition",
    "pearson", "spearman", "ndcg_full",
    "ndcg48", "ndcg64", "ndcg96",
    "recall48", "recall64", "recall96",
    "jaccard48", "jaccard64", "jaccard96",
    "saliency48", "saliency64", "saliency96",
]

CANDIDATE_CSV_FIELDS: list[str] = [
    "candidate", "split", "is_baseline", "token_competition",
    "count", "independent_image_units",
    "correct_pearson", "correct_spearman", "correct_ndcg_full",
    "correct_ndcg48", "correct_ndcg64", "correct_ndcg96",
    "correct_recall48", "correct_recall64", "correct_recall96",
    "correct_jaccard48", "correct_jaccard64", "correct_jaccard96",
    "correct_saliency48", "correct_saliency64", "correct_saliency96",
    "same_pearson", "same_spearman", "same_ndcg_full",
    "cross_pearson", "cross_spearman", "cross_ndcg_full",
    "delta_same_pearson_mean", "delta_same_pearson_ci_low", "delta_same_pearson_ci_high",
    "delta_same_pearson_dz", "delta_same_pearson_p", "delta_same_pearson_holm_p",
    "delta_same_spearman_mean", "delta_same_spearman_ci_low", "delta_same_spearman_ci_high",
    "delta_same_spearman_dz", "delta_same_spearman_p", "delta_same_spearman_holm_p",
    "delta_same_ndcg_full_mean", "delta_same_ndcg_full_ci_low", "delta_same_ndcg_full_ci_high",
    "delta_same_ndcg_full_dz", "delta_same_ndcg_full_p", "delta_same_ndcg_full_holm_p",
    "delta_cross_pearson_mean", "delta_cross_pearson_ci_low", "delta_cross_pearson_ci_high",
    "delta_cross_pearson_dz", "delta_cross_pearson_p", "delta_cross_pearson_holm_p",
    "delta_cross_spearman_mean", "delta_cross_spearman_ci_low", "delta_cross_spearman_ci_high",
    "delta_cross_spearman_dz", "delta_cross_spearman_p", "delta_cross_spearman_holm_p",
    "delta_cross_ndcg_full_mean", "delta_cross_ndcg_full_ci_low", "delta_cross_ndcg_full_ci_high",
    "delta_cross_ndcg_full_dz", "delta_cross_ndcg_full_p", "delta_cross_ndcg_full_holm_p",
    "exact_prompt_invariance_fraction", "tolerance_prompt_invariance_fraction", "invariance_pairs",
    "invariance_max_abs_diff_max", "invariance_max_abs_diff_median", "invariance_max_abs_diff_p95",
    "dominant_token_concentration", "dominant_token_concentration_status",
    "single_argmax_token_fraction", "mean_distinct_argmax_tokens",
    "hardware_fpga_scorer_total_ops_per_image", "hardware_prompt_query_tokens",
    "hardware_prompt_query_bytes_fp32", "hardware_prompt_query_bytes_int8",
    "gate_A_pass", "gate_B_pass", "gate_C_pass", "gate_D_pass", "gate_E_pass", "passed_all",
]


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def render_markdown(bundle: Mapping[str, Any]) -> str:
    gate = bundle.get("val_gate") or {}
    winner = bundle.get("winner") or {}
    hardware = bundle["hardware"]["per_reducer"]
    lines: list[str] = []
    lines.append("# 2026-10-02 冻结权重 scorer-reduction 消融（中文主报告）")
    lines.append("")
    lines.append(
        "> CPU-only、只读、冻结权重：**不训练、不调参、不加载 4B 模型、不使用 GPU、"
        "不联网、不跑 CV-Bench**。复用既有 472 条 NPZ 派生特征、当前 `Wt`/`Wv`、"
        "现有 train/val/test split 与 v2 统计工具。"
    )
    lines.append("")
    lines.append(
        f"- 预注册：`{bundle['preregistration_sha256']}`（seed {bundle['seed']}，"
        f"读取任何 test 数值前写盘）；候选顺序 `{', '.join(bundle['candidate_order'])}`"
    )
    lines.append("")
    lines.append("## 1. 目标")
    lines.append("")
    lines.append(
        "隔离并修复现有 prompt-conditioned selector 的 hard-max prompt collapse："
        "v2 诊断显示 `max_t cosine(Wv·group_mean, Wt·prompt_t)` 在每个 group 上由单一 "
        "prompt token 主导，同图 grounding/pointing 的 correct 分数 `np.array_equal` 精确相等。"
        "本消融只替换 **prompt-token 归约**，其余（视觉投影、prompt 投影、split、teacher target、指标定义）全部冻结。"
    )
    lines.append("")
    lines.append("## 2. 预注册候选集合（不得在 test 后增删/改参）")
    lines.append("")
    lines.append("| # | candidate | kind | params | token competition |")
    lines.append("| --- | --- | --- | --- | --- |")
    for index, candidate in enumerate(bundle["preregistration"]["candidates"]):
        lines.append(
            f"| {index + 1} | {candidate['name']} | {candidate['kind']} | "
            f"`{json.dumps(candidate['parameters'], ensure_ascii=False)}` | {candidate['token_competition']} |"
        )
    lines.append("")
    lines.append("## 3. val promotion gate（全部只看 val）")
    lines.append("")
    lines.append(
        "- A prompt sensitivity：same-image-other-task 的 Spearman 与 NDCG-full paired delta 均 >0，"
        "95% CI 下界 >0，|dz|≥0.2，Holm-adjusted p<0.05"
    )
    lines.append(
        "- B shuffled discrimination：cross-image-same-task 的 Spearman 与 NDCG-full paired delta 均 >0，"
        "95% CI 下界 >0，|dz|≥0.2，Holm-adjusted p<0.05"
    )
    lines.append("- C collapse removed：exact_prompt_invariance_fraction ≤ 0.05")
    lines.append(
        "- D teacher non-regression：候选 val correct Spearman 与 NDCG-full 均不低于 hard_max baseline 超过 0.01"
    )
    lines.append("- E 数值有限且硬件实现合同完整")
    lines.append(
        f"- Holm-Bonferroni family：{HOLM_FAMILY_DEFINITION}"
    )
    lines.append("")
    lines.append(f"- **winner = {winner.get('winner', 'N/A')}**；{winner.get('reason', '')}")
    lines.append("")
    lines.append("| candidate | A | B | C | D | E | passed_all | exact_inv frac |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for candidate in CANDIDATE_ORDER:
        entry = gate.get("candidates", {}).get(candidate)
        if entry is None:
            continue
        lines.append(
            "| {c} | {a} | {b} | {cc} | {d} | {e} | {p} | {x:.4f} |".format(
                c=candidate,
                a=entry["A_prompt_sensitivity"]["pass"],
                b=entry["B_shuffled_discrimination"]["pass"],
                cc=entry["C_collapse_removed"]["pass"],
                d=entry["D_teacher_non_regression"]["pass"],
                e=entry["E_numeric_and_contract"]["pass"],
                p=entry["passed_all"],
                x=entry["C_collapse_removed"]["exact_prompt_invariance_fraction"],
            )
        )
    lines.append("")
    lines.append("## 4. val 关键数值（correct 与两个 paired control，图像级聚类）")
    lines.append("")
    val = bundle["splits"].get("val")
    if val is not None:
        lines.append(
            f"- val：n_rec={val['count']}，n_img={val['independent_image_units']}，"
            f"prompt token min/mean/max = {val['prompt_token_count']['min']}/"
            f"{val['prompt_token_count']['mean']:.1f}/{val['prompt_token_count']['max']}，"
            f"G min/mean/max = {val['group_count']['min']}/{val['group_count']['mean']:.1f}/"
            f"{val['group_count']['max']}"
        )
        lines.append("")
        lines.append("| candidate | correct rho | correct NDCG-full | Δsame rho [CI] | Δsame NDCG [CI] | Δcross rho [CI] | Δcross NDCG [CI] |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for candidate in CANDIDATE_ORDER:
            correct = val["roles"][ROLE_CORRECT][candidate]
            parts = [candidate, f"{correct['spearman']:.4f}", f"{correct['ndcg_full']:.4f}"]
            for role in (ROLE_SAME, ROLE_CROSS):
                for metric in ("spearman", "ndcg_full"):
                    entry = val["deltas"][candidate][role][metric]
                    parts.append(
                        f"{entry['mean']:+.4f} [{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}]"
                    )
            lines.append("| " + " | ".join(parts) + " |")
        lines.append("")
        lines.append("### 4.1 dominant-token concentration（含 token competition 的 reducer）")
        lines.append("")
        lines.append("| candidate | concentration | single-argmax fraction | mean distinct tokens |")
        lines.append("| --- | --- | --- | --- |")
        for candidate in CANDIDATE_ORDER:
            entry = val["concentration"][candidate]
            if entry["dominant_token_concentration_status"] == "N/A":
                lines.append(f"| {candidate} | N/A | N/A | N/A |")
            else:
                lines.append(
                    f"| {candidate} | {entry['dominant_token_concentration']:.4f} | "
                    f"{entry['single_argmax_token_fraction']:.4f} | "
                    f"{entry['mean_distinct_argmax_tokens']:.3f} |"
                )
        lines.append("")
        lines.append("### 4.2 prompt invariance（correct vs 同图另一任务）")
        lines.append("")
        lines.append("| candidate | exact (np.array_equal) | tolerance 1e-9 | pairs | max_abs_diff max |")
        lines.append("| --- | --- | --- | --- | --- |")
        for candidate in CANDIDATE_ORDER:
            entry = val["invariance"][candidate]
            lines.append(
                f"| {candidate} | {entry['exact_prompt_invariance_fraction']:.4f} | "
                f"{entry['tolerance_prompt_invariance_fraction']:.4f} | {entry['pairs']} | "
                f"{entry['max_abs_diff_max']:.3e} |"
            )
    lines.append("")
    lines.append("### 4.3 K=48/64/96 teacher top-K overlap/recall/Jaccard/saliency mass（val correct）")
    lines.append("")
    lines.append("| candidate | recall48/64/96 | Jaccard48/64/96 | saliency48/64/96 | NDCG@48/64/96 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for candidate in CANDIDATE_ORDER:
        entry = _flat_bundle(val["roles"][ROLE_CORRECT][candidate])
        lines.append(
            "| {c} | {r} | {j} | {s} | {n} |".format(
                c=candidate,
                r="/".join(f"{entry[f'recall{k}']:.4f}" for k in (48, 64, 96)),
                j="/".join(f"{entry[f'jaccard{k}']:.4f}" for k in (48, 64, 96)),
                s="/".join(f"{entry[f'saliency{k}']:.4f}" for k in (48, 64, 96)),
                n="/".join(f"{entry[f'ndcg{k}']:.4f}" for k in (48, 64, 96)),
            )
        )
    lines.append("")
    lines.append("## 5. 硬件代价（解析符号计数，未综合、未实测 ZCU102）")
    lines.append("")
    lines.append(f"- {HARDWARE_SCOPE_NOTE}")
    lines.append(
        f"- 评估点：G={bundle['hardware']['evaluation_point']['G']}，"
        f"T={bundle['hardware']['evaluation_point']['T']}，D={SCORE_DIM}"
    )
    lines.append("")
    lines.append("| candidate | FPGA dot/group | cmp/group | exp/group | reduce/group | total/group | total/image | query shape | query INT8 B |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for candidate in CANDIDATE_ORDER:
        c = hardware[candidate]
        lines.append(
            "| {c} | {d} | {cmp} | {e} | {r} | {t} | {ti} | {q} | {qb} |".format(
                c=candidate,
                d=c["fpga_scorer_dot_mac_per_group"],
                cmp=c["fpga_scorer_compare_per_group"],
                e=c["fpga_scorer_exp_per_group"],
                r=c["fpga_scorer_reduction_add_per_group"],
                t=c["fpga_scorer_total_ops_per_group"],
                ti=c["fpga_scorer_total_ops_per_image"],
                q=str(c["host_prompt_query_shape"]),
                qb=c["prompt_query_transfer_bytes_int8_per_sample"],
            )
        )
    lines.append("")
    lines.append(
        "重点：`prompt_mean_vector` 只需 **1x128 INT8 query**（128 B）传输，FPGA 侧为 "
        "**Gx128 dot**（每 group 128 MAC，无 compare/exp/归约），彻底移除 token 竞争；"
        "其余矩阵型 reducer 需传输 `[T,128]` query 并在 FPGA 上做 `T*D` dot 与归约。"
    )
    lines.append("")
    lines.append("## 6. test（仅在 winner 写盘后评一次，不得据 test 反选）")
    lines.append("")
    test = bundle["splits"].get("test")
    if test is None:
        lines.append("- 本次运行未计算 test（val-only）。")
    else:
        lines.append(
            f"- evaluated={bundle['test_once']['evaluated']}，"
            f"evaluated_after_winner_written={bundle['test_once']['evaluated_after_winner_written']}，"
            f"used_for_selection={bundle['test_once']['used_for_selection']}"
        )
        lines.append(
            f"- test：n_rec={test['count']}，n_img={test['independent_image_units']}"
        )
        winner_name = winner.get("winner")
        if winner_name and winner_name != "NO_PROMOTION":
            correct = test["roles"][ROLE_CORRECT][winner_name]
            lines.append(
                f"- winner `{winner_name}` test correct Spearman={correct['spearman']:.4f}，"
                f"NDCG-full={correct['ndcg_full']:.4f}，"
                f"exact invariance={test['invariance'][winner_name]['exact_prompt_invariance_fraction']:.4f}"
            )
        else:
            baseline = test["roles"][ROLE_CORRECT][BASELINE_CANDIDATE]
            lines.append(
                f"- NO_PROMOTION：baseline `{BASELINE_CANDIDATE}` test correct Spearman="
                f"{baseline['spearman']:.4f}，NDCG-full={baseline['ndcg_full']:.4f}（仅冻结诊断）"
            )
    lines.append("")
    lines.append("## 7. 结论与下一决策")
    lines.append("")
    if winner.get("winner") == "NO_PROMOTION":
        lines.append(
            "- **winner = NO_PROMOTION**：本批预注册候选无人通过全部门槛 A-E，"
            "按预注册规则**不得事后挑“最不差”方案晋级**，也不得据 test 反选。"
        )
    else:
        lines.append(
            f"- **winner = {winner.get('winner')}**：通过全部门槛 A-E 后再按字典序"
            "（min primary delta → teacher alignment → 硬件代价 → 候选顺序）选出。"
        )
    lines.append(
        "- 所有数字均为 teacher-target 排序/相关性内部指标，**不是 CV-Bench 任务精度**；"
        "硬件代价为解析符号计数，**未综合、未实测 ZCU102**。"
    )
    lines.append("")
    lines.append("## 8. 复现")
    lines.append("")
    lines.append(f"- python {bundle['provenance']['python']} / numpy {bundle['provenance']['numpy']}")
    lines.append(f"- seed {bundle['provenance']['seed']}；derangement seeds {bundle['provenance']['derangement_seeds']}")
    lines.append(
        f"- index_sha256 `{bundle['provenance']['index_sha256'][:16]}…`；"
        f"projection_sha256 `{bundle['provenance']['projection_sha256'][:16]}…`；"
        f"preregistration `{bundle['preregistration_sha256'][:16]}…`"
    )
    lines.append(f"- elapsed {bundle['provenance']['elapsed_seconds']:.2f} s")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/prompt_projection_distillation.json"))
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/token_selector/prompt_projection_features/index.jsonl"),
    )
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument(
        "--projection",
        type=Path,
        default=Path("artifacts/token_selector/prompt_projection_WtWv.npz"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("reports/token_selector"))
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="comma-separated split list; val-only skips test entirely",
    )
    parser.add_argument("--seed", type=int, default=PREREGISTRATION_SEED)
    parser.add_argument("--val-only", action="store_true")
    args = parser.parse_args(argv)

    split_names = ["val"] if args.val_only else [s.strip() for s in args.splits.split(",") if s.strip()]
    if "val" not in split_names:
        parser.error("val must be included: the gates are val-only")
    bundle = run_ablation(
        config_path=args.config,
        index_path=args.index,
        features_dir=args.features_dir,
        projection_path=args.projection,
        split_names=split_names,
        seed=args.seed,
        out_dir=args.out_dir,
        write_artifacts=True,
    )
    paths = _output_paths(Path(args.out_dir))
    _write_json(paths["json"], bundle)
    paths["markdown"].write_text(render_markdown(bundle), encoding="utf-8")
    _write_csv(paths["candidates_csv"], _candidate_rows(bundle), CANDIDATE_CSV_FIELDS)
    _write_csv(paths["splits_csv"], _role_rows(bundle), ROLE_CSV_FIELDS)
    log_lines = [
        f"command: {' '.join(sys.argv)}",
        f"status: {bundle['status']}",
        f"seed: {bundle['seed']}",
        f"preregistration_sha256: {bundle['preregistration_sha256']}",
        f"winner: {(bundle.get('winner') or {}).get('winner')}",
        f"splits_evaluated: {bundle['splits_evaluated']}",
        f"python: {bundle['provenance']['python']} numpy: {bundle['provenance']['numpy']}",
        f"index_sha256: {bundle['provenance']['index_sha256']}",
        f"projection_sha256: {bundle['provenance']['projection_sha256']}",
        f"elapsed_seconds: {bundle['provenance']['elapsed_seconds']:.3f}",
    ]
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    paths["log"].write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    bundle["provenance"]["output_sha256"] = {
        "preregistration": hashlib.sha256(paths["preregistration"].read_bytes()).hexdigest(),
        "markdown": hashlib.sha256(paths["markdown"].read_bytes()).hexdigest(),
        "candidates_csv": hashlib.sha256(paths["candidates_csv"].read_bytes()).hexdigest(),
        "splits_csv": hashlib.sha256(paths["splits_csv"].read_bytes()).hexdigest(),
        "script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    _write_json(paths["json"], bundle)
    print(
        "ABLATION "
        + json.dumps(
            {
                "status": bundle["status"],
                "winner": (bundle.get("winner") or {}).get("winner"),
                "splits": bundle["splits_evaluated"],
                "preregistration_sha256": bundle["preregistration_sha256"],
                "artifacts": {k: str(v) for k, v in paths.items()},
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
