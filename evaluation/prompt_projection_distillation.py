"""Prompt-projection distillation contract for the B1 hardware-regular selector.

This module is the single source of truth for the *definitions* used to train
the explicit ``Wt[128,2560]`` / ``Wv[128,1024]`` projections of the B1
hardware-regular selector.  It contains no model loading, no CUDA and no
training loop: it defines the teacher target, the student score, the metrics,
the deterministic split, the projection hash, the on-disk feature-cache schema
and the listwise distillation losses.  The extractor and the trainer import
these definitions so that training and evaluation can never disagree.

Frozen definitions
------------------
Teacher (full dense visual path, 24 vision blocks)
    * ``merged_embeddings[G, 2560]``: the full-image merged visual embeddings
      (one row per 2x2 merge group, produced by the dense vision tower merger).
    * ``prompt_embeddings[T, 2560]``: token embeddings of the *raw user prompt
      text only* (``add_special_tokens=False``, no chat template, no image
      tokens) through the model input-embedding table.
    * ``teacher_raw[G] = max_t cosine(merged_embeddings[g], prompt_embeddings[t])``
      is saved as float32.
    * ``teacher_normalized = minmax(teacher_raw)`` in ``[0, 1]``.
    * ``teacher_soft = softmax(teacher_normalized / temperature)`` over groups.

Student (B1, after vision block 0 only)
    * ``group_mean[G, 1024]``: mean over the 4 patch positions of each merged
      group of the block-0 output.
    * ``Wv: 1024 -> 128`` and ``Wt: 2560 -> 128`` are stored as
      ``[128, source_dim]`` so ``features @ W.T`` projects.
    * ``student_raw[g] = max_t cosine(group_mean[g] @ Wv.T, prompt_embeddings[t] @ Wt.T)``.
      When ``group_mean`` is repeated over the 4 patch positions this is exactly
      ``hardware_regular_token_selector.semantic_cosine_scores``.

Metrics
    * ``pearson`` / ``spearman`` (average-rank ties, no SciPy) between the
      student score and the raw teacher score.
    * ``ndcg@K`` with the min-max teacher score as graded relevance.
    * for ``K`` in the configured budgets: teacher top-K ``recall``,
      ``jaccard`` and teacher saliency ``mass_retained`` (the teacher soft
      target restricted to the student top-K).
    * the same K metrics on the **real group set** produced by the frozen
      hardware selector at each budget, split into two explicit views:
      ``matched_real_budget`` (teacher top-``R`` with ``R = real_group_count``,
      the fair real-group comparison) and ``total_budget_coverage`` (teacher
      top-``K``; context groups occupy part of ``K`` so this is an upper-bound
      coverage, never a full-score).

Split contract
    * Every cached sample records the SHA256 of its source image file
      (``source_image_sha256``).  The train/val/test split is a hash draw over
      that image hash, so *all* prompts of one image land in exactly one split.
      Splitting by ``sample_id`` independently is forbidden for formal
      training, and a missing/empty image hash is a hard error.

Controls
    * Split evaluation reports the learned student with the correct prompts,
      an ``l2_baseline`` (per-group L2 norm of ``group_mean``) and a
      deterministic ``student_shuffled_prompt`` control that cyclically
      mismatches prompt embeddings within the split (sorted by ``sample_id``)
      using the same ``Wt``/``Wv``.  Splits with fewer than two samples report
      the shuffled control as ``unavailable``; nothing is fabricated.

The projections must be trained; this module never initialises or accepts
untrained random weights for inference.  The exported npz is compatible with
``evaluation.run_hf_hardware_regular_b1.load_projection``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from evaluation.hardware_regular_token_selector import (
    SOURCE_REAL,
    LowRankProjection,
    select_hardware_regular_groups,
)

FEATURE_DIM = 1024
PROMPT_DIM = 2560
SCORE_DIM = 128
MERGE_UNIT = 4
SUPPORTED_BUDGETS = (48, 64, 96)
DEFAULT_TEMPERATURE = 0.1
DEFAULT_LAMBDA_SEMANTIC = 0.5
DEFAULT_CONFIG_FILENAME = "prompt_projection_distillation.json"
_EPS = 1e-12

# npz keys written by the extractor.  The schema deliberately contains *no*
# dense hidden states: only the block-0 group mean, the raw prompt token
# embeddings and the teacher score vectors (plus scalar metadata).
CACHE_ARRAY_KEYS = (
    "group_mean",
    "prompt_embeddings",
    "teacher_scores",
    "teacher_scores_normalized",
    "teacher_soft_target",
)
CACHE_REQUIRED_ARRAY_KEYS = CACHE_ARRAY_KEYS
CACHE_DTYPE = {
    "group_mean": np.float16,
    "prompt_embeddings": np.float16,
    "teacher_scores": np.float32,
    "teacher_scores_normalized": np.float32,
    "teacher_soft_target": np.float32,
}
CACHE_METADATA_KEYS = (
    "sample_id",
    "task",
    "grid_h",
    "grid_w",
    "group_count",
    "prompt_token_count",
    "temperature",
    "dense_vision_blocks",
    "capture_layer",
    "prompt_tokenization",
    "add_special_tokens",
    "source_manifest",
    "source_image",
    "source_image_sha256",
    "config_sha256",
    "sample_sha256",
)
# Any key that would reveal the full dense hidden state is a schema violation.
FORBIDDEN_CACHE_KEYS = (
    "hidden",
    "hidden_states",
    "dense_hidden",
    "full_hidden",
    "block0_hidden",
    "deepstack",
    "image_embeds",
    "merged_embeddings",
)

PROJECTION_WT_KEY = "Wt"
PROJECTION_WV_KEY = "Wv"
PROJECTION_WEIGHTS_SHA256_KEY = "weights_sha256"
PROJECTION_METADATA_KEY = "metadata"
CANONICAL_PROJECTION_DTYPE = np.float32
SOURCE_IMAGE_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
IMAGE_HASH_CHUNK_BYTES = 1 << 20
PRIMARY_PROMOTION_METRICS = ("spearman", "ndcg_full")
SHUFFLED_UNAVAILABLE_REASON = "split has fewer than two samples"


class DistillationContractError(ValueError):
    """Raised when the distillation contract is violated."""


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------


def _finite_matrix(name: str, value: Any, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise DistillationContractError(
            f"{name} must have ndim={ndim}, got ndim={array.ndim}"
        )
    if array.size == 0:
        raise DistillationContractError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise DistillationContractError(f"{name} must contain only finite values")
    return array


def _check_axis(name: str, array: np.ndarray, axis: int, expected: int) -> None:
    if array.shape[axis] != expected:
        raise DistillationContractError(
            f"{name} axis {axis} must be {expected}, got {array.shape[axis]}"
        )


def validate_source_image_sha256(value: Any) -> str:
    """Return the lower-case 64-hex image SHA256 or reject the sample.

    Formal training groups samples by ``source_image_sha256`` so that every
    prompt of one image is kept in a single split.  A missing, empty or
    malformed hash is a hard error: silently falling back to ``sample_id``
    would reintroduce same-image leakage.
    """
    if value is None:
        raise DistillationContractError(
            "source_image_sha256 is missing; formal training refuses samples "
            "without an image hash"
        )
    text = str(value).strip().lower()
    if not SOURCE_IMAGE_SHA256_HEX.fullmatch(text):
        raise DistillationContractError(
            "source_image_sha256 must be a 64-character lower-case hex SHA256, "
            f"got {value!r}"
        )
    return text


def image_file_sha256(path: Path | str) -> str:
    """Streaming SHA256 of the raw image file bytes (not the decoded pixels)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(IMAGE_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_group_mean(group_mean: Any) -> np.ndarray:
    array = _finite_matrix("group_mean", group_mean, 2)
    _check_axis("group_mean", array, 1, FEATURE_DIM)
    return array


def validate_prompt_embeddings(prompt_embeddings: Any) -> np.ndarray:
    array = _finite_matrix("prompt_embeddings", prompt_embeddings, 2)
    _check_axis("prompt_embeddings", array, 1, PROMPT_DIM)
    return array


def validate_projection_matrices(Wt: Any, Wv: Any) -> tuple[np.ndarray, np.ndarray]:
    wt = np.asarray(Wt, dtype=np.float32)
    wv = np.asarray(Wv, dtype=np.float32)
    if wt.shape != (SCORE_DIM, PROMPT_DIM):
        raise DistillationContractError(
            f"Wt must have shape ({SCORE_DIM}, {PROMPT_DIM}), got {wt.shape}"
        )
    if wv.shape != (SCORE_DIM, FEATURE_DIM):
        raise DistillationContractError(
            f"Wv must have shape ({SCORE_DIM}, {FEATURE_DIM}), got {wv.shape}"
        )
    if not np.isfinite(wt).all() or not np.isfinite(wv).all():
        raise DistillationContractError("Wt/Wv must contain only finite values")
    return wt, wv


def _unit_rows(matrix: np.ndarray, name: str) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= _EPS):
        raise DistillationContractError(
            f"{name} contains a zero-norm row; cosine is undefined"
        )
    return matrix / norms[:, None]


# ---------------------------------------------------------------------------
# Teacher target and student score
# ---------------------------------------------------------------------------


def _max_cosine(visual: np.ndarray, prompt: np.ndarray, name: str) -> np.ndarray:
    if visual.shape[1] != prompt.shape[1]:
        raise DistillationContractError(
            f"{name}: visual dim {visual.shape[1]} != prompt dim {prompt.shape[1]}"
        )
    visual_unit = _unit_rows(visual, f"{name}.visual")
    prompt_unit = _unit_rows(prompt, f"{name}.prompt")
    return (visual_unit @ prompt_unit.T).max(axis=1)


def teacher_scores(merged_embeddings: Any, prompt_embeddings: Any) -> np.ndarray:
    """Raw teacher score: per merged group, max cosine over prompt tokens."""
    merged = _finite_matrix("merged_embeddings", merged_embeddings, 2)
    _check_axis("merged_embeddings", merged, 1, PROMPT_DIM)
    prompt = validate_prompt_embeddings(prompt_embeddings)
    return _max_cosine(merged, prompt, "teacher").astype(np.float32)


def student_cosine_scores(
    group_mean: Any, prompt_embeddings: Any, Wt: Any, Wv: Any
) -> np.ndarray:
    """Raw student score: projected group-mean vs projected prompt tokens."""
    mean = validate_group_mean(group_mean)
    prompt = validate_prompt_embeddings(prompt_embeddings)
    wt, wv = validate_projection_matrices(Wt, Wv)
    visual_proj = mean @ wv.T.astype(np.float64)
    prompt_proj = prompt @ wt.T
    return _max_cosine(visual_proj, prompt_proj, "student").astype(np.float32)


def minmax_normalize_1d(values: Any) -> np.ndarray:
    """Min-max normalise a 1-D vector into ``[0, 1]`` (all-equal -> zeros)."""
    array = _finite_matrix("values", values, 1)
    low = float(array.min())
    high = float(array.max())
    if high - low <= _EPS:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - low) / (high - low)).astype(np.float32)


def temperature_soft_target(normalized_scores: Any, temperature: float) -> np.ndarray:
    """``softmax(normalized / temperature)`` with an explicit temperature."""
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise DistillationContractError("temperature must be finite and > 0")
    array = _finite_matrix("normalized_scores", normalized_scores, 1)
    logits = (array - float(array.max())) / float(temperature)
    exp = np.exp(logits)
    soft = exp / exp.sum()
    return soft.astype(np.float32)


def teacher_target(
    merged_embeddings: Any, prompt_embeddings: Any, temperature: float
) -> dict[str, np.ndarray]:
    """Return raw, min-max normalised and temperature soft teacher targets."""
    raw = teacher_scores(merged_embeddings, prompt_embeddings)
    normalized = minmax_normalize_1d(raw)
    soft = temperature_soft_target(normalized, temperature)
    return {"raw": raw, "normalized": normalized, "soft": soft}


# ---------------------------------------------------------------------------
# Metrics (no SciPy)
# ---------------------------------------------------------------------------


def stable_rank(values: Any) -> np.ndarray:
    """Average ranks (1-based) with a deterministic tie policy.

    Equal values all receive the mean of the ranks they span; the tie groups
    are found on a stable sort so the result depends only on the values, never
    on the input order.
    """
    array = _finite_matrix("values", values, 1)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(array.shape[0], dtype=np.float64)
    index = 0
    while index < order.shape[0]:
        end = index
        while (
            end + 1 < order.shape[0]
            and array[order[end + 1]] == array[order[index]]
        ):
            end += 1
        average = 0.5 * ((index + 1) + (end + 1))
        ranks[order[index : end + 1]] = average
        index = end + 1
    return ranks


def pearson(a: Any, b: Any) -> float:
    """Pearson correlation; zero-variance input returns 0.0."""
    x = _finite_matrix("a", a, 1)
    y = _finite_matrix("b", b, 1)
    if x.shape != y.shape:
        raise DistillationContractError("pearson inputs must have equal shape")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = math.sqrt(float((x_centered**2).sum()) * float((y_centered**2).sum()))
    if denominator <= _EPS:
        return 0.0
    return float((x_centered * y_centered).sum() / denominator)


def spearman(a: Any, b: Any) -> float:
    """Spearman correlation via :func:`stable_rank` (average-rank ties)."""
    return pearson(stable_rank(a), stable_rank(b))


def topk_indices(scores: Any, k: int) -> np.ndarray:
    array = _finite_matrix("scores", scores, 1)
    if k <= 0:
        raise DistillationContractError("k must be positive")
    count = min(int(k), array.shape[0])
    order = np.argsort(-array, kind="stable")
    return order[:count]


def ndcg_at_k(scores: Any, relevance: Any, k: int) -> float:
    """NDCG@K with graded relevance; sorted by descending predicted score."""
    array = _finite_matrix("scores", scores, 1)
    rel = _finite_matrix("relevance", relevance, 1)
    if array.shape != rel.shape:
        raise DistillationContractError("ndcg scores and relevance must match")
    if np.any(rel < 0.0):
        raise DistillationContractError("ndcg relevance must be non-negative")
    k = min(int(k), array.shape[0])
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    gains = rel[topk_indices(array, k)]
    dcg = float((gains * discounts).sum())
    ideal_gains = np.sort(rel)[::-1][:k]
    idcg = float((ideal_gains * discounts).sum())
    if idcg <= _EPS:
        return 0.0
    return dcg / idcg


def ndcg_for_selected_indices(
    selected: Any, scores: Any, relevance: Any, k: int
) -> float:
    """NDCG@K restricted to ``selected`` groups, ranked by descending ``scores``.

    Only indices in ``selected`` may enter the predicted ranking.  The predicted
    list is the selected indices sorted by descending ``scores`` with the same
    deterministic low-index tie policy as :func:`topk_indices` (stable descending
    score, ties broken by ascending original group index).  DCG is computed on
    the first ``k`` entries of that restricted list against ``relevance``; the
    ideal DCG@K is the all-group ideal used by :func:`ndcg_at_k`, so an identity
    selection reproduces :func:`ndcg_at_k` exactly.  This helper never changes
    the semantics of :func:`ndcg_at_k`.

    Validation raises :class:`DistillationContractError` for non-finite inputs, a
    shape mismatch, negative relevance, a non-positive ``k``, and empty,
    duplicated or out-of-range selected indices.
    """
    array = _finite_matrix("scores", scores, 1)
    rel = _finite_matrix("relevance", relevance, 1)
    if array.shape != rel.shape:
        raise DistillationContractError("ndcg scores and relevance must match")
    if np.any(rel < 0.0):
        raise DistillationContractError("ndcg relevance must be non-negative")
    if int(k) <= 0:
        raise DistillationContractError("k must be positive")
    indices = np.asarray(selected, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise DistillationContractError("selected indices must be non-empty")
    if np.unique(indices).size != indices.size:
        raise DistillationContractError("selected indices must be unique")
    if indices.min() < 0 or indices.max() >= array.shape[0]:
        raise DistillationContractError("selected indices out of range for scores")
    count = min(int(k), int(indices.size))
    order = np.lexsort((indices, -array[indices]))[:count]
    ordered = indices[order]
    discounts = 1.0 / np.log2(np.arange(2, count + 2, dtype=np.float64))
    dcg = float((rel[ordered] * discounts).sum())
    ideal_k = min(int(k), array.shape[0])
    ideal_gains = np.sort(rel)[::-1][:ideal_k]
    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_k + 2, dtype=np.float64))
    idcg = float((ideal_gains * ideal_discounts).sum())
    if idcg <= _EPS:
        return 0.0
    return dcg / idcg


def jaccard(predicted: Iterable[int], teacher: Iterable[int]) -> float:
    pred_set = {int(value) for value in predicted}
    teacher_set = {int(value) for value in teacher}
    union = pred_set | teacher_set
    if not union:
        return 0.0
    return len(pred_set & teacher_set) / len(union)


def teacher_topk_recall(predicted: Iterable[int], teacher_topk: Iterable[int]) -> float:
    teacher_set = {int(value) for value in teacher_topk}
    if not teacher_set:
        return 0.0
    pred_set = {int(value) for value in predicted}
    return len(pred_set & teacher_set) / len(teacher_set)


def saliency_mass_retained(
    selected: Iterable[int], saliency: Any
) -> float:
    """Fraction of total teacher saliency mass inside ``selected``."""
    mass = _finite_matrix("saliency", saliency, 1)
    if np.any(mass < 0.0):
        raise DistillationContractError("saliency must be non-negative")
    total = float(mass.sum())
    if total <= _EPS:
        return 0.0
    indices = np.asarray(sorted({int(value) for value in selected}), dtype=np.int64)
    if indices.size == 0:
        return 0.0
    if indices.min() < 0 or indices.max() >= mass.shape[0]:
        raise DistillationContractError("selected index out of range for saliency")
    return float(mass[indices].sum()) / total


def _k_metrics(
    pred_scores: np.ndarray,
    teacher_raw: np.ndarray,
    teacher_normalized: np.ndarray,
    teacher_soft: np.ndarray,
    k: int,
) -> dict[str, float]:
    pred_topk = topk_indices(pred_scores, k)
    teacher_topk = topk_indices(teacher_raw, k)
    return {
        "k": int(k),
        "ndcg": ndcg_at_k(pred_scores, teacher_normalized, k),
        "topk_recall": teacher_topk_recall(pred_topk, teacher_topk),
        "jaccard": jaccard(pred_topk, teacher_topk),
        "saliency_mass_retained": saliency_mass_retained(pred_topk, teacher_soft),
    }


def metric_bundle(
    pred_scores: Any,
    teacher_raw: Any,
    teacher_normalized: Any,
    teacher_soft: Any,
    *,
    budgets: Sequence[int] = SUPPORTED_BUDGETS,
) -> dict[str, Any]:
    """Rank-level metrics for one sample at every budget ``K``."""
    pred = _finite_matrix("pred_scores", pred_scores, 1)
    raw = _finite_matrix("teacher_raw", teacher_raw, 1)
    normalized = _finite_matrix("teacher_normalized", teacher_normalized, 1)
    soft = _finite_matrix("teacher_soft", teacher_soft, 1)
    if not (pred.shape == raw.shape == normalized.shape == soft.shape):
        raise DistillationContractError(
            "metric_bundle inputs must all share the same shape"
        )
    return {
        "count": int(pred.shape[0]),
        "pearson": pearson(pred, raw),
        "spearman": spearman(pred, raw),
        "ndcg_full": ndcg_at_k(pred, normalized, pred.shape[0]),
        "per_budget": {
            str(int(k)): _k_metrics(pred, raw, normalized, soft, int(k))
            for k in budgets
        },
    }


def l2_baseline_scores(group_mean: Any) -> np.ndarray:
    """No-training L2 baseline: per-group L2 norm of ``group_mean``.

    This equals ``hardware_regular_token_selector.group_l2_saliency`` applied to
    the ``[G, 4, 1024]`` broadcast, because that function uses the per-group
    mean.  It is the baseline the formal projection must beat on val.
    """
    mean = validate_group_mean(group_mean)
    return np.linalg.norm(mean, axis=1).astype(np.float32)


def cyclic_shuffled_prompt_map(sample_ids: Iterable[str]) -> dict[str, str] | None:
    """Deterministic cyclic prompt mismatch within one split.

    ``sample_id`` values are sorted ascending and sample ``i`` receives the
    prompt embeddings of sample ``i+1 mod n``.  For ``n >= 2`` this is a
    derangement (no sample keeps its own prompt).  For ``n < 2`` the control is
    undefined and ``None`` is returned; callers must report it as unavailable
    rather than fabricate a value.
    """
    ordered = sorted(str(value) for value in sample_ids)
    if len(ordered) < 2:
        return None
    return {
        sample_id: ordered[(index + 1) % len(ordered)]
        for index, sample_id in enumerate(ordered)
    }


def rank_metric_differences(
    reference: Mapping[str, Any], comparison: Mapping[str, Any]
) -> dict[str, Any]:
    """Paired ``reference - comparison`` deltas for the rank/K metrics.

    Only the metric leaves are subtracted (``count`` is ignored).  Callers use
    this for ``student_correct - student_shuffled_prompt`` and the val
    promotion gate.
    """
    result: dict[str, Any] = {
        "pearson": float(reference["pearson"]) - float(comparison["pearson"]),
        "spearman": float(reference["spearman"]) - float(comparison["spearman"]),
        "ndcg_full": float(reference["ndcg_full"]) - float(comparison["ndcg_full"]),
        "per_budget": {},
    }
    for budget, entry in reference["per_budget"].items():
        other = comparison["per_budget"][budget]
        result["per_budget"][budget] = {
            "ndcg": float(entry["ndcg"]) - float(other["ndcg"]),
            "topk_recall": float(entry["topk_recall"]) - float(other["topk_recall"]),
            "jaccard": float(entry["jaccard"]) - float(other["jaccard"]),
            "saliency_mass_retained": float(entry["saliency_mass_retained"])
            - float(other["saliency_mass_retained"]),
        }
    return result


def _primary_metric_signs(
    deltas: Mapping[str, Any],
    *,
    primary: Sequence[str] = PRIMARY_PROMOTION_METRICS,
) -> dict[str, bool]:
    return {name: bool(float(deltas[name]) > 0.0) for name in primary}


def promotion_gate(
    split_metrics: Mapping[str, Any],
    *,
    val_split: str = "val",
    primary: Sequence[str] = PRIMARY_PROMOTION_METRICS,
) -> dict[str, Any]:
    """Sign-only val gate: student beats L2 and the correct prompts beat shuffled.

    No numeric margin is fixed a priori; the criterion is a strictly positive
    paired delta on every primary rank metric.  When the shuffled control is
    unavailable (fewer than two samples) the gate is reported as
    ``UNAVAILABLE`` instead of being silently passed.
    """
    gate: dict[str, Any] = {
        "primary_metrics": list(primary),
        "criterion": (
            "strictly positive paired delta on every primary metric; no "
            "arbitrary numeric threshold is fixed a priori"
        ),
        "test_discipline": (
            "test is evaluated once after the best epoch is frozen and never "
            "influences selection"
        ),
    }
    if val_split not in split_metrics:
        gate["status"] = "UNAVAILABLE"
        gate["reason"] = f"{val_split} split metrics are missing"
        return gate
    val = split_metrics[val_split]
    correct_rank = val["student_correct"]["rank"]
    l2_rank = val["l2_baseline"]["rank"]
    shuffled = val["student_shuffled_prompt"]
    correct_minus_l2 = rank_metric_differences(correct_rank, l2_rank)
    gate["student_correct_minus_l2_baseline"] = correct_minus_l2
    gate["student_beats_l2"] = all(_primary_metric_signs(correct_minus_l2, primary=primary).values())
    if not shuffled.get("available", False):
        gate["status"] = "UNAVAILABLE"
        gate["reason"] = str(shuffled.get("reason", SHUFFLED_UNAVAILABLE_REASON))
        gate["student_correct_minus_shuffled"] = {"available": False}
        gate["correct_beats_shuffled"] = None
        return gate
    correct_minus_shuffled = rank_metric_differences(correct_rank, shuffled["rank"])
    gate["student_correct_minus_shuffled"] = {
        "available": True,
        **correct_minus_shuffled,
    }
    gate["correct_beats_shuffled"] = all(
        _primary_metric_signs(correct_minus_shuffled, primary=primary).values()
    )
    gate["status"] = "EVALUATED"
    return gate


# ---------------------------------------------------------------------------
# Hardware-constrained real-group metrics
# ---------------------------------------------------------------------------


def _group_mean_to_hidden(group_mean: np.ndarray) -> np.ndarray:
    """Expand ``[G, 1024]`` to the selector's ``[G, 4, 1024]`` contract.

    Semantic cosine and L2 saliency are both functions of the per-group mean,
    so this broadcast is value-exact for every real-group decision the frozen
    hardware selector makes; it only avoids storing per-patch block-0 output.
    """
    return np.repeat(group_mean[:, None, :], MERGE_UNIT, axis=1)


def hardware_real_group_metrics(
    group_mean: Any,
    grid_h: int,
    grid_w: int,
    *,
    text_embeddings: Any,
    Wt: Any,
    Wv: Any,
    lambda_semantic: float,
    teacher_raw: Any,
    teacher_soft: Any,
    budgets: Sequence[int] = SUPPORTED_BUDGETS,
    region_grid: int = 3,
) -> dict[str, Any]:
    """Two clearly separated K metrics on the real group set.

    At a requested budget ``K`` the frozen selector emits ``real_group_count``
    real groups plus up to nine context groups, so a single top-K number is
    ambiguous.  Each budget therefore reports:

    * ``matched_real_budget``: with ``R = real_group_count``, compare the
      selected real groups against the teacher top-``R`` (recall, Jaccard,
      saliency mass).  This is the fair real-vs-real comparison.
    * ``total_budget_coverage``: compare the selected real groups against the
      teacher top-``K``.  Context groups share the ``K`` slots, so this is a
      coverage upper bound, explicitly flagged and never a full score.
    """
    mean = validate_group_mean(group_mean)
    group_count = mean.shape[0]
    if int(grid_h) * int(grid_w) != group_count:
        raise DistillationContractError(
            f"grid {grid_h}x{grid_w} does not match {group_count} groups"
        )
    raw = _finite_matrix("teacher_raw", teacher_raw, 1)
    soft = _finite_matrix("teacher_soft", teacher_soft, 1)
    projection = LowRankProjection(Wt=Wt, Wv=Wv)
    hidden = _group_mean_to_hidden(mean)
    results: dict[str, Any] = {}
    for budget in budgets:
        selection = select_hardware_regular_groups(
            hidden,
            int(grid_h),
            int(grid_w),
            budget=int(budget),
            text_embeddings=text_embeddings,
            projection=projection,
            lambda_semantic=lambda_semantic,
            region_grid=region_grid,
        )
        real = np.asarray(selection.group_indices)[
            np.asarray(selection.source_type) == SOURCE_REAL
        ]
        real_list = [int(value) for value in real]
        k = int(budget)
        real_budget = int(real.size)
        teacher_top_r = topk_indices(raw, real_budget) if real_budget > 0 else []
        teacher_top_k = topk_indices(raw, k)
        results[str(k)] = {
            "budget": k,
            "bypass": bool(selection.bypass),
            "real_group_count": real_budget,
            "context_group_count": int(selection.context_group_count),
            "matched_real_budget": {
                "real_budget": real_budget,
                "teacher_top_r_recall": teacher_topk_recall(real_list, teacher_top_r),
                "jaccard": jaccard(real_list, teacher_top_r),
                "saliency_mass_retained": saliency_mass_retained(real_list, soft),
                "definition": (
                    "R = real_group_count; selected real groups vs teacher "
                    "top-R. Fair real-vs-real comparison."
                ),
            },
            "total_budget_coverage": {
                "budget": k,
                "teacher_topk_recall": teacher_topk_recall(real_list, teacher_top_k),
                "jaccard": jaccard(real_list, teacher_top_k),
                "saliency_mass_retained": saliency_mass_retained(real_list, soft),
                "upper_bound_only": True,
                "definition": (
                    "selected real groups vs teacher top-K; context groups "
                    "occupy part of K, so this is a coverage upper bound, not "
                    "a full score."
                ),
            },
        }
    return results


# ---------------------------------------------------------------------------
# Deterministic hash-based split
# ---------------------------------------------------------------------------


def _validate_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    missing = [name for name in ("train", "val", "test") if name not in ratios]
    if missing:
        raise DistillationContractError(f"split ratios missing keys: {missing}")
    cleaned = {name: float(ratios[name]) for name in ("train", "val", "test")}
    if any(value < 0.0 or not math.isfinite(value) for value in cleaned.values()):
        raise DistillationContractError("split ratios must be finite and >= 0")
    if abs(sum(cleaned.values()) - 1.0) > 1e-9:
        raise DistillationContractError("split ratios must sum to 1.0")
    return cleaned


def assign_split(
    source_image_sha256: str,
    *,
    seed: int,
    ratios: Mapping[str, float],
    salt: str,
) -> str:
    """Map an image SHA256 to train/val/test by a stable SHA256 draw.

    The draw depends only on ``(salt, seed, source_image_sha256)`` so it is
    stable across processes and runs, and every prompt of one image is forced
    into exactly one split.  A missing/invalid image hash is rejected.
    """
    cleaned = _validate_ratios(ratios)
    image_hash = validate_source_image_sha256(source_image_sha256)
    key = f"{salt}|{int(seed)}|{image_hash}".encode("utf-8")
    draw = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / float(2**64)
    if draw < cleaned["train"]:
        return "train"
    if draw < cleaned["train"] + cleaned["val"]:
        return "val"
    return "test"


def _split_record_sample_id_and_image(
    record: Any,
) -> tuple[str, str]:
    if not isinstance(record, Mapping):
        raise DistillationContractError(
            "split inputs must be mappings with 'sample_id' and "
            "'source_image_sha256'; splitting by sample_id alone is forbidden "
            "for formal training"
        )
    sample_id = record.get("sample_id")
    if sample_id is None or str(sample_id).strip() == "":
        raise DistillationContractError("split record is missing sample_id")
    return str(sample_id), validate_source_image_sha256(
        record.get("source_image_sha256")
    )


def split_sample_ids(
    samples: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    ratios: Mapping[str, float],
    salt: str,
) -> dict[str, list[str]]:
    """Group-hash split by ``source_image_sha256`` with leakage assertions.

    Every record must carry ``sample_id`` and ``source_image_sha256``.  All
    records sharing an image hash are assigned to the same split, and the
    pairwise-disjoint check is performed on both sample ids and image hashes.
    """
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    seen_ids: set[str] = set()
    image_to_split: dict[str, str] = {}
    for record in samples:
        sample_id, image_hash = _split_record_sample_id_and_image(record)
        if sample_id in seen_ids:
            raise DistillationContractError(f"duplicate sample_id {sample_id!r}")
        seen_ids.add(sample_id)
        name = image_to_split.get(image_hash)
        if name is None:
            name = assign_split(image_hash, seed=seed, ratios=ratios, salt=salt)
            image_to_split[image_hash] = name
        splits[name].append(sample_id)
    for name in splits:
        splits[name].sort()
    as_sets = {name: set(value) for name, value in splits.items()}
    if sum(len(value) for value in as_sets.values()) != len(seen_ids):
        raise DistillationContractError("split assignment dropped or duplicated ids")
    if as_sets["train"] & as_sets["val"]:
        raise DistillationContractError("train/val split leakage")
    if as_sets["train"] & as_sets["test"]:
        raise DistillationContractError("train/test split leakage")
    if as_sets["val"] & as_sets["test"]:
        raise DistillationContractError("val/test split leakage")
    return splits


def split_group_summary(
    splits: Mapping[str, Sequence[str]], image_hashes: Mapping[str, str]
) -> dict[str, Any]:
    """Per-split sample/image counts plus a cross-split image leakage check."""
    summary: dict[str, Any] = {}
    for name, values in splits.items():
        hashes = {validate_source_image_sha256(image_hashes[sample_id]) for sample_id in values}
        summary[name] = {
            "sample_count": len(values),
            "image_count": len(hashes),
        }
    names = list(splits)
    for index, name in enumerate(names):
        current = {image_hashes[sample_id] for sample_id in splits[name]}
        for other in names[index + 1 :]:
            shared = current & {image_hashes[sample_id] for sample_id in splits[other]}
            if shared:
                raise DistillationContractError(
                    f"image hash leakage between {name} and {other}: {sorted(shared)}"
                )
    return summary


# ---------------------------------------------------------------------------
# Projection content hash and npz export
# ---------------------------------------------------------------------------


def canonical_projection_sha256(Wt: Any, Wv: Any) -> str:
    """Content hash matching ``run_hf_hardware_regular_b1`` exactly.

    Each matrix is cast to float32, made C-contiguous and hashed together with
    its key name, dtype string and shape, so any value/shape/dtype change
    changes the digest.
    """
    digest = hashlib.sha256()
    for name, array in (("Wt", Wt), ("Wv", Wv)):
        canonical = np.ascontiguousarray(
            np.asarray(array, dtype=CANONICAL_PROJECTION_DTYPE)
        )
        header = f"{name}:{canonical.dtype.str}:{canonical.shape}".encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def save_projection_npz(
    path: Path, Wt: Any, Wv: Any, metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Write float32 ``Wt``/``Wv`` plus the canonical ``weights_sha256`` key."""
    wt, wv = validate_projection_matrices(Wt, Wv)
    digest = canonical_projection_sha256(wt, wv)
    payload: dict[str, Any] = {
        PROJECTION_WT_KEY: wt,
        PROJECTION_WV_KEY: wv,
        PROJECTION_WEIGHTS_SHA256_KEY: np.asarray(digest),
    }
    meta = dict(metadata or {})
    meta.setdefault("weights_sha256", digest)
    meta.setdefault("dtype", "float32")
    meta.setdefault("Wt_shape", list(wt.shape))
    meta.setdefault("Wv_shape", list(wv.shape))
    payload[PROJECTION_METADATA_KEY] = np.asarray(
        json.dumps(meta, ensure_ascii=False, sort_keys=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return {"weights_sha256": digest, "bytes": path.stat().st_size, "metadata": meta}


# ---------------------------------------------------------------------------
# Feature-cache schema
# ---------------------------------------------------------------------------


def _cache_sample_fingerprint(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in CACHE_ARRAY_KEYS:
        array = np.ascontiguousarray(np.asarray(arrays[key]))
        digest.update(f"{key}:{array.dtype.str}:{array.shape}".encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def save_feature_cache(
    path: Path,
    *,
    group_mean: Any,
    prompt_embeddings: Any,
    teacher_raw: Any,
    teacher_normalized: Any,
    teacher_soft: Any,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist one sample with the frozen float16/float32 npz schema."""
    mean = np.ascontiguousarray(
        np.asarray(group_mean, dtype=CACHE_DTYPE["group_mean"])
    )
    prompt = np.ascontiguousarray(
        np.asarray(prompt_embeddings, dtype=CACHE_DTYPE["prompt_embeddings"])
    )
    raw = np.ascontiguousarray(
        np.asarray(teacher_raw, dtype=CACHE_DTYPE["teacher_scores"])
    )
    normalized = np.ascontiguousarray(
        np.asarray(teacher_normalized, dtype=CACHE_DTYPE["teacher_scores_normalized"])
    )
    soft = np.ascontiguousarray(
        np.asarray(teacher_soft, dtype=CACHE_DTYPE["teacher_soft_target"])
    )
    if mean.ndim != 2 or mean.shape[1] != FEATURE_DIM:
        raise DistillationContractError(f"group_mean must be [G,{FEATURE_DIM}]")
    if prompt.ndim != 2 or prompt.shape[1] != PROMPT_DIM:
        raise DistillationContractError(f"prompt_embeddings must be [T,{PROMPT_DIM}]")
    for name, array in (
        ("teacher_scores", raw),
        ("teacher_scores_normalized", normalized),
        ("teacher_soft_target", soft),
    ):
        if array.ndim != 1 or array.shape[0] != mean.shape[0]:
            raise DistillationContractError(f"{name} must be [G]")
    arrays = {
        "group_mean": mean,
        "prompt_embeddings": prompt,
        "teacher_scores": raw,
        "teacher_scores_normalized": normalized,
        "teacher_soft_target": soft,
    }
    meta = dict(metadata)
    fingerprint = _cache_sample_fingerprint(arrays)
    meta.setdefault("sample_sha256", fingerprint)
    meta.setdefault("group_count", int(mean.shape[0]))
    meta.setdefault("prompt_token_count", int(prompt.shape[0]))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = dict(arrays)
    for key in CACHE_METADATA_KEYS:
        if key in meta:
            payload[f"meta_{key}"] = np.asarray(meta[key])
    np.savez(path, **payload)
    return {
        "path": str(path),
        "sample_sha256": fingerprint,
        "bytes": path.stat().st_size,
        "schema_keys": list(arrays),
    }


def load_feature_cache(path: Path, *, validate: bool = True) -> dict[str, Any]:
    """Load and (optionally) validate one cached sample npz."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"feature cache does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        forbidden = sorted(key for key in FORBIDDEN_CACHE_KEYS if key in keys)
        if forbidden:
            raise DistillationContractError(
                f"cache {path} stores forbidden dense keys: {forbidden}"
            )
        missing = [key for key in CACHE_REQUIRED_ARRAY_KEYS if key not in keys]
        if missing:
            raise DistillationContractError(f"cache {path} missing keys: {missing}")
        arrays = {key: np.array(archive[key]) for key in CACHE_ARRAY_KEYS}
        metadata = {
            key: archive[f"meta_{key}"]
            for key in CACHE_METADATA_KEYS
            if f"meta_{key}" in keys
        }
    sample: dict[str, Any] = {"path": str(path), **arrays, "metadata": metadata}
    if validate:
        expected_dtypes = CACHE_DTYPE
        for key, dtype in expected_dtypes.items():
            if arrays[key].dtype != np.dtype(dtype):
                raise DistillationContractError(
                    f"cache {path} key {key} dtype {arrays[key].dtype} != {dtype}"
                )
        group_count = arrays["teacher_scores"].shape[0]
        if arrays["group_mean"].shape[0] != group_count:
            raise DistillationContractError(
                f"cache {path} group_mean rows != teacher_scores length"
            )
        for key in ("teacher_scores_normalized", "teacher_soft_target"):
            if arrays[key].shape[0] != group_count:
                raise DistillationContractError(f"cache {path} {key} length mismatch")
        if not np.isfinite(arrays["group_mean"]).all():
            raise DistillationContractError(f"cache {path} group_mean not finite")
        if not np.isfinite(arrays["prompt_embeddings"]).all():
            raise DistillationContractError(
                f"cache {path} prompt_embeddings not finite"
            )
        if not np.isfinite(arrays["teacher_scores"]).all():
            raise DistillationContractError(f"cache {path} teacher_scores not finite")
        if abs(float(arrays["teacher_soft_target"].sum()) - 1.0) > 1e-3:
            raise DistillationContractError(
                f"cache {path} teacher_soft_target does not sum to 1"
            )
        fingerprint = _cache_sample_fingerprint(arrays)
        stored = metadata.get("sample_sha256")
        if stored is not None and str(stored) != fingerprint:
            raise DistillationContractError(
                f"cache {path} sample_sha256 mismatch"
            )
    return sample


# ---------------------------------------------------------------------------
# Listwise distillation losses
# ---------------------------------------------------------------------------


def _require_torch() -> Any:
    import torch

    return torch


def minmax_normalize_torch(values: Any) -> Any:
    """Per-row min-max normalisation of a ``[B, G]`` tensor into ``[0, 1]``."""
    torch = _require_torch()
    low = values.min(dim=-1, keepdim=True).values
    high = values.max(dim=-1, keepdim=True).values
    span = (high - low).clamp_min(_EPS)
    return (values - low) / span


def student_cosine_scores_torch(
    group_mean: Any, prompt_embeddings: Any, Wt: Any, Wv: Any
) -> Any:
    """Differentiable student score ``[G]`` for one sample (torch tensors)."""
    torch = _require_torch()
    visual_proj = group_mean @ Wv.transpose(0, 1)
    prompt_proj = prompt_embeddings @ Wt.transpose(0, 1)
    visual_unit = visual_proj / visual_proj.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    prompt_unit = prompt_proj / prompt_proj.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    return (visual_unit @ prompt_unit.transpose(0, 1)).max(dim=-1).values


def listwise_kl_loss(
    student_scores: Any,
    teacher_soft: Any,
    temperature: float,
    *,
    temperature_squared: bool = True,
) -> Any:
    """KL(teacher || softmax(student/T)), optionally scaled by ``T**2``."""
    torch = _require_torch()
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise DistillationContractError("temperature must be finite and > 0")
    teacher = teacher_soft / teacher_soft.sum(dim=-1, keepdim=True).clamp_min(_EPS)
    student_log_prob = torch.log_softmax(student_scores / float(temperature), dim=-1)
    kl = (
        teacher * (torch.log(teacher.clamp_min(_EPS)) - student_log_prob)
    ).sum(dim=-1).mean()
    if temperature_squared:
        kl = kl * float(temperature) ** 2
    return kl


def score_mse_loss(student_scores: Any, teacher_normalized: Any) -> Any:
    """MSE between per-sample min-max normalised student and teacher scores."""
    torch = _require_torch()
    return torch.mean((minmax_normalize_torch(student_scores) - teacher_normalized) ** 2)


def distillation_loss(
    student_scores: Any,
    teacher_soft: Any,
    teacher_normalized: Any,
    *,
    temperature: float,
    listwise_kl_weight: float,
    score_mse_weight: float,
    temperature_squared: bool = True,
) -> tuple[Any, dict[str, float]]:
    """Weighted sum of the listwise KL and the optional score MSE."""
    if listwise_kl_weight < 0.0 or score_mse_weight < 0.0:
        raise DistillationContractError("loss weights must be >= 0")
    kl = listwise_kl_loss(
        student_scores,
        teacher_soft,
        temperature,
        temperature_squared=temperature_squared,
    )
    total = float(listwise_kl_weight) * kl
    parts = {
        "listwise_kl": float(kl.detach()),
        "score_mse": 0.0,
    }
    if score_mse_weight > 0.0:
        mse = score_mse_loss(student_scores, teacher_normalized)
        total = total + float(score_mse_weight) * mse
        parts["score_mse"] = float(mse.detach())
    parts["total"] = float(total.detach())
    return total, parts


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_distillation_config(path: Path | str) -> dict[str, Any]:
    """Load and validate the distillation JSON configuration."""
    text = Path(path).read_text(encoding="utf-8")
    config = json.loads(text)
    required = (
        "feature_dim",
        "prompt_dim",
        "score_dim",
        "merge_unit",
        "temperature",
        "lambda_semantic",
        "budgets",
        "seed",
        "loss",
        "optim",
        "split",
        "metrics",
        "controls",
        "promotion_criteria",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise DistillationContractError(f"config missing keys: {missing}")
    for key, expected in (
        ("feature_dim", FEATURE_DIM),
        ("prompt_dim", PROMPT_DIM),
        ("score_dim", SCORE_DIM),
        ("merge_unit", MERGE_UNIT),
    ):
        if int(config[key]) != expected:
            raise DistillationContractError(
                f"config {key} must be {expected}, got {config[key]}"
            )
    if float(config["temperature"]) <= 0.0:
        raise DistillationContractError("config temperature must be > 0")
    lambda_semantic = float(config["lambda_semantic"])
    if not 0.0 <= lambda_semantic <= 1.0:
        raise DistillationContractError("config lambda_semantic must be in [0, 1]")
    budgets = [int(value) for value in config["budgets"]]
    if not budgets or any(value % 16 != 0 or value <= 0 for value in budgets):
        raise DistillationContractError(
            "config budgets must be positive multiples of 16"
        )
    loss = config["loss"]
    for key in ("listwise_kl_weight", "score_mse_weight"):
        if float(loss[key]) < 0.0:
            raise DistillationContractError(f"config loss.{key} must be >= 0")
    optim = config["optim"]
    if int(optim["epochs"]) < 1:
        raise DistillationContractError("config optim.epochs must be >= 1")
    if float(optim["lr"]) <= 0.0:
        raise DistillationContractError("config optim.lr must be > 0")
    split = config["split"]
    if str(split.get("group_by", "source_image_sha256")) != "source_image_sha256":
        raise DistillationContractError(
            "config split.group_by must be source_image_sha256; per-sample_id "
            "independent splitting is forbidden for formal training"
        )
    if not bool(split.get("require_image_sha256", True)):
        raise DistillationContractError(
            "config split.require_image_sha256 must be true"
        )
    _validate_ratios(split["ratios"])
    controls = config["controls"]
    for name in ("l2_baseline", "shuffled_prompt"):
        if name not in controls:
            raise DistillationContractError(f"config controls missing {name!r}")
    promotion = config["promotion_criteria"]
    for name in (
        "val_beats_l2",
        "val_correct_beats_shuffled",
        "primary_metrics",
        "no_arbitrary_threshold",
        "test_discipline",
    ):
        if name not in promotion:
            raise DistillationContractError(
                f"config promotion_criteria missing {name!r}"
            )
    if list(promotion["primary_metrics"]) != list(PRIMARY_PROMOTION_METRICS):
        raise DistillationContractError(
            f"config promotion_criteria.primary_metrics must be "
            f"{list(PRIMARY_PROMOTION_METRICS)}"
        )
    return config


def config_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Frozen extraction-manifest pin (governance hardening)
# ---------------------------------------------------------------------------
#
# The extraction source must be frozen in the config, not chosen by the command
# line at run time.  ``config.extraction.manifest.path`` must be a repository
# relative path and ``.sha256`` the SHA256 of that file.  The extractor refuses
# to run (fail closed) unless the ``--manifest`` it was given resolves to that
# exact path with that exact hash.


def pinned_manifest_spec(config: Mapping[str, Any]) -> dict[str, str]:
    """Return the frozen ``{path, sha256}`` extraction manifest pin."""
    extraction = config.get("extraction")
    if not isinstance(extraction, Mapping):
        raise DistillationContractError("config extraction must be a mapping")
    pin = extraction.get("manifest")
    if not isinstance(pin, Mapping):
        raise DistillationContractError(
            "config extraction.manifest pin is required; the extractor refuses "
            "to run without a frozen extraction manifest"
        )
    path = pin.get("path")
    sha = pin.get("sha256")
    if not isinstance(path, str) or not path.strip():
        raise DistillationContractError(
            "config extraction.manifest.path must be a non-empty string"
        )
    if Path(path).is_absolute():
        raise DistillationContractError(
            "config extraction.manifest.path must be relative to the repository root"
        )
    if not isinstance(sha, str) or not SOURCE_IMAGE_SHA256_HEX.match(sha):
        raise DistillationContractError(
            "config extraction.manifest.sha256 must be 64 lowercase hex digits"
        )
    return {"path": path, "sha256": sha}


def validate_pinned_manifest(
    config: Mapping[str, Any],
    *,
    manifest_path: Path | str,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Fail-closed check that ``manifest_path`` equals the config manifest pin.

    The pinned path is resolved against ``repo_root`` (defaults to the project
    root, ``parents[1]`` of this module).  The command-line path is resolved as
    given.  The result is JSON-serialisable and ``status`` is ``PASS`` only when
    both the resolved path and the SHA256 match.
    """
    pin = pinned_manifest_spec(config)
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    pinned_abs = (root / pin["path"]).resolve()
    given_abs = Path(manifest_path).resolve()
    result: dict[str, Any] = {
        "status": "PASS",
        "pinned_path": pin["path"],
        "pinned_absolute_path": str(pinned_abs),
        "pinned_sha256": pin["sha256"],
        "manifest_path": str(given_abs),
        "path_matches": pinned_abs == given_abs,
        "pinned_manifest_exists": pinned_abs.is_file(),
        "manifest_exists": given_abs.is_file(),
        "manifest_sha256": None,
        "sha256_matches": False,
    }
    if not given_abs.is_file():
        result["status"] = "FAIL"
        result["error"] = f"manifest does not exist: {given_abs}"
        return result
    actual = hashlib.sha256(given_abs.read_bytes()).hexdigest()
    result["manifest_sha256"] = actual
    result["sha256_matches"] = actual == pin["sha256"]
    if not (result["path_matches"] and result["sha256_matches"]):
        result["status"] = "FAIL"
        result["error"] = (
            "CLI manifest does not match the frozen config pin "
            f"(path_matches={result['path_matches']}, "
            f"sha256_matches={result['sha256_matches']})"
        )
    return result


def canonical_feature_index_record(
    *,
    sample_id: str,
    task: str,
    npz_name: str,
    cache: Mapping[str, Any],
    config_hash: str,
    source_image_sha256: str,
) -> dict[str, Any]:
    """Build one deterministic JSONL index record for the extractor."""
    raw = np.asarray(cache["teacher_scores"], dtype=np.float64)
    return {
        "sample_id": str(sample_id),
        "task": str(task),
        "npz": str(npz_name),
        "group_count": int(np.asarray(cache["group_mean"]).shape[0]),
        "prompt_token_count": int(np.asarray(cache["prompt_embeddings"]).shape[0]),
        "grid_h": int(cache["grid_h"]),
        "grid_w": int(cache["grid_w"]),
        "teacher_score_min": float(raw.min()),
        "teacher_score_max": float(raw.max()),
        "sample_sha256": str(cache["sample_sha256"]),
        "source_image_sha256": validate_source_image_sha256(source_image_sha256),
        "config_sha256": str(config_hash),
    }


__all__ = [
    "CACHE_ARRAY_KEYS",
    "CACHE_DTYPE",
    "CACHE_METADATA_KEYS",
    "CACHE_REQUIRED_ARRAY_KEYS",
    "CANONICAL_PROJECTION_DTYPE",
    "DEFAULT_CONFIG_FILENAME",
    "DEFAULT_LAMBDA_SEMANTIC",
    "DEFAULT_TEMPERATURE",
    "DistillationContractError",
    "FEATURE_DIM",
    "FORBIDDEN_CACHE_KEYS",
    "MERGE_UNIT",
    "PROJECTION_METADATA_KEY",
    "PROJECTION_WEIGHTS_SHA256_KEY",
    "PROJECTION_WT_KEY",
    "PROJECTION_WV_KEY",
    "PROMPT_DIM",
    "PRIMARY_PROMOTION_METRICS",
    "SCORE_DIM",
    "SHUFFLED_UNAVAILABLE_REASON",
    "SUPPORTED_BUDGETS",
    "assign_split",
    "canonical_feature_index_record",
    "canonical_projection_sha256",
    "config_sha256",
    "cyclic_shuffled_prompt_map",
    "distillation_loss",
    "hardware_real_group_metrics",
    "image_file_sha256",
    "jaccard",
    "l2_baseline_scores",
    "listwise_kl_loss",
    "load_distillation_config",
    "load_feature_cache",
    "metric_bundle",
    "minmax_normalize_1d",
    "minmax_normalize_torch",
    "ndcg_at_k",
    "ndcg_for_selected_indices",
    "pearson",
    "pinned_manifest_spec",
    "promotion_gate",
    "rank_metric_differences",
    "saliency_mass_retained",
    "save_feature_cache",
    "save_projection_npz",
    "score_mse_loss",
    "spearman",
    "split_group_summary",
    "split_sample_ids",
    "stable_rank",
    "student_cosine_scores",
    "student_cosine_scores_torch",
    "teacher_scores",
    "teacher_target",
    "teacher_topk_recall",
    "temperature_soft_target",
    "topk_indices",
    "validate_group_mean",
    "validate_pinned_manifest",
    "validate_prompt_embeddings",
    "validate_projection_matrices",
    "validate_source_image_sha256",
]
