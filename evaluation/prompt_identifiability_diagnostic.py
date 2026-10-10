#!/usr/bin/env python3
"""Prompt identifiability / degeneration diagnostic for the B1 projection (v2).

This module is **diagnosis only**.  It never trains, never tunes a
hyper-parameter, never loads a 4B model and never touches the network.  It reads
the already-extracted feature cache (``group_mean``, ``prompt_embeddings`` and
the stored teacher targets), the frozen ``Wt``/``Wv`` npz and the existing
training report, then answers one question:

    Is the failure of the ``correct prompt beats shuffled prompt`` gate a
    teacher-signal problem, a student-collapse problem, an optimisation /
    generalisation problem, or a data/schema problem?

Every score/metric/load primitive is imported from the frozen modules
(``evaluation.prompt_projection_distillation`` and
``evaluation.train_prompt_projection``); the only new math here is the
diagnostic statistic itself (Jensen-Shannon divergence, paired bootstrap /
sign-flip permutation and a variance decomposition), documented below.

Version 2 corrects the statistical design that made the v1 conclusions unsafe:

* every donor map is built **strictly inside one split**, so the val diagnostic
  never reads a test/train sample;
* cross-image donors are a fixed-seed deterministic **derangement on images**
  (every image is recipient exactly once and donor exactly once, no self-pair);
  student donors keep the task (grounding->grounding, pointing->pointing),
  teacher donors keep the task *and* the group count ``G`` (singleton ``G``
  strata are excluded and reported), and the prompt-embedding control keeps the
  cross-task condition;
* the student paired delta is clustered at the **image** level (the same-image
  grounding/pointing deltas are averaged before the bootstrap/permutation);
* the prompt-embedding paired delta is per image (``same_i - cross_i``), never
  ``same_i`` minus a global cross mean;
* "bit-identical" uses ``np.array_equal`` and is accompanied by the per-pair
  ``max_abs_diff`` max/median/p95;
* the ``0.5`` prompt/image ratio floor is labelled a pre-registered engineering
  gate / heuristic criterion, not a literature-universal threshold;
* the root cause is reported as ``primary_blocker`` / ``contributing_factor`` /
  ``training_factor`` rather than a fixed if-order that lets the teacher gate
  pre-empt the student evidence.

Thresholds (see ``THRESHOLDS``) are module-level constants fixed *before* the
test split is read.  They are expressed as normalised effects
(standardised paired effect ``dz``), 95% confidence intervals and dimensionless
signal ratios, never as ad-hoc absolute score cuts.  Only the val split decides
the gate; test is reported exactly once after the gate is fixed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluation.prompt_projection_distillation import (
    DistillationContractError,
    FEATURE_DIM,
    PROMPT_DIM,
    SCORE_DIM,
    config_sha256,
    jaccard,
    l2_baseline_scores,
    load_distillation_config,
    metric_bundle,
    pearson,
    spearman,
    split_group_summary,
    split_sample_ids,
    student_cosine_scores,
    teacher_topk_recall,
    topk_indices,
    validate_projection_matrices,
)
from evaluation.train_prompt_projection import (
    DEFAULT_CONFIG_PATH,
    index_file_sha256,
    load_index_samples,
)
from evaluation.run_hf_hardware_regular_b1 import load_projection

_EPS = 1e-12
# ---------------------------------------------------------------------------
# Frozen diagnostic thresholds (fixed BEFORE the test split is evaluated).
# ---------------------------------------------------------------------------
BOOTSTRAP_RESAMPLES = 2000
PERMUTATION_RESAMPLES = 10000
ALPHA = 0.05
CI_LEVEL = 0.95
# Conventional "small effect" floor for the standardised paired effect
# (Cohen's dz).  A signal counts as present only when the 95% CI lower bound is
# strictly positive, |dz| reaches this floor and the two-sided sign-flip
# permutation p-value is below ALPHA.
EFFECT_SIZE_FLOOR = 0.2
# Dimensionless ratio floor: prompt-induced change must be at least half of the
# image-induced change for the teacher prompt signal to count as non-weak.  This
# is a **pre-registered engineering gate / heuristic criterion**, chosen before
# any test read, NOT a literature-universal threshold.  A ratio below it is
# *evidence* that the teacher prompt signal is small, not a standalone proof
# that the teacher "failed"; it can never by itself pre-empt the student
# evidence in the root-cause attribution.
PROMPT_SIGNAL_RATIO_FLOOR = 0.5
BOOTSTRAP_SEED = 20261002
PERMUTATION_SEED = 20261003
# Fixed seed offsets for the per-split deterministic derangements.  They are
# module constants so the pairing is fully pre-registered.
DERANGEMENT_SPLIT_OFFSETS: dict[str, int] = {"train": 0, "val": 101, "test": 202}

# The v2 report is self-describing about the corrected statistical design.
REPORT_VERSION = "2.0"

THRESHOLDS: dict[str, Any] = {
    "report_version": REPORT_VERSION,
    "fixed_before_test": True,
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "permutation_resamples": PERMUTATION_RESAMPLES,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "permutation_seed": PERMUTATION_SEED,
    "derangement_split_offsets": dict(DERANGEMENT_SPLIT_OFFSETS),
    "ci_level": CI_LEVEL,
    "alpha": ALPHA,
    "effect_size_floor_cohens_dz": EFFECT_SIZE_FLOOR,
    "prompt_signal_ratio_floor": PROMPT_SIGNAL_RATIO_FLOOR,
    "prompt_signal_ratio_floor_is_engineering_gate": True,
    "prompt_signal_ratio_floor_caveat": (
        "0.5 is a pre-registered engineering gate / heuristic criterion fixed "
        "before the test split was read, not a literature-universal threshold. "
        "It alone does not establish that the teacher signal is useless."
    ),
    "independent_unit_student_paired_delta": (
        "image: the grounding and pointing deltas of one image are averaged "
        "before the bootstrap and the sign-flip permutation; the CSV stays "
        "per-sample and the JSON reports sample_count and independent_image_units"
    ),
    "donor_scope": "strictly split-local; val never reads test/train samples",
    "cross_image_donor": (
        "fixed-seed deterministic derangement on images (bijective, no self-pair); "
        "student same task, teacher same G and same task, prompt control cross-task"
    ),
    "signal_present_rule": (
        "signal is present iff paired 95% bootstrap CI lower bound > 0 AND "
        "|Cohen's dz| >= effect_size_floor AND two-sided sign-flip permutation "
        "p < alpha; otherwise the signal is treated as absent"
    ),
    "TEACHER_PROMPT_SIGNAL_WEAK": (
        "the bootstrap 95% CI upper bound of the teacher prompt/image "
        "ranking-change ratio (same-image across-prompt / cross-image same-G "
        "donor, Spearman-distance based) is below prompt_signal_ratio_floor"
    ),
    "STUDENT_PROMPT_COLLAPSE": (
        "on val, none of the primary image-clustered paired deltas (correct - "
        "same-image other task, correct - split-local cross-image shuffled) is "
        "a present signal on spearman or ndcg_full under signal_present_rule"
    ),
    "GENERALIZATION_OVERFIT": (
        "best val epoch == 0 (val never improved after initialisation) AND the "
        "train listwise KL strictly decreased over the run; the normalised "
        "train-val gap is reported alongside"
    ),
    "DATA_OR_SCHEMA_INVALID": (
        "any integrity check fails: sample/image counts, same-image group_mean "
        "bytes, teacher_soft sums, projection shapes/finiteness or index hash"
    ),
}

CSV_FIELDS: list[str] = [
    "sample_id",
    "split",
    "task",
    "other_task",
    "source_image_sha256",
    "group_count",
    "grid_h",
    "grid_w",
    "prompt_token_count",
    "teacher_same_pearson_raw",
    "teacher_same_spearman_raw",
    "teacher_same_cosine_raw",
    "teacher_same_l1_normalized",
    "teacher_same_js_soft",
    "teacher_same_overlap48",
    "teacher_same_overlap64",
    "teacher_same_overlap96",
    "teacher_same_jaccard48",
    "teacher_same_jaccard64",
    "teacher_same_jaccard96",
    "teacher_cross_pearson_raw",
    "teacher_cross_spearman_raw",
    "teacher_cross_cosine_raw",
    "teacher_cross_l1_normalized",
    "teacher_cross_js_soft",
    "teacher_cross_overlap48",
    "teacher_cross_overlap64",
    "teacher_cross_overlap96",
    "teacher_cross_jaccard48",
    "teacher_cross_jaccard64",
    "teacher_cross_jaccard96",
    "student_correct_pearson",
    "student_correct_spearman",
    "student_correct_ndcg_full",
    "student_correct_recall48",
    "student_correct_recall64",
    "student_correct_recall96",
    "student_correct_jaccard48",
    "student_correct_jaccard64",
    "student_correct_jaccard96",
    "student_correct_saliency48",
    "student_correct_saliency64",
    "student_correct_saliency96",
    "student_same_pearson",
    "student_same_spearman",
    "student_same_ndcg_full",
    "student_same_recall48",
    "student_same_recall64",
    "student_same_recall96",
    "student_same_jaccard48",
    "student_same_jaccard64",
    "student_same_jaccard96",
    "student_same_saliency48",
    "student_same_saliency64",
    "student_same_saliency96",
    "student_cross_pearson",
    "student_cross_spearman",
    "student_cross_ndcg_full",
    "student_cross_recall48",
    "student_cross_recall64",
    "student_cross_recall96",
    "student_cross_jaccard48",
    "student_cross_jaccard64",
    "student_cross_jaccard96",
    "student_cross_saliency48",
    "student_cross_saliency64",
    "student_cross_saliency96",
    "student_delta_same_pearson",
    "student_delta_same_spearman",
    "student_delta_same_ndcg_full",
    "student_delta_cross_pearson",
    "student_delta_cross_spearman",
    "student_delta_cross_ndcg_full",
    "prompt_meanpool_cosine_same",
    "prompt_meanpool_cosine_cross",
    "teacher_normalized_mean",
    "teacher_soft_max",
]


# ---------------------------------------------------------------------------
# New diagnostic primitives (documented, no second copy of the frozen metrics)
# ---------------------------------------------------------------------------


def js_divergence(p: Any, q: Any) -> float:
    """Jensen-Shannon divergence (base 2, in ``[0, 1]``) of two distributions.

    ``p`` and ``q`` must be non-negative and sum to 1.  The teacher soft target
    is exactly a probability distribution (``softmax(normalized/T)``), so JS is
    the right divergence here; no re-normalisation is invented beyond dividing
    by the sum so float32 round-off cannot make the input invalid.
    """
    a = np.asarray(p, dtype=np.float64).reshape(-1)
    b = np.asarray(q, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise DistillationContractError("js_divergence inputs must share a shape")
    if a.size == 0:
        raise DistillationContractError("js_divergence inputs must not be empty")
    if np.any(a < -1e-9) or np.any(b < -1e-9):
        raise DistillationContractError("js_divergence inputs must be non-negative")
    a = np.clip(a, 0.0, None)
    b = np.clip(b, 0.0, None)
    a = a / max(float(a.sum()), _EPS)
    b = b / max(float(b.sum()), _EPS)
    m = 0.5 * (a + b)

    def _kl(x: np.ndarray, y: np.ndarray) -> float:
        mask = x > 0.0
        return float(np.sum(x[mask] * (np.log2(x[mask]) - np.log2(np.clip(y[mask], _EPS, None)))))

    return 0.5 * _kl(a, m) + 0.5 * _kl(b, m)


def cosine_similarity(a: Any, b: Any) -> float:
    """Cosine similarity of two 1-D vectors (0.0 when either norm is 0)."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise DistillationContractError("cosine inputs must share a shape")
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx <= _EPS or ny <= _EPS:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))


def mean_pooled(prompt_embeddings: Any) -> np.ndarray:
    """Mean over prompt tokens of ``[T, D]`` raw token embeddings."""
    array = np.asarray(prompt_embeddings, dtype=np.float64)
    if array.ndim != 2:
        raise DistillationContractError("prompt_embeddings must be 2-D")
    return array.mean(axis=0)


def paired_effect(
    deltas: Sequence[float],
    *,
    seed: int,
    n_bootstrap: int = BOOTSTRAP_RESAMPLES,
    n_permutation: int = PERMUTATION_RESAMPLES,
    sample_count: int | None = None,
    independent_image_units: int | None = None,
) -> dict[str, Any]:
    """Paired 95% bootstrap CI, Cohen's dz and two-sided sign-flip p-value.

    ``deltas`` are paired per-unit differences (same unit before/after).  The
    statistic is the mean difference.  The permutation null flips each paired
    difference's sign, which is the exact paired exchangeability null.

    ``sample_count`` and ``independent_image_units`` are reported alongside ``n``
    so an image-clustered statistic cannot be mistaken for a per-sample one.
    """
    d = np.asarray([float(value) for value in deltas], dtype=np.float64)
    n = int(d.size)
    result: dict[str, Any] = {
        "n": n,
        "sample_count": int(sample_count) if sample_count is not None else n,
        "independent_image_units": (
            int(independent_image_units) if independent_image_units is not None else n
        ),
        "mean": float(d.mean()) if n else None,
        "std": float(d.std(ddof=1)) if n > 1 else 0.0,
        "ci_low": None,
        "ci_high": None,
        "cohens_dz": None,
        "p_value_two_sided": None,
        "ci_level": CI_LEVEL,
        "bootstrap_resamples": int(n_bootstrap),
        "permutation_resamples": int(n_permutation),
        "signal_present": False,
    }
    if n == 0:
        return result
    std = float(d.std(ddof=1)) if n > 1 else 0.0
    result["cohens_dz"] = float(d.mean() / std) if std > _EPS else (
        0.0 if abs(float(d.mean())) <= _EPS else math.copysign(float("inf"), float(d.mean()))
    )
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(int(n_bootstrap), n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100.0 * (1.0 - CI_LEVEL) / 2.0, 100.0 * (1.0 + CI_LEVEL) / 2.0])
    result["ci_low"] = float(lo)
    result["ci_high"] = float(hi)
    observed = float(d.mean())
    signs = rng.integers(0, 2, size=(int(n_permutation), n)).astype(np.int8) * 2 - 1
    perm = (d * signs).mean(axis=1)
    hits = int(np.sum(np.abs(perm) >= abs(observed) - 1e-15))
    result["p_value_two_sided"] = float(hits + 1) / float(int(n_permutation) + 1)
    result["signal_present"] = bool(
        result["ci_low"] is not None
        and result["ci_low"] > 0.0
        and result["cohens_dz"] is not None
        and abs(float(result["cohens_dz"])) >= EFFECT_SIZE_FLOOR
        and result["p_value_two_sided"] < ALPHA
    )
    return result


def bootstrap_ratio_ci(
    same: Sequence[float],
    cross: Sequence[float],
    *,
    seed: int,
    n_bootstrap: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Bootstrap CI of the ratio ``mean(same) / mean(cross)`` over paired units.

    Units are resampled jointly (the same index is used for ``same`` and
    ``cross``) so the paired structure is preserved.
    """
    a = np.asarray([float(v) for v in same], dtype=np.float64)
    b = np.asarray([float(v) for v in cross], dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise DistillationContractError("bootstrap ratio inputs must be equal-length, non-empty")
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, a.size, size=(int(n_bootstrap), a.size))
    ratios = a[idx].mean(axis=1) / np.clip(b[idx].mean(axis=1), _EPS, None)
    lo, hi = np.percentile(ratios, [100.0 * (1.0 - CI_LEVEL) / 2.0, 100.0 * (1.0 + CI_LEVEL) / 2.0])
    return {
        "n": int(a.size),
        "ratio_of_means": float(a.mean() / max(float(b.mean()), _EPS)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "bootstrap_resamples": int(n_bootstrap),
        "ci_level": CI_LEVEL,
    }


def variance_components(values_by_image: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, float]:
    """Hierarchical variance decomposition of aligned per-group targets.

    For image ``i`` with ``K_i`` groups and two prompts ``k in {A,B}`` and value
    ``x[i,k,g]`` the pooled decomposition is::

        mu_i   = mean over (k,g) of x[i,k,g]
        mu_ik  = mean over g of x[i,k,g]
        W      = sum_i 2*K_i
        V_image  = sum_i 2*K_i * (mu_i - mu)^2 / W
        V_prompt = sum_i sum_k K_i * (mu_ik - mu_i)^2 / W
        V_group  = sum_i sum_k sum_g (x[i,k,g] - mu_ik)^2 / W
        V_total  = V_image + V_prompt + V_group

    ``V_prompt`` is the prompt signal (same image, different prompt), ``V_image``
    the image signal (different image) and ``V_group`` the within-prompt
    across-group dispersion.  ``prompt_signal_ratio = V_prompt / V_image`` and
    ``prompt_signal_fraction = V_prompt / (V_prompt + V_image)`` are the
    recomputable dimensionless ratios.
    """
    total_weight = 0.0
    grand = 0.0
    entries: list[tuple[float, float, np.ndarray]] = []  # (K_i, weight, flat values)
    for _image, tasks in values_by_image.items():
        arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in tasks.values()]
        if not arrays:
            continue
        length = arrays[0].shape[0]
        if any(array.shape[0] != length for array in arrays):
            raise DistillationContractError("variance_components requires equal group counts")
        stacked = np.concatenate(arrays)
        weight = float(stacked.size)
        total_weight += weight
        grand += float(stacked.sum())
        entries.append((weight, 0.0, stacked))
    if total_weight <= 0.0:
        raise DistillationContractError("variance_components received no values")
    grand /= total_weight
    v_image = 0.0
    v_prompt = 0.0
    v_group = 0.0
    for weight, _unused, stacked in entries:
        mu_i = float(stacked.mean())
        v_image += weight * (mu_i - grand) ** 2
    for _image, tasks in values_by_image.items():
        arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in tasks.values()]
        if not arrays:
            continue
        stacked_all = np.concatenate(arrays)
        weight = float(stacked_all.size)
        mu_i = float(stacked_all.mean())
        for array in arrays:
            mu_ik = float(array.mean())
            v_prompt += float(array.size) * (mu_ik - mu_i) ** 2
            v_group += float(np.sum((array - mu_ik) ** 2))
    v_image /= total_weight
    v_prompt /= total_weight
    v_group /= total_weight
    v_total = v_image + v_prompt + v_group
    return {
        "total_weight": total_weight,
        "grand_mean": grand,
        "v_total": v_total,
        "v_image": v_image,
        "v_prompt": v_prompt,
        "v_group": v_group,
        "prompt_signal_ratio": v_prompt / v_image if v_image > _EPS else None,
        "prompt_signal_fraction": v_prompt / (v_prompt + v_image)
        if (v_prompt + v_image) > _EPS
        else None,
    }


# ---------------------------------------------------------------------------
# Pairing maps (fixed-seed deterministic image derangements)
# ---------------------------------------------------------------------------


def image_task_index(samples: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    by_image: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for sample in samples:
        by_image[str(sample["source_image_sha256"])][str(sample["task"])] = sample
    return dict(by_image)


def same_image_partner_map(samples: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Map every sample_id to the other task's sample_id of the same image."""
    by_image = image_task_index(samples)
    partner: dict[str, str] = {}
    for tasks in by_image.values():
        ids = [str(sample["sample_id"]) for sample in tasks.values()]
        if len(ids) == 2:
            partner[ids[0]] = ids[1]
            partner[ids[1]] = ids[0]
    return partner


def _derangement_with_rng(ordered: Sequence[str], rng: np.random.Generator) -> dict[str, str]:
    """Deterministic derangement of ``ordered`` (no fixed point, bijective).

    A uniformly random permutation is drawn repeatedly (with the supplied RNG)
    until it has no fixed point; a fixed rotation is used as a guaranteed
    fallback so the function can never fail.  Every element is therefore
    recipient exactly once and donor exactly once.
    """
    values = list(ordered)
    n = len(values)
    if n < 2:
        return {}
    for _ in range(100000):
        perm = rng.permutation(n)
        if bool(np.all(perm != np.arange(n))):
            return {values[i]: values[int(perm[i])] for i in range(n)}
    rotation = np.roll(np.arange(n), 1)  # perm[i] = i-1 (mod n): no fixed point
    return {values[i]: values[int(rotation[i])] for i in range(n)}


def derange_images(image_ids: Sequence[str], *, seed: int) -> dict[str, str]:
    """Fixed-seed deterministic derangement on a set of image hashes."""
    return _derangement_with_rng(sorted(set(str(image) for image in image_ids)), np.random.default_rng(int(seed)))


def derange_images_by_stratum(
    image_to_group: Mapping[str, int], *, seed: int
) -> tuple[dict[str, str], list[str]]:
    """Per-``G``-stratum derangement; return ``(donors, excluded_images)``.

    A stratum with fewer than two images cannot provide a same-``G`` donor, so
    every image in it is excluded and returned in ``excluded_images`` (the
    caller reports the exclusion).  Strata are processed in ascending ``G``
    order with one shared RNG, so the result is deterministic.
    """
    strata: dict[int, list[str]] = defaultdict(list)
    for image, group_count in image_to_group.items():
        strata[int(group_count)].append(str(image))
    rng = np.random.default_rng(int(seed))
    donors: dict[str, str] = {}
    excluded: list[str] = []
    for group_count in sorted(strata):
        images = sorted(strata[group_count])
        if len(images) < 2:
            excluded.extend(images)
            continue
        donors.update(_derangement_with_rng(images, rng))
    return donors, sorted(excluded)


def sample_donor_map(
    samples: Sequence[Mapping[str, Any]],
    donor_by_image: Mapping[str, str],
    *,
    same_task: bool = True,
    allow_missing: bool = False,
) -> dict[str, str]:
    """Lift an image-level derangement to a sample-level donor map.

    With ``same_task`` true the donor of ``(image i, task t)`` is the sample of
    ``(donor image, task t)``; this isolates the instance semantics from the
    prompt swap.  Samples whose image has no donor are omitted when
    ``allow_missing`` is true and otherwise raise.
    """
    index = image_task_index(samples)
    donors: dict[str, str] = {}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        image = str(sample["source_image_sha256"])
        task = str(sample["task"])
        donor_image = donor_by_image.get(image)
        if donor_image is None:
            if allow_missing:
                continue
            raise DistillationContractError(f"no donor image for {sample_id}")
        donor_task = task if same_task else ("pointing" if task == "grounding" else "grounding")
        candidate = index.get(str(donor_image), {}).get(donor_task)
        if candidate is None:
            if allow_missing:
                continue
            raise DistillationContractError(
                f"no same-task donor sample for {sample_id} (donor_image={donor_image})"
            )
        donors[sample_id] = str(candidate["sample_id"])
    return donors


def cross_image_donor_map(
    samples: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    same_grid: bool,
    allow_missing: bool = False,
) -> dict[str, str]:
    """Split-agnostic helper: deterministic derangement donor map, same task.

    Kept for the focused tests and for callers that want a sample-level donor
    dictionary.  With ``same_grid`` true the derangement is performed inside
    each group-count stratum (singleton strata are excluded when
    ``allow_missing`` is true, otherwise an error is raised).  The donor image is
    always different from the recipient image and every image is recipient
    exactly once and donor exactly once.
    """
    by_id = {str(sample["sample_id"]): sample for sample in samples}
    if same_grid:
        image_to_group: dict[str, int] = {}
        for sample in samples:
            image_to_group[str(sample["source_image_sha256"])] = int(sample["group_mean"].shape[0])
        donor_by_image, excluded = derange_images_by_stratum(image_to_group, seed=seed)
        if excluded and not allow_missing:
            raise DistillationContractError(
                f"singleton G stratum images have no same-G donor: {excluded}"
            )
    else:
        donor_by_image = derange_images(
            [str(sample["source_image_sha256"]) for sample in samples], seed=seed
        )
    return sample_donor_map(
        samples, donor_by_image, same_task=True, allow_missing=allow_missing
    )


# ---------------------------------------------------------------------------
# Target / student pair metrics
# ---------------------------------------------------------------------------


def target_pair_metrics(
    a: Mapping[str, Any], b: Mapping[str, Any], budgets: Sequence[int]
) -> dict[str, float]:
    """Similarity/distance of two teacher targets (paired, same group grid)."""
    raw_a = np.asarray(a["teacher_raw"], dtype=np.float64)
    raw_b = np.asarray(b["teacher_raw"], dtype=np.float64)
    norm_a = np.asarray(a["teacher_normalized"], dtype=np.float64)
    norm_b = np.asarray(b["teacher_normalized"], dtype=np.float64)
    soft_a = np.asarray(a["teacher_soft"], dtype=np.float64)
    soft_b = np.asarray(b["teacher_soft"], dtype=np.float64)
    if not (raw_a.shape == raw_b.shape == norm_a.shape == norm_b.shape == soft_a.shape == soft_b.shape):
        raise DistillationContractError("teacher target pairs must share a shape")
    metrics: dict[str, float] = {
        "pearson_raw": pearson(raw_a, raw_b),
        "spearman_raw": spearman(raw_a, raw_b),
        "cosine_raw": cosine_similarity(raw_a, raw_b),
        "l1_normalized": float(np.mean(np.abs(norm_a - norm_b))),
        "js_soft": js_divergence(soft_a, soft_b),
    }
    for budget in budgets:
        k = int(budget)
        top_a = topk_indices(raw_a, k)
        top_b = topk_indices(raw_b, k)
        metrics[f"overlap@{k}"] = teacher_topk_recall(top_a, top_b)
        metrics[f"jaccard@{k}"] = jaccard(top_a, top_b)
    return metrics


def _flatten_bundle(bundle: Mapping[str, Any], budgets: Sequence[int]) -> dict[str, float]:
    flat = {
        "pearson": float(bundle["pearson"]),
        "spearman": float(bundle["spearman"]),
        "ndcg_full": float(bundle["ndcg_full"]),
    }
    for budget in budgets:
        entry = bundle["per_budget"][str(int(budget))]
        k = int(budget)
        flat[f"recall{k}"] = float(entry["topk_recall"])
        flat[f"jaccard{k}"] = float(entry["jaccard"])
        flat[f"saliency{k}"] = float(entry["saliency_mass_retained"])
    return flat


def student_condition_bundle(
    sample: Mapping[str, Any],
    prompt_embeddings: np.ndarray,
    Wt: np.ndarray,
    Wv: np.ndarray,
    budgets: Sequence[int],
) -> dict[str, Any]:
    """Student score + metric bundle for one (image, prompt-condition) pair."""
    scores = student_cosine_scores(
        sample["group_mean"], prompt_embeddings, Wt, Wv
    )
    bundle = metric_bundle(
        scores,
        sample["teacher_raw"],
        sample["teacher_normalized"],
        sample["teacher_soft"],
        budgets=budgets,
    )
    return {**_flatten_bundle(bundle, budgets), "scores": scores}


def _mean_flat(rows: Sequence[Mapping[str, float]], fields: Sequence[str]) -> dict[str, float]:
    return {field: float(np.mean([float(row[field]) for row in rows])) for field in fields}


def projection_spectrum(Wt: np.ndarray, Wv: np.ndarray) -> dict[str, Any]:
    """Singular-value spectrum and effective rank of the learned projections.

    Effective rank is the participation ratio
    ``(sum s)^2 / sum s^2``; a rank collapse of ``Wt`` would show up here.
    """
    result: dict[str, Any] = {}
    for name, matrix in (("Wt", Wt), ("Wv", Wv)):
        singular = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
        energy = float(np.sum(singular**2))
        result[name] = {
            "top5_singular_values": [float(value) for value in singular[:5]],
            "min_singular_value": float(singular.min()),
            "effective_rank_participation_ratio": float(singular.sum() ** 2 / energy) if energy > 0 else 0.0,
            "numerical_rank_singular_gt_1e-6_max": int(np.sum(singular > 1e-6 * singular[0])),
            "shape": list(np.asarray(matrix).shape),
        }
    return result


def score_invariance_stats(own: Any, other: Any, *, tolerance: float = 1e-9) -> dict[str, Any]:
    """Exact vs tolerance invariance of two score vectors.

    ``exact`` uses ``np.array_equal`` (the v2 definition of "bit-identical");
    the tolerance variant is reported separately and explicitly named.  The
    element-wise ``max_abs_diff`` is returned so a reader can see how close the
    two vectors actually are.
    """
    a = np.asarray(own)
    b = np.asarray(other)
    if a.shape != b.shape:
        raise DistillationContractError("score_invariance_stats requires equal shapes")
    max_abs_diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
    return {
        "exact": bool(np.array_equal(a, b)),
        "tolerance": bool(max_abs_diff <= float(tolerance)),
        "max_abs_diff": max_abs_diff,
    }


def score_reduction_collapse(
    samples: Sequence[Mapping[str, Any]],
    Wt: np.ndarray,
    Wv: np.ndarray,
    partner: Mapping[str, str],
) -> dict[str, Any]:
    """Diagnose the ``max`` reduction over prompt tokens in the student score.

    ``student_cosine_scores`` reduces the group x token cosine matrix with a max.
    If a single projected prompt token dominates every group, that max becomes
    independent of the prompt.  This reports, per sample, how many distinct
    tokens are the argmax across groups and whether the correct and same-image
    other-task scores are exactly equal (``np.array_equal``).
    """
    unique_counts: list[int] = []
    dominant_vectors: dict[str, np.ndarray] = {}
    exact_hits: list[bool] = []
    tolerance_hits: list[bool] = []
    max_abs_diffs: list[float] = []
    by_id = {str(sample["sample_id"]): sample for sample in samples}
    for sample in samples:
        visual = np.asarray(sample["group_mean"], dtype=np.float64) @ np.asarray(Wv, dtype=np.float64).T
        prompt = np.asarray(sample["prompt_embeddings"], dtype=np.float64) @ np.asarray(Wt, dtype=np.float64).T
        visual = visual / np.clip(np.linalg.norm(visual, axis=1, keepdims=True), _EPS, None)
        prompt = prompt / np.clip(np.linalg.norm(prompt, axis=1, keepdims=True), _EPS, None)
        cosines = visual @ prompt.T
        argmax = cosines.argmax(axis=1)
        unique_counts.append(int(np.unique(argmax).size))
        mode = int(np.bincount(argmax).argmax())
        dominant_vectors[str(sample["sample_id"])] = prompt[mode]
        other_id = partner.get(str(sample["sample_id"]))
        if other_id is not None and other_id in by_id:
            own = student_cosine_scores(sample["group_mean"], sample["prompt_embeddings"], Wt, Wv)
            other = student_cosine_scores(
                sample["group_mean"], by_id[other_id]["prompt_embeddings"], Wt, Wv
            )
            stats = score_invariance_stats(own, other)
            exact_hits.append(stats["exact"])
            tolerance_hits.append(stats["tolerance"])
            max_abs_diffs.append(stats["max_abs_diff"])
    dominant_cosines: list[float] = []
    seen: set[tuple[str, str]] = set()
    for sample_id, other_id in partner.items():
        key = tuple(sorted((sample_id, other_id)))
        if key in seen or sample_id not in dominant_vectors or other_id not in dominant_vectors:
            continue
        seen.add(key)
        dominant_cosines.append(cosine_similarity(dominant_vectors[sample_id], dominant_vectors[other_id]))
    diff_array = np.asarray(max_abs_diffs, dtype=np.float64)
    return {
        "definition": (
            "argmax token per group of the projected-prompt cosine matrix; "
            "'single_argmax_token_fraction' is the fraction of samples whose "
            "groups all select one prompt token; 'exact_prompt_invariance_fraction' "
            "is the fraction of samples where the correct and same-image "
            "other-task student scores are exactly equal under np.array_equal; "
            "'tolerance_invariance_fraction_1e-9' is the explicitly named "
            "tolerance-only variant"
        ),
        "exact_invariance_uses_np_array_equal": True,
        "tolerance_used": 1e-9,
        "sample_count": len(unique_counts),
        "single_argmax_token_fraction": float(np.mean([count == 1 for count in unique_counts])),
        "mean_distinct_argmax_tokens": float(np.mean(unique_counts)),
        "max_distinct_argmax_tokens": int(np.max(unique_counts)) if unique_counts else 0,
        "dominant_prompt_direction_cosine_same_image": float(np.mean(dominant_cosines))
        if dominant_cosines
        else None,
        "dominant_prompt_direction_pairs": len(dominant_cosines),
        "exact_prompt_invariance_fraction": float(np.mean(exact_hits)) if exact_hits else None,
        "tolerance_invariance_fraction_1e-9": float(np.mean(tolerance_hits)) if tolerance_hits else None,
        "prompt_invariance_pairs": len(exact_hits),
        "pairwise_max_abs_diff": {
            "max": float(diff_array.max()) if diff_array.size else None,
            "median": float(np.median(diff_array)) if diff_array.size else None,
            "p95": float(np.percentile(diff_array, 95)) if diff_array.size else None,
        },
    }


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


def integrity_report(
    samples: Sequence[Mapping[str, Any]],
    Wt: np.ndarray,
    Wv: np.ndarray,
) -> dict[str, Any]:
    """Fail-closed data/schema checks required by the diagnostic contract."""
    failures: list[str] = []
    by_image = image_task_index(samples)
    image_count = len(by_image)
    pair_count = 0
    for image_hash, tasks in by_image.items():
        if set(tasks) != {"grounding", "pointing"}:
            failures.append(f"image {image_hash}: tasks={sorted(tasks)}")
            continue
        pair_count += 1
        a, b = tasks["grounding"], tasks["pointing"]
        gm_a = np.asarray(a["group_mean"])
        gm_b = np.asarray(b["group_mean"])
        if gm_a.shape != gm_b.shape:
            failures.append(f"image {image_hash}: group_mean shape {gm_a.shape} != {gm_b.shape}")
        elif not np.array_equal(gm_a, gm_b):
            failures.append(f"image {image_hash}: group_mean bytes differ")
        if int(a["grid_h"]) != int(b["grid_h"]) or int(a["grid_w"]) != int(b["grid_w"]):
            failures.append(f"image {image_hash}: grid mismatch")
        if np.asarray(a["prompt_embeddings"]).shape[1] != PROMPT_DIM:
            failures.append(f"image {image_hash}: grounding prompt dim")
        if np.asarray(b["prompt_embeddings"]).shape[1] != PROMPT_DIM:
            failures.append(f"image {image_hash}: pointing prompt dim")
    for sample in samples:
        soft = np.asarray(sample["teacher_soft"], dtype=np.float64)
        if not np.isfinite(soft).all():
            failures.append(f"{sample['sample_id']}: teacher soft not finite")
        elif abs(float(soft.sum()) - 1.0) > 1e-3:
            failures.append(f"{sample['sample_id']}: teacher soft sums to {soft.sum()}")
        if abs(float(sample["teacher_normalized"].min())) > 1e-6 or abs(
            float(sample["teacher_normalized"].max()) - 1.0
        ) > 1e-6:
            failures.append(f"{sample['sample_id']}: teacher normalised not in [0,1]")
    try:
        validate_projection_matrices(Wt, Wv)
    except DistillationContractError as exc:  # pragma: no cover - defensive
        failures.append(f"projection: {exc}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "sample_count": len(samples),
        "image_count": image_count,
        "image_task_pairs": pair_count,
        "same_image_group_mean_equal": not any("group_mean bytes differ" in item for item in failures),
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------


def _split_of(
    split: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name, ids in split.items():
        for sample_id in ids:
            mapping[str(sample_id)] = name
    return mapping


def analyse_split(
    name: str,
    samples: Sequence[Mapping[str, Any]],
    *,
    budgets: Sequence[int],
    Wt: np.ndarray,
    Wv: np.ndarray,
    rows: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    """All per-split diagnostics (teacher, student, prompt embedding, variance).

    Everything is built from ``samples`` only: the donor maps are split-local by
    construction, so a val run can never read a test/train prompt or target.
    """
    by_id = {str(sample["sample_id"]): sample for sample in samples}
    ordered = sorted(by_id)
    budgets = [int(b) for b in budgets]
    teacher_metric_names = list(target_pair_metrics(samples[0], samples[0], budgets))
    student_flat_names = list(student_condition_bundle(samples[0], samples[0]["prompt_embeddings"], Wt, Wv, budgets))
    student_flat_names.remove("scores")
    primary = ("spearman", "ndcg_full")
    index_by_image = image_task_index(samples)
    image_ids_ordered = sorted(index_by_image)
    image_to_group = {
        image: int(next(iter(index_by_image[image].values()))["group_mean"].shape[0])
        for image in image_ids_ordered
    }

    # -- split-local deterministic derangements -----------------------------
    partner = same_image_partner_map(samples)
    donor_by_image = derange_images(image_ids_ordered, seed=int(seed))
    student_donor = sample_donor_map(samples, donor_by_image, same_task=True)
    teacher_donor_by_image, teacher_excluded = derange_images_by_stratum(
        image_to_group, seed=int(seed) + 1
    )
    teacher_donor = sample_donor_map(
        samples, teacher_donor_by_image, same_task=True, allow_missing=True
    )

    # -- per-image teacher same-image (grounding vs pointing) and cross control
    image_teacher_same: list[dict[str, float]] = []
    image_teacher_cross: list[dict[str, float]] = []
    image_ids: list[str] = []
    teacher_excluded_set = set(teacher_excluded)
    for image_hash in image_ids_ordered:
        has_pair = set(index_by_image[image_hash]) == {"grounding", "pointing"}
        donor_image = teacher_donor_by_image.get(image_hash)
        if not has_pair or donor_image is None:
            teacher_excluded_set.add(image_hash)
            continue
        a = index_by_image[image_hash]["grounding"]
        b = index_by_image[image_hash]["pointing"]
        # cross-image control: same task, different image, same G; average the
        # two per-task metrics within the image.
        per_task_cross: list[dict[str, float]] = []
        for task in ("grounding", "pointing"):
            donor_sample = index_by_image.get(donor_image, {}).get(task)
            if donor_sample is None:
                per_task_cross = []
                break
            per_task_cross.append(
                target_pair_metrics(index_by_image[image_hash][task], donor_sample, budgets)
            )
        if not per_task_cross:
            teacher_excluded_set.add(image_hash)
            continue
        image_ids.append(image_hash)
        image_teacher_same.append(target_pair_metrics(a, b, budgets))
        image_teacher_cross.append(
            {
                metric: float(np.mean([row[metric] for row in per_task_cross]))
                for metric in teacher_metric_names
            }
        )
    teacher_excluded_sorted = sorted(teacher_excluded_set)

    teacher_summary: dict[str, Any] = {
        "image_pairs": len(image_ids),
        "images_excluded_no_same_grid_donor": len(teacher_excluded_sorted),
        "excluded_image_hashes": teacher_excluded_sorted,
        "same_image": _mean_flat(image_teacher_same, teacher_metric_names),
        "cross_image_same_grid": _mean_flat(image_teacher_cross, teacher_metric_names),
    }
    distance_defs = {
        "pearson_raw": "1 - pearson",
        "spearman_raw": "1 - spearman",
        "cosine_raw": "1 - cosine",
        "l1_normalized": "L1 (already a distance)",
        "js_soft": "JS divergence (already a distance)",
        "overlap@48": "1 - overlap@48",
        "overlap@64": "1 - overlap@64",
        "overlap@96": "1 - overlap@96",
        "jaccard@48": "1 - jaccard@48",
        "jaccard@64": "1 - jaccard@64",
        "jaccard@96": "1 - jaccard@96",
    }
    teacher_paired: dict[str, Any] = {}
    teacher_paired_ratio: dict[str, Any] = {}
    for index, metric in enumerate(teacher_metric_names):
        same_values = np.asarray([row[metric] for row in image_teacher_same], dtype=np.float64)
        cross_values = np.asarray([row[metric] for row in image_teacher_cross], dtype=np.float64)
        invert = metric in {"pearson_raw", "spearman_raw", "cosine_raw"} or metric.startswith(
            ("overlap@", "jaccard@")
        )
        if invert:
            same_distance = 1.0 - same_values
            cross_distance = 1.0 - cross_values
        else:
            same_distance = same_values
            cross_distance = cross_values
        delta = cross_distance - same_distance
        teacher_paired[metric] = {
            "distance_definition": distance_defs.get(metric, "identity"),
            "mean_same_distance": float(same_distance.mean()),
            "mean_cross_distance": float(cross_distance.mean()),
            **paired_effect(
                delta,
                seed=BOOTSTRAP_SEED + index,
                sample_count=len(ordered),
                independent_image_units=len(image_ids),
            ),
            "paired_delta_definition": "cross_image_distance - same_image_prompt_distance",
        }
        teacher_paired_ratio[metric] = {
            "definition": "mean(same_distance) / mean(cross_distance)",
            **bootstrap_ratio_ci(
                same_distance,
                cross_distance,
                seed=BOOTSTRAP_SEED + 1000 + index,
            ),
        }
    teacher_summary["paired_cross_minus_same"] = teacher_paired
    teacher_summary["prompt_image_ratio"] = teacher_paired_ratio

    # -- per-sample student and prompt-embedding diagnostics
    student_rows: dict[str, dict[str, Any]] = {}
    for sample_id in ordered:
        sample = by_id[sample_id]
        image_hash = str(sample["source_image_sha256"])
        task = str(sample["task"])
        other_id = partner.get(sample_id)
        donor_id = student_donor.get(sample_id)
        correct = student_condition_bundle(sample, sample["prompt_embeddings"], Wt, Wv, budgets)
        same = (
            student_condition_bundle(sample, by_id[other_id]["prompt_embeddings"], Wt, Wv, budgets)
            if other_id is not None
            else None
        )
        cross = (
            student_condition_bundle(
                sample, by_id[donor_id]["prompt_embeddings"], Wt, Wv, budgets
            )
            if donor_id is not None
            else None
        )
        student_rows[sample_id] = {"correct": correct, "same": same, "cross": cross}

    # -- prompt-embedding control: per-image cross-task comparison -----------
    prompt_same_by_image: dict[str, float] = {}
    prompt_cross_by_image: dict[str, float] = {}
    for image_hash in image_ids_ordered:
        tasks = index_by_image[image_hash]
        if set(tasks) != {"grounding", "pointing"}:
            continue
        g_i, p_i = tasks["grounding"], tasks["pointing"]
        prompt_same_by_image[image_hash] = cosine_similarity(
            mean_pooled(g_i["prompt_embeddings"]), mean_pooled(p_i["prompt_embeddings"])
        )
        donor_image = donor_by_image.get(image_hash)
        donor_tasks = index_by_image.get(donor_image, {}) if donor_image else {}
        if set(donor_tasks) != {"grounding", "pointing"}:
            continue
        # keep the cross-task condition: g_i vs p_j and p_i vs g_j, averaged in
        # the image; this matches the same-image g_i vs p_i comparison.
        prompt_cross_by_image[image_hash] = 0.5 * (
            cosine_similarity(
                mean_pooled(g_i["prompt_embeddings"]),
                mean_pooled(donor_tasks["pointing"]["prompt_embeddings"]),
            )
            + cosine_similarity(
                mean_pooled(p_i["prompt_embeddings"]),
                mean_pooled(donor_tasks["grounding"]["prompt_embeddings"]),
            )
        )
    paired_image_ids = [image for image in image_ids_ordered if image in prompt_cross_by_image]
    prompt_delta = np.asarray(
        [
            prompt_same_by_image[image] - prompt_cross_by_image[image]
            for image in paired_image_ids
        ],
        dtype=np.float64,
    )
    prompt_summary = {
        "prompt_token_count": {
            "min": int(min(int(by_id[sid]["prompt_embeddings"].shape[0]) for sid in ordered)),
            "max": int(max(int(by_id[sid]["prompt_embeddings"].shape[0]) for sid in ordered)),
            "mean": float(np.mean([int(by_id[sid]["prompt_embeddings"].shape[0]) for sid in ordered])),
        },
        "mean_pooled_cosine_same_image_two_tasks": float(np.mean([prompt_same_by_image[i] for i in paired_image_ids]))
        if paired_image_ids
        else None,
        "mean_pooled_cosine_cross_image_fixed_seed": float(np.mean([prompt_cross_by_image[i] for i in paired_image_ids]))
        if paired_image_ids
        else None,
        "cross_image_std": float(np.std([prompt_cross_by_image[i] for i in paired_image_ids]))
        if paired_image_ids
        else None,
        "paired_same_minus_cross": {
            "definition": (
                "per image: same_image_cosine(g_i, p_i) - cross_image_cross_task_"
                "cosine, where the cross term averages (g_i vs p_j) and (p_i vs g_j) "
                "for the deranged donor image j; never same_i minus a global mean"
            ),
            **paired_effect(
                prompt_delta,
                seed=BOOTSTRAP_SEED + 2000,
                sample_count=len(ordered),
                independent_image_units=len(paired_image_ids),
            ),
        },
        "paired_image_units": len(paired_image_ids),
        "definition": (
            "cosine of the mean over prompt tokens of the float16 raw token "
            "embeddings; same-image pair = grounding vs pointing of one image; "
            "cross-image = split-local fixed-seed deranged donor image, cross-task"
        ),
    }

    # -- image-clustered student paired deltas ------------------------------
    delta_fields = ("pearson", "spearman", "ndcg_full")
    student_paired: dict[str, Any] = {}
    for condition in ("same", "cross"):
        per_image: dict[str, dict[str, list[float]]] = {field: defaultdict(list) for field in delta_fields}
        for sample_id in ordered:
            other = student_rows[sample_id][condition]
            if other is None:
                continue
            image_hash = str(by_id[sample_id]["source_image_sha256"])
            correct = student_rows[sample_id]["correct"]
            for field in delta_fields:
                per_image[field][image_hash].append(float(correct[field]) - float(other[field]))
        student_paired[condition] = {}
        for offset, field in enumerate(delta_fields):
            image_deltas = [
                float(np.mean(values)) for values in per_image[field].values() if values
            ]
            student_paired[condition][field] = {
                "definition": (
                    f"image-clustered mean of (student_correct_{field} - student_{condition}_{field}) "
                    "over the same-image grounding/pointing samples"
                ),
                "independent_unit": "image",
                **paired_effect(
                    image_deltas,
                    seed=BOOTSTRAP_SEED + 3000 + 10 * (condition == "cross") + offset,
                    sample_count=len(ordered),
                    independent_image_units=len(image_deltas),
                ),
            }
    student_primary = {
        "correct_beats_same_image_other_task": bool(
            any(
                student_paired["same"][field]["signal_present"] for field in primary
            )
        ),
        "correct_beats_cross_image_shuffled": bool(
            any(
                student_paired["cross"][field]["signal_present"] for field in primary
            )
        ),
    }
    student_summary = {
        "count": len(ordered),
        "independent_image_units": len(image_ids_ordered),
        "correct": _mean_flat([student_rows[sid]["correct"] for sid in ordered], student_flat_names),
        "same_image_other_task": _mean_flat(
            [student_rows[sid]["same"] for sid in ordered if student_rows[sid]["same"] is not None],
            student_flat_names,
        ),
        "cross_image_shuffled": _mean_flat(
            [student_rows[sid]["cross"] for sid in ordered if student_rows[sid]["cross"] is not None],
            student_flat_names,
        ),
        "l2_baseline": {
            "pearson": float(np.mean([pearson(l2_baseline_scores(s["group_mean"]), s["teacher_raw"]) for s in samples])),
            "spearman": float(np.mean([spearman(l2_baseline_scores(s["group_mean"]), s["teacher_raw"]) for s in samples])),
        },
        "paired_deltas": student_paired,
        "primary": student_primary,
    }

    # -- variance decomposition (teacher normalised + student correct scores)
    teacher_values: dict[str, dict[str, np.ndarray]] = {}
    student_values: dict[str, dict[str, np.ndarray]] = {}
    teacher_raw_values: dict[str, dict[str, np.ndarray]] = {}
    for sample_id in ordered:
        sample = by_id[sample_id]
        image_hash = str(sample["source_image_sha256"])
        task = str(sample["task"])
        teacher_values.setdefault(image_hash, {})[task] = np.asarray(sample["teacher_normalized"], dtype=np.float64)
        teacher_raw_values.setdefault(image_hash, {})[task] = np.asarray(sample["teacher_raw"], dtype=np.float64)
        student_values.setdefault(image_hash, {})[task] = np.asarray(
            student_rows[sample_id]["correct"]["scores"], dtype=np.float64
        )
    variance = {
        "teacher_normalized": variance_components(teacher_values),
        "teacher_raw": variance_components(teacher_raw_values),
        "student_correct_raw": variance_components(student_values),
        "formula": (
            "V_image = between-image variance of the per-image mean; V_prompt = "
            "within-image across-prompt variance of the per-prompt mean; V_group "
            "= within-prompt across-group variance; prompt_signal_ratio = "
            "V_prompt / V_image"
        ),
    }

    # -- assemble CSV rows for this split
    image_same_by_hash = {image_ids[i]: image_teacher_same[i] for i in range(len(image_ids))}
    image_cross_by_hash = {image_ids[i]: image_teacher_cross[i] for i in range(len(image_ids))}
    for sample_id in ordered:
        sample = by_id[sample_id]
        image_hash = str(sample["source_image_sha256"])
        other_id = partner.get(sample_id)
        same_teacher = image_same_by_hash.get(image_hash, {})
        cross_teacher = image_cross_by_hash.get(image_hash, {})
        correct = student_rows[sample_id]["correct"]
        same = student_rows[sample_id]["same"] or {}
        cross = student_rows[sample_id]["cross"] or {}
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "split": name,
            "task": str(sample["task"]),
            "other_task": str(by_id[other_id]["task"]) if other_id else "",
            "source_image_sha256": image_hash,
            "group_count": int(sample["group_mean"].shape[0]),
            "grid_h": int(sample["grid_h"]),
            "grid_w": int(sample["grid_w"]),
            "prompt_token_count": int(sample["prompt_embeddings"].shape[0]),
            "teacher_normalized_mean": float(np.mean(sample["teacher_normalized"])),
            "teacher_soft_max": float(np.max(sample["teacher_soft"])),
        }
        for metric in teacher_metric_names:
            row[f"teacher_same_{metric.replace('@', '')}"] = float(same_teacher.get(metric, float("nan")))
            row[f"teacher_cross_{metric.replace('@', '')}"] = float(cross_teacher.get(metric, float("nan")))
        for field, value in correct.items():
            if field == "scores":
                continue
            row[f"student_correct_{field}"] = float(value)
        for field, value in same.items():
            if field == "scores":
                continue
            row[f"student_same_{field}"] = float(value)
        for field, value in cross.items():
            if field == "scores":
                continue
            row[f"student_cross_{field}"] = float(value)
            if field in {"pearson", "spearman", "ndcg_full"}:
                row[f"student_delta_cross_{field}"] = float(correct[field]) - float(value)
                if same:
                    row[f"student_delta_same_{field}"] = float(correct[field]) - float(same[field])
        row["prompt_meanpool_cosine_same"] = float(prompt_same_by_image.get(image_hash, float("nan")))
        row["prompt_meanpool_cosine_cross"] = float(prompt_cross_by_image.get(image_hash, float("nan")))
        rows.append(row)

    return {
        "count": len(ordered),
        "image_count": len(image_ids_ordered),
        "pairing": {
            "scope": "split_local",
            "seed": int(seed),
            "student_donors": student_donor,
            "teacher_donors": teacher_donor,
            "teacher_excluded_image_hashes": teacher_excluded_sorted,
            "image_derangement": donor_by_image,
            "teacher_image_derangement": teacher_donor_by_image,
            "image_to_group": image_to_group,
        },
        "reduction_collapse": score_reduction_collapse(samples, Wt, Wv, partner),
        "teacher": teacher_summary,
        "student": student_summary,
        "prompt_embedding": prompt_summary,
        "variance": variance,
    }


def generalization_report(
    training_report: Mapping[str, Any],
    split_summaries: Mapping[str, Any],
    *,
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
    train_records: int | None = None,
    train_unique_images: int | None = None,
    train_group_targets: int | None = None,
) -> dict[str, Any]:
    """Parameter/sample budget plus a normalised train-val gap.

    The parameter budget is reported against records, unique images and teacher
    group targets together.  Group targets are correlated within an image and
    within a prompt, so ``parameters_per_group_target`` is *not* a
    parameters-per-independent-sample figure and must not be read as one.
    """
    metadata = training_report.get("metadata", {})
    Wt_shape = metadata.get("Wt_shape", [SCORE_DIM, PROMPT_DIM])
    Wv_shape = metadata.get("Wv_shape", [SCORE_DIM, FEATURE_DIM])
    params = int(np.prod(Wt_shape)) + int(np.prod(Wv_shape))
    train_samples = int(training_report.get("split_counts", {}).get("train", 0))
    records = int(train_records) if train_records is not None else train_samples
    images = int(train_unique_images) if train_unique_images is not None else 0
    group_targets = int(train_group_targets) if train_group_targets is not None else 0
    train_losses = [float(value) for value in training_report.get("epoch_train_losses", [])]
    val_losses = [float(value) for value in training_report.get("epoch_val_losses", [])]
    best_epoch = int(training_report.get("best_epoch", -1))
    best_val = float(training_report.get("best_val_listwise_kl", float("nan")))
    fields = ("pearson", "spearman", "ndcg_full")
    gaps: dict[str, Any] = {}
    for field in fields:
        train_values = np.asarray(split_summaries["train"]["student"]["per_sample"][field], dtype=np.float64)
        val_values = np.asarray(split_summaries["val"]["student"]["per_sample"][field], dtype=np.float64)
        if train_values.size < 2 or val_values.size < 2:
            continue
        pooled = math.sqrt(
            (
                (train_values.size - 1) * float(train_values.var(ddof=1))
                + (val_values.size - 1) * float(val_values.var(ddof=1))
            )
            / (train_values.size + val_values.size - 2)
        )
        mean_gap = float(train_values.mean() - val_values.mean())
        gaps[field] = {
            "train_mean": float(train_values.mean()),
            "val_mean": float(val_values.mean()),
            "absolute_gap": mean_gap,
            "cohens_d_gap": mean_gap / pooled if pooled > _EPS else None,
            "normalized_ratio_gap": mean_gap / float(val_values.mean())
            if abs(float(val_values.mean())) > _EPS
            else None,
        }
    train_decreased = bool(len(train_losses) >= 2 and train_losses[-1] < train_losses[0])
    val_never_improved = bool(
        best_epoch == 0
        and len(val_losses) >= 2
        and float(np.min(val_losses[1:])) > best_val
    )
    gap_ratio = (
        (val_losses[-1] - best_val) / max(train_losses[0] - train_losses[-1], _EPS)
        if train_losses and val_losses
        else None
    )
    return {
        "trainable_parameters": params,
        "Wt_parameters": int(np.prod(Wt_shape)),
        "Wv_parameters": int(np.prod(Wv_shape)),
        "train_samples": train_samples,
        "train_records": records,
        "train_unique_images": images,
        "train_group_targets": group_targets,
        "parameters_per_train_sample": params / train_samples if train_samples else None,
        "parameters_per_record": params / records if records else None,
        "parameters_per_image": params / images if images else None,
        "parameters_per_group_target": params / group_targets if group_targets else None,
        "best_epoch": best_epoch,
        "best_val_listwise_kl": best_val,
        "train_listwise_kl_first": train_losses[0] if train_losses else None,
        "train_listwise_kl_final": train_losses[-1] if train_losses else None,
        "val_listwise_kl_final": val_losses[-1] if val_losses else None,
        "train_loss_decreased": train_decreased,
        "val_never_improved_after_init": val_never_improved,
        "kl_gap_val_final_over_train_reduction": gap_ratio,
        "per_sample_metric_gaps": gaps,
        "definition": (
            "cohens_d_gap = (train_mean - val_mean) / pooled_std across the "
            "per-sample student metrics of the two splits; parameters = "
            "Wt.size + Wv.size.  train_group_targets is the number of "
            "(record, group) teacher targets, which are correlated within an "
            "image and within a prompt and are NOT independent samples; "
            "parameters_per_group_target is reported for completeness only."
        ),
    }


def run_diagnostic(
    *,
    config_path: Path,
    index_path: Path,
    features_dir: Path | None,
    projection_path: Path,
    training_report_path: Path,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute the full diagnostic report and the per-sample CSV rows."""
    started = time.time()
    config = load_distillation_config(config_path)
    samples = load_index_samples(index_path, features_dir)
    split = split_sample_ids(
        samples,
        seed=int(config["split"]["seed"]),
        ratios=config["split"]["ratios"],
        salt=str(config["split"]["salt"]),
    )
    image_hashes = {str(s["sample_id"]): str(s["source_image_sha256"]) for s in samples}
    group_summary = split_group_summary(split, image_hashes)
    projection, projection_sha = load_projection(projection_path)
    Wt = np.asarray(projection.Wt, dtype=np.float32)
    Wv = np.asarray(projection.Wv, dtype=np.float32)
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    integrity = integrity_report(samples, Wt, Wv)

    budgets = [int(value) for value in config["budgets"]]
    by_id = {str(s["sample_id"]): s for s in samples}
    rows: list[dict[str, Any]] = []

    def _analyse(name: str) -> dict[str, Any]:
        split_samples = [by_id[sample_id] for sample_id in split[name]]
        summary = analyse_split(
            name,
            split_samples,
            budgets=budgets,
            Wt=Wt,
            Wv=Wv,
            rows=rows,
            seed=int(seed) + DERANGEMENT_SPLIT_OFFSETS[name],
        )
        summary["student"]["per_sample"] = {
            key: [
                float(row[f"student_correct_{key}"])
                for row in rows
                if row["split"] == name
            ]
            for key in ("pearson", "spearman", "ndcg_full")
        }
        return summary

    # ----- gate first: only the val split decides the pre-registered labels ----
    # Thresholds and derangement seeds are module constants fixed before this
    # point; the test split is not read until the gate below is frozen.
    split_summaries: dict[str, Any] = {"val": _analyse("val")}
    teacher_ratio_spearman = split_summaries["val"]["teacher"]["prompt_image_ratio"]["spearman_raw"]
    teacher_weak = bool(
        teacher_ratio_spearman["ci_high"] is not None
        and teacher_ratio_spearman["ci_high"] < PROMPT_SIGNAL_RATIO_FLOOR
    )
    student_val_primary = split_summaries["val"]["student"]["primary"]
    student_collapse = bool(
        not student_val_primary["correct_beats_same_image_other_task"]
        and not student_val_primary["correct_beats_cross_image_shuffled"]
    )
    gate_fixed_before_test = True

    # ----- train (for the generalisation factor), then test reported once ------
    split_summaries["train"] = _analyse("train")
    split_summaries["test"] = _analyse("test")
    split_summaries = {name: split_summaries[name] for name in ("train", "val", "test")}

    train_samples = [by_id[sample_id] for sample_id in split["train"]]
    train_unique_images = len({str(s["source_image_sha256"]) for s in train_samples})
    train_group_targets = int(sum(int(s["group_mean"].shape[0]) for s in train_samples))
    generalization = generalization_report(
        training_report,
        split_summaries,
        train_metrics=split_summaries["train"]["student"]["correct"],
        val_metrics=split_summaries["val"]["student"]["correct"],
        train_records=len(train_samples),
        train_unique_images=train_unique_images,
        train_group_targets=train_group_targets,
    )
    overfit = bool(
        generalization["best_epoch"] == 0
        and generalization["train_loss_decreased"]
        and generalization["val_never_improved_after_init"]
    )

    data_invalid = integrity["status"] != "PASS"
    diagnosis = {
        "DATA_OR_SCHEMA_INVALID": data_invalid,
        "TEACHER_PROMPT_SIGNAL_WEAK": teacher_weak,
        "STUDENT_PROMPT_COLLAPSE": student_collapse,
        "GENERALIZATION_OVERFIT": overfit,
    }
    # v2 attribution: the teacher gate is a contributing factor, never allowed
    # to pre-empt the student exact/statistical evidence as the primary blocker.
    if data_invalid:
        primary_blocker = "DATA_OR_SCHEMA_INVALID"
        contributing_factor = "NONE"
        training_factor = "NONE"
    else:
        primary_blocker = (
            "STUDENT_PROMPT_COLLAPSE" if student_collapse else "NONE_OF_THE_PRE_REGISTERED_LABELS"
        )
        contributing_factor = (
            "TEACHER_PROMPT_SIGNAL_BELOW_ENGINEERING_GATE" if teacher_weak else "NONE"
        )
        training_factor = "GENERALIZATION_OVERFIT" if overfit else "NONE"

    projection_diagnostics = {
        "spectrum": projection_spectrum(Wt, Wv),
        "score_reduction_collapse": split_summaries["val"]["reduction_collapse"],
        "score_reduction_collapse_by_split": {
            name: split_summaries[name]["reduction_collapse"] for name in ("train", "val", "test")
        },
        "score_reduction_collapse_scope": (
            "split-local; the primary entry is the val split and every per-split "
            "value is computed only from that split's samples"
        ),
    }

    provenance = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "seed": int(seed),
        "derangement_seeds": {
            name: int(seed) + DERANGEMENT_SPLIT_OFFSETS[name] for name in ("train", "val", "test")
        },
        "config_sha256": config_sha256(config_path),
        "index_sha256": index_file_sha256(index_path),
        "projection_sha256": projection_sha,
        "training_report_sha256": hashlib.sha256(training_report_path.read_bytes()).hexdigest(),
        "thresholds": THRESHOLDS,
        "elapsed_seconds": None,
    }
    report: dict[str, Any] = {
        "schema_version": "2.0",
        "report_version": REPORT_VERSION,
        "status": "COMPUTED" if integrity["status"] == "PASS" else "DATA_OR_SCHEMA_INVALID",
        "scope": (
            "prompt identifiability / degeneration diagnostic v2; diagnosis only, "
            "no training, no tuning, no model load, CPU only"
        ),
        "inputs": {
            "config": str(config_path),
            "index": str(index_path),
            "features_dir": str(features_dir) if features_dir else str(index_path.parent),
            "projection": str(projection_path),
            "training_report": str(training_report_path),
        },
        "split_counts": {name: len(value) for name, value in split.items()},
        "split_group_summary": group_summary,
        "integrity": integrity,
        "projection": projection_diagnostics,
        "provenance": provenance,
        "splits": split_summaries,
        "generalization": generalization,
        "diagnosis": diagnosis,
        "primary_blocker": primary_blocker,
        "contributing_factor": contributing_factor,
        "training_factor": training_factor,
        "gate": {
            "split": "val",
            "fixed_before_test": gate_fixed_before_test,
            "test_reported_once_after_gate": True,
            "teacher_ratio_floor_is_engineering_gate": True,
        },
        "teacher_prompt_image_ratio_val_spearman": teacher_ratio_spearman,
        "notes": {
            "test_discipline": (
                "thresholds and derangement seeds are module constants fixed before "
                "the test split was evaluated; only val decides the gate and test is "
                "reported once afterwards"
            ),
            "donor_scope": "all donor maps are strictly split-local (no train/test donor in val)",
            "engineering_gate_caveat": (
                "the 0.5 prompt/image ratio floor is a pre-registered engineering "
                "heuristic, not a literature-universal threshold; it alone does not "
                "establish that the teacher signal is useless"
            ),
            "not_task_accuracy": "all numbers are internal diagnostics (teacher-target correlation / ranking), never CV-Bench task accuracy",
        },
    }
    provenance["elapsed_seconds"] = time.time() - started
    return report, rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_markdown(report: Mapping[str, Any]) -> str:
    """Chinese Markdown report of the diagnostic facts and pre-registered rules."""
    diag = report["diagnosis"]
    gen = report["generalization"]
    lines: list[str] = []
    lines.append("# 2026-10-02 Prompt 可辨识性 / 退化诊断 v2（中文报告）")
    lines.append("")
    lines.append(
        "> 本报告为 **v2**，用于修正 v1 的统计设计（split-local donor、"
        "image derangement、image-level 聚类、per-image prompt control、"
        "`np.array_equal` 精确不变量、工程门槛措辞、根因因子拆分）。"
        "v1 结论 **SUPERSEDED_FOR_INFERENCE_BY_V2**，不得作为最终推断；v1 原始数值未改动。"
    )
    lines.append("")
    lines.append(
        "本报告只做**诊断**：不训练、不调参、不加载 4B 模型、不联网、CPU 运行；"
        "所有 score/metric/load 原语复用 `evaluation/prompt_projection_distillation.py` "
        "与 `evaluation/train_prompt_projection.py`。"
    )
    lines.append("")
    lines.append("## 1. 四个预注册标签（阈值在读取 test 之前固定）")
    lines.append("")
    lines.append("| 标签 | 判定 |")
    lines.append("| --- | --- |")
    for name in (
        "DATA_OR_SCHEMA_INVALID",
        "TEACHER_PROMPT_SIGNAL_WEAK",
        "STUDENT_PROMPT_COLLAPSE",
        "GENERALIZATION_OVERFIT",
    ):
        lines.append(f"| `{name}` | {'**TRUE**' if diag[name] else 'false'} |")
    lines.append("")
    lines.append(
        f"- `primary_blocker` = **{report['primary_blocker']}**；"
        f"`contributing_factor` = **{report['contributing_factor']}**；"
        f"`training_factor` = **{report['training_factor']}**"
    )
    lines.append(
        "- 显著性判定：paired 95% bootstrap CI 下界 > 0 且 |Cohen's dz| ≥ "
        f"{THRESHOLDS['effect_size_floor_cohens_dz']} 且双侧符号翻转 permutation p < {ALPHA}。"
    )
    lines.append(
        f"- `TEACHER_PROMPT_SIGNAL_WEAK`：教师 prompt/image 排序变化比（Spearman 距离）"
        f"bootstrap 95% CI 上界 < {PROMPT_SIGNAL_RATIO_FLOOR}。"
        "**该 0.5 为预注册工程门槛 / 启发式判据，不是文献通用阈值**；"
        "它只说明教师 prompt 信号低于该工程门槛，不能单独宣称 teacher 失效。"
    )
    lines.append(
        "- gate 仅用 **val** 判定；阈值与 derangement seed 为模块常量，在读取 test 前固定；"
        "test 只在 gate 固定后报告一次。所有 donor 严格 **split-local**。"
    )
    lines.append("")
    lines.append("## 2. 关键数字（每个 split）")
    lines.append("")
    lines.append(
        "| split | n_rec | n_img | teacher same-image rho | teacher cross-image same-G rho | prompt/image ratio | "
        "student correct rho | student same-task rho | student cross-shuffle rho | "
        "delta(same) rho(图像级) | delta(cross) rho(图像级) |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in ("train", "val", "test"):
        entry = report["splits"][name]
        ratio = entry["teacher"]["prompt_image_ratio"]["spearman_raw"]
        lines.append(
            "| {name} | {n} | {ni} | {ts:.4f} | {tc:.4f} | {ratio:.3f} | {sc:.4f} | {ss:.4f} | {sx:.4f} | {ds:+.4f} | {dx:+.4f} |".format(
                name=name,
                n=entry["count"],
                ni=entry["image_count"],
                ts=entry["teacher"]["same_image"]["spearman_raw"],
                tc=entry["teacher"]["cross_image_same_grid"]["spearman_raw"],
                ratio=ratio["ratio_of_means"],
                sc=entry["student"]["correct"]["spearman"],
                ss=entry["student"]["same_image_other_task"]["spearman"],
                sx=entry["student"]["cross_image_shuffled"]["spearman"],
                ds=entry["student"]["paired_deltas"]["same"]["spearman"]["mean"],
                dx=entry["student"]["paired_deltas"]["cross"]["spearman"]["mean"],
            )
        )
    lines.append("")
    lines.append("### 2.1 教师 target 同图两任务 vs 跨图对照（val）")
    lines.append("")
    lines.append("| metric | same-image | cross-image | ratio | ratio CI |")
    lines.append("| --- | --- | --- | --- | --- |")
    teacher_val = report["splits"]["val"]["teacher"]
    for metric, entry in teacher_val["prompt_image_ratio"].items():
        lines.append(
            f"| {metric} | {teacher_val['same_image'][metric]:.4f} | "
            f"{teacher_val['cross_image_same_grid'][metric]:.4f} | {entry['ratio_of_means']:.3f} | "
            f"[{entry['ci_low']:.3f}, {entry['ci_high']:.3f}] |"
        )
    lines.append("")
    lines.append("### 2.2 学生配对检验（val，correct 减对照；独立单位为 image）")
    lines.append("")
    lines.append("| condition | metric | mean | CI low | CI high | dz | p | present | n_img | n_rec |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for condition in ("same", "cross"):
        for metric, entry in report["splits"]["val"]["student"]["paired_deltas"][condition].items():
            lines.append(
                f"| {condition} | {metric} | {entry['mean']:+.5f} | {entry['ci_low']:+.5f} | "
                f"{entry['ci_high']:+.5f} | {entry['cohens_dz']:.3f} | {entry['p_value_two_sided']:.4f} | "
                f"{entry['signal_present']} | {entry['independent_image_units']} | {entry['sample_count']} |"
            )
    lines.append("")
    lines.append("### 2.3 方差分解（val；公式见 JSON `variance.formula`）")
    lines.append("")
    lines.append("| target | V_image | V_prompt | V_group | prompt_signal_ratio | fraction |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for key in ("teacher_normalized", "teacher_raw", "student_correct_raw"):
        entry = report["splits"]["val"]["variance"][key]
        lines.append(
            "| {k} | {vi:.3e} | {vp:.3e} | {vg:.3e} | {r} | {f} |".format(
                k=key,
                vi=entry["v_image"],
                vp=entry["v_prompt"],
                vg=entry["v_group"],
                r=("n/a" if entry["prompt_signal_ratio"] is None else f"{entry['prompt_signal_ratio']:.4f}"),
                f=("n/a" if entry["prompt_signal_fraction"] is None else f"{entry['prompt_signal_fraction']:.4f}"),
            )
        )
    lines.append("")
    lines.append("### 2.4 prompt embedding 原始统计（val）")
    lines.append("")
    prompt = report["splits"]["val"]["prompt_embedding"]
    prompt_delta = prompt["paired_same_minus_cross"]
    lines.append(
        f"- token 数：min {prompt['prompt_token_count']['min']} / mean {prompt['prompt_token_count']['mean']:.2f} / "
        f"max {prompt['prompt_token_count']['max']}"
    )
    lines.append(
        f"- mean-pooled cosine 同图两任务 = {prompt['mean_pooled_cosine_same_image_two_tasks']:.4f}；"
        f"跨图 cross-task 对照 = {prompt['mean_pooled_cosine_cross_image_fixed_seed']:.4f} "
        f"(std {prompt['cross_image_std']:.4f})；逐 image delta = same_i - cross_i 的 mean = "
        f"{prompt_delta['mean']:+.4f} "
        f"CI [{prompt_delta['ci_low']:+.4f}, {prompt_delta['ci_high']:+.4f}]，"
        f"dz={prompt_delta['cohens_dz']:.3f}，p={prompt_delta['p_value_two_sided']:.4f}；"
        f"n_img={prompt_delta['independent_image_units']}，n_rec={prompt_delta['sample_count']}"
    )
    lines.append(
        "- 说明：cross 项为同一 donor image j 的 (g_i↔p_j) 与 (p_i↔g_j) 的 image 内平均，"
        "保持与同图 g_i↔p_i 相同的 cross-task 条件；**不使用** same_i 减全局 cross 均值。"
    )
    lines.append("")
    lines.append("### 2.5 学生 max-over-token 归约坍塌与投影谱")
    lines.append("")
    collapse = report["projection"]["score_reduction_collapse"]
    lines.append(
        f"- val split（split-local）中，所有 group 由**同一个** prompt token 取得 argmax 的样本占比 = "
        f"{collapse['single_argmax_token_fraction']:.4f}（均值 distinct argmax token 数 "
        f"{collapse['mean_distinct_argmax_tokens']:.3f}，最大 {collapse['max_distinct_argmax_tokens']}）"
    )
    lines.append(
        f"- correct 与同图另一任务 prompt 的 student 分数 **`np.array_equal` 精确相等** 的样本占比 = "
        f"{collapse['exact_prompt_invariance_fraction']:.4f}（{collapse['prompt_invariance_pairs']} 对）；"
        f"容差 1e-9 版本（明确命名）= {collapse['tolerance_invariance_fraction_1e-9']:.4f}"
    )
    diff = collapse["pairwise_max_abs_diff"]
    lines.append(
        f"- 逐对 max_abs_diff：max = {diff['max']:.3e}；median = {diff['median']:.3e}；"
        f"p95 = {diff['p95']:.3e}"
    )
    lines.append(
        f"- 同图两 prompt 的 dominant projected token 方向 cosine 均值 = "
        f"{collapse['dominant_prompt_direction_cosine_same_image']:.6f}（{collapse['dominant_prompt_direction_pairs']} 对）"
    )
    for name, entry in report["projection"]["spectrum"].items():
        lines.append(
            f"- {name} effective rank = {entry['effective_rank_participation_ratio']:.3f}，"
            f"min singular = {entry['min_singular_value']:.4f}，"
            f"数值秩(s>1e-6*s0) = {entry['numerical_rank_singular_gt_1e-6_max']}"
        )
    lines.append("")
    lines.append("## 3. 参数量与泛化")
    lines.append("")
    lines.append(
        f"- trainable parameters = {gen['trainable_parameters']} "
        f"(Wt {gen['Wt_parameters']} + Wv {gen['Wv_parameters']})"
    )
    lines.append(
        f"- train record 数 = {gen['train_records']}；唯一 image 数 = {gen['train_unique_images']}；"
        f"teacher group-target 总数 = {gen['train_group_targets']}"
    )
    lines.append(
        f"- params/record = {gen['parameters_per_record']:.2f}；params/image = "
        f"{gen['parameters_per_image']:.2f}；params/group-target = "
        f"{gen['parameters_per_group_target']:.2f}"
    )
    lines.append(
        "- 注意：group target 在同一 image / 同一 prompt 内相关，**不能当作独立样本**；"
        "params/group-target 仅为完整披露，不代表相对独立样本的参数比。"
    )
    lines.append(
        f"- best_epoch = {gen['best_epoch']}，best_val_KL = {gen['best_val_listwise_kl']:.6f}；"
        f"train KL {gen['train_listwise_kl_first']:.6f} → {gen['train_listwise_kl_final']:.6f}；"
        f"val KL final {gen['val_listwise_kl_final']:.6f}"
    )
    kl_gap = gen["kl_gap_val_final_over_train_reduction"]
    lines.append(
        f"- train_loss_decreased = {gen['train_loss_decreased']}；val_never_improved_after_init = "
        f"{gen['val_never_improved_after_init']}；归一化 KL gap 比 = "
        f"{'n/a' if kl_gap is None else f'{kl_gap:.3f}'}"
    )
    for field, entry in gen["per_sample_metric_gaps"].items():
        d_value = entry["cohens_d_gap"]
        lines.append(
            f"- 逐样本 {field}: train {entry['train_mean']:.4f} vs val {entry['val_mean']:.4f}；"
            f"gap {entry['absolute_gap']:+.4f}；Cohen's d = "
            f"{'n/a' if d_value is None else f'{d_value:.3f}'}"
        )
    lines.append("")
    lines.append("## 4. 结论与解释")
    lines.append("")
    lines.append(
        "诊断口径是**教师 target 的可辨识性**与**学生是否真正利用了 prompt**，"
        "不是 CV-Bench 任务精度。v2 的因子拆分为："
        f"`primary_blocker` = **{report['primary_blocker']}**（学生 evidence 优先）；"
        f"`contributing_factor` = **{report['contributing_factor']}**（教师信号低于**工程门槛**）；"
        f"`training_factor` = **{report['training_factor']}**。"
        "教师 0.5 门槛为预注册启发式判据，不得单独据此宣称 teacher 失效。"
    )
    lines.append("")
    lines.append(
        "**归约坍塌机制（2.5）**：学生分数 `max_t cosine(...)` 在每个 group 上由单一 prompt token 主导，"
        "而同图两 prompt 的该主导投影方向几乎相同，故 correct 与同图另一任务分数 `np.array_equal` 精确相等；"
        "这说明即使 `Wt` 数值满秩，`max` 归约仍使 prompt 信息不可用，属于学生打分函数的退化。"
    )
    lines.append("")
    lines.append("## 5. 下一门槛（不替 PI/Codex 做算法决策）")
    lines.append("")
    lines.append(
        "1. 若 `DATA_OR_SCHEMA_INVALID=true`：停止一切后续，先修复对应 schema/数据不一致并重跑本诊断。"
    )
    lines.append(
        "2. 若数据有效：任何 scorer ablation / 后续算法必须以 v2 的 `primary_blocker` / "
        "`contributing_factor` / `training_factor` 与 image-level CI 作为准入条件，"
        "由 PI 决定是否接受该负结果并冻结，或在明确授权下更换监督信号/优化协议后重训"
        "（不得读 test 调参）。"
    )
    lines.append(
        "3. v1 结论 **SUPERSEDED_FOR_INFERENCE_BY_V2**：在 v2 复核前，v1 的 "
        "`TEACHER_PROMPT_SIGNAL_WEAK` 根因判断不得用于后续推断。"
    )
    lines.append("")
    lines.append("---")
    lines.append(
        f"provenance：python {report['provenance']['python']} / numpy {report['provenance']['numpy']}；"
        f"seed {report['provenance']['seed']}；derangement_seeds {report['provenance']['derangement_seeds']}；"
        f"config_sha256 `{report['provenance']['config_sha256'][:16]}…`；"
        f"index_sha256 `{report['provenance']['index_sha256'][:16]}…`；projection_sha256 "
        f"`{report['provenance']['projection_sha256'][:16]}…`"
    )
    return "\n".join(lines) + "\n"


V1_MARKER = "SUPERSEDED_FOR_INFERENCE_BY_V2"
DEFAULT_V1_MD_PATH = Path("reports/token_selector/2026-10-02-prompt-identifiability-diagnostic.md")
DEFAULT_V1_JSON_PATH = Path("reports/token_selector/2026-10-02-prompt-identifiability-diagnostic.json")


def mark_v1_superseded(md_path: Path, json_path: Path) -> dict[str, Any]:
    """Add the v2 supersession marker to the v1 artefacts, values untouched.

    The Markdown gets a prominent blockquote prepended; the JSON gets a
    top-level marker object.  Existing v1 keys/values are copied verbatim, so
    the v1 raw numbers are not modified.  Idempotent.
    """
    result: dict[str, Any] = {"marker": V1_MARKER, "markdown": False, "json": False}
    if md_path.is_file():
        text = md_path.read_text(encoding="utf-8")
        if V1_MARKER not in text:
            banner = (
                f"> **{V1_MARKER}**：v1 的统计设计有缺陷（donor 跨 split、跨图有放回抽样、"
                "样本级独立性等），结论不得用于推断；请以 v2 报告为准。"
                "以下 v1 原始数值未改动。\n\n"
            )
            md_path.write_text(banner + text, encoding="utf-8")
        result["markdown"] = True
        result["markdown_sha256"] = hashlib.sha256(md_path.read_bytes()).hexdigest()
    if json_path.is_file():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        payload.setdefault(
            V1_MARKER,
            {
                "superseded_by": "2026-10-02-prompt-identifiability-diagnostic-v2",
                "reason": (
                    "v1 built donor maps on all 472 samples and passed them into "
                    "each split, sampled cross-image donors with replacement, and "
                    "used sample-level paired statistics; v2 fixes these"
                ),
                "v1_values_unchanged": True,
            },
        )
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result["json"] = True
        result["json_sha256"] = hashlib.sha256(json_path.read_bytes()).hexdigest()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
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
    parser.add_argument(
        "--training-report",
        type=Path,
        default=Path("reports/token_selector/2026-09-28-prompt-projection-training.json"),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("reports/token_selector/2026-10-02-prompt-identifiability-diagnostic-v2.json"),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("reports/token_selector/2026-10-02-prompt-identifiability-diagnostic-v2.md"),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("reports/token_selector/2026-10-02-prompt-identifiability-samples-v2.csv"),
    )
    parser.add_argument("--v1-md", type=Path, default=DEFAULT_V1_MD_PATH)
    parser.add_argument("--v1-json", type=Path, default=DEFAULT_V1_JSON_PATH)
    parser.add_argument(
        "--no-mark-v1",
        action="store_true",
        help="skip adding the SUPERSEDED_FOR_INFERENCE_BY_V2 marker to the v1 artefacts",
    )
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args(argv)

    report, rows = run_diagnostic(
        config_path=args.config,
        index_path=args.index,
        features_dir=args.features_dir,
        projection_path=args.projection,
        training_report_path=args.training_report,
        seed=args.seed,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(report), encoding="utf-8")
    write_csv(args.csv_out, rows, CSV_FIELDS)
    split_csv_paths: dict[str, str] = {}
    for name in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == name]
        split_path = args.csv_out.with_name(args.csv_out.stem + f"-{name}" + args.csv_out.suffix)
        write_csv(split_path, split_rows, CSV_FIELDS)
        split_csv_paths[name] = str(split_path)
    if not args.no_mark_v1:
        report["provenance"]["superseded_v1"] = mark_v1_superseded(args.v1_md, args.v1_json)
    report["provenance"]["command"] = " ".join(sys.argv)
    report["provenance"]["commands"] = {
        "diagnostic": " ".join(sys.argv),
        "focused_tests": f"{sys.executable} -m unittest evaluation.tests.test_prompt_identifiability_diagnostic -v",
        "full_regression": f"{sys.executable} -m unittest discover -s evaluation/tests",
    }
    report["provenance"]["output_sha256"] = {
        "markdown": hashlib.sha256(args.md_out.read_bytes()).hexdigest(),
        "samples_csv": hashlib.sha256(args.csv_out.read_bytes()).hexdigest(),
        "samples_csv_train": hashlib.sha256(Path(split_csv_paths["train"]).read_bytes()).hexdigest(),
        "samples_csv_val": hashlib.sha256(Path(split_csv_paths["val"]).read_bytes()).hexdigest(),
        "samples_csv_test": hashlib.sha256(Path(split_csv_paths["test"]).read_bytes()).hexdigest(),
        "script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "DIAGNOSTIC "
        + json.dumps(
            {
                "status": report["status"],
                "report_version": report["report_version"],
                "primary_blocker": report["primary_blocker"],
                "contributing_factor": report["contributing_factor"],
                "training_factor": report["training_factor"],
                "diagnosis": report["diagnosis"],
                "rows": len(rows),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
