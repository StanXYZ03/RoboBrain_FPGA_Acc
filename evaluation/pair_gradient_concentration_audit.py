#!/usr/bin/env python3
"""Train-only pair-gradient **contribution concentration** audit (zero-update, CPU-only).

This module is the follow-up to
:mod:`evaluation.pair_loss_scale_audit`.  That audit showed that the frozen
``lambda_target_0p25 = 0.25 / r_med = 14.280209646136903`` rescales the
concatenated pair gradient to ``0.25`` of the base gradient at initialisation.
The open question here is purely **descriptive**:

    Is the aggregate *pair* gradient dominated by a very small number of train
    image pairs, or is it spread across the 159 train images?

To answer it the audit computes, at the three fixed initial seeds and for the
draft arm ``B_KL_PLUS_ADAPTIVE_SWAP_HINGE``:

* the per-seed **aggregate** (train-image mean) base / pair gradient for
  ``Wt``, ``Wv`` and their concatenation;
* the **per-image-pair** ``L_base_i`` / ``L_pair_i`` gradient for ``Wt``,
  ``Wv`` and concatenation, keeping only *scalars* (never the full
  ``159 x 458752`` gradient tensor set);
* the **signed projection contribution**
  ``dot(g_i, sum_j g_j) / ||sum_j g_j||^2`` which is constructed so that the
  per-seed contributions sum to exactly 1 in exact arithmetic;
* concentration summaries (top-share, effective sample size, Gini) of the
  **positive** contributions and of the pair gradient norms;
* rank/cosine/signed-contribution structure and its stability across seeds.

Strict statistics interpretation (no pass/fail threshold is invented):

* a **sum of gradient norms is not the aggregate gradient** -- top-``k``
  norm-share is *not* a vector contribution and must not be read as one;
* only the **signed projection contribution** satisfies ``sum_i c_i = 1``
  and it **may be negative** (negative mass is reported explicitly);
* multiplying the pair loss by a positive ``lambda`` changes only the
  pair-vs-base scale; it leaves the pair-only concentration, the pair-only
  ranks and the pair-only cosines unchanged;
* a non-zero / concentrated initial gradient is **not** convergence and
  **not** accuracy; no winner / promotion is declared and the DRAFT is never
  modified.

Implementation reuse (no second scorer / loss / loader):

* split / metadata / train-only loader / frozen scorer / draft pair-loss come
  from :mod:`evaluation.pair_loss_feasibility_audit`;
* the per-seed aggregate forward and the candidate-``lambda`` derivation come
  from :mod:`evaluation.pair_loss_scale_audit`;
* distribution statistics and JS come from
  :mod:`evaluation.teacher_pair_signal_audit`;
* ``spearman`` comes from :mod:`evaluation.prompt_projection_distillation`;
* deterministic initialisation comes from
  :mod:`evaluation.train_prompt_projection`.

Hard isolation contract (enforced and unit-tested): the split is computed only
from index metadata; only the 318 train feature caches are opened; a spy loader
and corrupted val/test ``npz`` files must not move any train metric, the train
aggregate SHA or the index SHA; the draft config is only read.
"""

from __future__ import annotations

import argparse
import csv as _csv
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluation.pair_loss_feasibility_audit import (
    DRAFT_RELATIVE_PATH,
    EXPECTED_DRAFT_STATUS,
    EXPECTED_TRAIN_IMAGES,
    EXPECTED_TRAIN_RECORDS,
    FORBIDDEN_SOURCE_TOKENS,
    INIT_SEEDS,
    INIT_STD,
    STUDENT_TEMPERATURE,
    _cosine,
    _require_torch,
    _training_config_split,
    draft_contract_checks,
    load_draft_config,
    load_train_samples,
    pair_loss_terms,
    prompt_query,
    visual_unit,
)
from evaluation.pair_loss_scale_audit import (
    LAMBDA_CURRENT,
    TARGET_LAMBDA_RATIOS,
    _forward_pair_losses,
    candidate_lambdas,
)
from evaluation.prompt_projection_distillation import (
    DistillationContractError,
    config_sha256,
    load_distillation_config,
    spearman,
    split_group_summary,
    split_sample_ids,
)
from evaluation.teacher_pair_signal_audit import (
    describe,
    js_divergence_nats,
    load_index_metadata,
)
from evaluation.train_prompt_projection import index_file_sha256, init_projection

__all__ = [
    "FROZEN_LAMBDA_TARGET_0P25",
    "FROZEN_R_MEDIAN",
    "PARAM_GROUPS",
    "REPORT_STEM",
    "TOP_FRACTIONS",
    "concentration_metrics",
    "effective_sample_size",
    "gini_coefficient",
    "render_markdown",
    "run_audit",
    "signed_contribution_metrics",
    "top_fraction_share",
]

# ---------------------------------------------------------------------------
# Frozen audit constants and provenance pins.
# ---------------------------------------------------------------------------

REPORT_VERSION = "1.0"
REPORT_STEM = "2026-10-03-pair-gradient-concentration-train-only"
REFERENCE_LAMBDA_NAME = "lambda_target_0p25"
FROZEN_R_MEDIAN = 0.017506745782799503
FROZEN_LAMBDA_TARGET_0P25 = 14.280209646136903
PARAM_GROUPS: tuple[str, ...] = ("Wt", "Wv", "concat")
TOP_FRACTIONS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
TOP_FRACTION_KEYS: dict[float, str] = {
    0.01: "top_1pct",
    0.05: "top_5pct",
    0.10: "top_10pct",
    0.20: "top_20pct",
}
TOP_IMAGE_COUNT = 10

SCOPE_NOTE = (
    "train-only, CPU-only, zero-update concentration audit of the DRAFT pair "
    "loss at the frozen lambda_target_0p25 = 0.25 / r_med; per-image-pair L_base "
    "/ L_pair gradients are reduced to scalars, the signed projection "
    "contribution is constructed to sum to 1 per seed, and no training, no "
    "tuning, no parameter save, no val/test feature cache read, no model load, "
    "no GPU, no network, no promotion decision and no accuracy claim is made"
)

LIMITATIONS: tuple[str, ...] = (
    "梯度范数之和不是 aggregate 梯度：top-k norm share 只是模长分布，"
    "不能当作向量贡献来读；只有 signed projection contribution 才满足和约为 1。",
    "signed projection contribution 可为负，负贡献已被单独报告；"
    "正贡献的 share/ESS/Gini 只在 positive 子集上定义，不能掩盖负质量。",
    "lambda 为正时只改变 pair 与 base 的相对尺度，不改变 pair-only 的集中度、"
    "rank 或 cosine；本审计不据此宣布任何 lambda 为 winner。",
    "不设通过/失败阈值，不宣布 promotion/no-promotion，不修改 DRAFT，也不授权重训。",
    "所有数值都是 internal train-only 初始梯度诊断，不是 CV-Bench 任务精度，"
    "不能从初始梯度推断收敛、验证集改进或任务精度。",
)

FORMULAS: dict[str, str] = {
    "aggregate_base_gradient": "grad of the train-image mean L_base w.r.t. concat(Wt, Wv)",
    "aggregate_pair_gradient": "grad of the train-image mean L_pair w.r.t. concat(Wt, Wv)",
    "r_seed": "||aggregate_pair_grad|| / ||aggregate_base_grad|| (concatenated)",
    "signed_projection_contribution": (
        "dot(g_i, sum_j g_j) / ||sum_j g_j||^2, computed from the mean gradient as "
        "dot(g_i, mean) / (N * ||mean||^2); sums to 1 in exact arithmetic"
    ),
    "pair_over_base_norm_ratio": "||g_pair_i|| / ||g_base_i||",
    "pair_norm_concentration": "top-share / ESS / Gini of the positive per-image pair gradient norms",
    "signed_contribution_concentration": "top-share / ESS / Gini of the positive signed contributions",
    "effective_sample_size": "1 / sum_i p_i^2 with p_i = v_i / sum_j v_j over positive v",
    "gini": "Gini coefficient computed on the positive values only",
    "lambda_scaling": "lambda_0p25_scaled_pair_norm = lambda * pair_grad_norm; scaled ratio = lambda * pair/base",
}


# ---------------------------------------------------------------------------
# Distribution helpers (positive-only concentration statistics).
# ---------------------------------------------------------------------------


def _positive_array(values: Sequence[float]) -> np.ndarray:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.size == 0:
        return array
    if not np.all(np.isfinite(array)):
        raise DistillationContractError("concentration inputs must be finite")
    return array[array > 0.0]


def gini_coefficient(values: Sequence[float]) -> float | None:
    """Gini coefficient of the **positive** values; ``None`` if none/zero-sum."""
    array = _positive_array(values)
    if array.size == 0:
        return None
    total = float(array.sum())
    if total <= 0.0:
        return None
    ordered = np.sort(array)
    n = int(ordered.size)
    index = np.arange(1, n + 1, dtype=np.float64)
    gini = (2.0 * float(np.sum(index * ordered)) - (n + 1) * total) / (n * total)
    return float(min(max(gini, 0.0), 1.0))


def effective_sample_size(values: Sequence[float]) -> float | None:
    """``1 / sum_i p_i^2`` over the positive values renormalised to sum to 1."""
    array = _positive_array(values)
    if array.size == 0:
        return None
    total = float(array.sum())
    if total <= 0.0:
        return None
    normalized = array / total
    return float(1.0 / float(np.sum(normalized * normalized)))


def top_fraction_share(values: Sequence[float], fraction: float) -> float | None:
    """Share of the positive total held by the top ``ceil(fraction * n)`` values."""
    if not 0.0 < float(fraction) <= 1.0:
        raise DistillationContractError("fraction must be in (0, 1]")
    array = _positive_array(values)
    if array.size == 0:
        return None
    total = float(array.sum())
    if total <= 0.0:
        return None
    ordered = np.sort(array)[::-1]
    k = max(1, int(math.ceil(float(fraction) * ordered.size)))
    return float(float(ordered[:k].sum()) / total)


def concentration_metrics(values: Sequence[float]) -> dict[str, Any]:
    """Top-share / ESS / Gini for a non-negative concentration profile."""
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    finite = array[np.isfinite(array)] if array.size else array
    positive = finite[finite > 0.0] if finite.size else finite
    result: dict[str, Any] = {
        "count": int(finite.size),
        "positive_count": int(positive.size),
        "zero_count": int((finite == 0.0).sum()) if finite.size else 0,
        "sum": float(finite.sum()) if finite.size else None,
        "positive_sum": float(positive.sum()) if positive.size else None,
        "effective_sample_size": effective_sample_size(values),
        "gini": gini_coefficient(values),
    }
    for fraction in TOP_FRACTIONS:
        result[TOP_FRACTION_KEYS[fraction]] = top_fraction_share(values, fraction)
    return result


def signed_contribution_metrics(
    values: Sequence[float], *, expected_sum: float = 1.0
) -> dict[str, Any]:
    """Summary of the signed contributions: sum check, negative mass, positive shape."""
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.size and not np.all(np.isfinite(array)):
        raise DistillationContractError("signed contributions must be finite")
    total = float(array.sum()) if array.size else 0.0
    negative = array[array < 0.0] if array.size else array
    positive = array[array > 0.0] if array.size else array
    zero_count = int((array == 0.0).sum()) if array.size else 0
    result: dict[str, Any] = {
        "count": int(array.size),
        "sum": total,
        "expected_sum": float(expected_sum),
        "sum_error_abs": abs(total - float(expected_sum)),
        "negative_count": int(negative.size),
        "negative_fraction": (float(negative.size) / float(array.size)) if array.size else None,
        "negative_mass_signed": float(negative.sum()) if negative.size else 0.0,
        "negative_mass_abs": float(-negative.sum()) if negative.size else 0.0,
        "positive_count": int(positive.size),
        "positive_sum": float(positive.sum()) if positive.size else None,
        "zero_count": zero_count,
    }
    for fraction in TOP_FRACTIONS:
        result[TOP_FRACTION_KEYS[fraction]] = top_fraction_share(positive, fraction)
    result["positive_effective_sample_size"] = effective_sample_size(positive)
    result["positive_gini"] = gini_coefficient(positive)
    return result


def _describe_present(values: Sequence[Any]) -> dict[str, Any]:
    """``describe`` ignoring ``None`` entries (undefined cosines/ratios)."""
    return describe([float(value) for value in values if value is not None])


def _cross_seed_dict(dicts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recursively aggregate numeric leaves of identically-shaped dicts."""
    present = [entry for entry in dicts if entry is not None]
    if not present:
        return {}
    result: dict[str, Any] = {}
    for key in present[0]:
        values = [entry.get(key) for entry in present]
        if all(isinstance(value, Mapping) for value in values if value is not None):
            result[key] = _cross_seed_dict([value for value in values if value is not None])
            continue
        numbers = [
            float(value)
            for value in values
            if value is not None and not isinstance(value, bool)
        ]
        if numbers:
            result[key] = {
                "count": len(numbers),
                "mean": float(np.mean(numbers)),
                "median": float(np.median(numbers)),
                "min": float(np.min(numbers)),
                "max": float(np.max(numbers)),
            }
        else:
            result[key] = None
    return result


# ---------------------------------------------------------------------------
# Per-seed concentration audit (zero-update).
# ---------------------------------------------------------------------------


def _image_gradients(
    grounding: Mapping[str, Any],
    pointing: Mapping[str, Any],
    js_gp: float,
    Wt: Any,
    Wv: Any,
) -> tuple[dict[str, Any], dict[str, tuple[Any, Any, Any]]]:
    """Forward one image pair and return scalar diagnostics + per-param gradients."""
    torch = _require_torch()
    gm_g = torch.as_tensor(grounding["group_mean"], dtype=torch.float32)
    gm_p = torch.as_tensor(pointing["group_mean"], dtype=torch.float32)
    pe_g = torch.as_tensor(grounding["prompt_embeddings"], dtype=torch.float32)
    pe_p = torch.as_tensor(pointing["prompt_embeddings"], dtype=torch.float32)
    y_g = torch.as_tensor(grounding["teacher_soft"], dtype=torch.float32)
    y_p = torch.as_tensor(pointing["teacher_soft"], dtype=torch.float32)
    q_g = prompt_query(pe_g, Wt)
    q_p = prompt_query(pe_p, Wt)
    scores_g = visual_unit(gm_g, Wv) @ q_g
    scores_p = visual_unit(gm_p, Wv) @ q_p
    l_base, l_pair, d_swap, l_total, margin = pair_loss_terms(
        scores_g, scores_p, y_g, y_p, js_gp
    )
    base_grads = torch.autograd.grad(l_base, [Wt, Wv], retain_graph=True)
    pair_grads = torch.autograd.grad(l_pair, [Wt, Wv], retain_graph=True)
    scalars = {
        "l_base": float(l_base.item()),
        "l_pair": float(l_pair.item()),
        "d_swap": float(d_swap.item()),
        "l_total": float(l_total.item()),
        "hinge_margin": float(margin.item()),
        "hinge_active": bool(margin.item() > 0.0),
    }
    flat = lambda tensor: tensor.detach().reshape(-1)  # noqa: E731 - local alias
    groups = {
        "Wt": (flat(base_grads[0]), flat(pair_grads[0])),
        "Wv": (flat(base_grads[1]), flat(pair_grads[1])),
        "concat": (
            torch.cat([flat(base_grads[0]), flat(base_grads[1])]),
            torch.cat([flat(pair_grads[0]), flat(pair_grads[1])]),
        ),
    }
    return scalars, groups


def _gradient_mass_stats(tensor: Any) -> dict[str, Any]:
    flat = tensor.detach().reshape(-1)
    numel = int(flat.numel())
    zero = int((flat == 0.0).sum().item())
    nonfinite = int((~_require_torch().isfinite(flat)).sum().item())
    return {
        "numel": numel,
        "zero_count": zero,
        "nonfinite_count": nonfinite,
        "zero_fraction": (zero / numel) if numel else None,
        "nonfinite_fraction": (nonfinite / numel) if numel else None,
    }


def _seed_concentration(
    *,
    seed: int,
    image_pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], float]],
    lambda_value: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Zero-update per-seed aggregate + per-image concentration audit."""
    torch = _require_torch()
    torch.set_num_threads(1)
    Wt, Wv, loss_base, loss_pair = _forward_pair_losses(int(seed), image_pairs)
    agg_base_grads = torch.autograd.grad(loss_base, [Wt, Wv], retain_graph=True)
    agg_pair_grads = torch.autograd.grad(loss_pair, [Wt, Wv], retain_graph=False)

    flat = lambda tensor: tensor.detach().reshape(-1)  # noqa: E731 - local alias
    aggregate_groups = {
        "Wt": (flat(agg_base_grads[0]), flat(agg_pair_grads[0])),
        "Wv": (flat(agg_base_grads[1]), flat(agg_pair_grads[1])),
        "concat": (
            torch.cat([flat(agg_base_grads[0]), flat(agg_base_grads[1])]),
            torch.cat([flat(agg_pair_grads[0]), flat(agg_pair_grads[1])]),
        ),
    }
    aggregate: dict[str, Any] = {}
    for group, (base_vec, pair_vec) in aggregate_groups.items():
        base_norm = float(base_vec.norm())
        pair_norm = float(pair_vec.norm())
        mass_base = _gradient_mass_stats(base_vec)
        mass_pair = _gradient_mass_stats(pair_vec)
        aggregate[group] = {
            "base_grad_norm": base_norm,
            "pair_grad_norm": pair_norm,
            "pair_over_base_norm_ratio": (pair_norm / base_norm) if base_norm > 0.0 else None,
            "base_zero_fraction": mass_base["zero_fraction"],
            "base_nonfinite_fraction": mass_base["nonfinite_fraction"],
            "pair_zero_fraction": mass_pair["zero_fraction"],
            "pair_nonfinite_fraction": mass_pair["nonfinite_fraction"],
        }

    n_images = len(image_pairs)
    pair_denominator = {
        group: float(n_images) * float(vecs[1] @ vecs[1])
        for group, vecs in aggregate_groups.items()
    }
    base_denominator = {
        group: float(n_images) * float(vecs[0] @ vecs[0])
        for group, vecs in aggregate_groups.items()
    }

    values: dict[str, dict[str, list[Any]]] = {
        group: defaultdict(list) for group in PARAM_GROUPS
    }
    rows: list[dict[str, Any]] = []
    for grounding, pointing, js_gp in image_pairs:
        scalars, groups = _image_gradients(grounding, pointing, js_gp, Wt, Wv)
        image_hash = str(grounding["source_image_sha256"])
        row: dict[str, Any] = {
            "seed": int(seed),
            "image_sha256": image_hash,
            "grounding_sample_id": str(grounding["sample_id"]),
            "pointing_sample_id": str(pointing["sample_id"]),
            "group_count": int(np.asarray(grounding["teacher_soft"]).size),
            "js_nats": float(js_gp),
            "l_base": scalars["l_base"],
            "l_pair": scalars["l_pair"],
            "d_swap": scalars["d_swap"],
            "l_total": scalars["l_total"],
            "hinge_margin": scalars["hinge_margin"],
            "hinge_active": scalars["hinge_active"],
            "lambda_0p25": float(lambda_value),
        }
        for group in PARAM_GROUPS:
            base_vec, pair_vec = groups[group]
            agg_base_vec, agg_pair_vec = aggregate_groups[group]
            base_norm = float(base_vec.norm())
            pair_norm = float(pair_vec.norm())
            pair_over_base = (pair_norm / base_norm) if base_norm > 0.0 else None
            cos_pair = _cosine(pair_vec, agg_pair_vec)
            cos_base = _cosine(base_vec, agg_base_vec)
            pair_contribution = (
                float(pair_vec @ agg_pair_vec) / pair_denominator[group]
                if pair_denominator[group] > 0.0
                else None
            )
            base_contribution = (
                float(base_vec @ agg_base_vec) / base_denominator[group]
                if base_denominator[group] > 0.0
                else None
            )
            mass_pair = _gradient_mass_stats(pair_vec)
            mass_base = _gradient_mass_stats(base_vec)
            bucket = values[group]
            bucket["pair_grad_norm"].append(pair_norm)
            bucket["base_grad_norm"].append(base_norm)
            bucket["pair_over_base_norm_ratio"].append(pair_over_base)
            bucket["cos_pair_aggregate"].append(cos_pair)
            bucket["cos_base_aggregate"].append(cos_base)
            bucket["pair_signed_contribution"].append(pair_contribution)
            bucket["base_signed_contribution"].append(base_contribution)
            bucket["zero_fraction"].append(mass_pair["zero_fraction"])
            bucket["nonfinite_fraction"].append(mass_pair["nonfinite_fraction"])
            bucket["base_zero_fraction"].append(mass_base["zero_fraction"])
            bucket["base_nonfinite_fraction"].append(mass_base["nonfinite_fraction"])
            bucket["lambda_0p25_scaled_pair_norm"].append(float(lambda_value) * pair_norm)
            bucket["scaled_pair_over_base_norm_ratio"].append(
                (float(lambda_value) * pair_over_base)
                if pair_over_base is not None
                else None
            )
            row.update(
                {
                    f"{group}_base_grad_norm": base_norm,
                    f"{group}_pair_grad_norm": pair_norm,
                    f"{group}_pair_over_base_norm_ratio": pair_over_base,
                    f"{group}_cos_pair_aggregate": cos_pair,
                    f"{group}_cos_base_aggregate": cos_base,
                    f"{group}_pair_signed_contribution": pair_contribution,
                    f"{group}_base_signed_contribution": base_contribution,
                    f"{group}_lambda_0p25_scaled_pair_norm": float(lambda_value) * pair_norm,
                    f"{group}_scaled_pair_over_base_norm_ratio": (
                        (float(lambda_value) * pair_over_base)
                        if pair_over_base is not None
                        else None
                    ),
                    f"{group}_pair_zero_fraction": mass_pair["zero_fraction"],
                    f"{group}_pair_nonfinite_fraction": mass_pair["nonfinite_fraction"],
                }
            )
        rows.append(row)

    per_param: dict[str, Any] = {}
    for group in PARAM_GROUPS:
        bucket = values[group]
        per_param[group] = {
            "base_grad_norm": describe(bucket["base_grad_norm"]),
            "pair_grad_norm": describe(bucket["pair_grad_norm"]),
            "pair_over_base_norm_ratio": _describe_present(bucket["pair_over_base_norm_ratio"]),
            "cos_pair_aggregate": _describe_present(bucket["cos_pair_aggregate"]),
            "cos_base_aggregate": _describe_present(bucket["cos_base_aggregate"]),
            "pair_zero_fraction": describe(bucket["zero_fraction"]),
            "pair_nonfinite_fraction": describe(bucket["nonfinite_fraction"]),
            "base_zero_fraction": describe(bucket["base_zero_fraction"]),
            "base_nonfinite_fraction": describe(bucket["base_nonfinite_fraction"]),
            "pair_grad_norm_concentration": concentration_metrics(bucket["pair_grad_norm"]),
            "base_grad_norm_concentration": concentration_metrics(bucket["base_grad_norm"]),
            "pair_signed_contribution": signed_contribution_metrics(
                bucket["pair_signed_contribution"]
            ),
            "base_signed_contribution": signed_contribution_metrics(
                bucket["base_signed_contribution"]
            ),
            "lambda_0p25_scaled_pair_norm": describe(
                bucket["lambda_0p25_scaled_pair_norm"]
            ),
            "scaled_pair_over_base_norm_ratio": _describe_present(
                bucket["scaled_pair_over_base_norm_ratio"]
            ),
        }

    # Rank/correlation structure across images for this seed.
    correlations = _seed_correlations(rows)
    top_images = _top_images(rows)

    seed_block: dict[str, Any] = {
        "seed": int(seed),
        "image_count": n_images,
        "lambda_name": REFERENCE_LAMBDA_NAME,
        "lambda_value": float(lambda_value),
        "r_seed_concat_pair_over_base": aggregate["concat"]["pair_over_base_norm_ratio"],
        "loss_means": {
            "l_base": float(loss_base.item()),
            "l_pair": float(loss_pair.item()),
            "l_total": float(loss_base.item()) + float(loss_pair.item()),
        },
        "aggregate": aggregate,
        "per_param": per_param,
        "correlations": correlations,
        "top_images": top_images,
        "r_seed_pair_over_base_by_param": {
            group: aggregate[group]["pair_over_base_norm_ratio"] for group in PARAM_GROUPS
        },
    }
    return seed_block, rows


def _pair_by_image(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["image_sha256"]): row for row in rows}


def _spearman_safe(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    try:
        return float(spearman(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)))
    except DistillationContractError:
        return None


def _seed_correlations(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    js = [float(row["js_nats"]) for row in rows]
    group_count = [float(row["group_count"]) for row in rows]
    hinge_margin = [float(row["hinge_margin"]) for row in rows]
    active = [index for index, row in enumerate(rows) if row["hinge_active"]]
    correlations: dict[str, Any] = {}
    for group in PARAM_GROUPS:
        pair_norm = [float(row[f"{group}_pair_grad_norm"]) for row in rows]
        pair_contribution = [
            float(row[f"{group}_pair_signed_contribution"]) for row in rows
        ]
        l_pair = [float(row["l_pair"]) for row in rows]
        correlations[group] = {
            "js_vs_pair_norm": _spearman_safe(js, pair_norm),
            "js_vs_pair_signed_contribution": _spearman_safe(js, pair_contribution),
            "group_count_vs_pair_norm": _spearman_safe(group_count, pair_norm),
            "l_pair_vs_pair_norm": _spearman_safe(l_pair, pair_norm),
            "hinge_margin_vs_pair_norm": _spearman_safe(hinge_margin, pair_norm),
            "hinge_margin_vs_pair_norm_hinge_active_only": _spearman_safe(
                [hinge_margin[index] for index in active],
                [pair_norm[index] for index in active],
            ),
            "hinge_active_count": len(active),
        }
    return correlations


def _top_images(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def pack(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "image_sha256": str(row["image_sha256"]),
            "group_count": int(row["group_count"]),
            "js_nats": float(row["js_nats"]),
            "l_base": float(row["l_base"]),
            "l_pair": float(row["l_pair"]),
            "hinge_margin": float(row["hinge_margin"]),
            "hinge_active": bool(row["hinge_active"]),
            "pair_grad_norm_concat": float(row["concat_pair_grad_norm"]),
            "pair_signed_contribution_concat": float(
                row["concat_pair_signed_contribution"]
            ),
            "base_signed_contribution_concat": float(
                row["concat_base_signed_contribution"]
            ),
        }

    by_norm = sorted(
        rows, key=lambda row: float(row["concat_pair_grad_norm"]), reverse=True
    )[:TOP_IMAGE_COUNT]
    by_contribution = sorted(
        rows,
        key=lambda row: float(row["concat_pair_signed_contribution"]),
        reverse=True,
    )[:TOP_IMAGE_COUNT]
    return {
        "by_pair_norm": [pack(row) for row in by_norm],
        "by_pair_signed_contribution": [pack(row) for row in by_contribution],
    }


# ---------------------------------------------------------------------------
# Cross-seed aggregation.
# ---------------------------------------------------------------------------


def _pooled_per_param(
    rows: Sequence[Mapping[str, Any]], expected_sum: float
) -> dict[str, Any]:
    pooled: dict[str, Any] = {}
    for group in PARAM_GROUPS:
        pair_norm = [float(row[f"{group}_pair_grad_norm"]) for row in rows]
        base_norm = [float(row[f"{group}_base_grad_norm"]) for row in rows]
        pair_contribution = [
            float(row[f"{group}_pair_signed_contribution"]) for row in rows
        ]
        base_contribution = [
            float(row[f"{group}_base_signed_contribution"]) for row in rows
        ]
        pooled[group] = {
            "pair_grad_norm": describe(pair_norm),
            "pair_grad_norm_concentration": concentration_metrics(pair_norm),
            "base_grad_norm_concentration": concentration_metrics(base_norm),
            "pair_signed_contribution": signed_contribution_metrics(
                pair_contribution, expected_sum=expected_sum
            ),
            "base_signed_contribution": signed_contribution_metrics(
                base_contribution, expected_sum=expected_sum
            ),
        }
    return pooled


def _pooled_correlations(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    js = [float(row["js_nats"]) for row in rows]
    group_count = [float(row["group_count"]) for row in rows]
    hinge_margin = [float(row["hinge_margin"]) for row in rows]
    active = [index for index, row in enumerate(rows) if row["hinge_active"]]
    result: dict[str, Any] = {}
    for group in PARAM_GROUPS:
        pair_norm = [float(row[f"{group}_pair_grad_norm"]) for row in rows]
        pair_contribution = [
            float(row[f"{group}_pair_signed_contribution"]) for row in rows
        ]
        l_pair = [float(row["l_pair"]) for row in rows]
        result[group] = {
            "js_vs_pair_norm": _spearman_safe(js, pair_norm),
            "js_vs_pair_signed_contribution": _spearman_safe(js, pair_contribution),
            "group_count_vs_pair_norm": _spearman_safe(group_count, pair_norm),
            "l_pair_vs_pair_norm": _spearman_safe(l_pair, pair_norm),
            "hinge_margin_vs_pair_norm": _spearman_safe(hinge_margin, pair_norm),
            "hinge_margin_vs_pair_norm_hinge_active_only": _spearman_safe(
                [hinge_margin[index] for index in active],
                [pair_norm[index] for index in active],
            ),
            "hinge_active_count": len(active),
        }
    return result


def _cross_seed_rank(rows: Sequence[Mapping[str, Any]], group: str) -> dict[str, Any]:
    by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_seed[int(row["seed"])][str(row["image_sha256"])] = float(
            row[f"{group}_pair_grad_norm"]
        )
    seeds = sorted(by_seed)
    pairwise: dict[str, Any] = {}
    values: list[float] = []
    for first_index in range(len(seeds)):
        for second_index in range(first_index + 1, len(seeds)):
            first, second = seeds[first_index], seeds[second_index]
            shared = sorted(set(by_seed[first]) & set(by_seed[second]))
            if len(shared) < 2:
                continue
            rho = _spearman_safe(
                [by_seed[first][key] for key in shared],
                [by_seed[second][key] for key in shared],
            )
            pairwise[f"{first}_vs_{second}"] = rho
            if rho is not None:
                values.append(rho)
    return {
        "pairwise_spearman": pairwise,
        "shared_image_count": len(set.intersection(*[set(v) for v in by_seed.values()]))
        if by_seed
        else 0,
        "mean": float(np.mean(values)) if values else None,
        "min": float(np.min(values)) if values else None,
        "max": float(np.max(values)) if values else None,
    }


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------


def run_audit(
    *,
    config_path: Path,
    draft_path: Path,
    index_path: Path,
    features_dir: Path | None = None,
    seeds: Sequence[int] = INIT_SEEDS,
    cache_loader: Callable[..., dict[str, Any]] | None = None,
    lambda_value: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Full train-only concentration report plus per-image-per-param rows."""
    config = load_distillation_config(config_path)
    draft, draft_sha = load_draft_config(draft_path)
    contract = draft_contract_checks(draft)
    if not all(contract.values()):
        failed = sorted(key for key, value in contract.items() if not value)
        raise DistillationContractError(f"draft contract checks failed: {failed}")

    split_seed, ratios, salt = _training_config_split(config)
    metadata = load_index_metadata(index_path)
    split = split_sample_ids(metadata, seed=split_seed, ratios=ratios, salt=salt)
    image_hashes = {record["sample_id"]: record["source_image_sha256"] for record in metadata}
    group_summary = split_group_summary(split, image_hashes)

    opened: set[str] = set()
    train_samples, input_aggregate_sha, resolved_paths = load_train_samples(
        index_path,
        features_dir,
        split["train"],
        opened=opened,
        cache_loader=cache_loader,
    )
    directory = Path(features_dir) if features_dir else Path(index_path).parent
    val_test_ids = set(split["val"]) | set(split["test"])

    def _resolve(record: Mapping[str, Any]) -> str:
        path = Path(str(record["npz"]))
        if not path.is_absolute():
            path = directory / path
        return str(path)

    by_id = {record["sample_id"]: record for record in metadata}
    val_test_paths = {_resolve(by_id[sample_id]) for sample_id in val_test_ids}
    isolation = {
        "train_only": True,
        "split_from_index_metadata_only": True,
        "val_test_npz_opened": sorted(opened & val_test_paths),
        "opened_npz_count": len(opened),
        "train_npz_count": len(resolved_paths),
        "opened_only_train_npz": not (opened & val_test_paths),
        "val_test_samples_ignored": len(val_test_ids),
    }
    if isolation["val_test_npz_opened"]:
        raise DistillationContractError(
            f"isolation violation: val/test caches opened: {isolation['val_test_npz_opened']}"
        )

    by_image: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for sample in train_samples:
        by_image[sample["source_image_sha256"]][sample["task"]] = sample
    pair_failures: list[str] = []
    image_pairs: list[tuple[Mapping[str, Any], Mapping[str, Any], float]] = []
    for image_hash in sorted(by_image):
        tasks = by_image[image_hash]
        if set(tasks) != {"grounding", "pointing"}:
            pair_failures.append(f"image {image_hash}: tasks={sorted(tasks)}")
            continue
        js_gp = js_divergence_nats(
            tasks["grounding"]["teacher_soft"], tasks["pointing"]["teacher_soft"]
        )
        image_pairs.append((tasks["grounding"], tasks["pointing"], float(js_gp)))

    # The reference lambda is the frozen 0.25 / r_med scale; recompute it from the
    # same three-seed concat ratio and cross-check the frozen constant.  An
    # explicit ``lambda_value`` override is only used by focused tests.
    per_seed_blocks: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    computed_r_medians: list[float] = []
    if lambda_value is None:
        probe = [
            _seed_concentration(
                seed=int(seed),
                image_pairs=image_pairs,
                lambda_value=float(FROZEN_LAMBDA_TARGET_0P25),
            )
            for seed in seeds
        ]
        for block, _rows in probe:
            ratio = block["r_seed_concat_pair_over_base"]
            if ratio is not None:
                computed_r_medians.append(float(ratio))
        if len(computed_r_medians) != len(seeds):
            raise DistillationContractError(
                "the initial concatenated pair/base ratio must be defined for every seed"
            )
        r_median = float(np.median(np.asarray(computed_r_medians, dtype=np.float64)))
        resolved_lambda = float(candidate_lambdas(r_median)[REFERENCE_LAMBDA_NAME])
        if math.isclose(
            resolved_lambda, FROZEN_LAMBDA_TARGET_0P25, rel_tol=1e-12, abs_tol=1e-15
        ):
            # The frozen scale is reproduced; the probe already used it.
            per_seed_blocks = [block for block, _rows in probe]
            per_image_rows = [row for _block, rows in probe for row in rows]
        else:  # pragma: no cover - defensive; the frozen r_med is reproduced
            for seed in seeds:
                block, rows = _seed_concentration(
                    seed=int(seed),
                    image_pairs=image_pairs,
                    lambda_value=resolved_lambda,
                )
                per_seed_blocks.append(block)
                per_image_rows.extend(rows)
    else:
        resolved_lambda = float(lambda_value)
        r_median = None
        for seed in seeds:
            block, rows = _seed_concentration(
                seed=int(seed),
                image_pairs=image_pairs,
                lambda_value=resolved_lambda,
            )
            per_seed_blocks.append(block)
            per_image_rows.extend(rows)

    cross_seed_per_param = {
        group: _cross_seed_dict(
            [block["per_param"][group] for block in per_seed_blocks]
        )
        for group in PARAM_GROUPS
    }
    pooled = _pooled_per_param(per_image_rows, expected_sum=float(len(seeds)))
    cross_seed_rank = {
        group: _cross_seed_rank(per_image_rows, group) for group in PARAM_GROUPS
    }
    signed_sum_errors = [
        float(block["per_param"][group]["pair_signed_contribution"]["sum_error_abs"])
        for block in per_seed_blocks
        for group in PARAM_GROUPS
    ]
    signed_sum_error_max = max(signed_sum_errors) if signed_sum_errors else None

    lambda_matches_frozen = (
        math.isclose(resolved_lambda, FROZEN_LAMBDA_TARGET_0P25, rel_tol=1e-9, abs_tol=1e-12)
        if lambda_value is None
        else None
    )
    r_median_matches_frozen = (
        math.isclose(r_median, FROZEN_R_MEDIAN, rel_tol=1e-9, abs_tol=1e-15)
        if r_median is not None
        else None
    )

    provenance = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": _require_torch().__version__,
        "torch_cuda_available": bool(_require_torch().cuda.is_available()),
        "platform": platform.platform(),
        "seeds": [int(seed) for seed in seeds],
        "student_temperature": STUDENT_TEMPERATURE,
        "init_std": INIT_STD,
        "student_dim": 128,
        "config_sha256": config_sha256(config_path),
        "index_sha256": index_file_sha256(index_path),
        "draft_config_sha256": draft_sha,
        "draft_config_path": DRAFT_RELATIVE_PATH,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_npz_aggregate_sha256": input_aggregate_sha,
        "input_npz_scope": "train split only; aggregate over sorted (sample_id, file sha256)",
        "input_npz_count": len(resolved_paths),
    }

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "report_version": REPORT_VERSION,
        "report_stem": REPORT_STEM,
        "status": "COMPUTED" if not pair_failures else "PAIR_INCOMPLETE",
        "scope": SCOPE_NOTE,
        "draft_contract_checks": contract,
        "inputs": {
            "config": str(config_path),
            "draft_config": str(draft_path),
            "index": str(index_path),
            "features_dir": str(directory),
        },
        "split_counts": {name: len(value) for name, value in split.items()},
        "split_group_summary": group_summary,
        "isolation": isolation,
        "provenance": provenance,
        "integrity": {
            "train_record_count": len(train_samples),
            "train_image_count": len(image_pairs),
            "expected_train_records": EXPECTED_TRAIN_RECORDS,
            "expected_train_images": EXPECTED_TRAIN_IMAGES,
            "matches_expected": (
                len(train_samples) == EXPECTED_TRAIN_RECORDS
                and len(image_pairs) == EXPECTED_TRAIN_IMAGES
            ),
            "pair_failures": pair_failures,
        },
        "concentration_audit": {
            "reference_lambda_name": REFERENCE_LAMBDA_NAME,
            "reference_lambda_definition": "0.25 / r_median (frozen diagnostic target ratio)",
            "frozen_r_median": FROZEN_R_MEDIAN,
            "computed_r_median": r_median,
            "r_median_matches_frozen": r_median_matches_frozen,
            "frozen_lambda_target_0p25": FROZEN_LAMBDA_TARGET_0P25,
            "lambda_target_0p25_used": resolved_lambda,
            "lambda_matches_frozen": lambda_matches_frozen,
            "lambda_current_reference": LAMBDA_CURRENT,
            "target_lambda_ratios": dict(TARGET_LAMBDA_RATIOS),
            "signed_sum_error_max": signed_sum_error_max,
            "per_seed": per_seed_blocks,
            "cross_seed_per_param": cross_seed_per_param,
            "pooled": pooled,
            "pooled_correlations": _pooled_correlations(per_image_rows),
            "cross_seed_pair_norm_rank": cross_seed_rank,
            "hardware_contract": {
                "query_shape": [1, 128],
                "scorer": "Gx128 dot",
                "token_axis_reduction": "none (prompt mean vector)",
            },
        },
        "formulas": FORMULAS,
        "limitations": list(LIMITATIONS),
        "notes": {
            "zero_update": (
                "only forward and autograd.grad are used; no update rule is "
                "constructed, no parameter is written, no epoch is run"
            ),
            "norm_sum_is_not_aggregate": (
                "summing per-image gradient norms is NOT the aggregate gradient; "
                "norm shares characterise spread of magnitude only"
            ),
            "signed_contribution": (
                "dot(g_i, sum_j g_j) / ||sum_j g_j||^2 sums to 1 per seed in exact "
                "arithmetic and may be negative; the mean-gradient form "
                "dot(g_i, mean) / (N * ||mean||^2) is used to avoid materialising "
                "the full 159 x 458752 gradient set. The pooled block spans all "
                "seeds, so its expected sum is the seed count, not 1."
            ),
            "lambda_positive_scaling": (
                "a positive lambda only rescales pair vs base; pair-only "
                "concentration, ranks and cosines are invariant"
            ),
            "descriptive_only": (
                "no pass/fail threshold is set, no winner or promotion verdict is "
                "given, and the DRAFT is not modified"
            ),
            "not_task_accuracy": (
                "all numbers are internal train-only initial-gradient diagnostics, "
                "never CV-Bench task accuracy"
            ),
            "top_share_convention": (
                "top-k% share uses k = max(1, ceil(fraction * n)) over the positive "
                "values renormalised to sum to 1"
            ),
        },
    }
    return report, per_image_rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

CSV_FIELDS: tuple[str, ...] = (
    "seed",
    "image_sha256",
    "grounding_sample_id",
    "pointing_sample_id",
    "group_count",
    "js_nats",
    "l_base",
    "l_pair",
    "d_swap",
    "l_total",
    "hinge_margin",
    "hinge_active",
    "lambda_0p25",
    "Wt_base_grad_norm",
    "Wt_pair_grad_norm",
    "Wt_pair_over_base_norm_ratio",
    "Wt_cos_pair_aggregate",
    "Wt_cos_base_aggregate",
    "Wt_pair_signed_contribution",
    "Wt_base_signed_contribution",
    "Wt_lambda_0p25_scaled_pair_norm",
    "Wt_scaled_pair_over_base_norm_ratio",
    "Wt_pair_zero_fraction",
    "Wt_pair_nonfinite_fraction",
    "Wv_base_grad_norm",
    "Wv_pair_grad_norm",
    "Wv_pair_over_base_norm_ratio",
    "Wv_cos_pair_aggregate",
    "Wv_cos_base_aggregate",
    "Wv_pair_signed_contribution",
    "Wv_base_signed_contribution",
    "Wv_lambda_0p25_scaled_pair_norm",
    "Wv_scaled_pair_over_base_norm_ratio",
    "Wv_pair_zero_fraction",
    "Wv_pair_nonfinite_fraction",
    "concat_base_grad_norm",
    "concat_pair_grad_norm",
    "concat_pair_over_base_norm_ratio",
    "concat_cos_pair_aggregate",
    "concat_cos_base_aggregate",
    "concat_pair_signed_contribution",
    "concat_base_signed_contribution",
    "concat_lambda_0p25_scaled_pair_norm",
    "concat_scaled_pair_over_base_norm_ratio",
    "concat_pair_zero_fraction",
    "concat_pair_nonfinite_fraction",
)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = _csv.DictWriter(stream, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_FIELDS})


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _concentration_line(metrics: Mapping[str, Any]) -> str:
    if "effective_sample_size" in metrics:
        ess = metrics["effective_sample_size"]
        gini = metrics.get("gini")
    else:
        ess = metrics.get("positive_effective_sample_size")
        gini = metrics.get("positive_gini")
    return (
        f"top1%={_fmt(metrics.get('top_1pct'))}, top5%={_fmt(metrics.get('top_5pct'))}, "
        f"top10%={_fmt(metrics.get('top_10pct'))}, top20%={_fmt(metrics.get('top_20pct'))}, "
        f"ESS={_fmt(ess)}, Gini={_fmt(gini)}"
    )


def _signed_line(metrics: Mapping[str, Any]) -> str:
    return (
        f"sum={_fmt(metrics.get('sum'))}, |sum-expected|={_fmt(metrics.get('sum_error_abs'))}, "
        f"expected={_fmt(metrics.get('expected_sum'))}, "
        f"neg_count={metrics.get('negative_count')}, "
        f"neg_fraction={_fmt(metrics.get('negative_fraction'))}, "
        f"neg_mass={_fmt(metrics.get('negative_mass_abs'))}, "
        f"pos_sum={_fmt(metrics.get('positive_sum'))}"
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    provenance = report["provenance"]
    audit = report["concentration_audit"]
    integrity = report["integrity"]
    isolation = report["isolation"]
    split_counts = report["split_counts"]
    lines: list[str] = []
    lines.append(f"# Pair 梯度贡献集中度审计（train-only，{REPORT_STEM}）")
    lines.append("")
    lines.append(
        "本轮只做**零更新、CPU-only、train-only** 的成对梯度贡献集中度审计："
        "先用三 seed 初始 concat pair/base 比对 `r_med` 与冻结 "
        f"`lambda_target_0p25={_fmt(audit['frozen_lambda_target_0p25'])}`，"
        "再逐 image pair 计算 `L_base_i`/`L_pair_i` 的 Wt/Wv/concat 梯度并**只保留标量**，"
        "报告 signed projection contribution、负贡献、top-share、ESS、Gini、rank 稳定性与相关性。"
        "不训练、不调参、不保存 `Wt/Wv`、不读 val/test 特征缓存、不加载模型、不用 GPU、"
        "不联网、不设阈值、不给 promotion 结论。"
    )
    lines.append("")
    lines.append("## 1. 范围与硬性隔离")
    lines.append("")
    lines.append(
        f"- split 仅由 index metadata 用冻结 `split_sample_ids` 计算："
        f"train/val/test = {split_counts['train']}/{split_counts['val']}/{split_counts['test']} 条记录。"
    )
    lines.append(
        f"- 只打开 train NPZ：opened_npz_count={isolation['opened_npz_count']}，"
        f"val_test_npz_opened={isolation['val_test_npz_opened']}，"
        f"opened_only_train_npz={isolation['opened_only_train_npz']}。"
    )
    lines.append(
        f"- train 记录/图片 = {integrity['train_record_count']}/{integrity['train_image_count']}"
        f"（期望 {integrity['expected_train_records']}/{integrity['expected_train_images']}，"
        f"matches_expected={integrity['matches_expected']}）；status={report['status']}。"
    )
    lines.append(
        f"- draft 合同校验（只读）：全部通过 = {all(report['draft_contract_checks'].values())}；"
        f"draft 状态保持 `{EXPECTED_DRAFT_STATUS}`，本审计不修改 DRAFT。"
    )
    lines.append("")
    lines.append("## 2. 输入与 provenance")
    lines.append("")
    lines.append(
        f"- python {provenance['python']} / numpy {provenance['numpy']} / "
        f"torch {provenance['torch']}（cuda_available={provenance['torch_cuda_available']}）"
    )
    lines.append(
        f"- seeds={provenance['seeds']}，student temperature={STUDENT_TEMPERATURE}，"
        f"init_std={INIT_STD}，score_dim={provenance['student_dim']}。"
    )
    lines.append(f"- 训练 config SHA256 `{provenance['config_sha256']}`")
    lines.append(f"- index SHA256 `{provenance['index_sha256']}`")
    lines.append(f"- draft config SHA256 `{provenance['draft_config_sha256']}`（只读）")
    lines.append(f"- script SHA256 `{provenance['script_sha256']}`")
    lines.append(
        f"- 输入 NPZ aggregate SHA256 `{provenance['input_npz_aggregate_sha256']}`"
        f"（仅 train，{provenance['input_npz_count']} 个文件）"
    )
    lines.append("")
    lines.append("## 3. lambda 与 r_med 校验")
    lines.append("")
    lines.append(
        f"- 冻结 `r_med={_fmt(audit['frozen_r_median'], 18)}`；"
        f"本审计三 seed 重算 `r_med={_fmt(audit['computed_r_median'], 18)}`；"
        f"matches_frozen={audit['r_median_matches_frozen']}。"
    )
    lines.append(
        f"- 冻结 `lambda_target_0p25={_fmt(audit['frozen_lambda_target_0p25'], 18)}`；"
        f"本审计使用 `{_fmt(audit['lambda_target_0p25_used'], 18)}`；"
        f"matches_frozen={audit['lambda_matches_frozen']}（正 lambda 只改变 pair/base 相对尺度）。"
    )
    lines.append(
        f"- signed contribution 和校验最大绝对误差 = {_fmt(audit['signed_sum_error_max'])}。"
    )
    lines.append("")
    lines.append("## 4. 逐 seed 梯度集中度")
    lines.append("")
    for block in audit["per_seed"]:
        lines.append(f"### seed {block['seed']}")
        lines.append("")
        lines.append(
            f"- r_seed(concat)={_fmt(block['r_seed_concat_pair_over_base'])}，"
            f"L_base/L_pair/L_total mean={_fmt(block['loss_means']['l_base'])}/"
            f"{_fmt(block['loss_means']['l_pair'])}/{_fmt(block['loss_means']['l_total'])}，"
            f"aggregate pair/base(concat)={_fmt(block['aggregate']['concat']['pair_over_base_norm_ratio'])}。"
        )
        lines.append("")
        lines.append("| param | pair norm concentration | signed pair contribution |")
        lines.append("| --- | --- | --- |")
        for group in PARAM_GROUPS:
            per_param = block["per_param"][group]
            lines.append(
                f"| {group} | {_concentration_line(per_param['pair_grad_norm_concentration'])} | "
                f"{_signed_line(per_param['pair_signed_contribution'])} |"
            )
        lines.append("")
        lines.append(
            "| param | pair norm mean | pair/base mean | cos(pair,agg) mean | "
            "cos(base,agg) mean | pair zero frac mean | pair nonfinite frac mean |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for group in PARAM_GROUPS:
            per_param = block["per_param"][group]
            lines.append(
                "| "
                + " | ".join(
                    [
                        group,
                        _fmt(per_param["pair_grad_norm"]["mean"]),
                        _fmt(per_param["pair_over_base_norm_ratio"]["mean"]),
                        _fmt(per_param["cos_pair_aggregate"]["mean"]),
                        _fmt(per_param["cos_base_aggregate"]["mean"]),
                        _fmt(per_param["pair_zero_fraction"]["mean"]),
                        _fmt(per_param["pair_nonfinite_fraction"]["mean"]),
                    ]
                )
                + " |"
            )
        lines.append("")
        lines.append(
            f"- Spearman（本 seed，concat）：JS vs pair norm="
            f"{_fmt(block['correlations']['concat']['js_vs_pair_norm'])}，"
            f"JS vs signed contribution="
            f"{_fmt(block['correlations']['concat']['js_vs_pair_signed_contribution'])}，"
            f"G vs pair norm={_fmt(block['correlations']['concat']['group_count_vs_pair_norm'])}，"
            f"L_pair vs pair norm={_fmt(block['correlations']['concat']['l_pair_vs_pair_norm'])}，"
            f"hinge margin vs pair norm="
            f"{_fmt(block['correlations']['concat']['hinge_margin_vs_pair_norm'])}"
            f"（hinge active n={block['correlations']['concat']['hinge_active_count']}）。"
        )
        lines.append("")
    lines.append("## 5. 三 seed 聚合（pooled）")
    lines.append("")
    for group in PARAM_GROUPS:
        pooled = audit["pooled"][group]
        norm = pooled["pair_grad_norm_concentration"]
        signed = pooled["pair_signed_contribution"]
        lines.append(f"### {group}")
        lines.append("")
        lines.append(f"- pair norm 分布：{_concentration_line(norm)}")
        lines.append(f"- signed pair contribution：{_signed_line(signed)}")
        lines.append(
            f"- 正贡献集中度：top1%={_fmt(signed['top_1pct'])}，top5%={_fmt(signed['top_5pct'])}，"
            f"top10%={_fmt(signed['top_10pct'])}，top20%={_fmt(signed['top_20pct'])}，"
            f"ESS={_fmt(signed['positive_effective_sample_size'])}，Gini={_fmt(signed['positive_gini'])}"
        )
        lines.append(
            f"- base signed contribution：{_signed_line(pooled['base_signed_contribution'])}"
        )
        lines.append("")
    lines.append("## 6. 跨 seed 稳定性与相关性")
    lines.append("")
    for group in PARAM_GROUPS:
        rank = audit["cross_seed_pair_norm_rank"][group]
        lines.append(
            f"- {group} 同图 pair norm 跨 seed pairwise Spearman = {rank['pairwise_spearman']}，"
            f"mean={_fmt(rank['mean'])}（min={_fmt(rank['min'])}，max={_fmt(rank['max'])}，"
            f"shared images={rank['shared_image_count']}）。"
        )
    lines.append("")
    for group in PARAM_GROUPS:
        correlations = audit["pooled_correlations"][group]
        lines.append(
            f"- pooled {group}：JS vs pair norm={_fmt(correlations['js_vs_pair_norm'])}，"
            f"JS vs signed contribution={_fmt(correlations['js_vs_pair_signed_contribution'])}，"
            f"G vs pair norm={_fmt(correlations['group_count_vs_pair_norm'])}，"
            f"L_pair vs pair norm={_fmt(correlations['l_pair_vs_pair_norm'])}，"
            f"hinge margin vs pair norm={_fmt(correlations['hinge_margin_vs_pair_norm'])}，"
            f"active-only hinge margin vs pair norm="
            f"{_fmt(correlations['hinge_margin_vs_pair_norm_hinge_active_only'])}。"
        )
    lines.append("")
    lines.append("## 7. Top 10 图像（仅 hash）")
    lines.append("")
    for block in audit["per_seed"]:
        lines.append(f"### seed {block['seed']}")
        lines.append("")
        lines.append("| rank | image_sha256 | G | JS | L_pair | hinge_active | pair norm concat | pair signed contribution |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for rank, item in enumerate(block["top_images"]["by_pair_norm"], start=1):
            lines.append(
                f"| {rank} | {item['image_sha256']} | {item['group_count']} | "
                f"{_fmt(item['js_nats'])} | {_fmt(item['l_pair'])} | "
                f"{_fmt(item['hinge_active'])} | {_fmt(item['pair_grad_norm_concat'])} | "
                f"{_fmt(item['pair_signed_contribution_concat'])} |"
            )
        lines.append("")
        lines.append(
            "按 signed contribution 排序的 top 10 hash："
            + ", ".join(item["image_sha256"] for item in block["top_images"]["by_pair_signed_contribution"])
        )
        lines.append("")
    lines.append("## 8. 统计解释与限制")
    lines.append("")
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append(
        "- 本审计只描述初始点的每图梯度贡献分布，不改变 DRAFT 的任何字段，"
        "也不构成重训授权。"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = _project_root()
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "prompt_projection_distillation.json",
    )
    parser.add_argument("--draft", type=Path, default=root / DRAFT_RELATIVE_PATH)
    parser.add_argument(
        "--index",
        type=Path,
        default=root
        / "artifacts"
        / "token_selector"
        / "prompt_projection_features"
        / "index.jsonl",
    )
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=root / "reports" / "token_selector")
    parser.add_argument(
        "--harness-result",
        type=Path,
        default=Path("/tmp/robobrain_harness_result.json"),
    )
    parser.add_argument(
        "--harness-durable",
        type=Path,
        default=root
        / "reports"
        / "token_selector"
        / "logs"
        / f"{REPORT_STEM}-harness-result.json",
    )
    args = parser.parse_args(argv)

    report, rows = run_audit(
        config_path=args.config,
        draft_path=args.draft,
        index_path=args.index,
        features_dir=args.features_dir,
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{REPORT_STEM}.csv"
    json_path = out_dir / f"{REPORT_STEM}.json"
    md_path = out_dir / f"{REPORT_STEM}.md"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    output_hashes = {
        "csv": _sha256(csv_path),
        "json": _sha256(json_path),
        "md": _sha256(md_path),
    }
    audit = report["concentration_audit"]
    concat_pooled = audit["pooled"]["concat"]
    harness = {
        "status": "COMPUTED",
        "report": REPORT_STEM,
        "scope": report["scope"],
        "environment": {
            "python": report["provenance"]["python"],
            "numpy": report["provenance"]["numpy"],
            "torch": report["provenance"]["torch"],
            "platform": report["provenance"]["platform"],
        },
        "seeds": report["provenance"]["seeds"],
        "inputs": {
            "config_sha256": report["provenance"]["config_sha256"],
            "index_sha256": report["provenance"]["index_sha256"],
            "draft_config_sha256": report["provenance"]["draft_config_sha256"],
            "script_sha256": report["provenance"]["script_sha256"],
            "input_npz_aggregate_sha256": report["provenance"][
                "input_npz_aggregate_sha256"
            ],
        },
        "output_sha256": output_hashes,
        "artifacts": {
            "script": str(Path(__file__).resolve().relative_to(root)),
            "csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(md_path),
            "focused_tests": "evaluation/tests/test_pair_gradient_concentration_audit.py",
        },
        "isolation": report["isolation"],
        "key_metrics": {
            "train_images": report["integrity"]["train_image_count"],
            "train_records": report["integrity"]["train_record_count"],
            "computed_r_median": audit["computed_r_median"],
            "r_median_matches_frozen": audit["r_median_matches_frozen"],
            "lambda_target_0p25_used": audit["lambda_target_0p25_used"],
            "lambda_matches_frozen": audit["lambda_matches_frozen"],
            "signed_sum_error_max": audit["signed_sum_error_max"],
            "pair_norm_concentration_concat": concat_pooled[
                "pair_grad_norm_concentration"
            ],
            "signed_contribution_concentration_concat": concat_pooled[
                "pair_signed_contribution"
            ],
            "cross_seed_pair_norm_rank_concat": audit["cross_seed_pair_norm_rank"][
                "concat"
            ],
            "pooled_correlations_concat": report["concentration_audit"][
                "pooled_correlations"
            ]["concat"],
        },
        "limitations": list(report["limitations"]),
        "tests": {
            "focused": {
                "module": "evaluation/tests/test_pair_gradient_concentration_audit.py",
                "log": "reports/token_selector/logs/2026-10-03-pair-gradient-concentration-focused-tests.log",
                "command": None,
                "ran": None,
                "ok": None,
                "result": None,
            },
            "full": {
                "log": "reports/token_selector/logs/2026-10-03-pair-gradient-concentration-full-tests.log",
                "command": None,
                "baseline_before_change": None,
                "ran": None,
                "ok": None,
                "result": None,
            },
            "isolation_status": (
                f"opened_only_train_npz={report['isolation']['opened_only_train_npz']}; "
                f"val_test_npz_opened={report['isolation']['val_test_npz_opened']}; "
                f"split_from_index_metadata_only="
                f"{report['isolation']['split_from_index_metadata_only']}"
            ),
        },
        "next_decision": (
            "PI/Codex decision only: this audit is descriptive and does not train, "
            "does not modify the DRAFT and does not give a promotion/no-promotion "
            "verdict; no retraining is authorized by this report"
        ),
    }
    args.harness_result.parent.mkdir(parents=True, exist_ok=True)
    args.harness_result.write_text(
        json.dumps(harness, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.harness_durable.parent.mkdir(parents=True, exist_ok=True)
    args.harness_durable.write_text(
        json.dumps(harness, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"PAIR_GRADIENT_CONCENTRATION_AUDIT status={report['status']} "
        f"images={report['integrity']['train_image_count']}"
    )
    print(f"isolation opened_only_train_npz={report['isolation']['opened_only_train_npz']}")
    print(f"computed_r_median={audit['computed_r_median']}")
    print(f"lambda_target_0p25_used={audit['lambda_target_0p25_used']}")
    print(f"signed_sum_error_max={audit['signed_sum_error_max']}")
    print(
        "concat_pair_norm_concentration "
        f"top10pct={concat_pooled['pair_grad_norm_concentration']['top_10pct']} "
        f"ess={concat_pooled['pair_grad_norm_concentration']['effective_sample_size']} "
        f"gini={concat_pooled['pair_grad_norm_concentration']['gini']}"
    )
    print(
        "concat_signed_contribution "
        f"sum={concat_pooled['pair_signed_contribution']['sum']} "
        f"neg_fraction={concat_pooled['pair_signed_contribution']['negative_fraction']} "
        f"top10pct={concat_pooled['pair_signed_contribution']['top_10pct']} "
        f"ess={concat_pooled['pair_signed_contribution']['positive_effective_sample_size']} "
        f"gini={concat_pooled['pair_signed_contribution']['positive_gini']}"
    )
    print(
        "cross_seed_pair_norm_rank_concat "
        f"mean={audit['cross_seed_pair_norm_rank']['concat']['mean']}"
    )
    print(
        "pooled_correlations_concat "
        f"js_vs_pair_norm={report['concentration_audit']['pooled_correlations']['concat']['js_vs_pair_norm']} "
        f"js_vs_signed={report['concentration_audit']['pooled_correlations']['concat']['js_vs_pair_signed_contribution']}"
    )
    print(
        f"python={report['provenance']['python']} numpy={report['provenance']['numpy']} "
        f"torch={report['provenance']['torch']}"
    )
    print(f"input_config_sha256={report['provenance']['config_sha256']}")
    print(f"input_index_sha256={report['provenance']['index_sha256']}")
    print(f"input_draft_sha256={report['provenance']['draft_config_sha256']}")
    print(f"input_script_sha256={report['provenance']['script_sha256']}")
    print(f"input_npz_aggregate_sha256={report['provenance']['input_npz_aggregate_sha256']}")
    for name, digest in output_hashes.items():
        print(f"output_{name}_sha256={digest}")
    print(f"harness_result={args.harness_result}")
    print(f"harness_durable={args.harness_durable}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
