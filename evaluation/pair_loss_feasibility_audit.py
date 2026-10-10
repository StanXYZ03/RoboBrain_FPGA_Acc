#!/usr/bin/env python3
"""Train-only pair-loss **feasibility + initialisation-gradient** audit.

This module is a **zero-update, CPU-only, train-only** mathematical and
gradient-feasibility audit of the *draft* paired-supervision loss described in
``configs/prompt_mean_pairwise_retrain.draft.json``.  It never trains, never
constructs an update rule, never writes a new ``Wt``/``Wv``, never loads a
model, never uses a GPU, never touches the network and never opens a val/test
feature cache.

It answers exactly three bounded questions about the draft arm
``B_KL_PLUS_ADAPTIVE_SWAP_HINGE``:

1. **Teacher-optimum margin feasibility.**  With ``y_g``/``y_p`` the cached
   ``teacher_soft`` of the two prompts of one image, is the adaptive hinge
   ``relu(JS_nats(y_g, y_p) + L_base - D_swap)`` driven to zero when the student
   exactly reproduces the teacher?  At that point ``L_base = 0`` and
   ``D_swap = 0.5 (KL(y_g||y_p) + KL(y_p||y_g))``; the slack is
   ``D_swap - JS``.  Every per-image slack is reported, including any negative
   value (a negative slack would mean the hinge cannot vanish even for a perfect
   student).
2. **Does an initialisation gradient exist and is it finite?**  For the three
   fixed seeds and the frozen ``prompt_mean_vector`` scorer, ``autograd.grad``
   is called on the per-seed *train-image mean* of ``L_base``, ``L_pair`` and
   ``L_total`` with respect to ``Wt``/``Wv``; no parameter is updated.
3. **How aligned are the KL and pair gradients?**  The ``L_base``/``L_pair``
   gradient norms, their ratio and their cosine (for ``Wt``, ``Wv`` and the
   concatenation) are reported per seed.

The audit fixes **no pass/fail threshold** and makes **no claim** that a retrain
will succeed; a non-zero gradient is not accuracy and is not evidence of
learnability.

Isolation contract (enforced and unit-tested):

* the train/val/test split is computed **only from index metadata** using the
  frozen :func:`evaluation.prompt_projection_distillation.split_sample_ids`;
* only train feature caches are opened, and a spy loader / corrupted val-test
  npz must not move any train metric or the train aggregate SHA;
* the scorer is exactly the frozen ``prompt_mean_vector`` convention
  (``prompt tokens -> Wt -> row L2 normalise -> mean -> L2 normalise``;
  ``visual -> Wv -> row L2 normalise``; ``G x 128`` dot);
* the draft config is only *read*; its status remains ``DRAFT_...`` and it is
  never modified.
"""

from __future__ import annotations

import argparse
import csv as _csv
import hashlib
import json
import math
import platform
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluation.metrics import load_jsonl
from evaluation.prompt_projection_distillation import (
    DistillationContractError,
    config_sha256,
    load_distillation_config,
    load_feature_cache,
    split_group_summary,
    split_sample_ids,
)
from evaluation.teacher_pair_signal_audit import (
    aggregate_npz_sha256,
    describe,
    js_divergence_nats,
    kl_divergence_nats,
    load_index_metadata,
    symmetric_kl_nats,
)
from evaluation.train_prompt_projection import index_file_sha256, init_projection

__all__ = [
    "INIT_SEEDS",
    "REPORT_STEM",
    "STUDENT_TEMPERATURE",
    "draft_contract_checks",
    "kl_teacher_student",
    "load_draft_config",
    "load_train_samples",
    "pair_loss_terms",
    "prompt_query",
    "render_markdown",
    "run_audit",
    "student_distribution",
    "student_log_prob",
    "student_scores",
    "teacher_math_statistics",
    "visual_unit",
    "write_init_csv",
    "write_teacher_csv",
]

# ---------------------------------------------------------------------------
# Frozen audit constants.  ``INIT_SEEDS`` / ``STUDENT_TEMPERATURE`` /
# ``INIT_STD`` reproduce the *draft* optimiser contract and are cross-checked
# against the draft JSON at run time (see :func:`draft_contract_checks`).
# ---------------------------------------------------------------------------
REPORT_VERSION = "1.0"
REPORT_STEM = "2026-10-02-pair-loss-feasibility-train-only"
INIT_METRICS_STEM = "2026-10-02-pair-loss-feasibility-init-metrics"
INIT_SEEDS: tuple[int, ...] = (20261002, 20261003, 20261004)
STUDENT_TEMPERATURE = 0.1
INIT_STD = 0.02
EPS = 1e-12
EXPECTED_TRAIN_RECORDS = 318
EXPECTED_TRAIN_IMAGES = 159
# The draft file itself is an input; its digest is recorded (not pinned, because
# it is explicitly a DRAFT that only the PI may freeze).
DRAFT_RELATIVE_PATH = "configs/prompt_mean_pairwise_retrain.draft.json"
EXPECTED_DRAFT_STATUS = "DRAFT_REQUIRES_PI_APPROVAL"

# ``KL(p || (p+q)/2) <= 0.5 KL(p || q)`` by convexity of KL in its second
# argument in each direction, so ``JS <= 0.5 * symmetric_KL`` and the slack
# ``symmetric_KL - JS`` is non-negative.  The identity is stated for the report
# and *not* assumed: per-image slack values are reported as computed.
JS_SKL_INEQUALITY = (
    "JS(p,q) = 0.5 KL(p||m) + 0.5 KL(q||m) with m=(p+q)/2; convexity of KL in "
    "its second argument gives KL(p||m) <= 0.5 KL(p||q) and KL(q||m) <= "
    "0.5 KL(q||p), hence JS <= 0.5(KL(p||q)+KL(q||p)) = symmetric_KL and the "
    "adaptive-margin slack D_swap - JS is non-negative at the teacher optimum."
)
SCOPE_NOTE = (
    "train-only, CPU-only, zero-update mathematical and initialisation-gradient "
    "feasibility audit of the DRAFT pair loss; no training, no tuning, no "
    "parameter save, no val/test feature cache read, no model load, no GPU, "
    "no network, no promotion threshold, no accuracy claim"
)

# Static safety contract: these source tokens must never appear in this module.
# The literals are assembled at import time so that this very list cannot make
# the static safety test trip on itself.
FORBIDDEN_SOURCE_TOKENS: tuple[str, ...] = (
    "." + "step" + "(",
    "torch" + "." + "optim",
    "Adam" + "W",
    "save" + "_projection",
    "np" + "." + "savez",
)


# ---------------------------------------------------------------------------
# Draft config: read-only contract cross-check
# ---------------------------------------------------------------------------


def load_draft_config(path: Path) -> tuple[dict[str, Any], str]:
    """Read the *draft* JSON verbatim and return ``(payload, sha256)``."""
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise DistillationContractError(f"draft {path} is not a JSON object")
    return dict(payload), hashlib.sha256(raw).hexdigest()


def _temperature_from_draft(draft: Mapping[str, Any]) -> float | None:
    for arm in draft.get("arms", []):
        text = str(arm.get("definitions", {}).get("student_distribution", ""))
        match = re.search(r"softmax\(\s*student_score\s*/\s*([0-9.eE+-]+)\s*\)", text)
        if match:
            return float(match.group(1))
    return None


def draft_contract_checks(draft: Mapping[str, Any]) -> dict[str, bool]:
    """Verify the audit constants still match the draft; never edits the draft."""
    optimisation = draft.get("optimization", {})
    arm_losses = {
        str(arm.get("name")): str(arm.get("loss", "")) for arm in draft.get("arms", [])
    }
    b_loss = arm_losses.get("B_KL_PLUS_ADAPTIVE_SWAP_HINGE", "")
    b_defs = next(
        (
            arm.get("definitions", {})
            for arm in draft.get("arms", [])
            if str(arm.get("name")) == "B_KL_PLUS_ADAPTIVE_SWAP_HINGE"
        ),
        {},
    )
    seeds_match = [int(value) for value in optimisation.get("seeds", [])] == list(INIT_SEEDS)
    temperature = _temperature_from_draft(draft)
    return {
        "status_is_draft": str(draft.get("status")) == EXPECTED_DRAFT_STATUS,
        "seeds_match_audit": seeds_match,
        "init_std_0p02": "init_std=0.02" in str(optimisation.get("initialization", "")),
        "temperature_0p1": temperature is not None and math.isclose(
            float(temperature), STUDENT_TEMPERATURE, rel_tol=0.0, abs_tol=1e-12
        ),
        "loss_contains_hinge": "relu" in b_loss and "D_swap" in b_loss,
        "loss_contains_l_base": "L_base" in b_loss,
        "definitions_define_d_swap": "D_swap" in b_defs,
        "definitions_define_margin_js": "JS_nats" in str(b_defs.get("margin", "")),
    }


# ---------------------------------------------------------------------------
# Train-only feature load (with the visual/prompt tensors the scorer needs)
# ---------------------------------------------------------------------------


def _resolve_npz(record: Mapping[str, Any], directory: Path) -> Path:
    path = Path(str(record["npz"]))
    if not path.is_absolute():
        path = directory / path
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_train_samples(
    index_path: Path,
    features_dir: Path | None,
    train_ids: Sequence[str],
    *,
    opened: set[str] | None = None,
    cache_loader: Callable[..., dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    """Load **only** the train feature caches with the scorer's inputs.

    Mirrors :func:`evaluation.teacher_pair_signal_audit.load_train_samples` but
    additionally keeps ``group_mean`` and ``prompt_embeddings``.  ``opened``
    receives every touched path so a test can prove no val/test cache is read;
    ``cache_loader`` defaults to the frozen loader and is injectable.
    """
    loader = cache_loader or load_feature_cache
    directory = Path(features_dir) if features_dir else Path(index_path).parent
    by_id = {record["sample_id"]: record for record in load_index_metadata(index_path)}
    missing = [sample_id for sample_id in train_ids if sample_id not in by_id]
    if missing:
        raise DistillationContractError(f"train ids absent from index: {missing}")
    samples: list[dict[str, Any]] = []
    hashes: list[tuple[str, str]] = []
    resolved_paths: list[str] = []
    for sample_id in sorted(str(value) for value in train_ids):
        record = by_id[sample_id]
        npz_path = _resolve_npz(record, directory)
        if not npz_path.is_file():
            raise FileNotFoundError(f"train feature cache does not exist: {npz_path}")
        file_sha = _file_sha256(npz_path)
        if opened is not None:
            opened.add(str(npz_path))
        cache = loader(npz_path)
        hashes.append((sample_id, file_sha))
        resolved_paths.append(str(npz_path))
        samples.append(
            {
                "sample_id": sample_id,
                "task": str(record["task"]),
                "source_image_sha256": str(record["source_image_sha256"]),
                "grid_h": int(record["grid_h"]),
                "grid_w": int(record["grid_w"]),
                "group_count": int(record["group_count"]),
                "prompt_token_count": int(record["prompt_token_count"]),
                "npz": str(npz_path),
                "npz_sha256": file_sha,
                "group_mean": np.asarray(cache["group_mean"], dtype=np.float64),
                "prompt_embeddings": np.asarray(
                    cache["prompt_embeddings"], dtype=np.float64
                ),
                "teacher_soft": np.asarray(cache["teacher_soft_target"], dtype=np.float64),
            }
        )
    return samples, aggregate_npz_sha256(hashes), resolved_paths


# ---------------------------------------------------------------------------
# Part 1: teacher-optimum margin slack (pure numpy, cached teacher_soft)
# ---------------------------------------------------------------------------


def teacher_math_row(grounding: Mapping[str, Any], pointing: Mapping[str, Any]) -> dict[str, Any]:
    """Per-image teacher-optimum margin arithmetic for one train image pair."""
    y_g = np.asarray(grounding["teacher_soft"], dtype=np.float64).reshape(-1)
    y_p = np.asarray(pointing["teacher_soft"], dtype=np.float64).reshape(-1)
    if y_g.shape != y_p.shape:
        raise DistillationContractError(
            f"{grounding['sample_id']}: teacher_soft shapes differ"
        )
    js = js_divergence_nats(y_g, y_p)
    kl_gp = kl_divergence_nats(y_g, y_p)
    kl_pg = kl_divergence_nats(y_p, y_g)
    d_swap = 0.5 * (kl_gp + kl_pg)
    slack = d_swap - js
    ratio = (d_swap / js) if js > 0.0 else None
    group_count = int(np.asarray(y_g).size)
    return {
        "image_sha256": str(grounding["source_image_sha256"]),
        "grounding_sample_id": str(grounding["sample_id"]),
        "pointing_sample_id": str(pointing["sample_id"]),
        "group_count": group_count,
        "grid_h": int(grounding["grid_h"]),
        "grid_w": int(grounding["grid_w"]),
        "js_nats": float(js),
        "kl_grounding_pointing": float(kl_gp),
        "kl_pointing_grounding": float(kl_pg),
        "symmetric_kl": float(d_swap),
        "slack": float(slack),
        "slack_nonnegative": bool(slack >= 0.0),
        "js_positive": bool(js > 0.0),
        "d_swap_over_js": (None if ratio is None else float(ratio)),
    }


def teacher_math_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-image teacher slack / ratio distributions."""
    slacks = [float(row["slack"]) for row in rows]
    js_values = [float(row["js_nats"]) for row in rows]
    skl_values = [float(row["symmetric_kl"]) for row in rows]
    ratios = [float(row["d_swap_over_js"]) for row in rows if row["d_swap_over_js"] is not None]
    negative = [int(row["group_count"]) for row in rows if not row["slack_nonnegative"]]
    return {
        "image_count": len(rows),
        "independent_image_units": len(rows),
        "slack_stats": describe(slacks),
        "js_nats_stats": describe(js_values),
        "symmetric_kl_stats": describe(skl_values),
        "d_swap_over_js_stats": describe(ratios),
        "d_swap_over_js_count": len(ratios),
        "slack_nonnegative_fraction": (
            sum(1 for row in rows if row["slack_nonnegative"]) / float(len(rows))
            if rows
            else 0.0
        ),
        "slack_negative_image_count": len(negative),
        "slack_negative_group_counts": negative,
        "js_positive_fraction": (
            sum(1 for row in rows if row["js_positive"]) / float(len(rows)) if rows else 0.0
        ),
        "min_ratio_ge_2": (min(ratios) >= 2.0) if ratios else None,
        "min_slack_ge_0": (min(slacks) >= 0.0) if slacks else None,
        "slack_inequality_note": JS_SKL_INEQUALITY,
    }


# ---------------------------------------------------------------------------
# Part 2: frozen prompt_mean_vector scorer (torch, differentiable)
# ---------------------------------------------------------------------------


def _require_torch() -> Any:
    import torch

    return torch


def prompt_query(prompt_embeddings: Any, Wt: Any, *, eps: float = EPS) -> Any:
    """Frozen ``prompt_mean_vector`` query: project -> row-L2 -> mean -> L2."""
    torch = _require_torch()
    proj = prompt_embeddings @ Wt.transpose(0, 1)
    unit = proj / proj.norm(dim=-1, keepdim=True).clamp_min(eps)
    query = unit.mean(dim=0)
    return query / query.norm().clamp_min(eps)


def visual_unit(group_mean: Any, Wv: Any, *, eps: float = EPS) -> Any:
    """Frozen visual projection: ``group_mean @ Wv.T`` then row L2 normalise."""
    proj = group_mean @ Wv.transpose(0, 1)
    return proj / proj.norm(dim=-1, keepdim=True).clamp_min(eps)


def student_scores(
    group_mean: Any, prompt_embeddings: Any, Wt: Any, Wv: Any, *, eps: float = EPS
) -> Any:
    """Differentiable ``G``-vector of the frozen ``prompt_mean_vector`` scorer."""
    return visual_unit(group_mean, Wv, eps=eps) @ prompt_query(
        prompt_embeddings, Wt, eps=eps
    )


def student_log_prob(scores: Any, *, temperature: float = STUDENT_TEMPERATURE) -> Any:
    """``log softmax(score / temperature)`` of the student distribution."""
    torch = _require_torch()
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise DistillationContractError("student temperature must be finite and > 0")
    return torch.log_softmax(scores / float(temperature), dim=-1)


def student_distribution(scores: Any, *, temperature: float = STUDENT_TEMPERATURE) -> Any:
    """``softmax(score / temperature)`` of the student distribution."""
    return student_log_prob(scores, temperature=temperature).exp()


def kl_teacher_student(teacher: Any, student_log_probability: Any, *, eps: float = EPS) -> Any:
    """``KL(teacher || student)`` in nats for one sample; teacher is a constant."""
    torch = _require_torch()
    teacher = teacher / teacher.sum().clamp_min(eps)
    return torch.sum(
        teacher * (torch.log(teacher.clamp_min(eps)) - student_log_probability)
    )


def pair_loss_terms(
    scores_g: Any,
    scores_p: Any,
    teacher_g: Any,
    teacher_p: Any,
    js_gp: float,
    *,
    temperature: float = STUDENT_TEMPERATURE,
) -> tuple[Any, Any, Any, Any, Any]:
    """The draft pair loss for one image: ``(L_base, L_pair, D_swap, L_total, margin)``.

    ``L_base = 0.5 (KL(y_g||p_g) + KL(y_p||p_p))``,
    ``D_swap = 0.5 (KL(y_g||p_p) + KL(y_p||p_g))``,
    ``L_pair = relu(JS_nats(y_g,y_p) + L_base - D_swap)`` and
    ``L_total = L_base + L_pair``.  ``margin`` is the (pre-ReLU) hinge argument.
    """
    logp_g = student_log_prob(scores_g, temperature=temperature)
    logp_p = student_log_prob(scores_p, temperature=temperature)
    l_base = 0.5 * (
        kl_teacher_student(teacher_g, logp_g) + kl_teacher_student(teacher_p, logp_p)
    )
    d_swap = 0.5 * (
        kl_teacher_student(teacher_g, logp_p) + kl_teacher_student(teacher_p, logp_g)
    )
    margin = float(js_gp) + l_base - d_swap
    l_pair = _require_torch().relu(margin)
    l_total = l_base + l_pair
    return l_base, l_pair, d_swap, l_total, margin


# ---------------------------------------------------------------------------
# Part 2b: initialisation gradient diagnostics (zero-update)
# ---------------------------------------------------------------------------


def _cosine(first: Any, second: Any) -> float | None:
    a = first.detach().reshape(-1)
    b = second.detach().reshape(-1)
    na = float(a.norm())
    nb = float(b.norm())
    if na <= 0.0 or nb <= 0.0:
        return None
    return float((a * b).sum() / (na * nb))


def _grad_diagnostics(base_grads: Any, pair_grads: Any, total_grads: Any) -> dict[str, Any]:
    """Per-parameter gradient norms / ratio / cosine for base vs pair."""
    groups: dict[str, tuple[Any, Any, Any]] = {
        "Wt": (base_grads[0], pair_grads[0], total_grads[0]),
        "Wv": (base_grads[1], pair_grads[1], total_grads[1]),
        "concat": (
            _require_torch().cat([base_grads[0].detach().reshape(-1), base_grads[1].detach().reshape(-1)]),
            _require_torch().cat([pair_grads[0].detach().reshape(-1), pair_grads[1].detach().reshape(-1)]),
            _require_torch().cat([total_grads[0].detach().reshape(-1), total_grads[1].detach().reshape(-1)]),
        ),
    }
    result: dict[str, Any] = {}
    for name, (base_g, pair_g, total_g) in groups.items():
        base_norm = float(base_g.norm())
        pair_norm = float(pair_g.norm())
        total_norm = float(total_g.norm())
        result[name] = {
            "l_base_norm": base_norm,
            "l_pair_norm": pair_norm,
            "l_total_norm": total_norm,
            "l_base_finite": bool(_require_torch().isfinite(base_g).all()),
            "l_pair_finite": bool(_require_torch().isfinite(pair_g).all()),
            "l_total_finite": bool(_require_torch().isfinite(total_g).all()),
            "l_base_nonzero": bool(base_norm > 0.0),
            "l_pair_nonzero": bool(pair_norm > 0.0),
            "l_total_nonzero": bool(total_norm > 0.0),
            "pair_over_base_norm_ratio": (pair_norm / base_norm) if base_norm > 0.0 else None,
            "base_pair_cosine": _cosine(base_g, pair_g),
        }
    return result


def _init_seed_metrics(
    *,
    seed: int,
    image_pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], float]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the zero-update forward/grad audit for one seed; return (metrics, rows)."""
    torch = _require_torch()
    torch.set_num_threads(1)
    Wt, Wv = init_projection(seed, INIT_STD)
    base_terms: list[Any] = []
    pair_terms: list[Any] = []
    total_terms: list[Any] = []
    rows: list[dict[str, Any]] = []
    for grounding, pointing, js_gp in image_pairs:
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
        base_terms.append(l_base)
        pair_terms.append(l_pair)
        total_terms.append(l_total)
        with torch.no_grad():
            finite = int(torch.isfinite(scores_g).sum() + torch.isfinite(scores_p).sum())
            total_scores = int(scores_g.numel() + scores_p.numel())
            rows.append(
                {
                    "seed": int(seed),
                    "image_sha256": str(grounding["source_image_sha256"]),
                    "grounding_sample_id": str(grounding["sample_id"]),
                    "pointing_sample_id": str(pointing["sample_id"]),
                    "group_count": int(np.asarray(grounding["teacher_soft"]).size),
                    "js_nats": float(js_gp),
                    "l_base": float(l_base.item()),
                    "l_pair": float(l_pair.item()),
                    "d_swap": float(d_swap.item()),
                    "l_total": float(l_total.item()),
                    "hinge_margin": float(margin.item()),
                    "hinge_active": bool(margin.item() > 0.0),
                    "own_better_than_swap": bool(l_base.item() < d_swap.item()),
                    "query_cosine": float(torch.dot(q_g, q_p).item()),
                    "exact_query_equal": bool(torch.equal(q_g, q_p)),
                    "exact_score_equal": bool(torch.equal(scores_g, scores_p)),
                    "score_finite": bool(finite == total_scores),
                }
            )
    loss_base = torch.stack(base_terms).mean()
    loss_pair = torch.stack(pair_terms).mean()
    loss_total = torch.stack(total_terms).mean()
    base_grads = torch.autograd.grad(loss_base, [Wt, Wv], retain_graph=True)
    pair_grads = torch.autograd.grad(loss_pair, [Wt, Wv], retain_graph=True)
    total_grads = torch.autograd.grad(loss_total, [Wt, Wv], retain_graph=False)
    l_base_values = [row["l_base"] for row in rows]
    l_pair_values = [row["l_pair"] for row in rows]
    l_total_values = [row["l_total"] for row in rows]
    query_cosines = [row["query_cosine"] for row in rows]
    metrics = {
        "seed": int(seed),
        "image_count": len(rows),
        "parameter_count": int(Wt.numel() + Wv.numel()),
        "loss_stats": {
            "l_base": describe(l_base_values),
            "l_pair": describe(l_pair_values),
            "l_total": describe(l_total_values),
            "d_swap": describe([row["d_swap"] for row in rows]),
        },
        "loss_means": {
            "l_base": float(loss_base.item()),
            "l_pair": float(loss_pair.item()),
            "l_total": float(loss_total.item()),
        },
        "hinge_active_image_fraction": (
            sum(1 for row in rows if row["hinge_active"]) / float(len(rows))
        ),
        "own_better_than_swap_fraction": (
            sum(1 for row in rows if row["own_better_than_swap"]) / float(len(rows))
        ),
        "query_cosine_stats": describe(query_cosines),
        "exact_query_equality_fraction": (
            sum(1 for row in rows if row["exact_query_equal"]) / float(len(rows))
        ),
        "exact_score_equality_fraction": (
            sum(1 for row in rows if row["exact_score_equal"]) / float(len(rows))
        ),
        "score_finite_fraction": (
            sum(1 for row in rows if row["score_finite"]) / float(len(rows))
        ),
        "gradient": _grad_diagnostics(base_grads, pair_grads, total_grads),
    }
    return metrics, rows


def _aggregate_init_metrics(per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Three-seed aggregate over the per-seed metrics."""
    pooled_l_base: list[float] = []
    pooled_l_pair: list[float] = []
    pooled_l_total: list[float] = []
    for seed_metrics in per_seed:
        pooled_l_base.extend(
            [seed_metrics["loss_stats"]["l_base"]["mean"]] * seed_metrics["image_count"]
        )
        pooled_l_pair.extend(
            [seed_metrics["loss_stats"]["l_pair"]["mean"]] * seed_metrics["image_count"]
        )
        pooled_l_total.extend(
            [seed_metrics["loss_stats"]["l_total"]["mean"]] * seed_metrics["image_count"]
        )
    gradient: dict[str, Any] = {}
    for name in ("Wt", "Wv", "concat"):
        entries = [seed_metrics["gradient"][name] for seed_metrics in per_seed]
        ratios = [
            entry["pair_over_base_norm_ratio"]
            for entry in entries
            if entry["pair_over_base_norm_ratio"] is not None
        ]
        cosines = [
            entry["base_pair_cosine"]
            for entry in entries
            if entry["base_pair_cosine"] is not None
        ]
        gradient[name] = {
            "l_base_norm_mean": float(np.mean([entry["l_base_norm"] for entry in entries])),
            "l_pair_norm_mean": float(np.mean([entry["l_pair_norm"] for entry in entries])),
            "l_total_norm_mean": float(np.mean([entry["l_total_norm"] for entry in entries])),
            "l_base_nonzero_seed_count": sum(
                1 for entry in entries if entry["l_base_nonzero"]
            ),
            "l_pair_nonzero_seed_count": sum(
                1 for entry in entries if entry["l_pair_nonzero"]
            ),
            "l_total_nonzero_seed_count": sum(
                1 for entry in entries if entry["l_total_nonzero"]
            ),
            "all_finite": all(
                entry["l_base_finite"] and entry["l_pair_finite"] and entry["l_total_finite"]
                for entry in entries
            ),
            "pair_over_base_norm_ratio_mean": (
                float(np.mean(ratios)) if ratios else None
            ),
            "base_pair_cosine_mean": (float(np.mean(cosines)) if cosines else None),
            "base_pair_cosine_min": (float(np.min(cosines)) if cosines else None),
            "base_pair_cosine_max": (float(np.max(cosines)) if cosines else None),
        }
    return {
        "seed_count": len(per_seed),
        "seeds": [seed_metrics["seed"] for seed_metrics in per_seed],
        "pooled_loss_stats": {
            "l_base": describe(pooled_l_base),
            "l_pair": describe(pooled_l_pair),
            "l_total": describe(pooled_l_total),
        },
        "mean_of_seed_means": {
            "l_base": float(np.mean([m["loss_means"]["l_base"] for m in per_seed])),
            "l_pair": float(np.mean([m["loss_means"]["l_pair"] for m in per_seed])),
            "l_total": float(np.mean([m["loss_means"]["l_total"] for m in per_seed])),
        },
        "hinge_active_image_fraction_mean": float(
            np.mean([m["hinge_active_image_fraction"] for m in per_seed])
        ),
        "own_better_than_swap_fraction_mean": float(
            np.mean([m["own_better_than_swap_fraction"] for m in per_seed])
        ),
        "exact_query_equality_fraction_mean": float(
            np.mean([m["exact_query_equality_fraction"] for m in per_seed])
        ),
        "exact_score_equality_fraction_mean": float(
            np.mean([m["exact_score_equality_fraction"] for m in per_seed])
        ),
        "score_finite_fraction_mean": float(
            np.mean([m["score_finite_fraction"] for m in per_seed])
        ),
        "gradient": gradient,
    }


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------


def _training_config_split(config: Mapping[str, Any]) -> tuple[int, Mapping[str, float], str]:
    split = config["split"]
    return int(split["seed"]), split["ratios"], str(split["salt"])


def run_audit(
    *,
    config_path: Path,
    draft_path: Path,
    index_path: Path,
    features_dir: Path | None = None,
    seeds: Sequence[int] = INIT_SEEDS,
    cache_loader: Callable[..., dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Full train-only feasibility report plus teacher CSV and init CSV rows."""
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
    val_test_paths = {
        str(_resolve_npz(record, directory))
        for record in metadata
        if record["sample_id"] in set(split["val"]) | set(split["test"])
    }
    isolation = {
        "train_only": True,
        "split_from_index_metadata_only": True,
        "val_test_npz_opened": sorted(opened & val_test_paths),
        "opened_npz_count": len(opened),
        "train_npz_count": len(resolved_paths),
        "opened_only_train_npz": not (opened & val_test_paths),
        "val_test_samples_ignored": len(split["val"]) + len(split["test"]),
    }
    if isolation["val_test_npz_opened"]:
        raise DistillationContractError(
            f"isolation violation: val/test caches opened: {isolation['val_test_npz_opened']}"
        )

    by_image: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for sample in train_samples:
        by_image[sample["source_image_sha256"]][sample["task"]] = sample
    teacher_rows: list[dict[str, Any]] = []
    pair_failures: list[str] = []
    image_pairs: list[tuple[Mapping[str, Any], Mapping[str, Any], float]] = []
    for image_hash in sorted(by_image):
        tasks = by_image[image_hash]
        if set(tasks) != {"grounding", "pointing"}:
            pair_failures.append(f"image {image_hash}: tasks={sorted(tasks)}")
            continue
        row = teacher_math_row(tasks["grounding"], tasks["pointing"])
        teacher_rows.append(row)
        image_pairs.append((tasks["grounding"], tasks["pointing"], row["js_nats"]))

    math_stats = teacher_math_statistics(teacher_rows)
    per_seed_metrics: list[dict[str, Any]] = []
    init_rows: list[dict[str, Any]] = []
    for seed in seeds:
        metrics, rows = _init_seed_metrics(seed=int(seed), image_pairs=image_pairs)
        per_seed_metrics.append(metrics)
        init_rows.extend(rows)
    aggregate = _aggregate_init_metrics(per_seed_metrics)

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
            "train_image_count": len(teacher_rows),
            "expected_train_records": EXPECTED_TRAIN_RECORDS,
            "expected_train_images": EXPECTED_TRAIN_IMAGES,
            "matches_expected": (
                len(train_samples) == EXPECTED_TRAIN_RECORDS
                and len(teacher_rows) == EXPECTED_TRAIN_IMAGES
            ),
            "pair_failures": pair_failures,
        },
        "teacher_math_audit": math_stats,
        "initialization_audit": {
            "per_seed": per_seed_metrics,
            "aggregate": aggregate,
            "hardware_contract": {
                "query_shape": [1, 128],
                "scorer": "Gx128 dot",
                "token_axis_reduction": "none (prompt mean vector)",
            },
        },
        "formulas": {
            "js_nats": "JS(p,q) = 0.5 KL(p||m) + 0.5 KL(q||m), m=(p+q)/2, natural log",
            "l_base": "0.5 * (KL(y_g||p_g) + KL(y_p||p_p))",
            "d_swap": "0.5 * (KL(y_g||p_p) + KL(y_p||p_g))",
            "l_pair": "relu(JS_nats(y_g,y_p) + L_base - D_swap)",
            "l_total": "L_base + L_pair",
            "slack": "D_swap - JS at the teacher optimum (L_base = 0)",
            "student_distribution": f"softmax(student_score / {STUDENT_TEMPERATURE})",
            "scorer": (
                "prompt tokens -> Wt -> row L2 normalise -> mean -> L2 normalise; "
                "visual -> Wv -> row L2 normalise; Gx128 dot"
            ),
        },
        "notes": {
            "zero_update": (
                "only forward and autograd.grad are used; no update rule is "
                "constructed, no parameter is written, no epoch is run"
            ),
            "feasibility_not_success": (
                "a non-negative slack only means the adaptive-margin hinge can be "
                "zero at the teacher optimum; it is NOT evidence that retraining "
                "will succeed and a non-zero initial gradient is NOT accuracy"
            ),
            "js_skl_inequality": JS_SKL_INEQUALITY,
            "no_threshold": (
                "no pass/fail threshold is invented; the audit only reports "
                "whether the margin is feasible at the teacher optimum, whether "
                "the initial gradient exists/ is finite, and how KL and pair "
                "gradients align"
            ),
            "not_task_accuracy": (
                "all numbers are internal teacher/math diagnostics, never CV-Bench "
                "task accuracy"
            ),
        },
    }
    return report, teacher_rows, init_rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

TEACHER_CSV_FIELDS: tuple[str, ...] = (
    "image_sha256",
    "grounding_sample_id",
    "pointing_sample_id",
    "group_count",
    "grid_h",
    "grid_w",
    "js_nats",
    "kl_grounding_pointing",
    "kl_pointing_grounding",
    "symmetric_kl",
    "slack",
    "slack_nonnegative",
    "js_positive",
    "d_swap_over_js",
)

INIT_CSV_FIELDS: tuple[str, ...] = (
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
    "own_better_than_swap",
    "query_cosine",
    "exact_query_equal",
    "exact_score_equal",
    "score_finite",
)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = _csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_teacher_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_csv(path, TEACHER_CSV_FIELDS, rows)


def write_init_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_csv(path, INIT_CSV_FIELDS, rows)


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


def _stats_table(stats: Mapping[str, Mapping[str, Any]]) -> list[str]:
    header = "| metric | count | mean | median | std | p5 | p25 | p75 | p95 | min | max |"
    separator = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, separator]
    for key, row in stats.items():
        lines.append(
            "| "
            + " | ".join(
                [key]
                + [
                    _fmt(row.get(field))
                    for field in (
                        "count",
                        "mean",
                        "median",
                        "std",
                        "p5",
                        "p25",
                        "p75",
                        "p95",
                        "min",
                        "max",
                    )
                ]
            )
            + " |"
        )
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    provenance = report["provenance"]
    math_stats = report["teacher_math_audit"]
    init = report["initialization_audit"]
    aggregate = init["aggregate"]
    integrity = report["integrity"]
    isolation = report["isolation"]
    split_counts = report["split_counts"]
    lines: list[str] = []
    lines.append(f"# Pair-loss 数学与初始化梯度可行性审计（train-only，{REPORT_STEM}）")
    lines.append("")
    lines.append(
        "本轮只做**零更新、CPU-only、train-only** 的数学与梯度可行性审计："
        "判断草案自适应 margin 在 teacher optimum 是否可满足、初始化梯度是否存在且有限、"
        "以及 KL 与 pair 梯度同向/冲突程度。不训练、不调参、不保存新 `Wt/Wv`、"
        "不读 val/test 特征缓存、不加载模型、不用 GPU、不联网、不设阈值、不声称重训会成功。"
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
        f"draft 状态保持 `{EXPECTED_DRAFT_STATUS}`。"
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
    lines.append(f"- draft config SHA256 `{provenance['draft_config_sha256']}`")
    lines.append(f"- script SHA256 `{provenance['script_sha256']}`")
    lines.append(
        f"- 输入 NPZ aggregate SHA256 `{provenance['input_npz_aggregate_sha256']}`"
        f"（仅 train，{provenance['input_npz_count']} 个文件）"
    )
    lines.append("")
    lines.append("## 3. Teacher optimum margin 可行性（159 image pairs）")
    lines.append("")
    lines.append(
        "teacher target 用 cached `teacher_soft`；`JS=JS_nats(y_g,y_p)`；"
        "teacher-optimal 时 `L_base=0`，`D_swap=0.5(KL(y_g||y_p)+KL(y_p||y_g))=symmetric_KL`，"
        "`slack=D_swap-JS`；`L_pair=relu(JS+L_base-D_swap)` 在 teacher optimum 归零当且仅当 slack≥0。"
    )
    lines.append("")
    lines.append(f"- slack：{ {k: _fmt(v) for k, v in math_stats['slack_stats'].items()} }")
    lines.append(
        f"- slack≥0 比例 = **{_fmt(math_stats['slack_nonnegative_fraction'])}**；"
        f"slack<0 图像数 = **{math_stats['slack_negative_image_count']}**"
        f"（group_counts={math_stats['slack_negative_group_counts']}）。"
    )
    lines.append(
        f"- `D_swap/JS` 比值（JS>0，n={math_stats['d_swap_over_js_count']}）："
        f"{ {k: _fmt(v) for k, v in math_stats['d_swap_over_js_stats'].items()} }"
    )
    lines.append(
        f"- min slack ≥ 0 = {math_stats['min_slack_ge_0']}；min ratio ≥ 2 = {math_stats['min_ratio_ge_2']}。"
    )
    lines.append(f"- 不等式依据：{math_stats['slack_inequality_note']}")
    lines.append("")
    lines.append(
        "**数学解释（仅此）**：在 teacher optimum，`L_base=0` 且 hinge 归零，"
        "因此自适应 margin **在 teacher optimum 是可满足的**；这不代表可训练成功。"
    )
    lines.append("")
    lines.append("## 4. 初始化前向/梯度审计（zero-update）")
    lines.append("")
    lines.append(
        "固定 seeds 20261002/20261003/20261004；scorer 严格复用冻结 `prompt_mean_vector`；"
        "`student distribution = softmax(score/0.1)`；只对 train-image mean loss 调 `autograd.grad`，"
        "不更新参数。"
    )
    lines.append("")
    lines.append("### 4.1 逐 seed")
    lines.append("")
    lines.append(
        "| seed | L_base mean | L_pair mean | L_total mean | hinge active frac | own-better frac | "
        "query cos mean | exact query eq frac | exact score eq frac | score finite frac |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for seed_metrics in init["per_seed"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(seed_metrics["seed"]),
                    _fmt(seed_metrics["loss_means"]["l_base"]),
                    _fmt(seed_metrics["loss_means"]["l_pair"]),
                    _fmt(seed_metrics["loss_means"]["l_total"]),
                    _fmt(seed_metrics["hinge_active_image_fraction"]),
                    _fmt(seed_metrics["own_better_than_swap_fraction"]),
                    _fmt(seed_metrics["query_cosine_stats"]["mean"]),
                    _fmt(seed_metrics["exact_query_equality_fraction"]),
                    _fmt(seed_metrics["exact_score_equality_fraction"]),
                    _fmt(seed_metrics["score_finite_fraction"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("### 4.2 梯度范数 / 比值 / cosine")
    lines.append("")
    lines.append(
        "| seed | param | base norm | pair norm | total norm | pair/base | cos(base,pair) | finite | nonzeros |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for seed_metrics in init["per_seed"]:
        for name in ("Wt", "Wv", "concat"):
            entry = seed_metrics["gradient"][name]
            nonzeros = (
                f"b={int(entry['l_base_nonzero'])}/p={int(entry['l_pair_nonzero'])}/"
                f"t={int(entry['l_total_nonzero'])}"
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(seed_metrics["seed"]),
                        name,
                        _fmt(entry["l_base_norm"]),
                        _fmt(entry["l_pair_norm"]),
                        _fmt(entry["l_total_norm"]),
                        _fmt(entry["pair_over_base_norm_ratio"]),
                        _fmt(entry["base_pair_cosine"]),
                        _fmt(bool(entry["l_base_finite"] and entry["l_pair_finite"] and entry["l_total_finite"])),
                        nonzeros,
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append("### 4.3 三 seed aggregate")
    lines.append("")
    lines.append(
        f"- pooled L_base/L_pair/L_total mean = "
        f"{_fmt(aggregate['pooled_loss_stats']['l_base']['mean'])}/"
        f"{_fmt(aggregate['pooled_loss_stats']['l_pair']['mean'])}/"
        f"{_fmt(aggregate['pooled_loss_stats']['l_total']['mean'])}；"
        f"pooled p95 = {_fmt(aggregate['pooled_loss_stats']['l_base']['p95'])}/"
        f"{_fmt(aggregate['pooled_loss_stats']['l_pair']['p95'])}/"
        f"{_fmt(aggregate['pooled_loss_stats']['l_total']['p95'])}。"
    )
    lines.append(
        f"- hinge active image fraction（seed 均值）= {_fmt(aggregate['hinge_active_image_fraction_mean'])}；"
        f"own-better-than-swap fraction = {_fmt(aggregate['own_better_than_swap_fraction_mean'])}；"
        f"exact query equality = {_fmt(aggregate['exact_query_equality_fraction_mean'])}；"
        f"exact score equality = {_fmt(aggregate['exact_score_equality_fraction_mean'])}；"
        f"score finite = {_fmt(aggregate['score_finite_fraction_mean'])}。"
    )
    for name in ("Wt", "Wv", "concat"):
        entry = aggregate["gradient"][name]
        lines.append(
            f"- {name}：base norm mean={_fmt(entry['l_base_norm_mean'])}，"
            f"pair norm mean={_fmt(entry['l_pair_norm_mean'])}，"
            f"pair/base mean={_fmt(entry['pair_over_base_norm_ratio_mean'])}，"
            f"cos(base,pair) mean={_fmt(entry['base_pair_cosine_mean'])}"
            f"（[{_fmt(entry['base_pair_cosine_min'])}, {_fmt(entry['base_pair_cosine_max'])}]），"
            f"pair 非零 seed 数={entry['l_pair_nonzero_seed_count']}/{aggregate['seed_count']}，"
            f"all_finite={entry['all_finite']}。"
        )
    lines.append("")
    lines.append("## 5. 解释与限制")
    lines.append("")
    lines.append(
        "- 本审计只回答：teacher optimum 下 margin 是否可满足、初始化梯度是否存在且有限、"
        "KL 与 pair 梯度同向/冲突程度。"
    )
    lines.append(
        "- `slack≥0` 与该结论都是**数学可行性**，不是重训成功的证据；"
        "`gradient nonzero` 不是准确率。"
    )
    lines.append(
        "- 未自创通过阈值，未据此声称任何方法会成功；草案公式若有歧义只在本报告指出，不改草案。"
    )
    lines.append(
        "- 所有数值是 internal teacher/math 诊断，不是 CV-Bench 任务精度，也不代表 student 可学。"
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
    parser.add_argument(
        "--draft", type=Path, default=root / DRAFT_RELATIVE_PATH
    )
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

    report, teacher_rows, init_rows = run_audit(
        config_path=args.config,
        draft_path=args.draft,
        index_path=args.index,
        features_dir=args.features_dir,
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_csv = out_dir / f"{REPORT_STEM}.csv"
    json_path = out_dir / f"{REPORT_STEM}.json"
    md_path = out_dir / f"{REPORT_STEM}.md"
    init_csv = out_dir / f"{INIT_METRICS_STEM}.csv"
    write_teacher_csv(teacher_csv, teacher_rows)
    write_init_csv(init_csv, init_rows)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    output_hashes = {
        "csv": _sha256(teacher_csv),
        "init_csv": _sha256(init_csv),
        "json": _sha256(json_path),
        "md": _sha256(md_path),
    }
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
            "input_npz_aggregate_sha256": report["provenance"]["input_npz_aggregate_sha256"],
        },
        "output_sha256": output_hashes,
        "artifacts": {
            "script": str(Path(__file__).resolve().relative_to(root)),
            "csv": str(teacher_csv),
            "init_csv": str(init_csv),
            "json": str(json_path),
            "markdown": str(md_path),
            "focused_tests": "evaluation/tests/test_pair_loss_feasibility_audit.py",
        },
        "isolation": report["isolation"],
        "key_metrics": {
            "train_images": report["integrity"]["train_image_count"],
            "train_records": report["integrity"]["train_record_count"],
            "slack_nonnegative_fraction": report["teacher_math_audit"][
                "slack_nonnegative_fraction"
            ],
            "slack_negative_image_count": report["teacher_math_audit"][
                "slack_negative_image_count"
            ],
            "slack_mean": report["teacher_math_audit"]["slack_stats"]["mean"],
            "d_swap_over_js_median": report["teacher_math_audit"]["d_swap_over_js_stats"][
                "median"
            ],
            "hinge_active_image_fraction_mean": report["initialization_audit"][
                "aggregate"
            ]["hinge_active_image_fraction_mean"],
            "l_base_mean": report["initialization_audit"]["aggregate"][
                "mean_of_seed_means"
            ]["l_base"],
            "l_pair_mean": report["initialization_audit"]["aggregate"][
                "mean_of_seed_means"
            ]["l_pair"],
            "l_total_mean": report["initialization_audit"]["aggregate"][
                "mean_of_seed_means"
            ]["l_total"],
            "gradient": report["initialization_audit"]["aggregate"]["gradient"],
        },
    }
    args.harness_result.parent.mkdir(parents=True, exist_ok=True)
    args.harness_result.write_text(
        json.dumps(harness, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.harness_durable.parent.mkdir(parents=True, exist_ok=True)
    args.harness_durable.write_text(
        json.dumps(harness, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"PAIR_LOSS_FEASIBILITY_AUDIT status={report['status']} images={len(teacher_rows)}")
    print(f"isolation opened_only_train_npz={report['isolation']['opened_only_train_npz']}")
    print(
        f"slack_nonnegative_fraction={report['teacher_math_audit']['slack_nonnegative_fraction']:.6f} "
        f"slack_negative_image_count={report['teacher_math_audit']['slack_negative_image_count']}"
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
