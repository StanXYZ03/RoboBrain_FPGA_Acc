#!/usr/bin/env python3
"""Train-only pair-loss **scale-stability** audit (zero-update, CPU-only).

This module is the follow-up to
:mod:`evaluation.pair_loss_feasibility_audit`.  The feasibility audit showed
that, at the three fixed seeds, the concatenated ``L_pair`` initial gradient is
only about 1--2% of the ``L_base`` initial gradient, so the draft ``lambda = 1``
pair term may be too weak to produce a visible training signal.  This audit asks
one bounded, purely **descriptive** question:

    If the pair term is re-scaled by a *pre-fixed* ``lambda``, which pair/base
    and total/base gradient-norm ratios and which base/pair/total gradient
    cosines are realised at initialisation?

The candidate scales are fixed **before** any scaled-gradient number is
computed, from the three-seed median concatenated pair/base ratio ``r_med``:

* ``lambda_current = 1.0`` (the draft, un-rescaled),
* ``lambda_target_0p1 = 0.1 / r_med``,
* ``lambda_target_0p25 = 0.25 / r_med``,
* ``lambda_target_1p0 = 1.0 / r_med``.

``0.1`` / ``0.25`` / ``1.0`` are **diagnostic target ratios only**.  They are
neither literature standards nor accuracy thresholds.  No other ``lambda`` is
invented, no winner is declared, and a non-zero initial gradient is not
accuracy and is not evidence that retraining will converge.

Implementation reuse (no second algorithm):

* split / index metadata / train-only loader come from
  :mod:`evaluation.pair_loss_feasibility_audit`;
* the frozen ``prompt_mean_vector`` scorer and the draft pair-loss arithmetic
  come from the same module;
* gradient norm/ratio/cosine diagnostics reuse
  :func:`evaluation.pair_loss_feasibility_audit._grad_diagnostics` and
  :func:`evaluation.pair_loss_feasibility_audit._cosine`;
* distribution statistics reuse
  :func:`evaluation.teacher_pair_signal_audit.describe`;
* deterministic initialisation reuses
  :func:`evaluation.train_prompt_projection.init_projection`.

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
    _grad_diagnostics,
    _require_torch,
    _training_config_split,
    draft_contract_checks,
    load_draft_config,
    load_train_samples,
    pair_loss_terms,
    prompt_query,
    visual_unit,
)
from evaluation.prompt_projection_distillation import (
    DistillationContractError,
    config_sha256,
    load_distillation_config,
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
    "CANDIDATE_LAMBDA_NAMES",
    "LAMBDA_CURRENT",
    "REPORT_STEM",
    "TARGET_LAMBDA_RATIOS",
    "aggregate_lambda_metrics",
    "candidate_lambdas",
    "render_markdown",
    "run_audit",
]

# ---------------------------------------------------------------------------
# Frozen audit constants.  The target *ratios* (not the lambdas) are fixed here
# before any gradient is computed; the realised lambdas are derived from the
# three-seed median initial pair/base ratio ``r_med``.
# ---------------------------------------------------------------------------
REPORT_VERSION = "1.0"
REPORT_STEM = "2026-10-02-pair-loss-scale-train-only"
LAMBDA_CURRENT = 1.0
TARGET_LAMBDA_RATIOS: dict[str, float] = {
    "lambda_target_0p1": 0.1,
    "lambda_target_0p25": 0.25,
    "lambda_target_1p0": 1.0,
}
CANDIDATE_LAMBDA_NAMES: tuple[str, ...] = (
    "lambda_current",
    "lambda_target_0p1",
    "lambda_target_0p25",
    "lambda_target_1p0",
)
PARAM_GROUPS: tuple[str, ...] = ("Wt", "Wv", "concat")

SCOPE_NOTE = (
    "train-only, CPU-only, zero-update scale-stability audit of the DRAFT pair "
    "loss; the candidate lambda values are fixed before any scaled-gradient "
    "number is computed from the three-seed median initial pair/base ratio; no "
    "training, no tuning, no parameter save, no val/test feature cache read, no "
    "model load, no GPU, no network, no promotion decision, no accuracy claim"
)

LIMITATIONS: tuple[str, ...] = (
    "lambda=1 的 pair 梯度约为 base 的 1–2%（初始、concat 实测），"
    "不足以说明该成对监督会产生可见训练影响。",
    "0.1 / 0.25 / 1.0 只是预先固定的诊断目标 ratio，不是文献标准，也不是精度门槛。",
    "不能从零更新初始梯度推断收敛、验证集改进或任务精度。",
    "本审计不修改 DRAFT、不训练、不给出 promotion / no-promotion 结论，也不授权重训。",
    "所有数值都是 internal teacher/math 初始梯度诊断，不是 CV-Bench 任务精度。",
)

FORMULAS: dict[str, str] = {
    "r_seed": "||grad(L_pair)|| / ||grad(L_base)|| over concatenated (Wt, Wv) at the initial parameters",
    "r_median": "median over the three fixed seeds of r_seed (concatenated)",
    "lambda_current": "1.0",
    "lambda_target_x": "x / r_median for x in {0.1, 0.25, 1.0}, fixed before scaled gradients are computed",
    "L_scaled_total": "L_base + lambda * L_pair",
    "realised_scaled_pair_over_base": "||grad(lambda * L_pair)|| / ||grad(L_base)||",
    "total_over_base": "||grad(L_base + lambda * L_pair)|| / ||grad(L_base)||",
    "cosines": "cos(grad(total), grad(base)), cos(grad(total), grad(pair)), cos(grad(base), grad(pair))",
    "loss_ratio": "lambda * mean(L_pair) / mean(L_base)",
}


# ---------------------------------------------------------------------------
# Candidate lambda set (fixed, no extra scale)
# ---------------------------------------------------------------------------


def candidate_lambdas(r_median: float) -> dict[str, float]:
    """The four pre-fixed candidate scales; ``r_median`` must be finite and > 0."""
    if not math.isfinite(float(r_median)) or float(r_median) <= 0.0:
        raise DistillationContractError(
            f"r_median must be finite and > 0 to define the target scales, got {r_median!r}"
        )
    lambdas = {"lambda_current": LAMBDA_CURRENT}
    for name, target in TARGET_LAMBDA_RATIOS.items():
        lambdas[name] = float(target) / float(r_median)
    if tuple(lambdas) != CANDIDATE_LAMBDA_NAMES:
        raise DistillationContractError("candidate lambda names drifted from the frozen set")
    return lambdas


def _stats_with_cv(values: Sequence[float]) -> dict[str, Any]:
    """``describe`` (mean/median/min/max/std) plus the coefficient of variation."""
    stats = dict(describe([float(value) for value in values if value is not None]))
    mean = stats.get("mean")
    std = stats.get("std")
    if mean is None or std is None or not math.isfinite(float(mean)) or float(mean) == 0.0:
        stats["cv"] = None
    else:
        stats["cv"] = float(std) / float(mean)
    return stats


# ---------------------------------------------------------------------------
# Zero-update forward / gradient internals (reusing the feasibility pieces)
# ---------------------------------------------------------------------------


def _forward_pair_losses(
    seed: int,
    image_pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], float]],
) -> tuple[Any, Any, Any, Any]:
    """One differentiable train-image-mean ``(Wt, Wv, L_base, L_pair)`` per seed."""
    torch = _require_torch()
    torch.set_num_threads(1)
    Wt, Wv = init_projection(int(seed), INIT_STD)
    base_terms: list[Any] = []
    pair_terms: list[Any] = []
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
        l_base, l_pair, _d_swap, _l_total, _margin = pair_loss_terms(
            scores_g, scores_p, y_g, y_p, js_gp
        )
        base_terms.append(l_base)
        pair_terms.append(l_pair)
    return Wt, Wv, torch.stack(base_terms).mean(), torch.stack(pair_terms).mean()


def _grad_triples(base_grads: Any, pair_grads: Any, total_grads: Any) -> dict[str, tuple[Any, Any, Any]]:
    """Per-parameter ``(base, pair, total)`` gradient tensors, incl. concatenation."""
    torch = _require_torch()

    def flat(tensor: Any) -> Any:
        return tensor.detach().reshape(-1)

    return {
        "Wt": (base_grads[0], pair_grads[0], total_grads[0]),
        "Wv": (base_grads[1], pair_grads[1], total_grads[1]),
        "concat": (
            torch.cat([flat(base_grads[0]), flat(base_grads[1])]),
            torch.cat([flat(pair_grads[0]), flat(pair_grads[1])]),
            torch.cat([flat(total_grads[0]), flat(total_grads[1])]),
        ),
    }


def _scale_grad_diagnostics(base_grads: Any, scaled_pair_grads: Any, total_grads: Any) -> dict[str, Any]:
    """Reuse ``_grad_diagnostics`` and add total/base and total/pair cosines."""
    core = _grad_diagnostics(base_grads, scaled_pair_grads, total_grads)
    triples = _grad_triples(base_grads, scaled_pair_grads, total_grads)
    result: dict[str, Any] = {}
    for name, (base_g, pair_g, total_g) in triples.items():
        entry = core[name]
        base_norm = float(entry["l_base_norm"])
        total_norm = float(entry["l_total_norm"])
        result[name] = {
            "l_base_norm": base_norm,
            "l_scaled_pair_norm": float(entry["l_pair_norm"]),
            "l_total_norm": total_norm,
            "scaled_pair_over_base_norm_ratio": entry["pair_over_base_norm_ratio"],
            "total_over_base_norm_ratio": (
                total_norm / base_norm if base_norm > 0.0 else None
            ),
            "cos_total_base": _cosine(total_g, base_g),
            "cos_total_pair": _cosine(total_g, pair_g),
            "cos_base_pair": entry["base_pair_cosine"],
            "l_base_finite": bool(entry["l_base_finite"]),
            "l_scaled_pair_finite": bool(entry["l_pair_finite"]),
            "l_total_finite": bool(entry["l_total_finite"]),
            "l_base_nonzero": bool(entry["l_base_nonzero"]),
            "l_scaled_pair_nonzero": bool(entry["l_pair_nonzero"]),
            "l_total_nonzero": bool(entry["l_total_nonzero"]),
        }
    return result


def _lambda_entry(
    *,
    seed: int,
    name: str,
    lam: float,
    loss_base: Any,
    loss_pair: Any,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    base_value = float(loss_base.item())
    pair_value = float(loss_pair.item())
    return {
        "seed": int(seed),
        "lambda_name": name,
        "lambda_value": float(lam),
        "target_ratio": TARGET_LAMBDA_RATIOS.get(name),
        "loss_base_mean": base_value,
        "loss_pair_mean": pair_value,
        "scaled_pair_loss_over_base_loss": (
            float(lam) * pair_value / base_value if base_value > 0.0 else None
        ),
        "gradient": {group: dict(diagnostics[group]) for group in PARAM_GROUPS},
    }


def _seed_current_lambda_entry(
    seed: int,
    image_pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], float]],
) -> dict[str, Any]:
    """``lambda = 1`` entry; also supplies ``r_seed`` for the median."""
    torch = _require_torch()
    Wt, Wv, loss_base, loss_pair = _forward_pair_losses(seed, image_pairs)
    base_grads = torch.autograd.grad(loss_base, [Wt, Wv], retain_graph=True)
    pair_grads = torch.autograd.grad(loss_pair, [Wt, Wv], retain_graph=True)
    total_grads = torch.autograd.grad(loss_base + loss_pair, [Wt, Wv], retain_graph=False)
    diagnostics = _scale_grad_diagnostics(base_grads, pair_grads, total_grads)
    return _lambda_entry(
        seed=seed,
        name="lambda_current",
        lam=LAMBDA_CURRENT,
        loss_base=loss_base,
        loss_pair=loss_pair,
        diagnostics=diagnostics,
    )


def _seed_target_lambda_entries(
    seed: int,
    image_pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], float]],
    lambdas: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    """Per-seed entries for the three pre-fixed target scales (zero-update)."""
    torch = _require_torch()
    Wt, Wv, loss_base, loss_pair = _forward_pair_losses(seed, image_pairs)
    base_grads = torch.autograd.grad(loss_base, [Wt, Wv], retain_graph=True)
    entries: dict[str, dict[str, Any]] = {}
    for name in CANDIDATE_LAMBDA_NAMES:
        if name == "lambda_current":
            continue
        lam = float(lambdas[name])
        scaled_pair = loss_pair * lam
        scaled_grads = torch.autograd.grad(scaled_pair, [Wt, Wv], retain_graph=True)
        total_grads = torch.autograd.grad(
            loss_base + scaled_pair, [Wt, Wv], retain_graph=True
        )
        diagnostics = _scale_grad_diagnostics(base_grads, scaled_grads, total_grads)
        entries[name] = _lambda_entry(
            seed=seed,
            name=name,
            lam=lam,
            loss_base=loss_base,
            loss_pair=loss_pair,
            diagnostics=diagnostics,
        )
    return entries


def aggregate_lambda_metrics(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Cross-seed mean/median/min/max/CV for one candidate scale."""
    per_param: dict[str, Any] = {}
    for group in PARAM_GROUPS:
        per_param[group] = {
            "scaled_pair_over_base_norm_ratio": _stats_with_cv(
                [e["gradient"][group]["scaled_pair_over_base_norm_ratio"] for e in entries]
            ),
            "total_over_base_norm_ratio": _stats_with_cv(
                [e["gradient"][group]["total_over_base_norm_ratio"] for e in entries]
            ),
            "cos_total_base": _stats_with_cv(
                [e["gradient"][group]["cos_total_base"] for e in entries]
            ),
            "cos_total_pair": _stats_with_cv(
                [e["gradient"][group]["cos_total_pair"] for e in entries]
            ),
            "cos_base_pair": _stats_with_cv(
                [e["gradient"][group]["cos_base_pair"] for e in entries]
            ),
            "all_finite": all(
                e["gradient"][group]["l_base_finite"]
                and e["gradient"][group]["l_scaled_pair_finite"]
                and e["gradient"][group]["l_total_finite"]
                for e in entries
            ),
            "all_nonzero": all(
                e["gradient"][group]["l_base_nonzero"]
                and e["gradient"][group]["l_scaled_pair_nonzero"]
                and e["gradient"][group]["l_total_nonzero"]
                for e in entries
            ),
        }
    return {
        "lambda_name": str(entries[0]["lambda_name"]),
        "lambda_value": float(entries[0]["lambda_value"]),
        "target_ratio": entries[0]["target_ratio"],
        "seed_count": len(entries),
        "seeds": [int(e["seed"]) for e in entries],
        "scaled_pair_loss_over_base_loss": _stats_with_cv(
            [e["scaled_pair_loss_over_base_loss"] for e in entries]
        ),
        "per_param": per_param,
        "all_finite": all(group["all_finite"] for group in per_param.values()),
        "all_nonzero": all(group["all_nonzero"] for group in per_param.values()),
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
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Full scale-stability report plus per-seed rows and per-parameter CSV rows."""
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

    # Pass A: lambda_current (1.0) per seed; supplies r_seed -> r_median.
    current_entries: list[dict[str, Any]] = []
    for seed in seeds:
        current_entries.append(_seed_current_lambda_entry(int(seed), image_pairs))
    base_ratios = {
        int(entry["seed"]): entry["gradient"]["concat"]["scaled_pair_over_base_norm_ratio"]
        for entry in current_entries
    }
    concat_ratios = [
        float(value) for value in base_ratios.values() if value is not None
    ]
    if len(concat_ratios) != len(seeds):
        raise DistillationContractError(
            "the initial concatenated pair/base ratio must be defined for every seed"
        )
    r_median = float(np.median(np.asarray(concat_ratios, dtype=np.float64)))
    lambdas = candidate_lambdas(r_median)

    # Pass B: the three pre-fixed target scales, per seed.
    per_seed_entries: dict[int, dict[str, dict[str, Any]]] = {}
    for entry in current_entries:
        per_seed_entries[int(entry["seed"])] = {"lambda_current": entry}
    for seed in seeds:
        target_entries = _seed_target_lambda_entries(int(seed), image_pairs, lambdas)
        per_seed_entries[int(seed)].update(target_entries)

    per_seed: list[dict[str, Any]] = []
    lambda_csv_rows: list[dict[str, Any]] = []
    for seed in seeds:
        ordered = {
            name: per_seed_entries[int(seed)][name] for name in CANDIDATE_LAMBDA_NAMES
        }
        per_seed.append({"seed": int(seed), "lambdas": ordered})
        for name in CANDIDATE_LAMBDA_NAMES:
            entry = ordered[name]
            for group in PARAM_GROUPS:
                gradient = entry["gradient"][group]
                lambda_csv_rows.append(
                    {
                        "seed": int(seed),
                        "lambda_name": name,
                        "lambda_value": entry["lambda_value"],
                        "target_ratio": entry["target_ratio"],
                        "param": group,
                        "base_norm": gradient["l_base_norm"],
                        "scaled_pair_norm": gradient["l_scaled_pair_norm"],
                        "total_norm": gradient["l_total_norm"],
                        "scaled_pair_over_base_norm_ratio": gradient[
                            "scaled_pair_over_base_norm_ratio"
                        ],
                        "total_over_base_norm_ratio": gradient[
                            "total_over_base_norm_ratio"
                        ],
                        "cos_total_base": gradient["cos_total_base"],
                        "cos_total_pair": gradient["cos_total_pair"],
                        "cos_base_pair": gradient["cos_base_pair"],
                        "scaled_pair_loss_over_base_loss": entry[
                            "scaled_pair_loss_over_base_loss"
                        ],
                        "loss_base_mean": entry["loss_base_mean"],
                        "loss_pair_mean": entry["loss_pair_mean"],
                        "all_finite": bool(
                            gradient["l_base_finite"]
                            and gradient["l_scaled_pair_finite"]
                            and gradient["l_total_finite"]
                        ),
                        "all_nonzero": bool(
                            gradient["l_base_nonzero"]
                            and gradient["l_scaled_pair_nonzero"]
                            and gradient["l_total_nonzero"]
                        ),
                    }
                )

    cross_seed = {
        name: aggregate_lambda_metrics(
            [per_seed_entries[int(seed)][name] for seed in seeds]
        )
        for name in CANDIDATE_LAMBDA_NAMES
    }

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
        "scale_audit": {
            "r_seed_concat_pair_over_base": {
                str(seed): base_ratios[int(seed)] for seed in seeds
            },
            "r_median": r_median,
            "r_median_definition": "median over the three fixed seeds of the initial concatenated pair/base gradient norm ratio",
            "candidate_lambdas": lambdas,
            "target_lambda_ratios": dict(TARGET_LAMBDA_RATIOS),
            "lambda_current": LAMBDA_CURRENT,
            "per_seed": per_seed,
            "cross_seed": cross_seed,
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
            "descriptive_not_search": (
                "the four candidate scales are fixed before any scaled-gradient "
                "number is computed; this is a descriptive gradient-normalisation "
                "audit, not a hyper-parameter search, and no winner is declared"
            ),
            "not_task_accuracy": (
                "all numbers are internal train-only initial-gradient diagnostics, "
                "never CV-Bench task accuracy"
            ),
            "no_success_claim": (
                "a realised pair/base ratio equal to a diagnostic target does NOT "
                "imply convergence, validation improvement or task accuracy"
            ),
        },
    }
    return report, per_seed, lambda_csv_rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

CSV_FIELDS: tuple[str, ...] = (
    "seed",
    "lambda_name",
    "lambda_value",
    "target_ratio",
    "param",
    "base_norm",
    "scaled_pair_norm",
    "total_norm",
    "scaled_pair_over_base_norm_ratio",
    "total_over_base_norm_ratio",
    "cos_total_base",
    "cos_total_pair",
    "cos_base_pair",
    "scaled_pair_loss_over_base_loss",
    "loss_base_mean",
    "loss_pair_mean",
    "all_finite",
    "all_nonzero",
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


def render_markdown(report: Mapping[str, Any]) -> str:
    provenance = report["provenance"]
    scale = report["scale_audit"]
    integrity = report["integrity"]
    isolation = report["isolation"]
    split_counts = report["split_counts"]
    lines: list[str] = []
    lines.append(f"# Pair-loss 尺度稳定性审计（train-only，{REPORT_STEM}）")
    lines.append("")
    lines.append(
        "本轮只做**零更新、CPU-only、train-only** 的 pair-loss 尺度稳定性审计："
        "先用三 seed 初始 concat pair/base 梯度范数比求 `r_med`，再对**预先固定**的四个 "
        "`lambda` 候选逐一实际调用 `autograd.grad`，报告 scaled pair/base、total/base 比值与 "
        "base/pair/total 梯度 cosine。不训练、不调参、不保存 `Wt/Wv`、不读 val/test 特征缓存、"
        "不加载模型、不用 GPU、不联网、不设阈值、不给 promotion 结论、不宣布 winner。"
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
    lines.append("## 3. r_med 与预先固定的候选尺度")
    lines.append("")
    lines.append(
        "每个 seed 先算初始 concat `r_seed = ||grad(L_pair)|| / ||grad(L_base)||`，再取三 seed 中位数 "
        "`r_med`；四个候选 `lambda` 在任何 scaled 梯度数值之前固定，不新增其它 lambda。"
    )
    lines.append("")
    for seed, ratio in scale["r_seed_concat_pair_over_base"].items():
        lines.append(f"- seed {seed}: r_seed(concat) = {_fmt(ratio)}")
    lines.append(f"- **r_med (concat median) = {_fmt(scale['r_median'])}**")
    lines.append("")
    lines.append("| candidate | lambda | target pair/base ratio |")
    lines.append("| --- | --- | --- |")
    for name in CANDIDATE_LAMBDA_NAMES:
        target = scale["target_lambda_ratios"].get(name)
        lines.append(
            f"| {name} | {_fmt(scale['candidate_lambdas'][name])} | "
            f"{'N/A (current)' if target is None else _fmt(target)} |"
        )
    lines.append("")
    lines.append("## 4. 逐 seed / 逐 lambda 梯度诊断")
    lines.append("")
    for seed_block in scale["per_seed"]:
        lines.append(f"### seed {seed_block['seed']}")
        lines.append("")
        lines.append(
            "| lambda | lambda value | scaled_pair/base (Wt) | (Wv) | (concat) | "
            "total/base (Wt) | (Wv) | (concat) | cos(total,base) | cos(total,pair) | "
            "cos(base,pair) | scaled_pair_loss/base_loss |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for name in CANDIDATE_LAMBDA_NAMES:
            entry = seed_block["lambdas"][name]
            grad = entry["gradient"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        _fmt(entry["lambda_value"]),
                        _fmt(grad["Wt"]["scaled_pair_over_base_norm_ratio"]),
                        _fmt(grad["Wv"]["scaled_pair_over_base_norm_ratio"]),
                        _fmt(grad["concat"]["scaled_pair_over_base_norm_ratio"]),
                        _fmt(grad["Wt"]["total_over_base_norm_ratio"]),
                        _fmt(grad["Wv"]["total_over_base_norm_ratio"]),
                        _fmt(grad["concat"]["total_over_base_norm_ratio"]),
                        _fmt(grad["concat"]["cos_total_base"]),
                        _fmt(grad["concat"]["cos_total_pair"]),
                        _fmt(grad["concat"]["cos_base_pair"]),
                        _fmt(entry["scaled_pair_loss_over_base_loss"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.append("## 5. 跨 seed 聚合（mean/median/min/max/CV）")
    lines.append("")
    for name in CANDIDATE_LAMBDA_NAMES:
        agg = scale["cross_seed"][name]
        lines.append(
            f"### {name}（lambda = {_fmt(agg['lambda_value'])}，"
            f"target = {'N/A (current)' if agg['target_ratio'] is None else _fmt(agg['target_ratio'])}）"
        )
        lines.append("")
        lines.append(
            "| param | scaled_pair/base mean | median | min | max | CV | "
            "total/base mean | median | min | max | CV |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for group in PARAM_GROUPS:
            params = agg["per_param"][group]
            scaled = params["scaled_pair_over_base_norm_ratio"]
            total = params["total_over_base_norm_ratio"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        group,
                        _fmt(scaled["mean"]),
                        _fmt(scaled["median"]),
                        _fmt(scaled["min"]),
                        _fmt(scaled["max"]),
                        _fmt(scaled["cv"]),
                        _fmt(total["mean"]),
                        _fmt(total["median"]),
                        _fmt(total["min"]),
                        _fmt(total["max"]),
                        _fmt(total["cv"]),
                    ]
                )
                + " |"
            )
        concat = agg["per_param"]["concat"]
        lines.append("")
        lines.append(
            f"- concat cos(total,base) mean={_fmt(concat['cos_total_base']['mean'])}，"
            f"cos(total,pair) mean={_fmt(concat['cos_total_pair']['mean'])}，"
            f"cos(base,pair) mean={_fmt(concat['cos_base_pair']['mean'])}；"
            f"scaled_pair_loss/base_loss mean={_fmt(agg['scaled_pair_loss_over_base_loss']['mean'])}；"
            f"all_finite={agg['all_finite']}，all_nonzero={agg['all_nonzero']}。"
        )
        lines.append("")
    lines.append("## 6. 解释与限制")
    lines.append("")
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append(
        "- 本审计只描述初始点的梯度尺度与方向，不改变 DRAFT 的任何字段，也不构成重训授权。"
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

    report, _per_seed_rows, csv_rows = run_audit(
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
    write_csv(csv_path, csv_rows)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    output_hashes = {
        "csv": _sha256(csv_path),
        "json": _sha256(json_path),
        "md": _sha256(md_path),
    }
    scale = report["scale_audit"]
    key_concat = {
        name: {
            "lambda_value": scale["cross_seed"][name]["lambda_value"],
            "target_ratio": scale["cross_seed"][name]["target_ratio"],
            "scaled_pair_over_base_norm_ratio": scale["cross_seed"][name]["per_param"][
                "concat"
            ]["scaled_pair_over_base_norm_ratio"],
            "total_over_base_norm_ratio": scale["cross_seed"][name]["per_param"]["concat"][
                "total_over_base_norm_ratio"
            ],
            "cos_total_base": scale["cross_seed"][name]["per_param"]["concat"][
                "cos_total_base"
            ],
            "cos_total_pair": scale["cross_seed"][name]["per_param"]["concat"][
                "cos_total_pair"
            ],
            "cos_base_pair": scale["cross_seed"][name]["per_param"]["concat"][
                "cos_base_pair"
            ],
            "scaled_pair_loss_over_base_loss": scale["cross_seed"][name][
                "scaled_pair_loss_over_base_loss"
            ],
        }
        for name in CANDIDATE_LAMBDA_NAMES
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
            "focused_tests": "evaluation/tests/test_pair_loss_scale_audit.py",
        },
        "isolation": report["isolation"],
        "key_metrics": {
            "train_images": report["integrity"]["train_image_count"],
            "train_records": report["integrity"]["train_record_count"],
            "r_seed_concat_pair_over_base": scale["r_seed_concat_pair_over_base"],
            "r_median": scale["r_median"],
            "candidate_lambdas": scale["candidate_lambdas"],
            "key_concat_metrics": key_concat,
        },
        "limitations": list(report["limitations"]),
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
        f"PAIR_LOSS_SCALE_AUDIT status={report['status']} "
        f"images={report['integrity']['train_image_count']}"
    )
    print(f"isolation opened_only_train_npz={report['isolation']['opened_only_train_npz']}")
    print(f"r_median={scale['r_median']:.10f}")
    for name in CANDIDATE_LAMBDA_NAMES:
        print(
            f"lambda {name}={scale['candidate_lambdas'][name]:.10f} "
            f"concat_scaled_pair_over_base_mean="
            f"{scale['cross_seed'][name]['per_param']['concat']['scaled_pair_over_base_norm_ratio']['mean']:.10f} "
            f"concat_total_over_base_mean="
            f"{scale['cross_seed'][name]['per_param']['concat']['total_over_base_norm_ratio']['mean']:.10f}"
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
