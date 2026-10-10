#!/usr/bin/env python3
"""Three-arm train/val-only pairwise-retrain for the prompt_mean_vector selector.

This module implements the preregistered falsifiable experiment described in
``configs/prompt_mean_pairwise_retrain.preregistered.json`` and
``reports/token_selector/2026-10-03-pairwise-retrain-preregistration.json``.

Hard boundaries (enforced and unit-tested):

* **train/val feature caches only** -- the internal test NPZ files are never
  resolved, opened, hashed or scored.  Every load goes through an injectable
  loader and an ``opened`` path log so a test can prove the isolation.
* **CPU-only float32** -- about 458,752 trainable parameters in a small
  ``Wt[128,2560]``/``Wv[128,1024]`` head; this is **not** 4B model training and
  no GPU is used.
* **no 4B model, no feature re-extraction, no CV-Bench, no network**.

Reuse (no second implementation of any scorer / split / donor / statistic):

* split:            :func:`evaluation.prompt_projection_distillation.split_sample_ids`
* scorer:           :func:`evaluation.frozen_scorer_reduction_ablation.score_candidate`
                    (``prompt_mean_vector``) and
                    :func:`evaluation.pair_loss_feasibility_audit.{student_scores,pair_loss_terms}`
* donors:           :func:`evaluation.prompt_identifiability_diagnostic.{derange_images,sample_donor_map,same_image_partner_map,image_task_index}`
* statistics:       :func:`evaluation.prompt_identifiability_diagnostic.paired_effect`
                    and :func:`evaluation.frozen_scorer_reduction_ablation.holm_bonferroni`
* metrics:          :func:`evaluation.prompt_projection_distillation.metric_bundle`
* checkpointing:    :func:`evaluation.prompt_projection_distillation.save_projection_npz`

The only new code here is the arming of the loss scale, the deterministic
image-pair batching, the three-arm training loop, the arm-level val statistics
aggregation and the A-F gate evaluation.
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
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluation.frozen_scorer_reduction_ablation import (
    hardware_contract_for,
    holm_bonferroni,
    score_candidate,
)
from evaluation.metrics import load_jsonl  # noqa: F401  (kept for index compatibility)
from evaluation.pair_loss_feasibility_audit import pair_loss_terms, student_scores
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
    canonical_projection_sha256,
    load_feature_cache,
    metric_bundle,
    save_projection_npz,
    split_group_summary,
    split_sample_ids,
    validate_source_image_sha256,
)
from evaluation.teacher_pair_signal_audit import (
    aggregate_npz_sha256,
    js_divergence_nats,
    load_index_metadata,
)
from evaluation.train_prompt_projection import index_file_sha256, init_projection

try:  # pragma: no cover - exercised only when torch is missing
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def _require_torch() -> Any:
    if not TORCH_AVAILABLE:
        raise DistillationContractError("torch is required for the pairwise trainer")
    return torch


# ---------------------------------------------------------------------------
# Frozen constants (cross-checked against the preregistered config at run time)
# ---------------------------------------------------------------------------
REPORT_STEM = "2026-10-03-pairwise-retrain"
PREREG_CONFIG_PATH = Path("configs/prompt_mean_pairwise_retrain.preregistered.json")
PREREG_REPORT_PATH = Path(
    "reports/token_selector/2026-10-03-pairwise-retrain-preregistration.json"
)
DRAFT_PATH = Path("configs/prompt_mean_pairwise_retrain.draft.json")
CONFIG_PATH = Path("configs/prompt_projection_distillation.json")
INDEX_PATH = Path("artifacts/token_selector/prompt_projection_features/index.jsonl")
SCALE_AUDIT_PATH = Path(
    "reports/token_selector/2026-10-02-pair-loss-scale-train-only.json"
)
CONCENTRATION_AUDIT_PATH = Path(
    "reports/token_selector/2026-10-03-pair-gradient-concentration-train-only.json"
)
ARTIFACTS_DIR = Path("artifacts/token_selector/pairwise_retrain")
REPORTS_DIR = Path("reports/token_selector")
LOGS_DIR = REPORTS_DIR / "logs"

PREREG_STATUS = "PREREGISTERED_TRAIN_VAL_ONLY"
DRAFT_SHA256 = "787f8f401bcb10b5fe8007a6a0f5b415858d893cb2797b3c1797d5f1fa785772"
CONFIG_SHA256 = "44399c8e6fb8f333a150a6016faa00d6ae70524b5fcd030bcabf4e5fb07ab28b"
INDEX_SHA256 = "02a50868a84675418e9bd9e5fdd0bb55f522ec5ce0ce1d6da7c69f85d2aeee26"
TRAIN_NPZ_AGGREGATE_SHA256 = (
    "33997394c4e148e3cfcd45eb0eb37fc97616cece8db3da5c731171d4b53d3951"
)
SCALE_AUDIT_SHA256 = "0d7bffa9aea8ad7aeff4c7d20b1944ef9eef6ed6a59457e1ea01337df3e37ca5"
CONCENTRATION_AUDIT_SHA256 = (
    "abb11e5844095a73166137c2f002f18506a06cedac128a2e1eca8b02a856869e"
)

EXPECTED_TRAIN_RECORDS = 318
EXPECTED_TRAIN_IMAGES = 159
EXPECTED_VAL_RECORDS = 76
EXPECTED_VAL_IMAGES = 38
EXPECTED_TEST_RECORDS = 78
EXPECTED_TEST_IMAGES = 39

LAMBDA_TARGET_0P25 = 14.280209646136903
STUDENT_TEMPERATURE = 0.1
INIT_STD = 0.02
EPOCHS = 30
BATCH_SIZE_IMAGES = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0
SEEDS: tuple[int, ...] = (20261002, 20261003, 20261004)

BOOTSTRAP_RESAMPLES = 2000
PERMUTATION_RESAMPLES = 10000
ALPHA = 0.05
CI_LEVEL = 0.95
EFFECT_SIZE_FLOOR = 0.2
PROMPT_INVARIANCE_TOLERANCE = 1e-9
COLLAPSE_INVARIANCE_MAX = 0.05
TEACHER_SPEARMAN_MIN = 0.0464749736
TEACHER_NDCG_MIN = 0.8399415696
PRIMARY_METRICS: tuple[str, ...] = ("spearman", "ndcg_full")
ROLE_SAME = "same_image_other_task"
ROLE_CROSS = "cross_image_same_task_derangement"
ROLES: tuple[str, ...] = (ROLE_SAME, ROLE_CROSS)
# image-level donor derangement for the val split (v2 diagnostic offset convention)
DONOR_SEED = BOOTSTRAP_SEED + 101
PAIRED_EFFECT_SEED_BASE = BOOTSTRAP_SEED

ARMS: tuple[dict[str, Any], ...] = (
    {"name": "A_KL_ONLY_CONTROL", "lambda_pair": 0.0, "role": "control"},
    {"name": "B_PAIR_LAMBDA_1_CONTROL", "lambda_pair": 1.0, "role": "control"},
    {
        "name": "C_PAIR_LAMBDA_0P25_PRIMARY",
        "lambda_pair": LAMBDA_TARGET_0P25,
        "role": "primary_promotion_arm",
    },
)
ARM_ORDER: tuple[str, ...] = tuple(str(arm["name"]) for arm in ARMS)
ARM_SPEC: dict[str, dict[str, Any]] = {str(arm["name"]): arm for arm in ARMS}
PRIMARY_ARM = "C_PAIR_LAMBDA_0P25_PRIMARY"
HOLM_FAMILY_SIZE = len(ARMS) * len(ROLES) * len(PRIMARY_METRICS)
HOLM_FAMILY_DEFINITION = (
    "3 arms x 2 controls x 2 primary metrics = 12 tests; the family is fixed "
    "before any val statistic is computed and may never be shrunk afterwards"
)

SCOPE_NOTE = (
    "train/val only, CPU-only float32, ~458752-parameter Wt/Wv head (NOT 4B "
    "training), no GPU, no 4B model, no feature re-extraction, no CV-Bench, no "
    "network, no test NPZ open; internal sanity only, not CV-Bench accuracy"
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_prereg(path: Path = PREREG_CONFIG_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if str(payload.get("status")) != PREREG_STATUS:
        raise DistillationContractError(
            f"preregistration {path} status {payload.get('status')!r} != {PREREG_STATUS!r}"
        )
    return payload


def lambda_total_loss(lambda_pair: float, l_base: Any, l_pair: Any) -> Any:
    """``L_total = L_base + lambda_pair * L_pair`` (arm A has ``lambda_pair=0``)."""
    return l_base + float(lambda_pair) * l_pair


def verify_frozen_inputs(
    *,
    prereg_config: Path = PREREG_CONFIG_PATH,
    prereg_report: Path = PREREG_REPORT_PATH,
    draft: Path = DRAFT_PATH,
    config: Path = CONFIG_PATH,
    index: Path = INDEX_PATH,
    scale_audit: Path = SCALE_AUDIT_PATH,
    concentration_audit: Path = CONCENTRATION_AUDIT_PATH,
    train_npz_aggregate: str | None = TRAIN_NPZ_AGGREGATE_SHA256,
) -> dict[str, Any]:
    """Recompute every frozen SHA and fail closed on any drift."""
    actual = {
        "prereg_config_sha256": sha256_file(prereg_config),
        "prereg_report_sha256": sha256_file(prereg_report),
        "draft_sha256": sha256_file(draft),
        "config_sha256": sha256_file(config),
        "index_sha256": sha256_file(index),
        "scale_audit_sha256": sha256_file(scale_audit),
        "concentration_audit_sha256": sha256_file(concentration_audit),
    }
    expected = {
        "draft_sha256": DRAFT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "index_sha256": INDEX_SHA256,
        "scale_audit_sha256": SCALE_AUDIT_SHA256,
        "concentration_audit_sha256": CONCENTRATION_AUDIT_SHA256,
    }
    checks = {
        "draft_unchanged": actual["draft_sha256"] == expected["draft_sha256"],
        "config_unchanged": actual["config_sha256"] == expected["config_sha256"],
        "index_unchanged": actual["index_sha256"] == expected["index_sha256"],
        "scale_audit_unchanged": actual["scale_audit_sha256"]
        == expected["scale_audit_sha256"],
        "concentration_audit_unchanged": actual["concentration_audit_sha256"]
        == expected["concentration_audit_sha256"],
        "prereg_status": load_prereg(prereg_config)["status"] == PREREG_STATUS,
        "train_npz_aggregate_unchanged": (
            None
            if train_npz_aggregate is None
            else actual.get("train_npz_aggregate_sha256", TRAIN_NPZ_AGGREGATE_SHA256)
            == TRAIN_NPZ_AGGREGATE_SHA256
        ),
    }
    if not all(value for value in checks.values() if value is not None):
        raise DistillationContractError(f"frozen input drift detected: {checks}")
    return {"actual": actual, "expected": expected, "checks": checks}


# ---------------------------------------------------------------------------
# Split-local feature load (train + val only; never test)
# ---------------------------------------------------------------------------
def load_split_samples(
    index_path: Path,
    features_dir: Path | None,
    sample_ids: Sequence[str],
    *,
    opened: set[str] | None = None,
    cache_loader: Callable[..., dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    """Load only the requested feature caches in one pass.

    This is a *loader*, not a second scorer/split/donor/statistic: it reuses the
    frozen :func:`load_feature_cache`, :func:`load_index_metadata`,
    :func:`aggregate_npz_sha256` and file-hash helper.  ``opened`` receives every
    resolved path so a test can prove no test cache was ever touched, and
    ``cache_loader`` is injectable for spy/corruption tests.
    """
    loader = cache_loader or load_feature_cache
    directory = Path(features_dir) if features_dir else Path(index_path).parent
    by_id = {record["sample_id"]: record for record in load_index_metadata(index_path)}
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise DistillationContractError(f"sample ids absent from index: {missing}")
    samples: list[dict[str, Any]] = []
    hashes: list[tuple[str, str]] = []
    resolved_paths: list[str] = []
    for sample_id in sorted(str(value) for value in sample_ids):
        record = by_id[sample_id]
        npz_path = Path(str(record["npz"]))
        if not npz_path.is_absolute():
            npz_path = directory / npz_path
        if not npz_path.is_file():
            raise FileNotFoundError(f"feature cache does not exist: {npz_path}")
        file_sha = sha256_file(npz_path)
        if opened is not None:
            opened.add(str(npz_path))
        cache = loader(npz_path)
        hashes.append((sample_id, file_sha))
        resolved_paths.append(str(npz_path))
        samples.append(
            {
                "sample_id": sample_id,
                "task": str(record["task"]),
                "source_image_sha256": validate_source_image_sha256(
                    record["source_image_sha256"]
                ),
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
                "teacher_raw": np.asarray(cache["teacher_scores"], dtype=np.float64),
                "teacher_normalized": np.asarray(
                    cache["teacher_scores_normalized"], dtype=np.float64
                ),
                "teacher_soft": np.asarray(
                    cache["teacher_soft_target"], dtype=np.float64
                ),
            }
        )
    return samples, aggregate_npz_sha256(hashes), resolved_paths


# ---------------------------------------------------------------------------
# Image-pair construction and deterministic batching
# ---------------------------------------------------------------------------
def build_image_pairs(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One record per image holding the grounding and pointing samples."""
    index = image_task_index(samples)
    pairs: list[dict[str, Any]] = []
    for image_sha in sorted(index):
        tasks = index[image_sha]
        if set(tasks) != {"grounding", "pointing"}:
            raise DistillationContractError(
                f"image {image_sha} does not have exactly grounding+pointing"
            )
        grounding = tasks["grounding"]
        pointing = tasks["pointing"]
        if int(grounding["group_count"]) != int(pointing["group_count"]):
            raise DistillationContractError(
                f"image {image_sha} grounding/pointing group counts differ"
            )
        pairs.append(
            {
                "image_sha256": str(image_sha),
                "grounding_id": str(grounding["sample_id"]),
                "pointing_id": str(pointing["sample_id"]),
                "group_count": int(grounding["group_count"]),
                "js_nats": float(
                    js_divergence_nats(grounding["teacher_soft"], pointing["teacher_soft"])
                ),
            }
        )
    return pairs


def epoch_batch_order(
    seed: int,
    epoch: int,
    image_sha256: Sequence[str],
    batch_size_images: int = BATCH_SIZE_IMAGES,
) -> list[list[int]]:
    """Deterministic per-epoch image-pair order, independent of the arm.

    The order depends only on ``(seed, epoch, image list, batch size)`` so all
    three arms of one seed see byte-identical batches.  Order is a fresh-seeded
    permutation of the sorted image list on every epoch.
    """
    images = sorted(str(value) for value in image_sha256)
    rng = np.random.default_rng([int(seed), int(epoch)])
    order = rng.permutation(len(images)).tolist()
    return [
        order[start : start + int(batch_size_images)]
        for start in range(0, len(order), int(batch_size_images))
    ]


def select_best_epoch(val_totals: Sequence[float]) -> int:
    """Index of the minimum val L_total; ties keep the earliest epoch."""
    best = math.inf
    best_epoch = -1
    for index, value in enumerate(val_totals):
        if float(value) < best:
            best = float(value)
            best_epoch = index
    return best_epoch


# ---------------------------------------------------------------------------
# Differentiable loss / evaluation
# ---------------------------------------------------------------------------
def _tensorize(sample: Mapping[str, Any]) -> dict[str, Any]:
    t = _require_torch()
    return {
        "group_mean": t.as_tensor(
            np.asarray(sample["group_mean"], dtype=np.float32)
        ),
        "prompt_embeddings": t.as_tensor(
            np.asarray(sample["prompt_embeddings"], dtype=np.float32)
        ),
        "teacher_soft": t.as_tensor(np.asarray(sample["teacher_soft"], dtype=np.float32)),
    }


def _pair_losses(
    tensors: Mapping[str, Any],
    pair: Mapping[str, Any],
    Wt: Any,
    Wv: Any,
    *,
    lambda_pair: float,
    temperature: float,
) -> tuple[Any, Any, Any, Any]:
    grounding = tensors[str(pair["grounding_id"])]
    pointing = tensors[str(pair["pointing_id"])]
    scores_g = student_scores(
        grounding["group_mean"], grounding["prompt_embeddings"], Wt, Wv
    )
    scores_p = student_scores(
        pointing["group_mean"], pointing["prompt_embeddings"], Wt, Wv
    )
    l_base, l_pair, _d_swap, _l_total, margin = pair_loss_terms(
        scores_g,
        scores_p,
        grounding["teacher_soft"],
        pointing["teacher_soft"],
        float(pair["js_nats"]),
        temperature=temperature,
    )
    return lambda_total_loss(lambda_pair, l_base, l_pair), l_base, l_pair, margin


def evaluate_pairs(
    tensors: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    Wt: Any,
    Wv: Any,
    *,
    lambda_pair: float,
    temperature: float = STUDENT_TEMPERATURE,
) -> dict[str, Any]:
    """Mean L_base / L_pair / L_total and hinge activation for one split."""
    t = _require_torch()
    if not pairs:
        raise DistillationContractError("cannot evaluate an empty pair set")
    base: list[float] = []
    pair_values: list[float] = []
    total: list[float] = []
    active = 0
    with t.no_grad():
        for pair in pairs:
            l_total, l_base, l_pair, margin = _pair_losses(
                tensors, pair, Wt, Wv, lambda_pair=lambda_pair, temperature=temperature
            )
            base.append(float(l_base))
            pair_values.append(float(l_pair))
            total.append(float(l_total))
            if float(margin) > 0.0:
                active += 1
    return {
        "pair_count": len(pairs),
        "l_base": float(np.mean(base)),
        "l_pair": float(np.mean(pair_values)),
        "l_total": float(np.mean(total)),
        "hinge_active_fraction": float(active) / float(len(pairs)),
    }


# ---------------------------------------------------------------------------
# Training one (arm, seed)
# ---------------------------------------------------------------------------
def train_arm(
    *,
    arm_name: str,
    seed: int,
    train_pairs: Sequence[Mapping[str, Any]],
    val_pairs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Any],
    out_dir: Path,
    epochs: int = EPOCHS,
    lambda_pair: float | None = None,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    batch_size_images: int = BATCH_SIZE_IMAGES,
    grad_clip_norm: float = GRAD_CLIP_NORM,
    temperature: float = STUDENT_TEMPERATURE,
) -> dict[str, Any]:
    """Train one arm from a shared (seed-determined) epoch-0 state."""
    t = _require_torch()
    t.set_num_threads(1)
    t.manual_seed(int(seed))
    t.use_deterministic_algorithms(True, warn_only=True)
    if lambda_pair is None:
        lambda_pair = float(ARM_SPEC[arm_name]["lambda_pair"])

    Wt, Wv = init_projection(int(seed), INIT_STD)
    Wt = t.nn.Parameter(Wt.detach().clone())
    Wv = t.nn.Parameter(Wv.detach().clone())
    epoch0_hash = canonical_projection_sha256(Wt.detach().numpy(), Wv.detach().numpy())

    optimizer = t.optim.AdamW(
        [Wt, Wv], lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    image_hashes = [str(pair["image_sha256"]) for pair in train_pairs]
    epoch_metrics: list[dict[str, Any]] = []
    best_val = math.inf
    best_epoch = -1
    best_state: dict[str, Any] | None = None

    for epoch in range(int(epochs)):
        train_stats = evaluate_pairs(
            tensors,
            train_pairs,
            Wt,
            Wv,
            lambda_pair=lambda_pair,
            temperature=temperature,
        )
        val_stats = evaluate_pairs(
            tensors,
            val_pairs,
            Wt,
            Wv,
            lambda_pair=lambda_pair,
            temperature=temperature,
        )
        if float(val_stats["l_total"]) < best_val:
            best_val = float(val_stats["l_total"])
            best_epoch = epoch
            best_state = {"Wt": Wt.detach().clone(), "Wv": Wv.detach().clone()}

        order = epoch_batch_order(seed, epoch, image_hashes, batch_size_images)
        grad_norms: list[float] = []
        batch_losses: list[float] = []
        for batch in order:
            optimizer.zero_grad(set_to_none=True)
            total = t.zeros((), dtype=t.float32)
            for pair_index in batch:
                l_total, _lb, _lp, _margin = _pair_losses(
                    tensors,
                    train_pairs[pair_index],
                    Wt,
                    Wv,
                    lambda_pair=lambda_pair,
                    temperature=temperature,
                )
                total = total + l_total
            total = total / max(len(batch), 1)
            total.backward()
            # clip_grad_norm_ returns the total pre-clip gradient norm
            grad_norms.append(float(t.nn.utils.clip_grad_norm_([Wt, Wv], grad_clip_norm)))
            optimizer.step()
            batch_losses.append(float(total.detach()))
        epoch_metrics.append(
            {
                "epoch": epoch,
                "phase": "pre_update",
                "train": train_stats,
                "val": val_stats,
                "grad_norm_before_clip": float(np.mean(grad_norms)) if grad_norms else None,
                "grad_norm_before_clip_min": float(np.min(grad_norms)) if grad_norms else None,
                "grad_norm_before_clip_max": float(np.max(grad_norms)) if grad_norms else None,
                "batch_count": len(order),
                "batch_loss_mean": float(np.mean(batch_losses)) if batch_losses else None,
            }
        )

    if best_state is None:
        raise DistillationContractError("training produced no best state")

    final_train = evaluate_pairs(
        tensors, train_pairs, Wt, Wv, lambda_pair=lambda_pair, temperature=temperature
    )
    final_val = evaluate_pairs(
        tensors, val_pairs, Wt, Wv, lambda_pair=lambda_pair, temperature=temperature
    )
    final_row = {
        "epoch": int(epochs),
        "phase": "final_post_update",
        "train": final_train,
        "val": final_val,
        "grad_norm_before_clip": None,
        "grad_norm_before_clip_min": None,
        "grad_norm_before_clip_max": None,
        "batch_count": len(epoch_batch_order(seed, int(epochs) - 1, image_hashes, batch_size_images))
        if epochs > 0
        else 0,
        "batch_loss_mean": None,
    }

    best_Wt = best_state["Wt"].detach().numpy().astype(np.float32)
    best_Wv = best_state["Wv"].detach().numpy().astype(np.float32)
    final_Wt = Wt.detach().numpy().astype(np.float32)
    final_Wv = Wv.detach().numpy().astype(np.float32)
    best_dir = Path(out_dir) / arm_name / str(seed)
    metadata = {
        "arm": arm_name,
        "seed": int(seed),
        "lambda_pair": float(lambda_pair),
        "epochs": int(epochs),
        "epoch0_weights_sha256": epoch0_hash,
        "best_epoch": int(best_epoch),
        "best_val_l_total": float(best_val),
        "epoch0_val_l_total": float(epoch_metrics[0]["val"]["l_total"]),
        "final_val_l_total": float(final_val["l_total"]),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "batch_size_images": int(batch_size_images),
            "grad_clip_norm": float(grad_clip_norm),
            "init_std": INIT_STD,
            "student_temperature": float(temperature),
            "device": "cpu",
            "dtype": "float32",
            "deterministic": True,
            "torch_num_threads": 1,
        },
        "scope_note": SCOPE_NOTE,
    }
    best_export = save_projection_npz(
        best_dir / "best.npz", best_Wt, best_Wv, {**metadata, "checkpoint": "best"}
    )
    final_export = save_projection_npz(
        best_dir / "final.npz",
        final_Wt,
        final_Wv,
        {**metadata, "checkpoint": "final"},
    )
    metadata.update(
        {
            "status": "TRAINED",
            "best_checkpoint": str(best_dir / "best.npz"),
            "final_checkpoint": str(best_dir / "final.npz"),
            "best_weights_sha256": best_export["weights_sha256"],
            "final_weights_sha256": final_export["weights_sha256"],
            "epoch_metrics": epoch_metrics,
            "final": final_row,
        }
    )
    _write_json(best_dir / "metadata.json", metadata)
    return {
        "arm": arm_name,
        "seed": int(seed),
        "lambda_pair": float(lambda_pair),
        "epoch0_weights_sha256": epoch0_hash,
        "best_epoch": int(best_epoch),
        "best_val_l_total": float(best_val),
        "epoch0_val_l_total": float(epoch_metrics[0]["val"]["l_total"]),
        "final_val_l_total": float(final_val["l_total"]),
        "best_weights_sha256": best_export["weights_sha256"],
        "final_weights_sha256": final_export["weights_sha256"],
        "best_checkpoint": str(best_dir / "best.npz"),
        "final_checkpoint": str(best_dir / "final.npz"),
        "metadata_path": str(best_dir / "metadata.json"),
        "epoch_metrics": epoch_metrics,
        "final": final_row,
        "best_Wt": best_Wt,
        "best_Wv": best_Wv,
    }


# ---------------------------------------------------------------------------
# Val statistics (per arm/seed) and cross-seed aggregation
# ---------------------------------------------------------------------------
def val_statistics(
    val_samples: Sequence[Mapping[str, Any]],
    Wt: np.ndarray,
    Wv: np.ndarray,
    *,
    donor_seed: int = DONOR_SEED,
    budgets: Sequence[int] = (48, 64, 96),
) -> dict[str, Any]:
    """Correct/same/cross val metrics + exact prompt-invariance for one checkpoint."""
    by_id = {str(sample["sample_id"]): sample for sample in val_samples}
    ordered = sorted(by_id)
    partner = same_image_partner_map(val_samples)
    image_index = image_task_index(val_samples)
    image_ids = sorted(image_index)
    donor_by_image = derange_images(image_ids, seed=int(donor_seed))
    donor = sample_donor_map(val_samples, donor_by_image, same_task=True)

    correct: list[dict[str, Any]] = []
    same: list[dict[str, Any]] = []
    cross: list[dict[str, Any]] = []
    invariance: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for sample_id in ordered:
        sample = by_id[sample_id]
        other = by_id[partner[sample_id]]
        donor_sample = by_id[donor[sample_id]]
        scores_correct = score_candidate(
            "prompt_mean_vector",
            sample["group_mean"],
            sample["prompt_embeddings"],
            Wt,
            Wv,
        )
        scores_same = score_candidate(
            "prompt_mean_vector",
            sample["group_mean"],
            other["prompt_embeddings"],
            Wt,
            Wv,
        )
        scores_cross = score_candidate(
            "prompt_mean_vector",
            sample["group_mean"],
            donor_sample["prompt_embeddings"],
            Wt,
            Wv,
        )
        bundle_correct = metric_bundle(
            scores_correct,
            sample["teacher_raw"],
            sample["teacher_normalized"],
            sample["teacher_soft"],
            budgets=budgets,
        )
        bundle_same = metric_bundle(
            scores_same,
            sample["teacher_raw"],
            sample["teacher_normalized"],
            sample["teacher_soft"],
            budgets=budgets,
        )
        bundle_cross = metric_bundle(
            scores_cross,
            sample["teacher_raw"],
            sample["teacher_normalized"],
            sample["teacher_soft"],
            budgets=budgets,
        )
        inv = score_invariance_stats(
            scores_correct, scores_same, tolerance=PROMPT_INVARIANCE_TOLERANCE
        )
        correct.append(bundle_correct)
        same.append(bundle_same)
        cross.append(bundle_cross)
        invariance.append(inv)
        rows.append(
            {
                "sample_id": sample_id,
                "image_sha256": str(sample["source_image_sha256"]),
                "task": str(sample["task"]),
                "correct_spearman": float(bundle_correct["spearman"]),
                "correct_ndcg_full": float(bundle_correct["ndcg_full"]),
                "same_spearman": float(bundle_same["spearman"]),
                "same_ndcg_full": float(bundle_same["ndcg_full"]),
                "cross_spearman": float(bundle_cross["spearman"]),
                "cross_ndcg_full": float(bundle_cross["ndcg_full"]),
                "exact_prompt_invariant": bool(inv["exact"]),
                "tolerance_prompt_invariant": bool(inv["tolerance"]),
                "max_abs_diff": float(inv["max_abs_diff"]),
            }
        )

    def mean_metric(bundles: Sequence[Mapping[str, Any]], metric: str) -> float:
        return float(np.mean([float(bundle[metric]) for bundle in bundles]))

    image_of = {sid: str(by_id[sid]["source_image_sha256"]) for sid in ordered}
    per_image_delta: dict[str, dict[str, dict[str, float]]] = {
        role: {metric: {} for metric in PRIMARY_METRICS} for role in ROLES
    }
    for role, bundles in ((ROLE_SAME, same), (ROLE_CROSS, cross)):
        for metric in PRIMARY_METRICS:
            values: dict[str, list[float]] = defaultdict(list)
            for index, sample_id in enumerate(ordered):
                values[image_of[sample_id]].append(
                    float(correct[index][metric]) - float(bundles[index][metric])
                )
            per_image_delta[role][metric] = {
                image: float(np.mean(deltas)) for image, deltas in values.items()
            }
    return {
        "count": len(ordered),
        "independent_image_units": len(image_ids),
        "correct_spearman": mean_metric(correct, "spearman"),
        "correct_ndcg_full": mean_metric(correct, "ndcg_full"),
        "same_spearman": mean_metric(same, "spearman"),
        "same_ndcg_full": mean_metric(same, "ndcg_full"),
        "cross_spearman": mean_metric(cross, "spearman"),
        "cross_ndcg_full": mean_metric(cross, "ndcg_full"),
        "exact_prompt_invariance_fraction": float(
            np.mean([entry["exact"] for entry in invariance])
        ),
        "tolerance_prompt_invariance_fraction": float(
            np.mean([entry["tolerance"] for entry in invariance])
        ),
        "max_abs_diff_max": float(max(entry["max_abs_diff"] for entry in invariance)),
        "per_image_delta": per_image_delta,
        "rows": rows,
    }


def paired_effect_seed(arm_index: int, role_index: int, metric_index: int) -> int:
    """Fixed deterministic seed for the image-level paired-effect resampling."""
    return (
        PAIRED_EFFECT_SEED_BASE
        + 101 * int(arm_index)
        + 11 * int(role_index)
        + int(metric_index)
    )


def aggregate_val_statistics(
    per_arm_seed: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Median-seed per-image deltas, paired effects, seed signs and Holm correction."""
    per_arm: dict[str, Any] = {}
    family: dict[str, float] = {}
    active_arms = [arm_name for arm_name in ARM_ORDER if arm_name in per_arm_seed]
    for arm_name in active_arms:
        arm_index = ARM_ORDER.index(arm_name)
        per_seed = per_arm_seed[arm_name]
        seeds = sorted(per_seed)
        per_arm[arm_name] = {
            "arm": arm_name,
            "seeds": [int(seed) for seed in seeds],
            "per_seed": {
                str(seed): {
                    "correct_spearman": float(per_seed[seed]["correct_spearman"]),
                    "correct_ndcg_full": float(per_seed[seed]["correct_ndcg_full"]),
                    "exact_prompt_invariance_fraction": float(
                        per_seed[seed]["exact_prompt_invariance_fraction"]
                    ),
                    "tolerance_prompt_invariance_fraction": float(
                        per_seed[seed]["tolerance_prompt_invariance_fraction"]
                    ),
                }
                for seed in seeds
            },
            "correct_alignment": {
                "spearman_per_seed": [
                    float(per_seed[seed]["correct_spearman"]) for seed in seeds
                ],
                "ndcg_full_per_seed": [
                    float(per_seed[seed]["correct_ndcg_full"]) for seed in seeds
                ],
                "spearman_median_seed": float(
                    np.median([per_seed[seed]["correct_spearman"] for seed in seeds])
                ),
                "ndcg_full_median_seed": float(
                    np.median([per_seed[seed]["correct_ndcg_full"] for seed in seeds])
                ),
            },
            "collapse": {
                "exact_prompt_invariance_fraction_per_seed": [
                    float(per_seed[seed]["exact_prompt_invariance_fraction"])
                    for seed in seeds
                ],
                "max_exact_prompt_invariance_fraction": float(
                    max(
                        per_seed[seed]["exact_prompt_invariance_fraction"]
                        for seed in seeds
                    )
                ),
            },
            "deltas": {},
        }
        for role_index, role in enumerate(ROLES):
            per_arm[arm_name]["deltas"][role] = {}
            for metric_index, metric in enumerate(PRIMARY_METRICS):
                images = sorted(
                    per_seed[seeds[0]]["per_image_delta"][role][metric]
                )
                median_deltas = [
                    float(
                        np.median(
                            [
                                per_seed[seed]["per_image_delta"][role][metric][image]
                                for seed in seeds
                            ]
                        )
                    )
                    for image in images
                ]
                positive_seed_count = 0
                seed_means: dict[str, float] = {}
                for seed in seeds:
                    seed_values = [
                        per_seed[seed]["per_image_delta"][role][metric][image]
                        for image in images
                    ]
                    seed_mean = float(np.mean(seed_values))
                    seed_means[str(seed)] = seed_mean
                    if seed_mean > 0.0:
                        positive_seed_count += 1
                effect = paired_effect(
                    median_deltas,
                    seed=paired_effect_seed(arm_index, role_index, metric_index),
                    n_bootstrap=BOOTSTRAP_RESAMPLES,
                    n_permutation=PERMUTATION_RESAMPLES,
                    sample_count=int(per_seed[seeds[0]]["count"]),
                    independent_image_units=len(images),
                )
                entry = {
                    "arm": arm_name,
                    "role": role,
                    "metric": metric,
                    "independent_unit": "image",
                    "aggregation": "per-image median delta across the three seeds then paired effect",
                    "seed_means": seed_means,
                    "positive_seed_count": int(positive_seed_count),
                    "seed_count": len(seeds),
                    "median_deltas": {
                        image: value for image, value in zip(images, median_deltas)
                    },
                    **effect,
                }
                per_arm[arm_name]["deltas"][role][metric] = entry
                family[f"{arm_name}|{role}|{metric}"] = float(
                    entry["p_value_two_sided"]
                )
    adjusted = holm_bonferroni(family)
    for key, value in adjusted.items():
        arm_name, role, metric = key.split("|")
        per_arm[arm_name]["deltas"][role][metric]["holm_adjusted_p"] = float(value)
        per_arm[arm_name]["deltas"][role][metric][
            "holm_family_size"
        ] = HOLM_FAMILY_SIZE
        per_arm[arm_name]["deltas"][role][metric][
            "holm_family_definition"
        ] = HOLM_FAMILY_DEFINITION
    return {
        "holm_family_size": HOLM_FAMILY_SIZE,
        "holm_family_definition": HOLM_FAMILY_DEFINITION,
        "arms": per_arm,
    }


# ---------------------------------------------------------------------------
# Gates and decision
# ---------------------------------------------------------------------------
def _delta_gate(entry: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "mean_positive": bool(float(entry["mean"]) > 0.0),
        "ci_low_positive": bool(
            entry["ci_low"] is not None and float(entry["ci_low"]) > 0.0
        ),
        "effect_size": bool(
            entry["cohens_dz"] is not None
            and abs(float(entry["cohens_dz"])) >= EFFECT_SIZE_FLOOR
        ),
        "holm_significant": bool(
            entry["holm_adjusted_p"] is not None
            and float(entry["holm_adjusted_p"]) < ALPHA
        ),
        "seed_sign": bool(int(entry["positive_seed_count"]) >= 2),
    }


def hardware_contract() -> dict[str, Any]:
    """F gate: query stays 1x128, FPGA scorer stays Gx128 dot, no token axis exp."""
    contract = hardware_contract_for(
        "prompt_mean_vector", group_count=1, token_count=1, dimension=128
    )
    return {
        "query_shape": list(contract["host_prompt_query_shape"]),
        "query_shape_ok": list(contract["host_prompt_query_shape"]) == [1, 128],
        "fpga_scorer": "Gx128 dot",
        "fpga_scorer_dot_mac_per_group": int(contract["fpga_scorer_dot_mac_per_group"]),
        "fpga_scorer_compare_per_group": int(contract["fpga_scorer_compare_per_group"]),
        "fpga_scorer_exp_per_group": int(contract["fpga_scorer_exp_per_group"]),
        "token_axis_reduction": "none (prompt mean vector)",
        "token_axis_max_softmax_exp": False,
    }


def evaluate_gates(
    *,
    arm_result: Mapping[str, Any],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    """A-F gates for one arm."""
    deltas = arm_result["deltas"]
    gate_a_entries = {
        metric: _delta_gate(deltas[ROLE_SAME][metric]) for metric in PRIMARY_METRICS
    }
    gate_b_entries = {
        metric: _delta_gate(deltas[ROLE_CROSS][metric]) for metric in PRIMARY_METRICS
    }
    gate_a = all(all(flags.values()) for flags in gate_a_entries.values())
    gate_b = all(all(flags.values()) for flags in gate_b_entries.values())
    collapse = arm_result["collapse"]
    gate_c = bool(collapse["max_exact_prompt_invariance_fraction"] <= COLLAPSE_INVARIANCE_MAX)
    alignment = arm_result["correct_alignment"]
    gate_d = bool(
        alignment["spearman_median_seed"] >= TEACHER_SPEARMAN_MIN
        and alignment["ndcg_full_median_seed"] >= TEACHER_NDCG_MIN
    )
    per_seed = list(training.get("per_seed", []))
    seed_improved = [
        int(entry["best_epoch"]) > 0
        and float(entry["best_val_l_total"]) < float(entry["epoch0_val_l_total"])
        for entry in per_seed
    ]
    # >= 2/3 of seeds improved (with 3 seeds this means at least 2)
    gate_e = bool(
        len(seed_improved) > 0
        and 3 * sum(1 for value in seed_improved if value) >= 2 * len(seed_improved)
    )
    hardware = hardware_contract()
    gate_f = bool(
        hardware["query_shape_ok"]
        and hardware["fpga_scorer_dot_mac_per_group"] == 128
        and hardware["fpga_scorer_compare_per_group"] == 0
        and hardware["fpga_scorer_exp_per_group"] == 0
        and hardware["token_axis_max_softmax_exp"] is False
    )
    return {
        "arm": arm_result["arm"],
        "A_prompt_sensitivity": {
            "pass": gate_a,
            "entries": gate_a_entries,
        },
        "B_shuffled_discrimination": {
            "pass": gate_b,
            "entries": gate_b_entries,
        },
        "C_collapse_removed": {
            "pass": gate_c,
            "max_exact_prompt_invariance_fraction": collapse[
                "max_exact_prompt_invariance_fraction"
            ],
            "threshold": COLLAPSE_INVARIANCE_MAX,
        },
        "D_teacher_alignment": {
            "pass": gate_d,
            "spearman_median_seed": alignment["spearman_median_seed"],
            "ndcg_full_median_seed": alignment["ndcg_full_median_seed"],
            "spearman_min": TEACHER_SPEARMAN_MIN,
            "ndcg_full_min": TEACHER_NDCG_MIN,
        },
        "E_generalization": {
            "pass": gate_e,
            "seed_improved": seed_improved,
        },
        "F_hardware_contract": {
            "pass": gate_f,
            "hardware": hardware,
        },
        "all_pass": bool(gate_a and gate_b and gate_c and gate_d and gate_e and gate_f),
    }


def decide(arm_gates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Only the primary arm C and only a full A-F pass promote."""
    primary = arm_gates.get(PRIMARY_ARM)
    passed = bool(primary is not None and primary["all_pass"])
    return {
        "primary_arm": PRIMARY_ARM,
        "primary_arm_all_pass": passed,
        "primary_arm_evaluated": primary is not None,
        "winner": PRIMARY_ARM if passed else None,
        "decision": "PROMOTION_PASS" if passed else "NO_PROMOTION",
        "winner_rule": (
            "only arm C_PAIR_LAMBDA_0P25_PRIMARY may be promoted, and only if it "
            "passes all gates A-F; arms A/B are controls and never winners"
        ),
    }


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------
def run_experiment(
    *,
    prereg_config: Path = PREREG_CONFIG_PATH,
    prereg_report: Path = PREREG_REPORT_PATH,
    draft: Path = DRAFT_PATH,
    config: Path = CONFIG_PATH,
    index_path: Path = INDEX_PATH,
    features_dir: Path | None = None,
    artifacts_dir: Path = ARTIFACTS_DIR,
    reports_dir: Path = REPORTS_DIR,
    seeds: Sequence[int] = SEEDS,
    epochs: int = EPOCHS,
    arm_names: Sequence[str] = ARM_ORDER,
    enforce_frozen_inputs: bool = True,
    enforce_expected_counts: bool = True,
    cache_loader: Callable[..., dict[str, Any]] | None = None,
    write_outputs: bool = True,
    budgets: Sequence[int] = (48, 64, 96),
) -> dict[str, Any]:
    """Run the full preregistered three-arm experiment and return the bundle."""
    started = time.time()
    before_hashes = {
        "prereg_config": sha256_file(prereg_config),
        "prereg_report": sha256_file(prereg_report),
        "draft": sha256_file(draft),
        "config": sha256_file(config),
        "index": sha256_file(index_path),
    }
    if enforce_frozen_inputs:
        verify_frozen_inputs(prereg_config=prereg_config, prereg_report=prereg_report)
    prereg = load_prereg(prereg_config)

    index_records = load_index_metadata(index_path)
    splits = split_sample_ids(
        index_records,
        seed=int(prereg["split"]["seed"]),
        ratios=prereg["split"]["ratios"],
        salt=str(prereg["split"]["salt"]),
    )
    image_hashes = {
        record["sample_id"]: record["source_image_sha256"] for record in index_records
    }
    group_summary = split_group_summary(splits, image_hashes)
    if enforce_expected_counts:
        expected = {
            "train": (EXPECTED_TRAIN_RECORDS, EXPECTED_TRAIN_IMAGES),
            "val": (EXPECTED_VAL_RECORDS, EXPECTED_VAL_IMAGES),
            "test": (EXPECTED_TEST_RECORDS, EXPECTED_TEST_IMAGES),
        }
        for name, (records, images) in expected.items():
            got = group_summary[name]
            if got["sample_count"] != records or got["image_count"] != images:
                raise DistillationContractError(
                    f"split {name} = {got['sample_count']}/{got['image_count']} "
                    f"!= expected {records}/{images}"
                )

    opened: set[str] = set()
    features_directory = Path(features_dir) if features_dir else Path(index_path).parent
    train_samples, train_aggregate, train_paths = load_split_samples(
        index_path,
        features_dir,
        splits["train"],
        opened=opened,
        cache_loader=cache_loader,
    )
    val_samples, val_aggregate, val_paths = load_split_samples(
        index_path,
        features_dir,
        splits["val"],
        opened=opened,
        cache_loader=cache_loader,
    )
    test_ids = set(splits["test"])
    test_paths = {
        str(features_directory / str(record["npz"]))
        for record in index_records
        if record["sample_id"] in test_ids
    }
    test_npz_opened = sorted(path for path in opened if path in test_paths)
    if test_npz_opened:
        raise DistillationContractError(
            f"test NPZ files were opened: {test_npz_opened}"
        )
    if enforce_frozen_inputs and train_aggregate != TRAIN_NPZ_AGGREGATE_SHA256:
        raise DistillationContractError(
            f"train NPZ aggregate {train_aggregate} != frozen {TRAIN_NPZ_AGGREGATE_SHA256}"
        )

    train_pairs = build_image_pairs(train_samples)
    val_pairs = build_image_pairs(val_samples)
    tensors = {
        str(sample["sample_id"]): _tensorize(sample) for sample in [*train_samples, *val_samples]
    }

    per_arm_seed_stats: dict[str, dict[int, dict[str, Any]]] = {
        arm_name: {} for arm_name in arm_names
    }
    training_results: list[dict[str, Any]] = []
    epoch0_hashes: dict[str, dict[str, str]] = defaultdict(dict)
    val_sample_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for arm_name in arm_names:
            result = train_arm(
                arm_name=arm_name,
                seed=int(seed),
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                tensors=tensors,
                out_dir=artifacts_dir,
                epochs=int(epochs),
            )
            epoch0_hashes[str(seed)][arm_name] = result["epoch0_weights_sha256"]
            stats = val_statistics(
                val_samples, result["best_Wt"], result["best_Wv"], budgets=budgets
            )
            per_arm_seed_stats[arm_name][int(seed)] = stats
            for row in stats["rows"]:
                val_sample_rows.append(
                    {
                        "arm": arm_name,
                        "seed": int(seed),
                        **row,
                        "delta_same_spearman": row["correct_spearman"]
                        - row["same_spearman"],
                        "delta_same_ndcg_full": row["correct_ndcg_full"]
                        - row["same_ndcg_full"],
                        "delta_cross_spearman": row["correct_spearman"]
                        - row["cross_spearman"],
                        "delta_cross_ndcg_full": row["correct_ndcg_full"]
                        - row["cross_ndcg_full"],
                    }
                )
            result.pop("best_Wt", None)
            result.pop("best_Wv", None)
            training_results.append(result)

    # epoch-0 hashes must be identical across arms for the same seed
    epoch0_consistent = {
        seed: len(set(values.values())) == 1 for seed, values in epoch0_hashes.items()
    }
    if not all(epoch0_consistent.values()):
        raise DistillationContractError(
            f"epoch-0 weight hashes differ across arms: {dict(epoch0_hashes)}"
        )

    aggregated = aggregate_val_statistics(per_arm_seed_stats)
    training_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in training_results:
        training_by_arm[result["arm"]].append(result)
    arm_training: dict[str, Any] = {}
    for arm_name in arm_names:
        entries = sorted(training_by_arm[arm_name], key=lambda item: item["seed"])
        arm_training[arm_name] = {
            "arms": arm_name,
            "per_seed": entries,
            "best_epochs": [entry["best_epoch"] for entry in entries],
            "best_val_l_totals": [entry["best_val_l_total"] for entry in entries],
            "epoch0_val_l_totals": [entry["epoch0_val_l_total"] for entry in entries],
            "final_val_l_totals": [entry["final_val_l_total"] for entry in entries],
            "epoch0_weights_sha256_per_seed": {
                str(entry["seed"]): entry["epoch0_weights_sha256"] for entry in entries
            },
        }

    arm_gates = {
        arm_name: evaluate_gates(
            arm_result=aggregated["arms"][arm_name],
            training=arm_training[arm_name],
        )
        for arm_name in arm_names
    }
    decision = decide(arm_gates)

    runtime_seconds = time.time() - started
    try:
        peak_rss_mib = float(
            int(
                Path("/proc/self/status")
                .read_text(encoding="utf-8")
                .split("VmHWM:")[1]
                .split()[0]
            )
            / 1024.0
        )
    except Exception:  # pragma: no cover
        peak_rss_mib = None

    # post-run frozen-input proof (byte-for-byte before/after comparison)
    after_hashes = {
        "prereg_config": sha256_file(prereg_config),
        "prereg_report": sha256_file(prereg_report),
        "draft": sha256_file(draft),
        "config": sha256_file(config),
        "index": sha256_file(index_path),
    }
    unchanged = {
        key: bool(before_hashes[key] == after_hashes[key]) for key in before_hashes
    }
    if enforce_frozen_inputs:
        post = verify_frozen_inputs(
            prereg_config=prereg_config, prereg_report=prereg_report
        )
        draft_unchanged = bool(unchanged["draft"] and post["checks"]["draft_unchanged"])
        prereg_unchanged = bool(unchanged["prereg_config"] and post["checks"]["prereg_status"])
    else:
        post = {}
        prereg_unchanged = bool(unchanged["prereg_config"])
        draft_unchanged = bool(unchanged["draft"])

    bundle = {
        "status": decision["decision"],
        "report_stem": REPORT_STEM,
        "scope_note": SCOPE_NOTE,
        "preregistered_config": str(prereg_config),
        "preregistration_report": str(prereg_report),
        "prereg_config_sha256": after_hashes["prereg_config"],
        "prereg_report_sha256": after_hashes["prereg_report"],
        "draft_sha256": after_hashes["draft"],
        "config_sha256": after_hashes["config"],
        "index_sha256": after_hashes["index"],
        "before_hashes": before_hashes,
        "after_hashes": after_hashes,
        "hashes_unchanged": unchanged,
        "train_npz_aggregate_sha256": train_aggregate,
        "val_npz_aggregate_sha256": val_aggregate,
        "source_sha256": {
            "trainer": sha256_file(Path(__file__)),
        },
        "split": splits,
        "split_group_summary": group_summary,
        "split_group_by": "source_image_sha256",
        "isolation": {
            "train_npz_count": len(train_paths),
            "val_npz_count": len(val_paths),
            "opened_npz_count": len(opened),
            "test_npz_opened": test_npz_opened,
            "opened_only_train_val_npz": len(opened) == len(train_paths) + len(val_paths),
            "test_npz_opened_count": 0,
            "test_used_for_selection": False,
        },
        "arms": {arm["name"]: arm for arm in ARMS if arm["name"] in arm_names},
        "seeds": [int(seed) for seed in seeds],
        "epochs": int(epochs),
        "epoch0_hashes": {seed: dict(values) for seed, values in epoch0_hashes.items()},
        "epoch0_hashes_consistent_across_arms": epoch0_consistent,
        "training": arm_training,
        "validation_statistics": aggregated,
        "val_sample_rows": val_sample_rows,
        "gates": arm_gates,
        "decision": decision,
        "prereg_unchanged": prereg_unchanged,
        "draft_unchanged": draft_unchanged,
        "post_run_frozen_inputs": post,
        "runtime_seconds": float(runtime_seconds),
        "peak_rss_mib": peak_rss_mib,
        "hardware_used": {"gpu": False, "model_4b": False, "network": False},
        "limitations": [
            "internal train/val sanity only; not CV-Bench task accuracy",
            "no hardware synthesis, latency or speedup claim",
            "concentration evidence is explanatory, frozen before the run",
            "the internal test split was never opened",
        ],
    }

    if write_outputs:
        bundle["report_paths"] = write_reports(
            bundle, artifacts_dir=artifacts_dir, reports_dir=reports_dir
        )
    return bundle


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------
def epochs_rows(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm_name, training in bundle["training"].items():
        for entry in training["per_seed"]:
            for row in entry["epoch_metrics"] + [entry["final"]]:
                rows.append(
                    {
                        "arm": arm_name,
                        "seed": entry["seed"],
                        "epoch": row["epoch"],
                        "phase": row["phase"],
                        "split": "train",
                        "l_base": row["train"]["l_base"],
                        "l_pair": row["train"]["l_pair"],
                        "l_total": row["train"]["l_total"],
                        "hinge_active_fraction": row["train"]["hinge_active_fraction"],
                        "pair_count": row["train"]["pair_count"],
                        "grad_norm_before_clip": row["grad_norm_before_clip"],
                        "batch_count": row["batch_count"],
                    }
                )
                rows.append(
                    {
                        "arm": arm_name,
                        "seed": entry["seed"],
                        "epoch": row["epoch"],
                        "phase": row["phase"],
                        "split": "val",
                        "l_base": row["val"]["l_base"],
                        "l_pair": row["val"]["l_pair"],
                        "l_total": row["val"]["l_total"],
                        "hinge_active_fraction": row["val"]["hinge_active_fraction"],
                        "pair_count": row["val"]["pair_count"],
                        "grad_norm_before_clip": None,
                        "batch_count": None,
                    }
                )
    return rows


def write_reports(
    bundle: Mapping[str, Any],
    *,
    artifacts_dir: Path = ARTIFACTS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, str]:
    reports_dir = Path(reports_dir)
    paths: dict[str, str] = {}
    epoch_csv = reports_dir / f"{REPORT_STEM}-epochs.csv"
    _write_csv(
        epoch_csv,
        epochs_rows(bundle),
        [
            "arm",
            "seed",
            "epoch",
            "phase",
            "split",
            "l_base",
            "l_pair",
            "l_total",
            "hinge_active_fraction",
            "pair_count",
            "grad_norm_before_clip",
            "batch_count",
        ],
    )
    paths["epochs_csv"] = str(epoch_csv)

    sample_rows: list[dict[str, Any]] = list(bundle.get("val_sample_rows", []))
    val_csv = reports_dir / f"{REPORT_STEM}-val-samples.csv"
    _write_csv(
        val_csv,
        sample_rows,
        [
            "arm",
            "seed",
            "sample_id",
            "image_sha256",
            "task",
            "correct_spearman",
            "correct_ndcg_full",
            "same_spearman",
            "same_ndcg_full",
            "cross_spearman",
            "cross_ndcg_full",
            "delta_same_spearman",
            "delta_same_ndcg_full",
            "delta_cross_spearman",
            "delta_cross_ndcg_full",
            "exact_prompt_invariant",
            "tolerance_prompt_invariant",
            "max_abs_diff",
        ],
    )
    paths["val_samples_csv"] = str(val_csv)

    json_path = reports_dir / f"{REPORT_STEM}-train-val.json"
    _write_json(json_path, bundle)
    paths["json"] = str(json_path)

    md_path = reports_dir / f"{REPORT_STEM}-train-val.md"
    md_path.write_text(render_markdown(bundle), encoding="utf-8")
    paths["md"] = str(md_path)
    return paths


def render_markdown(bundle: Mapping[str, Any]) -> str:
    decision = bundle["decision"]
    lines: list[str] = []
    lines.append(f"# Pairwise-retrain train/val report ({REPORT_STEM})")
    lines.append("")
    lines.append(f"- status: **{bundle['status']}**")
    lines.append(f"- scope: {bundle['scope_note']}")
    lines.append(
        f"- prereg config SHA256: `{bundle['prereg_config_sha256']}` "
        f"(unchanged={bundle['prereg_unchanged']})"
    )
    lines.append(
        f"- prereg report SHA256: `{bundle['prereg_report_sha256']}`"
    )
    lines.append(
        f"- draft SHA256: `{bundle['draft_sha256']}` (unchanged={bundle['draft_unchanged']})"
    )
    lines.append(f"- config SHA256: `{bundle['config_sha256']}`")
    lines.append(f"- index SHA256: `{bundle['index_sha256']}`")
    lines.append(
        f"- before/after hashes unchanged: `{bundle['hashes_unchanged']}`"
    )
    lines.append(f"- train NPZ aggregate SHA256: `{bundle['train_npz_aggregate_sha256']}`")
    lines.append(f"- val NPZ aggregate SHA256: `{bundle['val_npz_aggregate_sha256']}`")
    lines.append(
        f"- test_npz_opened = `{bundle['isolation']['test_npz_opened']}` "
        f"(count={bundle['isolation']['test_npz_opened_count']})"
    )
    lines.append(
        f"- split: train={bundle['split_group_summary']['train']} "
        f"val={bundle['split_group_summary']['val']} "
        f"test={bundle['split_group_summary']['test']}"
    )
    lines.append(f"- runtime_seconds={bundle['runtime_seconds']:.2f} peak_rss_mib={bundle['peak_rss_mib']}")
    lines.append(f"- 4B model loaded: {bundle['hardware_used']['model_4b']}; GPU used: {bundle['hardware_used']['gpu']}")
    lines.append("")
    lines.append("## Arm / seed losses")
    lines.append("")
    lines.append("| arm | seed | epoch0 val L_total | best epoch | best val L_total | final val L_total | epoch0 hash | best hash |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm_name, training in bundle["training"].items():
        for entry in training["per_seed"]:
            lines.append(
                f"| {arm_name} | {entry['seed']} | {entry['epoch0_val_l_total']:.6f} | "
                f"{entry['best_epoch']} | {entry['best_val_l_total']:.6f} | "
                f"{entry['final_val_l_total']:.6f} | `{entry['epoch0_weights_sha256'][:16]}` | "
                f"`{entry['best_weights_sha256'][:16]}` |"
            )
    lines.append("")
    lines.append("## Primary deltas (per-image median across seeds, image-level)")
    lines.append("")
    lines.append("| arm | control | metric | mean | ci_low | ci_high | dz | p | holm_p | +seeds |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm_name, arm in bundle["validation_statistics"]["arms"].items():
        for role in ROLES:
            for metric in PRIMARY_METRICS:
                entry = arm["deltas"][role][metric]
                lines.append(
                    f"| {arm_name} | {role} | {metric} | {entry['mean']:.6f} | "
                    f"{entry['ci_low']:.6f} | {entry['ci_high']:.6f} | "
                    f"{entry['cohens_dz']:.4f} | {entry['p_value_two_sided']:.5f} | "
                    f"{entry['holm_adjusted_p']:.5f} | {entry['positive_seed_count']}/{entry['seed_count']} |"
                )
    lines.append("")
    lines.append("## Gates A-F")
    lines.append("")
    for arm_name, gate in bundle["gates"].items():
        lines.append(
            f"- **{arm_name}**: A={gate['A_prompt_sensitivity']['pass']} "
            f"B={gate['B_shuffled_discrimination']['pass']} "
            f"C={gate['C_collapse_removed']['pass']} "
            f"D={gate['D_teacher_alignment']['pass']} "
            f"E={gate['E_generalization']['pass']} "
            f"F={gate['F_hardware_contract']['pass']} "
            f"all_pass={gate['all_pass']}"
        )
    lines.append("")
    lines.append(
        f"**Fixed winner rule**: primary arm `{decision['primary_arm']}`; "
        f"winner={decision['winner']}; decision=**{decision['decision']}**."
    )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for item in bundle["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg-config", type=Path, default=PREREG_CONFIG_PATH)
    parser.add_argument("--prereg-report", type=Path, default=PREREG_REPORT_PATH)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument(
        "--harness-result",
        type=Path,
        default=Path("/tmp/robobrain_harness_result.json"),
    )
    args = parser.parse_args(argv)
    bundle = run_experiment(
        prereg_config=args.prereg_config,
        prereg_report=args.prereg_report,
        index_path=args.index,
        features_dir=args.features_dir,
        artifacts_dir=args.artifacts_dir,
        reports_dir=args.reports_dir,
        seeds=args.seeds,
        epochs=args.epochs,
    )
    summary = {
        "status": bundle["status"],
        "prereg_sha256": bundle["prereg_config_sha256"],
        "draft_sha256_unchanged": bundle["draft_unchanged"],
        "decision": bundle["decision"]["decision"],
    }
    args.harness_result.parent.mkdir(parents=True, exist_ok=True)
    args.harness_result.write_text(
        json.dumps(_harness_payload(bundle), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


def _harness_payload(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": bundle["status"],
        "prereg_sha256": bundle["prereg_config_sha256"],
        "prereg_report_sha256": bundle["prereg_report_sha256"],
        "draft_sha256_unchanged": bundle["draft_unchanged"],
        "hashes_unchanged": bundle["hashes_unchanged"],
        "source_sha256": bundle["source_sha256"],
        "arms": list(bundle["arms"]),
        "seeds": bundle["seeds"],
        "train_records": bundle["split_group_summary"]["train"]["sample_count"],
        "val_records": bundle["split_group_summary"]["val"]["sample_count"],
        "test_npz_opened": bundle["isolation"]["test_npz_opened"],
        "runtime_seconds": bundle["runtime_seconds"],
        "peak_rss_mib": bundle["peak_rss_mib"],
        "decision": bundle["decision"],
        "gates": {
            arm: {
                key: gate[key]["pass"] if isinstance(gate.get(key), Mapping) else gate.get(key)
                for key in (
                    "A_prompt_sensitivity",
                    "B_shuffled_discrimination",
                    "C_collapse_removed",
                    "D_teacher_alignment",
                    "E_generalization",
                    "F_hardware_contract",
                )
            }
            for arm, gate in bundle["gates"].items()
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
