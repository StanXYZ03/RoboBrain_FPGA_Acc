#!/usr/bin/env python3
"""Preregistered spatial-supervision train/val experiment for the selector.

Two arms, only B may pass:

* ``A_SEMANTIC_KL_CONTROL`` -- target = cached ``teacher_soft_target``.
* ``B_SPATIAL_KL_PRIMARY`` -- grounding target = exact bbox/grid-cell area
  target; pointing target = bilinear point target, both from the corrected
  :mod:`evaluation.spatial_teacher_feasibility_audit`.

Per-record loss is ``KL(target || softmax(student_score / 0.1))`` in nats; an
image-pair loss is the mean of the grounding and pointing record losses.  The
architecture, feature cache, split, optimizer and hardware contract are frozen.

Hard boundaries (enforced and unit-tested):

* **train/val feature caches only** -- the internal test NPZ files are never
  resolved, opened, hashed or scored.
* **CPU-only float32** -- about 458,752 trainable parameters in a small
  ``Wt[128,2560]``/``Wv[128,1024]`` head; this is **not** 4B model training.
* **no 4B model, no feature re-extraction, no raw image read, no CV-Bench, no
  network, no package installation**.

Reuse (no second implementation of any scorer / split / donor / statistic):

* split / loader:   :func:`evaluation.prompt_projection_distillation.split_sample_ids`
                    and :func:`evaluation.train_pairwise_prompt_selector.load_split_samples`
* scorer:           :func:`evaluation.pair_loss_feasibility_audit.student_scores`
                    (``prompt_mean_vector``) / :func:`...frozen_scorer_reduction_ablation.score_candidate`
* targets:          :func:`evaluation.spatial_teacher_feasibility_audit.{grounding_area_target,pointing_bilinear_target}`
* losses:           :func:`evaluation.pair_loss_feasibility_audit.{student_log_prob,kl_teacher_student}`
* donors:           :func:`evaluation.prompt_identifiability_diagnostic.{derange_images,sample_donor_map,same_image_partner_map,image_task_index}`
* statistics:       :func:`evaluation.prompt_identifiability_diagnostic.paired_effect`
                    and :func:`evaluation.frozen_scorer_reduction_ablation.holm_bonferroni`
* metrics:          :func:`evaluation.prompt_projection_distillation.{spearman,ndcg_at_k,ndcg_for_selected_indices,saliency_mass_retained,l2_baseline_scores}`
* deployed selector: :func:`evaluation.hardware_regular_token_selector.{combine_scores,select_real_groups}`
* checkpointing:    :func:`evaluation.prompt_projection_distillation.save_projection_npz`
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from evaluation.hardware_regular_token_selector import (
    InsufficientBudgetError,
    REGION_GRID,
    combine_scores,
    select_real_groups,
)
from evaluation.pair_loss_feasibility_audit import (
    kl_teacher_student,
    student_log_prob,
    student_scores,
)
from evaluation.prompt_identifiability_diagnostic import (
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
    l2_baseline_scores,
    load_feature_cache,
    ndcg_at_k,
    ndcg_for_selected_indices,
    saliency_mass_retained,
    save_projection_npz,
    spearman,
    split_group_summary,
    split_sample_ids,
    topk_indices,
    validate_source_image_sha256,
)
from evaluation.spatial_teacher_feasibility_audit import (
    SpatialContractError,
    grounding_area_target,
    load_clean_manifest,
    point_inside_bbox,
    pointing_bilinear_target,
    target_diagnostics,
)
from evaluation.teacher_pair_signal_audit import (
    aggregate_npz_sha256,
    load_index_metadata,
)
from evaluation.train_pairwise_prompt_selector import (
    epoch_batch_order,
    load_split_samples,
    select_best_epoch,
    sha256_file,
)
from evaluation.train_prompt_projection import init_projection

try:  # pragma: no cover - exercised only when torch is missing
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def _require_torch() -> Any:
    if not TORCH_AVAILABLE:
        raise DistillationContractError("torch is required for the spatial trainer")
    return torch


# ---------------------------------------------------------------------------
# Frozen constants (cross-checked against the preregistered config at run time)
# ---------------------------------------------------------------------------
REPORT_STEM = "2026-10-03-spatial-student-trainval"
PREREG_CONFIG_PATH = Path("configs/spatial_student_trainval.preregistered.json")
PREREG_REPORT_PATH = Path(
    "reports/token_selector/2026-10-03-spatial-student-trainval-preregistration.json"
)
CONFIG_PATH = Path("configs/prompt_projection_distillation.json")
INDEX_PATH = Path("artifacts/token_selector/prompt_projection_features/index.jsonl")
MANIFEST_PATH = Path("datasets/prompt_projection_clean_inductive_v1.jsonl")
LOCK_PATH = Path("datasets/prompt_projection_clean_inductive_v1.lock.json")
SPATIAL_AUDIT_JSON_PATH = Path(
    "reports/token_selector/2026-10-03-spatial-teacher-feasibility.json"
)
SPATIAL_AUDIT_SCRIPT_PATH = Path("evaluation/spatial_teacher_feasibility_audit.py")
SEMANTIC_CONTROL_PATH = Path("evaluation/train_pairwise_prompt_selector.py")
HARDWARE_CONFIG_PATH = Path("configs/hardware_regular_token_selector.json")
HARDWARE_CODE_PATH = Path("evaluation/hardware_regular_token_selector.py")
ARTIFACTS_DIR = Path("artifacts/token_selector/spatial_student_trainval")
REPORTS_DIR = Path("reports/token_selector")
LOGS_DIR = REPORTS_DIR / "logs"

PREREG_STATUS = "PREREGISTERED_TRAIN_VAL_ONLY"

CONFIG_SHA256 = "44399c8e6fb8f333a150a6016faa00d6ae70524b5fcd030bcabf4e5fb07ab28b"
INDEX_SHA256 = "02a50868a84675418e9bd9e5fdd0bb55f522ec5ce0ce1d6da7c69f85d2aeee26"
MANIFEST_SHA256 = "acab1a322a93021138ce87bb048c1e25493ec19a3f56f70dc3a24c0686c79213"
LOCK_SHA256 = "a723458da9b9e170c9ffcde3bfa8844ae18c3e84fe1ba15bbd3d4b3f57e2e105"
TRAIN_NPZ_AGGREGATE_SHA256 = (
    "33997394c4e148e3cfcd45eb0eb37fc97616cece8db3da5c731171d4b53d3951"
)
SPATIAL_AUDIT_JSON_SHA256 = (
    "1bd2b9e41f3e6d698548ff7ad85c9b0a0b513e0d3ef4d4b0c02cc259ffbcc739"
)
SPATIAL_AUDIT_SCRIPT_SHA256 = (
    "1d033c4036d5f2319c8828e393594fe0b6330ac9c43afe43af14499adf748aec"
)
SEMANTIC_CONTROL_SHA256 = (
    "07b504d79743915b7b8dacf7028dea1de91ba36bb2776dedd6ed9e4012032fec"
)
HARDWARE_CONFIG_SHA256 = (
    "6350c89bfc48bf59ae507bbacd80bda2eb3ba9f6cefbfde07ad1593c6e418142"
)
HARDWARE_CODE_SHA256 = (
    "c55cfdc39e798e7566cd609a77ac388dca85df463ec20f536448284e2f85932b"
)

EXPECTED_TRAIN_RECORDS = 318
EXPECTED_TRAIN_IMAGES = 159
EXPECTED_VAL_RECORDS = 76
EXPECTED_VAL_IMAGES = 38
EXPECTED_TEST_RECORDS = 78
EXPECTED_TEST_IMAGES = 39

STUDENT_TEMPERATURE = 0.1
INIT_STD = 0.02
EPOCHS = 30
BATCH_SIZE_IMAGES = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0
SEEDS: tuple[int, ...] = (20261003, 20261004, 20261005)
BUDGETS: tuple[int, ...] = (48, 64, 96)

BOOTSTRAP_RESAMPLES = 2000
PERMUTATION_RESAMPLES = 10000
ALPHA = 0.05
CI_LEVEL = 0.95
EFFECT_SIZE_FLOOR = 0.2
PROMPT_INVARIANCE_TOLERANCE = 1e-9
COLLAPSE_INVARIANCE_MAX = 0.05
STATISTICS_SEED = 20261003
DONOR_SEED = 20261103
POSITIVE_SEED_MIN = 2
ANCHOR_REGION_GRID = 3
LAMBDA_SEMANTIC = 0.5

ARM_ORDER: tuple[str, ...] = ("A_SEMANTIC_KL_CONTROL", "B_SPATIAL_KL_PRIMARY")
ARMS: tuple[dict[str, Any], ...] = (
    {
        "name": "A_SEMANTIC_KL_CONTROL",
        "target": "semantic_teacher",
        "role": "control",
        "can_win": False,
    },
    {
        "name": "B_SPATIAL_KL_PRIMARY",
        "target": "spatial",
        "role": "primary_promotion_arm",
        "can_win": True,
    },
)
ARM_SPEC: dict[str, dict[str, Any]] = {str(arm["name"]): arm for arm in ARMS}
PRIMARY_ARM = "B_SPATIAL_KL_PRIMARY"
CONTROL_ARM = "A_SEMANTIC_KL_CONTROL"

PRIMARY_TESTS: tuple[dict[str, Any], ...] = (
    {
        "key": "B_minus_A_correct_spearman",
        "kind": "arm_delta",
        "metric": "correct_spearman",
        "base": "correct_spearman",
        "definition": "arm B minus arm A correct-prompt spatial Spearman",
    },
    {
        "key": "B_minus_A_correct_ndcg_full",
        "kind": "arm_delta",
        "metric": "correct_ndcg_full",
        "base": "correct_ndcg_full",
        "definition": "arm B minus arm A correct-prompt spatial full NDCG",
    },
    {
        "key": "B_correct_minus_same_spearman",
        "kind": "correct_minus_control",
        "metric": "same_spearman",
        "base": "correct_spearman",
        "definition": "arm B correct minus same-image-other-task spatial Spearman",
    },
    {
        "key": "B_correct_minus_same_ndcg_full",
        "kind": "correct_minus_control",
        "metric": "same_ndcg_full",
        "base": "correct_ndcg_full",
        "definition": "arm B correct minus same-image-other-task spatial full NDCG",
    },
    {
        "key": "B_selector_minus_L2_anchor_mass_k48",
        "kind": "selector_minus_baseline",
        "metric": "selector_mass_k48",
        "base": "l2_anchor_mass_k48",
        "definition": "arm B deployed selector minus L2+anchor retained spatial-target mass at K48",
    },
)
HOLM_FAMILY_SIZE = len(PRIMARY_TESTS)

SCOPE_NOTE = (
    "train/val only, CPU-only float32, ~458752-parameter Wt/Wv head (NOT 4B "
    "training), no GPU, no 4B model, no feature re-extraction, no raw image "
    "read, no CV-Bench, no network, no internal test NPZ open; internal sanity "
    "only, not CV-Bench accuracy"
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
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


def verify_frozen_inputs(
    *,
    prereg_config: Path = PREREG_CONFIG_PATH,
    prereg_report: Path = PREREG_REPORT_PATH,
    config: Path = CONFIG_PATH,
    index: Path = INDEX_PATH,
    manifest: Path = MANIFEST_PATH,
    lock: Path = LOCK_PATH,
    spatial_audit_json: Path = SPATIAL_AUDIT_JSON_PATH,
    spatial_audit_script: Path = SPATIAL_AUDIT_SCRIPT_PATH,
    semantic_control: Path = SEMANTIC_CONTROL_PATH,
    hardware_config: Path = HARDWARE_CONFIG_PATH,
    hardware_code: Path = HARDWARE_CODE_PATH,
) -> dict[str, Any]:
    """Recompute every frozen SHA and fail closed on any drift."""
    actual = {
        "config_sha256": sha256_file(config),
        "index_sha256": sha256_file(index),
        "manifest_sha256": sha256_file(manifest),
        "lock_sha256": sha256_file(lock),
        "spatial_audit_json_sha256": sha256_file(spatial_audit_json),
        "spatial_audit_script_sha256": sha256_file(spatial_audit_script),
        "semantic_control_sha256": sha256_file(semantic_control),
        "hardware_config_sha256": sha256_file(hardware_config),
        "hardware_code_sha256": sha256_file(hardware_code),
    }
    expected = {
        "config_sha256": CONFIG_SHA256,
        "index_sha256": INDEX_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "lock_sha256": LOCK_SHA256,
        "spatial_audit_json_sha256": SPATIAL_AUDIT_JSON_SHA256,
        "spatial_audit_script_sha256": SPATIAL_AUDIT_SCRIPT_SHA256,
        "semantic_control_sha256": SEMANTIC_CONTROL_SHA256,
        "hardware_config_sha256": HARDWARE_CONFIG_SHA256,
        "hardware_code_sha256": HARDWARE_CODE_SHA256,
    }
    checks = {f"{key}_unchanged": actual[key] == expected[key] for key in expected}
    checks["prereg_status"] = load_prereg(prereg_config)["status"] == PREREG_STATUS
    # The preregistration report pins this config's SHA256; verify the pair.
    report = json.loads(Path(prereg_report).read_text(encoding="utf-8"))
    pinned_config = str(report["preregistered_config"]["sha256"])
    actual_config = sha256_file(prereg_config)
    checks["prereg_report_pins_config"] = pinned_config == actual_config
    checks["prereg_report_status"] = str(report.get("status")) == PREREG_STATUS
    if not all(checks.values()):
        raise DistillationContractError(f"frozen input drift detected: {checks}")
    return {
        "actual": actual,
        "expected": expected,
        "checks": checks,
        "prereg_config_sha256": actual_config,
        "prereg_report_sha256": sha256_file(prereg_report),
    }


# ---------------------------------------------------------------------------
# Spatial targets
# ---------------------------------------------------------------------------
def attach_spatial_targets(
    samples: Sequence[Mapping[str, Any]],
    manifest_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach the frozen spatial target per record (grounding / pointing)."""
    result: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        if sample_id not in manifest_by_id:
            raise SpatialContractError(f"sample {sample_id} missing from manifest")
        ground_truth = manifest_by_id[sample_id]["ground_truth"]
        grid_h = int(sample["grid_h"])
        grid_w = int(sample["grid_w"])
        if int(sample["group_count"]) != grid_h * grid_w:
            raise SpatialContractError(
                f"{sample_id}: group_count != grid_h*grid_w"
            )
        task = str(sample["task"])
        if task == "grounding":
            target = grounding_area_target(
                ground_truth["bbox_1000"], grid_h, grid_w
            )
        elif task == "pointing":
            target = pointing_bilinear_target(
                ground_truth["point_1000"], grid_h, grid_w
            )
        else:
            raise SpatialContractError(f"unknown task {task!r} for {sample_id}")
        if int(target.size) != int(sample["group_count"]):
            raise SpatialContractError(
                f"{sample_id}: spatial target size != group_count"
            )
        diagnostics = target_diagnostics(target)
        if not (
            diagnostics["finite"]
            and diagnostics["non_negative"]
            and diagnostics["sum_to_one_within_1e-9"]
        ):
            raise SpatialContractError(f"{sample_id}: invalid spatial target")
        bbox = np.asarray(ground_truth["bbox_1000"], dtype=np.float64)
        point = np.asarray(ground_truth["point_1000"], dtype=np.float64)
        entry = dict(sample)
        entry["spatial_target"] = target
        entry["spatial_diagnostics"] = diagnostics
        entry["point_inside_bbox"] = point_inside_bbox(point, bbox)
        result.append(entry)
    return result


def record_target(sample: Mapping[str, Any], arm_name: str) -> np.ndarray:
    """The frozen per-record training target for one arm."""
    if arm_name == CONTROL_ARM:
        return np.asarray(sample["teacher_soft"], dtype=np.float64)
    if arm_name == PRIMARY_ARM:
        return np.asarray(sample["spatial_target"], dtype=np.float64)
    raise DistillationContractError(f"unknown arm {arm_name!r}")


# ---------------------------------------------------------------------------
# Differentiable record/pair loss
# ---------------------------------------------------------------------------
def _tensorize(sample: Mapping[str, Any]) -> dict[str, Any]:
    t = _require_torch()
    return {
        "group_mean": t.as_tensor(np.asarray(sample["group_mean"], dtype=np.float32)),
        "prompt_embeddings": t.as_tensor(
            np.asarray(sample["prompt_embeddings"], dtype=np.float32)
        ),
        "teacher_soft": t.as_tensor(
            np.asarray(sample["teacher_soft"], dtype=np.float32)
        ),
        "spatial_target": t.as_tensor(
            np.asarray(sample["spatial_target"], dtype=np.float32)
        ),
    }


def record_kl(
    scores: Any,
    target: Any,
    *,
    temperature: float = STUDENT_TEMPERATURE,
) -> Any:
    """``KL(target || softmax(scores / temperature))`` in nats (no T^2 scaling)."""
    return kl_teacher_student(
        target, student_log_prob(scores, temperature=temperature)
    )


def _target_tensor(tensors: Mapping[str, Any], sample_id: str, arm_name: str) -> Any:
    entry = tensors[str(sample_id)]
    if arm_name == CONTROL_ARM:
        return entry["teacher_soft"]
    if arm_name == PRIMARY_ARM:
        return entry["spatial_target"]
    raise DistillationContractError(f"unknown arm {arm_name!r}")


def pair_kl_loss(
    tensors: Mapping[str, Any],
    pair: Mapping[str, Any],
    Wt: Any,
    Wv: Any,
    arm_name: str,
    *,
    temperature: float = STUDENT_TEMPERATURE,
) -> Any:
    """Mean of the grounding and pointing record KL losses for one image pair."""
    grounding = tensors[str(pair["grounding_id"])]
    pointing = tensors[str(pair["pointing_id"])]
    scores_g = student_scores(
        grounding["group_mean"], grounding["prompt_embeddings"], Wt, Wv
    )
    scores_p = student_scores(
        pointing["group_mean"], pointing["prompt_embeddings"], Wt, Wv
    )
    kl_g = record_kl(
        scores_g, _target_tensor(tensors, pair["grounding_id"], arm_name),
        temperature=temperature,
    )
    kl_p = record_kl(
        scores_p, _target_tensor(tensors, pair["pointing_id"], arm_name),
        temperature=temperature,
    )
    return 0.5 * (kl_g + kl_p)


def evaluate_split_kl(
    tensors: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    Wt: Any,
    Wv: Any,
    arm_name: str,
    *,
    temperature: float = STUDENT_TEMPERATURE,
) -> dict[str, Any]:
    """Mean image-pair KL for one split."""
    t = _require_torch()
    if not pairs:
        raise DistillationContractError("cannot evaluate an empty pair set")
    values: list[float] = []
    with t.no_grad():
        for pair in pairs:
            value = pair_kl_loss(
                tensors, pair, Wt, Wv, arm_name, temperature=temperature
            )
            values.append(float(value))
    return {
        "pair_count": len(pairs),
        "kl": float(np.mean(values)),
        "kl_min": float(np.min(values)),
        "kl_max": float(np.max(values)),
    }


# ---------------------------------------------------------------------------
# Image pairs
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
        if int(grounding["grid_h"]) != int(pointing["grid_h"]) or int(
            grounding["grid_w"]
        ) != int(pointing["grid_w"]):
            raise DistillationContractError(
                f"image {image_sha} grounding/pointing grids differ"
            )
        pairs.append(
            {
                "image_sha256": str(image_sha),
                "grounding_id": str(grounding["sample_id"]),
                "pointing_id": str(pointing["sample_id"]),
                "group_count": int(grounding["group_count"]),
                "grid_h": int(grounding["grid_h"]),
                "grid_w": int(grounding["grid_w"]),
            }
        )
    return pairs


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

    Wt, Wv = init_projection(int(seed), INIT_STD)
    Wt = t.nn.Parameter(Wt.detach().clone())
    Wv = t.nn.Parameter(Wv.detach().clone())
    epoch0_hash = canonical_projection_sha256(
        Wt.detach().numpy(), Wv.detach().numpy()
    )

    optimizer = t.optim.AdamW(
        [Wt, Wv], lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    image_hashes = [str(pair["image_sha256"]) for pair in train_pairs]
    epoch_metrics: list[dict[str, Any]] = []
    best_val = math.inf
    best_epoch = -1
    best_state: dict[str, Any] | None = None

    for epoch in range(int(epochs)):
        # epoch 0 is the pre-update evaluation
        train_stats = evaluate_split_kl(
            tensors, train_pairs, Wt, Wv, arm_name, temperature=temperature
        )
        val_stats = evaluate_split_kl(
            tensors, val_pairs, Wt, Wv, arm_name, temperature=temperature
        )
        if float(val_stats["kl"]) < best_val:
            best_val = float(val_stats["kl"])
            best_epoch = epoch
            best_state = {"Wt": Wt.detach().clone(), "Wv": Wv.detach().clone()}

        order = epoch_batch_order(seed, epoch, image_hashes, batch_size_images)
        grad_norms: list[float] = []
        batch_losses: list[float] = []
        for batch in order:
            optimizer.zero_grad(set_to_none=True)
            total = t.zeros((), dtype=t.float32)
            for pair_index in batch:
                total = total + pair_kl_loss(
                    tensors,
                    train_pairs[pair_index],
                    Wt,
                    Wv,
                    arm_name,
                    temperature=temperature,
                )
            total = total / max(len(batch), 1)
            total.backward()
            grad_norms.append(
                float(t.nn.utils.clip_grad_norm_([Wt, Wv], grad_clip_norm))
            )
            optimizer.step()
            batch_losses.append(float(total.detach()))
        epoch_metrics.append(
            {
                "epoch": epoch,
                "phase": "pre_update",
                "train": train_stats,
                "val": val_stats,
                "grad_norm_before_clip": float(np.mean(grad_norms))
                if grad_norms
                else None,
                "batch_count": len(order),
                "batch_loss_mean": float(np.mean(batch_losses))
                if batch_losses
                else None,
            }
        )

    if best_state is None:
        raise DistillationContractError("training produced no best state")

    final_train = evaluate_split_kl(
        tensors, train_pairs, Wt, Wv, arm_name, temperature=temperature
    )
    final_val = evaluate_split_kl(
        tensors, val_pairs, Wt, Wv, arm_name, temperature=temperature
    )
    final_row = {
        "epoch": int(epochs),
        "phase": "final_post_update",
        "train": final_train,
        "val": final_val,
        "grad_norm_before_clip": None,
        "batch_count": len(
            epoch_batch_order(seed, int(epochs) - 1, image_hashes, batch_size_images)
        )
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
        "target": ARM_SPEC[arm_name]["target"],
        "seed": int(seed),
        "epochs": int(epochs),
        "epoch0_weights_sha256": epoch0_hash,
        "best_epoch": int(best_epoch),
        "best_val_kl": float(best_val),
        "epoch0_val_kl": float(epoch_metrics[0]["val"]["kl"]),
        "final_val_kl": float(final_val["kl"]),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "loss_definition": "mean over image pairs of mean(grounding KL, pointing KL)",
        "student_temperature": float(temperature),
        "temperature_squared_scaling": False,
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
        "target": ARM_SPEC[arm_name]["target"],
        "epoch0_weights_sha256": epoch0_hash,
        "best_epoch": int(best_epoch),
        "best_val_kl": float(best_val),
        "epoch0_val_kl": float(epoch_metrics[0]["val"]["kl"]),
        "final_val_kl": float(final_val["kl"]),
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
# Validation metrics
# ---------------------------------------------------------------------------
def _select_groups(
    scores: np.ndarray,
    grid_h: int,
    grid_w: int,
    budget: int,
) -> np.ndarray:
    """Exact existing selector path; identity when the budget covers all groups."""
    group_count = int(np.asarray(scores).shape[0])
    effective = min(int(budget), group_count)
    if effective >= group_count:
        return np.arange(group_count, dtype=np.int64)
    try:
        selection = select_real_groups(scores, int(grid_h), int(grid_w), effective)
    except InsufficientBudgetError as exc:  # pragma: no cover - defensive
        raise DistillationContractError(str(exc)) from exc
    return np.asarray(selection.selected, dtype=np.int64)


def deployed_selection_metrics(
    combined: np.ndarray,
    l2_scores: np.ndarray,
    spatial_target: np.ndarray,
    grid_h: int,
    grid_w: int,
    *,
    budgets: Sequence[int] = BUDGETS,
) -> dict[str, float]:
    """Retained spatial-target mass and corrected selected-set NDCG per budget."""
    target = np.asarray(spatial_target, dtype=np.float64).reshape(-1)
    result: dict[str, float] = {}
    for budget in budgets:
        effective = min(int(budget), int(target.size))
        selected = _select_groups(combined, grid_h, grid_w, effective)
        baseline = _select_groups(l2_scores, grid_h, grid_w, effective)
        result[f"selector_mass_k{int(budget)}"] = float(
            saliency_mass_retained(selected, target)
        )
        result[f"selector_ndcg_k{int(budget)}"] = float(
            ndcg_for_selected_indices(selected, combined, target, effective)
        )
        result[f"l2_anchor_mass_k{int(budget)}"] = float(
            saliency_mass_retained(baseline, target)
        )
        result[f"l2_anchor_ndcg_k{int(budget)}"] = float(
            ndcg_for_selected_indices(baseline, l2_scores, target, effective)
        )
    return result


def val_statistics(
    val_samples: Sequence[Mapping[str, Any]],
    Wt: np.ndarray,
    Wv: np.ndarray,
    *,
    donor_seed: int = DONOR_SEED,
    budgets: Sequence[int] = BUDGETS,
    lambda_semantic: float = LAMBDA_SEMANTIC,
) -> dict[str, Any]:
    """Per-image correct / same / cross / deployed-selector val metrics."""
    by_id = {str(sample["sample_id"]): sample for sample in val_samples}
    ordered = sorted(by_id)
    partner = same_image_partner_map(val_samples)
    image_index = image_task_index(val_samples)
    image_ids = sorted(image_index)
    donor_by_image = derange_images(image_ids, seed=int(donor_seed))
    donor = sample_donor_map(val_samples, donor_by_image, same_task=True)

    image_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    invariance: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for sample_id in ordered:
        sample = by_id[sample_id]
        if sample_id not in partner:
            raise DistillationContractError(
                f"val sample {sample_id} has no same-image other-task partner"
            )
        other = by_id[partner[sample_id]]
        donor_sample = by_id[donor[sample_id]]
        target = np.asarray(sample["spatial_target"], dtype=np.float64)
        group_count = int(target.size)
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

        values: dict[str, float] = {
            "correct_spearman": float(spearman(scores_correct, target)),
            "correct_ndcg_full": float(
                ndcg_at_k(scores_correct, target, group_count)
            ),
            "same_spearman": float(spearman(scores_same, target)),
            "same_ndcg_full": float(ndcg_at_k(scores_same, target, group_count)),
            "cross_spearman": float(spearman(scores_cross, target)),
            "cross_ndcg_full": float(ndcg_at_k(scores_cross, target, group_count)),
        }
        for budget in budgets:
            values[f"correct_mass_k{int(budget)}"] = float(
                saliency_mass_retained(
                    topk_indices(scores_correct, int(budget)), target
                )
            )
            values[f"same_mass_k{int(budget)}"] = float(
                saliency_mass_retained(topk_indices(scores_same, int(budget)), target)
            )
            values[f"cross_mass_k{int(budget)}"] = float(
                saliency_mass_retained(topk_indices(scores_cross, int(budget)), target)
            )
        l2_scores = l2_baseline_scores(sample["group_mean"])
        combined = combine_scores(
            np.asarray(scores_correct, dtype=np.float64),
            l2_scores,
            float(lambda_semantic),
        )
        values.update(
            deployed_selection_metrics(
                combined,
                l2_scores,
                target,
                int(sample["grid_h"]),
                int(sample["grid_w"]),
                budgets=budgets,
            )
        )
        inv = score_invariance_stats(
            scores_correct, scores_same, tolerance=PROMPT_INVARIANCE_TOLERANCE
        )
        invariance.append(inv)

        image = str(sample["source_image_sha256"])
        for key, value in values.items():
            image_values[key][image].append(float(value))

        rows.append(
            {
                "sample_id": sample_id,
                "image_sha256": image,
                "task": str(sample["task"]),
                "group_count": group_count,
                **{key: float(value) for key, value in values.items()},
                "exact_prompt_invariant": bool(inv["exact"]),
                "tolerance_prompt_invariant": bool(inv["tolerance"]),
                "max_abs_diff": float(inv["max_abs_diff"]),
            }
        )

    image_metrics = {
        key: {image: float(np.mean(vals)) for image, vals in per_image.items()}
        for key, per_image in image_values.items()
    }

    def mean_metric(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    return {
        "count": len(ordered),
        "independent_image_units": len(image_ids),
        "correct_spearman": mean_metric("correct_spearman"),
        "correct_ndcg_full": mean_metric("correct_ndcg_full"),
        "same_spearman": mean_metric("same_spearman"),
        "same_ndcg_full": mean_metric("same_ndcg_full"),
        "cross_spearman": mean_metric("cross_spearman"),
        "cross_ndcg_full": mean_metric("cross_ndcg_full"),
        "exact_prompt_invariance_fraction": float(
            np.mean([entry["exact"] for entry in invariance])
        ),
        "tolerance_prompt_invariance_fraction": float(
            np.mean([entry["tolerance"] for entry in invariance])
        ),
        "max_abs_diff_max": float(
            max(entry["max_abs_diff"] for entry in invariance)
        ),
        "image_metrics": image_metrics,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Cross-seed aggregation and the five-test Holm family
# ---------------------------------------------------------------------------
def _median(values: Sequence[float]) -> float:
    return float(np.median([float(value) for value in values]))


def aggregate_validation(
    per_arm_seed: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    arm_names: Sequence[str] = ARM_ORDER,
    seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    """Median-seed per-image deltas, paired effects, seed signs and Holm."""
    active = [name for name in arm_names if name in per_arm_seed]
    per_arm: dict[str, Any] = {}
    for arm_name in active:
        per_seed = per_arm_seed[arm_name]
        arm_seeds = sorted(per_seed)
        per_arm[arm_name] = {
            "arm": arm_name,
            "target": ARM_SPEC[arm_name]["target"],
            "seeds": [int(seed) for seed in arm_seeds],
            "per_seed": {
                str(seed): {
                    "correct_spearman": float(
                        per_seed[seed]["correct_spearman"]
                    ),
                    "correct_ndcg_full": float(
                        per_seed[seed]["correct_ndcg_full"]
                    ),
                    "same_spearman": float(per_seed[seed]["same_spearman"]),
                    "same_ndcg_full": float(per_seed[seed]["same_ndcg_full"]),
                    "cross_spearman": float(per_seed[seed]["cross_spearman"]),
                    "cross_ndcg_full": float(per_seed[seed]["cross_ndcg_full"]),
                    "exact_prompt_invariance_fraction": float(
                        per_seed[seed]["exact_prompt_invariance_fraction"]
                    ),
                    "tolerance_prompt_invariance_fraction": float(
                        per_seed[seed]["tolerance_prompt_invariance_fraction"]
                    ),
                    "selector_mass_k48": float(
                        np.mean(
                            list(
                                per_seed[seed]["image_metrics"][
                                    "selector_mass_k48"
                                ].values()
                            )
                        )
                    ),
                    "l2_anchor_mass_k48": float(
                        np.mean(
                            list(
                                per_seed[seed]["image_metrics"][
                                    "l2_anchor_mass_k48"
                                ].values()
                            )
                        )
                    ),
                }
                for seed in arm_seeds
            },
            "correct_alignment": {
                "spearman_per_seed": [
                    float(per_seed[seed]["correct_spearman"]) for seed in arm_seeds
                ],
                "ndcg_full_per_seed": [
                    float(per_seed[seed]["correct_ndcg_full"])
                    for seed in arm_seeds
                ],
                "spearman_median_seed": _median(
                    [per_seed[seed]["correct_spearman"] for seed in arm_seeds]
                ),
                "ndcg_full_median_seed": _median(
                    [per_seed[seed]["correct_ndcg_full"] for seed in arm_seeds]
                ),
            },
            "collapse": {
                "exact_prompt_invariance_fraction_per_seed": [
                    float(per_seed[seed]["exact_prompt_invariance_fraction"])
                    for seed in arm_seeds
                ],
                "max_exact_prompt_invariance_fraction": float(
                    max(
                        per_seed[seed]["exact_prompt_invariance_fraction"]
                        for seed in arm_seeds
                    )
                ),
            },
        }

    primary: dict[str, Any] = {}
    family: dict[str, float] = {}
    if CONTROL_ARM in active and PRIMARY_ARM in active:
        control = per_arm_seed[CONTROL_ARM]
        primary_stats = per_arm_seed[PRIMARY_ARM]
        for test in PRIMARY_TESTS:
            metric = str(test["metric"])
            base = str(test["base"])
            images = sorted(primary_stats[seeds[0]]["image_metrics"][metric])
            per_seed_delta: dict[str, dict[str, float]] = {}
            seed_means: dict[str, float] = {}
            positive_seed_count = 0
            for seed in seeds:
                b_metrics = primary_stats[seed]["image_metrics"]
                if test["kind"] == "arm_delta":
                    a_metrics = control[seed]["image_metrics"]
                    deltas = {
                        image: float(b_metrics[metric][image])
                        - float(a_metrics[base][image])
                        for image in images
                    }
                else:
                    deltas = {
                        image: float(b_metrics[base][image])
                        - float(b_metrics[metric][image])
                        for image in images
                    }
                per_seed_delta[str(seed)] = deltas
                seed_mean = float(np.mean(list(deltas.values())))
                seed_means[str(seed)] = seed_mean
                if seed_mean > 0.0:
                    positive_seed_count += 1
            median_deltas = [
                _median([per_seed_delta[str(seed)][image] for seed in seeds])
                for image in images
            ]
            effect = paired_effect(
                median_deltas,
                seed=STATISTICS_SEED,
                n_bootstrap=BOOTSTRAP_RESAMPLES,
                n_permutation=PERMUTATION_RESAMPLES,
                sample_count=int(
                    sum(
                        primary_stats[seed]["count"] for seed in seeds
                    )
                    // max(len(seeds), 1)
                ),
                independent_image_units=len(images),
            )
            entry = {
                "test": str(test["key"]),
                "definition": str(test["definition"]),
                "kind": str(test["kind"]),
                "metric": metric,
                "base": base,
                "independent_unit": "image",
                "aggregation": (
                    "per-image median delta across the three seeds then paired effect"
                ),
                "seed_means": seed_means,
                "positive_seed_count": int(positive_seed_count),
                "seed_count": len(seeds),
                "median_deltas": {
                    image: value for image, value in zip(images, median_deltas)
                },
                **effect,
            }
            primary[str(test["key"])] = entry
            family[str(test["key"])] = float(entry["p_value_two_sided"])
        adjusted = holm_bonferroni(family)
        for key, value in adjusted.items():
            primary[key]["holm_adjusted_p"] = float(value)
            primary[key]["holm_family_size"] = HOLM_FAMILY_SIZE
    return {
        "holm_family_size": HOLM_FAMILY_SIZE,
        "holm_family_definition": (
            "exactly 5 preregistered tests: B-A correct Spearman and full NDCG; "
            "B correct minus same-image-other-task Spearman and full NDCG; "
            "B deployed selector minus L2+anchor retained target mass at K48"
        ),
        "statistics_seed": STATISTICS_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "permutation_resamples": PERMUTATION_RESAMPLES,
        "arms": per_arm,
        "primary_tests": primary,
    }


# ---------------------------------------------------------------------------
# Gates and decision
# ---------------------------------------------------------------------------
def _delta_gate(entry: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "mean_positive": bool(
            entry.get("mean") is not None and float(entry["mean"]) > 0.0
        ),
        "ci_low_positive": bool(
            entry.get("ci_low") is not None and float(entry["ci_low"]) > 0.0
        ),
        "effect_size": bool(
            entry.get("cohens_dz") is not None
            and abs(float(entry["cohens_dz"])) >= EFFECT_SIZE_FLOOR
        ),
        "holm_significant": bool(
            entry.get("holm_adjusted_p") is not None
            and float(entry["holm_adjusted_p"]) < ALPHA
        ),
        "seed_sign": bool(
            int(entry.get("positive_seed_count", 0)) >= POSITIVE_SEED_MIN
        ),
    }


def hardware_contract() -> dict[str, Any]:
    """G6: query stays [1,128], FPGA scorer stays Gx128 dot, no token-axis exp."""
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
        "anchor_region_grid": int(REGION_GRID),
        "anchor_region_grid_ok": int(REGION_GRID) == ANCHOR_REGION_GRID,
        "fixed_budget_budgets": list(BUDGETS),
    }


def evaluate_gates(
    *,
    validation: Mapping[str, Any],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    """G1-G6 for arm B (the only arm that can win)."""
    primary = validation.get("primary_tests", {})
    g1_entries = {
        metric: _delta_gate(primary[key])
        for key, metric in (
            ("B_minus_A_correct_spearman", "spearman"),
            ("B_minus_A_correct_ndcg_full", "ndcg_full"),
        )
        if key in primary
    }
    g2_entries = {
        metric: _delta_gate(primary[key])
        for key, metric in (
            ("B_correct_minus_same_spearman", "spearman"),
            ("B_correct_minus_same_ndcg_full", "ndcg_full"),
        )
        if key in primary
    }
    g3_entry = (
        _delta_gate(primary["B_selector_minus_L2_anchor_mass_k48"])
        if "B_selector_minus_L2_anchor_mass_k48" in primary
        else None
    )
    g1 = bool(
        len(g1_entries) == 2
        and all(all(flags.values()) for flags in g1_entries.values())
    )
    g2 = bool(
        len(g2_entries) == 2
        and all(all(flags.values()) for flags in g2_entries.values())
    )
    g3 = bool(g3_entry is not None and all(g3_entry.values()))

    collapse = validation["arms"].get(PRIMARY_ARM, {}).get("collapse", {})
    max_invariance = collapse.get("max_exact_prompt_invariance_fraction")
    g4 = bool(
        max_invariance is not None
        and float(max_invariance) <= COLLAPSE_INVARIANCE_MAX
    )

    per_seed = list(training.get("per_seed", []))
    seed_improved = [
        int(entry["best_epoch"]) > 0
        and float(entry["best_val_kl"]) < float(entry["epoch0_val_kl"])
        for entry in per_seed
    ]
    g5 = bool(
        len(seed_improved) > 0
        and 3 * sum(1 for value in seed_improved if value) >= 2 * len(seed_improved)
    )

    hardware = hardware_contract()
    g6 = bool(
        hardware["query_shape_ok"]
        and hardware["fpga_scorer_dot_mac_per_group"] == 128
        and hardware["fpga_scorer_compare_per_group"] == 0
        and hardware["fpga_scorer_exp_per_group"] == 0
        and hardware["token_axis_max_softmax_exp"] is False
        and hardware["anchor_region_grid_ok"]
    )
    return {
        "arm": PRIMARY_ARM,
        "G1_spatial_supervision_gain": {"pass": g1, "entries": g1_entries},
        "G2_prompt_sensitivity": {"pass": g2, "entries": g2_entries},
        "G3_deployed_K48_utility": {"pass": g3, "entry": g3_entry},
        "G4_collapse_removed": {
            "pass": g4,
            "max_exact_prompt_invariance_fraction": max_invariance,
            "threshold": COLLAPSE_INVARIANCE_MAX,
        },
        "G5_validation_learning": {"pass": g5, "seed_improved": seed_improved},
        "G6_hardware_contract": {"pass": g6, "hardware": hardware},
        "all_pass": bool(g1 and g2 and g3 and g4 and g5 and g6),
    }


def decide(primary_gate: Mapping[str, Any] | None) -> dict[str, Any]:
    """Only arm B may win, and only on a full G1-G6 pass."""
    passed = bool(primary_gate is not None and primary_gate["all_pass"])
    return {
        "primary_arm": PRIMARY_ARM,
        "control_arm": CONTROL_ARM,
        "primary_arm_all_pass": passed,
        "primary_arm_evaluated": primary_gate is not None,
        "decision": (
            "SPATIAL_STUDENT_LEARNABLE"
            if passed
            else "SPATIAL_STUDENT_NOT_ESTABLISHED"
        ),
        "winner_rule": (
            "only arm B_SPATIAL_KL_PRIMARY may win, and only if it passes G1-G6; "
            "arm A is a control and can never win"
        ),
        "internal_test_evaluated": False,
    }


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------
def _resolve(path_like: Path | None, default: Path) -> Path:
    return Path(path_like) if path_like is not None else default


def run_experiment(
    *,
    prereg_config: Path = PREREG_CONFIG_PATH,
    prereg_report: Path = PREREG_REPORT_PATH,
    config: Path = CONFIG_PATH,
    index_path: Path = INDEX_PATH,
    manifest_path: Path = MANIFEST_PATH,
    lock_path: Path = LOCK_PATH,
    spatial_audit_json: Path = SPATIAL_AUDIT_JSON_PATH,
    spatial_audit_script: Path = SPATIAL_AUDIT_SCRIPT_PATH,
    semantic_control: Path = SEMANTIC_CONTROL_PATH,
    hardware_config: Path = HARDWARE_CONFIG_PATH,
    hardware_code: Path = HARDWARE_CODE_PATH,
    features_dir: Path | None = None,
    artifacts_dir: Path = ARTIFACTS_DIR,
    reports_dir: Path = REPORTS_DIR,
    seeds: Sequence[int] = SEEDS,
    epochs: int = EPOCHS,
    arm_names: Sequence[str] = ARM_ORDER,
    budgets: Sequence[int] = BUDGETS,
    lambda_semantic: float = LAMBDA_SEMANTIC,
    enforce_frozen_inputs: bool = True,
    enforce_expected_counts: bool = True,
    cache_loader: Callable[..., dict[str, Any]] | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Run the full preregistered two-arm experiment and return the bundle."""
    started = time.time()
    prereg = load_prereg(prereg_config)
    before_hashes = {
        "prereg_config": sha256_file(prereg_config),
        "prereg_report": sha256_file(prereg_report),
        "config": sha256_file(config),
        "index": sha256_file(index_path),
        "manifest": sha256_file(manifest_path),
        "lock": sha256_file(lock_path),
        "spatial_audit_json": sha256_file(spatial_audit_json),
        "spatial_audit_script": sha256_file(spatial_audit_script),
        "semantic_control": sha256_file(semantic_control),
        "hardware_config": sha256_file(hardware_config),
        "hardware_code": sha256_file(hardware_code),
    }
    verification: dict[str, Any] = {}
    if enforce_frozen_inputs:
        verification = verify_frozen_inputs(
            prereg_config=prereg_config,
            prereg_report=prereg_report,
            config=config,
            index=index_path,
            manifest=manifest_path,
            lock=lock_path,
            spatial_audit_json=spatial_audit_json,
            spatial_audit_script=spatial_audit_script,
            semantic_control=semantic_control,
            hardware_config=hardware_config,
            hardware_code=hardware_code,
        )

    index_records = load_index_metadata(index_path)
    split_rule = prereg.get("split") or prereg["split_rule"]
    splits = split_sample_ids(
        index_records,
        seed=int(split_rule["seed"]),
        ratios=split_rule["ratios"],
        salt=str(split_rule["salt"]),
    )
    image_hashes = {
        record["sample_id"]: record["source_image_sha256"]
        for record in index_records
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

    manifest_by_id, _manifest_ids = load_clean_manifest(manifest_path)

    opened: set[str] = set()
    features_directory = (
        Path(features_dir) if features_dir else Path(index_path).parent
    )
    train_samples_raw, train_aggregate, train_paths = load_split_samples(
        index_path,
        features_dir,
        splits["train"],
        opened=opened,
        cache_loader=cache_loader,
    )
    val_samples_raw, val_aggregate, val_paths = load_split_samples(
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
            f"internal test NPZ files were opened: {test_npz_opened}"
        )
    if enforce_frozen_inputs and train_aggregate != TRAIN_NPZ_AGGREGATE_SHA256:
        raise DistillationContractError(
            f"train NPZ aggregate {train_aggregate} != frozen "
            f"{TRAIN_NPZ_AGGREGATE_SHA256}"
        )

    train_samples = attach_spatial_targets(train_samples_raw, manifest_by_id)
    val_samples = attach_spatial_targets(val_samples_raw, manifest_by_id)
    train_pairs = build_image_pairs(train_samples)
    val_pairs = build_image_pairs(val_samples)
    tensors = {
        str(sample["sample_id"]): _tensorize(sample)
        for sample in [*train_samples, *val_samples]
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
                val_samples,
                result["best_Wt"],
                result["best_Wv"],
                budgets=budgets,
                lambda_semantic=lambda_semantic,
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

    epoch0_consistent = {
        seed: len(set(values.values())) == 1
        for seed, values in epoch0_hashes.items()
    }
    if not all(epoch0_consistent.values()):
        raise DistillationContractError(
            f"epoch-0 weight hashes differ across arms: {dict(epoch0_hashes)}"
        )

    aggregated = aggregate_validation(
        per_arm_seed_stats, arm_names=arm_names, seeds=seeds
    )
    training_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in training_results:
        training_by_arm[result["arm"]].append(result)
    arm_training: dict[str, Any] = {}
    for arm_name in arm_names:
        entries = sorted(training_by_arm[arm_name], key=lambda item: item["seed"])
        arm_training[arm_name] = {
            "arm": arm_name,
            "per_seed": entries,
            "best_epochs": [entry["best_epoch"] for entry in entries],
            "best_val_kls": [entry["best_val_kl"] for entry in entries],
            "epoch0_val_kls": [entry["epoch0_val_kl"] for entry in entries],
            "final_val_kls": [entry["final_val_kl"] for entry in entries],
            "epoch0_weights_sha256_per_seed": {
                str(entry["seed"]): entry["epoch0_weights_sha256"]
                for entry in entries
            },
        }

    primary_gate = (
        evaluate_gates(
            validation=aggregated,
            training=arm_training[PRIMARY_ARM],
        )
        if PRIMARY_ARM in arm_names
        else None
    )
    decision = decide(primary_gate)

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

    after_hashes = {
        "prereg_config": sha256_file(prereg_config),
        "prereg_report": sha256_file(prereg_report),
        "config": sha256_file(config),
        "index": sha256_file(index_path),
        "manifest": sha256_file(manifest_path),
        "lock": sha256_file(lock_path),
        "spatial_audit_json": sha256_file(spatial_audit_json),
        "spatial_audit_script": sha256_file(spatial_audit_script),
        "semantic_control": sha256_file(semantic_control),
        "hardware_config": sha256_file(hardware_config),
        "hardware_code": sha256_file(hardware_code),
    }
    unchanged = {
        key: bool(before_hashes[key] == after_hashes[key]) for key in before_hashes
    }
    if enforce_frozen_inputs:
        post = verify_frozen_inputs(
            prereg_config=prereg_config,
            prereg_report=prereg_report,
            config=config,
            index=index_path,
            manifest=manifest_path,
            lock=lock_path,
            spatial_audit_json=spatial_audit_json,
            spatial_audit_script=spatial_audit_script,
            semantic_control=semantic_control,
            hardware_config=hardware_config,
            hardware_code=hardware_code,
        )
    else:
        post = {}

    bundle = {
        "status": decision["decision"],
        "report_stem": REPORT_STEM,
        "scope_note": SCOPE_NOTE,
        "preregistered_config": str(prereg_config),
        "preregistration_report": str(prereg_report),
        "prereg_config_sha256": after_hashes["prereg_config"],
        "prereg_report_sha256": after_hashes["prereg_report"],
        "before_hashes": before_hashes,
        "after_hashes": after_hashes,
        "hashes_unchanged": unchanged,
        "prereg_unchanged": bool(
            unchanged["prereg_config"] and unchanged["prereg_report"]
        ),
        "frozen_input_verification": verification,
        "post_run_frozen_inputs": post,
        "train_npz_aggregate_sha256": train_aggregate,
        "val_npz_aggregate_sha256": val_aggregate,
        "source_sha256": {"trainer": sha256_file(Path(__file__))},
        "split": splits,
        "split_group_summary": group_summary,
        "split_group_by": "source_image_sha256",
        "isolation": {
            "train_npz_count": len(train_paths),
            "val_npz_count": len(val_paths),
            "opened_npz_count": len(opened),
            "test_npz_opened": test_npz_opened,
            "opened_only_train_val_npz": len(opened)
            == len(train_paths) + len(val_paths),
            "test_npz_opened_count": 0,
            "test_used_for_selection": False,
        },
        "arms": {arm["name"]: arm for arm in ARMS if arm["name"] in arm_names},
        "seeds": [int(seed) for seed in seeds],
        "epochs": int(epochs),
        "budgets": [int(budget) for budget in budgets],
        "lambda_semantic": float(lambda_semantic),
        "epoch0_hashes": {seed: dict(values) for seed, values in epoch0_hashes.items()},
        "epoch0_hashes_consistent_across_arms": epoch0_consistent,
        "training": arm_training,
        "validation_statistics": aggregated,
        "val_sample_rows": val_sample_rows,
        "primary_gate": primary_gate,
        "decision": decision,
        "runtime_seconds": float(runtime_seconds),
        "peak_rss_mib": peak_rss_mib,
        "hardware_used": {"gpu": False, "model_4b": False, "network": False},
        "limitations": [
            "internal train/val sanity only; not CV-Bench task accuracy",
            "no hardware synthesis, latency or speedup claim",
            "no 4B model load, no feature re-extraction, no raw image read",
            "the internal test split was never opened or evaluated",
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
                for split_name in ("train", "val"):
                    stats = row[split_name]
                    rows.append(
                        {
                            "arm": arm_name,
                            "seed": entry["seed"],
                            "epoch": row["epoch"],
                            "phase": row["phase"],
                            "split": split_name,
                            "kl": stats["kl"],
                            "kl_min": stats["kl_min"],
                            "kl_max": stats["kl_max"],
                            "pair_count": stats["pair_count"],
                            "grad_norm_before_clip": row["grad_norm_before_clip"]
                            if split_name == "train"
                            else None,
                            "batch_count": row["batch_count"]
                            if split_name == "train"
                            else None,
                        }
                    )
    return rows


def write_reports(
    bundle: Mapping[str, Any],
    *,
    artifacts_dir: Path = ARTIFACTS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, str]:
    del artifacts_dir  # artifacts are written by train_arm; kept for symmetry
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
            "kl",
            "kl_min",
            "kl_max",
            "pair_count",
            "grad_norm_before_clip",
            "batch_count",
        ],
    )
    paths["epochs_csv"] = str(epoch_csv)

    val_fields = [
        "arm",
        "seed",
        "sample_id",
        "image_sha256",
        "task",
        "group_count",
        "correct_spearman",
        "correct_ndcg_full",
        "same_spearman",
        "same_ndcg_full",
        "cross_spearman",
        "cross_ndcg_full",
        "correct_mass_k48",
        "correct_mass_k64",
        "correct_mass_k96",
        "same_mass_k48",
        "cross_mass_k48",
        "selector_mass_k48",
        "selector_mass_k64",
        "selector_mass_k96",
        "selector_ndcg_k48",
        "l2_anchor_mass_k48",
        "l2_anchor_mass_k64",
        "l2_anchor_mass_k96",
        "l2_anchor_ndcg_k48",
        "exact_prompt_invariant",
        "tolerance_prompt_invariant",
        "max_abs_diff",
        "delta_same_spearman",
        "delta_same_ndcg_full",
        "delta_cross_spearman",
        "delta_cross_ndcg_full",
    ]
    val_csv = reports_dir / f"{REPORT_STEM}-val-samples.csv"
    _write_csv(val_csv, bundle.get("val_sample_rows", []), val_fields)
    paths["val_samples_csv"] = str(val_csv)

    json_path = reports_dir / f"{REPORT_STEM}.json"
    _write_json(json_path, bundle)
    paths["json"] = str(json_path)

    md_path = reports_dir / f"{REPORT_STEM}.md"
    md_path.write_text(render_markdown(bundle), encoding="utf-8")
    paths["md"] = str(md_path)
    return paths


def render_markdown(bundle: Mapping[str, Any]) -> str:
    decision = bundle["decision"]
    lines: list[str] = []
    lines.append(f"# Spatial-supervision student train/val report ({REPORT_STEM})")
    lines.append("")
    lines.append(f"- status: **{bundle['status']}**")
    lines.append(f"- scope: {bundle['scope_note']}")
    lines.append(
        f"- prereg config SHA256: `{bundle['prereg_config_sha256']}` "
        f"(unchanged={bundle['prereg_unchanged']})"
    )
    lines.append(f"- prereg report SHA256: `{bundle['prereg_report_sha256']}`")
    lines.append(f"- before/after hashes unchanged: `{bundle['hashes_unchanged']}`")
    lines.append(
        f"- train NPZ aggregate SHA256: `{bundle['train_npz_aggregate_sha256']}`"
    )
    lines.append(
        f"- val NPZ aggregate SHA256: `{bundle['val_npz_aggregate_sha256']}`"
    )
    lines.append(
        f"- test_npz_opened = `{bundle['isolation']['test_npz_opened']}` "
        f"(count={bundle['isolation']['test_npz_opened_count']})"
    )
    lines.append(
        f"- split: train={bundle['split_group_summary']['train']} "
        f"val={bundle['split_group_summary']['val']} "
        f"test={bundle['split_group_summary']['test']}"
    )
    lines.append(
        f"- runtime_seconds={bundle['runtime_seconds']:.2f} "
        f"peak_rss_mib={bundle['peak_rss_mib']}"
    )
    lines.append(
        f"- decision: **{decision['decision']}**; primary arm `{decision['primary_arm']}` "
        f"all_pass={decision['primary_arm_all_pass']}"
    )
    lines.append("")
    lines.append("## Arm / seed losses")
    lines.append("")
    lines.append(
        "| arm | seed | epoch0 val KL | best epoch | best val KL | final val KL | "
        "epoch0 hash | best hash |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm_name, training in bundle["training"].items():
        for entry in training["per_seed"]:
            lines.append(
                f"| {arm_name} | {entry['seed']} | {entry['epoch0_val_kl']:.6f} | "
                f"{entry['best_epoch']} | {entry['best_val_kl']:.6f} | "
                f"{entry['final_val_kl']:.6f} | "
                f"`{entry['epoch0_weights_sha256'][:16]}` | "
                f"`{entry['best_weights_sha256'][:16]}` |"
            )
    lines.append("")
    lines.append("## Primary five-test Holm family (image-level)")
    lines.append("")
    lines.append(
        "| test | mean | ci_low | ci_high | dz | raw p | Holm p | +seeds |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for key, entry in bundle["validation_statistics"]["primary_tests"].items():
        lines.append(
            f"| {key} | {entry['mean']:.6f} | {entry['ci_low']:.6f} | "
            f"{entry['ci_high']:.6f} | {entry['cohens_dz']:.4f} | "
            f"{entry['p_value_two_sided']:.5f} | {entry['holm_adjusted_p']:.5f} | "
            f"{entry['positive_seed_count']}/{entry['seed_count']} |"
        )
    lines.append("")
    lines.append("## Gates G1-G6 (arm B only)")
    lines.append("")
    gate = bundle.get("primary_gate")
    if gate is None:
        lines.append("- primary arm not evaluated in this run")
    else:
        lines.append(
            f"- G1={gate['G1_spatial_supervision_gain']['pass']} "
            f"G2={gate['G2_prompt_sensitivity']['pass']} "
            f"G3={gate['G3_deployed_K48_utility']['pass']} "
            f"G4={gate['G4_collapse_removed']['pass']} "
            f"G5={gate['G5_validation_learning']['pass']} "
            f"G6={gate['G6_hardware_contract']['pass']} "
            f"all_pass={gate['all_pass']}"
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
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument(
        "--harness-result",
        type=Path,
        default=Path("/tmp/robobrain_spatial_student_harness_result.json"),
    )
    args = parser.parse_args(argv)
    bundle = run_experiment(
        prereg_config=args.prereg_config,
        prereg_report=args.prereg_report,
        index_path=args.index,
        manifest_path=args.manifest,
        features_dir=args.features_dir,
        artifacts_dir=args.artifacts_dir,
        reports_dir=args.reports_dir,
        seeds=args.seeds,
        epochs=args.epochs,
    )
    summary = {
        "status": bundle["status"],
        "decision": bundle["decision"]["decision"],
        "prereg_sha256": bundle["prereg_config_sha256"],
    }
    args.harness_result.parent.mkdir(parents=True, exist_ok=True)
    args.harness_result.write_text(
        json.dumps(_harness_payload(bundle), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


def _primary_deltas(bundle: Mapping[str, Any]) -> dict[str, Any]:
    tests = bundle.get("validation_statistics", {}).get("primary_tests", {})
    return {
        key: {
            "mean": entry.get("mean"),
            "ci_low": entry.get("ci_low"),
            "ci_high": entry.get("ci_high"),
            "cohens_dz": entry.get("cohens_dz"),
            "p_value_two_sided": entry.get("p_value_two_sided"),
            "holm_adjusted_p": entry.get("holm_adjusted_p"),
            "positive_seed_count": entry.get("positive_seed_count"),
            "seed_count": entry.get("seed_count"),
        }
        for key, entry in tests.items()
    }


def _harness_payload(bundle: Mapping[str, Any]) -> dict[str, Any]:
    gate = bundle.get("primary_gate")
    return {
        "status": bundle["status"],
        "decision": bundle["decision"]["decision"],
        "prereg_sha256": bundle["prereg_config_sha256"],
        "prereg_report_sha256": bundle["prereg_report_sha256"],
        "hashes_unchanged": bundle["hashes_unchanged"],
        "source_sha256": bundle["source_sha256"],
        "arms": list(bundle["arms"]),
        "seeds": bundle["seeds"],
        "train_records": bundle["split_group_summary"]["train"]["sample_count"],
        "val_records": bundle["split_group_summary"]["val"]["sample_count"],
        "test_npz_opened": bundle["isolation"]["test_npz_opened"],
        "best_epochs": {
            arm: training["best_epochs"]
            for arm, training in bundle["training"].items()
        },
        "primary_deltas": _primary_deltas(bundle),
        "gates": (
            {
                key: gate[key]["pass"]
                for key in (
                    "G1_spatial_supervision_gain",
                    "G2_prompt_sensitivity",
                    "G3_deployed_K48_utility",
                    "G4_collapse_removed",
                    "G5_validation_learning",
                    "G6_hardware_contract",
                )
            }
            if gate is not None
            else None
        ),
        "runtime_seconds": bundle["runtime_seconds"],
        "peak_rss_mib": bundle["peak_rss_mib"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
