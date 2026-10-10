#!/usr/bin/env python3
"""Preregistered prompt-contrastive spatial train/val experiment for the selector.

Motivation: the prior preregistered spatial-supervision experiment
(:mod:`evaluation.train_spatial_prompt_selector`, read-only reuse source) passed
correct-prompt spatial alignment but failed the same-image other-task prompt
discrimination gate (G2).  This experiment adds a **fixed-weight** contrastive
term to the spatial KL base loss without changing the frozen ``prompt_mean_vector``
hardware structure.

Two arms, only B may pass:

* ``A_SPATIAL_KL_CONTROL`` -- ``base = 0.5 (KL(tg||pg_correct) + KL(tp||pp_correct))``.
* ``B_SPATIAL_KL_CONTRASTIVE`` -- ``total = base + 1.0 * contrastive`` where
  ``contrastive = 0.5 (softplus(KL_g_correct - KL_g_wrong) + softplus(KL_p_correct
  - KL_p_wrong))``; a grounding record's *wrong* prompt is the same image's
  pointing prompt and vice versa.  Every KL is ``KL(target || softmax(score/0.1))``
  with no ``T^2`` scaling.

Arm A selects its best checkpoint by val ``base``; arm B by val ``total`` (base,
contrastive and total are all reported).

Hard boundaries (enforced and unit-tested):

* **train/val feature caches only** -- the internal test NPZ files are never
  resolved, hashed, opened or scored.
* **CPU-only float32** -- about 458,752 trainable parameters in a small
  ``Wt[128,2560]``/``Wv[128,1024]`` head; this is **not** 4B model training.
* **no 4B model, no feature re-extraction, no raw image read, no CV-Bench, no
  network, no package installation**.
* the existing :mod:`evaluation.train_spatial_prompt_selector` and every existing
  report are read-only and byte-for-byte unchanged.

Reuse (no second implementation of any split / loader / target / scorer /
checkpoint / statistic / anchor / combine / selected-set NDCG):

* split / loader:   :func:`evaluation.prompt_projection_distillation.split_sample_ids`
                    and :func:`evaluation.train_pairwise_prompt_selector.load_split_samples`
* image pairs:      :func:`evaluation.train_spatial_prompt_selector.build_image_pairs`
* spatial targets:  :func:`evaluation.train_spatial_prompt_selector.attach_spatial_targets`
* scorer:           :func:`evaluation.pair_loss_feasibility_audit.student_scores`
                    (``prompt_mean_vector``)
* losses:           :func:`evaluation.train_spatial_prompt_selector.record_kl`
* validation:       :func:`evaluation.train_spatial_prompt_selector.val_statistics`
* statistics:       :func:`evaluation.prompt_identifiability_diagnostic.paired_effect`
                    and :func:`evaluation.frozen_scorer_reduction_ablation.holm_bonferroni`
* hardware:         :func:`evaluation.frozen_scorer_reduction_ablation.hardware_contract_for`
* checkpointing:    :func:`evaluation.prompt_projection_distillation.save_projection_npz`
"""

from __future__ import annotations

import argparse
import csv
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
)
from evaluation.hardware_regular_token_selector import REGION_GRID
from evaluation.prompt_identifiability_diagnostic import paired_effect
from evaluation.prompt_projection_distillation import (
    DistillationContractError,
    canonical_projection_sha256,
    save_projection_npz,
    split_group_summary,
    split_sample_ids,
)
from evaluation.spatial_teacher_feasibility_audit import load_clean_manifest
from evaluation.teacher_pair_signal_audit import load_index_metadata
from evaluation.train_pairwise_prompt_selector import (
    epoch_batch_order,
    load_split_samples,
    select_best_epoch,
    sha256_file,
)
from evaluation.train_prompt_projection import init_projection
from evaluation.train_spatial_prompt_selector import (
    _tensorize,
    attach_spatial_targets,
    build_image_pairs,
    record_kl,
    val_statistics,
)

try:  # pragma: no cover - exercised only when torch is missing
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def _require_torch() -> Any:
    if not TORCH_AVAILABLE:
        raise DistillationContractError("torch is required for the contrastive trainer")
    return torch


# ---------------------------------------------------------------------------
# Frozen constants (cross-checked against the preregistered config at run time)
# ---------------------------------------------------------------------------
REPORT_STEM = "2026-10-03-spatial-prompt-contrastive-trainval"
PREREG_CONFIG_PATH = Path("configs/spatial_prompt_contrastive_trainval.preregistered.json")
PREREG_REPORT_PATH = Path(
    "reports/token_selector/2026-10-03-spatial-prompt-contrastive-trainval-preregistration.json"
)
HARNESS_PROMPT_PATH = Path(
    "reports/token_selector/prompts/2026-10-03-spatial-prompt-contrastive-trainval-harness-prompt.md"
)
CONFIG_PATH = Path("configs/prompt_projection_distillation.json")
INDEX_PATH = Path("artifacts/token_selector/prompt_projection_features/index.jsonl")
MANIFEST_PATH = Path("datasets/prompt_projection_clean_inductive_v1.jsonl")
LOCK_PATH = Path("datasets/prompt_projection_clean_inductive_v1.lock.json")
SPATIAL_AUDIT_JSON_PATH = Path(
    "reports/token_selector/2026-10-03-spatial-teacher-feasibility.json"
)
SPATIAL_AUDIT_SCRIPT_PATH = Path("evaluation/spatial_teacher_feasibility_audit.py")
HARDWARE_CONFIG_PATH = Path("configs/hardware_regular_token_selector.json")
HARDWARE_CODE_PATH = Path("evaluation/hardware_regular_token_selector.py")
EXISTING_SPATIAL_TRAINER_PATH = Path("evaluation/train_spatial_prompt_selector.py")
PAIR_LOSS_AUDIT_PATH = Path("evaluation/pair_loss_feasibility_audit.py")
DISTILLATION_CODE_PATH = Path("evaluation/prompt_projection_distillation.py")
IDENTIFIABILITY_CODE_PATH = Path("evaluation/prompt_identifiability_diagnostic.py")
REDUCTION_CODE_PATH = Path("evaluation/frozen_scorer_reduction_ablation.py")
SEMANTIC_CONTROL_PATH = Path("evaluation/train_pairwise_prompt_selector.py")
ARTIFACTS_DIR = Path("artifacts/token_selector/spatial_prompt_contrastive_trainval")
REPORTS_DIR = Path("reports/token_selector")
LOGS_DIR = REPORTS_DIR / "logs"

PREREG_STATUS = "PREREGISTERED_TRAIN_VAL_ONLY"

# Frozen input SHA256 (also pinned inside the preregistered config).
FROZEN_INPUT_SHA256: dict[str, tuple[Path, str]] = {
    "config_sha256": (
        CONFIG_PATH,
        "44399c8e6fb8f333a150a6016faa00d6ae70524b5fcd030bcabf4e5fb07ab28b",
    ),
    "index_sha256": (
        INDEX_PATH,
        "02a50868a84675418e9bd9e5fdd0bb55f522ec5ce0ce1d6da7c69f85d2aeee26",
    ),
    "manifest_sha256": (
        MANIFEST_PATH,
        "acab1a322a93021138ce87bb048c1e25493ec19a3f56f70dc3a24c0686c79213",
    ),
    "lock_sha256": (
        LOCK_PATH,
        "a723458da9b9e170c9ffcde3bfa8844ae18c3e84fe1ba15bbd3d4b3f57e2e105",
    ),
    "spatial_audit_json_sha256": (
        SPATIAL_AUDIT_JSON_PATH,
        "1bd2b9e41f3e6d698548ff7ad85c9b0a0b513e0d3ef4d4b0c02cc259ffbcc739",
    ),
    "spatial_audit_script_sha256": (
        SPATIAL_AUDIT_SCRIPT_PATH,
        "1d033c4036d5f2319c8828e393594fe0b6330ac9c43afe43af14499adf748aec",
    ),
    "hardware_config_sha256": (
        HARDWARE_CONFIG_PATH,
        "6350c89bfc48bf59ae507bbacd80bda2eb3ba9f6cefbfde07ad1593c6e418142",
    ),
    "hardware_code_sha256": (
        HARDWARE_CODE_PATH,
        "c55cfdc39e798e7566cd609a77ac388dca85df463ec20f536448284e2f85932b",
    ),
    "existing_spatial_trainer_sha256": (
        EXISTING_SPATIAL_TRAINER_PATH,
        "1308e8a0f18cfb047aa56ab4e8fa40423ebd127f22df309361b3916eee389fd2",
    ),
    "pair_loss_audit_sha256": (
        PAIR_LOSS_AUDIT_PATH,
        "2913e43fadef1397f33a6d2f38ea093727ce2fc129378c330c0698e8c5a73dea",
    ),
    "distillation_code_sha256": (
        DISTILLATION_CODE_PATH,
        "cd2d4ccc3a4c5ba714f5e478f802b0cb9d64c6e279c15b6493661fb6fe0b0b98",
    ),
    "identifiability_code_sha256": (
        IDENTIFIABILITY_CODE_PATH,
        "f7311aa1a0ea1f82131a66dd4b38af930b93a527a743449069238924a0811bae",
    ),
    "reduction_code_sha256": (
        REDUCTION_CODE_PATH,
        "fef63fd2e6c82e3b4f38806f896940a8e0b1d5533feb51e99d97e145783c0729",
    ),
    "semantic_control_sha256": (
        SEMANTIC_CONTROL_PATH,
        "07b504d79743915b7b8dacf7028dea1de91ba36bb2776dedd6ed9e4012032fec",
    ),
}
TRAIN_NPZ_AGGREGATE_SHA256 = (
    "33997394c4e148e3cfcd45eb0eb37fc97616cece8db3da5c731171d4b53d3951"
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

CONTRASTIVE_WEIGHT = 1.0
CONTRASTIVE_MARGIN = 0.0
SOFTPLUS_TEMPERATURE = 1.0

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
NON_INFERIORITY_MEAN_FLOOR = -0.02
NON_INFERIORITY_CI_FLOOR = -0.05

ARM_ORDER: tuple[str, ...] = (
    "A_SPATIAL_KL_CONTROL",
    "B_SPATIAL_KL_CONTRASTIVE",
)
ARMS: tuple[dict[str, Any], ...] = (
    {
        "name": "A_SPATIAL_KL_CONTROL",
        "target": "spatial",
        "loss": "base",
        "role": "control",
        "can_win": False,
    },
    {
        "name": "B_SPATIAL_KL_CONTRASTIVE",
        "target": "spatial",
        "loss": "base+contrastive",
        "role": "primary_promotion_arm",
        "can_win": True,
    },
)
ARM_SPEC: dict[str, dict[str, Any]] = {str(arm["name"]): arm for arm in ARMS}
PRIMARY_ARM = "B_SPATIAL_KL_CONTRASTIVE"
CONTROL_ARM = "A_SPATIAL_KL_CONTROL"

PRIMARY_TESTS: tuple[dict[str, Any], ...] = (
    {
        "key": "P1_B_minus_A_correct_minus_same_spearman",
        "kind": "gap_arm_delta",
        "gap": "spearman",
        "definition": (
            "arm B minus arm A of the (correct minus same-image-other-task) "
            "spatial Spearman gap"
        ),
    },
    {
        "key": "P2_B_minus_A_correct_minus_same_ndcg_full",
        "kind": "gap_arm_delta",
        "gap": "ndcg_full",
        "definition": (
            "arm B minus arm A of the (correct minus same-image-other-task) "
            "spatial full NDCG gap"
        ),
    },
    {
        "key": "P3_B_correct_minus_same_spearman",
        "kind": "gap_absolute",
        "gap": "spearman",
        "definition": (
            "arm B absolute (correct minus same-image-other-task) spatial "
            "Spearman gap"
        ),
    },
    {
        "key": "P4_B_correct_minus_same_ndcg_full",
        "kind": "gap_absolute",
        "gap": "ndcg_full",
        "definition": (
            "arm B absolute (correct minus same-image-other-task) spatial full "
            "NDCG gap"
        ),
    },
    {
        "key": "P5_B_selector_minus_L2_anchor_mass_k48",
        "kind": "selector_minus_baseline",
        "metric": "selector_mass_k48",
        "base": "l2_anchor_mass_k48",
        "definition": (
            "arm B deployed selector minus L2+anchor retained spatial-target "
            "mass at K48 (selector minus baseline)"
        ),
    },
)
HOLM_FAMILY_SIZE = len(PRIMARY_TESTS)

GAP_SOURCES: dict[str, tuple[str, str]] = {
    "spearman": ("correct_spearman", "same_spearman"),
    "ndcg_full": ("correct_ndcg_full", "same_ndcg_full"),
}

SCOPE_NOTE = (
    "train/val development validation only, CPU-only float32, ~458752-parameter "
    "Wt/Wv head (NOT 4B training), no GPU, no 4B model, no feature re-extraction, "
    "no raw image read, no CV-Bench, no network, no internal test NPZ open; "
    "internal sanity only, not final or CV-Bench accuracy"
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
    harness_prompt: Path = HARNESS_PROMPT_PATH,
) -> dict[str, Any]:
    """Recompute every frozen SHA and fail closed on any drift."""
    actual = {
        key: sha256_file(path) for key, (path, _sha) in FROZEN_INPUT_SHA256.items()
    }
    expected = {
        key: sha for key, (_path, sha) in FROZEN_INPUT_SHA256.items()
    }
    checks = {f"{key}_unchanged": actual[key] == expected[key] for key in expected}
    checks["prereg_status"] = load_prereg(prereg_config)["status"] == PREREG_STATUS
    # The preregistration report pins this config's SHA256 and the prompt SHA256.
    report = json.loads(Path(prereg_report).read_text(encoding="utf-8"))
    pinned_config = str(report["preregistered_config"]["sha256"])
    actual_config = sha256_file(prereg_config)
    checks["prereg_report_pins_config"] = pinned_config == actual_config
    checks["prereg_report_status"] = str(report.get("status")) == PREREG_STATUS
    checks["prereg_report_pins_harness_prompt"] = (
        str(report["harness_prompt"]["sha256"]) == sha256_file(harness_prompt)
    )
    if not all(checks.values()):
        raise DistillationContractError(f"frozen input drift detected: {checks}")
    return {
        "actual": actual,
        "expected": expected,
        "checks": checks,
        "prereg_config_sha256": actual_config,
        "prereg_report_sha256": sha256_file(prereg_report),
        "harness_prompt_sha256": sha256_file(harness_prompt),
    }


# ---------------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------------
def softplus_contrastive(value: Any) -> Any:
    """Numerically stable ``softplus`` with the frozen margin and temperature."""
    t = _require_torch()
    scaled = (value - CONTRASTIVE_MARGIN) / SOFTPLUS_TEMPERATURE
    return SOFTPLUS_TEMPERATURE * t.logaddexp(scaled, t.zeros_like(scaled))


def pair_loss_components(
    tensors: Mapping[str, Any],
    pair: Mapping[str, Any],
    Wt: Any,
    Wv: Any,
    *,
    temperature: float = STUDENT_TEMPERATURE,
) -> dict[str, Any]:
    """All differentiable loss terms of one image pair.

    ``base = 0.5 (KL(tg||pg_correct) + KL(tp||pp_correct))`` and
    ``contrastive = 0.5 (softplus(KL_g_correct - KL_g_wrong) + softplus(
    KL_p_correct - KL_p_wrong))``.  The grounding record's wrong prompt is the
    same image's pointing prompt; the pointing record's wrong prompt is the same
    image's grounding prompt.  Every KL still uses the record's own target.
    """
    from evaluation.pair_loss_feasibility_audit import student_scores

    grounding = tensors[str(pair["grounding_id"])]
    pointing = tensors[str(pair["pointing_id"])]
    scores_g = student_scores(
        grounding["group_mean"], grounding["prompt_embeddings"], Wt, Wv
    )
    scores_p = student_scores(
        pointing["group_mean"], pointing["prompt_embeddings"], Wt, Wv
    )
    scores_g_wrong = student_scores(
        grounding["group_mean"], pointing["prompt_embeddings"], Wt, Wv
    )
    scores_p_wrong = student_scores(
        pointing["group_mean"], grounding["prompt_embeddings"], Wt, Wv
    )
    kl_g = record_kl(scores_g, grounding["spatial_target"], temperature=temperature)
    kl_p = record_kl(scores_p, pointing["spatial_target"], temperature=temperature)
    kl_g_wrong = record_kl(
        scores_g_wrong, grounding["spatial_target"], temperature=temperature
    )
    kl_p_wrong = record_kl(
        scores_p_wrong, pointing["spatial_target"], temperature=temperature
    )
    base = 0.5 * (kl_g + kl_p)
    contrastive = 0.5 * (
        softplus_contrastive(kl_g - kl_g_wrong)
        + softplus_contrastive(kl_p - kl_p_wrong)
    )
    return {
        "base": base,
        "contrastive": contrastive,
        "kl_g_correct": kl_g,
        "kl_g_wrong": kl_g_wrong,
        "kl_p_correct": kl_p,
        "kl_p_wrong": kl_p_wrong,
    }


def arm_total(components: Mapping[str, Any], arm_name: str) -> Any:
    """The differentiable training objective of one arm for one image pair."""
    if arm_name == CONTROL_ARM:
        return components["base"]
    if arm_name == PRIMARY_ARM:
        return components["base"] + CONTRASTIVE_WEIGHT * components["contrastive"]
    raise DistillationContractError(f"unknown arm {arm_name!r}")


def evaluate_split_losses(
    tensors: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    Wt: Any,
    Wv: Any,
    arm_name: str,
    *,
    temperature: float = STUDENT_TEMPERATURE,
) -> dict[str, Any]:
    """Mean base / contrastive / total of an arm over a split."""
    t = _require_torch()
    if not pairs:
        raise DistillationContractError("cannot evaluate an empty pair set")
    bases: list[float] = []
    contrastives: list[float] = []
    totals: list[float] = []
    with t.no_grad():
        for pair in pairs:
            components = pair_loss_components(
                tensors, pair, Wt, Wv, temperature=temperature
            )
            bases.append(float(components["base"]))
            contrastives.append(float(components["contrastive"]))
            totals.append(float(arm_total(components, arm_name)))
    return {
        "pair_count": len(pairs),
        "base": float(np.mean(bases)),
        "contrastive": float(np.mean(contrastives)),
        "total": float(np.mean(totals)),
        "base_min": float(np.min(bases)),
        "base_max": float(np.max(bases)),
        "contrastive_min": float(np.min(contrastives)),
        "contrastive_max": float(np.max(contrastives)),
        "total_min": float(np.min(totals)),
        "total_max": float(np.max(totals)),
    }


def arm_selection_value(stats: Mapping[str, Any], arm_name: str) -> float:
    """Val-only model-selection value: arm A uses base, arm B uses total."""
    if arm_name == CONTROL_ARM:
        return float(stats["base"])
    if arm_name == PRIMARY_ARM:
        return float(stats["total"])
    raise DistillationContractError(f"unknown arm {arm_name!r}")


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
    selection_values: list[float] = []
    best_val = math.inf
    best_epoch = -1
    best_state: dict[str, Any] | None = None

    for epoch in range(int(epochs)):
        train_stats = evaluate_split_losses(
            tensors, train_pairs, Wt, Wv, arm_name, temperature=temperature
        )
        val_stats = evaluate_split_losses(
            tensors, val_pairs, Wt, Wv, arm_name, temperature=temperature
        )
        selection = arm_selection_value(val_stats, arm_name)
        selection_values.append(selection)
        if selection < best_val:
            best_val = selection
            best_epoch = epoch
            best_state = {"Wt": Wt.detach().clone(), "Wv": Wv.detach().clone()}

        order = epoch_batch_order(seed, epoch, image_hashes, batch_size_images)
        grad_norms: list[float] = []
        batch_losses: list[float] = []
        for batch in order:
            optimizer.zero_grad(set_to_none=True)
            total = t.zeros((), dtype=t.float32)
            for pair_index in batch:
                components = pair_loss_components(
                    tensors,
                    train_pairs[pair_index],
                    Wt,
                    Wv,
                    temperature=temperature,
                )
                total = total + arm_total(components, arm_name)
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
                "selection_value": float(selection),
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
    selected_epoch = select_best_epoch(selection_values)
    if int(selected_epoch) != int(best_epoch):
        raise DistillationContractError(
            f"val-only selection mismatch: loop={best_epoch} select={selected_epoch}"
        )

    final_train = evaluate_split_losses(
        tensors, train_pairs, Wt, Wv, arm_name, temperature=temperature
    )
    final_val = evaluate_split_losses(
        tensors, val_pairs, Wt, Wv, arm_name, temperature=temperature
    )
    final_row = {
        "epoch": int(epochs),
        "phase": "final_post_update",
        "selection_value": float(arm_selection_value(final_val, arm_name)),
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
    best_stats = epoch_metrics[best_epoch]["val"]
    epoch0_stats = epoch_metrics[0]["val"]
    metadata = {
        "arm": arm_name,
        "target": ARM_SPEC[arm_name]["target"],
        "loss": ARM_SPEC[arm_name]["loss"],
        "seed": int(seed),
        "epochs": int(epochs),
        "epoch0_weights_sha256": epoch0_hash,
        "best_epoch": int(best_epoch),
        "selection_metric": "val_base" if arm_name == CONTROL_ARM else "val_total",
        "best_val_selection": float(best_val),
        "best_val_base": float(best_stats["base"]),
        "best_val_contrastive": float(best_stats["contrastive"]),
        "best_val_total": float(best_stats["total"]),
        "epoch0_val_base": float(epoch0_stats["base"]),
        "epoch0_val_contrastive": float(epoch0_stats["contrastive"]),
        "epoch0_val_total": float(epoch0_stats["total"]),
        "final_val_base": float(final_val["base"]),
        "final_val_contrastive": float(final_val["contrastive"]),
        "final_val_total": float(final_val["total"]),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "loss_definition": (
            "base = mean over image pairs of 0.5(KL_g_correct+KL_p_correct); "
            "contrastive = mean over image pairs of 0.5(softplus(KL_g_correct-"
            "KL_g_wrong)+softplus(KL_p_correct-KL_p_wrong)); total = base + "
            "1.0*contrastive for arm B and base for arm A"
        ),
        "contrastive_weight": CONTRASTIVE_WEIGHT,
        "contrastive_margin": CONTRASTIVE_MARGIN,
        "softplus_temperature": SOFTPLUS_TEMPERATURE,
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
        "loss": ARM_SPEC[arm_name]["loss"],
        "epoch0_weights_sha256": epoch0_hash,
        "best_epoch": int(best_epoch),
        "selection_metric": metadata["selection_metric"],
        "best_val_selection": float(best_val),
        "best_val_base": float(best_stats["base"]),
        "best_val_contrastive": float(best_stats["contrastive"]),
        "best_val_total": float(best_stats["total"]),
        "epoch0_val_base": float(epoch0_stats["base"]),
        "epoch0_val_contrastive": float(epoch0_stats["contrastive"]),
        "epoch0_val_total": float(epoch0_stats["total"]),
        "final_val_total": float(final_val["total"]),
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
# Cross-seed aggregation and the five-test Holm family
# ---------------------------------------------------------------------------
def _median(values: Sequence[float]) -> float:
    return float(np.median([float(value) for value in values]))


def _gap_image_metrics(
    image_metrics: Mapping[str, Mapping[str, float]], gap: str
) -> dict[str, float]:
    correct_key, same_key = GAP_SOURCES[gap]
    images = sorted(image_metrics[correct_key])
    return {
        image: float(image_metrics[correct_key][image])
        - float(image_metrics[same_key][image])
        for image in images
    }


def _per_seed_deltas(
    test: Mapping[str, Any],
    control: Mapping[int, Mapping[str, Any]],
    primary: Mapping[int, Mapping[str, Any]],
    seed: int,
    images: Sequence[str],
) -> dict[str, float]:
    b_metrics = primary[seed]["image_metrics"]
    if test["kind"] == "gap_arm_delta":
        gap_b = _gap_image_metrics(b_metrics, str(test["gap"]))
        gap_a = _gap_image_metrics(control[seed]["image_metrics"], str(test["gap"]))
        return {image: gap_b[image] - gap_a[image] for image in images}
    if test["kind"] == "gap_absolute":
        gap_b = _gap_image_metrics(b_metrics, str(test["gap"]))
        return {image: gap_b[image] for image in images}
    if test["kind"] == "selector_minus_baseline":
        metric = str(test["metric"])
        base = str(test["base"])
        return {
            image: float(b_metrics[metric][image]) - float(b_metrics[base][image])
            for image in images
        }
    raise DistillationContractError(f"unknown primary test kind {test['kind']!r}")


def _paired_entry(
    *,
    key: str,
    definition: str,
    per_seed_delta: Mapping[str, Mapping[str, float]],
    images: Sequence[str],
    seeds: Sequence[int],
    kind: str,
    metric: str | None,
    base: str | None,
    sample_count: int,
) -> dict[str, Any]:
    seed_means: dict[str, float] = {}
    positive_seed_count = 0
    for seed in seeds:
        deltas = per_seed_delta[str(seed)]
        seed_mean = float(np.mean([deltas[image] for image in images]))
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
        sample_count=int(sample_count),
        independent_image_units=len(images),
    )
    return {
        "test": key,
        "definition": definition,
        "kind": kind,
        "metric": metric,
        "base": base,
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
            "loss": ARM_SPEC[arm_name]["loss"],
            "seeds": [int(seed) for seed in arm_seeds],
            "per_seed": {
                str(seed): {
                    "correct_spearman": float(per_seed[seed]["correct_spearman"]),
                    "correct_ndcg_full": float(per_seed[seed]["correct_ndcg_full"]),
                    "same_spearman": float(per_seed[seed]["same_spearman"]),
                    "same_ndcg_full": float(per_seed[seed]["same_ndcg_full"]),
                    "cross_spearman": float(per_seed[seed]["cross_spearman"]),
                    "cross_ndcg_full": float(per_seed[seed]["cross_ndcg_full"]),
                    "gap_spearman": float(
                        per_seed[seed]["correct_spearman"]
                        - per_seed[seed]["same_spearman"]
                    ),
                    "gap_ndcg_full": float(
                        per_seed[seed]["correct_ndcg_full"]
                        - per_seed[seed]["same_ndcg_full"]
                    ),
                    "selector_mass_k48": float(
                        np.mean(
                            list(
                                per_seed[seed]["image_metrics"]["selector_mass_k48"].values()
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
                    "exact_prompt_invariance_fraction": float(
                        per_seed[seed]["exact_prompt_invariance_fraction"]
                    ),
                    "tolerance_prompt_invariance_fraction": float(
                        per_seed[seed]["tolerance_prompt_invariance_fraction"]
                    ),
                }
                for seed in arm_seeds
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
    noninferiority: dict[str, Any] = {}
    if CONTROL_ARM in active and PRIMARY_ARM in active:
        control = per_arm_seed[CONTROL_ARM]
        primary_stats = per_arm_seed[PRIMARY_ARM]
        images = sorted(primary_stats[seeds[0]]["image_metrics"]["correct_spearman"])
        avg_sample_count = int(
            sum(primary_stats[seed]["count"] for seed in seeds) // max(len(seeds), 1)
        )
        family: dict[str, float] = {}
        for test in PRIMARY_TESTS:
            per_seed_delta: dict[str, dict[str, float]] = {}
            for seed in seeds:
                per_seed_delta[str(seed)] = _per_seed_deltas(
                    test, control, primary_stats, seed, images
                )
            entry = _paired_entry(
                key=str(test["key"]),
                definition=str(test["definition"]),
                per_seed_delta=per_seed_delta,
                images=images,
                seeds=seeds,
                kind=str(test["kind"]),
                metric=str(test.get("metric")) if test.get("metric") else None,
                base=str(test.get("base")) if test.get("base") else None,
                sample_count=avg_sample_count,
            )
            primary[str(test["key"])] = entry
            family[str(test["key"])] = float(entry["p_value_two_sided"])
        adjusted = holm_bonferroni(family)
        for key, value in adjusted.items():
            primary[key]["holm_adjusted_p"] = float(value)
            primary[key]["holm_family_size"] = HOLM_FAMILY_SIZE

        # G1 descriptive non-inferiority of correct alignment (not in Holm).
        for metric_key in ("correct_spearman", "correct_ndcg_full"):
            per_seed_delta = {
                str(seed): {
                    image: float(primary_stats[seed]["image_metrics"][metric_key][image])
                    - float(control[seed]["image_metrics"][metric_key][image])
                    for image in images
                }
                for seed in seeds
            }
            entry = _paired_entry(
                key=f"G1_B_minus_A_{metric_key}",
                definition=f"arm B minus arm A correct-prompt {metric_key}",
                per_seed_delta=per_seed_delta,
                images=images,
                seeds=seeds,
                kind="arm_delta",
                metric=metric_key,
                base=metric_key,
                sample_count=avg_sample_count,
            )
            seed_means = {
                seed: float(value) for seed, value in entry["seed_means"].items()
            }
            noninferiority[metric_key] = {
                **entry,
                "mean_floor": NON_INFERIORITY_MEAN_FLOOR,
                "ci_lower_floor": NON_INFERIORITY_CI_FLOOR,
                "noninferior_seed_count": int(
                    sum(
                        1
                        for value in seed_means.values()
                        if value >= NON_INFERIORITY_MEAN_FLOOR
                    )
                ),
                "noninferior_seed_min": POSITIVE_SEED_MIN,
            }

    return {
        "holm_family_size": HOLM_FAMILY_SIZE,
        "holm_family_definition": (
            "exactly 5 preregistered tests: P1/P2 B-A (correct-same) Spearman and "
            "full NDCG gaps; P3/P4 B absolute (correct-same) Spearman and full "
            "NDCG gaps; P5 B deployed selector minus L2+anchor retained target "
            "mass at K48"
        ),
        "p5_sign_rule": "delta = selector_mass_k48 - l2_anchor_mass_k48",
        "statistics_seed": STATISTICS_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "permutation_resamples": PERMUTATION_RESAMPLES,
        "arms": per_arm,
        "primary_tests": primary,
        "correct_alignment_noninferiority": noninferiority,
    }


# ---------------------------------------------------------------------------
# Gates and decision
# ---------------------------------------------------------------------------
def _delta_gate(entry: Mapping[str, Any] | None) -> dict[str, bool]:
    if entry is None:
        return {
            "mean_positive": False,
            "ci_low_positive": False,
            "effect_size": False,
            "holm_significant": False,
            "seed_sign": False,
        }
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


def _noninferiority_gate(entry: Mapping[str, Any] | None) -> dict[str, bool]:
    if entry is None:
        return {
            "mean_within_floor": False,
            "ci_lower_within_floor": False,
            "seed_sign": False,
        }
    return {
        "mean_within_floor": bool(
            entry.get("mean") is not None
            and float(entry["mean"]) >= NON_INFERIORITY_MEAN_FLOOR
        ),
        "ci_lower_within_floor": bool(
            entry.get("ci_low") is not None
            and float(entry["ci_low"]) >= NON_INFERIORITY_CI_FLOOR
        ),
        "seed_sign": bool(
            int(entry.get("noninferior_seed_count", 0)) >= POSITIVE_SEED_MIN
        ),
    }


def hardware_contract() -> dict[str, Any]:
    """G7: query stays [1,128], FPGA scorer stays Gx128 dot, no token-axis exp."""
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
    """G1-G7 for arm B (the only arm that can win)."""
    primary = validation.get("primary_tests", {})
    noninferior = validation.get("correct_alignment_noninferiority", {})
    g1_entries = {
        metric: _noninferiority_gate(noninferior.get(metric))
        for metric in ("correct_spearman", "correct_ndcg_full")
    }
    g2_entries = {
        metric: _delta_gate(primary.get(key))
        for key, metric in (
            ("P1_B_minus_A_correct_minus_same_spearman", "spearman"),
            ("P2_B_minus_A_correct_minus_same_ndcg_full", "ndcg_full"),
        )
    }
    g3_entries = {
        metric: _delta_gate(primary.get(key))
        for key, metric in (
            ("P3_B_correct_minus_same_spearman", "spearman"),
            ("P4_B_correct_minus_same_ndcg_full", "ndcg_full"),
        )
    }
    g4_entry = _delta_gate(primary.get("P5_B_selector_minus_L2_anchor_mass_k48"))
    g1 = bool(
        len(g1_entries) == 2
        and all(all(flags.values()) for flags in g1_entries.values())
    )
    g2 = bool(
        len(g2_entries) == 2
        and all(all(flags.values()) for flags in g2_entries.values())
    )
    g3 = bool(
        len(g3_entries) == 2
        and all(all(flags.values()) for flags in g3_entries.values())
    )
    g4 = bool(all(g4_entry.values()))

    collapse = validation["arms"].get(PRIMARY_ARM, {}).get("collapse", {})
    max_invariance = collapse.get("max_exact_prompt_invariance_fraction")
    g5 = bool(
        max_invariance is not None
        and float(max_invariance) <= COLLAPSE_INVARIANCE_MAX
    )

    per_seed = list(training.get("per_seed", []))
    seed_improved = [
        int(entry["best_epoch"]) > 0
        and float(entry["best_val_total"]) < float(entry["epoch0_val_total"])
        for entry in per_seed
    ]
    g6 = bool(
        len(seed_improved) > 0
        and 3 * sum(1 for value in seed_improved if value) >= 2 * len(seed_improved)
    )

    hardware = hardware_contract()
    g7 = bool(
        hardware["query_shape_ok"]
        and hardware["fpga_scorer_dot_mac_per_group"] == 128
        and hardware["fpga_scorer_compare_per_group"] == 0
        and hardware["fpga_scorer_exp_per_group"] == 0
        and hardware["token_axis_max_softmax_exp"] is False
        and hardware["anchor_region_grid_ok"]
    )
    return {
        "arm": PRIMARY_ARM,
        "G1_correct_alignment_non_inferiority": {"pass": g1, "entries": g1_entries},
        "G2_prompt_discrimination_gain": {"pass": g2, "entries": g2_entries},
        "G3_prompt_discrimination_absolute": {"pass": g3, "entries": g3_entries},
        "G4_deployed_K48_utility": {"pass": g4, "entry": g4_entry},
        "G5_collapse_removed": {
            "pass": g5,
            "max_exact_prompt_invariance_fraction": max_invariance,
            "threshold": COLLAPSE_INVARIANCE_MAX,
        },
        "G6_validation_learning": {
            "pass": g6,
            "seed_improved": seed_improved,
        },
        "G7_hardware_contract": {"pass": g7, "hardware": hardware},
        "all_pass": bool(g1 and g2 and g3 and g4 and g5 and g6 and g7),
    }


def decide(primary_gate: Mapping[str, Any] | None) -> dict[str, Any]:
    """Only arm B may win, and only on a full G1-G7 pass."""
    passed = bool(primary_gate is not None and primary_gate["all_pass"])
    return {
        "primary_arm": PRIMARY_ARM,
        "control_arm": CONTROL_ARM,
        "primary_arm_all_pass": passed,
        "primary_arm_evaluated": primary_gate is not None,
        "decision": (
            "PROMPT_CONTRASTIVE_STUDENT_ESTABLISHED"
            if passed
            else "PROMPT_CONTRASTIVE_STUDENT_NOT_ESTABLISHED"
        ),
        "winner_rule": (
            "only arm B_SPATIAL_KL_CONTRASTIVE may win, and only if it passes "
            "G1-G7; arm A is a control and can never win"
        ),
        "internal_test_evaluated": False,
    }


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------
def run_experiment(
    *,
    prereg_config: Path = PREREG_CONFIG_PATH,
    prereg_report: Path = PREREG_REPORT_PATH,
    harness_prompt: Path = HARNESS_PROMPT_PATH,
    config: Path = CONFIG_PATH,
    index_path: Path = INDEX_PATH,
    manifest_path: Path = MANIFEST_PATH,
    lock_path: Path = LOCK_PATH,
    spatial_audit_json: Path = SPATIAL_AUDIT_JSON_PATH,
    spatial_audit_script: Path = SPATIAL_AUDIT_SCRIPT_PATH,
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
    """Run the full preregistered two-arm contrastive experiment."""
    started = time.time()
    prereg = load_prereg(prereg_config)
    if enforce_frozen_inputs:
        before_hashes = {
            key: sha256_file(path)
            for key, (path, _sha) in FROZEN_INPUT_SHA256.items()
        }
    else:
        before_hashes = {}
    before_hashes["prereg_config"] = sha256_file(prereg_config)
    before_hashes["prereg_report"] = sha256_file(prereg_report)
    before_hashes["harness_prompt"] = sha256_file(harness_prompt)
    verification: dict[str, Any] = {}
    if enforce_frozen_inputs:
        verification = verify_frozen_inputs(
            prereg_config=prereg_config,
            prereg_report=prereg_report,
            harness_prompt=harness_prompt,
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
        features_directory,
        splits["train"],
        opened=opened,
        cache_loader=cache_loader,
    )
    val_samples_raw, val_aggregate, val_paths = load_split_samples(
        index_path,
        features_directory,
        splits["val"],
        opened=opened,
        cache_loader=cache_loader,
    )
    # No test path is ever resolved, hashed or opened.  The opened set must be
    # exactly the train+val caches and nothing else.
    if len(opened) != len(train_paths) + len(val_paths):
        raise DistillationContractError(
            "opened NPZ set is not exactly the train+val caches"
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
                        "gap_spearman": row["correct_spearman"]
                        - row["same_spearman"],
                        "gap_ndcg_full": row["correct_ndcg_full"]
                        - row["same_ndcg_full"],
                        "selector_minus_l2_anchor_mass_k48": row["selector_mass_k48"]
                        - row["l2_anchor_mass_k48"],
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
            "selection_metrics": [entry["selection_metric"] for entry in entries],
            "best_val_totals": [entry["best_val_total"] for entry in entries],
            "epoch0_val_totals": [entry["epoch0_val_total"] for entry in entries],
            "final_val_totals": [entry["final_val_total"] for entry in entries],
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

    after_hashes = {}
    if enforce_frozen_inputs:
        after_hashes.update(
            {
                key: sha256_file(path)
                for key, (path, _sha) in FROZEN_INPUT_SHA256.items()
            }
        )
    after_hashes["prereg_config"] = sha256_file(prereg_config)
    after_hashes["prereg_report"] = sha256_file(prereg_report)
    after_hashes["harness_prompt"] = sha256_file(harness_prompt)
    unchanged = {
        key: bool(before_hashes[key] == after_hashes[key]) for key in before_hashes
    }
    if enforce_frozen_inputs:
        post = verify_frozen_inputs(
            prereg_config=prereg_config,
            prereg_report=prereg_report,
            harness_prompt=harness_prompt,
        )
    else:
        post = {}

    bundle = {
        "status": decision["decision"],
        "report_stem": REPORT_STEM,
        "scope_note": SCOPE_NOTE,
        "development_validation_only": True,
        "preregistered_config": str(prereg_config),
        "preregistration_report": str(prereg_report),
        "harness_prompt_path": str(harness_prompt),
        "prereg_config_sha256": after_hashes["prereg_config"],
        "prereg_report_sha256": after_hashes["prereg_report"],
        "harness_prompt_sha256": after_hashes["harness_prompt"],
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
        "source_sha256": {
            "trainer": sha256_file(Path(__file__)),
            "existing_spatial_trainer_readonly": sha256_file(
                EXISTING_SPATIAL_TRAINER_PATH
            ),
        },
        "split": splits,
        "split_group_summary": group_summary,
        "split_group_by": "source_image_sha256",
        "isolation": {
            "train_npz_count": len(train_paths),
            "val_npz_count": len(val_paths),
            "opened_npz_count": len(opened),
            "train_val_npz_count": len(train_paths) + len(val_paths),
            "test_npz_opened": [],
            "test_npz_opened_count": 0,
            "test_npz_hashed_count": 0,
            "test_npz_resolved_or_opened": False,
            "opened_only_train_val_npz": len(opened)
            == len(train_paths) + len(val_paths),
            "test_used_for_selection": False,
            "test_npz_action": "never resolved, hashed, opened or scored",
        },
        "arms": {arm["name"]: arm for arm in ARMS if arm["name"] in arm_names},
        "seeds": [int(seed) for seed in seeds],
        "epochs": int(epochs),
        "budgets": [int(budget) for budget in budgets],
        "lambda_semantic": float(lambda_semantic),
        "contrastive": {
            "weight": CONTRASTIVE_WEIGHT,
            "margin": CONTRASTIVE_MARGIN,
            "softplus_temperature": SOFTPLUS_TEMPERATURE,
            "wrong_prompt_definition": (
                "grounding uses same-image pointing prompt; pointing uses "
                "same-image grounding prompt"
            ),
        },
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
            "development validation only: the val split was already used during development; this is not final accuracy",
            "internal train/val sanity only; not CV-Bench task accuracy",
            "no hardware synthesis, latency or speedup claim",
            "no 4B model load, no feature re-extraction, no raw image read",
            "the internal test split was never resolved, hashed, opened or evaluated",
            "the prior spatial-student report's G3-style sign defect is only referenced historically; this experiment computes P5 as selector minus baseline",
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
                            "base": stats["base"],
                            "contrastive": stats["contrastive"],
                            "total": stats["total"],
                            "pair_count": stats["pair_count"],
                            "selection_value": row["selection_value"],
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
            "base",
            "contrastive",
            "total",
            "pair_count",
            "selection_value",
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
        "gap_spearman",
        "gap_ndcg_full",
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
        "selector_minus_l2_anchor_mass_k48",
        "exact_prompt_invariant",
        "tolerance_prompt_invariant",
        "max_abs_diff",
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
    lines.append(
        f"# Prompt-contrastive spatial student train/val report ({REPORT_STEM})"
    )
    lines.append("")
    lines.append(f"- status: **{bundle['status']}**")
    lines.append(f"- scope: {bundle['scope_note']}")
    lines.append(
        f"- development validation only: `{bundle['development_validation_only']}` "
        "(the val split was already used during development; not final accuracy)"
    )
    lines.append(
        f"- prereg config SHA256: `{bundle['prereg_config_sha256']}` "
        f"(unchanged={bundle['prereg_unchanged']})"
    )
    lines.append(f"- prereg report SHA256: `{bundle['prereg_report_sha256']}`")
    lines.append(f"- harness prompt SHA256: `{bundle['harness_prompt_sha256']}`")
    lines.append(f"- before/after hashes unchanged: `{bundle['hashes_unchanged']}`")
    lines.append(
        f"- train NPZ aggregate SHA256: `{bundle['train_npz_aggregate_sha256']}`"
    )
    lines.append(
        f"- val NPZ aggregate SHA256: `{bundle['val_npz_aggregate_sha256']}`"
    )
    lines.append(
        f"- test_npz_opened = `{bundle['isolation']['test_npz_opened']}` "
        f"(count={bundle['isolation']['test_npz_opened_count']}, "
        f"hashed={bundle['isolation']['test_npz_hashed_count']})"
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
        "| arm | seed | selection | epoch0 val total | best epoch | best val total | "
        "best val base | best val contrastive | final val total |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm_name, training in bundle["training"].items():
        for entry in training["per_seed"]:
            lines.append(
                f"| {arm_name} | {entry['seed']} | {entry['selection_metric']} | "
                f"{entry['epoch0_val_total']:.6f} | {entry['best_epoch']} | "
                f"{entry['best_val_total']:.6f} | {entry['best_val_base']:.6f} | "
                f"{entry['best_val_contrastive']:.6f} | "
                f"{entry['final_val_total']:.6f} |"
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
    lines.append("## G1 descriptive non-inferiority (not in Holm)")
    lines.append("")
    for metric_key, entry in bundle["validation_statistics"][
        "correct_alignment_noninferiority"
    ].items():
        lines.append(
            f"- {metric_key}: mean={entry['mean']:.6f} ci_low={entry['ci_low']:.6f} "
            f"noninferior_seeds={entry['noninferior_seed_count']}/{entry['seed_count']}"
        )
    lines.append("")
    lines.append("## Gates G1-G7 (arm B only)")
    lines.append("")
    gate = bundle.get("primary_gate")
    if gate is None:
        lines.append("- primary arm not evaluated in this run")
    else:
        lines.append(
            f"- G1={gate['G1_correct_alignment_non_inferiority']['pass']} "
            f"G2={gate['G2_prompt_discrimination_gain']['pass']} "
            f"G3={gate['G3_prompt_discrimination_absolute']['pass']} "
            f"G4={gate['G4_deployed_K48_utility']['pass']} "
            f"G5={gate['G5_collapse_removed']['pass']} "
            f"G6={gate['G6_validation_learning']['pass']} "
            f"G7={gate['G7_hardware_contract']['pass']} "
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
def _primary_values(bundle: Mapping[str, Any]) -> dict[str, Any]:
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
        "harness_prompt_sha256": bundle["harness_prompt_sha256"],
        "hashes_unchanged": bundle["hashes_unchanged"],
        "source_sha256": bundle["source_sha256"],
        "arms": list(bundle["arms"]),
        "seeds": bundle["seeds"],
        "train_records": bundle["split_group_summary"]["train"]["sample_count"],
        "val_records": bundle["split_group_summary"]["val"]["sample_count"],
        "val_images": bundle["split_group_summary"]["val"]["image_count"],
        "test_npz_opened": bundle["isolation"]["test_npz_opened"],
        "test_npz_hashed_count": bundle["isolation"]["test_npz_hashed_count"],
        "opened_only_train_val_npz": bundle["isolation"]["opened_only_train_val_npz"],
        "best_epochs": {
            arm: training["best_epochs"]
            for arm, training in bundle["training"].items()
        },
        "selection_metrics": {
            arm: training["selection_metrics"]
            for arm, training in bundle["training"].items()
        },
        "primary_values": _primary_values(bundle),
        "gates": (
            {
                key: gate[key]["pass"]
                for key in (
                    "G1_correct_alignment_non_inferiority",
                    "G2_prompt_discrimination_gain",
                    "G3_prompt_discrimination_absolute",
                    "G4_deployed_K48_utility",
                    "G5_collapse_removed",
                    "G6_validation_learning",
                    "G7_hardware_contract",
                )
            }
            if gate is not None
            else None
        ),
        "runtime_seconds": bundle["runtime_seconds"],
        "peak_rss_mib": bundle["peak_rss_mib"],
        "limitations": bundle["limitations"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg-config", type=Path, default=PREREG_CONFIG_PATH)
    parser.add_argument("--prereg-report", type=Path, default=PREREG_REPORT_PATH)
    parser.add_argument("--harness-prompt", type=Path, default=HARNESS_PROMPT_PATH)
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
        default=LOGS_DIR / f"{REPORT_STEM}-harness-result.json",
    )
    args = parser.parse_args(argv)
    bundle = run_experiment(
        prereg_config=args.prereg_config,
        prereg_report=args.prereg_report,
        harness_prompt=args.harness_prompt,
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


if __name__ == "__main__":
    raise SystemExit(main())
