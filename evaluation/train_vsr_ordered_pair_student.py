#!/usr/bin/env python3
"""Train the ordered three-slot VSR pair student on the frozen B1 margin teacher.

Contract
--------
* The CLI reads parameters and sha256 pins from a future preregistered config
  (``configs/2026-10-10-vsr-ordered-pair-student.preregistered.json``).  This
  task does not create that config and never runs ``main``.
* Only the *metadata* of ``student_index.jsonl`` is read first; the grouped
  train/val/internal-test split is the existing ``split_sample_ids`` image-hash
  split.  Before the validation gate passes, only train+val NPZ caches may be
  opened.  ``RestrictedFeatureLoader`` refuses any other sample id *before* it
  touches the file and records every opened sample id.
* Prompts are sliced to the frozen ``caption_token_spans``; the sliced token
  count ``T`` must lie in ``[1, max_prompt_tokens]`` (fail closed on overflow).
  No label, relation annotation, answer or VSR dev/test input is read.
* The same-image donor is a deterministic cycle over the prompts of one
  ``source_image_sha256`` sorted by ``sample_id``; single-prompt images have no
  same donor.  The cross-image donor is a diagnostic only.  Training applies the
  original listwise teacher KL to every sample and adds the existing softplus
  contrastive term only where a same-image donor exists.  Every prompt is scored
  against its *own* teacher; no relation label is used.
* ``OrderedPairScorer`` is reused: the visual projection is computed once per
  sample and the same projected visual is scored with the correct / same / cross
  prompt encodings.
* All point estimates and bootstrap CIs are image-equal: records sharing one
  ``source_image_sha256`` are averaged first, then images are averaged with equal
  weight.  Bootstrap seed/repetitions come from the config.

The teacher is a proxy signal, not task accuracy, and every number here is a
float software estimate, not an FPGA measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.metrics import load_jsonl
from evaluation.ordered_pair_scorer import (
    FEATURE_DIM,
    PROMPT_DIM,
    OrderedPairScorer,
    grid_positions,
)
from evaluation.prompt_projection_distillation import (
    distillation_loss,
    l2_baseline_scores,
    load_feature_cache,
    metric_bundle,
    split_sample_ids,
    validate_source_image_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/2026-10-10-vsr-ordered-pair-student.preregistered.json"
DEFAULT_INDEX = ROOT / "artifacts/token_selector/2026-10-10-vsr-b1-margin-teacher/student_index.jsonl"
DEFAULT_FEATURES = DEFAULT_INDEX.parent
DEFAULT_SPANS = DEFAULT_FEATURES / "caption_token_spans.json"
DEFAULT_OUTPUT = ROOT / "artifacts/token_selector/2026-10-10-vsr-ordered-pair-student"
DEFAULT_REPORT = ROOT / "reports/token_selector/2026-10-10-vsr-ordered-pair-student.json"

FROZEN_BUDGET = 96
FROZEN_PARTNER_K = 4
FROZEN_MAX_PROMPT_TOKENS = 16
METRIC_KEYS = ("spearman", "ndcg_full", "mass_k96")

DEFAULT_PARAMETERS: dict[str, Any] = {
    "seeds": [20261010, 20261011, 20261012],
    "epochs": 15,
    "batch_size": 8,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,
    "init_std": 0.02,
    "temperature": 0.1,
    "contrastive_weight": 0.5,
    "contrastive_margin_nats": 0.05,
    "contrastive_softplus_scale": 0.1,
    "partner_k": FROZEN_PARTNER_K,
    "max_prompt_tokens": FROZEN_MAX_PROMPT_TOKENS,
    "budget": FROZEN_BUDGET,
    "split_seed": 20261010,
    "split_salt": "vsr_b1_margin_student_2026_10_10",
    "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
    "device": "cpu",
    "bootstrap_seed": 20261010,
    "bootstrap_repetitions": 3000,
}

CONFIG_META_KEYS = frozenset(
    {
        "status",
        "scope",
        "architecture",
        "hypothesis",
        "single_change_from_failed_baseline",
        "wrong_prompt",
        "winner_selection",
        "promotion_gates",
        "hardware_contract",
        "stop_rule",
        "pins",
    }
)

ARCHITECTURE = "ordered_three_slot_bounded_partner"


class OrderedPairStudentError(RuntimeError):
    """Raised when the ordered-pair student contract is violated."""


class TestSplitOpenedError(OrderedPairStudentError):
    """Raised when a sample outside the permitted split set is requested."""


# ---------------------------------------------------------------------------
# Config / pins
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pins(config: Mapping[str, Any], root: Path = ROOT) -> None:
    """Verify every pinned input sha256; test NPZ files are never pinned here."""
    for name, pin in config.get("pins", {}).items():
        path = Path(str(pin["path"]))
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or sha256_file(path) != str(pin["sha256"]):
            raise OrderedPairStudentError(f"pinned input mismatch: {name}")


def resolve_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(DEFAULT_PARAMETERS)
    params.update({key: value for key, value in config.items() if key not in CONFIG_META_KEYS})
    validate_parameters(params)
    return params


def validate_parameters(cfg: Mapping[str, Any]) -> None:
    if not [int(seed) for seed in cfg["seeds"]]:
        raise OrderedPairStudentError("config seeds must be non-empty")
    if int(cfg["epochs"]) < 1 or int(cfg["batch_size"]) < 1:
        raise OrderedPairStudentError("epochs and batch_size must be >= 1")
    if int(cfg["partner_k"]) < 1:
        raise OrderedPairStudentError("partner_k must be >= 1")
    if not 1 <= int(cfg["max_prompt_tokens"]) <= FROZEN_MAX_PROMPT_TOKENS:
        raise OrderedPairStudentError("max_prompt_tokens must be in [1, 16]")
    if int(cfg["budget"]) < 1 or int(cfg["bootstrap_repetitions"]) < 1:
        raise OrderedPairStudentError("budget and bootstrap_repetitions must be >= 1")
    ratios = cfg["split_ratios"]
    for name in ("train", "val", "test"):
        if name not in ratios:
            raise OrderedPairStudentError(f"split_ratios missing {name}")
    if abs(sum(float(ratios[name]) for name in ("train", "val", "test")) - 1.0) > 1e-9:
        raise OrderedPairStudentError("split_ratios must sum to 1.0")
    if float(cfg["temperature"]) <= 0.0 or float(cfg["contrastive_softplus_scale"]) <= 0.0:
        raise OrderedPairStudentError("temperature and softplus scale must be > 0")


def verify_frozen(cfg: Mapping[str, Any]) -> None:
    if int(cfg["partner_k"]) != FROZEN_PARTNER_K:
        raise OrderedPairStudentError(f"partner_k must be frozen at {FROZEN_PARTNER_K}")
    if int(cfg["max_prompt_tokens"]) != FROZEN_MAX_PROMPT_TOKENS:
        raise OrderedPairStudentError(
            f"max_prompt_tokens must be frozen at {FROZEN_MAX_PROMPT_TOKENS}"
        )
    if int(cfg["budget"]) != FROZEN_BUDGET:
        raise OrderedPairStudentError(f"budget must be frozen at {FROZEN_BUDGET}")


def resolve_device(requested: Any) -> str:
    device = str(requested)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise OrderedPairStudentError(
            f"config requests {device}, but CUDA is unavailable; refusing an implicit CPU run"
        )
    return device


# ---------------------------------------------------------------------------
# Metadata-only index and restricted feature loading
# ---------------------------------------------------------------------------


def load_index_metadata(index_path: Path) -> list[dict[str, Any]]:
    """Read only ``student_index.jsonl`` metadata; never opens a cache NPZ."""
    records = load_jsonl(Path(index_path))
    if not records:
        raise OrderedPairStudentError(f"index {index_path} contains no samples")
    metadata: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id in seen:
            raise OrderedPairStudentError(f"duplicate index sample_id {sample_id}")
        seen.add(sample_id)
        if bool(record.get("contains_official_label", False)):
            raise OrderedPairStudentError(f"index {sample_id} carries an official label")
        if bool(record.get("contains_relation_annotation", False)):
            raise OrderedPairStudentError(f"index {sample_id} carries a relation annotation")
        metadata.append(
            {
                "sample_id": sample_id,
                "task": str(record.get("task", "")),
                "npz": str(record["npz"]),
                "grid_h": int(record["grid_h"]),
                "grid_w": int(record["grid_w"]),
                "group_count": int(record.get("group_count", int(record["grid_h"]) * int(record["grid_w"]))),
                "prompt_token_count": int(record.get("prompt_token_count", 0)),
                "source_image_sha256": validate_source_image_sha256(
                    record.get("source_image_sha256")
                ),
            }
        )
    return metadata


def split_metadata(records: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> dict[str, list[str]]:
    """Existing image-group hash split over metadata only."""
    return split_sample_ids(
        records,
        seed=int(cfg["split_seed"]),
        ratios=cfg["split_ratios"],
        salt=str(cfg["split_salt"]),
    )


class RestrictedFeatureLoader:
    """Open only the explicitly permitted sample ids and record every open.

    A disallowed sample id is rejected *before* any path resolution or file IO,
    which is what keeps the internal-test NPZ untouched until the gate passes.
    """

    def __init__(
        self,
        index_records: Sequence[Mapping[str, Any]],
        features_dir: Path,
        allowed_sample_ids: Sequence[str],
    ) -> None:
        self._records = {str(record["sample_id"]): dict(record) for record in index_records}
        self.features_dir = Path(features_dir)
        self._allowed = {str(value) for value in allowed_sample_ids}
        unknown = self._allowed - set(self._records)
        if unknown:
            raise OrderedPairStudentError(f"allowed ids missing from the index: {sorted(unknown)}")
        self.opened_sample_ids: list[str] = []
        self.blocked_sample_ids: list[str] = []

    @property
    def allowed_sample_ids(self) -> set[str]:
        return set(self._allowed)

    def record(self, sample_id: str) -> dict[str, Any]:
        sid = str(sample_id)
        if sid not in self._records:
            raise OrderedPairStudentError(f"unknown sample_id {sid}")
        return self._records[sid]

    def npz_path(self, sample_id: str) -> Path:
        record = self.record(sample_id)
        path = Path(record["npz"])
        return path if path.is_absolute() else self.features_dir / path

    def load(self, sample_id: str) -> dict[str, Any]:
        sid = str(sample_id)
        if sid not in self._records:
            raise OrderedPairStudentError(f"unknown sample_id {sid}")
        if sid not in self._allowed:
            self.blocked_sample_ids.append(sid)
            raise TestSplitOpenedError(
                f"sample {sid} is outside the permitted split set"
            )
        cache = load_feature_cache(self.npz_path(sid))
        self.opened_sample_ids.append(sid)
        return cache


# ---------------------------------------------------------------------------
# Caption span slicing / sample preparation
# ---------------------------------------------------------------------------


def slice_caption_prompt(
    prompt: Any, span: Mapping[str, Any], max_prompt_tokens: int
) -> np.ndarray:
    """Half-open caption slice with a fail-closed ``[1, max_prompt_tokens]`` bound."""
    array = np.asarray(prompt)
    full = int(array.shape[0])
    if int(span["full_token_count"]) != full:
        raise ValueError("prompt token count mismatch")
    start = int(span["start"])
    end = int(span["end_exclusive"])
    if start < 0 or end > full or end <= start:
        raise ValueError(f"caption span [{start},{end}) outside [0,{full}]")
    sliced = np.ascontiguousarray(array[start:end], dtype=np.float32)
    token_count = int(sliced.shape[0])
    if token_count < 1 or token_count > int(max_prompt_tokens):
        raise ValueError(
            f"caption token count {token_count} outside [1,{int(max_prompt_tokens)}]"
        )
    if "caption_token_count" in span and int(span["caption_token_count"]) != token_count:
        raise ValueError("caption_token_count disagrees with the half-open span")
    return sliced


def prepare_sample(
    meta: Mapping[str, Any],
    cache: Mapping[str, Any],
    spans: Mapping[str, Any],
    max_prompt_tokens: int,
) -> dict[str, Any]:
    sample_id = str(meta["sample_id"])
    if sample_id not in spans:
        raise OrderedPairStudentError(f"missing caption span for {sample_id}")
    prompt = np.asarray(cache["prompt_embeddings"], dtype=np.float32)
    sliced = slice_caption_prompt(prompt, spans[sample_id], max_prompt_tokens)
    # Keep the large visual cache in its stored FP16 form on the host.  Each
    # sample is promoted to FP32 only when it is transferred for scoring.
    group_mean = np.asarray(cache["group_mean"])
    grid_h, grid_w = int(meta["grid_h"]), int(meta["grid_w"])
    if group_mean.shape != (grid_h * grid_w, FEATURE_DIM):
        raise OrderedPairStudentError(f"group_mean shape mismatch for {sample_id}")
    if sliced.shape[1] != PROMPT_DIM:
        raise OrderedPairStudentError(f"prompt width mismatch for {sample_id}")
    positions = grid_positions(grid_h, grid_w).detach().cpu().numpy().astype(np.float32)
    return {
        "sample_id": sample_id,
        "source_image_sha256": validate_source_image_sha256(meta["source_image_sha256"]),
        "grid_h": grid_h,
        "grid_w": grid_w,
        "group_mean": group_mean,
        "prompt_embeddings": sliced,
        "positions": positions,
        "teacher_raw": np.asarray(cache["teacher_scores"], dtype=np.float32),
        "teacher_normalized": np.asarray(cache["teacher_scores_normalized"], dtype=np.float32),
        "teacher_soft": np.asarray(cache["teacher_soft_target"], dtype=np.float32),
    }


def load_samples(
    loader: RestrictedFeatureLoader,
    sample_ids: Sequence[str],
    spans: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    max_prompt_tokens = int(cfg["max_prompt_tokens"])
    samples: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        meta = loader.record(sample_id)
        cache = loader.load(sample_id)
        samples.append(prepare_sample(meta, cache, spans, max_prompt_tokens))
    return samples


# ---------------------------------------------------------------------------
# Deterministic donors
# ---------------------------------------------------------------------------


def same_image_donor_map(samples: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Cycle the prompts of each multi-prompt image; single-prompt images none."""
    by_image: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        by_image[str(sample["source_image_sha256"])].append(str(sample["sample_id"]))
    donors: dict[str, str] = {}
    for image in sorted(by_image):
        ordered = sorted(by_image[image])
        if len(ordered) < 2:
            continue
        for index, sample_id in enumerate(ordered):
            donors[sample_id] = ordered[(index + 1) % len(ordered)]
    return donors


def cross_image_donor_map(samples: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Deterministic next sample in the split with a different image (diagnostic)."""
    ordered = sorted(samples, key=lambda row: str(row["sample_id"]))
    donors: dict[str, str] = {}
    for index, sample in enumerate(ordered):
        image = str(sample["source_image_sha256"])
        for offset in range(1, len(ordered)):
            donor = ordered[(index + offset) % len(ordered)]
            if str(donor["source_image_sha256"]) != image:
                donors[str(sample["sample_id"])] = str(donor["sample_id"])
                break
    return donors


def same_donor_coverage(samples: Sequence[Mapping[str, Any]], donors: Mapping[str, str]) -> dict[str, int]:
    covered = [sample for sample in samples if str(sample["sample_id"]) in donors]
    return {
        "sample_count": len(covered),
        "image_count": len({str(sample["source_image_sha256"]) for sample in covered}),
    }


# ---------------------------------------------------------------------------
# Scoring / loss
# ---------------------------------------------------------------------------


def build_model(seed: int, cfg: Mapping[str, Any]) -> OrderedPairScorer:
    return OrderedPairScorer(
        seed=int(seed),
        init_std=float(cfg["init_std"]),
        max_prompt_tokens=int(cfg["max_prompt_tokens"]),
        partner_k=int(cfg["partner_k"]),
    )


def encode_and_score(
    model: OrderedPairScorer,
    visual: torch.Tensor,
    positions: torch.Tensor,
    prompt: Any,
    device: str,
) -> torch.Tensor:
    prompt_tensor = torch.as_tensor(np.asarray(prompt, dtype=np.float32), device=device)
    queries, geometry = model.encode_prompt(prompt_tensor)
    return model.score_projected(visual, positions, queries, geometry)


def ordered_pair_loss_terms(
    model: OrderedPairScorer,
    sample: Mapping[str, Any],
    same_prompt: Any,
    cfg: Mapping[str, Any],
    device: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Teacher KL for every sample plus the softplus contrastive term on same donors."""
    group_mean = torch.as_tensor(sample["group_mean"], dtype=torch.float32, device=device)
    positions = torch.as_tensor(sample["positions"], dtype=torch.float32, device=device)
    teacher_soft = torch.as_tensor(sample["teacher_soft"], dtype=torch.float32, device=device)
    teacher_normalized = torch.as_tensor(
        sample["teacher_normalized"], dtype=torch.float32, device=device
    )

    visual = model.project_visual(group_mean)
    correct = encode_and_score(model, visual, positions, sample["prompt_embeddings"], device)
    base, parts = distillation_loss(
        correct,
        teacher_soft,
        teacher_normalized,
        temperature=float(cfg["temperature"]),
        listwise_kl_weight=1.0,
        score_mse_weight=0.0,
        temperature_squared=True,
    )
    contrast = torch.zeros((), device=device)
    advantage = torch.zeros((), device=device)
    if same_prompt is not None:
        wrong = encode_and_score(model, visual, positions, same_prompt, device)
        temperature = float(cfg["temperature"])
        ce_correct = -(teacher_soft * F.log_softmax(correct / temperature, dim=-1)).sum()
        ce_wrong = -(teacher_soft * F.log_softmax(wrong / temperature, dim=-1)).sum()
        advantage = ce_wrong - ce_correct
        scale = float(cfg["contrastive_softplus_scale"])
        contrast = (
            F.softplus((float(cfg["contrastive_margin_nats"]) - advantage) / scale) * scale
        )
    total = base + float(cfg["contrastive_weight"]) * contrast
    info = {
        "base": float(parts["total"]),
        "contrast": float(contrast.detach()),
        "ce_advantage": float(advantage.detach()),
    }
    return total, info


@torch.no_grad()
def validation_objective(
    model: OrderedPairScorer,
    samples: Sequence[Mapping[str, Any]],
    same_donors: Mapping[str, str],
    cfg: Mapping[str, Any],
    device: str,
) -> float:
    if not samples:
        return 0.0
    model.eval()
    by_id = {str(sample["sample_id"]): sample for sample in samples}
    totals: list[float] = []
    for sample in samples:
        donor_id = same_donors.get(str(sample["sample_id"]))
        donor_prompt = by_id[donor_id]["prompt_embeddings"] if donor_id is not None else None
        loss, _ = ordered_pair_loss_terms(model, sample, donor_prompt, cfg, device)
        totals.append(float(loss.detach()))
    return float(np.mean(totals))


# ---------------------------------------------------------------------------
# Image-equal estimation
# ---------------------------------------------------------------------------


def image_equal_stat(
    values: Sequence[float], images: Sequence[str], *, seed: int, reps: int
) -> dict[str, float]:
    """Average records within an image first, then weight every image equally."""
    groups: dict[str, list[float]] = defaultdict(list)
    for value, image in zip(values, images):
        groups[str(image)].append(float(value))
    per_image = np.asarray([np.mean(groups[image]) for image in sorted(groups)], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    if int(reps) > 0:
        draws = per_image[rng.integers(0, per_image.size, size=(int(reps), per_image.size))].mean(axis=1)
    else:
        draws = per_image
    return {
        "mean": float(per_image.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "image_count": int(per_image.size),
        "sample_count": int(len(values)),
    }


def _metric_values(bundle: Mapping[str, Any]) -> dict[str, float]:
    return {
        "spearman": float(bundle["spearman"]),
        "ndcg_full": float(bundle["ndcg_full"]),
        "mass_k96": float(bundle["per_budget"][str(FROZEN_BUDGET)]["saliency_mass_retained"]),
    }


def _variant_summary(
    records: Sequence[Mapping[str, Any]], key: str, seed: int, reps: int
) -> dict[str, Any]:
    values: dict[str, list[float]] = {name: [] for name in METRIC_KEYS}
    images: list[str] = []
    for record in records:
        bundle = record[key]
        if bundle is None:
            continue
        images.append(str(record["source_image_sha256"]))
        for name in METRIC_KEYS:
            values[name].append(float(bundle[name]))
    if not images:
        return {"available": False, "sample_count": 0, "image_count": 0}
    summary: dict[str, Any] = {
        "available": True,
        "sample_count": int(len(images)),
        "image_count": int(len(set(images))),
    }
    for offset, name in enumerate(METRIC_KEYS):
        summary[name] = image_equal_stat(values[name], images, seed=int(seed) + offset, reps=reps)
    return summary


def aggregate_records(records: Sequence[Mapping[str, Any]], *, seed: int, reps: int) -> dict[str, Any]:
    return {
        "sample_count": int(len(records)),
        "image_count": int(len({str(record["source_image_sha256"]) for record in records})),
        "correct": _variant_summary(records, "correct", int(seed), reps),
        "l2": _variant_summary(records, "l2", int(seed) + 101, reps),
        "correct_minus_l2": _variant_summary(records, "correct_minus_l2", int(seed) + 211, reps),
        "correct_minus_cross": _variant_summary(records, "correct_minus_cross", int(seed) + 307, reps),
        "correct_minus_same": _variant_summary(records, "correct_minus_same", int(seed) + 401, reps),
    }


@torch.no_grad()
def evaluate_split(
    model: OrderedPairScorer,
    samples: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    *,
    same_donors: Mapping[str, str],
    cross_donors: Mapping[str, str],
    seed: int,
    reps: int,
) -> dict[str, Any]:
    device = str(cfg["device"])
    budget = int(cfg["budget"])
    model.eval()
    by_id = {str(sample["sample_id"]): sample for sample in samples}
    records: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        group_mean = torch.as_tensor(sample["group_mean"], dtype=torch.float32, device=device)
        positions = torch.as_tensor(sample["positions"], dtype=torch.float32, device=device)
        visual = model.project_visual(group_mean)

        def scores_for(prompt: Any) -> np.ndarray:
            return encode_and_score(model, visual, positions, prompt, device).detach().cpu().numpy()

        def bundle(scores: np.ndarray) -> dict[str, float]:
            return _metric_values(
                metric_bundle(
                    scores,
                    sample["teacher_raw"],
                    sample["teacher_normalized"],
                    sample["teacher_soft"],
                    budgets=[budget],
                )
            )

        correct = bundle(scores_for(sample["prompt_embeddings"]))
        l2 = bundle(l2_baseline_scores(sample["group_mean"]))
        cross_id = cross_donors.get(sample_id)
        same_id = same_donors.get(sample_id)
        cross = bundle(scores_for(by_id[cross_id]["prompt_embeddings"])) if cross_id is not None else None
        same = bundle(scores_for(by_id[same_id]["prompt_embeddings"])) if same_id is not None else None
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "source_image_sha256": str(sample["source_image_sha256"]),
            "correct": correct,
            "l2": l2,
            "correct_minus_l2": {name: correct[name] - l2[name] for name in METRIC_KEYS},
            "correct_minus_cross": (
                {name: correct[name] - cross[name] for name in METRIC_KEYS} if cross else None
            ),
            "correct_minus_same": (
                {name: correct[name] - same[name] for name in METRIC_KEYS} if same else None
            ),
        }
        records.append(record)
    return aggregate_records(records, seed=int(seed), reps=int(reps))


# ---------------------------------------------------------------------------
# Gates and winner selection
# ---------------------------------------------------------------------------


def validation_gate(val_metrics: Mapping[str, Any]) -> dict[str, Any]:
    correct_l2 = val_metrics["correct_minus_l2"]
    cross = val_metrics["correct_minus_cross"]
    same = val_metrics["correct_minus_same"]
    checks: dict[str, Any] = {
        "correct_minus_l2_spearman_positive": bool(correct_l2.get("available")) and correct_l2["spearman"]["mean"] > 0.0,
        "correct_minus_l2_ndcg_full_positive": bool(correct_l2.get("available")) and correct_l2["ndcg_full"]["mean"] > 0.0,
        "cross_image_ci_low_positive_all_three": bool(cross.get("available"))
        and all(cross[name]["ci_low"] > 0.0 for name in METRIC_KEYS),
        "same_image_ci_low_positive_all_three": bool(same.get("available"))
        and all(same[name]["ci_low"] > 0.0 for name in METRIC_KEYS),
    }
    checks["pass"] = all(checks.values())
    return checks


def _sensitivity_lows(variant: Mapping[str, Any]) -> list[float]:
    if not variant.get("available"):
        return [-math.inf] * len(METRIC_KEYS)
    return [float(variant[name]["ci_low"]) for name in METRIC_KEYS]


def winner_selection_key(run: Mapping[str, Any]) -> tuple[float, float, float]:
    val = run["val"]
    lows = _sensitivity_lows(val["correct_minus_cross"]) + _sensitivity_lows(val["correct_minus_same"])
    return (
        min(lows),
        float(val["correct_minus_l2"]["ndcg_full"]["mean"]),
        -float(run["best_val_objective"]),
    )


def maybe_evaluate_internal_test(
    *,
    gate_pass: bool,
    test_loader_factory: Callable[[], RestrictedFeatureLoader],
    evaluate_fn: Callable[[RestrictedFeatureLoader], Any],
) -> dict[str, Any]:
    """Load and evaluate internal test exactly once, and only when the gate passes."""
    if not gate_pass:
        return {"internal_test": None, "test_npz_opened": []}
    loader = test_loader_factory()
    internal_test = evaluate_fn(loader)
    return {"internal_test": internal_test, "test_npz_opened": list(loader.opened_sample_ids)}


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


def state_dict_float32(state: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, value in state.items():
        tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        arrays[str(name)] = np.ascontiguousarray(tensor.numpy(), dtype=np.float32)
    return arrays


def ordered_pair_state_sha256(state: Mapping[str, Any]) -> str:
    """Content hash over every float32 state entry (name, dtype, shape, bytes)."""
    digest = hashlib.sha256()
    for name in sorted(state):
        array = np.ascontiguousarray(np.asarray(state[name], dtype=np.float32))
        header = f"{name}:{array.dtype.str}:{array.shape}".encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def save_ordered_pair_npz(path: Path, state: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    arrays = state_dict_float32(state)
    digest = ordered_pair_state_sha256(arrays)
    payload: dict[str, Any] = dict(arrays)
    payload["state_sha256"] = np.asarray(digest)
    meta = dict(metadata)
    meta.setdefault("state_sha256", digest)
    meta.setdefault("dtype", "float32")
    meta.setdefault("parameter_count", int(sum(array.size for array in arrays.values())))
    payload["metadata"] = np.asarray(json.dumps(meta, ensure_ascii=False, sort_keys=True))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return {"state_sha256": digest, "bytes": path.stat().st_size, "metadata": meta}


def load_ordered_pair_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: np.array(archive[key]) for key in archive.files if key not in {"state_sha256", "metadata"}}


def rebuild_model(seed: int, state: Mapping[str, Any], cfg: Mapping[str, Any], device: str) -> OrderedPairScorer:
    model = build_model(int(seed), cfg)
    model.load_state_dict({name: torch.as_tensor(value) for name, value in state.items()})
    return model.to(device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_seed(
    seed: int,
    train: Sequence[Mapping[str, Any]],
    val: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    out_dir: Path,
    *,
    scorer_factory: Callable[[], OrderedPairScorer] | None = None,
) -> dict[str, Any]:
    device = str(cfg["device"])
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    rng = np.random.default_rng(int(seed))
    factory = scorer_factory or (lambda: build_model(int(seed), cfg))
    model = factory().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    train_by_id = {str(sample["sample_id"]): sample for sample in train}
    train_same = same_image_donor_map(train)
    val_same = same_image_donor_map(val)

    best: dict[str, Any] | None = None
    history: list[dict[str, float]] = []
    batch_size = int(cfg["batch_size"])
    for epoch in range(int(cfg["epochs"])):
        model.train()
        order = rng.permutation(len(train))
        epoch_values: list[float] = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device)
            for position in batch:
                sample = train[int(position)]
                donor_id = train_same.get(str(sample["sample_id"]))
                donor_prompt = train_by_id[donor_id]["prompt_embeddings"] if donor_id is not None else None
                item_loss, _ = ordered_pair_loss_terms(model, sample, donor_prompt, cfg, device)
                loss = loss + item_loss
            loss = loss / len(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip_norm"]))
            optimizer.step()
            epoch_values.append(float(loss.detach()))
        val_objective = validation_objective(model, val, val_same, cfg, device)
        history.append(
            {
                "epoch": int(epoch),
                "train_objective": float(np.mean(epoch_values)),
                "val_objective": val_objective,
            }
        )
        if best is None or val_objective < best["val_objective"]:
            best = {
                "epoch": int(epoch),
                "val_objective": val_objective,
                "state": {name: value.detach().cpu().float().clone() for name, value in model.state_dict().items()},
            }
    if best is None:
        raise OrderedPairStudentError("training completed without a best epoch")

    model.load_state_dict(best["state"])
    model.eval()
    val_metrics = evaluate_split(
        model,
        val,
        cfg,
        same_donors=val_same,
        cross_donors=cross_image_donor_map(val),
        seed=int(cfg["bootstrap_seed"]),
        reps=int(cfg["bootstrap_repetitions"]),
    )
    checkpoint = Path(out_dir) / f"seed-{int(seed)}.npz"
    export = save_ordered_pair_npz(
        checkpoint,
        best["state"],
        {
            "architecture": ARCHITECTURE,
            "seed": int(seed),
            "best_epoch": int(best["epoch"]),
            "best_val_objective": float(best["val_objective"]),
            "max_prompt_tokens": int(cfg["max_prompt_tokens"]),
            "partner_k": int(cfg["partner_k"]),
            "budget": int(cfg["budget"]),
            "teacher_proxy_not_task_accuracy": True,
            "float_software_not_fpga_measurement": True,
        },
    )
    return {
        "seed": int(seed),
        "best_epoch": int(best["epoch"]),
        "best_val_objective": float(best["val_objective"]),
        "checkpoint": str(checkpoint),
        "state_sha256": export["state_sha256"],
        "history": history,
        "val": val_metrics,
        "state": best["state"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--spans", type=Path, default=DEFAULT_SPANS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    raw_config = json.loads(args.config.read_text(encoding="utf-8"))
    verify_pins(raw_config)
    cfg = resolve_parameters(raw_config)
    verify_frozen(cfg)
    cfg["device"] = resolve_device(cfg["device"])

    records = load_index_metadata(args.index)
    split = split_metadata(records, cfg)
    spans = json.loads(args.spans.read_text(encoding="utf-8"))["spans"]

    loader = RestrictedFeatureLoader(records, args.features, allowed_sample_ids=split["train"] + split["val"])
    train = load_samples(loader, split["train"], spans, cfg)
    val = load_samples(loader, split["val"], spans, cfg)
    train_val_opened = list(loader.opened_sample_ids)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = [train_seed(int(seed), train, val, cfg, args.out_dir) for seed in cfg["seeds"]]
    states = {run["seed"]: run.pop("state") for run in runs}
    winner = max(runs, key=winner_selection_key)
    gates = validation_gate(winner["val"])

    def test_loader_factory() -> RestrictedFeatureLoader:
        return RestrictedFeatureLoader(records, args.features, allowed_sample_ids=split["test"])

    def evaluate_test(loader_: RestrictedFeatureLoader) -> dict[str, Any]:
        test_samples = load_samples(loader_, split["test"], spans, cfg)
        model = rebuild_model(winner["seed"], states[winner["seed"]], cfg, cfg["device"])
        model.eval()
        return evaluate_split(
            model,
            test_samples,
            cfg,
            same_donors=same_image_donor_map(test_samples),
            cross_donors=cross_image_donor_map(test_samples),
            seed=int(cfg["bootstrap_seed"]) + 5000,
            reps=int(cfg["bootstrap_repetitions"]),
        )

    test_result = maybe_evaluate_internal_test(
        gate_pass=bool(gates["pass"]),
        test_loader_factory=test_loader_factory,
        evaluate_fn=evaluate_test,
    )

    status = (
        "VSR_ORDERED_PAIR_STUDENT_PASS" if gates["pass"] else "VSR_ORDERED_PAIR_STUDENT_NOT_ESTABLISHED"
    )
    report = {
        "status": status,
        "scope": (
            "VSR zero-shot train internal grouped train/val/test only; no official VSR dev/test "
            "and no label/relation annotation input"
        ),
        "architecture": ARCHITECTURE,
        "config": str(args.config),
        "interpretation": {
            "teacher_proxy_not_task_accuracy": True,
            "float_software_not_fpga_measurement": True,
            "note": (
                "All metrics are distillation proxies against the frozen B1 margin teacher; "
                "they are not VSR task accuracy. Every number is float software, not a ZCU102 measurement."
            ),
        },
        "split_counts": {name: len(values) for name, values in split.items()},
        "same_donor_coverage": {
            "train": same_donor_coverage(train, same_image_donor_map(train)),
            "val": same_donor_coverage(val, same_image_donor_map(val)),
        },
        "opened_sample_ids": {"train_val": train_val_opened},
        "runs": runs,
        "winner_seed": winner["seed"],
        "winner_checkpoint": winner["checkpoint"],
        "winner_val": winner["val"],
        "gates": gates,
        "internal_test": test_result["internal_test"],
        "test_npz_opened": test_result["test_npz_opened"],
        "official_vsr_dev_used": False,
        "official_vsr_test_opened": False,
        "hardware_contract": {
            "fixed_budget": FROZEN_BUDGET,
            "partner_k": FROZEN_PARTNER_K,
            "max_prompt_tokens": FROZEN_MAX_PROMPT_TOKENS,
            "note": "analytic interface contract, not ZCU102 measurement",
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{status} winner={winner['seed']} gates={gates['pass']} test_opened={len(test_result['test_npz_opened'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
