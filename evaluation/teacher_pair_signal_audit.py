#!/usr/bin/env python3
"""Train-only teacher paired-signal (grounding vs pointing) learnability audit.

This module is **descriptive diagnosis only**, on the **train split only**.  It
never trains, never tunes a hyper-parameter, never loads a 4B model, never uses
a GPU, never touches the network and never reads a val/test feature cache.  It
answers one bounded question:

    Do the existing dense teacher targets of the two prompts of the *same*
    image (grounding / pointing) carry a non-zero, stable and spatially
    distinguishable paired signal?  And how strong / how heterogeneous is it?

It deliberately does **not** decide whether a future
``prompt_mean_vector`` scorer with a paired contrastive/delta loss should be
retrained, and it fixes **no promotion threshold**.

Isolation contract (enforced and unit-tested):

* the train/val/test split is computed **only from index metadata**
  (``sample_id`` + ``source_image_sha256``) using the frozen
  :func:`evaluation.prompt_projection_distillation.split_sample_ids`;
* only feature caches whose ``sample_id`` belongs to the **train** split are
  opened (``load_feature_cache`` is never called for a val/test path);
* the input aggregate SHA256 is taken over the opened train npz files only, so
  rewriting or corrupting a val/test npz cannot move any train metric or hash.

Every correlation / top-K primitive is imported from the frozen layer
(:mod:`evaluation.prompt_projection_distillation`).  The only new math is the
natural-log Jensen-Shannon divergence (this audit's spec asks for nats), the
numerically-stabilised symmetric KL, the total-variation distance and the
grid-coordinate distances, all documented below.
"""

from __future__ import annotations

import argparse
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

from evaluation.metrics import load_jsonl
from evaluation.prompt_projection_distillation import (
    DistillationContractError,
    config_sha256,
    jaccard,
    load_distillation_config,
    load_feature_cache,
    pearson,
    spearman,
    split_group_summary,
    split_sample_ids,
    teacher_topk_recall,
    topk_indices,
    validate_source_image_sha256,
)
from evaluation.train_prompt_projection import index_file_sha256

__all__ = [
    "describe",
    "grid_distance",
    "js_divergence_nats",
    "kl_divergence_nats",
    "load_index_metadata",
    "load_train_samples",
    "per_image_metrics",
    "render_markdown",
    "run_audit",
    "symmetric_kl_nats",
    "total_variation_distance",
]

# ---------------------------------------------------------------------------
# Frozen audit constants (fixed before this audit runs; no pass/fail threshold).
# ---------------------------------------------------------------------------
REPORT_VERSION = "1.0"
REPORT_STEM = "2026-10-02-teacher-pair-signal-train-only"
AUDIT_SEED = 20261002
KS: tuple[int, ...] = (48, 64, 96)
TOLERANCE = 1e-9
_EPS = 1e-12
# Natural-log Jensen-Shannon divergence is bounded on [0, ln 2]; equality gives
# exactly 0.  Reported in nats, not bits.
JS_NATS_MAX = math.log(2.0)
EXPECTED_TRAIN_RECORDS = 318
EXPECTED_TRAIN_IMAGES = 159
# v2 diagnostic reference (train-side split): teacher prompt/image Spearman
# ratio.  Quoted only as context for the "max-over-prompt-token may already be
# weak" caveat; not a threshold and not recomputed here.
V2_TRAIN_PROMPT_IMAGE_RATIO_SPEARMAN = 0.2539

NATURAL_LOG_JS_NOTE = (
    "JS divergence uses natural log (nats); its range is [0, ln 2] = "
    f"[0, {JS_NATS_MAX:.6f}], 0 iff the two teacher_soft distributions are equal."
)
GRID_INDEX_NOTE = (
    "grid distance uses the row-major flat index convention g -> "
    "(row=g // grid_w, col=g % grid_w); it is reported N/A if group_count != "
    "grid_h * grid_w for any train pair."
)

# Continuous per-image metrics that receive full summary statistics.
PER_IMAGE_METRIC_KEYS: tuple[str, ...] = (
    "js_nats",
    "pearson_raw_normalized",
    "spearman_raw_normalized",
    "normalized_l1",
    "normalized_l2",
    "normalized_max_abs",
    "k48_effective",
    "k48_jaccard",
    "k48_overlap",
    "k48_disagreement",
    "k64_effective",
    "k64_jaccard",
    "k64_overlap",
    "k64_disagreement",
    "k96_effective",
    "k96_jaccard",
    "k96_overlap",
    "k96_disagreement",
    "argmax_same",
    "argmax_manhattan",
    "argmax_euclidean",
    "kl_grounding_pointing",
    "kl_pointing_grounding",
    "symmetric_kl",
    "tv",
    "max_abs_soft_diff",
    "exact_equal",
    "tolerance_equal",
)

# Metrics whose per-image values sit in [0, 1] and are summarised as fractions.
BOOLEAN_METRIC_KEYS: tuple[str, ...] = (
    "argmax_same",
    "exact_equal",
    "tolerance_equal",
)


# ---------------------------------------------------------------------------
# Distribution primitives (new math for this audit, documented)
# ---------------------------------------------------------------------------


def _as_distribution(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise DistillationContractError("distribution must not be empty")
    if not np.isfinite(array).all():
        raise DistillationContractError("distribution must be finite")
    if np.any(array < -1e-9):
        raise DistillationContractError("distribution must be non-negative")
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= _EPS:
        raise DistillationContractError("distribution must have positive mass")
    return array / total


def js_divergence_nats(p: Any, q: Any) -> float:
    """Jensen-Shannon divergence in **nats** (natural log), range ``[0, ln 2]``.

    ``JS = 0.5 KL(p || m) + 0.5 KL(q || m)`` with ``m = (p + q) / 2``; the
    denominator is clamped to ``_EPS`` so a zero in the mixture cannot produce a
    ``nan``.  It is symmetric and equals 0 iff ``p == q``.
    """
    a = _as_distribution(p)
    b = _as_distribution(q)
    if a.shape != b.shape:
        raise DistillationContractError("js_divergence_nats inputs must share a shape")
    m = 0.5 * (a + b)

    def _kl(x: np.ndarray, y: np.ndarray) -> float:
        mask = x > 0.0
        return float(np.sum(x[mask] * (np.log(x[mask]) - np.log(np.clip(y[mask], _EPS, None)))))

    return 0.5 * _kl(a, m) + 0.5 * _kl(b, m)


def kl_divergence_nats(p: Any, q: Any) -> float:
    """Asymmetric ``KL(p || q)`` in nats with an ``_EPS`` denominator clamp."""
    a = _as_distribution(p)
    b = _as_distribution(q)
    if a.shape != b.shape:
        raise DistillationContractError("kl_divergence_nats inputs must share a shape")
    mask = a > 0.0
    return float(np.sum(a[mask] * (np.log(a[mask]) - np.log(np.clip(b[mask], _EPS, None)))))


def symmetric_kl_nats(p: Any, q: Any) -> float:
    """Numerically-stabilised symmetric KL: ``0.5 (KL(p||q) + KL(q||p))``."""
    return 0.5 * (kl_divergence_nats(p, q) + kl_divergence_nats(q, p))


def total_variation_distance(p: Any, q: Any) -> float:
    """Total variation distance ``0.5 * sum |p - q|`` in ``[0, 1]``."""
    a = _as_distribution(p)
    b = _as_distribution(q)
    if a.shape != b.shape:
        raise DistillationContractError("total_variation_distance inputs must share a shape")
    return 0.5 * float(np.abs(a - b).sum())


def grid_distance(
    first: int, second: int, *, grid_h: int, grid_w: int
) -> dict[str, float]:
    """Manhattan / Euclidean grid distance between two row-major flat indices."""
    width = int(grid_w)
    size = int(grid_h) * width
    for name, value in (("first", first), ("second", second)):
        if not 0 <= int(value) < size:
            raise DistillationContractError(
                f"grid_distance {name} index {value} outside [0, {size})"
            )
    r0, c0 = divmod(int(first), width)
    r1, c1 = divmod(int(second), width)
    drow = abs(r0 - r1)
    dcol = abs(c0 - c1)
    return {
        "manhattan": float(drow + dcol),
        "euclidean": float(math.hypot(drow, dcol)),
    }


def describe(values: Sequence[float]) -> dict[str, Any]:
    """count/mean/median/std/p5/p25/p75/p95/min/max for one metric."""
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    n = int(array.size)
    if n == 0:
        return {key: None for key in ("mean", "median", "std", "p5", "p25", "p75", "p95", "min", "max")} | {
            "count": 0
        }
    p5, p25, p75, p95 = np.percentile(array, [5.0, 25.0, 75.0, 95.0])
    return {
        "count": n,
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if n > 1 else 0.0,
        "p5": float(p5),
        "p25": float(p25),
        "p75": float(p75),
        "p95": float(p95),
        "min": float(array.min()),
        "max": float(array.max()),
    }


# ---------------------------------------------------------------------------
# Metadata-only index load + train-only feature cache load
# ---------------------------------------------------------------------------


def load_index_metadata(index_path: Path) -> list[dict[str, Any]]:
    """Read index metadata only; never opens a feature cache."""
    records = load_jsonl(index_path)
    if not records:
        raise DistillationContractError(f"index {index_path} contains no records")
    seen: set[str] = set()
    metadata: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id in seen:
            raise DistillationContractError(f"duplicate index sample_id {sample_id}")
        seen.add(sample_id)
        metadata.append(
            {
                "sample_id": sample_id,
                "task": str(record["task"]),
                "npz": str(record["npz"]),
                "npz_sha256": str(record.get("npz_sha256", "")),
                "group_count": int(record["group_count"]),
                "grid_h": int(record["grid_h"]),
                "grid_w": int(record["grid_w"]),
                "prompt_token_count": int(record.get("prompt_token_count", 0)),
                "source_image_sha256": validate_source_image_sha256(
                    record["source_image_sha256"]
                ),
            }
        )
    return metadata


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


def aggregate_npz_sha256(entries: Sequence[tuple[str, str]]) -> str:
    """Aggregate SHA256 over sorted ``(sample_id, file_sha256)`` train pairs."""
    digest = hashlib.sha256()
    for sample_id, file_sha256 in sorted(entries):
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_train_samples(
    index_path: Path,
    features_dir: Path | None,
    train_ids: Sequence[str],
    *,
    opened: set[str] | None = None,
    cache_loader: Callable[..., dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    """Load **only** the train feature caches; return samples, aggregate SHA, paths.

    ``opened`` (when given) receives every resolved npz path that was touched, so
    a test can assert that no val/test cache was ever opened.  ``cache_loader``
    defaults to the frozen :func:`load_feature_cache`; tests monkeypatch it.
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
                "teacher_raw": np.asarray(cache["teacher_scores"], dtype=np.float64),
                "teacher_normalized": np.asarray(
                    cache["teacher_scores_normalized"], dtype=np.float64
                ),
                "teacher_soft": np.asarray(cache["teacher_soft_target"], dtype=np.float64),
            }
        )
    return samples, aggregate_npz_sha256(hashes), resolved_paths


# ---------------------------------------------------------------------------
# Per-image paired metrics
# ---------------------------------------------------------------------------


def per_image_metrics(
    grounding: Mapping[str, Any],
    pointing: Mapping[str, Any],
    *,
    ks: Sequence[int] = KS,
) -> dict[str, Any]:
    """All descriptive paired metrics for one train image's two prompts."""
    g_soft = _as_distribution(grounding["teacher_soft"])
    p_soft = _as_distribution(pointing["teacher_soft"])
    g_raw = np.asarray(grounding["teacher_raw"], dtype=np.float64)
    p_raw = np.asarray(pointing["teacher_raw"], dtype=np.float64)
    g_norm = np.asarray(grounding["teacher_normalized"], dtype=np.float64)
    p_norm = np.asarray(pointing["teacher_normalized"], dtype=np.float64)

    grid_h = int(grounding["grid_h"])
    grid_w = int(grounding["grid_w"])
    group_count = int(grounding["group_count"])
    if group_count != g_soft.size or group_count != p_soft.size:
        raise DistillationContractError(
            f"{grounding['sample_id']}: group_count != teacher vector length"
        )
    if group_count != int(pointing["group_count"]):
        raise DistillationContractError(
            f"{grounding['sample_id']}: grounding/pointing group_count differ"
        )
    # grid_h * grid_w may disagree with G in principle; report N/A then.
    grid_consistent = group_count == grid_h * grid_w

    argmax_g = int(np.argmax(g_raw))
    argmax_p = int(np.argmax(p_raw))
    argmax_same = argmax_g == argmax_p
    if grid_consistent:
        distance = grid_distance(argmax_g, argmax_p, grid_h=grid_h, grid_w=grid_w)
        manhattan = distance["manhattan"]
        euclidean = distance["euclidean"]
    else:
        manhattan = None
        euclidean = None

    soft_diff = np.abs(g_soft - p_soft)
    row: dict[str, Any] = {
        "image_sha256": grounding["source_image_sha256"],
        "grounding_sample_id": grounding["sample_id"],
        "pointing_sample_id": pointing["sample_id"],
        "group_count": group_count,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "grid_consistent": bool(grid_consistent),
        "js_nats": js_divergence_nats(g_soft, p_soft),
        "pearson_raw_normalized": pearson(g_raw, g_norm),
        "spearman_raw_normalized": spearman(g_raw, g_norm),
        "normalized_l1": float(np.abs(g_norm - p_norm).sum()),
        "normalized_l2": float(np.linalg.norm(g_norm - p_norm)),
        "normalized_max_abs": float(np.abs(g_norm - p_norm).max()),
        "argmax_same": 1.0 if argmax_same else 0.0,
        "argmax_grounding": argmax_g,
        "argmax_pointing": argmax_p,
        "argmax_manhattan": manhattan,
        "argmax_euclidean": euclidean,
        "kl_grounding_pointing": kl_divergence_nats(g_soft, p_soft),
        "kl_pointing_grounding": kl_divergence_nats(p_soft, g_soft),
        "symmetric_kl": symmetric_kl_nats(g_soft, p_soft),
        "tv": total_variation_distance(g_soft, p_soft),
        "max_abs_soft_diff": float(soft_diff.max()),
        "exact_equal": 1.0 if np.array_equal(g_soft, p_soft) else 0.0,
        "tolerance_equal": 1.0 if float(soft_diff.max()) <= TOLERANCE else 0.0,
    }
    for k in ks:
        effective = min(int(k), group_count)
        g_top = topk_indices(g_raw, effective)
        p_top = topk_indices(p_raw, effective)
        row[f"k{int(k)}_effective"] = float(effective)
        row[f"k{int(k)}_jaccard"] = float(jaccard(g_top, p_top))
        row[f"k{int(k)}_overlap"] = float(teacher_topk_recall(g_top, p_top))
        row[f"k{int(k)}_disagreement"] = float(1.0 - jaccard(g_top, p_top))
        row[f"k{int(k)}_exceeds_g"] = bool(int(k) > group_count)
    return row


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _fraction(rows: Sequence[Mapping[str, Any]], key: str, *, predicate: Callable[[float], bool]) -> float:
    if not rows:
        return 0.0
    hits = sum(1 for row in rows if predicate(float(row[key])))
    return hits / float(len(rows))


def aggregate_rows(rows: Sequence[Mapping[str, Any]], ks: Sequence[int]) -> dict[str, Any]:
    """Summary statistics, fractions and G-stratified summaries."""
    stats = {key: describe([row[key] for row in rows]) for key in PER_IMAGE_METRIC_KEYS}
    fractions = {
        "exact_identical_fraction": _fraction(rows, "exact_equal", predicate=lambda v: v == 1.0),
        "tolerance_1e-9_identical_fraction": _fraction(
            rows, "tolerance_equal", predicate=lambda v: v == 1.0
        ),
        "js_positive_fraction": _fraction(rows, "js_nats", predicate=lambda v: v > 0.0),
        "tv_positive_fraction": _fraction(rows, "tv", predicate=lambda v: v > 0.0),
        "argmax_different_fraction": _fraction(
            rows, "argmax_same", predicate=lambda v: v == 0.0
        ),
        "grounding_swapped_kl_positive_fraction": _fraction(
            rows, "kl_grounding_pointing", predicate=lambda v: v > 0.0
        ),
        "pointing_swapped_kl_positive_fraction": _fraction(
            rows, "kl_pointing_grounding", predicate=lambda v: v > 0.0
        ),
        "both_directions_swapped_kl_positive_fraction": (
            sum(
                1
                for row in rows
                if float(row["kl_grounding_pointing"]) > 0.0
                and float(row["kl_pointing_grounding"]) > 0.0
            )
            / float(len(rows))
            if rows
            else 0.0
        ),
    }
    mean_disagreement = {
        f"k{int(k)}": float(np.mean([row[f"k{int(k)}_disagreement"] for row in rows]))
        if rows
        else None
        for k in ks
    }
    mean_jaccard = {
        f"k{int(k)}": float(np.mean([row[f"k{int(k)}_jaccard"] for row in rows]))
        if rows
        else None
        for k in ks
    }
    mean_overlap = {
        f"k{int(k)}": float(np.mean([row[f"k{int(k)}_overlap"] for row in rows]))
        if rows
        else None
        for k in ks
    }
    effective_ks = {
        f"k{int(k)}": sorted({int(row[f"k{int(k)}_effective"]) for row in rows}) for k in ks
    }
    exceeds_g_counts = {
        f"k{int(k)}": sum(1 for row in rows if row[f"k{int(k)}_exceeds_g"]) for k in ks
    }

    strata: dict[str, Any] = {}
    by_group: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[int(row["group_count"])].append(row)
    for group_count in sorted(by_group):
        group_rows = by_group[group_count]
        strata[str(group_count)] = {
            "image_count": len(group_rows),
            "mean_js_nats": float(np.mean([row["js_nats"] for row in group_rows])),
            "mean_symmetric_kl": float(np.mean([row["symmetric_kl"] for row in group_rows])),
            "mean_tv": float(np.mean([row["tv"] for row in group_rows])),
            "mean_pearson_raw_normalized": float(
                np.mean([row["pearson_raw_normalized"] for row in group_rows])
            ),
            "mean_spearman_raw_normalized": float(
                np.mean([row["spearman_raw_normalized"] for row in group_rows])
            ),
            "mean_disagreement": {
                f"k{int(k)}": float(
                    np.mean([row[f"k{int(k)}_disagreement"] for row in group_rows])
                )
                for k in ks
            },
            "argmax_different_fraction": _fraction(
                group_rows, "argmax_same", predicate=lambda v: v == 0.0
            ),
            "exact_identical_fraction": _fraction(
                group_rows, "exact_equal", predicate=lambda v: v == 1.0
            ),
        }

    return {
        "image_count": len(rows),
        "independent_image_units": len(rows),
        "sample_record_count": 2 * len(rows),
        "metric_stats": stats,
        "fractions": fractions,
        "mean_disagreement": mean_disagreement,
        "mean_jaccard": mean_jaccard,
        "mean_overlap": mean_overlap,
        "effective_k_values": effective_ks,
        "k_exceeds_g_counts": exceeds_g_counts,
        "strata_by_group_count": strata,
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
    index_path: Path,
    features_dir: Path | None = None,
    seed: int = AUDIT_SEED,
    ks: Sequence[int] = KS,
    cache_loader: Callable[..., dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Full train-only audit report plus the per-image CSV rows."""
    config = load_distillation_config(config_path)
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
    val_test_paths = {
        str(_resolve_npz(record, Path(features_dir) if features_dir else Path(index_path).parent))
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
    rows: list[dict[str, Any]] = []
    pair_failures: list[str] = []
    for image_hash in sorted(by_image):
        tasks = by_image[image_hash]
        if set(tasks) != {"grounding", "pointing"}:
            pair_failures.append(f"image {image_hash}: tasks={sorted(tasks)}")
            continue
        rows.append(per_image_metrics(tasks["grounding"], tasks["pointing"], ks=ks))

    aggregate = aggregate_rows(rows, ks)
    grid_consistent = all(bool(row["grid_consistent"]) for row in rows) and bool(rows)
    grid_note = GRID_INDEX_NOTE if grid_consistent else (
        GRID_INDEX_NOTE + " At least one train pair had group_count != grid_h * grid_w, "
        "so argmax grid distances are N/A for those rows."
    )

    provenance = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "seed": int(seed),
        "split_seed": int(split_seed),
        "config_sha256": config_sha256(config_path),
        "index_sha256": index_file_sha256(index_path),
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
        "scope": (
            "train-only descriptive teacher paired-signal (grounding vs pointing) "
            "learnability audit; no training, no tuning, no promotion threshold, "
            "no val/test feature cache read, CPU only"
        ),
        "inputs": {
            "config": str(config_path),
            "index": str(index_path),
            "features_dir": str(features_dir) if features_dir else str(Path(index_path).parent),
        },
        "split_counts": {name: len(value) for name, value in split.items()},
        "split_group_summary": group_summary,
        "isolation": isolation,
        "provenance": provenance,
        "integrity": {
            "train_record_count": len(train_samples),
            "train_image_count": len(rows),
            "expected_train_records": EXPECTED_TRAIN_RECORDS,
            "expected_train_images": EXPECTED_TRAIN_IMAGES,
            "matches_expected": (
                len(train_samples) == EXPECTED_TRAIN_RECORDS
                and len(rows) == EXPECTED_TRAIN_IMAGES
            ),
            "pair_failures": pair_failures,
            "grid_index_consistent": grid_consistent,
        },
        "ks": [int(k) for k in ks],
        "js_base": "natural_log",
        "js_range": [0.0, JS_NATS_MAX],
        "notes": {
            "js": NATURAL_LOG_JS_NOTE,
            "grid": grid_note,
            "independent_units": (
                "the independent unit is the image (each image contributes one "
                "grounding/pointing pair); the two directions are NOT two "
                "independent samples"
            ),
            "teacher_reduction": (
                "the existing teacher target is generated by "
                "max-over-prompt-token cosine of the full dense vision tower, so a "
                "non-zero paired signal can still be weak; the v2 train-side "
                f"prompt/image Spearman ratio was ~{V2_TRAIN_PROMPT_IMAGE_RATIO_SPEARMAN:.4f}"
            ),
            "no_threshold": (
                "this audit fixes no promotion threshold and makes no claim that a "
                "future prompt_mean_vector + paired contrastive/delta retrain would "
                "succeed; it only reports whether the paired teacher signal is "
                "non-zero, how strong and how heterogeneous it is"
            ),
            "not_task_accuracy": (
                "all numbers describe teacher-target distinguishability, never "
                "CV-Bench task accuracy or student learnability"
            ),
        },
        "oracle_swap_control": {
            "own_kl": 0.0,
            "own_kl_note": "KL(y||y) = 0 by construction for each direction",
            "kl_grounding_pointing": None,
            "kl_pointing_grounding": None,
        },
        "aggregate": aggregate,
    }
    report["oracle_swap_control"]["kl_grounding_pointing"] = aggregate["metric_stats"][
        "kl_grounding_pointing"
    ]
    report["oracle_swap_control"]["kl_pointing_grounding"] = aggregate["metric_stats"][
        "kl_pointing_grounding"
    ]
    return report, rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

CSV_FIELDS: tuple[str, ...] = (
    "image_sha256",
    "grounding_sample_id",
    "pointing_sample_id",
    "group_count",
    "grid_h",
    "grid_w",
    "grid_consistent",
    "js_nats",
    "pearson_raw_normalized",
    "spearman_raw_normalized",
    "normalized_l1",
    "normalized_l2",
    "normalized_max_abs",
    "k48_effective",
    "k48_jaccard",
    "k48_overlap",
    "k48_disagreement",
    "k48_exceeds_g",
    "k64_effective",
    "k64_jaccard",
    "k64_overlap",
    "k64_disagreement",
    "k64_exceeds_g",
    "k96_effective",
    "k96_jaccard",
    "k96_overlap",
    "k96_disagreement",
    "k96_exceeds_g",
    "argmax_same",
    "argmax_grounding",
    "argmax_pointing",
    "argmax_manhattan",
    "argmax_euclidean",
    "kl_grounding_pointing",
    "kl_pointing_grounding",
    "symmetric_kl",
    "tv",
    "max_abs_soft_diff",
    "exact_equal",
    "tolerance_equal",
)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv as _csv

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


def _stats_table(stats: Mapping[str, Mapping[str, Any]]) -> list[str]:
    header = "| metric | count | mean | median | std | p5 | p25 | p75 | p95 | min | max |"
    separator = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, separator]
    for key in PER_IMAGE_METRIC_KEYS:
        if key not in stats:
            continue
        row = stats[key]
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
    aggregate = report["aggregate"]
    fractions = aggregate["fractions"]
    integrity = report["integrity"]
    notes = report["notes"]
    split_counts = report["split_counts"]
    lines: list[str] = []
    lines.append(f"# Teacher 成对信号可学性审计（train-only，{REPORT_STEM}）")
    lines.append("")
    lines.append(
        "本轮只做**描述性诊断**：判断同一图片 grounding/pointing 两个 prompt 的现有 "
        "dense teacher target 是否提供非零、稳定且空间上可区分的成对信号，以及强弱与异质性。"
        "不训练、不调参、不设通过阈值、不读 val/test 特征缓存、不加载模型、不用 GPU、不联网。"
    )
    lines.append("")
    lines.append("## 1. 范围与硬性隔离")
    lines.append("")
    lines.append(
        f"- split 仅由 index metadata（sample_id + source_image_sha256）用冻结 "
        f"`split_sample_ids` 计算：train/val/test = "
        f"{split_counts['train']}/{split_counts['val']}/{split_counts['test']} 条记录。"
    )
    lines.append(
        f"- 只打开 train NPZ：opened_npz_count={report['isolation']['opened_npz_count']}，"
        f"val_test_npz_opened={report['isolation']['val_test_npz_opened']}，"
        f"opened_only_train_npz={report['isolation']['opened_only_train_npz']}。"
    )
    lines.append(
        f"- train 记录/图片 = {integrity['train_record_count']}/{integrity['train_image_count']}"
        f"（期望 {integrity['expected_train_records']}/{integrity['expected_train_images']}，"
        f"matches_expected={integrity['matches_expected']}）；status={report['status']}。"
    )
    lines.append(
        f"- 有效独立单位 = **image**（{aggregate['independent_image_units']} 个），"
        f"不把 grounding/pointing 两方向当作 2 个独立样本。"
    )
    lines.append("")
    lines.append("## 2. 输入与 provenance")
    lines.append("")
    lines.append(f"- python {provenance['python']} / numpy {provenance['numpy']} / seed {provenance['seed']}；split_seed {provenance['split_seed']}。")
    lines.append(f"- config SHA256 `{provenance['config_sha256']}`")
    lines.append(f"- index SHA256 `{provenance['index_sha256']}`")
    lines.append(f"- script SHA256 `{provenance['script_sha256']}`")
    lines.append(
        f"- 输入 NPZ aggregate SHA256 `{provenance['input_npz_aggregate_sha256']}`"
        f"（仅 train，{provenance['input_npz_count']} 个文件）"
    )
    lines.append("")
    lines.append("## 3. 指标定义")
    lines.append("")
    lines.append(f"- JS 散度（自然对数/nats）：{notes['js']}")
    lines.append(f"- grid 距离：{notes['grid']}")
    lines.append("- 对称 KL：0.5·(KL(y_g||y_p) + KL(y_p||y_g))，分母 clamp 到 1e-12。")
    lines.append("- TV：0.5·Σ|y_g − y_p|，范围 [0, 1]。")
    lines.append("- top-K：K=48/64/96，K>G 时取 min(K,G) 并在 CSV 标记 kK_exceeds_g。")
    lines.append("- oracle 互换对照：own KL(y||y)=0，swapped KL(y_g||y_p) 与 KL(y_p||y_g)；严格优势 = swapped KL > 0。")
    lines.append("")
    lines.append("## 4. 聚合统计（逐图，n={}）".format(aggregate["image_count"]))
    lines.append("")
    lines.extend(_stats_table(aggregate["metric_stats"]))
    lines.append("")
    lines.append("## 5. 比例与 oracle 互换对照")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("| --- | --- |")
    for key in (
        "exact_identical_fraction",
        "tolerance_1e-9_identical_fraction",
        "js_positive_fraction",
        "tv_positive_fraction",
        "argmax_different_fraction",
        "grounding_swapped_kl_positive_fraction",
        "pointing_swapped_kl_positive_fraction",
        "both_directions_swapped_kl_positive_fraction",
    ):
        lines.append(f"| {key} | {_fmt(fractions[key])} |")
    lines.append("")
    lines.append(
        "- K=48/64/96 平均 disagreement（1 − Jaccard）："
        + ", ".join(
            f"K={key}: {_fmt(value)}" for key, value in aggregate["mean_disagreement"].items()
        )
    )
    lines.append(
        "- K=48/64/96 平均 overlap/recall："
        + ", ".join(
            f"K={key}: {_fmt(value)}" for key, value in aggregate["mean_overlap"].items()
        )
    )
    lines.append(
        "- K=48/64/96 平均 Jaccard："
        + ", ".join(
            f"K={key}: {_fmt(value)}" for key, value in aggregate["mean_jaccard"].items()
        )
    )
    lines.append(
        "- 有效 K 取值（当 K>G 时证据）："
        + "; ".join(
            f"K={key}->{aggregate['effective_k_values'][key]}"
            for key in aggregate["effective_k_values"]
        )
    )
    lines.append(
        "- K>G 出现图片数："
        + ", ".join(
            f"K={key}: {aggregate['k_exceeds_g_counts'][key]}"
            for key in aggregate["k_exceeds_g_counts"]
        )
    )
    lines.append("")
    lines.append("## 6. 按共同 group_count（G）分层")
    lines.append("")
    lines.append(
        "| G | image_count | mean_js_nats | mean_symmetric_kl | mean_tv | "
        "mean_pearson_raw_normalized | mean_spearman_raw_normalized | "
        "mean_disagreement_k48 | mean_disagreement_k64 | mean_disagreement_k96 | "
        "argmax_different_fraction | exact_identical_fraction |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for group_count, stratum in aggregate["strata_by_group_count"].items():
        disagreement = stratum["mean_disagreement"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(group_count),
                    _fmt(stratum["image_count"]),
                    _fmt(stratum["mean_js_nats"]),
                    _fmt(stratum["mean_symmetric_kl"]),
                    _fmt(stratum["mean_tv"]),
                    _fmt(stratum["mean_pearson_raw_normalized"]),
                    _fmt(stratum["mean_spearman_raw_normalized"]),
                    _fmt(disagreement["k48"]),
                    _fmt(disagreement["k64"]),
                    _fmt(disagreement["k96"]),
                    _fmt(stratum["argmax_different_fraction"]),
                    _fmt(stratum["exact_identical_fraction"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## 7. 解释与限制")
    lines.append("")
    lines.append(
        "- 本审计只回答 teacher pair signal 是否非零、强弱和异质性；"
        "**不能**据此声称固定 `prompt_mean_vector` scorer + paired contrastive/delta loss 重训必然成功。"
    )
    lines.append(
        "- 现有 teacher 由 **max-over-prompt-token** 生成，已知 prompt/image signal ratio 约 0.27，"
        "因此即使 pair signal 非零也可能偏弱。"
    )
    lines.append(
        "- 未自创通过阈值；未访问 test 来定阈值；未读取任何 val/test 特征缓存。"
    )
    lines.append(
        "- 所有数值都是 internal teacher-target 诊断，不是 CV-Bench 任务精度，也不代表 student 可学。"
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
        "--index",
        type=Path,
        default=root
        / "artifacts"
        / "token_selector"
        / "prompt_projection_features"
        / "index.jsonl",
    )
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument(
        "--out-dir", type=Path, default=root / "reports" / "token_selector"
    )
    parser.add_argument("--seed", type=int, default=AUDIT_SEED)
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
        index_path=args.index,
        features_dir=args.features_dir,
        seed=args.seed,
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
    harness = {
        "status": "COMPUTED",
        "report": REPORT_STEM,
        "scope": report["scope"],
        "seed": report["provenance"]["seed"],
        "environment": {
            "python": report["provenance"]["python"],
            "numpy": report["provenance"]["numpy"],
            "platform": report["provenance"]["platform"],
        },
        "inputs": {
            "config_sha256": report["provenance"]["config_sha256"],
            "index_sha256": report["provenance"]["index_sha256"],
            "script_sha256": report["provenance"]["script_sha256"],
            "input_npz_aggregate_sha256": report["provenance"]["input_npz_aggregate_sha256"],
        },
        "output_sha256": output_hashes,
        "artifacts": {
            "script": str(Path(__file__).resolve().relative_to(root)),
            "csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(md_path),
            "focused_tests": "evaluation/tests/test_teacher_pair_signal_audit.py",
        },
        "isolation": report["isolation"],
        "key_metrics": {
            "train_images": report["integrity"]["train_image_count"],
            "train_records": report["integrity"]["train_record_count"],
            "js_positive_fraction": report["aggregate"]["fractions"]["js_positive_fraction"],
            "tv_positive_fraction": report["aggregate"]["fractions"]["tv_positive_fraction"],
            "argmax_different_fraction": report["aggregate"]["fractions"][
                "argmax_different_fraction"
            ],
            "exact_identical_fraction": report["aggregate"]["fractions"][
                "exact_identical_fraction"
            ],
            "mean_js_nats": report["aggregate"]["metric_stats"]["js_nats"]["mean"],
            "median_js_nats": report["aggregate"]["metric_stats"]["js_nats"]["median"],
            "mean_symmetric_kl": report["aggregate"]["metric_stats"]["symmetric_kl"]["mean"],
            "mean_tv": report["aggregate"]["metric_stats"]["tv"]["mean"],
            "mean_disagreement": report["aggregate"]["mean_disagreement"],
        },
    }
    args.harness_result.parent.mkdir(parents=True, exist_ok=True)
    args.harness_result.write_text(json.dumps(harness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.harness_durable.parent.mkdir(parents=True, exist_ok=True)
    args.harness_durable.write_text(
        json.dumps(harness, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"TEACHER_PAIR_SIGNAL_AUDIT status={report['status']} rows={len(rows)}")
    print(f"isolation opened_only_train_npz={report['isolation']['opened_only_train_npz']}")
    print(f"python={report['provenance']['python']} numpy={report['provenance']['numpy']} seed={report['provenance']['seed']}")
    print(f"input_config_sha256={report['provenance']['config_sha256']}")
    print(f"input_index_sha256={report['provenance']['index_sha256']}")
    print(f"input_script_sha256={report['provenance']['script_sha256']}")
    print(f"input_npz_aggregate_sha256={report['provenance']['input_npz_aggregate_sha256']}")
    for name, digest in output_hashes.items():
        print(f"output_{name}_sha256={digest}")
    print(f"harness_result={args.harness_result}")
    print(f"harness_durable={args.harness_durable}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
