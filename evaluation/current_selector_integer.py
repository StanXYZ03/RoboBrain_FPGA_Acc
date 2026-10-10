#!/usr/bin/env python3
"""Frozen software bridge for the 2026-10-03 "current Selector" (integer scorer).

This module adds **no new mathematics**.  It wires the already accepted
2026-10-03 chain into the real-model B1 path:

1. ``group_mean[G, 1024]`` float64
   -> :func:`evaluation.selector_int8_numerical_audit.quantize_activation`
      (one symmetric INT8 scale per image);
2. ``prompt_embeddings[T, 2560]`` + float ``Wt``
   -> :func:`evaluation.selector_int8_numerical_audit.primary_quantized_query`
      (float prompt-mean query, then one symmetric INT8 scale per query);
3. the accepted per-tensor INT8 ``Wv_q`` bundle
   -> :func:`evaluation.selector_integer_scorer_audit.integer_chain`
      (exact INT32 projection, signed INT64 dot, nonnegative INT64 norm2,
      exact floor sqrt, exact rational round-half-away, Q0.15 combine);
4. ``score_q`` -> ``combined = score_q / 2**15``
   -> :func:`evaluation.hardware_regular_token_selector.select_hardware_regular_groups`
      with ``combined_scores`` used verbatim (3x3 anchor + bounded Top-K +
      dynamic context reservation), i.e. the frozen software selection.

The deployment checkpoint is frozen to seed **20261003** because the accepted
compact HLS / IP / OOC / Zynq-BD chain is built from exactly that seed's Wv
bundle (``fpga_accel/integer_selector_pipeline_compact_hls`` fixtures and the
2026-10-03 scorer / interface / pipeline HLS reports reference only seed
20261003).  Seeds 20261004/20261005 appear only in the multi-seed numerical
audit statistics and never in an accepted hardware implementation.

This is a CPU-only, read-only bridge.  It loads no model and runs no GPU.  It
must never be described as FPGA execution or as end-to-end acceleration.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation import selector_int8_numerical_audit as base
from evaluation import selector_integer_scorer_audit as audit
from evaluation.hardware_regular_token_selector import (
    DEFAULT_LOCAL_CANDIDATE_CAP,
    MERGE_UNIT,
    REGION_GRID,
    HardwareSelection,
    LowRankProjection,
    select_hardware_regular_groups,
)
from evaluation.run_hf_hardware_regular_b1 import load_projection

ROOT = Path(__file__).resolve().parents[1]

#: Frozen deployment identity (seed selected only from accepted non-CV-Bench
#: hardware metadata; never from CV-Bench).
DEPLOYMENT_SEED = 20261003
DEPLOYMENT_CHECKPOINT = (
    "artifacts/token_selector/spatial_prompt_contrastive_trainval/"
    "B_SPATIAL_KL_CONTRASTIVE/20261003/best.npz"
)
DEPLOYMENT_CHECKPOINT_SHA256 = (
    "1f1d379e3229f310d11fee4ff620480e13fea9005abefc762c66f0d331ca72ff"
)
DEPLOYMENT_WV_PATH = (
    "artifacts/token_selector/selector_wv_scale_regularization/"
    "20261003/Wv_int8_rowmajor.bin"
)
DEPLOYMENT_WV_SHA256 = (
    "126bd8421dd128a318115c301bf9d1c7d336912150d19d2ec35203ed55d89d3b"
)
DEPLOYMENT_WV_SCALE_PATH = (
    "artifacts/token_selector/selector_wv_scale_regularization/"
    "20261003/Wv_per_tensor_scale_f32le.bin"
)
DEPLOYMENT_WV_SCALE_SHA256 = (
    "93585a3bd2d584a612b40d827f8d15f43c87acdc764d7e217d99df473428b13f"
)
SCORE_Q_SCALE = float(1 << 15)
BUDGETS = (48, 64, 96)


class CurrentSelectorError(ValueError):
    """A malformed input or a violated frozen-current-selector contract."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify(path: Path, expected_sha256: str, label: str) -> str:
    if not path.is_file():
        raise CurrentSelectorError(f"{label} does not exist: {path}")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise CurrentSelectorError(
            f"{label} sha256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return actual


def load_deployment_projection(
    root: Path = ROOT,
) -> tuple[LowRankProjection, dict[str, Any]]:
    """Load the frozen deployment checkpoint and verify its pinned SHA256."""
    path = Path(root) / DEPLOYMENT_CHECKPOINT
    projection, actual = load_projection(
        path, expected_sha256=DEPLOYMENT_CHECKPOINT_SHA256
    )
    metadata = {
        "seed": DEPLOYMENT_SEED,
        "path": DEPLOYMENT_CHECKPOINT,
        "sha256": actual,
        "Wt_shape": list(np.asarray(projection.Wt).shape),
        "Wv_shape": list(np.asarray(projection.Wv).shape),
    }
    return projection, metadata


def load_deployment_wv_bundle(root: Path = ROOT) -> dict[str, Any]:
    """Load the accepted per-tensor INT8 Wv tensor and its float32 scale."""
    wv_path = Path(root) / DEPLOYMENT_WV_PATH
    scale_path = Path(root) / DEPLOYMENT_WV_SCALE_PATH
    wv_sha = _verify(wv_path, DEPLOYMENT_WV_SHA256, "deployment Wv")
    scale_sha = _verify(scale_path, DEPLOYMENT_WV_SCALE_SHA256, "deployment Wv scale")
    wv_q = np.frombuffer(wv_path.read_bytes(), dtype=np.int8).reshape(
        audit.QUERY_DIM, audit.EMBED_DIM
    )
    scale = float(np.frombuffer(scale_path.read_bytes(), dtype="<f4")[0])
    if scale <= 0.0 or not np.isfinite(scale):
        raise CurrentSelectorError(
            f"deployment Wv scale must be finite and positive: {scale}"
        )
    return {
        "wv_q": wv_q.astype(np.int64),
        "wv_scale": scale,
        "wv_path": DEPLOYMENT_WV_PATH,
        "wv_sha256": wv_sha,
        "scale_path": DEPLOYMENT_WV_SCALE_PATH,
        "scale_sha256": scale_sha,
    }


def integer_scores(
    group_mean: Any,
    prompt_embeddings: Any,
    Wt: Any,
    wv_q: Any,
    grid_h: int,
    grid_w: int,
    *,
    budgets: Sequence[int] = BUDGETS,
) -> dict[str, Any]:
    """Run the frozen integer chain for one image and one prompt.

    Returns the exact ``score_q`` Q0.15 vector plus the intermediate
    activations and bounds needed to audit the fixed widths.  Raises
    :class:`CurrentSelectorError` if the received arrays are malformed.
    """
    mean = np.asarray(group_mean, dtype=np.float64)
    if mean.ndim != 2 or mean.shape[1] != audit.EMBED_DIM:
        raise CurrentSelectorError(
            f"group_mean must be [G, {audit.EMBED_DIM}], got {mean.shape}"
        )
    if int(grid_h) * int(grid_w) != mean.shape[0]:
        raise CurrentSelectorError(
            f"grid {grid_h}x{grid_w} does not match {mean.shape[0]} groups"
        )
    group_mean_q, group_mean_scale = base.quantize_activation(mean)
    query = base.primary_quantized_query(prompt_embeddings, Wt)
    chain = audit.integer_chain(
        group_mean_q,
        query["query_q"],
        wv_q,
        int(grid_h),
        int(grid_w),
        budgets=[int(value) for value in budgets],
    )
    return {
        "group_mean_q": group_mean_q,
        "group_mean_scale": float(group_mean_scale),
        "query_q": np.asarray(query["query_q"], dtype=np.int64),
        "query_scale": float(query["query_scale"]),
        "score_q": chain["score_q"],
        "semantic_norm_q15": chain["semantic_norm_q15"],
        "l2_norm_q15": chain["l2_norm_q15"],
        "selections": chain["selections"],
        "bounds": chain["bounds"],
    }


def score_q_to_combined(score_q: Any) -> np.ndarray:
    """Dequantise the Q0.15 score to the float ``combined`` consumed by selection."""
    values = np.asarray(score_q, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise CurrentSelectorError("score_q must be non-empty")
    return values / SCORE_Q_SCALE


def select_from_score_q(
    hidden_groups: Any,
    score_q: Any,
    grid_h: int,
    grid_w: int,
    budget: int,
    *,
    region_grid: int = REGION_GRID,
    local_candidate_cap: int = DEFAULT_LOCAL_CANDIDATE_CAP,
) -> HardwareSelection:
    """Frozen software selection on the integer ``score_q`` vector.

    ``hidden_groups`` stays float (``[G, 4, 1024]``) so the synthetic context
    group is pooled from real activations exactly as the deployed software path
    does.  Only ``combined`` comes from the integer chain; the selection code is
    reused verbatim.
    """
    combined = score_q_to_combined(score_q)
    if combined.size != int(hidden_groups.shape[0]):
        raise CurrentSelectorError(
            f"score_q has {combined.size} entries but {hidden_groups.shape[0]} groups"
        )
    return select_hardware_regular_groups(
        hidden_groups,
        int(grid_h),
        int(grid_w),
        budget=int(budget),
        combined_scores=combined,
        region_grid=region_grid,
        local_candidate_cap=local_candidate_cap,
    )


class IntegerSelectorScorer:
    """Callback used by the B1 integration's integer score mode.

    Holds the frozen ``Wt``/``Wv_q`` once and exposes one call per B1 selection.
    The last payload (``score_q``, query, bounds) is retained so the runner can
    write the per-sample audit fields.
    """

    def __init__(
        self,
        Wt: Any,
        wv_q: Any,
        *,
        budgets: Sequence[int] = BUDGETS,
    ) -> None:
        self.Wt = np.asarray(Wt)
        self.wv_q = np.asarray(wv_q, dtype=np.int64)
        self.budgets = tuple(int(value) for value in budgets)
        self.last_payload: dict[str, Any] | None = None

    def __call__(
        self,
        hidden_groups: Any,
        merged_h: int,
        merged_w: int,
        budget: int,
        text_embeddings: Any,
    ) -> dict[str, Any]:
        hidden = np.asarray(hidden_groups, dtype=np.float64)
        if hidden.ndim != 3 or hidden.shape[1] != MERGE_UNIT:
            raise CurrentSelectorError(
                f"hidden_groups must be [G, {MERGE_UNIT}, 1024], got {hidden.shape}"
            )
        if text_embeddings is None:
            raise CurrentSelectorError(
                "integer score mode requires the prompt text embeddings"
            )
        group_mean = hidden.mean(axis=1)
        payload = integer_scores(
            group_mean,
            text_embeddings,
            self.Wt,
            self.wv_q,
            merged_h,
            merged_w,
            budgets=self.budgets,
        )
        payload["budget"] = int(budget)
        payload["combined_scores"] = score_q_to_combined(payload["score_q"])
        self.last_payload = payload
        return payload


__all__ = [
    "BUDGETS",
    "CurrentSelectorError",
    "DEPLOYMENT_CHECKPOINT",
    "DEPLOYMENT_CHECKPOINT_SHA256",
    "DEPLOYMENT_SEED",
    "DEPLOYMENT_WV_PATH",
    "DEPLOYMENT_WV_SCALE_PATH",
    "IntegerSelectorScorer",
    "file_sha256",
    "integer_scores",
    "load_deployment_projection",
    "load_deployment_wv_bundle",
    "score_q_to_combined",
    "select_from_score_q",
]
