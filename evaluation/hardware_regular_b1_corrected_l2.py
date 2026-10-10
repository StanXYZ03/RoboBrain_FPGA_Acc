#!/usr/bin/env python3
"""Corrected patch-wise L2 scorer chain for the frozen B1 hardware selector.

Context
-------
A real-activation layer-5 replay proved (100/100) that the historical
``l2_anchor`` score is ``mean(norm(x_i))`` over the four raw patch vectors of a
merged group, whereas the current ``group_l2_saliency`` helper instead uses
``norm(mean(x_i))`` (norm of the group mean).  ``norm(mean)`` and ``mean(norm)``
differ whenever the four patches of a group cancel, so the current integer
scorer ranks a different quantity.  This experiment applies the verified
historical aggregation formula at B1 as a new same-cut comparison baseline; it
does not claim that the earlier historical run itself used the B1 cut.

This module adds the corrected patch-wise L2 chain and three scorer callbacks
that plug into the **unmodified** frozen B1 integration
:mod:`evaluation.hardware_regular_b1_integration` through its existing
``integer_scorer`` hook.  No frozen source, JSONL, report, checkpoint, HLS or
config is touched.

Arms
----
* ``corrected_l2_float`` (arm 1): per-group score is ``mean(norm(x_i))`` over
  the four raw patch vectors, then the existing float
  :func:`evaluation.hardware_regular_token_selector.minmax_normalize`.
* ``corrected_l2_integer`` (arm 2): all raw patch vectors are globally
  (per-image) symmetric-INT8 quantised with the accepted
  :func:`evaluation.selector_int8_numerical_audit.quantize_activation`; per
  patch compute the exact INT64 sum of squares and an integer floor square
  root; per group sum the four patch norms (rank/min-max equivalent to their
  mean); then the existing exact Q0.15
  :func:`evaluation.selector_integer_scorer_audit.minmax_normalize_q15`.
  No prompt term.
* ``prompt_plus_corrected_l2_integer`` (arm 3): the accepted integer semantic
  chain is preserved exactly (group-mean quantisation, frozen ``Wv_q``, frozen
  prompt query, semantic Q0.15); only the old ``norm(group_mean)`` L2 Q0.15 is
  replaced by arm 2's corrected patch-wise L2 Q0.15, then combined with the
  accepted ``lambda=0.5`` round-half-away/clamp rule via :func:`combine_q15`.

Every intermediate array and every fixed-width bound needed to audit the chain
is returned.  The module is CPU-safe (no CUDA at import, no model, no training)
and reuses accepted quantisation / rounding / min-max helpers instead of
duplicating their mathematics.

The only new numeric primitives are the patch-norm accumulation and
:func:`combine_q15`; both are checked against the accepted chain in the focused
tests.  ``combine_q15(semantic_norm_q15, l2_norm_q15)`` reproduces the accepted
``integer_chain`` ``score_q`` bit-for-bit when handed that chain's own
normalised semantic and L2 vectors.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation import selector_int8_numerical_audit as base
from evaluation import selector_integer_scorer_audit as audit
from evaluation.hardware_regular_token_selector import (
    MERGE_UNIT,
    minmax_normalize,
    validate_hidden,
)

#: Arm identifiers frozen by the preregistered config.
ARM_FLOAT = "corrected_l2_float"
ARM_INTEGER = "corrected_l2_integer"
ARM_PROMPT_INTEGER = "prompt_plus_corrected_l2_integer"
ALLOWED_ARMS = (ARM_FLOAT, ARM_INTEGER, ARM_PROMPT_INTEGER)

#: Fixed-point contract, aliased from the accepted audit so there is one source
#: of truth for the Q0.15 / Q16 widths.
SCORE_FRACTIONAL_BITS = audit.SCORE_FRACTIONAL_BITS
SCORE_Q_SCALE = 1 << SCORE_FRACTIONAL_BITS  # 32768
SCORE_Q_MAX = audit.SCORE_Q_MAX  # 32767
SEMANTIC_FRACTIONAL_BITS = audit.SEMANTIC_FRACTIONAL_BITS
LAMBDA_SEMANTIC = audit.LAMBDA_SEMANTIC  # 0.5
EMBED_DIM = audit.EMBED_DIM
INT64_MAX = audit.INT64_MAX
ACTIVATION_MAX_ABS = audit.ACTIVATION_MAX_ABS
WEIGHT_MAX_ABS = audit.WEIGHT_MAX_ABS
QUERY_MAX_ABS = audit.QUERY_MAX_ABS

#: Analytic patch-L2 width bound: each quantised patch entry is in [-128, 127]
#: so ``sum_1024 q_i^2 <= 1024 * 128 * 128`` and each patch norm is <= 4096.
PATCH_SQUARED_SUM_BOUND = EMBED_DIM * ACTIVATION_MAX_ABS * ACTIVATION_MAX_ABS
PATCH_NORM_BOUND = 4096
GROUP_PATCH_NORM_SUM_BOUND = MERGE_UNIT * PATCH_NORM_BOUND

BUDGETS = (48, 64, 96)


class CorrectedL2Error(ValueError):
    """A malformed input or a violated corrected-L2 contract."""


# ---------------------------------------------------------------------------
# Audit hashing
# ---------------------------------------------------------------------------
def array_sha256(array: Any) -> str:
    """Canonical SHA256 of an array's dtype, shape and C-order bytes.

    Used by the runner to record a portable, dtype-explicit digest of every
    score / index / metadata array in a raw record.
    """

    contiguous = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    header = f"{contiguous.dtype.str}:{contiguous.shape}".encode("ascii")
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Arm 1: corrected float patch-wise L2
# ---------------------------------------------------------------------------
def corrected_patch_l2_float(hidden_groups: Any) -> dict[str, Any]:
    """``mean(norm(x_i))`` over the four raw patches, then float min-max.

    This is the corrected no-prompt float baseline.  It is deliberately *not*
    :func:`evaluation.hardware_regular_token_selector.group_l2_saliency`, which
    computes ``norm(mean(x_i))`` and is the defective historical scorer.
    """

    hidden = validate_hidden(hidden_groups)
    # ``norm`` is taken per raw patch (axis=2) and only then averaged over the
    # four patches: ``mean(norm)``, never ``norm(mean)``.
    patch_norms = np.linalg.norm(hidden, axis=2)
    group_scores = patch_norms.mean(axis=1)
    combined = minmax_normalize(group_scores)
    return {
        "arm": ARM_FLOAT,
        "formula": "mean(norm(raw_patch_i)) over the 4 patches, then float min-max",
        "patch_norms": patch_norms,
        "group_scores": group_scores,
        "combined": combined,
        "bounds": {
            "group_count": int(hidden.shape[0]),
            "patch_norm_max": float(patch_norms.max()) if patch_norms.size else 0.0,
            "group_score_min": float(group_scores.min()) if group_scores.size else 0.0,
            "group_score_max": float(group_scores.max()) if group_scores.size else 0.0,
        },
    }


def score_corrected_l2_float(
    hidden_groups: Any,
    merged_h: int,
    merged_w: int,
    budget: int,
    text_embeddings: Any | None = None,
) -> dict[str, Any]:
    """B1 ``integer_scorer``-shaped callback for arm 1 (prompt ignored)."""

    del merged_h, merged_w, text_embeddings
    payload = corrected_patch_l2_float(hidden_groups)
    payload["budget"] = int(budget)
    payload["combined_scores"] = np.asarray(payload["combined"], dtype=np.float64)
    return payload


# ---------------------------------------------------------------------------
# Arm 2: corrected integer patch-wise L2
# ---------------------------------------------------------------------------
def _patch_norms_from_quantised(patch_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact INT64 patch sum-of-squares and integer floor sqrt per patch."""

    squared_sum = np.einsum("gpd,gpd->gp", patch_q, patch_q)
    squared_max = int(squared_sum.max()) if squared_sum.size else 0
    if squared_max > INT64_MAX:
        raise CorrectedL2Error("patch squared sum exceeds INT64")
    patch_norms = np.asarray(
        [[audit.floor_sqrt_int64(int(value)) for value in row] for row in squared_sum],
        dtype=np.int64,
    )
    return squared_sum, patch_norms


def corrected_patch_l2_integer(hidden_groups: Any) -> dict[str, Any]:
    """Bit-exact integer corrected patch-wise L2 (no prompt term)."""

    hidden = validate_hidden(hidden_groups)
    # One symmetric INT8 scale over *all* raw patch vectors of the image.
    patch_q, patch_scale = base.quantize_activation(hidden)
    squared_sum, patch_norms = _patch_norms_from_quantised(patch_q)
    group_norm_sum = patch_norms.sum(axis=1)
    l2_norm_q15 = audit.minmax_normalize_q15(group_norm_sum)
    combined = l2_norm_q15.astype(np.float64) / float(SCORE_Q_SCALE)
    return {
        "arm": ARM_INTEGER,
        "formula": (
            "global per-image INT8 raw patches; per-patch INT64 sum of squares "
            "+ integer floor sqrt; per-group sum of 4 patch norms; exact Q0.15 "
            "integer min-max"
        ),
        "patch_q": patch_q,
        "patch_scale": float(patch_scale),
        "patch_squared_sum": squared_sum,
        "patch_norms": patch_norms,
        "group_norm_sum": group_norm_sum,
        "l2_norm_q15": l2_norm_q15,
        "combined": combined,
        "bounds": {
            "group_count": int(hidden.shape[0]),
            "activation_absmax": float(np.max(np.abs(hidden))) if hidden.size else 0.0,
            "patch_squared_sum_max": int(squared_sum.max()) if squared_sum.size else 0,
            "patch_squared_sum_analytic_bound": int(PATCH_SQUARED_SUM_BOUND),
            "patch_norm_max": int(patch_norms.max()) if patch_norms.size else 0,
            "patch_norm_analytic_bound": int(PATCH_NORM_BOUND),
            "group_norm_sum_max": int(group_norm_sum.max()) if group_norm_sum.size else 0,
            "group_norm_sum_analytic_bound": int(GROUP_PATCH_NORM_SUM_BOUND),
            "l2_norm_q15_max": int(l2_norm_q15.max()) if l2_norm_q15.size else 0,
            "score_q_max_allowed": int(SCORE_Q_MAX),
        },
    }


def score_corrected_l2_integer(
    hidden_groups: Any,
    merged_h: int,
    merged_w: int,
    budget: int,
    text_embeddings: Any | None = None,
) -> dict[str, Any]:
    """B1 ``integer_scorer``-shaped callback for arm 2 (prompt ignored)."""

    del merged_h, merged_w, text_embeddings
    payload = corrected_patch_l2_integer(hidden_groups)
    payload["budget"] = int(budget)
    payload["combined_scores"] = np.asarray(payload["combined"], dtype=np.float64)
    return payload


# ---------------------------------------------------------------------------
# Accepted lambda=0.5 Q0.15 combine
# ---------------------------------------------------------------------------
def combine_q15(semantic_norm_q15: Any, l2_norm_q15: Any) -> np.ndarray:
    """Reproduce the accepted ``integer_chain`` lambda=0.5 Q0.15 combine.

    ``score_q = clamp(round_half_away((semantic + l2) / 2), 0, 32767)`` exactly
    as in :func:`evaluation.selector_integer_scorer_audit.integer_chain` step 6;
    the round-half-away division is the accepted exact integer helper.
    """

    semantic = np.asarray(semantic_norm_q15, dtype=np.int64).reshape(-1)
    l2 = np.asarray(l2_norm_q15, dtype=np.int64).reshape(-1)
    if semantic.shape != l2.shape:
        raise CorrectedL2Error(
            "semantic_norm_q15 and l2_norm_q15 must share a shape, got "
            f"{semantic.shape} vs {l2.shape}"
        )
    if semantic.size == 0:
        raise CorrectedL2Error("combine_q15 needs a non-empty score vector")
    for name, vector in (("semantic_norm_q15", semantic), ("l2_norm_q15", l2)):
        if int(vector.min()) < 0 or int(vector.max()) > SCORE_Q_MAX:
            raise CorrectedL2Error(
                f"{name} must lie in [0, {SCORE_Q_MAX}], got "
                f"[{int(vector.min())}, {int(vector.max())}]"
            )
    score_q = np.asarray(
        [
            min(
                max(
                    audit.round_half_away_div(
                        int(semantic_value) + int(l2_value), 2
                    ),
                    0,
                ),
                SCORE_Q_MAX,
            )
            for semantic_value, l2_value in zip(semantic, l2)
        ],
        dtype=np.uint16,
    )
    return score_q


# ---------------------------------------------------------------------------
# Arm 3: accepted semantic chain + corrected patch-wise L2
# ---------------------------------------------------------------------------
def prompt_plus_corrected_l2_integer(
    hidden_groups: Any,
    text_embeddings: Any,
    Wt: Any,
    wv_q: Any,
    grid_h: int,
    grid_w: int,
    *,
    budget: int,
) -> dict[str, Any]:
    """Accepted integer semantic chain with the corrected L2 Q0.15 term.

    The semantic half is produced by the accepted primitives
    (:func:`evaluation.selector_int8_numerical_audit.quantize_activation`,
    :func:`evaluation.selector_int8_numerical_audit.primary_quantized_query`,
    :func:`evaluation.selector_integer_scorer_audit.integer_chain`).  Only the
    L2 Q0.15 term is replaced by arm 2's corrected patch-wise value; the
    ``lambda=0.5`` combine is the accepted rule via :func:`combine_q15`.
    """

    hidden = validate_hidden(hidden_groups)
    if text_embeddings is None:
        raise CorrectedL2Error("arm 3 requires the prompt text embeddings")
    if Wt is None or wv_q is None:
        raise CorrectedL2Error("arm 3 requires the frozen Wt and Wv_q bundle")
    group_mean = hidden.mean(axis=1)
    group_mean_q, group_mean_scale = base.quantize_activation(group_mean)
    query = base.primary_quantized_query(text_embeddings, Wt)
    chain = audit.integer_chain(
        group_mean_q,
        query["query_q"],
        wv_q,
        int(grid_h),
        int(grid_w),
        budgets=[int(budget)],
    )
    corrected = corrected_patch_l2_integer(hidden)
    semantic_norm_q15 = chain["semantic_norm_q15"]
    l2_norm_q15 = corrected["l2_norm_q15"]
    score_q = combine_q15(semantic_norm_q15, l2_norm_q15)
    combined = score_q.astype(np.float64) / float(SCORE_Q_SCALE)
    return {
        "arm": ARM_PROMPT_INTEGER,
        "formula": (
            "accepted integer semantic Q0.15 combined with corrected patch-wise "
            "L2 Q0.15 via the accepted lambda=0.5 round-half-away/clamp rule"
        ),
        "group_mean_q": group_mean_q,
        "group_mean_scale": float(group_mean_scale),
        "query_q": np.asarray(query["query_q"], dtype=np.int64),
        "query_scale": float(query["query_scale"]),
        "semantic_q16": chain["semantic_q16"],
        "semantic_norm_q15": semantic_norm_q15,
        # The old (defective) group-mean L2 Q0.15 is retained only for audit; it
        # never enters ``score_q``.
        "old_group_mean_l2_norm_q15": chain["l2_norm_q15"],
        "corrected_patch_l2": corrected,
        "patch_q": corrected["patch_q"],
        "patch_scale": corrected["patch_scale"],
        "patch_norms": corrected["patch_norms"],
        "group_norm_sum": corrected["group_norm_sum"],
        "l2_norm_q15": l2_norm_q15,
        "score_q": score_q,
        "combined": combined,
        "integer_chain_bounds": chain["bounds"],
        "corrected_l2_bounds": corrected["bounds"],
        "bounds": {
            "semantic_norm_q15_max": (
                int(semantic_norm_q15.max()) if semantic_norm_q15.size else 0
            ),
            "l2_norm_q15_max": int(l2_norm_q15.max()) if l2_norm_q15.size else 0,
            "score_q_max": int(score_q.max()) if score_q.size else 0,
            "score_q_max_allowed": int(SCORE_Q_MAX),
            "lambda_semantic": float(LAMBDA_SEMANTIC),
        },
    }


# ---------------------------------------------------------------------------
# Scorer factory (three callbacks for the frozen B1 integer_scorer hook)
# ---------------------------------------------------------------------------
class CorrectedL2Scorer:
    """Callable B1 ``integer_scorer`` for one corrected-L2 arm at a time.

    The returned mapping contains ``combined_scores`` (consumed verbatim by the
    frozen :func:`evaluation.hardware_regular_token_selector.select_hardware_regular_groups`)
    plus every intermediate array needed to write the raw audit record.
    """

    def __init__(
        self,
        arm: str,
        *,
        Wt: Any | None = None,
        wv_q: Any | None = None,
        budgets: Sequence[int] = BUDGETS,
    ) -> None:
        if arm not in ALLOWED_ARMS:
            raise CorrectedL2Error(
                f"arm must be one of {ALLOWED_ARMS}, got {arm!r}"
            )
        self.arm = str(arm)
        self.Wt = None if Wt is None else np.asarray(Wt)
        self.wv_q = None if wv_q is None else np.asarray(wv_q, dtype=np.int64)
        if self.arm == ARM_PROMPT_INTEGER:
            if self.Wt is None or self.wv_q is None:
                raise CorrectedL2Error(
                    "prompt_plus_corrected_l2_integer requires Wt and wv_q"
                )
        elif self.Wt is not None or self.wv_q is not None:
            raise CorrectedL2Error(
                f"arm {self.arm!r} must not receive Wt/wv_q (no prompt term)"
            )
        self.budgets = tuple(int(value) for value in budgets)
        self.last_payload: dict[str, Any] | None = None

    def __call__(
        self,
        hidden_groups: Any,
        merged_h: int,
        merged_w: int,
        budget: int,
        text_embeddings: Any | None = None,
    ) -> dict[str, Any]:
        if self.arm == ARM_FLOAT:
            payload = corrected_patch_l2_float(hidden_groups)
        elif self.arm == ARM_INTEGER:
            payload = corrected_patch_l2_integer(hidden_groups)
        else:
            payload = prompt_plus_corrected_l2_integer(
                hidden_groups,
                text_embeddings,
                self.Wt,
                self.wv_q,
                merged_h,
                merged_w,
                budget=int(budget),
            )
        payload["arm"] = self.arm
        payload["budget"] = int(budget)
        payload["combined_scores"] = np.asarray(payload["combined"], dtype=np.float64)
        self.last_payload = payload
        return payload


def build_b1_scorer(
    arm: str,
    *,
    Wt: Any | None = None,
    wv_q: Any | None = None,
    budgets: Sequence[int] = BUDGETS,
) -> CorrectedL2Scorer:
    """Return the arm's B1 ``integer_scorer`` callback."""

    return CorrectedL2Scorer(arm, Wt=Wt, wv_q=wv_q, budgets=budgets)


#: Convenience wrappers for the two stateless arms.
SCORER_CALLBACKS: Mapping[str, Any] = {
    ARM_FLOAT: score_corrected_l2_float,
    ARM_INTEGER: score_corrected_l2_integer,
}


__all__ = [
    "ACTIVATION_MAX_ABS",
    "ALLOWED_ARMS",
    "ARM_FLOAT",
    "ARM_INTEGER",
    "ARM_PROMPT_INTEGER",
    "BUDGETS",
    "CorrectedL2Error",
    "CorrectedL2Scorer",
    "EMBED_DIM",
    "GROUP_PATCH_NORM_SUM_BOUND",
    "LAMBDA_SEMANTIC",
    "PATCH_NORM_BOUND",
    "PATCH_SQUARED_SUM_BOUND",
    "SCORE_Q_MAX",
    "SCORE_Q_SCALE",
    "SCORER_CALLBACKS",
    "array_sha256",
    "build_b1_scorer",
    "combine_q15",
    "corrected_patch_l2_float",
    "corrected_patch_l2_integer",
    "prompt_plus_corrected_l2_integer",
    "score_corrected_l2_float",
    "score_corrected_l2_integer",
]
