#!/usr/bin/env python3
"""Frozen adaptive hardware-rule budget formula for the layer-5 real-only arm.

The layer-5 adaptive-budget screening100 experiment keeps the frozen unbounded
real-only L2 + 3x3 anchor *selection algorithm* exactly as implemented in
:mod:`evaluation.hardware_regular_layer5_unbounded_l2`; only the per-sample
token budget becomes adaptive.  The budget is derived from the **real** number
of post-processor LLM visual tokens ``G = full_llm_visual_tokens`` with the
preregistered formula:

.. code-block:: text

    K_raw = max(96, ceil(0.25 * G))
    K_eff = min(G, 16 * ceil(K_raw / 16))
    identity bypass if K_eff >= G

Because ``K_eff`` is capped at ``G`` the identity branch is exactly the
``K_eff == G`` case (``G <= 96`` or an already-aligned small ``G``).

This module is deliberately tiny, dependency-free and CPU-only so that the GPU
runner, the CPU summariser and the tests all share one single source of truth
for the formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Minimum raw budget (the frozen screening budget floor).
MIN_BUDGET = 96
#: Target retention fraction as an exact rational 1/4.
RETENTION_NUMERATOR = 1
RETENTION_DENOMINATOR = 4
#: Hardware alignment in tokens (multiples of 16).
BUDGET_ALIGNMENT = 16

#: Human-readable, frozen formula identifier recorded in every artifact.
BUDGET_FORMULA = (
    "G = full_llm_visual_tokens; "
    "K_raw = max(96, ceil(0.25 * G)); "
    "K_eff = min(G, 16 * ceil(K_raw / 16)); "
    "identity_bypass if K_eff >= G"
)


def _ceil_div(numerator: int, denominator: int) -> int:
    """Integer ceiling division for positive operands."""

    return -(-numerator // denominator)


def compute_raw_budget(full_llm_visual_tokens: int) -> int:
    """``K_raw = max(96, ceil(0.25 * G))``."""

    groups = _require_group_count(full_llm_visual_tokens)
    return max(
        MIN_BUDGET,
        _ceil_div(groups * RETENTION_NUMERATOR, RETENTION_DENOMINATOR),
    )


def align_budget(budget: int) -> int:
    """Round ``budget`` up to the next multiple of :data:`BUDGET_ALIGNMENT`."""

    return BUDGET_ALIGNMENT * _ceil_div(budget, BUDGET_ALIGNMENT)


def _require_group_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"full_llm_visual_tokens must be a positive int, got {value!r}"
        )
    if value <= 0:
        raise ValueError(
            f"full_llm_visual_tokens must be positive, got {value}"
        )
    return value


@dataclass(frozen=True)
class AdaptiveBudget:
    """Resolved per-sample adaptive budget for one image."""

    full_llm_visual_tokens: int
    raw_budget: int
    effective_budget: int
    identity_bypass: bool
    retention: float
    budget_formula: str = BUDGET_FORMULA
    budget_min: int = MIN_BUDGET
    budget_retention_ratio: float = RETENTION_NUMERATOR / RETENTION_DENOMINATOR
    budget_alignment: int = BUDGET_ALIGNMENT

    def as_record(self) -> dict[str, Any]:
        """Flat per-sample fields persisted in the raw JSONL."""

        return {
            "full_llm_visual_tokens": self.full_llm_visual_tokens,
            "raw_budget": self.raw_budget,
            "effective_budget": self.effective_budget,
            "requested_budget": self.effective_budget,
            "adaptive_retention": self.retention,
            "identity_bypass": self.identity_bypass,
            "budget_formula": self.budget_formula,
            "budget_min": self.budget_min,
            "budget_retention_ratio": self.budget_retention_ratio,
            "budget_alignment": self.budget_alignment,
        }


def compute_adaptive_budget(full_llm_visual_tokens: int) -> AdaptiveBudget:
    """Resolve ``(K_raw, K_eff, retention, identity)`` from the real ``G``."""

    groups = _require_group_count(full_llm_visual_tokens)
    raw_budget = max(
        MIN_BUDGET,
        _ceil_div(groups * RETENTION_NUMERATOR, RETENTION_DENOMINATOR),
    )
    aligned = align_budget(raw_budget)
    effective_budget = min(groups, aligned)
    identity_bypass = effective_budget >= groups
    return AdaptiveBudget(
        full_llm_visual_tokens=groups,
        raw_budget=raw_budget,
        effective_budget=effective_budget,
        identity_bypass=identity_bypass,
        retention=effective_budget / groups,
    )


def verify_adaptive_budget_fields(row: Any) -> AdaptiveBudget:
    """Recompute the formula from ``G`` and prove every recorded field matches.

    Raises :class:`ValueError` on any mismatch so callers can fail closed.
    """

    if "full_llm_visual_tokens" not in row:
        raise ValueError("record is missing full_llm_visual_tokens")
    groups = row["full_llm_visual_tokens"]
    if isinstance(groups, bool) or not isinstance(groups, int):
        try:
            groups = int(groups)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"full_llm_visual_tokens is not an int: {row.get('full_llm_visual_tokens')!r}"
            ) from exc
    expected = compute_adaptive_budget(groups)

    def _require(field: str, value: Any) -> None:
        if field not in row:
            raise ValueError(f"record is missing {field}")

    for field in (
        "raw_budget",
        "effective_budget",
        "requested_budget",
        "adaptive_retention",
        "identity_bypass",
    ):
        _require(field, row.get(field))

    if int(row["raw_budget"]) != expected.raw_budget:
        raise ValueError(
            f"raw_budget {row['raw_budget']} != formula {expected.raw_budget} "
            f"for G={groups}"
        )
    if int(row["effective_budget"]) != expected.effective_budget:
        raise ValueError(
            f"effective_budget {row['effective_budget']} != formula "
            f"{expected.effective_budget} for G={groups}"
        )
    if int(row["requested_budget"]) != expected.effective_budget:
        raise ValueError(
            f"requested_budget {row['requested_budget']} != K_eff "
            f"{expected.effective_budget} for G={groups}"
        )
    if bool(row["identity_bypass"]) != expected.identity_bypass:
        raise ValueError(
            f"identity_bypass {row['identity_bypass']} != formula "
            f"{expected.identity_bypass} for G={groups}"
        )
    recorded_retention = float(row["adaptive_retention"])
    if abs(recorded_retention - expected.retention) > 1e-12:
        raise ValueError(
            f"adaptive_retention {recorded_retention} != formula "
            f"{expected.retention} for G={groups}"
        )
    return expected


__all__ = [
    "AdaptiveBudget",
    "BUDGET_ALIGNMENT",
    "BUDGET_FORMULA",
    "MIN_BUDGET",
    "RETENTION_DENOMINATOR",
    "RETENTION_NUMERATOR",
    "align_budget",
    "compute_adaptive_budget",
    "compute_raw_budget",
    "verify_adaptive_budget_fields",
]
