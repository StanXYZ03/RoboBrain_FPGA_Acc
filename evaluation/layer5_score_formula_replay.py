#!/usr/bin/env python3
"""Layer-5 score-formula replay: does the aggregation change explain r0250?

Question
--------
On the frozen CV-Bench ``screening100`` set, can the historical ``shallow_r0250``
index sets be reproduced by changing **only** the per-group score from the
current control formula

    control:   norm(mean(x_0, x_1, x_2, x_3))

to the historical formula

    candidate: mean(norm(x_0), norm(x_1), norm(x_2), norm(x_3))

Everything else is held fixed: one highest-score anchor in each non-empty 3x3
region, then stable global descending score with the lower group index as
tie-break, the exact per-sample ``K_C`` taken from the frozen reference, and
ascending output indices.  For every frozen reference sample ``K_C >= 16`` so
both paths use a 3x3 spatial anchor grid.

Scope
-----
This is a *diagnostic* module.  It is CPU/import safe (no CUDA call at import
time), never loads weights and never decodes text.  The vision-only replay path
reuses the frozen ``_unified_forward`` from
:mod:`evaluation.hardware_regular_layer5_forced_index_ablation` unchanged; it
does **not** call the historical end-to-end selector as the candidate
implementation.

The historical end-to-end selector (``structured_group_indices`` +
``mean-of-patch-L2``) is used *only in tests* as the equivalence oracle for the
candidate selector, never as the runtime candidate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from evaluation.hardware_regular_layer5_forced_index_ablation import (
    PRUNE_AFTER_LAYER,
    HardwareLayer5ForcedIndexAblationOutput,
    _make_selection,
    _unified_forward,
    select_control_groups,
    validate_diagnostic_budget,
    verify_diagnostic_selection,
)
from evaluation.hardware_regular_token_selector import (
    BOUNDARY_MAIN,
    REGION_GRID,
    HardwareSelection,
    InsufficientBudgetError,
    SelectorContractError,
    build_regions,
    minmax_normalize,
    scores_from_precomputed,
    validate_hidden,
)

ROOT = Path(__file__).resolve().parents[1]

#: Candidate mode label written into every record.  It is deliberately distinct
#: from the frozen control/forced mode labels.
MODE_CANDIDATE = "score_formula_candidate"
SELECTION_RULE_CANDIDATE = "historical_mean_patch_l2_plus_3x3_anchor"
SELECTION_RULE_CONTROL = "control_global_l2_plus_3x3_anchor"
ARM_LABEL = "layer5_score_formula_replay"
REFERENCE_SELECTOR = "shallow_r0250"

#: Frozen score-formula spellings (they appear verbatim in every record).
FORMULA_CONTROL = "norm(mean(x_0, x_1, x_2, x_3))"
FORMULA_CANDIDATE = "mean(norm(x_0), norm(x_1), norm(x_2), norm(x_3))"

#: The reference contract is only defined for a 3x3 anchor grid.
MIN_REFERENCE_K = 16
EXPECTED_SAMPLES = 100
CUTPOINT_PREFIX_BLOCKS = tuple(range(0, PRUNE_AFTER_LAYER + 1))

#: Anchor grid fixed by the reference contract.
ANCHOR_GRID = REGION_GRID

#: Diagnostic, non-deployable scope flags.
CAUSAL_DIAGNOSTIC_ONLY = True
FPGA_DEPLOYABLE_CANDIDATE = False
HARDWARE_ALIGNMENT_RELAXED_FOR_DIAGNOSTIC = True
ONLY_VARIABLE = "per_group_score_aggregation"

DEFAULT_OUTPUT_DIR = (
    ROOT / "artifacts" / "token_selector" / "2026-10-04-layer5-score-formula-replay"
)
DEFAULT_REPLAY_JSONL = DEFAULT_OUTPUT_DIR / "layer5_score_formula_replay_screening100.jsonl"
DEFAULT_REPORT_DIR = ROOT / "reports" / "cv_bench"
REPORT_STEM = "2026-10-04-layer5-score-formula-replay-screening100"


class ReplayContractError(RuntimeError):
    """Base class for fail-closed replay contract violations."""


class InputContractError(ReplayContractError):
    """A manifest / reference / control / replay input violates the contract."""


class UnexpectedIdentityBypassError(ReplayContractError):
    """The frozen reference never bypasses, so an identity bypass is unexpected."""


# ---------------------------------------------------------------------------
# Pure score formulas
# ---------------------------------------------------------------------------


def control_group_scores(hidden: Any) -> np.ndarray:
    """Control formula ``norm(mean(x_0, x_1, x_2, x_3))`` per group.

    This is the frozen ``group_l2_saliency`` aggregation, reproduced here as a
    pure numpy helper so the two formulas can be compared side by side.
    """
    array = validate_hidden(hidden)
    return np.linalg.norm(array.mean(axis=1), axis=1)


def historical_group_scores(hidden: Any) -> np.ndarray:
    """Historical formula ``mean(norm(x_0), norm(x_1), norm(x_2), norm(x_3))``.

    This is exactly the historical ``l2_anchor`` score
    ``hidden.view(G, 4, -1).norm(dim=-1).mean(dim=1)``.
    """
    array = validate_hidden(hidden)
    return np.linalg.norm(array, axis=2).mean(axis=1)


# ---------------------------------------------------------------------------
# Fixed 3x3 anchor + global-fill policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorFillSelection:
    """Auditable output of the shared anchor/global-fill policy."""

    anchors: tuple[int, ...]
    details: tuple[int, ...]
    selected: tuple[int, ...]


def _validated_scores(scores: Any) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise SelectorContractError("scores must not be empty")
    if not np.isfinite(array).all():
        raise SelectorContractError("scores must contain only finite values")
    return array


def anchor_fill_selection(
    scores: Any,
    grid_h: int,
    grid_w: int,
    *,
    budget: int,
    region_grid: int = REGION_GRID,
) -> AnchorFillSelection:
    """One highest-score anchor per non-empty region, then global descending fill.

    Ties always resolve to the lower original group index (the historical stable
    descending ``argsort``).  With ``region_grid=3`` and ``budget >= 16`` this is
    element-for-element the historical ``structured_group_indices`` policy.
    """
    score = _validated_scores(scores)
    group_count = int(score.size)
    grid_h = int(grid_h)
    grid_w = int(grid_w)
    if grid_h <= 0 or grid_w <= 0 or grid_h * grid_w != group_count:
        raise SelectorContractError(
            f"grid {grid_h}x{grid_w} does not match {group_count} scored groups"
        )
    budget = validate_diagnostic_budget(budget)
    if budget > group_count:
        raise InsufficientBudgetError(
            f"budget {budget} exceeds the {group_count} available groups"
        )
    regions = [
        region for region in build_regions(grid_h, grid_w, region_grid) if region.members
    ]
    if budget < len(regions):
        raise InsufficientBudgetError(
            f"budget {budget} cannot cover one anchor in each of {len(regions)} regions"
        )
    anchors = tuple(
        int(max(region.members, key=lambda group: (score[group], -group)))
        for region in regions
    )
    anchor_set = set(anchors)
    needed = budget - len(anchors)
    ranked = sorted(
        (group for group in range(group_count) if group not in anchor_set),
        key=lambda group: (-score[group], group),
    )
    if len(ranked) < needed:
        raise InsufficientBudgetError(
            f"global ranking has {len(ranked)} non-anchor groups but {needed} "
            f"detail groups are required"
        )
    details = tuple(int(group) for group in ranked[:needed])
    selected = tuple(sorted(anchors + details))
    if len(selected) != budget:
        raise SelectorContractError(
            f"internal error: selected {len(selected)} groups for budget {budget}"
        )
    return AnchorFillSelection(anchors=anchors, details=details, selected=selected)


def select_groups_by_score(
    scores: Any,
    grid_h: int,
    grid_w: int,
    *,
    budget: int,
    region_grid: int = REGION_GRID,
) -> np.ndarray:
    """Return the ascending index array of :func:`anchor_fill_selection`."""
    return np.asarray(
        anchor_fill_selection(
            scores, grid_h, grid_w, budget=budget, region_grid=region_grid
        ).selected,
        dtype=np.int64,
    )


def select_candidate_groups(
    hidden: Any,
    grid_h: int,
    grid_w: int,
    *,
    budget: int,
    region_grid: int = REGION_GRID,
    boundary: str = BOUNDARY_MAIN,
) -> HardwareSelection:
    """Candidate selector: historical mean-of-patch-L2 + fixed 3x3 anchor policy."""
    hidden_array = validate_hidden(hidden)
    scores = historical_group_scores(hidden_array)
    parts = anchor_fill_selection(
        scores, grid_h, grid_w, budget=budget, region_grid=region_grid
    )
    combined = minmax_normalize(scores)
    return _make_selection(
        hidden_array,
        grid_h,
        grid_w,
        np.asarray(parts.selected, dtype=np.int64),
        budget=budget,
        bypass=False,
        boundary=boundary,
        region_grid=region_grid,
        scores=scores_from_precomputed(l2=scores, combined=combined),
        anchors=parts.anchors,
        details=parts.details,
    )


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def indices_sha256(indices: Sequence[int]) -> str:
    """Canonical index hash: comma-joined decimal indices, SHA-256, UTF-8."""
    payload = ",".join(str(int(index)) for index in indices).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_index_match(left: Sequence[int], right: Sequence[int]) -> bool:
    return [int(value) for value in left] == [int(value) for value in right]


def jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    """Jaccard similarity of two index sets (empty vs empty is 1.0)."""
    left_set = {int(value) for value in left}
    right_set = {int(value) for value in right}
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def score_summary(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = bool(np.isfinite(array).all())
    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "all_finite": finite,
        }
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "all_finite": finite,
    }


def normalized_indices(raw: Any, *, group_count: int, label: str) -> list[int]:
    """Validate a 1-D unique in-range integer index list; return it ascending."""
    if not isinstance(raw, (list, tuple)) or not raw:
        raise InputContractError(f"{label} must be a non-empty list")
    values: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int):
            raise InputContractError(f"{label} must contain integers, got {value!r}")
        values.append(int(value))
    if len(set(values)) != len(values):
        raise InputContractError(f"{label} must be unique")
    ordered = sorted(values)
    if ordered[0] < 0 or ordered[-1] >= int(group_count):
        raise InputContractError(
            f"{label} must lie in [0, {int(group_count)}), got "
            f"min={ordered[0]} max={ordered[-1]}"
        )
    return ordered


# ---------------------------------------------------------------------------
# Fail-closed manifest / reference / control loading
# ---------------------------------------------------------------------------


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise InputContractError(f"missing required {label}: {path} (fail closed)")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputContractError(
                f"{label} line {line_number} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise InputContractError(f"{label} line {line_number} must be an object")
        rows.append(payload)
    if not rows:
        raise InputContractError(f"{label} is empty: {path} (fail closed)")
    return rows


def index_rows_by_sample_id(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for position, row in enumerate(rows):
        if "sample_id" not in row:
            raise InputContractError(f"{label} row {position} is missing sample_id")
        sample_id = str(row["sample_id"])
        if sample_id in index:
            raise InputContractError(f"duplicate {label} sample_id {sample_id!r}")
        index[sample_id] = row
    return index


def _require_same_ids(
    expected: Mapping[str, Any], observed: Mapping[str, Any], label: str
) -> None:
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing:
        raise InputContractError(
            f"{label} is missing {len(missing)} sample(s): {missing[:10]} (fail closed)"
        )
    if extra:
        raise InputContractError(
            f"{label} has {len(extra)} extra sample(s): {extra[:10]} (fail closed)"
        )


def normalize_reference_record(
    row: Mapping[str, Any], manifest_row: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and normalize one frozen historical ``shallow_r0250`` record."""
    sample_id = str(row.get("sample_id"))
    label = f"reference/{sample_id}"
    if str(row.get("task")) != str(manifest_row.get("task")):
        raise InputContractError(
            f"{label} task {row.get('task')!r} != manifest task "
            f"{manifest_row.get('task')!r}"
        )
    prune_after_layer = row.get("prune_after_layer")
    if (
        isinstance(prune_after_layer, bool)
        or not isinstance(prune_after_layer, int)
        or int(prune_after_layer) != PRUNE_AFTER_LAYER
    ):
        raise InputContractError(
            f"{label} prune_after_layer must be {PRUNE_AFTER_LAYER}, "
            f"got {prune_after_layer!r}"
        )
    anchor_grid = row.get("anchor_grid")
    if anchor_grid is not None and int(anchor_grid) != REGION_GRID:
        raise InputContractError(
            f"{label} anchor_grid must be {REGION_GRID}, got {anchor_grid!r}"
        )
    selector = row.get("selector")
    if selector is not None and str(selector) != "l2_anchor":
        raise InputContractError(
            f"{label} selector must be 'l2_anchor', got {selector!r}"
        )
    group_count = row.get("baseline_llm_visual_tokens")
    if isinstance(group_count, bool) or not isinstance(group_count, int):
        raise InputContractError(
            f"{label} baseline_llm_visual_tokens must be an int, got {group_count!r}"
        )
    group_count = int(group_count)
    if group_count <= 0:
        raise InputContractError(f"{label} G must be positive")
    indices = normalized_indices(
        row.get("selected_group_indices"),
        group_count=group_count,
        label=f"{label}.selected_group_indices",
    )
    recomputed = indices_sha256(indices)
    if row.get("selected_group_indices_sha256") != recomputed:
        raise InputContractError(
            f"{label}.selected_group_indices_sha256 mismatch: "
            f"recorded={row.get('selected_group_indices_sha256')!r} "
            f"recomputed={recomputed}"
        )
    k_c = len(indices)
    recorded_k = row.get("llm_visual_tokens")
    if isinstance(recorded_k, bool) or not isinstance(recorded_k, int) or int(
        recorded_k
    ) != k_c:
        raise InputContractError(
            f"{label}.llm_visual_tokens={recorded_k!r} != K_C={k_c}"
        )
    if k_c >= group_count:
        raise UnexpectedIdentityBypassError(
            f"{label} has K_C={k_c} >= G={group_count}: the frozen reference never "
            f"bypasses (fail closed)"
        )
    if k_c < MIN_REFERENCE_K:
        raise InputContractError(
            f"{label} K_C={k_c} < {MIN_REFERENCE_K}: the 3x3 anchor equivalence "
            f"contract requires K >= {MIN_REFERENCE_K} (fail closed)"
        )
    return {
        "sample_id": sample_id,
        "task": str(manifest_row.get("task")),
        "G": group_count,
        "K_C": k_c,
        "group_indices": indices,
        "group_indices_sha256": recomputed,
        "llm_visual_tokens": int(recorded_k),
        "image_grid_thw": [int(value) for value in row.get("image_grid_thw", [])],
    }


def normalize_control_record(
    row: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    reference_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Validate and normalize one frozen control-arm record."""
    sample_id = str(row.get("sample_id"))
    label = f"control/{sample_id}"
    if str(row.get("task")) != str(manifest_row.get("task")):
        raise InputContractError(
            f"{label} task {row.get('task')!r} != manifest task "
            f"{manifest_row.get('task')!r}"
        )
    if str(row.get("arm")) != "control" or str(row.get("mode")) != "control":
        raise InputContractError(
            f"{label} arm/mode must be control/control, got "
            f"{row.get('arm')!r}/{row.get('mode')!r}"
        )
    if row.get("selection_rule") != SELECTION_RULE_CONTROL:
        raise InputContractError(
            f"{label} selection_rule {row.get('selection_rule')!r} != "
            f"{SELECTION_RULE_CONTROL!r}"
        )
    if int(row.get("prune_after_layer", -1)) != PRUNE_AFTER_LAYER:
        raise InputContractError(f"{label} prune_after_layer must be {PRUNE_AFTER_LAYER}")
    if int(row.get("region_grid", -1)) != REGION_GRID:
        raise InputContractError(f"{label} region_grid must be {REGION_GRID}")
    if row.get("reference_jsonl_sha256") != reference_sha256:
        raise InputContractError(
            f"{label} reference_jsonl_sha256 does not match the current reference"
        )
    if row.get("manifest_sha256") != manifest_sha256:
        raise InputContractError(
            f"{label} manifest_sha256 does not match the current manifest"
        )
    group_count = int(reference["G"])
    k_c = int(reference["K_C"])
    if row.get("reference_G") != group_count:
        raise InputContractError(
            f"{label} reference_G={row.get('reference_G')!r} != reference G={group_count}"
        )
    if row.get("K_C") != k_c or row.get("reference_K_C") != k_c:
        raise InputContractError(
            f"{label} K_C/reference_K_C must equal reference K_C={k_c}"
        )
    indices = normalized_indices(
        row.get("selected_group_indices"),
        group_count=group_count,
        label=f"{label}.selected_group_indices",
    )
    recomputed = indices_sha256(indices)
    if row.get("selected_group_indices_sha256") != recomputed:
        raise InputContractError(
            f"{label}.selected_group_indices_sha256 mismatch: "
            f"recorded={row.get('selected_group_indices_sha256')!r} "
            f"recomputed={recomputed}"
        )
    if len(indices) != k_c:
        raise InputContractError(
            f"{label} selected {len(indices)} groups but K_C={k_c}"
        )
    if int(row.get("context_group_count", -1)) != 0:
        raise InputContractError(f"{label} context_group_count must be 0")
    if int(row.get("real_group_count", -1)) != k_c or int(
        row.get("output_group_count", -1)
    ) != k_c:
        raise InputContractError(
            f"{label} real/output group count must equal K_C={k_c}"
        )
    if row.get("identity_bypass") is not False:
        raise InputContractError(f"{label} identity_bypass must be False")
    if row.get("source_types_all_real") is not True:
        raise InputContractError(f"{label} source_types_all_real must be True")
    reference_indices = normalized_indices(
        row.get("reference_selected_group_indices"),
        group_count=group_count,
        label=f"{label}.reference_selected_group_indices",
    )
    if reference_indices != list(reference["group_indices"]):
        raise InputContractError(
            f"{label} embedded reference indices do not match the frozen reference"
        )
    return {
        "sample_id": sample_id,
        "task": str(manifest_row.get("task")),
        "G": group_count,
        "K_C": k_c,
        "group_indices": indices,
        "group_indices_sha256": recomputed,
    }


@dataclass(frozen=True)
class ReplayInputs:
    """Validated, fail-closed bundle of the three frozen inputs."""

    manifest_rows: tuple[dict[str, Any], ...]
    manifest_sha256: str
    reference_map: dict[str, dict[str, Any]]
    reference_sha256: str
    control_map: dict[str, dict[str, Any]]
    control_sha256: str

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(str(row["sample_id"]) for row in self.manifest_rows)


def load_replay_inputs(
    manifest_path: Path,
    reference_path: Path,
    control_path: Path,
    *,
    expected_samples: int | None = EXPECTED_SAMPLES,
) -> ReplayInputs:
    """Load and cross-validate manifest, reference and control JSONL fail closed."""
    manifest_sha256 = file_sha256(manifest_path)
    reference_sha256 = file_sha256(reference_path)
    control_sha256 = file_sha256(control_path)
    manifest_rows = read_jsonl(manifest_path, "manifest")
    manifest_index = index_rows_by_sample_id(manifest_rows, "manifest")
    if expected_samples is not None and len(manifest_index) != expected_samples:
        raise InputContractError(
            f"manifest has {len(manifest_index)} unique sample_ids; "
            f"expected {expected_samples}"
        )

    reference_rows = read_jsonl(reference_path, "reference")
    reference_index = index_rows_by_sample_id(reference_rows, "reference")
    _require_same_ids(manifest_index, reference_index, "reference")
    reference_map = {
        sample_id: normalize_reference_record(
            reference_index[sample_id], manifest_index[sample_id]
        )
        for sample_id in manifest_index
    }

    control_rows = read_jsonl(control_path, "control")
    control_index = index_rows_by_sample_id(control_rows, "control")
    _require_same_ids(manifest_index, control_index, "control")
    control_map = {
        sample_id: normalize_control_record(
            control_index[sample_id],
            manifest_index[sample_id],
            reference_map[sample_id],
            reference_sha256=reference_sha256,
            manifest_sha256=manifest_sha256,
        )
        for sample_id in manifest_index
    }

    return ReplayInputs(
        manifest_rows=tuple(manifest_rows),
        manifest_sha256=manifest_sha256,
        reference_map=reference_map,
        reference_sha256=reference_sha256,
        control_map=control_map,
        control_sha256=control_sha256,
    )


# ---------------------------------------------------------------------------
# Replay-record validation and safe resume
# ---------------------------------------------------------------------------


def _expected_provenance(provenance: Mapping[str, Any]) -> dict[str, str]:
    return {
        "manifest_sha256": str(provenance["manifest_sha256"]),
        "reference_jsonl_sha256": str(provenance["reference_jsonl_sha256"]),
        "control_jsonl_sha256": str(provenance["control_jsonl_sha256"]),
    }


def validate_replay_record(
    record: Mapping[str, Any],
    reference: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
) -> None:
    """Fail-closed per-record validation of a runner replay record."""
    sample_id = str(record.get("sample_id"))
    label = f"replay/{sample_id}"
    if sample_id != str(reference["sample_id"]) or sample_id != str(control["sample_id"]):
        raise InputContractError(f"{label} sample_id mismatch against inputs")
    if str(record.get("task")) != str(reference["task"]):
        raise InputContractError(
            f"{label} task {record.get('task')!r} != reference task {reference['task']!r}"
        )
    for field, expected in _expected_provenance(provenance).items():
        if record.get(field) != expected:
            raise InputContractError(
                f"{label} {field}={record.get(field)!r} != {expected!r} (fail closed)"
            )
    for flag, expected in (
        ("causal_diagnostic_only", CAUSAL_DIAGNOSTIC_ONLY),
        ("fpga_deployable_candidate", FPGA_DEPLOYABLE_CANDIDATE),
        ("only_variable", ONLY_VARIABLE),
        ("identity_bypass", False),
        ("uses_context_pooling", False),
        ("uses_context_slot_reservation", False),
        ("uses_local_candidate_cap", False),
        ("uses_b1_trained_wv", False),
        ("source_types_all_real", True),
    ):
        if record.get(flag) != expected:
            raise InputContractError(
                f"{label} {flag}={record.get(flag)!r} must be {expected!r}"
            )
    group_count = int(reference["G"])
    k_c = int(reference["K_C"])
    if record.get("reference_G") != group_count:
        raise InputContractError(f"{label} reference_G != {group_count}")
    if record.get("K_C") != k_c:
        raise InputContractError(f"{label} K_C != {k_c}")
    merged_grid = record.get("merged_grid")
    if (
        not isinstance(merged_grid, (list, tuple))
        or len(merged_grid) != 2
        or int(merged_grid[0]) * int(merged_grid[1]) != group_count
    ):
        raise InputContractError(
            f"{label} merged_grid {merged_grid!r} does not multiply to G={group_count}"
        )

    reference_indices = normalized_indices(
        record.get("reference_selected_group_indices"),
        group_count=group_count,
        label=f"{label}.reference_selected_group_indices",
    )
    if reference_indices != list(reference["group_indices"]):
        raise InputContractError(f"{label} reference indices do not match reference")
    candidate_indices = normalized_indices(
        record.get("candidate_selected_group_indices"),
        group_count=group_count,
        label=f"{label}.candidate_selected_group_indices",
    )
    control_indices = normalized_indices(
        record.get("control_selected_group_indices"),
        group_count=group_count,
        label=f"{label}.control_selected_group_indices",
    )
    if list(control["group_indices"]) != control_indices:
        raise InputContractError(
            f"{label} recomputed control indices do not match the frozen control arm"
        )
    if len(candidate_indices) != k_c or len(control_indices) != k_c:
        raise InputContractError(f"{label} candidate/control must emit exactly K_C={k_c}")
    if record.get("candidate_selected_group_indices_sha256") != indices_sha256(
        candidate_indices
    ):
        raise InputContractError(f"{label} candidate index hash mismatch")
    if record.get("control_selected_group_indices_sha256") != indices_sha256(
        control_indices
    ):
        raise InputContractError(f"{label} control index hash mismatch")
    if record.get("reference_selected_group_indices_sha256") != reference[
        "group_indices_sha256"
    ]:
        raise InputContractError(f"{label} reference index hash mismatch")

    exact = exact_index_match(candidate_indices, reference_indices)
    if record.get("candidate_equals_reference") is not exact:
        raise InputContractError(
            f"{label} candidate_equals_reference flag is inconsistent with the indices"
        )
    for field, left, right in (
        ("candidate_vs_reference_jaccard", candidate_indices, reference_indices),
        ("control_vs_reference_jaccard", control_indices, reference_indices),
    ):
        expected = jaccard(left, right)
        observed = record.get(field)
        if not isinstance(observed, (int, float)) or abs(float(observed) - expected) > 1e-9:
            raise InputContractError(
                f"{label} {field}={observed!r} != recomputed {expected:.12f}"
            )
    for field in ("candidate_score_summary", "control_score_summary"):
        summary = record.get(field)
        if not isinstance(summary, dict) or summary.get("all_finite") is not True:
            raise InputContractError(f"{label} {field} must be finite")


def resume_completed_ids(
    path: Path,
    allowed_ids: Sequence[str],
    *,
    inputs: ReplayInputs,
    provenance: Mapping[str, Any],
) -> set[str]:
    """Validate an existing replay JSONL and return its completed sample ids.

    Missing/empty file yields the empty set.  Otherwise every line must be a
    JSON object with a unique in-range sample_id, exact current provenance, the
    frozen diagnostic flags and internally consistent index hashes/jaccards.
    Any violation aborts: resume never silently drops or mixes work.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    allowed = {str(value) for value in allowed_ids}
    done: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputContractError(
                f"replay line {line_number} is not valid JSON: {exc} (fail closed)"
            ) from exc
        if not isinstance(record, dict):
            raise InputContractError(
                f"replay line {line_number} must be a JSON object (fail closed)"
            )
        if "sample_id" not in record:
            raise InputContractError(
                f"replay line {line_number} is missing sample_id (fail closed)"
            )
        sample_id = str(record["sample_id"])
        if sample_id in done:
            raise InputContractError(f"duplicate replay sample_id {sample_id!r}")
        if sample_id not in allowed:
            raise InputContractError(
                f"replay sample_id {sample_id!r} is not in the current sample set"
            )
        if sample_id not in inputs.reference_map or sample_id not in inputs.control_map:
            raise InputContractError(
                f"replay sample_id {sample_id!r} has no matching frozen reference/control"
            )
        validate_replay_record(
            record,
            inputs.reference_map[sample_id],
            inputs.control_map[sample_id],
            provenance=provenance,
        )
        done.add(sample_id)
    return done


# ---------------------------------------------------------------------------
# Vision-only unified replay path (reuses the frozen ``_unified_forward``)
# ---------------------------------------------------------------------------


@dataclass
class ScoreFormulaReplayOutput:
    """Both selections and both score vectors from one shared hidden state."""

    forward: HardwareLayer5ForcedIndexAblationOutput
    candidate_selection: HardwareSelection
    control_selection: HardwareSelection
    candidate_scores: np.ndarray
    control_scores: np.ndarray


def forward_layer5_score_formula_replay_vision(
    visual: Any,
    pixel_values: Any,
    grid_thw: Any,
    *,
    budget: int,
    region_grid: int = REGION_GRID,
    boundary: str = BOUNDARY_MAIN,
    prune_after_layer: int = PRUNE_AFTER_LAYER,
    synchronize: Callable[[], None] | None = None,
) -> ScoreFormulaReplayOutput:
    """Run the candidate selection through the frozen unified layer-5 vision path.

    The frozen ``_unified_forward`` is reused unchanged.  The single
    ``resolve_selection`` closure computes the candidate and control selections
    from the *same* hidden state and returns the candidate, so the only variable
    is the per-group score aggregation.  No LLM decode happens here.
    """
    budget = validate_diagnostic_budget(budget)
    captured: dict[str, Any] = {}

    def resolve_selection(hidden_np: np.ndarray, grid_h: int, grid_w: int) -> HardwareSelection:
        hidden = np.asarray(hidden_np, dtype=np.float64)
        candidate_scores = historical_group_scores(hidden)
        control_scores = control_group_scores(hidden)
        candidate = select_candidate_groups(
            hidden,
            grid_h,
            grid_w,
            budget=budget,
            region_grid=region_grid,
            boundary=boundary,
        )
        control = select_control_groups(
            hidden,
            grid_h,
            grid_w,
            budget=budget,
            region_grid=region_grid,
            boundary=boundary,
        )
        if candidate.bypass or control.bypass:
            raise UnexpectedIdentityBypassError(
                "the frozen reference never bypasses, but this sample selected "
                "the all-real identity path (fail closed)"
            )
        captured["candidate"] = candidate
        captured["control"] = control
        captured["candidate_scores"] = candidate_scores
        captured["control_scores"] = control_scores
        return candidate

    forward = _unified_forward(
        visual,
        pixel_values,
        grid_thw,
        resolve_selection=resolve_selection,
        budget=budget,
        mode=MODE_CANDIDATE,
        prune_after_layer=prune_after_layer,
        synchronize=synchronize,
    )
    for key in ("candidate", "control", "candidate_scores", "control_scores"):
        if key not in captured:  # pragma: no cover - _unified_forward always resolves
            raise ReplayContractError(
                f"internal error: the forward did not resolve a '{key}' selection"
            )
    verify_diagnostic_selection(captured["candidate"])
    verify_diagnostic_selection(captured["control"])
    return ScoreFormulaReplayOutput(
        forward=forward,
        candidate_selection=captured["candidate"],
        control_selection=captured["control"],
        candidate_scores=captured["candidate_scores"],
        control_scores=captured["control_scores"],
    )


def _as_int_list(values: Any) -> list[int]:
    """Flatten a numpy array / (possibly CUDA) tensor / sequence to ints."""
    if hasattr(values, "detach"):  # torch tensor, possibly on GPU
        values = values.detach().to("cpu")
        return [int(value) for value in values.reshape(-1).tolist()]
    return [int(value) for value in np.asarray(values).reshape(-1).tolist()]


def merged_grid_from_thw(
    grid_thw: Any, *, merge_size: int = 2
) -> tuple[int, int]:
    """Merged group grid ``(h, w)`` from a one-image ``image_grid_thw``."""
    values = _as_int_list(grid_thw)
    if len(values) != 3 or values[0] != 1:
        raise InputContractError(
            f"the score-formula replay supports one still image, got {values!r}"
        )
    if values[1] % merge_size or values[2] % merge_size:
        raise InputContractError("processor grid is not aligned to the visual merger")
    return values[1] // merge_size, values[2] // merge_size


def diagnostic_flags() -> dict[str, Any]:
    return {
        "causal_diagnostic_only": CAUSAL_DIAGNOSTIC_ONLY,
        "fpga_deployable_candidate": FPGA_DEPLOYABLE_CANDIDATE,
        "hardware_alignment_relaxed_for_diagnostic": (
            HARDWARE_ALIGNMENT_RELAXED_FOR_DIAGNOSTIC
        ),
        "only_variable": ONLY_VARIABLE,
        "score_formula_control": FORMULA_CONTROL,
        "score_formula_candidate": FORMULA_CANDIDATE,
    }


__all__ = [
    "ANCHOR_GRID",
    "ARM_LABEL",
    "AnchorFillSelection",
    "CAUSAL_DIAGNOSTIC_ONLY",
    "CUTPOINT_PREFIX_BLOCKS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_REPLAY_JSONL",
    "EXPECTED_SAMPLES",
    "FORMULA_CANDIDATE",
    "FORMULA_CONTROL",
    "FPGA_DEPLOYABLE_CANDIDATE",
    "HARDWARE_ALIGNMENT_RELAXED_FOR_DIAGNOSTIC",
    "InputContractError",
    "MIN_REFERENCE_K",
    "MODE_CANDIDATE",
    "ONLY_VARIABLE",
    "REFERENCE_SELECTOR",
    "REPORT_STEM",
    "ReplayContractError",
    "ReplayInputs",
    "SELECTION_RULE_CANDIDATE",
    "SELECTION_RULE_CONTROL",
    "ScoreFormulaReplayOutput",
    "UnexpectedIdentityBypassError",
    "anchor_fill_selection",
    "control_group_scores",
    "diagnostic_flags",
    "exact_index_match",
    "file_sha256",
    "forward_layer5_score_formula_replay_vision",
    "historical_group_scores",
    "indices_sha256",
    "jaccard",
    "load_replay_inputs",
    "merged_grid_from_thw",
    "normalize_control_record",
    "normalize_reference_record",
    "normalized_indices",
    "resume_completed_ids",
    "score_summary",
    "select_candidate_groups",
    "select_groups_by_score",
    "validate_replay_record",
]
