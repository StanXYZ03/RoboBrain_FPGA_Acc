#!/usr/bin/env python3
"""CPU-only, read-only postmortem of the frozen B1 answer-margin Oracle STOP.

The preregistered challenge/control pilot24 already decided ``PILOT_STOP``
(challenge repairs 10/12, control regressions 3/12, parse 24/24, exact
K96/87+9 layout 24/24).  This module does **not** re-run the pilot, does not
touch model weights / CUDA / the network, and does not overturn, re-tune or
re-argue the preregistered STOP.  It only reads the frozen raw records and
computes descriptive diagnostics:

* full fail-closed verification of every frozen input SHA256, the 24 unique
  identical sample ids, every recorded record/SHA field, the frozen transition
  labels and the exact K96 (output 96 / real 87 / context 9) layout;
* per-sample Oracle-vs-corrected-L2 selection overlap (all slots, real slots,
  context representatives), the fraction of the recorded Oracle absolute
  gradient saliency mass retained by each arm's selected **real** indices, the
  paired difference, the full-sequence answer margin, G, compression ratio and
  recorded timing / peak-memory fields as diagnostics;
* the three regression ids and the two unrepaired challenge ids;
* **exploratory** transition-group means/medians and Spearman associations
  (no confidence interval, no p-value, no significance language, no rule
  tuning) between the signed transition outcome and margin / G / Jaccard /
  retained-mass advantage;
* a machine-readable decision boundary: the direct absolute-gradient
  full-replacement teacher remains stopped.

Scientific status (must never be softened)
------------------------------------------
The 24-sample manifest is outcome-enriched (dense/control correctness used for
selection) and the Oracle uses the official ground-truth answer branch token.
This is outcome/label leakage plus ground-truth answer leakage: the numbers are
non-representative, non-deployable and are **not** official CV-Bench accuracy.
Nothing here is FPGA / latency / acceleration evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluation.b1_answer_margin_oracle import (  # noqa: E402
    ARM_ORACLE,
    BUDGET,
    EXPECTED_CONTEXT,
    EXPECTED_OUTPUT,
    EXPECTED_REAL,
    LOCAL_CANDIDATE_CAP,
    REGION_GRID_FIXED,
    array_sha256,
    file_sha256,
    record_sha256,
    retained_mass,
    selection_overlap,
)
from evaluation.build_cv_bench_pruning_report import answer_letter, parse_choice  # noqa: E402
from evaluation.hardware_regular_token_selector import (  # noqa: E402
    SOURCE_CONTEXT,
    SOURCE_REAL,
)

ROOT = Path(__file__).resolve().parents[1]
STEM = "2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24"
REPORT_STEM = "2026-10-05-b1-answer-margin-oracle-stop-postmortem"

CONFIG = ROOT / "configs" / f"{STEM}.preregistered.json"
MANIFEST = ROOT / "datasets" / f"{STEM}.jsonl"
ORACLE_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / STEM
    / "oracle_answer_margin_b1_saliency_k96.jsonl"
)
SUMMARY_JSON = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-05-b1-answer-margin-oracle-challenge-control-pilot-summary.json"
)
CONTROL_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-b1-corrected-l2-prompt-ablation-full2638"
    / "corrected_l2_integer_k96_full2638.jsonl"
)
REPORT_DIR = ROOT / "reports" / "token_selector"
REPORT_JSON = REPORT_DIR / f"{REPORT_STEM}.json"
REPORT_CSV = REPORT_DIR / f"{REPORT_STEM}.csv"
REPORT_MD = REPORT_DIR / f"{REPORT_STEM}.md"

CONTROL_ARM = "corrected_l2_integer"

#: Frozen SHA256 of the five inputs this postmortem is allowed to read.  The
#: preregistered config only pins the Oracle output weakly (``sha256: null``,
#: ``pending_gpu`` at pre-registration time), so the postmortem hard-codes the
#: realised hashes and fails closed on any drift.
FROZEN_INPUT_SHA256 = {
    "config": "a0e083b04e845d4bbd94051a3929ac301c2a6ee6b92e8ad72e4b09ec3f261de6",
    "manifest": "0bbe413d55750282a6d7f19650c13886d0e3355f0a82762afd7b5f968ba2a7ca",
    "oracle": "13be51fa120b4f62ef665d3cdebef0b5af96b748a8b5596e76a98d98e1273862",
    "summary": "63c3e790a06d6b697e214ec3cb1c0a7f95d3fa2d98219056bfafe327277a98db",
    "control": "d0e302f6b91279374ddd507db4d300ca0c29e16f522db57f47c8de059f103a5f",
}

PILOT_SAMPLE_COUNT = 24
FROZEN_STATUS = "PILOT_STOP"
FROZEN_TRANSITIONS = {
    "W->R": 10,
    "R->W": 3,
    "unchanged_correct": 9,
    "unchanged_wrong": 2,
}
FROZEN_DECISION = {
    "challenge_repairs": 10,
    "control_regressions": 3,
    "oracle_parse_count": 24,
    "layout_contract_pass_count": 24,
}

TRANSITIONS = ("W->R", "R->W", "unchanged_correct", "unchanged_wrong")

#: Variables summarised per transition group (descriptive only).
GROUP_VARIABLES = (
    "oracle_margin",
    "G",
    "compression_ratio",
    "all_slot_intersection",
    "all_slot_jaccard",
    "real_slot_intersection",
    "real_slot_jaccard",
    "context_slot_intersection",
    "context_slot_jaccard",
    "oracle_retained_mass_real",
    "l2_retained_mass_real",
    "retained_mass_advantage",
)

#: Variables correlated (Spearman) with the signed transition outcome.
SPEARMAN_VARIABLES = (
    "oracle_margin",
    "G",
    "all_slot_jaccard",
    "real_slot_jaccard",
    "context_slot_jaccard",
    "retained_mass_advantage",
)

CSV_FIELDS = (
    "sample_id",
    "task",
    "source",
    "source_dataset",
    "cell",
    "stratum",
    "ground_truth",
    "control_answer",
    "oracle_answer",
    "control_correct",
    "oracle_correct",
    "transition",
    "signed_transition",
    "oracle_margin",
    "G",
    "compression_ratio",
    "all_slot_intersection",
    "all_slot_union",
    "all_slot_jaccard",
    "real_slot_intersection",
    "real_slot_union",
    "real_slot_jaccard",
    "context_slot_intersection",
    "context_slot_union",
    "context_slot_jaccard",
    "oracle_real_count",
    "l2_real_count",
    "oracle_context_count",
    "l2_context_count",
    "oracle_retained_mass_real",
    "l2_retained_mass_real",
    "retained_mass_advantage",
    "oracle_retained_mass_all",
    "l2_retained_mass_all",
    "diagnostic_oracle_model_inference_seconds",
    "diagnostic_oracle_peak_cuda_bytes",
    "diagnostic_control_model_inference_seconds",
    "diagnostic_control_peak_cuda_bytes",
)

__all__ = [
    "CONTROL_JSONL",
    "FROZEN_INPUT_SHA256",
    "MANIFEST",
    "ORACLE_JSONL",
    "SUMMARY_JSON",
    "PostmortemInputError",
    "PostmortemPaths",
    "analyze",
    "build_parser",
    "build_report",
    "csv_rows",
    "main",
    "render_markdown",
    "signed_transition",
    "spearman_rho",
]


class PostmortemInputError(RuntimeError):
    """Raised when any frozen input violates the postmortem contract."""


@dataclass(frozen=True)
class PostmortemPaths:
    config: Path = CONFIG
    manifest: Path = MANIFEST
    oracle: Path = ORACLE_JSONL
    summary: Path = SUMMARY_JSON
    control: Path = CONTROL_JSONL


DEFAULT_PATHS = PostmortemPaths()


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------
def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PostmortemInputError(f"missing required input {label!r}: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PostmortemInputError(
                f"{label} line {line_number} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise PostmortemInputError(f"{label} line {line_number} is not an object")
        rows.append(row)
    return rows


def _index_unique(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if "sample_id" not in row:
            raise PostmortemInputError(f"{label} record is missing sample_id")
        sample_id = str(row["sample_id"])
        if sample_id in index:
            raise PostmortemInputError(f"{label} has duplicate sample_id {sample_id!r}")
        index[sample_id] = row
    return index


def signed_transition(control_correct: bool, oracle_correct: bool) -> int:
    """Signed outcome used only for exploratory association: +1 / 0 / -1."""

    return int(bool(oracle_correct)) - int(bool(control_correct))


def _rankdata(values: Sequence[float]) -> list[float]:
    """Tie-averaged 1-based ranks (no SciPy dependency)."""

    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while (
            end + 1 < len(order)
            and values[order[end + 1]] == values[order[position]]
        ):
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    return ranks


def spearman_rho(left: Sequence[float], right: Sequence[float]) -> float:
    """Spearman rank correlation with tie-averaged ranks, pure-numpy.

    Descriptive only.  Returns ``0.0`` when either variable is constant or the
    sample is too small.  No significance and no p-value are ever derived.
    """

    if len(left) != len(right):
        raise PostmortemInputError("spearman inputs must have equal length")
    if len(left) < 2:
        return 0.0
    rank_left = _rankdata([float(value) for value in left])
    rank_right = _rankdata([float(value) for value in right])
    array_left = np.asarray(rank_left, dtype=np.float64)
    array_right = np.asarray(rank_right, dtype=np.float64)
    if float(array_left.std()) == 0.0 or float(array_right.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(array_left, array_right)[0, 1])


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0


# ---------------------------------------------------------------------------
# Fail-closed verification of the frozen records
# ---------------------------------------------------------------------------
def _verify_array_fields(
    record: Mapping[str, Any], label: str, sample_id: str
) -> None:
    specs = (
        ("selected_group_indices", "selected_group_indices_sha256", np.int64),
        ("selected_patch_indices", "selected_patch_indices_sha256", np.int64),
        ("source_type", "source_type_sha256", np.int64),
        ("region_id", "region_id_sha256", np.int64),
        ("valid", "valid_sha256", bool),
    )
    for field, digest_field, dtype in specs:
        values = record.get(field)
        if not isinstance(values, list):
            raise PostmortemInputError(
                f"{label} sample {sample_id!r} field {field!r} is not a list"
            )
        actual = array_sha256(np.asarray(values, dtype=dtype))
        expected = record.get(digest_field)
        if str(expected) != actual:
            raise PostmortemInputError(
                f"{label} sample {sample_id!r} {digest_field} mismatch (fail closed)"
            )


def _validate_selection(
    record: Mapping[str, Any],
    label: str,
    sample_id: str,
    *,
    expected_arm: str,
) -> dict[str, Any]:
    """Return normalised selection arrays after contract checks."""

    if str(record.get("arm")) != expected_arm:
        raise PostmortemInputError(
            f"{label} sample {sample_id!r} arm={record.get('arm')!r} != {expected_arm!r}"
        )
    if str(record.get("cutpoint")) != "B1_after_vision_block_0":
        raise PostmortemInputError(f"{label} sample {sample_id!r} cutpoint mismatch")
    if int(record.get("prune_after_layer", -1)) != 0:
        raise PostmortemInputError(
            f"{label} sample {sample_id!r} prune_after_layer != 0"
        )
    for field, expected in (
        ("requested_budget", BUDGET),
        ("region_grid", REGION_GRID_FIXED),
        ("local_candidate_cap", LOCAL_CANDIDATE_CAP),
        ("output_group_count", EXPECTED_OUTPUT),
        ("real_group_count", EXPECTED_REAL),
        ("context_group_count", EXPECTED_CONTEXT),
    ):
        try:
            actual = int(record.get(field))
        except (TypeError, ValueError) as exc:
            raise PostmortemInputError(
                f"{label} sample {sample_id!r} {field}={record.get(field)!r} is not "
                f"an integer"
            ) from exc
        if actual != int(expected):
            raise PostmortemInputError(
                f"{label} sample {sample_id!r} {field}={actual} != {expected}"
            )
    if bool(record.get("identity_bypass")):
        raise PostmortemInputError(
            f"{label} sample {sample_id!r} is an identity bypass"
        )
    if bool(record.get("do_sample")):
        raise PostmortemInputError(f"{label} sample {sample_id!r} do_sample is not False")
    if int(record.get("max_new_tokens", -1)) != 16:
        raise PostmortemInputError(
            f"{label} sample {sample_id!r} max_new_tokens != 16"
        )
    full_groups = int(record.get("full_llm_visual_tokens", -1))
    if full_groups <= BUDGET:
        raise PostmortemInputError(
            f"{label} sample {sample_id!r} G={full_groups} <= {BUDGET}"
        )
    grid = [int(value) for value in record.get("image_grid_thw", [])]
    if len(grid) != 3 or grid[0] != 1 or grid[1] * grid[2] != int(
        record.get("full_patch_tokens", -1)
    ):
        raise PostmortemInputError(
            f"{label} sample {sample_id!r} image_grid_thw {grid} is inconsistent"
        )
    if int(record.get("full_patch_tokens", -1)) != full_groups * 4:
        raise PostmortemInputError(
            f"{label} sample {sample_id!r} full_patch_tokens != 4*G"
        )

    _verify_array_fields(record, label, sample_id)

    selected = np.asarray(record["selected_group_indices"], dtype=np.int64)
    patch = np.asarray(record["selected_patch_indices"], dtype=np.int64)
    source = np.asarray(record["source_type"], dtype=np.int64)
    region = np.asarray(record["region_id"], dtype=np.int64)
    valid = np.asarray(record["valid"], dtype=bool)

    for field, array, expected_size in (
        ("selected_group_indices", selected, EXPECTED_OUTPUT),
        ("selected_patch_indices", patch, EXPECTED_OUTPUT * 4),
        ("source_type", source, EXPECTED_OUTPUT),
        ("region_id", region, EXPECTED_OUTPUT),
        ("valid", valid, EXPECTED_OUTPUT),
    ):
        if int(array.size) != expected_size:
            raise PostmortemInputError(
                f"{label} sample {sample_id!r} {field} has {array.size} entries "
                f"!= {expected_size}"
            )
    if bool((np.diff(selected) <= 0).any()):
        raise PostmortemInputError(
            f"{label} sample {sample_id!r} selected_group_indices not strictly ascending"
        )
    if int(selected.min()) < 0 or int(selected.max()) >= full_groups:
        raise PostmortemInputError(
            f"{label} sample {sample_id!r} selected index outside [0, G)"
        )
    if len(set(int(value) for value in selected.tolist())) != EXPECTED_OUTPUT:
        raise PostmortemInputError(
            f"{label} sample {sample_id!r} selected indices are not unique"
        )
    if not bool(np.isin(source, (SOURCE_REAL, SOURCE_CONTEXT)).all()):
        raise PostmortemInputError(
            f"{label} sample {sample_id!r} has unknown source_type values"
        )
    real_count = int((source == SOURCE_REAL).sum())
    context_count = int((source == SOURCE_CONTEXT).sum())
    if real_count != EXPECTED_REAL or context_count != EXPECTED_CONTEXT:
        raise PostmortemInputError(
            f"{label} sample {sample_id!r} real/context counts "
            f"{real_count}/{context_count} != {EXPECTED_REAL}/{EXPECTED_CONTEXT}"
        )
    if bool(valid.all()) is False:
        raise PostmortemInputError(f"{label} sample {sample_id!r} has invalid slots")
    if int(region.min()) < 0 or int(region.max()) >= EXPECTED_CONTEXT:
        raise PostmortemInputError(
            f"{label} sample {sample_id!r} region_id outside [0, {EXPECTED_CONTEXT})"
        )
    expected_patch = (
        selected[:, None] * 4 + np.arange(4, dtype=np.int64)[None, :]
    ).reshape(-1)
    if not bool(np.array_equal(patch, expected_patch)):
        raise PostmortemInputError(
            f"{label} sample {sample_id!r} patch indices are not original positions"
        )

    previous = record_sha256(record)
    if str(record.get("record_sha256")) != previous:
        raise PostmortemInputError(
            f"{label} sample {sample_id!r} record_sha256 mismatch (fail closed)"
        )
    return {
        "selected": selected,
        "source": source,
        "region": region,
        "full_groups": full_groups,
        "real_indices": sorted(
            int(value) for value in selected[source == SOURCE_REAL].tolist()
        ),
        "context_indices": sorted(
            int(value) for value in selected[source == SOURCE_CONTEXT].tolist()
        ),
    }


def _validate_oracle(
    sample_id: str,
    record: Mapping[str, Any],
    *,
    manifest_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    for field, expected in (
        ("manifest_sha256", manifest_sha256),
        ("config_sha256", config_sha256),
    ):
        if str(record.get(field)) != str(expected):
            raise PostmortemInputError(
                f"oracle sample {sample_id!r} {field} mismatch (fail closed)"
            )
    if str(record.get("score_mode")) != "oracle_b1_answer_margin_saliency":
        raise PostmortemInputError(f"oracle sample {sample_id!r} score_mode mismatch")
    selection = _validate_selection(
        record, "oracle", sample_id, expected_arm=ARM_ORACLE
    )
    full_groups = selection["full_groups"]
    values = np.asarray(record.get("saliency_group_values", []), dtype=np.float64)
    if int(values.size) != full_groups:
        raise PostmortemInputError(
            f"oracle sample {sample_id!r} saliency size {values.size} != G {full_groups}"
        )
    if not bool(np.isfinite(values).all()):
        raise PostmortemInputError(
            f"oracle sample {sample_id!r} saliency is non-finite"
        )
    if str(record.get("saliency_sha256")) != array_sha256(values):
        raise PostmortemInputError(
            f"oracle sample {sample_id!r} saliency_sha256 mismatch (fail closed)"
        )
    margin = record.get("margin") or {}
    margin_value = margin.get("value")
    if margin_value is None or not math.isfinite(float(margin_value)):
        raise PostmortemInputError(
            f"oracle sample {sample_id!r} margin is not finite"
        )
    for field, expected in (
        ("output_group_count", EXPECTED_OUTPUT),
        ("real_group_count", EXPECTED_REAL),
        ("context_group_count", EXPECTED_CONTEXT),
    ):
        if int(record.get(field, -1)) != int(expected):
            raise PostmortemInputError(
                f"oracle sample {sample_id!r} {field} != {expected}"
            )
    layout = record.get("layout") or {}
    if not bool(layout.get("passed")):
        raise PostmortemInputError(
            f"oracle sample {sample_id!r} layout contract did not pass"
        )
    for field, expected in (
        ("output_group_count", EXPECTED_OUTPUT),
        ("real_group_count", EXPECTED_REAL),
        ("context_group_count", EXPECTED_CONTEXT),
    ):
        if int(layout.get(field, -1)) != int(expected):
            raise PostmortemInputError(
                f"oracle sample {sample_id!r} layout.{field} != {expected}"
            )
    if bool(layout.get("identity_bypass")):
        raise PostmortemInputError(
            f"oracle sample {sample_id!r} layout is an identity bypass"
        )
    if not bool(record.get("model_parameter_grads_all_none")):
        raise PostmortemInputError(
            f"oracle sample {sample_id!r} has non-None parameter grads"
        )
    selection["saliency"] = values
    selection["margin"] = float(margin_value)
    return selection


def _validate_control(
    sample_id: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    if str(record.get("score_mode")) != "integer":
        raise PostmortemInputError(
            f"control sample {sample_id!r} score_mode != integer"
        )
    return _validate_selection(
        record, "control", sample_id, expected_arm=CONTROL_ARM
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _overlap(
    left: Sequence[int], right: Sequence[int]
) -> dict[str, Any]:
    return selection_overlap(left, right)


def _retained(
    saliency: np.ndarray, indices: Sequence[int], label: str
) -> float:
    if len(indices) == 0:
        return 0.0
    array = np.asarray(indices, dtype=np.int64)
    if int(array.min()) < 0 or int(array.max()) >= int(saliency.size):
        raise PostmortemInputError(f"{label} retained-mass index outside saliency")
    return retained_mass(saliency, list(indices))


def _sample_metrics(
    sample_id: str,
    manifest_row: Mapping[str, Any],
    oracle: Mapping[str, Any],
    control: Mapping[str, Any],
    summary_row: Mapping[str, Any],
    *,
    manifest_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    meta = manifest_row.get("oracle_pilot")
    if not isinstance(meta, Mapping):
        raise PostmortemInputError(
            f"manifest sample {sample_id!r} has no oracle_pilot metadata"
        )
    if str(meta.get("cell")) != str(summary_row.get("cell")):
        raise PostmortemInputError(
            f"manifest/summary cell mismatch for sample {sample_id!r}"
        )
    if str(meta.get("stratum")) != str(summary_row.get("stratum")):
        raise PostmortemInputError(
            f"manifest/summary stratum mismatch for sample {sample_id!r}"
        )
    truth = answer_letter(str(manifest_row["answer"]))
    if truth != str(summary_row.get("ground_truth")):
        raise PostmortemInputError(
            f"summary ground_truth mismatch for sample {sample_id!r}"
        )

    choices = [str(choice) for choice in manifest_row.get("choices", [])]
    control_parsed = parse_choice(str(control["output_text"]), list(choices))
    oracle_parsed = parse_choice(str(oracle["output_text"]), list(choices))
    control_correct = bool(control_parsed is not None and control_parsed == truth)
    oracle_correct = bool(oracle_parsed is not None and oracle_parsed == truth)
    for arm, recomputed in (
        ("control_correct", control_correct),
        ("oracle_correct", oracle_correct),
    ):
        if bool(summary_row.get(arm)) != recomputed:
            raise PostmortemInputError(
                f"summary {arm} mismatch for sample {sample_id!r} (fail closed)"
            )
    expected_transition = (
        "W->R"
        if (not control_correct and oracle_correct)
        else "R->W"
        if (control_correct and not oracle_correct)
        else "unchanged_correct"
        if control_correct
        else "unchanged_wrong"
    )
    if str(summary_row.get("transition")) != expected_transition:
        raise PostmortemInputError(
            f"summary transition mismatch for sample {sample_id!r} "
            f"({summary_row.get('transition')!r} != {expected_transition!r})"
        )

    oracle_sel = _validate_oracle(
        sample_id,
        oracle,
        manifest_sha256=manifest_sha256,
        config_sha256=config_sha256,
    )
    control_sel = _validate_control(sample_id, control)
    if int(oracle_sel["full_groups"]) != int(control_sel["full_groups"]):
        raise PostmortemInputError(
            f"sample {sample_id!r} G differs between oracle and control"
        )
    full_groups = int(oracle_sel["full_groups"])
    saliency = oracle_sel["saliency"]

    oracle_indices = [int(value) for value in oracle_sel["selected"].tolist()]
    control_indices = [int(value) for value in control_sel["selected"].tolist()]
    all_slot = _overlap(oracle_indices, control_indices)
    real_slot = _overlap(oracle_sel["real_indices"], control_sel["real_indices"])
    context_slot = _overlap(
        oracle_sel["context_indices"], control_sel["context_indices"]
    )

    oracle_retained = _retained(
        saliency, oracle_sel["real_indices"], "oracle real"
    )
    l2_retained = _retained(saliency, control_sel["real_indices"], "l2 real")
    oracle_retained_all = _retained(saliency, oracle_indices, "oracle all")
    l2_retained_all = _retained(saliency, control_indices, "l2 all")

    margin = float(oracle_sel["margin"])
    summary_margin = summary_row.get("oracle_margin")
    if summary_margin is None or abs(float(summary_margin) - margin) > 1e-9:
        raise PostmortemInputError(
            f"summary oracle_margin mismatch for sample {sample_id!r}"
        )

    def _diag(record: Mapping[str, Any], field: str) -> Any:
        value = record.get(field)
        if value is None:
            value = record.get("timings", {}).get(field) if isinstance(
                record.get("timings"), Mapping
            ) else None
        return float(value) if isinstance(value, (int, float)) and not isinstance(
            value, bool
        ) else None

    return {
        "sample_id": sample_id,
        "task": str(manifest_row["task"]),
        "source": str(manifest_row["source"]),
        "source_dataset": str(manifest_row["source_dataset"]),
        "cell": str(meta.get("cell")),
        "stratum": str(meta.get("stratum")),
        "ground_truth": truth,
        "control_answer": control_parsed,
        "oracle_answer": oracle_parsed,
        "control_correct": control_correct,
        "oracle_correct": oracle_correct,
        "transition": expected_transition,
        "signed_transition": signed_transition(control_correct, oracle_correct),
        "oracle_margin": margin,
        "G": full_groups,
        "compression_ratio": EXPECTED_OUTPUT / full_groups,
        "all_slot_intersection": int(all_slot["intersection"]),
        "all_slot_union": int(all_slot["union"]),
        "all_slot_jaccard": float(all_slot["jaccard"]),
        "real_slot_intersection": int(real_slot["intersection"]),
        "real_slot_union": int(real_slot["union"]),
        "real_slot_jaccard": float(real_slot["jaccard"]),
        "context_slot_intersection": int(context_slot["intersection"]),
        "context_slot_union": int(context_slot["union"]),
        "context_slot_jaccard": float(context_slot["jaccard"]),
        "oracle_real_count": len(oracle_sel["real_indices"]),
        "l2_real_count": len(control_sel["real_indices"]),
        "oracle_context_count": len(oracle_sel["context_indices"]),
        "l2_context_count": len(control_sel["context_indices"]),
        "oracle_retained_mass_real": oracle_retained,
        "l2_retained_mass_real": l2_retained,
        "retained_mass_advantage": oracle_retained - l2_retained,
        "oracle_retained_mass_all": oracle_retained_all,
        "l2_retained_mass_all": l2_retained_all,
        "diagnostic_oracle_model_inference_seconds": _diag(
            oracle, "model_inference_seconds"
        ),
        "diagnostic_oracle_peak_cuda_bytes": _diag(oracle, "peak_cuda_bytes"),
        "diagnostic_control_model_inference_seconds": _diag(
            control, "model_inference_seconds"
        ),
        "diagnostic_control_peak_cuda_bytes": _diag(control, "peak_cuda_bytes"),
    }


def _group_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {"n": len(rows)}
    for variable in GROUP_VARIABLES:
        values = [float(row[variable]) for row in rows]
        stats[variable] = {
            "mean": _mean(values),
            "median": _median(values),
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }
    return stats


def _can_conclude(per_sample: Sequence[Mapping[str, Any]]) -> list[str]:
    mean_jaccard = _mean([row["real_slot_jaccard"] for row in per_sample])
    mean_advantage = _mean(
        [row["retained_mass_advantage"] for row in per_sample]
    )
    return [
        "the frozen pilot decision is PILOT_STOP (10/12 repairs, 3/12 control "
        "regressions, parse 24/24, exact K96/87+9 layout 24/24) and every frozen "
        "input hash, record hash, id, transition label and layout verified",
        "the direct absolute-gradient full-replacement teacher repaired 10/12 "
        "challenges but produced 3/12 control regressions and left 2/12 "
        "challenges unrepaired, so full replacement fails the preregistered "
        "control-regression gate",
        f"Oracle and corrected L2 select largely different slots (mean real-slot "
        f"Jaccard {mean_jaccard:.4f}); the Oracle retains a larger fraction of its "
        f"recorded absolute gradient saliency mass on the real slots "
        f"(mean paired advantage {mean_advantage:.4f})",
        "the observed associations between margin/G/Jaccard/retained-mass "
        "advantage and the signed transition outcome are exploratory only at "
        "n=24 with an outcome-enriched manifest",
    ]


def _cannot_conclude(per_sample: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        "any official CV-Bench accuracy, deployability, FPGA/latency/speedup or "
        "representative-sampling statement",
        "a causal mechanism for the 3 regressions or the 2 unrepaired "
        "challenges; n=24 outcome-enriched samples cannot support it",
        "that a lower dense answer margin alone explains the failures (the "
        "unrepaired challenge includes a 7.5-margin sample)",
        "that excessive selection churn relative to L2 explains the failures "
        "(the unrepaired challenges have the highest real-slot Jaccard)",
        "that the unsigned absolute |grad*activation| signal preserves the "
        "correct-answer margin after the nonlinear discrete K96 pruning; the "
        "pilot only records retained saliency mass, not a margin after pruning",
        "that the context/anchor reservation rule causes any observed "
        "difference; the per-slot scores are not recorded and no context/anchor "
        "ablation arm exists in this pilot",
        "any tuned numeric mixture, threshold, K or anchor change on these 24 "
        "samples",
    ]


def _explanation_audit(
    per_sample: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Any],
) -> dict[str, Any]:
    failure_rows = [
        row
        for row in per_sample
        if row["transition"] in {"R->W", "unchanged_wrong"}
    ]
    unchanged_correct = [
        row for row in per_sample if row["transition"] == "unchanged_correct"
    ]
    advantage_positive_all = all(
        row["retained_mass_advantage"] > 0 for row in per_sample
    )
    failure_advantage = _mean(
        [row["retained_mass_advantage"] for row in failure_rows]
    )
    stable_advantage = _mean(
        [row["retained_mass_advantage"] for row in unchanged_correct]
    )
    max_failure_margin = max(row["oracle_margin"] for row in failure_rows)
    return {
        "low_dense_answer_margin": {
            "verdict": "unproven_as_sufficient_explanation",
            "is_established_fact": False,
            "note": (
                "The full-sequence answer margin is recorded only for the Oracle "
                "arm. The failure groups are not confined to low margins "
                f"(largest failure-group margin = {max_failure_margin:.4f}); a "
                "low dense margin is therefore not a supported sufficient "
                "explanation of the STOP."
            ),
        },
        "excessive_selection_churn_relative_to_l2": {
            "verdict": "not_supported_by_observed_association",
            "is_established_fact": False,
            "note": (
                "Real-slot Jaccard is low overall (mean "
                f"{_mean([row['real_slot_jaccard'] for row in per_sample]):.4f}), "
                "but the unrepaired challenges have the highest mean Jaccard "
                f"({groups['unchanged_wrong']['real_slot_jaccard']['mean']:.4f}) "
                "and the unchanged-correct group the lowest "
                f"({groups['unchanged_correct']['real_slot_jaccard']['mean']:.4f}), "
                "so churn alone does not order the outcomes."
            ),
        },
        "unsigned_absolute_gradient_mass": {
            "verdict": "plausible_but_unproven",
            "is_established_fact": False,
            "retained_mass_advantage_positive_all_24": bool(advantage_positive_all),
            "failure_group_mean_advantage": failure_advantage,
            "unchanged_correct_mean_advantage": stable_advantage,
            "note": (
                "The Oracle retains more recorded absolute |grad*activation| "
                "mass than corrected L2 in every sample, yet still regresses 3 "
                "controls; the observed failure groups have a lower mean "
                "advantage. Because the score is unsigned and K96 pruning is "
                "nonlinear and discrete, retaining more absolute mass does not "
                "by itself imply the correct-answer margin survives. This is a "
                "plausible but untested mechanism."
            ),
        },
        "context_anchor_rule_interaction": {
            "verdict": "cannot_be_isolated_from_recorded_metadata",
            "is_established_fact": False,
            "note": (
                "Per-slot scores are not stored for either arm and there is no "
                "context-reservation / anchor ablation arm in this pilot, so the "
                "context/anchor rule interaction cannot be isolated from the "
                "frozen records. Context-representative overlap is measurable "
                "and reported, but it is descriptive only."
            ),
        },
    }


def _decision_boundary(
    groups: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    advantage = audit["unsigned_absolute_gradient_mass"]
    evidence_supports = bool(
        advantage["retained_mass_advantage_positive_all_24"]
        and advantage["failure_group_mean_advantage"]
        < advantage["unchanged_correct_mean_advantage"]
    )
    return {
        "direct_absolute_gradient_full_replacement_teacher": "STOPPED",
        "preregistered_pilot_status": FROZEN_STATUS,
        "preregistered_stop_overturned": False,
        "reproducible_on_frozen_records": True,
        "thresholds_K_anchors_or_mixtures_tuned_on_24_samples": False,
        "max_next_mechanism_classes": 1,
        "recommended_next_mechanism_class": (
            "conservative_l2_backbone_residual_task_aware_reranking"
            if evidence_supports
            else None
        ),
        "recommendation_status": (
            "unproven_hypothesis_for_next_falsifiable_test"
            if evidence_supports
            else "no_next_selector_mechanism_supported_by_this_evidence"
        ),
        "recommendation_evidence": {
            "retained_mass_advantage_positive_all_24": bool(
                advantage["retained_mass_advantage_positive_all_24"]
            ),
            "failure_group_mean_advantage_lt_unchanged_correct_mean": bool(
                advantage["failure_group_mean_advantage"]
                < advantage["unchanged_correct_mean_advantage"]
            ),
            "full_replacement_repairs": FROZEN_DECISION["challenge_repairs"],
            "full_replacement_regressions": FROZEN_DECISION[
                "control_regressions"
            ],
        },
        "numeric_mixture_specified": False,
        "note": (
            "The direct absolute-gradient full-replacement teacher remains "
            "stopped and the preregistered STOP is not overturned. If (and only "
            "if) the next step is a bounded, conservative residual re-ranking on "
            "the corrected-L2 backbone, it is proposed here solely as an "
            "unproven, falsifiable hypothesis; no numeric mixture, threshold, K "
            "or anchor is specified or tuned on these 24 samples."
        ),
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    manifest_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    manifest_index = _index_unique(manifest_rows, "pilot_manifest")
    oracle_index = _index_unique(oracle_rows, "oracle")
    control_index = _index_unique(control_rows, "control")

    if len(manifest_index) != PILOT_SAMPLE_COUNT:
        raise PostmortemInputError(
            f"manifest must hold {PILOT_SAMPLE_COUNT} unique samples, got "
            f"{len(manifest_index)}"
        )
    expected_ids = set(manifest_index)
    if set(oracle_index) != expected_ids:
        raise PostmortemInputError(
            "oracle sample_id set != manifest sample_id set (fail closed)"
        )
    if not expected_ids <= set(control_index):
        missing = sorted(expected_ids - set(control_index))
        raise PostmortemInputError(
            f"control is missing manifest sample_ids {missing[:5]}"
        )

    per_sample_summary = summary.get("per_sample")
    if not isinstance(per_sample_summary, list):
        raise PostmortemInputError("summary.per_sample must be a list")
    summary_index = _index_unique(per_sample_summary, "summary")
    if set(summary_index) != expected_ids:
        raise PostmortemInputError(
            "summary per_sample ids != manifest ids (fail closed)"
        )
    if int(summary.get("sample_count", -1)) != PILOT_SAMPLE_COUNT:
        raise PostmortemInputError("summary sample_count != 24")

    per_sample = [
        _sample_metrics(
            sample_id,
            manifest_index[sample_id],
            oracle_index[sample_id],
            control_index[sample_id],
            summary_index[sample_id],
            manifest_sha256=manifest_sha256,
            config_sha256=config_sha256,
        )
        for sample_id in sorted(
            expected_ids,
            key=lambda value: (
                [
                    cell
                    for cell in (
                        "ADE20K x Count",
                        "ADE20K x Relation",
                        "COCO x Count",
                        "COCO x Relation",
                        "Omni3D x Depth",
                        "Omni3D x Distance",
                    )
                ].index(str(manifest_index[value]["oracle_pilot"]["cell"])),
                ("challenge", "control").index(
                    str(manifest_index[value]["oracle_pilot"]["stratum"])
                ),
                value,
            ),
        )
    ]

    transitions = {
        transition: sum(
            1 for row in per_sample if row["transition"] == transition
        )
        for transition in TRANSITIONS
    }
    if transitions != FROZEN_TRANSITIONS:
        raise PostmortemInputError(
            f"frozen transition labels mismatch: {transitions} != {FROZEN_TRANSITIONS}"
        )
    summary_transitions = summary.get("transitions")
    if summary_transitions != FROZEN_TRANSITIONS:
        raise PostmortemInputError(
            "summary transitions != frozen transition labels (fail closed)"
        )

    decision = summary.get("decision") or {}
    if str(decision.get("status")) != FROZEN_STATUS:
        raise PostmortemInputError(
            f"summary status {decision.get('status')!r} != {FROZEN_STATUS!r}"
        )
    for field, expected in FROZEN_DECISION.items():
        if int(decision.get(field, -1)) != expected:
            raise PostmortemInputError(
                f"summary decision.{field}={decision.get(field)!r} != {expected}"
            )

    groups = {
        transition: _group_stats(
            [row for row in per_sample if row["transition"] == transition]
        )
        for transition in TRANSITIONS
    }
    signed = [row["signed_transition"] for row in per_sample]
    associations = {
        variable: spearman_rho(
            [row[variable] for row in per_sample], signed
        )
        for variable in SPEARMAN_VARIABLES
    }

    regressions = [
        row for row in per_sample if row["transition"] == "R->W"
    ]
    unrepaired = [
        row for row in per_sample if row["transition"] == "unchanged_wrong"
    ]
    if len(regressions) != FROZEN_DECISION["control_regressions"]:
        raise PostmortemInputError("control regression count mismatch")
    if len(unrepaired) != FROZEN_TRANSITIONS["unchanged_wrong"]:
        raise PostmortemInputError("unrepaired challenge count mismatch")
    for row in unrepaired:
        if row["stratum"] != "challenge":
            raise PostmortemInputError(
                f"unchanged_wrong sample {row['sample_id']!r} is not a challenge"
            )

    audit = _explanation_audit(per_sample, groups)
    boundary = _decision_boundary(groups, audit)

    return {
        "status": "PILOT_STOP_POSTMORTEM_COMPLETE",
        "frozen_status": FROZEN_STATUS,
        "design": STEM,
        "date": "2026-10-05",
        "sample_count": PILOT_SAMPLE_COUNT,
        "verification": {
            "ids_unique_and_identical": True,
            "record_hashes_verified": True,
            "array_sha256_fields_verified": True,
            "frozen_transitions": transitions,
            "frozen_decision": {
                "challenge_repairs": int(decision["challenge_repairs"]),
                "control_regressions": int(decision["control_regressions"]),
                "oracle_parse_count": int(decision["oracle_parse_count"]),
                "layout_contract_pass_count": int(
                    decision["layout_contract_pass_count"]
                ),
                "status": str(decision["status"]),
            },
            "layout_contract": {
                "output_group_count": EXPECTED_OUTPUT,
                "real_group_count": EXPECTED_REAL,
                "context_group_count": EXPECTED_CONTEXT,
                "pass_count": PILOT_SAMPLE_COUNT,
                "denominator": PILOT_SAMPLE_COUNT,
            },
            "manifest_sha256": manifest_sha256,
            "config_sha256": config_sha256,
        },
        "per_sample": per_sample,
        "transition_groups": groups,
        "associations": {
            "method": "spearman_rho tie-averaged ranks; descriptive only",
            "exploratory": True,
            "n": PILOT_SAMPLE_COUNT,
            "outcome_enriched": True,
            "uses_confidence_interval": False,
            "uses_p_value_or_significance": False,
            "signed_transition_outcome": (
                "int(oracle_correct) - int(control_correct) in {+1,0,-1}"
            ),
            "spearman": associations,
        },
        "regressions": regressions,
        "unrepaired_challenges": unrepaired,
        "explanation_audit": audit,
        "decision_boundary": boundary,
        "can_conclude": _can_conclude(per_sample),
        "cannot_conclude": _cannot_conclude(per_sample),
    }


# ---------------------------------------------------------------------------
# Pin verification / report assembly
# ---------------------------------------------------------------------------
def _verify_config_pins(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for section in ("inputs_sha256", "summarizer_inputs"):
        specs = config.get(section, {})
        if not isinstance(specs, Mapping):
            raise PostmortemInputError(f"config {section} must be an object")
        for name, spec in specs.items():
            if not isinstance(spec, Mapping):
                raise PostmortemInputError(
                    f"config pin {section}.{name} must be an object"
                )
            expected = spec.get("sha256")
            if expected is None:
                continue
            relative = spec.get("path")
            if relative is None:
                raise PostmortemInputError(
                    f"config pin {section}.{name} has no path"
                )
            path = Path(str(relative))
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                raise PostmortemInputError(
                    f"pinned file {path} is missing (fail closed)"
                )
            actual = file_sha256(path)
            if actual != str(expected):
                raise PostmortemInputError(
                    f"pin {section}.{name} sha256 mismatch: expected {expected}, "
                    f"got {actual} (fail closed)"
                )
            checks.append(
                {
                    "name": f"{section}.{name}",
                    "path": str(path),
                    "sha256": actual,
                    "matches": True,
                }
            )
    return {"pins_verified": True, "checks": checks}


def build_report(
    paths: PostmortemPaths = DEFAULT_PATHS,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    resolved = {
        "config": paths.config if paths.config.is_absolute() else root / paths.config,
        "manifest": paths.manifest
        if paths.manifest.is_absolute()
        else root / paths.manifest,
        "oracle": paths.oracle if paths.oracle.is_absolute() else root / paths.oracle,
        "summary": paths.summary
        if paths.summary.is_absolute()
        else root / paths.summary,
        "control": paths.control
        if paths.control.is_absolute()
        else root / paths.control,
    }
    for label, path in resolved.items():
        if not path.is_file():
            raise PostmortemInputError(f"missing required input {label!r}: {path}")
        actual = file_sha256(path)
        expected = FROZEN_INPUT_SHA256[label]
        if actual != expected:
            raise PostmortemInputError(
                f"{label} sha256 {actual} != frozen {expected} (fail closed)"
            )
    config = json.loads(resolved["config"].read_text(encoding="utf-8"))
    pins = _verify_config_pins(config, root)

    manifest_rows = _load_jsonl(resolved["manifest"], "manifest")
    oracle_rows = _load_jsonl(resolved["oracle"], "oracle")
    control_rows = _load_jsonl(resolved["control"], "control")
    summary = json.loads(resolved["summary"].read_text(encoding="utf-8"))

    result = analyze(
        manifest_rows=manifest_rows,
        oracle_rows=oracle_rows,
        control_rows=control_rows,
        summary=summary,
        manifest_sha256=FROZEN_INPUT_SHA256["manifest"],
        config_sha256=FROZEN_INPUT_SHA256["config"],
    )
    result["scope"] = (
        "CPU-only, read-only postmortem of the frozen preregistered STOP; no "
        "GPU/model weights/network/training and no threshold/K/anchor tuning"
    )
    result["gpu_runs"] = 0
    result["model_was_run"] = False
    result["training"] = False
    result["is_official_cv_bench_accuracy"] = False
    result["uses_confidence_interval"] = False
    result["uses_significance_or_pvalues"] = False
    result["associations_exploratory"] = True
    result["outcome_enriched_diagnostic"] = True
    result["outcome_label_leakage"] = True
    result["ground_truth_answer_leakage"] = True
    result["non_representative"] = True
    result["non_deployable"] = True
    result["deployable_selector"] = False
    result["is_paper_accuracy_claim"] = False
    result["is_fpga_or_latency_claim"] = False
    result["pins"] = pins
    result["inputs_sha256"] = {
        label: {"path": str(path), "sha256": FROZEN_INPUT_SHA256[label]}
        for label, path in resolved.items()
    }
    result["required_statements"] = {
        "frozen_stop_preserved": (
            "预注册 PILOT_STOP 保持不变，不被推翻、不接受重新调参。"
        ),
        "outcome_label_leakage": (
            "样本按 dense/control 正确性分层，属于 outcome/label leakage；"
            "非代表性、不可部署，禁止官方绝对精度主张。"
        ),
        "ground_truth_leakage": (
            "oracle 使用官方正确答案分叉 token，属于 ground-truth answer leakage，"
            "只能作为可证伪上界。"
        ),
        "descriptive_only": (
            "n=24 outcome-enriched 关联仅探索性，不使用 CI / p 值 / 显著性；"
            "不提出官方精度、部署、FPGA 加速或代表性采样主张。"
        ),
    }
    result["forbidden_claims"] = list(config.get("forbidden_claims", []))
    result["generated_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in report["per_sample"]:
        rows.append({field: row.get(field) for field in CSV_FIELDS})
    return rows


def render_markdown(report: Mapping[str, Any]) -> str:
    decision = report["verification"]["frozen_decision"]
    boundary = report["decision_boundary"]
    groups = report["transition_groups"]
    associations = report["associations"]["spearman"]
    lines = [
        "# B1 answer-margin Oracle challenge/control pilot24 STOP 事后尸检（CPU-only、只读）",
        "",
        f"- 日期：{report['date']}",
        f"- 设计：`{report['design']}`",
        f"- 状态：**{report['status']}**；冻结决策：**{report['frozen_status']}**",
        "- 范围：CPU-only、只读冻结结果；无 GPU / 模型权重 / 网络 / 训练；"
        "未改任何冻结 manifest / config / raw JSONL / summary；未调阈值 / K / anchor。",
        "",
        "## 1. 口径（必须先读）",
        "",
        "- 本报告**不推翻、不重新论证**预注册 STOP；只做描述性事后分析。",
        "- **outcome/label leakage**：样本按 dense/control 正确性分层，非代表性、不可部署；"
        "禁止官方绝对精度主张。",
        "- **ground-truth answer leakage**：oracle 用官方正确答案分叉 token，只是可证伪上界。",
        "- n=24 且 outcome-enriched；所有关联仅探索性，**不使用 CI / p 值 / 显著性**。",
        "",
        "## 2. 冻结决策与验证",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        f"| challenge 修复 | {decision['challenge_repairs']}/12 |",
        f"| control 回归 | {decision['control_regressions']}/12 |",
        f"| oracle parse | {decision['oracle_parse_count']}/24 |",
        f"| 精确 K96/87+9 布局 | {decision['layout_contract_pass_count']}/24 |",
        f"| 冻结状态 | **{decision['status']}** |",
        f"| 输入 SHA / record SHA / ID / 迁移标签 | 全部通过（{report['inputs_sha256']['manifest']['sha256'][:12]}…） |",
        "",
        "## 3. 逐样本指标",
        "",
        "| sample_id | cell | stratum | 迁移 | margin | G | 压缩比 | 全槽 Jaccard | real 槽 Jaccard | context 槽 Jaccard | Oracle 保留质量 | L2 保留质量 | 差值 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["per_sample"]:
        lines.append(
            f"| {row['sample_id']} | {row['cell']} | {row['stratum']} | "
            f"{row['transition']} | {row['oracle_margin']:.4f} | {row['G']} | "
            f"{row['compression_ratio']:.4f} | {row['all_slot_jaccard']:.4f} | "
            f"{row['real_slot_jaccard']:.4f} | {row['context_slot_jaccard']:.4f} | "
            f"{row['oracle_retained_mass_real']:.4f} | "
            f"{row['l2_retained_mass_real']:.4f} | "
            f"{row['retained_mass_advantage']:.4f} |"
        )
    lines += [
        "",
        "## 4. 3 个回归样本与 2 个未修复 challenge 样本",
        "",
        "| 类型 | sample_id | cell | stratum | margin | G | real 槽 Jaccard | 保留质量差值 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["regressions"]:
        lines.append(
            f"| control 回归 R→W | {row['sample_id']} | {row['cell']} | "
            f"{row['stratum']} | {row['oracle_margin']:.4f} | {row['G']} | "
            f"{row['real_slot_jaccard']:.4f} | {row['retained_mass_advantage']:.4f} |"
        )
    for row in report["unrepaired_challenges"]:
        lines.append(
            f"| 未修复 challenge | {row['sample_id']} | {row['cell']} | "
            f"{row['stratum']} | {row['oracle_margin']:.4f} | {row['G']} | "
            f"{row['real_slot_jaccard']:.4f} | {row['retained_mass_advantage']:.4f} |"
        )
    lines += [
        "",
        "## 5. 迁移组描述统计（mean / median；仅探索性）",
        "",
    ]
    for transition in TRANSITIONS:
        item = groups[transition]
        lines.append(f"### {transition}（n={item['n']}）")
        lines.append("")
        lines.append("| 变量 | mean | median | min | max |")
        lines.append("| --- | --- | --- | --- | --- |")
        for variable in GROUP_VARIABLES:
            stats = item[variable]
            lines.append(
                f"| {variable} | {stats['mean']:.4f} | {stats['median']:.4f} | "
                f"{stats['min']:.4f} | {stats['max']:.4f} |"
            )
        lines.append("")
    lines += [
        "### Spearman（有符号迁移结果 +1/0/−1；仅探索性，无 p 值）",
        "",
        "| 变量 | Spearman rho |",
        "| --- | --- |",
    ]
    for variable in SPEARMAN_VARIABLES:
        lines.append(f"| {variable} | {associations[variable]:.4f} |")
    lines += [
        "",
        "## 6. 可能解释的逐项审计",
        "",
    ]
    for name, audit in report["explanation_audit"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- 判定：`{audit['verdict']}`；已确立事实：`{audit['is_established_fact']}`")
        for key, value in audit.items():
            if key in {"verdict", "is_established_fact", "note"}:
                continue
            lines.append(f"- {key}：`{value}`")
        lines.append(f"- 说明：{audit['note']}")
        lines.append("")
    lines += [
        "## 7. 可结论 / 不可结论",
        "",
        "**可以结论：**",
    ]
    for item in report["can_conclude"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("**不能结论：**")
    for item in report["cannot_conclude"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## 8. 机器可读决策边界",
        "",
        "```json",
        json.dumps(boundary, ensure_ascii=False, indent=2),
        "```",
        "",
        "禁止据此报告声称：" + "、".join(report["forbidden_claims"]) + "。",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--oracle", type=Path, default=ORACLE_JSONL)
    parser.add_argument("--summary", type=Path, default=SUMMARY_JSON)
    parser.add_argument("--control", type=Path, default=CONTROL_JSONL)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--no-write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = PostmortemPaths(
        config=args.config,
        manifest=args.manifest,
        oracle=args.oracle,
        summary=args.summary,
        control=args.control,
    )
    try:
        report = build_report(paths)
    except (PostmortemInputError, OSError) as error:
        print(f"FAIL_CLOSED: {error}", file=sys.stderr)
        return 2

    if not args.no_write:
        report_dir = args.report_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / REPORT_JSON.name).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (report_dir / REPORT_MD.name).write_text(
            render_markdown(report), encoding="utf-8"
        )
        rows = csv_rows(report)
        with (report_dir / REPORT_CSV.name).open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(CSV_FIELDS))
            writer.writeheader()
            writer.writerows(rows)

    print(
        json.dumps(
            {
                "status": report["status"],
                "frozen_status": report["frozen_status"],
                "regression_ids": [
                    row["sample_id"] for row in report["regressions"]
                ],
                "unrepaired_ids": [
                    row["sample_id"] for row in report["unrepaired_challenges"]
                ],
                "recommended_next_mechanism_class": report[
                    "decision_boundary"
                ]["recommended_next_mechanism_class"],
                "gpu_runs": report["gpu_runs"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
