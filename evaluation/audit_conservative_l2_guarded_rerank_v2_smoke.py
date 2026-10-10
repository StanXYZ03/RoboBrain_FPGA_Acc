#!/usr/bin/env python3
"""CPU-only, read-only contract audit of the v2 six-cell GPU smoke.

This audit is deliberately separate from the full2638 development summariser.
It re-verifies, without loading a model, using a GPU, touching the network or
installing anything, that the already-produced raw smoke file

    artifacts/token_selector/2026-10-06-conservative-l2-guarded-reranker-v2-full2638/
        conservative_l2_guarded_rerank_v2_k96_smoke.jsonl

satisfies the frozen v2 preregistration contract.  The raw file was produced
earlier by the host GPU runner; this task performs **zero** GPU runs
(``gpu_runs=0``) and only audits one previously executed smoke
(``audited_gpu_runs=1``, ``audited_samples=6``).

Contract items (every one must pass for ``CONTRACT_SMOKE_PASS``)
----------------------------------------------------------------
1. Exactly six unique deterministic smoke ids, one per canonical
   ``source x task`` cell, recomputed from the frozen full manifest.
2. Every record SHA, the manifest / config / checkpoint / source pins and every
   leakage flag.
3. Identity-bypass legality for ``G <= 96`` and exact ``K=96`` / 87 real + 9
   context for every compressed sample.
4. Frozen anchors, same-region one-for-one swaps, per-region selected-real
   counts and the nonempty context invariants.
5. Every swap satisfies
   ``legal_gain == semantic_gain - l2_loss == utility_added - utility_removed > 0``,
   ``swap_count <= 16`` and the deterministic metadata lengths/hashes.
6. Every output parses and the per-sample answers are compared descriptively
   with the dense BF16, corrected integer-L2 and prior prompt-fusion full2638
   references.

This is a **contract** smoke only.  It is never an accuracy estimate, never a
steady-state latency measurement and never an FPGA latency / acceleration
claim.  4/6 (or any other count) is not a quality gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from evaluation.build_cv_bench_pruning_report import (  # noqa: E402
    answer_letter,
    parse_choice,
)
from evaluation.conservative_l2_guarded_reranker import (  # noqa: E402
    CONTEXT_POOLING_BASIS,
    LEGAL_GAIN_FORMULA,
    LOCAL_CANDIDATE_CAP,
    MAX_SWAPS,
    MECHANISM,
    REGION_GRID,
    UTILITY_FORMULA,
)
from evaluation.hardware_regular_b1_corrected_l2 import array_sha256  # noqa: E402
from evaluation.hardware_regular_token_selector import (  # noqa: E402
    MERGE_UNIT,
    SOURCE_CONTEXT,
    SOURCE_REAL,
    build_regions,
    region_id_by_group,
)
from evaluation.run_hf_conservative_l2_residual_rerank import (  # noqa: E402
    SMOKE_CELLS,
    select_smoke_samples,
)
from evaluation.run_hf_hardware_regular_b1_corrected_l2 import (  # noqa: E402
    record_sha256,
)
from evaluation.summarize_conservative_l2_guarded_rerank_full2638 import (  # noqa: E402
    EXPECTED_CONTEXT,
    EXPECTED_REAL,
    FULL_SAMPLE_COUNT,
    LEAKAGE_FLAGS,
)

ROOT = Path(__file__).resolve().parents[1]

RAW_SMOKE = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-06-conservative-l2-guarded-reranker-v2-full2638"
    / "conservative_l2_guarded_rerank_v2_k96_smoke.jsonl"
)
PREREGISTERED_CONFIG = (
    ROOT
    / "configs"
    / "2026-10-06-conservative-l2-guarded-reranker-v2-full2638-development.preregistered.json"
)
FULL_MANIFEST = ROOT / "datasets" / "cv_bench_full2638.jsonl"
DENSE_FULL_JSONL = (
    ROOT
    / "evaluation"
    / "results"
    / "cv_bench_l2_0875_full2638_20260916"
    / "shallow_r1000.jsonl"
)
CORRECTED_L2_INTEGER_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-b1-corrected-l2-prompt-ablation-full2638"
    / "corrected_l2_integer_k96_full2638.jsonl"
)
PROMPT_PLUS_CORRECTED_L2_INTEGER_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-b1-corrected-l2-prompt-ablation-full2638"
    / "prompt_plus_corrected_l2_integer_k96_full2638.jsonl"
)
REPORT_JSON = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-06-conservative-l2-guarded-reranker-v2-smoke.json"
)
REPORT_MD = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-06-conservative-l2-guarded-reranker-v2-smoke.md"
)

#: Claimed SHA256 of the audited raw smoke (verified, not trusted).
EXPECTED_RAW_SHA256 = (
    "e7648d9b3d015d434b49aa8b030ba9148d294bdd837f1ce9a68d769083a15916"
)
#: Frozen preregistration config SHA256 (also stated in the Chinese prereg report).
EXPECTED_CONFIG_SHA256 = (
    "7499b6ea00939c58506e73b496c1c62bed08162dc8d6d535229f2ffa703e74eb"
)
#: Expected observed shape, cross-checked but never substituted for recomputation.
EXPECTED_SMOKE_IDS = ("286", "374", "1039", "1392", "1467", "2566")
EXPECTED_BYPASS_IDS = ("286",)
EXPECTED_COMPRESSED_SWAP_COUNTS = {"374": 16, "1039": 16, "1392": 16, "1467": 16, "2566": 16}
EXPECTED_CORRECT_COUNTS = {
    "v2": 4,
    "dense": 4,
    "corrected_l2_integer": 4,
    "prompt_plus_corrected_l2_integer": 3,
}
EXPECTED_CONFIG_STATUS = "PREREGISTERED_NOT_EXECUTED"

DECISION_PASS = "CONTRACT_SMOKE_PASS"
DECISION_FAIL = "CONTRACT_SMOKE_FAIL"
NEXT_DECISION_PERMITTED = "PERMITTED_ONLY_SINGLE_FROZEN_FULL2638_DEVELOPMENT_RUN"
NEXT_DECISION_STOP = "STOP_NO_FULL2638_RUN"

ITEM_1 = "1_exactly_six_unique_deterministic_smoke_ids_one_per_cell"
ITEM_2 = "2_record_sha_manifest_config_checkpoint_source_pins_and_leakage"
ITEM_3 = "3_identity_bypass_legality_and_exact_k96_87_real_9_context"
ITEM_4 = "4_frozen_anchors_same_region_one_for_one_counts_nonempty_context"
ITEM_5 = "5_swap_legal_gain_relation_count_metadata_lengths_and_hashes"
ITEM_6 = "6_outputs_parse_and_descriptive_reference_comparison"

CONTRACT_ITEMS = (ITEM_1, ITEM_2, ITEM_3, ITEM_4, ITEM_5, ITEM_6)

#: The eight swap-audit arrays that must all share ``swap_count``.
SWAP_ARRAY_FIELDS = (
    "removed_group_indices",
    "added_group_indices",
    "swap_region_ids",
    "swap_semantic_gain_q15",
    "swap_l2_loss_q15",
    "swap_legal_gain_q16",
    "swap_utility_removed_q16",
    "swap_utility_added_q16",
)

#: SHA256 audit fields that must be 64-hex on the compressed path and null on
#: the identity bypass.
AUDIT_HASH_FIELDS = (
    "utility_q16_sha256",
    "semantic_norm_q15_sha256",
    "l2_norm_q15_sha256",
    "l2_combined_sha256",
    "baseline_selected_real_sha256",
    "reranked_selected_real_sha256",
)

#: Per-record arrays whose SHA256 is recomputable from the record itself.
RECOMPUTABLE_RECORD_HASH_FIELDS = (
    ("selected_group_indices", "selected_group_indices_sha256", np.int64),
    ("selected_patch_indices", "selected_patch_indices_sha256", np.int64),
    ("source_type", "source_type_sha256", np.int64),
    ("region_id", "region_id_sha256", np.int64),
    ("valid", "valid_sha256", bool),
)


class AuditError(RuntimeError):
    """A hard audit failure (missing input or unusable frozen artifact)."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AuditError(f"missing required input: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AuditError(f"{path} line {line_number} is not JSON: {exc}") from exc
    return rows


def unique_index(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in index:
            raise AuditError(f"{label} contains duplicate sample_id {sample_id!r}")
        index[sample_id] = row
    return index


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


class CheckRecorder:
    """Collects named boolean checks grouped under the six contract items."""

    def __init__(self) -> None:
        self.checks: dict[str, dict[str, Any]] = {}

    def add(self, item: str, name: str, passed: bool, detail: Any = "") -> None:
        key = f"{item}::{name}"
        if key in self.checks:
            raise AuditError(f"duplicate audit check {key!r}")
        self.checks[key] = {
            "item": item,
            "name": name,
            "passed": bool(passed),
            "detail": detail,
        }

    def group(self) -> dict[str, Any]:
        grouped: dict[str, Any] = {}
        for item in CONTRACT_ITEMS:
            entries = [c for c in self.checks.values() if c["item"] == item]
            grouped[item] = {
                "passed": all(c["passed"] for c in entries) if entries else False,
                "checks": {
                    c["name"]: {"passed": c["passed"], "detail": c["detail"]}
                    for c in entries
                },
            }
        return grouped

    @property
    def all_passed(self) -> bool:
        return bool(self.checks) and all(c["passed"] for c in self.checks.values())

    @property
    def failed_keys(self) -> list[str]:
        return sorted(key for key, c in self.checks.items() if not c["passed"])


@dataclass(frozen=True)
class InputPaths:
    raw_smoke: Path = RAW_SMOKE
    preregistered: Path = PREREGISTERED_CONFIG
    full_manifest: Path = FULL_MANIFEST
    dense_full_jsonl: Path = DENSE_FULL_JSONL
    corrected_l2_integer_jsonl: Path = CORRECTED_L2_INTEGER_JSONL
    prompt_plus_corrected_l2_integer_jsonl: Path = (
        PROMPT_PLUS_CORRECTED_L2_INTEGER_JSONL
    )
    report_json: Path = REPORT_JSON
    report_md: Path = REPORT_MD


DEFAULT_PATHS = InputPaths()


# ---------------------------------------------------------------------------
# Contract item 1: deterministic smoke ids
# ---------------------------------------------------------------------------
def _expected_smoke_by_cell(
    manifest_rows: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    chosen = select_smoke_samples(manifest_rows)
    return [
        {
            "source": str(sample.get("source")),
            "task": str(sample.get("task")),
            "sample_id": str(sample["sample_id"]),
        }
        for sample in chosen
    ]


def _audit_smoke_ids(
    recorder: CheckRecorder,
    *,
    observed_ids: Sequence[str],
    manifest_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    expected = _expected_smoke_by_cell(manifest_rows)
    expected_again = _expected_smoke_by_cell(manifest_rows)
    expected_ids = [entry["sample_id"] for entry in expected]

    recorder.add(
        ITEM_1,
        "manifest_unique_2638",
        len(manifest_rows) == FULL_SAMPLE_COUNT
        and len(unique_index(manifest_rows, label="manifest")) == FULL_SAMPLE_COUNT,
        {"rows": len(manifest_rows)},
    )
    recorder.add(
        ITEM_1,
        "recomputation_is_deterministic",
        expected == expected_again,
        expected_ids,
    )
    recorder.add(
        ITEM_1,
        "six_cells_present",
        len(expected) == len(SMOKE_CELLS),
        {"cells": [f"{s}::{t}" for s, t in SMOKE_CELLS], "chosen": len(expected)},
    )
    recorder.add(
        ITEM_1,
        "observed_equals_recomputed_ids",
        list(observed_ids) == expected_ids,
        {"observed": list(observed_ids), "recomputed": expected_ids},
    )
    recorder.add(
        ITEM_1,
        "six_unique_ids",
        len(observed_ids) == 6 and len(set(observed_ids)) == 6,
        list(observed_ids),
    )
    # Each chosen id must be the deterministic sha256(sample_id) winner of its
    # own cell, i.e. one sample per canonical cell and no cross-cell borrowing.
    winners_ok = True
    winner_detail: list[dict[str, str]] = []
    for source, task in SMOKE_CELLS:
        cell = [
            sample
            for sample in manifest_rows
            if str(sample.get("source")) == source and str(sample.get("task")) == task
        ]
        if not cell:
            winners_ok = False
            winner_detail.append({"cell": f"{source}::{task}", "winner": "MISSING"})
            continue
        winner = min(
            cell,
            key=lambda sample: (
                hashlib.sha256(str(sample["sample_id"]).encode("utf-8")).hexdigest(),
                str(sample["sample_id"]),
            ),
        )
        winner_id = str(winner["sample_id"])
        winner_detail.append({"cell": f"{source}::{task}", "winner": winner_id})
        if winner_id not in observed_ids:
            winners_ok = False
    recorder.add(
        ITEM_1,
        "one_deterministic_winner_per_cell_observed",
        winners_ok,
        winner_detail,
    )
    return {
        "expected_sample_ids": expected_ids,
        "expected_by_cell": expected,
        "expected_sample_ids_constant": list(EXPECTED_SMOKE_IDS),
    }


# ---------------------------------------------------------------------------
# Contract item 2: record SHA, pins and leakage
# ---------------------------------------------------------------------------
def _config_pin_table(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for name, entry in config.get("inputs_sha256", {}).items():
        table[name] = {
            "path": str(entry.get("path")),
            "sha256": entry.get("sha256"),
        }
    return table


def _expected_checkpoint_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    pins = config.get("checkpoint_pins", {})
    return {
        "checkpoint_sha256": pins.get("projection_npz", {}).get("sha256"),
        "wv_sha256": pins.get("wv_int8", {}).get("sha256"),
        "wv_scale_sha256": pins.get("wv_per_tensor_scale", {}).get("sha256"),
        "deployment_seed": pins.get("deployment_seed"),
        "projection_npz": pins.get("projection_npz", {}).get("path"),
    }


def _audit_record_pins(
    recorder: CheckRecorder,
    *,
    records: Sequence[Mapping[str, Any]],
    manifest_index: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    manifest_sha256: str,
) -> list[dict[str, Any]]:
    checkpoint = _expected_checkpoint_identity(config)
    per_record: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        manifest_row = manifest_index[sample_id]
        facts: dict[str, Any] = {"sample_id": sample_id}

        facts["arm"] = (
            str(record.get("arm")) == MECHANISM
            and str(record.get("mechanism")) == MECHANISM
        )
        facts["run_mode_smoke"] = str(record.get("run_mode")) == "smoke"
        facts["record_sha256"] = (
            str(record.get("record_sha256")) == record_sha256(dict(record))
        )
        facts["manifest_sha256"] = str(record.get("manifest_sha256")) == manifest_sha256
        facts["task_matches_manifest"] = str(record.get("task")) == str(
            manifest_row.get("task")
        )
        facts["checkpoint_identity"] = all(
            (
                str(record.get("checkpoint_sha256")) == str(checkpoint["checkpoint_sha256"]),
                str(record.get("wv_sha256")) == str(checkpoint["wv_sha256"]),
                str(record.get("wv_scale_sha256")) == str(checkpoint["wv_scale_sha256"]),
                int(record.get("deployment_seed", -1)) == int(checkpoint["deployment_seed"]),
                str(record.get("projection_npz")) == str(checkpoint["projection_npz"]),
            )
        )
        run = record.get("run") or {}
        facts["run_metadata_identity"] = all(
            (
                str(run.get("arm")) == MECHANISM,
                str(run.get("mechanism")) == MECHANISM,
                str(run.get("run_mode")) == "smoke",
                str(run.get("manifest_sha256")) == manifest_sha256,
                str(run.get("checkpoint_sha256")) == str(checkpoint["checkpoint_sha256"]),
                str(run.get("wv_sha256")) == str(checkpoint["wv_sha256"]),
                str(run.get("wv_scale_sha256")) == str(checkpoint["wv_scale_sha256"]),
                int(run.get("deployment_seed", -1)) == int(checkpoint["deployment_seed"]),
            )
        )
        facts["leakage_flags_false"] = all(
            not bool(record.get(flag)) for flag in LEAKAGE_FLAGS
        ) and all(not bool(run.get(flag)) for flag in LEAKAGE_FLAGS)
        facts["no_acceleration_claim"] = (
            record.get("selector_software_is_acceleration_claim") is False
            and record.get("cpu_selector_roundtrip_is_acceleration_claim") is False
            and run.get("cpu_selector_roundtrip_is_acceleration_claim") is False
        )
        facts["frozen_flags"] = (
            record.get("frozen_anchors") is True
            and record.get("same_region_one_for_one_swaps") is True
        )
        facts["fixed_contract_fields"] = all(
            (
                int(record.get("requested_budget", -1)) == 96,
                int(record.get("region_grid", -1)) == REGION_GRID,
                int(record.get("local_candidate_cap", -1)) == LOCAL_CANDIDATE_CAP,
                int(record.get("max_swaps", -1)) == MAX_SWAPS,
                str(record.get("utility_formula")) == UTILITY_FORMULA,
                str(record.get("legal_gain_formula")) == LEGAL_GAIN_FORMULA,
                str(record.get("context_pooling_basis")) == CONTEXT_POOLING_BASIS,
                int(record.get("prune_after_layer", -1)) == 0,
                str(record.get("cutpoint")) == "B1_after_vision_block_0",
                record.get("do_sample") is False,
                str(record.get("decoding")) == "greedy_argmax",
                str(record.get("dtype")) == "bfloat16",
                str(record.get("attention_implementation")) == "sdpa",
                int(record.get("max_new_tokens", -1)) == 16,
                str(record.get("score_mode")) == "integer",
                str(record.get("integration_score_mode")) == "integer",
            )
        )
        facts["deprecated_alias"] = (
            record.get("elapsed_seconds") == record.get("model_inference_seconds")
            and record.get("timed_stage_seconds") == record.get("model_inference_seconds")
            and record.get("vision_encoder_seconds") == record.get("vision_total_seconds")
        )
        per_record.append(facts)

    for key, name in (
        ("arm", "arm_mechanism_identity"),
        ("run_mode_smoke", "run_mode_is_smoke"),
        ("record_sha256", "record_sha256_recomputes"),
        ("manifest_sha256", "manifest_sha256_pin"),
        ("task_matches_manifest", "record_task_matches_manifest"),
        ("checkpoint_identity", "checkpoint_pins"),
        ("run_metadata_identity", "run_metadata_pins"),
        ("leakage_flags_false", "leakage_flags_all_false"),
        ("no_acceleration_claim", "no_acceleration_claim"),
        ("frozen_flags", "frozen_flag_booleans"),
        ("fixed_contract_fields", "fixed_budget_grid_formula_fields"),
        ("deprecated_alias", "timing_alias_identity"),
    ):
        recorder.add(
            ITEM_2,
            name,
            all(fact[key] for fact in per_record),
            [fact["sample_id"] for fact in per_record if not fact[key]],
        )
    return per_record


def _audit_frozen_inputs(
    recorder: CheckRecorder,
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    references: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    manifest_pin = (
        config.get("dataset_pins", {}).get("full_manifest", {}).get("sha256")
    )
    config_only_pin = (
        config.get("inputs_sha256", {}).get("full_manifest", {}).get("sha256")
    )
    recorder.add(
        ITEM_2,
        "config_sha256_matches_frozen_prereg",
        config_sha256 == EXPECTED_CONFIG_SHA256,
        {"observed": config_sha256, "expected": EXPECTED_CONFIG_SHA256},
    )
    recorder.add(
        ITEM_2,
        "config_status_still_preregistered_not_executed",
        str(config.get("status")) == EXPECTED_CONFIG_STATUS
        and config.get("executed") is False,
        {"status": config.get("status"), "executed": config.get("executed")},
    )
    full2638_output = (
        config.get("output_paths", {})
        .get(MECHANISM, {})
        .get("path", "")
    )
    recorder.add(
        ITEM_2,
        "full2638_arm_output_absent",
        full2638_output != "" and not (ROOT / full2638_output).exists(),
        full2638_output,
    )
    recorder.add(
        ITEM_2,
        "manifest_pin_consistency",
        manifest_sha256 == manifest_pin == config_only_pin,
        {
            "observed": manifest_sha256,
            "dataset_pins": manifest_pin,
            "inputs_sha256": config_only_pin,
        },
    )
    for name, reference in references.items():
        expected = reference["expected_sha256"]
        observed = reference["sha256"]
        recorder.add(
            ITEM_2,
            f"reference_pin_{name}",
            observed == expected and reference["unique_ids"] == FULL_SAMPLE_COUNT
            and reference["row_count"] == FULL_SAMPLE_COUNT,
            {
                "observed": observed,
                "expected": expected,
                "rows": reference["row_count"],
                "unique_ids": reference["unique_ids"],
            },
        )
    # Every non-null config pin must match on disk (v1/v2 source, config, raw).
    pin_table = _config_pin_table(config)
    verified: list[str] = []
    mismatched: list[str] = []
    not_yet_available: list[str] = []
    for name, entry in pin_table.items():
        if entry["sha256"] is None:
            not_yet_available.append(name)
            continue
        path = ROOT / entry["path"]
        if path.is_file() and sha256_file(path) == entry["sha256"]:
            verified.append(name)
        else:
            mismatched.append(name)
    recorder.add(
        ITEM_2,
        "all_nonnull_config_pins_byte_identical",
        not mismatched,
        {
            "verified_count": len(verified),
            "mismatched": mismatched,
            "not_yet_available": not_yet_available,
        },
    )
    return {
        "verified_pins": verified,
        "mismatched_pins": mismatched,
        "null_pins": not_yet_available,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
    }


# ---------------------------------------------------------------------------
# Contract items 3/4/5: record layout, anchors, swaps
# ---------------------------------------------------------------------------
def _recommended_region_lookup(grid: Sequence[int]) -> np.ndarray:
    grid_h = int(grid[1]) // 2
    grid_w = int(grid[2]) // 2
    return region_id_by_group(grid_h, grid_w, REGION_GRID)


def _audit_record_layout(
    recorder: CheckRecorder,
    *,
    records: Sequence[Mapping[str, Any]],
    observed_ids: Sequence[str],
) -> dict[str, Any]:
    bypass_ids: list[str] = []
    compressed_ids: list[str] = []
    per_sample: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        grid = [int(value) for value in record["image_grid_thw"]]
        full_patch = int(record["full_patch_tokens"])
        full_llm = int(record["full_llm_visual_tokens"])
        output_count = int(record["output_group_count"])
        real_count = int(record["real_group_count"])
        context_count = int(record["context_group_count"])
        bypass = bool(record.get("identity_bypass"))
        selected = [int(value) for value in record["selected_group_indices"]]
        grid_ok = (
            len(grid) == 3
            and grid[0] == 1
            and grid[1] * grid[2] == full_patch
            and full_llm * MERGE_UNIT == full_patch
        )
        selected_ok = (
            len(selected) == output_count
            and len(set(selected)) == len(selected)
            and all(0 <= value < full_llm for value in selected)
            and all(later > earlier for earlier, later in zip(selected, selected[1:]))
        )
        source_type = [int(value) for value in record["source_type"]]
        real_slots = [value for value in source_type if value == SOURCE_REAL]
        context_slots = [value for value in source_type if value == SOURCE_CONTEXT]
        expected_context = _expected_context_count(grid)
        if bypass:
            branch_ok = (
                full_llm <= 96
                and output_count == full_llm
                and real_count == full_llm
                and context_count == 0
                and selected == list(range(full_llm))
                and all(value == SOURCE_REAL for value in source_type)
            )
            bypass_ids.append(sample_id)
        else:
            branch_ok = (
                full_llm > 96
                and output_count == 96
                and real_count == EXPECTED_REAL
                and context_count == EXPECTED_CONTEXT
                and context_count == expected_context
                and len(real_slots) == EXPECTED_REAL
                and len(context_slots) == EXPECTED_CONTEXT
            )
            compressed_ids.append(sample_id)
        per_sample.append(
            {
                "sample_id": sample_id,
                "identity_bypass": bypass,
                "full_llm_visual_tokens": full_llm,
                "output_group_count": output_count,
                "real_group_count": real_count,
                "context_group_count": context_count,
                "expected_context_count": expected_context,
                "grid_ok": grid_ok,
                "selected_ok": selected_ok,
                "branch_ok": branch_ok,
            }
        )
    recorder.add(
        ITEM_3,
        "grid_and_merge_unit",
        all(row["grid_ok"] for row in per_sample),
        [row["sample_id"] for row in per_sample if not row["grid_ok"]],
    )
    recorder.add(
        ITEM_3,
        "selected_indices_strictly_ascending_unique_in_range",
        all(row["selected_ok"] for row in per_sample),
        [row["sample_id"] for row in per_sample if not row["selected_ok"]],
    )
    recorder.add(
        ITEM_3,
        "bypass_or_exact_k96_87_9",
        all(row["branch_ok"] for row in per_sample),
        [row["sample_id"] for row in per_sample if not row["branch_ok"]],
    )
    recorder.add(
        ITEM_3,
        "observed_bypass_shape_one_bypass",
        bypass_ids == list(EXPECTED_BYPASS_IDS) and len(compressed_ids) == 5,
        {"bypass": bypass_ids, "compressed": compressed_ids},
    )
    return {
        "bypass_ids": bypass_ids,
        "compressed_ids": compressed_ids,
        "per_sample": per_sample,
    }


def _expected_context_count(grid: Sequence[int]) -> int:
    grid_h = int(grid[1]) // 2
    grid_w = int(grid[2]) // 2
    return len(
        [
            region
            for region in build_regions(grid_h, grid_w, REGION_GRID)
            if region.members and len(region.members) >= 2
        ]
    )


def _audit_record_swaps(
    recorder: CheckRecorder,
    *,
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    per_sample: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        bypass = bool(record.get("identity_bypass"))
        swap_count = int(record.get("swap_count", -1))
        grid = [int(value) for value in record["image_grid_thw"]]
        region_lookup = _recommended_region_lookup(grid)
        selected = [int(value) for value in record["selected_group_indices"]]
        source_type = [int(value) for value in record["source_type"]]
        baseline_real = [int(value) for value in record["baseline_selected_real"]]
        reranked_real = [int(value) for value in record["reranked_selected_real"]]
        anchors = [int(value) for value in record["baseline_anchors"]]
        removed = [int(value) for value in record["removed_group_indices"]]
        added = [int(value) for value in record["added_group_indices"]]
        regions = [int(value) for value in record["swap_region_ids"]]
        sem_gain = [int(value) for value in record["swap_semantic_gain_q15"]]
        l2_loss = [int(value) for value in record["swap_l2_loss_q15"]]
        legal_gain = [int(value) for value in record["swap_legal_gain_q16"]]
        util_removed = [int(value) for value in record["swap_utility_removed_q16"]]
        util_added = [int(value) for value in record["swap_utility_added_q16"]]

        # --- item 5: lengths / count / legality ---
        lengths = {len(record[field]) for field in SWAP_ARRAY_FIELDS}
        lengths_ok = lengths == {swap_count}
        recorder.add(
            ITEM_5,
            f"swap_array_lengths[{sample_id}]",
            lengths_ok,
            {"swap_count": swap_count, "lengths": sorted(lengths)},
        )
        recorder.add(
            ITEM_5,
            f"swap_count_within_S[{sample_id}]",
            0 <= swap_count <= MAX_SWAPS,
            swap_count,
        )
        legal_ok = all(
            legal_gain[index] == sem_gain[index] - l2_loss[index]
            and legal_gain[index] == util_added[index] - util_removed[index]
            and legal_gain[index] > 0
            for index in range(len(legal_gain))
        )
        recorder.add(
            ITEM_5,
            f"legal_gain_identity_and_positive[{sample_id}]",
            legal_ok,
            {
                "semantic_gain": sem_gain,
                "l2_loss": l2_loss,
                "legal_gain": legal_gain,
            },
        )
        no_reuse = (
            len(set(removed)) == len(removed)
            and len(set(added)) == len(added)
            and not (set(removed) & set(added))
        )
        recorder.add(
            ITEM_5,
            f"no_reused_or_overlapping_groups[{sample_id}]",
            no_reuse,
            {"removed": removed, "added": added},
        )

        # --- item 4: anchors / same region / counts / context ---
        anchor_regions = sorted({int(region_lookup[group]) for group in anchors})
        expected_regions = sorted(
            {
                int(region_lookup[group])
                for group in range(int(record["full_llm_visual_tokens"]))
            }
        )
        anchors_ok = (
            len(anchors) >= 1
            and set(anchors).issubset(set(baseline_real))
            and set(anchors).issubset(set(reranked_real))
            and not (set(anchors) & set(removed))
            and anchor_regions == expected_regions
        )
        recorder.add(
            ITEM_4,
            f"anchors_frozen_and_one_per_region[{sample_id}]",
            anchors_ok if not bypass else (anchors == []),
            {
                "anchors": anchors,
                "anchor_regions": anchor_regions,
                "expected_regions": expected_regions,
            },
        )
        same_region_ok = all(
            int(region_lookup[removed[index]])
            == int(regions[index])
            == int(region_lookup[added[index]])
            for index in range(len(removed))
        )
        recorder.add(
            ITEM_4,
            f"same_region_one_for_one[{sample_id}]",
            same_region_ok,
            {"regions": regions},
        )
        removed_subset = set(removed).issubset(set(baseline_real))
        added_disjoint = not (set(added) & set(baseline_real))
        reconstruct = (
            (set(baseline_real) - set(removed)) | set(added)
        ) == set(reranked_real)
        recorder.add(
            ITEM_4,
            f"swap_audit_reconstructs_reranked_real[{sample_id}]",
            removed_subset and added_disjoint and reconstruct,
            {
                "removed_subset_baseline": removed_subset,
                "added_disjoint_baseline": added_disjoint,
                "reconstructs": reconstruct,
            },
        )
        if bypass:
            before = [int(value) for value in record["per_region_selected_counts_before"]]
            after = [int(value) for value in record["per_region_selected_counts_after"]]
            counts_ok = before == after
            context_ok = int(record["context_group_count"]) == 0
            context_detail = {"bypass": True}
        else:
            before = [
                sum(1 for group in baseline_real if int(region_lookup[group]) == region)
                for region in expected_regions
            ]
            after = [
                sum(1 for group in reranked_real if int(region_lookup[group]) == region)
                for region in expected_regions
            ]
            counts_ok = (
                before == after
                and before
                == [int(value) for value in record["per_region_selected_counts_before"]]
                and after
                == [int(value) for value in record["per_region_selected_counts_after"]]
            )
            real_slots = set(
                group
                for group, source in zip(selected, source_type)
                if source == SOURCE_REAL
            )
            context_groups = [
                group
                for group, source in zip(selected, source_type)
                if source == SOURCE_CONTEXT
            ]
            grid_h = int(grid[1]) // 2
            grid_w = int(grid[2]) // 2
            context_regions = [int(region_lookup[group]) for group in context_groups]
            nonempty = True
            for group in context_groups:
                region_id = int(region_lookup[group])
                members = [
                    candidate
                    for candidate in range(grid_h * grid_w)
                    if int(region_lookup[candidate]) == region_id
                ]
                if not [member for member in members if member not in real_slots]:
                    nonempty = False
            context_ok = (
                nonempty
                and sorted(context_regions) == expected_regions
                and set(context_groups) == (set(selected) - real_slots)
                and real_slots == set(reranked_real)
            )
            context_detail = {
                "context_groups": context_groups,
                "context_regions": sorted(context_regions),
                "expected_regions": expected_regions,
                "nonempty": nonempty,
                "real_slots_equal_reranked": real_slots == set(reranked_real),
            }
        recorder.add(
            ITEM_4,
            f"per_region_real_counts_preserved[{sample_id}]",
            counts_ok,
            {"before": before, "after": after},
        )
        recorder.add(
            ITEM_4,
            f"nonempty_context_invariant[{sample_id}]",
            context_ok,
            context_detail,
        )

        # --- item 5: recomputable hashes ---
        for array_field, hash_field, dtype in RECOMPUTABLE_RECORD_HASH_FIELDS:
            values = record[array_field]
            observed_hash = record.get(hash_field)
            recomputed = array_sha256(np.asarray(values, dtype=dtype))
            recorder.add(
                ITEM_5,
                f"{hash_field}[{sample_id}]",
                str(observed_hash) == recomputed,
                {"observed": observed_hash, "recomputed": recomputed},
            )
        # selected_patch_indices must be exactly group*4 + patch offset.
        expected_patches: list[int] = []
        for group in selected:
            expected_patches.extend(
                group * MERGE_UNIT + offset for offset in range(MERGE_UNIT)
            )
        recorder.add(
            ITEM_5,
            f"selected_patch_indices_derive_from_groups[{sample_id}]",
            [int(value) for value in record["selected_patch_indices"]]
            == expected_patches,
            None,
        )
        # baseline/reranked real hashes are recomputable on the compressed path.
        baseline_hash = record.get("baseline_selected_real_sha256")
        reranked_hash = record.get("reranked_selected_real_sha256")
        if bypass:
            recorder.add(
                ITEM_5,
                f"bypass_audit_arrays_empty_and_hashes_null[{sample_id}]",
                baseline_real == []
                and reranked_real == []
                and anchors == []
                and removed == []
                and added == []
                and all(record.get(field) is None for field in AUDIT_HASH_FIELDS),
                None,
            )
        else:
            recorder.add(
                ITEM_5,
                f"baseline_selected_real_sha256[{sample_id}]",
                str(baseline_hash)
                == array_sha256(np.asarray(baseline_real, dtype=np.int64)),
                {"observed": baseline_hash},
            )
            recorder.add(
                ITEM_5,
                f"reranked_selected_real_sha256[{sample_id}]",
                str(reranked_hash)
                == array_sha256(np.asarray(reranked_real, dtype=np.int64)),
                {"observed": reranked_hash},
            )
            recorder.add(
                ITEM_5,
                f"audit_array_hashes_are_hex64[{sample_id}]",
                all(_is_hex64(record.get(field)) for field in AUDIT_HASH_FIELDS),
                {field: _is_hex64(record.get(field)) for field in AUDIT_HASH_FIELDS},
            )
        # combined_scores_sha256 / score_vector_length consistency.
        combined_hash = record.get("combined_scores_sha256")
        score_length = record.get("score_vector_length")
        if bypass:
            combined_ok = combined_hash is None and score_length is None
        else:
            combined_ok = _is_hex64(combined_hash) and int(score_length) == int(
                record["full_llm_visual_tokens"]
            )
        recorder.add(
            ITEM_5,
            f"combined_scores_hash_and_length[{sample_id}]",
            combined_ok,
            {"combined_scores_sha256": combined_hash, "score_vector_length": score_length},
        )
        block_shapes = record.get("block_input_shapes")
        recorder.add(
            ITEM_5,
            f"block_input_shapes_metadata[{sample_id}]",
            isinstance(block_shapes, list)
            and len(block_shapes) == 24
            and all(int(value) > 0 for value in block_shapes),
            None,
        )
        per_sample.append(
            {
                "sample_id": sample_id,
                "identity_bypass": bypass,
                "swap_count": swap_count,
                "anchors": anchors,
                "removed_group_indices": removed,
                "added_group_indices": added,
                "swap_region_ids": regions,
                "swap_semantic_gain_q15": sem_gain,
                "swap_l2_loss_q15": l2_loss,
                "swap_legal_gain_q16": legal_gain,
            }
        )
    recorder.add(
        ITEM_5,
        "observed_swap_counts_match_expectation",
        all(
            per_sample[index]["swap_count"]
            == EXPECTED_COMPRESSED_SWAP_COUNTS.get(str(per_sample[index]["sample_id"]), 0)
            for index in range(len(per_sample))
        ),
        {row["sample_id"]: row["swap_count"] for row in per_sample},
    )
    return per_sample


# ---------------------------------------------------------------------------
# Contract item 6: parse + descriptive reference comparison
# ---------------------------------------------------------------------------
def _audit_descriptive_quality(
    recorder: CheckRecorder,
    *,
    records: Sequence[Mapping[str, Any]],
    manifest_index: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    arms = {
        "v2": {str(record["sample_id"]): record for record in records},
        "dense": references["dense"]["index"],
        "corrected_l2_integer": references["corrected_l2_integer"]["index"],
        "prompt_plus_corrected_l2_integer": references[
            "prompt_plus_corrected_l2_integer"
        ]["index"],
    }
    per_sample: dict[str, Any] = {}
    parse_ok_all = True
    for sample_id in [str(record["sample_id"]) for record in records]:
        manifest_row = manifest_index[sample_id]
        choices = list(manifest_row["choices"])
        ground_truth = answer_letter(str(manifest_row["answer"]))
        entry: dict[str, Any] = {"ground_truth": ground_truth}
        for arm, index in arms.items():
            row = index[sample_id]
            parsed = parse_choice(str(row["output_text"]), choices)
            if parsed is None:
                parse_ok_all = False
            entry[arm] = {
                "output_text": str(row["output_text"]),
                "parsed": parsed,
                "correct": parsed == ground_truth if parsed is not None else False,
            }
        per_sample[sample_id] = entry
    recorder.add(
        ITEM_6,
        "every_arm_output_parses",
        parse_ok_all,
        {
            sample_id: {
                arm: per_sample[sample_id][arm]["parsed"] for arm in arms
            }
            for sample_id in per_sample
        },
    )
    counts = {
        arm: sum(1 for entry in per_sample.values() if entry[arm]["correct"])
        for arm in arms
    }
    agreements = {
        f"v2_equals_{arm}": sum(
            1
            for entry in per_sample.values()
            if entry["v2"]["parsed"] == entry[arm]["parsed"]
        )
        for arm in arms
        if arm != "v2"
    }
    recorder.add(
        ITEM_6,
        "descriptive_quality_counts_computed",
        set(counts) == set(EXPECTED_CORRECT_COUNTS),
        counts,
    )
    return {
        "per_sample": per_sample,
        "correct_counts": counts,
        "expected_correct_counts": dict(EXPECTED_CORRECT_COUNTS),
        "counts_match_expected": counts == EXPECTED_CORRECT_COUNTS,
        "parsed_answer_agreement_counts": agreements,
    }


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------
def _load_references(
    paths: InputPaths, config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    baseline = config.get("baseline_references", {})

    def _load(
        name: str, path: Path, pin: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        rows = load_jsonl(path)
        index = unique_index(rows, label=name)
        expected = None
        if pin is not None:
            expected = pin.get("sha256")
        also_inputs = (
            config.get("inputs_sha256", {}).get(name, {}).get("sha256")
        )
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "expected_sha256": expected,
            "inputs_sha256": also_inputs,
            "row_count": len(rows),
            "unique_ids": len(index),
            "index": index,
        }

    return {
        "dense": _load("dense_full_jsonl", paths.dense_full_jsonl, baseline.get("dense_bf16")),
        "corrected_l2_integer": _load(
            "corrected_l2_integer_jsonl",
            paths.corrected_l2_integer_jsonl,
            baseline.get("corrected_l2_integer"),
        ),
        "prompt_plus_corrected_l2_integer": _load(
            "prompt_plus_corrected_l2_integer_jsonl",
            paths.prompt_plus_corrected_l2_integer_jsonl,
            baseline.get("prompt_plus_corrected_l2_integer"),
        ),
    }


def run_audit(paths: InputPaths = DEFAULT_PATHS) -> dict[str, Any]:
    """Run the full read-only smoke contract audit and return the report."""

    if not paths.raw_smoke.is_file():
        raise AuditError(f"missing required input: {paths.raw_smoke}")
    raw_sha256 = sha256_file(paths.raw_smoke)
    raw_rows = load_jsonl(paths.raw_smoke)
    config = json.loads(paths.preregistered.read_text(encoding="utf-8"))
    config_sha256 = sha256_file(paths.preregistered)
    manifest_rows = load_jsonl(paths.full_manifest)
    manifest_sha256 = sha256_file(paths.full_manifest)
    manifest_index = unique_index(manifest_rows, label="full_manifest")
    references = _load_references(paths, config)
    return audit_loaded(
        raw_sha256=raw_sha256,
        raw_rows=raw_rows,
        config=config,
        config_sha256=config_sha256,
        manifest_rows=manifest_rows,
        manifest_sha256=manifest_sha256,
        manifest_index=manifest_index,
        references=references,
        paths=paths,
    )


def audit_loaded(
    *,
    raw_sha256: str,
    raw_rows: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
    config_sha256: str,
    manifest_rows: Sequence[dict[str, Any]],
    manifest_sha256: str,
    manifest_index: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    paths: InputPaths,
) -> dict[str, Any]:
    recorder = CheckRecorder()
    observed_ids = [str(record["sample_id"]) for record in raw_rows]
    recorder.add(
        ITEM_1,
        "raw_smoke_sha256_matches_claim",
        raw_sha256 == EXPECTED_RAW_SHA256,
        {"observed": raw_sha256, "expected": EXPECTED_RAW_SHA256},
    )
    recorder.add(
        ITEM_1,
        "raw_smoke_six_records_no_duplicates",
        len(raw_rows) == 6 and len(set(observed_ids)) == 6,
        observed_ids,
    )
    smoke_ids = _audit_smoke_ids(
        recorder, observed_ids=observed_ids, manifest_rows=manifest_rows
    )
    record_pins = _audit_record_pins(
        recorder,
        records=raw_rows,
        manifest_index=manifest_index,
        config=config,
        manifest_sha256=manifest_sha256,
    )
    frozen = _audit_frozen_inputs(
        recorder,
        config=config,
        config_sha256=config_sha256,
        manifest_path=paths.full_manifest,
        manifest_sha256=manifest_sha256,
        references=references,
    )
    layout = _audit_record_layout(
        recorder, records=raw_rows, observed_ids=observed_ids
    )
    swaps = _audit_record_swaps(recorder, records=raw_rows)
    quality = _audit_descriptive_quality(
        recorder,
        records=raw_rows,
        manifest_index=manifest_index,
        references=references,
    )

    observed_shape = {
        "sample_ids": observed_ids,
        "expected_sample_ids": list(EXPECTED_SMOKE_IDS),
        "sample_ids_match_expected": observed_ids == list(EXPECTED_SMOKE_IDS),
        "bypass_ids": layout["bypass_ids"],
        "compressed_ids": layout["compressed_ids"],
        "swap_counts": {row["sample_id"]: row["swap_count"] for row in swaps},
        "correct_counts": quality["correct_counts"],
    }

    passed = recorder.all_passed
    decision = DECISION_PASS if passed else DECISION_FAIL
    report: dict[str, Any] = {
        "report": "2026-10-06-conservative-l2-guarded-reranker-v2-smoke",
        "date": "2026-10-06",
        "cpu_only": True,
        "read_only": True,
        "audited_gpu_runs": 1,
        "audited_samples": 6,
        "gpu_runs": 0,
        "model_runs": 0,
        "network_used": False,
        "training": False,
        "raw_smoke": {
            "path": str(paths.raw_smoke),
            "sha256_observed": raw_sha256,
            "sha256_claimed": EXPECTED_RAW_SHA256,
            "sha256_match": raw_sha256 == EXPECTED_RAW_SHA256,
            "records": len(raw_rows),
        },
        "config": {
            "path": str(paths.preregistered),
            "sha256_observed": config_sha256,
            "sha256_expected": EXPECTED_CONFIG_SHA256,
            "status": config.get("status"),
            "executed": config.get("executed"),
        },
        "full_manifest": {
            "path": str(paths.full_manifest),
            "sha256": manifest_sha256,
            "records": len(manifest_rows),
            "unique_sample_ids": len(manifest_index),
        },
        "sample_ids": observed_ids,
        "smoke_cell_assignment": smoke_ids,
        "contract_checks": recorder.group(),
        "contract_all_passed": passed,
        "failed_checks": recorder.failed_keys,
        "frozen_inputs": frozen,
        "record_pins": record_pins,
        "layout": layout,
        "swaps": swaps,
        "descriptive_quality": quality,
        "observed_shape": observed_shape,
        "decision": decision,
        "next_decision": NEXT_DECISION_PERMITTED if passed else NEXT_DECISION_STOP,
        "timing_scope": {
            "first_sample_model_inference_seconds": raw_rows[0].get(
                "model_inference_seconds"
            ),
            "first_sample_includes_first_execution_cost": True,
            "any_of_six_is_steady_state_latency_evidence": False,
            "selector_cpu_roundtrip_seconds": [
                record.get("selector_cpu_roundtrip_seconds") for record in raw_rows
            ],
            "selector_cpu_roundtrip_is_fpga_latency": False,
            "selector_cpu_roundtrip_is_acceleration_claim": False,
        },
        "limitations": [
            "六个样本的 4/6（或任一计数）是描述性诊断，不是精度估计，也不是任何门。",
            "首个样本 3.8937 s 含首次执行开销；六个计时都不是稳态时延证据。",
            "软件 selector roundtrip 不是 FPGA 时延，也不是加速主张。",
            "本审计不做论文/外部验证确认；通过仅表示可执行一次冻结的 full2638 发展性运行。",
            "full2638 是被反复观测的发展集，永不构成未触碰确认集。",
            "不得在看到 full2638 后调 v2；v1 永久 PREEXECUTION_DESIGN_STOP。",
            "审计只读，未加载模型权重、未联网、未训练、未安装依赖；gpu_runs=0。",
        ],
    }
    return report


# ---------------------------------------------------------------------------
# Markdown (Chinese)
# ---------------------------------------------------------------------------
def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# L2 守卫整数重排 v2 六单元 GPU 冒烟合同审计")
    lines.append("")
    lines.append(f"- 日期：{report['date']}")
    lines.append(f"- 决策：**`{report['decision']}`**")
    lines.append(f"- 下一决策：`{report['next_decision']}`")
    lines.append(
        f"- 口径：CPU-only、只读；`gpu_runs={report['gpu_runs']}`、"
        f"`audited_gpu_runs={report['audited_gpu_runs']}`、"
        f"`audited_samples={report['audited_samples']}`；未加载模型权重/未联网/未训练/未装依赖。"
    )
    raw = report["raw_smoke"]
    lines.append("")
    lines.append("## 1. 被审计原始冒烟")
    lines.append("")
    lines.append(f"- 路径：`{raw['path']}`")
    lines.append(f"- 记录数：{raw['records']}")
    lines.append(
        f"- SHA256（现场重算）：`{raw['sha256_observed']}`；与声明一致："
        f"`{raw['sha256_match']}`"
    )
    lines.append("")
    lines.append("## 2. 样本与单元")
    lines.append("")
    lines.append(f"- 观测样本 ID（按文件顺序）：{report['sample_ids']}")
    for entry in report["smoke_cell_assignment"]["expected_by_cell"]:
        lines.append(
            f"- `{entry['source']}::{entry['task']}` -> `{entry['sample_id']}`"
        )
    lines.append("")
    lines.append("## 3. 六项合同检查")
    lines.append("")
    lines.append("| 合同项 | 通过 | 子检查数 |")
    lines.append("| --- | --- | --- |")
    for item, group in report["contract_checks"].items():
        lines.append(
            f"| `{item}` | {'✅' if group['passed'] else '❌'} | {len(group['checks'])} |"
        )
    lines.append("")
    lines.append(
        f"- 全部合同检查通过：`{report['contract_all_passed']}`；"
        f"失败项：{report['failed_checks'] or '无'}"
    )
    lines.append("")
    lines.append("## 4. 布局与交换（逐样本）")
    lines.append("")
    lines.append(
        "| 样本 | bypass | G | K | real | context | swaps | anchors |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in report["layout"]["per_sample"]:
        lines.append(
            f"| {row['sample_id']} | {row['identity_bypass']} | "
            f"{row['full_llm_visual_tokens']} | {row['output_group_count']} | "
            f"{row['real_group_count']} | {row['context_group_count']} | "
            f"{[s['swap_count'] for s in report['swaps'] if s['sample_id'] == row['sample_id']][0]} | "
            f"{[s['anchors'] for s in report['swaps'] if s['sample_id'] == row['sample_id']][0]} |"
        )
    lines.append("")
    lines.append("## 5. 描述性逐样本答案对比（不是精度、不是门）")
    lines.append("")
    lines.append(
        "| 样本 | 正确答案 | v2 | dense | 修正整数 L2 | 既往 prompt 融合 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for sample_id, entry in report["descriptive_quality"]["per_sample"].items():
        cells = []
        for arm in (
            "v2",
            "dense",
            "corrected_l2_integer",
            "prompt_plus_corrected_l2_integer",
        ):
            mark = "✅" if entry[arm]["correct"] else "❌"
            cells.append(f"{mark} {entry[arm]['parsed']} ({entry[arm]['output_text']})")
        lines.append(
            f"| {sample_id} | {entry['ground_truth']} | " + " | ".join(cells) + " |"
        )
    counts = report["descriptive_quality"]["correct_counts"]
    lines.append("")
    lines.append(
        "- 描述性正确计数："
        + "；".join(f"{arm} {counts[arm]}/6" for arm in counts)
    )
    lines.append(
        "- **4/6 不是精度估计、不是门**；仅用于描述性对比。"
    )
    lines.append("")
    lines.append("## 6. 冻结输入字节一致性")
    lines.append("")
    frozen = report["frozen_inputs"]
    lines.append(
        f"- 非空 config pin 现场复核通过：{len(frozen['verified_pins'])}；"
        f"不匹配：{frozen['mismatched_pins'] or '无'}；"
        f"待生成（null）：{frozen['null_pins']}"
    )
    lines.append(
        "- 覆盖 v1/v2 核心、集成、runner、汇总器、测试、config、manifest、"
        "dense BF16、修正整数 L2、既往 prompt 融合参考与 3 个部署 checkpoint。"
    )
    lines.append("")
    lines.append("## 7. 时延口径")
    lines.append("")
    timing = report["timing_scope"]
    lines.append(
        f"- 首个样本 `model_inference_seconds` = "
        f"{timing['first_sample_model_inference_seconds']:.4f} s，**包含首次执行开销**。"
    )
    lines.append(
        "- 六个计时都**不是**稳态时延证据；模型加载时间不计入每样本计时段。"
    )
    lines.append(
        "- `selector_cpu_roundtrip_seconds` 是软件选择器往返，**不是 FPGA 时延**，"
        "**不是加速主张**。"
    )
    lines.append("")
    lines.append("## 8. 限制")
    lines.append("")
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    lines.append("## 9. 决策边界")
    lines.append("")
    if report["decision"] == DECISION_PASS:
        lines.append(
            "- `CONTRACT_SMOKE_PASS`：仅允许执行一次冻结的 full2638 发展性运行"
            "（唯一新臂 `conservative_l2_guarded_rerank_v2`）；"
            "不是论文确认、外部验证、可部署性或加速结论。"
        )
    else:
        lines.append(
            "- `CONTRACT_SMOKE_FAIL`：不得执行 full2638；先修复合同违约并重新预注册。"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def write_reports(report: Mapping[str, Any], paths: InputPaths) -> None:
    paths.report_json.parent.mkdir(parents=True, exist_ok=True)
    paths.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths.report_md.write_text(render_markdown(report), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-smoke", type=Path, default=RAW_SMOKE)
    parser.add_argument("--preregistered", type=Path, default=PREREGISTERED_CONFIG)
    parser.add_argument("--full-manifest", type=Path, default=FULL_MANIFEST)
    parser.add_argument("--dense-full-jsonl", type=Path, default=DENSE_FULL_JSONL)
    parser.add_argument(
        "--corrected-l2-integer-jsonl", type=Path, default=CORRECTED_L2_INTEGER_JSONL
    )
    parser.add_argument(
        "--prompt-plus-corrected-l2-integer-jsonl",
        type=Path,
        default=PROMPT_PLUS_CORRECTED_L2_INTEGER_JSONL,
    )
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Audit only; do not write the report files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = InputPaths(
        raw_smoke=args.raw_smoke,
        preregistered=args.preregistered,
        full_manifest=args.full_manifest,
        dense_full_jsonl=args.dense_full_jsonl,
        corrected_l2_integer_jsonl=args.corrected_l2_integer_jsonl,
        prompt_plus_corrected_l2_integer_jsonl=args.prompt_plus_corrected_l2_integer_jsonl,
        report_json=args.report_json,
        report_md=args.report_md,
    )
    report = run_audit(paths)
    if not args.no_write:
        write_reports(report, paths)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "contract_all_passed": report["contract_all_passed"],
                "failed_checks": report["failed_checks"],
                "sample_ids": report["sample_ids"],
                "raw_sha256": report["raw_smoke"]["sha256_observed"],
                "gpu_runs": report["gpu_runs"],
                "audited_gpu_runs": report["audited_gpu_runs"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["contract_all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
