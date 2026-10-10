#!/usr/bin/env python3
"""CPU-only, read-only audit + diagnostic postmortem of the v2 full2638 arm.

This audit is deliberately separate from the frozen full2638 development
summariser.  It re-verifies, without loading a model, using a GPU, touching the
network or installing anything, that the already-produced raw development file

    artifacts/token_selector/2026-10-06-conservative-l2-guarded-reranker-v2-full2638/
        conservative_l2_guarded_rerank_v2_k96_full2638.jsonl

satisfies the frozen v2 preregistration contract, and recomputes the frozen
decision from scratch.  The raw file was produced earlier by the host GPU
runner; this task performs **zero** GPU runs (``gpu_runs=0``) and audits one
previously executed run (``audited_gpu_runs=1``, ``audited_samples=2638``).

Audit items (every one must pass for an audit pass)
---------------------------------------------------
1. Raw SHA256, exactly 2638 unique manifest ids, no missing/extra/duplicate rows
   and every ``record_sha256``.
2. Model/config/checkpoint/source pins and every leakage flag.
3. Exact ``G<=96`` identity-bypass contract and exact ``K=96`` / 87 real + 9
   context for every compressed row.
4. Frozen anchors, same-region one-for-one swaps, preserved per-region real
   counts and nonempty context pools.
5. Every swap gain identity and every deterministic array hash.
6. Parsing + the frozen official primary metric ``score(...)["cv_bench_accuracy"]``
   for dense, corrected integer-L2, prior prompt fusion and v2.
7. The stratified paired bootstrap (v2 minus corrected integer-L2), exactly 5000
   repeats and seed 20261006, plus the 90%-of-dense floor and exact shortfall.
8. Every preregistered decision gate, requiring the frozen decision to remain
   exactly ``DEVELOPMENT_STOP`` whenever any gate fails.

The postmortem is **diagnostic / development-only**.  It is label-aware (it
uses the official CV-Bench ground-truth letters only to count repairs,
regressions and unchanged samples) and must never be used to tune or select a v3
rule from these labels.  full2638 has been observed repeatedly and is never an
untouched confirmation set.  No FPGA latency, board deployment, paper accuracy
or external-validation claim is made here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from evaluation.build_cv_bench_pruning_report import (  # noqa: E402
    answer_letter,
    paired_bootstrap_delta,
    parse_choice,
    score,
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

RAW_FULL2638 = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-06-conservative-l2-guarded-reranker-v2-full2638"
    / "conservative_l2_guarded_rerank_v2_k96_full2638.jsonl"
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
    / "2026-10-06-conservative-l2-guarded-reranker-v2-full2638-audit.json"
)
REPORT_MD = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-06-conservative-l2-guarded-reranker-v2-full2638-audit.md"
)
POSTMORTEM_JSON = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-06-conservative-l2-guarded-reranker-v2-full2638-postmortem.json"
)
POSTMORTEM_MD = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-06-conservative-l2-guarded-reranker-v2-full2638-postmortem.md"
)

#: Claimed SHA256 of the audited raw full2638 arm (verified, never trusted).
EXPECTED_RAW_SHA256 = (
    "3018f7593d31fd02854f3f63f992b6fc6293ba64c24185e2905877a71d88fdb1"
)
#: Frozen preregistration config SHA256.
EXPECTED_CONFIG_SHA256 = (
    "7499b6ea00939c58506e73b496c1c62bed08162dc8d6d535229f2ffa703e74eb"
)

#: Claimed observed facts (independently recomputed by this audit).
EXPECTED_RECORDS = 2638
EXPECTED_BYPASS = 143
EXPECTED_COMPRESSED = 2495
EXPECTED_DENSE_SCORE = 0.8727298774444869
EXPECTED_L2_SCORE = 0.7668517264725796
EXPECTED_PROMPT_SCORE = 0.7773617325561999
EXPECTED_V2_SCORE = 0.7837404943432142
EXPECTED_DELTA = 0.016888767870634647
EXPECTED_CI = (0.006157592750679508, 0.0271543129924543)
EXPECTED_FLOOR = 0.9 * EXPECTED_DENSE_SCORE
BOOTSTRAP_SEED = 20261006
BOOTSTRAP_REPEATS = 5000
CI_LEVEL = 0.95

#: Frozen preregistered decision outcomes.
OUTCOME_GO = "DEVELOPMENT_GO_EXTERNAL_VALIDATION"
OUTCOME_STOP = "DEVELOPMENT_STOP"

#: Audit-level decisions (distinct from the development decision).
AUDIT_PASS = "AUDIT_PASS"
AUDIT_FAIL = "AUDIT_FAIL"

#: Frozen original-G bins, copied verbatim from the heterogeneity diagnostic.
PRE_SELECTOR_BINS: tuple[tuple[str, Callable[[int], bool]], ...] = (
    ("<=96", lambda value: value <= 96),
    ("97-256", lambda value: 97 <= value <= 256),
    ("257-512", lambda value: 257 <= value <= 512),
    ("513-1024", lambda value: 513 <= value <= 1024),
    (">1024", lambda value: value > 1024),
)
PRE_SELECTOR_BIN_ORDER = tuple(label for label, _ in PRE_SELECTOR_BINS)

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
AUDIT_HASH_FIELDS = (
    "utility_q16_sha256",
    "semantic_norm_q15_sha256",
    "l2_norm_q15_sha256",
    "l2_combined_sha256",
    "baseline_selected_real_sha256",
    "reranked_selected_real_sha256",
)
RECOMPUTABLE_RECORD_HASH_FIELDS = (
    ("selected_group_indices", "selected_group_indices_sha256", np.int64),
    ("selected_patch_indices", "selected_patch_indices_sha256", np.int64),
    ("source_type", "source_type_sha256", np.int64),
    ("region_id", "region_id_sha256", np.int64),
    ("valid", "valid_sha256", bool),
)

ITEM_1 = "1_raw_sha_2638_unique_manifest_ids_and_record_hashes"
ITEM_2 = "2_model_config_checkpoint_source_pins_and_leakage_flags"
ITEM_3 = "3_identity_bypass_contract_and_exact_k96_87_real_9_context"
ITEM_4 = "4_anchors_frozen_same_region_swaps_counts_and_nonempty_context"
ITEM_5 = "5_swap_gain_identities_and_deterministic_array_hashes"
ITEM_6 = "6_frozen_official_primary_metric_all_four_arms"
ITEM_7 = "7_stratified_paired_bootstrap_5000_seed_20261006_and_90pct_dense_floor"
ITEM_8 = "8_decision_gates_recomputed_development_stop"

CONTRACT_ITEMS = (
    ITEM_1,
    ITEM_2,
    ITEM_3,
    ITEM_4,
    ITEM_5,
    ITEM_6,
    ITEM_7,
    ITEM_8,
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
    """Collects named boolean checks grouped under the eight audit items."""

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


def _acc(
    store: dict[str, dict[str, Any]],
    name: str,
    passed: bool,
    sample_id: str,
) -> None:
    entry = store.setdefault(name, {"checked": 0, "failed": 0, "failed_ids": []})
    entry["checked"] += 1
    if not passed:
        entry["failed"] += 1
        if len(entry["failed_ids"]) < 20:
            entry["failed_ids"].append(sample_id)


def _register_aggregates(
    recorder: CheckRecorder,
    item: str,
    store: Mapping[str, Mapping[str, Any]],
    names: Sequence[str],
) -> None:
    for name in names:
        entry = store.get(name, {"checked": 0, "failed": 0, "failed_ids": []})
        recorder.add(
            item,
            name,
            entry["failed"] == 0 and entry["checked"] > 0,
            {
                "checked": entry["checked"],
                "failed": entry["failed"],
                "failed_ids": entry["failed_ids"][:20],
            },
        )


@dataclass(frozen=True)
class InputPaths:
    raw_full2638: Path = RAW_FULL2638
    preregistered: Path = PREREGISTERED_CONFIG
    full_manifest: Path = FULL_MANIFEST
    dense_full_jsonl: Path = DENSE_FULL_JSONL
    corrected_l2_integer_jsonl: Path = CORRECTED_L2_INTEGER_JSONL
    prompt_plus_corrected_l2_integer_jsonl: Path = (
        PROMPT_PLUS_CORRECTED_L2_INTEGER_JSONL
    )
    report_json: Path = REPORT_JSON
    report_md: Path = REPORT_MD
    postmortem_json: Path = POSTMORTEM_JSON
    postmortem_md: Path = POSTMORTEM_MD


DEFAULT_PATHS = InputPaths()


# ---------------------------------------------------------------------------
# Frozen contract helpers
# ---------------------------------------------------------------------------
def _grid_hw(grid: Sequence[int]) -> tuple[int, int]:
    return int(grid[1]) // 2, int(grid[2]) // 2


def _expected_context_count(grid: Sequence[int]) -> int:
    grid_h, grid_w = _grid_hw(grid)
    return len(
        [
            region
            for region in build_regions(grid_h, grid_w, REGION_GRID)
            if region.members and len(region.members) >= 2
        ]
    )


def _region_lookup(grid: Sequence[int]) -> np.ndarray:
    grid_h, grid_w = _grid_hw(grid)
    return region_id_by_group(grid_h, grid_w, REGION_GRID)


def _expected_checkpoint_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    pins = config.get("checkpoint_pins", {})
    return {
        "checkpoint_sha256": pins.get("projection_npz", {}).get("sha256"),
        "wv_sha256": pins.get("wv_int8", {}).get("sha256"),
        "wv_scale_sha256": pins.get("wv_per_tensor_scale", {}).get("sha256"),
        "deployment_seed": pins.get("deployment_seed"),
        "projection_npz": pins.get("projection_npz", {}).get("path"),
    }


def _config_pin_table(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: {"path": str(entry.get("path")), "sha256": entry.get("sha256")}
        for name, entry in config.get("inputs_sha256", {}).items()
    }


def _arm_metrics(joined: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    """Reproduce the frozen summariser arm metrics exactly."""

    rows = [
        {
            "task": record["task"],
            "type": record["type"],
            "source": record["source"],
            "source_dataset": record["source_dataset"],
            "correct": record[f"{arm}_correct"],
            "answer": record[f"{arm}_answer"],
        }
        for record in joined
    ]
    result = score(rows, "correct")
    return {
        "cv_bench_accuracy": float(result["cv_bench_accuracy"]),
        "accuracy_2d": float(result["accuracy_2d"]),
        "accuracy_3d": float(result["accuracy_3d"]),
        "unweighted_sample_accuracy": float(result["accuracy"]),
        "parse_success_rate": float(result["parse_success_rate"]),
        "sample_count": int(result["sample_count"]),
    }


def _build_joined(
    sample_ids: Sequence[str],
    manifest_index: Mapping[str, Mapping[str, Any]],
    arm_indices: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    new_index: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        meta = manifest_index[sample_id]
        choices = list(meta["choices"])
        truth = answer_letter(str(meta["answer"]))
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "task": meta["task"],
            "type": meta["type"],
            "source": meta["source"],
            "source_dataset": meta["source_dataset"],
            "ground_truth": truth,
        }
        if new_index is not None:
            record["G"] = int(new_index[sample_id]["full_llm_visual_tokens"])
            record["identity_bypass"] = bool(
                new_index[sample_id].get("identity_bypass")
            )
            record["swap_count"] = int(new_index[sample_id].get("swap_count", 0))
        for arm, index in arm_indices.items():
            parsed = parse_choice(str(index[sample_id]["output_text"]), choices)
            record[f"{arm}_answer"] = parsed
            record[f"{arm}_correct"] = parsed == truth
        joined.append(record)
    return joined


# ---------------------------------------------------------------------------
# Item 1: raw identity
# ---------------------------------------------------------------------------
def _audit_raw_identity(
    recorder: CheckRecorder,
    *,
    raw_sha256: str,
    raw_rows: Sequence[Mapping[str, Any]],
    manifest_index: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    observed_ids = [str(record["sample_id"]) for record in raw_rows]
    recorder.add(
        ITEM_1,
        "raw_sha256_matches_claim",
        raw_sha256 == EXPECTED_RAW_SHA256,
        {"observed": raw_sha256, "expected": EXPECTED_RAW_SHA256},
    )
    recorder.add(
        ITEM_1,
        "raw_record_count_2638",
        len(raw_rows) == EXPECTED_RECORDS,
        {"records": len(raw_rows)},
    )
    recorder.add(
        ITEM_1,
        "raw_unique_sample_ids_2638",
        len(set(observed_ids)) == EXPECTED_RECORDS
        and len(set(observed_ids)) == len(observed_ids),
        {"unique": len(set(observed_ids))},
    )
    missing = sorted(set(manifest_index) - set(observed_ids))
    extra = sorted(set(observed_ids) - set(manifest_index))
    recorder.add(
        ITEM_1,
        "raw_ids_equal_manifest_no_missing_extra",
        not missing and not extra and set(observed_ids) == set(manifest_index),
        {"missing": missing[:20], "extra": extra[:20]},
    )
    for name, reference in references.items():
        recorder.add(
            ITEM_1,
            f"reference_2638_unique_ids_{name}",
            reference["row_count"] == EXPECTED_RECORDS
            and reference["unique_ids"] == EXPECTED_RECORDS
            and set(reference["index"]) == set(manifest_index),
            {
                "rows": reference["row_count"],
                "unique_ids": reference["unique_ids"],
            },
        )
    return {"sample_ids": observed_ids, "missing": missing, "extra": extra}


# ---------------------------------------------------------------------------
# Items 2/3/4/5: per-record contract
# ---------------------------------------------------------------------------
def _audit_record_contract(
    recorder: CheckRecorder,
    *,
    records: Sequence[Mapping[str, Any]],
    manifest_index: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    checkpoint = _expected_checkpoint_identity(config)
    pin_store: dict[str, dict[str, Any]] = {}
    layout_store: dict[str, dict[str, Any]] = {}
    anchor_store: dict[str, dict[str, Any]] = {}
    hash_store: dict[str, dict[str, Any]] = {}
    bypass_ids: list[str] = []
    compressed_ids: list[str] = []
    swap_histogram: Counter[int] = Counter()
    compressed_zero_swap = 0
    first_model_path: str | None = None
    first_model_label: str | None = None
    model_path_consistent = True
    model_label_consistent = True
    model_path_exists = True

    for record in records:
        sample_id = str(record["sample_id"])
        manifest_row = manifest_index[sample_id]
        run = record.get("run") or {}
        grid = [int(value) for value in record["image_grid_thw"]]
        full_patch = int(record["full_patch_tokens"])
        full_llm = int(record["full_llm_visual_tokens"])
        output_count = int(record["output_group_count"])
        real_count = int(record["real_group_count"])
        context_count = int(record["context_group_count"])
        bypass = bool(record.get("identity_bypass"))
        selected = [int(value) for value in record["selected_group_indices"]]
        source_type = [int(value) for value in record["source_type"]]
        region_id = [int(value) for value in record["region_id"]]
        valid = [bool(value) for value in record["valid"]]
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
        swap_count = int(record.get("swap_count", -1))

        # ---- item 2: identity / pins / leakage -------------------------
        _acc(
            pin_store,
            "arm_mechanism_identity",
            str(record.get("arm")) == MECHANISM
            and str(record.get("mechanism")) == MECHANISM,
            sample_id,
        )
        _acc(
            pin_store,
            "run_mode_full2638",
            str(record.get("run_mode")) == "full2638"
            and str(run.get("run_mode")) == "full2638",
            sample_id,
        )
        _acc(
            pin_store,
            "record_sha256_recomputes",
            str(record.get("record_sha256")) == record_sha256(dict(record)),
            sample_id,
        )
        _acc(
            pin_store,
            "manifest_sha256_pin",
            str(record.get("manifest_sha256")) == manifest_sha256,
            sample_id,
        )
        _acc(
            pin_store,
            "record_task_matches_manifest",
            str(record.get("task")) == str(manifest_row.get("task")),
            sample_id,
        )
        _acc(
            pin_store,
            "checkpoint_pins",
            all(
                (
                    str(record.get("checkpoint_sha256"))
                    == str(checkpoint["checkpoint_sha256"]),
                    str(record.get("wv_sha256")) == str(checkpoint["wv_sha256"]),
                    str(record.get("wv_scale_sha256"))
                    == str(checkpoint["wv_scale_sha256"]),
                    int(record.get("deployment_seed", -1))
                    == int(checkpoint["deployment_seed"]),
                    str(record.get("projection_npz")) == str(checkpoint["projection_npz"]),
                )
            ),
            sample_id,
        )
        _acc(
            pin_store,
            "run_metadata_pins",
            all(
                (
                    str(run.get("arm")) == MECHANISM,
                    str(run.get("mechanism")) == MECHANISM,
                    str(run.get("manifest_sha256")) == manifest_sha256,
                    str(run.get("checkpoint_sha256"))
                    == str(checkpoint["checkpoint_sha256"]),
                    str(run.get("wv_sha256")) == str(checkpoint["wv_sha256"]),
                    str(run.get("wv_scale_sha256"))
                    == str(checkpoint["wv_scale_sha256"]),
                    int(run.get("deployment_seed", -1))
                    == int(checkpoint["deployment_seed"]),
                )
            ),
            sample_id,
        )
        _acc(
            pin_store,
            "leakage_flags_all_false",
            all(not bool(record.get(flag)) for flag in LEAKAGE_FLAGS)
            and all(not bool(run.get(flag)) for flag in LEAKAGE_FLAGS),
            sample_id,
        )
        _acc(
            pin_store,
            "no_acceleration_claim",
            record.get("selector_software_is_acceleration_claim") is False
            and record.get("cpu_selector_roundtrip_is_acceleration_claim") is False
            and run.get("cpu_selector_roundtrip_is_acceleration_claim") is False,
            sample_id,
        )
        _acc(
            pin_store,
            "frozen_flag_booleans",
            record.get("frozen_anchors") is True
            and record.get("same_region_one_for_one_swaps") is True,
            sample_id,
        )
        _acc(
            pin_store,
            "fixed_budget_grid_formula_fields",
            all(
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
            ),
            sample_id,
        )
        _acc(
            pin_store,
            "timing_alias_identity",
            record.get("elapsed_seconds") == record.get("model_inference_seconds")
            and record.get("timed_stage_seconds")
            == record.get("model_inference_seconds")
            and record.get("vision_encoder_seconds")
            == record.get("vision_total_seconds"),
            sample_id,
        )
        if first_model_path is None:
            first_model_path = str(run.get("model_path"))
            first_model_label = str(run.get("model_label"))
            model_path_exists = Path(first_model_path).is_dir()
        if str(run.get("model_path")) != first_model_path:
            model_path_consistent = False
        if str(run.get("model_label")) != first_model_label:
            model_label_consistent = False

        # ---- item 3: layout -------------------------------------------
        grid_ok = (
            len(grid) == 3
            and grid[0] == 1
            and grid[1] * grid[2] == full_patch
            and full_llm * MERGE_UNIT == full_patch
        )
        _acc(layout_store, "grid_and_merge_unit", grid_ok, sample_id)
        selected_ok = (
            len(selected) == output_count
            and len(set(selected)) == len(selected)
            and all(0 <= value < full_llm for value in selected)
            and all(later > earlier for earlier, later in zip(selected, selected[1:]))
        )
        _acc(
            layout_store,
            "selected_indices_ascending_unique_in_range",
            selected_ok,
            sample_id,
        )
        expected_context = _expected_context_count(grid)
        region_ids_all = _region_lookup(grid)
        region_id_ok = len(region_id) == output_count and all(
            int(region_ids_all[group]) == int(region_id[index])
            for index, group in enumerate(selected)
        )
        _acc(layout_store, "region_id_matches_frozen_grid_map", region_id_ok, sample_id)
        valid_ok = len(valid) == output_count and all(valid)
        _acc(layout_store, "valid_flags_all_true", valid_ok, sample_id)
        if bypass:
            bypass_ids.append(sample_id)
            bypass_ok = (
                full_llm <= 96
                and output_count == full_llm
                and real_count == full_llm
                and context_count == 0
                and selected == list(range(full_llm))
                and all(value == SOURCE_REAL for value in source_type)
                and swap_count == 0
                and baseline_real == []
                and reranked_real == []
                and anchors == []
                and removed == []
                and added == []
                and all(record.get(field) is None for field in AUDIT_HASH_FIELDS)
                and record.get("combined_scores_sha256") is None
                and record.get("score_vector_length") is None
            )
            _acc(layout_store, "bypass_exact_contract", bypass_ok, sample_id)
            compressed_zero_swap += 0
        else:
            compressed_ids.append(sample_id)
            compressed_swap_ok = (
                full_llm > 96
                and output_count == 96
                and real_count == EXPECTED_REAL
                and context_count == EXPECTED_CONTEXT
                and context_count == expected_context
                and real_count + context_count == 96
            )
            _acc(
                layout_store,
                "compressed_exact_k96_87_real_9_context",
                compressed_swap_ok,
                sample_id,
            )
            if swap_count == 0:
                compressed_zero_swap += 1
        swap_histogram[swap_count] += 1

        # ---- item 5 (hashes) ------------------------------------------
        hash_ok = True
        for array_field, hash_field, dtype in RECOMPUTABLE_RECORD_HASH_FIELDS:
            observed = record.get(hash_field)
            recomputed = array_sha256(
                np.asarray(record[array_field], dtype=dtype)
            )
            if str(observed) != recomputed:
                hash_ok = False
        _acc(hash_store, "recomputable_record_hashes", hash_ok, sample_id)

        expected_patches: list[int] = []
        for group in selected:
            expected_patches.extend(
                group * MERGE_UNIT + offset for offset in range(MERGE_UNIT)
            )
        _acc(
            hash_store,
            "selected_patch_indices_derive_from_groups",
            [int(value) for value in record["selected_patch_indices"]]
            == expected_patches,
            sample_id,
        )
        if bypass:
            _acc(
                hash_store,
                "bypass_audit_arrays_empty_and_hashes_null",
                baseline_real == []
                and reranked_real == []
                and anchors == []
                and all(record.get(field) is None for field in AUDIT_HASH_FIELDS),
                sample_id,
            )
            _acc(
                hash_store,
                "bypass_combined_hash_null",
                record.get("combined_scores_sha256") is None
                and record.get("score_vector_length") is None,
                sample_id,
            )
        else:
            _acc(
                hash_store,
                "baseline_selected_real_sha256",
                str(record.get("baseline_selected_real_sha256"))
                == array_sha256(np.asarray(baseline_real, dtype=np.int64)),
                sample_id,
            )
            _acc(
                hash_store,
                "reranked_selected_real_sha256",
                str(record.get("reranked_selected_real_sha256"))
                == array_sha256(np.asarray(reranked_real, dtype=np.int64)),
                sample_id,
            )
            _acc(
                hash_store,
                "audit_array_hashes_are_hex64",
                all(_is_hex64(record.get(field)) for field in AUDIT_HASH_FIELDS),
                sample_id,
            )
            _acc(
                hash_store,
                "combined_scores_hash_and_length",
                _is_hex64(record.get("combined_scores_sha256"))
                and int(record.get("score_vector_length", -1)) == full_llm,
                sample_id,
            )
        _acc(
            hash_store,
            "block_input_shapes_metadata",
            isinstance(record.get("block_input_shapes"), list)
            and len(record["block_input_shapes"]) == 24
            and all(int(value) > 0 for value in record["block_input_shapes"]),
            sample_id,
        )

        # ---- item 5 (swap gain identities) ----------------------------
        lengths = {len(record[field]) for field in SWAP_ARRAY_FIELDS}
        _acc(
            hash_store,
            "swap_array_lengths_match_count",
            lengths == {swap_count},
            sample_id,
        )
        _acc(
            hash_store,
            "swap_count_within_S",
            0 <= swap_count <= MAX_SWAPS,
            sample_id,
        )
        legal_ok = all(
            legal_gain[index] == sem_gain[index] - l2_loss[index]
            and legal_gain[index] == util_added[index] - util_removed[index]
            and legal_gain[index] > 0
            for index in range(len(legal_gain))
        )
        _acc(hash_store, "legal_gain_identity_and_positive", legal_ok, sample_id)
        no_reuse = (
            len(set(removed)) == len(removed)
            and len(set(added)) == len(added)
            and not (set(removed) & set(added))
        )
        _acc(hash_store, "no_reused_or_overlapping_groups", no_reuse, sample_id)

        # ---- item 4: anchors / same region / counts / context ----------
        expected_regions = sorted({int(region_ids_all[group]) for group in range(full_llm)})
        if bypass:
            _acc(
                anchor_store,
                "anchors_frozen_one_per_region",
                anchors == [],
                sample_id,
            )
            _acc(anchor_store, "same_region_one_for_one", True, sample_id)
            _acc(anchor_store, "swap_audit_reconstructs_reranked_real", True, sample_id)
            _acc(anchor_store, "per_region_real_counts_preserved", True, sample_id)
            _acc(anchor_store, "nonempty_context_invariant", context_count == 0, sample_id)
        else:
            anchor_regions = sorted({int(region_ids_all[group]) for group in anchors})
            anchors_ok = (
                len(anchors) >= 1
                and set(anchors).issubset(set(baseline_real))
                and set(anchors).issubset(set(reranked_real))
                and not (set(anchors) & set(removed))
                and anchor_regions == expected_regions
            )
            _acc(anchor_store, "anchors_frozen_one_per_region", anchors_ok, sample_id)
            same_region_ok = all(
                int(region_ids_all[removed[index]])
                == int(regions[index])
                == int(region_ids_all[added[index]])
                for index in range(len(removed))
            )
            _acc(anchor_store, "same_region_one_for_one", same_region_ok, sample_id)
            reconstruct = (
                (set(baseline_real) - set(removed)) | set(added)
            ) == set(reranked_real)
            _acc(
                anchor_store,
                "swap_audit_reconstructs_reranked_real",
                set(removed).issubset(set(baseline_real))
                and not (set(added) & set(baseline_real))
                and reconstruct,
                sample_id,
            )
            before = [
                sum(1 for group in baseline_real if int(region_ids_all[group]) == region)
                for region in expected_regions
            ]
            after = [
                sum(1 for group in reranked_real if int(region_ids_all[group]) == region)
                for region in expected_regions
            ]
            counts_ok = (
                before == after
                and before
                == [int(value) for value in record["per_region_selected_counts_before"]]
                and after
                == [int(value) for value in record["per_region_selected_counts_after"]]
            )
            _acc(anchor_store, "per_region_real_counts_preserved", counts_ok, sample_id)
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
            context_regions = [int(region_ids_all[group]) for group in context_groups]
            nonempty = True
            for group in context_groups:
                region_id_value = int(region_ids_all[group])
                members = [
                    candidate
                    for candidate in range(full_llm)
                    if int(region_ids_all[candidate]) == region_id_value
                ]
                if not [member for member in members if member not in real_slots]:
                    nonempty = False
            context_ok = (
                nonempty
                and sorted(context_regions) == expected_regions
                and set(context_groups) == (set(selected) - real_slots)
                and real_slots == set(reranked_real)
                and len(real_slots) == EXPECTED_REAL
                and len(context_groups) == EXPECTED_CONTEXT
            )
            _acc(anchor_store, "nonempty_context_invariant", context_ok, sample_id)

    _register_aggregates(
        recorder,
        ITEM_2,
        pin_store,
        (
            "arm_mechanism_identity",
            "run_mode_full2638",
            "record_sha256_recomputes",
            "manifest_sha256_pin",
            "record_task_matches_manifest",
            "checkpoint_pins",
            "run_metadata_pins",
            "leakage_flags_all_false",
            "no_acceleration_claim",
            "frozen_flag_booleans",
            "fixed_budget_grid_formula_fields",
            "timing_alias_identity",
        ),
    )
    recorder.add(
        ITEM_2,
        "model_path_and_label_consistent",
        model_path_consistent and model_label_consistent,
        {
            "model_path": first_model_path,
            "model_label": first_model_label,
            "path_consistent": model_path_consistent,
            "label_consistent": model_label_consistent,
        },
    )
    recorder.add(
        ITEM_2,
        "model_path_exists_on_disk",
        bool(model_path_exists),
        {"model_path": first_model_path, "exists": model_path_exists},
    )
    recorder.add(
        ITEM_2,
        "model_identity_metadata_only_disclosed",
        True,
        {
            "note": (
                "the frozen config carries no model-weight SHA pin; model identity "
                "is metadata-only (model_path/model_label) while the three checkpoint "
                "files (projection npz, Wv int8, Wv scale) are SHA-pinned"
            ),
        },
    )
    _register_aggregates(
        recorder,
        ITEM_3,
        layout_store,
        (
            "grid_and_merge_unit",
            "selected_indices_ascending_unique_in_range",
            "region_id_matches_frozen_grid_map",
            "valid_flags_all_true",
            "bypass_exact_contract",
            "compressed_exact_k96_87_real_9_context",
        ),
    )
    recorder.add(
        ITEM_3,
        "observed_record_shape",
        len(bypass_ids) == EXPECTED_BYPASS
        and len(compressed_ids) == EXPECTED_COMPRESSED
        and len(bypass_ids) + len(compressed_ids) == EXPECTED_RECORDS,
        {
            "bypass_count": len(bypass_ids),
            "compressed_count": len(compressed_ids),
            "bypass_expected": EXPECTED_BYPASS,
            "compressed_expected": EXPECTED_COMPRESSED,
        },
    )
    _register_aggregates(
        recorder,
        ITEM_4,
        anchor_store,
        (
            "anchors_frozen_one_per_region",
            "same_region_one_for_one",
            "swap_audit_reconstructs_reranked_real",
            "per_region_real_counts_preserved",
            "nonempty_context_invariant",
        ),
    )
    _register_aggregates(
        recorder,
        ITEM_5,
        hash_store,
        (
            "recomputable_record_hashes",
            "selected_patch_indices_derive_from_groups",
            "bypass_audit_arrays_empty_and_hashes_null",
            "bypass_combined_hash_null",
            "baseline_selected_real_sha256",
            "reranked_selected_real_sha256",
            "audit_array_hashes_are_hex64",
            "combined_scores_hash_and_length",
            "block_input_shapes_metadata",
            "swap_array_lengths_match_count",
            "swap_count_within_S",
            "legal_gain_identity_and_positive",
            "no_reused_or_overlapping_groups",
        ),
    )
    return {
        "bypass_ids": bypass_ids,
        "compressed_ids": compressed_ids,
        "swap_histogram": {str(key): value for key, value in sorted(swap_histogram.items())},
        "compressed_zero_swap": compressed_zero_swap,
    }


# ---------------------------------------------------------------------------
# Item 6: frozen official primary metric
# ---------------------------------------------------------------------------
def _audit_metrics(
    recorder: CheckRecorder,
    *,
    joined: Sequence[Mapping[str, Any]],
    arms: Mapping[str, str],
    claimed: Mapping[str, float],
) -> dict[str, Any]:
    metrics = {name: _arm_metrics(joined, arm) for name, arm in arms.items()}
    for name, arm in arms.items():
        observed = float(metrics[name]["cv_bench_accuracy"])
        recorder.add(
            ITEM_6,
            f"{name}_cv_bench_accuracy_matches_claim",
            abs(observed - float(claimed[name])) < 1e-15,
            {"observed": observed, "claimed": float(claimed[name])},
        )
    true_parse = {
        name: {
            "failures": sum(
                1 for record in joined if record[f"{arm}_answer"] is None
            ),
            "parse_rate": sum(
                1 for record in joined if record[f"{arm}_answer"] is not None
            )
            / len(joined),
        }
        for name, arm in arms.items()
    }
    recorder.add(
        ITEM_6,
        "four_arms_parsed_with_frozen_parse_choice",
        all(metrics[name]["sample_count"] == EXPECTED_RECORDS for name in arms),
        {name: metrics[name]["sample_count"] for name in arms},
    )
    recorder.add(
        ITEM_6,
        "frozen_primary_metric_identifier",
        True,
        {
            "identifier": 'score(...)["cv_bench_accuracy"]',
            "implementation": "evaluation.build_cv_bench_pruning_report.score",
        },
    )
    return {"metrics": metrics, "true_parse": true_parse}


# ---------------------------------------------------------------------------
# Item 7: bootstrap + floor
# ---------------------------------------------------------------------------
def _audit_bootstrap(
    recorder: CheckRecorder,
    *,
    joined: Sequence[Mapping[str, Any]],
    new_metrics: Mapping[str, Any],
    l2_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [
        {
            "task": record["task"],
            "type": record["type"],
            "source": record["source"],
            "source_dataset": record["source_dataset"],
            "left_correct": record["new_correct"],
            "left_answer": record["new_answer"],
            "right_correct": record["corrected_l2_integer_correct"],
            "right_answer": record["corrected_l2_integer_answer"],
        }
        for record in joined
    ]
    point = float(
        score(rows, "left_correct")["cv_bench_accuracy"]
        - score(rows, "right_correct")["cv_bench_accuracy"]
    )
    ci = paired_bootstrap_delta(
        rows,
        "left_correct",
        "right_correct",
        BOOTSTRAP_SEED,
        repeats=BOOTSTRAP_REPEATS,
    )
    recorder.add(
        ITEM_7,
        "bootstrap_seed_and_repeats_frozen",
        BOOTSTRAP_SEED == 20261006 and BOOTSTRAP_REPEATS == 5000,
        {"seed": BOOTSTRAP_SEED, "repeats": BOOTSTRAP_REPEATS},
    )
    recorder.add(
        ITEM_7,
        "delta_point_estimate_matches_claim",
        abs(point - EXPECTED_DELTA) < 1e-15,
        {"observed": point, "claimed": EXPECTED_DELTA},
    )
    recorder.add(
        ITEM_7,
        "ci_matches_claim",
        abs(float(ci[0]) - EXPECTED_CI[0]) < 1e-15
        and abs(float(ci[1]) - EXPECTED_CI[1]) < 1e-15,
        {"observed": [float(ci[0]), float(ci[1])], "claimed": list(EXPECTED_CI)},
    )
    v2_score = float(new_metrics["cv_bench_accuracy"])
    dense_score = float(
        _arm_metrics(joined, "dense")["cv_bench_accuracy"]
    )
    floor = 0.9 * EXPECTED_DENSE_SCORE
    recorder.add(
        ITEM_7,
        "floor_equals_90pct_dense",
        abs(floor - EXPECTED_FLOOR) < 1e-18
        and abs(floor - 0.9 * dense_score) < 1e-15,
        {"floor": floor, "expected": EXPECTED_FLOOR, "dense": dense_score},
    )
    return {
        "strata": "task::source_dataset",
        "seed": BOOTSTRAP_SEED,
        "repeats": BOOTSTRAP_REPEATS,
        "ci_level": CI_LEVEL,
        "delta_point_estimate": point,
        "delta_ci95": [float(ci[0]), float(ci[1])],
        "floor": floor,
        "shortfall_absolute": floor - v2_score,
        "shortfall_relative_to_floor": (floor - v2_score) / floor,
        "shortfall_relative_to_dense": (floor - v2_score) / dense_score,
        "retained_dense_fraction": v2_score / dense_score,
        "per_repeat_shortfall": (floor - v2_score) * BOOTSTRAP_REPEATS,
    }


# ---------------------------------------------------------------------------
# Item 8: decision gates
# ---------------------------------------------------------------------------
def _decide(
    *,
    frozen_parse_rate: float,
    layout_ok: bool,
    new_score: float,
    ci_lower: float,
    leakage_ok: bool,
) -> dict[str, Any]:
    gates = {
        "all_2638_complete_parseable": bool(frozen_parse_rate >= 1.0),
        "exact_k96_layout_contract": bool(layout_ok),
        "score_at_least_90pct_dense": bool(new_score >= EXPECTED_FLOOR),
        "paired_ci_lower_nonnegative": bool(ci_lower >= 0.0),
        "no_forbidden_leakage": bool(leakage_ok),
    }
    failed = [name for name, passed in gates.items() if not passed]
    outcome = OUTCOME_STOP if failed else OUTCOME_GO
    return {
        "outcome": outcome,
        "failed_gates": failed,
        "reason": "failed gates: " + ", ".join(failed) if failed else "all gates passed",
        "gates": gates,
    }


def _audit_decision(
    recorder: CheckRecorder,
    *,
    layout: Mapping[str, Any],
    metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    leakage_ok: bool,
) -> dict[str, Any]:
    layout_ok = (
        len(layout["bypass_ids"]) + len(layout["compressed_ids"]) == EXPECTED_RECORDS
        and len(layout["compressed_ids"]) == EXPECTED_COMPRESSED
    )
    decision = _decide(
        frozen_parse_rate=float(metrics["new"]["parse_success_rate"]),
        layout_ok=layout_ok,
        new_score=float(metrics["new"]["cv_bench_accuracy"]),
        ci_lower=float(bootstrap["delta_ci95"][0]),
        leakage_ok=leakage_ok,
    )
    recorder.add(
        ITEM_8,
        "only_failed_gate_is_score_at_least_90pct_dense",
        decision["failed_gates"] == ["score_at_least_90pct_dense"],
        {"failed_gates": decision["failed_gates"]},
    )
    recorder.add(
        ITEM_8,
        "decision_remains_development_stop",
        decision["outcome"] == OUTCOME_STOP,
        {"outcome": decision["outcome"]},
    )
    recorder.add(
        ITEM_8,
        "score_gate_value",
        float(metrics["new"]["cv_bench_accuracy"]) < EXPECTED_FLOOR,
        {
            "v2": float(metrics["new"]["cv_bench_accuracy"]),
            "floor": EXPECTED_FLOOR,
            "shortfall": EXPECTED_FLOOR
            - float(metrics["new"]["cv_bench_accuracy"]),
        },
    )
    return decision


# ---------------------------------------------------------------------------
# Postmortem (diagnostic only)
# ---------------------------------------------------------------------------
def _g_bin(value: int) -> str:
    for label, predicate in PRE_SELECTOR_BINS:
        if predicate(int(value)):
            return label
    return "UNBINNED"


def _group_accuracy(
    joined: Sequence[Mapping[str, Any]],
    *,
    key: str,
    arm: str,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in joined:
        groups[str(record[key])].append(record)
    return {
        group: {
            "n": len(records),
            "correct": sum(1 for record in records if record[f"{arm}_correct"]),
            "accuracy": sum(1 for record in records if record[f"{arm}_correct"])
            / len(records),
        }
        for group, records in sorted(groups.items())
    }


def _migration(
    joined: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
) -> dict[str, Any]:
    # "left relative to right": a repair is right wrong -> left correct.
    repairs = sum(
        1
        for record in joined
        if record[f"{left}_correct"] and (not record[f"{right}_correct"])
    )
    regressions = sum(
        1
        for record in joined
        if (not record[f"{left}_correct"]) and record[f"{right}_correct"]
    )
    unchanged_correct = sum(
        1
        for record in joined
        if record[f"{left}_correct"] and record[f"{right}_correct"]
    )
    unchanged_wrong = sum(
        1
        for record in joined
        if (not record[f"{left}_correct"]) and (not record[f"{right}_correct"])
    )
    answer_changes = sum(
        1
        for record in joined
        if record[f"{left}_answer"] != record[f"{right}_answer"]
    )

    def _by(key: str) -> dict[str, Any]:
        buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in joined:
            buckets[str(record[key])].append(record)
        return {
            group: {
                "n": len(records),
                "repairs": sum(
                    1
                    for record in records
                    if record[f"{left}_correct"]
                    and (not record[f"{right}_correct"])
                ),
                "regressions": sum(
                    1
                    for record in records
                    if (not record[f"{left}_correct"])
                    and record[f"{right}_correct"]
                ),
                "unchanged_correct": sum(
                    1
                    for record in records
                    if record[f"{left}_correct"] and record[f"{right}_correct"]
                ),
                "unchanged_wrong": sum(
                    1
                    for record in records
                    if (not record[f"{left}_correct"])
                    and (not record[f"{right}_correct"])
                ),
                "answer_changes": sum(
                    1
                    for record in records
                    if record[f"{left}_answer"] != record[f"{right}_answer"]
                ),
            }
            for group, records in sorted(buckets.items())
        }

    return {
        "left": left,
        "right": right,
        "repairs": repairs,
        "regressions": regressions,
        "unchanged_correct": unchanged_correct,
        "unchanged_wrong": unchanged_wrong,
        "net": repairs - regressions,
        "answer_changes": answer_changes,
        "by_task": _by("task"),
        "by_source_dataset": _by("source_dataset"),
        "by_source": _by("source"),
        "by_original_G_bin": _by("g_bin"),
    }


def build_postmortem(
    *,
    joined: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    true_parse: Mapping[str, Mapping[str, Any]],
    layout: Mapping[str, Any],
) -> dict[str, Any]:
    arms = {
        "new": "new",
        "dense": "dense",
        "corrected_l2_integer": "corrected_l2_integer",
        "prompt_plus_corrected_l2_integer": "prompt_plus_corrected_l2_integer",
    }
    breakdown: dict[str, Any] = {}
    for name in arms:
        breakdown[name] = {
            "per_task": _group_accuracy(joined, key="task", arm=name),
            "per_source_dataset": _group_accuracy(
                joined, key="source_dataset", arm=name
            ),
            "per_source": _group_accuracy(joined, key="source", arm=name),
            "per_original_G_bin": _group_accuracy(joined, key="g_bin", arm=name),
        }

    parse_failures = {
        name: {
            "count": int(true_parse[name]["failures"]),
            "parse_rate": float(true_parse[name]["parse_rate"]),
            "sample_ids_prefix": [
                str(record["sample_id"])
                for record in joined
                if record[f"{arms[name]}_answer"] is None
            ][:20],
            "by_task": dict(
                Counter(
                    str(record["task"])
                    for record in joined
                    if record[f"{arms[name]}_answer"] is None
                )
            ),
            "by_source_dataset": dict(
                Counter(
                    str(record["source_dataset"])
                    for record in joined
                    if record[f"{arms[name]}_answer"] is None
                )
            ),
        }
        for name in arms
    }

    bypass_records = [record for record in joined if record["identity_bypass"]]
    compressed_records = [record for record in joined if not record["identity_bypass"]]

    return {
        "report": "2026-10-06-conservative-l2-guarded-reranker-v2-full2638-postmortem",
        "date": "2026-10-06",
        "marker": "DIAGNOSTIC_DEVELOPMENT_ONLY",
        "development_only": True,
        "label_aware": True,
        "is_a_gate": False,
        "do_not_tune_or_select_v3": True,
        "full2638_role": (
            "repeatedly observed development evidence; never an untouched "
            "confirmation set"
        ),
        "primary_metric": 'score(...)["cv_bench_accuracy"]',
        "correct_counts": {
            name: {
                "cv_bench_accuracy": float(metrics[name]["cv_bench_accuracy"]),
                "correct": sum(
                    1 for record in joined if record[f"{name}_correct"]
                ),
                "n": len(joined),
            }
            for name in arms
        },
        "breakdown_by_arm": breakdown,
        "v2_vs_corrected_l2_integer": _migration(
            joined, "new", "corrected_l2_integer"
        ),
        "v2_vs_dense": _migration(joined, "new", "dense"),
        "corrected_l2_integer_vs_dense": _migration(
            joined, "corrected_l2_integer", "dense"
        ),
        "prompt_fusion_vs_corrected_l2_integer": _migration(
            joined, "prompt_plus_corrected_l2_integer", "corrected_l2_integer"
        ),
        "parse_failures": parse_failures,
        "swap_count_breakdown": {
            "histogram": dict(layout["swap_histogram"]),
            "bypass_count": layout["bypass_count"],
            "compressed_count": layout["compressed_count"],
            "compressed_zero_swap": layout["compressed_zero_swap"],
            "compressed_positive_swap": layout["compressed_count"]
            - layout["compressed_zero_swap"],
        },
        "bypass_breakdown": {
            "count": len(bypass_records),
            "by_task": dict(
                Counter(str(record["task"]) for record in bypass_records)
            ),
            "by_source_dataset": dict(
                Counter(str(record["source_dataset"]) for record in bypass_records)
            ),
            "by_original_G_bin": dict(
                Counter(str(record["g_bin"]) for record in bypass_records)
            ),
            "G_min": min(int(record["G"]) for record in bypass_records),
            "G_max": max(int(record["G"]) for record in bypass_records),
        },
        "compressed_breakdown": {
            "count": len(compressed_records),
            "by_task": dict(
                Counter(str(record["task"]) for record in compressed_records)
            ),
            "by_source_dataset": dict(
                Counter(str(record["source_dataset"]) for record in compressed_records)
            ),
        },
        "limitations": [
            "本尸检仅诊断/发展用途，使用官方答案字母计数修复/回归/不变，绝不是门，也不是精度估计。",
            "禁止据这些标签调参、选择或声称 v3 规则；full2638 是被反复观测的发展集，永不构成未触碰确认集。",
            "无数值 CI/p 值；不做论文、外部验证、FPGA 时延、上板部署或加速主张。",
            "v2 相对修正整数 L2 的点估计为正且 CI 下界 >0，但未达到 90% dense 门，故最终判定 DEVELOPMENT_STOP。",
        ],
    }


# ---------------------------------------------------------------------------
# Reference loading
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
        expected = pin.get("sha256") if pin is not None else None
        also_inputs = config.get("inputs_sha256", {}).get(name, {}).get("sha256")
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "expected_sha256": expected,
            "inputs_sha256": also_inputs,
            "row_count": len(rows),
            "unique_ids": len(index),
            "index": index,
            "claimed_score": pin.get("official_score") if pin else None,
        }

    return {
        "dense_full_jsonl": _load(
            "dense_full_jsonl", paths.dense_full_jsonl, baseline.get("dense_bf16")
        ),
        "corrected_l2_integer_jsonl": _load(
            "corrected_l2_integer_jsonl",
            paths.corrected_l2_integer_jsonl,
            baseline.get("corrected_l2_integer"),
        ),
        "prompt_plus_corrected_l2_integer_jsonl": _load(
            "prompt_plus_corrected_l2_integer_jsonl",
            paths.prompt_plus_corrected_l2_integer_jsonl,
            baseline.get("prompt_plus_corrected_l2_integer"),
        ),
    }


# ---------------------------------------------------------------------------
# Top-level audit
# ---------------------------------------------------------------------------
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
    raw_identity = _audit_raw_identity(
        recorder,
        raw_sha256=raw_sha256,
        raw_rows=raw_rows,
        manifest_index=manifest_index,
        references=references,
    )

    # item 2: config pins
    recorder.add(
        ITEM_2,
        "config_sha256_matches_frozen",
        config_sha256 == EXPECTED_CONFIG_SHA256,
        {"observed": config_sha256, "expected": EXPECTED_CONFIG_SHA256},
    )
    recorder.add(
        ITEM_2,
        "config_preregistered_bytes_intact",
        str(config.get("status")) == "PREREGISTERED_NOT_EXECUTED"
        and config.get("executed") is False
        and config.get("arms", {})
        .get(MECHANISM, {})
        .get("sha256")
        is None,
        {
            "status": config.get("status"),
            "executed": config.get("executed"),
            "arm_sha256": config.get("arms", {}).get(MECHANISM, {}).get("sha256"),
        },
    )
    pin_table = _config_pin_table(config)
    verified: list[str] = []
    mismatched: list[str] = []
    null_pins: list[str] = []
    for name, entry in pin_table.items():
        if entry["sha256"] is None:
            null_pins.append(name)
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
            "null_pins": null_pins,
        },
    )
    recorder.add(
        ITEM_2,
        "null_pin_is_only_the_new_arm",
        null_pins == ["new_arm_jsonl"],
        {"null_pins": null_pins},
    )
    recorder.add(
        ITEM_2,
        "manifest_pin_consistency",
        str(manifest_sha256)
        == str(config.get("dataset_pins", {}).get("full_manifest", {}).get("sha256"))
        == str(config.get("inputs_sha256", {}).get("full_manifest", {}).get("sha256")),
        {"manifest_sha256": manifest_sha256},
    )
    for name, reference in references.items():
        recorder.add(
            ITEM_2,
            f"reference_pin_{name}",
            reference["sha256"] == reference["expected_sha256"]
            and reference["sha256"] == reference["inputs_sha256"],
            {
                "observed": reference["sha256"],
                "expected": reference["expected_sha256"],
                "inputs_sha256": reference["inputs_sha256"],
            },
        )

    contract = _audit_record_contract(
        recorder,
        records=raw_rows,
        manifest_index=manifest_index,
        config=config,
        manifest_sha256=manifest_sha256,
    )

    arms = {
        "new": "new",
        "dense": "dense",
        "corrected_l2_integer": "corrected_l2_integer",
        "prompt_plus_corrected_l2_integer": "prompt_plus_corrected_l2_integer",
    }
    claimed = {
        "new": EXPECTED_V2_SCORE,
        "dense": EXPECTED_DENSE_SCORE,
        "corrected_l2_integer": EXPECTED_L2_SCORE,
        "prompt_plus_corrected_l2_integer": EXPECTED_PROMPT_SCORE,
    }
    new_index = unique_index(raw_rows, label="new")
    joined = _build_joined(
        list(manifest_index),
        manifest_index,
        {
            "new": new_index,
            "dense": references["dense_full_jsonl"]["index"],
            "corrected_l2_integer": references["corrected_l2_integer_jsonl"]["index"],
            "prompt_plus_corrected_l2_integer": references[
                "prompt_plus_corrected_l2_integer_jsonl"
            ]["index"],
        },
        new_index=new_index,
    )
    metric_bundle = _audit_metrics(
        recorder, joined=joined, arms=arms, claimed=claimed
    )
    bootstrap = _audit_bootstrap(
        recorder,
        joined=joined,
        new_metrics=metric_bundle["metrics"]["new"],
        l2_metrics=metric_bundle["metrics"]["corrected_l2_integer"],
    )
    leakage_ok = all(
        not bool(record.get(flag))
        for record in raw_rows
        for flag in LEAKAGE_FLAGS
    )
    decision = _audit_decision(
        recorder,
        layout=contract,
        metrics=metric_bundle["metrics"],
        bootstrap=bootstrap,
        leakage_ok=leakage_ok,
    )

    # add derived g_bin for the postmortem
    for record in joined:
        record["g_bin"] = _g_bin(int(record["G"]))

    layout_summary = {
        "bypass_count": len(contract["bypass_ids"]),
        "compressed_count": len(contract["compressed_ids"]),
        "compressed_zero_swap": contract["compressed_zero_swap"],
        "swap_histogram": contract["swap_histogram"],
    }
    postmortem = build_postmortem(
        joined=joined,
        metrics=metric_bundle["metrics"],
        true_parse=metric_bundle["true_parse"],
        layout=layout_summary,
    )

    passed = recorder.all_passed
    report: dict[str, Any] = {
        "report": "2026-10-06-conservative-l2-guarded-reranker-v2-full2638-audit",
        "date": "2026-10-06",
        "cpu_only": True,
        "read_only": True,
        "gpu_runs": 0,
        "audited_gpu_runs": 1,
        "audited_samples": EXPECTED_RECORDS,
        "model_runs": 0,
        "network_used": False,
        "training": False,
        "raw_full2638": {
            "path": str(paths.raw_full2638),
            "sha256_observed": raw_sha256,
            "sha256_claimed": EXPECTED_RAW_SHA256,
            "sha256_match": raw_sha256 == EXPECTED_RAW_SHA256,
            "records": len(raw_rows),
            "unique_sample_ids": len(set(raw_identity["sample_ids"])),
            "missing_ids": raw_identity["missing"],
            "extra_ids": raw_identity["extra"],
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
        "frozen_inputs": {
            "verified_pins": verified,
            "verified_count": len(verified),
            "mismatched_pins": mismatched,
            "null_pins": null_pins,
        },
        "scores": metric_bundle["metrics"],
        "true_parse": metric_bundle["true_parse"],
        "paired_bootstrap": bootstrap,
        "layout": layout_summary,
        "contract_checks": recorder.group(),
        "contract_all_passed": passed,
        "failed_checks": recorder.failed_keys,
        "audit_decision": AUDIT_PASS if passed else AUDIT_FAIL,
        "development_decision": decision,
        "findings": [
            {
                "id": "parse_gate_implementation_defect",
                "severity": "implementation_defect_not_artifact",
                "summary": (
                    "the frozen gate all_2638_complete_parseable is computed from "
                    "score(...)['parse_success_rate'], which is identically 1.0 for "
                    "the summariser's arm rows because the 'correct' key has no "
                    "'_correct' substring to rewrite to '_answer'; honest parse_choice "
                    "shows real failures in every arm"
                ),
                "true_parse_failures": {
                    name: metric_bundle["true_parse"][name]["failures"]
                    for name in arms
                },
                "decision_impact": (
                    "none on the outcome: the score_at_least_90pct_dense gate "
                    "independently fails, so the decision is DEVELOPMENT_STOP either way"
                ),
            },
            {
                "id": "model_weight_sha_pin_absent",
                "severity": "informational",
                "summary": (
                    "the frozen config SHA-pins the three checkpoint files but not "
                    "the model weights; model identity is metadata-only "
                    "(model_path/model_label)"
                ),
            },
        ],
        "postmortem": {
            "json": str(paths.postmortem_json),
            "md": str(paths.postmortem_md),
            "marker": postmortem["marker"],
            "is_a_gate": False,
        },
        "limitations": [
            "full2638 has been repeatedly observed and is development evidence only; it is not an untouched confirmation set.",
            "This audit is CPU-only and read-only: no model weights were loaded, no GPU was used, no network, no training and no dependency installation; gpu_runs=0.",
            "The Selector path is a software bridge; no FPGA latency, board deployment or acceleration claim is made.",
            "No paper-accuracy or external-validation claim is made; the DEVELOPMENT_STOP outcome grants no external-validation permission for v2.",
            "Do not tune v2 after observing full2638. The postmortem is diagnostic/development-only and must not be used to select a v3 rule.",
            "v1 remains permanently PREEXECUTION_DESIGN_STOP and was neither run nor patched.",
        ],
    }
    return {"report": report, "postmortem": postmortem, "joined": joined}


def run_audit(paths: InputPaths = DEFAULT_PATHS) -> dict[str, Any]:
    """Run the read-only full2638 audit and return the audit report."""

    if not paths.raw_full2638.is_file():
        raise AuditError(f"missing required input: {paths.raw_full2638}")
    raw_sha256 = sha256_file(paths.raw_full2638)
    raw_rows = load_jsonl(paths.raw_full2638)
    config = json.loads(paths.preregistered.read_text(encoding="utf-8"))
    config_sha256 = sha256_file(paths.preregistered)
    manifest_rows = load_jsonl(paths.full_manifest)
    manifest_sha256 = sha256_file(paths.full_manifest)
    manifest_index = unique_index(manifest_rows, label="full_manifest")
    references = _load_references(paths, config)
    bundle = audit_loaded(
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
    bundle["report"]["generated_at"] = datetime.now(timezone.utc).isoformat()
    bundle["postmortem"]["generated_at"] = bundle["report"]["generated_at"]
    return bundle


# ---------------------------------------------------------------------------
# Markdown (Chinese)
# ---------------------------------------------------------------------------
def _fmt(value: Any, digits: int = 16) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# L2 守卫整数重排 v2 full2638 发展性结果只读审计")
    lines.append("")
    lines.append(f"- 日期：{report['date']}")
    lines.append(f"- 审计结论：**`{report['audit_decision']}`**")
    lines.append(
        f"- 冻结发展判定：**`{report['development_decision']['outcome']}`**"
        f"（理由：{report['development_decision']['reason']}）"
    )
    lines.append(
        f"- 口径：CPU-only、只读；`gpu_runs={report['gpu_runs']}`、"
        f"`audited_gpu_runs={report['audited_gpu_runs']}`、"
        f"`audited_samples={report['audited_samples']}`；未加载模型权重/未联网/未训练/未装依赖。"
    )
    raw = report["raw_full2638"]
    lines.append("")
    lines.append("## 1. 被审计原始 full2638")
    lines.append("")
    lines.append(f"- 路径：`{raw['path']}`")
    lines.append(f"- 记录数：{raw['records']}；唯一 `sample_id`：{raw['unique_sample_ids']}")
    lines.append(
        f"- SHA256（现场重算）：`{raw['sha256_observed']}`；与声明一致："
        f"`{raw['sha256_match']}`"
    )
    lines.append(f"- 缺失 ID：{raw['missing_ids'] or '无'}；多余 ID：{raw['extra_ids'] or '无'}")
    lines.append("")
    lines.append("## 2. 八项审计检查")
    lines.append("")
    lines.append("| 审计项 | 通过 | 子检查数 |")
    lines.append("| --- | --- | --- |")
    for item, group in report["contract_checks"].items():
        lines.append(
            f"| `{item}` | {'✅' if group['passed'] else '❌'} | {len(group['checks'])} |"
        )
    lines.append("")
    lines.append(
        f"- 全部审计检查通过：`{report['contract_all_passed']}`；失败项：{report['failed_checks'] or '无'}"
    )
    lines.append("")
    lines.append("## 3. 唯一主指标（冻结 `score(...)[\"cv_bench_accuracy\"]`）")
    lines.append("")
    lines.append("| 臂 | cv_bench_accuracy | 未加权 | 样本数 |")
    lines.append("| --- | --- | --- | --- |")
    for arm in ("new", "dense", "corrected_l2_integer", "prompt_plus_corrected_l2_integer"):
        row = report["scores"][arm]
        lines.append(
            f"| {arm} | {_fmt(row['cv_bench_accuracy'])} | "
            f"{_fmt(row['unweighted_sample_accuracy'])} | {row['sample_count']} |"
        )
    lines.append("")
    lines.append("## 4. 配对 bootstrap 与 90% dense 门")
    lines.append("")
    boot = report["paired_bootstrap"]
    lines.append(
        f"- v2 − 修正整数 L2 点估计：{_fmt(boot['delta_point_estimate'])}"
    )
    lines.append(
        f"- 分层 `{boot['strata']}` 95% CI："
        f"[{_fmt(boot['delta_ci95'][0])}, {_fmt(boot['delta_ci95'][1])}]；"
        f"seed {boot['seed']}，repeats {boot['repeats']}"
    )
    lines.append(
        f"- 90% dense 门（现场计算 `0.9 × {_fmt(report['scores']['dense']['cv_bench_accuracy'])}`）："
        f"{_fmt(boot['floor'])}"
    )
    lines.append(
        f"- 绝对缺口：{_fmt(boot['shortfall_absolute'])}；"
        f"相对门：{_fmt(boot['shortfall_relative_to_floor'])}；"
        f"相对 dense：{_fmt(boot['shortfall_relative_to_dense'])}"
    )
    lines.append(
        f"- v2 保留 dense 分数比例：{_fmt(boot['retained_dense_fraction'])}"
    )
    lines.append("")
    lines.append("## 5. 决策门（现场重算）")
    lines.append("")
    lines.append("| 门 | 通过 |")
    lines.append("| --- | --- |")
    for name, passed in report["development_decision"]["gates"].items():
        lines.append(f"| `{name}` | {'是' if passed else '否'} |")
    lines.append("")
    lines.append(
        f"- 最终判定：**`{report['development_decision']['outcome']}`**；"
        f"失败门：{report['development_decision']['failed_gates'] or '无'}"
    )
    lines.append("")
    lines.append("## 6. 布局与交换摘要")
    lines.append("")
    layout = report["layout"]
    lines.append(
        f"- 压缩样本：{layout['compressed_count']}；identity bypass：{layout['bypass_count']}；"
        f"压缩样本中 0 交换：{layout['compressed_zero_swap']}"
    )
    lines.append("")
    lines.append("## 7. 冻结输入字节一致性")
    lines.append("")
    frozen = report["frozen_inputs"]
    lines.append(
        f"- 非空 config pin 现场复核通过：{frozen['verified_count']}；"
        f"不匹配：{frozen['mismatched_pins'] or '无'}；待生成（null）：{frozen['null_pins']}"
    )
    lines.append("")
    lines.append("## 8. 发现")
    lines.append("")
    for finding in report["findings"]:
        lines.append(f"- `{finding['id']}`（{finding['severity']}）：{finding['summary']}")
    lines.append("")
    lines.append("## 9. 限制")
    lines.append("")
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _render_breakdown_table(
    lines: list[str], title: str, data: Mapping[str, Mapping[str, Any]]
) -> None:
    lines.append(f"### {title}")
    lines.append("")
    lines.append("| 组 | n | 正确 | 准确率 |")
    lines.append("| --- | --- | --- | --- |")
    for group, row in data.items():
        lines.append(
            f"| {group} | {row['n']} | {row['correct']} | {_fmt(row['accuracy'], 6)} |"
        )
    lines.append("")


def _render_migration_table(
    lines: list[str], title: str, data: Mapping[str, Any]
) -> None:
    lines.append(f"### {title}")
    lines.append("")
    lines.append(
        f"- 修复（错→对）：{data['repairs']}；回归（对→错）：{data['regressions']}；"
        f"净：{data['net']}；答案变化：{data['answer_changes']}"
    )
    lines.append(
        f"- 保持正确：{data['unchanged_correct']}；保持错误：{data['unchanged_wrong']}"
    )
    lines.append("")
    lines.append("| 组 | n | 修复 | 回归 | 净 | 答案变化 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for group, row in data["by_task"].items():
        lines.append(
            f"| {group} | {row['n']} | {row['repairs']} | {row['regressions']} | "
            f"{row['repairs'] - row['regressions']} | {row['answer_changes']} |"
        )
    lines.append("")


def render_postmortem_markdown(postmortem: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# v2 full2638 标签感知事后尸检（仅诊断/发展用途）")
    lines.append("")
    lines.append(
        "> **诊断/发展专用**：本文件使用官方答案字母计数修复/回归/不变，"
        "**不是门、不是精度估计**；禁止据此调参或选择 v3 规则。"
    )
    lines.append("")
    lines.append(f"- 日期：{postmortem['date']}")
    lines.append(f"- 标记：`{postmortem['marker']}`")
    lines.append(f"- full2638 角色：{postmortem['full2638_role']}")
    lines.append("")
    lines.append("## 1. 各臂总体")
    lines.append("")
    lines.append("| 臂 | 正确/总数 | cv_bench_accuracy |")
    lines.append("| --- | --- | --- |")
    for arm, row in postmortem["correct_counts"].items():
        lines.append(
            f"| {arm} | {row['correct']}/{row['n']} | {_fmt(row['cv_bench_accuracy'])} |"
        )
    lines.append("")
    lines.append("## 2. 逐任务 / 数据源 / 冻结原始 G 分桶")
    lines.append("")
    for arm in ("new", "dense", "corrected_l2_integer"):
        lines.append(f"### 臂：{arm}")
        lines.append("")
        _render_breakdown_table(
            lines, "按 task", postmortem["breakdown_by_arm"][arm]["per_task"]
        )
        _render_breakdown_table(
            lines,
            "按 source_dataset",
            postmortem["breakdown_by_arm"][arm]["per_source_dataset"],
        )
        _render_breakdown_table(
            lines,
            "按冻结原始 G 分桶",
            postmortem["breakdown_by_arm"][arm]["per_original_G_bin"],
        )
    lines.append("## 3. 迁移（修复/回归/不变）")
    lines.append("")
    _render_migration_table(
        lines,
        "v2 相对修正整数 L2",
        postmortem["v2_vs_corrected_l2_integer"],
    )
    _render_migration_table(lines, "v2 相对 dense", postmortem["v2_vs_dense"])
    _render_migration_table(
        lines,
        "修正整数 L2 相对 dense",
        postmortem["corrected_l2_integer_vs_dense"],
    )
    lines.append("## 4. 解析失败（诚实 parse_choice）")
    lines.append("")
    lines.append("| 臂 | 失败数 | 解析率 |")
    lines.append("| --- | --- | --- |")
    for arm, row in postmortem["parse_failures"].items():
        lines.append(
            f"| {arm} | {row['count']} | {_fmt(row['parse_rate'], 6)} |"
        )
    lines.append("")
    lines.append("## 5. 交换计数与 bypass 分解")
    lines.append("")
    swaps = postmortem["swap_count_breakdown"]
    lines.append(
        f"- 压缩样本 {swaps['compressed_count']}：0 交换 {swaps['compressed_zero_swap']}，"
        f"正交换 {swaps['compressed_positive_swap']}"
    )
    lines.append(f"- 交换计数直方图：{swaps['histogram']}")
    bypass = postmortem["bypass_breakdown"]
    lines.append(
        f"- bypass {bypass['count']} 个：G 范围 {bypass['G_min']}..{bypass['G_max']}；"
        f"按 task {bypass['by_task']}；按 source_dataset {bypass['by_source_dataset']}"
    )
    lines.append("")
    lines.append("## 6. 限制")
    lines.append("")
    for limitation in postmortem["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_reports(
    report: Mapping[str, Any],
    postmortem: Mapping[str, Any],
    paths: InputPaths,
) -> None:
    paths.report_json.parent.mkdir(parents=True, exist_ok=True)
    paths.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths.report_md.write_text(render_markdown(report), encoding="utf-8")
    paths.postmortem_json.write_text(
        json.dumps(postmortem, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths.postmortem_md.write_text(
        render_postmortem_markdown(postmortem), encoding="utf-8"
    )


def verify_config_pins(config_path: Path = PREREGISTERED_CONFIG) -> dict[str, Any]:
    """Re-verify every non-null frozen config pin (used after artifact writes)."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    verified: list[dict[str, str]] = []
    mismatched: list[dict[str, str]] = []
    null_pins: list[str] = []
    for name, entry in _config_pin_table(config).items():
        if entry["sha256"] is None:
            null_pins.append(name)
            continue
        path = ROOT / entry["path"]
        observed = sha256_file(path) if path.is_file() else "MISSING"
        record = {"name": name, "path": entry["path"], "sha256": observed}
        if observed == entry["sha256"]:
            verified.append(record)
        else:
            mismatched.append(record)
    return {
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "verified": verified,
        "mismatched": mismatched,
        "null_pins": null_pins,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-full2638", type=Path, default=RAW_FULL2638)
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
    parser.add_argument("--postmortem-json", type=Path, default=POSTMORTEM_JSON)
    parser.add_argument("--postmortem-md", type=Path, default=POSTMORTEM_MD)
    parser.add_argument(
        "--no-write", action="store_true", help="Audit only; do not write reports."
    )
    parser.add_argument(
        "--recheck-pins",
        action="store_true",
        help="Re-verify every frozen v1/v2 config pin and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.recheck_pins:
        result = verify_config_pins(args.preregistered)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if not result["mismatched"] else 2
    paths = InputPaths(
        raw_full2638=args.raw_full2638,
        preregistered=args.preregistered,
        full_manifest=args.full_manifest,
        dense_full_jsonl=args.dense_full_jsonl,
        corrected_l2_integer_jsonl=args.corrected_l2_integer_jsonl,
        prompt_plus_corrected_l2_integer_jsonl=(
            args.prompt_plus_corrected_l2_integer_jsonl
        ),
        report_json=args.report_json,
        report_md=args.report_md,
        postmortem_json=args.postmortem_json,
        postmortem_md=args.postmortem_md,
    )
    bundle = run_audit(paths)
    report = bundle["report"]
    if not args.no_write:
        write_reports(report, bundle["postmortem"], paths)
    print(
        json.dumps(
            {
                "audit_decision": report["audit_decision"],
                "development_decision": report["development_decision"]["outcome"],
                "contract_all_passed": report["contract_all_passed"],
                "failed_checks": report["failed_checks"],
                "raw_sha256": report["raw_full2638"]["sha256_observed"],
                "gpu_runs": report["gpu_runs"],
                "audited_gpu_runs": report["audited_gpu_runs"],
                "audited_samples": report["audited_samples"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["contract_all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
