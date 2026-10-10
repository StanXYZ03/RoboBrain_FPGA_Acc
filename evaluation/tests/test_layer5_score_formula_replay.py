#!/usr/bin/env python3
"""Focused tests for the layer-5 score-formula replay diagnostic.

Covers the two pure score formulas, the cancellation case that separates them,
exact tie behaviour, element-for-element equivalence of the fixed 3x3
anchor/global-fill candidate selector with the historical
``structured_group_indices`` across every frozen reference shape, the
comparison helpers, fail-closed input / resume / summariser validation, and the
vision-only unified replay path on a synthetic CPU vision tower.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation import layer5_score_formula_replay as replay
from evaluation import summarize_layer5_score_formula_replay as summary
from evaluation.hardware_regular_token_selector import FEATURE_DIM
from evaluation.shallow_block_pruning import structured_group_indices

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_JSONL = (
    ROOT
    / "evaluation"
    / "results"
    / "cv_bench_l2_budget64_pilot100_20260927"
    / "shallow_r0250.jsonl"
)


def _hidden(seed: int = 0, groups: int = 4, features: int = FEATURE_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(groups, 4, features))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _manifest_row(sample_id: str, task: str = "Count") -> dict:
    return {"sample_id": sample_id, "task": task}


def _reference_row(
    sample_id: str, *, g: int, k: int, indices: list[int], task: str = "Count"
) -> dict:
    return {
        "sample_id": sample_id,
        "task": task,
        "prune_after_layer": 5,
        "anchor_grid": 3,
        "selector": "l2_anchor",
        "baseline_llm_visual_tokens": g,
        "selected_group_indices": indices,
        "selected_group_indices_sha256": replay.indices_sha256(indices),
        "llm_visual_tokens": k,
    }


def _control_row(
    sample_id: str,
    *,
    g: int,
    k: int,
    indices: list[int],
    reference_indices: list[int],
    reference_sha256: str,
    manifest_sha256: str,
    task: str = "Count",
) -> dict:
    return {
        "sample_id": sample_id,
        "task": task,
        "arm": "control",
        "mode": "control",
        "selection_rule": replay.SELECTION_RULE_CONTROL,
        "prune_after_layer": 5,
        "region_grid": 3,
        "reference_jsonl_sha256": reference_sha256,
        "manifest_sha256": manifest_sha256,
        "reference_G": g,
        "K_C": k,
        "reference_K_C": k,
        "selected_group_indices": indices,
        "selected_group_indices_sha256": replay.indices_sha256(indices),
        "context_group_count": 0,
        "real_group_count": k,
        "output_group_count": k,
        "identity_bypass": False,
        "source_types_all_real": True,
        "reference_selected_group_indices": reference_indices,
    }


def _replay_record(
    sample_id: str,
    *,
    g: int,
    k: int,
    reference_indices: list[int],
    candidate_indices: list[int],
    control_indices: list[int],
    provenance: dict,
    task: str = "Count",
    grid: tuple[int, int] | None = None,
) -> dict:
    grid = grid or (8, 8)
    exact = replay.exact_index_match(candidate_indices, reference_indices)
    return {
        "sample_id": sample_id,
        "task": task,
        "reference_G": g,
        "K_C": k,
        "merged_grid": [grid[0], grid[1]],
        "reference_selected_group_indices": reference_indices,
        "reference_selected_group_indices_sha256": replay.indices_sha256(
            reference_indices
        ),
        "candidate_selected_group_indices": candidate_indices,
        "candidate_selected_group_indices_sha256": replay.indices_sha256(
            candidate_indices
        ),
        "control_selected_group_indices": control_indices,
        "control_selected_group_indices_sha256": replay.indices_sha256(
            control_indices
        ),
        "candidate_equals_reference": exact,
        "candidate_vs_reference_jaccard": replay.jaccard(
            candidate_indices, reference_indices
        ),
        "control_vs_reference_jaccard": replay.jaccard(
            control_indices, reference_indices
        ),
        "candidate_score_summary": {"all_finite": True},
        "control_score_summary": {"all_finite": True},
        "causal_diagnostic_only": replay.CAUSAL_DIAGNOSTIC_ONLY,
        "fpga_deployable_candidate": replay.FPGA_DEPLOYABLE_CANDIDATE,
        "only_variable": replay.ONLY_VARIABLE,
        "identity_bypass": False,
        "uses_context_pooling": False,
        "uses_context_slot_reservation": False,
        "uses_local_candidate_cap": False,
        "uses_b1_trained_wv": False,
        "source_types_all_real": True,
        "control_vs_reference_exact": replay.exact_index_match(
            control_indices, reference_indices
        ),
        "vision_total_seconds": 0.01,
        "model_inference_seconds": 0.01,
        "peak_cuda_bytes": 1024,
        "run": {
            "dtype": "bf16",
            "attn_implementation": "sdpa",
            "llm_decode": False,
            "training": False,
        },
        **provenance,
    }


@contextlib.contextmanager
def _synthetic_inputs(count: int = 100, g: int = 64, k: int = 16):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manifest_path = tmp_path / "manifest.jsonl"
        reference_path = tmp_path / "reference.jsonl"
        control_path = tmp_path / "control.jsonl"
        manifest_rows = [
            _manifest_row(str(index + 1)) for index in range(count)
        ]
        reference_rows = []
        control_rows = []
        for index in range(count):
            sample_id = str(index + 1)
            reference_indices = list(range(k))
            control_indices = list(range(k, 2 * k))
            reference_rows.append(
                _reference_row(
                    sample_id, g=g, k=k, indices=reference_indices
                )
            )
            control_rows.append(
                _control_row(
                    sample_id,
                    g=g,
                    k=k,
                    indices=control_indices,
                    reference_indices=reference_indices,
                    reference_sha256="PENDING",
                    manifest_sha256="PENDING",
                )
            )
        _write_jsonl(manifest_path, manifest_rows)
        _write_jsonl(reference_path, reference_rows)
        manifest_sha = replay.file_sha256(manifest_path)
        reference_sha = replay.file_sha256(reference_path)
        for row in control_rows:
            row["reference_jsonl_sha256"] = reference_sha
            row["manifest_sha256"] = manifest_sha
        _write_jsonl(control_path, control_rows)
        inputs = replay.load_replay_inputs(
            manifest_path, reference_path, control_path, expected_samples=count
        )
        provenance = {
            "manifest_sha256": inputs.manifest_sha256,
            "reference_jsonl_sha256": inputs.reference_sha256,
            "control_jsonl_sha256": inputs.control_sha256,
        }
        yield {
            "dir": tmp_path,
            "manifest": manifest_path,
            "reference": reference_path,
            "control": control_path,
            "inputs": inputs,
            "provenance": provenance,
        }


class ScoreFormulaTest(unittest.TestCase):
    def test_control_formula_matches_norm_of_mean(self) -> None:
        hidden = _hidden(seed=1)
        expected = np.linalg.norm(hidden.mean(axis=1), axis=1)
        np.testing.assert_allclose(replay.control_group_scores(hidden), expected)

    def test_candidate_formula_matches_mean_of_norms(self) -> None:
        hidden = _hidden(seed=2)
        expected = np.linalg.norm(hidden, axis=2).mean(axis=1)
        np.testing.assert_allclose(replay.historical_group_scores(hidden), expected)

    def test_cancellation_case_distinguishes_formulas(self) -> None:
        vector = np.zeros(FEATURE_DIM, dtype=np.float64)
        vector[0] = 3.0
        hidden = np.zeros((1, 4, FEATURE_DIM), dtype=np.float64)
        hidden[0, 0] = vector
        hidden[0, 1] = -vector
        hidden[0, 2] = vector
        hidden[0, 3] = -vector
        self.assertAlmostEqual(float(replay.control_group_scores(hidden)[0]), 0.0)
        self.assertAlmostEqual(float(replay.historical_group_scores(hidden)[0]), 3.0)

    def test_non_finite_hidden_is_rejected(self) -> None:
        hidden = _hidden(seed=3, groups=2)
        hidden[0, 0, 0] = np.nan
        with self.assertRaises(Exception):
            replay.control_group_scores(hidden)
        with self.assertRaises(Exception):
            replay.historical_group_scores(hidden)


class AnchorFillSelectionTest(unittest.TestCase):
    def test_exact_tie_breaks_favour_lower_index(self) -> None:
        score = np.zeros(6 * 6, dtype=np.float64)
        selected = replay.select_groups_by_score(score, 6, 6, budget=16)
        anchors = [
            row * 6 + col
            for row in (0, 2, 4)
            for col in (0, 2, 4)
        ]
        # The 9 anchors are the lowest index of every 3x3 region, then the
        # lowest remaining global indices fill to K.
        anchor_set = set(anchors)
        fill = [group for group in range(36) if group not in anchor_set][: 16 - 9]
        self.assertEqual(selected.tolist(), sorted(anchor_set | set(fill)))

    def test_matches_historical_structured_group_indices_across_reference_shapes(
        self,
    ) -> None:
        rows = [
            json.loads(line)
            for line in REFERENCE_JSONL.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        pairs = sorted(
            {
                (
                    int(row["image_grid_thw"][1]) // 2,
                    int(row["image_grid_thw"][2]) // 2,
                    len(row["selected_group_indices"]),
                )
                for row in rows
            }
        )
        rng = np.random.default_rng(20261004)
        checked = 0
        for grid_h, grid_w, k in pairs:
            group_count = grid_h * grid_w
            self.assertGreaterEqual(k, replay.MIN_REFERENCE_K)
            retention = (k - 0.5) / group_count
            for trial in range(3):
                if trial == 0:
                    score = rng.normal(size=group_count)
                elif trial == 1:
                    score = np.round(rng.normal(size=group_count), 1)
                else:
                    score = np.zeros(group_count)
                reference = structured_group_indices(
                    torch.tensor(score, dtype=torch.float32),
                    grid_h,
                    grid_w,
                    retention,
                    anchor_grid=3,
                ).tolist()
                candidate = replay.select_groups_by_score(
                    score, grid_h, grid_w, budget=k
                ).tolist()
                self.assertEqual(candidate, reference, msg=f"{grid_h}x{grid_w} K={k}")
                checked += 1
        self.assertEqual(checked, len(pairs) * 3)
        self.assertGreaterEqual(len(pairs), 20)

    def test_budget_below_anchor_count_is_rejected(self) -> None:
        score = np.zeros(36)
        with self.assertRaises(Exception):
            replay.select_groups_by_score(score, 6, 6, budget=8)

    def test_grid_mismatch_is_rejected(self) -> None:
        with self.assertRaises(Exception):
            replay.select_groups_by_score(np.zeros(35), 6, 6, budget=16)


class ComparisonHelperTest(unittest.TestCase):
    def test_exact_and_jaccard(self) -> None:
        self.assertTrue(replay.exact_index_match([1, 2], [1, 2]))
        self.assertFalse(replay.exact_index_match([1, 2], [2, 1]))
        self.assertAlmostEqual(replay.jaccard([1, 2, 3], [2, 3, 4]), 0.5)
        self.assertAlmostEqual(replay.jaccard([], []), 1.0)
        self.assertAlmostEqual(replay.jaccard([1], [2]), 0.0)

    def test_sha_matches_frozen_reference_convention(self) -> None:
        row = json.loads(
            REFERENCE_JSONL.read_text(encoding="utf-8").splitlines()[0]
        )
        self.assertEqual(
            replay.indices_sha256(sorted(row["selected_group_indices"])),
            row["selected_group_indices_sha256"],
        )

    def test_real_reference_contract_loads(self) -> None:
        rows = [
            json.loads(line)
            for line in REFERENCE_JSONL.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            normalized = replay.normalize_reference_record(
                row, {"sample_id": row["sample_id"], "task": row["task"]}
            )
            self.assertGreaterEqual(normalized["K_C"], replay.MIN_REFERENCE_K)
            self.assertLess(normalized["K_C"], normalized["G"])


class ProvenanceTest(unittest.TestCase):
    def test_valid_inputs_load(self) -> None:
        with _synthetic_inputs(count=5) as bundle:
            self.assertEqual(len(bundle["inputs"].manifest_rows), 5)

    def test_wrong_expected_sample_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest = tmp_path / "m.jsonl"
            reference = tmp_path / "r.jsonl"
            control = tmp_path / "c.jsonl"
            _write_jsonl(manifest, [_manifest_row("1")])
            _write_jsonl(
                reference, [_reference_row("1", g=64, k=16, indices=list(range(16)))]
            )
            _write_jsonl(
                control,
                [
                    _control_row(
                        "1",
                        g=64,
                        k=16,
                        indices=list(range(16, 32)),
                        reference_indices=list(range(16)),
                        reference_sha256="x",
                        manifest_sha256="y",
                    )
                ],
            )
            with self.assertRaises(replay.InputContractError):
                replay.load_replay_inputs(
                    manifest, reference, control, expected_samples=100
                )

    def test_duplicate_reference_id_is_rejected(self) -> None:
        with _synthetic_inputs(count=3) as bundle:
            rows = replay.read_jsonl(bundle["reference"], "reference")
            _write_jsonl(bundle["reference"], rows + [rows[0]])
            with self.assertRaises(replay.InputContractError):
                replay.load_replay_inputs(
                    bundle["manifest"], bundle["reference"], bundle["control"]
                )

    def test_extra_reference_id_is_rejected(self) -> None:
        with _synthetic_inputs(count=3) as bundle:
            rows = replay.read_jsonl(bundle["reference"], "reference")
            rows.append(_reference_row("999", g=64, k=16, indices=list(range(16))))
            _write_jsonl(bundle["reference"], rows)
            with self.assertRaises(replay.InputContractError):
                replay.load_replay_inputs(
                    bundle["manifest"], bundle["reference"], bundle["control"]
                )

    def test_task_mismatch_is_rejected(self) -> None:
        with _synthetic_inputs(count=2) as bundle:
            rows = replay.read_jsonl(bundle["reference"], "reference")
            rows[0]["task"] = "Depth"
            _write_jsonl(bundle["reference"], rows)
            with self.assertRaises(replay.InputContractError):
                replay.load_replay_inputs(
                    bundle["manifest"], bundle["reference"], bundle["control"]
                )

    def test_reference_hash_mismatch_is_rejected(self) -> None:
        with _synthetic_inputs(count=2) as bundle:
            rows = replay.read_jsonl(bundle["reference"], "reference")
            rows[0]["selected_group_indices_sha256"] = "0" * 64
            _write_jsonl(bundle["reference"], rows)
            with self.assertRaises(replay.InputContractError):
                replay.load_replay_inputs(
                    bundle["manifest"], bundle["reference"], bundle["control"]
                )

    def test_reference_index_out_of_range_is_rejected(self) -> None:
        with _synthetic_inputs(count=2) as bundle:
            rows = replay.read_jsonl(bundle["reference"], "reference")
            indices = list(range(15)) + [64]
            rows[0]["selected_group_indices"] = indices
            rows[0]["selected_group_indices_sha256"] = replay.indices_sha256(indices)
            _write_jsonl(bundle["reference"], rows)
            with self.assertRaises(replay.InputContractError):
                replay.load_replay_inputs(
                    bundle["manifest"], bundle["reference"], bundle["control"]
                )

    def test_reference_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest = tmp_path / "m.jsonl"
            reference = tmp_path / "r.jsonl"
            control = tmp_path / "c.jsonl"
            _write_jsonl(manifest, [_manifest_row("1")])
            indices = list(range(16))
            _write_jsonl(
                reference,
                [_reference_row("1", g=16, k=16, indices=indices)],
            )
            _write_jsonl(
                control,
                [
                    _control_row(
                        "1",
                        g=16,
                        k=16,
                        indices=indices,
                        reference_indices=indices,
                        reference_sha256="x",
                        manifest_sha256="y",
                    )
                ],
            )
            with self.assertRaises(replay.UnexpectedIdentityBypassError):
                replay.load_replay_inputs(
                    manifest, reference, control, expected_samples=1
                )

    def test_control_embedded_reference_mismatch_is_rejected(self) -> None:
        with _synthetic_inputs(count=2) as bundle:
            rows = replay.read_jsonl(bundle["control"], "control")
            rows[0]["reference_selected_group_indices"] = list(range(1, 17))
            _write_jsonl(bundle["control"], rows)
            with self.assertRaises(replay.InputContractError):
                replay.load_replay_inputs(
                    bundle["manifest"], bundle["reference"], bundle["control"]
                )


class ReplayRecordValidationTest(unittest.TestCase):
    def _record(self, bundle, sample_id="1", candidate=None):
        inputs = bundle["inputs"]
        reference = inputs.reference_map[sample_id]
        control = inputs.control_map[sample_id]
        reference_indices = list(reference["group_indices"])
        return _replay_record(
            sample_id,
            g=int(reference["G"]),
            k=int(reference["K_C"]),
            reference_indices=reference_indices,
            candidate_indices=list(candidate or reference_indices),
            control_indices=list(control["group_indices"]),
            provenance=bundle["provenance"],
        )

    def test_valid_record_passes(self) -> None:
        with _synthetic_inputs(count=2) as bundle:
            record = self._record(bundle)
            replay.validate_replay_record(
                record,
                bundle["inputs"].reference_map["1"],
                bundle["inputs"].control_map["1"],
                provenance=bundle["provenance"],
            )

    def test_stale_provenance_is_rejected(self) -> None:
        with _synthetic_inputs(count=2) as bundle:
            record = self._record(bundle)
            record["manifest_sha256"] = "0" * 64
            with self.assertRaises(replay.InputContractError):
                replay.validate_replay_record(
                    record,
                    bundle["inputs"].reference_map["1"],
                    bundle["inputs"].control_map["1"],
                    provenance=bundle["provenance"],
                )

    def test_inconsistent_candidate_flag_is_rejected(self) -> None:
        with _synthetic_inputs(count=2) as bundle:
            record = self._record(bundle)
            record["candidate_equals_reference"] = False
            with self.assertRaises(replay.InputContractError):
                replay.validate_replay_record(
                    record,
                    bundle["inputs"].reference_map["1"],
                    bundle["inputs"].control_map["1"],
                    provenance=bundle["provenance"],
                )

    def test_control_mismatch_in_record_is_rejected(self) -> None:
        with _synthetic_inputs(count=2) as bundle:
            record = self._record(bundle)
            record["control_selected_group_indices"] = list(range(16))  # == reference
            record["control_selected_group_indices_sha256"] = replay.indices_sha256(
                list(range(16))
            )
            with self.assertRaises(replay.InputContractError):
                replay.validate_replay_record(
                    record,
                    bundle["inputs"].reference_map["1"],
                    bundle["inputs"].control_map["1"],
                    provenance=bundle["provenance"],
                )


class ResumeTest(unittest.TestCase):
    def test_missing_file_yields_empty_set(self) -> None:
        with _synthetic_inputs(count=2) as bundle:
            done = replay.resume_completed_ids(
                bundle["dir"] / "missing.jsonl",
                ["1", "2"],
                inputs=bundle["inputs"],
                provenance=bundle["provenance"],
            )
            self.assertEqual(done, set())

    def test_valid_records_resume(self) -> None:
        with _synthetic_inputs(count=2) as bundle:
            record = _replay_record(
                "1",
                g=64,
                k=16,
                reference_indices=list(range(16)),
                candidate_indices=list(range(16)),
                control_indices=list(range(16, 32)),
                provenance=bundle["provenance"],
            )
            path = bundle["dir"] / "out.jsonl"
            _write_jsonl(path, [record])
            done = replay.resume_completed_ids(
                path,
                ["1", "2"],
                inputs=bundle["inputs"],
                provenance=bundle["provenance"],
            )
            self.assertEqual(done, {"1"})

    def test_stale_provenance_resume_aborts(self) -> None:
        with _synthetic_inputs(count=2) as bundle:
            record = _replay_record(
                "1",
                g=64,
                k=16,
                reference_indices=list(range(16)),
                candidate_indices=list(range(16)),
                control_indices=list(range(16, 32)),
                provenance={**bundle["provenance"], "control_jsonl_sha256": "0" * 64},
            )
            path = bundle["dir"] / "out.jsonl"
            _write_jsonl(path, [record])
            with self.assertRaises(replay.InputContractError):
                replay.resume_completed_ids(
                    path,
                    ["1", "2"],
                    inputs=bundle["inputs"],
                    provenance=bundle["provenance"],
                )

    def test_out_of_range_resume_id_aborts(self) -> None:
        with _synthetic_inputs(count=2) as bundle:
            record = _replay_record(
                "3",
                g=64,
                k=16,
                reference_indices=list(range(16)),
                candidate_indices=list(range(16)),
                control_indices=list(range(16, 32)),
                provenance=bundle["provenance"],
            )
            path = bundle["dir"] / "out.jsonl"
            _write_jsonl(path, [record])
            with self.assertRaises(replay.InputContractError):
                replay.resume_completed_ids(
                    path,
                    ["1", "2"],
                    inputs=bundle["inputs"],
                    provenance=bundle["provenance"],
                )


class SummarizerTest(unittest.TestCase):
    def _write_replay(self, bundle, *, mismatch: bool = False) -> Path:
        records = []
        for sample_id in [str(index + 1) for index in range(100)]:
            reference_indices = list(range(16))
            candidate_indices = list(reference_indices)
            if mismatch and sample_id == "7":
                candidate_indices = list(range(16, 32))
            records.append(
                _replay_record(
                    sample_id,
                    g=64,
                    k=16,
                    reference_indices=reference_indices,
                    candidate_indices=candidate_indices,
                    control_indices=list(range(16, 32)),
                    provenance=bundle["provenance"],
                )
            )
        path = bundle["dir"] / "replay.jsonl"
        _write_jsonl(path, records)
        return path

    def test_full_match_decision_and_outputs(self) -> None:
        with _synthetic_inputs(count=100) as bundle:
            path = self._write_replay(bundle)
            records = summary._load_records(
                path, bundle["inputs"], provenance=bundle["provenance"]
            )
            report = summary.build_report(
                bundle["inputs"], records, provenance=bundle["provenance"]
            )
            self.assertEqual(report["decision"], summary.DECISION_FULL)
            self.assertEqual(report["exact_candidate_reference_matches"], 100)
            self.assertEqual(report["candidate_vs_reference"]["mean"], 1.0)
            self.assertEqual(report["candidate_vs_reference"]["min"], 1.0)
            out_json = bundle["dir"] / "r.json"
            out_csv = bundle["dir"] / "r.csv"
            out_md = bundle["dir"] / "r.md"
            summary.write_json(out_json, report)
            summary.write_csv(out_csv, report)
            out_md.write_text(summary.render_markdown(report), encoding="utf-8")
            self.assertIn(summary.DECISION_FULL, out_md.read_text(encoding="utf-8"))
            self.assertEqual(len(out_csv.read_text(encoding="utf-8").splitlines()), 101)

    def test_partial_match_decision_lists_mismatch(self) -> None:
        with _synthetic_inputs(count=100) as bundle:
            path = self._write_replay(bundle, mismatch=True)
            records = summary._load_records(
                path, bundle["inputs"], provenance=bundle["provenance"]
            )
            report = summary.build_report(
                bundle["inputs"], records, provenance=bundle["provenance"]
            )
            self.assertEqual(report["decision"], summary.DECISION_PARTIAL)
            self.assertEqual(report["exact_candidate_reference_matches"], 99)
            self.assertEqual(report["mismatch_ids"], ["7"])

    def test_missing_record_fails_closed(self) -> None:
        with _synthetic_inputs(count=100) as bundle:
            path = self._write_replay(bundle)
            rows = replay.read_jsonl(path, "replay")
            _write_jsonl(path, rows[:-1])
            with self.assertRaises(replay.InputContractError):
                summary._load_records(
                    path, bundle["inputs"], provenance=bundle["provenance"]
                )

    def test_stale_record_provenance_fails_closed(self) -> None:
        with _synthetic_inputs(count=100) as bundle:
            path = self._write_replay(bundle)
            rows = replay.read_jsonl(path, "replay")
            rows[0]["manifest_sha256"] = "0" * 64
            _write_jsonl(path, rows)
            with self.assertRaises(replay.InputContractError):
                summary._load_records(
                    path, bundle["inputs"], provenance=bundle["provenance"]
                )

    def test_non_bf16_run_metadata_fails_closed(self) -> None:
        with _synthetic_inputs(count=100) as bundle:
            path = self._write_replay(bundle)
            rows = replay.read_jsonl(path, "replay")
            rows[0]["run"]["dtype"] = "fp16"
            _write_jsonl(path, rows)
            with self.assertRaises(replay.InputContractError):
                summary._load_records(
                    path, bundle["inputs"], provenance=bundle["provenance"]
                )


# --- synthetic CPU vision tower (mirrors the frozen forced-index tests) -----

OUT_HIDDEN = 2560
DEPTH = 24
DEEPSTACK_INDEXES = [5, 11, 17]
FAKE_GRID = 12  # G = 36 merged groups


class _FakeBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden, cu_seqlens=None, position_embeddings=None):
        output = hidden * self.scale
        if position_embeddings is None:
            return output
        cos, sin = position_embeddings
        rotary = torch.cat([cos, sin], dim=-1)
        if rotary.shape[-1] < self.hidden:
            rotary = F.pad(rotary, (0, self.hidden - rotary.shape[-1]))
        else:
            rotary = rotary[..., : self.hidden]
        return output + rotary


class _GroupMerger(nn.Module):
    def __init__(self, hidden: int, out_hidden: int, salt: int = 0) -> None:
        super().__init__()
        fan_in = hidden * 4
        self.proj = nn.Linear(fan_in, out_hidden, bias=False)
        weight = (
            torch.arange(out_hidden * fan_in, dtype=torch.float32).reshape(
                out_hidden, fan_in
            )
            + salt
        ) % 17 - 8.0
        with torch.no_grad():
            self.proj.weight.copy_(weight / fan_in)

    def forward(self, hidden):
        return self.proj(hidden.reshape(-1, hidden.shape[-1] * 4))


class _FakeVisual(nn.Module):
    dtype = torch.float32
    spatial_merge_size = 2
    deepstack_visual_indexes = list(DEEPSTACK_INDEXES)

    def __init__(self, hidden: int = FEATURE_DIM, depth: int = DEPTH) -> None:
        super().__init__()
        self.patch_embed = nn.Identity()
        self.blocks = nn.ModuleList(_FakeBlock(hidden) for _ in range(depth))
        self.deepstack_merger_list = nn.ModuleList(
            _GroupMerger(hidden, OUT_HIDDEN, salt=index)
            for index in range(len(DEEPSTACK_INDEXES))
        )
        self.merger = _GroupMerger(hidden, OUT_HIDDEN, salt=99)
        self.hidden = hidden

    def fast_pos_embed_interpolate(self, grid):
        return torch.zeros(int(grid.prod().item()), self.hidden)

    def rot_pos_emb(self, grid):
        count = int(grid.prod().item())
        base = torch.arange(count, dtype=torch.float32).unsqueeze(1)
        return torch.cat([base * 0.01, base * 0.03], dim=1)


def _pixels(grid: int) -> torch.Tensor:
    count = grid * grid
    base = torch.arange(count, dtype=torch.float32).unsqueeze(1)
    feature = torch.arange(FEATURE_DIM, dtype=torch.float32).unsqueeze(0)
    return torch.sin(0.01 * base + 0.001 * feature)


def _grid(grid: int) -> torch.Tensor:
    return torch.tensor([[1, grid, grid]])


class VisionReplayPathTest(unittest.TestCase):
    def test_candidate_path_uses_historical_scores_and_frozen_selector(self) -> None:
        visual = _FakeVisual()
        grid = _grid(FAKE_GRID)
        pixel_values = _pixels(FAKE_GRID)
        budget = 25
        with torch.inference_mode():
            result = replay.forward_layer5_score_formula_replay_vision(
                visual, pixel_values, grid, budget=budget
            )
        merged_h, merged_w = replay.merged_grid_from_thw(grid)
        hidden = (
            result.forward.cut_hidden.detach()
            .to(torch.float64)
            .reshape(result.forward.full_llm_visual_tokens, 4, FEATURE_DIM)
            .numpy()
        )
        expected_candidate = replay.select_groups_by_score(
            replay.historical_group_scores(hidden),
            merged_h,
            merged_w,
            budget=budget,
        ).tolist()
        self.assertEqual(
            result.candidate_selection.group_indices.tolist(), expected_candidate
        )
        self.assertEqual(len(expected_candidate), budget)
        self.assertEqual(
            result.forward.real_group_count, budget
        )
        self.assertFalse(result.forward.identity_bypass)
        self.assertTrue(np.isfinite(result.candidate_scores).all())
        self.assertTrue(np.isfinite(result.control_scores).all())
        self.assertEqual(result.forward.vision_total_seconds, (
            result.forward.vision_preamble_seconds
            + result.forward.cut_blocks_seconds
            + result.forward.selector_cpu_seconds
            + result.forward.remaining_vision_seconds
        ))

    def test_identity_bypass_is_rejected(self) -> None:
        visual = _FakeVisual()
        grid = _grid(FAKE_GRID)
        # G == budget triggers the control identity bypass: the frozen reference
        # never bypasses, so the replay must abort.
        with self.assertRaises(replay.UnexpectedIdentityBypassError):
            replay.forward_layer5_score_formula_replay_vision(
                visual, _pixels(FAKE_GRID), grid, budget=FAKE_GRID * FAKE_GRID // 4
            )


if __name__ == "__main__":
    unittest.main()
