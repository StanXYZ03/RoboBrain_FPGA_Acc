#!/usr/bin/env python3
"""CPU-only focused tests for the same-image dual-resolution summariser.

Synthetic 4-sample inputs live in a temporary directory and small count
overrides are used so the real frozen artifacts/configs are never modified.
The tests pin the fail-closed four-file contract (missing file / id / arm /
budget / G-threshold / hash / gate-report contamination), the deterministic
stratified paired bootstrap, both decision branches, and the JSON / CSV /
Chinese-Markdown rendering.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import (  # noqa: E402
    summarize_same_image_resolution_counterfactual as summ,
)
from evaluation.hardware_regular_b1_corrected_l2 import array_sha256  # noqa: E402

MANIFEST_SHA = "a" * 64
N = 4

#: (sample_id, task, original_G_bin, stratum, G_low, G_high)
PLAN = (
    ("s0", "Count", "G<=512", "G<=512::Count::integer_correct", 64, 520),
    ("s1", "Relation", "G<=512", "G<=512::Relation::integer_wrong", 80, 1024),
    ("s2", "Count", "G>512", "G>512::Count::integer_correct", 96, 2048),
    ("s3", "Relation", "G>512", "G>512::Relation::integer_wrong", 128, 3072),
)
LOW_G = {sid: low for sid, _, _, _, low, _ in PLAN}
HIGH_G = {sid: high for sid, _, _, _, _, high in PLAN}
STRATUM = {sid: stratum for sid, _, _, stratum, _, _ in PLAN}
GATE_BIN = {sid: gate for sid, _, gate, _, _, _ in PLAN}
TASK = {sid: task for sid, task, _, _, _, _ in PLAN}


def _manifest_rows() -> list[dict]:
    return [
        {
            "sample_id": sid,
            "task": task,
            "type": "2D",
            "source": "ADE20K",
            "source_dataset": "ADE20K Validation Set",
            "image": f"/images/{sid}.image",
            "prompt": "How many?",
            "choices": ["alpha", "beta", "gamma", "delta"],
            "answer": "(A)",
            "max_new_tokens": 16,
            "resolution_counterfactual": {
                "design": summ.DESIGN,
                "split": "selected",
                "stratum": stratum,
                "selection_rank": index,
                "original_G": low,
                "original_G_bin": gate,
                "original_integer_correct": index % 2 == 0,
                "selection_seed": summ.BOOTSTRAP_SEED,
            },
        }
        for index, (sid, task, gate, stratum, low, _) in enumerate(PLAN)
    ]


def _arm_record(
    arm: str,
    label: str,
    sid: str,
    correct: bool,
    *,
    manifest_sha: str = MANIFEST_SHA,
) -> dict:
    g = LOW_G[sid] if label == "low" else HIGH_G[sid]
    full_patch = g * summ.MERGE_UNIT
    # Genuine single-image grid with grid[1]*grid[2] == full_patch_tokens.
    grid = [1, summ.MERGE_UNIT, g]
    if g <= summ.BUDGET:
        bypass = True
        selected = list(range(g))
    else:
        bypass = False
        selected = list(range(summ.BUDGET))
    min_pixels, max_pixels = summ.PIXEL_BUDGETS[label]
    row = {
        "sample_id": sid,
        "task": TASK[sid],
        "source": "ADE20K",
        "source_dataset": "ADE20K Validation Set",
        "arm": arm,
        "score_mode": summ.SCORE_MODE,
        "integration_score_mode": summ.SCORE_MODE,
        "prune_after_layer": summ.PRUNE_AFTER_LAYER,
        "cutpoint": summ.CUTPOINT,
        "pixel_budget_label": label,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "image_grid_thw": grid,
        "full_patch_tokens": full_patch,
        "full_llm_visual_tokens": g,
        "output_group_count": len(selected),
        "identity_bypass": bypass,
        "selected_group_indices": selected,
        "selected_group_indices_sha256": array_sha256(
            np.asarray(selected, dtype=np.int64)
        ),
        "output_text": "(A)" if correct else "(B)",
        "requested_budget": summ.BUDGET,
        "region_grid": summ.REGION_GRID,
        "local_candidate_cap": summ.LOCAL_CANDIDATE_CAP,
        "do_sample": False,
        "decoding": summ.DECODING,
        "max_new_tokens": summ.MAX_NEW_TOKENS,
        "manifest_sha256": manifest_sha,
        "checkpoint_sha256": None,
        "wv_sha256": None,
        "wv_scale_sha256": None,
        "deployment_seed": None,
        "run": {
            "arm": arm,
            "budget": summ.BUDGET,
            "region_grid": summ.REGION_GRID,
            "local_candidate_cap": summ.LOCAL_CANDIDATE_CAP,
            "prune_after_layer": summ.PRUNE_AFTER_LAYER,
            "manifest_sha256": manifest_sha,
            "do_sample": False,
            "dtype": summ.RUN_DTYPE_LABEL,
            "same_image_resolution_counterfactual": True,
            "pixel_budget": {
                "design": summ.DESIGN,
                "label": label,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            },
        },
    }
    if arm == summ.ARM_PROMPT_INTEGER:
        row["checkpoint_sha256"] = summ.CHECKPOINT_SHA256
        row["wv_sha256"] = summ.WV_SHA256
        row["wv_scale_sha256"] = summ.WV_SCALE_SHA256
        row["deployment_seed"] = summ.DEPLOYMENT_SEED
    row["record_sha256"] = summ.record_sha256(row)
    return row


def _gate_report(manifest_sha: str = MANIFEST_SHA) -> dict:
    return {
        "design": summ.DESIGN,
        "mode": "processor",
        "authoritative": True,
        "model_was_run": False,
        "gate_threshold": summ.GATE_THRESHOLD,
        "manifest": {"path": "m", "sha256": manifest_sha, "count": N},
        "final_manifest": {"path": "s", "sha256": manifest_sha, "count": N},
        "plan": {
            "accepted": [sid for sid, *_ in PLAN],
            "accepted_count": N,
            "rejected": [],
            "replaced": [],
            "unresolved": [],
            "resolved": True,
        },
        "measurements": [
            {"sample_id": sid, "G_low": LOW_G[sid], "G_high": HIGH_G[sid]}
            for sid, *_ in PLAN
        ],
        "contract_checks": [{"name": "ok", "ok": True}],
    }


class SyntheticCase:
    """Writes a full set of synthetic inputs into a temp directory."""

    def __init__(self, tmp: Path, correctness: dict[str, dict[str, bool]] | None = None):
        self.tmp = tmp
        self.correctness = correctness or {
            sid: {
                "low_corrected_l2_integer": True,
                "high_corrected_l2_integer": True,
                "low_prompt_plus_corrected_l2_integer": True,
                "high_prompt_plus_corrected_l2_integer": True,
            }
            for sid, *_ in PLAN
        }
        self.manifest_rows = _manifest_rows()
        self.gate_report = _gate_report()

    def _write(self, name: str, rows: list[dict]) -> Path:
        path = self.tmp / name
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return path

    def _arm_rows(self, arm: str, label: str) -> list[dict]:
        return [
            _arm_record(
                arm,
                label,
                sid,
                self.correctness[sid][f"{label}_{arm}"],
            )
            for sid, *_ in PLAN
        ]

    def paths(self) -> summ.InputPaths:
        config = self.tmp / "config.json"
        config.write_text("{}", encoding="utf-8")
        gate = self.tmp / "gate.json"
        gate.write_text(json.dumps(self.gate_report), encoding="utf-8")
        return summ.InputPaths(
            preregistered=config,
            separated_manifest=self._write("manifest.jsonl", self.manifest_rows),
            gate_separation_report=gate,
            arm_integer_low_jsonl=self._write(
                "int_low.jsonl", self._arm_rows(summ.ARM_INTEGER, "low")
            ),
            arm_integer_high_jsonl=self._write(
                "int_high.jsonl", self._arm_rows(summ.ARM_INTEGER, "high")
            ),
            arm_prompt_low_jsonl=self._write(
                "prompt_low.jsonl", self._arm_rows(summ.ARM_PROMPT_INTEGER, "low")
            ),
            arm_prompt_high_jsonl=self._write(
                "prompt_high.jsonl",
                self._arm_rows(summ.ARM_PROMPT_INTEGER, "high"),
            ),
        )


class ContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.case = SyntheticCase(self.tmp)
        self.paths = self.case.paths()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _analyze(self, paths: summ.InputPaths | None = None, **kwargs):
        paths = paths or self.paths
        manifest_rows = [
            json.loads(line)
            for line in paths.separated_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        gate = json.loads(paths.gate_separation_report.read_text(encoding="utf-8"))
        arm_rows = {
            key: [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for key, path in summ.arm_paths(paths).items()
        }
        return summ.analyze(
            manifest_rows=manifest_rows,
            arm_rows=arm_rows,
            gate_report=gate,
            expected_sample_count=N,
            manifest_sha256=MANIFEST_SHA,
            **kwargs,
        )

    def _run(self, paths: summ.InputPaths | None = None, **kwargs):
        return summ.run(
            paths or self.paths,
            output_json=self.tmp / "report.json",
            output_csv=self.tmp / "report.csv",
            output_md=self.tmp / "report.md",
            verify_pins=False,
            expected_sample_count=N,
            manifest_sha256=MANIFEST_SHA,
            bootstrap_repeats=64,
            **kwargs,
        )

    def _mutate(
        self,
        key: str,
        mutate,
        *,
        paths: summ.InputPaths | None = None,
    ) -> summ.InputPaths:
        paths = paths or self.paths
        path = summ.arm_paths(paths)[key]
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        mutate(rows)
        for row in rows:
            if "record_sha256" in row:
                row["record_sha256"] = summ.record_sha256(row)
        new_path = self.tmp / f"mutated_{key.replace('::', '_')}.jsonl"
        new_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        key_to_attribute = {
            f"{summ.ARM_INTEGER}::low": "arm_integer_low_jsonl",
            f"{summ.ARM_INTEGER}::high": "arm_integer_high_jsonl",
            f"{summ.ARM_PROMPT_INTEGER}::low": "arm_prompt_low_jsonl",
            f"{summ.ARM_PROMPT_INTEGER}::high": "arm_prompt_high_jsonl",
        }
        overrides = {key_to_attribute[key]: new_path}
        return summ.InputPaths(**{**paths.__dict__, **overrides})

    def _arm_indices(self, paths: summ.InputPaths | None = None):
        paths = paths or self.paths
        return {
            key: {
                str(row["sample_id"]): row
                for row in (
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            }
            for key, path in summ.arm_paths(paths).items()
        }

    def test_missing_arm_file_fails_closed_without_writing(self) -> None:
        broken = summ.InputPaths(
            **{**self.paths.__dict__, "arm_prompt_high_jsonl": self.tmp / "absent.jsonl"}
        )
        code, report = self._run(paths=broken)
        self.assertEqual(code, 2)
        self.assertIsNone(report)
        for suffix in ("report.json", "report.csv", "report.md"):
            self.assertFalse((self.tmp / suffix).exists(), suffix)

    def test_missing_gate_report_fails_closed(self) -> None:
        broken = summ.InputPaths(
            **{**self.paths.__dict__, "gate_separation_report": self.tmp / "nope.json"}
        )
        code, report = self._run(paths=broken)
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_duplicate_sample_id_fails_closed(self) -> None:
        path = self.paths.arm_integer_low_jsonl
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows.append(dict(rows[0]))
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        code, report = self._run()
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_id_set_mismatch_fails_closed(self) -> None:
        broken = self._mutate(
            f"{summ.ARM_INTEGER}::low",
            lambda rows: rows[0].update({"sample_id": "not-in-manifest"}),
        )
        code, report = self._run(paths=broken)
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_manifest_sha_contamination_fails_closed(self) -> None:
        wrong = "b" * 64

        def _rewrite(rows):
            for row in rows:
                row["manifest_sha256"] = wrong

        broken = self._mutate(f"{summ.ARM_INTEGER}::low", _rewrite)
        code, report = self._run(paths=broken)
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_runner_schema_without_redundant_source_fields_passes(self) -> None:
        paths = self.paths
        for key in summ.arm_paths(paths):
            def _drop_source(rows):
                for row in rows:
                    row.pop("source", None)
                    row.pop("source_dataset", None)

            paths = self._mutate(key, _drop_source, paths=paths)
        code, report = self._run(paths=paths)
        self.assertEqual(code, 0)
        self.assertIsNotNone(report)

    def test_optional_source_field_mismatch_fails_closed(self) -> None:
        broken = self._mutate(
            f"{summ.ARM_INTEGER}::low",
            lambda rows: rows[0].update({"source_dataset": "wrong-source"}),
        )
        code, report = self._run(paths=broken)
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_tampered_record_hash_fails_closed(self) -> None:
        path = self.paths.arm_prompt_high_jsonl
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows[0]["output_text"] = "(B)"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        code, report = self._run()
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_wrong_arm_fails_closed(self) -> None:
        broken = self._mutate(
            f"{summ.ARM_INTEGER}::high",
            lambda rows: [row.update({"arm": summ.ARM_PROMPT_INTEGER}) for row in rows],
        )
        code, report = self._run(paths=broken)
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_wrong_pixel_budget_label_fails_closed(self) -> None:
        broken = self._mutate(
            f"{summ.ARM_INTEGER}::low",
            lambda rows: [row.update({"pixel_budget_label": "high"}) for row in rows],
        )
        code, report = self._run(paths=broken)
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_wrong_pixel_bounds_fail_closed(self) -> None:
        broken = self._mutate(
            f"{summ.ARM_PROMPT_INTEGER}::high",
            lambda rows: [row.update({"max_pixels": 16777215}) for row in rows],
        )
        code, report = self._run(paths=broken)
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_wrong_budget_grid_cap_fail_closed(self) -> None:
        for field, value in (
            ("requested_budget", 64),
            ("region_grid", 4),
            ("local_candidate_cap", 8),
        ):
            with self.subTest(field=field):
                broken = self._mutate(
                    f"{summ.ARM_INTEGER}::low",
                    lambda rows, f=field, v=value: [
                        row.update({f: v}) for row in rows
                    ],
                )
                code, report = self._run(paths=broken)
                self.assertEqual(code, 2)
                self.assertIsNone(report)

    def test_wrong_cutpoint_fails_closed(self) -> None:
        broken = self._mutate(
            f"{summ.ARM_INTEGER}::low",
            lambda rows: [
                row.update({"cutpoint": "B1_after_vision_block_1"}) for row in rows
            ],
        )
        code, report = self._run(paths=broken)
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_wrong_decoding_metadata_fails_closed(self) -> None:
        for field, value in (
            ("do_sample", True),
            ("decoding", "sampling"),
            ("max_new_tokens", 32),
        ):
            with self.subTest(field=field):
                broken = self._mutate(
                    f"{summ.ARM_INTEGER}::low",
                    lambda rows, f=field, v=value: [
                        row.update({f: v}) for row in rows
                    ],
                )
                code, report = self._run(paths=broken)
                self.assertEqual(code, 2)
                self.assertIsNone(report)

    def test_prompt_arm_checkpoint_hash_mismatch_fails_closed(self) -> None:
        broken = self._mutate(
            f"{summ.ARM_PROMPT_INTEGER}::low",
            lambda rows: [row.update({"wv_sha256": "0" * 64}) for row in rows],
        )
        code, report = self._run(paths=broken)
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_no_prompt_arm_checkpoint_hash_fails_closed(self) -> None:
        broken = self._mutate(
            f"{summ.ARM_INTEGER}::high",
            lambda rows: [
                row.update({"checkpoint_sha256": summ.CHECKPOINT_SHA256})
                for row in rows
            ],
        )
        code, report = self._run(paths=broken)
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_low_g_above_threshold_fails_closed(self) -> None:
        def _bump(rows):
            for row in rows:
                row["full_llm_visual_tokens"] = 600
                row["full_patch_tokens"] = 600 * summ.MERGE_UNIT
                row["image_grid_thw"] = [1, summ.MERGE_UNIT, 600]
                row["selected_group_indices"] = list(range(summ.BUDGET))
                row["output_group_count"] = summ.BUDGET
                row["identity_bypass"] = False
                row["selected_group_indices_sha256"] = array_sha256(
                    np.asarray(range(summ.BUDGET), dtype=np.int64)
                )

        broken = self._mutate(f"{summ.ARM_INTEGER}::low", _bump)
        code, report = self._run(paths=broken)
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_high_g_at_or_below_threshold_fails_closed(self) -> None:
        def _lower(rows):
            for row in rows:
                row["full_llm_visual_tokens"] = 512
                row["full_patch_tokens"] = 512 * summ.MERGE_UNIT
                row["image_grid_thw"] = [1, summ.MERGE_UNIT, 512]
                row["selected_group_indices"] = list(range(summ.BUDGET))
                row["output_group_count"] = summ.BUDGET
                row["identity_bypass"] = False
                row["selected_group_indices_sha256"] = array_sha256(
                    np.asarray(range(summ.BUDGET), dtype=np.int64)
                )

        broken = self._mutate(f"{summ.ARM_PROMPT_INTEGER}::high", _lower)
        code, report = self._run(paths=broken)
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_g_mismatch_with_gate_measurement_fails_closed(self) -> None:
        def _shift(rows):
            for row in rows:
                g = row["full_llm_visual_tokens"]
                row["full_llm_visual_tokens"] = g - 1
                row["full_patch_tokens"] = (g - 1) * summ.MERGE_UNIT
                row["image_grid_thw"] = [1, summ.MERGE_UNIT, g - 1]

        broken = self._mutate(f"{summ.ARM_INTEGER}::high", _shift)
        code, report = self._run(paths=broken)
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_same_budget_cross_arm_grid_contamination_fails_closed(self) -> None:
        # Re-tokenise the integer low arm to a different but still aligned grid
        # while keeping G / full_patch_tokens / the pixel budget identical, so
        # every per-record gate passes and only the same-image cross-arm grid
        # contract is violated. _mutate recomputes record_sha256, so the failure
        # is the pairing contract, not a stale hash.
        def _contaminate(rows):
            for row in rows:
                g = row["full_llm_visual_tokens"]
                row["image_grid_thw"] = [1, summ.MERGE_UNIT * 2, g // 2]

        broken = self._mutate(f"{summ.ARM_INTEGER}::low", _contaminate)
        with self.assertRaises(summ.InputContractError) as ctx:
            self._analyze(paths=broken)
        message = str(ctx.exception)
        self.assertIn("same-image contract", message)
        self.assertIn("selector arm", message)
        self.assertIn("image_grid_thw", message)
        code, report = self._run(paths=broken)
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_low_high_unchanged_grid_fails_closed(self) -> None:
        # The per-record gate keeps low G<=512 and high G>512 mutually
        # exclusive, so an identical low/high grid can never reach the pairing
        # check through the full validator; drive the pairing contract directly
        # with the four s0 recordings forced to the same grid and G. The mutated
        # rows have their record_sha256 recomputed, so this is not a hash error.
        indices = self._arm_indices()
        manifest_ids = [sid for sid, *_ in PLAN]
        for key in summ.arm_paths(self.paths):
            row = indices[key]["s0"]
            g = 64
            row["image_grid_thw"] = [1, summ.MERGE_UNIT, g]
            row["full_patch_tokens"] = g * summ.MERGE_UNIT
            row["full_llm_visual_tokens"] = g
            row["identity_bypass"] = True
            row["output_group_count"] = g
            row["selected_group_indices"] = list(range(g))
            row["selected_group_indices_sha256"] = array_sha256(
                np.asarray(range(g), dtype=np.int64)
            )
            row["record_sha256"] = summ.record_sha256(row)
        with self.assertRaises(summ.InputContractError) as ctx:
            summ.verify_same_image_pairing(manifest_ids, indices)
        message = str(ctx.exception)
        self.assertIn("low and high recordings", message)
        self.assertIn("identical image_grid_thw", message)
        self.assertIn("s0", message)

    def test_same_budget_cross_arm_token_mismatch_fails_closed(self) -> None:
        # Identical grid at the same sample / budget but the prompt arm reports
        # a different full_llm_visual_tokens (the gate measurement normally
        # locks this); drive the pairing contract directly with a recomputed
        # record_sha256 so this is not a hash error.
        indices = self._arm_indices()
        manifest_ids = [sid for sid, *_ in PLAN]
        row = indices[f"{summ.ARM_PROMPT_INTEGER}::low"]["s0"]
        row["full_llm_visual_tokens"] = 68
        row["record_sha256"] = summ.record_sha256(row)
        with self.assertRaises(summ.InputContractError) as ctx:
            summ.verify_same_image_pairing(manifest_ids, indices)
        message = str(ctx.exception)
        self.assertIn("full_llm_visual_tokens", message)
        self.assertIn("token mismatch", message)

    def test_non_authoritative_gate_report_fails_closed(self) -> None:
        self.case.gate_report["authoritative"] = False
        gate = self.tmp / "gate.json"
        gate.write_text(json.dumps(self.case.gate_report), encoding="utf-8")
        code, report = self._run()
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_unresolved_gate_report_fails_closed(self) -> None:
        self.case.gate_report["plan"]["resolved"] = False
        gate = self.tmp / "gate.json"
        gate.write_text(json.dumps(self.case.gate_report), encoding="utf-8")
        code, report = self._run()
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_replaced_gate_report_fails_closed(self) -> None:
        self.case.gate_report["plan"]["replaced"] = ["s0"]
        gate = self.tmp / "gate.json"
        gate.write_text(json.dumps(self.case.gate_report), encoding="utf-8")
        code, report = self._run()
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_missing_manifest_metadata_fails_closed(self) -> None:
        self.case.manifest_rows[0].pop("resolution_counterfactual")
        manifest = self.tmp / "manifest.jsonl"
        manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in self.case.manifest_rows),
            encoding="utf-8",
        )
        code, report = self._run()
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_unknown_stratum_fails_closed(self) -> None:
        self.case.manifest_rows[0]["resolution_counterfactual"]["stratum"] = "bogus"
        manifest = self.tmp / "manifest.jsonl"
        manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in self.case.manifest_rows),
            encoding="utf-8",
        )
        code, report = self._run()
        self.assertEqual(code, 2)
        self.assertIsNone(report)


class BootstrapTest(unittest.TestCase):
    def test_deterministic_for_same_seed(self) -> None:
        values = [1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0]
        strata = ["a", "a", "a", "b", "b", "b", "b", "b"]
        first = summ.bootstrap_mean_ci(values, strata, seed=7, repeats=500)
        second = summ.bootstrap_mean_ci(values, strata, seed=7, repeats=500)
        self.assertEqual(first, second)

    def test_different_seed_changes_interval(self) -> None:
        values = [((i % 7) - 3) / 3.0 for i in range(21)]
        strata = ["a"] * 7 + ["b"] * 14
        intervals = {
            summ.bootstrap_mean_ci(values, strata, seed=seed, repeats=500)
            for seed in range(6)
        }
        self.assertGreater(len(intervals), 1)

    def test_all_positive_is_tight(self) -> None:
        ci = summ.bootstrap_mean_ci([1.0] * 8, ["a"] * 8, repeats=200)
        self.assertEqual(ci, (1.0, 1.0))

    def test_all_zero_lower_bound_is_zero(self) -> None:
        ci = summ.bootstrap_mean_ci([0.0] * 8, ["a"] * 8, repeats=200)
        self.assertEqual(ci, (0.0, 0.0))

    def test_percentile_convention_matches_frozen_scorer(self) -> None:
        values = [float(i) for i in range(10)]
        repeats = 1000
        import random

        rng = random.Random(summ.BOOTSTRAP_SEED)
        deltas = []
        for _ in range(repeats):
            deltas.append(sum(rng.choice(values) for _ in range(10)) / 10)
        deltas.sort()
        expected = (
            deltas[int(0.025 * (repeats - 1))],
            deltas[int(0.975 * (repeats - 1))],
        )
        actual = summ.bootstrap_mean_ci(
            values, ["a"] * 10, seed=summ.BOOTSTRAP_SEED, repeats=repeats
        )
        self.assertEqual(actual, expected)

    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaises(summ.InputContractError):
            summ.bootstrap_mean_ci([1.0, 2.0], ["a"], repeats=10)


class DecisionTest(unittest.TestCase):
    def test_positive_lower_bound_confirms(self) -> None:
        decision = summ.decide(0.000001)
        self.assertEqual(decision["outcome"], summ.OUTCOME_CONFIRMED)
        self.assertTrue(decision["confirmed"])

    def test_zero_lower_bound_does_not_confirm(self) -> None:
        decision = summ.decide(0.0)
        self.assertEqual(decision["outcome"], summ.OUTCOME_NOT_CONFIRMED)
        self.assertFalse(decision["confirmed"])

    def test_negative_lower_bound_does_not_confirm(self) -> None:
        decision = summ.decide(-0.5)
        self.assertEqual(decision["outcome"], summ.OUTCOME_NOT_CONFIRMED)

    def test_formal_conclusion_unchanged(self) -> None:
        for lower in (-1.0, 0.0, 1.0):
            decision = summ.decide(lower)
            self.assertTrue(decision["formal_full2638_conclusion_unchanged"])
            self.assertEqual(
                decision["formal_full2638_conclusion"],
                summ.FORMAL_CONCLUSION_UNCHANGED,
            )
            self.assertEqual(
                decision["forbidden_metrics_never_used"], list(summ.FORBIDDEN_METRICS)
            )
            self.assertTrue(decision["uses_only_primary_estimand"])


class AnalyzeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run_case(self, correctness):
        case = SyntheticCase(self.tmp, correctness)
        paths = case.paths()
        return summ.run(
            paths,
            output_json=self.tmp / "r.json",
            output_csv=self.tmp / "r.csv",
            output_md=self.tmp / "r.md",
            verify_pins=False,
            expected_sample_count=N,
            manifest_sha256=MANIFEST_SHA,
            bootstrap_repeats=200,
        )

    def test_all_positive_effects_are_confirmed(self) -> None:
        correctness = {
            sid: {
                "low_corrected_l2_integer": False,
                "high_corrected_l2_integer": False,
                "low_prompt_plus_corrected_l2_integer": False,
                "high_prompt_plus_corrected_l2_integer": True,
            }
            for sid, *_ in PLAN
        }
        code, report = self._run_case(correctness)
        self.assertEqual(code, 0)
        self.assertEqual(report["primary_comparison"]["point_estimate"], 1.0)
        self.assertEqual(report["decision"]["outcome"], summ.OUTCOME_CONFIRMED)

    def test_all_zero_effects_are_not_confirmed(self) -> None:
        correctness = {
            sid: {
                "low_corrected_l2_integer": True,
                "high_corrected_l2_integer": True,
                "low_prompt_plus_corrected_l2_integer": True,
                "high_prompt_plus_corrected_l2_integer": True,
            }
            for sid, *_ in PLAN
        }
        code, report = self._run_case(correctness)
        self.assertEqual(code, 0)
        self.assertEqual(report["primary_comparison"]["point_estimate"], 0.0)
        self.assertEqual(report["primary_comparison"]["ci95"], [0.0, 0.0])
        self.assertEqual(report["decision"]["outcome"], summ.OUTCOME_NOT_CONFIRMED)

    def test_transitions_and_parse_rate(self) -> None:
        correctness = {
            "s0": {
                "low_corrected_l2_integer": False,
                "high_corrected_l2_integer": False,
                "low_prompt_plus_corrected_l2_integer": True,
                "high_prompt_plus_corrected_l2_integer": True,
            },
            "s1": {
                "low_corrected_l2_integer": True,
                "high_corrected_l2_integer": True,
                "low_prompt_plus_corrected_l2_integer": False,
                "high_prompt_plus_corrected_l2_integer": False,
            },
            "s2": {
                "low_corrected_l2_integer": True,
                "high_corrected_l2_integer": True,
                "low_prompt_plus_corrected_l2_integer": True,
                "high_prompt_plus_corrected_l2_integer": True,
            },
            "s3": {
                "low_corrected_l2_integer": False,
                "high_corrected_l2_integer": False,
                "low_prompt_plus_corrected_l2_integer": False,
                "high_prompt_plus_corrected_l2_integer": False,
            },
        }
        code, report = self._run_case(correctness)
        self.assertEqual(code, 0)
        low = report["budget_metrics"]["low"]
        high = report["budget_metrics"]["high"]
        self.assertEqual(low["wrong_to_right"], 1)
        self.assertEqual(low["right_to_wrong"], 1)
        self.assertEqual(low["net_correction"], 0)
        self.assertEqual(high["wrong_to_right"], 1)
        self.assertEqual(high["right_to_wrong"], 1)
        self.assertEqual(low["integer_parse_rate"], 1.0)
        self.assertEqual(high["prompt_parse_rate"], 1.0)
        # effects: s0 = (1-0)-(1-0)=0; s1 = (0-1)-(0-1)=0;
        # s2 = (1-1)-(1-1)=0; s3 = (0-0)-(0-0)=0.
        self.assertEqual(report["primary_comparison"]["point_estimate"], 0.0)


class ReportRenderingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.case = SyntheticCase(self.tmp)
        self.paths = self.case.paths()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self):
        return summ.run(
            self.paths,
            output_json=self.tmp / "report.json",
            output_csv=self.tmp / "report.csv",
            output_md=self.tmp / "report.md",
            verify_pins=False,
            expected_sample_count=N,
            manifest_sha256=MANIFEST_SHA,
            bootstrap_repeats=64,
        )

    def test_success_writes_json_csv_and_markdown(self) -> None:
        code, report = self._run()
        self.assertEqual(code, 0)
        for suffix in ("report.json", "report.csv", "report.md"):
            self.assertTrue((self.tmp / suffix).is_file(), suffix)

    def test_csv_is_per_sample(self) -> None:
        self._run()
        rows = (self.tmp / "report.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), N + 1)
        header = rows[0]
        for column in (
            "sample_id",
            "stratum",
            "low_integer_correct",
            "low_prompt_correct",
            "high_integer_correct",
            "high_prompt_correct",
            "effect",
        ):
            self.assertIn(column, header)

    def test_json_has_no_official_cv_bench_accuracy(self) -> None:
        _, report = self._run()
        self.assertNotIn("cv_bench_accuracy", report)
        self.assertNotIn("cv_bench_accuracy", report["budget_metrics"]["low"])
        self.assertEqual(
            report["primary_metric"]["identifier"], summ.PRIMARY_ESTIMAND_EXPANDED
        )
        self.assertEqual(
            report["primary_metric"]["preregistered_estimand_literal"],
            summ.PRIMARY_ESTIMAND,
        )
        self.assertEqual(
            report["decision"]["outcome"],
            summ.OUTCOME_NOT_CONFIRMED,
        )

    def test_markdown_is_chinese_and_mentions_contract(self) -> None:
        self._run()
        markdown = (self.tmp / "report.md").read_text(encoding="utf-8")
        for token in (
            "同图双分辨率反事实",
            "主估计量",
            "W→R",
            "R→W",
            "net",
            "parse",
            "禁用主指标",
            "不作时延或 FPGA 加速声明",
            "PROMPT_GAIN_NOT_CONFIRMED_FULL2638",
        ):
            self.assertIn(token, markdown)
        self.assertIn(summ.OUTCOME_NOT_CONFIRMED, markdown)

    def test_report_records_four_arm_files_and_inputs(self) -> None:
        _, report = self._run()
        self.assertEqual(
            set(report["arm_files"]),
            {
                f"{summ.ARM_INTEGER}::low",
                f"{summ.ARM_INTEGER}::high",
                f"{summ.ARM_PROMPT_INTEGER}::low",
                f"{summ.ARM_PROMPT_INTEGER}::high",
            },
        )
        for name in (
            "preregistered_config",
            "separated_manifest",
            "gate_separation_report",
        ):
            self.assertTrue(report["inputs_sha256"][name]["sha256"])

    def test_validation_flags_all_contract_checks(self) -> None:
        _, report = self._run()
        validation = report["validation"]
        for key in (
            "exactly_100_unique_ids_per_file",
            "four_file_id_sets_identical",
            "gate_report_resolved_100",
            "low_G_le_512",
            "high_G_gt_512",
            "G_matches_authoritative_measurements",
            "same_image_paired",
            "no_mixed_arm",
            "record_hashes_verified",
            "checkpoint_and_wv_hashes_verified",
        ):
            self.assertTrue(validation[key], key)


class RealConfigPinTest(unittest.TestCase):
    """Read-only checks against the real preregistered config / gate report."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.config = json.loads(
            summ.PREREGISTERED_CONFIG.read_text(encoding="utf-8")
        )
        self.gate = json.loads(
            summ.GATE_SEPARATION_REPORT.read_text(encoding="utf-8")
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_real_config_and_physical_pins_verify(self) -> None:
        result = summ.verify_config(
            self.config, self.gate, summ.DEFAULT_PATHS, expected_sample_count=100
        )
        self.assertTrue(result["pins_verified"])
        self.assertEqual(result["checkpoint_sha256"], summ.CHECKPOINT_SHA256)
        self.assertEqual(result["wv_sha256"], summ.WV_SHA256)
        self.assertEqual(result["wv_scale_sha256"], summ.WV_SCALE_SHA256)

    def test_tampered_config_manifest_pin_fails_closed(self) -> None:
        config = json.loads(json.dumps(self.config))
        config["manifest"]["sha256"] = "0" * 64
        with self.assertRaises(summ.InputContractError):
            summ.verify_config(
                config, self.gate, summ.DEFAULT_PATHS, expected_sample_count=100
            )

    def test_tampered_gate_config_sha_fails_closed(self) -> None:
        gate = json.loads(json.dumps(self.gate))
        gate["config"]["sha256"] = "0" * 64
        with self.assertRaises(summ.InputContractError):
            summ.verify_config(
                self.config, gate, summ.DEFAULT_PATHS, expected_sample_count=100
            )

    def test_missing_arm_files_fail_closed(self) -> None:
        paths = summ.InputPaths(
            **{
                **summ.DEFAULT_PATHS.__dict__,
                "arm_integer_low_jsonl": self.tmp / "absent_int_low.jsonl",
                "arm_integer_high_jsonl": self.tmp / "absent_int_high.jsonl",
                "arm_prompt_low_jsonl": self.tmp / "absent_prompt_low.jsonl",
                "arm_prompt_high_jsonl": self.tmp / "absent_prompt_high.jsonl",
            }
        )
        code, report = summ.run(
            paths,
            output_json=self.tmp / "r.json",
            output_csv=self.tmp / "r.csv",
            output_md=self.tmp / "r.md",
        )
        self.assertEqual(code, 2)
        self.assertIsNone(report)
        for suffix in ("r.json", "r.csv", "r.md"):
            self.assertFalse((self.tmp / suffix).exists(), suffix)


if __name__ == "__main__":
    unittest.main()
