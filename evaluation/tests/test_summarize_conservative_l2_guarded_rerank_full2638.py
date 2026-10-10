#!/usr/bin/env python3
"""CPU-only focused tests for the full2638 v2 L2-guarded summariser.

Synthetic inputs live in a temporary directory and ``expected_full_count=3`` is
used so the frozen 2638-sample artifacts are never touched.  The tests pin the
corrected development gate branches, the exact K96 / 87 real + 9 context layout
check, the widened L2-guarded swap audit, the fail-closed input contract and the
no-write-on-failure behaviour (exit code 2 while the v2 arm is absent).
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
    run_hf_conservative_l2_guarded_rerank as runner,
)
from evaluation import (  # noqa: E402
    summarize_conservative_l2_guarded_rerank_full2638 as summ,
)
from evaluation.conservative_l2_guarded_reranker import (  # noqa: E402
    MECHANISM,
    select_guarded_rerank,
)
from evaluation.hardware_regular_b1_corrected_l2 import (  # noqa: E402
    array_sha256,
    corrected_patch_l2_integer,
)

SOURCES = (
    ("s0", "ADE20K", "ADE20K Validation Set", "Count", "(A)"),
    ("s1", "COCO", "COCO 2017 Validation Set", "Relation", "(B)"),
    ("s2", "Omni3D", "Omni3D_SUNRGBD", "Depth", "(C)"),
)
GRID = [1, 44, 44]


def _manifest_rows() -> list[dict]:
    return [
        {
            "sample_id": sid,
            "task": task,
            "type": "2D" if source in {"ADE20K", "COCO"} else "3D",
            "source": source,
            "source_dataset": source_dataset,
            "choices": ["alpha", "beta", "gamma"],
            "answer": answer,
            "image": "x.jpg",
            "prompt": "p",
            "max_new_tokens": 16,
        }
        for sid, source, source_dataset, task, answer in SOURCES
    ]


def _new_record(
    sample_id: str,
    task: str,
    output_text: str,
    *,
    seed: int,
    manifest_sha: str,
    bypass: bool = False,
    leakage: str | None = None,
    break_hash: bool = False,
) -> dict:
    if bypass:
        grid = [1, 8, 8]
        gh = gw = 4
        hidden = np.zeros((gh * gw, 4, 1024), dtype=np.float64)
        semantic = np.zeros(gh * gw, dtype=np.int64)
    else:
        grid = GRID
        gh = grid[1] // 2
        gw = grid[2] // 2
        rng = np.random.default_rng(seed)
        hidden = rng.normal(size=(gh * gw, 4, 1024)) * rng.normal(
            size=(gh * gw, 1, 1)
        )
        semantic = rng.integers(0, 32768, size=gh * gw)
    l2 = corrected_patch_l2_integer(hidden)["l2_norm_q15"]
    result = select_guarded_rerank(
        hidden,
        gh,
        gw,
        semantic_norm_q15=semantic,
        l2_norm_q15=l2,
    )
    selection = result["selection"]
    audit = runner._guarded_audit_fields(result)
    full_patch = grid[1] * grid[2]
    full_llm = full_patch // 4
    selected = [int(v) for v in selection.group_indices]
    record = {
        "sample_id": sample_id,
        "task": task,
        "arm": MECHANISM,
        "mechanism": MECHANISM,
        "run_mode": "full2638",
        "utility_formula": "semantic_q15 + corrected_l2_q15",
        "legal_gain_formula": "semantic_gain - l2_loss = utility_added - utility_removed",
        "max_swaps": 16,
        "context_pooling_basis": "corrected_integer_l2_norm_q15",
        "manifest_sha256": manifest_sha,
        "requested_budget": 96,
        "region_grid": 3,
        "local_candidate_cap": 16,
        "image_grid_thw": grid,
        "full_patch_tokens": full_patch,
        "full_llm_visual_tokens": full_llm,
        "output_group_count": selection.output_group_count,
        "real_group_count": selection.real_group_count,
        "context_group_count": selection.context_group_count,
        "identity_bypass": bool(selection.bypass),
        "selected_group_indices": selected,
        "selected_group_indices_sha256": array_sha256(
            np.asarray(selected, dtype=np.int64)
        ),
        "output_text": output_text,
        "uses_labels": False,
        "uses_answer_tokens": False,
        "uses_gradients": False,
        "uses_floating_lambda": False,
        "uses_learned_new_weights": False,
        "uses_task_source_ids": False,
        **audit,
    }
    if bypass:
        record["real_group_count"] = full_llm
        record["context_group_count"] = 0
        record["swap_count"] = 0
    if leakage is not None:
        record[leakage] = True
    record["record_sha256"] = summ._record_sha256(record)
    if break_hash:
        record["record_sha256"] = "0" * 64
    return record


def _ref_rows(outputs: dict[str, str]) -> list[dict]:
    return [
        {"sample_id": sample_id, "output_text": outputs[sample_id]}
        for sample_id in outputs
    ]


class GateConstantsTest(unittest.TestCase):
    def test_exact_dense_and_computed_floor(self) -> None:
        self.assertEqual(summ.DENSE_SCORE, 0.8727298774444869)
        self.assertEqual(summ.GO_SCORE_FLOOR, 0.9 * summ.DENSE_SCORE)
        self.assertAlmostEqual(
            summ.GO_SCORE_FLOOR, 0.7854568897000382, places=16
        )
        self.assertGreater(
            summ.GO_SCORE_FLOOR, summ.PRIOR_PROMPT_FUSION_SCORE
        )

    def test_prior_fusion_is_not_a_gate(self) -> None:
        decision = summ.decide(
            new_score=summ.GO_SCORE_FLOOR,
            parse_rate=1.0,
            layout_ok=True,
            ci_lower=0.0,
            leakage_ok=True,
        )
        self.assertNotIn("point_no_worse_than_prior_baseline", decision["gates"])
        self.assertEqual(len(decision["gates"]), 5)
        self.assertEqual(decision["outcome"], summ.OUTCOME_GO)


class DecideGateTest(unittest.TestCase):
    def test_all_gates_go(self) -> None:
        decision = summ.decide(
            new_score=0.80,
            parse_rate=1.0,
            layout_ok=True,
            ci_lower=0.001,
            leakage_ok=True,
        )
        self.assertEqual(decision["outcome"], summ.OUTCOME_GO)
        self.assertTrue(all(decision["gates"].values()))

    def test_each_gate_stops(self) -> None:
        base = dict(
            new_score=0.80,
            parse_rate=1.0,
            layout_ok=True,
            ci_lower=0.001,
            leakage_ok=True,
        )
        for key, value in (
            ("parse_rate", 0.999),
            ("layout_ok", False),
            ("new_score", summ.GO_SCORE_FLOOR - 1e-9),
            ("ci_lower", -1e-9),
            ("leakage_ok", False),
        ):
            with self.subTest(key=key):
                kwargs = dict(base)
                kwargs[key] = value
                decision = summ.decide(**kwargs)
                self.assertEqual(decision["outcome"], summ.OUTCOME_STOP)

    def test_floor_is_inclusive(self) -> None:
        decision = summ.decide(
            new_score=summ.GO_SCORE_FLOOR,
            parse_rate=1.0,
            layout_ok=True,
            ci_lower=0.0,
            leakage_ok=True,
        )
        self.assertTrue(decision["gates"]["score_at_least_90pct_dense"])
        self.assertTrue(decision["gates"]["paired_ci_lower_nonnegative"])
        self.assertEqual(decision["outcome"], summ.OUTCOME_GO)


class SwapAuditTest(unittest.TestCase):
    def _row(self, **overrides) -> dict:
        row = {
            "swap_count": 1,
            "removed_group_indices": [1],
            "added_group_indices": [6],
            "swap_region_ids": [0],
            "swap_semantic_gain_q15": [500],
            "swap_l2_loss_q15": [200],
            "swap_legal_gain_q16": [300],
            "swap_utility_removed_q16": [20000],
            "swap_utility_added_q16": [20300],
            "per_region_selected_counts_before": [2],
            "per_region_selected_counts_after": [2],
            "baseline_anchors": [0],
            "baseline_selected_real": [0, 1],
            "reranked_selected_real": [0, 6],
            "image_grid_thw": [1, 12, 12],
        }
        row.update(overrides)
        return row

    def test_accepts_consistent_swap(self) -> None:
        summ._check_swap_audit(self._row(), "s0", max_swaps=16)

    def test_rejects_non_positive_legal_gain(self) -> None:
        with self.assertRaises(summ.InputContractError):
            summ._check_swap_audit(
                self._row(swap_legal_gain_q16=[0]), "s0", max_swaps=16
            )

    def test_rejects_inconsistent_legal_gain(self) -> None:
        with self.assertRaises(summ.InputContractError):
            summ._check_swap_audit(
                self._row(swap_legal_gain_q16=[299]), "s0", max_swaps=16
            )

    def test_rejects_region_mismatch(self) -> None:
        # groups 1 and 2 are in different 3x3 regions of a 6x6 merge grid.
        with self.assertRaises(summ.InputContractError):
            summ._check_swap_audit(
                self._row(swap_region_ids=[0], added_group_indices=[2]),
                "s0",
                max_swaps=16,
            )


class AnalyzeContractTest(unittest.TestCase):
    def _run_analyze(
        self,
        tmp: Path,
        *,
        new_outputs: dict[str, str],
        ref_outputs: dict[str, dict[str, str]],
        tamper: dict | None = None,
    ):
        manifest_rows = _manifest_rows()
        manifest_path = tmp / "manifest.jsonl"
        manifest_path.write_text(
            "\n".join(json.dumps(row) for row in manifest_rows) + "\n",
            encoding="utf-8",
        )
        manifest_sha = summ.sha256_file(manifest_path)
        new_rows = [
            _new_record(
                sid,
                dict((s[0], s[3]) for s in SOURCES)[sid],
                new_outputs[sid],
                seed=100 + index,
                manifest_sha=manifest_sha,
            )
            for index, sid in enumerate(new_outputs)
        ]
        if tamper:
            new_rows[0].update(tamper)
            if "record_sha256" not in tamper:
                new_rows[0]["record_sha256"] = summ._record_sha256(new_rows[0])
        dense_rows = _ref_rows(ref_outputs["dense"])
        corrected_rows = _ref_rows(ref_outputs["corrected"])
        paths = {}
        for name, rows in (
            ("new_arm_jsonl", new_rows),
            ("dense_full_jsonl", dense_rows),
            ("corrected_l2_integer_jsonl", corrected_rows),
        ):
            path = tmp / f"{name}.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            paths[name] = path
        input_paths = summ.InputPaths(
            full_manifest=manifest_path,
            **paths,
        )
        return summ.analyze(
            config={"fixed_configuration": {"max_swaps": 16}},
            manifest_rows=manifest_rows,
            dense_rows=dense_rows,
            corrected_l2_rows=corrected_rows,
            new_rows=new_rows,
            paths=input_paths,
            expected_full_count=3,
        )

    def test_go_when_new_arm_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = self._run_analyze(
                Path(tmp),
                new_outputs={"s0": "(A)", "s1": "(B)", "s2": "(C)"},
                ref_outputs={
                    "dense": {"s0": "(A)", "s1": "(B)", "s2": "(C)"},
                    "corrected": {"s0": "(B)", "s1": "(C)", "s2": "(A)"},
                },
            )
            self.assertEqual(report["layout"]["compressed_count"], 3)
            self.assertEqual(report["decision"]["outcome"], summ.OUTCOME_GO)
            self.assertEqual(report["metrics"]["new"]["cv_bench_accuracy"], 1.0)

    def test_stop_when_new_arm_wrong(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = self._run_analyze(
                Path(tmp),
                new_outputs={"s0": "(B)", "s1": "(C)", "s2": "(A)"},
                ref_outputs={
                    "dense": {"s0": "(A)", "s1": "(B)", "s2": "(C)"},
                    "corrected": {"s0": "(A)", "s1": "(B)", "s2": "(C)"},
                },
            )
            self.assertEqual(report["decision"]["outcome"], summ.OUTCOME_STOP)

    def test_fail_closed_on_layout_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(summ.InputContractError):
                self._run_analyze(
                    Path(tmp),
                    new_outputs={"s0": "(A)", "s1": "(B)", "s2": "(C)"},
                    ref_outputs={
                        "dense": {"s0": "(A)", "s1": "(B)", "s2": "(C)"},
                        "corrected": {"s0": "(A)", "s1": "(B)", "s2": "(C)"},
                    },
                    tamper={"real_group_count": 88},
                )

    def test_fail_closed_on_record_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(summ.InputContractError):
                self._run_analyze(
                    Path(tmp),
                    new_outputs={"s0": "(A)", "s1": "(B)", "s2": "(C)"},
                    ref_outputs={
                        "dense": {"s0": "(A)", "s1": "(B)", "s2": "(C)"},
                        "corrected": {"s0": "(A)", "s1": "(B)", "s2": "(C)"},
                    },
                    tamper={"record_sha256": "0" * 64},
                )

    def test_fail_closed_on_leakage_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = _manifest_rows()
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            manifest_sha = summ.sha256_file(manifest_path)
            new_rows = [
                _new_record(
                    sid,
                    task,
                    "(A)",
                    seed=index,
                    manifest_sha=manifest_sha,
                    leakage="uses_labels",
                )
                for index, (sid, _s, _sd, task, _a) in enumerate(SOURCES)
            ]
            with self.assertRaises(summ.InputContractError):
                summ.analyze(
                    config={"fixed_configuration": {"max_swaps": 16}},
                    manifest_rows=rows,
                    dense_rows=_ref_rows({s[0]: "(A)" for s in SOURCES}),
                    corrected_l2_rows=_ref_rows({s[0]: "(A)" for s in SOURCES}),
                    new_rows=new_rows,
                    paths=summ.InputPaths(full_manifest=manifest_path),
                    expected_full_count=3,
                )


class ConfigPinTest(unittest.TestCase):
    def test_verify_config_pins_skips_null_and_checks_real(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            target = tmp_path / "manifest.jsonl"
            target.write_text("hello", encoding="utf-8")
            digest = summ.sha256_file(target)
            config = {
                "inputs_sha256": {
                    "full_manifest": {
                        "path": str(target),
                        "sha256": digest,
                    },
                    "new_arm_jsonl": {"path": "x", "sha256": None},
                }
            }
            paths = summ.InputPaths(full_manifest=target)
            verified = summ.verify_config_pins(config, paths)
            self.assertEqual(verified["full_manifest"]["status"], "verified")
            self.assertEqual(
                verified["new_arm_jsonl"]["status"], "not_yet_available"
            )
            config["inputs_sha256"]["full_manifest"]["sha256"] = "0" * 64
            with self.assertRaises(summ.InputContractError):
                summ.verify_config_pins(config, paths)

    def test_cli_fail_closed_exit_2_and_no_report_when_arm_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            missing = tmp_path / "missing_arm.jsonl"
            config = {
                "fixed_configuration": {"max_swaps": 16},
                "arms": {MECHANISM: {"output_path": str(missing)}},
                "inputs_sha256": {},
            }
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            rows = [{"sample_id": "s0", "output_text": "(A)"}]
            for name in ("manifest", "dense", "corrected"):
                (tmp_path / f"{name}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n",
                    encoding="utf-8",
                )
            report_json = tmp_path / "report.json"
            report_md = tmp_path / "report.md"
            code = summ.main(
                [
                    "--preregistered",
                    str(config_path),
                    "--full-manifest",
                    str(tmp_path / "manifest.jsonl"),
                    "--dense-full-jsonl",
                    str(tmp_path / "dense.jsonl"),
                    "--corrected-l2-integer-jsonl",
                    str(tmp_path / "corrected.jsonl"),
                    "--new-arm-jsonl",
                    str(missing),
                    "--report-json",
                    str(report_json),
                    "--report-md",
                    str(report_md),
                ]
            )
            self.assertEqual(code, 2)
            self.assertFalse(report_json.exists())
            self.assertFalse(report_md.exists())


class RealPreregisteredConfigTest(unittest.TestCase):
    def test_real_preregistered_config_pins_verify(self) -> None:
        self.assertTrue(summ.PREREGISTERED_CONFIG.is_file())
        config = json.loads(summ.PREREGISTERED_CONFIG.read_text(encoding="utf-8"))
        verified = summ.verify_config_pins(config, summ.DEFAULT_PATHS)
        self.assertEqual(verified["new_arm_jsonl"]["status"], "not_yet_available")
        self.assertGreater(len(verified), 0)
        for name, entry in verified.items():
            if entry["status"] == "verified":
                self.assertEqual(
                    entry["sha256"],
                    config["inputs_sha256"][name]["sha256"],
                    name,
                )

    def test_default_cli_fail_closed_exit_2_no_report_when_arm_absent(self) -> None:
        code = summ.main([])
        self.assertEqual(code, 2)
        self.assertFalse(summ.REPORT_JSON.exists())
        self.assertFalse(summ.REPORT_MD.exists())


if __name__ == "__main__":
    unittest.main()
