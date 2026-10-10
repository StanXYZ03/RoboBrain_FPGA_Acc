"""CPU-only tests for the v2 decision-only summariser (GO/STOP + fail-closed).

Builds a fully synthetic 24-sample world: six cells, two challenge + two control
each, a frozen dense reference, a corrected-L2 control and an oracle arm.  No
model weights, no CUDA, no network.
"""

from __future__ import annotations

import unittest

import numpy as np

from evaluation.b1_answer_margin_oracle import (
    ARM_ORACLE,
    BUDGET,
    array_sha256,
    record_sha256,
    stable_sha256_rank,
)
from evaluation.b1_answer_margin_oracle_challenge_control import (
    CELLS,
    CHALLENGE,
    CONTROL,
    STRATA,
)
from evaluation.summarize_b1_answer_margin_oracle_challenge_control_pilot import (
    InputContractError,
    analyze,
)

A_SHA = "a" * 64
B_SHA = "b" * 64
FULL_GROUPS = 256


def _subdataset(cell, stratum: str, rank: int) -> str:
    if cell.source != "Omni3D":
        return "ADE20K Validation Set" if cell.source == "ADE20K" else "COCO 2017 Validation Set"
    if cell.task == "Depth":
        if stratum == CHALLENGE:
            return "Omni3D_Hypersim"
        return "Omni3D_SUNRGBD"
    return "Omni3D_nuScenes"


def _sample_id(cell, stratum: str, rank: int) -> str:
    return f"{cell.label()}|{stratum}|{rank}"


def oracle_record(sample_id: str, *, correct: bool = True, parse_ok: bool = True, **overrides) -> dict:
    values = np.linspace(0.5, 2.5, FULL_GROUPS, dtype=np.float64)
    if not parse_ok:
        output_text = "totally unclear"
    else:
        output_text = "(A)" if correct else "(B)"
    record = {
        "sample_id": sample_id,
        "arm": ARM_ORACLE,
        "manifest_sha256": A_SHA,
        "config_sha256": B_SHA,
        "requested_budget": BUDGET,
        "region_grid": 3,
        "local_candidate_cap": 16,
        "output_text": output_text,
        "output_token_ids": [1, 2],
        "full_patch_tokens": FULL_GROUPS * 4,
        "image_grid_thw": [1, 16, FULL_GROUPS * 4 // 16],
        "full_llm_visual_tokens": FULL_GROUPS,
        "output_group_count": BUDGET,
        "context_group_count": 9,
        "real_group_count": 87,
        "identity_bypass": False,
        "selected_group_indices": list(range(BUDGET)),
        "label_tokenization": {"contract_ok": True},
        "margin": {"value": 1.5},
        "saliency_group_values": [float(value) for value in values],
        "saliency_sha256": array_sha256(values),
        "model_parameter_grads_all_none": True,
        "layout": {
            "passed": True,
            "output_group_count": BUDGET,
            "real_group_count": 87,
            "context_group_count": 9,
            "identity_bypass": False,
        },
    }
    record.update(overrides)
    record["record_sha256"] = record_sha256(record)
    return record


def build_world(
    *,
    challenge_wrong=(),
    control_wrong=(),
    parse_fail=(),
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    manifest: list[dict] = []
    dense: list[dict] = []
    control: list[dict] = []
    oracle: list[dict] = []
    challenge_wrong = set(challenge_wrong)
    control_wrong = set(control_wrong)
    parse_fail = set(parse_fail)
    for cell in CELLS:
        for stratum in STRATA:
            for rank in range(2):
                sample_id = _sample_id(cell, stratum, rank)
                subdataset = _subdataset(cell, stratum, rank)
                manifest.append(
                    {
                        "sample_id": sample_id,
                        "task": cell.task,
                        "source": cell.source,
                        "source_dataset": subdataset,
                        "choices": ["0", "1", "2", "3"],
                        "answer": "(A)",
                        "oracle_pilot": {
                            "cell": cell.label(),
                            "stratum": stratum,
                            "dense_correct": True,
                            "control_correct": stratum == CONTROL,
                            "selection_sha256_rank": stable_sha256_rank(sample_id),
                        },
                    }
                )
                dense.append({"sample_id": sample_id, "output_text": "(A)"})
                control.append(
                    {
                        "sample_id": sample_id,
                        "output_text": "(A)" if stratum == CONTROL else "(B)",
                    }
                )
                correct = True
                if stratum == CHALLENGE and sample_id in challenge_wrong:
                    correct = False
                if stratum == CONTROL and sample_id in control_wrong:
                    correct = False
                oracle.append(
                    oracle_record(
                        sample_id,
                        correct=correct,
                        parse_ok=sample_id not in parse_fail,
                    )
                )
    return manifest, dense, control, oracle


def _challenge_ids() -> list[str]:
    return [
        _sample_id(cell, CHALLENGE, rank)
        for cell in CELLS
        for rank in range(2)
    ]


def _control_ids() -> list[str]:
    return [
        _sample_id(cell, CONTROL, rank) for cell in CELLS for rank in range(2)
    ]


class SummarizerDecisionTest(unittest.TestCase):
    def _analyze(self, manifest, dense, control, oracle):
        return analyze(
            manifest_rows=manifest,
            dense_rows=dense,
            control_rows=control,
            oracle_rows=oracle,
            manifest_sha256=A_SHA,
            config_sha256=B_SHA,
        )

    def test_full_repair_is_go(self) -> None:
        result = self._analyze(*build_world())
        decision = result["decision"]
        self.assertEqual(decision["status"], "PILOT_GO")
        self.assertEqual(decision["challenge_repairs"], 12)
        self.assertEqual(decision["control_regressions"], 0)
        self.assertEqual(decision["oracle_parse_count"], 24)
        self.assertEqual(decision["layout_contract_pass_count"], 24)
        self.assertTrue(all(decision["checks"].values()))
        self.assertFalse(decision["uses_confidence_interval"])
        self.assertFalse(decision["is_official_cv_bench_accuracy"])
        self.assertTrue(decision["dense_is_only_eligibility_reference"])

    def test_too_few_repairs_is_stop(self) -> None:
        wrong = _challenge_ids()[:5]
        result = self._analyze(*build_world(challenge_wrong=wrong))
        decision = result["decision"]
        self.assertEqual(decision["challenge_repairs"], 7)
        self.assertEqual(decision["status"], "PILOT_STOP")
        self.assertFalse(decision["checks"]["challenge_repairs_ge_min"])

    def test_too_many_regressions_is_stop(self) -> None:
        wrong = _control_ids()[:2]
        result = self._analyze(*build_world(control_wrong=wrong))
        decision = result["decision"]
        self.assertEqual(decision["control_regressions"], 2)
        self.assertEqual(decision["status"], "PILOT_STOP")
        self.assertFalse(decision["checks"]["control_regressions_le_max"])

    def test_one_regression_is_allowed(self) -> None:
        result = self._analyze(*build_world(control_wrong=_control_ids()[:1]))
        self.assertEqual(result["decision"]["control_regressions"], 1)
        self.assertEqual(result["decision"]["status"], "PILOT_GO")

    def test_parse_failure_is_stop(self) -> None:
        result = self._analyze(*build_world(parse_fail=_challenge_ids()[:1]))
        decision = result["decision"]
        self.assertEqual(decision["oracle_parse_count"], 23)
        self.assertEqual(decision["status"], "PILOT_STOP")
        self.assertFalse(decision["checks"]["oracle_parse_is_24"])

    def test_per_cell_counts_and_transitions(self) -> None:
        result = self._analyze(*build_world(challenge_wrong=_challenge_ids()[:1]))
        self.assertEqual(result["transitions"]["W->R"], 11)
        self.assertEqual(result["transitions"]["unchanged_wrong"], 1)
        for cell in CELLS:
            item = result["per_cell"][cell.label()]
            self.assertEqual(item["sample_count"], 4)
            self.assertEqual(item["challenge_count"], 2)
            self.assertEqual(item["control_count"], 2)

    def test_omni3d_subdataset_counts(self) -> None:
        result = self._analyze(*build_world())
        counts = result["manifest_structure"]["omni3d_subdataset_counts"]
        self.assertGreaterEqual(counts["Omni3D_Hypersim"], 2)
        self.assertGreaterEqual(counts["Omni3D_SUNRGBD"], 2)
        self.assertGreaterEqual(counts["Omni3D_nuScenes"], 2)

    def test_dense_control_supersets_are_allowed(self) -> None:
        manifest, dense, control, oracle = build_world()
        dense = dense + [{"sample_id": "extra-dense", "output_text": "(A)"}]
        control = control + [{"sample_id": "extra-control", "output_text": "(B)"}]
        result = self._analyze(manifest, dense, control, oracle)
        self.assertEqual(result["decision"]["status"], "PILOT_GO")

    def test_extra_oracle_sample_fails_closed(self) -> None:
        manifest, dense, control, oracle = build_world()
        oracle = oracle + [oracle_record("stale-extra")]
        with self.assertRaises(InputContractError):
            self._analyze(manifest, dense, control, oracle)

    def test_no_official_accuracy_fields(self) -> None:
        result = self._analyze(*build_world())
        self.assertNotIn("cv_bench_accuracy", result)
        self.assertNotIn("accuracy", {key for key in result})
        self.assertNotIn("significance", str(result))
        self.assertFalse(result["decision"]["is_official_cv_bench_accuracy"])


class SummarizerMalformedTest(unittest.TestCase):
    def _analyze(self, manifest, dense, control, oracle):
        return analyze(
            manifest_rows=manifest,
            dense_rows=dense,
            control_rows=control,
            oracle_rows=oracle,
            manifest_sha256=A_SHA,
            config_sha256=B_SHA,
        )

    def test_wrong_arm_fails_closed(self) -> None:
        world = build_world()
        world[3][0]["arm"] = "other"
        world[3][0]["record_sha256"] = record_sha256(world[3][0])
        with self.assertRaises(InputContractError):
            self._analyze(*world)

    def test_layout_failure_fails_closed(self) -> None:
        world = build_world()
        world[3][0]["layout"]["passed"] = False
        world[3][0]["record_sha256"] = record_sha256(world[3][0])
        with self.assertRaises(InputContractError):
            self._analyze(*world)

    def test_layout_count_mismatch_fails_closed(self) -> None:
        world = build_world()
        world[3][0]["layout"]["context_group_count"] = 8
        world[3][0]["record_sha256"] = record_sha256(world[3][0])
        with self.assertRaises(InputContractError):
            self._analyze(*world)

    def test_tampered_record_hash_fails_closed(self) -> None:
        world = build_world()
        world[3][0]["output_text"] = "(B)"
        with self.assertRaises(InputContractError):
            self._analyze(*world)

    def test_missing_sample_fails_closed(self) -> None:
        world = build_world()
        oracle = world[3][:-1]
        with self.assertRaises(InputContractError):
            self._analyze(world[0], world[1], world[2], oracle)

    def test_label_mismatch_fails_closed(self) -> None:
        world = build_world()
        manifest = world[0]
        first = next(
            row
            for row in manifest
            if row["oracle_pilot"]["stratum"] == CHALLENGE
        )
        first["oracle_pilot"]["stratum"] = CONTROL
        with self.assertRaises(InputContractError):
            self._analyze(manifest, world[1], world[2], world[3])

    def test_dense_eligibility_mismatch_fails_closed(self) -> None:
        world = build_world()
        dense = world[1]
        dense[0]["output_text"] = "(B)"
        with self.assertRaises(InputContractError):
            self._analyze(world[0], dense, world[2], world[3])


if __name__ == "__main__":
    unittest.main()
