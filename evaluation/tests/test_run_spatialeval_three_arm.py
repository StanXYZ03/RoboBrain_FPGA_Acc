from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation import run_spatialeval_three_arm as runner
from evaluation import spatialeval_external_contract as contract


def sample() -> dict:
    return {
        "sample_id": "spatialmap.vqa.0.0",
        "task": "spatialmap",
        "question_id": 0,
        "image": "/tmp/image.png",
        "image_sha256": "a" * 64,
        "prompt": "question",
        "max_new_tokens": 64,
        "oracle_answer": "northwest",
        "oracle_option": "A",
        "oracle_full_answer": "A. northwest",
    }


def frozen_dense() -> dict:
    return {
        "output_text": "northwest",
        "output_tokens": 1,
        "image_grid_thw": [1, 20, 20],
        "selected_group_indices": list(range(100)),
        "actual_token_retention": 1.0,
        "llm_visual_tokens": 100,
    }


class RunSpatialEvalThreeArmTests(unittest.TestCase):
    def test_normalize_dense_record(self) -> None:
        row = runner.normalize_output_record(
            contract.ARM_DENSE,
            sample(),
            frozen_dense(),
            manifest_sha256="b" * 64,
            run_metadata={"test": True},
        )
        self.assertEqual(row["output_group_count"], 100)
        self.assertEqual(row["output_token_count"], 1)
        self.assertFalse(row["output_token_ids_available"])
        self.assertFalse(row["uses_labels"])
        self.assertEqual(row["record_sha256"], contract.canonical_record_sha256(row))

    def test_dense_count_mismatch_fails(self) -> None:
        frozen = frozen_dense()
        frozen["selected_group_indices"] = list(range(99))
        with self.assertRaises(contract.SpatialEvalExternalContractError):
            runner.normalize_output_record(
                contract.ARM_DENSE,
                sample(),
                frozen,
                manifest_sha256="b" * 64,
                run_metadata={},
            )

    def test_grid_over_edge_budget_fails(self) -> None:
        with self.assertRaisesRegex(
            contract.SpatialEvalExternalContractError, "G="
        ):
            runner.validate_actual_grid([1, 50, 50])

    def test_resume_detects_corruption(self) -> None:
        row = runner.normalize_output_record(
            contract.ARM_DENSE,
            sample(),
            frozen_dense(),
            manifest_sha256="b" * 64,
            run_metadata={},
        )
        row["output_text"] = "tampered"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.jsonl"
            output.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                contract.SpatialEvalExternalContractError, "hash mismatch"
            ):
                runner.validate_resume(
                    output,
                    [sample()],
                    arm=contract.ARM_DENSE,
                    manifest_sha256="b" * 64,
                )

    def test_resume_rejects_wrong_arm(self) -> None:
        row = runner.normalize_output_record(
            contract.ARM_DENSE,
            sample(),
            frozen_dense(),
            manifest_sha256="b" * 64,
            run_metadata={},
        )
        row["arm"] = contract.ARM_V2
        row["record_sha256"] = contract.canonical_record_sha256(row)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.jsonl"
            output.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                contract.SpatialEvalExternalContractError, "mismatched"
            ):
                runner.validate_resume(
                    output,
                    [sample()],
                    arm=contract.ARM_DENSE,
                    manifest_sha256="b" * 64,
                )


if __name__ == "__main__":
    unittest.main()
