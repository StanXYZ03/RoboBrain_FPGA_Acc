#!/usr/bin/env python3
"""CPU-only contract tests for the layer-5 context-pooling ablation runner.

The runner itself is only executed on a GPU host; these tests pin its frozen
contract: cut point 5, score mode ``l2`` only, no projection/training knobs, the
single-variable ablation declaration and the B1/layer-5 field structure.  No
CUDA is required or touched.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import (  # noqa: E402
    run_hf_hardware_regular_layer5_context_ablation as runner,
)

REQUIRED_ARGS = [
    "--model",
    "model",
    "--model-label",
    "label",
    "--manifest",
    "manifest.jsonl",
    "--output",
    "out.jsonl",
    "--budget",
    "96",
]


class RunnerContractTest(unittest.TestCase):
    def test_frozen_constants(self) -> None:
        self.assertEqual(runner.SCORE_MODE, "l2")
        self.assertEqual(runner.PRUNE_AFTER_LAYER, 5)
        self.assertEqual(runner.CUTPOINT_PREFIX_BLOCKS, (0, 1, 2, 3, 4, 5))
        self.assertEqual(
            runner.ABLATION, "layer5_context_raw_representative_group"
        )
        self.assertEqual(
            runner.CONTEXT_GROUP_VALUES_SOURCE, "representative_raw_group"
        )
        self.assertIn("2026-10-04-layer5-l2-diagnostic", runner.BASELINE_POOLED_JSONL)

    def test_defaults_are_l2_and_layer5(self) -> None:
        args = runner.build_parser().parse_args(REQUIRED_ARGS)
        self.assertEqual(args.score_mode, "l2")
        self.assertEqual(args.prune_after_layer, 5)
        self.assertEqual(args.region_grid, 3)
        self.assertEqual(args.local_candidate_cap, 16)
        self.assertEqual(args.dtype, "bf16")
        self.assertEqual(args.device, "cuda:0")

    def test_other_score_modes_are_rejected(self) -> None:
        for mode in ("prompt", "integer", "float"):
            with self.subTest(mode=mode):
                with self.assertRaises(SystemExit):
                    runner.build_parser().parse_args(
                        REQUIRED_ARGS + ["--score-mode", mode]
                    )

    def test_other_cutpoints_are_rejected(self) -> None:
        for cut in ("0", "4", "6", "23", "-1"):
            with self.subTest(cut=cut):
                with self.assertRaises(SystemExit):
                    runner.build_parser().parse_args(
                        REQUIRED_ARGS + ["--prune-after-layer", cut]
                    )

    def test_projection_knob_does_not_exist(self) -> None:
        with self.assertRaises(SystemExit):
            runner.build_parser().parse_args(
                REQUIRED_ARGS + ["--projection-npz", "x.npz"]
            )

    def test_no_projection_or_training_helpers_are_exposed(self) -> None:
        for name in (
            "load_projection",
            "canonical_projection_sha256",
            "PROJECTION_WT_KEY",
            "PROJECTION_WV_KEY",
        ):
            self.assertFalse(hasattr(runner, name), name)

    def test_run_one_rejects_non_l2_before_touching_cuda(self) -> None:
        with self.assertRaises(ValueError):
            runner.run_one(
                None,
                None,
                {"sample_id": "s", "task": "t", "image": "i", "prompt": "p"},
                Path("manifest.jsonl"),
                budget=96,
                score_mode="prompt",
                region_grid=3,
                local_candidate_cap=16,
            )

    def test_file_sha256_matches_hashlib(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.bin"
            path.write_bytes(b"layer5-context-ablation")
            self.assertEqual(
                runner.file_sha256(path),
                hashlib.sha256(b"layer5-context-ablation").hexdigest(),
            )


class RunnerSourceContractTest(unittest.TestCase):
    def test_source_mentions_ablation_and_frozen_selection(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        for token in (
            "forward_hardware_regular_layer5_context_ablation_vision",
            "prepare_layer5_context_ablation_multimodal_inputs",
            "context_group_values_source",
            "uses_pooled_context_group_values",
            "selection_metadata_identical_to_frozen_pooled",
            "prune_after_layer",
            "greedy_argmax",
        ):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
