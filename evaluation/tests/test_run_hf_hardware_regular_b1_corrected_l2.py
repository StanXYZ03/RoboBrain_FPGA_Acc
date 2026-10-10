#!/usr/bin/env python3
"""CPU-only contract tests for the B1 corrected-L2 GPU runner.

The runner itself is only executed on a GPU host; these tests pin its frozen
contract (one of three arms, unified frozen B1 forward, strict offline local
model, fail-closed resume) and its raw-record digest.  No CUDA or model is
touched.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import run_hf_hardware_regular_b1_corrected_l2 as runner  # noqa: E402
from evaluation.hardware_regular_b1_corrected_l2 import (  # noqa: E402
    ARM_FLOAT,
    ARM_INTEGER,
    ARM_PROMPT_INTEGER,
)

MANIFEST_SHA = "a" * 64
CHECKPOINT_SHA = "b" * 64

REQUIRED_ARGS = [
    "--model",
    "model_dir",
    "--model-label",
    "label",
    "--manifest",
    "manifest.jsonl",
    "--output",
    "out.jsonl",
    "--arm",
    ARM_FLOAT,
]


def _sample(sample_id: str, *, bypass: bool = False) -> dict:
    if bypass:
        grid = [1, 16, 16]
    else:
        grid = [1, 60, 80]
    return {"sample_id": sample_id, "task": "Count", "image_grid_thw": grid}


def _record(
    sample_id: str,
    *,
    arm: str = ARM_FLOAT,
    manifest_sha256: str = MANIFEST_SHA,
    checkpoint_sha256: str | None = None,
    wv_sha256: str | None = None,
    wv_scale_sha256: str | None = None,
    budget: int = 96,
    region_grid: int = 3,
    local_candidate_cap: int = 16,
    bypass: bool = False,
    break_hash: bool = False,
    **overrides,
) -> dict:
    if bypass:
        full_patch_tokens, full_llm = 256, 64
    else:
        full_patch_tokens, full_llm = 4800, 1200
    output_group_count = full_llm if bypass else budget
    record: dict = {
        "sample_id": sample_id,
        "arm": arm,
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "wv_sha256": wv_sha256,
        "wv_scale_sha256": wv_scale_sha256,
        "requested_budget": budget,
        "region_grid": region_grid,
        "local_candidate_cap": local_candidate_cap,
        "output_text": "The answer is (B).",
        "output_token_ids": [1, 2, 3],
        "full_patch_tokens": full_patch_tokens,
        "image_grid_thw": [1, 16, 16] if bypass else [1, 60, 80],
        "full_llm_visual_tokens": full_llm,
        "output_group_count": output_group_count,
        "context_group_count": 0,
        "real_group_count": output_group_count,
        "identity_bypass": bypass,
        "selected_group_indices": list(range(output_group_count)),
    }
    record.update(overrides)
    record["record_sha256"] = runner.record_sha256(record)
    if break_hash:
        record["output_text"] = "tampered"
    return record


def _write(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _validate(path: Path, samples, **kwargs):
    defaults = dict(
        arm=ARM_FLOAT,
        manifest_sha256=MANIFEST_SHA,
        checkpoint_identity={
            "checkpoint_sha256": None,
            "wv_sha256": None,
            "wv_scale_sha256": None,
        },
        budget=96,
        region_grid=3,
        local_candidate_cap=16,
    )
    defaults.update(kwargs)
    return runner.validate_resume(path, samples, **defaults)


class RunnerContractTest(unittest.TestCase):
    def test_arm_is_required_and_restricted(self) -> None:
        with self.assertRaises(SystemExit):
            runner.build_parser().parse_args(REQUIRED_ARGS[:-2])
        for arm in ("l2", "integer", "prompt", "corrected_l2_typo"):
            with self.subTest(arm=arm):
                with self.assertRaises(SystemExit):
                    runner.build_parser().parse_args(
                        REQUIRED_ARGS + ["--arm", arm]
                    )

    def test_defaults_are_frozen(self) -> None:
        args = runner.build_parser().parse_args(REQUIRED_ARGS)
        self.assertEqual(args.arm, ARM_FLOAT)
        self.assertEqual(args.budget, 96)
        self.assertEqual(args.region_grid, 3)
        self.assertEqual(args.local_candidate_cap, 16)
        self.assertEqual(args.dtype, "bf16")
        self.assertIsNone(args.limit_per_task)

    def test_fp16_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            runner.build_parser().parse_args(REQUIRED_ARGS + ["--dtype", "fp16"])

    def test_source_uses_unified_b1_forward_and_offline_model(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        for token in (
            "forward_hardware_regular_b1_vision",
            "SCORE_MODE_INTEGER",
            "integer_scorer",
            "local_files_only=True",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "greedy_argmax",
            "record_sha256",
        ):
            self.assertIn(token, source)

    def test_arm_checkpoint_identity(self) -> None:
        for arm in (ARM_FLOAT, ARM_INTEGER):
            with self.subTest(arm=arm):
                Wt, wv_q, identity = runner.load_arm_checkpoint(arm)
                self.assertIsNone(Wt)
                self.assertIsNone(wv_q)
                self.assertIsNone(identity["checkpoint_sha256"])
        Wt, wv_q, identity = runner.load_arm_checkpoint(ARM_PROMPT_INTEGER)
        self.assertEqual(Wt.shape, (128, 2560))
        self.assertEqual(wv_q.shape, (128, 1024))
        self.assertEqual(
            identity["checkpoint_sha256"],
            "1f1d379e3229f310d11fee4ff620480e13fea9005abefc762c66f0d331ca72ff",
        )
        self.assertEqual(
            identity["wv_sha256"],
            "126bd8421dd128a318115c301bf9d1c7d336912150d19d2ec35203ed55d89d3b",
        )

    def test_record_sha256_is_canonical_and_sensitive(self) -> None:
        record = _record("s0")
        digest = runner.record_sha256(record)
        reordered = {key: record[key] for key in reversed(list(record))}
        self.assertEqual(runner.record_sha256(reordered), digest)
        changed = dict(record)
        changed["output_text"] = "different"
        self.assertNotEqual(runner.record_sha256(changed), digest)


class ResumeContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.samples = [_sample("s0"), _sample("s1"), _sample("s2", bypass=True)]

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_missing_and_empty_file_yield_empty_set(self) -> None:
        missing = self.tmp / "missing.jsonl"
        self.assertEqual(_validate(missing, self.samples), set())
        empty = self.tmp / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        self.assertEqual(_validate(empty, self.samples), set())

    def test_valid_records_resume(self) -> None:
        path = self.tmp / "out.jsonl"
        _write(path, [_record("s0"), _record("s2", bypass=True)])
        self.assertEqual(_validate(path, self.samples), {"s0", "s2"})

    def test_duplicate_id_fails_closed(self) -> None:
        path = self.tmp / "out.jsonl"
        _write(path, [_record("s0"), _record("s0")])
        with self.assertRaises(runner.ResumeContractError):
            _validate(path, self.samples)

    def test_unknown_id_fails_closed(self) -> None:
        path = self.tmp / "out.jsonl"
        _write(path, [_record("not-in-manifest")])
        with self.assertRaises(runner.ResumeContractError):
            _validate(path, self.samples)

    def test_wrong_arm_fails_closed(self) -> None:
        path = self.tmp / "out.jsonl"
        _write(path, [_record("s0", arm=ARM_INTEGER)])
        with self.assertRaises(runner.ResumeContractError):
            _validate(path, self.samples)

    def test_stale_manifest_hash_fails_closed(self) -> None:
        path = self.tmp / "out.jsonl"
        _write(path, [_record("s0", manifest_sha256="c" * 64)])
        with self.assertRaises(runner.ResumeContractError):
            _validate(path, self.samples)

    def test_checkpoint_hash_mismatch_fails_closed(self) -> None:
        path = self.tmp / "out.jsonl"
        _write(
            path,
            [
                _record(
                    "s0",
                    arm=ARM_PROMPT_INTEGER,
                    checkpoint_sha256=CHECKPOINT_SHA,
                    wv_sha256="d" * 64,
                    wv_scale_sha256="e" * 64,
                )
            ],
        )
        with self.assertRaises(runner.ResumeContractError):
            _validate(
                path,
                self.samples,
                arm=ARM_PROMPT_INTEGER,
                checkpoint_identity={
                    "checkpoint_sha256": CHECKPOINT_SHA,
                    "wv_sha256": "f" * 64,
                    "wv_scale_sha256": "e" * 64,
                },
            )

    def test_wrong_budget_grid_cap_fail_closed(self) -> None:
        for kwargs in (
            {"budget": 64},
            {"region_grid": 2},
            {"local_candidate_cap": 8},
        ):
            with self.subTest(kwargs=kwargs):
                path = self.tmp / "out.jsonl"
                _write(path, [_record("s0", **kwargs)])
                with self.assertRaises(runner.ResumeContractError):
                    _validate(path, self.samples)

    def test_wrong_k_and_grid_fail_closed(self) -> None:
        for record in (
            _record("s0", output_group_count=95),
            _record("s0", full_patch_tokens=4799),
            _record("s0", image_grid_thw=[1, 60, 79]),
            _record("s0", selected_group_indices=[0, 1, 2]),
        ):
            with self.subTest(record=record["output_group_count"]):
                path = self.tmp / "out.jsonl"
                _write(path, [record])
                with self.assertRaises(runner.ResumeContractError):
                    _validate(path, self.samples)

    def test_tampered_record_hash_fails_closed(self) -> None:
        path = self.tmp / "out.jsonl"
        _write(path, [_record("s0", break_hash=True)])
        with self.assertRaises(runner.ResumeContractError):
            _validate(path, self.samples)

    def test_partial_record_fails_closed(self) -> None:
        record = _record("s0")
        del record["output_token_ids"]
        record["record_sha256"] = runner.record_sha256(record)
        path = self.tmp / "out.jsonl"
        _write(path, [record])
        with self.assertRaises(runner.ResumeContractError):
            _validate(path, self.samples)

    def test_invalid_json_fails_closed(self) -> None:
        path = self.tmp / "out.jsonl"
        path.write_text("{not json}\n", encoding="utf-8")
        with self.assertRaises(runner.ResumeContractError):
            _validate(path, self.samples)


if __name__ == "__main__":
    unittest.main()
