#!/usr/bin/env python3
"""CPU-only focused tests for the edge576 three-arm runner adapter.

Covers the required contract without any CUDA, model weights or network:
three-arm dispatch to frozen implementations, per-sample ``image_grid_thw``
fail-closed assertion, duplicate sample ids, safe resume, ``parsed_answer=null``
records, the locked parameters / pinned r2 output paths, the manifest pin,
the anti-label-leakage sanitisation, the run-mode preflight order, the
complete-resume short-circuit, the real multimodal warmup (no output written)
and the fail-closed model identity verification.  The frozen implementations
are replaced by fakes; the real 2638-row manifest, the model identity manifest
and the SHA256 dependency pins are verified for real because that needs no
model weights.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import edge576_cvbench_runner_contract as contract  # noqa: E402
from evaluation import run_edge576_cvbench_three_arm as runner  # noqa: E402

MANIFEST_SHA = contract.EDGE576_MANIFEST_SHA256


def make_sample(
    sample_id: str,
    grid: tuple[int, int, int] = (1, 16, 16),
    *,
    prompt: str = "How many? (A) 1 (B) 2",
    answer: str = "(B)",
    choices: tuple[str, ...] = ("1", "2"),
) -> dict:
    prod = grid[0] * grid[1] * grid[2]
    return {
        "sample_id": sample_id,
        "task": "Count",
        "type": "2D",
        "source": "ADE20K",
        "source_dataset": "ADE20K Validation Set",
        "image": "img/0.image",
        "prompt": prompt,
        "answer": answer,
        "choices": list(choices),
        "max_new_tokens": 16,
        "image_grid_thw": list(grid),
        "full_patch_tokens": prod,
        "full_llm_visual_tokens": prod // contract.MERGE_UNIT,
    }


def fake_frozen(
    sample: dict,
    arm: str,
    *,
    grid: tuple[int, int, int] | None = None,
    output_text: str = "The answer is (B).",
    identity_bypass: bool | None = None,
) -> dict:
    actual_grid = list(sample["image_grid_thw"]) if grid is None else list(grid)
    full_llm = int(sample["full_llm_visual_tokens"])
    if identity_bypass is None:
        identity_bypass = True
    selected = list(range(full_llm)) if identity_bypass else list(range(96))
    record: dict = {
        "image_grid_thw": actual_grid,
        "output_text": output_text,
        "output_token_ids": [1, 2, 3],
        "selected_group_indices": selected,
    }
    if arm == contract.ARM_DENSE:
        record["actual_token_retention"] = 1.0
        record["llm_visual_tokens"] = full_llm
    if arm in (contract.ARM_L2, contract.ARM_V2):
        record["requested_budget"] = 96
        record["region_grid"] = 3
        record["local_candidate_cap"] = 16
        record["identity_bypass"] = bool(identity_bypass)
    if arm == contract.ARM_V2:
        record["max_swaps"] = 16
        record["swap_count"] = 0
        record["utility_formula"] = "semantic_q15 + corrected_l2_q15"
        record["legal_gain_formula"] = (
            "semantic_gain - l2_loss = utility_added - utility_removed"
        )
        record["uses_labels"] = False
        record["uses_answer_tokens"] = False
        record["uses_gradients"] = False
    return record


def normalized(
    arm: str, sample: dict, *, grid=None, output_text="The answer is (B)."
) -> dict:
    return runner.normalize_record(
        arm,
        sample,
        fake_frozen(sample, arm, grid=grid, output_text=output_text),
        manifest_sha256=MANIFEST_SHA,
        run_metadata={"arm": arm},
        run_mode="full2638",
    )


def write_records(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


class ArmDispatchTest(unittest.TestCase):
    def test_adapter_table_covers_all_arms(self) -> None:
        self.assertEqual(
            set(runner._ARM_ADAPTERS), set(contract.ARMS)
        )
        self.assertEqual(len(set(runner._ARM_ADAPTERS.values())), 3)

    def test_dispatch_selects_only_the_requested_frozen_adapter(self) -> None:
        calls: dict[str, dict] = {}
        originals = (runner._run_dense_one, runner._run_l2_one, runner._run_v2_one)

        def _recorder(name):
            def _call(model, processor, sample, manifest, **kwargs):
                calls[name] = kwargs
                return {"name": name}

            return _call

        runner._run_dense_one = _recorder("dense")
        runner._run_l2_one = _recorder("l2")
        runner._run_v2_one = _recorder("v2")
        try:
            for arm, expected in (
                (contract.ARM_DENSE, "dense"),
                (contract.ARM_L2, "l2"),
                (contract.ARM_V2, "v2"),
            ):
                calls.clear()
                result = runner.run_one_for_arm(
                    arm,
                    None,
                    None,
                    make_sample("s0"),
                    Path("manifest.jsonl"),
                    manifest_sha256=MANIFEST_SHA,
                    scorer="SCORER",
                    checkpoint_identity={"checkpoint_sha256": None},
                    run_mode="full2638",
                )
                self.assertEqual(result["name"], expected)
                self.assertEqual(list(calls), [expected])
                self.assertEqual(calls[expected]["manifest_sha256"], MANIFEST_SHA)
                self.assertEqual(calls[expected]["scorer"], "SCORER")
                self.assertEqual(calls[expected]["run_mode"], "full2638")
        finally:
            (
                runner._run_dense_one,
                runner._run_l2_one,
                runner._run_v2_one,
            ) = originals

    def test_unknown_arm_fails_closed(self) -> None:
        with self.assertRaises(contract.Edge576ContractError):
            runner.run_one_for_arm(
                "not_an_arm",
                None,
                None,
                make_sample("s0"),
                Path("m.jsonl"),
                manifest_sha256=MANIFEST_SHA,
                scorer=None,
                checkpoint_identity={},
                run_mode="full2638",
            )

    def test_adapters_really_call_frozen_run_one(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        for token in (
            "from evaluation.run_hf_shallow_block_pruning import",
            "from evaluation.run_hf_hardware_regular_b1_corrected_l2 import",
            "from evaluation.run_hf_conservative_l2_guarded_rerank import",
            "run_one as frozen_run_one",
            "retention=1.0",
            "ARM_INTEGER",
            "max_swaps=MAX_SWAPS",
        ):
            self.assertIn(token, source)


class GridAssertionTest(unittest.TestCase):
    def test_matching_grid_passes(self) -> None:
        sample = make_sample("s0")
        contract.assert_edge576_grid(sample, [1, 16, 16])

    def test_mismatched_grid_fails_closed(self) -> None:
        with self.assertRaises(contract.Edge576ContractError):
            contract.assert_edge576_grid(make_sample("s0"), [1, 16, 32])

    def test_missing_grid_fails_closed(self) -> None:
        with self.assertRaises(contract.Edge576ContractError):
            contract.assert_edge576_grid(make_sample("s0"), None)

    def test_execute_arm_aborts_before_writing_on_grid_mismatch(self) -> None:
        samples = [make_sample("s0")]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.jsonl"

            def bad_run_one(model, processor, sample, manifest):
                return fake_frozen(sample, contract.ARM_DENSE, grid=(1, 16, 32))

            with self.assertRaises(contract.Edge576ContractError):
                runner.execute_arm(
                    arm=contract.ARM_DENSE,
                    model=None,
                    processor=None,
                    samples=samples,
                    output=output,
                    manifest_path=Path("manifest.jsonl"),
                    manifest_sha256=MANIFEST_SHA,
                    run_one=bad_run_one,
                    run_metadata={},
                    run_mode="full2638",
                )
            self.assertFalse(output.exists() and output.read_text())

    def test_normalize_rejects_incomplete_frozen_record(self) -> None:
        with self.assertRaises(contract.Edge576ContractError):
            runner.normalize_record(
                contract.ARM_DENSE,
                make_sample("s0"),
                {"output_text": "x"},
                manifest_sha256=MANIFEST_SHA,
                run_metadata={},
                run_mode="full2638",
            )

    def test_normalize_rejects_missing_core_output_text(self) -> None:
        frozen = fake_frozen(make_sample("s0"), contract.ARM_DENSE)
        del frozen["output_text"]
        with self.assertRaises(contract.Edge576ContractError):
            runner.normalize_record(
                contract.ARM_DENSE,
                make_sample("s0"),
                frozen,
                manifest_sha256=MANIFEST_SHA,
                run_metadata={},
                run_mode="full2638",
            )

    def test_normalize_rejects_missing_core_selected_groups(self) -> None:
        frozen = fake_frozen(make_sample("s0"), contract.ARM_DENSE)
        del frozen["selected_group_indices"]
        with self.assertRaises(contract.Edge576ContractError):
            runner.normalize_record(
                contract.ARM_DENSE,
                make_sample("s0"),
                frozen,
                manifest_sha256=MANIFEST_SHA,
                run_metadata={},
                run_mode="full2638",
            )


class OutputTokenMappingTest(unittest.TestCase):
    """Dense/L2/v2 token-id behaviour: optional ids, mandatory real count."""

    def _normalize(self, arm, frozen, sample=None):
        sample = sample or make_sample("s0")
        return runner.normalize_record(
            arm,
            sample,
            frozen,
            manifest_sha256=MANIFEST_SHA,
            run_metadata={"arm": arm},
            run_mode="full2638",
        )

    def test_dense_without_token_ids_maps_count_from_output_tokens(self) -> None:
        frozen = fake_frozen(make_sample("s0"), contract.ARM_DENSE)
        frozen.pop("output_token_ids")
        frozen["output_tokens"] = 3
        record = self._normalize(contract.ARM_DENSE, frozen)
        self.assertNotIn("output_token_ids", record)
        self.assertFalse(record["output_token_ids_available"])
        self.assertEqual(record["output_token_count"], 3)
        self.assertEqual(
            record["record_sha256"], contract.record_sha256(record)
        )

    def test_dense_never_fabricates_token_ids(self) -> None:
        frozen = fake_frozen(make_sample("s0"), contract.ARM_DENSE)
        frozen.pop("output_token_ids")
        frozen["output_tokens"] = 0
        record = self._normalize(contract.ARM_DENSE, frozen)
        self.assertNotIn("output_token_ids", record)
        self.assertEqual(record["output_token_count"], 0)

    def test_l2_keeps_real_token_ids_and_count(self) -> None:
        frozen = fake_frozen(make_sample("s0"), contract.ARM_L2)
        frozen["output_tokens"] = 3
        record = self._normalize(contract.ARM_L2, frozen)
        self.assertEqual(record["output_token_ids"], [1, 2, 3])
        self.assertTrue(record["output_token_ids_available"])
        self.assertEqual(record["output_token_count"], 3)

    def test_v2_keeps_real_token_ids_and_count(self) -> None:
        frozen = fake_frozen(make_sample("s0"), contract.ARM_V2)
        frozen["output_tokens"] = 3
        record = self._normalize(contract.ARM_V2, frozen)
        self.assertEqual(record["output_token_ids"], [1, 2, 3])
        self.assertTrue(record["output_token_ids_available"])
        self.assertEqual(record["output_token_count"], 3)

    def test_token_id_count_mismatch_fails_closed(self) -> None:
        frozen = fake_frozen(make_sample("s0"), contract.ARM_L2)
        frozen["output_tokens"] = 4
        with self.assertRaises(contract.Edge576ContractError):
            self._normalize(contract.ARM_L2, frozen)

    def test_missing_token_fields_fails_closed(self) -> None:
        frozen = fake_frozen(make_sample("s0"), contract.ARM_L2)
        frozen.pop("output_token_ids")
        with self.assertRaises(contract.Edge576ContractError):
            self._normalize(contract.ARM_L2, frozen)

    def test_non_integer_token_id_fails_closed(self) -> None:
        frozen = fake_frozen(make_sample("s0"), contract.ARM_L2)
        frozen["output_token_ids"] = [1, "2", 3]
        with self.assertRaises(contract.Edge576ContractError):
            self._normalize(contract.ARM_L2, frozen)

    def test_resume_accepts_dense_record_without_token_ids(self) -> None:
        sample = make_sample("s0")
        frozen = fake_frozen(sample, contract.ARM_DENSE)
        frozen.pop("output_token_ids")
        frozen["output_tokens"] = 5
        record = self._normalize(contract.ARM_DENSE, frozen)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.jsonl"
            write_records(output, [record])
            done = contract.validate_resume_edge576(
                output,
                [sample],
                arm=contract.ARM_DENSE,
                manifest_sha256=MANIFEST_SHA,
            )
        self.assertEqual(done, {"s0"})

    def test_resume_rejects_missing_output_token_count(self) -> None:
        sample = make_sample("s0")
        record = normalized(contract.ARM_DENSE, sample)
        del record["output_token_count"]
        record["record_sha256"] = contract.record_sha256(record)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.jsonl"
            write_records(output, [record])
            with self.assertRaises(contract.Edge576ContractError):
                contract.validate_resume_edge576(
                    output,
                    [sample],
                    arm=contract.ARM_DENSE,
                    manifest_sha256=MANIFEST_SHA,
                )

    def test_resume_rejects_inconsistent_token_availability(self) -> None:
        sample = make_sample("s0")
        record = normalized(contract.ARM_DENSE, sample)
        record["output_token_ids_available"] = (
            not record["output_token_ids_available"]
        )
        record["record_sha256"] = contract.record_sha256(record)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.jsonl"
            write_records(output, [record])
            with self.assertRaises(contract.Edge576ContractError):
                contract.validate_resume_edge576(
                    output,
                    [sample],
                    arm=contract.ARM_DENSE,
                    manifest_sha256=MANIFEST_SHA,
                )


class ResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.samples = [
            make_sample("s0"),
            make_sample("s1", (1, 32, 32)),
        ]

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_resume_skips_done_and_appends_only_pending(self) -> None:
        output = self.tmp / "out.jsonl"
        write_records(output, [normalized(contract.ARM_L2, self.samples[0])])
        seen: list[str] = []

        def run_one(model, processor, sample, manifest):
            seen.append(str(sample["sample_id"]))
            return fake_frozen(sample, contract.ARM_L2)

        stats = runner.execute_arm(
            arm=contract.ARM_L2,
            model=None,
            processor=None,
            samples=self.samples,
            output=output,
            manifest_path=Path("manifest.jsonl"),
            manifest_sha256=MANIFEST_SHA,
            run_one=run_one,
            run_metadata={},
            run_mode="full2638",
        )
        self.assertEqual(seen, ["s1"])
        self.assertEqual(stats["already_done"], 1)
        self.assertEqual(stats["pending"], 1)
        self.assertEqual(stats["written"], 1)
        lines = output.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(
            contract.validate_resume_edge576(
                output, self.samples, arm=contract.ARM_L2, manifest_sha256=MANIFEST_SHA
            ),
            {"s0", "s1"},
        )

    def test_duplicate_sample_id_fails_closed(self) -> None:
        output = self.tmp / "out.jsonl"
        write_records(
            output,
            [
                normalized(contract.ARM_L2, self.samples[0]),
                normalized(contract.ARM_L2, self.samples[0]),
            ],
        )
        with self.assertRaises(contract.Edge576ContractError):
            contract.validate_resume_edge576(
                output, self.samples, arm=contract.ARM_L2, manifest_sha256=MANIFEST_SHA
            )

    def test_unknown_sample_id_fails_closed(self) -> None:
        output = self.tmp / "out.jsonl"
        write_records(output, [normalized(contract.ARM_L2, make_sample("ghost"))])
        with self.assertRaises(contract.Edge576ContractError):
            contract.validate_resume_edge576(
                output, self.samples, arm=contract.ARM_L2, manifest_sha256=MANIFEST_SHA
            )

    def test_tampered_record_hash_fails_closed(self) -> None:
        record = normalized(contract.ARM_L2, self.samples[0])
        record["output_text"] = "tampered"
        output = self.tmp / "out.jsonl"
        write_records(output, [record])
        with self.assertRaises(contract.Edge576ContractError):
            contract.validate_resume_edge576(
                output, self.samples, arm=contract.ARM_L2, manifest_sha256=MANIFEST_SHA
            )

    def test_wrong_manifest_hash_fails_closed(self) -> None:
        record = normalized(contract.ARM_L2, self.samples[0])
        record["manifest_sha256"] = "0" * 64
        record["record_sha256"] = contract.record_sha256(record)
        output = self.tmp / "out.jsonl"
        write_records(output, [record])
        with self.assertRaises(contract.Edge576ContractError):
            contract.validate_resume_edge576(
                output, self.samples, arm=contract.ARM_L2, manifest_sha256=MANIFEST_SHA
            )

    def test_wrong_arm_fails_closed(self) -> None:
        output = self.tmp / "out.jsonl"
        write_records(output, [normalized(contract.ARM_V2, self.samples[0])])
        with self.assertRaises(contract.Edge576ContractError):
            contract.validate_resume_edge576(
                output, self.samples, arm=contract.ARM_L2, manifest_sha256=MANIFEST_SHA
            )

    def test_stale_grid_in_existing_record_fails_closed(self) -> None:
        record = normalized(contract.ARM_L2, self.samples[0], grid=(1, 16, 32))
        output = self.tmp / "out.jsonl"
        write_records(output, [record])
        with self.assertRaises(contract.Edge576ContractError):
            contract.validate_resume_edge576(
                output, self.samples, arm=contract.ARM_L2, manifest_sha256=MANIFEST_SHA
            )

    def test_missing_field_fails_closed(self) -> None:
        record = normalized(contract.ARM_L2, self.samples[0])
        del record["parsed_answer"]
        record["record_sha256"] = contract.record_sha256(record)
        output = self.tmp / "out.jsonl"
        write_records(output, [record])
        with self.assertRaises(contract.Edge576ContractError):
            contract.validate_resume_edge576(
                output, self.samples, arm=contract.ARM_L2, manifest_sha256=MANIFEST_SHA
            )


class NullParseTest(unittest.TestCase):
    def test_unparseable_output_is_written_as_a_complete_record(self) -> None:
        sample = make_sample("s0")
        record = normalized(
            contract.ARM_V2, sample, output_text="I cannot tell from this image."
        )
        self.assertIsNone(record["parsed_answer"])
        self.assertFalse(record["parse_success"])
        self.assertFalse(record["correct"])
        for field in contract.REQUIRED_EDGE_RECORD_FIELDS:
            self.assertIn(field, record)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.jsonl"
            stats = runner.execute_arm(
                arm=contract.ARM_V2,
                model=None,
                processor=None,
                samples=[sample],
                output=output,
                manifest_path=Path("manifest.jsonl"),
                manifest_sha256=MANIFEST_SHA,
                run_one=lambda model, processor, s, m: fake_frozen(
                    s, contract.ARM_V2, output_text="no answer here"
                ),
                run_metadata={},
                run_mode="full2638",
            )
            self.assertEqual(stats["written"], 1)
            self.assertEqual(stats["parse_failure"], 1)
            self.assertEqual(
                contract.validate_resume_edge576(
                    output,
                    [sample],
                    arm=contract.ARM_V2,
                    manifest_sha256=MANIFEST_SHA,
                ),
                {"s0"},
            )

    def test_parse_failure_counts_as_wrong(self) -> None:
        record = normalized(
            contract.ARM_V2,
            make_sample("s0"),
            output_text="nothing",
        )
        self.assertFalse(record["correct"])
        self.assertTrue(record["official_answer"])


class LabelLeakageTest(unittest.TestCase):
    def test_sanitize_strips_every_label(self) -> None:
        sample = make_sample("s0")
        sample["ground_truth"] = {"bbox_1000": [0, 0, 1, 1]}
        sample["correct"] = True
        sample["parsed_answer"] = "B"
        sample["official_answer"] = "(B)"
        sanitized = contract.sanitize_sample_for_selector(sample)
        for key in contract.FORBIDDEN_SELECTOR_KEYS:
            self.assertNotIn(key, sanitized)
        self.assertEqual(sanitized["prompt"], sample["prompt"])
        self.assertEqual(sanitized["image_grid_thw"], sample["image_grid_thw"])

    def test_execute_arm_never_forwards_labels_to_the_runner(self) -> None:
        sample = make_sample("s0")
        sample["ground_truth"] = {"bbox_1000": [0, 0, 1, 1]}
        captured: dict = {}

        def run_one(model, processor, sanitized, manifest):
            captured.update(sanitized)
            return fake_frozen(sanitized, contract.ARM_DENSE)

        with tempfile.TemporaryDirectory() as tmp:
            runner.execute_arm(
                arm=contract.ARM_DENSE,
                model=None,
                processor=None,
                samples=[sample],
                output=Path(tmp) / "out.jsonl",
                manifest_path=Path("manifest.jsonl"),
                manifest_sha256=MANIFEST_SHA,
                run_one=run_one,
                run_metadata={},
                run_mode="full2638",
            )
        for key in contract.FORBIDDEN_SELECTOR_KEYS:
            self.assertNotIn(key, captured)
        self.assertEqual(captured["prompt"], sample["prompt"])

    def test_selector_whitelist_has_no_label_keys(self) -> None:
        for key in contract.FORBIDDEN_SELECTOR_KEYS:
            self.assertNotIn(key, contract.SELECTOR_SAMPLE_KEYS)


class ParameterLockTest(unittest.TestCase):
    def _args(self, *extra: str) -> list[str]:
        return ["--arm", contract.ARM_DENSE, *extra]

    def test_defaults_are_frozen(self) -> None:
        args = runner.build_parser().parse_args(self._args())
        self.assertEqual(args.budget, 96)
        self.assertEqual(args.region_grid, 3)
        self.assertEqual(args.local_candidate_cap, 16)
        self.assertEqual(args.max_swaps, 16)
        self.assertEqual(args.min_pixels, 65536)
        self.assertEqual(args.max_pixels, 589824)
        self.assertEqual(args.dtype, "bf16")

    def test_tampered_parameters_are_rejected(self) -> None:
        for extra in (
            ("--budget", "64"),
            ("--region-grid", "2"),
            ("--local-candidate-cap", "8"),
            ("--max-swaps", "8"),
            ("--min-pixels", "32768"),
            ("--max-pixels", "1048576"),
            ("--dtype", "fp16"),
        ):
            with self.subTest(extra=extra):
                with self.assertRaises(SystemExit):
                    runner.main(self._args(*extra, "--validate-only"))

    def test_arm_is_required_and_restricted(self) -> None:
        with self.assertRaises(SystemExit):
            runner.build_parser().parse_args(["--validate-only"])
        with self.assertRaises(SystemExit):
            runner.build_parser().parse_args(["--arm", "bogus"])

    def test_validate_only_and_dry_run_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            runner.main(self._args("--validate-only", "--dry-run"))

    def test_output_path_must_equal_the_pinned_r2_path(self) -> None:
        pinned = contract.ARM_OUTPUT_PATHS[contract.ARM_V2]
        self.assertEqual(
            contract.ensure_r2_output_path(contract.ARM_V2, None), pinned
        )
        self.assertEqual(
            contract.ensure_r2_output_path(contract.ARM_V2, pinned), pinned
        )
        with self.assertRaises(contract.Edge576ContractError):
            contract.ensure_r2_output_path(
                contract.ARM_V2, Path("/tmp/not-the-r2-path.jsonl")
            )

    def test_real_manifest_pin_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "manifest.jsonl"
            fake.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(contract.Edge576ContractError):
                contract.load_edge576_manifest(fake)

    def test_real_manifest_has_2638_unique_samples_within_budget(self) -> None:
        samples = contract.load_edge576_manifest()
        self.assertEqual(len(samples), 2638)
        self.assertEqual(len({str(s["sample_id"]) for s in samples}), 2638)
        self.assertLessEqual(
            max(int(s["full_llm_visual_tokens"]) for s in samples), 576
        )


class DryRunAndValidateOnlyTest(unittest.TestCase):
    def test_validate_only_reports_no_weights_and_full_input(self) -> None:
        samples = contract.load_edge576_manifest()
        summary = runner.validation_summary(
            contract.ARM_V2,
            samples,
            manifest_path=contract.EDGE576_MANIFEST,
            output=contract.ARM_OUTPUT_PATHS[contract.ARM_V2],
        )
        self.assertEqual(summary["sample_count"], 2638)
        self.assertEqual(summary["unique_sample_id_count"], 2638)
        self.assertTrue(summary["all_g_within_budget"])
        self.assertFalse(summary["weights_loaded"])
        self.assertFalse(summary["model_loaded"])
        self.assertEqual(summary["gpu_runs"], 0)
        self.assertTrue(summary["output_path_matches_r2"])
        self.assertTrue(summary["dependency_pins"]["all_match"])
        self.assertEqual(
            summary["dependency_pins"]["verified_count"],
            len(contract.RUNNER_DEPENDENCY_PINS),
        )
        self.assertFalse(summary["mechanism_identity"]["is_new_mechanism"])

    def test_dry_run_planned_command_names_the_frozen_arm(self) -> None:
        command = runner.planned_command(
            contract.ARM_L2,
            contract.ARM_OUTPUT_PATHS[contract.ARM_L2],
            contract.EDGE576_MANIFEST,
        )
        self.assertIn("--arm corrected_l2_integer_b1_k96", command)
        self.assertIn("run_edge576_cvbench_three_arm.py", command)


class DependencyPinTest(unittest.TestCase):
    def test_all_runner_dependency_pins_match_on_disk(self) -> None:
        result = contract.verify_dependency_pins()
        self.assertTrue(result["all_match"])
        self.assertEqual(result["verified_count"], len(contract.RUNNER_DEPENDENCY_PINS))

    def test_drifted_pin_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "r2.json"
            config.write_text(
                json.dumps(
                    {
                        "version": (
                            "2026-10-07-edge576-cvbench-three-arm-development-r2"
                        ),
                        "pins": {
                            "edge576_manifest": {
                                "path": "README.md",
                                "sha256": "0" * 64,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(contract.Edge576ContractError):
                contract.verify_dependency_pins(
                    pin_names=("edge576_manifest",), config_path=config
                )

    def test_missing_pin_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "r2.json"
            config.write_text(
                json.dumps(
                    {
                        "version": (
                            "2026-10-07-edge576-cvbench-three-arm-development-r2"
                        ),
                        "pins": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(contract.Edge576ContractError):
                contract.verify_dependency_pins(
                    pin_names=("edge576_manifest",), config_path=config
                )


class MechanismIdentityTest(unittest.TestCase):
    def test_v2_is_the_existing_frozen_mechanism(self) -> None:
        identity = contract.mechanism_identity(contract.ARM_V2)
        self.assertEqual(
            identity["mechanism_id"], "conservative_l2_guarded_rerank_v2"
        )
        self.assertEqual(
            identity["frozen_constant_mechanism_id"],
            "conservative_l2_guarded_rerank_v2",
        )
        self.assertFalse(identity["is_new_mechanism"])
        self.assertTrue(identity["ok"])

    def test_l2_identity_is_the_frozen_integer_arm(self) -> None:
        identity = contract.mechanism_identity(contract.ARM_L2)
        self.assertEqual(identity["mechanism_id"], "corrected_l2_integer")
        self.assertEqual(
            identity["frozen_constant_mechanism_id"], "corrected_l2_integer"
        )
        self.assertFalse(identity["is_new_mechanism"])

    def test_unknown_arm_fails_closed(self) -> None:
        with self.assertRaises(contract.Edge576ContractError):
            contract.mechanism_identity("bogus")


class RecordHashCompatibilityTest(unittest.TestCase):
    def test_edge_record_hash_matches_frozen_runner_algorithm(self) -> None:
        from evaluation.run_hf_hardware_regular_b1_corrected_l2 import (  # noqa: E402
            record_sha256 as frozen_record_sha256,
        )

        record = {"b": 2, "a": [1, 2, 3], "nested": {"y": "z", "x": 1}}
        self.assertEqual(
            contract.record_sha256(record), frozen_record_sha256(record)
        )


class ModelIdentityTest(unittest.TestCase):
    """Fail-closed ``--model`` identity verification (temp-dir positive/negative)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.model = self.tmp / "model"
        self.model.mkdir()
        (self.model / "config.json").write_bytes(b'{"model_type": "test"}')
        (self.model / "model.safetensors").write_bytes(b"FAKE-WEIGHTS" * 64)
        self.identity = self.tmp / "identity.json"
        self._write_identity()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _entries(self) -> list[dict]:
        return [
            {
                "relative_path": name,
                "bytes": (self.model / name).stat().st_size,
                "sha256": contract.sha256_file(self.model / name),
                "role": role,
            }
            for name, role in (
                ("config.json", "config"),
                ("model.safetensors", "weight_shard"),
            )
        ]

    def _write_identity(
        self,
        *,
        files: list[dict] | None = None,
        weight_shards: tuple[str, ...] = ("model.safetensors",),
        model_dir: Path | None = None,
    ) -> None:
        entries = self._entries() if files is None else files
        payload = {
            "schema_version": contract.MODEL_IDENTITY_SCHEMA_VERSION,
            "kind": contract.MODEL_IDENTITY_KIND,
            "hash_algorithm": contract.MODEL_IDENTITY_HASH_ALGORITHM,
            "model_dir": str(self.model if model_dir is None else model_dir),
            "model_label": "test-model",
            "file_count": len(entries),
            "weight_shards": list(weight_shards),
            "files": entries,
        }
        self.identity.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _verify(self, **kwargs) -> dict:
        kwargs.setdefault(
            "expected_manifest_sha256", contract.sha256_file(self.identity)
        )
        return contract.verify_model_identity(
            self.model, identity_path=self.identity, **kwargs
        )

    def test_real_pinned_manifest_is_loadable_and_declares_safetensors(self) -> None:
        manifest = contract.load_model_identity_manifest()
        relative_paths = {entry["relative_path"] for entry in manifest["files"]}
        self.assertIn("model.safetensors", relative_paths)
        self.assertEqual(manifest["kind"], contract.MODEL_IDENTITY_KIND)
        self.assertEqual(manifest["model_dir"], str(runner.DEFAULT_MODEL_PATH))

    def test_real_pinned_manifest_wrong_pin_fails_closed(self) -> None:
        with self.assertRaises(contract.Edge576ContractError):
            contract.load_model_identity_manifest(expected_sha256="0" * 64)

    def test_positive_identity_verifies_every_file(self) -> None:
        result = self._verify()
        self.assertTrue(result["ok"])
        self.assertEqual(result["verified_file_count"], 2)
        self.assertEqual(result["model_path"], str(self.model.resolve()))
        self.assertIn("model.safetensors", result["weight_shard_sha256"])
        self.assertEqual(
            result["weight_shard_sha256"]["model.safetensors"],
            contract.sha256_file(self.model / "model.safetensors"),
        )

    def test_missing_file_fails_closed(self) -> None:
        (self.model / "model.safetensors").unlink()
        with self.assertRaises(contract.Edge576ContractError):
            self._verify()

    def test_size_mismatch_fails_closed(self) -> None:
        entries = self._entries()
        entries[0]["bytes"] += 1
        self._write_identity(files=entries)
        with self.assertRaises(contract.Edge576ContractError):
            self._verify()

    def test_hash_mismatch_fails_closed(self) -> None:
        entries = self._entries()
        entries[1]["sha256"] = "0" * 64
        self._write_identity(files=entries)
        with self.assertRaises(contract.Edge576ContractError):
            self._verify()

    def test_required_weight_shard_undeclared_fails_closed(self) -> None:
        self._write_identity(weight_shards=())
        with self.assertRaises(contract.Edge576ContractError):
            self._verify()

    def test_wrong_model_dir_fails_closed(self) -> None:
        other = self.tmp / "elsewhere"
        other.mkdir()
        self._write_identity(model_dir=other)
        with self.assertRaises(contract.Edge576ContractError):
            self._verify()

    def test_manifest_pin_mismatch_fails_closed(self) -> None:
        with self.assertRaises(contract.Edge576ContractError):
            self._verify(expected_manifest_sha256="0" * 64)

    def test_relative_path_escape_fails_closed(self) -> None:
        entries = self._entries()
        entries[0]["relative_path"] = "../config.json"
        self._write_identity(files=entries)
        with self.assertRaises(contract.Edge576ContractError):
            self._verify()


class RunArmPreflightTest(unittest.TestCase):
    """Run-mode preflight order, complete-resume short-circuit and warmup wiring."""

    def _exercise(
        self,
        *,
        pending: int,
        model_path: Path | None = Path("/tmp/model"),
        identity_raises: bool = False,
    ):
        order: list[str] = []
        captured: dict = {}
        samples = [make_sample("s0"), make_sample("s1", (1, 32, 32))]

        def validation_fn(arm, samples, *, manifest_path, output):
            order.append("validation")
            return {
                "pending_count": pending,
                "existing_done_count": len(samples) - pending,
                "dependency_pins": {"all_match": True, "verified_count": 17},
                "mechanism_identity": {"ok": True, "is_new_mechanism": False},
            }

        def model_identity_fn(path):
            order.append("model_identity")
            if identity_raises:
                raise contract.Edge576ContractError("model identity mismatch")
            return {
                "ok": True,
                "identity_path": "identity.json",
                "identity_sha256": "a" * 64,
                "model_path": str(path),
                "model_label": "test-model",
                "file_count": 2,
                "total_bytes": 10,
                "weight_shards": ["model.safetensors"],
                "weight_shard_sha256": {"model.safetensors": "b" * 64},
                "files": [],
            }

        def require_cuda_fn(device):
            order.append("require_cuda")

        def build_state_fn(arm):
            order.append("build_state")
            return {
                "scorer": "SCORER",
                "checkpoint_identity": {"checkpoint_sha256": None},
            }

        def load_model_fn(path, device):
            order.append("load_model")
            return "MODEL"

        def load_processor_fn(path):
            order.append("load_processor")
            return "PROCESSOR"

        def warmup_fn(arm, model, processor, sample, manifest_path, **kwargs):
            order.append("warmup")
            captured["warmup_sample"] = sample
            return {
                "arm": arm,
                "count": 1,
                "sample_id": str(sample["sample_id"]),
                "max_new_tokens": 1,
                "wrote_record": False,
                "output_written": False,
                "elapsed_seconds": 0.25,
                "cuda_synchronized": True,
            }

        def execute_fn(**kwargs):
            order.append("execute")
            captured["run_metadata"] = kwargs["run_metadata"]
            captured["execute_samples"] = list(kwargs["samples"])
            return {
                "written": len(kwargs["samples"]),
                "parse_success": 1,
                "parse_failure": 0,
            }

        error = None
        result = None
        try:
            result = runner.run_arm(
                arm=contract.ARM_L2,
                samples=samples,
                manifest_path=contract.EDGE576_MANIFEST,
                output=Path("/tmp/edge576-out.jsonl"),
                model_path=model_path,
                model_label="test-model",
                device="cuda:0",
                validation_fn=validation_fn,
                model_identity_fn=model_identity_fn,
                require_cuda_fn=require_cuda_fn,
                build_state_fn=build_state_fn,
                load_model_fn=load_model_fn,
                load_processor_fn=load_processor_fn,
                warmup_fn=warmup_fn,
                execute_fn=execute_fn,
            )
        except contract.Edge576ContractError as exc:
            error = exc
        return result, error, order, captured, samples

    def test_preflight_order_puts_validation_before_any_model_load(self) -> None:
        result, error, order, captured, samples = self._exercise(pending=2)
        self.assertIsNone(error)
        self.assertEqual(order[0], "validation")
        for later in (
            "require_cuda",
            "model_identity",
            "build_state",
            "load_model",
            "load_processor",
            "warmup",
            "execute",
        ):
            self.assertLess(order.index("validation"), order.index(later))
        self.assertLess(order.index("model_identity"), order.index("build_state"))
        self.assertLess(order.index("model_identity"), order.index("load_model"))
        self.assertEqual(result["status"], "EDGE576_RUNNER_ARM_COMPLETE")
        self.assertTrue(result["model_identity_verified"])
        self.assertEqual(result["warmup_count"], 1)
        self.assertFalse(captured["run_metadata"]["warmup"]["wrote_record"])
        self.assertTrue(captured["run_metadata"]["model_identity_verified"])
        self.assertEqual(captured["execute_samples"], samples)

    def test_complete_resume_exits_without_model_identity_cuda_or_scorer(self) -> None:
        result, error, order, captured, samples = self._exercise(
            pending=0, model_path=None
        )
        self.assertIsNone(error)
        self.assertEqual(order, ["validation"])
        self.assertEqual(result["status"], "EDGE576_RUNNER_ARM_ALREADY_COMPLETE")
        self.assertEqual(result["pending"], 0)
        self.assertEqual(result["already_done"], 2)
        self.assertFalse(result["weights_loaded"])
        self.assertFalse(result["model_loaded"])
        self.assertIsNone(result["model_identity"])
        self.assertIsNone(result["warmup"])
        self.assertNotIn("run_metadata", captured)

    def test_model_identity_failure_aborts_before_scorer_or_weight_load(self) -> None:
        result, error, order, captured, samples = self._exercise(
            pending=1, identity_raises=True
        )
        self.assertIsInstance(error, contract.Edge576ContractError)
        self.assertIsNone(result)
        self.assertEqual(order, ["validation", "require_cuda", "model_identity"])
        self.assertNotIn("run_metadata", captured)


class WarmupTest(unittest.TestCase):
    """The warmup is one real call, max_new_tokens=1, synchronised and unwritten."""

    def test_warmup_is_one_call_with_max_new_tokens_one_and_synchronizes(self) -> None:
        sample = make_sample("s0")
        captured: dict = {}
        events: list[str] = []

        class FakeCuda:
            @staticmethod
            def synchronize() -> None:
                events.append("synchronize")

        fake_torch = types.SimpleNamespace(cuda=FakeCuda)

        def fake_run_one_for_arm(
            arm, model, processor, warmup_sample, manifest_path, **kwargs
        ):
            events.append("run_one")
            captured["warmup_sample"] = dict(warmup_sample)
            captured["kwargs"] = kwargs
            return fake_frozen(warmup_sample, arm)

        with mock.patch.object(runner, "run_one_for_arm", fake_run_one_for_arm):
            result = runner.run_multimodal_warmup(
                contract.ARM_V2,
                "MODEL",
                "PROCESSOR",
                sample,
                Path("manifest.jsonl"),
                manifest_sha256=MANIFEST_SHA,
                scorer="SCORER",
                checkpoint_identity={"checkpoint_sha256": None},
                run_mode="full2638",
                torch_module=fake_torch,
            )
        self.assertEqual(events.count("run_one"), 1)
        self.assertEqual(events.count("synchronize"), 1)
        self.assertEqual(captured["warmup_sample"]["max_new_tokens"], 1)
        self.assertEqual(sample["max_new_tokens"], 16)
        self.assertEqual(captured["kwargs"]["scorer"], "SCORER")
        self.assertEqual(captured["kwargs"]["run_mode"], "full2638")
        self.assertFalse(result["wrote_record"])
        self.assertFalse(result["output_written"])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["max_new_tokens"], 1)
        self.assertEqual(result["image_grid_thw"], [1, 16, 16])

    def test_warmup_writes_no_output_file(self) -> None:
        sample = make_sample("s0")
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(synchronize=lambda: None)
        )
        with tempfile.TemporaryDirectory() as tmp:

            def fake_run_one_for_arm(
                arm, model, processor, warmup_sample, manifest_path, **kwargs
            ):
                return fake_frozen(warmup_sample, arm)

            with mock.patch.object(runner, "run_one_for_arm", fake_run_one_for_arm):
                runner.run_multimodal_warmup(
                    contract.ARM_DENSE,
                    None,
                    None,
                    sample,
                    Path("manifest.jsonl"),
                    manifest_sha256=MANIFEST_SHA,
                    scorer=None,
                    checkpoint_identity={},
                    run_mode="full2638",
                    torch_module=fake_torch,
                )
            self.assertEqual(os.listdir(tmp), [])

    def test_warmup_grid_mismatch_fails_closed(self) -> None:
        sample = make_sample("s0")
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(synchronize=lambda: None)
        )

        def fake_run_one_for_arm(
            arm, model, processor, warmup_sample, manifest_path, **kwargs
        ):
            return fake_frozen(warmup_sample, arm, grid=(1, 16, 32))

        with mock.patch.object(runner, "run_one_for_arm", fake_run_one_for_arm):
            with self.assertRaises(contract.Edge576ContractError):
                runner.run_multimodal_warmup(
                    contract.ARM_V2,
                    None,
                    None,
                    sample,
                    Path("manifest.jsonl"),
                    manifest_sha256=MANIFEST_SHA,
                    scorer=None,
                    checkpoint_identity={},
                    run_mode="full2638",
                    torch_module=fake_torch,
                )


if __name__ == "__main__":
    unittest.main()
