#!/usr/bin/env python3
"""CPU-only contract tests for the layer-5 forced-index / control runner.

The runner is only executed on a GPU host; these tests pin its frozen CPU-side
contract: the CLI surface (per-arm ``--arm``, a fixed historical
``--reference-jsonl``, and **no** fixed-budget / projection / local-cap knob),
the fail-closed historical reference loading (duplicate / missing / extra /
task / index / range / hash), per-sample ``K_C``, the pure arm -> forward
parameter mapping, the diagnostic metadata and the "only the new vision forward,
old module reused only for LLM input assembly" rule, plus the fail-closed
resume validation of an existing ``--output`` JSONL (arm / input hashes /
diagnostic flags / unique in-range ids).  No CUDA is required or touched.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from evaluation import (  # noqa: E402
    run_hf_layer5_forced_index_ablation as runner,
)
from evaluation.hardware_regular_token_selector import (  # noqa: E402
    SelectorContractError,
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
    "--arm",
    "control",
]


def _manifest_row(sample_id: str, task: str) -> dict:
    return {
        "sample_id": sample_id,
        "task": task,
        "image": "img.png",
        "prompt": "prompt",
        "max_new_tokens": 16,
    }


def _ref_row(
    sample_id: str,
    task: str,
    group_count: int,
    indices,
    *,
    sha=None,
    prune_after_layer: int = 5,
    llm_visual_tokens=None,
    baseline=None,
) -> dict:
    ordered = sorted(int(index) for index in indices)
    if sha is None:
        sha = runner.indices_sha256(ordered)
    return {
        "sample_id": sample_id,
        "task": task,
        "prune_after_layer": prune_after_layer,
        "baseline_llm_visual_tokens": group_count if baseline is None else baseline,
        "llm_visual_tokens": (
            len(ordered) if llm_visual_tokens is None else llm_visual_tokens
        ),
        "selected_group_indices": [int(index) for index in indices],
        "selected_group_indices_sha256": sha,
    }


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _validated_reference(sample_id: str, task: str, group_count: int, indices):
    row = _ref_row(sample_id, task, group_count, indices)
    return runner.validate_reference_record(row, _manifest_row(sample_id, task))


def _output_record(
    sample_id: str,
    *,
    arm: str = "control",
    manifest_sha256: str = "manifest-sha",
    reference_sha256: str = "reference-sha",
    **overrides,
) -> dict:
    """A minimal output line consistent with the current run (resume contract)."""
    record = {
        "sample_id": sample_id,
        "arm": arm,
        "manifest_sha256": manifest_sha256,
        "reference_jsonl_sha256": reference_sha256,
        **runner.diagnostic_flags(),
    }
    record.update(overrides)
    return record


class CliContractTest(unittest.TestCase):
    def test_defaults(self) -> None:
        args = runner.build_parser().parse_args(REQUIRED_ARGS)
        self.assertEqual(args.arm, "control")
        self.assertEqual(args.region_grid, 3)
        self.assertEqual(args.dtype, "bf16")
        self.assertEqual(args.device, "cuda:0")
        self.assertEqual(Path(args.reference_jsonl), runner.REFERENCE_JSONL)
        self.assertFalse(hasattr(args, "budget"))
        self.assertFalse(hasattr(args, "score_mode"))
        self.assertFalse(hasattr(args, "prune_after_layer"))

    def test_arm_choices(self) -> None:
        for arm in ("control", "forced"):
            with self.subTest(arm=arm):
                args = runner.build_parser().parse_args(
                    REQUIRED_ARGS[:-1] + [arm]
                )
                self.assertEqual(args.arm, arm)
        with self.assertRaises(SystemExit):
            runner.build_parser().parse_args(REQUIRED_ARGS[:-1] + ["bogus"])

    def test_arm_is_required(self) -> None:
        without_arm = REQUIRED_ARGS[: REQUIRED_ARGS.index("--arm")]
        with self.assertRaises(SystemExit):
            runner.build_parser().parse_args(without_arm)

    def test_fixed_budget_knob_does_not_exist(self) -> None:
        for flag in ("--budget", "--local-candidate-cap", "--projection-npz"):
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit):
                    runner.build_parser().parse_args(REQUIRED_ARGS + [flag, "96"])

    def test_reference_jsonl_can_be_overridden(self) -> None:
        args = runner.build_parser().parse_args(
            REQUIRED_ARGS + ["--reference-jsonl", "custom.jsonl"]
        )
        self.assertEqual(Path(args.reference_jsonl), Path("custom.jsonl"))

    def test_limit_per_task_is_supported(self) -> None:
        args = runner.build_parser().parse_args(
            REQUIRED_ARGS + ["--limit-per-task", "5"]
        )
        self.assertEqual(args.limit_per_task, 5)


class ConstantsTest(unittest.TestCase):
    def test_frozen_constants(self) -> None:
        self.assertEqual(runner.ARM_CONTROL, "control")
        self.assertEqual(runner.ARM_FORCED, "forced")
        self.assertEqual(runner.ALLOWED_ARMS, ("control", "forced"))
        self.assertEqual(runner.CUTPOINT_PREFIX_BLOCKS, (0, 1, 2, 3, 4, 5))
        self.assertEqual(runner.PRUNE_AFTER_LAYER, 5)
        self.assertTrue(runner.REFERENCE_JSONL.name.endswith("shallow_r0250.jsonl"))
        self.assertEqual(runner.REFERENCE_SELECTOR, "shallow_r0250")

    def test_diagnostic_metadata(self) -> None:
        flags = runner.diagnostic_flags()
        self.assertIs(flags["hardware_alignment_relaxed_for_diagnostic"], True)
        self.assertIs(flags["causal_diagnostic_only"], True)
        self.assertIs(flags["fpga_deployable_candidate"], False)
        self.assertEqual(flags["only_variable"], "selected_group_indices")
        self.assertIn("selected_group_indices", flags["single_variable_contract"])
        self.assertIn("selected_group_indices", runner.SINGLE_VARIABLE_CONTRACT)

    def test_arm_selection_rule(self) -> None:
        self.assertNotEqual(
            runner.arm_selection_rule("control"),
            runner.arm_selection_rule("forced"),
        )
        with self.assertRaises(ValueError):
            runner.arm_selection_rule("bogus")


class ReferenceLoadingTest(unittest.TestCase):
    def test_valid_reference_is_mapped_by_sample_id(self) -> None:
        manifest = [_manifest_row("a", "Count"), _manifest_row("b", "Depth")]
        rows = [
            _ref_row("a", "Count", 8, [1, 3, 5]),
            _ref_row("b", "Depth", 10, [0, 4]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.jsonl"
            _write_jsonl(path, rows)
            mapping = runner.load_reference_records(path, manifest)
        self.assertEqual(set(mapping), {"a", "b"})
        self.assertEqual(mapping["a"]["K_C"], 3)
        self.assertEqual(mapping["a"]["G"], 8)
        self.assertEqual(mapping["a"]["group_indices"], [1, 3, 5])
        self.assertEqual(
            mapping["a"]["group_indices_sha256"], runner.indices_sha256([1, 3, 5])
        )

    def test_unsorted_indices_with_normalized_hash_are_accepted(self) -> None:
        manifest = [_manifest_row("a", "Count")]
        row = _ref_row("a", "Count", 8, [5, 1, 3], sha=runner.indices_sha256([1, 3, 5]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.jsonl"
            _write_jsonl(path, [row])
            mapping = runner.load_reference_records(path, manifest)
        self.assertEqual(mapping["a"]["group_indices"], [1, 3, 5])

    def test_duplicate_reference_record_fails(self) -> None:
        manifest = [_manifest_row("a", "Count")]
        rows = [_ref_row("a", "Count", 8, [1]), _ref_row("a", "Count", 8, [2])]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.jsonl"
            _write_jsonl(path, rows)
            with self.assertRaises(runner.ReferenceContractError):
                runner.load_reference_records(path, manifest)

    def test_missing_reference_record_fails(self) -> None:
        manifest = [_manifest_row("a", "Count"), _manifest_row("b", "Depth")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.jsonl"
            _write_jsonl(path, [_ref_row("a", "Count", 8, [1])])
            with self.assertRaises(runner.ReferenceContractError):
                runner.load_reference_records(path, manifest)

    def test_extra_reference_record_fails(self) -> None:
        manifest = [_manifest_row("a", "Count")]
        rows = [_ref_row("a", "Count", 8, [1]), _ref_row("zzz", "Depth", 8, [2])]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.jsonl"
            _write_jsonl(path, rows)
            with self.assertRaises(runner.ReferenceContractError):
                runner.load_reference_records(path, manifest)

    def test_task_mismatch_fails(self) -> None:
        manifest = [_manifest_row("a", "Count")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.jsonl"
            _write_jsonl(path, [_ref_row("a", "Depth", 8, [1])])
            with self.assertRaises(runner.ReferenceContractError):
                runner.load_reference_records(path, manifest)

    def test_missing_reference_file_fails(self) -> None:
        manifest = [_manifest_row("a", "Count")]
        with self.assertRaises(runner.ReferenceContractError):
            runner.load_reference_records(Path("/nonexistent/ref.jsonl"), manifest)

    def test_illegal_indices_fail(self) -> None:
        manifest = [_manifest_row("a", "Count")]
        base = _ref_row("a", "Count", 8, [1, 3])
        cases = {
            "not_a_list": "1,3",
            "empty": [],
            "nested": [[1], [3]],
            "float": [1.0, 3.0],
            "bool": [True, False],
            "duplicates": [1, 1],
            "negative": [-1, 3],
            "out_of_range": [1, 8],
            "sha_mismatch": [1, 3],
        }
        for name, bad in cases.items():
            with self.subTest(case=name):
                row = dict(base)
                row["selected_group_indices"] = bad
                if name == "sha_mismatch":
                    row["selected_group_indices_sha256"] = "0" * 64
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "ref.jsonl"
                    _write_jsonl(path, [row])
                    with self.assertRaises(runner.ReferenceContractError):
                        runner.load_reference_records(path, manifest)

    def test_non_layer5_and_inconsistent_k_fail(self) -> None:
        manifest = [_manifest_row("a", "Count")]
        wrong_cut = _ref_row("a", "Count", 8, [1], prune_after_layer=0)
        wrong_k = _ref_row("a", "Count", 8, [1], llm_visual_tokens=9)
        zero_g = _ref_row("a", "Count", 8, [1], baseline=0)
        for name, row in (
            ("wrong_cut", wrong_cut),
            ("wrong_k", wrong_k),
            ("zero_g", zero_g),
        ):
            with self.subTest(case=name):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "ref.jsonl"
                    _write_jsonl(path, [row])
                    with self.assertRaises(runner.ReferenceContractError):
                        runner.load_reference_records(path, manifest)

    def test_per_sample_k_c_equals_historical_length(self) -> None:
        manifest = [_manifest_row("a", "Count"), _manifest_row("b", "Depth")]
        rows = [
            _ref_row("a", "Count", 100, list(range(0, 100, 4))),
            _ref_row("b", "Depth", 100, list(range(0, 100, 5))),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.jsonl"
            _write_jsonl(path, rows)
            mapping = runner.load_reference_records(path, manifest)
        self.assertEqual(runner.reference_K(mapping["a"]), 25)
        self.assertEqual(runner.reference_K(mapping["b"]), 20)

    def test_reference_is_loaded_for_frozen_shallow_r0250(self) -> None:
        manifest_rows = [
            json.loads(line)
            for line in (
                ROOT / "datasets" / "cv_bench_screening100.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        mapping = runner.load_reference_records(
            runner.REFERENCE_JSONL, manifest_rows
        )
        self.assertEqual(len(mapping), len(manifest_rows))
        for record in mapping.values():
            self.assertEqual(record["K_C"], len(record["group_indices"]))
            self.assertTrue(record["G"] > 0)
            self.assertTrue(record["K_C"] > 0)


class ResumeValidationTest(unittest.TestCase):
    """`load_resume_completed_ids` must fail closed on a suspect output file."""

    MANIFEST_SHA = "manifest-sha"
    REFERENCE_SHA = "reference-sha"
    SAMPLES = [
        _manifest_row("a", "Count"),
        _manifest_row("b", "Depth"),
        _manifest_row("c", "Count"),
    ]

    def _call(self, path, *, arm="control", manifest_sha256=None, reference_sha256=None):
        return runner.load_resume_completed_ids(
            path,
            self.SAMPLES,
            arm=arm,
            manifest_sha256=(
                self.MANIFEST_SHA if manifest_sha256 is None else manifest_sha256
            ),
            reference_jsonl_sha256=(
                self.REFERENCE_SHA if reference_sha256 is None else reference_sha256
            ),
        )

    def test_missing_and_empty_files_return_empty_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.jsonl"
            self.assertEqual(self._call(missing), set())
            empty = Path(tmp) / "empty.jsonl"
            empty.write_text("", encoding="utf-8")
            self.assertEqual(self._call(empty), set())
            blank = Path(tmp) / "blank.jsonl"
            blank.write_text("\n   \n", encoding="utf-8")
            self.assertEqual(self._call(blank), set())

    def test_legal_resume_returns_completed_ids(self) -> None:
        rows = [_output_record("a"), _output_record("c")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            _write_jsonl(path, rows)
            self.assertEqual(self._call(path), {"a", "c"})

    def test_mixed_arms_fail(self) -> None:
        rows = [_output_record("a", arm="control"), _output_record("b", arm="forced")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            _write_jsonl(path, rows)
            with self.assertRaises(runner.ResumeContractError):
                self._call(path, arm="control")

    def test_manifest_hash_mismatch_fails(self) -> None:
        rows = [_output_record("a", manifest_sha256="stale-manifest")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            _write_jsonl(path, rows)
            with self.assertRaises(runner.ResumeContractError):
                self._call(path)

    def test_reference_hash_mismatch_fails(self) -> None:
        rows = [_output_record("a", reference_sha256="stale-reference")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            _write_jsonl(path, rows)
            with self.assertRaises(runner.ResumeContractError):
                self._call(path)

    def test_duplicate_sample_id_fails(self) -> None:
        rows = [_output_record("a"), _output_record("a")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            _write_jsonl(path, rows)
            with self.assertRaises(runner.ResumeContractError):
                self._call(path)

    def test_out_of_range_sample_id_fails(self) -> None:
        rows = [_output_record("zzz")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            _write_jsonl(path, rows)
            with self.assertRaises(runner.ResumeContractError):
                self._call(path)

    def test_bad_json_line_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            path.write_text("{not json\n", encoding="utf-8")
            with self.assertRaises(runner.ResumeContractError):
                self._call(path)

    def test_non_object_line_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            path.write_text("[1, 2, 3]\n", encoding="utf-8")
            with self.assertRaises(runner.ResumeContractError):
                self._call(path)

    def test_missing_sample_id_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            record = _output_record("a")
            del record["sample_id"]
            _write_jsonl(path, [record])
            with self.assertRaises(runner.ResumeContractError):
                self._call(path)

    def test_diagnostic_flag_mismatch_fails(self) -> None:
        cases = (
            "hardware_alignment_relaxed_for_diagnostic",
            "causal_diagnostic_only",
            "fpga_deployable_candidate",
        )
        for field in cases:
            with self.subTest(field=field):
                record = _output_record("a")
                record[field] = not runner.diagnostic_flags()[field]
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "out.jsonl"
                    _write_jsonl(path, [record])
                    with self.assertRaises(runner.ResumeContractError):
                        self._call(path)


class ArmMappingTest(unittest.TestCase):
    def test_control_passes_only_k_c(self) -> None:
        kwargs = runner.arm_forward_kwargs(
            "control", budget=7, reference_indices=[1, 3, 9]
        )
        self.assertEqual(kwargs["mode"], "control")
        self.assertEqual(kwargs["budget"], 7)
        self.assertIsNone(kwargs["forced_group_indices"])

    def test_forced_passes_the_historical_indices(self) -> None:
        kwargs = runner.arm_forward_kwargs(
            "forced", budget=3, reference_indices=[1, 3, 9]
        )
        self.assertEqual(kwargs["mode"], "forced")
        self.assertEqual(kwargs["budget"], 3)
        self.assertEqual(kwargs["forced_group_indices"], [1, 3, 9])

    def test_bad_arm_and_budget_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            runner.arm_forward_kwargs(
                "bogus", budget=1, reference_indices=[0]
            )
        with self.assertRaises(SelectorContractError):
            runner.arm_forward_kwargs(
                "control", budget=0, reference_indices=[]
            )


class RunOnePreconditionTest(unittest.TestCase):
    """`run_one` must reject broken contracts before any CUDA access."""

    def _call(self, sample, reference, arm="control"):
        return runner.run_one(
            None,
            None,
            sample,
            Path("manifest.jsonl"),
            reference,
            arm=arm,
            region_grid=3,
            manifest_sha256="m",
            reference_sha256="r",
        )

    def test_bad_arm_is_rejected(self) -> None:
        sample = _manifest_row("a", "Count")
        reference = _validated_reference("a", "Count", 8, [1, 3])
        with self.assertRaises(ValueError):
            self._call(sample, reference, arm="bogus")

    def test_sample_id_mismatch_is_rejected(self) -> None:
        sample = _manifest_row("a", "Count")
        reference = _validated_reference("b", "Count", 8, [1, 3])
        with self.assertRaises(ValueError):
            self._call(sample, reference)

    def test_task_mismatch_is_rejected(self) -> None:
        sample = _manifest_row("a", "Count")
        reference = _validated_reference("a", "Count", 8, [1, 3])
        reference = dict(reference, task="Depth")
        with self.assertRaises(ValueError):
            self._call(sample, reference)

    def test_inconsistent_k_c_is_rejected(self) -> None:
        sample = _manifest_row("a", "Count")
        reference = _validated_reference("a", "Count", 8, [1, 3])
        reference = dict(reference, K_C=5)
        with self.assertRaises(ValueError):
            self._call(sample, reference)


class SourceContractTest(unittest.TestCase):
    def test_only_the_new_vision_forward_and_shared_llm_assembly(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        for token in (
            "forward_hardware_regular_layer5_forced_index_ablation_vision",
            "prepare_layer5_unbounded_multimodal_inputs",
            "verify_diagnostic_selection",
            "single_variable_contract",
            "hardware_alignment_relaxed_for_diagnostic",
            "greedy_argmax",
            "prune_after_layer",
            "reference_selected_group_indices",
            "reference_jsonl_sha256",
        ):
            self.assertIn(token, source)
        # The old visual forward must never be called or imported.
        self.assertNotIn(
            "forward_hardware_regular_layer5_unbounded_l2_vision", source
        )

    def test_file_sha256_matches_hashlib(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.bin"
            path.write_bytes(b"layer5-forced-index")
            self.assertEqual(
                runner.file_sha256(path),
                hashlib.sha256(b"layer5-forced-index").hexdigest(),
            )

    def test_import_does_not_initialize_cuda(self) -> None:
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
