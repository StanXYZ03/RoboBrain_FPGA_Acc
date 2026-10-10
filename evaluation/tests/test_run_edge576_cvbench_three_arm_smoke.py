#!/usr/bin/env python3
"""CPU-only focused tests for the single-sample edge576 smoke entry.

Covers the required contract without CUDA, model weights or network:

* three-arm dispatch to the shared :func:`run_one_for_arm`;
* smoke output-path isolation from every full2638 arm output;
* exactly one sample per invocation;
* the real up-front validation (17 dependency pins + mechanism identity) for
  the dense arm, which needs no torch;
* the fail-closed preflight order (validation -> model identity -> CUDA ->
  scorer/weights -> warmup -> one frozen sample);
* the real multimodal warmup being called once, ``max_new_tokens=1``, and never
  written as a record;
* the actual ``image_grid_thw`` assertion failing closed without writing;
* the absence of any copied selection algorithm.

The heavy frozen implementations (scorer, model, processor call, warmup) are
replaced by fakes, while the real 2638-row edge576 manifest, the model identity
manifest and the SHA256 pins are verified for real because that needs no model
weights.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import edge576_cvbench_runner_contract as contract  # noqa: E402
from evaluation import run_edge576_cvbench_three_arm as runner  # noqa: E402
from evaluation import run_edge576_cvbench_three_arm_smoke as smoke  # noqa: E402

MANIFEST_SHA = contract.EDGE576_MANIFEST_SHA256

#: Sentinel meaning "the frozen runner produced no image_grid_thw at all".
MISSING_GRID = "MISSING_GRID"


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
    include_token_ids: bool = True,
    output_tokens: int | None = None,
) -> dict:
    if grid is None:
        actual_grid: list[int] | None = list(sample["image_grid_thw"])
    elif grid == MISSING_GRID:
        actual_grid = None
    else:
        actual_grid = list(grid)
    full_llm = int(sample["full_llm_visual_tokens"])
    selected = list(range(full_llm))
    record: dict = {
        "image_grid_thw": actual_grid,
        "output_text": output_text,
        "selected_group_indices": selected,
    }
    if include_token_ids:
        record["output_token_ids"] = [1, 2, 3]
    if output_tokens is not None:
        record["output_tokens"] = int(output_tokens)
    if arm == contract.ARM_DENSE:
        record["actual_token_retention"] = 1.0
        record["llm_visual_tokens"] = full_llm
    if arm in (contract.ARM_L2, contract.ARM_V2):
        record["requested_budget"] = 96
        record["region_grid"] = 3
        record["local_candidate_cap"] = 16
        record["identity_bypass"] = True
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


def fake_validation(arm, samples, *, manifest_path, output):
    return {
        "arm": arm,
        "mechanism_id": contract.ARM_MECHANISM_IDS[arm],
        "dependency_pins": {
            "all_match": True,
            "verified_count": 17,
            "requested_count": 17,
            "pins": {},
        },
        "mechanism_identity": {
            "arm": arm,
            "mechanism_id": contract.ARM_MECHANISM_IDS[arm],
            "frozen_constant_mechanism_id": contract.ARM_MECHANISM_IDS[arm],
            "is_new_mechanism": False,
            "ok": True,
        },
    }


def fake_metadata(arm, **kwargs):
    warmup = kwargs.get("warmup")
    return {
        "arm": arm,
        "warmup_count": 1 if warmup else 0,
        "warmup": warmup,
        "model_identity_verified": kwargs.get("model_identity") is not None,
    }


class SmokeHarness:
    """Records the call order and lets tests mutate the frozen fakes."""

    def __init__(self, tmp: Path, *, arm=contract.ARM_DENSE, grid=None,
                 output_text="The answer is (B).", processor=None,
                 include_token_ids: bool = True, output_tokens: int | None = None):
        self.tmp = tmp
        self.arm = arm
        self.grid = grid
        self.output_text = output_text
        self.include_token_ids = include_token_ids
        self.output_tokens = output_tokens
        self.order: list[str] = []
        self.captured: dict = {}
        self.processor = (
            contract.Edge576Processor(object())
            if processor is None
            else processor
        )

    def validation(self, arm, samples, *, manifest_path, output):
        self.order.append("validation")
        return fake_validation(arm, samples, manifest_path=manifest_path, output=output)

    def model_identity(self, path):
        self.order.append("model_identity")
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

    def require_cuda(self, device):
        self.order.append("require_cuda")

    def build_state(self, arm):
        self.order.append("build_state")
        return {
            "scorer": "SCORER",
            "checkpoint_identity": {"checkpoint_sha256": None},
        }

    def load_model(self, path, device):
        self.order.append("load_model")
        return "MODEL"

    def load_processor(self, path):
        self.order.append("load_processor")
        return self.processor

    def warmup(self, arm, model, processor, sample, manifest_path, **kwargs):
        self.order.append("warmup")
        warmup_sample = dict(sample)
        warmup_sample["max_new_tokens"] = int(kwargs.get("max_new_tokens", 1))
        self.captured["warmup_sample"] = warmup_sample
        self.captured["warmup_kwargs"] = kwargs
        return {
            "arm": arm,
            "count": 1,
            "sample_id": str(sample["sample_id"]),
            "max_new_tokens": int(kwargs.get("max_new_tokens", 1)),
            "wrote_record": False,
            "output_written": False,
            "elapsed_seconds": 0.1,
            "image_grid_thw": list(sample["image_grid_thw"]),
            "cuda_synchronized": True,
        }

    def run_one(self, arm, model, processor, sample, manifest_path, **kwargs):
        self.order.append("run_one")
        self.captured["run_one_sample"] = dict(sample)
        self.captured["run_one_kwargs"] = kwargs
        return fake_frozen(
            sample,
            arm,
            grid=self.grid,
            output_text=self.output_text,
            include_token_ids=self.include_token_ids,
            output_tokens=self.output_tokens,
        )

    def kwargs(self, *, smoke_dir: Path, output: Path | None = None):
        return dict(
            arm=self.arm,
            sample_id=self.captured.get("requested_id", "s0"),
            samples=self.captured.get("samples"),
            manifest_path=contract.EDGE576_MANIFEST,
            output=output,
            model_path=Path("/tmp/model"),
            model_label="test-model",
            device="cuda:0",
            smoke_dir=smoke_dir,
            validation_fn=self.validation,
            model_identity_fn=self.model_identity,
            require_cuda_fn=self.require_cuda,
            build_state_fn=self.build_state,
            load_model_fn=self.load_model,
            load_processor_fn=self.load_processor,
            warmup_fn=self.warmup,
            run_one_fn=self.run_one,
            metadata_fn=fake_metadata,
        )


def run_smoke_with(
    tmp: Path,
    *,
    arm: str,
    sample_id: str,
    samples: list[dict],
    grid=None,
    output_text="The answer is (B).",
    processor=None,
    output: Path | None = None,
    model_path: Path | None = Path("/tmp/model"),
    include_token_ids: bool = True,
    output_tokens: int | None = None,
):
    harness = SmokeHarness(
        tmp, arm=arm, grid=grid, output_text=output_text, processor=processor,
        include_token_ids=include_token_ids, output_tokens=output_tokens,
    )
    harness.captured["requested_id"] = sample_id
    harness.captured["samples"] = samples
    kwargs = harness.kwargs(smoke_dir=tmp, output=output)
    kwargs["model_path"] = model_path
    payload = smoke.run_smoke(**kwargs)
    return payload, harness


class SmokePathIsolationTest(unittest.TestCase):
    def test_default_smoke_path_is_pinned_and_named(self) -> None:
        for arm in contract.ARMS:
            path = smoke.smoke_output_path(arm, "1838")
            self.assertEqual(
                path,
                smoke.SMOKE_DIR / f"{arm}_1838.json",
            )
        self.assertTrue(str(smoke.SMOKE_DIR).endswith("2026-10-07-edge576-smoke"))

    def test_smoke_output_must_equal_pinned_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pinned = smoke.smoke_output_path(
                contract.ARM_DENSE, "s0", smoke_dir=tmp_path
            )
            self.assertEqual(
                smoke.ensure_smoke_output_path(
                    contract.ARM_DENSE, "s0", pinned, smoke_dir=tmp_path
                ),
                pinned,
            )
            with self.assertRaises(contract.Edge576ContractError):
                smoke.ensure_smoke_output_path(
                    contract.ARM_DENSE,
                    "s0",
                    tmp_path / "other.json",
                    smoke_dir=tmp_path,
                )

    def test_smoke_output_rejects_full2638_destination(self) -> None:
        with self.assertRaises(contract.Edge576ContractError):
            smoke.ensure_smoke_output_path(
                contract.ARM_V2,
                "1838",
                contract.ARM_OUTPUT_PATHS[contract.ARM_V2],
            )

    def test_smoke_dir_disjoint_from_full2638_dir(self) -> None:
        smoke_root = smoke.SMOKE_DIR.resolve()
        for full in contract.ARM_OUTPUT_PATHS.values():
            resolved = full.resolve()
            self.assertNotEqual(resolved.parent, smoke_root)
            with self.assertRaises(ValueError):
                resolved.relative_to(smoke_root)

    def test_sample_id_path_escape_rejected(self) -> None:
        for bad in ("../x", "a/b", "", ".", "..", "a\\b"):
            with self.subTest(bad=bad):
                with self.assertRaises(contract.Edge576ContractError):
                    smoke.smoke_output_path(contract.ARM_DENSE, bad)

    def test_default_smoke_sample_is_in_the_manifest(self) -> None:
        samples = contract.load_edge576_manifest()
        sample = smoke.find_smoke_sample(samples, smoke.DEFAULT_SMOKE_SAMPLE_ID)
        self.assertEqual(sample["sample_id"], smoke.DEFAULT_SMOKE_SAMPLE_ID)
        self.assertEqual(sample["full_llm_visual_tokens"], 576)
        self.assertEqual(sample["image_grid_thw"], [1, 36, 64])


class ThreeArmDispatchTest(unittest.TestCase):
    def test_run_smoke_dispatches_each_arm_to_the_frozen_run_one(self) -> None:
        samples = [make_sample("s0")]
        for arm in contract.ARMS:
            with self.subTest(arm=arm):
                with tempfile.TemporaryDirectory() as tmp:
                    payload, harness = run_smoke_with(
                        Path(tmp), arm=arm, sample_id="s0", samples=samples
                    )
                    self.assertEqual(harness.order[-1], "run_one")
                    self.assertEqual(
                        harness.captured["run_one_kwargs"]["run_mode"], "smoke"
                    )
                    self.assertEqual(
                        payload["record"]["mechanism_id"],
                        contract.ARM_MECHANISM_IDS[arm],
                    )
                    self.assertEqual(payload["record"]["arm"], arm)
                    self.assertTrue(payload["smoke_summary"]["mechanism_identity"]["ok"])

    def test_run_one_is_called_once_per_invocation(self) -> None:
        samples = [make_sample("s0")]
        with tempfile.TemporaryDirectory() as tmp:
            _, harness = run_smoke_with(
                Path(tmp), arm=contract.ARM_L2, sample_id="s0", samples=samples
            )
        self.assertEqual(harness.order.count("run_one"), 1)

    def test_source_does_not_copy_any_selection_algorithm(self) -> None:
        source = Path(smoke.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "combined_scores",
            "legal_gain",
            "rebuild_hardware_selection",
            "select_hardware_regular_groups",
            "l2_combined_float",
            "torch.matmul",
            "Wv_int8",
        ):
            self.assertNotIn(forbidden, source)
        for required in (
            "from evaluation.edge576_cvbench_runner_contract import",
            "from evaluation.run_edge576_cvbench_three_arm import",
            "run_one_for_arm",
            "normalize_record",
            "build_arm_state",
            "run_multimodal_warmup",
            "Edge576Processor",
            "verify_model_identity",
            "load_edge576_manifest",
        ):
            self.assertIn(required, source)

    def test_smoke_reuses_runner_helpers_by_identity(self) -> None:
        self.assertIs(smoke.run_one_for_arm, runner.run_one_for_arm)
        self.assertIs(smoke.normalize_record, runner.normalize_record)
        self.assertIs(smoke.build_arm_state, runner.build_arm_state)
        self.assertIs(smoke.run_multimodal_warmup, runner.run_multimodal_warmup)
        self.assertIs(smoke.load_edge576_manifest, contract.load_edge576_manifest)
        self.assertIs(smoke.verify_model_identity, contract.verify_model_identity)
        self.assertIs(smoke.Edge576Processor, contract.Edge576Processor)


class SingleSampleTest(unittest.TestCase):
    def test_only_the_requested_sample_is_run(self) -> None:
        samples = [make_sample("s0"), make_sample("s1", (1, 32, 32))]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            payload, harness = run_smoke_with(
                tmp_path, arm=contract.ARM_DENSE, sample_id="s1", samples=samples
            )
            self.assertEqual(
                harness.captured["run_one_sample"]["sample_id"], "s1"
            )
            self.assertEqual(harness.captured["warmup_sample"]["sample_id"], "s1")
            self.assertEqual(payload["record"]["sample_id"], "s1")
            written = tmp_path / f"{contract.ARM_DENSE}_s1.json"
            self.assertTrue(written.is_file())
            self.assertEqual(len(list(tmp_path.glob("*.json"))), 1)

    def test_output_file_contains_exactly_one_record(self) -> None:
        samples = [make_sample("s0"), make_sample("s1", (1, 32, 32))]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            smoke.run_smoke(
                arm=contract.ARM_V2,
                sample_id="s0",
                samples=samples,
                manifest_path=contract.EDGE576_MANIFEST,
                output=None,
                model_path=Path("/tmp/model"),
                model_label="test-model",
                device="cuda:0",
                smoke_dir=tmp_path,
                validation_fn=lambda arm, s, *, manifest_path, output: fake_validation(
                    arm, s, manifest_path=manifest_path, output=output
                ),
                model_identity_fn=lambda path: {"ok": True, "model_path": str(path)},
                require_cuda_fn=lambda device: None,
                build_state_fn=lambda arm: {
                    "scorer": object(),
                    "checkpoint_identity": {},
                },
                load_model_fn=lambda path, device: object(),
                load_processor_fn=lambda path: contract.Edge576Processor(object()),
                warmup_fn=lambda arm, m, p, s, mp, **kw: {
                    "arm": arm,
                    "count": 1,
                    "sample_id": str(s["sample_id"]),
                    "max_new_tokens": 1,
                    "wrote_record": False,
                },
                run_one_fn=lambda arm, m, p, s, mp, **kw: fake_frozen(s, arm),
                metadata_fn=fake_metadata,
            )
            payload = json.loads(
                (tmp_path / f"{contract.ARM_V2}_s0.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["sample_id"], "s0")
            self.assertEqual(payload["record"]["sample_id"], "s0")
            self.assertIn("record", payload)
            self.assertIn("smoke_summary", payload)
            self.assertEqual(len(list(tmp_path.glob("*.json"))), 1)

    def test_unknown_sample_id_fails_closed(self) -> None:
        samples = [make_sample("s0")]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(contract.Edge576ContractError):
                run_smoke_with(
                    Path(tmp),
                    arm=contract.ARM_DENSE,
                    sample_id="ghost",
                    samples=samples,
                )

    def test_duplicate_sample_id_fails_closed(self) -> None:
        samples = [make_sample("s0"), make_sample("s0")]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(contract.Edge576ContractError):
                run_smoke_with(
                    Path(tmp),
                    arm=contract.ARM_DENSE,
                    sample_id="s0",
                    samples=samples,
                )

    def test_two_arms_smoke_the_same_sample_to_distinct_files(self) -> None:
        samples = [make_sample("s0")]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for arm in (contract.ARM_DENSE, contract.ARM_V2):
                run_smoke_with(
                    tmp_path, arm=arm, sample_id="s0", samples=samples
                )
            names = sorted(p.name for p in tmp_path.glob("*.json"))
            self.assertEqual(
                names, [f"{contract.ARM_DENSE}_s0.json", f"{contract.ARM_V2}_s0.json"]
            )

    def test_smoke_never_writes_a_full2638_output(self) -> None:
        samples = [make_sample("s0")]
        with tempfile.TemporaryDirectory() as tmp:
            run_smoke_with(
                Path(tmp),
                arm=contract.ARM_DENSE,
                sample_id="s0",
                samples=samples,
            )
        for full in contract.ARM_OUTPUT_PATHS.values():
            self.assertFalse(full.exists(), f"smoke wrote {full}")


class OutputTokenSummaryTest(unittest.TestCase):
    """Smoke handles the dense producer's missing real token ids."""

    def test_dense_smoke_without_token_ids_writes_auditable_count(self) -> None:
        samples = [make_sample("s0")]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            payload, _ = run_smoke_with(
                tmp_path,
                arm=contract.ARM_DENSE,
                sample_id="s0",
                samples=samples,
                include_token_ids=False,
                output_tokens=4,
            )
            record = payload["record"]
            self.assertNotIn("output_token_ids", record)
            self.assertFalse(record["output_token_ids_available"])
            self.assertEqual(record["output_token_count"], 4)
            self.assertEqual(record["output_text"], "The answer is (B).")
            summary = payload["smoke_summary"]
            self.assertEqual(summary["output_token_count"], 4)
            self.assertFalse(summary["output_token_ids_available"])
            self.assertFalse(summary["output_token_ids_recorded"])
            on_disk = json.loads(
                (tmp_path / f"{contract.ARM_DENSE}_s0.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(on_disk["record"], record)

    def test_l2_smoke_with_real_token_ids_records_them(self) -> None:
        samples = [make_sample("s0")]
        with tempfile.TemporaryDirectory() as tmp:
            payload, _ = run_smoke_with(
                Path(tmp),
                arm=contract.ARM_L2,
                sample_id="s0",
                samples=samples,
                output_tokens=3,
            )
            record = payload["record"]
            self.assertEqual(record["output_token_ids"], [1, 2, 3])
            self.assertTrue(record["output_token_ids_available"])
            self.assertEqual(record["output_token_count"], 3)
            summary = payload["smoke_summary"]
            self.assertTrue(summary["output_token_ids_available"])
            self.assertTrue(summary["output_token_ids_recorded"])

    def test_dense_smoke_without_any_token_field_fails_closed(self) -> None:
        samples = [make_sample("s0")]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaises(contract.Edge576ContractError):
                run_smoke_with(
                    tmp_path,
                    arm=contract.ARM_DENSE,
                    sample_id="s0",
                    samples=samples,
                    include_token_ids=False,
                )
            self.assertEqual(list(tmp_path.glob("*.json")), [])


class WarmupNotCountedTest(unittest.TestCase):
    def test_warmup_called_once_with_max_new_tokens_one(self) -> None:
        samples = [make_sample("s0")]
        with tempfile.TemporaryDirectory() as tmp:
            payload, harness = run_smoke_with(
                Path(tmp),
                arm=contract.ARM_V2,
                sample_id="s0",
                samples=samples,
            )
        self.assertEqual(harness.order.count("warmup"), 1)
        self.assertLess(
            harness.order.index("warmup"), harness.order.index("run_one")
        )
        self.assertEqual(harness.captured["warmup_sample"]["max_new_tokens"], 1)
        self.assertEqual(
            harness.captured["warmup_kwargs"]["max_new_tokens"], 1
        )
        self.assertEqual(harness.captured["warmup_kwargs"]["run_mode"], "smoke")
        self.assertFalse(payload["smoke_summary"]["warmup"]["wrote_record"])
        self.assertFalse(payload["smoke_summary"]["warmup_recorded_in_output"])
        self.assertEqual(payload["smoke_summary"]["warmup_count"], 1)

    def test_record_is_scored_at_max_new_tokens_16_not_the_warmup(self) -> None:
        samples = [make_sample("s0")]
        with tempfile.TemporaryDirectory() as tmp:
            payload, _ = run_smoke_with(
                Path(tmp),
                arm=contract.ARM_L2,
                sample_id="s0",
                samples=samples,
            )
        self.assertEqual(payload["record"]["max_new_tokens"], 16)
        self.assertEqual(payload["smoke_summary"]["max_new_tokens"], 16)
        self.assertEqual(payload["smoke_summary"]["records_written"], 1)

    def test_warmup_failure_writes_nothing(self) -> None:
        samples = [make_sample("s0")]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            def bad_warmup(arm, m, p, s, mp, **kw):
                raise contract.Edge576ContractError("warmup failed")

            with self.assertRaises(contract.Edge576ContractError):
                smoke.run_smoke(
                    arm=contract.ARM_DENSE,
                    sample_id="s0",
                    samples=samples,
                    manifest_path=contract.EDGE576_MANIFEST,
                    output=None,
                    model_path=Path("/tmp/model"),
                    model_label="test-model",
                    device="cuda:0",
                    smoke_dir=tmp_path,
                    validation_fn=lambda arm, s, *, manifest_path, output: fake_validation(
                        arm, s, manifest_path=manifest_path, output=output
                    ),
                    model_identity_fn=lambda path: {"ok": True},
                    require_cuda_fn=lambda device: None,
                    build_state_fn=lambda arm: {
                        "scorer": None,
                        "checkpoint_identity": {},
                    },
                    load_model_fn=lambda path, device: "M",
                    load_processor_fn=lambda path: contract.Edge576Processor(object()),
                    warmup_fn=bad_warmup,
                    run_one_fn=lambda arm, m, p, s, mp, **kw: fake_frozen(s, arm),
                    metadata_fn=fake_metadata,
                )
            self.assertEqual(list(tmp_path.iterdir()), [])


class GridFailCloseTest(unittest.TestCase):
    def test_matching_grid_writes_the_record(self) -> None:
        samples = [make_sample("s0")]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            payload, _ = run_smoke_with(
                tmp_path, arm=contract.ARM_L2, sample_id="s0", samples=samples
            )
            self.assertEqual(payload["record"]["image_grid_thw"], [1, 16, 16])
            self.assertTrue(
                (tmp_path / f"{contract.ARM_L2}_s0.json").is_file()
            )

    def test_grid_mismatch_aborts_and_writes_nothing(self) -> None:
        samples = [make_sample("s0")]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaises(contract.Edge576ContractError):
                run_smoke_with(
                    tmp_path,
                    arm=contract.ARM_L2,
                    sample_id="s0",
                    samples=samples,
                    grid=(1, 16, 32),
                )
            self.assertEqual(list(tmp_path.iterdir()), [])

    def test_missing_grid_aborts_and_writes_nothing(self) -> None:
        samples = [make_sample("s0")]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaises(contract.Edge576ContractError):
                run_smoke_with(
                    tmp_path,
                    arm=contract.ARM_DENSE,
                    sample_id="s0",
                    samples=samples,
                    grid=MISSING_GRID,
                )
            self.assertEqual(list(tmp_path.iterdir()), [])


class PreflightOrderTest(unittest.TestCase):
    def test_validation_then_model_identity_then_cuda_then_weights(self) -> None:
        samples = [make_sample("s0")]
        with tempfile.TemporaryDirectory() as tmp:
            _, harness = run_smoke_with(
                Path(tmp), arm=contract.ARM_DENSE, sample_id="s0", samples=samples
            )
        order = harness.order
        self.assertEqual(order[0], "validation")
        for later in ("model_identity", "require_cuda", "build_state",
                      "load_model", "load_processor", "warmup", "run_one"):
            self.assertLess(order.index("validation"), order.index(later))
        self.assertLess(order.index("model_identity"), order.index("require_cuda"))
        self.assertLess(order.index("model_identity"), order.index("load_model"))
        self.assertLess(order.index("warmup"), order.index("run_one"))

    def test_model_identity_failure_aborts_before_cuda_state_or_weight(self) -> None:
        samples = [make_sample("s0")]
        order: list[str] = []

        def identity(_path):
            order.append("model_identity")
            raise contract.Edge576ContractError("identity mismatch")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(contract.Edge576ContractError):
                smoke.run_smoke(
                    arm=contract.ARM_DENSE,
                    sample_id="s0",
                    samples=samples,
                    manifest_path=contract.EDGE576_MANIFEST,
                    output=None,
                    model_path=Path("/tmp/model"),
                    model_label="test-model",
                    device="cuda:0",
                    smoke_dir=Path(tmp),
                    validation_fn=lambda arm, s, *, manifest_path, output: (
                        order.append("validation")
                        or fake_validation(
                            arm, s, manifest_path=manifest_path, output=output
                        )
                    ),
                    model_identity_fn=identity,
                    require_cuda_fn=lambda device: order.append("require_cuda"),
                    build_state_fn=lambda arm: order.append("build_state"),
                    load_model_fn=lambda path, device: order.append("load_model"),
                    load_processor_fn=lambda path: order.append("load_processor"),
                    warmup_fn=lambda *a, **k: order.append("warmup"),
                    run_one_fn=lambda *a, **k: order.append("run_one"),
                    metadata_fn=fake_metadata,
                )
            self.assertEqual(order, ["validation", "model_identity"])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_model_is_required(self) -> None:
        samples = [make_sample("s0")]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(contract.Edge576ContractError):
                run_smoke_with(
                    Path(tmp),
                    arm=contract.ARM_DENSE,
                    sample_id="s0",
                    samples=samples,
                    model_path=None,
                )

    def test_processor_must_be_edge576_wrapped(self) -> None:
        samples = [make_sample("s0")]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(contract.Edge576ContractError):
                run_smoke_with(
                    Path(tmp),
                    arm=contract.ARM_DENSE,
                    sample_id="s0",
                    samples=samples,
                    processor=object(),
                )


class RealValidationTest(unittest.TestCase):
    def test_dense_validation_verifies_17_pins_and_mechanism(self) -> None:
        samples = contract.load_edge576_manifest()
        output = smoke.smoke_output_path(contract.ARM_DENSE, "1838")
        summary = smoke.smoke_validation(
            contract.ARM_DENSE,
            samples,
            manifest_path=contract.EDGE576_MANIFEST,
            output=output,
        )
        self.assertEqual(summary["dependency_pins"]["verified_count"], 17)
        self.assertTrue(summary["dependency_pins"]["all_match"])
        self.assertEqual(summary["pin_count"], 17)
        self.assertEqual(summary["mechanism_id"], "dense_edge576")
        self.assertFalse(summary["mechanism_identity"]["is_new_mechanism"])
        self.assertFalse(summary["weights_loaded"])
        self.assertEqual(summary["gpu_runs"], 0)
        self.assertFalse(summary["touches_full2638_output"])

    def test_dry_run_summary_is_cpu_only_and_lists_planned_command(self) -> None:
        samples = contract.load_edge576_manifest()
        output = smoke.smoke_output_path(contract.ARM_DENSE, "1838")
        summary = smoke.dry_run_summary(
            contract.ARM_DENSE,
            "1838",
            samples,
            manifest_path=contract.EDGE576_MANIFEST,
            output=output,
        )
        self.assertEqual(summary["status"], smoke.SMOKE_STATUS_DRY_RUN_OK)
        self.assertIn("--arm dense_edge576", summary["planned_command"])
        self.assertIn("--sample-id 1838", summary["planned_command"])
        self.assertEqual(summary["gpu_runs"], 0)

    def test_main_dry_run_writes_no_smoke_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            code = smoke.main(
                [
                    "--arm",
                    contract.ARM_DENSE,
                    "--sample-id",
                    "1838",
                    "--dry-run",
                    "--smoke-dir",
                    str(tmp_path),
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(list(tmp_path.iterdir()), [])

    def test_cli_requires_arm_and_sample_id(self) -> None:
        with self.assertRaises(SystemExit):
            smoke.build_parser().parse_args(["--sample-id", "1838"])
        with self.assertRaises(SystemExit):
            smoke.build_parser().parse_args(["--arm", "dense_edge576"])
        with self.assertRaises(SystemExit):
            smoke.build_parser().parse_args(
                ["--arm", "bogus", "--sample-id", "1838"]
            )


class WarmupRealDispatchTest(unittest.TestCase):
    def test_warmup_uses_shared_runner_warmup(self) -> None:
        self.assertIs(smoke.run_multimodal_warmup, runner.run_multimodal_warmup)
        source = Path(smoke.__file__).read_text(encoding="utf-8")
        self.assertIn("warmup_fn: Callable[..., dict[str, Any]] = run_multimodal_warmup", source)
        self.assertIn("max_new_tokens=1", source)


if __name__ == "__main__":
    unittest.main()
