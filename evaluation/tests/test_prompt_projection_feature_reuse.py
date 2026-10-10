"""CPU-only tests for image reuse in the prompt-projection feature extractor.

These tests never load real weights and never touch CUDA.  They drive the
injectable execution layer
(``evaluation.extract_prompt_projection_features.run_prompt_projection_extraction``)
with fake image loading / vision / prompt callbacks and assert:

* one vision forward per image with two prompts, two prompt calls;
* one vision forward per distinct image;
* a partially completed image resumes with exactly one vision forward;
* a fully completed image resumes with zero vision forwards;
* a changed content hash for the same resolved path is a hard failure;
* grouped output is byte-identical to a per-sample legacy reference for all
  five cached arrays;
* the emitted index stays in manifest order even when image groups interleave.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.extract_prompt_projection_features import (
    _load_index_records,
    plan_image_groups,
    run_prompt_projection_extraction,
)
from evaluation.metrics import load_jsonl
from evaluation.prompt_projection_distillation import (
    FEATURE_DIM,
    PROMPT_DIM,
    DistillationContractError,
    load_feature_cache,
    save_feature_cache,
    teacher_target,
)

G = 4
T = 3
TEMPERATURE = 0.1
CACHE_ARRAYS = (
    "group_mean",
    "prompt_embeddings",
    "teacher_scores",
    "teacher_scores_normalized",
    "teacher_soft_target",
)


def _sha_for(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _seed_from(text: str) -> int:
    return int(_sha_for(text)[:8], 16)


class FakeModelStack:
    """Deterministic CPU stand-in for the processor + dense vision tower."""

    def __init__(
        self,
        sha_map: dict[str, str] | None = None,
        sha_sequence: list[str] | None = None,
    ) -> None:
        self.vision_calls: list[str] = []
        self.prompt_calls: list[str] = []
        self.save_calls: list[str] = []
        self.load_calls: list[str] = []
        self.release_calls = 0
        self._sha_map = dict(sha_map or {})
        self._sha_sequence = list(sha_sequence or [])

    def resolve(self, manifest: Path, image: str) -> Path:
        return Path("/synthetic/images") / image

    def sha256(self, path: Path) -> str:
        key = str(path)
        if self._sha_sequence:
            return self._sha_sequence.pop(0)
        if key in self._sha_map:
            return self._sha_map[key]
        return _sha_for(key)

    def load_image(self, path: Path) -> Any:
        self.load_calls.append(str(path))
        return {"path": str(path)}

    def vision(self, sample: dict, image: Any, image_path: Path) -> dict:
        self.vision_calls.append(str(image_path))
        rng = np.random.default_rng(_seed_from(str(image_path)))
        merged = rng.normal(size=(G, PROMPT_DIM)).astype(np.float32)
        mean = rng.normal(size=(G, FEATURE_DIM)).astype(np.float32)
        return {
            "merged_embeddings": torch.tensor(merged),
            "group_mean": torch.tensor(mean),
            "grid_h": 2,
            "grid_w": 2,
        }

    def prompt(self, sample: dict) -> torch.Tensor:
        self.prompt_calls.append(str(sample["sample_id"]))
        rng = np.random.default_rng(_seed_from(str(sample["sample_id"])))
        return torch.tensor(rng.normal(size=(T, PROMPT_DIM)).astype(np.float32))

    def save(self, path: Path, **kwargs: Any) -> dict:
        self.save_calls.append(str(path))
        return save_feature_cache(path, **kwargs)

    def release(self) -> None:
        self.release_calls += 1


def _sample(sample_id: str, image: str, task: str, prompt: str) -> dict:
    return {
        "sample_id": sample_id,
        "image": image,
        "task": task,
        "prompt": prompt,
    }


class ReuseExecutionTests(unittest.TestCase):
    def _run(
        self,
        tmp: Path,
        samples: list[dict],
        stack: FakeModelStack,
        *,
        completed: dict[str, dict] | None = None,
        overwrite: bool = False,
    ) -> tuple[dict, Path, Path]:
        out_dir = tmp / "out"
        index_path = out_dir / "index.jsonl"
        report = run_prompt_projection_extraction(
            samples=samples,
            manifest_path=tmp / "manifest.jsonl",
            out_dir=out_dir,
            index_path=index_path,
            temperature=TEMPERATURE,
            config_hash="cfg",
            resolve_image_fn=stack.resolve,
            image_sha256_fn=stack.sha256,
            image_loader_fn=stack.load_image,
            vision_capture_fn=stack.vision,
            prompt_embeddings_fn=stack.prompt,
            completed_records=completed,
            overwrite=overwrite,
            save_cache_fn=stack.save,
            release_fn=stack.release,
        )
        return report, out_dir, index_path

    def test_same_image_two_prompts_one_vision_two_prompt_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            samples = [
                _sample("A-g", "A.jpg", "grounding", "pa0"),
                _sample("A-p", "A.jpg", "pointing", "pa1"),
            ]
            stack = FakeModelStack()
            report, out_dir, index_path = self._run(tmp, samples, stack)
            self.assertEqual(report["vision_forwards"], 1)
            self.assertEqual(report["prompt_embeddings"], 2)
            self.assertEqual(report["records_written"], 2)
            self.assertEqual(stack.vision_calls, ["/synthetic/images/A.jpg"])
            self.assertCountEqual(stack.prompt_calls, ["A-g", "A-p"])
            self.assertEqual(stack.release_calls, 1)
            self.assertEqual(stack.load_calls, ["/synthetic/images/A.jpg"])
            records = load_jsonl(index_path)
            self.assertEqual([r["sample_id"] for r in records], ["A-g", "A-p"])
            for sample_id in ("A-g", "A-p"):
                self.assertTrue((out_dir / f"{sample_id}.npz").is_file())

    def test_different_images_one_vision_each(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            samples = [
                _sample("A-g", "A.jpg", "grounding", "pa"),
                _sample("B-g", "B.jpg", "grounding", "pb"),
            ]
            stack = FakeModelStack()
            report, _out_dir, _index = self._run(tmp, samples, stack)
            self.assertEqual(report["vision_forwards"], 2)
            self.assertEqual(report["prompt_embeddings"], 2)
            self.assertEqual(report["images_total"], 2)
            self.assertEqual(report["images_processed"], 2)
            self.assertCountEqual(
                stack.vision_calls,
                ["/synthetic/images/A.jpg", "/synthetic/images/B.jpg"],
            )

    def test_interleaved_groups_keep_manifest_index_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # Two images each with two prompts, interleaved: A0, B0, A1, B1.
            samples = [
                _sample("A0", "A.jpg", "grounding", "pa0"),
                _sample("B0", "B.jpg", "grounding", "pb0"),
                _sample("A1", "A.jpg", "pointing", "pa1"),
                _sample("B1", "B.jpg", "pointing", "pb1"),
            ]
            stack = FakeModelStack()
            report, _out_dir, index_path = self._run(tmp, samples, stack)
            self.assertEqual(report["vision_forwards"], 2)
            self.assertEqual(report["prompt_embeddings"], 4)
            records = load_jsonl(index_path)
            self.assertEqual(
                [r["sample_id"] for r in records], ["A0", "B0", "A1", "B1"]
            )

    def test_partial_resume_uses_exactly_one_vision_forward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            full = [
                _sample("A-g", "A.jpg", "grounding", "pa0"),
                _sample("A-p", "A.jpg", "pointing", "pa1"),
            ]
            first_stack = FakeModelStack()
            self._run(tmp, full[:1], first_stack)
            index_path = tmp / "out" / "index.jsonl"
            completed = _load_index_records(index_path)
            self.assertEqual(set(completed), {"A-g"})

            resume_stack = FakeModelStack()
            report, out_dir, index_path = self._run(
                tmp, full, resume_stack, completed=completed
            )
            self.assertEqual(report["vision_forwards"], 1)
            self.assertEqual(report["prompt_embeddings"], 1)
            self.assertEqual(report["records_written"], 1)
            self.assertEqual(resume_stack.vision_calls, ["/synthetic/images/A.jpg"])
            self.assertEqual(resume_stack.prompt_calls, ["A-p"])
            self.assertEqual(resume_stack.save_calls, [str(out_dir / "A-p.npz")])
            self.assertEqual(
                [r["sample_id"] for r in load_jsonl(index_path)], ["A-g", "A-p"]
            )

    def test_full_resume_uses_zero_vision_forwards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            samples = [
                _sample("A-g", "A.jpg", "grounding", "pa0"),
                _sample("A-p", "A.jpg", "pointing", "pa1"),
            ]
            first_stack = FakeModelStack()
            self._run(tmp, samples, first_stack)
            index_path = tmp / "out" / "index.jsonl"
            completed = _load_index_records(index_path)

            resume_stack = FakeModelStack()
            report, _out_dir, index_path = self._run(
                tmp, samples, resume_stack, completed=completed
            )
            self.assertEqual(report["vision_forwards"], 0)
            self.assertEqual(report["prompt_embeddings"], 0)
            self.assertEqual(report["records_written"], 0)
            self.assertEqual(resume_stack.vision_calls, [])
            self.assertEqual(resume_stack.prompt_calls, [])
            self.assertEqual(resume_stack.load_calls, [])
            self.assertEqual(len(load_jsonl(index_path)), 2)

    def test_same_path_content_hash_change_is_a_hard_failure_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            samples = [
                _sample("A-g", "A.jpg", "grounding", "pa0"),
                _sample("A-p", "A.jpg", "pointing", "pa1"),
            ]
            first_stack = FakeModelStack()
            self._run(tmp, samples[:1], first_stack)
            completed = _load_index_records(tmp / "out" / "index.jsonl")
            changed = FakeModelStack(
                sha_map={"/synthetic/images/A.jpg": _sha_for("changed-bytes")}
            )
            with self.assertRaises(DistillationContractError):
                self._run(tmp, samples, changed, completed=completed)
            self.assertEqual(changed.vision_calls, [])

    def test_same_path_content_hash_change_within_run_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            samples = [
                _sample("A-g", "A.jpg", "grounding", "pa0"),
                _sample("A-p", "A.jpg", "pointing", "pa1"),
            ]
            stack = FakeModelStack(
                sha_sequence=[_sha_for("v1"), _sha_for("v2")]
            )
            with self.assertRaises(DistillationContractError):
                self._run(tmp, samples, stack)
            self.assertEqual(stack.vision_calls, [])

    def test_grouped_output_matches_per_sample_legacy_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            samples = [
                _sample("A-g", "A.jpg", "grounding", "pa0"),
                _sample("A-p", "A.jpg", "pointing", "pa1"),
                _sample("B-g", "B.jpg", "grounding", "pb0"),
                _sample("B-p", "B.jpg", "pointing", "pb1"),
            ]
            stack = FakeModelStack()
            report, out_dir, _index = self._run(tmp, samples, stack)
            self.assertEqual(report["vision_forwards"], 2)

            legacy_dir = tmp / "legacy"
            legacy_dir.mkdir()
            legacy: dict[str, dict] = {}
            for sample in samples:
                image_path = Path("/synthetic/images") / sample["image"]
                image = stack.load_image(image_path)
                capture = stack.vision(sample, image, image_path)
                prompt = stack.prompt(sample)
                target = teacher_target(
                    capture["merged_embeddings"].cpu().numpy(),
                    prompt.cpu().numpy(),
                    TEMPERATURE,
                )
                save_feature_cache(
                    legacy_dir / f"{sample['sample_id']}.npz",
                    group_mean=capture["group_mean"],
                    prompt_embeddings=prompt,
                    teacher_raw=target["raw"],
                    teacher_normalized=target["normalized"],
                    teacher_soft=target["soft"],
                    metadata={"sample_id": sample["sample_id"]},
                )
                legacy[sample["sample_id"]] = load_feature_cache(
                    legacy_dir / f"{sample['sample_id']}.npz"
                )

            for sample in samples:
                sample_id = sample["sample_id"]
                grouped = load_feature_cache(out_dir / f"{sample_id}.npz")
                for key in CACHE_ARRAYS:
                    np.testing.assert_array_equal(
                        grouped[key],
                        legacy[sample_id][key],
                        err_msg=f"{sample_id}:{key}",
                    )
                self.assertEqual(grouped["metadata"]["sample_id"], sample_id)
                self.assertEqual(
                    int(grouped["metadata"]["grid_h"]), 2
                )

    def test_plan_image_groups_first_occurrence_and_manifest_indices(self) -> None:
        samples = [
            _sample("A0", "A.jpg", "grounding", "pa0"),
            _sample("B0", "B.jpg", "grounding", "pb0"),
            _sample("A1", "A.jpg", "pointing", "pa1"),
        ]
        groups = plan_image_groups(
            samples,
            resolve_image_fn=lambda _m, image: Path("/synthetic/images") / image,
            image_sha256_fn=lambda path: _sha_for(str(path)),
            completed_records={},
            manifest_path=Path("/synthetic/manifest.jsonl"),
        )
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0].resolved_image_path, "/synthetic/images/A.jpg")
        self.assertEqual(groups[0].manifest_indices, (0, 2))
        self.assertEqual(groups[1].manifest_indices, (1,))
        self.assertEqual(
            [str(s["sample_id"]) for s in groups[0].samples], ["A0", "A1"]
        )


class ValidateManifestOnlyIsolationTests(unittest.TestCase):
    def test_validate_manifest_only_imports_no_model_stack(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        config = repo_root / "configs" / "prompt_projection_distillation.json"
        manifest = (
            repo_root / "datasets" / "prompt_projection_clean_inductive_v1.jsonl"
        )
        if not (config.is_file() and manifest.is_file()):
            self.skipTest("frozen clean-inductive manifest is not present")
        script = textwrap.dedent(
            f"""
            import builtins, sys
            BLOCKED = {{"transformers", "PIL"}}
            real_import = builtins.__import__
            def guarded(name, *args, **kwargs):
                if name.split(".")[0] in BLOCKED:
                    raise AssertionError("forbidden import: " + name)
                return real_import(name, *args, **kwargs)
            builtins.__import__ = guarded
            sys.path.insert(0, {str(repo_root)!r})
            from evaluation.extract_prompt_projection_features import main
            code = main([
                "--manifest", {str(manifest)!r},
                "--config", {str(config)!r},
                "--validate-manifest-only",
            ])
            raise SystemExit(code)
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "PASS")
        self.assertFalse(payload["gpu_used"])
        self.assertFalse(payload["model_loaded"])


if __name__ == "__main__":
    unittest.main()
