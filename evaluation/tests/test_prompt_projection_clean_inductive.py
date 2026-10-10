"""CPU tests for the strict-inductive clean prompt-projection corpus.

The tests build a small synthetic repository (source manifests, images and
formal CV-Bench manifests) and exercise the pure builder/validator functions.
They never download anything, never load a model and never touch a GPU.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from evaluation.prompt_projection_clean_inductive import (
    CleanInductiveError,
    SAMPLE_ID_PREFIX,
    SCHEMA,
    aggregate_image_set_sha256,
    build_clean_records,
    build_lock,
    canonical_manifest_bytes,
    make_sample_id,
    per_image_facts,
    recover_cvbench_coco_image_ids,
    validate_manifest_and_lock,
)
from evaluation.prompt_projection_distillation import (
    DistillationContractError,
    assign_split,
    pinned_manifest_spec,
    validate_pinned_manifest,
)

SPLIT = {
    "salt": "synthetic-test",
    "seed": 7,
    "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
    "group_by": "source_image_sha256",
    "require_image_sha256": True,
}
CONFIG: dict[str, Any] = {"split": SPLIT, "extraction": {}}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _source_record(coco_image_id: int, task: str) -> dict[str, Any]:
    return {
        "coco_image_id": coco_image_id,
        "coco_annotation_id": 2000 + coco_image_id % 1000,
        "sample_id": f"orig-{coco_image_id}-{task}",
        "task": task,
        "image": f"coco/{coco_image_id:012d}.jpg",
        "prompt": f"prompt for {coco_image_id} {task}",
        "ground_truth": {"point_1000": [1.0, 2.0]},
    }


def _make_repo(root: Path, n_images: int = 20) -> dict[str, Any]:
    (root / "datasets").mkdir(parents=True)
    (root / "evaluation").mkdir(parents=True)
    (root / "evaluation" / "prompt_projection_clean_inductive.py").write_text(
        "# fixture construction script\n", encoding="utf-8"
    )
    image_dir = root / "datasets" / "coco"
    image_dir.mkdir()
    left: list[dict[str, Any]] = []
    right: list[dict[str, Any]] = []
    for index in range(n_images):
        coco_image_id = 1000 + index
        (image_dir / f"{coco_image_id:012d}.jpg").write_bytes(
            f"image-{coco_image_id}".encode()
        )
        left.append(_source_record(coco_image_id, "grounding"))
        right.append(_source_record(coco_image_id, "pointing"))
    _write_jsonl(root / "datasets" / "srcA.jsonl", left)
    _write_jsonl(root / "datasets" / "srcB.jsonl", right)
    for name in ("formal_a", "formal_b"):
        _write_jsonl(
            root / "datasets" / f"{name}.jsonl",
            [
                {
                    "sample_id": f"formal-{name}-0",
                    "image": f"formal/{name}.jpg",
                }
            ],
        )
    return {"images": n_images}


def _build_fixture(root: Path) -> dict[str, Any]:
    _make_repo(root)
    sources = ["datasets/srcA.jsonl", "datasets/srcB.jsonl"]
    cvbench_ids = [1000, 1001, 9999]
    records, stats = build_clean_records(sources, cvbench_ids, repo_root=root)
    facts = per_image_facts(records, repo_root=root)
    base_ids = set(int(value) for value in stats["base_coco_image_ids"])
    excluded = sorted(base_ids & set(cvbench_ids))
    retained = sorted(facts)
    manifest_path = root / "datasets" / "clean.jsonl"
    manifest_path.write_bytes(canonical_manifest_bytes(records))
    config = {
        "split": SPLIT,
        "extraction": {
            "manifest": {
                "path": "datasets/clean.jsonl",
                "sha256": sha256_bytes(manifest_path.read_bytes()),
            }
        },
    }
    formal_specs = [
        {"id": "fa", "path": "datasets/formal_a.jsonl"},
        {"id": "fb", "path": "datasets/formal_b.jsonl"},
    ]
    lock = build_lock(
        repo_root=root,
        manifest_path=manifest_path,
        config_path=root / "config.json",
        records=records,
        facts=facts,
        excluded_coco_image_ids=excluded,
        retained_coco_image_ids=retained,
        cvbench_coco_image_ids=cvbench_ids,
        source_stats=stats,
        config=config,
        formal_image_sha256_count=0,
        formal_image_sha256_digest=sha256_bytes(b""),
        formal_manifest_specs=formal_specs,
        parquet_specs=[],
    )
    lock_path = root / "clean.lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", "utf-8")
    return {
        "records": records,
        "stats": stats,
        "facts": facts,
        "excluded": excluded,
        "retained": retained,
        "manifest_path": manifest_path,
        "lock": lock,
        "lock_path": lock_path,
        "config": config,
        "cvbench_ids": cvbench_ids,
        "formal_sample_ids": {"formal-fa-0", "formal-fb-0"},
    }


class RecoverTests(unittest.TestCase):
    def test_recover_cvbench_coco_ids_from_parquet(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_2d.parquet"
            table = pa.table(
                {
                    "source": ["COCO", "COCO", "ADE20K", "Omni3D"],
                    "source_filename": [
                        "/coco/val2017/000000000802.jpg",
                        "/coco/val2017/000000001425.jpg",
                        "ADE_val_00000248.jpg",
                        "hypersim/frame.jpg",
                    ],
                }
            )
            pq.write_table(table, path)
            self.assertEqual(recover_cvbench_coco_image_ids([path]), [802, 1425])


class BuildRecordTests(unittest.TestCase):
    def test_new_sample_id_is_stable_and_traceable(self) -> None:
        record = _source_record(802, "grounding")
        cleaned = {
            **record,
            "sample_id": make_sample_id(802, record["coco_annotation_id"], "grounding"),
            "source_sample_id": record["sample_id"],
        }
        self.assertEqual(
            cleaned["sample_id"],
            f"{SAMPLE_ID_PREFIX}-000000000802-2802-grounding",
        )
        self.assertEqual(cleaned["source_sample_id"], record["sample_id"])

    def test_exclusion_drops_all_prompts_of_overlapping_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_repo(root)
            records, stats = build_clean_records(
                ["datasets/srcA.jsonl", "datasets/srcB.jsonl"],
                [1000, 1001],
                repo_root=root,
            )
            self.assertEqual(len(records), 36)  # 18 images x 2 prompts
            self.assertEqual(
                sum(stat["records_kept"] for stat in stats["sources"]), 36
            )
            self.assertEqual(
                sum(stat["records_excluded"] for stat in stats["sources"]), 4
            )
            image_ids = {int(record["coco_image_id"]) for record in records}
            self.assertNotIn(1000, image_ids)
            self.assertNotIn(1001, image_ids)
            self.assertTrue(
                all(
                    record["sample_id"].startswith(SAMPLE_ID_PREFIX)
                    for record in records
                )
            )
            self.assertTrue(
                all(record["source_sample_id"] for record in records)
            )

    def test_aggregate_image_set_matches_sorted_sha_lines(self) -> None:
        facts = {
            1: {"path": "a/one.jpg", "sha256": "a" * 64, "bytes": 1},
            2: {"path": "b/two.jpg", "sha256": "b" * 64, "bytes": 2},
        }
        expected = sha256_bytes(
            ("\n".join(sorted([f"{'b' * 64}  two.jpg", f"{'a' * 64}  one.jpg"]))).encode()
        )
        self.assertEqual(aggregate_image_set_sha256(facts), expected)


class ValidationTests(unittest.TestCase):
    def test_lock_has_required_frozen_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _build_fixture(Path(tmp))
            lock = fixture["lock"]
            self.assertEqual(lock["schema"], SCHEMA)
            self.assertTrue(lock["strict_inductive"])
            self.assertFalse(lock["sealed_test_used"])
            self.assertFalse(lock["prompt_distribution_matches_cvbench"])
            self.assertEqual(lock["manifest"]["records"], 36)
            self.assertEqual(lock["manifest"]["unique_images"], 18)
            self.assertEqual(lock["manifest"]["tasks"], {"grounding": 18, "pointing": 18})
            self.assertEqual(lock["cvbench_coco"]["excluded_image_ids"], [1000, 1001])
            self.assertEqual(len(lock["images"]), 18)
            self.assertTrue(
                all({"path", "bytes", "sha256"} <= set(entry) for entry in lock["images"])
            )
            self.assertTrue(lock["config_pin"]["matches_manifest"])
            self.assertEqual(lock["construction_script"]["path"], "evaluation/prompt_projection_clean_inductive.py")

    def test_validate_passes_on_clean_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _build_fixture(root)
            result = validate_manifest_and_lock(
                repo_root=root,
                manifest_path=fixture["manifest_path"],
                lock_path=fixture["lock_path"],
                config=fixture["config"],
                formal_sample_ids=fixture["formal_sample_ids"],
                formal_eval_image_sha256=set(),
                recovered_cvbench_coco_ids=fixture["cvbench_ids"],
            )
            self.assertEqual(result["status"], "PASS", result["failures"])
            self.assertTrue(result["gates"]["zero_cvbench_coco_overlap"]["passed"])
            self.assertTrue(
                result["gates"]["zero_formal_sample_id_overlap"]["passed"]
            )
            self.assertTrue(
                result["gates"]["zero_formal_eval_image_sha_overlap"]["passed"]
            )
            self.assertTrue(result["gates"]["same_image_same_split"]["passed"])
            self.assertTrue(
                result["gates"]["same_image_same_split"]["all_splits_nonempty"]
            )

    def test_validate_detects_sample_id_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _build_fixture(root)
            result = validate_manifest_and_lock(
                repo_root=root,
                manifest_path=fixture["manifest_path"],
                lock_path=fixture["lock_path"],
                config=fixture["config"],
                formal_sample_ids={fixture["records"][0]["sample_id"]},
                formal_eval_image_sha256=set(),
                recovered_cvbench_coco_ids=fixture["cvbench_ids"],
            )
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(
                result["gates"]["zero_formal_sample_id_overlap"]["passed"]
            )

    def test_validate_detects_formal_image_sha_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _build_fixture(root)
            retained_sha = fixture["lock"]["images"][0]["sha256"]
            result = validate_manifest_and_lock(
                repo_root=root,
                manifest_path=fixture["manifest_path"],
                lock_path=fixture["lock_path"],
                config=fixture["config"],
                formal_sample_ids=set(),
                formal_eval_image_sha256={retained_sha},
                recovered_cvbench_coco_ids=fixture["cvbench_ids"],
            )
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(
                result["gates"]["zero_formal_eval_image_sha_overlap"]["passed"]
            )

    def test_validate_detects_tampered_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _build_fixture(root)
            records = list(fixture["records"])
            records[0] = {**records[0], "prompt": "tampered"}
            fixture["manifest_path"].write_bytes(canonical_manifest_bytes(records))
            result = validate_manifest_and_lock(
                repo_root=root,
                manifest_path=fixture["manifest_path"],
                lock_path=fixture["lock_path"],
                config=fixture["config"],
                formal_sample_ids=set(),
                formal_eval_image_sha256=set(),
                recovered_cvbench_coco_ids=fixture["cvbench_ids"],
            )
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["manifest_sha256"]["passed"])


class PinTests(unittest.TestCase):
    def test_pinned_manifest_spec_rejects_missing_or_absolute(self) -> None:
        with self.assertRaises(DistillationContractError):
            pinned_manifest_spec({"extraction": {}})
        with self.assertRaises(DistillationContractError):
            pinned_manifest_spec(
                {
                    "extraction": {
                        "manifest": {"path": "/abs/x.jsonl", "sha256": "0" * 64}
                    }
                }
            )

    def test_validate_pinned_manifest_pass_and_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "datasets" / "clean.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("x\n", encoding="utf-8")
            sha = sha256_bytes(manifest.read_bytes())
            config = {
                "extraction": {
                    "manifest": {
                        "path": "datasets/clean.jsonl",
                        "sha256": sha,
                    }
                }
            }
            result = validate_pinned_manifest(
                config, manifest_path=manifest, repo_root=root
            )
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["path_matches"])
            self.assertTrue(result["sha256_matches"])

            wrong = root / "datasets" / "other.jsonl"
            wrong.write_text("y\n", encoding="utf-8")
            mismatch = validate_pinned_manifest(
                config, manifest_path=wrong, repo_root=root
            )
            self.assertEqual(mismatch["status"], "FAIL")
            self.assertFalse(mismatch["path_matches"])

            manifest.write_text("tampered\n", encoding="utf-8")
            stale = validate_pinned_manifest(
                config, manifest_path=manifest, repo_root=root
            )
            self.assertEqual(stale["status"], "FAIL")
            self.assertTrue(stale["path_matches"])
            self.assertFalse(stale["sha256_matches"])


if __name__ == "__main__":
    unittest.main()
