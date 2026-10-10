#!/usr/bin/env python3
"""Build the label-free student index for VSR B1 teacher caches."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.extract_vsr_b1_margin_teacher import DEFAULT_MANIFEST, DEFAULT_OUT_DIR, sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = DEFAULT_OUT_DIR / "student_index.jsonl"
REPORT = ROOT / "reports/token_selector/2026-10-10-vsr-b1-student-index.json"


def build() -> dict:
    manifest = {row["sample_id"]: row for row in (json.loads(line) for line in DEFAULT_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip())}
    teacher_index = DEFAULT_OUT_DIR / "index.jsonl"
    teacher = [json.loads(line) for line in teacher_index.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for record in teacher:
        sample_id = str(record["sample_id"])
        source = manifest[sample_id]
        audit = record["audit"]
        rows.append({
            "sample_id": sample_id,
            "task": "vsr_binary_spatial_reasoning",
            "npz": str(record["npz_name"]),
            "group_count": int(audit["group_count"]),
            "prompt_token_count": int(audit["prompt_token_count"]),
            "grid_h": int(audit["grid_h"]),
            "grid_w": int(audit["grid_w"]),
            "sample_sha256": str(record["cache_fingerprint"]),
            "source_image_sha256": str(source["image_sha256"]),
            "teacher_index_record_sha256": str(record["record_sha256"]),
            "contains_official_label": False,
            "contains_relation_annotation": False,
        })
    if len(rows) != 3483 or len({row["sample_id"] for row in rows}) != 3483:
        raise ValueError("student index must contain 3483 unique records")
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    OUTPUT.write_text(payload, encoding="utf-8")
    report = {
        "status": "VSR_B1_LABEL_FREE_STUDENT_INDEX_COMPLETE",
        "records": len(rows),
        "unique_images": len({row["source_image_sha256"] for row in rows}),
        "contains_official_label": False,
        "contains_relation_subject_object_annotations": False,
        "output": str(OUTPUT), "output_sha256": sha256_file(OUTPUT),
        "teacher_index_sha256": sha256_file(teacher_index),
        "source_manifest_sha256": sha256_file(DEFAULT_MANIFEST),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
