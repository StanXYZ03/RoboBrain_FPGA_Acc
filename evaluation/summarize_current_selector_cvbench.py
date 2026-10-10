#!/usr/bin/env python3
"""Summarise current-Selector CV-Bench raw JSONL with the official scorer.

The formal CV-Bench score is emitted **only** when one candidate JSONL contains
exactly the 2638 unique manifest samples, each with a complete record.  Any
partial run is reported as ``IN_PROGRESS`` with counts and missing ids; no
partial-set score is ever computed.  Scoring reuses the frozen official helpers
:func:`evaluation.build_cv_bench_pruning_report.score` and
:func:`...paired_bootstrap_delta` (the README formula
``((2D_ade + 2D_coco)/2 + 3D_omni)/2``); no new formula is defined here.

Forbidden sources: Pointing, Grounding, selected-set NDCG and retained spatial
mass never enter any field of this report.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.build_cv_bench_pruning_report import (
    answer_letter,
    paired_bootstrap_delta,
    parse_choice,
    score,
)
from evaluation.metrics import load_jsonl

STATUS_COMPLETE = "COMPLETE"
STATUS_IN_PROGRESS = "IN_PROGRESS"
FORMAL_SAMPLE_COUNT = 2638
FORBIDDEN_PRIMARY_METRICS = (
    "pointing",
    "grounding",
    "ndcg",
    "retained_mass",
)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != FORMAL_SAMPLE_COUNT:
        raise ValueError(
            f"manifest must hold {FORMAL_SAMPLE_COUNT} samples, got {len(rows)}"
        )
    return rows


def summarise_one(
    *,
    label: str,
    candidate_jsonl: Path,
    manifest: list[dict[str, Any]],
    native: Mapping[str, Any],
    dense: Mapping[str, Mapping[str, Any]],
    seed: int,
    bootstrap_repeats: int,
) -> dict[str, Any]:
    if not candidate_jsonl.is_file():
        return {
            "label": label,
            "status": STATUS_IN_PROGRESS,
            "candidate_jsonl": str(candidate_jsonl),
            "completed": 0,
            "expected": len(manifest),
            "reason": "file_missing",
        }
    candidate_rows = load_jsonl(candidate_jsonl)
    by_id: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in candidate_rows:
        sample_id = str(row["sample_id"])
        if sample_id in by_id:
            duplicates.append(sample_id)
        by_id[sample_id] = row
    expected_ids = {str(row["sample_id"]) for row in manifest}
    missing = sorted(expected_ids - set(by_id))
    extra = sorted(set(by_id) - expected_ids)
    result: dict[str, Any] = {
        "label": label,
        "status": STATUS_IN_PROGRESS,
        "candidate_jsonl": str(candidate_jsonl),
        "completed": len(by_id),
        "expected": len(expected_ids),
        "duplicate_sample_ids": duplicates,
        "extra_sample_ids": extra,
    }
    if duplicates or extra or missing or len(by_id) != len(expected_ids):
        result["reason"] = "incomplete_or_inconsistent"
        result["missing_sample_ids"] = missing[:200]
        result["missing_count"] = len(missing)
        return result

    joined: list[dict[str, Any]] = []
    manifest_by_id = {str(row["sample_id"]): row for row in manifest}
    for sample_id in manifest_by_id:
        meta = manifest_by_id[sample_id]
        row = by_id[sample_id]
        choices = list(meta["choices"])
        truth = answer_letter(str(meta["answer"]))
        native_row = native[str(sample_id)]
        dense_row = dense[sample_id]
        native_answer = native_row.get("parsed_answer") or parse_choice(
            native_row["output_text"], choices
        )
        dense_answer = parse_choice(dense_row["output_text"], choices)
        candidate_answer = row.get("parsed_answer")
        if candidate_answer is None:
            candidate_answer = parse_choice(row["output_text"], choices)
        joined.append(
            {
                "sample_id": sample_id,
                "task": meta["task"],
                "type": meta["type"],
                "source": meta["source"],
                "source_dataset": meta["source_dataset"],
                "ground_truth": truth,
                "native_answer": native_answer,
                "dense_answer": dense_answer,
                "candidate_answer": candidate_answer,
                "native_correct": native_answer == truth,
                "dense_correct": dense_answer == truth,
                "candidate_correct": candidate_answer == truth,
            }
        )

    candidate_score = score(joined, "candidate_correct")
    dense_score = score(joined, "dense_correct")
    native_score = score(joined, "native_correct")
    ci = paired_bootstrap_delta(
        joined, "candidate_correct", "dense_correct", seed, repeats=bootstrap_repeats
    )
    ci_native = paired_bootstrap_delta(
        joined, "candidate_correct", "native_correct", seed + 1, repeats=bootstrap_repeats
    )
    result.update(
        {
            "status": STATUS_COMPLETE,
            "sample_count": len(joined),
            "selector_path": "software",
            "official_scorer": (
                "evaluation.build_cv_bench_pruning_report."
                "parse_choice/answer_letter/score/paired_bootstrap_delta"
            ),
            "candidate": candidate_score,
            "dense_bf16_reference": dense_score,
            "native_bf16_reference": native_score,
            "absolute_delta_vs_dense": (
                candidate_score["cv_bench_accuracy"]
                - dense_score["cv_bench_accuracy"]
            ),
            "relative_retention_vs_dense": (
                candidate_score["cv_bench_accuracy"]
                / dense_score["cv_bench_accuracy"]
                if dense_score["cv_bench_accuracy"]
                else None
            ),
            "paired_bootstrap_ci95_vs_dense": ci,
            "paired_bootstrap_ci95_vs_native": ci_native,
            "bootstrap_seed": int(seed),
            "bootstrap_repeats": int(bootstrap_repeats),
            "forbidden_primary_metrics": list(FORBIDDEN_PRIMARY_METRICS),
        }
    )
    return result


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Current Selector CV-Bench gate summary",
        "",
        f"- generated_at: {report.get('generated_at')}",
        f"- manifest: `{report['manifest']}`",
        f"- manifest_sha256: `{report['manifest_sha256']}`",
        f"- status: {report['status']}",
        f"- forbidden primary metrics (never reported): "
        f"{', '.join(FORBIDDEN_PRIMARY_METRICS)}",
        "",
    ]
    for entry in report["candidates"]:
        lines.append(f"## {entry['label']}")
        if entry["status"] != STATUS_COMPLETE:
            lines.append(
                f"- IN_PROGRESS: {entry.get('completed')}/{entry.get('expected')} "
                f"({entry.get('reason')})"
            )
            lines.append("")
            continue
        lines.append(f"- status: COMPLETE, samples: {entry['sample_count']}")
        for name, block in (
            ("candidate", entry["candidate"]),
            ("dense_bf16", entry["dense_bf16_reference"]),
            ("native_bf16", entry["native_bf16_reference"]),
        ):
            lines.append(
                f"- {name}: combined={block['cv_bench_accuracy']:.6f} "
                f"2D={block['accuracy_2d']:.6f} 3D={block['accuracy_3d']:.6f} "
                f"parse={block['parse_success_rate']:.6f}"
            )
            lines.append(
                "  - tasks: "
                + ", ".join(
                    f"{task}={value:.6f}"
                    for task, value in sorted(block["task_accuracy"].items())
                )
            )
        lines.append(f"- absolute_delta_vs_dense: {entry['absolute_delta_vs_dense']:.6f}")
        lines.append(f"- relative_retention_vs_dense: {entry['relative_retention_vs_dense']:.6f}")
        lines.append(
            f"- paired_bootstrap_ci95_vs_dense: {entry['paired_bootstrap_ci95_vs_dense']}"
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native-json", type=Path, required=True)
    parser.add_argument("--dense-jsonl", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=2,
        metavar=("LABEL", "JSONL"),
        required=True,
        help="repeat for each K, e.g. --candidate K48 path/to/raw.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    return parser


def main() -> int:
    from datetime import datetime, timezone

    args = build_parser().parse_args()
    manifest = _load_manifest(args.manifest)
    native_payload = json.loads(args.native_json.read_text(encoding="utf-8"))
    native = {str(row["idx"]): row for row in native_payload["records"]}
    dense_rows = load_jsonl(args.dense_jsonl)
    dense = {str(row["sample_id"]): row for row in dense_rows}
    expected_ids = {str(row["sample_id"]) for row in manifest}
    if set(native) != expected_ids:
        raise ValueError("native baseline sample ids do not match the manifest")
    if set(dense) != expected_ids:
        raise ValueError("dense baseline sample ids do not match the manifest")

    import hashlib

    candidates = [
        summarise_one(
            label=label,
            candidate_jsonl=Path(path),
            manifest=manifest,
            native=native,
            dense=dense,
            seed=args.seed,
            bootstrap_repeats=args.bootstrap_repeats,
        )
        for label, path in args.candidate
    ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "native_json": str(args.native_json),
        "dense_jsonl": str(args.dense_jsonl),
        "status": (
            STATUS_COMPLETE
            if all(entry["status"] == STATUS_COMPLETE for entry in candidates)
            else STATUS_IN_PROGRESS
        ),
        "formal_sample_count": FORMAL_SAMPLE_COUNT,
        "forbidden_primary_metrics": list(FORBIDDEN_PRIMARY_METRICS),
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
