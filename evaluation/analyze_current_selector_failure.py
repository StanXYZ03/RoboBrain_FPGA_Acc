#!/usr/bin/env python3
"""Diagnose the 2026-10-04 current integer Selector CV-Bench failure.

Execution-only, CPU-only, read-only analysis of frozen artifacts:

* the formal quality report ``reports/cv_bench/2026-10-04-current-selector-cvbench.json``
* the K48/K64/K96 full-2638 raw JSONL under
  ``artifacts/token_selector/2026-10-04-current-selector-cvbench/``
* the frozen dense reference JSONL named by the quality report
* the already-frozen L2@25%-after-layer-5 report
  ``reports/cv_bench/2026-09-16-cv-bench-l2-0250-full2638.json``.

The analysis re-uses the **official** CV-Bench choice scorer
:func:`evaluation.build_cv_bench_pruning_report.parse_choice` /
:func:`answer_letter` / :func:`score`; no second accuracy formula is defined.

Two hard rules are enforced:

1. Pointing / Grounding / NDCG / retained-mass never become a primary metric.
2. The frozen L2@25% baseline is compared only as a *different cut point*
   (vision layer 5 vs the current B1 = after vision block 0).  Any accuracy
   difference is therefore **not** attributable to the ranker/selector alone;
   that sentence is emitted verbatim into the JSON and Markdown.

Metric-naming rule (added by the 2026-10-04 correction):

* The **only** formal CV-Bench primary is ``score(...)["cv_bench_accuracy"]``
  (``((ADE20K + COCO)/2 + Omni3D)/2``); the formal report match and the
  L2@25% comparison both use it.
* ``score(...)["accuracy"]`` (the plain 2638-sample mean) is kept only as
  ``unweighted_sample_accuracy``, a secondary diagnostic.  G-bucket means are
  labelled *secondary unweighted sample accuracy diagnostic* and never called
  the CV-Bench primary.

Every input is hashed (SHA256) and stored in the report.  Nothing here writes to
the raw JSONL, the formal quality report or ``configs/evaluation_protocol.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.build_cv_bench_pruning_report import (  # noqa: E402
    answer_letter,
    parse_choice,
    score,
)
from evaluation.metrics import load_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

FORMAL_JSON = ROOT / "reports" / "cv_bench" / "2026-10-04-current-selector-cvbench.json"
ARTIFACT_DIR = ROOT / "artifacts" / "token_selector" / "2026-10-04-current-selector-cvbench"
CANDIDATE_JSONL = {
    "K48": ARTIFACT_DIR / "current_selector_k48.jsonl",
    "K64": ARTIFACT_DIR / "current_selector_k64.jsonl",
    "K96": ARTIFACT_DIR / "current_selector_k96.jsonl",
}
MANIFEST = ROOT / "datasets" / "cv_bench_full2638.jsonl"
DENSE_JSONL = (
    ROOT
    / "evaluation"
    / "results"
    / "cv_bench_l2_0875_full2638_20260916"
    / "shallow_r1000.jsonl"
)
L2_LAYER5_JSON = ROOT / "reports" / "cv_bench" / "2026-09-16-cv-bench-l2-0250-full2638.json"
SCORER_MODULE = ROOT / "evaluation" / "build_cv_bench_pruning_report.py"

DEFAULT_OUTPUT_JSON = (
    ROOT / "reports" / "cv_bench" / "2026-10-04-current-selector-failure-diagnostic.json"
)
DEFAULT_OUTPUT_MD = (
    ROOT / "reports" / "cv_bench" / "2026-10-04-current-selector-failure-diagnostic.md"
)

FORMAL_SAMPLE_COUNT = 2638
CANDIDATE_ORDER = ("K48", "K64", "K96")
FORBIDDEN_PRIMARY_METRICS = ("pointing", "grounding", "ndcg", "retained_mass")

# The one and only formal CV-Bench primary metric.  ``score`` returns both a
# plain per-sample mean (``accuracy``) and the dataset-balanced official
# CV-Bench score (``cv_bench_accuracy``); only the latter may be called the
# CV-Bench primary score.  The former is retained strictly as a secondary,
# unweighted diagnostic.
PRIMARY_METRIC = 'score(...)["cv_bench_accuracy"]'
SECONDARY_METRIC = 'score(...)["accuracy"]'
SECONDARY_METRIC_LABEL = "secondary unweighted sample accuracy diagnostic"

# The four G buckets requested for the K96 focus; applied to every candidate.
G_BUCKETS = (
    ("G<=96", 0, 96),
    ("97-256", 97, 256),
    ("257-768", 257, 768),
    (">768", 769, None),
)

CUT_POINT_CAVEAT = (
    "The frozen L2@25% baseline prunes after vision layer 5 (prune_after_layer=5) "
    "while the current Selector prunes at boundary B1 (after vision block 0). "
    "The cut points differ, so any accuracy difference between the two must NOT "
    "be attributed to the ranker/selector alone."
)


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def g_bucket(group_count: int) -> str:
    for label, low, high in G_BUCKETS:
        if group_count >= low and (high is None or group_count <= high):
            return label
    raise ValueError(f"unbucketed group count: {group_count}")


def bucket_accuracy(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Secondary metric only: plain unweighted sample accuracy of a stratum.

    This is intentionally *not* the formal CV-Bench score: the official
    ``score(...)["cv_bench_accuracy"]`` averages the 2D sources and Omni3D
    equally, so a bucket mean can never stand in for the CV-Bench primary.
    """

    if not rows:
        return {
            "sample_count": 0,
            "unweighted_sample_accuracy": None,
            "parse_success_rate": None,
            "metric": SECONDARY_METRIC_LABEL,
        }
    return {
        "sample_count": len(rows),
        "unweighted_sample_accuracy": statistics.fmean(
            bool(r["official_correct"]) for r in rows
        ),
        "parse_success_rate": statistics.fmean(
            r["official_parsed"] is not None for r in rows
        ),
        "metric": SECONDARY_METRIC_LABEL,
    }


def _bucket_by(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {name: bucket_accuracy(grouped[name]) for name in sorted(grouped)}


def official_records(
    rows: Sequence[Mapping[str, Any]],
    manifest_index: Mapping[str, Mapping[str, Any]],
    label: str,
) -> list[dict[str, Any]]:
    """Recompute the official CV-Bench choice verdict per sample.

    ``parse_choice`` / ``answer_letter`` are the frozen official scorer, so the
    returned ``official_correct`` is the formal choice-accuracy verdict, not a
    field copied from the JSONL.
    """

    records: list[dict[str, Any]] = []
    correct_key = f"{label}_correct"
    answer_key = f"{label}_answer"
    for row in rows:
        sample_id = str(row["sample_id"])
        meta = manifest_index[sample_id]
        parsed = parse_choice(str(row["output_text"]), list(meta["choices"]))
        truth = answer_letter(str(meta["answer"]))
        enriched = dict(row)
        enriched["official_parsed"] = parsed
        enriched["official_correct"] = parsed == truth
        enriched[correct_key] = parsed == truth
        enriched[answer_key] = parsed
        records.append(enriched)
    return records


def verify_identity_bypass(
    label: str,
    budget: int,
    rows: Sequence[Mapping[str, Any]],
    dense_text: Mapping[str, str],
    manifest_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Check that G <= K records are true identity bypass and match dense."""

    identity_rows = [r for r in rows if bool(r["identity_bypass"])]
    g_le_k_rows = [r for r in rows if int(r["full_llm_visual_tokens"]) <= budget]
    identity_ids = {str(r["sample_id"]) for r in identity_rows}
    g_le_k_ids = {str(r["sample_id"]) for r in g_le_k_rows}

    text_matches = 0
    answer_matches = 0
    full_sequence_matches = 0
    retention_matches = 0
    mismatches: list[dict[str, Any]] = []
    for row in identity_rows:
        sample_id = str(row["sample_id"])
        group_count = int(row["full_llm_visual_tokens"])
        dense = dense_text.get(sample_id)
        parsed_dense = None
        if dense is not None:
            parsed_dense = parse_choice(dense, list(manifest_index[sample_id]["choices"]))
        text_ok = dense is not None and str(row["output_text"]) == dense
        answer_ok = parsed_dense == row["parsed_answer"]
        sequence_ok = list(row["selected_group_indices"]) == list(range(group_count))
        retention_ok = float(row["actual_token_retention"]) == 1.0 and int(
            row["output_group_count"]
        ) == group_count
        text_matches += int(text_ok)
        answer_matches += int(answer_ok)
        full_sequence_matches += int(sequence_ok)
        retention_matches += int(retention_ok)
        if not (text_ok and answer_ok and sequence_ok and retention_ok):
            mismatches.append(
                {
                    "sample_id": sample_id,
                    "group_count": group_count,
                    "text_match": text_ok,
                    "answer_match": answer_ok,
                    "full_sequence_match": sequence_ok,
                    "retention_match": retention_ok,
                }
            )

    return {
        "budget": budget,
        "identity_bypass_count": len(identity_rows),
        "g_le_k_count": len(g_le_k_rows),
        "identity_set_equals_g_le_k": identity_ids == g_le_k_ids,
        "missing_identity_for_g_le_k": sorted(g_le_k_ids - identity_ids),
        "spurious_identity_for_g_gt_k": sorted(identity_ids - g_le_k_ids),
        "dense_text_match_count": text_matches,
        "dense_answer_match_count": answer_matches,
        "identity_full_sequence_match_count": full_sequence_matches,
        "identity_retention_match_count": retention_matches,
        "all_identity_outputs_match_dense": bool(
            len(identity_rows) == text_matches == answer_matches
            and len(identity_rows) == full_sequence_matches == retention_matches
        ),
        "mismatches": mismatches,
    }


def analyse_candidate(
    label: str,
    budget: int,
    rows: Sequence[Mapping[str, Any]],
    formal_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    # Reuse the frozen official CV-Bench scorer for the overall / task / source
    # numbers; the G strata use the same official_correct boolean.
    official = score(list(rows), f"{label}_correct")
    official_accuracy = float(official["accuracy"])

    g_bucket_stats: dict[str, dict[str, Any]] = {}
    for bucket_label, low, high in G_BUCKETS:
        subset = [
            r
            for r in rows
            if int(r["full_llm_visual_tokens"]) >= low
            and (high is None or int(r["full_llm_visual_tokens"]) <= high)
        ]
        g_bucket_stats[bucket_label] = bucket_accuracy(subset)

    task_by_g: dict[str, dict[str, dict[str, Any]]] = {}
    source_by_g: dict[str, dict[str, dict[str, Any]]] = {}
    for bucket_label, _low, _high in G_BUCKETS:
        subset = [
            r
            for r in rows
            if g_bucket(int(r["full_llm_visual_tokens"])) == bucket_label
        ]
        task_by_g[bucket_label] = _bucket_by(subset, "task") if subset else {}
        source_by_g[bucket_label] = _bucket_by(subset, "source") if subset else {}

    formal_accuracy = float(formal_candidate["candidate"]["accuracy"])
    formal_cv_bench_accuracy = float(
        formal_candidate["candidate"]["cv_bench_accuracy"]
    )
    official_cv_bench_accuracy = float(official["cv_bench_accuracy"])
    return {
        "label": label,
        "budget": budget,
        "sample_count": len(rows),
        # Formal CV-Bench primary: the dataset-balanced official score and the
        # only acceptable primary metric.
        "cv_bench_accuracy": official_cv_bench_accuracy,
        "cv_bench_accuracy_2d": float(official["accuracy_2d"]),
        "cv_bench_accuracy_3d": float(official["accuracy_3d"]),
        # Secondary diagnostic: plain per-sample mean.  Never the CV-Bench
        # primary and never used for the formal report match or L2 comparison.
        "unweighted_sample_accuracy": official_accuracy,
        "parse_success_rate": float(official["parse_success_rate"]),
        # The formal report match must compare candidate.cv_bench_accuracy to
        # the recomputed official CV-Bench score, byte-for-byte semantics.
        "formal_report_cv_bench_accuracy": formal_cv_bench_accuracy,
        "formal_report_matches_recompute": abs(
            official_cv_bench_accuracy - formal_cv_bench_accuracy
        )
        <= 1e-9,
        "formal_report_unweighted_sample_accuracy": formal_accuracy,
        "relative_retention_vs_dense": float(
            formal_candidate["relative_retention_vs_dense"]
        ),
        "unweighted_task_accuracy": official["task_accuracy"],
        "unweighted_source_accuracy": official["source_accuracy"],
        "g_buckets": g_bucket_stats,
        "task_accuracy_by_g_bucket": task_by_g,
        "source_accuracy_by_g_bucket": source_by_g,
        "identity_bucket": bucket_accuracy(
            [r for r in rows if int(r["full_llm_visual_tokens"]) <= budget]
        ),
    }


def l2_layer5_comparison(
    candidate_reports: Mapping[str, Mapping[str, Any]],
    l2_path: Path,
) -> dict[str, Any]:
    l2 = json.loads(l2_path.read_text(encoding="utf-8"))
    strategy = l2["strategy"]
    l2_scores = l2["scores"]["l2_0250"]
    dense_scores = l2["scores"]["custom_dense"]
    comparison: dict[str, Any] = {
        "l2_layer5_cv_bench_accuracy": float(l2_scores["cv_bench_accuracy"]),
        "l2_layer5_unweighted_task_accuracy": l2_scores["task_accuracy"],
        "l2_layer5_unweighted_source_accuracy": l2_scores["source_accuracy"],
        "l2_layer5_retention": float(strategy["retention"]),
        "l2_layer5_prune_after_layer": int(strategy["prune_after_layer"]),
        "l2_layer5_anchor_grid": int(strategy["anchor_grid"]),
        "l2_layer5_position_policy": strategy["position_policy"],
        "l2_layer5_dense_accuracy": float(dense_scores["cv_bench_accuracy"]),
        "comparison_metric": PRIMARY_METRIC,
        "current_selector_boundary": "B1 (after vision block 0)",
        "cut_point_differs": True,
        "attribution_caveat": CUT_POINT_CAVEAT,
        "not_attributable_to_ranker": True,
        "candidates": {},
    }
    for label, report in candidate_reports.items():
        # The comparison against the frozen L2@25% cut point must use the
        # current formal CV-Bench primary, never the unweighted sample mean.
        current = report["cv_bench_accuracy"]
        comparison["candidates"][label] = {
            "current_selector_cv_bench_accuracy": current,
            "current_minus_l2_layer5": current
            - float(l2_scores["cv_bench_accuracy"]),
        }
    return comparison


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(root: Path = ROOT) -> dict[str, Any]:
    formal_path = root / FORMAL_JSON.relative_to(ROOT)
    artifact_dir = root / ARTIFACT_DIR.relative_to(ROOT)
    candidate_paths = {
        label: root / path.relative_to(ROOT) for label, path in CANDIDATE_JSONL.items()
    }
    manifest_path = root / MANIFEST.relative_to(ROOT)
    l2_path = root / L2_LAYER5_JSON.relative_to(ROOT)
    scorer_module = root / SCORER_MODULE.relative_to(ROOT)

    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    formal_candidates = {str(c["label"]): c for c in formal["candidates"]}
    dense_jsonl_path = Path(formal["dense_jsonl"])
    if not dense_jsonl_path.is_absolute():
        dense_jsonl_path = (root / dense_jsonl_path).resolve()

    manifest_rows = load_jsonl(manifest_path)
    manifest_index = {str(row["sample_id"]): row for row in manifest_rows}
    if len(manifest_index) != FORMAL_SAMPLE_COUNT:
        raise ValueError(
            f"manifest must hold {FORMAL_SAMPLE_COUNT} unique samples, got {len(manifest_index)}"
        )
    dense_rows = load_jsonl(dense_jsonl_path)
    dense_text = {str(row["sample_id"]): str(row["output_text"]) for row in dense_rows}

    candidates: dict[str, Any] = {}
    identity_checks: dict[str, Any] = {}
    for label in CANDIDATE_ORDER:
        path = candidate_paths[label]
        rows = load_jsonl(path)
        if len(rows) != FORMAL_SAMPLE_COUNT:
            raise ValueError(f"{label}: expected {FORMAL_SAMPLE_COUNT} rows, got {len(rows)}")
        unique_ids = {str(r["sample_id"]) for r in rows}
        if len(unique_ids) != FORMAL_SAMPLE_COUNT:
            raise ValueError(f"{label}: expected {FORMAL_SAMPLE_COUNT} unique ids, got {len(unique_ids)}")
        records = official_records(rows, manifest_index, label)
        budget = int(rows[0]["requested_budget"])
        candidates[label] = analyse_candidate(
            label, budget, records, formal_candidates[label]
        )
        identity_checks[label] = verify_identity_bypass(
            label, budget, records, dense_text, manifest_index
        )

    inputs_sha256 = {
        "formal_quality_report": {
            "path": str(formal_path),
            "sha256": sha256_file(formal_path),
        },
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "dense_jsonl": {
            "path": str(dense_jsonl_path),
            "sha256": sha256_file(dense_jsonl_path),
        },
        "l2_layer5_report": {"path": str(l2_path), "sha256": sha256_file(l2_path)},
        "official_scorer_module": {
            "path": str(scorer_module),
            "sha256": sha256_file(scorer_module),
        },
    }
    for label, path in candidate_paths.items():
        inputs_sha256[f"candidate_{label}_jsonl"] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "CPU-only read-only diagnostic of the frozen 2026-10-04 current "
            "Selector CV-Bench outputs; software Selector path, no FPGA claim"
        ),
        "formal_sample_count": FORMAL_SAMPLE_COUNT,
        "primary_metric": PRIMARY_METRIC,
        "secondary_metric": SECONDARY_METRIC,
        "metric_definitions": {
            "primary": (
                "Official CV-Bench score returned by the frozen scorer as "
                "score(...)[\"cv_bench_accuracy\"] = ((ADE20K + COCO)/2 + Omni3D)/2. "
                "This is the ONE formal CV-Bench primary metric."
            ),
            "secondary": (
                "unweighted_sample_accuracy = score(...)[\"accuracy\"], the plain "
                "mean over all 2638 samples. Reported only as a secondary "
                "diagnostic and never as the CV-Bench primary."
            ),
            "g_buckets": SECONDARY_METRIC_LABEL,
        },
        "forbidden_primary_metrics": list(FORBIDDEN_PRIMARY_METRICS),
        "inputs_sha256": inputs_sha256,
        "g_bucket_definition": {
            label: {"low": low, "high": high} for label, low, high in G_BUCKETS
        },
        "g_definition": "group_count = full_llm_visual_tokens (patch tokens / spatial_merge_size**2)",
        "candidates": candidates,
        "identity_bypass_verification": identity_checks,
        "l2_layer5_comparison": l2_layer5_comparison(candidates, l2_path),
        "limitations": [
            "Software Selector path only; this is not an FPGA or latency result.",
            "The comparison against L2@25% uses a different vision cut point "
            "(layer 5 vs B1/block 0) and therefore isolates no single component.",
            "G buckets are observational strata over CV-Bench, not a controlled sweep.",
            "G-bucket means are " + SECONDARY_METRIC_LABEL + " only and must not "
            "be presented as the formal CV-Bench primary score.",
            "Forbidden primary metrics (pointing/grounding/ndcg/retained_mass) are never used.",
        ],
    }
    report["all_identity_checks_pass"] = all(
        check["all_identity_outputs_match_dense"]
        and check["identity_set_equals_g_le_k"]
        for check in identity_checks.values()
    )
    report["formal_scores_reproduced"] = all(
        candidate["formal_report_matches_recompute"]
        for candidate in candidates.values()
    )
    return report


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# 当前 Selector CV-Bench 失败诊断（K48/K64/K96）")
    lines.append("")
    lines.append(f"- 生成时间：`{report['generated_at']}`")
    lines.append(f"- 范围：{report['scope']}")
    lines.append(f"- 正式样本数：{report['formal_sample_count']}")
    lines.append(
        "- 正式主指标（唯一）：`" + report["primary_metric"] + "`"
        "（官方 CV-Bench 组合分，`parse_choice`/`answer_letter`/`score`）；"
        "禁用主指标：" + ", ".join(report["forbidden_primary_metrics"])
    )
    lines.append(
        "- 次指标（仅诊断、不得冒充主分）：`unweighted_sample_accuracy` = "
        "`" + report["secondary_metric"] + "`（2638 样本简单平均）。"
    )
    lines.append("")
    lines.append("## 1. 总体准确率")
    lines.append("")
    lines.append(
        "| 候选 | 样本 | **CV-Bench 主分**（`cv_bench_accuracy`） | "
        "unweighted 样本准确率 | 解析率 | 正式报告一致（对 `candidate.cv_bench_accuracy`） | "
        "相对 dense 保留 |"
    )
    lines.append("|---|---:|---:|---:|---:|:---:|---:|")
    for label in CANDIDATE_ORDER:
        c = report["candidates"][label]
        lines.append(
            f"| {label} | {c['sample_count']} | **{_pct(c['cv_bench_accuracy'])}** | "
            f"{_pct(c['unweighted_sample_accuracy'])} | "
            f"{_pct(c['parse_success_rate'])} | "
            f"{'✅' if c['formal_report_matches_recompute'] else '❌'} | "
            f"{_pct(c['relative_retention_vs_dense'])} |"
        )
    lines.append("")
    lines.append(
        "> **两种口径的区别**：**CV-Bench 主分** "
        "`cv_bench_accuracy = ((ADE20K + COCO)/2 + Omni3D)/2` 对 2D 两来源与 3D 等权，"
        "是唯一正式口径；`unweighted_sample_accuracy` 是 2638 题的简单样本平均，"
        "只作次要诊断，**不得写作 CV-Bench 主分**。"
    )
    lines.append("")
    lines.append("## 2. 分任务 / 分来源")
    lines.append("")
    lines.append(
        "> 以下均为 **" + SECONDARY_METRIC_LABEL + "**（单层子集内的 unweighted 简单平均），"
        "不是正式 CV-Bench 主分。"
    )
    lines.append("")
    for label in CANDIDATE_ORDER:
        c = report["candidates"][label]
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| task | unweighted_sample_accuracy |")
        lines.append("|---|---:|")
        for task, accuracy in c["unweighted_task_accuracy"].items():
            lines.append(f"| {task} | {_pct(accuracy)} |")
        lines.append("")
        lines.append("| source | unweighted_sample_accuracy |")
        lines.append("|---|---:|")
        for source, accuracy in c["unweighted_source_accuracy"].items():
            lines.append(f"| {source} | {_pct(accuracy)} |")
        lines.append("")
    lines.append("## 3. G 分桶（重点 K96）")
    lines.append("")
    lines.append(
        "> G 桶均值仅为 **" + SECONDARY_METRIC_LABEL + "**，"
        "**不得当作正式 CV-Bench 主分**。"
    )
    lines.append("")
    lines.append("| 候选 | 桶 | samples | unweighted_sample_accuracy | 解析率 |")
    lines.append("|---|---|---:|---:|---:|")
    for label in CANDIDATE_ORDER:
        c = report["candidates"][label]
        for bucket_label, _low, _high in G_BUCKETS:
            stat = c["g_buckets"][bucket_label]
            lines.append(
                f"| {label} | {bucket_label} | {stat['sample_count']} | "
                f"{_pct(stat['unweighted_sample_accuracy'])} | {_pct(stat['parse_success_rate'])} |"
            )
        lines.append(
            f"| {label} | G<=K(identity) | {c['identity_bucket']['sample_count']} | "
            f"{_pct(c['identity_bucket']['unweighted_sample_accuracy'])} | "
            f"{_pct(c['identity_bucket']['parse_success_rate'])} |"
        )
    lines.append("")
    lines.append("### K96 任务 × G 桶（secondary unweighted sample accuracy diagnostic）")
    lines.append("")
    lines.append("| G 桶 | task | samples | unweighted_sample_accuracy |")
    lines.append("|---|---|---:|---:|")
    for bucket_label, _low, _high in G_BUCKETS:
        for task, stat in report["candidates"]["K96"]["task_accuracy_by_g_bucket"][
            bucket_label
        ].items():
            lines.append(
                f"| {bucket_label} | {task} | {stat['sample_count']} | "
                f"{_pct(stat['unweighted_sample_accuracy'])} |"
            )
    lines.append("")
    lines.append("## 4. G<=K identity bypass 与 dense 一致性")
    lines.append("")
    lines.append(
        "| 候选 | identity 样本 | G<=K 样本 | 集合相等 | dense 文本一致 | 集合全保留 | 全部一致 |"
    )
    lines.append("|---|---:|---:|:---:|---:|:---:|:---:|")
    for label in CANDIDATE_ORDER:
        check = report["identity_bypass_verification"][label]
        lines.append(
            f"| {label} | {check['identity_bypass_count']} | {check['g_le_k_count']} | "
            f"{'✅' if check['identity_set_equals_g_le_k'] else '❌'} | "
            f"{check['dense_text_match_count']} | "
            f"{check['identity_full_sequence_match_count']} | "
            f"{'✅' if check['all_identity_outputs_match_dense'] else '❌'} |"
        )
    lines.append("")
    lines.append("## 5. 与已有 L2@25% after-layer-5 的对照")
    lines.append("")
    cmp = report["l2_layer5_comparison"]
    lines.append(
        f"- 对照口径：`{cmp['comparison_metric']}`（当前 Selector 的正式 CV-Bench 主分）"
    )
    lines.append(f"- L2@25% after layer {cmp['l2_layer5_prune_after_layer']}：")
    lines.append(f"  - 组合准确率 `{_pct(cmp['l2_layer5_cv_bench_accuracy'])}`，dense `{_pct(cmp['l2_layer5_dense_accuracy'])}`，保留率 `{cmp['l2_layer5_retention']}`")
    lines.append(f"  - anchor grid `{cmp['l2_layer5_anchor_grid']}`，位置策略 `{cmp['l2_layer5_position_policy']}`")
    for label in CANDIDATE_ORDER:
        item = cmp["candidates"][label]
        lines.append(
            f"- {label} − L2@25%：`{item['current_minus_l2_layer5']:+.4f}`"
            f"（{label} CV-Bench 主分 `{_pct(item['current_selector_cv_bench_accuracy'])}`）"
        )
    lines.append("")
    lines.append(f"> **归因限制**：{cmp['attribution_caveat']}")
    lines.append("")
    lines.append("## 6. 输入 SHA256")
    lines.append("")
    lines.append("| 输入 | SHA256 |")
    lines.append("|---|---|")
    for name, item in report["inputs_sha256"].items():
        lines.append(f"| {name} | `{item['sha256']}` |")
    lines.append("")
    lines.append("## 7. 限制")
    lines.append("")
    for note in report["limitations"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append(
        f"- `all_identity_checks_pass` = {report['all_identity_checks_pass']}；"
        f"`formal_scores_reproduced` = {report['formal_scores_reproduced']}"
    )
    lines.append("")
    lines.append(
        "> 本报告为 CPU-only 只读诊断：未改原始 JSONL、正式 quality report 或 "
        "`configs/evaluation_protocol.json`；不含 FPGA/时延/加速声明。"
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args(argv)

    report = build_report(args.root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "all_identity_checks_pass": report["all_identity_checks_pass"],
                "formal_scores_reproduced": report["formal_scores_reproduced"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
