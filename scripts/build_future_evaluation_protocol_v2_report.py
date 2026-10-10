#!/usr/bin/env python3
"""Build the future-only evaluation-protocol-v2 Chinese report.

Read-only over existing frozen JSONLs: it recomputes the four-arm honest parse
counts with the frozen ``parse_choice`` parser and the v2 metric, recomputes the
four primary CV-Bench scores to prove they are unchanged, verifies the protocol
pins, and writes only the two protocol-v2 report files.  It never rewrites a
frozen report and never probes or claims a real default output as missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.build_cv_bench_pruning_report import (  # noqa: E402
    answer_letter,
    parse_choice,
    score as frozen_score,
)
from evaluation.cv_bench_metric_v2 import (  # noqa: E402
    LIFECYCLE_STATES,
    compute_cv_bench_metric,
)
from evaluation.metrics import load_jsonl  # noqa: E402
from scripts.build_model_identity_manifest import sha256_file  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "evaluation_protocol_v2.json"
MODEL_MANIFEST_PATH = ROOT / "artifacts" / "provenance" / "robobrain25-4b-model-identity.json"
MANIFEST_PATH = ROOT / "datasets" / "cv_bench_full2638.jsonl"

REPORT_JSON = ROOT / "reports" / "token_selector" / "2026-10-06-future-evaluation-protocol-v2.json"
REPORT_MD = ROOT / "reports" / "token_selector" / "2026-10-06-future-evaluation-protocol-v2.md"

EXPECTED_COUNT = 2638

#: arm label -> (path, expected official cv_bench_accuracy, role)
ARMS: dict[str, tuple[Path, float, str]] = {
    "dense": (
        ROOT / "evaluation" / "results" / "cv_bench_l2_0875_full2638_20260916" / "shallow_r1000.jsonl",
        0.8727298774444869,
        "custom dense BF16",
    ),
    "corrected_l2_integer": (
        ROOT
        / "artifacts"
        / "token_selector"
        / "2026-10-04-b1-corrected-l2-prompt-ablation-full2638"
        / "corrected_l2_integer_k96_full2638.jsonl",
        0.7668517264725796,
        "corrected integer-L2 K96 control",
    ),
    "prompt_plus_corrected_l2_integer": (
        ROOT
        / "artifacts"
        / "token_selector"
        / "2026-10-04-b1-corrected-l2-prompt-ablation-full2638"
        / "prompt_plus_corrected_l2_integer_k96_full2638.jsonl",
        0.7773617325561999,
        "descriptive strongest prior same-budget comparator",
    ),
    "v2": (
        ROOT
        / "artifacts"
        / "token_selector"
        / "2026-10-06-conservative-l2-guarded-reranker-v2-full2638"
        / "conservative_l2_guarded_rerank_v2_k96_full2638.jsonl",
        0.7837404943432142,
        "conservative_l2_guarded_rerank_v2 development arm",
    ),
}


class ReportError(RuntimeError):
    """Raised when the report contract is violated (fail closed)."""


def _records_for(manifest: Mapping[str, Mapping[str, Any]], arm_path: Path) -> list[dict[str, Any]]:
    outputs = {str(row["sample_id"]): row for row in load_jsonl(arm_path)}
    if len(outputs) != EXPECTED_COUNT:
        raise ReportError(f"{arm_path}: {len(outputs)} unique sample ids != {EXPECTED_COUNT}")
    records: list[dict[str, Any]] = []
    for sample_id, meta in manifest.items():
        output = outputs.get(str(sample_id))
        if output is None:
            raise ReportError(f"{arm_path}: missing sample_id {sample_id}")
        parsed = parse_choice(str(output["output_text"]), list(meta["choices"]))
        truth = answer_letter(str(meta["answer"]))
        records.append(
            {
                "sample_id": sample_id,
                "source": meta["source"],
                "task": meta["task"],
                "correct": parsed == truth,
                "parsed_answer": parsed,
            }
        )
    return records


def build_report() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest_rows = load_jsonl(MANIFEST_PATH)
    if len(manifest_rows) != EXPECTED_COUNT:
        raise ReportError(f"manifest has {len(manifest_rows)} rows != {EXPECTED_COUNT}")
    manifest = {str(row["sample_id"]): row for row in manifest_rows}
    if len(manifest) != EXPECTED_COUNT:
        raise ReportError("manifest has duplicate sample ids")

    # Verify the v2 pins before reporting on them.
    pins_verified: dict[str, Any] = {}
    for name, entry in config["pins"].items():
        path = ROOT / entry["path"]
        observed = sha256_file(path)
        pins_verified[name] = {
            "path": entry["path"],
            "expected_sha256": entry["sha256"],
            "observed_sha256": observed,
            "match": observed == entry["sha256"],
        }
        if observed != entry["sha256"]:
            raise ReportError(f"pin mismatch for {name}: {observed} != {entry['sha256']}")

    honest: dict[str, Any] = {}
    scores: dict[str, Any] = {}
    for arm, (path, expected, role) in ARMS.items():
        records = _records_for(manifest, path)
        metric = compute_cv_bench_metric(
            records,
            correct_key="correct",
            parsed_answer_key="parsed_answer",
            expected_count=EXPECTED_COUNT,
        )
        observed = float(metric["cv_bench_accuracy"])
        if abs(observed - expected) > 1e-15:
            raise ReportError(
                f"{arm}: recomputed cv_bench_accuracy {observed} != frozen {expected}"
            )
        frozen_reported = float(frozen_score(records, "correct")["parse_success_rate"])
        honest[arm] = {
            "role": role,
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "sample_count": int(metric["sample_count"]),
            "parse_success_count": int(metric["parse_success_count"]),
            "parse_failure_count": int(metric["parse_failure_count"]),
            "parse_success_rate": float(metric["parse_success_rate"]),
            "frozen_reported_parse_success_rate": frozen_reported,
        }
        scores[arm] = {
            "cv_bench_accuracy": observed,
            "expected_frozen_cv_bench_accuracy": expected,
            "unchanged": abs(observed - expected) <= 1e-15,
            "accuracy_2d": float(metric["accuracy_2d"]),
            "accuracy_3d": float(metric["accuracy_3d"]),
        }

    return {
        "date": "2026-10-06",
        "schema_version": "1.0",
        "marker": "PROTOCOL_V2_READY",
        "title": "未来评测协议 v2：生命周期安全 + 诚实解析门 + 模型身份固定",
        "scope": {
            "future_only": True,
            "retroactive": False,
            "gpu_runs": 0,
            "model_weights_loaded": False,
            "network_used": False,
            "training": False,
            "dependency_installation": False,
        },
        "protocol": {
            "config_path": str(CONFIG_PATH.relative_to(ROOT)),
            "config_sha256": sha256_file(CONFIG_PATH),
            "pins_verified": pins_verified,
            "all_pins_match": all(item["match"] for item in pins_verified.values()),
            "lifecycle_states": list(LIFECYCLE_STATES),
            "measurement_classes": sorted(config["measurement_classes"]),
        },
        "findings": [
            {
                "id": "parse_gate_key_alias",
                "severity": "protocol_defect_fixed_for_future_only",
                "summary": (
                    "冻结 scorer 用 correct_key.replace('_correct','_answer') 推导解析键；"
                    "当 correct_key='correct' 时该推导与布尔正确性字段同名，使解析门恒为 1.0，"
                    "而诚实失败为 dense=40、修正整数 L2=42、prompt=42、v2=43。"
                ),
                "affects_frozen_scores": False,
            },
            {
                "id": "model_weight_shards_unpinned",
                "severity": "provenance_gap_fixed_for_future_only",
                "summary": (
                    "冻结 v2 预注册固定了代码/checkpoint/数据，但只记录 model_path/model_label，"
                    "未固定本地模型权重分片；新增模型身份清单以流式 SHA256 记录全部分片与配置文件。"
                ),
                "affects_frozen_scores": False,
            },
        ],
        "honest_parse_counts": honest,
        "primary_scores": scores,
        "primary_scores_unchanged": all(item["unchanged"] for item in scores.values()),
        "development_decision_unchanged": "DEVELOPMENT_STOP",
        "model_identity_manifest": {
            "path": str(MODEL_MANIFEST_PATH.relative_to(ROOT)),
            "sha256": sha256_file(MODEL_MANIFEST_PATH),
            "model_dir": json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))["model_dir"],
            "file_count": json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))["file_count"],
        },
        "limitations": [
            "本报告为未来协议，不追溯修改任何冻结 v2 主分或 DEVELOPMENT_STOP 判定。",
            "诚实解析计数为只读重算，不改变官方主指标，也不构成新的精度主张。",
            "本轮 CPU-only：gpu_runs=0、未加载模型权重、未联网、未训练、未安装依赖。",
            "模型身份清单只是本地文件哈希，不是模型质量/精度/部署主张。",
        ],
        "next_allowed_action": (
            "在新协议下独立预注册新候选，按生命周期安全路径规则执行，并固定模型身份清单；"
            "在任何外部验证或硬件主张前先完成只读审计。"
        ),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# 未来评测协议 v2（生命周期安全 / 诚实解析门 / 模型身份固定）")
    lines.append("")
    lines.append(f"- 日期：{report['date']}")
    lines.append(f"- 标记：`{report['marker']}`")
    lines.append("- 范围：**仅面向未来**；不追溯修改任何冻结产物。")
    lines.append(
        f"- 口径：CPU-only、只读；`gpu_runs={report['scope']['gpu_runs']}`、"
        "未加载模型权重、未联网、未训练、未安装依赖。"
    )
    lines.append("")
    lines.append("## 1. 两个协议缺口")
    lines.append("")
    for finding in report["findings"]:
        lines.append(f"### `{finding['id']}`（{finding['severity']}）")
        lines.append("")
        lines.append(finding["summary"])
        lines.append("")
    lines.append("## 2. 四臂诚实解析计数（只读重算）")
    lines.append("")
    lines.append("| 臂 | 样本 | 解析成功 | 解析失败 | 诚实解析率 | 冻结 scorer 报告值 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for arm, row in report["honest_parse_counts"].items():
        lines.append(
            f"| {arm} | {row['sample_count']} | {row['parse_success_count']} | "
            f"{row['parse_failure_count']} | {row['parse_success_rate']:.10f} | "
            f"{row['frozen_reported_parse_success_rate']:.10f} |"
        )
    lines.append("")
    lines.append(
        "> 冻结 scorer 报告值恒为 1.0，是键别名缺陷；诚实失败为 dense=40、"
        "修正整数 L2=42、prompt=42、v2=43。"
    )
    lines.append("")
    lines.append("## 3. 主指标只读重算（未改变）")
    lines.append("")
    lines.append("| 臂 | cv_bench_accuracy | 冻结值 | 未改变 |")
    lines.append("| --- | ---: | ---: | --- |")
    for arm, row in report["primary_scores"].items():
        lines.append(
            f"| {arm} | {row['cv_bench_accuracy']:.16f} | "
            f"{row['expected_frozen_cv_bench_accuracy']:.16f} | "
            f"{'是' if row['unchanged'] else '否'} |"
        )
    lines.append("")
    lines.append(
        f"- 主指标全部未改变：`{report['primary_scores_unchanged']}`；"
        f"发展判定保持 **`{report['development_decision_unchanged']}`**。"
    )
    lines.append("- 本报告不追溯修改 v2 主分，也不改变 `DEVELOPMENT_STOP`。")
    lines.append("")
    lines.append("## 4. 协议固定与生命周期")
    lines.append("")
    protocol = report["protocol"]
    lines.append(f"- 协议 config：`{protocol['config_path']}`（`{protocol['config_sha256']}`）")
    for name, item in protocol["pins_verified"].items():
        lines.append(
            f"- pin `{name}`：`{item['path']}` 匹配 `{item['match']}`"
        )
    lines.append(f"- 生命周期状态：{', '.join('`' + s + '`' for s in protocol['lifecycle_states'])}")
    lines.append(f"- 测量类别：{', '.join('`' + s + '`' for s in protocol['measurement_classes'])}")
    lines.append("")
    lines.append("## 5. 模型身份清单")
    lines.append("")
    identity = report["model_identity_manifest"]
    lines.append(f"- 路径：`{identity['path']}`（`{identity['sha256']}`）")
    lines.append(f"- 模型目录：`{identity['model_dir']}`；文件数：{identity['file_count']}")
    lines.append("")
    lines.append("## 6. 限制")
    lines.append("")
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    lines.append("## 7. 下一步允许的动作")
    lines.append("")
    lines.append(report["next_allowed_action"])
    lines.append("")
    return "\n".join(lines) + "\n"


def write_report(report: Mapping[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    args = parser.parse_args(argv)
    report = build_report()
    write_report(report, args.report_json, args.report_md)
    print(
        json.dumps(
            {
                "marker": report["marker"],
                "report_json": str(args.report_json),
                "primary_scores_unchanged": report["primary_scores_unchanged"],
                "honest_parse_failures": {
                    arm: row["parse_failure_count"]
                    for arm, row in report["honest_parse_counts"].items()
                },
                "gpu_runs": report["scope"]["gpu_runs"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReportError, OSError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
