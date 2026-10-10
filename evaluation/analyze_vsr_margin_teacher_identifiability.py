#!/usr/bin/env python3
"""Measure whether the VSR margin teacher changes across prompts on one image."""

from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluation.prompt_projection_distillation import load_feature_cache, spearman, topk_indices

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts/token_selector/2026-10-10-vsr-b1-margin-teacher"
MANIFEST = ROOT / "datasets/external/vsr-zeroshot-b27a0af0/vsr_zeroshot_train_no_dev_image_overlap.jsonl"
REPORT_JSON = ROOT / "reports/token_selector/2026-10-10-vsr-margin-teacher-identifiability.json"
REPORT_MD = ROOT / "reports/token_selector/2026-10-10-vsr-margin-teacher-identifiability.md"


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"count": int(len(array)), "mean": float(array.mean()), "median": float(np.median(array)),
            "p05": float(np.percentile(array, 5)), "p95": float(np.percentile(array, 95)),
            "min": float(array.min()), "max": float(array.max())}


def main() -> int:
    source = {row["sample_id"]: row for row in (json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip())}
    index = [json.loads(line) for line in (CACHE / "student_index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in index:
        cache = load_feature_cache(CACHE / record["npz"], validate=True)
        sample = source[record["sample_id"]]
        by_image[record["source_image_sha256"]].append({
            "sample_id": record["sample_id"], "relation": sample["relation"], "label": int(sample["official_label"]),
            "scores": cache["teacher_scores"].astype(np.float64),
        })
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    pair_count = 0
    for rows in by_image.values():
        for left, right in itertools.combinations(rows, 2):
            if len(left["scores"]) != len(right["scores"]):
                raise ValueError("same image produced different visual group counts")
            pair_count += 1
            groups = len(left["scores"]); k = min(96, groups)
            top_left = set(int(x) for x in topk_indices(left["scores"], k))
            top_right = set(int(x) for x in topk_indices(right["scores"], k))
            jaccard = len(top_left & top_right) / len(top_left | top_right)
            rho = float(spearman(left["scores"], right["scores"]))
            left_norm = (left["scores"] - left["scores"].min()) / max(float(np.ptp(left["scores"])), 1e-12)
            right_norm = (right["scores"] - right["scores"].min()) / max(float(np.ptp(right["scores"])), 1e-12)
            distance = float(np.sqrt(np.mean((left_norm - right_norm) ** 2)))
            names = ["all_pairs"]
            names.append("different_relation" if left["relation"] != right["relation"] else "same_relation")
            names.append("different_label" if left["label"] != right["label"] else "same_label")
            for name in names:
                buckets[name]["spearman"].append(rho); buckets[name]["top96_jaccard"].append(jaccard); buckets[name]["normalized_rms_difference"].append(distance)
    results = {name: {metric: summary(values) for metric, values in metrics.items()} for name, metrics in sorted(buckets.items())}
    all_pairs = results["all_pairs"]
    conclusion = "TEACHER_MOSTLY_PROMPT_INVARIANT" if all_pairs["spearman"]["median"] >= 0.9 and all_pairs["top96_jaccard"]["median"] >= 0.8 else "TEACHER_HAS_MEANINGFUL_PROMPT_VARIATION"
    report = {"status": conclusion, "records": len(index), "images": len(by_image),
              "images_with_multiple_prompts": sum(len(v) > 1 for v in by_image.values()), "same_image_pairs": pair_count,
              "metrics": results,
              "interpretation": "High same-image cross-prompt similarity means absolute margin gradient saliency is mainly image-driven and is a weak sole target for a prompt-conditioned selector.",
              "next_decision": "If mostly invariant, retain it only as an image-saliency term and add an explicit relation/position contrast target derived from prompt-conditioned logit differences without inference-time labels."}
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_MD.write_text("\n".join([
        "# VSR Margin 教师 Prompt 可辨识性", "", f"状态：`{conclusion}`。", "",
        f"共 {len(index)} 条、{len(by_image)} 张图，其中 {report['images_with_multiple_prompts']} 张图有多条陈述，形成 {pair_count} 个同图 Prompt 对。", "",
        f"同图不同 Prompt 的教师 saliency：Spearman 中位数 {all_pairs['spearman']['median']:.4f}，Top-96 Jaccard 中位数 {all_pairs['top96_jaccard']['median']:.4f}，归一化 RMS 差中位数 {all_pairs['normalized_rms_difference']['median']:.4f}。", "",
        "如果相关性和 Top-96 重叠都很高，说明当前绝对梯度教师主要是图像显著性目标；它可以继续作为稳定项，但不足以单独监督 Prompt 条件化关系选择。下一版应增加显式关系/位置对比目标，同时保持推理端不读取 label、relation、subject、object 注释。", "",
    ]) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
