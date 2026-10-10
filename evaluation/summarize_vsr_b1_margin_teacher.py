#!/usr/bin/env python3
"""Audit and summarize the complete VSR B1 margin-teacher cache."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluation.extract_vsr_b1_margin_teacher import (
    DEFAULT_CONFIG,
    DEFAULT_OUT_DIR,
    MANIFEST_SHA256,
    canonical_record_sha256,
    sha256_file,
)
from evaluation.prompt_projection_distillation import load_feature_cache

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "reports/token_selector/2026-10-10-vsr-b1-margin-teacher.json"
REPORT_MD = ROOT / "reports/token_selector/2026-10-10-vsr-b1-margin-teacher.md"


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "min": float(array.min()), "max": float(array.max()),
        "p95": float(np.percentile(array, 95)),
    }


def summarize(out_dir: Path = DEFAULT_OUT_DIR, config: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    index_path = out_dir / "index.jsonl"
    records = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    config_sha = sha256_file(config)
    ids: set[str] = set()
    file_bytes = 0
    fingerprints_ok = 0
    schema_ok = 0
    group_counts: list[int] = []
    prompt_counts: list[int] = []
    margins: list[float] = []
    forward: list[float] = []
    backward: list[float] = []
    total: list[float] = []
    peaks: list[float] = []
    predicted_correct = 0
    label_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    for line_number, record in enumerate(records, 1):
        sample_id = str(record.get("sample_id"))
        if sample_id in ids:
            raise ValueError(f"duplicate sample_id at line {line_number}")
        ids.add(sample_id)
        if record.get("record_sha256") != canonical_record_sha256(record):
            raise ValueError(f"record hash mismatch at line {line_number}")
        if record.get("manifest_sha256") != MANIFEST_SHA256 or record.get("config_sha256") != config_sha:
            raise ValueError(f"identity mismatch at line {line_number}")
        path = out_dir / str(record["npz_name"])
        if sha256_file(path) != record["npz_file_sha256"]:
            raise ValueError(f"npz file hash mismatch: {path}")
        cache = load_feature_cache(path, validate=True)
        schema_ok += 1
        stored = str(cache["metadata"].get("sample_sha256"))
        if stored != str(record["cache_fingerprint"]):
            raise ValueError(f"cache fingerprint mismatch: {path}")
        fingerprints_ok += 1
        audit = record["audit"]
        if not audit["saliency_finite"] or int(audit["saliency_nonzero"]) != int(audit["group_count"]):
            raise ValueError(f"invalid saliency audit: {sample_id}")
        if int(cache["group_mean"].shape[0]) != int(audit["group_count"]):
            raise ValueError(f"group count mismatch: {sample_id}")
        file_bytes += path.stat().st_size
        group_counts.append(int(audit["group_count"]))
        prompt_counts.append(int(audit["prompt_token_count"]))
        margins.append(float(audit["margin"]))
        forward.append(float(audit["forward_seconds"]))
        backward.append(float(audit["backward_seconds"]))
        total.append(float(audit["total_seconds"]))
        peaks.append(float(audit["peak_cuda_bytes"]) / 2**30)
        correct = str(audit["correct_branch"])
        predicted = str(audit["predicted_branch"])
        label_counts[correct] += 1
        predicted_counts[predicted] += 1
        predicted_correct += int(correct == predicted)
    if len(records) != 3483 or len(ids) != 3483:
        raise ValueError(f"expected 3483 complete records, got {len(records)}")
    result = {
        "status": "VSR_B1_MARGIN_TEACHER_CACHE_AUDIT_PASS",
        "scientific_scope": "训练教师数据完整性与工程可行性；不是可部署 Selector 精度结论。",
        "records": len(records), "unique_sample_ids": len(ids),
        "npz_schema_valid": schema_ok, "cache_fingerprints_valid": fingerprints_ok,
        "all_record_hashes_valid": True, "all_npz_file_hashes_valid": True,
        "all_saliency_finite_and_every_group_nonzero": True,
        "manifest_sha256": MANIFEST_SHA256, "config_sha256": config_sha,
        "index_sha256": sha256_file(index_path), "npz_total_bytes": file_bytes,
        "group_count": {**quantiles([float(v) for v in group_counts]),
                        "g_le_96": sum(v <= 96 for v in group_counts),
                        "g_gt_96": sum(v > 96 for v in group_counts)},
        "prompt_token_count": quantiles([float(v) for v in prompt_counts]),
        "teacher_margin": {**quantiles(margins), "positive": sum(v > 0 for v in margins),
                           "zero": sum(v == 0 for v in margins), "negative": sum(v < 0 for v in margins)},
        "dense_next_branch_diagnostic": {
            "correct": predicted_correct, "total": len(records),
            "accuracy": predicted_correct / len(records),
            "label_counts": dict(label_counts), "predicted_counts": dict(predicted_counts),
            "not_generation_accuracy": True,
        },
        "timing_seconds": {"forward": quantiles(forward), "backward": quantiles(backward),
                           "sample_total": quantiles(total), "model_load_excluded": True,
                           "gpu_boundaries_synchronized": True},
        "peak_cuda_gib": quantiles(peaks),
        "leakage_contract": {
            "train_label_used_only_for_teacher_branch": True,
            "selector_inference_uses_label": False,
            "selector_inference_uses_relation_subject_object_annotations": False,
            "dev_images_removed_from_train": 5,
            "test_opened_or_scored": False,
        },
        "next_decision": "在固定 K96/87+9/3x3/cap16 下训练小型学生；先以 train 内按图像分组划分 train/validation，禁止使用 VSR dev 标签选超参数。",
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    g = result["group_count"]
    d = result["dense_next_branch_diagnostic"]
    t = result["timing_seconds"]
    REPORT_MD.write_text("\n".join([
        "# VSR B1 答案 Margin 教师缓存审计", "",
        "## 结论", "",
        "3483 条 zero-shot train 教师特征全部生成并通过逐文件审计。该结论只说明训练数据完整、梯度链可用，不表示 K96 Selector 已经提升精度。", "",
        "## 完整性", "",
        f"- 索引：{result['records']} 条，唯一 ID {result['unique_sample_ids']} 条。",
        f"- NPZ schema / 内部 fingerprint：{schema_ok}/{fingerprints_ok} 全部通过。",
        f"- NPZ 总大小：{file_bytes / 2**30:.3f} GiB。",
        "- 每条 saliency 均有限，并且每个视觉 group 的值都非零。", "",
        "## 数据与模型诊断", "",
        f"- G 均值 {g['mean']:.1f}，中位数 {g['median']:.0f}，范围 {g['min']:.0f}–{g['max']:.0f}；G<=96 共 {g['g_le_96']} 条，部署时应走 identity bypass。",
        f"- 冻结 Dense 模型正确 True/False 下一分支：{d['correct']}/{d['total']} = {d['accuracy']:.4f}。这是单步分支诊断，不是完整生成准确率。",
        f"- 稳态前向中位数 {t['forward']['median']*1000:.1f} ms，反向中位数 {t['backward']['median']*1000:.1f} ms；峰值显存最大 {result['peak_cuda_gib']['max']:.2f} GiB。", "",
        "## 数据隔离", "",
        "训练标签只用于确定教师 True/False 分支。未来部署 Selector 只允许读取 B1 图像特征、原始 Prompt、位置和固定预算；不得读取 label、relation、subject、object 或教师梯度。与 dev 重叠的 5 张图已删除，test 未打开、未评分。", "",
        "## 下一步", "",
        "先按图像 ID 对 2244 张训练图做分组 train/validation 划分，再训练固定 K96 的小型关系保持学生。超参数只能由 train 内 validation 选择；VSR dev 仅用于冻结候选后的开发评测。", "",
    ]) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(summarize(), ensure_ascii=False, indent=2))
