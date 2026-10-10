#!/usr/bin/env python3
"""Cross-validate a cheap, deployable-feature ROI resolution policy."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.analyze_roi_resolution_oracle import (
    load_deterministic_strategy,
    selection_utility,
    summarize_policy,
)
from evaluation.build_global_roi_gate_report import bootstrap_delta
from evaluation.metrics import load_jsonl, score_record


def grouped_folds(image_ids: list[int], folds: int, seed: int) -> dict[int, int]:
    unique = sorted(set(image_ids))
    random.Random(seed).shuffle(unique)
    return {image_id: index % folds for index, image_id in enumerate(unique)}


def resolve_image(manifest: Path, sample: dict[str, Any], original_dir: Path) -> Path:
    image = Path(str(sample["image"]))
    candidates = [manifest.parent / image, original_dir / image.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"image not found for {sample['sample_id']}: {candidates}")


def cheap_image_features(path: Path) -> tuple[np.ndarray, list[str]]:
    with Image.open(path) as opened:
        width, height = opened.size
        image = opened.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float64) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    gradient = np.concatenate((gx.ravel(), gy.ravel()))

    values: list[float] = [
        math.log1p(width),
        math.log1p(height),
        width / height,
        math.log1p(width * height),
    ]
    names = ["log_width", "log_height", "aspect", "log_pixels"]
    for channel, name in enumerate(("r", "g", "b")):
        values.extend((float(rgb[:, :, channel].mean()), float(rgb[:, :, channel].std())))
        names.extend((f"{name}_mean", f"{name}_std"))
        histogram, _ = np.histogram(rgb[:, :, channel], bins=8, range=(0.0, 1.0), density=True)
        values.extend(float(value) for value in histogram)
        names.extend(f"{name}_hist_{index}" for index in range(8))
    quantiles = np.quantile(gray, (0.1, 0.25, 0.5, 0.75, 0.9))
    values.extend(float(value) for value in quantiles)
    names.extend(("gray_q10", "gray_q25", "gray_q50", "gray_q75", "gray_q90"))
    values.extend((float(gradient.mean()), float(gradient.std())))
    names.extend(("gradient_mean", "gradient_std"))
    for threshold in (0.05, 0.1, 0.2):
        values.append(float((gradient >= threshold).mean()))
        names.append(f"edge_fraction_{threshold}")
    for row in range(4):
        for column in range(4):
            cell = gray[row * 16 : (row + 1) * 16, column * 16 : (column + 1) * 16]
            values.extend((float(cell.mean()), float(cell.std())))
            names.extend((f"grid_{row}_{column}_mean", f"grid_{row}_{column}_std"))
    return np.asarray(values, dtype=np.float64), names


def cheap_roi_features(
    path: Path, pixel_crop_box: list[int] | tuple[int, int, int, int]
) -> tuple[np.ndarray, list[str]]:
    with Image.open(path) as opened:
        source_width, source_height = opened.size
        crop_box = tuple(int(value) for value in pixel_crop_box)
        crop = opened.convert("RGB").crop(crop_box)
        crop_width, crop_height = crop.size
        resized = crop.resize((64, 64), Image.Resampling.BILINEAR)

    # Reuse the exact pixel-statistic contract through an in-memory implementation.
    rgb = np.asarray(resized, dtype=np.float64) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    gradient = np.concatenate((gx.ravel(), gy.ravel()))
    x1, y1, x2, y2 = crop_box
    values: list[float] = [
        math.log1p(crop_width),
        math.log1p(crop_height),
        crop_width / crop_height,
        math.log1p(crop_width * crop_height),
        x1 / source_width,
        y1 / source_height,
        x2 / source_width,
        y2 / source_height,
        (crop_width * crop_height) / (source_width * source_height),
    ]
    names = [
        "log_crop_width",
        "log_crop_height",
        "crop_aspect",
        "log_crop_pixels",
        "crop_x1_fraction",
        "crop_y1_fraction",
        "crop_x2_fraction",
        "crop_y2_fraction",
        "crop_area_fraction",
    ]
    for channel, name in enumerate(("r", "g", "b")):
        values.extend((float(rgb[:, :, channel].mean()), float(rgb[:, :, channel].std())))
        names.extend((f"{name}_mean", f"{name}_std"))
        histogram, _ = np.histogram(rgb[:, :, channel], bins=8, range=(0.0, 1.0), density=True)
        values.extend(float(value) for value in histogram)
        names.extend(f"{name}_hist_{index}" for index in range(8))
    values.extend(float(value) for value in np.quantile(gray, (0.1, 0.25, 0.5, 0.75, 0.9)))
    names.extend(("gray_q10", "gray_q25", "gray_q50", "gray_q75", "gray_q90"))
    values.extend((float(gradient.mean()), float(gradient.std())))
    names.extend(("gradient_mean", "gradient_std"))
    for threshold in (0.05, 0.1, 0.2):
        values.append(float((gradient >= threshold).mean()))
        names.append(f"edge_fraction_{threshold}")
    for row in range(4):
        for column in range(4):
            cell = gray[row * 16 : (row + 1) * 16, column * 16 : (column + 1) * 16]
            values.extend((float(cell.mean()), float(cell.std())))
            names.extend((f"grid_{row}_{column}_mean", f"grid_{row}_{column}_std"))
    return np.asarray(values, dtype=np.float64), names


def ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train = (train_x - mean) / scale
    test = (test_x - mean) / scale
    train = np.column_stack((np.ones(len(train)), train))
    test = np.column_stack((np.ones(len(test)), test))
    penalty = np.eye(train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(train.T @ train + penalty, train.T @ train_y)
    return test @ weights


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    positive = labels > 0
    negative = ~positive
    tpr = float((predictions[positive] > 0).mean()) if positive.any() else 0.0
    tnr = float((predictions[negative] <= 0).mean()) if negative.any() else 0.0
    return (tpr + tnr) / 2.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--original-image-dir", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--low-results", type=Path, required=True)
    parser.add_argument("--low-strategy", default="roi_only")
    parser.add_argument("--high-results", type=Path, required=True)
    parser.add_argument("--high-strategy", required=True)
    parser.add_argument("--expected-repeats", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--feature-region", choices=("full", "roi_crop"), default="full")
    parser.add_argument(
        "--vision-prefix-features",
        type=Path,
        help="Use precomputed ROI vision-prefix vectors instead of pixel statistics.",
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    samples = load_jsonl(args.manifest)
    baseline = load_deterministic_strategy(
        args.baseline_results, "dense", args.expected_repeats
    )
    low = load_deterministic_strategy(args.low_results, args.low_strategy, args.expected_repeats)
    high = load_deterministic_strategy(args.high_results, args.high_strategy, args.expected_repeats)
    common = set(baseline) & set(low) & set(high)
    samples = [sample for sample in samples if str(sample["sample_id"]) in common]
    categories = sorted({str(sample["category"]) for sample in samples})
    category_index = {category: index for index, category in enumerate(categories)}
    image_cache: dict[tuple[Any, ...], np.ndarray] = {}
    image_feature_names: list[str] = []
    vision_prefix_records: dict[Any, dict[str, Any]] = {}
    vision_features_by_sample = False
    if args.vision_prefix_features:
        prefix_rows = load_jsonl(args.vision_prefix_features)
        vision_features_by_sample = bool(prefix_rows and prefix_rows[0].get("sample_id"))
        if vision_features_by_sample:
            vision_prefix_records = {str(record["sample_id"]): record for record in prefix_rows}
            expected_keys = {str(sample["sample_id"]) for sample in samples}
        else:
            vision_prefix_records = {int(record["coco_image_id"]): record for record in prefix_rows}
            expected_keys = {int(sample["coco_image_id"]) for sample in samples}
        if set(vision_prefix_records) != expected_keys:
            raise ValueError("vision-prefix feature IDs do not match the action matrix")
        first_prefix = next(iter(vision_prefix_records.values()))
        image_feature_names = [str(name) for name in first_prefix["feature_names"]]
        for feature_id, record in vision_prefix_records.items():
            if [str(name) for name in record["feature_names"]] != image_feature_names:
                raise ValueError("vision-prefix feature names are inconsistent")
            image_cache[(feature_id,)] = np.asarray(record["feature_vector"], dtype=np.float64)
    else:
        for sample in samples:
            image_id = int(sample["coco_image_id"])
            sample_id = str(sample["sample_id"])
            if args.feature_region == "roi_crop":
                crop_box = tuple(int(value) for value in low[sample_id]["pixel_crop_box"])
                cache_key: tuple[Any, ...] = (image_id, *crop_box)
                if cache_key not in image_cache:
                    image_cache[cache_key], image_feature_names = cheap_roi_features(
                        resolve_image(args.manifest, sample, args.original_image_dir), crop_box
                    )
            else:
                cache_key = (image_id,)
                if cache_key not in image_cache:
                    image_cache[cache_key], image_feature_names = cheap_image_features(
                        resolve_image(args.manifest, sample, args.original_image_dir)
                    )

    fold_by_image = grouped_folds(
        [int(sample["coco_image_id"]) for sample in samples], args.folds, args.seed
    )
    analyses: dict[str, Any] = {}
    detail_rows: list[dict[str, Any]] = []
    all_gates_pass = True
    for task_index, task in enumerate(("grounding", "pointing")):
        task_samples = [sample for sample in samples if sample["task"] == task]
        x_rows: list[np.ndarray] = []
        gains: list[float] = []
        for sample in task_samples:
            sample_id = str(sample["sample_id"])
            category = np.zeros(len(categories), dtype=np.float64)
            category[category_index[str(sample["category"])]] = 1.0
            image_id = int(sample["coco_image_id"])
            if args.vision_prefix_features:
                feature_key = (sample_id if vision_features_by_sample else image_id,)
            elif args.feature_region == "roi_crop":
                crop_box = tuple(int(value) for value in low[sample_id]["pixel_crop_box"])
                feature_key: tuple[Any, ...] = (image_id, *crop_box)
            else:
                feature_key = (image_id,)
            x_rows.append(np.concatenate((image_cache[feature_key], category)))
            low_score = score_record(sample, low[sample_id])
            high_score = score_record(sample, high[sample_id])
            gains.append(selection_utility(task, high_score) - selection_utility(task, low_score))
        x = np.stack(x_rows)
        y = np.asarray(gains)
        predicted = np.zeros(len(task_samples), dtype=np.float64)
        folds = np.asarray([fold_by_image[int(sample["coco_image_id"])] for sample in task_samples])
        for fold in range(args.folds):
            train = folds != fold
            test = folds == fold
            predicted[test] = ridge_predict(x[train], y[train], x[test], args.ridge_alpha)
        choices = {
            str(sample["sample_id"]): "high" if prediction > 0 else "low"
            for sample, prediction in zip(task_samples, predicted)
        }
        policy = summarize_policy(
            task_samples, baseline, low, high, choices, active_escalation=False
        )
        if args.vision_prefix_features:
            completed_blocks = int(
                next(iter(vision_prefix_records.values()))["run"]["completed_vision_blocks"]
            )
            total_blocks = 24
            work = 0
            dense_work = 0
            for sample in task_samples:
                sample_id = str(sample["sample_id"])
                low_tokens = int(low[sample_id]["final_context_visual_tokens"])
                high_tokens = int(high[sample_id]["final_context_visual_tokens"])
                dense_tokens = int(baseline[sample_id]["final_context_visual_tokens"])
                work += (
                    low_tokens * total_blocks
                    if choices[sample_id] == "low"
                    else low_tokens * completed_blocks + high_tokens * total_blocks
                )
                dense_work += dense_tokens * total_blocks
            policy["vision_block_token_work_ratio_to_dense"] = work / dense_work
        differences: list[float] = []
        for sample, actual_gain, predicted_gain in zip(task_samples, y, predicted):
            sample_id = str(sample["sample_id"])
            chosen = high[sample_id] if choices[sample_id] == "high" else low[sample_id]
            chosen_score = score_record(sample, chosen)
            baseline_score = score_record(sample, baseline[sample_id])
            if task == "grounding":
                differences.append(float(chosen_score["iou"]) - float(baseline_score["iou"]))
            else:
                differences.append(float(chosen_score["pck_0_1"]) - float(baseline_score["pck_0_1"]))
            detail_rows.append(
                {
                    "sample_id": sample_id,
                    "coco_image_id": int(sample["coco_image_id"]),
                    "task": task,
                    "fold": int(fold_by_image[int(sample["coco_image_id"])]),
                    "actual_utility_gain_high_minus_low": float(actual_gain),
                    "predicted_utility_gain_high_minus_low": float(predicted_gain),
                    "choice": choices[sample_id],
                }
            )
        ci = bootstrap_delta(differences, args.seed + task_index, args.bootstrap)
        ba = balanced_accuracy(y, predicted)
        correlation = float(np.corrcoef(y, predicted)[0, 1]) if y.std() > 0 and predicted.std() > 0 else 0.0
        margin = -0.02
        task_gate = {
            "balanced_accuracy": ba >= 0.55,
            "compute_budget": (
                policy.get(
                    "vision_block_token_work_ratio_to_dense",
                    policy["final_token_ratio_to_dense"],
                )
                <= 0.5
            ),
            "noninferiority": ci["ci95_low"] >= margin,
        }
        task_gate["all"] = all(task_gate.values())
        all_gates_pass &= task_gate["all"]
        analyses[task] = {
            "samples": len(task_samples),
            "positive_high_better": int((y > 0).sum()),
            "action_balanced_accuracy": ba,
            "gain_pearson_correlation": correlation,
            "policy": policy,
            "primary_delta_vs_dense": ci,
            "gate": task_gate,
        }

    status_prefix = "VISION_PREFIX_POLICY_CV" if args.vision_prefix_features else "CHEAP_POLICY_CV"
    status = f"{status_prefix}_{'PASS' if all_gates_pass else 'FAIL'}"
    report = {
        "status": status,
        "evidence_level": "192-image revealed development grouped cross-validation",
        "feature_contract": {
            "feature_region": "vision_prefix" if args.vision_prefix_features else args.feature_region,
            "image_resolution": [64, 64],
            "image_features": image_feature_names,
            "prompt_feature": "one-hot category explicitly present in prompt",
            "oracle_roi_conditional": args.feature_region == "roi_crop",
            "ground_truth_values_used_as_direct_features": False,
            "feature_count": len(image_feature_names) + len(categories),
            "vision_prefix_features": str(args.vision_prefix_features) if args.vision_prefix_features else None,
            "vision_features_by_sample": vision_features_by_sample,
        },
        "cross_validation": {
            "folds": args.folds,
            "group": "coco_image_id",
            "seed": args.seed,
            "ridge_alpha": args.ridge_alpha,
            "decision_threshold": 0.0,
        },
        "analyses": analyses,
        "rows": detail_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 低成本ROI分辨率策略交叉验证",
        "",
        f"状态：`{status}`。特征区域=`{'vision_prefix' if args.vision_prefix_features else args.feature_region}`；使用折外尺度收益预测。",
        "",
        "| 任务 | n | 512更好 | 动作平衡准确率 | 收益相关系数 | 主指标差[95% CI] | 最终Token比 | Vision工作比 | 选512比例 | 门控 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for task, item in analyses.items():
        ci = item["primary_delta_vs_dense"]
        policy = item["policy"]
        lines.append(
            f"| {task} | {item['samples']} | {item['positive_high_better']} | "
            f"{item['action_balanced_accuracy']:.4f} | {item['gain_pearson_correlation']:.4f} | "
            f"{ci['mean_delta']:+.4f} [{ci['ci95_low']:+.4f},{ci['ci95_high']:+.4f}] | "
            f"{policy['final_token_ratio_to_dense']:.3f} | "
            f"{policy.get('vision_block_token_work_ratio_to_dense', policy['final_token_ratio_to_dense']):.3f} | "
            f"{policy['high_selected_rate']:.3f} | "
            f"{'PASS' if item['gate']['all'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "门控要求：动作平衡准确率至少0.55、Vision工作代理不高于Dense的50%、主指标相对Dense的配对95% CI下界不低于-0.02。",
            "",
            "证据边界：这是已揭示开发集的折外预测，不是校准或测试结果。当前ROI由真值生成，只能评价正确区域已给定后的尺度选择；不能验证区域定位。",
        ]
    )
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "samples": len(samples)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
