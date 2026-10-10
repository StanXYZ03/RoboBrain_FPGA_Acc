#!/usr/bin/env python3
"""Grouped-CV gate for low-resolution task-conditioned rectangular proposals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_roi_resolution_policy import grouped_folds
from evaluation.metrics import load_jsonl


def shifted_box(center_x: float, center_y: float, width: float, height: float) -> list[float]:
    """Build a fixed-size box around a center, shifting rather than shrinking at edges."""
    width = min(1000.0, max(1.0, float(width)))
    height = min(1000.0, max(1.0, float(height)))
    x1 = min(max(0.0, float(center_x) - width / 2.0), 1000.0 - width)
    y1 = min(max(0.0, float(center_y) - height / 2.0), 1000.0 - height)
    return [x1, y1, x1 + width, y1 + height]


def smooth_heatmap(values: np.ndarray) -> np.ndarray:
    """Apply a deterministic edge-aware 3x3 mean filter."""
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError("heatmap must be a non-empty matrix")
    padded = np.pad(values, 1, mode="edge")
    result = np.zeros_like(values, dtype=np.float64)
    for row in range(3):
        for column in range(3):
            result += padded[row : row + values.shape[0], column : column + values.shape[1]]
    return result / 9.0


def heatmap_center_1000(values: np.ndarray) -> tuple[float, float]:
    smoothed = smooth_heatmap(values)
    maximum = float(smoothed.max())
    candidates = np.flatnonzero(np.isclose(smoothed.reshape(-1), maximum))
    # Mean filtering can create a plateau around an isolated peak. Preserve the
    # original peak within that plateau instead of inheriting row-major bias.
    index = int(candidates[np.argmax(values.reshape(-1)[candidates])])
    row, column = divmod(index, smoothed.shape[1])
    return (
        (column + 0.5) * 1000.0 / smoothed.shape[1],
        (row + 0.5) * 1000.0 / smoothed.shape[0],
    )


def box_metrics(proposal: list[float], truth: list[float]) -> dict[str, float | bool]:
    px1, py1, px2, py2 = proposal
    tx1, ty1, tx2, ty2 = truth
    intersection = max(0.0, min(px2, tx2) - max(px1, tx1)) * max(
        0.0, min(py2, ty2) - max(py1, ty1)
    )
    truth_area = (tx2 - tx1) * (ty2 - ty1)
    proposal_area = (px2 - px1) * (py2 - py1)
    center_x, center_y = (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0
    return {
        "target_center_recalled": px1 <= center_x <= px2 and py1 <= center_y <= py2,
        "target_box_coverage": intersection / truth_area,
        "proposal_area_fraction": proposal_area / 1_000_000.0,
    }


def cell_features(record: dict[str, Any]) -> np.ndarray:
    compact = np.asarray(record["projected_cell_features"], dtype=np.float64)
    relevance = np.asarray(record["text_relevance"], dtype=np.float64)
    grid_h, grid_w = int(record["merged_grid_h"]), int(record["merged_grid_w"])
    if compact.shape[0] != grid_h * grid_w or relevance.shape != (grid_h * grid_w,):
        raise ValueError("invalid spatial scout record")
    rows, columns = np.divmod(np.arange(grid_h * grid_w), grid_w)
    x = (columns + 0.5) / grid_w
    y = (rows + 0.5) / grid_h
    task = str(record["task"])
    task_features = np.tile(
        np.asarray([task == "grounding", task == "pointing"], dtype=np.float64),
        (grid_h * grid_w, 1),
    )
    return np.column_stack(
        (
            compact,
            relevance,
            relevance * relevance,
            np.linalg.norm(compact, axis=1),
            x,
            y,
            x * x,
            y * y,
            x * y,
            np.full(grid_h * grid_w, grid_h / grid_w),
            task_features,
        )
    )


def gaussian_targets(record: dict[str, Any], truth: list[float]) -> np.ndarray:
    grid_h, grid_w = int(record["merged_grid_h"]), int(record["merged_grid_w"])
    rows, columns = np.divmod(np.arange(grid_h * grid_w), grid_w)
    x = (columns + 0.5) * 1000.0 / grid_w
    y = (rows + 0.5) * 1000.0 / grid_h
    x1, y1, x2, y2 = truth
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    sigma_x = max((x2 - x1) / 2.0, 1000.0 / grid_w)
    sigma_y = max((y2 - y1) / 2.0, 1000.0 / grid_h)
    return np.exp(-0.5 * (((x - center_x) / sigma_x) ** 2 + ((y - center_y) / sigma_y) ** 2))


def pooled_features(record: dict[str, Any]) -> np.ndarray:
    compact = np.asarray(record["projected_cell_features"], dtype=np.float64)
    relevance = np.asarray(record["text_relevance"], dtype=np.float64)
    grid_h, grid_w = int(record["merged_grid_h"]), int(record["merged_grid_w"])
    peak = int(np.argmax(smooth_heatmap(relevance.reshape(grid_h, grid_w))))
    peak_row, peak_column = divmod(peak, grid_w)
    task = str(record["task"])
    return np.concatenate(
        (
            compact.mean(axis=0),
            compact.std(axis=0),
            compact.min(axis=0),
            compact.max(axis=0),
            np.asarray(
                [
                    relevance.mean(),
                    relevance.std(),
                    relevance.min(),
                    relevance.max(),
                    (peak_column + 0.5) / grid_w,
                    (peak_row + 0.5) / grid_h,
                    grid_h / grid_w,
                    grid_h * grid_w / 100.0,
                    task == "grounding",
                    task == "pointing",
                ],
                dtype=np.float64,
            ),
        )
    )


def ridge_fit(
    train_x: np.ndarray,
    train_y: np.ndarray,
    alpha: float,
    weights: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    x = np.column_stack((np.ones(len(train_x)), (train_x - mean) / scale))
    if weights is None:
        weighted_x, weighted_y = x, train_y
    else:
        root = np.sqrt(np.asarray(weights, dtype=np.float64))[:, None]
        weighted_x, weighted_y = x * root, train_y * root[:, 0]
    penalty = np.eye(x.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_x.T @ weighted_x + penalty, weighted_x.T @ weighted_y
    )
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def ridge_apply(model: dict[str, np.ndarray], values: np.ndarray) -> np.ndarray:
    x = np.column_stack(
        (np.ones(len(values)), (values - model["mean"]) / model["scale"])
    )
    return x @ model["coefficients"]


def qwen_visual_tokens(width: int, height: int, *, min_pixels: int) -> tuple[int, int]:
    resized_h, resized_w = smart_resize(
        height, width, factor=32, min_pixels=min_pixels, max_pixels=16_777_216
    )
    encoder_tokens = (resized_h // 16) * (resized_w // 16)
    return encoder_tokens, encoder_tokens // 4


def roi_visual_tokens(
    proposal: list[float], original_width: int, original_height: int, long_edge: int
) -> tuple[int, int]:
    x1, y1, x2, y2 = proposal
    pixel_x1 = max(0, int(x1 * original_width / 1000.0))
    pixel_y1 = max(0, int(y1 * original_height / 1000.0))
    pixel_x2 = min(original_width, max(1, int(round(x2 * original_width / 1000.0))))
    pixel_y2 = min(original_height, max(1, int(round(y2 * original_height / 1000.0))))
    width, height = pixel_x2 - pixel_x1, pixel_y2 - pixel_y1
    if width <= 0 or height <= 0:
        raise ValueError("proposal produced an empty pixel crop")
    scale = long_edge / max(width, height)
    resized_w, resized_h = max(1, round(width * scale)), max(1, round(height * scale))
    return qwen_visual_tokens(resized_w, resized_h, min_pixels=1024)


def vision_layer_macs(tokens: int) -> int:
    return (
        4 * tokens * 1024 * 1024
        + 2 * tokens * 1024 * 4096
        + 2 * tokens * tokens * 1024
    )


def bootstrap_mean(values: list[float], *, seed: int, replicates: int) -> dict[str, float]:
    rng = random.Random(seed)
    means = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(replicates)
    ]
    means.sort()
    return {
        "mean": statistics.fmean(values),
        "ci95_low": means[int(0.025 * (replicates - 1))],
        "ci95_high": means[int(0.975 * (replicates - 1))],
    }


def summarize(
    rows: list[dict[str, Any]], gates: dict[str, float], *, seed: int, bootstrap: int
) -> dict[str, Any]:
    result: dict[str, Any] = {"samples": len(rows)}
    fields = (
        "target_center_recalled",
        "target_box_coverage",
        "proposal_area_fraction",
        "final_roi_visual_token_ratio",
        "cumulative_vision_macs_ratio",
    )
    for offset, field in enumerate(fields):
        result[field] = bootstrap_mean(
            [float(row[field]) for row in rows], seed=seed + offset, replicates=bootstrap
        )
    result["median_scout_prefix_ms"] = statistics.median(
        float(row["scout_prefix_ms"]) for row in rows
    )
    result["gate_checks"] = {
        "target_center_recall": result["target_center_recalled"]["mean"]
        >= gates["target_center_recall_min"],
        "target_box_coverage": result["target_box_coverage"]["mean"]
        >= gates["mean_target_box_coverage_min"],
        "final_roi_visual_tokens": result["final_roi_visual_token_ratio"]["mean"]
        <= gates["mean_final_roi_visual_token_ratio_max"],
        "cumulative_vision_macs": result["cumulative_vision_macs_ratio"]["mean"]
        <= gates["mean_cumulative_vision_macs_ratio_max"],
    }
    result["gate_pass"] = all(result["gate_checks"].values())
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if sha256(args.manifest) != lock["dataset"]["manifest_sha256"]:
        raise ValueError("manifest hash does not match the experiment lock")
    samples = load_jsonl(args.manifest)
    feature_rows = load_jsonl(args.features)
    sample_by_id = {str(sample["sample_id"]): sample for sample in samples}
    feature_by_id = {str(row["sample_id"]): row for row in feature_rows}
    if set(sample_by_id) != set(feature_by_id) or len(feature_rows) != len(feature_by_id):
        raise ValueError("scout features do not match the development manifest")
    for row in feature_rows:
        run = row["run"]
        if not run.get("source_is_full_image") or int(run["generation_calls"]) != 0:
            raise ValueError("features are not from the locked full-image scout")
        if int(run["global_long_edge"]) != int(lock["scout"]["input"].split()[-1]):
            raise ValueError("scout long edge does not match the lock")
        if int(run["completed_vision_blocks"]) != lock["scout"]["completed_vision_blocks"]:
            raise ValueError("scout depth does not match the lock")

    folds = grouped_folds(
        [int(sample["coco_image_id"]) for sample in samples],
        int(lock["dataset"]["folds"]),
        int(lock["dataset"]["seed"]),
    )
    learned: dict[str, dict[str, Any]] = {}
    for fold in range(int(lock["dataset"]["folds"])):
        train_samples = [
            sample for sample in samples if folds[int(sample["coco_image_id"])] != fold
        ]
        test_samples = [
            sample for sample in samples if folds[int(sample["coco_image_id"])] == fold
        ]
        train_cell_x = np.concatenate(
            [cell_features(feature_by_id[str(sample["sample_id"])]) for sample in train_samples]
        )
        train_cell_y = np.concatenate(
            [
                gaussian_targets(
                    feature_by_id[str(sample["sample_id"])],
                    [float(value) for value in sample["ground_truth"]["bbox_1000"]],
                )
                for sample in train_samples
            ]
        )
        heatmap_model = ridge_fit(
            train_cell_x,
            train_cell_y,
            float(lock["candidates"]["ridge_heatmap_scale"]["ridge_alpha"]),
            weights=1.0 + 4.0 * train_cell_y,
        )
        train_pooled = np.stack(
            [pooled_features(feature_by_id[str(sample["sample_id"])]) for sample in train_samples]
        )
        train_width = np.asarray(
            [
                math.log(
                    (float(sample["ground_truth"]["bbox_1000"][2])
                     - float(sample["ground_truth"]["bbox_1000"][0]))
                    / 1000.0
                )
                for sample in train_samples
            ]
        )
        train_height = np.asarray(
            [
                math.log(
                    (float(sample["ground_truth"]["bbox_1000"][3])
                     - float(sample["ground_truth"]["bbox_1000"][1]))
                    / 1000.0
                )
                for sample in train_samples
            ]
        )
        alpha = float(lock["candidates"]["ridge_heatmap_scale"]["ridge_alpha"])
        width_model = ridge_fit(train_pooled, train_width, alpha)
        height_model = ridge_fit(train_pooled, train_height, alpha)
        for sample in test_samples:
            sample_id = str(sample["sample_id"])
            record = feature_by_id[sample_id]
            grid_h, grid_w = int(record["merged_grid_h"]), int(record["merged_grid_w"])
            heatmap = ridge_apply(heatmap_model, cell_features(record)).reshape(grid_h, grid_w)
            center_x, center_y = heatmap_center_1000(heatmap)
            pooled = pooled_features(record)[None, :]
            multiplier = float(lock["candidates"]["ridge_heatmap_scale"]["scale_multiplier"])
            minimum, maximum = lock["candidates"]["ridge_heatmap_scale"]["predicted_extent_clip_1000"]
            width = min(maximum, max(minimum, 1000.0 * multiplier * math.exp(float(ridge_apply(width_model, pooled)[0]))))
            height = min(maximum, max(minimum, 1000.0 * multiplier * math.exp(float(ridge_apply(height_model, pooled)[0]))))
            learned[sample_id] = {
                "proposal_box_1000": shifted_box(center_x, center_y, width, height),
                "heatmap_center_1000": [center_x, center_y],
                "predicted_width_1000": width,
                "predicted_height_1000": height,
                "fold": fold,
            }

    detail_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    fixed = lock["candidates"]["cosine_fixed_box"]
    roi_long_edge = int(lock["roi_encoding"]["long_edge"])
    for sample in samples:
        sample_id = str(sample["sample_id"])
        feature = feature_by_id[sample_id]
        grid_h, grid_w = int(feature["merged_grid_h"]), int(feature["merged_grid_w"])
        relevance = np.asarray(feature["text_relevance"], dtype=np.float64).reshape(grid_h, grid_w)
        cosine_center = heatmap_center_1000(relevance)
        proposals = {
            "cosine_fixed_box": shifted_box(
                *cosine_center,
                float(fixed["box_width_1000"]),
                float(fixed["box_height_1000"]),
            ),
            "ridge_heatmap_scale": learned[sample_id]["proposal_box_1000"],
        }
        truth = [float(value) for value in sample["ground_truth"]["bbox_1000"]]
        dense_encoder, dense_llm = qwen_visual_tokens(
            int(feature["original_width"]), int(feature["original_height"]), min_pixels=65536
        )
        scout_encoder = int(feature["encoder_tokens"])
        for method, proposal in proposals.items():
            roi_encoder, roi_llm = roi_visual_tokens(
                proposal,
                int(feature["original_width"]),
                int(feature["original_height"]),
                roi_long_edge,
            )
            metrics = box_metrics(proposal, truth)
            row = {
                "sample_id": sample_id,
                "coco_image_id": int(sample["coco_image_id"]),
                "task": str(sample["task"]),
                "fold": int(folds[int(sample["coco_image_id"])]),
                "method": method,
                "proposal_box_1000": proposal,
                **metrics,
                "dense_visual_tokens": dense_llm,
                "scout_visual_tokens": int(feature["llm_visual_tokens"]),
                "roi_visual_tokens": roi_llm,
                "final_roi_visual_token_ratio": roi_llm / dense_llm,
                "cumulative_vision_macs_ratio": (
                    6 * vision_layer_macs(scout_encoder)
                    + 24 * vision_layer_macs(roi_encoder)
                )
                / (24 * vision_layer_macs(dense_encoder)),
                "scout_prefix_ms": float(feature["prefix_ms"]),
            }
            detail_rows.append(row)
            predictions.append(
                {
                    "sample_id": sample_id,
                    "coco_image_id": int(sample["coco_image_id"]),
                    "task": str(sample["task"]),
                    "fold": row["fold"],
                    "method": method,
                    "proposal_box_1000": proposal,
                    "target_center_recalled": metrics["target_center_recalled"],
                    "target_box_coverage": metrics["target_box_coverage"],
                    "roi_visual_tokens": roi_llm,
                }
            )

    analyses: dict[str, Any] = {}
    for method in ("cosine_fixed_box", "ridge_heatmap_scale"):
        analyses[method] = {}
        for task_offset, task in enumerate(("grounding", "pointing")):
            rows = [row for row in detail_rows if row["method"] == method and row["task"] == task]
            analyses[method][task] = summarize(
                rows,
                lock["gate"],
                seed=int(lock["dataset"]["seed"]) + task_offset,
                bootstrap=args.bootstrap,
            )
        analyses[method]["joint_gate_pass"] = all(
            analyses[method][task]["gate_pass"] for task in ("grounding", "pointing")
        )

    passing = [method for method, value in analyses.items() if value["joint_gate_pass"]]
    report = {
        "status": (
            "RECTANGULAR_ROI_LOCALIZATION_GATE_PASS"
            if passing
            else "RECTANGULAR_ROI_LOCALIZATION_GATE_FAIL"
        ),
        "lock": str(args.lock),
        "lock_sha256": sha256(args.lock),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "features": str(args.features),
        "features_sha256": sha256(args.features),
        "samples": len(samples),
        "unique_images": len({int(sample["coco_image_id"]) for sample in samples}),
        "analyses": analyses,
        "passing_candidates": passing,
        "calibration_accessed": False,
        "test_accessed": False,
        "next_action": (
            "Run one out-of-fold predicted-ROI BF16 VLM comparison."
            if passing
            else "Stop this controller branch without tuning on development data."
        ),
    }

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    lines = [
        "# 任务感知矩形ROI定位门",
        "",
        f"状态：`{report['status']}`。仅使用192张已揭示开发图像做按图像分组的5折交叉验证；校准集和测试集未访问。",
        "",
        "## 定位与成本",
        "",
        "| 方法 | 任务 | 中心召回 | 目标覆盖 | ROI面积比 | ROI Token比 | 累计Vision MAC比 | 定位门 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    names = {"cosine_fixed_box": "文本余弦固定框", "ridge_heatmap_scale": "岭回归热力图+尺度"}
    for method in ("cosine_fixed_box", "ridge_heatmap_scale"):
        for task in ("grounding", "pointing"):
            value = analyses[method][task]
            lines.append(
                f"| {names[method]} | {task} | {value['target_center_recalled']['mean']:.4f} | "
                f"{value['target_box_coverage']['mean']:.4f} | {value['proposal_area_fraction']['mean']:.4f} | "
                f"{value['final_roi_visual_token_ratio']['mean']:.4f} | "
                f"{value['cumulative_vision_macs_ratio']['mean']:.4f} | {value['gate_pass']} |"
            )
    lines.extend(
        [
            "",
            "## 预设门槛",
            "",
            "每个任务必须同时满足：目标中心召回不低于0.95、目标框平均覆盖不低于0.85、ROI最终视觉Token比不高于0.50、Scout加ROI的累计Vision MAC代理比不高于0.60。候选只有两项任务均通过才进入真实VLM生成。",
            "",
            f"通过候选：`{', '.join(passing) if passing else '无'}`。",
            "",
            "## 解释边界",
            "",
            "热力图与尺度头只在训练折读取COCO框，测试折提议不读取真值；同一图像的Grounding与Pointing始终位于同一折。MAC是视觉主算子的解析代理，不是目标硬件实测。定位门失败时按锁定规则停止，不以开发集结果重新调整框尺寸、深度或正则化。",
            "",
        ]
    )
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
