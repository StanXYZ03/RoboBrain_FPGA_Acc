#!/usr/bin/env python3
"""Summarize frozen retention points, paired accuracy, latency and localization."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluation.build_cv_bench_pruning_report import answer_letter, parse_choice, score, stats
from evaluation.metrics import load_jsonl, score_record, summarize


SOURCE_WEIGHTS = {"ADE20K": 0.25, "COCO": 0.25, "Omni3D": 0.5}


def write_curve_svg(path: Path, points: list[dict], latency: dict) -> None:
    """Use an XML serializer for a small scientific plot; no plotting dependency."""
    root = ET.Element("svg", {"xmlns": "http://www.w3.org/2000/svg", "width": "1080", "height": "430", "viewBox": "0 0 1080 430"})
    ET.SubElement(root, "rect", {"width": "1080", "height": "430", "fill": "white"})

    def label(x: float, y: float, value: str, **attributes) -> None:
        node = ET.SubElement(root, "text", {
            "x": str(x), "y": str(y), "font-family": "Noto Sans CJK SC, sans-serif",
            "font-size": "12", "fill": "#222222", **attributes,
        })
        node.text = value

    accuracy = [point["score"]["cv_bench_accuracy"] * 100 for point in points]
    ymin, ymax = min(accuracy) - 0.8, max(accuracy) + 0.8
    for panel, (values, title, color) in enumerate((
        ([point["retention"] * 100 for point in points], "视觉group保留率（%）", "#19835d"),
        ([latency[str(point["retention"])]["inference_ms"]["mean"] for point in points], "交错复测推理核心均值（ms）", "#bf473d"),
    )):
        left, top, width, height = 80 + panel * 535, 35, 430, 310
        xmin, xmax = min(values), max(values)
        margin = max(1.0, (xmax - xmin) * 0.08)
        xmin, xmax = xmin - margin, xmax + margin
        coordinates = []
        for tick in range(6):
            fraction = tick / 5
            x, y = left + fraction * width, top + (1 - fraction) * height
            ET.SubElement(root, "line", {"x1": str(left), "x2": str(left + width), "y1": str(y), "y2": str(y), "stroke": "#e3e3e3"})
            label(left - 8, y + 4, f"{ymin + fraction * (ymax - ymin):.2f}", **{"text-anchor": "end"})
            label(x, top + height + 23, f"{xmin + fraction * (xmax - xmin):.1f}", **{"text-anchor": "middle"})
        ET.SubElement(root, "path", {"d": f"M {left} {top} V {top + height} H {left + width}", "fill": "none", "stroke": "#555555"})
        for x_value, y_value in zip(values, accuracy):
            coordinates.append((left + (x_value - xmin) / (xmax - xmin) * width, top + (ymax - y_value) / (ymax - ymin) * height))
        ET.SubElement(root, "polyline", {"points": " ".join(f"{x:.3f},{y:.3f}" for x, y in coordinates), "fill": "none", "stroke": color, "stroke-width": "2"})
        for point, (x, y) in zip(points, coordinates):
            ET.SubElement(root, "circle", {"cx": str(x), "cy": str(y), "r": "4", "fill": color})
            label(x, y - 9, f"{point['retention']:.1%}", **{"text-anchor": "middle", "font-size": "10"})
        label(left + width / 2, 402, title, **{"text-anchor": "middle"})
        label(left - 53, top + height / 2, "CV-Bench准确率（%）", **{"text-anchor": "middle", "transform": f"rotate(-90 {left - 53} {top + height / 2})"})
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def paired_accuracy_ci(rows: list[dict], seed: int = 20260916, repeats: int = 5000) -> list[float]:
    """Exact stratified paired-bootstrap distribution for discrete correctness deltas."""
    grouped = defaultdict(list)
    source_counts = defaultdict(int)
    for row in rows:
        grouped[(row["source"], row["task"], row["source_dataset"])].append(
            int(row["pred_correct"]) - int(row["dense_correct"])
        )
        source_counts[row["source"]] += 1
    rng = np.random.default_rng(seed)
    deltas = np.zeros(repeats, dtype=np.float64)
    for (source, _, _), values in sorted(grouped.items()):
        frequencies = np.bincount(np.asarray(values, dtype=np.int64) + 1, minlength=3)
        draws = rng.multinomial(len(values), frequencies / len(values), size=repeats)
        deltas += SOURCE_WEIGHTS[source] * (draws[:, 2] - draws[:, 0]) / source_counts[source]
    return np.quantile(deltas, [0.025, 0.975]).tolist()


def unique_predictions(path: Path, sample_ids: set[str], retention: float) -> dict[str, dict]:
    rows = load_jsonl(path)
    indexed = {str(row["sample_id"]): row for row in rows}
    if len(rows) != len(indexed) or set(indexed) != sample_ids:
        raise ValueError(f"incomplete or duplicate predictions: {path}")
    for row in rows:
        if (row["selector"], row["prune_after_layer"], row["anchor_grid"]) != ("l2_anchor", 5, 3):
            raise ValueError(f"selector mismatch: {path}")
        if abs(row["requested_retention"] - retention) > 1e-12:
            raise ValueError(f"retention mismatch: {path}")
    return indexed


def csv_rows(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def latency_summary(path: Path, manifest: Path, retentions: list[float]) -> dict[str, dict]:
    records = load_jsonl(path)
    ids = {str(row["sample_id"]) for row in load_jsonl(manifest)}
    expected = {(sample_id, repeat, value) for sample_id in ids for repeat in range(3) for value in retentions}
    observed = [(str(row["sample_id"]), row["repeat_index"], row["requested_retention"]) for row in records]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError("latency records do not match the fixed three-repeat experiment")
    result = {}
    for value in retentions:
        subset = [row for row in records if row["requested_retention"] == value]
        by_sample = defaultdict(list)
        for row in subset:
            by_sample[str(row["sample_id"])].append(row)
        nondeterministic = [sample_id for sample_id, runs in by_sample.items() if len({(row["output_text"], row["selected_group_indices_sha256"]) for row in runs}) != 1]
        if nondeterministic:
            raise ValueError(f"output or selected mask changed across repeats: {nondeterministic[:3]}")
        result[str(value)] = {
            "measured_runs": len(subset),
            "distinct_samples": len(ids),
            "repeats": 3,
            "inference_ms": stats([row["elapsed_seconds"] * 1000 for row in subset]),
            "vision_ms": stats([row["vision_encoder_seconds"] * 1000 for row in subset]),
            "prefill_ms": stats([row["llm_prefill_seconds"] * 1000 for row in subset]),
            "decode_ms": stats([row["decoder_seconds"] * 1000 for row in subset]),
            "mean_output_tokens": statistics.fmean(row["output_tokens"] for row in subset),
            "output_cap_rate": statistics.fmean(row["output_tokens"] == 16 for row in subset),
            "peak_cuda_mib": max(row["peak_cuda_bytes"] for row in subset) / 1024**2,
            "repeat_mean_ms": [statistics.fmean(row["elapsed_seconds"] * 1000 for row in subset if row["repeat_index"] == repeat) for repeat in range(3)],
            "nondeterministic_groups": len(nondeterministic),
        }
    dense_mean = result["1.0"]["inference_ms"]["mean"]
    for point in result.values():
        point["mean_speedup_vs_dense"] = dense_mean / point["inference_ms"]["mean"]
    return result


def localization_summary(manifest: Path, output_dir: Path, retentions: list[float]) -> tuple[dict, list[dict]]:
    samples = load_jsonl(manifest)
    indexed = {str(row["sample_id"]): row for row in samples}
    result, raw_scored = {}, []
    scored_by_retention = {}
    for value in retentions:
        path = output_dir / f"shallow_r{round(value * 1000):04d}.jsonl"
        predictions = unique_predictions(path, set(indexed), value)
        scored = [score_record(sample, predictions[str(sample["sample_id"])]) for sample in samples]
        scored_by_retention[value] = {str(row["sample_id"]): row for row in scored}
        result[str(value)] = summarize(scored)
        raw_scored.extend({"retention": value, **row} for row in scored)
    dense = scored_by_retention[1.0]
    for value in retentions:
        intervals = {}
        for task, keys in (("pointing", ("pck_0_1", "pck_0_2")), ("grounding", ("iou", "acc_0_5"))):
            task_ids = sorted(str(row["sample_id"]) for row in samples if row["task"] == task)
            rng = np.random.default_rng(20260916)
            draws = rng.integers(0, len(task_ids), size=(5000, len(task_ids)))
            for key in keys:
                differences = np.asarray([float(scored_by_retention[value][sample_id][key]) - float(dense[sample_id][key]) for sample_id in task_ids])
                intervals[f"{task}_{key}"] = {
                    "mean_delta": float(differences.mean()),
                    "paired_delta_ci95": np.quantile(differences[draws].mean(axis=1), [0.025, 0.975]).tolist(),
                }
        result[str(value)]["paired_vs_dense"] = intervals
    return result, raw_scored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--existing-dir", type=Path, required=True)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--latency-jsonl", type=Path, required=True)
    parser.add_argument("--latency-manifest", type=Path, required=True)
    parser.add_argument("--localization-manifest", type=Path)
    parser.add_argument("--localization-dir", type=Path)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    if (args.localization_manifest is None) != (args.localization_dir is None):
        parser.error("provide both localization arguments or neither")
    retentions = [1.0, 0.875, 0.75, 0.5, 0.25]
    manifest = load_jsonl(args.manifest)
    ids = {str(row["sample_id"]) for row in manifest}
    if len(ids) != len(manifest) or len(ids) != 2638:
        raise ValueError("expected the complete 2638-sample frozen manifest")
    dense = unique_predictions(args.existing_dir / "shallow_r1000.jsonl", ids, 1.0)
    points, raw_accuracy = [], []
    for value in retentions:
        directory = args.existing_dir if value in (1.0, 0.875) else args.sweep_dir
        predictions = unique_predictions(directory / f"shallow_r{round(value * 1000):04d}.jsonl", ids, value)
        rows = []
        for sample in manifest:
            sample_id = str(sample["sample_id"])
            truth = answer_letter(sample["answer"])
            candidate = parse_choice(predictions[sample_id]["output_text"], sample["choices"])
            reference = parse_choice(dense[sample_id]["output_text"], sample["choices"])
            rows.append({
                "sample_id": sample_id, "retention": value, "source": sample["source"],
                "source_dataset": sample["source_dataset"], "task": sample["task"],
                "ground_truth": truth, "pred_answer": candidate, "dense_answer": reference,
                "pred_correct": candidate == truth, "dense_correct": reference == truth,
                "output_text": predictions[sample_id]["output_text"],
            })
        metrics = score(rows, "pred_correct")
        dense_metrics = score(rows, "dense_correct")
        raw_accuracy.extend(rows)
        raw = list(predictions.values())
        points.append({
            "retention": value, "score": metrics,
            "score_delta": metrics["cv_bench_accuracy"] - dense_metrics["cv_bench_accuracy"],
            "score_relative_retention": metrics["cv_bench_accuracy"] / dense_metrics["cv_bench_accuracy"],
            "paired_delta_ci95": paired_accuracy_ci(rows),
            "correct_to_incorrect": sum(row["dense_correct"] and not row["pred_correct"] for row in rows),
            "incorrect_to_correct": sum(not row["dense_correct"] and row["pred_correct"] for row in rows),
            "dense_correct_to_unparsed": sum(row["dense_correct"] and row["pred_answer"] is None for row in rows),
            "dense_correct_to_parsed_wrong": sum(row["dense_correct"] and row["pred_answer"] is not None and not row["pred_correct"] for row in rows),
            "parse_failure_count": sum(row["pred_answer"] is None for row in rows),
            "output_cap_count": sum(row["output_tokens"] == 16 for row in raw),
            "mean_output_tokens": statistics.fmean(row["output_tokens"] for row in raw),
            "actual_token_retention": statistics.fmean(row["actual_token_retention"] for row in raw),
            "vision_macs_ratio": statistics.fmean(row["encoder_macs_proxy"] / row["baseline_encoder_macs_proxy"] for row in raw),
            "prefill_macs_ratio": statistics.fmean(row["llm_prefill_macs_proxy"] / row["baseline_llm_prefill_macs_proxy"] for row in raw),
            "accuracy_run_core_ms": stats([row["elapsed_seconds"] * 1000 for row in raw]),
        })
    latency = latency_summary(args.latency_jsonl, args.latency_manifest, retentions)
    localization, raw_localization = ({}, [])
    if args.localization_manifest is not None:
        localization, raw_localization = localization_summary(args.localization_manifest, args.localization_dir, retentions)
    payload = {
        "strategy": "layer-5 l2_anchor, 3x3 anchors, original positions, BF16",
        "accuracy_scope": "CV-Bench complete 2638 samples",
        "latency_scope": "100 fixed samples, 3 interleaved repeats per configuration, inference core only",
        "localization_scope": "revealed 96-image COCO diagnostic; not blind generalization" if localization else "pending; no localization results included",
        "points": points, "latency": latency, "localization": localization,
    }
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    args.output_stem.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_rows(args.output_stem.with_suffix(".csv"), [
        {"retention": point["retention"], "cv_bench_accuracy": point["score"]["cv_bench_accuracy"], "delta": point["score_delta"], "relative_score_retention": point["score_relative_retention"], "ci95": point["paired_delta_ci95"], "mean_core_ms": latency[str(point["retention"])]["inference_ms"]["mean"], "speedup": latency[str(point["retention"])]["mean_speedup_vs_dense"], "actual_token_retention": point["actual_token_retention"], "vision_macs_ratio": point["vision_macs_ratio"], "prefill_macs_ratio": point["prefill_macs_ratio"]}
        for point in points
    ])
    csv_rows(args.output_stem.with_name(args.output_stem.name + "-accuracy-raw").with_suffix(".csv"), raw_accuracy)
    if localization:
        csv_rows(args.output_stem.with_name(args.output_stem.name + "-localization-raw").with_suffix(".csv"), raw_localization)

    lines = [
        "# L2 Token裁剪的保留率曲线", "", "## 实验做了什么", "",
        "保持同一模型、Prompt、图像、16 Token生成上限和选择器，只改变保留率。前6个视觉块不裁剪，在第5层后按完整2×2 group筛选；后续视觉层、DeepStack、Merger和LLM视觉序列同时压紧，保留原位置。",
        "L2评分是每个patch隐藏向量的平方和开方，再对group内4个patch取平均。3×3锚点是把空间网格分为最多9个区域，每区保留最高分group，再用全局高分group填满预算。预算向上取整，因此实际保留率可能略高于设定。",
        "全部2638题按CV-Bench官方来源组合公式计分，解析失败算错误。没有新建统一通过门槛，没有根据结果调整选择器；95%配对置信区间用于描述差异，不表示已证明非劣。", "",
        "组合公式为0.5×[(ADE20K准确率+COCO准确率)/2+Omni3D准确率]，不是直接用全部题目的正确率。本汇总用5000次分层配对bootstrap、多项分布抽样计算区间；单档报告用逐条重采样，有限次随机模拟会使区间端点略有不同，不影响点估计或本轮结论。", "",
        "## 完整集准确率", "",
        "| 保留率 | 官方组合分数 | 相对Dense变化（百分点） | 差值95% CI（百分点） | 分数相对保留 | 正确→错误 / 错误→正确 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for point in points:
        ci = point["paired_delta_ci95"]
        lines.append(f"| {point['retention']:.1%} | {point['score']['cv_bench_accuracy']:.4%} | {point['score_delta'] * 100:+.4f} | [{ci[0] * 100:+.4f}, {ci[1] * 100:+.4f}] | {point['score_relative_retention']:.4%} | {point['correct_to_incorrect']} / {point['incorrect_to_correct']} |")
    lines += ["", "| 保留率 | Count | Relation | Depth | Distance | 解析成功率 |", "|---:|---:|---:|---:|---:|---:|"]
    for point in points:
        tasks = point["score"]["task_accuracy"]
        lines.append(f"| {point['retention']:.1%} | {tasks['Count']:.4%} | {tasks['Relation']:.4%} | {tasks['Depth']:.4%} | {tasks['Distance']:.4%} | {point['score']['parse_success_rate']:.4%} |")
    lines += ["", "| 保留率 | 全部解析失败 | 达到16 Token上限 | Dense正确→未解析 | Dense正确→解析后答错 |", "|---:|---:|---:|---:|---:|"]
    for point in points:
        lines.append(f"| {point['retention']:.1%} | {point['parse_failure_count']} | {point['output_cap_count']} | {point['dense_correct_to_unparsed']} | {point['dense_correct_to_parsed_wrong']} |")
    lines += ["", "达到Token上限不必然表示答案被截断；这只是排查线索。所有配置使用同一16 Token上限，若某档解析失败明显增多，需要另设固定的长输出对照实验，不能只靠此表断言视觉信息丢失。"]
    lines += ["", "## 交错重复时延", "",
        "固定100题，各配置预热5次，每题3次正式重复，按题与重复轮次旋转配置顺序。以下数字来自同一运行的交错复测，不拿完整准确率扫描中不同时间的GPU状态比较加速比。",
        "| 保留率 | 核心均值 ms | p50 ms | p95 ms | 均值加速比 | Vision均值 ms | Prefill均值 ms | 平均输出Token |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for value in retentions:
        item = latency[str(value)]
        timing = item["inference_ms"]
        lines.append(f"| {value:.1%} | {timing['mean']:.3f} | {timing['median']:.3f} | {timing['p95']:.3f} | {item['mean_speedup_vs_dense']:.3f}× | {item['vision_ms']['mean']:.3f} | {item['prefill_ms']['mean']:.3f} | {item['mean_output_tokens']:.3f} |")
    lines += ["", "时延只包含推理核心，不含processor、H2D与图片解码；未测GPU功耗，也不是ZCU102或目标异构平台结果。输出长度变化会影响核心时延，因此同时列出输出Token，不能把全部时间差都归于矩阵工作量减少。", "", "## 定位诊断", "",
        "定位诊断方案使用已公开96张COCO图片，每图Grounding和Pointing各一条，共192条。真值框来自COCO，Pointing真值为框中心，不是人工标注的抓取可供性点。该集合只做误差归因，不作为新盲测，也不与CV-Bench加权合成分数。",
        "PCK@0.1/0.2：预测点与真值的距离不超过目标框对角线的10%/20%时计正确。Mean IoU：预测框与真值框交集面积除以并集面积的平均值。Acc@0.5：IoU至少0.5的比例。格式失败均计为错误。", "",
        "| 保留率 | PCK@0.1 | PCK@0.2 | Mean IoU | Acc@0.5 |", "|---:|---:|---:|---:|---:|"]
    for value in retentions:
        if localization:
            item = localization[str(value)]["tasks"]
            lines.append(f"| {value:.1%} | {item['pointing']['pck_0_1']:.4f} | {item['pointing']['pck_0_2']:.4f} | {item['grounding']['iou_mean']:.4f} | {item['grounding']['acc_0_5']:.4f} |")
    if not localization:
        lines += ["", "本次分档定位诊断尚未执行，表中不填估算值；上述段落说明计划使用的集合和指标，不是已完成实验。CV-Bench选择题结果不能替代精细定位验证。"]
    lines += ["", "## 计算量与可宣称范围", "",
        "| 保留率 | 实际Token保留 | Vision MAC代理保留 | Prefill MAC代理保留 |", "|---:|---:|---:|---:|"]
    for point in points:
        lines.append(f"| {point['retention']:.1%} | {point['actual_token_retention']:.4%} | {point['vision_macs_ratio']:.4%} | {point['prefill_macs_ratio']:.4%} |")
    lines += ["", "MAC代理只覆盖主要Transformer矩阵计算，排除patch embedding、Merger、DeepStack投影和文本Token。完整CV-Bench已用于观察固定曲线，后续不得把在此集合上选出的新策略称为未观察盲测。其他官方空间基准、通用VLM鲁棒性及真实Q8_0复验仍未完成；本次结果不能解除此前强裁剪定位退化的结论，也不授权Selector RTL或板端部署。", ""]

    lines += ["## 当前结论", "",
        "75%保留率的CV-Bench得分较Dense下降0.2454个百分点，推理核心均值加速约1.154倍，是本轮值得继续验证的低损失候选；差值置信区间包含零，不等于证明无损或统计非劣。",
        "50%与25%保留率分别下降2.6198和7.7244个百分点，差值置信区间均完全为负；对应核心加速约1.369和1.637倍。这是明确的质量—性能交换，不是全部方案都通过精度验证。",
        "100题时延集合各档平均输出约4.24 Token，输出长度接近；在这个集合中，收益与视觉和Prefill工作量减少一致。是否适用于精细空间定位、真实Q8后端和目标异构平台，还需独立验证。", ""]
    image_path = args.output_stem.with_suffix(".svg")
    write_curve_svg(image_path, points, latency)
    lines += [f"![完整集准确率与100题交错核心时延曲线]({image_path.name})", ""]
    args.output_stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps([{ "retention": point["retention"], "accuracy": point["score"]["cv_bench_accuracy"], "delta_ci95": point["paired_delta_ci95"], "latency": latency[str(point["retention"])]} for point in points], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
