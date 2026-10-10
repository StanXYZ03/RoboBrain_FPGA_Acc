#!/usr/bin/env python3
"""Official paired CV-Bench score and same-run refinement timing report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from collections import defaultdict

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.build_cv_bench_pruning_report import answer_letter, parse_choice, score, stats
from evaluation.build_l2_retention_curve import csv_rows, paired_accuracy_ci
from evaluation.evaluate_task_conditioned_roi_proposals import vision_layer_macs
from evaluation.metrics import load_jsonl

METHODS = ("global256", "cosine_global_roi")


def validate_protocol(rows: list[dict], manifest: Path, methods: list[str], repeats: int) -> None:
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    model = "/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B"
    signatures = {json.dumps(row["run"], sort_keys=True) for row in rows}
    if len(signatures) != 1:
        raise ValueError("mixed experiment signatures")
    run = rows[0]["run"]
    expected = {"model": model, "manifest_sha256": digest, "strategies": methods,
                "repeats": repeats, "global_long_edge": 256, "roi_long_edge": 384,
                "box_extent_1000": 600, "completed_blocks": 6, "dtype": "bf16",
                "ground_truth_at_inference": False, "adaptive_escalation": False,
                "learned_ridge_used": False}
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError("experiment does not match the frozen refinement protocol")


def index_unique(rows: list[dict], expected: set[str]) -> dict[str, dict]:
    indexed = {str(row["sample_id"]): row for row in rows}
    if len(rows) != len(indexed) or set(indexed) != expected:
        raise ValueError("duplicate, incomplete or foreign accuracy records")
    return indexed


def summarize_latency(rows: list[dict], ids: set[str], dense: dict) -> tuple[dict, int]:
    keys = [(str(row["sample_id"]), row["strategy"], row["repeat"]) for row in rows]
    expected = {(sample_id, method, repeat) for sample_id in ids
                for method in ("dense", *METHODS) for repeat in range(3)}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("latency must contain exactly all three repeats of all three methods")
    summary = {}
    identity_count = 0
    for method in ("dense", *METHODS):
        subset = [row for row in rows if row["strategy"] == method]
        grouped = defaultdict(list)
        for row in subset:
            grouped[str(row["sample_id"])].append(row)
            if method == "dense":
                if row["output_text"] != dense[str(row["sample_id"])]["output_text"]:
                    raise ValueError("native Dense output identity mismatch")
                identity_count += 1
        nondeterministic = [sample_id for sample_id, runs in grouped.items()
            if len({json.dumps([row["output_text"], row["effective_crop_1000"]]) for row in runs}) != 1]
        if nondeterministic:
            raise ValueError(f"repeat nondeterminism: {nondeterministic[:3]}")
        timing_fields = [name for name in subset[0] if name.endswith("_ms")]
        summary[method] = {
            "runs": len(subset), "samples": len(ids), "repeats": 3,
            "timing": {name: stats([row[name] for row in subset]) for name in timing_fields},
            "mean_output_tokens": statistics.fmean(row["output_tokens"] for row in subset),
            "peak_cuda_mib": max(row["peak_cuda_mib"] for row in subset),
            "nondeterministic_sample_groups": len(nondeterministic),
        }
    baseline = summary["dense"]["timing"]["model_end_to_end_ms"]["mean"]
    for item in summary.values():
        item["model_mean_speedup_vs_dense"] = baseline / item["timing"]["model_end_to_end_ms"]["mean"]
    return summary, identity_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--accuracy-jsonl", type=Path, required=True)
    parser.add_argument("--dense-jsonl", type=Path, required=True)
    parser.add_argument("--latency-jsonl", type=Path, required=True)
    parser.add_argument("--latency-manifest", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    samples = load_jsonl(args.manifest)
    if any(row["max_new_tokens"] != 16 for row in samples):
        raise ValueError("this report implements the frozen 16-token protocol only")
    ids = {str(row["sample_id"]) for row in samples}
    if len(samples) != 2638 or len(ids) != 2638:
        raise ValueError("full 2638-sample manifest required")
    dense = index_unique(load_jsonl(args.dense_jsonl), ids)
    accuracy = load_jsonl(args.accuracy_jsonl)
    if len(accuracy) != 2 * len(ids) or {row["strategy"] for row in accuracy} != set(METHODS):
        raise ValueError("both complete frozen methods required")
    validate_protocol(accuracy, args.manifest, list(METHODS), 1)
    predictions = {method: index_unique([row for row in accuracy if row["strategy"] == method], ids)
                   for method in METHODS}
    scored, points = [], {}
    for method in ("dense", *METHODS):
        rows = []
        for sample in samples:
            sample_id = str(sample["sample_id"])
            pred = dense[sample_id] if method == "dense" else predictions[method][sample_id]
            truth = answer_letter(sample["answer"])
            answer = parse_choice(pred["output_text"], sample["choices"])
            dense_answer = parse_choice(dense[sample_id]["output_text"], sample["choices"])
            rows.append({"sample_id": sample_id, "method": method,
                "task": sample["task"], "source": sample["source"],
                "source_dataset": sample["source_dataset"], "ground_truth": truth,
                "pred_answer": answer, "dense_answer": dense_answer,
                "pred_correct": answer == truth, "dense_correct": dense_answer == truth,
                "output_text": pred["output_text"]})
        scored.extend(rows)
        metrics = score(rows, "pred_correct")
        dense_score = score(rows, "dense_correct")["cv_bench_accuracy"]
        item = {
            "score": metrics, "score_delta_vs_dense": metrics["cv_bench_accuracy"] - dense_score,
            "relative_score_retention": metrics["cv_bench_accuracy"] / dense_score,
            "paired_delta_ci95": paired_accuracy_ci(rows),
            "correct_to_incorrect": sum(row["dense_correct"] and not row["pred_correct"] for row in rows),
            "incorrect_to_correct": sum(not row["dense_correct"] and row["pred_correct"] for row in rows),
            "parse_failure_count": sum(row["pred_answer"] is None for row in rows),
            "dense_correct_to_unparsed": sum(row["dense_correct"] and row["pred_answer"] is None for row in rows),
            "dense_correct_to_parsed_wrong": sum(row["dense_correct"] and row["pred_answer"] is not None and not row["pred_correct"] for row in rows),
        }
        raw_rows = list((dense if method == "dense" else predictions[method]).values())
        item["output_cap_count"] = sum(row["output_tokens"] == 16 for row in raw_rows)
        item["mean_output_tokens"] = statistics.fmean(row["output_tokens"] for row in raw_rows)
        if method != "dense":
            raw = predictions[method]
            ratios = [raw[i]["final_visual_tokens"] / dense[i]["baseline_llm_visual_tokens"] for i in ids]
            mac_ratios = [
                (24 * sum(vision_layer_macs(n) for n in raw[i]["encoder_tokens_per_image"])
                 + 6 * vision_layer_macs(raw[i]["scout_visual_tokens"] * 4))
                / (24 * vision_layer_macs(dense[i]["baseline_encoder_tokens"]))
                for i in ids
            ]
            item.update(final_visual_token_ratio=stats(ratios),
                        cumulative_vision_mac_ratio=stats(mac_ratios),
                        samples_with_more_visual_tokens_than_dense=sum(ratio > 1 for ratio in ratios))
        points[method] = item
    # Same-index correctness deltas isolate the benefit of adding the crop.
    global_correct = {row["sample_id"]: row["pred_correct"] for row in scored if row["method"] == "global256"}
    crop_rows = [{**row, "dense_correct": global_correct[row["sample_id"]]}
                 for row in scored if row["method"] == "cosine_global_roi"]
    crop_effect = {
        "score_delta": points["cosine_global_roi"]["score"]["cv_bench_accuracy"] - points["global256"]["score"]["cv_bench_accuracy"],
        "paired_delta_ci95": paired_accuracy_ci(crop_rows),
        "global_correct_to_wrong": sum(row["dense_correct"] and not row["pred_correct"] for row in crop_rows),
        "global_wrong_to_correct": sum(not row["dense_correct"] and row["pred_correct"] for row in crop_rows),
    }
    perf_ids = {str(row["sample_id"]) for row in load_jsonl(args.latency_manifest)}
    if len(perf_ids) != 100:
        raise ValueError("the fixed 100-sample performance manifest is required")
    latency_raw = load_jsonl(args.latency_jsonl)
    validate_protocol(latency_raw, args.latency_manifest, ["dense", *METHODS], 3)
    latency, identities = summarize_latency(latency_raw, perf_ids, dense)
    payload = {"accuracy_samples": len(ids), "points": points,
        "crop_effect_vs_global": crop_effect, "latency": latency,
        "dense_identity_matched_runs": identities,
        "scope": "A6000 BF16; fixed training-free cosine scout; not learned ridge or adaptive escalation",
        "source_lock": "configs/cv_bench_refinement_20260916.lock.json",
        "decision": "LOWRES_GLOBAL_USEFUL_BASELINE / COSINE_REFINEMENT_DOMINATED_AT16 / FORMAT_BUDGET_DIAGNOSTIC_REQUIRED / ADAPTIVE_IDEA_NOT_REJECTED",
        "sealed_data_access": False, "board_access": False,
        "quality_gate": "none; report official scores and measured tradeoff"}
    stem = args.output_stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_rows(stem.with_name(stem.name + "-accuracy-raw").with_suffix(".csv"), scored)
    csv_rows(stem.with_name(stem.name + "-latency-raw").with_suffix(".csv"), load_jsonl(args.latency_jsonl))
    csv_rows(stem.with_suffix(".csv"), [{"method": method,
        "cv_bench_score": item["score"]["cv_bench_accuracy"],
        "score_retention": item["relative_score_retention"],
        "delta": item["score_delta_vs_dense"],
        "mean_model_end_to_end_ms": latency[method]["timing"]["model_end_to_end_ms"]["mean"],
        "speedup": latency[method]["model_mean_speedup_vs_dense"]} for method, item in points.items()])
    names = {"dense": "原图Dense", "global256": "低清全图256", "cosine_global_roi": "低清全图＋预测局部384"}
    lines = ["# 低清全局与局部细化的官方基准复测", "", "## 为什么补测", "",
        "此前Scout因为没有通过内部ROI召回门而停止，未测最终VLM输出。本次不沿用召回门作为最终精度否决条件，而按完整CV-Bench官方得分与实际时延报告。",
        "旧学习式岭回归含Grounding/Pointing任务编码，不适用于未经验证的CV-Bench题型映射。本次只复用旧免训练文本余弦选区分支；不能把结果称作旧岭回归方案的复验。", "",
        "## 方法与实施", "",
        "低清对照：原图长边缩至256，不放大，模型处理完整低清图。细化方案：低清图先经过视觉前6层及第一个DeepStack投影，计算视觉组与非图像Prompt词嵌入的最大余弦相似度，经3×3均值平滑选峰；沿用原并列处理和固定600×600框（0至1000坐标），在边缘平移保持框大小，裁原图并将局部长边调整为384。最终输入同时保留低清全图和局部图，文字说明两图关系及实际裁剪坐标，原题与选项不改。",
        "推理不读取真值坐标或答案，不训练、调参或搜索阈值。每张图都细化一次，所以这是单次全局＋局部表示复测，不是自动判断是否细化、也不是循环探索方案。局部放大不会创造原图不存在的细节。",
        "完整2638题每方案一次确定性生成，固定16 Token上限。性能另用固定100题、每档1次预热、每题3次旋转配置顺序的正式重复，不能用完整扫描的单次计时替代。", "",
        "## 指标", "",
        "官方组合得分为0.5×[(ADE20K准确率＋COCO准确率)/2＋Omni3D准确率]。解析器与此前冻结版本相同，解析失败计错。相对得分保留为压缩方案得分除以Dense得分，不是全部任务成功率。配对差值95%区间采用5000次分层bootstrap，没有自行设置统一通过线。", "",
        "## 完整集得分", "",
        "| 方法 | 官方得分 | 相对Dense变化（百分点） | 分数保留 | 差值95%区间（百分点） |",
        "|---|---:|---:|---:|---:|"]
    for method, item in points.items():
        ci = item["paired_delta_ci95"]
        lines.append(f"| {names[method]} | {item['score']['cv_bench_accuracy']:.4%} | {item['score_delta_vs_dense']*100:+.4f} | {item['relative_score_retention']:.4%} | [{ci[0]*100:+.4f}, {ci[1]*100:+.4f}] |")
    lines += ["", "| 方法 | Count | Relation | Depth | Distance | 解析成功率 |", "|---|---:|---:|---:|---:|---:|"]
    for method, item in points.items():
        task = item["score"]["task_accuracy"]
        lines.append(f"| {names[method]} | {task['Count']:.4%} | {task['Relation']:.4%} | {task['Depth']:.4%} | {task['Distance']:.4%} | {item['score']['parse_success_rate']:.4%} |")
    lines += ["", "| 方法 | 解析失败题数 | 达到16 Token上限 | Dense正确→未解析 | Dense正确→解析后答错 |", "|---|---:|---:|---:|---:|"]
    for method, item in points.items():
        lines.append(f"| {names[method]} | {item['parse_failure_count']} | {item['output_cap_count']} | {item['dense_correct_to_unparsed']} | {item['dense_correct_to_parsed_wrong']} |")
    lines += ["", "达到输出上限并不必然表示答案被截断。如果细化方案解析失败增多，需另设相同长输出上限的对照排查输出格式/预算，而不能直接认定全部损失来自图像信息。当前仍按原冻结解析器和16 Token上限报告，不事后改规则。"]
    ci = crop_effect["paired_delta_ci95"]
    lines += ["", f"增加局部图相对仅低清全图的得分变化为{crop_effect['score_delta']*100:+.4f}个百分点，95%配对区间[{ci[0]*100:+.4f}, {ci[1]*100:+.4f}]；正确转错误{crop_effect['global_correct_to_wrong']}题，错误转正确{crop_effect['global_wrong_to_correct']}题。该对照包含局部图与必要的两图解释文字，不能单独归因于图像分辨率。", "", "## 交错重复性能", "",
        "模型端到端边界：原图已解码为RGB之后，到生成结束及取回必要结果；包括图像缩放、Scout全部处理、局部裁剪、processor、H2D、最终模型生成以及测量中的主机管理开销，不包括模型加载、JPEG解码和文本解码。请求端到端另包括JPEG与文本解码。Scout计时包含特征回传CPU，未把它伪装成纯PL或纯GPU时间。",
        "| 方法 | 模型端到端均值 ms | p50 ms | p95 ms | 均值加速比 | 请求均值 ms | 最终generate均值 ms | 输出Token均值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, item in latency.items():
        timing = item["timing"]["model_end_to_end_ms"]
        lines.append(f"| {names[method]} | {timing['mean']:.3f} | {timing['median']:.3f} | {timing['p95']:.3f} | {item['model_mean_speedup_vs_dense']:.3f}× | {item['timing']['request_end_to_end_ms']['mean']:.3f} | {item['timing']['generate_ms']['mean']:.3f} | {item['mean_output_tokens']:.3f} |")
    lines += ["", "| 方法 | 最终视觉Token平均比 | 累计Vision MAC平均比 | Token超过Dense题数 |", "|---|---:|---:|---:|"]
    for method in METHODS:
        item = points[method]
        lines.append(f"| {names[method]} | {item['final_visual_token_ratio']['mean']:.4f} | {item['cumulative_vision_mac_ratio']['mean']:.4f} | {item['samples_with_more_visual_tokens_than_dense']} |")
    lines += ["", "MAC为主要Attention/MLP矩阵的估算，累计包括Scout 6层与最终全局/局部各24层；不覆盖patch embedding、DeepStack/Merger投影、位置算子或文本序列，不是硬件实测。小原图可能使两图方案反而增加Token。", "", "## 验证与可宣称范围", "",
        f"性能Dense输出{identities}/{3*len(perf_ids)}次与此前冻结Dense逐字一致；每种表示的输出与裁剪框在3次重复间一致。原始JSONL保持追加，逐条得分及计时同时导出CSV，汇总为JSON与中文报告。",
        "本次只有A6000 BF16结果，不代表Xavier NX或ZCU102的速度、功耗和精度；未测板卡、功耗、TTFT或真实Q8。不得把官方选择题得分推广为机器人成功率或精细定位无损。没有访问封存校准/测试图，没有修改原始模型或应用。", ""]
    lines += ["## 当前结论与下一步", "",
        f"仅低清全图保留{points['global256']['relative_score_retention']:.4%}的Dense分数，模型端到端均值加速{latency['global256']['model_mean_speedup_vs_dense']:.3f}倍，是值得保留的有损压缩基线，不因绝对分数下降就称为不可用。",
        "当前16 Token配置下，免训练全局＋局部方案比仅低清全图更慢且得分更低，因此不将这一具体配置推进到Selector/ROI硬件开发。",
        f"但两图方案有{points['cosine_global_roi']['parse_failure_count']}题解析失败、{points['cosine_global_roi']['output_cap_count']}题达到输出上限。原始失败文本可见答案前的较长解释，支持输出格式/预算参与退化这一诊断线索；不能由本轮断言局部图像信息没有价值，更不能否定完整的任务感知自适应细化思想。",
        "下一项预先锁定为输出预算诊断：Dense、低清全图、全局＋局部统一使用64 Token上限，仍保持原题、选区、图像、解析器和模型不变。先在同一固定100题做3次重复，再完整评测2638题；不能只对失败题增加预算、不能事后修改解析器救分。旧16 Token报告和原始记录保留，新预算另存manifest与输出。", ""]
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"scores": {method: item["score"]["cv_bench_accuracy"] for method,item in points.items()},
                      "crop_effect": crop_effect,
                      "latency": {method: item["timing"]["model_end_to_end_ms"] for method,item in latency.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
