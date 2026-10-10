#!/usr/bin/env python3
"""Extract leakage-controlled VSR B1 answer-margin teacher features.

The frozen RoboBrain model supplies a train-only teacher signal.  For each VSR
zero-shot train statement, the correct True/False branch margin is propagated
to the output of vision block 0.  Per merged group, the teacher score is
``sum(abs(gradient * activation))`` over four patches and 1024 channels.

The saved cache contains only the deployable student's inputs
(``group_mean`` and raw-prompt embeddings) plus teacher score vectors.  Full
hidden states are never stored.  Official labels are used only to choose the
teacher branch; labels, relation names, subject and object annotations are not
student inputs and are forbidden at selector inference.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from evaluation.answer_margin_b1_preflight import choice_margin
from evaluation.b1_answer_margin_oracle import group_saliency_from_grad_activation
from evaluation.extract_prompt_projection_features import prompt_embeddings_from_text
from evaluation.prompt_projection_distillation import (
    minmax_normalize_1d,
    save_feature_cache,
    temperature_soft_target,
)
from evaluation.run_b1_answer_margin_gpu_smoke import (
    parse_label_branches,
    tokenize_labels,
)
from evaluation.run_hf_token_reduction import make_messages, processor_inputs
from evaluation.shallow_block_pruning import (
    forward_shallow_pruned_vision,
    prepare_pruned_multimodal_inputs,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B")
DEFAULT_MANIFEST = ROOT / "datasets/external/vsr-zeroshot-b27a0af0/vsr_zeroshot_train_no_dev_image_overlap.jsonl"
DEFAULT_CONFIG = ROOT / "configs/2026-10-10-vsr-b1-margin-teacher.preregistered.json"
DEFAULT_OUT_DIR = ROOT / "artifacts/token_selector/2026-10-10-vsr-b1-margin-teacher"
MANIFEST_SHA256 = "08ae6b609d2f1a1431d7e5d6dec1bf1720859e17159b13bd173a8502dc6bf4f4"
TEACHER_TEMPERATURE = 0.1
LABELS = ("False", "True")
PRUNE_AFTER_LAYER = 0
RETENTION = 1.0
MERGE_UNIT = 4


class VSRTeacherError(RuntimeError):
    """Fail-closed extraction contract violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_record_sha256(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or sha256_file(path) != MANIFEST_SHA256:
        raise VSRTeacherError(f"training manifest identity mismatch: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 3483 or len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise VSRTeacherError("expected 3483 unique leakage-controlled train records")
    for row in rows:
        if int(row.get("official_label", -1)) not in (0, 1):
            raise VSRTeacherError("every training record needs a binary teacher label")
    return rows


def verify_pins(config: Mapping[str, Any]) -> None:
    pins = config.get("pins")
    if not isinstance(pins, Mapping) or not pins:
        raise VSRTeacherError("preregistration has no dependency pins")
    for name, pin in pins.items():
        path = ROOT / str(pin.get("path", ""))
        expected = str(pin.get("sha256", ""))
        if not path.is_file() or sha256_file(path) != expected:
            raise VSRTeacherError(f"preregistered pin mismatch: {name}")


def validate_label_contract(tokenizer: Any) -> dict[str, Any]:
    token_ids = tokenize_labels(tokenizer, LABELS)
    branches = parse_label_branches(token_ids)
    if branches.common_prefix or branches.common_suffix:
        raise VSRTeacherError("True/False labels must each be exactly one branch token")
    decoded = [tokenizer.decode([token_id]) for token_id in branches.branch_token_ids]
    return {
        "labels": list(LABELS),
        "label_token_ids": token_ids,
        "branch_token_ids": list(branches.branch_token_ids),
        "decoded_branch_tokens": decoded,
        "single_distinct_token_per_label": True,
    }


def teacher_targets(saliency: Sequence[float], temperature: float = TEACHER_TEMPERATURE) -> dict[str, np.ndarray]:
    raw = np.asarray(saliency, dtype=np.float32)
    if raw.ndim != 1 or raw.size == 0 or not np.isfinite(raw).all() or np.any(raw < 0):
        raise VSRTeacherError("teacher saliency must be finite, non-negative and one-dimensional")
    normalized = minmax_normalize_1d(raw)
    soft = temperature_soft_target(normalized, temperature)
    return {"raw": raw, "normalized": normalized, "soft": soft}


def validate_resume(index_path: Path, out_dir: Path, allowed_ids: set[str], config_sha256: str) -> set[str]:
    if not index_path.is_file() or index_path.stat().st_size == 0:
        return set()
    done: set[str] = set()
    for line_number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(line)
        sample_id = str(record.get("sample_id"))
        if sample_id not in allowed_ids or sample_id in done:
            raise VSRTeacherError(f"invalid/duplicate resume record at line {line_number}")
        if record.get("manifest_sha256") != MANIFEST_SHA256 or record.get("config_sha256") != config_sha256:
            raise VSRTeacherError(f"resume identity mismatch at line {line_number}")
        if record.get("record_sha256") != canonical_record_sha256(record):
            raise VSRTeacherError(f"resume record hash mismatch at line {line_number}")
        cache_path = out_dir / str(record.get("npz_name"))
        if not cache_path.is_file() or sha256_file(cache_path) != record.get("npz_file_sha256"):
            raise VSRTeacherError(f"resume cache mismatch at line {line_number}")
        done.add(sample_id)
    return done


def _sync_clock() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _append_prefix(prepared: Any, model: Any, prefix: Sequence[int]) -> tuple[Any, Any, Any, Any]:
    inputs_embeds = prepared.inputs_embeds
    attention_mask = prepared.attention_mask
    position_ids = prepared.position_ids
    visual_pos_masks = prepared.visual_pos_masks
    if visual_pos_masks.ndim == 1:
        visual_pos_masks = visual_pos_masks.unsqueeze(0)
    if not prefix:
        return inputs_embeds, attention_mask, position_ids, visual_pos_masks
    prefix_ids = torch.tensor([list(prefix)], dtype=prepared.input_ids.dtype, device=prepared.input_ids.device)
    inputs_embeds = torch.cat((inputs_embeds, model.get_input_embeddings()(prefix_ids)), dim=1)
    attention_mask = torch.cat((attention_mask, torch.ones_like(prefix_ids, dtype=attention_mask.dtype)), dim=1)
    base = int(position_ids.max().item()) + 1
    positions = (torch.arange(len(prefix), device=position_ids.device) + base).view(1, 1, -1)
    positions = positions.expand(position_ids.shape[0], 1, -1)
    position_ids = torch.cat((position_ids, positions), dim=2)
    visual_pos_masks = torch.cat((visual_pos_masks, torch.zeros_like(prefix_ids, dtype=torch.bool)), dim=1)
    return inputs_embeds, attention_mask, position_ids, visual_pos_masks


def extract_one(model: Any, processor: Any, sample: Mapping[str, Any], label_contract: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    started = _sync_clock()
    image_path = Path(str(sample["image"]))
    if not image_path.is_file() or sha256_file(image_path) != str(sample["image_sha256"]):
        raise VSRTeacherError(f"image identity mismatch: {image_path}")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    messages = make_messages(image_path, str(sample["prompt"]))
    batch = processor_inputs(processor, messages, image).to(model.device)
    full_patch_tokens = int(batch.image_grid_thw.prod().item())
    if full_patch_tokens % MERGE_UNIT:
        raise VSRTeacherError("vision patches are not merge aligned")

    torch.cuda.reset_peak_memory_stats()
    forward_started = _sync_clock()
    vision = forward_shallow_pruned_vision(
        model.model.visual,
        batch.pixel_values,
        batch.image_grid_thw,
        retention=RETENTION,
        prune_after_layer=PRUNE_AFTER_LAYER,
        capture_prune_gradient=True,
    )
    prepared = prepare_pruned_multimodal_inputs(model, batch, vision)
    branches = parse_label_branches(label_contract["label_token_ids"])
    inputs_embeds, attention_mask, position_ids, visual_pos_masks = _append_prefix(
        prepared, model, branches.common_prefix
    )
    outputs = model.model.language_model(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=prepared.deepstack_visual_embeds,
    )
    branch_logits = model.lm_head(outputs.last_hidden_state[:, -1, :])
    correct_index = int(sample["official_label"])
    margin = choice_margin(branch_logits, [branches.branch_token_ids], [correct_index])
    option_ids = torch.tensor(branches.branch_token_ids, dtype=torch.long, device=branch_logits.device)
    option_logits = branch_logits[0].index_select(0, option_ids)
    forward_seconds = _sync_clock() - forward_started

    backward_started = _sync_clock()
    margin.sum().backward()
    backward_seconds = _sync_clock() - backward_started
    hidden = vision.prune_hidden_states
    if hidden is None or hidden.grad is None or int(hidden.shape[0]) != full_patch_tokens:
        raise VSRTeacherError("B1 activation-gradient chain is incomplete")
    saliency = group_saliency_from_grad_activation(hidden, hidden.grad, MERGE_UNIT)
    targets = teacher_targets(saliency)
    group_mean = hidden.detach().float().reshape(-1, MERGE_UNIT, hidden.shape[-1]).mean(dim=1).cpu().numpy()
    prompt_embeddings = prompt_embeddings_from_text(
        model.get_input_embeddings(), processor.tokenizer, str(sample["prompt"])
    ).cpu().numpy()
    peak = int(torch.cuda.max_memory_allocated())
    total_seconds = _sync_clock() - started
    cache = {
        "group_mean": group_mean,
        "prompt_embeddings": prompt_embeddings,
        "teacher_raw": targets["raw"],
        "teacher_normalized": targets["normalized"],
        "teacher_soft": targets["soft"],
    }
    audit = {
        "group_count": len(saliency),
        "grid_h": int(batch.image_grid_thw[0, 1].item()) // 2,
        "grid_w": int(batch.image_grid_thw[0, 2].item()) // 2,
        "prompt_token_count": int(prompt_embeddings.shape[0]),
        "correct_branch": LABELS[correct_index],
        "predicted_branch": LABELS[int(option_logits.detach().argmax().item())],
        "margin": float(margin.detach().item()),
        "option_logits": [float(value) for value in option_logits.detach().cpu().tolist()],
        "saliency_sum": float(targets["raw"].sum()),
        "saliency_nonzero": int(np.count_nonzero(targets["raw"])),
        "saliency_finite": bool(np.isfinite(targets["raw"]).all()),
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "total_seconds": total_seconds,
        "peak_cuda_bytes": peak,
        "gpu_boundaries_synchronized": True,
    }
    del outputs, branch_logits, margin, prepared, vision, hidden, batch
    gc.collect()
    torch.cuda.empty_cache()
    return cache, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise VSRTeacherError("CUDA is required")
    if not args.config.is_file():
        raise VSRTeacherError(f"missing preregistration: {args.config}")
    config_sha256 = sha256_file(args.config)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("dataset", {}).get("manifest_sha256") != MANIFEST_SHA256:
        raise VSRTeacherError("preregistration manifest pin mismatch")
    verify_pins(config)
    samples = load_manifest(args.manifest)
    if args.offset < 0 or (args.limit is not None and args.limit <= 0):
        raise VSRTeacherError("offset must be >=0 and limit must be >0")
    samples = samples[args.offset : None if args.limit is None else args.offset + args.limit]
    allowed_ids = {str(row["sample_id"]) for row in samples}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out_dir / "index.jsonl"
    completed = validate_resume(index_path, args.out_dir, allowed_ids, config_sha256)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model), dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa", low_cpu_mem_usage=True, local_files_only=True,
    ).eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    label_contract = validate_label_contract(processor.tokenizer)
    with index_path.open("a", encoding="utf-8", buffering=1) as index:
        for ordinal, sample in enumerate(samples, 1):
            sample_id = str(sample["sample_id"])
            if sample_id in completed:
                print(f"SKIP {ordinal}/{len(samples)} {sample_id}", flush=True)
                continue
            model.zero_grad(set_to_none=True)
            cache, audit = extract_one(model, processor, sample, label_contract)
            npz_name = f"{sample_id}.npz"
            temporary = args.out_dir / f".{npz_name}.tmp.npz"
            final = args.out_dir / npz_name
            metadata = {
                "sample_id": sample_id,
                "task": "vsr_binary_spatial_reasoning",
                "grid_h": audit["grid_h"], "grid_w": audit["grid_w"],
                "group_count": audit["group_count"],
                "prompt_token_count": audit["prompt_token_count"],
                "temperature": TEACHER_TEMPERATURE,
                "dense_vision_blocks": 24,
                "capture_layer": "B1_after_vision_block_0",
                "prompt_tokenization": "raw user prompt",
                "add_special_tokens": False,
                "source_manifest": str(args.manifest.resolve()),
                "source_image": str(sample["image"]),
                "source_image_sha256": str(sample["image_sha256"]),
                "config_sha256": config_sha256,
            }
            saved = save_feature_cache(
                temporary,
                group_mean=cache["group_mean"], prompt_embeddings=cache["prompt_embeddings"],
                teacher_raw=cache["teacher_raw"], teacher_normalized=cache["teacher_normalized"],
                teacher_soft=cache["teacher_soft"], metadata=metadata,
            )
            os.replace(temporary, final)
            record = {
                "sample_id": sample_id,
                "source_row_index": int(sample["source_row_index"]),
                "npz_name": npz_name,
                "npz_file_sha256": sha256_file(final),
                "cache_fingerprint": saved["sample_sha256"],
                "bytes": final.stat().st_size,
                "manifest_sha256": MANIFEST_SHA256,
                "config_sha256": config_sha256,
                "teacher": "correct True/False branch margin |gradient*activation| at B1",
                "teacher_uses_train_label": True,
                "selector_inference_uses_label": False,
                "selector_inference_uses_relation_annotation": False,
                "label_contract": label_contract,
                "audit": audit,
                "runtime": {"torch": torch.__version__, "python": platform.python_version()},
            }
            record["record_sha256"] = canonical_record_sha256(record)
            index.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            print(
                f"RESULT {ordinal}/{len(samples)} {sample_id} G={audit['group_count']} "
                f"margin={audit['margin']:.4f} fwd={audit['forward_seconds']:.3f}s "
                f"bwd={audit['backward_seconds']:.3f}s peak={audit['peak_cuda_bytes']/2**30:.2f}GiB",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
