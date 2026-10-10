#!/usr/bin/env python3
"""Single-image real-model B1 answer-choice-margin autograd smoke (GPU).

This is a deliberately narrow engineering smoke.  It answers exactly one
question on **one** frozen CV-Bench image: *does the real RoboBrain2.5-4B
autograd graph reach the B1 vision hidden states through a multiple-choice
answer margin?*

The path exercised on ``cuda:0`` (BF16 / SDPA) is::

    vision block 0 (full patch sequence P)
      -> forward_shallow_pruned_vision(..., prune_after_layer=0,
             capture_prune_gradient=True, retention=1.0)   # B1, no token cut
      -> prepare_pruned_multimodal_inputs(...)             # full prompt kept
      -> append common-prefix label embedding              # predict branch token
      -> model.model.language_model(...) + model.lm_head   # [V] logits
      -> choice_margin(...)                                # E vs best wrong
      -> backward()                                        # grad on B1 hidden

Hard rules encoded here:

* Exactly one sample (``sample_id == "1"``), never the full 2638 set.
* No training, no optimizer, ``model.requires_grad_(False)``.
* Labels ``"(A)".."(F)"`` are tokenised with the real tokenizer
  (``add_special_tokens=False``).  The smoke proceeds **only** when every
  label is a common prefix + exactly one distinct branch token + a common
  suffix.  Otherwise it fails closed and never fabricates a single-token
  margin.
* Every GPU timed segment is bracketed by ``torch.cuda.synchronize``.
  The recorded seconds are one-shot diagnostic numbers: they are **not**
  steady-state inference latency and **not** a selector speedup.
* This is not selector-accuracy evidence and must never be used to train a
  selector on the full2638 labels and re-score that same set.

The whole run fails closed (``FAIL_CLOSED``) when CUDA is unavailable, when the
single-sample contract is violated, or when the label branch contract does not
hold.  A genuine chain assertion failure also yields ``FAIL_CLOSED``.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from evaluation.answer_margin_b1_preflight import choice_margin
from evaluation.shallow_block_pruning import (
    forward_shallow_pruned_vision,
    prepare_pruned_multimodal_inputs,
)

__all__ = [
    "STATUS_PASS",
    "STATUS_FAIL",
    "AnswerMarginSmokeError",
    "SmokeConfig",
    "LabelBranches",
    "tokenize_labels",
    "answer_index",
    "parse_label_branches",
    "build_option_labels",
    "run_smoke",
    "render_markdown",
    "main",
]

STATUS_PASS = "REAL_MODEL_AUTOGRAD_SMOKE_PASS"
STATUS_FAIL = "FAIL_CLOSED"

REPORT_DATE = "2026-10-05"
REPORT_STEM = "2026-10-05-b1-answer-margin-gpu-smoke"
DEFAULT_REPORT_DIR = Path("reports/token_selector")
DEFAULT_MODEL = Path("/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B")
DEFAULT_MANIFEST = Path("datasets/cv_bench_full2638.jsonl")
DEFAULT_SAMPLE_ID = "1"

MODULE_PATH = Path("evaluation/run_b1_answer_margin_gpu_smoke.py")
TEST_PATH = Path("evaluation/tests/test_run_b1_answer_margin_gpu_smoke.py")

B1_PRUNE_AFTER_LAYER = 0
RETENTION = 1.0


class AnswerMarginSmokeError(ValueError):
    """Raised when a pure helper violates its contract."""


# ---------------------------------------------------------------------------
# Pure, CPU-testable helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelBranches:
    """Tokenised option labels reduced to ``prefix + one branch + suffix``."""

    common_prefix: list[int]
    branch_token_ids: list[int]
    common_suffix: list[int]
    prefix_length: int
    suffix_length: int


def build_option_labels(choice_count: int) -> list[str]:
    """Return ``["(A)", ...]`` for ``choice_count`` options."""
    if isinstance(choice_count, bool) or not isinstance(choice_count, int):
        raise AnswerMarginSmokeError("choice_count must be an int")
    if choice_count < 2:
        raise AnswerMarginSmokeError("need at least two options")
    if choice_count > 26:
        raise AnswerMarginSmokeError("at most 26 options are supported")
    return [f"({chr(ord('A') + index)})" for index in range(choice_count)]


def tokenize_labels(tokenizer: Any, labels: Sequence[str]) -> list[list[int]]:
    """Tokenise labels with ``add_special_tokens=False`` and no other change.

    ``tokenizer`` is called exactly as a HuggingFace tokenizer would be; this
    keeps the helper unit-testable with a light fake.
    """
    if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
        raise AnswerMarginSmokeError("labels must be a sequence of strings")
    tokenised: list[list[int]] = []
    for label in labels:
        if not isinstance(label, str):
            raise AnswerMarginSmokeError("every label must be a string")
        encoded = tokenizer(label, add_special_tokens=False)["input_ids"]
        ids = [int(value) for value in encoded]
        if not ids:
            raise AnswerMarginSmokeError(f"label {label!r} tokenised to nothing")
        tokenised.append(ids)
    return tokenised


def _longest_common_prefix(sequences: Sequence[Sequence[int]]) -> int:
    shortest = min(len(sequence) for sequence in sequences)
    length = 0
    while length < shortest:
        token = sequences[0][length]
        if all(sequence[length] == token for sequence in sequences):
            length += 1
        else:
            break
    return length


def _longest_common_suffix(sequences: Sequence[Sequence[int]]) -> int:
    shortest = min(len(sequence) for sequence in sequences)
    length = 0
    while length < shortest:
        token = sequences[0][len(sequences[0]) - 1 - length]
        if all(sequence[len(sequence) - 1 - length] == token for sequence in sequences):
            length += 1
        else:
            break
    return length


def parse_label_branches(label_token_ids: Sequence[Sequence[int]]) -> LabelBranches:
    """Reduce labels to a common prefix, one branch token and a common suffix.

    The result is valid only when every label is exactly
    ``common_prefix + [branch_i] + common_suffix`` with pairwise distinct
    ``branch_i``.  Anything else raises :class:`AnswerMarginSmokeError` so the
    caller can fail closed instead of inventing a single-token margin.
    """
    if isinstance(label_token_ids, (str, bytes)) or not isinstance(
        label_token_ids, Sequence
    ):
        raise AnswerMarginSmokeError("label_token_ids must be a sequence")
    sequences = [list(sequence) for sequence in label_token_ids]
    if len(sequences) < 2:
        raise AnswerMarginSmokeError("at least two labels are required for a margin")
    for index, sequence in enumerate(sequences):
        if not sequence:
            raise AnswerMarginSmokeError(f"label {index} has no tokens")
        for token in sequence:
            if isinstance(token, bool) or not isinstance(token, int):
                raise AnswerMarginSmokeError(
                    f"label {index} contains a non-int token id"
                )

    prefix_length = _longest_common_prefix(sequences)
    if any(len(sequence) == prefix_length for sequence in sequences):
        raise AnswerMarginSmokeError(
            "a label is fully consumed by the common prefix: no branch token"
        )
    remainders = [sequence[prefix_length:] for sequence in sequences]

    suffix_length = _longest_common_suffix(remainders)
    # Keep at least the branch token in every remainder.
    suffix_length = min(suffix_length, min(len(item) for item in remainders) - 1)
    # Equalise the suffix comparison so a label with extra inner tokens fails.
    suffix = remainders[0][len(remainders[0]) - suffix_length :] if suffix_length else []
    for index, remainder in enumerate(remainders):
        if len(remainder) != suffix_length + 1 or remainder[1:] != suffix:
            raise AnswerMarginSmokeError(
                "label tokens are not a single branch token plus a common suffix"
            )
    branch_token_ids = [remainder[0] for remainder in remainders]
    if len(set(branch_token_ids)) != len(branch_token_ids):
        raise AnswerMarginSmokeError("branch tokens are not pairwise distinct")
    return LabelBranches(
        common_prefix=sequences[0][:prefix_length],
        branch_token_ids=branch_token_ids,
        common_suffix=list(suffix),
        prefix_length=prefix_length,
        suffix_length=suffix_length,
    )


def answer_index(answer: str, labels: Sequence[str]) -> int:
    """Map an official answer string (e.g. ``"(E)"``) to its option index."""
    if not isinstance(answer, str):
        raise AnswerMarginSmokeError("answer must be a string")
    option_labels = list(labels)
    if not option_labels:
        raise AnswerMarginSmokeError("labels must not be empty")
    matches = [index for index, label in enumerate(option_labels) if label == answer]
    if len(matches) != 1:
        raise AnswerMarginSmokeError(
            f"answer {answer!r} matches {len(matches)} labels; expected exactly one"
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Real-model driver
# ---------------------------------------------------------------------------


@dataclass
class SmokeConfig:
    model: Path = DEFAULT_MODEL
    manifest: Path = DEFAULT_MANIFEST
    sample_id: str = DEFAULT_SAMPLE_ID
    device: str = "cuda:0"
    dtype: str = "bf16"
    model_label: str = "RoboBrain2.5-4B-BF16"
    report_dir: Path = DEFAULT_REPORT_DIR
    workspace_root: Path = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_clock() -> float:
    """Drain the GPU, then read the wall clock: a clean timed boundary."""
    torch.cuda.synchronize()
    return time.perf_counter()


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _load_sample(manifest: Path, sample_id: str, workspace_root: Path) -> dict[str, Any]:
    matches = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [record for record in matches if str(record.get("sample_id")) == sample_id]
    if len(matches) != 1:
        raise AnswerMarginSmokeError(
            f"manifest must contain exactly one sample_id={sample_id!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _resolve_image(manifest: Path, image: str) -> Path:
    path = Path(image)
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def _all_parameters_frozen(model: torch.nn.Module) -> tuple[bool, int]:
    parameters = list(model.parameters())
    frozen = all(not parameter.requires_grad for parameter in parameters)
    return frozen, len(parameters)


def _all_parameter_grads_none(model: torch.nn.Module) -> tuple[bool, int]:
    parameters = list(model.parameters())
    clean = all(parameter.grad is None for parameter in parameters)
    return clean, len(parameters)


def _base_result(config: SmokeConfig) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "date": REPORT_DATE,
        "status": STATUS_FAIL,
        "conclusion": STATUS_FAIL,
        "scope": (
            "Real-model single-image B1 answer-choice-margin autograd smoke on "
            "cuda:0, BF16/SDPA. No training, no optimizer, no network."
        ),
        "engineering_autograd_smoke_only": True,
        "is_selector_accuracy_evidence": False,
        "is_training": False,
        "not_steady_state_latency": True,
        "not_selector_speedup": True,
        "model": {
            "model_label": config.model_label,
            "model_path": str(config.model),
            "dtype": config.dtype,
            "device": config.device,
            "attn_implementation": "sdpa",
            "requires_grad": False,
        },
        "sample": None,
        "label_tokenization": None,
        "chain": None,
        "margin": None,
        "gradient": None,
        "model_parameter_grads_all_none": None,
        "all_model_parameters_requires_grad_false": None,
        "timing": {
            "gpu_boundaries_synchronized": True,
            "excludes_model_load": True,
            "not_steady_state_latency": True,
            "not_selector_speedup": True,
            "processor_seconds": None,
            "forward_seconds": None,
            "backward_seconds": None,
            "total_smoke_seconds": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        },
        "checks": [],
        "fail_closed_reason": None,
        "caveats": {
            "not_selector_accuracy_evidence": "这不是 selector 精度证据。",
            "not_training": "这不是训练；无 optimizer，模型参数全部 requires_grad=False。",
            "label_hygiene": (
                "官方答案标签不得在 CV-Bench full2638 上训练再对同集宣称结果。"
            ),
            "one_image_only": '只运行 sample_id="1" 这一张图，不得外推到 full2638。',
            "timing_caveat": (
                "计时为一次性诊断，不是稳态推理延时，也不代表 selector 加速。"
            ),
        },
    }


def _finish(result: dict[str, Any]) -> dict[str, Any]:
    """Set ``status``/``conclusion`` from the accumulated checks."""
    passed = bool(result["checks"]) and all(
        item["passed"] for item in result["checks"]
    )
    result["status"] = STATUS_PASS if passed else STATUS_FAIL
    result["conclusion"] = result["status"]
    return result


def _fail_closed(
    config: SmokeConfig,
    reason: str,
    *,
    detail: str,
    extra_checks: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = _base_result(config)
    result["fail_closed_reason"] = reason
    result["fail_closed_detail"] = detail
    if extra:
        result.update(extra)
    if extra_checks:
        result["checks"].extend(extra_checks)
    # ``FAIL_CLOSED`` is a hard stop: it must never be reclassified as a pass by
    # ``_finish`` just because the individual stop checks "passed".
    result["checks"].append(
        _check("fail_closed_hard_stop", True, f"{reason}: {detail}")
    )
    result["status"] = STATUS_FAIL
    result["conclusion"] = STATUS_FAIL
    return result


def run_smoke(config: SmokeConfig | None = None) -> dict[str, Any]:
    """Run the single-image smoke and always return a JSON-ready result."""
    config = config or SmokeConfig()

    # --- Resource gate: never load CUDA without a visible device -----------
    if not torch.cuda.is_available():
        return _fail_closed(
            config,
            "cuda_unavailable",
            detail=(
                "torch.cuda.is_available() is False; the GPU is not visible to "
                "this process, so no model was loaded and no margin was computed."
            ),
        )
    if not config.device.startswith("cuda"):
        return _fail_closed(
            config,
            "non_cuda_device",
            detail=f"device must be a cuda device, got {config.device!r}",
        )

    from transformers import AutoModelForImageTextToText, AutoProcessor

    from evaluation.run_hf_token_reduction import make_messages, processor_inputs

    result = _base_result(config)

    # --- Single sample, frozen manifest ------------------------------------
    manifest = config.manifest
    if not manifest.is_absolute():
        manifest = config.workspace_root / manifest
    try:
        sample = _load_sample(manifest, config.sample_id, config.workspace_root)
    except (OSError, AnswerMarginSmokeError, json.JSONDecodeError) as error:
        return _fail_closed(
            config, "sample_not_found", detail=f"{type(error).__name__}: {error}"
        )
    choices = [str(choice) for choice in sample.get("choices", [])]
    labels = build_option_labels(len(choices))
    answer = str(sample.get("answer", ""))
    try:
        correct_index = answer_index(answer, labels)
    except AnswerMarginSmokeError as error:
        return _fail_closed(
            config, "answer_contract_violation", detail=str(error)
        )
    result["sample"] = {
        "sample_id": str(sample["sample_id"]),
        "task": sample.get("task"),
        "source_dataset": sample.get("source_dataset"),
        "image": str(sample.get("image")),
        "prompt": str(sample.get("prompt")),
        "choices": choices,
        "labels": labels,
        "answer": answer,
        "correct_index": correct_index,
        "max_new_tokens": sample.get("max_new_tokens"),
    }

    # --- Real-tokenizer label contract (processor only, no weights) --------
    try:
        processor = AutoProcessor.from_pretrained(
            str(config.model), local_files_only=True
        )
    except Exception as error:  # pragma: no cover - environment dependent
        return _fail_closed(
            config,
            "processor_load_failed",
            detail=f"{type(error).__name__}: {error}",
        )
    tokenizer = processor.tokenizer
    try:
        label_token_ids = tokenize_labels(tokenizer, labels)
        branches = parse_label_branches(label_token_ids)
    except (AnswerMarginSmokeError, KeyError, TypeError) as error:
        return _fail_closed(
            config,
            "label_branch_contract_violation",
            detail=(
                "labels are not a common prefix plus exactly one distinct "
                f"branch token (and common suffix); {type(error).__name__}: {error}"
            ),
            extra={
                "label_tokenization": {
                    "labels": labels,
                    "label_token_ids": None,
                    "contract_ok": False,
                }
            },
        )
    decoded_branch_tokens = [
        tokenizer.decode([token]) for token in branches.branch_token_ids
    ]
    result["label_tokenization"] = {
        "labels": labels,
        "label_token_ids": label_token_ids,
        "decoded_label_tokens": [
            [tokenizer.decode([token]) for token in sequence]
            for sequence in label_token_ids
        ],
        "common_prefix": branches.common_prefix,
        "common_suffix": branches.common_suffix,
        "branch_token_ids": branches.branch_token_ids,
        "decoded_branch_tokens": decoded_branch_tokens,
        "prefix_length": branches.prefix_length,
        "suffix_length": branches.suffix_length,
        "single_branch_token_per_label": True,
        "branch_tokens_distinct": True,
        "contract_ok": True,
    }

    # --- Load the real model (excluded from every timed stage) -------------
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[config.dtype]
    torch.manual_seed(0)
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            str(config.model),
            dtype=dtype,
            device_map=config.device,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()
    except Exception as error:  # pragma: no cover - environment dependent
        return _fail_closed(
            config,
            "model_load_failed",
            detail=f"{type(error).__name__}: {error}",
        )
    model.requires_grad_(False)

    total_started = _sync_clock()
    exit_code = 0
    try:
        from PIL import Image

        image_path = _resolve_image(manifest, str(sample["image"]))
        messages = make_messages(image_path, str(sample["prompt"]))
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")

        # --- Processor ----------------------------------------------------
        processor_started = _sync_clock()
        processed = processor_inputs(processor, messages, image)
        processor_seconds = _sync_clock() - processor_started

        batch = processed.to(config.device)

        full_patch_tokens = int(batch.image_grid_thw.prod().item())
        grid_thw = [int(value) for value in batch.image_grid_thw[0].tolist()]
        merge_unit = int(model.config.vision_config.spatial_merge_size) ** 2
        if full_patch_tokens % merge_unit:
            raise AnswerMarginSmokeError(
                "processor grid is not aligned to the visual merger"
            )
        full_llm_visual_tokens = full_patch_tokens // merge_unit

        # --- Forward: B1 vision + LLM branch logits + margin --------------
        torch.cuda.reset_peak_memory_stats()
        forward_started = _sync_clock()
        vision = forward_shallow_pruned_vision(
            model.model.visual,
            batch.pixel_values,
            batch.image_grid_thw,
            retention=RETENTION,
            prune_after_layer=B1_PRUNE_AFTER_LAYER,
            capture_prune_gradient=True,
        )
        prepared = prepare_pruned_multimodal_inputs(model, batch, vision)

        inputs_embeds = prepared.inputs_embeds
        attention_mask = prepared.attention_mask
        position_ids = prepared.position_ids
        visual_pos_masks = prepared.visual_pos_masks
        if visual_pos_masks.ndim == 1:
            visual_pos_masks = visual_pos_masks.unsqueeze(0)

        prefix_length = branches.prefix_length
        if prefix_length:
            prefix_ids = torch.tensor(
                [branches.common_prefix],
                dtype=prepared.input_ids.dtype,
                device=prepared.input_ids.device,
            )
            prefix_embeds = model.get_input_embeddings()(prefix_ids)
            inputs_embeds = torch.cat([inputs_embeds, prefix_embeds], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], prefix_length),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )
            base_position = int(position_ids.max().item()) + 1
            extra_positions = (
                torch.arange(prefix_length, device=position_ids.device) + base_position
            ).view(1, 1, prefix_length)
            extra_positions = extra_positions.expand(
                position_ids.shape[0], 1, prefix_length
            )
            position_ids = torch.cat([position_ids, extra_positions], dim=2)
            visual_pos_masks = torch.cat(
                [
                    visual_pos_masks,
                    torch.zeros(
                        (visual_pos_masks.shape[0], prefix_length),
                        dtype=torch.bool,
                        device=visual_pos_masks.device,
                    ),
                ],
                dim=1,
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
        margin = choice_margin(
            branch_logits,
            [branches.branch_token_ids],
            [correct_index],
        )
        margin_value = float(margin.detach().item())

        option_ids = torch.tensor(
            branches.branch_token_ids, dtype=torch.long, device=branch_logits.device
        )
        option_logits = branch_logits[0].index_select(0, option_ids)
        option_logits_list = [float(value) for value in option_logits.detach().tolist()]
        predicted_index = int(option_logits.detach().argmax().item())
        forward_seconds = _sync_clock() - forward_started

        # --- Backward -----------------------------------------------------
        backward_started = _sync_clock()
        margin.sum().backward()
        backward_seconds = _sync_clock() - backward_started
        total_smoke_seconds = _sync_clock() - total_started

        # --- Chain evidence ----------------------------------------------
        b1_hidden = vision.prune_hidden_states
        if b1_hidden is None:
            raise AnswerMarginSmokeError(
                "capture_prune_gradient did not return B1 prune_hidden_states"
            )
        b1_shape = [int(value) for value in b1_hidden.shape]
        b1_grad = b1_hidden.grad
        b1_grad_present = b1_grad is not None
        b1_grad_finite = bool(
            torch.isfinite(b1_grad).all().item()
        ) if b1_grad_present else False
        b1_grad_nonzero = bool(
            (b1_grad.abs() > 0).any().item()
        ) if b1_grad_present else False
        b1_grad_norm = float(b1_grad.float().norm().item()) if b1_grad_present else None

        group_saliency: list[float] = []
        saliency_all_finite = False
        saliency_all_nonzero = False
        if b1_grad_present and b1_hidden.shape[0] == full_llm_visual_tokens * merge_unit:
            activation = b1_hidden.detach().float()
            gradient = b1_grad.float()
            product = (gradient * activation).reshape(
                full_llm_visual_tokens, merge_unit, -1
            )
            per_group = product.abs().sum(dim=(1, 2))
            group_saliency = [float(value) for value in per_group.tolist()]
            saliency_all_finite = bool(torch.isfinite(per_group).all().item())
            saliency_all_nonzero = bool((per_group > 0).all().item())

        grads_none, parameter_count = _all_parameter_grads_none(model)
        frozen, frozen_count = _all_parameters_frozen(model)

        result["chain"] = {
            "b1_hidden_shape": b1_shape,
            "image_grid_thw": grid_thw,
            "g": full_llm_visual_tokens,
            "merge_unit": merge_unit,
            "full_patch_tokens": full_patch_tokens,
            "retention": RETENTION,
            "prune_after_layer": B1_PRUNE_AFTER_LAYER,
            "token_pruning_applied": False,
            "prompt_token_count": int(prepared.input_ids.shape[1]),
            "branch_prediction_position": int(inputs_embeds.shape[1] - 1),
            "appended_prefix_tokens": prefix_length,
        }
        result["margin"] = {
            "definition": "logit((E)) - max(other branch-token logits)",
            "value": margin_value,
            "correct_index": correct_index,
            "correct_label": labels[correct_index],
            "predicted_index": predicted_index,
            "predicted_label": labels[predicted_index],
            "option_logits": option_logits_list,
            "branch_token_ids": branches.branch_token_ids,
            "decoded_branch_tokens": decoded_branch_tokens,
        }
        result["gradient"] = {
            "b1_hidden_grad_present": b1_grad_present,
            "b1_hidden_grad_finite": b1_grad_finite,
            "b1_hidden_grad_nonzero": b1_grad_nonzero,
            "b1_hidden_grad_norm": b1_grad_norm,
            "group_saliency": group_saliency,
            "group_saliency_all_finite": saliency_all_finite,
            "group_saliency_all_nonzero": saliency_all_nonzero,
            "group_count": len(group_saliency),
        }
        result["model_parameter_grads_all_none"] = grads_none
        result["all_model_parameters_requires_grad_false"] = frozen
        result["timing"] = {
            "gpu_boundaries_synchronized": True,
            "excludes_model_load": True,
            "not_steady_state_latency": True,
            "not_selector_speedup": True,
            "processor_seconds": processor_seconds,
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "total_smoke_seconds": total_smoke_seconds,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "timing_caveat": (
                "One-shot diagnostic timings; not steady-state inference latency "
                "and not a selector acceleration claim."
            ),
        }

        result["checks"].extend(
            [
                _check(
                    "single_sample_sample_id_1",
                    str(sample["sample_id"]) == DEFAULT_SAMPLE_ID,
                    f"sample_id={sample['sample_id']!r}",
                ),
                _check(
                    "labels_a_to_f",
                    labels == [f"({chr(ord('A') + i)})" for i in range(6)],
                    f"labels={labels}",
                ),
                _check(
                    "answer_is_E",
                    answer == "(E)" and correct_index == 4,
                    f"answer={answer!r} correct_index={correct_index}",
                ),
                _check(
                    "label_single_branch_token",
                    True,
                    f"prefix={branches.common_prefix} branches={branches.branch_token_ids} "
                    f"suffix={branches.common_suffix}",
                ),
                _check(
                    "b1_hidden_shape_equals_patch_tokens",
                    b1_shape == [full_patch_tokens, int(model.config.vision_config.hidden_size)],
                    f"b1_hidden_shape={b1_shape} full_patch_tokens={full_patch_tokens}",
                ),
                _check(
                    "b1_hidden_grad_present",
                    b1_grad_present,
                    "prune_hidden_states.grad is not None after backward",
                ),
                _check("b1_hidden_grad_finite", b1_grad_finite, "gradient is finite"),
                _check(
                    "b1_hidden_grad_nonzero",
                    b1_grad_nonzero,
                    "gradient has a non-zero entry",
                ),
                _check(
                    "group_saliency_all_finite",
                    saliency_all_finite,
                    f"{len(group_saliency)} groups finite",
                ),
                _check(
                    "group_saliency_all_nonzero",
                    saliency_all_nonzero,
                    f"{len(group_saliency)} groups non-zero",
                ),
                _check(
                    "all_model_parameter_grads_none",
                    grads_none,
                    f"{parameter_count} parameters have grad None",
                ),
                _check(
                    "all_model_parameters_requires_grad_false",
                    frozen,
                    f"{frozen_count} parameters are frozen",
                ),
                _check(
                    "margin_finite",
                    margin_value == margin_value and abs(margin_value) != float("inf"),
                    f"margin={margin_value}",
                ),
            ]
        )
        _finish(result)
        exit_code = 0 if result["status"] == STATUS_PASS else 1
        del batch, vision, prepared, inputs_embeds, outputs, branch_logits, margin
        del image
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as error:  # pragma: no cover - GPU environment dependent
        result["fail_closed_reason"] = "runtime_error"
        result["fail_closed_detail"] = f"{type(error).__name__}: {error}"
        result["checks"].append(
            _check(
                "no_unhandled_runtime_error",
                False,
                f"{type(error).__name__}: {error}",
            )
        )
        _finish(result)
        exit_code = 1

    result["exit_code"] = exit_code
    result["artifacts"] = {
        "module": {
            "path": str(MODULE_PATH),
            "sha256": _sha256(config.workspace_root / MODULE_PATH),
        },
        "test": {
            "path": str(TEST_PATH),
            "sha256": _sha256(config.workspace_root / TEST_PATH),
        },
    }
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return f"`{value}`"


def render_markdown(result: dict[str, Any]) -> str:
    """Render the Chinese Markdown companion for the machine-readable JSON."""
    model = result["model"]
    sample = result.get("sample") or {}
    tokenization = result.get("label_tokenization") or {}
    chain = result.get("chain") or {}
    margin = result.get("margin") or {}
    gradient = result.get("gradient") or {}
    timing = result["timing"]
    check_rows = "\n".join(
        f"| {item['name']} | {'✅' if item['passed'] else '❌'} | {item['detail']} |"
        for item in result["checks"]
    )
    lines = [
        "# B1 答案 choice-margin 真实模型单图 GPU smoke",
        "",
        f"- 日期：{result['date']}",
        f"- 结论：**{result['status']}**",
        f"- 模型：`{model['model_path']}`（{model['model_label']}，{model['dtype']}，"
        f"{model['device']}，SDPA）",
        f"- 范围：{result['scope']}",
        "",
        "## 1. 这说明了什么，不说明什么",
        "",
        "本 smoke **只**验证一次真实模型前向 + 反向的自动微分链路能否把选择答案的",
        "margin 梯度送回 B1 视觉 hidden states。它**不是** selector 精度证据，**不是**训练，",
        "也**不是**稳态推理延时或 selector 加速。",
        "",
        "- `engineering_autograd_smoke_only = true`、`is_selector_accuracy_evidence = false`、",
        "  `is_training = false`。",
        "- 只运行 `sample_id=\"1\"` 这一张图，不得外推到 full2638。",
        "- 官方答案标签不得在 CV-Bench full2638 上训练再对同集宣称结果。",
        "",
        "## 2. 固定样本",
        "",
        f"- sample_id：`{sample.get('sample_id')}`，task：`{sample.get('task')}`，"
        f"source：`{sample.get('source_dataset')}`。",
        f"- answer：`{sample.get('answer')}`（正确索引 `{sample.get('correct_index')}`）。",
        f"- prompt：`{sample.get('prompt')}`。",
        "",
        "## 3. 标签分叉 token（真实 tokenizer，add_special_tokens=False）",
        "",
        f"- 原始标签 token 序列：`{tokenization.get('label_token_ids')}`。",
        f"- 共同前缀：`{tokenization.get('common_prefix')}`；共同后缀："
        f"`{tokenization.get('common_suffix')}`。",
        f"- 分叉 token id：`{tokenization.get('branch_token_ids')}`；解码："
        f"`{tokenization.get('decoded_branch_tokens')}`。",
        "- 仅当每个标签都是「共同前缀 + 恰好一个互不重复的分叉 token + 共同后缀」时才继续；",
        "  否则 `FAIL_CLOSED`，绝不伪造单 token margin。",
        "",
        "## 4. 真实链路断言",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        f"| B1 hidden shape | {_fmt(chain.get('b1_hidden_shape'))} |",
        f"| image_grid_thw | {_fmt(chain.get('image_grid_thw'))} |",
        f"| G（full_llm_visual_tokens） | {_fmt(chain.get('g'))} |",
        f"| merge_unit | {_fmt(chain.get('merge_unit'))} |",
        f"| retention / prune_after_layer | {_fmt(chain.get('retention'))} / "
        f"{_fmt(chain.get('prune_after_layer'))} |",
        f"| 是否裁剪 token | {_fmt(chain.get('token_pruning_applied'))} |",
        "",
        "### margin 与预测",
        "",
        f"- margin 定义：`{margin.get('definition')}`",
        f"- margin 值：{_fmt(margin.get('value'))}",
        f"- 预测选项：`{margin.get('predicted_label')}`（索引 {margin.get('predicted_index')}），"
        f"正确选项：`{margin.get('correct_label')}`。",
        f"- 各分支 token logit：`{margin.get('option_logits')}`。",
        "",
        "### 梯度",
        "",
        f"- B1 grad 存在：`{gradient.get('b1_hidden_grad_present')}`，"
        f"有限：`{gradient.get('b1_hidden_grad_finite')}`，"
        f"非零：`{gradient.get('b1_hidden_grad_nonzero')}`，"
        f"范数：{_fmt(gradient.get('b1_hidden_grad_norm'))}。",
        f"- 每组 grad×activation saliency 有限："
        f"`{gradient.get('group_saliency_all_finite')}`，非零："
        f"`{gradient.get('group_saliency_all_nonzero')}`，组数："
        f"`{gradient.get('group_count')}`。",
        f"- 模型全部 parameter.grad 为 None："
        f"`{result.get('model_parameter_grads_all_none')}`；"
        f"全部参数 requires_grad=False："
        f"`{result.get('all_model_parameters_requires_grad_false')}`。",
        "",
        "## 5. 一次性计时与显存",
        "",
        "所有 GPU 计时段前后均 `torch.cuda.synchronize()`，且不计模型加载。",
        "",
        "| 阶段 | 秒 |",
        "| --- | --- |",
        f"| processor | {_fmt(timing.get('processor_seconds'))} |",
        f"| forward（B1 视觉 + LLM + margin） | {_fmt(timing.get('forward_seconds'))} |",
        f"| backward | {_fmt(timing.get('backward_seconds'))} |",
        f"| 总 smoke | {_fmt(timing.get('total_smoke_seconds'))} |",
        "",
        f"- peak allocated：`{timing.get('peak_allocated_bytes')}` bytes；"
        f"peak reserved：`{timing.get('peak_reserved_bytes')}` bytes。",
        "- **明确**：这些数字不是稳态推理延时，也不代表 selector 加速。",
        "",
        "## 6. 校验清单",
        "",
        "| 检查 | 通过 | 说明 |",
        "| --- | --- | --- |",
        check_rows,
        "",
        f"- fail_closed_reason：`{result.get('fail_closed_reason')}`",
        "",
        "## 7. 口径与禁止声明",
        "",
        "- **这不是 selector 精度证据**，不得据此声称选择器精度、CV-Bench 分数或部署收益。",
        "- **这不是训练**：无 optimizer，`model.requires_grad_(False)`，模型参数梯度全为 None。",
        "- **不得用 full2638 标签训练后再对同集评估**；如需训练必须使用独立留出/外部冻结集。",
        "- 只运行一张图，不得外推；计时为一次性诊断，不是稳态延时或 selector 加速。",
        "",
        f"- **下一道门**：{result.get('next_gate', '待定')}",
        "",
    ]
    return "\n".join(lines)


def _next_gate() -> str:
    return (
        "若本 smoke PASS，才允许在独立留出集上设计 B1 margin teacher 的小规模"
        "可行性实验；本 smoke 本身不得升级为精度或训练结论。"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-label", default="RoboBrain2.5-4B-BF16")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--workspace-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace_root: Path = args.workspace_root
    report_dir = args.report_dir
    if not report_dir.is_absolute():
        report_dir = workspace_root / report_dir

    config = SmokeConfig(
        model=args.model,
        manifest=args.manifest,
        sample_id=args.sample_id,
        device=args.device,
        dtype=args.dtype,
        model_label=args.model_label,
        report_dir=report_dir,
        workspace_root=workspace_root,
    )
    result = run_smoke(config)
    result["next_gate"] = _next_gate()
    if "artifacts" not in result:
        result["artifacts"] = {
            "module": {
                "path": str(MODULE_PATH),
                "sha256": _sha256(workspace_root / MODULE_PATH),
            },
            "test": {
                "path": str(TEST_PATH),
                "sha256": _sha256(workspace_root / TEST_PATH),
            },
        }

    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{REPORT_STEM}.json"
    md_path = report_dir / f"{REPORT_STEM}.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(result), encoding="utf-8")

    summary = {
        "status": result["status"],
        "fail_closed_reason": result.get("fail_closed_reason"),
        "json": str(json_path),
        "markdown": str(md_path),
        "checks_passed": sum(1 for item in result["checks"] if item["passed"]),
        "checks_total": len(result["checks"]),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if result["status"] == STATUS_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
