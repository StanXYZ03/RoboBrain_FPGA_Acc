#!/usr/bin/env python3
"""CPU-only autograd preflight for a future B1 answer-choice-margin teacher.

This module is deliberately small and side-effect free.  It answers one narrow
engineering question: **can the B1 hidden states stay in the autograd graph all
the way to a multiple-choice answer margin?**

The path exercised here is::

    hidden_states [P, D]                     # B1 (after vision block 0)
      -> differentiable_group_gather(...)    # [K * merge_unit, D], no detach
      -> frozen fake lm_head                 # logits
      -> choice_margin(...)                  # correct logit - max(other logits)
      -> backward()                          # finite, non-zero grad on B1 hidden

Nothing here loads model weights, touches CUDA, uses the network, trains, or
reads any evaluation label.  It is a *tensor and gradient contract* check, not
a selector-quality measurement and not a claim about CV-Bench accuracy.

Label hygiene that must hold before this becomes a real teacher signal:

* Official CV-Bench answer labels may never be used to train a selector on the
  full2638 set and then be re-scored on that same set as if it were held out.
* Any follow-up GPU smoke test is limited to a single image and must not train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "PREFLIGHT_STATUS_PASS",
    "PREFLIGHT_STATUS_FAIL",
    "AnswerMarginContractError",
    "PreflightSettings",
    "differentiable_group_gather",
    "choice_margin",
    "choice_margin_from_positions",
    "run_preflight",
    "render_markdown",
    "main",
]

PREFLIGHT_STATUS_PASS = "PREFLIGHT_PASS"
PREFLIGHT_STATUS_FAIL = "PREFLIGHT_FAIL"

REPORT_DATE = "2026-10-05"
REPORT_STEM = f"{REPORT_DATE}-b1-answer-margin-autograd-preflight"
DEFAULT_REPORT_DIR = Path("reports/token_selector")
MODULE_PATH = Path("evaluation/answer_margin_b1_preflight.py")
TEST_PATH = Path("evaluation/tests/test_answer_margin_b1_preflight.py")

_ISELECT_INT_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


class AnswerMarginContractError(ValueError):
    """Raised when a gather or choice-margin input violates the contract."""


# ---------------------------------------------------------------------------
# Differentiable group gather
# ---------------------------------------------------------------------------


def _as_long_index_tensor(
    group_indices: torch.Tensor | Sequence[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    if torch.is_tensor(group_indices):
        if group_indices.ndim != 1:
            raise AnswerMarginContractError(
                "group_indices must be a one-dimensional tensor"
            )
        if group_indices.dtype == torch.bool or group_indices.is_floating_point():
            raise AnswerMarginContractError(
                "group_indices must use an integer dtype"
            )
        if group_indices.dtype not in _ISELECT_INT_DTYPES:
            raise AnswerMarginContractError(
                f"group_indices dtype {group_indices.dtype} is not an integer index type"
            )
        return group_indices.to(device=device, dtype=torch.long)
    if isinstance(group_indices, (str, bytes)) or not isinstance(
        group_indices, Sequence
    ):
        raise AnswerMarginContractError(
            "group_indices must be a one-dimensional tensor or a sequence of ints"
        )
    values = list(group_indices)
    if not values:
        raise AnswerMarginContractError("group_indices must not be empty")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise AnswerMarginContractError(
                "group_indices entries must be plain ints"
            )
    return torch.tensor(values, dtype=torch.long, device=device)


def differentiable_group_gather(
    hidden_states: torch.Tensor,
    group_indices: torch.Tensor | Sequence[int],
    merge_unit: int = 4,
) -> torch.Tensor:
    """Gather complete merge groups while keeping autograd intact.

    ``hidden_states`` has shape ``[P, D]`` where ``P = G * merge_unit`` patch
    tokens.  ``group_indices`` has shape ``[K]`` and selects complete groups in
    the order given.  The result is ``[K * merge_unit, D]`` where row
    ``k * merge_unit + j`` is exactly ``hidden_states[g_k * merge_unit + j]``.

    The operation is a pure ``index_select`` on the *same* device and dtype.
    It never calls ``detach``/``numpy``/``cpu`` and never rebuilds a tensor from
    host memory, so gradients flow back to ``hidden_states``.
    """
    if not torch.is_tensor(hidden_states):
        raise AnswerMarginContractError("hidden_states must be a torch.Tensor")
    if hidden_states.ndim != 2:
        raise AnswerMarginContractError(
            f"hidden_states must be [P, D], got ndim={hidden_states.ndim}"
        )
    if not hidden_states.is_floating_point():
        raise AnswerMarginContractError(
            "hidden_states must use a floating-point dtype"
        )
    patch_tokens, hidden_dim = int(hidden_states.shape[0]), int(hidden_states.shape[1])
    if patch_tokens <= 0 or hidden_dim <= 0:
        raise AnswerMarginContractError("hidden_states must not be empty")
    if isinstance(merge_unit, bool) or not isinstance(merge_unit, int) or merge_unit <= 0:
        raise AnswerMarginContractError("merge_unit must be a positive integer")
    if patch_tokens % merge_unit != 0:
        raise AnswerMarginContractError(
            f"hidden_states rows ({patch_tokens}) must be divisible by merge_unit "
            f"({merge_unit})"
        )
    group_count = patch_tokens // merge_unit

    indices = _as_long_index_tensor(group_indices, device=hidden_states.device)
    if indices.numel() == 0:
        raise AnswerMarginContractError("group_indices must not be empty")
    if int(indices.min().item()) < 0 or int(indices.max().item()) >= group_count:
        raise AnswerMarginContractError(
            f"group_indices must be in [0, {group_count}), got "
            f"[{int(indices.min().item())}, {int(indices.max().item())}]"
        )
    if int(torch.unique(indices).numel()) != int(indices.numel()):
        raise AnswerMarginContractError("group_indices must not contain duplicates")

    offsets = torch.arange(
        merge_unit, device=hidden_states.device, dtype=torch.long
    )
    patch_indices = (indices[:, None] * merge_unit + offsets[None, :]).reshape(-1)
    return hidden_states.index_select(0, patch_indices)


# ---------------------------------------------------------------------------
# Choice margin
# ---------------------------------------------------------------------------


def _validate_choice_options(
    option_token_ids: Sequence[Sequence[int]],
    correct_index: Sequence[int] | torch.Tensor,
    *,
    question_count: int,
    vocab_size: int,
) -> tuple[list[list[int]], list[int]]:
    if isinstance(option_token_ids, (str, bytes)) or not isinstance(
        option_token_ids, Sequence
    ):
        raise AnswerMarginContractError(
            "option_token_ids must be a sequence with one entry per question"
        )
    if len(option_token_ids) != question_count:
        raise AnswerMarginContractError(
            f"option_token_ids has {len(option_token_ids)} questions but "
            f"answer_logits has {question_count}"
        )
    if torch.is_tensor(correct_index):
        if correct_index.ndim != 1:
            raise AnswerMarginContractError("correct_index must be one-dimensional")
        if correct_index.dtype == torch.bool or correct_index.is_floating_point():
            raise AnswerMarginContractError("correct_index must use an integer dtype")
        correct_values = [int(value) for value in correct_index.tolist()]
    else:
        if isinstance(correct_index, (str, bytes)) or not isinstance(
            correct_index, Sequence
        ):
            raise AnswerMarginContractError(
                "correct_index must be a one-dimensional tensor or a sequence of ints"
            )
        correct_values = list(correct_index)
    if len(correct_values) != question_count:
        raise AnswerMarginContractError(
            f"correct_index has {len(correct_values)} entries but answer_logits "
            f"has {question_count}"
        )

    normalized_options: list[list[int]] = []
    for question, raw_options in enumerate(option_token_ids):
        if isinstance(raw_options, (str, bytes)) or not isinstance(
            raw_options, Sequence
        ):
            raise AnswerMarginContractError(
                f"question {question} options must be a sequence of token ids"
            )
        options = list(raw_options)
        if len(options) < 2:
            raise AnswerMarginContractError(
                f"question {question} needs at least two options to define a margin"
            )
        for token in options:
            if isinstance(token, bool) or not isinstance(token, int):
                raise AnswerMarginContractError(
                    f"question {question} option token ids must be plain ints"
                )
            if not (0 <= token < vocab_size):
                raise AnswerMarginContractError(
                    f"question {question} option token id {token} is outside "
                    f"[0, {vocab_size})"
                )
        if len(set(options)) != len(options):
            raise AnswerMarginContractError(
                f"question {question} contains duplicate option token ids"
            )
        chosen = correct_values[question]
        if isinstance(chosen, bool) or not isinstance(chosen, int):
            raise AnswerMarginContractError(
                f"question {question} correct_index must be a plain int"
            )
        if not (0 <= chosen < len(options)):
            raise AnswerMarginContractError(
                f"question {question} correct_index {chosen} is outside "
                f"[0, {len(options)})"
            )
        normalized_options.append(options)
    return normalized_options, correct_values


def choice_margin(
    answer_logits: torch.Tensor,
    option_token_ids: Sequence[Sequence[int]],
    correct_index: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Return ``correct_logit - max(other_option_logits)`` per question.

    ``answer_logits`` is ``[Q, V]`` with one row per question.  Each question
    supplies its own option token ids, so the option count may vary (4, 5, 6
    for A-D/A-E/A-F, or any explicit token ids).  Option token ids must be
    unique within a question; duplicates, out-of-range ids, and out-of-range
    ``correct_index`` values fail closed.

    The result is ``[Q]`` and remains differentiable with respect to
    ``answer_logits``.
    """
    if not torch.is_tensor(answer_logits):
        raise AnswerMarginContractError("answer_logits must be a torch.Tensor")
    if answer_logits.ndim != 2:
        raise AnswerMarginContractError(
            f"answer_logits must be [Q, V], got ndim={answer_logits.ndim}"
        )
    if not answer_logits.is_floating_point():
        raise AnswerMarginContractError(
            "answer_logits must use a floating-point dtype"
        )
    question_count, vocab_size = int(answer_logits.shape[0]), int(answer_logits.shape[1])
    if question_count <= 0 or vocab_size <= 0:
        raise AnswerMarginContractError("answer_logits must not be empty")

    normalized_options, correct_values = _validate_choice_options(
        option_token_ids,
        correct_index,
        question_count=question_count,
        vocab_size=vocab_size,
    )

    margins: list[torch.Tensor] = []
    for question in range(question_count):
        ids = torch.tensor(
            normalized_options[question],
            dtype=torch.long,
            device=answer_logits.device,
        )
        option_logits = answer_logits[question].index_select(0, ids)
        chosen = correct_values[question]
        correct_logit = option_logits[chosen]
        others = torch.cat((option_logits[:chosen], option_logits[chosen + 1 :]))
        margins.append(correct_logit - others.max())
    return torch.stack(margins)


def choice_margin_from_positions(
    token_logits: torch.Tensor,
    positions: torch.Tensor | Sequence[int],
    option_token_ids: Sequence[Sequence[int]],
    correct_index: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Select ``[Q]`` answer positions from ``[T, V]`` logits, then margin.

    This is the shape a real teacher would use: full sequence logits plus the
    answer-token positions.  It only performs a differentiable ``index_select``
    so gradients reach the logits that produced the selected positions.
    """
    if not torch.is_tensor(token_logits) or token_logits.ndim != 2:
        raise AnswerMarginContractError("token_logits must be a [T, V] tensor")
    if torch.is_tensor(positions):
        if positions.ndim != 1:
            raise AnswerMarginContractError("positions must be one-dimensional")
        if positions.dtype == torch.bool or positions.is_floating_point():
            raise AnswerMarginContractError("positions must use an integer dtype")
        position_tensor = positions.to(device=token_logits.device, dtype=torch.long)
    else:
        if isinstance(positions, (str, bytes)) or not isinstance(positions, Sequence):
            raise AnswerMarginContractError(
                "positions must be a one-dimensional tensor or sequence of ints"
            )
        values = list(positions)
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise AnswerMarginContractError("positions entries must be plain ints")
        position_tensor = torch.tensor(
            values, dtype=torch.long, device=token_logits.device
        )
    if position_tensor.numel() == 0:
        raise AnswerMarginContractError("positions must not be empty")
    if int(position_tensor.min().item()) < 0 or int(position_tensor.max().item()) >= int(
        token_logits.shape[0]
    ):
        raise AnswerMarginContractError("positions are outside the token axis")
    answer_logits = token_logits.index_select(0, position_tensor)
    return choice_margin(answer_logits, option_token_ids, correct_index)


# ---------------------------------------------------------------------------
# Preflight driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreflightSettings:
    """Small CPU fixture sizes; nothing here is a production configuration."""

    seed: int = 20261005
    hidden_dim: int = 64
    vocab_size: int = 32
    group_count: int = 30
    selected_group_count: int = 8
    merge_unit: int = 4


def _deterministic_questions() -> tuple[list[list[int]], list[int]]:
    """A-D, A-F and a 3-option question with explicit token ids."""
    return (
        [[3, 7, 11, 19], [0, 4, 8, 12, 16, 20], [1, 2, 5]],
        [2, 5, 0],
    )


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def run_preflight(settings: PreflightSettings | None = None) -> dict[str, Any]:
    """Run the deterministic CPU preflight and return a JSON-ready result."""
    settings = settings or PreflightSettings()
    merge_unit = int(settings.merge_unit)
    if merge_unit <= 0:
        raise AnswerMarginContractError("merge_unit must be positive")
    if settings.group_count <= 0 or settings.hidden_dim <= 0 or settings.vocab_size <= 0:
        raise AnswerMarginContractError("preflight dimensions must be positive")
    if not (1 <= settings.selected_group_count <= settings.group_count):
        raise AnswerMarginContractError(
            "selected_group_count must be in [1, group_count]"
        )

    generator = torch.Generator().manual_seed(int(settings.seed))
    patch_tokens = settings.group_count * merge_unit
    hidden_states = torch.randn(
        patch_tokens, settings.hidden_dim, generator=generator, dtype=torch.float32
    ).requires_grad_(True)

    selected = torch.randperm(settings.group_count, generator=generator)[
        : settings.selected_group_count
    ].sort().values

    lm_head = nn.Linear(settings.hidden_dim, settings.vocab_size, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(
            torch.randn(
                settings.vocab_size,
                settings.hidden_dim,
                generator=generator,
                dtype=torch.float32,
            )
            / math.sqrt(settings.hidden_dim)
        )
    lm_head.weight.requires_grad_(False)
    lm_head.eval()

    option_token_ids, correct_index = _deterministic_questions()
    question_count = len(option_token_ids)

    # --- Differentiable path -------------------------------------------------
    compact = differentiable_group_gather(hidden_states, selected, merge_unit)
    readout = compact.mean(dim=0)
    logits = lm_head(readout)
    answer_logits = logits.unsqueeze(0).expand(question_count, settings.vocab_size)
    margins = choice_margin(answer_logits, option_token_ids, correct_index)
    loss = margins.sum()
    loss.backward()

    grad = hidden_states.grad
    grad_present = grad is not None
    grad_finite = bool(torch.isfinite(grad).all().item()) if grad_present else False
    grad_nonzero = bool((grad.abs() > 0).any().item()) if grad_present else False

    # Gather order check (detached, comparison only).
    sample_k = 0
    sample_group = int(selected[sample_k].item())
    order_matches = bool(
        torch.equal(
            compact.detach()[sample_k * merge_unit : (sample_k + 1) * merge_unit],
            hidden_states.detach()[
                sample_group * merge_unit : (sample_group + 1) * merge_unit
            ],
        )
    )

    # --- Negative control: detach must stop the gradient --------------------
    hidden_control = hidden_states.detach().clone().requires_grad_(True)
    compact_control = differentiable_group_gather(
        hidden_control, selected, merge_unit
    ).detach()
    control_logits = lm_head(compact_control.mean(dim=0))
    control_margins = choice_margin(
        control_logits.unsqueeze(0).expand(question_count, settings.vocab_size),
        option_token_ids,
        correct_index,
    )
    control_graph_connected = bool(control_margins.requires_grad)
    control_backward_raises = False
    if not control_graph_connected:
        # The detached path intentionally cannot even start a backward pass;
        # catching that RuntimeError is itself part of the negative control.
        try:
            control_margins.sum().backward()
        except RuntimeError:
            control_backward_raises = True
    negative_control_grad_is_none = hidden_control.grad is None

    # A positional variant must also stay differentiable.
    hidden_positions = hidden_states.detach().clone().requires_grad_(True)
    position_compact = differentiable_group_gather(
        hidden_positions, selected, merge_unit
    )
    position_logits = lm_head(position_compact)
    position_margins = choice_margin_from_positions(
        position_logits,
        [0, 1, 2],
        option_token_ids,
        correct_index,
    )
    position_margins.sum().backward()
    position_grad_present = hidden_positions.grad is not None
    position_grad_nonzero = bool(
        position_grad_present and (hidden_positions.grad.abs() > 0).any().item()
    )

    margins_list = [float(value) for value in margins.detach().tolist()]
    checks = [
        _check(
            "differentiable_group_gather_shape",
            tuple(compact.shape)
            == (settings.selected_group_count * merge_unit, settings.hidden_dim),
            f"compact shape {tuple(compact.shape)}",
        ),
        _check(
            "differentiable_group_gather_requires_grad",
            bool(compact.requires_grad),
            "gathered tensor is connected to the B1 hidden states",
        ),
        _check(
            "differentiable_group_gather_order",
            order_matches,
            "row k*merge_unit+j equals hidden[group_k*merge_unit+j]",
        ),
        _check(
            "choice_margin_shape",
            tuple(margins.shape) == (question_count,),
            f"margins shape {tuple(margins.shape)}",
        ),
        _check(
            "choice_margin_finite",
            all(math.isfinite(value) for value in margins_list),
            f"margins {margins_list}",
        ),
        _check(
            "b1_hidden_grad_present",
            grad_present,
            "backward populated hidden_states.grad",
        ),
        _check("b1_hidden_grad_finite", grad_finite, "gradient is finite"),
        _check("b1_hidden_grad_nonzero", grad_nonzero, "gradient has a non-zero entry"),
        _check(
            "negative_control_detach_blocks_grad",
            negative_control_grad_is_none
            and not control_graph_connected
            and control_backward_raises,
            "detaching the compact tensor disconnects the graph, leaves the "
            "control hidden grad as None and makes backward raise",
        ),
        _check(
            "positional_margin_grad_present",
            position_grad_present and position_grad_nonzero,
            "choice_margin_from_positions keeps a non-zero gradient",
        ),
    ]
    passed = all(item["passed"] for item in checks)
    status = PREFLIGHT_STATUS_PASS if passed else PREFLIGHT_STATUS_FAIL

    return {
        "schema_version": "1.0",
        "date": REPORT_DATE,
        "status": status,
        "scope": (
            "CPU-only engineering feasibility preflight for a future B1 "
            "answer-choice-margin teacher signal. No model weights, no GPU, "
            "no network, no training, no dependency installation."
        ),
        "engineering_feasibility_only": True,
        "is_selector_accuracy_evidence": False,
        "settings": {
            "seed": int(settings.seed),
            "hidden_dim": int(settings.hidden_dim),
            "vocab_size": int(settings.vocab_size),
            "group_count": int(settings.group_count),
            "selected_group_count": int(settings.selected_group_count),
            "merge_unit": merge_unit,
            "patch_tokens": patch_tokens,
            "question_count": question_count,
            "option_counts": [len(options) for options in option_token_ids],
        },
        "gather": {
            "input_shape": [patch_tokens, int(settings.hidden_dim)],
            "output_shape": list(compact.shape),
            "merge_unit": merge_unit,
            "selected_group_indices": [int(value) for value in selected.tolist()],
            "order_preserved": order_matches,
            "uses_detach": False,
            "uses_numpy": False,
            "uses_cpu_rebuild": False,
        },
        "forward": {
            "readout_shape": list(readout.shape),
            "logits_shape": list(logits.shape),
            "answer_logits_shape": list(answer_logits.shape),
        },
        "margin": {
            "definition": "correct_option_logit - max(other_option_logits)",
            "per_question": margins_list,
            "option_token_ids": [list(options) for options in option_token_ids],
            "correct_index": list(correct_index),
            "differentiable": True,
        },
        "gradient": {
            "b1_hidden_grad_present": grad_present,
            "b1_hidden_grad_finite": grad_finite,
            "b1_hidden_grad_nonzero": grad_nonzero,
            "b1_hidden_grad_norm": (
                float(grad.detach().norm().item()) if grad_present else None
            ),
            "positional_grad_present": position_grad_present,
            "positional_grad_nonzero": position_grad_nonzero,
        },
        "negative_control": {
            "description": (
                "Detach the compact tensor before the fake lm_head; the original "
                "B1 hidden states must then receive no gradient."
            ),
            "control_graph_connected": control_graph_connected,
            "control_backward_raises": control_backward_raises,
            "b1_control_grad_is_none": negative_control_grad_is_none,
        },
        "checks": checks,
        "label_hygiene": (
            "Official CV-Bench answer labels must never be used to train a "
            "selector on the full2638 set and then be re-scored on that same "
            "set as if it were held out."
        ),
        "gpu_smoke_constraint": (
            "Any follow-up GPU smoke test may use exactly one image and must "
            "not train."
        ),
        "required_statements": {
            "engineering_feasibility_not_accuracy": (
                "这只是工程可行性，不是 selector 精度证据。"
            ),
            "label_hygiene": (
                "官方答案标签不得在 CV-Bench full2638 上训练再对同集宣称结果。"
            ),
            "gpu_smoke": "后续 GPU smoke 只能一张图，不得训练。",
        },
        "next_gate": (
            "GPU single-image smoke of the frozen B1 forward that reuses "
            "differentiable_group_gather and choice_margin with a real lm_head; "
            "one image only, no training, no CV-Bench accuracy claim."
        ),
        "forbidden_claims": [
            "selector accuracy improvement",
            "CV-Bench full2638 accuracy",
            "FPGA speedup or latency",
            "teacher-signal quality",
        ],
    }


def render_markdown(result: dict[str, Any]) -> str:
    """Render a Chinese Markdown report for the machine-readable result."""
    settings = result["settings"]
    gather = result["gather"]
    forward = result["forward"]
    margin = result["margin"]
    gradient = result["gradient"]
    control = result["negative_control"]
    check_rows = "\n".join(
        f"| {item['name']} | {'✅' if item['passed'] else '❌'} | {item['detail']} |"
        for item in result["checks"]
    )
    per_question = "\n".join(
        f"| Q{index} | {len(options)} | `{options}` | {correct} | `{value:.6f}` |"
        for index, (options, correct, value) in enumerate(
            zip(margin["option_token_ids"], margin["correct_index"], margin["per_question"])
        )
    )
    lines = [
        "# B1 答案 choice-margin 自动微分预检（CPU-only）",
        "",
        f"- 日期：{result['date']}",
        f"- 状态：**{result['status']}**",
        f"- 范围：{result['scope']}",
        "",
        "## 1. 这说明了什么，不说明什么",
        "",
        "本报告只证明**张量与梯度合同可行**：B1 hidden states 能够保持在同一自动微分图中，",
        "经过可微分组 gather、冻结 fake lm_head 与 choice margin，最终得到有限且非零的梯度。",
        "",
        "- `engineering_feasibility_only = true`：这只是工程可行性。",
        "- `is_selector_accuracy_evidence = false`：**这不是 selector 精度证据**，不代表任何",
        "  选择器更准、更差或可用于部署。",
        "- 本预检未加载任何模型权重、未使用 GPU、未联网、未训练、未安装依赖。",
        "",
        "## 2. 可微分组 gather",
        "",
        f"- 输入 `hidden_states`：`{gather['input_shape']}`，`merge_unit = {gather['merge_unit']}`。",
        f"- 输出：`{gather['output_shape']}`，第 `k*merge_unit + j` 行等于 "
        f"`hidden_states[group_k*merge_unit + j]`（`order_preserved = {gather['order_preserved']}`）。",
        f"- 选中组索引：`{gather['selected_group_indices']}`。",
        f"- 实现约束：`uses_detach = {gather['uses_detach']}`、"
        f"`uses_numpy = {gather['uses_numpy']}`、`uses_cpu_rebuild = {gather['uses_cpu_rebuild']}`；",
        "  即不 detach、不 numpy、不在 CPU 重建，梯度可回到 B1 hidden。",
        "",
        "## 3. 前向 shape 与 choice margin",
        "",
        f"- readout `{forward['readout_shape']}` → logits `{forward['logits_shape']}` → "
        f"answer logits `{forward['answer_logits_shape']}`。",
        f"- margin 定义：`{margin['definition']}`，支持每题可变选项数（A-D / A-F / 任意给定 token ids）。",
        "",
        "| 题 | 选项数 | 选项 token ids | 正确索引 | margin |",
        "| --- | --- | --- | --- | --- |",
        per_question,
        "",
        "## 4. 梯度与负对照",
        "",
        f"- B1 hidden 梯度存在：`{gradient['b1_hidden_grad_present']}`，"
        f"有限：`{gradient['b1_hidden_grad_finite']}`，"
        f"非零：`{gradient['b1_hidden_grad_nonzero']}`，"
        f"范数：`{gradient['b1_hidden_grad_norm']}`。",
        f"- 位置选择路径（`choice_margin_from_positions`）梯度存在且非零："
        f"`{gradient['positional_grad_present']}` / `{gradient['positional_grad_nonzero']}`。",
        f"- 负对照：`{control['description']}`",
        f"  - 结果 `b1_control_grad_is_none = {control['b1_control_grad_is_none']}`："
        "detach 后原 B1 hidden **确实得不到梯度**，证明正向梯度来自真实计算图而非偶然。",
        "",
        "## 5. 校验清单",
        "",
        "| 检查 | 通过 | 说明 |",
        "| --- | --- | --- |",
        check_rows,
        "",
        "## 6. 标签卫生与后续门",
        "",
        "- **这只是工程可行性，不是 selector 精度证据**；不得据此报告声称选择器精度、"
        "CV-Bench 分数或任何部署收益。",
        "- **官方答案标签不得在 CV-Bench full2638 上训练再对同集宣称结果**；"
        "若需训练必须使用独立留出/外部冻结集。",
        "- **后续 GPU smoke 只能一张图、不得训练**；只验证张量与梯度合同在真实模型上成立。",
        "",
        f"- 结构化标签卫生声明：{result['label_hygiene']}",
        f"- 结构化 GPU 约束：{result['gpu_smoke_constraint']}",
        f"- **下一道门**：{result['next_gate']}",
        "",
        "禁止据此报告声称：" + "、".join(result["forbidden_claims"]) + "。",
        "",
        "机器可读结果：同目录 JSON（`status`、逐项 `checks`、margin 数值与梯度证据）。",
        "",
    ]
    return "\n".join(lines)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only B1 answer-margin autograd preflight"
    )
    parser.add_argument("--seed", type=int, default=PreflightSettings.seed)
    parser.add_argument(
        "--report-dir", type=Path, default=DEFAULT_REPORT_DIR
    )
    parser.add_argument(
        "--workspace-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args(argv)

    result = run_preflight(PreflightSettings(seed=args.seed))
    workspace_root: Path = args.workspace_root
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

    report_dir = args.report_dir
    if not report_dir.is_absolute():
        report_dir = workspace_root / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{REPORT_STEM}.json"
    md_path = report_dir / f"{REPORT_STEM}.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(result), encoding="utf-8")

    summary = {
        "status": result["status"],
        "json": str(json_path),
        "markdown": str(md_path),
        "checks_passed": sum(1 for item in result["checks"] if item["passed"]),
        "checks_total": len(result["checks"]),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if result["status"] == PREFLIGHT_STATUS_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
