#!/usr/bin/env python3
"""Generate the 2026-10-07 edge576 CV-Bench r4 three-arm preregistration.

CPU-only, read-only over every pinned input.  This script never runs a model.

r4 exists for exactly one reason: the first real r3 GPU smoke attempt
(``dense_edge576``, ``sample_id=1838``) completed model load, one multimodal
warmup and one sample inference and then fail-closed inside
``normalize_record`` with

    frozen dense_edge576 record is missing 'output_token_ids' (fail closed)

because r3 hard-required ``output_token_ids`` while the frozen dense producer
``evaluation/run_hf_shallow_block_pruning.py::run_one`` never emits token ids
(only the real integer count ``output_tokens``).  r4 re-pins the fixed
execution layer:

* ``output_token_ids`` becomes optional and is never fabricated;
* the mandatory, always-auditable ``output_token_count`` is mapped from the
  real frozen fields (``output_token_ids`` length and/or ``output_tokens``,
  which must agree);
* ``output_token_ids_available`` records whether real ids exist;
* core fields (``output_text``, ``selected_group_indices``) and the token
  consistency rules stay fail-closed.

r4 freezes the r3 arms, primary metric, success gates, manifest contract,
shared arm contract and frozen v2 mechanism chain **byte-for-byte unchanged**
and keeps the execution order:

    model identity -> three-arm single-sample smoke all pass ->
    Dense full2638 -> corrected integer L2 full2638 -> v2 full2638 ->
    official scoring + task::source_dataset paired bootstrap.

r3 is kept byte-identical as evidence and marked
``SUPERSEDED_AFTER_FIRST_GPU_SMOKE_FAILURE``; it wrote no record and touched no
full2638 output, so superseding it costs no results.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "2026-10-07-edge576-cvbench-three-arm-development-r4.preregistered.json"
)
DEFAULT_REPORT_JSON = (
    ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cvbench-prereg-r4.json"
)
DEFAULT_REPORT_MD = (
    ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cvbench-prereg-r4.md"
)

# --- superseded r3 evidence (kept byte-identical; never re-generated) ---------
R3_CONFIG = (
    ROOT
    / "configs"
    / "2026-10-07-edge576-cvbench-three-arm-development-r3.preregistered.json"
)
R3_REPORT_JSON = (
    ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cvbench-prereg-r3.json"
)
R3_REPORT_MD = (
    ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cvbench-prereg-r3.md"
)
R3_GENERATOR = ROOT / "scripts" / "preregister_edge576_cvbench_r3_20261007.py"
R3_TEST = ROOT / "evaluation" / "tests" / "test_edge576_cvbench_prereg_r3.py"

R3_FAILURE_JSON = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-07-edge576-cvbench-r3-first-gpu-attempt-failure.json"
)
R3_FAILURE_MD = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-07-edge576-cvbench-r3-first-gpu-attempt-failure.md"
)
R3_SUPERSEDED_JSON = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-07-edge576-cvbench-r3-superseded.json"
)
R3_SUPERSEDED_MD = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-07-edge576-cvbench-r3-superseded.md"
)

# --- execution layer refreshed by the r4 fix ----------------------------------
RUNNER_CONTRACT = ROOT / "evaluation" / "edge576_cvbench_runner_contract.py"
THREE_ARM_RUNNER = ROOT / "evaluation" / "run_edge576_cvbench_three_arm.py"
THREE_ARM_RUNNER_TEST = (
    ROOT / "evaluation" / "tests" / "test_run_edge576_cvbench_three_arm.py"
)
SMOKE_ENTRY = ROOT / "evaluation" / "run_edge576_cvbench_three_arm_smoke.py"
SMOKE_TEST = (
    ROOT / "evaluation" / "tests" / "test_run_edge576_cvbench_three_arm_smoke.py"
)
PREREG_TEST_R4 = ROOT / "evaluation" / "tests" / "test_edge576_cvbench_prereg_r4.py"

#: r3 pin names whose files changed for the token-contract fix.  r4 re-pins the
#: fixed bytes under the same names; every other r3 pin must be byte-identical.
REFRESHED_R3_PINS = {
    "edge576_runner_contract": RUNNER_CONTRACT,
    "edge576_runner": THREE_ARM_RUNNER,
    "edge576_runner_test": THREE_ARM_RUNNER_TEST,
    "smoke_entry": SMOKE_ENTRY,
    "smoke_test": SMOKE_TEST,
}

SMOKE_OUTPUT_DIR = "artifacts/token_selector/2026-10-07-edge576-smoke"
SMOKE_SAMPLE_ID = "1838"
SMOKE_SAMPLE_G = 576
SMOKE_SAMPLE_GRID = [1, 36, 64]
MODEL_PATH = "/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B"
R3_PIN_COUNT = 54


class PreregistrationError(RuntimeError):
    """Raised when a required pin input is missing or drifts (fail closed)."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin(path: Path, *, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise PreregistrationError(f"missing pinned input {path} (fail closed)")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "role": role,
    }


def count_focused_tests(path: Path) -> int:
    if not path.is_file():
        raise PreregistrationError(f"missing focused test {path} (fail closed)")
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("def test_")
    )


def _recompute_r3_pins(r3_config: Mapping[str, Any]) -> dict[str, Any]:
    """Carry forward every r3 pin; refresh only the five fixed files.

    All non-refreshed r3 pins must still hash to their r3 value, so r4 proves it
    changed nothing else.  The refreshed names must still point at the declared
    execution-layer paths.
    """

    pins: dict[str, Any] = {}
    for name, entry in r3_config["pins"].items():
        path = ROOT / entry["path"]
        if not path.is_file():
            raise PreregistrationError(
                f"r3 pin input {name!r} is missing: {path} (fail closed)"
            )
        actual = sha256_file(path)
        if name in REFRESHED_R3_PINS:
            expected = REFRESHED_R3_PINS[name]
            if path.resolve() != expected.resolve():
                raise PreregistrationError(
                    f"r3 pin {name!r} path {path} != refreshed path {expected} "
                    "(fail closed)"
                )
            pins[name] = {
                "path": str(expected.relative_to(ROOT)),
                "sha256": actual,
                "role": entry.get("role"),
            }
            continue
        if actual != entry["sha256"]:
            raise PreregistrationError(
                f"r3 pin {name!r} drifted: {actual} != {entry['sha256']} "
                "(fail closed)"
            )
        pins[name] = {
            "path": entry["path"],
            "sha256": actual,
            "role": entry.get("role"),
        }
    return pins


def build_config() -> dict[str, Any]:
    r3_config = json.loads(R3_CONFIG.read_text(encoding="utf-8"))
    pins = _recompute_r3_pins(r3_config)
    pins.update(
        {
            "r3_first_gpu_attempt_failure_json": pin(
                R3_FAILURE_JSON,
                role="r3_first_gpu_smoke_attempt_failure_evidence",
            ),
            "r3_first_gpu_attempt_failure_md": pin(
                R3_FAILURE_MD,
                role="r3_first_gpu_smoke_attempt_failure_evidence_md",
            ),
            "r3_superseded_json": pin(
                R3_SUPERSEDED_JSON, role="r3_superseded_evidence"
            ),
            "r3_superseded_md": pin(
                R3_SUPERSEDED_MD, role="r3_superseded_evidence_md"
            ),
            "r3_prereg_config": pin(R3_CONFIG, role="superseded_r3_config"),
            "r3_prereg_report_json": pin(
                R3_REPORT_JSON, role="superseded_r3_report_json"
            ),
            "r3_prereg_report_md": pin(
                R3_REPORT_MD, role="superseded_r3_report_md"
            ),
            "prereg_generator_r4": pin(
                Path(__file__).resolve(), role="edge576_prereg_r4_generator"
            ),
            "prereg_test_r4": pin(
                PREREG_TEST_R4, role="edge576_prereg_r4_test"
            ),
        }
    )

    supersede_reason = (
        "r3 hard-required 'output_token_ids' for every arm, but the frozen dense "
        "producer evaluation/run_hf_shallow_block_pruning.py::run_one never "
        "emits token ids (only the real integer count 'output_tokens'). The "
        "first real r3 GPU smoke attempt (dense_edge576, sample_id=1838) "
        "completed model load, one warmup and one sample inference and then "
        "fail-closed inside normalize_record before writing any file. r3 wrote "
        "no record and touched no full2638 output, so superseding it costs no "
        "results; this is the only reason r3 is superseded."
    )

    execution_order = [
        {
            "step": 1,
            "id": "model_identity",
            "description": (
                "Verify the pinned model identity manifest and every declared "
                "model file (relative path + byte size + SHA256, including "
                "model.safetensors) before any CUDA, scorer or weight load."
            ),
            "fail_closed": True,
            "executed": False,
        },
        {
            "step": 2,
            "id": "three_arm_single_sample_smoke",
            "description": (
                "Run the independent single-sample smoke for dense_edge576, "
                "corrected_l2_integer_b1_k96 and v2_edge576_b1_k96 with "
                "evaluation/run_edge576_cvbench_three_arm_smoke.py; each "
                "invocation writes only its own file under "
                f"{SMOKE_OUTPUT_DIR} and must save a complete record plus a "
                "smoke summary. The dense arm has no real token ids; its record "
                "must instead carry output_token_count and "
                "output_token_ids_available=false."
            ),
            "arms": list(r3_config["arms"]),
            "sample_id": SMOKE_SAMPLE_ID,
            "must_all_pass": True,
            "blocks_full2638_until_pass": True,
            "executed": False,
        },
        {
            "step": 3,
            "id": "dense_edge576_full2638",
            "arm": "dense_edge576",
            "output_path": r3_config["arms"]["dense_edge576"]["output_path"],
            "executed": False,
        },
        {
            "step": 4,
            "id": "corrected_l2_integer_b1_k96_full2638",
            "arm": "corrected_l2_integer_b1_k96",
            "output_path": r3_config["arms"]["corrected_l2_integer_b1_k96"][
                "output_path"
            ],
            "executed": False,
        },
        {
            "step": 5,
            "id": "v2_edge576_b1_k96_full2638",
            "arm": "v2_edge576_b1_k96",
            "output_path": r3_config["arms"]["v2_edge576_b1_k96"]["output_path"],
            "executed": False,
        },
        {
            "step": 6,
            "id": "official_scoring_and_paired_bootstrap",
            "description": (
                "Score every completed arm with evaluation/cv_bench_metric_v2.py "
                "(explicit parsed_answer field, parse failures count as wrong) "
                "and run the task::source_dataset paired bootstrap (seed "
                "20261007, 5000 repeats, 95%) before any external validation "
                "or hardware claim."
            ),
            "executed": False,
        },
    ]

    smoke_commands = [
        (
            "python evaluation/run_edge576_cvbench_three_arm_smoke.py"
            f" --arm {arm}"
            f" --sample-id {SMOKE_SAMPLE_ID}"
            f" --model {MODEL_PATH}"
            " --model-label RoboBrain2.5-4B-BF16"
            " --manifest datasets/2026-10-07-cv-bench-edge576-full2638.jsonl"
            f" --output {SMOKE_OUTPUT_DIR}/{arm}_{SMOKE_SAMPLE_ID}.json"
        )
        for arm in ("dense_edge576", "corrected_l2_integer_b1_k96", "v2_edge576_b1_k96")
    ]

    config: dict[str, Any] = {
        "version": "2026-10-07-edge576-cvbench-three-arm-development-r4",
        "revision": "r4",
        "date": "2026-10-07",
        "status": "PREREGISTERED_NOT_EXECUTED",
        "arm_status": "NOT_EXECUTED",
        "executed": False,
        "future_only": True,
        "retroactive": False,
        "gpu_runs": 0,
        "purpose": (
            "Token-contract fix revision of the edge deployment study: r4 keeps "
            "the r3 three-arm comparison (Dense@edge576, corrected integer "
            "L2@B1/K96, frozen existing conservative_l2_guarded_rerank_v2@B1/K96) "
            "byte-for-byte unchanged, makes output_token_ids optional (never "
            "fabricated), requires the auditable output_token_count mapped from "
            "the real frozen fields, and re-pins the fixed runner/smoke "
            "execution layer and the r3 failure/supersede evidence."
        ),
        "measurement_class": r3_config["measurement_class"],
        "development_retest_statement": r3_config["development_retest_statement"],
        "development_set_statement": r3_config["development_set_statement"],
        "non_retroactive_statement": r3_config["non_retroactive_statement"],
        "supersedes": {
            "path": str(R3_CONFIG.relative_to(ROOT)),
            "sha256": sha256_file(R3_CONFIG),
            "report_json": str(R3_REPORT_JSON.relative_to(ROOT)),
            "report_json_sha256": sha256_file(R3_REPORT_JSON),
            "report_md": str(R3_REPORT_MD.relative_to(ROOT)),
            "report_md_sha256": sha256_file(R3_REPORT_MD),
            "generator": str(R3_GENERATOR.relative_to(ROOT)),
            "generator_sha256": sha256_file(R3_GENERATOR),
            "status": "SUPERSEDED_AFTER_FIRST_GPU_SMOKE_FAILURE",
            "executed": True,
            "gpu_runs": 1,
            "arm_runs_completed": 0,
            "smoke_files_written": 0,
            "records_written": 0,
            "full2638_outputs_touched": False,
            "never_executed": False,
            "must_not_be_run": True,
            "reason": supersede_reason,
            "failure_evidence": str(R3_FAILURE_JSON.relative_to(ROOT)),
            "failure_error": (
                "frozen dense_edge576 record is missing 'output_token_ids' "
                "(fail closed)"
            ),
            "superseded_evidence": str(R3_SUPERSEDED_JSON.relative_to(ROOT)),
        },
        "execution_order": execution_order,
        "record_contract": {
            "core_required_fields": ["output_text", "selected_group_indices"],
            "optional_fields": ["output_token_ids"],
            "required_audit_fields": [
                "output_token_count",
                "output_token_ids_available",
            ],
            "output_token_count_mapping": (
                "From the real frozen fields only: output_token_ids length "
                "and/or the integer output_tokens; both must agree when both "
                "are present; neither present fails closed."
            ),
            "token_ids_never_fabricated": True,
            "token_field_consistency_fail_closed": True,
            "dense_producer_emits_output_token_ids": False,
            "l2_producer_emits_output_token_ids": True,
            "v2_producer_emits_output_token_ids": True,
            "dense_producer_count_field": "output_tokens",
            "metrics_require_token_ids": False,
            "metrics_source_of_truth": (
                "evaluation/build_cv_bench_pruning_report.parse_choice over "
                "output_text plus the official answer"
            ),
        },
        "smoke_contract": {
            "entry": str(SMOKE_ENTRY.relative_to(ROOT)),
            "output_dir": SMOKE_OUTPUT_DIR,
            "output_pattern": "<arm>_<sample_id>.json",
            "one_sample_per_invocation": True,
            "writes_only_the_smoke_file": True,
            "must_not_touch_full2638_outputs": True,
            "full2638_outputs": [
                r3_config["arms"][arm]["output_path"]
                for arm in ("dense_edge576", "corrected_l2_integer_b1_k96",
                            "v2_edge576_b1_k96")
            ],
            "required_checks": [
                "17 frozen dependency SHA256 pins",
                "arm mechanism identity",
                "--model identity (relative path + byte size + SHA256 incl. model.safetensors)",
                "real multimodal warmup with max_new_tokens=1",
                "actual processor image_grid_thw equals the edge576 manifest grid",
                "complete normalised record plus smoke summary saved",
                "optional output_token_ids never fabricated; output_token_count present",
            ],
            "real_warmup": {
                "max_new_tokens": 1,
                "record_discarded": True,
                "cuda_synchronize": True,
            },
            "warmup_not_counted_in_record": True,
            "actual_grid_assertion": True,
            "selection_algorithm_copied": False,
            "smoke_sample_id": SMOKE_SAMPLE_ID,
            "smoke_sample_g": SMOKE_SAMPLE_G,
            "smoke_sample_grid": SMOKE_SAMPLE_GRID,
            "default_smoke_dir_disjoint_from_full2638": True,
            "output_token_ids_optional": True,
            "output_token_count_required": True,
            "dense_arm_smoke_expects_missing_token_ids": True,
            "commands": smoke_commands,
        },
        "edge_runtime_domain": copy.deepcopy(r3_config["edge_runtime_domain"]),
        "manifest_contract": copy.deepcopy(r3_config["manifest_contract"]),
        "shared_arm_contract": copy.deepcopy(r3_config["shared_arm_contract"]),
        "arms": copy.deepcopy(r3_config["arms"]),
        "frozen_candidate_mechanism": copy.deepcopy(
            r3_config["frozen_candidate_mechanism"]
        ),
        "primary_metric": copy.deepcopy(r3_config["primary_metric"]),
        "success_gates": copy.deepcopy(r3_config["success_gates"]),
        "model_identity": copy.deepcopy(r3_config["model_identity"]),
        "forbidden_claims": list(r3_config["forbidden_claims"])
        + [
            "the r3 first GPU smoke attempt produced any record or smoke file",
            "r3 touched any full2638 output",
            "output_token_ids may be fabricated for the dense arm",
        ],
        "next_allowed_action": (
            "Follow the execution_order exactly: verify the model identity, run "
            "the three-arm single-sample smoke until all three pass, and only "
            "then run Dense full2638, corrected integer L2 full2638 and v2 "
            "full2638 to the pinned output paths; score with cv_bench_metric_v2 "
            "and the task::source_dataset paired bootstrap before any external "
            "validation or hardware claim."
        ),
        "runner_adaptation_entry": {
            "dense": r3_config["runner_adaptation_entry"]["dense"],
            "corrected_l2_integer": r3_config["runner_adaptation_entry"][
                "corrected_l2_integer"
            ],
            "v2_candidate": r3_config["runner_adaptation_entry"]["v2_candidate"],
            "runner_contract": str(RUNNER_CONTRACT.relative_to(ROOT)),
            "three_arm_runner": str(THREE_ARM_RUNNER.relative_to(ROOT)),
            "smoke_entry": str(SMOKE_ENTRY.relative_to(ROOT)),
            "required_edge576_inputs": r3_config["runner_adaptation_entry"][
                "required_edge576_inputs"
            ],
            "selection_algorithm_copied": False,
            "status": "ADAPTER_AND_SMOKE_FIXED_NO_MODEL_RUN",
        },
        "verification": {
            "py_compile": [
                str(SMOKE_ENTRY.relative_to(ROOT)),
                str(SMOKE_TEST.relative_to(ROOT)),
                str(Path(__file__).resolve().relative_to(ROOT)),
                str(PREREG_TEST_R4.relative_to(ROOT)),
            ],
            "smoke_focused_tests": count_focused_tests(SMOKE_TEST),
            "runner_focused_tests": count_focused_tests(THREE_ARM_RUNNER_TEST),
            "prereg_r4_focused_tests": count_focused_tests(PREREG_TEST_R4),
            "pins_recomputed_on_disk": True,
            "gpu_runs": 0,
            "weights_loaded": False,
            "model_loaded": False,
            "network": False,
            "training": False,
        },
        "pins": pins,
    }
    return config


def render_markdown(config: Mapping[str, Any]) -> str:
    pins = config["pins"]
    counts = {
        "r3_deps": R3_PIN_COUNT,
        "new": len(pins) - R3_PIN_COUNT,
        "total": len(pins),
    }
    smoke = config["smoke_contract"]
    step_cn = {
        1: (
            "模型身份：在任何 CUDA/scorer/权重加载之前，校验 pinned 模型身份清单与每个"
            "声明文件（相对路径 + 字节数 + SHA256，含 `model.safetensors`）。"
        ),
        2: (
            "三臂单样本 smoke：用 `evaluation/run_edge576_cvbench_three_arm_smoke.py` "
            "依次对 `dense_edge576`、`corrected_l2_integer_b1_k96`、`v2_edge576_b1_k96` "
            "各跑一个样本（`sample_id=1838`）；每次调用只写自己的 smoke 文件，必须保存"
            "完整记录 + smoke 摘要。Dense 无真实 token id，记录必须带 "
            "`output_token_count` 且 `output_token_ids_available=false`。"
            "**三臂全部通过前不得进入任何 full2638 运行。**"
        ),
        3: "Dense@edge576 full2638（写 pinned 输出路径）。",
        4: "修正整数 L2@B1/K96 full2638（写 pinned 输出路径）。",
        5: "v2@B1/K96 full2638（写 pinned 输出路径）。",
        6: (
            "官方评分与分层配对 bootstrap：用 `evaluation/cv_bench_metric_v2.py` 对每条完成臂"
            "评分（显式 `parsed_answer`、解析失败计错），并做 `task::source_dataset` 配对 "
            "bootstrap（seed 20261007、5000 次、95%）之后，才允许外部验证或硬件主张。"
        ),
    }
    lines = [
        "# edge576 CV-Bench 三臂预注册 r4（2026-10-07，token 契约修复层）",
        "",
        f"- 状态：`{config['status']}`（`executed=false`、`gpu_runs=0`、仅 CPU/只读、未加载模型权重）；"
        "三臂状态 `NOT_EXECUTED`",
        f"- 取代 r3：`{config['supersedes']['path']}` SHA `{config['supersedes']['sha256']}`；"
        f"r3 `SUPERSEDED_AFTER_FIRST_GPU_SMOKE_FAILURE`（首次 GPU smoke 于归一化阶段 fail-closed，"
        f"写出记录 0、未触碰 full2638），字节全部保留且不得再运行",
        f"- 失败证据：`{config['supersedes']['failure_evidence']}`；"
        f"作废记录：`{config['supersedes']['superseded_evidence']}`",
        f"- pin 总数：**{counts['total']}** = r3 既有 {counts['r3_deps']} 项 + 新增 {counts['new']} 项",
        "",
        "## 唯一作废 r3 的理由",
        "",
        "**r3 硬性要求 `output_token_ids`，而冻结 dense 生产者从不返回 token id。** "
        "`evaluation/run_hf_shallow_block_pruning.py::run_one` 只返回 `output_text`、真实整数计数 "
        "`output_tokens` 和 `selected_group_indices`；只有冻结 L2/v2 生产者返回 `output_token_ids`。"
        "r3 的第一次真实执行（`dense_edge576`、`sample_id=1838`、`cuda:0`）在模型加载、一次 warmup、"
        "一次样本推理之后，于写文件前 fail-closed："
        "`frozen dense_edge576 record is missing 'output_token_ids' (fail closed)`。"
        "r3 **未写任何记录、未触碰任何 full2638 输出**，作废不损失任何结果；"
        "这是 r3 被取代的**唯一**理由。",
        "",
        "> 机器可读英文原文：" + config["supersedes"]["reason"],
        "",
        "## r4 冻结不变（与 r3 逐字节一致）",
        "",
        "- 三臂：`dense_edge576` / `corrected_l2_integer_b1_k96` / `v2_edge576_b1_k96`；",
        "- 主指标：官方 `cv_bench_accuracy = ((ADE20K + COCO)/2 + Omni3D)/2`；",
        "- 成功门、`task::source_dataset` 分层配对 bootstrap（seed 20261007、5000 次、95%）、"
        "K96 布局（87 real + 9 context，`G<=96` identity bypass）、2638 唯一完整输出"
        "（`parsed_answer` 可 `null`、解析失败计错并报告、无 100% 解析门）、无标签/答案泄漏；",
        "- 三条 full2638 输出路径不变；edge576 运行域与处理参数不变。",
        "",
        "## r4 修的 token 契约（唯一改动）",
        "",
        f"- `output_token_ids` **可选**，只有冻结生产者真实返回时才记录，**绝不伪造**；",
        "- 必备可审计字段 `output_token_count` 由真实字段规范映射："
        "有 `output_token_ids` 取其真实长度、有 `output_tokens` 取其真实整数值、"
        "两者都在必须一致、两者都无则 fail-closed；",
        "- 必备 `output_token_ids_available`（bool）标注 token id 是否真实存在；",
        "- 核心字段 `output_text`、`selected_group_indices` 缺失仍 fail-closed；",
        "- 主指标不消费 token id（只用 `output_text` + 官方答案诚实解析）。",
        "",
        "## 独立单样本 smoke 执行层",
        "",
        f"- 入口：`{smoke['entry']}`（SHA `{pins['smoke_entry']['sha256']}`）；",
        f"- 测试：`{pins['smoke_test']['path']}`（SHA `{pins['smoke_test']['sha256']}`）；",
        f"- 唯一输出目录：`{smoke['output_dir']}`，文件名 `{smoke['output_pattern']}`；"
        "**绝不触碰** full2638 输出：",
    ]
    for full in smoke["full2638_outputs"]:
        lines.append(f"  - `{full}`")
    lines += [
        f"- 每次只跑一个 arm + 一个 `sample_id`（`{smoke['smoke_sample_id']}`，"
        f"G={smoke['smoke_sample_g']}，grid={smoke['smoke_sample_grid']}）；",
        "- 复用（不复制）`load_edge576_manifest` / `verify_dependency_pins` / "
        "`mechanism_identity` / `verify_model_identity` / `Edge576Processor` / "
        "`build_arm_state` / `run_multimodal_warmup` / `run_one_for_arm` / `normalize_record`；",
        "- 必须执行：17 项依赖 pin + 机制身份、模型身份（含 `model.safetensors`）、"
        "真实 warmup（`max_new_tokens=1`、记录丢弃、`torch.cuda.synchronize`）、"
        "实际 `image_grid_thw` 断言；保存一个完整正式样本记录 + smoke 摘要；",
        "- 禁止复制任何选择算法（`selection_algorithm_copied=false`）。",
        "",
        "## 明确执行顺序（r4 冻结）",
        "",
    ]
    for step in config["execution_order"]:
        lines.append(f"{step['step']}. `{step['id']}` — {step_cn[step['step']]}")
    lines += [
        "",
        "**硬门**：第 2 步三臂单样本 smoke 必须全部通过，否则不得开始任何 full2638 运行。",
        "",
        "## 新增/继承 pin",
        "",
        "| pin | 文件 | SHA256 |",
        "| --- | --- | --- |",
    ]
    for name in (
        "edge576_runner_contract",
        "edge576_runner",
        "edge576_runner_test",
        "smoke_entry",
        "smoke_test",
        "r3_first_gpu_attempt_failure_json",
        "r3_first_gpu_attempt_failure_md",
        "r3_superseded_json",
        "r3_superseded_md",
        "r3_prereg_config",
        "r3_prereg_report_json",
        "r3_prereg_report_md",
        "prereg_generator_r4",
        "prereg_test_r4",
    ):
        entry = pins[name]
        lines.append(f"| `{name}` | `{entry['path']}` | `{entry['sha256']}` |")
    lines += [
        "",
        f"（其余 r3 冻结依赖现场重算后原样继承；r3 未重新生成，字节保留。）",
        "",
        "## 验证",
        "",
        f"- `py_compile`：{len(config['verification']['py_compile'])} 个新增/修改文件；",
        f"- runner 聚焦测试：**{config['verification']['runner_focused_tests']} OK**；",
        f"- smoke 聚焦测试：**{config['verification']['smoke_focused_tests']} OK**；",
        f"- r4 预注册聚焦测试：**{config['verification']['prereg_r4_focused_tests']} OK**；",
        "- 全部 pin 现场重算 `pins_recomputed_on_disk=true`；",
        "- `gpu_runs=0`、`weights_loaded=false`、`model_loaded=false`、未联网、未训练。",
        "",
        "## 边界与风险",
        "",
        "- 本轮**不启动模型、不联网、不训练**；r4 与修复后的 smoke 均未执行任何 GPU 运行；",
        "- r3 的失败尝试消耗了 1 次 GPU smoke，未产出任何记录，不构成精度/加速/部署证据；",
        "- 模型身份核验会在加载前流式读取并 SHA256 整个 `model.safetensors`（约 9.65 GB），"
        "每次正式运行增加一次性 CPU/IO 开销；",
        "- 三臂 smoke 与 full2638 的 GPU 路径尚未在修复后与真实模型联调；",
        "- full2638 是反复观测的发展集，通过门也只允许一次未触碰外部基准确认，"
        "不构成发表/部署/加速主张；",
        "- r3 及其预注册、报告、生成器、测试、失败/作废证据字节全部保留，不得重新生成。",
        "",
    ]
    return "\n".join(lines)


def check_pins(config_path: Path) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for name, entry in config["pins"].items():
        path = ROOT / entry["path"]
        if not path.is_file():
            mismatches.append(f"{name}: missing {path}")
            continue
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            mismatches.append(f"{name}: {actual} != {entry['sha256']}")
    if mismatches:
        print("PIN_AUDIT_FAIL", file=sys.stderr)
        for item in mismatches:
            print(f"  {item}", file=sys.stderr)
        return 2
    print(
        f"PIN_AUDIT_OK pins={len(config['pins'])} "
        f"config_sha256={sha256_file(config_path)}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        return check_pins(args.config)

    config = build_config()
    config["generated_at"] = datetime.now(timezone.utc).isoformat()
    config["generator"] = "scripts/preregister_edge576_cvbench_r4_20261007.py"
    config["generator_sha256"] = sha256_file(Path(__file__).resolve())

    # Self-check: every declared pin must re-verify on the spot.
    for name, entry in config["pins"].items():
        observed = sha256_file(ROOT / entry["path"])
        if observed != entry["sha256"]:
            raise PreregistrationError(
                f"pin {name} mismatch: {observed} != {entry['sha256']}"
            )

    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "status": config["status"],
        "arm_status": config["arm_status"],
        "revision": config["revision"],
        "date": config["date"],
        "gpu_runs": 0,
        "executed": False,
        "pins_verified": True,
        "pin_count": len(config["pins"]),
        "pins": config["pins"],
        "config": str(args.config.relative_to(ROOT)),
        "config_sha256": sha256_file(args.config),
        "supersedes": config["supersedes"],
        "execution_order": config["execution_order"],
        "record_contract": config["record_contract"],
        "smoke_contract": config["smoke_contract"],
        "manifest_contract": config["manifest_contract"],
        "shared_arm_contract": config["shared_arm_contract"],
        "arms": config["arms"],
        "frozen_candidate_mechanism": config["frozen_candidate_mechanism"],
        "primary_metric": config["primary_metric"],
        "success_gates": config["success_gates"],
        "runner_adaptation_entry": config["runner_adaptation_entry"],
        "verification": config["verification"],
        "next_allowed_action": config["next_allowed_action"],
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report_md.write_text(render_markdown(config), encoding="utf-8")
    print(
        "EDGE576_R4_PREREGISTERED "
        f"config={args.config.name} sha256={report['config_sha256']} "
        f"pins={report['pin_count']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreregistrationError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
