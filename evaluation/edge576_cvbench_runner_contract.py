#!/usr/bin/env python3
"""Shared, CPU-only edge576 three-arm run contract (no model, no CUDA).

This module is the single source of truth for the 2026-10-07 edge576 CV-Bench
development run domain frozen by
``configs/2026-10-07-edge576-cvbench-three-arm-development-r2.preregistered.json``:

* the three arms ``dense_edge576`` / ``corrected_l2_integer_b1_k96`` /
  ``v2_edge576_b1_k96`` and their pinned r2 output paths;
* the edge576 manifest contract (2638 unique samples, ``G<=576``, per-sample
  ``image_grid_thw`` recorded by the authoritative processor);
* the fixed processor budget ``min_pixels=65536`` / ``max_pixels=589824`` and a
  processor wrapper that injects exactly those bounds into the frozen runners;
* per-sample fail-closed ``image_grid_thw`` assertion helpers;
* label-free selector input sanitisation;
* append-only, fail-closed resume validation;
* SHA256 pin verification of the frozen dependencies and a mechanism-identity
  check that the third arm really is the existing
  ``conservative_l2_guarded_rerank_v2`` mechanism;
* a fail-closed ``--model`` identity verifier that re-checks the pinned
  ``artifacts/provenance/robobrain25-4b-model-identity.json`` and every declared
  model file (relative path, byte size and SHA256, including
  ``model.safetensors``) before any weight is opened.

It contains **no selection algorithm**.  Every selection / scorer / forward
implementation stays in the frozen modules that the runner imports; this file
only carries the runtime contract around them.  Importing it must never load
model weights or touch CUDA.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.build_cv_bench_pruning_report import answer_letter, parse_choice

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Frozen edge576 domain constants (must match the r2 preregistration)
# ---------------------------------------------------------------------------
EDGE576_MANIFEST = (
    ROOT / "datasets" / "2026-10-07-cv-bench-edge576-full2638.jsonl"
)
EDGE576_MANIFEST_SHA256 = (
    "b5355ba1636011b4af101ace16c170b061ecf44a69e5c75fe2a64f4288c7dd1f"
)
EDGE576_SAMPLE_COUNT = 2638

R2_CONFIG = (
    ROOT
    / "configs"
    / "2026-10-07-edge576-cvbench-three-arm-development-r2.preregistered.json"
)

MIN_PIXELS = 65536
MAX_PIXELS = 589824
BUDGET_G = 576
PATCH_SIZE = 16
MERGE_SIZE = 2
MERGE_UNIT = 4
K_BUDGET = 96
REGION_GRID = 3
LOCAL_CANDIDATE_CAP = 16
MAX_SWAPS = 16
MAX_NEW_TOKENS = 16
DTYPE = "bf16"
ATTENTION_IMPLEMENTATION = "sdpa"
DECODING = "greedy_argmax"
DO_SAMPLE = False
IDENTITY_BYPASS_G_LE = 96

#: Official CV-Bench sources (must equal ``cv_bench_metric_v2.CV_BENCH_SOURCES``).
CV_BENCH_SOURCES = ("ADE20K", "COCO", "Omni3D")

# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
ARM_DENSE = "dense_edge576"
ARM_L2 = "corrected_l2_integer_b1_k96"
ARM_V2 = "v2_edge576_b1_k96"
ARMS = (ARM_DENSE, ARM_L2, ARM_V2)

ARM_MECHANISM_IDS = {
    ARM_DENSE: "dense_edge576",
    ARM_L2: "corrected_l2_integer",
    ARM_V2: "conservative_l2_guarded_rerank_v2",
}

#: None of the three arms invents a new mechanism; the third arm is the frozen
#: existing ``conservative_l2_guarded_rerank_v2`` re-run in the edge576 domain.
ARM_IS_NEW_MECHANISM = {ARM_DENSE: False, ARM_L2: False, ARM_V2: False}

ARM_OUTPUT_PATHS = {
    ARM_DENSE: (
        ROOT
        / "artifacts"
        / "token_selector"
        / "2026-10-07-edge576-cvbench"
        / "dense_edge576_full2638.jsonl"
    ),
    ARM_L2: (
        ROOT
        / "artifacts"
        / "token_selector"
        / "2026-10-07-edge576-cvbench"
        / "corrected_l2_integer_k96_full2638.jsonl"
    ),
    ARM_V2: (
        ROOT
        / "artifacts"
        / "token_selector"
        / "2026-10-07-edge576-cvbench"
        / "v2_edge576_k96_full2638.jsonl"
    ),
}

#: Frozen files the runner directly depends on; every one is SHA256-pinned in
#: the r2 config and re-verified by ``--validate-only``/``--dry-run``.
RUNNER_DEPENDENCY_PINS = (
    "edge576_manifest",
    "edge576_summary",
    "source_full_manifest",
    "corrected_l2_runner",
    "v2_mechanism_core",
    "v2_integration",
    "v2_runner",
    "v2_summarizer",
    "integer_token_selector_core",
    "integer_corrected_l2_core",
    "integer_b1_integration_core",
    "integer_current_selector_core",
    "v1_layout_rebuild_core",
    "cv_bench_metric_v2",
    "evaluator",
    "metrics",
    "model_identity_manifest",
)

# ---------------------------------------------------------------------------
# Model identity contract (fail-closed, CPU-only, no weights loaded)
# ---------------------------------------------------------------------------
#: Frozen manifest that pins every local model/processor/tokenizer file by
#: relative path, byte size and SHA256.  It is itself pinned in the r2 config
#: dependency list (``model_identity_manifest``).
MODEL_IDENTITY_MANIFEST = (
    ROOT / "artifacts" / "provenance" / "robobrain25-4b-model-identity.json"
)
MODEL_IDENTITY_MANIFEST_SHA256 = (
    "bbfcda188605f54a9c7fea83e7fc2a46eca8ac635c086d23861b294c9950f8e3"
)
MODEL_IDENTITY_KIND = "robobrain_model_identity"
MODEL_IDENTITY_SCHEMA_VERSION = "1.0"
MODEL_IDENTITY_HASH_ALGORITHM = "sha256"
#: At least one weight shard must be present and hash-verified before any load.
REQUIRED_MODEL_WEIGHT_SHARDS = ("model.safetensors",)

#: Fields the selector may see.  ``answer``/``ground_truth`` and every derived
#: correctness field are deliberately absent so labels can never leak in.
SELECTOR_SAMPLE_KEYS = (
    "sample_id",
    "task",
    "type",
    "source",
    "source_dataset",
    "image",
    "prompt",
    "choices",
    "max_new_tokens",
    "image_grid_thw",
    "full_llm_visual_tokens",
    "full_patch_tokens",
)

FORBIDDEN_SELECTOR_KEYS = (
    "answer",
    "ground_truth",
    "correct",
    "parsed_answer",
    "official_answer",
    "parse_success",
)

#: Fields every normalised edge576 record must carry for a safe resume.  The
#: dense producer emits no token ids, so ``output_token_ids`` is optional and is
#: replaced by the mandatory, always-auditable ``output_token_count``.
REQUIRED_EDGE_RECORD_FIELDS = (
    "sample_id",
    "arm",
    "mechanism_id",
    "manifest_sha256",
    "min_pixels",
    "max_pixels",
    "image_grid_thw",
    "full_llm_visual_tokens",
    "output_text",
    "output_token_count",
    "output_token_ids_available",
    "parsed_answer",
    "correct",
    "source",
    "task",
    "prompt_sha256",
    "max_new_tokens",
    "do_sample",
    "decoding",
    "dtype",
    "attention_implementation",
    "output_group_count",
    "identity_bypass",
    "run_mode",
    "record_sha256",
)

ARM_EXTRA_RECORD_FIELDS = {
    ARM_DENSE: ("actual_token_retention", "llm_visual_tokens", "selected_group_indices"),
    ARM_L2: (
        "requested_budget",
        "region_grid",
        "local_candidate_cap",
        "selected_group_indices",
    ),
    ARM_V2: (
        "requested_budget",
        "region_grid",
        "local_candidate_cap",
        "max_swaps",
        "selected_group_indices",
        "swap_count",
        "utility_formula",
        "legal_gain_formula",
        "uses_labels",
        "uses_answer_tokens",
        "uses_gradients",
    ),
}


class Edge576ContractError(RuntimeError):
    """Raised when the edge576 run contract is violated (fail closed)."""


# ---------------------------------------------------------------------------
# Output token contract (real fields only; token ids are never fabricated)
# ---------------------------------------------------------------------------
#: The frozen dense producer (``run_hf_shallow_block_pruning.run_one``) emits
#: only the integer count ``output_tokens`` (== ``len(generated)``) and never
#: emits token ids.  The frozen L2 and v2 producers emit both
#: ``output_token_ids`` and ``output_tokens``.  Token ids are therefore optional
#: and must never be synthesised; the mandatory ``output_token_count`` is mapped
#: from whatever real field exists.
OUTPUT_TOKEN_IDS_FIELD = "output_token_ids"
OUTPUT_TOKEN_COUNT_FIELD = "output_token_count"
OUTPUT_TOKEN_IDS_AVAILABLE_FIELD = "output_token_ids_available"
FROZEN_OUTPUT_TOKEN_COUNT_FIELD = "output_tokens"


def resolve_output_token_fields(
    frozen_record: Mapping[str, Any],
) -> tuple[list[int] | None, int]:
    """Return ``(token_ids_or_None, count)`` from real frozen fields only.

    Fail-closed rules:

    * token ids, when present, must be a list/tuple of non-bool integers;
    * the frozen count, when present, must be a non-negative non-bool integer;
    * if both are present they must agree on the count;
    * if neither is present the record has no auditable output length.
    """

    raw_ids = frozen_record.get(OUTPUT_TOKEN_IDS_FIELD)
    raw_count = frozen_record.get(FROZEN_OUTPUT_TOKEN_COUNT_FIELD)

    token_ids: list[int] | None = None
    if raw_ids is not None:
        if not isinstance(raw_ids, (list, tuple)):
            raise Edge576ContractError(
                f"{OUTPUT_TOKEN_IDS_FIELD} must be a list of ints, got "
                f"{type(raw_ids).__name__} (fail closed)"
            )
        token_ids = []
        for value in raw_ids:
            if isinstance(value, bool) or not isinstance(value, int):
                raise Edge576ContractError(
                    f"{OUTPUT_TOKEN_IDS_FIELD} contains a non-integer "
                    f"{value!r} (fail closed)"
                )
            token_ids.append(int(value))

    frozen_count: int | None = None
    if raw_count is not None:
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            raise Edge576ContractError(
                f"{FROZEN_OUTPUT_TOKEN_COUNT_FIELD} must be an int, got "
                f"{type(raw_count).__name__} (fail closed)"
            )
        if int(raw_count) < 0:
            raise Edge576ContractError(
                f"{FROZEN_OUTPUT_TOKEN_COUNT_FIELD}={raw_count!r} is negative "
                "(fail closed)"
            )
        frozen_count = int(raw_count)

    if token_ids is not None:
        if frozen_count is None:
            frozen_count = len(token_ids)
        elif frozen_count != len(token_ids):
            raise Edge576ContractError(
                f"{OUTPUT_TOKEN_IDS_FIELD} length {len(token_ids)} != "
                f"{FROZEN_OUTPUT_TOKEN_COUNT_FIELD} {frozen_count} (fail closed)"
            )
    if frozen_count is None:
        raise Edge576ContractError(
            f"record carries neither {OUTPUT_TOKEN_IDS_FIELD!r} nor "
            f"{FROZEN_OUTPUT_TOKEN_COUNT_FIELD!r}; no auditable output length "
            "(fail closed)"
        )
    return token_ids, frozen_count


def _validate_token_fields_in_record(
    record: Mapping[str, Any], *, label: str
) -> None:
    """Fail closed unless ``record``'s token fields are internally consistent."""

    count = record.get(OUTPUT_TOKEN_COUNT_FIELD)
    if isinstance(count, bool) or not isinstance(count, int) or int(count) < 0:
        raise Edge576ContractError(
            f"{label} {OUTPUT_TOKEN_COUNT_FIELD}={count!r} must be a "
            "non-negative int (fail closed)"
        )
    count = int(count)
    available = record.get(OUTPUT_TOKEN_IDS_AVAILABLE_FIELD)
    if not isinstance(available, bool):
        raise Edge576ContractError(
            f"{label} {OUTPUT_TOKEN_IDS_AVAILABLE_FIELD}={available!r} must be "
            "a bool (fail closed)"
        )
    raw_ids = record.get(OUTPUT_TOKEN_IDS_FIELD)
    if raw_ids is None:
        if available:
            raise Edge576ContractError(
                f"{label} declares {OUTPUT_TOKEN_IDS_AVAILABLE_FIELD}=true but "
                f"has no {OUTPUT_TOKEN_IDS_FIELD} (fail closed)"
            )
        return
    if not available:
        raise Edge576ContractError(
            f"{label} has {OUTPUT_TOKEN_IDS_FIELD} but declares "
            f"{OUTPUT_TOKEN_IDS_AVAILABLE_FIELD}=false (fail closed)"
        )
    if not isinstance(raw_ids, (list, tuple)):
        raise Edge576ContractError(
            f"{label} {OUTPUT_TOKEN_IDS_FIELD} must be a list (fail closed)"
        )
    for value in raw_ids:
        if isinstance(value, bool) or not isinstance(value, int):
            raise Edge576ContractError(
                f"{label} {OUTPUT_TOKEN_IDS_FIELD} contains non-integer "
                f"{value!r} (fail closed)"
            )
    if len(raw_ids) != count:
        raise Edge576ContractError(
            f"{label} {OUTPUT_TOKEN_IDS_FIELD} length {len(raw_ids)} != "
            f"{OUTPUT_TOKEN_COUNT_FIELD} {count} (fail closed)"
        )



# ---------------------------------------------------------------------------
# Hashing / IO helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_sha256(record: Mapping[str, Any]) -> str:
    """Canonical digest of a raw record, excluding the digest field itself.

    Bit-for-bit compatible with the frozen runners' ``record_sha256``.
    """

    payload = {
        key: value for key, value in record.items() if key != "record_sha256"
    }
    blob = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()


def grid_llm_tokens(grid: Sequence[int]) -> int:
    values = [int(value) for value in grid]
    if len(values) != 3:
        raise Edge576ContractError(f"image_grid_thw must have length 3, got {values}")
    return (values[0] * values[1] * values[2]) // MERGE_UNIT


# ---------------------------------------------------------------------------
# Manifest contract
# ---------------------------------------------------------------------------
def _validate_manifest_record(record: Mapping[str, Any], index: int) -> None:
    label = record.get("sample_id", f"index {index}")
    for field in (
        "sample_id",
        "task",
        "source",
        "image",
        "prompt",
        "answer",
        "choices",
        "max_new_tokens",
        "image_grid_thw",
        "full_patch_tokens",
        "full_llm_visual_tokens",
        "processing",
    ):
        if field not in record:
            raise Edge576ContractError(
                f"edge576 manifest sample {label!r} is missing {field!r} (fail closed)"
            )
    processing = record["processing"]
    if processing.get("edge576") is not True:
        raise Edge576ContractError(
            f"sample {label!r} processing.edge576 is not True (fail closed)"
        )
    for field, expected in (
        ("min_pixels", MIN_PIXELS),
        ("max_pixels", MAX_PIXELS),
        ("patch_size", PATCH_SIZE),
        ("merge_size", MERGE_SIZE),
        ("merge_unit", MERGE_UNIT),
        ("budget_g", BUDGET_G),
        ("keep_aspect_ratio", True),
    ):
        if processing.get(field) != expected:
            raise Edge576ContractError(
                f"sample {label!r} processing.{field}={processing.get(field)!r} "
                f"!= pinned {expected!r} (fail closed)"
            )
    grid = [int(value) for value in record["image_grid_thw"]]
    if len(grid) != 3 or grid[0] != 1:
        raise Edge576ContractError(
            f"sample {label!r} image_grid_thw={grid!r} is not a single image "
            "(fail closed)"
        )
    full_patch_tokens = int(record["full_patch_tokens"])
    if grid[1] * grid[2] != full_patch_tokens:
        raise Edge576ContractError(
            f"sample {label!r} grid {grid} does not match full_patch_tokens "
            f"{full_patch_tokens} (fail closed)"
        )
    full_llm = int(record["full_llm_visual_tokens"])
    if full_llm * MERGE_UNIT != full_patch_tokens:
        raise Edge576ContractError(
            f"sample {label!r} full_llm_visual_tokens {full_llm} does not match "
            f"full_patch_tokens {full_patch_tokens} (fail closed)"
        )
    if full_llm > BUDGET_G:
        raise Edge576ContractError(
            f"sample {label!r} G={full_llm} exceeds the edge576 budget "
            f"{BUDGET_G} (fail closed)"
        )
    if str(record["task"]) == "" or str(record["source"]) == "":
        raise Edge576ContractError(
            f"sample {label!r} has an empty task/source (fail closed)"
        )
    if str(record["source"]) not in CV_BENCH_SOURCES:
        raise Edge576ContractError(
            f"sample {label!r} source {record['source']!r} is not an official "
            f"CV-Bench source (fail closed)"
        )
    if int(record["max_new_tokens"]) != MAX_NEW_TOKENS:
        raise Edge576ContractError(
            f"sample {label!r} max_new_tokens={record['max_new_tokens']!r} "
            f"!= pinned {MAX_NEW_TOKENS} (fail closed)"
        )


def load_edge576_manifest(
    path: Path = EDGE576_MANIFEST,
    *,
    expected_sha256: str = EDGE576_MANIFEST_SHA256,
    expected_count: int = EDGE576_SAMPLE_COUNT,
) -> list[dict[str, Any]]:
    """Load and fully validate the pinned edge576 manifest (fail closed)."""

    path = Path(path)
    if not path.is_file():
        raise Edge576ContractError(f"missing required input {path} (fail closed)")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise Edge576ContractError(
            f"edge576 manifest SHA256 {actual_sha} != pinned {expected_sha256} "
            "(fail closed)"
        )
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Edge576ContractError(
                f"manifest line {line_number} is not valid JSON: {exc} (fail closed)"
            ) from exc
        if not isinstance(record, dict):
            raise Edge576ContractError(
                f"manifest line {line_number} is not a JSON object (fail closed)"
            )
        sample_id = str(record.get("sample_id"))
        if sample_id in seen:
            raise Edge576ContractError(
                f"duplicate manifest sample_id {sample_id!r} (fail closed)"
            )
        seen.add(sample_id)
        _validate_manifest_record(record, len(samples))
        samples.append(record)
    if len(samples) != expected_count:
        raise Edge576ContractError(
            f"edge576 manifest has {len(samples)} samples != expected "
            f"{expected_count} (fail closed)"
        )
    return samples


def g_histogram(samples: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    buckets = {"g_le_96": 0, "g_97_256": 0, "g_257_512": 0, "g_513_576": 0}
    for sample in samples:
        value = int(sample["full_llm_visual_tokens"])
        if value <= 96:
            buckets["g_le_96"] += 1
        elif value <= 256:
            buckets["g_97_256"] += 1
        elif value <= 512:
            buckets["g_257_512"] += 1
        else:
            buckets["g_513_576"] += 1
    return buckets


# ---------------------------------------------------------------------------
# Processor wrapper: the only place the edge576 pixel budget is applied
# ---------------------------------------------------------------------------
class Edge576Processor:
    """Inject the frozen ``images_kwargs`` bounds into every processor call.

    The frozen runners call ``evaluation.run_hf_token_reduction.processor_inputs``
    which forwards ``**kwargs`` to the real processor.  For arms whose frozen
    ``run_one`` does not expose pixel bounds, this wrapper guarantees that the
    authoritative processor still runs with exactly the pinned edge576 bounds,
    and refuses any attempt to override them.
    """

    def __init__(
        self,
        processor: Any,
        *,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
    ) -> None:
        if int(min_pixels) != MIN_PIXELS or int(max_pixels) != MAX_PIXELS:
            raise Edge576ContractError(
                f"processor bounds {min_pixels}/{max_pixels} are not the pinned "
                f"edge576 bounds {MIN_PIXELS}/{MAX_PIXELS} (fail closed)"
            )
        self._processor = processor
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._processor, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        if "images_kwargs" in kwargs:
            raise Edge576ContractError(
                "images_kwargs was supplied explicitly; the edge576 pixel budget "
                "must be applied by Edge576Processor only (fail closed)"
            )
        kwargs["images_kwargs"] = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }
        return self._processor(*args, **kwargs)


# ---------------------------------------------------------------------------
# Label-free selector inputs and per-sample grid assertion
# ---------------------------------------------------------------------------
def sanitize_sample_for_selector(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Return the label-free projection of ``sample`` handed to a frozen runner.

    Only whitelisted keys survive; ``answer``/``ground_truth`` and the derived
    correctness fields are never included, so no label can reach the selector.
    """

    sanitized = {
        key: sample[key] for key in SELECTOR_SAMPLE_KEYS if key in sample
    }
    leaked = [key for key in FORBIDDEN_SELECTOR_KEYS if key in sanitized]
    if leaked:
        raise Edge576ContractError(
            f"selector input would leak labels {leaked} (fail closed)"
        )
    return sanitized


def assert_edge576_grid(
    sample: Mapping[str, Any], actual_grid: Sequence[int] | None
) -> None:
    """Fail closed unless the actual processor grid equals the manifest grid."""

    if actual_grid is None:
        raise Edge576ContractError(
            f"sample {sample.get('sample_id')!r} produced no image_grid_thw "
            "(fail closed)"
        )
    expected = [int(value) for value in sample["image_grid_thw"]]
    actual = [int(value) for value in actual_grid]
    if actual != expected:
        raise Edge576ContractError(
            f"sample {sample.get('sample_id')!r} processor image_grid_thw "
            f"{actual} != edge576 manifest grid {expected} (fail closed)"
        )


# ---------------------------------------------------------------------------
# Output path / resume contract
# ---------------------------------------------------------------------------
def ensure_r2_output_path(arm: str, output: Path | None) -> Path:
    """Return the pinned r2 output path; reject any other destination."""

    validate_arm(arm)
    pinned = ARM_OUTPUT_PATHS[arm]
    if output is None:
        return pinned
    candidate = Path(output)
    try:
        same = candidate.resolve() == pinned.resolve()
    except OSError:
        same = False
    if not same:
        raise Edge576ContractError(
            f"--output {candidate} != pinned r2 path {pinned} for arm {arm!r} "
            "(fail closed)"
        )
    return pinned


def validate_arm(arm: str) -> str:
    if arm not in ARMS:
        raise Edge576ContractError(
            f"unknown arm {arm!r}; expected one of {ARMS} (fail closed)"
        )
    return arm


def _validate_resume_record(
    record: Mapping[str, Any],
    *,
    line_number: int,
    allowed_ids: set[str],
    done: set[str],
    arm: str,
    manifest_sha256: str,
    samples_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    missing = [
        field for field in REQUIRED_EDGE_RECORD_FIELDS if field not in record
    ]
    for field in ARM_EXTRA_RECORD_FIELDS[arm]:
        if field not in record:
            missing.append(field)
    if missing:
        raise Edge576ContractError(
            f"output line {line_number} is missing fields {missing} "
            "(partial record, fail closed)"
        )
    _validate_token_fields_in_record(record, label=f"output line {line_number}")
    sample_id = str(record["sample_id"])
    if sample_id in done:
        raise Edge576ContractError(
            f"duplicate output sample_id {sample_id!r} (fail closed)"
        )
    if sample_id not in allowed_ids:
        raise Edge576ContractError(
            f"output sample_id {sample_id!r} is not in the current manifest "
            "(stale output, fail closed)"
        )
    if str(record.get("arm")) != arm:
        raise Edge576ContractError(
            f"output sample {sample_id!r} arm={record.get('arm')!r} != {arm!r} "
            "(fail closed)"
        )
    if str(record.get("mechanism_id")) != ARM_MECHANISM_IDS[arm]:
        raise Edge576ContractError(
            f"output sample {sample_id!r} mechanism_id="
            f"{record.get('mechanism_id')!r} != {ARM_MECHANISM_IDS[arm]!r} "
            "(fail closed)"
        )
    if record.get("manifest_sha256") != manifest_sha256:
        raise Edge576ContractError(
            f"output sample {sample_id!r} manifest_sha256 differs from the "
            "current run (fail closed)"
        )
    for field, expected in (
        ("min_pixels", MIN_PIXELS),
        ("max_pixels", MAX_PIXELS),
        ("max_new_tokens", MAX_NEW_TOKENS),
        ("do_sample", DO_SAMPLE),
        ("decoding", DECODING),
        ("dtype", DTYPE),
        ("attention_implementation", ATTENTION_IMPLEMENTATION),
    ):
        if record.get(field) != expected:
            raise Edge576ContractError(
                f"output sample {sample_id!r} {field}={record.get(field)!r} != "
                f"pinned {expected!r} (fail closed)"
            )
    sample = samples_by_id[sample_id]
    assert_edge576_grid(sample, record["image_grid_thw"])
    full_llm = int(record["full_llm_visual_tokens"])
    if full_llm != grid_llm_tokens(record["image_grid_thw"]):
        raise Edge576ContractError(
            f"output sample {sample_id!r} full_llm_visual_tokens {full_llm} "
            "does not match its grid (fail closed)"
        )
    if full_llm != int(sample["full_llm_visual_tokens"]):
        raise Edge576ContractError(
            f"output sample {sample_id!r} G={full_llm} != manifest G="
            f"{sample['full_llm_visual_tokens']} (fail closed)"
        )
    parsed = record["parsed_answer"]
    if parsed is not None and not isinstance(parsed, str):
        raise Edge576ContractError(
            f"output sample {sample_id!r} parsed_answer must be str or None "
            "(fail closed)"
        )
    if not isinstance(record["correct"], bool):
        raise Edge576ContractError(
            f"output sample {sample_id!r} correct must be bool (fail closed)"
        )
    if record.get("prompt_sha256") != prompt_sha256(sample["prompt"]):
        raise Edge576ContractError(
            f"output sample {sample_id!r} prompt_sha256 does not match the "
            "edge576 manifest prompt (fail closed)"
        )
    output_group_count = int(record["output_group_count"])
    selected = [int(value) for value in record["selected_group_indices"]]
    if len(selected) != output_group_count:
        raise Edge576ContractError(
            f"output sample {sample_id!r} has {len(selected)} selected groups "
            f"but output_group_count {output_group_count} (fail closed)"
        )
    if selected and not all(
        later > earlier for earlier, later in zip(selected, selected[1:])
    ):
        raise Edge576ContractError(
            f"output sample {sample_id!r} selected_group_indices are not strictly "
            "ascending (fail closed)"
        )
    identity_bypass = bool(record["identity_bypass"])
    if arm in (ARM_L2, ARM_V2):
        if identity_bypass:
            if output_group_count != full_llm:
                raise Edge576ContractError(
                    f"output sample {sample_id!r} identity bypass must emit G="
                    f"{full_llm} groups, got {output_group_count} (fail closed)"
                )
        elif output_group_count != K_BUDGET:
            raise Edge576ContractError(
                f"output sample {sample_id!r} non-bypass must emit K={K_BUDGET}, "
                f"got {output_group_count} (fail closed)"
            )
    elif output_group_count != full_llm:
        raise Edge576ContractError(
            f"dense output sample {sample_id!r} must emit G={full_llm} groups, "
            f"got {output_group_count} (fail closed)"
        )
    if str(record.get("record_sha256")) != record_sha256(record):
        raise Edge576ContractError(
            f"output sample {sample_id!r} record_sha256 mismatch (stale or "
            "corrupted record, fail closed)"
        )


def validate_resume_edge576(
    path: Path,
    samples: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    manifest_sha256: str,
) -> set[str]:
    """Validate an existing edge576 output JSONL; return its completed ids.

    A missing or empty file yields the empty set.  Otherwise every non-blank
    line must be a complete, unique, hash-valid edge576 record for the current
    arm and manifest.  Any violation aborts instead of silently continuing.
    """

    validate_arm(arm)
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    samples_by_id = {str(sample["sample_id"]): sample for sample in samples}
    done: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Edge576ContractError(
                f"output line {line_number} is not valid JSON: {exc} (fail closed)"
            ) from exc
        if not isinstance(record, dict):
            raise Edge576ContractError(
                f"output line {line_number} must be a JSON object (fail closed)"
            )
        _validate_resume_record(
            record,
            line_number=line_number,
            allowed_ids=set(samples_by_id),
            done=done,
            arm=arm,
            manifest_sha256=manifest_sha256,
            samples_by_id=samples_by_id,
        )
        done.add(str(record["sample_id"]))
    return done


# ---------------------------------------------------------------------------
# Mechanism identity + dependency pins
# ---------------------------------------------------------------------------
def _load_r2_config(path: Path = R2_CONFIG) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise Edge576ContractError(f"missing r2 config {path} (fail closed)")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise Edge576ContractError(
            f"r2 config {path} is not valid JSON: {exc} (fail closed)"
        ) from exc
    if config.get("version") != "2026-10-07-edge576-cvbench-three-arm-development-r2":
        raise Edge576ContractError(
            f"unexpected r2 config version {config.get('version')!r} (fail closed)"
        )
    return config


def verify_dependency_pins(
    *,
    pin_names: Sequence[str] = RUNNER_DEPENDENCY_PINS,
    config_path: Path = R2_CONFIG,
) -> dict[str, Any]:
    """Recompute every pinned dependency SHA256; fail closed on any drift."""

    config = _load_r2_config(config_path)
    pins = config["pins"]
    verified: dict[str, dict[str, Any]] = {}
    for name in pin_names:
        if name not in pins:
            raise Edge576ContractError(
                f"r2 config has no dependency pin {name!r} (fail closed)"
            )
        entry = pins[name]
        path = ROOT / entry["path"]
        if not path.is_file():
            raise Edge576ContractError(
                f"pinned dependency {name!r} is missing: {path} (fail closed)"
            )
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise Edge576ContractError(
                f"pinned dependency {name!r} SHA256 drifted: {actual} != "
                f"{entry['sha256']} (fail closed)"
            )
        verified[name] = {"path": entry["path"], "sha256": actual, "role": entry.get("role")}
    return {
        "verified_count": len(verified),
        "requested_count": len(pin_names),
        "all_match": True,
        "pins": verified,
    }


def mechanism_identity(
    arm: str, *, config_path: Path = R2_CONFIG
) -> dict[str, Any]:
    """Confirm the arm's mechanism identity, including the frozen v2 constant."""

    validate_arm(arm)
    expected = ARM_MECHANISM_IDS[arm]
    config = _load_r2_config(config_path)
    config_arm = config["arms"][arm]
    frozen_constant: str | None = None
    if arm == ARM_V2:
        from evaluation.conservative_l2_guarded_reranker import (  # noqa: PLC0415
            MECHANISM,
        )

        frozen_constant = MECHANISM
        if config_arm.get("mechanism_id") != MECHANISM:
            raise Edge576ContractError(
                f"r2 v2 arm mechanism_id {config_arm.get('mechanism_id')!r} != "
                f"frozen {MECHANISM!r} (fail closed)"
            )
        if bool(config_arm.get("is_new_mechanism")):
            raise Edge576ContractError(
                "r2 v2 arm declares is_new_mechanism=true (fail closed)"
            )
    elif arm == ARM_L2:
        from evaluation.hardware_regular_b1_corrected_l2 import (  # noqa: PLC0415
            ARM_INTEGER,
        )

        frozen_constant = ARM_INTEGER
    if str(config_arm.get("output_path")) != str(
        ARM_OUTPUT_PATHS[arm].relative_to(ROOT)
    ):
        raise Edge576ContractError(
            f"r2 arm {arm!r} output_path does not match the pinned path (fail closed)"
        )
    return {
        "arm": arm,
        "mechanism_id": expected,
        "frozen_constant_mechanism_id": frozen_constant,
        "is_new_mechanism": bool(ARM_IS_NEW_MECHANISM[arm]),
        "config_output_path": str(config_arm.get("output_path")),
        "allowed_mechanism_scope": config_arm.get("allowed_mechanism_scope"),
        "ok": True,
    }


# ---------------------------------------------------------------------------
# Model identity verification (fail closed, before any weight load)
# ---------------------------------------------------------------------------
def load_model_identity_manifest(
    path: Path = MODEL_IDENTITY_MANIFEST,
    *,
    expected_sha256: str | None = MODEL_IDENTITY_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Load and structurally validate the frozen model identity manifest.

    The manifest file itself is SHA256-pinned, so a swapped or edited identity
    record aborts here before any ``--model`` directory is trusted.
    """

    path = Path(path)
    if not path.is_file():
        raise Edge576ContractError(
            f"missing model identity manifest {path} (fail closed)"
        )
    actual_sha = sha256_file(path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise Edge576ContractError(
            f"model identity manifest SHA256 {actual_sha} != pinned "
            f"{expected_sha256} (fail closed)"
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Edge576ContractError(
            f"model identity manifest {path} is not valid JSON: {exc} "
            "(fail closed)"
        ) from exc
    if not isinstance(manifest, dict):
        raise Edge576ContractError(
            f"model identity manifest {path} is not a JSON object (fail closed)"
        )
    if manifest.get("kind") != MODEL_IDENTITY_KIND:
        raise Edge576ContractError(
            f"model identity manifest kind={manifest.get('kind')!r} != "
            f"{MODEL_IDENTITY_KIND!r} (fail closed)"
        )
    if manifest.get("schema_version") != MODEL_IDENTITY_SCHEMA_VERSION:
        raise Edge576ContractError(
            f"model identity schema_version={manifest.get('schema_version')!r} "
            f"!= {MODEL_IDENTITY_SCHEMA_VERSION!r} (fail closed)"
        )
    if manifest.get("hash_algorithm") != MODEL_IDENTITY_HASH_ALGORITHM:
        raise Edge576ContractError(
            f"model identity hash_algorithm={manifest.get('hash_algorithm')!r} "
            f"!= {MODEL_IDENTITY_HASH_ALGORITHM!r} (fail closed)"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise Edge576ContractError(
            "model identity manifest carries no files (fail closed)"
        )
    return manifest


def _validate_identity_relative_path(relative_path: str) -> Path:
    candidate = Path(str(relative_path))
    if not str(relative_path) or candidate.is_absolute():
        raise Edge576ContractError(
            f"model identity relative_path {relative_path!r} must be relative "
            "(fail closed)"
        )
    if ".." in candidate.parts:
        raise Edge576ContractError(
            f"model identity relative_path {relative_path!r} escapes the model "
            "directory (fail closed)"
        )
    return candidate


def verify_model_identity(
    model_path: Path,
    *,
    identity_path: Path = MODEL_IDENTITY_MANIFEST,
    expected_manifest_sha256: str | None = MODEL_IDENTITY_MANIFEST_SHA256,
    required_weight_shards: Sequence[str] = REQUIRED_MODEL_WEIGHT_SHARDS,
) -> dict[str, Any]:
    """Verify every declared ``--model`` file against the frozen identity.

    Checks, all fail closed and all CPU-only, before any model weight is opened:

    * the identity manifest matches its pinned SHA256 and structure;
    * ``--model`` resolves to the manifest's declared ``model_dir``;
    * every declared relative path exists, has the recorded byte size and the
      recorded SHA256 (including the ``model.safetensors`` weight shard);
    * each required weight shard is declared and verified.
    """

    identity_path = Path(identity_path)
    manifest = load_model_identity_manifest(
        identity_path, expected_sha256=expected_manifest_sha256
    )
    identity_sha = sha256_file(identity_path)
    model_root = Path(model_path).expanduser().resolve()
    if not model_root.is_dir():
        raise Edge576ContractError(
            f"--model is not a local directory: {model_root} (fail closed)"
        )
    declared_root_raw = manifest.get("model_dir")
    if not declared_root_raw:
        raise Edge576ContractError(
            "model identity manifest declares no model_dir (fail closed)"
        )
    declared_root = Path(str(declared_root_raw)).expanduser().resolve()
    if declared_root != model_root:
        raise Edge576ContractError(
            f"--model {model_root} != identity manifest model_dir {declared_root} "
            "(fail closed)"
        )

    verified: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in manifest["files"]:
        if not isinstance(entry, Mapping):
            raise Edge576ContractError(
                "model identity file entry is not an object (fail closed)"
            )
        relative_path = str(entry.get("relative_path", ""))
        relative = _validate_identity_relative_path(relative_path)
        if relative_path in seen:
            raise Edge576ContractError(
                f"duplicate model identity relative_path {relative_path!r} "
                "(fail closed)"
            )
        seen.add(relative_path)
        target = model_root / relative
        if not target.is_file():
            raise Edge576ContractError(
                f"model identity file missing: {target} (fail closed)"
            )
        actual_bytes = target.stat().st_size
        expected_bytes = int(entry.get("bytes", -1))
        if actual_bytes != expected_bytes:
            raise Edge576ContractError(
                f"model identity file {relative_path!r} size {actual_bytes} != "
                f"{expected_bytes} (fail closed)"
            )
        expected_sha = str(entry.get("sha256", ""))
        actual_sha = sha256_file(target)
        if actual_sha != expected_sha:
            raise Edge576ContractError(
                f"model identity file {relative_path!r} SHA256 {actual_sha} != "
                f"{expected_sha} (fail closed)"
            )
        verified.append(
            {
                "relative_path": relative_path,
                "bytes": actual_bytes,
                "sha256": actual_sha,
                "role": entry.get("role"),
            }
        )

    declared_shards = {str(value) for value in manifest.get("weight_shards", [])}
    for shard in required_weight_shards:
        if shard not in seen:
            raise Edge576ContractError(
                f"required weight shard {shard!r} is not in the model identity "
                "manifest (fail closed)"
            )
        if shard not in declared_shards:
            raise Edge576ContractError(
                f"required weight shard {shard!r} is not declared in "
                "weight_shards (fail closed)"
            )
    shard_hashes = {
        item["relative_path"]: item["sha256"]
        for item in verified
        if item["relative_path"] in declared_shards
    }
    return {
        "ok": True,
        "identity_path": str(identity_path),
        "identity_sha256": identity_sha,
        "model_path": str(model_root),
        "model_label": manifest.get("model_label"),
        "total_bytes": sum(item["bytes"] for item in verified),
        "file_count": len(verified),
        "verified_file_count": len(verified),
        "weight_shards": sorted(declared_shards),
        "required_weight_shards": list(required_weight_shards),
        "weight_shard_sha256": shard_hashes,
        "files": verified,
    }


def compact_model_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return the small model-identity summary stored in run metadata."""

    return {
        "ok": bool(identity.get("ok")),
        "identity_path": identity.get("identity_path"),
        "identity_sha256": identity.get("identity_sha256"),
        "model_path": identity.get("model_path"),
        "model_label": identity.get("model_label"),
        "file_count": identity.get("file_count"),
        "total_bytes": identity.get("total_bytes"),
        "weight_shards": list(identity.get("weight_shards", [])),
        "weight_shard_sha256": dict(identity.get("weight_shard_sha256", {})),
    }
