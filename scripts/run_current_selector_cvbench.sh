#!/usr/bin/env bash
# run_current_selector_cvbench.sh
#
# Host-side driver for the 2026-10-04 current-Selector CV-Bench gate.
#
# Stages (strict order, each resumable from its own raw JSONL):
#   1. identity      : <= 5 samples, B1 integer selector forced to G <= K
#                      identity bypass; compares generated token ids, decoded
#                      text and dense vision image_embeds with the official
#                      dense path.  Any mismatch stops the run (set -e).
#   2. smoke         : <= 5 samples, structural checks only; quality is written
#                      but never aggregated and never influences a choice.
#   3. full K=48/64/96 : one complete deterministic pass per K over the frozen
#                      2638-sample manifest, then the official summariser.
#   4. latency       : NOT run here; the exact stratified-100 x 3-repeat command
#                      is printed at the end.
#
# Every stage appends to its own JSONL and never recomputes a sample id that is
# already present with the same configuration.  Re-running this script is safe.
#
# No GPU is launched by the Harness; Codex runs this on the host.  No dependency
# is installed.  The software selector is never labelled FPGA/board acceleration.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-/home/zhaoxiangyu/conda-envs/robobrain25/bin/python}"
MODEL="${MODEL:-/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/datasets/cv_bench_full2638.jsonl}"
SCREENING="${SCREENING:-${REPO_ROOT}/datasets/cv_bench_screening100.jsonl}"
NATIVE_JSON="${NATIVE_JSON:-${REPO_ROOT}/reports/cv_bench/cv_bench_bf16_full2638.json}"
DENSE_JSONL="${DENSE_JSONL:-${REPO_ROOT}/evaluation/results/cv_bench_l2_0875_full2638_20260916/shallow_r1000.jsonl}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/artifacts/token_selector/2026-10-04-current-selector-cvbench}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/reports/token_selector/logs/2026-10-04-current-selector-cvbench}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bf16}"

RUNNER=("${PYTHON}" -m evaluation.run_hf_current_selector_cvbench)
SUMMARIZER=("${PYTHON}" -m evaluation.summarize_current_selector_cvbench)

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

echo "=== stage 1/3 identity gate (<=5 samples) ==="
"${RUNNER[@]}" \
  --model "${MODEL}" --model-label RoboBrain2.5-4B-bf16 \
  --manifest "${MANIFEST}" \
  --output "${OUT_DIR}/identity.jsonl" \
  --mode identity --limit-per-task 1 \
  --device "${DEVICE}" --dtype "${DTYPE}" \
  2>&1 | tee "${LOG_DIR}/identity.log"

echo "=== stage 2/3 structural smoke (<=5 samples, no quality aggregation) ==="
"${RUNNER[@]}" \
  --model "${MODEL}" --model-label RoboBrain2.5-4B-bf16 \
  --manifest "${MANIFEST}" \
  --output "${OUT_DIR}/smoke_k64.jsonl" \
  --mode smoke --budget 64 --limit-per-task 1 \
  --device "${DEVICE}" --dtype "${DTYPE}" \
  2>&1 | tee "${LOG_DIR}/smoke_k64.log"

echo "=== stage 3/3 full2638 K=48/64/96 ==="
for K in 48 64 96; do
  echo "--- K=${K} ---"
  "${RUNNER[@]}" \
    --model "${MODEL}" --model-label RoboBrain2.5-4B-bf16 \
    --manifest "${MANIFEST}" \
    --output "${OUT_DIR}/current_selector_k${K}.jsonl" \
    --mode full --budget "${K}" \
    --device "${DEVICE}" --dtype "${DTYPE}" \
    2>&1 | tee -a "${LOG_DIR}/full_k${K}.log"
done

echo "=== official summary (COMPLETE only when 2638 unique complete samples) ==="
"${SUMMARIZER[@]}" \
  --manifest "${MANIFEST}" \
  --native-json "${NATIVE_JSON}" \
  --dense-jsonl "${DENSE_JSONL}" \
  --candidate K48 "${OUT_DIR}/current_selector_k48.jsonl" \
  --candidate K64 "${OUT_DIR}/current_selector_k64.jsonl" \
  --candidate K96 "${OUT_DIR}/current_selector_k96.jsonl" \
  --output "${REPO_ROOT}/reports/cv_bench/2026-10-04-current-selector-cvbench.json" \
  2>&1 | tee "${LOG_DIR}/summary.log"

echo "=== latency command (NOT run by this script) ==="
printf '%q ' "${RUNNER[@]}"
echo "--model ${MODEL} --model-label RoboBrain2.5-4B-bf16 \\"
echo "  --manifest ${SCREENING} --output ${OUT_DIR}/latency100_k64_r3.jsonl \\"
echo "  --mode latency --budget 64 --repeats 3 --warmup 1 \\"
echo "  --device ${DEVICE} --dtype ${DTYPE}"
echo
echo "=== raw JSONL sha256 ==="
sha256sum "${OUT_DIR}"/*.jsonl | tee "${LOG_DIR}/raw-jsonl.sha256"
echo "DONE"
