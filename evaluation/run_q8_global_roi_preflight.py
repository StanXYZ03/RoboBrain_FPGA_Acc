#!/usr/bin/env python3
"""Run one real Q8_0 two-image interface preflight through llama-mtmd-cli."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

MARKER = "Q8_MULTI_IMAGE_RESPONSE"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--global-image", type=Path, required=True)
    parser.add_argument("--roi-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    required = [args.executable, args.model, args.mmproj, args.global_image, args.roi_image]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = args.output_dir / "stdout.txt"
    stderr_path = args.output_dir / "stderr.txt"
    if stdout_path.exists() or stderr_path.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    prompt = (
        "The first image is a low-resolution full scene. The second image is a high-resolution "
        "crop from the first image. Use the first for context and the second for localization. "
        "Locate the couch in the second image and return exactly one bounding box as "
        "[x1, y1, x2, y2], with integer coordinates relative to the second image, normalized 0-1000."
    )
    command = [
        str(args.executable), "-m", str(args.model), "--mmproj", str(args.mmproj),
        "--image", f"{args.global_image},{args.roi_image}", "-p", prompt,
        "-ngl", "0", "--no-mmproj-offload", "--temp", "0", "--seed", "42",
        "--image-min-tokens", "1", "--image-max-tokens", "512", "-n", str(args.max_tokens),
    ]
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        MTMD_TEST_RESPONSE_MARKER=MARKER,
    )
    started = time.monotonic()
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open(
        "x", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, env=env, timeout=1800)
    wall = time.monotonic() - started
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    output = stdout_text.split(MARKER, 1)[1].strip() if MARKER in stdout_text else ""
    encoded_image_batches = stderr_text.count("encoding mtmd batch")
    status = (
        "Q8_MULTI_IMAGE_FORWARD_PASS"
        if completed.returncode == 0 and output and encoded_image_batches >= 2
        else "Q8_MULTI_IMAGE_FORWARD_FAIL"
    )
    report = {
        "status": status,
        "scope": "single revealed interface preflight; not an accuracy or steady-state latency benchmark",
        "exit_code": completed.returncode,
        "output_text": output,
        "process_wall_seconds_including_model_load": wall,
        "global_image": str(args.global_image.resolve()),
        "roi_image": str(args.roi_image.resolve()),
        "encoded_image_batches": encoded_image_batches,
        "accuracy_warning": "runtime recommends at least 1024 image tokens for grounding; this low-budget run is interface-only",
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status.endswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
