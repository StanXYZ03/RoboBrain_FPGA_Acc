#!/usr/bin/env python3
"""Run a deterministic spatial baseline against a llama.cpp OpenAI endpoint."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import load_jsonl


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def request_json(url: str, payload: dict | None, timeout: float) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_image(manifest: Path, image: str) -> Path:
    path = Path(image)
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8090")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task", choices=("pointing", "grounding"))
    args = parser.parse_args()

    health = request_json(args.base_url.rstrip("/") + "/health", None, 10)
    if health.get("status") not in {"ok", "no slot available"}:
        raise RuntimeError(f"endpoint is not healthy: {health}")
    samples = load_jsonl(args.manifest)
    if args.task:
        samples = [sample for sample in samples if sample["task"] == args.task]
    if args.limit is not None:
        samples = samples[: args.limit]
    done = {record["sample_id"] for record in load_jsonl(args.output)} if args.output.is_file() else set()
    pending = [sample for sample in samples if sample["sample_id"] not in done]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8", buffering=1) as output:
        for index, sample in enumerate(pending, 1):
            image_path = resolve_image(args.manifest, sample["image"])
            payload = {
                "model": args.model_label,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri(image_path)}},
                            {"type": "text", "text": sample["prompt"]},
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": int(sample["max_new_tokens"]),
                "stream": False,
            }
            started = time.perf_counter()
            response = request_json(
                args.base_url.rstrip("/") + "/v1/chat/completions",
                payload,
                args.timeout,
            )
            elapsed = time.perf_counter() - started
            text = response["choices"][0]["message"]["content"]
            record = {
                "sample_id": sample["sample_id"],
                "task": sample["task"],
                "output_text": text,
                "elapsed_seconds": elapsed,
                "usage": response.get("usage"),
                "run": {
                    "model_label": args.model_label,
                    "base_url": args.base_url,
                    "deterministic_decode": True,
                },
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"RESULT {index}/{len(pending)} id={sample['sample_id']} elapsed={elapsed:.3f}s output={text!r}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
