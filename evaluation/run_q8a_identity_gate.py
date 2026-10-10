"""Run local CPU Q8-A smoke cases with append-only raw logs."""
import argparse
import json
import hashlib
import os
import re
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("q8a", "q8b"), default="q8a")
    args = parser.parse_args()
    output = ROOT / "evaluation/results" / time.strftime(f"{args.stage}_identity_%Y%m%dT%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    sample = json.loads((ROOT / "datasets/dac_cutpoint_oracle_dev.jsonl").read_text().splitlines()[0])
    model_root = Path("/home/zhaoxiangyu/RoboBrain2.5/deployment/models")
    command = [str(ROOT / "profiling_src/llama.cpp/build-capture/bin/llama-mtmd-cli"),
               "-m", str(model_root / "RoboBrain2.5-4B.Q8_0.gguf"),
               "--mmproj", str(model_root / "RoboBrain2.5-4B.mmproj-Q8_0.gguf"),
               "--image", str(ROOT / "datasets" / sample["image"]),
               "-p", sample["prompt"], "-n", str(sample["max_new_tokens"]),
               "-c", "2048", "-b", "512", "-ub", "256", "-t", "4", "-tb", "4",
               "-ngl", "0", "--no-mmproj-offload", "--temp", "0", "--seed", "42"]
    records = []
    for mode in ("stock", "identity"):
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
                   MTMD_Q8A_PRINT_TOKEN_IDS="1", MTMD_TEST_RESPONSE_MARKER="Q8A_RESPONSE_BEGIN")
        env.pop("MTMD_Q8A_KEEP_INDICES", None)
        env.pop("MTMD_Q8B_KEEP_INDICES", None)
        env.pop("GGML_CPU_PROFILE", None)
        if mode == "identity":
            env[f"MTMD_{args.stage.upper()}_KEEP_INDICES"] = "identity"
        print(f"STAGE {args.stage} {mode}: CPU-only numerical smoke, at most 32 output tokens; logs={output}", flush=True)
        start = time.monotonic()
        with (output / f"{mode}.stdout").open("x") as stdout, (output / f"{mode}.stderr").open("x") as stderr:
            try:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=1800)
                exit_code = result.returncode
            except subprocess.TimeoutExpired:
                exit_code = 124
        log_text = (output / f"{mode}.stdout").read_text(errors="replace") + (output / f"{mode}.stderr").read_text(errors="replace")
        matches = re.findall(r"Q8A_TOKEN_IDS=([0-9,]+)", log_text)
        token_ids = [int(x) for x in matches[-1].split(",")] if matches else []
        record = {"stage": args.stage, "mode": mode, "command": command, "exit_code": exit_code, "token_ids": token_ids,
                  "executable_sha256": hashlib.sha256(Path(command[0]).read_bytes()).hexdigest(),
                  "process_wall_s_including_load": time.monotonic() - start,
                  "sample": sample, "performance_benchmark": False}
        records.append(record)
        with (output / "runs.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"DONE {mode}: exit={exit_code}; tokens={len(token_ids)}", flush=True)
        if exit_code:
            break
    passed = (len(records) == 2 and all(r["exit_code"] == 0 and r["token_ids"] for r in records)
              and records[0]["token_ids"] == records[1]["token_ids"])
    summary = {"stage": args.stage, "status": "IDENTITY_TOKEN_SMOKE_PASS" if passed else "IDENTITY_GATE_NOT_PASSED",
               "records": records, "limitations": "One development image; not an accuracy or latency benchmark; sparse gather remains unverified."}
    with (output / "summary.json").open("x") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(summary["status"], flush=True)
    print(f"OUTPUT {output}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
