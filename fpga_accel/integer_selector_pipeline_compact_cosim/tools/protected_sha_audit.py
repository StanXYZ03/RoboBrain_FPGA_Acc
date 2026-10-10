#!/usr/bin/env python3
"""SHA-256 snapshot / verify for the frozen protected files.

The RTL co-simulation phase must not modify any pre-existing file under the
four accepted HLS kernel trees, the ``evaluation/`` tree, or the accepted
compact-interface reports / artifacts / logs.  This tool takes a byte-level
snapshot of exactly those paths and can later verify that nothing changed.

Usage:
  protected_sha_audit.py snapshot --out <path.json>
  protected_sha_audit.py verify   --before <path.json> --after <path.json> \
                                  --out <verify.json>

The snapshot JSON is a plain mapping of repository-relative path -> sha256 plus
an aggregate digest over the sorted (path, sha256) pairs, so it is stable and
diffable.
"""

import argparse
import datetime
import hashlib
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# fpga_accel/integer_selector_pipeline_compact_cosim/tools -> repo root
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

# Directories that are frozen recursively.
PROTECTED_DIRS = [
    "fpga_accel/integer_selector_scorer_hls",
    "fpga_accel/token_selector_hls",
    "fpga_accel/integer_selector_pipeline_hls",
    "fpga_accel/integer_selector_pipeline_compact_hls",
    "evaluation",
]

# The accepted compact-interface evidence (reports + artifact tree + logs).
PROTECTED_DIRS += [
    "artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls",
]

# Pre-existing compact-interface related files (reports / logs / prompts).
PROTECTED_FILES = [
    "reports/token_selector/2026-10-03-integer-selector-compact-interface-hls.md",
    "reports/token_selector/2026-10-03-integer-selector-compact-interface-hls.json",
    "reports/token_selector/logs/2026-10-03-integer-selector-compact-interface-hls-harness-result.json",
    "reports/token_selector/prompts/2026-10-03-integer-selector-compact-interface-hls-harness-prompt.md",
    "reports/token_selector/prompts/2026-10-03-integer-selector-compact-rtl-cosim-harness-prompt.md",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_protected():
    seen = set()
    for rel_dir in PROTECTED_DIRS:
        abs_dir = os.path.join(REPO_ROOT, rel_dir)
        if not os.path.isdir(abs_dir):
            continue
        for dirpath, dirnames, filenames in os.walk(abs_dir):
            dirnames.sort()
            for name in sorted(filenames):
                abs_path = os.path.join(dirpath, name)
                rel = os.path.relpath(abs_path, REPO_ROOT)
                if rel in seen:
                    continue
                seen.add(rel)
                yield rel, abs_path
    for rel in PROTECTED_FILES:
        abs_path = os.path.join(REPO_ROOT, rel)
        if os.path.isfile(abs_path) and rel not in seen:
            seen.add(rel)
            yield rel, abs_path


def aggregate(files):
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(files[rel].encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def snapshot():
    files = {}
    bytes_total = 0
    for rel, abs_path in iter_protected():
        files[rel] = sha256_file(abs_path)
        bytes_total += os.path.getsize(abs_path)
    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "repo_root": REPO_ROOT,
        "protected_dirs": PROTECTED_DIRS,
        "protected_files": PROTECTED_FILES,
        "total_files": len(files),
        "total_bytes": bytes_total,
        "aggregate_sha256": aggregate(files),
        "files": files,
    }


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("snapshot")
    ps.add_argument("--out", required=True)

    pv = sub.add_parser("verify")
    pv.add_argument("--before", required=True)
    pv.add_argument("--after", required=True)
    pv.add_argument("--out", required=True)

    args = ap.parse_args()

    if args.cmd == "snapshot":
        snap = snapshot()
        with open(args.out, "w") as fh:
            json.dump(snap, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(
            "snapshot: files=%d bytes=%d aggregate=%s -> %s"
            % (
                snap["total_files"],
                snap["total_bytes"],
                snap["aggregate_sha256"],
                args.out,
            )
        )
        return 0

    with open(args.before) as fh:
        before = json.load(fh)
    with open(args.after) as fh:
        after = json.load(fh)
    b = before["files"]
    a = after["files"]
    changed = sorted(k for k in b if k in a and a[k] != b[k])
    missing = sorted(k for k in b if k not in a)
    added = sorted(k for k in a if k not in b)
    result = {
        "before": args.before,
        "after": args.after,
        "before_total_files": len(b),
        "after_total_files": len(a),
        "before_aggregate_sha256": before["aggregate_sha256"],
        "after_aggregate_sha256": after["aggregate_sha256"],
        "changed": changed,
        "missing": missing,
        "added": added,
        "unchanged": not (changed or missing or added),
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(
        "verify: before=%d after=%d changed=%d missing=%d added=%d unchanged=%s"
        % (len(b), len(a), len(changed), len(missing), len(added), result["unchanged"])
    )
    return 0 if result["unchanged"] else 1


if __name__ == "__main__":
    sys.exit(main())
