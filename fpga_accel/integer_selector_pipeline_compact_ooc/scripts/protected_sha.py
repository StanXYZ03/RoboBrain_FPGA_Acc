#!/usr/bin/env python3
"""Protected-file SHA-256 audit for the compact-IP OOC route task.

Snapshots the byte content (sha256 + size) of every protected tree/file into a
JSON manifest, and compares a before/after pair.  Only reads files.

Protected set (frozen by the task):
  * fpga_accel/integer_selector_scorer_hls/
  * fpga_accel/token_selector_hls/
  * fpga_accel/integer_selector_pipeline_hls/
  * fpga_accel/integer_selector_pipeline_compact_hls/
  * fpga_accel/integer_selector_pipeline_compact_cosim/
  * evaluation/
  * the existing compact-interface and compact-RTL-cosim reports, logs, prompts
    and artifacts (globbed explicitly, excluding the new ooc-route paths)

The new paths (integer_selector_pipeline_compact_ooc/,
2026-10-03-integer-selector-compact-ip-ooc-route*) are NOT part of the protected
set and are intentionally excluded.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

PROTECTED_DIRS = [
    "fpga_accel/integer_selector_scorer_hls",
    "fpga_accel/token_selector_hls",
    "fpga_accel/integer_selector_pipeline_hls",
    "fpga_accel/integer_selector_pipeline_compact_hls",
    "fpga_accel/integer_selector_pipeline_compact_cosim",
    "evaluation",
]

PROTECTED_GLOBS = [
    "reports/token_selector/2026-10-03-integer-selector-compact-interface-hls.*",
    "reports/token_selector/2026-10-03-integer-selector-compact-rtl-cosim.*",
    "reports/token_selector/logs/2026-10-03-integer-selector-compact-interface-hls*",
    "reports/token_selector/logs/2026-10-03-integer-selector-compact-rtl-cosim*",
    "reports/token_selector/prompts/*integer-selector-compact-interface*",
    "reports/token_selector/prompts/*integer-selector-compact-rtl-cosim*",
    "artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/**",
    "artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/**",
]

EXCLUDE_SUBSTRINGS = [
    "integer-selector-compact-ip-ooc-route",
    "integer_selector_pipeline_compact_ooc",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_paths(repo, root):
    if not os.path.isdir(root):
        return
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            yield os.path.join(dirpath, name)


def collect(repo):
    entries = {}
    for rel in PROTECTED_DIRS:
        for p in iter_paths(repo, os.path.join(repo, rel)):
            rp = os.path.relpath(p, repo)
            if any(s in rp for s in EXCLUDE_SUBSTRINGS):
                continue
            entries[rp] = p
    for pat in PROTECTED_GLOBS:
        for p in glob.glob(os.path.join(repo, pat), recursive=True):
            if not os.path.isfile(p):
                continue
            rp = os.path.relpath(p, repo)
            if any(s in rp for s in EXCLUDE_SUBSTRINGS):
                continue
            entries[rp] = p
    manifest = {}
    for rp in sorted(entries):
        p = entries[rp]
        manifest[rp] = {"sha256": sha256_file(p), "size": os.path.getsize(p)}
    return manifest


def aggregate(manifest):
    h = hashlib.sha256()
    for rp in sorted(manifest):
        h.update(rp.encode())
        h.update(b"\0")
        h.update(manifest[rp]["sha256"].encode())
        h.update(b"\n")
    return h.hexdigest()


def cmd_snapshot(args):
    repo = os.path.abspath(args.repo)
    manifest = collect(repo)
    out = {
        "repo": repo,
        "file_count": len(manifest),
        "aggregate_sha256": aggregate(manifest),
        "files": manifest,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps({"file_count": len(manifest), "aggregate_sha256": out["aggregate_sha256"]}))
    return 0


def cmd_compare(args):
    before = json.load(open(args.before))
    after = json.load(open(args.after))
    b = before["files"]
    a = after["files"]
    changed = sorted(rp for rp in b if rp in a and b[rp]["sha256"] != a[rp]["sha256"])
    missing = sorted(rp for rp in b if rp not in a)
    added = sorted(rp for rp in a if rp not in b)
    result = {
        "before_file_count": len(b),
        "after_file_count": len(a),
        "before_aggregate_sha256": before["aggregate_sha256"],
        "after_aggregate_sha256": after["aggregate_sha256"],
        "changed": changed,
        "missing": missing,
        "added": added,
        "protected_files_unchanged": (
            not changed and not missing and before["aggregate_sha256"] == after["aggregate_sha256"]
        ),
        "all_protected_present_and_identical": len(b) == len(a) and not changed and not missing and not added,
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--repo", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_snapshot)
    c = sub.add_parser("compare")
    c.add_argument("--before", required=True)
    c.add_argument("--after", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_compare)
    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
