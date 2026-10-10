#!/usr/bin/env python3
"""Resolve and validate the five bounded RTL-co-simulation cases.

The case list must use the *accepted* compact-interface manifest lines verbatim:
``case_name arena_file budget local_cap scale grid_h grid_w``.  This tool
refuses to continue (exit 1, with explicit evidence) if any required
representative slot is missing from the accepted fixture set, so a weaker case
can never be silently substituted.

Slots (in order):
  1 production non-identity G80/K48
  2 real correct-prompt G260/K64
  3 production stress G400/K96
  4 identity G80/K96
  5 deterministic all-zero or all-equal/tie case at a supported budget
"""

import argparse
import hashlib
import json
import os
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True, help="accepted fixture directory")
    ap.add_argument("--cases", required=True, help="new case-list file")
    ap.add_argument("--out", required=False, default=None)
    args = ap.parse_args()

    manifest_txt = os.path.join(args.fixtures, "manifest.txt")
    manifest_json = os.path.join(args.fixtures, "manifest.json")
    report = {"fixtures": args.fixtures, "case_list": args.cases, "checks": []}

    def check(name, ok, detail):
        report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        return ok

    with open(manifest_txt) as fh:
        accepted_lines = [ln.strip() for ln in fh if ln.strip()]
    with open(manifest_json) as fh:
        mj = json.load(fh)
    accepted_cases = mj["cases"]

    with open(args.cases) as fh:
        case_lines = [ln.strip() for ln in fh if ln.strip()]

    check("five_cases", len(case_lines) == 5, "case list has %d lines" % len(case_lines))

    resolved = []
    for line in case_lines:
        fields = line.split()
        if len(fields) != 7:
            check("line_format:%s" % line, False, "expected 7 fields")
            resolved.append(None)
            continue
        name, arena_file, budget, local_cap, scale, grid_h, grid_w = fields
        exact = line in accepted_lines
        check("accepted_line:%s" % name, exact,
              "verbatim line present in accepted manifest.txt" if exact
              else "NOT present verbatim in accepted manifest.txt")
        info = accepted_cases.get(name)
        check("manifest_case:%s" % name, info is not None,
              "present in accepted manifest.json cases" if info
              else "absent from accepted manifest.json cases")
        if info is None:
            resolved.append(None)
            continue
        check("budget:%s" % name, int(budget) == info["budget"],
              "case list budget %s vs manifest %s" % (budget, info["budget"]))
        check("grid:%s" % name,
              [int(grid_h), int(grid_w)] == list(info["grid"]),
              "case list grid vs manifest grid %s" % (info["grid"],))
        check("cap:%s" % name, int(local_cap) == info["local_cap"],
              "case list local_cap %s vs manifest %s" % (local_cap, info["local_cap"]))
        exp_path = os.path.join(args.fixtures, info["file"])
        exists = os.path.isfile(exp_path)
        check("expected_file:%s" % name, exists, exp_path)
        if exists:
            digest = sha256_file(exp_path)
            check("expected_sha:%s" % name, digest == info["sha256"],
                  "expected sha256 %s vs manifest %s" % (digest, info["sha256"]))
        arena_path = os.path.join(args.fixtures, arena_file)
        scene = info["scene"]
        scene_info = mj["scenes"].get(scene, {})
        check("arena_file:%s" % name, os.path.isfile(arena_path), arena_path)
        if scene_info:
            check("arena_matches_csv:%s" % name, scene_info.get("file") == arena_file,
                  "arena file %s vs scene %s" % (arena_file, scene_info.get("file")))
        resolved.append({
            "case": name,
            "scene": scene,
            "group_count": info["group_count"],
            "grid": info["grid"],
            "budget": info["budget"],
            "local_cap": info["local_cap"],
            "scale": info["scale"],
            "status": info["status"],
            "output_count": info["output_count"],
            "real_count": info["real_count"],
            "context_count": info["context_count"],
            "total_bytes": info["total_bytes"],
            "expected_sha256": info["sha256"],
        })

    # ---- semantic slot resolution (no weaker substitution) ----
    slots = [
        ("production_non_identity_G80_K48",
         lambda c: c and c["group_count"] == 80 and c["budget"] == 48
         and c["group_count"] > c["budget"]),
        ("real_correct_prompt_G260_K64",
         lambda c: c and c["group_count"] == 260 and c["budget"] == 64
         and c["scene"].startswith("real_") and "correct" in c["scene"]),
        ("production_stress_G400_K96",
         lambda c: c and c["group_count"] == 400 and c["budget"] == 96),
        ("identity_G80_K96",
         lambda c: c and c["group_count"] == 80 and c["budget"] == 96
         and c["group_count"] <= c["budget"]),
        ("zero_or_all_equal_tie_supported_budget",
         lambda c: c and ("zero" in c["scene"] or "all_equal" in c["scene"])
         and c["budget"] in (48, 64, 96)),
    ]
    slot_report = []
    for slot_name, pred in slots:
        hit = None
        for idx, c in enumerate(resolved):
            if pred(c):
                hit = {"index": idx, "case": c["case"]}
                break
        check("slot:%s" % slot_name, hit is not None,
              "resolved to %s" % (hit["case"] if hit else "NO MATCH"))
        slot_report.append({"slot": slot_name, "resolved_case": hit["case"] if hit else None})

    report["resolved_cases"] = resolved
    report["slots"] = slot_report
    report["all_ok"] = all(c["ok"] for c in report["checks"])

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    print(text)
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
