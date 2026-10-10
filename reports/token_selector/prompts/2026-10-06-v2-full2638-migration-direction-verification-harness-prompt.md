# Harness task: verify the corrected full2638 migration direction

Work read-only in `/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`.

Do not edit files, run GPU/model/network/training, or run the full `evaluation/tests` suite. The old full suite contains state-dependent pre-execution tests that rewrite the frozen development report when the full arm exists.

Verify only:

1. `evaluation/audit_conservative_l2_guarded_rerank_v2_full2638.py::_migration(left,right)` defines “left relative to right”: repair = right wrong and left correct; regression = right correct and left wrong.
2. Independently recompute from frozen manifest and four JSONL outputs:
   - v2 vs corrected integer-L2 = 117 repairs, 73 regressions, net +44, 223 answer changes;
   - v2 vs dense = 55 repairs, 287 regressions, net -232, 409 answer changes.
3. Confirm the regenerated postmortem JSON/MD and the top two entries of `CURRENT_STATUS.md` state those directions correctly.
4. Run py_compile, only `evaluation.tests.test_audit_conservative_l2_guarded_rerank_v2_full2638`, and `python evaluation/audit_conservative_l2_guarded_rerank_v2_full2638.py --recheck-pins`.
5. Confirm frozen raw SHA remains `3018f7593d31fd02854f3f63f992b6fc6293ba64c24185e2905877a71d88fdb1` and frozen development report `generated_at` remains `2026-10-05T16:51:40.293686+00:00`.

Return compact JSON only: `status`, `migration_counts`, `reports_consistent`, `status_consistent`, `focused_tests`, `pins`, `raw_sha256`, `frozen_generated_at`, `gpu_runs`, `files_modified`. Expected `files_modified=[]` and `gpu_runs=0`.
