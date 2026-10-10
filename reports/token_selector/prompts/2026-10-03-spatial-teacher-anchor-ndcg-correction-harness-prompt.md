# Harness correction task: anchor-baseline NDCG

Codex independently audited the completed spatial-teacher feasibility experiment and found one bounded implementation defect. Fix only this defect, rerun the audit/tests, and issue a transparent correction. Keep the preregistered spatial-target formulas, JS/TV decision rule, data split, inputs, and decision label frozen.

Project root: `/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`.
Python: `/home/zhaoxiangyu/conda-envs/robobrain25/bin/python`.

Security remains unchanged: CPU-only, no GPU/model/network/package install; no raw images; train NPZ only; no val/test NPZ; never modify frozen input files.

## Confirmed defect

In `evaluation/spatial_teacher_feasibility_audit.py::anchor_baseline_record`, the anchor-selected indices are used for retained target mass, but NDCG is computed as:

`ndcg_at_k(scores, target, effective)`

This ignores the anchor-selected set and therefore duplicates pure-L2 NDCG even when anchor retained mass differs.

## Required correction

1. Keep the exact existing `select_real_groups` selection rule and L2 input score.
2. For anchor NDCG, rank only the selected groups, ordered by descending original L2 score with the existing deterministic low-index tie policy. Compute DCG on that selected ordered list against the spatial target relevance, divided by the ideal DCG@K over all groups. Do not allow any unselected group into the predicted top K.
3. Prefer adding/reusing a small general metric helper such as `ndcg_for_selected_indices(selected, scores, relevance, k)` with explicit validation. Do not change the existing global `ndcg_at_k` semantics.
4. Identity bypass must still produce the same NDCG as pure L2 when all groups are selected.
5. Add focused tests that would fail under the old code:
   - anchor selection differs from pure L2 top-K and anchor NDCG differs accordingly;
   - selected-set ordering follows descending L2 with deterministic low-index tie handling;
   - unselected high-relevance groups do not enter anchor DCG;
   - identity bypass equals pure-L2 NDCG;
   - retained mass is unchanged by this correction.
6. Do not edit either preregistration file. Verify their SHA256 values remain:
   - config: `bfa1963597993ea9e3482735f16b2d430638ac3bf8e0682b5078f4e112974490`
   - report: `ce296bab00b82793b7f59781da126825680bb0d3cd8434e0e638f1f5bf1fcca9`
7. Rerun the full real audit, focused tests, and full `evaluation/tests` suite. Confirm the primary semantic/spatial pair metrics, paired JS/TV CIs, and `SPATIAL_SUPERVISION_SIGNAL_PRESENT` are bitwise/numerically unchanged. Anchor NDCG is expected to change; anchor retained mass must remain unchanged.
8. Add a clearly labeled correction note to the report JSON/MD and top `CURRENT_STATUS.md` entry with old and corrected anchor NDCG values. Do not rewrite older history or conceal the defect.
9. Refresh run/focused/full logs and durable harness result. Record old/new script and report/output SHA256 values.

Final response: compact JSON only with `status`, `defect_confirmed`, `correction`, `prereg_unchanged`, `primary_decision_unchanged`, `primary_metrics_unchanged`, `anchor_ndcg_old`, `anchor_ndcg_corrected`, `anchor_mass_unchanged`, `focused_tests`, `full_tests`, `files_modified`, `limitations`, `next_decision`.
