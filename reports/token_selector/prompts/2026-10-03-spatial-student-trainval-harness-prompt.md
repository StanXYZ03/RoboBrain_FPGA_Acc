# Harness task: preregistered spatial-supervision student train/val experiment

Execute this experiment end to end. Codex owns the research decision; keep reasoning/output concise. Reuse existing project functions and do not alter any prior frozen input, preregistration, report, checkpoint, app, or model file.

Project root: `/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`
Python: `/home/zhaoxiangyu/conda-envs/robobrain25/bin/python`

Security: CPU-only float32 training; no GPU, 4B model, feature extraction, raw-image read, CV-Bench, network, or package install. Use cached derived NPZ locally only; never upload tensor payloads. TRAIN/VAL only; never resolve/hash/open/score internal test NPZ.

## Question

Does replacing the weak task-specific semantic teacher with the preregistered bbox/point spatial teacher make the existing hardware-cheap `prompt_mean_vector` student learn task-conditioned spatial token scores?

Architecture remains frozen:
- `Wt[128,2560]`, `Wv[128,1024]`
- `query = normalize(mean_t(normalize(prompt_embeddings @ Wt.T)))`
- `visual = row_normalize(group_mean @ Wv.T)`
- score = `visual @ query`, student distribution = `softmax(score/0.1)`
- runtime query `[1,128]`, FPGA scorer `Gx128` dot, no token-axis max/softmax/exp

## Pre-registration before any new validation read

Write and freeze:
- `configs/spatial_student_trainval.preregistered.json`
- `reports/token_selector/2026-10-03-spatial-student-trainval-preregistration.json`

Pin paths/SHA256 for the clean config/index/manifest/lock, all train NPZ aggregate, corrected spatial-feasibility JSON and script, existing semantic control training code, hardware selector config/code, and prereg files. Record formulas, arms, optimizer, seeds, statistics, gates, prohibitions. Verify before/after byte immutability.

## Frozen arms and optimization

Two arms, only B can pass:
- `A_SEMANTIC_KL_CONTROL`: target = cached `teacher_soft_target`.
- `B_SPATIAL_KL_PRIMARY`: grounding target = exact bbox/grid-cell area target; pointing target = bilinear point target from corrected `evaluation.spatial_teacher_feasibility_audit`.

Loss for each record: `KL(target || softmax(student_score/0.1))`; image-pair batch loss is the mean of grounding and pointing. No pair hinge, target blending, margin, rank loss, auxiliary loss, or hyperparameter search.

Optimization:
- seeds `20261003, 20261004, 20261005`
- same epoch-0 Wt/Wv and same deterministic image-pair batch order across arms for each seed
- AdamW, lr `1e-3`, weight decay `0`, 30 epochs, batch 4 image pairs, grad clip 1.0
- CPU float32, one torch thread, deterministic algorithms
- epoch 0 is pre-update
- best checkpoint selected only by each arm's own VAL KL, tie earliest
- save epoch0 hash and best/final only; no per-epoch weight files

## Frozen validation metrics

Independent unit is image. For each seed/best checkpoint compute:
1. Correct-prompt spatial alignment per record: Spearman, full NDCG, target mass at K48/64/96.
2. Same-image other-task prompt control: score the same visual groups using the paired task prompt; deltas are correct minus wrong for Spearman/NDCG.
3. Cross-image same-task deterministic derangement as descriptive secondary control.
4. Deployed selection proxy at K48: use existing score normalization/combine rule from `configs/hardware_regular_token_selector.json` and exact existing `select_real_groups` 3x3 anchor rule. Compare B prompt-conditioned selector retained spatial-target mass against L2+anchor baseline. Do not invent another anchor rule. Use corrected selected-set NDCG helper where NDCG is reported.
5. Exact prompt-score invariance/collapse fraction.

Across seeds, take per-image median delta, then image-level bootstrap 2000, two-sided sign-flip 10000, Cohen dz, deterministic seed 20261003. Primary Holm family is exactly 5 tests:
- B minus A: correct spatial Spearman and full NDCG (2)
- B correct minus same-image other-task: Spearman and full NDCG (2)
- B deployed selector minus L2+anchor: K48 retained target mass (1)

Report raw and Holm-adjusted p-values, 95% CI, dz, and positive-seed counts. Other K/method metrics are descriptive and cannot decide the result.

## Frozen gates for B only

- G1 spatial-supervision gain: both B-A Spearman/NDCG deltas mean>0, CI lower>0, |dz|>=0.2, Holm p<0.05, positive mean in >=2/3 seeds.
- G2 prompt sensitivity: both correct-vs-same-image-other-task Spearman/NDCG satisfy the same five conditions.
- G3 deployed K48 utility: B selector minus L2+anchor retained target-mass delta satisfies the same five conditions.
- G4 collapse removed: exact prompt-score invariance fraction <=0.05 in every seed.
- G5 validation learning: >=2/3 B seeds have best_epoch>0 and best VAL spatial KL < epoch0 VAL spatial KL.
- G6 hardware contract remains `[1,128]` + `Gx128` dot and existing fixed-budget 3x3 anchor selection.

Decision:
- `SPATIAL_STUDENT_LEARNABLE` only if B passes G1-G6.
- otherwise `SPATIAL_STUDENT_NOT_ESTABLISHED`.
- A is control and can never win.
- no internal test evaluation in either case.

## Outputs

Create new files only:
- `evaluation/train_spatial_prompt_selector.py`
- `evaluation/tests/test_train_spatial_prompt_selector.py`
- `artifacts/token_selector/spatial_student_trainval/<arm>/<seed>/{best.npz,final.npz,metadata.json}`
- `reports/token_selector/2026-10-03-spatial-student-trainval-epochs.csv`
- `reports/token_selector/2026-10-03-spatial-student-trainval-val-samples.csv`
- `reports/token_selector/2026-10-03-spatial-student-trainval.{json,md}`
- run/focused/full/harness-result logs with the same stem under `reports/token_selector/logs/`

Tests must cover target choice by arm/task, shared initialization and batches, exact KL reference, val-only best selection, corrupted internal-test NPZ isolation, five-test Holm family, all gates, corrected selected-set NDCG use, checkpoints, prereg immutability, and report schemas.

Run focused tests, the formal two-arm/three-seed/30-epoch experiment, then full `evaluation/tests`. Update the top of `CURRENT_STATUS.md` after completion with decision, key values, limitations, evidence, and exact next decision. Final response: compact JSON only with `status`, `decision`, `prereg_sha256`, `arms`, `seeds`, `train_records`, `val_records`, `test_npz_opened`, `best_epochs`, `primary_deltas`, `gates`, `focused_tests`, `full_tests`, `runtime_seconds`, `peak_rss_mib`, `files_created`, `limitations`, `next_decision`.
