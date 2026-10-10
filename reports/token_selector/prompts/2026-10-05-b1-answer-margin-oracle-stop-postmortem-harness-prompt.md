# Harness task: B1 answer-margin Oracle PILOT_STOP postmortem

Work only in `/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102` using `/home/zhaoxiangyu/conda-envs/robobrain25/bin/python`.

Implement a concise, CPU-only, read-only-over-frozen-results postmortem of the completed challenge/control pilot. Do not run GPU/model weights/network/training, do not change any frozen manifest/config/raw result/summary, do not tune thresholds/K/anchors, and do not propose that the preregistered STOP be overturned.

Frozen evidence:

- manifest: `datasets/2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24.jsonl`
- config: `configs/2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24.preregistered.json`
- Oracle raw: `artifacts/token_selector/2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24/oracle_answer_margin_b1_saliency_k96.jsonl`
- Oracle summary: `reports/token_selector/2026-10-05-b1-answer-margin-oracle-challenge-control-pilot-summary.json`
- corrected L2 control: `artifacts/token_selector/2026-10-04-b1-corrected-l2-prompt-ablation-full2638/corrected_l2_integer_k96_full2638.jsonl`

Known frozen decision: repairs 10/12 pass, regressions 3/12 fail, parse 24/24, exact 96/87/9 layout 24/24, final `PILOT_STOP`.

Create a new analysis script and tests, then generate:

- `reports/token_selector/2026-10-05-b1-answer-margin-oracle-stop-postmortem.json`
- `reports/token_selector/2026-10-05-b1-answer-margin-oracle-stop-postmortem.csv`
- `reports/token_selector/2026-10-05-b1-answer-margin-oracle-stop-postmortem.md`

Required analyses, using only already-recorded fields:

1. Verify all input SHA256, 24 unique identical IDs, record hashes, frozen transition labels and exact K96/87+9 layout before analysis.
2. Per sample and by transition (`W->R`, `R->W`, unchanged-correct, unchanged-wrong), compute:
   - all-slot selected-index Jaccard/intersection between Oracle and corrected L2;
   - real-slot-only Jaccard/intersection using each arm's `source_type`;
   - context representative overlap;
   - fraction of the recorded Oracle absolute gradient saliency mass retained by Oracle-selected real indices and by L2-selected real indices; report paired difference;
   - full-sequence answer margin, G, compression ratio, and any recorded timing/peak-memory fields only as diagnostics.
3. Report the three regression sample IDs and the two unrepaired challenge IDs explicitly, with their metrics and cells.
4. Test only descriptive associations that do not tune a rule: compare transition-group means/medians and Spearman correlations of margin/G/Jaccard/retained-mass advantage with the signed transition outcome. Because n=24 and outcome-enriched, label all associations exploratory and do not use CI/significance/p-values.
5. State what can and cannot be concluded. In particular audit these possible explanations separately:
   - low dense answer margin;
   - excessive selection churn relative to L2;
   - Oracle retains more absolute gradient mass but absolute `|grad*activation|` is unsigned and may not preserve the correct-answer margin after nonlinear discrete K96 pruning;
   - context/anchor rule interaction cannot be isolated if required metadata are absent.
   Mark unsupported mechanisms as unproven, not facts.
6. End with a machine-readable decision boundary: the direct absolute-gradient full-replacement teacher remains stopped. Recommend at most one next mechanism class based on evidence, without specifying/tuning a numeric mixture on these 24 samples. A permissible candidate is a conservative L2-backbone/residual task-aware re-ranking mechanism, but only recommend it if the observed evidence supports it; otherwise say no next selector mechanism is supported.
7. Update `CURRENT_STATUS.md` with the frozen STOP and postmortem evidence. Do not claim official accuracy, deployability, FPGA speedup or representative sampling.

Add focused CPU tests for malformed inputs/hashes/layout, index/source-type consistency, metric formulas and frozen expected aggregate values. Run py_compile, focused tests, and full `evaluation/tests`. Final stdout one compact JSON with `status`, `files_changed`, `regression_ids`, `unrepaired_ids`, `key_metrics`, `supported_conclusions`, `unsupported_conclusions`, `focused_tests`, `full_tests`, `gpu_runs`.
