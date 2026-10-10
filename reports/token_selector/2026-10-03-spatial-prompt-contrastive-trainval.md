# Prompt-contrastive spatial student train/val report (2026-10-03-spatial-prompt-contrastive-trainval)

- status: **PROMPT_CONTRASTIVE_STUDENT_NOT_ESTABLISHED**
- scope: train/val development validation only, CPU-only float32, ~458752-parameter Wt/Wv head (NOT 4B training), no GPU, no 4B model, no feature re-extraction, no raw image read, no CV-Bench, no network, no internal test NPZ open; internal sanity only, not final or CV-Bench accuracy
- development validation only: `True` (the val split was already used during development; not final accuracy)
- prereg config SHA256: `12d0aa9f856692fdd0da8e4d93783ef540ca9cadc38c9e4ee704ba903dd1dbbd` (unchanged=True)
- prereg report SHA256: `5513f89ca327fe992923f7f3e7fc6612fb506d74f1e3d1ec3e65748dd6f1816b`
- harness prompt SHA256: `d02fb7c94b633c22949e96bbae1315fd98a89258eede83e5ec3b0643888967c7`
- before/after hashes unchanged: `{'config_sha256': True, 'index_sha256': True, 'manifest_sha256': True, 'lock_sha256': True, 'spatial_audit_json_sha256': True, 'spatial_audit_script_sha256': True, 'hardware_config_sha256': True, 'hardware_code_sha256': True, 'existing_spatial_trainer_sha256': True, 'pair_loss_audit_sha256': True, 'distillation_code_sha256': True, 'identifiability_code_sha256': True, 'reduction_code_sha256': True, 'semantic_control_sha256': True, 'prereg_config': True, 'prereg_report': True, 'harness_prompt': True}`
- train NPZ aggregate SHA256: `33997394c4e148e3cfcd45eb0eb37fc97616cece8db3da5c731171d4b53d3951`
- val NPZ aggregate SHA256: `0439b7cb798f9dc49751d8135dd009294df723b2b10839837d7145778ca68f4b`
- test_npz_opened = `[]` (count=0, hashed=0)
- split: train={'sample_count': 318, 'image_count': 159} val={'sample_count': 76, 'image_count': 38} test={'sample_count': 78, 'image_count': 39}
- runtime_seconds=326.89 peak_rss_mib=2290.078125
- decision: **PROMPT_CONTRASTIVE_STUDENT_NOT_ESTABLISHED**; primary arm `B_SPATIAL_KL_CONTRASTIVE` all_pass=False

## Arm / seed losses

| arm | seed | selection | epoch0 val total | best epoch | best val total | best val base | best val contrastive | final val total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_SPATIAL_KL_CONTROL | 20261003 | val_base | 3.292221 | 27 | 2.928261 | 2.928261 | 0.627962 | 3.052110 |
| A_SPATIAL_KL_CONTROL | 20261004 | val_base | 3.361956 | 5 | 2.912712 | 2.912712 | 0.644539 | 3.099121 |
| A_SPATIAL_KL_CONTROL | 20261005 | val_base | 3.367746 | 4 | 2.915483 | 2.915483 | 0.649719 | 3.119844 |
| B_SPATIAL_KL_CONTRASTIVE | 20261003 | val_total | 3.982109 | 3 | 3.566311 | 2.929424 | 0.636888 | 3.853758 |
| B_SPATIAL_KL_CONTRASTIVE | 20261004 | val_total | 4.064741 | 5 | 3.566685 | 2.938110 | 0.628575 | 3.975294 |
| B_SPATIAL_KL_CONTRASTIVE | 20261005 | val_total | 4.057460 | 4 | 3.557198 | 2.921846 | 0.635352 | 4.046325 |

## Primary five-test Holm family (image-level)

| test | mean | ci_low | ci_high | dz | raw p | Holm p | +seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1_B_minus_A_correct_minus_same_spearman | 0.006824 | -0.002665 | 0.016364 | 0.2200 | 0.18968 | 0.33617 | 2/3 |
| P2_B_minus_A_correct_minus_same_ndcg_full | 0.011873 | 0.003461 | 0.021408 | 0.4114 | 0.01300 | 0.03900 | 3/3 |
| P3_B_correct_minus_same_spearman | 0.014950 | -0.005207 | 0.034627 | 0.2332 | 0.16808 | 0.33617 | 3/3 |
| P4_B_correct_minus_same_ndcg_full | 0.052582 | 0.023012 | 0.083270 | 0.5518 | 0.00120 | 0.00480 | 3/3 |
| P5_B_selector_minus_L2_anchor_mass_k48 | 0.226936 | 0.120464 | 0.335391 | 0.6760 | 0.00040 | 0.00200 | 3/3 |

## G1 descriptive non-inferiority (not in Holm)

- correct_spearman: mean=-0.005820 ci_low=-0.013497 noninferior_seeds=2/3
- correct_ndcg_full: mean=-0.004087 ci_low=-0.010059 noninferior_seeds=3/3

## Gates G1-G7 (arm B only)

- G1=True G2=False G3=False G4=True G5=True G6=True G7=True all_pass=False

## Limitations

- development validation only: the val split was already used during development; this is not final accuracy
- internal train/val sanity only; not CV-Bench task accuracy
- no hardware synthesis, latency or speedup claim
- no 4B model load, no feature re-extraction, no raw image read
- the internal test split was never resolved, hashed, opened or evaluated
- the prior spatial-student report's G3-style sign defect is only referenced historically; this experiment computes P5 as selector minus baseline
