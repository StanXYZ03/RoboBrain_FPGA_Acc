# Spatial-supervision student train/val report (2026-10-03-spatial-student-trainval)

- status: **SPATIAL_STUDENT_NOT_ESTABLISHED**
- scope: train/val only, CPU-only float32, ~458752-parameter Wt/Wv head (NOT 4B training), no GPU, no 4B model, no feature re-extraction, no raw image read, no CV-Bench, no network, no internal test NPZ open; internal sanity only, not CV-Bench accuracy
- prereg config SHA256: `0ee94dca8e20535461f6526b026860a9160f7c8eea21920d0a38bf8610019abf` (unchanged=True)
- prereg report SHA256: `8da7bd2e7d7df8c94a776cb1bf1e0434e379e317b470f58523a96c5d9374d4a8`
- before/after hashes unchanged: `{'prereg_config': True, 'prereg_report': True, 'config': True, 'index': True, 'manifest': True, 'lock': True, 'spatial_audit_json': True, 'spatial_audit_script': True, 'semantic_control': True, 'hardware_config': True, 'hardware_code': True}`
- train NPZ aggregate SHA256: `33997394c4e148e3cfcd45eb0eb37fc97616cece8db3da5c731171d4b53d3951`
- val NPZ aggregate SHA256: `0439b7cb798f9dc49751d8135dd009294df723b2b10839837d7145778ca68f4b`
- test_npz_opened = `[]` (count=0)
- split: train={'sample_count': 318, 'image_count': 159} val={'sample_count': 76, 'image_count': 38} test={'sample_count': 78, 'image_count': 39}
- runtime_seconds=185.53 peak_rss_mib=2287.64453125
- decision: **SPATIAL_STUDENT_NOT_ESTABLISHED**; primary arm `B_SPATIAL_KL_PRIMARY` all_pass=False

## Arm / seed losses

| arm | seed | epoch0 val KL | best epoch | best val KL | final val KL | epoch0 hash | best hash |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_SEMANTIC_KL_CONTROL | 20261003 | 2.668297 | 12 | 2.503504 | 2.676422 | `f6467f8fea80cd31` | `2cf6a215c65114e6` |
| A_SEMANTIC_KL_CONTROL | 20261004 | 2.712207 | 2 | 2.520495 | 2.633174 | `386dc2fc71bb0298` | `26bfddde4807406d` |
| A_SEMANTIC_KL_CONTROL | 20261005 | 2.662997 | 2 | 2.511975 | 2.671205 | `2a2311a1ff430bd9` | `da4611c693540f9f` |
| B_SPATIAL_KL_PRIMARY | 20261003 | 3.292221 | 27 | 2.928261 | 3.052110 | `f6467f8fea80cd31` | `a66c6f55d7555929` |
| B_SPATIAL_KL_PRIMARY | 20261004 | 3.361956 | 5 | 2.912712 | 3.099121 | `386dc2fc71bb0298` | `b26354df4a58f82f` |
| B_SPATIAL_KL_PRIMARY | 20261005 | 3.367746 | 4 | 2.915483 | 3.119844 | `2a2311a1ff430bd9` | `da3d32464f5a0d14` |

## Primary five-test Holm family (image-level)

| test | mean | ci_low | ci_high | dz | raw p | Holm p | +seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B_minus_A_correct_spearman | 0.116239 | 0.072465 | 0.160201 | 0.8425 | 0.00030 | 0.00120 | 3/3 |
| B_minus_A_correct_ndcg_full | 0.072518 | 0.023511 | 0.118175 | 0.4787 | 0.00480 | 0.01170 | 3/3 |
| B_correct_minus_same_spearman | 0.010183 | -0.002846 | 0.022837 | 0.2438 | 0.15068 | 0.15068 | 3/3 |
| B_correct_minus_same_ndcg_full | 0.033670 | 0.011946 | 0.056137 | 0.4770 | 0.00390 | 0.01170 | 3/3 |
| B_selector_minus_L2_anchor_mass_k48 | -0.246583 | -0.352744 | -0.141682 | -0.7394 | 0.00010 | 0.00050 | 0/3 |

## Gates G1-G6 (arm B only)

- G1=True G2=False G3=False G4=True G5=True G6=True all_pass=False

## Limitations

- internal train/val sanity only; not CV-Bench task accuracy
- no hardware synthesis, latency or speedup claim
- no 4B model load, no feature re-extraction, no raw image read
- the internal test split was never opened or evaluated
