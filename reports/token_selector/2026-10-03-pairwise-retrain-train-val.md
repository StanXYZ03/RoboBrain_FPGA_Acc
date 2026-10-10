# Pairwise-retrain train/val report (2026-10-03-pairwise-retrain)

- status: **NO_PROMOTION**
- scope: train/val only, CPU-only float32, ~458752-parameter Wt/Wv head (NOT 4B training), no GPU, no 4B model, no feature re-extraction, no CV-Bench, no network, no test NPZ open; internal sanity only, not CV-Bench accuracy
- prereg config SHA256: `c8327ddb529970377707fd15df8fee410f293e747ddf80cd88f23177c771ef3c` (unchanged=True)
- prereg report SHA256: `44f21c9cb6d1a7c1dca41900d367e08cc7a350ce44a99508cf5915f221562d99`
- draft SHA256: `787f8f401bcb10b5fe8007a6a0f5b415858d893cb2797b3c1797d5f1fa785772` (unchanged=True)
- config SHA256: `44399c8e6fb8f333a150a6016faa00d6ae70524b5fcd030bcabf4e5fb07ab28b`
- index SHA256: `02a50868a84675418e9bd9e5fdd0bb55f522ec5ce0ce1d6da7c69f85d2aeee26`
- before/after hashes unchanged: `{'prereg_config': True, 'prereg_report': True, 'draft': True, 'config': True, 'index': True}`
- train NPZ aggregate SHA256: `33997394c4e148e3cfcd45eb0eb37fc97616cece8db3da5c731171d4b53d3951`
- val NPZ aggregate SHA256: `0439b7cb798f9dc49751d8135dd009294df723b2b10839837d7145778ca68f4b`
- test_npz_opened = `[]` (count=0)
- split: train={'sample_count': 318, 'image_count': 159} val={'sample_count': 76, 'image_count': 38} test={'sample_count': 78, 'image_count': 39}
- runtime_seconds=283.71 peak_rss_mib=2282.8125
- 4B model loaded: False; GPU used: False

## Arm / seed losses

| arm | seed | epoch0 val L_total | best epoch | best val L_total | final val L_total | epoch0 hash | best hash |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_KL_ONLY_CONTROL | 20261002 | 2.858936 | 2 | 2.507506 | 2.670728 | `40bc0c1b3d6e2538` | `b4cc1b8b3ce340ac` |
| A_KL_ONLY_CONTROL | 20261003 | 2.668297 | 12 | 2.503504 | 2.676422 | `f6467f8fea80cd31` | `2cf6a215c65114e6` |
| A_KL_ONLY_CONTROL | 20261004 | 2.712207 | 2 | 2.520495 | 2.633174 | `386dc2fc71bb0298` | `26bfddde4807406d` |
| B_PAIR_LAMBDA_1_CONTROL | 20261002 | 2.917403 | 2 | 2.566526 | 2.711077 | `40bc0c1b3d6e2538` | `f1f2641ddaa5b80f` |
| B_PAIR_LAMBDA_1_CONTROL | 20261003 | 2.727380 | 12 | 2.560553 | 2.748783 | `f6467f8fea80cd31` | `ec4e39382a80aede` |
| B_PAIR_LAMBDA_1_CONTROL | 20261004 | 2.775997 | 6 | 2.574429 | 2.705466 | `386dc2fc71bb0298` | `447db8c400811ea0` |
| C_PAIR_LAMBDA_0P25_PRIMARY | 20261002 | 3.693848 | 23 | 3.210765 | 3.280221 | `40bc0c1b3d6e2538` | `92f1d13df72f9474` |
| C_PAIR_LAMBDA_0P25_PRIMARY | 20261003 | 3.512015 | 5 | 3.191505 | 3.362547 | `f6467f8fea80cd31` | `b65dfa918bc4c3a1` |
| C_PAIR_LAMBDA_0P25_PRIMARY | 20261004 | 3.623136 | 28 | 3.244140 | 3.274614 | `386dc2fc71bb0298` | `2e36deec05ea3981` |

## Primary deltas (per-image median across seeds, image-level)

| arm | control | metric | mean | ci_low | ci_high | dz | p | holm_p | +seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_KL_ONLY_CONTROL | same_image_other_task | spearman | 0.000456 | -0.000741 | 0.001771 | 0.1147 | 0.49165 | 1.00000 | 3/3 |
| A_KL_ONLY_CONTROL | same_image_other_task | ndcg_full | 0.000055 | -0.000293 | 0.000356 | 0.0524 | 0.76022 | 1.00000 | 2/3 |
| A_KL_ONLY_CONTROL | cross_image_same_task_derangement | spearman | 0.000099 | -0.001159 | 0.001258 | 0.0253 | 0.87401 | 1.00000 | 1/3 |
| A_KL_ONLY_CONTROL | cross_image_same_task_derangement | ndcg_full | -0.000240 | -0.000787 | 0.000204 | -0.1488 | 0.41626 | 1.00000 | 1/3 |
| B_PAIR_LAMBDA_1_CONTROL | same_image_other_task | spearman | 0.001333 | -0.001603 | 0.004520 | 0.1377 | 0.40536 | 1.00000 | 3/3 |
| B_PAIR_LAMBDA_1_CONTROL | same_image_other_task | ndcg_full | 0.000416 | -0.000433 | 0.001299 | 0.1474 | 0.38006 | 1.00000 | 3/3 |
| B_PAIR_LAMBDA_1_CONTROL | cross_image_same_task_derangement | spearman | -0.001230 | -0.005325 | 0.002064 | -0.1060 | 0.55284 | 1.00000 | 1/3 |
| B_PAIR_LAMBDA_1_CONTROL | cross_image_same_task_derangement | ndcg_full | -0.000069 | -0.001413 | 0.001115 | -0.0175 | 0.92181 | 1.00000 | 1/3 |
| C_PAIR_LAMBDA_0P25_PRIMARY | same_image_other_task | spearman | 0.006232 | -0.003820 | 0.015884 | 0.1951 | 0.24338 | 1.00000 | 3/3 |
| C_PAIR_LAMBDA_0P25_PRIMARY | same_image_other_task | ndcg_full | 0.000666 | -0.000972 | 0.002342 | 0.1241 | 0.44266 | 1.00000 | 3/3 |
| C_PAIR_LAMBDA_0P25_PRIMARY | cross_image_same_task_derangement | spearman | 0.001635 | -0.003532 | 0.006851 | 0.0959 | 0.55714 | 1.00000 | 1/3 |
| C_PAIR_LAMBDA_0P25_PRIMARY | cross_image_same_task_derangement | ndcg_full | 0.000020 | -0.000733 | 0.000722 | 0.0085 | 0.95850 | 1.00000 | 0/3 |

## Gates A-F

- **A_KL_ONLY_CONTROL**: A=False B=False C=True D=True E=True F=True all_pass=False
- **B_PAIR_LAMBDA_1_CONTROL**: A=False B=False C=True D=False E=True F=True all_pass=False
- **C_PAIR_LAMBDA_0P25_PRIMARY**: A=False B=False C=True D=False E=True F=True all_pass=False

**Fixed winner rule**: primary arm `C_PAIR_LAMBDA_0P25_PRIMARY`; winner=None; decision=**NO_PROMOTION**.

## Limitations

- internal train/val sanity only; not CV-Bench task accuracy
- no hardware synthesis, latency or speedup claim
- concentration evidence is explanatory, frozen before the run
- the internal test split was never opened

## Evidence: SHAs, tests, isolation

- source trainer SHA256: `07b504d79743915b7b8dacf7028dea1de91ba36bb2776dedd6ed9e4012032fec`
- source test SHA256: `52b1bf36ebb784e09906d12f81cbb4c07aed2bc96d0d46be50b87de9d9db9dfd`
- preregistered config SHA256: `c8327ddb529970377707fd15df8fee410f293e747ddf80cd88f23177c771ef3c`
- preregistration report SHA256: `44f21c9cb6d1a7c1dca41900d367e08cc7a350ce44a99508cf5915f221562d99`
- epochs CSV SHA256: `712806acfa16bc77d359aad1936e81a15b4c35f8e4f0151ff1459e004bf4df70`
- val-samples CSV SHA256: `8961f47bfea87e7f4eecb62ae4fe89c24f3e6610a912cfcc09b1c43f3729341c`
- checkpoint / metadata SHA256:
  - `artifacts/token_selector/pairwise_retrain/A_KL_ONLY_CONTROL/20261002/best.npz` = `72a1f765fca6094c95c94b8fce45839cdaac43f394ec986c727117772d5b413b`
  - `artifacts/token_selector/pairwise_retrain/A_KL_ONLY_CONTROL/20261002/final.npz` = `da1e6b151c84b00ebf11a019373c2672feeb9cf8b412d3cdd7142cd9590e0672`
  - `artifacts/token_selector/pairwise_retrain/A_KL_ONLY_CONTROL/20261002/metadata.json` = `4ef86229550d66a38024a072ff9f33fe57151b9e2b799322c0c44ff2d496026d`
  - `artifacts/token_selector/pairwise_retrain/A_KL_ONLY_CONTROL/20261003/best.npz` = `8a2aeb13f2e38a37410a9e67d964a0a739ebbe20051914f4df31ff0d5ff82a51`
  - `artifacts/token_selector/pairwise_retrain/A_KL_ONLY_CONTROL/20261003/final.npz` = `02444b4ac7bf5e17e42782127f27c43a8b5b7d4e731a2616b18e4b5191819b3a`
  - `artifacts/token_selector/pairwise_retrain/A_KL_ONLY_CONTROL/20261003/metadata.json` = `e250ed092c4ce11acec85e17163203f69f30663035493ff7e08c350c796d7bf8`
  - `artifacts/token_selector/pairwise_retrain/A_KL_ONLY_CONTROL/20261004/best.npz` = `598cee7fe8697f3c574e69ab232c3c2efaf0a405fa5976b1369e2b2e655eb110`
  - `artifacts/token_selector/pairwise_retrain/A_KL_ONLY_CONTROL/20261004/final.npz` = `8138ea6f548d2d3187c6e47f0a2646ce5162fbcb29231d575849a439e8367045`
  - `artifacts/token_selector/pairwise_retrain/A_KL_ONLY_CONTROL/20261004/metadata.json` = `6c867d0d3879b82907c4d33040b99cb59b1990d4c9bda8603e7029f3fa0c091d`
  - `artifacts/token_selector/pairwise_retrain/B_PAIR_LAMBDA_1_CONTROL/20261002/best.npz` = `974525a78b51fe2dd130be6eb778e426584d13d45671d0f5f7dbcb4641d6129b`
  - `artifacts/token_selector/pairwise_retrain/B_PAIR_LAMBDA_1_CONTROL/20261002/final.npz` = `920c1d2b7b3eaa8a6f1c15147393c773fd6d4437607090304a3ceb74b2522b71`
  - `artifacts/token_selector/pairwise_retrain/B_PAIR_LAMBDA_1_CONTROL/20261002/metadata.json` = `71fdb0bc0b3fbe43536b752ed842efab54f2e7149c1f8c3aaf3f00eafc1dddae`
  - `artifacts/token_selector/pairwise_retrain/B_PAIR_LAMBDA_1_CONTROL/20261003/best.npz` = `2161474a422a5bd62420e00c3bb0958259673baaccbd30966ae2118e365e60c8`
  - `artifacts/token_selector/pairwise_retrain/B_PAIR_LAMBDA_1_CONTROL/20261003/final.npz` = `5c7153b20857814824eb78c99dbb7ce2bcc57079a1f64fe59188b15161669883`
  - `artifacts/token_selector/pairwise_retrain/B_PAIR_LAMBDA_1_CONTROL/20261003/metadata.json` = `fbf7bf4f6927f2faf6a8c2fccac368a2fe60e9b2c6ef464e6c9503f9e2bacdfb`
  - `artifacts/token_selector/pairwise_retrain/B_PAIR_LAMBDA_1_CONTROL/20261004/best.npz` = `5c4b61c432e43e02571e6036bea6001adddf0350700ac2261bbcf90494b56df9`
  - `artifacts/token_selector/pairwise_retrain/B_PAIR_LAMBDA_1_CONTROL/20261004/final.npz` = `ffa4963e48c61424ec19655da2b4aa371eda6c3ac2ace974219d60e160ebadc6`
  - `artifacts/token_selector/pairwise_retrain/B_PAIR_LAMBDA_1_CONTROL/20261004/metadata.json` = `c1daf2a29e50a4d73f2904448415420c2a6555e2be0ca607970a07d22551bd78`
  - `artifacts/token_selector/pairwise_retrain/C_PAIR_LAMBDA_0P25_PRIMARY/20261002/best.npz` = `0fbd585294d1ae4b23cd4665c49bc263f4aa7362d145d92882926b4dc04c3386`
  - `artifacts/token_selector/pairwise_retrain/C_PAIR_LAMBDA_0P25_PRIMARY/20261002/final.npz` = `10e57dca509da047cfb1d8a28bb0974c0ba180aa646cd043a3e4095fbf2fb10c`
  - `artifacts/token_selector/pairwise_retrain/C_PAIR_LAMBDA_0P25_PRIMARY/20261002/metadata.json` = `7a7c594cb02317ed58aaaec57b06df1e4ed91dfd16f0901d7a940274621ad00d`
  - `artifacts/token_selector/pairwise_retrain/C_PAIR_LAMBDA_0P25_PRIMARY/20261003/best.npz` = `da07e0aa3d0dd345fd01dc8e53d01c9774f55acbe104e0bfd1ff0e28a9a08281`
  - `artifacts/token_selector/pairwise_retrain/C_PAIR_LAMBDA_0P25_PRIMARY/20261003/final.npz` = `0651227763d7dd9d8cdb2090eb7945891147173dcb2d8bafcd8e55acce7f7200`
  - `artifacts/token_selector/pairwise_retrain/C_PAIR_LAMBDA_0P25_PRIMARY/20261003/metadata.json` = `a7c25f2bb3941f8fc60a0cb48f45bd2056b358d5fe57efff11520cdacdecb2e7`
  - `artifacts/token_selector/pairwise_retrain/C_PAIR_LAMBDA_0P25_PRIMARY/20261004/best.npz` = `77a54709f17cea18d749a4b9c79b99dcdbf40b89a71f4b8c8a963ddc7ab82ab6`
  - `artifacts/token_selector/pairwise_retrain/C_PAIR_LAMBDA_0P25_PRIMARY/20261004/final.npz` = `413f9adac51fdb165c4cfb9ac3289136351b6714acc663797fb7caf9262b4338`
  - `artifacts/token_selector/pairwise_retrain/C_PAIR_LAMBDA_0P25_PRIMARY/20261004/metadata.json` = `fbc5a3fcb2af74bc8d53db59f51ee1ed42f54754b7ee9928d11a3b180ac69c9d`
- focused tests: Ran 17 tests OK (0 failures, 0 errors)
- full tests: Ran 491 tests OK (0 failures, 0 errors)
- `test_npz_opened = []`
- before/after frozen hashes unchanged: `{'config': True, 'draft': True, 'index': True, 'prereg_config': True, 'prereg_report': True}`

