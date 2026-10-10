# Current Selector CV-Bench gate summary

- generated_at: 2026-10-03T17:44:25.775164+00:00
- manifest: `/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/datasets/cv_bench_full2638.jsonl`
- manifest_sha256: `22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91`
- status: COMPLETE
- forbidden primary metrics (never reported): pointing, grounding, ndcg, retained_mass

## K48
- status: COMPLETE, samples: 2638
- candidate: combined=0.653429 2D=0.591024 3D=0.715833 parse=0.988628
  - tasks: Count=0.408629, Depth=0.728333, Distance=0.703333, Relation=0.818462
- dense_bf16: combined=0.872730 2D=0.798793 3D=0.946667 parse=0.984837
  - tasks: Count=0.682741, Depth=0.981667, Distance=0.911667, Relation=0.947692
- native_bf16: combined=0.872730 2D=0.798793 3D=0.946667 parse=0.984837
  - tasks: Count=0.682741, Depth=0.981667, Distance=0.911667, Relation=0.947692
- absolute_delta_vs_dense: -0.219301
- relative_retention_vs_dense: 0.748718
- paired_bootstrap_ci95_vs_dense: [-0.23824303572655103, -0.20129053702667976]

## K64
- status: COMPLETE, samples: 2638
- candidate: combined=0.703331 2D=0.634995 3D=0.771667 parse=0.989007
  - tasks: Count=0.469543, Depth=0.786667, Distance=0.756667, Relation=0.843077
- dense_bf16: combined=0.872730 2D=0.798793 3D=0.946667 parse=0.984837
  - tasks: Count=0.682741, Depth=0.981667, Distance=0.911667, Relation=0.947692
- native_bf16: combined=0.872730 2D=0.798793 3D=0.946667 parse=0.984837
  - tasks: Count=0.682741, Depth=0.981667, Distance=0.911667, Relation=0.947692
- absolute_delta_vs_dense: -0.169399
- relative_retention_vs_dense: 0.805897
- paired_bootstrap_ci95_vs_dense: [-0.18659766173108439, -0.15283475856858297]

## K96
- status: COMPLETE, samples: 2638
- candidate: combined=0.764869 2D=0.699739 3D=0.830000 parse=0.986353
  - tasks: Count=0.546954, Depth=0.835000, Distance=0.825000, Relation=0.895385
- dense_bf16: combined=0.872730 2D=0.798793 3D=0.946667 parse=0.984837
  - tasks: Count=0.682741, Depth=0.981667, Distance=0.911667, Relation=0.947692
- native_bf16: combined=0.872730 2D=0.798793 3D=0.946667 parse=0.984837
  - tasks: Count=0.682741, Depth=0.981667, Distance=0.911667, Relation=0.947692
- absolute_delta_vs_dense: -0.107860
- relative_retention_vs_dense: 0.876410
- paired_bootstrap_ci95_vs_dense: [-0.12297794442318433, -0.09383393188307687]

