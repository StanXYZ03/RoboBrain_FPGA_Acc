# edge576 CV-Bench manifest 构建报告（2026-10-07）

- 状态：`EDGE576_MANIFEST_BUILT`（CPU-only、`gpu_runs=0`、未加载模型权重）
- 样本数：2638（唯一 `sample_id` 2638）
- 输入冻结 manifest：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/datasets/cv_bench_full2638.jsonl` SHA `22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91`
- 输出 manifest：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/datasets/2026-10-07-cv-bench-edge576-full2638.jsonl` SHA `b5355ba1636011b4af101ace16c170b061ecf44a69e5c75fe2a64f4288c7dd1f`（3280730 B）
- 处理参数：保持长宽比、`min_pixels=65536`、`max_pixels=589824`、`patch_size=16`、`merge_size=2`、`G<= 576`
- 公式：full_llm_visual_tokens = (image_grid_thw[0]*image_grid_thw[1]*image_grid_thw[2]) // (spatial_merge_size**2)

## G 分布（full_llm_visual_tokens）

- min/median/mean/p90/p95/max = 64/300.0/357.07/576.0/576.0/576
- 恰好等于 576 的样本数：400
- `G<=96`（identity bypass）：143；`96<G<=576`（压缩）：2495
- 分桶：<=96=143、97-256=781、257-512=706、513-576=1008
- 源分布：{'ADE20K': 633, 'COCO': 805, 'Omni3D': 1200}
- 任务分布：{'Count': 788, 'Depth': 600, 'Distance': 600, 'Relation': 650}

## 合同

- 全部 2638 条均以权威 `image_grid_thw` 实测，未事后删除任何高 G 样本；
- ID/标签/源/任务与冻结 full manifest 完全一致；
- 每条记录含图片 SHA256、原始尺寸、`image_grid_thw`、patch/LLM token、处理参数与 `record_sha256`；
- 任意 `G>576` 即 fail closed 且不写出 manifest。

## 声明

- 本构建只打开 processor，未加载模型权重、未使用 GPU、未联网、未训练；
- full2638 是发展集，不是未触碰确认集；
- 本 manifest 冻结不代表任何精度/部署/加速主张。
