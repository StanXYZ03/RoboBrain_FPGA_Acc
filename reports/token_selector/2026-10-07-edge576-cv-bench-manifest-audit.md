# edge576 CV-Bench manifest 独立审计报告（2026-10-07）

- 状态：`EDGE576_MANIFEST_AUDIT_PASS`（CPU-only、只读、`gpu_runs=0`、未加载模型权重）
- 审计脚本：`evaluation/audit_edge576_cv_bench_manifest.py`，
  SHA256 `45dcd04976e5c5c8acaf232989f219b1817b5a13fb87bfbe27dd4cbfe81ea683`
- 被审计 manifest：`datasets/2026-10-07-cv-bench-edge576-full2638.jsonl`，
  SHA256 `b5355ba1636011b4af101ace16c170b061ecf44a69e5c75fe2a64f4288c7dd1f`（3280730 B）
- 源冻结 manifest：`datasets/cv_bench_full2638.jsonl`，
  SHA256 `22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91`

## 合同复核结果（全部通过）

| 合同 | 结果 |
|---|---|
| 恰 2638 条、唯一 `sample_id` | ✅ |
| ID 集合与冻结 full manifest 一致 | ✅ |
| `source`/`source_dataset`/`task`/`type`/`prompt`/`choices`/`answer`/`max_new_tokens` 逐字段一致 | ✅ |
| `record_sha256` 全部现场重算一致 | ✅ |
| 2638 张图片 SHA256 现场重算一致 | ✅ |
| 2638 条原始宽高现场重算一致 | ✅ |
| `full_patch_tokens == product(image_grid_thw)` 且 `G == full_patch_tokens//4` | ✅ |
| 全部 `G <= 576` | ✅ |
| 处理参数与冻结常量一致 | ✅ |
| 摘要 JSON 与现场重算统计一致 | ✅ |
| 用权威 processor 逐样本重算 `image_grid_thw`（2638/2638）完全一致 | ✅ |

## 声明

- 审计只打开 processor，未加载模型权重、未使用 GPU、未联网、未训练；
- 审计只读 manifest/summary，不修改任何既有文件；
- 通过审计不代表任何精度/部署/加速/FPGA 时延主张；full2638 仍是发展集。
