# edge576 CV-Bench manifest 独立复审报告 r2（2026-10-07）

- 状态：`EDGE576_MANIFEST_AUDIT_PASS`（CPU-only、只读、`gpu_runs=0`、未加载模型权重）
- 审计脚本（已修复 `--model` 丢失缺陷）：
  `evaluation/audit_edge576_cv_bench_manifest.py`，
  SHA256 `a999fe0a48b962104b9991df1c62329789a2164f530fe977767dd95ddbad352b`
- 实际处理器路径（写入报告）：
  `/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B`（`processor_model_path`）
- 被审计 manifest：`datasets/2026-10-07-cv-bench-edge576-full2638.jsonl`，
  SHA256 `b5355ba1636011b4af101ace16c170b061ecf44a69e5c75fe2a64f4288c7dd1f`（3280730 B）
- 源冻结 manifest：`datasets/cv_bench_full2638.jsonl`，
  SHA256 `22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91`
- 本报告 JSON：`reports/token_selector/2026-10-07-edge576-cv-bench-manifest-audit-r2.json`，
  SHA256 `34f1385fbe71aa7a947798614b71fe4da9e64ef56449d3b41ac6c52e7886469e`

## 修复说明（缺陷 C）

- 旧审计器 `audit()` 不接受模型路径，`main` 解析 `--model` 后丢弃，`verify_processor`
  硬用 `DEFAULT_MODEL`。
- 现在 `audit()` 显式接受 `model_path`，`main` 传入 `--model`，处理器只从该路径加载，
  报告记录 `processor_model_path`。新增 4 个 CPU 聚焦回归测试（含「显式路径必须被线程传入
  加载器」的负例与「传入 processor 即跳过加载器」的正例）。

## 合同复核结果（全部通过）

| 合同 | 结果 |
|---|---|
| 恰 2638 条、唯一 `sample_id` | ✅ |
| ID 集合与冻结 full manifest 一致 | ✅ |
| 冻结标签字段逐字段一致 | ✅ |
| `record_sha256` 全部现场重算一致 | ✅ |
| 2638 张图片 SHA256 现场重算一致 | ✅ |
| 2638 条原始宽高现场重算一致 | ✅ |
| `full_patch_tokens == product(image_grid_thw)` 且 `G == full_patch_tokens//4` | ✅ |
| 全部 `G <= 576` | ✅ |
| 处理参数与冻结常量一致 | ✅ |
| 摘要 JSON 与现场重算统计一致 | ✅ |
| 用 `--model` 指定处理器逐样本重算 `image_grid_thw`（**2638/2638**）完全一致 | ✅ |

## 声明

- 复审只打开 processor，未加载模型权重、未使用 GPU、未联网、未训练；
- 复审只读 manifest/summary，不修改任何既有文件；r1 旧审计报告字节保持不变；
- 通过审计不代表任何精度/部署/加速/FPGA 时延主张；full2638 仍是发展集。
