# 任务对嵌套负样本有效性审计（2026-10-03）

- 结论：**SYMMETRIC_NEGATIVE_INVALID**（解释性诊断，不改变既有 `OPERATIONAL_PROMPT_UTILITY_NOT_ESTABLISHED`，不作精度/晋级声明）
- SAME_IMAGE_TASK_TARGETS_NESTED：True（阈值 0.8，实测 1.000000）
- grounding B K48 gap mean：-0.001925；pointing B K48 gap mean：+0.287765

## 1. point 是否落在 bbox 内
- 记录级 overall 比例（判定用）：1.000000（76/76）
- 图像级跨任务比例（次要）：1.000000（38/38）
- 同图 grounding/pointing 共享 ground_truth 的图像数：38/38

## 2. bbox 面积分布（归一化）
- min 0.023580 / p25 0.060638 / median 0.138215 / p75 0.206783 / max 0.983556
- mean 0.202331，零面积 0

## 3. B 臂三 seed 中位数 K48 correct−wrong retained-mass gap
- grounding：mean -0.001925，median -0.000409（n=38）
- pointing：mean +0.287765，median +0.144167（n=38）
- overall：mean +0.142920，median +0.101759

## 4. 分层与 Spearman
- point_inside=true：n=38，grounding gap -0.0019252304813808796，pointing gap 0.2877651809180278
- point_inside=false：n=0，grounding gap None，pointing gap None
- 说明：one stratum is empty; point-inside is a constant across the 38 val images, so no between-stratum contrast exists
- bbox 面积 vs grounding gap Spearman：-0.049568（n=38）

## 5. correct/错误 selected-set Jaccard 分层
- grounding：mean 0.246018，median 0.230769
- pointing：mean 0.246018，median 0.230769
- point_inside=true：grounding 0.24601759836970738，pointing 0.24601759836970738

## 6. 与 operational JSON 交叉核对
- 一致（容差 1e-09）：True

## 7. 隔离与不变性
- 仅读取：/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/configs/task_pair_nesting_audit.preregistered.json, /home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/datasets/prompt_projection_clean_inductive_v1.jsonl, /home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/reports/token_selector/2026-10-03-operational-prompt-utility-diagnostic-records.csv, /home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/reports/token_selector/2026-10-03-operational-prompt-utility-diagnostic.json, /home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/reports/token_selector/2026-10-03-task-pair-nesting-audit-preregistration.json
- NPZ 打开：[]；原图/权重/GPU/联网/训练/内部测试：全部 False
- 预注册与输入运行前后逐字节不变：True

## 限制
- Explanatory diagnostic on already-used validation; not a final accuracy conclusion.
- The prior OPERATIONAL_PROMPT_UTILITY_NOT_ESTABLISHED decision is unchanged.
- No NPZ, raw image, model weight, GPU, network or internal test was touched.
- point-inside is constant across the 38 val images, so the stratification cannot separate strata.
- Per-image statistics treat the 38 images as independent units.
