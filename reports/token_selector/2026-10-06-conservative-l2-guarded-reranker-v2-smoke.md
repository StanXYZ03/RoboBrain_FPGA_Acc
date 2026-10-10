# L2 守卫整数重排 v2 六单元 GPU 冒烟合同审计

- 日期：2026-10-06
- 决策：**`CONTRACT_SMOKE_PASS`**
- 下一决策：`PERMITTED_ONLY_SINGLE_FROZEN_FULL2638_DEVELOPMENT_RUN`
- 口径：CPU-only、只读；`gpu_runs=0`、`audited_gpu_runs=1`、`audited_samples=6`；未加载模型权重/未联网/未训练/未装依赖。

## 1. 被审计原始冒烟

- 路径：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/artifacts/token_selector/2026-10-06-conservative-l2-guarded-reranker-v2-full2638/conservative_l2_guarded_rerank_v2_k96_smoke.jsonl`
- 记录数：6
- SHA256（现场重算）：`e7648d9b3d015d434b49aa8b030ba9148d294bdd837f1ce9a68d769083a15916`；与声明一致：`True`

## 2. 样本与单元

- 观测样本 ID（按文件顺序）：['286', '374', '1039', '1392', '1467', '2566']
- `ADE20K::Count` -> `286`
- `ADE20K::Relation` -> `374`
- `COCO::Count` -> `1039`
- `COCO::Relation` -> `1392`
- `Omni3D::Depth` -> `1467`
- `Omni3D::Distance` -> `2566`

## 3. 六项合同检查

| 合同项 | 通过 | 子检查数 |
| --- | --- | --- |
| `1_exactly_six_unique_deterministic_smoke_ids_one_per_cell` | ✅ | 8 |
| `2_record_sha_manifest_config_checkpoint_source_pins_and_leakage` | ✅ | 20 |
| `3_identity_bypass_legality_and_exact_k96_87_real_9_context` | ✅ | 4 |
| `4_frozen_anchors_same_region_one_for_one_counts_nonempty_context` | ✅ | 30 |
| `5_swap_legal_gain_relation_count_metadata_lengths_and_hashes` | ✅ | 89 |
| `6_outputs_parse_and_descriptive_reference_comparison` | ✅ | 2 |

- 全部合同检查通过：`True`；失败项：无

## 4. 布局与交换（逐样本）

| 样本 | bypass | G | K | real | context | swaps | anchors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 286 | True | 64 | 64 | 64 | 0 | 0 | [] |
| 374 | False | 180 | 96 | 87 | 9 | 16 | [15, 29, 35, 66, 72, 79, 135, 143, 164] |
| 1039 | False | 280 | 96 | 87 | 9 | 16 | [28, 74, 81, 94, 142, 148, 201, 210, 251] |
| 1392 | False | 260 | 96 | 87 | 9 | 16 | [26, 69, 77, 125, 130, 137, 175, 195, 233] |
| 1467 | False | 768 | 96 | 87 | 9 | 16 | [36, 127, 208, 263, 414, 491, 576, 671, 751] |
| 2566 | False | 1400 | 96 | 87 | 9 | 16 | [200, 249, 416, 638, 767, 850, 927, 1050, 1199] |

## 5. 描述性逐样本答案对比（不是精度、不是门）

| 样本 | 正确答案 | v2 | dense | 修正整数 L2 | 既往 prompt 融合 |
| --- | --- | --- | --- | --- | --- |
| 286 | B | ❌ D ((D) 4) | ❌ D ((D) 4) | ❌ D ((D) 4) | ❌ D ((D) 4) |
| 374 | A | ✅ A ((A) left) | ✅ A ((A) left) | ✅ A ((A) left) | ❌ B ((B) right) |
| 1039 | E | ❌ D ((D) 1) | ❌ D ((D) 1) | ❌ D ((D) 1) | ❌ D ((D) 1) |
| 1392 | A | ✅ A ((A) left) | ✅ A ((A) left) | ✅ A ((A) left) | ✅ A ((A) left) |
| 1467 | A | ✅ A ((A) books) | ✅ A ((A) books) | ✅ A ((A) books) | ✅ A ((A) books) |
| 2566 | A | ✅ A ((A) truck) | ✅ A ((A) truck) | ✅ A ((A) truck) | ✅ A ((A) truck) |

- 描述性正确计数：v2 4/6；dense 4/6；corrected_l2_integer 4/6；prompt_plus_corrected_l2_integer 3/6
- **4/6 不是精度估计、不是门**；仅用于描述性对比。

## 6. 冻结输入字节一致性

- 非空 config pin 现场复核通过：36；不匹配：无；待生成（null）：['new_arm_jsonl']
- 覆盖 v1/v2 核心、集成、runner、汇总器、测试、config、manifest、dense BF16、修正整数 L2、既往 prompt 融合参考与 3 个部署 checkpoint。

## 7. 时延口径

- 首个样本 `model_inference_seconds` = 3.8937 s，**包含首次执行开销**。
- 六个计时都**不是**稳态时延证据；模型加载时间不计入每样本计时段。
- `selector_cpu_roundtrip_seconds` 是软件选择器往返，**不是 FPGA 时延**，**不是加速主张**。

## 8. 限制

- 六个样本的 4/6（或任一计数）是描述性诊断，不是精度估计，也不是任何门。
- 首个样本 3.8937 s 含首次执行开销；六个计时都不是稳态时延证据。
- 软件 selector roundtrip 不是 FPGA 时延，也不是加速主张。
- 本审计不做论文/外部验证确认；通过仅表示可执行一次冻结的 full2638 发展性运行。
- full2638 是被反复观测的发展集，永不构成未触碰确认集。
- 不得在看到 full2638 后调 v2；v1 永久 PREEXECUTION_DESIGN_STOP。
- 审计只读，未加载模型权重、未联网、未训练、未安装依赖；gpu_runs=0。

## 9. 决策边界

- `CONTRACT_SMOKE_PASS`：仅允许执行一次冻结的 full2638 发展性运行（唯一新臂 `conservative_l2_guarded_rerank_v2`）；不是论文确认、外部验证、可部署性或加速结论。

