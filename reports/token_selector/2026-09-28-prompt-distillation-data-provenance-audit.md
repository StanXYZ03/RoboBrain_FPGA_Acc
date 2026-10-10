# 2026-09-28 Prompt 投影蒸馏数据来源与污染风险审计（只读，无 GPU/无模型）

## 结论

- 状态：`LEAKAGE_RISK_STATIC_AUDIT_COMPLETE / CONTAMINATION_OBSERVED=false / CONTAMINATION_PROSPECTIVE=true / GPU_NOT_USED`。
- 本轮只读审计：未加载模型、未运行抽取器/训练器、未访问 GPU、未构造新 split、未修改任何算法/配置/数据、未下载数据。
- **核心发现**：当前记录在案的抽取命令把蒸馏特征来源指向正式 64-Token CV-Bench pilot100 manifest（`datasets/cv_bench_screening100_budget64_diagnostic.jsonl`）。配置本身不锁定 manifest。按配置的 `source_image_sha256` 分组 split 重放后，**内部 train/val/test 的图像与问题 100% 落在正式 CV-Bench 评测集内**。因此若按该命令执行，将产生 benchmark contamination（正式评测图像与问题参与训练与选参）。
- 当前仓库**不存在**抽取缓存/索引/已训练权重（`artifacts/token_selector/prompt_projection_features/` 不存在，无 `prompt_projection_WtWv.npz`），所以污染尚未发生，属**将产生**的风险。
- 本地可验证、且与 CV-Bench COCO 子集不相交的独立数据：`visual_refinement_development + calibration` 去掉 52 张 CV-Bench COCO 重叠图后得 **236 张原图 / 472 条记录**（聚合图像集 SHA256 见第 6 节）。

## 1. 数据流（现状）

```
configs/prompt_projection_distillation.json
  ├─ extraction.cache_dir = artifacts/token_selector/prompt_projection_features   (当前不存在)
  ├─ extraction.index     = .../index.jsonl                                       (当前不存在)
  └─ 不锁定 --manifest（来源完全由操作命令决定）

evaluation/extract_prompt_projection_features.py --manifest <M>
  ├─ 教师：full dense 24-block vision tower + merger → merged[G,2560]
  ├─ 学生输入：visual.blocks[0] hook → 每组 4 patch 均值 [G,1024]
  ├─ prompt：仅原始 prompt 文本，add_special_tokens=False → [T,2560]
  ├─ 教师分数：逐组 max_t cosine；minmax → softmax(/T)
  └─ 每样本 npz + JSONL index；写入 source_image_sha256（源图片字节流式 SHA256）

evaluation/train_prompt_projection.py
  ├─ split.group_by = source_image_sha256
  ├─ assign_split = sha256(salt|seed|source_image_sha256) → 累计 0.70/0.15/0.15
  ├─ best epoch 仅按 val listwise KL；test 冻结后评估一次
  └─ 导出 Wt[128,2560]/Wv[128,1024] + weights_sha256

正式评测源（外部）：
  ├─ 预注册 16-Token：datasets/cv_bench_full2638.jsonl / cv_bench_screening100.jsonl
  └─ 64-Token 诊断：datasets/cv_bench_{full2638,screening100}_budget64_diagnostic.jsonl
```

**来源缺口**：`extract_prompt_projection_features.py` 必须传 `--manifest`，而 `configs/prompt_projection_distillation.json` 没有固定该字段；唯一有记录的方案（`reports/token_selector/2026-09-27-prompt-projection-distillation.md` 第 11 节）用的是正式 CV-Bench 评测 manifest。

## 2. 审计方法（只读、可复现）

- 用 `evaluation.prompt_projection_distillation.assign_split` 对 manifest 自身图像字节做确定性重放；图像身份用**文件字节 SHA256**（CV-Bench 内一致）。
- 对 COCO 侧 manifest 用 `coco_image_id` 作身份键（不同分辨率副本不能用 SHA256 对齐）；`coco_image_id` 从已校验的 `datasets/CV-Bench/test_{2d,3d}.parquet` 的 `source_filename` 恢复。
- `datasets/CV-Bench/test_2d.parquet`/`test_3d.parquet` SHA256 与 screening 锁 `dataset_files` 完全一致：`33196034…ecbc` / `ef91fe8b…bbc25`。
- 冻结源码哈希与本轮实测一致（见第 9 节），说明审计对象未被改动。

## 3. Manifest 统计

| manifest | 题数 | 唯一图像(路径/SHA) | 复用图题的图像数 | 任务分布 | 类型/来源 |
|---|---:|---:|---:|---|---|
| `cv_bench_full2638.jsonl` (16-tok) | 2638 | 2638 / 2573 | 65 | Count 788, Relation 650, Depth 600, Distance 600 | 2D 1438 / 3D 1200；ADE20K 633, COCO 805, Omni3D 1200 |
| `cv_bench_screening100.jsonl` (16-tok) | 100 | 100 / 100 | 0 | Count 20, Relation 20, Depth 30, Distance 30 | 2D 40 / 3D 60；ADE20K 20, COCO 20, Omni3D 60 |
| `cv_bench_full2638_budget64_diagnostic.jsonl` | 2638 | 2638 / 2573 | 65 | 同上 | 同上 |
| `cv_bench_screening100_budget64_diagnostic.jsonl` | 100 | 100 / 100 | 0 | 同上 | 同上 |
| `visual_refinement_20260912_development.jsonl` | 384 | 192 | — | grounding 192 / pointing 192 | COCO val2017 原图 |
| `visual_refinement_20260912_calibration.jsonl` | 192 | 96 | — | grounding 96 / pointing 96 | COCO val2017 原图 |
| `visual_refinement_20260912_test.jsonl` | 1024 | 512 | — | grounding 512 / pointing 512 | COCO val2017 原图；`sealed_test=true` |

- `screening100` 的 sample_id 与图像 SHA 均为 `full2638` 的子集（交集=100）；`full2638` 的 16-token 与 64-token 版本 sample_id 完全相同，只是 `max_new_tokens` 16→64。
- `full2638` 有 65 条题共用同一图像字节（2638 题 / 2573 唯一 SHA）。
- 图像目录实测：`cv_bench_full2638_images` 2638 文件 / 404,722,392 B；`cv_bench_screening100_images` 100 文件 / 14,738,161 B；COCO `visual_refinement_original` 800 文件 / 118,854,183 B；`val2017-original` 240 文件 / 34,787,198 B。

## 4. 内部 train/val/test 与正式评测的重叠（静态重放）

对配置 split（salt=`prompt_projection_distillation_2026_09_27`, seed=20260927, 0.70/0.15/0.15）重放：

| 抽取来源 | split | 记录/图像 | 与 full2638 图像重叠 | 与 pilot100 图像重叠 | sample_id 重叠(pilot100) | prompt 重叠(pilot100) |
|---|---|---:|---:|---:|---:|---:|
| documented：`screening100_budget64` + `--limit-per-task 4` | train | 11 / 11 | **11 / 11** | **11 / 11** | 11 | 11 |
| 同上 | val | 2 / 2 | **2 / 2** | **2 / 2** | 2 | 2 |
| 同上 | test | 3 / 3 | **3 / 3** | **3 / 3** | 3 | 3 |
| `screening100_64tok` 全部 100 | train | 69 / 69 | 69 / 69 | 69 / 69 | 69 | 68 |
| `screening100_64tok` 全部 100 | val | 15 / 15 | 15 / 15 | 15 / 15 | 15 | 15 |
| `screening100_64tok` 全部 100 | test | 16 / 16 | 16 / 16 | 16 / 16 | 16 | 16 |
| `screening100_16tok` 全部 100 | 同 64tok | 同 | 同 | 同 | 同 | 同 |
| `full2638_64tok` 全部 | train | 1812 / 1769 | **1769 / 1769** | 69 | 1812 | 80 |
| `full2638_64tok` 全部 | val | 417 / 408 | **408 / 408** | 15 | 417 | 26 |
| `full2638_64tok` 全部 | test | 409 / 396 | **396 / 396** | 16 | 409 | 24 |

documented limit-4 子集具体样本：
- train：`55,123,371,402,426,1482,1501,1506,2050,2115,2142`
- val：`1543,2104`
- test：`21,81,346`

结论：**任何以 CV-Bench manifest 作为抽取来源的方案，其全部内部图像都与正式评测图像重叠**；用 `screening100` 时连外部问题文本也重叠。内部哈希分组只保证「同图不跨内部 split」，**不能**保证与外部评测集不相交。

### COCO 身份层面的交叉重叠（`coco_image_id`）

| 本地 COCO manifest | 与 CV-Bench 805 个 COCO 图重叠 | 与 pilot100 COCO 图重叠 |
|---|---:|---:|
| `visual_refinement_development` (192) | 33 | 1 |
| `visual_refinement_calibration` (96) | 19 | 0 |
| `visual_refinement_test` (512，sealed) | 86 | 2 |
| `dac_cutpoint_oracle_dev` (16) | 3 | 0 |
| `dac_spatial_eval_expanded` (96) | 14 | 1 |
| `dac_spatial_eval_smoke` (32) | 4 | 0 |
| `dac_spatial_l2res_blind` (96) | 24 | 0 |
| `dac_spatial_selector_calibration` (32) | 6 | 1 |
| `dac_spatial_selector_validation` (96) | 10 | 0 |
| `dac_spatial_task_routing_test` (96) | 12 | 0 |
| `q8a_r0875_failure_attribution` (48) | 15 | 0 |
| `q8b_r0875_failure_attribution` (51) | 16 | 0 |
| `coco2017/val2017-original` (240) | 43 | — |

即：本地历史 COCO manifest **都不是** CV-Bench 纯净来源；使用前必须按 `coco_image_id` 扣除 CV-Bench COCO 子集。

## 5. 污染判定

- **判定：`LEAKAGE_RISK`（将产生，尚未产生）。**
- **无标签属性**：抽取器/训练器从不读取 `answer`/`choices`，教师目标来自 dense 模型自身的显著性 soft target。因此不存在传统监督标签泄漏。
- **输入/问题泄漏**：一旦用 CV-Bench 图像与问题抽取，Wt/Wv 会在评测输入上拟合；配置又在 val 上选 best epoch，而 val 本身属于正式评测集 → 这是**输入与问题层面的污染**。
- **严格归纳（strict inductive）**：**违反**。归纳协议要求投影的训练与选参图像/问题与正式 CV-Bench 评测集完全不相交；documented 方案完全重叠。
- **无标签直推（transductive）**：若明确声明为直推协议，复用无标签评测输入在原则上有讨论空间，但必须满足：(a) 任何评测图像不得参与 epoch/超参选择；(b) 结果必须标注 transductive，不能当作主归纳数字；(c) 内部 test 不再是独立泛化估计。当前 config/report 声称的是 `test_used_for_selection=false` 的标准归纳纪律，并未声明直推，**不能默认适用**；需要 PI 明确授权。
- **合规路径**：仅在 CV-Bench 之外的数据上训练并冻结投影，再对正式 CV-Bench 只做一次最终评测。
- 内部 split 的 `test_evaluated_once_after_freeze` 在当前来源下具有误导性：该内部 test 是正式评测集的子集，冻结一次并不构成独立泛化证据。

## 6. 本地可用的无泄漏独立数据（仅列实际存在且可校验者）

### 候选 A：`visual_refinement_development + calibration` 去 CV-Bench COCO 重叠
- 状态：`MATERIALIZED_AS_CLEAN_INDUCTIVE_V1 / STRICT_INDUCTIVE_VALIDATION_PASS`（2026-09-28 更新；原状态 `AVAILABLE_LOCALLY_VERIFIED`）。
- 冻结产物：`datasets/prompt_projection_clean_inductive_v1.jsonl`（SHA256 `acab1a32…9213`，236 图 / 472 记录）与 `datasets/prompt_projection_clean_inductive_v1.lock.json`（`strict_inductive=true`）。构造/验证脚本 `evaluation/prompt_projection_clean_inductive.py`；`configs/prompt_projection_distillation.json` 已 pin 该 manifest，抽取器启动 fail-closed 校验。overlap 门（零 CV-Bench COCO id、零正式 eval image SHA、零正式 sample_id）全部 PASS；报告见 `reports/token_selector/2026-09-28-prompt-distillation-clean-manifest.{md,json}`。抽取/训练仍未运行（GPU 门禁）。
- 清单：`datasets/visual_refinement_20260912_development.jsonl` (SHA256 `647e963e3971d606a0ea95996a2d11f5d3b7ffa8fea1d9169811261572b180ae`)、`datasets/visual_refinement_20260912_calibration.jsonl` (`d7746bf21061dfb70cb779bbf7aa6b006f15e41c3f465501d07ad5c6c8d88247`)。
- 基础：288 张唯一 COCO 原图 / 576 条记录；扣除与 `cv_bench_full2638` COCO 子集重叠的 52 张（dev 33 + cal 19）后：
  - **236 张唯一原图 / 472 条记录**（dev 318 + cal 154），文件全部存在。
  - 图像目录 `datasets/coco2017/visual_refinement_original`，236 文件合计 **35,054,288 B**。
  - 聚合图像集 SHA256：`ca8f8a6ac13c98d661d989c531b3427ce29438a01d9f0e4434eb097501a0bb6b`（对 236 行 `"<sha256>  <filename>"` 排序后拼接再 SHA256）。
  - 逐图 sha256/字节数清单见报告 JSON 的 `local_clean_candidates.candidates[0].per_image_sha256`。
- 已交叉核对不相交：CV-Bench full2638 COCO 子集（805 个 id）、历史 dac COCO manifest 并集、`visual_refinement_test`(sealed)。
- 边界：grounding/pointing 的 prompt 分布与 CV-Bench 的 Count/Relation/Depth/Distance 不同；内部 test 只能当内部 sanity check，不能替代 CV-Bench 结论。

### 候选 B：`coco2017/val2017-original`
- 状态：`IMAGES_VERIFIED_NO_EXISTING_PROMPT_MANIFEST`。
- 240 张原图，其中 197 张与 CV-Bench COCO 子集不相交；但这 240 张都已被历史 dac manifest 使用，且**没有现成 image+prompt manifest** 与之配对。按任务约束不得现场发明 split/manifest，故当前**不可直接使用**。

### 明确不可用
- `datasets/cv_bench_*`：正式评测源。
- `datasets/dac_*`、`q8*a_failure_attribution*`：COCO 图与 CV-Bench COCO 子集有 3–24 张重叠，且已参与过 visual-refinement selector 开发。
- `visual_refinement_20260912_test.jsonl`：`sealed_test=true`，禁止用于训练/选参。

## 7. 下一步最小无泄漏数据方案候选（仅候选，本轮不执行）

1. 用候选 A 的 236 张原图 / 472 条记录构造一个**新的** CV-Bench 独立抽取 manifest —— **已于 2026-09-28 构造并冻结**（`datasets/prompt_projection_clean_inductive_v1.jsonl` + `.lock.json`，`strict_inductive=true`，overlap 门 PASS）。
2. GPU 审计通过后，仅以该 manifest 运行 `extract_prompt_projection_features.py`。
3. 运行 `train_prompt_projection.py`；内部 test 仅作 sanity check，不写成 CV-Bench 泛化。
4. 冻结 Wt/Wv 后，对正式 CV-Bench 只做一次最终评测。
5. 若 PI 选择 transductive 协议，则必须独立授权、禁止用评测图像选 epoch/超参，并在结果中明确标注 transductive。

## 8. 告警与边界

- 配置不锁定抽取 manifest 是**根因治理缺口**；provenance 取决于操作者命令。
- 内部图像哈希分组**无法**检测外部评测集重叠，只能用于内部两两不交。
- COCO 身份必须用 `coco_image_id`，不能跨分辨率用 SHA256。
- `CURRENT_STATUS.md` 较早条目有「按 sample_id 哈希 split」的陈旧表述；冻结实现是按 `source_image_sha256`。
- 本轮没有任何真实特征/权重产生；所有 split 均为**静态重放**，未伪造任何真实 cache/manifest。

## 9. 证据与哈希

- 冻结源码哈希（本轮实测，与 `artifacts/token_selector/2026-09-27-prompt-projection-distillation-spec.json` 一致）：
  - `evaluation/prompt_projection_distillation.py` = `21c6bec09af608a13789a40822095052bbd35b0006b1b490715055e98cf6196f`
  - `evaluation/extract_prompt_projection_features.py` = `aeaf2049bfab91e819d7d31fb771034f368746de2c8684bd37d00e1eff21781c`
  - `evaluation/train_prompt_projection.py` = `f34a7865f815299f812c9e4d95d311a509ee9d7394f38a2cf9b07de0a3648983`
  - `configs/prompt_projection_distillation.json` = `6c46c28216378f22a91d6e3e31da045dce8144f0bd13bc5ade15f617eb3c7bca`
- Manifest 哈希：full2638 `22399a767f4c…ece3d91`；screening100 `31bf313286c8…768b50c`；full2638_64 `a82813547444…221f8a6`；screening100_64 `236e38a5f88c…ec2e668`。
- 机器可读报告：`reports/token_selector/2026-09-28-prompt-distillation-data-provenance-audit.json`。
- 关键命令证据：`reports/token_selector/2026-09-27-prompt-projection-distillation.md` 第 11 节。
- 只读工具：`evaluation.prompt_projection_distillation.assign_split`；`datasets/CV-Bench/test_{2d,3d}.parquet` 的 `source_filename` 恢复 COCO 身份。

## 10. 合规声明

- 未加载模型、未使用 GPU、未运行抽取/训练/推理、未下载数据、未修改算法/配置/数据。
- 未访问封存数据（`visual_refinement_test` 仅做重叠统计，未用于训练）、未访问 ZCU102。
- 未 kill/暂停/修改任何进程。
