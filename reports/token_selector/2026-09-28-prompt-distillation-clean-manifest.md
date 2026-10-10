# 2026-09-28 Prompt 投影蒸馏严格归纳数据治理：干净语料冻结与验证（无 GPU/无模型）

## 结论

- 状态：`CLEAN_INDUCTIVE_MANIFEST_FROZEN / VALIDATION_PASS / GPU_NOT_USED`。
- 按已审计候选 A 构造 `datasets/prompt_projection_clean_inductive_v1.jsonl`：仅合并 `visual_refinement_20260912_development.jsonl` 与 `visual_refinement_20260912_calibration.jsonl`，按从冻结 CV-Bench parquet 恢复的 **805** 个 COCO image id 删除全部重叠，得到 **236 唯一图 / 472 记录**（dev 318 + cal 154，grounding 236 + pointing 236）。
- 生成 `datasets/prompt_projection_clean_inductive_v1.lock.json`，冻结源 manifest、CV-Bench parquet、四个正式 CV-Bench manifest、排除 52 id、保留 236 id、每图 path/size/sha256、记录/任务数、最终 manifest hash、构造脚本 hash、`strict_inductive=true`。
- 修复治理缺口：`configs/prompt_projection_distillation.json` 显式 pin 抽取 manifest 相对路径与 sha256；`extract_prompt_projection_features.py` 启动时 fail-closed 校验 CLI manifest 解析路径与 sha256 等于配置 pin，并提供仅 CPU 的 `--validate-manifest-only`。
- 本轮未加载模型、未使用 GPU、未运行抽取/训练/推理、未联网、未访问封存数据 `visual_refinement_20260912_test.jsonl`、未访问 ZCU102。

## 1. 干净语料统计

| 项 | 值 |
|---|---:|
| 唯一图像 | 236 |
| 记录数 | 472 |
| 任务 | grounding 236 / pointing 236 |
| 源记录 | development 384→318；calibration 192→154 |
| 按 COCO id 排除 | 52（dev 33 + cal 19） |
| 图像总字节 | 35,054,288 |
| 聚合图像集 SHA256 | `ca8f8a6ac13c98d661d989c531b3427ce29438a01d9f0e4434eb097501a0bb6b` |
| 最终 manifest SHA256 | `acab1a322a93021138ce87bb048c1e25493ec19a3f56f70dc3a24c0686c79213` |
| 最终 manifest 字节 | 639,668 |
| lock SHA256 | `a723458da9b9e170c9ffcde3bfa8844ae18c3e84fe1ba15bbd3d4b3f57e2e105` |

- `sample_id` 新命名且稳定：`ppclean-v1-<coco_image_id:012d>-<coco_annotation_id>-<task>`；原记录 id 保存在 `source_sample_id`，并记录 `source_manifest`、`source_record_index`。
- 内部确定性 split（salt/seed/比例来自 config，按 `source_image_sha256` 分组）：train 159 图 / 318 记录；val 38 图 / 76 记录；test 39 图 / 78 记录，三者非空。

## 2. 来源与冻结哈希

源 manifest（`datasets/`）：

| 源 | SHA256 | 读入 | 保留 | 排除 |
|---|---|---:|---:|---:|
| `visual_refinement_20260912_development.jsonl` | `647e963e3971d606a0ea95996a2d11f5d3b7ffa8fea1d9169811261572b180ae` | 384 | 318 | 66 |
| `visual_refinement_20260912_calibration.jsonl` | `d7746bf21061dfb70cb779bbf7aa6b006f15e41c3f465501d07ad5c6c8d88247` | 192 | 154 | 38 |

CV-Bench 冻结 parquet（COCO id 恢复来源，与 screening 锁 `dataset_files` 一致）：

| parquet | SHA256 |
|---|---|
| `datasets/CV-Bench/test_2d.parquet` | `33196034ef4bf3265cae4a7ff5c4071b2ff1cc21123e8e285c6a91393897ecbc` |
| `datasets/CV-Bench/test_3d.parquet` | `ef91fe8b5392eb2a16e318ca68fa02449d45ba1e152afece12a0a526e9fbbc25` |

四个正式 CV-Bench manifest（仅用于 overlap 门与哈希冻结，不作为训练来源）：

| id | 路径 | SHA256 |
|---|---|---|
| full2638_16tok | `datasets/cv_bench_full2638.jsonl` | `22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91` |
| screening100_16tok | `datasets/cv_bench_screening100.jsonl` | `31bf313286c81fa50911ca3822118184ac03ca74df2cf2c260250cab2768b50c` |
| full2638_64tok | `datasets/cv_bench_full2638_budget64_diagnostic.jsonl` | `a82813547444a5ab89dd7c9d3a0d1483e01a937fcf1d90f7c06e7faf0221f8a6` |
| screening100_64tok | `datasets/cv_bench_screening100_budget64_diagnostic.jsonl` | `236e38a5f88ced36a8b77844f84c5d0c1c1b1401944ae8fd76632f351ec2e668` |

## 3. 排除与保留 id

- 排除（52）：见 lock `cvbench_coco.excluded_image_ids`，digest `2bea4ed0f38841f779ba8e9d8b76a1bfa4f154a1e0deba14176f8c3f068e5d79`。
- 保留（236）：见 lock `cvbench_coco.retained_image_ids`，digest `3af2dd92741979da9a7592876feb6cc80b74dca05c2ce834d61f6f3971adbb90`。
- 805 个 CV-Bench COCO id：digest `821ea2b3a959232a24aee381bec04e01846c06319a741fafdd32bfa5b22bcfe4`。
- 每图 `path/size/sha256` 共 236 条（`lock.images`），实际字节与哈希在验证时逐图重算比对。

## 4. Overlap 门（全部 PASS）

| 门 | 结果 |
|---|---|
| 零 CV-Bench COCO id 重叠 | PASS（保留 236 与 805 交集 0） |
| 零正式 eval image SHA 重叠 | PASS（保留 236 与正式 2573 唯一图 SHA 交集 0） |
| 零正式 `sample_id` 重叠 | PASS（472 个新 id 与 2638 个正式 id 交集 0） |
| 图像全部存在且 path/size/sha256 一致 | PASS |
| 同图所有 prompt 只能同 split | PASS（按 `source_image_sha256` 分组，三 split 非空） |
| train/val/test 均非空 | PASS |
| 源/parquet/正式 manifest/构造脚本 哈希一致 | PASS |
| config pin 与 manifest 一致 | PASS |

- **不声称 prompt 文本分布与 CV-Bench 相同**：本语料是 COCO grounding/pointing prompt，与 CV-Bench Count/Relation/Depth/Distance 的任务分布不同；内部 test 仅作 sanity check，不替代 CV-Bench 结论。该声明写入 lock 的 `warnings` 与 `prompt_distribution_matches_cvbench=false`。

## 5. 治理缺口修复（fail-closed）

- `configs/prompt_projection_distillation.json` → `extraction.manifest = {path, sha256, strict_inductive}`，pin 到 `datasets/prompt_projection_clean_inductive_v1.jsonl` / `acab1a32…9213`。
- `extract_prompt_projection_features.py` 启动顺序：加载 config → 校验 pin（解析路径 + sha256）→ `--validate-manifest-only` 直接退出 → GPU 审计 → 模型加载。pin 不通过返回 3 并打印 JSON，绝不进入 GPU 审计或模型路径。
- `--validate-manifest-only` 仅 CPU：不调用 `gpu_preflight_audit`、不 import transformers、不创建模型、不做任何 CUDA 调用，输出 `gpu_used=false`、`model_loaded=false`。
- 实测：对旧文档化的污染来源 `datasets/cv_bench_screening100_budget64_diagnostic.jsonl` 运行 validate-only，`status=FAIL`、`path_matches=false`、退出码 1；对干净 manifest，`status=PASS`、退出码 0。

## 6. 新增/修改文件

| 文件 | 作用 | SHA256 |
|---|---|---|
| `evaluation/prompt_projection_clean_inductive.py` | 可复现构造/验证脚本（build/validate） | `d80186b664823f2abf6799b4037d5ad3382b375efdf5a5e5f30efe237356a4c9` |
| `evaluation/tests/test_prompt_projection_clean_inductive.py` | 干净语料与 pin 单元测试（11 项） | `5f20a4be3ddc5f421b1cf916e66e51e41a890e448a76d7299b9dcb137a08e162` |
| `evaluation/prompt_projection_distillation.py` | 新增 `pinned_manifest_spec` / `validate_pinned_manifest` | `93a607e41c8ba03543e9b1bd1f228466ad8e0925da8781a6aa659e73e28cc727` |
| `evaluation/extract_prompt_projection_features.py` | 启动 fail-closed pin 校验 + `--validate-manifest-only` | `18e0e48f4d6e87bef9ca6a3871052a1668db2b9c4902cee122d3c382ecce3fb5` |
| `evaluation/tests/test_prompt_projection_distillation.py` | 新增 pin/validate-only 测试（共 37 项） | `1b96daea36173abb3de6df2c69ded17f235f1847657e06727a1bf2392893bca0` |
| `configs/prompt_projection_distillation.json` | 显式 pin manifest | `0b03f8f56422f5efdc34561b0cbce40b98510ff0ede88942854849307108ef99` |
| `datasets/prompt_projection_clean_inductive_v1.jsonl` | 干净归纳语料 | `acab1a322a93021138ce87bb048c1e25493ec19a3f56f70dc3a24c0686c79213` |
| `datasets/prompt_projection_clean_inductive_v1.lock.json` | 冻结 lock | `a723458da9b9e170c9ffcde3bfa8844ae18c3e84fe1ba15bbd3d4b3f57e2e105` |
| `reports/token_selector/2026-09-28-prompt-distillation-clean-manifest.json` | 机器可读报告 | `7b707da26b0660213351b7b0c21ae92c64466d6959b0c9d746380197a56c94e0` |

## 7. 验证

- `py_compile`：5 个 Python 文件通过。
- `python -m unittest evaluation.tests.test_prompt_projection_clean_inductive evaluation.tests.test_prompt_projection_distillation evaluation.tests.test_hardware_regular_token_selector evaluation.tests.test_metrics` → **76/76 OK**。
- `python -m unittest evaluation.tests.test_hardware_regular_b1_integration` → **19/19 OK**。
- `validate --json-out …`：`status=PASS`，failures 为空。
- `extract_prompt_projection_features.py … --validate-manifest-only`：`status=PASS`、`gpu_used=false`、`model_loaded=false`。

精确命令：

```bash
PY=/home/zhaoxiangyu/conda-envs/robobrain25/bin/python
$PY evaluation/prompt_projection_clean_inductive.py build
$PY evaluation/prompt_projection_clean_inductive.py validate --json-out /tmp/ppclean_validate_final.json
$PY evaluation/extract_prompt_projection_features.py \
  --manifest datasets/prompt_projection_clean_inductive_v1.jsonl \
  --config configs/prompt_projection_distillation.json \
  --validate-manifest-only
```

## 8. 告警与边界

- 本轮只冻结并验证**输入语料**，未抽取特征、未训练 `Wt/Wv`、未做任何 CV-Bench 评测；不得据此宣称任何精度、加速或可学性。
- grounding/pointing prompt 分布与 CV-Bench 不同；内部 test 只作 sanity check。
- COCO 身份必须用 `coco_image_id`；不同分辨率副本不能用 SHA256 对齐。
- 旧文档化命令 `cv_bench_screening100_budget64_diagnostic.jsonl` 现被 fail-closed 拒绝；污染仍属**将产生**而非已发生（无 cache/index/WtWv）。
- 未访问 `visual_refinement_20260912_test.jsonl`（sealed）、未访问 ZCU102。

## 9. 下一步门

`GPU_GATED_CLEAN_EXTRACTION`：GPU 属主/空闲显存门禁通过后，仅以 `datasets/prompt_projection_clean_inductive_v1.jsonl` 运行抽取器，再训练并在 best epoch 冻结后对正式 CV-Bench 只做一次最终评测。
