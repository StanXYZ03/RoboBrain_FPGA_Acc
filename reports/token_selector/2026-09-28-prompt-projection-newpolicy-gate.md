# 2026-09-28 干净归纳 Prompt 投影特征抽取/训练内部验证（新共享 GPU 策略门禁）→ BLOCKED

> **[已被取代 / 2026-09-28 16:05]** 本报告的 **24 GiB 空闲显存门槛（24576 MiB）已被「实际可运行优先」策略取代**：在用户最新明确授权下，预检门槛降为**空闲显存 ≥ 14336 MiB 且 5 s 平均利用率 < 80%**，仍允许与其他用户 CUDA 进程共存且**不得 kill/暂停/修改他人进程**；模型加载后仅要求仍有 ≥ 4096 MiB 余量。本报告作为历史记录**原样保留**（仅加此标注），其 FAIL 结论不再作为阻塞依据。取代后的实际运行见 `2026-09-28-prompt-projection-shared-gpu-run.{md,json}`。

- 日期/时间：2026-09-28 15:52（本地）
- 执行者：`zhaoxiangyu`（uid 1005）
- 决策：`GPU_GATE_BLOCKED_NEW_POLICY / MODEL_LOAD_NOT_ACTIVATED / PROMPT_PROJECTION_EXTRACTION_NOT_RUN / TRAINING_NOT_RUN`
- 结论：**BLOCKED**。新策略下唯一不满足的门槛是**空闲显存 22317 MiB < 24576 MiB（24 GiB）**；GPU 平均利用率 0% < 50% 达标，**他人 CUDA 上下文本身不再构成阻塞**。
- 边界：未 kill/暂停/修改任何他人进程；未联网/下载；未修改原模型、`inference.py`、`api.py`；未创建抽取输出目录；未加载模型、未抽取、未训练、未评估。

## 1. 资源门禁（新策略，取代旧 G1「存在他人进程即阻塞」）

| 条件 | 阈值 | 观测 | 结果 |
|---|---|---|---|
| 空闲显存 | ≥ 24 GiB（24576 MiB） | **22317 MiB（21.79 GiB）** | **FAIL** |
| 5 s 平均 GPU 利用率 | < 50% | 0%（采样 `0,0,0,0,0`） | PASS |
| 运行期为他人保留 | ≥ 12 GiB | 未进入运行期（模型未加载） | N/A |
| 不得 kill/暂停/修改他人进程 | 强制 | 仅只读 `nvidia-smi`/`ps` | PASS |

- 空闲显存缺口：**2259 MiB**。
- 旧报告 `2026-09-28-prompt-distillation-gpu-gate.{md,json}` 已**原样保留并加注**「G1 已被新策略取代」；其 G1 结论不再作为阻塞依据。
- 运行期审计（模型加载后、抽取结束后）**未执行**（未加载模型）。

## 2. GPU 与进程审计（只读）

| 项 | 观测值 |
|---|---|
| GPU | NVIDIA RTX A6000（索引 0） |
| 驱动 / CUDA | 580.178.04 / 13.0 |
| 显存总量 / 已用 / 空闲 | 46068 / 23169 / **22317 MiB** |
| 利用率 | 0%（5 s × 1 s 采样全 0） |
| 温度 / 功耗 | 32–33 °C / 11.0–16.74 W（上限 300 W） |

compute 进程（`nvidia-smi --query-compute-apps` + `ps` 核对归属）：

| PID | 用户 | uid | 显存 | 命令 | 状态 |
|---|---|---|---:|---|---|
| 16507 | `seey6` | 1001 | **23038 MiB** | `python main.py --port 7860 --listen 0.0.0.0` | `Sl+`，已运行 2 天 20:39:20 |

- 非本用户 compute 进程数 = 1。**按新策略不因该上下文阻塞**；但该进程占用 23038 MiB，导致空闲显存低于 24 GiB。
- 未对该进程做任何 kill/暂停/修改/信号操作。
- 原始日志：`reports/token_selector/logs/gpu_audit_20260928T155208_newpolicy.log`

## 3. 数据完整性（CPU，只读，PASS）

| 项 | 值 |
|---|---|
| 唯一训练源 | `datasets/prompt_projection_clean_inductive_v1.jsonl` |
| manifest SHA256 实测 | `acab1a322a93021138ce87bb048c1e25493ec19a3f56f70dc3a24c0686c79213` |
| 与 config pin 一致 | **是** |
| `--validate-manifest-only` | `status=PASS`，`gpu_used=false`，`model_loaded=false` |
| 记录 / 唯一图 | **472 / 236**（grounding 236 + pointing 236） |
| split（按 `source_image_sha256` 严格同图不跨 split） | train 159 图/318 记录；val 38 图/76 记录；test 39 图/78 记录 |
| 三 split 非空 | 是 |
| CV-Bench 禁入训练 | 零 COCO id 重叠、零正式 sample_id 重叠、零正式 eval image SHA 重叠，全部 PASS |
| 图像存在且 path/size/sha256 冻结一致 | PASS |
| 构造脚本哈希一致 | PASS |

- `validate-manifest-only` 证据：`reports/token_selector/logs/2026-09-28-prompt-projection-validate-manifest-only-newpolicy.json`
- 完整 CPU 校验证据：`reports/token_selector/logs/2026-09-28-prompt-projection-clean-inductive-validate-newpolicy.json`

## 4. 抽取 / 训练 / 控制（全部 NOT_RUN）

因门禁 BLOCKED，以下均未执行：

- **抽取**：未加载 `RoboBrain2.5-4B`，未创建 `artifacts/token_selector/prompt_projection_features`，未写 cache/index，未做任何视觉/提示前向；`actual_vision_forwards`、峰值显存、各阶段计时、`torch.cuda.synchronize` 均无数据。预期同图复用视觉前向 = 236（472 prompt records）。
- **训练**：未训练 128 维 `Wt[128,2560]` / `Wv[128,1024]`；val 选 checkpoint、test 仅一次、train/val/test 的 Pearson/Spearman/NDCG@K、K=48/64/96 top-K recall/Jaccard/saliency mass 全部 NOT_RUN。
- **控制**：L2 基线、shuffled-prompt 均 NOT_RUN。
- 未做任何 CV-Bench 任务精度声明。

## 5. 聚焦测试与全量回归（CPU，PASS）

| 项 | 结果 |
|---|---|
| `test_prompt_projection_clean_inductive` + `test_prompt_projection_distillation` + `test_prompt_projection_feature_reuse` + `test_metrics` | **63 tests OK** |
| `python -m unittest discover -s evaluation/tests` | **Ran 300 tests OK** |

- 全量日志：`reports/token_selector/logs/2026-09-28-prompt-projection-full-regression.log`
- 本轮未改任何源码，故无需 `py_compile` 修复。

## 6. 合规

`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`；无联网/下载；未改原模型/`inference.py`/`api.py`；未访问 ZCU102；未访问封存 `visual_refinement_20260912_test.jsonl`。

## 7. 限制

- 门禁在新策略下仅因**显存不足**阻塞（22317 MiB < 24576 MiB）；若未来空闲显存恢复但 `seey6` 进程仍在，`extract_prompt_projection_features.py` **内置** `gpu_gate.owner_rule = "every GPU compute process must belong to the current user"` 仍会拦截，需在另行授权下对该 gate 做**最小非语义调整**（不改算法/超参数）或明确 bypass；本轮未改。
- 按 image hash 的隔离仅以冻结 lock/manifest 在 CPU 复核，未在抽取运行期核验。
- 无蒸馏指标、无任务精度、无端到端加速声明。

## 8. 精确下一步（未执行）

当且仅当 **空闲显存 ≥ 24576 MiB（24 GiB）且 5 s 平均利用率 < 50%** 时：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/zhaoxiangyu/conda-envs/robobrain25/bin/python \
  evaluation/extract_prompt_projection_features.py \
  --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B \
  --manifest datasets/prompt_projection_clean_inductive_v1.jsonl \
  --config configs/prompt_projection_distillation.json \
  --out-dir artifacts/token_selector/prompt_projection_features
```

随后再训练 `Wt/Wv`（val 选 checkpoint、test 仅一次）并跑 L2 / shuffled 控制与回归。运行前需按新策略对齐 extractor 内置 gate。**不得 kill 他人进程。**

证据：`reports/token_selector/2026-09-28-prompt-projection-newpolicy-gate.{md,json}`、`reports/token_selector/logs/`、`/tmp/robobrain_harness_result.json`、`CURRENT_STATUS.md`。
