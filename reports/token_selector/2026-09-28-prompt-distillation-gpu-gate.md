# 干净归纳 Prompt 投影蒸馏 GPU 门禁（只读审计）→ BLOCKED

> **[已废弃策略标注 / 2026-09-28 15:52]** 本报告的门槛 **G1「不得存在其他用户计算进程」已被新策略取代**。新策略（见 `2026-09-28-prompt-projection-newpolicy-gate.{md,json}`）允许与其他用户低占用 CUDA 上下文共存，仅在**空闲显存 < 24 GiB 或平均利用率 ≥ 50%** 时判 BLOCKED；不得 kill/暂停/修改他人进程。本报告作为历史记录**原样保留**（仅加此标注），其 G1 结论不再作为阻塞依据。新策略下当次（14:47）本可通过显存/利用率门槛，但当时未授权共存；2026-09-28 15:52 复核时因他人进程占用 23038 MiB、空闲仅 22317 MiB < 24576 MiB 而**再次 BLOCKED**（原因变为显存不足，而非进程归属）。

- 日期：2026-09-28 14:47（本地）
- 执行者：`zhaoxiangyu`（uid 1005）
- 任务：执行既定 Prompt 投影蒸馏 GPU 门禁；仅在门禁通过时才允许抽取/训练。
- 结论：**BLOCKED**（阻塞门槛 G1：存在其他用户计算进程）。`model_load_authorized=false`。
- 决策：`GPU_GATE_BLOCKED / MODEL_LOAD_NOT_ACTIVATED / PROMPT_PROJECTION_EXTRACTION_NOT_RUN`。

## 1. GPU 审计（`nvidia-smi` + `ps` + `/proc`）

| 项 | 观测值 |
|---|---|
| GPU | NVIDIA RTX A6000（索引 0） |
| 驱动 / CUDA | 580.178.04 / 13.0 |
| 显存总量 | 46068 MiB |
| 显存已用 / 空闲 | 471 MiB / **45015 MiB** |
| GPU 利用率 | **0%**（12 s 采样 avg 0 / max 0） |
| 显存利用率 | 1% |
| 功耗 | 9.29–14.54 W / 300 W |
| 温度 | 27 °C |

12 s、1 s 间隔采样（`util%`）：`0,0,0,0,0,0,0,0,0,0,0,0`；功耗 9.29–14.54 W；温度恒 27 °C；显存已用恒 471 MiB。**非持续高利用率。**

### compute 进程清单（`nvidia-smi --query-compute-apps`）

| PID | 用户 | uid | 显存 | 命令 |
|---|---|---|---|---|
| 16507 | **seey6** | 1001 | 382 MiB | `python main.py --port 7860 --listen 0.0.0.0` |
| — | — | — | — | （仅此 1 个 compute 进程） |

`/proc/16507`：属主 `seey6`（uid 1001），`stat=Sl+`，`ELAPSED=2-19:34:44`，`RSS=72428 KB`，`%CPU=0.4`。该进程**非**执行者 `zhaoxiangyu`（uid 1005）。

## 2. 硬门槛判定

| 门槛 | 规则 | 结果 | 证据 |
|---|---|---|---|
| **G1** | 不得存在其他用户计算进程 | **FAIL** | PID 16507 owner=seey6 uid=1001 ≠ 1005 |
| G2 | 空闲显存 ≥ 20480 MiB | PASS | 45015 MiB |
| G3 | GPU 不能持续高利用率 | PASS | 12 采样 util 全 0% |

**总判：BLOCKED**（阻塞门槛 G1）。

> 报告后复核（14:49）：又出现第二个非本用户 compute 进程 PID **1215914**（`chenbowu`/uid 1003，2004 MiB，`.../cbw/bin/python -u fault_research_framework_20260924/clip_study/token_transfer.py stage_b ...`）。其他用户计算进程数 = 2，**判定仍为 BLOCKED 且进一步确认**。因门禁未通过：

- 未加载模型、未激活 `robobrain25` 环境、未 `import transformers/PIL`；
- **未创建空输出目录** `artifacts/token_selector/prompt_projection_features`（确认该目录不存在）；
- 未做任何视觉/提示前向、未抽取、未训练、未评估 CV-Bench；
- 未 kill/暂停/修改任何进程；未联网/下载；未改原模型、`inference.py`、`api.py`。

## 3. 冻结输入与数据源校验（静态，只读）

- 唯一数据源：`datasets/prompt_projection_clean_inductive_v1.jsonl`。
- manifest SHA256 观测 = `acab1a322a93021138ce87bb048c1e25493ec19a3f56f70dc3a24c0686c79213`，与 pin 一致 **PASS**。
- 冻结 pin：config `0b03f8f5…ef99`、extractor `cac52d49…eb70`、trainer `f34a7865…f815`、manifest_builder `d80186b6…a4c9`。
- split 合同：按 `source_image_sha256` 严格同图不跨 split（预期 159/38/39 图、318/76/78 记录），CV-Bench 禁止进入训练。
- 注：因门禁 BLOCKED，按 image hash 的运行期 split 隔离**未核验（NOT_RUN）**。

## 4. 环境合规

`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 保持；无联网；未触碰封存盲测/验证 manifest；ZCU102 未访问。

## 5. 限制

- 本报告只记录门禁审计，不含任何蒸馏指标、不称任务精度、不称端到端加速。
- 无缓存/权重产物；无 index、无 NPZ。

## 6. 精确下一步（未执行）

当且仅当 **GPU 上无非 `zhaoxiangyu` 计算进程 + 空闲显存 ≥ 20480 MiB + 利用率非持续满载** 时，重跑本只读门禁；仅在门禁 READY 且独立授权门开启后，才运行：

```
conda activate /home/zhaoxiangyu/conda-envs/robobrain25
python evaluation/extract_prompt_projection_features.py \
  --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B \
  --manifest datasets/prompt_projection_clean_inductive_v1.jsonl \
  --config configs/prompt_projection_distillation.json \
  --out-dir artifacts/token_selector/prompt_projection_features
```

预期同图复用视觉 forward = 236（472 prompt records）。

证据：`reports/token_selector/2026-09-28-prompt-distillation-gpu-gate.{md,json}`、`/tmp/robobrain_harness_result.json`、`CURRENT_STATUS.md`。
