# 2026-09-28 干净归纳 Prompt 投影特征抽取 + 训练内部验证（「实际可运行优先」共享 GPU）→ 执行完成（val 未通过 shuffled 判据）

- 日期/时间：2026-09-28（本地）
- 执行者：`zhaoxiangyu`（uid 1005）
- 决策：`PROMPT_PROJECTION_EXTRACTION_DONE / TRAINING_DONE_VAL_SELECTED / VAL_CORRECT_BEATS_SHUFFLED_FALSE / TEST_EVALUATED_ONCE / INTERNAL_SANITY_ONLY_NOT_CV_BENCH_ACCURACY`
- 边界：未 kill/暂停/修改任何他人进程；无联网/下载；未修改原模型、`inference.py`、`api.py`；未访问封存 `visual_refinement_20260912_test.jsonl`、未访问 ZCU102。
- 本次**实际使用了 GPU**：RTX A6000 上完整跑通 236 次视觉前向 + 472 条 prompt 记录抽取与 128 维 `Wt/Wv` 训练。

## 1. 共享 GPU 策略与门禁

按用户最新明确授权「只要 GPU 能实际运行就使用」，预检规则改为：

| 条件 | 阈值 | 观测 | 结果 |
|---|---|---|---|
| 空闲显存 | ≥ 14336 MiB | 22317 MiB | PASS |
| 5 s 平均利用率 | < 80% | 0%（采样 `0,0,0,0,0`） | PASS |
| 运行期他人进程 | 只读审计，不阻断 | `seey6` PID 16507 / 23038 MiB（u） | 容忍，未做任何操作 |
| 模型加载后余量 | ≥ 4096 MiB | 10227 MiB | PASS |

- 实现：extractor 新增 **opt-in** `--allow-shared-gpu`、`--min-free-mib`（默认 14336）、`--max-util-pct`（默认 80）。**不传该 flag 时旧严格属主门禁完全不变**（聚焦测试 `test_legacy_default_still_blocks_non_owner` 通过）。
- 运行期只在**模型加载后**与**抽取结束后**各做一次单次 `nvidia-smi` 快照，不频繁轮询；捕获 OOM/余量不足时释放本进程模型并保存阶段，不影响他人。
- 旧 24 GiB 门禁报告 `2026-09-28-prompt-projection-newpolicy-gate.{md,json}` 与旧独占门禁报告 `2026-09-28-prompt-feature-extraction.{md,json}` **原样保留并加注「已被本策略取代」**。

## 2. 数据完整性（CPU，只读，PASS）

| 项 | 值 |
|---|---|
| 唯一数据源 | `datasets/prompt_projection_clean_inductive_v1.jsonl` |
| manifest SHA256 | `acab1a32…9213` == config pin（PASS） |
| `--validate-manifest-only` | `status=PASS, gpu_used=false, model_loaded=false` |
| 记录 / 唯一图 | **472 / 236**（grounding 236 + pointing 236） |
| split（按 `source_image_sha256`） | train 159 图/318；val 38 图/76；test 39 图/78 |
| 跨 split 同图泄漏 | train∩val=0，train∩test=0，val∩test=0 |
| CV-Bench 训练泄漏 | 无（冻结语料已排除全部 CV-Bench COCO id / eval image hash / sample_id） |

## 3. 抽取（GPU，完成）

| 项 | 值 |
|---|---|
| 模型 | `/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B`（offline / `local_files_only`） |
| index | `artifacts/token_selector/prompt_projection_features/index.jsonl` |
| index SHA256 / 字节 | `02a50868…ee26` / 326962 |
| 记录 / NPZ | 472 / 472（`364,136,400` B） |
| **视觉前向次数** | **236**（同图一次，2 条 prompt 复用） |
| prompt embedding | 472 |
| 输出顺序 | 严格 manifest 顺序 |
| schema | 仅 `group_mean[G,1024]f16`、`prompt_embeddings[T,2560]f16`、教师 3 数组 f32、`meta_*`；无 dense hidden/block0 patch/deepstack |
| index config hash | 全部 == 新 config `44399c8e…28b` |

**GPU 计时（前后 `torch.cuda.synchronize`）**：`91.58 s`（236 图 + 472 prompt）。
**峰值显存**：`torch.cuda.max_memory_allocated = 8552.6 MiB`，`max_memory_reserved = 9248.0 MiB`；加载后余量 10227 MiB。

脚本错误修复（**一次、语义不变**）：真实 GPU 适配层 `_vision_capture/_prompt_embeddings` 原先把 CUDA tensor 直接交给 `np.asarray`，触发 `can't convert cuda:0 device type tensor to numpy`。修复为在**生产适配层** `.detach().cpu()`（数值/dtype 不变），共享 CPU 测试核心与特征定义未改。

## 4. 训练（val 选 checkpoint，test 仅最终一次）

- 模型：`Wt[128,2560]`、`Wv[128,1024]`，config 超参数未改（30 epoch、AdamW、lr 1e-3、seed 20260927、listwise KL）。
- `best_epoch = 0`，`best_val_listwise_kl = 0.025189`；train KL 从 0.025643 降到 0.014819（`loss_decreased=true`）。
- 训练在 GPU 上的非确定性 warning 用 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 复跑，`weights_sha256=ece4a916…cdb3` **完全一致**。
- 导出 `artifacts/token_selector/prompt_projection_WtWv.npz`：`weights_sha256=ece4a916…`，`file_sha256=132a6d9f…`，`load_projection` 复核通过；metadata `trained=true, best_epoch=0, config/index hash` 与本次一致。

### 4.1 指标（内部 sanity，**不是 CV-Bench 任务精度**）

val（选 checkpoint 用）：

| 视图 | Pearson | Spearman | NDCG(full) |
|---|---|---|---|
| student_correct | 0.0731 | 0.0565 | 0.8499 |
| l2_baseline | 0.0251 | 0.0300 | 0.8412 |
| student_shuffled_prompt | 0.0781 | 0.0630 | 0.8510 |
| correct − shuffled | **−0.0051** | **−0.0065** | **−0.0010** |

test（冻结后仅一次）：

| 视图 | Pearson | Spearman | NDCG(full) |
|---|---|---|---|
| student_correct | 0.0673 | 0.0536 | 0.8493 |
| l2_baseline | 0.0343 | 0.0125 | 0.8425 |
| student_shuffled_prompt | 0.0644 | 0.0519 | 0.8488 |
| correct − shuffled | +0.0028 | +0.0017 | +0.0005 |

test student_correct 的 K=48/64/96：

| K | top-K recall | Jaccard | saliency mass | NDCG@K |
|---|---|---|---|---|
| 48 | 0.2257 | 0.1305 | 0.2814 | 0.5319 |
| 64 | 0.2929 | 0.1755 | 0.3537 | 0.5615 |
| 96 | 0.4180 | 0.2716 | 0.4957 | 0.6165 |

- **控制结论**：`student_beats_l2=true`（val 三个主指标均胜 L2），但 **`correct_beats_shuffled=false`**（val 每个主指标 paired delta 均为负）→ 正式 promotion 判据**未通过**，不作任何 promotion/精度声明。
- test 指标仅为冻结后一次性记录，不参与选参。
- CSV：`reports/token_selector/2026-09-28-prompt-projection-training-metrics.csv`；原始 JSON：`reports/token_selector/2026-09-28-prompt-projection-training.json`。

## 5. 聚焦测试与全量回归（CPU，PASS）

| 项 | 结果 |
|---|---|
| clean_inductive + distillation + feature_reuse + metrics | **68 tests OK**（含新增 5 个 `SharedGpuGateTests`） |
| `unittest discover -s evaluation/tests` | **305 tests OK** |

日志：`reports/token_selector/logs/2026-09-28-prompt-projection-{focused-tests,full-regression}-shared-gpu.log`。

## 6. 源码 pin / provenance（旧 hash → 新 hash）

| 文件 | 旧 SHA256 | 新 SHA256 | 说明 |
|---|---|---|---|
| `evaluation/extract_prompt_projection_features.py` | `cac52d49…eb70` | `ccfd0ea8…cf69` | 新增共享门禁 flag + 运行期审计 + 设备适配修复 |
| `configs/prompt_projection_distillation.json` | `0b03f8f5…ef99` | `44399c8e…28b` | 仅文档化 shared_gpu_policy；特征/split/算法/超参未改 |
| `evaluation/train_prompt_projection.py` | `f34a7865…8983` | `f34a7865…8983` | 未改 |
| `evaluation/prompt_projection_clean_inductive.py` | `d80186b6…a4c9` | `d80186b6…a4c9` | 未改 |
| `evaluation/tests/test_prompt_projection_distillation.py` | — | `a2a50f80…fe0a` | 新增共享门禁测试 |

完整 provenance：`reports/token_selector/2026-09-28-prompt-projection-shared-gpu-provenance.json`。

## 7. 限制

- 本结果为**内部归纳 sanity check**，**不得称为 CV-Bench 任务精度**，也不与 Count/Relation/Depth/Distance 可比。
- clean 语料的 prompt 文本分布（grounding/pointing）与 CV-Bench 模板不同。
- **val 最优就是 epoch 0**：后续 epoch 训练 KL 下降但 val KL 上升（过拟合），导出权重仅比确定性初值多训 1 个 epoch，因此在 val 上 prompt-specific 信号很弱，`correct_beats_shuffled` 判据失败。
- 抽取 wall-clock 受与他人共享 GPU 影响，**不构成加速声明**；峰值显存为本进程 `torch.cuda` 统计。

## 8. 精确下一步（未执行）

在 PI 明确授权下二选一：

1. **冻结为负结果**：接受 `best_epoch=0` / val shuffled 判据失败，不再调参，记录终止；或
2. **仅调整优化协议**（如学习率 warmup / early-stopping 指标 / init scale），**不改数据、split、特征与评测契约**，重跑训练后再次如实报告 val gate。

证据：`reports/token_selector/2026-09-28-prompt-projection-shared-gpu-run.{md,json}`、`...-provenance.json`、`...-training.{json,csv}`、`artifacts/token_selector/prompt_projection_features/`、`artifacts/token_selector/prompt_projection_WtWv.npz`、`/tmp/robobrain_harness_result.json`、`CURRENT_STATUS.md`。
