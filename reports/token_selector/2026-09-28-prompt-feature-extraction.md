# Prompt 投影特征抽取 — GPU 门禁结果：BLOCKED（抽取未运行）

> **[已被取代 / 2026-09-28 16:05]** 本报告的**严格独占属主门禁（存在他人 compute 进程即阻塞）已被「实际可运行优先」共享策略取代**：经用户明确授权，在 `--allow-shared-gpu` 下允许与其他用户 CUDA 进程共存，仅要求**空闲显存 ≥ 14336 MiB 且 5 s 平均利用率 < 80%**，模型加载后仍有 ≥ 4096 MiB 余量；不得 kill/暂停/修改他人进程。本报告作为历史记录**原样保留**（仅加此标注），其 BLOCKED 结论不再作为阻塞依据。取代后的实际运行（抽取 472 记录 / 236 次视觉前向）见 `2026-09-28-prompt-projection-shared-gpu-run.{md,json}`。

- 日期：2026-09-28
- 审计窗口：2026-09-28T01:34:13+08:00 – 2026-09-28T01:34:22+08:00
- 决策：`GPU_GATE_BLOCKED / MODEL_LOAD_NOT_ACTIVATED / PROMPT_PROJECTION_EXTRACTION_NOT_RUN`
- `model_load_authorized = false`

## 1. 门禁判据与结论

| 判据 | 阈值 | 观测 | 结果 |
|---|---|---|---|
| G1 无非 zhaoxiangyu（uid 1005）compute 进程 | 无 | 2 个：`seey6`(uid 1001)、`zhaoshengtao`(uid 1004) | **FAIL** |
| G2 空闲显存 | >= 20480 MiB | 42060 MiB（已用 3426 / 总 46068） | PASS |
| G3 非明显满载 | 非满载 | util 97–100%、254–296 W / 300 W、85–86 °C | **FAIL** |

总判：**BLOCKED**。未激活模型加载路径，未运行抽取器。

## 2. 只读证据

### GPU
```
NVIDIA RTX A6000, driver 580.178.04, CUDA 13.0
Memory: 3426MiB / 46068MiB  (free 42060 MiB)
Util 99% (samples 99/99/99/97/99/100), 86C, 292-296W / 300W cap, P2
```

### compute 进程（均非 zhaoxiangyu）
- PID 16507 — owner `seey6` / uid 1001 — 382 MiB — `python main.py --port 7860 --listen 0.0.0.0` — 起于 Fri Sep 25 19:12:53 2026，State S。
- PID 884366 — owner `zhaoshengtao` / uid 1004 — 2950 MiB — `python sensitivity/weight_sensitivity.py --max-images 0 --max-layerwise-images 0 --max-layerwise-layers 0 --resume-layerwise --run-dir-name phase1_weight_sensitivity_kodak24` — 起于 Mon Sep 28 00:48:16 2026，State R，CPU 100%。

display 进程：`/usr/lib/xorg/Xorg` (35 MiB)、`/usr/bin/gnome-shell` (9 MiB)。

原始日志：`reports/token_selector/logs/gpu_audit_20260928T013413.log`

## 3. 抽取状态

- `evaluation/extract_prompt_projection_features.py` sha256 `cac52d49…eb70`
- `configs/prompt_projection_distillation.json` sha256 `0b03f8f5…ef99`
- `datasets/prompt_projection_clean_inductive_v1.jsonl` sha256 `acab1a32…9213`
- 期望 out-dir：`artifacts/token_selector/prompt_projection_features`
- 该目录此前**不存在**，本轮**未创建**；未加载模型、未 import transformers/PIL、未执行任何视觉/提示前向。

因为门禁 BLOCKED，以下核验**全部未执行**：index=472 唯一 sample、npz=472、唯一 `source_image_sha256`=236、`vision_forwards`=236、prompt calls=472、禁止键检查、schema/dtype/shape、source/config/manifest hash 一致性、按图 hash 重放 split（159/38/39 图，318/76/78 记录）。

## 4. 计时与峰值显存

- 抽取 wall time：不适用（未启动）。
- 峰值显存：不适用（审计期观测 3426 MiB 为常驻进程，非抽取峰值）。
- 若运行，wall time 将含模型加载，属非稳态推理；本轮未产生该数字。

## 5. 边界与纪律

未训练、未跑 CV-Bench、未联网、未改模型、未 kill/暂停任何其他进程、未删除任何缓存（无缓存可删）、未宣称任何精度。

## 6. 下一步（未执行）

待出现零非 zhaoxiangyu compute 进程、空闲 >= 20480 MiB、且 GPU 非满载时，重跑本只读门禁；仅门禁 READY 后在独立授权门下激活 pinned 抽取器与模型加载路径。
