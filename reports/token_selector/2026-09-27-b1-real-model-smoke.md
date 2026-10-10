# 2026-09-27 B1 真实模型 Smoke Test（dense / identity / L2 K=64）—— BLOCKED

## 结论

- 状态：`BLOCKED_GPU_AUDIT_NON_OWNER_COMPUTE_PROCESS`
- 判定阶段：模型加载前的 GPU 前置审计。
- 原因：GPU 上存在**非 `zhaoxiangyu`** 的计算进程，触发任务规定的 BLOCKED 条件。按规则**未加载任何模型**、**未选取样本**、**未保存 smoke manifest**、**未执行任何 CUDA 推理**。

## GPU 前置审计（只读）

时间（UTC）：2026-09-27T07:41:21Z；命令：`nvidia-smi`、`nvidia-smi --query-compute-apps=...`、`ps -o user,pid,cmd`。

| 项目 | 值 |
| --- | --- |
| GPU 型号 | NVIDIA RTX A6000 |
| 驱动 / CUDA | 580.178.04 / 13.0 |
| 总显存 | 46068 MiB |
| 已用显存 | 4886 MiB |
| 空闲显存 | 40600 MiB（≈39.65 GiB） |
| 功耗 | 298.95 W / 上限 300.00 W（瞬时 298.44 W） |
| 利用率 / 温度 | 100% / 86 ℃ |

### GPU 计算进程归属（`ps -o user,pid,cmd` 核对）

| PID | 用户 | 命令 | 显存 | 归属 zhaoxiangyu |
| --- | --- | --- | --- | --- |
| 16507 | `seey6` | `python main.py --port 7860 --listen 0.0.0.0` | 382 MiB | 否 |
| 562557 | `ismy` | `/home/ismy/for_dac/.venv-woq/bin/python -m woq.run --model facebook/opt-1.3b --method awq --seed 0 --tag main` | 4410 MiB | 否 |

两者均经 `ps -o user,pid,cmd` 与 `/proc/<pid>` 属主双重核对，确认为非本用户计算进程。

## 门限判定

| 门限 | 阈值 | 实测 | 通过 |
| --- | --- | --- | --- |
| 空闲显存 | ≥ 20 GiB | 39.65 GiB | 是 |
| 计算进程归属 | 仅 `zhaoxiangyu` | 存在 `seey6`、`ismy` | **否** |

→ 因“存在非 `zhaoxiangyu` 的计算进程”，按任务规则进入 BLOCKED 分支。

## 未执行项（因 BLOCKED 中止）

- 未激活 `robobrain25` 环境，未设置 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`（未进入执行阶段）。
- A) 官方 dense `model.generate(do_sample=False)`：未执行。
- B) 新 B1 runner identity（budget 取 ≥ 该样本 G 的最小 16 倍数）：未执行。
- C) 新 B1 L2 K=64：未执行。
- 未做 dense 与 identity 的逐 token ids 一致性检查。
- 未做 K64 的 64 视觉 group / 64 placeholder / valid / group index 升序唯一 / 分阶段计时求和误差检查。

## 合规声明

- 未 kill、未暂停、未修改任何进程。
- 未修改源代码、未修改原始模型、未删除文件。
- 未访问封存（blind/validation）数据，未访问 ZCU102。
- 未用历史结果冒充本次结果。

## 证据文件

- `reports/token_selector/smoke/2026-09-27-b1-real-model-smoke-gpu-audit.json`
- `reports/token_selector/smoke/2026-09-27-b1-real-model-smoke-status.jsonl`
- `reports/token_selector/2026-09-27-b1-real-model-smoke.json`
- `/tmp/robobrain_harness_result.json`

## 备选样本（仅供解除阻断后使用，本次未选取）

- 预期 manifest：`datasets/cv_bench_screening100_budget64_diagnostic.jsonl`（SHA256 `236e38a5f88ced36a8b77844f84c5d0c1c1b1401944ae8fd76632f351ec2e668`，即 CURRENT_STATUS 记录的锁定 64-Token 100 题 manifest）。
- 该 manifest 第一条 `sample_id`：`21`（`task=Count`、`max_new_tokens=64`）。
- 本次**未**生成 smoke manifest。

## 恢复条件

当 GPU 上不再存在非 `zhaoxiangyu` 计算进程且空闲显存 ≥ 20 GiB 时，重跑 `nvidia-smi` 与 `ps -o user,pid,cmd` 审计通过后，再执行 dense / identity / L2 K=64 的同一加载 smoke test。
