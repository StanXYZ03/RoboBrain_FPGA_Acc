# edge576 CV-Bench r3 作废记录（2026-10-07）

- 状态：`SUPERSEDED_AFTER_FIRST_GPU_SMOKE_FAILURE`
- 对象：`configs/2026-10-07-edge576-cvbench-three-arm-development-r3.preregistered.json`（SHA `e7ff56c9…3eed`）
- 失败证据：`reports/token_selector/2026-10-07-edge576-cvbench-r3-first-gpu-attempt-failure.{json,md}`
- 取代者：`configs/2026-10-07-edge576-cvbench-three-arm-development-r4.preregistered.json`
- 本轮撰写：`gpu_runs=0`、`weights_loaded=false`、未联网、未训练、未装依赖

## 为什么作废 r3

r3 的归一化契约**强制要求每条记录都带 `output_token_ids`**，但冻结 dense 生产者
`evaluation/run_hf_shallow_block_pruning.py::run_one` **只返回** `output_text`、真实整数计数
`output_tokens` 和 `selected_group_indices`，**从不返回 token id**；只有冻结 L2/v2 生产者返回
`output_token_ids`。

r3 的第一次真实模型执行（`dense_edge576`、`sample_id=1838`、`cuda:0`）在完成模型加载、一次 warmup
与一次样本推理后，于写文件前 fail-closed：

```
Edge576ContractError: frozen dense_edge576 record is missing 'output_token_ids' (fail closed)
```

**未写任何文件、未触碰任何 full2638 输出**，因此作废 r3 不损失任何结果。

## r3 生命周期（诚实记录）

| 项目 | 值 |
| --- | --- |
| `executed` | true（发生过一次真实 GPU smoke 尝试） |
| `gpu_runs` | 1 |
| 完成臂数 | 0 |
| smoke 文件数 | 0 |
| 写出记录数 | 0 |
| full2638 输出被触碰 | false |
| `never_executed` | false |
| `must_not_be_run` | true（修复并由 r4 重新钉住前不得再运行） |
| 字节保留 | true |

## r3 执行层 pin 漂移（被 r4 重新钉住）

| 文件 | r3 SHA256 |
| --- | --- |
| `evaluation/edge576_cvbench_runner_contract.py` | `fdea7df6…90db` |
| `evaluation/run_edge576_cvbench_three_arm.py` | `b1fb2ca1…eb85` |
| `evaluation/tests/test_run_edge576_cvbench_three_arm.py` | `6d4f5135…fa55` |
| `evaluation/run_edge576_cvbench_three_arm_smoke.py` | `3c956d7e…c255` |
| `evaluation/tests/test_run_edge576_cvbench_three_arm_smoke.py` | `458a4510…d1068` |

r3 的 config/报告/生成器/测试字节**全部保留**，不得重新生成；r4 在 `supersedes` 中记录这些旧 SHA。

## r3 中保持不变、由 r4 继承的内容

- 三臂：`dense_edge576` / `corrected_l2_integer_b1_k96` / `v2_edge576_b1_k96`（第三臂仍为既有
  `conservative_l2_guarded_rerank_v2`，`is_new_mechanism=false`）；
- 主指标官方 `cv_bench_accuracy`、成功门、`task::source_dataset` 分层配对 bootstrap
  （seed 20261007、5000 次、95%）、K96/87+9 布局、无泄漏门；
- 三条 full2638 输出路径不变；
- 单样本 smoke `sample_id=1838`；
- 执行顺序：模型身份 → **三臂 smoke 全通过** → Dense full2638 → 修正整数 L2 full2638 →
  v2 full2638 → 官方评分与分层配对 bootstrap。

## r4 的修复要点

- `output_token_ids` 改为可选，只有冻结生产者真实返回时才记录，**绝不伪造**；
- 新增必备可审计 `output_token_count`，由真实字段 `output_token_ids` 长度和/或 `output_tokens`
  规范映射（两者都在则必须一致）；
- 新增 `output_token_ids_available` 标注 token id 是否真实存在；
- 核心字段缺失（`output_text`、`selected_group_indices`）、token 字段全缺失或互相矛盾仍 fail-closed；
- smoke 入口与聚焦测试覆盖 Dense 缺失 / L2 与 v2 存在 / 字段映射 / 核心字段缺失四种行为。
