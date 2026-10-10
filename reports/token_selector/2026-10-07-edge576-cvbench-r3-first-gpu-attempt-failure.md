# edge576 CV-Bench r3 首次 GPU smoke 尝试失败证据（Dense / sample 1838）

- 状态：`FIRST_GPU_SMOKE_ATTEMPT_FAIL_CLOSED`（fail-closed，未写出任何记录）
- 日期：2026-10-07
- 对象：r3 预注册的一次真实模型执行——`dense_edge576`，`sample_id=1838`（G=576、grid=[1,36,64]），`cuda:0`
- r3 config：`configs/2026-10-07-edge576-cvbench-three-arm-development-r3.preregistered.json`（SHA `e7ff56c9…3eed`）
- 本轮证据任务：`gpu_runs=0`、`weights_loaded=false`、未联网、未改任何冻结历史

## 发生了什么

该次尝试已完成：

1. 17 项依赖 pin 与机制身份校验；
2. `--model` 身份校验（相对路径 + 字节数 + SHA256，含 `model.safetensors`）；
3. CUDA 要求、冻结 scorer 与本地模型/processor 加载；
4. 一次真实多模态 warmup（`max_new_tokens=1`，记录丢弃）；
5. 一次真实样本推理（`sample_id=1838`）。

随后在**写文件之前**的归一化阶段 fail-closed：

```
Edge576ContractError: frozen dense_edge576 record is missing 'output_token_ids' (fail closed)
```

失败位置：`evaluation/run_edge576_cvbench_three_arm.py::normalize_record`。

## 未触碰 / 未写出的证据

| 项目 | 结果 |
| --- | --- |
| smoke 目录 `artifacts/token_selector/2026-10-07-edge576-smoke/` | 不存在 |
| smoke 文件数 | 0 |
| full2638 目录 `artifacts/token_selector/2026-10-07-edge576-cvbench/` | 不存在 |
| full2638 输出文件数 | 0 |
| 输出流是否打开 | 否（异常发生在 `output.open(...)` 之前） |
| `gpu_runs`（该次尝试） | 1 |
| `weights_loaded` / `model_loaded` | true / true |
| `records_written` | 0 |

## 根因（以现有 runner 真实字段为准）

r3 的归一化契约**强制要求每条记录都带 `output_token_ids`**，但三条臂的冻结生产者返回 schema 并不一致：

- **dense**：`evaluation/run_hf_shallow_block_pruning.py::run_one` 的 record（358–403 行）只有
  `output_text`、真实整数计数 `output_tokens`（== `len(generated)`）、`selected_group_indices`、`image_grid_thw`，
  **从不返回 `output_token_ids`**；
- **L2**：`evaluation/run_hf_hardware_regular_b1_corrected_l2.py` 返回 `output_token_ids` 与 `output_tokens`；
- **v2**：`evaluation/run_hf_conservative_l2_guarded_rerank.py` 返回 `output_token_ids` 与 `output_tokens`。

即 r3 假设「三臂 record schema 完全一致」被证伪。主指标只用 `output_text` 走 `parse_choice` 诚实解析，**不需要 token id**；`output_tokens` 是真实等价可审计字段。

## 修复决定（r4）

- `output_token_ids` 改为**可选**：只有冻结生产者真实返回时才记录，**绝不伪造**；
- 新增**必备**可审计字段 `output_token_count`，由真实字段规范映射：
  - 有 `output_token_ids` → 取其真实列表长度；
  - 有 `output_tokens` → 取其真实整数值；
  - 两者都有 → 必须一致，否则 fail-closed；
  - 两者都无 → fail-closed；
- 保留核心字段 fail-closed：缺 `output_text` 或 `selected_group_indices` 仍然拒绝；
- 新增 `output_token_ids_available` 明确标注 token id 是否真实存在；
- resume 校验同步要求 `output_token_count` 为非负整数、`output_token_ids_available` 为 bool，且与可选 `output_token_ids` 一致。

## 处置

- r3 标记为 **superseded**（见 `2026-10-07-edge576-cvbench-r3-superseded.{json,md}`）；
- r3 config/报告/生成器/测试**字节全部保留**，不得重新生成；
- 由 r4 预注册钉住修复后的执行层，执行顺序仍为：模型身份 → **三臂 smoke 全通过** → Dense full2638 → 修正整数 L2 full2638 → v2 full2638 → 官方评分与分层配对 bootstrap。

## 边界

本证据由操作者报告 + 磁盘状态 + 冻结代码路径确定性记录；撰写本证据期间**未运行 GPU、未加载模型权重、未联网**。该次尝试不是精度/加速/部署证据，也未产生任何 full2638 结果。
