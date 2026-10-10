# B1 答案 choice-margin 自动微分预检（CPU-only）

- 日期：2026-10-05
- 状态：**PREFLIGHT_PASS**
- 范围：CPU-only engineering feasibility preflight for a future B1 answer-choice-margin teacher signal. No model weights, no GPU, no network, no training, no dependency installation.

## 1. 这说明了什么，不说明什么

本报告只证明**张量与梯度合同可行**：B1 hidden states 能够保持在同一自动微分图中，
经过可微分组 gather、冻结 fake lm_head 与 choice margin，最终得到有限且非零的梯度。

- `engineering_feasibility_only = true`：这只是工程可行性。
- `is_selector_accuracy_evidence = false`：**这不是 selector 精度证据**，不代表任何
  选择器更准、更差或可用于部署。
- 本预检未加载任何模型权重、未使用 GPU、未联网、未训练、未安装依赖。

## 2. 可微分组 gather

- 输入 `hidden_states`：`[120, 64]`，`merge_unit = 4`。
- 输出：`[32, 64]`，第 `k*merge_unit + j` 行等于 `hidden_states[group_k*merge_unit + j]`（`order_preserved = True`）。
- 选中组索引：`[0, 1, 3, 6, 12, 14, 19, 22]`。
- 实现约束：`uses_detach = False`、`uses_numpy = False`、`uses_cpu_rebuild = False`；
  即不 detach、不 numpy、不在 CPU 重建，梯度可回到 B1 hidden。

## 3. 前向 shape 与 choice margin

- readout `[64]` → logits `[32]` → answer logits `[3, 32]`。
- margin 定义：`correct_option_logit - max(other_option_logits)`，支持每题可变选项数（A-D / A-F / 任意给定 token ids）。

| 题 | 选项数 | 选项 token ids | 正确索引 | margin |
| --- | --- | --- | --- | --- |
| Q0 | 4 | `[3, 7, 11, 19]` | 2 | `-0.226357` |
| Q1 | 6 | `[0, 4, 8, 12, 16, 20]` | 5 | `-0.259998` |
| Q2 | 3 | `[1, 2, 5]` | 0 | `-0.216041` |

## 4. 梯度与负对照

- B1 hidden 梯度存在：`True`，有限：`True`，非零：`True`，范数：`0.3603646755218506`。
- 位置选择路径（`choice_margin_from_positions`）梯度存在且非零：`True` / `True`。
- 负对照：`Detach the compact tensor before the fake lm_head; the original B1 hidden states must then receive no gradient.`
  - 结果 `b1_control_grad_is_none = True`：detach 后原 B1 hidden **确实得不到梯度**，证明正向梯度来自真实计算图而非偶然。

## 5. 校验清单

| 检查 | 通过 | 说明 |
| --- | --- | --- |
| differentiable_group_gather_shape | ✅ | compact shape (32, 64) |
| differentiable_group_gather_requires_grad | ✅ | gathered tensor is connected to the B1 hidden states |
| differentiable_group_gather_order | ✅ | row k*merge_unit+j equals hidden[group_k*merge_unit+j] |
| choice_margin_shape | ✅ | margins shape (3,) |
| choice_margin_finite | ✅ | margins [-0.22635748982429504, -0.25999757647514343, -0.21604061126708984] |
| b1_hidden_grad_present | ✅ | backward populated hidden_states.grad |
| b1_hidden_grad_finite | ✅ | gradient is finite |
| b1_hidden_grad_nonzero | ✅ | gradient has a non-zero entry |
| negative_control_detach_blocks_grad | ✅ | detaching the compact tensor disconnects the graph, leaves the control hidden grad as None and makes backward raise |
| positional_margin_grad_present | ✅ | choice_margin_from_positions keeps a non-zero gradient |

## 6. 标签卫生与后续门

- **这只是工程可行性，不是 selector 精度证据**；不得据此报告声称选择器精度、CV-Bench 分数或任何部署收益。
- **官方答案标签不得在 CV-Bench full2638 上训练再对同集宣称结果**；若需训练必须使用独立留出/外部冻结集。
- **后续 GPU smoke 只能一张图、不得训练**；只验证张量与梯度合同在真实模型上成立。

- 结构化标签卫生声明：Official CV-Bench answer labels must never be used to train a selector on the full2638 set and then be re-scored on that same set as if it were held out.
- 结构化 GPU 约束：Any follow-up GPU smoke test may use exactly one image and must not train.
- **下一道门**：GPU single-image smoke of the frozen B1 forward that reuses differentiable_group_gather and choice_margin with a real lm_head; one image only, no training, no CV-Bench accuracy claim.

禁止据此报告声称：selector accuracy improvement、CV-Bench full2638 accuracy、FPGA speedup or latency、teacher-signal quality。

机器可读结果：同目录 JSON（`status`、逐项 `checks`、margin 数值与梯度证据）。
