# B1 答案 choice-margin 真实模型单图 GPU smoke

- 日期：2026-10-05
- 结论：**REAL_MODEL_AUTOGRAD_SMOKE_PASS**
- 模型：`/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B`（RoboBrain2.5-4B-BF16，bf16，cuda:0，SDPA）
- 范围：Real-model single-image B1 answer-choice-margin autograd smoke on cuda:0, BF16/SDPA. No training, no optimizer, no network.

## 1. 这说明了什么，不说明什么

本 smoke **只**验证一次真实模型前向 + 反向的自动微分链路能否把选择答案的
margin 梯度送回 B1 视觉 hidden states。它**不是** selector 精度证据，**不是**训练，
也**不是**稳态推理延时或 selector 加速。

- `engineering_autograd_smoke_only = true`、`is_selector_accuracy_evidence = false`、
  `is_training = false`。
- 只运行 `sample_id="1"` 这一张图，不得外推到 full2638。
- 官方答案标签不得在 CV-Bench full2638 上训练再对同集宣称结果。

## 2. 固定样本

- sample_id：`1`，task：`Count`，source：`ADE20K Validation Set`。
- answer：`(E)`（正确索引 `4`）。
- prompt：`How many cushions are in the image? Select from the following choices.
(A) 4
(B) 5
(C) 8
(D) 0
(E) 6
(F) 7`。

## 3. 标签分叉 token（真实 tokenizer，add_special_tokens=False）

- 原始标签 token 序列：`[[4346, 8], [5349, 8], [3025, 8], [5432, 8], [10722, 8], [7832, 8]]`。
- 共同前缀：`[]`；共同后缀：`[8]`。
- 分叉 token id：`[4346, 5349, 3025, 5432, 10722, 7832]`；解码：`['(A', '(B', '(C', '(D', '(E', '(F']`。
- 仅当每个标签都是「共同前缀 + 恰好一个互不重复的分叉 token + 共同后缀」时才继续；
  否则 `FAIL_CLOSED`，绝不伪造单 token margin。

## 4. 真实链路断言

| 项 | 值 |
| --- | --- |
| B1 hidden shape | `[2688, 1024]` |
| image_grid_thw | `[1, 42, 64]` |
| G（full_llm_visual_tokens） | `672` |
| merge_unit | `4` |
| retention / prune_after_layer | 1.000000 / `0` |
| 是否裁剪 token | `False` |

### margin 与预测

- margin 定义：`logit((E)) - max(other branch-token logits)`
- margin 值：2.000000
- 预测选项：`(E)`（索引 4），正确选项：`(E)`。
- 各分支 token logit：`[18.125, 20.25, 18.125, 15.5, 22.25, 18.875]`。

### 梯度

- B1 grad 存在：`True`，有限：`True`，非零：`True`，范数：12.834799。
- 每组 grad×activation saliency 有限：`True`，非零：`True`，组数：`672`。
- 模型全部 parameter.grad 为 None：`True`；全部参数 requires_grad=False：`True`。

## 5. 一次性计时与显存

所有 GPU 计时段前后均 `torch.cuda.synchronize()`，且不计模型加载。

| 阶段 | 秒 |
| --- | --- |
| processor | 0.088575 |
| forward（B1 视觉 + LLM + margin） | 0.610485 |
| backward | 0.702670 |
| 总 smoke | 1.458584 |

- peak allocated：`13821010944` bytes；peak reserved：`13977518080` bytes。
- **明确**：这些数字不是稳态推理延时，也不代表 selector 加速。

## 6. 校验清单

| 检查 | 通过 | 说明 |
| --- | --- | --- |
| single_sample_sample_id_1 | ✅ | sample_id='1' |
| labels_a_to_f | ✅ | labels=['(A)', '(B)', '(C)', '(D)', '(E)', '(F)'] |
| answer_is_E | ✅ | answer='(E)' correct_index=4 |
| label_single_branch_token | ✅ | prefix=[] branches=[4346, 5349, 3025, 5432, 10722, 7832] suffix=[8] |
| b1_hidden_shape_equals_patch_tokens | ✅ | b1_hidden_shape=[2688, 1024] full_patch_tokens=2688 |
| b1_hidden_grad_present | ✅ | prune_hidden_states.grad is not None after backward |
| b1_hidden_grad_finite | ✅ | gradient is finite |
| b1_hidden_grad_nonzero | ✅ | gradient has a non-zero entry |
| group_saliency_all_finite | ✅ | 672 groups finite |
| group_saliency_all_nonzero | ✅ | 672 groups non-zero |
| all_model_parameter_grads_none | ✅ | 713 parameters have grad None |
| all_model_parameters_requires_grad_false | ✅ | 713 parameters are frozen |
| margin_finite | ✅ | margin=2.0 |

- fail_closed_reason：`None`

## 7. 口径与禁止声明

- **这不是 selector 精度证据**，不得据此声称选择器精度、CV-Bench 分数或部署收益。
- **这不是训练**：无 optimizer，`model.requires_grad_(False)`，模型参数梯度全为 None。
- **不得用 full2638 标签训练后再对同集评估**；如需训练必须使用独立留出/外部冻结集。
- 只运行一张图，不得外推；计时为一次性诊断，不是稳态延时或 selector 加速。

- **下一道门**：若本 smoke PASS，才允许在独立留出集上设计 B1 margin teacher 的小规模可行性实验；本 smoke 本身不得升级为精度或训练结论。
