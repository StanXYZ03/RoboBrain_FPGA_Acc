> **SUPERSEDED_FOR_INFERENCE_BY_V2**：v1 的统计设计有缺陷（donor 跨 split、跨图有放回抽样、样本级独立性等），结论不得用于推断；请以 v2 报告为准。以下 v1 原始数值未改动。

# 2026-10-02 Prompt 可辨识性 / 退化诊断（中文报告）

本报告只做**诊断**：不训练、不调参、不加载 4B 模型、不联网、CPU 运行；所有 score/metric/load 原语复用 `evaluation/prompt_projection_distillation.py` 与 `evaluation/train_prompt_projection.py`。

## 1. 四个预注册标签（阈值在读取 test 之前固定）

| 标签 | 判定 |
| --- | --- |
| `DATA_OR_SCHEMA_INVALID` | false |
| `TEACHER_PROMPT_SIGNAL_WEAK` | **TRUE** |
| `STUDENT_PROMPT_COLLAPSE` | **TRUE** |
| `GENERALIZATION_OVERFIT` | **TRUE** |

- `root_cause`（按数据 → 教师 → 学生 → 泛化优先级）= **TEACHER_PROMPT_SIGNAL_WEAK**
- 显著性判定：paired 95% bootstrap CI 下界 > 0 且 |Cohen's dz| ≥ 0.2 且双侧符号翻转 permutation p < 0.05。
- `TEACHER_PROMPT_SIGNAL_WEAK`：教师 prompt/image 排序变化比（Spearman 距离）bootstrap 95% CI 上界 < 0.5。

## 2. 关键数字（每个 split）

| split | n | teacher same-image rho | teacher cross-image rho | prompt/image ratio | student correct rho | student same-task rho | student cross-shuffle rho | lat 学生 delta(same) rho | delta(cross) rho |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 318 | 0.7511 | 0.0198 | 0.254 | 0.0831 | 0.0831 | 0.0836 | +0.0000 | -0.0005 |
| val | 76 | 0.7304 | 0.0160 | 0.274 | 0.0565 | 0.0565 | 0.0635 | +0.0000 | -0.0071 |
| test | 78 | 0.7746 | 0.0149 | 0.229 | 0.0536 | 0.0536 | 0.0466 | +0.0000 | +0.0070 |

### 2.1 教师 target 同图两任务 vs 跨图对照（val）

| metric | same-image | cross-image | ratio | ratio CI |
| --- | --- | --- | --- | --- |
| pearson_raw | 0.8026 | 0.0178 | 0.201 | [0.146, 0.261] |
| spearman_raw | 0.7304 | 0.0160 | 0.274 | [0.220, 0.335] |
| cosine_raw | 0.9805 | 0.8872 | 0.173 | [0.130, 0.227] |
| l1_normalized | 0.0654 | 0.2054 | 0.318 | [0.262, 0.381] |
| js_soft | 0.0766 | 0.8360 | 0.092 | [0.051, 0.149] |
| overlap@48 | 0.7810 | 0.1976 | 0.273 | [0.208, 0.342] |
| jaccard@48 | 0.6672 | 0.1116 | 0.375 | [0.298, 0.455] |
| overlap@64 | 0.7737 | 0.2554 | 0.304 | [0.239, 0.377] |
| jaccard@64 | 0.6539 | 0.1485 | 0.406 | [0.330, 0.485] |
| overlap@96 | 0.7926 | 0.3749 | 0.332 | [0.270, 0.395] |
| jaccard@96 | 0.6718 | 0.2337 | 0.428 | [0.359, 0.500] |

### 2.2 学生配对检验（val，correct 减对照）

| condition | metric | mean | CI low | CI high | dz | p | present |
| --- | --- | --- | --- | --- | --- | --- | --- |
| same | pearson | +0.00000 | +0.00000 | +0.00000 | 0.000 | 1.0000 | False |
| same | spearman | +0.00000 | +0.00000 | +0.00000 | 0.000 | 1.0000 | False |
| same | ndcg_full | +0.00000 | +0.00000 | +0.00000 | 0.000 | 1.0000 | False |
| cross | pearson | -0.00391 | -0.02177 | +0.01005 | -0.057 | 0.7473 | False |
| cross | spearman | -0.00707 | -0.02621 | +0.00725 | -0.095 | 0.4923 | False |
| cross | ndcg_full | -0.00229 | -0.00662 | +0.00081 | -0.134 | 0.5029 | False |

### 2.3 方差分解（val；公式见 JSON `variance.formula`）

| target | V_image | V_prompt | V_group | prompt_signal_ratio | fraction |
| --- | --- | --- | --- | --- | --- |
| teacher_normalized | 3.676e-03 | 2.668e-04 | 3.197e-02 | 0.0726 | 0.0677 |
| teacher_raw | 2.756e-05 | 1.332e-06 | 2.419e-04 | 0.0483 | 0.0461 |
| student_correct_raw | 2.405e-03 | 0.000e+00 | 1.867e-03 | 0.0000 | 0.0000 |

### 2.4 prompt embedding 原始统计（val）

- token 数：min 38 / mean 40.63 / max 45
- mean-pooled cosine 同图两任务 = 0.9580；跨图随机 = 0.9755 (std 0.0208)；paired delta mean = -0.0174 CI [-0.0175, -0.0173]，dz=-53.763，p=0.0001

### 2.5 学生 max-over-token 归约坍塌与投影谱

- 全部样本中，所有 group 由**同一个** prompt token 取得 argmax 的样本占比 = 0.9873（均值 distinct argmax token 数 1.013，最大 2）
- correct 与同图另一任务 prompt 的 student 分数**逐元素完全相等**的样本占比 = 1.0000（472 对）
- 同图两 prompt 的 dominant projected token 方向 cosine 均值 = 1.000000（236 对）
- Wt effective rank = 123.918，min singular = 0.7981，数值秩(s>1e-6*s0) = 128
- Wv effective rank = 119.950，min singular = 0.4217，数值秩(s>1e-6*s0) = 128

## 3. 参数量与泛化

- trainable parameters = 458752 (Wt 327680 + Wv 131072)；train samples = 318；参数/样本 = 1442.62
- best_epoch = 0，best_val_KL = 0.025189；train KL 0.025643 → 0.014819；val KL final 0.027907
- train_loss_decreased = True；val_never_improved_after_init = True；归一化 KL gap 比 = 0.251
- 逐样本 pearson: train 0.1079 vs val 0.0731；gap +0.0348；Cohen's d = 0.238
- 逐样本 spearman: train 0.0831 vs val 0.0565；gap +0.0267；Cohen's d = 0.179
- 逐样本 ndcg_full: train 0.8549 vs val 0.8499；gap +0.0049；Cohen's d = 0.135

## 4. 结论与解释

诊断口径是**教师 target 的可辨识性**与**学生是否真正利用了 prompt**，不是 CV-Bench 任务精度。若 `TEACHER_PROMPT_SIGNAL_WEAK` 为真，说明同图换 prompt 对 teacher target 的影响远小于换图，correct 与 shuffled 的监督几乎不可分；若 `STUDENT_PROMPT_COLLAPSE` 为真，说明学生输出的排序在 correct / 同图另一任务 / 跨图 shuffled 三种 prompt 下统计不可分（即输出近似 prompt 无关）。两者可同时成立。

**归约坍塌机制（2.5）**：学生分数 `max_t cosine(...)` 在每个 group 上由单一 prompt token 主导，而同图两 prompt 的该主导投影方向几乎相同，故 correct 与同图另一任务分数逐元素完全相等；这说明即使 `Wt` 数值满秩，`max` 归约仍使 prompt 信息不可用，属于学生打分函数的退化。

## 5. 下一门槛（可执行选项，最多 2 个；不替 PI 做研究决策）

1. 若 `DATA_OR_SCHEMA_INVALID=true`：停止一切后续，先修复对应的 schema/数据不一致并重跑本诊断。
2. 若数据有效：把本报告的 `TEACHER_PROMPT_SIGNAL_WEAK` / `STUDENT_PROMPT_COLLAPSE` / `GENERALIZATION_OVERFIT` 标签与 CI 作为下一阶段准入条件，由 PI 决定是否 (a) 接受该负结果并冻结，或 (b) 在明确授权下更换监督信号/优化协议后重训（不得读 test 调参）。

---
provenance：python 3.10.20 / numpy 2.2.6；seed 20261002；config_sha256 `44399c8e6fb8f333…`；index_sha256 `02a50868a8467541…`；projection_sha256 `132a6d9fcd35ee94…`
