# 2026-10-02 Prompt 可辨识性 / 退化诊断 v2（中文报告）

> 本报告为 **v2**，用于修正 v1 的统计设计（split-local donor、image derangement、image-level 聚类、per-image prompt control、`np.array_equal` 精确不变量、工程门槛措辞、根因因子拆分）。v1 结论 **SUPERSEDED_FOR_INFERENCE_BY_V2**，不得作为最终推断；v1 原始数值未改动。

本报告只做**诊断**：不训练、不调参、不加载 4B 模型、不联网、CPU 运行；所有 score/metric/load 原语复用 `evaluation/prompt_projection_distillation.py` 与 `evaluation/train_prompt_projection.py`。

## 1. 四个预注册标签（阈值在读取 test 之前固定）

| 标签 | 判定 |
| --- | --- |
| `DATA_OR_SCHEMA_INVALID` | false |
| `TEACHER_PROMPT_SIGNAL_WEAK` | **TRUE** |
| `STUDENT_PROMPT_COLLAPSE` | **TRUE** |
| `GENERALIZATION_OVERFIT` | **TRUE** |

- `primary_blocker` = **STUDENT_PROMPT_COLLAPSE**；`contributing_factor` = **TEACHER_PROMPT_SIGNAL_BELOW_ENGINEERING_GATE**；`training_factor` = **GENERALIZATION_OVERFIT**
- 显著性判定：paired 95% bootstrap CI 下界 > 0 且 |Cohen's dz| ≥ 0.2 且双侧符号翻转 permutation p < 0.05。
- `TEACHER_PROMPT_SIGNAL_WEAK`：教师 prompt/image 排序变化比（Spearman 距离）bootstrap 95% CI 上界 < 0.5。**该 0.5 为预注册工程门槛 / 启发式判据，不是文献通用阈值**；它只说明教师 prompt 信号低于该工程门槛，不能单独宣称 teacher 失效。
- gate 仅用 **val** 判定；阈值与 derangement seed 为模块常量，在读取 test 前固定；test 只在 gate 固定后报告一次。所有 donor 严格 **split-local**。

## 2. 关键数字（每个 split）

| split | n_rec | n_img | teacher same-image rho | teacher cross-image same-G rho | prompt/image ratio | student correct rho | student same-task rho | student cross-shuffle rho | delta(same) rho(图像级) | delta(cross) rho(图像级) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 318 | 159 | 0.7503 | 0.0174 | 0.254 | 0.0831 | 0.0831 | 0.0852 | +0.0000 | -0.0021 |
| val | 76 | 38 | 0.7331 | 0.0165 | 0.271 | 0.0565 | 0.0565 | 0.0677 | +0.0000 | -0.0112 |
| test | 78 | 39 | 0.7686 | 0.0374 | 0.240 | 0.0536 | 0.0536 | 0.0421 | +0.0000 | +0.0115 |

### 2.1 教师 target 同图两任务 vs 跨图对照（val）

| metric | same-image | cross-image | ratio | ratio CI |
| --- | --- | --- | --- | --- |
| pearson_raw | 0.8069 | 0.0113 | 0.195 | [0.139, 0.265] |
| spearman_raw | 0.7331 | 0.0165 | 0.271 | [0.211, 0.339] |
| cosine_raw | 0.9805 | 0.8853 | 0.170 | [0.126, 0.226] |
| l1_normalized | 0.0653 | 0.2063 | 0.316 | [0.256, 0.384] |
| js_soft | 0.0730 | 0.8360 | 0.087 | [0.043, 0.145] |
| overlap@48 | 0.7861 | 0.1899 | 0.264 | [0.194, 0.344] |
| jaccard@48 | 0.6744 | 0.1064 | 0.364 | [0.283, 0.451] |
| overlap@64 | 0.7781 | 0.2557 | 0.298 | [0.224, 0.375] |
| jaccard@64 | 0.6600 | 0.1479 | 0.399 | [0.323, 0.480] |
| overlap@96 | 0.7965 | 0.3693 | 0.323 | [0.257, 0.391] |
| jaccard@96 | 0.6771 | 0.2289 | 0.419 | [0.342, 0.494] |

### 2.2 学生配对检验（val，correct 减对照；独立单位为 image）

| condition | metric | mean | CI low | CI high | dz | p | present | n_img | n_rec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| same | pearson | +0.00000 | +0.00000 | +0.00000 | 0.000 | 1.0000 | False | 38 | 76 |
| same | spearman | +0.00000 | +0.00000 | +0.00000 | 0.000 | 1.0000 | False | 38 | 76 |
| same | ndcg_full | +0.00000 | +0.00000 | +0.00000 | 0.000 | 1.0000 | False | 38 | 76 |
| cross | pearson | -0.00931 | -0.02783 | +0.00002 | -0.164 | 0.2533 | False | 38 | 76 |
| cross | spearman | -0.01124 | -0.03364 | +0.00001 | -0.163 | 0.5012 | False | 38 | 76 |
| cross | ndcg_full | -0.00275 | -0.00821 | +0.00000 | -0.164 | 0.4949 | False | 38 | 76 |

### 2.3 方差分解（val；公式见 JSON `variance.formula`）

| target | V_image | V_prompt | V_group | prompt_signal_ratio | fraction |
| --- | --- | --- | --- | --- | --- |
| teacher_normalized | 3.676e-03 | 2.668e-04 | 3.197e-02 | 0.0726 | 0.0677 |
| teacher_raw | 2.756e-05 | 1.332e-06 | 2.419e-04 | 0.0483 | 0.0461 |
| student_correct_raw | 2.405e-03 | 0.000e+00 | 1.867e-03 | 0.0000 | 0.0000 |

### 2.4 prompt embedding 原始统计（val）

- token 数：min 38 / mean 40.63 / max 45
- mean-pooled cosine 同图两任务 = 0.9580；跨图 cross-task 对照 = 0.9541 (std 0.0011)；逐 image delta = same_i - cross_i 的 mean = +0.0039 CI [+0.0036, +0.0044]，dz=3.063，p=0.0001；n_img=38，n_rec=76
- 说明：cross 项为同一 donor image j 的 (g_i↔p_j) 与 (p_i↔g_j) 的 image 内平均，保持与同图 g_i↔p_i 相同的 cross-task 条件；**不使用** same_i 减全局 cross 均值。

### 2.5 学生 max-over-token 归约坍塌与投影谱

- val split（split-local）中，所有 group 由**同一个** prompt token 取得 argmax 的样本占比 = 0.9737（均值 distinct argmax token 数 1.026，最大 2）
- correct 与同图另一任务 prompt 的 student 分数 **`np.array_equal` 精确相等** 的样本占比 = 1.0000（76 对）；容差 1e-9 版本（明确命名）= 1.0000
- 逐对 max_abs_diff：max = 0.000e+00；median = 0.000e+00；p95 = 0.000e+00
- 同图两 prompt 的 dominant projected token 方向 cosine 均值 = 1.000000（38 对）
- Wt effective rank = 123.918，min singular = 0.7981，数值秩(s>1e-6*s0) = 128
- Wv effective rank = 119.950，min singular = 0.4217，数值秩(s>1e-6*s0) = 128

## 3. 参数量与泛化

- trainable parameters = 458752 (Wt 327680 + Wv 131072)
- train record 数 = 318；唯一 image 数 = 159；teacher group-target 总数 = 86640
- params/record = 1442.62；params/image = 2885.23；params/group-target = 5.29
- 注意：group target 在同一 image / 同一 prompt 内相关，**不能当作独立样本**；params/group-target 仅为完整披露，不代表相对独立样本的参数比。
- best_epoch = 0，best_val_KL = 0.025189；train KL 0.025643 → 0.014819；val KL final 0.027907
- train_loss_decreased = True；val_never_improved_after_init = True；归一化 KL gap 比 = 0.251
- 逐样本 pearson: train 0.1079 vs val 0.0731；gap +0.0348；Cohen's d = 0.238
- 逐样本 spearman: train 0.0831 vs val 0.0565；gap +0.0267；Cohen's d = 0.179
- 逐样本 ndcg_full: train 0.8549 vs val 0.8499；gap +0.0049；Cohen's d = 0.135

## 4. 结论与解释

诊断口径是**教师 target 的可辨识性**与**学生是否真正利用了 prompt**，不是 CV-Bench 任务精度。v2 的因子拆分为：`primary_blocker` = **STUDENT_PROMPT_COLLAPSE**（学生 evidence 优先）；`contributing_factor` = **TEACHER_PROMPT_SIGNAL_BELOW_ENGINEERING_GATE**（教师信号低于**工程门槛**）；`training_factor` = **GENERALIZATION_OVERFIT**。教师 0.5 门槛为预注册启发式判据，不得单独据此宣称 teacher 失效。

**归约坍塌机制（2.5）**：学生分数 `max_t cosine(...)` 在每个 group 上由单一 prompt token 主导，而同图两 prompt 的该主导投影方向几乎相同，故 correct 与同图另一任务分数 `np.array_equal` 精确相等；这说明即使 `Wt` 数值满秩，`max` 归约仍使 prompt 信息不可用，属于学生打分函数的退化。

## 5. 下一门槛（不替 PI/Codex 做算法决策）

1. 若 `DATA_OR_SCHEMA_INVALID=true`：停止一切后续，先修复对应 schema/数据不一致并重跑本诊断。
2. 若数据有效：任何 scorer ablation / 后续算法必须以 v2 的 `primary_blocker` / `contributing_factor` / `training_factor` 与 image-level CI 作为准入条件，由 PI 决定是否接受该负结果并冻结，或在明确授权下更换监督信号/优化协议后重训（不得读 test 调参）。
3. v1 结论 **SUPERSEDED_FOR_INFERENCE_BY_V2**：在 v2 复核前，v1 的 `TEACHER_PROMPT_SIGNAL_WEAK` 根因判断不得用于后续推断。

---
provenance：python 3.10.20 / numpy 2.2.6；seed 20261002；derangement_seeds {'train': 20261002, 'val': 20261103, 'test': 20261204}；config_sha256 `44399c8e6fb8f333…`；index_sha256 `02a50868a8467541…`；projection_sha256 `132a6d9fcd35ee94…`
