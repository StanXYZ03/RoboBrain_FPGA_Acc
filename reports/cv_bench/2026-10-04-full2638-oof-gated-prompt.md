# full2638 out-of-fold G-gated Prompt 诊断（探索性、只读）

- 生成时间：`2026-10-04T22:07:34.404155+08:00`
- 范围：Exploratory out-of-fold G-gated Prompt model selection on the frozen full2638 corrected-L2 arms; CPU-only, read-only; NOT a new formal claim and NOT a gate.
- 探索性模型选择，**不是新正式主张、不构成门**；冻结正式结论不变：**PROMPT_GAIN_NOT_CONFIRMED_FULL2638**
> Threshold discovery here uses the same full2638 corpus under 5-fold cross-validation, so the OOF score is an internally cross-validated estimate, not an untouched external evaluation. A paper/formal claim still requires replaying the frozen policy on an external frozen dataset.

> ⚠️ 阈值发现与评估共用同一 full2638 语料（5-fold 交叉验证），因此 OOF 分数只是内部交叉验证估计，不是未触碰的外部评测；在提出论文/正式主张之前，必须把冻结策略在外部冻结数据集上重放。

- 唯一主指标：`score(...)["cv_bench_accuracy"]`；unweighted 仅次要诊断。
- dense 角色：descriptive_reference_only。

## 1. 固定设计

- 5-fold，seed=20261004，分层=`task::source_dataset`；分配：strata processed in sorted order; ids sorted (numeric-aware) and shuffled with one random.Random(fold_seed); dealt round-robin across the folds。
- 候选策略（固定顺序）：['never_prompt', 'always_prompt', 'prompt_if_G_gt_96', 'prompt_if_G_gt_256', 'prompt_if_G_gt_512', 'prompt_if_G_gt_1024']。
- 策略规则：never_prompt uses the integer-L2 arm on every sample; always_prompt uses the Prompt+integer-L2 arm on every sample; prompt_if_G_gt_T uses the Prompt arm iff the pre-selector G = full_llm_visual_tokens is strictly greater than T, else the integer-L2 arm. No task/source/dataset identity enters any policy.
- 选择规则：In each fold keep the other four folds, score all six candidate policies with the official cv_bench_accuracy, pick the maximum; exact ties prefer the lower Prompt-use rate, then the fixed order (never_prompt, always_prompt, prompt_if_G_gt_96, prompt_if_G_gt_256, prompt_if_G_gt_512, prompt_if_G_gt_1024). Apply the winner unchanged to the held-out fold.
- bootstrap：seed=20261004 / repeats=5000 / level=0.95 / 分层=task::source_dataset；**不作门**。

## 2. 输入合同与复现

- 输入 SHA pin 全部通过：True
- 复现冻结基线：never=integer 臂 True；always=Prompt 臂 True；dense True。

## 3. Fold 选择

| fold | train n | held-out n | 选中策略 | train official | held-out official | held-out Prompt 使用率 |
|---:|---:|---:|---|---:|---:|---:|
| 0 | 2108 | 530 | prompt_if_G_gt_512 | 0.786460 | 0.785258 | 0.373585 |
| 1 | 2110 | 528 | prompt_if_G_gt_512 | 0.788114 | 0.778663 | 0.384470 |
| 2 | 2111 | 527 | prompt_if_G_gt_512 | 0.789928 | 0.771433 | 0.390892 |
| 3 | 2111 | 527 | prompt_if_G_gt_512 | 0.780210 | 0.810201 | 0.370019 |
| 4 | 2112 | 526 | prompt_if_G_gt_512 | 0.786377 | 0.785541 | 0.391635 |

- 选择跨 fold 稳定：**True**；各策略被选次数：{'prompt_if_G_gt_512': 5}。

## 4. OOF 结果与对比

- OOF official `cv_bench_accuracy`=**0.786218**；unweighted=0.781274；Prompt 使用率=**0.382108**（1008/2638）。
- never_prompt official=0.766852；always_prompt official=0.777362；dense（描述性）=0.872730。
- OOF−never = **+0.019366**，95% CI [+0.007391, +0.030968]。
- OOF−always = **+0.008856**，95% CI [-0.000686, +0.018654]。
- OOF−dense（仅描述，无 bootstrap）= -0.086512。

## 5. 迁移（相对 never / always）

- 相对 never_prompt（integer 臂）：W→R=133、R→W=86、不变对=1928、不变错=491、净修正=+47。
- 相对 always_prompt（Prompt 臂）：W→R=108、R→W=87、不变对=1953、不变错=490、净修正=+21。

## 6. 按 task

| 层 | n | Prompt 使用率 | OOF(unweighted) | never | always | OOF−never | OOF−always | 对 never 净修正 | OOF official |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Count | 788 | 0.200508 | 0.535533 | 0.517766 | 0.548223 | +0.017766 | -0.012690 | +14 | n/a |
| Depth | 600 | 0.666667 | 0.878333 | 0.870000 | 0.868333 | +0.008333 | +0.010000 | +5 | n/a |
| Distance | 600 | 0.666667 | 0.863333 | 0.810000 | 0.836667 | +0.053333 | +0.026667 | +32 | n/a |
| Relation | 650 | 0.076923 | 0.913846 | 0.920000 | 0.900000 | -0.006154 | +0.013846 | -4 | n/a |

## 7. 按 source

| 层 | n | Prompt 使用率 | OOF(unweighted) | never | always | OOF−never | OOF−always | 对 never 净修正 | OOF official |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ADE20K | 633 | 0.328594 | 0.660348 | 0.644550 | 0.660348 | +0.015798 | +0.000000 | +10 | n/a |
| COCO | 805 | 0.000000 | 0.742857 | 0.742857 | 0.744099 | +0.000000 | -0.001242 | +0 | n/a |
| Omni3D | 1200 | 0.666667 | 0.870833 | 0.840000 | 0.852500 | +0.030833 | +0.018333 | +37 | n/a |

## 8. 按 source_dataset

| 层 | n | Prompt 使用率 | OOF(unweighted) | never | always | OOF−never | OOF−always | 对 never 净修正 | OOF official |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ADE20K Validation Set | 633 | 0.328594 | 0.660348 | 0.644550 | 0.660348 | +0.015798 | +0.000000 | +10 | n/a |
| COCO 2017 Validation Set | 805 | 0.000000 | 0.742857 | 0.742857 | 0.744099 | +0.000000 | -0.001242 | +0 | n/a |
| Omni3D_Hypersim | 400 | 1.000000 | 0.830000 | 0.767500 | 0.830000 | +0.062500 | +0.000000 | +25 | n/a |
| Omni3D_SUNRGBD | 400 | 0.000000 | 0.982500 | 0.982500 | 0.927500 | +0.000000 | +0.055000 | +0 | n/a |
| Omni3D_nuScenes | 400 | 1.000000 | 0.800000 | 0.770000 | 0.800000 | +0.030000 | +0.000000 | +12 | n/a |

## 9. 按 pre-selector G 桶（G = full_llm_visual_tokens）

| 层 | n | Prompt 使用率 | OOF(unweighted) | never | always | OOF−never | OOF−always | 对 never 净修正 | OOF official |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| <=96 | 143 | 0.000000 | 0.727273 | 0.727273 | 0.727273 | +0.000000 | +0.000000 | +0 | n/a |
| 97-256 | 781 | 0.000000 | 0.878361 | 0.878361 | 0.860435 | +0.000000 | +0.017926 | +0 | 0.914510 |
| 257-512 | 706 | 0.000000 | 0.737960 | 0.737960 | 0.728045 | +0.000000 | +0.009915 | +0 | 0.784807 |
| 513-1024 | 448 | 1.000000 | 0.799107 | 0.747768 | 0.799107 | +0.051339 | +0.000000 | +23 | n/a |
| >1024 | 560 | 1.000000 | 0.700000 | 0.657143 | 0.700000 | +0.042857 | +0.000000 | +24 | n/a |

## 10. 输入 SHA256

| 输入 | SHA256 |
|---|---|
| formal_report | `53ec2ee10fde7b31b08a840263bd26409e35781ed25b89b20166ea90e84af6ea` |
| full_manifest | `22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91` |
| corrected_l2_integer_arm | `d0e302f6b91279374ddd507db4d300ca0c29e16f522db57f47c8de059f103a5f` |
| prompt_plus_corrected_l2_integer_arm | `6d3b97e8ce768066a99c9b9f517ddf602480fa75d57ce2e36b54e26baabc90fa` |

## 11. 限制

- Threshold discovery here uses the same full2638 corpus under 5-fold cross-validation, so the OOF score is an internally cross-validated estimate, not an untouched external evaluation. A paper/formal claim still requires replaying the frozen policy on an external frozen dataset.
- This is exploratory model selection over frozen outputs; it changes no raw output, config or prior report and creates no formal claim.
- The official cv_bench_accuracy requires all three sources (ADE20K, COCO, Omni3D); inside single-source task/source strata only the unweighted sample accuracy is reported and it is not a gate.
- Threshold discovery and evaluation share the same full2638 corpus under cross-validation; selections being stable across folds does not make the score an external evaluation.
- No GPU, model, board, network or training is used; the whole diagnostic is a CPU-only read-only re-analysis of frozen JSONLs.

> CPU-only 只读诊断：未改任何权威输入、既有配置/报告或 `configs/evaluation_protocol.json`；无 FPGA/时延/加速声明。
