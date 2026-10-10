# G>512 门控的数据源混杂诊断（full2638，探索性）

- 生成时间：2026-10-04T22:25:48.052536+08:00
- 分析：`full2638_gate_source_confounding`
- **结论：`SOURCE_CONFOUNDING_NOT_RESOLVED`**
- 正式结论（不变）：`PROMPT_GAIN_NOT_CONFIRMED_FULL2638`，`formal_result_unchanged=true`
- 范围：Exploratory source-dataset confounding quantification of the frozen prompt_if_G_gt_512 gate on the frozen full2638 corrected-L2 arms; CPU-only, read-only; NOT a new formal claim and NOT a gate.

> 本报告所有子组 / bootstrap / 依赖诊断数字均为冻结结论 PROMPT_GAIN_NOT_CONFIRMED_FULL2638 的事后探索性描述；Cramer's V 与互信息仅为描述性关联度量，不是因果证据、不作为门。正式 CV-Bench 结论不变。

## 1. 门控定义

- 门：`prompt_if_G_gt_512`，阈值 `512`
- 规则：prompt_if_G_gt_512 uses the Prompt+integer-L2 arm iff the pre-selector G = full_llm_visual_tokens is strictly greater than 512, else the integer-L2 arm. No task/source/dataset identity enters the gate.
- G 定义：G = full_llm_visual_tokens, the pre-selector real visual group count (one group per 2x2 merge unit)
- 门内样本 `1008`、门外 `1630`（门内占比 0.382108）

## 2. source_dataset × G-bin 分布

| source_dataset | <=96 | 97-256 | 257-512 | 513-1024 | >1024 | n | G>512 | G<=512 |
|---|---|---|---|---|---|---|---|---|
| ADE20K Validation Set | 112 | 263 | 50 | 48 | 160 | 633 | 208 | 425 |
| COCO 2017 Validation Set | 31 | 341 | 433 | 0 | 0 | 805 | 0 | 805 |
| Omni3D_Hypersim | 0 | 0 | 0 | 400 | 0 | 400 | 400 | 0 |
| Omni3D_SUNRGBD | 0 | 177 | 223 | 0 | 0 | 400 | 0 | 400 |
| Omni3D_nuScenes | 0 | 0 | 0 | 0 | 400 | 400 | 400 | 0 |

## 3. 门控 / 数据源依赖诊断（探索性，不作门）

- Cramer's V（`G>512` × source_dataset）= 0.880783
- 互信息（`G>512` × source_dataset）= 0.740323 bits
- Cramer's V（五档 G-bin × source_dataset）= 0.665422
- 互信息（五档 G-bin × source_dataset）= 1.163543 bits
- 上述两指标均为描述性关联度量，仅探索性，不是因果证据、不作为门。

## 4. 每个 source_dataset 内 Prompt−integer 效应

| source_dataset | n | W→R | R→W | net | unweighted Δ | 95% CI |
|---|---|---|---|---|---|---|
| ADE20K Validation Set | 633 | 44 | 34 | +10 | +0.015798 | [-0.011058, +0.044234] |
| COCO 2017 Validation Set | 805 | 61 | 60 | +1 | +0.001242 | [-0.024845, +0.028571] |
| Omni3D_Hypersim | 400 | 61 | 36 | +25 | +0.062500 | [+0.015000, +0.107500] |
| Omni3D_SUNRGBD | 400 | 6 | 28 | -22 | -0.055000 | [-0.082500, -0.027500] |
| Omni3D_nuScenes | 400 | 48 | 36 | +12 | +0.030000 | [-0.015000, +0.075000] |

（CI 为分层配对 bootstrap，重采样层为 `task::source_dataset`；
单数据源内官方 `cv_bench_accuracy` 未定义，故用无权重样本准确率，仅次要诊断。）

## 5. 焦点：ADE20K 同源效应差（interaction）

- ADE20K `G<=512`：n=425，W→R=20，R→W=20，net=+0，Δ=+0.000000
- ADE20K `G>512`：n=208，W→R=24，R→W=14，net=+10，Δ=+0.048077
- **interaction = Δ(G>512) − Δ(G<=512) = +0.048077**
- **95% 分层配对 bootstrap CI = [-0.018824, +0.114977]**
- 分层方式：`task::gate_bin`，seed `20261004`，repeats `5000`
- 判定规则：仅当 interaction CI 整体 `> 0`（下界 > 0）才可视为同源 G 效应得到区分；否则必须为 `SOURCE_CONFOUNDING_NOT_RESOLVED`。

## 6. 门控策略的样本内描述效应

- 说明：in-sample descriptive gate effect; the frozen out-of-fold estimate lives in the OOF report and is not recomputed as a new claim
- 门控官方 `cv_bench_accuracy` = 0.786218；integer = 0.766852；prompt = 0.777362
- 门控 − integer（无权重） = 0.017817；门控 − prompt（无权重） = 0.007961

## 7. 冻结复现与契约

- 冻结官方 Prompt−integer Δ = 0.010510；重算 = 0.010510；复现 = yes
- [x] exactly 2638 identical unique sample_ids in manifest and both arms
- [x] task/source/source_dataset come from the manifest
- [x] G taken from the integer arm full_llm_visual_tokens
- [x] gate is exactly G > 512 with no task/source identity
- [x] recomputed overall prompt-minus-integer delta reproduces the frozen report
- [x] inputs unchanged (re-hashed) before and after the analysis

## 8. 复现命令

```bash
python evaluation/analyze_gate_source_confounding.py --json reports/cv_bench/2026-10-04-gate-source-confounding.json --csv reports/cv_bench/2026-10-04-gate-source-confounding.csv --md reports/cv_bench/2026-10-04-gate-source-confounding.md
```
