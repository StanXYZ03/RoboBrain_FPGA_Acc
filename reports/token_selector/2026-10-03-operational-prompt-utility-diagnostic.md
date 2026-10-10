# 部署态 Prompt 效用诊断（探索性）

- 日期：2026-10-03；状态：`OPERATIONAL_PROMPT_UTILITY_NOT_ESTABLISHED`
- 这是**已使用 validation 的零训练探索性诊断**，不是最终精度，不改上一 contrastive 结论，未触碰 internal test。

## 主 Holm 家族（恰好 2 项，仅 K48 retained mass）

| 检验 | mean | CI low | CI high | dz | raw p | Holm p | +seeds | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1_B_correct_minus_wrong_selector_mass_k48 | +0.137908 | +0.072964 | +0.205801 | 0.6503 | 0.00040 | 0.00080 | 3/3 | True |
| P2_B_minus_A_correct_minus_wrong_gap_mass_k48 | +0.085962 | +0.035535 | +0.136801 | 0.5247 | 0.00180 | 0.00180 | 3/3 | True |

## 辅助门槛 S1–S4（不进入 Holm）

- S1 grounding mean = -0.0019252304813808796, pointing mean = 0.2877651809180278；pass = False
- S2 K48 gap NDCG mean = 0.05281094218280773；pass = True
- S3 K48 correct-vs-wrong Jaccard mean = 0.24601759836970738 (threshold 0.95)；pass = True
- S4 prior gates all pass = True, pins unchanged = True, identity = True；pass = True

## 各任务 K48（arm B，correct − wrong）

| 任务 | gap mass mean | gap NDCG mean | correct-vs-wrong Jaccard |
| --- | --- | --- | --- |
| grounding | -0.001925 | -0.014836 | 0.2460 |
| pointing | +0.287765 | +0.125925 | 0.2460 |

## K48/64/96（arm B 与 A 的 correct − wrong gap mass，overall）

| budget | B gap mass mean | A gap mass mean |
| --- | --- | --- |
| k48 | +0.137908 | +0.051946 |
| k64 | +0.133796 | +0.054134 |
| k96 | +0.117520 | +0.016057 |

## 决策

- **OPERATIONAL_PROMPT_UTILITY_NOT_ESTABLISHED**（P1/P2 = True，S1–S4 = False）

## 隔离与溯源

- 打开 NPZ 数：76（仅 val：True）
- test NPZ 打开：[]；train NPZ 打开：[]
- test 路径解析/哈希/打开：False
- 预注册 config 运行前后 SHA 不变：True
- 预注册 report 运行前后 SHA 不变：True
- 所有 pin 输入不变：True
- val NPZ aggregate 与预注册一致：True

## 限制

- Exploratory diagnostic on already-used validation; not a final accuracy conclusion.
- No internal test NPZ was resolved, hashed, opened, parsed or scored.
- The val split was already used during development; this is not final or CV-Bench accuracy.
- No hardware synthesis, latency or speedup claim.
- The prior contrastive decision PROMPT_CONTRASTIVE_STUDENT_NOT_ESTABLISHED is unchanged; this diagnostic only adds exploratory evidence.
- The prior report JSON is parsed only for the S4 gate booleans; its statistics and val-samples CSV are never metric sources.
- Per-image statistics treat the 38 images as independent units.
- The selector chain is descriptive; passing P1/P2 does not by itself establish task accuracy or promotion.
