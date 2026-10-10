# 同图双分辨率反事实：预注册计划（尚未执行）

- 设计 ID：`2026-10-04-same-image-resolution-counterfactual`
- 状态：**`PREREGISTERED_NOT_EXECUTED`**（未加载模型、未使用 GPU、未生成任何臂输出）
- 预注册 config：`configs/2026-10-04-same-image-resolution-counterfactual.preregistered.json`
- 冻结样本 manifest：`datasets/2026-10-04-same-image-resolution-counterfactual-ade20k100.jsonl`（100 条，SHA256 `c53c7b7d…d4c5`）
- 储备样本 manifest：`datasets/2026-10-04-same-image-resolution-counterfactual-ade20k-reserve.jsonl`（533 条，SHA256 `b6d9333b…005c`）
- 抽样报告：`reports/cv_bench/2026-10-04-same-image-resolution-counterfactual-selection.json`
- 门分离报告（processor 权威模式）：`reports/cv_bench/2026-10-04-same-image-resolution-counterfactual-gate-separation.json`（**待生成**）
- 门分离预测（analytic 非权威）：`reports/cv_bench/2026-10-04-same-image-resolution-counterfactual-gate-separation-prediction.json`

---

## 1. 为什么做这个反事实

`reports/cv_bench/2026-10-04-gate-source-confounding.json` 的结论是
**`SOURCE_CONFOUNDING_NOT_RESOLVED`**：在 full2638 中，`G>512` 门控与
`source_dataset` 高度混同（Cramer's V ≈ 0.8808，互信息 ≈ 0.7403 bits；`Omni3D_Hypersim`
全部落在门内、`Omni3D_SUNRGBD` 全部落在门外），因此“门控有效”无法与“数据源身份”
区分。ADE20K 是唯一同时跨越 `G<=512` 与 `G>512` 的数据源：

- ADE20K `G<=512`（n=425）：Prompt−integer 无权重 Δ = **0.000000**，W→R 20 / R→W 20 / net 0；
- ADE20K `G>512`（n=208）：Prompt−integer 无权重 Δ = **+0.048077**，W→R 24 / R→W 14 / net +10；
- 同源 interaction = `Δ(G>512) − Δ(G<=512)` = **+0.048077**，
  95% 分层配对 bootstrap CI = **[-0.018824, +0.114977]**（下界 <0，未整体 >0）。

由于这是不同图片之间的对比，仍无法排除“图片内容/数据源”混杂。本反事实把**同一张图**
分别放到低、高两个 processor pixel budget 下，让 `G` 在同一图像内跨越 512，从而从图像
内部识别分辨率（G）效应。

**正式 CV-Bench 结论不变**：`PROMPT_GAIN_NOT_CONFIRMED_FULL2638`。本实验即使确认，
也只是新的探索性证据，不能改写正式结论。

## 2. 冻结样本（已由脚本确定并输出 manifest）

- 抽样脚本：`evaluation/select_ade20k_resolution_counterfactual_samples.py`
- 仅取 `source_dataset = "ADE20K Validation Set"`（633 张，全部跨 `G` 两侧）。
- 分层因子（8 层）：原始门档 `{G<=512, G>512}` × task `{Count, Relation}` ×
  原始 integer 臂正确性 `{integer_correct, integer_wrong}`。
- 配额：在 8 层上按 `n_available` 做最大余数比例分配，每层下限 `min(4, n_available)`，
  抽样种子 `20261004`，层内按 `sample_id` 升序、`random.Random("<seed>|<stratum>")`
  确定性洗牌后取前 k。
- 结果：**100 条 selected**（每层 7–14 条；`G<=512` 53 条、`G>512` 47 条；Count 54、
  Relation 46；integer_correct 53、integer_wrong 47），其余 533 条冻结为 reserve。
- 每条记录在 `resolution_counterfactual` 字段下携带 `stratum / selection_rank /
  original_G / original_G_bin / original_integer_correct / selection_seed`。

## 3. 两个 processor pixel budget（预注册）

| 档位 | `min_pixels` | `max_pixels` | 目标 |
|---|---|---|---|
| `low` | 65536（默认） | 524288 = 512×1024 | 强制 `G <= 512` |
| `high` | 525312 = 513×1024 | 16777216（默认最大） | 强制 `G >= 513` |

- `G = full_llm_visual_tokens = (H_resized/32) × (W_resized/32)`（Qwen `smart_resize`，
  factor=32，patch=16，merge=2×2）。
- 已用冻结 full2638 的 633 张 ADE20K 图验证：analytic `smart_resize` 公式与记录的
  `full_llm_visual_tokens` **633/633 完全一致**。
- analytic 预测（**非权威**）：100/100 样本满足 `G_low ∈ [64,504] <= 512 < [520,3588] ∈ G_high`，
  0 剔除、0 替换、0 未解。

## 4. 运行前强制门：processor-only dry run

**必须先执行**（权威模式，只加载 processor，不跑模型）：

```bash
python evaluation/verify_same_image_resolution_gate_separation.py \
  --mode processor \
  --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B
```

- 分离判据：`G_low <= 512` 且 `G_high > 512`。
- 未分离样本按预注册规则**剔除/替换**：从同层 reserve 中按 `selection_rank` 升序取
  下一个尚未使用、且自身两档也分离的候选；该层 reserve 用尽则该样本 `unresolved`，
  dry run **fail closed**，必须 100/100 全部 resolved 才能跑模型。
- freeze 后的最终 manifest 为
  `datasets/2026-10-04-same-image-resolution-counterfactual-ade20k100-separated.jsonl`，
  其 SHA256 写入权威 gate-separation 报告；runner 会强制 `--manifest` 与该 SHA 一致。
- 注意：`Omni3D_Hypersim` 在全量中完全落在 `G>512` 侧，ADE `G>512::Relation::integer_wrong`
  层仅 7 条且已全部入选、**无 reserve**；若该层出现未分离样本将直接 `unresolved` 并
  fail closed（预注册已记录该风险）。

## 5. 两臂与冻结前向

两臂各自在**低档和高档**分别运行，共 4 组输出：

- `corrected_l2_integer`（integer 臂，不使用 B1-trained Wv）
- `prompt_plus_corrected_l2_integer`（Prompt+integer 臂，使用 B1-trained Wv）

固定条件：cutpoint `B1_after_vision_block_0`、`prune_after_layer=0`、`budget K=96`、
`region_grid=3`、`local_candidate_cap=16`、BF16、SDPA、greedy argmax、`do_sample=false`、
`max_new_tokens=16`、checkpoint seed `20261003`（`checkpoint_sha256 1f1d379e…`、
`wv_sha256 126bd842…`、`wv_scale_sha256 93585a3b…`）。

runner（脚手架，未执行）：`evaluation/run_hf_same_image_resolution_counterfactual.py`；
它复用冻结的 B1 前向/续跑/解码逻辑，只向 processor 调用注入 `min_pixels/max_pixels`，
并把 `pixel_budget_label / min_pixels / max_pixels` 写入每条记录。

## 6. 主估计量与判定规则

- 每图效应：`(Prompt_correct − Integer_correct)`，分别在 `high`、`low` 两档计算。
- **主估计量**：`mean_i [ (Prompt−Integer)_high − (Prompt−Integer)_low ]`。
- 主指标：无权重逐样本配对差（单数据源内官方 `cv_bench_accuracy` 未定义，仅作次要参考）。
- 分层配对 bootstrap：按原始层 `stratum` 重采样，seed `20261004`，repeats `5000`，
  CI 95%。
- 判定：
  - 若 CI 下界 **> 0** → `RESOLUTION_DEPENDENT_PROMPT_GAIN_CONFIRMED`；
  - 否则（含 0 或整体 <0）→ `RESOLUTION_DEPENDENT_PROMPT_GAIN_NOT_CONFIRMED`。
- 次要诊断：各档 W→R / R→W / net、各层效应差、parse rate。
- **不得**用 pointing/grounding/ndcg/retained-mass 作主指标；**不得**声明显存/时延/加速。

## 7. 执行顺序（下一步）

```bash
# 1) 权威 processor-only dry run（必须先做；解析出的 100-separated manifest 才会生成）
python evaluation/verify_same_image_resolution_gate_separation.py \
  --mode processor \
  --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B

# 2) 四组臂运行（示例：低档 integer；其余三组同理，输出文件分开）
python evaluation/run_hf_same_image_resolution_counterfactual.py \
  --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B \
  --model-label RoboBrain2.5-4B-BF16 \
  --manifest datasets/2026-10-04-same-image-resolution-counterfactual-ade20k100-separated.jsonl \
  --arm corrected_l2_integer --pixel-budget low \
  --output artifacts/token_selector/2026-10-04-same-image-resolution-counterfactual/corrected_l2_integer_low.jsonl
```

> 本次仅搭建：以上命令**尚未执行**，也没有任何臂输出或 counterfactual 分析结果。
