# B1 corrected-L2 / Prompt 消融（full2638 正式确认）

- 生成时间：`2026-10-04T10:17:02.726567+00:00`
- 范围：**formal full2638 CV-Bench conclusion for the frozen corrected-L2 / Prompt protocol**
- 升格自：2026-10-04-b1-corrected-l2-prompt-ablation-screening100 (CORRECTED_PROMPT_SELECTOR_SCREENING_PASS; not tuned)
> Formal full2638 CV-Bench evaluation of the frozen corrected-L2 / Prompt protocol. The Selector path is a software bridge only; no FPGA, latency or acceleration claim is made.

- 正式主指标（唯一）：`score(...)["cv_bench_accuracy"]` = `((ADE20K + COCO)/2 + Omni3D)/2`；`unweighted_sample_accuracy` 仅为次要诊断，**不得写作 CV-Bench 主分**，也不参与决策。
- 禁用主指标：pointing, grounding, ndcg, retained_mass

## 1. 固定配置

- 样本：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/datasets/cv_bench_full2638.jsonl`（2638）；K=96；切点：B1 = after vision block 0；region_grid=3；local_cap=16
- dtype=bfloat16；attention=sdpa；decoding=greedy_argmax；bootstrap seed=20261004 / repeats=5000；quantization CI 下界阈值=-0.02；prompt CI 下界阈值=0.0

## 2. 输入合同校验

- full manifest 样本数：2638
- 三臂样本数：{'corrected_l2_float': 2638, 'corrected_l2_integer': 2638, 'prompt_plus_corrected_l2_integer': 2638}
- 每臂恰好 2638 个唯一 ID 且与 manifest 一致：✅
- 三臂 budget=96 / region_grid=3 / local_cap=16：✅
- 非 bypass 臂输出 K=96；bypass 输出 G：✅
- record_sha256 / manifest_sha256 / checkpoint 校验：✅
- 三臂输入 grid 一致（(tuple(image_grid_thw), full_patch_tokens, full_llm_visual_tokens)）：✅

## 3. 五臂正式主分

| 臂 | 角色 | 样本 | **CV-Bench 主分** | unweighted（次要） |
|---|---|---:|---:|---:|
| corrected_l2_float | primary | 2638 | **0.766117** | 0.763078 |
| corrected_l2_integer | primary | 2638 | **0.766852** | 0.763457 |
| prompt_plus_corrected_l2_integer | primary | 2638 | **0.777362** | 0.773313 |
| dense | descriptive | 2638 | **0.872730** | 0.868082 |
| old_current_integer | descriptive | 2638 | **0.764869** | 0.761562 |

## 4. 配对分层 bootstrap（95% CI，主指标）

| 对比 | 决策相关 | 点估计 | 95% CI |
|---|---:|---:|---:|
| corrected_l2_integer − corrected_l2_float | 是 | +0.000735 | [-0.004192, +0.005833] |
| prompt_plus_corrected_l2_integer − corrected_l2_integer | 是 | +0.010510 | [-0.005430, +0.025699] |
| dense − corrected_l2_float | 否（descriptive） | +0.106613 | [+0.092894, +0.120699] |
| old_current_integer − corrected_l2_float | 否（descriptive） | -0.001247 | [-0.017502, +0.014129] |

## 5. 预注册决策

- 结论：**PROMPT_GAIN_NOT_CONFIRMED_FULL2638**
- 依据：quantization gate passes (CI lower -0.004192 >= -0.02) but prompt_minus_integer_l2 CI lower -0.005430 <= 0.0
- integer_minus_float：[-0.004192, +0.005833]，下界 -0.004192；量化门 ✅ 通过（阈值 >= -0.02）
- prompt_minus_integer_l2：[-0.005430, +0.025699]，下界 -0.005430；Prompt 增益 ❌ 未确认（阈值 > 0.0）
- dense 与旧 current-integer 为 descriptive reference，**不得覆盖**上述决策。

## 6. 输入 SHA256

| 输入 | SHA256 |
|---|---|
| full_manifest | `22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91` |
| dense_full_jsonl | `3ff3b77ea4bddae96a8231c653bf08db1ce396a9c125fb414466417b1ea69f63` |
| old_current_integer_full_jsonl | `b8976a5befdde7ae84cae925d14e37b2f5dd4e4d60b30eb4bdd879b18b87241d` |
| corrected_l2_float_jsonl | `0dff41fb39dde4893c8a87a60b44309fa029fe85360b71438ee183ffd708ff76` |
| corrected_l2_integer_jsonl | `d0e302f6b91279374ddd507db4d300ca0c29e16f522db57f47c8de059f103a5f` |
| prompt_plus_corrected_l2_integer_jsonl | `6d3b97e8ce768066a99c9b9f517ddf602480fa75d57ce2e36b54e26baabc90fa` |
| preregistered_config | `6251a31cdb1b0e2cd176a0076337d0d1dfa76ca49b3af57fa046438738d7c2d8` |

## 7. 限制

- Software Selector bridge only; no FPGA, latency or acceleration claim.
- Dense and the old current-integer selector are descriptive references only and never drive the decision.
- The decision uses score(...)["cv_bench_accuracy"] exclusively, via paired stratified bootstrap CI lower bounds.
- Forbidden primary metrics (pointing/grounding/ndcg/retained_mass) are never reported.

> CPU-only 只读汇总：未改原始 JSONL、正式 quality report 或 `configs/evaluation_protocol.json`；无 FPGA/时延/加速声明。
