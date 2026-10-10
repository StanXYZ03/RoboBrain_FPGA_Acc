# B1 answer-margin oracle challenge/control pilot24 v2 预注册（未执行 GPU）

- 日期：2026-10-05
- 状态：**PREREGISTERED_NOT_EXECUTED**
- 设计：`2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24`
- 范围：Outcome-enriched challenge/control diagnostic for the B1 answer-margin oracle; CPU-only pre-registration, GPU not executed.

## 1. 结论口径（必须保留）

- **outcome/label leakage = true**：样本按 dense/control 正确性分层，是结果富集诊断，
  **非代表性、不可部署**，禁止用于官方绝对精度主张。
- **ground-truth answer leakage = true**：oracle 用官方正确答案分叉 token 计算
  B1 `|grad×activation|` group saliency，只是可证伪上界。
- 24 样本**不是论文精度**；本轮 **未运行 GPU**、未加载模型权重、未训练、
  未生成任何 oracle 输出，因此不得使用 CI 或『精度提升已确认』措辞。
- `gpu_executed = false`、`model_was_run = false`、`gpu_runs = 0`。

## 2. 固定 24 样本（六单元各 4：2 challenge + 2 control）

- manifest：`datasets/2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24.jsonl`（SHA256 `0bbe413d55750282a6d7f19650c13886d0e3355f0a82762afd7b5f968ba2a7ca`，24 条）。
- 样本 ID：`['229', '290', '334', '305', '519', '425', '439', '471', '748', '969', '719', '636', '1406', '1221', '1234', '1328', '1701', '1814', '1908', '1932', '2108', '2178', '2566', '2518']`。
- 各单元数量：`{'ADE20K x Count': 4, 'ADE20K x Relation': 4, 'COCO x Count': 4, 'COCO x Relation': 4, 'Omni3D x Depth': 4, 'Omni3D x Distance': 4}`。
- 结果分层数量：`{'challenge': 12, 'control': 12}`。

## 3. 选样规则（明确使用 dense/control 正确性 = outcome leakage）

- 只用 `datasets/cv_bench_full2638.jsonl` 原始元数据 + 冻结 dense BF16 与
  corrected integer-L2 B1/K96 正确性 + 真实本地 tokenizer 标签分叉合同 +
  权威 processor `G>96`。
- 每单元 challenge=dense 对且 corrected L2 错；control=dense 对且 corrected L2 对。
- 按 `sha256(str(sample_id))` 升序取每层前 2；Omni3D 8 条要求 Hypersim/SUNRGBD/
  nuScenes 各 ≥2，否则用最小化有序 SHA-rank 元组的确定性约束选样。
- 排除：`sample_id=1`、screening100 全部 ID、同图反事实 100 全部 ID、既有随机 v1
  pilot24 全部 ID。排除计数：`223`。

## 4. 固定运行合同

| 项 | 值 |
| --- | --- |
| 模型 | `/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B`，BF16/SDPA |
| cutpoint | `B1_after_vision_block_0`（prune_after_layer=0） |
| K / region_grid / local_cap | 96 / 3 / 16 |
| 解码 | greedy，max_new_tokens=16 |
| 期望布局 | output=96，real=87，context=9 |
| runner | `evaluation/run_b1_answer_margin_oracle_pilot.py`（复用，未改） |

- saliency 定义：per merged group g: sum over its 4 patches j and feature dim d of abs(grad[g*4+j, d] * activation[g*4+j, d]), from the full (uncut) B1 visual sequence and the official correct-answer branch token
- score 注入：existing integer_scorer hook of forward_hardware_regular_b1_vision -> combined_scores of select_hardware_regular_groups (layout unchanged)

## 5. 汇总 GO/STOP 门（仅此富集诊断）

- challenge 修复 ≥ 8/12；
- control 回归 ≤ 1/12；
- oracle parse = 24/24；
- 24/24 精确 K96 布局合同通过；
- 满足 → `GO`，否则 `STOP`；不使用置信区间。

## 6. 输入与脚本 SHA256 pin

- `full_manifest`: `datasets/cv_bench_full2638.jsonl` → `22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91`
- `screening100_manifest`: `datasets/cv_bench_screening100.jsonl` → `31bf313286c81fa50911ca3822118184ac03ca74df2cf2c260250cab2768b50c`
- `counterfactual100_manifest`: `datasets/2026-10-04-same-image-resolution-counterfactual-ade20k100.jsonl` → `c53c7b7dd733b11f55edc85ba65745d72951d31f456c7403ec21e91b0031d4c5`
- `random_v1_pilot24_manifest`: `datasets/2026-10-05-b1-answer-margin-oracle-pilot24.jsonl` → `9c8d15a65ef429eb1315ae1b73a80246de0ff2005373370fc184f0de1a2813c9`
- `pilot_manifest`: `datasets/2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24.jsonl` → `0bbe413d55750282a6d7f19650c13886d0e3355f0a82762afd7b5f968ba2a7ca`
- `shared_helpers`: `evaluation/b1_answer_margin_oracle.py` → `58f2843ad329f723a15388f2a95422b4a55572b769ca62100df585a5bb28c752`
- `challenge_control_helpers`: `evaluation/b1_answer_margin_oracle_challenge_control.py` → `3b45059a5e1090dd3c57f738ab838a100417b2660411167fde69c2d73c517fa9`
- `selection_script`: `evaluation/select_b1_answer_margin_oracle_challenge_control_pilot24.py` → `5630dd5cf4278fe026a709da68e609e32b52b3d85125d195906ad14013f0b5ce`
- `gpu_runner`: `evaluation/run_b1_answer_margin_oracle_pilot.py` → `972b504afa41027b1c07588e2cd5cd26f310e7d7ccc3e5d395c82dfb117fb6c2`
- `summarizer`: `evaluation/summarize_b1_answer_margin_oracle_challenge_control_pilot.py` → `0c4b5c86b9330834de59ee95cd41455b833d65e9db33bd7434af7b70c84b96fd`
- `helpers_test`: `evaluation/tests/test_b1_answer_margin_oracle_challenge_control.py` → `8099f924958730572b3e0835e64cc9678a0ad62a02db71c7f7a1e971a9c97216`
- `selection_test`: `evaluation/tests/test_select_b1_answer_margin_oracle_challenge_control_pilot24.py` → `e6fd3ad001753dab820b6be292262d5acba91135e8944a88450e5e8ecb65ae45`
- `runner_test`: `evaluation/tests/test_run_b1_answer_margin_oracle_challenge_control_pilot.py` → `5f6a1a0f428285ea015e7907f3a87ee6ead24fcd4e08f362bf54cc200ccc0592`
- `summarizer_test`: `evaluation/tests/test_summarize_b1_answer_margin_oracle_challenge_control_pilot.py` → `f9749857b3c0a5882f50ceaa05524f89a136f72e4ce76019d31160b88414a818`

## 7. 汇总器输入

- dense（仅资格参照）：`evaluation/results/cv_bench_l2_0875_full2638_20260916/shallow_r1000.jsonl` → `3ff3b77ea4bddae96a8231c653bf08db1ce396a9c125fb414466417b1ea69f63`
- control：`artifacts/token_selector/2026-10-04-b1-corrected-l2-prompt-ablation-full2638/corrected_l2_integer_k96_full2638.jsonl` → `d0e302f6b91279374ddd507db4d300ca0c29e16f522db57f47c8de059f103a5f`
- oracle（待 GPU 生成）：`artifacts/token_selector/2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24/oracle_answer_margin_b1_saliency_k96.jsonl`

## 8. 下一条命令（尚未执行）

```
python evaluation/run_b1_answer_margin_oracle_pilot.py --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --model-label RoboBrain2.5-4B-BF16 --manifest datasets/2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24.jsonl --config configs/2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24.preregistered.json --output artifacts/token_selector/2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24/oracle_answer_margin_b1_saliency_k96.jsonl
```

禁止据此报告声称：deployable selector、paper accuracy、official CV-Bench accuracy from 24 enriched samples、CV-Bench accuracy improvement confirmed、FPGA speedup or latency、representative sampling、using the 24 enriched samples for training or threshold tuning。
