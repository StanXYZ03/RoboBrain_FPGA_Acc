# B1 answer-margin oracle challenge/control pilot24 v2 汇总（decision-only）

- 日期：2026-10-05
- 状态：**PILOT_STOP**
- 设计：`2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24`

## 1. 口径

- **outcome/label leakage**：样本按 dense/control 正确性分层，非代表性、不可部署；
  禁止把 24 样本结果当作官方 CV-Bench 精度。
- **ground-truth answer leakage**：oracle 用官方正确答案分叉 token，只是可证伪上界。
- 本报告**不计算置信区间**、不使用显著性/『精度提升已确认』措辞，dense 仅资格参照。

## 2. 预注册 GO/STOP 决策

| 条件 | 要求 | 实际 | 通过 |
| --- | --- | --- | --- |
| challenge 修复 | ≥ 8/12 | 10 | ✅ |
| control 回归 | ≤ 1/12 | 3 | ❌ |
| oracle parse | 24/24 | 24 | ✅ |
| 精确 K96 布局 | 24/24 | 24 | ✅ |
| 哈希/ID/分层标签 | 全部通过 | ✅ | ✅ |

- 结论：**PILOT_STOP**（满足全部门才 GO）。
- 该门只是 outcome-enriched 诊断的筛选决策，**不是精度、部署或加速结论**。

## 3. 各单元修复/回归

| 单元 | 样本 | challenge | control | 修复 | 回归 | W→R | R→W |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ADE20K x Count | 4 | 2 | 2 | 2 | 1 | 2 | 1 |
| ADE20K x Relation | 4 | 2 | 2 | 1 | 1 | 1 | 1 |
| COCO x Count | 4 | 2 | 2 | 1 | 0 | 1 | 0 |
| COCO x Relation | 4 | 2 | 2 | 2 | 0 | 2 | 0 |
| Omni3D x Depth | 4 | 2 | 2 | 2 | 0 | 2 | 0 |
| Omni3D x Distance | 4 | 2 | 2 | 2 | 1 | 2 | 1 |

## 4. 逐样本迁移

| sample_id | cell | stratum | truth | control | oracle | 迁移 | margin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 229 | ADE20K x Count | challenge | A | D | A | W->R | 4.2500 |
| 290 | ADE20K x Count | challenge | E | B | E | W->R | 0.3750 |
| 305 | ADE20K x Count | control | C | C | C | unchanged_correct | 2.2500 |
| 334 | ADE20K x Count | control | C | C | E | R->W | 1.2500 |
| 425 | ADE20K x Relation | challenge | A | B | B | unchanged_wrong | 0.2500 |
| 519 | ADE20K x Relation | challenge | A | B | A | W->R | 2.5000 |
| 439 | ADE20K x Relation | control | A | A | A | unchanged_correct | 2.1250 |
| 471 | ADE20K x Relation | control | A | A | B | R->W | 0.5000 |
| 748 | COCO x Count | challenge | B | D | B | W->R | 0.5000 |
| 969 | COCO x Count | challenge | C | B | B | unchanged_wrong | 7.5000 |
| 636 | COCO x Count | control | B | B | B | unchanged_correct | 5.8750 |
| 719 | COCO x Count | control | C | C | C | unchanged_correct | 6.5000 |
| 1221 | COCO x Relation | challenge | A | B | A | W->R | 0.2500 |
| 1406 | COCO x Relation | challenge | B | A | B | W->R | 3.5000 |
| 1234 | COCO x Relation | control | B | B | B | unchanged_correct | 5.6250 |
| 1328 | COCO x Relation | control | A | A | A | unchanged_correct | 3.7500 |
| 1701 | Omni3D x Depth | challenge | A | B | A | W->R | 3.6250 |
| 1814 | Omni3D x Depth | challenge | B | A | B | W->R | 5.3750 |
| 1908 | Omni3D x Depth | control | A | A | A | unchanged_correct | 6.5000 |
| 1932 | Omni3D x Depth | control | A | A | A | unchanged_correct | 3.5000 |
| 2108 | Omni3D x Distance | challenge | A | B | A | W->R | 3.2500 |
| 2178 | Omni3D x Distance | challenge | B | A | B | W->R | 4.7500 |
| 2518 | Omni3D x Distance | control | A | A | B | R->W | 3.6250 |
| 2566 | Omni3D x Distance | control | A | A | A | unchanged_correct | 4.5000 |

## 5. 契约与输入 SHA256

- manifest：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/datasets/2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24.jsonl` → `0bbe413d55750282a6d7f19650c13886d0e3355f0a82762afd7b5f968ba2a7ca`
- control：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/artifacts/token_selector/2026-10-04-b1-corrected-l2-prompt-ablation-full2638/corrected_l2_integer_k96_full2638.jsonl` → `d0e302f6b91279374ddd507db4d300ca0c29e16f522db57f47c8de059f103a5f`
- oracle：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/artifacts/token_selector/2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24/oracle_answer_margin_b1_saliency_k96.jsonl` → `13be51fa120b4f62ef665d3cdebef0b5af96b748a8b5596e76a98d98e1273862`
- dense（仅资格参照，不作第三臂）：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/evaluation/results/cv_bench_l2_0875_full2638_20260916/shallow_r1000.jsonl`

禁止据此报告声称：deployable selector、paper accuracy、official CV-Bench accuracy from 24 enriched samples、CV-Bench accuracy improvement confirmed、FPGA speedup or latency、representative sampling、using the 24 enriched samples for training or threshold tuning。
