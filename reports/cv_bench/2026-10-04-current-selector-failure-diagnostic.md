# 当前 Selector CV-Bench 失败诊断（K48/K64/K96）

- 生成时间：`2026-10-04T00:28:42.006137+00:00`
- 范围：CPU-only read-only diagnostic of the frozen 2026-10-04 current Selector CV-Bench outputs; software Selector path, no FPGA claim
- 正式样本数：2638
- 正式主指标（唯一）：`score(...)["cv_bench_accuracy"]`（官方 CV-Bench 组合分，`parse_choice`/`answer_letter`/`score`）；禁用主指标：pointing, grounding, ndcg, retained_mass
- 次指标（仅诊断、不得冒充主分）：`unweighted_sample_accuracy` = `score(...)["accuracy"]`（2638 样本简单平均）。

## 1. 总体准确率

| 候选 | 样本 | **CV-Bench 主分**（`cv_bench_accuracy`） | unweighted 样本准确率 | 解析率 | 正式报告一致（对 `candidate.cv_bench_accuracy`） | 相对 dense 保留 |
|---|---:|---:|---:|---:|:---:|---:|
| K48 | 2638 | **0.6534** | 0.6494 | 0.9886 | ✅ | 0.7487 |
| K64 | 2638 | **0.7033** | 0.6990 | 0.9890 | ✅ | 0.8059 |
| K96 | 2638 | **0.7649** | 0.7616 | 0.9864 | ✅ | 0.8764 |

> **两种口径的区别**：**CV-Bench 主分** `cv_bench_accuracy = ((ADE20K + COCO)/2 + Omni3D)/2` 对 2D 两来源与 3D 等权，是唯一正式口径；`unweighted_sample_accuracy` 是 2638 题的简单样本平均，只作次要诊断，**不得写作 CV-Bench 主分**。

## 2. 分任务 / 分来源

> 以下均为 **secondary unweighted sample accuracy diagnostic**（单层子集内的 unweighted 简单平均），不是正式 CV-Bench 主分。

### K48

| task | unweighted_sample_accuracy |
|---|---:|
| Count | 0.4086 |
| Depth | 0.7283 |
| Distance | 0.7033 |
| Relation | 0.8185 |

| source | unweighted_sample_accuracy |
|---|---:|
| ADE20K | 0.5671 |
| COCO | 0.6149 |
| Omni3D | 0.7158 |

### K64

| task | unweighted_sample_accuracy |
|---|---:|
| Count | 0.4695 |
| Depth | 0.7867 |
| Distance | 0.7567 |
| Relation | 0.8431 |

| source | unweighted_sample_accuracy |
|---|---:|
| ADE20K | 0.6066 |
| COCO | 0.6634 |
| Omni3D | 0.7717 |

### K96

| task | unweighted_sample_accuracy |
|---|---:|
| Count | 0.5470 |
| Depth | 0.8350 |
| Distance | 0.8250 |
| Relation | 0.8954 |

| source | unweighted_sample_accuracy |
|---|---:|
| ADE20K | 0.6603 |
| COCO | 0.7391 |
| Omni3D | 0.8300 |

## 3. G 分桶（重点 K96）

> G 桶均值仅为 **secondary unweighted sample accuracy diagnostic**，**不得当作正式 CV-Bench 主分**。

| 候选 | 桶 | samples | unweighted_sample_accuracy | 解析率 |
|---|---|---:|---:|---:|
| K48 | G<=96 | 143 | 0.6993 | 0.9580 |
| K48 | 97-256 | 781 | 0.7490 | 0.9898 |
| K48 | 257-768 | 1144 | 0.6031 | 0.9869 |
| K48 | >768 | 570 | 0.5930 | 0.9982 |
| K48 | G<=K(identity) | 0 | n/a | n/a |
| K64 | G<=96 | 143 | 0.7203 | 0.9650 |
| K64 | 97-256 | 781 | 0.7990 | 0.9885 |
| K64 | 257-768 | 1144 | 0.6608 | 0.9869 |
| K64 | >768 | 570 | 0.6333 | 1.0000 |
| K64 | G<=K(identity) | 61 | 0.6721 | 0.9672 |
| K96 | G<=96 | 143 | 0.7273 | 0.9650 |
| K96 | 97-256 | 781 | 0.8489 | 0.9885 |
| K96 | 257-768 | 1144 | 0.7474 | 0.9808 |
| K96 | >768 | 570 | 0.6789 | 1.0000 |
| K96 | G<=K(identity) | 143 | 0.7273 | 0.9650 |

### K96 任务 × G 桶（secondary unweighted sample accuracy diagnostic）

| G 桶 | task | samples | unweighted_sample_accuracy |
|---|---|---:|---:|
| G<=96 | Count | 81 | 0.5556 |
| G<=96 | Relation | 62 | 0.9516 |
| 97-256 | Count | 148 | 0.5878 |
| 97-256 | Depth | 104 | 0.9519 |
| 97-256 | Distance | 73 | 0.8904 |
| 97-256 | Relation | 456 | 0.9035 |
| 257-768 | Count | 430 | 0.5837 |
| 257-768 | Depth | 296 | 0.8446 |
| 257-768 | Distance | 327 | 0.8410 |
| 257-768 | Relation | 91 | 0.8681 |
| >768 | Count | 129 | 0.3721 |
| >768 | Depth | 200 | 0.7600 |
| >768 | Distance | 200 | 0.7750 |
| >768 | Relation | 41 | 0.7805 |

## 4. G<=K identity bypass 与 dense 一致性

| 候选 | identity 样本 | G<=K 样本 | 集合相等 | dense 文本一致 | 集合全保留 | 全部一致 |
|---|---:|---:|:---:|---:|:---:|:---:|
| K48 | 0 | 0 | ✅ | 0 | 0 | ✅ |
| K64 | 61 | 61 | ✅ | 61 | 61 | ✅ |
| K96 | 143 | 143 | ✅ | 143 | 143 | ✅ |

## 5. 与已有 L2@25% after-layer-5 的对照

- 对照口径：`score(...)["cv_bench_accuracy"]`（当前 Selector 的正式 CV-Bench 主分）
- L2@25% after layer 5：
  - 组合准确率 `0.7955`，dense `0.8727`，保留率 `0.25`
  - anchor grid `3`，位置策略 `preserve original positions`
- K48 − L2@25%：`-0.1421`（K48 CV-Bench 主分 `0.6534`）
- K64 − L2@25%：`-0.0922`（K64 CV-Bench 主分 `0.7033`）
- K96 − L2@25%：`-0.0306`（K96 CV-Bench 主分 `0.7649`）

> **归因限制**：The frozen L2@25% baseline prunes after vision layer 5 (prune_after_layer=5) while the current Selector prunes at boundary B1 (after vision block 0). The cut points differ, so any accuracy difference between the two must NOT be attributed to the ranker/selector alone.

## 6. 输入 SHA256

| 输入 | SHA256 |
|---|---|
| formal_quality_report | `d4ee60fe777cf0b6b075467c72acc1d2aec9ee2eb9cd7537523c5b7fe6344311` |
| manifest | `22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91` |
| dense_jsonl | `3ff3b77ea4bddae96a8231c653bf08db1ce396a9c125fb414466417b1ea69f63` |
| l2_layer5_report | `36ff7f9fc971bf389fee252fc4ab0af97321e3ad3d5c10fb1f2b1017f2cc6f55` |
| official_scorer_module | `58a6bc67399e32b6edac08acbc96babf7c4d5e83173b3cc8eb41a2db64d8a61b` |
| candidate_K48_jsonl | `de5a5761e6c0e7f7718f6a48c5af2ea20d7ffce23235559691cb423d898a95c5` |
| candidate_K64_jsonl | `7d1f48acefd7193dfc31665468de4089ad537beb2f24eefbab04879c554a006e` |
| candidate_K96_jsonl | `b8976a5befdde7ae84cae925d14e37b2f5dd4e4d60b30eb4bdd879b18b87241d` |

## 7. 限制

- Software Selector path only; this is not an FPGA or latency result.
- The comparison against L2@25% uses a different vision cut point (layer 5 vs B1/block 0) and therefore isolates no single component.
- G buckets are observational strata over CV-Bench, not a controlled sweep.
- G-bucket means are secondary unweighted sample accuracy diagnostic only and must not be presented as the formal CV-Bench primary score.
- Forbidden primary metrics (pointing/grounding/ndcg/retained_mass) are never used.

- `all_identity_checks_pass` = True；`formal_scores_reproduced` = True

> 本报告为 CPU-only 只读诊断：未改原始 JSONL、正式 quality report 或 `configs/evaluation_protocol.json`；不含 FPGA/时延/加速声明。
