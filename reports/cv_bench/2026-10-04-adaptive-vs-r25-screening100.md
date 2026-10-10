# Adaptive (A) vs fixed-K96 (B) vs historical r25 (C) — screening100 严格诊断

- 生成时间：2026-10-04T13:03:28.798144+08:00
- 范围：fixed 100-sample CV-Bench screening; CPU-only read-only diagnostic; NOT a formal full2638 conclusion
- 唯一主指标：`score(...)["cv_bench_accuracy"]`；unweighted = `score(...)["accuracy"]`（**仅次要**）
- 三臂：A=adaptive layer-5 budget；B=fixed K96；C=historical shallow r0250；同一 100 样本、同一 task、`prune_after_layer=5`。

## 1. 官方分数（`score(...)["cv_bench_accuracy"]`）

| 臂 | cv_bench_accuracy | A−C | A−B | unweighted(次要) | 2D | 3D |
| --- | --- | --- | --- | --- | --- | --- |
| A_adaptive | 0.762500 | -0.033333 | 0.016667 | 0.770000 | 0.725000 | 0.800000 |
| B_fixed_k96 | 0.745833 | -0.050000 | 0.000000 | 0.760000 | 0.675000 | 0.816667 |
| C_r25 | 0.795833 | 0.000000 | 0.050000 | 0.810000 | 0.725000 | 0.866667 |

- A 相对 C 的官方分差 **-0.033333**；A 相对 B 的官方分差 **0.016667**。

## 2. 每样本 G / K / retention（汇总）

| 臂 | G 范围 | K 范围 | K 均值 | retention 均值 | identity 数 |
| --- | --- | --- | --- | --- | --- |
| A_adaptive | 64–3588 | 64–912 | 200.480000 | 0.356865 | 2 |
| B_fixed_k96 | 64–3588 | 64–96 | 95.360000 | 0.277290 | 2 |
| C_r25 | 64–3588 | 16–897 | 181.110000 | 0.250253 | 0 |

- A−C 的 K 差：min=0、max=69、mean=19.370000；A 的 K 在所有样本上 >= C：**True**。

## 3. 正确性迁移与文本变化

| 对比 | 都对 | 仅参考对 | 仅处理对 | 都错 | 文本变化 | 答案变化 |
| --- | --- | --- | --- | --- | --- | --- |
| A_vs_C | 71 | 10 | 6 | 13 | 18 | 18 |
| A_vs_B | 69 | 7 | 8 | 16 | 15 | 15 |
| B_vs_C | 68 | 13 | 8 | 11 | 24 | 24 |

## 4. 选择集合关系

| 对比 | 平均 Jaccard | 平均交集 | 平均并集 | A⊇other | other⊇A | 相等 |
| --- | --- | --- | --- | --- | --- | --- |
| A_vs_C | 0.539596 | 140.100000 | 241.490000 | 9 | 0 | 0 |
| A_vs_B | 0.685344 | 95.360000 | 200.480000 | 100 | 41 | 41 |

> A_vs_C 中 left=A、right=C；A_vs_B 中 left=A、right=B。

## 5. 分 task 统计

| task | n | A(unweighted) | B(unweighted) | C(unweighted) | A−C | 平均 A−C K | 平均 Jaccard | A⊇C | C⊇A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Count | 20 | 0.550000 | 0.450000 | 0.550000 | 0.000000 | 27.150000 | 0.529250 | 2 | 0 |
| Depth | 30 | 0.933333 | 0.866667 | 0.933333 | 0.000000 | 8.900000 | 0.604034 | 0 | 0 |
| Distance | 30 | 0.666667 | 0.766667 | 0.800000 | -0.133333 | 8.400000 | 0.554048 | 0 | 0 |
| Relation | 20 | 0.900000 | 0.900000 | 0.900000 | 0.000000 | 43.750000 | 0.431606 | 7 | 0 |

## 6. 按 G 桶统计（unweighted 次要）

| G 桶 | n | A | B | C | A−C | 平均 A−C K | A⊇C | C⊇A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G<=96 | 2 | 0.500000 | 0.500000 | 0.500000 | 0.000000 | 48.000000 | 2 | 0 |
| 97-256 | 27 | 0.851852 | 0.851852 | 0.777778 | 0.074074 | 48.444444 | 6 | 0 |
| 257-768 | 46 | 0.760870 | 0.760870 | 0.869565 | -0.108696 | 9.956522 | 1 | 0 |
| G>768 | 25 | 0.720000 | 0.680000 | 0.760000 | -0.040000 | 3.000000 | 0 | 0 |

## 7. 按 K 差（A−C）桶统计（unweighted 次要）

| K 差桶 | n | A | B | C | A−C | 平均 Jaccard | A⊇C | C⊇A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 23 | 0.739130 | 0.695652 | 0.826087 | -0.086957 | 0.567886 | 0 | 0 |
| 1-15 | 37 | 0.756757 | 0.756757 | 0.837838 | -0.081081 | 0.608981 | 0 | 0 |
| 16-31 | 11 | 0.727273 | 0.727273 | 0.818182 | -0.090909 | 0.520955 | 1 | 0 |
| 32+ | 29 | 0.827586 | 0.827586 | 0.758621 | 0.068966 | 0.435703 | 8 | 0 |

## 8. 严格归因判断

- A=**0.762500**、C=**0.795833**、B=**0.745833**。
- A−C 官方差：**-0.033333**；A−B 官方差：**0.016667**。
- A 的 K>=C 在所有样本上成立：**True**（min A−C K=0，max=69）。
- 选择集合是否为嵌套：**False**（A⊇C=9/100，C⊇A=0/100）。
- **结论：NOT_DIRECT_EVIDENCE_OF_TOKEN_INSUFFICIENCY** — 不是 token 数量不足的直接证据；A 的 K 在每个样本上都 >= C，但官方分更低，且 A/C 选择集合大多非嵌套（A 超集 C 仅少量样本、C 超集 A 为 0），因此主要可观测差异是选择集合/实现路径，而不是预算大小。
- 免责：仅凭本筛选集上的观测不能声称因果；unweighted/子组数字均为 secondary 诊断。

## 9. 最小因果实验（同一实现、同一预算、仅替换索引集合；本次未执行、无 GPU）

- **E1_forced_index_set_A_vs_C**：同一实现（evaluation/hardware_regular_layer5_unbounded_l2.py 的 layer-5 cut=5、blocks 0..5 → compact hidden → DeepStack index 5 → blocks 6..23 路径）、同一预算、仅替换选中的 selected_group_indices；不改变 score_mode/region_grid/BF16/SDPA/greedy。 对每个样本固定预算 K=K_C(K_C<=K_A)，臂 X 强制使用 C 的冻结索引集合，臂 Y 用同一选择算法以预算 K_C 重新选出索引集合；两臂仅索引集合不同。
  - 实现=evaluation/hardware_regular_layer5_unbounded_l2.py；预算=per-sample K_C (fixed, identical in both arms)；唯一变量=selected_group_indices；已执行=False；需要模型前向=True。
  - runner 变更：需要给 runner 增加只读 --forced-selected-indices JSONL 入口（不改既有默认路径）。

## 10. 输入 SHA256（只读校验）

| 输入 | SHA256 |
| --- | --- |
| screening_manifest | `31bf313286c81fa50911ca3822118184ac03ca74df2cf2c260250cab2768b50c` |
| adaptive_arm | `8f9442dd0f533e8a475fd51784682b729a64decb31f3166a7f32d72b4405d3e3` |
| fixed_k96_arm | `fd22c5a4b1941a8ad0a2c47bf15f0abcbeaa56553a9f9a790deb6e14054df94c` |
| r25_arm | `ce019edd1ee35d11c83a3ff7c11bd9db46539cf56745aadcf0780f15abdbf7d0` |
| adaptive_prereg | `55a39d9f1b2f0df264391e6e8c99198d8c4086221116bbbd2509d3e00e52025f` |
| unbounded_prereg | `33d46aa0fa5779bf586fa9262b656bf72caf48eb45f958028cd1ca70e6cff020` |

## 11. 限制

- screening100 only; not the formal 2638-sample benchmark
- unweighted and subgroup accuracies are secondary diagnostics
- no GPU/model/board execution in this analysis
- A/C also differ in implementation path, not only in selected indices
