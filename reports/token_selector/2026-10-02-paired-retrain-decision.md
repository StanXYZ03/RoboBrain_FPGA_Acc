# Prompt 成对监督重训决策草案

状态：**等待 PI 批准，尚未训练**。

## 为什么现在可以讨论重训

冻结权重归约消融已经证明：去掉 hard max 可以消除逐元素 Prompt 不变量，但不能让当前 `Wt/Wv` 自动恢复正确 Prompt 判别。Train-only teacher 审计进一步证明，同图 grounding/pointing 的监督并非完全相同：159 个训练图像对全部 `JS>0`，Top-K 集合平均有约 28%–29% 差异，但 JS 中位数只有 0.00924，且只有 11.95% 的图像对改变最高分 group。信号存在，但弱且高度不均匀。

因此，下一实验不能对所有图像强加同一固定 margin，也不应同时引入低秩、量化、新 teacher 和多组超参数。最小可证伪问题是：**保持 scorer 与硬件接口不变，仅增加与 teacher 可辨识度成比例的成对 Prompt 判别监督，能否在 val 上恢复 Prompt 条件性？**

## 推荐的唯一实验

固定 scorer：

```text
prompt token embeddings
  -> Wt[128,2560]
  -> 每 token L2 归一化
  -> token 均值并再次归一化
  -> 1x128 prompt query

group_mean[G,1024]
  -> Wv[128,1024]
  -> 每 group L2 归一化
  -> 与 prompt query 做 Gx128 点积
```

这保持 FPGA 侧每个 group 128 MAC、每个 Prompt 只传 128 个 INT8 元素的接口，不再产生 `G x T` token 竞争。

只比较两个训练臂：

1. `A_KL_ONLY_CONTROL`：用同一 scorer 只拟合两个 teacher soft target，判断单纯重训是否足够。
2. `B_KL_PLUS_ADAPTIVE_SWAP_HINGE`：在 KL 上增加成对 swap-hinge。正确 Prompt 的 teacher KL 必须优于交换 Prompt，margin 直接取该图像对 teacher target 的 `JS_nats`。

成对损失为：

```text
L_base = 0.5 * [KL(y_g || p_g) + KL(y_p || p_p)]
D_swap = 0.5 * [KL(y_g || p_p) + KL(y_p || p_g)]
L_pair = ReLU(JS(y_g, y_p) + L_base - D_swap)
L_total = L_base + L_pair
```

所有量均使用自然对数，单位一致。teacher 差异小的图像自动获得小 margin，差异大的图像获得大 margin，不新增人工 margin 或可调 loss weight。

## 为什么暂时不做低秩

原训练有 458,752 个参数且 `best_epoch=0`，低秩很可能有助于泛化与硬件成本；但现在同时改变参数化会无法判断改进来自成对监督还是容量约束。本阶段保留全尺寸 `Wt/Wv`。只有 B 臂通过后，才单独研究 rank 16/32、INT8 和空间 anchor，使每一步都有清晰因果归因。

## 验证纪律

- 只读 train/val NPZ，不读取现有 internal test NPZ。
- 三个固定 seed：20261002、20261003、20261004。
- 以 image 为独立统计单位；same-image 与 split-local cross-image controls 均需通过。
- 两个臂共 8 个 primary hypothesis，统一 Holm-Bonferroni 校正。
- B 必须通过 Prompt 敏感性、shuffled 判别、exact collapse、teacher alignment、跨 seed 泛化和硬件合同六项 gate；A 仅是对照，不能作为“最不差”方案晋级。
- 现有 internal test 已在旧实验中被观察，不能再称为全新最终 holdout；正式论文精度必须使用后来冻结的外部 benchmark 协议。

完整草案位于 `configs/prompt_mean_pairwise_retrain.draft.json`。只有 PI 批准后，才允许将状态改为 `PREREGISTERED`、由 Harness 实现并开始 train/val 运行。
