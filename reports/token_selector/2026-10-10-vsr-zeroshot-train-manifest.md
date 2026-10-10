# VSR Zero-Shot Train 防泄漏清单

- 官方 train：3489 条 / 2249 张图。
- 与 dev 重叠：5 张图，对应删除 6 条 train 记录。
- 最终训练域：3483 条 / 2244 张图。
- 标签：False=1685，True=1798。
- 清单 SHA256：`08ae6b609d2f1a1431d7e5d6dec1bf1720859e17159b13bd173a8502dc6bf4f4`。

训练标签只允许构造教师 margin/蒸馏目标；正式 Selector 输入不得包含标签、relation、subject 或 object。zero-shot test 未打开。
