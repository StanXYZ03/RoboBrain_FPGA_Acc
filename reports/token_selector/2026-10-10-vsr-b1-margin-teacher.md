# VSR B1 答案 Margin 教师缓存审计

## 结论

3483 条 zero-shot train 教师特征全部生成并通过逐文件审计。该结论只说明训练数据完整、梯度链可用，不表示 K96 Selector 已经提升精度。

## 完整性

- 索引：3483 条，唯一 ID 3483 条。
- NPZ schema / 内部 fingerprint：3483/3483 全部通过。
- NPZ 总大小：2.410 GiB。
- 每条 saliency 均有限，并且每个视觉 group 的值都非零。

## 数据与模型诊断

- G 均值 281.0，中位数 300，范围 80–400；G<=96 共 4 条，部署时应走 identity bypass。
- 冻结 Dense 模型正确 True/False 下一分支：2735/3483 = 0.7852。这是单步分支诊断，不是完整生成准确率。
- 稳态前向中位数 70.3 ms，反向中位数 90.4 ms；峰值显存最大 11.02 GiB。

## 数据隔离

训练标签只用于确定教师 True/False 分支。未来部署 Selector 只允许读取 B1 图像特征、原始 Prompt、位置和固定预算；不得读取 label、relation、subject、object 或教师梯度。与 dev 重叠的 5 张图已删除，test 未打开、未评分。

## 下一步

先按图像 ID 对 2244 张训练图做分组 train/validation 划分，再训练固定 K96 的小型关系保持学生。超参数只能由 train 内 validation 选择；VSR dev 仅用于冻结候选后的开发评测。

