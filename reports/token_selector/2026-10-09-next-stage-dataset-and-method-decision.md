# 下一阶段数据与方法决策

## 为什么不能继续在 SpatialEval 上调

SpatialEval 本次外部确认已经解封并给出失败结论。官方 Hugging Face 数据卡只提供 test split，
三个模态均为 4635 条；官方 GitHub README 仍将数据生成脚本标为待发布，当前仓库中也没有生成器。
因此，直接在这 4500 条上改变 K、评分权重或锚点会把外部集变成发展集，后续不能再据此声称泛化。

- 官方数据卡：https://huggingface.co/datasets/MilaWang/SpatialEval/blob/main/README.md
- 官方代码：https://github.com/jiayuww/SpatialEval

## 建议采用 VSR Zero-Shot 作为下一阶段数据

VSR（Visual Spatial Reasoning，TACL 2023）是专门评估视觉空间关系的公开数据集。每条样本由
真实图像、描述两个对象空间关系的句子和 True/False 标签组成。固定官方提交的 zero-shot
划分实际为 train 3489、dev 340、test 1222；train/dev 的对象概念重叠为 0，图像仅重叠 5 张。
相比之下，random train/dev 虽有 7680/1097 条，但图像重叠达到 558 张，不适合训练视觉
Selector 后再把 dev 当作独立验证。VSR 的短答案比 SpatialEval 的解释型 64-token 输出更接近
边缘侧结构化决策，也能按空间关系类别定位 Selector 的失效原因。

- 官方仓库：https://github.com/cambridgeltl/visual-spatial-reasoning
- TACL 论文：https://aclanthology.org/2023.tacl-1.37/

限制：VSR 图像来自 COCO，元数据约 3.2 MB，但图像需单独取得。执行前应先审计现有 COCO
缓存能覆盖多少样本，再决定是否下载缺失图片，避免直接下载完整 COCO train2017。

## 建议实验顺序

1. VSR random 与 zero-shot 元数据均已固定；random 因 train/dev 图像重叠 558 张而不作为主划分。
2. 将 zero-shot train/dev 定义为发展域，zero-shot test 在冻结方法、Prompt、预算和门槛前保持未解析。
3. 在 dev 上先运行 Dense、整数 L2 和当前 v2 的 K96/K144/K192 离散预算曲线。统一 edge576、
   greedy、答案只允许 True/False、`max_new_tokens<=4`，同时报告准确率和完整应用时延。
4. 只在发展域设计新的关系保持 Selector：Prompt 语义相关性、规则网格覆盖和成对空间结构共同
   决定分数；预算只能从少量硬件友好的离散档位选择，输出仍为连续、对齐的索引列表。
5. 与固定 K 的 L2、当前 v2、随机选择和 Dense 比较，按关系类别报告准确率，并做配对 bootstrap。
6. 只有候选在 dev 同时满足相对 Dense 的质量门、显著优于同预算 L2、且按实测阶段模型存在净
   时延收益，才预注册并一次性运行 VSR test。

## 与硬件创新的对应关系

新方案不把“裁掉多少 token”单独当创新点，而是联合解决三个问题：

1. 关系保持：选择成对对象及其相对位置所需的 token，而不是只保留单点高响应区域。
2. 离散自适应预算：K 只取少量固定档位，便于 ZCU102 规则化输出和 DMA 对齐。
3. 短结构化决策：用 True/False、坐标或动作码等短输出评估，避免解释型长 Decode 掩盖视觉压缩收益。

如果最终仍需自然语言解释，应将解释生成与实时动作决策分为两条路径；333 ms 约束只施加在动作
决策路径，不能要求 64-token 解释也达到同一实时门槛。

## 当前决策点

元数据与缓存审计已完成：zero-shot dev 的 227 张图中本机已有 5 张，缺失 222 张；train 缺失
2204 张。建议下一步只补齐 dev 的 222 张图片，先运行 Dense/L2/v2 的短输出预算曲线；只有结果
支持继续训练关系保持 Selector 时，才补齐 train，避免不必要的流量和磁盘占用。
