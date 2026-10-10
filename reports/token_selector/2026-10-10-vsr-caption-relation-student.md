# VSR Caption-only 关系学生实验

日期：2026-10-10

## 目的

验证此前 Prompt 条件化失败是否主要由固定指令文本稀释造成。实验只把输入查询从完整
Prompt 改为图片描述主体的 token span，评分结构、数据划分、三组随机种子、优化参数、
教师目标和晋级门槛全部保持不变。

## 数据与边界

- 数据：VSR zero-shot train 的内部按图分组划分，train/val/test 为 2800/317/366 条。
- Caption token 数：最少 6、最多 15、平均 8.45；完整 Prompt 平均约 30.45 token。
- 教师：RoboBrain2.5-4B 在视觉第 0 块之后的 True/False margin 梯度显著性。
- 模型：`Wt[128,2560]`、`Wv[128,1024]`，caption token 投影后逐行归一化、求均值并
  再归一化为一个 128 维查询；每个视觉分组与该查询做点积。
- 未使用 VSR dev，未打开 VSR test，标签和关系标注不进入 Selector 推理输入。

## 结果

胜出种子为 20261010，但状态为 `CAPTION_RELATION_STUDENT_NOT_ESTABLISHED`。

| validation 指标 | 正确 Prompt | 跨图错误 Prompt | 正确减错误 | 正确减 L2 |
| --- | ---: | ---: | ---: | ---: |
| Spearman | 0.207252 | 0.206942 | +0.000310 | +0.147662 |
| NDCG-full | 0.624834 | 0.624888 | -0.000054 | +0.038726 |
| K96 教师质量保留量 | 0.681487 | 0.667412 | +0.014076 | +0.297264 |

按源图聚类 bootstrap 后，正确减错误 Prompt 的 95% 区间为：

- Spearman：`[-0.001720, +0.005144]`
- NDCG-full：`[-0.002156, +0.002727]`
- K96 教师质量保留量：`[-0.003601, +0.035271]`

三项区间都跨过 0，因此 Prompt 敏感性门失败。内部 test 的三项区间也全部跨 0；这只
用于一次冻结诊断，不用于回选模型。

caption-only 还弱于完整 Prompt 均值模型：正确 Prompt 的 Spearman 从 0.262761 降到
0.207252，NDCG 从 0.635606 降到 0.624834，K96 教师质量保留量从 0.759646 降到
0.681487。由此不能再把主要失败原因归结为固定指令 token 稀释。

## 结论

单个均值查询能够学习一般图像显著性，并稳定优于纯 L2，但无法稳定表达 VSR 中由词序、
实体对和空间关系共同决定的条件化选择。继续修改 caption span、学习率或门槛会变成在
同一内部划分上的事后调参，因此停止该模型族，不进入 VSR dev 全模型推理。

下一候选必须显式保留关系结构，同时维持固定 K96、规则化输出和 FPGA 可执行工作量。
在实现前先独立审查以下选择：少量实体/关系槽查询、受限 token-word late interaction，
或将 Selector 切点后移以获得关系化视觉特征。软件估计、A6000 完整模型结果和 ZCU102
板端结果必须继续分开报告。

## 产物

- 预注册：`configs/2026-10-10-vsr-caption-relation-student.preregistered.json`
- 机器报告：`reports/token_selector/2026-10-10-vsr-caption-relation-student.json`
- 三组检查点：`artifacts/token_selector/2026-10-10-vsr-caption-relation-student/`
- 报告 SHA256：`9ae8d0765fbc7b5ce5de4877fd39e4b31affcfbc53fd9b1cd6751bb7883e53d8`
- 运行时间：5 分 56 秒；最大主机常驻内存约 5.95 GiB；观察到 GPU 显存约 5.3 GiB。
