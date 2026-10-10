# VSR 有序三槽位关系保持学生：预注册执行与停止

本报告记录配置 `configs/2026-10-10-vsr-ordered-pair-student.preregistered.json`
（状态 `PREREGISTERED_NOT_EXECUTED`）在内部划分上的执行结果，机器可读结果见同目录
`2026-10-10-vsr-ordered-pair-student.json`，运行日志见
`logs/2026-10-10-vsr-ordered-pair-student.log`。最终状态为
`VSR_ORDERED_PAIR_STUDENT_NOT_ESTABLISHED`，严格门未全部通过。

## 这次做了什么

沿用冻结的 B1 margin 教师，只把上一次失败的「完整 Prompt 均值池化查询」替换为
一个对词序敏感的三槽位有界配对打分器 `ordered_three_slot_bounded_partner`：
最多 16 个原始 Prompt token 经 128 维非对称局部文本编码器得到 subject / object /
relation 三个隐槽位，视觉侧由 B1 分组特征投影到 128 维，取全局 top-4 主体/客体
配对并加入六项相对几何，不构造 G×G 矩阵。切点仍为 vision block 1，预算固定
K96（87 real + 9 context、3×3 anchor、局部候选上限 16），Prompt 最多 16 token，
参数规模 511238。

其余合同与预注册一致：三个种子 20261010 / 20261011 / 20261012，15 epoch，
batch size 8，学习率 0.001，temperature 0.1，seed 20261010 划分，5000 次按图聚类
bootstrap。加性改动只有查询侧这一处，未改动 B1、K96、caption span、划分、
优化器家族、教师或 epoch。

本次训练命令为 `python evaluation/train_vsr_ordered_pair_student.py`，wall clock
**6:53.87**，峰值 RSS **4252412 KiB**，退出状态 0。这是 float 软件实验，任何数字
都不是 ZCU102 板端实测。

提交前仅规范化了 `evaluation/ordered_pair_scorer.py` 文件末尾的额外空行；运行时文件
SHA256 为 `2ae423f3...ae313`，规范化后为 `88ac2e0b...f82a`。Python 源码 AST 与行为
不变，当前配置 pin 对应规范化后的文件；算法、参数和实验输出均未修改。

## 数据划分与同图 donor 覆盖

内部按图划分的 train / val / test 为 **2800 / 317 / 366**。其中同图 donor
（同一 `source_image_sha256` 里存在多条不同 Prompt）覆盖：train **1580** 个样本、
**564** 张图；val **154** 个样本、**62** 张图。test 侧只登记、未打开。

## 胜出种子与验证结果

按预注册的胜出规则（六项按图聚类 95% CI 下界的最大值优先，再比较
correct−L2 NDCG-full，最后比较验证目标），胜出种子为 **20261012**，
其内部验证集（317 个样本 / 225 张图）结果如下。

正确 Prompt 的教师代理指标（三者为 Spearman / NDCG-full / K96 教师质量）：

- 正确 Prompt：**0.136835889 / 0.596526661 / 0.486185966**
  （95% CI：`[0.105284578, 0.167955941]` / `[0.584245518, 0.608530447]` /
  `[0.431629108, 0.540044729]`）
- 修正整数 L2：**0.065094348 / 0.593374688 / 0.401341404**
  （95% CI：`[0.040255656, 0.090449631]` / `[0.582544922, 0.604139081]` /
  `[0.344959977, 0.456819466]`）
- 正确 Prompt − L2：**+0.071741542 / +0.003151973 / +0.084844562**
  （95% CI：`[0.028483419, 0.114673380]` / `[-0.005328824, 0.012266129]` /
  `[0.006942330, 0.160669332]`）

也就是说，在教师排序这一代理目标上，有序三槽位查询相对 L2 的三个点估计都为正；
但 NDCG-full 的差值是 +0.003151973，其 95% CI 跨 0。

## 严格门判定

teacher alignment 是两个点估计门：正确 Prompt − L2 的 Spearman 与 NDCG-full
点估计都 > 0，**两门通过**（`gates.correct_minus_l2_spearman_positive=true`、
`gates.correct_minus_l2_ndcg_full_positive=true`）。

Prompt 敏感性门都要求三项指标按图聚类的 95% CI 下界全部 > 0，两组**均未通过**：

- 正确 Prompt − 跨图错误 Prompt（跨图 donor 对照，317 个样本 / 225 张图）：
  Spearman **+0.000029060**，95% CI `[-0.000768164, +0.000806661]`；
  NDCG-full **-0.000046629**，95% CI `[-0.000147317, +0.000046023]`；
  K96 教师质量 **-0.001538223**，95% CI `[-0.013188071, +0.008986409]`。
  三项 CI 下界分别为负、负、负，未全为正。
- 正确 Prompt − 同图错误 Prompt（同图 donor 对照，154 个样本 / 62 张图）：
  Spearman **+0.000030490**，95% CI `[-0.000317827, +0.000373563]`；
  NDCG-full **+0.000025048**，95% CI `[-0.000032273, +0.000088990]`；
  K96 教师质量 **+0.000028764**，95% CI `[-0.000037337, +0.000130868]`。
  三项 CI 下界分别为负、负、负，同样未全为正。

因此 `gates.cross_image_ci_low_positive_all_three=false`、
`gates.same_image_ci_low_positive_all_three=false`、`gates.pass=false`。预注册的
停止规则随即生效：`internal_test` 保持 **null**，`test_npz_opened` 为 **[]**，
`official_vsr_dev_used` 与 `official_vsr_test_opened` 均为 **false**——VSR 官方
dev 与 test 都没有被使用。

## 口径与不得越界的结论

全部指标都是相对冻结 B1 margin 教师的蒸馏代理，**不是 VSR 任务精度**：同图
教师代理也不等于 grounding/pointing 之类任务准确率。所有数字都是 float 软件
结果，**不是 FPGA 实测**。因此不能据此声称 K96 全模型精度，也不能声称任何
加速或部署收益。本次没有涉及量化、HLS 综合、布线时序或 ZCU102 板端测量。

## 结论与停止

有序三槽位候选在教师排序的两个点估计上优于 L2，但**没有建立起 Prompt 敏感性**：
跨图与同图的正确−错误 Prompt 对照，三项指标的 95% CI 下界都没有稳定大于 0，
同图对照的点估计本身也接近 0。据此按预注册停止该候选：不事后调整学习率、
损失或门槛，不进入量化、HLS 或全模型 dev。

下一步只回到研究决策本身：重新判断 B1 原始 embedding 是否提供了可学习的关系
信号，或者是否需要更换切点/监督目标；重大方向需与用户讨论后再决定，本报告
不启动任何新实验。
