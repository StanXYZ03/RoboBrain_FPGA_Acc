# 后续任务

更新时间：2026-10-10

- [x] 审计 SpatialEval 官方论文、数据 schema 与 evaluator 缺陷。
- [x] 冻结 VQA 主集 4500 条、Prompt、三臂、主指标、门槛和全部 SHA pin。
- [x] 将 4500 条问题对应图像按内容哈希物化为 1459 个 `/tmp` 缓存文件。
- [x] 完成官方兼容 metric、三臂运行器、汇总器和 35 项 CPU 测试。
- [x] r1 在 93 条未评分 Dense 前缀后发现 checkpoint 直接固定缺口；保留前缀并发布 r2 行政修正版（35 pins）。
- [x] 按 r2 完成 `dense_edge576` 4500 条；唯一 ID、记录哈希与恢复检查通过。
- [x] 完成 `corrected_l2_integer_b1_k96` 4500 条；唯一 ID、记录哈希与恢复检查通过。
- [x] 完成 `v2_edge576_b1_k96` 4500 条；唯一 ID、记录哈希与恢复检查通过。
- [x] 只运行一次 `evaluation/summarize_spatialeval_three_arm.py`，生成 JSON/CSV/中文 Markdown。
- [x] 外部确认判定为 FAIL：v2/Dense=0.744195；v2-L2=-0.004667，95% CI [-0.013778,+0.004444]。
- [x] 固化失败归因：视觉 token 减少 79.53%，但 Decode 占 Dense elapsed 85.57%，当前软件 v2 慢 3.24%。
- [ ] 禁止在 SpatialEval test 上调参；把当前固定 K96 v2 从主方案降级为负结果/消融。
- [ ] 在既有发展数据上设计自适应预算和空间关系保持机制，并与结构化短解码或非自回归动作头联合评估。
- [ ] 新方法冻结后选择另一个未观察公开基准，只做一次外部确认。
- [x] 固定并审计 VSR random/zero-shot 元数据；random train/dev 有 558 张图重叠，不作为主划分。
- [x] VSR zero-shot train/dev 概念重叠为 0、图像重叠 5；test 只核对 hash/行数，未解析 JSON。
- [x] 只下载并校验 zero-shot dev 缺失的 222 张 COCO 图；dev 现为 227/227，未下载完整 COCO。
- [x] 用 zero-shot dev 全 340 条短 True/False 输出完成 Dense、L2 K96/K144/K192、v2 K96/K144；v2 K192 结构性 fail-closed 且 0 记录。
- [x] 固化预算曲线与异构投影：K96 v2/Dense=0.8662、距0.88线4条；1GbE 投影0.972x，假设10GbE 1.162x，均非板测。
- [x] 只补齐 VSR zero-shot train 所需图像，并建立 train-only 的关系保持 K96 训练/验证特征清单；test 继续封存。→ 内部按图划分 train/val/test = **2800/317/366**；同图 donor 覆盖 train **1580 样本/564 图**、val **154 样本/62 图**；test 仍只登记未打开。
- [x] 固定 K96/3x3/整数布局，设计对象对与空间关系保持目标；禁止通过扩大 K 掩盖质量缺口。→ 已按预注册执行有序三槽位有界配对候选 `ordered_three_slot_bounded_partner`（seed 20261012、K96/87+9、vision block 1）：教师排序点估计优于 L2（正确−L2 Spearman/NDCG/K96 = +0.071741542/+0.003151973/+0.084844562），但跨图与同图 Prompt 敏感性三项 95% CI 下界均未 >0，状态 `VSR_ORDERED_PAIR_STUDENT_NOT_ESTABLISHED`；按停止规则停止该候选，不事后调参、不进入量化/HLS/全模型 dev，test 未打开。
- [ ] 回到研究决策：重新判断 B1 原始 embedding 是否提供可学习的关系信号，或更换切点/监督目标；重大方向需与用户讨论后再决定，不自行启动实验。
- [ ] 新候选必须在 dev 同时满足相对 Dense >=0.88、同预算 L2 配对 CI 下界 >=0，以及可辩护的异构净收益后才允许冻结 test。
- [ ] 评估 Jetson Xavier NX 与 ZCU102 的 PCIe/共享内存可行性；1 GbE 全 block0 特征往返不得作为主架构。
- [ ] Jetson Xavier NX 接入后重测真实预处理、视觉、Prefill、Decode 与链路延迟。
- [ ] ZCU102 只在真实净收益合同成立后实施 Selector IP/数据通路联调；当前不得声称板端加速。

## 独立继任实现（`2026-10-09-spatialeval-vqa-external-confirmation.preregistered.json`，CPU-only）

- [x] 新增官方兼容 evaluator（importlib 上游 `evaluation.py`，固定 `model_name=robobrain25`，子串主分 + strict 诊断 + task/qid/macro/micro）。
- [x] 新增 SpatialEval 4500 三臂合同/runner（复用冻结 Dense/L2/v2 生产者，图像物化、标签禁入、append-only resume、warmup、CUDA sync、原始 JSONL）。
- [x] 新增 fail-closed 汇总器（三臂各 4500 同集合；v2/dense macro>=0.90；task::qid 配对 bootstrap seed=20261009/reps=5000 CI 下界>=0；逐任务只报告）。
- [x] 新增继任预注册 config（`PREREGISTERED_NOT_EXECUTED`、`labels_observed=true`、`not_label_blind`、`not_used_for_selector_tuning`、全部 pin 现场复核 12/12）。
- [x] 新增并运行 4 个 CPU 聚焦测试，**64 OK**；`py_compile` 通过。
- [~] 本实现尚未执行任何 GPU/模型权重/预测；阻塞：`/tmp` 两个固定 parquet 当前缺失，无法生成 run manifest。
- [ ] 恢复 parquet 并复核 SHA 后，依次 `python evaluation/spatialeval_vqa_contract.py` → 三臂 `--validate-only` → 只执行一次正式三臂 → 只运行一次汇总器。
