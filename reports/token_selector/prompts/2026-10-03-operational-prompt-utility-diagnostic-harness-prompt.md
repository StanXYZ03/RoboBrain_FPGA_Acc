# Frozen harness prompt — Operational Prompt Utility Diagnostic

This file is written **before** any contrastive val CSV/JSON parsing, any val NPZ
scoring and any best-checkpoint scoring for this diagnostic.  It is the verbatim
execution contract supplied by the operator.

你是执行工程师。执行一次零训练、预注册的“部署态 Prompt 效用诊断”，不要提出计划，不输出思考。项目根目录和 Python 同前。

目的：判定上一项 contrastive 学生虽然 Spearman gate 未过，但在真实 fixed-budget selector 输出上是否具有可用的 Prompt 条件化效用。不得改变上一实验结论，只能新增探索性诊断。

合同：
1. 在读取 contrastive val CSV/JSON、val NPZ、best checkpoint 前，先保存本提示到 reports/token_selector/prompts/2026-10-03-operational-prompt-utility-diagnostic-harness-prompt.md，并创建 configs/operational_prompt_utility_diagnostic.preregistered.json 与对应 preregistration JSON，固定输入SHA、统计与门槛。
2. 禁止 internal test NPZ 解析/哈希/打开；禁止原图、4B、GPU、联网、训练、修改既有代码/报告。新增 evaluation/operational_prompt_utility_diagnostic.py 与测试。
3. 使用 contrastive 实验 A/B 两臂三 seed best checkpoint、同一冻结76 val records/38 images。复用 val_statistics 所需 scorer、空间target、combine_scores(lambda=.5)、select_real_groups 3x3 anchor、selected-set NDCG。
4. 对每条记录/seed/K=48,64,96分别用：
 correct prompt selector；
 same-image-other-task wrong prompt selector；
 对两者均以相同 L2 融合和anchor规则选择。
 输出 retained spatial-target mass、selected-set NDCG、correct-vs-wrong selected Jaccard/overlap。
5. 每图先对 grounding/pointing记录按目标定义汇总，再跨3 seed取中位数；独立图像 bootstrap=2000、sign-flip=10000、dz。primary Holm family恰好2项且仅K48 retained mass：
 P1 B 的 correct selector - wrong selector；
 P2 (B correct-wrong gap) - (A correct-wrong gap)。
 二者均需 mean>0、CI lower>0、dz>=0.2、Holm<0.05、>=2/3 seed positive。
6. 辅助门槛（不进入Holm）：
 S1 B grounding与pointing的K48 correct-wrong mass mean均>0；
 S2 B K48 correct-wrong selected-set NDCG mean>0；
 S3 B K48 correct-vs-wrong selection Jaccard mean<0.95；
 S4 上一实验G1非劣、G4部署效用、G5无collapse、G7硬件合同均已通过且输入SHA一致。
 仅 P1/P2+S1-S4全过，结论 OPERATIONAL_PROMPT_UTILITY_SUPPORTED；否则 OPERATIONAL_PROMPT_UTILITY_NOT_ESTABLISHED。
7. 汇总 overall/task/grid/group_count，报告K48/64/96；明确这是已使用validation的探索性机制证据，不是最终精度。不得posthoc改阈值。
8. 测试至少覆盖：correct/wrong prompt交换、相同部署选择链、P1/P2方向与Holm、三seed每图中位数、task/grid分层、Jaccard、test隔离、prereg不变、上一报告gate/SHA引用、schema。
9. 输出 stem=2026-10-03-operational-prompt-utility-diagnostic 的 json/md/records.csv，以及 logs 下 run/focused-tests/full-tests/harness-result；更新CURRENT_STATUS顶部。运行聚焦与全量 unittest。
10. 最终只输出<=25行JSON：decision、P1/P2、S1-S4、各任务K48、K64/96、Jaccard、isolation、tests、files、limitations、next_decision。
