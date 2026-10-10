# 任务提示原文（冻结）

你是执行工程师。Codex 已完成独立审计，现在只执行“空间学生部署链路失效分解”，不要提出研究方向，不要训练，不要碰 internal test。项目根目录：/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102。Python：/home/zhaoxiangyu/conda-envs/robobrain25/bin/python。

执行合同：
1. 在读取现有 spatial-student 的 val-samples.csv/json、val NPZ 或 best checkpoint 之前，先把本提示原文保存到 reports/token_selector/prompts/2026-10-03-spatial-selector-failure-diagnostic-harness-prompt.md，并创建 configs/spatial_selector_failure_diagnostic.preregistered.json 与对应 preregistration JSON，记录创建时间和所有输入路径/SHA256。声明这是已使用 validation 的探索性诊断，只能指导下一次预注册，不能作最终精度结论。
2. 禁止打开/哈希/解析 internal test NPZ；禁止读取原始图片、4B 权重、密钥、代理配置；禁止 GPU、联网、训练、改已有报告和训练代码。
3. 复用 evaluation/train_spatial_prompt_selector.py 中空间 target、score_candidate、combine_scores、select_real_groups、ndcg_for_selected_indices 等，不复制实现。使用 B_SPATIAL_KL_PRIMARY 三个 best checkpoint、原冻结 split 的 76 条 val/38 张图。
4. 对每条记录、每 seed、K=48/64/96计算：
   - L2 plain top-K
   - L2+3x3 anchor
   - B semantic plain top-K
   - B semantic+3x3 anchor
   - combined(minmax B semantic + L2) + anchor，lambda=0.5
   - exploratory lambda sweep + anchor：lambda=[0,0.05,0.1,0.2,0.3,0.5,0.7,1.0]
   - exact spatial target + anchor 作为可达上界
   每种至少输出 retained spatial-target mass；涉及锚点排序时输出修正后的 selected-set NDCG；输出相对 L2+anchor 的 delta、selected-set Jaccard/overlap。
5. 汇总必须按 overall、grounding、pointing、group_count/grid 规模分层；三 seed 先对每图取中位数，再按独立图像汇总 mean、median、95% bootstrap CI、better/equal/worse 计数。lambda sweep 仅描述，不做显著性/晋级声明。
6. 明确判因，可多标签，但定义必须在预注册中冻结：
   FUSION_DAMAGE：semantic+anchor 优于 L2+anchor，但 lambda=.5 combined+anchor 不优于 L2+anchor；
   HEAD_SIGNAL_WEAK：semantic+anchor 不优于 L2+anchor，且所有 lambda>0 在 K48 overall 及两个任务均不优于 lambda=0；
   ANCHOR_BUDGET_DAMAGE：semantic plain 优于 L2 plain，但 semantic+anchor 不优于 L2+anchor；
   TASK_SPECIFIC：grounding 与 pointing 的 K48 delta 符号相反；
   HEADROOM_PRESENT：oracle+anchor 相对 L2+anchor 的 K48 retained-mass mean delta >0.10。
   “优于”定义为图像级 mean delta >0，不做 p 值判断。
7. 新增 evaluation/spatial_selector_failure_diagnostic.py 和对应测试。测试至少覆盖 test NPZ 隔离、七类模式/8个 lambda、正确 selected-set NDCG、三 seed 每图中位数、分任务/规模汇总、判因规则、预注册哈希不变、报告 schema。
8. 输出：
   reports/token_selector/2026-10-03-spatial-selector-failure-diagnostic.{json,md}
   reports/token_selector/2026-10-03-spatial-selector-failure-diagnostic-records.csv
   reports/token_selector/logs/同 stem 的 run/focused-tests/full-tests/harness-result
   更新 CURRENT_STATUS.md 顶部，写清结果、限制、精确下一决策条件。
9. 运行聚焦测试和全量 unittest discover -s evaluation/tests。不得改已有测试以让失败消失。
10. 最终只输出不超过25行 JSON：status、diagnosis、K48关键数值、task分层、lambda sweep最佳描述值、headroom、test isolation、tests、files、limitations。不要输出思考过程。
