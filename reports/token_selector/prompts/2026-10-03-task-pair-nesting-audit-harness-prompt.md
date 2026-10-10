# 2026-10-03 Task-pair Nesting Audit Harness Prompt (authority text, verbatim)

执行一个短小的“Grounding/Pointing 负样本有效性审计”，不训练。项目根目录当前目录。先把本提示保存为 reports/token_selector/prompts/2026-10-03-task-pair-nesting-audit-harness-prompt.md，并在读取 operational records.csv 或 clean manifest 内容前创建 configs/task_pair_nesting_audit.preregistered.json 和对应 preregistration JSON，冻结输入SHA和规则；只可先哈希输入。

输入仅限：
- datasets/prompt_projection_clean_inductive_v1.jsonl
- reports/token_selector/2026-10-03-operational-prompt-utility-diagnostic-records.csv
- 对应 operational JSON
禁止打开任何NPZ、原图、权重、internal test，禁止训练/GPU/联网，禁止改既有文件。

计算38个val图像（sample id/split映射可从 operational records CSV，不读NPZ）：
1. point是否落在bbox内，overall比例；
2. bbox面积分布；
3. 每图B臂三seed中位数的grounding/pointing K48 correct-wrong retained-mass gap；
4. point-inside与gap的分层；bbox面积与grounding gap的Spearman；
5. grounding与pointing正确/错误selected-set Jaccard分层（若records已有）。
冻结判定：
- SAME_IMAGE_TASK_TARGETS_NESTED iff point-inside-bbox比例>=0.80；
- SYMMETRIC_NEGATIVE_INVALID iff TARGETS_NESTED 且 grounding gap<=0、pointing gap>0；
- 否则 NOT_ESTABLISHED。
这只是解释性诊断，不改变前一实验的NOT_ESTABLISHED，也不作精度/晋级声明。

新增 evaluation/task_pair_nesting_audit.py 和测试；输出 reports/token_selector/2026-10-03-task-pair-nesting-audit.{json,md}、records.csv、logs四件套；测试覆盖几何边界、聚合、判定、输入隔离、prereg不变、schema；运行聚焦与全量unittest，更新CURRENT_STATUS。最终<=20行JSON，仅结果。
