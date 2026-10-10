# Current Selector CV-Bench Preflight Only

项目根：`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`。只执行工程工作，输出简练。

重要环境事实：Harness 自身隔离命名空间看不到 `/dev/nvidia*`，不要运行/解释 nvidia-smi，不要写 GPU BLOCKED。Codex 已在宿主只读确认 RTX A6000 46068 MiB、used 471 MiB、free 45015 MiB、util 0%；唯一 GPU PID 16507 属于 seey6、382 MiB，资源充足且绝不修改。Harness 本轮只做 CPU preflight、代码、测试、预注册和宿主命令；GPU 由 Codex 在宿主运行。

遵循 `reports/token_selector/prompts/2026-10-04-current-selector-cvbench-gate-harness-prompt.md` 的评测标准、安全边界和输出要求，但本轮只完成以下事项，严禁读取任何新的 proposed CV-Bench 输出：

1. 只读审计当前真实 B1 runner、2026-10-03 integer scorer/compact HLS 对应的唯一算法链和候选 checkpoint。必须只依据已有非 CV-Bench validation 元数据，冻结唯一 deployment checkpoint/seed；若没有唯一选择规则则写清歧义并 fail，不得按 CV-Bench 选择。优先检查已有配置/报告是否已明确部署 checkpoint；不得自行平均三 seed。
2. 在读取 proposed CV-Bench 输出前写不可变预注册：`configs/current_selector_cvbench_20261004.preregistered.json` 和对应 prereg report。锁定 full2638 manifest/图片、官方 evaluator、dense baseline、checkpoint、K=48/64/96、model path、max_new_tokens、do_sample=False、质量/延迟字段、bootstrap、代码与输入 SHA256、禁止 post-hoc 回选。Pointing/Grounding/NDCG/retained-mass 不得作为主指标或选择依据。
3. 审计并最小修改/新增 runner，使真实 RoboBrain B1 prompt-conditioned 当前 Selector 可以：identity gate、结构 smoke、full2638、断点续跑；逐样本保存 raw JSONL；严格按 `configs/evaluation_protocol.json` 的 model-E2E/application边界计时并在 GPU 段 synchronize。禁止修改原模型、app、既有 HLS/cosim/OOC/BD 和既有报告。
4. 新增结果汇总器：只在发现恰 2638 个唯一完整样本时输出正式 CV-Bench 官方组合分数/2D/3D/四任务/absolute delta/relative retention/parse rate/paired bootstrap CI；部分结果只能 IN_PROGRESS，绝不计正式分。使用既有官方 scorer，不复制或发明评分公式。
5. 新增 host shell driver，阶段严格为：identity <=5 -> proposed structural smoke <=5（不聚合质量）-> K48/K64/K96 full2638。必须支持从 raw JSONL 安全恢复，不能重复样本。另生成固定分层100样本×3 repeats性能命令，但本轮不跑。
6. 运行 CPU 测试、py_compile、CLI help；不得安装依赖。保存日志和 SHA。更新 CURRENT_STATUS.md 为 PREFLIGHT_READY 或明确失败，不能宣称精度/加速。
7. 最终 stdout 不超过80行：状态、冻结 checkpoint及依据、创建/修改文件、测试结果、Codex下一条精确宿主命令。不要给建议，不要运行GPU。
