# 当前工作状态与接续入口

- [x] 2026-10-10 **VSR zero-shot dev 短决策预算曲线完成，下一主线固定为 K96 关系保持与通信规避**：官方提交 `b27a0af0…a241a` 的 zero-shot dev 340 条、227 张 COCO 原图已全部物化并逐图校验；test 仅保持既有 hash/行数审计，未解析、未评分。预注册使用 edge576、BF16/SDPA、greedy、`max_new_tokens=4`、严格 True/False，完成 Dense、修正整数 L2 K96/K144/K192、v2 K96/K144 共 **2040** 条记录，解析成功率均 100%；v2 K192 因冻结整数 Prompt scorer 内部 cap16 审计选择只能提供 144 候选而需求 183，在任何 warmup/正式记录前 fail-closed，未修改已 HLS 审计核心。准确率：Dense **0.7912**；L2 K96/K144/K192 **0.6647/0.7176/0.7706**；v2 K96/K144 **0.6853/0.7382**。v2 K96/K144 相对 Dense **0.8662/0.9331**；同预算 v2−L2 都为 **+0.0206**，95% CI 分别 `[-0.0088,+0.0529]`、`[-0.0029,+0.0471]`，尚未证明稳定优于 L2。现有已接受 compact HLS 只支持 `K<=96`；K96 达到 0.88 宽松开发线还差 **4** 条正确。A6000 软件 v2 K96 CPU selector 往返均值 **43.05 ms**；按已接受 200 MHz HLS 周期公式和本轮真实 G 投影，FPGA compute **4.14 ms**、平均双向载荷 **1.835 MB**，1 GbE 串行模型段 **91.56 ms** 对 Dense model inference **88.98 ms** 为 **0.972x**，假设 10 GbE 为 **76.55 ms / 1.162x**，均非板端实测。**决策**：当前 K96 不晋级、不扩大 K；下一阶段固定 K96/3x3/整数接口，在 VSR zero-shot train 发展成对关系保持目标，先补 4 条质量缺口并使同预算 v2−L2 CI 不跨 0；同时优先 PCIe/共享内存或调整切点避免 1 GbE 全特征往返。报告：`reports/token_selector/2026-10-10-vsr-zeroshot-dev-budget-curve.{json,csv,md}` 与 `…-outcome-analysis.{json,md}`。

更新时间：2026-10-10（VSR dev 340 条短决策曲线完成；K96 质量差4条且1GbE无净收益；转入固定K96关系保持训练与通信规避）

- [x] 2026-10-09 **外部失败后的下一发展域已审计为 VSR zero-shot，dev 图像现已补齐**：SpatialEval 官方只提供 test，仓库截至当前未发布生成器，不能继续用其 test 调参。固定 VSR 官方提交 `b27a0af0…a241a`。random split 的 train/dev 图像重叠 **558** 张，不作为主划分；zero-shot train/dev/test 为 **3489/340/1222**，train/dev 对象概念重叠 **0**、图像重叠 **5**。zero-shot test 只核对 SHA、字节数和行数，未解析 JSON。dev 的 227 张图已全部物化；后续执行结果见上一条。

更新时间：2026-10-09（VSR random 因 train/dev 558 图重叠弃用；zero-shot 概念重叠0、dev仅缺222图；test保持未解析；等待小规模图片下载授权）

- [x] 2026-10-09 **SpatialEval VQA 4500 三臂外部确认完成，冻结 v2@B1/K96 未通过，状态 `CURRENT_K96_SELECTOR_REJECTED_BY_EXTERNAL_CONFIRMATION`**：按 r2 预注册完成 Dense、修正整数 L2@K96、v2@K96 各 4500 条，共 13500 次推理；三臂均为 4500 个唯一 ID、每任务 1500 条，记录哈希、无标签/答案进入 Selector、append-only 恢复及 `ALREADY_COMPLETE` 检查全部通过。官方兼容三任务等权项目宏平均：Dense **0.583778**、L2 **0.439111**、v2 **0.434444**；v2/Dense **0.744195**，未达到冻结的 0.90 门槛；v2-L2 **-0.004667**，按 `task::question_id` 分层配对 bootstrap（seed 20261009、5000 次）95% CI **[-0.013778,+0.004444]**，下界未达到 0。两项外部门均失败，且 v2 没有优于同预算 L2 的证据。平均视觉 token 从 Dense **468.98** 降到 **96**（减少 **79.53%**），但 A6000 软件路径 mean elapsed 从 **1426.78 ms** 增到 **1472.96 ms**（慢 **3.24%**）；Dense Decode **1220.92 ms**、占 **85.57%**，即使把所有非 Decode 阶段理想化为零，Amdahl 上限也仅 **1.169x**。Dense **4231/4500**、v2 **4310/4500** 条输出撞到 64-token 上限，因此 Decode 主导只适用于本次官方解释型 Prompt，不能直接外推到坐标/动作码短输出。Prefill 从 80.78 ms 降到 37.84 ms（2.135x），只节省 42.94 ms；v2 CPU selector 往返 81.07 ms，直接乐观减为零也只约 **1.025x**，且不是 FPGA 预测。正式汇总见 `reports/token_selector/2026-10-09-spatialeval-three-arm-summary.{json,csv,md}`；失败归因见 `reports/token_selector/2026-10-09-spatialeval-external-failure-analysis.{json,md}`。**决策**：停止将当前固定 K96 v2 作为主方案晋级；SpatialEval test 已解封，不得用于调参或再次充当未见外部集。下一步仅在原发展数据研究自适应预算、空间关系保持与结构化短解码/非自回归动作头的联合设计，冻结新版后换一个未观察公开基准确认；Jetson/ZCU102 实测不能推翻本次质量失败。

更新时间：2026-10-09（SpatialEval 4500×3 外部确认完成；v2/Dense 74.42%，v2-L2 CI 跨零，两门失败；当前 K96 v2 被拒绝；转向视觉压缩与 Decode 联合优化）

- [~] 2026-10-09 **SpatialEval VQA 4500 外部确认：独立继任实现已 `PREREGISTERED_NOT_EXECUTED`（CPU-only；本实现未运行 GPU、未加载模型权重、未写任何预测结果）**：在既有草案 `configs/2026-10-09-spatialeval-vqa-external-confirmation.draft.json`（字节不变、SHA `20e7c890…cd3e`）之外新增**继任预注册** `configs/2026-10-09-spatialeval-vqa-external-confirmation.preregistered.json`，状态 `PREREGISTERED_NOT_EXECUTED`、`executed=false`、`gpu_runs=0`、`weights_loaded=false`、`labels_observed=true`、`not_label_blind=true`、`not_used_for_selector_tuning=true`；明确 `SpatialReal135` 仅探索、不进主门，官方 `evaluation.py` 无跨任务总分，项目汇总命名为 `project_task_macro`/`project_micro`。**新文件**：官方兼容 evaluator `evaluation/spatialeval_official_evaluator.py`（SHA `b16c7cb2…edfb0`；importlib 加载 pin 住的上游 `evaluation.py` commit `d82ba382`，固定 `model_name=robobrain25`，调用原始 spatialmap/mazenav/spatialgrid parser，主分严格用 `int(str(oracle_answer).lower() in str(parsed_answer).lower())`，解析异常/None 计错，另出 normalized strict 诊断，按 task/qid/3-task macro/4500 micro 汇总）、合同 `evaluation/spatialeval_vqa_contract.py`（`f13c9219…f5a1a`；两 `/tmp` parquet 按 reference manifest 行索引物化去重图像到 `/tmp/zhaoxiangyu-spatialeval-vqa/images`，校验 parquet SHA/bytes/rows、图像 SHA 与宽高、prompt=原 text+官方 w_reason 原句、max64/BF16/SDPA/65536-589824/K96/3x3/cap16/S16、标签禁入 selector、append-only resume）、三臂 runner `evaluation/run_spatialeval_vqa_three_arm.py`（`1edbebef…22ee`；复用 `evaluation/run_edge576_cvbench_three_arm.py` 冻结 Dense/L2/v2 生产者，未复制 selector 算法，每臂一次 1-token warmup + CUDA sync + 原始 JSONL）、fail-closed 汇总器 `evaluation/summarize_spatialeval_vqa_three_arm.py`（`9589d403…94aad`；要求三臂各 4500 且 sample 集合一致，项目门 v2/dense task-macro `>=0.90` 与 `task::qid` 分层配对 bootstrap seed=20261009/reps=5000 的 v2−l2 95% CI 下界 `>=0`，逐任务只报告）。config 现场复核 **12 个 runner 依赖 pin 全匹配**（另含两 parquet SHA/bytes/rows、reference manifest `03649b9c…f4050`、官方三文件 SHA、模型身份 `bbfcda18…f8e3`、全部新文件 SHA）。**新增 4 个 CPU 聚焦测试**（`test_spatialeval_official_evaluator.py`、`test_spatialeval_vqa_contract.py`、`test_run_spatialeval_vqa_three_arm.py`、`test_summarize_spatialeval_vqa_three_arm.py`）**64 OK**。口径：本实现全程 CPU-only，未联网下载、未运行三臂、未写预测；工作区内另有并行的 `…-three-arm-external-confirmation-r2` 运行（见下条，由其它进程执行），与本实现的文件/配置/输出路径无关且未被本实现触碰。**下一步（宿主，尚未执行）**：先恢复 `/tmp` 两个 parquet 后运行 `python evaluation/spatialeval_vqa_contract.py` 生成 run manifest，再依次 `python evaluation/run_spatialeval_vqa_three_arm.py --arm <arm> --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --validate-only` 与正式三臂。

- [~] 2026-10-09 **SpatialEval VQA 外部确认 r2 已正式冻结，Dense 第一臂从未评分前缀续跑（不得调参）**：选择 NeurIPS 2024 SpatialEval 的 VQA test 作为未用于 Selector 调参的公开空间推理外部确认集。官方论文对 Spatial-Map、Maze-Nav、Spatial-Grid 分别报告三个问题的平均准确率，并将 Spatial-Real 单独报告；固定官方 commit `d82ba382…` 的 `evaluation.py` 无 Spatial-Real handler、也无四任务官方总分。因此主集严格固定为前三个任务各 1500、共 **4500** 条，Spatial-Real 135 条只作未来独立探索诊断。r1 config `16d26764…e002f` 在 Dense 生成 93 条后发现 provenance 缺口：直接 pin 了未使用的旧投影，未直接 pin v2 loader 实际使用的三个 checkpoint；当时只查看 G/K/计时/token 数，**未读取或评分答案**，随即停止。r1 标记 `R1_SUPERSEDED_BEFORE_SCORING`，93 条原始前缀保持不变并快照为 `dense_prefix93_v1_unscored.jsonl`（801929 B，SHA `179b68b9…0196`）。正式 r2 配置 `configs/2026-10-09-spatialeval-vqa-three-arm-external-confirmation-r2.preregistered.json` SHA **`bd8e1a23…708014`**，状态 `PREREGISTERED_AMENDMENT_BEFORE_SCORING`，新增直接固定真实 `best.npz`/Wv int8/scale 和前缀快照，共 **35 pins** 现场复核通过；算法、数据、Prompt、指标、门槛、输出路径均未改变。明确 `labels_observed=true`、`label_blind=false`、但未用于 Selector 调参，观察输出后禁止调机制或门槛。主指标逐条复现官方 parser + `oracle_answer in parsed_answer` 子串语义；另报严格相等诊断。官方无跨任务总分，项目主汇总明确命名为三任务等权宏平均；成功门冻结为 v2/Dense 宏平均保留率 `>=0.90`，以及 `task::question_id` 分层配对 bootstrap（seed 20261009、5000 次）v2−L2 95% CI 下界 `>=0`，逐任务只报告不设门。三臂固定 edge576、BF16/SDPA、greedy、max64、B1/K96/grid3/cap16/v2 S16。两个官方 parquet 共 712,137,895 B、SHA 全匹配；引用 manifest 4635 条 SHA `03649b9c…f4050`；CPU 物化主集得到 4500 问题/1459 个去重图像，`/tmp` 缓存 196 MB，逐图 SHA/尺寸通过。新增官方兼容 metric、外部合同、三臂 runner、汇总器及测试，聚焦 **35 OK**。A6000 启动前检查：空闲 45,009 MiB、利用率 0%，他人进程 `seey6` 仅占 382 MiB且未触碰。Dense 将从第 94 条 append-only 续跑；首批 25 条仅做性能估算：模型推理均值约 1.443 s，Decode 1.230 s，输出均达到 64 token，预计 Dense 单臂约 1.80 小时。当前禁止生成任何精度结论；Dense 完成后依次运行 L2、v2，最后只运行一次冻结汇总器。r1 作废证据见 `reports/token_selector/2026-10-09-spatialeval-prereg-r1-superseded.{json,md}`。

更新时间：2026-10-09（SpatialEval 主集4500 r2冻结；r1在93条未评分前缀后因checkpoint直接固定缺口作废；r2 35 pins/35 tests通过；Dense从第94条续跑，不得调参）

- [x] 2026-10-08 **edge576 CV-Bench 三臂 r4 真实 GPU 执行完成 → `DEVELOPMENT_GATES_PASS_GO_EXTERNAL_VALIDATION`（仅发展集；非论文级精度、非外部泛化确认）**：r3 首次真实 GPU smoke（`dense_edge576`、`sample_id=1838`、`cuda:0`）因冻结 dense 生产者只返回 `output_text`、真实整数计数 `output_tokens` 与 `selected_group_indices`、**从不返回 `output_token_ids`**，在写文件前的 `normalize_record` fail-closed（`Edge576ContractError`），**未写任何文件、未触碰任何 full2638 输出**；r3 标记 **superseded**（字节全保留、`must_not_be_run=true`）。r4 config `configs/2026-10-07-edge576-cvbench-three-arm-development-r4.preregistered.json` SHA `4f11d47981fd6c04c07608eba8b7e85e27e035bcb8d2d459a7ebc8a8c4b682c5`、**63 个 pin 现场复核全过**，修复为 `output_token_ids` **可选**（仅真实返回才记录、**绝不伪造**）、`output_token_count` **必备**（由真实 `output_token_ids` 长度和/或真实整数 `output_tokens` 规范映射，两者并存必须一致，全部缺失即 fail-closed）。**三臂 smoke 全通过**后完成三份 full2638（各 2638 唯一、三臂同 `sample_id` 集合且与固定 edge576 manifest 一致）：Dense raw SHA `707868c6…e784`、修正整数 L2 `e11dba93…8a8c`、v2 `8b2a2885…92f8`。官方 `cv_bench_accuracy`：Dense **0.8739581555**、修正整数 L2 **0.7937869555**（相对同输入 Dense **0.908267**）、v2 **0.8078459446**（相对同输入 Dense **0.924353**）；显式 `parsed_answer` 诚实解析失败 Dense/L2/v2 = **36/42/41**，解析失败计为错误、不设 100% 解析门。`task::source_dataset` 分层配对 bootstrap（seed **20261007**、**5000** 次、95%）：L2−Dense **-0.0801712** CI `[-0.0931172,-0.0675909]`、v2−Dense **-0.0661122** CI `[-0.0778464,-0.0547949]`、v2−L2 **+0.0140590** CI `[+0.0044580,+0.0233407]`。**两个预注册发展门全部通过**：v2/Dense `>=0.90` 与 v2−L2 95% CI 下界 `>=0`，状态记为 **`DEVELOPMENT_GATES_PASS_GO_EXTERNAL_VALIDATION`**；**不得**写成论文级精度或外部泛化已确认。**软件时延（非 FPGA 实测、非板端加速结论）**：L2 `application_end_to_end` mean/p95 **150.08/190.14 ms**、v2 **185.06/240.65 ms**；v2 `selector_cpu_roundtrip` mean/p95 **52.79/86.23 ms**；Dense **没有** `application_end_to_end_seconds` 字段，必须记为 **null**，其 `elapsed_seconds` 只映射为 `model_inference` mean/p95 **157.04/196.48 ms**，**禁止**直接与上述 application 字段比较。正式汇总 `reports/token_selector/2026-10-08-edge576-cvbench-three-arm-r4-summary.{json,csv,md}`（artifact SHA `1957c323…6549`），汇总器/测试 **28 OK**。**下一步**：在查看结果前冻结一个**此前未用于调参**的公开空间 VLM 外部确认基准（样本/Prompt/官方指标/阈值/模型身份/输出路径全部预先固定），然后**只运行一次**确认；**不得**继续在 CV-Bench full2638 上调 v2。并行工程任务：接入 **Jetson Xavier NX** 后重测真实阶段/链路延迟；ZCU102 仅在净收益合同下评估 selector，**当前不宣称 FPGA 加速**。

更新时间：2026-10-08（**edge576 三臂 r4 完成 → `DEVELOPMENT_GATES_PASS_GO_EXTERNAL_VALIDATION`（仅发展集）**：r3 dense smoke 因缺 `output_token_ids` 在写文件前 fail-closed、未写文件、r3 superseded；r4 config `4f11d479…82c5` 63 pins，token ids 可选、`output_token_count` 必备且不伪造；三臂 smoke 全通过，三份 full2638 唯一同集合，raw SHA dense `707868c6…`/L2 `e11dba93…`/v2 `8b2a2885…`；官方分 dense `0.8739581555`、L2 `0.793787`（0.908267 Dense）、v2 `0.807846`（0.924353 Dense），诚实解析失败 36/42/41 计错；`task::source_dataset` 配对 bootstrap（seed 20261007/5000）L2−Dense `-0.0801712`、v2−Dense `-0.0661122`、v2−L2 `+0.0140590` CI 下界 `+0.0044580`，v2/Dense `>=0.90` 与 v2−L2 CI 下界 `>=0` 两门通过；软件时延 L2/v2 application mean/p95 `150.08/190.14`、`185.06/240.65` ms、v2 selector roundtrip `52.79/86.23` ms，Dense application 字段必须 null、`elapsed_seconds` 只映射 model inference `157.04/196.48` ms，非 FPGA 实测；汇总 `1957c323…`、28 OK；下一步先冻结未调参公开空间 VLM 外部确认基准并只跑一次、不在 CV-Bench full2638 调 v2，并行接入 Jetson Xavier NX 重测真实阶段/链路延迟，ZCU102 仅在净收益合同下评估 selector、不宣称 FPGA 加速）

- [x] 2026-10-07 **edge576 CV-Bench 预执行修正 r2 完成（CPU-only；无 GPU/模型权重/网络/训练/依赖安装；未改 edge manifest/builder/源 full manifest/model identity/metric_v2；未改冻结历史）→ `R1_PREEXECUTION_SUPERSEDED` + `EDGE576_AUDITOR_MODEL_PATH_FIXED` + `EDGE576_MANIFEST_AUDIT_PASS`（复审 2638/2638）+ `EDGE576_R2_PREREGISTERED_NOT_EXECUTED`（第三臂 `NOT_EXECUTED`）**：主审校三点全部处置。**A/B r1 缺陷证据保留**：r1 config `8958cbd6…`、r1 报告 json `c728c312…`/md `37252dad…`、r1 生成器 `c0177675…`、r1 测试 `5eb3b8c4…` 与旧审计 json `0caacd74…`/md `380ab4d2…` **字节全部不变**；r1 **从未执行、不得运行**，新增中文作废报告 `reports/token_selector/2026-10-07-edge576-cvbench-r1-preexecution-superseded.{json,md}`（`PREEXECUTION_SUPERSEDED`，记录缺陷 A 第三臂 `mechanism_id=null`、B `record_completeness` 写「全部可解析」与诚实解析冲突、C 审计器 `--model` 丢失；并记录 r1 测试因 auditor pin 过期而预期 1 项失败为作废正证据）。**C 审计器修复**：`evaluation/audit_edge576_cv_bench_manifest.py`（新 SHA `a999fe0a…`）`audit()` 显式接受 `model_path`，`main` 传入 `--model`，处理器只从该路径加载，报告记录 `processor_model_path`；新增 4 个 CPU 聚焦回归测试（正例：显式 processor 跳过加载器；负例：`verify_processor` 必须把显式路径线程传入加载器；`main` 必须转发 `--model`），审计器测试 14→**18 OK**。**复审**：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 仅加载本地 processor、`--verify-processor --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B`，2638 条 record/图片 SHA/原始尺寸/公式/预算/摘要现场重算全过，权威 processor **逐样本重算 2638/2638 `image_grid_thw` 完全一致**；新报告 `reports/token_selector/2026-10-07-edge576-cv-bench-manifest-audit-r2.json`（SHA `34f1385f…`，记录实际 `processor_model_path`），旧审计报告字节不动。**r2 未来专用预注册**：`configs/2026-10-07-edge576-cvbench-three-arm-development-r2.preregistered.json`（SHA `8f265557…`、`status=PREREGISTERED_NOT_EXECUTED`、`arm_status=NOT_EXECUTED`、`executed=false`、`gpu_runs=0`）**40 个 SHA pin 现场复核 0 不匹配**，含修正审计器+复审报告、v2 机制 core/integration/runner/summariser/4 测试、projection npz `1f1d379e…`、Wv int8 `126bd842…`、per-tensor scale `93585a3b…`、整数评分/L2 核心（token_selector/corrected_l2/b1_integration/current_selector_integer/v1 layout rebuild/int8 数值审计/整数评分审计）、evaluator/metrics、冻结原生 v2 输出 `3018f759…`、r1 作废证据；三臂仍 **Dense@edge576 / corrected integer L2@B1 K96 / v2@B1 K96**，第三臂明确冻结为现有 `conservative_l2_guarded_rerank_v2`（`is_new_mechanism=false`、不发明新机制），输出路径 `artifacts/token_selector/2026-10-07-edge576-cvbench/v2_edge576_k96_full2638.jsonl`、`output_sha256=null`。**r2 成功门**：主指标仅官方 `cv_bench_accuracy`、相对**同输入** Dense@edge576 保留 `>=0.90`、相对同切点/同 K 修正整数 L2 `task::source_dataset` 分层配对 bootstrap（5000/seed 20261007/95%）CI 下界 `>=0`、K96/87+9（`G<=96` bypass）、无泄漏；**record completeness 只要求 2638 唯一完整输出，`parsed_answer` 可 `null`，解析失败计错并报告成功/失败数与率，`requires_100_percent_parse=false` 不设 100% 解析门**。明确这是**现有 v2 在新边缘运行域（`G<=576`）的发展性复测**，不追溯改写 2026-10-06 原生 `DEVELOPMENT_STOP`。**验证**：`py_compile` 通过；聚焦 **52 OK**（builder 17 + auditor 18 + r2 prereg 17），r1 prereg 模块预期 1 项失败（auditor pin 过期，已披露）；r2 pin 审计 **40/40 通过**；`gpu_runs=0`、`weights_loaded=false`。中文 r2 报告 `reports/token_selector/2026-10-07-edge576-cvbench-prereg-r2.{json,md}`；持久结果 `reports/token_selector/2026-10-07-edge576-r2-harness-result.json`。**下次允许动作**：为 edge576 适配三臂 runner（消费 edge manifest、`min_pixels=65536`/`max_pixels=589824`、逐样本断言 `image_grid_thw`），入口见 r2 config `runner_adaptation_entry`（dense `evaluation/run_hf_shallow_block_pruning.py --retentions 1.0`；L2 `evaluation/run_hf_hardware_regular_b1_corrected_l2.py`；v2 `evaluation/run_hf_conservative_l2_guarded_rerank.py`）；适配前不启动任何模型。

更新时间：2026-10-07（**edge576 r2 预执行修正完成 → `R1_PREEXECUTION_SUPERSEDED` + `EDGE576_AUDITOR_MODEL_PATH_FIXED` + 复审 `EDGE576_MANIFEST_AUDIT_PASS` 2638/2638 + `EDGE576_R2_PREREGISTERED_NOT_EXECUTED`**：r1 字节全保留、从未执行不得运行；审计器 `a999fe0a…` 显式 `model_path`/`--model`/报告实际路径、18 测试；复审报告 `34f1385f…`；r2 config `8f265557…` 40 pins 0 不匹配、第三臂=现有 `conservative_l2_guarded_rerank_v2`（`is_new_mechanism=false`、`NOT_EXECUTED`、`v2_edge576_k96`）、记录完整性 2638 唯一完整输出且 `parsed_answer` 可 null、解析失败计错并报告、无 100% 解析门；保留官方主分/同输入 Dense>=0.90/分层配对 bootstrap CI>=0/K96 87+9/无泄漏门；现有 v2 新边缘域发展性复测、原生 STOP 不追溯；聚焦 52 OK、r2 pin 审计 40/40、`gpu_runs=0`）

- [x] 2026-10-07 **edge576 CV-Bench 基础设施冻结完成（CPU-only；无 GPU/模型权重/网络/训练/依赖安装；未改原模型/app/冻结历史 config/raw/report）→ `EDGE576_MANIFEST_BUILT` + `EDGE576_MANIFEST_AUDIT_PASS` + `PREREGISTERED_NOT_EXECUTED`**：新增确定性构建器 `evaluation/build_edge576_cv_bench_manifest.py`（SHA `12258b15…b48a`）、独立只读审计器 `evaluation/audit_edge576_cv_bench_manifest.py`（`45dcd049…e683`）与 3 个 CPU 聚焦测试（`d7556f31…e514`/`59ed16da…a026`/`5eb3b8c4…d40`）。以本地 RoboBrain2.5-4B processor、`HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`、仅 CPU、保持长宽比、`min_pixels=65536`、`max_pixels=589824` 处理完整 CV-Bench **2638** 张图（**未加载模型权重**）。manifest `datasets/2026-10-07-cv-bench-edge576-full2638.jsonl` SHA **`b5355ba1…dd1f`**（3280730 B、2638 唯一、ID/标签/源/任务与冻结 full manifest `22399a76…3d91` 完全一致）；摘要 `…summary.json` SHA `492c8295…1a6b`。**G 分布**（权威 `image_grid_thw`，`G=patch//4`）：min/median/mean/max = **64/300/357.07/576**、P90=P95=576、恰 576 有 **400**；分桶 `<=96=143`、`97-256=781`、`257-512=706`、`513-576=1008`；全部 `G<=576`、无样本事后删除；任何 `G>576` 即 fail closed 不写 manifest。**独立审计 `EDGE576_MANIFEST_AUDIT_PASS`**（`reports/token_selector/2026-10-07-edge576-cv-bench-manifest-audit.{json,md}`，SHA `0caacd74…30f8`/`380ab4d2…6c35`）：2638 条 record/图片 SHA/原始尺寸/公式/预算/处理参数/摘要统计全部现场重算一致，并用权威 processor **逐样本重算 2638/2638 `image_grid_thw` 完全一致**。**未来专用预注册 `PREREGISTERED_NOT_EXECUTED`**：`configs/2026-10-07-edge576-cvbench-three-arm-development.preregistered.json`（SHA `8958cbd6…b3f3`）**13 个 SHA pin 现场复核全过**（builder/auditor/3 测试/edge manifest+summary/源 full manifest/model identity `bbfcda18…f8e3`/`cv_bench_metric_v2` `2a2f2004…a3a1`/corrected-L2 runner+raw/dense 参考），锁定三臂 **Dense@edge576 / corrected integer L2@B1 K96 / next candidate@B1 K96** 共享同一 edge576 manifest、同一 Prompt、BF16/SDPA、greedy、`max_new_tokens=16`，且每臂必须逐样本断言 `image_grid_thw` 与 manifest 一致；主指标仅 `cv_bench_accuracy`（显式 `parsed_answer_key` 诚实解析门），质量保留率 `>=0.90`，相对修正整数 L2 按 `task::source_dataset` 分层配对 bootstrap（5000/seed 20261007/95%）CI 下界 `>=0`，无标签泄漏；next candidate 机制 `null`（待独立策略审查）；full2638 为 development、原生分辨率 `DEVELOPMENT_STOP` 不追溯、高分辨率仅 stress。**验证**：`py_compile` 6 文件 OK、聚焦 **39 OK**；审计真实通过；`gpu_runs=0`、未启动三臂推理。**下一允许命令（宿主，尚未执行）**：先为 edge576 改编三臂 runner（消费 edge manifest、使用 pinned min/max pixels 并逐样本断言 grid），再运行 Dense@edge576；在 manifest/模型身份/代码 SHA/门/输出路径全部冻结前不启动全量 GPU。

更新时间：2026-10-07（**edge576 基础设施冻结 → `EDGE576_MANIFEST_BUILT` + `EDGE576_MANIFEST_AUDIT_PASS` + `PREREGISTERED_NOT_EXECUTED`**：2638 唯一、G=64..576（分桶 143/781/706/1008）、全部 `G<=576`、manifest `b5355ba1…`、摘要 `492c8295…`；独立审计重算 record/图片 SHA/尺寸/公式/预算/摘要并用权威 processor 复算 2638/2638 grid；预注册 config `8958cbd6…` 13 pins 全过，锁定三臂同 edge576 输入/Prompt/BF16/SDPA/greedy/max16、保留率>=0.90、`task::source_dataset` 配对 bootstrap CI 下界>=0、诚实解析门、无标签泄漏；full2638 development、原生 `DEVELOPMENT_STOP` 不追溯、高分辨率仅 stress；聚焦 39 OK、py_compile 6 文件 OK、`gpu_runs=0`；三臂未运行）

- [x] 2026-10-07 **边缘部署主运行域与未来成功判定已冻结到规划文档**：后续 Selector
  主实验不再要求固定 K96 覆盖 Qwen3-VL 全部原生动态高分辨率范围；完整 CV-Bench 2638
  样本统一保持长宽比缩放到 Patch Merger 后 `G<=576`，Dense、修正整数 L2 和候选 Selector
  必须使用完全相同的 edge576 输入、Prompt、greedy 解码和输出上限。主预算固定 K96；
  `G<=96` identity bypass，`96<G<=576` 压缩。原生动态分辨率结果（median G=300、P90=1400、
  max G=3588、560/2638 条 `G>1024`）保留为只读压力测试，其既有 `DEVELOPMENT_STOP`
  不追溯改判，但不再阻断边缘主档位。未来主质量指标仅为 CV-Bench 官方
  `cv_bench_accuracy=((ADE20K+COCO)/2+Omni3D)/2`；成功门为相对同输入
  Dense@edge576 保留 `>=90%`，并且相对同切点/同 K 的修正整数 L2 基线之
  `task::source_dataset` 分层配对 bootstrap 95% CI 下界 `>=0`。使用未来评测协议 v2
  显式区分 correct/parsed answer，诚实报告解析成功/失败；General/Pointing/Grounding/
  Trajectory 只作为 API 功能回归，不再以 PCK/IoU/ADE/FDE 或自定义 E/G/Q 充当主门。
  full2638 仍是发展集，通过后只允许一个新的未用于调参的公开空间 VLM 基准做预注册确认。
  **下一步**：先生成并审计 edge576 manifest，再运行 Dense@edge576、L2@K96 和预注册候选；
  在 manifest、模型身份、代码 SHA、门和输出路径冻结前不启动全量 GPU。权威条款见
  `docs/PL_ACCELERATION_RESEARCH_PLAN_2027.md` 顶部 2026-10-07 更新及 `docs/ROADMAP.md`。

更新时间：2026-10-07（边缘主档位 `G<=576`、K96、同分辨率 Dense 对照、CV-Bench 官方
主分和诚实解析门已固化；高分辨率仅作压力测试，旧冻结结果不改判）

- [x] 2026-10-06 **未来评测协议 v2（生命周期安全 + 诚实解析门 + 模型身份固定）CPU-only 完成 → `PROTOCOL_V2_READY`（仅面向未来，不追溯修改任何冻结产物）**：复用冻结 `parse_choice`/`answer_letter`，新增未来专用 `evaluation/cv_bench_metric_v2.py`（SHA `2a2f2004834803927f34114a916e0e867a6091ada0e66d05be6ee109183a73a1`）：显式要求 `correct_key` 与 `parsed_answer_key` 两个**不同**字段、校验记录数/键/类型/source/task/唯一 `sample_id` 并 fail-closed，按冻结权重 `((ADE20K+COCO)/2+Omni3D)/2` 计分并输出诚实解析成功数/失败数/成功率；**绝不从 `correct_key.replace("_correct","_answer")` 推断解析键**。现场只读重算四臂诚实解析失败：dense **40**、修正整数 L2 **42**、prompt 融合 **42**、v2 **43**（冻结 scorer 因键别名恒报 1.0）；四臂主分现场重算与冻结值逐位一致（dense `0.8727298774444869`、修正整数 L2 `0.7668517264725796`、prompt `0.7773617325561999`、v2 `0.7837404943432142`），故**不追溯改变 v2 主分或 `DEVELOPMENT_STOP`**。新增流式模型身份构建器 `scripts/build_model_identity_manifest.py`（SHA `33caea5818bf06a7d5655513ae23dd0fc17fcd9a5d8108ae600dd9c82fb8545f`；1 MiB 分块 SHA256、不将权重载入 RAM、拒绝缺失的 index 引用分片、确定性排序 JSON）并生成 `artifacts/provenance/robobrain25-4b-model-identity.json`（SHA `bbfcda188605f54a9c7fea83e7fc2a46eca8ac635c086d23861b294c9950f8e3`；10 个文件、1 个权重分片 `model.safetensors` SHA `e2f6e3c372f4ef76afb5e9e28ec56679944fcc3b4f2ce2ebda8b21264e96eb2f`、9665586903 字节；`--check` 重算 `MATCH`）。新增协议 `configs/evaluation_protocol_v2.json`（`future_only=true`、`retroactive=false`；pin 上述 metric/manifest/builder 三 SHA；显式生命周期 `PREREGISTERED_NOT_EXECUTED`→`EXECUTED_UNAUDITED`→`AUDITED_STOP`/`AUDITED_GO_EXTERNAL_VALIDATION`；缺输出测试必须使用临时/显式路径、列出禁止探测的真实默认输出；区分 development/untouched confirmation/software latency/FPGA estimate/board measurement）。新增聚焦测试 `evaluation/tests/test_cv_bench_metric_v2.py` 与 `evaluation/tests/test_build_model_identity_manifest.py`（均只用临时路径、不改既有报告）；新增报告生成器 `scripts/build_future_evaluation_protocol_v2_report.py` 并产出中文报告 `reports/token_selector/2026-10-06-future-evaluation-protocol-v2.{json,md}`（`46be99f9…4f3e`/`0c12b313…50f8`）。**验证**：`py_compile` 5 文件 OK；聚焦 **40 OK**、既有审计测试模块 **37 OK**；**未运行全量 `evaluation/tests`**（5 项冻结预执行测试状态过期、1 项会在臂存在时重写冻结 development 报告）；v2 raw SHA 复核仍 `3018f7593d31fd02854f3f63f992b6fc6293ba64c24185e2905877a71d88fdb1`、冻结 development `generated_at` 仍 `2026-10-05T16:51:40.293686+00:00`、36 个非空 v2 pin 复核 **36/0**（1 个 null=`new_arm_jsonl`）、config SHA 仍 `7499b6ea…74eb`。**口径**：CPU-only/只读、`gpu_runs=0`、未加载模型权重/未联网/未训练/未装依赖；模型身份清单只是本地文件哈希，不是精度/部署主张。**下一命令（宿主，复核模型身份与 pin）**：`python scripts/build_model_identity_manifest.py --check && python evaluation/audit_conservative_l2_guarded_rerank_v2_full2638.py --recheck-pins`；**下一允许动作**：在新协议下独立预注册新候选。

更新时间：2026-10-06（**未来评测协议 v2 完成 → `PROTOCOL_V2_READY`（未来专用、非追溯）**：新 `cv_bench_metric_v2.py` 强制显式 `correct_key`/`parsed_answer_key` 不同并 fail-closed，诚实解析失败 dense 40/修正整数 L2 42/prompt 42/v2 43（冻结 scorer 恒 1.0 键别名缺陷）；四臂主分现场重算逐位不变 → 不改 `DEVELOPMENT_STOP`；模型身份清单 `bbfcda18…` 流式 SHA256（`model.safetensors` `e2f6e3c3…eb2f`、10 文件、9665586903 字节、`--check` MATCH）；协议 `configs/evaluation_protocol_v2.json` pin metric/manifest/builder，显式四生命周期与临时路径缺输出测试策略，区分开发/未触碰确认/软件时延/FPGA 估计/板端测量；聚焦 40 OK + 审计模块 37 OK、未跑全量；v2 raw SHA 与冻结 generated_at 不变、36/0 pin、config SHA 不变；CPU-only、gpu_runs=0）

- [x] 2026-10-06 **v2 full2638 发展性结果 CPU-only 只读审计 + 标签感知尸检完成 → 审计 `AUDIT_PASS` + 冻结判定 `DEVELOPMENT_STOP`（唯一失败门 `score_at_least_90pct_dense`）**：现场重算 raw SHA256 = `3018f7593d31fd02854f3f63f992b6fc6293ba64c24185e2905877a71d88fdb1`（与声明一致）、恰 2638 条唯一记录（0 缺失/0 多余/0 重复，与 frozen manifest 及 dense/修正整数 L2/prompt 融合参考 ID 集合一致）；143 identity bypass（`G=64..96`，合法 `G<=96`、`selected=range(G)`、`real=G`、`context=0`、`swap=0`）+ 2495 压缩样本（每条精确 `K=96`/`real=87`/`context=9`、anchor 冻结从不移除、同区域一对一、每区域 real 计数 before==after、context 池非空且覆盖全部 9 区域、`legal_gain=semantic_gain-l2_loss=utility_added-utility_removed>0`、selected/patch/source_type/region_id/valid/baseline/reranked real 及全部 audit 数组 hash 现场重算一致）。冻结主指标 `score(...)["cv_bench_accuracy"]` 现场重算：dense **0.8727298774444869**、修正整数 L2 **0.7668517264725796**、既往 prompt 融合 **0.7773617325561999**、v2 **0.7837404943432142**；v2−L2 = **+0.016888767870634647**，`task::source_dataset` 分层配对 bootstrap 95% CI **[0.006157592750679508, 0.0271543129924543]**（5000 次、seed 20261006），90% dense 门 **0.7854568897000382**，绝对缺口 **0.0017163953568240**（相对门 **0.2185219048%**、相对 dense **0.1966697143%**），v2 保留 dense 分数 **0.8980333028567213**。五门重算：`all_2638_complete_parseable` ✅、`exact_k96_layout_contract` ✅、`score_at_least_90pct_dense` ❌、`paired_ci_lower_nonnegative` ✅、`no_forbidden_leakage` ✅ → 决策保持 `DEVELOPMENT_STOP`（**不得**在看到 full2638 后放宽 preregistered 90% 门）。**发现（不改变判定）**：冻结 `all_2638_complete_parseable` 由 `score(...)["parse_success_rate"]` 计算，而汇总器 arm 行的 `correct` 键无可替换的 `_correct` 子串，使该值恒为 1.0；诚实 `parse_choice` 实测解析失败 new **43**/dense **40**/修正整数 L2 **42**/prompt **42**（仍因 score 门独立失败而 STOP）；冻结 config 未 SHA-pin 模型权重（模型身份仅 `model_path/model_label` metadata），3 个部署 checkpoint 已 pin。**36 个非空 config pin 现场字节一致、1 个 null（未生成的 new_arm），创建审计后再次复核仍 36/0、config SHA `7499b6ea…74eb` 不变**；v1 全部 pin 前后一致，v1 从未运行/未修补。**标签感知尸检（仅诊断/发展、禁止据标签调参或选 v3）**：v2−修正整数 L2 修复 **117**、回归 **73**、净 **+44**、答案变化 **223**；v2−dense 修复 **55**、回归 **287**、净 **−232**；按 task v2 Count 423/788、Relation 599/650、Depth 537/600、Distance 499/600；按 source_dataset Hypersim 324/400、SUNRGBD 393/400、nuScenes 319/400、COCO 611/805、ADE20K 411/633；按冻结原始 G 桶（`<=96`/`97-256`/`257-512`/`513-1024`/`>1024`）104/143、690/781、528/706、350/448、386/560；压缩样本 swap 直方图 `{16:2406, 15:11, 14:17, 13:18, 12:10, 11:11, 10:9, 9:5, 8:5, 7:2, 6:1}`，无 0-swap 压缩样本。**新建**：审计脚本 `evaluation/audit_conservative_l2_guarded_rerank_v2_full2638.py`（`e781c923…a667`）、聚焦测试 `evaluation/tests/test_audit_conservative_l2_guarded_rerank_v2_full2638.py`（`50860cc6…4b8`）、中文审计 `reports/token_selector/2026-10-06-conservative-l2-guarded-reranker-v2-full2638-audit.{json,md}`（`bdb2f0a9…dbdf`/`5f420bdc…f4d2`）、中文尸检 `…-postmortem.{json,md}`（`e4317fe7…7d05`/`e6b17f82…8e30`）。**验证**：`py_compile` 2 文件 OK；聚焦 **37 OK**（真实退出码 0）；全量 `evaluation/tests` 实收 **1859**、其中 **5 项为预先存在的过期断言失败**（冻结 smoke 审计断言 full2638 臂输出不存在、冻结汇总器测试断言臂缺失时退出 2；二者在 full2638 臂早已生成后必然失败，与本次审计无关，本轮未改任何冻结测试/源码），另有 1854 OK，真实退出码 **1**；运行全量套件会使冻结 `reports/cv_bench/2026-10-06-…-development.{json,md}` 被汇总器等价重写（仅 `generated_at` 变化），已按原始 `generated_at=2026-10-05T16:51:40.293686+00:00` 逐字节还原。**口径**：`gpu_runs=0`（本审计任务）、`audited_gpu_runs=1`、`audited_samples=2638`；CPU-only/只读、未加载模型权重/未联网/未训练/未装依赖；full2638 已被反复观测，**不是**未触碰确认集；无 FPGA 时延/上板部署/论文精度/外部验证主张；本机制不再获得外部验证许可。**下一命令（宿主，仅复核冻结 pin）**：`python evaluation/audit_conservative_l2_guarded_rerank_v2_full2638.py --recheck-pins`。

更新时间：2026-10-06（**v2 full2638 只读审计 → `AUDIT_PASS` + `DEVELOPMENT_STOP`**：raw SHA `3018f759…fdb1`、2638 唯一、143 bypass + 2495 压缩（K96/87+9）、全部 record/数组 hash 与 36 pin 现场复核；四臂主分 dense `0.8727298774444869`、修正整数 L2 `0.7668517264725796`、prompt 融合 `0.7773617325561999`、v2 `0.7837404943432142`；v2−L2 `+0.016888767870634647`、CI `[0.006157592750679508, 0.0271543129924543]`（5000/seed 20261006）；门 `0.7854568897000382`，绝对缺口 `0.0017163953568240`，保留 dense `0.8980333028567213`；唯一冻结失败门 `score_at_least_90pct_dense` → 保持 STOP；解析门恒 1.0 缺陷已披露（真实失败 43/40/42/42）；尸检 v2−L2 净 **+44**（修复 117/回归 73），仅诊断；新增审计+测试+4 报告；`gpu_runs=0`、`audited_gpu_runs=1`、`audited_samples=2638`；聚焦 37 OK、全量 1859（1854 OK + 5 项预先存在过期断言、退出 1）；无外部验证/论文/FPGA 主张）

- [x] 2026-10-06 **v2 六单元合同冒烟 CPU-only 只读审计完成 → `CONTRACT_SMOKE_PASS`（仅允许执行一次冻结 full2638 发展性运行）**：对被审计 raw `artifacts/token_selector/2026-10-06-conservative-l2-guarded-reranker-v2-full2638/conservative_l2_guarded_rerank_v2_k96_smoke.jsonl` 现场重算 SHA256 = `e7648d9b3d015d434b49aa8b030ba9148d294bdd837f1ce9a68d769083a15916`（与声明一致）、恰 6 条唯一记录；从冻结 full manifest 独立重算确定性六单元样本 ID = `[286,374,1039,1392,1467,2566]`（ADE20K/COCO × Count/Relation + Omni3D × Depth/Distance 各 1，逐单元取 `sha256(sample_id)` 最小者），与文件观测一致。新增只读审计 `evaluation/audit_conservative_l2_guarded_rerank_v2_smoke.py`（`5a5db4ff…af29`）+ 聚焦测试 `evaluation/tests/test_audit_conservative_l2_guarded_rerank_v2_smoke.py`（`c6f96b0d…9858`）+ 中文报告 `reports/token_selector/2026-10-06-conservative-l2-guarded-reranker-v2-smoke.{json,md}`（`02286917…2599` / `4daebf1b…c793`）。**六项合同全部通过（共 153 项子检查）**：(1) 六唯一确定性 ID 每单元 1 且逐单元 sha256 最小者；(2) 每条 `record_sha256` 用冻结 runner 规则重算一致、`manifest_sha256`/config/checkpoint/source pin 与全部 leakage flag 全过，**36 个非空 config pin 现场复核字节一致**（含 v1/v2 核心/集成/runner/汇总器/测试/config、manifest、dense BF16、修正整数 L2、prompt 融合参考、3 个部署 checkpoint；唯一 null = 未生成的 full2638 新臂，且该路径不存在）；(3) `286` 为合法 `G=64<=96` identity bypass（K=G、real=G、context=0、swap=0、selected=range(G)），其余 5 个压缩样本精确 `K=96`/`real=87`/`context=9`、selected 严格升序唯一且在界；(4) anchor 冻结（9 个、每区域 1 个、⊆baseline∩reranked、从不被移除）、每次替换同区域一对一、每区域 real 计数 before==after 现场重算一致、每 context 区域剩余未选 real 非空（9 context 覆盖 9 区域）、`reranked_selected_real` 恰为 real 槽且 context 槽 = selected−reranked；(5) 每 swap `legal_gain == semantic_gain - l2_loss == utility_added - utility_removed > 0`、压缩样本 swap_count 全为 16 且 `<=S=16`、8 个 swap 数组长度一致、selected/patch(=`group*4+j`)/source_type/region_id/valid/baseline/reranked real 及 audit 数组 hash 全部现场重算一致、`combined_scores_sha256`/`score_vector_length=G`、`block_input_shapes` 24 段；(6) 四臂输出全部可解析，描述性逐样本计数 v2 **4/6**、dense **4/6**、修正整数 L2 **4/6**、既往 prompt 融合 **3/6**（v2 与 dense/修正 L2 逐样本答案 6/6 一致、与 prompt 融合 5/6 一致，差异样本 `374`）。**口径**：`gpu_runs=0`（本审计任务），`audited_gpu_runs=1`、`audited_samples=6`（被审计 raw 为此前宿主 GPU 生成）；CPU-only、只读、无 GPU/模型权重/网络/训练/依赖安装、未改 v1/v2 源码/config/raw JSONL 或既往报告。**明确声明**：4/6 不是精度估计也不是门；首个样本 3.8937 s 含首次执行开销，六个计时都不是稳态时延证据；软件 selector roundtrip 不是 FPGA 时延也不是加速证据；无论文/外部验证主张。**验证**：`py_compile` 2 文件 OK、聚焦 **26 OK**、全量 `evaluation/tests` **1822 OK**（基线 1796 + 26）、真实退出码 0。**下一决策**：`CONTRACT_SMOKE_PASS` 仅允许执行一次冻结的 full2638 发展性运行（命令见 config/预注册报告）；不得据本冒烟作质量判定。

更新时间：2026-10-06（**v2 六单元合同冒烟 CPU-only 只读审计 → `CONTRACT_SMOKE_PASS`**：raw SHA `e7648d9b…5916` 现场复核、6 唯一记录；从 frozen manifest 重算六 ID `[286,374,1039,1392,1467,2566]`；六项合同 153 子检查全过——36 pins 字节一致（v1/v2 代码/config/raw 未改）、`286` 合法 G64 bypass、5 压缩样本 K96/87+9、anchor 冻结/同区域一对一/区域计数不变/context 非空、每 swap `legal_gain=semantic_gain-l2_loss=utility_added-utility_removed>0` 且 swap_count=16<=S、全部 hash 现场重算；描述性 v2/dense/修正 L2 4/6、prompt 融合 3/6（**非精度、非门**）；首样本 3.8937 s 含首次执行开销、六计时非稳态时延、软件 roundtrip 非 FPGA 时延/加速；新增审计 `5a5db4ff…` + 测试 `c6f96b0d…` + 中文报告 `02286917…`/`4daebf1b…`；聚焦 26、全量 1822 OK、退出 0；`gpu_runs=0`、`audited_gpu_runs=1`、`audited_samples=6`；仅允许一次冻结 full2638 发展性运行，无论文/外部验证主张）

- [x] 2026-10-06 **v1 执行前设计 STOP + L2 守卫整数重排 v2 预注册完成（CPU-only；无 GPU/模型权重/网络/训练/依赖安装；未改原模型/app；v1 源码/配置/报告字节不变且从未运行；未生成任何新臂输出）→ v1 `PREEXECUTION_DESIGN_STOP` + v2 `PREREGISTERED_NOT_EXECUTED`**：独立复核确认预注册 v1 排序量 `(semantic_q15[j]-L2_q15[j])-(semantic_q15[i]-L2_q15[i])` 展开为 `semantic_gain + l2_loss`，**符号错误地奖励 L2 损失**；反例 `semantic_i=semantic_j=10000`、`L2_i=30000`、`L2_j=1000` → v1 gain `29000>0` 错误接受、修正 `legal_gain=-29000<0` 正确拒绝。**不修补、不运行 v1**；新增中文 STOP 报告 `reports/token_selector/2026-10-06-conservative-l2-residual-reranker-v1-preexecution-stop.{json,md}`（SHA `1993a2d4…bc95`/`4ae0f48e…4245`，pin v1 config/核心/集成/runner/汇总器/4 测试/预注册报告 11 个 SHA，`executed=false`、`gpu_runs=0`、`next_host_commands=[]`）。**唯一加性修正族 `conservative_l2_guarded_rerank_v2`**：`utility_q16=semantic_q15+l2_q15`（`[0,65534]`，int64）；`semantic_gain=sem_j-sem_i`；`l2_loss=l2_i-l2_j`；`legal_gain=semantic_gain-l2_loss=utility_added-utility_removed`，仅 `>0` 可交换；候选按 `utility_q16` 降序、平局取更小原始索引、每区域 `local_candidate_cap=16`；全局按 `legal_gain` 降序再 `(added_id,removed_id,region_id)` 升序，至多 `S=16` 个非冲突同区域一对一交换；anchor 冻结、每区域 real 计数不变、**K 恒 96**、当前全尺寸 **87 real + 9 context**、**context pooling 仍修正整数 L2**；无标签/答案/梯度/浮点 lambda/新权重/task-source ID/扫描。新增文件：核心 `evaluation/conservative_l2_guarded_reranker.py`（`9486771f…357b`；复用 v1 已验证的 `rebuild_hardware_selection`/`l2_combined_float` 与冻结 baseline 选择）、最小 B1 集成 `evaluation/hardware_regular_b1_guarded_rerank.py`（`0eaab5ed…15d2`；复用已测 v1 additive forward、仅重贴 v2 bypass 标签）、真实模型 runner `evaluation/run_hf_conservative_l2_guarded_rerank.py`（`834e5c81…2146`；离线 BF16/SDPA、greedy、`max_new_tokens=16`、六单元合同冒烟拒绝写 full2638、fail-closed 续跑）、fail-closed 汇总器 `evaluation/summarize_conservative_l2_guarded_rerank_full2638.py`（`7da4e93d…14d7`；缺 v2 臂即退出码 2 且不写报告）与 4 个聚焦测试（`87f1e4c8…eedd`/`95786e96…625d`/`ea309b6d…0898`/`7e744613…471c`）。预注册 config `configs/2026-10-06-conservative-l2-guarded-reranker-v2-full2638-development.preregistered.json`（`7499b6ea…74eb`；`status=PREREGISTERED_NOT_EXECUTED`、`executed=false`、新臂 SHA null、`go_score_floor_90pct_dense=0.9*0.8727298774444869=0.7854568897000382` 计算存储、bootstrap seed `20261006`/5000、先验融合 `0.7773617325561999` **仅描述不设门**），现场复核 **36 个非空 pin 全过**（含语义 Q0.15 源 `selector_int8_numerical_audit.py`/`selector_integer_scorer_audit.py`、修正 patch-L2 源 `hardware_regular_b1_corrected_l2.py`、3 个部署 checkpoint，**明确用整数 INT8+floor_sqrt 链而非旧 mean-of-patch 浮点 helper**）。中文 v2 预注册报告 `reports/token_selector/2026-10-06-conservative-l2-guarded-reranker-v2-prereg.{json,md}`（`d1d84fdf…dc5a`/`95cacd5f…73ae`）。**验证**：`py_compile` 8 文件 OK；聚焦 v2 **56 OK**；全量 `evaluation/tests` **1796 OK**（基线 1740 + 56）、真实退出码 0；v1 11 个文件前后 SHA 全等；缺 v2 臂汇总器退出 2 且不写报告；`gpu_runs=0`。通过仅表示可获取/运行一个新的未触碰外部空间推理基准，**不是**论文确认/官方未触碰验证/FPGA 执行/加速/时延证据；full2638 仍为受污染发展性 triage。

更新时间：2026-10-06（**v1 `PREEXECUTION_DESIGN_STOP` + v2 `conservative_l2_guarded_rerank_v2` 预注册完成**：v1 排序量符号错误（反例 `10000/10000/30000/1000` → v1 `+29000` 接受、v2 `-29000` 拒绝），v1 字节不变且从未运行，STOP 报告 `1993a2d4…`/`4ae0f48e…`；v2 `utility_q16=sem_q15+l2_q15`、`legal_gain=semantic_gain-l2_loss=utility_added-utility_removed>0`、候选按 utility 降序+低索引、`cap16`、全局 `S=16`、K96/87+9、修正 L2 池化；新增核心/集成/runner/汇总器+4 测试 `9486771f…`/`0eaab5ed…`/`834e5c81…`/`7da4e93d…` 与 `87f1e4c8…`/`95786e96…`/`ea309b6d…`/`7e744613…`；config `7499b6ea…`、36 pins 复核通过、修正门 `>=0.9*0.8727298774444869=0.7854568897000382`、seed 20261006/5000、CI 下界 `>=0`、先验 `0.7773617325561999` 仅描述；中文 v2 预注册 `d1d84fdf…`/`95cacd5f…`；聚焦 56、全量 1796 OK、退出 0、缺臂 fail-closed 退出 2 不写报告；CPU-only、`gpu_runs=0`、未生成新臂输出；仅发展性、非论文/部署/时延/加速主张）

- [x] 2026-10-05 **保守修正 L2 主干 + 有界整数任务残差重排 v1 完整预注册包就绪（CPU-only；无 GPU/模型权重/网络/训练/依赖安装；未改原模型/app、冻结历史 config/报告/raw 输出；未改 `hardware_regular_b1_integration.py` 等冻结评估代码；新臂输出未生成）→ `PREREGISTERED_NOT_EXECUTED`**：实现唯一新加性选择族 **`conservative_l2_residual_rerank_v1`**。合同：基线 real 选择取自既有修正整数 L2 K96 路径（每非空 3x3 区域 1 个 L2 anchor + 既有动态 context 预留）；**anchor 全部冻结、从不替换**；每区域被选 real 数不变，故每次替换都是**同区域一对一交换**，不会消耗该区域预留的未选 context 分组；有符号残差严格定义 `semantic_norm_q15 - corrected_l2_norm_q15`（复用已接受整数语义链 seed-20261003 Q0.15 + 修正整数 patch-L2 Q0.15；**无**标签/答案 token/梯度/浮点 lambda/学习新权重/task-source ID）；每区域候选枚举受既有 `local_candidate_cap=16` 限制，平局按**更小原始分组索引**确定性打破，只取严格正残差增益；全局至多 `S=16` 个最高正、非冲突同区域交换（不足 16 则少做；`S` 是独立固定硬件工作量子，**不是** local candidate cap、也非从已停止 pilot24 推断）；用既有布局合同重建 **87 real + 9 context**，**context pooling 权重仍基于修正整数 L2**，Prompt 只影响有界残差交换；任何违约 fail closed；不做任何 `S/K/cut/anchor/公式/阈值/分数混合` 扫描。**新增文件**：核心 `evaluation/conservative_l2_residual_reranker.py`（SHA `64b69975…5bf4`）、独立 B1 集成 `evaluation/hardware_regular_b1_residual_rerank.py`（`fc9c130a…5026`；因冻结 selector 的单一 `combined_scores` 同时驱动选择排名与 context 池化，无法表达本机制，故新文件镜像冻结 forward、仅替换选择步；0-swap 时与冻结路径逐元素一致）、真实模型 runner `evaluation/run_hf_conservative_l2_residual_rerank.py`（`ffd41f46…cb1c`；复用冻结修正 L2 runner 的 provenance/resume 机制，支持 `--smoke` 六单元合同冒烟 + 可续跑 full2638；本地模型 BF16/SDPA、greedy、`max_new_tokens=16`、`HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`、计时段 `torch.cuda.synchronize`、模型加载不计入稳态计时、raw 逐样本 JSONL）、fail-closed 汇总器 `evaluation/summarize_conservative_l2_residual_rerank_full2638.py`（`b7f71603…1c79`；严格校验 2638 唯一 ID、K96/87+9 布局或合法 bypass、swap 审计、record/config/manifest SHA、泄漏 flag；缺新臂即退出码 2 且**不写报告**）与 4 个 CPU 聚焦测试（`5712c95c…a1c8e` / `a116a132…db04` / `7f8b018d…a806` / `9f92dfca…5545`）。预注册 config `configs/2026-10-05-conservative-l2-residual-reranker-full2638-development.preregistered.json`（SHA `6b158d22…7512`；`status=PREREGISTERED_NOT_EXECUTED`、`executed=false`、新臂 SHA null），**30 个输入现场 SHA256 pin 全部复核通过**（full manifest/dense/corrected-L2 float+integer/prompt/旧 current integer/修正 L2 正式报告与 config/全部新代码与 4 测试/既有 selector 与 audit 模块/evaluator/metrics/`evaluation_protocol.json`/3 个部署 checkpoint）；中文预注册报告 `reports/token_selector/2026-10-05-conservative-l2-residual-reranker-prereg.{json,md}`（`1a936b8d…1c3c` / `67fdd032…56cf`）。**发展性门（全满足才 `DEVELOPMENT_GO_EXTERNAL_VALIDATION`）**：2638 全完成且可解析；每压缩样本精确 K96 与冻结 real/context 布局（合法 `G<=K` bypass 单独计数）；官方 `cv_bench_accuracy >= 0.78545688966`（dense 90%）；点估计 `>= 0.777362`（最强同预算发展基线）；新臂减修正整数 L2 的配对 bootstrap 95% CI 下界 `>= 0.0`（5000 次、seed `20261005`、按 `task::source_dataset` 分层）；无禁用标签/答案/task/source 泄漏；否则 `DEVELOPMENT_STOP`。通过仅为**获取/运行新的未触碰外部空间推理基准**的许可，**不是**论文确认/可部署性/FPGA 时延主张；看到 full2638 后不得调 v1。**验证**：`py_compile` 4 文件 OK、聚焦 **37 OK**、全量 `evaluation/tests` **1740 OK**（基线 1703 + 37）、17 个冻结先验输入逐字节前后一致（0 mismatch）、汇总器缺新臂实测 `FAIL_CLOSED` 退出码 2 且无报告写出。`gpu_runs=0`。**下一道门（宿主、尚未执行）**：先 `--smoke` 六单元合同冒烟，再跑一次冻结 v1 full2638（命令见中文报告与 config；`python evaluation/summarize_conservative_l2_residual_rerank_full2638.py` 出判定）。

更新时间：2026-10-05（**conservative L2-backbone residual task-aware re-ranking v1 预注册完成 → `PREREGISTERED_NOT_EXECUTED`**：唯一新加性选择族 `conservative_l2_residual_rerank_v1`；残差 `semantic_norm_q15 - corrected_l2_norm_q15`；anchor 冻结、同区域一对一、每区域计数不变、候选 cap16、全局 `S=16`、K 恒 96、87 real+9 context、context 池化仍修正整数 L2；新增核心/集成/runner/汇总器 + 4 测试（`64b69975…`/`fc9c130a…`/`ffd41f46…`/`b7f71603…` + `5712c95c…`/`a116a132…`/`7f8b018d…`/`9f92dfca…`）；config `6b158d22…`、30 pins 复核通过、中文预注册 json/md `1a936b8d…`/`67fdd032…`；门 90% dense `0.78545688966`、先验 `0.777362`、CI 下界 `>=0`（seed 20261005/5000）、无泄漏；聚焦 37、全量 1740 OK、冻结先验 0 mismatch、缺新臂 fail-closed 退出 2 不写报告；CPU-only、`gpu_runs=0`、未生成新臂输出；仅发展性、非论文/部署/时延/加速主张）

- [x] 2026-10-05 **B1 answer-margin oracle challenge/control pilot24 v2 执行完成 → `PILOT_STOP`，并完成 CPU-only 只读 STOP 事后尸检 → `PILOT_STOP_POSTMORTEM_COMPLETE`**：冻结执行结果 challenge 修复 **10/12**（≥8 通过）、control 回归 **3/12**（>1 失败）、oracle parse **24/24**、精确 K96/87+9 布局 **24/24**，因 control 回归门槛不合格，最终 **`PILOT_STOP`**（不推翻、不重调、不再论证）。新增只读尸检脚本 `evaluation/analyze_b1_answer_margin_oracle_stop_postmortem.py`（SHA `26f2f4c9…5ba0`；fail-closed：5 个冻结输入硬编码 SHA256、24 个唯一同 ID、oracle/control `record_sha256` 与全部 `*_sha256` 数组字段、`saliency_sha256`、冻结迁移 10/3/9/2、精确 96/87/9 布局、两臂 G 一致、index/source_type 一致性全部现场复核通过）与聚焦测试 `evaluation/tests/test_analyze_b1_answer_margin_oracle_stop_postmortem.py`（SHA `a3707c2e…a755`）。产物 `reports/token_selector/2026-10-05-b1-answer-margin-oracle-stop-postmortem.{json,csv,md}`（SHA json `9d438c11…58cb`、csv `c74f4b85…9582`、md `b1a0cfcb…99b1`）。关键描述统计（仅探索性、无 CI/p 值）：24 样本全槽 Jaccard 均值 **0.238592**、real 槽 Jaccard 均值 **0.206124**（Oracle 与 corrected L2 选择差异大）；Oracle 保留的绝对梯度 saliency 质量相对 L2 的配对差值 **24/24 全为正**（均值 **0.313594**），但失败组（R→W + 未修复 challenge）均值 **0.239897** 低于 unchanged-correct 的 **0.352701**；3 个回归样本 **334/471/2518**（ADE20K Count / ADE20K Relation / Omni3D Distance），2 个未修复 challenge **425/969**（ADE20K Relation / COCO Count，后者 margin 7.5）。逐项审计：低 dense margin **不能**充分解释（失败含 7.5 margin）、高 selection churn **未获观测支持**（未修复组 real Jaccard 最高 0.3108）、无符号绝对 `|grad×activation|` 可能是原因但 **未证明**（plausible-but-unproven）、context/anchor 规则交互 **无法**从现有记录隔离（无 per-slot 分数、无消融臂）。机器可读决策边界：直接 absolute-gradient 全替换 teacher **保持 STOPPED**；最多推荐一个下一机制类 = **conservative L2-backbone / residual task-aware re-ranking**，状态为 **unproven_hypothesis_for_next_falsifiable_test**，**不指定也不调**任何数值混合/阈值/K/anchor。口径：CPU-only、只读；无 GPU/模型权重/网络/训练；`gpu_runs=0`；outcome/label + ground-truth 双重泄漏，非代表性、不可部署、禁止官方绝对精度/部署/FPGA 加速/代表性采样主张。**验证**：`py_compile` 2 文件 OK、聚焦 **29 OK**、全量 `evaluation/tests` **1703 OK**（基线 1674 + 29）。**下一道门**：若继续，只能把该 residual re-ranking 作为新的、独立预注册的可证伪实验；禁止在 24 样本上调参，禁止用冻结 STOP 结果宣称精度。

更新时间：2026-10-05（**B1 answer-margin oracle challenge/control pilot24 v2 执行 → `PILOT_STOP`；STOP 事后尸检完成 → `PILOT_STOP_POSTMORTEM_COMPLETE`**：修复 10/12、回归 3/12、parse 24/24、布局 24/24 → STOP；全槽 Jaccard 0.238592、real 槽 0.206124、保留质量差值 24/24 为正（均值 0.313594）；回归 334/471/2518、未修复 425/969；低 margin/churn/无符号梯度/context-anchor 逐项审计，仅无符号梯度为 plausible-but-unproven；决策边界保持 absolute-gradient 全替换 STOPPED，唯一候选 conservative L2-backbone residual task-aware re-ranking（unproven hypothesis，无数值混合）；新增脚本 `26f2f4c9…` + 测试 `a3707c2e…`，聚焦 29、全量 1703 OK；CPU-only、只读、gpu_runs=0；非代表性、不可部署，禁止官方精度/部署/加速/代表性主张）

- [x] 2026-10-05 **B1 answer-margin oracle challenge/control pilot24 v2 完整预注册包就绪（CPU-only；无 GPU/模型权重/网络/训练/依赖安装；未改原模型/app/冻结历史 config 与既有结果；oracle 输出未生成）→ `PREREGISTERED_NOT_EXECUTED`**：新增共享纯合同 `evaluation/b1_answer_margin_oracle_challenge_control.py`（SHA `3b45059a…7fa9`；六单元 `source×task` 每单元 4 样本 = 2 challenge + 2 control、确定性 SHA256 排序、约束 Omni3D 子数据集覆盖、outcome/label leakage 声明、门常量 8/12、1/12、24/24）、选样脚本 `evaluation/select_b1_answer_margin_oracle_challenge_control_pilot24.py`（`5630dd5c…b5ce`）、fail-closed decision-only 汇总器 `evaluation/summarize_b1_answer_margin_oracle_challenge_control_pilot.py`（`0c4b5c86…96fd`）与 4 个 CPU 聚焦测试（`8099f924…7216`/`e6fd3ad0…ae45`/`5f6a1a0f…0592`/`f9749857…a818`）；**复用既有 GPU runner `evaluation/run_b1_answer_margin_oracle_pilot.py`，未复制任何模型前向代码**。固定 24 样本 manifest `datasets/2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24.jsonl`（SHA `0bbe413d…a7ca`）：单元 ADE20K/COCO × Count/Relation + Omni3D × Depth/Distance 各 4（2+2），样本 ID `[229,290,334,305,519,425,439,471,748,969,719,636,1406,1221,1234,1328,1701,1814,1908,1932,2108,2178,2566,2518]`；排除 `sample_id=1` + screening100 + 同图反事实100 + 随机 v1 pilot24，并集 **223**；用冻结 dense `evaluation/results/cv_bench_l2_0875_full2638_20260916/shallow_r1000.jsonl` 与 corrected integer-L2 `artifacts/token_selector/2026-10-04-b1-corrected-l2-prompt-ablation-full2638/corrected_l2_integer_k96_full2638.jsonl` 官方 `parse_choice`/`answer_letter` 分层（**outcome/label leakage，明确非代表性、不可部署、禁止官方绝对精度主张**），真实本地 tokenizer 分叉合同 + 权威 processor `G>96`：dense-correct 候选 **2107**，标签分叉拒绝 0，G≤96 拒绝 90；按 `sha256(sample_id)` 升序每层前 2；Omni3D naive 仅 Hypersim2/nuScenes6/SUNRGBD0，触发确定性约束选样（最小化全局有序 SHA-rank 元组）得 Hypersim2/SUNRGBD2/nuScenes4，失败即 fail closed。预注册 config `configs/2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24.preregistered.json`（SHA `a0e083b0…1de6`）pin 全部输入/脚本/runner/汇总器/测试/manifest/dense/corrected-L2 SHA（16 pins 现场复核 `pins_verified=true`）；中文预注册报告 `reports/token_selector/2026-10-05-b1-answer-margin-oracle-challenge-control-pilot-prereg.{json,md}`（`db4507ba…00f3`/`71939e1c…c028`，`status=PREREGISTERED_NOT_EXECUTED`、`gpu_runs=0`）。固定合同：本地 RoboBrain2.5-4B、BF16/SDPA、greedy、max_new_tokens=16、B1 after block0、K96、region_grid=3、local_cap=16；oracle saliency 经既有 `integer_scorer` 合同注入，布局 output=96/real=87/context=9 原位置（非 `fixed_indices`）。汇总门：challenge 修复 ≥8/12、control 回归 ≤1/12、oracle parse 24/24、精确 K96 布局 24/24、全部哈希/ID/分层标签通过 → GO 否则 STOP，**不用 CI、不用显著性/「精度提升已确认」措辞**。**验证**：`py_compile` 7 文件 OK；聚焦 **51 OK**；全量 `evaluation/tests` **1674 OK**（基线 1623 + 51）；汇总器无 oracle 输出时 `FAIL_CLOSED` 退出码 2 且不写正式报告；以真实 config/manifest/dense/control + 合成 oracle 跑通 GO=12/0/24/24。**下一道门（宿主、尚未执行）**：`python evaluation/run_b1_answer_margin_oracle_pilot.py --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --model-label RoboBrain2.5-4B-BF16 --manifest datasets/2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24.jsonl --config configs/2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24.preregistered.json --output artifacts/token_selector/2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24/oracle_answer_margin_b1_saliency_k96.jsonl`，随后 `python evaluation/summarize_b1_answer_margin_oracle_challenge_control_pilot.py`。

更新时间：2026-10-05（**B1 answer-margin oracle challenge/control pilot24 v2 预注册完成 → `PREREGISTERED_NOT_EXECUTED`**：六单元各 2 challenge + 2 control；manifest `0bbe413d…a7ca`、config `a0e083b0…1de6`；排除 223；dense-correct 2107、G≤96 拒 90；Omni3D 约束选样 H2/S2/N4；复用既有 runner；门 8/12、1/12、24/24、布局 24/24，无 CI；outcome/label + ground-truth 双重泄漏，不可部署、非代表性；gpu_runs=0；聚焦 51、全量 1674 OK；脚本+测试+manifest+config+中文预注册报告就绪）

- [x] 2026-10-05 **B1 answer-margin label-aware oracle K96 pilot 完整预注册包就绪（CPU-only；无 GPU/模型权重/网络/训练/依赖安装；未改模型/app/冻结历史 config 与既有结果；oracle 输出未生成）→ `PREREGISTERED_NOT_EXECUTED`**：新增共享纯合同 `evaluation/b1_answer_margin_oracle.py`（`|grad×activation|` group saliency、saliency 注入现有 `select_hardware_regular_groups`/`forward_hardware_regular_b1_vision` 合同、87+9/K96 布局校验、retained mass/Jaccard 诊断）、选样脚本 `evaluation/select_b1_answer_margin_oracle_pilot24.py`、GPU runner `evaluation/run_b1_answer_margin_oracle_pilot.py`（**本轮未执行**）、fail-closed 汇总器 `evaluation/summarize_b1_answer_margin_oracle_pilot.py` 与 3 个 CPU 测试；固定 24 样本 `datasets/2026-10-05-b1-answer-margin-oracle-pilot24.jsonl`（SHA `9c8d15a6…13c9`）按 ADE20K Validation Set / COCO 2017 Validation Set / Omni3D_Hypersim 三个 `source_dataset` 各 8，仅用原 manifest 元数据 + 真实本地 tokenizer 标签分叉合同 + `full_llm_visual_tokens>96` + `sha256(sample_id)` 升序，排除 `sample_id=1`、screening100 全部 ID、同图反事实 100 全部 ID（并集 199），**不读 dense/L2 正确性**；预注册 config `configs/2026-10-05-b1-answer-margin-oracle-pilot24.preregistered.json`（SHA `59d7e53f…b6a9`）pin 全部输入/脚本/测试/manifest SHA；中文报告 `reports/token_selector/2026-10-05-b1-answer-margin-oracle-pilot-prereg.{md,json}`。固定合同：本地 RoboBrain2.5-4B、BF16/SDPA、greedy、max_new_tokens=16、B1 after block0、K96、region_grid=3、local_cap=16；每样本先用完整视觉序列 + 官方正确答案分叉 token 计算 B1 saliency（模型参数全冻结、无 optimizer、grad 全 None），再把 saliency 作为 score 注入既有合同；fake tensor 已验证 G>96 → output 96 / real 87 / context 9 且为原位置 patch（非 fixed_indices）。汇总门：oracle−control net≥3、W→R≥4、R→W≤1、oracle parse≥23/24、24/24 精确 K96 布局全过 → GO 否则 STOP，**不用 CI、不用「精度提升已确认」措辞**；明确 ground-truth answer leakage、不可部署、非论文精度、GPU 未运行（`gpu_runs=0`）。**验证**：`py_compile` 4 文件 + 3 测试 OK；聚焦 **51 OK**；全量 `evaluation/tests` **1623 OK**（基线 1572 + 51）。**下一道门（宿主、尚未执行）**：`python evaluation/run_b1_answer_margin_oracle_pilot.py --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --model-label RoboBrain2.5-4B-BF16 --manifest datasets/2026-10-05-b1-answer-margin-oracle-pilot24.jsonl --config configs/2026-10-05-b1-answer-margin-oracle-pilot24.preregistered.json --output artifacts/token_selector/2026-10-05-b1-answer-margin-oracle-pilot24/oracle_answer_margin_b1_saliency_k96.jsonl`，随后 `python evaluation/summarize_b1_answer_margin_oracle_pilot.py`。

更新时间：2026-10-05（**B1 answer-margin oracle K96 pilot 预注册完成 → `PREREGISTERED_NOT_EXECUTED`**：24 样本 = ADE20K/COCO/Omni3D_Hypersim 各 8，manifest `9c8d15a6…`、config `59d7e53f…`；仅元数据+真实 tokenizer 分叉+G>96+sha256 排序，排除 sample_id=1/screening100/反事实100 共 199，不读正确性；固定 B1/K96/grid3/cap16/greedy/max16；saliency 经既有 score 合同注入，fake tensor 验证 96/87/9 原位置；汇总门 net≥3、W→R≥4、R→W≤1、parse≥23/24、布局全过，否则 STOP，无 CI；GPU 未运行、gpu_runs=0；聚焦 51、全量 1623 OK；脚本+测试+manifest+config+中文预注册报告就绪）

- [x] 2026-10-05 **B1 真实模型单图答案 choice-margin GPU smoke 完成 → `REAL_MODEL_AUTOGRAD_SMOKE_PASS`**：新增 `evaluation/run_b1_answer_margin_gpu_smoke.py`（SHA `39805aa5…8492`）与 CPU 单测 `evaluation/tests/test_run_b1_answer_margin_gpu_smoke.py`（SHA `09c24a7f…9fa8`，**22 聚焦测试 OK**）；宿主环境以本地 RoboBrain2.5-4B、BF16/SDPA、`cuda:0` 离线执行固定 `sample_id="1"` 一张图，未训练、无 optimizer、模型 713 个参数张量全部 `requires_grad=False` 且 backward 后 `parameter.grad=None`。真实 tokenizer 将 `"(A)"…"(F)"` 分解为 6 个互不重复分叉 token 加共同后缀 `)`；正确答案 `(E)` 的 margin 为 **+2.0**，模型在 6 个候选中也预测 `(E)`。真实链路为 B1 `prune_after_layer=0`、`retention=1.0`（不裁 token）：`image_grid_thw=[1,42,64]`、B1 hidden `[2688,1024]`、G=672、merge_unit=4；B1 梯度存在、有限且非零，范数 **12.834799**，672/672 个 group 的 `|grad×activation|` saliency 均有限且非零，13/13 硬检查通过。一次性诊断计时：processor **88.575 ms**、真实 forward **610.485 ms**、backward **702.670 ms**、总 smoke **1458.584 ms**；peak allocated/reserved 为 **13,821,010,944 / 13,977,518,080 bytes**。这些数字包含梯度图且只有一次，不是稳态推理延时、不是 selector 加速。正式报告 `reports/token_selector/2026-10-05-b1-answer-margin-gpu-smoke.{json,md}`，SHA 分别 `27238ea2…a425` / `eb29eb26…a167`；Harness 沙箱曾因隐藏 `/dev/nvidia*` 写出临时 `FAIL_CLOSED`，已由宿主真实执行覆盖。**口径**：这里只证明真实答案边际可反传至 B1，不是 selector 精度证据；不得用 full2638 标签训练后再对同集宣称结果。**验证**：`py_compile` 2 文件 OK、聚焦 **22 OK**、全量 `evaluation/tests` **1572 OK**（基线 1550 + 22）；运行后 GPU 已释放，仍仅有 `seey6` PID16507 占 382 MiB。**下一道门**：先做小规模、明确标记为 label-aware oracle 上界的配对诊断，检验该 saliency 在相同 B1/K96 下是否比 corrected integer L2 保留更多官方答案信息；若无优势立即停止，不进入训练。

更新时间：2026-10-05（**B1 真实模型单图 answer choice-margin GPU smoke → `REAL_MODEL_AUTOGRAD_SMOKE_PASS`**：宿主 A6000 离线真实运行，A-F 分叉 token 合同成立；margin +2.0 且预测 `(E)` 正确；B1 `[2688,1024]`、G=672，梯度存在/有限/非零、范数 12.834799，672/672 group saliency 有限非零，13/13 检查通过；模型参数全冻结且 grad=None；一次性 forward/backward 610.485/702.670 ms、峰值 allocated 13.82 GB，仅为自动微分 smoke，非 selector 精度或稳态时延证据；22/1572 测试 OK）

- [x] 2026-10-05 **B1 官方答案 choice-margin 自动微分预检（CPU-only、无模型权重/GPU/网络/训练/依赖安装；未改原模型、app、冻结 config、历史报告与现有实验结果）→ `PREFLIGHT_PASS`**：新增 `evaluation/answer_margin_b1_preflight.py`（SHA `189b2694…a7d9`）+ `evaluation/tests/test_answer_margin_b1_preflight.py`（SHA `05d8dad7…2450`，**18 聚焦测试 OK**）+ 中文报告与机器可读 JSON `reports/token_selector/2026-10-05-b1-answer-margin-autograd-preflight.{md,json}`（JSON SHA `d44a8541…ec8d`、MD SHA `2c79571e…eb0f`）。核心合同：`differentiable_group_gather(hidden_states[P,D], group_indices[K], merge_unit=4) -> [K*4,D]` 只用纯 `index_select` 保持自动微分（不 detach/numpy/CPU 重建），第 `k*4+j` 行严格等于 `hidden[g_k*4+j]` 且保持组顺序；`choice_margin` 支持每题可变选项数（A-D / A-F / 任意给定 token ids），定义为正确选项 logit 减其余选项最大 logit，严格 fail-closed 校验 shape、索引范围与题内重复 token id。CPU fake 张量与冻结 fake lm_head 串联后 B1 hidden 梯度存在、有限且非零（范数 `0.3603646755218506`）；明确负对照显示 `detach` 后原 B1 hidden **得不到梯度**；内部 10/10 检查通过。**口径**：`engineering_feasibility_only=true`、`is_selector_accuracy_evidence=false`，**不是 selector 精度证据**；官方答案标签不得在 CV-Bench full2638 上训练再对同集宣称结果；后续 GPU smoke 只能一张图、不得训练。`py_compile` 2 文件 OK、聚焦 18 OK、全量 `evaluation/tests` **1550 OK**（基线 1532 + 18）。**下一道门**：真实模型单图 GPU smoke，复用同一 gather+margin 合同，不训练、不宣称 CV-Bench 精度。

更新时间：2026-10-05（**B1 答案 choice-margin 自动微分预检完成 → `PREFLIGHT_PASS`**：新增可微 `differentiable_group_gather`（纯 index_select、不 detach/numpy/CPU 重建、顺序保持）+ 支持每题可变选项数与任意 token id 的 `choice_margin`（正确 logit − 其余最大 logit，严格 fail-closed）；CPU fake 张量/fake lm_head 得到有限非零 B1 hidden 梯度，detach 负对照无梯度；10/10 检查、聚焦 18、全量 1550 OK；明确仅为工程可行性、非 selector 精度证据，禁止 full2638 标签训练后同集宣称，GPU smoke 限一张图且不得训练）

- [x] 2026-10-05 **同图双分辨率反事实四臂真实 GPU 实验完成 -> `RESOLUTION_DEPENDENT_PROMPT_GAIN_NOT_CONFIRMED`**：在单张 A6000 上以本地 RoboBrain2.5-4B、BF16/SDPA、greedy、max_new_tokens=16 串行完成 `corrected_l2_integer` / `prompt_plus_corrected_l2_integer` x low/high 共 **4 臂，每臂 100 个相同 ADE20K 样本**；离线、无训练，未修改预注册合同或 raw JSONL。预注册唯一主估计量 `mean_i[(Prompt-Integer)_high-(Prompt-Integer)_low]` = **-0.07**，original-stratum 配对 bootstrap 95% CI **[-0.20,+0.06]**，未达到 CI 下界 >0 的确认门。low：integer/prompt accuracy **0.56/0.62**、W->R/R->W **16/10**、净 **+6**；high：**0.53/0.52**、**11/12**、净 **-1**。因此不能确认 Prompt 增益随 `G>512` 或更高视觉分辨率增强；先前 full2638 OOF `G>512` 门的正收益更可能来自数据源/内容相关性，**停止把该门控作为因果机制或主要创新点，也不继续调 G 阈值**。正式报告：`reports/cv_bench/2026-10-04-same-image-resolution-counterfactual.{json,csv,md}`，SHA256 分别 `f8c4fe01...f541` / `dce436b5...1ead` / `d2260ad3...1af4`；execution receipt `0fe87a61...d56a`，其 15 个带 SHA 的内部产物现场复核全部一致。测试：汇总器聚焦 **50 OK**、相关 **183 OK**、全量 `evaluation/tests` **1532 OK**。当前研究判断：K96 硬件规则选择器仍可作为压缩基线/次要组成，但 Prompt 条件分支及 `G>512` 路由尚无可信因果依据；下一步须改为新的、可证伪的空间保持机制，而不是继续在同一门控上调参。

更新时间：2026-10-05（**同图双分辨率真实反事实完成**：四臂 x 100 A6000 BF16/SDPA；交互效应 -0.07，95% CI [-0.20,+0.06] -> 不确认高分辨率 Prompt 增益；停止把 `G>512` 门控作为主要创新机制；正式报告/receipt 已固化，50/183/1532 测试通过）

- [x] 2026-10-04 **同图双分辨率反事实 fail-closed 汇总器就绪（CPU-only、只读；无 GPU/模型加载/联网/训练/依赖安装；未改预注册 config、既有 gate 报告与 `configs/evaluation_protocol.json`）→ `SUMMARIZER_READY_GPU_RESULTS_NOT_EXECUTED`**：新增 `evaluation/summarize_same_image_resolution_counterfactual.py`（SHA `3cffeaec49909f267f112ba70329a85ce9c3b508d1faf49a3507160b80fc67c4`）+ `evaluation/tests/test_summarize_same_image_resolution_counterfactual.py`（SHA `0f9f0c3815470c83f6751c70d457a46116f6b7a88670bdc35b1effce417e4949`，**48 聚焦测试 OK**）。汇总器严格 fail-closed 校验四个输入（`corrected_l2_integer` / `prompt_plus_corrected_l2_integer` × `low`/`high`）：每文件恰好 100 个唯一 `sample_id` 且四集合一致并等于 separated manifest（`c53c7b7d…d4c5`）；manifest/config/gate-report/checkpoint（`1f1d379e…ca72ff`）/Wv（`126bd842…d89d3b`）/Wv-scale（`93585a3b…28b13f`）物理 SHA 与 arm / pixel_budget_label / min/max_pixels / K96 / B1（`prune_after_layer=0`、`cutpoint=B1_after_vision_block_0`）/ grid3 / local_cap16 / greedy（`do_sample=false`）/ max_new_tokens=16 冻结元数据一致；low 每样本 G<=512、high 每样本 G>512 且逐样本等于权威 processor dry-run 测量；同图 low/high 成对且同一 sample、同一 pixel budget 下 integer 与 prompt 两臂 `image_grid_thw` 与 `full_llm_visual_tokens` 完全一致（跨臂 grid 污染、low/high grid 未变化一律 fail closed）；重复/额外 ID、混臂、断点续跑污染一律拒绝。唯一主估计量 = 预注册 `mean_i[(Prompt−Integer)_high−(Prompt−Integer)_low]`（unweighted per-sample paired difference；单一 ADE20K 源内 `score(...)["cv_bench_accuracy"]` 未定义，**不计算**），按 original stratum 配对 bootstrap（seed 20261004 / repeats 5000 / 95%）；CI 下界 >0 → `RESOLUTION_DEPENDENT_PROMPT_GAIN_CONFIRMED`，否则 `RESOLUTION_DEPENDENT_PROMPT_GAIN_NOT_CONFIRMED`；输出各档 W→R / R→W / net、parse rate、逐样本 CSV、JSON 与中文 MD；不使用 pointing/grounding/nDCG/retained-mass，不作时延/FPGA 加速声明。**四臂文件缺失 → CLI 退出码 2 且不写任何正式结果**（实测 `FAIL_CLOSED: missing required input 'corrected_l2_integer::low'`，`reports/cv_bench/2026-10-04-same-image-resolution-counterfactual.{json,csv,md}` 未生成）。相关聚焦 181 OK、全量 `evaluation/tests` **1530 OK**（基线 1482 + 48）。**未执行**：四条 GPU 模型臂（两臂 × low/high）仍全部未运行、`model_runs=0`、无任何臂输出；正式结论 `PROMPT_GAIN_NOT_CONFIRMED_FULL2638` 不变。更新 receipt `reports/cv_bench/2026-10-04-same-image-resolution-counterfactual-execution-receipt.json`。**下一条命令**：先执行已记录的四条 GPU 臂，再运行 `python evaluation/summarize_same_image_resolution_counterfactual.py`。

更新时间：2026-10-04（**同图双分辨率反事实汇总器就绪 → `SUMMARIZER_READY_GPU_RESULTS_NOT_EXECUTED`**：新增 fail-closed 汇总器 `3cffeaec…` + 48 聚焦测试；严格校验四输入（100 唯一 ID × 四文件集合=separated manifest、manifest/config/gate/checkpoint/Wv 物理哈希与 arm/pixel budget/K96/B1/grid3/cap16/greedy/max_new_tokens=16 一致、low G<=512 与 high G>512 且等于权威 dry-run、同图成对且同预算跨臂 grid/G 一致、无混臂/续跑污染）；主估计量仅预注册 `mean_i[(Prompt−Integer)_high−(Prompt−Integer)_low]`，original-stratum 配对 bootstrap seed 20261004/5000，CI 下界>0 → CONFIRMED 否则 NOT_CONFIRMED；输出 W→R/R→W/net、parse rate、逐样本 CSV、JSON/中文 MD；四臂缺失即 fail closed 不写正式结果；相关聚焦 181 OK、全量 1530 OK；四条 GPU 臂仍未执行、正式结论不变）

- [x] 2026-10-04 **同图双分辨率反事实：权威 processor 分离门 dry run（CPU-only；只加载本地 Qwen3VLProcessor、**未加载模型权重/未用 GPU/未联网/未训练**；未改预注册预算/阈值/样本/评价规则、未改 `configs/evaluation_protocol.json`、未改冻结输入；未跑任何模型臂）→ `PROCESSOR_DRY_RUN_PASS_MODEL_NOT_EXECUTED`**：命令 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python evaluation/verify_same_image_resolution_gate_separation.py --mode processor --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B`。**结果 `mode=processor`、`authoritative=true`、`model_was_run=false`；100/100 分离、rejected/replaced/**unresolved 全 0**、`resolved=true`；G_low 范围 **64..504（全部 ≤512）**、G_high 范围 **520..3588（全部 >512）**；100 个唯一 `sample_id`；全部 `contract_checks` OK**。因零替换，权威 separated manifest 与预注册 manifest **逐字节一致**：`datasets/2026-10-04-same-image-resolution-counterfactual-ade20k100-separated.jsonl` SHA **`c53c7b7d…d4c5`**（= config pin，100 条）；报告 `reports/cv_bench/2026-10-04-same-image-resolution-counterfactual-gate-separation.json` SHA **`2d2abe2d…7021e`**；config SHA **`ae1ba239…857f83`**、manifest pin `c53c7b7d…d4c5`、reserve `b6d9333b…005c` 三处现场复核一致；`strata_without_reserve=[G>512::Relation::integer_wrong]`（本次无 drop，未触及 reserve）。**测试**：新增四组聚焦测试 **26+10+16+15 = 67 OK**（`test_analyze_gate_source_confounding` / `test_select_ade20k_resolution_counterfactual_samples` / `test_verify_same_image_resolution_gate_separation` / `test_run_hf_same_image_resolution_counterfactual`）+ `py_compile` **5 文件 OK**；本次未改任何测试或被测代码（无脚本 bug 需修）。**执行 receipt 更正**：原 receipt 中 selection/verify/runner 的 `test_count`（13/17/11）与实测 unittest 收集数（10/16/15）不符，三者总数 41 不变，已更正为实测值。**未执行**：四条 GPU 模型臂（`corrected_l2_integer` / `prompt_plus_corrected_l2_integer` × low/high）**全部未运行**，`model_runs=0`、无任何臂输出，正式结论 `PROMPT_GAIN_NOT_CONFIRMED_FULL2638` 不变。更新 receipt `reports/cv_bench/2026-10-04-same-image-resolution-counterfactual-execution-receipt.json`。**下一条命令（宿主、四条 GPU 臂，尚未执行）**：`python evaluation/run_hf_same_image_resolution_counterfactual.py --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --model-label RoboBrain2.5-4B-BF16 --manifest datasets/2026-10-04-same-image-resolution-counterfactual-ade20k100-separated.jsonl --arm corrected_l2_integer --pixel-budget low --output artifacts/token_selector/2026-10-04-same-image-resolution-counterfactual/corrected_l2_integer_low.jsonl`（再以 `--pixel-budget high`、`--arm prompt_plus_corrected_l2_integer` 重复，共四条）。

更新时间：2026-10-04（**同图双分辨率反事实 processor 分离门通过 → `PROCESSOR_DRY_RUN_PASS_MODEL_NOT_EXECUTED`**：权威 `--mode processor` 100/100 分离、0 拒绝/0 替换/0 未解、resolved=true；G_low 64..504 ≤512、G_high 520..3588 >512；零替换 → separated manifest 与预注册 manifest 同为 SHA `c53c7b7d…d4c5`；报告 `2d2abe2d…`、config `ae1ba239…`、reserve `b6d9333b…` pin 一致；四组聚焦测试 67 OK + py_compile 5 文件 OK（receipt test_count 由 13/17/11 更正为 10/16/15，总数 41 不变）；仅加载 processor、未加载权重/未用 GPU/未联网/未训练；四条 GPU 臂未执行、正式结论不变）

- [x] 2026-10-04 **G>512 门控数据源混杂量化 + 同图双分辨率反事实预注册（CPU-only、只读；无 GPU/联网/训练/依赖安装；未改冻结输入、既有报告与 `configs/evaluation_protocol.json`）→ `SOURCE_CONFOUNDING_NOT_RESOLVED` + `PREREGISTERED_NOT_EXECUTED`**：**目标一**新增 `evaluation/analyze_gate_source_confounding.py` + 26 聚焦测试。**结论 `SOURCE_CONFOUNDING_NOT_RESOLVED`**，正式结论 `PROMPT_GAIN_NOT_CONFIRMED_FULL2638` 不变（`formal_result_unchanged=true`）。G>512 门控与 `source_dataset` 严重混同：Cramer's V **0.880783**、互信息 **0.740323 bits**（五档 G-bin：V **0.665422** / MI **1.163543 bits**；**均仅探索性、不作门**）；门内 1008 / 门外 1630（占比 0.382108）。source×G-bin：ADE20K 633（G≤512 425 / G>512 208，**唯一跨门数据源**）、COCO 805 全门外、Hypersim 400 全门内（513-1024）、SUNRGBD 400 全门外（97-512）、nuScenes 400 全门内（>1024）。各源 Prompt−integer 无权重 Δ（分层配对 bootstrap CI）：ADE20K +0.015798 [-0.011058,+0.044234]、COCO +0.001242 [-0.024845,+0.028571]、Hypersim +0.062500 [+0.015000,+0.107500]、SUNRGBD −0.055000 [-0.082500,-0.027500]、nuScenes +0.030000 [-0.015000,+0.075000]。**焦点 ADE20K 同源 interaction = Δ(G>512) − Δ(G≤512) = +0.048077，95% 分层配对 bootstrap CI [-0.018824,+0.114977]（下界 ≤0，非全 >0）** → 门控效应无法与数据源身份区分；门控样本内官方 `cv_bench_accuracy` 0.786218（integer 0.766852 / prompt 0.777362，仅描述）。产物 `reports/cv_bench/2026-10-04-gate-source-confounding.{json,csv,md}` + execution receipt。**目标二（只搭建、未跑模型）**：由 `evaluation/select_ade20k_resolution_counterfactual_samples.py` 冻结 100 条 ADE20K 分层样本（原始 G 高低 × Count/Relation × integer 正确性共 8 层，seed 20261004；G≤512 53 / G>512 47，Count 54 / Relation 46）→ `datasets/2026-10-04-same-image-resolution-counterfactual-ade20k100.jsonl`（SHA `c53c7b7d…d4c5`）+ 533 reserve（`…-ade20k-reserve.jsonl`，SHA `b6d9333b…005c`）；预注册 config `configs/2026-10-04-same-image-resolution-counterfactual.preregistered.json`（low `max_pixels=524288` 强制 G≤512、high `min_pixels=525312` 强制 G≥513；K96/grid3/local_cap16/BF16/SDPA/greedy；主估计量 `mean_i[(Prompt−Integer)_high − (Prompt−Integer)_low]`，bootstrap seed 20261004/5000，CI 下界 >0 → CONFIRMED 否则 NOT_CONFIRMED）+ 中文计划 `reports/cv_bench/2026-10-04-same-image-resolution-counterfactual-plan.md` + 抽样报告 + runner 脚手架 `evaluation/run_hf_same_image_resolution_counterfactual.py`（强制 `--manifest` 与 dry-run 最终 manifest SHA 一致、逐记录强制 pixel budget）。**processor-only dry run 脚本 `evaluation/verify_same_image_resolution_gate_separation.py`：权威 `--mode processor` 待执行；`--mode analytic` 预测 100/100 分离（smart_resize 公式对 633 张 ADE20K 记录 G 逐张复核 **633/633 一致**），属非权威**；未分离样本按预注册规则从同层 reserve 剔除/替换，reserve 用尽即 fail closed。**未加载模型、未用 GPU、无任何臂输出。**

更新时间：2026-10-04（**G>512 数据源混杂量化完成 → `SOURCE_CONFOUNDING_NOT_RESOLVED`**：Cramer's V 0.880783、MI 0.740323 bits（仅探索性）；门内 1008/门外 1630；ADE20K 同源 interaction +0.048077 CI[-0.018824,+0.114977] 下界≤0，Hypersim +0.0625 [+0.015,+0.1075]、SUNRGBD −0.055 [-0.0825,-0.0275]；正式结论不变。**同图双分辨率反事实预注册完成 → `PREREGISTERED_NOT_EXECUTED`**：冻结 ADE20K100（`c53c7b7d…`）+ 533 reserve、固定 low/high pixel budget、processor dry run 待执行（analytic 预测 100/100）、runner fail-closed；新增 **67** 测试（26+13+17+11），相关 **206 OK**、全量 `evaluation/tests` **1482 OK**；CPU-only/只读、未改冻结输入/既有报告/`configs/evaluation_protocol.json`。**下一条命令**：`python evaluation/verify_same_image_resolution_gate_separation.py --mode processor --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B`，之后 `python evaluation/run_hf_same_image_resolution_counterfactual.py --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --model-label RoboBrain2.5-4B-BF16 --manifest datasets/2026-10-04-same-image-resolution-counterfactual-ade20k100-separated.jsonl --arm corrected_l2_integer --pixel-budget low --output artifacts/token_selector/2026-10-04-same-image-resolution-counterfactual/corrected_l2_integer_low.jsonl`）

- [x] 2026-10-04 **full2638 out-of-fold G-gated Prompt 探索性诊断（CPU-only、只读；无 GPU/联网/训练/依赖安装；未改权威输入/配置/既有报告与 `configs/evaluation_protocol.json`）→ OOF 策略 `prompt_if_G_gt_512`（5/5 fold 稳定）**：新增 `evaluation/analyze_full2638_oof_gated_prompt.py` + `evaluation/tests/test_analyze_full2638_oof_gated_prompt.py`（36 聚焦测试；按 `task::source_dataset` 分层 5-fold、seed 20261004、每样本恰属一个 held-out fold；候选=`never_prompt`/`always_prompt`/`prompt_if_G_gt_{96,256,512,1024}` 先于评分固定；每折仅用其余四折最大化官方 `cv_bench_accuracy`，平局先低 Prompt 使用率再固定顺序）。**OOF 主分 0.786218**（Prompt 使用率 **0.382108**，1008/2638）；never **0.766852**、always **0.777362**、dense **0.872730**（仅描述）。**OOF−never +0.019366**，95% CI **[+0.007391,+0.030968]**（整体 >0）；**OOF−always +0.008856**，95% CI **[-0.000686,+0.018654]**（含 0）；OOF−dense **−0.086512**（描述、无 bootstrap）。迁移：对 never W→R 133/R→W 86/净 +47；对 always 108/87/净 +21；分桶 W→R/R→W/净：97-256 0/0/0、257-512 0/0/0、513-1024 65/42/**+23**、>1024 68/44/**+24**。产物 `reports/cv_bench/2026-10-04-full2638-oof-gated-prompt.{json,csv,md}` + execution receipt（已随最小更正更新脚本/测试/报告哈希与测试计数）。测试：39 聚焦 + 41 full2638 summarizer + 30 异质性 + 4 CV-Bench scorer = **114 OK**，全量 `evaluation/tests` **1415 OK**。**2026-10-04 最小更正**：分组表末列标题由 `official Δ` 更正为 `OOF official` 并改为无正号渲染（该值本就是子集 OOF official 分数，不是差值；`0.914510`/`0.784807` 分别为 97-256 / 257-512 桶的子集 OOF official 分数），新增 3 个聚焦回归测试；**核心数字与结论不变**。**限制**：阈值发现与评估共用同一 full2638 语料（交叉验证），只是内部 OOF 估计，**不是未触碰外部评测**；提出论文/正式主张前必须在外部冻结数据集上重放冻结策略。

- [x] 2026-10-04 **full2638 Prompt 增益异质性事后诊断（CPU-only、只读；无 GPU/联网/训练/依赖安装；未改权威输入、既有配置/报告或 `configs/evaluation_protocol.json`）→ `PROMPT_GAIN_NOT_CONFIRMED_FULL2638` 正式结论不变**：新增 `evaluation/analyze_full2638_prompt_gain_heterogeneity.py`（严格按 `sample_id` 配对两臂并校验 SHA/record/config/索引哈希；仅用既有 `parse_choice`/`answer_letter`/`score`/`paired_bootstrap_delta`）与 30 个聚焦测试。**Prompt−integerL2 总体**：W→R **220**、R→W **194**、不变对 **1820**、不变错 **404**，净修正 **+26**（unweighted 配对差 **+0.009856**）；official `cv_bench_accuracy` Δ **+0.010510**（95% CI **[-0.005430,+0.025699]**，与冻结报告逐位一致）。**净增益是大额对向流残差**：task Count +24 / Distance +16 / Depth −1 / Relation −13；source_dataset Hypersim +25 / nuScenes +12 / ADE20K +10 / COCO +1 / SUNRGBD **−22**；pre-selector 桶（=`full_llm_visual_tokens`）97-256 −14 / 257-512 −7 / 513-1024 +23 / >1024 +24。identity bypass 143 样本迁移全 0；非 bypass 2495 样本索引精确匹配 **0**、Jaccard mean/min **0.285257/0.043478**。探索性分层 bootstrap（seed 20261004/5000、unweighted、**不作门**）仅 `source_dataset::Omni3D_SUNRGBD` 的 95% CI 整体 <0（n=400，Δ **−0.055**，CI **[-0.0825,-0.0275]**）。产物 `reports/cv_bench/2026-10-04-full2638-prompt-gain-heterogeneity.{json,csv,md}` + execution receipt；聚焦 30 OK、相关既有 102 OK、全量 `evaluation/tests` **1376 OK**；所有子组数字均标注事后探索、不构成新主张。

更新时间：2026-10-04（**full2638 Prompt 增益异质性事后诊断完成 → `PROMPT_GAIN_NOT_CONFIRMED_FULL2638` 不变**：总体 W→R/R→W/不变对/不变错=220/194/1820/404、净 +26、official Δ+0.010510 CI[-0.005430,+0.025699] 与冻结报告一致；净增益为对向流残差（Count +24、Hypersim +25 vs Relation −13、SUNRGBD −22）；bypass 143 全 0；非 bypass 2495 索引精确匹配 0、Jaccard 0.285257/0.043478；探索性分层 bootstrap 仅 SUNRGBD CI 整体 <0；pre-selector 桶 97-256 −14 / 257-512 −7 / 513-1024 +23 / >1024 +24；新增诊断脚本+30 聚焦测试，相关既有 102、全量 1376 OK；CPU-only/只读、未改权威输入/配置/既有报告）

- [x] 2026-10-04 **B1 corrected-L2 / Prompt full2638 三臂正式确认（A6000、BF16/SDPA、B1=after vision block 0、K96、region_grid3/local_cap16、greedy；真实模型推理，无训练/联网/上板）→ `PROMPT_GAIN_NOT_CONFIRMED_FULL2638`**：在不调参地升格 screening100（`CORRECTED_PROMPT_SELECTOR_SCREENING_PASS`）后，三臂唯一主分 `score(...)["cv_bench_accuracy"]` 为 corrected_l2_float **0.766117**、corrected_l2_integer **0.766852**、prompt_plus_corrected_l2_integer **0.777362**；dense **0.872730**、旧缺陷 integer **0.764869** 仅描述参考。**量化门**：integer−float **+0.000735**，95% CI **[-0.004192,+0.005833]**，下界 ≥ −0.02 ✅。**Prompt 增益**：prompt−integer **+0.010510**，95% CI **[-0.005430,+0.025699]**，下界 ≤ 0 ❌ 未确认。决策 **PROMPT_GAIN_NOT_CONFIRMED_FULL2638**（只用主分；dense/旧 integer 不得覆盖）。三臂各 2638 条、sample_id 集合一致、record/manifest/checkpoint/selected-index 哈希全过。产物 `reports/cv_bench/2026-10-04-b1-corrected-l2-prompt-ablation-full2638.{json,csv,md}` + `…-full2638-execution-receipt.json`（记录 preregistration、manifest、三臂、正式报告、源脚本、screening 升格证据的当前 SHA256；formal JSON SHA `53ec2ee1…af6ea`）。相关既有 102 测试、全量 `evaluation/tests` **1412 OK**。原始 raw 为真实 A6000 BF16/SDPA 推理；Selector 为软件桥，**不代表 FPGA/时延/加速**。只读复核命令：`python evaluation/summarize_b1_corrected_l2_prompt_ablation_full2638.py`。

更新时间：2026-10-04（**B1 corrected-L2 / Prompt full2638 三臂正式确认完成 → `PROMPT_GAIN_NOT_CONFIRMED_FULL2638`**：float/integer/prompt 主分 0.766117/0.766852/0.777362（dense 0.872730、旧 integer 0.764869 仅描述）；量化门 integer−float +0.000735 CI[-0.004192,+0.005833] 通过；prompt−integer +0.010510 CI[-0.005430,+0.025699] 下界 ≤0 → Prompt 增益未确认；升格自 screening100 未调参；2638×3 臂全哈希/契约通过；产物正式报告 json/csv/md + execution receipt；相关 102、全量 1412 OK；真实 A6000 BF16/SDPA 推理，selector 软件桥不代表 FPGA/时延/加速）

- [x] 2026-10-04 **B1 corrected-L2 / Prompt screening100 三臂正式消融（A6000、BF16/SDPA、K96、greedy；无训练/联网/上板）→ `CORRECTED_PROMPT_SELECTOR_SCREENING_PASS`**：统一 B1 forward、同一 `select_hardware_regular_groups`、region_grid=3/local_cap=16 下，浮点 `mean(norm(4 patches))` 主分 **0.733333**，其全局 INT8 + INT64/整数 sqrt + Q0.15 定点近似主分 **0.733333**（integer−float=0，95% CI [0,0]，量化门通过）；冻结 seed-20261003 Prompt 整数语义与纠正 L2 组合主分 **0.820833**（相对定点 L2 **+0.0875**，95% CI **[+0.016667,+0.1625]**，Prompt 增益确认）。dense 描述参考 0.891667，旧缺陷 L2 0.6625，旧 Prompt+缺陷 L2 0.8125。浮点/定点索引 Jaccard mean/min=0.968518/0.900990，输出仅 1/100 改变；Prompt/定点 L2 索引 Jaccard mean=0.292528，输出 16/100 改变。软件 Python Selector 计时不代表 FPGA 加速（vision mean：float 43.0 ms、integer 83.7 ms、prompt 118.4 ms）。正式报告 JSON SHA `abe35625…693a9`，固定 100 样本诊断；下一步必须在不调参的情况下升格到 full2638 三臂确认。

- [x] 2026-10-04 **Layer-5 L2 组内聚合公式真实激活回放（A6000、BF16/SDPA、vision-only、screening100；无 LLM decode/训练/联网/上板）→ `L2_AGGREGATION_DIFFERENCE_FULLY_EXPLAINS_INDEX_SET`**：在统一 layer-5 forward、逐样本同一 K_C、同一固定 3x3 空间 anchor、同一稳定全局填充下，只把 control 的 `norm(mean(4 patches))` 换为历史 `mean(norm(each patch))`；候选与冻结 `shallow_r0250` 索引 **100/100 精确一致**（Jaccard mean/min=1.0/1.0），重新计算的 control 与冻结 control 也 **100/100 一致**，control/reference Jaccard mean/min=0.603351/0.333333。结合此前同预算因果结果（历史索引官方 CV-Bench 0.795833，control 0.670833，差值 +0.125，95% CI [0.041667, 0.208333]），现已确认此前退化来自 L2 组内聚合定义，而非预算、锚点、实现路径或浮点回放偏差。原始回放 SHA `6a8b13da…a81aca`，正式报告 JSON SHA `26c6303b…45784`；固定 100 样本诊断，不是全量 2638 或 FPGA 部署结论。下一步：将 `mean(norm)` 固定为正确的无 Prompt 硬件规则基线，再在相同 cut/K/布局下隔离评估 Prompt 条件化分量。

- [x] 2026-10-04 **Adaptive(A) vs fixed-K96(B) vs historical r25(C) screening100 严格只读诊断（CPU-only、无 GPU/联网/训练/依赖安装；未改任何输入/既有结果）→ `ADAPTIVE_VS_R25_SCREENING100_DIAGNOSTIC_COMPLETE`**：新增 `evaluation/analyze_adaptive_vs_r25_screening100.py`（fail-closed：三臂各 100 唯一 `sample_id` 且集合相同、task 与 manifest 一致、`prune_after_layer=5`、`selected_group_indices` 严格升序唯一且 SHA256 可复核、G 三臂逐样本一致、输入 SHA256 pin + 既有预注册 config 19/16 pins 复核、运行前后输入不变；唯一主指标 `score(...)["cv_bench_accuracy"]`，unweighted=`score(...)["accuracy"]` 仅次要）与 `evaluation/tests/test_analyze_adaptive_vs_r25_screening100.py`（24 测试：输入合同/官方 score/集合统计/迁移/fail-closed/渲染）。**官方分**：A **0.762500**、B **0.745833**、C **0.795833**；A−C **−0.033333**、A−B **+0.016667**。**A 的 K 全 100 样本 >= C**（K差 min/max/mean = 0/69/19.37），A/C 选择集合非嵌套（A⊇C 9/100、C⊇A 0/100，平均 Jaccard 0.5396）；A/B 同一实现且 A⊇B 100/100（41 equal），平均 Jaccard 0.6853。**A/C 迁移**：都对 71、仅 C 对 10、仅 A 对 6、都错 13，文本/答案变化 18；**A/B**：69/7/8/16，变化 15。**结论 `NOT_DIRECT_EVIDENCE_OF_TOKEN_INSUFFICIENCY`**：0.7625<0.795833 不是 token 数量不足的直接证据，主要可观测差异是选择集合/实现路径；审校后仅保留 1 个合格最小因果实验（同一实现、同一预算、仅替换 `selected_group_indices`，需 forced-indices 入口，本次未执行/无 GPU）。产物 `reports/cv_bench/2026-10-04-adaptive-vs-r25-screening100.{json,csv,md}`（CSV 每样本一行）；`py_compile` OK、聚焦 24 OK。

更新时间：2026-10-04（**Adaptive vs r25 screening100 严格诊断完成 → `ADAPTIVE_VS_R25_SCREENING100_DIAGNOSTIC_COMPLETE`**：官方 `cv_bench_accuracy` A=0.7625、B=0.745833、C=0.795833（A−C −0.033333、A−B +0.016667）；A 的 K 全样本 >=C（0/69/19.37）且 A/C 非嵌套（A⊇C 9、C⊇A 0，Jaccard 0.5396），A/B 同实现 A⊇B 100/100；A/C 迁移 71/10/6/13、文本变化 18；→ 不是 token 不足的直接证据，主因可观测差异=选择集合/实现路径；审校后仅保留 1 个同实现同预算仅换索引集合的最小因果实验（未执行）；新增 24 测试 OK；CPU-only/只读、未改输入）

- [x] 2026-10-04 **Layer-5 unbounded real-only vs raw-representative screening100 诊断工程搭建（CPU-only、只读、无 GPU/训练/板卡；未改任何既有 layer5/context-ablation/B1 integration、runner、冻结 JSONL、报告、config 或 `configs/evaluation_protocol.json`；未跑 B 臂、未跑汇总、未写正式报告）→ `LAYER5_UNBOUNDED_VS_REPRESENTATIVE_SCREENING100_SCAFFOLD_READY`**：新增独立 integration `evaluation/hardware_regular_layer5_unbounded_l2.py`（SHA `de05d387…b78a`；单图 cut=5，仅 L2，K96/region_grid3；**禁止** context pooling / context slot reservation / local_candidate_cap；选择规则=`unbounded_real_only_global_l2_3x3_anchor`：每个非空 3x3 区域 1 个 anchor（combined 最高、平局取低 index）+ 其余 K−anchors 取**全局** L2 排名（非 anchor 组、降序、平局取低 index）；`G<=96` 为全 real identity bypass；`verify_real_only_selection` 断言全 SOURCE_REAL/context=0/精确 K 或 identity/严格升序唯一；blocks0..5 全量 → compact（原始选中 2x2 group）→ DeepStack index5 在 compact hidden 上 → blocks6..23；复用冻结 `group_l2_saliency`/`minmax_normalize`/`build_regions`/`region_id_by_group`，显式拒绝 B1-trained Wv/projection/text_embeddings）。新增 runner `evaluation/run_hf_hardware_regular_layer5_unbounded_l2.py`（SHA `bd4a84ab…8a79`；`--score-mode` 仅 `l2`、`--prune-after-layer` 仅 `5`；**无** `--local-candidate-cap`、**无** `--projection-npz`；记录 `arm_label`/`selection_rule`/`uses_context_pooling=false`/`uses_context_slot_reservation=false`/`uses_local_candidate_cap=false`/`local_candidate_cap=null`/`uses_pooled_context_group_values=false`/`source_types_all_real`/`identity_bypass`，并 fail-closed 调用 `verify_real_only_selection`）。新增预注册 config `configs/2026-10-04-layer5-unbounded-vs-representative-screening100.preregistered.json`（SHA `33d46aa0…`；A=冻结 representative 臂 `artifacts/token_selector/2026-10-04-layer5-context-pooling-ablation/layer5_context_ablation_k96_screening100.jsonl` SHA `f09f5a53…e796c`，B=`artifacts/token_selector/2026-10-04-layer5-unbounded-real-only/layer5_unbounded_real_only_k96_screening100.jsonl` 待生成；固定 screening100/K96/layer5/L2/region_grid3/BF16/SDPA/greedy；唯一主指标 `score(...)["cv_bench_accuracy"]`，unweighted 仅次要，禁用 pointing/grounding/ndcg/retained_mass；bootstrap 分层 `task::source_dataset` seed 20261004 repeats 5000；**unbounded−representative** 95% CI 下界 >0.02 → `HARDWARE_SELECTION_CONSTRAINT_HARM_CONFIRMED` 否则 `HARDWARE_SELECTION_CONSTRAINT_HARM_NOT_CONFIRMED`；dense 仅参考；16 个 frozen/新模块 SHA pin 全部现场复核通过，B JSONL pin 为 null 待生成）。新增 fail-closed 汇总器 `evaluation/summarize_layer5_unbounded_vs_representative_screening100.py`（SHA `e0c2f2a6…ae5`；硬校验 100 唯一 ID、K96、layer5、A/B 输入 grid 一致；A 声明 `representative_raw_group`+`uses_pooled_context_group_values=false`；B 必须证明全 real（source_type 全 0）、无 context（context_group_count=0、uses_context_pooling/uses_context_slot_reservation/uses_local_candidate_cap=false、local_candidate_cap 非正）、`selection_rule` 匹配、`real_group_count==output_group_count` 且非 bypass 精确 K=96 或 identity；明确**不要求**跨臂 selection metadata 一致；缺失或任何校验失败 `InputContractError` 且不写报告）。**测试**：新增 integration 25 + runner 10 + summarizer 57 = **92 OK**；既有 layer5 聚焦 159 OK、全量 `unittest discover -s evaluation/tests` **1088 OK**（基线 996）；`py_compile` 6 文件 OK、两个 CLI `--help` OK；frozen pins 逐文件 SHA 复核 15/15 非空匹配。**未做**：未生成 B JSONL、未运行 GPU、未跑汇总、未写正式 quality report。**下一步宿主命令**：`python evaluation/run_hf_hardware_regular_layer5_unbounded_l2.py --model <frozen BF16 RoboBrain2.5-4B> --model-label <label> --manifest datasets/cv_bench_screening100.jsonl --output artifacts/token_selector/2026-10-04-layer5-unbounded-real-only/layer5_unbounded_real_only_k96_screening100.jsonl --score-mode l2 --prune-after-layer 5 --budget 96 --region-grid 3 --dtype bf16 --device cuda:0`，随后把 config 的 `unbounded_layer5_l2_screening_jsonl` 与 arm SHA 回填，再 `python evaluation/summarize_layer5_unbounded_vs_representative_screening100.py`。

更新时间：2026-10-04（**Layer-5 unbounded real-only vs raw-representative screening100 诊断工程搭建完成 → `LAYER5_UNBOUNDED_VS_REPRESENTATIVE_SCREENING100_SCAFFOLD_READY`**：新增独立 real-only 无界全局 L2 + 3x3 anchor 集成（禁 context pooling/slot reservation/local cap、G<=96 identity、DeepStack index5 在 compact hidden、blocks6..23）、runner（无 local-cap/projection 旋钮、all-real/no-context 声明）、预注册 config（A SHA `f09f5a53…`、B 待生成、主指标 `cv_bench_accuracy`、bootstrap seed 20261004/5000、CI 下界>0.02 → `HARDWARE_SELECTION_CONSTRAINT_HARM_CONFIRMED` 否则 NOT_CONFIRMED、dense 仅参考）、fail-closed 汇总器（100 IDs/K96/layer5/grid 一致 + B 全 real/无 context/精确 K 或 identity，且不要求跨臂 metadata 一致）；新增 92 测试 OK、全量 1088 OK（基线 996）；CPU-only/只读、未改冻结文件、未生成 B 结果或正式报告）

- [x] 2026-10-04 **Layer-5 context-pooling 严格单变量消融工程搭建（CPU-only、只读、无 GPU/训练/板卡；未改任何既有 layer5/B1 integration、runner、冻结 JSONL、报告或 `configs/evaluation_protocol.json`；未生成代表性臂 JSONL、未跑汇总、未写正式报告）→ `LAYER5_CONTEXT_POOLING_ABLATION_SCAFFOLD_READY`**：新增 `evaluation/hardware_regular_layer5_context_ablation.py`（独立单图 cut=5、仅 L2、K96/region_grid3/local_cap16；复用冻结 `group_l2_saliency`/`minmax_normalize`/`select_hardware_regular_groups`，并把同一调用重跑一遍以显式断言 `group_indices`/`patch_indices`/`source_type`/`region_id`/`valid`/budget/bypass/两计数/grid/anchors/details 与 frozen pooled 算法逐元素一致、且每个 SOURCE_REAL 槽与其原始 2x2 group 逐位相等；唯一变化=SOURCE_CONTEXT 槽改用 `cut_hidden[group_indices[slot]]` 代表原始 2x2 group；cos/sin 仍来自相同代表 patch 位置，DeepStack index5 仍在 compact hidden 上合并，blocks 6..23 在消融后 compact 序列上运行；显式拒绝 B1-trained Wv/projection）。新增 `evaluation/run_hf_hardware_regular_layer5_context_ablation.py`（`--score-mode` 仅 `l2`、`--prune-after-layer` 仅 `5`、无 projection/training 入口；记录 `ablation`/`context_group_values_source=representative_raw_group`/`uses_pooled_context_group_values=false`/`context_slots_replaced`/`selection_metadata_identical_to_frozen_pooled`，并在元数据不一致时 fail closed）。新增 `configs/2026-10-04-layer5-context-pooling-ablation-screening100.preregistered.json`（screening100/K96/region_grid3/local_cap16/BF16/SDPA/greedy；基线=冻结 pooled `layer5_l2_k96_screening100.jsonl` SHA `5395d5e2…a571e`，新臂待生成；唯一主指标 `score(...)["cv_bench_accuracy"]`，unweighted 仅次要，禁用 pointing/grounding/ndcg/retained_mass；bootstrap 分层 `task::source_dataset` seed 20261004 repeats 5000；representative−pooled 95% CI 下界 >0.02 → `CONTEXT_POOLING_HARM_CONFIRMED` 否则 `CONTEXT_POOLING_HARM_NOT_CONFIRMED`；dense 仅参考）。新增 `evaluation/summarize_layer5_context_pooling_ablation_screening100.py`（100 唯一 ID/K96/L2/layer5/输入 grid/两臂 selection metadata+indices+positions 逐元素一致/消融声明 + 输入 SHA pin；缺失或任何校验失败 `InputContractError` fail closed、不写报告）。**测试**：新增 integration 20 + runner 9 + summarizer 52 = **81 OK**；既有 layer5 聚焦 21+9+48 全过，全量 `unittest discover -s evaluation/tests` **996 OK**；`py_compile` 与新 runner/汇总器 `--help` OK。**未做**：未生成 representative JSONL、未运行 GPU、未跑汇总、未写正式报告；下一步宿主 `python evaluation/run_hf_hardware_regular_layer5_context_ablation.py --model <frozen BF16 RoboBrain2.5-4B> --model-label <label> --manifest datasets/cv_bench_screening100.jsonl --output artifacts/token_selector/2026-10-04-layer5-context-pooling-ablation/layer5_context_ablation_k96_screening100.jsonl --score-mode l2 --prune-after-layer 5 --budget 96 --region-grid 3 --local-candidate-cap 16 --dtype bf16 --device cuda:0`。

更新时间：2026-10-04（**Layer-5 context-pooling 严格单变量消融工程搭建完成 → `LAYER5_CONTEXT_POOLING_ABLATION_SCAFFOLD_READY`**：新增独立 integration（重跑 frozen selector 断言 metadata/indices/positions 完全一致；SOURCE_REAL 不变；仅 SOURCE_CONTEXT 用代表原始 2x2 group；DeepStack index5 在 compact hidden；禁用 B1-trained Wv）、runner（L2/layer5-only + 消融声明字段）、预注册 config（基线 pooled SHA `5395d5e2…`，主指标 `cv_bench_accuracy`，bootstrap seed 20261004/5000，CI 下界>0.02 → CONFIRMED 否则 NOT_CONFIRMED，dense 仅参考）、fail-closed 汇总器（100 IDs/K96/L2/layer5/grid/metadata 一致/SHA pin）；新增 81 测试 OK、全量 996 OK；CPU-only/只读、未改冻结文件、未生成结果或报告）

- [x] 2026-10-04 **Layer-5 vs B1 L2 screening100 工程搭建（CPU-only、只读、无 GPU/训练/板卡；未改 `hardware_regular_b1_integration.py`、既有 runner、冻结 JSONL/报告/`configs/evaluation_protocol.json`；未生成 layer5 结果、未写正式报告）→ `LAYER5_VS_B1_L2_SCREENING100_SCAFFOLD_READY`**：新增 `evaluation/hardware_regular_layer5_integration.py`（单图；固定 `prune_after_layer=5`；仅 `score_mode=l2`；复用冻结 `group_l2_saliency`/`minmax_normalize`/`select_hardware_regular_groups`，`region_grid=3`/`local_candidate_cap=16`/`BUDGET=96`；先跑视觉 blocks 0..5 全序列，再用 `selection.groups` 构造 compact hidden；DeepStack index 5 严格在 compact hidden 上调对应 `deepstack_merger_list`，随后 blocks 6..23，并断言 DeepStack 与 `image_embeds` 对齐；显式拒绝 B1-trained Wv/projection/text_embeddings）。新增 `evaluation/run_hf_hardware_regular_layer5_l2.py`（沿用 B1 runner 的 synchronize 计时/greedy/字段，输出 `score_mode=l2`、`prune_after_layer=5`、`requested_budget`、grid、文本；`--score-mode` 仅 `l2`、`--prune-after-layer` 仅 `5`、无 projection/training 入口）。新增 `configs/2026-10-04-layer5-vs-b1-l2-screening100.preregistered.json`（screening100/K96/BF16/SDPA/greedy；B1 L2 输入=既有 `b1_l2_k96_screening100.jsonl` SHA `983300c1…13849c6`，layer5 待生成；唯一主指标 `score(...)["cv_bench_accuracy"]`；bootstrap 分层 `task::source_dataset` seed 20261004 repeats 5000；layer5−B1 95% CI 下界 > 0.02 → `CUTPOINT_EFFECT_CONFIRMED`，否则 `CUTPOINT_EFFECT_NOT_CONFIRMED`；dense 仅参考）。新增 `evaluation/summarize_layer5_vs_b1_l2_screening100.py`（硬校验 100 IDs、K96、`score_mode=l2`、layer=5、B1/layer5 网格一致、输入 SHA pin；缺失或任何校验失败 `InputContractError` fail closed，不写报告）。**测试**：新增集成 21 + runner 9 + summarizer 48，聚焦（含既有 B1 集成 19 / B1 汇总 41）**138 OK**；`py_compile`、两个 CLI `--help` OK。**未做**：未生成 layer5 JSONL、未运行 GPU、未写正式 quality report；下一步宿主 `python evaluation/run_hf_hardware_regular_layer5_l2.py … --score-mode l2 --prune-after-layer 5`。

更新时间：2026-10-04（**Layer-5 vs B1 L2 screening100 工程搭建完成 → `LAYER5_VS_B1_L2_SCREENING100_SCAFFOLD_READY`**：新增 layer5 L2 集成（单图、cut=5、DeepStack index5 在 compact hidden 上合并、blocks 6..23、禁用 B1-trained Wv）、layer5-only runner、预注册 config（K96/BF16/SDPA/greedy，B1 L2 SHA `983300c1…`，layer5 待生成，唯一主指标 `cv_bench_accuracy`，bootstrap seed 20261004/5000，CI 下界>0.02 → CONFIRMED 否则 NOT_CONFIRMED，dense 仅参考）、汇总器（100 IDs/K96/l2/layer5/网格/SHA fail-closed）；聚焦 138 测试 OK；CPU-only/只读、未改冻结文件、未生成 layer5 结果或正式报告）

- [x] 2026-10-04 **当前 Selector CV-Bench 失败诊断 + 异构时延投影（CPU-only、只读、纯分析；无 GPU/上板/链路实测/训练/调参；未改原始 JSONL、正式 quality report、`configs/evaluation_protocol.json`）→ `CURRENT_SELECTOR_DIAGNOSTIC_COMPLETE` + `PROJECTION_ANALYSIS_ONLY`**：新增 `evaluation/analyze_current_selector_failure.py` 与 `evaluation/project_current_selector_heterogeneous_latency.py` 及对应测试。**CV-Bench 正式主指标**（唯一）= `score(...)["cv_bench_accuracy"]`（复用 `parse_choice`/`answer_letter`/`score`）：K48 **0.6534**、K64 **0.7033**、K96 **0.7649**，2638/2638 且与正式 report 的 `candidate.cv_bench_accuracy` 逐位一致；**unweighted 样本准确率**（`score(...)["accuracy"]`，仅次要诊断，不得称 CV-Bench 主分）：K48 **0.6494**、K64 **0.6990**、K96 **0.7616**。**G 桶（重点 K96；桶均值仅为 secondary unweighted sample accuracy diagnostic，非正式 CV-Bench 主分）**：`G<=96` **143/0.7273**、`97-256` **781/0.8489**、`257-768` **1144/0.7474**、`>768` **570/0.6789**；并给出 task/source 分桶及 K96 task×G。**Identity bypass 验证**：K48 0、K64 61/61、K96 143/143，`G<=K` 集合与 identity 集合逐样本相等、输出文本与冻结 dense 一致、`selected_group_indices==range(G)`、retention=1.0 → 全过。**L2@25% after-layer-5 对照**：既有 **0.7955**（cut point=layer 5），当前为 B1（after vision block 0），对照一律用当前 `cv_bench_accuracy`（K48/K64/K96 差 **-0.1421/-0.0922/-0.0306**，K96 = -0.030616494），明确 **cut point 不同、不得归因于排序器**（JSON/MD 均写死）。**异构时延投影（纯分析、非 ZCU102/Jetson 实测）**：Wv resident、`G<=K` 本地 bypass、无重叠；K48/64/96 FPGA compute mean **8.97/9.03/9.16 ms**、1GbE（110 MB/s）传输 mean **35.50/36.09/37.29 ms**、假设 10GbE（1.1 GB/s）**3.55/3.61/3.73 ms**、model-E2E 投影 1GbE mean **155.98/157.53/166.88 ms**、10GbE 假设 mean **124.03/125.04/133.32 ms**；accepted 公式复算 G80/K48=**328,698.8**、G260/K64=**752,341.8**、G400/K96=**1,097,801.8** cycles 逐位一致。**产物**：`reports/cv_bench/2026-10-04-current-selector-failure-diagnostic.{json,md}`、`reports/token_selector/2026-10-04-current-selector-heterogeneous-latency-projection.{json,md}`（所有输入 SHA256 入库；投影 `measurement_status=not_measured`、`scope=PURE ANALYSIS ESTIMATE`）。**测试**：新增 31/31 OK，全量 `unittest discover -s evaluation/tests` **790 OK**；2026-10-04 口径更正后聚焦 `evaluation/tests/test_analyze_current_selector_failure.py` **20/20 OK**、全量 `unittest discover -s evaluation/tests` **796 OK**（新增 6 个锁定「主分=`cv_bench_accuracy` vs 次分=unweighted」「formal 匹配对 `candidate.cv_bench_accuracy`」「L2 对照用当前 `cv_bench_accuracy`」「G 桶只标次要」「无 `absolute_delta_vs_dense`」「Markdown 双口径并解释差异」的用例）。下一步：若需时延结论，先做 PS 侧寄存器/DDR 实测并用真实链路替换 110 MB/s 假设。

更新时间：2026-10-04（**Current Selector 失败诊断 + 异构时延投影完成**：K48/K64/K96 正式 CV-Bench 主分 `cv_bench_accuracy` 0.6534/0.7033/0.7649（unweighted 样本准确率 0.6494/0.6990/0.7616 仅次要诊断）；K96 G 桶 143/781/1144/570 → 0.7273/0.8489/0.7474/0.6789；identity bypass K64 61/61、K96 143/143 与 dense 文本逐字一致；L2@25% layer5 0.7955 仅对照且 cut point 不同不可归因排序器；纯分析投影 FPGA compute 8.97/9.03/9.16 ms、1GbE 传输 35.50/36.09/37.29 ms、model-E2E 155.98/157.53/166.88 ms、10GbE 假设 124.03/125.04/133.32 ms；accepted 公式三用例逐位复现；新增 31 测试、全量 790 OK；CPU-only/只读、未改原始 JSONL/正式 quality report/`evaluation_protocol`、非实测）

- [x] 2026-10-04 **当前 Selector CV-Bench Gate CPU 预检（只执行工程、CPU-only、只读审计、未读取任何 proposed CV-Bench 输出、无 GPU/联网/训练/调参/依赖安装；未改原模型、既有 HLS/cosim/OOC/BD 与既有报告）→ `PREFLIGHT_READY`**：只读审计确认「当前 Selector」= 2026-10-03 integer scorer（`quantize_activation`→`primary_quantized_query`→已验收 per-tensor INT8 `Wv_q`→`integer_chain` 精确 INT32/INT64/Q0.15）+ 已验收紧凑选择合同（`select_hardware_regular_groups`，`combined = score_q / 2**15`）。**唯一部署 checkpoint 冻结为 seed 20261003**（`artifacts/token_selector/spatial_prompt_contrastive_trainval/B_SPATIAL_KL_CONTRASTIVE/20261003/best.npz` SHA `1f1d379e…ca72ff`；Wv `126bd842…d89d3b`、scale `93585a3b…28b13f`），依据=已验收紧凑 HLS/IP/OOC/Zynq-BD 全链只由该 seed 构建、`fpga_accel/integer_selector_pipeline_compact_hls` fixture manifest pin 该 bundle；20261004/05 仅存在于多 seed 审计统计 → **未三 seed 平均、未按 CV-Bench 选择、无歧义**。**预注册（先于任何 proposed CV-Bench 输出写盘）**：`configs/current_selector_cvbench_20261004.preregistered.json`（`f49f86b0…09864`）+ `reports/token_selector/2026-10-04-current-selector-cvbench-preregistration.{json,md}`（json `942c829f…426dd`，`all_pins_verified=true`），锁定 manifest `22399a76…ce3d91` 与图片聚合 `abac17f6…110fc`（2638/404,722,392 B）、官方 evaluator `58a6bc67…d8a61b`（公式 `((2D_ADE+2D_COCO)/2+3D_Omni3D)/2`）、dense BF16 基线、模型 `model.safetensors e2f6e3c3…6eb2f`、K=48/64/96、`do_sample=false`、`max_new_tokens=16`、GPU 段 synchronize 的 model-E2E/application-E2E 边界、bootstrap 5000/seed 20261004、禁止 post-hoc 回选、Pointing/Grounding/NDCG/retained-mass 不得作主指标。**新增**：`evaluation/current_selector_integer.py`（`34728308…b8ed7`，零新数学整数链软件桥）；`evaluation/run_hf_current_selector_cvbench.py`（`5870e847…f2bfc2`，identity ≤5 / 结构 smoke ≤5 不聚合质量 / full2638 / latency；`resume_state` 对 mode/budget/seed/score_mode 不一致 fail-closed、绝不重复样本；逐样本 raw JSONL 含 token ids/正确性/全部计时/峰值显存；selector 恒标 `software`，不冒充 FPGA）；`evaluation/summarize_current_selector_cvbench.py`（`1e1b2cc4…bc053`，仅当恰 2638 唯一完整样本时输出官方组合分/2D/3D/四任务/absolute delta/relative retention/parse rate/paired bootstrap CI，部分结果只能 `IN_PROGRESS`）；`scripts/run_current_selector_cvbench.sh`（`da1f8523…386b`，host driver：identity→smoke→K48/64/96 可安全续跑；100×3 latency 命令只打印未运行）。**最小修改**：`evaluation/hardware_regular_b1_integration.py`（`c5fda60c…3cffa`）新增 `SCORE_MODE_INTEGER` 与可选 `integer_scorer` 回调，既有 l2/prompt 行为不变。**CPU 验证**：py_compile OK、`bash -n` OK、两个 CLI `--help` OK、聚焦 **15/15 OK**、全量 `unittest discover -s evaluation/tests` **730 OK**；日志与 SHA 见 `reports/token_selector/logs/2026-10-04-current-selector-cvbench-preflight/`。**未运行 GPU、未读 proposed CV-Bench 输出、无精度/加速/上板声明**；下一条宿主命令 `bash scripts/run_current_selector_cvbench.sh`（Codex 在宿主运行）。

更新时间：2026-10-04（**Current Selector CV-Bench 预检完成 → `PREFLIGHT_READY`**：唯一部署 seed 冻结 **20261003**（已验收紧凑 HLS/IP/OOC/BD 链唯一引用，非 CV-Bench 选择、未平均三 seed）；预注册 config `f49f86b0…09864` + report `942c829f…426dd` 全 pin 通过，写于任何 proposed CV-Bench 输出之前；新增整数链软件桥/真实 B1 runner（identity/smoke/full/latency、安全续跑、raw JSONL、协议计时）/官方汇总器（2638 门槛 + IN_PROGRESS）/host driver；B1 integration 仅加 integer score mode；聚焦 15 + 全量 730 测试 OK、py_compile/CLI help/bash -n OK；CPU-only/只读、无 GPU/精度/加速声明；下一步宿主 `bash scripts/run_current_selector_cvbench.sh`）

- [x] 2026-10-03 **最小 ZCU102 Zynq MPSoC Block Design 验证（Vivado 2023.2、CPU-only、只读复用已验收紧凑 IP `robobrain.local:hls:integer_selector_pipeline_compact:1.0` 的 `export.zip`；无 bitstream、无板卡、无 GPU/联网/训练/调参；未改四个受保护 HLS 目录、`evaluation/`、紧凑 HLS/cosim/OOC 工作区与既有报告/日志/prompt/artifact）→ `INTEGER_SELECTOR_MINIMAL_ZYNQ_BD_PASS`**：新增 `fpga_accel/integer_selector_minimal_zynq_bd/`（`scripts/build_bd.tcl` 建 BD/连线/地址/`validate_bd_design`/`generate_target all`/`make_wrapper`/`write_bd_tcl`；`scripts/synth_bd.tcl` 有界 `launch_runs synth_1` 综合、无 implementation/bitstream；`scripts/run_bd.sh` 单进程驱动并把有界文本证据回拷；`scripts/protected_sha.py`、`scripts/summarize_bd.py`）。**全门通过**：part 恰 `xczu9eg-ffvb1156-2-e`、board part `xilinx.com:zcu102:part0:3.4`（available；3.3/3.4 均装）；VLNV 恰 `robobrain.local:hls:integer_selector_pipeline_compact:1.0`；接口恰 **3×128-bit AXI4 master**（`m_axi_input_arena`/`m_axi_wv_q`/`m_axi_output_arena`）+ **1×32-bit AXI4-Lite slave**（`s_axi_control`），`control_r` 出现 0 次、无 score 端口；三 master 经 3→1 `smartconnect_ddr` 到 PS `S_AXI_HP0_FPD`（128-bit AXI4，`PSU__USE__S_AXI_GP2=1`），PS `M_AXI_HPM0_FPD`（128-bit AXI4）经 `smartconnect_control` 到 `control`；`validate_bd_design` **0 ERROR / 0 CRITICAL WARNING**（15 条设计 WARNING，另有 17 条证据导出的 BD 5-234）；输出产物 + HDL wrapper 已生成、`write_bd_tcl` 已导出。**地址映射**：control base **`0xA0000000`**（range `0x10000`）；三 DDR 段各 offset `0x0`、range **`0x80000000`**（HP0_DDR_LOW，非空）。**时钟/复位**：`pl_clk0`（IOPLL、DIVISOR0=5）目标 200 MHz、实测 **199.998 MHz**（`clk_pl_0` period 5.000 ns）；`pl_resetn0`→`proc_sys_reset`（`C_EXT_RESET_HIGH=0`，低有效）→`peripheral_aresetn`→`ap_rst_n`。**有界综合**：`synth_1` rc=0、wall **284 s**、post-synth **CLB LUT 32,514（11.86%）/ FF 37,264（6.80%）/ BRAM 122.5（13.43%）/ DSP 210（8.33%）/ URAM 0**、WNS **+1.184 ns**（5.0 ns 约束）、TNS 0（post-synth 未布线，非时序签核）；`BITSTREAM_FILES=0`。**受保护 SHA 2315/2315 before=after 逐字节不变**（aggregate `68c2c1b318be66e632e5b14f646881fd4e069528082b61924bf1ab29c8bada35`）。**产物**：`artifacts/token_selector/2026-10-03-integer-selector-minimal-zynq-bd/`、`reports/token_selector/2026-10-03-integer-selector-minimal-zynq-bd.{json,md}`、`reports/token_selector/logs/2026-10-03-integer-selector-minimal-zynq-bd-harness-result.json`。**限制**：无 Ethernet/Jetson、无 PS 软件、无 DDR 争用/仲裁带宽测量、无全系统时序签核、无 bitstream/板卡、无模型端到端时延、无任务精度声明。

更新时间：2026-10-03（**最小 ZCU102 BD 验证完成**：part/board 已装；恰 3×128-bit AXI4 master + 1×AXI4-Lite `control`（无 `control_r`/score 端口）；3→1 SmartConnect 汇聚到 PS `S_AXI_HP0_FPD`；control base `0xA0000000`、三 DDR 段 `0x0..0x7FFFFFFF` 非空；PL0 实测 199.998 MHz；validate 0 ERROR/0 CRITICAL，target/wrapper/`write_bd_tcl` 已生成；有界综合 rc=0、LUT 11.86%/FF 6.80%/BRAM 13.43%/DSP 8.33%、WNS +1.184 ns；受保护 2315 文件不变；→ **`INTEGER_SELECTOR_MINIMAL_ZYNQ_BD_PASS`**；无 bitstream/上板/PS 软件/DDR 争用/全系统时序/端到端/精度声明；复现命令 `bash fpga_accel/integer_selector_minimal_zynq_bd/scripts/run_bd.sh`；下一步实现 PS 侧 bare-metal/Linux 寄存器+DDR 测试（静态 JPG 派生 fixture），再做 Jetson 以太网集成）

- [x] 2026-10-03 **已验收紧凑整数 Selector 顶层的 IP 导出 + Vivado 2023.2 有界 OOC 综合/实现/布线（只读复用已验收紧凑顶层的单一源码与接口；Vitis HLS `csynth_design` + `export_design -rtl verilog -format ip_catalog -flow syn`；Vivado `synth_design -mode out_of_context` + `opt_design` + `place_design` + `phys_opt_design` + `route_design`；无 bitstream、无板卡、无 GPU/联网/训练/调参；未改四个受保护 HLS 目录、`evaluation/` 与既有紧凑接口/cosim 报告、日志、prompt、fixture、artifact）→ `INTEGER_SELECTOR_COMPACT_IP_OOC_ROUTE_PASS`**：新增独立工作区 `fpga_accel/integer_selector_pipeline_compact_ooc/`（`scripts/run_hls_export.tcl` 只读引用已验收 `integer_selector_pipeline_compact_hls/src/integer_selector_pipeline_compact.cpp`，接口=3×128-bit `m_axi` + 1×AXI-Lite；`scripts/run_vivado_ooc.tcl` 读取导出 IP 的 `hdl/verilog` RTL 与 `constraints/*_ooc.xdc`；`scripts/run_ooc.sh` 单进程驱动并只回拷有界证据；另 `protected_sha.py`、`summarize_ooc.py`）。**全门通过**：HLS csynth 0 error/0 fatal、estimated **3.650 ns ≤ 5.0 ns**，资源与已验收紧凑接口 csynth **逐字段一致**（BRAM_18K 295 / DSP 212 / FF 45,156 / LUT 91,089 / URAM 0；`identity_with_accepted.identical=true`）；IP 导出 VLNV **`robobrain.local:hls:integer_selector_pipeline_compact:1.0`**、`export.zip` **1,051,551 B**（sha256 `b4e0654792e6591afd388aeb34bff713cd2b8688d068be1942a987bdff9a4090`）；**生成接口门（component.xml + 顶层端口双重解析）**：恰 3 个 128-bit AXI4 master `input_arena`/`wv_q`/`output_arena`、恰 1 个 AXI4-Lite slave `control`（WDATA 32 bit）、`control_r` 出现 0 次、无外部 `score_q` 端口；part 恰 `xczu9eg-ffvb1156-2-e`、`ap_clk` 恰 5.000 ns（导出 OOC XDC 仅 `create_clock -name ap_clk -period 5.000`，**无 set_clock_uncertainty**；HLS 默认 1.35 ns 仅用于 csynth 调度，post-route 约束不含）。**Vivado OOC 全成功**：synth 164 s / opt 15 s / place 96 s / phys_opt 1 s / route 64 s，全 rc=0，总 **413 s** 未触发 90 min 超时；route **routable 59,967 = fully routed 59,967、routing errors 0、unrouted 0**；**post-route WNS +0.529 / TNS 0 / WHS +0.028 / THS 0 ns（116,438 endpoints、0 failing）**，由 5 ns 需求推导 **achieved 223.66 MHz**；post-route LUT **27,607/274,080 = 10.07%**、FF **30,997/548,160 = 5.65%**、BRAM **122.5/912 = 13.43%**、DSP **210/2,520 = 8.33%**、URAM 0（设备无 URAM；全部 <80%）；DRC **Error 0 / Critical Warning 0**（125 Warning：DPIP-2 16 / DPOP-3 36 / DPOP-4 72 / RTSTAT-10 1，OOC 典型）；关键 setup 路径 `grp_ts_pool_context_packed…/weights_U/ram_reg_bram_0/CLKARDCLK` → `…/grp_ts_pool_context_packed_Pipeline_VITIS_LOOP_276_2_VITIS_LOOP_281_3…/mul_ln290_reg_1294_reg/DSP_OUTPUT_INST/ALU_OUT[15]`（slack **+0.529 ns**、data path **4.418 ns**、5 DSP 级）；唯一 console CRITICAL WARNING 为宿主 `[Common 17-741]` Tcl store 写权限，非设计 DRC、不影响实现/时序。**受保护 2260/2260 文件 before=after 逐字节不变**（aggregate `8fec4b4988ed014b5ac0e4e453ab46153ec731dc825ab2d6810608240ca5dfc5`）；artifact **40,851,315 B / 44 文件**（最大 routed DCP 33,501,830 B ≤200 MiB 已归档，sha256 `25f1ca2f…`；未复制完整工程/.cache/.gen/.runs/RTL 树/仿真数据）；日志无 `write_bitstream`、无 bitstream。**产物**：`artifacts/token_selector/2026-10-03-integer-selector-compact-ip-ooc-route/`（hls、ip、vivado、logs、scripts）、`reports/token_selector/2026-10-03-integer-selector-compact-ip-ooc-route.{json,md}`、`reports/token_selector/logs/2026-10-03-integer-selector-compact-ip-ooc-route-harness-result.json`。**限制**：仅 OOC 证据（不含 PS/DDR/AXI 互联争用、完整 block design、bitstream、板卡实测、端到端时延/吞吐或任务精度声明）。**精确下一步（未执行，复现命令 `bash fpga_accel/integer_selector_pipeline_compact_ooc/scripts/run_ooc.sh`）**：以该导出 IP 构建最小 Zynq UltraScale+ MPSoC block design（PS、时钟/复位、3×128-bit AXI4 master、AXI4-Lite control、memory map 与 DMA 路径）并完成设计验证，之后才考虑 bitstream。

- [x] 2026-10-03 **已验收紧凑整数 Selector 顶层的有界 Vitis HLS 2023.2 C/RTL 协同仿真（CPU-only、只读引用已验收源码/头文件与 fixture、只 `csynth_design`+`cosim_design -rtl verilog -tool xsim -trace_level none`；无 IP 导出/Vivado implementation/bitstream/板卡/GPU/联网/训练/调参；未改四个受保护 HLS 目录、`evaluation/` 与既有紧凑接口报告/fixture）→ `INTEGER_SELECTOR_COMPACT_RTL_COSIM_PASS`**：新增独立工作区 `fpga_accel/integer_selector_pipeline_compact_cosim/`（专用 RTL-cosim testbench **不使用 `dlopen`**，直接从已验收 fixture 读取冻结 expected arena/标量并逐字节比较；用例解析器逐行对已验收 `manifest.txt` 校验并做槽位语义检查，缺失即失败、不替代）。**恰 5 个代表用例**：`synth_g80_K48_cap16`（G80/8x10/K48，production 非 identity）、`real_grounding_correct_K64_cap16`（G260/13x20/K64，real correct prompt）、`synth_g400_K96_cap16`（G400/20x20/K96，production stress）、`synth_g80_K96_cap16`（G80/8x10/K96，identity bypass）、`zero_g128_K48_cap16`（G128/8x16/K48，确定性全 0）。**门全过**：host g++ 与 ASan+UBSan 各 5/5 用例、51/51 checks、0 scalar/arena mismatch；XSim C/RTL `*** C/RTL co-simulation finished: PASS ***`、C TB 与 C post-check 各 5 用例 51/51、0 mismatch、0 error、0 fatal；生成接口仍为**恰 3 个 128-bit `m_axi`（`input_arena`/`wv_q`/`output_arena`）+ 恰 1 个 AXI-Lite `control`（`control_r` 出现 0 次、无 score 端口）**；`trace_level none`、`/tmp` 工程内 wdb/vcd/saif/fsdb 全 0、仓库 artifact **1,284,950 B / 17 文件**（最大 364 KB，未复制仿真工作目录/RTL 树）；**逐 transaction latency** 338,225 / 737,081 / 1,062,296 / 266,036 / 384,196 cycles（total **2,787,614**；min/avg/max 266,036/557,566/1,062,296 @200 MHz），与已验收公式估计 G80/K48 328,699、G260/K64 752,342、G400/K96 1,097,802 一致；墙钟 **596 s**，未触发 60 min XSim 超时；**受保护 2229/2229 文件 before=after 逐字节不变**（aggregate `2478d5e7…`）。**产物**：`artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/`、`reports/token_selector/2026-10-03-integer-selector-compact-rtl-cosim.{json,md}`、`reports/token_selector/logs/2026-10-03-integer-selector-compact-rtl-cosim-harness-result.json`。**限制**：仅证明 RTL 事务行为与冻结 expected arena 一致，不证明布线时序、bitstream、板卡时延、模型端到端速度或任务精度。

更新时间：2026-10-03（**紧凑整数 Selector 顶层有界 C/RTL 协同仿真完成**：5 个冻结代表用例 host g++/ASan 与 XSim C/RTL 全过（各 51/51、0 mismatch、0 error/fatal）；接口仍 3×128-bit `m_axi` + 1×AXI-Lite `control`（无 `control_r`/无 score 端口）；`trace_level none`、无波形文件、artifact 1.28 MB/17 文件；逐 transaction 338,225/737,081/1,062,296/266,036/384,196 cycles（total 2,787,614，与公式估计一致）；墙钟 596 s 未超时；受保护 2229/2229 不变；→ **`INTEGER_SELECTOR_COMPACT_RTL_COSIM_PASS`**；仅功能 cosim，非布线时序/bitstream/板卡/端到端/精度声明；建议下一步才导出 IP 并做 ZCU102 OOC 综合/实现）

- [x] 2026-10-03 **整数 Selector 流水线紧凑 AXI 接口 HLS 封装（CPU-only、只读复用、只 `csynth_design`；无 cosim/IP 导出/implementation/bitstream/板卡/GPU/联网/训练/调参；未改四个受保护目录）→ `INTEGER_SELECTOR_COMPACT_INTERFACE_HLS_PASS`**：新增 `fpga_accel/integer_selector_pipeline_compact_hls/`（顶层 `integer_selector_pipeline_compact`，`score_q` 仍为片上局部 `uint16_t[G]`、非端口）。**接口门由生成报告/元数据证明**：恰好 3 个 128-bit `m_axi`（`input_arena`/`wv_q`/`output_arena`）+ 恰好 1 个 AXI-Lite `control`（**`control_r` 出现 0 次**；做法：每个 m_axi 指针额外显式 `s_axilite port=<ptr> bundle=control` 合并 offset 寄存器，沿用已验收 q8q8 模式）；无 score 端口。**冻结内存合同**：输入 arena `tokens@0/G*4096` + `group_mean@G*4096/G*1024` + `query@G*5120/128` = `G*5120+128` B（全 64 B 对齐、无额外 padding）；`wv_q` 128 KiB/8192 words；输出 arena 权威 `fpga_transfer_layout(K)`，K=48/64/96 = **196,928/262,464/393,792 B**，5 段各自 64 B 对齐，gap/未用项/尾部全 0；tokens 直写 arena 起点、K≤96 metadata 片上缓存后打包；identity（G≤K）仍写固定 K arena 且保留 `output_count` 语义。**复用**：两个已验收 `.cpp` 只读 `#include`，直接调用 `is_scorer_core<false>`/`token_selector_core_packed`，无第二套数学；状态码 `static_assert` 与 `integer_selector_pipeline` 逐值一致。**四道门全过**：Python **16/16 OK**（K=16/32/48/64/80/96 输出与 G=9/16/50/80/128/260/400/4096 输入：公式 + 冻结权威 + 编译 C++ `ispc_fpga_layout`/`ispc_input_layout` 镜像逐字段一致；23 用例 expected arena padding/unused 全 0）、g++ **244/0 PASS**、ASan+UBSan **244/0 PASS**、HLS C-sim **0 error 244/0 PASS**；三者在 g++/ASan/csim 下均 `dlopen` 已验收 `integer_selector_pipeline` 并逐字节比对 arena/status/三计数/`scale_out`，另与 `integer_selector_scorer`+`token_selector_core_packed` 顺序组合逐字节一致，覆盖 G80/K48、G260/K64、G400/K96、identity、全 0/全等分数、空 region（2x25）、真实 grounding/pointing correct+wrong prompt、canary 与重复确定性。**csynth**：0 error、estimated **3.650 ns ≤ requested 5.0 ns / effective 3.650 ns**（默认 uncertainty、slack 0.000、Fmax 273.97 MHz）、资源 BRAM **16.17%** / DSP **8.41%** / FF **8.24%** / LUT **33.23%** / URAM **0**（均 <50%）；**公式推导** G80/K48 **328,699 cycles = 1.643 ms**、G260/K64 **752,342 = 3.762 ms**、G400/K96 **1,097,802 = 5.489 ms @200 MHz**（工具报告 worst 9,337,705 为 compile-time 上界）；1 GbE 传输按 fixed arena + Wv-resident 模型给出（非测量）。**限制**：HLS-only 证据，非 RTL cosim、非 routed timing、非上板、非端到端模型时延、非任务精度；生产用例周期为公式估计；传输为带宽假设模型；未导出 IP/未 implementation。**下一步**：将该 3×128-bit `m_axi` + 1×AXI-Lite 组件接入 ZCU102 block design，`wv_q` 固定到 PL 侧常驻地址、输入用连续 arena，并做 C/RTL cosim 与上板带宽/时延测量。**产物**：`fpga_accel/integer_selector_pipeline_compact_hls/`、`artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/{fixtures,logs,csynth}`（含 `summary.json`、`interface_evidence.json`、`layout_evidence.json`、`protected_trees_before.json`）、`reports/token_selector/2026-10-03-integer-selector-compact-interface-hls.{json,md}`、`reports/token_selector/logs/2026-10-03-integer-selector-compact-interface-hls-harness-result.json`。

更新时间：2026-10-03（**紧凑 AXI 接口封装完成**：3×128-bit m_axi + 1×AXI-Lite `control`（无 `control_r`）、无 score 端口；输入 arena `G*5120+128` B、输出 fixed arena K48/64/96=196,928/262,464/393,792 B 且 padding 全 0；Python 16 / g++ 244 / ASan+UBSan 244 / HLS C-sim 0 error 244 全过并 dlopen 已验收 pipeline 逐字节比对；csynth 0 error、3.650 ns≤5.0、BRAM 16.17 / DSP 8.41 / FF 8.24 / LUT 33.23 / URAM 0 %；推导 G80/K48 1.643、G260/K64 3.762、G400/K96 5.489 ms@200MHz；受保护 4 树逐字节未变；→ **`INTEGER_SELECTOR_COMPACT_INTERFACE_HLS_PASS`**；HLS-only、非 cosim/上板/端到端/精度声明）

- [x] 2026-10-03 **Codex 审校有界更正：`integer_selector_pipeline` 传输审计 5 段 64 B 布局计数（CPU-only、只读、无 GPU/联网/模型/训练/板卡；只改允许文件；未重跑 g++/ASan/csim/csynth）→ `INTEGER_SELECTOR_PIPELINE_HLS_PASS` 保持**：Codex 审校接受功能/csim/csynth 证据，但指出旧 `fpga_accel/integer_selector_pipeline_hls/tools/transfer_audit.py::image_bytes` 把 `selected_tokens + metadata` 只做**一次**尾部 64 B `align_up`，未按冻结合同 `evaluation.hardware_regular_token_selector.fpga_transfer_layout(budget)` 的 5 段逐 offset 对齐，K=48/K=96 的 FPGA→Jetson 总量与派生网络/DDR 时延被低估。**更正**：`image_bytes` **直接调用**冻结 `fpga_transfer_layout(budget)`/`describe_fpga_layout(budget)` 实时产生 `fixed_budget_arena_bytes`（不复制第二套 offset 算法），保存 5 段 offset/size；新增 `logical_payload_bytes`（实际 `output_count` 个 token + 四种 metadata 原始有效字节，无 padding）；网络/DDR 保守时间**统一基于 fixed arena**，scatter-gather payload 时间仅标注为**下界**；G=80/K=96 identity bypass 的 `output_count=80` 与 runtime 固定预算 arena K=96 分开报告。**关键数字**：fixed arena K=48/64/96 = **196,928 / 262,464 / 393,792 B**（由冻结函数实时产生、与逐字段一致、5 段 offset 全 64 B 对齐；预期一致）；旧→新 FPGA→Jetson（cold）：K=48 196,864→**196,928**（+64）、K=64 262,464→262,464（+0）、K=96/G=260,400 393,728→**393,792**（+64）、K=96/G=80 identity 328,128→**393,792**（+65,664）；1 GbE F→J arena central（110 MB/s）K=96 由旧 3,579.1→**3,579.9 µs**。Jetson→FPGA、score_q DDR 节省、HLS 功能/时序/资源/周期数字**全部不变**；决策不变、旧传输数字作废以更正后字段为准。**验证**：聚焦 Python **14/14 OK**（新增 7 个会令旧实现失败的用例：K=48/64/96 offsets/sizes/total 与冻结函数逐字段一致、K=48/K=96 明确防回归、G=80/K=96 payload+arena 双口径、网络/DDR 时间 arena-based、Jetson→FPGA/score_q 不变、无第二套 offset 算法）；受保护 SHA 在本更正前后逐字节不变 **174/174**（pipeline HLS header/source/TB/TCL/Makefile 8 + 两既有核 27 + fixtures 38 + csynth 96 + 原 pipeline artifact 日志 4 + 原 pipeline harness-result 1，`all_must_be_unchanged_groups_ok=true`）。**产物**：主 `reports/token_selector/2026-10-03-integer-selector-pipeline-hls.{json,md}`（JSON `authority_prompt` = `reports/token_selector/prompts/2026-10-03-integer-selector-pipeline-transfer-correction-harness-prompt.md` SHA256 `ff8af13c8a3115511afc7fac750a4e185f6327777f25163e81a6ef59d34b9114`，含 `transfer_audit_correction` + `frozen_asset_sha256_check`）、`reports/token_selector/logs/2026-10-03-integer-selector-pipeline-transfer-correction-harness-result.json`、`...-transfer-correction-focused-tests.log`、`...-transfer-correction-frozen-assets-baseline.json`；**原 pipeline harness-result 未覆盖**。**下一步**：无（有界更正完成）；如需复核运行 `cd fpga_accel/integer_selector_pipeline_hls && /home/zhaoxiangyu/conda-envs/robobrain25/bin/python -m unittest discover -s tests -p 'test_*.py' -v`（不重生成 fixtures），再 `/home/zhaoxiangyu/conda-envs/robobrain25/bin/python tools/build_report.py`。

更新时间：2026-10-03（**传输审计有界更正完成**：`image_bytes` 改调冻结 `fpga_transfer_layout`/`describe_fpga_layout`；fixed arena K=48/64/96 = 196,928/262,464/393,792 B，K=48/K=96 旧值分别低估 +64 / +65,664 B（G=80 identity）、K=96 G≥96 +64 B；Jetson→FPGA 与 score_q 节省、HLS 功能/时序/资源数字不变；聚焦 Python 14/14 OK；受保护 SHA 174/174 不变；`authority_prompt` 非 null；→ **`INTEGER_SELECTOR_PIPELINE_HLS_PASS` 保持**，旧传输数字作废；CPU-only/只读，未重跑 g++/ASan/csim/csynth）

- [x] 2026-10-03 **整数打分核与硬件规则选择核片上串接 HLS 实现 + 受控 csynth（CPU-only、只读复用、只 `csynth_design`；无 cosim/IP 导出/implementation/bitstream/板卡/GPU/联网/训练/调参；未改两个既有 HLS 目录、evaluation golden、模型/checkpoint/manifest/config/既有报告）→ `INTEGER_SELECTOR_PIPELINE_HLS_PASS`**：新增 `fpga_accel/integer_selector_pipeline_hls/`（单一顶层 `integer_selector_pipeline`：`tokens_q`/`group_mean_q`/`wv_q`/`query_q` 均为 128-bit packed 有符号 INT8、16 lane/word 小端，`group_count 1..4096`，状态码 union `OK/NULL/GROUP_COUNT(=GROUP_LIMIT)/GRID/BUDGET/INSUFFICIENT/CANDIDATE_POOL/INTERNAL/LOCAL_CAP`；输出沿用 token selector 5 段语义 + `scale_out`/三计数）。**中间 `score_q[G]` 是顶层函数内 `uint16_t score_local[IS_MAX_GROUPS]` 片上局部缓冲，源码静态检查与综合接口表双重确认：无 score m_axi bundle、不是端口、不往返 DDR**。**复用策略（不复制第二套数学）**：两个已验收 `.cpp` 被 `#include` 进 wrapper，直接调用 `is_scorer_core<false>` 与 `token_selector_core_packed`；`is_scorer_core` 因位于匿名命名空间（静态可见性）无法独立链接，只读 include 是最小可审计暴露。综合可组合性审计：Vitis HLS 2023.2 `HLSGen` 在“两核融合进 helper 函数/C++ 模板”时 segfault（已用最小变体复现），故融合体直接写在顶层 `extern "C"` 函数内，host-only debug 孪生用 `#ifndef __SYNTHESIS__` 隔离（上游 core 自身模板未改）。**功能门全过**：Python 7/7 OK、g++ 383/383 PASS、ASan+UBSan 383/383 PASS、HLS C-sim 0 error 383/383 PASS；覆盖 budget 48/64/96、G 80/260/400（真实 grounding/pointing correct prompt，G=260/13x20）、identity bypass（G=16/K=48、G=9/K=96）、全 0 与全等 score、空 region（2x25）、非法 G/grid/budget/local-cap 与空指针、逐字节重复确定性；测试台逐字节核对 I 片上 score_q vs 冻结 Python scorer golden、II 最终 5 段输出 vs 冻结 Python selector golden、III pipeline vs “既有 standalone scorer + packed selector”顺序组合、IV packed v4 vs int8 v3。**csynth**：0 error、142 warning、**无 HLS 200-871 / 200-1016**（1× 200-790 loop constraint not satisfied、5× II 200-885、2× 200-878，来自一次性 Wv 载入 achieved II=8）；estimated **3.650 ns ≤ effective 3.650 ns**（target 5.0、默认 uncertainty 1.35、slack 0.000、Fmax 273.97 MHz）；资源 BRAM_18K **17.98%** / DSP **8.41%** / FF **9.16%** / LUT **33.16%** / URAM 0（均 <50%）；**公式推导** G=400/K=96 总周期 **1,073,128 = 5.366 ms @200 MHz ≤ 6,000,000 cycles / 30 ms**（报告 worst 9,312,990 cycles = 46.565 ms @200 MHz 为 compile-time 上界；模型在最大界 9,462,678 vs 工具 9,312,990、保守 1.6%）。**异构传输审计（模型估算，非测量）**：单列每图 `tokens_q`/`group_mean_q`/`query_q`（+Wv cold/resident）与 F→J selected tokens + metadata；1 GbE 假设有效载荷 central 110 MB/s、敏感性区间 [90,118] MB/s，另给理论 6 GB/s 本地 DDR 时间；片上融合只省中间 `score_q` 的 **4G bytes FPGA-DDR 往返**（G=400 1,600 B、0.267 us @6 GB/s），**不消除** Jetson↔FPGA 往返；`group_mean_q` 仍为额外输入，PL 内重建仅列为未来潜在节省、本轮不改数值合同。**产物**：`fpga_accel/integer_selector_pipeline_hls/{include,src,tb,tools,tests,scripts,README.md,Makefile}`、`artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/{fixtures,csynth,logs}`、`reports/token_selector/2026-10-03-integer-selector-pipeline-hls.{md,json}`、`reports/token_selector/logs/2026-10-03-integer-selector-pipeline-hls-harness-result.json`。**精确下一步（未执行）**：只有独立授权后才做 C/RTL cosim、IP 导出、implementation/bitstream 与板卡，或探索 PL 内重建 `group_mean_q`；本轮综合估计非上板实测、非端到端模型加速。

更新时间：2026-10-03（**整数打分核与 packed selector 片上串接完成**：新增 `integer_selector_pipeline_hls`，`score_q` 为片上局部缓冲（源码+综合接口双重确认无端口/不落 DDR），两核 `.cpp` 只读 include 直接调用、无第二套数学；四道门 Python 7 / g++ 383 / ASan+UBSan 383 / HLS C-sim 0 error 383 全过；csynth estimated **3.650 ns ≤ 3.650 ns**（默认 1.35 ns uncertainty 未放宽）、资源 BRAM 17.98 / DSP 8.41 / FF 9.16 / LUT 33.16 / URAM 0 %、0 error 且无 200-871/1016；推导 G=400/K=96 **1,073,128 cycles / 5.366 ms @200 MHz**（≤6M/30ms）；异构传输审计给出 1 GbE（90–118 MB/s）与 6 GB/s DDR 模型时间，片上融合省 4G bytes DDR 往返但不省 Jetson↔FPGA 往返 → **`INTEGER_SELECTOR_PIPELINE_HLS_PASS`**；CPU-only/只读复用、未改既有文件、未 cosim/未导出 IP/未 implementation/未上板、非端到端模型加速声明）

- [x] 2026-10-03 **独立位精确整数 Selector 打分核 HLS 实现 + 受控 csynth（CPU-only、只读复用、只 `csynth_design`；无 cosim/IP 导出/implementation/bitstream/板卡/GPU/联网/训练；未改既有 Python 审计、selector/HLS kernel、模型/checkpoint/cache/manifest/config/report）→ `INTEGER_SELECTOR_SCORER_HLS_PASS`**：新增 `fpga_accel/integer_selector_scorer_hls/`（顶层 `integer_selector_scorer`：`group_mean_q`/`wv_q`/`query_q` 均 128-bit packed 有符号 INT8、16 lane/word 小端，`score_q` uint16 Q0.15，`group_count 1..4096`，4 个独立 m_axi bundle + s_axilite control，`OK/ERR_NULL/ERR_GROUP_COUNT`），位精确实现 `evaluation/selector_integer_scorer_audit.py::integer_chain`：INT8×INT8→INT32 投影、INT64 dot/norm2、精确 floor 整数 sqrt、精确有符号有理 round-half-away（`dot*2^16` 用乘法，非负值左移）、两项独立整数 min-max 到 Q0.15、λ=0.5 半进位 + clamp；DUT 内 float/double=0、`sqrt(`/`pow(`=0，`static_assert` 冻结分析界。片上 128-bank Wv 缓存（128 KiB，`dim=1 complete`）+ `semantic_arr`(int32)/`l2_arr`(uint16) 两趟 min-max/输出。fixture 生成器直接复用冻结 golden（不复制数学）：真实 grounding/pointing（correct+wrong）、合成 zero/all-equal/G=1/mixed-sign、G=400 流式 xorshift64*（不嵌 400 行）、原语向量（含正/负精确 .5 tie）。**四道门全过**：Python 21/21 OK、g++ 103/103 PASS、ASan+UBSan 103/103 PASS、HLS C-sim 0 error 103/103 PASS。**csynth**：estimated **3.650 ns ≤ effective 3.650 ns**（target 5.0、默认 uncertainty 1.35、slack 0.000、Fmax 273.97 MHz）；资源 BRAM_18K **10.42%** / DSP **5.24%** / FF **2.32%** / LUT **17.99%** / URAM **0%**（均 <50%）；0 error、无 HLS 200-871/1016；Wv 载入 **65,539 cycles**、pass1 1442 + pass2 403 = **1845 cycles/group**；报告推导 G=80/260/400 = 213,139 / 545,239 / **803,539 cycles = 1.066 / 2.726 / 4.018 ms @200 MHz**（含 Wv 载入，≤4,000,000 cycles / 20 ms；综合估计非测量）。`score_q` 与既有 token selector 输入同 dtype/Q0.15/范围，**可直接消费**（host 串接/选择本轮未做）。产物：`fpga_accel/integer_selector_scorer_hls/`、`artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/{fixtures,csynth,logs}`、`reports/token_selector/2026-10-03-integer-selector-scorer-hls.{md,json}`。

更新时间：2026-10-03（**独立整数 Selector 打分核 HLS 实现与受控 csynth 完成**：四道数值门逐字节通过（Python 21、g++ 103、ASan+UBSan 103、HLS C-sim 0 error 103）；csynth estimated 3.650 ns ≤ 3.650 ns（默认 1.35 ns uncertainty 未放宽）、资源 BRAM 10.42/DSP 5.24/FF 2.32/LUT 17.99/URAM 0 %、0 error；Wv 载入 65,539 + 1845 cycles/group → G=400 803,539 cycles / 4.018 ms @200 MHz（≤4M/20 ms，综合估计）；→ **`INTEGER_SELECTOR_SCORER_HLS_PASS`**；CPU-only/只读复用已接受 Wv 与 fixture、未改既有文件、未 cosim/未导出 IP/未 implementation/未上板、非端到端 selector/非任务精度/非加速声明）

- [x] 2026-10-03 **位精确整数 Selector scorer 审计（零训练、预注册、validation-only、CPU-only、只读；未训练/未调参/未加载 4B/无 GPU/无联网/未读原图/未解析-哈希-打开 internal test NPZ 或报告/未改既有代码报告、checkpoint、cache、manifest、预注册、selector HLS 与前一 Wv 审计）→ `INTEGER_SCORER_NUMERICS_PASS`**：按预注册合同先写盘冻结 `configs/selector_integer_scorer_audit.preregistered.json`（SHA256 `70d9c6b1…` 为其中 frozen_inputs 之一；本 config 自身状态 `PREREGISTERED_VALIDATION_ONLY`，authority prompt `reports/token_selector/prompts/2026-10-03-selector-integer-scorer-audit-harness-prompt.md` SHA256 `1a2960fe…0fc5` 运行前后一致）。新增 `evaluation/selector_integer_scorer_audit.py`（SHA256 `3fcf91d2…0e74`）与聚焦测试 `evaluation/tests/test_selector_integer_scorer_audit.py`（SHA256 `3e4fde2c…61a3`），复用 `selector_wv_scale_regularization.quantize_weight_tensor`/`per_tensor_scale_vector`、`selector_int8_numerical_audit.{quantize_activation,primary_quantized_query,quantized_visual_projection,quantized_combined_scores,float_reference_scores,exact_int_matmul,selected_indices,bootstrap_mean_ci,load_prereg,verify_prereguration_identity}`、`prompt_projection_distillation.{split_sample_ids,jaccard,saliency_mass_retained,ndcg_for_selected_indices,spearman}.spearman,load_clean_manifest,load_index_metadata,load_split_samples,attach_spatial_targets,load_projection`，未复制第二套 split/loader/scorer/metric/selection/统计。**冻结整数链**（输入量化后无任何 float）：全局 A8 `group_mean_q`、per-tensor Wv INT8、逐向量 INT8 prompt query；`p[G,128]=group_mean_q@Wv_q^T` 精确 INT32；`dot=sum_d p·query_q` 与 `norm2=sum_d p^2` 显式 checked INT64；`semantic_q16=round_half_away((dot<<16)/floor_sqrt(norm2))`（零 norm→0，精确有符号有理 round-half-away，非 float）；`l2_int=floor_sqrt(sum_1024 group_mean_q^2)`；两项独立精确有理 min-max 到 [0,32767]（全等→0）；`score_q=clamp(round_half_away((semantic_norm_q15+l2_norm_q15)/2),0,32767)`；semantic 小数位冻结 16、不 sweep。参考臂 `PER_TENSOR_FLOAT_NORM`（已接受 per-tensor Wv float-normalisation 链路）、原始 float `FLOAT_REFERENCE`（原 Wt/Wv 冻结 scorer）、候选 `INTEGER_SCORER_Q16`。**隔离**：仅打开 76 val NPZ（38 图），train/test NPZ opened=[]、val aggregate `0439b7cb…68f4b` 匹配；运行前后 17 pins（8 frozen inputs + 3 checkpoints + 3 accepted Wv bundle×2 文件）逐字节不变；未解析/哈希/打开 internal-test NPZ 或报告、未读原图、未加载 4B、无 GPU/联网/训练/调参。**验证候选链不复制 Wv**：三 seed 从已接受 `selector_wv_scale_regularization/<seed>/{Wv_int8_rowmajor.bin,Wv_per_tensor_scale_f32le.bin}` 读入并 pin hash，与 `quantize_weight_tensor` 重算逐位一致；artifact 只在 metadata 中引用路径+SHA256，fixture npz 不含 Wv。**结果（76 val records、三 B seed 20261003/04/05、correct/wrong、K=48/64/96；每记录三 seed 中位、38 图两任务均值、2000 bootstrap 95% CI）**：候选 vs per-tensor Jaccard **0.994092/0.993939/0.996744**（CI 下界 0.990857/0.991118/0.995117）、replacement count **0.144737/0.197368/0.157895**、score Spearman **0.999957**（CI [0.999954,0.999960]）、mass delta **+0.000000/−0.000040/−0.000000**、NDCG delta **+0.000005/−0.000043/+0.000006**、exact-set rate（仅描述）**0.855263/0.815789/0.842105**；候选 vs 原始 float Jaccard **0.959012/0.962775/0.974957**、mass delta **−0.000244/−0.000164/−0.002779**、NDCG delta **−0.000171/−0.000029/−0.000223**；K48 Pointing prompt-utility retained-mass gap delta（候选−float）**−0.009432**（CI [−0.038977,+0.007120]）；Grounding 仅描述 **+0.000375**；三 seed K48 候选 vs per-tensor Jaccard **0.986617/0.988786/0.991944**。**整数宽度/操作合同**（`artifacts/token_selector/selector_integer_scorer_audit/integer_width_contract.json`，SHA256 `384d4d33…e45e`）：解析上界 projection INT32 16646144、dot INT64 272730423296、norm2 INT64 35468046088798208、shifted numerator INT64 17873661021126656、semantic_q16 INT32 201326592、l2_squared_sum INT64 16777216、l2_int INT32 4096；观测最大 projection 145258、dot 156390982、norm2 558499699697、shifted 10249239396352、semantic_q16 20902468、l2_int 607、score_q 30581，全部 `analytical_all_fit=true`/`observed_all_fit=true`/`all_fit_declared_widths=true`。**fixtures**：三 seed 各 grounding/pointing fixture 回读重放 `group_mean_q/query_q/Wv_q` 得 semantic_q16/l2_int/normalized/score_q/selected 与导出逐位一致（`roundtrip.ok=true`），含 projection checksum/bounds。**8/8 判据全过 → `INTEGER_SCORER_NUMERICS_PASS`**（非 synthesis/时延/精度/加速证明）。**测试**：聚焦 **25/25 OK**、全量 `unittest discover -s evaluation/tests` **715 OK**（前基线 690）；运行 runtime 8.41 s、peak RSS ≈785 MiB；`gpu_used=false`/`model_loaded=false`/`network_used=false`。证据：`reports/token_selector/2026-10-03-selector-integer-scorer-audit.{json,md}`（`b6a5b748…3067`/`02ad3330…b909`）、`...-records.csv`（`3f8ea377…b74b`）、`reports/token_selector/logs/2026-10-03-selector-integer-scorer-audit-{run,harness-result.json,focused-tests,full-tests}.log/json`、`artifacts/token_selector/selector_integer_scorer_audit/`。**限制**：Python 位精确整数参考而非 HLS 综合/上板；任意精度整数仅用于证明固定宽度与精确有理除法/开方；参考 float-normalisation 臂沿用已接受链路未重导；validation 已在开发中使用、internal test 保持封存。**精确下一决策（未执行，仅由 PI/Codex 决定）**：`INTEGER_SCORER_NUMERICS_PASS` 仅授权实现 HLS；须另立预注册与综合/时延/资源/端到端精度实验，不得据本审计声称上板加速、任务精度或 promotion；不得打开 internal test。

更新时间：2026-10-03（**位精确整数 Selector scorer 审计完成**：预注册冻结、运行前后 17 pins 逐字节不变；仅 76 val NPZ（38 图），test/train opened=[]、val aggregate 匹配；输入量化后纯整数链（INT32 投影、checked INT64 dot/norm2、精确有理 round-half-away 与 floor_sqrt、冻结 semantic 16 位、整数 min-max、λ=0.5）；候选 vs per-tensor K48/64/96 Jaccard 0.994092/0.993939/0.996744、replacement 0.145/0.197/0.158、Spearman 0.999957、mass delta ≈0、NDCG delta ≈0；候选 vs float mass delta −0.000244/−0.000164/−0.002779、NDCG delta −0.000171/−0.000029/−0.000223、Jaccard 0.959012/0.962775/0.974957；K48 Pointing gap delta（候选−float）−0.009432；三 seed K48 Jaccard 0.986617/0.988786/0.991944；整数宽度合同解析/观测全部放得下；三 seed fixture 回读逐位一致且不复制 Wv（引用已接受 bundle 并 pin hash）；8/8 判据全过 → **`INTEGER_SCORER_NUMERICS_PASS`**；聚焦 25 + 全量 715 测试 OK；CPU-only/只读、非完整定点 FPGA/HLS/时延/精度/加速/promotion 证明）

- [x] 2026-10-03 **Wv 尺度硬件正则化审计（零训练、预注册、validation-only、CPU-only、只读；未训练/未调参/未加载 4B/无 GPU/无联网/未读原图/未解析-哈希-打开 internal test NPZ 或报告/未改既有代码报告、checkpoint、cache、manifest、预注册、selector HLS 与前一审计）→ `PER_TENSOR_WV_ACCEPTABLE_FOR_HLS`**：新增 `evaluation/selector_wv_scale_regularization.py`（`443be77d…48e7`）与聚焦测试 `evaluation/tests/test_selector_wv_scale_regularization.py`（`b6b2ad60…ce54`），复用冻结 `split_sample_ids`/`load_split_samples`/`attach_spatial_targets`/`float_reference_scores`/`quantize_activation`/`quantize_weight_rows`/`primary_quantized_query`/`quantized_visual_projection`/`quantized_combined_scores`/`selected_indices`/`bootstrap_mean_ci`/`load_projection` 与 `jaccard`/`saliency_mass_retained`/`ndcg_for_selected_indices`/`spearman`，只新增逐张量量化原语 `quantize_weight_tensor`（`d=max(abs(Wv))/127; q=clamp(round_half_away(Wv/d),-127,127); Wv_hat=q*d`）。参考臂 `PER_ROW_WV`（逐输出行 Wv scale）与候选 `PER_TENSOR_WV`（整张 Wv 单一 scale）除 Wv 量化外完全一致：float Wt prompt_mean_vector→逐向量 INT8 query、每图单一 INT8 activation scale、精确 INT32 G×1024 投影、software-golden float 归一化/L2/min-max、Q0.15 score、3x3 anchor；76 val records/38 images、三 B seed（20261003/04/05）、K=48/64/96。**运行前后 14 pins 逐字节不变**；仅打开 76 val NPZ（test/train opened=[]、val aggregate `0439b7cb…68f4b` 匹配）；INT32 解析最坏界 1024/2560/128=16646144/41615360/2080768<2^31-1，观测最大 |累加| 逐行 227969 / 逐张量 145258，无溢出；三 seed 导出 fixture 回读重算 score_q/selected 与导出逐位一致、`wt_exported=false`、逐张量 scale 为 4 字节 little-endian float32。**候选 vs 逐行（correct，图像级 mean）**：K=48/64/96 Jaccard **0.998389/0.998785/0.998644**（CI 下界 0.996241/0.997571/0.997558）、replacement count **0.039474/0.039474/0.065789**、score Spearman **0.999982**、mass delta **0.000000/0.000000/0.000000**、NDCG delta **−0.000072/−0.000063/−0.000066**、exact-set rate（仅描述）**0.960526/0.960526/0.934211**；**候选 vs float**：mass delta **−0.000321/−0.000373/−0.003206**、NDCG delta **−0.000255/+0.000048/−0.000385**、Jaccard 0.959612/0.962334/0.976286；K48 Pointing prompt-utility retained-mass gap delta（候选−float）**−0.009432**（CI [−0.038977,+0.007120]）；三 seed K48 候选-vs-逐行 mean Jaccard **0.998926/0.994629/0.994629**。8 项冻结判据全 True → **`PER_TENSOR_WV_ACCEPTABLE_FOR_HLS`**。**产物**：扁平 canonical `reports/token_selector/2026-10-03-selector-wv-scale-regularization.{json(3f110220…f22e),md(41115085…0737)}` 与 `…-records.csv(d84b78ee…95ed)`，并在同名目录镜像同一三件套；`reports/token_selector/logs/2026-10-03-selector-wv-scale-regularization-{harness-result.json,run.log,focused-tests.log,full-tests.log}`，以及 `artifacts/token_selector/selector_wv_scale_regularization/<seed>/{Wv_int8_rowmajor.bin,Wv_per_tensor_scale_f32le.bin,fixture_grounding.npz,fixture_pointing.npz,metadata.json}`（每 seed ~676 KB、共 ~2.0 MB；主 bundle 不导出 Wt）。**测试**：聚焦 **27/27 OK**、全量 `unittest discover -s evaluation/tests` **690 OK**（基线 663）。**限制**：float L2 归一化/倒数平方根/min-max 仍是 software golden，本审计未证明完整定点 FPGA scorer、HLS/位级等价、延迟、端到端精度或发表 readiness；Grounding prompt-utility 仅描述（目标嵌套）。**精确下一步**：运行 `/home/zhaoxiangyu/conda-envs/robobrain25/bin/python -m evaluation.selector_wv_scale_regularization` 与 `/home/zhaoxiangyu/conda-envs/robobrain25/bin/python -m unittest evaluation.tests.test_selector_wv_scale_regularization -v` 复核；在获得新授权前不进入 HLS/位级等价、延迟或精度声明。

更新时间：2026-10-03（**Wv 尺度硬件正则化审计完成**：预注册冻结、运行前后 14 pins 逐字节不变；仅 76 val NPZ，test/train opened=[]；候选 vs 逐行 K48/64/96 Jaccard 0.998389/0.998785/0.998644、replacement 0.039/0.039/0.066、Spearman 0.999982；候选 vs float mass delta −0.000321/−0.000373/−0.003206、NDCG delta −0.000255/+0.000048/−0.000385；K48 Pointing gap delta −0.009432；三 seed K48 Jaccard 0.998926/0.994629/0.994629；8/8 判据全过 → **`PER_TENSOR_WV_ACCEPTABLE_FOR_HLS`**；fixture 回读逐位一致、不导出 Wt；聚焦 27 + 全量 690 测试 OK；CPU-only/只读、非完整定点 FPGA 证明、无精度/延迟/加速/promotion 声明）

- [x] 2026-10-03 **Selector INT8 数值审计（零训练、预注册、validation-only、CPU-only、只读；未训练/未调参/未加载 4B/无 GPU/无联网/未读原图/未解析-哈希-打开 internal test NPZ 或报告/未改既有文件与预注册）→ `INT8_NUMERICS_NOT_ESTABLISHED`**：新增 `evaluation/selector_int8_numerical_audit.py`（`4329e116…05dc`）与聚焦测试 `evaluation/tests/test_selector_int8_numerical_audit.py`（`81e2bda2…3ad9`），复用冻结 `split_sample_ids`/`load_split_samples`/`attach_spatial_targets`/`score_candidate`/`combine_scores`/`_select_groups`/`saliency_mass_retained`/`ndcg_for_selected_indices`/`jaccard`/`spearman` 与三 B 臂 best checkpoint、76 val records/38 images；实现两条 round-half-away-from-zero INT8 链：主臂 `SPLIT_WV_Q8_A8_Q8QUERY`（float Wt 查询→逐向量 INT8；每图全局 group_mean INT8；Wv 逐输出行 INT8；精确 INT32 投影；反量化行 L2；Q0.15 score 选择）与描述性对照 `FULL_WT_WV_Q8_A8`（另量化 Wt/每 prompt 张量，不参与决策）。**运行前后 15 pins 逐字节不变**；仅打开 76 val NPZ（test/train opened=[]、val aggregate `0439b7cb…68f4b` 匹配）；INT32 解析最坏界 1024/2560/128 = 16646144/41615360/2080768 < 2^31-1，观测最大累加 227969/1023052，无溢出。**主臂 vs float（图像级 mean）**：K=48/64/96 Jaccard **0.9591/0.9623/0.9766**、exact-set rate **0.2237/0.1447/0.1842**、retained-mass delta **−0.000253/−0.000373/−0.002636**、selected-NDCG delta **−0.000093/+0.000200/−0.000054**；K48 Pointing prompt-utility retained-mass gap delta（quant−float）**−0.009432**（CI [−0.038977,+0.007120]）；三 seed K48 mean Jaccard **0.9598/0.9512/0.9614**。按冻结规则仅 **exact-set match rate ≥ 0.75** 失败（其余 7 项判据全 True）→ **`INT8_NUMERICS_NOT_ESTABLISHED`**。**产物**：`reports/token_selector/2026-10-03-selector-int8-numerical-audit.{json,md,-records.csv}`（JSON `d2539b05…9100`、MD `af97da0c…3427d`、CSV `c720168d…e3db84`）、run/focused/full/harness 日志，以及 `artifacts/token_selector/selector_int8_numerical_audit/<seed>/{Wv_int8_rowmajor.bin,Wv_row_scales_f32le.bin,fixture_grounding.npz,fixture_pointing.npz,metadata.json}`（每 seed ~689 KB、共 ~2.1 MB；主 bundle 不导出 Wt；fixture 回读重算 score_q/selected 与导出逐位一致）。**测试**：聚焦 **24/24 OK**、全量 `unittest discover -s evaluation/tests` **663 OK**（基线 639）。**限制**：float L2 归一化/倒数平方根/min-max 仍是 software golden，本审计未证明完整定点 FPGA scorer、无精度/延迟/端到端加速/promotion 声明；Grounding prompt-utility 仅描述（目标嵌套）。

更新时间：2026-10-03（**Selector INT8 数值审计完成**：预注册冻结、运行前后 15 pins 逐字节不变；仅 76 val NPZ，test/train opened=[]；主臂 K48/64/96 Jaccard 0.9591/0.9623/0.9766、mass delta −0.000253/−0.000373/−0.002636、NDCG delta −0.000093/+0.000200/−0.000054，K48 Pointing gap delta −0.009432，三 seed K48 Jaccard 0.9598/0.9512/0.9614；仅 exact-set rate 0.2237/0.1447/0.1842 < 0.75 → **`INT8_NUMERICS_NOT_ESTABLISHED`**；INT32 界 16646144/41615360/2080768 无溢出；聚焦 24 + 全量 663 测试 OK；CPU-only/只读、非完整定点 FPGA 证明、无精度/加速/promotion 声明）

- [x] 2026-10-03 **任务对嵌套负样本有效性审计（零训练、CPU-only、只读、解释性；未打开 NPZ/原图/权重、无 GPU/联网、未访问 internal test、未改既有文件与预注册）→ `SYMMETRIC_NEGATIVE_INVALID`**：先保存提示 `reports/token_selector/prompts/2026-10-03-task-pair-nesting-audit-harness-prompt.md`（SHA256 `6e369bad…ad92`），并在读取任何 operational records CSV / clean manifest 内容前，仅以 SHA256 哈希三个输入后写盘冻结 `configs/task_pair_nesting_audit.preregistered.json`（`a9ca0acc…0316`）与 `reports/token_selector/2026-10-03-task-pair-nesting-audit-preregistration.json`（`35ea5443…f892`，`self_sha256_embedded=false`）；输入 pins：clean manifest `acab1a32…9213`、records CSV `8fe570a5…6cf4`、operational JSON `6980fda9…8b9f`，运行前后逐字节不变。新增 `evaluation/task_pair_nesting_audit.py`（仅复用 `point_inside_bbox`/`spearman`/`sha256_file`，未复制第二套 split/target/metric/statistic）与聚焦测试 `evaluation/tests/test_task_pair_nesting_audit.py`。**数据**：仅从 records CSV 恢复 val（76 records/38 images，grounding+pointing 各 1；CSV 全为 val，不读 index/NPZ）；几何仅取 clean manifest `ground_truth.{point_1000,bbox_1000}`。**结果**：point-inside-bbox 记录级 overall 比例 **1.000000（76/76）**、图像级跨任务 **1.000000（38/38）**，38/38 图 grounding 与 pointing 共享 ground_truth → `SAME_IMAGE_TASK_TARGETS_NESTED=true`（≥0.80）；bbox 面积（归一化）median **0.138215**（min 0.023580 / p25 0.060638 / p75 0.206783 / max 0.983556，零面积 0）；B 臂三 seed 中位数 K48 correct−wrong retained-mass gap：grounding mean **−0.001925**（median −0.000409）、pointing mean **+0.287765**（median +0.144167）、overall +0.142920；point_inside=true 层 n=38、false 层 n=0（无法分层对比）；bbox 面积 vs grounding gap Spearman **−0.049568**；correct−wrong selected Jaccard：grounding mean **0.246018**、pointing **0.246018**；与 operational JSON `task_k48` 容差 1e-9 **交叉核对一致**。按冻结规则 `NESTED 且 grounding gap<=0 且 pointing gap>0` → **`SYMMETRIC_NEGATIVE_INVALID`**（同图另一任务 prompt 对 pointing 不是有效负样本，因两任务目标嵌套）。**隔离/不变**：读路径集合恰为 3 输入 + 2 预注册文件，`npz_opened=[]`、原图/权重/GPU/联网/训练/内部测试全 False，`numpy.load` 未被调用；预注册与 3 输入运行前后逐字节不变。**测试**：聚焦 **33/33 OK**、全量 `unittest discover -s evaluation/tests` **639 OK**（基线 606）。**产物**：JSON `reports/token_selector/2026-10-03-task-pair-nesting-audit.json`、MD 同名 `.md`、CSV `…-records.csv`（76 行）、logs 四件套（run/focused/full/harness-result）。不改前一实验 `OPERATIONAL_PROMPT_UTILITY_NOT_ESTABLISHED`，不作精度/晋级声明。下一步命令：`/home/zhaoxiangyu/conda-envs/robobrain25/bin/python -m evaluation.task_pair_nesting_audit --summary`。

更新时间：2026-10-03（**任务对嵌套负样本有效性审计完成**：预注册先于读取输入冻结；仅 3 文本输入 + 2 预注册文件，未开 NPZ/原图/权重/internal test；point-inside-bbox 76/76、38/38 图共享 ground_truth → TARGETS_NESTED；grounding K48 gap −0.001925<=0、pointing +0.287765>0 → **`SYMMETRIC_NEGATIVE_INVALID`**；面积中位 0.1382、面积-gap Spearman −0.0496、Jaccard 0.2460；与 operational JSON 逐位一致；聚焦 33 + 全量 639 测试 OK；解释性、非精度、无 promotion、前一 NOT_ESTABLISHED 不变）


- [x] 2026-10-03 **部署态 Prompt 效用诊断（零训练、预注册、validation-only、CPU-only、只读；未训练/未调参/未加载 4B/无 GPU/无联网/未读原图/未重抽取/未解析-哈希-打开 internal test NPZ/未改预注册与既有代码报告）→ `OPERATIONAL_PROMPT_UTILITY_NOT_ESTABLISHED`**：先按 authority prompt 原文（`reports/token_selector/prompts/2026-10-03-operational-prompt-utility-diagnostic-harness-prompt.md`，SHA256 `a3edca6c…da1d9c`）冻结 `configs/operational_prompt_utility_diagnostic.preregistered.json`（`8413da0f591782bb9c7eddc80ab8a25da4df7fae66a58ad0ccefea11cb1c6775`）与 `reports/token_selector/2026-10-03-operational-prompt-utility-diagnostic-preregistration.json`（`626a3375…6bcdd`），pins 10 个输入 + A/B 两臂×3 seed 共 6 个 best checkpoint（16 项）运行前后逐字节不变，`verify_prereg_identity` 确认 config/report/authority prompt SHA 一致。新增 `evaluation/operational_prompt_utility_diagnostic.py`（`b7af91ba…c1f75`）与聚焦测试 `evaluation/tests/test_operational_prompt_utility_diagnostic.py`（`94cd57dd…80153`）；复用 contrastive 实验 A/B 六 best checkpoint 与同一冻结 76 val records/38 images，逐记录/seed/K=48,64,96 分别用 correct prompt 与 same-image-other-task wrong prompt，二者走**完全相同**部署链：冻结 `prompt_mean_vector` scorer → `combine_scores(lambda=0.5)` → 3x3 `select_real_groups` anchor → `ndcg_for_selected_indices`；输出 retained spatial-target mass、selected-set NDCG、correct-vs-wrong selected Jaccard/overlap。**聚合（冻结）**：每图先对 grounding/pointing 记录取均值，再跨 3 seed 取中位数，最后按 38 张独立图像做 2000 bootstrap + 10000 sign-flip + dz；primary Holm 家族恰 2 项且仅 K48 retained mass。**结果**：P1（B correct−wrong selector mass K48）mean **+0.137908**（CI [+0.072964, +0.205801]、dz 0.6503、raw p 0.0004、Holm 0.0008、+3/3）；P2（(B gap)−(A gap)）mean **+0.085962**（CI [+0.035535, +0.136801]、dz 0.5247、raw p 0.0018、Holm 0.0018、+3/3）→ **P1/P2 全过**；S2 B K48 gap NDCG mean **+0.052811**>0、S3 B K48 correct-vs-wrong Jaccard mean **0.246018**<0.95、S4 上一 contrastive G1/G4/G5/G7 全过且 pins/identity 一致，但 **S1 失败**：grounding K48 correct−wrong mass mean **−0.001925**（pointing **+0.287765**，非同号正）→ 按冻结 decision rule `OPERATIONAL_PROMPT_UTILITY_NOT_ESTABLISHED`。描述性：K64 B gap mass +0.133796、K96 +0.117520、A 臂 K48 +0.051946；B K48 jaccard mean 0.246018（CI [0.219642,0.273705]）。**隔离/不变**：仅打开 76 val NPZ（`opened_only_val_npz=true`、`test_npz_opened=[]`、`train_npz_opened=[]`、test 未解析/哈希/打开）；val NPZ aggregate `0439b7cb…68f4b` 与预注册一致；上一 contrastive 报告 JSON 仅用于 S4 的 G1/G4/G5/G7 布尔（同时被 pins 逐字节哈希），其 validation 统计与 val-samples CSV 均未解析、绝不作为 metric source；预注册 config/report 运行前后 SHA 不变。**输出**：json `6980fda9…78b9f`、md `2bda3221…53a1d7`、records.csv `8fe570a5…f6cf4`，logs run/focused-tests/full-tests/harness-result（`bdfb63af…0e8bb`）。**限制**：已使用 validation 的探索性机制证据，不是最终精度/CV-Bench 精度；不改上一 `PROMPT_CONTRASTIVE_STUDENT_NOT_ESTABLISHED` 结论；无 promotion、无硬件/加速声明；通过 P1/P2 本身不构成任务精度或晋级证据。

更新时间：2026-10-03（**部署态 Prompt 效用诊断完成**：预注册冻结 + 运行前后逐字节校验；仅 76 val NPZ、test/train 打开 []；P1 **+0.1379**（Holm 0.0008）、P2 **+0.0860**（Holm 0.0018）全过，S2/S3/S4 过但 S1 grounding **−0.0019**（pointing +0.2878）→ `OPERATIONAL_PROMPT_UTILITY_NOT_ESTABLISHED`；K64 +0.1338、K96 +0.1175；聚焦 21 + 全量 606 测试 OK；探索性/已用 validation/非最终精度/无 promotion、未改预注册与既有代码报告）

- [x] 2026-10-03 **预注册 Prompt-contrastive 空间学生 train/val 实验（development validation、CPU-only float32、两臂、三 seed、30 epoch；未加载 4B/未重抽取/未读原图/未跑 CV-Bench/无 GPU/无联网/未解析-哈希-打开 internal test NPZ/未改既有 train_spatial_prompt_selector 与既有报告）→ `PROMPT_CONTRASTIVE_STUDENT_NOT_ESTABLISHED`**：先写盘冻结提示 `reports/token_selector/prompts/2026-10-03-spatial-prompt-contrastive-trainval-harness-prompt.md`（`d02fb7c9…67c7`）、config `configs/spatial_prompt_contrastive_trainval.preregistered.json`（`12d0aa9f…dbbd`）与 `reports/token_selector/2026-10-03-spatial-prompt-contrastive-trainval-preregistration.json`（`5513f89c…816b`，`self_sha256_embedded=false`），均在首次读 val NPZ 之前完成；新增 `evaluation/train_spatial_prompt_contrastive_selector.py`（`51fd7ad6…9a94`）与聚焦测试 `evaluation/tests/test_train_spatial_prompt_contrastive_selector.py`（`8c1bcdc3…7024`），复用 `train_spatial_prompt_selector.{attach_spatial_targets,build_image_pairs,record_kl,val_statistics,_tensorize}`、`pair_loss_feasibility_audit.student_scores`、`prompt_identifiability_diagnostic.paired_effect`、`frozen_scorer_reduction_ablation.{holm_bonferroni,hardware_contract_for}`、`prompt_projection_distillation.{split_sample_ids,save_projection_npz}`、`train_pairwise_prompt_selector.{load_split_samples,epoch_batch_order,select_best_epoch}`，未复制第二套 split/loader/target/scorer/statistics；架构冻结 `Wt[128,2560]`/`Wv[128,1024]`/`prompt_mean_vector`/`[1,128]` query/Gx128 dot。**两臂**：`A_SPATIAL_KL_CONTROL` total=base=(KL(tg‖pg_correct)+KL(tp‖pp_correct))/2；`B_SPATIAL_KL_CONTRASTIVE` total=base+1.0·contrastive、contrastive=(softplus(KL_g_correct−KL_g_wrong)+softplus(KL_p_correct−KL_p_wrong))/2，wrong 对 grounding 用同图 pointing prompt、反之亦然；所有 KL=softmax(score/0.1) 无 T²；margin=0、softplus temperature=1，未调参。A 按 val base、B 按 val total 选 best（同时报告 base/contrastive/total）。**优化/一致性**：seeds 20261003/04/05、AdamW lr=1e-3 wd=0、batch=4 image pairs、clip=1、torch 单线程、确定性；每 seed 两臂 epoch0 权重 hash 完全一致（`f6467f8f…`/`386dc2fc…`/`2a2311a1…`）、batch 顺序共享。**隔离/不变**：只打开 318 train+76 val=394 NPZ（`opened_only_train_val_npz=true`、`test_npz_opened=[]`、`test_npz_hashed_count=0`、`test_used_for_selection=false`）；prereg config/report 与 14 个 pinned 输入 + `evaluation/train_spatial_prompt_selector.py`（`1308e8a0…9fd2`）运行前后逐字节不变（18/18 checks True），val NPZ aggregate `0439b7cb…68f4b`。**五项 primary（image-level、per-image 三 seed 中位 delta、2000 bootstrap、10000 sign-flip）**：P1 B−A (correct−same) Spearman **+0.006824**（CI[−0.002665,+0.016364]、dz 0.220、Holm 0.33617、+2/3）；P2 同 NDCG **+0.011873**（CI[+0.003461,+0.021408]、dz 0.411、Holm 0.039、+3/3）；P3 B 绝对 (correct−same) Spearman **+0.014950**（CI[−0.005207,+0.034627]、dz 0.233、Holm 0.33617、+3/3）；P4 同 NDCG **+0.052582**（CI[+0.023012,+0.083270]、dz 0.552、Holm 0.0048、+3/3）；P5 B selector−L2+anchor K48 retained mass **+0.226936**（CI[+0.120464,+0.335391]、dz 0.676、Holm 0.002、+3/3，按 selector−baseline 正确符号、单测防回归）。**Gates**：G1 非劣 True（B−A correct Spearman mean −0.005820/CI_low −0.013497、NDCG −0.004087/−0.010059，均 ≥ −0.02/−0.05 且 ≥2/3 seed）、G2 **False**（P1 CI lower<0）、G3 **False**（P3 CI lower<0）、G4 True、G5 True（B 每 seed exact invariance=0）、G6 True（B best_epoch=[3,5,4]、best val total 3.566/3.567/3.557 < epoch0 3.982/4.065/4.057）、G7 True → 因 Spearman 判别未显著仅 `NOT_ESTABLISHED`；B best val base 2.929/2.938/2.922 与 A 2.928/2.913/2.915 基本持平（正确对齐非劣）。**结论（development validation，不得称最终精度；门槛未 posthoc 改动）**：固定权重同图错误 Prompt contrastive 显著提升 NDCG 型判别（P2/P4）并保持/提升部署 retained-mass 效用（P5 +0.227、Holm 0.002），但 Spearman 型判别 (P1/P3) CI 下界仍 ≤0 → 合同判 NOT_ESTABLISHED；既有 G3 符号缺陷仅历史引用、本实验 P5 用 selector−baseline。**测试**：聚焦 **18/18 OK**、全量 `unittest discover -s evaluation/tests` **585 OK**。**证据**：`reports/token_selector/2026-10-03-spatial-prompt-contrastive-trainval.{json,md,epochs.csv,val-samples.csv}`（json `4fb9a9ac…7597`、md `9ac2b483…fede`）、`reports/token_selector/logs/2026-10-03-spatial-prompt-contrastive-trainval-{run,focused-tests,full-tests}.log` 与 `…-harness-result.json`（`2c14f88f…f721`）、`artifacts/token_selector/spatial_prompt_contrastive_trainval/<arm>/<seed>/{best.npz,final.npz,metadata.json}`。**精确下一步命令（未执行，需 PI 授权）**：`/home/zhaoxiangyu/conda-envs/robobrain25/bin/python -m evaluation.train_spatial_prompt_contrastive_selector --harness-result reports/token_selector/logs/2026-10-03-spatial-prompt-contrastive-trainval-harness-result.json`（仅复现冻结合同；下一步判读为 `next_decision=HOLD_NOT_ESTABLISHED_NO_PROMOTION`，不再改阈值/权重/门控）。

更新时间：2026-10-03（**Prompt-contrastive 空间学生 train/val 完成**：预注册+运行前后 SHA 不变；仅 394 train/val NPZ、test 未解析/哈希/打开；P2 NDCG +0.0119（Holm 0.039）、P4 +0.0526（Holm 0.0048）、P5 selector−baseline +0.2269（Holm 0.002）显著正，但 P1 Spearman +0.0068（CI lower −0.0027）、P3 +0.0150（CI lower −0.0052）未显著 → G2/G3 False → `PROMPT_CONTRASTIVE_STUDENT_NOT_ESTABLISHED`；聚焦 18 + 全量 585 测试 OK；development validation、非最终精度、无 promotion/无硬件声明）

- [x] 2026-10-03 **接管完成：空间学生部署链路失效分解（EXPLORATORY、validation 已使用、CPU-only、只读；未训练/未调参/未加载 4B/无 GPU/无联网/未解析 internal test NPZ/未改既有报告、训练代码与两份预注册）→ 判因 `HEADROOM_PRESENT`；已修正新脚本/新测试的 3 处问题**：审校 `reports/token_selector/prompts/2026-10-03-spatial-selector-failure-diagnostic-harness-prompt.md` 后确认先前进程冻结的预注册（config `2593616a…d1e67`、preregistration report `615882c7…e49ac`、prompt `13092c8e…0283e`）与执行合同一致且**逐字节未改**；修复：(1) `interpretation.within_task_heterogeneity`/`summary` 原为手写字符串（含硬编码 28/38、10/38、10x16），改为由冻结数值实时计算；(2) `isolation.restricted_val_report_files_parsed` 原恒为 False，与实际读取 trainval report JSON 做 G3 符号核对自相矛盾，改为按 `prior_report_consistency.available` 如实上报并加注（val-samples CSV 仍从不解析、绝不作为 metric source）；(3) `harness_payload` 补齐合同 item 10 决策字段（`task_stratification`、`lambda_sweep_best`、`limitations`、`l2_anchor_k48_overall_mass_mean`），`render_markdown` 增加实时生成的判因说明。**数据与隔离**：仅打开 76 个 val NPZ（38 图），`test_npz_opened=[]`、`train_npz_opened=[]`、`test_npz_resolved_hashed_or_opened=False`；`B_SPATIAL_KL_PRIMARY` 三 seed（20261003/04/05）best checkpoint；预注册 config/report 运行前后 SHA 不变、`verify_prereg_pins` 103 项 pin 全 True（含 76 val NPZ、19 code、3 checkpoint+metadata、2 restricted）、val NPZ aggregate `0439b7cb…68f4b` 与预注册一致。**K48 retained spatial-target mass 图像级均值 delta（mode − L2+anchor）**：semantic+anchor **+0.237814**、combined λ=0.5+anchor **+0.246583**、semantic_plain − L2_plain **+0.261678**、oracle+anchor **+0.737096**（>0.10 → `HEADROOM_PRESENT`）；分任务 grounding **+0.121060**、pointing **+0.371635**（同号，`TASK_SPECIFIC` 不触发）；λ sweep 仅描述，K48 overall retained-mass mean 最佳 λ=**0.70**（0.401517），λ=0.5 为 0.396948、λ=1.0 为 0.388179、λ=0 为 0.150366；`FUSION_DAMAGE`/`HEAD_SIGNAL_WEAK`/`ANCHOR_BUDGET_DAMAGE` 均不触发（semantic_anchor>0 且 combined>0，30/30 图 hetero 28 better/10 worse，唯一负网格 10x16 G=160 为 −0.487764）。**发现既有报告符号缺陷（只读、未改）**：`2026-10-03-spatial-student-trainval.json` 的 `B_selector_minus_L2_anchor_mass_k48` mean = **−0.246583**，但 `aggregate_validation` 对 `selector_minus_baseline` 计算 `base − metric = l2_anchor − selector`，与其 stated definition `selector − L2+anchor` 反向；按定义重算 **+0.246583**，`sign_consistent_with_stated_definition=False`，既有报告未编辑。**验证**：聚焦 `evaluation/tests/test_spatial_selector_failure_diagnostic.py` **22/22 OK**、全量 `unittest discover -s evaluation/tests` **567/567 OK**（未改任何既有测试）。**files**：json `9a7ea53c…e683d`、md `6e5462d8…11fb`、records.csv `cb2c078d…163a`（8892 行 = 3 seed×76 rec×3 K×13 mode）、harness-result `c5541352…396f`、run `d30e86bd…b21a`、focused `cc76f2c2…e7b`、full `87111c8d…e76d`；script **`2beda63f…fbd`**、test **`bcbb4a38…115`**（下方同名旧条目的 script `167f5fcc…`/test `69196a98…` 为接管前版本，SHA 已被本条取代，历史保留）。**限制**：探索性、validation 已使用、非最终精度/promotion 结论、未触碰 internal test、oracle 仅为可达上界、图像级统计、λ sweep 无显著性。**精确下一决策条件（需 PI/Codex 决定，本任务不执行）**：先决定是否为既有 G3 符号反向发布只读更正说明（禁止改原报告）；若继续空间学生路线，须新预注册并只针对 `HEADROOM_PRESENT` 与最小网格（10x16/G=160）异质性做 train/val-only 实验；下一验证命令：`cd <root> && /home/zhaoxiangyu/conda-envs/robobrain25/bin/python -m evaluation.spatial_selector_failure_diagnostic --root .`。

更新时间：2026-10-03（**接管并修复完成、合同闭环**：预注册冻结文件未改；仅 76 val NPZ、test/train 打开 []；K48 图像级均值 delta semantic+anchor **+0.238**、combined λ=0.5 **+0.247**、oracle **+0.737**，仅 **`HEADROOM_PRESENT`**；λ=0.70 描述性最佳（0.4015）；既有 G3 mean −0.247 实为符号反向、按定义应为 **+0.247**（未改原报告）；聚焦 22 + 全量 567 测试 OK；探索性/非精度/无 promotion/无训练授权）

- [x] 2026-10-03 **空间学生部署链路失效分解（EXPLORATORY、validation 已使用、CPU-only、只读；未训练/未调参/未加载 4B/无 GPU/无联网/未解析 internal test NPZ/未改既有报告与训练代码）→ 判因 `HEADROOM_PRESENT`；部署质量链路未失效，既有 G3 符号被反向**：按执行合同在首次读取 val 之前写盘冻结 `configs/spatial_selector_failure_diagnostic.preregistered.json`（SHA256 `2593616a…d1e67`）与 `reports/token_selector/2026-10-03-spatial-selector-failure-diagnostic-preregistration.json`（SHA256 `615882c7…e49ac`，`self_sha256_embedded=false`，记录 `created_at_utc` 与全部输入路径/SHA256），并将提示原文保存到 `reports/token_selector/prompts/2026-10-03-spatial-selector-failure-diagnostic-harness-prompt.md`（`13092c8e…0283e`）。冻结 7 个模式族 / 13 个具体模式（`l2_plain`、`l2_anchor`、`semantic_plain`、`semantic_anchor`、`combined_anchor λ=0.5`、8 个 λ∈{0,0.05,0.1,0.2,0.3,0.5,0.7,1.0} 的 `lambda_sweep_anchor`、`oracle_anchor`）、K=48/64/96、按 overall/grounding/pointing/group_count/grid/size_band 分层、三 seed 先对每图取中位数再按 38 张独立图像汇总 mean/median/2000-bootstrap CI/better-equal-worse，以及 5 条判因定义（均冻结于预注册）。新增 `evaluation/spatial_selector_failure_diagnostic.py`（`167f5fcc…5a5b`）与聚焦测试 `evaluation/tests/test_spatial_selector_failure_diagnostic.py`（`69196a98…30a9`）；全程复用 `train_spatial_prompt_selector.{attach_spatial_targets,_select_groups}`、`frozen_scorer_reduction_ablation.score_candidate`、`hardware_regular_token_selector.{combine_scores,select_real_groups}`、`prompt_projection_distillation.{ndcg_for_selected_indices,ndcg_at_k,saliency_mass_retained,topk_indices,l2_baseline_scores,jaccard}`、`prompt_identifiability_diagnostic.paired_effect`、`run_hf_hardware_regular_b1.load_projection`，未复制第二套算法。**数据与隔离**：仅打开 76 个 val NPZ（38 图），train/test NPZ 打开均为 `[]`；使用 `B_SPATIAL_KL_PRIMARY` 三 seed（20261003/04/05）best checkpoint；预注册 config/report 运行前后 SHA 不变、全部 76 val NPZ + 19 code + 3 checkpoint + 2 restricted SHA 不变、val NPZ aggregate `0439b7cb…68f4b` 与 pin 一致。**K48 retained spatial-target mass 图像级均值 delta（mode − L2+anchor）**：semantic+anchor **+0.237814**、combined λ=0.5+anchor **+0.246583**、semantic_plain − L2_plain **+0.261678**、oracle+anchor **+0.737096**；修正 selected-set NDCG delta：combined λ=0.5 **+0.144019**、semantic+anchor **+0.167890**。**分任务（λ=0.5+anchor）**：grounding **+0.121060**、pointing **+0.371635**（同号，`TASK_SPECIFIC` 未触发）；better/equal/worse = **28/0/10**；size band medium +0.2944、large +0.2336、small（n=1，10x16，G=160）−0.4878。**λ sweep（仅描述、不做显著性/晋级）**：mass 最佳 λ=0.70 **+0.251152**，NDCG 最佳 λ=1.00 **+0.167890**；λ=0 与 L2+anchor 逐位相同（Jaccard=1.0）。**判因**：仅 **`HEADROOM_PRESENT`**（oracle 相对 L2+anchor +0.737096 > 0.10）；`FUSION_DAMAGE` / `HEAD_SIGNAL_WEAK` / `ANCHOR_BUDGET_DAMAGE` / `TASK_SPECIFIC` 均未触发。**只读符号核对**：既有 `2026-10-03-spatial-student-trainval.json` 的 `primary_tests.B_selector_minus_L2_anchor_mass_k48` mean = **−0.246583**（CI[−0.352744,−0.141682]），但按预注册定义 selector−L2+anchor、由冻结 val CSV 与本次重算独立得到 **+0.246583**；根因是 `aggregate_validation` 对该 `selector_minus_baseline` 键按 `base−metric = l2_anchor−selector` 计算，与其定义反向。**既有报告/训练代码未改，仅报告。** **限制**：这是已使用 validation 的探索性诊断，只能指导下一次预注册，不能作最终精度/promotion 结论；retained mass 非任务精度；oracle 只是可达上界；internal test 未解析/未跑；无 GPU/训练/联网/4B。**测试**：聚焦 **22/22 OK**、全量 `unittest discover -s evaluation/tests` **567 OK**（基线 545）。**精确下一决策条件（未执行）**：PI/Codex 必须先在独立复核中判定 `aggregate_validation` 的 `selector_minus_baseline` 符号缺陷并按其定义重算 G3；若仅 G3 修复而 G2（same-image prompt 敏感）仍失败，则 `SPATIAL_STUDENT_NOT_ESTABLISHED` 不变、不得晋级；只有在下一次预注册中冻结修正后的 G3 判据、明确 fresh/held-out 确认方式并获训练授权后，才可讨论 λ=0.5 部署。证据：`reports/token_selector/2026-10-03-spatial-selector-failure-diagnostic.{json,md,records.csv}`、`reports/token_selector/logs/2026-10-03-spatial-selector-failure-diagnostic-{run,focused-tests,full-tests,harness-result}`。

更新时间：2026-10-03（**空间学生部署链路失效分解完成**：预注册冻结 + 运行前后逐字节校验；仅 76 val NPZ、test/train 打开 []；K48 retained mass 图像级均值 delta semantic+anchor **+0.238**、combined λ=0.5+anchor **+0.247**、oracle **+0.737**，分任务同号；仅 **`HEADROOM_PRESENT`**；发现既有 G3 mean −0.247 实为 `l2_anchor−selector` 符号反向、按定义应为 **+0.247**；既有报告未改；聚焦 22 + 全量 567 测试 OK；探索性/非精度/无 promotion/无训练授权）

- [x] 2026-10-03 **预注册空间监督学生训练/验证实验（train/val-only、CPU-only float32、两臂、~458,752 参数小头、非 4B；未加载 4B/未重抽取/未读原图/未跑 CV-Bench/无 GPU/无联网/未开 internal test NPZ）→ `SPATIAL_STUDENT_NOT_ESTABLISHED`**：按预注册合同先写盘冻结 `configs/spatial_student_trainval.preregistered.json`（SHA256 `0ee94dca…19abf`）与 `reports/token_selector/2026-10-03-spatial-student-trainval-preregistration.json`（SHA256 `8da7bd2e…d4a8`，`self_sha256_embedded=false`），首次读 val 前完成；pin 9 个输入（config `44399c8e…ab28b`、index `02a50868…ee26`、clean manifest `acab1a32…9213`、lock `a723458d…e105`、train NPZ aggregate `33997394…3951`、corrected spatial-feasibility JSON `1bd2b9e4…c739` 与 script `1d033c40…8aec`、semantic control code `07b504d7…2fec`、hardware config `6350c89b…8142` 与 code `c55cfdc3…5932b`），运行前后逐文件逐字节不变（`hashes_unchanged` 全 True）。新增 `evaluation/train_spatial_prompt_selector.py`（SHA256 `1308e8a0…9fd2`）与聚焦测试 `evaluation/tests/test_train_spatial_prompt_selector.py`（SHA256 `2c529f90…98906`）；复用冻结 split/loader/scorer/空间目标/KL/统计/Holm/`select_real_groups`+`combine_scores`/`ndcg_for_selected_indices`，未复制第二套。**两臂**：`A_SEMANTIC_KL_CONTROL` target=cached `teacher_soft_target`；`B_SPATIAL_KL_PRIMARY` grounding=bbox/grid-cell 精确面积、pointing=bilinear point；逐记录 `KL(target‖softmax(score/0.1))`（无 T² 缩放），图对损失=grounding/pointing 记录 KL 均值；无 hinge/混合/margin/rank/辅助 loss/调参。**优化**：seeds `20261003/04/05`、AdamW lr=1e-3 wd=0、30 epoch、batch 4 图对、grad clip 1.0、CPU float32、单 torch 线程、确定性；每 seed 两臂 epoch0 权重 hash 完全一致（`f6467f8f…`/`386dc2fc…`/`2a2311a1…`）；best 仅按各臂自身 VAL KL、tie 取最早；只存 epoch0 hash + best/final。**结果（image 独立单位、per-image 跨 seed 中位 delta + 2000 bootstrap + 10000 sign-flip、seed 20261003）**：`best_epoch` A=[12,2,2]、B=[27,5,4]。主 Holm 家族恰 5 检验：B−A correct Spearman **+0.116239**（CI[0.072465,0.160201]、dz 0.8425、raw p=0.0003、Holm 0.0012、+3/3）；B−A correct full NDCG **+0.072518**（CI[0.023511,0.118175]、dz 0.4787、Holm 0.0117、+3/3）；B correct−same-image-other-task Spearman **+0.010183**（CI[−0.002846,0.022837]、dz 0.2438、Holm 0.1507、+3/3 → 不显著）；同 NDCG **+0.033670**（CI[0.011946,0.056137]、dz 0.4770、Holm 0.0117、+3/3）；B 部署 selector − L2+anchor K48 retained spatial mass **−0.246583**（CI[−0.352744,−0.141682]、dz −0.7394、Holm 0.0005、+0/3 → 显著更差）。**Gates**：G1=True、**G2=False**（Spearman CI 下界<0）、**G3=False**（selector 保留空间质量显著低于 L2+anchor）、G4=True（collapse=0 全 seed）、G5=True、G6=True → `all_pass=False` → `SPATIAL_STUDENT_NOT_ESTABLISHED`；A 为对照永不获胜。**隔离**：split 仅由 index metadata 计算（train/val/test=318/76/78 记录、159/38/39 图），只打开 train+val 394 NPZ（`opened_only_train_val_npz=true`、`test_npz_opened=[]`、`test_used_for_selection=false`）；train aggregate 与冻结一致。**测试**：聚焦 **16/16 OK**、全量 `unittest discover -s evaluation/tests` **545 OK**（基线 529）。**证据**：`reports/token_selector/2026-10-03-spatial-student-trainval.{json,md}`、`-epochs.csv`、`-val-samples.csv`、`artifacts/token_selector/spatial_student_trainval/<arm>/<seed>/{best.npz,final.npz,metadata.json}`、`reports/token_selector/logs/2026-10-03-spatial-student-trainval-{run,focused-tests,full-tests,harness-result}`。**限制**：仅内部 train/val sanity，非 CV-Bench 任务精度；无综合/时延/加速/上板声明；无 4B/重抽取/原图；internal test 未开未评。**精确下一步决策**：不再堆叠本 Wt/Wv 空间监督重训；若继续，需 PI 明确授权并预注册与 L2+anchor 基线正面对齐的新监督/选择目标（G3 显示 prompt 条件化部署选择器在 K48 反而低于 L2+anchor），或转向推理期 anchor 预算与 INT8/低秩约束，不得复用 internal test。

更新时间：2026-10-03（**预注册空间监督学生 train/val 实验完成**：两臂×三 seed×30 epoch、CPU-only float32、未开 internal test NPZ；B−A correct Spearman/NDCG 显著正（Holm 0.0012/0.0117，+3/3），但 G2 same-image-other-task Spearman CI 下界<0、G3 部署 selector K48 保留空间质量显著低于 L2+anchor（−0.247）→ `SPATIAL_STUDENT_NOT_ESTABLISHED`；聚焦 16 + 全量 545 测试 OK；预注册/输入运行前后 SHA 未变；非 CV-Bench 精度、无硬件/加速声明）

- [x] 2026-10-03 **Codex 独立审计缺陷更正：anchor-baseline NDCG 选择集（bounded correction；CPU-only、只读、train-only、无 GPU/模型/联网；预注册冻结不变）→ `SPATIAL_SUPERVISION_SIGNAL_PRESENT` 保持**：Codex 审校确认 `evaluation/spatial_teacher_feasibility_audit.py::anchor_baseline_record` 旧实现虽用 anchor 选定集计算保留目标质量，却用 `ndcg_at_k(scores,target,effective)` 计算 NDCG，忽略选定集并使 anchor NDCG 恒等于纯 L2 NDCG（旧 anchor NDCG 与 L2 逐位相同）。**更正**：在冻结度量层新增通用 helper `evaluation/prompt_projection_distillation.py::ndcg_for_selected_indices(selected,scores,relevance,k)`（显式校验：shape 匹配/非负 relevance/正 k/非空/去重/越界），anchor NDCG 改为**仅对 anchor 选定组按原始 L2 降序排序**（`np.lexsort` 低索引打破平局）计算 DCG，分母仍为全体组 ideal DCG@K；任何未选组不得进入预测 top-K；`ndcg_at_k` 语义未改。`select_real_groups` 选择规则、L2 输入、预注册空间目标公式、JS/TV 决策规则、split、输入、决策标签全部冻结。**旧→校正 anchor NDCG（mean，overall/grounding/pointing）**：K=48 `0.134532→0.135611` / `0.217840→0.219177` / `0.051223→0.052045`；K=64 `0.154315→0.155977` / `0.240733→0.243338` / `0.067897→0.068616`；K=96 `0.187391→0.190026` / `0.287698→0.291110` / `0.087084→0.088943`（9/9 桶变化；旧值即纯 L2 NDCG）。**anchor 保留目标质量逐位不变**（max abs diff `0.0`）。**主指标不变**：以 monkeypatch 旧公式在内存重建旧报告并与新报告逐字段比较，`decision_label`/`semantic_pair_signal`/`spatial_pair_signal`/`paired_differences`/`semantic_spatial_compatibility`/`l2_baseline`/`integrity`/`isolation`/`split_counts`/`geometry` 与 provenance 的 config/index/manifest/train NPZ aggregate 全部**精确相等**（重建旧 CSV sha 与冻结旧值一致 = `03bd34b0…9f94`）。**隔离**：`opened_only_train_npz=true`、`val_test_npz_opened=[]`、opened_npz_count=318、train aggregate `33997394…3951`。**预注册未编辑**：config `bfa19635…4490`、report `ce296bab…fcca9` 运行前后 SHA256 未变。**SHA 记录**：script 旧 `c800d12d…9983`→新 `1d033c40…8aec`；报告 JSON 旧 `85f83651…b154`→新 `1bd2b9e4…c739`；MD 旧 `f4fb194d…99a0`→新 `a28aadfd…18a1`；CSV 不变 `03bd34b0…9f94`（anchor 不入 CSV）；durable harness 旧 `a22ea2ba…e8e30`→新 `edd07f3f…a6abe`；冻结度量层 `prompt_projection_distillation.py` `cd2d4ccc…b98`（仅新增 helper）。**测试**：聚焦 **38/38 OK**（新增 7 个会令旧代码失败的用例：选定集 NDCG 与纯 L2 不同、选定集按 L2 降序低索引平局、未选高相关不进入 anchor DCG、identity bypass 等于纯 L2、retained mass 不变、显式校验、报告更正块一致性）；全量 `unittest discover -s evaluation/tests` **529 OK**（基线 522）。报告 JSON/MD 已加 `correction` 块（含旧/校正值、旧 SHA、`anchor_mass_unchanged=true`），旧历史未改写、缺陷未隐藏。**限制**：anchor baseline 仍是 zero-update 描述性参考而非训练 selector；不声称任务精度/FPGA 加速；可学习性与下游精度仍未验证，本更正不授权训练。证据：`reports/token_selector/2026-10-03-spatial-teacher-feasibility.{json,md}`、`reports/token_selector/logs/2026-10-03-spatial-teacher-feasibility-{run,focused-tests,full-tests}.log`、`reports/token_selector/logs/2026-10-03-spatial-teacher-feasibility-harness-result.json`。精确下一步（未执行）：交 Codex/PI 复核本更正；若接受，再单独决定是否设计可训练 scorer 与 promotion-gated 实验。

更新时间：2026-10-03（**Codex 审计 anchor-baseline NDCG 选择集缺陷已更正并复核**：新增 `ndcg_for_selected_indices`，anchor NDCG 仅对选定组按 L2 降序（低索引平局）排序；旧值=纯 L2；校正后 K=48/64/96 overall `0.134532→0.135611` / `0.154315→0.155977` / `0.187391→0.190026`，anchor retained mass 逐位不变（0.0），主指标/配对 JS/TV CI/`SPATIAL_SUPERVISION_SIGNAL_PRESENT` 精确不变，预注册 SHA 未变；聚焦 38 + 全量 529 测试 OK；CPU-only/只读、未训练/未调参、非 CV-Bench 精度、无加速声明）

- [x] 2026-10-03 **参数无关空间监督目标可行性审计（TRAIN-only、zero-update、CPU-only、只读；未训练/未调参/未保存新 Wt-Wv/未加载 4B/无 GPU/无联网/未读 val-test NPZ/未跑 CV-Bench）→ `SPATIAL_SUPERVISION_SIGNAL_PRESENT`**：按预注册要求先写盘冻结 `configs/spatial_teacher_feasibility.preregistered.json`（SHA256 `bfa1963597993ea9e3482735f16b2d430638ac3bf8e0682b5078f4e112974490`）与 `reports/token_selector/2026-10-03-spatial-teacher-feasibility-preregistration.json`（SHA256 `ce296bab00b82793b7f59781da126825680bb0d3cd8434e0e638f1f5bf1fcca9`，`self_sha256_embedded=false`），二者均逐文件 pin 6 个输入的路径+SHA256（config `44399c8e…ab28b`、index `02a50868…ee26`、clean manifest `acab1a32…9213`、lock `a723458d…e105`、teacher pair signal `09dbce65…52dd`、hardware regular `6350c89b…8142`）；脚本在计算任何新指标前校验 pin，运行前后各校验一次并报告逐字节不变（`byte_for_byte_unchanged=true`）。新增 `evaluation/spatial_teacher_feasibility_audit.py`（script sha `c800d12d…9983`）与聚焦测试 `evaluation/tests/test_spatial_teacher_feasibility_audit.py`，复用冻结 `split_sample_ids`/`load_feature_cache`/`pearson`/`spearman`/`topk_indices`/`jaccard`/`ndcg_at_k`/`saliency_mass_retained`/`l2_baseline_scores`、teacher-pair 审计的 JS/KL/TV/`describe`/index/manifest 加载与 npz aggregate、identifiability 的 `paired_effect`（2000 bootstrap+10000 sign-flip）与 `hardware_regular_token_selector.select_real_groups` 精确 3x3 anchor 规则；未复制第二套 split/metric/statistics，未改既有文件/app/models/报告。**冻结目标**：`grounding_area_target`=bbox 与 `H x W` 单元精确交集面积/总面积；`pointing_bilinear_target`=`u=x*W/1000-0.5`、`v=y*H/1000-0.5` 夹取后可分离四邻双线性权重、合并重复边界索引并归一化（无高斯/无带宽）；不与语义 teacher 混合、不看指标调公式。**数据合同**：join 472/472、缺/重/任务或图像身份不匹配=0、每图 `{grounding,pointing}` 完整、`group_count==grid_h*grid_w` 100%、bbox/point 有效 318/318、point-inside-bbox 318/318（1.0）、target 最大 |sum-1|=4.44e-16。**隔离**：split 仅由 index metadata 计算（train/val/test=318/76/78 记录、159/38/39 图），只打开 train 318 NPZ（`val_test_npz_opened=[]`、`opened_only_train_npz=true`、opened_npz_count=318、train npz aggregate `33997394…3951`）；测试用 spy loader 与「把 val/test NPZ 改写成非法字节」两法证明 train 指标/每图行/aggregate SHA 不变。**同图任务分离（159 图，image 为独立单位）**：语义 teacher（缓存 teacher_soft）JS mean **0.051067**/median 0.009236、TV 0.127210、argmax 不同比例 **0.119497**（与 2026-10-02 teacher-pair 审计 JS 0.0511 一致）；空间 teacher JS mean **0.518015**/median 0.539648、TV **0.866312**、argmax 不同比例 **0.962264**。**配对差 spatial−semantic（seed 20261003）**：JS mean **0.466949**、median 0.502143、std 0.165044、95% CI **[0.440491, 0.492981]**、p=**9.999e-05**、Cohen dz **2.829235**；TV mean **0.739101**、CI **[0.700647, 0.774453]**、dz **3.135090**；top-K disagreement 差 k48 +0.492348 CI[0.433178,0.552661]、k64 +0.412434、k96 +0.282972（均描述性）。**语义/空间兼容性**（Spearman、full NDCG，relevance=teacher_soft_target）：overall 0.125667/0.388966、grounding 0.180759/0.411134、pointing 0.070574/0.366799 → 目标不同，非自动判定某一方错误。**硬件规整基线**：L2（`l2_baseline_scores` on cached group_mean）overall K=48/64/96 NDCG 0.134532/0.154315/0.187391、保留目标质量 0.167937/0.240022/0.348751；anchor（`select_real_groups` 精确复用、以 L2 saliency 作排序分，`select_real_groups` 952 + `identity_bypass` 2）overall NDCG 同值、质量 0.171013/0.244024/0.356207。**运行**：runtime 1.65 s、peak RSS ≈1215 MiB、`gpu_used=false`/`model_loaded=false`/`network_used=false`；outputs csv `03bd34b0…9f94`、json `85f83651…f154`、md `f4fb194d…c99a0`。**测试**：聚焦 **31/31 OK**、全量 `unittest discover -s evaluation/tests` **522 OK**。**限制**：不训练、不建 student、不声称可学习；无下游/CV-Bench 精度、无 FPGA 加速；语义比较器仅为既有缓存 max-over-prompt-token teacher；grounding 面积目标与 pointing 双线性目标支撑结构不同（多单元面积 vs ≤4 点），部分分离来自该结构差异而非仅任务几何；anchor 基线在本 zero-update 审计中只是 L2 排序的 3x3 参考，非已训练 selector。**下一步（交 Codex/PI 决策，未执行）**：因 `SPATIAL_SUPERVISION_SIGNAL_PRESENT`，需 Codex/PI 决定是否设计可训练 scorer 并预注册 promotion-gated 实验；本审计不授权任何训练，且**可学习性与任务精度仍未验证**。证据：`configs/spatial_teacher_feasibility.preregistered.json`、`reports/token_selector/2026-10-03-spatial-teacher-feasibility-preregistration.json`、`evaluation/spatial_teacher_feasibility_audit.py`、`evaluation/tests/test_spatial_teacher_feasibility_audit.py`、`reports/token_selector/2026-10-03-spatial-teacher-feasibility.{csv,json,md}`、`reports/token_selector/logs/2026-10-03-spatial-teacher-feasibility-{run,focused-tests,full-tests}.log`、`reports/token_selector/logs/2026-10-03-spatial-teacher-feasibility-harness-result.json`。

更新时间：2026-10-03（**参数无关空间监督目标可行性审计完成**：预注册冻结+运行前后逐字节校验；train-only 318 NPZ、val_test_npz_opened=[]；空间同图 JS/TV 远高于语义 teacher → `SPATIAL_SUPERVISION_SIGNAL_PRESENT`（JS diff CI 下界 0.4405、TV 0.7006 均 >0）；兼容性 overall Spearman 0.126/NDCG 0.389；L2 与 3x3 anchor 基线 K=48/64/96 已报告；聚焦 31 + 全量 522 测试 OK；CPU-only/只读、无训练/调参/模型/GPU/联网、未读 val/test NPZ、非 CV-Bench 精度、无加速声明；**可学习性与任务精度未验证，训练未授权**）

- [x] 2026-10-03 **轻量 Prompt 条件化 Token Selector 三臂重训可证伪实验（train/val-only、CPU-only float32、~458,752 参数小头、非 4B；未加载 4B/未重抽取/未跑 CV-Bench/无 GPU/无联网/未开 internal test NPZ）→ `NO_PROMOTION`**：按预注册合同先写盘冻结 `configs/prompt_mean_pairwise_retrain.preregistered.json`（SHA256 `c8327ddb529970377707fd15df8fee410f293e747ddf80cd88f23177c771ef3c`）与 `reports/token_selector/2026-10-03-pairwise-retrain-preregistration.json`（SHA256 `44f21c9cb6d1a7c1dca41900d367e08cc7a350ce44a99508cf5915f221562d99`，状态 `PREREGISTERED_TRAIN_VAL_ONLY`，不自嵌 SHA），首次读 val 前完成。三臂严格固定：`A_KL_ONLY_CONTROL`（L_total=L_base）、`B_PAIR_LAMBDA_1_CONTROL`（+1.0·L_pair）、`C_PAIR_LAMBDA_0P25_PRIMARY`（+14.280209646136903·L_pair，来自冻结 scale audit）；`L_base=0.5(KL(y_g‖p_g)+KL(y_p‖p_p))`、`D_swap=0.5(KL(y_g‖p_p)+KL(y_p‖p_g))`、`L_pair=relu(JS_nats(y_g,y_p)+L_base−D_swap)`。新增 `evaluation/train_pairwise_prompt_selector.py` + `evaluation/tests/test_train_pairwise_prompt_selector.py`；复用冻结 split/scorer（`prompt_mean_vector`）/donor（image derangement，seed 20261103，无自配同 task）/统计（`paired_effect` 2000 bootstrap + 10000 sign-flip、`holm_bonferroni`）实现，未复制第二套。**运行**：3 arms × 3 seeds × 30 epochs 全跑满，`torch.set_num_threads(1)`、AdamW lr=1e-3 wd=0 bs=4 images grad_clip=1.0；每 epoch 更新前评估并记录 train/val 的 L_base/L_pair/L_total/hinge_active_fraction/grad_norm_before_clip/batch_count；best 仅按该 arm val L_total 最小、tie 取更早；同 seed 三臂 epoch0 权重 hash 完全一致（`40bc0c1b…`/`f6467f8f…`/`386dc2fc…`）。`best_epoch`：A=2/12/2、B=2/12/6、C=23/5/28，全部 best<epoch0。**结果**：collapse 全臂 `exact_prompt_invariance_fraction=0.0`（C 通过）；median-seed val correct Spearman/NDCG：A 0.04986/0.84980、B 0.04593/0.84941、C 0.02860/0.84696；**primary C 未过 A（same-image：spearman mean +0.00623、CI lower −0.00382、p=0.2434、Holm=1.0；ndcg mean +0.00067、CI lower −0.00097）、未过 B（cross-image 两 metric CI lower<0 且 dz≈0.01/0.096）、未过 D（Spearman 0.02860 < 0.046475）**；A/B 亦未过 A/B（仅对照报告）。Holm family 固定 3 arms×2 controls×2 metrics=12。按冻结 winner 规则仅 C 且 A–F 全过才晋升 → `decision=NO_PROMOTION`、`winner=null`。**隔离**：loader 访问日志 + spy loader + 损坏 synthetic test NPZ 三测试证明只打开 318 train + 76 val = 394 NPZ，`test_npz_opened=[]`、test 78 records/39 images 未解析未评分；运行前后 prereg/draft/config/index 逐字节未变（`hashes_unchanged` 全 true）。**验证**：聚焦 `python -m unittest evaluation.tests.test_train_pairwise_prompt_selector -v` **17 OK**、全量 `python -m unittest discover -s evaluation/tests` **491 OK**；运行 `runtime_seconds=283.71`、`peak_rss_mib=2282.81`；无 4B/无 GPU/无联网。证据：`configs/prompt_mean_pairwise_retrain.preregistered.json`、`evaluation/train_pairwise_prompt_selector.py`、`evaluation/tests/test_train_pairwise_prompt_selector.py`、`reports/token_selector/2026-10-03-pairwise-retrain-{preregistration.json,epochs.csv,val-samples.csv,train-val.json,train-val.md}`、`reports/token_selector/logs/2026-10-03-pairwise-retrain-{run,focused-tests,full-tests}.log`、`reports/token_selector/logs/2026-10-03-pairwise-retrain-harness-result.json`（durable；`/tmp/robobrain_harness_result.json` 在本沙箱每次调用后不持久）、`artifacts/token_selector/pairwise_retrain/{A_KL_ONLY_CONTROL,B_PAIR_LAMBDA_1_CONTROL,C_PAIR_LAMBDA_0P25_PRIMARY}/{20261002,20261003,20261004}/{best,final}.npz+metadata.json`。精确下一步（未执行，需 PI/Codex 决策）：按冻结 `after_gate.if_fail` **停止投影 scorer 重训并重设计 teacher/监督**；不添加事后 margin/reducer/rank/epoch，不评估 internal test；若要重判须另立新预注册。

- [x] 2026-10-03 **Pair 梯度贡献集中度审计（train-only、zero-update、CPU-only、只读；未训练/未调参/未保存新 Wt-Wv/未加载 4B/无 GPU/无联网/未读 val-test NPZ/未跑 CV-Bench）**：新增 `evaluation/pair_gradient_concentration_audit.py` 与聚焦测试 `evaluation/tests/test_pair_gradient_concentration_audit.py`（复用 feasibility 的 split/load/scorer/`pair_loss_terms`/`load_train_samples`/`_cosine`/`draft` 校验与 scale 审计的 `_forward_pair_losses`/`candidate_lambdas`，未复制第二套 scorer/loss；未改既有报告/训练器/app/models）。**隔离**：split 仅由 index metadata 计算（train/val/test=318/76/78 记录、159/38/39 图），只打开 train 318 NPZ（`val_test_npz_opened=[]`、`opened_only_train_npz=true`、opened_npz_count=318）；spy loader 与损坏 val/test NPZ 两法证明 train 指标、每图行、三 seed 聚合与 train NPZ aggregate SHA 不变。**审计设定**：固定 seeds 20261002/20261003/20261004、temp=0.1、init_std=0.02，只 forward + `torch.autograd.grad`；每 seed 先取 train-image mean 的 aggregate base/pair 梯度，再逐 image pair 计算 `L_base_i`/`L_pair_i` 的 Wt/Wv/concat 梯度并只保留标量（不把 159×458752 梯度同时驻留）。**结果**：三 seed concat r_seed=0.010005/0.017507/0.022286 → `r_med=0.017506745782799503`（与冻结一致），`lambda_target_0p25=14.280209646136903`（与冻结一致；正 lambda 只改 pair/base 相对尺度）；signed projection contribution 每 seed/每 param 和≈1，最大绝对误差 **1.728e-06**。concat pooled（477 行）：pair grad norm top1/5/10/20% share=0.087464/0.356141/0.539848/0.734687、ESS=123.742059（positive 466/zero 11）、Gini=0.680267；signed pair contribution sum=3.000005（期望=seed 数 3）、negative count=189/477（fraction **0.396226**，mass=0.946076）、positive ESS=**48.937040**、positive Gini=**0.777647**、positive top10% share=0.660550。逐 seed concat signed sum=1.000001/1.000002/1.000002，negative fraction=0.383648/0.389937/0.415094；pair zero fraction(Wt) mean=0.012579/0.025157/0.031447、nonfinite=0。跨 seed 同图 concat pair norm pairwise Spearman=0.969071/0.975754/0.947484（mean **0.964103**）；pooled concat Spearman：JS vs pair norm **0.927240**、JS vs signed contribution 0.356084、G vs pair norm −0.023203、L_pair vs pair norm 0.915456、hinge margin vs pair norm 0.915450（active-only 0.909327）。top 10 图像仅记录 hash。**统计解释（不含阈值/winner/promotion）**：梯度范数之和不是 aggregate 梯度，norm share 不能当向量贡献；只有 signed projection contribution 和约为 1 且可为负（负质量已单列）；lambda 正缩放不改变 pair-only 集中度/rank/cosine；不能从初始梯度推断收敛或精度。**测试**：聚焦 **24/24 OK**、全量 `unittest discover -s evaluation/tests` **474 OK**（基线 450）。**inputs**：config `44399c8e…ab28b`、index `02a50868…ee26`、draft `787f8f40…85772`（未改，状态仍 `DRAFT_REQUIRES_PI_APPROVAL`，重训项保持 `[ ]`）、train NPZ aggregate `33997394…3951`、script `7981c4ab…0607`、focused test source `ae13942b…ce25`；**outputs**：md `214de023…677e`、json `abb11e58…869e`、csv `edbb9b18…5840`。证据：`reports/token_selector/2026-10-03-pair-gradient-concentration-train-only.{md,json,csv}`、`reports/token_selector/logs/2026-10-03-pair-gradient-concentration-{run,focused-tests,full-tests}.log`、`reports/token_selector/logs/2026-10-03-pair-gradient-concentration-train-only-harness-result.json`、`/tmp/robobrain_harness_result.json`。精确下一步（未执行、交 Codex/PI）：是否将该集中度证据与 pair/base 尺度证据一并纳入重训决策；本审计不授权重训。

更新时间：2026-10-03（**Pair 梯度贡献集中度审计完成**：只读 DRAFT 未改、重训项保持 `[ ]`；r_med/lambda 与冻结一致；signed contribution 和≈1（max err 1.7e-06），concat pair norm Gini=0.680/ESS=123.7，signed positive Gini=0.778/ESS=48.9、negative fraction 39.6%，跨 seed rank mean 0.964，JS vs pair norm 0.927；聚焦 24 + 全量 474 测试 OK；CPU-only/只读、未读 val/test NPZ、未训练/未调参、非 CV-Bench 精度、无 promotion 结论、无重训授权）

- [x] 2026-10-02 **Pair-loss 尺度稳定性审计（train-only、zero-update、CPU-only、只读；未训练/未调参/未保存新 Wt-Wv/未加载 4B/无 GPU/无联网/未读 val-test NPZ/未跑 CV-Bench）**：新增 `evaluation/pair_loss_scale_audit.py` 与聚焦测试 `evaluation/tests/test_pair_loss_scale_audit.py`（复用 feasibility 审计的 split/load/scorer/loss/`_grad_diagnostics`/`_cosine`，未复制第二套算法；未改既有报告/训练器/app/models）。**隔离**：split 仅由 index metadata 计算（train/val/test=318/76/78 记录、159/38/39 图），只打开 train 318 NPZ（`val_test_npz_opened=[]`、`opened_only_train_npz=true`、opened_npz_count=318）；spy loader 与损坏 val/test NPZ 两法证明 train 指标、per-seed/per-lambda 行、r_med、cross-seed 聚合与 train NPZ aggregate SHA 不变；draft 合同只读且校验全通过。**尺度（固定 seeds 20261002/20261003/20261004，student temp=0.1，init_std=0.02，只 forward+`autograd.grad`）**：三 seed 初始 concat pair/base 比 = **0.010005/0.017507/0.022286**，`r_med`=**0.0175067458**；候选尺度**预先固定**为 `lambda_current=1.0`、`lambda_target_0p1=0.1/r_med=5.712084`、`lambda_target_0p25=0.25/r_med=14.280210`、`lambda_target_1p0=1.0/r_med=57.120839`（未新增其它 lambda、未宣布 winner）。**逐 seed/逐 lambda 实际 `autograd.grad`（Wt/Wv/concat 全部 finite 且 nonzero）**：concat scaled_pair/base（median）在四候选中为 0.017507/0.100000/0.250000/1.000001，对应 total/base mean=**1.000440/1.006537/1.034103/1.406463**；`lambda_current` 的 cos(total,base)/cos(total,pair)/cos(base,pair) mean=0.999863/0.033332/0.016752，scaled_pair_loss/base_loss mean=0.018482；跨 seed CV：lambda_current concat scaled_pair/base CV=0.373（seed 异质）。**明确限制**：lambda=1 的 pair 梯度约 base 的 1–2%，不足以说明该监督会产生可见训练影响；0.1/0.25/1.0 只是预先固定的诊断目标，不是文献标准、不是精度门槛；不能从零更新梯度推断收敛、val 改进或任务精度；不修改 DRAFT、不训练、不给 promotion/no-promotion。**测试**：聚焦 **23/23 OK**、全量 `unittest discover -s evaluation/tests` **450 OK**（基线 427）。**inputs**：config `44399c8e…ab28b`、index `02a50868…ee26`、draft `787f8f40…85772`（未改，状态仍 `DRAFT_REQUIRES_PI_APPROVAL`，重训项保持 `[ ]`）、train NPZ aggregate `33997394…3951`、script `b1b80810…53e9`；**outputs**：md `791df44f…f785`、json `0d7bffa9…7ca5`、csv `2940d704…848a`。授权记录：`reports/token_selector/2026-10-02-harness-data-authorization.md`。证据：`reports/token_selector/2026-10-02-pair-loss-scale-train-only.{md,json,csv}`、`reports/token_selector/logs/2026-10-02-pair-loss-scale-{run,focused-tests,full-tests}.log`、`reports/token_selector/logs/2026-10-02-pair-loss-scale-train-only-harness-result.json`。精确下一步（未执行、且本审计不授权重训）：如需复算运行 `/home/zhaoxiangyu/conda-envs/robobrain25/bin/python3 -m evaluation.pair_loss_scale_audit`；是否按 DRAFT 重训仍须 PI 批准。

更新时间：2026-10-02（**Pair-loss 尺度稳定性审计完成**：只读 DRAFT 未改、重训项保持 `[ ]`；三 seed concat pair/base=0.010005/0.017507/0.022286 → r_med=0.0175067458，预先固定四候选 lambda（1.0、5.712084、14.280210、57.120839），逐 seed/逐 lambda autograd.grad 验证 concat realized ratio median=0.017507/0.1/0.25/1.0、total/base mean=1.0004/1.0065/1.0341/1.4065，全部 finite/nonzero；lambda=1 pair 梯度仅 base 的 1–2%，0.1/0.25/1.0 仅为固定诊断目标；聚焦 23 + 全量 450 测试 OK；CPU-only/只读、未读 val/test NPZ、未训练/未调参、非 CV-Bench 精度、无 promotion 结论、无重训授权）

- [x] 2026-10-02 **Pair-loss 数学与初始化梯度可行性审计（train-only、zero-update、CPU-only、只读；未训练/未调参/未保存新 Wt-Wv/未加载 4B/无 GPU/无联网/未读 val-test NPZ/未跑 CV-Bench）**：新增 `evaluation/pair_loss_feasibility_audit.py` 与聚焦测试 `evaluation/tests/test_pair_loss_feasibility_audit.py`；只读 DRAFT `configs/prompt_mean_pairwise_retrain.draft.json`（SHA256 `787f8f40…85772` **未改**，状态仍 `DRAFT_REQUIRES_PI_APPROVAL`，重训项保持 `[ ]`），未改既有训练器/原报告/app/models。**隔离**：split 仅由 index metadata 计算（train/val/test=318/76/78 记录、159/38/39 图），只打开 train 318 NPZ（`val_test_npz_opened=[]`、`opened_only_train_npz=true`）；spy loader + 损坏 val/test NPZ 两法证明 train 数学指标、三 seed 初始化指标与 train aggregate SHA 完全不变；静态测试禁止源码出现参数更新/优化器/权重落盘构造。**数学（159 image pairs，cached teacher_soft，nats）**：`JS` mean 0.0511/median 0.00924，`symmetric_KL` mean 0.3852/median 0.0513，`slack=D_swap−JS` mean 0.3341/median 0.0416/min **0.000549**/p95 1.8393；`slack≥0` 比例 **1.0**、`slack<0` 图像 **0**；`D_swap/JS`（JS>0，n=159）min **4.114**/median 5.203/mean 5.594/max 10.334（min 比值 ≥ 2）。因 `KL(p||(p+q)/2)≤0.5·KL(p||q)` ⇒ `JS≤0.5·symmetric_KL`，故 teacher optimum 下 `L_base=0`、`D_swap=symmetric_KL`、`relu(JS−D_swap)=0`：**自适应 margin 在 teacher optimum 对全部 159 对可满足（仅为数学可行性，非可训练成功）**。**初始化/梯度（seeds 20261002/20261003/20261004；严格 frozen `prompt_mean_vector`，softmax(score/0.1)，只 forward + `autograd.grad`，零更新）**：pooled L_base/L_pair/L_total mean **2.7696/0.0511/2.8207**、p95 2.9006/0.0519/2.9503；hinge active 图像比例 seed 均值 **0.9769**、own-better-than-swap **0.4654**、query cosine mean≈0.965、exact query equality **0**、exact score equality **0**、score finite **1.0**；三 seed 对 Wt/Wv/concat 的 L_base/L_pair/L_total 梯度**均存在、有限、非零**，pair/base 范数比均值 Wt/Wv/concat=**0.0174/0.0159/0.0166**，base-vs-pair 梯度 cosine 均值 **0.0300/0.0040/0.0168**（弱正、近正交；pair 梯度约 base 的 1–2%）。**结论仅可行性**：margin 数学可满足、初始化梯度存在且有限、KL 与 pair 梯度弱同向；不设通过阈值、不据此声称重训成功、不把 gradient nonzero 当准确率。草案公式未发现阻断性歧义（仅记录上述 JS≤0.5·SKL 使 margin 在 teacher optimum 自动满足这一性质），不改草案，交 Codex/PI。**测试**：聚焦 **20/20 OK**、全量 `unittest discover -s evaluation/tests` **427 OK**。**inputs**：config `44399c8e…ab28b`、index `02a50868…eee26`、draft `787f8f40…85772`、script `2913e43f…73dea`、train NPZ aggregate `33997394…3951`；**outputs**：md `3124f73e…720f`、json `93e89249…64fc`、csv `fdbd1dd6…16e0`、init-csv `647481a8…623a`（重跑逐字节一致）。证据：`reports/token_selector/2026-10-02-pair-loss-feasibility-train-only.{md,json,csv}`、`reports/token_selector/2026-10-02-pair-loss-feasibility-init-metrics.csv`、`reports/token_selector/logs/2026-10-02-pair-loss-feasibility-{run,focused-tests,full-tests}.log`、`reports/token_selector/logs/2026-10-02-pair-loss-feasibility-train-only-harness-result.json`、`/tmp/robobrain_harness_result.json`。

更新时间：2026-10-02（**Pair-loss 数学/初始化梯度可行性审计完成（train-only / zero-update / CPU-only）**：只读 DRAFT 未改、重训项保持 `[ ]`；split 仅由 index metadata、只开 train 318 NPZ；159 对 teacher optimum slack≥0=1.0、slack<0=0（min 0.000549）、D_swap/JS∈[4.114,10.334] → margin 数学可满足；三 seed 初始化 L_base/L_pair/L_total≈2.77/0.05/2.82、hinge active≈0.977、梯度存在/有限/非零、pair/base≈1–2%、base-pair cosine≈0.004–0.030（弱同向）；聚焦 20 + 全量 427 测试 OK；非 CV-Bench 精度、无训练/无阈值/无重训成功声明）

- [ ] 2026-10-02 **Prompt 成对监督重训实验合同已形成草案，等待 PI 批准，尚未训练**：基于冻结归约负结果与 train-only teacher pair 审计，Codex 建议只比较两个因果清晰的训练臂：固定 `prompt_mean_vector` scorer 的 `A_KL_ONLY_CONTROL`，以及 `B_KL_PLUS_ADAPTIVE_SWAP_HINGE`；B 的 margin 逐图取 teacher `JS_nats(y_g,y_p)`，使弱 teacher pair 自动获得小 margin，且不引入手调 loss weight。两臂均保留全尺寸 `Wt[128,2560]/Wv[128,1024]`、`1x128` query 与 FPGA `Gx128` dot，三固定 seed，train/val-only，existing internal test 不再读取；B 必须通过 Prompt 敏感、shuffled 判别、collapse、teacher alignment、跨 seed 泛化与硬件合同 A-F 六门，A 仅是对照，失败则 `NO_PROMOTION`。暂不混入低秩/量化/新 teacher，以隔离显式 Prompt 判别监督的因果效果；B 通过后才进入 rank 16/32、INT8 与空间 anchor 固定预算阶段。草案：`configs/prompt_mean_pairwise_retrain.draft.json`；决策说明：`reports/token_selector/2026-10-02-paired-retrain-decision.md`。**本项尚未授权执行，不是预注册完成状态。**

- [x] 2026-10-02 **Train-only teacher 成对信号可学性审计（描述性、CPU-only、只读；未训练/未调参/未加载 4B/无 GPU/无联网/未读 val-test NPZ/未跑 CV-Bench）**：新增 `evaluation/teacher_pair_signal_audit.py` 与聚焦测试 `evaluation/tests/test_teacher_pair_signal_audit.py`（复用冻结 `split_sample_ids`/`load_feature_cache`/`pearson`/`spearman`/`topk_indices`/`jaccard`/`teacher_topk_recall`，未改 v1/v2/frozen 报告与 `app/`、`models/`）。**硬隔离**：split 仅由 index metadata 计算（train/val/test=318/76/78 记录、159/38/39 图），只打开 train 的 318 个 NPZ（`val_test_npz_opened=[]`、`opened_only_train_npz=true`）；测试既用 spy loader 又用「把 val/test NPZ 改写成非法字节」两种方式证明 train 指标、input aggregate SHA 与 index SHA 不变，且 val/test 从不被打开。**159 图结果**：JS（自然对数，范围 [0, ln 2]）mean **0.0511** / median **0.00924** / p95 0.2666；对称 KL mean **0.3852** / median 0.0513；TV mean **0.1272** / median 0.0520；`teacher_raw` vs `teacher_normalized` Pearson=Spearman=**1.0**（min=max=1）；normalized L1/L2/max-abs mean 17.238/1.637/0.457；**exact 与 1e-9 tolerance identical fraction 均 = 0**；JS>0、TV>0、两方向 swapped KL>0 比例均 **1.0**；argmax 不同比例 **0.1195**（Manhattan 中位 0、p95 13.1）；K=48/64/96 平均 disagreement(1−Jaccard) **0.2808/0.2918/0.2909**、overlap 0.8162/0.8126/0.8191；K=96 对唯一一张 G=80 图按 min(K,G)=80 并在 CSV 标记。按共同 G 分 **15 层（80–400）**报 count 与主要指标，异质性明显（G=260 n=37 平均 JS 0.0749 vs G=176 n=3 平均 0.0036）。**结论仅描述**：pair signal 非零但整体偏弱且强异质，且现有 teacher 为 max-over-prompt-token（v2 train prompt/image ratio≈0.2539），不能据此声称固定 `prompt_mean_vector` + paired contrastive/delta 重训必然成功；未设任何通过阈值、未访问 test 定阈值。**测试**：聚焦 **25/25 OK**、全量 `unittest discover -s evaluation/tests` **407 OK**。**inputs**：config `44399c8e…28b`、index `02a50868…ee26`、script `95c70317…c2ec`、train NPZ aggregate `33997394…3951`；**outputs**：md `6b926634…b508`、json `09dbce65…52dd`、csv `06f84551…812e`。证据：`reports/token_selector/2026-10-02-teacher-pair-signal-train-only.{md,json,csv}`、`reports/token_selector/logs/2026-10-02-teacher-pair-signal-{run,focused-tests,full-tests}.log`、`reports/token_selector/logs/2026-10-02-teacher-pair-signal-train-only-harness-result.json`（/tmp 在本 harness 不跨命令持久，durable copy 为准）。决策 `TEACHER_PAIR_SIGNAL_TRAIN_ONLY_DESCRIPTIVE / NONZERO_BUT_WEAK_AND_HETEROGENEOUS / NO_THRESHOLD / NO_TRAINING / NO_VAL_TEST_READ`。精确下一步（未执行，仅由 PI/Codex 决定）：若要验证固定 `prompt_mean_vector` + paired delta 是否可学，需先在独立授权下预注册并重训，仍不得读 test 调参、不得访问 val/test 特征缓存做本轮 train-only 审计之外的用途。

更新时间：2026-10-02（**Train-only teacher 成对信号审计完成**：split 仅由 index metadata、只打开 train NPZ；159 图 JS/KL/TV/top-K/argmax 全量描述，signal 非零但弱且异质；exact 与 1e-9 tolerance identical 均为 0，argmax 不同 11.95%，K=48/64/96 平均 disagreement≈0.29；聚焦 25 + 全量 407 测试 OK；CPU-only/只读、未读 val/test NPZ、未训练/未调参、非 CV-Bench 精度、无加速声明）

- [x] 2026-10-02 **Codex 对冻结权重 scorer-reduction 消融完成独立审校**：检查预注册、split 加载顺序、image-level donor/统计、Holm `m=40`、promotion gate 与硬件计数；在独立临时目录复跑 val，11 个候选 Spearman/NDCG 与正式报告最大绝对差 **0.0**，`winner=NO_PROMOTION`、passing `[]` 与 preregistration hash 完全一致；聚焦 **36/36 OK**、全量 **382/382 OK**。解释修正：多数平滑归约已消除 exact prompt invariance，但冻结 `Wt/Wv` 未恢复可用 prompt 判别；`mean_cos` 与 `prompt_mean_vector` 组内排序数学等价，不能算两份独立算法证据，不过后者仍对应更低成本的 `1x128` query 硬件形式。证据：`reports/token_selector/2026-10-02-frozen-scorer-reduction-codex-audit.md`。下一步不再堆叠冻结权重归约函数；是否预注册并重训固定硬件友好 scorer/改变监督，需 PI 确认。

- [x] 2026-10-02 **CPU-only 冻结权重 scorer-reduction 消融（隔离 hard-max prompt collapse）→ 负结果 `winner=NO_PROMOTION`；未训练/未调参/未加载 4B/无 GPU/无联网/未跑 CV-Bench**：新增 `evaluation/frozen_scorer_reduction_ablation.py` 与聚焦测试 `evaluation/tests/test_frozen_scorer_reduction_ablation.py`（仅做复用式新增，未改 v1/v2 诊断与蒸馏/训练模块，未改 `models/`、`app/`、`inference.py`、`api.py`）。**预注册**（读取任何 test 数值前写盘）`reports/token_selector/2026-10-02-frozen-scorer-reduction-preregistration.json`，sha256 `b2686bc394833c40…`，seed **20261002**，候选顺序固定为 11 个：`hard_max`（baseline）、`mean_cos`、`topk_mean_k2`、`topk_mean_k4`、`norm_lse_tau003/010/030`、`softmax_mean_tau003/010/030`、`prompt_mean_vector`（每个 prompt token 的 Wt 投影按现有 cosine 约定归一化→均值→再 L2 归一化，逐 group 只与 1 条 128-d query 做 cosine/dot）；未新增 centered/IDF/文本规则/温度。**统计沿用 v2**：donor 严格 split-local、逐图确定性 derangement（双射无自配）、student donor same-task、同图 grounding/pointing 先聚合到 image 再 paired 统计、2000 次 image-clustered bootstrap + 10000 次双侧 sign-flip permutation、精确不变量用 `np.array_equal` 并同时报 tolerance 1e-9、仅 val 选型、winner/NO_PROMOTION 写盘后才首次加载 test 且只评一次；非 baseline 候选 val primary gates 做 Holm-Bonferroni（family 明确为 10 非 baseline 候选 × 2 条件 × 2 primary metric = **m=40**）。**val gate 结果（全部只看 val，n=76/38 图）**：passing = **[]**，A（same-image prompt 敏感）与 B（cross-image same-task shuffled 判别）对**全部 11 个候选均为 False**；C（exact invariance ≤0.05）只有 `hard_max`（1.0000）与 `topk_mean_k2`（0.9474）失败，其余 9 个通过；D（teacher alignment 非退化 ≥ baseline−0.01）在 `mean_cos`(rho 0.0129/N 0.8343)、`norm_lse_tau030`(0.0292/0.8397)、`softmax_mean_tau030`(0.0427/0.8458)、`prompt_mean_vector`(0.0129/0.8343) 失败；E 全通过。**关键 val 数**：baseline `hard_max` correct Spearman **0.0565**、NDCG-full **0.8499**、exact invariance **1.0000**；`prompt_mean_vector` correct **0.0129/0.8343**、exact invariance **0.0000**、1×128 query 使 token 竞争彻底消失，但 same-image ΔSpearman 仅 **+0.0004（CI[−0.0017,+0.0027], p≈1.0, Holm p=1.0）**、cross-image Δ≈**+0.0001**，即“collapse 消失但 teacher 对齐与 prompt 判别同时变差/仍不可辨”；best correct Spearman = `softmax_mean_tau010` 0.0593，best NDCG-full = `topk_mean_k4` 0.8513，最大 |same-image ΔSpearman| ≤0.0015、所有 CI 跨 0、Holm p≈1；dominant-token concentration：`hard_max` 0.9999、`norm_lse_tau003` 0.9975、`softmax_mean_tau003` 0.9975、`norm_lse/softmax_mean_tau030` 0.1567，`mean_cos`/`prompt_mean_vector` = N/A。**test 仅一次冻结诊断**：`hard_max` Spearman 0.0536 / NDCG-full 0.8493，与 val 同向，未用于反选。**硬件代价（解析符号，未综合/未实测 ZCU102）**：`prompt_mean_vector` = **1×128 INT8 query（128 B）+ FPGA G×128 dot（128 MAC/group，无 compare/exp/归约）**；矩阵型 reducer 需传 `[T,128]` query（T=41 → 5248 B）并在 FPGA 做 T·D=5248 MAC/group 加比较/exp/归约（total/group 5288–5412）。**隔离测试**：改写/删除 test NPZ 后 val 指标、winner、预注册 hash 完全不变，且 val 阶段从不打开任何 test npz（loader 记录断言）。**测试**：聚焦 **36/36 OK**、全量 `unittest discover -s evaluation/tests` **382 OK**。**输出**：`reports/token_selector/2026-10-02-frozen-scorer-reduction.{json,md}`、`-candidates.csv`、`-splits.csv`、`-preregistration.json`、`-val-gate.json`、`logs/2026-10-02-frozen-scorer-reduction-{run,console,focused-tests,full-tests,harness-result}.log/json`、`/tmp/robobrain_harness_result.json`。**决策 `FROZEN_SCORER_REDUCTION_ABLATION_DONE / NO_PROMOTION / COLLAPSE_REMOVED_BUT_PROMPT_SIGNAL_ABSENT`；下一决策（需 PI 授权）：冻结 Wt/Wv 下仅改归约无法恢复 prompt 可利用性，需重训/改监督并预注册固定 reducer，不得据 test 事后挑“最不差”方案。**

更新时间：2026-10-02（**冻结权重 scorer-reduction 消融完成**：预注册 11 候选、Holm family m=40；winner=**NO_PROMOTION**（无人通过 A–E），`prompt_mean_vector` 移除 collapse 但 prompt 判别仍不可辨且 teacher 对齐下降；test 仅一次诊断；聚焦 36 + 全量 382 测试 OK；CPU-only/只读、非 CV-Bench 精度、无综合/无 ZCU102 实测/无加速声明）

- [x] 2026-10-02 **Prompt 可辨识性诊断 v2 纠错（CPU-only、只读、无训练/无调参/无 4B 加载/无联网/CV-Bench 未进入）→ 修正 v1 统计设计；`primary_blocker=STUDENT_PROMPT_COLLAPSE`，`contributing_factor=TEACHER_PROMPT_SIGNAL_BELOW_ENGINEERING_GATE`，`training_factor=GENERALIZATION_OVERFIT`**：最小修改 `evaluation/prompt_identifiability_diagnostic.py` 与聚焦测试 `evaluation/tests/test_prompt_identifiability_diagnostic.py`，落实 12 项纠错：(1) 所有 donor **严格 split-local**，val 不再读取 test/train 的 prompt/target；(2) 跨图 donor 改为**按 image 的固定 seed 确定性 derangement**（双射、无自配）：student 同 task、teacher 同 G 且同 task（同 G stratum 仅 1 图则排除并报告）、prompt control 跨图跨 task；(3) student paired delta 先对同图 grounding/pointing 两条 delta 取均值再在 **image 级**做 2000 bootstrap + 10000 双侧 sign-flip permutation（CSV 仍逐样本，JSON 同时报 sample_count 与 independent_image_units）；(4) prompt paired delta 逐 image `same_i - cross_i`，禁止减全局 cross 均值；(5) "bit-identical" 改用 `np.array_equal` 并报每对 max_abs_diff 的 max/median/p95，容差版本显式命名；(6) teacher 0.5 明确为**预注册工程门槛/启发式判据，非文献通用阈值**，不单独宣称 teacher 失效；(7) 根因改为 `primary_blocker`/`contributing_factor`/`training_factor`，保留四个布尔标签；(8) 参数/样本同时报 params/record、params/image、params/group-target 并说明 group target 非独立样本；(9)(10) 阈值与 derangement seed 为模块常量、读取 test 前固定，gate 仅用 val，test 固定后报一次；v1 报告加 `SUPERSEDED_FOR_INFERENCE_BY_V2`（v1 原始数值未改）。**v1→v2 关键数**：val teacher Spearman prompt/image ratio 0.2740 CI[0.220,0.335]（n=35 图）→ **0.2713 CI[0.211,0.339]（n=30/38 图；8 图因同 G stratum 仅 1 图排除，train 4、test 3）**；student val same delta Spearman 0.0（n=76）→ **0.0（image 级 n=38）**，cross delta −0.00707（n=76）→ **−0.01124 CI[−0.03364,+0.00001] p=0.5012（image 级 n=38，n_rec=76）** 均 not present；prompt mean-pooled delta −0.0174 → **+0.0039 CI[+0.0036,+0.0044] p=1e-4（逐 image，n=38）**；`exact_prompt_invariance_fraction` val **1.0000 经 `np.array_equal`**（tolerance 1e-9 版本同为 1.0，max/median/p95 max_abs_diff 均 0），single-argmax val **0.9737**；Wt/Wv effective rank **123.918/119.950**；split 318/76/78。**参数**：458752（train record 318、唯一 image 159、teacher group-target 86640）→ params/record **1442.62**、params/image **2885.23**、params/group-target **5.29**；best_epoch=0、train KL 0.025643→0.014819、val KL final 0.027907。**测试**：聚焦 **41/41 OK**（v1 基线 26，新增 15 项覆盖 split-local donor、derangement 双射无自配、同 task donor、teacher 同 G donor、image-clustered n、prompt control 非全局均值、array_equal 精确不变量、test prompt 修改不影响 val）、全量 `unittest discover -s evaluation/tests` **346 OK**（v1 基线 331）。**inputs**：config `44399c8e…28b`、index `02a50868…ee26`、projection `132a6d9f…d0c`、training_report `f1a46979…079`；**outputs**：v2 md `b5e49fd2…c9`、v2 json `f4e0b5b5…a90`、samples-v2 `8bcefa50…f37`。证据：`reports/token_selector/2026-10-02-prompt-identifiability-diagnostic-v2.{md,json}`、`...-samples-v2{,-train,-val,-test}.csv`、`reports/token_selector/logs/2026-10-02-prompt-identifiability-v2-{run,tests}.log`、`/tmp/robobrain_harness_result.json`。决策 `PROMPT_IDENTIFIABILITY_V2_CORRECTED / PRIMARY_STUDENT_PROMPT_COLLAPSE / CONTRIBUTING_TEACHER_BELOW_ENGINEERING_GATE / TRAINING_GENERALIZATION_OVERFIT / V1_SUPERSEDED_FOR_INFERENCE / NO_TRAINING_NO_MODEL_LOAD_NO_NETWORK`。精确下一步（未执行，仅由 PI/Codex 决定）：任何后续 scorer ablation 必须以 v2 gate 为准，仍不得读 test 调参、不得事后修改阈值/seed，且应先经 PI 授权。

更新时间：2026-10-02（**Prompt 可辨识性 v2 纠错完成**：split-local donor + image derangement + image-level 统计 + `np.array_equal` 精确不变量 + 工程门槛措辞 + 因子拆分；`primary_blocker=STUDENT_PROMPT_COLLAPSE`、`contributing_factor=TEACHER_PROMPT_SIGNAL_BELOW_ENGINEERING_GATE`、`training_factor=GENERALIZATION_OVERFIT`；v1 已加 `SUPERSEDED_FOR_INFERENCE_BY_V2` 且数值未改；聚焦 41 + 全量 346 测试 OK；CPU-only/只读、非 CV-Bench 精度、无加速声明）

- [x] 2026-10-02 **Prompt 可辨识性/退化诊断（CPU-only、只读、无训练/无调参/无 4B 加载/无联网/CV-Bench 未进入）→ 数据有效；三标签命中，根因 `TEACHER_PROMPT_SIGNAL_WEAK` + `STUDENT_PROMPT_COLLAPSE`**（**SUPERSEDED_FOR_INFERENCE_BY_V2：统计设计有缺陷，v1 结论不得作为最终推断**）：新增 `evaluation/prompt_identifiability_diagnostic.py`（新增算法仅 JS 散度、paired bootstrap/符号翻转 permutation、分层方差分解；score/metric/load 全部复用 `prompt_projection_distillation.py` 与 `train_prompt_projection.py`，无第二套算法）与聚焦测试 `evaluation/tests/test_prompt_identifiability_diagnostic.py`。**完整性 PASS**：472 样本 / 236 图、每图 grounding+pointing 各 1、同图 `group_mean` 逐元素相等（0 处不一致，hash/数值一致）、`teacher_soft_target` 均和 1、Wt[128,2560]/Wv[128,1024] 有限；split 318/76/78。**预注册阈值（读取 test 前固定）写入 JSON `thresholds`**：paired 95% bootstrap（**2000** 次）CI 下界 > 0 且 |Cohen's dz| ≥ **0.2** 且双侧符号翻转 permutation（**10000** 次）p < **0.05** 才算 signal present；`TEACHER_PROMPT_SIGNAL_WEAK` = 教师 prompt/image Spearman 距离比 bootstrap CI 上界 < **0.5**；`GENERALIZATION_OVERFIT` = best_epoch==0 且 train KL 下降且 val 无改善；`DATA_OR_SCHEMA_INVALID` = 任一完整性检查失败。**关键数值（val；train/test 同向）**：teacher 同图两 prompt Spearman **0.7304** vs 跨图同 G 对照 **0.0160**，ratio **0.274**，CI **[0.220, 0.335] < 0.5** → WEAK（train 0.2539 CI [0.231,0.279]、test 0.2288 CI [0.194,0.271]）；`teacher_normalized` V_prompt/V_image = **0.0726**；student correct−同图另一任务 Spearman/NDCG/pearson 差值**恰为 0**（CI [0,0]），correct−固定 seed 跨图 shuffled Spearman **−0.0071**（CI [−0.0262,+0.0073]，p=0.49），两个 primary 条件全部 not present → **COLLAPSE**；坍塌机制：**98.73%** 样本所有 group 由**单一** prompt token 取 max，同图两 prompt 的 dominant 投影方向 cosine **=1.0**，**472/472** 样本 correct 与同图另一任务分数逐元素完全相等（Wt/Wv 数值满秩，effective rank ≈ **123.9/120.0**，非权重秩坍塌）；`GENERALIZATION_OVERFIT`：best_epoch=**0**，train KL 0.025643→0.014819，val KL 0.025189→0.027907（val 在 init 后从未改善）；params **458752**/train **318** = **1442.6**；逐样本 Spearman train−val gap +0.0267（Cohen's d=0.179）。**测试**：聚焦 **26/26 OK**、全量 `unittest discover -s evaluation/tests` **331 OK**（基线 305）。**inputs**：index `02a50868…ee26`、config `44399c8e…28b`、projection `132a6d9f…9d0c`、training_report `f1a46979…6079`；**outputs**：JSON `f25f2684…`、MD `4dbce21c…`、samples CSV `64c48888…`、脚本 `1735323f…`、测试脚本 `bce2f5d4…`、tests log `1eacc259…`。决策 `PROMPT_IDENTIFIABILITY_DIAGNOSED / DATA_VALID / TEACHER_PROMPT_SIGNAL_WEAK_TRUE / STUDENT_PROMPT_COLLAPSE_TRUE / GENERALIZATION_OVERFIT_TRUE / TEST_DIAGNOSTIC_ONLY / NO_TRAINING_NO_TUNING_NO_MODEL_LOAD`。证据：`reports/token_selector/2026-10-02-prompt-identifiability-diagnostic.{md,json}`、`2026-10-02-prompt-identifiability-samples{,-train,-val,-test}.csv`、`reports/token_selector/logs/2026-10-02-prompt-identifiability-{run,tests}.log`、`evaluation/prompt_identifiability_diagnostic.py`、`evaluation/tests/test_prompt_identifiability_diagnostic.py`、`/tmp/robobrain_harness_result.json`。精确下一步（未执行，最多 2 选项）：(a) PI 接受该负结果并将 epoch-0/val 失败冻结；(b) 仅在明确独立授权下先修复 student `max`-over-token 归约退化/更换监督信号后重训（不改数据/split/特征/评测契约、不得读 test 调参）；本轮不进入 CV-Bench、不调参、不重载 4B、不 kill 他人进程。

更新时间：2026-10-02（**Prompt 可辨识性诊断完成**：数据有效；teacher prompt/image 排序变化比 CI 上界 <0.5 → `TEACHER_PROMPT_SIGNAL_WEAK`；student correct 与同图另一任务分数逐元素完全相等、与跨图 shuffled 不可分 → `STUDENT_PROMPT_COLLAPSE`；best_epoch=0 且 val 从不改善 → `GENERALIZATION_OVERFIT`；诊断仅 CPU/只读，聚焦 26 + 全量 331 测试 OK，非 CV-Bench 任务精度、无加速声明）

- [x] 2026-09-28 **干净归纳 Prompt 投影特征抽取 + 训练内部验证（「实际可运行优先」共享 GPU）→ 执行完成；val shuffled 判据未通过**：按用户最新明确授权「只要 GPU 能实际运行就使用，允许与其他用户 CUDA 进程共享，禁止 kill/暂停/修改他人进程」，预检门槛降为**空闲显存 ≥ 14336 MiB 且 5 s 平均 util < 80%**，模型加载后仅要求 ≥ 4096 MiB。extractor 做**最小、默认不变**修改：新增 opt-in `--allow-shared-gpu`/`--min-free-mib`/`--max-util-pct`，不传 flag 时旧严格属主门禁完全不变；新增加载后/抽取后各一次单次快照审计。15:15 预检 PASS：空闲 **22317 MiB**、5×1 s util 全 **0%（avg 0%）**、非属主 `seey6` PID 16507/23038 MiB **按新策略容忍、未做任何操作**；加载后空闲 **10227 MiB ≥ 4096 PASS**。**实际 GPU 执行**：模型 `/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B`（offline），抽取 **472 记录 / 236 次视觉前向**（同图一次前向、2 prompt 复用，472 次 prompt embedding），index SHA256 `02a50868…ee26`、472 npz（364,136,400 B）、schema 无 dense hidden；`torch.cuda.synchronize` 前后计时 **91.58 s**、峰值 allocated **8552.6 MiB** / reserved **9248.0 MiB**。数据完整性 PASS：manifest SHA256 `acab1a32…9213`==pin、`--validate-manifest-only` PASS、472/236、split train 159/318、val 38/76、test 39/78、跨 split 同图重叠 0、CV-Bench 训练泄漏 0。训练既定 128 维 `Wt[128,2560]`/`Wv[128,1024]`（超参未改）：`best_epoch=0`、`best_val_kl=0.025189`、train KL 0.025643→0.014819；`CUBLAS_WORKSPACE_CONFIG=:4096:8` 复跑 `weights_sha256=ece4a916…cdb3` 完全一致；导出 `artifacts/token_selector/prompt_projection_WtWv.npz`（`load_projection` 通过）。控制与指标（**内部 sanity，非 CV-Bench 任务精度**）：val `student_beats_l2=true`，但 `correct_beats_shuffled=false`（val Pearson 0.0731/Spearman 0.0565/NDCG 0.8499 vs shuffled 0.0781/0.0630/0.8510，主指标 paired delta 全负）→ **正式 promotion 未通过**；test 冻结后仅评一次（Pearson 0.0673/Spearman 0.0536/NDCG 0.8493；K=48/64/96 recall 0.2257/0.2929/0.4180、Jaccard 0.1305/0.1755/0.2716、saliency mass 0.2814/0.3537/0.4957）。测试：聚焦 **68 OK**（含新增 5 个 `SharedGpuGateTests`）、全量 `unittest discover -s evaluation/tests` **305 OK**。源码 pin（旧→新）：extractor `cac52d49…eb70`→`ccfd0ea8…cf69`、config `0b03f8f5…ef99`→`44399c8e…28b`（trainer/clean builder 未改）。一次语义不变修复：生产适配层 `.detach().cpu()` 解决 CUDA tensor→`np.asarray` 报错，数值/dtype/特征/split/算法/超参未改。旧 24 GiB/独占门禁报告已加注被取代。决策 `PROMPT_PROJECTION_EXTRACTION_DONE / TRAINING_DONE_VAL_SELECTED / VAL_CORRECT_BEATS_SHUFFLED_FALSE / TEST_EVALUATED_ONCE / INTERNAL_SANITY_ONLY_NOT_CV_BENCH_ACCURACY`。证据：`reports/token_selector/2026-09-28-prompt-projection-shared-gpu-run.{md,json}`、`2026-09-28-prompt-projection-shared-gpu-provenance.json`、`2026-09-28-prompt-projection-training.{json}`、`2026-09-28-prompt-projection-training-metrics.csv`、`artifacts/token_selector/prompt_projection_features/`、`artifacts/token_selector/prompt_projection_WtWv.npz`、`/tmp/robobrain_harness_result.json`。精确下一步（未执行）：由 PI 决定 (a) 冻结该 epoch-0/val 失败为负结果，或 (b) 仅在明确授权下调整优化协议（不改数据/split/特征/评测契约）后重训并再次如实报告 val gate；**不得 kill 他人进程**。

- [!] 2026-09-28 **干净归纳 Prompt 投影特征抽取/训练内部验证（新共享 GPU 低占用共存策略门禁）→ BLOCKED（仅因空闲显存不足）**：按用户最新授权，资源门禁改为「空闲显存 ≥ 24 GiB 且 5 s 平均利用率 < 50% 即可使用，允许与其他用户低占用 CUDA 上下文共存，不因存在他人进程阻塞；禁止 kill/暂停/修改他人进程」；旧报告 `2026-09-28-prompt-distillation-gpu-gate.{md,json}` 原样保留并加注「G1 被新策略取代」。15:52 只读审计 A6000（驱动 580.178.04 / CUDA 13.0）：总 46068 MiB、已用 23169 MiB、**空闲 22317 MiB（21.79 GiB）< 24576 MiB（24 GiB），缺口 2259 MiB → 门禁 FAIL**；5×1 s 采样利用率全 **0%（avg 0% < 50% PASS）**，温度 32–33 °C、功耗 11.0–16.74 W；唯一 compute 进程 PID 16507 `seey6`/uid 1001/`python main.py --port 7860 --listen 0.0.0.0`/**23038 MiB**/已运行 2 天 20:39，**按新策略该上下文本身不阻塞**，仅因其占用导致显存不足；未对该进程做任何 kill/暂停/修改/信号操作。故 `model_load_authorized=false`：未激活模型、未 import transformers/PIL、未创建输出目录、未抽取/训练/评估，运行期两次审计（模型加载后、抽取结束后）NOT_RUN。CPU 只读校验 PASS：唯一训练源 `datasets/prompt_projection_clean_inductive_v1.jsonl` SHA256 **`acab1a32…9213` == config pin**；`--validate-manifest-only` `status=PASS, gpu_used=false, model_loaded=false`；472 记录 / 236 唯一图（grounding 236 + pointing 236）；按 `source_image_sha256` 严格同图不跨 split = train 159 图/318、val 38 图/76、test 39 图/78；CV-Bench 零 COCO id / 零正式 sample_id / 零正式 eval image SHA 重叠；图像与构造脚本哈希冻结一致。测试（CPU）：`test_prompt_projection_clean_inductive+distillation+feature_reuse+metrics` **63/63 OK**，全量 `unittest discover -s evaluation/tests` **300 OK**。决策 `GPU_GATE_BLOCKED_NEW_POLICY / MODEL_LOAD_NOT_ACTIVATED / PROMPT_PROJECTION_EXTRACTION_NOT_RUN / TRAINING_NOT_RUN`。证据：`reports/token_selector/2026-09-28-prompt-projection-newpolicy-gate.{md,json}`、`reports/token_selector/logs/gpu_audit_20260928T155208_newpolicy.log`、`reports/token_selector/logs/2026-09-28-prompt-projection-{validate-manifest-only,clean-inductive-validate}-newpolicy.json`、`reports/token_selector/logs/2026-09-28-prompt-projection-full-regression.log`、`/tmp/robobrain_harness_result.json`。精确下一步（未执行）：待**空闲显存 ≥ 24576 MiB 且平均利用率 < 50%** 后运行 pinned 抽取器（预期 `vision_forwards=236`）→ 训练 128 维 `Wt/Wv`（val 选 checkpoint、test 仅一次）→ L2/shuffled 控制 → 聚焦与全量回归；运行前需按新策略对 extractor 内置 `owner_rule` gate 做**最小非语义调整**（本轮未改，不改算法/超参数），**不得 kill 他人进程**。

更新时间：2026-09-28（**干净归纳 Prompt 投影抽取 + 训练内部验证已实际在共享 GPU 上跑通**：extractor 新增 opt-in 共享门禁（默认行为不变），预检 22317 MiB / util 0% / 加载后 10227 MiB 全 PASS；抽取 472 记录 / 236 视觉前向 / 91.58 s / 峰值 reserved 9248 MiB；训练 128 维 Wt/Wv，`best_epoch=0`，val `student_beats_l2=true` 但 `correct_beats_shuffled=false` → promotion 未通过；test 仅评一次；聚焦 68 + 全量 305 测试 OK；旧门禁报告已加注被取代；**非 CV-Bench 任务精度、无加速声明**）

- [x] 2026-09-28 **Selector→LayerNorm-Q8_0 adapter 固定预算合同修正 + 参数片上 cache + 受控 csynth（bias PASS / adapter timing FAIL）**：审校修正 `fpga_accel/selector_qkv_adapter_hls`。**(1) 合法 M 收紧**：Selector 合法 K 仅 48/64/96，adapter 合法 rows 严格为 **M=192/256/384**，`SQA_MAX_ROWS=384` 并新增显式 `sqa_is_legal_adapter_rows()`；`0/-1/1/191/193/385/4320` 等一律 `SQA_ERR_DIMENSION`；bias-add 保留通用 `SQA_BIAS_MAX_ROWS=4320`，本轮正式 qkv case 为 `M=192/256/384, N=3072`。**(2) 参数存储**：`norm_gamma/norm_beta` 不再每行从 m_axi 重复读取，改为调用开始各读一次到 256×128-bit（各 4 KiB、共 8 KiB）片上 cache，`#pragma HLS BIND_STORAGE ... type=ram_2p impl=bram`；数学、Q8_0 payload/scale、端口布局不变。**(3) 回归**：重新生成 fixture（adapter M=192 sha256 `02e16b51…b68` 不变；bias M=384,N=3072 sha256 `7461ad9b…00a9c`），g++/ASan+UBSan/HLS C-sim 均 `passed=18 total=18`，adapter/bias 逐字节 golden mismatch **0**；聚焦 Python **24/24 OK**、全量 `unittest discover -s evaluation/tests` **300 OK**。**(4) 受控 csynth**（只 `csynth_design`；part `xczu9eg-ffvb1156-2-e`、`create_clock -period 5.0`、默认 uncertainty 1.35 ns 不放宽；工程仅 `/tmp`）：**bias PASS**（estimated 3.650 ns = effective budget 3.650 ns、slack 0、worst 21,012,512 cycles、资源 BRAM 1.75%/DSP 0.36%/FF 0.73%/LUT 1.41%，正式 M=384,N=3072 由 achieved II=2 推导 589,847 cycles≈2.949 ms）；**adapter 0 error、有限 latency（worst 2,601,752 cycles≈13.009 ms<33.3 ms）、资源全 <50%（BRAM 2.96%/DSP 1.90%/FF 2.08%/LUT 6.42%）但 estimated period 3.945 ns > 3.650 ns、slack -0.295 ns，如实判 timing FAIL**；根因：Q8_0 `quantize_block` 的 32 路 amax 归约 loop-carried `amax→fcmp→select→amax`（fcmp≈2.782 ns+select 0.449+store 0.427=3.945 ns），同一路径也存在于现有 `pl_resident_ffn` csynth；允许的最多 2 轮调度/存储 pragma 均无法关闭（II=2 不生效；`BIND_OP op=fcmp` 在 xczu9eg 上不支持，pre-synthesis error），故**未继续 cosim/route/IP/BD/驱动/上板**。合法 M 周期：M=192 → 1,294,848 cycles/6.474 ms、M=384 → 2,601,216 cycles/13.006 ms（工具直接报告），M=256 仅有界推导 [1,726,464, 1,734,144] cycles（非单独报告值）。未改任何现有 selector/q8q8/pl_resident_ffn HLS 源码与模型文件。决策 `SELECTOR_QKV_ADAPTER_HLS_IMPLEMENTED / ADAPTER_LEGAL_M_192_256_384 / PARAM_CACHE_8KIB_BRAM / GPP_ASAN_HLS_CSIM_PASS / BIAS_CSYNTH_GATE_PASS / ADAPTER_CSYNTH_TIMING_FAIL / NO_COSIM_NO_ROUTE / ATTENTION_STILL_MISSING / NO_END_TO_END_SPEEDUP_CLAIM`。证据：`reports/token_selector/2026-09-28-selector-qkv-adapter-csynth.{md,json}`、`reports/token_selector/2026-09-28-selector-qkv-adapter-csim.{md,json}`、`artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/{adapter,adapter_round0,adapter_round1,adapter_round2_bindop_unsupported,bias}/`、`artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/`、`scripts/validate_selector_qkv_adapter_csynth.py`、`scripts/run_csynth.*`、`/tmp/robobrain_harness_result.json`。精确下一步（未执行）：仅在独立授权下用**不等价**手段（如把 32 路 amax 归约改平衡树/分段最大值并对 fcmp 打拍）打破 3.945 ns 路径后重跑同一条 csynth；在此之前不 cosim/导出 IP/route/建 BD/写驱动/上板。

更新时间：2026-09-28（**Selector→LayerNorm-Q8_0 adapter 合同修正 + 参数 8 KiB BRAM cache + 受控 csynth 完成**：合法 M 收紧为 192/256/384；fixture/g++/ASan/HLS C-sim 逐字节 0 mismatch，聚焦 24 + 全量 300 测试 OK；bias csynth PASS（3.650 ns），adapter csynth 0 error/有限 latency/资源 <50% 但 timing FAIL（3.945 ns > 3.650 ns，固有 fcmp 归约路径），未 cosim/route；无 GPU/板卡/联网，attention 仍缺失，不声称加速/精度）

- [x] 2026-09-28 **Selector→LayerNorm-Q8_0 适配器 + 独立 QKV bias-add 参考 HLS 核（g++/ASan+UBSan/HLS C-sim）→ PASS（仅 C-sim，无综合/cosim/IP/板卡/GPU/联网）**：新建独立目录 `fpga_accel/selector_qkv_adapter_hls`，实现修正版拓扑中**唯一 model-correct 直接边界**的前两级 PL 算子。**(1) `selector_ln_q8_adapter`**：输入 Selector 段0 `int8[M,1024]`（128-bit 小端 packed，lane `L=bits[8L+7:8L]`，M∈{192,256,384} 对应 K=48/64/96）+ 全局 `s>0`（`x≈q/s`，clamp `[-128,127]`，zero-point 0）+ 真实 `norm1` gamma/beta（4×fp32/128-bit）；输出直接匹配现有 q8q8/pl_resident_ffn 的 Q8_0 activation 合同：每行 32 block、每 block 32 int8、256-bit payload、per-block binary16 scale。**(2) `qkv_bias_add`**：独立 elementwise `y_in+bias[col]`，输入现有 Q8Q8 的 4×fp32/128-bit `[M,3072]` 与 fp32 `bias[3072]`，输出同布局；**仅独立算子，不假装 attention 完整**。**数学等价**：`LN(q/s)=((q-mean_q)/sqrt(var_q+eps*s*s))*gamma+beta`（推导：`mean_x=mean_q/s`、`var_x=var_q/s²`），在 int8 域直接计算不物化 `x`；variance 为 **population variance**（`E[q²]-E[q]²`，负值截 0），与 `ffn_vector.cpp::row_statistics` 完全一致；`eps*s*s` 左结合；常量行取 `inverse_norm=0` → 恰为 `beta`；`q=-128` 参与统计、仅输出 Q8_0 饱和到 `-127`；用 **direct dequant→FP32 LayerNorm** 独立参考对拍。**真实张量 fixture（只读、不加载完整模型）**：`tools/generate_fixtures.py` 按 header+byte-range 读 `model-00002-of-00002.safetensors`（3,908,490,048 B，header sha256 `87e63413…b94e`，读后未改动）的 `blocks.1.norm1.weight/bias`（BF16 [1024]，raw `41b623b3…`/`855c11e4…`）与 `blocks.1.attn.qkv.bias`（BF16 [3072]，raw `83c5f50d…`），生成 `real_block1_adapter_fixture.bin`（413,740 B，sha256 `02e16b51…b68`）与 `real_block1_bias_fixture.bin`（405,532 B，sha256 `a9ff53c0…b99`）。**逐字节对拍**：adapter payload/scale 与 bias 输出对 Python 冻结 golden 的 mismatch 均为 **0**（g++、ASan+UBSan、HLS C-sim 三处一致；`SQA_TEST_SUMMARY passed=16 total=16 result=PASS`）。**numerics**：adapter 真实 block1（M=192,K=48,s=33.7386）fused vs direct `max_abs≈9.5e-7`、rel_L2≈8.0e-8、cosine≈1.0，Q8_0 dequant vs direct `max_abs≈0.0277`、RMSE≈0.00509、rel_L2≈0.00722、cosine≈0.9999739；必需案例 M=192/256/384（K=48/64/96，多 seed）fused vs direct rel_L2 **7.7e-8–9.4e-8**、dequant vs direct rel_L2 **≈0.0052–0.0054**、cosine **≈0.999986**。**测试**：新增 `evaluation/tests/test_selector_qkv_adapter_hls.py` **17/17 OK**（fusion 恒等、population variance、常量行=beta、-128、half 逐字节复刻、Q8_0 clamp、fixture 复算 golden、真实 gamma/beta/bias 对应），全量 `unittest discover -s evaluation/tests` **293 OK**（基线 276）；原始日志：`artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/{gpp_test.log,asan_ubsan_run.log,hls_csim_adapter.log,hls_csim_bias.log}`。**workaround（仅 host testbench，复用仓库既有）**：`LD_LIBRARY_PATH=/home/xilinx/xic/lib/lnx64.o/SuSE`（libtinfo.so.5）+ `COMPILER_PATH`→系统 `ld`、`LIBRARY_PATH=/usr/lib/x86_64-linux-gnu`、自带 GCC 8.3.0、`config_csim -ldflags "-no-pie"` 规避 binutils 2.37 无法读 `.relr.dyn`；ASan/UBSan 用 `-fno-sanitize=shift` 规避 HLS `ap_int` 头内部 left-shift-of-negative 误报（内核位移全在无符号值上，未丢失内核侧检查）。边界：**仅 HLS C-sim**，不是综合/时序/资源/板级；激活为合成，仅 norm1 gamma/beta 与 qkv.bias 为真实；链止于 `+qkv.bias`；attention（reshape/split、RoPE、softmax、PV、attn.proj、残差、norm2、fc1/GELU/fc2）与 blocks 2..23 仍缺失，**不声称端到端加速、不判任务精度**；未改任何现有 selector/q8q8/pl_resident_ffn HLS 源码与模型文件。决策 `SELECTOR_QKV_ADAPTER_HLS_IMPLEMENTED / FUSED_LAYERNORM_Q8_0_BYTE_EXACT_VS_PYTHON_GOLDEN / INDEPENDENT_QKV_BIAS_ADD_KERNEL / GPP_ASAN_HLS_CSIM_PASS / CSIM_NOT_CSYNTH / ATTENTION_STILL_MISSING / NO_END_TO_END_SPEEDUP_CLAIM`。证据：`fpga_accel/selector_qkv_adapter_hls/`、`scripts/validate_selector_qkv_adapter_csim.py`、`evaluation/tests/test_selector_qkv_adapter_hls.py`、`reports/token_selector/2026-09-28-selector-qkv-adapter-csim.{md,json}`、`artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim/`、`/tmp/robobrain_harness_result.json`。精确下一步（未执行）：在 `+qkv.bias` 之后实现 attention 各算子，仅独立授权后再考虑 csynth/BD/驱动/上板；同时采集真实 block0 hidden 激活做 fixture 级数值验证。

更新时间：2026-09-28（**Selector→LayerNorm-Q8_0 适配器 + 独立 QKV bias-add HLS 核完成**：g++/ASan+UBSan/HLS C-sim 全部 PASS；adapter Q8_0 payload+binary16 scale 与 bias 输出对 Python 冻结 golden 逐字节一致；fused LN 与 direct dequant→FP32 LN 对拍 rel_L2≈1e-7；真实 block1 norm1 gamma/beta 与 qkv.bias 只读 fixture；17/17 聚焦 + 293 全量测试 OK；仅 C-sim，无综合/cosim/IP/板卡/GPU/联网，attention 仍缺失，无端到端加速/精度声明）

- [!] 2026-09-28 **干净归纳 Prompt 投影蒸馏 GPU 门禁（只读）→ BLOCKED，未加载模型、未抽取、未训练**：按 `nvidia-smi` + `ps` + `/proc` 审计 A6000（驱动 580.178.04 / CUDA 13.0，总 46068 MiB，已用 471 MiB，空闲 **45015 MiB ≥ 20480 PASS**；12 s 采样 util 全 **0%**、功耗 9.29–14.54 W、温度 27 °C → **G3 非持续满载 PASS**），但存在**非本用户** compute 进程：PID **16507** owner **seey6/uid 1001**（本用户 `zhaoxiangyu`/uid 1005），382 MiB，`python main.py --port 7860 --listen 0.0.0.0`，`/proc` 属主核对一致，已运行 2 天 19 小时 → 门槛 **G1（不得存在其他用户计算进程）FAIL**，总判 **BLOCKED**，`model_load_authorized=false`。按要求：未激活 `robobrain25`、未 import transformers/PIL、未加载模型、**未创建输出目录** `artifacts/token_selector/prompt_projection_features`（确认不存在）、未做视觉/提示前向、未抽取/训练/评估 CV-Bench、未 kill/暂停/修改任何进程、未联网、未改原模型/`inference.py`/`api.py`；`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。静态校验（只读）：唯一数据源 `datasets/prompt_projection_clean_inductive_v1.jsonl`，manifest SHA256 观测 `acab1a32…9213` == pin **PASS**；冻结 pin config `0b03f8f5…ef99`、extractor `cac52d49…eb70`、trainer `f34a7865…f815`、manifest_builder `d80186b6…a4c9`；split 合同按 `source_image_sha256` 严格同图不跨 split（159/38/39 图、318/76/78 记录），CV-Bench 禁入训练，运行期隔离未核验（NOT_RUN）。决策 `GPU_GATE_BLOCKED / MODEL_LOAD_NOT_ACTIVATED / PROMPT_PROJECTION_EXTRACTION_NOT_RUN`。证据：`reports/token_selector/2026-09-28-prompt-distillation-gpu-gate.{md,json}`、`/tmp/robobrain_harness_result.json`。精确下一步（未执行）：待 GPU 零非 `zhaoxiangyu` compute 进程 + 空闲 ≥20480 MiB + 非持续满载后重跑只读门禁；仅在门禁 READY 且独立授权门开启后才运行抽取器与训练。

- [x] 2026-09-28 **Selector → 真实 block1 fused QKV CPU 数值链验证（合成激活）→ PASS（仅接口/数值实现自洽）**（无 GPU/无板卡/无综合/无 cosim/无联网；只读真实 BF16 safetensors 的 4 个 block1 张量；不加载完整模型；未改任何 HLS/RTL/模型文件）：新增可复现 `scripts/validate_selector_qkv_numeric_chain.py` 与 `evaluation/tests/test_selector_qkv_numeric_chain.py`。逐张量只读 `model.visual.blocks.1.norm1.weight/bias`、`attn.qkv.weight/bias`：BF16、shard `model-00002-of-00002.safetensors`，shape 分别 `[1024]/[1024]/[3072,1024]/[3072]`（确认 `W[3072,1024]`、`bias[3072]`），记录 header/张量 raw sha256 且读后文件 mtime/size 不变。用确定性合成（`default_rng(seed)` + 固定低频 feature bias，**非真实激活**）block0 输出，3 个 seed 各一个完整 M=192（K=48）、hidden=1024、N=3072 case + 1 个 M=8 case；对 A=`FP32 LN(x)·W+b` 逐级报 B=selector 全局对称 INT8（`q=round_half_away(x*s)`、`x≈q/s`、zero point 0、饱和 `[-128,127]`，本报告 `s=128/amax`）→ LayerNorm(norm1,eps=1e-6) → FP32 W；C=在 B 上 LN 输出经既有 Q8_0（复用 `export_real_tensor_case.py::q8_quantize`，block32、`d=amax/127`、饱和 `[-127,127]`、scale 存 binary16）量/反量化；D=在 C 上真实 W 也 Q8_0 化，按 `q8q8_gemm.cpp` 语义做 INT32 每 32 块累加再 +bias。结果（D vs A）：relative_L2 **0.0074–0.0084**、cosine **0.999965–0.999973**、max_abs **0.049–0.068**；诊断：selector 输入量化 rel_L2 **0.0088–0.0113**、LayerNorm 后 **0.0086–0.0110**、激活 Q8_0 **0.0072**、权重 Q8_0 **0.0062**；INT32 模拟 vs 反量化 FP32 rel_L2≈**2e-7**（自洽）。明确 selector `-128` 与 Q8_0 `-127` 差异，且 `1/s` 广播 **≠** LayerNorm。自洽检查全 PASS：Q8_0 通用实现与既有参考在整除 32 时逐字节一致、分块 FP32 与直接 FP32 一致、INT32 累加与反量化 FP32 一致、fused QKV `[0:1024]|[1024:2048]|[2048:3072]` 三段等于三次 GEMM、确定性重复逐字节一致。权重加载（0.020 s）与一次性 W Q8_0 量化（0.063 s）**不计入**每 case kernel 链计时（M=192 kernel_total≈0.26–0.29 s，peak RSS≈0.19 GB）；BLAS 固定单线程。测试：新聚焦 `test_selector_qkv_numeric_chain` **20/20 OK**（含全局 INT8 round/clamp/dequant、LayerNorm 顺序、Q8_0 block32 尾界通用、bias 必加、fused QKV 三段布局、确定性、只读权重加载、分块/INT32 等价），全量 `unittest discover -s evaluation/tests` **276 OK**（基线 256）。边界：**CPU 软件模拟**，激活为**合成非真实**，数值链只到 `+qkv.bias`，**无时延/加速声明**，**不判任务精度**。决策 `SELECTOR_QKV_CPU_NUMERIC_CHAIN_IMPLEMENTED / INTERFACE_AND_IMPLEMENTATION_SELF_CONSISTENT / SYNTHETIC_ACTIVATION / NO_TASK_ACCURACY_CLAIM`。证据：`reports/token_selector/2026-09-28-selector-qkv-numeric-chain.{md,json,csv}`、`scripts/validate_selector_qkv_numeric_chain.py`、`evaluation/tests/test_selector_qkv_numeric_chain.py`、`/tmp/robobrain_harness_result.json`、`CURRENT_STATUS.md`。精确下一步（未执行）：仅在本 CPU 链被接受后，再决定是否在 HLS 中实现 int8→fp32/LayerNorm-Q8_0 适配器与显式 bias-add 算子；无独立授权不得综合/建 BD/写驱动/上板；任务精度仍需完整下游路径。

更新时间：2026-09-28（**Selector → 真实 block1 fused QKV CPU 数值链验证完成**：合成激活、CPU-only、只读真实 BF16 block1 norm1/qkv 四张量、分级 B/C/D 相对 A 误差与分项误差、INT32 Q8Q8 等价模拟；20/20 聚焦 + 276 全量测试 OK；无 GPU/板卡/综合/联网/加速声明，不判任务精度；数值链只到 `+qkv.bias`）

- [x] 2026-09-28 **Selector 下游兼容性审计「真实模型拓扑修正版」→ CORRECTED（只读/无 GPU/无板卡/无综合/无联网，未改算法/RTL/HLS/驱动，未实施数值链）**：修正首版 3 个真实模型拓扑错误——(1) `Qwen3VLVisionAttention` 实际是**单个 fused `attn.qkv` Linear[1024→3072] 带 bias** + `attn.proj`[1024→1024] 带 bias（`modeling_qwen3_vl.py:175-176`，Transformers 4.57.0，sha256 `dd63ed3b…b22da`），**不存在**独立 Q/K/V 权重；block0/1 全部张量为 BF16、位于 `model-00002-of-00002.safetensors`，仅读 header 取证未加载整模型（index sha256 `58a7841d…cf2b`）。(2) B1 在 block0 后、block1 前，block1 真实拓扑 `hidden + attn(norm1(hidden))` 然后 `hidden + mlp(norm2(hidden))`（`:267-274`，LayerNorm eps=1e-6）：selector 段0 不能直接喂 qkv/fc1，必须先按 `quantization_contract.md` 反量化（`x≈q/s`）→ LayerNorm(norm1.weight/bias) → Q8_0 requant；**`x_scale=1/s` 广播只是反量化，未证明且不成立为 LayerNorm 等价**。(3) `pl_resident_ffn` LayerNorm-Q8_0 核输入是 4×fp32/128-bit（`ffn_vector.h:11-12`），不是 selector int8，需 int8-global-scale→fp32 adapter 或新建 int8 变体；`q8q8` 签名无 bias，fused QKV 的 [3072] bias 需新算子。修正后**唯一 model-correct 直接候选**：`Selector→Dequant→LayerNorm-Q8_0→fused QKV GEMM(M=4K,N=3072,K=1024)→+qkv.bias`；fc1/O/fc2 均**非直接边界**（中间有完整 attention+残差+norm2）。`+qkv.bias` 之后仍缺 reshape/split、RoPE、softmax、PV、proj+bias、residual、norm2、fc1/GELU/fc2 与 blocks 2..23，**下游不完整，不得声称完整**。最小验证候选改为 **CPU 数值链**（真实 block1 norm1/qkv 切片或完整一层流式读取，对 BF16/FP32 参考报 dequant/+LayerNorm/+Q8_0/+Q8Q8/+bias 各级 max_abs/rmse），**本轮不实现，需另行授权**。零返回路径仍只有 shared-DDR/片内且仍有 materialization，且必须先插入该 elementwise 链。证据：`reports/token_selector/2026-09-28-selector-downstream-compatibility.{md,json}`、`/tmp/robobrain_harness_result.json`、`CURRENT_STATUS.md`。

- [x] 2026-09-28 **Selector v4 与现有 Q8Q8/PE16 DATAFLOW/PL-resident FFN 下游核接口兼容性审计 → AUDIT_ONLY（无 GPU/板卡/综合/bitstream/联网，未改算法/RTL/HLS/驱动，未实施融合）** [**已被上方 2026-09-28「真实模型拓扑修正版」取代：Q/K/V 实为 fused qkv、fc1 非直接边界、必须先 Dequant→LayerNorm→Q8_0；原文保留为历史记录**]：只引用实际存在的源文件与报告。Selector v4 输出为 `[K,4,1024]` int8（group 主序、patch 次序、128-bit 小端 lane；real 原样复制、context 池化），加 `group_index/source_type/region_id/valid` 4 段 metadata；`scale` 仅 float 透传、不入 5 段布局；`patch_indices` 未输出但可由 `group_index` 推 `4g+[0,1,2,3]`。`M=4K, N=1024` 可无歧义恢复为行主序激活矩阵，`M=192/256/384`（K=48/64/96），与 Q8Q8 `activations_q[gm*k_blocks+kb]` 在 `k=1024` 时**逐字节一致**。下游清单：`q8q8_pe16_dataflow_hls`（TM=16/TN=64/PE_N=16/QK=32/OUTPUT_LANES=4，M≤4320/N≤9728/K≤9728，Q8_0 每 32 元素 binary16 scale，5 个 `m_axi` 256-bit/16-bit/128-bit + `s_axilite` offsets 0x00–0x64）；PE16 基线与 PE8 族同构；Decoder `gemv_stream6` 为 M=1 GEMV（Vision prefill **INCOMPATIBLE**）；`pl_resident_ffn` 向量核（4×fp32 128-bit→Q8_0 256-bit+fp16）可作为中间激活/重量化桥梁。真实 Vision 形状（config.json）：hidden=1024、intermediate=4096、heads=16、head_dim=64。兼容矩阵：Q/K/V/fc1 **维度 PASS + token 字节 DIRECT + scale ADAPTER_REQUIRED**（全局 float `s` → `[M,32]` binary16，`x_scale=1/s`）；O 投影与 FFN fc2 因输入非 selector 段 0 判 **INCOMPATIBLE as selector boundary**（间接边界）；**无维度层 INCOMPATIBLE**；权重侧阻塞：Vision tower 的 Q8_0 权重仓库不存在（现有 Q8_0 GGUF 仅语言模型）。**score 生成不在 selector 内**（`score_q` 是输入），与异构模型 `T_score` 外置一致。J_F_CONTINUE：两核均为 `m_axi+s_axilite`、BD 仅 SmartConnect+PS HPC/HP 且**无 `axi_dma`**、无 selector BD，故 `m_axi` master→master **不能直连 stream**；当前唯一可行是 **shared-DDR（或片内共享）零返回**，但仍有 K*4096 B（≤394 KB）的 memory materialization；真正零 materialization 需 **AXI4-Stream variant + 128→256 宽度转换/FIFO**（仓库不存在，属 HLS 接口改动）。最小可验证候选 **F1**：selector 段 0 → scale-adapter → `q8q8_pe16_dataflow`（Q/K/V 或 fc1），先 CPU 逐字节对拍 + 合成 scale 再做 HLS dataflow C-sim，**不建 BD/不改接口/不综合/不上板**。阻塞：scale adapter 缺失、无 selector↔Q8Q8 BD/驱动、无 AXI4-Stream 数据面、无 Vision Q8_0 权重、`T_fpga_downstream` 未实测、selector 无整机/上板签核。证据：`reports/token_selector/2026-09-28-selector-downstream-compatibility.{md,json}`、`/tmp/robobrain_harness_result.json`。
- [x] 2026-09-28 **Jetson Xavier NX + ZCU102 Selector 异构需求模型：基线混淆修正 + 两套严格公式（只建模/只参数扫描）→ PASS**（无 GPU/无板卡/无联网/无实测加速声明）：修正 v1 的基线混淆——旧单一 `J_ONLY` 已含 Jetson 软件 selector 与 K-token 后续视觉，故 `J_F_J` 与它共享同一 K-token 后续视觉，post-B1 节省不能抵消 FPGA 往返；改对 Dense 基线后 score 生成也不再抵消。改为四个 placement：`J_DENSE`（无 score/selector，dense 后续视觉）、`J_PRUNED`（Jetson score+软件 selector+K-token 后续视觉）、`J_F_J`（Jetson score+FPGA selector 往返+同一 K-token Jetson 后续视觉）、`J_F_CONTINUE`（Jetson score+送 FPGA selector+真实 FPGA 下游）。输出两套严格公式：A 算法收益 vs `J_DENSE`：`J_PRUNED: S_prune > T_score+T_sel_sw`、`J_F_J: S_prune > T_score+T_link_dense+T_sel_fpga+T_link_return+2F`、`J_F_CONTINUE: T_vision_dense-T_fpga_downstream > T_score+T_link_dense+T_sel_fpga+F`；B 部署选择 vs `J_PRUNED`：`J_F_J` 仅当 `T_sel_sw > T_link_dense+T_sel_fpga+T_link_return+2F` 才更快（K-token 后续视觉严格抵消，公式**不含任何 vision saving**），`J_F_CONTINUE` 要求 `T_sel_sw+T_vision_pruned > T_link_dense+T_sel_fpga+F+T_fpga_downstream`。`T_score`/`T_sel_sw`/`T_vision_dense`/`T_vision_pruned`/`T_fpga_downstream` 均未实测，数表只给 communication+FPGA-selector known lower bound（例：G=256,K=64 csynth worst/2000 MB/s/0.5 ms → `J_F_J` vs `J_PRUNED` 要求 `T_sel_sw > 10.1964 ms`；`J_F_J` vs `J_DENSE` 要求 `S_prune > T_score + 10.1964 ms`），**禁止再写相对旧 `J_ONLY` 的 9–12 ms 完整 break-even**。字节合同/selector 证据不变（dense `G*4096+G*2`、5 段 64B 对齐 `K*4096` 输出；csynth worst 1,708,124 cycles、cosim min/avg/max 5,668/209,518/440,899、OOC 5.000 ns WNS +0.689 ns PASS）。CSV 字段更新为 comparison/baseline/placement/known_lower_bound_ms/required_expression/symbolic_missing_terms/numeric_threshold_ms/is_exact_threshold/lower_bound_only；JSON/harness 新增 `baselines`、`symbolic_missing_terms`、`key_thresholds`（harness 顶层仅 status/baselines/contracts/evidence/placements/sweep/key_thresholds/tests/warnings/files_changed/next_gate）。验证：`py_compile` 2 文件 PASS；`test_heterogeneous_selector_break_even` **34/34**（含“`J_F_J` vs `J_PRUNED` 公式不含任何 vision saving”“`J_F_J` vs `J_DENSE` 含 `T_score` 符号项”“仅 B/J_F_J 数值精确”）；全量 `unittest discover -s evaluation/tests` **256 tests OK**。更新 `configs/heterogeneous_selector_break_even.json`（v2）、`scripts/analyze_heterogeneous_selector_break_even.py`、`evaluation/tests/test_heterogeneous_selector_break_even.py`、`reports/token_selector/2026-09-28-heterogeneous-break-even.{md,csv,json}`、`/tmp/robobrain_harness_result.json`。决策：`HETEROGENEOUS_SELECTOR_BASELINE_CORRECTED / NO_FULL_BREAK_EVEN / NO_MEASURED_SPEEDUP_CLAIM`。精确下一步（未执行）：在 Jetson 实测 `T_score`、`T_sel_sw`、`T_vision_dense`、`T_vision_pruned`，并实测 Jetson↔ZCU102 有效吞吐与固定软硬件+DMA 开销；`J_F_CONTINUE` 还需真实并实测的 FPGA 下游核；在此之前不得声称加速或给出完整 break-even 毫秒数。

- [!] 2026-09-28 **干净归纳 Prompt 投影特征抽取的 GPU 门禁（只读）→ BLOCKED，抽取未运行、不加载模型**：按 `nvidia-smi` + `ps` + `/proc` 审计。GPU `NVIDIA RTX A6000`（驱动 580.178.04、CUDA 13.0）总 46068 MiB、已用 3426 MiB、空闲 **42060 MiB（≥20480 MiB 达标）**；但 **GPU 利用率 97–100%**（采样 99/99/99/97/99/100）、功耗 254–296 W / 300 W、温度 85–86 °C，属明显满载。compute 进程 2 个且**均非 zhaoxiangyu（uid 1005）**：PID 16507 `seey6`/uid 1001/382 MiB/`python main.py --port 7860 --listen 0.0.0.0`；PID 884366 `zhaoshengtao`/uid 1004/2950 MiB/`python sensitivity/weight_sensitivity.py --resume-layerwise … phase1_weight_sensitivity_kodak24`/R/CPU 100%。门槛 G1（无非本用户 compute）**FAIL**、G2（空闲 ≥20480 MiB）PASS、G3（非满载）**FAIL** → 总判 **BLOCKED**，`model_load_authorized=false`。**未激活模型加载路径**：未 source 激活环境运行抽取器、未 import transformers/PIL、未加载模型、未创建 `artifacts/token_selector/prompt_projection_features`（此前不存在）、未做任何视觉/提示前向；故 index=472/npz=472/唯一图 236/`vision_forwards`=236/prompt calls=472、禁止键/schema/dtype/shape、source/config/manifest hash 一致性、按图 hash 重放 split（159/38/39 图、318/76/78 记录）**全部未核验（NOT_RUN）**；无缓存可 resume、未删除任何内容。pinned 待运行输入：script `cac52d49…eb70`、config `0b03f8f5…ef99`、manifest `acab1a32…9213`。未训练、未跑 CV-Bench、未联网、未改模型、未 kill/暂停他人进程、未宣称精度。决策：`GPU_GATE_BLOCKED / MODEL_LOAD_NOT_ACTIVATED / PROMPT_PROJECTION_EXTRACTION_NOT_RUN`。证据：`reports/token_selector/2026-09-28-prompt-feature-extraction.{md,json}`、`reports/token_selector/logs/gpu_audit_20260928T013413.log`、`/tmp/robobrain_harness_result.json`。精确下一步（未执行）：出现零非 zhaoxiangyu compute 进程 + 空闲 ≥20480 MiB + 非满载后重跑本只读门禁；仅在门禁 READY 且独立授权门开启后才运行 `extract_prompt_projection_features.py --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --manifest datasets/prompt_projection_clean_inductive_v1.jsonl --config configs/prompt_projection_distillation.json --out-dir artifacts/token_selector/prompt_projection_features`。

- [x] 2026-09-28 **Prompt 投影特征抽取器同图复用（CPU 审计 + 可注入执行层）→ PASS**（无 GPU/无模型/无真实抽取/训练/推理）：审计确认 `processor image(pixel_values,image_grid_thw)`、block0 `group_mean[G,1024]`、完整 dense vision+merger `merged_embeddings[G,2560]`、grid 均为**图像相关且与 prompt 无关**；`prompt_embeddings[T,2560]` 与 teacher max-cosine 逐记录独立。据此按 `(resolved path, source_image_sha256)` 全 manifest 分组实现**一次视觉前向、多 prompt 复用**，prompt embedding/teacher 目标不共享。关键数据结构事实：冻结 `datasets/prompt_projection_clean_inductive_v1.jsonl`（472 记录/236 图，sha256 `acab1a32…9213`；236 唯一图片文件 35,054,288 B；每图 2 prompt=grounding 236+pointing 236）**非图像连续**（先全部 grounding 再全部 pointing，首现块 472≠236），故“只缓存当前图”的朴素流式实现会做 472 次前向；本实现跨全 manifest 分组并用有序写入器 `_OrderedIndexWriter` 保证输出 index **严格 manifest 顺序**。resume：`--overwrite` 截断重建、否则追加，已完成后不写 NPZ/index；同图部分 pending 时最多一次前向、全部完成零前向；缓存仅驻留当前图（`finally` 释放，CUDA 下每组后 `empty_cache`）；同路径内容 hash 变化（跨 run resume 或 run 内）**hard fail**。新增可注入 `run_prompt_projection_extraction(...)`（`resolve_image_fn`/`image_sha256_fn`/`image_loader_fn`/`vision_capture_fn`/`prompt_embeddings_fn`/`save_cache_fn`/`index_record_fn`/`release_fn`），真实 `main()` 仍先 fail-closed manifest pin + GPU 审计再加载模型，`--validate-manifest-only` 纯 CPU（子进程 import guard 证明不 import transformers/PIL 且 `gpu_used=false,model_loaded=false`）。验证：`py_compile` 5 文件通过；新增 `test_prompt_projection_feature_reuse` 10/10（同图 2 prompt=1 vision+2 prompt、异图各 1、partial resume=1、full resume=0、同路径 hash 变化 hard fail×2、与逐样本 legacy 参考 5 数组逐数组相等、交错组 index 仍 manifest 序、validate-only 隔离）；`test_prompt_projection_distillation` 37/37；全量 `unittest discover -s evaluation/tests` **222 tests OK**。理论收益（**仅为结构计数，非实测，不伪造任何 wall-clock**）：vision forward 472→236，**2.0× 减少**；prompt embedding 仍 472；NPZ schema/教师/学生/split/manifest pin 均未改。决策：`PROMPT_FEATURE_IMAGE_REUSE_IMPLEMENTED / CPU_TESTS_PASS / EXTRACTOR_NOT_RUN / GPU_NOT_USED`。证据：`evaluation/extract_prompt_projection_features.py`、`evaluation/tests/test_prompt_projection_feature_reuse.py`、`reports/token_selector/2026-09-28-prompt-feature-image-reuse.{md,json}`、`/tmp/robobrain_harness_result.json`。精确下一步（未执行）：GPU 审计通过（无非本用户 compute、空闲 ≥20480 MiB、非满载）后，在独立授权门下运行 `extract_prompt_projection_features.py --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --manifest datasets/prompt_projection_clean_inductive_v1.jsonl --out-dir artifacts/token_selector/prompt_projection_features`，预期 `vision_forwards=236`，再训练 `Wt/Wv`；在此之前不得加载模型或用 GPU。

- [!] 2026-09-28 **Prompt 投影蒸馏数据来源与污染风险审计（只读，无 GPU/无模型）→ LEAKAGE_RISK**：审计 `configs/prompt_projection_distillation.json`、`evaluation/{extract,train}_prompt_projection*.py`、全部 CV-Bench 锁定 manifest/index 与本地 COCO/CV-Bench 数据。数据流：抽取来源由 `--manifest` 决定、**config 不锁定**；内部 split 按 `source_image_sha256` 的 `sha256(salt|seed|hash)` 分组；正式评测源为 `datasets/cv_bench_{full2638,screening100}.jsonl`（16-token 预注册）与 `..._budget64_diagnostic.jsonl`（64-token 诊断）。**核心风险**：唯一有记录的抽取命令用的是正式 64-Token pilot100 manifest（`datasets/cv_bench_screening100_budget64_diagnostic.jsonl`，`--limit-per-task 4`）；静态重放配置 split 后 limit-4 子集的 train/val/test 图像 11/2/3 张**全部**与 pilot100+full2638 重叠（100%），sample_id/prompt 亦 100% 重叠；source 换 full2638 则 1769/408/396 张图像全重叠。内部哈希分组只保证同图不跨内部 split，**无法**防外部评测集重叠。当前无 cache/index/WtWv，污染属**将产生**而非已发生；无监督标签泄漏（教师为无标签 dense 显著性）。严格归纳下 documented 方案违反 `test once` 纪律；transductive 复用需 PI 明确授权并禁止用评测图选参。本地可验证无泄漏独立数据：`visual_refinement_development+calibration` 去 52 张 CV-Bench COCO 重叠后 **236 图/472 记录**（聚合图集 SHA256 `ca8f8a6a…0bb6b`，35,054,288 B）；`val2017-original` 197 张虽净但无现成 prompt manifest、不可直接用。决策：`LEAKAGE_RISK_STATIC_AUDIT_COMPLETE / CONTAMINATION_OBSERVED_FALSE / CONTAMINATION_PROSPECTIVE_TRUE / GPU_NOT_USED`。证据：`reports/token_selector/2026-09-28-prompt-distillation-data-provenance-audit.{md,json}`、`/tmp/robobrain_harness_result.json`。精确下一步（未执行）：先由 PI 确认 inductive/transductive 协议并选择 CV-Bench 独立抽取 manifest，再按 GPU 门禁抽取与训练；在此之前不得运行 documented 的 `cv_bench_screening100_budget64_diagnostic.jsonl` 抽取命令。

- [x] 2026-09-28 **Prompt 投影蒸馏严格归纳数据治理：干净语料冻结 + fail-closed manifest pin（无 GPU/无模型）→ PASS**：新增可复现 `evaluation/prompt_projection_clean_inductive.py`（build/validate），仅合并 `visual_refinement_20260912_development.jsonl`(384) 与 `..._calibration.jsonl`(192)，按从冻结 CV-Bench parquet（`test_2d` SHA256 `33196034…ecbc` / `test_3d` `ef91fe8b…bbc25`）恢复的 **805** 个 COCO image id 删除全部重叠，构造 `datasets/prompt_projection_clean_inductive_v1.jsonl` = **236 唯一图 / 472 记录**（dev 318 + cal 154；grounding 236 / pointing 236；排除 52 = 33+19；图集 SHA256 `ca8f8a6a…0bb6b`；35,054,288 B；manifest SHA256 `acab1a32…9213`），`sample_id` 新命名 `ppclean-v1-<coco_image_id:012d>-<coco_annotation_id>-<task>`，保留 `source_sample_id`/`source_manifest`/`source_record_index`。生成 `datasets/prompt_projection_clean_inductive_v1.lock.json`（SHA256 `a723458d…e105`，`strict_inductive=true`，冻结两源/两 parquet/四个正式 CV-Bench manifest、52 排除 id、236 保留 id、每图 path/size/sha256、记录/任务数、最终 manifest hash、构造脚本 hash）。overlap 门全部 PASS：零 CV-Bench COCO id、零正式 eval image SHA（236∩2573=0）、零正式 sample_id（472∩2638=0）、图像全存在、同图同 split、train/val/test = 159图/318、38图/76、39图/78 非空。治理缺口修复：`configs/prompt_projection_distillation.json` 显式 pin `extraction.manifest={path,sha256}`；`extract_prompt_projection_features.py` 启动 fail-closed 校验 CLI manifest 解析路径+sha256 == pin（失败返回 3），新增仅 CPU `--validate-manifest-only`（不跑 GPU 审计/不加载模型，实测 `status=PASS,gpu_used=false,model_loaded=false`）；对旧污染来源 `cv_bench_screening100_budget64_diagnostic.jsonl` 运行 validate-only 返回 `FAIL`/exit 1。测试：`py_compile` 5 文件通过；`test_prompt_projection_clean_inductive` 11/11 + `test_prompt_projection_distillation` 37/37 + `test_hardware_regular_token_selector` + `test_metrics` 合计 76/76，`test_hardware_regular_b1_integration` 19/19。**不声称 prompt 文本分布与 CV-Bench 相同**（grounding/pointing ≠ Count/Relation/Depth/Distance；内部 test 仅 sanity check）。未使用 GPU/模型/推理/训练/联网，未访问 `visual_refinement_20260912_test.jsonl`(sealed)、未访问 ZCU102。决策：`CLEAN_INDUCTIVE_MANIFEST_FROZEN / STRICT_INDUCTIVE_VALIDATION_PASS / EXTRACTOR_PIN_FAIL_CLOSED / GPU_NOT_USED`。证据：`datasets/prompt_projection_clean_inductive_v1.{jsonl,lock.json}`、`evaluation/prompt_projection_clean_inductive.py`、`evaluation/tests/test_prompt_projection_clean_inductive.py`、`reports/token_selector/2026-09-28-prompt-distillation-clean-manifest.{md,json}`、`/tmp/robobrain_harness_result.json`。精确下一步（未执行）：GPU 属主 + 空闲 ≥20 GiB + 非满载门禁通过后，仅以该 manifest 运行抽取器，再训练并在冻结 best epoch 后对正式 CV-Bench 只做一次最终评测；旧 documented 污染命令已被 fail-closed 拒绝。

- [!] 2026-09-28 **GPU 安全审计（只读）→ BLOCKED，不加载模型**：按 `nvidia-smi` 完整审计（型号/显存/利用率/功耗/温度/compute PID）并用 `ps` + `/proc` 核对归属。GPU `NVIDIA RTX A6000`（驱动 580.178.04、CUDA 13.0）：总 46068 MiB、已用 3426 MiB、空闲 **42060 MiB（≈41.07 GiB，≥20 GiB 达标）**；但 **GPU 利用率 98%**（14 s 采样 avg 93%/max 99%）、显存利用率 58%、**功耗 292.52 W 平均 / 308.59 W 瞬时 vs 300 W 上限**、**温度 85 °C**（目标 84/最高 93/slowdown 95）、风扇 67%、P2，属明显满载。compute 进程 2 个且**均非 zhaoxiangyu（uid 1005）**：PID 16507 `seey6`/uid 1001/382 MiB/`python main.py --port 7860 --listen 0.0.0.0`/已运行 2 天 6 小时；PID 884366 `zhaoshengtao`/uid 1004/2950 MiB/`python sensitivity/weight_sensitivity.py --resume-layerwise … phase1_weight_sensitivity_kodak24`/Rl/100% CPU/25 分 30 秒。门槛 G1（无非本用户 compute）**FAIL**、G2（≥20 GiB）PASS、G3（非满载）**FAIL** → 总判 **BLOCKED**，`model_load_authorized=false`。本轮只读：未加载模型、未跑推理、未改代码、未 kill/暂停他人进程。证据：`reports/token_selector/gpu_audit_2026-09-28.json`、`reports/token_selector/gpu_audit_2026-09-28.md`、`/tmp/robobrain_harness_result.json`。精确下一步（未执行）：待 GPU 零非 zhaoxiangyu compute 进程 + 空闲 ≥20 GiB + 非满载后重跑本审计，方可在独立授权门后加载模型。

- [x] 2026-09-28 Selector v4 **Verilog IP 导出 + Vivado OOC synth/opt/place/route（PASS）**（未上板/未生成 bitstream/未写驱动/未跑 GPU）：新增可复现脚本 `scripts/export_selector_v4_ip_ooc.tcl`（Vitis HLS 2023.2：`csynth_design` + `export_design -rtl verilog -format ip_catalog -flow impl`，等价于既有成功 HLS-IP OOC 流程 `fpga_accel/q8q8_pe16_hls/scripts/run_hls.tcl`）、`scripts/run_selector_v4_ip_ooc.sh`（先 df 审计）与 collector `scripts/summarize_selector_ooc.py`。top `token_selector_top_packed`、part `xczu9eg-ffvb1156-2-e`、`create_clock -period 5.0 -name default`（不传 `-uncertainty`，保持 HLS 默认 27% = 1.35 ns；clock 未放宽），选择数学/量化/host 5 段布局/IP 端口均未改（`src/token_selector_hls.cpp`/`include/token_selector_hls.h` sha256 与 v4 csynth 冻结值一致：`078a3a18…d7da`/`f76ac1dd…ade9c`）。所有 HLS/Vivado project、cache、checkpoint、IP zip 在 `/tmp`（`/tmp/robobrain_token_selector_hls_export_v4`、`/tmp/robobrain_selector_ip_ooc_v4/ip/token_selector_top_packed_ip.zip` = 719,284 B、sha256 `3ea8b74f…f265`），仓库只存小报告/hash。结果（Vivado v.2023.2）：route `logical 145683 / routable 52474 / fully routed 52474 / routing errors 0 / unrouted 0`；timing `ap_clk 5.000 ns`，**WNS +0.689 ns、TNS 0、WHS +0.036 ns、THS 0、failing endpoints 0**；utilization **LUT 24,949/274,080 = 9.10%、FF 29,458/548,160 = 5.37%、BRAM 42/912 = 4.61%（RAMB18 等价 84/1,824 = 4.61%）、DSP 85/2,520 = 3.37%**；DRC **Error 0 / Critical Warning 0**（130 Warning：DPIP-2 16 / DPOP-3 38 / DPOP-4 75 / RTSTAT-10 1，OOC 典型）；post-route 真实 critical path 在**未改动的 `ts_pool_context_packed`**：RAMB18E2 `weights_U` CLKARDCLK → DSP48E2 `mul_ln290` DSP_OUTPUT ALU_OUT[10]，data path 4.258 ns（logic 2.278 / route 1.980）、5 级 DSP、slack MET +0.689 ns。硬门槛（export/route 0 error、WNS/WHS≥0、unrouted=0、DRC Error=0、四类资源<50%）全过，status **PASS**（`SELECTOR_V4_IP_OOC_ROUTE_PASS`）。**边界**：HLS 导出的 OOC XDC 只含 `create_clock -period 5.000`，不含 HLS 的 1.35 ns 调度 uncertainty，故 post-route 数字是本流程标准 OOC 结果，**不等于** csynth pre-route 3.650 ns/0 ns 估算（pre-route 估算）、**不等于** cosim 功能 PASS（只证明功能），也不等于含 PS/DDR 的整机/上板签核。同时修正文档：正式 `score_q` 合同为 unsigned Q0.15、合法上限 **32767**；`bounddiff` 的 0/65535 交替仅是 `uint16` 原始载荷鲁棒性压力测试、**不是合同内量化输入**（用例保留），并修正“每年区域”→“每个区域”等笔误。证据：`reports/token_selector/2026-09-28-selector-ooc-v4.md`、`artifacts/token_selector/2026-09-28-selector-ooc-v4/{summary.json,route_status_routed.rpt,timing_summary_routed.rpt,utilization_placed.rpt,drc_routed_summary.rpt,clock_utilization_routed.rpt,critical_paths_routed.rpt,critical_path_routed.md,df_before.txt,df_after.txt}`、`scripts/export_selector_v4_ip_ooc.tcl`、`scripts/run_selector_v4_ip_ooc.sh`、`scripts/summarize_selector_ooc.py`、`README.md`、`fpga_accel/token_selector_hls/README.md`、`/tmp/robobrain_harness_result.json`。精确下一步（未执行）：如要上板，需独立授权门后把 IP 集成进含 PS/DDR 的 ZCU102 系统做整机 route、生成 bitstream、写驱动并做板级 200 MHz/功能验证。

- [x] 2026-09-28 Selector v4 **测试缺口修复 + 独立 C/RTL cosim（功能性 PASS）**（未上板/未导出 IP/未导出 bitstream/未写驱动/未跑 GPU）：保留 220 组内存内随机差分，新增 **48 组确定性边界差分** `bounddiff_v4_00..47`（显式覆盖全相等、0/65535 交替、重复档位、随机饱和值；网格 32x128/64x64/128x32/31x127/127x31/16x16/24x24/32x32；K=16/32/48/64/80/96；local_cap=1/16），逐例释放内存，每例新 packed v4 对冻结 v3-int8 逐字节；普通与 ASan/UBSan 均 `passed=475 failed=0`（207+220+48，无 sanitizer 报错），Python 4 模块 `Ran 76 tests OK`。更正措辞：旧 `score_mod=1..6` 只是量化并**提高 tie 频率**，不是“强制大量 tie”。新建独立 cosim testbench `tb/tb_token_selector_cosim_v4.cpp` 与 `scripts/run_cosim_v4.tcl`（cosim 工程 `/tmp/robobrain_token_selector_hls_cosim_v4` ≠ csynth 工程 `/tmp/robobrain_token_selector_hls_csynth_v4`），只回放 3 个冻结代表 case（identity 4x4/K48；16x16/K64 全 tie+context；32x32/K96 极值+context），C golden 由冻结 Python 参考固化为 TSIN/TSEX。Vitis HLS 2023.2：C simulation `0 errors`；C/RTL cosim RTL **Verilog**、`-trace_level port`、xsim，**`C/RTL co-simulation finished: PASS`**，post-check `passed=3 failed=0`，逐例核对 status/count/scale/全部 token 字节/全部 metadata。**边界：cosim 只验证功能，不证明 post-route 200 MHz；v4 csynth 仍是 pre-route 0 ns 估算裕量。** 证据：`reports/token_selector/2026-09-28-selector-cosim-v4.md`、`artifacts/token_selector/2026-09-28-selector-cosim-v4/`、`fpga_accel/token_selector_hls/README.md`、`/tmp/robobrain_harness_result.json`。精确下一步（未执行）：仅在独立板卡/授权门后才写驱动、导出 IP 或上板。

- [x] 2026-09-28 Selector packed top v4：**关闭 5 ns 时序门槛（PASS）**（未上板/未 cosim/未导出 IP/未写驱动/未跑 GPU）。先提取 v3 真实 critical path：`token_selector_top_packed` 估计 4.423 ns、effective budget 3.650 ns（默认 1.35 ns uncertainty 不变），残余路径为 `entry` 物化循环的 `gmem_out_group` bus request 3.650 ns + control path 0.773 ns，根因是完全分区的 `local_top[9][16]`/pool/selected/entry 稳定插入以可变下标访问寄存器堆（摘要 `artifacts/token_selector/2026-09-28-selector-csynth-v4/v3_critical_path_summary.md`）。v4 在不改选择数学/host 5 段布局/default uncertainty/128-bit packed context 路径的前提下：删除 `local_top` bank 与所有 16/144 路插入，区内每个 region 的精确 top-`local_cap` 改为**顺序确定性多遍 max-select**（每 rank 扫描 region、跳过 anchor 与已选、score 高优先、tie 低 index，候选集合与 v3 完全相同），全局 details / selected 升序 / entry tuple 排序全部改为单比较器多遍 select + 标量 best 寄存器；centroid 代表点显式窄化 int32（已证明 `|delta|<=675721074`），两个 32x32→64 平方用 `#pragma HLS BIND_OP ... impl=dsp latency=2`；`ts_pool_context_packed` 逐行不变。v3 bounded-insertion int8 kernel/top 保留在 `#ifndef __SYNTHESIS__` 下仅作测试参考。验证：g++ 普通与 ASan/UBSan 均 `passed=475 failed=0`（184 fixtures + 220 组内存内随机差分 `randdiff_v3_000..219` + 48 组确定性边界差分 `bounddiff_v4_00..47`，新 multi-pass packed vs 旧 v3 insertion int8 逐字节）；Python 4 模块 `Ran 76 tests OK`（新增 `test_v4_multipass_matches_v3_insertion_200plus`，220 例 v3 insertion 差分）。受控 csynth（`scripts/run_csynth_v4.tcl`，top packed、part/5 ns/default uncertainty 不变、只 `csynth_design`，**1 轮修复即通过**）：**0 error；estimated period 3.650 ns = 3.650 ns budget、slack 0.000 ns、无 HLS 200-871/200-1016**；latency 有限 **worst 1,708,124 cycles = 8.541 ms @200 MHz < 33.3 ms**；**BRAM 7.57% / DSP 3.45% / FF 7.56% / LUT 16.41% / URAM 0** 全部 <50%。四项门槛全过，status **PASS**。证据：`reports/token_selector/2026-09-28-selector-csynth-v4.md`、`artifacts/token_selector/2026-09-28-selector-csynth-v4/{summary.json,csynth.rpt,csynth.xml,token_selector_top_packed_csynth.rpt,console_v4.log,v3_critical_path_summary.md,core_selection_diff.txt}`、`artifacts/token_selector/2026-09-28-selector-spec-v4.json`、`fpga_accel/token_selector_hls/README.md`、`/tmp/robobrain_harness_result.json`。精确下一步（未执行）：仅在通过独立板卡/授权门后才写驱动与导出 IP。C/RTL cosim 已由本文件顶部 2026-09-28 cosim 里程碑完成功能性验证；本轮不导出、不上板。

更新时间：2026-09-28（**Selector v4 下游兼容性审计已出「真实模型拓扑修正版」**：Q/K/V 为 fused `attn.qkv`[1024→3072]+bias 与 `attn.proj`[1024→1024]+bias（无独立 Q/K/V），block1 为 `hidden+attn(norm1(hidden))` 后 `hidden+mlp(norm2(hidden))`，必须 Dequant(global s)→LayerNorm(eps=1e-6)→Q8_0→fused QKV(M=4K,N=3072,K=1024)→+bias；fc1/O/fc2 **非直接边界**，`1/s` 广播**不等价** LayerNorm，`+qkv.bias` 后下游不完整，最小候选改为 CPU 数值链且未实施；以下为被修正的首版记录——新增 **Selector v4 下游核接口兼容性审计**：只读审计 selector 输出 `[K,4,1024]` 与 Q8Q8/PE16 DATAFLOW/PL-resident FFN 的维度/字节/scale/AXI 兼容性；Q/K/V/fc1 判 `ADAPTER_REQUIRED`（全局 float scale→per-32 binary16），O/fc2 为 selector 边界 `INCOMPATIBLE`；`J_F_CONTINUE` 现有仅 shared-DDR 零返回且仍有 memory materialization，无 AXI4-Stream；给出最小候选 F1 与阻塞项，未实施融合/未上板/无 GPU；异构 selector 需求模型**基线混淆修正**：新增 `J_DENSE`/`J_PRUNED` 两基线 + `J_DENSE`/`J_PRUNED`/`J_F_J`/`J_F_CONTINUE` 四 placement + 两套严格公式；数表只给 communication+FPGA-selector lower bound，禁止相对旧 `J_ONLY` 的完整 break-even；只建模/只参数扫描、读取 v4 csynth/cosim artifacts、无 GPU/无板卡/无联网、未声称加速，256 tests OK；干净归纳 Prompt 投影特征抽取的 GPU 门禁仍 BLOCKED；Selector v4 已导出 Verilog IP 并完成 Vivado OOC synth/opt/place/route，全部硬门槛 PASS；未上板/未生成 bitstream/未写驱动）

- [x] 2026-09-27 Selector token 数据面改 **128-bit packed AXI** + 受控 csynth v3（未上板/未 cosim/未导出 IP/未跑 GPU）：保留冻结的 int8 `token_selector_core`/`token_selector_top` 用于软件回归，新增 packed 孪生 `token_selector_core_packed`/`token_selector_top_packed`。word 为 `ap_uint<128>`（g++ 为位精确兼容的 `unsigned __int128` + lane 访问器），16 个连续 int8 小端 lane，每组 4096 B = 256 words，`word(group,w)` ↔ 原字节 `group*4096+w*16`；host 5 段布局/64 B 对齐/总字节数与 metadata 端口完全不变。context 池化按 region 只读未选 group，每 word 一次取 16 lane、16 路 `int8*uint16→int64` 并行累加，round-half-away 与全零均匀不变；real group 256 word 原样复制；不缓存全部 G tokens。核心 diff 仅注释与 token I/O（`artifacts/.../core_selection_diff.txt`）。验证：g++ 普通与 ASan/UBSan 均 `passed=207 failed=0`（184 fixtures：34 固定 + 150 随机，int8 与 packed 双 top 逐字节对拍 golden），新增 pack/unpack 任意 byte offset/lane 测试；Python 4 模块 `Ran 75 tests OK`（新增 9 项 packed-contract）。受控 csynth（`scripts/run_csynth_v3.tcl`，top packed，part `xczu9eg-ffvb1156-2-e`、5.0 ns、默认 1.35 ns uncertainty 不变，只 `csynth_design`，2 轮修复）：0 error；latency 有限 **worst 2,499,989 cycles = 12.50 ms @200 MHz < 33.3 ms**；资源 **BRAM 7.13% / DSP 2.66% / FF 33.87% / LUT 33.48%** 全部 <50%；但 5 ns 默认 uncertainty 下 estimated **4.423 ns**、slack **-0.773 ns**（`ts_pool_context_packed` 自身 3.650 ns 达标，残余关键路径在共享选择逻辑）。按客观门槛判 **`REDESIGN_REQUIRED`**，**未写 PASS**。证据：`reports/token_selector/2026-09-27-selector-csynth-v3.md`、`artifacts/token_selector/2026-09-27-selector-csynth-v3/{summary.json,csynth.rpt,csynth.xml,token_selector_top_packed_csynth.rpt,core_selection_diff.txt}`、`artifacts/token_selector/2026-09-27-selector-packed-spec-v3.json`、`evaluation/token_selector_packed_contract.py`、`fpga_accel/token_selector_hls/README.md`/`quantization_contract.md`/`register_map_draft.md`、`/tmp/robobrain_harness_result.json`。精确下一步（未执行）：在不改选择数学/host 布局/默认 uncertainty 前提下给共享选择逻辑关键路径（重点 `local_top[9][16]` 完全分区插入 mux 与代表点/质心）打拍或流水化，使 estimated period `<=3.65 ns`，再重跑同一条 `csynth_design`；在此之前不写驱动、不 cosim、不导出 IP、不上板。

- [x] 2026-09-27 定点 Selector v2 有界重构 + 等价性回归 + 受控 csynth 重跑（未上板/未 cosim/未导出 IP/未跑 GPU）：按审校方案固化 `TS_MAX_BUDGET=96` 与 `local_cap<=16`（Python `MAX_BUDGET`/`MAX_LOCAL_CANDIDATE_CAP` 与 C++ 同值；运行时仍接受 16 的正倍数且 `<=96`，保留 16/32/48/64/80/96 边界，`K=112/128` 返回 `ERR_BUDGET`）；区内改为稳定有界插入只维护精确 `top-local_cap`（不再收集最多 4096 项后全排序），全局候选池 `9*16=144` 有界插入选择；anchor 扫描缓存 `anchor_score`；所有动态 G 循环加 `LOOP_TRIPCOUNT max=4096`；输出 `m_axi` 按 tokens/每段 metadata 独立 bundle（host 5 段布局未变）；context 池化位精确不变、region/selected 改流式扫描。等价性：新增 150 组随机 fixtures + Python `_legacy_select`（旧“收集全部成员+全排序”独立实现）120 组随机逐项对拍；Python `Ran 66 tests OK`，g++ 普通与 ASan/UBSan 均 `passed=205 failed=0`，`csynth_design` 0 error、资源任一类 <50%（BRAM 40/2.19%、DSP 73/2.90%、FF 184171/33.60%、LUT 89111/32.51%、URAM 0）、顶层 latency **已有限**（best 2 / avg 592391 / worst 20561009 cycles）。两轮修复后仍 **门控未全满足**：worst 20561009 cycles 按 200 MHz 换算 102.81 ms > 33.3 ms；5 ns 默认 uncertainty 1.35 ns 下估计 4.423 ns、slack `-0.773 ns`（未改 uncertainty 掩盖）。主瓶颈是 context 池化 `VITIS_LOOP_610_28`（18,996,156 cycles），其内层受 `gmem_in_tokens` 8-bit 端口读带宽限制 II=8（1 byte/cycle，HLS 200-880）。如实判 `REDESIGN_REQUIRED`，**未写 PASS**。证据：`reports/token_selector/2026-09-27-selector-csynth-v2.md`、`artifacts/token_selector/2026-09-27-selector-csynth-v2/{summary.json,csynth.rpt,csynth.xml,token_selector_top_csynth.rpt}`、`artifacts/token_selector/2026-09-27-fixedpoint-selector-fixtures/`（184 case）、`scripts/run_csynth_v2.tcl`、`scripts/summarize_selector_csynth.py`、`/tmp/robobrain_harness_result.json`。精确下一步（未执行）：在**不改 host 5 段布局/选择数学/默认 uncertainty** 前提下加宽 token 取数通路（更宽内部 token 端口或预取 staging buffer）并/或把 int64 MAC 带寄存流水化，使 worst < 6.66e6 cycles 且估计周期 `<=3.65 ns`，再重跑同一条 `csynth_design`；在此之前不写驱动、不 cosim、不导出 IP、不上板。
- [x] 2026-09-27 定点 Selector 溢出边界 + 受控 csynth（未上板/未 cosim/未导出 IP）：把加权质心的 `int64` 安全性固化到 `MAX_GRID_DIM=128`（Python `MAX_GRID_DIM`、C++ `TS_MAX_GRID_DIM` 同值），单维 >128 返回 `ERR_GRID`；删除内核中全部 `__int128`。数学证明：区域成员 `n<=491`、`W<=16088597`、`|delta|<=675721074`、`distance<=9.13e17<2^63-1`（穷举最大 `4.4776e17`，grid `31x127`），证明与源码哈希见 `artifacts/token_selector/2026-09-27-selector-overflow-proof.json`。新增边界 128/129、`1xN/2xN` fixtures 与测试；Python 回归 `Ran 65 tests OK`，g++ 普通与 ASan/UBSan 均 `passed=49 failed=0`。HLS 源码静态审计：无动态分配/递归/异常/虚函数/STL 堆容器。
- [x] 新增 `scripts/run_csynth.tcl`（`open_project /tmp/robobrain_token_selector_hls_csynth`、top `token_selector_top`、part `xczu9eg-ffvb1156-2-e`、clock 5.0 ns、只 `csynth_design`），运行前记录 `df`，全部中间文件留在 `/tmp`（59 MiB），小报告复制到 `artifacts/token_selector/2026-09-27-selector-csynth/`。第 1 轮综合把数据通路优化为空（latency 0、154 dangling），根因是 core 未强制内联；加 `#pragma HLS INLINE`（`src/token_selector_hls.cpp:237`）后第 2 轮得到真实数据通路。
- [!] csynth 结果：资源在器件内 BRAM 128(7%)/DSP 75(2%)/FF 25933(4%)/LUT 29026(10%)/URAM 0；但顶层 latency 报告为 `?`（不有限），`VITIS_LOOP_344_10` II 违反（最高 II=11），timing slack `-0.11 ns`（估计 3.762 ns，可用 3.65 ns）。按客观门槛判 `FAIL_REDESIGN_REQUIRED`，**未写 PASS**。根因：`budget` 无编译期上界导致多数循环无法定 tripcount。证据：`artifacts/token_selector/2026-09-27-selector-csynth/summary.json`、`csynth.rpt`、`token_selector_top_csynth.rpt`。
- 决策：`SELECTOR_OVERFLOW_BOUND_128_PROVEN / CSIM_AND_REGRESSION_PASS / HLS_SOURCE_SYNTHESIZABLE_STATIC_AUDIT_PASS / CSYNTH_FAIL_REDESIGN_REQUIRED`。精确下一步（未执行）：给内核加编译期循环上界（显式 `budget<=4096` 契约 + `#pragma HLS LOOP_TRIPCOUNT` 或常量模板预算）后重跑同一条 csynth，直到顶层 latency 有限且 5.0 ns 收敛；在此之前不写驱动、不 cosim、不导出 IP、不上板。

- [x] 2026-09-27 定点 Selector 选择核 + g++ C 仿真（未综合/未上板）：新增位精确 golden `evaluation/hardware_regular_selector_fixedpoint.py` 与 `fpga_accel/token_selector_hls/`（m_axi+s_axilite 顶层、g++ 可编译、fixtures 生成器、tb、Makefile、README/寄存器草案/量化契约）。24 个冻结 fixtures（identity、K48/64/96、9x9/10x12/2x9/5x4、全同分、全零分、非法预算、唯一/anchor/context、布局）在 g++ 普通与 ASan/UBSan 下均 `passed=34 failed=0`；Python 新测试 20/20，与既有 selector/B1 回归合并 62/62；定点选择结构的 group_index/source_type/region_id/anchors/details 与既有浮点 selector 完全一致。决策 `FIXEDPOINT_SELECTOR_CSIM_PASS`。证据：`reports/token_selector/2026-09-27-fixedpoint-selector-csim.md`、`artifacts/token_selector/2026-09-27-fixedpoint-selector-spec.json`、`artifacts/token_selector/2026-09-27-fixedpoint-selector-fixtures/`。未改任何既有 selector 语义，未运行 HLS/csynth/cosim，未访问 ZCU102。精确下一步（未执行）：如需硬件化，仅运行 csynth 取得寄存器头并覆盖 `register_map_draft.md` offset。

- [x] 2026-09-27 HLS Selector 只读审计（未启动综合）：Vivado/Vitis/Vitis_HLS 2023.2、ZCU102 `xczu9eg-ffvb1156-2-e` 可用；PL 侧无任何 Selector/TopK/context 池化 HLS、RTL、寄存器草案或 testbench，仅 Q8×Q8 矩阵/GEMV/FFN 引擎；INT8 token 段只有字节数无量化契约、INT4 缺失、旧合同 JSON 哈希陈旧。决策 `READ_ONLY_HLS_SELECTOR_AUDIT_COMPLETE`。证据：`reports/token_selector/2026-09-27-hls-selector-audit.md`、`artifacts/token_selector/2026-09-27-hls-selector-audit.json`。

## 2026-09-27 B1 Prompt 投影蒸馏评估契约加固（已实现，未运行抽取/GPU）

- [x] 图像分组 split：抽取时对源图片文件字节做流式 SHA256，写 npz `meta_source_image_sha256` 与索引 `source_image_sha256`；正式训练按 `sha256(salt|seed|source_image_sha256)` 划分，**同图所有 prompt 必同 split**，缺/非法 hash 直接抛错拒绝；合成 12 样本/4 图覆盖同图绝不泄漏。
- [x] `hardware_real_group_metrics` 每个 K 拆成 `matched_real_budget`（R=real_group_count，对 teacher top-R 的 recall/jaccard/mass，公平真实预算比较）与 `total_budget_coverage`（对 teacher top-K，`upper_bound_only=true`，明确非满分）；删除含糊的顶层 `teacher_topk_recall`。
- [x] split 评估加入 `l2_baseline`（每组 L2 范数）与确定性 `student_shuffled_prompt`（同 split 按 sample_id 升序循环错配 prompt，同一 `Wt`/`Wv`），报告 `student_correct`/`l2_baseline`/`student_shuffled_prompt` 及 `correct_minus_shuffled` 在 pearson/spearman/ndcg 与 K 指标的差值；样本 <2 标记 `available=false` 不伪造。
- [x] config 与报告写清晋升判据：正式方法在 val 须优于 L2、正确 prompt 须优于 shuffled（只取差值符号，主指标 `spearman`+`ndcg_full`，不预设任意数值阈值）；test 仅冻结后评估一次，报告写 `test_used_for_selection=false`、`test_evaluated_once_after_freeze=true`。
- [x] 验证：`py_compile` 4 个 py 文件通过；新测试 33/33 OK；`test_hardware_regular_token_selector`+`test_hardware_regular_b1_integration` 回归 42/42 OK；三模块可导入。
- [!] 本轮未运行抽取器、未加载模型、未使用 GPU、未访问封存数据或 ZCU102；无真实特征/权重/指标。
- 决策：`PROMPT_PROJECTION_DISTILLATION_EVAL_CONTRACT_HARDENED / CPU_SYNTHETIC_TESTS_PASS / EXTRACTOR_NOT_RUN / GPU_NOT_USED`。证据：`evaluation/{prompt_projection_distillation,extract_prompt_projection_features,train_prompt_projection}.py`、`evaluation/tests/test_prompt_projection_distillation.py`、`configs/prompt_projection_distillation.json`、`reports/token_selector/2026-09-27-prompt-projection-distillation.md`、`artifacts/token_selector/2026-09-27-prompt-projection-distillation-spec.json`、`/tmp/robobrain_harness_result.json`。
- 精确下一步（未执行）：待 GPU 无非本用户计算进程且空闲显存 ≥20 GiB 后，先跑抽取器再跑训练器，命令见报告 MD 第 11 节。

更新时间：2026-09-27（B1 Prompt 投影蒸馏评估契约加固完成，抽取/GPU 未运行）

## 2026-09-27 B1 Prompt 投影蒸馏基础设施（已实现，未运行抽取/GPU）

- [x] 新增（未改任何既有源文件/模型）：`evaluation/prompt_projection_distillation.py`（冻结定义）、`evaluation/extract_prompt_projection_features.py`（本地 offline 抽取器 + 可注销 block0 hook + nvidia-smi 归属门禁）、`evaluation/train_prompt_projection.py`（确定性训练器与 float32 导出）、`evaluation/tests/test_prompt_projection_distillation.py`、`configs/prompt_projection_distillation.json`、`reports/token_selector/2026-09-27-prompt-projection-distillation.md`、`artifacts/token_selector/2026-09-27-prompt-projection-distillation-spec.json`。
- [x] 教师契约：完整 24 层视觉编码后 merged embedding `[G,2560]` 与原始 prompt（仅 tokenize 文本、`add_special_tokens=False`）`[T,2560]` 的最大 cosine；保存 minmax 原分数与 temperature soft target。学生：B1 block0 后每组 4 patch 均值 `[G,1024]`，`Wt[128,2560]`/`Wv[128,1024]` 投到 128 维后最大 cosine。
- [x] 缓存 schema：每样本 npz 仅 `group_mean:float16[G,1024]`、`prompt_embeddings:float16[T,2560]`、`teacher_scores:float32[G]`、`teacher_scores_normalized`、`teacher_soft_target` 与标量元数据；加载时禁止任何 dense hidden 键；JSONL 索引断点续跑。
- [x] 训练：确定性 seed、按 `sample_id` 哈希 split（train/val/test 两两无泄漏）、listwise KL 主损失 + 可配置 score MSE；best epoch 仅按 val 选，test 只冻结后评估一次；导出 float32 Wt/Wv + `weights_sha256`，与既有 `load_projection` 逐位兼容。
- [x] 指标：自实现 Pearson、Spearman（stable 平均秩，不用 SciPy）、NDCG@K、K=48/64/96 的 teacher top-K recall/Jaccard/saliency mass retained；另在 `source_type==REAL` 的真实 group 集合上报告同类 K 指标。
- [x] 验证：`py_compile` 通过；新测试 20/20 OK；既有 `test_hardware_regular_token_selector`+`test_hardware_regular_b1_integration` 回归 42/42 OK。
- [!] 本轮未运行抽取器、未加载模型、未使用 GPU；GPU 门禁已用注入式假命令在 CPU 覆盖四种状态。真实分布、真实训练与任何精度/加速均未验证。
- 决策：`PROMPT_PROJECTION_DISTILLATION_INFRA_IMPLEMENTED / CPU_SYNTHETIC_TESTS_PASS / EXTRACTOR_NOT_RUN / GPU_NOT_USED`。精确下一步：待 GPU 无非本用户计算进程且空闲显存 ≥20 GiB 后，先跑抽取器（`--gpu-audit-only` 或正式），再跑训练器导出 `artifacts/token_selector/prompt_projection_WtWv.npz`，最后用 `--score-mode prompt --projection-npz` 推理；命令见报告 MD 第 8 节。

更新时间：2026-09-27（Prompt 投影蒸馏基础设施实现完成，抽取/训练/GPU 未运行）

## 2026-09-27 B1 真实模型 smoke test（blocked，未加载模型）

- [!] GPU 前置审计未通过：A6000 46068MiB，已用4886MiB，空闲40600MiB（≥20GiB），功耗298.95W/300W，利用率100%，温度86℃；但存在**非`zhaoxiangyu`**计算进程：`seey6` PID16507（`python main.py --port 7860`，382MiB）、`ismy` PID562557（`woq.run facebook/opt-1.3b awq`，4410MiB），均经`ps -o user,pid,cmd`与`/proc`属主核对。
- [!] 按任务规则进入BLOCKED：未激活`robobrain25`、未加载任何模型、未选取样本/未保存smoke manifest、未执行dense/identity/L2 K=64、未做逐token一致性与K64结构检查。
- [x] 合规：未kill/暂停/修改任何进程，未改源代码与原始模型，未访问封存数据或ZCU102。
- 决策：`BLOCKED_GPU_AUDIT_NON_OWNER_COMPUTE_PROCESS`。证据：`reports/token_selector/smoke/2026-09-27-b1-real-model-smoke-{gpu-audit.json,status.jsonl}`、`reports/token_selector/2026-09-27-b1-real-model-smoke.{json,md}`、`/tmp/robobrain_harness_result.json`。
- 精确下一步：待GPU无非`zhaoxiangyu`计算进程且空闲≥20GiB时，重跑`nvidia-smi`与`ps -o user,pid,cmd`审计通过后，对锁定64-Token manifest（SHA256 `236e38a5…ec2e668`）第一条样本在同一加载下执行 dense/identity/L2 K=64 smoke test。

更新时间：2026-09-27（B1真实模型smoke test因GPU存在非本用户计算进程而blocked，无推理结果）

## 2026-09-27 粗到细64-Token 100题判别实验（completed，取代下方blocked占位）

- [x] GPU前置检查通过：`nvidia-smi`正常，A6000 46068MiB仅471MiB占用、0%利用率；唯一计算进程PID16507归用户`seey6`（`python main.py --port 7860`，382MiB），未kill、未修改。审计`reports/cv_bench/2026-09-27-coarse-to-fine-64token-pilot100-gpu-audit.txt`。
- [x] 复用锁定manifest（SHA256 `236e38a5…ec2e668`）、锁config及两个既有运行器（哈希与blocked记录一致`8b05cbf9…`/`41a7e9c8…`），未改算法或解析器；新报告脚本仅聚合既有字段。refinement 900条（dense/global256/cosine_global_roi×3重复×100题），pruning 300条（retention 1.0/0.5/0.25×100题），共1200条64-Token推理；pruning r1000与refinement dense逐字一致率1.0。
- [x] 结果：官方组合分Dense 89.1667%、l2_50 85.0000%、l2_25 79.5833%、global256 85.0000%、global_local 74.1667%；相对Dense保留100%/95.3271%/89.2523%/95.3271%/83.1776%；解析成功率100%/100%/100%/99%/89%；输出上限命中0/0/0/1/11；平均合并视觉Token 724.1/362.1/181.1/44.7/148.5，Token保留100%/50.0155%/25.0253%/14.4782%/47.5813%；模型核心均值219.153/156.503/128.670/105.941/271.400 ms，模型端到端243.196/179.422/151.514/121.787/308.394 ms。
- [x] global_local相对global256配对差值-10.8333pp、95%CI[-18.3333,-3.7500]；相对l2_50 -10.8333pp、CI[-18.3333,-3.3333]；本轮100题下区间均不含零。
- [x] 帕累托（官方分越高、模型端到端越低）：前沿`{dense, global256}`；global256支配global_local/l2_25/l2_50，dense支配global_local。仅数学结果，未设置通过线。
- [x] 证据：`reports/cv_bench/2026-09-27-coarse-to-fine-64token-pilot100.{json,csv,md}`（JSON含500条逐题记录，CSV为5配置汇总，另存`-perquestion.csv`500条）；`artifacts/cv_bench/2026-09-27-coarse-to-fine-64token-pilot100-manifest.json`；原始JSONL `evaluation/results/cv_bench_refinement_budget64_pilot100_20260927.jsonl`与`evaluation/results/cv_bench_l2_budget64_pilot100_20260927/`；汇总脚本`evaluation/build_cv_bench_coarse_to_fine_report.py`；`/tmp/robobrain_harness_result.json`（1178字节）。
- [x] 合规：未跑完整2638，未跑Grounding/Pointing/PCK/IoU，未搜索新算法，未复用16-Token结果，未修改原始模型/`app/inference.py`/`api.py`，未删除文件，未kill进程，未访问封存数据或ZCU102。
- 决策：`COARSE_TO_FINE_64TOKEN_PILOT100_COMPLETE / GLOBAL_LOCAL_DOMINATED_BY_GLOBAL256_AND_L2_50_AT64 / PARETO_FRONTIER_DENSE_GLOBAL256`。
- 精确下一步（如继续）：对同5配置扩大到更多固定分层题或完整2638的64-Token复测以收窄区间；不得改Prompt或解析器。复现本轮两条命令：
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/zhaoxiangyu/conda-envs/robobrain25/bin/python evaluation/run_cv_bench_refinement.py --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --manifest datasets/cv_bench_screening100_budget64_diagnostic.jsonl --output evaluation/results/cv_bench_refinement_budget64_pilot100_20260927.jsonl --strategies dense global256 cosine_global_roi --repeats 3 --warmup 1
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/zhaoxiangyu/conda-envs/robobrain25/bin/python evaluation/run_hf_shallow_block_pruning.py --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --model-label RoboBrain2.5-4B-BF16 --manifest datasets/cv_bench_screening100_budget64_diagnostic.jsonl --output-dir evaluation/results/cv_bench_l2_budget64_pilot100_20260927 --retentions 1.0 0.5 0.25 --prune-after-layer 5 --anchor-grid 3 --selector l2_anchor --device cuda:0 --dtype bf16
```

## 2026-09-27 粗到细64-Token 100题判别实验（blocked）

- [x] 已读取`AGENTS.md`、`CURRENT_STATUS.md`、锁定manifest`datasets/cv_bench_screening100_budget64_diagnostic.jsonl`（100题，SHA256 `236e38a5…ec2e668`）、锁`configs/cv_bench_refinement_budget64_diagnostic.lock.json`及既有运行器`evaluation/run_cv_bench_refinement.py`、`evaluation/run_hf_shallow_block_pruning.py`。
- [!] 前置GPU检查未通过：`nvidia-smi`报`Failed to initialize NVML: Unknown Error`；robobrain25环境`torch.cuda.is_available()==False`（CUDA error 304）；`/dev/nvidiactl`、`/dev/nvidia0`、`/dev/nvidia-uvm`均`PermissionError [Errno 13]`。驱动`580.178.04`、GPU `NVIDIA RTX A6000`（只读`/proc`可读）。
- [!] `danger-full-access`提权被拒绝：`no approval channel is available`。按任务规则立即停止，未轮询、未抢占、未kill进程。
- [x] 未执行任何CUDA推理；未修改原始模型、`app/inference.py`、`api.py`；未删除文件；未访问封存数据、未访问ZCU102；未用16-Token旧结果冒充本次结果。
- 决策：`BLOCKED_GPU_ACCESS_UNAVAILABLE_IN_SANDBOX`。证据：`reports/cv_bench/2026-09-27-coarse-to-fine-64token-pilot100.{json,csv,md}`、`artifacts/cv_bench/2026-09-27-coarse-to-fine-64token-pilot100-manifest.json`、`/tmp/robobrain_harness_result.json`。
- 精确下一步（需在可访问GPU的会话执行，先重跑`nvidia-smi`）：见本报告MD中的两条原命令（refinement dense/global256/cosine_global_roi 与 pruning l2_anchor 1.0/0.5/0.25，manifest均为budget64 100题，`max_new_tokens=64`）。

更新时间：2026-09-27（粗到细64-Token判别实验因会话GPU不可达而blocked，无精度/时延结果）

## 2026-09-16 全局低清与局部细化复测（16 Token协议完成）

- [x] 用户明确暂用Jetson Xavier NX作为主AI算力平台；本轮仍在A6000验证算法，不访问NX或ZCU102。NX内存容量、JetPack和运行后端尚待设备审计。
- [x] GPU访问恢复：A6000仅75 MiB占用，GPU上只有gdm的Xorg/gnome-shell系统桌面进程，没有其他计算任务；不修改任何现有进程。
- [x] 审计旧Scout：学习式岭回归含Grounding/Pointing二元任务编码，不能无依据映射到CV-Bench四类题目。本轮只复用旧免训练cosine_fixed_box分支，未重训或调参；这不是旧岭回归结果的复现，也不是完整的自适应再次探索方案。
- [x] 新输入适配器、测试和预先锁定协议已保存：`evaluation/run_cv_bench_refinement.py`、`evaluation/tests/test_cv_refinement.py`、`configs/cv_bench_refinement_20260916.lock.json`。
- [x] 固定100题Dense/global256/cosine_global_roi，各预热1次、正式3次交错重复，共900条完成；Dense 300/300次与冻结输出逐字一致，各档输出/裁剪框无重复非确定性。模型端到端均值247.905/116.620/184.217 ms，相对Dense加速1.000/2.126/1.346倍；请求均值250.705/119.462/186.936 ms。平均输出Token 4.24/4.34/5.75，不把全部时间差归于图像计算。
- [x] 交错性能原始JSONL及汇总JSON/逐条CSV已保存：`evaluation/results/cv_bench_refinement_latency100_20260916.jsonl`及`reports/cv_bench/2026-09-16-refinement-latency100.{json,csv}`；源哈希保存为`2026-09-16-refinement-protocol-source-hashes.json`。
- [x] 全部2638题global256/cosine_global_roi两档各一次确定性准确率扫描，共5276条完成；不拿单次扫描作为正式时延。
- [x] 官方计分、配对差值区间、原始CSV/JSON及中文报告已保存：`reports/cv_bench/2026-09-16-global-local-refinement-full2638.{json,csv,md}`，逐条评分CSV共7914行、正式时延CSV 900行。
- [x] 完整集Dense/global256/cosine_global_roi得分87.2730%/81.3830%/72.8233%，相对分数保留100%/93.2510%/83.4431%；低清全图是有损加速基线，不因下降就否定可用性。
- [x] 当前两图方案相对仅低清全图下降8.5597个百分点，配对CI [-10.0197,-7.0679]个百分点，同时模型端到端更慢（184.217 vs 116.620 ms）；仅对当前16 Token具体配置判定被对照方案支配，不进入硬件开发。
- [x] 输出格式诊断已保存`reports/cv_bench/2026-09-16-refinement-format-diagnostic.json`。两图399题解析失败、400题触及16 Token上限；失败分布Count 167、Relation 23、Depth 1、Distance 208。原始失败文本可见未完成的解释；Dense正确转错418题中264题未解析、154题解析后答错。因此不能把整体退化全部归因于ROI信息无效，不能据此否定完整自适应细化思想或旧学习式Scout。
- [x] 下一项输出预算诊断已预先锁定`configs/cv_bench_refinement_budget64_diagnostic.lock.json`，派生完整/100题manifest只把max_new_tokens从16改为64，ID、图片、题目、选项、答案均不变；审计保存为`2026-09-16-budget64-derived-manifest-audit.json`。该诊断尚未开始推理，不是已有精度结果。
- [x] 本轮模型会话正常退出。最终GPU占用1131 MiB，其中新出现的1048 MiB计算进程PID 2131226归chenbowu，不是本轮模型；未修改或结束该进程。后续模型加载前必须重新检查GPU。
- [x] 最终验证88项测试通过；两项原协议续跑均pending=0，冻结协议/源代码哈希一致；完整JSON/CSV数量与ID覆盖校验通过，审计见`reports/cv_bench/2026-09-16-refinement-artifact-audit.json`。原始失败文本示例另存`2026-09-16-refinement-failure-examples.json`。
- 决策：`LOWRES_GLOBAL_USEFUL_BASELINE / COSINE_REFINEMENT_DOMINATED_AT16 / FORMAT_BUDGET_DIAGNOSTIC_REQUIRED / ADAPTIVE_IDEA_NOT_REJECTED / LEARNED_SCOUT_CVBENCH_NOT_TESTED / XAVIER_NX_DEVICE_AUDIT_PENDING / SELECTOR_RTL_HOLD`。

100题及完整集均已完成；以下原协议命令再执行应pending=0，不覆盖原始记录：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/zhaoxiangyu/conda-envs/robobrain25/bin/python evaluation/run_cv_bench_refinement.py --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --manifest datasets/cv_bench_screening100.jsonl --output evaluation/results/cv_bench_refinement_latency100_20260916.jsonl --strategies dense global256 cosine_global_roi --repeats 3 --warmup 1
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/zhaoxiangyu/conda-envs/robobrain25/bin/python evaluation/run_cv_bench_refinement.py --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --manifest datasets/cv_bench_full2638.jsonl --output evaluation/results/cv_bench_refinement_full2638_20260916.jsonl --strategies global256 cosine_global_roi --repeats 1 --warmup 1
```

下一条可执行命令：先在服务器终端核查`nvidia-smi`和进程归属；显存充足后，对同一固定100题做统一64 Token三档诊断，不调整Prompt或解析器：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/zhaoxiangyu/conda-envs/robobrain25/bin/python evaluation/run_cv_bench_refinement.py --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --manifest datasets/cv_bench_screening100_budget64_diagnostic.jsonl --output evaluation/results/cv_bench_refinement_budget64_latency100_20260916.jsonl --strategies dense global256 cosine_global_roi --repeats 3 --warmup 1
```

随后完整64 Token诊断应三档全部运行，另存`evaluation/results/cv_bench_refinement_budget64_full2638_20260916.jsonl`，预计7914条；不能只重测原失败题。现有16 Token报告器明确拒绝64 Token manifest，后续需新增或扩展预算诊断报告器，再做同预算配对评分。

更新时间：2026-09-16（细化方案16 Token完整基准与重复性能复测已完成，64 Token预算诊断待执行）  
状态：`L2_0875_FROZEN / PE16_BOARD_BASELINE_PASS / DATAFLOW_BOARD_PASS / GGML_Q8Q8_BRIDGE_CONTRACT_PASS / A53_NEON_Q8_QUANT_PASS / PL_RESIDENT_FULL_SYSTEM_ROUTE_PASS / Q8_A_MECHANICS_PASS_QUALITY_FAIL / Q8_B_MECHANICS_PASS_QUALITY_FAIL / Q8_EXPANDED_DIAGNOSTIC_COMPLETE / BF16_SAME_MASK_DIAGNOSTIC_COMPLETE / VISUAL_REFINEMENT_SPLIT_LOCK_PASS / GLOBAL_ROI_HF_INTERFACE_PASS / GLOBAL_ROI_Q8_INTERFACE_PASS_ACCURACY_UNVERIFIED / ROI_RESOLUTION_ACTION_MATRIX192_COMPLETE / P2_REPRESENTATION_CONDITIONALLY_SUPPORTED / NAIVE_REENCODE_BUDGET_FAIL / CHEAP_POLICY_CV_FAIL / VISION_PREFIX_DEPTH_1_3_6_CV_FAIL / TASK_RELEVANCE_POLICY_CV_FAIL / ROI_SCALE_CONTROLLER_BRANCH_STOP / ROI_L2_MATCHED_BUDGET192_COMPLETE / GEOMETRIC_SPARSE_DIAGNOSTIC_FAIL / LOW_BUDGET_POST_LAYER5_SPARSE_TOKEN_STOP / RECTANGULAR_ROI_LOCALIZATION_GATE_FAIL / LOWRES_SCOUT_BRANCH_STOP / TOKEN_REDUCTION_RESEARCH_PAUSED / SELECTOR_RTL_HOLD / BOARD_DATAFLOW_IDLE_UART_SHELL`

## 2026-09-16 CV-Bench完整评测

### 保留率曲线扩展（准确率与时延完成，定位诊断待补）

- [x] 用户授权固定同一layer-5 L2+3x3锚点策略，增加75%/50%/25%三档，全部使用2638题，不调整评分器或策略。
- [x] 预先保存`configs/cv_bench_l2_sweep_full2638.lock.json`，定位诊断仅用已公开96图开发记录，不访问封存96/512图。
- [x] 三档完整准确率、官方分数与配对置信区间完成，新增7914条推理记录。
- [x] 75%档2638题完成，组合分数`0.8702758235`，相对Dense下降0.2454个百分点、分数相对保留99.7188%；配对差值CI `[-0.0081058476,+0.0029585406]`，33个正确转错误、27个错误转正确。保留为曲线点，不称为严格无损或统计非劣。
- [x] 50%档2638题完成，组合分数`0.8465315882`，下降2.6198个百分点、分数相对保留96.9981%；CI `[-0.0356000829,-0.0173704778]`，121个正确转错误、48个错误转正确。存在整体退化证据，仍保留为质量—收益曲线点，不自行定义不可用门槛。
- [x] 25%档2638题完成，组合分数`0.7954859905`，下降7.7244个百分点、分数相对保留91.1492%；CI `[-0.0897425745,-0.0652782398]`，263个正确转错误、51个错误转正确。解析率98.9765%，不支持“主要是解析失败”作为总体退化解释。
- [x] 固定100题、5档配置、每题3次交错重复，共1500条时延记录完成；每档预热5次，重复输出与掩码全部一致。核心均值Dense/87.5%/75%/50%/25%分别为225.161/208.930/195.189/164.468/137.507 ms，对应加速1.000/1.078/1.154/1.369/1.637倍。只计推理核心，不含图片解码、processor和H2D。
- [x] 时延原始JSONL与汇总JSON/CSV已保存：`evaluation/results/cv_bench_l2_sweep_latency100_20260916.jsonl`及`reports/cv_bench/2026-09-16-l2-interleaved-latency100.{json,csv}`。
- [x] 完整集质量—计算—交错时延曲线及中文汇总已生成：`reports/cv_bench/2026-09-16-l2-retention-quality-latency-curve.{json,csv,md,svg}`，逐条评分CSV共13190行。
- [ ] Pointing/Grounding分档定位诊断未启动：此前沙箱外GPU检查审批两次超时，本轮GPU访问已恢复；本轮优先完成用户新讨论的全局/局部官方基准复测，这一独立定位诊断仍待补，不填估算数据。
- [x] 验证83项单元测试通过，报告JSON/CSV/SVG完成校验，冻结模型推理代码与manifest/baseline哈希保持一致。报告允许不带定位参数生成，但明确标记定位待完成。
- [x] 审计证据保存到`reports/cv_bench/2026-09-16-l2-retention-sweep-artifact-audit.json`：三档各2638条、交错时延1500条、曲线CSV 5行、逐条评分CSV 13190行；4项锁定哈希校验均通过。已完成GPU会话退出成功；由于最后GPU检查审批超时，本轮不能再次确认最终显存占用，不把沙箱报错当作服务器驱动故障。
- 本轮决策：`CV_BENCH_RETENTION_CURVE_COMPLETE / INTERLEAVED_CORE_LATENCY_COMPLETE / L2_0750_LOW_LOSS_CANDIDATE / L2_0500_0250_MEASURED_QUALITY_TRADEOFF / LOCALIZATION_GPU_ACCESS_PENDING / Q8_AND_SELECTOR_RTL_HOLD`。75%档可以继续验证；50%/25%有明确整体退化，不称为无损；CV-Bench不能替代精细定位、真实Q8与目标异构平台测量。
- [x] 新增交错时延运行器与曲线汇总器；复用冻结选择器，没有改动推理实现；新增顺序、续跑和组合公式测试通过，绘图使用标准库XML序列化SVG，未安装新依赖。

准确率续跑命令（已完成，再运行应pending=0）：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/zhaoxiangyu/conda-envs/robobrain25/bin/python evaluation/run_hf_shallow_block_pruning.py --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --model-label RoboBrain2.5-4B-BF16 --manifest datasets/cv_bench_full2638.jsonl --output-dir evaluation/results/cv_bench_l2_sweep_full2638_20260916 --retentions 0.75 0.5 0.25 --prune-after-layer 5 --anchor-grid 3 --selector l2_anchor --device cuda:0 --dtype bf16
```

交错时延已完成，再运行应跳过全部已有记录；复现或续跑命令：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/zhaoxiangyu/conda-envs/robobrain25/bin/python evaluation/run_l2_interleaved_latency.py --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --manifest datasets/cv_bench_screening100.jsonl --output-jsonl evaluation/results/cv_bench_l2_sweep_latency100_20260916.jsonl --retentions 1.0 0.875 0.75 0.5 0.25 --repeats 3 --warmup 5
```

本轮精确下一步：先在可访问GPU的服务器终端执行`nvidia-smi`及`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`，按PID用`ps -o user,pid,args -p <PID>`确认归属。GPU空闲且显存充足后，在本目录运行以下960条定位诊断；不读取封存数据，不进行策略调参：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/zhaoxiangyu/conda-envs/robobrain25/bin/python evaluation/run_hf_shallow_block_pruning.py --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --model-label RoboBrain2.5-4B-BF16 --manifest datasets/dac_spatial_eval_expanded.jsonl --output-dir evaluation/results/l2_retention_sweep_localization96_20260916 --retentions 1.0 0.875 0.75 0.5 0.25 --prune-after-layer 5 --anchor-grid 3 --selector l2_anchor --device cuda:0 --dtype bf16
```

其余下方条目为此前87.5%单档里程碑；旧顺序时延已由上面的交错复测补充，不应误读为仍未做重复计时。

- [x] 用户上传官方CV-Bench文件后完成完整性检查，2D 1438条、3D 1200条、共2638条。旧Hugging Face下载阻塞已通过本地数据解决。
- [x] 完成100条固定分层筛选，再扩大到官方完整2638条；冻结ID、原始图片字节哈希、parquet哈希、baseline哈希及策略参数，不在数据上调参。
- [x] 复用手机端最可信的`layer-5 L2@87.5% + 3x3区域锚点`真实实现；不使用激进34%裁剪、几何Oracle、ROI Scout或学习式残差头。
- [x] 全集完成原生BF16、同路径Dense、L2三组各2638条。原生`model.generate()`与自定义Dense输出2638/2638逐字一致。
- [x] 官方来源组合分数：BF16/Dense `0.8727298774`，L2 `0.8723349327`，下降0.0395个百分点；相对分数保留`0.9995474605`。配对分层bootstrap 5000次，差值95% CI `[-0.0045079013,+0.0037806757]`；没有预注册非劣门，不能宣称严格无损或统计非劣。
- [x] 正确转错误18条，错误转正确17条；答案一致率98.3321%。解析成功率98.4837%降至98.2563%，解析失败计为错误。
- [x] Count `0.6827 -> 0.6865`，Relation `0.9477 -> 0.9415`，Depth/Distance均不变；不得用总体分数掩盖Relation下降。
- [x] 实际Token平均减少12.3998%，Vision/Prefill MAC代理分别减少11.3211%/12.9234%。本次推理核心均值`220.769 -> 205.800 ms`（1.0727x），p50/p95 `156.847/463.975 -> 154.726/421.484 ms`；Vision/Prefill均值54.114/87.590降至48.408/78.118 ms。
- [!] 上述性能为A6000顺序运行、每题一次的观测，只含推理核心，不含H2D/预处理；未做交错多轮时延或功耗测量，不是ZCU102/目标异构平台实测。真实Q8_0同策略及其他四个官方空间基准尚未验证。
- [x] 校准96图与封存512图未访问，ZCU102未访问、复位或烧录，原模型与原应用未修改。
- [x] 最终验证：79项evaluation单元测试通过；两份完整CSV各2638行，JSON可解析，manifest与baseline锁哈希一致；GPU进程结束，显存释放至75 MiB。原始记录保持不覆盖，元数据抽样措辞勘误另存。
- 决策：`CV_BENCH_FULL2638_BF16_COMPLETE / RETAIN_L2_0875_SAFE_BASELINE / Q8_SAME_STRATEGY_PENDING / NOT_MAIN_ACCELERATION_SCHEME`。

证据与接续：

- `reports/cv_bench/2026-09-16-cv-bench-l2-0875-full2638.{json,csv,md}`
- `reports/cv_bench/cv_bench_bf16_full2638.{json,csv}`
- `evaluation/results/cv_bench_l2_0875_full2638_20260916/shallow_r1000.jsonl`
- `evaluation/results/cv_bench_l2_0875_full2638_20260916/shallow_r0875.jsonl`
- `configs/cv_bench_l2_0875_full2638.lock.json`
- `datasets/cv_bench_full2638.jsonl`
- `reports/cv_bench/README.md`
- `reports/cv_bench/2026-09-16-cv-bench-artifact-audit.json`
- `reports/cv_bench/2026-09-16-full2638-metadata-erratum.json`

精确下一步：先审计真实Q8_0视觉路径，确定第5层后L2 group gather、DeepStack压紧与原位置保留的插入点；先做100%身份门，再对同一2638题配对复验，不使用真值Oracle代替可部署策略。下一条只读命令：

```bash
rg -n 'early_keep|keep_indices|deepstack|for \(int|layer|qwen3vl' profiling_src/llama.cpp/tools/mtmd/models/qwen3vl.cpp
```

## 2026-09-16按文献通行方式复核

- [x] 已撤销自定义`E/G/Q`、0.75/0.25权重和统一质量门槛。`configs/evaluation_protocol.json`现要求按RoboBrain2.5原论文分别报告五个空间基准的官方分数及算术平均，通用VLM鲁棒性另表报告；同时报告Token/FLOPs、TTFT/TPOT、实测端到端延迟、功耗和能耗。
- [x] 对当前最可信的`layer-5 L2@87.5% + 3x3区域锚定`重新评分；只读取冻结原始输出，不重新选参、不访问封存的512图测试集。
- [x] 96张COCO图、192项任务的manifest及Full/L2输出哈希与2026-08-26锁定记录完全一致。
- [x] 相对Full，L2的PCK@0.1/PCK@0.2/Mean IoU/Acc@0.5保持率为`1.0182/1.0541/1.0047/1.0000`，格式成功率`1.0000`；PCK@0.1与Mean IoU的诊断性平均保持率为`1.0114`，按图像配对bootstrap 95%区间`[0.9537,1.0761]`。
- [x] 上述区间包含1.0，只能说明“轻微下降、无变化或轻微提高”均与现有数据相容；没有预注册非劣界限，不能宣称统计非劣或改善，也不能替代五个官方空间基准的主质量表。
- [x] Token减少12.5%，视觉编码MAC代理减少10.3%，LLM Prefill MAC代理减少12.7%；旧A6000记录的推理核心均值仅加速`1.007x`，不支持显著端到端收益。旧计时不含H2D，不能作为新协议的正式模型端到端结果。
- [!] 已尝试启动CV-Bench固定分层100题筛选；官方数据本地不存在，服务器直连Hugging Face超时，现有mihomo对Hugging Face TLS握手失败（其他公共站点可达），因此本轮没有伪造正式基准结果。代理选择已恢复`Auto`。
- [!] 决策：`RETAIN_AS_FROZEN_SAFE_BASELINE / NOT_MAIN_ACCELERATION_SCHEME`。L2@87.5%继续作为安全基线，但不能单独作为论文主加速方案。

新增证据：

- `evaluation/recheck_frozen_token_pruning.py`
- `reports/2026-09-16-l2-0875-new-standard-recheck.{json,md}`
- `configs/evaluation_protocol.json`

## 唯一工作目录

最新研究计划：`reports/2026-09-12-active-visual-refinement-plan.md`；最新执行结论：`reports/2026-09-15-task-conditioned-rectangular-roi-conclusion.md`。
按需视觉细化已经完成ROI分辨率矩阵、ROI/L2逐样本同预算对照、几何Oracle归因和低分辨率任务感知矩形提议器交叉验证。岭回归提议器的Grounding/Pointing中心召回仅0.849/0.828，低于0.95预设门槛；虽累计Vision MAC代理约0.43，但定位不可靠，因此没有运行完整VLM生成。当前低清Scout分支与约34%预算的post-layer5离散Token路线均停止，Token压缩研究暂停；L2@87.5%保留为轻度裁剪基线，Selector/ROI控制器RTL保持暂停。校准96图和测试512图继续封存。

异构平台选型之后的Token压缩工作已整理为阶段汇报：`reports/2026-09-16-heterogeneous-platform-token-pruning-stage-review.md`。该报告汇总研究演进、定量结果、积极成果、主要难点、尚未解决问题、可宣称边界及建议的12页汇报结构；不改变既有实验决策。

后续桌面端与手机 Remote Codex 都应在以下目录继续：

```text
/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102
```

`/home/zhaoxiangyu/RoboBrain2.5` 是较早的独立副本，不包含所有最新文件。不要在未做逐文件审计前覆盖或双向同步两个目录。

新会话开始时先读取：

1. `CURRENT_STATUS.md`
2. `docs/PL_ACCELERATION_RESEARCH_PLAN_2027.md`
3. `reports/2026-08-26-l2-residual-blind-validation.md`

## 2026-09-14按需视觉细化进展

- [x] 锁定COCO 2017全新800图：开发192、校准96、测试512；划分之间及与432张历史图像均零重叠。
- [x] BF16同掩码诊断完成：固定Token删除在HF BF16同样退化，排除“仅Q8量化/自定义后端导致”的解释。
- [x] 32图比例分层开发门控完成，覆盖Grounding和Pointing各32条、每条3次正式重复；所有重复输出一致。
- [x] Dense平均278.1视觉Token；ROI-only平均80.6，Token比0.290。Grounding Mean IoU `0.8111 -> 0.7989`，差值-0.0122，95% CI [-0.0657,+0.0518]；Pointing PCK@0.1 `0.7188 -> 0.6875`，差值-0.0312，95% CI [-0.1875,+0.1250]。结果是“证据不足”，不是精度通过。
- [x] 同次空闲A6000阶段计时显示ROI-only将Vision约`21 -> 10 ms`、Prefill约`47 -> 29 ms`，Decode基本不变；应用p50只下降约7%至11%，当前端到端仍由Decode主导。
- [x] ROI边距0.25/0.5/1.0网格完成。0.25和1.0的Pointing PCK@0.1均为0.6250，低于0.5的0.6875；没有证据支持继续调边距。边距运行遭遇不同GPU负载，跨运行延迟已从报告中禁用。
- [x] 真值ROI-only、`margin=0.5`、`long-edge=512`完成相同32图与三次重复：Token比0.516；Grounding Mean IoU相对Dense +0.0219，Pointing PCK@0.1相对Dense -0.0625。单纯提高ROI分辨率没有统一恢复精定位。
- [x] 384/512逐图真值动作上界完成：Grounding有23/32张偏好512；Pointing为14张偏好512、15张偏好384。Pointing one-shot oracle以0.389 Token比达到Dense PCK@0.1，但active escalation累计Token比为0.520；尚无不读真值的控制器。
- [x] 按`configs/roi_resolution_action_matrix192.lock.json`补齐完整192图开发集的Dense、ROI-384和ROI-512三列动作矩阵；三列各1152条，缺失与非确定性组均为0。
- [x] 192图固定策略结果：ROI-384 Token比0.342，Grounding/Pointing主指标差-0.0214/-0.0469；ROI-512 Token比0.609，主指标差+0.0061/-0.0208。均未同时满足Token不高于0.5和两任务非劣目标。
- [x] 192图逐图真值动作上界：Grounding one-shot Token比0.497、IoU 0.8452；Pointing Token比0.478、PCK@0.1 0.6615。朴素active escalation累计Token比升至0.696/0.653，判定预算失败。
- [x] 可部署特征尺度预测器已完成并失败，384/512控制器分支按预设规则停止；当前转入固定ROI/L2逐样本同预算对照。校准/测试集未访问。
- [x] 低成本策略消融完成：全图统计、ROI统计、Vision Prefix 1/3/6层均未同时通过动作准确率、成本和非劣门控。
- [x] Vision Prefix GPU边界实测中位数：1/3/6层分别2.804/3.553/4.638 ms；只含视觉前缀，不含预处理、LLM或目标平台通信。
- [x] 唯一预注册的6层Patch-Text空间相关性候选完成：Grounding/Pointing动作平衡准确率0.4973/0.5262，Vision工作比0.532/0.518，均失败。
- [!] 决策：当前384/512尺度控制器分支停止；不再在开发集调特征、深度、岭系数或阈值，不访问校准/测试集。下一步必须改变动作空间/监督目标，或补同预算固定ROI/L2对照。
- [x] 固定ROI/L2逐样本同预算对照完成。修正processor边界后，自定义Dense与原生Dense输出384/384逐字一致；L2与ROI逐样本Token一致，2条ROI目标超过Dense时按锁定规则回退Dense。
- [x] 最终Token比均约0.342；ROI/L2的Vision计算代理比分别0.309/0.482。L2相对Dense的Grounding IoU与Pointing PCK@0.1分别下降0.2780/0.3021，95%区间均完全为负，非劣门失败。
- [x] L2覆盖归因：目标中心仅保留150/384，全部目标相交网格完整保留10/384，目标相交网格总体保留率36.7%。
- [x] 同预算layer-5几何Oracle诊断完成：目标中心保留372/384、完整目标网格270/384、总体目标网格覆盖75.9%，但相对ROI-384仍下降Grounding 0.2179、Pointing 0.3281，95%区间均完全为负。
- [!] 决策：停止约34%预算的post-layer5离散Token裁剪，不进入Selector RTL；保留L2@87.5%轻度裁剪基线。下一候选改为低分辨率任务感知矩形提议器，先做分组交叉验证定位门，再决定是否运行预测ROI端到端推理。
- [x] 校准96图与测试512图均未访问；首次processor默认值不一致的结果作为审计证据保留，没有覆盖。
- [x] 低分辨率任务感知矩形提议器定位门完成：256px全图、视觉前6层、384条Scout前向，不调用4B文本生成；5折按图像分组交叉验证无同图泄漏。
- [x] 文本余弦固定框的Grounding/Pointing中心召回为0.510/0.474；岭回归热力图与尺度头提高到0.849/0.828，但仍低于0.95门槛，目标覆盖也仅0.735/0.729。
- [x] 两候选ROI Token比约0.41至0.43、累计Vision MAC代理约0.42至0.43，成本门通过；失败来自定位而非成本。按锁定规则没有运行完整VLM生成。
- [!] 决策：停止当前低清Scout分支，不在已观察开发集调参；Token压缩研究暂停，校准/测试继续封存，Selector与ROI控制器RTL保持暂停。

关键证据：

- `reports/2026-09-12-visual-refinement-split-lock.{json,md}`
- `reports/2026-09-12-hf-same-mask-expanded-diagnostic.{json,md}`
- `reports/2026-09-12-global-roi-bf16-representation-gate32-stratified.{json,md}`
- `reports/2026-09-12-global-roi-failure-attribution32-stratified.{json,md}`
- `reports/2026-09-14-roi-margin-grid32-quality.{json,md}`
- `reports/2026-09-14-roi-resolution512-gate32-quality.{json,md}`
- `reports/2026-09-14-roi-resolution-oracle32.{json,md}`
- `configs/roi_resolution_action_matrix192.lock.json`
- `reports/2026-09-14-roi-resolution-action-matrix192-quality.{json,md}`
- `reports/2026-09-14-roi-resolution-oracle192.{json,md}`
- `reports/2026-09-14-roi-resolution-action-matrix192-conclusion.md`
- `reports/2026-09-14-roi-resolution-policy-ablation.md`
- `reports/2026-09-14-roi-resolution-task-relevance-policy-cv192.{json,md}`
- `configs/roi_l2_matched_budget192.lock.json`
- `evaluation/run_hf_shallow_block_pruning.py`
- `evaluation/build_roi_l2_matched_report.py`
- `configs/roi_l2_matched_budget192_nativegrid_correction.lock.json`
- `configs/geometric_l2_roi384_matched_budget192.lock.json`
- `reports/2026-09-15-roi-l2-geometric-matched-budget192.{json,md}`
- `reports/2026-09-15-roi-l2-matched-failure-attribution192.{json,csv,md}`
- `reports/2026-09-15-low-budget-sparse-token-mechanism-conclusion.md`
- `configs/task_conditioned_rectangular_roi_cv.lock.json`
- `reports/2026-09-15-task-conditioned-rectangular-roi-cv192.{json,csv,md}`
- `reports/2026-09-15-task-conditioned-rectangular-roi-conclusion.md`
- `evaluation/results/global_roi_bf16_representation_gate32_stratified_20260912.jsonl`
- `evaluation/results/global_roi_bf16_roi_m0250_s384_stratified32_exactwarmup_20260914.jsonl`
- `evaluation/results/global_roi_bf16_roi_m1000_s384_stratified32_exactwarmup_20260914.jsonl`

## 已完成并冻结的结论

- ZCU102 PS端已完成RoboBrain2.5-4B Q8_0文字与图片推理验收。
- ZCU102 PL端统一Q8xQ8 GEMM v3已完成200 MHz板端数值、时序和性能验证，但尚未接入完整Transformer Block。
- 96图盲集上完成full、L2@87.5%和冻结L2残差头各192条，共576条推理。
- 冻结残差头能保持full点估计，但没有证明优于同预算L2；所有相对L2的95%置信区间跨0。
- 候选交互头在开发坐标NLL门控通过，但在开发校准真实输出上使Pointing PCK@0.1下降0.0625。
- 后验提出并预先锁定的“Grounding交互头、Pointing回退L2”路由，在全新96图测试上没有任何主指标改善。
- 决策：停止当前学习式联合删除路线；保留L2@87.5%为冻结Token基线，学习头不进入PL。

关键报告：

- `reports/2026-08-26-l2-residual-blind-validation.md`
- `reports/2026-08-26-l2-residual-blind-summary.json`
- `reports/2026-08-26-l2-residual-blind.lock.json`
- `reports/2026-08-25-zcu102-q8q8-output-burst-v3.md`
- `reports/2026-08-26-joint-deletion-coordinate-nll-pilot.md`
- `reports/2026-08-27-joint-deletion-scale-and-additive-selector.md`
- `reports/2026-08-27-candidate-interaction-calibration.md`
- `reports/2026-08-27-task-routing-test.md`

## 数据隔离规则

- `datasets/dac_spatial_l2res_blind.jsonl` 已经揭盲，只能用于复现已冻结结果，不能再用于选参、训练或下一算法设计。
- `datasets/dac_spatial_task_routing_test.jsonl` 已经揭示，同样只能用于复现冻结测试。
- 固定64/32图开发拆分及其真实输出校准结果也不得再用于宣称新方法的泛化改进。
- 新方法只有在开发集真实输出门控通过后才能冻结，并且需要另建从未观察的新测试集。
- 不得把模型原始输出当作唯一真值；Pointing和Grounding继续使用真实COCO标注及PCK/IoU指标。

## 联合删除阶段

目标：以多个视觉group联合删除后的真实坐标NLL构造集合级监督，检验是否能可靠超过
同预算L2@87.5%。

已完成并冻结：

1. 完整96图开发集完成192条任务记录、1344次候选前向；同图双任务共享相同候选掩码。
2. 113参数逐组可加头校准pairwise accuracy为0.4903，低于0.52门槛，路线停止。
3. 545参数候选交互头 `full_task_lambda_0p5` 在开发坐标NLL门控通过：
   pairwise accuracy 0.5374、top-1 regret 0.0288、相对L2平均NLL降低0.0101。
4. 开发校准真实输出中，交互头使Pointing PCK@0.1由0.6250降至0.5625，统一头停止。
5. 后验形成的任务路由在推理前锁定，并在与全部320张既有图片零重叠的新96图测试上运行。
6. 新测试Pointing与L2逐条文本和掩码96/96一致；Grounding有54/96条输出变化。
7. 路由相对L2：PCK@0.1和PCK@0.2均为0变化，Mean IoU下降0.0099，
   Acc@0.5下降0.0104；未满足“至少一个主指标严格改善”的预锁定条件。
8. 最终决策：`STOP_LEARNED_JOINT_DELETION / RETAIN_L2_0875 / PL_HOLD_FOR_LEARNED_HEADS`。

这些结果只否定当前特征、候选生成和小头结构，不等价于学习式Token选择普遍无效。
开发坐标NLL通过不能替代真实标注PCK/IoU证据。

## 当前硬件阶段

- 保留已通过ZCU102板端验证的统一Q8×Q8 output-burst v3作为参考设计。
- [x] 不覆盖v3；新建独立 `fpga_accel/q8q8_pe16_hls` 分支。
- [x] PE16 C testbench 7/7通过，最大绝对误差2.5034e-05。
- [x] 首次N-group优先遍历综合为II=2，冻结失败证据后改为M-row优先交错。
- [x] 第二次综合恢复II=1，估算周期3.884 ns（257.47 MHz）。
- [x] HLS资源：738 BRAM18K、400 DSP、31,056 FF、39,438 LUT，均低于器件80%门控。
- [x] 一阶排程投影：Prefill 1.591×、Vision 1.432×、Decoder 1.375×；
  这些不是PE16板端实测。
- [x] 导出独立PE16 IP：`robobrain.local:hls:q8q8_pe16:1.0`，
  SHA256=`c5a6bc869fc139f96343fa6a5885f04f5ac41b42cf0fc395d864c69cb822f81a`。
- [x] 独立Vivado完整系统通过200 MHz布局布线；最终bitstream重建结果：
  WNS=+0.067 ns、WHS=+0.011 ns、未布线网络=0、AXI输出宽度=128 bit。
- [x] 完整系统资源：25,592 LUT、29,406 FF、366.5 BRAM tile、656 DSP；
  分别占器件9.34%、5.36%、40.19%、26.03%。
- [x] Vivado vectorless功耗估算5.044 W（Medium confidence），不是板级实测。
- [x] 默认DRC无Error；613项Warning均为DSP流水化建议，144项Advisory为DSP属性建议。
  两个时钟方法学Advisory与PE8基线相同。
- [x] 生成的`compute_cached_tile`控制RTL完成定向XSim：
  16个PE bank各写24次，覆盖12个合法地址且每地址写2次。
- [!] Vitis HLS顶层AXI cosim因工具生成约30.74 GiB Verilog、`xelab` RSS采样约
  20.5 GiB而主动停止；不能记为RTL数值通过。全尺寸数值仍由C testbench 7/7覆盖。
- [x] 生成PE16 bitstream/XSA；独立bitstream与XSA内嵌bitstream哈希一致。
  bitstream SHA256=`d8ab6e8215d1ce41f320be8afad51aff5d4e7d90b4135e7994c90d9e1f810f65`。
- [x] 生成与PE16 XSA匹配的JTAG-only FSBL，保留SPD动态双Rank 16 GB DDR补丁；
  FSBL SHA256=`9280d4b92adb9aec54daeeae7681bb4206bb4e65d19c3ef24805bddd973a4efb`。
- [x] PE16寄存器映射10/10通过；主机运行时测试4/4、APM单位测试、数值基准自检及
  ASan运行时/基准测试均通过。
- [x] 生成独立版本化AArch64运行时与RAM Linux包：
  `deployment/artifacts/ram_linux_pe16_20260828`。内核UIO、16 GB设备树、
  64 MiB DMA保留区、initramfs载荷和网络配置完成离线一致性检查。
  Linux/BusyBox源码仅从旧独立副本按固定提交只读引用，没有同步或修改旧副本。
- [x] 最终PE16离线上板包门控通过：`Q8Q8_PE16_BUNDLE_PASS`；
  `board_programmed=no`。
- [x] 用户于2026-08-28明确授权后完成JTAG易失性启动；没有写入SD或QSPI。
- [x] 板端FSBL从SPD识别2 Rank、16 GB DDR；PL configured、PS–PL isolation/reset、
  AXI-Lite idle、Linux UIO全部通过。
- [x] 启动后与全部性能测试结束后各执行一次`M=2,N=4,K=32`自检，均为
  8输出完全一致、`max_abs=0`；三类大矩阵所有抽样同样`max_abs=0`。
- [x] 三次均值：Language prefill `242.808 ms / 105.027 GOPS`、Vision encoder
  `448.839 ms / 80.739 GOPS`、Decoder GEMV `14.647 ms / 3.401 GOPS`。
- [x] 相对v3 PE8板端基线分别加速`1.518× / 1.400× / 1.264×`；这仍是算子级
  microbenchmark，不是完整RoboBrain端到端加速比。
- [x] APM traffic/latency全部无溢出；Prefill/Vision继续判为compute/schedule-bound，
  Decoder继续判为memory/weight-stream-bound。
- [x] 测试后控制寄存器`0x00000004`（idle），UART shell保留；主机侧日志采集和
  `hw_server`已关闭，不影响板上RAM Linux继续运行。
- 当前板上运行的是DATAFLOW自检用RAM Linux并停在UART shell，不是RoboBrain模型或网页服务；
  PE16基线RAM Linux已被2026-08-30授权的JTAG易失性DATAFLOW启动替换。
- 当前不声称完整Transformer Block或模型端到端PE16收益，也不声称板级温度/功耗实测。

### 2026-08-29 PE16 DATAFLOW HLS里程碑

- [x] 保留已上板PE16基线，新建独立 `fpga_accel/q8q8_pe16_dataflow_hls` 分支；未访问、
  复位或烧录板卡。
- [x] 第一次参数式ping-pong写法通过C仿真，但Vitis HLS因指针别名无法证明bank独立而
  拒绝DATAFLOW综合；失败日志冻结为
  `artifacts/20260829T130834Z_vitis_hls_csynth.log`。
- [x] 改为本地FIFO连接的load/compute/store三级流式流水；进一步把每个M tile的
  1026周期累加器清零融合进首个K-block。
- [x] 最终C testbench 9/9通过，覆盖3个M tile及双N tile重用，最大绝对误差
  `3.05175781e-05`。
- [x] HLS明确综合出 `process_n_tile_dataflow` DATAFLOW区域；主MAC循环II=1，
  估算周期3.884 ns（257.47 MHz）。
- [x] HLS资源：816 BRAM18K（44.74%）、406 DSP（16.11%）、37,826 FF（6.90%）、
  42,110 LUT（15.36%），通过冻结资源门槛。
- [x] 保守排程模型相对PE16串行HLS模型：Language Prefill降低20.82%，Vision Encoder
  降低38.11%；均为综合模型，不是RTL或板端实测。Decoder继续只观察、不宣称加速。
- 决策：`DATAFLOW_HLS_GATE_PASS / DIRECTED_RTL_NEXT`。下一步对生成的融合首块初始化
  计算控制RTL做定向XSim；在RTL和布局布线门槛前不生成板端结论。

### 2026-08-29 PE16 DATAFLOW定向RTL里程碑

- [x] HLS生成的融合初始化计算控制RTL通过定向XSim：16个PE bank各写24次，覆盖
  12个合法地址，每个地址写2次；`kb=0`直接写路径包含在调度中。
- [x] 生成RTL静态结构检查确认三个独立process和完整连接：
  `load → 256/32-bit activation FIFO → compute → 128-bit output FIFO → store`。
- [x] 最终通过目录：
  `fpga_accel/q8q8_pe16_dataflow_hls/artifacts/compute_rtl_smoke/20260829T132655Z`。
- [!] 沙箱内XSim在时间仿真入口发生内部异常；完全相同的snapshot在沙箱外运行通过。
  早期缺依赖和工具异常目录全部保留，未覆盖。
- 证据边界：算术数值仍由9组C testbench覆盖；定向RTL只验证控制、地址和握手。
  已知会产生约30.7 GiB仿真模型的顶层AXI cosim未重试；当前没有bitstream或板端结果。
- 决策：`DIRECTED_RTL_AND_DATAFLOW_STRUCTURE_PASS / EXPORT_IP_NEXT`。

### 2026-08-29 PE16 DATAFLOW IP导出与OOC实现里程碑

- [x] 导出独立IP `robobrain.local:hls:q8q8_pe16_dataflow:1.0`，ZIP SHA256=
  `309a4bad50a4d377eedfa63a1a19fa9b1007a9ad81c9b355159868631306f723`。
- [x] ZCU102器件200 MHz OOC实现通过：WNS=+0.015 ns、WHS=+0.028 ns、
  TNS/THS=0、未布线网络=0。
- [x] 实现后资源：21,534 LUT、22,636 FF、661 DSP、838 BRAM18K等价资源；
  DSP与BRAM tile分别占器件26.23%和45.94%。
- [x] 默认DRC为0 Error；606项Warning和144项Advisory均为DSP实现建议或无负载提示。
- [!] 方法学1000项TIMING-18来自OOC外部AXI/控制端口没有I/O delay；不能替代完整
  PS+AXI系统的时序签核。
- [!] Vivado vectorless OOC功耗估算1.478 W（Medium confidence），不包含完整PS和互连，
  不是板级实测或论文能效结论。
- 本阶段未生成bitstream，未访问、复位或烧录板卡。
- 决策：`DATAFLOW_IP_OOC_ROUTE_PASS / FULL_SYSTEM_ROUTE_NEXT`。

### 2026-08-29 PE16 DATAFLOW完整系统BD验证里程碑

- [x] 新建独立 `fpga_accel/q8q8_pe16_dataflow_zcu102_vivado`，未覆盖已上板PE16分支。
- [x] 完整ZynqMP Block Design通过 `validate_bd_design` 和输出产品生成。
- [x] 保持HPC0承载WQ、HPC1承载XQ、HP0共享WS/XS/Y，输出AXI宽度仍为128 bit。
- [x] PS控制寄存器地址保持 `0xA0000000 / 64 KiB`，高低DDR窗口均已映射。
- [x] validate原始运行目录为
  `artifacts/runs/20260829T134918Z_validate`；Vivado日志SHA256=
  `41a635e139b94e043987b05962ac2c4c57295931cebda6afba546909e1e23705`。
- [!] AXI USER/LOCK宽度提示与既有完整系统拓扑同类；最终结论仍取决于综合、布局布线、
  DRC和时序门控。
- 决策：`DATAFLOW_SYSTEM_VALIDATE_PASS / FULL_SYSTEM_ROUTE_NEXT`。

### 2026-08-29 PE16 DATAFLOW完整系统布局布线里程碑

- [x] 完整PS+SmartConnect+PL系统通过200 MHz布局布线：WNS=+0.078 ns、
  WHS=+0.010 ns、TNS/THS=0、未布线网络=0。
- [x] 完整系统资源：25,966 LUT（9.47%）、30,473 FF（5.56%）、
  419 BRAM tile（45.94%）、661 DSP（26.23%），全部低于80%门槛。
- [x] 相对已上板PE16完整系统增加374 LUT、1,067 FF、52.5 BRAM tile和5 DSP；
  DATAFLOW的主要硬件代价是激活/输出FIFO的BRAM。
- [x] 新旧IP运行时寄存器头文件逐字节一致，SHA256均为
  `0810fe9a8a030fd2858dfd00bba2ea1ac7f3c9c1dcb6c30c2603934c36a0144d`。
- [x] 默认DRC为0 Error；方法学仅有与PE16基线相同的CLKC-40/CLKC-56 Advisory。
- [!] 完整系统Vivado vectorless估算4.684 W（Medium confidence）；不注入实测活动率，
  不宣称相对PE16的0.358 W估算差为功耗改进。
- [!] Language Prefill 198.942 ms与Vision Encoder 293.048 ms仍为一阶排程投影，
  不是DATAFLOW板端实测；Decoder继续不宣称加速。
- 原始运行目录：
  `fpga_accel/q8q8_pe16_dataflow_zcu102_vivado/artifacts/runs/20260829T135117Z_impl`。
- routed DCP SHA256=
  `8eda6c2deba32c331044a88009ff39bec6d3fd4f3b02b1d968b40248aa33efca`。
- 本阶段未生成bitstream，未访问、复位或烧录板卡。
- 决策：`DATAFLOW_FULL_SYSTEM_ROUTE_PASS / OFFLINE_BITSTREAM_NEXT`。

### 2026-08-30 PE16 DATAFLOW bitstream离线里程碑

- [x] 从已通过完整系统布局布线的独立工程生成bitstream和含bitstream的XSA。
- [x] bitstream SHA256=
  `c70d08a386d2a69b4319094107eea28e0591119ad1a86bfabf5598a806600036`；
  XSA内嵌bit与独立bit逐字节哈希一致。
- [x] 生成门控保留WNS=+0.078 ns、WHS=+0.010 ns、未布线网络=0，默认DRC为0 Error。
- [x] XSA包含27个成员及完整硬件描述、PS初始化文件和驱动元数据。
- [x] 原始运行目录：
  `fpga_accel/q8q8_pe16_dataflow_zcu102_vivado/artifacts/runs/20260829T141332Z_bitstream`。
- [x] 本阶段未访问、复位或配置ZCU102；当前板上PE16 RAM Linux和UART shell不变。
- 决策：`DATAFLOW_BITSTREAM_PASS / OFFLINE_PREBOARD_NEXT`。下一步生成与新XSA匹配且保留
  SPD动态双Rank 16 GB DDR补丁的JTAG-only FSBL，并验证运行时与启动包一致性。

### 2026-08-30 PE16 DATAFLOW离线上板包里程碑

- [x] 使用DATAFLOW XSA生成匹配的JTAG-only AArch64 FSBL；FSBL SHA256=
  `0f2c39fe97f121a31bef848996d460be71f174ae10b65dc28f05b90d8fe4481a`。
- [x] 保留SPD动态双Rank 16 GB DDR补丁；新旧`psu_init.c`及补丁后
  `xfsbl_ddr_init.c`均逐字节一致，关键启动/隔离/复位符号齐全。
- [x] DATAFLOW IP寄存器映射10/10通过；runtime 4/4、ASan/UBSan 4/4、benchmark普通/
  消毒器自检、APM单测和DT检查均通过。
- [x] 新编译AArch64 selftest/benchmark与已验证RAM Linux载荷逐字节一致；16 GB设备树、
  64 MiB DMA保留区、UIO和initramfs哈希门控通过。
- [x] JTAG脚本的四项路径覆盖和Tcl完整性通过；候选路径固定在
  `deployment/artifacts/q8q8_pe16_dataflow_jtag.env`。
- [x] 最终离线输出：`Q8Q8_PE16_DATAFLOW_BUNDLE_PASS`；板卡访问/配置均为`NO/NO`。
- 决策：`DATAFLOW_OFFLINE_PREBOARD_PASS / BOARD_AUTHORIZATION_REQUIRED`。板测会复位当前
  易失性RAM Linux并替换PL，必须取得明确授权后执行；不写SD/QSPI。

### 2026-08-30 任务无关描述符与DMA自动分块里程碑

- [x] 新增只包含`operation/layout/M/N/K`的PS工作负载描述符；不包含RoboBrain四种任务名，
  General/Pointing/Grounding/Trajectory继续共用同一PL ABI。
- [x] 实现GGML Q8_0交织块到硬件`INT8 payload + FP16 scale`双平面的逐位保持拆分。
- [x] 实现根据64 MiB UIO保留区、64-byte对齐和硬件上限自动选择最大M chunk，并生成
  多chunk `q8q8_job`；权重DDR缓冲无需重复打包，但当前PL仍会在每次调用重新读取权重。
- [x] 普通与ASan/UBSan测试各5/5通过，包含2-chunk模拟硬件端到端执行，12个输出
  `max_abs=0`；AArch64交叉编译通过。
- [x] 64 MiB下规划结果：视觉Prefill `976+104`，Vision Encoder `3584+736`，
  Decoder `1`。相对冻结512行策略分别从3/9/1次调用降至2/2/1次；尚未实测性能。
- [x] 新建`ram_linux_pe16_dataflow_20260830`并通过QEMU启动；QEMU不验证PL。
- 决策：`TASK_INDEPENDENT_DISPATCH_PASS / DATAFLOW_BOARD_MEASUREMENT_REQUIRED`。

### 2026-08-30 DATAFLOW板测协议与APM公式审计里程碑

- [x] 保留冻结`language_prefill M=512`作为PE8/PE16/DATAFLOW同形状kernel对比；新增独立
  `visual_prefill M=1080`用于固定`512+512+56`与自动`976+104`的同工作量A/B，避免混用口径。
- [x] 新AArch64 benchmark完成普通、ASan/UBSan、描述符5/5和DATAFLOW寄存器10/10测试；
  版本化RAM Linux包含完整视觉Prefill case和测量协议，QEMU启动通过但不作为PL证据。
- [x] 板端协议固定15阶段：协议前后自检、同形状kernel对比、视觉Prefill/Vision/Decoder
  固定与自动分块A/B、自动策略traffic/latency APM；每阶段后检查`ap_idle`并拒绝并发运行。
- [x] 协议dry-run与模拟硬件15/15 idle门控通过；UART PTY状态机只在新启动自检PASS后发送
  一次测量命令，日志拒绝覆盖；板测汇总器4项正/负测试通过。
- [x] 审计发现旧PE16汇总器把累计延迟除以平均每次事务数，导致平均延迟统一放大3倍；
  已追加机器可读修正并修复未来脚本。原始日志、最大延迟、流量、kernel性能与瓶颈结论不变。
- [x] 最新离线包再次输出`Q8Q8_PE16_DATAFLOW_BUNDLE_PASS`，明确
  `board_accessed=no board_programmed=no`；当前板上易失性PE16系统未改变。
- 决策：`DATAFLOW_MEASUREMENT_PROTOCOL_PASS / APM_LATENCY_FORMULA_CORRECTED /
  BOARD_AUTHORIZATION_REQUIRED`。

### 2026-08-30 GGML Q8_0到PL桥接契约里程碑

- [x] 新建项目自有`fpga_accel/ggml_q8q8_bridge`，遵守上游约束，未修改
  `profiling_src/llama.cpp`任何文件。
- [x] 固化GGML `src0[K,N] × src1[K,M] → dst[N,M]`到硬件
  `X[M,K] × W[N,K]^T`映射；二维连续Q8_0权重、Q8_0/FP32激活和FP32输出可分类。
- [x] batch、broadcast、view、布局/shape/类型/容量/硬件上限不满足时显式回退；8种负例通过。
- [x] 原生与ASan/UBSan各4/4通过，mock端到端12输出`max_abs=0`。
- [x] 当前上游Q8_0块布局为34 byte、scale偏移0、payload偏移2；FP32→Q8_0差分
  19行/152块逐字节一致。
- [x] 公共`ggml_tensor`适配器和两个AArch64测试程序交叉编译通过；尚未在板上执行。
- [x] 本阶段未访问、复位或配置板卡；不宣称完整llama.cpp接入、模型端到端精度或加速。
- 决策：`GGML_Q8Q8_BRIDGE_CONTRACT_PASS / PRODUCTION_HOOK_AND_COVERAGE_NEXT`。

关键证据：

- `reports/2026-08-30-ggml-q8q8-bridge.md`
- `fpga_accel/ggml_q8q8_bridge/artifacts/ggml_bridge_summary_20260830.json`
- `fpga_accel/ggml_q8q8_bridge/artifacts/ggml_bridge_20260830_final/`

### 2026-08-30 PE16 DATAFLOW板端里程碑

- [x] 用户明确授权后按冻结操作单执行JTAG易失性启动；没有写入SD或QSPI。
- [x] FSBL从SPD识别双Rank、16 GiB DDR；PL配置、PS–PL隔离/复位释放、AXI idle、
  U-Boot/ATF JTAG接管和Linux UIO全部通过。
- [x] 15阶段协议完整通过；每阶段后控制寄存器均为`0x00000004`，协议前后自检和全部
  benchmark抽样`max_abs=0`，APM overflow为0，UART shell保留。
- [x] 同形状对比：Language Prefill `174.545 ms / 146.102 GOPS / 1.391×`；
  Vision Encoder `272.232 ms / 133.117 GOPS / 1.649×`；Decoder
  `14.670 ms / 3.395 GOPS / 0.998×`。加速比相对冻结PE16板端基线。
- [x] 同工作量分块A/B：视觉Prefill `380.711→368.889 ms`（`1.032×`）；
  Vision Encoder `272.232→252.071 ms`（`1.080×`）；Decoder基本不变（`1.000×`）。
- [x] 自动分块APM实效带宽约为视觉Prefill `1.468 GB/s`、Vision Encoder
  `1.510 GB/s`、Decoder `1.835 GB/s`；OP/B分别约`99.34/95.24/1.85`，继续支持
  大矩阵compute/schedule-bound、Decoder memory/weight-stream-bound判断。
- [x] 汇总器初次因把可用内存写死为旧initramfs值而拒绝；修复为严格校验总内存
  `16776192 KiB`和合理可用范围，5项正/负单测通过，对同一原始日志完成汇总，未重跑板测。
- 决策：`DATAFLOW_BOARD_PASS / DENSE_LARGE_M_ACCELERATION_CONFIRMED /
  DECODER_GEMV_NO_GAIN / FULL_MODEL_BACKEND_INTEGRATION_NEXT`。

关键证据：

- `deployment/artifacts/20260830T113814Z-q8q8-pe16-dataflow-board-summary.json`
- `reports/20260830T113814Z-zcu102-q8q8-pe16-dataflow-board.md`
- `reports/20260830T113814Z-zcu102-q8q8-pe16-dataflow-board-uart.log`
- `reports/20260830T113814Z-zcu102-q8q8-pe16-dataflow-board-xsct.log`

### 2026-08-30 GGML真实覆盖率与板端准备开销里程碑

- [x] 使用既有Grounding逐算子日志做只读覆盖率审计，未读取或使用盲集/验证集精度数据。
  合同可命中调用数：decode `252/253`、KV cache `249/253`、text prefill `252/253`、
  vision encode `104/106`、vision prefill `747/759`。
- [x] 可命中部分占各阶段MUL_MAT时间分别为decode `90.43%`、KV cache近`100%`、
  text prefill `99.77%`、vision encode `99.30%`、vision prefill近`100%`；这只证明shape/类型
  覆盖，不代表所有命中都值得下沉。M=1 decoder按板测结果强制保留CPU路径。
- [x] 通过UART把独立AArch64准备开销基准写入板端`/tmp`运行，二进制SHA256回读一致；
  没有复位/重配PL，没有写SD或QSPI。标量FP32→Q8_0在M=512/1080/4320分别为
  `31.758/66.970/107.172 ms`。
- [x] 新增AArch64 NEON量化和最多4个row worker；上游152块差分、half-way舍入、NaN回退、
  原生/ASan测试通过。板端六个真实尺度中，标量、NEON单核、NEON四核全部输出逐字节一致。
- [x] 板端NEON单核约`3.23–3.37×`；四核在M=512/1080/4320为
  `11.76×/12.23×/12.40×`，时间降至`2.701/5.474/8.644 ms`，只占同shape PL kernel
  `1.55%/1.48%/3.43%`。激活量化不再是主要准备瓶颈。
- [x] 权重拆分仍是关键问题：`N=9728,K=2560`在cacheable DDR约`145–148 ms`，既有真实
  UIO权重准备约`354 ms`。生产路径禁止每次调用重复拆分；下一阶段比较受控预打包驻留与
  PL直接消费原生34-byte GGML Q8_0两条路线。
- [!] 极简RAM Linux缺少动态`libpthread.so.0`，首个动态测试程序未进入benchmark；失败日志
  保留。改用自包含静态测试程序后通过，未修改initramfs或板端持久介质。
- 决策：`GGML_PROFILE_COVERAGE_PASS / A53_NEON_PARALLEL_QUANTIZATION_PASS /
  WEIGHT_REPACK_PER_CALL_REJECTED / NATIVE_OR_PREPACKED_WEIGHT_PATH_NEXT`。

关键证据：

- `reports/2026-08-30-ggml-q8q8-profile-coverage.md`
- `fpga_accel/ggml_q8q8_bridge/artifacts/grounding_profile_coverage_20260830.json`
- `reports/2026-08-30-zcu102-ggml-q8q8-prepare-benchmark.md`
- `fpga_accel/ggml_q8q8_bridge/artifacts/prepare_benchmark_20260830/board_summary.json`
- `reports/2026-08-30-zcu102-ggml-q8q8-neon-quantization.md`
- `fpga_accel/ggml_q8q8_bridge/artifacts/prepare_benchmark_neon_numeric_20260830/board_summary.json`
- `reports/20260830T123000Z-zcu102-ggml-neon-numeric-benchmark-uart.log`
- `reports/20260830T121500Z-zcu102-ggml-neon-prepare-benchmark-uart.log`（动态pthread失败证据）

### 2026-08-30 快速UIO数据路径与Block驻留决策里程碑

- [x] 新增NEON/多线程Q8_0双平面拆分；原生与ASan测试全量一致。板端三种权重shape在
  cacheable DDR和真实非缓存UIO均逐字节通过，且基准未访问PL控制寄存器。
- [x] UIO四核权重staging：6.96/26.46/4.46 MB分别为`3.207/12.038/2.188 ms`，
  相对旧逐字节路径为`18.28×/18.50×/17.24×`。对大M，权重stage+激活量化合计约为
  当前kernel的`4.30–8.44%`，可先进入完整Block bring-up；原生Q8_0 PL前端降为后续项。
- [x] 新增UIO→cacheable DDR NEON/多线程回读并做三种输出全量对照。19.92/37.98/58.72 MB
  四核回读仍需`48.957/83.899/150.351 ms`，虽相对逐字节快`47.16–52.47×`，带宽仅
  `0.39–0.45 GB/s`。
- [x] 新增默认关闭的`rb_ggml_q8q8_execute_fast` bring-up入口，统一使用快速量化、权重/
  activation拆分和输出回读；原生与ASan mock参考/快速双路径均`max_abs=0`。它仍是逐算子
  PS往返，不是最终生产后端。
- [x] 架构决策：完整Block不能把每个`MUL_MAT`结果都回PS。下一阶段先实现LLM SwiGLU FFN和
  Vision GELU FFN的PL驻留子图，以整个子图的buffer live range选择64 MiB内的chunk，增加
  Norm、SiLU/GELU、逐元素乘、残差和F32→Q8_0向量操作。
- [x] 第一版64-byte对齐活跃区间规划器和属性测试通过；同时存活buffer物理重叠为0。
  64 MiB最大安全chunk为LLM SwiGLU `432`（1080行3块、峰值65,678,848 B）和Vision GELU
  `2896`（4320行2块、峰值66,917,376 B）。旧单GEMM `976/3584`不能直接用于完整FFN。
- 决策：`FAST_UIO_WEIGHT_STAGING_PASS / FAST_UIO_READBACK_PASS /
  PER_OP_PS_ROUNDTRIP_REJECTED_AS_FINAL_ARCHITECTURE / PL_RESIDENT_FFN_NEXT`。

关键证据：

- `reports/2026-08-30-zcu102-ggml-q8q8-weight-path.md`
- `fpga_accel/ggml_q8q8_bridge/artifacts/weight_path_benchmark_20260830/board_summary.json`
- `reports/2026-08-30-zcu102-ggml-q8q8-output-path.md`
- `fpga_accel/ggml_q8q8_bridge/artifacts/output_path_benchmark_20260830/board_summary.json`
- `fpga_accel/ggml_q8q8_bridge/artifacts/ggml_fast_execute_20260830/`
- `docs/PL_RESIDENT_TRANSFORMER_BLOCK_ARCHITECTURE.md`
- `reports/2026-08-30-pl-resident-ffn-live-range-planner.md`

关键报告：

- `reports/2026-08-27-zcu102-q8q8-pe16-attempt1.md`
- `reports/2026-08-27-zcu102-q8q8-pe16-hls.md`
- `reports/2026-08-27-zcu102-q8q8-pe16-rtl-validation.md`
- `reports/2026-08-27-zcu102-q8q8-pe16-vivado.md`
- `reports/2026-08-28-zcu102-q8q8-pe16-bitstream.md`
- `reports/2026-08-28-zcu102-q8q8-pe16-preboard.md`
- `reports/2026-08-28-zcu102-q8q8-pe16-board.md`
- `reports/2026-08-28-zcu102-q8q8-pe16-board-uart.log`
- `reports/2026-08-28-zcu102-q8q8-pe16-board-xsct.log`
- `fpga_accel/q8q8_pe16_hls/artifacts/pe16_hls_summary.json`
- `fpga_accel/q8q8_pe16_hls/artifacts/pe16_rtl_validation_summary.json`
- `fpga_accel/q8q8_pe16_zcu102_vivado/artifacts/pe16_vivado_summary.json`
- `fpga_accel/q8q8_pe16_zcu102_vivado/artifacts/pe16_bitstream_summary.json`
- `deployment/artifacts/q8q8_pe16_preboard_summary.json`
- `deployment/artifacts/q8q8_pe16_board_summary.json`
- `reports/2026-08-29-zcu102-q8q8-pe16-dataflow-hls.md`
- `fpga_accel/q8q8_pe16_dataflow_hls/artifacts/pe16_dataflow_hls_summary.json`
- `reports/2026-08-29-zcu102-q8q8-pe16-dataflow-rtl.md`
- `fpga_accel/q8q8_pe16_dataflow_hls/artifacts/pe16_dataflow_rtl_summary.json`
- `reports/2026-08-29-zcu102-q8q8-pe16-dataflow-ip-export.md`
- `fpga_accel/q8q8_pe16_dataflow_hls/artifacts/pe16_dataflow_ip_export_summary.json`
- `reports/2026-08-29-zcu102-q8q8-pe16-dataflow-vivado.md`
- `fpga_accel/q8q8_pe16_dataflow_zcu102_vivado/artifacts/pe16_dataflow_vivado_summary.json`
- `reports/2026-08-30-zcu102-q8q8-pe16-dataflow-bitstream.md`
- `fpga_accel/q8q8_pe16_dataflow_zcu102_vivado/artifacts/pe16_dataflow_bitstream_summary.json`
- `reports/2026-08-30-zcu102-q8q8-pe16-dataflow-preboard.md`
- `deployment/artifacts/q8q8_pe16_dataflow_preboard_summary.json`
- `fpga_accel/q8q8_runtime/artifacts/dataflow_dispatch_20260830/dispatch_unit_test.log`
- `deployment/reports/2026-08-30-ram-linux-pe16-dataflow-qemu.log`
- `deployment/reports/2026-08-30-ram-linux-pe16-dataflow-measurement-qemu.log`
- `docs/DATAFLOW_BOARD_MEASUREMENT_RUNBOOK.md`
- `reports/2026-08-30-zcu102-q8q8-pe16-apm-latency-correction.md`
- `deployment/artifacts/q8q8_pe16_apm_latency_correction.json`

DATAFLOW上板阶段已经完成，NEON基准后板卡仍停在DATAFLOW RAM Linux UART shell，PL idle。
当前不需要再次复位或配置板卡；安全接续命令重放冻结门并验证PL驻留FFN规划器：

```bash
cd /home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102
python3 deployment/scripts/verify_q8q8_pe16_dataflow_board.py \
  --summary deployment/artifacts/20260830T113814Z-q8q8-pe16-dataflow-board-summary.json \
  --markdown reports/20260830T113814Z-zcu102-q8q8-pe16-dataflow-board.md
make -C fpga_accel/ggml_q8q8_bridge all ARTIFACTS=artifacts/ggml_bridge_<new_run_id>
python3 fpga_accel/ggml_q8q8_bridge/tests/test_profile_coverage.py
python3 fpga_accel/pl_resident_ffn/test_live_range_planner.py
```

### 2026-08-31 DATAFLOW 板端稳定性复测

- [x] 用户明确授权后，直接从既有 DATAFLOW RAM Linux UART shell 重跑冻结板端协议；未复位、
  未重新配置 PL、未写 SD/QSPI。
- [x] 协议前后自检与全部抽样均为 `max_abs=0`，APM 无溢出，最终控制寄存器
  `0x00000004`（idle）。
- [x] 本次板测：Language M512 `174.559 ms / 146.090 GOPS`、Visual Prefill M1080 auto
  `368.879 ms / 145.826 GOPS`、Vision Encoder M4320 auto `252.063 ms / 143.769 GOPS`、
  Decoder M1 `14.660 ms / 3.398 GOPS`。
- [x] 相对 2026-08-30 的四项延迟差异均小于 `0.06%`；大 M 矩阵核吞吐假设稳定。
- [!] 用本次吞吐更新的一阶整图投影为 `538.727 s / 8.979 min`。这不是端到端板测：完整
  GGML/llama.cpp Transformer 后端尚未连接 PL，小 M/转置 shape、完整调度和驻留 Block
  仍会带来误差。
- 决策：`BOARD_RERUN_PASS / KERNEL_ASSUMPTION_STABLE / END_TO_END_NOT_MEASURED`。

关键证据：

- `reports/20260831T-board-rerun-uart.log`
- `reports/2026-08-31-zcu102-dataflow-rerun.json`
- `reports/2026-08-31-zcu102-dataflow-rerun.md`

精确下一命令（继续 PL 驻留 FFN，而不是宣称整图已实测）：

```bash
python3 fpga_accel/pl_resident_ffn/test_live_range_planner.py
```

### 2026-08-31 PL驻留FFN统一向量IP里程碑

- [x] 新建独立`fpga_accel/pl_resident_ffn_hls`，未覆盖已上板DATAFLOW分支，未访问板卡、
  未生成系统bitstream、未修改SD卡。
- [x] 一个任务无关模式核实现RMSNorm/LayerNorm、SiLU门控、GELU、残差加和Q8_0重量化；
  7个AXI主口压缩为3个，保留双输入读与结果写并发。
- [x] 单算子、零方差、非法参数及两类完整FFN软件组合共9/9通过；LLM/Vision组合相对独立
  FP32参考的最大误差分别为0.000711/0.002926。
- [x] HLS全部循环约束满足，估算Fmax 253.49 MHz；FP32统计反馈显式`II=5`，共享参数
  预载`II=2`，不再以高资源换取非关键孤立吞吐。
- [x] 独立IP布局布线通过：周期4.198 ns（约238.21 MHz），WNS=+0.802 ns、
  WHS=+0.023 ns、未布线0；26,685 LUT、47,063 FF、154 DSP、64 BRAM Tile。
- [x] IP SHA256=`0a8959b1ea6478f228608fe20df4deb3fe6fc566bd5e8b10d19f88095b786ba0`。
- [ ] 当前组合测试使用确定性小Q8矩阵；尚未通过冻结RoboBrain真实张量精度门。
- [ ] 当前是OOC结果；DATAFLOW GEMM+向量核完整ZynqMP系统的SmartConnect开销和时序尚未
  实现，因此不宣称Block或模型端到端加速。
- 决策：`PL_RESIDENT_VECTOR_IP_ROUTE_PASS / FULL_SYSTEM_INTEGRATION_NEXT`。

关键证据：

- `fpga_accel/pl_resident_ffn_hls/artifacts/pl_resident_ffn_vector_summary.json`
- `fpga_accel/pl_resident_ffn_hls/artifacts/20260830T165327Z_vitis_hls_export_impl.log`
- `reports/2026-08-31-pl-resident-ffn-vector-ip.md`

精确下一命令（创建并验证双IP完整系统后执行）：

```bash
make -C fpga_accel/pl_resident_ffn_zcu102_vivado validate
```

### 2026-08-31 PL驻留FFN双IP完整系统里程碑

- [x] 新建独立`fpga_accel/pl_resident_ffn_zcu102_vivado`，把既有PE16 DATAFLOW GEMM和
  统一FFN向量核接入同一ZynqMP PS+AXI+PL系统；未覆盖已上板分支。
- [x] 固定GEMM/向量核AXI-Lite地址为`0xA0000000`/`0xA0010000`；BD验证通过。
- [x] 200 MHz完整布局布线通过：WNS=+0.060 ns、WHS=+0.009 ns、TNS/THS=0、
  未布线网络=0，DRC与Methodology Error均为0。
- [x] 完整系统资源：58,445 LUT（21.32%）、88,156 FF（16.08%）、483 BRAM Tile
  （52.96%）、815 DSP（32.34%），全部低于80%门控。
- [x] Vivado vectorless估算5.942 W（Medium confidence）；仅作设计审计，不是板级实测，
  不支持低功耗或能效结论。
- [ ] 当前两个IP仍由PS顺序启动；尚无PL命令队列或自主完整FFN控制器。
- [ ] 冻结RoboBrain真实FFN张量数值门未完成，因此没有生成bitstream、没有访问/配置板卡、
  没有修改SD/QSPI，也不宣称Block或模型端到端加速。
- 决策：`PL_RESIDENT_FULL_SYSTEM_ROUTE_PASS / REAL_TENSOR_NUMERIC_GATE_NEXT`。

关键证据：

- `fpga_accel/pl_resident_ffn_zcu102_vivado/artifacts/pl_resident_ffn_system_summary.json`
- `fpga_accel/pl_resident_ffn_zcu102_vivado/artifacts/runs/20260830T171212Z_impl`
- `reports/2026-08-31-pl-resident-ffn-full-system.md`

精确下一命令（只读解析模型并冻结真实张量测试向量；不加载CUDA）：

```bash
python3 fpga_accel/pl_resident_ffn_hls/scripts/export_real_tensor_case.py --help
```

### 2026-08-31 RoboBrain真实Q8权重FFN代理数值门

- [x] 只读解析旧独立部署目录中的已验哈希Q8_0 GGUF；没有复制/覆盖旧项目，没有加载CUDA。
- [x] 使用第0层完整`ffn_norm/gate/up/down`真实权重与4个真实token embedding代理输入，
  计算完整2560→9728→2560 FFN候选/FP32参考差分。
- [x] 完整代理用例：max abs=0.03017、RMSE=0.002165、relative L2=1.235%，通过预设
  8%门槛。
- [x] 冻结2560→64→64真实权重切片并接入当前HLS C源码；原9项回归+新增切片10/10通过，
  切片max abs=0.001716、RMSE=0.000346。
- [!] 输入是模型真实embedding派生的代理激活，不是运行时FFN隐藏状态；不得写成“真实激活
  全层精度已通过”，也不证明端到端模型精度。
- [x] 未生成bitstream、未访问板卡、未修改SD/QSPI。
- 决策：`REAL_WEIGHT_PROXY_NUMERIC_PASS / RUNTIME_ACTIVATION_CAPTURE_NEXT`。

关键证据：

- `fpga_accel/pl_resident_ffn_hls/artifacts/real_weight_proxy_20260831_v2/summary.json`
- `fpga_accel/pl_resident_ffn_hls/artifacts/real_weight_proxy_20260831_v2/real_weight_proxy_slice.bin`
- `fpga_accel/pl_resident_ffn_hls/artifacts/20260830T174858Z_vitis_hls_csim.log`
- `reports/2026-08-31-real-weight-proxy-numeric.md`

精确下一命令（先构建CPU运行时FFN输入捕获钩子测试，不加载CUDA）：

```bash
make -C fpga_accel/runtime_activation_capture test
```

### 2026-09-01 机器人计算平台与FPGA路线调研文档

- [x] 完成现有机器人GPU/NPU/封闭车规平台对RoboBrain2.5-4B的部署适用性分析，单列
  Jetson AGX Thor规格、官方并发VLM基准及其不可直接等同于本模型单请求性能的边界。
- [x] 按“公开板载平台/仅合作或评估但主控未披露”区分国内外人形机器人案例，未根据训练
  平台或厂商合作关系反推量产机器人的板载芯片。
- [x] 区分单板FPGA低功耗路线和GPU/NPU+FPGA异构路线；核对Hummingbird、DATE 2025、
  ReCoVLM和NANOMIND的模型、平台、吞吐、带宽利用率及任务边界。
- [x] 汇总本项目已上板DATAFLOW GEMM与尚未上板GEMM+FFN双IP的两套资源口径，保留
  算子级板测、整图一阶投影和端到端未完成之间的证据边界。
- 验证：文档人工回读完成；外部参数链接到厂商官方页面、正式论文或论文预印本；未访问板卡、
  未加载CUDA、未修改SD/QSPI。当前目录不是Git工作树，因此无法执行`git diff --check`。
- 决策：短期继续真实运行时FFN激活数值门；获得端到端结果后，再根据吞吐和板级功耗选择
  单板ZCU102低功耗路线或Jetson+FPGA异构路线。

关键证据：

- `docs/ROBOT_COMPUTE_PLATFORM_AND_FPGA_ROLE_SURVEY.md`
- `reports/2026-08-31-zcu102-dataflow-rerun.md`
- `fpga_accel/pl_resident_ffn_zcu102_vivado/artifacts/pl_resident_ffn_system_summary.json`

精确下一命令：

```bash
make -C fpga_accel/runtime_activation_capture test
```

### 2026-09-03 边缘AI平台全档位选型补充

- [x] 将实验室已有原始Ascend 310、用户图片中的Orange Pi AIpro 8/20T纳入平台选择。
- [x] 按请求准备、图像解码/预处理、Patch Embedding、Vision Attention/FFN、
  DeepStack/Projector、视觉与文本Prefill、KV Cache、Decode、采样和实时控制逐阶段分析。
- [x] 建立从RDK X5、RK3588和低功耗NPU，到Jetson Orin/Thor、嵌入式dGPU、Atlas 300I
  Duo、DGX Spark、RTX A6000及Versal VEK280的统一Markdown总表。
- [x] 成本采用“已有/A–E/询价”分级，并保留NVIDIA与AMD公开美元价；没有把电源额定功率写成
  实测板级功耗，也没有跨精度直接比较TOPS。
- [x] 以4.274 GB/token权重流计算有公开带宽平台的Decode理想上限；明确这些值不是实测吞吐。
- [!] 原始Ascend 310的具体Atlas型号、板载内存和CANN版本未知；Orange Pi商品截图的1094.09元
  是8/20 TOPS、8/16 GB混合商品起售价，不能据此锁定实验室实物SKU。
- [!] 当前llama.cpp CANN已验证硬件主要为310P/910B，310P的Q8_0量化`MUL_MAT`仍非完整
  支持；原始310和310B均需先做设备与算子兼容门。
- 验证：Markdown人工回读、项目内证据路径检查及外部厂商/上游文档复核通过；未加载CUDA、
  未访问或编程ZCU102、未修改SD/QSPI。
- 决策：已有A6000先建立统一高性能基线；两套昇腾设备先做零成本兼容性验证；新增机器人主机
  的平衡选择为AGX Orin 64 GB，最高性能选择为Thor；ZCU102继续作为定制低功耗/确定性子图。

关键证据：

- `docs/EDGE_AI_PLATFORM_TOTAL_COMPARISON_2026.md`
- `reports/2026-08-22-zcu102-operator-bottleneck.md`
- `fpga_accel/heterogeneous_partition/artifacts/partition_gate_20260901/report.md`
- `third_party/llama.cpp/docs/backend/CANN.md`

精确下一命令（在实验室Ascend设备终端确认实际型号）：

```bash
npu-smi info
```

### 2026-09-03 ZCU102异构通信门控补充

- [x] 将平台选型从“算力/内存/功耗”扩展为“计算平台+切分边界+物理链路”联合筛选。
- [x] 以5 Hz、通信占周期20%为严格门槛，冻结40 ms单向预算；按既有模型合同计算1/2.5/5/10GbE、
  USB 3.0及PCIe Gen2×4规划速率下的视觉隐藏状态和KV Cache传输下界。
- [x] 通信门结论：Vision Merger边界需要至少2.5GbE或USB 3.0；1GbE只允许Q8视觉状态过渡验证；
  每帧跨板迁移完整KV Cache在当前ZCU102可实现链路上全部淘汰。
- [x] 核对ZCU102 P1为PS PCIe Gen2×4 Root Port，不能与同为Root Port的Jetson开发套件PCIe插槽
  直接互连；可将P1用于安装10GbE NIC，或在PL SFP+上另建10GbE MAC/PCS/DMA。
- [x] 平台通信筛选后，首选冻结为`ZCU102 + AGX Orin 64 GB + 10GbE`，高预算候选为
  `ZCU102 + Thor + 5/10GbE`，实验室算法基线为`ZCU102 + A6000服务器 + 10GbE`。
- [x] 默认任务划分修正为：主AI平台保留Vision Encoder、全部Prefill、KV Cache和Decode；ZCU102
  承担传感器/ROI/Token削减、动作头、安全约束和确定性I/O。停止主平台Prefill、ZCU102 Decode。
- [!] 10GbE、USB 3.0 Device及任何新链路均为规划值，尚未在当前板卡上实测；不得将传输下界写成
  已有吞吐。QSFP28与ZCU102 SFP+的速率/模块兼容也未验证。
- 验证：Markdown表格列数检查通过；内部模型数据路径与厂商接口资料复核完成；未访问或编程ZCU102，
  未修改SD/QSPI，未加载CUDA。

关键证据：

- `docs/EDGE_AI_PLATFORM_TOTAL_COMPARISON_2026.md`
- `fpga_accel/heterogeneous_partition/artifacts/partition_gate_20260901/report.md`
- AMD ZCU102 User Guide UG1182
- NVIDIA Jetson AGX Orin Developer Kit Guide

当时下一命令（已由2026-09-04路线调整取代，仅保留历史）：

```bash
make -C fpga_accel/runtime_activation_capture test
```

### 2026-09-03 异构通信论文与GTH/SFP+报告补充

- [x] 在四节式主调研报告和平台综合报告中补充ZCU102的GTH、SFP+、Ethernet MAC/PCS、DMA与AXI4-Stream层次，明确GTH只是PL侧SerDes，不等同于可直接使用的Linux网卡。
- [x] 核对Quad 230的4条GTH到SFP0～SFP3连接，并加入AMD已验证的1×10GbE/1×40GbE参考架构；40G由4×10.3125 Gb/s通道经MAC/PCS聚合，不是插入4根线即可自动形成。
- [x] 增加GLITCHES、HPU、INF²、HillInfer和ReCoVLM的异构切分、物理链路、传输内容及通信隐藏方法；公开资料没有说明ReCoVLM物理接口，因此保持未知，不推断为PCIe或以太网。
- [x] 将PL 40GbE按4.0 GB/s规划有效载荷加入通信下界。85.32 MiB Q8 KV线路下界为22.4 ms，可进入40 ms预算，但ZCU102 Decode带宽和PS中转仍不形成系统收益，默认切分不变。
- [x] 冻结当前工程解释：相关服务器论文主要使用PCIe DMA；本项目采用10/40GbE是因为ZCU102与Jetson是两块独立开发板。正式数据面应走PL AXI4-Stream→MAC/DMA，KV继续驻留主AI平台。
- 验证：Markdown标题编号、表格列数和新增链接格式检查通过；未访问或编程ZCU102，未修改SD/QSPI，未加载CUDA。

关键证据：

- `docs/EDGE_AI_PLATFORM_TOTAL_COMPARISON_2026.md`
- `docs/ROBOT_COMPUTE_PLATFORM_AND_FPGA_ROLE_SURVEY.md`
- AMD ZCU102 User Guide UG1182、AMD XAPP1326
- GLITCHES（HPEC 2024）、HPU、INF²、HillInfer与ReCoVLM论文/会议页面

当时下一命令（已由2026-09-04路线调整取代，仅保留历史）：

```bash
make -C fpga_accel/runtime_activation_capture test
```

### 2026-09-04 ZCU102平台无关稀疏协处理路线冻结

- [x] 修改总研究计划：不再预先选择Jetson、昇腾、服务器GPU或其他NPU作为唯一主平台，先冻结跨平台不变的ZCU102职责。
- [x] ZCU102固定定位为模型工作量削减协处理器：生成并执行结构保持的视觉Token评分、空间约束、选择、合并、压紧、原始位置/MRoPE/DeepStack映射和块稀疏描述符。
- [x] 把硬件分成控制面、Token数据面和条件启用的稀疏计算面；规则稠密GEMM默认留给主平台，只有`PL计算+通信`实测小于主平台被替代计算时才启用稀疏Attention/GEMM/GEMV。
- [x] 固定Patch/浅层前、Vision layer-5、Merger后三个候选边界；每帧最多一次主张量往返，不允许24/36层逐层跨板，也不迁移完整KV Cache。
- [x] 明确逐阶段参与方式：Vision投影/FFN和LLM Prefill主要通过Token行数减少间接受益；文本Prefill与Decode默认留在主平台；Decode只保留权重/KV常驻且跨越实测收益门的条件分支。
- [x] 重新编排WP2–WP7：现有稠密DATAFLOW与驻留FFN冻结为能力/Roofline基线；下一主线依次为稀疏合同与边界选择、PL Selector/Merger/Compactor、传输与主平台适配、可选稀疏计算面、系统评测。
- [x] 评价口径限定为模型端到端：独立新图和连续帧分开报告，不把动作头、传感器I/O或机器人控制计入模型加速贡献；功耗计入主平台、ZCU102和互连总系统。
- [x] 数据隔离规则保持不变：L2@87.5%只作冻结安全/硬件合同基线；新方法必须使用全新开发和未观察测试集，禁止用已揭示blind/validation/task-routing清单选参。
- 验证：修改章节人工回读；旧路线冲突关键词审计完成；未加载CUDA、未访问/复位/配置ZCU102、未写SD/QSPI。
- 决策：`PLATFORM_AGNOSTIC_SPARSE_COPROCESSOR_PLAN_FROZEN / DENSE_FULL_BLOCK_DEFERRED / SPARSE_TOKEN_CONTRACT_AUDIT_NEXT`。

关键证据：

- `docs/PL_ACCELERATION_RESEARCH_PLAN_2027.md`
- `docs/EDGE_AI_PLATFORM_TOTAL_COMPARISON_2026.md`
- `reports/2026-08-27-task-routing-test.md`
- `reports/2026-08-31-zcu102-dataflow-rerun.md`

精确下一命令（只审计可复用字段，不加载模型、不访问板卡）：

```bash
rg -n "token.*mask|keep_count|original.*index|mrope|deepstack" fpga_accel evaluation profiling_src --glob '*.[ch]' --glob '*.cpp' --glob '*.py'
```

### 2026-09-04 FPGA角色复核与多切入点Oracle运行前门控

- [x] 复核HPCA/FPL/MICRO/ICCAD/ASPLOS/DAC/FCCM相关设计，冻结FPGA子图的最低条件：至少具备单向模型数据流，或使权重/KV/连续子图跨操作驻留；条件必要但不充分。
- [x] 从相关工作收紧创新边界为“具身空间结构保持Token削减 + Patch/浅层到Compactor单向驻留 + 精度/计算/通信联合切入点选择”。不再把外置小Selector或与主平台竞争稠密GEMM作为默认主张。
- [x] 删除已撤回`arXiv:2512.24713`作为方案依据，改用DAC 2024 FNM-Trans，并补充HeatViT、Fitop-Trans、DFX、FlightLLM、Hummingbird、HILOS、HPU和ReCoVLM的实际驻留角色。
- [x] 新建`datasets/dac_cutpoint_oracle_dev.jsonl`：16张COCO新图、32条Pointing/Grounding请求，与六套既有manifest共416张图像重合为0；数据角色明确为development oracle。
- [x] 扩展真实模型前向到五个切入点：Patch Embedding、视觉块1、视觉块3、第一个DeepStack层5（零基索引5、已完成6块）及Merger后；支持100/75/50/25%规则group保留。
- [x] 几何Oracle使用真实框，优先保留所有目标相交group，再以空间锚点和邻近上下文填充；只用于不可部署精度上界。
- [x] 41/41单元测试通过，新增所有切入点下的DeepStack形状对齐仿真、Oracle确定性/目标覆盖和通信字节测试。
- [x] 静态审计同时覆盖真实512像素开发网格和既有4320 Patch板端合同。后者25%保留时，FP16浅层张量单向为2.113 MiB/10GbE 2.216 ms，外置往返为10.551 MiB/11.064 ms；1GbE往返下界110.635 ms。均未计协议、DMA、驱动、布局转换和计算，不能称为链路实测。
- [!] A6000检查发现用户`chenbowu`的训练进程；首次检查87°C、286.13 W，复查仍有32%利用率。遵守共享GPU隔离要求，未加载本项目9 GB模型，真实Oracle输出门延后至GPU空闲。
- [x] 未访问、复位或配置ZCU102；未修改SD/QSPI。
- 决策：`CUTPOINT_ORACLE_PREFLIGHT_PASS / REAL_MODEL_GATE_PENDING / SELECTOR_RTL_NOT_STARTED`。

关键证据：

- `reports/2026-09-04-fpga-role-and-cutpoint-gate.md`，sha256 `18f0c92382dc00e0a0a22c79ccb8332a70da796f9fbc4dfa31b78e3d063b1c98`
- `reports/2026-09-04-cutpoint-oracle-preflight.json`，sha256 `f3ce974bcdd43aa72872ccf60e2a6cf0ee1cd594731ae79d25811aaf5c44c82f`
- `reports/2026-09-04-cutpoint-oracle-preflight.md`，sha256 `17053e47246c36c8eddd830d64a7d9decdb682ea3b82e83e26c72fed55bfe36a`
- `reports/2026-09-04-cutpoint-oracle.lock.json`，sha256 `02d3d3a16c9590a7f47f80ab7bf6c24dbd6d936a2c9ed3575a5d03d927853f05`
- `docs/PL_ACCELERATION_RESEARCH_PLAN_2027.md`，sha256 `729b20e0abe739995dcbb17ac13a775b039056b9b3b41ed51242e9f102b103eb`

精确下一命令（执行前必须再次检查GPU显存、温度和进程归属）：

```bash
/home/zhaoxiangyu/conda-envs/robobrain25/bin/python evaluation/run_hf_cutpoint_oracle.py --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --model-label RoboBrain2.5-4B-BF16 --manifest datasets/dac_cutpoint_oracle_dev.jsonl --output-dir evaluation/results/cutpoint_oracle_dev --limit-per-task 4 --cut-points patch_embedding vision_block_1 vision_block_3 deepstack_layer_5 post_merger --retentions 1.0 0.75 0.5 0.25
```

完成后运行：

```bash
/home/zhaoxiangyu/conda-envs/robobrain25/bin/python evaluation/build_cutpoint_oracle_report.py --manifest datasets/dac_cutpoint_oracle_dev.jsonl --results-dir evaluation/results/cutpoint_oracle_dev --json reports/2026-09-04-cutpoint-oracle-summary.json --markdown reports/2026-09-04-cutpoint-oracle-summary.md --limit-per-task 4
```

### 2026-09-04 BF16切入点烟雾门完成，最终Q8_0复验启动

- [x] 共享A6000空闲后再次检查进程归属、显存和温度，确认无其他用户计算进程，才加载本项目BF16模型；运行结束后模型已卸载。
- [x] 五个切入点×四种保留率×每配置8条样本全部完成，共20个配置、160条原始记录；所有100%保留配置跨切入点输出一致。
- [x] 小样本BF16结果中，Patch Embedding@75%与Vision block 3@75%未通过空间精度门；Vision block 1@75%、首个DeepStack@75%和Merger后@75%仅列为临时候选。其中Vision block 1@75%理论MAC比例为0.746，10GbE单向序列化下界1.083 ms。
- [!] 这些结果只证明BF16结构可行性，样本量为每任务4条且Oracle使用真值框，不能作为可部署Selector、论文精度或最终Q8_0结论。
- [x] 用户确认最终系统固定为4B Q8_0：`RoboBrain2.5-4B.Q8_0.gguf`主模型与`RoboBrain2.5-4B.mmproj-Q8_0.gguf`视觉投影模型。计划、报告和实验锁均已改为Q8强制复验。
- [x] 审计本地`llama.cpp` Qwen3-VL多模态路径：当前视觉图按固定`n_patches`构图，24个视觉块后执行Merger并拼接DeepStack；`clip_image_batch_encode`还要求最终Token数等于静态输出数。HF切入点代码不能直接代表Q8 GGUF执行路径。
- [x] 冻结两级Q8实现顺序：Q8-A先实现Merger后Token列压紧，并同步DeepStack、占位Token和MRoPE，100%旁路必须逐位/逐字一致；Q8-B再在视觉第1块后加入固定预算group gather，使余下23个视觉块及LLM视觉Prefill真实减算。
- [!] Q8-A与Q8-B在真实主模型+mmproj路径通过前，不冻结Selector描述符，不进入Selector/Merger/Compactor RTL，也不宣称最终精度或端到端加速。
- [x] 本里程碑未访问、复位或配置ZCU102，未修改SD/QSPI。
- 决策：`CUTPOINT_ORACLE_BF16_SMOKE_PASS / FINAL_Q8_0_TWO_STAGE_GATE_REQUIRED / SELECTOR_RTL_BLOCKED_BY_Q8_GATE`。

关键证据：

- `reports/2026-09-04-cutpoint-oracle-summary.json`，sha256 `4a76f22f12ada73031562996f24f513c0bc319c603972da6a3a958f8706c1709`
- `reports/2026-09-04-cutpoint-oracle-summary.md`，sha256 `24cf76240a9662bc7260ed5e12c1a5aff5a352252d8f5d94ba87afe47446df41`
- `reports/2026-09-04-q8-cutpoint-path-audit.md`，sha256 `ac48209b5501f1b9348fe361475e938ba8de03d1e5d36e4d14fe65a79f59ae2b`
- `reports/2026-09-04-fpga-role-and-cutpoint-gate.md`，sha256 `cb845e69a2e9f459eb14657249a4a1d7a4b18af4bdf75a2a4ab703412a4af7f6`
- `reports/2026-09-04-cutpoint-oracle.lock.json`，sha256 `5e53f3175ff22a8899f6a2447cba34a4e622174ee7a5f9f8d628d1929b6b6d53`
- `docs/PL_ACCELERATION_RESEARCH_PLAN_2027.md`，sha256 `a340905b0037d3d5fc3028700993d612550ad0565a81f99099ee25e66ba1aaaa`

精确下一命令（不加载CUDA、不访问板卡）：

```bash
sed -n '1,240p' reports/2026-09-04-q8-cutpoint-path-audit.md
```

之后实现Q8-A离线回放和100%旁路一致性测试；Q8-A通过后才实施Q8-B视觉图动态缩短。

## 最近验证

```text
evaluation tests: 34/34 PASS
blind inference count: 576
report summary sha256: 9c9a9faf8ef726929c09c9f323fd2a4aa4cb81b0c4b2d5e97e01f6bda014c721
report markdown sha256: caf3e1488016f3cb4d0238790f6f5bb919385197b1a65b8808abff4f74a72b9d
joint-deletion pilot masks: 14/14 unique and finite
joint-deletion repeat max_abs_nll_delta: 0
joint-deletion raw sha256: 90a8731f26752652345c5240a916b68a49f31cb2dcf05fa04380ba1ac755ba1a
joint-deletion summary sha256: f708fc2e46cf6281a4b873e76a5864fcfd1a6ecf0683cdccea6500586aeb5595
joint-deletion development records: 192/192
joint-deletion development candidate forwards: 1344
joint-deletion development sha256: a16477a52dc9b29a3d81c5396249b9f79e22a4f0e21fa2dd6b83c1a9b29031d0
additive selector development gate: STOP (pairwise accuracy 0.4903 < 0.52)
additive selector summary sha256: ecd72f52be07a73f688077a486216199d017a19a921e57a5c167a0e3e8240721
candidate interaction development gate: PASS (coordinate-NLL only)
candidate interaction artifact sha256: 33e3f9a2926896d67ee89cd93c98e8f38c31a9e3b330aadc61b9a5a44233bac6
candidate interaction summary sha256: 6148bf49a9345bbc4b05bf7cb053ccffc2dc36004e676588fd9292c8592a6d02
candidate interaction real-output gate: FAIL (Pointing PCK@0.1 delta -0.0625)
candidate interaction calibration json sha256: 8249531103c67902692994a4dac51d0a217f58fb5984b78313740c391b8b48d2
task-routing untouched manifest images: 96, overlap with all prior images: 0
task-routing L2 raw sha256: 737f214ddbc0c3496f968fe5069be2553ffbbafc94c5a402b08872997d3ee13f
task-routing routed raw sha256: 12dbe3fac7df95d97e04a940578b592c99223a308d0299147b582b5d7c0ed2db
task-routing report json sha256: c80005ea06e9b7f2bb23f6ed1d77f4ccb79be7ddbffcd8797f64293449afd0b1
task-routing gate: FAIL; final token baseline: L2@87.5%
PE16 HLS IP sha256: c5a6bc869fc139f96343fa6a5885f04f5ac41b42cf0fc395d864c69cb822f81a
PE16 compute-control RTL: PASS (16 banks, 24 writes/bank, 12 addresses/bank)
PE16 top-level AXI cosim: STOP_RESOURCE_RISK (no RTL result)
PE16 RTL summary sha256: 8f6fbd0b98c6916a99166f0a9c421d579517dc600f04f465a8c5ad285f7c87c8
PE16 final bitstream route: PASS (WNS +0.067 ns, WHS +0.011 ns, unrouted 0)
PE16 Vivado summary sha256: 49f4d4df4800e8a1838d129139aceeb074184d7e7990c75d4de53f8229d2ef09
PE16 final routed DCP sha256: 185a2b7d8fe68736abc30a169ca6ac88140772e58a76d6995069b1edb20a3f20
PE16 bitstream sha256: d8ab6e8215d1ce41f320be8afad51aff5d4e7d90b4135e7994c90d9e1f810f65
PE16 bitstream/XSA integrity gate: PASS (8/8)
PE16 matching FSBL sha256: 9280d4b92adb9aec54daeeae7681bb4206bb4e65d19c3ef24805bddd973a4efb
PE16 runtime register map: PASS (10/10)
PE16 versioned RAM-Linux bundle: PASS
PE16 offline pre-board bundle: PASS
PE16 pre-board summary sha256: 5de0cb7d87289bbcc9fcf4d98f1bc4d761b6c3258efd87bb2f4d46a1657fa28d
PE16 bitstream generated: YES
PE16 JTAG volatile board boot: PASS (DDR/PL/AXI/UIO)
PE16 initial/final board self-test: PASS (max_abs=0)
PE16 board benchmark: PASS (Prefill 242.808 ms; Vision 448.839 ms; Decoder 14.647 ms)
PE16 board APM: PASS (traffic + latency; overflow=0)
PE16 board summary sha256: be16c875906413c76ed15bd426546c579cdf116da0c43bf6c78fcbed38296a36
PE16 UART raw sha256: 9ee996a06cae1874e18f5f95b25b1b17df53d852842d4e8222e2cdd5f38d733a
PE16 XSCT raw sha256: 4d68c927fa840ff7a6aa5d5495039692b3445cb3dcfa63cb4bae8a0ef781c113
PE16 board programmed: YES (volatile PL/RAM only; no SD/QSPI write)
PE16 final control: 0x00000004 (idle); UART shell alive
PE16 DATAFLOW C testbench: PASS (9/9; max_abs 3.05175781e-05)
PE16 DATAFLOW HLS: PASS (DATAFLOW synthesized; compute II=1; estimated 3.884 ns)
PE16 DATAFLOW resources: BRAM18K 816/1824; DSP 406/2520; FF 37826; LUT 42110
PE16 DATAFLOW conservative schedule reduction: Prefill 20.82%; Vision 38.11%
PE16 DATAFLOW summary sha256: 853e788fb4add714ec39cb9fe39ae8791670ab2e31091eb7216d3b1c300cd550
PE16 DATAFLOW report sha256: e8f2355f55adb80ab693bdcd99475ee3dde719f152148a87b23a0a37b6b3c1c3
PE16 DATAFLOW board programmed: NO; existing board state unchanged
PE16 DATAFLOW directed RTL: PASS (16 banks; 24 writes/bank; init fused)
PE16 DATAFLOW generated structure: PASS (load -> activation FIFOs -> compute -> output FIFO -> store)
PE16 DATAFLOW RTL summary sha256: 42777e76e1ada309bc7ac3abaf1c5f6a3fee7d50f0bd9ae00da10d39478d4cf3
PE16 DATAFLOW RTL report sha256: 7af20e8bce0d10bcec7d88d6bcc4cb68c341b1caf6e397eb8d858c2cc3a3f0f3
PE16 DATAFLOW IP sha256: 309a4bad50a4d377eedfa63a1a19fa9b1007a9ad81c9b355159868631306f723
PE16 DATAFLOW OOC route: PASS (WNS +0.015 ns, WHS +0.028 ns, unrouted 0)
PE16 DATAFLOW full-system BD validate: PASS (AXI output 128 bit; control 0xA0000000)
PE16 DATAFLOW BD validate log sha256: 41a635e139b94e043987b05962ac2c4c57295931cebda6afba546909e1e23705
PE16 DATAFLOW full-system route: PASS (WNS +0.078 ns; WHS +0.010 ns; unrouted 0)
PE16 DATAFLOW full-system summary sha256: 57e36b4b1ecbc84df41a65c2181610573933adbc521a2e1daf9f88775590368e
PE16 DATAFLOW routed DCP sha256: 8eda6c2deba32c331044a88009ff39bec6d3fd4f3b02b1d968b40248aa33efca
PE16 DATAFLOW bitstream: PASS (WNS +0.078 ns; WHS +0.010 ns; unrouted 0; DRC Error 0)
PE16 DATAFLOW bitstream sha256: c70d08a386d2a69b4319094107eea28e0591119ad1a86bfabf5598a806600036
PE16 DATAFLOW XSA embedded bit integrity: PASS
PE16 DATAFLOW board accessed/programmed: NO/NO; existing board state unchanged
PE16 DATAFLOW FSBL: PASS (SPD dynamic dual-rank 16GB; JTAG-only)
PE16 DATAFLOW FSBL sha256: 0f2c39fe97f121a31bef848996d460be71f174ae10b65dc28f05b90d8fe4481a
PE16 DATAFLOW runtime: PASS (unit 4/4; sanitizer 4/4; benchmark; APM; DT; register map 10/10)
PE16 DATAFLOW offline pre-board bundle: PASS
PE16 DATAFLOW board authorization: REQUIRED; no board access performed
PE16 DATAFLOW task-independent descriptor: PASS (5/5 native; 5/5 sanitizer; task fields 0)
PE16 DATAFLOW 64MiB auto chunks: visual prefill 976x2; vision 3584x2; decoder 1x1
PE16 DATAFLOW RAM Linux QEMU smoke: PASS (not PL evidence)
PE16 DATAFLOW full visual prefill benchmark case: PASS (M=1080 distinct from frozen M=512 kernel shape)
PE16 DATAFLOW measurement protocol: PASS (15 stages; 15 mock idle gates; task fields 0)
PE16 DATAFLOW board summarizer: PASS (4 tests; APM latency formula total/total)
PE16 DATAFLOW UART capture state machine: PASS (PTY; command after boot gate; append-only log)
PE16 DATAFLOW measurement RAM Linux Image sha256: 73c131879b1237d4cfb8de4eb24cf7a288d27f6defda488609d9d86c87457e6b
PE16 DATAFLOW measurement initramfs sha256: 927cb283c56687a0f8f7b53b5d3c48fc03030443206e1c371ce069b6461d8972
PE16 DATAFLOW measurement benchmark sha256: 9efd7c52dd27d5f4041b0600fc9126a02b0460adc24d203cd3348f12445e8925
PE16 DATAFLOW measurement QEMU log sha256: edc702314dc4233f597a652a867bd1c16ae563b9f626f0d510c5f10f1aa5cb11
PE16 DATAFLOW offline bundle verifier sha256: d195e6ec679897abba48f542eccded4958db39d14581581aad32a85900dfb766
PE16 DATAFLOW pre-board summary sha256: 8867f005826eac484b30ebd1d9e44ae44778f502b46051426c6da9feda83390b
PE16 DATAFLOW pre-board report sha256: 0c4a849b72836f20d088036567d242046e88944398baed48f9bc34c38464f98d
PE16 APM latency correction: PASS (old average values divided by 3; other conclusions unchanged)
PE16 APM latency correction json sha256: 2d1fbeef0cc02f6c2824234ad4f2c7aa40f95976973619914189e985e4c257b2
PE16 DATAFLOW offline pre-board stage accessed/programmed: NO/NO (historical gate)
PE16 DATAFLOW JTAG volatile board boot: PASS (DDR/PL/AXI/UIO; no SD/QSPI write)
PE16 DATAFLOW protocol: PASS (15/15 stages idle; initial/final max_abs=0; APM overflow=0)
PE16 DATAFLOW board kernel speedup vs PE16: language 1.391091x; vision 1.648739x; decoder 0.998410x
PE16 DATAFLOW auto chunk speedup: visual prefill 1.032047x; vision 1.079980x; decoder 1.000114x
PE16 DATAFLOW UART raw sha256: 179c25a1c1139841165fa89bba9044b2bf60cb7bed205087c19d3570a98d15ae
PE16 DATAFLOW XSCT raw sha256: 4d68c927fa840ff7a6aa5d5495039692b3445cb3dcfa63cb4bae8a0ef781c113
PE16 DATAFLOW board summary sha256: 41013e32ef7353f24f80112f8b36caf4071ec481720f985eb9c49bf20aa34807
PE16 DATAFLOW board report sha256: 77176707c557c589595d76ef5bd23b6177812e68da8481782f1fb155041ecd2e
PE16 DATAFLOW final control: 0x00000004 (idle); UART shell alive
GGML Q8Q8 bridge native/sanitizer: PASS (4/4 each; safe fallbacks 8)
GGML Q8_0 upstream layout: PASS (QK 32; bytes 34; scale offset 0; payload offset 2)
GGML Q8_0 upstream quantizer differential: PASS (19 rows; 152 blocks; byte equal)
GGML public tensor adapter: PASS (view fallback enabled)
GGML bridge AArch64 cross compile: PASS (not board-executed)
GGML bridge upstream modified / board accessed / board programmed: NO / NO / NO
GGML profile coverage: PASS (decode 252/253; kv 249/253; text prefill 252/253; vision encode 104/106; vision prefill 747/759)
GGML A53 prepare benchmark: PASS (6 shapes; 12 initial measurements; RAM-only UART)
GGML A53 NEON numeric: PASS (6/6 full byte equality; 24 measurements)
GGML A53 NEON speedup: M512 11.76x; M1080 12.23x; M4320 12.40x vs scalar
GGML A53 NEON fast4/kernel ratio: M512 1.55%; M1080 1.48%; M4320 3.43%
GGML weight repack per call: REJECT (cacheable DDR 145-148 ms; existing UIO ~354 ms for N9728/K2560)
GGML pthread dynamic attempt: STOP_MISSING_LIB (no benchmark entered; static retry PASS)
GGML preparation board access/program/reset/persistent writes: YES / NO / NO / NO
GGML fast UIO weight staging: PASS (3 shapes x 2 destinations; full byte equality)
GGML UIO fast4 weight staging: 3.207 ms / 12.038 ms / 2.188 ms
GGML fast UIO output readback: PASS (3 shapes; full byte equality; 0.39-0.45 GB/s)
GGML UIO fast4 output readback: 48.957 ms / 83.899 ms / 150.351 ms
GGML fast execute bring-up path: PASS (native/sanitizer mock reference+fast; max_abs=0)
GGML final per-op PS roundtrip architecture: REJECT; PL-resident FFN + live-range chunking NEXT
PL-resident FFN planner: PASS (LLM chunk 432x3; Vision chunk 2896x2; overlap 0)
PL-resident FFN vector C simulation: PASS (9/9; LLM max_abs 0.000711; Vision 0.002926)
PL-resident FFN vector HLS: PASS (all loop constraints satisfied; estimated 253.49 MHz)
PL-resident FFN vector OOC route: PASS (WNS +0.802 ns; WHS +0.023 ns; unrouted 0)
PL-resident FFN vector resources: 26,685 LUT; 47,063 FF; 154 DSP; 64 BRAM Tile
PL-resident FFN vector IP sha256: 0a8959b1ea6478f228608fe20df4deb3fe6fc566bd5e8b10d19f88095b786ba0
PL-resident FFN board accessed/programmed/SD modified: NO / NO / NO
PL-resident FFN full-system BD validate: PASS (GEMM 0xA0000000; vector 0xA0010000)
PL-resident FFN full-system route: PASS (WNS +0.060 ns; WHS +0.009 ns; unrouted 0)
PL-resident FFN full-system resources: 58,445 LUT; 88,156 FF; 483 BRAM Tile; 815 DSP
PL-resident FFN full-system summary sha256: d4699598995cc219d8834abea42d1931bc6841c9ec1de7a85d665dd88698f495
PL-resident FFN full-system routed DCP sha256: 8cb2c550cdcbafcbb32ca344141b1b91123b3fa73e2b6174739c46e06159ec2d
PL-resident FFN full-system bitstream/board/SD: NO / NO / NO
RoboBrain real-weight full proxy: PASS (max_abs 0.03017; RMSE 0.002165; relative_L2 1.235%)
RoboBrain real-weight HLS slice: PASS (10/10 total; max_abs 0.001716; RMSE 0.000346)
RoboBrain real-weight proxy summary sha256: 269ebad274ae0c891ecdbb81228774c01a03699b1c9f5c92d67466f915a7e7ca
RoboBrain runtime FFN activation captured: NO (proxy input only)
RoboBrain real-weight gate CUDA/board/bitstream/SD: NO / NO / NO / NO
```

### 2026-09-10 Q8-A真实Q8_0身份门与75%压紧门完成

- [x] 修复Qwen3-VL静态图`temporal_merge=2`被Q8-A误拒绝的问题；仅放行最终形成单个二维MRoPE网格的静态图，多时间片视频仍拒绝。
- [x] C API测试夹具改为双条目、2:1 temporal merge，稀疏索引、重复索引拒绝、MRoPE原位置和100%身份旁路测试通过。
- [x] 真实Q8_0主模型+mmproj完成stock/identity门：两路退出码0，20个生成Token ID逐项相同，输出均为`[470, 16, 918, 427]`。
- [x] Q8-A 75%真实压紧完成每任务4条、共8条开发样本：176视觉Token压到132，8/8退出成功且结构化输出解析率100%。
- [!] 75%质量门失败：Grounding Mean IoU `0.7701 -> 0.7799`、Acc@0.5不变；Pointing PCK@0.1 `0.75 -> 0.50`，下降0.25超过预注册0.125门限。不得宣称post-merger@75%为Q8候选。
- [!] 几何Oracle使用真实框且每任务仅4条；只作开发诊断。进程时间包含模型加载，不作为稳态性能或加速证据。
- [x] 未使用共享GPU，未访问、复位或配置ZCU102，未修改SD/QSPI。
- 决策：`Q8_A_IDENTITY_AND_MECHANICS_PASS / POST_MERGER_75_QUALITY_FAIL / Q8_B_IMPLEMENTATION_NEXT`。

关键证据：

- `reports/2026-09-10-q8a-real-q8-validation.md`
- `evaluation/results/q8a_identity_20260910T143750/summary.json`，sha256 `1b82946ea18c78c38476ed42df25a48961132d357c61eb511ade9cb06246dcb8`
- `evaluation/results/q8a_oracle75_20260910T144447/raw_records.jsonl`，sha256 `9559e4aca45a3a052576b5372dddadb5d5ed24b387128ea3af97612c707da195`
- `evaluation/results/q8a_oracle75_20260910T144447/summary.json`，sha256 `c5f035b36bbeec9cf530a26862d1730ca01f6a4d95e662cc1d2671fb98aae0bb`

精确下一命令（CPU静态审计，不访问GPU或板卡）：

```bash
sed -n '1,240p' profiling_src/llama.cpp/tools/mtmd/models/qwen3vl.cpp
```

随后实现Q8-B视觉块1后的完整2x2 group gather，先通过100%身份门，再运行相同8样本75%几何Oracle门。Q8-B通过前不冻结Selector描述符或RTL。

### 2026-09-10 Q8-B真实Q8_0身份门与质量门完成

- [x] 在Qwen3-VL视觉块1后实现完整2x2 Merger group gather；后续23个视觉块、DeepStack、Merger和LLM视觉Prefill均使用缩短序列。
- [x] 新增独立Q8-B early-keep C API，与Q8-A互斥；稀疏索引、冲突拒绝和身份旁路C测试通过。
- [x] 清理手工增量归档产生的重复静态库成员，并用原始对象清单重建`libmtmd.a`；最终CLI SHA256为`e0e8b34088e96ed3b632519d2c9d5d0ad9ca070183f1448f3a24405aa6531ec4`。
- [x] 最终100% identity：stock与Q8-B路径退出码均为0，20个生成Token ID逐项一致，输出均为`[470, 16, 918, 427]`。
- [x] 最终干净二进制完成Q8-B@87.5%每任务4条、共8条开发样本：176个group Token压到154，对应后续视觉patch为704压到616；8/8退出成功且解析率100%。
- [!] 87.5%质量门失败：Grounding Mean IoU `0.7701 -> 0.7993`、Acc@0.5不变；Pointing PCK@0.1不变，PCK@0.2 `1.0 -> 0.75`，下降0.25超过预注册0.125门限。
- [!] 75%中间构建复核也因Pointing PCK@0.2下降0.25失败，只作为附加证据；正式结论引用最终干净二进制87.5%结果。
- [!] 几何Oracle使用真实框且每任务仅4条，不能部署或形成论文级精度结论；运行墙钟包含模型加载，不是稳态延迟或实测加速。
- [x] 发现87.5%原始记录Oracle布尔元数据沿用旧模式名而误写，已修复后续脚本并新增绑定原始SHA256的勘误文件；不覆盖原始数据，指标不受影响。
- [x] 本里程碑未使用共享GPU，未访问、复位或配置ZCU102，未修改SD/QSPI。
- 决策：`Q8_B_IDENTITY_AND_MECHANICS_PASS / Q8_B_75_AND_0875_QUALITY_FAIL / SELECTOR_RTL_HOLD / PLATFORM_AGNOSTIC_BENEFIT_MODEL_NEXT`。

关键证据：

- `reports/2026-09-10-q8b-real-q8-validation.md`
- `evaluation/results/q8b_identity_20260910T152554/summary.json`，sha256 `1447198d1e697a579a032dad7a306f518991d0ae5325c12831ee6eaa85f32689`
- `evaluation/results/q8b_oracle0875_20260910T160715/raw_records.jsonl`，sha256 `e6656c2d49787611b0306df37f0bcc1a63f4186a922a71bae33c1ef20166c6ab`
- `evaluation/results/q8b_oracle0875_20260910T160715/summary.json`，sha256 `17b554ad2505c76655bc888ff906ec9f7342c2e7a700f2af3d663a69d2372d5c`
- `evaluation/results/q8b_oracle0875_20260910T160715/metadata_erratum.json`

精确下一步（CPU静态建模，不访问GPU或板卡）：建立平台无关的切入点收益模型，输入阶段时间、Token/patch数量、张量字节数、链路带宽与驻留策略，输出单向驻留和外置往返的净延迟/能量交叉点。当前不进入Selector RTL，也不在已观察的8条样本上继续调参。

### 2026-09-10 平台无关异构收益模型完成

- [x] 新增参数化配置、纯Python模型和3个单元测试；JSON/CSV/中文Markdown三种输出均通过解析检查。
- [x] 直接复用既有A6000 BF16 Oracle原始记录中的视觉前缀、压缩后视觉块、视觉其他部分和LLM Prefill阶段中位数，不按层数比例伪造时间。
- [x] 扫描5个切入点、4档链路、4档视觉前缀加速倍数，共80种组合；分别输出主平台软件压缩、单向驻留和外挂往返的延迟，以及基于假设功耗的能量敏感性。
- [x] 输出给定链路下最低前缀加速倍数，以及给定加速倍数下最低链路带宽，供未来AI核心平台选型后直接回填实测参数。
- [!] 乐观的零固定开销模型中，外挂往返相对主平台软件压缩为`0/80`获胜；单向驻留为`25/80`获胜。该结果支持“连续子图驻留”，不支持把FPGA只作为外挂Selector。
- [!] 10 Gb/s、有效率0.8、前缀4倍的示例中，Vision block 1仍比软件压缩慢0.587 ms，达到交叉点约需21.814 Gb/s；Vision block 3仅快0.192 ms，余量很小。所有数值均是BF16计时驱动的敏感性估算，不是Q8、ZCU102或目标AI平台实测。
- [!] 能量结果使用30 W主算力、5 W加速器、3 W链路的显式假设，只能用于趋势扫描，不得作为板级功耗或论文实测值。
- [x] 本阶段不使用GPU，不访问、复位或配置ZCU102，不修改SD/QSPI。
- 决策：`PLATFORM_AGNOSTIC_BENEFIT_MODEL_PASS / EXTERNAL_SELECTOR_ROUND_TRIP_REJECT / TARGET_PLATFORM_PARAMETERS_PENDING`。

关键证据：

- `reports/2026-09-10-platform-agnostic-benefit-model.md`，sha256 `68953bda209fc28c9d907c9aa92a6999e18ce6374d2a8477937fd9855dfd045e`
- `reports/2026-09-10-platform-agnostic-benefit-model.json`，sha256 `774e362cd9f242b0504a28067d9c017f96109ec8c34caa533f9be336092717ab`
- `reports/2026-09-10-platform-agnostic-benefit-model.csv`，sha256 `b3577377a760a8c3171a15c1d06e02652f3ffc18aa1868471da733929b08d0e3`
- `evaluation/heterogeneous_benefit_model.py`，sha256 `fd68ec69962d6a4eeea8ef9d27997d9335e6cb169026193ce3b9e797a8cf5df4`
- `configs/platform_agnostic_benefit_model.json`，sha256 `9bda45a2e4470d47562bd5aebf2830aa8033f2637bba528b38850ef9ba9ab234`

精确下一步：确定AI核心平台前，冻结本模型接口；不为失败的Oracle Selector写RTL。可继续补充`Token/Block Sparse Descriptor v1`的纯数据合同，但只能定义通用全长/索引/位置/CRC格式，不得把当前失败的75%或87.5%选择策略固化为硬件算法。

### 2026-09-10 Token/Block Sparse Descriptor v1完成

- [x] 完成64字节little-endian头、严格递增原索引、0至4维原始位置、Identity旁路和版本化flag定义。
- [x] payload与header分别使用IEEE CRC-32；保留字段、未知flag、长度、索引范围和Identity约束均由解析器验证。
- [x] C参考编码器和零动态分配解析器完成，普通构建10/10与ASan/UBSan 10/10测试通过。
- [x] C生成124字节golden binary，独立Python codec解析并重新编码后逐字节一致；binary SHA256为`53b85bfafda5d661a5c229d80cf4a22d0f25265df52f8eb9e0da395feb8793e8`。
- [x] 合同不包含任务名、Selector评分、固定保留率或模型输出规则，不固化当前Q8质量门失败的Oracle选择策略。
- [!] 当前仅为主机C软件合同；未进行HLS、RTL、综合、时序、板端或跨端字节一致性验证。
- [x] 本阶段不使用GPU，不访问、复位或配置ZCU102，不修改SD/QSPI。
- 决策：`TOKEN_SPARSE_DESCRIPTOR_V1_SOFTWARE_CONTRACT_PASS / SELECTOR_ALGORITHM_UNFROZEN / HLS_RTL_HOLD`。

关键证据：

- `reports/2026-09-10-token-sparse-descriptor-v1.md`
- `fpga_accel/token_sparse_descriptor/include/token_sparse_descriptor.h`，sha256 `5a36c11bf710138b5c92691cbf340349a8b64537ea590c54a359ccef0f74ccf1`
- `fpga_accel/token_sparse_descriptor/src/token_sparse_descriptor.c`，sha256 `c1fc07e73a0f00782598e3c0efb9172ab3c0bd82474e04cc21a265fb1fe267b1`
- `fpga_accel/token_sparse_descriptor/tests/test_token_sparse_descriptor.c`，sha256 `7b7c1868645964817bf7de9fb95f30e83339cb70b638865dcadc135e2593106c`
- `fpga_accel/token_sparse_descriptor/artifacts/golden_sparse_v1.bin`，sha256 `53b85bfafda5d661a5c229d80cf4a22d0f25265df52f8eb9e0da395feb8793e8`
- `fpga_accel/token_sparse_descriptor/artifacts/golden_sparse_v1.json`，sha256 `b09a1cdd1aa83ee792ba01d879353e98cc60c4f3e1de2fae252887fda944b72a`

精确下一步：在核心AI算力平台未定、当前Selector质量门失败的条件下，暂停算法绑定的RTL。收益模型和稀疏数据合同已具备，待平台确定后优先回填实测链路/功耗/Q8阶段时间；新Selector必须使用新的开发划分，并在冻结后进入未观察测试集，不能继续用已观察8样本调参。

### 2026-09-11 Q8 87.5%扩大诊断与失效归因

- [x] 将Q8-B评估从每任务4条扩大到每任务96条：96张COCO图片、30个类别、192条任务，stock/Q8-B共384条推理全部成功。
- [x] 预先冻结配对Bootstrap种子、10,000次重采样和非劣界限；不再用单张失败代替总体结论。
- [x] Q8-B Pointing PCK@0.1由0.6458降至0.5104，差值-0.1354，95%区间[-0.2190,-0.0521]，确认超过5个百分点退化界限。
- [x] Q8-B Pointing PCK@0.2只下降0.0104，95%区间[-0.0521,+0.0312]；Grounding Mean IoU下降0.0152且区间跨0，Acc@0.5上升0.0104且区间跨0。这些指标不得称为已确认退化。
- [x] 在同一完整集补跑Q8-A Merger后87.5%裁剪192条；PCK@0.1下降0.0938、Mean IoU下降0.0510，两者相对零变化区间均为负，说明退化不依赖早期23层视觉计算被跳过。
- [x] 同掩码阶段归因：48条支持Token集合或共享压紧路径影响，15条支持早期裁剪影响；Q8-B相对Q8-A的PCK@0.1差异区间跨0，不能证明早期裁剪普遍更差。
- [x] 新增目标网格覆盖审计：目标中心192/192保留，目标相交网格8510/8566保留，188/192任务完整覆盖。Q8-B的16个严格Pointing退化中15个发生在完整覆盖样本，排除“主要因目标被直接删掉”。
- [x] 代码审计确认视觉紧凑位置与LLM MRoPE继续使用原始Token索引；49项evaluation单元测试通过。
- [!] 数据已用于旧L2研究且Oracle读取真值框，本轮是扩大诊断而不是新盲测或可部署精度结论。
- [x] BF16运行器新增显式已揭示诊断模式、固定掩码逐索引检查和配对Bootstrap中文汇总器；50项evaluation单元测试通过。
- [!] BF16同索引跨后端复现尚未执行：检查时GPU由用户`seey6`训练进程占用并高负载，未并发干扰、未结束其进程。
- 决策：`Q8_EXPANDED_DIAGNOSTIC_COMPLETE / STRICT_POINTING_DEGRADATION_CONFIRMED / DIRECT_TARGET_DELETION_REJECTED / CONTEXT_OR_SHARED_COMPACTION_SUSPECTED / BF16_SAME_MASK_PENDING / SELECTOR_RTL_HOLD`。

关键证据：

- `reports/2026-09-11-q8-token-pruning-failure-analysis.md`
- `reports/2026-09-11-q8b-expanded-diagnostic.{json,csv,md}`
- `reports/2026-09-11-q8a-expanded-diagnostic.{json,csv,md}`
- `reports/2026-09-11-q8-stage-attribution-full96.{json,csv,md}`
- `reports/2026-09-11-q8b-expanded-diagnostic.lock.json`
- `evaluation/results/q8_geometric_oracle_masks_r0875_expanded96_20260911.jsonl`

精确下一步：GPU空闲后运行`reports/2026-09-11-q8-token-pruning-failure-analysis.md`末尾的BF16同掩码命令；只有该结果才能进一步区分“上下文/裁剪策略问题”和“Q8量化或自定义后端交互”。
