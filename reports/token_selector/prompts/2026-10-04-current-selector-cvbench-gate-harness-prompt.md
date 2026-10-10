# Current Selector CV-Bench Gate

你是执行工程师，只执行，不做研究方向决策，结果尽量简练。项目根目录：
`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`。

目标：按 `configs/evaluation_protocol.json` 对“当前已实现的 B1 prompt-conditioned integer/hardware-regular Token Selector”建立相对未裁剪 RoboBrain2.5-4B BF16 的正式 CV-Bench 质量与模型端到端延迟证据。Pointing/Grounding 仅是历史内部诊断，本任务不得读取其指标作 promotion、不得写入主结论。

安全：先在宿主执行 nvidia-smi + 进程属主审计。不得 kill/暂停其他用户进程；若空闲显存不足 20 GiB 则保存 BLOCKED 报告并停止。只用本地模型 `/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B`，设置 HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1、do_sample=False。不得修改原模型、`app/`、既有 HLS/cosim/OOC/BD 目录及既有报告。新代码仅放 `evaluation/`，新配置放 `configs/`，结果放 `reports/cv_bench/`、`reports/token_selector/`、`artifacts/token_selector/`。不得安装依赖。

执行顺序与硬门：

1. 只读审计当前 B1 runner、当前最终权重/checkpoint、整数 scorer/HLS 输入合同、CV-Bench full2638 manifest/图片/既有 dense BF16 基线。以 2026-10-03 已实现并通过 integer scorer + compact HLS 的算法链为“当前 Selector”，不能退回 L2，也不能发明新算法。
2. 在读取任何新的 proposed CV-Bench 输出前，写预注册 JSON：锁定唯一部署 checkpoint/seed（只能依据已有非 CV-Bench validation 元数据选择；若已有冻结部署 checkpoint 则必须用它）、K={48,64,96} 三个固定 operating points、官方 CV-Bench evaluator、质量/延迟字段、代码与输入 SHA256、禁止依据 CV-Bench 回选。记录 checkpoint 选择依据。若无法唯一确定部署 checkpoint，输出 BLOCKED，不得擅自平均或按 CV-Bench 选择。
3. 实现/复用真实模型 B1 软件路径。先做不超过 5 条的 identity gate：G<=K 或显式 identity，要求与官方 dense 的 generated token IDs、文本、视觉输出逐项一致；这一步只验证实现，不汇总质量。失败则修实现并重试，不得改算法或评测标准。
4. 做不超过 5 条的 proposed 结构 smoke，仅检查 shape、K、MRoPE/placeholder、有限值和计时字段，不查看/汇总正确率，不据此选参。失败可修工程错误，不得调权重、lambda、K 或规则。
5. 门通过后，对官方 CV-Bench full2638 将 K48/K64/K96 各执行一次完整、确定性质量评测。不得根据中途结果停止某个 K；断点续跑必须保证样本唯一、顺序和配置一致。保存每条原始 JSONL/CSV：官方题型、答案、解析成功、正确性、G/K、实际保留率、输出文本/token ids、image/processor/H2D、vision preamble、block0、selector（软件路径需单列，不能冒充 FPGA）、remaining vision、projector/prepare、prefill、decode、postprocess、model-E2E、application-E2E、峰值显存。所有 GPU 段前后 synchronize；模型加载不计入稳态。
6. 正式质量只报告 CV-Bench 官方组合分数、2D/3D、Count/Relation/Depth/Distance、绝对差、相对 dense BF16 保持率、解析成功率和配对 bootstrap 95% CI。主比较必须是当前 Selector vs unpruned BF16；L2 只作为既有消融参照。禁止用 retained mass、NDCG、Pointing、Grounding 代替模型输出质量。
7. 性能正式值：至少固定分层 100 条，在模型加载后 warmup>=1，每配置 repeats>=3，保存每次原始记录，报告 mean/p50/p95。若真实 FPGA 尚未上板，则软件 selector 的整模型时间只能标为 software path；另生成 FPGA-substitution 仿真：用当前 compact RTL/cycle 模型替换 selector 段，并分别给出 (a) local/high-speed、(b) 1GbE 110 MB/s 纯序列化、(c) 90/118 MB/s 敏感性。必须保留 query/量化、链路固定开销等未知项，称 lower-bound/proxy，不能称板上加速。
8. 至少加入 matched-budget L2 对照。Random 与 uniform 若已有安全 runner 就执行相同 100 条性能/质量 screen；若没有，不要阻塞 proposed full2638，明确列为后续缺口，不得临时手写未经测试的实现污染主结果。
9. 测试新增/修改代码，保存 pytest/unittest/py_compile 日志。更新 `CURRENT_STATUS.md`，中文主报告：
   - `reports/cv_bench/2026-10-04-current-selector-cvbench.md/.json/.csv`
   - `reports/token_selector/2026-10-04-current-selector-end-to-end-proxy.md/.json`
   - 原始结果与运行日志使用清晰目录，避免复制大模型或巨大波形。

若完整 2638 运行在本次 Harness 时限内无法完成：必须留下可恢复 checkpoint、已完成数、ETA、命令和状态 `IN_PROGRESS`，不得用部分集得分冒充完整结果。最终 stdout 仅输出：状态、冻结 checkpoint、各 K 完成数/官方分数（若完整）、关键延迟、文件、阻塞/下一条命令，不超过 80 行。
