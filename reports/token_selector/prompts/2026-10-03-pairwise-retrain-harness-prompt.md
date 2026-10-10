你是执行工程师。Codex 已批准进入“轻量 Prompt 条件化 Token Selector 三臂重训可证伪实验”。严格按下述预注册合同实现、运行、测试和写报告；不要替 Codex/PI 改研究方向，最终只返回精简 JSON。

项目根目录：
`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`

## 授权与数据边界

- 用户已授权 DeepSeek Harness 经 `api.deepseek.com` 处理必要代码、指标和衍生数据。
- 严禁读取、上传或输出原始图片、4B 模型权重、API key、SSH key、代理配置和任何凭据。
- 只允许本地读取 train/val 的 prompt_projection NPZ 衍生特征；internal test NPZ 严禁打开。
- 不加载 RoboBrain2.5-4B，不做特征重抽取，不运行 CV-Bench，不使用网络。
- 不修改 `app/`、`models/`、原始模型、既有历史报告和 `configs/prompt_mean_pairwise_retrain.draft.json`。

## 冻结输入与已验证事实

- draft SHA256=`787f8f401bcb10b5fe8007a6a0f5b415858d893cb2797b3c1797d5f1fa785772`
- config SHA256=`44399c8e6fb8f333a150a6016faa00d6ae70524b5fcd030bcabf4e5fb07ab28b`
- index SHA256=`02a50868a84675418e9bd9e5fdd0bb55f522ec5ce0ce1d6da7c69f85d2aeee26`
- train NPZ aggregate SHA256=`33997394c4e148e3cfcd45eb0eb37fc97616cece8db3da5c731171d4b53d3951`
- fixed scorer: `prompt_mean_vector`，`Wt[128,2560]`、`Wv[128,1024]`，runtime query `1x128`，FPGA scorer `Gx128 dot`。
- seeds=`20261002/20261003/20261004`，student temperature=`0.1`，init_std=`0.02`。
- frozen `lambda_target_0p25=14.280209646136903`。
- concentration evidence: per-seed pair-norm ESS≈41/159、signed-positive ESS≈16、约38%–42% image pair 对 aggregate pair gradient 为负投影；这是解释变量，不允许事后改变实验。

## 一、先写盘并冻结预注册

必须在首次读取 val NPZ 前完成。

1. 保留旧 draft 字节不变，新建：
   - `configs/prompt_mean_pairwise_retrain.preregistered.json`
   - `reports/token_selector/2026-10-03-pairwise-retrain-preregistration.json`
2. prereg 状态写 `PREREGISTERED_TRAIN_VAL_ONLY`，记录 draft/config/index/scale-audit/concentration-audit SHA。为避免自哈希悖论，preregistration JSON 不嵌入自身 SHA；由运行报告记录它的外部 SHA。首次读取 val 后禁止改动。
3. 三个训练臂严格固定：
   - `A_KL_ONLY_CONTROL`: `L_total=L_base`
   - `B_PAIR_LAMBDA_1_CONTROL`: `L_total=L_base+1.0*L_pair`
   - `C_PAIR_LAMBDA_0P25_PRIMARY`: `L_total=L_base+14.280209646136903*L_pair`
4. 损失定义：
   - `L_base=0.5*(KL(y_g||p_g)+KL(y_p||p_p))`
   - `D_swap=0.5*(KL(y_g||p_p)+KL(y_p||p_g))`
   - `L_pair=relu(JS_nats(y_g,y_p)+L_base-D_swap)`
5. A/B 为对照，C 为唯一 primary promotion arm。若 C 不通过全部 gate，结论必须是 `NO_PROMOTION`；即使 A/B 某项更好也不得事后改 winner。
6. 不添加第四臂，不搜索 lambda/rank/temp/lr/epoch/margin，不更换 teacher，不看 test。

## 二、训练合同

1. 新增：
   - `evaluation/train_pairwise_prompt_selector.py`
   - `evaluation/tests/test_train_pairwise_prompt_selector.py`
2. split 沿用 source_image_sha256 grouped split：train=318 records/159 images，val=76 records/38 images；每 image 恰有 grounding+pointing 两条。
3. train/val 可读；test 78 records/39 images 的 NPZ 绝对不能打开。用 loader 访问日志、spy loader、损坏 test NPZ 三种测试证明 test 未读且 train/val 结果与 hash 不变。
4. 三 arm × 三 seed；每个 seed 各 arm 使用完全相同 epoch-0 Wt/Wv 和完全相同每 epoch image-pair batch 顺序。
5. 固定优化：CPU-only、float32、AdamW、epochs=30、lr=1e-3、weight_decay=0、batch_size_images=4、grad_clip_norm=1.0、deterministic CPU、`torch.set_num_threads(1)`；不使用 val 调超参。明确说明这是约45.9万参数小头训练，不是4B训练。
6. 每 epoch 记录 train/val 的 L_base、L_pair、L_total、hinge_active_fraction、grad_norm_before_clip、batch count；epoch 0 必须在更新前评估。
7. best epoch 只按该 arm 的 val L_total 最小值选择，tie 取更早 epoch。保存 epoch0 权重 hash、best checkpoint、final checkpoint及每 epoch metrics；不保存每 epoch 完整权重。相同 seed 各 arm epoch0 hash 必须相同。
8. 每次正式 run 前后校验 prereg、draft、config、index SHA；运行后必须证明 prereg 和 draft 字节未变。

## 三、验证与统计合同

只使用 val，禁止 test。尽可能复用：

- `evaluation/frozen_scorer_reduction_ablation.py`
- `evaluation/prompt_identifiability_diagnostic.py` v2
- `evaluation/train_prompt_projection.py`
- `evaluation/pair_loss_feasibility_audit.py`

禁止复制第二套 scorer、split、donor 或统计实现。

对每 arm/seed 的 best checkpoint，在 val 上计算：

1. correct-prompt teacher alignment：record-level Spearman、NDCG-full。
2. same-image-other-task Prompt 敏感性：correct prompt metric − 同图另一任务 prompt metric；grounding/pointing 两条先按 image 平均，再以 image 为独立单位。
3. cross-image-same-task shuffled discrimination：split-local、固定 seed、image-level deterministic derangement、无自配、同 task donor；correct prompt metric − donor prompt metric。
4. exact_prompt_invariance_fraction（`np.array_equal`）及 tolerance `1e-9` 版本。
5. 每个 primary delta（same-image/cross-image × Spearman/NDCG）：2000 image-level bootstrap 95% CI、10000 双侧 sign-flip permutation p、Cohen's dz；三 seed 先对同 image delta 取 median 后做统计，并要求 delta sign 在至少2/3 seed为正。
6. Holm-Bonferroni family 固定为 3 arms × 2 controls × 2 primary metrics = 12 tests，不得事后缩小。
7. teacher alignment gate：median-seed val correct Spearman ≥ `0.0564749736−0.01`，且 NDCG-full ≥ `0.8499415696−0.01`。
8. Gate：
   - A_prompt_sensitivity：same-image 两 primary delta>0、CI lower>0、`|dz|>=0.2`、Holm p<0.05、>=2/3 seed 正。
   - B_shuffled_discrimination：cross-image 两 primary 同上。
   - C_collapse_removed：每个 seed exact_prompt_invariance_fraction<=0.05。
   - D_teacher_alignment：满足上述两个阈值。
   - E_generalization：>=2/3 seed best_epoch>0 且 best val L_total < epoch0 val L_total。
   - F_hardware_contract：query仍`1x128`、FPGA scorer仍`Gx128 dot`、无 token-axis max/softmax/exp。
9. 只有 arm C 同时通过 A-F 才 `PROMOTION_PASS`，否则 `NO_PROMOTION`。A/B 只报告对照结果。
10. 不读取、评估或写出 internal test 指标。

## 四、证据与输出

新增：

- `configs/prompt_mean_pairwise_retrain.preregistered.json`
- `evaluation/train_pairwise_prompt_selector.py`
- `evaluation/tests/test_train_pairwise_prompt_selector.py`
- `artifacts/token_selector/pairwise_retrain/`（9组 best/final checkpoint、metadata/hash）
- `reports/token_selector/2026-10-03-pairwise-retrain-preregistration.json`
- `reports/token_selector/2026-10-03-pairwise-retrain-epochs.csv`
- `reports/token_selector/2026-10-03-pairwise-retrain-val-samples.csv`
- `reports/token_selector/2026-10-03-pairwise-retrain-train-val.{md,json}`
- `reports/token_selector/logs/2026-10-03-pairwise-retrain-{run,focused-tests,full-tests}.log`
- `reports/token_selector/logs/2026-10-03-pairwise-retrain-harness-result.json`
- `/tmp/robobrain_harness_result.json`（若不持久，以 durable copy 为准）
- `CURRENT_STATUS.md` 顶部新增完成条目。

所有 JSON/CSV 保存原始数值，不只平均值。报告必须列：每 arm/seed epoch0、best、final loss，best epoch，四组 primary delta/CI/p/dz/Holm，collapse、alignment、A-F gate，winner 固定判决，checkpoint/hash、输入/源码/输出 SHA，`test_npz_opened=[]`，训练总耗时、峰值 RSS；明确不含4B/GPU。

## 五、测试与安全

1. 聚焦测试至少覆盖：三 arm loss 数学、同 seed arm 的 epoch0 hash 相同、deterministic batch order 相同、best epoch tie取早、val-only selection、test NPZ spy/corrupt isolation、prereg/draft immutable、Holm m=12、image-level aggregation/derangement/no self、winner rule仅 C、checkpoint round-trip。
2. 运行聚焦测试和 `python -m unittest discover -s evaluation/tests` 全量测试。
3. 禁止 kill/暂停/修改其他用户进程；本任务不需要 GPU。
4. 若合同无法无歧义实现，停止训练并写 BLOCKED 合同报告；不要自行改门槛/臂/损失。
5. 三臂三 seed 全部跑满30 epochs，不缩减 epochs/seed/sample。

最终 stdout 只输出一行 JSON，字段：`status, prereg_sha256, draft_sha256_unchanged, arms, seeds, train_records, val_records, test_npz_opened, focused_tests, full_tests, runtime_seconds, peak_rss_mib, arm_results, primary_arm_gates, decision, files_created, limitations, next_decision`。不要输出长篇过程。
