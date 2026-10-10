# Harness prompt (verbatim) — 2026-10-03 spatial prompt-contrastive train/val

This file preserves the exact executing harness prompt that authorized and
defined the `2026-10-03-spatial-prompt-contrastive-trainval` experiment.  It is
written to disk **before any validation NPZ is read** for this experiment, as
required by the execution contract.

```
你是执行工程师。实施下一项预注册实验，不要提出替代方案，不要输出思考过程。项目根目录：/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102；Python：/home/zhaoxiangyu/conda-envs/robobrain25/bin/python。

研究目的：在不改变 prompt_mean_vector 硬件结构的前提下，解决 spatial-student 唯一未通过的同图错误 Prompt 判别 G2，同时保持已经证实为正的部署 retained-mass 效用。

严格合同：
1. 在读取任何 val NPZ 前，先把本提示原文保存为 reports/token_selector/prompts/2026-10-03-spatial-prompt-contrastive-trainval-harness-prompt.md，并创建 configs/spatial_prompt_contrastive_trainval.preregistered.json 与对应 preregistration JSON，固定路径/SHA、指标、统计、门槛。internal test NPZ 禁止解析、哈希、打开。
2. 禁止原图、4B 权重、GPU、联网、重抽特征；CPU float32。不得修改既有训练脚本/报告，尤其不改 evaluation/train_spatial_prompt_selector.py；其 G3 符号错误只作为已知历史问题引用。
3. 新脚本 evaluation/train_spatial_prompt_contrastive_selector.py；复用既有 split/loader/空间 target/student scorer/checkpoint/统计/anchor/combine/selected-set NDCG。架构冻结 Wt[128,2560]、Wv[128,1024]、prompt_mean_vector、[1,128] query、Gx128 dot。
4. 两臂、seeds=20261003/4/5、30 epochs、AdamW lr=1e-3 wd=0、batch=4 image pairs、clip=1、同 seed 两臂 epoch0 与 batch 顺序完全一致：
 A_SPATIAL_KL_CONTROL：base=(KL(tg||pg_correct)+KL(tp||pp_correct))/2。
 B_SPATIAL_KL_CONTRASTIVE：total=base+contrastive；contrastive=(softplus(KL_g_correct-KL_g_wrong)+softplus(KL_p_correct-KL_p_wrong))/2。wrong 对 grounding 使用同图 pointing prompt，反之亦然。所有 KL 都用 softmax(score/0.1)，无 T²；contrastive 权重固定1.0、margin=0、softplus temperature=1，不得调参。
 A 按 val base 选 best；B 按 val total 选 best，同时报告 base/contrastive/total。
5. 验证仍用76 records/38 images，三 seed 对每图 delta 取中位数，image-level bootstrap=2000、sign-flip=10000、dz；primary Holm family恰好5项：
 P1 B-A 的(correct-same) spatial Spearman gap；
 P2 B-A 的(correct-same) full-NDCG gap；
 P3 B 绝对 correct-same spatial Spearman；
 P4 B 绝对 correct-same full-NDCG；
 P5 B deployed selector - L2+anchor retained spatial-target mass @K48。P5必须按 selector减baseline正确符号计算并有单测防回归。
6. Gates：
 G1 correct spatial alignment非劣：B-A Spearman/NDCG mean>=-0.02，95% CI lower>=-0.05，且>=2/3 seed mean>=-0.02（描述性非劣，不进入Holm）。
 G2 P1/P2均 mean>0、CI lower>0、dz>=0.2、Holm<0.05、>=2/3 seeds positive。
 G3 P3/P4同样全部通过。
 G4 P5同样全部通过。
 G5 exact prompt invariance <=0.05 every B seed。
 G6 >=2/3 B seeds best_epoch>0 且 best val total低于epoch0。
 G7硬件合同不变。
 仅全部通过才 PROMPT_CONTRASTIVE_STUDENT_ESTABLISHED，否则 PROMPT_CONTRASTIVE_STUDENT_NOT_ESTABLISHED。
7. 既有 val 已用于开发，本实验只能作为 development validation；不得声称最终精度。正式结果后不得 posthoc 改门槛。
8. 测试必须覆盖：两臂 target/loss 精确参考、wrong prompt交换、同初始化/批次、val-only checkpoint、五项Holm、G1-G7、P5正确符号、测试集隔离、prereg不变、checkpoint/report schema。
9. 输出 stem 统一 2026-10-03-spatial-prompt-contrastive-trainval：
 reports/token_selector/<stem>-epochs.csv
 reports/token_selector/<stem>-val-samples.csv
 reports/token_selector/<stem>.json/.md
 artifacts/token_selector/spatial_prompt_contrastive_trainval/<arm>/<seed>/
 reports/token_selector/logs/<stem>-run/focused-tests/full-tests/harness-result
 并更新 CURRENT_STATUS.md 顶部。
10. 运行聚焦和全量 unittest。最终只输出不超过25行JSON：decision、五项primary值、G1-G7、best epochs、isolation、tests、files、limitations、next_decision。
```
