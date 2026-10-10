# edge576 CV-Bench 三臂未来预注册（2026-10-07）

- 状态：`PREREGISTERED_NOT_EXECUTED`（`executed=false`、`gpu_runs=0`、仅 CPU/只读、未加载模型权重）
- edge576 manifest：`datasets/2026-10-07-cv-bench-edge576-full2638.jsonl` SHA `b5355ba1636011b4af101ace16c170b061ecf44a69e5c75fe2a64f4288c7dd1f`
- 源冻结 full manifest：SHA `22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91`
- 样本数：2638（唯一 2638）；全部 `G<=576`：True
- G：min/max = 64/576；分桶 <=96=143、97-256=781、257-512=706、513-576=1008

## 三臂共享合同（已锁定）

- 同一 edge576 manifest 与同一 Prompt；BF16/SDPA；`cuda:0`；
- `do_sample=false`（greedy）、`max_new_tokens=16`；
- 处理参数 `min_pixels=65536`、`max_pixels=589824`、保持长宽比、`patch_size=16`、`merge_size=2`；
- 每臂必须断言 processor 的 `image_grid_thw` 与 manifest 记录逐样本一致，否则 fail closed；禁止使用原生分辨率预算。

## 三臂

1. `dense_edge576`（质量分母）；
2. `corrected_l2_integer_b1_k96`（B1 after vision block 0、K96、grid3、cap16、`G<=96` identity bypass）——冻结整数 L2 控制臂；
3. `next_candidate_b1_k96`（同切点、同 K、单变量；机制待独立策略审查后选定，`mechanism_id=null`）。

## 成功门

- 质量保留率 = `selector_cv_bench_accuracy / dense_edge576_cv_bench_accuracy` `>= 0.90`；
- 相对同切点/同 K 修正整数 L2，按 `task::source_dataset` 分层配对 bootstrap（5000 次、seed 20261007、95%）CI 下界 `>=0`；
- 主指标仅 `cv_bench_accuracy`，诚实解析门使用显式 `parsed_answer_key`；
- 无标签泄漏；2638 全集完整且可解析。

## 口径与边界

- full2638 是反复观测的发展集，通过只允许一次新的未触碰外部基准确认；
- 2026-10-06 原生分辨率 `DEVELOPMENT_STOP` 不追溯修改；
- 原生高分辨率仅作压力测试，不作边缘主档位晋升门；
- 本预注册不代表任何精度/部署/加速/FPGA 时延主张，也未启动任何模型推理。
