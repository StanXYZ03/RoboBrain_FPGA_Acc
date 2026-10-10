# edge576 CV-Bench r2 预注册作废报告（2026-10-07，PREEXECUTION_SUPERSEDED）

- 状态：`PREEXECUTION_SUPERSEDED`
- 对象：`configs/2026-10-07-edge576-cvbench-three-arm-development-r2.preregistered.json`
  SHA `8f265557bfb86998135b6985023e136d7f397eb8e06eb64551a97308cc958d09`
- 机器结果：`reports/token_selector/2026-10-07-edge576-cvbench-r2-preexecution-superseded.json`
- 口径：CPU-only、只读；`gpu_runs=0`、`executed=false`、未加载模型权重、未联网、未训练、
  未安装依赖；r2 全部字节保持不变。

## 唯一作废理由

**r2 没有 pin 新增的 adapter / smoke 执行层。** r2 只 pin 了 40 项冻结依赖与既有
`conservative_l2_guarded_rerank_v2` 机制链，未 pin：

1. 新增 runner 契约 `evaluation/edge576_cvbench_runner_contract.py`；
2. 新增三臂 runner `evaluation/run_edge576_cvbench_three_arm.py`；
3. 新增 runner 聚焦测试 `evaluation/tests/test_run_edge576_cvbench_three_arm.py`；
4. runner 适配报告 `.../2026-10-07-edge576-cvbench-three-arm-runner-adaptation.{json,md}`；
5. 新增独立单样本 smoke 入口 `evaluation/run_edge576_cvbench_three_arm_smoke.py`；
6. 新增 smoke 聚焦测试 `evaluation/tests/test_run_edge576_cvbench_three_arm_smoke.py`。

因此 r2 所声明的可执行面与实际将执行的字节已经不一致；r2 **从未执行**（`gpu_runs=0`、
`arm_runs=0`），作废不损失任何结果。r3 在同一三臂/指标/门槛之上补齐上述 pin。

## 未 pin 的新增执行层（现场 SHA256）

| 文件 | SHA256 | 作用 |
| --- | --- | --- |
| `evaluation/edge576_cvbench_runner_contract.py` | `fdea7df6…90db` | 共享 edge576 运行契约与 fail-closed 模型身份核验 |
| `evaluation/run_edge576_cvbench_three_arm.py` | `b1fb2ca1…3eb85` | 三臂 runner 适配（预检 + 续跑 + 热身） |
| `evaluation/tests/test_run_edge576_cvbench_three_arm.py` | `6d4f5135…0fa55` | runner 适配聚焦 CPU 测试 |
| `.../2026-10-07-edge576-cvbench-three-arm-runner-adaptation.json` | `485947b8…3784` | runner 适配机器报告 |
| `.../2026-10-07-edge576-cvbench-three-arm-runner-adaptation.md` | `6e496348…655fe` | runner 适配中文报告 |
| `evaluation/run_edge576_cvbench_three_arm_smoke.py` | `3c956d7e…c255` | 独立单样本 smoke 入口 |
| `evaluation/tests/test_run_edge576_cvbench_three_arm_smoke.py` | `458a4510…1068` | smoke 聚焦 CPU 测试 |

## r2 的既有修正仍然有效（不是作废理由）

- **A（第三臂已预注册）**：r2 将第三臂冻结为**现有**
  `conservative_l2_guarded_rerank_v2`（`is_new_mechanism=false`），r3 不变继承。
- **B（诚实记录完整性）**：r2 只要求 2638 个唯一完整输出，`parsed_answer` 允许 `null`、
  解析失败计错并单独报告，无 100% 解析门；r3 不变继承。
- **C（审计器 `--model` 已修复）**：r2 使用修正后的审计器与复审报告；r3 不变继承。

以上三点是 r2 相对 r1 的有效修正，**不构成**本次作废理由，r3 也不修改它们。

## r2 未改内容（r3 原样冻结）

- 三臂：`dense_edge576` / `corrected_l2_integer_b1_k96` / `v2_edge576_b1_k96`；
- 主指标：官方 `cv_bench_accuracy`（`((ADE20K+COCO)/2+Omni3D)/2`）；
- 成功门：同输入 Dense@edge576 保留率 `>=0.90`、相对修正整数 L2 的
  `task::source_dataset` 分层配对 bootstrap（seed 20261007）95% CI 下界 `>=0`、
  K96 布局、2638 唯一完整输出、无标签泄漏；
- 三条 full2638 输出路径不变。

## 取代关系

- 取代者：`configs/2026-10-07-edge576-cvbench-three-arm-development-r3.preregistered.json`
  （生成器 `scripts/preregister_edge576_cvbench_r3_20261007.py`）。
- r2 配置文件、预注册报告 JSON/MD、生成器、聚焦测试字节全部保留，不得重新生成，也不得执行。
