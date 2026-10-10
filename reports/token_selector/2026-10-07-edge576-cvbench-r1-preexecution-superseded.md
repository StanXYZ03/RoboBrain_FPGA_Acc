# edge576 CV-Bench r1 预注册作废报告（2026-10-07）

- 状态：`PREEXECUTION_SUPERSEDED`（`executed=false`、`gpu_runs=0`、仅 CPU/只读）
- 结论：**r1 从未执行，且不得运行**；其全部文件字节保持不变，作为未执行设计缺陷证据。
- 被作废预注册：`configs/2026-10-07-edge576-cvbench-three-arm-development.preregistered.json`
  SHA256 `8958cbd6f61589217b73e983e0f57257f0cf9bc951c5adbd8506df5d6156b3f3`
- 配套报告：`reports/token_selector/2026-10-07-edge576-cvbench-prereg.json`
  SHA `c728c312…`、`reports/token_selector/2026-10-07-edge576-cvbench-prereg.md`
  SHA `37252dad…`
- 生成器（同样保留字节不变）：`scripts/preregister_edge576_cvbench_20261007.py`
  SHA `c0177675…`

## 三项设计缺陷

| 编号 | 缺陷 | r2 修正 |
|---|---|---|
| A | 第三臂 `mechanism_id=null`，状态 `TO_BE_SELECTED…`，**不是已预注册候选** | 冻结为现有 `conservative_l2_guarded_rerank_v2`，`mechanism_id` 明确、`is_new_mechanism=false`，pin 其完整机制链，不发明新机制 |
| B | `success_gates.record_completeness` 写「2638 全部可解析」，与诚实解析口径冲突（Dense 也会有不可解析输出） | 只要求「2638 个唯一完整输出」；`parsed_answer` 可 `null`，解析失败按错误计并报告成功/失败数/率，**不设 100% 解析门** |
| C | 审计器 CLI 接受 `--model` 却未传入 `audit()`，`verify_processor` 硬用 `DEFAULT_MODEL` | `audit()` 显式接受 `model_path`；`main` 传入 `--model`；报告记录实际处理器路径 |

## 审计器 pin 说明

- r1 当时 pin 的审计器 SHA 为 `45dcd04976e5c5c8acaf232989f219b1817b5a13fb87bfbe27dd4cbfe81ea683`。
- 修复缺陷 C 后，审计器实际 SHA 变为
  `a999fe0a48b962104b9991df1c62329789a2164f530fe977767dd95ddbad352b`，因此 r1 配置中的
  该 pin 已不再匹配磁盘文件。
- r1 已作废且**不得重新生成或执行**；重新审计改用修正后的审计器，并单独记录
  `reports/token_selector/2026-10-07-edge576-cv-bench-manifest-audit-r2.json`。

## r1 自检现状（预期）

- r1 聚焦测试模块 `evaluation/tests/test_edge576_cvbench_prereg.py`（SHA
  `5eb3b8c4…`）字节保留不变。
- 由于修复缺陷 C 后审计器 SHA 变化，该模块的 `test_every_pin_reverifies`
  现在会**预期失败 1 项**（r1 配置的 auditor pin `45dcd049…` 已过期）；其余 r1 断言仍成立。
- 这一预期失败正是 r1 被作废、不可执行的正面证据，**不是** r2 或修正审计器聚焦测试的回归。

## 未改动的冻结输入

- edge576 manifest：`b5355ba1636011b4af101ace16c170b061ecf44a69e5c75fe2a64f4288c7dd1f`
- edge576 summary：`492c82958804bda54ec36c607bc9b8e2222ce2ef3d7deafc89b64517176b1a6b`
- 源冻结 full manifest：`22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91`
- 模型身份清单：`bbfcda188605f54a9c7fea83e7fc2a46eca8ac635c086d23861b294c9950f8e3`
- `cv_bench_metric_v2.py`：`2a2f2004834803927f34114a916e0e867a6091ada0e66d05be6ee109183a73a1`

## 后续

- 正确预注册见 r2：`configs/2026-10-07-edge576-cvbench-three-arm-development-r2.preregistered.json`，
  生成器 `scripts/preregister_edge576_cvbench_r2_20261007.py`。
- 口径：CPU-only、只读；无 GPU、无模型权重、无网络、无训练、无依赖安装；不代表任何精度/
  部署/加速/FPGA 时延主张；full2638 仍是发展集，2026-10-06 原生分辨率 `DEVELOPMENT_STOP`
  不追溯改写。
