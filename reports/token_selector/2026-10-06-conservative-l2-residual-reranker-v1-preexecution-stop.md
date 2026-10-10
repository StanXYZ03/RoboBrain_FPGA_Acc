# 保守修正 L2 主干 + 有界整数任务残差重排 v1 执行前 STOP 报告

- 机制：`conservative_l2_residual_rerank_v1`
- 状态：**`PREEXECUTION_DESIGN_STOP`**（任何 GPU 运行之前）
- 口径：**仅发展性审计**；不是精度结论、不是可部署性、不是 FPGA 时延/加速主张。
- 执行：`executed=false`、`gpu_runs=0`、v1 新臂输出**从未生成**。

## 1. 阻断缺陷（独立复核确认）

v1 预注册的交换排序量为：

`(semantic_q15[j] - L2_q15[j]) - (semantic_q15[i] - L2_q15[i])`

对同区域候选 `j` 与被选 `i`，展开为 `semantic_gain + l2_loss`，即它**奖励 L2 损失**，
与“保护修正整数 L2”的意图符号相反。

### 精确反例

| 量 | 值 |
| --- | --- |
| `semantic_i` | `10000` |
| `semantic_j` | `10000` |
| `L2_i` | `30000` |
| `L2_j` | `1000` |
| `semantic_gain` | `0` |
| `l2_loss = L2_i - L2_j` | `29000` |
| v1 gain = `semantic_gain + l2_loss` | `29000 > 0`（**错误接受**） |
| 修正 `legal_gain = semantic_gain - l2_loss` | `-29000 < 0`（正确拒绝） |

v1 会接受一个“无语义改进、且把修正 L2 从 30000 降到 1000”的候选。

## 2. 处置

- **不修补 v1、不运行 v1**；v1 源码/配置/报告保持**字节不变**。
- 新增独立、加性的修正族 `conservative_l2_guarded_rerank_v2`：
  - `utility_q16 = semantic_q15 + L2_q15`（`[0, 65534]`，加宽整数）；
  - `legal_gain = semantic_gain - l2_loss = utility_added - utility_removed`；
  - 仅接受 `legal_gain > 0`；其余合同（anchor 冻结、同区域一对一、每区域计数不变、
    `local_candidate_cap=16`、全局 `S=16`、K=96、87 real + 9 context、修正整数 L2 池化）不变。

## 3. v1 冻结哈希（现场复核，字节不变）

| 文件 | SHA256 |
| --- | --- |
| `mechanism_core` | `64b699757fa9ea04eb12438509b5679644a263eb4783139e50d945e196fb5bf4` |
| `integration` | `fc9c130a11fc11605398b8177f2f2f2d13126aa270f95f5ae14ad7867a555026` |
| `runner` | `ffd41f46c776eabc441c7d90b957c3f664f3510f3d21ce797415fbd70c27cb1c` |
| `summarizer` | `b7f71603fbe006715d6a8ee662730f79efe6fd2fc95ab23cc5525b49ba0f1c79` |
| `test_mechanism` | `5712c95c463525d0a0db6c03314099bb7a935782a254880c86dfc3f4ad8a1c8e` |
| `test_integration` | `a116a132a626f6d9e0d2d5769f3620fe9e521e9411246b07082acd7c2370db04` |
| `test_runner` | `7f8b018ddc1536c770f8e6649db5882ea2d1359475bccdb2173ff0e9845ca806` |
| `test_summarizer` | `9f92dfca40b010a6dcdbe76e9648f0c27335a599703a501ed21cfb35d94e5545` |
| `preregistered_config` | `6b158d22d52f1df66eba75b1b7bb4c7c614793681152c1501c43079fecd97512` |
| `prereg_report_json` | `1a936b8d8eaa1fbc960b60f401766ce608b620fd45df5524f6f8ae0db5de1c3c` |
| `prereg_report_md` | `67fdd0329ae79f0febb85dd2e9ca5fb66b1a66385660fe357fbc30c5a98356cf` |

- v1 预注册 config：`configs/2026-10-05-conservative-l2-residual-reranker-full2638-development.preregistered.json`
  （SHA `6b158d22d52f1df66eba75b1b7bb4c7c614793681152c1501c43079fecd97512`）
- v1 预注册报告：`reports/token_selector/2026-10-05-conservative-l2-residual-reranker-prereg.json`
  （SHA `1a936b8d8eaa1fbc960b60f401766ce608b620fd45df5524f6f8ae0db5de1c3c`）

## 4. 下一道命令

**无**。禁止运行 v1、禁止修补 v1、禁止生成 v1 新臂输出。后续只允许 v2 的独立预注册路径。

## 5. 限制

- 本报告是执行前设计 STOP，不是对 v1 的精度/性能结论。
- v1 full2638 输出不得生成；看到 full2638 后也不得修补 v1。
- full2638 是反复观测过的发展集，不是未触碰确认集。
- Selector 路径是软件桥；不做 FPGA/板卡/时延/加速主张。
