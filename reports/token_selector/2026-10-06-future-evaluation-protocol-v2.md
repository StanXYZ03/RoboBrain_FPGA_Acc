# 未来评测协议 v2（生命周期安全 / 诚实解析门 / 模型身份固定）

- 日期：2026-10-06
- 标记：`PROTOCOL_V2_READY`
- 范围：**仅面向未来**；不追溯修改任何冻结产物。
- 口径：CPU-only、只读；`gpu_runs=0`、未加载模型权重、未联网、未训练、未安装依赖。

## 1. 两个协议缺口

### `parse_gate_key_alias`（protocol_defect_fixed_for_future_only）

冻结 scorer 用 correct_key.replace('_correct','_answer') 推导解析键；当 correct_key='correct' 时该推导与布尔正确性字段同名，使解析门恒为 1.0，而诚实失败为 dense=40、修正整数 L2=42、prompt=42、v2=43。

### `model_weight_shards_unpinned`（provenance_gap_fixed_for_future_only）

冻结 v2 预注册固定了代码/checkpoint/数据，但只记录 model_path/model_label，未固定本地模型权重分片；新增模型身份清单以流式 SHA256 记录全部分片与配置文件。

## 2. 四臂诚实解析计数（只读重算）

| 臂 | 样本 | 解析成功 | 解析失败 | 诚实解析率 | 冻结 scorer 报告值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 2638 | 2598 | 40 | 0.9848369977 | 1.0000000000 |
| corrected_l2_integer | 2638 | 2596 | 42 | 0.9840788476 | 1.0000000000 |
| prompt_plus_corrected_l2_integer | 2638 | 2596 | 42 | 0.9840788476 | 1.0000000000 |
| v2 | 2638 | 2595 | 43 | 0.9836997726 | 1.0000000000 |

> 冻结 scorer 报告值恒为 1.0，是键别名缺陷；诚实失败为 dense=40、修正整数 L2=42、prompt=42、v2=43。

## 3. 主指标只读重算（未改变）

| 臂 | cv_bench_accuracy | 冻结值 | 未改变 |
| --- | ---: | ---: | --- |
| dense | 0.8727298774444869 | 0.8727298774444869 | 是 |
| corrected_l2_integer | 0.7668517264725796 | 0.7668517264725796 | 是 |
| prompt_plus_corrected_l2_integer | 0.7773617325561999 | 0.7773617325561999 | 是 |
| v2 | 0.7837404943432142 | 0.7837404943432142 | 是 |

- 主指标全部未改变：`True`；发展判定保持 **`DEVELOPMENT_STOP`**。
- 本报告不追溯修改 v2 主分，也不改变 `DEVELOPMENT_STOP`。

## 4. 协议固定与生命周期

- 协议 config：`configs/evaluation_protocol_v2.json`（`aeb963f1887b22c206531fb9902d9629e7b26aefa9529244cd8bdf6112c885ed`）
- pin `cv_bench_metric_v2_source`：`evaluation/cv_bench_metric_v2.py` 匹配 `True`
- pin `model_identity_manifest`：`artifacts/provenance/robobrain25-4b-model-identity.json` 匹配 `True`
- pin `model_identity_builder`：`scripts/build_model_identity_manifest.py` 匹配 `True`
- 生命周期状态：`PREREGISTERED_NOT_EXECUTED`, `EXECUTED_UNAUDITED`, `AUDITED_STOP`, `AUDITED_GO_EXTERNAL_VALIDATION`
- 测量类别：`board_measurement`, `development`, `fpga_estimate`, `software_latency`, `untouched_confirmation`

## 5. 模型身份清单

- 路径：`artifacts/provenance/robobrain25-4b-model-identity.json`（`bbfcda188605f54a9c7fea83e7fc2a46eca8ac635c086d23861b294c9950f8e3`）
- 模型目录：`/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B`；文件数：10

## 6. 限制

- 本报告为未来协议，不追溯修改任何冻结 v2 主分或 DEVELOPMENT_STOP 判定。
- 诚实解析计数为只读重算，不改变官方主指标，也不构成新的精度主张。
- 本轮 CPU-only：gpu_runs=0、未加载模型权重、未联网、未训练、未安装依赖。
- 模型身份清单只是本地文件哈希，不是模型质量/精度/部署主张。

## 7. 下一步允许的动作

在新协议下独立预注册新候选，按生命周期安全路径规则执行，并固定模型身份清单；在任何外部验证或硬件主张前先完成只读审计。

