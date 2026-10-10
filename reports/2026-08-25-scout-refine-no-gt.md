# 无真值 Scout–Refine 规则区域选择器实验

日期：2026-08-25  
状态：`256_INFERENCES_COMPLETE / ACCURACY_GATE_FAIL / PL_GATE_FAIL`

## 1. 方法

第一遍用 12.5% 请求分辨率输出预测点/框；第二遍不读取标注，Pointing 在预测点周围
取固定方形区域，Grounding 对预测框按固定比例扩展，再将第二遍坐标映射回原图。
`task hybrid` 只依据公开任务协议选参数：Pointing=tight，Grounding=wide。
总 Token、MAC proxy 和延迟均包含两遍，没有隐藏 scout 开销。

## 2. 完整结果

| 配置 | 实际 Token | Encoder MAC | LLM prefill MAC | PCK@0.1 | PCK@0.2 | Mean IoU | Acc@0.5 | Point p50/s | Ground p50/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform@1.0 | 1.0000 | 1.0000 | 1.0000 | 0.5625 | 0.7812 | 0.8090 | 0.9062 | 0.2934 | 0.5333 |
| uniform@0.25 | 0.2389 | 0.2180 | 0.2360 | 0.3438 | 0.6875 | 0.6274 | 0.6875 | 0.2860 | 0.5271 |
| uniform@0.50 | 0.4593 | 0.4308 | 0.4554 | 0.4688 | 0.7812 | 0.7415 | 0.8438 | 0.2916 | 0.5299 |
| oracle@0.25 | 0.2462 | 0.2249 | 0.2433 | 0.5312 | 0.8125 | 0.7545 | 0.8750 | 0.2837 | 0.5197 |
| scout-refine low/tight | 0.2087 | 0.1871 | 0.2057 | 0.3438 | 0.6250 | 0.5912 | 0.6562 | 0.5712 | 1.0550 |
| scout-refine low/wide | 0.2020 | 0.1811 | 0.1991 | 0.3750 | 0.5938 | 0.4967 | 0.6250 | 0.5692 | 1.0544 |
| scout-refine high/tight | 0.3576 | 0.3252 | 0.3531 | 0.5000 | 0.7500 | 0.6095 | 0.6875 | 0.5700 | 1.0548 |
| scout-refine high/wide | 0.3353 | 0.3043 | 0.3310 | 0.4688 | 0.7188 | 0.6790 | 0.7812 | 0.5692 | 1.0499 |
| scout-refine task hybrid | 0.3494 | 0.3174 | 0.3449 | 0.5000 | 0.7500 | 0.6790 | 0.7812 | 0.5700 | 1.0499 |

## 3. 门控结论

task hybrid 实际使用 34.9% Token。相对 uniform@0.50，
Token 减少 11.0%（绝对保留率），PCK@0.1 +0.0312，
但 Mean IoU -0.0625、Acc@0.5 -0.0625。

它形成一个可复现的低 Token Pareto 点，但没有满足完整输入的精度门，也没有统一超过
matched uniform。两遍 A6000 中位延迟约为单遍的两倍；在 PS 上还会重复文本 prefill
与 decode。因此该方案只保留为软件基线，不接入 PL。

按既有 ZCU102 PS 阶段时间一阶投影，并把文本 prefill/decode 固定成本支付两次，
task hybrid 约为 675.6 s/图、2.51×、
0.001480 Hz；这只是工作量投影，不是板端实测。

## 4. 负结果带来的设计约束

- Pointing 与 Grounding 的最佳固定裁剪范围不同，固定统一参数不可行。
- 极低分辨率 scout 的定位误差会把第二遍高分辨率计算集中到错误区域。
- 裁剪重建局部坐标系，无法天然保持原图 MRoPE/DeepStack 的全局空间参考。
- 下一候选应是单遍浅层特征选择：保留原 position ID，以规则 2×2/4×4 块压缩，
  同时保留分区全局锚点；不能再依赖完整 scout decode。

## 5. 复现

- 原始结果：`evaluation/results/scout_refine_full`
- 派生 hybrid：`evaluation/results/scout_refine_full/task_profile_hybrid.jsonl`
- 机器汇总：`reports/2026-08-25-scout-refine-summary.json`
- 运行器：`evaluation/run_hf_scout_refine.py`
- 汇总器：`evaluation/build_scout_refine_report.py`
