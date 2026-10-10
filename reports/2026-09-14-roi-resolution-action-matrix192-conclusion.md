# ROI分辨率动作矩阵192图结论

日期：2026-09-14。状态：`P2_REPRESENTATION_CONDITIONALLY_SUPPORTED / NAIVE_REENCODE_BUDGET_FAIL / DEPLOYABLE_SELECTOR_UNVERIFIED`。

本轮使用192张已揭示COCO开发图，每张同时构造Grounding和Pointing任务，共384个任务记录。Dense、真值ROI-384和真值ROI-512各执行3次确定性推理，共3456条纳入比较的原始记录。真值ROI只用于表示与动作上界诊断，不是可部署区域选择器。

| 表示 | Dense Token比 | Grounding Mean IoU | 相对Dense | Pointing PCK@0.1 | 相对Dense |
|---|---:|---:|---:|---:|---:|
| Dense | 1.000 | 0.8103 | 0 | 0.6458 | 0 |
| ROI-384 | 0.342 | 0.7890 | -0.0214 [-0.0471,+0.0034] | 0.5990 | -0.0469 [-0.1094,+0.0156] |
| ROI-512 | 0.609 | 0.8164 | +0.0061 [-0.0154,+0.0275] | 0.6250 | -0.0208 [-0.0833,+0.0365] |

## 动作互补性

- Grounding中512优于384为117/192、相同4/192、劣于71/192；Pointing中分别为98/192、7/192、87/192。固定单一尺度不能覆盖所有图像。
- 读取真值逐图选择尺度的one-shot上界：Grounding Mean IoU 0.8452、Token比0.497；Pointing PCK@0.1 0.6615、Token比0.478。该结果证明存在动作互补，但不证明可以预测正确动作。
- 若先编码ROI-384再决定是否重编码ROI-512，质量上界相同，但累计Token比分别为Grounding 0.696、Pointing 0.653，超过当前不高于Dense 50%的设计目标。

## 工程决策

1. P2表示假设“有条件支持”：低/高ROI尺度组合存在低于50%最终Token且接近或超过Dense点估计的真值上界。
2. 拒绝把朴素`ROI-384 -> ROI-512`完整重编码作为最终主动细化实现；其重复视觉计算抵消了大部分Token收益。
3. 下一步只评估不读取真值的低成本尺度收益预测器，特征必须来自输入图像、任务文本或可复用的低成本Scout状态。先做按图像分组的交叉验证，不能在同图双任务间泄漏。
4. 即使尺度预测通过，区域定位仍是独立未解决问题；当前真值ROI结果不能替代可部署Scout的定位精度和开销验证。
5. 当前各配置跨运行GPU负载不同，本报告不使用延迟差作为结论。正式延迟需冻结策略后在无竞争GPU条件下一次性配对重测。

关键数据：

- `reports/2026-09-14-roi-resolution-action-matrix192-quality.{json,md}`
- `reports/2026-09-14-roi-resolution-oracle192.{json,md}`
- `evaluation/results/global_roi_bf16_representation_gate32_stratified_20260912.jsonl`
- `evaluation/results/global_roi_bf16_roi_m0500_s512_stratified32_exactwarmup_20260914.jsonl`

证据边界：以上均为已揭示开发集结果，没有访问校准集或512图测试集，不得表述为泛化精度或部署完成。
