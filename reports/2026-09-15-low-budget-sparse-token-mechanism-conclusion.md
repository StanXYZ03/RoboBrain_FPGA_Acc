# 低预算稀疏Token机制结论

日期：2026-09-15  
状态：`MATCHED_BUDGET_COMPLETE / LOW_BUDGET_POST_LAYER5_SPARSE_TOKEN_STOP / RECTANGULAR_ROI_CONTROLLER_NEXT`

## 受控对照

实验使用192张已揭示开发图像，每图各一条Grounding和Pointing，共384条任务。校准96图和测试512图均未访问。所有候选逐样本匹配ROI-384最终视觉Token；两条ROI目标超过原图Dense Token时，稀疏候选按预注册规则回退为Dense。

原生Dense与修正后的自定义Dense输出`384/384`逐字一致，排除了自定义greedy和processor网格差异。正式原始结果均保留，首次processor默认值不一致的运行也未覆盖。

| 方法 | 最终Token比 | Vision计算代理比 | Grounding IoU | Pointing PCK@0.1 |
|---|---:|---:|---:|---:|
| Dense | 1.000 | 1.000 | 0.8103 | 0.6458 |
| 真值矩形ROI-384 | 0.342 | 0.309 | 0.7890 | 0.5990 |
| layer-5 L2稀疏Token | 0.342 | 0.482 | 0.5324 | 0.3438 |
| layer-5几何Oracle稀疏Token | 0.342 | 0.482 | 0.5710 | 0.2708 |

L2相对Dense的主指标差为Grounding `-0.2780 [-0.3268,-0.2291]`、Pointing `-0.3021 [-0.3750,-0.2292]`。几何Oracle相对ROI-384仍下降Grounding `-0.2179 [-0.2524,-0.1841]`、Pointing `-0.3281 [-0.4115,-0.2448]`。括号为10,000次逐图配对Bootstrap的95%区间。

## 失效归因

L2仅保留目标中心网格`150/384`，目标相交网格总体保留率为36.7%，说明非任务感知排序确实经常漏掉目标。但这不是全部原因：

- 几何Oracle保留目标中心`372/384`，完整保留全部目标相交网格`270/384`，目标相交网格总体保留率75.9%。
- 几何Oracle只使Grounding相对L2提高0.0387，95%区间跨0；Pointing反而下降0.0729，区间同样跨0。
- 同样的最终Token预算下，矩形ROI远好于两种离散Token选择，而且ROI从输入开始缩小网格，Vision计算代理仅0.309；layer-5选择器因前6层仍处理Dense网格，计算代理为0.482。

因此，当前RoboBrain原模型在约34%预算下依赖连续矩形视觉序列、局部放大重采样或其共同形成的输入分布。只在原图网格中保留框内离散Token，不能复现矩形ROI效果。该分析是机制关联证据，不证明单一因果。

## 决策

1. 停止未微调RoboBrain上的`layer-5 + 约34%离散Token`路线，不进入Selector RTL。
2. 保留L2@87.5%作为轻度裁剪安全基线；本结果不与其矛盾，因为裁剪强度和实验目的不同。
3. 保留矩形ROI作为下一表示路径，但其当前使用真值框且ROI-384尚未通过两任务非劣门，不能称为可部署方案。
4. 下一算法候选改为低分辨率任务感知矩形提议器：从全局低清视觉前缀和文本相关性预测一个连续ROI及尺度，不用4B模型生成长文本作Scout。
5. 先在192图开发集做按图像分组交叉验证，评价框中心召回、目标覆盖、ROI面积和累计计算；定位门通过后才运行out-of-fold预测ROI的真实VLM输出。校准集用于冻结阈值，测试集只在最终方案冻结后访问一次。

硬件含义：后续优先定义矩形ROI描述符、DDR突发读取、Resize/归一化和批量Patch准备；离散Token selector/compactor保持暂停。是否下沉到ZCU102仍取决于选定主算力平台后的链路与端到端收益门。

## 证据

- `configs/roi_l2_matched_budget192.lock.json`
- `configs/roi_l2_matched_budget192_nativegrid_correction.lock.json`
- `configs/geometric_l2_roi384_matched_budget192.lock.json`
- `reports/2026-09-15-roi-l2-geometric-matched-budget192.{json,md}`
- `reports/2026-09-15-roi-l2-matched-failure-attribution192.{json,csv,md}`
- `evaluation/results/roi_l2_matched_budget192_nativegrid/`
- `evaluation/results/geometric_matched_roi384_nativegrid/`
