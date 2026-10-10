# 任务感知矩形ROI提议器结论

日期：2026-09-15  
状态：`LOCALIZATION_GATE_FAIL / NO_VLM_GENERATION / CURRENT_LOWRES_SCOUT_BRANCH_STOP`

## 实验边界

实验严格使用192张已揭示开发图像、384条Grounding/Pointing任务，按COCO图像ID进行5折分组交叉验证。同一图像的两类任务不会跨训练折与验证折。96张校准图和512张测试图均未访问。

Scout输入为长边256的完整图像，调用冻结RoboBrain视觉编码器前6层，在第一个DeepStack投影处形成任务文本与视觉网格的余弦相关性。每条Scout只执行视觉前缀，不调用4B语言模型生成。比较两个在运行前锁定的候选：文本余弦峰值加固定600×600框，以及共享岭回归热力图头加尺度回归头。

## 结果

| 方法 | 任务 | 目标中心召回 | 目标框覆盖 | ROI面积比 | ROI Token比 | Scout+ROI Vision MAC比 |
|---|---|---:|---:|---:|---:|---:|
| 文本余弦固定框 | Grounding | 0.5104 | 0.4332 | 0.3600 | 0.4113 | 0.4165 |
| 文本余弦固定框 | Pointing | 0.4740 | 0.3846 | 0.3600 | 0.4113 | 0.4165 |
| 岭回归热力图+尺度 | Grounding | 0.8490 | 0.7345 | 0.5595 | 0.4276 | 0.4335 |
| 岭回归热力图+尺度 | Pointing | 0.8281 | 0.7290 | 0.5553 | 0.4237 | 0.4298 |

预设门槛为每个任务同时达到中心召回不低于0.95、目标覆盖不低于0.85、ROI Token比不高于0.50、累计Vision MAC比不高于0.60。两个候选均只通过成本门，未通过定位门。因此不运行预测ROI的完整BF16 VLM生成。

Scout视觉前6层的A6000中位计时约为Grounding 4.171 ms、Pointing 4.160 ms。该时间不含JPEG解码、processor、ROI读取与重编码、LLM Prefill和Decode，也不是ZCU102或未来主算力平台的时间。

## 事后归因

以下只用于解释失败，不用于重新选参：

- 学习候选对large目标的中心召回为0.9722，但small目标仅0.6912。
- edge目标中心召回为0.8072，interior目标为0.9615；主要困难集中在小目标和边缘目标。
- 学习候选平均ROI面积已经达到约56%，仍有15%至17%的目标中心遗漏，说明问题不是框过小或计算预算过紧，而是低清浅层特征未可靠定位目标。
- 同图两类任务的学习候选框中心平均距离为22.27（0至1000坐标），明显小于余弦基线的241.77，说明监督头抑制了提示词格式扰动，但仍未达到部署所需召回。

## 决策

1. 按锁定规则停止当前`256px + 6层浅视觉前缀 + 轻量热力图/尺度头`分支，不在这192张图上调整深度、框大小、投影维数、正则化或阈值。
2. 不访问校准集和测试集，不把训练折拟合或真值框结果包装成可部署精度。
3. 不启动ROI控制器或Selector RTL；现有矩形ROI结果继续作为表示上界，而非可部署算法。
4. Token压缩研究暂时保留两项可靠基线：L2@87.5%轻度裁剪，以及真值矩形ROI表示上界。约34%的中层离散裁剪和当前低清Scout均停止。
5. 在主AI算力平台确定前，后续工作回到平台无关的模型压缩与阶段接口：量化敏感性、权重/KV驻留条件、Decode带宽下界和端到端预算。任何新的ROI路线都必须引入实质不同的信息源，例如经过定位监督训练的视觉骨干或外部检测器，并建立新开发划分，不能继续在当前开发集试参数。

## 证据

- `configs/task_conditioned_rectangular_roi_cv.lock.json`
- `evaluation/extract_task_conditioned_roi_scout.py`
- `evaluation/evaluate_task_conditioned_roi_proposals.py`
- `evaluation/results/task_conditioned_roi_scout256_prefix6_dev384_20260915.jsonl`
- `evaluation/results/task_conditioned_rectangular_roi_cv_predictions_20260915.jsonl`
- `reports/2026-09-15-task-conditioned-rectangular-roi-cv192.{json,csv,md}`
