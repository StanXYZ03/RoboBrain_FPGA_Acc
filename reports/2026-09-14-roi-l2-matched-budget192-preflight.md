# ROI与L2逐样本同预算对照：执行前审计

日期：2026-09-14  
状态：`LOCKED / SOFTWARE_READY / GPU_RUN_PENDING`

> 2026-09-15更新：本预检中锁定的实验已经执行完成。首次运行发现一张小图的processor最小像素边界与原生Dense不一致，原始结果保留作审计；按修正锁重新运行后，自定义Dense与原生Dense达到`384/384`逐字一致。正式结论见`reports/2026-09-15-low-budget-sparse-token-mechanism-conclusion.md`，本文件仅保留为运行前记录。

## 已完成

- 实验锁：`configs/roi_l2_matched_budget192.lock.json`。
- 数据仅使用192张已揭示开发图像、384条Grounding/Pointing任务；96张校准集和512张测试集未访问。
- 从ROI-384三次确定性原始记录读取每条任务的`final_context_visual_tokens`，作为L2逐样本目标。
- 扩展`evaluation/run_hf_shallow_block_pruning.py`，支持逐样本Token目标、目标重复一致性检查、Dense封顶和实际输出Token断言。
- 新增`evaluation/build_roi_l2_matched_report.py`，固定10,000次配对Bootstrap、种子20260914与主指标-0.02非劣门限。
- 69项`evaluation/tests`测试全部通过，锁文件可正常解析。

## 公平性边界

L2和ROI-384匹配的是最终送入LLM的视觉Token数。L2在视觉第5层之后才选择Token，因此前6个视觉块仍处理Dense网格；两者总Vision计算量不相同，报告必须同时列出最终Token比和Vision Encoder计算代理比。

ROI-384读取COCO真值框，是不可部署的表示诊断。该开发对照只能判断“保留目标局部”与“保留全局高L2 Token”的相对质量，不能证明新控制器有效，也不能替代封存测试集。

## 当前阻塞

GPU运行前只读审计显示RTX A6000空闲：显存75/46068 MiB、利用率0%、无计算进程。随后两次沙箱外正式命令均在权限审核阶段超时，未创建结果文件。这不属于模型加载、CUDA、推理代码或算法失败。

解除权限阻塞后，先运行锁定的Full与L2共768条确定性开发推理，再运行报告器；不得在查看结果后改变预算、层号、配额或非劣门限。
