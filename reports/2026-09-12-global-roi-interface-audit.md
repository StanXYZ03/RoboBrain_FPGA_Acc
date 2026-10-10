# 全局低清图加高分辨率ROI接口审计

日期：2026-09-12。状态：`HF_MULTI_IMAGE_FORWARD_PASS / Q8_MULTI_IMAGE_FORWARD_PASS / ACCURACY_UNVERIFIED`。

## 结论

- HF处理器对同一请求中的两张图生成两个独立`image_grid_thw`；4条预检中，合并后视觉Token总数与文本中的图像占位符数逐条一致。
- RoboBrain BF16在A6000上完成一次双图预热和24次双图正式生成，说明当前Transformers实现的MRoPE、DeepStack和语言Prefill可接受双图输入。
- 当前`llama-mtmd-cli --image`在帮助中明确支持逗号分隔的多个文件；源码按媒体数量插入标记并逐个`load_media()`。
- 真实Q8_0主模型与Q8_0 mmproj完成一条双图CPU推理，退出码0，日志中有两次独立`encoding mtmd batch`，输出`[100, 780, 260, 996]`。

上述证据只证明接口和执行路径成立。Q8预检为缩短运行时间使用`--image-min-tokens 1`，运行时明确警告Grounding建议至少1024图像Token，因此该输出不得用于质量判断。进程总时间378.737秒包含模型加载和CPU生成，也不是稳态延迟。

## 坐标契约

第二张图为从第一张原始场景裁剪得到的ROI。模型输出相对ROI的0–1000坐标；主机按实际整数像素裁剪边界换算得到的`effective_crop_box_1000`映射回原图。该规则避免理想浮点框与实际像素取整产生的系统误差。

## 旧实现差异

`run_hf_scout_refine.py`先读取低分辨率模型输出，再执行一次只含裁剪图的完整生成；总成本是两次生成之和，且第二轮缺少显式全局图。新参考路径在最终一次请求中同时输入低清全局图和ROI。真值ROI版本仍只是受控诊断；部署版本必须加入Scout/策略全部成本。

## 证据文件

- CPU处理器逐样本记录：`reports/2026-09-12-global-roi-processor-preflight.json`
- BF16 96次原始记录：`evaluation/results/global_roi_bf16_pilot_20260912.jsonl`
- Q8原始日志与摘要：`evaluation/results/q8_global_roi_preflight_20260912/`
- 预算字段规范：`artifacts/visual_refinement_budget_schema.json`

## 尚未通过的门

1. 尚未在真实Q8路径按可接受Token预算比较Dense、低清、ROI-only和全局加ROI。
2. 尚未证明多图表示在空间任务上优于同预算低分辨率表示。
3. 尚未实现跨轮KV缓存复用；当前HF参考路径是单次双图完整Prefill。
4. 尚未测量未来异构链路和ROI预处理的能耗与端到端收益。

