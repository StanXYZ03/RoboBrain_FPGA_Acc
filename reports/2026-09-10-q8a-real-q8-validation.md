# RoboBrain2.5-4B Q8-A 真实 Q8_0 路径验证

## 结论

Q8-A 的软件实现门通过，`post-merger@75%` 的开发集精度门未通过。

- 真实模型：`RoboBrain2.5-4B.Q8_0.gguf` 与 `RoboBrain2.5-4B.mmproj-Q8_0.gguf`。
- 100% identity：原路径与身份旁路均退出成功，20 个生成 Token ID 逐项相同，输出均为`[470, 16, 918, 427]`。
- 75% compact：8/8 开发样本成功，视觉 Token 从176压到132，所有输出均可解析，未出现 DeepStack 维度错误或 MRoPE 越界。
- 质量门：Grounding Mean IoU `0.7701 -> 0.7799`，Acc@0.5 `1.0 -> 1.0`；Pointing PCK@0.1 `0.75 -> 0.50`，PCK@0.2 `1.0 -> 1.0`。PCK@0.1下降0.25，超过预注册最大下降0.125，因此判定失败。

本结果证明 Q8-A 能在真实 Q8_0 执行路径完成 Merger 后 Token 列压紧，并保持主投影与 DeepStack 特征行同步；它不证明75%配置满足最终精度要求。

## 修复与验证

首次真实 identity 测试返回`rc=2`。根因是Qwen3-VL即使处理单张静态图，也以两个预处理条目配合`temporal_merge=2`表示；旧检查错误地只允许单条目、`temporal_merge=1`。

修复后仅允许“最终输出仍为一个二维 MRoPE 网格”的静态图压紧；多时间片视频仍拒绝，避免在没有时间位置映射验证时扩大支持范围。C API 测试夹具同步改为Qwen3-VL的双条目静态图表示，测试通过。

## 证据

- 身份门摘要：`evaluation/results/q8a_identity_20260910T143750/summary.json`，SHA256 `1b82946ea18c78c38476ed42df25a48961132d357c61eb511ade9cb06246dcb8`
- 8样本原始记录：`evaluation/results/q8a_oracle75_20260910T144447/raw_records.jsonl`，SHA256 `9559e4aca45a3a052576b5372dddadb5d5ed24b387128ea3af97612c707da195`
- 8样本摘要：`evaluation/results/q8a_oracle75_20260910T144447/summary.json`，SHA256 `c5f035b36bbeec9cf530a26862d1730ca01f6a4d95e662cc1d2671fb98aae0bb`
- Q8-A实现源文件：`profiling_src/llama.cpp/tools/mtmd/mtmd.cpp`，SHA256 `b4a8b7dfd87f7933a4846a16dc3bf4150f401ebbf00f307a10fe07c9e86fa1dc`
- CLI可执行文件：SHA256 `85a442ce1916e4ae48826031b52f576033eb45cfba4cc7e9b563707b7f46be5c`

## 限制

几何 Oracle 掩码读取真实框，只用于开发诊断，不能部署，也不是全局精度上界。每任务仅4条，不能形成论文级精度结论。记录的进程时间包含模型加载，不是稳态延迟，不能用于宣称加速。Q8-A只减少LLM视觉Prefill，未减少24层视觉编码计算。

## 决策

状态：`Q8_A_IDENTITY_AND_MECHANICS_PASS / POST_MERGER_75_QUALITY_FAIL / Q8_B_IMPLEMENTATION_NEXT`。

Q8-A已经满足Q8-B的基础设施前置门。下一步在Qwen3-VL GGML视觉图第1块后按完整2x2 Merger group执行gather，使后续23个视觉块和LLM视觉Prefill同时真实减算；Q8-B仍须独立通过100% identity和75%质量门。
