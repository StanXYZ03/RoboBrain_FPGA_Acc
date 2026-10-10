# Q8_0切入点实现路径审计

日期：2026-09-04  
状态：`Q8_0_REQUIRED / POST_MERGER_FIRST / INTERNAL_GATHER_SECOND`

## 目标模型

- 主模型：`RoboBrain2.5-4B.Q8_0.gguf`，4,693,672,576 byte。
- 视觉模型：`RoboBrain2.5-4B.mmproj-Q8_0.gguf`，453,974,336 byte。
- 最终算法精度、延迟和硬件边界必须由这两个文件验证；BF16只作结构可行性与教师参考。

## 本地执行路径

本地Q8_0运行时基于`profiling_src/llama.cpp`，当前提交`178a6c44937154dc4c4eff0d166f4a044c4fceba`。仓库已有服务器阶段计时修改，后续补丁必须保留这些改动。

Qwen3-VL视觉图位于`tools/mtmd/models/qwen3vl.cpp`：

1. Patch Embedding、位置编码后设置固定`n_pos=n_patches`；
2. 24个视觉块均按该固定长度构图；
3. DeepStack在指定层产生逐group特征；
4. 最后执行主merger，并把主视觉Embedding与DeepStack沿特征维拼接；
5. `clip_image_batch_encode()`要求图输出Token数严格等于`clip_n_output_tokens()`计算的完整数量。

因此，现有HF版`forward_shallow_pruned_vision()`不能代表Q8_0 GGUF结果，现有llama.cpp也不能通过命令行直接在视觉块1后删Token。

## 两级实现顺序

### Q8-A：Merger后压紧

先在`clip_image_encode`输出之后、送入LLM之前，按固定group索引压紧完整Q8视觉Embedding的Token列，并同步减少图像占位Token及二维/MRoPE位置。该步骤不改变Q8视觉计算图，适合先验证：

- Q8主模型与Q8 mmproj的真实输出精度；
- 主Embedding和全部DeepStack通道是否按同一Token索引压紧；
- 100%旁路是否逐字一致；
- 75%几何Oracle和L2基线的Q8任务指标。

Q8-A只能减少LLM视觉Prefill，不能减少Vision Encoder，不能作为最终性能方案。

### Q8-B：视觉块1后图内压紧

Q8-A通过后，在Qwen3-VL GGML图增加可选固定预算gather：

- 第1个视觉块后对hidden state按完整2×2 merger group收集；
- 为后续层提供同步压紧的视觉RoPE位置；
- 将后续Attention的Token长度改为实际保留数；
- DeepStack只在压紧后的序列上生成，最终merger和LLM占位数量同步；
- 100%模式保持原图路径和逐字输出。

这一阶段才同时减少后续23个Vision Block与LLM视觉Prefill，是BF16候选`vision_block_1@75%`对应的真实Q8验证。

## 进入RTL的门槛

必须全部满足：

1. Q8-A和Q8-B的100%旁路与原Q8_0输出逐条一致；
2. Q8-B在独立开发数据上通过Pointing与Grounding精度门；
3. 记录实际删除的Vision/LLM算子行数与Token数，而非只报告mask；
4. 单向驻留的PL计算与通信模型优于外置往返；
5. 冻结后建立新的未观察测试集。

在此之前不扩大BF16调参、不冻结视觉块1，也不编写Selector RTL。

