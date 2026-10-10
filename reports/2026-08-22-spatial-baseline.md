# Qwen / RoboBrain / Q8 spatial smoke baseline

Manifest: `datasets/dac_spatial_eval_smoke.jsonl`

> This COCO-derived set is a reproducible pipeline gate, not the final publication benchmark. Pointing is defined as object-center pointing and does not measure affordance quality.

实验锁文件：`reports/2026-08-22-spatial-baseline.lock.json`（数据、模型、输出 SHA-256，运行环境与复现命令）。

| Model | Point parse | PCK@0.1 | PCK@0.2 | Point NME / box diag | Point p50 (s) | Box parse | Mean IoU | Acc@0.5 | Box p50 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-4B BF16 | 0.9688 | 0.5000 | 0.8750 | 0.1061 | 0.2784 | 1.0000 | 0.8403 | 1.0000 | 0.4971 |
| RoboBrain2.5-4B BF16 | 1.0000 | 0.5312 | 0.7500 | 0.1318 | 0.2636 | 1.0000 | 0.8118 | 0.9062 | 0.4767 |
| RoboBrain2.5-4B Q8_0 | 1.0000 | 0.5938 | 0.8438 | 0.1009 | 2.2930 | 1.0000 | 0.8023 | 0.8438 | 33.9552 |

## Interpretation gate

- Compare Qwen BF16 with RoboBrain BF16 to isolate embodied post-training effects at 4B scale.
- Compare RoboBrain BF16 with RoboBrain Q8_0 to isolate the deployed format/runtime effect.
- Latency is reported for audit only: BF16 uses A6000/Transformers while Q8_0 uses host CPU/llama.cpp, so it is not a quantization speed comparison.
- Q8_0 Pointing follows Grounding on the same images and hits llama.cpp's vision/prefix cache; its 2.293 s median is a warm-cache value. The 33.955 s Grounding median is the cold-image CPU path.
- Do not enter token-pruning accuracy claims until all three rows are reproducible and the final held-out benchmark is frozen.

## 当前结论

- 流水线门槛通过：三种模型共 192 条预测全部完成；除 Qwen Pointing 的 1 条格式失败外，其余输出均可解析。
- RoboBrain BF16 相对 Qwen BF16：严格 PCK@0.1 提高 3.1 个百分点，但 PCK@0.2、平均 IoU 和 Acc@0.5 更低；不能宣称 RoboBrain 在该小集上全面优于底模。
- RoboBrain Q8_0 相对其 BF16：Grounding 平均 IoU 下降 0.0095，Acc@0.5 下降 6.25 个百分点；Pointing 指标未退化。该结果支持“Q8 未发生全面精度崩溃”，不证明 Q8 提升精度。
- Q8_0 GGUF 并非“所有张量都是 INT8”：语言模型含 254 个 Q8_0 与 145 个 F32 张量，mmproj 含 104 个 Q8_0 与 212 个 F32 张量；因此不能把本实验写成严格的 W8A16 对比。
- 本集只测试物体中心 Pointing 与 2D Grounding。下一阶段在做 token 剪枝/PL 加速前，仍需冻结真正的 affordance、复杂 referring expression 和轨迹任务测试集。
