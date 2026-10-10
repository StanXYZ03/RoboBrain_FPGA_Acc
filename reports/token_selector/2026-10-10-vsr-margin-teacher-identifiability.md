# VSR Margin 教师 Prompt 可辨识性

状态：`TEACHER_HAS_MEANINGFUL_PROMPT_VARIATION`。

共 3483 条、2244 张图，其中 703 张图有多条陈述，形成 2256 个同图 Prompt 对。

同图不同 Prompt 的教师 saliency：Spearman 中位数 0.7669，Top-96 Jaccard 中位数 0.5610，归一化 RMS 差中位数 0.0852。

如果相关性和 Top-96 重叠都很高，说明当前绝对梯度教师主要是图像显著性目标；它可以继续作为稳定项，但不足以单独监督 Prompt 条件化关系选择。下一版应增加显式关系/位置对比目标，同时保持推理端不读取 label、relation、subject、object 注释。

