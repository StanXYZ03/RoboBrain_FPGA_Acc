# PL驻留FFN活跃区间规划器

结论：`LIVE_RANGE_PLANNER_PASS / 64M_NO_OVERLAP / SUBGRAPH_CHUNK_DIFFERS_FROM_GEMM_CHUNK`。

| 子图 | total M | 单GEMM旧chunk | FFN安全chunk | chunk数 | arena峰值 | 64 MiB利用率 |
|---|---:|---:|---:|---:|---:|---:|
| LLM SwiGLU | 1080 | 976 | 432 | 3 | 65,678,848 B | 97.87% |
| Vision GELU | 4320 | 3584 | 2896 | 2 | 66,917,376 B | 99.71% |

规划器按半开生命周期`[birth, death)`分配64-byte对齐物理区间，允许生命周期不重叠的buffer复用
同一offset。属性测试覆盖两类子图、小M、边界M和最大chunk+16拒绝，所有同时存活buffer物理
区间零重叠。

LLM中`input_q8 → ffn_q8 → down_f32`复用offset `30,883,840`；Vision中
`input_q8/fc1_q8/fc2_f32`复用offset `16,318,464`。这表明live-range复用有效，但完整FFN的
最大chunk仍显著小于单GEMM容量规划值。

边界：当前只是确定性内存/生命周期规划，不包含向量核RTL、DMA并发或上板性能。下一门是两类
FFN的C golden与向量算子实现，不能把规划结果当成加速比。

证据：

- `fpga_accel/pl_resident_ffn/live_range_planner.py`
- `fpga_accel/pl_resident_ffn/test_live_range_planner.py`
- `fpga_accel/pl_resident_ffn/artifacts/20260830/llm_swiglu_m1080_plan.json`
- `fpga_accel/pl_resident_ffn/artifacts/20260830/vision_gelu_m4320_plan.json`
