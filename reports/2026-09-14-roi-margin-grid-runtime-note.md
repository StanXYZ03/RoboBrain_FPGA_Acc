# ROI边距网格运行说明

日期：2026-09-14。精度与Token汇总见`reports/2026-09-14-roi-margin-grid32-quality.md`。

- 三个ROI边距配置使用相同的32张比例分层开发图、Grounding与Pointing双任务、`do_sample=False`和三次正式重复；每个配置内部输出确定性一致。
- `margin=1.0`运行期间存在其他用户GPU进程；`margin=0.25`运行中途也观察到端到端时间由约0.45秒突增到约1.2秒，随后恢复。
- 因运行日期和共享GPU负载不同，不能把这些运行的墙钟时间用于边距配置间加速比较。统一报告已显式设置`latency_comparable=false`并隐藏延迟比。
- 结构化输出、配对PCK/IoU和processor产生的视觉Token仍可用于开发质量筛选；本说明不把这些结果提升为盲测或部署策略证据。

结论：`margin=0.25`和`margin=1.0`均未改善Pointing PCK@0.1，当前保留`margin=0.5`作为下一分辨率归因实验的固定边距。
