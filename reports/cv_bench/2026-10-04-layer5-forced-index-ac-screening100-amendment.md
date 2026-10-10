# Layer-5 forced-index A/C 预注册修正记录

日期：2026-10-04  
类型：`NON_ANALYTIC_METADATA_CONTRACT_FIX`

## 发生了什么

两条 GPU arm 已按预注册命令完成后，首次运行汇总器时触发 fail-closed：runner 将命令行选项 `--dtype bf16` 原样记录为 `run.dtype="bf16"`，初版汇总器却要求语义名称 `"bfloat16"`。因此没有生成任何正式 JSON/CSV/MD 报告。

## 修正内容

- 将汇总器和预注册元数据合同统一为 runner 的实际记录值 `bf16`；runner 仍将它映射为 `torch.bfloat16`。
- 更新汇总器测试中的合成记录。
- 未修改或重跑 control/forced 原始 JSONL。
- 未修改模型、视觉索引、输出文本、官方评分函数、bootstrap seed/repeats/分层方式或判定阈值。

## 审计哈希

- control 原始 JSONL：`838cbb847c957412b73010b4c54999d1cad486a83c6e54592a1c42d35420d989`
- forced 原始 JSONL：`f78530ee2e781774dae66feb9e9287e678f1beca378c55ba186f7e6470b8b083`
- 修正后 summarizer：`6356d3ac9571ded14f717a562a946af3f6aed2a437286bed7d2bb441b6528087`
- 修正后 preregistration：`ef1ffeae6c2805b4ec1b3ff3f3551169b17c9ccf1b125ac956fb7f5603ce4457`

该修正只解决字符串契约不一致，不接触实验结果，属于预先声明判据下的非分析性实现修正。
