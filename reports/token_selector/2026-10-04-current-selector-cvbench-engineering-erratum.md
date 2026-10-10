# Current Selector CV-Bench 工程修复附录

## 修复范围

本附录不修改 2026-10-04 预注册中的模型、数据集、部署 checkpoint、Selector 数学、K=48/64/96、解码方式、正式指标、完整样本数或统计方法。仅修复正式运行前 Smoke 发现的两个工程问题：

1. 当 `G <= K` 时，Selector 按合同执行 identity bypass，不产生 INT32 scorer 中间界限。旧门禁错误地要求旁路样本也具有四项界限，造成合法样本被误报失败。修复后，旁路样本跳过该项要求；真正执行压缩的样本仍必须完整记录四项界限。
2. Shell 驱动把带参数的 Python 命令保存为单个字符串，导致系统将整串内容误当作可执行文件。修复为 Bash 参数数组，不改变传入 runner/summarizer 的任何参数。

## 代码指纹

| 文件 | 预注册时 SHA256 | 修复后 SHA256 |
|---|---|---|
| `evaluation/run_hf_current_selector_cvbench.py` | `5870e8477dcbccd52c572e8c5c03d6991e426e97522286709cdeea01a5f2bfc2` | `0098faf3af47b648d54489b9a08e92c47859ee2b3735292b3f2f291324f01d2d` |
| `evaluation/tests/test_current_selector_cvbench.py` | `8d875117f0d7829cc54dc078cdf614ac9e16c8f833b8fa58fe1ec2369d1d9939` | `6c403a8c85de14b2065dadced3e2e62a11528496c88ac934d0b1acdb01549439` |
| `scripts/run_current_selector_cvbench.sh` | `da1f8523dbbf4af118664b623ddb95297e107190c2e0ad7f54dabfd5c86c386b` | `3eefbb08542cf2263f4efe3d9c3485b796223d1c3a382c104d3eea2150bcd411` |

## 验证

- 定向测试：19 项通过。
- 项目全量 CPU 测试：734 项通过，0 失败，耗时 45.251 s。
- 全量日志：`reports/token_selector/logs/2026-10-04-current-selector-cvbench/full-tests-after-smoke-fix-host.log`
- 日志 SHA256：`ff6866c9816c9bf0735f412ba75eb29858f29ec3fd770dd2e4696b3578592973`

旧 Smoke 文件必须留档，不能在原文件上续跑；修复后门禁从空的新文件重新运行。正式 2638 样本评测只有在修复后的 identity 与 Smoke 门禁通过后才允许启动。
