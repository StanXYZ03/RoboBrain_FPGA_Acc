# edge576 三臂 runner 适配完成（CPU-only，未运行模型；run 模式预检/续跑/热身/模型身份已收紧）

- 状态：`EDGE576_RUNNER_ADAPTATION_COMPLETE_NO_MODEL_RUN`
- 机器结果：`reports/token_selector/2026-10-07-edge576-cvbench-three-arm-runner-adaptation.json`
- 口径：本轮只实现并验证 runner 适配；`gpu_runs=0`、`weights_loaded=false`、`model_loaded=false`、未联网/未训练/未安装依赖；**未修改任何冻结文件，未修改 `CURRENT_STATUS.md`，未创建 r3 预注册**。

## 新增文件（仅新增，不改动冻结文件）

| 文件 | SHA256 | 作用 |
| --- | --- | --- |
| `evaluation/edge576_cvbench_runner_contract.py` | `fdea7df6…90db` | 共享 edge576 运行契约：2638 输入校验、pinned `min_pixels=65536`/`max_pixels=589824` processor 包装、逐样本 `image_grid_thw` fail-closed 断言、标签隔离、resume 校验、r2 输出路径锁、依赖 SHA pin、机制身份校验，以及 fail-closed **模型身份核验**（相对路径/字节数/SHA256，含 `model.safetensors`） |
| `evaluation/run_edge576_cvbench_three_arm.py` | `b1fb2ca1…3eb85` | 三臂 runner 适配入口（`--arm`、`--validate-only`、`--dry-run`）：run 模式预检、完整续跑短路、模型身份核验、每臂一次真实多模态 warmup |
| `evaluation/tests/test_run_edge576_cvbench_three_arm.py` | `6d4f5135…0fa55` | 新增 54 项 CPU 聚焦测试（含 38 项原有 + 16 项新增） |
| `reports/token_selector/2026-10-07-edge576-cvbench-three-arm-runner-adaptation.{json,md}` | 本报告 | 中文报告 + 机器结果 JSON |

## 运行契约

- 三臂：`dense_edge576` / `corrected_l2_integer_b1_k96` / `v2_edge576_b1_k96`，输出路径严格等于 r2 config 三条路径，禁止其它目的地。
- 选择算法**零复制、零重写**：分别调用冻结 `run_hf_shallow_block_pruning.run_one`（retention 1.0、`prune_after_layer=5`、`selector=l2_anchor`）、`run_hf_hardware_regular_b1_corrected_l2.run_one`（`corrected_l2_integer`、K96）、`run_hf_conservative_l2_guarded_rerank.run_one`（现有 `conservative_l2_guarded_rerank_v2`、K96、S16）。
- 三臂共享同一 edge576 manifest、同一 prompt、BF16/SDPA、`do_sample=false`、greedy、`max_new_tokens=16`。
- processor 由 `Edge576Processor` 统一注入 `images_kwargs={min_pixels:65536, max_pixels:589824}`，并拒绝任何覆盖。
- 每样本先断言实际 `image_grid_thw` 与 manifest 一致，再写行；不一致立即 fail closed 且不写该样本。
- 逐条 append + flush 持久化；resume 按 `sample_id` 去重、跳过已完成；`parsed_answer` 允许 `null` 且仍写完整记录（解析失败计错，无 100% 解析门）。
- selector 输入经白名单 `sanitize_sample_for_selector`，`answer`/`ground_truth`/`correct`/`parsed_answer` 永不进入冻结 runner。

## run 模式新增的 fail-closed 顺序（全部在任何权重加载之前）

1. **与 `--validate-only` 完全相同的 17 项依赖 SHA256 pin 现场校验 + 机制身份校验**，任一漂移立即中止；同时执行 append-only resume 校验。
2. **完整续跑短路**：若 2638 条已完成，直接返回 `EDGE576_RUNNER_ARM_ALREADY_COMPLETE` 成功退出，不要求 `--model`、不要求 CUDA、不加载 scorer/权重/processor、不跑 warmup。
3. **`--model` 模型身份 fail-closed 核验**：读取并校验 pinned `artifacts/provenance/robobrain25-4b-model-identity.json`（自身 SHA 与结构），要求 `--model` 解析后等于 manifest `model_dir`，逐文件核对相对路径、字节数与 SHA256（含 `model.safetensors`，且必须声明为 `weight_shards`）。任一项不符立即中止，绝不加载权重。
4. CUDA 校验 → 冻结 scorer checkpoint → 本地模型/processor。
5. **每臂一次真实多模态 warmup**：同一 processor、同一机制、同一参数，仅把 `max_new_tokens` 置 1；记录被丢弃、绝不写入 JSONL；warmup 后执行 `torch.cuda.synchronize()`；warmup 摘要写入 run metadata 与最终 stats，其耗时**不混入任何样本**。
6. 之后才 append-only 写样本，并按 arm/resume 计数。

## validate-only / dry-run 结果（不加载权重）

三臂 `--validate-only` 均为 `EDGE576_RUNNER_VALIDATE_ONLY_OK`：2638 条唯一输入、全部 `G<=576`、参数锁一致、输出路径等于 r2 pin、机制身份 `is_new_mechanism=false`、17/17 依赖 SHA pin 现场复核通过、`pending=2638`、`weights_loaded=false`、`gpu_runs=0`。

`dense_edge576 --dry-run` 返回 `EDGE576_RUNNER_DRY_RUN_OK` 并给出启动命令。

## 主审校 processor wrapper 证据（真实本地 processor，CPU-only）

主审校已用真实本地 `RoboBrain2.5-4B` processor（`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、仅 CPU、不加载权重）在 **G=64/240/576** 三个样本上验证 `Edge576Processor` 注入的 `min_pixels=65536`/`max_pixels=589824` 与冻结 `processor_inputs` 组合后，重算 `image_grid_thw` 与 manifest 完全一致；本报告复核并落盘该证据：

| sample_id | G(manifest) | manifest grid | wrapper grid | 匹配 | 与显式 bounds 裸 processor 一致 |
| --- | --- | --- | --- | --- | --- |
| `0` | 64 | `[1,16,16]` | `[1,16,16]` | ✅ | ✅ |
| `646` | 240 | `[1,24,40]` | `[1,24,40]` | ✅ | ✅ |
| `1838` | 576 | `[1,36,64]` | `[1,36,64]` | ✅ | ✅ |

- 三样本 `wrapper_matches_manifest=true`、`all_wrapper_match=true`；`gpu_runs=0`、`weights_loaded=false`。
- 逐样本 `image_sha256` 已随证据记录，便于复核：`416ad33e…a922` / `fd991bcc…258d` / `5f16e394…d37`。

## 验证

- `py_compile`：3 个新增文件通过。
- 新增聚焦测试：**54 OK**（原有 38 项全部保留通过；新增 16 项覆盖：run 预检顺序、完整续跑不加载、模型身份失败先于 scorer/权重中止、warmup 单次 `max_new_tokens=1` + 单次 `torch.cuda.synchronize` + 不写文件 + 不改原样本、模型身份正例/缺文件/大小错/hash 错/未声明权重分片/model_dir 不符/manifest pin 不符/路径逃逸）。
- 既有 edge576 聚焦测试回归：**52 OK**（builder 17 + auditor 18 + r2 prereg 17）。合计 **106 OK**。
- r2 依赖 pin 现场复核：**17/17 匹配**（本轮未改任何被 pin 文件）。

## 三臂实际启动命令（宿主执行，尚未运行）

```
python evaluation/run_edge576_cvbench_three_arm.py --arm dense_edge576 --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --model-label RoboBrain2.5-4B-BF16 --manifest datasets/2026-10-07-cv-bench-edge576-full2638.jsonl --output artifacts/token_selector/2026-10-07-edge576-cvbench/dense_edge576_full2638.jsonl
python evaluation/run_edge576_cvbench_three_arm.py --arm corrected_l2_integer_b1_k96 --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --model-label RoboBrain2.5-4B-BF16 --manifest datasets/2026-10-07-cv-bench-edge576-full2638.jsonl --output artifacts/token_selector/2026-10-07-edge576-cvbench/corrected_l2_integer_k96_full2638.jsonl
python evaluation/run_edge576_cvbench_three_arm.py --arm v2_edge576_b1_k96 --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --model-label RoboBrain2.5-4B-BF16 --manifest datasets/2026-10-07-cv-bench-edge576-full2638.jsonl --output artifacts/token_selector/2026-10-07-edge576-cvbench/v2_edge576_k96_full2638.jsonl
```

## 风险

- 三臂本轮均未在 GPU 上实际运行，runner 的 GPU 路径未与真实模型联调。
- 模型身份核验会在加载前流式读取并 SHA256 整个 **9.65 GB `model.safetensors`**，每次正式 run 增加一次性的 CPU/IO 开销（这是 fail-closed 身份核验的要求，未做缓存或跳过）。
- 真实多模态 warmup（`max_new_tokens=1` + `torch.cuda.synchronize`）已接线并以 fake 单测覆盖，但本轮**未在 GPU 上真实执行**。
- 断点续跑为 fail closed：混臂、混 manifest、混像素预算或损坏记录一律中止，需人工清理后才能继续。
- `parsed_answer=null` 计错但保留完整记录；评分必须以 `cv_bench_metric_v2` 的 `parsed_answer` 字段为准。
- edge576 full2638 仍是发展集，通过门也不构成发表或部署主张。
