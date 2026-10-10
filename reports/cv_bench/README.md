# CV-Bench评测接续说明

## 全局低清与局部细化复测

2026-09-16新增`configs/cv_bench_refinement_20260916.lock.json`与`evaluation/run_cv_bench_refinement.py`。复用旧免训练文本余弦Scout、6层前缀、3×3平滑及600×600固定框，比较仅低清全图256与低清全图＋预测局部384；不使用真值框，不使用旧Grounding/Pointing专用学习式岭回归，不重训或调参。每图细化一次，并不是自适应决定是否再次观察的完整方案。

900条性能记录与完整集两档5276条均已完成：固定100题、3档各1次预热及3次交错重复；300次Dense输出与冻结基线一致。续跑命令见`CURRENT_STATUS.md`。原始JSONL追加、不覆盖。

最新结果见`2026-09-16-global-local-refinement-full2638.md`：低清全图保留93.2510%基线得分、模型端到端加速2.126倍；当前全局＋局部保留83.4431%、加速1.346倍，比低清全图更慢且得分更低。但两图方案399题解析失败、400题达到16 Token上限，因此需先做统一64 Token输出预算诊断，不能据此否定全部按需细化思想。预算诊断已锁定、派生manifest已生成，尚未执行推理。

两项任务均完成后，不加载模型生成官方得分、配对区间及性能报告：

```bash
/home/zhaoxiangyu/conda-envs/robobrain25/bin/python evaluation/build_cv_refinement_report.py --manifest datasets/cv_bench_full2638.jsonl --accuracy-jsonl evaluation/results/cv_bench_refinement_full2638_20260916.jsonl --dense-jsonl evaluation/results/cv_bench_l2_0875_full2638_20260916/shallow_r1000.jsonl --latency-jsonl evaluation/results/cv_bench_refinement_latency100_20260916.jsonl --latency-manifest datasets/cv_bench_screening100.jsonl --output-stem reports/cv_bench/2026-09-16-global-local-refinement-full2638
```

脚本要求完整ID覆盖、三次性能重复、Dense身份一致及无输出/裁剪非确定性，拒绝用不完整数据生成正式报告。模型端到端含缩放、Scout、裁剪、processor与H2D，排除JPEG和文本解码；请求端到端另含这两项。该新时延不能与之前排除processor/H2D的L2核心时延直接混成同口径加速比。

后续主平台暂定Jetson Xavier NX，尚未设备审计；本轮测量平台仍为A6000，不授权板端操作。

## 新增保留率曲线

当前最新汇总为`2026-09-16-l2-retention-quality-latency-curve.md`。同一冻结策略的75%/50%/25%三档各完成2638题；准确率全集新增7914条。固定100题、5档配置、各3次交错复测共1500条，输出与掩码在重复间完全一致。时延仅为A6000推理核心，不含图片解码、processor或H2D，非上板或完整端到端结果。

75%档得分下降0.2454个百分点，核心加速1.154倍，保留为低损失候选；50%/25%分别下降2.6198/7.7244个百分点，核心加速1.369/1.637倍，属于明确的质量—性能交换。没有新建统一精度通过门。

分档定位诊断待沙箱外GPU访问审批恢复，尚未启动。当前报告不填定位估算值；已公开96图仅用于诊断，不能作为新的盲测。续跑命令见`CURRENT_STATUS.md`。

不加载模型即可重新生成已完成的曲线：

```bash
/home/zhaoxiangyu/conda-envs/robobrain25/bin/python evaluation/build_l2_retention_curve.py --manifest datasets/cv_bench_full2638.jsonl --existing-dir evaluation/results/cv_bench_l2_0875_full2638_20260916 --sweep-dir evaluation/results/cv_bench_l2_sweep_full2638_20260916 --latency-jsonl evaluation/results/cv_bench_l2_sweep_latency100_20260916.jsonl --latency-manifest datasets/cv_bench_screening100.jsonl --output-stem reports/cv_bench/2026-09-16-l2-retention-quality-latency-curve
```

定位5档各192条完成后，在上述命令增加`--localization-manifest datasets/dac_spatial_eval_expanded.jsonl --localization-dir evaluation/results/l2_retention_sweep_localization96_20260916`即可补全定位表和配对区间。未完成时脚本会拒绝生成部分定位结果。

## 当前结果

已完成100条筛选及全部2638条原生BF16、同路径Dense、冻结L2@87.5%评测。正式结果以`2026-09-16-cv-bench-l2-0875-full2638.md`为准，100条结果只保留为筛选证据。

完整集原生记录中的`metadata.sampling`沿用了筛选运行器的措辞，实际上已按idx运行全部2638条，没有抽样。原始JSON不覆盖，勘误保存在`2026-09-16-full2638-metadata-erratum.json`，不影响数据或指标。

数据来自用户上传的官方CV-Bench文件，`datasets/CV-Bench`只读链接到旧工作区的数据目录，没有同步两个项目。完整2638条原始图片字节已提取到当前工作区`datasets/cv_bench_full2638_images`，正式推理可不依赖该链接。

策略复用`evaluation/shallow_block_pruning.py`，没有重写算法：前6个视觉块完整运行，第5层后按2×2 group的L2分数与3×3区域锚点保留87.5%，后续视觉/DeepStack/Merger/Prefill使用缩短序列，保留原位置。

## 复现或续跑

在唯一工作目录运行，先检查GPU是否空闲，不要终止其他用户进程。

```bash
source /home/chenbowu/miniconda3/bin/activate /home/zhaoxiangyu/conda-envs/robobrain25
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python evaluation/run_hf_shallow_block_pruning.py --model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B --model-label RoboBrain2.5-4B-BF16 --manifest datasets/cv_bench_full2638.jsonl --output-dir evaluation/results/cv_bench_l2_0875_full2638_20260916 --retentions 1.0 0.875 --prune-after-layer 5 --anchor-grid 3 --selector l2_anchor --device cuda:0 --dtype bf16
```

上述运行器自动跳过已有ID，不覆盖JSONL；当前两组均已完整，再执行应显示`pending=0`。

重新生成报告：

```bash
python evaluation/build_cv_bench_pruning_report.py --manifest datasets/cv_bench_full2638.jsonl --baseline-json reports/cv_bench/cv_bench_bf16_full2638.json --dense-jsonl evaluation/results/cv_bench_l2_0875_full2638_20260916/shallow_r1000.jsonl --pruned-jsonl evaluation/results/cv_bench_l2_0875_full2638_20260916/shallow_r0875.jsonl --lock-file configs/cv_bench_l2_0875_full2638.lock.json --output-stem reports/cv_bench/2026-09-16-cv-bench-l2-0875-full2638
```

## 限制与下一步

完整官方组合公式已核对本地`datasets/CV-Bench/README.md`。冻结多选解析器、16 Token上限及失败计错规则随代码公开；原生Dense和自定义Dense全部2638条逐字一致。不得把本次分数变化称为严格无损、统计非劣或提升，因为没有事前非劣界限。

推理核心时延不含H2D和预处理。已有时延是顺序运行、每题一次，不作为完整模型端到端或板端功耗结论。下一阶段应先在真实Q8_0路径实现同一layer-5 L2策略并通过100%身份门，再评测同一完整集；不能用已有几何Oracle的Q8_A/Q8_B结果代替它。其余四个官方空间基准仍待数据与官方评测器接入。
