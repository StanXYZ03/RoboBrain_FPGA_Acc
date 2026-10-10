# RoboBrain2.5-4B Q8-B 真实 Q8_0 路径验证

## 结论

Q8-B 的软件机制与100%身份门通过，但75%和87.5%保留率均未通过预注册质量门，因此当前不得冻结Selector描述符、切入边界或进入Selector/Merger/Compactor RTL。

- 真实模型：`RoboBrain2.5-4B.Q8_0.gguf`与`RoboBrain2.5-4B.mmproj-Q8_0.gguf`。
- 100% identity：stock与Q8-B身份旁路均退出成功，20个生成Token ID逐项相同，输出均为`[470, 16, 918, 427]`。
- 87.5%正式复核：8/8开发样本退出成功且结构化输出解析率100%；视觉Merger group由176压到154，对应后续视觉块的patch行数由704降到616。
- 质量门：Grounding Mean IoU `0.7701 -> 0.7993`，Acc@0.5保持`1.0`；Pointing PCK@0.1保持`0.75`，PCK@0.2由`1.0`降到`0.75`。PCK@0.2下降0.25，超过预注册最大下降0.125，判定失败。

Q8-B在Qwen3-VL第一个完整视觉块后按2x2 Merger group选择原始patch，使后续23个视觉块、DeepStack、Merger及LLM视觉Prefill处理更短序列。该结果证明真实Q8_0图可以执行这一缩短路径，不证明存在端到端加速，也不证明87.5%配置满足任务精度。

## 实现与机制验证

- 新增独立Q8-B early-keep API，与Q8-A Merger后压紧互斥，非法组合会被拒绝。
- 每个保留group展开为四个原始patch索引；视觉块1输出后执行行gather，后续视觉块使用压紧后的MRoPE位置。
- DeepStack及Merger沿用同一组保留位置，最终输出Token数与LLM占位Token保持一致。
- C API覆盖稀疏索引、冲突拒绝和100%身份路径；最终干净静态库无重复archive member。

75%初步测试同样为8条开发样本，视觉Token为`176 -> 132`。Grounding Mean IoU `0.7701 -> 0.7997`，Pointing PCK@0.1保持`0.75`，PCK@0.2由`1.0`降到`0.75`，因此同样失败。该轮使用清理重复archive member之前的中间构建，只作为方向一致的附加证据；正式结论引用下述最终干净二进制87.5%复核。

## 正式证据

- 最终CLI可执行文件SHA256：`e0e8b34088e96ed3b632519d2c9d5d0ad9ca070183f1448f3a24405aa6531ec4`
- 最终100%身份门：`evaluation/results/q8b_identity_20260910T152554/summary.json`，SHA256 `1447198d1e697a579a032dad7a306f518991d0ae5325c12831ee6eaa85f32689`
- 87.5%掩码：`evaluation/results/q8_geometric_oracle_masks_r0875_8sample.jsonl`，SHA256 `4a7d362031b4b022222fea4fcba3afbc9a3fc3df461848ecc2a46510f504a3a0`
- 87.5%原始记录：`evaluation/results/q8b_oracle0875_20260910T160715/raw_records.jsonl`，SHA256 `e6656c2d49787611b0306df37f0bcc1a63f4186a922a71bae33c1ef20166c6ab`
- 87.5%摘要：`evaluation/results/q8b_oracle0875_20260910T160715/summary.json`，SHA256 `17b554ad2505c76655bc888ff906ec9f7342c2e7a700f2af3d663a69d2372d5c`
- 元数据勘误：`evaluation/results/q8b_oracle0875_20260910T160715/metadata_erratum.json`。原始记录未覆盖，指标不受影响。

## 证据边界

几何Oracle掩码读取真实框，只能用于开发诊断，不能部署。每任务仅4条，离论文级统计充分性很远。PCK是离散小样本指标，一条样本即可造成0.25变化；这不允许忽略门控失败，也不足以宣称Q8-B路线普遍无效。

运行记录中的进程墙钟时间包含模型加载，不是稳态延迟。当前没有逐算子Q8-B计时、GPU结果或ZCU102板测，因此不得将`704 -> 616`的序列缩短直接表述成实测加速比。本阶段未访问、复位或配置板卡，也未写入SD/QSPI。

87.5%正式原始记录中的`development_oracle_uses_ground_truth`因旧的模式名判断被误写为`false`；`summary.json`中的`oracle_uses_ground_truth_bbox=true`正确。勘误文件绑定原始记录哈希，保留原始数据不变。

## 决策与下一步

状态：`Q8_B_IDENTITY_AND_MECHANICS_PASS / Q8_B_75_AND_0875_QUALITY_FAIL / SELECTOR_RTL_HOLD`。

两级真实Q8门均已完成：Q8-A@75%的Pointing PCK@0.1失败；Q8-B@75%与@87.5%的Pointing PCK@0.2失败。由于当前Oracle上调保留率仍未通过，不再在这8条已观察样本上继续调参，也不冻结硬件Selector。

在AI核心平台尚未确定时，下一项不会白做的工作是建立平台无关的符号收益模型：输入视觉切入点实测时间、Token数、张量字节数、链路带宽和驻留策略，输出单向驻留与外置往返的净延迟/能量交叉点。Q8-B代码保留为真实减算实验路径；新选择算法必须使用新的开发划分，并在冻结后建立未观察测试集。
