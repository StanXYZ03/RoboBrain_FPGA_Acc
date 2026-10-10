# SpatialEval VQA test4635 Schema / 官方评分语义审计（CPU-only）

- 状态：`SCHEMA_AUDIT_PASS`
- 机器结果：`reports/token_selector/2026-10-09-spatialeval-vqa-schema-audit.json`
- 口径：CPU-only、无网络、无 GPU、无模型加载、无推理、无训练；只读固定输入；只新建本报告两个文件，未修改任何其它文件（含 `CURRENT_STATUS.md`、configs、evaluation 代码）。只输出聚合统计，未逐条打印 prompt/答案。

## 唯一结论

**`SCHEMA_AUDIT_PASS`**：固定 SpatialEval VQA test4635 manifest 与官方 `d82ba382` evaluator/inference 语义已现场复核——四个输入 SHA256 全部匹配、4635 个唯一 ID、源 parquet 2318+2317、四 task×question_id 结构与 oracle 三字段均已确认，官方评分语义已读清，可作为**未用于 Selector 调参的外部确认集**冻结。但官方 `evaluation.py` 只定义逐 task `Acc = correct/line_count`（无总分公式）、用 `str(oracle_answer).lower()` 的**子串包含**而非相等比较、只评估 `m-*_w_reason.jsonl`，且 **spatialreal 无 handler**（会复用上一条 `model_answer`）。因此冻结前必须显式固定 task 子集、总分口径与解析器端口。现有 `evaluation/spatialeval_metric.py` 是草案 exact-match，**绝不能称官方复现**。

## 1. 固定输入 SHA256 复核

| 角色 | 路径 | 期望 SHA256 | 实测 | 结果 |
| --- | --- | --- | --- | --- |
| manifest | `datasets/external/2026-10-09-spatialeval-vqa-test4635.reference.jsonl` | `03649b9c…f4050` | `03649b9c…f4050` | 匹配 |
| summary | `datasets/external/2026-10-09-spatialeval-vqa-test4635.reference.summary.json` | — | — | 交叉核对 |
| evaluator | `artifacts/provenance/spatialeval-official-d82ba382/evaluation.py` | `88d2a1b3…cd952` | `88d2a1b3…cd952` | 匹配 |
| inference | `artifacts/provenance/spatialeval-official-d82ba382/inference_vlm.py` | `0a69b4cd…b39337` | `0a69b4cd…b39337` | 匹配 |
| configs | `artifacts/provenance/spatialeval-official-d82ba382/inference_configs.py` | `79d9c59a…e44972` | `79d9c59a…e44972` | 匹配 |

官方 commit：`d82ba382805265f205693cb83a30c4866ea6c35b`。summary 声明 `row_count=4635`、`unique_id_count=4635`、`split=test`、`MANIFEST_BUILT_OFFLINE`。

## 2. manifest 完整性

- 行数 `4635`；唯一 `sample_id` `4635`（无重复）；唯一 `record_sha256` `4635`；唯一 `text` `3325`；唯一 `image_sha256` `1594`。
- 源 parquet 拆分：`test-00000-of-00002.parquet` `2318` 行（SHA256 `624aad41…b897`）+ `test-00001-of-00002.parquet` `2317` 行（SHA256 `b9301e70…dadd`）= `4635`，与 summary 一致；`(parquet,row_index)` 全唯一。
- `record_sha256` 方案 = `sha256(json.dumps(去 record_sha256, sort_keys=True, ensure_ascii=False, separators=(',',':')))`，`4635/4635` 全部重算匹配。
- 冻结指纹：`sha256(sorted(sample_id) 换行拼接)` = `4ea00a88…a26f`；`sha256(按文件序拼接 record_sha256)` = `68ab8b6e…82d8`。
- 字段 schema（每行 13 字段，均 4635 非空键）：`sample_id, text, oracle_answer, oracle_option, oracle_full_answer, width, height, image_sha256, image_storage, source_parquet_path, source_parquet_row_index, source_parquet_sha256, record_sha256`。

## 3. ID 结构与四 task / question_id 计数

- ID 结构：`<task>.vqa.<int>[|sa_<int>].<question_id>`，全部 4 段；第 2 段全为 `vqa`；索引段 `<int>` 4500、`sa_<int>` 135。
- 官方抽取：`task = id.split('.')[0]`；`question_id = int(id.split('.')[-1])`。

| task | question_id 0 | 1 | 2 | 合计 |
| --- | --- | --- | --- | --- |
| spatialmap | 500 | 500 | 500 | 1500 |
| mazenav | 500 | 500 | 500 | 1500 |
| spatialgrid | 500 | 500 | 500 | 1500 |
| spatialreal | 135 | 0 | 0 | 135 |
| **合计** | **1635** | **1500** | **1500** | **4635** |

- `spatialmap`：qid0 方向选择题；qid1 实体/人物位置；qid2 计数。
- `mazenav`：qid0 右转数；qid1 总转向数；qid2 二值 Yes/No（option A=Yes 333、B=No 167，且仅 1 条选项字母行）。
- `spatialgrid`：qid0 方块计数；qid1/qid2 动物识别（5 类动物）。
- `spatialreal`：qid0 真实图像空间 QA；**官方 `evaluation.py` 无该 handler**。

## 4. oracle 三字段审计

| 字段 | 类型 | 空值 | 关键规范/格式 | 取值规模 |
| --- | --- | --- | --- | --- |
| `oracle_answer` | 100% `str` | 0 null / 0 空串 | strip 干净率 1.0；全小写率 0.6542；仅 3 条以 `.` 结尾 | 308 distinct；integer_string 2015 + free_text 2620 |
| `oracle_option` | 100% `str` | 0 | 100% 匹配 `^[A-D]$` | 仅 A/B/C/D |
| `oracle_full_answer` | 100% `str` | 0 | 100% 匹配 `^[A-D]\. .+$`；100% 恒等于 `<option>. <answer>` | 随 answer |

- `oracle_answer` 分类（task）：spatialmap int 500/text 1000；mazenav int 1000/text 500；spatialgrid int 500/text 1000；spatialreal int 15/text 120。
- 关系：`oracle_full_answer = oracle_option + ". " + oracle_answer` 精确成立 **4635/4635**。
- 但 `oracle_full_answer` 作为 prompt 中**精确整行**只有 3466 条；其余 **1169** 条只是该选项行的前缀（真实选项文本把答案作为子串包含）。
- 4 选项样本 4135 条中，选项文本与 `oracle_answer` 精确相等仅 3133 条（0.7577），另 1002 条为“包含答案”。官方评分不使用 `oracle_option`/`oracle_full_answer`，只比较 `oracle_answer`。

## 5. text 与选项格式、图尺寸

- `text` 100% 含 `Available options`；0 条以 `?` 结尾；末字符：`.` 4500、`\n` 135（spatialreal）。
- 字符长度：min 139 / mean 566.98 / median 430 / p90 1071 / max 1071；词数：min 26 / mean 97.01 / median 71 / max 192。
- 按 task 字符长度：spatialmap 287–386（median 342）；mazenav 788–1071（median 1071）；spatialgrid 346–462（median 431）；spatialreal 139–401（median 180）。
- 选项行数：4 行（A,B,C,D 顺序）4135 条；1 行 500 条（全部 mazenav.qid2）。
- 图尺寸：宽度 512–3757（median 1168）、高度 512–2463（median 1168）；67 种 (w,h)；正方形 4434、横图 180、竖图 21。
- 主尺寸：`1168×1168` 1500（mazenav）、`512×512` 1500（spatialgrid）、`1281×1281` 1434（spatialmap 主）、spatialreal 46 种以 `2250×1500` 为主。
- `image_storage=parquet_embedded_or_local` 100%；manifest **不含图像字节**，仅 `image_sha256/width/height`。

## 6. 官方 evaluator 语义（`evaluation.py`，SHA `88d2a1b3…`）

- 抽取：`task = id.split('.')[0]`；`question_id = int(id.split('.')[-1])`。
- 比较字段：**`oracle_answer`**（非 option、非 full_answer）。
- 比较算式：`eval_result = int(str(data["oracle_answer"]).lower() in str(model_answer).lower())`，即 **子串包含**，方向为“oracle 为 needle、解析答案为 haystack”，并非相等。
- 逐 task 公式：`Acc = correct_answers / line_count`；`line_count` 在 try 之前对每行自增，异常跳过仍计入分母。
- **总分公式**：该 artifact 内**不存在**跨 task 总分，只输出逐 task CSV；任何 overall 数字必须自行定义，不能称官方总分。
- 输入文件名契约：`extract_model_name` 只接受 `m-*_w_reason.jsonl`，其它文件被 `main()` 静默跳过。
- handler：只有 spatialmap/mazenav/spatialgrid；**spatialreal 缺失** → 首行 `NameError`，后续行**复用上一条 `model_answer`**，属静默错误。
- 子串膨胀风险：oracle_answer 有 73 组“一个是另一个真子串”的成对情形；14 个纯数字答案中 7 对存在子串关系（如 `0` 会被 `10` 判对）。

## 7. 官方 inference 语义（`inference_vlm.py` / `inference_configs.py`）

- `--w_reason`：追加 “First, provide a concise answer in one sentence. Then, elaborate on the reasoning …”；bunny VQA 包一层 `USER: <image>\n{text}\n{reason} ASSISTANT:`，其它模型为 `{text}\n{reason}`。官方 evaluator 文件名契约与之耦合（只评 `m-*_w_reason.jsonl`）。
- `--completion` 追加 `Answer:`，不是 evaluator 识别的路径。
- 采样/生成**不统一**：CLI 默认 `temperature=0.2, top_p=0.9, repetition_penalty=1.0, max_new_tokens=1024`；bunny 走 greedy 且忽略 CLI 温标；qwen 走 `model.chat` 默认；cogvlm `do_sample=False, max_length=2048`；instructblip `do_sample=False, num_beams=5, top_p=0.9, repetition_penalty=1.5`；llava/merged bunny 传 `args.temperature`。冻结必须**按所选后端逐项固定**。
- inference 按 question_id 分组，`--first_k` 可按题型截断；输出字段 `id, answer, oracle_answer, oracle_option, oracle_full_answer, prompt, image`。

## 8. 现有 `spatialeval_metric.py` 的定位（禁止误称）

- 路径：`evaluation/spatialeval_metric.py`，内部名 `draft_answer_only_exact_match_not_official_parser`。
- 它是**草案 answer-only 精确匹配**，不含官方模型特定启发式解析、不含 w_reason 抽取分支、也不使用官方“子串包含”语义。
- **规则：不得将其标注、引用或报告为官方 SpatialEval 复现或官方 evaluator。**

## 9. 标签 materialize 与“非标签盲测”声明

- oracle 标签由离线 manifest builder 依据官方 parquet **程序化 materialize** 并已冻结；本轮仅以聚合方式审计，`labels_observed=true`。
- **未观察任何三臂预测或分数**：`gpu_runs=0`、`model_loaded=false`、`three_arm_predictions_seen=false`、`three_arm_scores_seen=false`。
- 因此：该集是**未用于 Selector 调参的外部确认集**；但因标签已 materialize/审计，**不是标签盲测（label-blind）**，不得据此做算法设计或超参选择。
- 现有草案 config `configs/2026-10-09-spatialeval-vqa-external-confirmation.draft.json` 仍写 `labels_observed=false`；后续预注册继任版应改为 `true`，本审计不修改该草案。

## 10. 正式预注册需冻结字段（建议清单，本审计不落 config）

1. **数据身份**：`MilaWang/SpatialEval`、config `vqa`、split `test`、`row_count=4635`、许可状态（仍 `NOT_EXPLICITLY_STATED`）。
2. **来源**：两个 parquet 路径/SHA256/行数（2318+2317）、字节数、manifest 路径/字节 5282303/SHA `03649b9c…`。
3. **schema 与完整性**：13 字段 schema、唯一 ID 集哈希 `4ea00a88…`、`record_sha256` 方案与链哈希 `68ab8b6e…`、`unique_id_count=4635`、`text`/`oracle_*` 逐字节冻结值。
4. **官方代码**：三份 artifact SHA256、commit `d82ba382…`、并声明 `spatialeval_metric.py` **非**官方解析器。
5. **task/评分语义**：task×question_id 映射；**spatialreal 处置**（排除或端口修正 handler + 单测）；比较字段 `oracle_answer`；子串包含口径（或另命名修正指标）；逐 task `Acc`；**显式定义并命名跨 task 聚合**；`m-*_w_reason.jsonl` 文件名契约。
6. **三臂与运行时**：`dense_edge576` / `corrected_integer_l2_b1_k96` / `conservative_l2_guarded_rerank_v2_b1_k96`；机制 id、selector checkpoint/config 哈希、K/G、模型身份（`model.safetensors` SHA）、processor 像素界 `min_pixels=65536`/`max_pixels=589824` 与 `image_grid_thw` 断言、prompt 字节（含 w_reason/completion 后缀）、逐后端生成参数、parquet 图像字节 materialize 与哈希方式。
7. **统计与判定**：bootstrap seed/次数、分层键（如 `task::question_id`）、判定门 `v2/dense ≥ 0.9` 与 `v2−l2 95% CI 下界 ≥ 0.0`、resume 去重键 `sample_id`、输出与 eval_summary 路径、`no-tuning` 与 `labels_observed=true` 声明。

## 11. 关键发现

1. **F1（info）**：四个固定 SHA256 全部匹配；4635 行 / 4635 唯一 ID；summary 的 2318+2317 与两 parquet SHA 一致；`record_sha256` 4635/4635 重算通过。
2. **F2（info）**：四 task = spatialmap/mazenav/spatialgrid 各 1500、spatialreal 135；question_id {0:1635, 1:1500, 2:1500}；mazenav qid2 为二值（A=Yes 333 / B=No 167）单选项行，其余均为有序 A–D 四选项。
3. **F3（high）**：官方 `evaluation.py` 用 `oracle_answer` **子串包含**、逐 task `Acc=correct/line_count`（无总分）、只读 `m-*_w_reason.jsonl`；**spatialreal 无 handler**，会复用上一条 `model_answer`（首行则 NameError）。
4. **F4（medium）**：`oracle_full_answer` 恒等于 `<option>. <answer>`，但只有 3466 条是 prompt 精确整行，1169 条仅是选项行前缀（选项文本包含答案）；官方评分不读 option/full_answer。
5. **F5（medium）**：`spatialeval_metric.py` 为草案 exact-match，绝非官方复现；标签已程序化 materialize 且无任何三臂预测/分数被观察，故属未用于 Selector 调参的外部确认集，但**不是标签盲测**。

## 12. 限制

- manifest 不含图像字节（`parquet_embedded_or_local`），仅审计 `image_sha256/width/height`；任何图像重物化必须重 pin parquet SHA。
- 只读到三段 provenance；官方仓库的 `utils/format_filename`、数据加载器与任何总分聚合不可得，未能审计。
- spatialreal 缺口来自源码阅读，未实际执行（本轮无推理）。
- 许可状态仍 `NOT_EXPLICITLY_STATED`，本审计不解决许可/分发权。
- 全程无模型/GPU/网络/推理，仅验证 schema 与官方评分语义。
