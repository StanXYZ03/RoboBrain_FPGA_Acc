# SpatialEval 外部确认 r1 作废说明

- 状态：`R1_SUPERSEDED_BEFORE_SCORING`
- r1 配置：`configs/2026-10-09-spatialeval-vqa-three-arm-external-confirmation.preregistered.json`
- r1 SHA256：`16d26764d6a587af3cd68e12d79929b7301e93150ca7e09300713461921e002f`
- 继任 r2：`configs/2026-10-09-spatialeval-vqa-three-arm-external-confirmation-r2.preregistered.json`
- r2 SHA256：`bd8e1a23e0d3fba03315c93d10d88856c4864aa133efc32331360b9d4d708014`

r1 的 pin 列表错误地直接固定了一个实际未被 v2 loader 使用的旧投影文件，而 v2 真正加载的
`best.npz`、`Wv_int8_rowmajor.bin` 和 `Wv_per_tensor_scale_f32le.bin` 只通过旧 r4 配置间接引用。
这不会改变已经运行的 Dense 算法，但不能满足外部确认对候选 checkpoint 的直接 fail-closed
身份验证要求。

发现缺口时 Dense 已生成 93 条 append-only 记录。未读取或评分输出答案，只查看了 G、K、阶段
计时和输出 token 数以估算运行时间。进程随后由本项目主动以 SIGINT 停止，93 条记录原文件保持
不变，并复制为不可变证据：

`artifacts/token_selector/2026-10-09-spatialeval-vqa-three-arm/dense_prefix93_v1_unscored.jsonl`

该快照为 801929 字节，SHA256
`179b68b9f3b008357f4458956c75236e8a181439abe890cbb1f42263bbca0196`。r2 直接固定快照和三个
实际 checkpoint；算法、数据、Prompt、输出路径、指标、bootstrap 和成功门全部不变。Dense
从第 94 条 append-only 续跑，前缀不得删除、改写或单独评分。r1 不得再用于启动新运行。
