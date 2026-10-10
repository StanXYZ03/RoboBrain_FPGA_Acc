# VSR 发展数据审计

- 状态：`VSR_METADATA_AUDIT_PASS_IMAGES_INCOMPLETE`。
- 官方提交：`b27a0af0ee1462d2b6b92c8c83e869d9254a241a`。
- test 文件只核对 SHA、字节数和行数；未解析 JSON，未读取标签、caption 或 relation。

## 元数据

- train：7680 条，4968 张图，64 种关系。
- dev：1097 条，1017 张图，60 种关系。
- train/dev 唯一图像重叠：558 张。

## 本机覆盖

- train：缓存 115/4968，缺失 4853，覆盖率 2.31%。
- dev：缓存 20/1017，缺失 997，覆盖率 1.97%。
- 去重后的 train+dev 缺失图像总数：5307。

## 决策

当前不能启动 VSR GPU 实验。建议先只下载 997 张缺失 dev 图像，完成 Dense/L2/v2 的短输出预算曲线；
是否再下载 train 图像，取决于 dev 诊断是否支持继续训练关系保持 Selector。
