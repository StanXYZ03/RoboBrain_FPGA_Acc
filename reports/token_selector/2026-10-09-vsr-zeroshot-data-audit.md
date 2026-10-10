# VSR Zero-Shot 数据审计

- 状态：`VSR_ZEROSHOT_METADATA_AUDIT_PASS_IMAGES_INCOMPLETE`。
- 官方提交：`b27a0af0ee1462d2b6b92c8c83e869d9254a241a`。
- test 只核对 SHA、字节数和行数；未解析 JSON，未读取标签、caption 或 relation。

## 隔离情况

- train：3489 条、2249 张图、60 种关系。
- dev：340 条、227 张图、40 种关系。
- train/dev 概念重叠：0；图像重叠：5。
- 相比 random split 的 558 张 train/dev 图像重叠，zero-shot 更适合关系保持 Selector 的发展验证。

## 本机覆盖

- train：缓存 45/2249，缺失 2204。
- dev：缓存 5/227，缺失 222。

## 决策

建议使用 zero-shot train/dev 发展，继续封存 zero-shot test。先只补齐 222 张 dev 图像并运行短输出预算曲线；
只有关系保持候选需要训练时，才补齐 2204 张 train 图像。
