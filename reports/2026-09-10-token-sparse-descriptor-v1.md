# Token/Block Sparse Descriptor v1 软件合同

## 结论

完成与任务名称、Selector算法、保留率和AI主平台无关的视觉Token稀疏描述符v1。C参考编码器/零拷贝解析器在普通构建和ASan/UBSan下均通过10类测试，可作为未来PS软件、主算力平台适配层和HLS前端之间的稳定数据合同。

本里程碑只冻结二进制格式，不固化当前未通过Q8质量门的75%或87.5% Oracle选择策略，不代表已完成HLS、RTL、综合或板端验证。

## 合同摘要

- 64字节little-endian固定头，magic为ASCII `RTSD`，版本为1。
- 条目包含严格递增且唯一的原序列`uint32`索引，并可携带0至4个原始位置分量。
- 支持全长Identity标志；Identity要求`kept_count == full_count`且索引严格为`0..N-1`。
- payload与头部分别使用IEEE CRC-32；头CRC计算时CRC字段自身置0。
- v1保留字段必须为0，未知flag、长度不一致、截断、重复或越界索引均拒绝。
- 解析器不分配内存，返回指向输入payload的只读视图，便于嵌入式运行时使用。

## 验证结果

普通构建与Sanitizer构建均通过以下10类测试：

1. CRC-32标准向量。
2. 4维原始位置的稀疏序列化/解析往返。
3. 全长Identity往返。
4. 错误magic拒绝。
5. payload CRC损坏拒绝。
6. header CRC损坏拒绝。
7. CRC合法但保留字段非零时拒绝。
8. 截断描述符拒绝。
9. 重复与越界索引拒绝。
10. 输出容量不足拒绝。

编译选项包括`-std=c11 -Wall -Wextra -Werror -pedantic`；Sanitizer复验包括AddressSanitizer和UndefinedBehaviorSanitizer。

随后完成独立跨语言互操作门：C工具生成124字节固定稀疏描述符，Python独立实现解析后重新编码，与C输出逐字节一致。golden条目为原索引`[1, 4, 7]`及4维原始位置；它只验证格式，不代表任何Selector策略。

## 证据

- 接口：`fpga_accel/token_sparse_descriptor/include/token_sparse_descriptor.h`，SHA256 `5a36c11bf710138b5c92691cbf340349a8b64537ea590c54a359ccef0f74ccf1`
- 实现：`fpga_accel/token_sparse_descriptor/src/token_sparse_descriptor.c`，SHA256 `c1fc07e73a0f00782598e3c0efb9172ab3c0bd82474e04cc21a265fb1fe267b1`
- 测试：`fpga_accel/token_sparse_descriptor/tests/test_token_sparse_descriptor.c`，SHA256 `7b7c1868645964817bf7de9fb95f30e83339cb70b638865dcadc135e2593106c`
- 格式规范：`fpga_accel/token_sparse_descriptor/README.md`，SHA256 `7ef9ece1159208574bd63291ecf64f0a788932f51f287ca9f9c3644ef99ccbe1`
- 普通测试二进制：SHA256 `3c5e8b39bc2554d7d5bb5afd8d26dd91ba777453637b344866cf1a9b8b11d9fa`
- Sanitizer测试二进制：SHA256 `5ca71bf86e4faaa7dde3fc4890559397ee1c786914edce599e0a8a9c49dbe9db`
- Golden binary：`fpga_accel/token_sparse_descriptor/artifacts/golden_sparse_v1.bin`，124字节，SHA256 `53b85bfafda5d661a5c229d80cf4a22d0f25265df52f8eb9e0da395feb8793e8`
- Golden语义摘要：`fpga_accel/token_sparse_descriptor/artifacts/golden_sparse_v1.json`，SHA256 `b09a1cdd1aa83ee792ba01d879353e98cc60c4f3e1de2fae252887fda944b72a`
- 独立Python codec：`evaluation/token_sparse_descriptor.py`，SHA256 `303747441a2e3492d09a053b48060df3f02b3a37feae350f5a88cbad369e5b56`
- 跨语言构建脚本：`evaluation/build_token_sparse_descriptor_fixture.py`，SHA256 `d9d451e49f30819bc7202b02fa625c5a49de31505a768cdfff17882b7e0ad75a`

复验命令：

```bash
make -C fpga_accel/token_sparse_descriptor test
python3 evaluation/build_token_sparse_descriptor_fixture.py
```

## 下一步边界

只有新的可部署Selector通过独立Q8质量门后，才把其输出接入该合同。后续HLS/RTL只需实现v1校验、索引读取和gather/compact数据面，不应包含任务名或特定模型输出规则。当前不生成bitstream、不访问板卡。
