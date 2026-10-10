# Token/Block Sparse Descriptor v1

该目录定义与任务、Selector算法和AI主平台无关的视觉Token索引合同。它不固化当前未通过质量门的75%或87.5% Oracle策略。

## 二进制格式

所有整数均为little-endian。64字节头部后紧跟kept_count个定长条目：

| 偏移 | 字节 | 字段 |
|---:|---:|---|
| 0 | 4 | magic，ASCII RTSD |
| 4 | 2 | version，当前为1 |
| 6 | 2 | header_bytes，固定64 |
| 8 | 4 | flags |
| 12 | 4 | full_count |
| 16 | 4 | kept_count |
| 20 | 2 | entry_bytes，4乘以(1+position_dims) |
| 22 | 2 | position_dims，0至4 |
| 24 | 4 | payload_bytes |
| 28 | 4 | payload CRC32 |
| 32 | 4 | header CRC32；计算时该字段置0 |
| 36 | 28 | 保留，v1必须为0 |

条目第一个uint32是原序列索引，后跟0至4个原始位置分量。索引必须严格递增、唯一且小于full_count。CRC使用IEEE CRC-32多项式0xEDB88320。

Flags：

- HAS_POSITIONS=0x1
- IDENTITY=0x2，要求kept_count等于full_count且索引为0至N-1
- SORTED_UNIQUE=0x4，v1必须设置

解析器不分配内存，返回指向输入payload的只读视图，适合PS软件和后续HLS前端共同使用。v1只描述保留集合与原始位置，不定义评分、任务名、保留率或硬件调度策略。

## 验证

    make test

测试覆盖标准CRC向量、稀疏/身份往返、magic、头部CRC、payload CRC、非零保留字段、截断、重复/越界索引和容量不足。
