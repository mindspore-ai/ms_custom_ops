# lightning_indexer

## 描述

DeepSeek V3.2中用于从历史token中获取与当前token最相关kv索引的计算方法。
更多参考：https://gitcode.com/cann/cann-recipes-infer/blob/master/ops/ascendc/docs/custom-npu_lightning_indexer.md。

## 输入参数

| Name | DType | Shape | Optional | Inplace | Format | Description |
|------|-------|-------|----------|---------|--------|-------------|
| query | Tensor[bfloat16] | - BNSD：(batch, s, num_heads, 128)<br>- TND：(num_tokens, num_heads, 128) | No | No | ND | 查询向量中不参与位置编码计算的部分。 |
| key | Tensor[bfloat16] | (block_num, block_size, 1, 128) | No | No | ND | key向量中参与位置编码计算的部分。 |
| weights | Tensor[bfloat16] | - BSND：(batch, s, num_heads, 1)<br>- TND：(num_tokens, num_heads, 1) | No | No | ND | 计算`query`与历史kv相关性的权重系数。 |
| sparse_indices | Tensor[int32] | (batch, q_s, 1, sparse_size) | No | No | ND | 表示需要参与计算的kv的索引。 |
| actual_seq_lengths_query | Tensor[int32] | (batch,) | Yes | No | ND | 表示每个batch的query的token长度。<br>如果不指定，表示每个batch的token长度与`query`的shape的`S`值相同。<br>`query`的layout为TND时必须传入，且每个元素的值表示当前batch与之前所有batch的token数总和（前缀和）。 |
| actual_seq_lenghts_key | Tensor[int32] | (batch,) | Yes | No | ND | 表示每个batch的key和value的token长度。<br>如果不指定，表示每个batch的token长度与`key`的shape的`S`值相同。 |
| block_table | Tensor[int32] | (batch, max_kv_seq_len_per_block) | Yes | No | ND | 表示PagedAttention计算中kv的block映射表。 |
| layout_query | int | - | No | - | - | 表示`query`的数据排布格式，支持`0`-"BSND"和`1`-"TND"，默认值`0`。 |
| layout_key | int | - | No | - | - | 表示`key`的数据排布格式，当前仅支持`0`-"PA_BSND"。默认值`0`。|
| sparse_count | int | - | No | - | - | 表示需topk极端需要保留的block数量，支持`1-2048`，默认值`2048`。 |
| sparse_mode | int | - | No | - | - | 表示sparse模式：<br>- 0：代表全部计算。<br>- 3：代表rightDownCausal模式的mask，对应以右下定点往左上为划分线的下三角场景。默认值`3`。 |

## 输出参数

| Name | DType | Shape | Format | Description |
|------|-------|-------|--------|-------------|
| sparse_indices | Tensor[int32] | layout_query为"TND"：(num_tokens, 1, sparse_count)<br>layout_query为"BSND"：(batch, s, 1, sparse_count) | ND | 选取的kv的索引 |

## 约束限制

- `num_heads`仅支持`64`。
- `block_size`取值为`16`的倍数，最大支持`1024`

## 支持产品

- Atlas A3推理系列产品

## 使用示例

```python
import mindspore as ms
import ms_custom_ops
import numpy as np

query = ms.Tensor(np.random.uniform(-10.0, 10.0, size=(1, 1, 64, 128)), dtype=ms.bfloat16)
key = ms.Tensor(np.random.uniform(-5.0, 10.0, size=(64, 256, 1, 128)), dtype=ms.bfloat16)
weights = ms.Tensor(np.random.uniform(-1.0, 1.0, size=(1, 1, 64, 1)), dtype=ms.bfloat16)
actual_seq_lengths_query = ms.Tensor([1], dtype=ms.int32)
actual_seq_lengths_key = ms.Tensor([8192], dtype=ms.int32)
block_table = ms.Tensor([range(64)], dtype=ms.int32).reshape(1, -1)

out = ms_custom_ops.lightning_indexer(query, key, weights,
                                      actual_seq_lengths_query, actual_seq_lengths_key,
                                      block_table, 0, 0, 2048, 3)
print(out)
```