# sparse_flash_attention

## 描述

DeepSeek V3.2推出的稀疏Attention实现，通过选取topk个相关性最高的历史token与当前query向量进行计算来减少计算量。
更多参考：https://gitcode.com/cann/cann-recipes-infer/blob/master/ops/ascendc/docs/custom-npu_sparse_flash_attention.md。

## 输入参数

| Name | DType | Shape | Optional | Inplace | Format | Description |
|------|-------|-------|----------|---------|--------|-------------|
| query | Tensor[bfloat16] | - BNSD：(batch, s, num_heads, 512)<br>- TND：(num_tokens, num_heads, 512) | No | No | ND | 查询向量中不参与位置编码计算的部分。 |
| key | Tensor[bfloat16] | (block_num, block_size, 1, 512) | No | No | ND | key向量中参与位置编码计算的部分。 |
| value | Tensor[bfloat16] | (block_num, block_size, 1, 512) | No | No | ND | value向量。 |
| sparse_indices | Tensor[int32] | (batch, q_s, 1, sparse_size) | No | No | ND | 表示需要参与计算的kv的索引。 |
| block_table | Tensor[int32] | (batch, max_kv_seq_len_per_block) | Yes | No | ND | 表示PagedAttention计算中kv的block映射表。 |
| actual_seq_lengths_query | Tensor[int32] | (batch,) | Yes | No | ND | 表示每个batch的query的token长度。<br>如果不指定，表示每个batch的token长度与`query`的shape的`S`值相同。<br>`query`的layout为TND时必须传入，且每个元素的值表示当前batch与之前所有batch的token数总和（前缀和）。 |
| actual_seq_lenghts_kv | Tensor[int32] | (batch,) | Yes | No | ND | 表示每个batch的key和value的token长度。<br>如果不指定，表示每个batch的token长度与`key`的shape的`S`值相同。 |
| query_rope | Tensor[bfloat16] | BNSD：(batch, s, num_heads, 64)<br>- TND：(num_tokens, num_heads, 64) | Yes | No | ND | MLA查询向量中参与位置编码计算的部分。 |
| key_rope | Tensor[bfloat16] | (block_num, block_size, 1, 64) | Yes | No | ND | MLA key向量中参与位置编码计算的部分。 |
| scale_value | float | - | No | - | - | `query`和`key`矩阵乘结果的缩放系数，取值范围(0, 1]，默认值`1.0`。 |
| sparse_block_size | int | - | No | - | - | 在计算importance score时使用，表示sparse阶段block大小，默认值`1`。 |
| layout_query | int | - | No | - | - | 表示`query`的数据排布格式，支持`0`-"BSND"和`1`-"TND"，默认值`0`。 |
| layout_kv | int | - | No | - | - | 表示`key`的数据排布格式，支持0-"BSND"和1-"PA_BSND"，"PA_BSND"在使能PagedAttention时使用。默认值`0`。 |
| sparse_mode | int | - | No | - | - | 表示sparse模式：<br>- 0：代表全部计算。<br>- 3：代表rightDownCausal模式的mask，对应以右下定点往左上为划分线的下三角场景。默认值`3`. |

## 输出参数

| Name | DType | Shape | Format | Description |
|------|-------|-------|--------|-------------|
| attention_out | Tensor[bfloat16] | 与输入`query`保持一致 | ND | Attention计算输出 |

## 约束限制

- `num_heads`支持`1/2/4/8/16/32/64/128`。
- `block_size`取值为`16`的倍数，最大支持`1024`。
- `layout_query`为`TND`且`layout_kv`为`BSND`场景不支持。

## 支持产品

- Atlas A3推理系列产品

## 使用示例

```python
import mindspore as ms
import ms_custom_ops
import numpy as np

query = ms.Tensor(np.random.uniform(-10.0, 10.0, size=(4, 1, 128, 512)), dtype=ms.bfloat16)
key = ms.Tensor(np.random.uniform(-5.0, 10.0, size=(4, 8192, 1, 512)), dtype=ms.bfloat16)
value = ms.Tensor(np.random.uniform(-5.0, 10.0, size=(4, 8192, 1, 512)), dtype=ms.bfloat16)
sparse_indices = ms.Tensor(np.random.uniform(0, 16, size=(4, 1, 1, 2048)), dtype=ms.int32)
query_rope = ms.Tensor(np.random.uniform(-10.0, 10.0, size=(4, 1, 128, 64)), dtype=ms.bfloat16)
key_rope = ms.Tensor(np.random.uniform(-5.0, 10.0, size=(4, 8192, 1, 64)), dtype=ms.bfloat16)
actual_seq_lengths_query = ms.Tensor([1] * 4, dtype=ms.int32)
actual_seq_lengths_kv = ms.Tensor([4096] * 4, dtype=ms.int32)

out = ms_custom_ops.sparse_flash_attention(query, key, value, sparse_indices, None,
                                           actual_seq_lengths_query, actual_seq_lengths_kv,
                                           query_rope, key_rope, 0.4, 1, 0, 0, 3)
print(out)
```