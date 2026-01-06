# paged_cache_load

## 描述

load and concat key, value from kv_cache using block_tables and context_lens.
Support dtype: fp16, bf16, int8
Support format: ND, NZ

## 输入参数

| Name | DType | Shape | Optional | Inplace | Format | Description |
|------|-------|-------|----------|---------|--------|-------------|
| key_cache | Tensor(fp16, bf16, int8) | [num_blocks, block_size, num_heads, head_size_k] | No | No | ND, NZ | 输入tensor，数据类型为(fp16, bf16, int8), 原始key_cache |
| value_cache | Tensor(fp16, bf16, int8) | [num_blocks, block_size, num_heads, head_size_v] | No | No | ND, NZ | 输入tensor，数据类型为(fp16, bf16, int8), 原始value_cache |
| block_tables | Tensor(int32) | [batch, block_indices] | No | No | ND | 输入tensor，数据类型为(int32) |
| seq_lens | Tensor(int32) | [batch 或 batch+1] | No | No | ND | 输入tensor，数据类型为(int32), 记录每个batch的context length， 支持两种类型：每个元素是一个batch的长度 或 累加和模式|
| seq_starts | Tensor(int32) | [batch] | No | No | ND | 可选输入tensor，数据类型为(int32), 记录seq的起始点 |
| kv_cache_cfg | int |  | No | No |  | default 0, 0->nd, 1->nz |
| is_seq_lens_cumsum_type | bool |  | No | No |  | default false，false表示不使用累加和模式，只有ND格式下支持true |
| has_seq_starts | bool |  | No | No |  | default false，false表示没有seq_starts输入，只有ND格式下支持true |

## 输出参数

| Name | DType | Shape | Description |
|------|-------|-------|-------------|
| key_out | Tensor(fp16, bf16, int8) | [num_tokens, num_heads, head_size_k] | 拼接后的key，ND格式 |
| value_out | Tensor(fp16, bf16, int8) | [num_tokens, num_heads, head_size_k] | 拼接后的value，ND格式 |

## 使用示例

### 基本使用示例（常规模式）

```python

import os
import numpy as np
from mindspore import Tensor, context
import mindspore as ms
import random
import ms_custom_ops
class AsdPagedCacheLoadCustom(ms.nn.Cell):
def __init__(self):
    super().__init__()

def construct(self, key_cache, value_cache, block_table, seq_lens, seq_starts, kv_cache_cfg,
              is_seq_lens_cumsum_type, has_seq_starts):
    return ms_custom_ops.paged_cache_load(key_cache, value_cache, block_table, seq_lens, seq_starts, kv_cache_cfg,
                                  is_seq_lens_cumsum_type, has_seq_starts)

#  ND INPUT WITH SEQ_STARTS
# dtype is in [ms.float16, ms.bfloat16, ms.int8]
if dtype == ms.float16:
    key_cache = np.random.randint(1, 11,
                                  size=(num_blocks, block_size, num_heads, head_size_k)).astype(np.float16)
    value_cache = np.random.randint(1, 11,
                                    size=(num_blocks, block_size, num_heads, head_size_v)).astype(np.float16)
elif dtype == ms.bfloat16:
    key_cache = np.random.randint(1, 11,
                                  size=(num_blocks, block_size, num_heads, head_size_k)).astype(np.float32)
    value_cache = np.random.randint(1, 11,
                                    size=(num_blocks, block_size, num_heads, head_size_v)).astype(np.float32)
else:
    key_cache = np.random.randint(1, 11,
                                  size=(num_blocks, block_size, num_heads, head_size_k)).astype(np.int8)
    value_cache = np.random.randint(1, 11,
                                    size=(num_blocks, block_size, num_heads, head_size_v)).astype(np.int8)
context_lens = [random.randint(1, 1024) for _ in range(num_tokens)]
max_context_len = max(context_lens)
max_num_blocks_per_req = (max_context_len + block_size -1) // block_size + 4
block_tables = []
for _ in range(num_tokens):
    block_table = [
        random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_req)
    ]
    block_tables.append(block_table)
cu_context_lens = [0]
for elem in context_lens:
    cu_context_lens.append(cu_context_lens[-1] + elem)
seq_starts = [random.randint(0, 4) * block_size for _ in range(num_tokens)]
context_lens = np.array(cu_context_lens).astype(np.int32)
block_tables = np.array(block_tables).astype(np.int32)
seq_starts = np.array(seq_starts).astype(np.int32)
sum_context_lens = context_lens[-1]
seq_starts_tensor = None if seq_starts is None else Tensor(seq_starts)
net = AsdPagedCacheLoadCustom()
key_out, value_out = net(
    Tensor(key_cache).astype(dtype),
    Tensor(value_cache).astype(dtype),
    Tensor(block_tables),
    Tensor(context_lens),
    seq_starts_tensor,
    format_type, cu_seq_lens, has_seq_starts
)
print("key_out is ", key_out)
print("value_out is ", value_out)

# NZ INPUT WITHOUT SEQ_STARTS
# dtype is in [ms.float16, ms.bfloat16, ms.int8]
if dtype == ms.float16:
    key_cache = np.random.randint(
        1, 11, size=(num_blocks, num_heads * head_size_k // 16, block_size, 16)).astype(np.float16)
    value_cache = np.random.randint(
        1, 11, size=(num_blocks, num_heads * head_size_k // 16, block_size, 16)).astype(np.float16)
elif dtype == ms.bfloat16:
    key_cache = np.random.randint(
        1, 11, size=(num_blocks, num_heads * head_size_k // 16, block_size, 16)).astype(np.float32)
    value_cache = np.random.randint(
        1, 11, size=(num_blocks, num_heads * head_size_k // 16, block_size, 16)).astype(np.float32)
else:
    key_cache = np.random.randint(
        1, 11, size=(num_blocks, num_heads * head_size_k // 32, block_size, 32)).astype(np.int8)
    value_cache = np.random.randint(
        1, 11, size=(num_blocks, num_heads * head_size_k // 32, block_size, 32)).astype(np.int8)
context_lens = [random.randint(1, 1024) for _ in range(num_tokens)]
max_context_len = max(context_lens)
max_num_blocks_per_req = (max_context_len + block_size -1) // block_size
block_tables = []
for _ in range(num_tokens):
    block_table = [
        random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_req)
    ]
    block_tables.append(block_table)

context_lens = np.array(context_lens).astype(np.int32)
block_tables = np.array(block_tables).astype(np.int32)
sum_context_lens = sum(context_lens)
seq_starts_tensor = None if seq_starts is None else Tensor(seq_starts)
net = AsdPagedCacheLoadCustom()
key_out, value_out = net(
    Tensor(key_cache).astype(dtype),
    Tensor(value_cache).astype(dtype),
    Tensor(block_tables),
    Tensor(context_lens),
    seq_starts_tensor,
    format_type, cu_seq_lens, has_seq_starts
)
print("key_out is ", key_out)
print("value_out is ", value_out)

```
