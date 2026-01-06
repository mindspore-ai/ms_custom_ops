# reshape_and_cache_npd算子

## 描述

reshape_and_cache_npd - store KV cache in NPD format, N-KV head number, P - block Size, D - embedding size

## 输入参数

| Name                   | DType                             | Shape                                                                                                                   | Optional | Inplace | Format | Description                    |
|------------------------|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------|----------|---------|--------|--------------------------------|
| key                    | Tensor[float16/bfloat16]     | (num_tokens, num_head*head_dim)                                                                                        | No       | No      | ND     | key 张量                       |
| value                  | Tensor[float16/bfloat16]     | (num_tokens, num_head*head_dim)                                                                                        | No      | No      | ND     | value 张量                     |
| key_cache              | Tensor[float16/bfloat16]     | NPD: (num_blocks, num_head, block_size, head_dim) | No       | No     | NPD  | key_cache 张量                 |
| value_cache            | Tensor[float16/bfloat16]     | NPD: (num_blocks, num_head, block_size, head_dim) | No       | No     | NPD  | value_cache 张量               |
| slot_mapping           | Tensor[int32]                      | (num_tokens,)                                                                                                           | No       | No      | ND     | slot_mapping 张量              |
| cache_mode             | int                                | -                                                                                                                       | No       | -       | -      | 缓存模式：0 表示 ND 格式，1 表示 NPD 格式 |

## 输出参数

| Name   | DType           | Shape                                | Description |
|--------|-----------------|--------------------------------------|-------------|
| key_out | Tensor[float16/bfloat16] | (num_tokens/block_size, num_head, block_size, head_dim)     | key out in NPD format    |
| value_out | Tensor[float16/bfloat16] | (num_tokens/block_size, num_head, block_size, head_dim)     | key out in NPD format    |

## 使用示例

```python
import mindspore as ms
import ms_custom_ops
import numpy as np

# 创建输入张量
key = ms.Tensor(np.random.rand(16, 4096), ms.float16)
value = ms.Tensor(np.random.rand(16, 4096), ms.float16)
key_cache = ms.Tensor(np.random.rand(1024, 32, 16, 128), ms.float16)
value_cache = ms.Tensor(np.random.rand(1024, 32, 16, 128), ms.float16)
slot_mapping = ms.Tensor(np.arange(128), ms.int32)
q_seq = ms.Tensor(np.arange(1), ms.int32)
kv_seq = ms.Tensor(np.arange(1), ms.int32)
block_tbl = ms.Tensor(np.arange(128).reshape(1,128), ms.int32)
# 调用算子
k_out, v_out = ms_custom_ops.reshape_and_cache_npd(
    key=key,
    value=value,
    key_cache=key_cache,
    value_cache=value_cache,
    slot_mapping=slot_mapping,
    q_seq=q_seq,
    kv_seq=kv_seq,
    block_tbl=block_tbl,
    cache_mode=1 #ND0,NPD1
)
```