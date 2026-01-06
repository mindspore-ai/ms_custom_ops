# FlashAttention Encoder 算子

## 描述

FlashAttention is a high-performance implementation of self-attention, mainly reducing memory usage and improving throughput through blocking/recomputation, mask compression, and better memory access via NPD shape.
where N is head number, P is page size and D is head size

## 接口与输入输出

### 名称

- 算子名：`unpad_fa_npd`

### 输入参数

| Name             | DType                     | Shape                     | Optional | Format | Description |
|------------------|---------------------------|---------------------------|----------|--------|-------------|
| query            | Tensor[float16/bfloat16]  | TH                        | No       | ND     | Query tensor |
| key              | Tensor[float16/bfloat16]  | TH/NPD                    | No       | ND     | Key tensor |
| value            | Tensor[float16/bfloat16]  | TH/NPD                    | No       | ND     | Value tensor |
| attn_mask        | Tensor[float16/bfloat16]  | (128,128)                 | No       | ND     | Upper triangular mask or None |
| actual_seq_qlen  | Tensor[int32]             | (batch,)                  | No       | ND     | Number of query tokens in each batch element |
| actual_seq_kvlen | Tensor[int32]             | (batch,)                  | No       | ND     | Number of key/value tokens in each batch element |
| head_num         | int                       | -                         | No       | -      | 注意力头数（H），默认 32（需显式设置 >0） |
| scale_value      | float                     | -                         | No       | -      | QK 缩放系数 `qk_scale`（通常为 1/sqrt(head_dim)），默认 1.0 |
| q_input_layout   | string                    | -                         | No       | -      | Query input layout: TH only TH is supported ，默认 TH |
| kv_input_layout  | string                    | -                         | No       | -      | Key/Value input layout: TH or NPD，默认 NPD |
| block_size       | int                       | -                         | No       | -      | Page table, block size，默认 32 |

#### 参数补充说明

- attn_mask
    - The mask should be an upper triangular matrix containing values of 0 and 1 for bfloat16, and values of 0 and -10000 for float16.

- q_seq_len / kv_seq_len
    - Both must be provided and should be located on CPU memory.

### 输出参数

| Name          | DType                    | Shape           | Description |
|---------------|--------------------------|-----------------|-------------|
| attention_out | Tensor[float16/bfloat16] | 与 query 对齐   | 注意力输出   |

### miscellaneous 杂项

- Typically used with reshape_and_cache_npd to efficiently manage all key/value permutations within an operator.

### Python 使用示例

```python
import mindspore as ms
from mindspore import Tensor, context
import ms_custom_ops
import numpy as np

# 创建输入张量
np.random.seed(0)
context.set_context(device_target="Ascend", mode=context.PYNATIVE_MODE)

head_num = 8
kv_head_num = 4
head_dim = 128
scale_value = float(1.0 / np.sqrt(head_dim))
block_size = 32

q_seq = np.array([block_size, block_size*2], dtype=np.int32)
kv_seq = np.array([block_size, block_size*2], dtype=np.int32)
npd_value = npd_value.transpose(0,2,1,3)
npd_key = npd_key.reshape(kv_tokens, kv_head_num * head_dim)
npd_value = npd_value.reshape(kv_tokens, kv_head_num * head_dim)

attn_mask = Tensor(np.ones(shape=(128, 128)).astype(np.float32), ms.float16)
npd_key = Tensor(npd_key, ms.float16)
npd_value = Tensor(npd_value, ms.float16)

# 调用算子
attention_out = ms_custom_ops.unpad_fa_npd(
    query=query,
    key=npd_key,
    value=npd_value,
    attn_mask=attn_mask,
    actual_seq_qlen=q_seq_len,
    actual_seq_kvlen=kv_seq_len,
    head_num=head_num,
    scale_value=scale_value,
    q_input_layout="TH",
    kv_input_layout="NPD",
    block_size=block_size
)
```
