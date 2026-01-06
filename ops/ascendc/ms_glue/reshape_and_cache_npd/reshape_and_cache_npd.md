# reshape_and_cache_npd算子

## 描述

reshape_and_cache_npd - the operator responsible for storing key/value pairs in a KV cache in ND or NPD format. It also produces two outputs: the current key tensor and the value tensor, along with a history if required.

### 名称

- 算子名：`reshape_and_cache_npd`

### 输入参数

T - token number
BN - block number
N - head number
P - block size
D - embedding
B - Batch

| Name             | DType                     | Shape                                   | Optional | Format | Description |
|------------------|---------------------------|--------------------------------------   |----------|--------|-------------|
| key              | Tensor[float16/bfloat16/int8]  | TH/NPD                                  | No       | ND     | Key tensor |
| value            | Tensor[float16/bfloat16/int8]  | TH/NPD                                  | No       | ND     | Value tensor |
| key_cache        | Tensor[float16/bfloat16/int8]  | BNSD: (BN, N, P, D); BSND: (BN, P, N, D)| No       | ND     | key cache |
| value_cache      | Tensor[float16/bfloat16/int8]  | BNSD: (BN, N, P, D); BSND: (BN, P, N, D)| No       | ND     | value cache |
| actual_seq_qlen  | Tensor[int32]             | (B,)                                    | No       | ND     | Number of query tokens in each B element |
| actual_seq_kvlen | Tensor[int32]             | (B,)                                    | No       | ND     | Number of key/value tokens in each batch element |
| slot_mapping     | Tensor[int32]             | (T,)                                    | No       | ND     | Mapping of token into Key value cache |
| block_tbl        | Tensor[int32]             | (B, max_query_len)                      | No       | ND     | Mapping of page into Key value cache |
| kv_cache_layout  | string                    | -                                       | No       | -      | Key value cache in\out layout: supported values ND or NPD，默认 NPD |
| key_value_layout | string                    | -                                       | No       | -      | Key/Value output layout: TH or NPD，默认 NPD |

#### 参数补充说明

- q_seq_len / kv_seq_len
    - Both must be provided and should be located on CPU memory.

## 输出参数

| Name      | DType                    | Shape                                     | Description   |
|-----------|--------------------------|-------------------------------------------|---------------|
| key_out   | Tensor[float16/bfloat16/int8] | NPD:(T+History, N * P * D) TH: (T, N * D) | key output    |
| value_out | Tensor[float16/bfloat16/int8] | NPD:(T+History, N * P * D) TH: (T, N * D) | value output  |

## 使用示例

```python
import mindspore as ms
import ms_custom_ops
import numpy as np
from mindspore import Tensor, context


np.random.seed(0)
context.set_context(device_target="Ascend", mode=context.GRAPH_MODE)

T = 16
BN = 1024
P = 32
N = 8
D = 128
B = 1

key = ms.Tensor(np.random.rand(T, N * D), ms.float16)
value = ms.Tensor(np.random.rand(T, N * D), ms.float16)
key_cache = ms.Tensor(np.random.rand(BN, N, P, D), ms.float16)
value_cache = ms.Tensor(np.random.rand(BN, N, P, D), ms.float16)
slot_mapping = ms.Tensor(np.arange(T), ms.int32)
q_seq = ms.Tensor(np.full(B, T).astype(np.int32)).move_to("CPU")
kv_seq = ms.Tensor(np.full(B, T).astype(np.int32)).move_to("CPU")
block_tbl = ms.Tensor(np.arange(BN).reshape(B,BN), ms.int32)
# 调用算子
k_out, v_out = ms_custom_ops.reshape_and_cache_npd(
    key=key,
    value=value,
    key_cache=key_cache,
    value_cache=value_cache,
    slot_mapping=slot_mapping,
    actual_seq_qlen=q_seq,
    actual_seq_kvlen=kv_seq,
    block_tbl=block_tbl,
    kv_cache_layout="BNSD",  
    key_value_layout="NPD"
)
print("shape ", k_out.shape)
```
