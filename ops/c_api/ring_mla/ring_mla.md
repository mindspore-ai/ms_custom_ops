## RingMLA 算子

## 描述

RingMLA（Ring Multi-head Latent Attention）是一个多头潜在注意力算子，会将 Query/Key 拆分为基础部分（base）和 RoPE 部分（rope）分别输入，并支持 KV 头分组。
算子可选地在 `calc_type=0` 时结合 `o_prev`、`lse_prev` 实现 ring 形式的增量更新，在 `calc_type=1` 时计算独立注意力。适用于长序列或分块推理场景。

## 接口与输入输出

### 名称

- **算子名**：`ring_mla`

### 输入参数

| Name         | DType                        | Shape                                             | Optional | Format | Description |
|--------------|------------------------------|---------------------------------------------------|----------|--------|-------------|
| query        | Tensor[float16/bfloat16]     | (num_tokens, num_head, base_dim)                 | No       | ND | 不含 RoPE 部分的 Query 基础向量 |
| query_rope   | Tensor[float16/bfloat16]     | (num_tokens, num_head, rope_dim)                 | No       | ND | Query 的 RoPE 位置编码部分 |
| key          | Tensor[float16/bfloat16]     | (num_kv_tokens, num_kv_head, base_dim)           | No       | ND | 不含 RoPE 部分的 Key 基础向量 |
| key_rope     | Tensor[float16/bfloat16]     | (num_kv_tokens, num_kv_head, rope_dim)           | No       | ND | Key 的 RoPE 位置编码部分 |
| value        | Tensor[float16/bfloat16]     | (num_kv_tokens, num_kv_head, value_dim)          | No       | ND | Value 向量 |
| mask         | Tensor[float16/bfloat16]     | (batch, max_seq, max_seq) 或 (max_seq, max_seq)  | Yes      | ND | 因果 Mask，默认值：None |
| alibi_coeff  | Tensor[float32]              | 暂不支持                          | Yes      | ND    | ALiBi 系数，不支持；默认值：None |
| deq_scale_qk | Tensor[float32]              | 暂不支持                                       | Yes      | ND    | QK logits 反量化 scale，不支持；默认值：None |
| deq_offset_qk| Tensor[float32]              | 暂不支持                                       | Yes      | ND    | QK logits 反量化 offset，不支持；默认值：None |
| deq_scale_pv | Tensor[float32]              | 暂不支持                                       | Yes      | ND    | PV 反量化 scale，不支持；默认值：None |
| deq_offset_pv| Tensor[float32]              | 暂不支持                                       | Yes      | ND    | PV 反量化 offset，不支持；默认值：None |
| quant_p      | Tensor[float32]              | 暂不支持                                       | Yes      | ND    | 概率值量化scale，不支持；默认值：None |
| log_n        | Tensor[float32]              | 暂不支持                                       | Yes      | ND    | 归一化 log 因子，不支持；默认值：None |
| o_prev       | Tensor[float16/bfloat16]     | (num_tokens, num_head, value_dim)                | Yes      | ND | Ring 更新时使用的上一阶段注意力输出；默认值：None |
| lse_prev     | Tensor[float32]              | (num_head, num_tokens)                           | Yes      | ND    | Ring 更新时使用的上一阶段 log-sum-exp（LSE）；默认值：None |
| q_seq_lens   | Tensor[int32]                | (batch,)                                         | Yes      | ND    | 每个样本的 Query 序列长度；默认值：None |
| context_lens | Tensor[int32]                | (batch,)                                         | Yes      | ND    | 每个样本的 Key/Value 序列长度（上下文长度）；默认值：None |
| head_num     | int                           | -                                                | No       | -     | Query 的注意力头数；默认值：0（调用时通常需显式设置为 >0） |
| scale_value  | float                         | -                                                | No       | -     | 作用于 QKᵀ 的缩放系数，默认值：1.0 |
| kv_head_num  | int                           | -                                                | No       | -     | KV 头数，支持 KV 头分组；默认值：0 |
| mask_type    | int                           | -                                                | Yes      | -     | Mask 类型，`0` 表示不使用 Mask，`1` 表示三角因果 Mask；默认 0 |
| calc_type    | int                           | -                                                | Yes      | -     | 计算模式，`0` 表示启用 ring 更新（使用 `o_prev`/`lse_prev`），`1` 表示独立计算当前注意力；默认 0 |

> 注：当前版本中，`alibi_coeff`、deq/quant 相关张量、`log_n` 等参数主要为接口占位，未必在所有场景实际使用，可以统一传入 `None`。

### 输出参数

| Name          | DType                    | Shape                            | Description |
|---------------|--------------------------|----------------------------------|-------------|
| attention_out | Tensor[float16/bfloat16] | (num_tokens, num_head, value_dim)| 注意力输出，与 `value` 的 dtype 一致 |
| lse           | Tensor[float32]          | (num_head, num_tokens)           | 对应每个 head、每个 token 的 log-sum-exp |

## Python 使用示例

```python
import math
import numpy as np
from mindspore import Tensor
import ms_custom_ops

num_tokens = 4
num_kv_tokens = 8
num_head = 16
num_kv_head = 16
base_dim, rope_dim, value_dim = 128, 64, 128

scale_value = 1.0 / math.sqrt(base_dim + rope_dim)

q_nope = Tensor(np.random.randn(num_tokens, num_head, base_dim).astype(np.float16))
q_rope = Tensor(np.random.randn(num_tokens, num_head, rope_dim).astype(np.float16))
k_nope = Tensor(np.random.randn(num_kv_tokens, num_kv_head, base_dim).astype(np.float16))
k_rope = Tensor(np.random.randn(num_kv_tokens, num_kv_head, rope_dim).astype(np.float16))
value = Tensor(np.random.randn(num_kv_tokens, num_kv_head, value_dim).astype(np.float16))

mask = None
alibi = None
deq_scale_qk = None
deq_offset_qk = None
deq_scale_pv = None
deq_offset_pv = None
quant_p = None
log_n = None

o_prev = Tensor(np.zeros((num_tokens, num_head, value_dim), dtype=np.float16))
lse_prev = Tensor(np.zeros((num_head, num_tokens), dtype=np.float32))

q_seq_lens = Tensor(np.array([num_tokens], dtype=np.int32))
kv_seq_lens = Tensor(np.array([num_kv_tokens], dtype=np.int32))

attention_out, lse = ms_custom_ops.ring_mla(
    q_nope, q_rope, k_nope, k_rope, value,
    mask, alibi,
    deq_scale_qk, deq_offset_qk, deq_scale_pv, deq_offset_pv,
    quant_p, log_n,
    o_prev, lse_prev,
    q_seq_lens, kv_seq_lens,
    num_head, scale_value, num_kv_head,
    mask_type=0,
    calc_type=1
)

print(attention_out.shape, lse.shape)
# 期望输出: (num_tokens, num_head, value_dim) (num_head, num_tokens)
```
