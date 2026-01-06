
# apply_rotary_pos_emb 算子

## 描述

旋转位置编码（Rotary Position Embedding，RoPE），以旋转矩阵的方式在q、k中注入位置信息，使得attention计算时能感受到token的位置关系，在各大模型中，RoPE被广泛应用。RoPE以绝对位置编码的方式实现了相对位置编码，能有效保持位置信息相对关系，并且可以通过编码外推的方式支持超过训练长度的位置编码。支持query/key输入为2/3/4维，其中2维仅支持TH unpad方案。

## 输入参数

| Name                | DType           | Shape                                  | Optional | Inplace | Format | Description                                            |
|---------------------|-----------------|----------------------------------------|----------|---------|--------|--------------------------------------------------------|
| query               | Tensor(float16/bf16)  |[batch, seqlen, hidden_size_q]/ [ntokens, hidden_size_q]/ [batch, seqlen, head_num_q, head_size]| No       | No      | ND     |当前step多个token的query。支持2维TH, 3维BSH, 4维BSND 。 |
| key                 | Tensor(float16/bf16)  |[batch, seqlen, hidden_size_k]/ [ntokens, hidden_size_k]/ [batch, seqlen, head_num_K, head_size]| No       | No      | ND     | 当前step多个token的key。。支持2维TH, 3维BSH, 4维BSND 。   |  
| cos                 | Tensor(float16/float/bf16) | [ntokens, head_size]/ [max_seqlen, head_size] | No       | No      | ND     | ROPE高精度模式，需要输入cos的数据类型为float时生效。 |
| sin                 | Tensor(float16/float/bf16) | [ntokens, head_size]/ [max_seqlen, head_size] | No       | No      | ND     | ROPE高精度模式，需要输入sin的数据类型为float时生效。 |
| position_ids              | Tensor(uint32) | [batch]  | No      | No      | Nd | 在推理prefill阶段表示每个batch的sequence length，在推理decode阶段表示每个batch递推的index。  |
| cos_format         | int          |             | No      | No      |   |可取值0,1,2,3。推荐使用2或3，当取值为0或1时sin/cos的shape为[max_seqlen, head_size]，当取值为2或3时sin/cos的shape为[ntokens, head_size]，当取值为0或2时表示half模式，当取值为1或3时为interleave模式    |

## 输出参数

| Name   | DType      | Shape      | Description           |
|--------|------------|------------|-----------------------|
| out_query| Tensor(float16/bf16)  | [batch, seqlen, hidden_size_q]/ [ntokens, hidden_size_q]/ [batch, seqlen, head_num_q, head_size]| query旋转位置编码后的结果 |
| out_key| Tensor(float16/bf16) | [batch, seqlen, hidden_size_k]/ [ntokens, hidden_size_k]/ [batch, seqlen, head_num_K, head_size]| key旋转位置编码后的结果 |

## 约束说明

- 支持产品：Atlas 800I A2 推理产品
- 输入tensor数据类型需保持一致，高精度模式例外。
- cos、sin传入数据类型为float时，中间计算结果以float保存。
- hidden_size_q和hidden_size_k必须是head_size的整数倍，满足`hidden_size_q = head_size * head_num_q、 hidden_size_k = head_size * head_num_k`，其中head_num_q可以大于head_num_k,hidden_size_q和hidden_size_k需要32bytes对齐。
- ntokens = sum(seqlen[i])，i=0...batch-1。
- query和key支持2/3/4维，Format均为ND,[ntokens, hidden_size]/[batch, seqlen, hidden_size]/[batch, seqlen, head_num, head_size]；当query和key为2维时，仅支持TH unpad方案
- Decoder阶段要取cos和sin表中seqlen对应的cos/sin值输入。
- 多batch场景需要组合使用gather算子。

## 使用示例

```python
import numpy as np
import mindspore as ms
import ms_custom_ops

inv_freq = 1.0 / (10000 ** (np.arange(0, 128, 2).astype(np.float32) * (1 / 128)))
t = np.arange(2048, dtype=inv_freq.dtype)
freqs = np.outer(t, inv_freq)
emb = np.concatenate((freqs, freqs), axis=-1)
cos = np.cos(emb).astype(np.float16)
sin = np.sin(emb).astype(np.float16)
query = np.random.rand(2, 1, 128).astype(np.float16)
key = np.random.rand(2, 1, 128).astype(np.float16)
position_ids = np.random.randint(0, 2048, [2], dtype=np.int32)
cos = cos[position_ids]
sin = sin[position_ids]
query_tensor = ms.Tensor(query, dtype=ms.float16)
key_tensor = ms.Tensor(key, dtype=ms.float16)
cos_tensor = ms.Tensor(cos, dtype=ms.float16)
sin_tensor = ms.Tensor(sin, dtype=ms.float16)
pos_tensor = ms.Tensor(position_ids, dtype=ms.float16)
out_query, out_key = ms_custom_ops.apply_rotary_pos_emb(query_tensor, key_tensor, cos_tensor, sin_tensor, pos_tensor, 2)
print("query out: ", out_query)
print("key out: ", out_key)
```
