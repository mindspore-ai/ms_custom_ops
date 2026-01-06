
# apply_rotary_pos_emb_atb 算子

## 描述

此算子是对接ATB算子库的Rope算子，旋转位置编码（Rotary Position Embedding，RoPE），以旋转矩阵的方式在q、k中注入位置信息，使得attention计算时能感受到token的位置关系，在各大模型中，RoPE被广泛应用。RoPE以绝对位置编码的方式实现了相对位置编码，能有效保持位置信息相对关系，并且可以通过编码外推的方式支持超过训练长度的位置编码。

## 计算公式

参见官网描述:https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/API/ascendtbapi/ascendtb_01_0075.html

## 输入参数

| Name                | DType           | Shape                                  | Optional | Inplace | Format | Description                                            |
|---------------------|-----------------|----------------------------------------|----------|---------|--------|--------------------------------------------------------|
| query               | Tensor(float16/bf16)  | [ntokens, hiddenSizeQ] | No       | No      | ND     | 当前step多个token的query。  |
| key                 | Tensor(float16/bf16)  |[ntokens, hiddenSizeK]| No       | No      | ND     | 当前step多个token的key。   |  
| cos                 | Tensor(float16/float/bf16) | [ntokens, head_size] / [ntokens, head_size / 2] | No       | No      | ND     | 当cos的第二个维度与参数rotaryCoeff不相等时，其值为head_size。ROPE高精度模式，需要输入cos的数据类型为float时生效。cos的第二个维度需要是参数rotaryCoeff的整数倍。 |
| sin                 | Tensor(float16/float/bf16) | [ntokens, head_size] / [ntokens, head_size / 2] | No       | No      | ND     | 当sin的第二个维度与参数rotaryCoeff不相等时，其值为head_size。ROPE高精度模式，需要输入sin的数据类型为float时生效。sin的第二个维度需要是参数rotaryCoeff的整数倍。 |
| seq_len              | Tensor(uint32/int32) | [batch]  | No      | No      | Nd |                    |
| rotary_coeff         | int          |           | No      | No      |   | 旋转系数，对半旋转是2，支持配置2、4、head_size/2 |
| cos_format         | int          |             | Yes | No      |   | 训练用参数，支持配置0或1，推理采用默认值0  |

## 输出参数

| Name   | DType      | Shape      | Description           |
|--------|------------|------------|-----------------------|
| out_query| Tensor(float16/bf16)  | [ntokens, hiddenSizeQ]| query旋转位置编码后的结果 |
| out_key| Tensor(float16/bf16) | [ntokens, hiddenSizeK]| key旋转位置编码后的结果 |

## 约束说明

- 输入tensor数据类型需保持一致，高精度模式例外。
- cos、sin传入数据类型为float时，中间计算结果以float保存。
- hiddenSizeQ和hiddenSizeK必须是head_size的整数倍，满足`hiddenSizeQ = head_size * headNumQ、hiddenSizeK = head_size * headNumK`，其中headNumQ可以大于headNumK,hiddenSizeQ和hiddenSizeK需要32bytes对齐。
- ntokens = sum(seqlen[i])，i=0...batch-1。
- query和key要求两维，有部分模型使用了4维，这种情况下维度是：
  [batch, seqlen, headNum, head_size]；对应的ropeQ、ropeK也是四维，维度输入输出对应。
- Decoder阶段要取cos和sin表中seqlen对应的cos/sin值输入。
- 多batch场景需要组合使用gather算子。

## 另见

|算子|特点|
|----|----|
|[apply_rotary_pos_emb_cops](https://gitee.com/mindspore/ms_custom_ops/blob/master/ops/c_api/apply_rotary_pos_emb_cops/apply_rotary_pos_emb_cops.md)|输入参数query/key/cos/sin要求四维, 当前仅支持head_size=128和half模式|
|[apply_rotary_pos_emb_ms](https://gitee.com/mindspore/ms_custom_ops/blob/master/ops/ascendc/apply_rotary_pos_emb_ms/apply_rotary_pos_emb_ms.md) |输入参数query/key要求三维, cos/sin要求两维, 支持query_head_size >= cos_head_size，当前仅支持Atlas推理系列产品下的interleave模式|

## 使用示例

```python
import numpy as np
import mindspore as ms
import ms_custom_ops

ms.set_device("Ascend")
batch_size = 1
seq_len = 32
q_num_head = 4
k_num_head = 2
head_dim = 128
query_data = np.random.uniform(0, 1, [batch_size * seq_len, head_dim * k_num_head]).astype(query_dtype)
key_data = np.random.uniform(0, 1, [batch_size * seq_len, head_dim * k_num_head]).astype(query_dtype)
cos_data = np.random.uniform(0, 1, [batch_size * seq_len, head_dim]).astype(query_dtype)
sin_data = np.random.uniform(0, 1, [batch_size * seq_len, head_dim]).astype(query_dtype)
rotary_coeff = 2
query = ms.Tensor(query_data, dtype=ms.bfloat16)
key = ms.Tensor(key_data, dtype=ms.bfloat16)
cos = ms.Tensor(cos_data, dtype=ms.bfloat16)
sin = ms.Tensor(sin_data, dtype=ms.bfloat16)
seqlen = ms.Tensor(batch_size * seq_len, dtype=ms.int32)

query_emb, key_emb = ms_custom_ops.apply_rotary_pos_emb_atb(query, key, cos, sin, seqlen, rotary_coeff)
```