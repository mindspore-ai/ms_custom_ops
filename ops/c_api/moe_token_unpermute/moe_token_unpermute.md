# moe_token_unpermute 算子

## 描述

根据sorted_indices存储的下标，获取permuted_tokens中存储的输入数据，permuted_tokens会与probs相乘；最后进行累加求和，并输出计算结果。

## 输入参数

| Name | DType | Shape | Optional | Inplace | Format | Description |
|------|-------|-------|----------|---------|--------|-------------|
| permuted_tokens | Tensor(float16) | [tokens_num * topK_num, hidden_size] | No | No | ND | 要进行反排列的已排列标记的Tensor |
| sorted_indices | Tensor(int32) | [tokens_num * topK_num] | No | No | ND | 表示需要计算的数据在permuted_tokens中的位置 |
| probs | Tensor(float16) | [tokens_num, topK_num] | No | No | ND | 与已排列标记对应的概率Tensor |
| padded_mode | bool |  | Yes | No |  | true表示开启paddedMode，false表示关闭paddedMode，paddedMode解释见restoreShape参数。<br>目前仅支持 False, 不对输出的shape进行变换。默认False |
| restore_shape | Tuple(int) |  | Yes | No |  | padded_mode=true时生效，否则不会对其进行操作。paddedMode=true时，out的shape将表征为restoreShape。目前仅支持None。默认None |

## 输出参数

| Name | DType | Shape | Description |
|------|-------|-------|-------------|
| out | Tensor(float16) | [tokens_num, hidden_size] | 加权反排列后的Tensor |

## 支持产品

- Atlas 推理系列产品

## 特殊说明

- 当前仅支持：`padded_mode = false， restore_shape = None`。
- topK 支持 1,2,4,8
- hidden_size 支持 2048，5120，7168

## 使用示例

### 基本使用示例（常规模式）

```python
import numpy as np
from mindspore import Tensor
import ms_custom_ops

token_num = 128
hidden_size = 7168
top_k = 8
permuted_token = np.random.randn(token_num * top_k, hidden_size).astype(np.float16)
sorted_idx = np.arange(token_num * top_k, dtype=np.int32)
np.random.shuffle(sorted_idx)
probs = np.random.randn(token_num, top_k).astype(np.float16)
out = ms_custom_ops.moe_token_unpermute(Tensor(permuted_token), Tensor(sorted_idx), Tensor(probs))
```
