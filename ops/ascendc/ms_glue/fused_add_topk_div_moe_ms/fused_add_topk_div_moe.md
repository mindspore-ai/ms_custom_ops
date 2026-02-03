# fused_add_topk_div_moe算子

## 描述

提供一个类型为tensor的专家分数logits以及一个对应的类型为tensor偏置权重bias，然后对logits进行softmax处理，再加上偏置权重bias，选择出8个专家数量，最后对8个专家分数进行归一化处理，得到最终输出的专家数量与归一化后的分数。

## 输入参数

| Name                | DType           | Shape                                  | Optional | Inplace | Format | Description                                            |
|---------------------|-----------------|----------------------------------------|----------|---------|--------|--------------------------------------------------------|
| logits              | Tensor(float32) | [tokens, 128]                          | No       | No      | ND     | 每个token中128个专家的得分 |
| bias                | Tensor(float32) | [128]                                  | No       | No      | ND     | 对每个专家得分的偏置加权 |
| num_groups          | int             | -                                      | Yes      | No      | -      | 分组数量，当前仅支持1 |
| group_topk          | int             | -                                      | Yes      | No      | -      | 选择组数，当前仅支持1 |
| topk_div_group_topk | int             | -                                      | Yes      | No      | -      | 组内选择topk_div_group_topk个最大值求和，当前仅支持8 |
| topk                | int             | -                                      | Yes      | No      | -      | topk选择前k个值，当前仅支持8 |
| activate_type       | int             | -                                      | Yes      | No      | -      | 激活类型，当前仅支持0 |
| is_norm             | bool            | -                                      | Yes      | No      | -      | 是否启动归一化，当前仅支持True |
| scale               | float           | -                                      | Yes      | No      | -      | 归一化后的乘系数，当前仅支持1.0 |

## 输出参数

| Name | DType | Shape | Description |
|------|-------|-------|-------------|
| expert_weight | Tensor(float32) | [tokens, 8] | 归一化后专家的最终分数，按专家分数降序排列 |
| expert_index | Tensor(int32) | [tokens, 8] | 选出的8个专家的下标，按专家分数降序排列 |

## 支持产品

- Atlas 推理系列产品

## 特殊说明

1. 当前仅支持(n, 128)格式的输入，同时n<=2048
2. bias的维度只能是(128)
3. topk只支持8
4. num_expert只支持128
5. num_groups只支持1, group_topk只支持1, topk_div_group_topk只支持8, topk只支持8, activate_type只支持0, is_norm只支持True, scale只支持1.0

## 使用示例

### 基本使用示例

```python

import mindspore as ms
import numpy as np
import ms_custom_ops
from mindspore import context, Tensor

ms.set_context(device_target="Ascend", mode=context.GRAPH_MODE)
ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

num_tokens = 2048
expert_num = 128
num_groups = 1
group_topk = 1
topk_div_group_topk = 8
topk = 8
activate_type = 0
is_norm = True
scale = 1.0

np_logits = np.random.random((num_tokens, 128)).astype(np.float32)
np_bias = np.random.random(128).astype(np.float32)
logits = Tensor(np_logits)
bias = Tensor(np_bias)
expert_weight, expert_index = ms_custom_ops.fused_add_topk_div_moe(logits, bias, num_groups, group_topk,
                                                                     topk_div_group_topk, topk, activate_type,
                                                                     is_norm, scale)
