# moe_gating_group_topk算子

## 描述

moe_gating_group_topk算子实现了MoE专家分组计算，对输入x做Sigmoid计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。主要应用于Pangu模型。

## 输入参数

| Name | DType | Shape | Optional | Inplace | Format | Description |
|------|-------|-------|----------|---------|--------|-------------|
| x | Tensor(float16/float32/bf16) | [n,expert] | No | No | ND | 专家组分数 |
| bias | Tensor(float16/float32/bf16) | [expert] | No | No | ND | 用于与x相加，数据类型和格式与x一致,bias当前仅支持None |
| k | int |  | No | No |  | topk的k值。要求1 <= k <= x_shape[-1] / groupCount * kGroup。 |
| k_group| int |  | No | No |  | 分组排序后取的group个数。要求1 <= kGroup <= groupCount，并且kGroup * x_shape[-1] / groupCount的值要大于等于k。 |
| group_count | int |  | No | No |  | 分组的总个数。要求groupCount > 0，x_shape[-1]能够被groupCount整除且整除后的结果大于2，并且整除的结果按照32个数对齐后乘groupCount的结果不大于2048。 |
| group_select_mode | int |  | No | No |  | 0表示使用最大值对group进行排序, 1表示使用topk2的sum值对group进行排序。仅支持0|
| renorm | int |  | No | No |  | renorm标记，当前仅支持0，表示先进行norm操作，再计算topk。 |
| norm_type | int |  | No | No |  | 0表示使用Softmax函数，1表示使用Sigmoid函数。仅支持0。 |
| out_flag | bool |  | No | No |  | true表示输出，false表示不输出。仅支持false。|
| routed_scaling_factor | float | | No| No |  | 用于计算yOut使用的routedScalingFactor系数 |
| eps | float |  | No| No |  | 用于计算yOut使用的eps系数 |

注意：

- enableExpertMapping参数控制是否启用逻辑专家模式。当enableExpertMapping为false时，输入只有x和add_num；当为true时，输入包括x、add_num、mapping_num和mapping_table。

- a表示batch大小，b表示专家数量，c表示最大冗余专家数（最多128）。

## 输出参数

| Name | DType | Shape | Description |
|------|-------|-------|-------------|
| y_out | Tensor(float16/float32/bf16) | [n, topk] | 分组排序topk后计算的结果 |
| expert_idx_out | Tensor(int32) | [n, k] | 专家的序号 |
| norm_out | Tensor(float32) | [n,expert] | norm计算的输出结果, 当前无输出 |

## 特殊说明

- 当前仅支持：`group_select_mode = 0， renorm = 0， norm_type = 0， out_flag = false`。
- expert能够被group_count整除，expert不超过2048，当前仅支持`1 < group_count < 32，1 < expert/group_count < 64，expert/group_count能被8整除，group_count = k_group = k`。

## 使用示例

### 基本使用示例（常规模式）

```python
import numpy as np
import mindspore as ms
import ms_custom_ops

x = np.random.uniform(-2, 2, (8, 64)).astype(np.float16)
x_tensor = ms.Tensor(x, dtype=ms.float16)
bias = None
k = 4
k_group = 4
group_count = 4
group_select_mode = 0
renorm = 0
norm_type = 0
out_flag = False
routed_scaling_factor = 1.0
eps = 1e-20
y_out, expert_idx_out, _ = ms_custom_ops.moe_gating_group_topk(x_tensor, bias, k, k_group, group_count, group_select_mode, renorm, norm_type, out_flag, routed_scaling_factor, eps)
print("y_out:", y_out)
print("expert_idx_out:", expert_idx_out)
```
