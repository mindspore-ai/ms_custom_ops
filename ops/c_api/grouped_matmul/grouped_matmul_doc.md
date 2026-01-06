# grouped_matmul 算子

## 描述

`grouped_matmul`（分组矩阵乘法）算子针对输入张量 `x` 与权重张量 `weight` 按照分组信息 `group_list` 逐组执行矩阵乘法操作，可选地支持bias（偏置）、scale（缩放因子）、per_token_scale（token级缩放）和antiquant_scale（反量化缩放）等参数。每组输入和权重做独立矩阵乘法，并拼接形成整体输出。该算子适用于高效实现分组全连接、Mixture-of-Experts (MoE) 等需要基于动态分组的场景，并针对Ascend芯片做性能优化。

### 计算公式

- **非量化场景（FP16）**：

  ```python
  y = x * weight[i] + bias[i]
  ```

- **量化场景（per-channel scale）**：

  ```python
  y = (x * weight[i] + bias[i]) * scale[i]
  ```

- **量化场景（per-token scale）**：

  ```python
  y = (x * weight[i] + bias[i]) * scale[i] * per_token_scale
  ```

其中 `i` 表示第`i`组，`x` 为该组输入片段，`weight[i]`、`bias[i]`、`scale[i]` 为每组权重、偏置、缩放因子，`per_token_scale` 通常 shape 为 [M] 或 [M, 1]。

---

## 输入参数

| Name            | DType                                      | Shape                    | Optional                | Inplace | Format      | Description                                                                                                  |
|-----------------|--------------------------------------------|--------------------------|-------------------------|--------|-------------|--------------------------------------------------------------------------------------------------------------|
| x               | Tensor(float16/int8)                       | [M, K] 或 [K, M]         | 否                      | 否     | ND/FRACTAL_NZ          | 输入特征矩阵，支持 float16、int8。                                                                           |
| weight          | Tensor(float16/int8)                       | [E, K, N] 或 [E, N, K]*  | 否                      | 否     | FRACTAL_NZ  | 分组权重张量，支持 float16、int8，E为分组数，必须为NZ格式。                                                   |
| group_list      | Tensor(int32)                              | [E]                      | 否                      | 否     | ND          | 分组边界列表，升序，指定每组在x中的范围，必填。                                                              |
| bias            | Tensor(float16/int32)                      | [E, N]                   | 是，默认None           | 否     | ND          | 每组偏置项，可选，支持 float16 或 int32。                                                                    |
| scale           | Tensor(float32/int64/uint64)               | [E, N]                   | 是，默认None           | 否     | ND          | 每组dequant缩放因子，量化推理时必需，支持 float32、int64 或 uint64。                                          |
| per_token_scale | Tensor(float32)                            | [M] 或 [M, 1]            | 是，默认None           | 否     | ND          | 每token缩放系数（动态量化/补偿时可选），仅支持 float32。                                                      |
| antiquant_scale | Tensor(float16/float32)                    | [E, K, 1]                | 是，默认None           | 否     | ND          | 权重量化反量化缩放因子，int8量化推理配合使用。                                                                |
| transpose_a     | bool                                       | 标量                     | 是，默认False          | 否     | -           | x是否转置，默认为False。                                                                                     |
| transpose_b     | bool                                       | 标量                     | 是，默认False          | 否     | -           | weight是否转置，默认为False。                                                                                |
| x_format        | int                                        | 标量                     | 是，默认0              | 否     | -           | 输入x的数据格式：0表示ND，1表示FRACTAL_NZ。大shape场景下支持将x转换为FRACTAL_NZ格式时高性能推理。             |

> - x 的 shape[0] 必须等于 group_list[-1]（即最终分组后总token数量一致）。
> - K轴：在fp16输入场景下必须为16的整数倍，在int8输入场景下必须为32的整数倍。
> - N轴：必须为16的整数倍。
> - E轴（分组数）：必须为8的整数倍.
> - weight 应转换为 FRACTAL_NZ 格式，推荐用 `ms_custom_ops.trans_data(weight, transdata_type=1)`。
> - bias、scale、antiquant_scale、per_token_scale等参数如不需要可不提供，不同量化方案可据实际场景选用。
> - 权重量化场景下，需设置transpose_b=True，此时权重weight的shape支持[E, N, K]。
> - 支持PyNative/Graph模式下使用。
> - 对于 int4 量化场景，请使用 `grouped_matmul_w4` 算子。

---

## 输出参数

| Name   | DType                      | Shape     | Format | Description                       |
|--------|----------------------------|-----------|--------|----------------------------|
| out    | float16 | [M, N]    | ND     | 输出拼接后的分组MatMul结果 |

---

## 支持平台

- 仅支持昇腾Atlas推理系列芯片。

---

## 常见问题

1. group_list设置不当会导致shape mismatch错误，请确保 `sum(group样本数) == M`。
2. 权重格式须为NZ，否则算子将报错。
3. 若使用int8量化推理，请正确提供scale, antiquant_scale相关参数与数据类型。

---

## 适用场景

- MoE（Mixture-of-Experts）动态专家分组全连接
- 稀疏或分组全连接、主干-分支等模型高效推理
- 多路分组稀疏矩阵乘法（多分支/专家并行）

---

## 使用示例

```python
import mindspore as ms
from mindspore import Tensor
import numpy as np
import ms_custom_ops

# 假设分4组: 总M=8, K=16, N=32, E=4
M, K, N, E = 8, 16, 32, 4
x = Tensor(np.random.randn(M, K).astype(np.float16))
weight = Tensor(np.random.randn(E, K, N).astype(np.float16))
group_list = Tensor(np.array([2, 4, 6, 8]).astype(np.int32))  # 每组样本数量累加
bias = Tensor(np.random.randn(E, N).astype(np.float16))
scale = Tensor(np.ones((E, N)).astype(np.float32))

out = ms_custom_ops.grouped_matmul(
    x, weight, group_list, bias, scale
)
print(out.shape)  # (8, 32)
```
