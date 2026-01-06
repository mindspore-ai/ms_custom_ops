# grouped_matmul_w4 算子

## 描述

`grouped_matmul_w4`（分组矩阵乘法，int4量化版本）算子针对输入张量 `x` 与权重张量 `weight`（int4量化）按照分组信息 `group_list` 逐组执行矩阵乘法操作，支持bias（偏置）、x_scale（输入缩放因子）和weight_scale（权重量化缩放因子）等参数。每组输入和权重做独立矩阵乘法，并拼接形成整体输出。该算子适用于高效实现分组全连接、Mixture-of-Experts (MoE) 等需要基于动态分组的场景，并针对Ascend芯片做性能优化。

### 计算公式

- **int4量化场景**：

  ```python
  y = (x * weight[i] + bias[i]) * x_scale * weight_scale[i]
  ```

其中 `i` 表示第`i`组，`x` 为该组输入片段，`weight[i]`、`bias[i]`、`weight_scale[i]` 为每组权重、偏置、权重量化缩放因子，`x_scale` 为输入缩放因子。

---

## 输入参数

| Name         | DType                    | Shape                    | Optional                | Format      | Description                                                                                                  |
|--------------|--------------------------|--------------------------|-------------------------|-------------|--------------------------------------------------------------------------------------------------------------|
| x            | Tensor(int8)             | [M, K]                   | No                      | ND          | 输入特征矩阵，支持 int8。                                                                                    |
| weight       | Tensor(qint4x2)          | [E, N, K/2]              | No                      | FRACTAL_NZ  | 分组权重张量，支持 qint4x2（int4量化），E为分组数，必须为NZ格式。                                            |
| group_list   | Tensor(int32)            | [E]                      | No                      | ND          | 分组边界列表，升序，指定每组在x中的范围，必填。                                                              |
| bias         | Tensor(float32)          | [E, N]                   | No                      | ND          | 每组偏置项，仅支持 float32。                                                                                 |
| x_scale      | Tensor(float32)          | [M] 或 [M, 1]            | No                      | ND          | 输入缩放因子，仅支持 float32。                                                                               |
| weight_scale | Tensor(float32)          | [E, K//g, N] 或类似      | No                      | ND          | 权重量化缩放因子，int4量化推理配合使用，仅支持 float32。                                                      |

> - x 的 shape[0] 必须等于 group_list[-1]（即最终分组后总token数量一致）。
> - M轴：支持动态shape。
> - K、N轴：仅支持以下组合：[256, 7168]、[512, 7168]、[7168, 512]、[7168, 1024]。
> - E轴（分组数）：仅支持 256。
> - g值（权重量化分组大小）：仅支持 256。
> - weight 为 int4 量化格式（qint4x2），存储为 [E, N, K/2] 形状，其中 K/2 是因为两个4bit值打包到一个8bit中。
> - weight 必须转换为FRACTAL_NZ格式，推荐用 `ms_custom_ops.trans_data(weight, transdata_type=1)`。
> - bias、x_scale、weight_scale 为必需参数，必须提供。
> - 支持PyNative/Graph模式下使用。
> - 对于 float16 和 int8 量化场景，请使用 `grouped_matmul` 算子。

---

## 输出参数

| 名称   | DType    | Shape  | Format | 说明                       |
|--------|----------|--------|--------|----------------------------|
| out    | float16  | [M, N] | ND     | 输出拼接后的分组MatMul结果 |

---

## 支持平台

- 仅支持昇腾Atlas推理系列芯片。

---

## 常见问题

1. group_list设置不当会导致shape mismatch错误，请确保 `sum(group样本数) == M`。
2. 权重格式须为NZ，否则算子将报错。
3. 若使用int4量化推理，请正确提供bias、x_scale、weight_scale相关参数与数据类型。

---

## 适用场景

- MoE（Mixture-of-Experts）动态专家分组全连接（int4量化）
- 稀疏或分组全连接、主干-分支等模型高效推理（int4量化）
- 多路分组稀疏矩阵乘法（多分支/专家并行，int4量化）

---

## 使用示例

```python
import mindspore as ms
from mindspore import Tensor
import numpy as np
import ms_custom_ops

# 假设分4组: 总M=8, K=16, N=32, E=4
M, K, N, E = 8, 16, 32, 4
x = Tensor(np.random.randint(-128, 127, (M, K)).astype(np.int8))
# weight 为 int4 量化格式，需要转换为 qint4x2
weight = Tensor(np.random.randint(0, 255, (E, N, K // 2)).astype(np.uint8))
weight = ms_custom_ops.type_cast(weight, ms.qint4x2)
group_list = Tensor(np.array([2, 4, 6, 8]).astype(np.int32))
bias = Tensor(np.random.randn(E, N).astype(np.float16))
x_scale = Tensor(np.random.randn(M).astype(np.float32))
weight_scale = Tensor(np.random.randn(E, K // 256, N).astype(np.float16))

# weight 必须转换为 NZ 格式
w_i8 = ms_custom_ops.type_cast(weight, ms.int8)
w_i8_nz = ms_custom_ops.trans_data(w_i8, transdata_type=1)
weight_nz = ms_custom_ops.type_cast(w_i8_nz, ms.qint4x2)

out = ms_custom_ops.grouped_matmul_w4(
    x, weight_nz, group_list, bias, x_scale, weight_scale
)
print(out.shape)  # (8, 32)
```
