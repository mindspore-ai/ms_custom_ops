# matmul 算子

## 描述

matmul 算子用于执行矩阵乘法操作。该算子支持输入矩阵的转置操作，并可指定输入和输出的数据格式。

## 输入参数

| Name           | DType              | Shape     | Optional | Inplace | Format        | Description                                          |
|----------------|--------------------|-----------|----------|---------|---------------|------------------------------------------------------|
| x1              | Tensor(float16/bf16) | 2维及以上 | No       | No      | ND/FRACTAL_NZ | 第一个输入矩阵                                        |
| x2              | Tensor(float16/bf16) | 2维及以上 | No       | No      | ND/FRACTAL_NZ | 第二个输入矩阵                                        |
| transpose_a     | bool               | -         | Yes      | -       | -             | 是否对 x1（A 矩阵）进行转置，默认为 False              |
| transpose_b     | bool               | -         | Yes      | -       | -             | 是否对 x2（B 矩阵）进行转置，默认为 False              |
| x1_format       | int                | -         | Yes      | -       | -             | x1 的数据格式：0 表示 ND，1 表示 FRACTAL_NZ，默认为 0 |
| x2_format       | int                | -         | Yes      | -       | -             | x2 的数据格式：0 表示 ND，1 表示 FRACTAL_NZ，默认为 0 |
| output_format   | int                | -         | Yes      | -       | -             | 输出的数据格式：0 表示 ND，1 表示 FRACTAL_NZ，默认为 0 |

## 输出参数

| Name | DType              | Shape                  | Description      |
|------|--------------------|------------------------|------------------|
| y    | Tensor(float16/bf16) | 符合矩阵乘法规则的形状 | 矩阵乘法的计算结果 |

## 支持产品

- Atlas 800I A2 推理产品、Atlas 推理系列产品

## 约束说明

1. Atlas 推理系列产品仅支持 float16 数据类型
2. Atlas 800I A2 推理产品仅支持 ND 格式
3. Atlas 推理系列产品支持 ND 和 FRACTAL_NZ 数据格式，推荐组合如下：

   | x1         | x2         | y          | 推荐场景                                         |
   |------------|------------|------------|-------------------------------------------------|
   | FRACTAL_NZ | FRACTAL_NZ | ND         | prefill 阶段，x1 第 0 维度较大的情况               |
   | ND         | FRACTAL_NZ | ND         | decode 阶段，x1 第 0 维度较小的情况               |
   | FRACTAL_NZ | FRACTAL_NZ | FRACTAL_NZ | 通用情况                                         |

   当输入为 FRACTAL_NZ 格式时，需要对该输入先调用 `trans_data` 算子进行格式转换；当输出为 FRACTAL_NZ 格式时，需要对 MatMul 的输出调用 `trans_data` 算子进行格式转换。

   **注意**：Pynative Mode 下仅支持 x1 和 y 为 ND 格式的情况。

## 使用示例

### 基本使用示例

```python
import mindspore as ms
import numpy as np
import ms_custom_ops

ms.set_context(device_target="Ascend")

@ms.jit
def matmul_func(x1, x2, transpose_a=False, transpose_b=False,
                x1_format=0, x2_format=0, output_format=0):
    return ms_custom_ops.mat_mul(x1, x2, transpose_a=transpose_a, transpose_b=transpose_b,
                                 x1_format=x1_format, x2_format=x2_format, output_format=output_format)

# 示例1：基础矩阵乘法
batch = 2
m = 128
k = 256
n = 128
x1 = np.random.randn(batch, m, k).astype(np.float16)
x2 = np.random.randn(batch, k, n).astype(np.float16)

ms_x1 = ms.Tensor(x1)
ms_x2 = ms.Tensor(x2)
output = matmul_func(ms_x1, ms_x2)
print("Output shape:", output.shape)
```

### 使用 FRACTAL_NZ 格式的示例

```python
import mindspore as ms
import numpy as np
import ms_custom_ops

ms.set_context(device_target="Ascend")

@ms.jit
def matmul_with_nz_func(x1, x2):
    # 将 ND 格式转换为 FRACTAL_NZ 格式
    x1_nz = ms_custom_ops.trans_data(x1, transdata_type=1)  # ND_TO_FRACTAL_NZ
    x2_nz = ms_custom_ops.trans_data(x2, transdata_type=1)  # ND_TO_FRACTAL_NZ

    # 执行矩阵乘法，指定输入和输出格式为 FRACTAL_NZ
    out_nz = ms_custom_ops.mat_mul(x1_nz, x2_nz, transpose_a=False, transpose_b=False,
                                    x1_format=1, x2_format=1, output_format=1)

    # 将输出从 FRACTAL_NZ 格式转换回 ND 格式
    out = ms_custom_ops.trans_data(out_nz, transdata_type=0)  # FRACTAL_NZ_TO_ND
    return out

# x1 形状: (batch, m, k)
# x2 形状: (batch, k, n)
# 输出形状: (batch, m, n)
batch = 2
m = 128
k = 256
n = 128
x1 = np.random.randn(batch, m, k).astype(np.float16)
x2 = np.random.randn(batch, k, n).astype(np.float16)

ms_x1 = ms.Tensor(x1)
ms_x2 = ms.Tensor(x2)
output = matmul_with_nz_func(ms_x1, ms_x2)
print("Output shape:", output.shape)
```
