# batch_matmul 算子

## 描述

batch_matmul 算子用于执行批量矩阵乘法操作。该算子支持3维的Tensor输入，第一维是batch维度，最后两个维度做矩阵乘法。也支持其中一个输入的batch轴为1时做broadcast。

## 输入参数

| Name           | DType              | Shape     | Optional | Inplace | Format | Description                                          |
|----------------|--------------------|-----------|----------|---------|--------|------------------------------------------------------|
| x1              | Tensor(float16/bfloat16/float32) | 3维 | No       | No      | ND     | 第一个输入矩阵，shape为[A, M, K]。数据类型支持说明见下方 |
| x2              | Tensor(float16/bfloat16/float32) | 3维 | No       | No      | ND     | 第二个输入矩阵，shape为[B, K, N]。 |
| cube_math_type  | int                | -         | Yes      | -       | -      | Cube单元的计算逻辑，控制矩阵乘法的精度模式，默认为0。详细说明见下方"cube_math_type 参数说明" |

## cube_math_type 参数说明

`cube_math_type` 参数用于控制 Ascend 芯片上 Cube 单元（矩阵乘法加速单元）的计算精度模式。不同取值对应不同的精度策略：

- **0 (KEEP_DTYPE)**：保持输入数据类型，不进行精度转换。这是默认值，推荐在大多数场景下使用。算子会按照输入张量的原始数据类型进行计算，不会自动进行精度转换。

- **1 (ALLOW_FP32_DOWN_PRECISION)**：允许将 FP32 降精度到 FP16 进行计算，以提高性能。当输入为 FP32 类型时，Cube 单元会将其转换为 FP16 进行计算，从而获得更好的性能，但可能会损失一定的精度。适用于对精度要求不高的场景，或者需要平衡性能和精度的场景。

- **2 (USE_FP16 / FORCE_FP16)**：强制使用 FP16 精度进行计算，无论输入数据类型如何。即使输入是其他数据类型，也会强制转换为 FP16 进行计算。适用于需要统一使用 FP16 的场景，或者需要最大化性能的场景。

- **3 (USE_HF32 / FORCE_HF32)**：强制使用 HF32（Half Float 32）精度进行计算。HF32 是 Ascend 芯片特有的高精度浮点格式，提供比 FP16 更高的精度（接近 FP32），同时保持比 FP32 更好的性能。适用于对精度要求较高，但又需要一定性能的场景。

**使用建议**：

- 默认情况下使用 `0 (KEEP_DTYPE)`，这样可以保持计算的准确性。
- 如果对性能要求较高且可以接受一定的精度损失，可以使用 `1 (ALLOW_FP32_DOWN_PRECISION)` 或 `2 (USE_FP16)`。
- 如果对精度要求较高，可以使用 `3 (USE_HF32)`，它在精度和性能之间提供了良好的平衡。

## 输出参数

| Name | DType              | Shape                  | Description      |
|------|--------------------|------------------------|------------------|
| y    | Tensor(与输入类型相同) | 符合批量矩阵乘法规则的形状 | 批量矩阵乘法的计算结果，输出类型与输入类型相同 |

## 计算公式

out = x1 @ x2

其中：

- x1的shape是[A, M, K]，x2的shape是[A, K, N]，输出out的shape是[A, M, N]
- 第一维相等，后两维做矩阵乘运算
- 如果x1的shape是[A, M, K]，x2的shape是[1, K, N]，输出out的shape是[A, M, N]（B矩阵第一维为1，会broadcast到A）
- 如果x1的shape是[1, M, K]，x2的shape是[B, K, N]，输出out的shape是[B, M, N]（A矩阵第一维为1，会broadcast到B）

## 支持产品

- Atlas A3 训练系列产品、Atlas A3 推理系列产品
- Atlas A2 训练系列产品、Atlas A2 推理系列产品
- Atlas 训练系列产品
- Atlas 推理系列产品

## 约束说明

1. 仅支持3维的Tensor传入，第一维是batch维度，最后两个维度做矩阵乘法
2. 支持其中一个输入的batch轴为1时做broadcast
3. **数据类型限制**：aclnnBatchMatMul 支持的数据类型取决于芯片型号：
    - **Atlas 训练系列产品、Atlas 推理系列产品**：数据类型支持 FLOAT16、FLOAT32
    - **Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品**：数据类型支持 BFLOAT16、FLOAT16、FLOAT32

   如果需要在不支持的芯片上使用其他数据类型，请使用 MindSpore 内置的 BatchMatMul 算子。详细说明见下方"数据类型支持说明"
4. mat2的Reduce维度需要与self的Reduce维度大小相等
5. **注意**：aclnnBatchMatMul 不支持 transpose 参数。如果需要在调用前转置输入，请先使用 transpose 算子对输入进行转置，然后再调用 batch_matmul。

### 数据类型支持说明

`aclnnBatchMatMul` API 在不同芯片上的数据类型支持情况：

- **Atlas 训练系列产品、Atlas 推理系列产品**：
    - ✅ 支持 **FLOAT16**：标准的半精度浮点数
    - ✅ 支持 **FLOAT32**：单精度浮点数，提供最高精度

- **Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品**：
    - ✅ 支持 **BFLOAT16**：Brain Float 16，提供比 float16 更大的数值范围
    - ✅ 支持 **FLOAT16**：标准的半精度浮点数
    - ✅ 支持 **FLOAT32**：单精度浮点数，提供最高精度

**使用建议**：

- 在 Atlas A2/A3 系列芯片上，可以根据精度和性能需求选择合适的数据类型
- 在 Atlas 系列芯片上，可以使用 FLOAT16 或 FLOAT32
- 如果需要跨芯片兼容，建议统一使用 FLOAT16

## 使用示例

### 基本使用示例

```python
import mindspore as ms
import numpy as np
import ms_custom_ops

ms.set_context(device_target="Ascend")

@ms.jit
def batch_matmul_func(x1, x2, cube_math_type=0):
    return ms_custom_ops.batch_matmul(x1, x2, cube_math_type=cube_math_type)

# 示例1：基础批量矩阵乘法
# x1 shape: [A, M, K] = [2, 128, 256]
# x2 shape: [A, K, N] = [2, 256, 128]
# 输出 shape: [2, 128, 128]
batch = 2
m = 128
k = 256
n = 128
x1 = np.random.randn(batch, m, k).astype(np.float16)
x2 = np.random.randn(batch, k, n).astype(np.float16)

ms_x1 = ms.Tensor(x1)
ms_x2 = ms.Tensor(x2)
output = batch_matmul_func(ms_x1, ms_x2)
print("Output shape:", output.shape)
```

### Broadcast 示例

```python
# 示例2：Broadcast batch维度
# x1 shape: [A, M, K] = [2, 128, 256]
# x2 shape: [1, K, N] = [1, 256, 128]  (batch维度为1，会broadcast)
# 输出 shape: [2, 128, 128]
x1 = np.random.randn(2, 128, 256).astype(np.float16)
x2 = np.random.randn(1, 256, 128).astype(np.float16)

ms_x1 = ms.Tensor(x1)
ms_x2 = ms.Tensor(x2)
output = batch_matmul_func(ms_x1, ms_x2)
print("Output shape:", output.shape)
```

### 转置示例

```python
# 示例3：使用转置（需要先转置输入）
# 原始 x1 shape: [2, 256, 128]，需要转置为 [2, 128, 256]
# 原始 x2 shape: [2, 128, 256]，需要转置为 [2, 256, 128]
# 输出 shape: [2, 128, 128]
x1 = np.random.randn(2, 256, 128).astype(np.float16)
x2 = np.random.randn(2, 128, 256).astype(np.float16)

ms_x1 = ms.Tensor(x1)
ms_x2 = ms.Tensor(x2)

# 使用 transpose 算子先转置
x1_transposed = ms.ops.transpose(ms_x1, (0, 2, 1))  # [2, 128, 256]
x2_transposed = ms.ops.transpose(ms_x2, (0, 2, 1))  # [2, 256, 128]
output = batch_matmul_func(x1_transposed, x2_transposed)
print("Output shape:", output.shape)
```

