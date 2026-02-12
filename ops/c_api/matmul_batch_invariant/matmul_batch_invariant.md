# matmul_batch_invariant 算子

## 描述

matmul_batch_invariant 算子用于执行批不变矩阵乘法操作。该算子支持1D到6D的Tensor输入，遵循标准的矩阵乘法语义，支持batch维度广播。"BatchInvariant"特性确保对于batch[0]的计算结果不受总batch大小的影响，这对于需要确定性计算的场景非常有用。

## 输入参数

| Name           | DType              | Shape     | Optional | Inplace | Format | Description                                          |
|----------------|--------------------|-----------|----------|---------|--------|------------------------------------------------------|
| x1              | Tensor(float16/bfloat16/float32) | 1-6维 | No       | No      | ND     | 左矩阵，shape为[..., M, K]。对于1D输入[K]，会自动补全为[1, K] |
| x2              | Tensor(float16/bfloat16/float32) | 1-6维 | No       | No      | ND     | 右矩阵，shape为[..., K, N]。对于1D输入[K]，会自动补全为[K, 1] |
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
| y    | Tensor(与输入类型相同) | 符合矩阵乘法规则的形状 | 矩阵乘法的计算结果，输出类型与x1相同 |

## 计算公式

out = x1 @ x2

其中：

- 对于2D输入：x1的shape是[M, K]，x2的shape是[K, N]，输出out的shape是[M, N]
- 对于多维输入：batch维度遵循广播规则
    - x1的shape是[..., M, K]，x2的shape是[..., K, N]，输出out的shape是[..., M, N]
- 对于1D输入的特殊处理：
    - 如果x1是1D [K]，x2是2D [K, N]，输出是1D [N]（向量-矩阵乘法）
    - 如果x1是2D [M, K]，x2是1D [K]，输出是1D [M]（矩阵-向量乘法）
    - 如果x1是1D [K1]，x2是1D [K2]（K1=K2=K），输出是标量（点积）

## 支持产品

- Atlas A3 训练系列产品、Atlas A3 推理系列产品
- Atlas A2 训练系列产品、Atlas A2 推理系列产品
- Atlas 训练系列产品
- Atlas 推理系列产品

## 约束说明

1. 支持1D到6D的Tensor输入
2. 1D输入会自动补全到2D：x1的[K]补全为[1, K]，x2的[K]补全为[K, 1]
3. 补全后的维度范围是2D到6D
4. batch维度遵循标准广播规则
5. x2的K维度需要与x1的K维度大小相等
6. **数据类型限制**：aclnnMatmulBatchInvariant 支持的数据类型取决于芯片型号：
    - **Atlas 训练系列产品、Atlas 推理系列产品**：数据类型支持 FLOAT16、FLOAT32
    - **Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品**：数据类型支持 BFLOAT16、FLOAT16、FLOAT32

### 数据类型支持说明

`aclnnMatmulBatchInvariant` API 在不同芯片上的数据类型支持情况：

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

### 基本使用示例（2D矩阵乘法）

```python
import mindspore as ms
import numpy as np
import ms_custom_ops

ms.set_context(device_target="Ascend")

@ms.jit
def matmul_batch_invariant_func(x1, x2, cube_math_type=0):
    return ms_custom_ops.matmul_batch_invariant(x1, x2, cube_math_type=cube_math_type)

# 示例1：基础2D矩阵乘法
# x1 shape: [M, K] = [128, 256]
# x2 shape: [K, N] = [256, 64]
# 输出 shape: [128, 64]
m, k, n = 128, 256, 64
x1 = np.random.randn(m, k).astype(np.float16)
x2 = np.random.randn(k, n).astype(np.float16)

ms_x1 = ms.Tensor(x1)
ms_x2 = ms.Tensor(x2)
output = matmul_batch_invariant_func(ms_x1, ms_x2)
print("Output shape:", output.shape)  # (128, 64)
```

### 批量矩阵乘法示例（3D）

```python
# 示例2：批量矩阵乘法
# x1 shape: [batch, M, K] = [4, 128, 256]
# x2 shape: [batch, K, N] = [4, 256, 64]
# 输出 shape: [4, 128, 64]
batch, m, k, n = 4, 128, 256, 64
x1 = np.random.randn(batch, m, k).astype(np.float16)
x2 = np.random.randn(batch, k, n).astype(np.float16)

ms_x1 = ms.Tensor(x1)
ms_x2 = ms.Tensor(x2)
output = matmul_batch_invariant_func(ms_x1, ms_x2)
print("Output shape:", output.shape)  # (4, 128, 64)
```

### 高维批量矩阵乘法示例（4D）

```python
# 示例3：4D批量矩阵乘法（常见于多头注意力）
# x1 shape: [batch, heads, M, K] = [2, 8, 64, 128]
# x2 shape: [batch, heads, K, N] = [2, 8, 128, 64]
# 输出 shape: [2, 8, 64, 64]
batch, heads, m, k, n = 2, 8, 64, 128, 64
x1 = np.random.randn(batch, heads, m, k).astype(np.float16)
x2 = np.random.randn(batch, heads, k, n).astype(np.float16)

ms_x1 = ms.Tensor(x1)
ms_x2 = ms.Tensor(x2)
output = matmul_batch_invariant_func(ms_x1, ms_x2)
print("Output shape:", output.shape)  # (2, 8, 64, 64)
```

### 1D输入示例

```python
# 示例4：向量-矩阵乘法
# x1 shape: [K] = [256]（自动补全为[1, 256]）
# x2 shape: [K, N] = [256, 64]
# 输出 shape: [64]（移除补全的维度）
k, n = 256, 64
x1 = np.random.randn(k).astype(np.float16)
x2 = np.random.randn(k, n).astype(np.float16)

ms_x1 = ms.Tensor(x1)
ms_x2 = ms.Tensor(x2)
output = matmul_batch_invariant_func(ms_x1, ms_x2)
print("Output shape:", output.shape)  # (64,)

# 示例5：矩阵-向量乘法
# x1 shape: [M, K] = [128, 256]
# x2 shape: [K] = [256]（自动补全为[256, 1]）
# 输出 shape: [128]（移除补全的维度）
m, k = 128, 256
x1 = np.random.randn(m, k).astype(np.float16)
x2 = np.random.randn(k).astype(np.float16)

ms_x1 = ms.Tensor(x1)
ms_x2 = ms.Tensor(x2)
output = matmul_batch_invariant_func(ms_x1, ms_x2)
print("Output shape:", output.shape)  # (128,)
```

### Broadcast 示例

```python
# 示例6：Broadcast batch维度
# x1 shape: [4, 128, 256]
# x2 shape: [1, 256, 64]（batch维度为1，会broadcast到4）
# 输出 shape: [4, 128, 64]
x1 = np.random.randn(4, 128, 256).astype(np.float16)
x2 = np.random.randn(1, 256, 64).astype(np.float16)

ms_x1 = ms.Tensor(x1)
ms_x2 = ms.Tensor(x2)
output = matmul_batch_invariant_func(ms_x1, ms_x2)
print("Output shape:", output.shape)  # (4, 128, 64)
```
