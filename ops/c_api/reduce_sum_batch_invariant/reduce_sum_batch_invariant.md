# reduce_sum_batch_invariant

## 功能说明

`reduce_sum_batch_invariant` 是一个具有 batch 不变性的求和算子。该算子在指定维度上对输入张量进行求和操作，同时保证对于相同的 batch[0] 输入数据，无论总 batch 大小如何，都能产生相同的输出结果。

## 产品支持情况

| 产品 | 是否支持 |
|:-----|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数原型

```python
ms_custom_ops.reduce_sum_batch_invariant(input, dims, keep_dims, output_dtype)
```

## 参数说明

- **input** (Tensor): 输入张量，shape 支持 0-8 维，支持非连续的 Tensor，数据格式支持 ND。
  - 数据类型支持：FLOAT16、FLOAT32、BFLOAT16

- **dims** (tuple[int]): 指定要进行求和的维度，数据类型为 INT64，取值范围为 [-input.dim(), input.dim()-1]。空列表表示对所有维度求和。

- **keep_dims** (bool): 指定是否在输出张量中保留被求和的维度。
  - True: 保留维度，被求和的维度大小变为 1
  - False: 不保留维度，输出张量的维度数减少

- **output_dtype** (mindspore.dtype): 指定输出张量的数据类型，如 `ms.float32`。
  - 数据类型支持：FLOAT16、FLOAT32、BFLOAT16

## 返回值

- **out** (Tensor): 输出张量，shape 根据 dims 和 keep_dims 参数确定，数据类型由 dtype 参数指定。

## 约束说明

1. 输入张量维度范围为 0-8 维
2. dims 参数中的维度值必须在有效范围内
3. dims 参数中不能有重复的维度值
4. dtype 的数据类型必须在支持范围内

## 确定性计算

`reduce_sum_batch_invariant` 默认使用确定性实现，保证 batch 不变性。

## 调用示例

```python
import numpy as np
import mindspore as ms
from mindspore import Tensor, context
import ms_custom_ops

# 设置运行环境
ms.set_context(device_target="Ascend", mode=context.PYNATIVE_MODE)

# 创建输入数据
input_data = np.random.randn(4, 32, 128).astype(np.float32)
input_tensor = Tensor(input_data, dtype=ms.float32)

# 调用 reduce_sum_batch_invariant
# 在最后一个维度上求和，不保留维度
dims = [-1]
keep_dims = False
output_dtype = ms.float32

out = ms_custom_ops.reduce_sum_batch_invariant(input_tensor, dims, keep_dims, output_dtype)
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {out.shape}")
# 输出: Input shape: (4, 32, 128)
#       Output shape: (4, 32)

# 在多个维度上求和，保留维度
dims = [1, 2]
keep_dims = True

out = ms_custom_ops.reduce_sum_batch_invariant(input_tensor, dims, keep_dims, output_dtype)
print(f"Output shape with keep_dims=True: {out.shape}")
# 输出: Output shape with keep_dims=True: (4, 1, 1)
```

## 与标准 ops.sum 的区别

`reduce_sum_batch_invariant` 与标准的 `mindspore.ops.sum` 功能类似，主要区别在于：

1. **Batch 不变性**: 该算子保证对于相同的 batch[0] 输入，无论整体 batch 大小如何变化，输出结果保持一致
2. **确定性计算**: 默认使用确定性算法，确保多次运行结果一致
3. **专用优化**: 针对 Ascend 硬件进行了专门优化

## 使用场景

该算子特别适用于以下场景：

1. 需要保证计算确定性的推理服务
2. 动态 batch 场景下要求输出一致性的应用
3. 分布式计算中需要保证跨节点结果一致性的场景
