# grouped_matmul_v4算子

## 描述

grouped_matmul_v4算子用于执行分组矩阵乘法操作，支持多种量化格式（如int8、int4），并提供丰富的配置选项，包括偏置、缩放、偏移、激活函数等。该算子适用于高效处理大规模矩阵乘法计算，尤其在深度学习模型中。

## 输入参数

| Name | DType | Shape | Optional | Inplace | Format | Description |
|------|-------|-------|----------|---------|--------|-------------|
| x | List[Tensor] | 多维张量列表 | No | No | ND | 输入特征张量列表 |
| weight | List[Tensor] | 多维张量列表 | No | No | ND/FRACTAL_NZ | 权重张量列表，支持int4量化 |
| bias | List[Tensor] | 适当广播的形状列表 | Yes | No | ND | 偏置张量列表 |
| scale | List[Tensor] | 适当广播的形状列表 | Yes | No | ND | 缩放因子列表，用于量化/反量化过程 |
| offset | List[Tensor] | 适当广播的形状列表 | Yes | No | ND | 偏移量列表 |
| antiquant_scale | List[Tensor] | 适当广播的形状列表 | Yes | No | ND | 反量化缩放因子列表 |
| antiquant_offset | List[Tensor] | 适当广播的形状列表 | Yes | No | ND | 反量化偏移量列表 |
| per_token_scale | List[Tensor] | 适当广播的形状列表 | Yes | No | ND | 逐token缩放因子列表 |
| group_list | Tensor | 一维张量 | Yes | - | ND | 分组信息列表 |
| activation_input | List[Tensor] | 多维张量列表 | Yes | No | ND | 激活函数输入张量列表 |
| activation_quant_scale | List[Tensor] | 适当广播的形状列表 | Yes | No | ND | 激活函数量化缩放因子列表 |
| activation_quant_offset | List[Tensor] | 适当广播的形状列表 | Yes | No | ND | 激活函数量化偏移量列表 |
| split_item | int64 | - | Yes | - | - | 分割项配置 |
| group_type | int64 | - | Yes | - | - | 分组类型配置，0表示常规分组，-1表示多输出 |
| group_list_type | int64 | - | Yes | - | - | 分组列表类型配置 |
| act_type | int64 | - | Yes | - | - | 激活函数类型配置 |
| weight_format | str | - | Yes | - | - | 权重格式，可选值为"ND"或"FRACTAL_NZ"，默认为"ND" |
| output_dtype | int64 | - | Yes | - | - | 输出数据类型，支持float16、bfloat16 |

## 输出参数

| Name | DType | Shape | Description |
|------|-------|-------|-------------|
| output | Tensor/List[Tensor] | 符合分组矩阵乘法规则的形状 | 分组矩阵乘法的计算结果，当group_type为-1时返回多个输出 |

更多详细信息请参考：[aclnnGroupedMatmulV4](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/aolapi/context/aclnnGroupedMatmulV4.md)

## 支持平台

- Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件
- Atlas 推理系列产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品

## 使用示例

### 基本使用示例

```python
import mindspore as ms
import numpy as np
import ms_custom_ops

ms.set_device("Ascend")

@ms.jit
def grouped_matmul_v4_func(x, weight, bias=None, scale=None,
                          offset=None, antiquant_scale=None,
                          antiquant_offset=None, per_token_scale=None,
                          group_list=None, activation_input=None,
                          activation_quant_scale=None,
                          activation_quant_offset=None, split_item=0,
                          group_type=0, group_list_type=0, act_type=0,
                          weight_format="ND", output_dtype=ms.float16):
    return ms_custom_ops.grouped_matmul_v4_cops(
        x, weight, bias, scale, offset,
        antiquant_scale, antiquant_offset, per_token_scale,
        group_list, activation_input, activation_quant_scale,
        activation_quant_offset, split_item, group_type,
        group_list_type, act_type, weight_format, output_dtype)

# 准备输入数据
expert_num = 4
seq_len = 128
hidden_size = 768

# 创建输入张量列表
x = [ms.Tensor(np.random.randint(-128, 127, size=(seq_len, hidden_size)), dtype=ms.int8)]
# 创建权重张量列表
weight = [ms.Tensor(np.random.randint(-8, 7, size=(expert_num, hidden_size, hidden_size)), dtype=ms.int8)]
# 创建分组列表
group_list = ms.Tensor([0, 1, 2, 3], dtype=ms.int64)

# 执行分组矩阵乘法
output = grouped_matmul_v4_func(
    x, weight, group_list=group_list,
    split_item=3, group_type=0, group_list_type=0, act_type=0,
    weight_format="ND", output_dtype=ms.float16
)
print("Output shape:", output[0].shape)
```

### 多输出模式示例

```python
import mindspore as ms
import numpy as np
import ms_custom_ops

ms.set_device("Ascend")

# 准备多个输入和权重
seq_len = 64
input_size = 512
output_size = 256

# 创建多个输入和权重
x = [
    ms.Tensor(np.random.randint(-128, 127, size=(seq_len, input_size)), dtype=ms.int8),
    ms.Tensor(np.random.randint(-128, 127, size=(seq_len, input_size)), dtype=ms.int8)
]
weight = [
    ms.Tensor(np.random.randint(-8, 7, size=(input_size, output_size)), dtype=ms.int8),
    ms.Tensor(np.random.randint(-8, 7, size=(input_size, output_size)), dtype=ms.int8)
]

# 多输出模式 (group_type = -1)
outputs = ms_custom_ops.grouped_matmul_v4_cops(
    x, weight,
    split_item=0, group_type=-1, group_list_type=0, act_type=0,
    weight_format="ND"
)

print("Number of outputs:", len(outputs))
for i, out in enumerate(outputs):
    print(f"Output {i+1} shape:", out.shape)
```