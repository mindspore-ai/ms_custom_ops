# scatter_nd_update算子

## 描述

scatter_nd_update算子用于计算旋转编码操作。
该算子底层调用的是aclnnScatterNdUpdate算子。

## 输入参数

| Name                | DType           | Shape                                  | Optional | Inplace | Format | Description                                            |
|---------------------|-----------------|----------------------------------------|----------|---------|--------|--------------------------------------------------------|
| input               | Tensor          | 1~8维 | No       | Yes      | ND     | 数据类型需与updates一致。同时作为输出。                              |
| indices                 | Tensor          | 至少2维 | No       | No      | ND     | 索引张量，数据类型为int32或者int64,索引数据不支持越界                              |  
| updates                 | Tensor          |           | No       | No      | ND     | 更新值张量，数据类型需要与VarRef一致                                |

Note:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品​：

+ input和updates支持：FLOAT16、BFLOAT16、FLOAT32、INT64、BOOL、INT8

+ indices支持：INT32、INT64

Atlas 训练系列产品、Atlas 推理系列产品​：

+ input和updates支持：FLOAT16、FLOAT32、BOOL
+ indices支持：INT32、INT64

约束说明:
1.indices至少是2维，其最后1维的大小（记为a）不能超过input的维度大小
2.updates的形状必须等于indices除最后1维外的形状加上input除前a维外的形状
例如：input的shape是(4, 5, 6)，indices的shape是(3, 2)，则updates的shape必须是(3, 6)

## 输出参数

该接口无输出返回值，计算结果将就地更新至输入input中。

更多详细信息请参考：[aclnnScatterNdUpdate](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/API/aolapi/context/aclnnScatterNdUpdate.md)

## 特殊说明

## 使用示例

### 基本使用示例

```python
import mindspore as ms
import numpy as np
import ms_custom_ops
from mindspore import context, Tensor

ms.set_device("Ascend")

@ms.jit
def scatter_nd_update_func(input, indices, updates):
   return ms_custom_ops.scatter_nd_update(input, indices, updates)

data_input = np.random.uniform(0, 1, [24, 128]).astype(np.float16)
data_indices = np.random.uniform(0, 12, [12, 1]).astype(np.int32)
data_updates = np.random.uniform(1, 2, [12, 128]).astype(np.float16)


input = Tensor(data_input, dtype=ms.float16)
indices = Tensor(data_indices, dtype=ms.int32)
updates = Tensor(data_updates, dtype=ms.float16)
scatter_nd_update_func(input, indices, updates)
print("result:", input)
```
