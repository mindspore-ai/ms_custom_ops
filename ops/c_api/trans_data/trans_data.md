# trans_data算子

## 描述

trans_data算子用于进行数据格式转换，支持ND格式与FRACTAL_NZ格式之间的相互转换，主要用于深度学习模型中的张量格式适配。

> **警告**：这是实验性接口，可能存在接口变化或删除的风险。

## 输入参数

| Name                   | DType           | Shape                                                                    | Description                    |
|------------------------|-----------------|--------------------------------------------------------------------------|--------------------------------|
| input                  | Tensor[float16/bfloat16/int8] | rank>=2的任意形状（最后两维为H、W） | 输入张量                       |
| transdata_type         | int             | -                                                                        | 转换类型                       |
|                        |                 |                                                                          | 0: FRACTAL_NZ_TO_ND           |
|                        |                 |                                                                          | 1: ND_TO_FRACTAL_NZ           |

## 输出参数

| Name   | DType           | Shape                                | Description |
|--------|-----------------|--------------------------------------|-------------|
| output | Tensor[float16/bfloat16/int8] | 与输入相同或根据转换规则调整的形状   | 转换后的张量    |

## 功能说明

### 转换类型说明

1. **ND_TO_FRACTAL_NZ (1)**：将ND格式张量转换为FRACTAL_NZ格式
   - 适用于需要加速计算的场景
   - 将张量重新组织为分块的内存布局

2. **FRACTAL_NZ_TO_ND (0)**：将FRACTAL_NZ格式张量转换为ND格式
   - 适用于需要标准张量操作的场景
   - 将分块的内存布局恢复为连续的ND格式

### 重要特性说明

#### 数据对齐规则

**H、W维度定义**：

- 对于rank>=2的张量，H是倒数第2维，W是倒数第1维
- 例如：shape=[batch, H, W] 或 shape=[N1, N2, ..., H, W]

**对齐常量**：

- H维度: 始终对齐到 16（所有数据类型）
- W维度:
    - float16/bfloat16: 对齐到 16
    - int8/uint8: 对齐到 32

**对齐要求与验证**：

> **注意**：以下对齐验证规则和异常行为**仅适用于 PyNative 模式**。Graph 模式下不会进行前端对齐检查，未对齐的输入可能导致硬件层报错或未定义行为。

**PyNative 模式的对齐验证**：

```text
ND_TO_FRACTAL_NZ 转换的对齐要求 (PyNative模式):

输入维度验证:
- H % 16 == 0   (必须是16的倍数，否则抛出异常)
- W % align == 0 (必须是对齐值的倍数，否则抛出异常)

其中 align = 16 (float16/bf16) 或 32 (int8)

示例(float16):
- 合法: H=16, W=16  (16%16=0, 16%16=0)
- 合法: H=32, W=32  (32%16=0, 32%16=0)
- 非法: H=15, W=16  (15%16!=0) -> 抛出 RuntimeError
- 非法: H=16, W=17  (17%16!=0) -> 抛出 RuntimeError

示例(int8):
- 合法: H=16, W=32  (16%16=0, 32%32=0)
- 合法: H=32, W=64  (32%16=0, 64%32=0)
- 非法: H=16, W=16  (16%32!=0) -> 抛出 RuntimeError
- 非法: H=16, W=33  (33%32!=0) -> 抛出 RuntimeError
```

**Graph 模式说明**：

在 Graph 模式下，算子不会进行前端对齐检查，未对齐的输入会直接传递到底层硬件。建议在 Graph 模式下使用前确保输入满足对齐要求，以避免潜在的运行时错误。

**形状转换公式**(仅适用于已对齐的输入)：

```text
ND转FRACTAL_NZ (3D输入为例，要求输入已对齐):
输入:  [batch, H, W]       (H%16=0, W%align=0)
输出:  [batch, W/align, H, align]

其中 align = 16 (float16/bf16) 或 32 (int8)

具体例子(float16):
输入:  [2, 16, 16]
输出:  [2, 1, 16, 16]  (16/16=1)

输入:  [2, 32, 64]
输出:  [2, 4, 32, 16]  (64/16=4)
```

## 使用示例

### 基本用法

```python
import mindspore as ms
import numpy as np
import ms_custom_ops

# 创建输入张量
input_data = ms.Tensor(np.random.rand(2, 16, 16), ms.float16)

# ND到FRACTAL_NZ转换
output_nz = ms_custom_ops.trans_data(
    input=input_data,
    transdata_type=1  # ND_TO_FRACTAL_NZ
)

# FRACTAL_NZ到ND转换 (显式指定原始形状)
output_nd = ms_custom_ops.trans_data(
    input=output_nz,
    transdata_type=0,  # FRACTAL_NZ_TO_ND
)
```

## 支持的运行模式

- **Graph Mode**：支持静态图模式执行
- **PyNative Mode**：支持动态图模式执行
