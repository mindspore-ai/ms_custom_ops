# fa_update算子

## 描述

fa_update算子实现了attention部分中间结果卡间同步。

## 输入参数

| Name | DType | Shape | Optional | Inplace | Format | Description |
|------|-------|-------|----------|---------|--------|-------------|
| lse | Tensor(fp32) | [sp, batch * seqLen * headNum] | No | No | ND | 输入tensor，数据类型为float32, 各SP域计算的lse |
| local_out | Tensor(fp32) | [sp, batch * seqLen * headNum, head_size] | No | No | ND | 输入tensor，数据类型为float32， 各SP域计算的output |
| fa_update_type | int |  | No | No |  |指定下标需要执行的操作类型，目前只支持0：DECODE_UPDATE |
| sp | int |  | No | No |  |序列并行的并行度SP，取值范围[1, 8] |

注意：

- head_size必须是[8, 512]， 且是8的倍数

## 输出参数

| Name | DType | Shape | Description |
|------|-------|-------|-------------|
| output | Tensor(fp32) | [batch * seqLen * headNum, head_size] | 全局output，数据类型float32 |

## 使用示例

### 基本使用示例（常规模式）

```python

import numpy as np
import pytest
from mindspore import Tensor, context
import mindspore as ms
import random
import ms_custom_ops

context.set_context(mode=context_mode, device_target="Ascend")
context.set_context(jit_config={"jit_level": "O0"})

class AsdFaUpdateCustom(ms.nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, lse, local_out, fa_update_type, sp):
        return ms_custom_ops.fa_update(lse, local_out, fa_update_type, sp)

sp = 1
head_size = 8
batch = 11
seq_len = 1
head_num = 13
shape0 = (sp, batch * seq_len * head_num)
shape1 = (sp, batch * seq_len * head_num, head_size)
lse = np.random.rand(*shape0).astype(np.float32)
local_out = np.random.rand(*shape1).astype(np.float32)
net = AsdFaUpdateCustom()
output = net(Tensor(lse), Tensor(local_out), 0, sp)

```
