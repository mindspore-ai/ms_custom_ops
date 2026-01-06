# moe_init_routing_v2 算子

## 描述

MoE的routing计算，根据 gating TopK 的计算结果做routing处理。

## 输入参数

| Name | DType | Shape | Optional | Inplace | Format | Description |
|------|-------|-------|----------|---------|--------|-------------|
| x | Tensor(float16) | [num_rows, hidden_size] | No | No | ND | MOE的输入即token特征输入 |
| expert_idx | Tensor(int32) | [num_rows, k] | No | No | ND | 代表每个token激活的专家 |
| active_num | int |  | No | No | | 表示是否为Active场景，该属性在dropPadMode为0时生效，值范围大于等于0；<br>0表示Dropless场景，大于0时表示Active场景，约束所有专家共同处理tokens总量。<br> 当前仅支持0 |
| expert_capacity | int |  | No | No |  | 表示每个专家能够处理的tokens数，值范围大于等于0。目前仅支持0 |
| expert_num | int |  | No | No | | 表示专家数，值范围大于等于0 |
| drop_pad_mode | int |  | No | No |  | 表示是否为Drop/Pad场景，取值为0和1。目前仅支持0，表示非Drop/Pad场景，该场景下不校验 expert_capacity |
| expert_tokens_count_or_cumsum_flag | int | | No | No | | 当前只支持1，表示expertTokensCountOrCumsumOut输出的值为各个专家处理的token数量的累计值。 |
| expert_tokens_before_capacity_flag | bool | | No | No | | 当前只支持False，表示不输出expertTokensBeforeCapacityOut |

## 输出参数

| Name | DType | Shape | Description |
|------|-------|-------|-------------|
| expanded_x | Tensor(float16) | [num_row * k, hidden_size] | 根据expert_idx进行扩展过的特征 |
| expanded_row_idx | Tensor(int32) | [num_rows * k] | expanded_x 和 x 的索引映射关系 |
| expert_tokens_count_or_cumsum | Tensor(int32) | [expert_num] | 输出每个专家处理的token数量的统计结果及累加值，通过 expert_tokens_count_or_cumsum_flag 参数控制是否输出 |
| expert_tokens_before_capacity | Tensor(int32) | [expert_num] | 输出drop之前每个专家处理的token数量的统计结果, 当前输出无意义 |

## 特殊说明

- 当前仅在`hidden_size = 7168， k = 8` 时支持 `num_rows > 256`。
- `hidden_size` 整除 1024, 并且小于等于 7168
- `expert_num <= 256`

## 支持产品

- Atlas 推理系列产品

## 使用示例

### 基本使用示例（常规模式）

```python
import numpy as np
import mindspore as ms
import ms_custom_ops

num_rows = 64
hidden_size = 7168
expert_num = 256
k = 8
active_num = 0
expert_capacity = 0
drop_pad_mode = 0
expert_tokens_count_or_cumsum_flag = 1
expert_tokens_before_capacity_flag = False

x = np.random.uniform(-1, 1, size=(num_rows, hidden_size)).astype(np.float16)
expert_idx = np.random.randint(0, expert_num, size=(num_rows, k)).astype(np.int32)
x_ms = Tensor(x, ms.float16)
expert_idx_ms = Tensor(expert_idx, ms.int32)
probs = np.random.randn(token_num, top_k).astype(np.float16)
expanded_x, row_idx, group_list, _ = ms_custom_ops.moe_init_routing_v2(
                                        x_ms, expert_idx_ms, 0, active_num, expert_capacity,
                                        expert_num, drop_pad_mode, expert_tokens_count_or_cumsum_flag,
                                        expert_tokens_before_capacity_flag)
print("out:", out)
```
