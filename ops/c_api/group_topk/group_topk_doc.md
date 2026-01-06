# group_topk 算子

## 描述

group_topk算子将输入token中最后一维度的数据分为group_num个组，每个组取最大值，然后选出每个组最大值中的前k个，最后将非前k个组的数据全部置零。

## 输入参数

| Name            | DType                       | Shape                        | Inplace | Format | Description                                         |
|-----------------|-----------------------------|------------------------------|---------|--------|-----------------------------------------------------|
| token           | Tensor[float16/bfloat16]    | (num_tokens, expert_num)/(batch_size, seq_len, expert_num)     | Yes     | ND     | 若为二维Tensor，维度0为token数，维度1为专家总数。<br>若为三维Tensor，维度0乘维度1为token数，维度2为专家总数。 |
| idx_arr         | Tensor[int32]               | (idx_arr_dim)                | No      | ND     | 一维Tensor，长度大于专家总数的等差序列，用于辅助计算。推荐使用固定长度1024，[0,1,2,...,1023]。 |
| group_num       | int                         | -                            | -       | -      | 每个token分组数量，取值范围为[1, expert_num]。 |
| k               | int                         | -                            | -       | -      | 选择top K专家数量，取值范围为[1, group_num]。 |
| k_inner         | int                         | -                            | -       | -      | 计算每组得分时取最大值的数量，默认值为1。 |

## 输出参数

该接口无输出返回值，计算结果将原地更新至输入token中。

## 规格约束

- 1 ≤ expert_num ≤ 1024;
- expert_num ≥ group_num ≥ k ≥ 1;
- expert_num能够被group_num整除;
- 1 ≤ k_inner ≤ expert_num/group_num;
- idx_arr_dim ≥ expert_num;

## 使用示例

```python
import numpy as np
import mindspore as ms
import ms_custom_ops


# 创建输入张量
token = ms.Tensor(np.random.rand(512, 256), ms.float16)
idx_arr = ms.Tensor(np.arange(1024, dtype=np.int32))
group_num = 8
k = 8
k_inner = 2

# 调用算子
ms_custom_ops.group_topk(token=token, idx_arr=idx_arr, group_num=group_num,
                         k=k, k_inner=k_inner)
```