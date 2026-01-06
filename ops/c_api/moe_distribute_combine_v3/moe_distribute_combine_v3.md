# moe_distribute_combine_v3

## 描述

算子功能：当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）。该算子底层调用的是aclnnMoeDistributeCombineV3算子。注意该接口必须与moe_distribute_dispatch_v3配套使用，相当于按moe_distribute_dispatch_v3接口收集数据的路径原路返还。

## 输入参数

| Name | DType | Shape | Optional | Inplace | Format | Description |
|------|-------|-------|----------|---------|--------|-------------|
| expand_x | Tensor | 2维，参见约束说明 | No | No | ND | 根据expertIds进行扩展过的token特征,支持Float16/BFloat16 |
| expert_ids | Tensor | 2维[Bs, K] | No | No | ND | 每个token的topK个专家索引 |
| assist_info_for_combine | Tensor | 1维，参见约束说明 | No | No | ND | moe_distribute_dispatch_ve中的assist_info_for_combine输出 |
| ep_send_counts | Tensor | 1维，参见约束说明 | No | No | ND | moe_distribute_dispatch_ve中的ep_recv_counts输出 |
| expert_scales | Tensor | 2维[Bs, K] | No | No | ND | 每个token的topK个专家的权重 |
| ep_world_size | int | - | No | No | - | EP通信域size |
| ep_rank_id | int | - | No | No | - | EP域本卡Id |
| moe_expert_num | int | - | No | No | - | MoE专家数量 |
| group_ep | str | 字符串长度在[1, 128) | No | No | - | EP通信域名称，专家并行的通信域 |
| tp_send_counts | Tensor | 1维，参见约束说明 | Yes | No | ND | moe_distribute_dispatch_ve中的tp_recv_counts输出 |
| x_activemask | Tensor | 1维或2维，参见约束说明 | Yes | No | ND | 表示token是否参与通信，默认值None |
| activation_scale | Tensor | - | Yes | No | - | 预留参数，当前版本不支持，默认值None |
| weight_scale | Tensor | - | Yes | No | - | 预留参数，当前版本不支持，默认值None |
| group_list | Tensor | - | Yes | No | - | 预留参数，当前版本不支持，默认值None |
| expand_scales | Tensor | 1维，参见约束说明 | Yes | No | ND | moe_distribute_dispatch_ve中的expand_scales输出，默认值None |
| shared_expert_x | Tensor | 2维或3维，参见约束说明 | Yes | No | ND | 共享专家计算后的token，默认值None |
| elastic_info | Tensor | 1维，参见约束说明 | Yes | No | ND | 表示RP通信域的动态缩容信息，当某些通信卡因异常而从通信域中剔除，实际参与通信的卡数可从本参数中获取，默认值None |
| ori_x | Tensor | 2维，参见约束说明 | Yes | No | ND | 表示未经过FFN的token数据，默认值None |
| const_expert_alpha1 | Tensor | 1维，参见约束说明 | Yes | No | ND | 在使能const_expert的场景下需要输入的计算系数，默认值None |
| const_expert_alpha2 | Tensor | 1维，参见约束说明 | Yes | No | ND | 在使能const_expert的场景下需要输入的计算系数，默认值None |
| const_expert_v | Tensor | 2维，参见约束说明 | Yes | No | ND | 在使能const_expert的场景下需要输入的计算系数，默认值None |
| group_tp | str | 字符串，参见约束说明 | Yes | No | - | - | TP通信域名称，数据并行的通信域，默认值None |
| tp_world_size | int | - | Yes | No | - | TP通信域的size，默认值0 |
| tp_rank_id | int | - | Yes | No | - | TP域本卡Id，默认值0 |
| expert_shard_type | int | - | Yes | No | - | 表示共享专家卡分布类型，默认值0 |
| shared_expert_num | int | - | Yes | No | - | 表示共享专家数量，一个共享专家可以复制部署到多个卡上，默认值1 |
| shared_expert_rank_num | int | - | Yes | No | - | 表示共享专家卡数量，默认值0 |
| global_bs | int | - | Yes | No | - | EP域全局的batch_size大小，默认值0 |
| out_dtype | int | - | Yes | No | - | 预留参数，当前版本不支持，默认值0 |
| common_quant_mode | int | - | Yes | No | - | 通信量化类型，默认值0 |
| group_list_type | int | - | Yes | No | - | 预留参数，当前版本不支持，默认值0 |
| comm_alg | str | - | Yes | No | - | 表示通信亲和内存布局算法，参见约束说明，默认值None |
| zero_expert_num | int | - | Yes | No | - | 表示零专家的数量，默认值0 |
| copy_expert_num | int | - | Yes | No | - | 表示拷贝专家的数量，默认值0 |
| const_expert_num | int | - | Yes | No | - | 表示常量专家的数量，默认值0 |

## 参数说明和约束

- 支持产品：Atlas A3 训练系列产品/Atlas A3 推理系列产品，Atlas A2 训练系列产品/Atlas A2 推理系列产品
- **expand_x**: shape为 (max(tp_world_size, 1) x A , H)，A表示本卡需要分发的最大token数量，H为hidden_size，支持dtype为Float16/BFloat16
- **assist_info_for_combine**: 1D tensor，shape=(A x 128, )
- **ep_send_counts**:
    - A2机器上，shape(moe_expert_num + 2 x global_bs x K x server_num, )
    - A3机器上，shape(ep_world_size x max(tp_world_size, 1) x local_expert_num, )
- **ep_world_size**:
    - A2机器上，取值支持16、32、64；
    - A3机器上，取值支持[2, 768]
- **ep_rank_id**: 取值范围为[0, ep_world_size),同一个EP通信域内各卡的ep_rank_id不重复
- **moe_expert_num**: 需要满足moe_expert_num % (ep_world_size -shared_expert_rank_num) = 0
    - A2机器上，取值范围(0, 512],且需要满足moe_expert_num / (ep_world_size - shared_expert_rank_num) <= 24
    - A3机器上，取值范围为(0, 1024]
- **tp_send_counts**: 若有TP域通信需要传参，若无TP域通信，传None即可
    - A2机器上，当前不支持TP域通信
    - A3机器上，有TP域通信时，shape为 (tp_world_size, )
- **x_active_mask**:
    - A2机器上，当前不支持，传None即可
    - A3机器上，可传入1维或者2维tensor。当输入为1维tensor，shape为(Bs)；当传入2维tensor,shape为(Bs, K).数据类型为Bool,可选择传入None或有效数据。当输入为1维时,参数为True表示对应Token参与通信，且True必须排到False之前，非法输入示例:[True, False, True].当参数为2维tensor，参数为True表示当前token对应expert_ids参与通信。若当前token对应的K个BOOL值全为False，表示当前token不会参与通信。默认所有token都会参与通信。当每张卡的Bs数量不一致时，所有token必须全部有效
- **expand_scales**:
    - A2机器上，1维tensor，shape为 (A, )
    - A3机器上，当前不支持，传默认值None
- **shared_expert_x**:
    - A2机器上，当前不支持，传默认值None
    - A3机器上，要求是一个2D或3D的Tensor，当Tensor为2D时，shape为 (Bs, H)；当Tensor为3D时，前两位的乘积需等于Bs，第三维需等于H。数据类型需跟expand_x保持一致。可传/可不传，传入时，shared_expert_rank_num需为0
- **elastic_info**:
    - A2机器上, 当前不支持，传None
    - A3机器上，可传入None或实际有效数据；传入None表示不使能动态缩容功能；当传入有效数据，要求为1维的(4 + 2 x ep_world_size)tensor。Tensor中前四个数字表示(是否缩容，缩容后实际rank数，缩容后共享专家使用的rank数，缩容后moe专家的个数),后续2 x ep_world_size表示2个rank映射表，缩容后本卡中因部分rank异常而从EP通信域中剔除，第一个Table的映射关系为Table1[epRankId]=localEpRankId或-1，localEpRankId表示新EP通信域中的rank Index，-1表示epRankId这张卡从通信域中被剔除，第二个Table映射关系为Table2[localEpRankId] = epRankId
- **ori_x**: 在使能copy_expert或使能const_expert的场景下需要本输入数据
    - A2机器上，当前不支持，传默认值None
    - A3机器上，可选择传入有效数据或None，当copy_expert_num不为0或const_expert_num不为0时必须传入有效输入；当传入有效数据时，要求是一个2D的Tensor，shape为 (Bs, H)，数据类型需跟expand_x保持一致
- **const_expert_alpha1**: 在使能const_expert的场景下需要本输入数据
    - A2机器上，当前不支持，传默认值None
    - A3机器上，可选择传入有效数据或填空指针，当const_expert_num不为0时必须传入有效输入；当传入有效数据时，要求是一个1D的Tensor，shape为 (const_expert_num, )，数据类型需跟expand_x保持一致
- **const_expert_alpha2**: 在使能const_expert的场景下需要本输入数据
    - A2机器上，当前不支持，传默认值None
    - A3机器上，可选择传入有效数据或填空指针，当const_expert_num不为0时必须传入有效输入；当传入有效数据时，要求是一个1D的Tensor，shape为 (const_expert_num, )，数据类型需跟expand_x保持一致
- **const_expert_v**: 在使能const_expert的场景下需要本输入数据
    - A2机器上，当前不支持，传默认值None
    - A3机器上，可选择传入有效数据或填空指针，当const_expert_num不为0时必须传入有效输入；当传入有效数据时，要求是一个2D的Tensor，shape为 (const_expert_um, H)，数据类型需跟expand_x保持一致
- **group_tp**:
    - A2机器上，当前版本不支持，传空字符
    - A3机器上，字符串长度范围为[1, 128)，不能和group_ep相同
- **tp_world_size**:
    - A2机器上，当前版本不支持，传0
    - A3机器上，取值范围[0, 2]，0和1表示无TP域通信，有TP域通信时仅支持2
- **tp_rank_id**:
    - A2机器上，当前版本不支持，传0
    - A3机器上，取值范围[0, 1]，同一个TP通信域中各卡的tp_rank_id不重复。无TP域通信时，传0即可
- **expert_shard_type**:
    - A2机器上，当前版本不支持，传0
    - A3机器上，当前仅支持传0，表示共享专家卡排在MoE专家卡前面
- **shared_expert_num**:
    - A2机器上，当前版本不支持，传0
    - A3机器上，当前取值范围[0, 4]
- **shared_expert_rank_num**:
    - A2机器上，当前版本不支持，传0
    - A3机器上，当前取值范围[0, ep_world_size)，为0时需满足shared_expert_num为0或1，不为0时需满足shared_expert_rank_num % shared_expert_num = 0
- **global_bs**: 当每个rank的Bs数一致时，global_bs = Bs x ep_world_size 或 global_bs = 0；当每个rank的Bs数不一致时，global_bs = max_bs x ep_world_size，其中max_bs表示单卡Bs最大值
- **common_quant_mode**:
    - A2机器上，取值范围0或者2，0表示通信时不进行量化，2表示通信时进行int8量化，2仅当comm_alg配置为"hierarchy"或HCCL_INTRA_PCIE_ENABLE为1且HCCL_INTRA_ROCE_ENABLE为0且驱动版本不低于25.0.RC1.1时支持
    - A3机器上，取值范围0或者2，0表示通信时不进行量化，2表示通信时进行int8量化，int8量化当且仅当tp_world_size < 2时可使能
- **comm_alg**:
    - A2机器上,当前版本支持nullptr,"","fullmesh","hierarchy"四种输入方式.推荐配置"hierarchy"并搭配25.0.RC1.1及以上版本驱动使用
        - nullptr和"": 仅在此场景下,HCCL_INTRA_PCIE_ENABLE和HCCL_INTRA_ROCE_ENABLE配置生效.当HCCL_INTRA_PCIE_ENABLE=1&&HCCL_INTRA_ROCE_ENABLE=0时,调用"hierarchy"算法,否则调用"fullmesh"算法.不推荐使用该方式
        - "fullmesh": token数据直接通过RDMA方式发往topk个目标专家所在的卡
        - "hierarchy": token数据经过跨机、机内两次发送，仅不同server同号卡之间使用RDMA通信，server内使用HCCS通信
    - A3机器上，当前不支持，传默认值None
- **zero_expert_num**:
    - A2机器上，当前不支持，传入0即可
    - A3机器上，取值范围:[0, MAX_INT32),MAX_INT32 = 2^31 - 1, 合法的零专家的ID的值是[moe_expert_num, moe_expert_num + zero_expert_num)
- **copy_expert_num**:    - A2机器上，当前不支持，传入0即可
    - A3机器上，取值范围:[0, MAX_INT32),MAX_INT32 = 2^31 - 1, 合法的零专家的ID的值是[moe_expert_num + zero_expert_num, moe_expert_num + zero_expert_num + copy_expert_num)
- **const_expert_num**:
    - A2机器上，当前不支持，传入0即可
    - A3机器上，取值范围:[0, MAX_INT32),MAX_INT32 = 2^31 - 1, 合法的零专家的ID的值是[moe_expert_num + zero_expert_num + copy_expert_num, moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num)

## 输出参数

| Name   | DType      | Shape      | Description           |
|--------|------------|------------|-----------------------|
| x_out| Tensor(float16/bf16)  | [Bs, H]| 表示处理后的token |

更多详细信息请参考：[aclnnMoeDistributeCombineV3](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/API/aolapi/context/aclnnMoeDistributeCombineV3.md)

## 使用示例

```python
# Atlas A3 机器
import numpy as np
import mindspore as ms
import ms_custom_ops
from mindspore.communication import init, get_rank, GlobalComm
bs = 8
h = 7168
k = 8
ep_world_size = 8
moe_expert_num = 16
global_bs = bs * ep_world_size
x = Tensor(np.random.randn(bs, h), ms.float16)
expert_ids = Tensor(np.random.randint(0, moe_expert_num, (bs, k)), ms.int32)
expert_scales = Tensor(np.random.randn(bs, k), ms.float32)
init()
rank_id = get_rank()
expand_x, _, assist_info_for_combine, _, ep_recv_count, _, _ = ms_custom_ops.moe_distribute_dispatch_v3(
    x=x,
    expert_ids=expert_ids,
    group_ep='ep-0-1-2-3-4-5-6-7',
    ep_world_size=ep_world_size,
    ep_rank_id=rank_id,
    moe_expert_num=moe_expert_num,
    global_bs=global_bs)
out_x = ms_custom_ops.moe_distribute_combine_v3(
    expand_x=expand_x,
    expert_ids=expert_ids,
    assist_info_for_combine=assist_info_for_combine,
    expert_scales=expert_scales,
    ep_send_counts=ep_recv_count,
    group_ep='ep-0-1-2-3-4-5-6-7',
    ep_world_size=ep_world_size,
    ep_rank_id=rank_id,
    moe_expert_num=moe_expert_num,
    global_bs=global_bs)
print(out_x)
```