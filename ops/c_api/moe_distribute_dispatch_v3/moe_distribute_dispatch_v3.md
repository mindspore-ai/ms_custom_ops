# moe_distribute_dispatch_v3算子

## 描述

moe_distribute_dispatch_v3算子对token数据进行量化(可选)，当存在TP通信域时，先进行EP(Expert parallelism)域的AllToAll通信，再进行TP(Tensor Parallelism)域的AllGatherV通信；当不存在TP通信域时，进行TP域的AllToAllV通信。该算子底层调用的是aclnnMoeDistributeDispatchV3算子。

## 相对于aclnnMoeDistributeDispatchV2算子

- 新增支持动态缩容场景，支持在创建通信域后，出现故障卡，将故障卡冲通信域中剔除，算子可正常执行，无需重新编译，通过传入elastic_info参数使能本特性
- 新增支持特殊专家场景
    - zeroExpertNum非0时使能该特性, $MoE(oriXOptional) = 0$
    - copyExpertNum非0时使能该特性, 同时还需要传入有效的oriXOptional参数,$MoE(oriXOptional) = oriXOptional$
    - constExpertNum非0时使能该特性, 同时还需传入有效的oriXOptional、constExpertAlpha1Optional、constExpertAlpha2Optional、constExpertVOptional参数.$MoE(oriXOptional) = constExpertAlpha1Optional*oriXOptional+constExpertAlpha2Optional*constExpertVOptional$

## 输入参数

| Name | DType | Shape | Optional | Inplace | Format | Description |
|------|-------|-------|----------|---------|--------|-------------|
| x | Tensor | 2维[Bs, H] | No | No | ND | 表示本卡发送的token数据 |
| expert_ids | Tensor | 2维[Bs, K] | No | No | ND | 每个token的topK个专家索引 |
| ep_world_size | int | No | No | No | - | EP通信域size |
| ep_rank_id | int | No | No | No | - | EP域中本卡ID |
| moe_expert_num | int | No | No | No | - | MoE专家数量 |
| scales | Tensor | 2维[sharedExpertNum + moeExpertNum, H] | Yes | No | ND | 每个专家的量化平滑参数 |
| x_activemask | Tensor | 1维或2维，参见约束说明 | Yes | No | ND | 表示token是否参与通信 |
| expert_scales | Tensor | 2维，参见约束说明 | Yes | No | ND | 每个token的topK个专家权重 |
| elastic_info | Tensor | 1维，参见约束说明 | Yes | No | ND | 表示RP通信域的动态缩容信息，当某些通信卡因异常而从通信域中剔除，实际参与通信的卡数可从本参数中获取 |
| group_ep | str | 字符串长度在[1, 128) | No | No | - | EP通信域名称，专家并行的通信域 |
| group_tp | str | 字符串，参见约束说明 | Yes | No | - | TP通信域名称，数据并行的通信域 |
| tp_world_size | int | No | No | No | - | TP通信域的size |
| tp_rank_id | int | No | No | No | - | TP域本卡Id |
| expert_shard_type | int | No | No | No | - | 表示共享专家卡分布类型 |
| shared_expert_num | int | No | No | No | - | 表示共享专家数量，一个共享专家可以复制部署到多个卡上 |
| shared_expert_rank_num | int | No | No | No | - | 表示共享专家卡数量 |
| quant_mode | int | No | No | No | - | 表示量化模式，0：非量化，2：动态量化 |
| global_bs | int | No | No | No | - | EP域全局的batch_size大小 |
| expert_token_nums_type | int | No | No | No | - | 输出expertTokenNUms中值的语义类型。0：expert_token_nums输出为每个专家处理token数的前缀和;1:expert_token_nums输出为每个专家处理的token数量 |
| comm_alg | str | No | No | No | - | 表示通信亲和内存布局算法，参见约束说明 |
| zero_expert_num | int | No | No | No | - | 表示零专家的数量 |
| copy_expert_num | int | No | No | No | - | 表示拷贝专家的数量 |
| const_expert_num | int | No | No | No | - | 表示常量专家的数量 |

## 参数说明和约束

- **x**: 本卡发送的Token数据，shape=(Bs, H), Bs为batch_size, H为hidden_size,支持dtype为Float16/BFloat16
- **expert_ids**: 每个Token的topK个专家索引,2D tensor,shape=(Bs, K)
- **ep_world_size**: 在A2机器上，取值支持16/32/64;在A3机器上取值为区间[2, 768]
- **ep_rank_id**: 取值范围为[0, epWorldSize),同一个EP通信域内各卡的ep_rank_id不重复
- **moe_expert_num**: 需要满足moe_expert_num % (ep_world_size -shared_expert_rank_num) = 0.在A2机器上，取值范围(0, 512],且需要满足moe_expert_num / (ep_world_size - shared_expert_rank_num) <= 24.在A3机器上取值范围为(0, 1024]
- **scales**: 每个专家的量化平滑参数，2维tensor,shape=(shared_expert_num + moe_expert_num, H).非量化场景传None，动态量化场景可选择传入有效数据或者None
    - A2机器上，当comm_alg配置为"hierarchy"或者配置HCCL_INTRA_PCIE_ENABLE=1&&HCCL_INTRA_ROCE_ENABLE=0,要求传入None
    - A3机器上无特殊要求
- **x_active_mask**: 在A2机器上当前不支持，传None即可
    - 在A3机器上，可传入1维或者2维tensor。当输入为1维tensor，shape为(Bs);当传入2维tensor,shape为(Bs, K).数据类型为Bool,可选择传入None或有效数据。当输入为1维时,参数为True表示对应Token参与通信，且True必须排到False之前，非法输入示例:[True, False, True].当参数为2维tensor，参数为True表示当前token对应expert_ids参与通信。若当前token对应的K个BOOL值全为False，表示当前token不会参与通信。默认所有token都会参与通信。当每张卡的Bs数量不一致时，所有token必须全部有效
- **expert_scales**: 在A2机器上是2维tensor，shape=(Bs, K).A3机器上，当前不支持，传默认值None
- **elastic_info**:
    - A2机器上当前不支持，传None
    - A3上，可传入None或实际有效数据;传入None表示不使能动态缩容功能;当传入有效数据，要求为1维的(4 + 2 x ep_world_size)tensor。Tensor中前四个数字表示(是否缩容，缩容后实际rank数，缩容后共享专家使用的rank数，缩容后moe专家的个数),后续2 x ep_world_size表示2个rank映射表，缩容后本卡中因部分rank异常而从EP通信域中剔除，第一个Table的映射关系为Table1[epRankId]=localEpRankId或-1，localEpRankId表示新EP通信域中的rank Index，-1表示epRankId这张卡从通信域中被剔除，第二个Table映射关系为Table2[localEpRankId] = epRankId
- **group_ep**: 字符串长度范围为[1, 128)，不能和groupTp相同
- **group_tp**: A2上,当前版本不支持，传空字符.A3上,字符串长度范围为[1, 128)，不能和groupEp相同
- **tp_world_size**: A2上,当前版本不支持,传0即可.A3上,取值范围[0, 2],0和1表示无TP域通信,有TP域通信时仅支持2
- **tp_rank_id**: A2上,当前版本不支持,传0即可.A3上,取值范围[0, 1],同一个TP通信域中各卡的tpRankId不重复.无TP域通信时,传0即可
- **expert_shard_type**: A2上,当前版本不支持,传0即可.A3上,当前仅支持传0,表示共享专家卡排在MoE专家卡前面
- **shared_expert_num**: A2上,当前版本不支持,传0即可.A3上,当前取值范围为[0,4]
- **shared_expert_rank_num**: A2上,当前版本不支持,传0即可.A3上,当前取值范围[0, epWorldSize),为0时需满足sharedExpertNum为0或1,不为0时需满足sharedExpertRankNum % sharedExpertNum = 0
- **quant_mode**: 支持0:非量化，2:动态量化
- **global_bs**: 当每个rank的Bs数一致时,globalBs = Bs x epWorldSize 或 globalBs = 0;当每个rank的Bs数不一致时,globalBs = maxBs x  epWorldSize,其中maxBs表示单卡Bs最大值
- **expert_token_nums_type**: 支持0:expertTokenNums中的输出为每个专家处理的token数的前缀和,1:expertTokenNums中的输出为每个专家处理的token数量
- **comm_alg**: A3机器上当前版本不支持;A2机器上,当前版本支持nullptr,"","fullmesh","hierarchy"四种输入方式.推荐配置"hierarchy"并搭配25.0.RC1.1及以上版本驱动使用
    - nullptr和"": 仅在此场景下,HCCL_INTRA_PCIE_ENABLE和HCCL_INTRA_ROCE_ENABLE配置生效.当HCCL_INTRA_PCIE_ENABLE=1&&HCCL_INTRA_ROCE_ENABLE=0时,调用"hierarchy"算法,否则调用"fullmesh"算法.不推荐使用该方式
    - "fullmesh": token数据直接通过RDMA方式发往topk个目标专家所在的卡
    - "hierarchy": token数据经过跨机、机内两次发送，仅不同server同号卡之间使用RDMA通信，server内使用HCCS通信
- **zero_expert_num**: A2上,当前不支持,传入0即可;A3上,取值范围:[0, MAX_INT32),MAX_INT32 = 2^31 - 1, 合法的零专家的ID的值是[moeExpertNum, moeExpertNum + zeroExpertNum)
- **copy_expert_num**: A2上,当前不支持,传入0即可;A3上,取值范围:[0, MAX_INT32),MAX_INT32 = 2^31 - 1, 合法的拷贝专家的ID的值是[moeExpertNum + zeroExpertNum, moeExpertNum + zeroExpertNum + copyExpertNum)
- **const_expert_num**: A2上,当前不支持,传入0即可;A3上,取值范围:[0, MAX_INT32),MAX_INT32 = 2^31 - 1, 合法的常量专家的ID的值是[moeExpertNum + zeroExpertNum + copyExpertNum, moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum)

## 输出参数

| Name | DType | Shape | Description |
|------|-------|-------|-------------|
| expand_x | Tensor | [max(tp_world_size, 1) * A, H] | 根据expert_ids进行扩展过的token特征 |
| dynamic_scales | Tensor | [A] | 表示计算得到的动态量化参数 |
| assist_info_for_combine | Tensor | [A*128] | 表示给同一专家发送的token个数，对应Combine算子中的assistInfoForCombine |
| expert_token_nums | Tensor | [local_expert_num] | 表示每个专家收到的token个数 |
| ep_recv_counts | Tensor | 1维，shape见约束说明 | 从EP通信域各卡接收的token数,对应Combine算子中的epSendCounts |
| tp_recv_counts | Tensor | 1维，shape见约束说明 | 从TP通信域各卡接收的token数，对应Combine算子中的tpSendCounts |
| expand_scales | Tensor | 1维，shape见约束说明 | 表示本卡输出token的权重，对应Combine算子中的expertScalesOptional |

## 输出约束说明

- **dynamic_scales**: 当quant_mode为2时,才有该输出
- **ep_recv_counts**:
    - A2机器上,要求shape为 (moeExpertNum + 2 x globalBs x K x serverNum, ),前moeExpertNum个数表示从EP通信域各卡接收的token数,2 x globalBs x K x serverNum存储了机间机内做通信前combine可以提前做reduce的token个数和token在通信区中的偏移,globalBs传入0时在此处应当按照Bs x epWorldSize计算
    - A3机器上,要求shape为 (epWorldSize x max(tpWorldSize, 1) x localExpertNum)
- **tp_recv_counts**: A2机器上当前不支持TP域通信.A3上当有TP域通信时,要求是一个1D的Tensor,shape为 (tpWorldSize, )
- **expand_scales**: A2上要求是一个1D的Tensor，shape为 (A).A3上,当前版本不支持该输出

expand_x中A(表示本卡需要分发的最大token数量)大小计算:

- 不使能动态缩容场景时:
    - 对于共享专家，要满足于:$A = Bs x epWorldSize x \frac{sharedExpertNum}{sharedExpertRankNum}$
    - 对于MoE专家，当global_bs为0时,要满足:$A \gt= Bs x epWorldSize x min(localExpertNum, K)$;当global_bs非0时，要满足$A >= globalBs x min(localExpertNum, K)$
- 使能动态缩容场景时:
    - 当global_bs = 0时, $A >= max(Bs x epWorldSize x \frac{sharedExpertNum}{sharedExpertRankNum}, Bs x ep_world_size x min(localExpertNum, K))$
    - 当global_bs非0时, $A >= max(Bs x epWorldSize x \frac{sharedExpertNum}{sharedExpertRankNum}, globalBs x min(localExpertNum, K))$

H:表示是hidden_size隐藏层大小

- A2机器上取值是(0, 7168],且需保证是32的整数倍
- A3机器上取值为[1024, 8192]

Bs:表示为Batch sequence size,本卡最终输出的token数量

- A2机器上取值为(0, 256]
- A3机器上取值为(0, 512]

K: 表示选取的topK个专家,$K \in (0, 16]$并且满足$K \in (0, moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum]$

localExpertNum:本卡专家数量

- 对于共享专家卡， localExpertNum = 1
- 对于MoE专家卡，$localExpertNum = \frac{moeExpertNum}{(epWorldSize - sharedExpertRankNum)}$, localExpertNum > 1,不支持TP域通信

更多详细信息请参考：[aclnnMoeDistributeDispatchV3](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/API/aolapi/context/aclnnMoeDistributeDispatchV3.md)

## 特殊说明

## 使用示例

### 基本使用示例

```python
import mindspore as ms
import numpy as np
import ms_custom_ops
from mindspore.communication import init, get_rank, create_group

class MoeDistributeDispatchNet(ms.nn.Cell):

    def __init__(self,
                 global_gs=0,
                 moe_expert_num=16,
                 ep_world_size=16,
                 group_ep="ep"):
        super().__init__()
        self.moe_distribute_dispatch = ms_custom_ops.moe_distribute_dispatch_v3
        self.rank = get_rank()
        self.moe_expert_num = moe_expert_num
        self.ep_world_size = ep_world_size
        self.group_ep = group_ep
        self.ep_rank_id = get_rank(group=group_ep)
        self.global_gs = global_gs

    @jit
    def construct(self, x, expert_ids, scales=None):
        out = self.moe_distribute_dispatch(x=x,
                                           expert_ids=expert_ids,
                                           group_ep=self.group_ep,
                                           ep_world_size=self.ep_world_size,
                                           ep_rank_id=self.ep_rank_id,
                                           moe_expert_num=self.moe_expert_num,
                                           scales=scales,
                                           global_bs=self.global_gs)
        return out

mode = ms.PYNATIVE_MODE #or ms.GRAPH_MODE
batch_size = 8
k = 8
hidden_size = 7168
moe_expert_num = 256
ep_world_size = 8
global_bs =  0
np.ranom.seed(1)
ms.set_context(mode)
init()
rand_id = get_rank()

ep_rank_list = [list(range(ep_world_size))]
for ep_rank in ep_rank_list:
    if rand_id in ep_rank:
        group_ep = "ep-" + "-".join([str(i) for i in ep_rank])
        create_group(group_ep, ep_rank)
x = np.random.uniform(low=-5, high=5, size=(bs, h)).astype(np.float16)
expert_ids = np.argsort(np.random.randn(bs, moe_expert_num),
                            axis=1)[:, :k].astype(np.int32)
expert_scales = np.random.randn(bs, k).astype(np.float32)
ms_x = ms.Tensor(x)
ms_expert_ids = ms.Tensor(expert_ids)
ms_expert_scales = ms.Tensor(expert_scales)
dispatch = MoeDistributeDispatchNet(global_gs=global_bs,
                                        moe_expert_num=moe_expert_num,
                                        ep_world_size=ep_world_size,
                                        group_ep=group_ep)
expand_x, _, assist_info_for_combine, _, ep_recv_counts, tp_recv_counts, _ = dispatch(
        ms_x, ms_expert_ids)
print(expand_x)
```
