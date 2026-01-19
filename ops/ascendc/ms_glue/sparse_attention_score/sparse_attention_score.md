# SparseAttentionScore

## 描述

- 算子功能：推理场景下，使用SparseAttention算法实现selected-attention（选择注意力）的计算。

- 计算公式：

    注意力的正向计算公式如下：

  $$
  selected\_key = Gather(key, selected\_indices[i]),0<=i<selected\_block\_count \\
  selected\_value = Gather(value, selected\_indices[i]),0<=i<selected\_block\_count
  $$
  $$
  attention\_out = Softmax(Mask(scale * (query @ selected\_key^T), atten\_mask)) @ selected\_value
  $$

- 整体计算流程：

1. 对于当前的block块，利用sparse_attention_mask与slect_block_id，判断该block是否需要被计算：是，进入步骤2；否，进入下一个block，回到步骤1；

2. query与转置后的key做matmul计算后得到最初步的attention_score，然后与位置编码pse相加后再乘以缩放系数scale_value。此时的结果通过atten_mask进行select操作，将atten_mask中为true的位置进行遮蔽，得到结果masked_attention_score，即atten_mask中为true的位置在select后结果为负的极小值，经过softmax计算之后变成0从而达到遮蔽效果。

3. 为了实现SparseAttention加速算法，使用sparse_softmax操作对masked_attention_score进行运算，用以代替原公式中的softmax运算，而后将结果与value做matmul运算。由于sparse_softmax操作对masked_attention_score的Skv(输入key、value的sequence length)方向进行了切分，故实现过程中存在一个刷新流程，具体如下：
    1. 每次FlashSoftmax计算只对切分后的一个SkvSplit（SkvSplit是针对Skv轴进行切分之后的序列长度的简称）进行操作，并从第二次循环开始记录exp，其中 i 表示Skv切分后的循环变量，针对exp的i是从1开始 ，exp的计算公式如下：
       $$
       exp[i] = e^{max_{i - 1} - max_{i}}
       $$
    2. 当i = 0时，计算出的MM[PV]结果直接保存到ub_attention_out[0]的ub中。
    3. 从i = 1开始，需要增加Mul和Add操作，即将上一次的MM[PV]的结果和当前exp相乘，相乘完的结果和本次MM[PV]的结果相加得到的结果保存到ub_attention_out[1]的ub中。以此类推，遍历Skv计算完成。
    4. 由于FlashSoftmax计算中的除sum被后移到输出attention_out之前，因此最后需要将ub中的ub_attention_out按行除以softmax_sum并将最终完整的结果保存到输出内存attention_out(Final)上。

## 支持产品

| 硬件型号                                     | 是否支持 |
|---------------------------------------------|----------|
| Atlas A3 推理系列产品/Atlas A3 训练系列产品   | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品   | √        |
| Atlas 训练系列产品                            | √        |
| Atlas 推理系列产品                            | √        |

## 接口与输入输出

### 名称

- 算子名：`sparse_attention_score`

### 输入参数与输出参数

| Name                   | DType                             | Shape                                                                                                                   | Optional | Inplace | Format | Description                    |
|------------------------|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------|----------|---------|--------|--------------------------------|
| query                    | Tensor[float16/bfloat16/float32]     | TH/NPD                                                                                        | No       | No      | ND     | query 张量                       |
| key                    | Tensor[float16/bfloat16/float32]     | TH/NPD                                                                                       | No       | No      | ND     | key 张量                       |
| value                  | Tensor[float16/bfloat16/float32]     | TH/NPD                                                                                        | No      | No      | ND     | value 张量                     |
| real_shift_optional              | Tensor[float]     | - | Yes       | No     | - | **预留参数，暂未使用**                 |
| drop_mask_optional            | Tensor[float]     | - | Yes      | No     | - | **预留参数，暂未使用**               |
| padding_mask_optional           | Tensor[float]                      |       -                                                                                                    | Yes       | No      |  - | **预留参数，暂未使用**             |
| atten_mask_optional             | Tensor[bool/uint8]                                 | [B,N,S,S\]、\[B,1,S,S\]、\[1,1,S,S\]、\[S,S\]                                                                                                               | Yes       | No     | ND     | 取值为1代表该位不参与计算（不生效），为0代表该位参与计算 |
| prefix_optional               | Tensor[int32]                                  | -                                                                                                                       | Yes      | No      | -      | **预留参数，暂未使用**              |
| select_block_idx_optional               | Tensor[int32]                                | [B, N_kv, S_q, selected_block_count]                                                                                                                     | Yes      | No       | ND     | 表示所选数据块block的索引（在该行block中的序号，$0<=idx<(Skv/select_block_en)$），则该block的offset=id*select_block_len             |
| scale_value_optional               | float                               | -                                                                                                                       | Yes      | No       | -      | 代表缩放系数，作为计算流中Muls的scalar值，一般设置为D^-0.5             |
| keep_prob_optional               | float                                | -                                                                                                                       | Yes      | No      | -      | 代表dropMaskOptional中1的比例             |
| pre_tokens_optional               | int                                | -                                                                                                                       | Yes      | No       | -      | 用于稀疏计算 ，表示slides window的左边界             |
| next_tokens_optional               | int                                | -                                                                                                                       | Yes      | No      | -      | 用于稀疏计算，表示slides window的右边界            |
| head_num               | int                                | -                                                                                                                       | Yes      | No       | -      | 代表单卡的head个数，即输入query的N轴长度              |
| input_layout               | string                                | BSH、SBH、BSND、BNSD                                                                                                                      | Yes      | No       | -      | 代表输入query、key、value的数据排布格式              |
| inner_precise_optional               | int                                | -                                                                                                                       | Yes      | No      | -      | 用于提升精度，默认配置为0即可              |
| sparse_mode_optional               | int                                | -                                                                                                                       | Yes      | No      | -      | 表示sparse的模式，默认配置为0即可，支持配置值为0、1、2、3、4、5、6。当整网的sparse_mode_optional都相同且shape小于2048\*2048时，建议使用defaultMask模式，来减少内存使用量            |
| select_block_len_optional               | int                                | -                                                                                                                       | Yes      | No       | -      | 表示select的每个block长度。不输入时，则select_block_len_optional也不会起作用，代表非稀疏运算            |
| softmax_max_out               | Tensor[float]                                | [B,N,Sq,8]                                                                                                                    | No      | Yes       | ND     | Softmax计算的Max中间结果，用于反向计算             |
| softmax_sum_out               | Tensor[float]                                | [B,N,Sq,8]                                                                                                                    | No      | Yes       | ND     | Softmax计算的Sum中间结果             |
| softmax_out_out               | -                              | -                                                                                                                    | No      | Yes       | -     |   **预留参数，暂未使用**          |
| attention_out_out               | Tensor[float16/bfloat16/float32]                                | [B,N,Sq,D]                                                                                                                    | No      | Yes       | ND     | 计算公式的最终输出            |

### 约束与限制

- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- 输入query、key、value的B：batchsize必须相等。
- 输入query、key、value的D：Head-Dim必须相等。
- 输入query、key、value的input_layout必须一致。
- 输入query、key、value、real_shift_optional的数据类型必须一致。
- 输入key/value的shape必须一致。
- 支持输入query的N和key/value的N不相等，但必须成比例关系，即Nq/Nkv必须是非0整数，Nq取值范围1~256。当Nq/Nkv > 1时，即为GQA(grouped-query attention)；当Nkv=1时，即为MQA(multi-query attention)。本文如无特殊说明，N表示的是Nq。
- 关于数据shape的约束，以input_layout的BSND、BNSD为例（BSH、SBH下H=N\*D），其中：
    - B：取值范围为1\~2M。带prefix_optional的时候B最大支持2K。
    - N：取值范围为1\~256。
    - S：取值范围为1\~1M。
    - D：取值范围为1\~512。
- 部分场景下，如果计算量过大可能会导致算子执行超时，此时建议做轴切分处理，注：这里的计算量会受B、S、N、D等参数的影响，值越大计算量越大。
- keep_prob_optional的取值范围为(0, 1]。
- prefix_optional稀疏计算场景即sparse_mode_optional=5或者sparse_mode_optional=6，当Sq > Skv时，prefix的N值取值范围\[0, Skv\]，当Sq <= Skv时，prefix的N值取值范围\[Skv-Sq, Skv\]。
- band场景，pre_tokens_optional和next_tokens_optional之间必须要有交集。
- sparse_mode_optional配置为1、2、3、5、6时，用户配置的pre_tokens_optional、next_tokens_optional不会生效；sparse_mode_optional配置为0、4时，须保证atten_mask_optional与pre_tokens_optional、next_tokens_optional的范围一致。
- sparse_mode_optional为1、2、3、4、5、6时，应传入对应正确的atten_mask_optional，否则将导致计算结果错误。当atten_mask_optional输入为None时，sparse_mode_optional、pre_tokens_optional、next_tokens_optional参数不生效，固定为全计算。
- real_shift_optional Sq大于1024时如果配置BNHS、1NHS，需要Sq和Skv等长。
- block_size最小可选[64,128], 建议选[128, 512]或[128, 256]

## Python 使用示例

```python
import mindspore
import ms_custom_ops

context.set_context(device_target="Ascend", mode=context.PYNATIVE_MODE)

batch_size = 1,
num_heads = 14,
head_size = 128,
seq_len = 131072,
scale = None,
dtype = mindspore.bfloat16,
epsilon = 0.1,
block_height = 128
block_width = 256
num_kv_heads = num_heads
scale = scale if scale is not None else float(1.0 / (head_size ** (1/2)))

query = mindspore.mint.rand([batch_size, num_heads, seq_len, head_size], dtype=dtype)
key = mindspore.mint.rand([batch_size, num_kv_heads, seq_len, head_size], dtype=dtype)
value = mindspore.mint.rand([batch_size, num_kv_heads, seq_len, head_size], dtype=dtype)
score = mindspore.mint.zeros([batch_size, num_heads, seq_len, head_size], dtype=dtype)

mask = mindspore.mint.eq(
        mindspore.mint.tril(mindspore.mint.ones([2048, 2048], dtype=mindspore.uint8)),
        mindspore.mint.zeros([2048, 2048], dtype=mindspore.uint8))

blocks = make_block_indices(
        batch_size=batch_size, num_kv_heads=num_kv_heads, seq_len=seq_len, block_height=block_height,
        block_width=block_width)

real_shift_optional = mindspore.mint.rand([1], dtype=mindspore.float)
drop_mask_optional = mindspore.mint.rand([1], dtype=mindspore.float)
padding_mask_optional = mindspore.mint.rand([1], dtype=mindspore.float)
atten_mask_optional = mask
prefix_optional = mindspore.mint.rand([1], dtype=mindspore.int32)
select_block_idx_optional = blocks.to(mindspore.int32)
scale_value_optional = scale
keep_prob_optional = float(1.0)
pre_tokens_optional = int(2147483647)
next_tokens_optional = int(2147483647)
head_num = num_heads
input_layout = "BNSD"
inner_precise_optional = int(0)
sparse_mode_optional = int(2)
select_block_len_optional = int(block_width)
softmax_max_out = mindspore.mint.zeros([batch_size, num_heads, seq_len, 8], dtype=mindspore.float)
softmax_sum_out = mindspore.mint.zeros([batch_size, num_heads, seq_len, 8], dtype=mindspore.float)
softmax_out_out = mindspore.mint.rand([1], dtype=mindspore.float)
attention_out_out = score

ms_custom_ops.sparse_attention_score(
        query,
        key,
        value,
        real_shift_optional,
        drop_mask_optional,
        padding_mask_optional,
        atten_mask_optional,
        prefix_optional,
        select_block_idx_optional,
        scale_value_optional,
        keep_prob_optional,
        pre_tokens_optional,
        next_tokens_optional,
        head_num,
        input_layout,
        inner_precise_optional,
        sparse_mode_optional,
        select_block_len_optional,
        softmax_max_out,
        softmax_sum_out,
        softmax_out_out,
        attention_out_out)
```
