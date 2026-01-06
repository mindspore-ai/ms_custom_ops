# PagedAttention

## 描述

PagedAttention 是针对增量推理/分页 KV-Cache 的注意力算子，支持未量化、反量化融合、QKV 全量化（Offline/Online），多 Token 推理（MTP）、多种掩码、不同输入布局与 ND/NZ 输入格式。

## 支持产品

| 硬件型号                            | 是否支持 |
|-------------------------------------|----------|
| Atlas A3 推理系列产品/Atlas A3 训练系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |
| Atlas 训练系列产品                    | √        |
| Atlas 推理系列产品                    | √        |
| Atlas 200I/500 A2 推理产品          | x        |

## 接口与输入输出

### 名称

- 算子名：`paged_attention`

### 输入参数

| Name               | DType                                 | Shape                                   | Optional (Default) | Format     | Description |
|--------------------|---------------------------------------|-----------------------------------------|--------------------|------------|-------------|
| query              | float16/bfloat16 或 int8              | ND: [num_tokens, q_head_num, head_size]；Atlas 推理系列产品/NZ: [num_tokens, q_head_num*head_size] | No | ND/NZ | 查询张量；全量化时为 int8。|
| key_cache          | float16/bfloat16 或 int8              | ND: [num_blocks, block_size, kv_head_num, head_size_k]；NZ: [num_blocks, block_size, kv_head_num*head_size_k]| No | ND/NZ | 分页 KV 的 K 缓存；量化场景为 int8。 |
| value_cache        | float16/bfloat16 或 int8              | ND: [num_blocks, block_size, kv_head_num, head_size_v]；NZ: [num_blocks, block_size, kv_head_num*head_size_v]     | No                 | ND/NZ      | 分页 KV 的 V 缓存；量化场景为 int8。 |
| block_tables       | int32                                  | [num_tokens, max_num_blocks_per_query]  | No                 | ND         | 每个 token 对应的 block 索引表。 |
| context_lens       | int32                                  | [batch]                                  | No                 | ND | 每 batch 的 KV token 数。Atlas A2 训练系列产品/Atlas A2 推理系列产品和Atlas A3 推理系列产品/Atlas A3 训练系列产品：作为 param 从 CPU 读取；Atlas 推理系列产品：保持为 NPU 张量输入。 |
| attn_mask          | float16/bfloat16             | 较复杂，参考用例 | Yes (None)        | ND/NZ      | 注意力掩码；不同 `mask_type` 对应不同构造。 |
| batch_run_status   | int32                                  | [batch]                                  | Yes (None)         | ND         | 控制可计算 batch 的标志；需与 `batch_run_status_enable` 配合。 |
| k_descale          | float32 或 int64        | [kv_head_num*head_size]    | Yes (None)         | ND         | 反量化融合/全量化。 |
| k_offset           | int32                   | [kv_head_num*head_size]    | Yes (None)         | ND         | 非对称反量化偏移，`has_quant_offset=True` 时使用。 |
| v_descale          | float32 或 int64        | [kv_head_num*head_size]    | Yes (None)         | ND         | 反量化融合/全量化步长。 |
| v_offset           | int32                   | [kv_head_num*head_size]    | Yes (None)         | ND         | 非对称反量化偏移，`has_quant_offset=True` 时使用。 |
| razor_offset       | float32                 | [num_blocks, block_size]   | Yes (None)       | ND     | Razor Rope 场景偏移。 |
| p_scale            | float32                 | [q_head_num]               | Yes (None)         | ND         | 离线全量化时传入 P 矩阵量化 scale。 |
| log_n              | float32                 | [batch]                    | Yes (None)         | ND         | `scale_type=LOGN` 时为各 batch 缩放系数。 |
| q_seq_lens         | int32                   | [batch]                    | Cond (None)        | ND（CPU）  | 并行解码/MTP（`calc_type=1`）时必需，始终按 param 绑定（CPU）。 |
| q_head_num         | int                                    | -                                        | Yes (0)            | -          | Q 头数；不允许为 0。 |
| qk_scale           | float                                  | -                                        | Yes (1.0)          | -          | QK 缩放（TOR）。 |
| kv_head_num        | int                                    | -                                        | Yes (0)            | -          | KV 头数。 |
| mask_type          | int                                    | -                                        | Yes (0)            | -          | 掩码类型，见下文枚举。 |
| batch_run_status_enable | bool                              | -                                        | Yes (False)        | -          | 是否启用 `batch_run_status`。 |
| quant_type         | int                                    | -                                        | Yes (0)            | -          | 量化类型，见下文枚举。 |
| out_data_type      | int                                    | -                                        | Yes (-1)           | -          | 全量化输出类型：1=fp16，27=bf16。 |
| has_quant_offset   | bool                                   | -                                        | Yes (False)        | -          | 是否使用非对称反量化偏置。 |
| compress_type      | int                                    | -                                        | Yes (0)            | -          | 压缩类型，见下文枚举。 |
| calc_type          | int                                    | -                                        | Yes (0)            | -          | 计算模式（MTP），见下文枚举。 |
| scale_type         | int                                    | -                                        | Yes (0)            | -          | 缩放类型（TOR/LOGN）。 |
| input_layout       | int                                    | -                                        | Yes (0)            | -          | 输入布局：0=BSND，1=BNSD。 |
| mla_v_dim          | int                                    | -                                        | Yes (0)            | -          | MLA 输出 head 维度（0 关闭）。 |
| input_format       | int                                    | -                                        | Yes (0)            | -          | 0=ND（Atlas A2 训练系列产品/Atlas A2 推理系列产品和Atlas A3 推理系列产品/Atlas A3 训练系列产品），1=NZ（Atlas 推理系列产品）。 |

说明：`context_lens` 与 `q_seq_lens` 虽为输入张量，但以 param 方式绑定到 host；在 Atlas 推理系列产品/NZ 路径下 `context_lens` 保持为 NPU 输入张量，而 `q_seq_lens` 始终通过 CPU 侧 param 传入。

### 输出参数

| Name           | DType                                   | Shape                              | Description |
|----------------|-----------------------------------------|------------------------------------|-------------|
| attention_out  | float16/bfloat16（或由 `out_data_type` 指定） | [num_tokens, q_head_num, head_size_out] | 注意力输出；当 `mla_v_dim>0` 时 `head_size_out=mla_v_dim`，否则与输入 `head_size` 一致。 |

## 参数含义与枚举值

- mask_type（掩码类型）
    - 0：PA_MASK_UNDEFINED
    - 1：PA_MASK_TYPE_NORM（倒三角mask）
    - 2：PA_MASK_TYPE_ALIBI
    - 3：PA_MASK_TYPE_SPEC（并行解码mask）
    - 4：PA_MASK_TYPE_MASK_FREE（仅 fp16 支持）
- quant_type（量化类型）
    - 0：UNQUANT/UNDEFINED（未量化）
    - 1：DEQUANT_FUSION（KV int8 反量化融合）
    - 2：QUANT_QKV_OFFLINE（全量化离线）
    - 3：QUANT_QKV_ONLINE（全量化在线）
- out_data_type（全量化输出类型）
    - -1：未指定（默认）
    - 1：float16
    - 27：bfloat16
- compress_type（压缩类型）
    - 0：UNDEFINED
    - 1：KVHEAD
    - 2：KVHEAD_ROPE
    - 3：MAX（非法）
- calc_type（计算模式）
    - 0：UNDEFINED（常规单 token 解码）
    - 1：SPEC（MTP，多 Token 推理；需提供 `q_seq_lens`）
- scale_type（缩放类型）
    - 0：TOR
    - 1：LOGN
    - 2：MAX
- input_layout（输入布局）
    - 0：BSND
    - 1：BNSD
- input_format（输入格式）
    - 0：ND（Atlas A2 训练系列产品/Atlas A2 推理系列产品和Atlas A3 推理系列产品/Atlas A3 训练系列产品 默认）
    - 1：NZ（Atlas 推理系列产品，需对 Q/K/V/Mask 使用 `trans_data(..., 1)` 转换）

更多关于输入/输出约束说明，请参考 ATB 文档：[PagedAttentionOperation 输入输出列表](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/ascendtbapi/ascendtb_01_0197.html)。

## Python 使用示例

```python
import numpy as np
from mindspore import Tensor, context, ops
import ms_custom_ops

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

batch_size = 4
head_num = 32
kv_head_num = 32
head_dim = 128
block_size = 128

kv_seq_lens = [192, 193, 194, 195]
q_seq_lens = [1] * batch_size

num_tokens = sum(q_seq_lens)
max_kv_len = max(kv_seq_lens)
max_blocks_per_query = (max_kv_len + block_size - 1) // block_size
num_blocks = batch_size * max_blocks_per_query

query = Tensor(np.random.randn(num_tokens, head_num, head_dim).astype(np.float16))
key_cache = Tensor(np.random.randn(num_blocks, block_size, kv_head_num, head_dim).astype(np.float16))
value_cache = Tensor(np.random.randn(num_blocks, block_size, kv_head_num, head_dim).astype(np.float16))
block_tables_np = np.stack([
    np.arange(i * max_blocks_per_query, (i + 1) * max_blocks_per_query, dtype=np.int32)
    for i in range(batch_size)
])
block_tables = Tensor(block_tables_np)
context_lens = Tensor(np.array(kv_seq_lens, dtype=np.int32))

context_lens_cpu = ops.move_to(context_lens, "CPU")

qk_scale = float(1.0 / np.sqrt(head_dim))
out = ms_custom_ops.paged_attention(
    query, key_cache, value_cache, block_tables, context_lens_cpu,
    attn_mask=None, q_seq_lens=None,
    q_head_num=head_num, qk_scale=qk_scale, kv_head_num=kv_head_num,
    mask_type=0, batch_run_status_enable=False,
    quant_type=0, out_data_type=-1, has_quant_offset=False,
    compress_type=0, calc_type=0, scale_type=0,
    input_layout=0, mla_v_dim=0, input_format=0
)
print(out.shape)  # (num_tokens, head_num, head_dim)
```
