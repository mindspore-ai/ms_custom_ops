# FlashAttention Encoder 算子

## 描述

FlashAttention 是一种高性能的自注意力实现，主要通过分块/重计算、mask压缩与更优的访存/算子融合来降低显存占用和提升吞吐。本项目中的 `flash_attention_encoder` 为自定义算子接口，统一对接 AscendTransformerBoost 的 SelfAttentionOperation 系列内核。

参考（ATB 文档，SelfAttentionOperation 参数与特性说明）：

- [SelfAttentionOperation 参数列表](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendtbapi/ascendtb_01_0278.html)

## 接口与输入输出

### 名称

- 算子名：`flash_attention_encoder`

### 输入参数

| Name            | DType                     | Shape                     | Optional | Format | Description |
|-----------------|---------------------------|---------------------------|----------|--------|-------------|
| query           | Tensor[float16/bfloat16]  | TH/TND                    | No       | ND/NZ | 查询向量 Q |
| key             | Tensor[float16/bfloat16]  | 同 query                  | No       | ND/NZ | 键向量 K |
| value           | Tensor[float16/bfloat16]  | 同 query                  | No       | ND/NZ | 值向量 V |
| layer_id        | Tensor[int32]             | (1,) 或按实现要求         | Yes      | ND    | 层索引（保留，可选） |
| mask            | Tensor[float16/bfloat16]  | 见不同 mask 类型要求      | Yes      | ND/NZ | 注意力掩码，支持 NORM/ALIBI/SWA 等 |
| alibi_coeff     | Tensor[float32]           | 见 ALIBI 需求             | Yes      | ND    | ALIBI 相关（可选） |
| deq_scale_qk    | Tensor[float32]           | (head_num,)               | Yes      | ND    | 量化相关（保留） |
| deq_offset_qk   | Tensor[float32]           | (head_num,)               | Yes      | ND    | 量化相关（保留） |
| deq_scale_pv    | Tensor[float32]           | (head_num,)               | Yes      | ND    | 量化相关（保留） |
| deq_offset_pv   | Tensor[float32]           | (head_num,)               | Yes      | ND    | 量化相关（保留） |
| quant_p         | Tensor[float32]           | 按需                      | Yes      | ND    | 量化相关（保留） |
| logN            | Tensor[float32]           | (1,) 或按实现要求         | Yes      | ND    | 预留（本版未使用） |
| q_seq_len       | Tensor[int32]             | (batch,)                  | Cond     | ND    | TH/TND 布局必需：每 batch 的 Q 序列长度 |
| kv_seq_len      | Tensor[int32]             | (batch,)                  | Cond     | ND    | TH/TND 布局必需：每 batch 的 KV 序列长度 |
| head_num        | int                       | -                         | Yes      | -     | 注意力头数（H），默认 0（需显式设置 >0） |
| scale_value     | float                     | -                         | Yes      | -     | QK 缩放系数 `qk_scale`（通常为 1/sqrt(head_dim)），默认 1.0 |
| kv_head_num     | int                       | -                         | Yes      | -     | KV 头数（GQA 场景），默认 0 表示与 head_num 对齐 |
| mask_type       | int                       | -                         | Yes      | -     | 掩码类型，默认 0（UNDEFINED）；详见下文“参数补充说明” |
| kernel_type     | int                       | -                         | Yes      | -     | 内核精度，默认 0：半精度；1：高精度（FP32 BMM1） |
| window_size     | int                       | -                         | Yes      | -     | SWA 窗口大小，默认 0（关闭 SWA） |
| cache_type      | int                       | -                         | Yes      | -     | 缓存类型，默认 0：NORM；1：SWA（SWA 优化） |
| input_format    | int                       | -                         | Yes      | -     | 输入格式选择，默认 0：ND；1：NZ |

注：当前版本未接线量化/online-offline QKV/clamp/ring/prefix 等高级特性，相关张量为占位，可保持为 None。

#### 产品支持情况

| 硬件型号                               | 是否支持 | 特殊说明                                                                |
| ---------------------------------- | ---- | ------------------------------------------------------------------- |
| Atlas A3 推理系列产品/Atlas A3 训练系列产品    | √    | -                                                                  |
| Atlas 800I A2 推理产品/Atlas A2 训练系列产品 | √    | -                                                                  |
| Atlas 训练系列产品                       | √    | 部分场景支持，BNSD维度输入 |
| Atlas 推理系列产品                       | √    | 部分场景支持，BNSD维度输入，高精度，压缩mask |
| Atlas 200I/500 A2 推理产品             | ×    | -                                                                  |

更多详情请参考 [SelfAttentionOperation 产品支持情况](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendtbapi/ascendtb_01_0272.html)。

#### 参数补充说明

- mask_type（按枚举数值）
    - 默认值：0。
    - 取值：
    - 0：MASK_TYPE_UNDEFINED（全 0 mask）
    - 1：MASK_TYPE_NORM（倒三角/因果 mask）
    - 2：MASK_TYPE_ALIBI（ALIBI mask）
    - 3：MASK_TYPE_NORM_COMPRESS（倒三角压缩）
    - 4：MASK_TYPE_ALIBI_COMPRESS（ALIBI 压缩）
    - 5：MASK_TYPE_ALIBI_COMPRESS_SQRT（ALIBI 压缩开平方）
    - 6：MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN（ALIBI 压缩左对齐，平台限定）
    - 7：MASK_TYPE_SLIDING_WINDOW_NORM（SWA 常规）
    - 8：MASK_TYPE_SLIDING_WINDOW_COMPRESS（SWA 压缩）
    - 9：MASK_TYPE_CAUSAL_MASK（内部生成的因果 mask）

- q_seq_len / kv_seq_len
    - 默认值：None（TH/TND 场景必传，且需同时提供；类型 `int32` 且在 CPU）。

### 输出参数

| Name          | DType                    | Shape           | Description |
|---------------|--------------------------|-----------------|-------------|
| attention_out | Tensor[float16/bfloat16] | 与 query 对齐   | 注意力输出   |

### Python 使用示例

```python
import numpy as np
from mindspore import context, Tensor
import ms_custom_ops

np.random.seed(0)
context.set_context(device_target="Ascend", mode=context.PYNATIVE_MODE)

head_num = 8
kv_head_num = 4
head_dim = 128
scale_value = float(1.0 / np.sqrt(head_dim))

q_seq = np.array([16, 20], dtype=np.int32)
kv_seq = np.array([16, 20], dtype=np.int32)
q_tokens = int(q_seq.sum())
kv_tokens = int(kv_seq.sum())

q = Tensor(np.random.uniform(-1, 1, size=(q_tokens, head_num, head_dim)).astype(np.float16))
k = Tensor(np.random.uniform(-1, 1, size=(kv_tokens, kv_head_num, head_dim)).astype(np.float16))
v = Tensor(np.random.uniform(-1, 1, size=(kv_tokens, kv_head_num, head_dim)).astype(np.float16))

q_seq_len = Tensor(q_seq).move_to("CPU")
kv_seq_len = Tensor(kv_seq).move_to("CPU")

out = ms_custom_ops.flash_attention_encoder(
    q, k, v,
    layer_id=None, mask=None, alibi_coeff=None,
    deq_scale_qk=None, deq_offset_qk=None, deq_scale_pv=None, deq_offset_pv=None,
    quant_p=None, logN=None,
    q_seq_len=q_seq_len, kv_seq_len=kv_seq_len,
    head_num=head_num, scale_value=scale_value, kv_head_num=kv_head_num,
    kernel_type=0, mask_type=0, window_size=0, cache_type=0
)

print(out.shape)  # 期望: (q_tokens, head_num, head_dim)
```
