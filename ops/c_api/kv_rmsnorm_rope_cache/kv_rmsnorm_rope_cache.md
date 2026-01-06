# kv_rmsnorm_rope_cache算子

## 描述

kv_rmsnorm_rope_cache算子用于对输入张量(kv)的尾轴，拆分出左半边用于rms_norm计算，右半边用于rope计算，再将计算结果分别scatter到两块cache中。该算子底层调用的是aclnnKvRmsNormRopeCache算子。

## 输入参数

| Name                | DType           | Shape                                  | Optional | Inplace | Format | Description                                            |
|---------------------|-----------------|----------------------------------------|----------|---------|--------|--------------------------------------------------------|
| kv               | Tensor          | 4维[B_kv, N, S_kv, D] | No       | No      | ND     | 用于切分出RMSNorm计算所需数据Dv和RoPE计算所需数据Dk的输入数据                              |
| gamma                 | Tensor          | 1维[D_v] | No       | No      | ND     | 用于RMSNorm计算的输入数据                              |  
| cos                 | Tensor          | 4维[Bkv, 1, Skv, Dk] 或 [Bkv, 1, 1, Dk]         | No       | No      | ND     | 用于计算Rope的余弦变换数据                                |
| sin                 | Tensor          | 4维[Bkv, 1, Skv, Dk] 或 [Bkv, 1, 1, Dk]          | No       | No      | ND     | 表用于计算Rope的正弦变换数据                                |
| index              | Tensor          | 多维，参见约束说明                                 | No      | No      | ND | 用于指定写入cache的具体索引位置，value为-1时跳过更新                                 |
| k_cache         | Tensor          |    4维，参见约束说明                                  | No      | Yes      | ND | 提前申请的Cache                              |
| c_kv_cache         | Tensor          | 4维，参见约束说明                                  | No      | Yes      | ND | 提前申请的Cache                              |
| k_rope_scale         | Tensor          | 多维，参见约束说明                               | Yes      | No      | ND | k_cache为int8时量化参数                              |
| c_kv_scale         | Tensor          | 多维，参见约束说明                                   | Yes      | No      | ND | 提c_kv_cache int8时量化参数                              |
| k_rope_offet         | Tensor          | 多维，参见约束说明                                 | Yes      | No      | ND | k_cache为int8时量化参数且非对称量化时的偏移参数                              |
| c_kv_offset_         | Tensor          | 多维，参见约束说明                                 | Yes      | No      | ND | c_kv_cache为int8时量化参数且非对称量化时的偏移参数                              |
| epsilon         | float          | No                                     | No      | No      | - | Rmsnorm进行计算时防除零的epsilon值                              |
| cache_mode         | int          | No                                     | No      | No      | - | 提cache格式选择枚举，枚举含义详见约束说明                              |
| is_output_kv         | bool          | No                                     | No      | No      | - | 控制是否输出k_rope_out和c_kv_out的标志位                              |

Note:
形状约束

+ ​kv输入: shape为[Bkv, N, Skv, D]，其中D = Dk + Dv

+ ​gamma输入: shape为[Dv,]

+ ​cos/sin输入: shape为[Bkv, 1, Skv, Dk] 或 [Bkv, 1, 1, Dk]，必须与cos保持一致

​index输入:

+ cacheMode为Norm时: shape为2维[Bkv, Skv]

+ cacheMode为PA_BNSD/PA_NZ时: shape为1维[Bkv * Skv]

+ cacheMode为PA_BLK_BSND/PA_BLK_NZ时: shape为1维[Bkv * ceil_div(Skv, BlockSize)]

数据类型约束

+ ​kv/gamma/cos/sin: 支持FLOAT16、BFLOAT16

+ ​index: 支持INT64

+ ​cache: 支持与输入kv相同的数据类型或INT8

数值约束

+ ​N值: 仅支持N=1（与DeepSeekV3网络结构强相关）

+ ​Dk值: 必须为偶数（满足RoPE规则）

​对齐要求:

+ NZ场景下Dk、Dv需32B对齐

+ PA场景下BlockSize需32B对齐

+ 不同数据类型对齐要求不同（INT8需32B对齐，FLOAT16需16B对齐）

index约束

+ ​Norm模式: value范围[-1, Scache)，不同Bkv下value可重复

+ ​PA_BNSD/PA_NZ模式: value范围[-1, BlockNum * BlockSize)，value不能重复

+ ​PA_BLK_BSND/PA_BLK_NZ模式: value范围[-1, BlockNum * BlockSize)，value/BlockSize的值不能重复

cache_mode枚举值约束:

+ "Norm" => 0

+ "PA" => 1

+ "PA_BNSD" => 2

+ "PA_NZ" => 3

+ "PA_BLK_BNSD" => 4

+ "PA_BLK_NZ" => 5

+ 其余枚举值输入无效，会被拦截;

## 输出参数

| Name   | DType      | Shape      | Description           |
|--------|------------|------------|-----------------------|
| k_rope_out| Tensor   | [B_kv, N, S_kv, D_k] | Rope计算的结果(is_output_kv为True时输出) |
| c_kv_out | Tensor    | [B_kv, N, S_kv, D_v] | Rmsnorm计算的结果(is_output_kv为True时输出) |

k_rope_out数据类型和kv相同。
c_kv_out数据类型和kv相同。

更多详细信息请参考：[aclnnKvRmsNormRopeCache](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/API/aolapi/context/aclnnKvRmsNormRopeCache.md)

## 特殊说明

## 使用示例

### 基本使用示例

```python
import mindspore as ms
import numpy as np
import ms_custom_ops

ms.set_device("Ascend")

@ms.jit
def kv_rmsnorm_rope_cache_func(kv,
        gamma,
        cos,
        sin,
        index,
        kCacheRef,
        ckvCacheRef,
        kRopeScale=None,
        ckvScale=None,
        kRopeOffset=None,
        cKvOffset=None,
        epsilon=1e-5,
        cache_mode=0,
        is_output_kv=False,
    ):
    return ms_custom_ops.kv_rmsnorm_rope_cache(
        kv,
        gamma,
        cos,
        sin,
        index,
        kCacheRef,
        ckvCacheRef,
        kRopeScale,
        ckvScale,
        kRopeOffset,
        cKvOffset,
        epsilon,
        cache_mode,
        is_output_kv,
    )

def to_tensor(arr, dtype=None):
    if arr is None:
        return None
    if dtype == np.int8:
        return Tensor(arr, dtype=ms.int8)
    return Tensor(arr, dtype=get_ms_dtype(dtype))

def generate_inputs(
    batch_size,
    seq_len,
    page_num,
    page_size,
    quant_mode,
    cache_mode,
    output_mode,
    input_dtype,
    epsilon,
):
    # 生成基础输入张量
    kv = np.random.randn(batch_size, 1, seq_len, 576).astype(input_dtype)
    gamma = np.random.randn(512).astype(input_dtype)
    cos = np.random.randn(batch_size, 1, seq_len, 64).astype(input_dtype)
    sin = np.random.randn(batch_size, 1, seq_len, 64).astype(input_dtype)

    # 初始化缓存相关变量
    k_cache = None
    ckv_cache = None
    index = None
    k_rope_scale = None
    c_kv_scale = None

    # 处理缓存模式
    if cache_mode != "Norm":
        # 创建初始缓存（全9张量）
        k_cache = np.ones((page_num, page_size, 1, 64), dtype=input_dtype) * 9
        ckv_cache = np.ones((page_num, page_size, 1, 512), dtype=input_dtype) * 9

        # 创建索引数组
        if "BLK" in cache_mode:
            total_blocks = batch_size * ((seq_len + page_size - 1) // page_size)
            index = np.arange(0, total_blocks * page_size, page_size, dtype=np.int64)
        else:
            index = np.arange(0, batch_size * seq_len, 1, dtype=np.int64)

    # 处理量化模式
    if quant_mode == 1:
        if k_cache is not None:
            k_cache = k_cache.astype(np.int8)
        if ckv_cache is not None:
            ckv_cache = ckv_cache.astype(np.int8)
        k_rope_scale = np.random.randn(64).astype(np.float32)
        c_kv_scale = np.random.randn(512).astype(np.float32)

    # 应用与原始代码相同的变换
    kv = 8 * kv - 10  # (-2 + 10) = 8
    gamma = 990 * gamma - 1000  # (-10 + 1000) = 990
    sin = 0.02 * sin - 0.01  # (0.01 + 0.01) = 0.02
    return (
        kv,
        gamma,
        cos,
        sin,
        index,
        k_cache,
        ckv_cache,
        k_rope_scale,
        c_kv_scale,
        cache_mode,
        output_mode,
        input_dtype,
        epsilon,
    )


def get_kv_rmsnorm_rope_cache_mode_enum(cache_mode_str):
    if cache_mode_str == "PA":
        return 1
    elif cache_mode_str == "PA_BNSD":
        return 2
    elif cache_mode_str == "PA_NZ":
        return 3
    elif cache_mode_str == "PA_BLK_BNSD":
        return 4
    elif cache_mode_str == "PA_BLK_NZ":
        return 5
    return 0  # "Norm"

batch_size = 64
seq_len = 1
page_num = 576
page_size = 128
quant_mode = 0
cache_mode = "PA_BNSD"
output_mode = False
input_dtype = np.float16
epsilon = 1e-5
(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, cache_mode, output_mode, _, _,) = generate_inputs(
    batch_size,
    seq_len,
    page_num,
    page_size,
    quant_mode,
    cache_mode,
    output_mode,
    input_dtype,
    epsilon)

kv_tensor = to_tensor(kv, input_dtype)
gamma_tensor = to_tensor(gamma, input_dtype)
cos_tensor = to_tensor(cos, input_dtype)
sin_tensor = to_tensor(sin, input_dtype)
index_tensor = to_tensor(index, np.int64)
k_rope_scale_tensor = None
c_kv_scale_tensor = None
if quant_mode == 1:
    k_cahce_tensor = to_tensor(k_cache, np.int8)
    ckv_cache_tensor = to_tensor(ckv_cache, np.int8)
    k_rope_scale_tensor = to_tensor(k_rope_scale, np.float32)
    c_kv_scale_tensor = to_tensor(c_kv_scale, np.float32)
else:
    k_cache_tensor = to_tensor(k_cache, input_dtype)
    ckv_cache_tensor = to_tensor(ckv_cache, input_dtype)

cache_mode_enum = get_kv_rmsnorm_rope_cache_mode_enum(cache_mode)
k_rope_offset = None
c_kv_offset = None
k_rope, c_kv = kv_rmsnorm_rope_cache_func(
        kv_tensor,
        gamma_tensor,
        cos_tensor,
        sin_tensor,
        index_tensor,
        k_cache_tensor,
        ckv_cache_tensor,
        k_rope_scale_tensor,
        c_kv_scale_tensor,
        k_rope_offset,
        c_kv_offset,
        epsilon,
        cache_mode_enum,
        output_mode,
    )
print(k_cache_tensor)
print(ckv_cache_tensor)
```
