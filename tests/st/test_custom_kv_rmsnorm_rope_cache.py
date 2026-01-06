# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Test cases for kv_rmsnorm_rope_cache custom operator.

This module contains comprehensive test cases for the kv_rmsnorm_rope_cache
custom operator implementation in MindSpore. It tests various data types,
shapes, execution modes, and cache configurations to ensure the operator
functions correctly across different scenarios.
"""

import copy
import pytest
import numpy as np

import mindspore as ms
from mindspore import context, Tensor
from mindspore.common.np_dtype import bfloat16
import ms_custom_ops


def get_ms_dtype(query_dtype):
    """Convert numpy dtype to MindSpore dtype.

    Args:
        query_dtype: Numpy data type to convert.

    Returns:
        Corresponding MindSpore data type.
    """
    if query_dtype == np.float32:
        ms_dtype = ms.float32
    elif query_dtype == np.float16:
        ms_dtype = ms.float16
    elif query_dtype == bfloat16:
        ms_dtype = ms.bfloat16
    elif query_dtype == np.int64:
        ms_dtype = ms.int64
    return ms_dtype


def get_kv_rmsnorm_rope_cache_mode_enum(cache_mode_str):
    """Convert cache mode string to enum value.

    Args:
        cache_mode_str: Cache mode string representation.

    Returns:
        Integer enum value corresponding to the cache mode.
    """
    if cache_mode_str == "PA":
        return 1
    if cache_mode_str == "PA_BNSD":
        return 2
    if cache_mode_str == "PA_NZ":
        return 3
    if cache_mode_str == "PA_BLK_BNSD":
        return 4
    if cache_mode_str == "PA_BLK_NZ":
        return 5
    return 0  # "Norm"


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
    """Generate test inputs for kv_rmsnorm_rope_cache operator.
    Args:
        batch_size: Batch size for the test.
        seq_len: Sequence length for the test.
        page_num: Number of pages for caching.
        page_size: Size of each page.
        quant_mode: Quantization mode (0 or 1).
        cache_mode: Cache mode string.
        output_mode: Output mode flag.
        input_dtype: Input data type.
        epsilon: Small constant for numerical stability.

    Returns:
        Tuple of generated test inputs.
    """
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


def generate_inputs_mindspore(
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
    """Generate MindSpore tensors for kv_rmsnorm_rope_cache operator test.

    Args:
        batch_size: Batch size for the test.
        seq_len: Sequence length for the test.
        page_num: Number of pages for caching.
        page_size: Size of each page.
        quant_mode: Quantization mode (0 or 1).
        cache_mode: Cache mode string.
        output_mode: Output mode flag.
        input_dtype: Input data type.
        epsilon: Small constant for numerical stability.

    Returns:
        Tuple of generated test inputs including both numpy arrays and MindSpore tensors.
    """
    # 使用 NumPy 函数生成相同的数据
    results = generate_inputs(
        batch_size,
        seq_len,
        page_num,
        page_size,
        quant_mode,
        cache_mode,
        output_mode,
        input_dtype,
        epsilon,
    )

    # 解包结果
    (
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
        dtype,
        _,
    ) = results

    def to_tensor(arr, dtype=None):
        """Convert numpy array to MindSpore tensor.

        Args:
            arr: Numpy array to convert.
            tensor_dtype: Target data type for the tensor.

        Returns:
            MindSpore tensor or None if input is None.
        """
        if arr is None:
            return None
        if dtype == np.int8:
            return Tensor(arr, dtype=ms.int8)
        return Tensor(arr, dtype=get_ms_dtype(dtype))

    kv_tensor = to_tensor(kv, dtype)
    gamma_tensor = to_tensor(gamma, dtype)
    cos_tensor = to_tensor(cos, dtype)
    sin_tensor = to_tensor(sin, dtype)
    index_tensor = to_tensor(index, np.int64)
    k_rope_scale_tensor = None
    c_kv_scale_tensor = None
    if quant_mode == 1:
        k_cache_tensor = to_tensor(k_cache, np.int8)
        ckv_cache_tensor = to_tensor(ckv_cache, np.int8)
        k_rope_scale_tensor = to_tensor(k_rope_scale, np.float32)
        c_kv_scale_tensor = to_tensor(c_kv_scale, np.float32)
    else:
        k_cache_tensor = to_tensor(k_cache, dtype)
        ckv_cache_tensor = to_tensor(ckv_cache, dtype)

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
        kv_tensor,
        gamma_tensor,
        cos_tensor,
        sin_tensor,
        index_tensor,
        k_cache_tensor,
        ckv_cache_tensor,
        k_rope_scale_tensor,
        c_kv_scale_tensor,
        cache_mode,
        output_mode,
        input_dtype,
        epsilon,
    )


def supported_op_exec_numpy(
        kv,
        gamma,
        cos,
        sin,
        index,
        k_cache,
        ckv_cache,
        k_rope_scale=None,
        c_kv_scale=None,
        k_rope_offset=None,
        c_kv_offset=None,
        epsilon=1e-05,
        cache_mode="Norm",
        is_output_kv=False,
):
    """
    NumPy实现aclnnKvRmsNormRopeCache功能

    参数:
        kv: 输入张量 [batch_size, N, seq_len, D]
        gamma: RMSNorm缩放因子 [Dv]
        cos: 旋转位置编码余弦部分 [batch_size, 1, seq_len, Dk] 或 [batch_size, 1, 1, Dk]
        sin: 旋转位置编码正弦部分 [batch_size, 1, seq_len, Dk] 或 [batch_size, 1, 1, Dk]
        index: 索引张量，形状取决于cache_mode
        k_cache: k缓存 (输入/输出)
        ckv_cache: ckv缓存 (输入/输出)
        k_rope_scale: k的量化缩放因子 (可选)
        c_kv_scale: ckv的量化缩放因子 (可选)
        k_rope_offset: k的量化偏移 (可选)
        c_kv_offset: ckv的量化偏移 (可选)
        epsilon: RMSNorm的小常数
        cache_mode: 缓存模式 ("Norm", "PA", "PA_BNSD", "PA_NZ", "PA_BLK_NZ", "PA_BLK_BNSD")
        is_output_kv: 是否输出中间结果
    """

    # 辅助函数
    def round_float_to_int8(src_array):
        """将浮点数组四舍五入并转换为int8"""
        rounded_array = np.round(src_array)
        clipped_array = np.clip(rounded_array, -128, 127)
        return clipped_array.astype(np.int8)

    def rotate_half(x):
        """旋转输入的一半隐藏维度"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return np.concatenate((-x2, x1), axis=-1)

    # 获取输入形状
    batch_size, _, seq_len, dim = kv.shape
    d_v = gamma.shape[0]  # RMSNorm部分的维度
    d_k = dim - d_v  # ROPE部分的维度

    # 确定量化模式
    quant_mode = 0 if k_rope_scale is None else 1
    d0 = 32 if quant_mode == 1 else 16

    # 对于PA模式，获取缓存形状
    if "PA" in cache_mode:
        block_num, block_size, _, _ = k_cache.shape

    # 对于BLK模式，计算索引页长度
    if "BLK" in cache_mode:
        index_page_id_length = index.shape[0] // batch_size

    # 拆分输入
    rms_in = kv[..., :d_v].astype(np.float32)  # RMSNorm部分
    rope_in = kv[..., d_v:].astype(np.float32)  # ROPE部分

    # 计算RMSNorm
    # 计算平方和平均值
    square_x = np.square(rms_in)
    mean_square_x = np.mean(square_x, axis=-1, keepdims=True)
    rms = np.sqrt(mean_square_x + epsilon)

    # 应用RMSNorm
    gamma_f32 = gamma.astype(np.float32)
    y = rms_in / rms * gamma_f32

    # 计算ROPE
    # 重塑输入以匹配旋转要求
    k = rope_in.reshape(batch_size, 1, seq_len, 32, 2)
    k = np.transpose(k, (0, 1, 2, 4, 3))  # 交换最后两个维度
    k = k.reshape(batch_size, 1, seq_len, 64)

    # 广播cos和sin到正确的形状
    if cos.shape[2] == 1:  # [batch_size, 1, 1, Dk]
        cos_broadcast = np.tile(cos, (1, 1, seq_len, 1))
    else:  # [batch_size, 1, seq_len, Dk]
        cos_broadcast = cos

    if sin.shape[2] == 1:  # [batch_size, 1, 1, Dk]
        sin_broadcast = np.tile(sin, (1, 1, seq_len, 1))
    else:  # [batch_size, 1, seq_len, Dk]
        sin_broadcast = sin

    # 应用旋转位置编码
    k_embed = k * cos_broadcast.astype(np.float32) + rotate_half(
        k
    ) * sin_broadcast.astype(np.float32)

    # 复制输出结果
    k_embed_out = copy.deepcopy(k_embed).astype(kv.dtype)
    y_out = copy.deepcopy(y).astype(kv.dtype)

    # 准备缓存数据
    if quant_mode == 1:
        # 应用量化
        if k_rope_offset is not None:
            k_embed = (k_embed - k_rope_offset) * k_rope_scale
        else:
            k_embed = k_embed * k_rope_scale
        k_embed = round_float_to_int8(k_embed)

        if c_kv_offset is not None:
            y = (y - c_kv_offset) * c_kv_scale
        else:
            y = y * c_kv_scale
        y = round_float_to_int8(y)
    else:
        # 转换为缓存数据类型
        k_embed = k_embed.astype(k_cache.dtype)
        y = y.astype(ckv_cache.dtype)

    # 根据缓存模式更新缓存
    if cache_mode == "Norm":
        # Norm模式 - 直接更新缓存
        for b in range(batch_size):
            for s in range(seq_len):
                pos = index[b, s]
                if pos != -1:  # -1表示跳过更新
                    k_cache[b, 0, pos] = k_embed[b, 0, s]
                    ckv_cache[b, 0, pos] = y[b, 0, s]

    elif cache_mode in ("PA", "PA_BNSD"):
        # PA和PA_BNSD模式
        k_cache_flat = k_cache.reshape(-1, d_k)
        ckv_cache_flat = ckv_cache.reshape(-1, d_v)

        for b in range(batch_size):
            for s in range(seq_len):
                offset = index[b * seq_len + s]
                if offset >= 0:
                    k_cache_flat[offset] = k_embed[b, 0, s]
                    ckv_cache_flat[offset] = y[b, 0, s]

        # 恢复缓存形状
        k_cache[:] = k_cache_flat.reshape(block_num, block_size, 1, d_k)
        ckv_cache[:] = ckv_cache_flat.reshape(block_num, block_size, 1, d_v)

    elif cache_mode == "PA_BLK_NZ":
        # PA_BLK_NZ模式
        k_cache_reshaped = k_cache.reshape(block_num, 1, -1, block_size, d0)
        ckv_cache_reshaped = ckv_cache.reshape(block_num, 1, -1, block_size, d0)

        for b in range(batch_size):
            for s in range(seq_len):
                index_page_id = s // block_size
                page_offset = index[b * index_page_id_length + index_page_id]

                if page_offset >= 0:
                    page_id = page_offset // block_size
                    token_offset = s % block_size

                    # 更新缓存
                    k_embed_reshaped = k_embed[b, 0, s].reshape(-1, d0)
                    k_cache_reshaped[page_id, 0, :, token_offset] = k_embed_reshaped

                    y_reshaped = y[b, 0, s].reshape(-1, d0)
                    ckv_cache_reshaped[page_id, 0, :, token_offset] = y_reshaped

        # 恢复缓存形状
        k_cache[:] = k_cache_reshaped.reshape(block_num, block_size, 1, d_k)
        ckv_cache[:] = ckv_cache_reshaped.reshape(block_num, block_size, 1, d_v)

    elif cache_mode == "PA_BLK_BNSD":
        # PA_BLK_BNSD模式
        k_cache_reshaped = k_cache.reshape(block_num, block_size, 1, -1)
        ckv_cache_reshaped = ckv_cache.reshape(block_num, block_size, 1, -1)

        for b in range(batch_size):
            for s in range(seq_len):
                index_page_id = s // block_size
                page_offset = index[b * index_page_id_length + index_page_id]

                if page_offset >= 0:
                    page_id = page_offset // block_size
                    token_offset = s % block_size

                    # 更新缓存
                    k_cache_reshaped[page_id, token_offset, 0] = k_embed[b, 0, s]
                    ckv_cache_reshaped[page_id, token_offset, 0] = y[b, 0, s]

        # 恢复缓存形状
        k_cache[:] = k_cache_reshaped.reshape(block_num, block_size, 1, d_k)
        ckv_cache[:] = ckv_cache_reshaped.reshape(block_num, block_size, 1, d_v)

    elif cache_mode == "PA_NZ":
        # PA_NZ模式
        if quant_mode == 1:
            d0 = 32
        else:
            d0 = 16

        k_cache_reshaped = k_cache.reshape(block_num, 1, -1, block_size, d0)
        ckv_cache_reshaped = ckv_cache.reshape(block_num, 1, -1, block_size, d0)

        for b in range(batch_size):
            for s in range(seq_len):
                page_offset = index[b * seq_len + s]
                if page_offset >= 0:
                    page_id = page_offset // block_size
                    token_offset = page_offset % block_size

                    # 更新缓存
                    k_embed_reshaped = k_embed[b, 0, s].reshape(-1, d0)
                    k_cache_reshaped[page_id, 0, :, token_offset] = k_embed_reshaped

                    y_reshaped = y[b, 0, s].reshape(-1, d0)
                    ckv_cache_reshaped[page_id, 0, :, token_offset] = y_reshaped

        # 恢复缓存形状
        k_cache[:] = k_cache_reshaped.reshape(block_num, block_size, 1, d_k)
        ckv_cache[:] = ckv_cache_reshaped.reshape(block_num, block_size, 1, d_v)

    # 返回结果
    if is_output_kv:
        return k_cache, ckv_cache, k_embed_out, y_out

    return k_cache, ckv_cache


class KvRmsNormRopeCacheNet(ms.nn.Cell):
    """Neural network cell for kv_rmsnorm_rope_cache operator.

    This class provides a wrapper around the custom kv_rmsnorm_rope_cache
    operator to integrate it with MindSpore's neural network framework.
    """

    def _init__(self):
        """Initialize the kv_rmsnorm_rope_cache network cell."""
        super().__init__()

    def construct(
            self,
            kv,
            gamma,
            cos,
            sin,
            index,
            k_cache_ref,
            c_kv_cache_ref,
            k_rope_scale=None,
            c_kv_scale=None,
            k_rope_offset=None,
            c_kv_offset=None,
            epsilon=1e-5,
            cache_mode=0,
            is_output_kv=False,
    ):
        """Construct the kv_rmsnorm_rope_cache operation.

        Args:
            kv: Input key-value tensor.
            gamma: RMSNorm scaling factor.
            cos: Cosine component for rotary position encoding.
            sin: Sine component for rotary position encoding.
            index: Index tensor for caching.
            k_cache_ref: Key cache reference.
            c_kv_cache_ref: Key-value cache reference.
            k_rope_scale: Quantization scale for key rope (optional).
            c_kv_scale: Quantization scale for key-value cache (optional).
            k_rope_offset: Quantization offset for key rope (optional).
            c_kv_offset: Quantization offset for key-value cache (optional).
            epsilon: Small constant for numerical stability.
            cache_mode: Cache mode enum value.
            is_output_kv: Whether to output intermediate results.

        Returns:
            Result of the kv_rmsnorm_rope_cache operation.
        """
        return ms_custom_ops.kv_rmsnorm_rope_cache(
            kv,
            gamma,
            cos,
            sin,
            index,
            k_cache_ref,
            c_kv_cache_ref,
            k_rope_scale,
            c_kv_scale,
            k_rope_offset,
            c_kv_offset,
            epsilon,
            cache_mode,
            is_output_kv,
        )


def run(
        net,
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
    """Execute kv_rmsnorm_rope_cache operator test and validate results.

    This function generates test data, executes the operator through the
    provided network, and validates the output against a reference implementation.

    Args:
        net: Neural network containing the kv_rmsnorm_rope_cache operator.
        batch_size: Batch size for the test.
        seq_len: Sequence length for the test.
        page_num: Number of pages for caching.
        page_size: Size of each page.
        quant_mode: Quantization mode (0 or 1).
        cache_mode: Cache mode string.
        output_mode: Output mode flag.
        input_dtype: Input data type.
        epsilon: Small constant for numerical stability.

    Raises:
        AssertionError: If the output does not match the reference implementation.
    """
    (
        kv,
        gamma,
        cos,
        sin,
        index,
        k_cache,
        ckv_cache,
        k_rope_scale,
        c_kv_scale,
        kv_tensor,
        gamma_tensor,
        cos_tensor,
        sin_tensor,
        index_tensor,
        k_cache_tensor,
        ckv_cache_tensor,
        k_rope_scale_tensor,
        c_kv_scale_tensor,
        cache_mode,
        output_mode,
        input_dtype,
        epsilon,
    ) = generate_inputs_mindspore(
        batch_size,
        seq_len,
        page_num,
        page_size,
        quant_mode,
        cache_mode,
        output_mode,
        input_dtype,
        epsilon,
    )
    k_rope_offset = None
    c_kv_offset = None
    k_cache_golden = None
    ckv_cache_golden = None
    k_rope_golden = None
    ckv_out_golden = None
    if not output_mode:
        k_cache_golden, ckv_cache_golden = supported_op_exec_numpy(
            kv,
            gamma,
            cos,
            sin,
            index,
            k_cache,
            ckv_cache,
            k_rope_scale,
            c_kv_scale,
            k_rope_offset,
            c_kv_offset,
            epsilon,
            cache_mode,
            is_output_kv=output_mode,
        )
    else:
        k_cache_golden, ckv_cache_golden, k_rope_golden, ckv_out_golden = (
            supported_op_exec_numpy(
                kv,
                gamma,
                cos,
                sin,
                index,
                k_cache,
                ckv_cache,
                k_rope_scale,
                c_kv_scale,
                k_rope_offset,
                c_kv_offset,
                epsilon,
                cache_mode,
                is_output_kv=output_mode,
            )
        )

    cache_mode_enum = get_kv_rmsnorm_rope_cache_mode_enum(cache_mode)
    k_rope, c_kv = net(
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
    k_cache = k_cache_tensor
    ckv_cache = ckv_cache_tensor
    k_cache_np = k_cache.asnumpy()
    ckv_cache_np = ckv_cache.asnumpy()
    if output_mode:
        k_rope_np = k_rope.asnumpy()
        c_kv_np = c_kv.asnumpy()
    else:
        pass

    if input_dtype == np.float16:
        atol = 0.01
        rtol = 0.01
    else:
        atol = 1e-5
        rtol = 1e-5

    np.testing.assert_allclose(k_cache_golden, k_cache_np, rtol=rtol, atol=atol)
    np.testing.assert_allclose(ckv_cache_golden, ckv_cache_np, rtol=rtol, atol=atol)
    if output_mode:
        np.testing.assert_allclose(k_rope_golden, k_rope_np, rtol=rtol, atol=atol)
        np.testing.assert_allclose(ckv_out_golden, c_kv_np, rtol=rtol, atol=atol)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("seq_len", [1])
@pytest.mark.parametrize("page_num", [576])
@pytest.mark.parametrize("page_size", [128])
@pytest.mark.parametrize("quant_mode", [0])
@pytest.mark.parametrize(
    "cache_mode",
    ["PA_BLK_NZ", "PA_BLK_BNSD", "PA_NZ", "PA_BNSD"],
)
@pytest.mark.parametrize("output_mode", [False])
@pytest.mark.parametrize(
    "input_dtype",
    [np.float16],
)
def test_kv_rmsnorm_rope_cache(
        exec_mode,
        batch_size,
        seq_len,
        page_num,
        page_size,
        quant_mode,
        cache_mode,
        output_mode,
        input_dtype,
):
    """Test the kv_rmsnorm_rope_cache custom operator.

    Feature: Test the kv_rmsnorm_rope_cache custom operator functionality.
    Description: This test verifies the kv_rmsnorm_rope_cache operator under various
                 execution modes and cache configurations. It tests different combinations
                 of batch sizes, sequence lengths, page numbers, page sizes, quantization
                 modes, cache modes, and output modes to ensure correct behavior.
    Expectation: The operator should produce correct results for all tested configurations
                 without errors or exceptions. The output should match the expected values
                 within acceptable numerical tolerance.
    """
    epsilon = 1e-5
    ms.set_context(device_target="Ascend", mode=exec_mode)
    net = KvRmsNormRopeCacheNet()
    run(
        net,
        batch_size,
        seq_len,
        page_num,
        page_size,
        quant_mode,
        cache_mode,
        output_mode,
        input_dtype,
        epsilon,
    )
