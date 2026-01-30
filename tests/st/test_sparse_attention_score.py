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
"""Tests for ms_custom_ops.sparse_attention_score()"""

import typing
import pytest
import mindspore
import ms_custom_ops

def ceildiv(x, y):
    """Divide and round toward positive infinity."""
    return -(x // -y)

def make_block_indices(
        *, batch_size: int, num_kv_heads: int, seq_len: int, block_height: int, block_width: int
) -> mindspore.Tensor:
    """Make blocks to cover the full causal-attention area.

    Keyword args:
        batch_size: Batch size
        num_kv_heads: Number of attention heads
        seq_len: Sequence length
        block_height: Height of a block
        block_width: Width of a block
    """
    # Block-wise shape for the pattern
    pattern_height: int = ceildiv(seq_len, block_height)
    pattern_width: int = ceildiv(seq_len, block_width)

    # Construct row-wise block indices to cover the whole causal-attention area.
    # We need to replace with -1 block indices that will be ignored.
    rowwise_block_idx: mindspore.Tensor = mindspore.tensor(
        list(range(pattern_width))
    ).broadcast_to([pattern_height, -1])
    rowwise_elemwise_idx: mindspore.Tensor = (
        rowwise_block_idx * block_width
    )
    rowwise_elemwise_upperbound: mindspore.Tensor = (
        mindspore.tensor(list(range(1, pattern_height + 1))).reshape(
            [pattern_height, 1]
        )
        * block_height
    )
    rowwise_block_idx: mindspore.Tensor = rowwise_block_idx.where(
        rowwise_elemwise_idx < rowwise_elemwise_upperbound, -1
    )

    result: mindspore.Tensor = rowwise_block_idx.broadcast_to([batch_size, num_kv_heads, pattern_height, pattern_width])
    return result

def compute_relative_error(
        *, lhs: mindspore.Tensor, rhs: mindspore.Tensor, p = 'fro', dim: typing.Tuple[int, int] = (-2, -1)
) -> mindspore.Tensor:
    """Compute relative error between to (batch)matrices using norms.

    Keyword args:
        lhs: matrix to use as reference
        rhs: matrix to compare to the reference
        p: order of the norm (see mindspore.mint.norm())
        dim: dimensions to calculate the norm across (see mindspore.mint.norm())

    Details:
        The function computes a relative error between two matrices using norms.
        The lower the relative error the better.

        The function assumes that both matrices are in row-major layout and that additional batch dimensions are
        outermost. Specify the dim tuple otherwise.
    """
    diff_norm: mindspore.Tensor = mindspore.mint.norm(lhs - rhs, p=p, dim=dim)
    lhs_norm: mindspore.Tensor = mindspore.mint.norm(lhs, p=p, dim=dim)
    relative_error: mindspore.Tensor = diff_norm / lhs_norm
    return relative_error

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
def test_sparse_attention_score(
        *,
        batch_size: int = 1,
        num_heads: int = 14,
        head_size: int = 128,
        seq_len: int = 131072,
        scale: typing.Optional[float] = None,
        dtype: mindspore.dtype = mindspore.bfloat16,
        epsilon: float = 0.1,
):
    """
    Feature: Test ms_custom_ops.sparse_attention_score()
    Description:
        The function creates block indices and other parameters such that ms_custom_ops.sparse_attention_score() should
        compute the same area as mindspore.ops.flash_attention_score() and then compares the relative error against
        an accepted threshold (epsilon).
    Expectation: Precision results are correct within the given threshold, no exceptions.
    Keyword args:
        batch_size: Batch size
        num_heads: Number of attention heads
        head_size: Head size
        seq_len: Sequence length
        scale: scaling value
        dtype: Data type for query, key, value and result
        epsilon: threshold for relative error check
    """
    block_height: int = 128
    block_width: int = 256
    num_kv_heads: int = num_heads
    scale: float = scale if scale is not None else float(1.0 / (head_size ** (1/2)))

    query: mindspore.Tensor = mindspore.mint.rand([batch_size, num_heads, seq_len, head_size], dtype=dtype)
    key: mindspore.Tensor = mindspore.mint.rand([batch_size, num_kv_heads, seq_len, head_size], dtype=dtype)
    value: mindspore.Tensor = mindspore.mint.rand([batch_size, num_kv_heads, seq_len, head_size], dtype=dtype)
    score: mindspore.Tensor = mindspore.mint.zeros([batch_size, num_heads, seq_len, head_size], dtype=dtype)

    # attn_mask for sparse_mode=2
    mask: mindspore.Tensor = mindspore.mint.eq(
            mindspore.mint.tril(mindspore.mint.ones([2048, 2048], dtype=mindspore.uint8)),
            mindspore.mint.zeros([2048, 2048], dtype=mindspore.uint8))

    blocks = make_block_indices(
            batch_size=batch_size, num_kv_heads=num_kv_heads, seq_len=seq_len, block_height=block_height,
            block_width=block_width)

    real_shift_optional: mindspore.Tensor = mindspore.mint.rand([1], dtype=mindspore.float)
    drop_mask_optional: mindspore.Tensor = mindspore.mint.rand([1], dtype=mindspore.float)
    padding_mask_optional: mindspore.Tensor = mindspore.mint.rand([1], dtype=mindspore.float)
    atten_mask_optional: mindspore.Tensor = mask
    prefix_optional: mindspore.Tensor = mindspore.mint.rand([1], dtype=mindspore.int32)
    select_block_idx_optional: mindspore.Tensor = blocks.to(mindspore.int32)
    scale_value_optional: float  = scale
    keep_prob_optional: float = float(1.0)
    pre_tokens_optional: int = int(2147483647)
    next_tokens_optional: int = int(2147483647)
    head_num: int = num_heads
    input_layout: str = "BNSD"
    inner_precise_optional: int = int(0)
    sparse_mode_optional: int = int(2)
    select_block_len_optional: int = int(block_width)
    softmax_max_out: mindspore.Tensor = mindspore.mint.zeros([batch_size, num_heads, seq_len, 8], dtype=mindspore.float)
    softmax_sum_out: mindspore.Tensor = mindspore.mint.zeros([batch_size, num_heads, seq_len, 8], dtype=mindspore.float)
    softmax_out_out: mindspore.Tensor = mindspore.mint.rand([1], dtype=mindspore.float)
    attention_out_out: mindspore.Tensor = score

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
    mindspore.runtime.synchronize()

    ref: mindspore.Tensor = mindspore.ops.flash_attention_score(
            query, key, value, head_num, attn_mask=mask, scalar_value=scale,
            input_layout=input_layout, sparse_mode=sparse_mode_optional)
    err: mindspore.Tensor = compute_relative_error(lhs=ref, rhs=attention_out_out, p=float("inf"), dim=(-2, -1))

    assert mindspore.mint.lt(err, epsilon).all()
