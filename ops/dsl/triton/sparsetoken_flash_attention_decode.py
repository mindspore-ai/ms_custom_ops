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

"""FlexAttention Triton kernel for Sparse Token Decoding with Online Softmax

This module provides the flexible attention mechanism for sparse token decoding.
It implements an efficient Triton kernel that
handles sparse attention with online softmax normalization.
"""

import triton
import triton.language as tl


@triton.jit
def sparsetoken_flash_attention_decode(
    q_ptr,
    k_ptr,
    v_ptr,
    sparse_ind_ptr,
    sparse_nnz_ptr,
    output_ptr,
    batch_size,
    qo_heads,
    head_dim,
    gqa_group_size,
    softmax_scale,
    stride_q_b,
    stride_q_h,
    stride_q_1,
    stride_k_b,
    stride_k_h,
    stride_k_s,
    stride_v_b,
    stride_v_h,
    stride_v_s,
    stride_sparse_ind_b,
    stride_sparse_ind_h,
    stride_sparse_ind_l,
    stride_sparse_nnz_b,
    stride_sparse_nnz_h,
    stride_output_b,
    stride_output_h,
    stride_output_1,
    blokc_head: tl.constexpr,
):
    """
    Triton kernel for sparse token decoding with online softmax.
    Each program instance handles one (b, h) pair.
    """
    pid = tl.program_id(0)
    total_tasks = batch_size * qo_heads
    if pid >= total_tasks:
        return

    # Compute b and h from pid
    b = pid // qo_heads
    h = pid % qo_heads
    k_h = h // gqa_group_size

    # Load nnz from sparse_nnz
    nnz_offset = b * stride_sparse_nnz_b + h * stride_sparse_nnz_h
    nnz = tl.load(sparse_nnz_ptr + nnz_offset).to(tl.int32)
    # If nnz is 0, set to 1
    nnz = tl.where(nnz == 0, 1, nnz)

    # Load query vector
    q_offset = (
        b * stride_q_b + h * stride_q_h
        + 0 * stride_q_1 + tl.arange(0, blokc_head)
    )
    q_vec = tl.load(
        q_ptr + q_offset,
        mask=tl.arange(0, blokc_head) < head_dim,
        other=0.0,
    )

    # Initialize online softmax variables
    max_score = -float("inf")
    sum_exp = 0.0
    out_vec = tl.zeros([blokc_head], dtype=tl.float32)

    # Loop over nnz indices
    for i in range(0, nnz):
        # Load index from sparse_ind
        ind_offset = (
            b * stride_sparse_ind_b
            + h * stride_sparse_ind_h
            + i * stride_sparse_ind_l
        )
        idx = tl.load(sparse_ind_ptr + ind_offset).to(tl.int32)

        # Load key vector
        k_offset = (
            b * stride_k_b + k_h * stride_k_h
            + idx * stride_k_s + tl.arange(0, blokc_head)
        )
        k_vec = tl.load(
            k_ptr + k_offset,
            mask=tl.arange(0, blokc_head) < head_dim,
            other=0.0,
        )

        # Compute attention score
        dot_product = tl.sum(q_vec * k_vec)
        score = dot_product * softmax_scale

        # Load value vector
        v_offset = (
            b * stride_v_b + k_h * stride_v_h
            + idx * stride_v_s + tl.arange(0, blokc_head)
        )
        v_vec = tl.load(
            v_ptr + v_offset,
            mask=tl.arange(0, blokc_head) < head_dim,
            other=0.0,
        )

        # Online softmax update
        new_max = tl.maximum(max_score, score)
        exp_scale = tl.exp(max_score - new_max)
        exp_score = tl.exp(score - new_max)

        sum_exp = sum_exp * exp_scale + exp_score
        out_vec = out_vec * exp_scale + exp_score * v_vec
        max_score = new_max

    # Normalize the output
    out_vec = out_vec / sum_exp

    # Store the result
    output_offset = (
        b * stride_output_b + h * stride_output_h
        + 0 * stride_output_1 + tl.arange(0, blokc_head)
    )
    tl.store(
        output_ptr + output_offset,
        out_vec,
        mask=tl.arange(0, blokc_head) < head_dim,
    )
