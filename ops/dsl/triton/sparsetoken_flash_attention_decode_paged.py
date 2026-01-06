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

"""FlexAttention Triton Kernel for Sparse Token Decoding with Paged KV Cache

This module provides the flexible attention mechanism for sparse token decoding
using a paged key-value cache. It implements an efficient Triton kernel that
handles sparse attention with online softmax normalization.
"""

import triton
import triton.language as tl


@triton.jit
def sparsetoken_flash_attention_decode_paged(
    q_ptr,
    paged_kv_cache_ptr,
    kv_page_indptr_ptr,
    kv_page_indices_ptr,
    sparse_ind_ptr,
    sparse_nnz_ptr,
    output_ptr,
    h_q: tl.constexpr,
    h_k: tl.constexpr,
    dim: tl.constexpr,
    page_size: tl.constexpr,
    l_max: tl.constexpr,
    sqrt_d: tl.constexpr,
):
    """
    Triton 内核实现稀疏 token 分页注意力
    每个程序处理一个 (batch, head) 对
    """
    pid = tl.program_id(0)
    b = pid // h_q
    h = pid % h_q

    # 加载 nnz
    offset_nnz = b * h_q + h
    nnz = tl.load(sparse_nnz_ptr + offset_nnz).to(tl.int32)

    # 计算输出偏移
    output_offset = b * h_q * dim + h * dim + tl.arange(0, dim)

    if nnz == 0:
        # 如果 nnz 为 0，存储零输出
        tl.store(output_ptr + output_offset, 0.0)
        return

    # 加载查询向量 q [1, dim]
    q_offset = b * h_q * dim + h * dim + tl.arange(0, dim)
    q = tl.load(q_ptr + q_offset)

    # 计算 GQA 组和 KV 头索引
    groups = h_q // h_k
    h_k = h // groups

    # 初始化在线 softmax 状态
    m = -float("inf")
    l = 0.0
    acc = tl.zeros([dim], dtype=tl.float32)

    # 循环处理每个 token
    for i in range(0, nnz):
        # 加载 token 索引
        offset_ind = b * h_q * l_max + h * l_max + i
        token_idx = tl.load(sparse_ind_ptr + offset_ind).to(tl.int32)

        # 计算页面索引和偏移
        page_idx = token_idx // page_size
        offset_in_page = token_idx % page_size

        # 加载页面起始指针
        ptr_start = tl.load(kv_page_indptr_ptr + b).to(tl.int32)

        # 加载页面 ID
        page_id_offset = ptr_start + page_idx
        page_id = tl.load(kv_page_indices_ptr + page_id_offset).to(tl.int32)

        # 加载 K 向量 [1, dim]
        k_offset = (
            page_id * (2 * h_k * page_size * dim)
            + 0 * (h_k * page_size * dim)
            + h_k * (page_size * dim)
            + offset_in_page * dim
            + tl.arange(0, dim)
        )
        k = tl.load(paged_kv_cache_ptr + k_offset)

        # 加载 V 向量 [1, dim]
        v_offset = (
            page_id * (2 * h_k * page_size * dim)
            + 1 * (h_k * page_size * dim)
            + h_k * (page_size * dim)
            + offset_in_page * dim
            + tl.arange(0, dim)
        )
        v = tl.load(paged_kv_cache_ptr + v_offset)

        # 计算注意力分数
        dot_product = tl.sum(q * k)
        score = dot_product / sqrt_d  # 使用预计算的 sqrt_d

        # 更新在线 softmax
        m_new = tl.maximum(m, score)
        alpha = tl.exp(m - m_new)
        beta = tl.exp(score - m_new)
        l_new = l * alpha + beta
        acc_new = acc * alpha + beta * v
        m = m_new
        l = l_new
        acc = acc_new

    # 计算最终输出
    out = acc / l
    tl.store(output_ptr + output_offset, out)
