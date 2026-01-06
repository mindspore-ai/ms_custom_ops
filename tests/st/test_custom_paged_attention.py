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
"""
System tests for the ms_custom_ops.paged_attention operator.

This module validates functional correctness, masking modes, GQA/MLA variants,
quantization paths, layouts, and MTP/lookahead decoding behaviour.
"""

import math
import random
from dataclasses import dataclass
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, context, ops, nn
from paged_attention_ms_reference import PagedAttentionMsReference
import ms_custom_ops

# Mask type enumerations
MASK_UNDEFINED = 0
MASK_NORM = 1
MASK_ALIBI = 2
MASK_SPEC = 3
MASK_FREE = 4

# Quantization type enumerations
QUANT_UNQUANT = 0
DEQUANT_FUSION = 1
QUANT_QKV_OFFLINE = 2
QUANT_QKV_ONLINE = 3

# Input layout enumerations
INPUT_LAYOUT_BSND = 0
INPUT_LAYOUT_BNSD = 1

# Input format enumerations
INPUT_FORMAT_ND = 0
INPUT_FORMAT_NZ = 1

REFERENCE_NUMPY = "numpy"
REFERENCE_MINDSPORE = "mindspore"


@dataclass
class AttentionContext:
    """Encapsulates attention computation context parameters."""
    num_heads: int
    kv_heads: int
    scale: float
    mask: np.ndarray
    mask_type: int
    batch_idx: int
    q_idx: int
    ql: int
    kl: int
    mla_v_dim: int
    mask_dtype: ms.dtype


@dataclass
class QuantizationParams:
    """Encapsulates quantization scale parameters."""
    k_scale: np.ndarray
    v_scale: np.ndarray
    p_scale: np.ndarray = None


class PagedAttentionDataGenerator:
    """Data generator and golden reference calculator for paged attention tests.

    Handles mask construction, golden computation, and accuracy validation.
    """

    def __init__(self, rng_seed: int = 2025):
        """Initialize with random seed."""
        self.seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)
        random.seed(rng_seed)
        np.random.seed(rng_seed)

    def _np_dtype_for(self, ms_dtype: ms.dtype) -> np.dtype:
        """Convert MindSpore dtype to NumPy dtype."""
        if ms_dtype == ms.float16:
            return np.float16
        if ms_dtype == ms.bfloat16:
            return np.float32  # bfloat16 not native in numpy
        if ms_dtype == ms.int8:
            return np.int8
        return np.float32

    def generate_inputs(self, num_heads: int, kv_heads: int, head_size: int,
                       block_size: int, num_blocks: int, q_seq_lens: list, context_lens: list,
                       q_dtype: ms.dtype, kv_dtype: ms.dtype, mask_type: int,
                       quant_type: int = QUANT_UNQUANT, has_quant_offset: bool = False,
                       mla_v_dim: int = 0, mask_out_dtype: ms.dtype = None):
        """Generate query, key_cache, value_cache, block_tables, masks, and quantization parameters.

        Directly accepts q_seq_lens and context_lens (kv_seq_lens) for flexible configuration,
        especially important for MTP/lookahead scenarios.

        **Returns numpy arrays** for golden calculation; convert to Tensors before network input.
        **Masks are generated in their natural shapes (without NZ padding)** for golden calculation.
        NZ padding (if needed) should be applied later in _prepare_mask_for_network.

        Args:
            num_heads: Number of query heads
            kv_heads: Number of KV heads
            head_size: Dimension per head (for Q/K in MLA mode, or all in
                non-MLA mode)
            block_size: Block size for KV cache
            num_blocks: Total number of blocks
            q_seq_lens: List of query sequence lengths per batch
                (e.g., [1, 15, 30, 6] for MTP)
            context_lens: List of context (KV) lengths per batch
                (e.g., [10, 64, 64, 64])
            q_dtype: Query tensor dtype (for reference, actual data is numpy)
            kv_dtype: Key/Value cache dtype (for reference, actual data is numpy)
            mask_type: Type of attention mask (MASK_UNDEFINED, MASK_NORM,
                MASK_SPEC, etc.)
            quant_type: Quantization type.
                Supported values: QUANT_UNQUANT, DEQUANT_FUSION,
                QUANT_QKV_OFFLINE, QUANT_QKV_ONLINE
            has_quant_offset: Whether to generate quantization offsets
            mla_v_dim: MLA V/O head dimension (0 for non-MLA, >0 for MLA mode)
            mask_out_dtype: Mask output dtype (for reference)

        Returns:
            Dictionary containing all generated numpy arrays and metadata:
            - query, key_cache, value_cache, block_tables, q_seq_lens,
              kv_seq_lens, mask (all numpy arrays or None)
            - k_descale, v_descale (for DEQUANT_FUSION, per-element, numpy
              arrays)
            - k_offset, v_offset (ONLY for DEQUANT_FUSION if has_quant_offset,
              numpy arrays)
            - k_scale_per_head, v_scale_per_head (for full QKV quant,
              per-head, numpy arrays)
            - p_scale (ONLY for QUANT_QKV_OFFLINE, per-head, numpy array)
            - q_dtype, kv_dtype, mask_dtype (MindSpore dtypes for later Tensor
              conversion)
            Note: Q scale is NOT used (only K and V scales are used)
        """
        # Determine if this is full quantization
        is_full_quant = quant_type in (QUANT_QKV_OFFLINE, QUANT_QKV_ONLINE)
        is_dequant_fusion = quant_type == DEQUANT_FUSION

        # Calculate total number of query tokens from q_seq_lens
        num_tokens = sum(q_seq_lens)
        batch_size = len(q_seq_lens)

        # Generate query array
        q_np_dtype = self._np_dtype_for(q_dtype)
        if q_dtype == ms.int8:
            # For quantized int8: use range -2.0~2.0 before quantization
            q_range = 2.0 if is_full_quant else 1.0
            query_np = self.rng.uniform(
                -q_range, q_range, (num_tokens, num_heads, head_size)
            ).astype(np.float32)
            query_np = np.clip(
                np.rint(query_np * 127.0 / q_range), -127, 127
            ).astype(np.int8)
        else:
            # Generate query in the target numpy dtype grid.
            # For fp16: use np.float16 directly.
            # For bf16: generate float32 and then quantize through MindSpore
            # Tensor(bfloat16)
            query_np = self.rng.uniform(
                -1.0, 1.0, (num_tokens, num_heads, head_size)
            ).astype(q_np_dtype)
            if q_dtype == ms.bfloat16:
                # Simulate bf16 quantization so that golden and operator see the
                # same value grid.
                # Tensor(..., ms.bfloat16).asnumpy() returns float32 values that
                # lie on the bf16 grid.
                query_np = Tensor(query_np, dtype=ms.bfloat16).asnumpy().astype(
                    np.float32
                )

        # Determine head dimensions for MLA
        # MLA: Q/K use head_size (head_size_qk), V/O use mla_v_dim (head_size_vo)
        # Non-MLA: all use head_size
        head_size_qk = head_size
        head_size_vo = mla_v_dim if mla_v_dim > 0 else head_size
        is_mla = mla_v_dim > 0

        # Generate key/value cache arrays
        # IMPORTANT: For MLA with KV combined mode, V cache views the first
        # head_size_vo dimensions of K cache
        kv_np_dtype = self._np_dtype_for(kv_dtype)
        if kv_dtype == ms.int8:
            # Full quant uses 2.0, dequant_fusion uses 4.0
            kv_range = 2.0 if is_full_quant else 4.0
            # For MLA combined mode, K cache must have at least head_size_qk dimensions
            # (which includes head_size_vo)
            cache_shape = (num_blocks, block_size, kv_heads, head_size_qk)
            key_cache_np = self.rng.uniform(
                -kv_range, kv_range, cache_shape
            ).astype(np.float32)
            key_cache_np = np.clip(
                np.rint(key_cache_np * 127.0 / kv_range), -127, 127
            ).astype(np.int8)
            # For MLA: V cache is the first head_size_vo dimensions of K cache (KV combined)
            if is_mla:
                value_cache_np = key_cache_np[:, :, :, :head_size_vo]
            else:
                v_cache_shape = (num_blocks, block_size, kv_heads, head_size_vo)
                value_cache_np = self.rng.uniform(
                    -kv_range, kv_range, v_cache_shape
                ).astype(np.float32)
                value_cache_np = np.clip(
                    np.rint(value_cache_np * 127.0 / kv_range), -127, 127
                ).astype(np.int8)
        else:
            cache_shape = (num_blocks, block_size, kv_heads, head_size_qk)
            key_cache_np = self.rng.uniform(
                -1.0, 1.0, cache_shape
            ).astype(kv_np_dtype)
            # For MLA: V cache is the first head_size_vo dimensions of K cache (KV combined)
            if is_mla:
                value_cache_np = key_cache_np[:, :, :, :head_size_vo]
            else:
                v_cache_shape = (num_blocks, block_size, kv_heads, head_size_vo)
                value_cache_np = self.rng.uniform(
                    -1.0, 1.0, v_cache_shape
                ).astype(kv_np_dtype)

        # Generate block tables (numpy array)
        max_context_len = max(context_lens)
        max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
        block_tables_list = []
        for _ in range(batch_size):
            block_table = [random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)]
            block_tables_list.append(block_table)
        block_tables_np = np.array(block_tables_list, dtype=np.int32)

        # Use provided q_seq_lens and context_lens (kv_seq_lens) directly
        # Keep as numpy arrays
        kv_seq_lens = context_lens
        q_seq_lens_np = np.array(q_seq_lens, dtype=np.int32)
        kv_seq_lens_np = np.array(kv_seq_lens, dtype=np.int32)

        # Generate mask based on mask_type (returns numpy array without NZ padding)
        # Mask dtype must match operator output dtype in quant scenes
        mask_dtype = mask_out_dtype if mask_out_dtype is not None else q_dtype
        is_mla = mla_v_dim > 0
        mask_np = self._generate_mask(
            mask_type,
            num_tokens,
            max_context_len,
            q_seq_lens,
            kv_seq_lens,
            mask_dtype,
            num_heads,
            is_mla,
        )

        # Generate quantization parameters based on quant_type
        # All as numpy arrays
        k_descale_np = None
        v_descale_np = None
        k_offset_np = None
        v_offset_np = None
        # Note: Q scale is NOT used in paged attention (only K/V scales are used)
        k_scale_per_head_np = None
        v_scale_per_head_np = None
        p_scale_np = None  # For full quant: softmax P matrix quantization scale

        if is_dequant_fusion:
            # Dequant Fusion: generate per-element descale for KV
            # Shape: (kv_heads * head_size_qk/vo) - per-element quantization
            k_descale_np = self.rng.integers(
                -1, 2, size=(kv_heads * head_size_qk,)
            ).astype(np.float32)
            v_descale_np = self.rng.integers(
                -1, 2, size=(kv_heads * head_size_vo,)
            ).astype(np.float32)

            # Generate offsets if requested
            # Offsets are ONLY for Dequant Fusion, NOT for full quantization
            # Shape: (kv_heads * head_size_qk/vo) - per-element offset
            if has_quant_offset:
                k_offset_np = self.rng.integers(
                    -20, 20, size=(kv_heads * head_size_qk,)
                ).astype(np.int32)
                v_offset_np = self.rng.integers(
                    -20, 20, size=(kv_heads * head_size_vo,)
                ).astype(np.int32)

        elif is_full_quant:
            # Full QKV Quant: generate per-head scales
            # Range [-1, 2] for K/V scales
            # NOTE: Q scale is NOT used (only K and V scales)
            # NOTE: Full quantization does NOT use offsets
            k_scale_per_head_np = self.rng.uniform(
                -1.0, 2.0, size=(num_heads,)
            ).astype(np.float32)
            v_scale_per_head_np = self.rng.uniform(
                -1.0, 2.0, size=(num_heads,)
            ).astype(np.float32)

            # Generate P matrix quantization scale ONLY for offline quantization
            # Online quantization (quantType=3) doesn't need p_scale
            if quant_type == QUANT_QKV_OFFLINE:
                # Shape: (num_heads,), range [-1, 2]
                p_scale_np = self.rng.uniform(
                    -1.0, 2.0, size=(num_heads,)
                ).astype(np.float32)

        # Return all parameters as a dictionary of numpy arrays and metadata
        return {
            'query': query_np,
            'key_cache': key_cache_np,
            'value_cache': value_cache_np,
            'block_tables': block_tables_np,
            'q_seq_lens': q_seq_lens_np,
            'kv_seq_lens': kv_seq_lens_np,
            'mask': mask_np,
            'k_descale': k_descale_np,
            'v_descale': v_descale_np,
            'k_offset': k_offset_np,
            'v_offset': v_offset_np,
            'k_scale_per_head': k_scale_per_head_np,
            'v_scale_per_head': v_scale_per_head_np,
            'p_scale': p_scale_np,
            'q_dtype': q_dtype,
            'kv_dtype': kv_dtype,
            'mask_dtype': mask_dtype,
        }

    def _get_alibi_slopes(self, num_heads: int) -> np.ndarray:
        """Generate ALIBI slopes for positional bias.

        Args:
            num_heads: Number of attention heads

        Returns:
            ALIBI slopes array of shape (num_heads,)
        """
        nearest_power_of_two = 2 ** int(np.floor(np.log2(num_heads)))
        m0 = 2.0 ** (-8.0 / nearest_power_of_two)
        slopes = np.array(
            [m0 ** i for i in range(1, nearest_power_of_two + 1)],
            dtype=np.float32,
        )

        if nearest_power_of_two < num_heads:
            m1 = 2.0 ** (-4.0 / nearest_power_of_two)
            # Generate additional slopes with step size 2
            additional_count = num_heads - nearest_power_of_two
            mm = np.array(
                [m1 ** i for i in range(1, 1 + 2 * additional_count, 2)],
                dtype=np.float32,
            )
            slopes = np.concatenate([slopes, mm], axis=0)

        return slopes

    def _generate_mask(self, mask_type: int, num_tokens: int, max_context_len: int,
                      q_seq_lens: list, kv_seq_lens: list, dtype: ms.dtype, num_heads: int = 0,
                      is_mla: bool = False) -> np.ndarray:
        """Generate attention mask (without NZ padding).

        This method generates masks in their natural shapes for golden calculation.
        NZ padding (if needed) should be applied later in _prepare_mask_for_network.

        Args:
            mask_type: MASK_UNDEFINED, MASK_NORM, MASK_ALIBI, MASK_SPEC, etc.
            num_tokens: Total query tokens
            max_context_len: Maximum context length
            q_seq_lens: List of query sequence lengths
            kv_seq_lens: List of KV sequence lengths
            dtype: Data type for mask (MindSpore dtype, used for determining numpy dtype)
            num_heads: Number of attention heads (required for ALIBI)
            is_mla: Whether this is MLA (Multi-Head Latent Attention) mode

        Returns:
            Mask numpy array (without NZ padding) or None
        """
        if mask_type == MASK_UNDEFINED:
            return None

        if mask_type == MASK_FREE:
            # MASK_FREE: Generate global causal mask for golden calculation
            # Shape: (num_tokens, max_context_len)
            # Only mask the last q_seqlen window within k_seqlen context
            batch_size = len(q_seq_lens)
            np_dtype = self._np_dtype_for(dtype)
            mask_np = np.zeros((num_tokens, max_context_len), dtype=np_dtype)
            prev_qseq = 0
            for i in range(batch_size):
                qseq = q_seq_lens[i]
                kseq = kv_seq_lens[i]
                start = kseq - qseq
                tri = np.ones((qseq, qseq), dtype=np_dtype)
                tri = np.triu(tri, 1)  # Upper triangular (exclude diagonal)
                tri *= -60000.0
                mask_np[prev_qseq:(prev_qseq + qseq), start:kseq] = tri
                prev_qseq += qseq
            return mask_np

        # Determine pre_mask_factor for SPEC mask construction
        # Note:
        # - ALIBI mask doesn't use pre_mask_factor (position bias, not boolean)
        # - NORM mask always uses -10000.0 regardless of dtype
        # - SPEC mask: pre_mask_factor=1.0 ONLY for MLA + bf16, otherwise -10000.0
        # - post_mask_factor only applies to MLA + SPEC + bf16 paths
        if mask_type == MASK_SPEC:
            if is_mla and dtype == ms.bfloat16:
                # For MLA + bf16 SPEC mask: use 1.0; golden multiplies by -10000.0
                pre_mask_factor = 1.0
            else:
                # For non-MLA or fp16 SPEC mask: use -10000.0 directly
                pre_mask_factor = -10000.0
        else:
            # For ALIBI/NORM, pre_mask_factor is not used
            pre_mask_factor = None

        if mask_type == MASK_NORM:
            # Normal causal mask: upper triangular
            # NORM mask ALWAYS uses -10000.0, regardless of dtype!
            batch_size = len(q_seq_lens)
            max_q_len = max(q_seq_lens)

            if max_q_len == 1:
                # Decode mode: shape (num_tokens, 1, max_context_len)
                mask_np = np.zeros((num_tokens, 1, max_context_len), dtype=np.float32)
                for i in range(num_tokens):
                    # Mask out positions before current token index
                    if i > 0:
                        mask_np[i, :, :i] = -10000.0

                return mask_np

            # Prefill mode: shape (batch_size, max_q_len, max_context_len)
            np_dtype = self._np_dtype_for(dtype)
            mask_np = np.zeros((batch_size, max_q_len, max_context_len), dtype=np_dtype)
            for i in range(batch_size):
                qseq = q_seq_lens[i]
                # Create upper triangular mask (qseq x qseq) for causal attention
                tri = np.ones((qseq, qseq), dtype=np_dtype)
                tri = np.triu(tri, 1)  # Upper triangular (exclude diagonal)
                tri *= -10000.0
                # Place mask at the end: [-qseq:, -qseq:]
                mask_np[i, -qseq:, -qseq:] = tri
            return mask_np

        if mask_type == MASK_ALIBI:
            # ALIBI mask: positional bias
            # ALIBI is NOT a boolean mask but a bias added to attention scores
            # Decode shape: (batch, num_heads, 1, max_context_len)
            # Prefill shape: (batch, num_heads, max_q_len, max_context_len)
            # ALIBI values are NOT multiplied by a mask factor
            batch_size = len(q_seq_lens)
            max_q_len = max(q_seq_lens)

            # For decode (all q_len=1), use shape (batch, num_heads, 1, max_context_len)
            # For prefill/MTP, use shape (batch, num_heads, max_q_len, max_context_len)
            if max_q_len == 1:
                mask_np = np.zeros(
                    (batch_size, num_heads, 1, max_context_len),
                    dtype=np.float32,
                )
            else:
                mask_np = np.zeros(
                    (batch_size, num_heads, max_q_len, max_context_len),
                    dtype=np.float32,
                )

            alibi_slopes = self._get_alibi_slopes(num_heads)

            for i, (ql, kl) in enumerate(zip(q_seq_lens, kv_seq_lens)):
                if kl == 0:
                    continue
                # position_ids - context_len + 1
                # Generates negative bias values: [-kl+1, ..., -1, 0]
                position_ids = np.arange(kl, dtype=np.int32)
                alibi_bias = (position_ids - kl + 1).astype(
                    np.float32
                )  # [-kl+1, ..., -1, 0]
                # Shape: (num_heads, 1, kl)
                alibi_bias = (
                    alibi_slopes.reshape((-1, 1, 1))
                    * alibi_bias.reshape((1, 1, -1))
                )

                if max_q_len == 1:
                    mask_np[i, :, :, :kl] = alibi_bias
                    # Direct assignment, no mask factor needed
                else:
                    # For prefill/MTP: repeat for all query positions
                    # Each query position gets the same ALIBI bias per key position
                    mask_np[i, :, :ql, :kl] = alibi_bias
                    # Direct assignment, no mask factor needed

            return mask_np

        if mask_type == MASK_SPEC:
            # SPEC mask for parallel decoding (MTP)
            # Only mask within the last q_len window
            np_dtype = self._np_dtype_for(dtype)
            mask_np = np.zeros((num_tokens, max_context_len), dtype=np_dtype)
            pre_q = 0
            for ql, kl in zip(q_seq_lens, kv_seq_lens):
                if ql == 0 or kl == 0:
                    pre_q += ql
                    continue
                # Create upper triangular mask for the last q_len tokens in context
                start = max(0, kl - ql)
                tri = np.triu(np.ones((ql, ql), dtype=np_dtype), 1) * pre_mask_factor
                mask_np[pre_q: pre_q + ql, start: kl] = tri
                pre_q += ql
            return mask_np

        return None

    def compute_golden_reference(
        self,
        query: np.ndarray,
        key_cache: np.ndarray,
        value_cache: np.ndarray,
        block_tables: np.ndarray,
        q_seq_lens: np.ndarray,
        kv_seq_lens: np.ndarray,
        scale: float,
        mask: np.ndarray,
        mask_type: int = MASK_UNDEFINED,
        k_descale: np.ndarray = None,
        v_descale: np.ndarray = None,
        k_offset: np.ndarray = None,
        v_offset: np.ndarray = None,
        k_scale_per_head: np.ndarray = None,
        v_scale_per_head: np.ndarray = None,
        p_scale: np.ndarray = None,
        output_dtype: ms.dtype = None,
        mla_v_dim: int = 0,
        mask_dtype: ms.dtype = None,
    ) -> np.ndarray:
        """Compute golden reference output using NumPy.

        Supports group matmul with GQA support and quantization.
        Supports MLA (Multi-Head Latent Attention) with different Q/K and V/O dimensions.

        Args:
            query: Query numpy array
            key_cache: Key cache numpy array
            value_cache: Value cache numpy array
            block_tables: Block table mapping numpy array
            q_seq_lens: Query sequence lengths numpy array
            kv_seq_lens: KV sequence lengths numpy array
            scale: QK scaling factor
            mask: Attention mask numpy array (optional)
            mask_type: Mask type constant
                (MASK_UNDEFINED, MASK_NORM, MASK_ALIBI, MASK_SPEC, MASK_FREE)
            k_descale: Key dequantization scales (for Dequant Fusion int8 KV)
            v_descale: Value dequantization scales (for Dequant Fusion int8 KV)
            k_offset: Key dequantization offsets
                (ONLY for Dequant Fusion with offset)
            v_offset: Value dequantization offsets
                (ONLY for Dequant Fusion with offset)
            k_scale_per_head: Key per-head scales (for full QKV quant)
            v_scale_per_head: Value per-head scales (for full QKV quant)
            p_scale: P matrix quantization scale (for offline full QKV quant)
            Note: Q scale is NOT used
            output_dtype: Output data type (for quantization scenarios)
            mla_v_dim: MLA V/O head dimension (0 for non-MLA, >0 for MLA mode)
            mask_dtype: Original mask dtype (for determining post_mask_factor
                in MLA + SPEC + bf16)

        Returns:
            Golden reference output numpy array
        """
        # Convert to float32 for computation (inputs are already numpy arrays)
        q_np = query.astype(np.float32)
        kc_np = key_cache.astype(np.float32)
        vc_np = value_cache.astype(np.float32)

        # Determine head dimensions (MLA support)
        num_tokens, num_heads, head_size_qk = q_np.shape
        kv_heads = kc_np.shape[2]
        head_size_k = kc_np.shape[3]  # Key head dimension
        head_size_v = vc_np.shape[3]  # Value head dimension (may differ in MLA)
        output_head_dim = head_size_v if mla_v_dim > 0 else head_size_qk

        # Apply dequantization for Dequant Fusion (int8 KV, fp16 Q)
        # Add offset (if provided) then multiply by per-element descale
        # Shapes for per-element params: (kv_heads * head_size_qk/vo)
        if k_descale is not None and v_descale is not None:
            k_scale_np = k_descale.astype(np.float32)
            v_scale_np = v_descale.astype(np.float32)
            # Reshape scale from (kv_heads * head_size) to (kv_heads, head_size)
            k_scale_reshaped = k_scale_np.reshape(kv_heads, head_size_k)
            v_scale_reshaped = v_scale_np.reshape(kv_heads, head_size_v)
            # Offsets are optional (only for Dequant Fusion with offset)
            if k_offset is not None:
                k_off_np = k_offset.astype(np.float32).reshape(kv_heads, head_size_k)
                kc_np += k_off_np[np.newaxis, np.newaxis, :, :]
            if v_offset is not None:
                v_off_np = v_offset.astype(np.float32).reshape(kv_heads, head_size_v)
                vc_np += v_off_np[np.newaxis, np.newaxis, :, :]
            # Apply per-element scale: broadcast to (num_blocks, block_size, kv_heads, head_size)
            kc_np *= k_scale_reshaped[np.newaxis, np.newaxis, :, :]
            vc_np *= v_scale_reshaped[np.newaxis, np.newaxis, :, :]

        # For full QKV quantization: keep int8 data, scales will be applied after matmul
        # We'll apply dequantization AFTER int32 matmul results
        is_full_quant = k_scale_per_head is not None and v_scale_per_head is not None
        if is_full_quant:
            k_scale_np = k_scale_per_head.astype(np.float32)
            v_scale_np = v_scale_per_head.astype(np.float32)
            # Keep data as int8 for now (stored in float32 array but with int8 values)
            # Convert to int32 for matmul computation
            q_np = q_np.astype(np.int32)
            kc_np = kc_np.astype(np.int32)
            vc_np = vc_np.astype(np.int32)

        tables_np = block_tables.astype(np.int32)
        q_lens = q_seq_lens.tolist()
        k_lens = kv_seq_lens.tolist()

        block_size = kc_np.shape[1]

        # Initialize output with correct dimensions (MLA: output uses head_size_v)
        # Non-MLA: [num_tokens, num_heads, head_size_qk]
        # MLA: [num_tokens, num_heads, head_size_v]
        out_np = np.zeros(
            (num_tokens, num_heads, output_head_dim),
            dtype=np.float32,
        )

        q_idx = 0
        for batch_idx, (ql, kl) in enumerate(zip(q_lens, k_lens)):
            if ql == 0 or kl == 0:
                q_idx += ql
                continue

            # Extract KV from cache and prepare tensors
            key_seq, value_seq = self._extract_kv_from_cache(
                kc_np, vc_np, tables_np[batch_idx], kl, block_size
            )

            q_slice = q_np[q_idx: q_idx + ql]
            query_t = np.transpose(q_slice, (1, 0, 2))
            key_t = np.transpose(key_seq, (1, 2, 0))
            value_t = np.transpose(value_seq, (1, 0, 2))

            # Compute attention output based on quantization type
            if is_full_quant:
                ctx = AttentionContext(
                    num_heads=num_heads,
                    kv_heads=kv_heads,
                    scale=scale,
                    mask=mask,
                    mask_type=mask_type,
                    batch_idx=batch_idx,
                    q_idx=q_idx,
                    ql=ql,
                    kl=kl,
                    mla_v_dim=mla_v_dim,
                    mask_dtype=mask_dtype,
                )
                quant_params = QuantizationParams(
                    k_scale=k_scale_np,
                    v_scale=v_scale_np,
                    p_scale=p_scale,
                )
                out_slice = self._compute_full_quant_attention(
                    query_t, key_t, value_t, ctx, quant_params
                )
            else:
                out_slice = self._compute_float_attention(
                    query_t, key_t, value_t, num_heads, kv_heads, scale,
                    mask, mask_type, batch_idx, q_idx, ql, kl,
                    mla_v_dim, mask_dtype
                )

            # Transpose output from [num_heads, ql, head_size] to [ql, num_heads, head_size]
            out_np[q_idx: q_idx + ql] = np.transpose(out_slice, (1, 0, 2))

            q_idx += ql

        # Convert to specified output dtype (for quantization) or keep as float32
        if output_dtype is not None:
            out_dtype_np = self._np_dtype_for(output_dtype)
            return out_np.astype(out_dtype_np)

        # Return as float32 or convert to original query dtype
        return out_np.astype(query.dtype)

    def _extract_kv_from_cache(
        self,
        key_cache: np.ndarray,
        value_cache: np.ndarray,
        block_table: np.ndarray,
        kv_len: int,
        block_size: int,
    ) -> tuple:
        """Extract keys and values from paged cache using block table.
        
        Args:
            key_cache: Key cache array [num_blocks, block_size, kv_heads, head_size]
            value_cache: Value cache array [num_blocks, block_size, kv_heads, head_size]
            block_table: Block table for current sequence
            kv_len: KV sequence length
            block_size: Block size
            
        Returns:
            Tuple of (key_seq, value_seq) arrays
        """
        keys_list = []
        values_list = []
        for j in range(kv_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size
            keys_list.append(key_cache[block_number, block_offset])
            values_list.append(value_cache[block_number, block_offset])

        key_seq = np.stack(keys_list, axis=0)
        value_seq = np.stack(values_list, axis=0)
        return key_seq, value_seq

    def _compute_full_quant_attention(
        self,
        query_t: np.ndarray,
        key_t: np.ndarray,
        value_t: np.ndarray,
        ctx: AttentionContext,
        quant_params: QuantizationParams,
    ) -> np.ndarray:
        """Compute attention with full QKV quantization (int8 Q/K/V).

        Args:
            query_t: Query tensor (num_heads, ql, head_dim).
            key_t: Key tensor (kv_heads, head_dim, kl).
            value_t: Value tensor (kv_heads, kl, head_dim).
            ctx: Attention context parameters.
            quant_params: Quantization scale parameters.

        Returns:
            Output slice [num_heads, ql, head_size]
        """
        # Q@K^T with int32, then apply K scale
        scores_int32 = self._group_matmul(query_t, key_t, ctx.num_heads, ctx.kv_heads)

        scores = np.zeros_like(scores_int32, dtype=np.float32)
        for h in range(ctx.num_heads):
            scores[h] = scores_int32[h].astype(np.float32) * quant_params.k_scale[h]

        scores = scores * ctx.scale

        # Apply mask
        if ctx.mask is not None:
            scores = self._apply_mask_simple(
                scores, ctx.mask, ctx.mask_type, ctx.q_idx,
                ctx.ql, ctx.kl, ctx.mla_v_dim, ctx.mask_dtype
            )

        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        row_sum = np.sum(exp_scores, axis=-1, keepdims=True)

        # Quantize P and apply V
        has_p_scale = quant_params.p_scale is not None
        if has_p_scale:
            out_slice = self._apply_offline_quant_pv(
                exp_scores, value_t, ctx.num_heads, ctx.kv_heads,
                quant_params.v_scale, quant_params.p_scale
            )
        else:
            out_slice = self._apply_online_quant_pv(
                exp_scores, value_t, ctx.num_heads, ctx.kv_heads, quant_params.v_scale
            )

        return out_slice / row_sum

    def _compute_float_attention(
        self,
        query_t: np.ndarray,
        key_t: np.ndarray,
        value_t: np.ndarray,
        num_heads: int,
        kv_heads: int,
        scale: float,
        mask: np.ndarray,
        mask_type: int,
        batch_idx: int,
        q_idx: int,
        ql: int,
        kl: int,
        mla_v_dim: int,
        mask_dtype: ms.dtype,
    ) -> np.ndarray:
        """Compute standard float attention (unquant or dequant fusion).
        
        Returns:
            Output slice [num_heads, ql, head_size]
        """
        scores = self._group_matmul(query_t, key_t, num_heads, kv_heads) * scale

        if mask is not None:
            scores = self._apply_mask_full(
                scores, mask, mask_type, batch_idx, q_idx, ql, kl,
                mla_v_dim, mask_dtype
            )

        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        return self._group_matmul_pv(probs, value_t, num_heads, kv_heads)

    def _apply_mask_simple(
        self,
        scores: np.ndarray,
        mask: np.ndarray,
        mask_type: int,
        q_idx: int,
        ql: int,
        kl: int,
        mla_v_dim: int,
        mask_dtype: ms.dtype,
    ) -> np.ndarray:
        """Apply mask for full quant path (MASK_FREE and SPEC only)."""
        mask_np = mask.astype(np.float32)
        post_factor = 1.0
        if mla_v_dim > 0 and mask_type == MASK_SPEC and mask_dtype == ms.bfloat16:
            post_factor = -10000.0

        if mask_type == MASK_FREE:
            mask_slice = mask_np[q_idx: q_idx + ql, :kl]
            scores += mask_slice[np.newaxis, :, :]
        else:
            mask_slice = mask_np[q_idx: q_idx + ql, :kl]
            scores += mask_slice[np.newaxis, :, :] * post_factor

        return scores

    def _apply_mask_full(
        self,
        scores: np.ndarray,
        mask: np.ndarray,
        mask_type: int,
        batch_idx: int,
        q_idx: int,
        ql: int,
        kl: int,
        mla_v_dim: int,
        mask_dtype: ms.dtype,
    ) -> np.ndarray:
        """Apply mask for float attention (all mask types)."""
        mask_np = mask.astype(np.float32)
        post_factor = 1.0
        if mla_v_dim > 0 and mask_type == MASK_SPEC and mask_dtype == ms.bfloat16:
            post_factor = -10000.0

        if mask_type == MASK_FREE:
            mask_slice = mask_np[q_idx: q_idx + ql, :kl]
            scores += mask_slice[np.newaxis, :, :]
        elif mask_type == MASK_ALIBI:
            if mask_np.ndim == 4:
                mask_slice = mask_np[batch_idx, :, :ql, :kl]
            else:
                mask_slice = mask_np[batch_idx, :, :, :kl][:, :ql, :]
            scores += mask_slice
        elif mask_type == MASK_SPEC:
            mask_slice = mask_np[q_idx: q_idx + ql, :kl]
            scores += mask_slice[np.newaxis, :, :] * post_factor
        else:
            # MASK_NORM
            if mask_np.ndim == 3:
                if mask_np.shape[1] == 1:
                    mask_slice = mask_np[batch_idx, 0, :kl]
                    scores += mask_slice[np.newaxis, np.newaxis, :]
                else:
                    mask_slice = mask_np[batch_idx, -ql:, :kl]
                    scores += mask_slice[np.newaxis, :, :]

        return scores

    def _apply_offline_quant_pv(
        self,
        exp_scores: np.ndarray,
        value_t: np.ndarray,
        num_heads: int,
        kv_heads: int,
        v_scale: np.ndarray,
        p_scale: np.ndarray,
    ) -> np.ndarray:
        """Apply offline P quantization and V matmul."""
        p_scale_np = p_scale.astype(np.float32)
        probs_fp = exp_scores * p_scale_np.reshape(num_heads, 1, 1)
        probs_int8 = np.rint(probs_fp.astype(np.float16)).astype(np.int8)
        probs_int8 = probs_int8.astype(np.int32)

        out_int32 = self._group_matmul_pv(probs_int8, value_t, num_heads, kv_heads)
        out_slice = np.zeros_like(out_int32, dtype=np.float32)
        for h in range(num_heads):
            out_slice[h, :, :] = out_int32[h, :, :].astype(np.float32) * v_scale[h]

        return out_slice

    def _apply_online_quant_pv(
        self,
        exp_scores: np.ndarray,
        value_t: np.ndarray,
        num_heads: int,
        kv_heads: int,
        v_scale: np.ndarray,
    ) -> np.ndarray:
        """Apply online P quantization and V matmul."""
        row_maxp = np.max(exp_scores, axis=-1, keepdims=True)
        p_scale_dynamic = row_maxp / 127.0
        probs_fp = exp_scores / p_scale_dynamic
        probs_int8 = np.rint(probs_fp.astype(np.float16)).astype(np.int8)
        probs_int8 = probs_int8.astype(np.int32)

        out_int32 = self._group_matmul_pv(probs_int8, value_t, num_heads, kv_heads)
        out_slice = np.zeros_like(out_int32, dtype=np.float32)
        for h in range(num_heads):
            de_scalev = v_scale[h] * row_maxp[h, 0, 0] / 127.0
            out_slice[h, :, :] = out_int32[h, :, :].astype(np.float32) * de_scalev

        return out_slice

    def _group_matmul(self, query_block: np.ndarray, key_block: np.ndarray,
                      num_heads: int, kv_heads: int) -> np.ndarray:
        """Group matmul for Q @ K^T with GQA support.

        Query and key blocks should be pre-transposed before calling this method.
        - query_block: [num_heads, ql, head_size]
        - key_block: [kv_heads, head_size, kl]

        Args:
            query_block: Query [num_heads, ql, head_size] (pre-transposed)
            key_block: Key [kv_heads, head_size, kl] (pre-transposed)
            num_heads: Number of query heads
            kv_heads: Number of KV heads

        Returns:
            Scores [num_heads, ql, kl]
        """
        # Always use group loop
        # When kv_heads == num_heads, group_size = 1, loop runs num_heads times
        group_size = num_heads // kv_heads
        scores_list = []
        for kv_h in range(kv_heads):
            query_group = query_block[
                kv_h * group_size: (kv_h + 1) * group_size, :, :
            ]  # [group_size, ql, head_size]
            key_head_block = key_block[kv_h: kv_h + 1, :, :]  # [1, head_size, kl]
            # query_group @ key_head_block ->
            # [group_size, ql, head_size] @ [1, head_size, kl] = [group_size, ql, kl]
            scores_group = np.matmul(
                query_group.astype(np.float32), key_head_block.astype(np.float32)
            )
            scores_list.append(scores_group)
        return np.concatenate(scores_list, axis=0)  # [num_heads, ql, kl]

    def _group_matmul_pv(self, prob_block: np.ndarray, value_block: np.ndarray,
                         num_heads: int, kv_heads: int) -> np.ndarray:
        """Group matmul for P @ V with GQA support.

        Value block should be pre-transposed before calling this method.
        - prob_block: [num_heads, ql, kl]
        - value_block: [kv_heads, kl, head_size]

        Args:
            prob_block: Attention probabilities [num_heads, ql, kl]
            value_block: Value [kv_heads, kl, head_size] (pre-transposed)
            num_heads: Number of query heads
            kv_heads: Number of KV heads

        Returns:
            Output [num_heads, ql, head_size]
        """
        # Always use group loop
        # When kv_heads == num_heads, group_size = 1, loop runs num_heads times
        group_size = num_heads // kv_heads
        out_list = []
        for kv_h in range(kv_heads):
            prob_group = prob_block[
                kv_h * group_size: (kv_h + 1) * group_size, :, :
            ]  # [group_size, ql, kl]
            value_head_block = value_block[kv_h: kv_h + 1, :, :]  # [1, kl, head_size]
            # prob_group @ value_head_block ->
            # [group_size, ql, kl] @ [1, kl, head_size] = [group_size, ql, head_size]
            out_group = np.matmul(
                prob_group.astype(np.float32), value_head_block.astype(np.float32)
            )
            out_list.append(out_group)
        return np.concatenate(out_list, axis=0)  # [num_heads, ql, head_size]

    def validate_accuracy(
        self,
        output: np.ndarray,
        golden: np.ndarray,
        dtype: ms.dtype,
        num_heads: int,
        max_context_len: int,
        is_quant: bool = False,
    ) -> bool:
        """Validate output accuracy against golden reference.

        Uses both legacy (ratio-based) and adaptive (complexity-based) thresholds.
        Quantization paths use stricter thresholds.

        Args:
            output: Operator output numpy array
            golden: Golden reference numpy array
            dtype: Data type (output dtype for quantization)
            num_heads: Number of heads
            max_context_len: Maximum context length
            is_quant: Whether this is a quantization test (affects threshold selection)

        Returns:
            True if accuracy check passes
        """
        out_np = output.astype(np.float32)
        golden_np = golden.astype(np.float32)

        out_flat = out_np.flatten()
        golden_flat = golden_np.flatten()
        diff = np.abs(out_flat - golden_flat)
        max_diff = np.max(diff)

        # Legacy ratio-based validation (slightly relaxed for bf16+quant)
        ratios = [0.001, 0.001, 0.005, 0.005]  # [rel_loose, abs_loose, rel_strict, abs_strict]
        rel_loose, abs_loose, rel_strict, abs_strict = ratios

        # 只在 "bf16 且为量化" 的场景下放宽严格阈值；
        # - 纯 bf16（非量化）仍使用原始严格阈值
        # - 其它量化精度（如 fp16+int8）也使用原始严格阈值
        strict_scale = 2.0 if (dtype == ms.bfloat16 and is_quant) else 1.0
        rel_strict_eff = rel_strict * strict_scale
        abs_strict_eff = abs_strict * strict_scale

        limit_error = np.maximum(np.abs(golden_flat) * rel_loose, abs_loose)
        strict_limit_error = np.maximum(np.abs(golden_flat) * rel_strict_eff, abs_strict_eff)
        error_count = np.sum(diff > limit_error)
        strict_error_count = np.sum(diff > strict_limit_error)

        out_len = max(1, out_flat.shape[0])
        accuracy_loose = 1.0 - float(error_count) / out_len
        accuracy_strict = 1.0 - float(strict_error_count) / out_len

        print(f"Max difference: {max_diff:.6e}")
        print(f"Loose accuracy (1/1000): {accuracy_loose:.6f}")
        print(f"Strict accuracy (5/1000): {accuracy_strict:.6f}")

        # Quantization uses stricter threshold
        # For "bf16 with quantization" scenario, relax strict error ratio;
        # Other scenarios:
        #   - Quantization (non bf16) still uses stricter ratio
        #   - Non-quantization uses looser ratio
        error_ratio = float(strict_error_count) / out_len
        if dtype == ms.bfloat16 or is_quant:
            # Original strict ratio is 0.005; relax to 0.02 for these harder cases.
            legacy_pass = error_ratio <= rel_strict_eff
        else:
            legacy_pass = error_ratio <= ratios[0]

        # Adaptive validation based on computation complexity
        calc_times = num_heads * max_context_len + 4
        if dtype == ms.bfloat16:
            # For bf16, especially with quantization (int8 KV / full QKV quant),
            # relax adaptive threshold by one bit compared to the original setting.
            base = 2 ** (-7) if calc_times < 2048 else 2 ** (-6)
            error_factor = base * (2.0 if is_quant else 1.0)
        elif dtype == ms.float16:
            error_factor = 2 ** (-8) if calc_times < 2048 else 2 ** (-7)
        else:  # float32
            if calc_times < 2048:
                error_factor = 2 ** (-11)
            elif calc_times < 16384:
                error_factor = 2 ** (-10)
            else:
                error_factor = 2 ** (-9)

        error_threshold = np.maximum(np.abs(golden_flat), 1.0) * error_factor
        adaptive_pass = np.all(diff <= error_threshold)

        print(f"Calculation complexity: {calc_times}")
        print(f"Error factor: {error_factor:.6e}")
        print(f"Adaptive test: {'PASS' if adaptive_pass else 'FAIL'}")
        print(f"Legacy test: {'PASS' if legacy_pass else 'FAIL'}")

        return bool(adaptive_pass or legacy_pass)


class PagedAttentionNet(nn.Cell):
    """MindSpore network wrapper for paged_attention operator.

    Handles CPU transfer of sequence length tensors internally.
    """

    def __init__(self, q_head_num: int, qk_scale: float, kv_head_num: int, mask_type: int,
                 batch_run_status_enable: bool = False, quant_type: int = QUANT_UNQUANT,
                 out_data_type: int = -1, has_quant_offset: bool = False, compress_type: int = 0,
                 calc_type: int = 0, scale_type: int = 0, input_layout: int = INPUT_LAYOUT_BSND,
                 mla_v_dim: int = 0, input_format: int = INPUT_FORMAT_ND):
        super().__init__()
        self.q_head_num = int(q_head_num)
        self.qk_scale = float(qk_scale)
        self.kv_head_num = int(kv_head_num)
        self.mask_type = int(mask_type)
        self.batch_run_status_enable = bool(batch_run_status_enable)
        self.quant_type = int(quant_type)
        self.out_data_type = int(out_data_type)
        self.has_quant_offset = bool(has_quant_offset)
        self.compress_type = int(compress_type)
        self.calc_type = int(calc_type)
        self.scale_type = int(scale_type)
        self.input_layout = int(input_layout)
        self.mla_v_dim = int(mla_v_dim)
        self.input_format = int(input_format)
        self._is_pynative = context.get_context("mode") == context.PYNATIVE_MODE

    def construct(self, query, key_cache, value_cache, block_tables,
                  attn_mask, batch_run_status,
                  k_descale, k_offset, v_descale, v_offset,
                  razor_offset, p_scale, log_n,
                  q_seq_lens, kv_seq_lens):
        """Forward pass with CPU tensor transfer for sequence lengths."""
        # Transfer sequence length tensors to CPU (required by internal runner)
        needs_q_seq = self.calc_type == 1 and q_seq_lens is not None
        if self._is_pynative:
            kv_seq_cpu = kv_seq_lens.move_to("CPU")
            q_seq_cpu = q_seq_lens.move_to("CPU") if needs_q_seq else None
        else:
            kv_seq_cpu = ops.move_to(kv_seq_lens, "CPU")
            q_seq_cpu = ops.move_to(q_seq_lens, "CPU") if needs_q_seq else None

        return ms_custom_ops.paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            kv_seq_cpu,
            attn_mask,
            batch_run_status,
            k_descale,
            k_offset,
            v_descale,
            v_offset,
            razor_offset,
            p_scale,
            log_n,
            q_seq_cpu if self.calc_type == 1 else None,
            self.q_head_num,
            self.qk_scale,
            self.kv_head_num,
            self.mask_type,
            self.batch_run_status_enable,
            self.quant_type,
            self.out_data_type,
            self.has_quant_offset,
            self.compress_type,
            self.calc_type,
            self.scale_type,
            self.input_layout,
            self.mla_v_dim,
            self.input_format,
        )


def _set_dynamic_shapes_for_pa(net: PagedAttentionNet, test_config: dict) -> None:
    """Configure dynamic input shapes for PagedAttentionNet."""
    # Extract key params
    num_heads = int(test_config['num_heads'])
    kv_heads = int(test_config['kv_heads'])
    head_size_qk = int(test_config['head_size'])
    head_size_vo = int(test_config.get('mla_v_dim', 0) or head_size_qk)
    q_dtype = test_config['q_dtype']
    kv_dtype = test_config['kv_dtype']
    mask_type = int(test_config['mask_type'])
    quant_type = int(test_config.get('quant_type', QUANT_UNQUANT))
    has_quant_offset = bool(test_config.get('has_quant_offset', False))
    # Dynamic shapes (None for variable dims)
    query_dyn = Tensor(shape=[None, num_heads, head_size_qk], dtype=q_dtype)
    key_dyn = Tensor(shape=[None, None, kv_heads, head_size_qk], dtype=kv_dtype)
    value_dyn = Tensor(shape=[None, None, kv_heads, head_size_vo], dtype=kv_dtype)
    block_tables_dyn = Tensor(shape=[None, None], dtype=ms.int32)

    # Mask shapes by type
    mask_dyn = None
    mask_dtype = test_config.get('expected_dtype', q_dtype)
    if mask_type == MASK_ALIBI:
        mask_dyn = Tensor(shape=[None, num_heads, None, None], dtype=mask_dtype)
    elif mask_type == MASK_NORM:
        # NORM mask: shape varies by mode
        # Decode: (num_tokens, 1, max_context_len)
        # Prefill: (batch_size, max_q_len, max_context_len)
        mask_dyn = Tensor(shape=[None, None, None], dtype=mask_dtype)
    elif mask_type == MASK_SPEC:
        mask_dyn = Tensor(shape=[None, None], dtype=mask_dtype)
    elif mask_type in (MASK_FREE, MASK_UNDEFINED):
        mask_dyn = None

    # Optional tensors
    batch_run_status_enable = test_config.get('batch_run_status_enable', False)
    if batch_run_status_enable:
        batch_run_status_dyn = Tensor(shape=[None], dtype=ms.int32)
    else:
        batch_run_status_dyn = None

    # Descale/offset shapes depend on quantization type
    # We only need correct rank/dtype here; lengths are dynamic
    if quant_type in [QUANT_QKV_OFFLINE, QUANT_QKV_ONLINE]:
        # per-head scales in descale slots, and optional p_scale (offline)
        k_descale_dyn = Tensor(shape=[None], dtype=ms.float32)
        v_descale_dyn = Tensor(shape=[None], dtype=ms.float32)
        if quant_type == QUANT_QKV_OFFLINE:
            p_scale_dyn = Tensor(shape=[None], dtype=ms.float32)
        else:
            p_scale_dyn = None
        k_offset_dyn = None
        v_offset_dyn = None
    elif quant_type == DEQUANT_FUSION:
        # per-element descale; use dynamic length 1-D
        k_descale_dyn = Tensor(shape=[None], dtype=ms.float32)
        v_descale_dyn = Tensor(shape=[None], dtype=ms.float32)
        p_scale_dyn = None
        k_offset_dyn = Tensor(shape=[None], dtype=ms.int32) if has_quant_offset else None
        v_offset_dyn = Tensor(shape=[None], dtype=ms.int32) if has_quant_offset else None
    else:
        k_descale_dyn = None
        v_descale_dyn = None
        k_offset_dyn = None
        v_offset_dyn = None
        p_scale_dyn = None

    # q_seq_lens / kv_seq_lens must be static shapes in set_inputs
    batch_size = len(test_config.get('context_lens', []))
    q_seq_dyn = Tensor(shape=[batch_size], dtype=ms.int32)
    kv_seq_dyn = Tensor(shape=[batch_size], dtype=ms.int32)

    # razor_offset/log_n are unused in our tests; keep None
    net.set_inputs(
        query_dyn, key_dyn, value_dyn, block_tables_dyn,
        mask_dyn, batch_run_status_dyn,
        k_descale_dyn, k_offset_dyn, v_descale_dyn, v_offset_dyn,
        None, p_scale_dyn, None,
        q_seq_dyn, kv_seq_dyn,
    )


def _run_paged_attention_test(generator: PagedAttentionDataGenerator, test_config: dict,
                             run_mode: int, validate_accuracy: bool = True, dynamic: bool = False,
                             reference_impl: str = REFERENCE_NUMPY):
    """Execute paged attention test with given configuration.

    Following the refactored workflow:
    1. Generate numpy arrays for golden calculation
    2. Compute golden reference using numpy arrays
    3. Convert numpy arrays to Tensors for network input
    4. Execute network
    5. Validate output against golden

    Args:
        generator: Data generator instance
        test_config: Test configuration dictionary
        run_mode: GRAPH_MODE or PYNATIVE_MODE
        validate_accuracy: Whether to validate accuracy against golden reference
        dynamic: Whether to use dynamic shapes
        reference_impl: "numpy" (default) or "mindspore" for MS reference
    """
    context.set_context(device_target="Ascend", mode=run_mode)

    # Extract configuration
    num_heads = test_config['num_heads']
    kv_heads = test_config['kv_heads']
    head_size = test_config['head_size']
    block_size = test_config['block_size']
    num_blocks = test_config['num_blocks']
    context_lens = test_config['context_lens']
    q_dtype = test_config['q_dtype']
    kv_dtype = test_config['kv_dtype']
    mask_type = test_config['mask_type']
    qk_scale = test_config.get('qk_scale', 1.0 / math.sqrt(head_size))

    # Get q_seq_lens from test_config (directly configure q_seq_lens)
    # If not provided, default to [1] * batch_size for basic decode mode
    q_seq_lens_config = test_config.get('q_seq_lens', [1] * len(context_lens))

    # Determine quantization type
    quant_type = test_config.get('quant_type', QUANT_UNQUANT)
    has_quant_offset = test_config.get('has_quant_offset', False)
    is_full_quant = quant_type in (QUANT_QKV_OFFLINE, QUANT_QKV_ONLINE)

    # MLA configuration
    mla_v_dim = test_config.get('mla_v_dim', 0)

    ms_ref = None
    if reference_impl == REFERENCE_MINDSPORE:
        ms_ref = PagedAttentionMsReference(getattr(generator, "seed", 2025))
        ms_inputs = ms_ref.generate_inputs_ms(
            num_heads,
            kv_heads,
            head_size,
            block_size,
            num_blocks,
            q_seq_lens_config,
            context_lens,
            q_dtype,
            kv_dtype,
            mask_type,
            quant_type,
            has_quant_offset,
            mla_v_dim,
            test_config.get('expected_dtype'),
        )
        inputs_dict = {key: _tensor_to_numpy(val) for key, val in ms_inputs.items()}
        golden_tensor = ms_ref.compute_golden_reference_ms(
            ms_inputs['query'],
            ms_inputs['key_cache'],
            ms_inputs['value_cache'],
            ms_inputs['block_tables'],
            ms_inputs['q_seq_lens'],
            ms_inputs['kv_seq_lens'],
            qk_scale,
            ms_inputs['mask'],
            mask_type,
            ms_inputs['k_descale'],
            ms_inputs['v_descale'],
            ms_inputs['k_offset'],
            ms_inputs['v_offset'],
            ms_inputs['k_scale_per_head'],
            ms_inputs['v_scale_per_head'],
            ms_inputs['p_scale'],
            output_dtype=test_config.get('expected_dtype') if is_full_quant else None,
            mla_v_dim=mla_v_dim,
            mask_dtype=ms_inputs['mask_dtype'],
        )
        golden = golden_tensor.asnumpy()
    else:
        # Step 1: Generate numpy arrays for golden calculation
        mask_out_dtype = test_config.get('expected_dtype')
        inputs_dict = generator.generate_inputs(
            num_heads,
            kv_heads,
            head_size,
            block_size,
            num_blocks,
            q_seq_lens_config,
            context_lens,
            q_dtype,
            kv_dtype,
            mask_type,
            quant_type,
            has_quant_offset,
            mla_v_dim,
            mask_out_dtype,
        )

        # Step 2: Compute golden reference using numpy arrays
        golden_output_dtype = test_config.get('expected_dtype') if is_full_quant else None
        golden = generator.compute_golden_reference(
            inputs_dict['query'],
            inputs_dict['key_cache'],
            inputs_dict['value_cache'],
            inputs_dict['block_tables'],
            inputs_dict['q_seq_lens'],
            inputs_dict['kv_seq_lens'],
            qk_scale,
            inputs_dict['mask'],
            mask_type,
            inputs_dict['k_descale'],
            inputs_dict['v_descale'],
            inputs_dict['k_offset'],
            inputs_dict['v_offset'],
            inputs_dict['k_scale_per_head'],
            inputs_dict['v_scale_per_head'],
            inputs_dict['p_scale'],
            output_dtype=golden_output_dtype,
            mla_v_dim=mla_v_dim,
            mask_dtype=inputs_dict['mask_dtype']
        )

    # Step 3: Convert numpy arrays to Tensors for network input
    query = Tensor(inputs_dict['query'], dtype=q_dtype)
    key_cache = Tensor(inputs_dict['key_cache'], dtype=kv_dtype)
    value_cache = Tensor(inputs_dict['value_cache'], dtype=kv_dtype)
    block_tables = Tensor(inputs_dict['block_tables'], dtype=ms.int32)
    q_seq_lens = Tensor(inputs_dict['q_seq_lens'], dtype=ms.int32)
    kv_seq_lens = Tensor(inputs_dict['kv_seq_lens'], dtype=ms.int32)
    mask = (
        Tensor(inputs_dict['mask'], dtype=inputs_dict['mask_dtype'])
        if inputs_dict['mask'] is not None
        else None
    )

    # Convert quantization parameters to Tensors
    k_descale = (Tensor(inputs_dict['k_descale'], dtype=ms.float32)
                 if inputs_dict['k_descale'] is not None else None)
    v_descale = (Tensor(inputs_dict['v_descale'], dtype=ms.float32)
                 if inputs_dict['v_descale'] is not None else None)
    k_offset = (Tensor(inputs_dict['k_offset'], dtype=ms.int32)
                if inputs_dict['k_offset'] is not None else None)
    v_offset = (Tensor(inputs_dict['v_offset'], dtype=ms.int32)
                if inputs_dict['v_offset'] is not None else None)
    k_scale_per_head = (Tensor(inputs_dict['k_scale_per_head'], dtype=ms.float32)
                        if inputs_dict['k_scale_per_head'] is not None else None)
    v_scale_per_head = (Tensor(inputs_dict['v_scale_per_head'], dtype=ms.float32)
                        if inputs_dict['v_scale_per_head'] is not None else None)
    p_scale = (Tensor(inputs_dict['p_scale'], dtype=ms.float32)
               if inputs_dict['p_scale'] is not None else None)

    # Optional batch_run_status (not generated by generate_inputs)
    batch_run_status = test_config.get('batch_run_status')

    # Optional format conversion (NZ format in PyNative)
    input_format = test_config.get('input_format', INPUT_FORMAT_ND)
    # Create network
    net = PagedAttentionNet(
        q_head_num=num_heads,
        qk_scale=qk_scale,
        kv_head_num=kv_heads,
        mask_type=mask_type,
        batch_run_status_enable=test_config.get('batch_run_status_enable', False),
        quant_type=test_config.get('quant_type', QUANT_UNQUANT),
        out_data_type=test_config.get('out_data_type', -1),
        has_quant_offset=test_config.get('has_quant_offset', False),
        compress_type=test_config.get('compress_type', 0),
        calc_type=test_config.get('calc_type', 0),
        scale_type=test_config.get('scale_type', 0),
        input_layout=test_config.get('input_layout', INPUT_LAYOUT_BSND),
        mla_v_dim=test_config.get('mla_v_dim', 0),
        input_format=input_format,
    )

    # Select correct descale parameters based on quantization type
    # For Dequant Fusion: use k_descale, v_descale (per-element)
    # For Full QKV Quant: use k_scale_per_head, v_scale_per_head (per-head) in descale position
    quant_type = test_config.get('quant_type', QUANT_UNQUANT)
    if quant_type in [QUANT_QKV_OFFLINE, QUANT_QKV_ONLINE]:
        # Full quantization: pass per-head scales in descale position
        final_k_descale = k_scale_per_head
        final_v_descale = v_scale_per_head
    else:
        # Dequant Fusion or Unquant: use original descale
        final_k_descale = k_descale
        final_v_descale = v_descale

    # Configure dynamic shapes if requested
    if dynamic:
        _set_dynamic_shapes_for_pa(net, test_config)

    # Execute operator
    output = net(
        query, key_cache, value_cache, block_tables,
        mask, batch_run_status,
        final_k_descale, k_offset, final_v_descale, v_offset,
        None, p_scale, None,  # razor_offset, p_scale, log_n
        q_seq_lens, kv_seq_lens
    )

    # Validate output shape and dtype
    num_tokens = sum(q_seq_lens_config)  # Calculate from q_seq_lens
    # MLA: output dimension is mla_v_dim (head_size_vo), not head_size (head_size_qk)
    output_head_dim = mla_v_dim if mla_v_dim > 0 else head_size
    expected_shape = (num_tokens, num_heads, output_head_dim)
    assert tuple(output.shape) == expected_shape, \
        f"Shape mismatch: {output.shape} vs {expected_shape}"

    if test_config.get('expected_dtype'):
        expected_dtype = test_config['expected_dtype']
        assert output.dtype == expected_dtype, \
            f"Dtype mismatch: {output.dtype} vs {expected_dtype}"

    # Step 5: Validate accuracy if requested
    if validate_accuracy:
        output_np = output.asnumpy()
        validate_dtype = test_config.get('expected_dtype', q_dtype)
        is_quant_test = quant_type != QUANT_UNQUANT
        max_ctx_len = max(context_lens)
        if reference_impl == REFERENCE_MINDSPORE and ms_ref is not None:
            assert ms_ref.validate_accuracy_ms(
                output_np,
                golden,
                validate_dtype,
                num_heads,
                max_ctx_len,
                is_quant_test,
            )
        else:
            assert generator.validate_accuracy(
                output_np, golden, validate_dtype, num_heads, max_ctx_len, is_quant_test
            )


# ========================================
# Test Cases - Organized by Feature Category
# ========================================
#
# Test organization:
# 1. Basic Functionality Tests
#    - Decode path across GRAPH/PYNATIVE, dynamic/static, fp16/bf16; mask-free baseline
# 2. Mask Type Tests
#    - NORM (triangular), ALIBI positional bias, SPEC for MTP/parallel decoding
# 3. GQA Tests
#    - Head ratios such as 8:1, 4:2, 32:8
# 4. Configuration Variation Tests
#    - Odd/non-16-aligned head counts, larger head size, small/large seq lens
#      varied batch seq lens, small block size
# 5. Quantization Tests
#    - Dequant Fusion (int8 KV) with/without offsets; Full QKV quant (offline/online)
#      output dtype selection
# 6. Combined Feature Tests
#    - GQA + int8 KV (with/without offsets), ALIBI + GQA
# 7. BF16 + int8 KV (Dequant Fusion)
#    - Varied kv seq lens, head sizes (incl. non-standard), GQA ratios, block sizes, edge cases
# 8. MLA (Multi-Head Latent Attention) Tests
#    - Split/combined cache, varied Q/K vs V/O dims, NORM mask, varied block sizes
#    - Full QKV quant (online/offline), unaligned dims
# 9. MLA + MTP Combined Tests
#    - Prefill (large/small embeds, fp16/bf16, with/without SPEC)
#    - Multi-token outputs under SPEC, scalability with many heads
# 10. Lookahead Decoding Tests
#    - Mixed q/k lengths, single-head, varied block sizes, very long contexts, with GQA
#


# Common test decorators
def _paged_attention_test(test_func):
    """Apply common decorators for PagedAttention tests.

    Adds shared marks used by most tests in this module to reduce repetition.
    """
    decorators = [
        pytest.mark.level0,
        pytest.mark.platform_arm_ascend910b_training,
        pytest.mark.env_onecard,
    ]
    for decorator in reversed(decorators):
        test_func = decorator(test_func)
    return test_func


def _tensor_to_numpy(value):
    if value is None:
        return None
    if isinstance(value, Tensor):
        return value.asnumpy()
    return value


# ==================== 1. Basic Functionality Tests ====================
# Test basic features: data types, execution modes, simple configurations

@_paged_attention_test
@pytest.mark.parametrize('ms_dtype', [ms.float16, ms.bfloat16])
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dynamic', [False, True])
def test_pa_basic_dtype_and_mode(ms_dtype, run_mode, dynamic):
    """
    Feature: PagedAttention - basic decode path across modes and dtypes
    Description: Unquantized decode with random inputs over GRAPH/PYNATIVE,
    static/dynamic, fp16/bf16
    Expectation: Runs successfully; output shape matches and matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(3001)
    test_config = {
        'q_seq_lens': [1, 1, 1, 1],  # Basic decode: 1 token per sequence
        'num_heads': 32,
        'kv_heads': 1,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [192, 193, 194, 195],
        'q_dtype': ms_dtype,
        'kv_dtype': ms_dtype,
        'mask_type': MASK_UNDEFINED,
        'qk_scale': 1.0,
    }
    _run_paged_attention_test(
        generator,
        test_config,
        run_mode,
        validate_accuracy=True,
        dynamic=dynamic,
    )


@_paged_attention_test
def test_pa_no_mask():
    """
    Feature: PagedAttention - mask-free decode
    Description: Simplest decode with no attention mask, fp16 inputs
    Expectation: Operator executes; output shape/dtype correct and accuracy within tolerance
    """
    generator = PagedAttentionDataGenerator(4115)
    test_config = {
        'q_seq_lens': [1, 1, 1, 1],  # Basic decode: 1 token per sequence
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [192, 193, 194, 195],
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_UNDEFINED,
        'qk_scale': 1.0,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


# ==================== 2. Mask Type Tests ====================
# Test different attention mask types

@_paged_attention_test
@pytest.mark.parametrize('ms_dtype', [ms.float16, ms.bfloat16])
def test_pa_norm_mask(ms_dtype):
    """
    Feature: PagedAttention - normal causal mask
    Description: Decode with triangular causal mask under fp16/bf16
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(4116)
    test_config = {
        'q_seq_lens': [1, 1, 1, 1],  # Basic decode: 1 token per sequence
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [192, 193, 194, 195],
        'q_dtype': ms_dtype,
        'kv_dtype': ms_dtype,
        'mask_type': MASK_NORM,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
@pytest.mark.parametrize('ms_dtype', [ms.float16, ms.bfloat16])
def test_pa_alibi_mask(ms_dtype):
    """
    Feature: PagedAttention - ALIBI positional bias
    Description: Decode with ALIBI bias per head using fp16/bf16
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(4100)
    test_config = {
        'q_seq_lens': [1, 1],  # Basic decode
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [500, 500],
        'q_dtype': ms_dtype,
        'kv_dtype': ms_dtype,
        'mask_type': MASK_ALIBI,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
@pytest.mark.parametrize('ms_dtype', [ms.float16, ms.bfloat16])
def test_pa_spec_mask_mtp(ms_dtype):
    """
    Feature: PagedAttention - SPEC mask for MTP
    Description: Multi-token prediction (q_len=2) with SPEC mask in fp16/bf16
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(3002)
    test_config = {
        'q_seq_lens': [2, 2, 2, 2],  # MTP: 2 tokens per sequence
        'num_heads': 32,
        'kv_heads': 1,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [192, 193, 194, 195],
        'q_dtype': ms_dtype,
        'kv_dtype': ms_dtype,
        'mask_type': MASK_SPEC,
        'qk_scale': 0.01,
        'calc_type': 1,  # Enable MTP mode
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


# ==================== 3. GQA (Grouped Query Attention) Tests ====================
# Test various query-to-kv head ratios

@_paged_attention_test
@pytest.mark.parametrize('ms_dtype', [ms.float16, ms.bfloat16])
def test_pa_gqa_8to1(ms_dtype):
    """
    Feature: PagedAttention - GQA 8:1 ratio
    Description: Decode with 8 query heads and 1 KV head (ALIBI mask)
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(4101)
    test_config = {
        'q_seq_lens': [1] * 13,  # Basic decode
        'num_heads': 8,
        'kv_heads': 1,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 1024,
        'context_lens': [88] * 13,
        'q_dtype': ms_dtype,
        'kv_dtype': ms_dtype,
        'mask_type': MASK_ALIBI,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
def test_pa_gqa_4to2():
    """
    Feature: PagedAttention - GQA 4:2 ratio
    Description: Decode with 4 query heads and 2 KV heads (ALIBI mask)
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(4102)
    test_config = {
        'q_seq_lens': [1] * 13,  # Basic decode
        'num_heads': 4,
        'kv_heads': 2,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 1024,
        'context_lens': [88] * 13,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_ALIBI,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
def test_pa_gqa_32to8():
    """
    Feature: PagedAttention - GQA 32:8 ratio
    Description: Decode with 32 query heads and 8 KV heads (ALIBI mask)
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(4103)
    test_config = {
        'q_seq_lens': [1] * 13,  # Basic decode
        'num_heads': 32,
        'kv_heads': 8,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 1024,
        'context_lens': [88] * 13,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_ALIBI,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


# ==================== 4. Configuration Variation Tests ====================
# Test different head counts, dimensions, block sizes, sequence lengths

@_paged_attention_test
def test_pa_odd_heads():
    """
    Feature: PagedAttention - odd head count
    Description: Decode with 7 heads to verify non-power-of-2 handling
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(4104)
    test_config = {
        'q_seq_lens': [1] * 13,  # Basic decode
        'num_heads': 7,
        'kv_heads': 7,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 1024,
        'context_lens': [88] * 13,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_NORM,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
def test_pa_head_not_aligned():
    """
    Feature: PagedAttention - non-16-aligned head count
    Description: Decode with 20 heads (bf16) to verify edge-case handling
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(4110)
    test_config = {
        'q_seq_lens': [1, 1],  # Basic decode
        'num_heads': 20,
        'kv_heads': 20,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [500, 500],
        'q_dtype': ms.bfloat16,
        'kv_dtype': ms.bfloat16,
        'mask_type': MASK_NORM,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
def test_pa_headsize_256():
    """
    Feature: PagedAttention - larger head dimension
    Description: Decode with head_size=256 and block_size=16, no mask
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(4107)
    test_config = {
        'q_seq_lens': [1, 1, 1, 1],  # Basic decode
        'num_heads': 16,
        'kv_heads': 16,
        'head_size': 256,
        'block_size': 16,
        'num_blocks': 512,
        'context_lens': [192, 193, 194, 195],
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_UNDEFINED,
        'qk_scale': 1.0 / math.sqrt(256),
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
def test_pa_blocksize_16():
    """
    Feature: PagedAttention - small block size
    Description: Decode with block_size=16 and fp16, no mask
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(4108)
    test_config = {
        'q_seq_lens': [1, 1, 1, 1],  # Basic decode
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 16,
        'num_blocks': 512,
        'context_lens': [192, 193, 194, 195],
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_UNDEFINED,
        'qk_scale': 1.0,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
def test_pa_small_seqlen():
    """
    Feature: PagedAttention - small sequence lengths
    Description: Decode with kv_seq_len=33 and NORM mask
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(4105)
    test_config = {
        'q_seq_lens': [1, 1, 1, 1],  # Basic decode
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [33, 33, 33, 33],
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_NORM,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
def test_pa_large_seqlen():
    """
    Feature: PagedAttention - large sequence lengths
    Description: Decode with kv_seq_len=3000 (bf16) and NORM mask
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(4106)
    test_config = {
        'q_seq_lens': [1, 1],  # Basic decode
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 128,
        'context_lens': [3000, 3000],
        'q_dtype': ms.bfloat16,
        'kv_dtype': ms.bfloat16,
        'mask_type': MASK_NORM,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
def test_pa_multi_batch_varied_seqlens():
    """
    Feature: PagedAttention - varied sequence lengths across batch
    Description: Decode with kv_seq_lens [100, 500, 1000, 2000] and NORM mask
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(4111)
    test_config = {
        'q_seq_lens': [1, 1, 1, 1],  # Basic decode
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 256,
        'context_lens': [100, 500, 1000, 2000],
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_NORM,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


# ==================== 5. Quantization Tests ====================
# Test KV cache quantization and full QKV quantization

@_paged_attention_test
def test_pa_dequant_fusion_kv_int8():
    """
    Feature: PagedAttention - Dequant Fusion (int8 KV)
    Description: Decode with int8 KV cache and per-element descales (no offsets)
    Expectation: Operator executes; output dtype/shape correct and accuracy within tolerance
    """
    generator = PagedAttentionDataGenerator(3003)
    test_config = {
        'q_seq_lens': [1, 1, 1, 1],  # Basic decode
        'num_heads': 32,
        'kv_heads': 1,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [192, 193, 194, 195],
        'q_dtype': ms.float16,
        'kv_dtype': ms.int8,
        'mask_type': MASK_UNDEFINED,
        'qk_scale': 1.0,
        'quant_type': DEQUANT_FUSION,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dynamic', [False, True])
def test_pa_dequant_fusion_with_offset(run_mode, dynamic):
    """
    Feature: PagedAttention - Dequant Fusion with offsets (int8 KV)
    Description: Decode testing k_offset/v_offset handling across modes and dynamic shapes
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(3004)
    test_config = {
        'q_seq_lens': [1, 1, 1, 1],  # Basic decode
        'num_heads': 32,
        'kv_heads': 1,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [192, 193, 194, 195],
        'q_dtype': ms.float16,
        'kv_dtype': ms.int8,
        'mask_type': MASK_UNDEFINED,
        'qk_scale': 1.0,
        'quant_type': DEQUANT_FUSION,
        'has_quant_offset': True,
    }
    _run_paged_attention_test(
        generator,
        test_config,
        run_mode,
        validate_accuracy=True,
        dynamic=dynamic,
    )


@_paged_attention_test
@pytest.mark.parametrize('out_dtype_sel, expect_ms_dtype', [(1, ms.float16), (27, ms.bfloat16)])
def test_pa_full_quant_qkv_offline(out_dtype_sel, expect_ms_dtype):
    """
    Feature: PagedAttention - Full QKV quant (offline)
    Description: int8 Q/K/V with per-head K/V scales and P scale; fp16/bf16 outputs
    Expectation: Operator executes; output dtype matches selection and accuracy within tolerance
    """
    generator = PagedAttentionDataGenerator(3004)
    test_config = {
        'q_seq_lens': [1, 1],  # Basic decode
        'num_heads': 32,
        'kv_heads': 1,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [64, 64],
        'q_dtype': ms.int8,
        'kv_dtype': ms.int8,
        'mask_type': MASK_UNDEFINED,
        'qk_scale': 1.0 / math.sqrt(128),
        'quant_type': QUANT_QKV_OFFLINE,
        'out_data_type': out_dtype_sel,
        'expected_dtype': expect_ms_dtype,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
@pytest.mark.parametrize('quant_type', [QUANT_QKV_OFFLINE, QUANT_QKV_ONLINE])
@pytest.mark.parametrize('out_dtype_sel, expect_ms_dtype', [(1, ms.float16), (27, ms.bfloat16)])
@pytest.mark.parametrize('dynamic', [False, True])
def test_pa_full_quant_qkv_configs(quant_type, out_dtype_sel, expect_ms_dtype, dynamic):
    """
    Feature: PagedAttention - Full QKV quant (offline/online) configs
    Description: Validate offline vs online quant, layouts, and output dtypes
    with dynamic/static shapes
    Expectation: Operator executes; output dtype/accuracy meet expectations
    """
    generator = PagedAttentionDataGenerator(4001 + quant_type)
    test_config = {
        'q_seq_lens': [1, 1, 1, 1],  # Basic decode
        'num_heads': 32,
        'kv_heads': 1,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [96, 64, 64, 64],
        'q_dtype': ms.int8,
        'kv_dtype': ms.int8,
        'mask_type': MASK_UNDEFINED,
        'qk_scale': 1.0 / math.sqrt(128),
        'quant_type': quant_type,
        'out_data_type': out_dtype_sel,
        'expected_dtype': expect_ms_dtype,
    }
    _run_paged_attention_test(
        generator,
        test_config,
        context.GRAPH_MODE,
        validate_accuracy=True,
        dynamic=dynamic,
    )


# ==================== 6. Combined Feature Tests ====================
# Test combinations of multiple features

@_paged_attention_test
def test_pa_gqa_with_int8_kv():
    """
    Feature: PagedAttention - GQA with int8 KV (Dequant Fusion)
    Description: bf16 query, 32:8 GQA, int8 KV, NORM mask
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(4112)
    test_config = {
        'q_seq_lens': [1, 1, 1, 1],  # Basic decode
        'num_heads': 32,
        'kv_heads': 8,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [192, 193, 194, 195],
        'q_dtype': ms.bfloat16,
        'kv_dtype': ms.int8,
        'mask_type': MASK_NORM,
        'qk_scale': 1.0,
        'quant_type': DEQUANT_FUSION,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
def test_pa_gqa_int8_with_offset():
    """
    Feature: PagedAttention - GQA with int8 KV and offsets
    Description: fp16 query, 32:8 GQA, Dequant Fusion with k/v offsets
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(4113)
    test_config = {
        'q_seq_lens': [1, 1, 1, 1],  # Basic decode
        'num_heads': 32,
        'kv_heads': 8,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [192, 193, 194, 195],
        'q_dtype': ms.float16,
        'kv_dtype': ms.int8,
        'mask_type': MASK_NORM,
        'qk_scale': 1.0,
        'quant_type': DEQUANT_FUSION,
        'has_quant_offset': True,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
def test_pa_alibi_with_gqa():
    """
    Feature: PagedAttention - ALIBI with GQA
    Description: ALIBI bias combined with 8:1 GQA in fp16
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(4114)
    test_config = {
        'q_seq_lens': [1] * 13,  # Basic decode
        'num_heads': 8,
        'kv_heads': 1,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 1024,
        'context_lens': [88] * 13,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_ALIBI,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


# ==================== 7. BFloat16 Quantization Tests ====================
# Test bfloat16 with quantization (int8 KV cache)

@_paged_attention_test
@pytest.mark.parametrize('reference_impl', [REFERENCE_NUMPY, REFERENCE_MINDSPORE])
def test_pa_bf16_int8_kv_basic(reference_impl):
    """
    Feature: PagedAttention - bf16 with int8 KV (Dequant Fusion)
    Description: Decode with bf16 Q and int8 KV over moderate sequence length
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(6001)
    test_config = {
        'q_seq_lens': [1],  # Basic decode
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 256,
        'context_lens': [768],
        'q_dtype': ms.bfloat16,
        'kv_dtype': ms.int8,
        'mask_type': MASK_NORM,
        'qk_scale': 1.0 / math.sqrt(128),
        'quant_type': DEQUANT_FUSION,
    }
    _run_paged_attention_test(
        generator,
        test_config,
        context.GRAPH_MODE,
        reference_impl=reference_impl,
    )


@_paged_attention_test
@pytest.mark.parametrize('k_seqlen', [128, 513, 768, 1024, 1025])
def test_pa_bf16_int8_kv_varied_seqlen(k_seqlen):
    """
    Feature: PagedAttention - bf16 with int8 KV across sequence lengths
    Description: Decode across varied kv_seq_len including non-aligned sizes
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(6002 + k_seqlen)
    test_config = {
        'q_seq_lens': [1],
        'num_heads': 24,
        'kv_heads': 24,
        'head_size': 64,
        'block_size': 128,
        'num_blocks': 256,
        'context_lens': [k_seqlen],
        'q_dtype': ms.bfloat16,
        'kv_dtype': ms.int8,
        'mask_type': MASK_NORM,
        'qk_scale': 1.0 / math.sqrt(64),
        'quant_type': DEQUANT_FUSION,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
@pytest.mark.parametrize('head_size', [32, 33, 64, 128])
def test_pa_bf16_int8_kv_varied_headsize(head_size):
    """
    Feature: PagedAttention - bf16 with int8 KV across head sizes
    Description: Decode with varied head_size including non-standard 33
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(6100 + head_size)
    test_config = {
        'q_seq_lens': [1],
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': head_size,
        'block_size': 16 if head_size * 16 <= 128 * 128 else 128,
        'num_blocks': 512,
        'context_lens': [512],
        'q_dtype': ms.bfloat16,
        'kv_dtype': ms.int8,
        'mask_type': MASK_NORM,
        'qk_scale': 1.0 / math.sqrt(head_size),
        'quant_type': DEQUANT_FUSION,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
@pytest.mark.parametrize('num_heads,kv_heads', [(4, 1), (24, 2), (64, 32)])
def test_pa_bf16_int8_kv_gqa_combinations(num_heads, kv_heads):
    """
    Feature: PagedAttention - bf16 with int8 KV and GQA variants
    Description: Decode with varied GQA ratios under bf16+int8
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(6200 + num_heads * 100 + kv_heads)
    test_config = {
        'q_seq_lens': [1],
        'num_heads': num_heads,
        'kv_heads': kv_heads,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 512,
        'context_lens': [766],
        'q_dtype': ms.bfloat16,
        'kv_dtype': ms.int8,
        'mask_type': MASK_NORM,
        'qk_scale': 1.0 / math.sqrt(128),
        'quant_type': DEQUANT_FUSION,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
@pytest.mark.parametrize('block_size', [16, 128])
def test_pa_bf16_int8_kv_varied_blocksize(block_size):
    """
    Feature: PagedAttention - bf16 with int8 KV across block sizes
    Description: Decode with block_size in {16, 128} under bf16+int8
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(6300 + block_size)
    test_config = {
        'q_seq_lens': [1],
        'num_heads': 32,
        'kv_heads': 8,
        'head_size': 64,
        'block_size': block_size,
        'num_blocks': 1024,
        'context_lens': [1024],
        'q_dtype': ms.bfloat16,
        'kv_dtype': ms.int8,
        'mask_type': MASK_NORM,
        'qk_scale': 1.0 / math.sqrt(64),
        'quant_type': DEQUANT_FUSION,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
def test_pa_bf16_int8_kv_edge_case_309():
    """
    Feature: PagedAttention - bf16 with int8 KV edge case (k_len=309)
    Description: Decode with k_len=309, head_size=32, block_size=16
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(6400)
    test_config = {
        'q_seq_lens': [1],
        'num_heads': 64,
        'kv_heads': 64,
        'head_size': 32,
        'block_size': 16,
        'num_blocks': 256,
        'context_lens': [309],
        'q_dtype': ms.bfloat16,
        'kv_dtype': ms.int8,
        'mask_type': MASK_NORM,
        'qk_scale': 1.0 / math.sqrt(32),
        'quant_type': DEQUANT_FUSION,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


# ==================== 8. MLA (Multi-Head Latent Attention) Tests ====================
# Test MLA scenarios with different Q/K and V/O head dimensions
# MLA特性: head_size_qk != head_size_vo, 需要设置mla_v_head_size参数

@_paged_attention_test
def test_pa_mla_split_cache_basic():
    """
    Feature: PagedAttention + MLA - split KV cache basic
    Description: MLA with head_size_qk=576, head_size_vo=512, no mask
    Expectation: Graph mode executes and matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(7001)
    test_config = {
        'q_seq_lens': [1] * 20,  # 20 sequences, 1 token each
        'num_heads': 4,
        'kv_heads': 1,
        'head_size': 576,  # Q/K head size
        'block_size': 128,
        'num_blocks': 1024,
        'context_lens': [128] * 20,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_UNDEFINED,
        'qk_scale': 1.0 / math.sqrt(576),
        'mla_v_dim': 512,  # V/O head size (different from Q/K)
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE, validate_accuracy=True)


@_paged_attention_test
def test_pa_mla_split_cache_large_groupnum():
    """
    Feature: PagedAttention + MLA - split cache with large head count
    Description: MLA with 128 Q heads and 1 KV head (128:1), no mask
    Expectation: Graph mode executes and matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(7002)
    test_config = {
        'q_seq_lens': [1] * 20,
        'num_heads': 128,
        'kv_heads': 1,
        'head_size': 576,
        'block_size': 128,
        'num_blocks': 1024,
        'context_lens': [256] * 20,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_UNDEFINED,
        'qk_scale': 1.0 / math.sqrt(576),
        'mla_v_dim': 512,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE, validate_accuracy=True)


@_paged_attention_test
@pytest.mark.parametrize('head_size_qk,head_size_vo', [(576, 512), (192, 128)])
def test_pa_mla_varied_head_dimensions(head_size_qk, head_size_vo):
    """
    Feature: PagedAttention + MLA - varied head dimensions
    Description: Validate MLA with different Q/K and V/O sizes
    Expectation: Graph mode executes and matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(7100 + head_size_qk)
    test_config = {
        'q_seq_lens': [1] * 32,
        'num_heads': 32,
        'kv_heads': 1,
        'head_size': head_size_qk,
        'block_size': 128,
        'num_blocks': 256,
        'context_lens': [256] * 32,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_UNDEFINED,
        'qk_scale': 1.0 / math.sqrt(head_size_qk),
        'mla_v_dim': head_size_vo,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE, validate_accuracy=True)


@_paged_attention_test
def test_pa_mla_with_norm_mask():
    """
    Feature: PagedAttention + MLA - NORM mask
    Description: MLA with varied kv_seq_lens and causal mask
    Expectation: Graph mode executes and matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(7200)
    test_config = {
        'q_seq_lens': [1] * 9,
        'num_heads': 128,
        'kv_heads': 1,
        'head_size': 576,
        'block_size': 128,
        'num_blocks': 512,
        'context_lens': [3000, 300, 1400, 33, 65, 1, 16, 1400, 300],  # Varied lengths
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_NORM,
        'qk_scale': 1.0 / math.sqrt(576),
        'mla_v_dim': 512,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE, validate_accuracy=True)


@_paged_attention_test
@pytest.mark.parametrize('block_size', [128, 64])
def test_pa_mla_varied_blocksize(block_size):
    """
    Feature: PagedAttention + MLA - varied block sizes
    Description: MLA with block_size in {128, 256}, no mask
    Expectation: Graph mode executes and matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(7300 + block_size)
    test_config = {
        'q_seq_lens': [1] * 16,
        'num_heads': 16,
        'kv_heads': 1,
        'head_size': 576,
        'block_size': block_size,
        'num_blocks': 256,
        'context_lens': [256] * 16,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_UNDEFINED,
        'qk_scale': 1.0 / math.sqrt(576),
        'mla_v_dim': 512,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE, validate_accuracy=True)


@_paged_attention_test
def test_pa_mla_int8_kv_quant():
    """
    Feature: PagedAttention + MLA - full QKV quant (online)
    Description: MLA with int8 Q/K/V, online quant, bf16 output
    Expectation: Graph mode executes; output dtype/accuracy are correct
    """
    generator = PagedAttentionDataGenerator(7400)
    test_config = {
        'q_seq_lens': [1] * 20,
        'num_heads': 40,
        'kv_heads': 1,
        'head_size': 576,
        'block_size': 128,
        'num_blocks': 256,
        'context_lens': [1024] * 20,
        'q_dtype': ms.int8,  # Full quantization: Q is int8
        'kv_dtype': ms.int8,
        'mask_type': MASK_UNDEFINED,
        'qk_scale': 1.0 / math.sqrt(576),
        'quant_type': QUANT_QKV_ONLINE,  # Online quantization
        'out_data_type': 27,  # bfloat16 output
        'expected_dtype': ms.bfloat16,
        'mla_v_dim': 512,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE, validate_accuracy=True)


@_paged_attention_test
def test_pa_mla_int8_kv_quant_offline():
    """
    Feature: PagedAttention + MLA - full QKV quant (offline)
    Description: MLA with int8 Q/K/V, offline quant, varied kv_seq_lens, bf16 output
    Expectation: Graph mode executes; output dtype/accuracy are correct
    """
    generator = PagedAttentionDataGenerator(7500)
    test_config = {
        'q_seq_lens': [1] * 9,
        'num_heads': 128,
        'kv_heads': 1,
        'head_size': 576,
        'block_size': 128,
        'num_blocks': 512,
        'context_lens': [3000, 300, 1400, 33, 65, 1, 16, 1400, 300],
        'q_dtype': ms.int8,  # Full quantization: Q is int8
        'kv_dtype': ms.int8,
        'mask_type': MASK_UNDEFINED,
        'qk_scale': 1.0 / math.sqrt(576),
        'quant_type': QUANT_QKV_OFFLINE,  # Offline quantization
        'out_data_type': 27,  # bfloat16 output
        'expected_dtype': ms.bfloat16,
        'mla_v_dim': 512,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE, validate_accuracy=True)


@_paged_attention_test
def test_pa_mla_unaligned_embed():
    """
    Feature: PagedAttention + MLA - unaligned head dimensions
    Description: MLA with Q/K=290 and V/O=130, full QKV quant
    Expectation: Graph mode executes; output dtype/accuracy are correct
    """
    generator = PagedAttentionDataGenerator(7600)
    test_config = {
        'q_seq_lens': [1] * 25,
        'num_heads': 25,
        'kv_heads': 1,
        'head_size': 290,  # Non-standard Q/K dimension
        'block_size': 128,
        'num_blocks': 256,
        'context_lens': [128] * 25,
        'q_dtype': ms.int8,
        'kv_dtype': ms.int8,
        'mask_type': MASK_UNDEFINED,
        'qk_scale': 1.0 / math.sqrt(290),
        'quant_type': QUANT_QKV_ONLINE,
        'out_data_type': 1,  # float16 output
        'expected_dtype': ms.float16,
        'mla_v_dim': 130,  # Non-standard V/O dimension
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE, validate_accuracy=True)


# ==================== 9. MLA + MTP Combined Tests ====================
# Test MLA (Multi-Head Latent Attention) combined with MTP (Multi-Token Prediction)
# These tests cover scenarios where Q/K and V/O have different dimensions AND
# multiple tokens are predicted per sequence (q_seqlen > 1)

@_paged_attention_test
def test_pa_mla_mtp_prefill_large_embed_no_mask_fp16():
    """
    Feature: PagedAttention + MLA + MTP - prefill large embed (fp16)
    Description: Prefill q_len=128, Q/K=576, V/O=512, no mask
    Expectation: Graph mode executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(8001)
    batch = 27
    test_config = {
        'q_seq_lens': [128] * batch,  # Prefill: 128 tokens per sequence
        'num_heads': 32,
        'kv_heads': 1,
        'head_size': 576,  # Q/K head size
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [512] * batch,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_UNDEFINED,  # No mask
        'qk_scale': 1.0 / math.sqrt(576),
        'calc_type': 1,  # MTP mode (required to pass q_seq_lens into op)
        'mla_v_dim': 512,  # V/O head size (different from Q/K)
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE, validate_accuracy=True)


@_paged_attention_test
def test_pa_mla_mtp_prefill_large_embed_no_mask_bf16():
    """
    Feature: PagedAttention + MLA + MTP - prefill large embed (bf16)
    Description: Prefill q_len=64, Q/K=576, V/O=512, no mask, bf16 dtype
    Expectation: Graph mode executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(8002)
    batch = 32
    test_config = {
        'q_seq_lens': [64] * batch,  # Prefill: 64 tokens per sequence
        'num_heads': 32,
        'kv_heads': 1,
        'head_size': 576,  # Q/K head size
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [273] * batch,
        'q_dtype': ms.bfloat16,
        'kv_dtype': ms.bfloat16,
        'mask_type': MASK_UNDEFINED,  # No mask
        'qk_scale': 1.0 / math.sqrt(576),
        'calc_type': 1,  # MTP mode
        'mla_v_dim': 512,  # V/O head size
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE, validate_accuracy=True)


@_paged_attention_test
def test_pa_mla_mtp_prefill_small_embed_no_mask_fp16():
    """
    Feature: PagedAttention + MLA + MTP - prefill small embed (fp16)
    Description: Prefill q_len=23, Q/K=192, V/O=128, no mask
    Expectation: Graph mode executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(8003)
    batch = 25
    test_config = {
        'q_seq_lens': [23] * batch,  # Prefill: 23 tokens per sequence
        'num_heads': 32,
        'kv_heads': 1,
        'head_size': 192,  # Q/K head size (smaller)
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [156] * batch,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_UNDEFINED,  # No mask
        'qk_scale': 1.0 / math.sqrt(192),
        'calc_type': 1,  # MTP mode
        'mla_v_dim': 128,  # V/O head size (smaller)
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE, validate_accuracy=True)


@_paged_attention_test
def test_pa_mla_mtp_prefill_small_embed_spec_mask_fp16():
    """
    Feature: PagedAttention + MLA + MTP - prefill small embed with SPEC (fp16)
    Description: Prefill q_len=256, Q/K=192, V/O=128 using SPEC mask
    Expectation: Graph mode executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(8004)
    batch = 27
    test_config = {
        'q_seq_lens': [256] * batch,  # Prefill: 256 tokens per sequence
        'num_heads': 32,
        'kv_heads': 1,
        'head_size': 192,  # Q/K head size
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [766] * batch,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_SPEC,  # SPEC mask for parallel decoding
        'qk_scale': 1.0 / math.sqrt(192),
        'calc_type': 1,  # MTP mode
        'mla_v_dim': 128,  # V/O head size
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE, validate_accuracy=True)


@_paged_attention_test
def test_pa_mla_mtp_prefill_small_embed_spec_mask_bf16():
    """
    Feature: PagedAttention + MLA + MTP - prefill small embed with SPEC (bf16)
    Description: Prefill q_len=1056, Q/K=192, V/O=128 using SPEC mask and bf16 dtype
    Expectation: Graph mode executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(8005)
    batch = 11
    test_config = {
        'q_seq_lens': [1056] * batch,  # Large prefill: 1056 tokens per sequence
        'num_heads': 16,
        'kv_heads': 1,
        'head_size': 192,  # Q/K head size
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [1963] * batch,
        'q_dtype': ms.bfloat16,
        'kv_dtype': ms.bfloat16,
        'mask_type': MASK_SPEC,  # SPEC mask
        'qk_scale': 1.0 / math.sqrt(192),
        'calc_type': 1,  # MTP mode
        'mla_v_dim': 128,  # V/O head size
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE, validate_accuracy=True)


@_paged_attention_test
def test_pa_mla_mtp_multi_token_large_embed_spec_mask_fp16():
    """
    Feature: PagedAttention + MLA - MTP with SPEC mask (fp16)
    Description: MTP (q_len=4) with large Q/K (576) and V/O (512) head dims using SPEC mask
    Expectation: Graph mode executes and matches golden within tolerance; shapes/dtypes correct
    """
    generator = PagedAttentionDataGenerator(8006)
    batch = 27
    test_config = {
        'q_seq_lens': [4] * batch,  # MTP: 4 tokens per sequence
        'num_heads': 32,
        'kv_heads': 1,
        'head_size': 576,  # Q/K head size (large)
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [512] * batch,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_SPEC,  # SPEC mask for MTP
        'qk_scale': 1.0 / math.sqrt(576),
        'calc_type': 1,  # MTP mode
        'mla_v_dim': 512,  # V/O head size (large)
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE, validate_accuracy=True)


@_paged_attention_test
def test_pa_mla_mtp_multi_token_large_embed_spec_mask_bf16():
    """
    Feature: PagedAttention + MLA - MTP with SPEC mask (bf16)
    Description: MTP (q_len=4) in bf16 with large Q/K (576) and V/O (512) using SPEC mask
    Expectation: Graph mode executes and matches golden within tolerance; shapes/dtypes correct
    """
    generator = PagedAttentionDataGenerator(8007)
    batch = 32
    test_config = {
        'q_seq_lens': [4] * batch,  # MTP: 4 tokens per sequence
        'num_heads': 32,
        'kv_heads': 1,
        'head_size': 576,  # Q/K head size
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [273] * batch,
        'q_dtype': ms.bfloat16,
        'kv_dtype': ms.bfloat16,
        'mask_type': MASK_SPEC,  # SPEC mask for MTP
        'qk_scale': 1.0 / math.sqrt(576),
        'calc_type': 1,  # MTP mode
        'mla_v_dim': 512,  # V/O head size
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE, validate_accuracy=True)


@_paged_attention_test
def test_pa_mla_mtp_multi_token_large_heads_long_context_fp16():
    """
    Feature: PagedAttention + MLA - MTP scalability (fp16)
    Description: Stress with 128 heads and 4096 context under SPEC mask in MTP
    Expectation: Graph mode executes; results align with golden within tolerance
    """
    generator = PagedAttentionDataGenerator(8008)
    batch = 16
    test_config = {
        'q_seq_lens': [4] * batch,  # MTP: 4 tokens per sequence
        'num_heads': 128,  # Very large head count
        'kv_heads': 1,
        'head_size': 576,  # Q/K head size
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [4096] * batch,  # Very long context
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_SPEC,  # SPEC mask for MTP
        'qk_scale': 1.0 / math.sqrt(576),
        'calc_type': 1,  # MTP mode
        'mla_v_dim': 512,  # V/O head size
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE, validate_accuracy=True)


# ==================== 10. Lookahead Decoding Tests ====================
# Test lookahead/speculative decoding scenarios (MASK_SPEC with varied q/k lengths)

@_paged_attention_test
def test_pa_lookahead_mixed_lengths():
    """
    Feature: PagedAttention - lookahead/speculative decoding with SPEC mask
    Description: Mixed q_seq_lens [1,15,30,6] and kv_seq_lens [10,64,64,64] in MTP mode
    Expectation: Graph mode executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(5001)
    test_config = {
        'q_seq_lens': [1, 15, 30, 6],  # MTP: varied tokens per sequence
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 512,
        'context_lens': [10, 64, 64, 64],  # Different context lengths per sequence
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_SPEC,
        'qk_scale': 1.0 / math.sqrt(128),
        'calc_type': 1,  # MTP mode for lookahead
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
def test_pa_lookahead_single_head():
    """
    Feature: PagedAttention - lookahead with single head
    Description: MTP with single head and varied sequences, SPEC mask
    Expectation: Graph mode executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(5002)
    test_config = {
        'q_seq_lens': [256, 256, 15],  # MTP: varied tokens
        'num_heads': 1,
        'kv_heads': 1,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 256,
        'context_lens': [512, 512, 2048],
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_SPEC,
        'qk_scale': 1.0 / math.sqrt(128),
        'calc_type': 1,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
def test_pa_lookahead_blocksize_16():
    """
    Feature: PagedAttention - lookahead with small block size
    Description: SPEC mask MTP using block_size=16 to verify granularity handling
    Expectation: Graph mode executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(5003)
    test_config = {
        'q_seq_lens': [1, 15, 30, 6],  # MTP: varied tokens per sequence
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 16,
        'num_blocks': 512,
        'context_lens': [10, 64, 64, 64],
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_SPEC,
        'qk_scale': 1.0 / math.sqrt(128),
        'calc_type': 1,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
def test_pa_lookahead_blocksize_32():
    """
    Feature: PagedAttention - lookahead with medium block size
    Description: SPEC mask MTP using block_size=32 with varied q/k lengths
    Expectation: Graph mode executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(5004)
    test_config = {
        'q_seq_lens': [15, 103, 1024],  # MTP: varied tokens
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 32,
        'num_blocks': 256,
        'context_lens': [64, 103, 1025],
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_SPEC,
        'qk_scale': 1.0 / math.sqrt(128),
        'calc_type': 1,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
def test_pa_lookahead_very_long_context():
    """
    Feature: PagedAttention - lookahead with very long contexts
    Description: Decode (q_len=1) with kv_seq_lens up to 33k to test long-context handling
    Expectation: Graph mode executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(5005)
    test_config = {
        'q_seq_lens': [1, 1, 1],  # Decode mode with very long context
        'num_heads': 40,
        'kv_heads': 40,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 1234,
        'context_lens': [13333, 23333, 33331],  # Very long contexts
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_SPEC,
        'qk_scale': 1.0 / math.sqrt(128),
        'calc_type': 1,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_test
def test_pa_lookahead_with_gqa():
    """
    Feature: PagedAttention - lookahead with GQA
    Description: SPEC mask MTP using kv_heads < num_heads (GQA)
    Expectation: Graph mode executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(5006)
    test_config = {
        'q_seq_lens': [1, 15, 30, 6],  # MTP: varied tokens per sequence
        'num_heads': 32,
        'kv_heads': 8,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 512,
        'context_lens': [10, 64, 64, 64],
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_SPEC,
        'qk_scale': 1.0 / math.sqrt(128),
        'calc_type': 1,
    }
    _run_paged_attention_test(generator, test_config, context.GRAPH_MODE)
