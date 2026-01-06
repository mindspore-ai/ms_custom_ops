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

"""System tests for paged_attention operator on Ascend 310P with NZ format.

This module tests the paged_attention operator on 310P hardware with NZ data format.
NZ format conversion (ND->NZ) is performed within the graph using trans_data operator.

Supported Features (310P NZ path):
- Basic decode with fp16
- Mask types: UNDEFINED, NORM, ALIBI, SPEC, FREE
- Grouped Query Attention (GQA)
- Multi-Token Prediction (MTP) / Lookahead decoding
- Prefill scenarios

Unsupported Features (310P NZ path):
- Dequant Fusion / Full QKV quantization (INT8 KV cache)
- MLA (Multi-Head Latent Attention) - use dedicated MLA operator instead
"""

import math
from typing import Dict, Optional

import numpy as np
import pytest
from test_custom_paged_attention import (
    PagedAttentionDataGenerator,
    MASK_UNDEFINED,
    MASK_NORM,
    MASK_ALIBI,
    MASK_SPEC,
    MASK_FREE,
    QUANT_UNQUANT,
    QUANT_QKV_OFFLINE,
    QUANT_QKV_ONLINE,
    INPUT_LAYOUT_BSND,
    INPUT_FORMAT_NZ,
)
import mindspore as ms
from mindspore import Tensor, context, nn, ops
import ms_custom_ops

# ========== Constants ==========

# Trans_data operation types
_TRANSDATA_ND_TO_NZ = 1  # Convert ND format to NZ format
_TRANSDATA_NZ_TO_ND = 0  # Convert NZ format to ND format

# ========== Helper Functions ==========


def _select_descale_tensors(
    quant_type: int,
    k_descale: Optional[Tensor],
    v_descale: Optional[Tensor],
    k_scale_per_head: Optional[Tensor],
    v_scale_per_head: Optional[Tensor]
) -> tuple:
    """Select appropriate descale tensors based on quantization type.

    Args:
        quant_type: Quantization type
        k_descale: Per-element K descale (for DEQUANT_FUSION)
        v_descale: Per-element V descale (for DEQUANT_FUSION)
        k_scale_per_head: Per-head K scale (for QUANT_QKV_*)
        v_scale_per_head: Per-head V scale (for QUANT_QKV_*)

    Returns:
        Tuple of (selected_k_descale, selected_v_descale)
    """
    if quant_type in (QUANT_QKV_OFFLINE, QUANT_QKV_ONLINE):
        return k_scale_per_head, v_scale_per_head
    return k_descale, v_descale


def _check_310p_capability(test_config: Dict) -> None:
    """Check if test configuration is supported on 310P NZ path.

    Raises:
        pytest.skip: If configuration uses unsupported features
    """
    quant_type = test_config.get('quant_type', QUANT_UNQUANT)
    has_quant_offset = test_config.get('has_quant_offset', False)
    mla_v_dim = test_config.get('mla_v_dim', 0)

    # Check quantization support
    is_full_quant = quant_type in (QUANT_QKV_OFFLINE, QUANT_QKV_ONLINE)
    if quant_type != QUANT_UNQUANT or is_full_quant or has_quant_offset:
        pytest.skip(
            "310P paged_attention does not support quantization on NZ path "
            "(Dequant Fusion/Full Quant/Offsets)."
        )

    # Check MLA support
    if int(mla_v_dim or 0) > 0:
        pytest.skip(
            "310P paged_attention does not support MLA on NZ path. "
            "Use dedicated MLA operator for MLA functionality."
        )


# ========== Network Definition ==========

class PagedAttentionNzNet(nn.Cell):
    """MindSpore network wrapper for paged_attention with NZ format conversion.

    This network:
    1) Converts ND format inputs to NZ format via trans_data
    2) Executes paged_attention
    3) Converts NZ output back to ND format

    Platform-specific handling:
    - 310P (NZ): context_lens stays on NPU as tensor input
    - 910B (ND): context_lens moved to CPU and passed via param

    Note: No manual padding/reshape is required; trans_data and kernel handle layout/alignment internally.
    """

    def __init__(
        self,
        q_head_num: int,
        qk_scale: float,
        kv_head_num: int,
        mask_type: int,
        batch_run_status_enable: bool = False,
        quant_type: int = QUANT_UNQUANT,
        out_data_type: int = -1,
        has_quant_offset: bool = False,
        compress_type: int = 0,
        calc_type: int = 0,
        scale_type: int = 0,
        input_layout: int = INPUT_LAYOUT_BSND,
        mla_v_dim: int = 0,
        input_format: int = INPUT_FORMAT_NZ,
    ):
        """Initialize PagedAttentionNzNet.

        Args:
            q_head_num: Number of query heads
            qk_scale: QK scale factor
            kv_head_num: Number of KV heads (for GQA)
            mask_type: Attention mask type
            batch_run_status_enable: Enable batch run status
            quant_type: Quantization type
            out_data_type: Output data type override
            has_quant_offset: Whether quantization uses offsets
            compress_type: Compression type
            calc_type: Calculation type (0=decode, 1=MTP)
            scale_type: Scale type
            input_layout: Input tensor layout
            mla_v_dim: MLA V dimension (0 means no MLA)
            input_format: Input format (0=ND, 1=NZ)
        """
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

    def construct(
        self,
        query_2d: Tensor,
        key_cache_3d: Tensor,
        value_cache_3d: Tensor,
        block_tables: Tensor,
        attn_mask_nd: Optional[Tensor],
        batch_run_status: Optional[Tensor],
        k_descale: Optional[Tensor],
        k_offset: Optional[Tensor],
        v_descale: Optional[Tensor],
        v_offset: Optional[Tensor],
        razor_offset: Optional[Tensor],
        p_scale: Optional[Tensor],
        log_n: Optional[Tensor],
        q_seq_lens: Optional[Tensor],
        kv_seq_lens: Tensor,
    ) -> Tensor:
        """Forward pass with ND->NZ conversion and paged attention execution.

        This method performs the following transformations:
        1. Query: (T, H*D) -> trans_data -> NZ format
        2. KV Cache: (B, S, KH*D) -> trans_data -> NZ format
        3. Mask: ND format -> trans_data -> NZ format (if present)
        4. Execute paged_attention with NZ inputs
        5. Output: NZ format -> trans_data -> (T, H*D) -> (T, H, D)

        Note: Input tensors are expected to be already reshaped to 2D/3D format.
        Use _prepare_inputs_for_nz_network() to prepare inputs before calling this network.

        Args:
            query_2d: Query tensor in 2D format (tokens, num_heads * head_size)
            key_cache_3d: Key cache tensor in 3D format (num_blocks, block_size, kv_heads * head_size_qk)
            value_cache_3d: Value cache tensor in 3D format (num_blocks, block_size, kv_heads * head_size_vo)
            block_tables: Block table mapping (batch_size, max_blocks_per_seq)
            attn_mask_nd: Attention mask in ND format (optional)
            batch_run_status: Batch run status (optional)
            k_descale: Key descale factors (optional)
            k_offset: Key quantization offsets (optional)
            v_descale: Value descale factors (optional)
            v_offset: Value quantization offsets (optional)
            razor_offset: Razor offset (unused)
            p_scale: P scale factor (optional)
            log_n: Log N tensor (unused)
            q_seq_lens: Query sequence lengths (optional, for MTP)
            kv_seq_lens: KV sequence lengths (context lengths)

        Returns:
            Output tensor in ND format (tokens, num_heads, head_size)
        """
        # Step 1: Ensure query is contiguous for trans_data
        # query_2d = query_2d.contiguous()

        # Step 2: Convert ND to NZ format (2D/3D ND -> 4D NZ)
        query_nz = ms_custom_ops.trans_data(query_2d, transdata_type=_TRANSDATA_ND_TO_NZ)
        key_cache_nz = ms_custom_ops.trans_data(key_cache_3d, transdata_type=_TRANSDATA_ND_TO_NZ)
        value_cache_nz = ms_custom_ops.trans_data(value_cache_3d, transdata_type=_TRANSDATA_ND_TO_NZ)

        # Step 3: Convert mask if present (ND -> NZ via trans_data)
        # Masks are in ND format with proper shapes:
        # - ALIBI: (batch*num_heads, 16, max_context_len_pad)
        # - NORM: (num_tokens, 16, max_context_len_pad) for decode or (batch, max_q_len, max_context_len) for prefill
        # - SPEC: (num_tokens, max_context_len)
        # - MASK_FREE: (1, 128, 128)
        attn_mask_nz = None
        if attn_mask_nd is not None:
            attn_mask_nz = ms_custom_ops.trans_data(attn_mask_nd, transdata_type=_TRANSDATA_ND_TO_NZ)

        # Step 4: Handle seq_lens based on platform
        # 310P NZ: kv_seq_lens stays on NPU, q_seq_lens moves to CPU
        kv_seq_input = kv_seq_lens  # Keep on NPU for 310P

        q_seq_input = None
        if self.calc_type == 1 and q_seq_lens is not None:
            # MTP mode: move q_seq_lens to CPU
            if self._is_pynative:
                q_seq_input = q_seq_lens.move_to("CPU")
            else:
                q_seq_input = ops.move_to(q_seq_lens, "CPU")

        # Step 5: Execute paged attention with NZ format
        out_nz = ms_custom_ops.paged_attention(
            query_nz, key_cache_nz, value_cache_nz, block_tables, kv_seq_input,
            attn_mask_nz, batch_run_status,
            k_descale, k_offset, v_descale, v_offset,
            razor_offset, p_scale, log_n,
            q_seq_input,
            self.q_head_num, self.qk_scale, self.kv_head_num, self.mask_type,
            self.batch_run_status_enable, self.quant_type, self.out_data_type,
            self.has_quant_offset, self.compress_type, self.calc_type,
            self.scale_type, self.input_layout, self.mla_v_dim, self.input_format
        )

        # Step 6: Convert output from NZ back to ND format
        out_nd_2d = ms_custom_ops.trans_data(out_nz, transdata_type=_TRANSDATA_NZ_TO_ND)

        # Step 7: Reshape output from 2D to 3D TND format
        # Calculate token count and output head size from query_2d shape
        token_count = query_2d.shape[0]
        output_head_size = self.mla_v_dim if self.mla_v_dim > 0 else (query_2d.shape[1] // self.q_head_num)
        # (tokens, H*D_out) -> (tokens, H_out, D_out)
        out_nd = ops.reshape(out_nd_2d, (token_count, self.q_head_num, output_head_size))

        return out_nd


# ========== Test Execution ==========

def _prepare_inputs_for_nz_network(
    query: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    num_heads: int,
    kv_heads: int,
    q_dtype: ms.dtype,
    kv_dtype: ms.dtype
) -> tuple:
    """Prepare query and KV cache tensors for NZ network input.

    This function reshapes numpy arrays to the format expected by PagedAttentionNzNet:
    - Query: (tokens, num_heads, head_size) -> (tokens, num_heads * head_size)
    - Key cache: (num_blocks, block_size, kv_heads, head_size_qk) -> (num_blocks, block_size, kv_heads * head_size_qk)
    - Value cache: (num_blocks, block_size, kv_heads, head_size_vo) -> (num_blocks, block_size, kv_heads * head_size_vo)

    Args:
        query: Query numpy array (tokens, num_heads, head_size)
        key_cache: Key cache numpy array (num_blocks, block_size, kv_heads, head_size_qk)
        value_cache: Value cache numpy array (num_blocks, block_size, kv_heads, head_size_vo)
        num_heads: Number of query heads
        kv_heads: Number of KV heads
        q_dtype: Query dtype
        kv_dtype: KV cache dtype

    Returns:
        Tuple of (query_2d_tensor, key_cache_3d_tensor, value_cache_3d_tensor)
    """
    # Reshape query: (tokens, num_heads, head_size) -> (tokens, num_heads * head_size)
    tokens = query.shape[0]
    head_size = query.shape[2]
    query_2d = query.reshape(tokens, num_heads * head_size)

    # Reshape key cache: (num_blocks, block_size, kv_heads, head_size_qk) ->
    # (num_blocks, block_size, kv_heads * head_size_qk)
    num_blocks = key_cache.shape[0]
    block_size = key_cache.shape[1]
    head_size_qk = key_cache.shape[3]
    key_cache_3d = key_cache.reshape(num_blocks, block_size, kv_heads * head_size_qk)

    # Reshape value cache: (num_blocks, block_size, kv_heads, head_size_vo) ->
    # (num_blocks, block_size, kv_heads * head_size_vo)
    head_size_vo = value_cache.shape[3]
    value_cache_3d = value_cache.reshape(num_blocks, block_size, kv_heads * head_size_vo)

    # Convert to Tensors
    query_tensor = Tensor(query_2d, dtype=q_dtype)
    key_cache_tensor = Tensor(key_cache_3d, dtype=kv_dtype)
    value_cache_tensor = Tensor(value_cache_3d, dtype=kv_dtype)

    return query_tensor, key_cache_tensor, value_cache_tensor


def _prepare_mask_for_network(
    mask: Optional[np.ndarray],
    mask_type: int,
    num_tokens: int,
    batch_size: int,
    num_heads: int,
    context_lens: list,
    q_seq_lens: list,
    dtype: ms.dtype
) -> tuple:
    """Prepare mask tensor for network input (ND format with NZ padding, ready for trans_data).

    This function handles:
    1. NZ padding for MASK_NORM and MASK_ALIBI (16-alignment)
    2. Reshape for ALIBI: (batch, num_heads, q_len, kv_len) -> (batch*num_heads, q_len, kv_len)
    3. Special handling for MASK_FREE and MASK_UNDEFINED

    Args:
        mask: Original mask numpy array (without NZ padding, or None)
        mask_type: Mask type constant
        num_tokens: Total number of query tokens
        batch_size: Batch size
        num_heads: Number of query heads
        context_lens: Context lengths for each sequence
        q_seq_lens: Query sequence lengths for each sequence
        dtype: Target dtype

    Returns:
        Tuple of (prepared_mask, effective_mask_type)
        - prepared_mask: Mask tensor in ND format with NZ padding (or None)
        - effective_mask_type: Actual mask type to use (may differ from input for MASK_UNDEFINED)
    """
    effective_mask_type = mask_type
    np_dtype = ms.dtype_to_nptype(dtype)
    max_context_len = max(context_lens)
    max_q_len = max(q_seq_lens)

    # Handle different mask types
    if mask_type == MASK_ALIBI:
        # ALIBI: Apply NZ padding then reshape
        if mask is None:
            return None, effective_mask_type

        # Step 1: Apply NZ padding (16-alignment)
        # Original shape: (batch, num_heads, q_len, max_context_len)
        # Padded shape: (batch, num_heads, 16, max_context_len_pad)
        max_context_len_pad = ((max_context_len + 15) // 16) * 16
        q_len_for_pad = 1 if max_q_len == 1 else max_q_len
        mask_padded = np.zeros((batch_size, num_heads, 16, max_context_len_pad), dtype=np_dtype)
        mask_padded[:, :, :q_len_for_pad, :max_context_len] = mask

        # Step 2: Reshape from (batch, num_heads, 16, max_context_len_pad) to
        # (batch*num_heads, 16, max_context_len_pad)
        mask_np = mask_padded.reshape((batch_size * num_heads, 16, max_context_len_pad))
        return Tensor(mask_np, dtype=dtype), effective_mask_type

    if mask_type == MASK_FREE:
        # MASK_FREE: create fixed 128x128 upper triangular mask
        mask_size = 128
        mask_np = np.zeros((mask_size, mask_size), dtype=np.float16)
        mask_np[np.triu_indices(mask_size, k=1)] = 1.0
        mask_np = mask_np * -60000.0
        # Reshape to 3D (1, 128, 128) before trans_data
        mask_nd = mask_np.reshape((1, mask_size, mask_size))
        return Tensor(mask_nd, dtype=dtype), effective_mask_type

    if mask_type == MASK_NORM:
        # NORM: Apply NZ padding
        if mask is None:
            return None, effective_mask_type

        if max_q_len == 1:
            # Decode mode: Apply NZ padding
            # Original shape: (num_tokens, 1, max_context_len)
            # Padded shape: (num_tokens, 16, max_context_len_pad)
            max_context_len_pad = ((max_context_len + 15) // 16) * 16
            mask_padded = np.zeros((num_tokens, 16, max_context_len_pad), dtype=np_dtype)
            mask_padded[:, :1, :max_context_len] = mask
            return Tensor(mask_padded, dtype=dtype), effective_mask_type
        # Prefill mode: no padding needed (already in correct shape)
        return Tensor(mask, dtype=dtype), effective_mask_type

    if mask_type == MASK_SPEC:
        # SPEC: keep original shape (no padding needed)
        if mask is None:
            return None, effective_mask_type
        return Tensor(mask, dtype=dtype), effective_mask_type

    # MASK_UNDEFINED
    # For decode mode (q_len=1), create zero-filled NORM mask to force masked kernel path
    # This ensures NZ padding columns don't contribute to softmax denominator
    return None, effective_mask_type


def _run_paged_attention_nz_test(
    generator: PagedAttentionDataGenerator,
    test_config: Dict,
    run_mode: int,
    validate_accuracy: bool = True,
    dynamic: bool = False,
) -> None:
    """Execute paged attention test with NZ format conversion.

    Following the pattern of test_custom_flash_attention_encoder_nz.py:
    1. Generate numpy arrays for golden calculation (no padding/reshaping)
    2. Compute golden reference using numpy arrays
    3. Reshape/prepare tensors for network input (ND format)
    4. Network internally converts ND->NZ via trans_data
    5. Validate output against golden

    Args:
        generator: Data generator for inputs and golden reference
        test_config: Test configuration dictionary
        run_mode: Execution mode (GRAPH_MODE or PYNATIVE_MODE)
        validate_accuracy: Whether to validate against golden reference
        dynamic: Whether to use dynamic shapes

    Raises:
        AssertionError: If output shape/dtype/accuracy validation fails
        pytest.skip: If configuration uses unsupported 310P features
    """
    context.set_context(device_target="Ascend", mode=run_mode)

    # Check 310P capability constraints
    _check_310p_capability(test_config)

    # Extract test parameters
    num_heads = test_config['num_heads']
    kv_heads = test_config['kv_heads']
    head_size = test_config['head_size']
    q_seq_lens_config = test_config.get('q_seq_lens', [1] * len(test_config['context_lens']))
    context_lens = test_config['context_lens']
    qk_scale = test_config.get('qk_scale', 1.0 / math.sqrt(head_size))
    quant_type = test_config.get('quant_type', QUANT_UNQUANT)
    mla_v_dim = test_config.get('mla_v_dim', 0)
    num_tokens = sum(q_seq_lens_config)
    batch_size = len(q_seq_lens_config)

    # Step 1: Generate numpy arrays for golden calculation (without NZ padding)
    # Masks are in their natural shapes for golden calculation
    # NZ padding will be applied later in _prepare_mask_for_network
    inputs_dict = generator.generate_inputs(
        num_heads, kv_heads, head_size,
        test_config['block_size'], test_config['num_blocks'],
        q_seq_lens_config, context_lens,
        test_config['q_dtype'], test_config['kv_dtype'],
        test_config['mask_type'], quant_type,
        test_config.get('has_quant_offset', False), mla_v_dim,
        test_config.get('expected_dtype')
    )

    # Step 2: Compute golden reference using numpy arrays (float32 for precision)
    # This is done before any tensor conversion or padding
    is_full_quant = quant_type in (QUANT_QKV_OFFLINE, QUANT_QKV_ONLINE)
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
        test_config['mask_type'],
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

    # Step 3: Prepare ND format tensors for network input
    # Apply NZ padding and reshape masks in _prepare_mask_for_network
    mask_nd, effective_mask_type = _prepare_mask_for_network(
        inputs_dict['mask'],
        test_config['mask_type'],
        num_tokens,
        batch_size,
        num_heads,
        context_lens,
        q_seq_lens_config,
        test_config['q_dtype']
    )

    # Step 4: Create network
    net = PagedAttentionNzNet(
        q_head_num=num_heads,
        qk_scale=qk_scale,
        kv_head_num=kv_heads,
        mask_type=effective_mask_type,  # Use effective mask type (may be switched from UNDEFINED to NORM)
        batch_run_status_enable=test_config.get('batch_run_status_enable', False),
        quant_type=quant_type,
        out_data_type=test_config.get('out_data_type', -1),
        has_quant_offset=test_config.get('has_quant_offset', False),
        compress_type=test_config.get('compress_type', 0),
        calc_type=test_config.get('calc_type', 0),
        scale_type=test_config.get('scale_type', 0),
        input_layout=test_config.get('input_layout', INPUT_LAYOUT_BSND),
        mla_v_dim=mla_v_dim,
        input_format=INPUT_FORMAT_NZ,
    )

    # Configure dynamic shapes if requested
    if dynamic:
        _configure_dynamic_shapes(net, test_config)

    # Select descale tensors based on quantization type
    k_descale, v_descale = _select_descale_tensors(
        quant_type,
        inputs_dict['k_descale'], inputs_dict['v_descale'],
        inputs_dict['k_scale_per_head'], inputs_dict['v_scale_per_head']
    )

    # Step 5: Prepare query and KV cache tensors (reshape to 2D/3D format)
    query_2d_tensor, key_cache_3d_tensor, value_cache_3d_tensor = _prepare_inputs_for_nz_network(
        inputs_dict['query'],
        inputs_dict['key_cache'],
        inputs_dict['value_cache'],
        num_heads,
        kv_heads,
        test_config['q_dtype'],
        test_config['kv_dtype']
    )

    # Convert other numpy arrays to Tensors
    block_tables_tensor = Tensor(inputs_dict['block_tables'], dtype=ms.int32)
    q_seq_lens_tensor = Tensor(inputs_dict['q_seq_lens'], dtype=ms.int32)
    kv_seq_lens_tensor = Tensor(inputs_dict['kv_seq_lens'], dtype=ms.int32)

    # Convert quantization parameters to Tensors
    k_descale_tensor = Tensor(k_descale, dtype=ms.float32) if k_descale is not None else None
    v_descale_tensor = Tensor(v_descale, dtype=ms.float32) if v_descale is not None else None
    k_offset_tensor = Tensor(inputs_dict['k_offset'], dtype=ms.int32) if inputs_dict['k_offset'] is not None else None
    v_offset_tensor = Tensor(inputs_dict['v_offset'], dtype=ms.int32) if inputs_dict['v_offset'] is not None else None
    p_scale_tensor = Tensor(inputs_dict['p_scale'], dtype=ms.float32) if inputs_dict['p_scale'] is not None else None

    # Step 6: Execute network (trans_data happens inside)
    output = net(
        query_2d_tensor,
        key_cache_3d_tensor,
        value_cache_3d_tensor,
        block_tables_tensor,
        mask_nd,  # Use prepared ND mask (already a Tensor from _prepare_mask_for_network)
        test_config.get('batch_run_status'),
        k_descale_tensor, k_offset_tensor,
        v_descale_tensor, v_offset_tensor,
        None, p_scale_tensor, None,  # razor_offset, p_scale, log_n
        q_seq_lens_tensor,
        kv_seq_lens_tensor
    )

    # Step 7: Validate output
    _validate_output(output, test_config, num_heads, head_size, mla_v_dim, num_tokens)

    # Step 8: Validate accuracy against golden reference
    if validate_accuracy:
        # Convert output to numpy for validation
        output_np = output.asnumpy()

        validate_dtype = test_config.get('expected_dtype', test_config['q_dtype'])
        is_quant_test = quant_type != QUANT_UNQUANT

        assert generator.validate_accuracy(
            output_np, golden, validate_dtype, num_heads, max(context_lens), is_quant_test
        ), "Accuracy validation failed: output does not match golden reference within tolerance"


def _configure_dynamic_shapes(net: PagedAttentionNzNet, test_config: Dict) -> None:
    """Configure dynamic input shapes for the network.

    Note: Input shapes reflect the reshaped format (2D for query, 3D for KV cache).

    Args:
        net: Network instance to configure
        test_config: Test configuration dictionary
    """
    num_heads = test_config['num_heads']
    kv_heads = test_config['kv_heads']
    head_size = test_config['head_size']
    q_dtype = test_config['q_dtype']
    kv_dtype = test_config['kv_dtype']
    mask_type = test_config['mask_type']
    batch_size = len(test_config['context_lens'])
    mla_v_dim = test_config.get('mla_v_dim', 0)

    # Dynamic tensor shapes (already reshaped to 2D/3D)
    # Query: (tokens, num_heads * head_size)
    query_dyn = Tensor(shape=[None, num_heads * head_size], dtype=q_dtype)
    # Key cache: (num_blocks, block_size, kv_heads * head_size_qk)
    key_dyn = Tensor(shape=[None, None, kv_heads * head_size], dtype=kv_dtype)
    # Value cache: (num_blocks, block_size, kv_heads * head_size_vo)
    head_size_vo = mla_v_dim if mla_v_dim > 0 else head_size
    value_dyn = Tensor(shape=[None, None, kv_heads * head_size_vo], dtype=kv_dtype)
    block_tables_dyn = Tensor(shape=[None, None], dtype=ms.int32)

    # Mask shape depends on mask type
    mask_dyn = None
    if mask_type == MASK_ALIBI:
        mask_dyn = Tensor(shape=[None, num_heads, None, None], dtype=q_dtype)
    elif mask_type == MASK_NORM:
        # NORM mask: shape varies by mode
        # Decode: (num_tokens, 1, max_context_len)
        # Prefill: (batch_size, max_q_len, max_context_len)
        mask_dyn = Tensor(shape=[None, None, None], dtype=q_dtype)
    elif mask_type == MASK_SPEC:
        # SPEC mask: (num_tokens, max_context_len)
        mask_dyn = Tensor(shape=[None, None], dtype=q_dtype)
    elif mask_type == MASK_FREE:
        # For NZ tests we provide a 3D (1, 128, 128) mask tensor before trans_data
        mask_dyn = Tensor(shape=[None, None, None], dtype=q_dtype)

    # Sequence lengths (static batch size)
    q_seq_dyn = Tensor(shape=[batch_size], dtype=ms.int32)
    kv_seq_dyn = Tensor(shape=[batch_size], dtype=ms.int32)

    net.set_inputs(
        query_dyn, key_dyn, value_dyn, block_tables_dyn,
        mask_dyn, None,  # mask, batch_run_status
        None, None, None, None,  # k_descale, k_offset, v_descale, v_offset
        None, None, None,  # razor_offset, p_scale, log_n
        q_seq_dyn, kv_seq_dyn
    )


def _validate_output(
    output: Tensor,
    test_config: Dict,
    num_heads: int,
    head_size: int,
    mla_v_dim: int,
    num_tokens: int,
) -> None:
    """Validate output tensor shape and dtype.

    Args:
        output: Output tensor to validate
        test_config: Test configuration
        num_heads: Number of query heads
        head_size: Head dimension size
        mla_v_dim: MLA V dimension (0 if not MLA)
        num_tokens: Expected number of tokens

    Raises:
        AssertionError: If shape or dtype validation fails
    """
    output_head_dim = mla_v_dim if mla_v_dim > 0 else head_size
    expected_shape = (num_tokens, num_heads, output_head_dim)

    assert tuple(output.shape) == expected_shape, \
        f"Output shape mismatch: got {output.shape}, expected {expected_shape}"

    if 'expected_dtype' in test_config:
        assert output.dtype == test_config['expected_dtype'], \
            f"Output dtype mismatch: got {output.dtype}, expected {test_config['expected_dtype']}"


# ========== Test Decorator ==========

def _paged_attention_nz_test(test_func):
    """Apply common decorators for 310P NZ tests."""
    decorators = [
        pytest.mark.level0,
        pytest.mark.platform_ascend310p,
        pytest.mark.env_onecard,
    ]
    for decorator in reversed(decorators):
        test_func = decorator(test_func)
    return test_func


# ==================== 1. Basic Functionality Tests (310P NZ) ====================

@_paged_attention_nz_test
@pytest.mark.parametrize('dynamic', [False, True])
def test_pa_nz_basic_decode_graph(dynamic):
    """
    Feature: PagedAttention (NZ 310P) - basic decode path (Graph mode)
    Description: Basic decode with fp16 inputs, ND->NZ conversion within graph, dynamic shapes
    Expectation: Operator executes successfully; output shape matches and matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(8001)
    test_config = {
        'q_seq_lens': [1, 1, 1, 1],  # Basic decode: 1 token per sequence (total=4, not 16-aligned)
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
    _run_paged_attention_nz_test(generator, test_config, context.GRAPH_MODE, validate_accuracy=True, dynamic=dynamic)


@_paged_attention_nz_test
@pytest.mark.parametrize('dynamic', [False, True])
def test_pa_nz_basic_decode_pynative(dynamic):
    """
    Feature: PagedAttention (NZ 310P) - basic decode path (PyNative mode)
    Description: Basic decode with fp16 inputs, ND->NZ conversion, PyNative mode with 16-aligned num_tokens
    Expectation: Operator executes successfully; output shape matches and matches golden within tolerance

    Note: PyNative mode requires num_tokens (H dimension) to be 16-aligned for trans_data.
    """
    generator = PagedAttentionDataGenerator(8001)
    test_config = {
        'q_seq_lens': [1] * 16,  # PyNative: num_tokens=16 (16-aligned)
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [192] * 16,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_UNDEFINED,
        'qk_scale': 1.0,
    }
    _run_paged_attention_nz_test(generator, test_config, context.PYNATIVE_MODE, validate_accuracy=True, dynamic=dynamic)


@_paged_attention_nz_test
def test_pa_nz_no_mask_graph():
    """
    Feature: PagedAttention (NZ 310P) - mask-free decode (Graph mode)
    Description: Simplest decode with no attention mask, fp16 inputs, NZ format
    Expectation: Operator executes; output shape/dtype correct and accuracy within tolerance
    """
    generator = PagedAttentionDataGenerator(8002)
    test_config = {
        'q_seq_lens': [1, 1, 1, 1],
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
    _run_paged_attention_nz_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_nz_test
def test_pa_nz_no_mask_pynative():
    """
    Feature: PagedAttention (NZ 310P) - mask-free decode (PyNative mode)
    Description: Simplest decode with no attention mask, fp16 inputs, NZ format, 16-aligned num_tokens
    Expectation: Operator executes; output shape/dtype correct and accuracy within tolerance

    Note: PyNative mode requires num_tokens (H dimension) to be 16-aligned for trans_data.
    """
    generator = PagedAttentionDataGenerator(8002)
    test_config = {
        'q_seq_lens': [1] * 16,  # PyNative: num_tokens=16 (16-aligned)
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [192] * 16,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_UNDEFINED,
        'qk_scale': 1.0,
    }
    _run_paged_attention_nz_test(generator, test_config, context.PYNATIVE_MODE)


# ==================== 2. Mask Type Tests (310P NZ) ====================

@_paged_attention_nz_test
def test_pa_nz_norm_mask_graph():
    """
    Feature: PagedAttention (NZ 310P) - normal causal mask (Graph mode)
    Description: Decode with triangular causal mask, fp16, NZ format
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(8003)
    test_config = {
        'q_seq_lens': [1, 1, 1, 1],
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [192, 193, 194, 195],
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_NORM,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_nz_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_nz_test
def test_pa_nz_norm_mask_pynative():
    """
    Feature: PagedAttention (NZ 310P) - normal causal mask (PyNative mode)
    Description: Decode with triangular causal mask, fp16, NZ format, 16-aligned num_tokens
    Expectation: Operator executes; output matches golden within tolerance

    Note: PyNative mode requires num_tokens (H dimension) to be 16-aligned for trans_data.
    """
    generator = PagedAttentionDataGenerator(8003)
    test_config = {
        'q_seq_lens': [1] * 16,  # PyNative: num_tokens=16 (16-aligned)
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [192] * 16,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_NORM,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_nz_test(generator, test_config, context.PYNATIVE_MODE)


@_paged_attention_nz_test
def test_pa_nz_alibi_mask_graph():
    """
    Feature: PagedAttention (NZ 310P) - ALIBI positional bias (Graph mode)
    Description: Decode with ALIBI bias per head using fp16, NZ format
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(8004)
    test_config = {
        'q_seq_lens': [1, 1],
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [500, 500],
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_ALIBI,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_nz_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_nz_test
def test_pa_nz_alibi_mask_pynative():
    """
    Feature: PagedAttention (NZ 310P) - ALIBI positional bias (PyNative mode)
    Description: Decode with ALIBI bias per head using fp16, NZ format, 16-aligned num_tokens
    Expectation: Operator executes; output matches golden within tolerance

    Note: PyNative mode requires num_tokens (H dimension) to be 16-aligned for trans_data.
    """
    generator = PagedAttentionDataGenerator(8004)
    test_config = {
        'q_seq_lens': [1] * 16,  # PyNative: num_tokens=16 (16-aligned)
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [500] * 16,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_ALIBI,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_nz_test(generator, test_config, context.PYNATIVE_MODE)


@_paged_attention_nz_test
def test_pa_nz_spec_mask_mtp_graph():
    """
    Feature: PagedAttention (NZ 310P) - SPEC mask for MTP (Graph mode)
    Description: Multi-token prediction (q_len=2) with SPEC mask in fp16, NZ format
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(8005)
    test_config = {
        'q_seq_lens': [2, 2, 2, 2],  # MTP: 2 tokens per sequence (total=8, not 16-aligned)
        'num_heads': 32,
        'kv_heads': 1,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [192, 193, 194, 195],
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_SPEC,
        'qk_scale': 0.01,
        'calc_type': 1,  # Enable MTP mode
    }
    _run_paged_attention_nz_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_nz_test
def test_pa_nz_spec_mask_mtp_pynative():
    """
    Feature: PagedAttention (NZ 310P) - SPEC mask for MTP (PyNative mode)
    Description: Multi-token prediction with SPEC mask in fp16, NZ format, 16-aligned num_tokens
    Expectation: Operator executes; output matches golden within tolerance

    Note: PyNative mode requires num_tokens (H dimension) to be 16-aligned for trans_data.
    """
    generator = PagedAttentionDataGenerator(8005)
    test_config = {
        'q_seq_lens': [2] * 8,  # PyNative: num_tokens=16 (16-aligned), 2 tokens per sequence
        'num_heads': 32,
        'kv_heads': 1,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 64,
        'context_lens': [192] * 8,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_SPEC,
        'qk_scale': 0.01,
        'calc_type': 1,  # Enable MTP mode
    }
    _run_paged_attention_nz_test(generator, test_config, context.PYNATIVE_MODE)


# ==================== 3. GQA Tests (310P NZ) ====================

@_paged_attention_nz_test
def test_pa_nz_gqa_8to1_graph():
    """
    Feature: PagedAttention (NZ 310P) - GQA 8:1 (Graph mode)
    Description: Grouped Query Attention with 8 query heads per 1 KV head, ALIBI mask, fp16, NZ format
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(8006)
    test_config = {
        'q_seq_lens': [1] * 13,  # total=13, not 16-aligned
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
    _run_paged_attention_nz_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_nz_test
def test_pa_nz_gqa_8to1_pynative():
    """
    Feature: PagedAttention (NZ 310P) - GQA 8:1 (PyNative mode)
    Description: Grouped Query Attention with 8 query heads per 1 KV head, ALIBI mask, fp16, NZ format, 16-aligned
    Expectation: Operator executes; output matches golden within tolerance

    Note: PyNative mode requires num_tokens (H dimension) to be 16-aligned for trans_data.
    """
    generator = PagedAttentionDataGenerator(8006)
    test_config = {
        'q_seq_lens': [1] * 16,  # PyNative: num_tokens=16 (16-aligned)
        'num_heads': 8,
        'kv_heads': 1,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 1024,
        'context_lens': [88] * 16,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_ALIBI,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_nz_test(generator, test_config, context.PYNATIVE_MODE)


@_paged_attention_nz_test
def test_pa_nz_gqa_32to8_graph():
    """
    Feature: PagedAttention (NZ 310P) - GQA 32:8 (Graph mode)
    Description: Grouped Query Attention with 32 query heads per 8 KV heads, ALIBI mask, fp16, NZ format
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(8007)
    test_config = {
        'q_seq_lens': [1] * 13,  # total=13, not 16-aligned
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
    _run_paged_attention_nz_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_nz_test
def test_pa_nz_gqa_32to8_pynative():
    """
    Feature: PagedAttention (NZ 310P) - GQA 32:8 (PyNative mode)
    Description: Grouped Query Attention with 32 query heads per 8 KV heads, ALIBI mask, fp16, NZ format, 16-aligned
    Expectation: Operator executes; output matches golden within tolerance

    Note: PyNative mode requires num_tokens (H dimension) to be 16-aligned for trans_data.
    """
    generator = PagedAttentionDataGenerator(8007)
    test_config = {
        'q_seq_lens': [1] * 16,  # PyNative: num_tokens=16 (16-aligned)
        'num_heads': 32,
        'kv_heads': 8,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 1024,
        'context_lens': [88] * 16,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_ALIBI,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_nz_test(generator, test_config, context.PYNATIVE_MODE)


# ==================== 4. Configuration Variation Tests (310P NZ) ====================

@_paged_attention_nz_test
def test_pa_nz_odd_heads_graph():
    """
    Feature: PagedAttention (NZ 310P) - Non-power-of-2 heads (Graph mode)
    Description: MHA with 7 query/KV heads (non-power-of-2), NORM mask, fp16, NZ format
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(8008)
    test_config = {
        'q_seq_lens': [1] * 13,  # total=13, not 16-aligned
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
    _run_paged_attention_nz_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_nz_test
def test_pa_nz_odd_heads_pynative():
    """
    Feature: PagedAttention (NZ 310P) - Non-power-of-2 heads (PyNative mode)
    Description: MHA with 7 query/KV heads (non-power-of-2), NORM mask, fp16, NZ format, 16-aligned
    Expectation: Operator executes; output matches golden within tolerance

    Note: PyNative mode requires num_tokens (H dimension) to be 16-aligned for trans_data.
    """
    generator = PagedAttentionDataGenerator(8008)
    test_config = {
        'q_seq_lens': [1] * 16,  # PyNative: num_tokens=16 (16-aligned)
        'num_heads': 7,
        'kv_heads': 7,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 1024,
        'context_lens': [88] * 16,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_NORM,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_nz_test(generator, test_config, context.PYNATIVE_MODE)


@_paged_attention_nz_test
def test_pa_nz_small_blocksize_graph():
    """
    Feature: PagedAttention (NZ 310P) - Small block size (Graph mode)
    Description: Decode with small block size (16), fp16, NZ format
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(8009)
    test_config = {
        'q_seq_lens': [1, 1, 1, 1],  # total=4, not 16-aligned
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 16,
        'num_blocks': 512,
        'context_lens': [192, 193, 194, 195],
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_UNDEFINED,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_nz_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_nz_test
def test_pa_nz_small_blocksize_pynative():
    """
    Feature: PagedAttention (NZ 310P) - Small block size (PyNative mode)
    Description: Decode with small block size (16), fp16, NZ format, 16-aligned num_tokens
    Expectation: Operator executes; output matches golden within tolerance

    Note: PyNative mode requires num_tokens (H dimension) to be 16-aligned for trans_data.
    """
    generator = PagedAttentionDataGenerator(8009)
    test_config = {
        'q_seq_lens': [1] * 16,  # PyNative: num_tokens=16 (16-aligned)
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 16,
        'num_blocks': 512,
        'context_lens': [192] * 16,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_UNDEFINED,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_nz_test(generator, test_config, context.PYNATIVE_MODE)


@_paged_attention_nz_test
def test_pa_nz_varied_seqlens_graph():
    """
    Feature: PagedAttention (NZ 310P) - Varied context lengths (Graph mode)
    Description: Decode with diverse context lengths [100, 500, 1000, 2000], NORM mask, fp16, NZ format
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(8010)
    test_config = {
        'q_seq_lens': [1, 1, 1, 1],  # total=4, not 16-aligned
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
    _run_paged_attention_nz_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_nz_test
def test_pa_nz_varied_seqlens_pynative():
    """
    Feature: PagedAttention (NZ 310P) - Varied context lengths (PyNative mode)
    Description: Decode with diverse context lengths, NORM mask, fp16, NZ format, 16-aligned num_tokens
    Expectation: Operator executes; output matches golden within tolerance

    Note: PyNative mode requires num_tokens (H dimension) to be 16-aligned for trans_data.
    """
    generator = PagedAttentionDataGenerator(8010)
    test_config = {
        'q_seq_lens': [1] * 16,  # PyNative: num_tokens=16 (16-aligned)
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 256,
        # PyNative: all 16-aligned
        'context_lens': [96, 496, 1008, 2000, 96, 496, 1008, 2000,
                         96, 496, 1008, 2000, 96, 496, 1008, 2000],
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_NORM,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_nz_test(generator, test_config, context.PYNATIVE_MODE)


# ==================== 5. Combined Feature Tests (310P NZ) ====================

@_paged_attention_nz_test
def test_pa_nz_alibi_with_gqa_graph():
    """
    Feature: PagedAttention (NZ 310P) - ALIBI + GQA (Graph mode)
    Description: ALIBI mask combined with GQA (8:1), fp16, NZ format
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(8014)
    test_config = {
        'q_seq_lens': [1] * 13,  # total=13, not 16-aligned
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
    _run_paged_attention_nz_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_nz_test
def test_pa_nz_alibi_with_gqa_pynative():
    """
    Feature: PagedAttention (NZ 310P) - ALIBI + GQA (PyNative mode)
    Description: ALIBI mask combined with GQA (8:1), fp16, NZ format, 16-aligned num_tokens
    Expectation: Operator executes; output matches golden within tolerance

    Note: PyNative mode requires num_tokens (H dimension) to be 16-aligned for trans_data.
    """
    generator = PagedAttentionDataGenerator(8014)
    test_config = {
        'q_seq_lens': [1] * 16,  # PyNative: num_tokens=16 (16-aligned)
        'num_heads': 8,
        'kv_heads': 1,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 1024,
        'context_lens': [88] * 16,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_ALIBI,
        'qk_scale': 1.0 / math.sqrt(128),
    }
    _run_paged_attention_nz_test(generator, test_config, context.PYNATIVE_MODE)


# ==================== 6. Lookahead/MTP Tests (310P NZ) ====================

@_paged_attention_nz_test
def test_pa_nz_lookahead_mixed_lengths_graph():
    """
    Feature: PagedAttention (NZ 310P) - Lookahead with mixed seq lengths (Graph mode)
    Description: Speculative decoding/lookahead with mixed query lengths [1, 15, 30, 6], SPEC mask, fp16, NZ format
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(8017)
    test_config = {
        'q_seq_lens': [1, 15, 30, 6],  # total=52, not 16-aligned
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 512,
        'context_lens': [10, 64, 64, 64],
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_SPEC,
        'qk_scale': 1.0 / math.sqrt(128),
        'calc_type': 1,  # Enable MTP mode
    }
    _run_paged_attention_nz_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_nz_test
def test_pa_nz_lookahead_mixed_lengths_pynative():
    """
    Feature: PagedAttention (NZ 310P) - Lookahead with mixed seq lengths (PyNative mode)
    Description: Speculative decoding/lookahead with mixed query lengths, SPEC mask, fp16, NZ format, 16-aligned
    Expectation: Operator executes; output matches golden within tolerance

    Note: PyNative mode requires num_tokens (H dimension) to be 16-aligned for trans_data.
    """
    generator = PagedAttentionDataGenerator(8017)
    test_config = {
        'q_seq_lens': [4, 4, 4, 4],  # PyNative: num_tokens=16 (16-aligned)
        'num_heads': 32,
        'kv_heads': 32,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 512,
        'context_lens': [10, 64, 64, 64],
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_SPEC,
        'qk_scale': 1.0 / math.sqrt(128),
        'calc_type': 1,  # Enable MTP mode
    }
    _run_paged_attention_nz_test(generator, test_config, context.PYNATIVE_MODE)


@_paged_attention_nz_test
def test_pa_nz_lookahead_with_gqa_graph():
    """
    Feature: PagedAttention (NZ 310P) - Lookahead + GQA (Graph mode)
    Description: Speculative decoding/lookahead with GQA (32:8), SPEC mask, fp16, NZ format
    Expectation: Operator executes; output matches golden within tolerance
    """
    generator = PagedAttentionDataGenerator(8018)
    test_config = {
        'q_seq_lens': [1, 15, 30, 6],  # total=52, not 16-aligned
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
        'calc_type': 1,  # Enable MTP mode
    }
    _run_paged_attention_nz_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_nz_test
def test_pa_nz_lookahead_with_gqa_pynative():
    """
    Feature: PagedAttention (NZ 310P) - Lookahead + GQA (PyNative mode)
    Description: Speculative decoding/lookahead with GQA (32:8), SPEC mask, fp16, NZ format, 16-aligned
    Expectation: Operator executes; output matches golden within tolerance

    Note: PyNative mode requires num_tokens (H dimension) to be 16-aligned for trans_data.
    """
    generator = PagedAttentionDataGenerator(8018)
    test_config = {
        'q_seq_lens': [4, 4, 4, 4],  # PyNative: num_tokens=16 (16-aligned)
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
        'calc_type': 1,  # Enable MTP mode
    }
    _run_paged_attention_nz_test(generator, test_config, context.PYNATIVE_MODE)


# ==================== 7. Prefill Tests (310P NZ) ====================

@_paged_attention_nz_test
def test_pa_nz_prefill_mask_free_graph():
    """
    Feature: PagedAttention (NZ 310P) - Prefill without mask (Graph mode)
    Description: Prefill scenario with long query lengths (139), mask-free, fp16, NZ format
    Expectation: Operator executes; output matches golden within tolerance
    """
    batch = 2
    generator = PagedAttentionDataGenerator(8019)
    test_config = {
        'q_seq_lens': [128 + 11] * batch,  # total=278, not 16-aligned
        'num_heads': 8,
        'kv_heads': 8,
        'head_size': 64,
        'block_size': 128,
        'num_blocks': 1024,
        'context_lens': [128 * 2 + 11] * batch,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_FREE,
        'qk_scale': 1.0 / math.sqrt(64),
        'calc_type': 1,  # Enable MTP/prefill mode
    }
    _run_paged_attention_nz_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_nz_test
def test_pa_nz_prefill_mask_free_pynative():
    """
    Feature: PagedAttention (NZ 310P) - Prefill without mask (PyNative mode)
    Description: Prefill scenario with long query lengths, mask-free, fp16, NZ format, 16-aligned
    Expectation: Operator executes; output matches golden within tolerance

    Note: PyNative mode requires num_tokens (H dimension) to be 16-aligned for trans_data.
    """
    batch = 2
    generator = PagedAttentionDataGenerator(8019)
    test_config = {
        'q_seq_lens': [128] * batch,  # PyNative: num_tokens=256 (16-aligned)
        'num_heads': 8,
        'kv_heads': 8,
        'head_size': 64,
        'block_size': 128,
        'num_blocks': 1024,
        'context_lens': [128 * 2] * batch,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_FREE,
        'qk_scale': 1.0 / math.sqrt(64),
        'calc_type': 1,  # Enable MTP/prefill mode
    }
    _run_paged_attention_nz_test(generator, test_config, context.PYNATIVE_MODE)


@_paged_attention_nz_test
def test_pa_nz_prefill_spec_mask_graph():
    """
    Feature: PagedAttention (NZ 310P) - Prefill with SPEC mask (Graph mode)
    Description: Prefill scenario with long query lengths (139), SPEC mask, GQA (8:1), fp16, NZ format
    Expectation: Operator executes; output matches golden within tolerance
    """
    batch = 1
    generator = PagedAttentionDataGenerator(8020)
    test_config = {
        'q_seq_lens': [128 * 1 + 11] * batch,  # total=139, not 16-aligned
        'num_heads': 8,
        'kv_heads': 1,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 1024,
        'context_lens': [128 * 2 + 11] * batch,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_SPEC,
        'qk_scale': 1.0 / math.sqrt(128),
        'calc_type': 1,  # Enable MTP/prefill mode
    }
    _run_paged_attention_nz_test(generator, test_config, context.GRAPH_MODE)


@_paged_attention_nz_test
def test_pa_nz_prefill_spec_mask_pynative():
    """
    Feature: PagedAttention (NZ 310P) - Prefill with SPEC mask (PyNative mode)
    Description: Prefill scenario with long query lengths, SPEC mask, GQA (8:1), fp16, NZ format, 16-aligned
    Expectation: Operator executes; output matches golden within tolerance

    Note: PyNative mode requires num_tokens (H dimension) to be 16-aligned for trans_data.
    """
    batch = 1
    generator = PagedAttentionDataGenerator(8020)
    test_config = {
        'q_seq_lens': [128] * batch,  # PyNative: num_tokens=128 (16-aligned)
        'num_heads': 8,
        'kv_heads': 1,
        'head_size': 128,
        'block_size': 128,
        'num_blocks': 1024,
        'context_lens': [128 * 2] * batch,
        'q_dtype': ms.float16,
        'kv_dtype': ms.float16,
        'mask_type': MASK_SPEC,
        'qk_scale': 1.0 / math.sqrt(128),
        'calc_type': 1,  # Enable MTP/prefill mode
    }
    _run_paged_attention_nz_test(generator, test_config, context.PYNATIVE_MODE)
