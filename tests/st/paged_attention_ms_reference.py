"""MindSpore reference implementation for paged attention tests.

This module mirrors the numpy-based logic in ``test_custom_paged_attention.py``
using MindSpore Tensor operators so that test data generation, mask construction,
and golden reference computation can be performed fully inside MindSpore.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import mindspore as ms
from mindspore import Tensor, ops

MASK_UNDEFINED = 0
MASK_NORM = 1
MASK_ALIBI = 2
MASK_SPEC = 3
MASK_FREE = 4

QUANT_UNQUANT = 0
DEQUANT_FUSION = 1
QUANT_QKV_OFFLINE = 2
QUANT_QKV_ONLINE = 3


class PagedAttentionMsReference:
    """MindSpore version of paged attention data generator and golden calc."""

    def __init__(self, seed: int = 2025):
        """Initialize the MindSpore reference implementation.

        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = int(seed)
        ms.set_seed(self.seed)
        self.np_rng = np.random.default_rng(self.seed)
        self.round = ops.round
        self.clip = ops.clip_by_value
        self.concat = ops.concat
        self.stack = ops.stack
        # Use explicit max / exp / sum for softmax to mirror numpy reference.
        self.exp = ops.exp
        self.transpose = ops.transpose
        self.matmul = ops.matmul
        self.maximum = ops.maximum
        self.abs = ops.abs

    # ----------------------------------------------------------------------
    # Random helpers

    def _uniform_np(self, shape, low, high):
        """Generate numpy array from uniform distribution."""
        return self.np_rng.uniform(low, high, size=shape)

    def _randint_np(self, low, high, shape):
        """Generate numpy array of random integers."""
        return self.np_rng.integers(low, high, size=shape, endpoint=False, dtype=np.int32)

    # ----------------------------------------------------------------------
    # Public APIs

    def generate_inputs_ms(
        self,
        num_heads: int,
        kv_heads: int,
        head_size: int,
        block_size: int,
        num_blocks: int,
        q_seq_lens: List[int],
        context_lens: List[int],
        q_dtype: ms.dtype,
        kv_dtype: ms.dtype,
        mask_type: int,
        quant_type: int = QUANT_UNQUANT,
        has_quant_offset: bool = False,
        mla_v_dim: int = 0,
        mask_out_dtype: Optional[ms.dtype] = None,
    ) -> Dict[str, Tensor]:
        """MindSpore variant of generate_inputs (returns Tensors)."""
        is_full_quant = quant_type in (QUANT_QKV_OFFLINE, QUANT_QKV_ONLINE)
        is_dequant_fusion = quant_type == DEQUANT_FUSION
        is_mla = mla_v_dim > 0

        batch_size = len(q_seq_lens)
        num_tokens = sum(q_seq_lens)
        head_size_qk = head_size
        head_size_vo = mla_v_dim if mla_v_dim > 0 else head_size

        # Query
        if q_dtype == ms.int8:
            q_range = 2.0 if is_full_quant else 1.0
            query_np = self._uniform_np((num_tokens, num_heads, head_size_qk), -q_range, q_range).astype(np.float32)
            query_np = np.clip(np.rint(query_np * (127.0 / q_range)), -127, 127).astype(np.int8)
            query = Tensor(query_np, dtype=ms.int8)
        else:
            query_np = self._uniform_np((num_tokens, num_heads, head_size_qk), -1.0, 1.0).astype(np.float32)
            query = Tensor(query_np, dtype=q_dtype)

        # Key/Value cache
        if kv_dtype == ms.int8:
            kv_range = 2.0 if is_full_quant else 4.0
            key_np = self._uniform_np(
                (num_blocks, block_size, kv_heads, head_size_qk),
                -kv_range,
                kv_range,
            ).astype(np.float32)
            key_np = np.clip(
                np.rint(key_np * (127.0 / kv_range)),
                -127,
                127,
            ).astype(np.int8)
            key_cache = Tensor(key_np, dtype=ms.int8)
            if is_mla:
                value_cache = Tensor(key_np[:, :, :, :head_size_vo], dtype=ms.int8)
            else:
                value_np = self._uniform_np(
                    (num_blocks, block_size, kv_heads, head_size_vo),
                    -kv_range,
                    kv_range,
                ).astype(np.float32)
                value_np = np.clip(
                    np.rint(value_np * (127.0 / kv_range)),
                    -127,
                    127,
                ).astype(np.int8)
                value_cache = Tensor(value_np, dtype=ms.int8)
        else:
            key_np = self._uniform_np((num_blocks, block_size, kv_heads, head_size_qk), -1.0, 1.0).astype(np.float32)
            key_cache = Tensor(key_np, dtype=kv_dtype)
            if is_mla:
                value_cache = Tensor(key_np[:, :, :, :head_size_vo], dtype=kv_dtype)
            else:
                value_np = self._uniform_np(
                    (num_blocks, block_size, kv_heads, head_size_vo),
                    -1.0,
                    1.0,
                ).astype(np.float32)
                value_cache = Tensor(value_np, dtype=kv_dtype)

        # Block tables
        max_context_len = max(context_lens)
        max_blocks_per_seq = (max_context_len + block_size - 1) // block_size
        block_tables_np = self._randint_np(0, num_blocks, (batch_size, max_blocks_per_seq))
        block_tables = Tensor(block_tables_np, dtype=ms.int32)

        q_seq_tensor = Tensor(q_seq_lens, ms.int32)
        kv_seq_tensor = Tensor(context_lens, ms.int32)

        mask_dtype = mask_out_dtype if mask_out_dtype is not None else q_dtype
        mask = self._generate_mask_tensor(
            mask_type,
            num_tokens,
            max_context_len,
            q_seq_lens,
            context_lens,
            mask_dtype,
            num_heads,
            is_mla,
        )

        k_descale = v_descale = k_offset = v_offset = None
        k_scale_per_head = v_scale_per_head = p_scale = None

        if is_dequant_fusion:
            k_descale_np = self._uniform_np((kv_heads * head_size_qk,), -1.0, 1.0).astype(np.float32)
            v_descale_np = self._uniform_np((kv_heads * head_size_vo,), -1.0, 1.0).astype(np.float32)
            k_descale = Tensor(k_descale_np, ms.float32)
            v_descale = Tensor(v_descale_np, ms.float32)
            if has_quant_offset:
                k_offset_np = self._randint_np(-20, 20, (kv_heads * head_size_qk,))
                v_offset_np = self._randint_np(-20, 20, (kv_heads * head_size_vo,))
                k_offset = Tensor(k_offset_np, ms.int32)
                v_offset = Tensor(v_offset_np, ms.int32)
        elif is_full_quant:
            k_scale_np = self._uniform_np((num_heads,), -1.0, 2.0).astype(np.float32)
            v_scale_np = self._uniform_np((num_heads,), -1.0, 2.0).astype(np.float32)
            k_scale_per_head = Tensor(k_scale_np, ms.float32)
            v_scale_per_head = Tensor(v_scale_np, ms.float32)
            if quant_type == QUANT_QKV_OFFLINE:
                p_scale_np = self._uniform_np((num_heads,), -1.0, 2.0).astype(np.float32)
                p_scale = Tensor(p_scale_np, ms.float32)

        return {
            "query": query,
            "key_cache": key_cache,
            "value_cache": value_cache,
            "block_tables": block_tables,
            "q_seq_lens": q_seq_tensor,
            "kv_seq_lens": kv_seq_tensor,
            "mask": mask,
            "k_descale": k_descale,
            "v_descale": v_descale,
            "k_offset": k_offset,
            "v_offset": v_offset,
            "k_scale_per_head": k_scale_per_head,
            "v_scale_per_head": v_scale_per_head,
            "p_scale": p_scale,
            "q_dtype": q_dtype,
            "kv_dtype": kv_dtype,
            "mask_dtype": mask_dtype,
        }

    # ------------------------------------------------------------------
    # Mask construction

    def _generate_mask_tensor(
        self,
        mask_type: int,
        num_tokens: int,
        max_context_len: int,
        q_seq_lens: List[int],
        kv_seq_lens: List[int],
        dtype: ms.dtype,
        num_heads: int,
        is_mla: bool,
    ) -> Optional[Tensor]:
        """Generate attention mask as MindSpore Tensor.

        Args:
            mask_type: Type of mask (MASK_UNDEFINED/NORM/ALIBI/SPEC/FREE).
            num_tokens: Total number of query tokens.
            max_context_len: Maximum context length.
            q_seq_lens: Query sequence lengths per batch.
            kv_seq_lens: Key/value sequence lengths per batch.
            dtype: Data type for the mask.
            num_heads: Number of attention heads.
            is_mla: Whether using MLA architecture.

        Returns:
            Attention mask tensor or None if mask_type is MASK_UNDEFINED.
        """
        mask_np = self._generate_mask_np(
            mask_type,
            num_tokens,
            max_context_len,
            q_seq_lens,
            kv_seq_lens,
            dtype,
            num_heads,
            is_mla,
        )
        if mask_np is None:
            return None
        return Tensor(mask_np, dtype=dtype)

    def _generate_mask_np(
        self,
        mask_type: int,
        num_tokens: int,
        max_context_len: int,
        q_seq_lens: List[int],
        kv_seq_lens: List[int],
        dtype: ms.dtype,
        num_heads: int,
        is_mla: bool,
    ) -> Optional[np.ndarray]:
        """Generate attention mask as numpy array.

        Args:
            mask_type: Type of mask (MASK_UNDEFINED/NORM/ALIBI/SPEC/FREE).
            num_tokens: Total number of query tokens.
            max_context_len: Maximum context length.
            q_seq_lens: Query sequence lengths per batch.
            kv_seq_lens: Key/value sequence lengths per batch.
            dtype: Data type for the mask.
            num_heads: Number of attention heads.
            is_mla: Whether using MLA architecture.

        Returns:
            Attention mask array or None if mask_type is MASK_UNDEFINED.
        """
        if mask_type == MASK_UNDEFINED:
            return None

        if mask_type == MASK_FREE:
            mask = np.zeros((num_tokens, max_context_len), dtype=np.float32)
            row = 0
            for q_len, k_len in zip(q_seq_lens, kv_seq_lens):
                tri = np.triu(np.ones((q_len, q_len), dtype=np.float32), k=1) * -60000.0
                start = k_len - q_len
                mask[row : row + q_len, start:k_len] = tri
                row += q_len
            return mask

        if mask_type == MASK_NORM:
            max_q_len = max(q_seq_lens)
            if max_q_len == 1:
                mask = np.zeros((num_tokens, 1, max_context_len), dtype=np.float32)
                for token_idx in range(num_tokens):
                    if token_idx > 0:
                        mask[token_idx, :, :token_idx] = -10000.0
                return mask

            mask = np.zeros((len(q_seq_lens), max_q_len, max_context_len), dtype=np.float32)
            for batch_idx, q_len in enumerate(q_seq_lens):
                tri = np.triu(np.ones((q_len, q_len), dtype=np.float32), k=1) * -10000.0
                mask[batch_idx, -q_len:, -q_len:] = tri
            return mask

        if mask_type == MASK_ALIBI:
            max_q_len = max(q_seq_lens)
            base_shape = (
                len(q_seq_lens),
                num_heads,
                1 if max_q_len == 1 else max_q_len,
                max_context_len,
            )
            mask = np.zeros(base_shape, dtype=np.float32)
            slopes = self._get_alibi_slopes_numpy(num_heads)
            for i, (q_len, k_len) in enumerate(zip(q_seq_lens, kv_seq_lens)):
                if k_len == 0:
                    continue
                positions = np.arange(k_len, dtype=np.float32)
                alibi_bias = (positions - (k_len - 1)).astype(np.float32)
                alibi_bias = alibi_bias.reshape(1, 1, k_len)
                bias = slopes[:, :, None] * alibi_bias
                if max_q_len == 1:
                    mask[i, :, :, :k_len] = bias
                else:
                    mask[i, :, :q_len, :k_len] = bias
            return mask

        if mask_type == MASK_SPEC:
            mask = np.zeros((num_tokens, max_context_len), dtype=np.float32)
            row = 0
            pre_mask_factor = 1.0 if (is_mla and dtype == ms.bfloat16) else -10000.0
            for q_len, k_len in zip(q_seq_lens, kv_seq_lens):
                # Keep row (global token index) semantics aligned with numpy reference:
                # - Always advance row by q_len whenever there are query tokens,
                #   even if k_len == 0 (no valid context); this matches the
                #   behaviour of pre_q += qseq in the numpy implementation.
                if q_len == 0:
                    continue
                if k_len == 0:
                    row += q_len
                    continue
                start = max(0, k_len - q_len)
                tri = np.triu(np.ones((q_len, q_len), dtype=np.float32), k=1) * pre_mask_factor
                mask[row : row + q_len, start:k_len] = tri
                row += q_len
            return mask

        return None

    def _get_alibi_slopes(self, num_heads: int) -> Tensor:
        """Compute ALiBi position bias slopes as MindSpore Tensor.

        Args:
            num_heads: Number of attention heads.

        Returns:
            Tensor of shape (num_heads, 1, 1) containing slope values.
        """
        nearest_pow = 2 ** int(math.floor(math.log2(num_heads)))
        base = 2.0 ** (-8.0 / nearest_pow)
        slopes = [base ** i for i in range(1, nearest_pow + 1)]
        if nearest_pow < num_heads:
            extra_base = 2.0 ** (-4.0 / nearest_pow)
            extra = [
                extra_base ** i for i in range(1, 1 + 2 * (num_heads - nearest_pow), 2)
            ]
            slopes.extend(extra)
        return Tensor(slopes[:num_heads], ms.float32).reshape((num_heads, 1, 1))

    def _get_alibi_slopes_numpy(self, num_heads: int) -> np.ndarray:
        """Compute ALiBi position bias slopes as numpy array.

        Args:
            num_heads: Number of attention heads.

        Returns:
            Array of shape (num_heads, 1, 1) containing slope values.
        """
        nearest_pow = 2 ** int(math.floor(math.log2(num_heads)))
        base = 2.0 ** (-8.0 / nearest_pow)
        slopes = [base ** i for i in range(1, nearest_pow + 1)]
        if nearest_pow < num_heads:
            extra_base = 2.0 ** (-4.0 / nearest_pow)
            extra = [
                extra_base ** i for i in range(1, 1 + 2 * (num_heads - nearest_pow), 2)
            ]
            slopes.extend(extra)
        slopes = np.array(slopes[:num_heads], dtype=np.float32)
        return slopes.reshape((num_heads, 1, 1))

    # ------------------------------------------------------------------
    # Golden reference computation

    def compute_golden_reference_ms(
        self,
        query: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        block_tables: Tensor,
        q_seq_lens: Tensor,
        kv_seq_lens: Tensor,
        scale: float,
        mask: Optional[Tensor],
        mask_type: int = MASK_UNDEFINED,
        k_descale: Optional[Tensor] = None,
        v_descale: Optional[Tensor] = None,
        k_offset: Optional[Tensor] = None,
        v_offset: Optional[Tensor] = None,
        k_scale_per_head: Optional[Tensor] = None,
        v_scale_per_head: Optional[Tensor] = None,
        p_scale: Optional[Tensor] = None,
        output_dtype: Optional[ms.dtype] = None,
        mla_v_dim: int = 0,
        mask_dtype: Optional[ms.dtype] = None,
    ) -> Tensor:
        """Compute golden reference using MindSpore operators, mirroring numpy reference.

        To align numerically with the numpy implementation used in
        ``test_custom_paged_attention.py``, all heavy computations are forced
        onto CPU in float32/int32, and softmax is implemented via explicit
        max/exp/sum rather than a fused op.
        """
        original_q_dtype = query.dtype
        q = ops.cast(query, ms.float32)
        kc = ops.cast(key_cache, ms.float32)
        vc = ops.cast(value_cache, ms.float32)
        num_tokens, num_heads, head_size_qk = q.shape
        kv_heads = kc.shape[2]
        head_size_v = vc.shape[3]
        output_head_dim = head_size_v if mla_v_dim > 0 else head_size_qk

        # Dequant fusion (int8 KV → float domain)
        if k_descale is not None and v_descale is not None:
            k_scale = ops.reshape(ops.cast(k_descale, ms.float32), (kv_heads, -1))
            v_scale = ops.reshape(ops.cast(v_descale, ms.float32), (kv_heads, -1))
            if k_offset is not None:
                k_off = ops.reshape(ops.cast(k_offset, ms.float32), (kv_heads, -1))
                kc = kc + k_off[None, None, :, :]
            if v_offset is not None:
                v_off = ops.reshape(ops.cast(v_offset, ms.float32), (kv_heads, -1))
                vc = vc + v_off[None, None, :, :]
            kc = kc * k_scale[None, None, :, :]
            vc = vc * v_scale[None, None, :, :]

            # --------------------------------------------------------------
            # Simulate low-precision matmul behaviour for quantized KV:
            # - For bf16/fp16 Q inputs, many hardware kernels effectively
            #   run Q/K/V matmuls in that low precision and accumulate to
            #   fp32. To make golden closer to the real kernel (and more
            #   consistent with the numpy reference thresholds), we round
            #   Q/K/V onto the corresponding grid and then lift back to
            #   float32 for the rest of the computation.
            # --------------------------------------------------------------
            if original_q_dtype in (ms.bfloat16, ms.float16):
                lowp_dtype = ms.bfloat16 if original_q_dtype == ms.bfloat16 else ms.float16
                q = ops.cast(ops.cast(q, lowp_dtype), ms.float32)
                kc = ops.cast(ops.cast(kc, lowp_dtype), ms.float32)
                vc = ops.cast(ops.cast(vc, lowp_dtype), ms.float32)

        is_full_quant = k_scale_per_head is not None and v_scale_per_head is not None
        if is_full_quant:
            q = ops.cast(q, ms.int32)
            kc = ops.cast(kc, ms.int32)
            vc = ops.cast(vc, ms.int32)

        out = ops.zeros(
            (num_tokens, num_heads, output_head_dim), ms.float32
        )
        q_seq_list = q_seq_lens.asnumpy().tolist()
        kv_seq_list = kv_seq_lens.asnumpy().tolist()
        block_tables_np = block_tables.asnumpy().tolist()
        block_size = kc.shape[1]
        q_ptr = 0

        for batch_idx, (ql, kl) in enumerate(zip(q_seq_list, kv_seq_list)):
            if ql == 0 or kl == 0:
                q_ptr += ql
                continue

            keys = []
            values = []
            block_table = block_tables_np[batch_idx]
            for j in range(kl):
                block_id = int(block_table[j // block_size])
                offset = j % block_size
                keys.append(kc[block_id, offset])
                values.append(vc[block_id, offset])
            key_seq = self.stack(keys, axis=0)
            value_seq = self.stack(values, axis=0)

            q_slice = q[q_ptr : q_ptr + ql]
            q_t = self.transpose(q_slice, (1, 0, 2))
            k_t = self.transpose(key_seq, (1, 2, 0))
            v_t = self.transpose(value_seq, (1, 0, 2))

            if is_full_quant:
                out_slice = self._compute_full_quant_attention_ms(
                    q_t,
                    k_t,
                    v_t,
                    num_heads,
                    kv_heads,
                    scale,
                    k_scale_per_head,
                    v_scale_per_head,
                    p_scale,
                    mask,
                    mask_type,
                    batch_idx,
                    q_ptr,
                    ql,
                    kl,
                    mla_v_dim,
                    mask_dtype,
                )
            else:
                out_slice = self._compute_float_attention_ms(
                    q_t,
                    k_t,
                    v_t,
                    num_heads,
                    kv_heads,
                    scale,
                    mask,
                    mask_type,
                    batch_idx,
                    q_ptr,
                    ql,
                    kl,
                    mla_v_dim,
                    mask_dtype,
                )

            out_slice = ops.reshape(out_slice, (num_heads, ql, output_head_dim))
            out[q_ptr : q_ptr + ql] = self.transpose(out_slice, (1, 0, 2))
            q_ptr += ql

        if output_dtype is not None:
            return ops.cast(out, output_dtype)
        return ops.cast(out, query.dtype)

    def _compute_full_quant_attention_ms(
        self,
        query_t: Tensor,
        key_t: Tensor,
        value_t: Tensor,
        num_heads: int,
        kv_heads: int,
        scale: float,
        k_scale: Tensor,
        v_scale: Tensor,
        p_scale: Optional[Tensor],
        mask: Optional[Tensor],
        mask_type: int,
        batch_idx: int,
        q_ptr: int,
        ql: int,
        kl: int,
        mla_v_dim: int,
        mask_dtype: Optional[ms.dtype],
    ) -> Tensor:
        """Compute attention with full quantization (int8 Q/K/V).

        Args:
            query_t: Query tensor (num_heads, ql, head_dim).
            key_t: Key tensor (kv_heads, head_dim, kl).
            value_t: Value tensor (kv_heads, kl, head_dim).
            num_heads: Number of query heads.
            kv_heads: Number of key/value heads.
            scale: Attention scale factor.
            k_scale: Per-head key dequantization scale.
            v_scale: Per-head value dequantization scale.
            p_scale: Optional attention probability quantization scale.
            mask: Optional attention mask.
            mask_type: Type of mask.
            batch_idx: Current batch index.
            q_ptr: Current query token pointer.
            ql: Query sequence length.
            kl: Key/value sequence length.
            mla_v_dim: MLA value dimension (0 if not MLA).
            mask_dtype: Mask data type.

        Returns:
            Attention output tensor.
        """
        scores_int32 = self._group_matmul_ms(query_t, key_t, num_heads, kv_heads)
        scores = ops.zeros_like(scores_int32, ms.float32)
        for h in range(num_heads):
            scores[h] = ops.cast(scores_int32[h], ms.float32) * k_scale[h]
        scores = scores * Tensor(scale, ms.float32)

        if mask is not None:
            scores = self._apply_mask_ms(
                scores, mask, mask_type, batch_idx, q_ptr, ql, kl, mla_v_dim, mask_dtype
            )

        scores_max = ops.amax(scores, axis=-1, keepdims=True)
        exp_scores = self.exp(scores - scores_max)
        row_sum = ops.sum(exp_scores, dim=-1, keepdim=True)

        if p_scale is not None:
            probs = exp_scores * ops.reshape(p_scale, (num_heads, 1, 1))
            probs = self.round(probs.astype(ms.float16)).astype(ms.int32)
            out_int32 = self._group_matmul_pv_ms(probs, value_t, num_heads, kv_heads)
            out = ops.zeros_like(out_int32, ms.float32)
            for h in range(num_heads):
                out[h] = ops.cast(out_int32[h], ms.float32) * v_scale[h]
        else:
            row_max = ops.amax(exp_scores, axis=-1, keepdims=True)
            p_scale_dyn = row_max / 127.0
            probs = exp_scores / p_scale_dyn
            probs = self.round(probs.astype(ms.float16)).astype(ms.int32)
            out_int32 = self._group_matmul_pv_ms(probs, value_t, num_heads, kv_heads)
            out = ops.zeros_like(out_int32, ms.float32)
            for h in range(num_heads):
                de_scale = v_scale[h] * row_max[h, 0, 0] / 127.0
                out[h] = ops.cast(out_int32[h], ms.float32) * de_scale

        return out / row_sum

    def _compute_float_attention_ms(
        self,
        query_t: Tensor,
        key_t: Tensor,
        value_t: Tensor,
        num_heads: int,
        kv_heads: int,
        scale: float,
        mask: Optional[Tensor],
        mask_type: int,
        batch_idx: int,
        q_ptr: int,
        ql: int,
        kl: int,
        mla_v_dim: int,
        mask_dtype: Optional[ms.dtype],
    ) -> Tensor:
        """Compute attention in floating-point precision.

        Args:
            query_t: Query tensor (num_heads, ql, head_dim).
            key_t: Key tensor (kv_heads, head_dim, kl).
            value_t: Value tensor (kv_heads, kl, head_dim).
            num_heads: Number of query heads.
            kv_heads: Number of key/value heads.
            scale: Attention scale factor.
            mask: Optional attention mask.
            mask_type: Type of mask.
            batch_idx: Current batch index.
            q_ptr: Current query token pointer.
            ql: Query sequence length.
            kl: Key/value sequence length.
            mla_v_dim: MLA value dimension (0 if not MLA).
            mask_dtype: Mask data type.

        Returns:
            Attention output tensor.
        """
        # Mirror numpy reference:
        #   scores = group_matmul(q, k) * scale
        #   scores = apply_mask(scores)
        #   scores_max = max(scores, axis=-1, keepdims=True)
        #   exp_scores = exp(scores - scores_max)
        #   probs = exp_scores / sum(exp_scores, axis=-1, keepdims=True)
        #   out = group_matmul_pv(probs, v)
        scores = self._group_matmul_ms(query_t, key_t, num_heads, kv_heads)
        scores = scores * Tensor(scale, ms.float32)
        if mask is not None:
            scores = self._apply_mask_ms(
                scores, mask, mask_type, batch_idx, q_ptr, ql, kl, mla_v_dim, mask_dtype
            )

        scores_max = ops.amax(scores, axis=-1, keepdims=True)
        exp_scores = self.exp(scores - scores_max)
        denom = ops.sum(exp_scores, dim=-1, keepdim=True)
        probs = exp_scores / denom

        return self._group_matmul_pv_ms(probs, value_t, num_heads, kv_heads)

    def _apply_mask_ms(
        self,
        scores: Tensor,
        mask: Tensor,
        mask_type: int,
        batch_idx: int,
        q_ptr: int,
        ql: int,
        kl: int,
        mla_v_dim: int,
        mask_dtype: Optional[ms.dtype],
    ) -> Tensor:
        """Apply attention mask to scores.

        Args:
            scores: Attention scores tensor.
            mask: Mask tensor.
            mask_type: Type of mask.
            batch_idx: Current batch index.
            q_ptr: Current query token pointer.
            ql: Query sequence length.
            kl: Key/value sequence length.
            mla_v_dim: MLA value dimension.
            mask_dtype: Mask data type.

        Returns:
            Masked attention scores.
        """
        post_factor = 1.0
        if mla_v_dim > 0 and mask_type == MASK_SPEC and mask_dtype == ms.bfloat16:
            post_factor = -10000.0
        mask_float = ops.cast(mask, ms.float32)
        if mask_type == MASK_FREE:
            mask_slice = mask_float[q_ptr : q_ptr + ql, :kl]
            scores = scores + mask_slice[None, :, :]
        elif mask_type == MASK_ALIBI:
            mask_slice = mask_float[batch_idx, :, :ql, :kl]
            scores = scores + mask_slice
        elif mask_type == MASK_SPEC:
            mask_slice = mask_float[q_ptr : q_ptr + ql, :kl]
            scores = scores + mask_slice[None, :, :] * post_factor
        else:  # MASK_NORM
            # Align MASK_NORM behaviour with numpy reference implementation:
            # - Decode mode (max_q_len == 1): mask shape is (num_tokens, 1, max_context_len)
            #   Use batch_idx to index the current sequence and add a 1D mask
            #   broadcasted over (num_heads, ql, kl).
            # - Prefill mode: mask shape is (batch_size, max_q_len, max_context_len)
            #   Use last ql rows for current batch, same as numpy path.
            if mask_float.ndim == 3 and mask_float.shape[1] == 1:
                # Decode mode: take 1D mask [kl] for current sequence.
                mask_slice = mask_float[batch_idx, 0, :kl]  # [kl]
                scores = scores + mask_slice[None, None, :]  # -> broadcast to [num_heads, ql, kl]
            else:
                # Prefill mode: causal upper triangular on the last ql positions.
                mask_slice = mask_float[batch_idx, -ql:, :kl]  # [ql, kl]
                scores = scores + mask_slice[None, :, :]       # -> [1, ql, kl] broadcast over heads
        return scores

    def _group_matmul_ms(
        self, query_block: Tensor, key_block: Tensor, num_heads: int, kv_heads: int
    ) -> Tensor:
        """Perform grouped query-key matmul for multi-query/grouped-query attention.

        Args:
            query_block: Query tensor (num_heads, seq_len, head_dim).
            key_block: Key tensor (kv_heads, head_dim, kv_len).
            num_heads: Number of query heads.
            kv_heads: Number of key/value heads.

        Returns:
            QK^T scores of shape (num_heads, seq_len, kv_len).
        """
        group_size = num_heads // kv_heads
        outputs = []
        for kv_h in range(kv_heads):
            q_group = query_block[kv_h * group_size : (kv_h + 1) * group_size]
            k_head = key_block[kv_h : kv_h + 1]
            outputs.append(self.matmul(q_group, k_head))
        return self.concat(outputs, axis=0)

    def _group_matmul_pv_ms(
        self, prob_block: Tensor, value_block: Tensor, num_heads: int, kv_heads: int
    ) -> Tensor:
        """Perform grouped probability-value matmul for multi-query/grouped-query attention.

        Args:
            prob_block: Attention probabilities (num_heads, seq_len, kv_len).
            value_block: Value tensor (kv_heads, kv_len, head_dim).
            num_heads: Number of query heads.
            kv_heads: Number of key/value heads.

        Returns:
            Attention output of shape (num_heads, seq_len, head_dim).
        """
        group_size = num_heads // kv_heads
        outputs = []
        for kv_h in range(kv_heads):
            p_group = prob_block[kv_h * group_size : (kv_h + 1) * group_size]
            v_head = value_block[kv_h : kv_h + 1]
            outputs.append(self.matmul(p_group, v_head))
        return self.concat(outputs, axis=0)

    # ------------------------------------------------------------------
    # Accuracy check

    def validate_accuracy_ms(
        self,
        output,
        golden,
        dtype: ms.dtype,
        num_heads: int,
        max_context_len: int,
        is_quant: bool = False,
    ) -> bool:
        """Validate kernel output against the reference with multiple thresholds."""
        # Convert to numpy float32 for comparison
        out_np = np.array(output, dtype=np.float32)
        golden_np = np.array(golden, dtype=np.float32)

        out_flat = out_np.reshape(-1)
        golden_flat = golden_np.reshape(-1)
        diff = np.abs(out_flat - golden_flat)
        max_diff = np.max(diff)

        ratios = [0.001, 0.001, 0.005, 0.005]
        rel_loose, abs_loose, rel_strict, abs_strict = ratios
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

        print(f"[MS-REF] Max difference: {max_diff:.6f}")
        print(f"[MS-REF] Loose accuracy (1/1000): {accuracy_loose:.6f}")
        print(f"[MS-REF] Strict accuracy (5/1000): {accuracy_strict:.6f}")

        error_ratio = float(strict_error_count) / out_len
        if dtype == ms.bfloat16 or is_quant:
            legacy_pass = error_ratio <= rel_strict_eff
        else:
            legacy_pass = error_ratio <= ratios[0]

        calc_times = num_heads * max_context_len + 4
        if dtype == ms.bfloat16:
            base = 2 ** (-7) if calc_times < 2048 else 2 ** (-6)
            error_factor = base * (2.0 if is_quant else 1.0)
        elif dtype == ms.float16:
            error_factor = 2 ** (-8) if calc_times < 2048 else 2 ** (-7)
        else:
            if calc_times < 2048:
                error_factor = 2 ** (-11)
            elif calc_times < 16384:
                error_factor = 2 ** (-10)
            else:
                error_factor = 2 ** (-9)

        error_threshold = np.maximum(np.abs(golden_flat), 1.0) * error_factor
        adaptive_pass = np.all(diff <= error_threshold)

        print(f"[MS-REF] Calculation complexity: {calc_times}")
        print(f"[MS-REF] Error factor: {error_factor:.6e}")
        print(f"[MS-REF] Adaptive test: {'PASS' if adaptive_pass else 'FAIL'}")
        print(f"[MS-REF] Legacy test: {'PASS' if legacy_pass else 'FAIL'}")

        return bool(adaptive_pass or legacy_pass)

__all__ = ["PagedAttentionMsReference"]
