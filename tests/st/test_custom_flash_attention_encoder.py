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

import math
import numpy as np
import pytest
import mindspore as ms
import ms_custom_ops
from mindspore import Tensor, context, ops, nn
from mindspore.common.np_dtype import bfloat16 as np_bfloat16

MASK_NONE = 0
MASK_NORM = 1
MASK_ALIBI = 2
MASK_NORM_COMPRESS = 3
MASK_SWA_NORM = 7
MASK_SWA_COMPRESS = 8

KERNEL_DEFAULT = 0
KERNEL_HIGH_PRECISION = 1

CACHE_NORM = 0
CACHE_SWA = 1

class FlashAttentionTestSuite:
    """Unified test suite for Flash Attention encoder operations.
    
    Provides comprehensive testing with support for different layouts (TH/TND),
    data types (fp16/bf16), mask types, and GQA configurations.
    """
    
    def __init__(self, rng_seed: int = None):
        """Initialize test suite with random generator.

        Args:
            rng_seed: Random seed for reproducibility. If None, uses non-deterministic seed.
        """
        self.rng = np.random.default_rng(rng_seed)
    
    # ========== Utility Methods ==========
    
    @staticmethod
    def _ms_tensor(x: np.ndarray) -> Tensor:
        """Convert NumPy array to MindSpore Tensor, handling bfloat16."""
        if x is None:
            return None
        return Tensor(x.astype(np.float32)).astype(ms.bfloat16) if x.dtype == np_bfloat16 else Tensor(x)
    
    @staticmethod
    def _get_alibi_slopes(n_heads: int) -> np.ndarray:
        """Generate ALIBI slopes for positional bias"""
        n = 2 ** int(np.floor(np.log2(n_heads)))
        m0 = 2.0 ** (-8.0 / n)
        slopes = np.array([m0 ** i for i in range(1, n + 1)], dtype=np.float32)
        
        if n < n_heads:
            m1 = 2.0 ** (-4.0 / n)
            # Generate additional slopes with step size 2
            additional_count = n_heads - n
            mm = np.array([m1 ** i for i in range(1, 1 + 2 * additional_count, 2)], dtype=np.float32)
            slopes = np.concatenate([slopes, mm], axis=0)
        
        return slopes
    
    # ========== Input Generation ==========
    
    def build_inputs(self, q_seq: np.ndarray, kv_seq: np.ndarray, heads: int, head_dim: int, 
                    np_dtype: np.dtype, kv_heads: int = None, layout: str = 'TH') -> tuple:
        """Build test inputs for either TH or TND layout."""
        q_ntok = int(q_seq.sum())
        kv_ntok = int(kv_seq.sum())
        kvh = heads if kv_heads is None else kv_heads
        
        if layout == 'TH':
            # TH layout: [tokens, heads*head_dim]
            q_np = self.rng.standard_normal((q_ntok, heads * head_dim)).astype(np_dtype)
            k_np = self.rng.standard_normal((kv_ntok, kvh * head_dim)).astype(np_dtype)
            v_np = self.rng.standard_normal((kv_ntok, kvh * head_dim)).astype(np_dtype)
        else:  # TND layout
            # TND layout: [tokens, heads, head_dim]
            q_np = self.rng.standard_normal((q_ntok, heads, head_dim)).astype(np_dtype)
            k_np = self.rng.standard_normal((kv_ntok, kvh, head_dim)).astype(np_dtype)
            v_np = self.rng.standard_normal((kv_ntok, kvh, head_dim)).astype(np_dtype)
        
        return q_np, k_np, v_np
    
    # ========== Mask Generation ==========
    
    @staticmethod
    def _get_pre_mask_coef(np_dtype: np.dtype, is_alibi: bool = False) -> float:
        """Get pre-mask coefficient for operator input (following reference implementation)."""
        if np_dtype == np.float16:
            return -10000.0
        elif np_dtype == np_bfloat16 and is_alibi:
            return -float("inf")
        elif np_dtype == np.float32 and is_alibi:
            return 1.0
        else:  # bf16 non-alibi
            return 1.0
    
    @staticmethod
    def _get_post_mask_coef(np_dtype: np.dtype, is_alibi: bool = False) -> float:
        """Get post-mask coefficient for golden computation (following reference implementation)."""
        if np_dtype == np.float16:
            return 1.0
        elif np_dtype == np_bfloat16 and is_alibi:
            return 1.0
        elif np_dtype == np.float32 and is_alibi:
            return 1.0
        else:  # bf16 non-alibi
            return -3e38
    
    @classmethod
    def _build_swa_mask(cls, max_seq: int, window_size: int, mask_coef: float, mask_shape: tuple,
                       is_compress: bool, head_dim: int = None) -> np.ndarray:
        """Build SWA mask (normal or compressed)"""
        if is_compress:
            swa_mask = np.ones((1, 512, 512), dtype=np.float32) * mask_coef
            
            # Calculate true window size
            pp_n = 128 if head_dim <= 128 else 64
            if window_size <= pp_n * 3:
                true_size = window_size
            elif window_size % pp_n == 0:
                true_size = pp_n * 3
            else:
                true_size = pp_n * 2 + window_size % pp_n
            
            # Apply SWA compress pattern
            triu_mask = np.triu(swa_mask, 1)
            tril_mask = np.tril(swa_mask, -true_size)
            swa_mask = triu_mask + tril_mask
        else:
            # Normal SWA: use the provided mask_shape
            swa_mask = np.ones(mask_shape, dtype=np.float32) * mask_coef
            
            # For encoder: apply SWA pattern if window_size < max_seq
            if window_size < max_seq:
                triu_mask = np.triu(swa_mask, 1)
                tril_mask = np.tril(swa_mask, -window_size)
                swa_mask = triu_mask + tril_mask
            else:
                # Window larger than sequence: just upper triangle
                swa_mask = np.triu(swa_mask, 1)
        return swa_mask
    
    @classmethod
    def _build_alibi_mask(cls, max_seq: int, heads: int, alibi_slopes: np.ndarray, 
                         mask_coef: float, mask_shape: tuple) -> np.ndarray:
        """Build ALIBI mask with positional bias."""
        if alibi_slopes is None or heads is None:
            # Fallback to triangular mask
            mask = np.ones(mask_shape, dtype=np.float32) * mask_coef
            return np.triu(mask, 1)
        
        # Create base triangular mask + ALIBI bias
        mask = np.ones(mask_shape, dtype=np.float32) * mask_coef
        base_triu = np.triu(mask, 1)
        
        # Add ALIBI positional bias
        q_pos = np.arange(max_seq, dtype=np.float32).reshape(-1, 1)
        k_pos = np.arange(max_seq, dtype=np.float32).reshape(1, -1)
        alibi_bias = k_pos - q_pos  # Relative position matrix
        
        batch = mask_shape[0]
        for b in range(batch):
            for h in range(heads):
                base_triu[b, h] += alibi_slopes[h] * alibi_bias
                
        return base_triu
    
    def build_operator_mask(self, mask_type: int, max_seq: int, np_dtype: np.dtype, batch: int,
                           window_size: int = 0, heads: int = None, head_dim: int = None, 
                           alibi_slopes: np.ndarray = None) -> np.ndarray:
        """Build mask tensor for operator input."""
        # Get pre-mask coefficient for operator input and determine shape
        mask_coef = self._get_pre_mask_coef(np_dtype, mask_type == MASK_ALIBI)
        if mask_type == MASK_ALIBI:
            mask_shape = (batch, heads, max_seq, max_seq)
        elif mask_type == MASK_NORM:
            mask_shape = (batch, max_seq, max_seq)
        else:
            mask_shape = (max_seq, max_seq)
        
        # Build mask based on type
        if mask_type == MASK_NONE:
            return None
        elif mask_type == MASK_NORM:
            mask = np.ones(mask_shape, dtype=np.float32) * mask_coef
            return np.triu(mask, 1)
        elif mask_type == MASK_NORM_COMPRESS:
            # NORM_COMPRESS uses fixed (128, 128) shape as in reference implementation
            compress_mask = np.ones((128, 128), dtype=np.float32) * mask_coef
            result_mask = np.triu(compress_mask, 1)
            return result_mask
        elif mask_type in (MASK_SWA_NORM, MASK_SWA_COMPRESS):
            swa_mask = self._build_swa_mask(max_seq, window_size, mask_coef, mask_shape, 
                                           mask_type == MASK_SWA_COMPRESS, head_dim)
            # SWA_COMPRESS returns (1, 512, 512), reshape to (512, 512) for operator as in reference
            if mask_type == MASK_SWA_COMPRESS and swa_mask.shape == (1, 512, 512):
                return swa_mask.reshape(512, 512)
            return swa_mask
        elif mask_type == MASK_ALIBI:
            return self._build_alibi_mask(max_seq, heads, alibi_slopes, mask_coef, mask_shape)
        else:
            return np.zeros(mask_shape, dtype=np.float32)
    
    def build_golden_mask(self, mask_type: int, ql: int, kl: int, np_dtype: np.dtype, 
                         batch_idx: int = 0, head_idx: int = 0, op_mask: np.ndarray = None, **kwargs) -> np.ndarray:
        """Build mask slice for golden reference computation."""
        if mask_type == MASK_NONE:
            return None
        
        # Special handling for compress masks: golden calculation uses actual sequence length
        if mask_type == MASK_NORM_COMPRESS:
            # Generate actual-size mask for golden computation (not from op_mask)
            pre_mask_coef = self._get_pre_mask_coef(np_dtype, False)
            post_mask_coef = self._get_post_mask_coef(np_dtype, False)
            golden_mask = np.ones((ql, kl), dtype=np.float32) * pre_mask_coef
            result_mask = np.triu(golden_mask, 1)
            # Apply post coefficient for golden computation
            return result_mask * post_mask_coef
        
        elif mask_type == MASK_SWA_COMPRESS:
            # Generate actual-size SWA mask for golden computation (not from 512x512 op_mask)
            pre_mask_coef = self._get_pre_mask_coef(np_dtype, False)
            post_mask_coef = self._get_post_mask_coef(np_dtype, False)
            golden_mask = np.ones((ql, kl), dtype=np.float32) * pre_mask_coef
            
            # Apply SWA pattern with actual sequence length
            window_size = kwargs.get('window_size', 0)
            max_seq = max(ql, kl)
            if window_size < max_seq:
                triu_mask = np.triu(golden_mask, 1)
                tril_mask = np.tril(golden_mask, -window_size)
                result_mask = triu_mask + tril_mask
            else:
                result_mask = np.triu(golden_mask, 1)
            
            # Apply post coefficient for golden computation
            return result_mask * post_mask_coef
        
        # For other mask types, extract from op_mask
        if op_mask is None:
            return None
            
        # Extract appropriate slice based on mask dimensions
        if mask_type == MASK_ALIBI and op_mask.ndim == 4:
            mask_slice = op_mask[batch_idx, head_idx, :ql, :kl]
        elif op_mask.ndim >= 3:
            mask_slice = op_mask[batch_idx, :ql, :kl]
        else:
            mask_slice = op_mask[:ql, :kl]
            
        # Apply post coefficient for golden computation
        post_mask_coef = self._get_post_mask_coef(np_dtype, mask_type == MASK_ALIBI)
        return mask_slice.astype(np.float32) * post_mask_coef
    
    # ========== Golden Reference ==========
    
    def compute_golden_attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, q_seq: np.ndarray,
                                kv_seq: np.ndarray, heads: int, scale: float, kv_heads: int = None,
                                mask: np.ndarray = None, mask_type: int = MASK_NONE, **kwargs) -> np.ndarray:
        """Compute golden reference attention output."""
        kv_heads = kv_heads or heads
        head_dim = q.shape[-1] // heads
        q_tokens = q.reshape(-1, heads, head_dim).astype(np.float32)
        k_tokens = k.reshape(-1, kv_heads, head_dim).astype(np.float32)
        v_tokens = v.reshape(-1, kv_heads, head_dim).astype(np.float32)
        
        out = np.zeros_like(q_tokens)
        q_off = kv_off = 0
        
        for b in range(len(q_seq)):
            ql, kl = int(q_seq[b]), int(kv_seq[b])
            if ql == 0:
                continue
                
            qs = q_tokens[q_off:q_off + ql]
            if kl == 0:
                out[q_off:q_off + ql] = 0.0
            else:
                ks = k_tokens[kv_off:kv_off + kl]
                vs = v_tokens[kv_off:kv_off + kl]
                
                for h in range(heads):
                    kh = h // (heads // kv_heads)  # GQA support
                    logits = np.einsum("qd,kd->qk", qs[:, h, :], ks[:, kh, :]) * scale
                    
                    # Apply mask if present
                    mask_slice = self.build_golden_mask(
                        mask_type, ql, kl, kwargs.get('np_dtype', np.float16), b, h, mask,
                        window_size=kwargs.get('window_size', 0)
                    )
                    if mask_slice is not None:
                        logits += mask_slice
                    
                    # Softmax and attention
                    logits_max = np.max(logits, axis=1, keepdims=True)
                    exp_logits = np.exp(logits - logits_max)
                    attn_weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                    out[q_off:q_off + ql, h, :] = np.einsum("qk,kd->qd", attn_weights, vs[:, kh, :])
                    
            q_off += ql
            kv_off += kl
            
        return out.reshape(-1, heads * head_dim)
    
    # ========== Result Validation ==========
    
    def validate_output(self, out: np.ndarray, golden: np.ndarray, np_dtype: np.dtype,
                        heads: int, max_seq: int) -> bool:
        """Validate operator output against golden reference with adaptive precision."""
        golden_flat = golden.flatten().astype(np.float32)
        out_flat = out.flatten().astype(np.float32)
        diff = np.abs(golden_flat - out_flat)
        out_len = out_flat.shape[0]
        max_diff = np.max(diff)

        # Legacy standard with fixed ratios
        if np_dtype == np_bfloat16:
            ratios = [0.001, 0.001, 0.005, 0.005]  # [rel_loose, abs_loose, rel_strict, abs_strict]
        else:  # fp16
            ratios = [0.001, 0.001, 0.005, 0.005]

        # Calculate accuracy metrics
        limit_error = np.maximum(np.abs(golden_flat) * ratios[0], ratios[1])
        strict_limit_error = np.maximum(np.abs(golden_flat) * ratios[2], ratios[3])
        error_count = np.sum(diff > limit_error)
        strict_error_count = np.sum(diff > strict_limit_error)

        accuracy_loose = 1.0 - float(error_count) / out_len
        accuracy_strict = 1.0 - float(strict_error_count) / out_len

        print(f"Max difference: {max_diff:.6f}")
        print(f"Loose accuracy (1/1000): {accuracy_loose:.6f}")
        print(f"Strict accuracy (5/1000): {accuracy_strict:.6f}")

        # Legacy pass calculation with data type distinction
        if np_dtype == np_bfloat16:
            legacy_pass = (float(strict_error_count) / max(1, out_len)) <= ratios[2]
        else:
            legacy_pass = (float(strict_error_count) / max(1, out_len)) <= ratios[0]

        # Adaptive validation based on computation complexity
        calc_times = heads * max_seq + 4
        if np_dtype == np_bfloat16:
            error_factor = 2 ** (-7 if calc_times < 2048 else -6)
        elif np_dtype == np.float16:
            error_factor = 2 ** (-8 if calc_times < 2048 else -7)
        else:  # float32
            if calc_times < 2048:
                error_factor = 2 ** (-11)
            elif calc_times < 16384:
                error_factor = 2 ** (-10)
            else:
                error_factor = 2 ** (-9)

        # Adaptive threshold: max(|golden|, 1.0) * error_factor
        error_threshold = np.maximum(np.abs(golden_flat), 1.0) * error_factor
        adaptive_pass = np.all(diff <= error_threshold)

        print(f"Calculation complexity: {calc_times}")
        print(f"Error factor: {error_factor:.6e}")
        print(f"Adaptive precision test: {'PASS' if adaptive_pass else 'FAIL'}")
        print(f"Legacy precision test: {'PASS' if legacy_pass else 'FAIL'}")

        return adaptive_pass or legacy_pass

    # ========== Dynamic Shape Configuration ==========
    
    def set_dynamic_shapes(self, net, test_case) -> None:
        """Set dynamic input shapes for the network."""
        ms_dtype = ms.float16 if test_case.np_dtype == np.float16 else ms.bfloat16
        
        # Input shapes depend on layout
        if test_case.layout == 'TH':
            q_shape = [None, test_case.hidden]
            k_shape = v_shape = [None, test_case.kv_heads * test_case.head_dim]
        else:  # TND layout
            q_shape = [None, test_case.heads, test_case.head_dim]
            k_shape = v_shape = [None, test_case.kv_heads, test_case.head_dim]
        
        # Mask shape depends on mask type and backend:
        #  - 910B (ND path): ALIBI mask is 4D [batch, heads, max_seq, max_seq]
        #  - 310P (NZ path): ALIBI mask is 3D [heads, max_seq, max_seq]
        is_310p_nz = bool(getattr(net, 'input_format', 0) == 1)
        if test_case.mask_type == MASK_ALIBI:
            mask_shape = [None, None, None] if is_310p_nz else [None, None, None, None]
        elif test_case.mask_type == MASK_NORM:
            mask_shape = [None, None, None]
        else:
            mask_shape = [None, None]
        mask_tensor = Tensor(shape=mask_shape, dtype=ms_dtype) if test_case.mask_type != MASK_NONE else None
        # ALIBI slopes (optional)
        alibi_dyn = None
        
        net.set_inputs(
            Tensor(shape=q_shape, dtype=ms_dtype),                # q
            Tensor(shape=k_shape, dtype=ms_dtype),                # k
            Tensor(shape=v_shape, dtype=ms_dtype),                # v
            mask_tensor,                                          # mask
            alibi_dyn,                                            # alibi_slopes
            Tensor(shape=[None], dtype=ms.int32),                 # q_seq_lens
            Tensor(shape=[None], dtype=ms.int32)                  # kv_seq_lens
        )
    
    # ========== Main Test Execution ==========
    
    def run_test_case(self, test_case, dynamic: bool = False) -> None:
        """Execute a complete test case with all necessary components."""
        # 1. Generate ALIBI slopes if needed
        alibi_slopes = self._get_alibi_slopes(test_case.heads) if (test_case.mask_type == MASK_ALIBI or test_case.use_alibi) else None
        
        # 2. Build inputs based on layout
        q_data, k_data, v_data = self.build_inputs(
            test_case.q_seq, test_case.kv_seq, 
            test_case.heads, test_case.head_dim, test_case.np_dtype, 
            test_case.kv_heads, test_case.layout
        )
        
        # 3. Build operator mask
        mask_np = self.build_operator_mask(
            test_case.mask_type, test_case.max_seq, test_case.np_dtype, test_case.batch,
            window_size=test_case.window_size, heads=test_case.heads, head_dim=test_case.head_dim,
            alibi_slopes=alibi_slopes
        )
        
        # 4. Prepare inputs for golden calculation (flatten TND to TH layout)
        if test_case.layout == 'TND':
            q_golden = q_data.reshape(q_data.shape[0], -1).astype(np.float32)
            k_golden = k_data.reshape(k_data.shape[0], -1).astype(np.float32)
            v_golden = v_data.reshape(v_data.shape[0], -1).astype(np.float32)
        else:  # TH layout
            q_golden = q_data.astype(np.float32)
            k_golden = k_data.astype(np.float32)
            v_golden = v_data.astype(np.float32)
        
        # 5. Compute golden reference
        golden = self.compute_golden_attention(
            q_golden, k_golden, v_golden,
            test_case.q_seq, test_case.kv_seq, test_case.heads, test_case.scale_value, 
            kv_heads=test_case.kv_heads, mask=mask_np, mask_type=test_case.mask_type,
            alibi_slopes=alibi_slopes, window_size=test_case.window_size, np_dtype=test_case.np_dtype
        ).astype(test_case.np_dtype)
        
        # 6. Create and configure network
        net = FlashAttentionEncoderNet(
            test_case.heads, test_case.scale_value, test_case.kv_heads, test_case.mask_type,
            test_case.kernel_type, test_case.window_size, test_case.cache_type
        )
        
        if dynamic:
            self.set_dynamic_shapes(net, test_case)
        
        # 7. Execute operator
        ms_dtype = ms.float16 if test_case.np_dtype == np.float16 else ms.bfloat16
        out = net(
            self._ms_tensor(q_data), self._ms_tensor(k_data), self._ms_tensor(v_data),
            self._ms_tensor(mask_np).astype(ms_dtype) if mask_np is not None else None,
            None,
            self._ms_tensor(test_case.q_seq), self._ms_tensor(test_case.kv_seq)
        )
        
        # 8. Verify results
        out_np = (out.float().asnumpy() if test_case.np_dtype == np_bfloat16 else out.asnumpy()).astype(np.float32)
        assert self.validate_output(out_np, golden.astype(np.float32), test_case.np_dtype, test_case.heads, int(test_case.q_seq.max()))


class FlashAttentionEncoderNet(nn.Cell):
    """MindSpore network wrapper for flash_attention_encoder operator."""
    
    def __init__(self, heads: int, scale_value: float, kv_heads: int, mask_type: int,
                 kernel_type: int, window_size: int, cache_type: int):
        super().__init__()
        self.heads = heads
        self.scale_value = scale_value
        self.kv_heads = kv_heads
        self.mask_type = mask_type
        self.kernel_type = kernel_type
        self.window_size = window_size
        self.cache_type = cache_type
        # determine execution mode once during initialization
        self._is_pynative = (context.get_context("mode") == context.PYNATIVE_MODE)
    
    def construct(self, q, k, v, mask, alibi_slopes, q_seq_lens, kv_seq_lens):
        if self._is_pynative:
            q_lens_cpu = q_seq_lens.move_to("CPU")
            kv_lens_cpu = kv_seq_lens.move_to("CPU")
        else:
            q_lens_cpu = ops.move_to(q_seq_lens, "CPU")
            kv_lens_cpu = ops.move_to(kv_seq_lens, "CPU")
        return ms_custom_ops.flash_attention_encoder(
            q, k, v, None, mask, alibi_slopes, None, None, None, None, None, None,
            q_lens_cpu, kv_lens_cpu,
            self.heads, self.scale_value, self.kv_heads, self.mask_type, 
            self.kernel_type, self.window_size, self.cache_type)


class TestCase:
    """Configuration for a single test case."""
    def __init__(self, layout: str, heads: int, head_dim: int, q_seq: list, kv_seq: list,
                 np_dtype: np.dtype, kv_heads: int = None, mask_type: int = MASK_NONE,
                 kernel_type: int = KERNEL_DEFAULT, window_size: int = 0,
                 cache_type: int = CACHE_NORM, use_alibi: bool = False):
        self.layout = layout
        self.heads = heads
        self.head_dim = head_dim
        self.q_seq = np.array(q_seq, dtype=np.int32)
        self.kv_seq = np.array(kv_seq, dtype=np.int32)
        self.np_dtype = np_dtype
        self.kv_heads = kv_heads or heads
        self.mask_type = mask_type
        self.kernel_type = kernel_type
        self.window_size = window_size
        self.cache_type = cache_type
        self.use_alibi = use_alibi
        self.q_ntokens = int(sum(q_seq))
        self.kv_ntokens = int(sum(kv_seq))
        self.hidden = heads * head_dim
        self.max_seq = max(max(q_seq), max(kv_seq))
        self.batch = len(q_seq)
        self.scale_value = 1.0 / math.sqrt(float(head_dim))


def _run_test_with_case(test_case: TestCase, run_mode: int, dynamic: bool = False):
    """Execute a test case with the given run mode using the unified test suite."""
    context.set_context(device_target="Ascend", mode=run_mode)
    test_suite = FlashAttentionTestSuite()
    test_suite.run_test_case(test_case, dynamic)


# Common test decorators
def _flash_attention_test(test_func):
    """Apply common decorators for flash attention encoder tests."""
    decorators = [
        pytest.mark.level0,
        pytest.mark.platform_ascend910b,
        pytest.mark.env_onecard,
        pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE]),
        pytest.mark.parametrize('dynamic', [True, False]),
        pytest.mark.parametrize('np_dtype', [np.float16, np_bfloat16])
    ]
    
    for decorator in reversed(decorators):
        test_func = decorator(test_func)
    return test_func


@_flash_attention_test
def test_flash_attention_encoder_th_layout(run_mode, dynamic, np_dtype):
    """
    Feature: FlashAttention encoder - TH layout basic
    Description: Random Q/K/V in TH layout with fp16/bf16 across GRAPH/PYNATIVE and dynamic/static shapes
    Expectation: Operator executes successfully and output matches golden attention within tolerance
    """
    test_case = TestCase('TH', 4, 32, [48, 16], [48, 16], np_dtype)
    _run_test_with_case(test_case, run_mode, dynamic)


@_flash_attention_test
def test_flash_attention_encoder_tnd_layout(run_mode, dynamic, np_dtype):
    """
    Feature: FlashAttention encoder - TND layout basic
    Description: Random Q/K/V in TND layout with fp16/bf16 across GRAPH/PYNATIVE and dynamic/static shapes
    Expectation: Operator executes successfully and output matches golden attention within tolerance
    """
    test_case = TestCase('TND', 2, 64, [20, 12], [20, 12], np_dtype)
    _run_test_with_case(test_case, run_mode, dynamic)


@_flash_attention_test
def test_flash_attention_encoder_th_high_precision(run_mode, dynamic, np_dtype):
    """
    Feature: FlashAttention encoder - TH layout high precision kernel
    Description: TH layout, fp16/bf16, kernel_type=HIGH_PRECISION across GRAPH/PYNATIVE and dynamic/static shapes
    Expectation: Operator executes successfully and output matches golden attention within tolerance
    """
    test_case = TestCase('TH', 4, 32, [48, 16], [48, 16], np_dtype, kernel_type=KERNEL_HIGH_PRECISION)
    _run_test_with_case(test_case, run_mode, dynamic)


@_flash_attention_test
def test_flash_attention_encoder_tnd_high_precision(run_mode, dynamic, np_dtype):
    """
    Feature: FlashAttention encoder - TND layout high precision kernel
    Description: TND layout, fp16/bf16, kernel_type=HIGH_PRECISION across GRAPH/PYNATIVE and dynamic/static shapes
    Expectation: Operator executes successfully and output matches golden attention within tolerance
    """
    test_case = TestCase('TND', 2, 64, [20, 12], [20, 12], np_dtype, kernel_type=KERNEL_HIGH_PRECISION)
    _run_test_with_case(test_case, run_mode, dynamic)


@_flash_attention_test
def test_flash_attention_encoder_th_norm_mask(run_mode, dynamic, np_dtype):
    """
    Feature: FlashAttention encoder - TH layout with normal causal mask
    Description: TH layout with triangular mask, fp16/bf16 across modes and dynamic/static shapes
    Expectation: Operator executes successfully and output matches golden attention within tolerance
    """
    test_case = TestCase('TH', 4, 32, [48, 16], [48, 16], np_dtype, mask_type=MASK_NORM)
    _run_test_with_case(test_case, run_mode, dynamic)


@_flash_attention_test
def test_flash_attention_encoder_tnd_norm_mask(run_mode, dynamic, np_dtype):
    """
    Feature: FlashAttention encoder - TND layout with normal causal mask
    Description: TND layout with triangular mask, fp16/bf16 across modes and dynamic/static shapes
    Expectation: Operator executes successfully and output matches golden attention within tolerance
    """
    test_case = TestCase('TND', 2, 64, [20, 12], [20, 12], np_dtype, mask_type=MASK_NORM)
    _run_test_with_case(test_case, run_mode, dynamic)


@_flash_attention_test
def test_flash_attention_encoder_tnd_gqa(run_mode, dynamic, np_dtype):
    """
    Feature: FlashAttention encoder - TND layout with GQA
    Description: TND layout using grouped-query attention (kv_heads < heads), fp16/bf16 across modes
    Expectation: Operator executes successfully and output matches golden attention within tolerance
    """
    test_case = TestCase('TND', 4, 32, [24, 16], [24, 16], np_dtype, kv_heads=2)
    _run_test_with_case(test_case, run_mode, dynamic)


@_flash_attention_test
def test_flash_attention_encoder_tnd_swa_full_window(run_mode, dynamic, np_dtype):
    """
    Feature: FlashAttention encoder - TND layout with SWA full window
    Description: Sliding Window Attention with window_size == max_seq, TND layout, fp16/bf16 across modes
    Expectation: Operator executes successfully and output matches golden attention within tolerance
    """
    test_case = TestCase('TND', 2, 64, [20, 12], [20, 12], np_dtype, 
                        mask_type=MASK_SWA_NORM, window_size=20, cache_type=CACHE_SWA)
    _run_test_with_case(test_case, run_mode, dynamic)


@_flash_attention_test
def test_flash_attention_encoder_tnd_swa_compress(run_mode, dynamic, np_dtype):
    """
    Feature: FlashAttention encoder - TND layout with SWA compress mask
    Description: Sliding Window Attention compressed mask (512x512 op mask), TND layout, fp16/bf16
    Expectation: Operator executes successfully and output matches golden attention within tolerance
    """
    test_case = TestCase('TND', 2, 64, [20, 12], [20, 12], np_dtype, 
                        mask_type=MASK_SWA_COMPRESS, window_size=8, cache_type=CACHE_SWA)
    _run_test_with_case(test_case, run_mode, dynamic)


@_flash_attention_test
def test_flash_attention_encoder_th_nomask(run_mode, dynamic, np_dtype):
    """
    Feature: FlashAttention encoder - TH layout without mask
    Description: No mask applied, TH layout, fp16/bf16 across GRAPH/PYNATIVE and dynamic/static shapes
    Expectation: Operator executes successfully and output matches golden attention within tolerance
    """
    test_case = TestCase('TH', 2, 64, [20, 12], [20, 12], np_dtype)
    _run_test_with_case(test_case, run_mode, dynamic)


@_flash_attention_test
def test_flash_attention_encoder_tnd_norm_compress(run_mode, dynamic, np_dtype):
    """
    Feature: FlashAttention encoder - TND layout with normal compress mask
    Description: Triangular mask with compressed operator mask (128x128), TND layout, fp16/bf16
    Expectation: Operator executes successfully and output matches golden attention within tolerance
    """
    test_case = TestCase('TND', 2, 64, [20, 12], [20, 12], np_dtype, mask_type=MASK_NORM_COMPRESS)
    _run_test_with_case(test_case, run_mode, dynamic)


@_flash_attention_test
def test_flash_attention_encoder_tnd_alibi(run_mode, dynamic, np_dtype):
    """
    Feature: FlashAttention encoder - TND layout with ALIBI mask
    Description: ALIBI positional bias mask per head with generated slopes, TND layout, fp16/bf16 across modes
    Expectation: Operator executes successfully and output matches golden attention within tolerance
    """
    test_case = TestCase('TND', 4, 32, [24, 16], [24, 16], np_dtype, 
                        mask_type=MASK_ALIBI, use_alibi=True)
    _run_test_with_case(test_case, run_mode, dynamic)

