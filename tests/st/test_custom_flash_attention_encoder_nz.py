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

"""Test suite for custom flash attention encoder with NZ format using MindSpore.

This module contains tests to validate the integration and behavior of the
flash_attention_encoder operator with inputs and outputs in the NZ (non-contiguous)
format, ensuring proper conversion and correctness under various scenarios.
"""
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, context, ops, nn
from mindspore.common.np_dtype import bfloat16 as np_bfloat16
from test_custom_flash_attention_encoder import (
    FlashAttentionTestSuite,
    TestCase,
    MASK_NORM,
    MASK_ALIBI,
    MASK_SWA_NORM,
    MASK_SWA_COMPRESS,
    CACHE_SWA,
)
import ms_custom_ops

class FlashAttentionEncoderNzNet(nn.Cell):
    """Network wrapper for flash_attention_encoder with NZ format conversion.

    This network converts ND format inputs to NZ format within the graph,
    ensuring trans_data operations are included in graph mode compilation.
    """
    def __init__(self, heads: int, scale_value: float, kv_heads: int, mask_type: int,
                 kernel_type: int, window_size: int, cache_type: int,
                 input_format: int = 1, inner_precise: int = 0):
        super().__init__()
        self.heads = heads
        self.scale_value = scale_value
        self.kv_heads = kv_heads
        self.mask_type = mask_type
        self.kernel_type = kernel_type
        self.window_size = window_size
        self.cache_type = cache_type
        self.input_format = input_format
        self.inner_precise = inner_precise
        self._is_pynative = context.get_context("mode") == context.PYNATIVE_MODE

    def construct(self, q_nd, k_nd, v_nd, mask_nd, alibi_slopes, q_seq_lens, kv_seq_lens):
        """
        Construct the forward pass of the network.

        Converts input tensors from ND to NZ format, applies the flash_attention_encoder
        operation in NZ format, and converts the output back to ND format. Sequence lengths
        are moved to CPU for operator compatibility. This ensures all format conversions are
        included inside the graph for graph mode compilation.

        Args:
            q_nd (Tensor): Query tensor in ND format.
            k_nd (Tensor): Key tensor in ND format.
            v_nd (Tensor): Value tensor in ND format.
            mask_nd (Tensor): Mask tensor in ND format, or None.
            alibi_slopes (Tensor): ALIBI slopes for positional bias, or None.
            q_seq_lens (Tensor): Query sequence lengths tensor.
            kv_seq_lens (Tensor): Key/Value sequence lengths tensor.

        Returns:
            Tensor: Output tensor in ND format after flash attention and format conversions.
        """
        q_nz = ms_custom_ops.trans_data(q_nd, transdata_type=1)
        k_nz = ms_custom_ops.trans_data(k_nd, transdata_type=1)
        v_nz = ms_custom_ops.trans_data(v_nd, transdata_type=1)

        # Convert mask to NZ if present
        if mask_nd is not None:
            mask_nz = ms_custom_ops.trans_data(mask_nd, transdata_type=1)
        else:
            mask_nz = None

        # Move sequence lengths to CPU
        if self._is_pynative:
            q_lens_cpu = q_seq_lens.move_to("CPU")
            kv_lens_cpu = kv_seq_lens.move_to("CPU")
        else:
            q_lens_cpu = ops.move_to(q_seq_lens, "CPU")
            kv_lens_cpu = ops.move_to(kv_seq_lens, "CPU")

        # Execute flash attention with NZ format
        out_nz = ms_custom_ops.flash_attention_encoder(
            q_nz, k_nz, v_nz, None, mask_nz, alibi_slopes, None, None, None, None, None, None,
            q_lens_cpu, kv_lens_cpu,
            self.heads, self.scale_value, self.kv_heads, self.mask_type,
            self.kernel_type, self.window_size, self.cache_type,
            input_format=self.input_format, inner_precise=self.inner_precise)

        # Convert output back to ND format within graph
        out_nd = ms_custom_ops.trans_data(out_nz, transdata_type=0)
        return out_nd


def _ms_tensor(x: np.ndarray) -> Tensor:
    if x is None:
        return None
    return Tensor(x.astype(np.float32)).astype(ms.bfloat16) if x.dtype == np_bfloat16 else Tensor(x)


def _run_nz_test(test_case: TestCase, run_mode: int, dynamic: bool = False,
                 inner_precise: int = 0):
    """
    Run FlashAttention encoder kernel in NZ format and validate against reference implementation.

    Args:
        test_case (TestCase): Test parameters including shapes, types, and mask config.
        run_mode (int): MindSpore context mode (GRAPH_MODE or PYNATIVE_MODE).
        dynamic (bool, optional): Use dynamic shape testing if True.

    Returns:
        None. Asserts if test does not match expected output.
    """
    context.set_context(device_target="Ascend", mode=run_mode)
    test_suite = FlashAttentionTestSuite()

    alibi_slopes = (
        test_suite.get_alibi_slopes(test_case.heads)
        if (test_case.mask_type == MASK_ALIBI or getattr(test_case, 'use_alibi', False))
        else None
    )
    q_data, k_data, v_data = test_suite.build_inputs(
        test_case.q_seq, test_case.kv_seq,
        test_case.heads, test_case.head_dim, test_case.np_dtype,
        test_case.kv_heads, test_case.layout
    )

    mask_np = test_suite.build_operator_mask(
        test_case.mask_type, test_case.max_seq, test_case.np_dtype, test_case.batch,
        window_size=test_case.window_size, heads=test_case.heads, head_dim=test_case.head_dim,
        alibi_slopes=alibi_slopes
    )

    if test_case.layout == 'TND':
        q_golden = q_data.reshape(q_data.shape[0], -1).astype(np.float32)
        k_golden = k_data.reshape(k_data.shape[0], -1).astype(np.float32)
        v_golden = v_data.reshape(v_data.shape[0], -1).astype(np.float32)
    else:
        q_golden = q_data.astype(np.float32)
        k_golden = k_data.astype(np.float32)
        v_golden = v_data.astype(np.float32)

    golden = test_suite.compute_golden_attention(
        q_golden, k_golden, v_golden,
        test_case.q_seq, test_case.kv_seq, test_case.heads, test_case.scale_value,
        kv_heads=test_case.kv_heads, mask=mask_np, mask_type=test_case.mask_type,
        alibi_slopes=alibi_slopes, window_size=test_case.window_size, np_dtype=test_case.np_dtype
    ).astype(test_case.np_dtype)

    net = FlashAttentionEncoderNzNet(
        test_case.heads, test_case.scale_value, test_case.kv_heads, test_case.mask_type,
        test_case.kernel_type, test_case.window_size, test_case.cache_type,
        input_format=1, inner_precise=inner_precise
    )

    if dynamic:
        # Reuse dynamic shape setup from original suite
        test_suite.set_dynamic_shapes(net, test_case)

    ms_dtype = ms.float16 if test_case.np_dtype == np.float16 else ms.bfloat16

    # Prepare ND format tensors (trans_data will be called inside Net)
    q_ms = _ms_tensor(q_data).astype(ms_dtype)
    k_ms = _ms_tensor(k_data).astype(ms_dtype)
    v_ms = _ms_tensor(v_data).astype(ms_dtype)

    # Prepare mask in ND format
    if test_case.mask_type == MASK_ALIBI:
        mask_np = mask_np.reshape(test_case.batch * test_case.heads, test_case.max_seq, test_case.max_seq)
    if mask_np is not None:
        mask_ms = _ms_tensor(mask_np).astype(ms_dtype)
    else:
        mask_ms = None

    # Call Net with ND format inputs (trans_data happens inside the graph)
    out_nd = net(
        q_ms, k_ms, v_ms,
        mask_ms,
        None,
        _ms_tensor(test_case.q_seq), _ms_tensor(test_case.kv_seq)
    )

    # Validate output (already in ND format from Net)
    out_np = (out_nd.float().asnumpy() if test_case.np_dtype == np_bfloat16 else out_nd.asnumpy()).astype(np.float32)
    assert test_suite.validate_output(
        out_np,
        golden.astype(np.float32),
        test_case.np_dtype,
        test_case.heads,
        int(test_case.q_seq.max())
    )


def _flash_attention_nz_test(func):
    """
    Decorator to parametrize NZ FlashAttention encoder tests.

    Applies the following pytest markers and parameterizations:
      - level0
      - platform_ascend310p
      - env_onecard
      - run_mode: GRAPH_MODE and PYNATIVE_MODE
      - dynamic: True and False
      - np_dtype: np.float16

    Args:
        func (function): The test function to decorate.

    Returns:
        function: The decorated test function.
    """
    decorators = [
        pytest.mark.level0,
        pytest.mark.platform_ascend310p,
        pytest.mark.env_onecard,
        pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE]),
        pytest.mark.parametrize('dynamic', [True, False]),
        pytest.mark.parametrize('np_dtype', [np.float16])
    ]
    for d in reversed(decorators):
        func = d(func)
    return func


@_flash_attention_nz_test
def test_flash_attention_encoder_th_nz_nomask(run_mode, dynamic, np_dtype):
    """
    Feature: FlashAttention encoder (NZ) - TH layout without mask
    Description: Convert ND Q/K/V to NZ via trans_data, input_format=1, run across modes and dynamic/static shapes
    Expectation: Operator executes successfully and output matches golden attention within tolerance
    """
    test_case = TestCase('TH', 2, 64, [16, 16], [16, 16], np_dtype)
    _run_nz_test(test_case, run_mode, dynamic)


@_flash_attention_nz_test
def test_flash_attention_encoder_th_nz_norm_mask(run_mode, dynamic, np_dtype):
    """
    Feature: FlashAttention encoder (NZ) - TH layout with normal causal mask
    Description: Trans_data (ND->NZ) within Net graph for Q/K/V/Mask, input_format=1,
        run across modes and dynamic/static shapes
    Expectation: Operator executes successfully and output matches golden attention
        within tolerance
    """
    test_case = TestCase('TH', 2, 64, [16, 16], [16, 16], np_dtype, mask_type=MASK_NORM)
    _run_nz_test(test_case, run_mode, dynamic)


@_flash_attention_nz_test
def test_flash_attention_encoder_th_nz_alibi_mask(run_mode, dynamic, np_dtype):
    """
    Feature: FlashAttention encoder (NZ) - TH layout with ALIBI mask
    Description: ALIBI mask per head flattened to [B*H,S,S], trans_data (ND->NZ) within Net graph, input_format=1
    Expectation: Operator executes successfully and output matches golden attention within tolerance
    """
    test_case = TestCase('TH', 4, 32, [16, 16], [16, 16], np_dtype, mask_type=MASK_ALIBI, use_alibi=True)
    _run_nz_test(test_case, run_mode, dynamic)


@_flash_attention_nz_test
def test_flash_attention_encoder_th_nz_swa_norm(run_mode, dynamic, np_dtype):
    """
    Feature: FlashAttention encoder (NZ) - TH layout with SWA normal mask
    Description: Sliding Window Attention (window_size < max_seq), trans_data (ND->NZ) within Net graph, input_format=1
    Expectation: Operator executes successfully and output matches golden attention within tolerance
    """
    test_case = TestCase('TH', 2, 64, [16, 16], [16, 16], np_dtype,
                         mask_type=MASK_SWA_NORM, window_size=16, cache_type=CACHE_SWA)
    _run_nz_test(test_case, run_mode, dynamic)


@_flash_attention_nz_test
def test_flash_attention_encoder_th_nz_swa_compress(run_mode, dynamic, np_dtype):
    """
    Feature: FlashAttention encoder (NZ) - TH layout with SWA compress mask
    Description: Compressed SWA mask, trans_data (ND->NZ) within Net graph, input_format=1, window_size=8
    Expectation: Operator executes successfully and output matches golden attention within tolerance
    """
    test_case = TestCase('TH', 2, 64, [16, 16], [16, 16], np_dtype,
                         mask_type=MASK_SWA_COMPRESS, window_size=8, cache_type=CACHE_SWA)
    _run_nz_test(test_case, run_mode, dynamic)


@_flash_attention_nz_test
def test_flash_attention_encoder_th_nz_inner_precise(run_mode, dynamic, np_dtype):
    """
    Feature: FlashAttention encoder (NZ) - TH layout with inner_precise enabled
    Description: Enable higher internal precision via ``inner_precise=1`` while keeping
        TH layout and no mask, to verify the encoder works correctly with this flag
        in both GRAPH/PYNATIVE modes and dynamic/static shapes.
    Expectation: Operator executes successfully and output matches golden attention
        within tolerance when inner_precise is enabled.
    """
    test_case = TestCase('TH', 2, 64, [16, 16], [16, 16], np_dtype)
    _run_nz_test(test_case, run_mode, dynamic, inner_precise=1)
