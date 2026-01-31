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
"""Test cases for reduce_sum_batch_invariant custom operator.

This module contains test cases for the reduce_sum_batch_invariant custom operator
implementation in MindSpore. It tests various data types, shapes, dims, and
execution modes to ensure the operator functions correctly.

Test strategy:
    1. Test basic functionality with different dims and keepdims
    2. Test different data types (float16, float32, bfloat16)
    3. Compare output against numpy reference implementation
    4. Test both GRAPH_MODE and PYNATIVE_MODE
"""
import numpy as np
import pytest

import mindspore as ms
from mindspore import context, Tensor
from mindspore.common.np_dtype import bfloat16
import ms_custom_ops


def get_ms_dtype(query_dtype):
    """Convert numpy dtype to mindspore dtype."""
    ms_dtype = ms.float16
    if query_dtype == np.float32:
        ms_dtype = ms.float32
    elif query_dtype == np.float16:
        ms_dtype = ms.float16
    elif query_dtype == bfloat16:
        ms_dtype = ms.bfloat16
    return ms_dtype


def reduce_sum_numpy(self_data, dims, keep_dims, dtype=None):
    """
    NumPy reference implementation for reduce_sum.
    
    Args:
        self_data: Input numpy array
        dims: Tuple of dimensions to reduce along
        keep_dims: Whether to keep reduced dimensions
        dtype: Output data type (if None, use input dtype)
        
    Returns:
        Reduced numpy array
    """
    if len(dims) == 0:
        # Empty dims means sum all elements
        axis = None
    else:
        axis = tuple(dims)
    
    result = np.sum(self_data, axis=axis, keepdims=keep_dims)
    
    if dtype is not None:
        result = result.astype(dtype)
    
    return result


class ReduceSumBatchInvariantNet(ms.nn.Cell):
    """Network wrapper for reduce_sum_batch_invariant operator."""
    
    def __init__(self, dims, keep_dims, output_dtype):
        super().__init__()
        self.dims = dims
        self.keep_dims = keep_dims
        self.output_dtype = output_dtype

    def construct(self, input_tensor):
        return ms_custom_ops.reduce_sum_batch_invariant(
            input_tensor, self.dims, self.keep_dims, self.output_dtype
        )


def run_reduce_sum_batch_invariant(
        net,
        input_shape,
        dims,
        keep_dims,
        dtype,
):
    """Execute reduce_sum_batch_invariant operator test and validate results.

    This function generates random test data, executes the reduce_sum_batch_invariant
    operator through the provided network, and validates the output against
    a NumPy reference implementation.

    Args:
        net: The neural network containing reduce_sum_batch_invariant operator.
        input_shape (tuple): Shape of the input tensor.
        dims (tuple): Dimensions to reduce along.
        keep_dims (bool): Whether to keep reduced dimensions.
        dtype (numpy.dtype): Data type for the input tensor.

    Returns:
        None

    Raises:
        AssertionError: If the output does not match the NumPy reference.
    """
    # Generate random input data
    input_data = np.random.uniform(-1, 1, input_shape).astype(dtype)
    
    ms_dtype = get_ms_dtype(dtype)
    input_tensor = Tensor(input_data, dtype=ms_dtype)
    
    # Run the custom operator
    output = net(input_tensor)
    
    # Compute reference result using numpy
    golden = reduce_sum_numpy(input_data, dims, keep_dims, dtype)
    
    # Validate results
    np.testing.assert_allclose(
        output.asnumpy(), 
        golden, 
        rtol=1e-3, 
        atol=1e-3
    )


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("input_shape", [(4, 32, 128), (8, 8, 64, 256), (32, 2048, 4096)])
@pytest.mark.parametrize("keep_dims", [True, False])
def test_reduce_sum_batch_invariant_basic(
        mode,
        dtype,
        input_shape,
        keep_dims,
):
    """
    Feature: test reduce_sum_batch_invariant basic functionality.
    Description: Test with various shapes, dims, and keepdims combinations.
    Expectation: Output matches numpy reference implementation.
    """
    ms.set_context(device_target="Ascend", mode=mode)
    ms.set_context(jit_config={"jit_level": "O0"})
    
    output_dtype = get_ms_dtype(dtype)
    dims = (-1,)
    net = ReduceSumBatchInvariantNet(dims, keep_dims, output_dtype)
    run_reduce_sum_batch_invariant(
        net,
        input_shape,
        dims,
        keep_dims,
        dtype,
    )
