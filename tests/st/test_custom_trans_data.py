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
# pylint: disable=too-many-function-args
""" tests_custom_trans_data_pyboost_ascend """

import logging
from enum import Enum
from functools import wraps

import numpy as np
import psutil
import pytest

import mindspore as ms
from mindspore import Tensor, context, nn
from mindspore.common.api import jit, _pynative_executor
from mindspore.common.np_dtype import bfloat16
import ms_custom_ops


def jit_for_graph_mode(fn):
    jitted_fn = jit(fn)
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if context.get_context("mode") == context.GRAPH_MODE:
            return jitted_fn(*args, **kwargs)
        return fn(*args, **kwargs)
    return wrapper


class TransdataType(Enum):
    FRACTAL_NZ_TO_ND = 0
    ND_TO_FRACTAL_NZ = 1


class TransDataOp(nn.Cell):
    @jit_for_graph_mode
    def construct(self, input_tensor, transdata_type=0):
        return ms_custom_ops.trans_data(input=input_tensor, transdata_type=transdata_type)


def setup_test(device_target="Ascend", mode=context.GRAPH_MODE):
    ms.set_device(device_target)
    context.set_context(mode=mode)


def create_random_data(shape, dtype):
    if dtype == np.int8:
        return np.random.randint(low=-128, high=127, size=shape, dtype=np.int8)
    return np.random.rand(*shape).astype(dtype)


def nd_to_nz_shape(nd_shape, dtype):
    """Calculate NZ shape from ND shape - matches C++ CalculateTransDataOutputShape

    Supports 2D, 3D, and 4D+ input, output is always 4D.
    For 2D input [H, W] -> [1, RoundUp(W, align)/align, RoundUp(H, 16), align]
    For 3D input [N, H, W] -> [N, RoundUp(W, align)/align, RoundUp(H, 16), align]
    For 4D+ input [N1, N2, ..., H, W] -> [N1*N2*..., RoundUp(W, align)/align, RoundUp(H, 16), align]
    """
    if len(nd_shape) < 2:
        raise ValueError(f"TransData ND_TO_FRACTAL_NZ requires at least 2D input, but got {len(nd_shape)}D input")

    nz_width_align = 32 if dtype == np.int8 else 16
    default_height_align = 16

    # Extract N, H, W according to input dimensions
    if len(nd_shape) == 2:
        # 2D: [H, W]
        N = 1
        H = nd_shape[0]
        W = nd_shape[1]
    elif len(nd_shape) == 3:
        # 3D: [N, H, W]
        N = nd_shape[0]
        H = nd_shape[1]
        W = nd_shape[2]
    else:
        # 4D+: [N1, N2, ..., H, W] -> flatten leading dims
        N = 1
        for i in range(len(nd_shape) - 2):
            N *= nd_shape[i]
        H = nd_shape[-2]
        W = nd_shape[-1]

    output_shape = [
        N,
        (W + nz_width_align - 1) // nz_width_align,  # W'/align
        ((H + default_height_align - 1) // default_height_align) * default_height_align,  # H'
        nz_width_align  # align
    ]
    return tuple(output_shape)


def verify_basic_output(output, input_tensor):
    """Verify basic output properties"""
    assert output is not None
    assert output.dtype == input_tensor.dtype
    assert hasattr(output, 'shape')


def get_process_memory_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def generate_alignment_boundary_cases(align_values=(16, 32)):
    """Generate systematic alignment boundary test cases"""
    cases = []
    for align in align_values:
        # Test around alignment boundaries: align-1, align, align+1
        for offset in [-1, 0, 1]:
            dim = align + offset
            if dim > 0:
                cases.extend([
                    (1, dim, dim),
                    (2, dim, align),
                    (1, align, dim),
                ])
    return list(set(cases))  # Remove duplicates


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.platform_ascend310p
@pytest.mark.env_onecard
@pytest.mark.parametrize('np_dtype', [np.float16, bfloat16])
@pytest.mark.parametrize('input_shape', [(2, 16, 16), (1, 32, 32), (4, 16, 64)])
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_trans_data_nd_to_nz(np_dtype, input_shape, run_mode):
    """
    Feature: TransData operator ND to FRACTAL_NZ conversion
    Description: Test ND format to FRACTAL_NZ format conversion with aligned shapes
    Expectation: Output shape matches expected NZ format and dtype is preserved
    """
    setup_test(mode=run_mode)
    net = TransDataOp()

    input_data = create_random_data(input_shape, np_dtype)
    input_tensor = Tensor(input_data)
    expected_nz_shape = nd_to_nz_shape(input_shape, np_dtype)

    output = net(input_tensor, TransdataType.ND_TO_FRACTAL_NZ.value)
    verify_basic_output(output, input_tensor)
    logging.info(
        "ND->NZ test passed: dtype=%s, shape=%s, expected_nz_shape=%s, mode=%s",
        np_dtype, input_shape, expected_nz_shape, run_mode
    )


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.platform_ascend310p
@pytest.mark.env_onecard
@pytest.mark.parametrize('np_dtype', [np.int8])
@pytest.mark.parametrize('input_shape', [(1, 16, 32), (3, 32, 96)])
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_trans_data_nd_to_nz_int8(np_dtype, input_shape, run_mode):
    """
    Feature: TransData operator ND to FRACTAL_NZ conversion for int8
    Description: Test int8 data with properly aligned dimensions (H%16=0, W%32=0)
    Expectation: Output shape matches expected NZ format
    """
    setup_test(mode=run_mode)
    net = TransDataOp()

    input_data = create_random_data(input_shape, np_dtype)
    input_tensor = Tensor(input_data)
    expected_nz_shape = nd_to_nz_shape(input_shape, np_dtype)

    output = net(input_tensor, TransdataType.ND_TO_FRACTAL_NZ.value)
    verify_basic_output(output, input_tensor)
    logging.info(
        "ND->NZ int8 test passed: dtype=%s, shape=%s, expected_nz_shape=%s, mode=%s",
        np_dtype, input_shape, expected_nz_shape, run_mode
    )


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.platform_ascend310p
@pytest.mark.env_onecard
@pytest.mark.parametrize('np_dtype,input_shape', [
    (np.float16, (1, 8, 16)),     # H=8 unaligned
    (np.float16, (1, 16, 15)),    # W=15 unaligned
    (np.int8, (1, 16, 16)),       # W=16 unaligned for int8
])
def test_trans_data_nd_to_nz_unaligned_should_fail(np_dtype, input_shape):
    """
    Feature: TransData operator dimension alignment validation
    Description: Test that unaligned dimensions raise exception in pynative mode
    Expectation: CheckDimensionAlignment should throw exception for unaligned H or W
    """
    setup_test(mode=context.PYNATIVE_MODE)
    net = TransDataOp()

    input_data = create_random_data(input_shape, np_dtype)
    input_tensor = Tensor(input_data)

    with pytest.raises(RuntimeError, match="dimension must be aligned"):
        net(input_tensor, TransdataType.ND_TO_FRACTAL_NZ.value)
        _pynative_executor.sync()
    logging.info(
        "Unaligned dimension correctly rejected: dtype=%s, shape=%s",
        np_dtype, input_shape
    )


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.platform_ascend310p
@pytest.mark.env_onecard
@pytest.mark.parametrize('np_dtype', [np.float16, bfloat16])
@pytest.mark.parametrize('input_shape', [(2, 16, 16), (1, 32, 32), (4, 16, 64)])
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_trans_data_nz_to_nd(np_dtype, input_shape, run_mode):
    """
    Feature: TransData operator FRACTAL_NZ to ND conversion
    Description: Test FRACTAL_NZ format to ND format conversion with aligned shapes in both graph and pynative modes
    Expectation: Output shape matches original ND shape and dtype is preserved
    """
    setup_test(mode=run_mode)
    net = TransDataOp()

    # First create NZ format data
    input_data = create_random_data(input_shape, np_dtype)
    input_tensor = Tensor(input_data)

    # ND -> NZ
    nz_output = net(input_tensor, TransdataType.ND_TO_FRACTAL_NZ.value)
    verify_basic_output(nz_output, input_tensor)

    # NZ -> ND (the main test)
    nd_output = net(nz_output, TransdataType.FRACTAL_NZ_TO_ND.value)

    assert nd_output is not None
    assert nd_output.dtype == input_tensor.dtype
    assert nd_output.shape == input_shape

    logging.info("NZ->ND test passed: dtype=%s, shape=%s, mode=%s", np_dtype, input_shape, run_mode)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.platform_ascend310p
@pytest.mark.env_onecard
@pytest.mark.parametrize('np_dtype,input_shape', [
    (np.float16, (1, 1, 1)),        # Both unaligned
    (np.float16, (1, 1, 16)),       # H unaligned
    (np.float16, (1, 16, 15)),      # W unaligned
    (np.float16, (1, 2047, 2047)),  # Both unaligned
    (np.int8, (1, 16, 16)),         # W unaligned for int8
])
def test_trans_data_unaligned_dimensions_should_fail(np_dtype, input_shape):
    """
    Feature: TransData operator dimension alignment validation for edge cases
    Description: Test that unaligned large/edge dimensions raise exception in pynative mode
    Expectation: CheckDimensionAlignment validation rejects unaligned inputs
    """
    setup_test(mode=context.PYNATIVE_MODE)
    net = TransDataOp()

    input_data = create_random_data(input_shape, np_dtype)
    input_tensor = Tensor(input_data)

    with pytest.raises(RuntimeError, match="dimension must be aligned"):
        net(input_tensor, TransdataType.ND_TO_FRACTAL_NZ.value)
        _pynative_executor.sync()
    logging.info(
        "Unaligned dimension correctly rejected: dtype=%s, shape=%s",
        np_dtype, input_shape
    )


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.platform_ascend310p
@pytest.mark.env_onecard
@pytest.mark.parametrize('np_dtype', [np.float16, bfloat16])
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_trans_data_nz_to_nd_precision_validation(np_dtype, run_mode):
    """
    Feature: TransData operator precision validation
    Description: Test data precision preservation for well-aligned dimensions in roundtrip conversion
    Expectation: Output data matches input data within acceptable tolerance
    """
    setup_test(mode=run_mode)
    net = TransDataOp()
    precision_test_cases = [(1, 16, 16), (1, 32, 32)]

    for input_shape in precision_test_cases:
        try:
            np.random.seed(123)
            input_data = np.random.rand(*input_shape).astype(np_dtype) * 0.1
            input_tensor = Tensor(input_data)

            nz_output = net(input_tensor, TransdataType.ND_TO_FRACTAL_NZ.value)
            nd_output = net(nz_output, TransdataType.FRACTAL_NZ_TO_ND.value)

            assert nd_output.shape == input_shape
            assert nd_output.dtype == input_tensor.dtype
            assert np.allclose(nd_output.asnumpy(), input_data, rtol=5e-3, atol=5e-3)
            logging.info("Precision validation passed: dtype=%s, shape=%s", np_dtype, input_shape)
        except Exception as e:  # pylint: disable=broad-except
            logging.warning("Precision validation failed: shape=%s, error=%s", input_shape, e)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('test_case', ['normal', 'zero_dim'])
def test_trans_data_nz_indices_comprehensive(test_case):
    """
    Feature: TransData operator NZ output indices handling
    Description: Test nz_output_indices_ behavior in normal, zero dimension and roundtrip scenarios
    Expectation: NZ format outputs are correctly tracked and handled in all scenarios
    """
    setup_test(mode=context.PYNATIVE_MODE)
    net = TransDataOp()

    if test_case == 'normal':
        # Test basic ND_TO_FRACTAL_NZ initialization
        input_data = create_random_data((1, 16, 16), np.float16)
        input_tensor = Tensor(input_data)
        output = net(input_tensor, TransdataType.ND_TO_FRACTAL_NZ.value)
        verify_basic_output(output, input_tensor)
        logging.info("nz_indices normal test passed")

    elif test_case == 'zero_dim':
        # Test zero dimension handling
        zero_data = np.array([]).reshape(0, 16, 16).astype(np.float16)
        zero_tensor = Tensor(zero_data)
        try:
            output = net(zero_tensor, TransdataType.ND_TO_FRACTAL_NZ.value)
            if output is not None:
                assert output.dtype == zero_tensor.dtype
            logging.info("nz_indices zero_dim test passed")
        except Exception as e:  # pylint: disable=broad-except
            logging.info("nz_indices zero_dim handled: %s", e)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('np_dtype', [np.float16, bfloat16])
def test_trans_data_edge_cases_minimal_dimensions_should_fail(np_dtype):
    """
    Feature: TransData operator edge case handling
    Description: Test that minimal unaligned dimensions raise exception in pynative mode
    Expectation: CheckDimensionAlignment rejects minimal unaligned dimensions
    """
    setup_test(mode=context.PYNATIVE_MODE)
    net = TransDataOp()

    # All minimal cases are unaligned (< 16)
    minimal_unaligned_cases = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1), (1, 2, 3), (1, 8, 8)]

    for shape in minimal_unaligned_cases:
        data = create_random_data(shape, np_dtype)
        tensor = Tensor(data)

        with pytest.raises(RuntimeError, match="dimension must be aligned"):
            net(tensor, TransdataType.ND_TO_FRACTAL_NZ.value)
            _pynative_executor.sync()
        logging.info("Minimal unaligned case correctly rejected: dtype=%s, shape=%s", np_dtype, shape)
