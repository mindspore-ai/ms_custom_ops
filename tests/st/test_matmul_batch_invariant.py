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

""" tests matmul_batch_invariant """

import pytest
from functools import wraps
import numpy as np
import mindspore as ms
from mindspore import Tensor, context
import ms_custom_ops


def jit(func):
    @wraps(func)
    def decorator(*args, **kwargs):
        if ms.get_context("mode") == ms.PYNATIVE_MODE:
            return func(*args, **kwargs)
        return ms.jit(func, jit_level="O0", infer_boost="on")(*args, **kwargs)
    return decorator


class MatmulBatchInvariantNet(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul_batch_invariant = ms_custom_ops.matmul_batch_invariant

    @jit
    def construct(self, x1, x2, cube_math_type=0):
        out = self.matmul_batch_invariant(x1, x2, cube_math_type=cube_math_type)
        return out


@pytest.mark.skip("CI need to install ops-batchinvariant-dev package")
@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [ms.float16, ms.bfloat16])
def test_matmul_batch_invariant_2d_basic(exec_mode, dtype):
    """
    Feature: Test matmul_batch_invariant 2D basic functionality.
    Description: Test basic 2D matrix multiplication.
    Expectation: Assert that results are consistent with mindspore.mint.matmul.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    matmul_net = MatmulBatchInvariantNet()

    m, k, n = 128, 256, 64
    np_dtype = np.float16 if dtype == ms.float16 else np.float32
    x1 = np.random.randn(m, k).astype(np_dtype)
    x2 = np.random.randn(k, n).astype(np_dtype)

    ms_x1 = Tensor(x1, dtype=dtype)
    ms_x2 = Tensor(x2, dtype=dtype)

    # Golden data using mint.matmul
    expected = ms.mint.matmul(ms_x1, ms_x2)
    output = matmul_net(ms_x1, ms_x2)

    assert output.shape == expected.shape
    assert np.allclose(expected.astype(ms.float32).asnumpy(),
                       output.astype(ms.float32).asnumpy(), rtol=1e-2, atol=1e-2)


@pytest.mark.skip("bmm is not supported now")
@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [ms.float16, ms.bfloat16])
def test_matmul_batch_invariant_3d_batch(exec_mode, dtype):
    """
    Feature: Test matmul_batch_invariant 3D batch functionality.
    Description: Test 3D batch matrix multiplication.
    Expectation: Assert that results are consistent with mindspore.mint.matmul.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    matmul_net = MatmulBatchInvariantNet()

    batch, m, k, n = 4, 64, 128, 32
    np_dtype = np.float16 if dtype == ms.float16 else np.float32
    x1 = np.random.randn(batch, m, k).astype(np_dtype)
    x2 = np.random.randn(batch, k, n).astype(np_dtype)

    ms_x1 = Tensor(x1, dtype=dtype)
    ms_x2 = Tensor(x2, dtype=dtype)

    # Golden data using mint.matmul
    expected = ms.mint.matmul(ms_x1, ms_x2)
    output = matmul_net(ms_x1, ms_x2)

    assert output.shape == expected.shape
    assert np.allclose(expected.astype(ms.float32).asnumpy(),
                       output.astype(ms.float32).asnumpy(), rtol=1e-2, atol=1e-2)


@pytest.mark.skip("bmm is not supported now")
@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [ms.float16, ms.bfloat16])
def test_matmul_batch_invariant_4d_multihead(exec_mode, dtype):
    """
    Feature: Test matmul_batch_invariant 4D multi-head attention style.
    Description: Test 4D batch matrix multiplication (common in multi-head attention).
    Expectation: Assert that results are consistent with mindspore.mint.matmul.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    matmul_net = MatmulBatchInvariantNet()

    batch, heads, m, k, n = 2, 8, 32, 64, 32
    np_dtype = np.float16 if dtype == ms.float16 else np.float32
    x1 = np.random.randn(batch, heads, m, k).astype(np_dtype)
    x2 = np.random.randn(batch, heads, k, n).astype(np_dtype)

    ms_x1 = Tensor(x1, dtype=dtype)
    ms_x2 = Tensor(x2, dtype=dtype)

    # Golden data using mint.matmul
    expected = ms.mint.matmul(ms_x1, ms_x2)
    output = matmul_net(ms_x1, ms_x2)

    assert output.shape == expected.shape
    assert np.allclose(expected.astype(ms.float32).asnumpy(),
                       output.astype(ms.float32).asnumpy(), rtol=1e-2, atol=1e-2)


@pytest.mark.skip("bmm is not supported now")
@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_matmul_batch_invariant_5d(exec_mode):
    """
    Feature: Test matmul_batch_invariant 5D functionality.
    Description: Test 5D batch matrix multiplication.
    Expectation: Assert that results are consistent with mindspore.mint.matmul.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    matmul_net = MatmulBatchInvariantNet()

    d1, d2, d3, m, k, n = 2, 3, 4, 16, 32, 16
    x1 = np.random.randn(d1, d2, d3, m, k).astype(np.float16)
    x2 = np.random.randn(d1, d2, d3, k, n).astype(np.float16)

    ms_x1 = Tensor(x1, dtype=ms.float16)
    ms_x2 = Tensor(x2, dtype=ms.float16)

    # Golden data using mint.matmul
    expected = ms.mint.matmul(ms_x1, ms_x2)
    output = matmul_net(ms_x1, ms_x2)

    assert output.shape == expected.shape
    assert np.allclose(expected.astype(ms.float32).asnumpy(),
                       output.astype(ms.float32).asnumpy(), rtol=1e-2, atol=1e-2)


@pytest.mark.skip("bmm is not supported now")
@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_matmul_batch_invariant_6d(exec_mode):
    """
    Feature: Test matmul_batch_invariant 6D functionality.
    Description: Test 6D batch matrix multiplication.
    Expectation: Assert that results are consistent with mindspore.mint.matmul.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    matmul_net = MatmulBatchInvariantNet()

    d1, d2, d3, d4, m, k, n = 2, 2, 2, 3, 16, 32, 16
    x1 = np.random.randn(d1, d2, d3, d4, m, k).astype(np.float16)
    x2 = np.random.randn(d1, d2, d3, d4, k, n).astype(np.float16)

    ms_x1 = Tensor(x1, dtype=ms.float16)
    ms_x2 = Tensor(x2, dtype=ms.float16)

    # Golden data using mint.matmul
    expected = ms.mint.matmul(ms_x1, ms_x2)
    output = matmul_net(ms_x1, ms_x2)

    assert output.shape == expected.shape
    assert np.allclose(expected.astype(ms.float32).asnumpy(),
                       output.astype(ms.float32).asnumpy(), rtol=1e-2, atol=1e-2)


@pytest.mark.skip("CI need to install ops-batchinvariant-dev package")
@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_matmul_batch_invariant_1d_vector_matrix(exec_mode):
    """
    Feature: Test matmul_batch_invariant 1D vector-matrix multiplication.
    Description: Test 1D x 2D -> 1D (vector-matrix multiplication).
    Expectation: Assert that results are consistent with mindspore.mint.matmul.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    matmul_net = MatmulBatchInvariantNet()

    k, n = 128, 64
    x1 = np.random.randn(k).astype(np.float16)
    x2 = np.random.randn(k, n).astype(np.float16)

    ms_x1 = Tensor(x1, dtype=ms.float16)
    ms_x2 = Tensor(x2, dtype=ms.float16)

    # Golden data using mint.matmul
    expected = ms.mint.matmul(ms_x1, ms_x2)
    output = matmul_net(ms_x1, ms_x2)

    assert output.shape == expected.shape
    assert np.allclose(expected.astype(ms.float32).asnumpy(),
                       output.astype(ms.float32).asnumpy(), rtol=1e-2, atol=1e-2)


@pytest.mark.skip("CI need to install ops-batchinvariant-dev package")
@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_matmul_batch_invariant_1d_matrix_vector(exec_mode):
    """
    Feature: Test matmul_batch_invariant 1D matrix-vector multiplication.
    Description: Test 2D x 1D -> 1D (matrix-vector multiplication).
    Expectation: Assert that results are consistent with mindspore.mint.matmul.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    matmul_net = MatmulBatchInvariantNet()

    m, k = 64, 128
    x1 = np.random.randn(m, k).astype(np.float16)
    x2 = np.random.randn(k).astype(np.float16)

    ms_x1 = Tensor(x1, dtype=ms.float16)
    ms_x2 = Tensor(x2, dtype=ms.float16)

    # Golden data using mint.matmul
    expected = ms.mint.matmul(ms_x1, ms_x2)
    output = matmul_net(ms_x1, ms_x2)

    assert output.shape == expected.shape
    assert np.allclose(expected.astype(ms.float32).asnumpy(),
                       output.astype(ms.float32).asnumpy(), rtol=1e-2, atol=1e-2)


@pytest.mark.skip("bmm is not supported now")
@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_matmul_batch_invariant_broadcast_batch(exec_mode):
    """
    Feature: Test matmul_batch_invariant batch broadcast functionality.
    Description: Test batch dimension broadcasting (one input has batch=1).
    Expectation: Assert that results are consistent with mindspore.mint.matmul.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    matmul_net = MatmulBatchInvariantNet()

    batch, m, k, n = 4, 32, 64, 16
    x1 = np.random.randn(batch, m, k).astype(np.float16)
    x2 = np.random.randn(1, k, n).astype(np.float16)  # batch=1, will broadcast

    ms_x1 = Tensor(x1, dtype=ms.float16)
    ms_x2 = Tensor(x2, dtype=ms.float16)

    # Golden data using mint.matmul
    expected = ms.mint.matmul(ms_x1, ms_x2)
    output = matmul_net(ms_x1, ms_x2)

    assert output.shape == expected.shape
    assert np.allclose(expected.astype(ms.float32).asnumpy(),
                       output.astype(ms.float32).asnumpy(), rtol=1e-2, atol=1e-2)


@pytest.mark.skip("CI need to install ops-batchinvariant-dev package")
@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_matmul_batch_invariant_broadcast_different_ndim(exec_mode):
    """
    Feature: Test matmul_batch_invariant with different ndim broadcasting.
    Description: Test 3D x 2D broadcasting.
    Expectation: Assert that results are consistent with mindspore.mint.matmul.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    matmul_net = MatmulBatchInvariantNet()

    batch, m, k, n = 4, 32, 64, 16
    x1 = np.random.randn(batch, m, k).astype(np.float16)
    x2 = np.random.randn(k, n).astype(np.float16)  # 2D, will broadcast to batch

    ms_x1 = Tensor(x1, dtype=ms.float16)
    ms_x2 = Tensor(x2, dtype=ms.float16)

    # Golden data using mint.matmul
    expected = ms.mint.matmul(ms_x1, ms_x2)
    output = matmul_net(ms_x1, ms_x2)

    assert output.shape == expected.shape
    assert np.allclose(expected.astype(ms.float32).asnumpy(),
                       output.astype(ms.float32).asnumpy(), rtol=1e-2, atol=1e-2)


@pytest.mark.skip("CI need to install ops-batchinvariant-dev package")
@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('cube_math_type', [0, 1, 2])
def test_matmul_batch_invariant_cube_math_type(exec_mode, cube_math_type):
    """
    Feature: Test matmul_batch_invariant with different cube_math_type.
    Description: Test cube_math_type parameter for precision control.
    Expectation: Assert that output shape is correct.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    matmul_net = MatmulBatchInvariantNet()

    m, k, n = 64, 128, 32
    x1 = np.random.randn(m, k).astype(np.float16)
    x2 = np.random.randn(k, n).astype(np.float16)

    ms_x1 = Tensor(x1, dtype=ms.float16)
    ms_x2 = Tensor(x2, dtype=ms.float16)

    output = matmul_net(ms_x1, ms_x2, cube_math_type=cube_math_type)

    # Verify output shape is correct
    assert output.shape == (m, n)
