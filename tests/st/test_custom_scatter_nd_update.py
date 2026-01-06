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
"""Test cases for scatter_nd_update custom operator.

This module contains test cases for the scatter_nd_update custom operator
implementation in MindSpore. It tests various data types, shapes, and
execution modes to ensure the operator functions correctly.
"""
import numpy as np
import pytest

import mindspore as ms
from mindspore import context, Tensor
from mindspore.common.np_dtype import bfloat16
import ms_custom_ops


def get_ms_dtype(query_dtype):
    ms_dtype = ms.float16
    if query_dtype == np.float32:
        ms_dtype = ms.float32
    elif query_dtype == np.float16:
        ms_dtype = ms.float16
    elif query_dtype == bfloat16:
        ms_dtype = ms.bfloat16
    return ms_dtype


def scatter_nd_update(var_ref, indices, updates):
    """
    NumPy实现scatter_nd_update功能
    参数:
        varRef: 原始数据张量 (n维)
        indices: 索引张量 (至少2维，最后一维大小k <= varRef.ndim)
        updates: 更新值张量 (形状需符合约束条件)
    返回:
        更新后的张量副本
    """
    # 复制原始数据避免修改输入
    result = np.copy(var_ref)

    # 获取关键维度信息
    k = indices.shape[-1]  # 索引维度
    idx_shape = indices.shape[:-1]  # 索引前缀形状
    var_shape = var_ref.shape

    # 验证updates形状: [idx_shape] + [var_shape[k:]]
    expected_updates_shape = idx_shape + var_shape[k:]
    if updates.shape != expected_updates_shape:
        raise ValueError(
            f"Updates shape mismatch. Expected {expected_updates_shape}, got {updates.shape}"
        )

    # 重塑索引为二维数组 [N, k]
    flat_indices = indices.reshape(-1, k)
    # 重塑更新值为 [N, ...] 形状
    flat_updates = updates.reshape(-1, *var_shape[k:])

    # 遍历每个索引
    for i in range(flat_indices.shape[0]):
        idx = tuple(flat_indices[i])
        # 验证索引边界
        if any((idx[j] < 0 or idx[j] >= var_shape[j]) for j in range(k)):
            raise IndexError(
                f"Index {idx} out of bounds for tensor shape {var_shape[:k]}"
            )

        # 构建完整切片索引
        full_idx = idx
        # 添加剩余维度的全切片
        if k < len(var_shape):
            full_idx += (slice(None),) * (len(var_shape) - k)

        # 执行更新
        result[full_idx] = flat_updates[i]

    return result


class ScatterNdUpdateNet(ms.nn.Cell):
    def _init__(self):
        super().__init__()

    def construct(self, var, indices, updates):
        return ms_custom_ops.scatter_nd_update(var, indices, updates)


def run(
        net,
        var_shape,
        indices_shape,
        updates_shape,
        dtype,
        indices_dtype,
):
    """Execute scatter_nd_update operator test and validate results.

    This function generates random test data, executes the scatter_nd_update
    operator through the provided network, and validates the output against
    a golden reference implementation.

    Args:
        net: The neural network or function containing scatter_nd_update operator.
        var_shape (list): Shape of the input tensor to be updated.
        indices_shape (list): Shape of the indices tensor.
        updates_shape (list): Shape of the updates tensor.
        dtype (numpy.dtype): Data type for the varRef and updates tensors.
        indices_dtype (numpy.dtype): Data type for the indices tensor.

    Returns:
        None

    Raises:
        AssertionError: If the output does not match the golden reference.
    """
    var = np.random.uniform(0, 1, var_shape).astype(dtype)
    indices = np.random.uniform(0, 12, indices_shape).astype(indices_dtype)
    updates = np.random.uniform(1, 2, updates_shape).astype(dtype)

    var_tensor = Tensor(var, dtype=get_ms_dtype(dtype))
    indices_tensor = Tensor(indices, dtype=ms.int64)
    updates_tensor = Tensor(updates, dtype=get_ms_dtype(dtype))
    _ = net(var_tensor, indices_tensor, updates_tensor)

    golden = scatter_nd_update(var, indices, updates)

    np.testing.assert_allclose(var_tensor.asnumpy(), golden, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("indice_dtype", [np.int64, np.int32])
@pytest.mark.parametrize("var_shape", [[24, 128]])
@pytest.mark.parametrize("indices_shape", [[12, 1]])
@pytest.mark.parametrize("updates_shape", [[12, 128]])
def test_scatter_nd_update(
        mode,
        dtype,
        indice_dtype,
        var_shape,
        indices_shape,
        updates_shape,
):
    """
    Feature: test scatter_nd_update ops precision.
    Description:test common scene.
    Expectation: success or failure.
    """
    ms.set_context(device_target="Ascend", mode=mode)
    ms.set_context(jit_config={"jit_level": "O0"})
    net = ScatterNdUpdateNet()
    run(
        net,
        var_shape,
        indices_shape,
        updates_shape,
        dtype,
        indice_dtype,
    )
