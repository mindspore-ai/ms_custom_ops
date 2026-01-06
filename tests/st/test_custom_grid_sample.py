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
"""Tests for ms_custom_ops.grid_sample."""

import logging
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, nn, context, Profiler
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics
import ms_custom_ops

def bilinear_interpolate(input_tensor, x, y, h, w):
    """bilinear_interpolate"""
    x1 = int(np.floor(x))
    y1 = int(np.floor(y))
    x2 = min(x1 + 1, w - 1)
    y2 = min(y1 + 1, h - 1)

    # 边界检查
    if x1 < 0 or x1 >= w or y1 < 0 or y1 >= h:
        return np.zeros(input_tensor.shape[0], dtype=np.float32)

    # 计算权重
    wx = x - x1
    wy = y - y1

    # 双线性插值
    result = (input_tensor[y1, x1, :] * (1 - wx) * (1 - wy) +
              input_tensor[y1, x2, :] * wx * (1 - wy) +
              input_tensor[y2, x1, :] * (1 - wx) * wy +
              input_tensor[y2, x2, :] * wx * wy)
    return result


def golden_grid_sample(input_data, grid_data, align_corners, padding_mode, interpolation_mode):
    """golden_grid_sample"""
    n, h_in, w_in, c = input_data.shape
    n, h_out, w_out, _ = grid_data.shape
    output_data = np.zeros([n, h_out, w_out, c]).astype(np.float32)

    for n in range(n):
        for h in range(h_out):
            for w in range(w_out):
                # 获取归一化坐标
                x_norm = grid_data[n, h, w, 0]
                y_norm = grid_data[n, h, w, 1]

                # 映射到input坐标空间
                if align_corners:
                    x = (x_norm + 1) * (w_in - 1) / 2
                    y = (y_norm + 1) * (h_in - 1) / 2
                else:
                    x = (x_norm + 1) * w_in / 2 - 0.5
                    y = (y_norm + 1) * h_in / 2 - 0.5

                # 边界处理
                if padding_mode == 0:  # zeros
                    if x < 0 or x >= w_in or y < 0 or y >= h_in:
                        output_data[n, h, w, :] = 0
                        continue
                elif padding_mode == 1:  # border
                    x = np.clip(x, 0, w_in - 1)
                    y = np.clip(y, 0, h_in - 1)

                # 插值采样
                if interpolation_mode == 0:  # bilinear
                    output_data[n, h, w, :] = bilinear_interpolate(
                        input_data[n], x, y, h_in, w_in)
                elif interpolation_mode == 1:  # nearest
                    x_idx = int(x)
                    y_idx = int(y)
                    x_idx = np.clip(x_idx, 0, w_in - 1)
                    y_idx = np.clip(y_idx, 0, h_in - 1)
                    output_data[n, h, w, :] = input_data[n, y_idx, x_idx, :]
    return output_data


class GridSampleNet(nn.Cell):
    """GridSampleNet"""

    def construct(self, input_data, grid, interpolation_mode, padding_mode, align_corners):
        return ms_custom_ops.grid_sample(input_data, grid, interpolation_mode, padding_mode, align_corners)


def run_grid_sample(net, input_dtype, grid_dtype, align_corners, padding_mode, interpolation_mode,
                    n, c, h_in, w_in, h_out, w_out, is_profiler=False):
    """run_grid_sample"""
    np_input = np.random.random((n, h_in, w_in, c)).astype(input_dtype)
    np_grid = np.random.uniform(-1, 1, (n, h_out, w_out, 2)).astype(grid_dtype)
    input_data = Tensor(np_input)
    grid = Tensor(np_grid)
    np_output = golden_grid_sample(np_input, np_grid, align_corners, padding_mode, interpolation_mode)
    if is_profiler is False:
        output_data = net(input_data, grid, interpolation_mode, padding_mode, align_corners)
        np.testing.assert_allclose(np_output, output_data.asnumpy(), rtol=1e-4, atol=1e-4, err_msg=" grid_sample ")
    else:
        profiler = Profiler(profiler_level=ProfilerLevel.Level2,
                    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                    aic_metrics=AicoreMetrics.AiCoreNone)
        for _ in range(50):
            output_data = net(input_data, grid, interpolation_mode, padding_mode, align_corners)
        profiler.analyse()


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("input_dtype", [np.float32])
@pytest.mark.parametrize("grid_dtype", [np.float32])
@pytest.mark.parametrize("align_corners", [False])
@pytest.mark.parametrize("padding_mode", [1])
@pytest.mark.parametrize("interpolation_mode", [0])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("c", [1536, 8])
@pytest.mark.parametrize("h_in,w_in", [(24, 24)])
@pytest.mark.parametrize("h_out,w_out", [(1024, 1)])
def test_grid_sample(exec_mode, input_dtype, grid_dtype, align_corners, padding_mode, interpolation_mode,
                     n, c, h_in, w_in, h_out, w_out):
    """
    Feature: test grid_sample operator.
    Description: test correctness of grid_sample operator.
    Expectation:should pass for all testcases.
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    net =  GridSampleNet()
    run_grid_sample(net, input_dtype, grid_dtype, align_corners, padding_mode,
                    interpolation_mode, n, c, h_in, w_in, h_out, w_out)



@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("input_dtype", [np.float32])
@pytest.mark.parametrize("grid_dtype", [np.float32])
@pytest.mark.parametrize("align_corners", [False])
@pytest.mark.parametrize("padding_mode", [1])
@pytest.mark.parametrize("interpolation_mode", [0])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("c", [17])
@pytest.mark.parametrize("h_in,w_in", [(24, 24)])
@pytest.mark.parametrize("h_out,w_out", [(1024, 1)])
def test_grid_sample_align_c(exec_mode, input_dtype, grid_dtype, align_corners, padding_mode, interpolation_mode,
                             n, c, h_in, w_in, h_out, w_out):
    """
    Feature: test grid_sample operator.
    Description: test align c.
    Expectation: Unsupported c correctly rejected.
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    net =  GridSampleNet()
    with pytest.raises(ValueError, match="c should be aligned with"):
        run_grid_sample(net, input_dtype, grid_dtype, align_corners, padding_mode, interpolation_mode,
                        n, c, h_in, w_in, h_out, w_out)
    logging.info(
        "Unsupported c correctly rejected: c=%s", c
    )


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("input_dtype", [np.float32])
@pytest.mark.parametrize("grid_dtype", [np.float32])
@pytest.mark.parametrize("align_corners", [False])
@pytest.mark.parametrize("padding_mode", [1])
@pytest.mark.parametrize("interpolation_mode", [1])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("c", [16])
@pytest.mark.parametrize("h_in,w_in", [(2, 2)])
@pytest.mark.parametrize("h_out,w_out", [(2, 2)])
def test_grid_sample_unsupported_mode(exec_mode, input_dtype, grid_dtype, align_corners, padding_mode,
                                     interpolation_mode, n, c, h_in, w_in, h_out, w_out):
    """
    Feature: test grid_sample operator.
    Description: test unsupported mode.
    Expectation: Unsupported mode correctly rejected.
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0"})
    net =  GridSampleNet()
    with pytest.raises(ValueError, match="mode only supports"):
        run_grid_sample(net, input_dtype, grid_dtype, align_corners, padding_mode, interpolation_mode,
                        n, c, h_in, w_in, h_out, w_out)
    logging.info("Unsupported mode correctly rejected")


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("input_dtype", [np.float32])
@pytest.mark.parametrize("grid_dtype", [np.float32])
@pytest.mark.parametrize("align_corners", [False])
@pytest.mark.parametrize("padding_mode", [1])
@pytest.mark.parametrize("interpolation_mode", [0])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("h_in,w_in", [(24, 24)])
@pytest.mark.parametrize("h_out,w_out", [(1024, 1)])
def test_grid_sample_3d_input(exec_mode, input_dtype, grid_dtype, align_corners, padding_mode,
                              interpolation_mode, n, h_in, w_in, h_out, w_out):
    """
    Feature: test grid_sample operator.
    Description: test 3d input.
    Expectation: 3d input correctly rejected.
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    net =  GridSampleNet()
    with pytest.raises(RuntimeError, match="dim of inputs should"):
        np_input = np.random.random((n, h_in, w_in)).astype(input_dtype)
        np_grid = np.random.uniform(-1, 1, (n, h_out, w_out, 2)).astype(grid_dtype)
        input_data = Tensor(np_input)
        grid = Tensor(np_grid)
        output_data = net(input_data, grid, interpolation_mode, padding_mode, align_corners)
        _ = output_data.asnumpy()
    logging.info("3d input correctly rejected")
