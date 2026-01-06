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
import os
import random
import numpy as np
import pytest
from functools import wraps

from st_utils import custom_compare

import mindspore as ms
from mindspore import context, Tensor
from mindspore.nn import Cell
import ms_custom_ops

np.set_printoptions(precision=2, suppress=True, linewidth=200)


def jit(func):
    @wraps(func)
    def decorator(*args, **kwargs):
        if context.get_context("mode") == context.PYNATIVE_MODE:
            return func(*args, **kwargs)
        return ms.jit(func, jit_level="O0", infer_boost="on")(*args, **kwargs)
    return decorator


def split_x(x, group_list):
    x_split = []
    for i in range(len(group_list)):
        if i == 0:
            x_split.append(x[0: group_list[i],])
        else:
            x_split.append(x[group_list[i - 1]: group_list[i],])
    return x_split


def split_w(w):
    tmp_split = np.split(w, w.shape[0], axis=0)
    w_split = []
    for t in tmp_split:
        w_split.append(np.squeeze(t, 0))
    return w_split


def generate_random_numbers(m, e):
    # 生成e-1个互不相同的随机数，范围是1到n，但不包括m
    random_numbers = random.choices([i for i in range(1, m+1)], k=e-1)
    # 将m添加到列表的末尾
    random_numbers.append(m)
    # 将列表从小到大排序
    random_numbers.sort()
    return random_numbers


class Net(Cell):
    def __init__(self):
        super().__init__()

    @jit
    def construct(self, x, weight, group_list, bias, x_scale, weight_scale):
        out = ms_custom_ops.grouped_matmul_w4(
            x, weight, group_list, bias, x_scale, weight_scale)
        return out


def grouped_matmul_int4_swft(m, k, n, e):
    os.environ['INTERNAL_PRINT_TILING'] = "on"
    group_list_np = np.array(generate_random_numbers(m, e)).astype(np.int32)
    g = 256

    def dyn_quant(x_fp16):
        x_abs = np.abs(x_fp16)
        x_max = np.max(x_abs, axis=-1, keepdims=True)
        anti_scale = x_max.astype(np.float32) / 127.0
        x_int8 = np.round(x_fp16.astype(np.float32) /
                          anti_scale).astype(np.int8)
        return x_int8, anti_scale

    def quant(y_fp16):
        y_fp = y_fp16.reshape((e, k // g, g, n))
        y_max = np.max(np.abs(y_fp), keepdims=True, axis=-2)
        scale = (y_max.astype(np.float32) / 7.0)
        y_int8 = np.round(y_fp.astype(np.float32) /
                          scale).astype(np.int8).reshape((e, k, n))
        return y_int8, scale

    np_x_fp = np.random.uniform(-0.3, 0.3, [m, k]).astype(np.float16)
    np_x_all, np_x_scale = dyn_quant(np_x_fp)
    np_w_fp = np.random.uniform(-0.3, 0.3, [e, k, n]).astype(np.float16)
    np_w_all, np_y_scale = quant(np_w_fp)
    bias = np.ones([e, 1, k]).astype(np.float16) * 8
    np_w_fp16 = np_w_all.reshape(e, k//g, g, n).astype(np.float32) * np_y_scale
    np_w_fp16 = np_w_fp16.reshape(e, k, n)
    np_bias = np.matmul(bias, np_w_fp16).astype(np.float32)

    np_x = split_x(np_x_all, group_list_np)
    np_w = split_w(np_w_fp16)
    np_p = split_x(np_x_scale, group_list_np)
    res_np = [(np.matmul(x0.astype(np.float16), w0) * p0)
              for x0, w0, p0 in zip(np_x, np_w, np_p)]
    expect_np = np.concatenate(res_np, axis=0)

    def i8toi4(y_int8):
        input_x = ((y_int8 + 16) % 16).astype(np.uint8).reshape(-1)
        input_y = (input_x[1::2] << 4) | input_x[::2]
        return input_y
    np_w_all_int4 = i8toi4(np_w_all.transpose(0, 2, 1)).reshape(e, n, k // 2)

    x = Tensor(np_x_all)
    w = Tensor(np_w_all_int4, dtype=ms.qint4x2)
    weight_scale = Tensor(np_y_scale.reshape(e, k // g, n))
    bias_tensor = Tensor(np_bias.reshape(e, n))
    x_scale = Tensor(np_x_scale.reshape(m,))

    group_list = Tensor(group_list_np, dtype=ms.int32)

    # weight must be NZ format so do transdata before
    w_i8 = ms_custom_ops.type_cast(w, ms.int8)
    w_i8_nz = ms_custom_ops.trans_data(w_i8, transdata_type=1)
    w_i4_nz = ms_custom_ops.type_cast(w_i8_nz, ms.qint4x2)
    weight = ms.Parameter(w_i4_nz, requires_grad=False)

    gmm_net = Net()
    output = gmm_net(x, weight, group_list, bias_tensor, x_scale, weight_scale)
    res = custom_compare(output.astype(
        ms.float32).asnumpy(), expect_np, ms.float16)
    assert res, "matmul compare fail."


@pytest.mark.level0
@pytest.mark.platform_ascend310p
@pytest.mark.env_onecard
@pytest.mark.parametrize('batch_size', [5, 17])
@pytest.mark.parametrize('inputs_shape', [[256, 7168, 256]])
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_gmm_w4_swft_0(batch_size, inputs_shape, exec_mode):
    """
    Feature: test grouped matmul w4 with scale
    Description: test grouped matmul w4 with scale
    Expectation: the result is correct
    """
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    grouped_matmul_int4_swft(batch_size, *inputs_shape)


@pytest.mark.level1
@pytest.mark.platform_ascend310p
@pytest.mark.env_onecard
@pytest.mark.parametrize('batch_size', [8, 32])
@pytest.mark.parametrize('inputs_shape', [[7168, 512, 256], [512, 7168, 256], [7168, 1024, 256]])
def test_gmm_w4_swft_1(batch_size, inputs_shape):
    """
    Feature: test matmul operator in graph mode
    Description: test matmul.
    Expectation: the result is correct
    """
    ms.set_device("Ascend")
    ms.set_context(mode=context.GRAPH_MODE)
    grouped_matmul_int4_swft(batch_size, *inputs_shape)

