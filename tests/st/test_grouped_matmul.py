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
from mindspore import context, Tensor, ops, Profiler
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


def process_deq_scale(deq_scale) -> np.ndarray:
    new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.uint32)
    return new_deq_scale.astype(np.int64)


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


def np_qbmm_compute(a, b, tmp_scale, bias=None, tmp_pertoken_scale=None):
    c = np.dot(a.astype(np.float32), b.astype(np.float32)).astype(np.int32)
    if bias is not None:
        c = c + bias
    c = c.astype(np.float32) * tmp_scale
    if tmp_pertoken_scale is not None:
        per_token_scale = tmp_pertoken_scale[:, np.newaxis]
        c = c * per_token_scale
    c = c.astype(np.float16)
    return c


class Net(Cell):
    def __init__(self):
        super().__init__()

    @jit
    def construct(self, x, weight, group_list, bias=None, scale=None, per_token_scale=None, antiquant_scale=None, transpose_a=False, transpose_b=False):
        out = ms_custom_ops.grouped_matmul(
            x, weight, group_list, bias, scale, per_token_scale, antiquant_scale, transpose_a, transpose_b)
        return out


def custom_grouped_matmul(m, k, n, e, trans_a=False, trans_b=False, profiling=False, with_bias=True, with_pertoken_scale=False, scale_dtype=ms.int64):
    os.environ['INTERNAL_PRINT_TILING'] = "on"

    # numpy calculate
    np_x_all = np.random.uniform(-20, 20, size=[m, k]).astype(np.int8)
    np_w_all = np.random.uniform(-20, 20, size=[e, k, n]).astype(np.int8)
    np_b_all = np.random.randint(-10, 10, (e, n)).astype(np.int32)
    np_s_all = np.random.rand(e, n).astype(np.float32) / 1000
    scale = process_deq_scale(np_s_all)
    np_ps_all = np.random.rand(m).astype(np.float32)
    group_list_np = np.array(generate_random_numbers(m, e)).astype(np.int32)

    def compute_numpy_result(np_x_all, np_w_all, np_b_all, np_s_all, np_ps_all, group_list_np, with_bias, with_pertoken_scale):
        # use group_list split x. [(G0, k), (G1, k)....(GN, k)]
        np_x = split_x(np_x_all, group_list_np)
        np_w = split_w(np_w_all)  # [(k, n), (k, n)....(k, n)]
        np_b = split_w(np_b_all)  # [(n), (n)....(n)]
        np_s = split_w(np_s_all)  # [(n), (n)....(n)]
        # use group_list split per_token_scale. [(G0,), (G1,)....(GN,)]
        np_ps = split_x(np_ps_all, group_list_np)

        if not with_bias and not with_pertoken_scale:
            res_np = [np_qbmm_compute(x0, w0, s0)
                      for x0, w0, s0 in zip(np_x, np_w, np_s)]
        elif with_bias and not with_pertoken_scale:
            res_np = [np_qbmm_compute(x0, w0, s0, b0)
                      for x0, w0, s0, b0 in zip(np_x, np_w, np_s, np_b)]
        if not with_bias and with_pertoken_scale:
            res_np = [np_qbmm_compute(x0, w0, s0, None, ps0)
                      for x0, w0, s0, ps0 in zip(np_x, np_w, np_s, np_ps)]
        elif with_bias and with_pertoken_scale:
            res_np = [np_qbmm_compute(x0, w0, s0, b0, ps0) for x0, w0, s0, b0, ps0 in zip(
                np_x, np_w, np_s, np_b, np_ps)]

        expect_np = np.concatenate(res_np, axis=0)
        return expect_np

    expect_np = compute_numpy_result(
        np_x_all, np_w_all, np_b_all, np_s_all, np_ps_all, group_list_np, with_bias, with_pertoken_scale)

    # ms calculate
    if trans_b:
        np_w_all = np.transpose(np_w_all, (0, 2, 1))
    x = Tensor(np_x_all)  # [m, k]
    w = Tensor(np_w_all)  # [e, k, n]

    # weight must be NZ format so do transdata before
    w_nz = ms_custom_ops.trans_data(w, transdata_type=1)
    weight = ms.Parameter(w_nz, requires_grad=False)

    if scale_dtype == ms.float32:
        s = Tensor(np_s_all, ms.float32)  # [e, n]
    elif scale_dtype == ms.int64:
        s = Tensor(scale, ms.int64)  # [e, n]
    else:
        raise ValueError(
            f"scale_dtype must be float32 or int64, but got {scale_dtype}")
    scale = ms.Parameter(s, requires_grad=False) if s is not None else None

    ps = ms.Parameter(Tensor(np_ps_all, ms.float32),
                      requires_grad=False) if with_pertoken_scale else None
    b = ms.Parameter(Tensor(np_b_all, ms.int32),
                     requires_grad=False) if with_bias else None

    group_list = Tensor(group_list_np, dtype=ms.int32)
    gmm_net = Net()

    for _ in range(50 if profiling else 1):
        output = gmm_net(x, weight, group_list, b, scale, ps,
                         transpose_a=trans_a, transpose_b=trans_b)
    if profiling:
        return
    res = custom_compare(output.astype(
        ms.float32).asnumpy(), expect_np, ms.float16)
    assert res, "matmul compare fail."


def generate_random_numbers(m, e):
    # 生成e-1个互不相同的随机数，范围是1到n，但不包括m
    random_numbers = random.choices([i for i in range(1, m+1)], k=e-1)
    # 将m添加到列表的末尾
    random_numbers.append(m)
    # 将列表从小到大排序
    random_numbers.sort()
    return random_numbers


@pytest.mark.level0
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize('m', [32, 1000])
@pytest.mark.parametrize('with_pertoken_scale', [True, False])
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.env_onecard
def test_gmm_v4_with_scale(m, with_pertoken_scale, exec_mode):
    """
    Feature: test matmul operator in graph mode or pynative mode
    Description: test matmul with per_token_scale and without per_token_scale
    Expectation: the result is correct
    """
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    custom_grouped_matmul(m, 64, 128, 8, trans_b=True,
                          with_pertoken_scale=with_pertoken_scale, scale_dtype=ms.float32)


@pytest.mark.level1
@pytest.mark.platform_ascend310p
@pytest.mark.env_onecard
@pytest.mark.parametrize('inputs_shape', [[40, 512, 7168], [1024, 2048, 7168], [1024, 7168, 4096]])
def test_gmm_v4_with_scale_ds(inputs_shape):
    """
    Feature: test grouped matmul with scale
    Description: test grouped matmul with scale
    Expectation: the result is correct
    """
    ms.set_device("Ascend")
    ms.set_context(mode=context.GRAPH_MODE)
    e = 256
    custom_grouped_matmul(*inputs_shape, e, trans_b=True,
                          with_pertoken_scale=True, scale_dtype=ms.float32)


def grouped_quant_matmul(m, k, n, e, trans_a=False, trans_b=False, profiling=False, with_bias=False):
    os.environ['INTERNAL_PRINT_TILING'] = "on"

    # numpy calculate
    np_x_all = np.random.uniform(-20, 20, size=[m, k]).astype(np.int8)
    np_w_all = np.random.uniform(-20, 20, size=[e, k, n]).astype(np.int8)
    np_b_all = np.random.randint(-10, 10, (e, n)).astype(np.int32)
    np_s_all = np.random.rand(e, n).astype(np.float32) / 1000

    scale = process_deq_scale(np_s_all)
    group_list_np = np.array(generate_random_numbers(m, e)).astype(np.int32)

    # use group_list split x. [(G0, n), (G1, n)....(GN, n)]
    np_x = split_x(np_x_all, group_list_np)
    np_w = split_w(np_w_all)  # [(k, n), (k, n)....(k, n)]
    np_b = split_w(np_b_all)  # [(n), (n)....(n)]
    np_s = split_w(np_s_all)  # [(n), (n)....(n)]
    if not with_bias:
        res_np = [np_qbmm_compute(x0, w0, s0)
                  for x0, w0, s0 in zip(np_x, np_w, np_s)]
    else:
        res_np = [np_qbmm_compute(x0, w0, s0, b0)
                  for x0, w0, s0, b0 in zip(np_x, np_w, np_s, np_b)]
    expect_np = np.concatenate(res_np, axis=0)

    # ms calculate
    if trans_b:
        np_w_all = np.transpose(np_w_all, (0, 2, 1))
    x = ms.Tensor(np_x_all)  # [m, k]
    w = ms.Tensor(np_w_all)  # [e, k, n]
    s = ms.Tensor(scale, ms.int64)  # [e, n]

    # [e, n]
    b = ms.Parameter(Tensor(np_b_all, ms.int32),
                     requires_grad=False) if with_bias else None

    group_list = Tensor(group_list_np, dtype=ms.int32)
    w_nz = ms_custom_ops.trans_data(w, transdata_type=1)
    weight = ms.Parameter(w_nz, requires_grad=False)

    gmm_net = Net()

    for _ in range(50 if profiling else 1):
        output = gmm_net(x, weight, group_list, bias=b, scale=s, per_token_scale=None, antiquant_scale=None,
                         transpose_a=trans_a, transpose_b=trans_b)
    if profiling:
        return
    res = custom_compare(output.astype(
        ms.float32).asnumpy(), expect_np, ms.float16)
    assert res, "matmul compare fail."


def grouped_matmul(m, k, n, e, trans_a=False, trans_b=False, profiling=False):
    os.environ['INTERNAL_PRINT_TILING'] = "on"

    # numpy calculate
    np_x_all = np.random.uniform(0.1, 2, size=[m, k]).astype(np.float16)
    np_w_all = np.random.uniform(0.1, 1, size=[e, k, n]).astype(np.float16)
    group_list_np = np.array(generate_random_numbers(m, e)).astype(np.int32)

    # use group_list split x. [(G0, n), (G1, n)....(GN, n)]
    np_x = split_x(np_x_all, group_list_np)
    np_w = split_w(np_w_all)  # [(k, n), (k, n)....(k, n)]
    res_np = [np.matmul(x0, w0) for x0, w0 in zip(np_x, np_w)]
    expect_np = np.concatenate(res_np, axis=0)

    # ms calculate
    if trans_b:
        np_w_all = np.transpose(np_w_all, (0, 2, 1))
    x = ms.Tensor(np_x_all)  # [m, k]
    w = ms.Tensor(np_w_all)  # [e, k, n]

    group_list = ms.Tensor(group_list_np, dtype=ms.int32)
    w_nz = ms_custom_ops.trans_data(w, transdata_type=1)
    weight = ms.Parameter(w_nz, requires_grad=False)
    gmm_net = Net()
    for _ in range(50 if profiling else 1):
        output = gmm_net(x, weight, group_list, bias=None, scale=None, per_token_scale=None, antiquant_scale=None,
                         transpose_a=trans_a, transpose_b=trans_b)
    if profiling:
        return
    res = custom_compare(output.astype(
        ms.float32).asnumpy(), expect_np, ms.float16)
    assert res, "matmul compare fail."


@pytest.mark.level0
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize('input_shape', [[32, 32, 64, 8], [1000, 256, 512, 16]])
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.env_onecard
def test_gmm_increment(input_shape, exec_mode):
    """
    Feature: test matmul operator in graph mode
    Description: test matmul.
    Expectation: the result is correct
    """
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    grouped_matmul(*input_shape, trans_b=True)
    grouped_quant_matmul(*input_shape, trans_b=True)


@pytest.mark.level0
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize('input_shape', [[32, 32, 64, 8], [1000, 256, 512, 16]])
@pytest.mark.env_onecard
def test_gmm_quant_with_bias(input_shape):
    """
    Feature: test matmul operator in graph mode
    Description: test matmul.
    Expectation: the result is correct
    """
    ms.set_device("Ascend")
    ms.set_context(mode=context.GRAPH_MODE)
    grouped_quant_matmul(*input_shape, trans_b=True, with_bias=True)


@pytest.mark.level1
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize('m', [1024])
@pytest.mark.parametrize('e', [8])
@pytest.mark.parametrize("prof_flag_str", [0])
@pytest.mark.env_onecard
def test_moe_real_shape(m, e, prof_flag_str):
    """
    Feature: test matmul operator in graph mode
    Description: test matmul.
    Expectation: the result is correct
    """
    ms.set_device("Ascend")
    ms.set_context(mode=context.GRAPH_MODE)
    prof_flag = bool(int(prof_flag_str))
    profiler = Profiler(start_profile=False, output_path="profiler")
    profiler.start()
    grouped_matmul(m, 5120, 2688, e, profiling=prof_flag, trans_b=True)
    grouped_matmul(m, 5120, 1344, e, profiling=prof_flag, trans_b=True)
    profiler.stop()
    profiler.analyse()
