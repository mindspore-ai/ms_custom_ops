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
"""
Unit test module for grouped_matmul_v4 operator

This module contains tests for various functions and scenarios of the grouped_matmul_v4 operator,
including correctness verification under different input dimensions, quantization formats,
group types, and other conditions.
"""
import numpy as np
import pytest
import mindspore as ms
from mindspore import dtype as mstype
from mindspore import context, Tensor
from mindspore.nn import Cell
from tests.mark_utils import arg_mark
import ms_custom_ops

def split_x(x, group_list):
    """
    Split input tensor according to group list
    
    Args:
        x: Input tensor
        group_list: Group index list
        
    Returns:
        list: List of split tensors
    """
    x_split = []
    for i, end_idx in enumerate(group_list):
        if i == 0:
            x_split.append(x[0: end_idx,])
        else:
            x_split.append(x[group_list[i - 1]: end_idx,])
    return x_split


def split_w(w):
    """
    Split weight tensor along the first dimension
    
    Args:
        w: Weight tensor to split
        
    Returns:
        list: List of split weight tensors
    """
    tmp_split = np.split(w, w.shape[0], axis=0)
    w_split = []
    for t in tmp_split:
        w_split.append(np.squeeze(t, 0))
    return w_split


class GroupedMatmulV4Net(Cell):
    """
    Network wrapper class for grouped_matmul_v4 operator
    
    Used to encapsulate the grouped_matmul_v4_cops operator in tests, providing a unified interface for computation.
    """
    def __init__(self):
        super().__init__()
        self.gmm_v4 = ms_custom_ops.grouped_matmul_v4_cops

    def construct(self, x, weight, bias=None, scale=None, offset=None, antiquant_scale=None,
                  antiquant_offset=None, pertoken_scale=None, group_list=None, split_item=3,
                  group_type=-1, group_list_type=0, weight_format="ND", output_dtype=None):
        out = self.gmm_v4(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset,
                          pertoken_scale, group_list, split_item=split_item, group_type=group_type,
                          group_list_type=group_list_type, weight_format=weight_format, output_dtype=output_dtype)
        return out


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_grouped_matmul_v4_x2d_w2d_splititem0_grouptypeneg1_none(mode):
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_context(device_target="Ascend")
    if mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O0')
    elif mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    gmm_v4_net = GroupedMatmulV4Net()

    split_item = 0
    group_type = -1

    m0 = 16
    k0 = 256
    n0 = 128

    m1 = 127
    k1 = 88
    n1 = 64

    # numpy calculate
    np_x0 = np.random.uniform(1, 2, size=[2, 3, 4, 5, m0, k0]).astype(np.float32)
    np_w0 = np.random.uniform(1, 2, size=[k0, n0]).astype(np.float32)
    np_b0 = np.random.uniform(1, 5, size=[n0]).astype(np.float32)

    np_x1 = np.random.uniform(1, 2, size=[2, 3, 4, 5, m1, k1]).astype(np.float32)
    np_w1 = np.random.uniform(1, 2, size=[k1, n1]).astype(np.float32)
    np_b1 = np.random.uniform(1, 5, size=[n1]).astype(np.float32)

    except0 = np.matmul(np_x0, np_w0) + np_b0
    except1 = np.matmul(np_x1, np_w1) + np_b1

    # ms calculate
    x = [ms.Tensor(np_x0, dtype=mstype.bfloat16), ms.Tensor(np_x1, dtype=mstype.bfloat16)]
    w = [ms.Tensor(np_w0, dtype=mstype.bfloat16), ms.Tensor(np_w1, dtype=mstype.bfloat16)]
    b = [ms.Tensor(np_b0), ms.Tensor(np_b1)]

    res = gmm_v4_net(x, w, b, split_item=split_item, group_type=group_type)

    # compare
    np.testing.assert_allclose(except0, res[0].float().asnumpy(), rtol=4e-3)
    np.testing.assert_allclose(except1, res[1].float().asnumpy(), rtol=4e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_grouped_matmul_v4_x2d_w3d_splititem3_grouptype0_a16w8(mode):
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_context(device_target="Ascend")
    if mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O0')
    elif mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    gmm_v4_net = GroupedMatmulV4Net()

    split_item = 3
    group_type = 0
    group_list_type = 0

    m0 = 32
    k0 = 256
    n0 = 128
    e0 = 8
    group_list_np = [1, 3, 10, 14, 18, 22, 24, 30]  # last value can be less than total token numbers.

    # numpy calculate
    np_x_all = np.random.uniform(-128, 127, size=[m0, k0]).astype(np.float16)
    np_w_all = np.random.uniform(-128, 127, size=[e0, k0, n0]).astype(np.int8)
    antiquant_scale0 = np.array(np.full([e0, n0], 0.01)).astype(np.float16)
    antiquant_offset0 = np.array(np.full([e0, n0], 1)).astype(np.float16)

    np_x = split_x(np_x_all, group_list_np)
    np_w = split_w(np_w_all)
    np_s = split_w(antiquant_scale0)
    np_o = split_w(antiquant_offset0)
    res_np = [np.matmul(x0, (w0 + o0) * s0) for x0, w0, s0, o0 in zip(np_x, np_w, np_s, np_o)]
    except_np = np.concatenate(res_np, axis=0)

    # ms calculate
    x = [ms.Tensor(np_x_all)]
    w = [ms.Tensor(np_w_all)]
    antiquant_scale = [ms.Tensor(antiquant_scale0)]
    antiquant_offset = [ms.Tensor(antiquant_offset0)]

    b = None
    scale = None
    offset = None
    pertoken_scale = None
    group_list = ms.Tensor(group_list_np, dtype=mstype.int64)

    res = gmm_v4_net(x, w, b, scale, offset, antiquant_scale, antiquant_offset, pertoken_scale, group_list,
                     split_item, group_type, group_list_type)

    # compare
    np.testing.assert_allclose(except_np, res[0][:30].asnumpy(), rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_grouped_matmul_v4_x2d_w3d_splititem3_grouptype0_a16w4(mode):
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_context(device_target="Ascend")
    if mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O0')
    elif mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    gmm_v4_net = GroupedMatmulV4Net()

    split_item = 3
    group_type = 0
    group_list_type = 0

    m0 = 32
    k0 = 256
    n0 = 128
    e0 = 8
    group_list_np = [1, 3, 10, 14, 18, 22, 24, 30] # last value can be less than total token numbers

    # numpy calculate
    np_x_all = np.random.uniform(-128, 127, size=[m0, k0]).astype(np.float16)
    np_w_all = np.random.uniform(0, 2, size=[e0, k0, n0]).astype(np.int8)
    antiquant_scale0 = np.array(np.full([e0, n0], 0.01)).astype(np.float16)
    antiquant_offset0 = np.array(np.full([e0, n0], 1)).astype(np.float16)

    for i in range(e0):
        for j in range(k0):
            for k in range(n0):
                np_w_all[i, j, k] = np_w_all[i, j, k] & 0xf

    np_w_all_int4 = np.ones((e0 * k0 * n0 // 2,), dtype=np.int8)
    np_w_all_one_rank = np_w_all.reshape(-1,)
    for i in range(e0 * k0 * n0 // 2):
        np_w_all_int4[i] = np_w_all_one_rank[i * 2] | ((np_w_all_one_rank[(i * 2) + 1] & 15) << 4)

    np_w_all_int4_3_rank = np_w_all_int4.reshape((e0, k0, n0 // 2))

    np_x = split_x(np_x_all, group_list_np)
    np_w = split_w(np_w_all)
    np_s = split_w(antiquant_scale0)
    np_o = split_w(antiquant_offset0)
    res_np = [np.matmul(x0, (w0 + o0) * s0) for x0, w0, s0, o0 in zip(np_x, np_w, np_s, np_o)]
    expect_np = np.concatenate(res_np, axis=0)

    # ms calculate
    x = [ms.Tensor(np_x_all)]
    w = [ms.Tensor(np_w_all_int4_3_rank, dtype=ms.qint4x2)]
    antiquant_scale = [ms.Tensor(antiquant_scale0)]
    antiquant_offset = [ms.Tensor(antiquant_offset0)]

    b = None
    scale = None
    offset = None
    pertoken_scale = None
    group_list = ms.Tensor(group_list_np, dtype=mstype.int64)

    res = gmm_v4_net(x, w, b, scale, offset, antiquant_scale, antiquant_offset, pertoken_scale, group_list,
                     split_item, group_type, group_list_type)

    # compare
    np.testing.assert_allclose(expect_np, res[0][:30].asnumpy(), rtol=1e-3, atol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_grouped_matmul_v4_x2d_w3d_splititem3_grouptype0_none_pertoken(mode):
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_context(device_target="Ascend")
    if mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O0')
    elif mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    gmm_v4_net = GroupedMatmulV4Net()

    split_item = 3
    group_type = 0
    group_list_type = 1

    m0 = 32
    k0 = 256
    n0 = 128
    e0 = 8
    group_list_np = [1, 2, 7, 4, 4, 4, 2, 8]

    # numpy calculate
    np_x_all = np.random.uniform(-128, 127, size=[m0, k0]).astype(np.int8)
    np_w_all = np.random.uniform(-128, 127, size=[e0, k0, n0]).astype(np.int8)
    np_s_all = np.array(np.full([e0, n0], 10)).astype(np.float32)
    np_pts = np.array([10] * m0).astype(np.float32)

    np_x = split_x(np_x_all, np.cumsum(group_list_np))
    np_w = split_w(np_w_all)
    np_s = split_w(np_s_all)
    res_np = [np.matmul(x0, w0 * s0) for x0, w0, s0 in zip(np_x, np_w, np_s)]
    except_np = np.concatenate(res_np, axis=0) * np_pts.reshape(m0, 1)

    # ms calculate
    x = [ms.Tensor(np_x_all)]
    w = [ms.Tensor(np_w_all)]
    scale = [ms.Tensor(np_s_all, dtype=mstype.bfloat16)]
    pertoken_scale = [ms.Tensor(np_pts)]

    b = None
    offset = None
    antiquant_scale = None
    antiquant_offset = None
    group_list = ms.Tensor(group_list_np, dtype=mstype.int64)

    res = gmm_v4_net(x, w, b, scale, offset, antiquant_scale, antiquant_offset, pertoken_scale, group_list,
                     split_item, group_type, group_list_type)

    # compare
    np.testing.assert_allclose(except_np, res[0].float().asnumpy(), rtol=4e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_grouped_matmul_v4_x2d_w3d_splititem3_grouptype0_none_perchannel(mode):
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_context(device_target="Ascend")
    if mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O0')
    elif mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    gmm_v4_net = GroupedMatmulV4Net()

    split_item = 3
    group_type = 0
    group_list_type = 1

    m0 = 32
    k0 = 256
    n0 = 128
    e0 = 8
    group_list_np = [1, 2, 7, 4, 4, 4, 2, 8]

    # numpy calculate
    np_x_all = np.random.uniform(-128, 127, size=[m0, k0]).astype(np.int8)
    np_w_all = np.random.uniform(-128, 127, size=[e0, k0, n0]).astype(np.int8)
    np_s_all = np.array(np.full([e0, n0], 10)).astype(np.float32)
    np_b_all = np.array(np.full([e0, n0], 1)).astype(np.float32)

    np_x = split_x(np_x_all, np.cumsum(group_list_np))
    np_w = split_w(np_w_all)
    np_s = split_w(np_s_all)
    np_b = split_w(np_b_all)
    res_np = [np.matmul(x0, w0 * s0) + b0 * s0 for x0, w0, s0, b0 in zip(np_x, np_w, np_s, np_b)]
    except_np = np.concatenate(res_np, axis=0)

    # ms calculate
    x = [ms.Tensor(np_x_all)]
    w = [ms.Tensor(np_w_all)]
    scale = [ms.Tensor(np_s_all, dtype=mstype.bfloat16)]
    bias = [ms.Tensor(np_b, dtype=mstype.int32)]

    offset = None
    antiquant_scale = None
    antiquant_offset = None
    group_list = ms.Tensor(group_list_np, dtype=mstype.int64)

    res = gmm_v4_net(x, w, bias, scale, offset, antiquant_scale, antiquant_offset, None, group_list,
                     split_item, group_type, group_list_type)

    # compare
    np.testing.assert_allclose(except_np, res[0].float().asnumpy(), rtol=4e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_ops_grouped_matmul_v4_multi_dyn(mode):
    """
    Feature: Pyboost function.
    Description: Test GroupedMatmulV4 forward with dynamic rank/shape.
    Expectation: Success.
    """
    context.set_context(device_target="Ascend")
    if mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O0')
    elif mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    gmm_v4_net = GroupedMatmulV4Net()

    split_item = 0
    group_type = -1
    group_list_type = 0
    weight_format = "ND"

    x = ms.mutable([Tensor(shape=(None, None), dtype=mstype.float16), Tensor(shape=(None, None), dtype=mstype.float16)])
    weight = ms.mutable([Tensor(shape=(None, None), dtype=mstype.float16),
                         Tensor(shape=(None, None), dtype=mstype.float16)])
    gmm_v4_net.set_inputs(x, weight, None, None, None, None, None, None, None, split_item,
                          group_type, group_list_type, weight_format, None)

    np_x0 = np.random.uniform(0.1, 2, size=[16, 256]).astype(np.float32)
    np_w0 = np.random.uniform(0.1, 1, size=[256, 128]).astype(np.float32)
    expect0 = np.matmul(np_x0, np_w0)

    np_x1 = np.random.uniform(0.1, 2, size=[127, 88]).astype(np.float32)
    np_w1 = np.random.uniform(0.1, 1, size=[88, 64]).astype(np.float32)
    expect1 = np.matmul(np_x1, np_w1)

    x1 = ms.mutable([ms.Tensor(np_x0, dtype=mstype.float16), ms.Tensor(np_x1, dtype=mstype.float16)])
    weight1 = ms.mutable([ms.Tensor(np_w0, dtype=mstype.float16), ms.Tensor(np_w1, dtype=mstype.float16)])
    res1 = gmm_v4_net(x1, weight1, split_item=split_item, group_type=group_type)
    np.testing.assert_allclose(expect0, res1[0].asnumpy(), rtol=1e-1)
    np.testing.assert_allclose(expect1, res1[1].asnumpy(), rtol=1e-1)

    x2 = ms.mutable([ms.Tensor(np_x0, dtype=mstype.float16), ms.Tensor(np_x1, dtype=mstype.float16)])
    weight2 = ms.mutable([ms.Tensor(np_w0, dtype=mstype.float16), ms.Tensor(np_w1, dtype=mstype.float16)])
    res2 = gmm_v4_net(x2, weight2, split_item=split_item, group_type=group_type)
    np.testing.assert_allclose(expect0, res2[0].asnumpy(), rtol=1e-1)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
@pytest.mark.parametrize('output_dtype', [ms.float16, ms.bfloat16])
def test_ops_grouped_mamtul_v4_a8w4(mode, output_dtype):
    """
    Feature: pyboost function.
    Description: test GroupedMatmulV4 forward with a8w4.
    Expectation: success.
    """
    np.random.seed(1)
    context.set_context(device_target="Ascend")
    if mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O0')
    elif mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    gmm_v4_net = GroupedMatmulV4Net()

    e = 8
    m = 32
    k = 256
    n = 128
    split_item = 3
    group_type = 0
    group_list_type = 1

    x_np = np.random.randint(-5, 5, size=(m, k)).astype(np.int8)
    w_np = np.random.randint(-5, 5, size=(e, k, n)).astype(np.int8)
    w_int4_np = w_np.reshape(-1) & 0x000F
    w_int4_np = w_int4_np[0::2] | (w_int4_np[1::2] << 4)
    w_int4_np = w_int4_np.reshape(e, k, n // 2)
    scale_np = np.random.normal(0, 0.01, size=(e, 1, n)).astype(np.float32)
    scale_np_uint64 = np.frombuffer(scale_np.tobytes(), dtype=np.uint32).astype(np.uint64).reshape(e, 1, n)
    bias_np = 8 * (w_np.astype(np.float32) * scale_np).sum(axis=1)
    pertoken_scale_np = np.random.normal(0, 0.01, (m, 1)).astype(np.float32)
    group_list_np = np.array([1, 2, 7, 4, 4, 4, 2, 8], dtype=np.int64)

    index = np.cumsum(group_list_np)
    x_np_split = np.split(x_np, index, axis=0)
    pertoken_scale_np_split = np.split(pertoken_scale_np, index, axis=0)
    out_list = []
    scale_fp32 = scale_np_uint64.astype(np.uint32)
    scale_fp32.dtype = np.float32
    for i in range(e):
        mm = np.matmul(x_np_split[i].astype(np.int32), w_np[i].astype(np.int32)).astype(np.float32)
        mm = mm * scale_fp32[i] * pertoken_scale_np_split[i]
        out_list.append(mm)
    expect = np.concatenate(out_list, axis=0)

    x = [Tensor(x_np, ms.int8)]
    weight = [Tensor(w_int4_np, dtype=ms.qint4x2)]
    bias = [Tensor(bias_np, ms.float32)]
    scale = [Tensor(scale_np_uint64, ms.uint64)]
    pertoken_scale = [Tensor(pertoken_scale_np, ms.float32)]
    group_list = Tensor(group_list_np, ms.int64)
    out = gmm_v4_net(x, weight, bias=bias, scale=scale, pertoken_scale=pertoken_scale,
                     group_list=group_list, split_item=split_item, group_type=group_type,
                     group_list_type=group_list_type, output_dtype=output_dtype)[0]
    cnt = expect.shape[0]
    np.testing.assert_allclose(expect.astype(np.float32), out[:cnt].astype(ms.float32).asnumpy(), rtol=5e-3, atol=5e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
@pytest.mark.parametrize('weight_format', ["ND", "FRACTAL_NZ"])
def test_ops_grouped_mamtul_v4_fractal_nz(mode, weight_format):
    """
    Feature: pyboost function.
    Description: test GroupedMatmulV4 forward with fractal_nz.
    Expectation: success.
    """
    np.random.seed(1)
    context.set_context(device_target="Ascend")
    if mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O0')
    elif mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    gmm_v4_net = GroupedMatmulV4Net()

    split_item = 0
    group_type = -1

    np_x0 = np.random.uniform(0.1, 2, size=[16, 256]).astype(np.float16)
    np_w0 = np.random.uniform(0.1, 1, size=[256, 128]).astype(np.float16)
    expect0 = np.matmul(np_x0, np_w0)

    ms_x0 = [ms.Tensor(np_x0, dtype=mstype.float16)]
    if weight_format == "FRACTAL_NZ":
        ms_w0 = [ms_custom_ops.trans_data(ms.Tensor(np_w0, dtype=mstype.float16), transdata_type=1)]
    else:
        ms_w0 = [ms.Tensor(np_w0, dtype=mstype.float16)]

    res = gmm_v4_net(ms_x0, ms_w0, split_item=split_item, group_type=group_type, weight_format=weight_format)
    np.testing.assert_allclose(expect0, res[0].asnumpy(), rtol=1e-1)
