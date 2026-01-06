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
"""Tests for ms_custom_ops.rope using numpy golden reference."""
from functools import wraps

import mindspore as ms
import numpy as np
import pytest
from mindspore import context, nn
from mindspore.common.api import jit

import ms_custom_ops


def rotate_half(x):
    """
    rotate half
    """
    x0, x1 = x.chunk(2, -1)
    return ms.ops.cat((-x1, x0), axis=-1)


def golden_rotary_coeff_2(batch, seqlen, query, key, cos, sin):
    """
    golden calculate
    """
    origin_type = query.dtype
    cal_type = cos.dtype
    if origin_type is ms.bfloat16:
        cal_type = ms.float32
    ntoken = batch * seqlen
    batch = ntoken // seqlen
    _, hidden_size = query.shape
    _, hidden_size1 = key.shape
    _, head_size = cos.shape
    head_num = hidden_size // head_size
    head_num1 = hidden_size1 // head_size
    q = query.view(batch, seqlen, head_num, head_size).astype(cal_type)
    k = key.view(batch, seqlen, head_num1, head_size).astype(cal_type)
    cos = cos.view(batch, seqlen, head_size).unsqueeze(2).astype(cal_type)
    sin = sin.view(batch, seqlen, head_size).unsqueeze(2).astype(cal_type)
    q_embed = ((q * cos) + (rotate_half(q) * sin)).view(ntoken, hidden_size)
    k_embed = ((k * cos) + (rotate_half(k) * sin)).view(ntoken, hidden_size1)
    return q_embed.astype(origin_type), k_embed.astype(origin_type)


def jit_for_graph_mode(fn):
    """
    A decorator that conditionally applies jit to a function at runtime based on the context mode.
    """
    jitted_fn = jit(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if context.get_context("mode") == context.GRAPH_MODE:
            return jitted_fn(*args, **kwargs)
        return fn(*args, **kwargs)
    return wrapper


class AsdOpRopeNet(nn.Cell):
    """rope"""
    @jit_for_graph_mode
    def construct(self, query, key, cos, sin, seq_len, rotary_coeff, cos_format=0):
        return ms_custom_ops.rope(query, key, cos, sin, seq_len, rotary_coeff, cos_format)


def run_case(query_dtype, cos_dtype, rotary_coeff, batch, seqlen, head_num_q, head_num_k, head_size):
    """run case"""
    tokens = batch * seqlen
    np_query = np.random.rand(tokens, head_num_q * head_size)
    np_key = np.random.rand(tokens, head_num_k * head_size)
    np_cos = np.random.rand(tokens, head_size)
    np_sin = np.random.rand(tokens, head_size)
    # np_sql = np.array([seqlen])
    np_sql = np.full((batch), seqlen)
    tensor_query = ms.Tensor(np_query, dtype=query_dtype)
    tensor_key = ms.Tensor(np_key, dtype=query_dtype)
    tensor_cos = ms.Tensor(np_cos, dtype=cos_dtype)
    tensor_sin = ms.Tensor(np_sin, dtype=cos_dtype)
    tensor_seqlen = ms.Tensor(np_sql, dtype=ms.int32)
    cos_format = 0
    net = AsdOpRopeNet()
    q_out, k_out = net(tensor_query, tensor_key, tensor_cos,
                       tensor_sin, tensor_seqlen, rotary_coeff, cos_format)
    golden_q, golden_k = golden_rotary_coeff_2(
        batch, seqlen, tensor_query, tensor_key, tensor_cos, tensor_sin)

    np.testing.assert_allclose(golden_q.astype(ms.float32).asnumpy(), q_out.astype(ms.float32).asnumpy(),
                               rtol=1e-4, atol=1e-4, err_msg=" query ")
    np.testing.assert_allclose(golden_k.astype(ms.float32).asnumpy(), k_out.astype(ms.float32).asnumpy(),
                               rtol=1e-4, atol=1e-4, err_msg=" key ")
    return q_out, k_out


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend310p
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("query_dtype", [ms.float16])
@pytest.mark.parametrize("cos_dtype", [ms.float16, ms.float32])
@pytest.mark.parametrize("rotary_coeff", [2])
@pytest.mark.parametrize("batch", [8, 91])
@pytest.mark.parametrize("seqlen", [128])
@pytest.mark.parametrize("head_num_q", [32])
@pytest.mark.parametrize("head_num_k", [2])
@pytest.mark.parametrize("head_size", [128])
def test_asdop_rope_fp16(exec_mode, query_dtype, cos_dtype, rotary_coeff, batch, seqlen, head_num_q, head_num_k,
                         head_size):
    """
    Feature:  asdop rope kernel.
    Description: test for ApplyRotaryPosEmbExt ops.
    Expectation: should pass for all testcases.
    """
    np.random.seed(0)
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    run_case(query_dtype, cos_dtype, rotary_coeff, batch,
             seqlen, head_num_q, head_num_k, head_size)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("query_dtype", [ms.bfloat16])
@pytest.mark.parametrize("cos_dtype", [ms.bfloat16])
@pytest.mark.parametrize("rotary_coeff", [2])
@pytest.mark.parametrize("batch", [1, 71])
@pytest.mark.parametrize("seqlen", [47, 128])
@pytest.mark.parametrize("head_num_q", [8])
@pytest.mark.parametrize("head_num_k", [1])
@pytest.mark.parametrize("head_size", [64, 128])
def test_asdop_rope_bf16(exec_mode, query_dtype, cos_dtype, rotary_coeff, batch, seqlen, head_num_q, head_num_k,
                         head_size):
    """
    Feature:  asdop rope kernel.
    Description: test for ApplyRotaryPosEmbExt ops.
    Expectation: should pass for all testcases.
    """
    np.random.seed(0)
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    run_case(query_dtype, cos_dtype, rotary_coeff, batch,
             seqlen, head_num_q, head_num_k, head_size)
