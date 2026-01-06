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
"""Test cases for fused_add_topk_div_moe custom operator."""
import logging
from functools import wraps

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, nn, context, Profiler
from mindspore.common.api import jit
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics

import ms_custom_ops

k = 8


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


def softmax(logits):
    exp_logits = np.exp(logits)
    sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
    softmax_output = exp_logits / sum_exp
    return softmax_output


def golden_fused_add_topk_div_moe(logits, bias):
    """Compute golden reference output for fused_add_topk_div_moe."""
    sigmoid_out = softmax(logits)
    add_out = sigmoid_out + bias
    idx = np.argpartition(add_out, -k, -1)
    topk_idx = np.take(idx, range(-k, 0), -1)
    topk_scores = np.take_along_axis(add_out, topk_idx, -1)
    sorted_indices = np.argsort(-topk_scores, axis=-1)
    sorted_topk_scores = np.take_along_axis(topk_scores, sorted_indices, -1)
    sorted_topk_idx = np.take_along_axis(topk_idx, sorted_indices, -1)
    sum_out = np.sum(sorted_topk_scores, axis=-1, keepdims=True)
    div_out = sorted_topk_scores / sum_out
    sorted_topk_idx = sorted_topk_idx.astype(np.int32)
    return div_out, sorted_topk_idx


class FusedAddTopkDivMoeNet(nn.Cell):
    """Fused add topk div moe operation network"""

    @jit_for_graph_mode
    def construct(self, logits, bias):
        return ms_custom_ops.fused_add_topk_div_moe(logits, bias, 1, 1, 8,
                                                    8, 0, True, 1.0)


def run_fused_add_topk_div_moe(net, logits_dtype, bias_dtype, num_tokens, is_profiler=False):
    """Run fused_add_topk_div_moe test with given parameters."""
    np_logits = np.random.random((num_tokens, 128)).astype(logits_dtype)
    np_bias = np.zeros(128).astype(bias_dtype)
    logits = Tensor(np_logits)
    bias = Tensor(np_bias)
    np_expert_weight, _ = golden_fused_add_topk_div_moe(
        np_logits, np_bias)
    if not is_profiler:
        expert_weight, _ = net(logits, bias)
        np.testing.assert_allclose(np_expert_weight, expert_weight.asnumpy(),
                                   rtol=1e-4, atol=1e-4, err_msg=" fused_add_topk_div_moe ")
    else:
        profiler = Profiler(profiler_level=ProfilerLevel.Level2,
                            activities=[ProfilerActivity.CPU,
                                        ProfilerActivity.NPU],
                            aic_metrics=AicoreMetrics.AiCoreNone)
        for _ in range(50):
            _, _ = net(logits, bias)
        profiler.analyse()


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("logits_dtype", [np.float32])
@pytest.mark.parametrize("bias_dtype", [np.float32])
@pytest.mark.parametrize("num_tokens", [2048])
def test_fused_add_topk_div(exec_mode, logits_dtype, bias_dtype, num_tokens):
    """
    Feature:aclnnFusedAddTopkDivMoe kernel.
    Description: test for FusedAddTopkDivMoeExt ops.
    Expectation:should pass for all testcases.
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0"})

    net = FusedAddTopkDivMoeNet()
    run_fused_add_topk_div_moe(
        net, logits_dtype, bias_dtype, num_tokens)
    logging.info("Correctly handle standard scenarios")


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("logits_dtype", [np.float32])
@pytest.mark.parametrize("bias_dtype", [np.float32])
@pytest.mark.parametrize("num_tokens", [2047])
def test_fused_add_topk_div_misalignment(exec_mode, logits_dtype, bias_dtype, num_tokens):
    """
    Feature:aclnnFusedAddTopkDivMoe kernel.
    Description: test for FusedAddTopkDivMoeExt ops.
    Expectation:should pass for all testcases.
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0"})

    net = FusedAddTopkDivMoeNet()
    run_fused_add_topk_div_moe(
        net, logits_dtype, bias_dtype, num_tokens)
    logging.info("Correctly handle misalignment scenarios")


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("logits_dtype", [np.float32])
@pytest.mark.parametrize("bias_dtype", [np.float32])
@pytest.mark.parametrize("num_tokens", [5])
def test_fused_add_topk_div_nums_token_less_8(exec_mode, logits_dtype, bias_dtype, num_tokens):
    """
    Feature:aclnnFusedAddTopkDivMoe kernel.
    Description: test for FusedAddTopkDivMoeExt ops.
    Expectation:should pass for all testcases.
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0"})

    net = FusedAddTopkDivMoeNet()
    run_fused_add_topk_div_moe(
        net, logits_dtype, bias_dtype, num_tokens)
    logging.info("Correctly handle scenarios where num_tokens < 8")


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("logits_dtype", [np.float32])
@pytest.mark.parametrize("bias_dtype", [np.int32])
@pytest.mark.parametrize("num_tokens", [2048])
def test_fused_add_topk_div_invalid_data(exec_mode, logits_dtype, bias_dtype, num_tokens):
    """
    Feature:aclnnFusedAddTopkDivMoe kernel.
    Description: test for FusedAddTopkDivMoeExt ops.
    Expectation:should pass for all testcases.
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0"})

    net = FusedAddTopkDivMoeNet()
    with pytest.raises(RuntimeError, match="the dtype of 'bias' should be Float32"):
        run_fused_add_topk_div_moe(
            net, logits_dtype, bias_dtype, num_tokens)
    logging.info("Correctly handle invalid data scenarios")


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("logits_dtype", [np.float32])
@pytest.mark.parametrize("bias_dtype", [np.float32])
@pytest.mark.parametrize("num_tokens", [1])
def test_fused_add_topk_div_nums_token_equal_1(exec_mode, logits_dtype, bias_dtype, num_tokens):
    """
    Feature:aclnnFusedAddTopkDivMoe kernel.
    Description: test for FusedAddTopkDivMoeExt ops.
    Expectation:should pass for all testcases.
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0"})

    net = FusedAddTopkDivMoeNet()
    run_fused_add_topk_div_moe(
        net, logits_dtype, bias_dtype, num_tokens)
    logging.info("Correctly handle scenarios where num_tokens == 1")
