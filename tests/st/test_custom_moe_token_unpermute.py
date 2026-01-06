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
""" tests_custom_moe_token_unpermute_pyboost_ascend """

import numpy as np
import pytest
import logging
from mindspore import nn
from mindspore import Tensor, context

# Local imports
import ms_custom_ops

np.set_printoptions(precision=2, suppress=True, linewidth=200)


class MoeTokenUnpermuteNet(nn.Cell):
    """MoeTokenUnpermuteNet"""
    def construct(self, permuted_tokens, sorted_indices, probs, padded_mode=False, restore_shape=None):
        return ms_custom_ops.moe_token_unpermute(
            permuted_tokens, sorted_indices, probs, padded_mode, restore_shape
        )


def moe_token_unpermute_op_impl(permute_token, sorted_idx, probs):
    token_num = probs.shape[0]
    top_k = probs.shape[1]
    hidden = permute_token.shape[1]
    out = np.zeros((token_num, hidden), dtype=np.float16)

    for i in range(token_num):
        for k in range(top_k):
            dst_row = permute_token[sorted_idx[i * top_k + k], :]
            out[i, :] += probs[i, k] * dst_row
    return out


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("token_num", [1, 16, 1001])
@pytest.mark.parametrize("hidden", [7168, 2048])
@pytest.mark.parametrize("top_k", [8])
def test_moe_token_unpermute(exec_mode, token_num, hidden, top_k):
    """
    Feature: test moe_token_unpermute operator
    Description: test different mode and input dimension of the operator correctness
    Expectation: the result is correct
    """
    context.set_context(mode=exec_mode, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0"})
    net = MoeTokenUnpermuteNet()

    # 生成输入数据
    permute_token = np.random.randn(token_num * top_k, hidden).astype(np.float16)
    sorted_idx = np.arange(token_num * top_k, dtype=np.int32)
    np.random.shuffle(sorted_idx)
    probs = np.random.randn(token_num, top_k).astype(np.float16)

    # 计算期望输出
    expected = moe_token_unpermute_op_impl(permute_token, sorted_idx, probs)

    # 运行算子
    output = net(Tensor(permute_token), Tensor(sorted_idx), Tensor(probs))

    # 验证结果
    assert np.allclose(output.asnumpy(), expected, rtol=1e-3, atol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("token_num", [16])
@pytest.mark.parametrize("hidden", [1024])
@pytest.mark.parametrize("top_k", [8])
def test_moe_token_unpermute_unsupported_hidden_size(exec_mode, token_num, hidden, top_k):
    """
    Feature: test moe_token_unpermute operator
    Description: test unsupported hidden_size
    Expectation: Unsupported hidden size correctly rejected
    """
    context.set_context(mode=exec_mode, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0"})
    net = MoeTokenUnpermuteNet()

    # 生成输入数据
    permute_token = np.random.randn(token_num * top_k, hidden).astype(np.float16)
    sorted_idx = np.arange(token_num * top_k, dtype=np.int32)
    np.random.shuffle(sorted_idx)
    probs = np.random.randn(token_num, top_k).astype(np.float16)

    # 运行算子
    with pytest.raises(RuntimeError, match="Tiling error"):
        output = net(Tensor(permute_token), Tensor(sorted_idx), Tensor(probs))
        out = output.asnumpy()
    logging.info(
        f"Unsupported hidden_size correctly rejected: hidden_size={hidden}",
    )


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("token_num", [16])
@pytest.mark.parametrize("hidden", [7168])
@pytest.mark.parametrize("top_k", [7])
def test_moe_token_unpermute_unsupported_top_k(exec_mode, token_num, hidden, top_k):
    """
    Feature: test moe_token_unpermute operator
    Description: test unsupported top_k
    Expectation: Unsupported top_k correctly rejected
    """
    context.set_context(mode=exec_mode, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0"})
    net = MoeTokenUnpermuteNet()

    # 生成输入数据
    permute_token = np.random.randn(token_num * top_k, hidden).astype(np.float16)
    sorted_idx = np.arange(token_num * top_k, dtype=np.int32)
    np.random.shuffle(sorted_idx)
    probs = np.random.randn(token_num, top_k).astype(np.float16)

    # 运行算子
    with pytest.raises(RuntimeError, match="Tiling error"):
        output = net(Tensor(permute_token), Tensor(sorted_idx), Tensor(probs))
        out = output.asnumpy()
    logging.info(
        f"Unsupported top_k correctly rejected: top_k={top_k}",
    )


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("token_num", [16])
@pytest.mark.parametrize("hidden", [7168])
@pytest.mark.parametrize("top_k", [8])
def test_moe_token_unpermute_1d_input(exec_mode, token_num, hidden, top_k):
    """
    Feature: test moe_token_unpermute operator
    Description: test 1d input
    Expectation: 1d input correctly rejected
    """
    context.set_context(mode=exec_mode, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0"})
    net = MoeTokenUnpermuteNet()

    # 生成输入数据
    permute_token = np.random.randn(hidden).astype(np.float16)
    sorted_idx = np.arange(token_num * top_k, dtype=np.int32)
    np.random.shuffle(sorted_idx)
    probs = np.random.randn(token_num, top_k).astype(np.float16)

    # 运行算子
    with pytest.raises(RuntimeError, match="must be a 2D tensor"):
        output = net(Tensor(permute_token), Tensor(sorted_idx), Tensor(probs))
        out = output.asnumpy()
    logging.info(
        f"1d input correctly rejected",
    )
