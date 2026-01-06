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

import numpy as np
import pytest
import logging
import mindspore as ms
from mindspore import context
from mindspore.nn import Cell
from mindspore import Tensor
import ms_custom_ops

class MoeInitRoutingV2Net(Cell):
    """MoeInitRoutingV2."""
    def __init__(self):
        super().__init__()
        self.forward_func = ms_custom_ops.moe_init_routing_v2

    def construct(self, *args):
        return self.forward_func(*args)

def moe_init_routing_v2_exec(x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode,
                             expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag):
    num_rows = x.shape[0]
    hidden_size = x.shape[-1]
    k = expert_idx.shape[-1]
    sorted_row_idx = np.argsort(expert_idx.reshape((-1,)), axis=-1, kind="stable") # 原顺序
    sorted_expert_idx = np.sort(expert_idx.reshape((-1,)), axis=-1)
    if drop_pad_mode == 1 and expert_num <= 0:
        raise Exception("[Error] expert_num must be greater than 0 when drop pad mode is enabled")

    expert_tokens_count_or_cumsum = None
    expert_tokens_before_capacity = None
    # expert_token_idx
    expert_idx_hist, _ = np.histogram(sorted_expert_idx, bins=expert_num, range=(0, expert_num - 1))
    expert_token_idx = np.cumsum(expert_idx_hist)
    # 每个专家用了几次
    # 0 0 0 1 1 1 2 3 3 3
    # export个数 0 3 6 7 10 -> grouplist
    if drop_pad_mode == 1 and expert_tokens_before_capacity_flag:
        expert_tokens_before_capacity = expert_idx_hist.astype("int32")
    if drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag == 1: # 只用这个
        expert_tokens_count_or_cumsum = expert_token_idx.astype("int32")
    elif drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag == 2:
        expert_tokens_count_or_cumsum = expert_idx_hist.astype("int32")

    if drop_pad_mode == 0:
        expanded_row_idx = np.zeros(sorted_row_idx.shape, dtype=np.int32)
        expanded_row_idx[sorted_row_idx] = np.arange(sorted_row_idx.shape[-1], dtype=np.int32) # 取值
        if active_num == 0:
            active_num = num_rows * k
        else:
            active_num = min(active_num, num_rows * k)
        expanded_x = x[sorted_row_idx[:active_num] // k, :]
    else:
        # Handle drop/pad mode
        sort_row_tmp = np.full((expert_num * expert_capacity), -1, dtype=int)
        offset = 0
        last_expert_id = 0
        for i, val in enumerate(sorted_row_idx):
            if val != -1:
                if last_expert_id != sorted_expert_idx[i]:
                    offset = 0
                    last_expert_id = sorted_expert_idx[i]
                sort_row_tmp[sorted_expert_idx[i] * expert_capacity + offset] = sorted_row_idx[i]
                offset = offset + 1

        expanded_row_idx = np.full(sorted_row_idx.shape, -1)
        for i, val in enumerate(sort_row_tmp):
            if val != -1:
                expanded_row_idx[val] = i

        expanded_x = np.full((expert_num * expert_capacity, hidden_size), 0, dtype=x.dtype)
        for i, val in enumerate(sort_row_tmp):
            if val != -1:
                expanded_x[i] = x[val // k]
        expanded_x = expanded_x.reshape(expert_num, expert_capacity, hidden_size)
    if expert_tokens_count_or_cumsum is not None:
        return expanded_x, expanded_row_idx.astype("int32"), expert_tokens_count_or_cumsum
    if expert_tokens_before_capacity is not None:
        return expanded_x, expanded_row_idx.astype("int32"), expert_tokens_before_capacity
    return expanded_x, expanded_row_idx.astype("int32")

def _test_moe_init_routing_v2(exec_mode, num_rows, hidden_size, k, expert_num, expert_capacity,
                              drop_pad_mode, active_num=0, expert_tokens_count_or_cumsum_flag=0,
                              expert_tokens_before_capacity_flag=False, is_dyn=False):
    context.set_context(mode=exec_mode, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0"})

    # Generate input data
    np.random.seed(0)
    x = np.random.uniform(-1, 1, size=(num_rows, hidden_size)).astype(np.float16)
    expert_idx = np.random.randint(0, expert_num, size=(num_rows, k)).astype(np.int32)

    # Calculate expected output
    expected = moe_init_routing_v2_exec(x, expert_idx, active_num, expert_capacity, expert_num,
                                        drop_pad_mode, expert_tokens_count_or_cumsum_flag,
                                        expert_tokens_before_capacity_flag)

    # Run on MindSpore
    net = MoeInitRoutingV2Net()
    if is_dyn and exec_mode == context.GRAPH_MODE:
        x_dyn = Tensor(shape=[None, None], dtype=ms.float16)
        expert_idx_dyn = Tensor(shape=[None, None], dtype=ms.int32)
        net.set_inputs(x_dyn, expert_idx_dyn, active_num, expert_capacity, expert_num,
                       drop_pad_mode, expert_tokens_count_or_cumsum_flag,
                       expert_tokens_before_capacity_flag)

    x_ms = Tensor(x, ms.float16)
    expert_idx_ms = Tensor(expert_idx, ms.int32)
    output = net(x_ms, expert_idx_ms, active_num, expert_capacity, expert_num,
                 drop_pad_mode, expert_tokens_count_or_cumsum_flag,
                 expert_tokens_before_capacity_flag)

    # Compare results
    for np_out, ms_out in zip(expected, output):
        np.testing.assert_allclose(ms_out.asnumpy(), np_out, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE])
@pytest.mark.parametrize('num_rows', [1, 15, 256])
@pytest.mark.parametrize('hidden_size', [7168])
@pytest.mark.parametrize('k', [8,])
@pytest.mark.parametrize('expert_num', [256])
@pytest.mark.parametrize('expert_tokens_count_or_cumsum_flag', [1,])
@pytest.mark.parametrize('expert_capacity', [0])
@pytest.mark.parametrize('drop_pad_mode', [0,])
@pytest.mark.parametrize('is_dyn', [False, True])
@pytest.mark.env_onecard
def test_moe_init_routing_v2_graph_mode(exec_mode, num_rows, hidden_size, k, expert_num, expert_tokens_count_or_cumsum_flag,
                                   expert_capacity, drop_pad_mode, is_dyn):
    """
    Feature: Test MoeInitRoutingV2 basic functionality
    Description: Test MoeInitRoutingV2 ops in graph mode
    Expectation: Run success
    """
    _test_moe_init_routing_v2(exec_mode, num_rows, hidden_size, k, expert_num,
                              expert_tokens_count_or_cumsum_flag=expert_tokens_count_or_cumsum_flag,
                              expert_capacity=expert_capacity, drop_pad_mode=drop_pad_mode, is_dyn=is_dyn)


@pytest.mark.level0
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize('exec_mode', [context.PYNATIVE_MODE])
@pytest.mark.parametrize('num_rows', [1, 15, 256])
@pytest.mark.parametrize('hidden_size', [5120, 7168])
@pytest.mark.parametrize('k', [8,])
@pytest.mark.parametrize('expert_num', [128, 256])
@pytest.mark.parametrize('expert_tokens_count_or_cumsum_flag', [1,])
@pytest.mark.parametrize('expert_capacity', [0])
@pytest.mark.parametrize('drop_pad_mode', [0,])
@pytest.mark.parametrize('is_dyn', [False])
@pytest.mark.env_onecard
def test_moe_init_routing_v2_pynative_mode(exec_mode, num_rows, hidden_size, k, expert_num, expert_tokens_count_or_cumsum_flag,
                                   expert_capacity, drop_pad_mode, is_dyn):
    """
    Feature: Test MoeInitRoutingV2 basic functionality
    Description: Test MoeInitRoutingV2 ops in PyNative mode
    Expectation: Run success
    """
    _test_moe_init_routing_v2(exec_mode, num_rows, hidden_size, k, expert_num,
                              expert_tokens_count_or_cumsum_flag=expert_tokens_count_or_cumsum_flag,
                              expert_capacity=expert_capacity, drop_pad_mode=drop_pad_mode, is_dyn=is_dyn)


@pytest.mark.level1
@pytest.mark.platform_ascend310p
@pytest.mark.env_onecard
def test_moe_init_routing_v2_large_rows():
    """
    Feature: Test MoeInitRoutingV2 with large rows
    Description: Test MoeInitRoutingV2 ops in Ascend backend
    Expectation: Run success
    """
    _test_moe_init_routing_v2(exec_mode=context.GRAPH_MODE, num_rows=2000, hidden_size=7168, k=8, expert_num=256, expert_capacity=0,
                              expert_tokens_count_or_cumsum_flag=1, drop_pad_mode=0, is_dyn=False)


@pytest.mark.level1
@pytest.mark.platform_ascend310p
@pytest.mark.env_onecard
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('hidden_size', [7165])
def test_moe_init_routing_v2_unsupported_hidden_size(exec_mode, hidden_size):
    """
    Feature: Test MoeInitRoutingV2 with unsupported hidden size
    Description: Test MoeInitRoutingV2 ops in Ascend backend
    Expectation: Unsupported hidden size correctly rejected
    """
    with pytest.raises(ValueError, match="the hidden_size of"):
        _test_moe_init_routing_v2(exec_mode=exec_mode, num_rows=10, hidden_size=hidden_size, k=8, expert_num=256, expert_capacity=0,
                                expert_tokens_count_or_cumsum_flag=1, drop_pad_mode=0, is_dyn=False)
    logging.info(f"Unsupported hidden size correctly rejected: hidden_size={hidden_size}")


@pytest.mark.level1
@pytest.mark.platform_ascend310p
@pytest.mark.env_onecard
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('num_rows', [10])
@pytest.mark.parametrize('k', [8,])
@pytest.mark.parametrize('hidden_size', [7168])
@pytest.mark.parametrize('drop_pad_mode', [0,])
@pytest.mark.parametrize('expert_num', [256])
def test_moe_init_routing_v2_1d_input(exec_mode, num_rows, k, hidden_size, drop_pad_mode, expert_num):
    """
    Feature: Test MoeInitRoutingV2 with 1d input
    Description: Test MoeInitRoutingV2 ops in Ascend backend
    Expectation: 1d input correctly rejected
    """
    context.set_context(mode=exec_mode, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0"})

    # Generate input data
    np.random.seed(0)
    x = np.random.uniform(-1, 1, size=(hidden_size)).astype(np.float16)
    expert_idx = np.random.randint(0, expert_num, size=(num_rows, k)).astype(np.int32)

    # Run on MindSpore
    net = MoeInitRoutingV2Net()

    x_ms = Tensor(x, ms.float16)
    expert_idx_ms = Tensor(expert_idx, ms.int32)

    with pytest.raises(RuntimeError, match="the rank of"):
        output = net(x_ms, expert_idx_ms, 0, 0, expert_num, drop_pad_mode, 1, False)
        out = output[0].asnumpy()
    logging.info(f"Unsupported hidden size correctly rejected: hidden_size={hidden_size}")
