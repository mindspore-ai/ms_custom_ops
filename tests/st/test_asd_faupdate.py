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

"""
faupdate operator testcase
"""

import mindspore as ms
import numpy as np
import pytest
from mindspore import Tensor, context
from st_utils import custom_compare

import ms_custom_ops


class AsdFaUpdateCustom(ms.nn.Cell):
    def construct(self, lse, local_out, fa_update_type, sp):
        return ms_custom_ops.fa_update(lse, local_out, fa_update_type, sp)


def golden_calc(lse, local_out):
    """
    golden calculation
    """
    all_lse = lse.copy()
    all_out = local_out.copy()
    sp = all_out.shape[0]
    hd = all_out.shape[-1]

    all_lse = np.transpose(all_lse, (1, 0))
    all_out = np.transpose(all_out, (1, 2, 0)).reshape(-1, sp * hd)
    # (b * s * hc, sp)
    lse_exp = np.exp(all_lse)
    # (b * s * hc, 1)
    sum_lse_exp = np.sum(lse_exp, axis=-1, keepdims=True)
    # (b * s * hc, sp)
    lse_exp = lse_exp / sum_lse_exp

    # oi = lse_exp * oi (b * s * hc, hd, sp) * (b * s * hc, hd, sp)
    lse_exp = np.repeat(lse_exp[:, np.newaxis, :], hd, axis=1)
    all_out = all_out.reshape(-1, hd, sp)
    all_out = all_out * lse_exp

    # o = sum(oi) (b * s * hc, hd)
    all_out = np.sum(all_out, axis=-1, keepdims=True)
    return all_out


def fa_update_function(sp, batch, seq_len, head_num, head_size):
    """
    testcase main function
    """
    shape0 = (sp, batch * seq_len * head_num)
    shape1 = (sp, batch * seq_len * head_num, head_size)
    lse = np.random.rand(*shape0).astype(np.float32)
    local_out = np.random.rand(*shape1).astype(np.float32)
    net = AsdFaUpdateCustom()
    output = net(Tensor(lse), Tensor(local_out), 0, sp)
    output_golden = golden_calc(lse, local_out)
    custom_compare(output.asnumpy(), output_golden, ms.float32)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('context_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_fa_update_base(context_mode):
    """
    Feature: test fa_update operator
    Description: test fa_update
    Expectation: the result is correct
    """
    context.set_context(mode=context_mode, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0"})
    sp = 1
    head_size = 8
    batch = 11
    seq_len = 1
    head_num = 13
    fa_update_function(sp, batch, seq_len, head_num, head_size)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('context_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_fa_update_large(context_mode):
    """
    Feature: test fa_update operator
    Description: test fa_update
    Expectation: the result is correct
    """
    context.set_context(mode=context_mode, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0"})
    sp = 3
    head_size = 128
    batch = 1000
    seq_len = 1
    head_num = 13
    fa_update_function(sp, batch, seq_len, head_num, head_size)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('context_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_fa_update_prefill(context_mode):
    """
    Feature: test fa_update operator
    Description: test fa_update
    Expectation: the result is correct
    """
    context.set_context(mode=context_mode, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0"})
    sp = 3
    head_size = 128
    batch = 2
    seq_len = 4000
    head_num = 32
    fa_update_function(sp, batch, seq_len, head_num, head_size)
