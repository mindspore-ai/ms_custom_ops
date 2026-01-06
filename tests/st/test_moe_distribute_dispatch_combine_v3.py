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
test dispatch and combine
"""

import numpy as np
from functools import wraps
import pytest
import mindspore as ms
import ms_custom_ops
from mindspore.communication import init, get_rank, create_group


def jit(func):

    @wraps(func)
    def decorator(*args, **kwargs):
        if ms.get_context("mode") == "PYNATIVE_MODE":
            return func(*args, **kwargs)
        return ms.jit(func, jit_level="O0", infer_boost="on")(*args, **kwargs)

    return decorator


class MoeDistributeDispatchNet(ms.nn.Cell):

    def __init__(self,
                 global_bs=0,
                 moe_expert_num=16,
                 ep_world_size=16,
                 group_ep="ep"):
        super().__init__()
        self.moe_distribute_dispatch = ms_custom_ops.moe_distribute_dispatch_v3
        self.rank = get_rank()
        self.moe_expert_num = moe_expert_num
        self.ep_world_size = ep_world_size
        self.group_ep = group_ep
        self.ep_rank_id = get_rank(group=group_ep)
        self.global_bs = global_bs

    @jit
    def construct(self, x, expert_ids, scales=None):
        out = self.moe_distribute_dispatch(x=x,
                                           expert_ids=expert_ids,
                                           group_ep=self.group_ep,
                                           ep_world_size=self.ep_world_size,
                                           ep_rank_id=self.ep_rank_id,
                                           moe_expert_num=self.moe_expert_num,
                                           scales=scales,
                                           global_bs=self.global_bs)
        return out


class MoeDistributeCombineNet(ms.nn.Cell):

    def __init__(self,
                 global_bs=0,
                 moe_expert_num=16,
                 ep_world_size=16,
                 group_ep="ep"):
        super().__init__()
        self.moe_distribute_combine = ms_custom_ops.moe_distribute_combine_v3
        self.rank = get_rank()
        self.moe_expert_num = moe_expert_num
        self.ep_world_size = ep_world_size
        self.group_ep = group_ep
        self.ep_rank_id = get_rank(group=group_ep)
        self.global_bs = global_bs

    @jit
    def construct(self, expand_x, expert_ids, assist_info_for_combine,
                  expert_scales, ep_send_counts, tp_send_counts):
        out = self.moe_distribute_combine(
            expand_x=expand_x,
            expert_ids=expert_ids,
            assist_info_for_combine=assist_info_for_combine,
            expert_scales=expert_scales,
            group_ep=self.group_ep,
            ep_world_size=self.ep_world_size,
            ep_rank_id=self.ep_rank_id,
            moe_expert_num=self.moe_expert_num,
            ep_send_counts=ep_send_counts,
            tp_send_counts=tp_send_counts,
            global_bs=self.global_bs)
        return out


def run_moe_distribute_dispatch_and_combine(mode,
                                            bs,
                                            k,
                                            h,
                                            moe_expert_num,
                                            ep_world_size=16,
                                            global_bs=0):
    np.random.seed(1)
    ms.set_context(mode=mode)
    init()
    rank_id = get_rank()

    ep_rank_list = [list(range(ep_world_size))]
    for ep_rank in ep_rank_list:
        if rank_id in ep_rank:
            group_ep = "ep-" + "-".join([str(i) for i in ep_rank])
            create_group(group_ep, ep_rank)
    x = np.random.uniform(low=-5, high=5, size=(bs, h)).astype(np.float16)
    expert_ids = np.argsort(np.random.randn(bs, moe_expert_num),
                            axis=1)[:, :k].astype(np.int32)
    expert_scales = np.random.randn(bs, k).astype(np.float32)
    ms_x = ms.Tensor(x)
    ms_expert_ids = ms.Tensor(expert_ids)
    ms_expert_scales = ms.Tensor(expert_scales)
    dispatch = MoeDistributeDispatchNet(global_bs=global_bs,
                                        moe_expert_num=moe_expert_num,
                                        ep_world_size=ep_world_size,
                                        group_ep=group_ep)
    combine = MoeDistributeCombineNet(global_bs=global_bs,
                                      moe_expert_num=moe_expert_num,
                                      ep_world_size=ep_world_size,
                                      group_ep=group_ep)
    expand_x, _, assist_info_for_combine, _, ep_recv_counts, tp_recv_counts, _ = dispatch(
        ms_x, ms_expert_ids)
    out = combine(expand_x, ms_expert_ids, assist_info_for_combine,
                  ms_expert_scales, ep_recv_counts, tp_recv_counts)


@pytest.mark.level2
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("bs", [8, 16])
@pytest.mark.parametrize("k", [8])
@pytest.mark.parametrize("h", [2048, 7168])
@pytest.mark.parametrize("moe_expert_num", [256])
def test_dispatch_and_combine_v3_8p(mode, bs, k, h, moe_expert_num):
    """
    Feature:aclnnMoeDistributeDispatchV3 and aclnnMoeDistributeCombineV3 kernel.
    Description: test for moe_distribute_dispatch_v3 and moe_distribute_combine_v3 ops.
    Expectation:should pass for all testcases.
    Command: msrun --worker_num=8 --local_worker_num=8 --join True pytest -s -v  \
             test_moe_distribute_dispatch_combine_v3.py::test_dispatch_and_combine_v3_8p
    """
    ep_world_size = 8
    run_moe_distribute_dispatch_and_combine(mode,
                                            bs,
                                            k,
                                            h,
                                            moe_expert_num,
                                            ep_world_size,
                                            global_bs=0)
    run_moe_distribute_dispatch_and_combine(mode,
                                            bs,
                                            k,
                                            h,
                                            moe_expert_num,
                                            ep_world_size,
                                            global_bs=bs * ep_world_size)
