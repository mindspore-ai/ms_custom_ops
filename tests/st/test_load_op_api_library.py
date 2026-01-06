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
"""tests_load_op_api_library"""

from functools import wraps
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, context, nn
from mindspore.common.api import jit

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

class ReshapeAndCacheNpdAll(nn.Cell):
    """Reshape and cache operation for NPD format"""
    def __init__(self):
        super().__init__()
        ms.ops.add(1, 2) # 触发mindspore执行LoadOpApiLib()
        # pylint: disable=import-outside-toplevel
        import ms_custom_ops
        self.reshape_and_cache_npd = ms_custom_ops.reshape_and_cache_npd

    @jit_for_graph_mode
    def construct(self, key, value, key_cache, value_cache, slot_map, actual_seq_qlen, actual_seq_kvlen, block_tbl):
        """ReshapeAndCacheNpdAll construct"""
        return self.reshape_and_cache_npd(
            key=key, value=value, key_cache=key_cache, value_cache=value_cache,
            slot_mapping=slot_map, q_seq=actual_seq_qlen, kv_seq=actual_seq_kvlen,
            block_tbl=block_tbl)

def create_inputs(dtype, k_head_dim, v_head_dim, batch, head_num):
    """Create NPD format inputs"""
    # Global constants
    slot_size = 16
    seq_len = slot_size
    key_new_shape = (batch * seq_len, head_num * k_head_dim)
    value_new_shape = (batch * seq_len, head_num * v_head_dim)
    num_tokens = key_new_shape[0]
    page_num = ((seq_len + slot_size - 1) // slot_size) * batch

    # NPD format specific cache shapes
    key_cache_shape = (page_num, head_num, slot_size, k_head_dim)
    value_cache_shape = (page_num, head_num, slot_size, v_head_dim)

    key_update = np.random.rand(*key_new_shape).astype(dtype)
    value_update = np.random.rand(*value_new_shape).astype(dtype)
    key_cache = np.random.rand(*key_cache_shape).astype(dtype)
    value_cache = np.random.rand(*value_cache_shape).astype(dtype)
    slot_map = np.random.choice(np.arange(num_tokens), num_tokens, replace=False).astype(np.int32)
    q_seq = np.full(batch, seq_len).astype(np.int32)
    kv_seq = np.full(batch, seq_len).astype(np.int32)
    block_tbl = np.arange(page_num, dtype=np.int32).reshape(batch, (page_num + batch - 1) // batch)
    return key_update, value_update, key_cache, value_cache, slot_map, q_seq, kv_seq, block_tbl

def create_ms_inputs(np_k, np_v, np_k_cache, np_v_cache, np_slot_map, np_q_seq, np_kv_seq, np_block_tbl):
    """Create MindSpore inputs"""
    return (
        Tensor(np_k), Tensor(np_v), Tensor(np_k_cache), Tensor(np_v_cache),
        Tensor(np_slot_map), Tensor(np_q_seq), Tensor(np_kv_seq), Tensor(np_block_tbl)
    )

class InferenceEngine:
    """Inference engine for NPD format"""

    @staticmethod
    def up_div(x, y):
        return (x + y - 1) // y

    @staticmethod
    def npd_permute(np_arr: np.ndarray, heads: int, embed: int, batch: int, ps: int):
        """npd permute function for 2D input"""
        t_total, hd = np_arr.shape
        assert hd == heads * embed, "Shape mismatch: second dim must be heads*embed"
        assert t_total % batch == 0, "Total tokens must be divisible by batch size"

        t = t_total // batch  # tokens per batch
        # Step 1: reshape to [b, t, h, d]
        np_arr = np_arr.reshape(batch, t, heads, embed)

        # Step 2: pad t per batch to make it divisible by ps
        padded_t = InferenceEngine.up_div(t, ps) * ps
        pad_len = padded_t - t
        if pad_len > 0:
            pad = np.zeros((batch, pad_len, heads, embed), dtype=np_arr.dtype)
            np_arr = np.concatenate([np_arr, pad], axis=1)  # pad along token dim

        # Step 3: reshape to [batch, blocks, ps, heads, embed] → permute to [batch*blocks, heads, ps, embed]
        blocks = InferenceEngine.up_div(t, ps)
        np_arr = np_arr.reshape(batch * blocks, ps, heads, embed).transpose(0, 2, 1, 3)
        np_arr = np_arr.reshape(-1, heads, embed)

        return np_arr.reshape(-1, heads, embed)

    @staticmethod
    def npd_cache_inference(key, value, key_cache, value_cache, slot_map):
        """NPD format inference"""
        key_tmp = key.copy()
        value_tmp = value.copy()
        key_cache_ans = key_cache.copy()
        value_cache_ans = value_cache.copy()

        key_head = key_cache.shape[1]
        key_head_dim = key_cache.shape[3]
        value_head = value_cache.shape[1]
        value_head_dim = value_cache.shape[3]
        slot_size = key_cache.shape[2]

        key_tmp = key_tmp.reshape(-1, key_head, key_head_dim)
        value_tmp = value_tmp.reshape(-1, value_head, value_head_dim)

        for i, slot in enumerate(slot_map):
            slot_idx = slot // slot_size
            slot_offset = slot % slot_size
            for h in range(key_head):
                key_cache_ans[slot_idx][h][slot_offset] = key_tmp[i][h]
                value_cache_ans[slot_idx][h][slot_offset] = value_tmp[i][h]
        return (key_cache_ans, value_cache_ans)

    @staticmethod
    def npd_kv_inference(key, value, key_cache, value_cache, block_tbl):
        """NPD format inference"""
        # Fixed dimensions for NPD format
        key_head = key_cache.shape[1]
        key_head_dim = key_cache.shape[3]
        value_head = value_cache.shape[1]
        value_head_dim = value_cache.shape[3]
        slot_size = key_cache.shape[2]
        batch = block_tbl.shape[0]

        return (
            InferenceEngine.npd_permute(key, key_head, key_head_dim, batch, slot_size),
            InferenceEngine.npd_permute(value, value_head, value_head_dim, batch, slot_size),
        )

def assert_equal(ms_cache, np_cache, rtol = 0.001, atol = 0.001):
    ms_cache_np = ms_cache.asnumpy()
    ms_cache_np = ms_cache_np[: np_cache.shape[0]].flatten()
    np_cache = np_cache.flatten()
    assert np.allclose(ms_cache_np, np_cache, rtol=rtol, atol=atol)

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize("run_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_reshape_and_cache_inside_cell(run_mode):
    """
    Feature: Test ReshapeAndCacheNpd Operation.
    Description: Test ReshapeAndCacheNpd when 'import ms_custom_ops' in cell init.
    Expectation: Assert that results are consistent with numpy.
    """
    np_dtype = np.float16
    kv_embed = 32
    batch = 2
    head_num = 4
    context.set_context(mode=run_mode)

    net = ReshapeAndCacheNpdAll()

    # Create inputs for NPD format
    np_k, np_v, np_k_cache, np_v_cache, np_slot_map, np_q_seq, np_kv_seq, np_block_tbl = create_inputs(
        np_dtype, kv_embed, kv_embed, batch, head_num
    )

    # Generate golden results for NPD format
    np_k_out, np_v_out = InferenceEngine.npd_kv_inference(
        np_k, np_v, np_k_cache, np_v_cache, np_block_tbl
    )
    np_k_cache_out, np_v_cache_out = InferenceEngine.npd_cache_inference(
        np_k, np_v, np_k_cache, np_v_cache, np_slot_map
    )

    # Create MindSpore inputs
    ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map, ms_q_seq, ms_kv_seq, ms_block_tbl = create_ms_inputs(
        np_k, np_v, np_k_cache, np_v_cache, np_slot_map, np_q_seq, np_kv_seq, np_block_tbl
    )
    ms_q_seq = ms_q_seq.move_to("CPU")
    ms_kv_seq = ms_kv_seq.move_to("CPU")

    # Run test
    ms_k_out, ms_v_out = net(ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map, ms_q_seq, ms_kv_seq, ms_block_tbl)

    # Verify Output
    assert_equal(ms_k_cache, np_k_cache_out)
    assert_equal(ms_v_cache, np_v_cache_out)
    assert_equal(ms_k_out, np_k_out)
    assert_equal(ms_v_out, np_v_out)

