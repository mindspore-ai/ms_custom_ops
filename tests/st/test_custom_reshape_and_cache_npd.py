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
""" tests_custom_pyboost_ascend """

# Standard library imports
from enum import Enum
from functools import wraps
from typing import Tuple, Optional, Dict, Any

# Third-party imports
import numpy as np
import pytest

# MindSpore imports
import mindspore as ms
from mindspore import Tensor, context, nn
from mindspore.common.api import jit
from mindspore.common.np_dtype import bfloat16

# Local imports
import ms_custom_ops

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

# Global constants
NUM_SLOTS = 20
SLOT_SIZE = 16
BATCH_SIZE = 13
SEQ_LEN = SLOT_SIZE
NUM_HEADS = 32
K_HEAD_DIM = 128
V_HEAD_DIM = 128

class CacheFormat(Enum):
    """Cache format enumeration"""
    ND = "nd"
    NPD = "npd"


class DataType(Enum):
    """Data type enumeration"""
    FLOAT16 = np.float16
    BFLOAT16 = bfloat16


class ReshapeAndCacheNpdAll(nn.Cell):
    """Reshape and cache operation for ND/NPD format with all parameters"""

    @jit_for_graph_mode
    def construct(self, key, value, key_cache, value_cache, slot_map, q_seq, kv_seq, block_tbl, cache_mode=1):
        k_out, value_out =  ms_custom_ops.reshape_and_cache_npd(
            key, value, key_cache, value_cache, slot_map, q_seq, kv_seq, block_tbl, cache_mode)
        return k_out, value_out


class MindSporeInputFactory:
    """Factory for creating MindSpore inputs"""

    @staticmethod
    def create_inputs(np_k: np.ndarray, np_v: np.ndarray,
                     np_k_cache: np.ndarray, np_v_cache: np.ndarray,
                     np_slot_map: np.ndarray, np_q_seq: np.ndarray, np_kv_seq: np.ndarray,
                     np_block_tbl: np.ndarray) -> Tuple[Tensor, ...]:
        """Create MindSpore inputs"""
        ms_key = Tensor(np_k)
        ms_value = Tensor(np_v)
        ms_key_cache = Tensor(np_k_cache)
        ms_value_cache = Tensor(np_v_cache)
        ms_slot_map = Tensor(np_slot_map)
        ms_q_seq = Tensor(np_q_seq)
        ms_kv_seq = Tensor(np_kv_seq)
        ms_block_tbl = Tensor(np_block_tbl)
        return ms_key, ms_value, ms_key_cache, ms_value_cache, ms_slot_map, ms_q_seq, ms_kv_seq, ms_block_tbl


def create_ms_inputs(np_k, np_v, np_k_cache, np_v_cache, np_slot_map, q_seq, kv_seq, block_tbl):
    """Legacy function for backward compatibility"""
    return MindSporeInputFactory.create_inputs(np_k, np_v, np_k_cache, np_v_cache, np_slot_map, q_seq, kv_seq,
                                               block_tbl)


class TestResultVerifier:
    """Verify test results"""

    @staticmethod
    def verify_results(ms_cache: Tensor, np_cache: np.ndarray,
                      dtype: np.dtype, truncate: bool = False, rtol: float = 0.001, atol: float = 0.001) -> None:
        """Verify results with appropriate dtype handling"""
        if dtype == bfloat16:
            ms_cache_np = ms_cache.float().asnumpy()
            np_cache = np_cache.astype(np.float32)
        else:
            ms_cache_np = ms_cache.asnumpy()

        if truncate is False:
            ms_cache_np =  ms_cache_np.flatten()
            np_cache =  np_cache.flatten()
        else:
            ms_cache_np =  ms_cache_np[:np_cache.shape[0]].flatten()
            np_cache =  np_cache.flatten()

        assert np.allclose(ms_cache_np, np_cache, rtol=rtol, atol=atol)


class TestConfig:
    """Test configuration"""

    def __init__(self, device_target: str = "Ascend", mode: context = context.GRAPH_MODE,
                 jit_config: Optional[Dict[str, Any]] = None):
        self.device_target = device_target
        self.mode = mode
        self.jit_config = jit_config or {}

    def apply(self):
        """Apply test configuration"""
        ms.set_device(self.device_target)
        context.set_context(mode=self.mode)
        if self.jit_config:
            context.set_context(jit_config=self.jit_config)


class DimensionTestHelper:
    """Helper class for testing different dimension combinations"""

    @staticmethod
    def run_with_dimensions(k_head_dim: int, v_head_dim: int, test_func):
        """Run test with specified dimensions and restore original values"""
        global K_HEAD_DIM, V_HEAD_DIM
        original_k_head_dim = K_HEAD_DIM
        original_v_head_dim = V_HEAD_DIM

        try:
            K_HEAD_DIM = k_head_dim
            V_HEAD_DIM = v_head_dim
            test_func()
        finally:
            K_HEAD_DIM = original_k_head_dim
            V_HEAD_DIM = original_v_head_dim


# ===============================
#        RESHAPE AND CACHE NPD TEST ARCHITECTURE
# ===============================
"""
Test Structure Overview:

1. ND FORMAT TESTS (cache_mode=0):
   - Direct ND format testing without format conversion
   - Data flow: Input(ND) → ReshapeAndCacheNpd → Output(ND) → Verify
   - Tests: test_reshape_and_cache_nd_*

2. NZ FORMAT TESTS (cache_mode=1): 
   - Tests NPD format with format conversion using trans_data
   - Data flow: Input(ND) → TransData(ND→NZ) → ReshapeAndCacheNpd → TransData(NZ→ND) → Verify
   - Tests: test_reshape_and_cache_npd_*
   
3. KEY COMPONENTS:
   - create_nd_inputs(): Generate ND format test data
   - create_npd_inputs(): Generate NZ-compatible test data (different layout)
   - get_nd_cached_slots(): Extract verification data from ND format cache
   - get_npd_cached_slots(): Extract verification data from NZ format cache (legacy)
   - npd_inference(): Generate golden reference results

4. VERIFICATION STRATEGY:
   - ND tests: Both actual and golden use ND format → direct comparison
   - NPD tests: Convert actual results back to ND format → compare with ND golden
"""

# ===============================
#        NPD FORMAT TESTS
# ===============================
class TestDataGenerator:
    """Data generator for test inputs"""

    @staticmethod
    def create_random_data(shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Create random data with specified shape and dtype"""
        if dtype == np.int8:
            return np.random.randint(low=-128, high=127, size=shape, dtype=np.int8)
        return np.random.rand(*shape).astype(dtype)

    @staticmethod
    def create_slot_map(num_tokens: int) -> np.ndarray:
        """Create slot mapping"""
        return np.random.choice(np.arange(num_tokens), num_tokens, replace=False).astype(np.int32)

    @staticmethod
    def create_q_seq(batch: int, seq:int) -> np.ndarray:
        """Create q seq lens"""
        return np.full(batch, seq).astype(np.int32)

    @staticmethod
    def create_kv_seq(batch: int, seq:int) -> np.ndarray:
        """Create kv seq lens"""
        return np.full(batch, seq).astype(np.int32)

    @staticmethod
    def create_blk_tbl(page_num: int, batch: int) -> np.ndarray:
        return np.arange(page_num, dtype=np.int32).reshape(batch, (page_num + batch - 1) // batch)

    @staticmethod
    def get_update_shapes(kv_dim: int, k_head_dim=None, v_head_dim=None, batch=None, head_num = None
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], int, int]:
        """Get update shapes for key and value, and number of tokens based on dimension"""
        # Use provided dimensions or fall back to global constants
        actual_k_head_dim = k_head_dim if k_head_dim is not None else K_HEAD_DIM
        actual_v_head_dim = v_head_dim if v_head_dim is not None else V_HEAD_DIM
        actual_head_num = head_num if head_num is not None else NUM_HEADS
        actual_batch = batch if batch is not None else BATCH_SIZE
        if kv_dim == 2:
            key_update_shape = (actual_batch * SEQ_LEN, actual_head_num * actual_k_head_dim)
            value_update_shape = (actual_batch * SEQ_LEN, actual_head_num * actual_v_head_dim)
            num_tokens = key_update_shape[0]
        elif kv_dim == 3:
            key_update_shape = (actual_batch, SEQ_LEN, actual_head_num * actual_k_head_dim)
            value_update_shape = (actual_batch, SEQ_LEN, actual_head_num * actual_v_head_dim)
            num_tokens = key_update_shape[0] * key_update_shape[1]
        else:
            raise ValueError(f"Key's dim should be 2 or 3, but got {kv_dim}")

        page_num = ((SEQ_LEN + SLOT_SIZE) // SLOT_SIZE) * actual_batch
        return key_update_shape, value_update_shape, num_tokens, page_num

    @staticmethod
    def get_update_shape(kv_dim: int, is_key: bool = True, k_head_dim=None, v_head_dim=None, batch=None, head_num=None
    ) -> Tuple[Tuple[int, ...], int, int]:
        """Legacy method for backward compatibility"""
        key_shape, value_shape, num_tokens, page_num = TestDataGenerator.get_update_shapes(kv_dim, k_head_dim,
                                                                                           v_head_dim, batch)
        return (key_shape if is_key else value_shape), num_tokens, page_num


class NPDDataGenerator(TestDataGenerator):
    """Data generator for NPD format"""

    @staticmethod
    def create_npd_inputs(dtype: np.dtype, kv_dim: int, k_head_dim=None, v_head_dim=None, batch=None, head_num=None
    ) -> Tuple[np.ndarray, ...]:
        """Create NPD format inputs"""
        # Use provided dimensions or fall back to global constants
        actual_k_head_dim = k_head_dim if k_head_dim is not None else K_HEAD_DIM
        actual_v_head_dim = v_head_dim if v_head_dim is not None else V_HEAD_DIM
        actual_batch = batch if batch is not None else BATCH_SIZE
        actual_head_num = head_num if head_num is not None else NUM_HEADS

        key_new_shape, value_new_shape, num_tokens, page_num = TestDataGenerator.get_update_shapes(kv_dim, k_head_dim,
                                                                                                   v_head_dim, batch,
                                                                                                   head_num)
        key_cache_shape = (page_num, actual_head_num, SLOT_SIZE, actual_k_head_dim)
        value_cache_shape = (page_num, actual_head_num, SLOT_SIZE, actual_v_head_dim)

        key_update = TestDataGenerator.create_random_data(key_new_shape, dtype)
        value_update = TestDataGenerator.create_random_data(value_new_shape, dtype)
        key_cache = TestDataGenerator.create_random_data(key_cache_shape, dtype)
        value_cache = TestDataGenerator.create_random_data(value_cache_shape, dtype)
        slot_map = TestDataGenerator.create_slot_map(num_tokens)
        q_seq = TestDataGenerator.create_q_seq(actual_batch, SEQ_LEN)
        kv_seq = TestDataGenerator.create_kv_seq(actual_batch, SEQ_LEN)
        block_tbl = TestDataGenerator.create_blk_tbl(page_num, actual_batch)

        return key_update, value_update, key_cache, value_cache, slot_map, q_seq, kv_seq, block_tbl


def create_npd_inputs(dtype=np.float16, kv_dim=3, k_head_dim=None, v_head_dim=None, batch=None, head_num=None):
    """Legacy function for backward compatibility"""
    return NPDDataGenerator.create_npd_inputs(dtype, kv_dim, k_head_dim, v_head_dim, batch, head_num)


class InferenceEngine:
    """Inference engine for different formats"""
    @staticmethod
    def up_div(x, y):
        return (x + y - 1) // y

    @staticmethod
    def npd_permute(np_arr: np.ndarray, heads:int, embed:int, batch:int, ps:int):
        if np_arr.ndim == 2:
            t_total, hd = np_arr.shape
        else:
            b, s, hd = np_arr.shape
            t_total = b * s

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
        np_arr = np_arr.reshape(batch*blocks, ps, heads, embed).transpose(0, 2, 1, 3)
        np_arr = np_arr.reshape(-1, heads, embed)

        return np_arr.reshape(-1, heads,embed)

    @staticmethod
    def npd_inference(key: np.ndarray, value: np.ndarray, key_cache: np.ndarray, value_cache: np.ndarray,
                      slot_map: np.ndarray, block_tbl:np.array
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """NPD format inference"""
        key_tmp = key.copy()
        value_tmp = value.copy()
        key_cache_ans = key_cache.copy()
        value_cache_ans = value_cache.copy()

        # Use different dimensions for key and value
        key_head = key_cache.shape[1]
        key_head_dim = key_cache.shape[3]
        value_head = value_cache.shape[1]
        value_head_dim = value_cache.shape[3]
        slot_size = key_cache.shape[2]
        batch = block_tbl.shape[0]
        key_tmp = key_tmp.reshape(-1, key_head, key_head_dim)
        value_tmp = value_tmp.reshape(-1, value_head, value_head_dim)

        for i, slot in enumerate(slot_map):
            slot_idx = slot // slot_size
            slot_offset = slot % slot_size
            for h in range(key_head):
                key_cache_ans[slot_idx][h][slot_offset] = key_tmp[i][h]
                value_cache_ans[slot_idx][h][slot_offset] = value_tmp[i][h]
        return (InferenceEngine.npd_permute(key, key_head, key_head_dim, batch, slot_size),
                InferenceEngine.npd_permute(value, value_head, value_head_dim, batch, slot_size), key_cache_ans,
                value_cache_ans)

def npd_inference(key, value, key_cache, value_cache, slot_map, block_tbl):
    """Legacy function for backward compatibility"""
    return InferenceEngine.npd_inference(key, value, key_cache, value_cache, slot_map, block_tbl)

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('np_dtype', [np.float16, bfloat16])
@pytest.mark.parametrize('kv_embed', [32, 128, 256])
@pytest.mark.parametrize('batch', [1, 2, 8, 13, 16])
@pytest.mark.parametrize('head_num', [4, 8, 32])
def test_reshape_and_cache_nd_key_value(np_dtype, kv_embed, batch, head_num, run_mode):
    """
    Feature: Test ReshapeAndCacheNpd.
    Description: Test ND format with key and value.
    Expectation: Assert that results are consistent with numpy.
    """
    kv_dim = 2
    test_config = TestConfig(device_target="Ascend", mode=run_mode)
    test_config.apply()

    net = ReshapeAndCacheNpdAll()

    np_k, np_v, np_k_cache, np_v_cache, np_slot_map, np_q_seq, np_kv_seq, np_block_tbl = create_npd_inputs(
        np_dtype, kv_dim, kv_embed, kv_embed, batch, head_num)
    np_k_out, np_v_out, np_k_cache_out, np_v_cache_out = npd_inference(
        np_k, np_v, np_k_cache, np_v_cache, np_slot_map, np_block_tbl)

    ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map, ms_q_seq, ms_kv_seq, ms_block_tbl = create_ms_inputs(
        np_k, np_v, np_k_cache, np_v_cache, np_slot_map, np_q_seq, np_kv_seq, np_block_tbl)

    # Run test
    ms_k_out, ms_v_out = net(ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map, ms_q_seq, ms_kv_seq, ms_block_tbl, 1)

    # Verify Output
    TestResultVerifier.verify_results(ms_k_cache, np_k_cache_out, np_dtype)
    TestResultVerifier.verify_results(ms_v_cache, np_v_cache_out, np_dtype)
    TestResultVerifier.verify_results(ms_k_out, np_k_out, np_dtype, True)
    TestResultVerifier.verify_results(ms_v_out, np_v_out, np_dtype, True)
