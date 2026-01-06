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
"""tests_custom_pyboost_ascend"""

# Standard library imports
from enum import Enum
from functools import wraps
from typing import Any, Dict, Optional, Tuple

# MindSpore imports
import mindspore as ms
# Third-party imports
import numpy as np
import pytest
from mindspore import Tensor, context, ops
from mindspore.common.api import jit
from mindspore.common.np_dtype import bfloat16
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore

# Local imports
import ms_custom_ops

ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})


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


class DataType(Enum):
    """Data type enumeration"""

    FLOAT16 = np.float16
    BFLOAT16 = bfloat16


class FlashAttention(Cell):
    """Flash Attention Layer."""

    def __init__(self, head_num, scale_value):
        super().__init__()
        self.head_num = head_num
        self.scale_value = scale_value
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()
        self.flash_attention = FlashAttentionScore(head_num=head_num, scale_value=scale_value, input_layout="TH")

    @jit_for_graph_mode
    def construct(
        self,
        query=None,
        key=None,
        value=None,
        key_cache=None,
        value_cache=None,
        slot_mapping=None,
        attn_mask=None,
        actual_seq_qlen=None,
        actual_seq_kvlen=None,
    ):
        """Forward process of the FlashAttention."""
        key = key.contiguous()
        value = value.contiguous()
        self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
        _, _, _, context_ = self.flash_attention(
            query, key, value, None, None, None, attn_mask, None, actual_seq_qlen, actual_seq_kvlen
        )
        return context_


class FlashAttentionNpd(Cell):
    """Flash Attention NPD Layer."""

    def __init__(self, head_num, scale_value, block_size):
        super().__init__()
        self.head_num = head_num
        self.scale_value = scale_value
        self.cache_layout = "BSND"
        self.kv_layout = "NPD"
        self.block_size = block_size

    @jit_for_graph_mode
    def construct(
        self,
        query=None,
        key=None,
        value=None,
        key_cache=None,
        value_cache=None,
        slot_mapping=None,
        attn_mask=None,
        actual_seq_qlen=None,
        actual_seq_kvlen=None,
        block_table=None,
    ):
        """ReshapeAndCacheNpd construct"""
        key = key.contiguous()
        value = value.contiguous()
        k_out, value_out = ms_custom_ops.reshape_and_cache_npd(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            actual_seq_qlen,
            actual_seq_kvlen,
            block_table,
            self.cache_layout,
            self.kv_layout,
        )
        out = ms_custom_ops.unpad_fa_npd(
            query=query,
            key=k_out,
            value=value_out,
            attn_mask=attn_mask,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            head_num=self.head_num,
            scale_value=self.scale_value,
            q_input_layout="TH",
            kv_input_layout=self.kv_layout,
            block_size=self.block_size,
        )
        return out


class MindSporeInputFactory:
    """Factory for creating MindSpore inputs"""

    @staticmethod
    def create_inputs(
        np_q: np.ndarray,
        np_k: np.ndarray,
        np_v: np.ndarray,
        np_k_cache: np.ndarray,
        np_v_cache: np.ndarray,
        np_slot_map: np.ndarray,
        np_attn_mask: np.ndarray,
        np_q_seq: np.ndarray,
        np_kv_seq: np.ndarray,
        np_block_tbl: np.ndarray,
    ) -> Tuple[Tensor, ...]:
        """Create MindSpore inputs"""
        ms_query = Tensor(np_q)
        ms_key = Tensor(np_k)
        ms_value = Tensor(np_v)
        ms_key_cache = Tensor(np_k_cache)
        ms_value_cache = Tensor(np_v_cache)
        ms_slot_map = Tensor(np_slot_map)
        ms_attn_mask = Tensor(np_attn_mask)
        ms_q_seq = Tensor(np_q_seq)
        ms_kv_seq = Tensor(np_kv_seq)
        ms_block_tbl = Tensor(np_block_tbl)
        return (
            ms_query,
            ms_key,
            ms_value,
            ms_key_cache,
            ms_value_cache,
            ms_slot_map,
            ms_attn_mask,
            ms_q_seq,
            ms_kv_seq,
            ms_block_tbl,
        )


def create_ms_inputs(np_q, np_k, np_v, np_k_cache, np_v_cache, np_slot_map, np_attn_mask, q_seq, kv_seq, block_tbl):
    return MindSporeInputFactory.create_inputs(
        np_q, np_k, np_v, np_k_cache, np_v_cache, np_slot_map, np_attn_mask, q_seq, kv_seq, block_tbl
    )


class TestResultVerifier:
    """Verify test results"""

    @staticmethod
    def verify_results(golden: Tensor, out: Tensor, dtype: np.dtype, rtol: float = 0.001, atol: float = 0.001) -> None:
        """Verify results with appropriate dtype handling"""

        golden_np = golden.float().asnumpy().astype(np.float32).flatten()
        out_np = out.float().asnumpy().astype(np.float32).flatten()
        if dtype == bfloat16:
            rtol = 0.01
        assert np.allclose(golden_np, out_np, rtol=rtol, atol=atol)


class TestConfig:
    """Test configuration"""

    def __init__(
        self,
        device_target: str = "Ascend",
        mode: context = context.GRAPH_MODE,
        jit_config: Optional[Dict[str, Any]] = None,
    ):
        self.device_target = device_target
        self.mode = mode
        self.jit_config = jit_config or {}

    def apply(self):
        """Apply test configuration"""
        ms.set_device(self.device_target)
        context.set_context(mode=self.mode)
        if self.jit_config:
            context.set_context(jit_config=self.jit_config)


# ===============================
#       UNPAD FA NPD TEST ARCHITECTURE
# ===============================
# """
# Test Structure Overview:
#
# 1. BSND FORMAT TESTS (cache_mode=0):
#  - Direct BSND format testing without format conversion
#  - Data flow: Input(BSND) → flash attention → Output(BSND) → Verify
#  - Tests: test_reshape_and_cache_nd_*
#
# 3. KEY COMPONENTS:
#  - create_inputs(): Generate BSND format test data
#  - FlashAttention - generate golden output
#  - FlashAttentionNpd - new attention output
#
# 4. VERIFICATION STRATEGY:
#  - compare results with FlashAttention


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
    def create_slot_map(batch: int, seq: int, ps: int) -> np.ndarray:
        """Create slot mapping"""
        slot_map = np.zeros((batch, seq), dtype=np.int32)
        round_seq = ((seq + ps - 1) // ps) * ps
        for i in range(batch):
            slot_map[i, :seq] = np.arange(seq) + i * round_seq
        return slot_map.reshape(batch * seq)

    @staticmethod
    def create_q_seq(batch: int, seq: int) -> np.ndarray:
        """Create q seq lens"""
        return np.full(batch, seq).astype(np.int32)

    @staticmethod
    def create_kv_seq(batch: int, seq: int) -> np.ndarray:
        """Create kv seq lens"""
        return np.full(batch, seq).astype(np.int32)

    @staticmethod
    def create_blk_tbl(batch: int, page_num: int) -> np.ndarray:
        blk_tbl = np.ones((batch, page_num), dtype=np.int32) * -1
        for i in range(batch):
            blk_tbl[i, :page_num] = np.arange(page_num) + i * page_num
        return blk_tbl

    @staticmethod
    def create_attn_mask(dtype: np.dtype) -> np.ndarray:
        amask = np.ones(shape=(128, 128))
        amask = np.triu(amask, 1)
        if dtype == bfloat16:
            return amask.astype(dtype)
        return (amask * -10000).astype(dtype)

    @staticmethod
    def get_update_shapes(
        batch_dim: int, seq: int, q_head_num: int, kv_head_num: int, head_dim: int, block_size: int
    ) -> Tuple[Tuple[int, ...]]:
        """Get update shapes for key and value, and number of tokens based on dimension"""

        query = (batch_dim * seq, q_head_num * head_dim)
        key = (batch_dim * seq, kv_head_num * head_dim)
        value = (batch_dim * seq, kv_head_num * head_dim)
        num_tokens = key[0]

        block_num = (seq + block_size - 1) // block_size
        k_cache = (block_num * batch_dim, block_size, kv_head_num, head_dim)
        v_cache = (block_num * batch_dim, block_size, kv_head_num, head_dim)
        slot_map = num_tokens
        q_seq = num_tokens
        k_seq = num_tokens
        block_tbl = (batch_dim, block_num)

        return query, key, value, k_cache, v_cache, slot_map, q_seq, k_seq, block_tbl


class DataGenerator(TestDataGenerator):
    """Data generator for NPD format"""

    @staticmethod
    def create_inputs(
        dtype: np.dtype, batch_dim: int, seq: int, q_head_num: int, kv_head_num: int, head_dim: int, block_size: int
    ) -> Tuple[np.ndarray, ...]:
        """create inputs"""
        (
            query_shape,
            key_shape,
            value_shape,
            k_cache_shape,
            v_cache_shape,
            _,
            _,
            _,
            block_tbl_shape,
        ) = TestDataGenerator.get_update_shapes(batch_dim, seq, q_head_num, kv_head_num, head_dim, block_size)
        query = TestDataGenerator.create_random_data(query_shape, dtype)
        key = TestDataGenerator.create_random_data(key_shape, dtype)
        value = TestDataGenerator.create_random_data(value_shape, dtype)
        key_cache = TestDataGenerator.create_random_data(k_cache_shape, dtype)
        value_cache = TestDataGenerator.create_random_data(v_cache_shape, dtype)
        slot_map = TestDataGenerator.create_slot_map(batch_dim, seq, block_size)
        q_seq = TestDataGenerator.create_q_seq(batch_dim, seq)
        kv_seq = TestDataGenerator.create_kv_seq(batch_dim, seq)
        block_tbl = TestDataGenerator.create_blk_tbl(block_tbl_shape[0], block_tbl_shape[1])
        attn_mask = TestDataGenerator.create_attn_mask(dtype)

        return query, key, value, key_cache, value_cache, slot_map, attn_mask, q_seq, kv_seq, block_tbl


def create_inputs(dtype, batch_dim: int, seq: int, q_head_num: int, kv_head_num: int, head_dim: int, block_size: int):
    """Legacy function for backward compatibility"""
    return DataGenerator.create_inputs(dtype, batch_dim, seq, q_head_num, kv_head_num, head_dim, block_size)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize("np_dtype", [np.float16, bfloat16])
@pytest.mark.parametrize("run_mode", [context.PYNATIVE_MODE, context.GRAPH_MODE])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("seq", [128, 300, 8192])
@pytest.mark.parametrize("head_num", [32])
@pytest.mark.parametrize("kv_head_num", [8])
@pytest.mark.parametrize("embed", [128])
def test_unpad_fa_npd(np_dtype, run_mode, batch, seq, head_num, kv_head_num, embed):
    """
    Feature: Test unpad flash attention.
    Description: Test BSND format query, key, value and mask.
    Expectation: Assert that results are consistent with numpy.
    """
    test_config = TestConfig(device_target="Ascend", mode=run_mode)
    test_config.apply()

    block_size = 32
    net0 = FlashAttention(head_num, 1.0)
    net1 = FlashAttentionNpd(head_num, 1.0, block_size)

    (
        np_query,
        np_key,
        np_value,
        np_key_cache,
        np_value_cache,
        np_slot_map,
        np_attn_mask,
        np_q_seq,
        np_kv_seq,
        np_block_tbl,
    ) = create_inputs(np_dtype, batch, seq, head_num, kv_head_num, embed, block_size)

    (
        ms_query,
        ms_key,
        ms_value,
        ms_key_cache,
        ms_value_cache,
        ms_slot_map,
        ms_attn_mask,
        ms_q_seq,
        ms_kv_seq,
        ms_block_tbl,
    ) = create_ms_inputs(
        np_query,
        np_key,
        np_value,
        np_key_cache,
        np_value_cache,
        np_slot_map,
        np_attn_mask,
        np_q_seq,
        np_kv_seq,
        np_block_tbl,
    )

    # Run test
    golden_out = net0(
        ms_query, ms_key, ms_value, ms_key_cache, ms_value_cache, ms_slot_map, ms_attn_mask, ms_q_seq, ms_kv_seq
    )
    ms_q_seq = ms_q_seq.move_to("CPU")
    ms_kv_seq = ms_kv_seq.move_to("CPU")
    out = net1(
        ms_query,
        ms_key,
        ms_value,
        ms_key_cache,
        ms_value_cache,
        ms_slot_map,
        ms_attn_mask,
        ms_q_seq,
        ms_kv_seq,
        ms_block_tbl,
    )
    # Verify Output
    TestResultVerifier.verify_results(golden_out, out, np_dtype)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize("np_dtype", [np.float16, bfloat16])
@pytest.mark.parametrize("run_mode", [context.PYNATIVE_MODE, context.GRAPH_MODE])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("seq", [128, 300, 8192])
@pytest.mark.parametrize("head_num", [16])
@pytest.mark.parametrize("kv_head_num", [16])
@pytest.mark.parametrize("embed", [128])
def test_unpad_fa_npd_none_mask(np_dtype, run_mode, batch, seq, head_num, kv_head_num, embed):
    """
    Feature: Test unpad flash attention.
    Description: Test BSND format query, key, value and mask.
    Expectation: Assert that results are consistent with numpy.
    """
    test_config = TestConfig(device_target="Ascend", mode=run_mode)
    test_config.apply()

    block_size = 32
    net0 = FlashAttention(head_num, 1.0)
    net1 = FlashAttentionNpd(head_num, 1.0, block_size)

    (
        np_query,
        np_key,
        np_value,
        np_key_cache,
        np_value_cache,
        np_slot_map,
        np_attn_mask,
        np_q_seq,
        np_kv_seq,
        np_block_tbl,
    ) = create_inputs(np_dtype, batch, seq, head_num, kv_head_num, embed, block_size)

    (
        ms_query,
        ms_key,
        ms_value,
        ms_key_cache,
        ms_value_cache,
        ms_slot_map,
        ms_attn_mask,
        ms_q_seq,
        ms_kv_seq,
        ms_block_tbl,
    ) = create_ms_inputs(
        np_query,
        np_key,
        np_value,
        np_key_cache,
        np_value_cache,
        np_slot_map,
        np_attn_mask,
        np_q_seq,
        np_kv_seq,
        np_block_tbl,
    )

    # Run test
    ms_attn_mask = None
    golden_out = net0(
        ms_query, ms_key, ms_value, ms_key_cache, ms_value_cache, ms_slot_map, ms_attn_mask, ms_q_seq, ms_kv_seq
    )
    ms_q_seq = ms_q_seq.move_to("CPU")
    ms_kv_seq = ms_kv_seq.move_to("CPU")
    out = net1(
        ms_query,
        ms_key,
        ms_value,
        ms_key_cache,
        ms_value_cache,
        ms_slot_map,
        ms_attn_mask,
        ms_q_seq,
        ms_kv_seq,
        ms_block_tbl,
    )
    # Verify Output
    TestResultVerifier.verify_results(golden_out, out, np_dtype)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize("np_dtype", [np.float16, bfloat16])
@pytest.mark.parametrize("run_mode", [context.GRAPH_MODE])
@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("seq", [16384, 32768])
@pytest.mark.parametrize("head_num", [8])
@pytest.mark.parametrize("kv_head_num", [2])
@pytest.mark.parametrize("embed", [32, 64, 128, 256])
@pytest.mark.parametrize("chunk_size", [8192])
def test_unpad_fa_npd_prefix(np_dtype, run_mode, batch, seq, head_num, kv_head_num, embed, chunk_size):
    """
    Feature: Test unpad flash attention.
    Description: Test BSND format query, key, value and mask.
    Expectation: Assert that results are consistent with numpy.
    """
    test_config = TestConfig(device_target="Ascend", mode=run_mode)
    test_config.apply()

    block_size = 32
    net0 = FlashAttention(head_num, 1.0)
    net1 = FlashAttentionNpd(head_num, 1.0, block_size)

    (
        np_query,
        np_key,
        np_value,
        np_key_cache,
        np_value_cache,
        np_slot_map,
        np_attn_mask,
        np_q_seq,
        np_kv_seq,
        np_block_tbl,
    ) = create_inputs(np_dtype, batch, seq, head_num, kv_head_num, embed, block_size)

    (
        ms_query,
        ms_key,
        ms_value,
        ms_key_cache,
        ms_value_cache,
        ms_slot_map,
        ms_attn_mask,
        ms_q_seq,
        ms_kv_seq,
        ms_block_tbl,
    ) = create_ms_inputs(
        np_query,
        np_key,
        np_value,
        np_key_cache,
        np_value_cache,
        np_slot_map,
        np_attn_mask,
        np_q_seq,
        np_kv_seq,
        np_block_tbl,
    )

    # Run test
    # ms_attn_mask= None
    golden_out = net0(
        ms_query, ms_key, ms_value, ms_key_cache, ms_value_cache, ms_slot_map, ms_attn_mask, ms_q_seq, ms_kv_seq
    )

    block_num = np_block_tbl.shape[1]
    chunks = (seq + chunk_size - 1) // chunk_size
    last_chunk = chunk_size if (seq % chunk_size == 0) else seq % chunk_size
    concat = ops.Concat(axis=1)

    ms_type = ms.bfloat16
    if np_dtype == np.float16:
        ms_type = ms.float16
    out = Tensor(shape=(batch, 0), dtype=ms_type)
    for c in range(chunks):
        st_off = c * chunk_size
        cur_chunk_size = chunk_size if (c + 1 != chunks) else last_chunk
        ed_off = st_off + cur_chunk_size

        # slice inputs according to chunk
        cur_q = np_query.reshape(batch, seq, head_num * embed)[:, st_off:ed_off, :].reshape(-1, head_num * embed)
        cur_k = np_key.reshape(batch, seq, kv_head_num * embed)[:, st_off:ed_off, :].reshape(-1, kv_head_num * embed)
        cur_v = np_value.reshape(batch, seq, kv_head_num * embed)[:, st_off:ed_off, :].reshape(-1, kv_head_num * embed)

        cur_slot_mapping = np_slot_map.reshape(batch, seq)[:, st_off:ed_off].reshape(-1)
        cur_block_table = np.full(block_num * batch, -1).reshape(batch, block_num).astype(np.int32)
        cur_block_table[:, : (ed_off + block_size - 1) // block_size] = np_block_tbl[
            :, : (ed_off + block_size - 1) // block_size
        ]
        actual_seq_qlen = np.full(batch, cur_chunk_size).astype(np.int32)
        actual_seq_kvlen = np.full(batch, ed_off).astype(np.int32)

        (
            ms_query,
            ms_key,
            ms_value,
            ms_key_cache,
            ms_value_cache,
            ms_slot_map,
            ms_attn_mask,
            ms_q_seq,
            ms_kv_seq,
            ms_block_tbl,
        ) = create_ms_inputs(
            cur_q,
            cur_k,
            cur_v,
            np_key_cache,
            np_value_cache,
            cur_slot_mapping,
            np_attn_mask,
            actual_seq_qlen,
            actual_seq_kvlen,
            cur_block_table,
        )

        ms_q_seq = ms_q_seq.move_to("CPU")
        ms_kv_seq = ms_kv_seq.move_to("CPU")

        out_chunk = net1(
            ms_query,
            ms_key,
            ms_value,
            ms_key_cache,
            ms_value_cache,
            ms_slot_map,
            ms_attn_mask,
            ms_q_seq,
            ms_kv_seq,
            ms_block_tbl,
        )
        out_chunk = out_chunk.reshape(batch, -1)
        out = concat((out, out_chunk))
        np_key_cache = ms_key_cache.asnumpy()
        np_value_cache = ms_value_cache.asnumpy()

    # Verify Output
    TestResultVerifier.verify_results(golden_out, out, np_dtype)
