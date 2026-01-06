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
"""apply_rotary_pos_emb test case"""
import mindspore as ms
import numpy as np
import pytest
from mindspore import Tensor, context, nn
from mindspore.common.np_dtype import bfloat16

import ms_custom_ops


def get_ms_dtype(query_dtype):
    ms_dtype = ms.float32
    if query_dtype == np.float32:
        ms_dtype = ms.float32
    elif query_dtype == np.float16:
        ms_dtype = ms.float16
    elif query_dtype == bfloat16:
        ms_dtype = ms.bfloat16
    return ms_dtype


class RotaryEmbedding(nn.Cell):
    """RotaryEmbedding"""
    # cosFormat=0  shape是[max_seqlen, head_dim]，    cos/sin不交替
    # cosFormat=1  shape是[max_seqlen, head_dim]，    cos/sin交替
    # cosFormat=2  shape是[batch*seqLen, head_dim]， cos/sin不交替
    # cosFormat=3  shape是[batch*seqLen, head_dim]， cos/sin交替

    def __init__(self, dim, base=10000, max_seq_len=2048, cos_dtype=np.float32, cos_format=0):
        super().__init__()
        inv_freq = 1.0 / \
            (base ** (np.arange(0, dim, 2).astype(np.float32) * (1 / dim)))
        t = np.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = np.outer(t, inv_freq)
        if cos_format in (0, 2):
            emb = np.concatenate((freqs, freqs), axis=-1)
        else:
            freqs = np.expand_dims(freqs, 2)
            emb = np.concatenate((freqs, freqs), axis=-1)
            emb = emb.reshape(max_seq_len, dim)
        self.cos_np = np.cos(emb).astype(cos_dtype)
        self.sin_np = np.sin(emb).astype(cos_dtype)
        self.cos = Tensor(np.cos(emb), dtype=get_ms_dtype(cos_dtype))
        self.sin = Tensor(np.sin(emb), dtype=get_ms_dtype(cos_dtype))
        self.dim = dim
        self.cos_format = cos_format
        self.cos_dtype = cos_dtype

    def construct(self, query, key, position_ids):
        """forward"""
        query_embed, key_embed = ms_custom_ops.apply_rotary_pos_emb(
            query, key, self.cos, self.sin, position_ids, self.cos_format)
        return query_embed, key_embed

    def rope_compute(self, batch, head_dim, hiddensize, hidden_size_q, hidden_size_k, batch_valid_len, head_num,
                     head_num_q, head_num_k, query, key, qtype):
        """rope_compute"""
        calculate_type = qtype
        if self.cos_dtype in (ms.float32, ms.bfloat16):
            calculate_type = ms.float32
        rotary_coeff = 2
        cos = self.cos.asnumpy()
        sin = self.sin.asnumpy()
        q_shape = query.shape
        k_shape = key.shape
        query = query.reshape(q_shape[0] * q_shape[1], q_shape[2])
        key = key.reshape(k_shape[0] * k_shape[1], k_shape[2])
        q = query.asnumpy()
        kk = key.asnumpy()
        seqlen = batch_valid_len.asnumpy()
        ntokens = np.sum(seqlen)
        rope_q = np.zeros(shape=(ntokens, hidden_size_q)).astype(calculate_type)
        rope_k = np.zeros(shape=(ntokens, hidden_size_k)).astype(calculate_type)
        prefix_ntokens = 0
        cos_list = [cos[:x, :] for x in seqlen]
        sin_list = [sin[:x, :] for x in seqlen]
        cos = np.squeeze(np.concatenate(cos_list, axis=0))
        sin = np.squeeze(np.concatenate(sin_list, axis=0))
        cos_table = np.zeros(shape=(ntokens, hiddensize)).astype(calculate_type)
        for i in range(ntokens):
            for j in range(head_num):
                cos_table[i][j*head_dim:(j+1)*head_dim] = cos[i][:]
        for i in range(batch):
            curr_seq_len = seqlen[i]
            q1 = np.zeros(shape=(curr_seq_len, hidden_size_q)).astype(calculate_type)
            k1 = np.zeros(shape=(curr_seq_len, hidden_size_k)).astype(calculate_type)

            for i in range(prefix_ntokens, prefix_ntokens + curr_seq_len):
                q1[i-prefix_ntokens] = q[i] * cos_table[i][:hidden_size_q]
                k1[i-prefix_ntokens] = kk[i] * cos_table[i][:hidden_size_k]
            q2 = np.zeros(shape=(curr_seq_len, hidden_size_q)).astype(calculate_type)
            k2 = np.zeros(shape=(curr_seq_len, hidden_size_k)).astype(calculate_type)
            for k in range(head_num):
                src_ = k * head_dim
                rotary_strdie = head_dim // rotary_coeff
                rotary_times_perhead = rotary_coeff / 2
                for cycle in range(int(rotary_times_perhead)):
                    src = src_ + cycle * rotary_strdie * 2
                    dst = src + rotary_strdie * 2
                    for curr_seqleni in range(curr_seq_len):
                        if k < head_num_q:
                            q2[curr_seqleni][src:src + rotary_strdie] = q[prefix_ntokens +
                                                                          curr_seqleni][src + rotary_strdie:dst] * (-1)
                            q2[curr_seqleni][src + rotary_strdie:dst] = q[prefix_ntokens +
                                                                          curr_seqleni][src:src+rotary_strdie]
                            q2[curr_seqleni][src:dst] = q2[curr_seqleni][src:dst] * sin[prefix_ntokens +
                                                                                        curr_seqleni][cycle *
                                                                                                      rotary_strdie * 2:
                                                                                                      (cycle + 1) *
                                                                                                      rotary_strdie * 2]
                        if k < head_num_k:
                            k2[curr_seqleni][src:src + rotary_strdie] = kk[prefix_ntokens +
                                                                           curr_seqleni][src + rotary_strdie:dst] * (-1)
                            k2[curr_seqleni][src + rotary_strdie:dst] = kk[prefix_ntokens +
                                                                           curr_seqleni][src:src+rotary_strdie]
                            k2[curr_seqleni][src:dst] = k2[curr_seqleni][src:dst] * sin[prefix_ntokens +
                                                                                        curr_seqleni][cycle *
                                                                                                      rotary_strdie * 2:
                                                                                                      (cycle + 1) *
                                                                                                      rotary_strdie * 2]
            rope_q[prefix_ntokens:prefix_ntokens + curr_seq_len] += q1 + q2
            rope_k[prefix_ntokens:prefix_ntokens + curr_seq_len] += k1 + k2

            prefix_ntokens += curr_seq_len
        rope_q = rope_q.reshape(q_shape[0], q_shape[1], q_shape[2])  # pylint: disable=too-many-function-args
        rope_k = rope_k.reshape(k_shape[0], k_shape[1], k_shape[2])  # pylint: disable=too-many-function-args
        return rope_q.astype(qtype), rope_k.astype(qtype)


def run(net, seq_lens, num_head_q, num_head_k, hidden_dim, query_dtype):
    """run case"""
    batch = len(seq_lens)
    seq_len_in = int(sum(seq_lens)/2)
    hidden_size_q = num_head_q * hidden_dim
    hidden_size_k = num_head_k * hidden_dim
    # query = np.random.rand(batch, seqLen, hidden_size_q).astype(np.float32)
    # key = np.random.rand(batch, seqLen, hidden_size_k).astype(np.float32)
    query = np.random.rand(1, seq_len_in, hidden_size_q).astype(np.float32)
    key = np.random.rand(1, seq_len_in, hidden_size_k).astype(np.float32)
    query = np.concatenate((query, query))
    key = np.concatenate((key, key))
    # 判断 q/k 前一半和后一半相等
    np.testing.assert_allclose(
        query[0:1, :, :], query[1:2, :, :], rtol=1e-2, atol=1e-2, err_msg="in query 前一半和后一半要相等")
    np.testing.assert_allclose(
        key[0:1, :, :], key[1:2, :, :], rtol=1e-2, atol=1e-2, err_msg="in key 前一半和后一半要相等")
    query = query.reshape(1, sum(seq_lens), -1)  # pylint: disable=too-many-function-args
    key = key.reshape(1, sum(seq_lens), -1)  # pylint: disable=too-many-function-args
    in_query = Tensor(query, dtype=get_ms_dtype(query_dtype))
    in_key = Tensor(key, dtype=get_ms_dtype(query_dtype))
    batch_valid_len = Tensor(seq_lens, dtype=ms.int32)
    out_query, out_key = net(in_query, in_key, batch_valid_len)

    out_query_np = out_query.astype(ms.float32).asnumpy()
    out_key_np = out_key.astype(ms.float32).asnumpy()
    # pylint: disable=too-many-function-args
    np.testing.assert_allclose(out_query_np[:, :seq_len_in, :], out_query_np[:,
                               seq_len_in:, :], rtol=1e-2, atol=1e-2, err_msg="out query 前一半和后一半要相等")
    # pylint: disable=too-many-function-args
    np.testing.assert_allclose(out_key_np[:, :seq_len_in, :], out_key_np[:,
                               seq_len_in:, :], rtol=1e-2, atol=1e-2,  err_msg="out key 前一半和后一半要相等")
    hiddensize = max(hidden_size_q, hidden_size_k)
    head_num = max(num_head_q, num_head_k)
    golden_query, golden_key = net.rope_compute(batch, hidden_dim, hiddensize, hidden_size_q,
                                                hidden_size_k, batch_valid_len, head_num, num_head_q, num_head_k,
                                                in_query, in_key, query_dtype)
    golden_query = golden_query.astype(np.float32)
    golden_key = golden_key.astype(np.float32)
    np.testing.assert_allclose(
        out_query_np, golden_query, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(out_key_np, golden_key, rtol=1e-2, atol=1e-2)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize('query_dtype', [np.float16])
@pytest.mark.parametrize('cos_dtype', [np.float16])
@pytest.mark.parametrize('cos_format', [2])
@pytest.mark.parametrize('seq_len', [[4, 9, 4, 9]])
@pytest.mark.parametrize('num_head', [40])
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_rope_float16_unpad_special(query_dtype, cos_dtype, cos_format, seq_len, num_head, exec_mode):
    """
    Feature:apply_rotary_pos_emb kernel.
    Description: test for apply_rotary_pos_emb ops.
    Expectation:should pass for all testcases.
    """
    hidden_dim = 128
    base = 10000
    max_seq_len = 8192
    np.random.seed(0)
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0"})
    net = RotaryEmbedding(hidden_dim, base, max_seq_len, cos_dtype, cos_format)
    seqlens = np.array(seq_len, np.int32)
    run(net, seqlens, num_head, num_head, hidden_dim,
        query_dtype)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize('query_dtype', [np.float16])
@pytest.mark.parametrize('cos_dtype', [np.float16, np.float32])
@pytest.mark.parametrize('cos_format', [2])
@pytest.mark.parametrize('seq_len', [[32, 32], [1, 1], [8192, 8192]])
@pytest.mark.parametrize('num_head', [8, 16])
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_rope_float16_unpad(query_dtype, cos_dtype, cos_format, seq_len, num_head, exec_mode):
    """
    Feature:apply_rotary_pos_emb kernel.
    Description: test for apply_rotary_pos_emb ops.
    Expectation:should pass for all testcases.
    """
    hidden_dim = 128
    base = 10000
    max_seq_len = 8192
    np.random.seed(0)
    ms.set_context(device_target="Ascend", mode=exec_mode, jit_syntax_level=ms.STRICT)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    net = RotaryEmbedding(hidden_dim, base, max_seq_len, cos_dtype, cos_format)
    seqlens = np.array(seq_len, np.int32)
    run(net, seqlens, num_head, num_head, hidden_dim,
        query_dtype)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('query_dtype', [bfloat16])
@pytest.mark.parametrize('cos_dtype', [bfloat16])
@pytest.mark.parametrize('cos_format', [2])
@pytest.mark.parametrize('seq_len', [[32, 32], [1, 1], [8192, 8192]])
@pytest.mark.parametrize('num_head', [8, 16])
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_rope_float16_unpad_bf16(query_dtype, cos_dtype, cos_format, seq_len, num_head, exec_mode):
    """
    Feature:apply_rotary_pos_emb kernel.
    Description: test for apply_rotary_pos_emb ops.
    Expectation:should pass for all testcases.
    """
    hidden_dim = 128
    base = 10000
    max_seq_len = 8192
    np.random.seed(0)
    ms.set_context(device_target="Ascend", mode=exec_mode, jit_syntax_level=ms.STRICT)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    net = RotaryEmbedding(hidden_dim, base, max_seq_len, cos_dtype, cos_format)
    seqlens = np.array(seq_len, np.int32)
    run(net, seqlens, num_head, num_head, hidden_dim,
        query_dtype)
